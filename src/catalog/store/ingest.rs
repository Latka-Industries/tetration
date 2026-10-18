//! Stage parts and publish one generation through a [`PackageStore`](super::PackageStore).
//!
//! Same roles as local `tet ingest` / `tet publish`: N writers PUT immutable parts, one
//! committer merges them and flips `CURRENT` with a conditional put (THI-471). A rejected
//! CAS leaves the previous `CURRENT` and its `packed.tet` untouched. Readers who already
//! pinned that generation keep its object key.
//!
//! Cross-host election: exclusive `publishing/<G>` claim with heartbeat; a second host that
//! loses the claim does not write `CURRENT`. Stale takeover requires [`PublishOptions::force`].

use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::catalog::append::merge_tet_files;
use crate::catalog::package::{self, GenerationManifest, PACKAGE_PROTOCOL_VERSION};
use crate::catalog::shard::{PartMeta, find_claim_overlap};
use crate::catalog::{CatalogError, DatasetClaim, read_tet_summary_v1};

use super::publish_lock::{self, PublishOptions};
use super::{PackageStore, ReadAt, WritePart};

/// Upload one sealed layout-v1 part under `staging/<session>/`. Does not touch `CURRENT`.
///
/// `PART_META.json` is written after the part bytes (meta-last).
///
/// # Errors
///
/// Invalid ids, empty catalog, session collision, or store I/O.
pub fn stage_part(
    store: &(impl ReadAt + WritePart),
    session_id: &str,
    part_id: &str,
    tet_bytes: &[u8],
) -> Result<PartMeta, CatalogError> {
    if !package::is_safe_path_segment(session_id) {
        return Err(CatalogError::Package("invalid staging session id"));
    }
    if !package::is_safe_path_segment(part_id) {
        return Err(CatalogError::Package("invalid part id"));
    }
    let summary = read_tet_summary_v1(tet_bytes)?;
    if summary.datasets.is_empty() {
        return Err(CatalogError::Package("ingest source has no datasets"));
    }
    let claims: Vec<DatasetClaim> = summary
        .datasets
        .iter()
        .map(|d| DatasetClaim::full_grid(d.name.clone(), d.dtype, &d.shape, &d.chunk_shape))
        .collect();

    let session = format!("staging/{session_id}");
    store.create_dir_exclusive(&session)?;
    let part_key = format!("{session}/part-{part_id}.tet");
    store.write_part(&part_key, tet_bytes)?;
    let stored = store.read_all(&part_key)?;
    if stored.as_slice() != tet_bytes {
        return Err(CatalogError::Package("staged part bytes mismatch"));
    }

    let meta = PartMeta {
        part_id: part_id.to_string(),
        writer_session_id: session_id.to_string(),
        protocol_version: PACKAGE_PROTOCOL_VERSION,
        sealed: true,
        part_byte_len: tet_bytes.len() as u64,
        part_sha256: sha256_hex(tet_bytes),
        claims,
    };
    let meta_bytes = serde_json::to_vec_pretty(&meta)
        .map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))?;
    store.write_part(&format!("{session}/PART_META.json"), &meta_bytes)?;
    Ok(meta)
}

/// Merge `part_ids` into `generations/<G>/` without flipping `CURRENT`.
///
/// Uses default [`PublishOptions`] (no force takeover).
///
/// # Errors
///
/// Missing parts, claim overlap, publish lock held, merge failure, or store I/O.
pub fn seal_staged_generation(
    store: &impl PackageStore,
    generation: u64,
    part_ids: &[String],
) -> Result<GenerationManifest, CatalogError> {
    seal_staged_generation_with_options(store, generation, part_ids, &PublishOptions::default())
}

/// [`seal_staged_generation`] with explicit [`PublishOptions`] (force / stale window).
///
/// # Errors
///
/// Missing parts, claim overlap, publish lock held, merge failure, or store I/O.
pub fn seal_staged_generation_with_options(
    store: &impl PackageStore,
    generation: u64,
    part_ids: &[String],
    options: &PublishOptions,
) -> Result<GenerationManifest, CatalogError> {
    if part_ids.is_empty() {
        return Err(CatalogError::Package(
            "publish requires a frozen part-id list",
        ));
    }
    let staged = list_staged(store)?;
    let mut frozen = Vec::with_capacity(part_ids.len());
    for id in part_ids {
        let Some(meta) = staged.iter().find(|m| m.part_id == *id).cloned() else {
            return Err(CatalogError::Package("frozen part id not found in staging"));
        };
        let part_key = part_object_key(&meta);
        let bytes = store.read_all(&part_key)?;
        if bytes.len() as u64 != meta.part_byte_len || sha256_hex(&bytes) != meta.part_sha256 {
            return Err(CatalogError::Package("staged part hash mismatch"));
        }
        frozen.push((PathBuf::from(&part_key), meta, bytes));
    }
    let overlap_input: Vec<(PathBuf, PartMeta)> = frozen
        .iter()
        .map(|(path, meta, _)| (path.clone(), meta.clone()))
        .collect();
    if let Some(overlap) = find_claim_overlap(&overlap_input) {
        return Err(CatalogError::ClaimOverlap(overlap));
    }

    let tmp = TempDir::new(generation)?;
    let mut part_paths = Vec::with_capacity(frozen.len());
    for (i, (_, _, bytes)) in frozen.iter().enumerate() {
        let path = tmp.0.join(format!("part-{i}.tet"));
        fs::write(&path, bytes)?;
        part_paths.push(path);
    }
    let refs: Vec<&Path> = part_paths.iter().map(PathBuf::as_path).collect();
    let packed_path = tmp.0.join("packed.tet");
    merge_tet_files(&packed_path, &refs)?;
    let packed = fs::read(&packed_path)?;

    publish_lock::begin_publish_store(store, generation, options)?;
    let lock = publish_lock::publishing_lock_key(generation);
    let manifest = GenerationManifest {
        generation,
        protocol_version: PACKAGE_PROTOCOL_VERSION,
        part_ids: part_ids.to_vec(),
        packed_byte_len: packed.len() as u64,
        packed_sha256: Some(sha256_hex(&packed)),
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))?;
    store.write_part(&format!("{lock}/packed.tet"), &packed)?;
    store.write_part(&format!("{lock}/MANIFEST.json"), &manifest_bytes)?;
    let _ = publish_lock::touch_publish_heartbeat(store, generation);
    let _ = store.remove_object(&publish_lock::heartbeat_key(generation));
    let dest = format!("generations/{}", package::format_generation_id(generation));
    if let Err(err) = store.rename_prefix(&lock, &dest) {
        let _ = store.remove_prefix(&lock);
        return Err(err);
    }
    let _ = tmp;
    Ok(manifest)
}

/// Seal `part_ids` into generation `G` and CAS-publish `CURRENT`.
///
/// On [`CatalogError::InvalidCurrent`] the previous generation's `packed.tet` is left as it was.
///
/// # Errors
///
/// Same as [`seal_staged_generation`], plus a lost conditional put.
pub fn publish_staged_parts(
    store: &impl PackageStore,
    generation: u64,
    part_ids: &[String],
) -> Result<GenerationManifest, CatalogError> {
    publish_staged_parts_with_options(store, generation, part_ids, &PublishOptions::default())
}

/// [`publish_staged_parts`] with explicit [`PublishOptions`] (force / stale window).
///
/// # Errors
///
/// Same as [`seal_staged_generation_with_options`], plus a lost conditional put.
pub fn publish_staged_parts_with_options(
    store: &impl PackageStore,
    generation: u64,
    part_ids: &[String],
    options: &PublishOptions,
) -> Result<GenerationManifest, CatalogError> {
    let manifest = seal_staged_generation_with_options(store, generation, part_ids, options)?;
    let expected = store.current_generation()?;
    store.publish_generation(generation, expected)?;
    Ok(manifest)
}

fn list_staged(store: &impl ReadAt) -> Result<Vec<PartMeta>, CatalogError> {
    let keys = store.list_prefix("staging/")?;
    let mut out = Vec::new();
    for key in keys {
        if !key.ends_with("/PART_META.json") {
            continue;
        }
        let bytes = store.read_all(&key)?;
        let meta: PartMeta = serde_json::from_slice(&bytes)
            .map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))?;
        if !meta.sealed {
            continue;
        }
        if out.iter().any(|m: &PartMeta| m.part_id == meta.part_id) {
            return Err(CatalogError::Package("duplicate staged part id"));
        }
        out.push(meta);
    }
    Ok(out)
}

fn part_object_key(meta: &PartMeta) -> String {
    format!(
        "staging/{}/{}",
        meta.writer_session_id,
        meta.part_file_name()
    )
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(generation: u64) -> Result<Self, CatalogError> {
        // Include thread id so two committers racing the same G never share a scratch dir.
        let path = std::env::temp_dir().join(format!(
            "tetration-publish-{}-{}-{}-{:?}",
            std::process::id(),
            package::format_generation_id(generation),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
            std::thread::current().id()
        ));
        fs::create_dir_all(&path)?;
        Ok(Self(path))
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
