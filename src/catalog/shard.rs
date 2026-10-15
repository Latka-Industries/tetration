//! Shard writer for `.tet.d` staging (writer protocol v2 / THI-463).
//!
//! Each [`TetShardWriter`] owns one immutable part under `staging/<session>/`:
//! a layout-v1 `part-<id>.tet` plus meta-last [`PartMeta`] (`PART_META.json`).
//! Does **not** update `CURRENT` — publish is a separate committer step.
//!
//! Prefer [`TetWriterSession`] when you only need a standalone sealed `.tet`.
//! Prefer [`TetShardWriter`] when multiple producers write into one package.
//!
//! Spec: [`docs/writer_protocol_v2.md`](../../docs/writer_protocol_v2.md).

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::CatalogError;
use super::append::merge_tet_files;
use super::package::{
    GenerationManifest, PACKAGE_PROTOCOL_VERSION, TetPackage, is_safe_path_segment, read_json,
    sha256_file_hex, write_json_atomic,
};
use super::session::{TetDatasetWrite, TetFile, TetWriterSession};
use crate::utils::fs_device;

const PART_META_FILE: &str = "PART_META.json";

/// Chunk-grid claim for one dataset inside a sealed part (decision E).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetClaim {
    pub name: String,
    pub dtype: u32,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    /// When `true`, this part owns every chunk of the dataset grid.
    pub full_grid: bool,
    /// Per-axis half-open chunk index ranges when [`Self::full_grid`] is `false`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chunk_ranges: Vec<[u64; 2]>,
}

impl DatasetClaim {
    /// Claim the entire chunk grid for a dataset written by the shard writer.
    #[must_use]
    pub fn full_grid(
        name: impl Into<String>,
        dtype: u32,
        shape: &[u64],
        chunk_shape: &[u64],
    ) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            chunk_shape: chunk_shape.to_vec(),
            full_grid: true,
            chunk_ranges: Vec::new(),
        }
    }

    /// `true` when two claims name the same dataset and their chunk ownership overlaps.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.name != other.name {
            return false;
        }
        if self.full_grid || other.full_grid {
            return true;
        }
        if self.chunk_ranges.len() != other.chunk_ranges.len() {
            return true;
        }
        self.chunk_ranges
            .iter()
            .zip(other.chunk_ranges.iter())
            .all(|(a, b)| ranges_overlap(a[0], a[1], b[0], b[1]))
    }
}

fn ranges_overlap(a0: u64, a1: u64, b0: u64, b1: u64) -> bool {
    a0 < b1 && b0 < a1
}

/// Staging `PART_META.json` written last when a shard seals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartMeta {
    pub part_id: String,
    pub writer_session_id: String,
    pub protocol_version: u32,
    /// Must be `true` for committers to accept this part.
    pub sealed: bool,
    pub part_byte_len: u64,
    pub part_sha256: String,
    pub claims: Vec<DatasetClaim>,
}

impl PartMeta {
    /// Relative filename of the layout-v1 part inside the staging session dir.
    #[must_use]
    pub fn part_file_name(&self) -> String {
        format!("part-{}.tet", self.part_id)
    }
}

/// Concurrent producer: buffers datasets, seals one immutable staging part.
///
/// # Example flow
///
/// ```ignore
/// let mut w = TetShardWriter::create(&pkg, "writer-a", "part-a")?;
/// w.push_dataset(TetDatasetWrite::f32_row_major(...)?)?;
/// let meta = w.seal()?;
/// ```
pub struct TetShardWriter {
    package_root: PathBuf,
    session_id: String,
    part_id: String,
    session_dir: PathBuf,
    temp_part_path: PathBuf,
    final_part_path: PathBuf,
    session: Option<TetWriterSession>,
    claims: Vec<DatasetClaim>,
}

impl TetShardWriter {
    /// Create `staging/<session_id>/` and start a writer targeting a temp part file.
    ///
    /// # Errors
    ///
    /// Invalid ids, staging collision, or I/O errors.
    pub fn create(
        package: &TetPackage,
        session_id: impl Into<String>,
        part_id: impl Into<String>,
    ) -> Result<Self, CatalogError> {
        let session_id = session_id.into();
        let part_id = part_id.into();
        if !is_safe_path_segment(&session_id) {
            return Err(CatalogError::Package("invalid staging session id"));
        }
        if !is_safe_path_segment(&part_id) {
            return Err(CatalogError::Package("invalid part id"));
        }

        let session_dir = package.create_staging_session(&session_id)?;
        let temp_part_path = session_dir.join(format!(".{part_id}.tet.tmp"));
        let final_part_path = session_dir.join(format!("part-{part_id}.tet"));
        let session = TetWriterSession::create(&temp_part_path);

        Ok(Self {
            package_root: package.root().to_path_buf(),
            session_id,
            part_id,
            session_dir,
            temp_part_path,
            final_part_path,
            session: Some(session),
            claims: Vec::new(),
        })
    }

    /// Package root this writer targets.
    #[must_use]
    pub fn package_root(&self) -> &Path {
        &self.package_root
    }

    /// Staging session directory.
    #[must_use]
    pub fn session_dir(&self) -> &Path {
        &self.session_dir
    }

    /// Part id that will appear in [`PartMeta`] and the `part-*.tet` filename.
    #[must_use]
    pub fn part_id(&self) -> &str {
        &self.part_id
    }

    /// Queue an in-memory dataset (full-grid claim) onto the underlying session.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError`] when the writer is already sealed or the dataset is invalid.
    pub fn push_dataset(&mut self, dataset: TetDatasetWrite) -> Result<(), CatalogError> {
        let session = self
            .session
            .as_mut()
            .ok_or(CatalogError::Package("TetShardWriter already sealed"))?;
        let claim = DatasetClaim::full_grid(
            dataset.name.clone(),
            dataset.dtype,
            &dataset.shape,
            &dataset.chunk_shape,
        );
        session.push_dataset(dataset)?;
        self.claims.push(claim);
        Ok(())
    }

    /// Commit the layout-v1 part, rename into place, hash, then write `PART_META.json` last.
    ///
    /// Does not touch package `CURRENT`.
    ///
    /// # Errors
    ///
    /// Already sealed, empty datasets, I/O, or hash failures.
    pub fn seal(mut self) -> Result<PartMeta, CatalogError> {
        let session = self
            .session
            .take()
            .ok_or(CatalogError::Package("TetShardWriter already sealed"))?;
        if self.claims.is_empty() {
            return Err(CatalogError::Package("shard requires at least one dataset"));
        }

        let committed = session.commit()?;
        if committed != self.temp_part_path {
            return Err(CatalogError::Package("shard commit path mismatch"));
        }
        fs_device::publish_file(&self.temp_part_path, &self.final_part_path)?;

        let part_byte_len = fs::metadata(&self.final_part_path)?.len();
        let part_sha256 = sha256_file_hex(&self.final_part_path)?;

        let meta = PartMeta {
            part_id: self.part_id.clone(),
            writer_session_id: self.session_id.clone(),
            protocol_version: PACKAGE_PROTOCOL_VERSION,
            sealed: true,
            part_byte_len,
            part_sha256,
            claims: self.claims.clone(),
        };
        write_json_atomic(&self.session_dir.join(PART_META_FILE), &meta)?;
        Ok(meta)
    }
}

/// Read `PART_META.json` from a staging session directory.
///
/// # Errors
///
/// Missing file or JSON / I/O errors.
pub fn read_part_meta(session_dir: &Path) -> Result<PartMeta, CatalogError> {
    let path = session_dir.join(PART_META_FILE);
    if !path.is_file() {
        return Err(CatalogError::Package("PART_META.json missing"));
    }
    read_json(&path)
}

/// Verify a sealed part's on-disk bytes match [`PartMeta::part_sha256`].
///
/// # Errors
///
/// Missing part file, hash mismatch, or I/O.
pub fn verify_part_meta(session_dir: &Path, meta: &PartMeta) -> Result<(), CatalogError> {
    if !meta.sealed {
        return Err(CatalogError::Package("part is not sealed"));
    }
    if meta.protocol_version != PACKAGE_PROTOCOL_VERSION {
        return Err(CatalogError::UnsupportedProtocolVersion(
            meta.protocol_version,
        ));
    }
    let part_path = session_dir.join(meta.part_file_name());
    if !part_path.is_file() {
        return Err(CatalogError::Package("sealed part file missing"));
    }
    let len = fs::metadata(&part_path)?.len();
    if len != meta.part_byte_len {
        return Err(CatalogError::Package("part byte length mismatch"));
    }
    let digest = sha256_file_hex(&part_path)?;
    if digest != meta.part_sha256 {
        return Err(CatalogError::Package("part sha256 mismatch"));
    }
    Ok(())
}

/// Open the layout-v1 part file for a sealed [`PartMeta`].
///
/// # Errors
///
/// Verification failure or [`TetFile::open`] I/O.
pub fn open_sealed_part(session_dir: &Path, meta: &PartMeta) -> Result<TetFile, CatalogError> {
    verify_part_meta(session_dir, meta)?;
    let part_path = session_dir.join(meta.part_file_name());
    Ok(TetFile::open(part_path)?)
}

/// Scan `staging/` for sealed parts (meta-last + hash OK).
///
/// Unsealed or incomplete sessions are skipped (not errors).
///
/// # Errors
///
/// I/O while reading the staging directory, or corrupt sealed meta that fails verify.
pub fn list_sealed_parts(package: &TetPackage) -> Result<Vec<(PathBuf, PartMeta)>, CatalogError> {
    let staging = package.staging_dir();
    if !staging.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(&staging)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let session_dir = entry.path();
        let meta_path = session_dir.join(PART_META_FILE);
        if !meta_path.is_file() {
            continue;
        }
        let meta: PartMeta = read_json(&meta_path)?;
        if !meta.sealed {
            continue;
        }
        verify_part_meta(&session_dir, &meta)?;
        out.push((session_dir, meta));
    }
    out.sort_by(|a, b| a.1.part_id.cmp(&b.1.part_id));
    Ok(out)
}

/// Detect overlapping dataset claims across a frozen publish set (decision E).
#[must_use]
pub fn find_claim_overlap(parts: &[(PathBuf, PartMeta)]) -> Option<(String, String, String)> {
    for i in 0..parts.len() {
        for j in (i + 1)..parts.len() {
            for ca in &parts[i].1.claims {
                for cb in &parts[j].1.claims {
                    if ca.overlaps(cb) {
                        return Some((
                            parts[i].1.part_id.clone(),
                            parts[j].1.part_id.clone(),
                            ca.name.clone(),
                        ));
                    }
                }
            }
        }
    }
    None
}

/// Resolve a frozen `part_ids` list against sealed staging (decision C).
///
/// # Errors
///
/// Missing part id, verify failure, or I/O.
pub fn resolve_part_ids(
    package: &TetPackage,
    part_ids: &[String],
) -> Result<Vec<(PathBuf, PartMeta)>, CatalogError> {
    if part_ids.is_empty() {
        return Err(CatalogError::Package(
            "publish requires a frozen part-id list",
        ));
    }
    let sealed = list_sealed_parts(package)?;
    let mut out = Vec::with_capacity(part_ids.len());
    for id in part_ids {
        let found = sealed.iter().find(|(_, m)| m.part_id == *id).cloned();
        let Some(pair) = found else {
            return Err(CatalogError::Package("frozen part id not found in staging"));
        };
        out.push(pair);
    }
    Ok(out)
}

/// Stage an existing layout-v1 `.tet` as a sealed package part (copy + `PART_META` last).
///
/// Builds full-grid [`DatasetClaim`]s from the source catalog. Does not update `CURRENT`.
///
/// # Errors
///
/// Invalid ids, empty catalog, staging collision, or I/O / hash failures.
pub fn ingest_tet_file(
    package: &TetPackage,
    source: &Path,
    session_id: impl Into<String>,
    part_id: impl Into<String>,
) -> Result<PartMeta, CatalogError> {
    let session_id = session_id.into();
    let part_id = part_id.into();
    if !is_safe_path_segment(&session_id) {
        return Err(CatalogError::Package("invalid staging session id"));
    }
    if !is_safe_path_segment(&part_id) {
        return Err(CatalogError::Package("invalid part id"));
    }
    if !source.is_file() {
        return Err(CatalogError::Package("ingest source .tet missing"));
    }

    let summary = {
        let file = TetFile::open(source)?;
        file.summary()?
    };
    if summary.datasets.is_empty() {
        return Err(CatalogError::Package("ingest source has no datasets"));
    }
    let claims: Vec<DatasetClaim> = summary
        .datasets
        .iter()
        .map(|d| DatasetClaim::full_grid(d.name.clone(), d.dtype, &d.shape, &d.chunk_shape))
        .collect();

    let session_dir = package.create_staging_session(&session_id)?;
    let final_part_path = session_dir.join(format!("part-{part_id}.tet"));
    // Copy into a temp name first, then rename (meta-last still applies after hash).
    let temp_part_path = session_dir.join(format!(".{part_id}.tet.tmp"));
    fs::copy(source, &temp_part_path)?;
    fs_device::publish_file(&temp_part_path, &final_part_path)?;

    let part_byte_len = fs::metadata(&final_part_path)?.len();
    let part_sha256 = sha256_file_hex(&final_part_path)?;
    let meta = PartMeta {
        part_id,
        writer_session_id: session_id,
        protocol_version: PACKAGE_PROTOCOL_VERSION,
        sealed: true,
        part_byte_len,
        part_sha256,
        claims,
    };
    write_json_atomic(&session_dir.join(PART_META_FILE), &meta)?;
    Ok(meta)
}

/// Suggest a filesystem-safe part id from a source path stem (falls back to `part`).
#[must_use]
pub fn part_id_from_path(path: &Path) -> String {
    let raw = path.file_stem().and_then(|s| s.to_str()).unwrap_or("part");
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() || !is_safe_path_segment(&cleaned) {
        "part".to_owned()
    } else {
        cleaned
    }
}

/// Single committer: freeze `part_ids` → merge into `packed.tet` → seal generation `G`.
///
/// Rejects overlapping claims. Does not GC staging. Crash before seal leaves `CURRENT` unchanged.
/// Compact of parts happens inside this step (decision I).
///
/// # Errors
///
/// Missing parts, claim overlap, publish lock held, merge failure, or I/O.
pub fn publish_parts(
    package: &TetPackage,
    generation: u64,
    part_ids: &[String],
) -> Result<GenerationManifest, CatalogError> {
    let frozen = resolve_part_ids(package, part_ids)?;
    if let Some((a, b, name)) = find_claim_overlap(&frozen) {
        let _ = (a, b, name);
        return Err(CatalogError::Package(
            "overlapping dataset claims in publish set",
        ));
    }

    let part_paths: Vec<PathBuf> = frozen
        .iter()
        .map(|(dir, meta)| dir.join(meta.part_file_name()))
        .collect();
    let path_refs: Vec<&Path> = part_paths.iter().map(PathBuf::as_path).collect();

    let packed_tmp = package
        .root()
        .join(format!(".publish-{generation}.tet.tmp"));
    // Best-effort cleanup if we fail after creating the temp.
    let merge_result = merge_tet_files(&packed_tmp, &path_refs);
    if let Err(e) = merge_result {
        let _ = fs::remove_file(&packed_tmp);
        return Err(e);
    }

    if let Err(e) = package.begin_publish(generation) {
        let _ = fs::remove_file(&packed_tmp);
        return Err(e);
    }

    let seal = package.seal_generation(generation, &packed_tmp, part_ids.to_vec());
    let _ = fs::remove_file(&packed_tmp);
    seal?;
    package.read_manifest(generation)
}
