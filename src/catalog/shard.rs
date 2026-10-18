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

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::CatalogError;
use super::append::{copy_stored_payload, merge_tet_files};
use super::package::{
    GenerationManifest, PACKAGE_PROTOCOL_VERSION, TetPackage, is_safe_path_segment, read_json,
    sha256_file_hex, write_json_atomic,
};
use super::read_tet_summary_v1;
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

    /// Human-readable chunk ownership for overlap rejects (`full_grid` or `chunk_ranges=[…]`).
    #[must_use]
    pub fn ranges_display(&self) -> String {
        if self.full_grid {
            return "full_grid".to_owned();
        }
        let axes: Vec<String> = self
            .chunk_ranges
            .iter()
            .map(|r| format!("[{},{})", r[0], r[1]))
            .collect();
        format!("chunk_ranges=[{}]", axes.join(","))
    }
}

fn ranges_overlap(a0: u64, a1: u64, b0: u64, b1: u64) -> bool {
    a0 < b1 && b0 < a1
}

/// Structured overlap between two parts in a publish set (decision E).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimOverlap {
    pub part_a: String,
    pub part_b: String,
    pub dataset: String,
    pub ranges_a: String,
    pub ranges_b: String,
}

impl fmt::Display for ClaimOverlap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "overlapping dataset claims in publish set: dataset={} parts={} + {} ({} vs {})",
            self.dataset, self.part_a, self.part_b, self.ranges_a, self.ranges_b
        )
    }
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
///
/// Returns the first overlapping pair with part ids, dataset name, and range displays.
/// Callers that allow identical-byte overlaps should use [`reject_differing_claim_overlaps`].
#[must_use]
pub fn find_claim_overlap(parts: &[(PathBuf, PartMeta)]) -> Option<ClaimOverlap> {
    for i in 0..parts.len() {
        for j in (i + 1)..parts.len() {
            for ca in &parts[i].1.claims {
                for cb in &parts[j].1.claims {
                    if ca.overlaps(cb) {
                        return Some(ClaimOverlap {
                            part_a: parts[i].1.part_id.clone(),
                            part_b: parts[j].1.part_id.clone(),
                            dataset: ca.name.clone(),
                            ranges_a: ca.ranges_display(),
                            ranges_b: cb.ranges_display(),
                        });
                    }
                }
            }
        }
    }
    None
}

/// `true` when two parts have identical `part_sha256`, or the overlapping dataset's chunk
/// payloads match byte-for-byte in both part files.
fn overlapping_content_identical(
    dir_a: &Path,
    meta_a: &PartMeta,
    dir_b: &Path,
    meta_b: &PartMeta,
    dataset: &str,
) -> Result<bool, CatalogError> {
    if meta_a.part_sha256 == meta_b.part_sha256 && !meta_a.part_sha256.is_empty() {
        return Ok(true);
    }
    let path_a = dir_a.join(meta_a.part_file_name());
    let path_b = dir_b.join(meta_b.part_file_name());
    let bytes_a = fs::read(&path_a)?;
    let bytes_b = fs::read(&path_b)?;
    overlapping_dataset_bytes_match(&bytes_a, &bytes_b, dataset)
}

/// Compare one named dataset's geometry + stored chunk payloads across two part files.
pub(crate) fn overlapping_dataset_bytes_match(
    bytes_a: &[u8],
    bytes_b: &[u8],
    dataset: &str,
) -> Result<bool, CatalogError> {
    let sum_a = read_tet_summary_v1(bytes_a)?;
    let sum_b = read_tet_summary_v1(bytes_b)?;
    let (id_a, ds_a) = match sum_a
        .datasets
        .iter()
        .enumerate()
        .find(|(_, d)| d.name == dataset)
    {
        Some((i, d)) => (i as u64, d),
        None => return Ok(false),
    };
    let (id_b, ds_b) = match sum_b
        .datasets
        .iter()
        .enumerate()
        .find(|(_, d)| d.name == dataset)
    {
        Some((i, d)) => (i as u64, d),
        None => return Ok(false),
    };
    if ds_a.dtype != ds_b.dtype || ds_a.shape != ds_b.shape || ds_a.chunk_shape != ds_b.chunk_shape
    {
        return Ok(false);
    }
    let mut map_a = std::collections::BTreeMap::new();
    for entry in &sum_a.chunks {
        if entry.dataset_id == id_a {
            map_a.insert(entry.chunk_index, copy_stored_payload(bytes_a, entry)?);
        }
    }
    let mut map_b = std::collections::BTreeMap::new();
    for entry in &sum_b.chunks {
        if entry.dataset_id == id_b {
            map_b.insert(entry.chunk_index, copy_stored_payload(bytes_b, entry)?);
        }
    }
    Ok(map_a == map_b)
}

/// Reject overlapping claims whose content differs (decision E).
///
/// Identical `part_sha256` or matching dataset chunk bytes are allowed (idempotent retry).
///
/// # Errors
///
/// [`CatalogError::ClaimOverlap`] when overlapping claims have differing bytes, or I/O while
/// reading part files.
pub fn reject_differing_claim_overlaps(parts: &[(PathBuf, PartMeta)]) -> Result<(), CatalogError> {
    for i in 0..parts.len() {
        for j in (i + 1)..parts.len() {
            for ca in &parts[i].1.claims {
                for cb in &parts[j].1.claims {
                    if !ca.overlaps(cb) {
                        continue;
                    }
                    let identical = overlapping_content_identical(
                        &parts[i].0,
                        &parts[i].1,
                        &parts[j].0,
                        &parts[j].1,
                        &ca.name,
                    )?;
                    if !identical {
                        return Err(CatalogError::ClaimOverlap(ClaimOverlap {
                            part_a: parts[i].1.part_id.clone(),
                            part_b: parts[j].1.part_id.clone(),
                            dataset: ca.name.clone(),
                            ranges_a: ca.ranges_display(),
                            ranges_b: cb.ranges_display(),
                        }));
                    }
                }
            }
        }
    }
    Ok(())
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

/// Result of [`validate_publish_parts`] / [`dry_run_publish_parts`]: claim checks passed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishValidation {
    /// Generation id that a real publish would seal (caller-supplied).
    pub generation: u64,
    /// Frozen part ids in publish-set order.
    pub part_ids: Vec<String>,
}

/// Resolve `part_ids` and reject differing-byte overlaps (decision E). Does not seal or lock.
///
/// Identical-byte overlaps are allowed (idempotent retry). Shared by [`dry_run_publish_parts`]
/// and [`publish_parts`]. Future schema / replace checks belong here too.
///
/// # Errors
///
/// Missing parts, differing claim overlap, or I/O while reading staging meta / part bytes.
pub fn validate_publish_parts(
    package: &TetPackage,
    generation: u64,
    part_ids: &[String],
) -> Result<(PublishValidation, Vec<(PathBuf, PartMeta)>), CatalogError> {
    let frozen = resolve_part_ids(package, part_ids)?;
    reject_differing_claim_overlaps(&frozen)?;
    Ok((
        PublishValidation {
            generation,
            part_ids: part_ids.to_vec(),
        },
        frozen,
    ))
}

/// Claim / schema checks only: no `publishing/` lock, no `packed.tet`, no `CURRENT` flip.
///
/// # Errors
///
/// Same as [`validate_publish_parts`].
pub fn dry_run_publish_parts(
    package: &TetPackage,
    generation: u64,
    part_ids: &[String],
) -> Result<PublishValidation, CatalogError> {
    let (validation, _) = validate_publish_parts(package, generation, part_ids)?;
    Ok(validation)
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
    let (_, frozen) = validate_publish_parts(package, generation, part_ids)?;

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
