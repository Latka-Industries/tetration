//! Optional secondary indexes attached when a generation is sealed.
//!
//! Writers update staging parts only. Compact/publish recomputes [`GenerationSecondaryIndex`]
//! from the merged `packed.tet` and writes `INDEX.json` beside it when at least one kind
//! qualifies (`value_postings`, `sorted_axis_range`). The manifest token is `"present"` or
//! `"omitted"`. A later generation gets a new document; sealed bytes are not edited in place.
//!
//! Contract: [`docs/secondary_indexes.md`](../../docs/secondary_indexes.md) (THI-568 / THI-613).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::metadata::{CoordAxisV1, DatasetMetadataV1};
use super::zone::{fill_le_bytes, hex_le, sha256_hex, stored_payload};
use super::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, DATASET_DTYPE_TAG_V1, DatasetRecordV1,
    TetFileSummaryV1, elem_size_for_wire_tag, read_tet_summary_v1,
};

/// File name beside `packed.tet` inside `generations/<G>/`.
pub const SECONDARY_INDEX_FILE: &str = "INDEX.json";

/// Manifest `secondary_indexes` value when [`SECONDARY_INDEX_FILE`] was written.
pub const SECONDARY_INDEX_PRESENT: &str = "present";

/// Manifest `secondary_indexes` value when seal ran and no dataset qualified.
pub const SECONDARY_INDEX_OMITTED: &str = "omitted";

/// On-disk secondary-index document version.
pub const SECONDARY_INDEX_VERSION: u32 = 1;

/// `indexes[].kind` for exact cell-value → chunk-list postings.
pub const VALUE_POSTINGS_KIND: &str = "value_postings";

/// `indexes[].kind` for monotonic numeric footer coords → chunk spans (THI-613).
pub const SORTED_AXIS_RANGE_KIND: &str = "sorted_axis_range";

/// Distinct non-null values per dataset above which value-postings are omitted for that dataset.
///
/// A partial list would look like “this value is absent” and drop a chunk that has it.
pub const VALUE_POSTINGS_MAX_CARDINALITY: usize = 64;

/// One exact raw value and the chunk-grid coordinates that contain it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ValuePosting {
    /// Lowercase hex of the raw little-endian element (same encoding as zone `min_le`).
    pub value_le: String,
    /// Rank-length chunk coordinates. Sorted lexicographically.
    pub chunks: Vec<Vec<u64>>,
}

/// One chunk-slab span on a sorted coordinate axis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SortedAxisSegment {
    /// Lowercase hex of the `f64` LE min coordinate covering this slab (inclusive).
    pub lo_le: String,
    /// Lowercase hex of the `f64` LE max coordinate covering this slab (inclusive).
    pub hi_le: String,
    /// Rank-length chunk coordinates on this axis slab. Sorted lexicographically.
    pub chunks: Vec<Vec<u64>>,
}

/// One index entry inside [`GenerationSecondaryIndex`].
///
/// Unknown `kind` values deserialize as [`Self::Unknown`] so a later kind does not reject the file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum SecondaryIndexEntry {
    /// Exact non-null cell values for one dataset.
    #[serde(rename = "value_postings")]
    ValuePostings {
        dataset_id: u64,
        dtype: u32,
        postings: Vec<ValuePosting>,
    },
    /// Monotonic numeric footer coords → per-chunk spans for one dataset axis.
    #[serde(rename = "sorted_axis_range")]
    SortedAxisRange {
        dataset_id: u64,
        axis: u32,
        ascending: bool,
        segments: Vec<SortedAxisSegment>,
    },
    /// A kind this crate does not interpret.
    #[serde(other)]
    Unknown,
}

/// Secondary indexes for one sealed generation. Recomputed at compact/publish.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GenerationSecondaryIndex {
    pub version: u32,
    pub generation: u64,
    /// SHA-256 of the `packed.tet` these postings were computed from.
    pub packed_sha256: String,
    pub indexes: Vec<SecondaryIndexEntry>,
}

/// Bytes to write at seal, or an explicit omit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSecondaryIndex {
    /// [`SECONDARY_INDEX_PRESENT`] or [`SECONDARY_INDEX_OMITTED`].
    pub status: &'static str,
    /// SHA-256 of [`Self::bytes`] when status is present.
    pub sha256: Option<String>,
    /// Pretty JSON (trailing newline) when status is present.
    pub bytes: Option<Vec<u8>>,
}

/// Build the seal-time index blob.
///
/// `None` indexes (status [`SECONDARY_INDEX_OMITTED`]) means no kind qualified for any
/// dataset. The caller must not write `INDEX.json`.
///
/// # Errors
///
/// Catalog parse failures, unsupported dtypes, or payload decode failures.
pub fn prepare_secondary_index(
    packed: &[u8],
    generation: u64,
    packed_sha256: &str,
) -> Result<PreparedSecondaryIndex, CatalogError> {
    let Some(doc) = secondary_index_for_packed(packed, generation, packed_sha256)? else {
        return Ok(PreparedSecondaryIndex {
            status: SECONDARY_INDEX_OMITTED,
            sha256: None,
            bytes: None,
        });
    };
    let bytes = encode_secondary_index(&doc)?;
    let sha = sha256_hex(&bytes);
    Ok(PreparedSecondaryIndex {
        status: SECONDARY_INDEX_PRESENT,
        sha256: Some(sha),
        bytes: Some(bytes),
    })
}

/// Known index kinds for datasets that qualify at seal time.
///
/// Returns `Ok(None)` when no kind qualifies.
///
/// # Errors
///
/// Catalog parse failures, unsupported dtypes, or payload decode failures.
pub fn secondary_index_for_packed(
    packed: &[u8],
    generation: u64,
    packed_sha256: &str,
) -> Result<Option<GenerationSecondaryIndex>, CatalogError> {
    let summary = read_tet_summary_v1(packed)?;
    let mut indexes = value_postings_for_summary(packed, &summary)?;
    indexes.extend(sorted_axis_entries_for_summary(&summary)?);
    indexes.sort_by_key(index_sort_key);
    if indexes.is_empty() {
        return Ok(None);
    }
    Ok(Some(GenerationSecondaryIndex {
        version: SECONDARY_INDEX_VERSION,
        generation,
        packed_sha256: packed_sha256.to_owned(),
        indexes,
    }))
}

fn index_sort_key(entry: &SecondaryIndexEntry) -> (u64, u8, u32) {
    match entry {
        SecondaryIndexEntry::ValuePostings { dataset_id, .. } => (*dataset_id, 0, 0),
        SecondaryIndexEntry::SortedAxisRange {
            dataset_id, axis, ..
        } => (*dataset_id, 1, *axis),
        SecondaryIndexEntry::Unknown => (u64::MAX, 255, u32::MAX),
    }
}

fn value_postings_for_summary(
    packed: &[u8],
    summary: &TetFileSummaryV1,
) -> Result<Vec<SecondaryIndexEntry>, CatalogError> {
    let mut by_dataset: BTreeMap<u64, DatasetAcc> = BTreeMap::new();
    for entry in &summary.chunks {
        let dataset_idx =
            usize::try_from(entry.dataset_id).map_err(|_| CatalogError::TooLargeForPlatform {
                field: "dataset_id",
                value: entry.dataset_id,
            })?;
        let dataset = summary
            .datasets
            .get(dataset_idx)
            .ok_or(CatalogError::InvalidWriteSpec(
                "chunk dataset_id out of range while building secondary indexes",
            ))?;
        let acc = by_dataset.entry(entry.dataset_id).or_insert(DatasetAcc {
            dtype: dataset.dtype,
            values: BTreeMap::new(),
            omitted: false,
        });
        if acc.omitted {
            continue;
        }
        let ndim = dataset.shape.len();
        let elem_size = elem_size_for_wire_tag(dataset.dtype)?;
        let stored = stored_payload(packed, entry.payload_offset, entry.stored_byte_len)?;
        let raw = CHUNK_PAYLOAD_CODEC_V1.decode_tile_payload(
            stored,
            entry.raw_byte_len,
            entry.stored_byte_len,
            entry.codec,
        )?;
        let fill = fill_le_bytes(
            dataset.dtype,
            summary
                .metadata
                .datasets
                .get(&dataset.name)
                .map(|m| &m.attrs),
        );
        let chunk_index = entry.chunk_index[..ndim].to_vec();
        if !accumulate_chunk(
            acc,
            &chunk_index,
            dataset.dtype,
            elem_size,
            raw.as_ref(),
            fill.as_deref(),
        )? {
            acc.omitted = true;
            acc.values.clear();
        }
    }
    Ok(by_dataset
        .into_iter()
        .filter(|(_, acc)| !acc.omitted)
        .map(|(dataset_id, acc)| SecondaryIndexEntry::ValuePostings {
            dataset_id,
            dtype: acc.dtype,
            postings: acc
                .values
                .into_iter()
                .map(|(bytes, chunks)| ValuePosting {
                    value_le: hex_le(&bytes),
                    chunks: chunks.into_iter().collect(),
                })
                .collect(),
        })
        .collect())
}

fn sorted_axis_entries_for_summary(
    summary: &TetFileSummaryV1,
) -> Result<Vec<SecondaryIndexEntry>, CatalogError> {
    let mut out = Vec::new();
    for (dataset_id, dataset) in summary.datasets.iter().enumerate() {
        let Some(meta) = summary.metadata.datasets.get(&dataset.name) else {
            continue;
        };
        let ndim = dataset.shape.len();
        for axis in 0..ndim {
            let Some((values, ascending)) =
                monotonic_numeric_axis(meta, axis, dataset.shape[axis])
            else {
                continue;
            }; // lookup uses shape extent when dim_names are absent (HDF5 CF import)
            let segments = axis_segments(summary, dataset_id as u64, dataset, axis, &values)?;
            if segments.is_empty() {
                continue;
            }
            out.push(SecondaryIndexEntry::SortedAxisRange {
                dataset_id: dataset_id as u64,
                axis: u32::try_from(axis).map_err(|_| CatalogError::TooLargeForPlatform {
                    field: "axis",
                    value: axis as u64,
                })?,
                ascending,
                segments,
            });
        }
    }
    Ok(out)
}

fn monotonic_numeric_axis(
    meta: &DatasetMetadataV1,
    axis: usize,
    extent: u64,
) -> Option<(Vec<f64>, bool)> {
    let coord = coord_axis_for_dimension(meta, axis, Some(extent))?;
    let extent_usize = usize::try_from(extent).ok()?;
    if coord.labels.len() != extent_usize || extent_usize == 0 {
        return None;
    }
    let mut values = Vec::with_capacity(extent_usize);
    for label in &coord.labels {
        let v: f64 = label.parse().ok()?;
        if !v.is_finite() {
            return None;
        }
        values.push(v);
    }
    if values.len() == 1 {
        return Some((values, true));
    }
    let ascending = values.windows(2).all(|w| w[0] < w[1]);
    let descending = values.windows(2).all(|w| w[0] > w[1]);
    if ascending {
        Some((values, true))
    } else if descending {
        Some((values, false))
    } else {
        None
    }
}

fn coord_axis_for_dimension(
    meta: &DatasetMetadataV1,
    d: usize,
    extent: Option<u64>,
) -> Option<&CoordAxisV1> {
    let coords = meta.coords.as_ref()?;
    if let Some(names) = &meta.dim_names
        && let Some(name) = names.get(d)
        && let Some(c) = coords.get(name)
    {
        return Some(c);
    }
    if let Some(c) = coords.get(d.to_string().as_str()) {
        return Some(c);
    }
    // HDF5 CF import stores coords under axis names without dim_names. Match by
    // length only when exactly one axis has that extent (avoids lat/lon/time ties).
    let extent = extent.and_then(|e| usize::try_from(e).ok())?;
    let mut matches = coords.values().filter(|c| c.labels.len() == extent);
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn axis_segments(
    summary: &TetFileSummaryV1,
    dataset_id: u64,
    dataset: &DatasetRecordV1,
    axis: usize,
    values: &[f64],
) -> Result<Vec<SortedAxisSegment>, CatalogError> {
    let ndim = dataset.shape.len();
    let chunk_len = dataset.chunk_shape.get(axis).copied().ok_or_else(|| {
        CatalogError::InvalidWriteSpec("chunk_shape missing axis while building sorted-axis index")
    })?;
    if chunk_len == 0 {
        return Err(CatalogError::InvalidWriteSpec(
            "chunk_shape axis extent is zero while building sorted-axis index",
        ));
    }
    let n_axis_chunks = dataset.shape[axis].div_ceil(chunk_len);
    let mut segments = Vec::with_capacity(
        usize::try_from(n_axis_chunks).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "axis_chunks",
            value: n_axis_chunks,
        })?,
    );
    for c in 0..n_axis_chunks {
        let g0 = c.checked_mul(chunk_len).ok_or(CatalogError::InvalidWriteSpec(
            "sorted-axis chunk start overflow",
        ))?;
        let g1 = (c.saturating_add(1))
            .checked_mul(chunk_len)
            .map(|v| v.min(dataset.shape[axis]))
            .unwrap_or(dataset.shape[axis]);
        let start = usize::try_from(g0).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "axis_start",
            value: g0,
        })?;
        let stop = usize::try_from(g1).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "axis_stop",
            value: g1,
        })?;
        let slice = &values[start..stop];
        if slice.is_empty() {
            continue;
        }
        let mut lo = slice[0];
        let mut hi = slice[0];
        for &v in &slice[1..] {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let mut chunks: Vec<Vec<u64>> = summary
            .chunks
            .iter()
            .filter(|e| {
                e.dataset_id == dataset_id
                    && e.chunk_index.get(axis).copied() == Some(c)
            })
            .map(|e| e.chunk_index[..ndim].to_vec())
            .collect();
        chunks.sort();
        segments.push(SortedAxisSegment {
            lo_le: hex_le(&lo.to_le_bytes()),
            hi_le: hex_le(&hi.to_le_bytes()),
            chunks,
        });
    }
    segments.sort_by(|a, b| a.lo_le.cmp(&b.lo_le));
    Ok(segments)
}

/// Serialize `doc` the same way seal writes `INDEX.json`.
///
/// # Errors
///
/// JSON encoding failure.
pub fn encode_secondary_index(doc: &GenerationSecondaryIndex) -> Result<Vec<u8>, CatalogError> {
    let mut bytes = serde_json::to_vec_pretty(doc)
        .map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Read `INDEX.json` from `path`.
///
/// # Errors
///
/// I/O or JSON errors.
pub fn read_secondary_index_file(path: &Path) -> Result<GenerationSecondaryIndex, CatalogError> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))
}

/// `value_postings` entries only, in document order.
#[must_use]
pub fn value_postings(doc: &GenerationSecondaryIndex) -> Vec<&SecondaryIndexEntry> {
    doc.indexes
        .iter()
        .filter(|entry| matches!(entry, SecondaryIndexEntry::ValuePostings { .. }))
        .collect()
}

/// `sorted_axis_range` entries only, in document order.
#[must_use]
pub fn sorted_axis_ranges(doc: &GenerationSecondaryIndex) -> Vec<&SecondaryIndexEntry> {
    doc.indexes
        .iter()
        .filter(|entry| matches!(entry, SecondaryIndexEntry::SortedAxisRange { .. }))
        .collect()
}

/// Known kinds only (skips [`SecondaryIndexEntry::Unknown`]), in document order.
#[must_use]
pub fn known_index_entries(doc: &GenerationSecondaryIndex) -> Vec<&SecondaryIndexEntry> {
    doc.indexes
        .iter()
        .filter(|entry| !matches!(entry, SecondaryIndexEntry::Unknown))
        .collect()
}

struct DatasetAcc {
    dtype: u32,
    values: BTreeMap<Vec<u8>, BTreeSet<Vec<u64>>>,
    omitted: bool,
}

/// `Ok(false)` when this chunk pushes the dataset over the cardinality cap.
fn accumulate_chunk(
    acc: &mut DatasetAcc,
    chunk_index: &[u64],
    dtype: u32,
    elem_size: usize,
    raw: &[u8],
    fill: Option<&[u8]>,
) -> Result<bool, CatalogError> {
    if elem_size == 0 || !raw.len().is_multiple_of(elem_size) {
        return Err(CatalogError::InvalidWriteSpec(
            "secondary-index payload length is not a multiple of the element size",
        ));
    }
    let mut distinct = BTreeSet::new();
    collect_non_null(dtype, elem_size, raw, fill, &mut distinct)?;
    for value in distinct {
        if !acc.values.contains_key(&value) && acc.values.len() >= VALUE_POSTINGS_MAX_CARDINALITY {
            return Ok(false);
        }
        acc.values
            .entry(value)
            .or_default()
            .insert(chunk_index.to_vec());
    }
    Ok(true)
}

fn collect_non_null(
    dtype: u32,
    elem_size: usize,
    raw: &[u8],
    fill: Option<&[u8]>,
    out: &mut BTreeSet<Vec<u8>>,
) -> Result<(), CatalogError> {
    let tags = DATASET_DTYPE_TAG_V1;
    if tags.is_f32(dtype) {
        collect_float::<4, f32>(raw, fill, f32::from_le_bytes, f32::is_nan, out);
        return Ok(());
    }
    if tags.is_f64(dtype) {
        collect_float::<8, f64>(raw, fill, f64::from_le_bytes, f64::is_nan, out);
        return Ok(());
    }
    if tags.is_f16(dtype) {
        collect_float::<2, half::f16>(raw, fill, half::f16::from_le_bytes, half::f16::is_nan, out);
        return Ok(());
    }
    if matches_int(dtype, elem_size) {
        for le in raw.chunks_exact(elem_size) {
            if fill.is_some_and(|f| f == le) {
                continue;
            }
            out.insert(le.to_vec());
        }
        return Ok(());
    }
    Err(CatalogError::InvalidWriteSpec(
        "unsupported dataset dtype tag",
    ))
}

fn matches_int(dtype: u32, elem_size: usize) -> bool {
    let tags = DATASET_DTYPE_TAG_V1;
    (tags.is_i64(dtype) && elem_size == 8)
        || (tags.is_i32(dtype) && elem_size == 4)
        || (tags.is_i16(dtype) && elem_size == 2)
        || (tags.is_u64(dtype) && elem_size == 8)
        || (tags.is_u32(dtype) && elem_size == 4)
        || (tags.is_u16(dtype) && elem_size == 2)
        || (tags.is_u8(dtype) && elem_size == 1)
}

fn collect_float<const N: usize, T: Copy>(
    raw: &[u8],
    fill: Option<&[u8]>,
    from_le: fn([u8; N]) -> T,
    is_nan: fn(T) -> bool,
    out: &mut BTreeSet<Vec<u8>>,
) {
    for le in raw.as_chunks::<N>().0 {
        let v = from_le(*le);
        if is_nan(v) || fill.is_some_and(|f| f == le.as_slice()) {
            continue;
        }
        out.insert(le.to_vec());
    }
}
