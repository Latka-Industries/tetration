//! Optional value-posting index attached when a generation is sealed.
//!
//! Writers update staging parts only. Compact/publish recomputes [`GenerationSecondaryIndex`]
//! from the merged `packed.tet` and writes `INDEX.json` beside it when at least one dataset
//! qualifies. The manifest token is `"present"` or `"omitted"`. A later generation gets a new
//! document; sealed bytes are not edited in place.
//!
//! Contract: [`docs/secondary_indexes.md`](../../docs/secondary_indexes.md) (THI-568).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::zone::{fill_le_bytes, hex_le, sha256_hex, stored_payload};
use super::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, DATASET_DTYPE_TAG_V1, elem_size_for_wire_tag,
    read_tet_summary_v1,
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

/// Distinct non-null values per dataset above which that dataset is left out of the index.
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
/// `None` indexes (status [`SECONDARY_INDEX_OMITTED`]) means every dataset was above the
/// cardinality cap, or the image has no datasets. The caller must not write `INDEX.json`.
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

/// Posting lists for datasets at or under [`VALUE_POSTINGS_MAX_CARDINALITY`].
///
/// Returns `Ok(None)` when no dataset qualifies.
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
    let indexes = by_dataset
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
        .collect::<Vec<_>>();
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
