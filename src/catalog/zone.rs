//! Per-chunk min / max / null stats attached when a generation is sealed.
//!
//! Writers update staging parts only. Compact/publish recomputes [`GenerationZoneMaps`]
//! from the merged `packed.tet` and writes `ZONE.json` beside it. A later generation
//! gets a new document; sealed stats are not edited in place.

use std::fs;
use std::path::Path;

use half::f16;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, DATASET_DTYPE_TAG_V1, elem_size_for_wire_tag,
    read_tet_summary_v1,
};

/// File name beside `packed.tet` inside `generations/<G>/`.
pub const ZONE_MAPS_FILE: &str = "ZONE.json";

/// Manifest `zone_maps` value for seals that wrote [`ZONE_MAPS_FILE`].
pub const ZONE_MAPS_REQUIRED: &str = "required";

/// On-disk zone-map document version.
pub const ZONE_MAPS_VERSION: u32 = 1;

/// Min/max/null for one chunk in a sealed generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ChunkZoneStat {
    pub dataset_id: u64,
    /// Chunk grid coordinates (rank-length; unused axes omitted).
    pub chunk_index: Vec<u64>,
    pub element_count: u64,
    /// Fill-value matches and IEEE NaN elements. Excluded from `min_le` / `max_le`.
    pub null_count: u64,
    /// Little-endian hex of the minimum non-null element. Absent when every element is null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_le: Option<String>,
    /// Little-endian hex of the maximum non-null element. Absent when every element is null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_le: Option<String>,
}

/// Zone maps for one sealed generation. Recomputed at compact/publish.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GenerationZoneMaps {
    pub version: u32,
    pub generation: u64,
    /// SHA-256 of the `packed.tet` these stats were computed from.
    pub packed_sha256: String,
    pub chunks: Vec<ChunkZoneStat>,
}

/// Pretty JSON bytes (trailing newline) and their SHA-256 hex.
///
/// # Errors
///
/// Returns [`CatalogError`] when the packed image cannot be summarized or a chunk cannot be decoded.
pub fn zone_maps_json_for_packed(
    packed: &[u8],
    generation: u64,
    packed_sha256: &str,
) -> Result<(Vec<u8>, String), CatalogError> {
    let doc = zone_maps_for_packed(packed, generation, packed_sha256)?;
    let bytes = encode_zone_maps(&doc)?;
    let sha = sha256_hex(&bytes);
    Ok((bytes, sha))
}

/// Build zone maps by decoding every chunk in `packed`.
///
/// # Errors
///
/// Catalog parse failures, unsupported dtypes, or payload decode failures.
pub fn zone_maps_for_packed(
    packed: &[u8],
    generation: u64,
    packed_sha256: &str,
) -> Result<GenerationZoneMaps, CatalogError> {
    let summary = read_tet_summary_v1(packed)?;
    let chunks: Vec<ChunkZoneStat> = summary
        .chunks
        .par_iter()
        .map(|entry| {
            let dataset_idx = usize::try_from(entry.dataset_id).map_err(|_| {
                CatalogError::TooLargeForPlatform {
                    field: "dataset_id",
                    value: entry.dataset_id,
                }
            })?;
            let dataset =
                summary
                    .datasets
                    .get(dataset_idx)
                    .ok_or(CatalogError::InvalidWriteSpec(
                        "chunk dataset_id out of range while building zone maps",
                    ))?;
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
            chunk_zone_stat_from_raw(
                entry.dataset_id,
                &entry.chunk_index[..ndim],
                dataset.dtype,
                elem_size,
                raw.as_ref(),
                fill.as_deref(),
            )
        })
        .collect::<Result<Vec<_>, CatalogError>>()?;
    Ok(GenerationZoneMaps {
        version: ZONE_MAPS_VERSION,
        generation,
        packed_sha256: packed_sha256.to_owned(),
        chunks,
    })
}

/// Zone row for one raw (decoded) tile buffer — used during streaming write to avoid a second pass.
///
/// # Errors
///
/// Payload length / dtype problems.
pub(crate) fn chunk_zone_stat_from_raw(
    dataset_id: u64,
    chunk_index: &[u64],
    dtype: u32,
    elem_size: usize,
    raw: &[u8],
    fill: Option<&[u8]>,
) -> Result<ChunkZoneStat, CatalogError> {
    let bounds = scan_chunk(dtype, elem_size, raw, fill)?;
    let element_count =
        u64::try_from(raw.len() / elem_size).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "element_count",
            value: u64::MAX,
        })?;
    Ok(ChunkZoneStat {
        dataset_id,
        chunk_index: chunk_index.to_vec(),
        element_count,
        null_count: bounds.null_count,
        min_le: bounds.min.map(|b| hex_le(&b)),
        max_le: bounds.max.map(|b| hex_le(&b)),
    })
}

/// Serialize `doc` the same way seal writes `ZONE.json`.
///
/// # Errors
///
/// JSON encoding failure.
pub fn encode_zone_maps(doc: &GenerationZoneMaps) -> Result<Vec<u8>, CatalogError> {
    let mut bytes = serde_json::to_vec_pretty(doc)
        .map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Read `ZONE.json` from `path`.
///
/// # Errors
///
/// I/O or JSON errors.
pub fn read_zone_maps_file(path: &Path) -> Result<GenerationZoneMaps, CatalogError> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|e| CatalogError::InvalidPackageProtocol(e.to_string()))
}

/// SHA-256 hex of `bytes`.
#[must_use]
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

struct Bounds {
    null_count: u64,
    min: Option<Vec<u8>>,
    max: Option<Vec<u8>>,
}

pub(crate) fn stored_payload(packed: &[u8], offset: u64, len: u64) -> Result<&[u8], CatalogError> {
    let start = usize::try_from(offset).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "payload_offset",
        value: offset,
    })?;
    let n = usize::try_from(len).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "stored_byte_len",
        value: len,
    })?;
    let end = start
        .checked_add(n)
        .ok_or(CatalogError::InvalidWriteSpec("payload span overflow"))?;
    packed
        .get(start..end)
        .ok_or(CatalogError::PayloadOutOfBounds {
            file_len: packed.len() as u64,
            start: offset,
            end: offset.saturating_add(len),
        })
}

pub(crate) fn fill_le_bytes(
    dtype: u32,
    attrs: Option<&std::collections::BTreeMap<String, String>>,
) -> Option<Vec<u8>> {
    let attrs = attrs?;
    let raw = ["_FillValue", "missing_value", "fill_value"]
        .iter()
        .find_map(|k| attrs.get(*k))?;
    let trimmed = raw.trim();
    let tags = DATASET_DTYPE_TAG_V1;
    if tags.is_f32(dtype) {
        return Some(trimmed.parse::<f32>().ok()?.to_le_bytes().to_vec());
    }
    if tags.is_f64(dtype) {
        return Some(trimmed.parse::<f64>().ok()?.to_le_bytes().to_vec());
    }
    if tags.is_f16(dtype) {
        let v = trimmed.parse::<f32>().ok()?;
        return Some(f16::from_f32(v).to_le_bytes().to_vec());
    }
    if tags.is_i64(dtype) {
        return Some(trimmed.parse::<i64>().ok()?.to_le_bytes().to_vec());
    }
    if tags.is_u64(dtype) {
        return Some(trimmed.parse::<u64>().ok()?.to_le_bytes().to_vec());
    }
    if tags.is_i32(dtype) {
        return Some(trimmed.parse::<i32>().ok()?.to_le_bytes().to_vec());
    }
    if tags.is_u32(dtype) {
        return Some(trimmed.parse::<u32>().ok()?.to_le_bytes().to_vec());
    }
    if tags.is_i16(dtype) {
        return Some(trimmed.parse::<i16>().ok()?.to_le_bytes().to_vec());
    }
    if tags.is_u16(dtype) {
        return Some(trimmed.parse::<u16>().ok()?.to_le_bytes().to_vec());
    }
    if tags.is_u8(dtype) {
        return Some(vec![trimmed.parse::<u8>().ok()?]);
    }
    None
}

fn scan_chunk(
    dtype: u32,
    elem_size: usize,
    raw: &[u8],
    fill: Option<&[u8]>,
) -> Result<Bounds, CatalogError> {
    if elem_size == 0 || !raw.len().is_multiple_of(elem_size) {
        return Err(CatalogError::InvalidWriteSpec(
            "zone-map payload length is not a multiple of the element size",
        ));
    }
    let tags = DATASET_DTYPE_TAG_V1;
    if tags.is_f32(dtype) {
        return Ok(scan_f32(raw, fill));
    }
    if tags.is_f64(dtype) {
        return Ok(scan_f64(raw, fill));
    }
    if tags.is_f16(dtype) {
        return Ok(scan_f16(raw, fill));
    }
    if tags.is_i64(dtype) {
        return Ok(scan_int::<i64, 8>(raw, fill, i64::from_le_bytes));
    }
    if tags.is_i32(dtype) {
        return Ok(scan_int::<i32, 4>(raw, fill, i32::from_le_bytes));
    }
    if tags.is_i16(dtype) {
        return Ok(scan_int::<i16, 2>(raw, fill, i16::from_le_bytes));
    }
    if tags.is_u64(dtype) {
        return Ok(scan_int::<u64, 8>(raw, fill, u64::from_le_bytes));
    }
    if tags.is_u32(dtype) {
        return Ok(scan_int::<u32, 4>(raw, fill, u32::from_le_bytes));
    }
    if tags.is_u16(dtype) {
        return Ok(scan_int::<u16, 2>(raw, fill, u16::from_le_bytes));
    }
    if tags.is_u8(dtype) {
        return Ok(scan_int::<u8, 1>(raw, fill, u8::from_le_bytes));
    }
    Err(CatalogError::InvalidWriteSpec(
        "unsupported dataset dtype tag",
    ))
}

fn scan_f32(raw: &[u8], fill: Option<&[u8]>) -> Bounds {
    scan_float::<f32, 4>(raw, fill, f32::from_le_bytes, f32::is_nan, f32::total_cmp)
}

fn scan_f64(raw: &[u8], fill: Option<&[u8]>) -> Bounds {
    scan_float::<f64, 8>(raw, fill, f64::from_le_bytes, f64::is_nan, f64::total_cmp)
}

fn scan_f16(raw: &[u8], fill: Option<&[u8]>) -> Bounds {
    scan_float::<f16, 2>(raw, fill, f16::from_le_bytes, f16::is_nan, f16::total_cmp)
}

fn scan_float<T: Copy, const N: usize>(
    raw: &[u8],
    fill: Option<&[u8]>,
    from_le: fn([u8; N]) -> T,
    is_nan: fn(T) -> bool,
    cmp: fn(&T, &T) -> std::cmp::Ordering,
) -> Bounds {
    let mut bounds = Bounds {
        null_count: 0,
        min: None,
        max: None,
    };
    let mut min_v: Option<T> = None;
    let mut max_v: Option<T> = None;
    for le in raw.as_chunks::<N>().0 {
        let v = from_le(*le);
        if is_nan(v) || fill.is_some_and(|f| f == le.as_slice()) {
            bounds.null_count += 1;
            continue;
        }
        match min_v {
            None => {
                min_v = Some(v);
                max_v = Some(v);
                bounds.min = Some(le.to_vec());
                bounds.max = Some(le.to_vec());
            }
            Some(lo) => {
                if cmp(&v, &lo).is_lt() {
                    min_v = Some(v);
                    bounds.min = Some(le.to_vec());
                }
                if let Some(hi) = max_v
                    && cmp(&v, &hi).is_gt()
                {
                    max_v = Some(v);
                    bounds.max = Some(le.to_vec());
                }
            }
        }
    }
    bounds
}

fn scan_int<T, const N: usize>(raw: &[u8], fill: Option<&[u8]>, from_le: fn([u8; N]) -> T) -> Bounds
where
    T: Copy + Ord,
{
    let mut bounds = Bounds {
        null_count: 0,
        min: None,
        max: None,
    };
    let mut min_v: Option<T> = None;
    let mut max_v: Option<T> = None;
    for le in raw.as_chunks::<N>().0 {
        let v = from_le(*le);
        if fill.is_some_and(|f| f == le.as_slice()) {
            bounds.null_count += 1;
            continue;
        }
        match min_v {
            None => {
                min_v = Some(v);
                max_v = Some(v);
                bounds.min = Some(le.to_vec());
                bounds.max = Some(le.to_vec());
            }
            Some(lo) => {
                if v < lo {
                    min_v = Some(v);
                    bounds.min = Some(le.to_vec());
                }
                if let Some(hi) = max_v
                    && v > hi
                {
                    max_v = Some(v);
                    bounds.max = Some(le.to_vec());
                }
            }
        }
    }
    bounds
}

pub(crate) fn hex_le(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}
