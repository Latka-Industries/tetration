//! Dataset directory records (v1 blob encoding and write validation).

use serde::Serialize;

use crate::utils::wire;

use super::execution::FileExecutionSettingsV1;
use super::tile;
use super::{
    CHUNK_PAYLOAD_CODEC_V1, CatalogError, DATASET_DTYPE_TAG_V1, MAX_NDIM, OneChunkRawWrite,
};

/// Full tensor + tiling for [`super::write_raw_array_file`](crate::catalog::write_raw_array_file).
#[derive(Debug, Clone)]
pub struct RawArrayWrite<'a> {
    pub name: &'a str,
    pub dtype: u32,
    pub shape: &'a [u64],
    pub chunk_shape: &'a [u64],
    /// Per-chunk payload codec (`u32` wire tag). Use [`CHUNK_PAYLOAD_CODEC_V1`](crate::catalog::CHUNK_PAYLOAD_CODEC_V1)
    /// (see [`ChunkPayloadCodecV1`](crate::catalog::ChunkPayloadCodecV1)).
    pub chunk_codec: u32,
    /// Row-major tensor bytes (`4 * product(shape)` for [`DATASET_DTYPE_TAG_V1`](crate::catalog::DATASET_DTYPE_TAG_V1) `.f32` / `.i32`, `8 *` for `.f64` / `.i64`).
    pub data: &'a [u8],
    /// Optional execution settings written into the chunk index header; `None` = engine defaults.
    pub file_execution: Option<FileExecutionSettingsV1>,
}

impl<'a> RawArrayWrite<'a> {
    /// Build a write spec from tensor geometry and row-major bytes (no validation).
    #[must_use]
    pub fn from_tensor(
        name: &'a str,
        dtype: u32,
        shape: &'a [u64],
        chunk_shape: &'a [u64],
        chunk_codec: u32,
        data: &'a [u8],
        file_execution: Option<FileExecutionSettingsV1>,
    ) -> Self {
        Self {
            name,
            dtype,
            shape,
            chunk_shape,
            chunk_codec,
            data,
            file_execution,
        }
    }
}

/// Parsed dataset record from the on-disk catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct DatasetRecordV1 {
    pub name: String,
    pub dtype: u32,
    pub shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
}

/// Rank, non-zero extents, supported dtype, and chunk grid bounds (no tensor buffer).
///
/// Shared by [`validate_raw_array_write`](validate_raw_array_write) and
/// [`crate::catalog::validate_array_write_meta`].
///
/// # Errors
///
/// Returns [`CatalogError`] when geometry is invalid or the chunk grid overflows.
pub(super) fn validate_tensor_geometry(
    shape: &[u64],
    chunk_shape: &[u64],
    dtype: u32,
) -> Result<(), CatalogError> {
    if shape.len() != chunk_shape.len() {
        return Err(CatalogError::InvalidWriteSpec(
            "shape and chunk_shape must have the same rank",
        ));
    }
    let ndim = shape.len();
    if ndim == 0 {
        return Err(CatalogError::InvalidWriteSpec(
            "tensor rank must be at least 1",
        ));
    }
    if ndim > MAX_NDIM {
        return Err(CatalogError::BadNdim { ndim });
    }
    for i in 0..ndim {
        if chunk_shape[i] == 0 || shape[i] == 0 {
            return Err(CatalogError::InvalidWriteSpec(
                "shape and chunk_shape entries must be non-zero",
            ));
        }
    }
    if !DATASET_DTYPE_TAG_V1.is_supported(dtype) {
        return Err(CatalogError::InvalidWriteSpec(
            "only dataset dtype tags in DATASET_DTYPE_TAG_V1 (f32/f64/i32/i64/u8/u16/i16/u32/f16/u64) are supported",
        ));
    }
    let _ = super::tensor_bytes_from_shape(shape, dtype)
        .ok_or(CatalogError::InvalidWriteSpec("payload size overflow"))?;
    let counts = tile::chunk_grid_counts(shape, chunk_shape)?;
    tile::total_chunk_count(&counts)?;
    Ok(())
}

pub(super) fn validate_raw_array_write(spec: &RawArrayWrite<'_>) -> Result<(), CatalogError> {
    validate_tensor_geometry(spec.shape, spec.chunk_shape, spec.dtype)?;
    let need = super::tensor_bytes_from_shape(spec.shape, spec.dtype)
        .ok_or(CatalogError::InvalidWriteSpec("payload size overflow"))?;
    if spec.data.len() as u64 != need {
        return Err(CatalogError::InvalidWriteSpec(
            "tensor data length must equal element_size * product(shape)",
        ));
    }
    if !CHUNK_PAYLOAD_CODEC_V1.is_supported(spec.chunk_codec) {
        return Err(CatalogError::UnsupportedCodec {
            codec: spec.chunk_codec,
        });
    }
    Ok(())
}

pub(super) fn validate_write_spec(spec: &OneChunkRawWrite<'_>) -> Result<(), CatalogError> {
    validate_raw_array_write(&RawArrayWrite {
        name: spec.name,
        dtype: spec.dtype,
        shape: spec.shape,
        chunk_shape: spec.chunk_shape,
        chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
        data: spec.payload,
        file_execution: None,
    })?;
    let counts = tile::chunk_grid_counts(spec.shape, spec.chunk_shape)?;
    let n = tile::total_chunk_count(&counts)?;
    if n != 1 {
        return Err(CatalogError::InvalidWriteSpec(
            "use write_raw_array_file for multi-chunk tiling (this helper requires a single chunk)",
        ));
    }
    Ok(())
}

pub(super) fn encode_dataset_blob(
    name: &str,
    dtype: u32,
    shape: &[u64],
    chunk_shape: &[u64],
) -> Result<Vec<u8>, CatalogError> {
    let ndim = shape.len();
    let name_bytes = name.as_bytes();
    let name_len = u32::try_from(name_bytes.len()).map_err(|_| CatalogError::BadDatasetName)?;
    let ndim_u32 = u32::try_from(ndim).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "tensor_rank",
        value: u64::try_from(ndim).unwrap_or(u64::MAX),
    })?;
    let mut out = Vec::new();
    out.extend_from_slice(&name_len.to_le_bytes());
    out.extend_from_slice(&dtype.to_le_bytes());
    out.extend_from_slice(&ndim_u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(name_bytes);
    let pad = wire::padding_to_align8(out.len());
    out.extend(std::iter::repeat_n(0u8, pad));
    for &s in shape {
        out.extend_from_slice(&s.to_le_bytes());
    }
    for &s in chunk_shape {
        out.extend_from_slice(&s.to_le_bytes());
    }
    Ok(out)
}

pub(super) fn parse_one_dataset_record(
    data: &[u8],
    start: usize,
    blob_end: usize,
) -> Result<(DatasetRecordV1, usize), CatalogError> {
    if start + 16 > blob_end {
        return Err(CatalogError::TooShort {
            need: start + 16,
            got: blob_end,
        });
    }
    let mut o = start;
    let name_len = wire::take_u32_le(data, &mut o) as usize;
    let dtype = wire::take_u32_le(data, &mut o);
    let ndim = wire::take_u32_le(data, &mut o) as usize;
    let _res = wire::take_u32_le(data, &mut o);
    if ndim == 0 || ndim > MAX_NDIM {
        return Err(CatalogError::BadNdim { ndim });
    }
    let name_start = o;
    let name_end = name_start
        .checked_add(name_len)
        .ok_or(CatalogError::TooShort {
            need: usize::MAX,
            got: blob_end,
        })?;
    if name_end > blob_end {
        return Err(CatalogError::TooShort {
            need: name_end,
            got: blob_end,
        });
    }
    let name_bytes = &data[name_start..name_end];
    let name = std::str::from_utf8(name_bytes)
        .map_err(|_| CatalogError::BadDatasetName)?
        .to_owned();
    o = name_end;
    let pad = wire::padding_to_align8(name_end - start);
    o += pad;
    let shapes_need = o.checked_add(ndim * 16).ok_or(CatalogError::TooShort {
        need: usize::MAX,
        got: blob_end,
    })?;
    if shapes_need > blob_end {
        return Err(CatalogError::TooShort {
            need: shapes_need,
            got: blob_end,
        });
    }
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(wire::take_u64_le(data, &mut o));
    }
    let mut chunk_shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        chunk_shape.push(wire::take_u64_le(data, &mut o));
    }
    Ok((
        DatasetRecordV1 {
            name,
            dtype,
            shape,
            chunk_shape,
        },
        o,
    ))
}
