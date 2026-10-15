//! Shared layout math and on-disk assembly for multi-dataset writers.
//!
//! Used by [`crate::catalog::write`], [`crate::catalog::stream_write`], and
//! [`crate::catalog::append`] so create/append/streaming paths share chunk counting, `TIDX` sizing,
//! superblock construction, and payload cursor advancement.

use std::fs::File;
use std::io::Write;

use crate::layout::{LAYOUT_VERSION_V1, SuperblockV1};
use crate::utils::wire;

use super::dataset::RawArrayWrite;
use super::execution::FileExecutionSettingsV1;
use super::index::{self, ChunkIndexEntryV1};
use super::tile;
use super::{CHUNK_PAYLOAD_CODEC_V1, CatalogError, ChunkGridPlan, MAX_NDIM, usize_from_u64};

/// Map a catalog dataset index (0-based position in the write batch) to wire `dataset_id`.
///
/// # Errors
///
/// Returns [`CatalogError::TooLargeForPlatform`] when `dataset_id` does not fit in `u64`.
pub(crate) fn wire_dataset_id(dataset_id: usize) -> Result<u64, CatalogError> {
    u64::try_from(dataset_id).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "dataset_id",
        value: u64::MAX,
    })
}

/// `usize` → `u64` for wire lengths and offsets.
///
/// # Errors
///
/// Returns [`CatalogError::TooLargeForPlatform`] on overflow.
pub(crate) fn u64_from_usize(field: &'static str, v: usize) -> Result<u64, CatalogError> {
    u64::try_from(v).map_err(|_| CatalogError::TooLargeForPlatform {
        field,
        value: u64::MAX,
    })
}

/// Sum tile counts across datasets given `(shape, chunk_shape)` pairs.
///
/// # Errors
///
/// Returns [`CatalogError::InvalidWriteSpec`] when any grid overflows or the total exceeds `u64::MAX`.
pub(crate) fn sum_chunk_counts<'a>(
    shapes: impl IntoIterator<Item = (&'a [u64], &'a [u64])>,
) -> Result<u64, CatalogError> {
    let mut total = 0u64;
    for (shape, chunk_shape) in shapes {
        let counts = tile::chunk_grid_counts(shape, chunk_shape)?;
        total = total
            .checked_add(tile::total_chunk_count(&counts)?)
            .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;
    }
    Ok(total)
}

/// Layout v1 offsets after the dataset directory blob is written.
///
/// Returns `(chunk_index_offset, chunk_index_length, payload_start)` per `docs/layout_v1.md`.
///
/// # Errors
///
/// Returns [`CatalogError::InvalidWriteSpec`] when index or payload math overflows.
pub(crate) fn chunk_index_layout(
    dataset_blob_len: u64,
    n_chunks_total: u64,
) -> Result<(u64, u64, u64), CatalogError> {
    let index_base = wire::align8_u64(40u64 + dataset_blob_len);
    let index_header_len = index::CHUNK_INDEX_HEADER_V1.header_len as u64;
    let entries_len_u64 = n_chunks_total
        .checked_mul(ChunkIndexEntryV1::WIRE_LEN as u64)
        .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;
    let chunk_index_length = index_header_len
        .checked_add(entries_len_u64)
        .ok_or(CatalogError::InvalidWriteSpec("chunk index size overflow"))?;
    let payload_start = index_base
        .checked_add(chunk_index_length)
        .ok_or(CatalogError::InvalidWriteSpec("payload start overflow"))?;
    Ok((index_base, chunk_index_length, payload_start))
}

/// Superblock for a freshly written multi-dataset file.
///
/// # Errors
///
/// Returns [`CatalogError::TooLargeForPlatform`] when `dataset_count` does not fit in `u32`.
pub(crate) fn layout_superblock(
    dataset_count: usize,
    index_base: u64,
    chunk_index_length: u64,
) -> Result<SuperblockV1, CatalogError> {
    Ok(SuperblockV1 {
        layout_version: LAYOUT_VERSION_V1,
        dataset_count: u32::try_from(dataset_count).map_err(|_| {
            CatalogError::TooLargeForPlatform {
                field: "dataset_count",
                value: dataset_count as u64,
            }
        })?,
        flags: 0,
        chunk_index_offset: index_base,
        chunk_index_length,
    })
}

/// Serialize the `TIDX` header plus fixed-size chunk index entries.
///
/// # Errors
///
/// Returns [`CatalogError`] when `chunk_index_length` does not fit in `usize`.
pub(crate) fn build_chunk_index_bytes(
    entries: &[ChunkIndexEntryV1],
    n_chunks_total: u64,
    chunk_index_length: u64,
    file_execution: FileExecutionSettingsV1,
) -> Result<Vec<u8>, CatalogError> {
    let mut index_bytes = Vec::with_capacity(usize_from_u64(
        "chunk_index_byte_length",
        chunk_index_length,
    )?);
    index::write_chunk_index_header(&mut index_bytes, n_chunks_total, file_execution);
    for e in entries {
        index_bytes.extend_from_slice(&e.to_bytes());
    }
    debug_assert_eq!(index_bytes.len() as u64, chunk_index_length);
    Ok(index_bytes)
}

/// Write superblock, dataset blob length + blob, alignment padding, and chunk index bytes.
///
/// Payload bytes are written separately after this preamble.
///
/// # Errors
///
/// Propagates I/O errors and [`CatalogError::TooLargeForPlatform`] from layout math.
pub(crate) fn write_file_preamble(
    f: &mut File,
    sb: &SuperblockV1,
    blob: &[u8],
    index_base: u64,
    index_bytes: &[u8],
) -> Result<(), CatalogError> {
    f.write_all(&sb.to_bytes())?;
    f.write_all(&(blob.len() as u64).to_le_bytes())?;
    f.write_all(blob)?;
    let after_blob = 40usize + blob.len();
    let pad = usize_from_u64("chunk_index_base", index_base)?.saturating_sub(after_blob);
    if pad > 0 {
        f.write_all(&vec![0u8; pad])?;
    }
    f.write_all(index_bytes)?;
    Ok(())
}

/// One encoded chunk payload plus index fields; [`Self::push`] sets `payload_offset` from `cursor`.
pub(crate) struct EncodedChunkPush {
    pub dataset_id: u64,
    pub chunk_index: [u64; MAX_NDIM],
    pub raw_byte_len: u64,
    pub chunk_codec: u32,
    pub stored: Vec<u8>,
}

impl EncodedChunkPush {
    /// Append index row and payload; return the next payload file offset.
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::InvalidWriteSpec`] when the payload cursor overflows.
    pub(crate) fn push(
        self,
        entries: &mut Vec<ChunkIndexEntryV1>,
        payloads: &mut Vec<Vec<u8>>,
        cursor: u64,
    ) -> Result<u64, CatalogError> {
        let stored_len = u64_from_usize("chunk_stored_len", self.stored.len())?;
        entries.push(ChunkIndexEntryV1 {
            dataset_id: self.dataset_id,
            chunk_index: self.chunk_index,
            payload_offset: cursor,
            raw_byte_len: self.raw_byte_len,
            stored_byte_len: stored_len,
            codec: self.chunk_codec,
        });
        let next = cursor
            .checked_add(stored_len)
            .ok_or(CatalogError::InvalidWriteSpec("payload cursor overflow"))?;
        payloads.push(self.stored);
        Ok(next)
    }
}

/// Slice every tile from an in-memory row-major tensor and append encoded payloads.
///
/// # Errors
///
/// Returns [`CatalogError`] when tile extraction, codec encode, or cursor math fails.
pub(crate) fn push_raw_tiles_from_tensor(
    spec: &RawArrayWrite<'_>,
    grid: &ChunkGridPlan,
    dataset_id: u64,
    mut cursor: u64,
    entries: &mut Vec<ChunkIndexEntryV1>,
    payloads: &mut Vec<Vec<u8>>,
) -> Result<u64, CatalogError> {
    for k in 0..grid.n_chunks {
        let coord = tile::chunk_coord_from_linear(k, &grid.counts, grid.ndim);
        let tile_bytes = tile::extract_tile_row_major(
            spec.data,
            spec.shape,
            spec.chunk_shape,
            &coord[..grid.ndim],
            grid.ndim,
            grid.elem_size,
        )?;
        let raw_len = u64_from_usize("chunk_payload_len", tile_bytes.len())?;
        let stored_vec =
            CHUNK_PAYLOAD_CODEC_V1.encode_tile_payload(spec.chunk_codec, tile_bytes)?;
        cursor = EncodedChunkPush {
            dataset_id,
            chunk_index: ChunkIndexEntryV1::padded_chunk_index(&coord[..grid.ndim], grid.ndim),
            raw_byte_len: raw_len,
            chunk_codec: spec.chunk_codec,
            stored: stored_vec,
        }
        .push(entries, payloads, cursor)?;
    }
    Ok(cursor)
}
