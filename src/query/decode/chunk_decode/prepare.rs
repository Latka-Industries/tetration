//! Chunk payload preparation: mmap slices, codec decode, and tile validation.

use std::borrow::Cow;

use crate::catalog::{CHUNK_PAYLOAD_CODEC_V1, MAX_NDIM, tile};
use crate::query::types::{PlannedChunkIo, ReadPlan, TetError};
use crate::utils::wire;

use super::elem::ChunkElem;

pub(super) struct PreparedChunk<'a> {
    pub(super) bytes_read: u64,
    pub(super) raw_bytes: Cow<'a, [u8]>,
    pub(super) nelem: usize,
    pub(super) tile: Vec<u64>,
    pub(super) chunk_coord: [u64; MAX_NDIM],
    pub(super) ndim: usize,
}

fn u64_to_usize(field: &'static str, v: u64) -> Result<usize, TetError> {
    usize::try_from(v)
        .map_err(|_| TetError::Validation(format!("{field}={v} is too large for this platform")))
}

pub(super) fn map_span_err(e: wire::SpanError, chunk_index: &[u64], mmap_len: usize) -> TetError {
    match e {
        wire::SpanError::AddOverflow => TetError::Validation("payload byte range overflow".into()),
        wire::SpanError::OutOfBounds { end } => TetError::Validation(format!(
            "chunk_index={chunk_index:?}: payload byte range extends past mmap length {mmap_len} (end={end})"
        )),
    }
}

pub(super) fn map_codec_err(e: &crate::catalog::CatalogError, chunk_index: &[u64]) -> TetError {
    TetError::Validation(format!("chunk_index={chunk_index:?}: {e}"))
}

pub(crate) fn planned_chunk_stored_slice<'a>(
    mmap: &'a [u8],
    c: &PlannedChunkIo,
) -> Result<&'a [u8], TetError> {
    let off = u64_to_usize("payload_offset", c.payload_offset)?;
    let len = u64_to_usize("stored_byte_len", c.stored_byte_len)?;
    let range = wire::checked_usize_subslice(off, len, mmap.len())
        .map_err(|e| map_span_err(e, &c.chunk_index, mmap.len()))?;
    Ok(&mmap[range])
}

pub(crate) fn decode_planned_chunk_bytes<'a>(
    stored: &'a [u8],
    c: &PlannedChunkIo,
) -> Result<Cow<'a, [u8]>, TetError> {
    CHUNK_PAYLOAD_CODEC_V1
        .decode_tile_payload(stored, c.raw_byte_len, c.stored_byte_len, c.codec)
        .map_err(|e| map_codec_err(&e, &c.chunk_index))
}

/// Map each planned chunk to a subslice of `mmap` (zero-copy).
///
/// # Errors
///
/// Returns [`TetError::Validation`] if any planned chunk uses a non-raw codec, if payload
/// offsets or lengths are invalid or extend past `mmap`, or if stored payload decode fails.
pub fn planned_chunk_mmap_slices<'a>(
    mmap: &'a [u8],
    plan: &ReadPlan,
) -> Result<Vec<&'a [u8]>, TetError> {
    let mut out = Vec::with_capacity(plan.chunks.len());
    for c in &plan.chunks {
        if !CHUNK_PAYLOAD_CODEC_V1.is_raw(c.codec) {
            return Err(TetError::Validation(format!(
                "planned_chunk_mmap_slices requires codec {} (raw); got codec={} for chunk_index={:?}",
                CHUNK_PAYLOAD_CODEC_V1.raw, c.codec, c.chunk_index
            )));
        }
        let stored = planned_chunk_stored_slice(mmap, c)?;
        decode_planned_chunk_bytes(stored, c)?;
        out.push(stored);
    }
    Ok(out)
}

pub(super) fn prepare_chunk<'a, E: ChunkElem>(
    mmap: &'a [u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
) -> Result<PreparedChunk<'a>, TetError> {
    let ndim = plan.dataset_shape.len();
    if c.chunk_index.len() != ndim {
        return Err(TetError::Validation(format!(
            "chunk_index={:?} rank {} != dataset rank {}",
            c.chunk_index,
            c.chunk_index.len(),
            ndim
        )));
    }
    let stored = planned_chunk_stored_slice(mmap, c)?;
    let bytes_read = stored.len() as u64;
    let raw_bytes = decode_planned_chunk_bytes(stored, c)?;
    if !raw_bytes.len().is_multiple_of(E::ELEM_SIZE) {
        return Err(TetError::Validation(format!(
            "chunk raw length {} is not a multiple of {} (chunk_index={:?})",
            raw_bytes.len(),
            E::ELEM_SIZE,
            c.chunk_index
        )));
    }
    let nelem = raw_bytes.len() / E::ELEM_SIZE;
    let mut chunk_coord = [0u64; MAX_NDIM];
    chunk_coord[..ndim].copy_from_slice(&c.chunk_index[..ndim]);
    let tile = tile::tile_extent(&plan.dataset_shape, &plan.chunk_shape, &chunk_coord, ndim);
    let tile_elems: u64 = tile
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or_else(|| TetError::Validation("tile element count overflow".into()))?;
    let tile_elems_us = usize::try_from(tile_elems)
        .map_err(|_| TetError::Validation("tile element count too large for this host".into()))?;
    if tile_elems_us != nelem {
        return Err(TetError::Validation(format!(
            "chunk_index={:?}: tile has {tile_elems_us} values but raw_byte_len implies {nelem}",
            c.chunk_index
        )));
    }
    Ok(PreparedChunk {
        bytes_read,
        raw_bytes,
        nelem,
        tile,
        chunk_coord,
        ndim,
    })
}
