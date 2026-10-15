//! Chunk index region (`TIDX` header + fixed-size entries).

use argus::{LeReader, LeWriter};
use serde::Serialize;

use crate::utils::wire;

use super::execution::FileExecutionSettingsV1;
use super::{CHUNK_PAYLOAD_CODEC_V1, CatalogError, MAX_NDIM};

/// Wire-fixed fields for the chunk index region header (layout v1).
///
/// `version` is the **chunk index header** format version (bytes `4..8` after `magic`), not the
/// file superblock `layout_version` (`TETR` block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkIndexHeaderV1 {
    pub magic: &'static [u8; 4],
    pub header_len: usize,
    pub version: u32,
}

/// Defined chunk index header for layout v1 (`TIDX` magic, 32-byte header, version 1).
pub const CHUNK_INDEX_HEADER_V1: ChunkIndexHeaderV1 = ChunkIndexHeaderV1 {
    magic: b"TIDX",
    header_len: 32,
    version: 1,
};

/// Chunk index entry (fixed size, little-endian). Unused `chunk_index` slots are zero.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ChunkIndexEntryV1 {
    pub dataset_id: u64,
    /// Chunk grid coordinates (`i0 … i7`); only the first *rank* values are meaningful.
    pub chunk_index: [u64; MAX_NDIM],
    pub payload_offset: u64,
    pub raw_byte_len: u64,
    pub stored_byte_len: u64,
    /// Payload codec (`u32`): compare to [`CHUNK_PAYLOAD_CODEC_V1`](crate::catalog::CHUNK_PAYLOAD_CODEC_V1)
    /// ([`ChunkPayloadCodecV1`](crate::catalog::ChunkPayloadCodecV1)).
    pub codec: u32,
}

impl ChunkIndexEntryV1 {
    /// Copy `coord` into a fixed `MAX_NDIM` index slot (unused axes stay zero).
    #[must_use]
    pub(crate) fn padded_chunk_index(coord: &[u64], ndim: usize) -> [u64; MAX_NDIM] {
        let mut chunk_index = [0u64; MAX_NDIM];
        chunk_index[..ndim].copy_from_slice(&coord[..ndim]);
        chunk_index
    }

    /// On-disk byte length of one chunk index entry (little-endian).
    pub const WIRE_LEN: usize = 8 + MAX_NDIM * 8 + 8 + 8 + 8 + 4 + 4;
    /// Byte offset of `payload_offset` within a wire entry.
    pub const WIRE_PAYLOAD_OFFSET: usize = 8 + MAX_NDIM * 8;
    /// Byte offset of `raw_byte_len` within a wire entry.
    pub const WIRE_RAW_BYTE_LEN_OFFSET: usize = Self::WIRE_PAYLOAD_OFFSET + 8;
    /// Byte offset of `stored_byte_len` within a wire entry.
    pub const WIRE_STORED_BYTE_LEN_OFFSET: usize = Self::WIRE_RAW_BYTE_LEN_OFFSET + 8;
    /// Byte offset of `codec` within a wire entry.
    pub const WIRE_CODEC_OFFSET: usize = Self::WIRE_STORED_BYTE_LEN_OFFSET + 8;

    pub(super) fn to_bytes(&self) -> [u8; Self::WIRE_LEN] {
        let mut out = [0u8; Self::WIRE_LEN];
        let mut w = LeWriter::new(&mut out);
        w.put_u64(self.dataset_id);
        for c in &self.chunk_index {
            w.put_u64(*c);
        }
        w.put_u64(self.payload_offset);
        w.put_u64(self.raw_byte_len);
        w.put_u64(self.stored_byte_len);
        w.put_u32(self.codec);
        w.put_u32(0);
        debug_assert_eq!(w.position(), Self::WIRE_LEN);
        out
    }

    pub(super) fn from_bytes(data: &[u8], off: usize) -> Result<Self, CatalogError> {
        let need = off + Self::WIRE_LEN;
        if data.len() < need {
            return Err(CatalogError::TooShort {
                need,
                got: data.len(),
            });
        }
        let mut r = LeReader::new(&data[off..off + Self::WIRE_LEN]);
        let dataset_id = r.take_u64();
        let mut chunk_index = [0u64; MAX_NDIM];
        for slot in &mut chunk_index {
            *slot = r.take_u64();
        }
        let payload_offset = r.take_u64();
        let raw_byte_len = r.take_u64();
        let stored_byte_len = r.take_u64();
        let codec = r.take_u32();
        r.skip(4);
        Ok(Self {
            dataset_id,
            chunk_index,
            payload_offset,
            raw_byte_len,
            stored_byte_len,
            codec,
        })
    }
}

pub(super) fn parse_chunk_index(
    idx_bytes: &[u8],
) -> Result<(Vec<ChunkIndexEntryV1>, FileExecutionSettingsV1), CatalogError> {
    let hdr = CHUNK_INDEX_HEADER_V1;
    let mut r = LeReader::require(idx_bytes, "chunk_index_header", hdr.header_len).map_err(
        |e| CatalogError::TooShort {
            need: e.need,
            got: e.got,
        },
    )?;
    let m = r.take_4();
    if &m != hdr.magic {
        return Err(CatalogError::BadIndexMagic(m));
    }
    let ver = r.take_u32();
    if ver != hdr.version {
        return Err(CatalogError::UnsupportedIndexVersion(ver));
    }
    let count = r.take_u64();
    let file_execution =
        FileExecutionSettingsV1::from_index_header_tail(&idx_bytes[16..hdr.header_len]);
    let entries_bytes = count
        .checked_mul(ChunkIndexEntryV1::WIRE_LEN as u64)
        .ok_or(CatalogError::BadIndexLength {
            count,
            region: idx_bytes.len() as u64,
        })?;
    let need =
        (hdr.header_len as u64)
            .checked_add(entries_bytes)
            .ok_or(CatalogError::BadIndexLength {
                count,
                region: idx_bytes.len() as u64,
            })?;
    if need != idx_bytes.len() as u64 {
        return Err(CatalogError::BadIndexLength {
            count,
            region: idx_bytes.len() as u64,
        });
    }
    let n = usize::try_from(count).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "chunk_index_entry_count",
        value: count,
    })?;
    // `need == idx_bytes.len()` above bounds `n` by the mmap region; still avoid trusting
    // a mismatched count if length math ever drifts.
    let max_fit = idx_bytes.len().saturating_sub(hdr.header_len) / ChunkIndexEntryV1::WIRE_LEN;
    if n > max_fit {
        return Err(CatalogError::BadIndexLength {
            count,
            region: idx_bytes.len() as u64,
        });
    }
    let mut out = Vec::with_capacity(n);
    let mut off = hdr.header_len;
    for _ in 0..n {
        let e = ChunkIndexEntryV1::from_bytes(idx_bytes, off)?;
        off += ChunkIndexEntryV1::WIRE_LEN;
        out.push(e);
    }
    Ok((out, file_execution))
}

pub(super) fn write_chunk_index_header(
    out: &mut Vec<u8>,
    entry_count: u64,
    file_execution: FileExecutionSettingsV1,
) {
    let hdr = CHUNK_INDEX_HEADER_V1;
    let mut header = [0u8; 32];
    {
        let mut w = LeWriter::new(&mut header);
        w.put_bytes(hdr.magic);
        w.put_u32(hdr.version);
        w.put_u64(entry_count);
        w.put_zeros(16);
        debug_assert_eq!(w.position(), hdr.header_len);
    }
    file_execution.write_index_header_tail(&mut header[16..hdr.header_len]);
    out.extend_from_slice(&header);
}

/// Validate chunk index payload spans against a file length.
///
/// # Errors
///
/// Returns [`CatalogError`] when a codec is unsupported, raw/stored lengths disagree for codec 0,
/// or a payload span extends past `file_len`.
pub fn validate_chunk_payloads(
    chunks: &[ChunkIndexEntryV1],
    file_len: u64,
) -> Result<(), CatalogError> {
    for c in chunks {
        if !CHUNK_PAYLOAD_CODEC_V1.is_supported(c.codec) {
            return Err(CatalogError::UnsupportedCodec { codec: c.codec });
        }
        if CHUNK_PAYLOAD_CODEC_V1.is_raw(c.codec) && c.raw_byte_len != c.stored_byte_len {
            return Err(CatalogError::RawStoredMismatch {
                raw: c.raw_byte_len,
                stored: c.stored_byte_len,
            });
        }
        ensure_span_in_file(c.payload_offset, c.stored_byte_len, file_len)?;
    }
    Ok(())
}

fn ensure_span_in_file(offset: u64, len: u64, file_len: u64) -> Result<(), CatalogError> {
    match wire::checked_u64_byte_span(offset, len, file_len) {
        Ok(()) => Ok(()),
        Err(wire::SpanError::AddOverflow) => Err(CatalogError::PayloadOutOfBounds {
            file_len,
            start: offset,
            end: u64::MAX,
        }),
        Err(wire::SpanError::OutOfBounds { end }) => Err(CatalogError::PayloadOutOfBounds {
            file_len,
            start: offset,
            end,
        }),
    }
}
