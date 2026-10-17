//! Dataset directory and chunk index (layout v1 extension).
//!
//! See `docs/layout_v1.md` for byte layout after the 32-byte superblock.
//!
//! Multi-dataset writers share geometry and file assembly via [`file_layout`] (chunk grid planning,
//! index/payload layout, encoded chunk push). [`write`] and [`stream_write`] create new files;
//! [`append`] extends an existing catalog in place.

mod append;
mod dataset;
pub mod execution;
mod file_layout;
mod history;
mod index;
pub mod metadata;
pub mod package;
pub mod session;
pub mod shard;
pub mod store;
mod stream_write;
pub mod tile;
mod write;

use std::borrow::Cow;
use std::io;

use argus::{LeReader, PayloadCodec};
use serde::Serialize;
use thiserror::Error;

use crate::layout::{self, SuperblockV1};
use crate::utils::{dtype::ElementDtype, wire};

pub use append::{
    append_multi_mixed, append_multi_raw_array_file, append_multi_raw_array_streaming,
    merge_tet_files,
};
pub use dataset::{DatasetRecordV1, RawArrayWrite};
pub use execution::{DEFAULT_MEMORY_BUDGET_PERCENT_BPS, FileExecutionSettingsV1};
#[cfg(test)]
pub(crate) use history::rewrite_footer_bytes_for_test;
pub use history::{
    FooterBlobV1, HistoryEvent, HistoryEventV1, HistoryFooterWireV1, MetadataBlobRefV1,
    append_convert_history, append_history_events, payload_file_len, read_footer_blob,
    read_metadata, unix_timestamp_now, write_footer_blob,
};
pub use index::{CHUNK_INDEX_HEADER_V1, ChunkIndexEntryV1, ChunkIndexHeaderV1};
pub use metadata::{
    CoordAxisV1, DatasetMetadataV1, FileMetadataV1, MetadataLimitsV1, TetMetadataV1,
};
pub use package::{
    DEFAULT_PUBLISH_LOCK_STALE, GenerationManifest, PACKAGE_PROTOCOL_VERSION, PackageProtocol,
    TetPackage, TetPackagePin, format_generation_id,
};
pub use session::{
    FileMetadataDraft, TetDatasetStreamSpec, TetDatasetWrite, TetFile, TetWriterSession,
};
pub use shard::{
    DatasetClaim, PartMeta, TetShardWriter, find_claim_overlap, ingest_tet_file, list_sealed_parts,
    open_sealed_part, part_id_from_path, publish_parts, read_part_meta, resolve_part_ids,
    verify_part_meta,
};
pub use store::{
    BlockCache, ByteRange, CachingStore, GenerationPin, LocalStore, MemoryStore, ObjectFetchPlan,
    ObjectMeta, PackageStore, ParsedTetUri, PinnedPackedTet, PublishGeneration, PublishOptions,
    ReadAt, TetOpenTarget, TetUriError, WritePart, assemble_object_from_ranges,
    begin_publish_store, fetch_ranges, heartbeat_key, merge_byte_ranges, packed_tet_key,
    parse_tet_uri, pin_packed_tet, plan_catalog_fetch, publish_lock_exists, publish_lock_is_stale,
    publish_staged_parts, publish_staged_parts_with_options, publishing_lock_key,
    quarantine_publish_lock, read_superblock_at, read_tet_summary_at_store, seal_staged_generation,
    seal_staged_generation_with_options, stage_part, touch_publish_heartbeat,
};
#[cfg(feature = "tetration-s3")]
pub use store::{S3Config, S3Store};
pub use stream_write::{
    ArrayWriteMeta, StreamTileJob, StreamWriteProgress, total_chunk_count_for_meta,
    validate_array_write_meta, write_multi_raw_array_streaming,
};
pub use tile::{chunk_coords_intersecting_global_box, chunk_coords_intersecting_strided};
pub use write::{write_multi_raw_array_file, write_one_chunk_raw_file, write_raw_array_file};

/// Verify APIs moved to [`crate::verify`]; re-exported for compatibility.
pub use crate::verify::{
    TetVerifyReport, VerifyFinding, VerifyFixHint, VerifyRecommendation, VerifySeverity,
    VerifySummary, format_verify_json, format_verify_quiet, format_verify_text,
    looks_like_tet_package, verify_tet_bytes, verify_tet_file, verify_tet_package,
};

/// v1 chunk payload codec wire tags (`u32` values written per chunk in the index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkPayloadCodecV1 {
    /// Raw little-endian tensor bytes (`stored_byte_len == raw_byte_len`).
    pub raw: u32,
    /// **zstd**–compressed bytes at `payload_offset`; decompressed size is `raw_byte_len`.
    pub zstd: u32,
}

/// Defined chunk payload codecs for layout v1 (see `docs/layout_v1.md`).
pub const CHUNK_PAYLOAD_CODEC_V1: ChunkPayloadCodecV1 = ChunkPayloadCodecV1 { raw: 0, zstd: 1 };

impl ChunkPayloadCodecV1 {
    #[must_use]
    pub const fn is_raw(self, codec: u32) -> bool {
        codec == self.raw
    }

    #[must_use]
    pub const fn is_zstd(self, codec: u32) -> bool {
        codec == self.zstd
    }

    #[must_use]
    pub const fn is_supported(self, codec: u32) -> bool {
        self.is_raw(codec) || self.is_zstd(codec)
    }

    /// Encode one tile’s uncompressed row-major bytes for the given wire `codec`.
    ///
    /// Raw (`0`) returns `tile_bytes` without copying. Zstd (`1`) compresses via
    /// [`argus::encode`].
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::UnsupportedCodec`] when `codec` is neither [`ChunkPayloadCodecV1::raw`]
    /// nor [`ChunkPayloadCodecV1::zstd`], or maps zstd failures from argus.
    pub fn encode_tile_payload(
        self,
        codec: u32,
        tile_bytes: Vec<u8>,
    ) -> Result<Vec<u8>, CatalogError> {
        let payload =
            PayloadCodec::from_u32(codec).map_err(|_| CatalogError::UnsupportedCodec { codec })?;
        // Keep raw path zero-copy on the owned tile buffer (argus::encode would re-allocate).
        if matches!(payload, PayloadCodec::Raw) {
            return Ok(tile_bytes);
        }
        argus::encode(payload, &tile_bytes).map_err(map_codec_error)
    }

    /// Decode stored chunk bytes to uncompressed tile payload (`raw_byte_len` bytes).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::UnsupportedCodec`], [`CatalogError::RawStoredMismatch`],
    /// [`CatalogError::ZstdDecode`], or [`CatalogError::DecodedLengthMismatch`].
    pub fn decode_tile_payload(
        self,
        stored: &[u8],
        raw_byte_len: u64,
        stored_byte_len: u64,
        codec: u32,
    ) -> Result<Cow<'_, [u8]>, CatalogError> {
        let payload =
            PayloadCodec::from_u32(codec).map_err(|_| CatalogError::UnsupportedCodec { codec })?;
        // Catalog index carries stored_byte_len; argus raw checks use `stored.len()`.
        if matches!(payload, PayloadCodec::Raw) && stored_byte_len != raw_byte_len {
            return Err(CatalogError::RawStoredMismatch {
                raw: raw_byte_len,
                stored: stored_byte_len,
            });
        }
        let _ = stored_byte_len;
        argus::decode(payload, stored, raw_byte_len).map_err(map_codec_error)
    }
}

/// Map [`argus::CodecError`] into the catalog error surface (stable wire messages).
fn map_codec_error(err: argus::CodecError) -> CatalogError {
    match err {
        argus::CodecError::Unsupported { codec } => CatalogError::UnsupportedCodec { codec },
        argus::CodecError::RawStoredMismatch { raw, stored } => {
            CatalogError::RawStoredMismatch { raw, stored }
        }
        argus::CodecError::DecodedLengthMismatch { decoded, raw } => {
            CatalogError::DecodedLengthMismatch { decoded, raw }
        }
        argus::CodecError::Zstd(msg) => CatalogError::ZstdDecode(msg),
    }
}

/// v1 dataset element `dtype` wire tags (`u32` in each dataset directory record).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetDtypeTagV1 {
    /// IEEE754 binary32 (`f32`), row-major within each chunk.
    pub f32: u32,
    /// IEEE754 binary64 (`f64`), row-major within each chunk.
    pub f64: u32,
    /// Two's-complement `i32`, row-major within each chunk.
    pub i32: u32,
    /// Two's-complement `i64`, row-major within each chunk.
    pub i64: u32,
    /// Unsigned `u8`, row-major within each chunk.
    pub u8: u32,
    /// Unsigned `u16`, row-major within each chunk.
    pub u16: u32,
    /// Two's-complement `i16`, row-major within each chunk.
    pub i16: u32,
    /// Unsigned `u32`, row-major within each chunk.
    pub u32: u32,
    /// IEEE754 binary16 (`f16`), row-major within each chunk.
    pub f16: u32,
    /// Unsigned `u64`, row-major within each chunk.
    pub u64: u32,
}

/// Defined dataset element dtypes for layout v1 (see `docs/layout_v1.md`).
pub const DATASET_DTYPE_TAG_V1: DatasetDtypeTagV1 = DatasetDtypeTagV1 {
    f32: 1,
    f64: 2,
    i32: 3,
    i64: 4,
    u8: 5,
    u16: 6,
    i16: 7,
    u32: 8,
    f16: 9,
    u64: 10,
};

impl DatasetDtypeTagV1 {
    #[must_use]
    pub const fn is_f32(self, dtype: u32) -> bool {
        dtype == self.f32
    }

    #[must_use]
    pub const fn is_f64(self, dtype: u32) -> bool {
        dtype == self.f64
    }

    #[must_use]
    pub const fn is_i32(self, dtype: u32) -> bool {
        dtype == self.i32
    }

    #[must_use]
    pub const fn is_i64(self, dtype: u32) -> bool {
        dtype == self.i64
    }

    #[must_use]
    pub const fn is_u8(self, dtype: u32) -> bool {
        dtype == self.u8
    }

    #[must_use]
    pub const fn is_u16(self, dtype: u32) -> bool {
        dtype == self.u16
    }

    #[must_use]
    pub const fn is_i16(self, dtype: u32) -> bool {
        dtype == self.i16
    }

    #[must_use]
    pub const fn is_u32(self, dtype: u32) -> bool {
        dtype == self.u32
    }

    #[must_use]
    pub const fn is_f16(self, dtype: u32) -> bool {
        dtype == self.f16
    }

    #[must_use]
    pub const fn is_u64(self, dtype: u32) -> bool {
        dtype == self.u64
    }

    #[must_use]
    pub const fn is_supported(self, dtype: u32) -> bool {
        self.is_f32(dtype)
            || self.is_f64(dtype)
            || self.is_i32(dtype)
            || self.is_i64(dtype)
            || self.is_u8(dtype)
            || self.is_u16(dtype)
            || self.is_i16(dtype)
            || self.is_u32(dtype)
            || self.is_f16(dtype)
            || self.is_u64(dtype)
    }
}

/// Maximum tensor rank supported by the v1 catalog on disk.
pub const MAX_NDIM: usize = 8;

/// Element count × element size for a tensor with `shape` and wire `dtype`.
#[must_use]
pub fn tensor_bytes_from_shape(shape: &[u64], dtype: u32) -> Option<u64> {
    crate::utils::dtype::ElementDtype::try_from_wire_tag(dtype)?.tensor_bytes_for_shape(shape)
}

/// Element size in bytes for a supported catalog wire `dtype` tag.
///
/// # Errors
///
/// Returns [`CatalogError::InvalidWriteSpec`] when `dtype` is not a supported v1 tag.
pub fn elem_size_for_wire_tag(dtype: u32) -> Result<usize, CatalogError> {
    crate::utils::dtype::ElementDtype::try_from_wire_tag(dtype)
        .map(ElementDtype::elem_size)
        .ok_or(CatalogError::InvalidWriteSpec(
            "unsupported dataset dtype tag",
        ))
}

/// Per-dataset chunk grid geometry for tile iteration during writes.
///
/// Produced by [`chunk_grid_plan`]; consumed by [`file_layout::push_raw_tiles_from_tensor`]
/// and streaming index builders.
pub(crate) struct ChunkGridPlan {
    /// Tensor rank (`shape.len()`).
    pub ndim: usize,
    /// Chunks per axis: `ceil(shape[d] / chunk_shape[d])`.
    pub counts: Vec<u64>,
    /// `product(counts)` — linear tile index runs `0..n_chunks`.
    pub n_chunks: u64,
    /// Element size in bytes for `dtype`'s wire tag.
    pub elem_size: usize,
}

/// Chunk grid counts, total tiles, and element size for a supported wire dtype.
///
/// # Errors
///
/// Returns [`CatalogError`] when the grid overflows or `dtype` is unsupported.
pub(crate) fn chunk_grid_plan(
    shape: &[u64],
    chunk_shape: &[u64],
    dtype: u32,
) -> Result<ChunkGridPlan, CatalogError> {
    let ndim = shape.len();
    let counts = tile::chunk_grid_counts(shape, chunk_shape)?;
    let n_chunks = tile::total_chunk_count(&counts)?;
    let elem_size = elem_size_for_wire_tag(dtype)?;
    Ok(ChunkGridPlan {
        ndim,
        counts,
        n_chunks,
        elem_size,
    })
}

/// Typed helpers for callers that already know the element type.
#[must_use]
pub fn f32_tensor_bytes_from_shape(shape: &[u64]) -> Option<u64> {
    ElementDtype::F32.tensor_bytes_for_shape(shape)
}

/// Element count × 8 for an `f64` tensor with `shape`.
#[must_use]
pub fn f64_tensor_bytes_from_shape(shape: &[u64]) -> Option<u64> {
    ElementDtype::F64.tensor_bytes_for_shape(shape)
}

/// Element count × 4 for an `i32` tensor with `shape`.
#[must_use]
pub fn i32_tensor_bytes_from_shape(shape: &[u64]) -> Option<u64> {
    ElementDtype::I32.tensor_bytes_for_shape(shape)
}

/// Element count × 8 for an `i64` tensor with `shape`.
#[must_use]
pub fn i64_tensor_bytes_from_shape(shape: &[u64]) -> Option<u64> {
    ElementDtype::I64.tensor_bytes_for_shape(shape)
}

/// High-level view of a mapped `.tet` file (superblock + catalog).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TetFileSummaryV1 {
    pub superblock: SuperblockV1,
    pub datasets: Vec<DatasetRecordV1>,
    pub chunks: Vec<ChunkIndexEntryV1>,
    /// Execution preferences from the chunk index header (defaults when all zero).
    pub file_execution: FileExecutionSettingsV1,
    /// Optional provenance/history footer (`[["convert","h5"|"nc","<unix_secs>"], …]`).
    pub history: Vec<HistoryEventV1>,
    /// Optional `metadata` object from the same `THST` footer JSON.
    pub metadata: TetMetadataV1,
}

/// Catalog read, index validation, codec, and writer failures.
#[derive(Debug, Error)]
pub enum CatalogError {
    #[error(transparent)]
    Layout(#[from] layout::LayoutError),
    #[error("file too short for catalog: need at least {need} bytes, got {got}")]
    TooShort { need: usize, got: usize },
    #[error("dataset_count is {count} but dataset directory is missing (file length {len})")]
    MissingDatasetDirectory { count: u32, len: usize },
    #[error("invalid UTF-8 in dataset name")]
    BadDatasetName,
    #[error("ndim {ndim} out of range (max {MAX_NDIM})")]
    BadNdim { ndim: usize },
    #[error("dataset blob length mismatch: declared {declared}, parsed {parsed}")]
    DatasetBlobMismatch { declared: u64, parsed: u64 },
    #[error(
        "dataset_count {count} cannot fit in dataset blob of {blob_len} bytes (min {min_record} B/record)"
    )]
    DatasetCountExceedsBlob {
        count: u32,
        blob_len: u64,
        min_record: u64,
    },
    #[error("chunk index offset mismatch: superblock says {sb}, expected {expected}")]
    ChunkIndexOffsetMismatch { sb: u64, expected: u64 },
    #[error("bad chunk index magic: expected TIDX, got {0:?}")]
    BadIndexMagic([u8; 4]),
    #[error("unsupported chunk index version: {0}")]
    UnsupportedIndexVersion(u32),
    #[error("chunk index entry count {count} does not fit in region length {region}")]
    BadIndexLength { count: u64, region: u64 },
    #[error("chunk payload [{start}, {end}) out of bounds for file length {file_len}")]
    PayloadOutOfBounds { file_len: u64, start: u64, end: u64 },
    #[error("unsupported codec {codec} (supported: 0 = raw, 1 = zstd)")]
    UnsupportedCodec { codec: u32 },
    #[error("raw/stored length mismatch for codec=0: raw={raw}, stored={stored}")]
    RawStoredMismatch { raw: u64, stored: u64 },
    #[error("zstd decode failed: {0}")]
    ZstdDecode(String),
    #[error("decoded payload length {decoded} != raw_byte_len {raw}")]
    DecodedLengthMismatch { decoded: usize, raw: u64 },
    #[error("invalid one-chunk write spec: {0}")]
    InvalidWriteSpec(&'static str),
    #[error("catalog numeric value too large for this platform ({field}={value})")]
    TooLargeForPlatform { field: &'static str, value: u64 },
    #[error("tet package: {0}")]
    Package(&'static str),
    #[error("not a tetration package (missing PROTOCOL): {0}")]
    NotAPackage(std::path::PathBuf),
    #[error("package path already exists: {0}")]
    PackagePathExists(std::path::PathBuf),
    #[error("unsupported package protocol_version: {0}")]
    UnsupportedProtocolVersion(u32),
    #[error("invalid package PROTOCOL / JSON: {0}")]
    InvalidPackageProtocol(String),
    #[error("invalid package CURRENT pointer")]
    InvalidCurrent,
    #[error("package generation {0} is not sealed (missing packed.tet)")]
    GenerationNotSealed(u64),
    #[error("publish lock held for generation {0}")]
    PublishLockHeld(u64),
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Arguments for writing a single-dataset, single-chunk raw payload file (no compression).
#[derive(Debug, Clone)]
pub struct OneChunkRawWrite<'a> {
    pub name: &'a str,
    pub dtype: u32,
    pub shape: &'a [u64],
    pub chunk_shape: &'a [u64],
    pub payload: &'a [u8],
}

/// Read superblock, dataset directory, and chunk index from a mapped file.
///
/// Minimum v1 dataset record size (header + ≥1 axis of shapes). Used to bound
/// `dataset_count` before allocating (THI-376).
const MIN_DATASET_RECORD_V1: u64 = 32;

/// # Errors
///
/// Returns [`CatalogError`] when layout rules are violated or regions are inconsistent.
pub fn read_tet_summary_v1(data: &[u8]) -> Result<TetFileSummaryV1, CatalogError> {
    let sb = layout::read_superblock_v1(data)?;
    if sb.dataset_count == 0 {
        let flags = sb.flags;
        return Ok(TetFileSummaryV1 {
            superblock: sb,
            datasets: Vec::new(),
            chunks: Vec::new(),
            file_execution: FileExecutionSettingsV1::default_engine(),
            history: history::read_history(data, flags)?,
            metadata: history::read_metadata(data, flags)?,
        });
    }

    let (datasets, blob_end) = parse_dataset_directory(data, &sb)?;

    let blob_end_u64 = u64::try_from(blob_end).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "dataset_blob_end",
        value: u64::MAX,
    })?;
    let expected_index_off = wire::align8_u64(blob_end_u64);
    if sb.chunk_index_offset != expected_index_off {
        return Err(CatalogError::ChunkIndexOffsetMismatch {
            sb: sb.chunk_index_offset,
            expected: expected_index_off,
        });
    }

    let idx_start = usize_from_u64("chunk_index_offset", sb.chunk_index_offset)?;
    let idx_len = usize_from_u64("chunk_index_length", sb.chunk_index_length)?;
    let idx_end = idx_start
        .checked_add(idx_len)
        .filter(|&e| e <= data.len())
        .ok_or(CatalogError::TooShort {
            need: idx_start.saturating_add(idx_len),
            got: data.len(),
        })?;
    let idx_bytes = &data[idx_start..idx_end];
    let (chunks, file_execution) = index::parse_chunk_index(idx_bytes)?;

    let payload_len = history::payload_file_len(data, sb.flags)?;
    index::validate_chunk_payloads(&chunks, payload_len)?;
    let history = history::read_history(data, sb.flags)?;
    let metadata = history::read_metadata(data, sb.flags)?;

    Ok(TetFileSummaryV1 {
        superblock: sb,
        datasets,
        chunks,
        file_execution,
        history,
        metadata,
    })
}

fn parse_dataset_directory(
    data: &[u8],
    sb: &SuperblockV1,
) -> Result<(Vec<DatasetRecordV1>, usize), CatalogError> {
    if data.len() < 40 {
        return Err(CatalogError::MissingDatasetDirectory {
            count: sb.dataset_count,
            len: data.len(),
        });
    }

    let dataset_blob_len_u64 = {
        let mut r = LeReader::require(&data[32..], "dataset_blob_len", 8).map_err(|e| {
            CatalogError::TooShort {
                need: 32 + e.need,
                got: data.len(),
            }
        })?;
        r.take_u64()
    };
    let dataset_blob_len =
        usize::try_from(dataset_blob_len_u64).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "dataset_blob_len",
            value: dataset_blob_len_u64,
        })?;
    let blob_start = 40usize;
    let blob_end = blob_start
        .checked_add(dataset_blob_len)
        .ok_or(CatalogError::TooShort {
            need: usize::MAX,
            got: data.len(),
        })?;
    if data.len() < blob_end {
        return Err(CatalogError::TooShort {
            need: blob_end,
            got: data.len(),
        });
    }

    let max_fit = (dataset_blob_len as u64) / MIN_DATASET_RECORD_V1;
    if u64::from(sb.dataset_count) > max_fit {
        return Err(CatalogError::DatasetCountExceedsBlob {
            count: sb.dataset_count,
            blob_len: dataset_blob_len as u64,
            min_record: MIN_DATASET_RECORD_V1,
        });
    }
    let mut datasets = Vec::with_capacity(sb.dataset_count as usize);
    let mut cursor = blob_start;
    for _ in 0..sb.dataset_count {
        let (rec, next) = dataset::parse_one_dataset_record(data, cursor, blob_end)?;
        datasets.push(rec);
        cursor = next;
    }
    if cursor != blob_end {
        return Err(CatalogError::DatasetBlobMismatch {
            declared: dataset_blob_len as u64,
            parsed: (cursor - blob_start) as u64,
        });
    }
    Ok((datasets, blob_end))
}

/// Validate chunk index payload spans against a file length (same rules as [`read_tet_summary_v1`]).
///
/// Exposed for integration tests in [`crate::tests`].
pub use index::validate_chunk_payloads;

pub(super) fn usize_from_u64(field: &'static str, v: u64) -> Result<usize, CatalogError> {
    usize::try_from(v).map_err(|_| CatalogError::TooLargeForPlatform { field, value: v })
}
