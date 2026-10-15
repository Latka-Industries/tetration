//! Shared `.tet` test fixtures (compiled into each `src/tests/*.rs` module via `mod fixture`).
#![allow(dead_code)]

use std::path::Path;

use crate::catalog::{
    CHUNK_INDEX_HEADER_V1, CHUNK_PAYLOAD_CODEC_V1, ChunkIndexEntryV1, DATASET_DTYPE_TAG_V1,
    FileExecutionSettingsV1, RawArrayWrite, read_tet_summary_v1, write_raw_array_file,
};
use crate::layout::mmap_file_read;

/// Patch helpers for corrupting chunk index entries in on-disk fixtures.
pub mod index_patch {
    use super::*;

    pub const CHUNK_INDEX_HDR_LEN: usize = CHUNK_INDEX_HEADER_V1.header_len;
    /// Byte offset of `raw_byte_len` within a chunk index entry.
    pub const ENTRY_RAW_BYTE_LEN_OFFSET: usize = ChunkIndexEntryV1::WIRE_RAW_BYTE_LEN_OFFSET;
    pub const ENTRY_STORED_BYTE_LEN_OFFSET: usize = ChunkIndexEntryV1::WIRE_STORED_BYTE_LEN_OFFSET;
    pub const ENTRY_PAYLOAD_OFFSET: usize = ChunkIndexEntryV1::WIRE_PAYLOAD_OFFSET;
    pub const ENTRY_CODEC_OFFSET: usize = ChunkIndexEntryV1::WIRE_CODEC_OFFSET;
    /// Byte offset of `chunk_index[0]` within a wire entry (`dataset_id` is 8 bytes).
    pub const ENTRY_CHUNK_INDEX0_OFFSET: usize = 8;

    pub fn first_index_field_offset(path: &Path, field_offset_in_entry: usize) -> usize {
        entry_field_offset(path, 0, field_offset_in_entry)
    }

    /// Absolute file offset of a field inside chunk-index entry `entry_idx` (0-based).
    pub fn entry_field_offset(
        path: &Path,
        entry_idx: usize,
        field_offset_in_entry: usize,
    ) -> usize {
        let mmap0 = mmap_file_read(path).unwrap();
        let s = read_tet_summary_v1(&mmap0).unwrap();
        let idx = usize::try_from(s.superblock.chunk_index_offset).unwrap();
        idx + CHUNK_INDEX_HDR_LEN + entry_idx * ChunkIndexEntryV1::WIRE_LEN + field_offset_in_entry
    }

    pub fn patch_first_index_entry_u64(path: &Path, field_offset_in_entry: usize, value: u64) {
        patch_entry_u64(path, 0, field_offset_in_entry, value);
    }

    pub fn patch_entry_u64(
        path: &Path,
        entry_idx: usize,
        field_offset_in_entry: usize,
        value: u64,
    ) {
        let patch_at = entry_field_offset(path, entry_idx, field_offset_in_entry);
        let mut bytes = std::fs::read(path).unwrap();
        assert!(
            patch_at + 8 <= bytes.len(),
            "patch offset out of range for test fixture"
        );
        bytes[patch_at..patch_at + 8].copy_from_slice(&value.to_le_bytes());
        std::fs::write(path, &bytes).unwrap();
    }

    pub fn patch_first_index_entry_raw_and_stored(
        path: &Path,
        raw_byte_len: u64,
        stored_byte_len: u64,
    ) {
        let raw_at = first_index_field_offset(path, ENTRY_RAW_BYTE_LEN_OFFSET);
        let stored_at = first_index_field_offset(path, ENTRY_STORED_BYTE_LEN_OFFSET);
        let mut bytes = std::fs::read(path).unwrap();
        assert!(stored_at + 8 <= bytes.len(), "patch offset out of range");
        bytes[raw_at..raw_at + 8].copy_from_slice(&raw_byte_len.to_le_bytes());
        bytes[stored_at..stored_at + 8].copy_from_slice(&stored_byte_len.to_le_bytes());
        std::fs::write(path, &bytes).unwrap();
    }

    /// Overwrite `len` bytes at absolute `file_off` (for corrupting stored payloads).
    pub fn patch_bytes_at(path: &Path, file_off: usize, junk: &[u8]) {
        let mut bytes = std::fs::read(path).unwrap();
        assert!(
            file_off + junk.len() <= bytes.len(),
            "payload patch out of range"
        );
        bytes[file_off..file_off + junk.len()].copy_from_slice(junk);
        std::fs::write(path, &bytes).unwrap();
    }

    pub fn truncate_file(path: &Path, new_len: u64) {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(new_len)
            .unwrap();
    }
}

/// Shape `[2, 3]` used across multi-chunk examples.
pub const SHAPE_2X3: [u64; 2] = [2, 3];
/// Chunk shape `[2, 2]` (two tiles along the last axis).
pub const CHUNK_2X2: [u64; 2] = [2, 2];

/// Row-major `f32` tensor values 1..=6 as little-endian bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_f32_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 24];
    for (slot, n) in data.chunks_exact_mut(4).zip(1_u8..=6) {
        slot.copy_from_slice(&f32::from(n).to_le_bytes());
    }
    data
}

/// Row-major `u8` tensor values 1..=6 (`shape` [2, 3]).
pub fn le_row_major_2x3_u8_one_to_six() -> Vec<u8> {
    (1_u8..=6).collect()
}

/// Row-major `u16` tensor values 1..=6 as LE bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_u16_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 12];
    for (slot, n) in data.chunks_exact_mut(2).zip(1_u16..=6) {
        slot.copy_from_slice(&n.to_le_bytes());
    }
    data
}

/// Row-major `i16` tensor values 1..=6 as LE bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_i16_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 12];
    for (slot, n) in data.chunks_exact_mut(2).zip(1_i16..=6) {
        slot.copy_from_slice(&n.to_le_bytes());
    }
    data
}

/// Row-major `i32` tensor values 1..=6 as little-endian bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_i32_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 24];
    for (slot, n) in data.chunks_exact_mut(4).zip(1_i32..=6) {
        slot.copy_from_slice(&n.to_le_bytes());
    }
    data
}

/// Row-major `i64` tensor values 1..=6 as little-endian bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_i64_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 48];
    for (slot, n) in data.chunks_exact_mut(8).zip(1_i64..=6) {
        slot.copy_from_slice(&n.to_le_bytes());
    }
    data
}

/// Row-major `f64` tensor values 1..=6 as little-endian bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_f64_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 48];
    for (slot, n) in data.chunks_exact_mut(8).zip(1_u64..=6) {
        slot.copy_from_slice(&(n as f64).to_le_bytes());
    }
    data
}

fn write_multichunk_2x3(
    path: &Path,
    dataset_name: &str,
    chunk_codec: u32,
    dtype: u32,
    data: &[u8],
) {
    write_raw_array_file(
        path,
        &RawArrayWrite {
            name: dataset_name,
            dtype,
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            chunk_codec,
            data,
            file_execution: None,
        },
    )
    .unwrap();
}

fn write_multichunk_2x3_f32(path: &Path, dataset_name: &str, chunk_codec: u32, data: &[u8]) {
    write_multichunk_2x3(
        path,
        dataset_name,
        chunk_codec,
        DATASET_DTYPE_TAG_V1.f32,
        data,
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `f32` file (values 1..6).
pub fn write_multichunk_2x3_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3_f32(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        &le_row_major_2x3_f32_one_to_six(),
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `f64` file (values 1..6).
pub fn write_multichunk_2x3_f64_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3_f64(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        &le_row_major_2x3_f64_one_to_six(),
    );
}

fn write_multichunk_2x3_f64(path: &Path, dataset_name: &str, chunk_codec: u32, data: &[u8]) {
    write_multichunk_2x3(
        path,
        dataset_name,
        chunk_codec,
        DATASET_DTYPE_TAG_V1.f64,
        data,
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `u16` file (values 1..6).
pub fn write_multichunk_2x3_u16_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        DATASET_DTYPE_TAG_V1.u16,
        &le_row_major_2x3_u16_one_to_six(),
    );
}

/// Row-major `u32` tensor values 1..=6 as LE bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_u32_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 24];
    for (slot, n) in data.chunks_exact_mut(4).zip(1_u32..=6) {
        slot.copy_from_slice(&n.to_le_bytes());
    }
    data
}

/// Row-major `u64` tensor values 1..=6 as LE bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_u64_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 48];
    for (slot, n) in data.chunks_exact_mut(8).zip(1_u64..=6) {
        slot.copy_from_slice(&n.to_le_bytes());
    }
    data
}

/// Row-major `f16` tensor values 1..=6 as LE bytes (`shape` [2, 3]).
pub fn le_row_major_2x3_f16_one_to_six() -> Vec<u8> {
    let mut data = vec![0u8; 12];
    for (slot, n) in data.chunks_exact_mut(2).zip(1_u32..=6) {
        let v = half::f16::from_f32(n as f32);
        slot.copy_from_slice(&v.to_bits().to_le_bytes());
    }
    data
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `u32` file (values 1..6).
pub fn write_multichunk_2x3_u32_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        DATASET_DTYPE_TAG_V1.u32,
        &le_row_major_2x3_u32_one_to_six(),
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `u64` file (values 1..6).
pub fn write_multichunk_2x3_u64_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        DATASET_DTYPE_TAG_V1.u64,
        &le_row_major_2x3_u64_one_to_six(),
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `f16` file (values 1..6).
pub fn write_multichunk_2x3_f16_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        DATASET_DTYPE_TAG_V1.f16,
        &le_row_major_2x3_f16_one_to_six(),
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `i16` file (values 1..6).
pub fn write_multichunk_2x3_i16_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        DATASET_DTYPE_TAG_V1.i16,
        &le_row_major_2x3_i16_one_to_six(),
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `u8` file (values 1..6).
pub fn write_multichunk_2x3_u8_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        DATASET_DTYPE_TAG_V1.u8,
        &le_row_major_2x3_u8_one_to_six(),
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `i32` file (values 1..6).
pub fn write_multichunk_2x3_i32_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        DATASET_DTYPE_TAG_V1.i32,
        &le_row_major_2x3_i32_one_to_six(),
    );
}

/// Write a single-dataset `[2,3]` / `[2,2]` multi-chunk raw `i64` file (values 1..6).
pub fn write_multichunk_2x3_i64_tiles(path: &Path, dataset_name: &str) {
    write_multichunk_2x3(
        path,
        dataset_name,
        CHUNK_PAYLOAD_CODEC_V1.raw,
        DATASET_DTYPE_TAG_V1.i64,
        &le_row_major_2x3_i64_one_to_six(),
    );
}

/// Same geometry as [`write_multichunk_2x3_tiles`], but chunk payloads are **zstd**-compressed.
pub fn write_multichunk_2x3_zstd(path: &Path, dataset_name: &str, data: &[u8]) {
    write_multichunk_2x3_f32(path, dataset_name, CHUNK_PAYLOAD_CODEC_V1.zstd, data);
}

/// Same geometry as [`write_multichunk_2x3_tiles`], with per-file execution settings in the index header.
pub fn write_multichunk_2x3_with_execution(
    path: &Path,
    dataset_name: &str,
    file_execution: FileExecutionSettingsV1,
) {
    write_raw_array_file(
        path,
        &RawArrayWrite {
            name: dataset_name,
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &le_row_major_2x3_f32_one_to_six(),
            file_execution: Some(file_execution),
        },
    )
    .unwrap();
}

/// Same geometry as [`write_multichunk_2x3_tiles`], but chunk payloads are **zstd**-compressed
/// (all-zero `f32` tensor so frames shrink on disk).
pub fn write_multichunk_2x3_zero_zstd(path: &Path, dataset_name: &str) {
    write_multichunk_2x3_zstd(path, dataset_name, &vec![0u8; 24]);
}

/// Shape `[34, 64]` with chunk `[4, 4]` → **144** tiles (quick verify samples 128).
pub const VERIFY_LARGE_SHAPE: [u64; 2] = [34, 64];
pub const VERIFY_LARGE_CHUNK: [u64; 2] = [4, 4];

/// Write a raw `f32` grid with more than [`crate::verify::DEEP_DECODE_MAX_CHUNKS`] chunks.
pub fn write_verify_large_f32(path: &Path, dataset_name: &str) {
    let ne = usize::try_from(VERIFY_LARGE_SHAPE[0] * VERIFY_LARGE_SHAPE[1]).unwrap();
    let mut data = vec![0u8; ne * 4];
    for (slot, i) in data.chunks_exact_mut(4).zip(0_u32..) {
        slot.copy_from_slice(&(i as f32).to_le_bytes());
    }
    write_raw_array_file(
        path,
        &RawArrayWrite {
            name: dataset_name,
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &VERIFY_LARGE_SHAPE,
            chunk_shape: &VERIFY_LARGE_CHUNK,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &data,
            file_execution: None,
        },
    )
    .unwrap();
}

fn corrupt_footer_json_bytes(data: &mut [u8]) {
    const TAIL: usize = 16;
    let json_len = u64::from_le_bytes(
        data[data.len() - TAIL..data.len() - TAIL + 8]
            .try_into()
            .unwrap(),
    );
    let json_end = data.len() - TAIL;
    let json_start = json_end - usize::try_from(json_len).unwrap();
    for b in &mut data[json_start..json_end] {
        *b = b'X';
    }
}

/// Multichunk `f32` plus a corrupt history footer (for `tet verify` / `tet repair plan`).
pub fn write_repair_plan_fixture(path: &Path, dataset_name: &str) {
    use crate::catalog::{FooterBlobV1, TetMetadataV1, write_footer_blob};

    write_multichunk_2x3_tiles(path, dataset_name);
    write_footer_blob(
        path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1::default()),
            metadata_ref: None,
        },
    )
    .unwrap();
    let mut data = std::fs::read(path).unwrap();
    corrupt_footer_json_bytes(&mut data);
    std::fs::write(path, &data).unwrap();
}

/// Regenerate tracked files under `fixtures/small/tet/` (see `fixtures/small/tet/README.md`).
pub fn write_tracked_small_tet_fixtures(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    write_multichunk_2x3_tiles(&dir.join("sample.tet"), "temperature");
    write_verify_large_f32(&dir.join("large.tet"), "a");
    write_repair_plan_fixture(&dir.join("plan.tet"), "a");
    write_multichunk_2x3_u8_tiles(&dir.join("multichunk_u8.tet"), "a");
    write_multichunk_2x3_u32_tiles(&dir.join("multichunk_u32.tet"), "a");
    write_multichunk_2x3_f16_tiles(&dir.join("multichunk_f16.tet"), "a");
}

/// Path to committed manual fixtures: `fixtures/small/tet/`.
pub fn tracked_small_tet_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/small/tet")
}

/// Load paired query documents from `fixtures/queries/{stem}.{json|toml}`.
pub mod query_files {
    use std::path::PathBuf;

    use crate::query::{QueryDocument, parse_query_json, parse_query_toml};

    #[must_use]
    pub fn dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/queries")
    }

    #[must_use]
    pub fn path(stem: &str, ext: &str) -> PathBuf {
        dir().join(format!("{stem}.{ext}"))
    }

    #[must_use]
    pub fn read(stem: &str, ext: &str) -> String {
        let path = path(stem, ext);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    #[must_use]
    pub fn json(stem: &str) -> QueryDocument {
        parse_query_json(&read(stem, "json")).unwrap()
    }

    #[must_use]
    pub fn toml(stem: &str) -> QueryDocument {
        parse_query_toml(&read(stem, "toml")).unwrap()
    }
}
