//! THI-378: conformance reject corpus — TIDX / payload offset faults.
//!
//! Checked-in under `fixtures/small/tet/reject/`. Each case must make
//! [`verify_tet_bytes`] fail with a stable finding `check` name (no panic).
//! Fuzz seeds are refreshed via `mise fuzz-reseed`.

use std::path::{Path, PathBuf};

use crate::catalog::{CHUNK_PAYLOAD_CODEC_V1, read_tet_summary_v1};
use crate::layout::mmap_file_read;
use crate::verify::{VerifyOptions, verify_tet_bytes};

use super::fixture::{
    self,
    index_patch::{self, ENTRY_CHUNK_INDEX0_OFFSET, ENTRY_PAYLOAD_OFFSET},
    write_multichunk_2x3_tiles, write_multichunk_2x3_zero_zstd,
};

/// One reject fixture: basename under `reject/` + expected failing verify `check`.
struct RejectCase {
    name: &'static str,
    /// Primary finding `check` that must fail (`ok == false`).
    expected_check: &'static str,
    build: fn(&Path),
}

fn reject_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/small/tet/reject")
}

fn write_base_raw(path: &Path) {
    write_multichunk_2x3_tiles(path, "a");
}

fn build_payload_offset_past_eof(path: &Path) {
    write_base_raw(path);
    index_patch::patch_first_index_entry_u64(path, ENTRY_PAYLOAD_OFFSET, u64::MAX / 2);
}

fn build_duplicate_payload_offset(path: &Path) {
    write_base_raw(path);
    let off0 = {
        let mmap = mmap_file_read(path).unwrap();
        let s = read_tet_summary_v1(&mmap).unwrap();
        assert!(s.chunks.len() >= 2, "need multi-chunk fixture");
        s.chunks[0].payload_offset
    }; // drop mmap before rewrite (Windows rejects write on mapped files)
    index_patch::patch_entry_u64(path, 1, ENTRY_PAYLOAD_OFFSET, off0);
}

fn build_raw_stored_len_mismatch(path: &Path) {
    write_base_raw(path);
    // codec=0 requires raw == stored; diverge them.
    index_patch::patch_first_index_entry_raw_and_stored(path, 16, 8);
}

fn build_chunk_coord_out_of_grid(path: &Path) {
    write_base_raw(path);
    // shape [2,3] / chunk [2,2] → grid [1,2]; set axis-0 coord out of range.
    index_patch::patch_first_index_entry_u64(path, ENTRY_CHUNK_INDEX0_OFFSET, 99);
}

fn build_truncated_after_index(path: &Path) {
    write_base_raw(path);
    let cut = {
        let mmap = mmap_file_read(path).unwrap();
        let s = read_tet_summary_v1(&mmap).unwrap();
        let first = &s.chunks[0];
        // Keep index intact but chop the first payload mid-span → parse or decode fails.
        let start = first.payload_offset;
        (start + first.stored_byte_len / 2).max(start + 1)
    };
    index_patch::truncate_file(path, cut);
}

fn build_truncated_tidx(path: &Path) {
    write_base_raw(path);
    let idx = {
        let mmap = mmap_file_read(path).unwrap();
        let s = read_tet_summary_v1(&mmap).unwrap();
        s.superblock.chunk_index_offset
    };
    // Truncate inside the index region (after header, mid entry).
    index_patch::truncate_file(path, idx + 40);
}

fn build_zstd_garbage_payload(path: &Path) {
    write_multichunk_2x3_zero_zstd(path, "a");
    let (off, n) = {
        let mmap = mmap_file_read(path).unwrap();
        let s = read_tet_summary_v1(&mmap).unwrap();
        assert!(
            s.chunks
                .iter()
                .any(|c| CHUNK_PAYLOAD_CODEC_V1.is_zstd(c.codec)),
            "expected zstd chunks"
        );
        let ch = &s.chunks[0];
        let off = usize::try_from(ch.payload_offset).unwrap();
        let n = usize::try_from(ch.stored_byte_len.min(8)).unwrap().max(1);
        (off, n)
    };
    index_patch::patch_bytes_at(path, off, &vec![0xA5u8; n]);
}

fn build_inflated_stored_past_eof(path: &Path) {
    write_base_raw(path);
    let bad_stored = {
        let mmap = mmap_file_read(path).unwrap();
        let s = read_tet_summary_v1(&mmap).unwrap();
        let file_len = mmap.len() as u64;
        let ch = &s.chunks[0];
        // Keep offset valid but inflate stored so span crosses EOF.
        // Keep raw == stored so codec=0 does not fail earlier on RawStoredMismatch.
        file_len
            .saturating_sub(ch.payload_offset)
            .saturating_add(64)
    };
    index_patch::patch_first_index_entry_raw_and_stored(path, bad_stored, bad_stored);
}

const CASES: &[RejectCase] = &[
    RejectCase {
        name: "payload_offset_past_eof.tet",
        expected_check: "parse",
        build: build_payload_offset_past_eof,
    },
    RejectCase {
        name: "duplicate_payload_offset.tet",
        expected_check: "payload_offsets",
        build: build_duplicate_payload_offset,
    },
    RejectCase {
        name: "raw_stored_len_mismatch.tet",
        expected_check: "parse",
        build: build_raw_stored_len_mismatch,
    },
    RejectCase {
        name: "chunk_coord_out_of_grid.tet",
        expected_check: "dataset_tensor_bytes",
        build: build_chunk_coord_out_of_grid,
    },
    RejectCase {
        name: "truncated_payload.tet",
        expected_check: "parse",
        build: build_truncated_after_index,
    },
    RejectCase {
        name: "truncated_tidx.tet",
        expected_check: "parse",
        build: build_truncated_tidx,
    },
    RejectCase {
        name: "zstd_garbage_payload.tet",
        expected_check: "chunk_decode",
        build: build_zstd_garbage_payload,
    },
    RejectCase {
        name: "inflated_stored_past_eof.tet",
        expected_check: "parse",
        build: build_inflated_stored_past_eof,
    },
];

/// Legacy THI-376 crash seeds — only assert `!ok` (finding codes vary by seed).
const LEGACY_REJECT: &[&str] = &[
    "overflow_tile_extent.tet",
    "oom_dataset_count.tet",
    "zero_chunk_shape.tet",
];

fn assert_reject_bytes(name: &str, data: &[u8], path: &Path, expected_check: &str) {
    let report = verify_tet_bytes(data, Some(path), VerifyOptions::default());
    assert!(
        !report.ok,
        "{name}: expected verify failure, got ok findings={:?}",
        report.findings
    );
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == expected_check && !f.ok),
        "{name}: expected failing check `{expected_check}`, findings={:?}",
        report.findings
    );
}

#[test]
fn reject_corpus_cases_fail_with_stable_checks() {
    let dir = tempfile::tempdir().unwrap();
    for case in CASES {
        let path = dir.path().join(case.name);
        (case.build)(&path);
        let data = std::fs::read(&path).unwrap();
        assert_reject_bytes(case.name, &data, &path, case.expected_check);
    }
}

#[test]
fn reject_corpus_checked_in_fixtures_match_builders() {
    let root = reject_dir();
    for case in CASES {
        let path = root.join(case.name);
        assert!(
            path.is_file(),
            "missing checked-in reject fixture {} (run UPDATE_REJECT_CORPUS=1 cargo test regenerate_reject_corpus -- --ignored)",
            path.display()
        );
        let data = std::fs::read(&path).unwrap();
        assert_reject_bytes(case.name, &data, &path, case.expected_check);
    }
    for &name in LEGACY_REJECT {
        let path = root.join(name);
        let data = std::fs::read(&path).expect("legacy reject fixture");
        let report = verify_tet_bytes(&data, Some(&path), VerifyOptions::default());
        assert!(!report.ok, "{name}: legacy reject must fail verify");
    }
}

/// Write / refresh `fixtures/small/tet/reject/*.tet` for the THI-378 case set.
///
/// ```bash
/// UPDATE_REJECT_CORPUS=1 cargo test --lib regenerate_reject_corpus -- --ignored --nocapture
/// mise fuzz-reseed
/// ```
#[test]
#[ignore = "set UPDATE_REJECT_CORPUS=1 to rewrite fixtures/small/tet/reject"]
fn regenerate_reject_corpus() {
    assert_eq!(
        std::env::var("UPDATE_REJECT_CORPUS").ok().as_deref(),
        Some("1"),
        "refusing to rewrite fixtures without UPDATE_REJECT_CORPUS=1"
    );
    let root = reject_dir();
    std::fs::create_dir_all(&root).unwrap();
    for case in CASES {
        let path = root.join(case.name);
        (case.build)(&path);
        eprintln!("wrote {}", path.display());
        let data = std::fs::read(&path).unwrap();
        assert_reject_bytes(case.name, &data, &path, case.expected_check);
    }
    // Keep fixture helper referenced for docs / discoverability.
    let _ = fixture::SHAPE_2X3;
}
