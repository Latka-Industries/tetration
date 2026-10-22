//! Fuzz seal-time secondary index build from a tiny packed `.tet` (value-postings path).
//!
//! Exercises [`tetration::catalog::prepare_secondary_index`] / embed after writing mutated
//! tile bytes. Must not panic — including high-cardinality float tiles that used to allocate
//! a full per-chunk distinct set before the 64-cap could fire.
//!
//! ```bash
//! cargo +nightly fuzz run secondary_index_seal
//! # or: mise fuzz-index-seal / mise fuzz-index-seal-smoke
//! ```

#![no_main]

use std::path::Path;

use libfuzzer_sys::fuzz_target;
use tetration::catalog::{
    CHUNK_PAYLOAD_CODEC_V1, DATASET_DTYPE_TAG_V1, EMBEDDED_STANDALONE_GENERATION, RawArrayWrite,
    embed_indexes_in_tet, prepare_secondary_index, secondary_index_for_packed, write_raw_array_file,
};

/// Cap input so each iteration stays cheap (tile + header).
const MAX_INPUT: usize = 8 * 1024;
/// Max elements in the synthetic dataset (keeps packed.tet tiny).
const MAX_ELEMS: u64 = 256;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 || data.len() > MAX_INPUT {
        return;
    }

    let (dtype, elem_size) = match data[0] % 5 {
        0 => (DATASET_DTYPE_TAG_V1.u8, 1usize),
        1 => (DATASET_DTYPE_TAG_V1.f32, 4),
        2 => (DATASET_DTYPE_TAG_V1.f64, 8),
        3 => (DATASET_DTYPE_TAG_V1.i32, 4),
        _ => (DATASET_DTYPE_TAG_V1.u16, 2),
    };

    let n_elems = ((data[1] as u64) % MAX_ELEMS).max(1);
    let chunk_elems = ((data[2] as u64) % n_elems).max(1);
    let need = usize::try_from(n_elems)
        .ok()
        .and_then(|n| n.checked_mul(elem_size))
        .unwrap_or(0);
    if need == 0 || need > MAX_INPUT {
        return;
    }

    let mut payload = vec![0u8; need];
    let src = &data[3..];
    let copy = src.len().min(need);
    payload[..copy].copy_from_slice(&src[..copy]);

    let Ok(dir) = tempfile::tempdir() else {
        return;
    };
    let path = dir.path().join("seal.tet");
    let shape = [n_elems];
    let chunk_shape = [chunk_elems];
    if write_raw_array_file(
        &path,
        &RawArrayWrite {
            name: "a",
            dtype,
            shape: &shape,
            chunk_shape: &chunk_shape,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &payload,
            file_execution: None,
        },
    )
    .is_err()
    {
        return;
    }

    let Ok(packed) = std::fs::read(&path) else {
        return;
    };
    // Seal-time builders (may omit INDEX when cardinality exceeds the cap).
    let _ = secondary_index_for_packed(&packed, 1, "fuzz");
    let _ = prepare_secondary_index(&packed, 1, "fuzz");
    // Standalone embed path (zones + optional INDEX spills; no full-file SHA).
    let _ = embed_indexes_in_tet(Path::new(&path), EMBEDDED_STANDALONE_GENERATION);
});
