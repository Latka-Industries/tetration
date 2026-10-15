//! THI-377: structured proptest — valid write → TIDX / payload offset invariants.
//!
//! Generates small legal tensors (dtype × chunk geometry × raw/zstd), writes via
//! public catalog APIs, then checks summary geometry, non-overlapping payloads,
//! and full-selection materialize round-trip.

use std::collections::HashSet;

use proptest::prelude::*;

use crate::catalog::tile::{chunk_grid_counts, tile_extent, tile_raw_byte_len, total_chunk_count};
use crate::catalog::{
    CHUNK_PAYLOAD_CODEC_V1, DATASET_DTYPE_TAG_V1, RawArrayWrite, elem_size_for_wire_tag,
    read_tet_summary_v1, write_raw_array_file,
};
use crate::layout::mmap_file_read;
use crate::query::decode::indexing::{coords_from_linear_row_major, linear_rm_index};
use crate::query::{
    materialize_read_plan_f32_le, materialize_read_plan_f64_le, materialize_read_plan_i32_le,
    parse_query_json, plan_query_with_tet_mmap, validate_query,
};
use crate::utils::dtype::ElementDtype;

const DATASET: &str = "field";

/// Cap product(shape) so CI stays fast under zstd + materialize.
const MAX_ELEMS: u64 = 48;
/// Cap chunk tiles so index enumeration stays small.
const MAX_CHUNKS: u64 = 16;

#[derive(Debug, Clone)]
struct LegalTensor {
    dtype: u32,
    shape: Vec<u64>,
    chunk_shape: Vec<u64>,
    codec: u32,
    data: Vec<u8>,
}

fn dtype_tag(choice: u8) -> u32 {
    let tags = DATASET_DTYPE_TAG_V1;
    match choice % 3 {
        0 => tags.f32,
        1 => tags.f64,
        _ => tags.i32,
    }
}

fn fill_pattern(dtype: u32, nelem: usize) -> Vec<u8> {
    let elem = ElementDtype::try_from_wire_tag(dtype).expect("supported dtype");
    let es = elem.elem_size();
    let mut data = vec![0u8; nelem * es];
    for (i, slot) in data.chunks_exact_mut(es).enumerate() {
        match elem {
            ElementDtype::F32 => {
                slot.copy_from_slice(&(i as f32 * 0.25 + 1.0).to_le_bytes());
            }
            ElementDtype::F64 => {
                slot.copy_from_slice(&(i as f64 * 0.25 + 1.0).to_le_bytes());
            }
            ElementDtype::I32 => {
                slot.copy_from_slice(&(i as i32 + 1).to_le_bytes());
            }
            _ => unreachable!("strategy only uses f32/f64/i32"),
        }
    }
    data
}

fn try_build(
    dtype: u32,
    codec: u32,
    shape: Vec<u64>,
    chunk_shape: Vec<u64>,
) -> Option<LegalTensor> {
    let nelem = shape.iter().try_fold(1u64, |a, &b| a.checked_mul(b))?;
    if nelem == 0 || nelem > MAX_ELEMS {
        return None;
    }
    let counts = chunk_grid_counts(&shape, &chunk_shape).ok()?;
    let n_chunks = total_chunk_count(&counts).ok()?;
    if n_chunks == 0 || n_chunks > MAX_CHUNKS {
        return None;
    }
    let data = fill_pattern(dtype, usize::try_from(nelem).ok()?);
    Some(LegalTensor {
        dtype,
        shape,
        chunk_shape,
        codec,
        data,
    })
}

/// One axis: extent + chunk size with `1 <= chunk <= extent <= 6`.
fn axis_strategy() -> impl Strategy<Value = (u64, u64)> {
    (1u64..=6).prop_flat_map(|extent| (Just(extent), 1u64..=extent))
}

fn legal_tensor_strategy() -> impl Strategy<Value = LegalTensor> {
    prop_oneof![
        (any::<u8>(), any::<bool>()).prop_flat_map(|(dtype_choice, use_zstd)| {
            let dtype = dtype_tag(dtype_choice);
            let codec = if use_zstd {
                CHUNK_PAYLOAD_CODEC_V1.zstd
            } else {
                CHUNK_PAYLOAD_CODEC_V1.raw
            };
            axis_strategy().prop_filter_map("1d", move |(e0, c0)| {
                try_build(dtype, codec, vec![e0], vec![c0])
            })
        }),
        (any::<u8>(), any::<bool>()).prop_flat_map(|(dtype_choice, use_zstd)| {
            let dtype = dtype_tag(dtype_choice);
            let codec = if use_zstd {
                CHUNK_PAYLOAD_CODEC_V1.zstd
            } else {
                CHUNK_PAYLOAD_CODEC_V1.raw
            };
            (axis_strategy(), axis_strategy()).prop_filter_map("2d", move |((e0, c0), (e1, c1))| {
                try_build(dtype, codec, vec![e0, e1], vec![c0, c1])
            })
        }),
        (any::<u8>(), any::<bool>()).prop_flat_map(|(dtype_choice, use_zstd)| {
            let dtype = dtype_tag(dtype_choice);
            let codec = if use_zstd {
                CHUNK_PAYLOAD_CODEC_V1.zstd
            } else {
                CHUNK_PAYLOAD_CODEC_V1.raw
            };
            (axis_strategy(), axis_strategy(), axis_strategy()).prop_filter_map(
                "3d",
                move |((e0, c0), (e1, c1), (e2, c2))| {
                    try_build(dtype, codec, vec![e0, e1, e2], vec![c0, c1, c2])
                },
            )
        }),
    ]
}

fn materialize_full_le(mmap: &[u8], dtype: u32) -> Vec<u8> {
    let doc = parse_query_json(&format!(r#"{{"dataset":"{DATASET}"}}"#)).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let tags = DATASET_DTYPE_TAG_V1;
    if tags.is_f32(dtype) {
        let (vals, _, _) = materialize_read_plan_f32_le(mmap, rp, None).unwrap();
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if tags.is_f64(dtype) {
        let (vals, _, _) = materialize_read_plan_f64_le(mmap, rp, None).unwrap();
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else if tags.is_i32(dtype) {
        let (vals, _, _) = materialize_read_plan_i32_le(mmap, rp, None).unwrap();
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        panic!("unexpected dtype {dtype}");
    }
}

fn assert_write_invariants(t: &LegalTensor) -> Result<(), TestCaseError> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prop.tet");
    write_raw_array_file(
        &path,
        &RawArrayWrite {
            name: DATASET,
            dtype: t.dtype,
            shape: &t.shape,
            chunk_shape: &t.chunk_shape,
            chunk_codec: t.codec,
            data: &t.data,
            file_execution: None,
        },
    )
    .expect("legal write must succeed");

    let mmap = mmap_file_read(&path).unwrap();
    let file_len = mmap.len() as u64;
    let summary = read_tet_summary_v1(&mmap).expect("writer output must parse");

    prop_assert_eq!(summary.datasets.len(), 1);
    let ds = &summary.datasets[0];
    prop_assert_eq!(ds.dtype, t.dtype);
    prop_assert_eq!(&ds.shape[..], &t.shape[..]);
    prop_assert_eq!(&ds.chunk_shape[..], &t.chunk_shape[..]);

    let counts = chunk_grid_counts(&t.shape, &t.chunk_shape).unwrap();
    let expected_chunks = total_chunk_count(&counts).unwrap();
    prop_assert_eq!(summary.chunks.len() as u64, expected_chunks);

    let elem_size = elem_size_for_wire_tag(t.dtype).unwrap();
    let ndim = t.shape.len();
    let mut seen_coords: HashSet<Vec<u64>> = HashSet::new();
    let mut spans: Vec<(u64, u64)> = Vec::with_capacity(summary.chunks.len());

    for c in &summary.chunks {
        prop_assert_eq!(c.dataset_id, 0);
        prop_assert_eq!(c.codec, t.codec);

        let coord: Vec<u64> = c.chunk_index[..ndim].to_vec();
        for d in 0..ndim {
            prop_assert!(
                coord[d] < counts[d],
                "chunk coord {coord:?} out of grid {counts:?}"
            );
        }
        prop_assert!(
            seen_coords.insert(coord.clone()),
            "duplicate chunk coord {coord:?}"
        );

        let expect_raw =
            tile_raw_byte_len(&t.shape, &t.chunk_shape, &coord, ndim, elem_size).unwrap();
        prop_assert_eq!(c.raw_byte_len, expect_raw);

        let end = c
            .payload_offset
            .checked_add(c.stored_byte_len)
            .expect("offset+len");
        prop_assert!(
            end <= file_len,
            "payload span past EOF: offset={} stored={} file_len={}",
            c.payload_offset,
            c.stored_byte_len,
            file_len
        );
        if CHUNK_PAYLOAD_CODEC_V1.is_raw(c.codec) {
            prop_assert_eq!(c.raw_byte_len, c.stored_byte_len);
        }
        spans.push((c.payload_offset, end));
    }
    prop_assert_eq!(seen_coords.len(), summary.chunks.len());

    spans.sort_by_key(|&(start, _)| start);
    for w in spans.windows(2) {
        let (_, end0) = w[0];
        let (start1, _) = w[1];
        prop_assert!(end0 <= start1, "overlapping payloads: {spans:?}");
        prop_assert_ne!(w[0].0, w[1].0);
    }

    let nelem = t.shape.iter().product::<u64>();
    let sample_n = usize::try_from(nelem.min(8)).unwrap();
    for k in 0..sample_n {
        let li = if sample_n <= 1 {
            0usize
        } else {
            usize::try_from(k as u64 * (nelem - 1) / (sample_n as u64 - 1)).unwrap()
        };
        let global = coords_from_linear_row_major(li, &t.shape).unwrap();
        let mut chunk_coord = vec![0u64; ndim];
        let mut local = vec![0usize; ndim];
        for d in 0..ndim {
            let g = global[d] as u64;
            chunk_coord[d] = g / t.chunk_shape[d];
            local[d] = usize::try_from(g % t.chunk_shape[d]).unwrap();
        }
        prop_assert!(
            seen_coords.contains(&chunk_coord),
            "missing chunk for linear {li} → global {global:?} chunk {chunk_coord:?}"
        );
        let tile = tile_extent(&t.shape, &t.chunk_shape, &chunk_coord, ndim);
        let local_li = linear_rm_index(&local, &tile).unwrap();
        let back = coords_from_linear_row_major(local_li, &tile).unwrap();
        prop_assert_eq!(back, local);
    }

    let got = materialize_full_le(&mmap, t.dtype);
    prop_assert_eq!(
        got.as_slice(),
        t.data.as_slice(),
        "materialize round-trip mismatch (dtype={}, codec={}, shape={:?}, chunk={:?})",
        t.dtype,
        t.codec,
        t.shape,
        t.chunk_shape
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 48,
        ..ProptestConfig::default()
    })]

    #[test]
    fn valid_write_offset_and_roundtrip_invariants(t in legal_tensor_strategy()) {
        assert_write_invariants(&t)?;
    }
}
