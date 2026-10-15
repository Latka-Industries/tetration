//! Dense embedder materialize API (`materialize_query_selection`, `materialize_query_transform_ram`).

use super::fixture::{
    write_multichunk_2x3_f16_tiles, write_multichunk_2x3_tiles, write_multichunk_2x3_u32_tiles,
    write_multichunk_2x3_u64_tiles,
};
use crate::layout::mmap_file_read;
use crate::query::{
    DenseBuffer, estimate_query_selection_bytes, materialize_query_selection,
    materialize_query_transform_ram, parse_query_json, plan_query_with_tet_mmap,
    plan_read_for_document, validate_query,
};

fn pop_std_1_to_6() -> f64 {
    let mean = 3.5;
    let var = (1..=6)
        .map(|n| {
            let d = f64::from(n) - mean;
            d * d
        })
        .sum::<f64>()
        / 6.0;
    var.sqrt()
}

#[test]
fn materialize_query_selection_f32_full_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dense.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let out = materialize_query_selection(&doc, &mmap).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    match out.buffer {
        DenseBuffer::F32(vals) => {
            assert_eq!(vals.len(), 6);
            let want = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
            for (got, expected) in vals.iter().zip(want) {
                assert!((got - expected).abs() < 1e-5, "got {got}, want {expected}");
            }
        }
        other => panic!("expected f32 buffer, got {other:?}"),
    }
}

#[test]
fn materialize_query_selection_honors_slice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("slice.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(
        r#"{"dataset":"a","selection":[{"start":0,"stop":1},{"start":0,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let out = materialize_query_selection(&doc, &mmap).unwrap();
    assert_eq!(out.shape, vec![1, 3]);
    match out.buffer {
        DenseBuffer::F32(vals) => assert_eq!(vals, vec![1.0, 2.0, 3.0]),
        other => panic!("expected f32 buffer, got {other:?}"),
    }
}

#[test]
fn materialize_query_selection_rejects_operation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("op.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","mean":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let err = materialize_query_selection(&doc, &mmap).unwrap_err();
    assert!(err.to_string().contains("selection-only"), "{err}");
}

#[test]
fn estimate_query_selection_bytes_full_f32() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("est.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    assert_eq!(estimate_query_selection_bytes(&doc, &mmap).unwrap(), 24);
}

#[test]
fn materialize_query_selection_rejects_over_budget() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("budget.tet");
    write_multichunk_2x3_tiles(&path, "a");
    // 6 f32 = 24 bytes; budget of 8 must fail before decode.
    let doc = parse_query_json(r#"{"dataset":"a","execution":{"memory_budget_bytes":8}}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let err = materialize_query_selection(&doc, &mmap).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("exceeds memory_budget_bytes"), "{msg}");
    assert!(msg.contains("24"), "{msg}");
    assert!(msg.contains('8'), "{msg}");
}

#[test]
fn materialize_query_selection_honors_raised_budget() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("okbudget.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc =
        parse_query_json(r#"{"dataset":"a","execution":{"memory_budget_bytes":24}}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let out = materialize_query_selection(&doc, &mmap).unwrap();
    match out.buffer {
        DenseBuffer::F32(vals) => assert_eq!(vals.len(), 6),
        other => panic!("expected f32 buffer, got {other:?}"),
    }
}

#[test]
fn materialize_query_selection_u32() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u32.tet");
    write_multichunk_2x3_u32_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let out = materialize_query_selection(&doc, &mmap).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    match out.buffer {
        DenseBuffer::U32(vals) => assert_eq!(vals, vec![1, 2, 3, 4, 5, 6]),
        other => panic!("expected u32 buffer, got {other:?}"),
    }
}

#[test]
fn materialize_query_selection_u64() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u64.tet");
    write_multichunk_2x3_u64_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let out = materialize_query_selection(&doc, &mmap).unwrap();
    match out.buffer {
        DenseBuffer::U64(vals) => assert_eq!(vals, vec![1, 2, 3, 4, 5, 6]),
        other => panic!("expected u64 buffer, got {other:?}"),
    }
}

#[test]
fn materialize_query_selection_f16() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f16.tet");
    write_multichunk_2x3_f16_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let out = materialize_query_selection(&doc, &mmap).unwrap();
    match out.buffer {
        DenseBuffer::F16(vals) => {
            assert_eq!(vals.len(), 6);
            for (i, v) in vals.iter().enumerate() {
                let expected = half::f16::from_f32((i + 1) as f32);
                assert_eq!(*v, expected, "index {i}");
            }
        }
        other => panic!("expected f16 buffer, got {other:?}"),
    }
}

#[test]
fn plan_read_for_document_matches_query_read_plan() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plan.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(
        r#"{"dataset":"a","selection":[{"start":0,"stop":1},{"start":0,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let planned = plan_read_for_document(&doc, &mmap).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = response.read_plan.as_ref().unwrap();
    assert_eq!(
        planned.read_plan.logical_selection_shape,
        rp.logical_selection_shape
    );
    assert_eq!(planned.read_plan.chunk_count, rp.chunk_count);
    assert_eq!(planned.read_plan.chunks.len(), rp.chunks.len());
}

#[test]
fn materialize_query_transform_ram_zscore_full_buffer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zscore.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","transform":{"method":"zscore"},"write":"ram"}"#)
        .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let out = materialize_query_transform_ram(&doc, &mmap, &path, None).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    let std = pop_std_1_to_6();
    match out.buffer {
        DenseBuffer::F32(vals) => {
            assert_eq!(vals.len(), 6);
            for (i, &v) in vals.iter().enumerate() {
                let raw = f64::from((i + 1) as f32);
                let expected = ((raw - 3.5) / std) as f32;
                assert!(
                    (v - expected).abs() < 1e-4,
                    "index {i}: got {v}, expected {expected}"
                );
            }
        }
        other => panic!("expected f32 buffer, got {other:?}"),
    }
}

#[test]
fn materialize_query_transform_ram_rejects_sidecar_write() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sidecar.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(
        r#"{"dataset":"a","transform":{"method":"zscore"},"write":{"target":"sidecar","timestamp":false}}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let err = materialize_query_transform_ram(&doc, &mmap, &path, None).unwrap_err();
    assert!(err.to_string().contains("ram"), "{err}");
}

#[test]
fn materialize_query_transform_ram_requires_transform_operation() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let err = materialize_query_transform_ram(&doc, &mmap, &path, None).unwrap_err();
    assert!(err.to_string().contains("transform"), "{err}");
}
