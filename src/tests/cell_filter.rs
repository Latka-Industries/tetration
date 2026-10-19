//! Cell filter on kept chunks (THI-563). Zone skip still drops impossible chunks;
//! folds must ignore non-matching cells inside a chunk that was kept.

use std::fs;

use crate::catalog::{
    DATASET_DTYPE_TAG_V1, TetDatasetWrite, TetPackage, TetShardWriter, publish_parts,
};
use crate::query::{
    ExecuteQueryOptions, SpillPathAllowlist, execute_query_document, materialize_read_plan_f32_le,
    materialize_read_plan_i32_le, parse_query_json, plan_query_with_tet_mmap,
    plan_query_with_tet_mmap_ex, validate_query,
};

fn f32_le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_le(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn publish_row(
    pkg: &TetPackage,
    dtype: u32,
    shape: &[u64],
    chunk_shape: &[u64],
    data: Vec<u8>,
    fill: Option<&str>,
) {
    let mut write =
        TetDatasetWrite::row_major("a", dtype, shape, chunk_shape, data).expect("dataset");
    if let Some(fill) = fill {
        write.attrs.insert("_FillValue".to_owned(), fill.to_owned());
    }
    let mut w = TetShardWriter::create(pkg, "s", "only").unwrap();
    w.push_dataset(write).unwrap();
    w.seal().unwrap();
    publish_parts(pkg, 1, &["only".to_owned()]).unwrap();
}

fn execute_sum(path: &std::path::Path, bytes: &[u8], query: &str) -> crate::query::QueryResponse {
    let doc = parse_query_json(query).unwrap();
    validate_query(&doc).unwrap();
    execute_query_document(
        &doc,
        path,
        bytes,
        ExecuteQueryOptions::execute_no_preview(),
        None,
    )
    .unwrap()
}

#[test]
fn kept_chunk_sum_excludes_non_matching_cells() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("cells.tet.d")).unwrap();
    // Chunks of 2: [-1, 4] kept, [-2, 8] kept, [-9, -8] skipped by ge 0.
    publish_row(
        &pkg,
        DATASET_DTYPE_TAG_V1.f32,
        &[6],
        &[2],
        f32_le(&[-1.0, 4.0, -2.0, 8.0, -9.0, -8.0]),
        None,
    );
    let packed = pkg.packed_tet_path(1);
    let bytes = fs::read(&packed).unwrap();

    let filtered = execute_sum(
        &packed,
        &bytes,
        r#"{"dataset":"a","filter":{"ge":0},"sum":[]}"#,
    );
    let plan = filtered.read_plan.expect("plan");
    assert_eq!(plan.zone_filter_skipped_chunks, 1);
    assert_eq!(
        plan.chunk_count, 2,
        "mixed chunks stay; only the all-negative chunk skips"
    );
    let ex = filtered.execution.expect("execution");
    assert_eq!(ex.operation_sum, Some(12.0));
    assert_eq!(ex.operation_element_count, Some(2));

    let unfiltered = execute_sum(&packed, &bytes, r#"{"dataset":"a","sum":[]}"#);
    let full = unfiltered.execution.expect("execution");
    assert_eq!(full.operation_sum, Some(-8.0));
    assert_ne!(full.operation_sum, ex.operation_sum);
}

#[test]
fn partial_sum_on_kept_chunk_drops_negative_cells() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("partial.tet.d")).unwrap();
    // One chunk so the zone map cannot drop it: [[-1, 4], [-2, 5]].
    publish_row(
        &pkg,
        DATASET_DTYPE_TAG_V1.f32,
        &[2, 2],
        &[2, 2],
        f32_le(&[-1.0, 4.0, -2.0, 5.0]),
        None,
    );
    let packed = pkg.packed_tet_path(1);
    let bytes = fs::read(&packed).unwrap();
    let resp = execute_sum(
        &packed,
        &bytes,
        r#"{"dataset":"a","filter":{"ge":0},"sum":["1"]}"#,
    );
    let plan = resp.read_plan.expect("plan");
    assert_eq!(plan.chunk_count, 1);
    assert_eq!(plan.zone_filter_skipped_chunks, 0);
    let ex = resp.execution.expect("execution");
    assert_eq!(ex.operation_reduced_sum.as_deref(), Some(&[4.0, 5.0][..]));
}

#[test]
fn is_null_counts_nan_inside_a_kept_chunk() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("nan.tet.d")).unwrap();
    publish_row(
        &pkg,
        DATASET_DTYPE_TAG_V1.f32,
        &[3],
        &[3],
        f32_le(&[f32::NAN, 1.0, 2.0]),
        None,
    );
    let packed = pkg.packed_tet_path(1);
    let bytes = fs::read(&packed).unwrap();
    let resp = execute_sum(
        &packed,
        &bytes,
        r#"{"dataset":"a","filter":{"is_null":true},"count":[]}"#,
    );
    let plan = resp.read_plan.expect("plan");
    assert_eq!(plan.chunk_count, 1);
    let ex = resp.execution.expect("execution");
    assert_eq!(ex.operation_element_count, Some(1));
}

#[test]
fn fill_value_is_excluded_from_numeric_fold() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("fill.tet.d")).unwrap();
    publish_row(
        &pkg,
        DATASET_DTYPE_TAG_V1.f32,
        &[2],
        &[2],
        f32_le(&[5.0, 4.0]),
        Some("5"),
    );
    let packed = pkg.packed_tet_path(1);
    let bytes = fs::read(&packed).unwrap();
    let resp = execute_sum(
        &packed,
        &bytes,
        r#"{"dataset":"a","filter":{"ge":0},"sum":[]}"#,
    );
    let plan = resp.read_plan.expect("plan");
    assert_eq!(plan.chunk_count, 1, "non-fill 4 keeps the chunk under ge 0");
    let ex = resp.execution.expect("execution");
    assert_eq!(ex.operation_sum, Some(4.0));
    assert_eq!(ex.operation_element_count, Some(1));
}

#[test]
fn i32_fold_uses_promoted_compare() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("i32.tet.d")).unwrap();
    publish_row(
        &pkg,
        DATASET_DTYPE_TAG_V1.i32,
        &[3],
        &[3],
        i32_le(&[-1, 2, 5]),
        None,
    );
    let packed = pkg.packed_tet_path(1);
    let bytes = fs::read(&packed).unwrap();
    let resp = execute_sum(
        &packed,
        &bytes,
        r#"{"dataset":"a","filter":{"ge":0},"sum":[]}"#,
    );
    let ex = resp.execution.expect("execution");
    assert_eq!(ex.operation_sum, Some(7.0));
    assert_eq!(ex.operation_element_count, Some(2));
}

#[test]
fn materialize_keeps_shape_and_masks_non_matches() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("dense.tet.d")).unwrap();
    publish_row(
        &pkg,
        DATASET_DTYPE_TAG_V1.f32,
        &[3],
        &[3],
        f32_le(&[-1.0, 4.0, 2.0]),
        None,
    );
    let packed = pkg.packed_tet_path(1);
    let bytes = fs::read(&packed).unwrap();
    let path = packed.display().to_string();
    let doc = parse_query_json(r#"{"dataset":"a","filter":{"ge":0}}"#).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, Some(&path), &bytes, None)
        .unwrap()
        .read_plan
        .expect("plan");
    assert_eq!(plan.logical_element_count, 3);
    let (vals, _, _) = materialize_read_plan_f32_le(&bytes, &plan, None).unwrap();
    assert_eq!(vals.len(), 3);
    assert!(vals[0].is_nan());
    assert_eq!(vals[1], 4.0);
    assert_eq!(vals[2], 2.0);
}

#[test]
fn integer_materialize_uses_fill_and_refuses_without_it() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("idense.tet.d")).unwrap();
    publish_row(
        &pkg,
        DATASET_DTYPE_TAG_V1.i32,
        &[3],
        &[3],
        i32_le(&[1, -3, 4]),
        Some("-99"),
    );
    let packed = pkg.packed_tet_path(1);
    let bytes = fs::read(&packed).unwrap();
    let path = packed.display().to_string();
    let doc = parse_query_json(r#"{"dataset":"a","filter":{"ge":0}}"#).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, Some(&path), &bytes, None)
        .unwrap()
        .read_plan
        .expect("plan");
    let (vals, _, _) = materialize_read_plan_i32_le(&bytes, &plan, None).unwrap();
    assert_eq!(vals, vec![1, -99, 4]);

    let bare = TetPackage::create(dir.path().join("nofill.tet.d")).unwrap();
    publish_row(
        &bare,
        DATASET_DTYPE_TAG_V1.i32,
        &[2],
        &[2],
        i32_le(&[-1, 2]),
        None,
    );
    let bare_packed = bare.packed_tet_path(1);
    let bare_bytes = fs::read(&bare_packed).unwrap();
    let bare_path = bare_packed.display().to_string();
    let bare_plan = plan_query_with_tet_mmap(&doc, Some(&bare_path), &bare_bytes, None)
        .unwrap()
        .read_plan
        .expect("plan");
    let err = materialize_read_plan_i32_le(&bare_bytes, &bare_plan, None).unwrap_err();
    assert!(err.to_string().contains("requires a dataset fill"), "{err}");
}

#[test]
fn transform_zscore_masks_non_matching_cells() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("z.tet.d")).unwrap();
    publish_row(
        &pkg,
        DATASET_DTYPE_TAG_V1.f32,
        &[3],
        &[3],
        f32_le(&[1.0, -1.0, 3.0]),
        None,
    );
    let packed = pkg.packed_tet_path(1);
    let bytes = fs::read(&packed).unwrap();
    let doc = parse_query_json(
        r#"{"dataset":"a","filter":{"ge":0},"transform":{"method":"zscore"},"write":"ram"}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let policy =
        SpillPathAllowlist::from_roots(dir.path().to_path_buf(), [dir.path().to_path_buf()]);
    let resp = plan_query_with_tet_mmap_ex(
        &doc,
        Some(packed.to_str().unwrap()),
        &bytes,
        Some(3),
        Some(&policy),
    )
    .unwrap();
    let ex = resp.execution.expect("execution");
    assert!((ex.operation_mean.unwrap() - 2.0).abs() < 1e-5);
    assert_eq!(ex.f32_preview.len(), 3);
    assert!((ex.f32_preview[0] - (-1.0)).abs() < 1e-4);
    assert!(ex.f32_preview[1].is_nan());
    assert!((ex.f32_preview[2] - 1.0).abs() < 1e-4);
}
