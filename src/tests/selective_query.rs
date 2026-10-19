//! Selective value-filter demo: zone pushdown must skip chunks (THI-547).
//!
//! Fails if `zone_filter_skipped_chunks` is zero on a multi-chunk package where
//! a selective `filter` cannot match every chunk's zone bounds.

use std::fs;

use crate::catalog::{TetDatasetWrite, TetPackage, TetShardWriter};
use crate::query::{ExecuteQueryOptions, execute_query_document, parse_query_json, validate_query};

fn f32_le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Eight elements, chunk shape 2 → four chunks with disjoint ranges:
/// [0,1], [10,11], [20,21], [30,31].
fn publish_ladder(pkg: &TetPackage) {
    let vals = [0.0_f32, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
    let mut w = TetShardWriter::create(pkg, "s", "ladder").unwrap();
    w.push_dataset(
        TetDatasetWrite::f32_row_major("temp", &[vals.len() as u64], &[2], f32_le(&vals)).unwrap(),
    )
    .unwrap();
    w.seal().unwrap();
    crate::catalog::publish_parts(pkg, 1, &["ladder".to_owned()]).unwrap();
}

#[test]
fn selective_filter_must_skip_chunks_on_zone_maps() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("selective.tet.d")).unwrap();
    publish_ladder(&pkg);
    let packed = pkg.packed_tet_path(1);
    let bytes = fs::read(&packed).unwrap();

    let full = parse_query_json(r#"{"dataset":"temp","sum":[]}"#).unwrap();
    validate_query(&full).unwrap();
    let full_resp = execute_query_document(
        &full,
        &packed,
        &bytes,
        ExecuteQueryOptions::execute_no_preview(),
        None,
    )
    .unwrap();
    let full_plan = full_resp.read_plan.expect("full plan");
    assert_eq!(full_plan.chunk_count, 4, "fixture must be multi-chunk");
    assert_eq!(full_plan.zone_filter_skipped_chunks, 0);

    // Only the top ladder rung can match.
    let selective = parse_query_json(r#"{"dataset":"temp","filter":{"ge":25},"sum":[]}"#).unwrap();
    validate_query(&selective).unwrap();
    let sel_resp = execute_query_document(
        &selective,
        &packed,
        &bytes,
        ExecuteQueryOptions::execute_no_preview(),
        None,
    )
    .unwrap();
    let sel_plan = sel_resp.read_plan.expect("selective plan");

    assert!(
        sel_plan.zone_filter_skipped_chunks > 0,
        "selective filter must skip at least one chunk via zone maps (got skip=0, kept={})",
        sel_plan.chunk_count
    );
    assert!(
        sel_plan.chunk_count < full_plan.chunk_count,
        "selective plan must touch fewer chunks than geometric full (kept={}, full={})",
        sel_plan.chunk_count,
        full_plan.chunk_count
    );
    assert_eq!(sel_plan.zone_filter_skipped_chunks, 3);
    assert_eq!(sel_plan.chunk_count, 1);
    assert_eq!(sel_plan.chunks[0].chunk_index, vec![3]);

    let full_ex = full_resp.execution.expect("full exec");
    let sel_ex = sel_resp.execution.expect("selective exec");
    assert!(
        sel_ex.total_bytes_read_from_disk < full_ex.total_bytes_read_from_disk,
        "selective execute must read fewer payload bytes (sel={}, full={})",
        sel_ex.total_bytes_read_from_disk,
        full_ex.total_bytes_read_from_disk
    );
}
