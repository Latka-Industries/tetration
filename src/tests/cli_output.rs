//! CLI query output formatting (`QueryOutputFormat`).

use super::fixture;
use crate::layout::mmap_file_read;
use crate::query::cli::{QueryOutputFormat, format_query_response, format_query_stderr_hints};
use crate::query::{parse_query_json, plan_query_empty, plan_query_with_tet_mmap, validate_query};

#[test]
fn format_full_and_json_include_operation_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fmt.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = super::fixture::query_files::json("mean_a");
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();

    let full = format_query_response(&response, QueryOutputFormat::Full).unwrap();
    assert!(full.contains("\"operation_mean\""));

    let compact = format_query_response(&response, QueryOutputFormat::Json).unwrap();
    assert!(!compact.contains('\n'));
    assert!(compact.contains("\"operation_mean\""));
}

#[test]
fn format_stats_omits_chunk_rows_and_previews() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("stats.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = super::fixture::query_files::json("sum_a");
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(8)).unwrap();

    let stats = format_query_response(&response, QueryOutputFormat::Stats).unwrap();
    assert!(stats.contains("\"operation_sum\""));
    assert!(stats.contains("\"chunk_count\""));
    assert!(!stats.contains("\"chunks\""));
    assert!(!stats.contains("f32_preview"));
}

#[test]
fn format_quiet_scalar_mean() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("quiet.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = super::fixture::query_files::json("mean_a");
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("dataset=a"));
    assert!(line.contains("status=ok"));
    assert!(line.contains("op=mean"));
    assert!(line.contains("mean=3.5"));
    assert!(line.contains("elements=6"));
    assert!(!line.contains('\n'));
}

#[test]
fn format_quiet_partial_sum_along_axis() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("quiet_partial.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = super::fixture::query_files::json("sum_axis0_a");
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("op=sum"));
    assert!(line.contains("axes=[0]"));
    assert!(line.contains("values=[5,7,9]"));
    assert!(line.contains("reduced_shape=3"));
}

#[test]
fn format_quiet_plan_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plan.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("status=planned"));
    assert!(line.contains("chunks=2"));
    assert!(line.contains("logical_shape=2×3"));
}

#[test]
fn format_quiet_unmatched_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("miss.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"missing"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("status=not_found"));
    assert!(line.contains("available=[a]"));
}

#[test]
fn format_plan_omits_chunks_and_execution() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plan_fmt.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let plan_only = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let planned = format_query_response(&plan_only, QueryOutputFormat::Plan).unwrap();
    assert!(planned.contains("\"chunk_count\": 2"));
    assert!(!planned.contains("\"chunks\""));
    assert!(!planned.contains("\"execution\""));

    let doc_op = super::fixture::query_files::json("mean_a");
    validate_query(&doc_op).unwrap();
    let executed = plan_query_with_tet_mmap(&doc_op, None, &mmap, Some(0)).unwrap();
    let planned_exec = format_query_response(&executed, QueryOutputFormat::Plan).unwrap();
    assert!(!planned_exec.contains("operation_mean"));
    assert!(!planned_exec.contains("\"execution\""));
}

#[test]
fn format_catalog_miss_hint_lists_datasets() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("hint.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"missing"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, None).unwrap();

    let hint = format_query_stderr_hints(&response).expect("catalog miss hint");
    assert!(hint.contains("hint: dataset"));
    assert!(hint.contains("available datasets:"));
    assert!(hint.contains("  a\n"));
    assert!(hint.contains(&format!("tip: run `tet info {}`", path.display())));
}

#[test]
fn format_quiet_validated_without_tet() {
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let response = plan_query_empty(&doc);

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("validated"));
    assert!(line.contains("hint=pass --tet"));
}

#[test]
fn format_table_scalar_mean() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("table.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = super::fixture::query_files::json("mean_a");
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();

    let text = format_query_response(&response, QueryOutputFormat::Table).unwrap();
    assert!(text.contains("query:"));
    assert!(text.contains("mean"));
    assert!(text.contains("3.5"));
    assert!(text.contains("shape"));
    assert!(text.contains("result:"));
    assert!(!text.contains("read_plan:"));
    assert!(!text.contains("chunk_touch_policy"));
    // Compact result is the scalar only — no elements row.
    let result = text.split("result:").nth(1).unwrap();
    assert!(!result.contains("elements"));

    let verbose = format_query_response(&response, QueryOutputFormat::TableVerbose).unwrap();
    assert!(verbose.contains("read_plan:"));
    assert!(verbose.contains("result:"));
    assert!(verbose.contains("touch_policy"));
    assert!(verbose.contains("dense"));
    assert!(!verbose.contains("dense_half_open_intervals"));
}

#[test]
fn format_table_partial_sum() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("table_partial.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = super::fixture::query_files::json("sum_axis0_a");
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();

    let text = format_query_response(&response, QueryOutputFormat::Table).unwrap();
    assert!(text.contains("reduced_shape=3"));
    assert!(text.contains('5'));
    assert!(text.contains('7'));
    assert!(text.contains('0'));
    assert!(text.contains('1'));
}

#[test]
fn format_table_slice_grid_2x3() {
    let path = fixture::tracked_small_tet_dir().join("sample.tet");
    let doc = super::fixture::query_files::json("slice_full_temperature");
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(6)).unwrap();

    let text = format_query_response(&response, QueryOutputFormat::Table).unwrap();
    assert!(text.contains("slice (f32, shape=2×3)"));
    assert!(text.contains("c0"));
    assert!(text.contains("r0"));
    assert!(text.contains("1"));
    assert!(text.contains("6"));
}

#[test]
fn format_quiet_preview_execute() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prev.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();

    let line = format_query_response(&response, QueryOutputFormat::Quiet).unwrap();
    assert!(line.contains("status=preview"));
    assert!(line.contains("preview=4 f32"));
    assert!(line.contains("sample=[1,2,3,4]"));
}
