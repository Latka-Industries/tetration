//! Query engine integration tests: JSON validation, mmap planning, materialize, operations.

use std::path::{Path, PathBuf};

use super::fixture::{
    self, CHUNK_2X2, SHAPE_2X3, write_multichunk_2x3_f64_tiles, write_multichunk_2x3_tiles,
    write_multichunk_2x3_zero_zstd,
};
use crate::catalog::{
    CHUNK_PAYLOAD_CODEC_V1, DATASET_DTYPE_TAG_V1, FileExecutionSettingsV1, OneChunkRawWrite,
    RawArrayWrite, read_tet_summary_v1, write_one_chunk_raw_file, write_raw_array_file,
};
use crate::layout::{create_empty_v1_file, mmap_file_read};
use crate::query::{
    CHUNK_TOUCH_POLICY, Operation, OutputHint, QueryInputFormat, QueryLimits, SpillPathAllowlist,
    TempSpillFile, ValueCompareOp, ValuePredicate, detect_query_input_format,
    materialize_read_plan_f32_le, materialize_read_plan_f32_le_into,
    materialize_read_plan_f32_le_into_parallel, materialize_read_plan_f32_le_parallel,
    parse_query_json, parse_query_text, parse_query_toml, plan_query_empty,
    plan_query_with_tet_mmap, plan_query_with_tet_mmap_ex, validate_query,
};

/// JSON string literal for a filesystem path embedded in test query fixtures.
fn json_path_handle(path: &Path) -> String {
    let s = path.display().to_string();
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

// --- JSON / plan-only ---

#[test]
fn sample_query_toml_parses_like_json() {
    let doc = super::fixture::query_files::toml("mean_strided_temperature");
    validate_query(&doc).unwrap();
    assert_eq!(doc.dataset, "temperature");
    assert!(matches!(doc.operation, Some(Operation::Mean { .. })));
    let axes = doc.selection.as_ref().unwrap();
    assert_eq!(axes.len(), 2);
    assert_eq!(axes[0].start, Some(0));
    assert_eq!(axes[0].stop, Some(100));
    assert_eq!(axes[0].step, Some(2));
}

#[test]
fn parse_query_text_auto_detects_json_and_toml() {
    let json = super::fixture::query_files::read("mean_a", "json");
    let toml = super::fixture::query_files::read("mean_a", "toml");
    let json_path = super::fixture::query_files::path("mean_a", "json");
    let toml_path = super::fixture::query_files::path("mean_a", "toml");
    assert_eq!(
        detect_query_input_format(Some(json_path.to_str().unwrap()), &json),
        QueryInputFormat::Json
    );
    assert_eq!(
        detect_query_input_format(Some(toml_path.to_str().unwrap()), &toml),
        QueryInputFormat::Toml
    );
    assert_eq!(
        detect_query_input_format(None, &json),
        QueryInputFormat::Json
    );
    assert_eq!(
        detect_query_input_format(None, &toml),
        QueryInputFormat::Toml
    );
    parse_query_text(&json, QueryInputFormat::Auto).unwrap();
    parse_query_text(&toml, QueryInputFormat::Auto).unwrap();
}

#[test]
fn toml_parametric_op_tables() {
    let doc = super::fixture::query_files::toml("quantile_axis0_a");
    validate_query(&doc).unwrap();
    assert!(matches!(
        doc.operation,
        Some(Operation::Quantile { q, .. }) if (q - 0.5).abs() < f64::EPSILON
    ));
}

#[test]
fn sample_query_parses_and_plans() {
    let doc = super::fixture::query_files::json("mean_strided_temperature");
    validate_query(&doc).unwrap();
    let plan = plan_query_empty(&doc);
    assert!(plan.accepted);
    assert_eq!(plan.dataset, "temperature");
    assert_eq!(plan.selection_axes, Some(2));
}

#[test]
fn rejects_nested_operation_object() {
    let json = r#"{"dataset":"a","operation":{"mean":{"axes":[]}}}"#;
    let err = parse_query_json(json).unwrap_err();
    assert!(err.to_string().contains("operation"), "{err}");
}

#[test]
fn rejects_nested_output_object() {
    let json = r#"{"dataset":"a","output":{"preferred":{"spill_array":{"handle":"out.bin"}}}}"#;
    let err = parse_query_json(json).unwrap_err();
    assert!(err.to_string().contains("output"), "{err}");
}

#[test]
fn parses_flat_spill_roundtrip() {
    let json = r#"{"dataset":"a","spill":"slice.bin"}"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
    let out = doc.output.as_ref().unwrap();
    assert!(matches!(
        out.preferred,
        Some(OutputHint::SpillArray { ref handle }) if handle == "slice.bin"
    ));
    let roundtrip = serde_json::to_string(&doc).unwrap();
    assert!(roundtrip.contains(r#""spill":"slice.bin""#));
    assert!(!roundtrip.contains("output"));
}

#[test]
fn parses_flat_mean_on_axis_zero() {
    let json = r#"{
        "dataset": "temperature",
        "mean": 0,
        "execution": { "memory_budget_percent": 40 }
    }"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
    let op = doc.operation.as_ref().unwrap();
    assert!(matches!(op, Operation::Mean { axes } if axes.as_slice() == ["0"]));
    assert_eq!(
        doc.execution.as_ref().unwrap().memory_budget_percent_bps,
        Some(4000)
    );
    let roundtrip = serde_json::to_string_pretty(&doc).unwrap();
    assert!(roundtrip.contains(r#""mean": 0"#));
    assert!(roundtrip.contains(r#""memory_budget_percent": 40"#));
    assert!(!roundtrip.contains("operation"));
}

#[test]
fn rejects_invalid_operation_axis_token() {
    // Single-letter names like "x" are valid dimension names (Phase 9); reject malformed tokens.
    let json = r#"{"dataset":"a","sum":"9bad"}"#;
    let err = parse_query_json(json).unwrap_err();
    assert!(err.to_string().contains("invalid axis label"), "{err}");
}

#[test]
fn accepts_decimal_operation_axis_indices() {
    let json = r#"{"dataset":"a","sum":0}"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
}

#[test]
fn accepts_min_max_count_operations() {
    for json in [
        r#"{"dataset":"a","min":[]}"#,
        r#"{"dataset":"a","max":1}"#,
        r#"{"dataset":"a","count":[]}"#,
        r#"{"dataset":"a","var":[]}"#,
        r#"{"dataset":"a","std":0}"#,
        r#"{"dataset":"a","product":[]}"#,
        r#"{"dataset":"a","median":[]}"#,
        r#"{"dataset":"a","quantile":{"q":0.5}}"#,
        r#"{"dataset":"a","histogram":{"bins":4}}"#,
    ] {
        let doc = parse_query_json(json).unwrap();
        validate_query(&doc).unwrap();
    }
}

#[test]
fn rejects_empty_dataset() {
    let json = r#"{"dataset": "   "}"#;
    let doc = parse_query_json(json).unwrap();
    assert!(validate_query(&doc).is_err());
}

#[test]
fn rejects_unknown_query_fields() {
    let json = r#"{"dataset":"a","extra":1}"#;
    let err = parse_query_json(json).unwrap_err();
    assert!(err.to_string().contains("unknown"), "{err}");
}

#[test]
fn parses_value_filter_range_and_roundtrips() {
    let json = r#"{"dataset":"temperature","filter":{"ge":0,"lt":40},"mean":[]}"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
    let filter = doc.filter.as_ref().unwrap();
    assert_eq!(filter.predicates.len(), 2);
    assert!(filter.predicates.iter().any(|p| matches!(
        p,
        ValuePredicate::Compare {
            op: ValueCompareOp::Ge,
            value
        } if (*value - 0.0).abs() < f64::EPSILON
    )));
    assert!(filter.predicates.iter().any(|p| matches!(
        p,
        ValuePredicate::Compare {
            op: ValueCompareOp::Lt,
            value
        } if (*value - 40.0).abs() < f64::EPSILON
    )));
    let roundtrip = serde_json::to_string(&doc).unwrap();
    assert!(roundtrip.contains(r#""filter""#));
    assert!(roundtrip.contains(r#""ge":0"#) || roundtrip.contains(r#""ge":0.0"#));
    let again = parse_query_json(&roundtrip).unwrap();
    validate_query(&again).unwrap();
    assert_eq!(again.filter, doc.filter);
}

#[test]
fn parses_value_filter_is_null_toml() {
    let toml = r#"
dataset = "a"
[filter]
is_null = true
"#;
    let doc = parse_query_toml(toml).unwrap();
    validate_query(&doc).unwrap();
    assert_eq!(
        doc.filter.as_ref().unwrap().predicates,
        vec![ValuePredicate::IsNull]
    );
}

#[test]
fn rejects_empty_value_filter() {
    let err = parse_query_json(r#"{"dataset":"a","filter":{}}"#).unwrap_err();
    assert!(err.to_string().contains("filter"), "{err}");
}

#[test]
fn rejects_unknown_value_filter_keys() {
    let err = parse_query_json(r#"{"dataset":"a","filter":{"between":[0,1]}}"#).unwrap_err();
    assert!(
        err.to_string().contains("unknown field `filter.between`"),
        "{err}"
    );
}

#[test]
fn rejects_non_finite_value_filter() {
    let err = parse_query_json(r#"{"dataset":"a","filter":{"gt":"inf"}}"#).unwrap_err();
    assert!(err.to_string().contains("filter.gt"), "{err}");
    let doc = parse_query_json(r#"{"dataset":"a","filter":{"gt":null}}"#);
    assert!(doc.is_err());
}

#[test]
fn rejects_empty_value_filter_range() {
    let doc = parse_query_json(r#"{"dataset":"a","filter":{"gt":10,"lt":5}}"#).unwrap();
    let err = validate_query(&doc).unwrap_err();
    assert!(err.to_string().contains("empty"), "{err}");
}

#[test]
fn rejects_is_null_with_numeric_compare() {
    let doc = parse_query_json(r#"{"dataset":"a","filter":{"is_null":true,"gt":0}}"#).unwrap();
    let err = validate_query(&doc).unwrap_err();
    assert!(err.to_string().contains("is_null"), "{err}");
}

#[test]
fn rejects_sql_style_filter_expression() {
    let err = parse_query_json(r#"{"dataset":"a","filter":"temperature > 100"}"#).unwrap_err();
    assert!(err.to_string().contains("filter"), "{err}");
    let err = parse_query_json(r#"{"dataset":"a","where":{"gt":1}}"#).unwrap_err();
    assert!(err.to_string().contains("unknown"), "{err}");
}

#[test]
fn rejects_oversized_query_json() {
    let limits = QueryLimits::DEFAULT;
    let pad = "x".repeat(limits.max_json_bytes);
    let json = format!(r#"{{"dataset":"{pad}"}}"#);
    let err = parse_query_json(&json).unwrap_err();
    assert!(err.to_string().contains("maximum size"), "{err}");
}

#[test]
fn rejects_oversized_dataset_name() {
    let limits = QueryLimits::DEFAULT;
    let name = "a".repeat(limits.max_dataset_name_len + 1);
    let json = format!(r#"{{"dataset":"{name}"}}"#);
    let doc = parse_query_json(&json).unwrap();
    let err = validate_query(&doc).unwrap_err();
    assert!(err.to_string().contains("dataset"), "{err}");
}

#[test]
fn rejects_selection_rank_above_max_ndim() {
    use crate::catalog::MAX_NDIM;
    let slices = (0..=MAX_NDIM)
        .map(|_| r#"{ "start": 0, "stop": 1 }"#)
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(r#"{{"dataset":"a","selection":[{slices}]}}"#);
    let doc = parse_query_json(&json).unwrap();
    let err = validate_query(&doc).unwrap_err();
    assert!(err.to_string().contains("selection rank"), "{err}");
}

#[test]
fn rejects_deeply_nested_query_json() {
    let limits = QueryLimits::DEFAULT;
    let mut inner = "null".to_string();
    for _ in 0..=limits.max_json_depth {
        inner = format!(r#"{{"x":{inner}}}"#);
    }
    let json = format!(r#"{{"dataset":"a","junk":{inner}}}"#);
    let err = parse_query_json(&json).unwrap_err();
    assert!(err.to_string().contains("nesting depth"), "{err}");
}

// --- strided read plan ---

#[test]
fn read_plan_strided_step_touches_fewer_chunks_than_dense() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("strided.tet");
    let shape = [4u64, 3];
    let chunk_shape = [2u64, 3];
    let mut data = vec![0u8; 4 * 12];
    for (i, slot) in data.chunks_exact_mut(4).enumerate() {
        let v = (i + 1) as f32;
        slot.copy_from_slice(&v.to_le_bytes());
    }
    write_raw_array_file(
        &path,
        &RawArrayWrite {
            name: "a",
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &shape,
            chunk_shape: &chunk_shape,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &data,
            file_execution: None,
        },
    )
    .unwrap();

    let mmap = mmap_file_read(&path).unwrap();

    let doc_dense = parse_query_json(
        r#"{"dataset":"a","selection":[{"start":1,"stop":3},{"start":0,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc_dense).unwrap();
    let dense = plan_query_with_tet_mmap(&doc_dense, None, &mmap, None).unwrap();
    assert_eq!(
        dense.read_plan.as_ref().unwrap().chunk_touch_policy,
        CHUNK_TOUCH_POLICY.dense_half_open_unit_step
    );
    assert_eq!(dense.read_plan.as_ref().unwrap().chunk_count, 2);

    let doc_strided = parse_query_json(
        r#"{"dataset":"a","selection":[{"start":1,"stop":3,"step":2},{"start":0,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc_strided).unwrap();
    let strided = plan_query_with_tet_mmap(&doc_strided, None, &mmap, None).unwrap();
    assert_eq!(
        strided.read_plan.as_ref().unwrap().chunk_touch_policy,
        CHUNK_TOUCH_POLICY.strided_half_open
    );
    assert_eq!(strided.read_plan.as_ref().unwrap().chunk_count, 1);
    assert_eq!(
        strided.read_plan.as_ref().unwrap().chunks[0].chunk_index,
        vec![0, 0]
    );
}

// --- mmap planning, materialize, operations ---

#[test]
fn plan_query_f32_preview_zstd_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zstd_q.tet");
    write_multichunk_2x3_zero_zstd(&path, "temperature");
    let doc = parse_query_json(r#"{"dataset":"temperature","layout_version":1}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(32)).unwrap();
    let ex = plan.execution.as_ref().expect("preview requested");
    let s = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(
        ex.total_bytes_read_from_disk,
        s.chunks[0].stored_byte_len + s.chunks[1].stored_byte_len
    );
    assert!(!ex.f32_preview_truncated);
    assert!(ex.f32_preview.iter().all(|&x| x == 0.0));
}

#[test]
fn plan_query_resolves_dataset_in_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q.tet");
    write_multichunk_2x3_tiles(&path, "temperature");

    let doc = parse_query_json(r#"{"dataset":"temperature","layout_version":1}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, Some("q.tet"), &mmap, None).unwrap();
    assert_eq!(plan.tet_file.as_deref(), Some("q.tet"));
    let cat = plan.catalog.as_ref().unwrap();
    assert!(cat.matched);
    assert_eq!(cat.dataset_index, Some(0));
    assert_eq!(cat.shape.as_ref().unwrap(), &SHAPE_2X3);
    assert_eq!(cat.chunk_shape.as_ref().unwrap(), &CHUNK_2X2);
    assert_eq!(cat.chunk_index_rows, Some(2));
    let rp = plan.read_plan.as_ref().unwrap();
    assert_eq!(rp.chunk_count, 2);
    assert_eq!(
        rp.chunk_touch_policy,
        CHUNK_TOUCH_POLICY.dense_half_open_unit_step
    );
    assert_eq!(rp.total_stored_bytes, 24);
    assert_eq!(rp.chunks[0].chunk_index, vec![0, 0]);
    assert_eq!(rp.chunks[1].chunk_index, vec![0, 1]);
}

#[test]
fn plan_query_unknown_dataset_lists_available() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("one.tet");
    let shape = [1u64, 1];
    let mut payload = vec![0u8; 4];
    payload.copy_from_slice(&1.0f32.to_le_bytes());
    write_one_chunk_raw_file(
        &path,
        &OneChunkRawWrite {
            name: "only_me",
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &shape,
            chunk_shape: &shape,
            payload: &payload,
        },
    )
    .unwrap();

    let doc = parse_query_json(r#"{"dataset":"missing"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let cat = plan.catalog.as_ref().unwrap();
    assert!(!cat.matched);
    assert_eq!(
        cat.available_datasets.as_ref().unwrap(),
        &vec!["only_me".to_string()]
    );
    assert!(plan.read_plan.is_none());
}

#[test]
fn plan_query_empty_file_catalog() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.tet");
    create_empty_v1_file(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"x"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let cat = plan.catalog.as_ref().unwrap();
    assert!(!cat.matched);
    assert_eq!(
        cat.available_datasets.as_ref().unwrap(),
        &Vec::<String>::new()
    );
    assert!(plan.read_plan.is_none());
}

#[test]
fn plan_query_read_plan_respects_narrow_selection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("narrow.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let doc = parse_query_json(
        r#"{"dataset":"a","selection":[{"start":0,"stop":2},{"start":2,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    assert_eq!(rp.chunk_count, 1);
    assert_eq!(rp.chunks[0].chunk_index, vec![0, 1]);
}

#[test]
fn plan_query_selection_wrong_rank_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rank.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let doc = parse_query_json(r#"{"dataset":"a","selection":[{"start":0,"stop":1}]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let err = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("exactly 2 axes"), "{msg}");
}

#[test]
fn plan_query_f32_preview_full_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prev.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(32)).unwrap();
    let ex = plan.execution.as_ref().expect("preview requested");
    assert_eq!(ex.total_bytes_read_from_disk, 24);
    assert!(!ex.f32_preview_truncated);
    let want = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(ex.f32_preview.len(), want.len());
    for (a, b) in ex.f32_preview.iter().zip(want.iter()) {
        assert!((a - b).abs() < 1e-5, "got {a} want {b}");
    }
}

#[test]
fn plan_query_f32_preview_narrow_selection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prev2.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(
        r#"{"dataset":"a","selection":[{"start":0,"stop":2},{"start":2,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(8)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.total_bytes_read_from_disk, 8);
    assert_eq!(ex.f32_preview.len(), 2);
    assert!((ex.f32_preview[0] - 3.0).abs() < 1e-5);
    assert!((ex.f32_preview[1] - 6.0).abs() < 1e-5);
}

#[test]
fn materialize_read_plan_f32_decodes_full_planned_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mat.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (vals, truncated, bytes) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    assert_eq!(bytes, 24);
    assert!(!truncated);
    assert_eq!(vals.len(), 6);
    let want = [1f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    for (a, b) in vals.iter().zip(want.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn materialize_read_plan_f32_cap_truncates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mat2.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (vals, truncated, _) = materialize_read_plan_f32_le(&mmap, rp, Some(4)).unwrap();
    assert!(truncated);
    assert_eq!(vals.len(), 4);
    assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn plan_query_operation_requires_explicit_preview_limit() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("op_req.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    assert!(plan_query_with_tet_mmap(&doc, None, &mmap, None).is_err());
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert!(ex.f32_preview.is_empty());
    assert_eq!(ex.operation_element_count, Some(6));
    assert!((ex.operation_sum.unwrap() - 21.0).abs() < 1e-5);
}

#[test]
fn plan_query_preview_cap_zero_without_operation_errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prev0.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    assert!(plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).is_err());
}

#[test]
fn plan_query_operation_min_max_count_scalar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("minmax.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();

    let min_doc = parse_query_json(r#"{"dataset":"a","min":[]}"#).unwrap();
    validate_query(&min_doc).unwrap();
    let min_plan = plan_query_with_tet_mmap(&min_doc, None, &mmap, Some(8)).unwrap();
    assert!((min_plan.execution.as_ref().unwrap().operation_min.unwrap() - 1.0).abs() < 1e-5);

    let max_doc = parse_query_json(r#"{"dataset":"a","max":[]}"#).unwrap();
    validate_query(&max_doc).unwrap();
    let max_plan = plan_query_with_tet_mmap(&max_doc, None, &mmap, Some(8)).unwrap();
    assert!((max_plan.execution.as_ref().unwrap().operation_max.unwrap() - 6.0).abs() < 1e-5);

    let count_doc = parse_query_json(r#"{"dataset":"a","count":[]}"#).unwrap();
    validate_query(&count_doc).unwrap();
    let count_plan = plan_query_with_tet_mmap(&count_doc, None, &mmap, Some(0)).unwrap();
    let ex = count_plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_element_count, Some(6));
    assert!(ex.f32_preview.is_empty());
}

#[test]
fn plan_query_operation_var_std_scalar_and_partial() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("varstd.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();

    let var_doc = parse_query_json(r#"{"dataset":"a","var":[]}"#).unwrap();
    validate_query(&var_doc).unwrap();
    let var_plan = plan_query_with_tet_mmap(&var_doc, None, &mmap, Some(0)).unwrap();
    let var_ex = var_plan.execution.as_ref().unwrap();
    // population var of 1..6
    assert!((var_ex.operation_var.unwrap() - 2.916_666_7).abs() < 1e-5);

    let std_doc = parse_query_json(r#"{"dataset":"a","std":[]}"#).unwrap();
    validate_query(&std_doc).unwrap();
    let std_plan = plan_query_with_tet_mmap(&std_doc, None, &mmap, Some(0)).unwrap();
    let std_ex = std_plan.execution.as_ref().unwrap();
    assert!((std_ex.operation_std.unwrap() - 1.707_825_1).abs() < 1e-5);

    let partial_doc = parse_query_json(r#"{"dataset":"a","var":0}"#).unwrap();
    validate_query(&partial_doc).unwrap();
    let partial_plan = plan_query_with_tet_mmap(&partial_doc, None, &mmap, Some(4)).unwrap();
    let partial_ex = partial_plan.execution.as_ref().unwrap();
    assert_eq!(
        partial_ex.operation_reduced_shape.as_deref(),
        Some(&[3u64][..])
    );
    let vars = partial_ex.operation_reduced_var.as_ref().unwrap();
    assert_eq!(vars.len(), 3);
    for v in vars {
        assert!((*v - 2.25).abs() < 1e-5, "expected 2.25, got {v}");
    }
}

#[test]
fn plan_query_operation_product_scalar_and_partial() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("product.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();

    let doc = parse_query_json(r#"{"dataset":"a","product":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    assert!((plan.execution.as_ref().unwrap().operation_product.unwrap() - 720.0).abs() < 1e-5);

    let partial_doc = parse_query_json(r#"{"dataset":"a","product":0}"#).unwrap();
    validate_query(&partial_doc).unwrap();
    let partial_plan = plan_query_with_tet_mmap(&partial_doc, None, &mmap, Some(4)).unwrap();
    let ex = partial_plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_reduced_shape.as_deref(), Some(&[3u64][..]));
    let products = ex.operation_reduced_product.as_ref().unwrap();
    assert_eq!(products.len(), 3);
    assert!((products[0] - 4.0).abs() < 1e-5);
    assert!((products[1] - 10.0).abs() < 1e-5);
    assert!((products[2] - 18.0).abs() < 1e-5);
}

#[test]
fn plan_query_operation_min_along_axis_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("min_axis0.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","min":0}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_reduced_shape.as_deref(), Some(&[3u64][..]));
    let mins = ex.operation_reduced_min.as_ref().unwrap();
    assert_eq!(mins.len(), 3);
    assert!((mins[0] - 1.0).abs() < 1e-5);
    assert!((mins[1] - 2.0).abs() < 1e-5);
    assert!((mins[2] - 3.0).abs() < 1e-5);
}

#[test]
fn named_axis_resolves_via_footer_dim_names() {
    use crate::catalog::{DatasetMetadataV1, FooterBlobV1, TetMetadataV1, write_footer_blob};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dim_names.tet");
    write_multichunk_2x3_tiles(&path, "a");
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1 {
                datasets: [(
                    "a".to_owned(),
                    DatasetMetadataV1 {
                        dim_names: Some(vec!["row".to_owned(), "col".to_owned()]),
                        ..Default::default()
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            }),
            metadata_ref: None,
        },
    )
    .unwrap();

    let mmap = mmap_file_read(&path).unwrap();
    let by_name = parse_query_json(r#"{"dataset":"a","min":"row"}"#).unwrap();
    validate_query(&by_name).unwrap();
    let by_idx = parse_query_json(r#"{"dataset":"a","min":0}"#).unwrap();
    let r_name = plan_query_with_tet_mmap(&by_name, None, &mmap, Some(4)).unwrap();
    let r_idx = plan_query_with_tet_mmap(&by_idx, None, &mmap, Some(4)).unwrap();
    let ex_name = r_name.execution.as_ref().unwrap();
    let ex_idx = r_idx.execution.as_ref().unwrap();
    assert_eq!(ex_name.operation_reduced_min, ex_idx.operation_reduced_min);
    assert_eq!(
        ex_name.operation_reduced_shape,
        ex_idx.operation_reduced_shape
    );
}

#[test]
fn named_axis_requires_dim_names_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no_dim_names.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","mean":"row"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let err = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap_err();
    assert!(err.to_string().contains("dim_names"));
}

#[test]
fn group_by_wire_rejects_unknown_and_oversize() {
    let unknown = parse_query_json(r#"{"dataset":"a","group":["row"],"sum":[]}"#).unwrap_err();
    assert!(unknown.to_string().contains("unknown field"));

    let empty = parse_query_json(r#"{"dataset":"a","group_by":[],"sum":[]}"#).unwrap_err();
    assert!(empty.to_string().contains("at least one"));

    let missing_op = parse_query_json(r#"{"dataset":"a","group_by":["0"]}"#).unwrap();
    let err = validate_query(&missing_op).unwrap_err();
    assert!(err.to_string().contains("requires a reduction"));

    let dup = parse_query_json(r#"{"dataset":"a","group_by":["row","row"],"sum":[]}"#).unwrap();
    let err = validate_query(&dup).unwrap_err();
    assert!(err.to_string().contains("more than once"));

    let too_many = parse_query_json(
        r#"{"dataset":"a","group_by":["0","1","2","3","4","5","6","7","8"],"sum":[]}"#,
    )
    .unwrap_err();
    assert!(too_many.to_string().contains("maximum"));
}

#[test]
fn group_by_named_dimension_streams_fold() {
    use crate::catalog::{DatasetMetadataV1, FooterBlobV1, TetMetadataV1, write_footer_blob};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group_by.tet");
    write_multichunk_2x3_tiles(&path, "a");
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1 {
                datasets: [(
                    "a".to_owned(),
                    DatasetMetadataV1 {
                        dim_names: Some(vec!["row".to_owned(), "col".to_owned()]),
                        ..Default::default()
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            }),
            metadata_ref: None,
        },
    )
    .unwrap();

    let doc = parse_query_json(r#"{"dataset":"a","group_by":["col"],"sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let roundtrip = parse_query_json(&serde_json::to_string(&doc).unwrap()).unwrap();
    assert_eq!(
        roundtrip
            .group_by
            .as_ref()
            .map(crate::query::GroupBySpec::axes),
        Some(&["col".to_owned()][..])
    );

    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    let logical = plan.read_plan.as_ref().unwrap().logical_element_count;
    assert_eq!(ex.memory_strategy, Some("streaming_fold"));
    assert_eq!(ex.group_count, Some(3));
    assert!(logical > ex.group_count.unwrap());
    assert_eq!(ex.operation_reduced_shape.as_deref(), Some(&[3u64][..]));
    let sums = ex.operation_reduced_sum.as_ref().unwrap();
    assert_eq!(sums.len(), 3);
    assert!((sums[0] - 5.0).abs() < 1e-5);
    assert!((sums[1] - 7.0).abs() < 1e-5);
    assert!((sums[2] - 9.0).abs() < 1e-5);
    assert!(ex.spill_f32_path.is_none());
}

#[test]
fn group_by_tier_c_materializes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group_by_median.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","group_by":["1"],"median":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, path.to_str(), &mmap, Some(0)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.group_count, Some(3));
    assert_ne!(ex.memory_strategy, Some("streaming_fold"));
    assert!(ex.memory_strategy.unwrap_or("").contains("materialize"));
    assert_eq!(ex.operation_reduced_median.as_ref().map(Vec::len), Some(3));
}

#[test]
fn group_by_labels_collapses_repeated_coords() {
    use crate::catalog::{
        CoordAxisV1, DatasetMetadataV1, FooterBlobV1, TetMetadataV1, write_footer_blob,
    };

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group_by_labels.tet");
    write_multichunk_2x3_tiles(&path, "a");
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1 {
                datasets: [(
                    "a".to_owned(),
                    DatasetMetadataV1 {
                        dim_names: Some(vec!["row".to_owned(), "col".to_owned()]),
                        coords: Some(
                            [(
                                "col".to_owned(),
                                CoordAxisV1 {
                                    labels: vec![
                                        "east".to_owned(),
                                        "east".to_owned(),
                                        "west".to_owned(),
                                    ],
                                },
                            )]
                            .into_iter()
                            .collect(),
                        ),
                        ..Default::default()
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            }),
            metadata_ref: None,
        },
    )
    .unwrap();

    let doc =
        parse_query_json(r#"{"dataset":"a","group_by":{"by":"labels","axes":["col"]},"sum":[]}"#)
            .unwrap();
    validate_query(&doc).unwrap();
    assert!(
        doc.group_by
            .as_ref()
            .is_some_and(crate::query::GroupBySpec::by_labels)
    );
    let roundtrip = parse_query_json(&serde_json::to_string(&doc).unwrap()).unwrap();
    assert_eq!(roundtrip.group_by, doc.group_by);

    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.memory_strategy, Some("streaming_fold"));
    assert_eq!(ex.group_count, Some(2));
    assert_eq!(ex.operation_reduced_shape.as_deref(), Some(&[2u64][..]));
    let sums = ex.operation_reduced_sum.as_ref().unwrap();
    assert_eq!(sums.len(), 2);
    // Index group_by col sums were 5, 7, 9 — east collapses col0+col1.
    assert!((sums[0] - 12.0).abs() < 1e-5);
    assert!((sums[1] - 9.0).abs() < 1e-5);
}

#[test]
fn group_by_labels_missing_coords_fail_closed() {
    use crate::catalog::{DatasetMetadataV1, FooterBlobV1, TetMetadataV1, write_footer_blob};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("group_by_labels_missing.tet");
    write_multichunk_2x3_tiles(&path, "a");
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1 {
                datasets: [(
                    "a".to_owned(),
                    DatasetMetadataV1 {
                        dim_names: Some(vec!["row".to_owned(), "col".to_owned()]),
                        ..Default::default()
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            }),
            metadata_ref: None,
        },
    )
    .unwrap();

    let doc =
        parse_query_json(r#"{"dataset":"a","group_by":{"by":"labels","axes":["col"]},"sum":[]}"#)
            .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let err = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap_err();
    assert!(err.to_string().contains("coordinate labels"));
}

#[test]
fn group_by_labels_wire_rejects_unknown() {
    let unknown = parse_query_json(
        r#"{"dataset":"a","group_by":{"by":"labels","axes":["0"],"how":"dense"},"sum":[]}"#,
    )
    .unwrap_err();
    assert!(unknown.to_string().contains("unknown field `group_by"));

    let bad_by =
        parse_query_json(r#"{"dataset":"a","group_by":{"by":"values","axes":["0"]},"sum":[]}"#)
            .unwrap_err();
    assert!(bad_by.to_string().contains("group_by.by"));
}

#[test]
fn pipeline_lowers_to_group_by_fold() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pipeline.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let raw = r#"{"dataset":"a","pipeline":[{"select":[{},{}]},{"group_by":["1"],"sum":[]}]}"#;
    let doc = parse_query_json(raw).unwrap();
    validate_query(&doc).unwrap();
    assert!(doc.pipeline.as_ref().is_some_and(|steps| steps.len() == 2));
    assert_eq!(
        doc.group_by.as_ref().map(crate::query::GroupBySpec::axes),
        Some(&["1".to_owned()][..])
    );
    let again = parse_query_json(&serde_json::to_string(&doc).unwrap()).unwrap();
    assert_eq!(again.group_by, doc.group_by);
    assert!(again.pipeline.is_some());

    let flat = parse_query_json(r#"{"dataset":"a","group_by":["1"],"sum":[]}"#).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let piped = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let direct = plan_query_with_tet_mmap(&flat, None, &mmap, Some(0)).unwrap();
    assert_eq!(
        piped.execution.as_ref().unwrap().operation_reduced_sum,
        direct.execution.as_ref().unwrap().operation_reduced_sum
    );
    assert_eq!(piped.execution.as_ref().unwrap().group_count, Some(3));
    assert_eq!(
        piped.execution.as_ref().unwrap().memory_strategy,
        Some("streaming_fold")
    );
}

#[test]
fn pipeline_rejects_reorder_unknown_and_oversize() {
    let reorder = parse_query_json(r#"{"dataset":"a","pipeline":[{"sum":[]},{"select":[{},{}]}]}"#)
        .unwrap_err();
    assert!(reorder.to_string().contains("out of order"));

    let unknown = parse_query_json(r#"{"dataset":"a","pipeline":[{"map":1}]}"#).unwrap_err();
    assert!(unknown.to_string().contains("unknown pipeline step"));

    let mixed =
        parse_query_json(r#"{"dataset":"a","sum":[],"pipeline":[{"sum":[]}]}"#).unwrap_err();
    assert!(mixed.to_string().contains("cannot be combined"));

    let steps = (0..9)
        .map(|_| r#"{"select":[]}"#)
        .collect::<Vec<_>>()
        .join(",");
    let over = parse_query_json(&format!(r#"{{"dataset":"a","pipeline":[{steps}]}}"#)).unwrap_err();
    assert!(over.to_string().contains("maximum"));
}

#[test]
fn window_wire_rejects_unknown_and_bad_size() {
    let unknown = parse_query_json(
        r#"{"dataset":"a","window":{"axis":0,"size":2,"mode":"bins","frame":"rows"},"sum":[]}"#,
    )
    .unwrap_err();
    assert!(unknown.to_string().contains("unknown field `window"));

    let zero =
        parse_query_json(r#"{"dataset":"a","window":{"axis":0,"size":0,"mode":"bins"},"sum":[]}"#)
            .unwrap_err();
    assert!(zero.to_string().contains("window.size"));

    let missing =
        parse_query_json(r#"{"dataset":"a","window":{"axis":0,"size":2,"mode":"bins"}}"#).unwrap();
    let err = validate_query(&missing).unwrap_err();
    assert!(err.to_string().contains("requires a reduction"));

    let mixed = parse_query_json(
        r#"{"dataset":"a","group_by":["0"],"window":{"axis":1,"size":2,"mode":"bins"},"sum":[]}"#,
    )
    .unwrap();
    let err = validate_query(&mixed).unwrap_err();
    assert!(err.to_string().contains("cannot be combined"));

    let legacy = parse_query_json(
        r#"{"dataset":"a","window":{"axis":0,"size":2,"mode":"tumbling"},"sum":[]}"#,
    )
    .unwrap_err();
    assert!(legacy.to_string().contains("bins"));
    let legacy_slide = parse_query_json(
        r#"{"dataset":"a","window":{"axis":0,"size":2,"mode":"sliding"},"sum":[]}"#,
    )
    .unwrap_err();
    assert!(legacy_slide.to_string().contains("rolling"));

    let bad_pad = parse_query_json(
        r#"{"dataset":"a","window":{"axis":0,"size":2,"mode":"bins","pad":"edge"},"sum":[]}"#,
    )
    .unwrap_err();
    assert!(bad_pad.to_string().contains("window pad"));

    let bins_step = parse_query_json(
        r#"{"dataset":"a","window":{"axis":0,"size":2,"mode":"bins","step":1},"sum":[]}"#,
    )
    .unwrap_err();
    assert!(bins_step.to_string().contains("window.step"));

    let bad_align = parse_query_json(
        r#"{"dataset":"a","window":{"axis":0,"size":2,"mode":"rolling","align":"middle"},"sum":[]}"#,
    )
    .unwrap_err();
    assert!(bad_align.to_string().contains("window align"));

    let rolling_end_no_pad = parse_query_json(
        r#"{"dataset":"a","window":{"axis":0,"size":2,"mode":"rolling","align":"end"},"sum":[]}"#,
    )
    .unwrap_err();
    assert!(rolling_end_no_pad.to_string().contains("pad"));

    let min_over = parse_query_json(
        r#"{"dataset":"a","window":{"axis":0,"size":2,"mode":"bins","min_count":3},"sum":[]}"#,
    )
    .unwrap_err();
    assert!(min_over.to_string().contains("min_count"));
}

#[test]
fn window_bins_and_rolling_match_reference() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("window.tet");
    write_multichunk_2x3_tiles(&path, "a");
    // Logical [[1,2,3],[4,5,6]] — window along col (axis 1).
    let tumble =
        parse_query_json(r#"{"dataset":"a","window":{"axis":1,"size":2,"mode":"bins"},"sum":[]}"#)
            .unwrap();
    validate_query(&tumble).unwrap();
    let roundtrip = parse_query_json(&serde_json::to_string(&tumble).unwrap()).unwrap();
    assert_eq!(roundtrip.window.as_ref().unwrap().size, 2);

    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&tumble, None, &mmap, Some(0)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.memory_strategy, Some("streaming_fold"));
    assert_eq!(ex.window_count, Some(2));
    assert_eq!(ex.operation_reduced_shape.as_deref(), Some(&[2u64, 2][..]));
    let sums = ex.operation_reduced_sum.as_ref().unwrap();
    assert_eq!(sums.len(), 4);
    // Reference: row0 [1+2, 3], row1 [4+5, 6]
    assert!((sums[0] - 3.0).abs() < 1e-5);
    assert!((sums[1] - 3.0).abs() < 1e-5);
    assert!((sums[2] - 9.0).abs() < 1e-5);
    assert!((sums[3] - 6.0).abs() < 1e-5);

    let slide = parse_query_json(
        r#"{"dataset":"a","window":{"axis":1,"size":2,"mode":"rolling"},"sum":[]}"#,
    )
    .unwrap();
    validate_query(&slide).unwrap();
    let plan = plan_query_with_tet_mmap(&slide, None, &mmap, Some(0)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.window_count, Some(2));
    let sums = ex.operation_reduced_sum.as_ref().unwrap();
    // Reference: row0 [1+2, 2+3], row1 [4+5, 5+6]
    assert!((sums[0] - 3.0).abs() < 1e-5);
    assert!((sums[1] - 5.0).abs() < 1e-5);
    assert!((sums[2] - 9.0).abs() < 1e-5);
    assert!((sums[3] - 11.0).abs() < 1e-5);
}

#[test]
fn window_step_pad_align_min_count_and_u8() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("window_ext.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();

    // Rolling step 2 along cols: starts at 0 only (extent 3, size 2) → one window per row.
    let step = parse_query_json(
        r#"{"dataset":"a","window":{"axis":1,"size":2,"mode":"rolling","step":2},"sum":[]}"#,
    )
    .unwrap();
    validate_query(&step).unwrap();
    let plan = plan_query_with_tet_mmap(&step, None, &mmap, Some(0)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.window_count, Some(1));
    let sums = ex.operation_reduced_sum.as_ref().unwrap();
    assert_eq!(sums.len(), 2);
    assert!((sums[0] - 3.0).abs() < 1e-5); // 1+2
    assert!((sums[1] - 9.0).abs() < 1e-5); // 4+5

    // Bins + pad zero: short last bin padded to size → row0 [1+2, 3+0], row1 [4+5, 6+0].
    let pad = parse_query_json(
        r#"{"dataset":"a","window":{"axis":1,"size":2,"mode":"bins","pad":"0"},"sum":[]}"#,
    )
    .unwrap();
    let plan = plan_query_with_tet_mmap(&pad, None, &mmap, Some(0)).unwrap();
    let sums = plan
        .execution
        .as_ref()
        .unwrap()
        .operation_reduced_sum
        .as_ref()
        .unwrap();
    assert!((sums[0] - 3.0).abs() < 1e-5);
    assert!((sums[1] - 3.0).abs() < 1e-5);
    assert!((sums[2] - 9.0).abs() < 1e-5);
    assert!((sums[3] - 6.0).abs() < 1e-5);

    // min_count 2 on short last bin → NaN for the short windows.
    let mc = parse_query_json(
        r#"{"dataset":"a","window":{"axis":1,"size":2,"mode":"bins","min_count":2},"sum":[]}"#,
    )
    .unwrap();
    let plan = plan_query_with_tet_mmap(&mc, None, &mmap, Some(0)).unwrap();
    let sums = plan
        .execution
        .as_ref()
        .unwrap()
        .operation_reduced_sum
        .as_ref()
        .unwrap();
    assert!((sums[0] - 3.0).abs() < 1e-5);
    assert!(sums[1].is_nan());
    assert!((sums[2] - 9.0).abs() < 1e-5);
    assert!(sums[3].is_nan());

    // Bins align end: short first — row0 [1, 2+3], row1 [4, 5+6].
    let end = parse_query_json(
        r#"{"dataset":"a","window":{"axis":1,"size":2,"mode":"bins","align":"end"},"sum":[]}"#,
    )
    .unwrap();
    let plan = plan_query_with_tet_mmap(&end, None, &mmap, Some(0)).unwrap();
    let sums = plan
        .execution
        .as_ref()
        .unwrap()
        .operation_reduced_sum
        .as_ref()
        .unwrap();
    assert!((sums[0] - 1.0).abs() < 1e-5);
    assert!((sums[1] - 5.0).abs() < 1e-5);
    assert!((sums[2] - 4.0).abs() < 1e-5);
    assert!((sums[3] - 11.0).abs() < 1e-5);

    // u8 dtype windowed sum (same geometry as f32 reference).
    let u8_path = dir.path().join("window_u8.tet");
    fixture::write_multichunk_2x3_u8_tiles(&u8_path, "a");
    let u8_mmap = mmap_file_read(&u8_path).unwrap();
    let u8_doc =
        parse_query_json(r#"{"dataset":"a","window":{"axis":1,"size":2,"mode":"bins"},"sum":[]}"#)
            .unwrap();
    let plan = plan_query_with_tet_mmap(&u8_doc, None, &u8_mmap, Some(0)).unwrap();
    let sums = plan
        .execution
        .as_ref()
        .unwrap()
        .operation_reduced_sum
        .as_ref()
        .unwrap();
    assert!((sums[0] - 3.0).abs() < 1e-5);
    assert!((sums[1] - 3.0).abs() < 1e-5);
    assert!((sums[2] - 9.0).abs() < 1e-5);
    assert!((sums[3] - 6.0).abs() < 1e-5);
}

#[test]
fn align_wire_rejects_unknown_and_combinations() {
    let unknown =
        parse_query_json(r#"{"dataset":"a","align":{"with":"b","axes":["0"],"join":true}}"#)
            .unwrap_err();
    assert!(unknown.to_string().contains("unknown field `align"));

    let empty = parse_query_json(r#"{"dataset":"a","align":{"with":"b","axes":[]}}"#).unwrap_err();
    assert!(empty.to_string().contains("at least one axis"));

    let mixed =
        parse_query_json(r#"{"dataset":"a","align":{"with":"b","axes":["0"]},"sum":[]}"#).unwrap();
    let err = validate_query(&mixed).unwrap_err();
    assert!(err.to_string().contains("cannot be combined"));

    let same = parse_query_json(r#"{"dataset":"a","align":{"with":"a","axes":["0"]}}"#).unwrap();
    let err = validate_query(&same).unwrap_err();
    assert!(err.to_string().contains("different dataset"));
}

#[test]
fn align_check_ok_and_extent_mismatch() {
    use std::collections::BTreeMap;

    use crate::catalog::{CoordAxisV1, TetDatasetWrite, TetWriterSession};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("align.tet");
    let shape = [2_u64, 3];
    let chunk = [2_u64, 2];
    let mut data = vec![0u8; 24];
    for (slot, n) in data.chunks_exact_mut(4).zip(1_u8..=6) {
        slot.copy_from_slice(&f32::from(n).to_le_bytes());
    }
    let mut session = TetWriterSession::create(&path);
    for name in ["a", "b"] {
        let mut ds = TetDatasetWrite::f32_row_major(name, &shape, &chunk, data.clone()).unwrap();
        ds.dim_names = Some(vec!["row".into(), "col".into()]);
        let mut coords = BTreeMap::new();
        coords.insert(
            "row".into(),
            CoordAxisV1 {
                labels: vec!["r0".into(), "r1".into()],
            },
        );
        ds.coords = Some(coords);
        session.push_dataset(ds).unwrap();
    }
    session.commit().unwrap();

    let doc =
        parse_query_json(r#"{"dataset":"a","align":{"with":"b","axes":["row","col"]}}"#).unwrap();
    validate_query(&doc).unwrap();
    let roundtrip = parse_query_json(&serde_json::to_string(&doc).unwrap()).unwrap();
    assert_eq!(roundtrip.align.as_ref().unwrap().with, "b");

    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let align = plan.execution.as_ref().unwrap().alignment.as_ref().unwrap();
    assert!(align.aligned);
    assert_eq!(align.axes.len(), 2);

    // Mismatch: second file with different col extent.
    let path2 = dir.path().join("align_bad.tet");
    let mut session = TetWriterSession::create(&path2);
    let mut a = TetDatasetWrite::f32_row_major("a", &shape, &chunk, data.clone()).unwrap();
    a.dim_names = Some(vec!["row".into(), "col".into()]);
    session.push_dataset(a).unwrap();
    let shape_b = [2_u64, 2];
    let chunk_b = [2_u64, 2];
    let mut data_b = vec![0u8; 16];
    for (slot, n) in data_b.chunks_exact_mut(4).zip(1_u8..=4) {
        slot.copy_from_slice(&f32::from(n).to_le_bytes());
    }
    let mut b = TetDatasetWrite::f32_row_major("b", &shape_b, &chunk_b, data_b).unwrap();
    b.dim_names = Some(vec!["row".into(), "col".into()]);
    session.push_dataset(b).unwrap();
    session.commit().unwrap();

    let mmap = mmap_file_read(&path2).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let align = plan.execution.as_ref().unwrap().alignment.as_ref().unwrap();
    assert!(!align.aligned);
    assert!(align.axes.iter().any(|ax| ax.axis == "col"
        && !ax.ok
        && ax.reason.as_deref().is_some_and(|r| r.contains("extent"))));
}

#[test]
fn pipeline_window_lowers_and_matches_flat() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pipeline_window.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let raw =
        r#"{"dataset":"a","pipeline":[{"window":{"axis":1,"size":2,"mode":"bins"},"sum":[]}]}"#;
    let doc = parse_query_json(raw).unwrap();
    validate_query(&doc).unwrap();
    assert!(doc.window.is_some());
    let flat =
        parse_query_json(r#"{"dataset":"a","window":{"axis":1,"size":2,"mode":"bins"},"sum":[]}"#)
            .unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let piped = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let direct = plan_query_with_tet_mmap(&flat, None, &mmap, Some(0)).unwrap();
    assert_eq!(
        piped.execution.as_ref().unwrap().operation_reduced_sum,
        direct.execution.as_ref().unwrap().operation_reduced_sum
    );
}

#[test]
fn pipeline_cli_quiet_group_sum() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pipeline_cli.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mut cmd = std::process::Command::new(env!("CARGO"));
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    cmd.args([
        "run",
        "--quiet",
        "--bin",
        "tet",
        "--",
        "query",
        r#"{"dataset":"a","pipeline":[{"group_by":["1"],"sum":[]}]}"#,
        "-t",
        path.to_str().unwrap(),
        "-x",
        "-q",
    ]);
    let output = cmd.output().unwrap();
    assert!(
        output.status.success(),
        "stderr {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("group_count=3") && stdout.contains("values="),
        "stdout {stdout}"
    );
}

#[test]
fn plan_query_scalar_sum_fold_matches_materialize_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fold_sum.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(8)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (full, _, _) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    let manual_sum: f64 = full.iter().map(|&x| f64::from(x)).sum();
    assert_eq!(ex.operation_element_count, Some(6));
    assert!((ex.operation_sum.unwrap() - manual_sum).abs() < 1e-5);
    assert_eq!(ex.f32_preview.len(), 6);
    assert!(!ex.f32_preview_truncated);
}

#[test]
fn plan_query_operation_sum_full_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("op_sum.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(64)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_element_count, Some(6));
    assert!((ex.operation_sum.unwrap() - 21.0).abs() < 1e-5);
    assert!(ex.operation_mean.is_none());
    assert!(!ex.f32_preview_truncated);
}

#[test]
fn plan_query_operation_mean_narrow_selection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("op_mean.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(
        r#"{"dataset":"a","mean":[],"selection":[{"start":0,"stop":2},{"start":2,"stop":3}]}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(8)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_element_count, Some(2));
    assert!((ex.operation_mean.unwrap() - 4.5).abs() < 1e-5);
    assert!(ex.operation_sum.is_none());
}

#[test]
fn plan_query_operation_sum_preview_truncated_but_aggregate_full() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("op_trunc.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(2)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert!(ex.f32_preview_truncated);
    assert_eq!(ex.f32_preview.len(), 2);
    assert_eq!(ex.operation_element_count, Some(6));
    assert!((ex.operation_sum.unwrap() - 21.0).abs() < 1e-5);
}

#[test]
fn plan_query_operation_sum_along_axis_zero() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("op_axis0.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","sum":0}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.operation_reduced_shape.as_deref(), Some(&[3u64][..]));
    let sums = ex.operation_reduced_sum.as_ref().unwrap();
    assert_eq!(sums.len(), 3);
    assert!((sums[0] - 5.0).abs() < 1e-5);
    assert!((sums[1] - 7.0).abs() < 1e-5);
    assert!((sums[2] - 9.0).abs() < 1e-5);
    assert!(ex.operation_sum.is_none());
}

#[test]
fn plan_query_execute_multi_chunk_matches_parallel_materialize() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("exec_par.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, Some(64)).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    assert!(rp.chunk_count > 1, "fixture must touch multiple chunks");
    let ex = plan.execution.as_ref().unwrap();
    let (par, par_trunc, par_bytes) =
        materialize_read_plan_f32_le_parallel(&mmap, rp, Some(64)).unwrap();
    assert_eq!(ex.f32_preview, par);
    assert_eq!(ex.f32_preview_truncated, par_trunc);
    assert_eq!(ex.total_bytes_read_from_disk, par_bytes);
}

#[test]
fn materialize_read_plan_f32_parallel_matches_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("par.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (seq, seq_trunc, seq_bytes) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    let (par, par_trunc, par_bytes) =
        materialize_read_plan_f32_le_parallel(&mmap, rp, None).unwrap();
    assert_eq!(seq, par);
    assert_eq!(seq_trunc, par_trunc);
    assert_eq!(seq_bytes, par_bytes);

    let mut seq_buf = vec![0.0f32; 6];
    let mut par_buf = vec![0.0f32; 6];
    let seq_into = materialize_read_plan_f32_le_into(&mmap, rp, None, &mut seq_buf).unwrap();
    let par_into =
        materialize_read_plan_f32_le_into_parallel(&mmap, rp, None, &mut par_buf).unwrap();
    assert_eq!(
        seq_into.logical_element_count,
        par_into.logical_element_count
    );
    assert_eq!(seq_into.elements_written, par_into.elements_written);
    assert_eq!(seq_into.truncated, par_into.truncated);
    assert_eq!(
        seq_into.total_bytes_read_from_disk,
        par_into.total_bytes_read_from_disk
    );
    assert_eq!(seq_buf, par_buf);
}

#[test]
fn materialize_read_plan_f32_parallel_zstd_matches_sequential() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("par_zstd.tet");
    write_multichunk_2x3_zero_zstd(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (seq, _, seq_bytes) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    let (par, _, par_bytes) = materialize_read_plan_f32_le_parallel(&mmap, rp, None).unwrap();
    assert_eq!(seq, par);
    assert_eq!(seq_bytes, par_bytes);
    assert!(seq.iter().all(|&x| x == 0.0));
}

#[test]
fn materialize_read_plan_f32_le_into_matches_vec() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("into.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let plan = plan_query_with_tet_mmap(&doc, None, &mmap, None).unwrap();
    let rp = plan.read_plan.as_ref().unwrap();
    let (want, _, _) = materialize_read_plan_f32_le(&mmap, rp, None).unwrap();
    let mut buf = vec![0.0f32; 6];
    let out = materialize_read_plan_f32_le_into(&mmap, rp, None, &mut buf).unwrap();
    assert_eq!(out.logical_element_count, 6);
    assert_eq!(out.elements_written, 6);
    assert!(!out.truncated);
    assert_eq!(buf, want);
}

// --- JSON hardening property tests ---

use proptest::prelude::*;

// --- file execution settings / memory budget ---

#[test]
fn file_execution_settings_roundtrip_in_index_header() {
    use crate::catalog::{FileExecutionSettingsV1, read_tet_summary_v1};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("exec.tet");
    let settings = FileExecutionSettingsV1 {
        memory_budget_percent_bps: 5000,
        memory_budget_bytes: 64 * 1024 * 1024,
    };
    fixture::write_multichunk_2x3_with_execution(&path, "t", settings);
    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.file_execution, settings);
}

#[test]
fn query_execution_reports_dataset_and_budget_fields() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "temperature");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"temperature","execution":{"memory_budget_bytes":999999}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(4)).unwrap();
    let catalog = resp.catalog.as_ref().unwrap();
    assert_eq!(catalog.dataset_f32_bytes, Some(24));
    assert_eq!(
        catalog.file_execution,
        Some(FileExecutionSettingsV1::default_engine())
    );
    let exec = resp.execution.as_ref().unwrap();
    assert_eq!(exec.memory_budget_bytes, Some(999_999));
    assert_eq!(exec.logical_selection_f32_bytes, Some(24));
}

#[test]
fn plan_query_operation_argmin_argmax_scalar_and_partial() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a","arg_min":[]}"#).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(64)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.operation_argmin_index, Some(0));
    assert_eq!(ex.memory_strategy, Some("streaming_fold"));

    let doc = parse_query_json(r#"{"dataset":"a","arg_max":0}"#).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(64)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.operation_reduced_argmax, Some(vec![1, 1, 1]));
}

#[test]
fn spill_path_allowlist_rejects_outside_root() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let outside_dir = tempfile::tempdir().unwrap();
    let policy = SpillPathAllowlist::default_for_tet(&path, std::iter::empty::<PathBuf>()).unwrap();
    let spill_out = outside_dir.path().join("out.bin");
    let json = format!(
        r#"{{"dataset":"a","spill":{}}}"#,
        json_path_handle(&spill_out)
    );
    let doc = parse_query_json(&json).unwrap();
    let err = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(2), Some(&policy)).unwrap_err();
    assert!(err.to_string().contains("allowed root"), "{err}");

    let json = r#"{"dataset":"a","spill":"out.bin"}"#;
    let doc = parse_query_json(json).unwrap();
    let resp = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(2), Some(&policy)).unwrap();
    let resp_zero_preview =
        plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(0), Some(&policy)).unwrap();
    assert!(
        resp_zero_preview
            .execution
            .as_ref()
            .unwrap()
            .spill_f32_path
            .is_some()
    );
    assert!(
        resp.execution
            .as_ref()
            .unwrap()
            .spill_f32_path
            .as_ref()
            .unwrap()
            .ends_with("out.bin")
    );

    let allow = dir.path().join("allowed");
    std::fs::create_dir_all(&allow).unwrap();
    let policy = SpillPathAllowlist::default_for_tet(&path, [allow.clone()]).unwrap();
    let spill_ok = allow.join("out.bin");
    let json = format!(
        r#"{{"dataset":"a","spill":{}}}"#,
        json_path_handle(&spill_ok)
    );
    let doc = parse_query_json(&json).unwrap();
    let resp = plan_query_with_tet_mmap_ex(&doc, None, &mmap, Some(2), Some(&policy)).unwrap();
    let spilled = resp
        .execution
        .as_ref()
        .unwrap()
        .spill_f32_path
        .as_ref()
        .unwrap();
    assert!(
        spilled.ends_with("allowed/out.bin") || spilled.ends_with("allowed\\out.bin"),
        "{spilled}"
    );
}

#[test]
fn capped_preview_materialize_allocates_only_cap_not_full_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(2)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.f32_preview.len(), 2);
    assert!(ex.f32_preview_truncated);
    assert_eq!(ex.memory_strategy, Some("capped_in_memory"));
}

#[test]
fn plan_query_operation_median_scalar_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","median":[],"execution":{"memory_budget_bytes":999999}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(64)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert!((ex.operation_median.unwrap() - 3.5).abs() < 1e-5);
    assert_eq!(ex.operation_element_count, Some(6));
    assert_eq!(ex.memory_strategy, Some("in_memory_materialize"));
}

#[test]
fn plan_query_operation_median_scalar_temp_spill_when_over_budget() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    // 8 bytes = 2 f32; logical selection is 6 f32 (24 bytes).
    let json = r#"{"dataset":"a","median":[],"execution":{"memory_budget_bytes":8}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(2)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert!((ex.operation_median.unwrap() - 3.5).abs() < 1e-5);
    assert_eq!(ex.memory_strategy, Some("temp_spill_materialize"));
    assert!(ex.spill_f32_path.is_none());
}

#[test]
fn plan_query_u16_preview_and_sum() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u16.tet");
    fixture::write_multichunk_2x3_u16_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.u16_preview.len(), 4);
    assert!(ex.u16_preview_truncated);
    assert_eq!(ex.operation_sum.unwrap(), 21.0);
    assert_eq!(resp.catalog.as_ref().unwrap().dataset_u16_bytes, Some(12));
}

#[test]
fn plan_query_i16_preview_and_sum() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i16.tet");
    fixture::write_multichunk_2x3_i16_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a","sum":[]}"#).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.i16_preview.len(), 4);
    assert!(ex.i16_preview_truncated);
    assert_eq!(ex.operation_sum.unwrap(), 21.0);
    assert_eq!(resp.catalog.as_ref().unwrap().dataset_i16_bytes, Some(12));
}

#[test]
fn plan_query_u8_preview_and_sum() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u8.tet");
    fixture::write_multichunk_2x3_u8_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","sum":[]}"#;
    let doc = parse_query_json(json).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.u8_preview.len(), 4);
    assert!(ex.u8_preview_truncated);
    assert_eq!(ex.f32_preview.len(), 0);
    assert_eq!(ex.operation_sum.unwrap(), 21.0);
    assert_eq!(resp.catalog.as_ref().unwrap().dataset_u8_bytes, Some(6));
}

#[test]
fn plan_query_i32_preview_and_sum() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i32.tet");
    fixture::write_multichunk_2x3_i32_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","sum":[]}"#;
    let doc = parse_query_json(json).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.i32_preview.len(), 4);
    assert!(ex.i32_preview_truncated);
    assert_eq!(ex.f32_preview.len(), 0);
    assert_eq!(ex.operation_sum.unwrap(), 21.0);
    assert_eq!(resp.catalog.as_ref().unwrap().dataset_i32_bytes, Some(24));
}

#[test]
fn plan_query_i64_preview_materialize() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("i64.tet");
    fixture::write_multichunk_2x3_i64_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(6)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.i64_preview, vec![1, 2, 3, 4, 5, 6]);
    assert!(!ex.i64_preview_truncated);
    assert_eq!(resp.catalog.as_ref().unwrap().dataset_i64_bytes, Some(48));
}

#[test]
fn plan_query_f64_preview_and_sum() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64.tet");
    write_multichunk_2x3_f64_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","sum":[]}"#;
    let doc = parse_query_json(json).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(4)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.f64_preview.len(), 4);
    assert!(ex.f64_preview_truncated);
    assert_eq!(ex.f32_preview.len(), 0);
    assert!((ex.operation_sum.unwrap() - 21.0).abs() < 1e-9);
    assert_eq!(resp.catalog.as_ref().unwrap().dataset_f64_bytes, Some(48));
}

#[test]
fn plan_query_operation_median_partial_axis_0() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","median":0,"execution":{"memory_budget_bytes":999999}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(0)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.operation_reduced_shape, Some(vec![3]));
    let medians = ex.operation_reduced_median.as_ref().unwrap();
    assert_eq!(medians.len(), 3);
    assert!((medians[0] - 2.5).abs() < 1e-5);
    assert!((medians[1] - 3.5).abs() < 1e-5);
    assert!((medians[2] - 4.5).abs() < 1e-5);
}

#[test]
fn plan_query_operation_quantile_scalar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json =
        r#"{"dataset":"a","quantile":{"q":0.95},"execution":{"memory_budget_bytes":999999}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(0)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert!((ex.operation_quantile.unwrap() - 5.75).abs() < 1e-5);
}

#[test]
fn histogram_scalar_respects_min_max_edges() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("hist.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","histogram":{"bins":2,"min":0,"max":10},"execution":{"memory_budget_bytes":999999}}"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(0)).unwrap();
    let edges = resp
        .execution
        .as_ref()
        .unwrap()
        .operation_histogram_edges
        .as_ref()
        .unwrap();
    assert_eq!(edges.len(), 3);
    assert!((edges[0] - 0.0).abs() < 1e-9);
    assert!((edges[2] - 10.0).abs() < 1e-9);
}

#[test]
fn nan_count_scalar_on_f32_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nan.tet");
    let data = fixture::le_row_major_2x3_f32_one_to_six();
    let mut patched = data.clone();
    patched[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
    write_raw_array_file(
        &path,
        &RawArrayWrite {
            name: "a",
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            data: &patched,
            dtype: DATASET_DTYPE_TAG_V1.f32,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            file_execution: None,
        },
    )
    .unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a","nan_count":[]}"#).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    assert_eq!(
        resp.execution.as_ref().unwrap().operation_nan_count,
        Some(1.0)
    );
}

#[test]
fn inf_count_scalar_on_f32_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("inf.tet");
    let mut patched = fixture::le_row_major_2x3_f32_one_to_six();
    patched[0..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
    write_raw_array_file(
        &path,
        &RawArrayWrite {
            name: "a",
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            data: &patched,
            dtype: DATASET_DTYPE_TAG_V1.f32,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            file_execution: None,
        },
    )
    .unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a","inf_count":[]}"#).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    assert_eq!(
        resp.execution.as_ref().unwrap().operation_inf_count,
        Some(1.0)
    );
}

#[test]
fn any_inf_scalar_on_f32_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("inf.tet");
    let mut patched = fixture::le_row_major_2x3_f32_one_to_six();
    patched[0..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
    write_raw_array_file(
        &path,
        &RawArrayWrite {
            name: "a",
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            data: &patched,
            dtype: DATASET_DTYPE_TAG_V1.f32,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            file_execution: None,
        },
    )
    .unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a","any_inf":[]}"#).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    assert_eq!(
        resp.execution.as_ref().unwrap().operation_any_inf,
        Some(true)
    );
}

#[test]
fn null_count_uses_footer_fill_value() {
    use crate::catalog::{DatasetMetadataV1, FooterBlobV1, TetMetadataV1, write_footer_blob};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fill.tet");
    let mut data = fixture::le_row_major_2x3_u8_one_to_six();
    data[0] = 99;
    write_raw_array_file(
        &path,
        &RawArrayWrite {
            name: "a",
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            data: &data,
            dtype: DATASET_DTYPE_TAG_V1.u8,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            file_execution: None,
        },
    )
    .unwrap();
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1 {
                datasets: [(
                    "a".to_owned(),
                    DatasetMetadataV1 {
                        attrs: [("_FillValue".to_owned(), "99".to_owned())].into(),
                        ..Default::default()
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            }),
            metadata_ref: None,
        },
    )
    .unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a","null_count":[]}"#).unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    assert_eq!(
        resp.execution.as_ref().unwrap().operation_null_count,
        Some(1.0)
    );
}

#[test]
fn selection_coord_labels_resolve_to_row_slice() {
    use crate::catalog::{
        CoordAxisV1, DatasetMetadataV1, FooterBlobV1, TetMetadataV1, write_footer_blob,
    };

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("coords.tet");
    write_raw_array_file(
        &path,
        &RawArrayWrite {
            name: "a",
            shape: &SHAPE_2X3,
            chunk_shape: &CHUNK_2X2,
            data: &fixture::le_row_major_2x3_f32_one_to_six(),
            dtype: DATASET_DTYPE_TAG_V1.f32,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            file_execution: None,
        },
    )
    .unwrap();
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1 {
                datasets: [(
                    "a".to_owned(),
                    DatasetMetadataV1 {
                        dim_names: Some(vec!["row".into(), "col".into()]),
                        coords: Some({
                            let mut m = std::collections::BTreeMap::new();
                            m.insert(
                                "row".into(),
                                CoordAxisV1 {
                                    labels: vec!["r0".into(), "r1".into()],
                                },
                            );
                            m
                        }),
                        ..Default::default()
                    },
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            }),
            metadata_ref: None,
        },
    )
    .unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(
        r#"{
            "dataset":"a",
            "selection":[
                {"start_label":"r0","stop_label":"r1"},
                {"start":0,"stop":3}
            ],
            "mean":[]
        }"#,
    )
    .unwrap();
    let resp = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    assert_eq!(resp.execution.as_ref().unwrap().operation_mean, Some(2.0));
}

#[test]
fn covariance_on_2x3_selection_obs_axis_1() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cov.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(
        r#"{"dataset":"a","covariance":1,"execution":{"memory_budget_bytes":999999}}"#,
    )
    .unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(0)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.operation_covariance_order, Some(2));
    let cov = ex.operation_covariance.as_ref().unwrap();
    assert_eq!(cov.len(), 4);
    assert_eq!(ex.operation_element_count, Some(6));
    let expected = crate::query::materialize::covariance::run_covariance_correlation(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        1,
        false,
    )
    .unwrap()
    .covariance
    .unwrap();
    for (a, b) in cov.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5, "got {a} expected {b}");
    }
}

#[test]
fn correlation_on_2x3_selection() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("corr.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let doc = parse_query_json(
        r#"{"dataset":"a","correlation":{"axis":1},"execution":{"memory_budget_bytes":999999}}"#,
    )
    .unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(0)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert_eq!(ex.operation_correlation_order, Some(2));
    let corr = ex.operation_correlation.as_ref().unwrap();
    assert!((corr[0] - 1.0).abs() < 1e-9);
    assert!((corr[3] - 1.0).abs() < 1e-9);
}

#[test]
fn plan_query_operation_histogram_scalar() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json =
        r#"{"dataset":"a","histogram":{"bins":3},"execution":{"memory_budget_bytes":999999}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(0)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    let counts = ex.operation_histogram_counts.as_ref().unwrap();
    let edges = ex.operation_histogram_edges.as_ref().unwrap();
    assert_eq!(counts.len(), 3);
    assert_eq!(edges.len(), 4);
    assert_eq!(counts.iter().sum::<f64>() as u32, 6);
}

#[test]
fn plan_query_f64_median_temp_spill_when_over_budget() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f64.tet");
    write_multichunk_2x3_f64_tiles(&path, "a");
    let mmap = mmap_file_read(&path).unwrap();
    let json = r#"{"dataset":"a","median":[],"execution":{"memory_budget_bytes":16}}"#;
    let doc = parse_query_json(json).unwrap();
    let resp =
        plan_query_with_tet_mmap(&doc, Some(path.to_str().unwrap()), &mmap, Some(2)).unwrap();
    let ex = resp.execution.as_ref().unwrap();
    assert!((ex.operation_median.unwrap() - 3.5).abs() < 1e-5);
    assert_eq!(ex.memory_strategy, Some("temp_spill_materialize"));
}

// --- spill policy (from src/query/engine/spill_policy.rs) ---

#[test]
fn spill_path_allowlist_default_for_tet_includes_parent() {
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("data.tet");
    std::fs::write(&tet, b"x").unwrap();
    let policy = SpillPathAllowlist::default_for_tet(&tet, std::iter::empty::<PathBuf>()).unwrap();
    let resolved = policy.validate(Path::new("spill.bin")).unwrap();
    assert!(resolved.starts_with(std::fs::canonicalize(dir.path()).unwrap()));
}

#[test]
fn temp_spill_file_deleted_on_drop() {
    let dir = tempfile::tempdir().unwrap();
    let file_path = dir.path().join("spill-test.bin");
    std::fs::write(&file_path, b"x").unwrap();
    {
        let guard = TempSpillFile::with_path_for_test(file_path.clone());
        assert!(file_path.exists());
        drop(guard);
    }
    assert!(!file_path.exists());
}

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
fn parses_transform_zscore_and_write_roundtrip() {
    let json = r#"{"dataset":"a","transform":{"method":"zscore"},"write":"ram"}"#;
    let doc = parse_query_json(json).unwrap();
    validate_query(&doc).unwrap();
    assert!(matches!(
        doc.operation,
        Some(Operation::Transform {
            method: crate::query::TransformMethod::Zscore,
            ..
        })
    ));
    assert_eq!(
        doc.write.as_ref().unwrap().target,
        crate::query::WriteTarget::Ram
    );
    let roundtrip = serde_json::to_string(&doc).unwrap();
    assert!(roundtrip.contains(r#""transform""#));
    assert!(roundtrip.contains(r#""method":"zscore""#));
    assert!(roundtrip.contains(r#""write":"ram""#));
}

#[test]
fn rejects_spill_and_write_together() {
    let json = r#"{"dataset":"a","transform":{"method":"zscore"},"spill":"out.bin","write":"ram"}"#;
    let doc = parse_query_json(json).unwrap();
    let err = validate_query(&doc).unwrap_err();
    assert!(err.to_string().contains("spill"), "{err}");
}

#[test]
fn plan_query_zscore_scalar_ram() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zscore.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","transform":{"method":"zscore"},"write":"ram"}"#)
        .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let policy =
        SpillPathAllowlist::from_roots(dir.path().to_path_buf(), [dir.path().to_path_buf()]);
    let plan = plan_query_with_tet_mmap_ex(
        &doc,
        Some(path.to_str().unwrap()),
        &mmap,
        Some(6),
        Some(&policy),
    )
    .unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.memory_strategy, Some("transform_ram"));
    assert!((ex.operation_mean.unwrap() - 3.5).abs() < 1e-5);
    let std = pop_std_1_to_6();
    assert!((ex.operation_std.unwrap() - std).abs() < 1e-5);
    assert_eq!(ex.f32_preview.len(), 6);
    for (i, &v) in ex.f32_preview.iter().enumerate() {
        let raw = f64::from((i + 1) as f32);
        let expected = ((raw - 3.5) / std) as f32;
        assert!(
            (v - expected).abs() < 1e-4,
            "index {i}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn plan_query_minmax_scalar_ram() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("minmax.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(r#"{"dataset":"a","transform":{"method":"minmax"},"write":"ram"}"#)
        .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let policy =
        SpillPathAllowlist::from_roots(dir.path().to_path_buf(), [dir.path().to_path_buf()]);
    let plan = plan_query_with_tet_mmap_ex(
        &doc,
        Some(path.to_str().unwrap()),
        &mmap,
        Some(6),
        Some(&policy),
    )
    .unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.memory_strategy, Some("transform_ram"));
    assert_eq!(ex.operation_min, Some(1.0));
    assert_eq!(ex.operation_max, Some(6.0));
    assert!((ex.f32_preview[0] - 0.0).abs() < 1e-5);
    assert!((ex.f32_preview[5] - 1.0).abs() < 1e-5);
    assert!((ex.f32_preview[3] - 0.6).abs() < 1e-5); // value 4 → (4-1)/5
}

#[test]
fn plan_query_zscore_switch_spills_when_over_budget() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("zscore_spill.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(
        r#"{"dataset":"a","transform":{"method":"zscore"},"write":"switch","execution":{"memory_budget_bytes":1}}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let policy =
        SpillPathAllowlist::from_roots(dir.path().to_path_buf(), [dir.path().to_path_buf()]);
    let plan = plan_query_with_tet_mmap_ex(
        &doc,
        Some(path.to_str().unwrap()),
        &mmap,
        Some(2),
        Some(&policy),
    )
    .unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.memory_strategy, Some("transform_spill"));
    assert!(ex.spill_f32_path.is_some());
    assert_eq!(ex.spill_f32_bytes, Some(24));
}

#[test]
fn plan_query_zscore_sidecar_publishes_tet_beside_source() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let doc = parse_query_json(
        r#"{"dataset":"a","transform":{"method":"zscore"},"write":{"target":"sidecar","timestamp":false}}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let policy =
        SpillPathAllowlist::from_roots(dir.path().to_path_buf(), [dir.path().to_path_buf()]);
    let plan = plan_query_with_tet_mmap_ex(
        &doc,
        Some(path.to_str().unwrap()),
        &mmap,
        Some(6),
        Some(&policy),
    )
    .unwrap();
    let ex = plan.execution.as_ref().unwrap();
    assert_eq!(ex.memory_strategy, Some("transform_sidecar"));
    let published = ex.spill_f32_path.as_ref().unwrap();
    assert!(
        published.ends_with("data.zscore.tet"),
        "unexpected sidecar path: {published}"
    );
    let published_path = Path::new(published);
    assert!(published_path.is_file());
    let sidecar_mmap = mmap_file_read(published_path).unwrap();
    let summary = read_tet_summary_v1(&sidecar_mmap).unwrap();
    assert_eq!(summary.datasets.len(), 1);
    assert_eq!(summary.datasets[0].name, "a-zscore");
    assert_eq!(summary.history.len(), 1);
    assert_eq!(summary.history[0].op, "transform");
    assert_eq!(
        summary.history[0].params.get("method").map(String::as_str),
        Some("zscore")
    );
    assert_eq!(
        summary.history[0].params.get("dataset").map(String::as_str),
        Some("a")
    );
    assert_eq!(
        summary.history[0]
            .params
            .get("output_dataset")
            .map(String::as_str),
        Some("a-zscore")
    );
    assert_eq!(
        summary
            .metadata
            .file
            .as_ref()
            .and_then(|f| f.tool.as_deref()),
        Some("tet query transform")
    );
    // Same payload + naming through TetWriterSession must match catalog shape.
    let ds = &summary.datasets[0];
    let mut session = crate::catalog::TetWriterSession::create(dir.path().join("writer.tet"));
    session.metadata.tool = Some("tet query transform".to_owned());
    let payload_elems = ds.shape.iter().product::<u64>() as usize;
    let payload = vec![0u8; payload_elems * 4];
    let write = crate::catalog::TetDatasetWrite::f32_row_major(
        "a-zscore",
        &ds.shape,
        &ds.chunk_shape,
        payload,
    )
    .unwrap();
    session.push_dataset(write).unwrap();
    session.commit().unwrap();
    let writer_summary =
        read_tet_summary_v1(&mmap_file_read(&dir.path().join("writer.tet")).unwrap()).unwrap();
    assert_eq!(writer_summary.datasets[0].name, ds.name);
    assert_eq!(writer_summary.datasets[0].shape, ds.shape);
    assert_eq!(writer_summary.datasets[0].chunk_shape, ds.chunk_shape);
    assert_eq!(writer_summary.datasets[0].dtype, ds.dtype);
}

#[test]
fn rejects_write_timestamp_for_non_sidecar() {
    let doc = parse_query_json(
        r#"{"dataset":"a","transform":{"method":"zscore"},"write":{"target":"spill","timestamp":false}}"#,
    )
    .unwrap();
    let err = validate_query(&doc).unwrap_err();
    assert!(err.to_string().contains("timestamp"), "{err}");
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        ..ProptestConfig::default()
    })]

    #[test]
    fn parse_query_json_never_panics(raw in "\\PC{0,4096}") {
        let _ = parse_query_json(&raw);
    }

    #[test]
    fn validate_query_never_panics_after_parse(raw in "\\PC{0,4096}") {
        if let Ok(doc) = parse_query_json(&raw) {
            let _ = validate_query(&doc);
        }
    }
}
