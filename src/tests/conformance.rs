//! SPEC conformance pack gate (THI-592).
//!
//! Drives [`conformance/manifest.v1.json`](../../../conformance/manifest.v1.json) so
//! “compatible reader” is CI-testable. Pack format: [`conformance/README.md`](../../../conformance/README.md).

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;

use crate::layout::mmap_file_read;
use crate::query::{parse_query_json, plan_query_with_tet_mmap, validate_query};
use crate::verify::{VerifyOptions, verify_tet_bytes, verify_tet_file};

#[derive(Debug, Deserialize)]
struct Manifest {
    pack_version: u32,
    accept: Vec<AcceptCase>,
    reject_verify: Vec<RejectVerifyCase>,
    reject_query: Vec<RejectQueryCase>,
}

#[derive(Debug, Deserialize)]
struct AcceptCase {
    id: String,
    tet: String,
    #[serde(default)]
    verify: Option<String>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    expect: Option<serde_json::Map<String, Value>>,
    #[serde(default)]
    expect_tol: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RejectVerifyCase {
    id: String,
    tet: String,
    #[serde(default)]
    expect_check: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RejectQueryCase {
    id: String,
    query: String,
    expect_error_substr: String,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_manifest() -> Manifest {
    let path = repo_root().join("conformance/manifest.v1.json");
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("missing conformance manifest {}: {e}", path.display());
    });
    let m: Manifest = serde_json::from_str(&text).expect("conformance manifest JSON");
    assert_eq!(m.pack_version, 1, "unsupported pack_version");
    m
}

fn resolve(rel: &str) -> PathBuf {
    let p = repo_root().join(rel);
    assert!(
        p.is_file(),
        "conformance path missing: {rel} → {}",
        p.display()
    );
    p
}

fn assert_expect_bool(ex: &crate::query::QueryExecutionPreview, key: &str, want: bool) {
    let got = match key {
        "execution.alignment.aligned" => ex
            .alignment
            .as_ref()
            .map(|a| a.aligned)
            .unwrap_or_else(|| panic!("missing execution.alignment")),
        other => panic!("unsupported bool expect key `{other}` — extend conformance runner"),
    };
    assert_eq!(got, want, "{key}: got {got} want {want}");
}

fn assert_expect_number(ex: &crate::query::QueryExecutionPreview, key: &str, want: f64, tol: f64) {
    let got = match key {
        "execution.operation_mean" => ex.operation_mean,
        "execution.operation_sum" => ex.operation_sum,
        "execution.operation_var" => ex.operation_var,
        "execution.operation_element_count" => ex.operation_element_count.map(|n| n as f64),
        "execution.window_count" => ex.window_count.map(|n| n as f64),
        "execution.group_count" => ex.group_count.map(|n| n as f64),
        other => panic!("unsupported expect key `{other}` — extend conformance runner"),
    };
    let got = got.unwrap_or_else(|| panic!("missing {key} in execution preview"));
    assert!(
        (got - want).abs() <= tol,
        "{key}: got {got} want {want} (tol {tol})"
    );
}

fn assert_expect_f64_array(
    ex: &crate::query::QueryExecutionPreview,
    key: &str,
    want: &[f64],
    tol: f64,
) {
    let got = match key {
        "execution.operation_reduced_sum" => ex.operation_reduced_sum.as_deref(),
        "execution.operation_reduced_mean" => ex.operation_reduced_mean.as_deref(),
        other => panic!("unsupported array expect key `{other}` — extend conformance runner"),
    };
    let got = got.unwrap_or_else(|| panic!("missing {key} in execution preview"));
    assert_eq!(
        got.len(),
        want.len(),
        "{key}: len mismatch got {} want {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{key}[{i}]: got {g} want {w} (tol {tol})"
        );
    }
}

fn run_accept(case: &AcceptCase) {
    let tet = resolve(&case.tet);
    if let Some(v) = case.verify.as_deref() {
        assert_eq!(
            v, "ok",
            "{}: only verify=\"ok\" is supported in pack_version 1",
            case.id
        );
    }
    let report = verify_tet_file(&tet).unwrap_or_else(|e| panic!("{}: verify err {e}", case.id));
    assert!(
        report.ok,
        "{}: verify must succeed, findings={:?}",
        case.id, report.findings
    );

    let Some(query_rel) = case.query.as_ref() else {
        return;
    };
    let query_text = fs::read_to_string(resolve(query_rel)).unwrap();
    let doc = parse_query_json(&query_text).unwrap_or_else(|e| {
        panic!("{}: parse query: {e}", case.id);
    });
    validate_query(&doc).unwrap_or_else(|e| panic!("{}: validate: {e}", case.id));

    let mmap = mmap_file_read(&tet).unwrap();
    let tet_str = tet
        .to_str()
        .unwrap_or_else(|| panic!("{}: non-UTF-8 path", case.id));
    let resp = plan_query_with_tet_mmap(&doc, Some(tet_str), &mmap, Some(0))
        .unwrap_or_else(|e| panic!("{}: execute: {e}", case.id));
    let ex = resp
        .execution
        .as_ref()
        .unwrap_or_else(|| panic!("{}: missing execution", case.id));

    let tol = case.expect_tol.unwrap_or(1e-5);
    let expect = case
        .expect
        .as_ref()
        .unwrap_or_else(|| panic!("{}: query case needs expect", case.id));
    for (key, val) in expect {
        if let Some(arr) = val.as_array() {
            let want: Vec<f64> = arr
                .iter()
                .map(|v| {
                    v.as_f64().unwrap_or_else(|| {
                        panic!("{}: expect `{key}` array entries must be numbers", case.id)
                    })
                })
                .collect();
            assert_expect_f64_array(ex, key, &want, tol);
        } else if let Some(want) = val.as_bool() {
            assert_expect_bool(ex, key, want);
        } else {
            let want = val.as_f64().unwrap_or_else(|| {
                panic!(
                    "{}: expect `{key}` must be a number, bool, or array",
                    case.id
                )
            });
            assert_expect_number(ex, key, want, tol);
        }
    }
}

fn run_reject_verify(case: &RejectVerifyCase) {
    let path = resolve(&case.tet);
    let data = fs::read(&path).unwrap();
    let report = verify_tet_bytes(&data, Some(&path), VerifyOptions::default());
    assert!(
        !report.ok,
        "{}: expected verify failure, findings={:?}",
        case.id, report.findings
    );
    if let Some(check) = case.expect_check.as_deref() {
        assert!(
            report.findings.iter().any(|f| f.check == check && !f.ok),
            "{}: expected failing check `{check}`, findings={:?}",
            case.id,
            report.findings
        );
    }
}

fn run_reject_query(case: &RejectQueryCase) {
    let text = fs::read_to_string(resolve(&case.query)).unwrap();
    let err = match parse_query_json(&text) {
        Err(e) => e.to_string(),
        Ok(doc) => match validate_query(&doc) {
            Err(e) => e.to_string(),
            Ok(()) => panic!("{}: query should have been rejected", case.id),
        },
    };
    let needle = case.expect_error_substr.to_ascii_lowercase();
    assert!(
        err.to_ascii_lowercase().contains(&needle),
        "{}: error `{err}` missing substr `{}`",
        case.id,
        case.expect_error_substr
    );
}

#[test]
fn conformance_pack_v1() {
    let m = load_manifest();
    assert!(
        !m.accept.is_empty() && !m.reject_verify.is_empty() && !m.reject_query.is_empty(),
        "manifest must list accept, reject_verify, and reject_query cases"
    );
    for case in &m.accept {
        run_accept(case);
    }
    for case in &m.reject_verify {
        run_reject_verify(case);
    }
    for case in &m.reject_query {
        run_reject_query(case);
    }
}

#[test]
fn conformance_readme_and_manifest_present() {
    assert!(repo_root().join("conformance/README.md").is_file());
    assert!(repo_root().join("conformance/manifest.v1.json").is_file());
}

/// Touch path helper so `cargo test` links stay honest under rename.
#[test]
fn conformance_reject_query_files_exist() {
    for rel in [
        "conformance/reject/queries/unknown_top_level_key.json",
        "conformance/reject/queries/nested_operation.json",
        "conformance/reject/queries/nested_output.json",
        "conformance/reject/queries/unknown_filter_key.json",
        "conformance/reject/queries/empty_dataset.json",
        "conformance/reject/queries/dual_sum_mean.json",
        "conformance/reject/queries/invalid_axis_label.json",
        "conformance/reject/queries/unknown_write_field.json",
        "conformance/reject/queries/unknown_window_field.json",
        "conformance/reject/queries/window_size_zero.json",
        "conformance/reject/queries/window_without_reduction.json",
        "conformance/reject/queries/window_with_group_by.json",
        "conformance/reject/queries/window_mode_legacy.json",
        "conformance/reject/queries/unknown_align_field.json",
        "conformance/reject/queries/align_with_reduction.json",
        "conformance/reject/queries/align_empty_axes.json",
        "conformance/reject/queries/unknown_group_by_field.json",
        "conformance/reject/queries/group_by_labels_bad_by.json",
        "conformance/reject/queries/window_pad_unknown.json",
        "conformance/reject/queries/window_step_on_bins.json",
        "conformance/reject/queries/window_align_unknown.json",
    ] {
        assert!(Path::new(&resolve(rel)).is_file());
    }
}

#[test]
fn conformance_accept_query_files_exist() {
    for rel in [
        "conformance/accept/queries/mean_strided_sample.json",
        "conformance/accept/queries/window_bins_sum_col.json",
        "conformance/accept/queries/window_rolling_sum_col.json",
        "conformance/accept/queries/window_rolling_step2_sum_col.json",
        "conformance/accept/queries/window_bins_pad0_sum_col.json",
        "conformance/accept/queries/window_bins_align_end_sum_col.json",
        "conformance/accept/queries/align_pair_ok.json",
        "conformance/accept/queries/group_by_labels_col_sum.json",
    ] {
        assert!(Path::new(&resolve(rel)).is_file());
    }
}

#[test]
fn conformance_spec_milestone_is_0_6_3() {
    let path = repo_root().join("conformance/manifest.v1.json");
    let v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(
        v.get("spec_milestone").and_then(|x| x.as_str()),
        Some("0.6.3")
    );
}
