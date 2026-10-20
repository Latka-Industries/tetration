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

fn assert_expect_number(ex: &crate::query::QueryExecutionPreview, key: &str, want: f64, tol: f64) {
    let got = match key {
        "execution.operation_mean" => ex.operation_mean,
        "execution.operation_sum" => ex.operation_sum,
        "execution.operation_var" => ex.operation_var,
        "execution.operation_element_count" => ex.operation_element_count.map(|n| n as f64),
        other => panic!("unsupported expect key `{other}` — extend conformance runner"),
    };
    let got = got.unwrap_or_else(|| panic!("missing {key} in execution preview"));
    assert!(
        (got - want).abs() <= tol,
        "{key}: got {got} want {want} (tol {tol})"
    );
}

fn run_accept(case: &AcceptCase) {
    let tet = resolve(&case.tet);
    if let Some(v) = case.verify.as_deref() {
        assert_eq!(v, "ok", "{}: only verify=\"ok\" is supported in pack_version 1", case.id);
    }
    let report = verify_tet_file(&tet).unwrap_or_else(|e| panic!("{}: verify err {e}", case.id));
    assert!(
        report.ok,
        "{}: verify must succeed, findings={:?}",
        case.id,
        report.findings
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
    let tet_str = tet.to_str().unwrap_or_else(|| panic!("{}: non-UTF-8 path", case.id));
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
        let want = val
            .as_f64()
            .unwrap_or_else(|| panic!("{}: expect `{key}` must be a number", case.id));
        assert_expect_number(ex, key, want, tol);
    }
}

fn run_reject_verify(case: &RejectVerifyCase) {
    let path = resolve(&case.tet);
    let data = fs::read(&path).unwrap();
    let report = verify_tet_bytes(&data, Some(&path), VerifyOptions::default());
    assert!(
        !report.ok,
        "{}: expected verify failure, findings={:?}",
        case.id,
        report.findings
    );
    if let Some(check) = case.expect_check.as_deref() {
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.check == check && !f.ok),
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
        "conformance/reject/queries/unknown_filter_key.json",
    ] {
        assert!(Path::new(&resolve(rel)).is_file());
    }
}
