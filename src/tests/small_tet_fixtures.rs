//! Committed `fixtures/small/tet/*.tet` — manual CLI smoke + CI gate.
//!
//! Regenerate: `UPDATE_SMALL_TET=1 cargo test --lib regenerate_tracked_small_tet_fixtures -- --ignored --nocapture`

use std::path::Path;
use std::process::Command;

use crate::query::plan_query_with_tet_mmap;
use crate::verify::{VerifyOptions, verify_tet_bytes, verify_tet_file};
use crate::{layout::mmap_file_read, verify::DEEP_DECODE_MAX_CHUNKS};

use super::fixture::{
    VERIFY_LARGE_CHUNK, VERIFY_LARGE_SHAPE, tracked_small_tet_dir, write_tracked_small_tet_fixtures,
};
use super::verify::assert_tet_verify_ok;

fn tet_bin() -> Command {
    if let Ok(exe) = std::env::var("CARGO_BIN_EXE_tet") {
        return Command::new(exe);
    }
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for rel in ["target/debug/tet", "target/release/tet"] {
        let bin = root.join(rel);
        if bin.is_file() {
            return Command::new(bin);
        }
    }
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "--quiet", "--bin", "tet", "--"]);
    cmd
}

fn fixture(name: &str) -> std::path::PathBuf {
    tracked_small_tet_dir().join(name)
}

fn assert_fixture_exists(path: &Path) {
    assert!(
        path.is_file(),
        "missing {}; run UPDATE_SMALL_TET=1 cargo test --lib regenerate_tracked_small_tet_fixtures -- --ignored --nocapture",
        path.display()
    );
}

#[test]
#[ignore = "regenerate fixtures/small/tet/"]
fn regenerate_tracked_small_tet_fixtures() {
    if std::env::var("UPDATE_SMALL_TET").is_err() {
        eprintln!("set UPDATE_SMALL_TET=1 to write fixtures/small/tet/");
        return;
    }
    write_tracked_small_tet_fixtures(&tracked_small_tet_dir());
}

#[test]
fn tracked_small_tet_verify_and_query() {
    let sample = fixture("sample.tet");
    let large = fixture("large.tet");
    let plan = fixture("plan.tet");
    let u8 = fixture("multichunk_u8.tet");
    let u32 = fixture("multichunk_u32.tet");
    let f16 = fixture("multichunk_f16.tet");
    let recording = fixture("recording.tet");
    for p in [&sample, &large, &u8, &u32, &f16, &recording] {
        assert_fixture_exists(p);
    }
    assert_fixture_exists(&plan);

    assert_tet_verify_ok(&sample);
    assert_tet_verify_ok(&u8);
    assert_tet_verify_ok(&u32);
    assert_tet_verify_ok(&f16);

    let large_report = verify_tet_file(&large).unwrap();
    assert!(large_report.ok, "{:?}", large_report.findings);
    assert!(
        large_report
            .findings
            .iter()
            .any(|f| f.check == "chunk_decode_skipped"),
        "large.tet should exceed quick decode sample"
    );
    let n_chunks = (VERIFY_LARGE_SHAPE[0].div_ceil(VERIFY_LARGE_CHUNK[0])
        * VERIFY_LARGE_SHAPE[1].div_ceil(VERIFY_LARGE_CHUNK[1])) as usize;
    assert!(n_chunks > DEEP_DECODE_MAX_CHUNKS);

    let data = std::fs::read(&large).unwrap();
    let deep = verify_tet_bytes(&data, Some(&large), VerifyOptions { deep_decode: true });
    assert!(deep.ok);
    assert!(deep.summary.as_ref().is_some_and(|s| s.deep_chunk_decode));

    let plan_report = verify_tet_file(&plan).unwrap();
    assert!(!plan_report.ok);
    assert!(
        plan_report
            .recommendations
            .iter()
            .any(|r| r.code == "footer_invalid"),
        "{:?}",
        plan_report.recommendations
    );

    const EXPECT_SUM: f64 = 21.0;
    const EXPECT_VAR: f64 = 17.5 / 6.0;

    for path in [&u8, &u32, &f16] {
        let mmap = mmap_file_read(path).unwrap();
        let sum_doc = super::fixture::query_files::json("sum_a");
        let sum_ex = plan_query_with_tet_mmap(&sum_doc, None, &mmap, Some(0))
            .unwrap()
            .execution
            .unwrap();
        assert_eq!(
            sum_ex.operation_sum.unwrap(),
            EXPECT_SUM,
            "{}",
            path.display()
        );

        let var_doc = super::fixture::query_files::json("var_a");
        let var_ex = plan_query_with_tet_mmap(&var_doc, None, &mmap, Some(0))
            .unwrap()
            .execution
            .unwrap();
        assert!(
            (var_ex.operation_var.unwrap() - EXPECT_VAR).abs() < 1e-5,
            "var {} {}",
            var_ex.operation_var.unwrap(),
            path.display()
        );
    }
}

#[test]
fn tracked_small_tet_cli_verify_query_repair() {
    let sample = fixture("sample.tet");
    let large = fixture("large.tet");
    let plan = fixture("plan.tet");
    let u8 = fixture("multichunk_u8.tet");
    assert_fixture_exists(&sample);
    assert_fixture_exists(&large);
    assert_fixture_exists(&plan);
    assert_fixture_exists(&u8);

    let out = tet_bin()
        .args(["verify", sample.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("status: ok"));

    let out = tet_bin()
        .args(["verify", large.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("status: ok"));
    assert!(stdout.contains("chunk_decode_skipped") || stdout.contains("decode-check skipped"));

    let out = tet_bin()
        .args(["verify", "--deep", large.to_str().unwrap(), "-q"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("status=ok"));

    let out = tet_bin()
        .args(["repair", plan.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("footer_invalid") || stdout.contains("plan"));

    let query = super::fixture::query_files::path("sum_a", "json");
    let out = tet_bin()
        .args([
            "query",
            query.to_str().unwrap(),
            "-t",
            u8.to_str().unwrap(),
            "-x",
            "-q",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("sum=21"));
}
