//! `tet run` CLI smoke (THI-646).

use std::path::Path;
use std::process::{Command, Output};

fn tet_cmd() -> Command {
    if let Ok(exe) = std::env::var("CARGO_BIN_EXE_tet") {
        return Command::new(exe);
    }
    let mut cmd = Command::new(env!("CARGO"));
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    cmd.args(["run", "--quiet", "--bin", "tet", "--"]);
    cmd
}

#[test]
fn cli_run_dry_run_prints_plan_and_executes_nothing() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/jobs");
    let root = tempfile::tempdir().unwrap();
    let out = tet_cmd()
        .args([
            "run",
            fixtures.join("csv_verify_query.toml").to_str().unwrap(),
            "--dry-run",
            "--root",
            root.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr={stderr}");
    // Status word is at the end: `[1/3] convert …  plan`
    assert!(
        stderr.contains("[1/3] convert ") && stderr.contains(" plan"),
        "{stderr}"
    );
    assert!(
        stderr.contains("[3/3] query dataset=temp @ ") && stderr.contains(" plan"),
        "{stderr}"
    );
    assert!(stderr.contains("dry run ok: 3 step(s) planned"), "{stderr}");
    assert!(!root.path().join("work").exists());

    let bad = tet_cmd()
        .args(["run", "fixtures/jobs/missing.toml", "--dry-run"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .unwrap();
    assert!(!bad.status.success());
    assert!(String::from_utf8_lossy(&bad.stderr).contains("error:"));
}

#[test]
fn cli_run_step_failure_names_the_missing_path() {
    let dir = tempfile::tempdir().unwrap();
    let job = dir.path().join("job.toml");
    std::fs::write(
        &job,
        "schema_version = 1\n[[step]]\nop = \"verify\"\npath = \"nope/missing.tet\"\n",
    )
    .unwrap();
    let out = tet_cmd()
        .args(["run", job.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Match step identity + basename only: Windows I/O wording differs, and temp
    // dirs may surface as short (`RUNNER~1`) vs long paths in Display.
    assert!(
        stderr.contains("error: step 1 (verify) failed:") && stderr.contains("missing.tet"),
        "{stderr}"
    );
}

/// Temp dir holding `raw/table.csv` so the fixture job can write `work/` safely.
fn scratch_with_csv() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("raw")).unwrap();
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/jobs/raw/table.csv"),
        dir.path().join("raw/table.csv"),
    )
    .unwrap();
    dir
}

fn stdout_json(out: &Output) -> serde_json::Value {
    let stdout = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!(
            "stdout is not JSON ({e}): {stdout}\nstderr={}",
            String::from_utf8_lossy(&out.stderr)
        )
    })
}

#[test]
fn cli_run_json_prints_job_report_on_stdout_and_progress_on_stderr() {
    let dir = scratch_with_csv();
    let job = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/jobs/csv_verify_query.toml");
    let out = tet_cmd()
        .args([
            "run",
            job.to_str().unwrap(),
            "--json",
            "--root",
            dir.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr={stderr}");
    // Progress chrome stays on stderr.
    assert!(stderr.contains("[1/3] convert "), "{stderr}");
    assert!(stderr.contains("job ok: 3 step(s)"), "{stderr}");

    let report = stdout_json(&out);
    assert_eq!(report["dry_run"], false);
    assert!(report.get("error").is_none());
    let steps = report["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0]["op"], "convert");
    assert_eq!(steps[0]["data"]["dataset_count"], 2);
    assert_eq!(steps[1]["op"], "verify");
    assert_eq!(steps[1]["data"]["ok"], true);
    assert_eq!(steps[2]["op"], "query");
    assert!(
        steps[2]["data"]["execution"]["operation_mean"].is_number(),
        "{}",
        steps[2]["data"]
    );
}

#[test]
fn cli_run_json_keeps_multiline_step_dumps_off_stdout() {
    let dir = tempfile::tempdir().unwrap();
    let job = dir.path().join("job.toml");
    std::fs::write(
        &job,
        "schema_version = 1\n[[step]]\nop = \"convert\"\nfrom = \"t.csv\"\nto = \"t.tet\"\n\
         [[step]]\nop = \"info\"\npath = \"t.tet\"\nquiet = false\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("t.csv"), "a,b\n1,2\n3,4\n").unwrap();
    let out = tet_cmd()
        .args(["run", job.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(out.status.success());
    // Multi-line `info` text would corrupt stdout if it were printed there.
    let report = stdout_json(&out);
    assert!(report["steps"][1]["data"]["summary"].is_object());
    assert!(report["steps"][1]["stdout"].is_string());
}

#[test]
fn cli_run_json_dry_run_is_a_valid_planned_report() {
    let dir = tempfile::tempdir().unwrap();
    let job = Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/jobs/csv_verify_query.toml");
    let out = tet_cmd()
        .args([
            "run",
            job.to_str().unwrap(),
            "--dry-run",
            "--json",
            "--root",
            dir.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report = stdout_json(&out);
    assert_eq!(report["dry_run"], true);
    let steps = report["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 3);
    assert!(steps.iter().all(|s| s["status"] == "planned"));
    assert!(steps.iter().all(|s| s["data"].is_null()));
    assert!(!dir.path().join("work").exists());
}

#[test]
fn cli_run_json_step_failure_prints_partial_report_and_exits_nonzero() {
    let dir = scratch_with_csv();
    let job = dir.path().join("job.toml");
    std::fs::write(
        &job,
        "schema_version = 1\n\
         [[step]]\nop = \"convert\"\nfrom = \"raw/table.csv\"\nto = \"work/a.tet\"\n\
         [[step]]\nop = \"verify\"\npath = \"work/missing.tet\"\n",
    )
    .unwrap();
    let out = tet_cmd()
        .args(["run", job.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("error: step 2 (verify) failed:"),
        "{stderr}"
    );
    let report = stdout_json(&out);
    assert_eq!(report["steps"].as_array().unwrap().len(), 1);
    assert_eq!(report["steps"][0]["data"]["dataset_count"], 2);
    // stderr is 1-based ("step 2"); JSON error.index is 0-based.
    assert_eq!(report["error"]["index"], 1);
    assert_eq!(report["error"]["op"], "verify");
    assert!(report["error"]["message"].is_string());

    // First-step failure: nothing completed → empty stdout (same contract as pre-flight).
    std::fs::write(
        &job,
        "schema_version = 1\n[[step]]\nop = \"verify\"\npath = \"work/missing.tet\"\n",
    )
    .unwrap();
    let first = tet_cmd()
        .args(["run", job.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert_eq!(first.status.code(), Some(1));
    assert!(
        first.stdout.is_empty(),
        "first-step failure must not print partial JSON: {}",
        String::from_utf8_lossy(&first.stdout)
    );
    let first_err = String::from_utf8_lossy(&first.stderr);
    assert!(
        first_err.contains("error: step 1 (verify) failed:"),
        "{first_err}"
    );

    // Pre-flight failure: nothing completed, so stdout stays empty.
    std::fs::write(&job, "schema_version = 1\n[[step]]\nop = \"nope\"\n").unwrap();
    let bad = tet_cmd()
        .args(["run", job.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(1));
    assert!(bad.stdout.is_empty());
}
