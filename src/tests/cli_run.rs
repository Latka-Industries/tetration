//! `tet run` CLI smoke (THI-646).

use std::path::Path;
use std::process::Command;

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
