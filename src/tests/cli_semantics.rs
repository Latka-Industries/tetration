//! `tet semantics` CLI smoke (THI-583).

use std::{path::PathBuf, process::Command};

use crate::catalog::read_tet_summary_v1;
use crate::layout::mmap_file_read;

use super::fixture::write_multichunk_2x3_tiles;

fn tet_command(args: &[&str]) -> Command {
    // Lib tests do not set `CARGO_BIN_EXE_tet`; avoid stale `target/*/tet` from other target dirs.
    if let Ok(exe) = std::env::var("CARGO_BIN_EXE_tet") {
        let mut cmd = Command::new(exe);
        cmd.args(args);
        return cmd;
    }
    let mut cmd = Command::new(env!("CARGO"));
    let mut full = vec!["run", "--quiet", "--bin", "tet", "--"];
    full.extend(args);
    cmd.args(full);
    cmd
}

#[test]
fn cli_semantics_set_then_info_and_scaffold() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.tet");
    write_multichunk_2x3_tiles(&path, "frames");
    let path_s = path.to_str().unwrap();

    let set = tet_command(&[
        "semantics",
        "set",
        path_s,
        "--dataset",
        "frames",
        "--set",
        "frame_rate=30",
        "--set",
        "time_axis=0",
        "--set",
        "media_kind=video",
    ])
    .output()
    .expect("spawn tet semantics set");
    assert!(
        set.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&set.stderr)
    );

    let info = tet_command(&["info", path_s])
        .output()
        .expect("spawn tet info");
    assert!(info.status.success());
    let info_out = String::from_utf8_lossy(&info.stdout);
    assert!(info_out.contains("frame_rate"), "{info_out}");
    assert!(info_out.contains("30"), "{info_out}");
    assert!(info_out.contains("time_axis"), "{info_out}");
    assert!(info_out.contains('0'), "{info_out}");

    let scaffold = tet_command(&["semantics", "scaffold", path_s, "--format", "toml"])
        .output()
        .expect("spawn tet semantics scaffold");
    assert!(
        scaffold.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&scaffold.stderr)
    );
    let stub = String::from_utf8_lossy(&scaffold.stdout);
    assert!(stub.contains("schema_version = 1"), "{stub}");
    assert!(stub.contains("[datasets.frames]"), "{stub}");
    assert!(stub.contains("[profiles.media.v1]"), "{stub}");
    assert!(stub.contains("# frame_rate_unit ="), "{stub}");
    assert!(!stub.contains("id = "), "{stub}");

    let out_path = dir.path().join("tet-semantics.json");
    let scaffold_json = tet_command(&[
        "semantics",
        "scaffold",
        path_s,
        "--format",
        "json",
        "-o",
        out_path.to_str().unwrap(),
    ])
    .output()
    .expect("spawn scaffold -o");
    assert!(scaffold_json.status.success());
    let written = std::fs::read_to_string(&out_path).unwrap();
    assert!(written.contains("\"schema_version\": 1"), "{written}");
    assert!(written.contains("\"datasets\""), "{written}");
    assert!(written.contains("\"media.v1\""), "{written}");
    assert!(!written.contains("\"declared\""), "{written}");
}

#[test]
fn cli_semantics_apply_prints_stamped_vs_left() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.tet");
    write_multichunk_2x3_tiles(&path, "frames");
    let toml = dir.path().join("tet-semantics.toml");
    std::fs::write(
        &toml,
        r#"
schema_version = 1

[datasets.frames]
dim_names = ["y", "x"]

[datasets.frames.attrs]
semantics_profile = "media.v1"
frame_rate = 30
time_axis = 0

[profiles.media.v1]
index_to_seconds = "index / frame_rate"

[units.Hz]
display = "Hz"
"#,
    )
    .unwrap();
    let before = std::fs::read(&toml).unwrap();
    let out = tet_command(&[
        "semantics",
        "apply",
        path.to_str().unwrap(),
        "--from",
        toml.to_str().unwrap(),
    ])
    .output()
    .expect("spawn tet semantics apply");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("dim_names=[y, x]"), "{err}");
    assert!(err.contains("profiles media.v1"), "{err}");
    assert!(err.contains("units Hz"), "{err}");
    assert!(!err.contains("index_to_seconds"), "{err}");
    assert_eq!(std::fs::read(&toml).unwrap(), before);

    let info = tet_command(&["info", path.to_str().unwrap()])
        .output()
        .expect("spawn tet info");
    assert!(info.status.success());
    let info_out = String::from_utf8_lossy(&info.stdout);
    assert!(info_out.contains("frame_rate"), "{info_out}");
    assert!(info_out.contains("30"), "{info_out}");

    let annotated = dir.path().join("annotated.toml");
    let again = tet_command(&[
        "semantics",
        "apply",
        path.to_str().unwrap(),
        "--from",
        toml.to_str().unwrap(),
        "--annotate",
        "-o",
        annotated.to_str().unwrap(),
    ])
    .output()
    .expect("spawn tet semantics apply --annotate");
    assert!(
        again.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&again.stderr)
    );
    assert_eq!(std::fs::read(&toml).unwrap(), before);
    let body = std::fs::read_to_string(&annotated).unwrap();
    assert!(body.contains("# media_kind ="), "{body}");
    assert!(
        body.contains("index_to_seconds = \"index / frame_rate\""),
        "{body}"
    );
    assert!(body.contains("[units.Hz]"), "{body}");

    let same = tet_command(&[
        "semantics",
        "apply",
        path.to_str().unwrap(),
        "--from",
        toml.to_str().unwrap(),
        "--annotate",
        "-o",
        toml.to_str().unwrap(),
    ])
    .output()
    .expect("spawn apply -o --from");
    assert!(!same.status.success());
    assert_eq!(std::fs::read(&toml).unwrap(), before);
}

#[test]
fn cli_scaffold_bare_file_links_docs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bare.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let out = tet_command(&[
        "semantics",
        "scaffold",
        path.to_str().unwrap(),
        "--format",
        "json",
    ])
    .output()
    .expect("spawn scaffold");
    assert!(out.status.success());
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(body.contains("No declared semantics attrs"));
    assert!(body.contains("docs.tetration.sh/guides/semantics"));
}

#[test]
fn tracked_recording_fixture_opens() {
    let tracked =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/small/tet/recording.tet");
    if !tracked.is_file() {
        // Allow local trees before regenerate; unit test covers write_recording_fixture.
        return;
    }
    let mmap = mmap_file_read(&tracked).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets[0].name, "frames");
    assert_eq!(
        summary
            .metadata
            .datasets
            .get("frames")
            .and_then(|d| d.attrs.get("frame_rate"))
            .map(String::as_str),
        Some("30")
    );
}
