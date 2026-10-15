//! `tet verify` / [`verify_tet_file`] integration tests.

use std::path::Path;
use std::process::Command;

use crate::catalog::{FooterBlobV1, TetMetadataV1, write_footer_blob};
use crate::verify::{
    VerifyOptions, format_verify_quiet, format_verify_text, verify_tet_bytes, verify_tet_file,
};

use super::fixture::{
    index_patch::{self, ENTRY_PAYLOAD_OFFSET},
    write_multichunk_2x3_tiles,
};

/// Assert [`verify_tet_file`] passes (used after writer/convert tests — CI gate).
pub(crate) fn assert_tet_verify_ok(path: &Path) {
    let report = verify_tet_file(path).unwrap();
    assert!(
        report.ok,
        "verify failed for {}: {:?}",
        path.display(),
        report.findings
    );
}

fn tet_verify_command(tet_path: &std::path::Path) -> Command {
    let path_arg = tet_path.to_str().expect("utf-8 path");
    if let Ok(exe) = std::env::var("CARGO_BIN_EXE_tet") {
        let mut cmd = Command::new(exe);
        cmd.args(["verify", path_arg]);
        return cmd;
    }
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "--quiet", "--bin", "tet", "--", "verify", path_arg]);
    cmd
}

#[test]
fn verify_ok_on_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ok.tet");
    write_multichunk_2x3_tiles(&path, "temperature");

    let report = verify_tet_file(&path).unwrap();
    assert!(report.ok, "{:?}", report.findings);
    assert!(
        report
            .summary
            .as_ref()
            .is_some_and(|s| s.dataset_count == 1)
    );
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "chunk_decode" && f.ok)
    );
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "dataset_tensor_bytes" && f.ok)
    );

    let text = format_verify_text(&report);
    assert!(text.contains("status: ok"));
}

#[test]
fn verify_quiet_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let data = std::fs::read(&path).unwrap();
    let report = verify_tet_bytes(&data, Some(&path), VerifyOptions::default());
    let line = format_verify_quiet(&report);
    assert!(line.contains("status=ok"));
}

#[test]
fn verify_fails_on_bad_payload_offset() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.tet");
    write_multichunk_2x3_tiles(&path, "a");
    index_patch::patch_first_index_entry_u64(&path, ENTRY_PAYLOAD_OFFSET, u64::MAX);

    let report = verify_tet_file(&path).unwrap();
    assert!(!report.ok);
    assert!(
        report.findings.iter().any(|f| !f.ok),
        "{:?}",
        report.findings
    );
    assert!(!report.recommendations.is_empty());
    assert!(
        report.recommendations.iter().all(|r| {
            r.fix.as_ref().is_none_or(|f| {
                f.command
                    .as_deref()
                    .is_none_or(|c| !c.contains("tet repair"))
            })
        }),
        "non-repairable issues should not get tet repair commands"
    );
}

#[test]
fn verify_fails_on_tile_raw_byte_len_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("raw.tet");
    write_multichunk_2x3_tiles(&path, "a");
    index_patch::patch_first_index_entry_raw_and_stored(&path, 4, 4);

    let report = verify_tet_file(&path).unwrap();
    assert!(!report.ok);
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "dataset_tensor_bytes" && !f.ok),
        "{:?}",
        report.findings
    );
}

#[test]
fn verify_deep_decode_option_on_fixture() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deep.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let data = std::fs::read(&path).unwrap();
    let report = verify_tet_bytes(&data, Some(&path), VerifyOptions { deep_decode: true });
    assert!(report.ok);
    assert!(report.summary.as_ref().is_some_and(|s| s.deep_chunk_decode));
}

#[test]
fn verify_ok_with_footer_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("footer.tet");
    write_multichunk_2x3_tiles(&path, "a");
    write_footer_blob(
        &path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1::default()),
            metadata_ref: None,
        },
    )
    .unwrap();

    let report = verify_tet_file(&path).unwrap();
    assert!(report.ok);
    assert!(report.summary.as_ref().is_some_and(|s| s.history_footer));
}

#[test]
fn tet_verify_binary_smoke() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("smoke.tet");
    write_multichunk_2x3_tiles(&path, "a");
    let out = tet_verify_command(&path).output().unwrap();
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("status: ok"));
}

/// LibFuzzer crash (THI-376): malformed chunk coords used to panic in `tile_extent` overflow.
#[test]
fn verify_tet_bytes_no_panic_on_overflow_tile_extent_fixture() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/small/tet/reject/overflow_tile_extent.tet");
    let data = std::fs::read(&path).expect("reject fixture present");
    let report = verify_tet_bytes(&data, Some(&path), VerifyOptions::default());
    assert!(!report.ok, "malformed fixture must fail verify: {report:?}");
}

/// LibFuzzer crash (THI-376): huge `dataset_count` must not `Vec::with_capacity` OOM.
#[test]
fn verify_tet_bytes_no_oom_on_inflated_dataset_count_fixture() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/small/tet/reject/oom_dataset_count.tet");
    let data = std::fs::read(&path).expect("reject fixture present");
    let report = verify_tet_bytes(&data, Some(&path), VerifyOptions::default());
    assert!(!report.ok, "malformed fixture must fail verify: {report:?}");
}

/// LibFuzzer crash (THI-376): zero `chunk_shape` must not `div_ceil` panic.
#[test]
fn verify_tet_bytes_no_panic_on_zero_chunk_shape_fixture() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/small/tet/reject/zero_chunk_shape.tet");
    let data = std::fs::read(&path).expect("reject fixture present");
    let report = verify_tet_bytes(&data, Some(&path), VerifyOptions::default());
    assert!(!report.ok, "malformed fixture must fail verify: {report:?}");
}

#[test]
fn tile_extent_out_of_grid_coords_do_not_panic() {
    use crate::catalog::tile::tile_extent;
    // coord * chunk_shape would overflow or land past shape — must not panic.
    let extent = tile_extent(&[2, 3], &[2, 2], &[u64::MAX / 2, 0], 2);
    assert_eq!(extent.len(), 2);
    let _ = tile_extent(&[2, 3], &[2, 2], &[0, u64::MAX], 2);
}
