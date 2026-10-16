//! CLI: `tet convert` skip report and `--strict` (THI-485).

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

fn write_mixed_zarr(root: &Path) {
    let data = root.join("data");
    std::fs::create_dir_all(data.join("c")).unwrap();
    std::fs::create_dir_all(root.join("label")).unwrap();
    std::fs::write(
        root.join("zarr.json"),
        r#"{"zarr_format":3,"node_type":"group"}"#,
    )
    .unwrap();
    std::fs::write(
        data.join("zarr.json"),
        r#"{"zarr_format":3,"node_type":"array","shape":[2],"data_type":"float32","chunk_grid":{"name":"regular","configuration":{"chunk_shape":[2]}},"codecs":[{"name":"bytes","configuration":{"endian":"little"}}]}"#,
    )
    .unwrap();
    let mut bytes = Vec::new();
    bytes.extend(1.0f32.to_le_bytes());
    bytes.extend(2.0f32.to_le_bytes());
    std::fs::write(data.join("c/0"), bytes).unwrap();
    std::fs::write(
        root.join("label/zarr.json"),
        r#"{"zarr_format":3,"node_type":"array","shape":[2],"data_type":"string","chunk_grid":{"name":"regular","configuration":{"chunk_shape":[2]}},"codecs":[{"name":"bytes","configuration":{"endian":"little"}}]}"#,
    )
    .unwrap();
}

#[test]
fn cli_convert_prints_skips_and_strict_fails() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("mixed.zarr");
    write_mixed_zarr(&store);
    let output = dir.path().join("mixed.tet");

    let ok = tet_cmd()
        .args([
            "convert",
            store.to_str().unwrap(),
            output.to_str().unwrap(),
            "--jobs",
            "1",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&ok.stderr);
    let stdout = String::from_utf8_lossy(&ok.stdout);
    assert!(ok.status.success(), "convert: {stderr}");
    assert!(stderr.contains("skipped 1 dataset"), "{stderr}");
    assert!(stderr.contains("label"), "{stderr}");
    assert!(stdout.contains("\"path\": \"label\""), "{stdout}");
    assert!(output.is_file());

    let strict_out = dir.path().join("strict.tet");
    let strict = tet_cmd()
        .args([
            "convert",
            "--strict",
            store.to_str().unwrap(),
            strict_out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let strict_err = String::from_utf8_lossy(&strict.stderr);
    assert!(!strict.status.success(), "strict should fail: {strict_err}");
    assert!(strict_err.contains("label"), "{strict_err}");
    assert!(strict_err.contains("--strict"), "{strict_err}");
    assert!(!strict_out.exists());
}
