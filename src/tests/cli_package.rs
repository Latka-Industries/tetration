//! CLI smoke: parallel ingest → publish → query (THI-467).

use std::path::Path;
use std::process::Command;

use crate::catalog::{TetDatasetWrite, TetWriterSession};
use crate::verify::looks_like_tet_package;

fn f32_pair(a: f32, b: f32) -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    buf[..4].copy_from_slice(&a.to_le_bytes());
    buf[4..].copy_from_slice(&b.to_le_bytes());
    buf
}

fn write_one_tet(path: &Path, name: &str, a: f32, b: f32) {
    let mut s = TetWriterSession::create(path);
    s.push_dataset(TetDatasetWrite::f32_row_major(name, &[2], &[2], f32_pair(a, b)).unwrap())
        .unwrap();
    s.commit().unwrap();
}

fn tet_cmd() -> Command {
    // Prefer the just-built bin when cargo test sets CARGO_BIN_EXE_tet (integration).
    // Lib tests: always `cargo run` so we do not pick a stale target/debug/tet.
    if let Ok(exe) = std::env::var("CARGO_BIN_EXE_tet") {
        return Command::new(exe);
    }
    let mut cmd = Command::new(env!("CARGO"));
    cmd.current_dir(env!("CARGO_MANIFEST_DIR"));
    cmd.args(["run", "--quiet", "--bin", "tet", "--"]);
    cmd
}

#[test]
fn cli_ingest_publish_query_pack_smoke() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("temp.tet");
    let b = dir.path().join("pressure.tet");
    write_one_tet(&a, "temp", 1.0, 2.0);
    write_one_tet(&b, "pressure", 3.0, 4.0);

    let pkg = dir.path().join("climate.tet.d");
    let ingest = tet_cmd()
        .args([
            "ingest",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
            "--out",
            pkg.to_str().unwrap(),
            "--jobs",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        ingest.status.success(),
        "ingest: {}",
        String::from_utf8_lossy(&ingest.stderr)
    );
    assert!(looks_like_tet_package(&pkg));

    let publish = tet_cmd()
        .args(["publish", pkg.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        publish.status.success(),
        "publish: {}",
        String::from_utf8_lossy(&publish.stderr)
    );

    let query = tet_cmd()
        .args([
            "query",
            r#"{"dataset":"temp","mean":[]}"#,
            "-t",
            pkg.to_str().unwrap(),
            "-x",
            "-q",
        ])
        .output()
        .unwrap();
    assert!(
        query.status.success(),
        "query: {}\n{}",
        String::from_utf8_lossy(&query.stderr),
        String::from_utf8_lossy(&query.stdout)
    );
    let qout = String::from_utf8_lossy(&query.stdout);
    assert!(
        qout.contains("mean") || qout.contains("1.5") || qout.contains("temp"),
        "unexpected query stdout: {qout}"
    );

    let packed = dir.path().join("climate.tet");
    let pack = tet_cmd()
        .args([
            "pack",
            pkg.to_str().unwrap(),
            "-o",
            packed.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        pack.status.success(),
        "pack: {}",
        String::from_utf8_lossy(&pack.stderr)
    );
    assert!(packed.is_file());

    let verify = tet_cmd()
        .args(["verify", pkg.to_str().unwrap(), "-q"])
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "verify package: {}",
        String::from_utf8_lossy(&verify.stderr)
    );
}

#[test]
fn ingest_tet_file_library_roundtrip() {
    use crate::catalog::{TetPackage, ingest_tet_file, list_sealed_parts, publish_parts};

    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("one.tet");
    write_one_tet(&src, "a", 1.0, 2.0);
    let pkg = TetPackage::create(dir.path().join("p.tet.d")).unwrap();
    let meta = ingest_tet_file(&pkg, &src, "s0", "one").unwrap();
    assert!(meta.sealed);
    assert_eq!(meta.claims.len(), 1);
    assert_eq!(list_sealed_parts(&pkg).unwrap().len(), 1);
    let manifest = publish_parts(&pkg, 1, &["one".into()]).unwrap();
    assert_eq!(manifest.generation, 1);
    let file = pkg.open_current().unwrap();
    assert_eq!(file.summary().unwrap().datasets[0].name, "a");
}
