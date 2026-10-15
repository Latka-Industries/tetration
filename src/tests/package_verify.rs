//! Package verify / torn-generation reject cases (THI-466, THI-480).

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use proptest::prelude::*;

use crate::catalog::{
    GenerationManifest, PACKAGE_PROTOCOL_VERSION, TetDatasetWrite, TetPackage, TetShardWriter,
    format_generation_id, publish_parts, read_tet_summary_v1,
};
use crate::layout::mmap_file_read;
use crate::verify::{
    TetVerifyReport, VerifyOptions, looks_like_tet_package, verify_tet_package,
    verify_tet_package_with_options,
};

use super::fixture::index_patch::{self, ENTRY_CHUNK_INDEX0_OFFSET};

fn f32_le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn seal_dataset(pkg: &TetPackage, session: &str, part: &str, shape: &[u64], chunk: &[u64], data: Vec<u8>) {
    let mut w = TetShardWriter::create(pkg, session, part).unwrap();
    w.push_dataset(TetDatasetWrite::f32_row_major("a", shape, chunk, data).unwrap())
        .unwrap();
    w.seal().unwrap();
}

fn seal_one(pkg: &TetPackage, session: &str, part: &str) {
    seal_dataset(pkg, session, part, &[2], &[2], f32_le(&[1.0, 2.0]));
}

fn seal_multichunk(pkg: &TetPackage, session: &str, part: &str) {
    seal_dataset(pkg, session, part, &[4], &[2], f32_le(&[1.0, 2.0, 3.0, 4.0]));
}

/// Create `{stem}.tet.d`, seal one part, publish generation 1.
fn published_pkg(stem: &str) -> (tempfile::TempDir, PathBuf, TetPackage) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join(format!("{stem}.tet.d"));
    let pkg = TetPackage::create(&root).unwrap();
    seal_one(&pkg, "s", "p0");
    publish_parts(&pkg, 1, &["p0".into()]).unwrap();
    (dir, root, pkg)
}

fn write_manifest(pkg: &TetPackage, generation: u64, manifest: &GenerationManifest) {
    fs::write(
        pkg.manifest_path(generation),
        serde_json::to_string_pretty(manifest).unwrap() + "\n",
    )
    .unwrap();
}

fn assert_check_fails(report: &TetVerifyReport, check: &str) {
    assert!(!report.ok, "expected failure for {check}: {:?}", report.findings);
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == check && !f.ok),
        "missing failing check `{check}`: {:?}",
        report.findings
    );
}

fn assert_warn(report: &TetVerifyReport, check: &str) {
    assert!(
        report.findings.iter().any(|f| {
            f.check == check && f.severity == crate::verify::VerifySeverity::Warning
        }),
        "missing warn `{check}`: {:?}",
        report.findings
    );
}

#[test]
fn healthy_package_verify_ok() {
    let (_dir, root, pkg) = published_pkg("ok");
    assert!(pkg.read_manifest(1).unwrap().packed_sha256.is_some());
    assert!(looks_like_tet_package(&root));
    let report = verify_tet_package(&root).unwrap();
    assert!(report.ok, "{:?}", report.findings);
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "manifest_sha256" && f.ok)
    );
}

#[test]
fn current_points_at_missing_generation() {
    let (_dir, root, _pkg) = published_pkg("missing");
    fs::write(root.join("CURRENT"), "9\n").unwrap();
    assert_check_fails(&verify_tet_package(&root).unwrap(), "current_packed");
}

#[test]
fn manifest_byte_len_mismatch() {
    let (_dir, root, pkg) = published_pkg("len");
    let mut manifest = pkg.read_manifest(1).unwrap();
    manifest.packed_byte_len = 1;
    write_manifest(&pkg, 1, &manifest);
    assert_check_fails(&verify_tet_package(&root).unwrap(), "manifest_byte_len");
}

#[test]
fn manifest_sha256_mismatch() {
    let (_dir, root, pkg) = published_pkg("hash");
    let mut manifest = pkg.read_manifest(1).unwrap();
    manifest.packed_sha256 = Some("0".repeat(64));
    write_manifest(&pkg, 1, &manifest);
    assert_check_fails(&verify_tet_package(&root).unwrap(), "manifest_sha256");
}

#[test]
fn incomplete_publish_warns() {
    let (_dir, root, pkg) = published_pkg("inc");
    pkg.begin_publish(2).unwrap();
    let report = verify_tet_package(&root).unwrap();
    assert!(report.ok, "incomplete publish must not fail CURRENT");
    assert_warn(&report, "publishing_incomplete");
}

#[test]
fn bad_protocol_version_fails() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("proto.tet.d");
    let _pkg = TetPackage::create(&root).unwrap();
    fs::write(
        root.join("PROTOCOL"),
        format!(
            "{{\"protocol_version\":{}}}\n",
            PACKAGE_PROTOCOL_VERSION + 1
        ),
    )
    .unwrap();
    assert_check_fails(&verify_tet_package(&root).unwrap(), "protocol");
}

#[test]
fn truncate_packed_fails_layout() {
    let (_dir, root, pkg) = published_pkg("torn");
    let packed = pkg.packed_tet_path(1);
    let meta = fs::metadata(&packed).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(&packed)
        .unwrap()
        .set_len(meta.len() / 2)
        .unwrap();
    // Keep MANIFEST lens in sync so we hit packed_layout, not byte_len.
    let mut manifest = pkg.read_manifest(1).unwrap();
    manifest.packed_byte_len = meta.len() / 2;
    manifest.packed_sha256 = None;
    write_manifest(&pkg, 1, &manifest);

    assert_check_fails(
        &verify_tet_package_with_options(&root, VerifyOptions::default()).unwrap(),
        "packed_layout",
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Corrupt CURRENT text must not panic; verify fails or reports invalid current.
    #[test]
    fn fuzz_current_pointer_text(s in ".*") {
        let (_dir, root, _pkg) = published_pkg("fuzz");
        fs::write(root.join("CURRENT"), s.as_bytes()).unwrap();
        let report = verify_tet_package(&root).unwrap();
        // Empty/whitespace-only CURRENT is treated as invalid by parse; any non-"1" breaks pin.
        if s.trim() == "1" || s.trim() == "{\"generation\":1}" {
            assert!(report.ok || report.findings.iter().any(|f| f.check == "staging"));
        } else {
            assert!(!report.ok || report.findings.iter().any(|f| !f.ok && (
                f.check == "current"
                    || f.check == "current_packed"
                    || f.check == "current_manifest"
                    || f.check == "packed_layout"
                    || f.check == "manifest_byte_len"
                    || f.check == "manifest_sha256"
            )), "{s:?} => {:?}", report.findings);
        }
    }
}

#[test]
fn missing_current_with_sealed_generation_fails() {
    let (_dir, root, _pkg) = published_pkg("nocur");
    fs::remove_file(root.join("CURRENT")).unwrap();
    assert_check_fails(&verify_tet_package(&root).unwrap(), "current");
}

#[test]
fn duplicate_chunk_coords_in_packed_fails() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("dup.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    seal_multichunk(&pkg, "s", "p0");
    publish_parts(&pkg, 1, &["p0".into()]).unwrap();

    let packed = pkg.packed_tet_path(1);
    let coord0 = {
        let mmap = mmap_file_read(&packed).unwrap();
        let s = read_tet_summary_v1(&mmap).unwrap();
        assert!(s.chunks.len() >= 2, "need multi-chunk packed.tet");
        s.chunks[0].chunk_index[0]
    };
    // Make entry 1 share entry 0's chunk coords (simulates torn/overlapping merge).
    index_patch::patch_entry_u64(&packed, 1, ENTRY_CHUNK_INDEX0_OFFSET, coord0);
    // Drop SHA so we hit layout chunk_coords, not manifest_sha256.
    let mut manifest = pkg.read_manifest(1).unwrap();
    manifest.packed_sha256 = None;
    write_manifest(&pkg, 1, &manifest);

    assert_check_fails(&verify_tet_package(&root).unwrap(), "chunk_coords");
}

#[test]
fn quarantine_stale_lock_clears_publishing_warn() {
    let (_dir, root, pkg) = published_pkg("q");
    pkg.begin_publish(2).unwrap();

    // max_age=0 → any existing lock is stale.
    assert!(
        pkg.publish_lock_is_stale(2, Duration::from_secs(0))
            .unwrap()
    );
    let dest = pkg
        .quarantine_publish_lock(2, Duration::from_secs(0))
        .unwrap();
    assert!(dest.exists());
    assert!(!pkg.publishing_dir().join(format_generation_id(2)).exists());

    let report = verify_tet_package(&root).unwrap();
    assert!(report.ok, "{:?}", report.findings);
    assert!(
        report.findings.iter().any(|f| f.check == "publishing"
            && f.detail
                .as_deref()
                .is_some_and(|d| d.contains("quarantine"))),
        "{:?}",
        report.findings
    );
}
