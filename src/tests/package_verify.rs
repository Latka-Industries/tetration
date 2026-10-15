//! Package verify / torn-generation reject cases (THI-466).

use std::fs;
use std::time::Duration;

use proptest::prelude::*;

use crate::catalog::{
    PACKAGE_PROTOCOL_VERSION, TetDatasetWrite, TetPackage, TetShardWriter, format_generation_id,
    publish_parts,
};
use crate::verify::{
    VerifyOptions, looks_like_tet_package, verify_tet_package, verify_tet_package_with_options,
};

fn f32_pair(a: f32, b: f32) -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    buf[..4].copy_from_slice(&a.to_le_bytes());
    buf[4..].copy_from_slice(&b.to_le_bytes());
    buf
}

fn seal_one(pkg: &TetPackage, session: &str, part: &str) {
    let mut w = TetShardWriter::create(pkg, session, part).unwrap();
    w.push_dataset(TetDatasetWrite::f32_row_major("a", &[2], &[2], f32_pair(1.0, 2.0)).unwrap())
        .unwrap();
    w.seal().unwrap();
}

#[test]
fn healthy_package_verify_ok() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("ok.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    seal_one(&pkg, "s", "p0");
    let manifest = publish_parts(&pkg, 1, &["p0".into()]).unwrap();
    assert!(manifest.packed_sha256.is_some());

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
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("missing.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    seal_one(&pkg, "s", "p0");
    publish_parts(&pkg, 1, &["p0".into()]).unwrap();

    // Tear CURRENT → G=9 with no generations/00000009.
    fs::write(root.join("CURRENT"), "9\n").unwrap();

    let report = verify_tet_package(&root).unwrap();
    assert!(!report.ok);
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "current_packed" && !f.ok),
        "{:?}",
        report.findings
    );
}

#[test]
fn manifest_byte_len_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("len.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    seal_one(&pkg, "s", "p0");
    publish_parts(&pkg, 1, &["p0".into()]).unwrap();

    let mut manifest = pkg.read_manifest(1).unwrap();
    manifest.packed_byte_len = 1;
    let path = pkg.manifest_path(1);
    fs::write(
        &path,
        serde_json::to_string_pretty(&manifest).unwrap() + "\n",
    )
    .unwrap();

    let report = verify_tet_package(&root).unwrap();
    assert!(!report.ok);
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "manifest_byte_len" && !f.ok),
        "{:?}",
        report.findings
    );
}

#[test]
fn manifest_sha256_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("hash.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    seal_one(&pkg, "s", "p0");
    publish_parts(&pkg, 1, &["p0".into()]).unwrap();

    let mut manifest = pkg.read_manifest(1).unwrap();
    manifest.packed_sha256 = Some("0".repeat(64));
    fs::write(
        pkg.manifest_path(1),
        serde_json::to_string_pretty(&manifest).unwrap() + "\n",
    )
    .unwrap();

    let report = verify_tet_package(&root).unwrap();
    assert!(!report.ok);
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "manifest_sha256" && !f.ok),
        "{:?}",
        report.findings
    );
}

#[test]
fn incomplete_publish_warns() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("inc.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    seal_one(&pkg, "s", "p0");
    publish_parts(&pkg, 1, &["p0".into()]).unwrap();
    pkg.begin_publish(2).unwrap();

    let report = verify_tet_package(&root).unwrap();
    assert!(report.ok, "incomplete publish must not fail CURRENT");
    assert!(
        report.findings.iter().any(|f| {
            f.check == "publishing_incomplete"
                && f.severity == crate::verify::VerifySeverity::Warning
        }),
        "{:?}",
        report.findings
    );
}

#[test]
fn bad_protocol_version_fails() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("proto.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    let _ = pkg;
    fs::write(
        root.join("PROTOCOL"),
        format!(
            "{{\"protocol_version\":{}}}\n",
            PACKAGE_PROTOCOL_VERSION + 1
        ),
    )
    .unwrap();

    let report = verify_tet_package(&root).unwrap();
    assert!(!report.ok);
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "protocol" && !f.ok),
        "{:?}",
        report.findings
    );
}

#[test]
fn truncate_packed_fails_layout() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("torn.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    seal_one(&pkg, "s", "p0");
    publish_parts(&pkg, 1, &["p0".into()]).unwrap();

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
    fs::write(
        pkg.manifest_path(1),
        serde_json::to_string_pretty(&manifest).unwrap() + "\n",
    )
    .unwrap();

    let report = verify_tet_package_with_options(&root, VerifyOptions::default()).unwrap();
    assert!(!report.ok);
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "packed_layout" && !f.ok),
        "{:?}",
        report.findings
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Corrupt CURRENT text must not panic; verify fails or reports invalid current.
    #[test]
    fn fuzz_current_pointer_text(s in ".*") {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("fuzz.tet.d");
        let pkg = TetPackage::create(&root).unwrap();
        seal_one(&pkg, "s", "p0");
        publish_parts(&pkg, 1, &["p0".into()]).unwrap();
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
fn quarantine_stale_lock_clears_publishing_warn() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("q.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    seal_one(&pkg, "s", "p0");
    publish_parts(&pkg, 1, &["p0".into()]).unwrap();
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
