//! Publish sealed staging parts into a generation (THI-464, THI-480).

use std::fs;
use std::path::PathBuf;

use crate::catalog::{
    TetDatasetWrite, TetPackage, TetShardWriter, format_generation_id, list_sealed_parts,
    publish_parts,
};
use crate::verify::{VerifySeverity, verify_tet_package};

fn f32_le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn seal_named(pkg: &TetPackage, session: &str, part: &str, name: &str, vals: &[f32]) {
    let mut w = TetShardWriter::create(pkg, session, part).unwrap();
    w.push_dataset(
        TetDatasetWrite::f32_row_major(name, &[2], &[2], f32_le(vals)).unwrap(),
    )
    .unwrap();
    w.seal().unwrap();
}

/// Create `{stem}.tet.d`, seal dataset `a`, publish generation 1.
fn published_single(stem: &str) -> (tempfile::TempDir, PathBuf, TetPackage) {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join(format!("{stem}.tet.d"));
    let pkg = TetPackage::create(&root).unwrap();
    seal_named(&pkg, "s", "only", "a", &[1.0, 2.0]);
    publish_parts(&pkg, 1, &["only".into()]).unwrap();
    (dir, root, pkg)
}

fn assert_current_is_a(pkg: &TetPackage) {
    assert_eq!(pkg.current().unwrap(), Some(1));
    let still = pkg.open_current().unwrap();
    assert_eq!(still.summary().unwrap().datasets[0].name, "a");
}

#[test]
fn n_shards_publish_matches_union_of_datasets() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("pub.tet.d")).unwrap();

    seal_named(&pkg, "s0", "p0", "temp", &[1.0, 2.0]);
    seal_named(&pkg, "s1", "p1", "pressure", &[3.0, 4.0]);

    let manifest = publish_parts(&pkg, 1, &["p0".into(), "p1".into()]).unwrap();
    assert_eq!(manifest.generation, 1);
    assert_eq!(manifest.part_ids, vec!["p0", "p1"]);
    assert!(manifest.packed_sha256.is_some());
    assert_eq!(pkg.current().unwrap(), Some(1));

    let file = pkg.open_current().unwrap();
    let summary = file.summary().unwrap();
    let names: Vec<_> = summary.datasets.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, vec!["temp", "pressure"]);
    assert!(summary.chunks.len() >= 2);
}

#[test]
fn crash_mid_publish_leaves_prior_generation() {
    let (_dir, _root, pkg) = published_single("crash");
    assert_current_is_a(&pkg);

    // Simulate crashed publish of G=2: lock held, CURRENT still 1.
    pkg.begin_publish(2).unwrap();
    assert_current_is_a(&pkg);
    pkg.abort_publish(2).unwrap();
}

#[test]
fn crash_mid_seal_torn_publishing_prior_readable() {
    let (_dir, root, pkg) = published_single("torn");

    // Crash after begin_publish + partial packed copy into publishing/<G>/.
    let lock = pkg.begin_publish(2).unwrap();
    fs::write(lock.join("packed.tet"), b"torn").unwrap();

    assert_current_is_a(&pkg);
    // Incomplete G is never reader-visible via CURRENT.
    assert!(pkg.open_generation(2).is_err());

    let report = verify_tet_package(&root).unwrap();
    assert!(
        report.ok,
        "torn publishing must not fail CURRENT: {:?}",
        report.findings
    );
    assert!(
        report.findings.iter().any(|f| {
            f.check == "publishing_incomplete" && f.severity == VerifySeverity::Warning
        }),
        "{:?}",
        report.findings
    );

    pkg.abort_publish(2).unwrap();
}

#[test]
fn crash_after_rename_before_current_flip() {
    let (_dir, root, pkg) = published_single("rename");

    // Simulate seal_generation after rename into generations/ but before set_current:
    // G=2 tree exists, CURRENT still points at G=1.
    let g2 = pkg.generation_dir(2);
    fs::create_dir_all(&g2).unwrap();
    fs::copy(pkg.packed_tet_path(1), g2.join("packed.tet")).unwrap();
    let mut manifest = pkg.read_manifest(1).unwrap();
    manifest.generation = 2;
    fs::write(
        g2.join("MANIFEST.json"),
        serde_json::to_string_pretty(&manifest).unwrap() + "\n",
    )
    .unwrap();
    assert!(!pkg.publishing_dir().join(format_generation_id(2)).exists());

    assert_current_is_a(&pkg);
    // G=2 is on disk but not CURRENT — readers pin CURRENT only.
    assert_eq!(pkg.open_current().unwrap().path(), pkg.packed_tet_path(1));

    let report = verify_tet_package(&root).unwrap();
    assert!(report.ok, "{:?}", report.findings);
}

#[test]
fn overlapping_claims_reject_publish() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("overlap.tet.d")).unwrap();

    for (session, part) in [("s0", "a"), ("s1", "b")] {
        seal_named(&pkg, session, part, "same", &[1.0, 2.0]);
    }
    assert_eq!(list_sealed_parts(&pkg).unwrap().len(), 2);

    let err = publish_parts(&pkg, 1, &["a".into(), "b".into()]).unwrap_err();
    assert!(matches!(
        err,
        crate::catalog::CatalogError::Package("overlapping dataset claims in publish set")
    ));
    assert_eq!(pkg.current().unwrap(), None);
}
