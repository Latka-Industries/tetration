//! `.tet.d` package skeleton (THI-468 / writer protocol v2).

use std::time::Duration;

use crate::catalog::{PACKAGE_PROTOCOL_VERSION, TetDatasetWrite, TetPackage, TetWriterSession};

fn tiny_tet(path: &std::path::Path) {
    let mut session = TetWriterSession::create(path);
    let data = {
        let mut buf = vec![0u8; 8];
        buf[..4].copy_from_slice(&1.0_f32.to_le_bytes());
        buf[4..].copy_from_slice(&2.0_f32.to_le_bytes());
        buf
    };
    session
        .push_dataset(TetDatasetWrite::f32_row_major("a", &[2], &[2], data).unwrap())
        .unwrap();
    session.commit().unwrap();
}

#[test]
fn create_open_empty_package() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("demo.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    assert_eq!(pkg.protocol().protocol_version, PACKAGE_PROTOCOL_VERSION);
    assert!(pkg.staging_dir().is_dir());
    assert!(pkg.generations_dir().is_dir());
    assert!(pkg.publishing_dir().is_dir());
    assert_eq!(pkg.current().unwrap(), None);

    let again = TetPackage::open(&root).unwrap();
    assert_eq!(again.protocol().protocol_version, PACKAGE_PROTOCOL_VERSION);
}

#[test]
fn seal_generation_and_open_current() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("pkg.tet.d");
    let pkg = TetPackage::create(&root).unwrap();

    let packed_src = dir.path().join("src.tet");
    tiny_tet(&packed_src);

    pkg.begin_publish(1).unwrap();
    pkg.seal_generation(1, &packed_src, vec!["part-a".into()])
        .unwrap();

    assert_eq!(pkg.current().unwrap(), Some(1));
    assert!(pkg.packed_tet_path(1).is_file());
    let manifest = pkg.read_manifest(1).unwrap();
    assert_eq!(manifest.generation, 1);
    assert_eq!(manifest.part_ids, vec!["part-a"]);
    assert!(manifest.packed_byte_len > 0);

    let file = pkg.open_current().unwrap();
    let summary = file.summary().unwrap();
    assert_eq!(summary.datasets.len(), 1);
    assert_eq!(summary.datasets[0].name, "a");
}

#[test]
fn publish_lock_exclusive() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("lock.tet.d")).unwrap();
    pkg.begin_publish(3).unwrap();
    let err = pkg.begin_publish(3).unwrap_err();
    assert!(matches!(
        err,
        crate::catalog::CatalogError::PublishLockHeld(3)
    ));
    pkg.abort_publish(3).unwrap();
    pkg.begin_publish(3).unwrap();
}

#[test]
fn quarantine_requires_stale() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("q.tet.d")).unwrap();
    pkg.begin_publish(9).unwrap();
    let err = pkg
        .quarantine_publish_lock(9, Duration::from_secs(3600))
        .unwrap_err();
    assert!(matches!(
        err,
        crate::catalog::CatalogError::PublishLockHeld(9)
    ));
    // Force stale by age 0.
    let q = pkg.quarantine_publish_lock(9, Duration::ZERO).unwrap();
    assert!(q.exists());
    assert!(
        !pkg.publishing_dir()
            .join(crate::catalog::format_generation_id(9))
            .exists()
    );
}

#[test]
fn staging_session_collision() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("st.tet.d")).unwrap();
    pkg.create_staging_session("w1").unwrap();
    let err = pkg.create_staging_session("w1").unwrap_err();
    assert!(matches!(
        err,
        crate::catalog::CatalogError::PackagePathExists(_)
    ));
}
