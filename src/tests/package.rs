//! `.tet.d` package skeleton (THI-468 / writer protocol v2).

use std::time::Duration;

use crate::catalog::{
    CatalogError, PACKAGE_PROTOCOL_VERSION, PackGenerationOptions, TetDatasetWrite, TetPackage,
    TetWriterSession,
};
use crate::verify::verify_tet_file;

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
fn pin_current_without_generation() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("empty.tet.d")).unwrap();
    let err = pkg.pin_current().unwrap_err();
    assert!(matches!(err, CatalogError::NoCurrentGeneration), "{err:?}");
}

#[test]
fn abort_publish_refuses_current() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("cur.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    let src = dir.path().join("src.tet");
    tiny_tet(&src);
    pkg.begin_publish(1).unwrap();
    pkg.seal_generation(1, &src, vec!["a".into()]).unwrap();
    assert_eq!(pkg.current().unwrap(), Some(1));
    let err = pkg.abort_publish(1).unwrap_err();
    assert!(
        matches!(err, crate::catalog::CatalogError::Package(msg) if msg.contains("CURRENT")),
        "{err:?}"
    );
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
fn pin_survives_current_advance() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("pin.tet.d");
    let pkg = TetPackage::create(&root).unwrap();

    let g1 = dir.path().join("g1.tet");
    let g2 = dir.path().join("g2.tet");
    tiny_tet(&g1);
    tiny_tet(&g2);

    pkg.begin_publish(1).unwrap();
    pkg.seal_generation(1, &g1, vec!["a".into()]).unwrap();
    let pin = pkg.pin_current().unwrap();
    assert_eq!(pin.generation(), 1);

    pkg.begin_publish(2).unwrap();
    pkg.seal_generation(2, &g2, vec!["b".into()]).unwrap();
    assert_eq!(pkg.current().unwrap(), Some(2));

    // Pin still maps generation 1 even though CURRENT advanced.
    assert_eq!(pin.generation(), 1);
    assert_eq!(pin.packed_tet_path(), pkg.packed_tet_path(1));
    let file = pin.open().unwrap();
    assert_eq!(file.path(), pkg.packed_tet_path(1));
    let summary = file.summary().unwrap();
    assert_eq!(summary.datasets[0].name, "a");

    // Fresh open_current follows CURRENT.
    assert_eq!(pkg.open_current().unwrap().path(), pkg.packed_tet_path(2));
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

#[test]
fn pack_generation_keeps_package_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("keep.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    let src = dir.path().join("src.tet");
    tiny_tet(&src);
    pkg.begin_publish(1).unwrap();
    pkg.seal_generation(1, &src, vec!["a".into()]).unwrap();

    let dest = dir.path().join("out.tet");
    pkg.pack_generation(1, &dest).unwrap();
    assert!(dest.is_file());
    assert!(root.is_dir());
    assert!(verify_tet_file(&dest).unwrap().ok);
}

#[test]
fn pack_generation_remove_package_after_verify() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("rm.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    let src = dir.path().join("src.tet");
    tiny_tet(&src);
    pkg.begin_publish(1).unwrap();
    pkg.seal_generation(1, &src, vec!["a".into()]).unwrap();

    let dest = dir.path().join("out.tet");
    pkg.pack_generation_with(
        1,
        &dest,
        &PackGenerationOptions {
            remove_package: true,
        },
    )
    .unwrap();
    assert!(dest.is_file());
    assert!(!root.exists());
    assert!(verify_tet_file(&dest).unwrap().ok);
}

#[test]
fn pack_remove_package_refuses_dest_inside_package() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("inside.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    let src = dir.path().join("src.tet");
    tiny_tet(&src);
    pkg.begin_publish(1).unwrap();
    pkg.seal_generation(1, &src, vec!["a".into()]).unwrap();

    let dest = root.join("nested.tet");
    let err = pkg
        .pack_generation_with(
            1,
            &dest,
            &PackGenerationOptions {
                remove_package: true,
            },
        )
        .unwrap_err();
    assert!(
        matches!(err, crate::catalog::CatalogError::Package(msg) if msg.contains("inside the package")),
        "unexpected err: {err:?}"
    );
    assert!(root.is_dir());
    assert!(dest.is_file());
}
