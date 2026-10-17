//! Package store trait: local `.tet.d` + in-memory backends (THI-469).

use crate::catalog::{
    LocalStore, MemoryStore, PackageStore, PublishGeneration, ReadAt, TetDatasetWrite,
    TetWriterSession, format_generation_id, read_tet_summary_v1,
};

fn tiny_tet_bytes() -> Vec<u8> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.tet");
    let mut session = TetWriterSession::create(&path);
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
    std::fs::read(&path).unwrap()
}

fn assert_store_roundtrip<S: PackageStore>(store: &S) {
    store
        .create_dir_exclusive("staging/w1")
        .expect("staging session");
    let packed = tiny_tet_bytes();
    store
        .write_part("staging/w1/part-a.tet", &packed)
        .expect("write part");

    let meta = store.object_meta("staging/w1/part-a.tet").unwrap().unwrap();
    assert_eq!(meta.len, packed.len() as u64);

    let head = store.read_at("staging/w1/part-a.tet", 0, 4).unwrap();
    assert_eq!(head, &packed[..4]);

    let listed = store.list_prefix("staging/").unwrap();
    assert!(
        listed.iter().any(|k| k == "staging/w1/part-a.tet"),
        "list_prefix: {listed:?}"
    );

    let g = format_generation_id(1);
    let pub_key = format!("publishing/{g}");
    store.create_dir_exclusive(&pub_key).unwrap();
    store
        .write_part(&format!("publishing/{g}/packed.tet"), &packed)
        .unwrap();
    store
        .write_part(
            &format!("publishing/{g}/MANIFEST.json"),
            br#"{"generation":1,"protocol_version":2,"part_ids":["a"],"packed_byte_len":1}"#,
        )
        .unwrap();

    store
        .rename_prefix(&format!("publishing/{g}"), &format!("generations/{g}"))
        .unwrap();

    assert!(
        store
            .object_meta(&format!("generations/{g}/packed.tet"))
            .unwrap()
            .is_some()
    );
    assert!(
        store
            .object_meta(&format!("publishing/{g}/packed.tet"))
            .unwrap()
            .is_none()
    );

    store.publish_generation(1, None).unwrap();
    assert_eq!(store.current_generation().unwrap(), Some(1));

    let err = store.publish_generation(2, None).unwrap_err();
    assert!(matches!(err, crate::catalog::CatalogError::InvalidCurrent));

    store.publish_generation(2, Some(1)).unwrap();
    assert_eq!(store.current_generation().unwrap(), Some(2));

    // CAS lost: expected stale.
    let err = store.publish_generation(3, Some(1)).unwrap_err();
    assert!(matches!(err, crate::catalog::CatalogError::InvalidCurrent));
    assert_eq!(store.current_generation().unwrap(), Some(2));

    let sealed = store
        .read_all(&format!("generations/{g}/packed.tet"))
        .unwrap();
    let summary = read_tet_summary_v1(&sealed).unwrap();
    assert_eq!(summary.datasets[0].name, "a");
}

#[test]
fn memory_store_read_write_publish() {
    assert_store_roundtrip(&MemoryStore::new());
}

#[test]
fn local_store_read_write_publish() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("demo.tet.d");
    std::fs::create_dir_all(root.join("staging")).unwrap();
    std::fs::create_dir_all(root.join("generations")).unwrap();
    std::fs::create_dir_all(root.join("publishing")).unwrap();
    assert_store_roundtrip(&LocalStore::new(root));
}

#[test]
fn tet_package_current_uses_local_store() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = crate::catalog::TetPackage::create(dir.path().join("pkg.tet.d")).unwrap();
    assert_eq!(pkg.current().unwrap(), None);
    assert_eq!(pkg.store().current_generation().unwrap(), None);

    let packed = tiny_tet_bytes();
    let packed_path = dir.path().join("src.tet");
    std::fs::write(&packed_path, &packed).unwrap();
    pkg.begin_publish(1).unwrap();
    pkg.seal_generation(1, &packed_path, vec!["part-a".into()])
        .unwrap();

    assert_eq!(pkg.current().unwrap(), Some(1));
    assert_eq!(pkg.store().current_generation().unwrap(), Some(1));
    let via_store = pkg.store().read_all("CURRENT").unwrap();
    assert_eq!(std::str::from_utf8(&via_store).unwrap().trim(), "1");
}
