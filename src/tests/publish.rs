//! Publish sealed staging parts into a generation (THI-464).

use crate::catalog::{
    TetDatasetWrite, TetPackage, TetShardWriter, list_sealed_parts, publish_parts,
};

fn f32_pair(a: f32, b: f32) -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    buf[..4].copy_from_slice(&a.to_le_bytes());
    buf[4..].copy_from_slice(&b.to_le_bytes());
    buf
}

#[test]
fn n_shards_publish_matches_union_of_datasets() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("pub.tet.d")).unwrap();

    let mut w0 = TetShardWriter::create(&pkg, "s0", "p0").unwrap();
    w0.push_dataset(
        TetDatasetWrite::f32_row_major("temp", &[2], &[2], f32_pair(1.0, 2.0)).unwrap(),
    )
    .unwrap();
    w0.seal().unwrap();

    let mut w1 = TetShardWriter::create(&pkg, "s1", "p1").unwrap();
    w1.push_dataset(
        TetDatasetWrite::f32_row_major("pressure", &[2], &[2], f32_pair(3.0, 4.0)).unwrap(),
    )
    .unwrap();
    w1.seal().unwrap();

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
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("crash.tet.d")).unwrap();

    let mut w = TetShardWriter::create(&pkg, "s", "only").unwrap();
    w.push_dataset(TetDatasetWrite::f32_row_major("a", &[2], &[2], f32_pair(1.0, 2.0)).unwrap())
        .unwrap();
    w.seal().unwrap();
    publish_parts(&pkg, 1, &["only".into()]).unwrap();
    assert_eq!(pkg.current().unwrap(), Some(1));

    // Simulate crashed publish of G=2: lock held, CURRENT still 1.
    pkg.begin_publish(2).unwrap();
    assert_eq!(pkg.current().unwrap(), Some(1));
    let still = pkg.open_current().unwrap();
    assert_eq!(still.summary().unwrap().datasets[0].name, "a");

    pkg.abort_publish(2).unwrap();
}

#[test]
fn overlapping_claims_reject_publish() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("overlap.tet.d")).unwrap();

    for (session, part) in [("s0", "a"), ("s1", "b")] {
        let mut w = TetShardWriter::create(&pkg, session, part).unwrap();
        w.push_dataset(
            TetDatasetWrite::f32_row_major("same", &[2], &[2], f32_pair(1.0, 2.0)).unwrap(),
        )
        .unwrap();
        w.seal().unwrap();
    }
    assert_eq!(list_sealed_parts(&pkg).unwrap().len(), 2);

    let err = publish_parts(&pkg, 1, &["a".into(), "b".into()]).unwrap_err();
    assert!(matches!(
        err,
        crate::catalog::CatalogError::Package("overlapping dataset claims in publish set")
    ));
    assert_eq!(pkg.current().unwrap(), None);
}
