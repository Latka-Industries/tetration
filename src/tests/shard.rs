//! Shard writer staging parts (THI-463).

use crate::catalog::{
    DatasetClaim, TetDatasetWrite, TetPackage, TetShardWriter, find_claim_overlap,
    list_sealed_parts, open_sealed_part,
};

fn f32_pair() -> Vec<u8> {
    let mut buf = vec![0u8; 8];
    buf[..4].copy_from_slice(&1.0_f32.to_le_bytes());
    buf[4..].copy_from_slice(&2.0_f32.to_le_bytes());
    buf
}

#[test]
fn n_shard_writers_produce_readable_parts() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("multi.tet.d")).unwrap();

    let mut a = TetShardWriter::create(&pkg, "w-a", "a").unwrap();
    a.push_dataset(TetDatasetWrite::f32_row_major("temp", &[2], &[2], f32_pair()).unwrap())
        .unwrap();
    let meta_a = a.seal().unwrap();
    assert!(meta_a.sealed);
    assert_eq!(meta_a.part_id, "a");
    assert_eq!(meta_a.claims.len(), 1);
    assert!(!meta_a.part_sha256.is_empty());

    let mut b = TetShardWriter::create(&pkg, "w-b", "b").unwrap();
    b.push_dataset(TetDatasetWrite::f32_row_major("pressure", &[2], &[2], f32_pair()).unwrap())
        .unwrap();
    let meta_b = b.seal().unwrap();

    let sealed = list_sealed_parts(&pkg).unwrap();
    assert_eq!(sealed.len(), 2);
    assert_eq!(sealed[0].1.part_id, "a");
    assert_eq!(sealed[1].1.part_id, "b");
    assert!(find_claim_overlap(&sealed).is_none());

    let file_a = open_sealed_part(&sealed[0].0, &meta_a).unwrap();
    let summary_a = file_a.summary().unwrap();
    assert_eq!(summary_a.datasets[0].name, "temp");

    let file_b = open_sealed_part(&sealed[1].0, &meta_b).unwrap();
    assert_eq!(file_b.summary().unwrap().datasets[0].name, "pressure");

    // Staging only — no CURRENT yet.
    assert_eq!(pkg.current().unwrap(), None);
}

#[test]
fn overlapping_full_grid_claims_detected() {
    let a = DatasetClaim::full_grid("x", 1, &[4], &[2]);
    let b = DatasetClaim::full_grid("x", 1, &[4], &[2]);
    let c = DatasetClaim::full_grid("y", 1, &[4], &[2]);
    assert!(a.overlaps(&b));
    assert!(!a.overlaps(&c));
}

#[test]
fn seal_requires_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("empty.tet.d")).unwrap();
    let w = TetShardWriter::create(&pkg, "w", "p").unwrap();
    let err = w.seal().unwrap_err();
    assert!(matches!(
        err,
        crate::catalog::CatalogError::Package("shard requires at least one dataset")
    ));
}

#[test]
fn incomplete_session_skipped_by_list() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("partial.tet.d")).unwrap();
    let mut w = TetShardWriter::create(&pkg, "w", "p").unwrap();
    w.push_dataset(TetDatasetWrite::f32_row_major("a", &[2], &[2], f32_pair()).unwrap())
        .unwrap();
    // Drop without seal.
    drop(w);
    assert!(list_sealed_parts(&pkg).unwrap().is_empty());
}
