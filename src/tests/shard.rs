//! Shard writer staging parts (THI-463).

use std::fs;
use std::path::PathBuf;

use crate::catalog::package::{PACKAGE_PROTOCOL_VERSION, sha256_file_hex, write_json_atomic};
use crate::catalog::{
    ClaimOverlap, DATASET_DTYPE_TAG_V1, DatasetClaim, PartMeta, TetDatasetWrite, TetPackage,
    TetShardWriter, find_claim_overlap, list_sealed_parts, open_sealed_part, publish_parts,
    write_sparse_raw_array_file,
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
    assert_eq!(a.ranges_display(), "full_grid");
}

#[test]
fn find_claim_overlap_reports_part_ids_and_ranges() {
    let left = DatasetClaim {
        name: "temp".into(),
        dtype: 1,
        shape: vec![8],
        chunk_shape: vec![2],
        full_grid: false,
        chunk_ranges: vec![[0, 3]],
    };
    let right = DatasetClaim {
        name: "temp".into(),
        dtype: 1,
        shape: vec![8],
        chunk_shape: vec![2],
        full_grid: false,
        chunk_ranges: vec![[2, 4]],
    };
    assert!(left.overlaps(&right));
    assert_eq!(left.ranges_display(), "chunk_ranges=[[0,3)]");
    assert_eq!(right.ranges_display(), "chunk_ranges=[[2,4)]");

    let parts = vec![
        (
            PathBuf::from("staging/s0"),
            PartMeta {
                part_id: "p0".into(),
                writer_session_id: "s0".into(),
                protocol_version: 2,
                sealed: true,
                part_byte_len: 1,
                part_sha256: "aa".into(),
                claims: vec![left],
            },
        ),
        (
            PathBuf::from("staging/s1"),
            PartMeta {
                part_id: "p1".into(),
                writer_session_id: "s1".into(),
                protocol_version: 2,
                sealed: true,
                part_byte_len: 1,
                part_sha256: "bb".into(),
                claims: vec![right],
            },
        ),
    ];
    let overlap = find_claim_overlap(&parts).expect("overlap");
    assert_eq!(
        overlap,
        ClaimOverlap {
            part_a: "p0".into(),
            part_b: "p1".into(),
            dataset: "temp".into(),
            ranges_a: "chunk_ranges=[[0,3)]".into(),
            ranges_b: "chunk_ranges=[[2,4)]".into(),
        }
    );
    let msg = overlap.to_string();
    assert!(
        msg.contains("dataset=temp") && msg.contains("parts=p0 + p1") && msg.contains("[0,3)"),
        "{msg}"
    );
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

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn stage_sparse_claim(
    pkg: &TetPackage,
    session: &str,
    part: &str,
    claim: DatasetClaim,
    tiles: &[(u64, &[u8])],
) {
    let session_dir = pkg.create_staging_session(session).unwrap();
    let part_path = session_dir.join(format!("part-{part}.tet"));
    write_sparse_raw_array_file(
        &part_path,
        &claim.name,
        claim.dtype,
        &claim.shape,
        &claim.chunk_shape,
        tiles,
    )
    .unwrap();
    let part_byte_len = fs::metadata(&part_path).unwrap().len();
    let part_sha256 = sha256_file_hex(&part_path).unwrap();
    let meta = PartMeta {
        part_id: part.into(),
        writer_session_id: session.into(),
        protocol_version: PACKAGE_PROTOCOL_VERSION,
        sealed: true,
        part_byte_len,
        part_sha256,
        claims: vec![claim],
    };
    write_json_atomic(&session_dir.join("PART_META.json"), &meta).unwrap();
}

#[test]
fn disjoint_chunk_range_claims_publish() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("tile.tet.d")).unwrap();

    // Global shape [4] / chunk [2] → two chunks. Left owns [0,1), right owns [1,2).
    let left = DatasetClaim::ranges("temp", DATASET_DTYPE_TAG_V1.f32, &[4], &[2], vec![[0, 1]]);
    let right = DatasetClaim::ranges("temp", DATASET_DTYPE_TAG_V1.f32, &[4], &[2], vec![[1, 2]]);
    assert!(!left.overlaps(&right));

    let left_tile = f32_bytes(&[1.0, 2.0]);
    let right_tile = f32_bytes(&[3.0, 4.0]);
    stage_sparse_claim(&pkg, "s0", "left", left, &[(0, left_tile.as_slice())]);
    stage_sparse_claim(&pkg, "s1", "right", right, &[(1, right_tile.as_slice())]);

    let sealed = list_sealed_parts(&pkg).unwrap();
    assert!(find_claim_overlap(&sealed).is_none());

    let manifest = publish_parts(&pkg, 1, &["left".into(), "right".into()]).unwrap();
    assert_eq!(manifest.generation, 1);
    let summary = pkg.open_current().unwrap().summary().unwrap();
    assert_eq!(summary.datasets.len(), 1);
    assert_eq!(summary.datasets[0].name, "temp");
    assert_eq!(summary.datasets[0].shape, vec![4]);
    assert_eq!(summary.chunks.len(), 2);
}

#[test]
fn push_dataset_with_claim_rejects_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("claim-mismatch.tet.d")).unwrap();
    let mut w = TetShardWriter::create(&pkg, "w", "p").unwrap();
    let ds = TetDatasetWrite::f32_row_major("temp", &[2], &[2], f32_pair()).unwrap();
    let bad = DatasetClaim::ranges("other", DATASET_DTYPE_TAG_V1.f32, &[2], &[2], vec![[0, 1]]);
    let err = w.push_dataset_with_claim(ds, bad).unwrap_err();
    assert!(matches!(
        err,
        crate::catalog::CatalogError::Package(
            "dataset claim must match queued dataset name/dtype/shape/chunk_shape"
        )
    ));
}

#[test]
fn ranges_display_and_constructor() {
    let c = DatasetClaim::ranges("x", 1, &[8], &[2], vec![[0, 2]]);
    assert!(!c.full_grid);
    assert_eq!(c.ranges_display(), "chunk_ranges=[[0,2)]");
}
