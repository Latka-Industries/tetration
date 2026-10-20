//! Publish sealed staging parts into a generation (THI-464, THI-480).

use std::fs;
use std::path::PathBuf;

use crate::catalog::{
    PublishPartsOptions, TetDatasetWrite, TetPackage, TetShardWriter, dry_run_publish_parts,
    format_generation_id, list_sealed_parts, publish_parts, publish_parts_with_options,
};
use crate::verify::{VerifySeverity, verify_tet_package};

fn f32_le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn seal_named(pkg: &TetPackage, session: &str, part: &str, name: &str, vals: &[f32]) {
    seal_shaped(pkg, session, part, name, &[2], &[2], vals);
}

fn seal_shaped(
    pkg: &TetPackage,
    session: &str,
    part: &str,
    name: &str,
    shape: &[u64],
    chunk_shape: &[u64],
    vals: &[f32],
) {
    let mut w = TetShardWriter::create(pkg, session, part).unwrap();
    w.push_dataset(TetDatasetWrite::f32_row_major(name, shape, chunk_shape, f32_le(vals)).unwrap())
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
    let mut zone = crate::catalog::read_zone_maps_file(
        &pkg.generation_dir(1).join(crate::catalog::ZONE_MAPS_FILE),
    )
    .unwrap();
    zone.generation = 2;
    let zone_bytes = crate::catalog::encode_zone_maps(&zone).unwrap();
    fs::write(g2.join(crate::catalog::ZONE_MAPS_FILE), &zone_bytes).unwrap();
    let mut index = crate::catalog::read_secondary_index_file(
        &pkg.generation_dir(1)
            .join(crate::catalog::SECONDARY_INDEX_FILE),
    )
    .unwrap();
    index.generation = 2;
    let index_bytes = crate::catalog::encode_secondary_index(&index).unwrap();
    fs::write(g2.join(crate::catalog::SECONDARY_INDEX_FILE), &index_bytes).unwrap();
    let mut manifest = pkg.read_manifest(1).unwrap();
    manifest.generation = 2;
    manifest.zone_maps_sha256 = Some(crate::catalog::zone::sha256_hex(&zone_bytes));
    manifest.secondary_indexes_sha256 = Some(crate::catalog::zone::sha256_hex(&index_bytes));
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

    // Same claim ranges, different payload bytes → reject.
    seal_named(&pkg, "s0", "a", "same", &[1.0, 2.0]);
    seal_named(&pkg, "s1", "b", "same", &[3.0, 4.0]);
    assert_eq!(list_sealed_parts(&pkg).unwrap().len(), 2);

    let err = publish_parts(&pkg, 1, &["a".into(), "b".into()]).unwrap_err();
    match err {
        crate::catalog::CatalogError::ClaimOverlap(overlap) => {
            assert_eq!(overlap.dataset, "same");
            assert_eq!(overlap.part_a, "a");
            assert_eq!(overlap.part_b, "b");
            assert_eq!(overlap.ranges_a, "full_grid");
            assert_eq!(overlap.ranges_b, "full_grid");
            let msg = overlap.to_string();
            assert!(msg.contains("dataset=same"), "{msg}");
            assert!(msg.contains("parts=a + b"), "{msg}");
            assert!(msg.contains("full_grid"), "{msg}");
        }
        other => panic!("expected ClaimOverlap, got {other:?}"),
    }
    assert_eq!(pkg.current().unwrap(), None);
}

#[test]
fn identical_bytes_overlap_allows_publish() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("idempotent.tet.d")).unwrap();

    // Same claim + matching chunk bytes (part wrappers may differ; chunk compare allows).
    seal_named(&pkg, "s0", "a", "same", &[1.0, 2.0]);
    seal_named(&pkg, "s1", "b", "same", &[1.0, 2.0]);

    let manifest = publish_parts(&pkg, 1, &["a".into(), "b".into()]).unwrap();
    assert_eq!(manifest.generation, 1);
    assert_eq!(pkg.current().unwrap(), Some(1));
    let file = pkg.open_current().unwrap();
    let summary = file.summary().unwrap();
    assert_eq!(summary.datasets.len(), 1);
    assert_eq!(summary.datasets[0].name, "same");
}

#[test]
fn identical_file_reingest_overlap_allows_via_part_sha256() {
    use crate::catalog::ingest_tet_file;

    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src.tet");
    {
        let mut s = crate::catalog::TetWriterSession::create(&src);
        s.push_dataset(
            TetDatasetWrite::f32_row_major("temp", &[2], &[2], f32_le(&[1.0, 2.0])).unwrap(),
        )
        .unwrap();
        s.commit().unwrap();
    }
    let pkg = TetPackage::create(dir.path().join("reingest.tet.d")).unwrap();
    ingest_tet_file(&pkg, &src, "s0", "a").unwrap();
    ingest_tet_file(&pkg, &src, "s1", "b").unwrap();
    let sealed = list_sealed_parts(&pkg).unwrap();
    assert_eq!(sealed[0].1.part_sha256, sealed[1].1.part_sha256);

    let manifest = publish_parts(&pkg, 1, &["a".into(), "b".into()]).unwrap();
    assert_eq!(manifest.generation, 1);
    assert_eq!(
        pkg.open_current()
            .unwrap()
            .summary()
            .unwrap()
            .datasets
            .len(),
        1
    );
}

#[test]
fn dry_run_publish_ok_leaves_current_unset() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("dry-ok.tet.d")).unwrap();
    seal_named(&pkg, "s0", "p0", "temp", &[1.0, 2.0]);
    seal_named(&pkg, "s1", "p1", "pressure", &[3.0, 4.0]);

    let validation = dry_run_publish_parts(&pkg, 1, &["p0".into(), "p1".into()]).unwrap();
    assert_eq!(validation.generation, 1);
    assert_eq!(validation.part_ids, vec!["p0", "p1"]);
    assert_eq!(pkg.current().unwrap(), None);
    assert!(!pkg.generation_dir(1).exists());
    assert!(!pkg.publishing_dir().join(format_generation_id(1)).exists());
}

#[test]
fn dry_run_publish_rejects_overlap_without_sealing() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("dry-overlap.tet.d")).unwrap();
    seal_named(&pkg, "s0", "a", "same", &[1.0, 2.0]);
    seal_named(&pkg, "s1", "b", "same", &[9.0, 8.0]);

    let err = dry_run_publish_parts(&pkg, 1, &["a".into(), "b".into()]).unwrap_err();
    assert!(matches!(err, crate::catalog::CatalogError::ClaimOverlap(_)));
    assert_eq!(pkg.current().unwrap(), None);
    assert!(!pkg.generation_dir(1).exists());
}

#[test]
fn schema_change_vs_current_rejects_without_replace() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("schema.tet.d")).unwrap();
    seal_named(&pkg, "s0", "p0", "temp", &[1.0, 2.0]);
    publish_parts(&pkg, 1, &["p0".into()]).unwrap();
    assert_eq!(pkg.current().unwrap(), Some(1));

    seal_shaped(&pkg, "s1", "p1", "temp", &[4], &[2], &[1.0, 2.0, 3.0, 4.0]);
    let err = publish_parts(&pkg, 2, &["p1".into()]).unwrap_err();
    match err {
        crate::catalog::CatalogError::SchemaConflict(c) => {
            assert_eq!(c.dataset, "temp");
            assert_eq!(c.prior_shape, vec![2]);
            assert_eq!(c.new_shape, vec![4]);
            let msg = c.to_string();
            assert!(msg.contains("--replace temp"), "{msg}");
        }
        other => panic!("expected SchemaConflict, got {other:?}"),
    }
    assert_eq!(pkg.current().unwrap(), Some(1));
    assert!(!pkg.generation_dir(2).exists());
}

#[test]
fn schema_change_with_replace_publishes_g_plus_one() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("replace.tet.d")).unwrap();
    seal_named(&pkg, "s0", "p0", "temp", &[1.0, 2.0]);
    publish_parts(&pkg, 1, &["p0".into()]).unwrap();
    let g1 = pkg.open_current().unwrap().summary().unwrap();
    assert_eq!(g1.datasets[0].shape, vec![2]);

    seal_shaped(&pkg, "s1", "p1", "temp", &[4], &[2], &[1.0, 2.0, 3.0, 4.0]);
    let opts = PublishPartsOptions {
        replace: vec!["temp".into()],
    };
    let manifest = publish_parts_with_options(&pkg, 2, &["p1".into()], &opts).unwrap();
    assert_eq!(manifest.generation, 2);
    assert_eq!(pkg.current().unwrap(), Some(2));
    // Sealed G=1 untouched.
    let still_g1 = pkg.open_generation(1).unwrap().summary().unwrap();
    assert_eq!(still_g1.datasets[0].shape, vec![2]);
    let g2 = pkg.open_current().unwrap().summary().unwrap();
    assert_eq!(g2.datasets[0].shape, vec![4]);
}
