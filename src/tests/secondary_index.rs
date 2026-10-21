//! Secondary indexes written at compact/publish (THI-571 / THI-613).

use std::collections::BTreeMap;
use std::fs;

use serde_json::json;

use crate::catalog::store::ReadAt;
use crate::catalog::{
    CoordAxisV1, DATASET_DTYPE_TAG_V1, GenerationPin, MemoryStore, SECONDARY_INDEX_FILE,
    SECONDARY_INDEX_OMITTED, SECONDARY_INDEX_PRESENT, SecondaryIndexEntry, TetDatasetWrite,
    TetPackage, TetShardWriter, VALUE_POSTINGS_MAX_CARDINALITY, publish_staged_parts,
    read_secondary_index_file, secondary_index_key, stage_part,
};
use crate::verify::{
    VerifyOptions, VerifySeverity, verify_tet_package, verify_tet_package_with_options,
};

fn f32_le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn publish_f32(pkg: &TetPackage, generation: u64, session: &str, part: &str, vals: &[f32]) {
    let mut w = TetShardWriter::create(pkg, session, part).unwrap();
    w.push_dataset(
        TetDatasetWrite::f32_row_major("a", &[vals.len() as u64], &[2], f32_le(vals)).unwrap(),
    )
    .unwrap();
    w.seal().unwrap();
    crate::catalog::publish_parts(pkg, generation, &[part.to_owned()]).unwrap();
}

fn publish_u8(pkg: &TetPackage, generation: u64, vals: &[u8]) {
    let mut w = TetShardWriter::create(pkg, "s", "only").unwrap();
    w.push_dataset(
        TetDatasetWrite::row_major(
            "a",
            DATASET_DTYPE_TAG_V1.u8,
            &[vals.len() as u64],
            &[vals.len() as u64],
            vals.to_vec(),
        )
        .unwrap(),
    )
    .unwrap();
    w.seal().unwrap();
    crate::catalog::publish_parts(pkg, generation, &["only".to_owned()]).unwrap();
}

#[test]
fn publish_writes_index_without_touching_staging() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("idx.tet.d")).unwrap();
    publish_f32(&pkg, 1, "s", "only", &[1.0, 2.0, 1.0, 3.0]);

    let staging = pkg.root().join("staging");
    for entry in fs::read_dir(&staging).unwrap() {
        let session = entry.unwrap().path();
        assert!(
            !session.join(SECONDARY_INDEX_FILE).is_file(),
            "staging must not carry secondary indexes"
        );
    }

    let index_path = pkg.generation_dir(1).join(SECONDARY_INDEX_FILE);
    let first = fs::read(&index_path).unwrap();
    let doc = read_secondary_index_file(&index_path).unwrap();
    assert_eq!(doc.generation, 1);
    let SecondaryIndexEntry::ValuePostings { postings, .. } = &doc.indexes[0] else {
        panic!("expected value postings");
    };
    assert_eq!(postings.len(), 3);
    assert_eq!(postings[0].value_le, "00000040");
    assert_eq!(postings[0].chunks, vec![vec![0]]);
    assert_eq!(postings[1].value_le, "00004040");
    assert_eq!(postings[1].chunks, vec![vec![1]]);
    assert_eq!(postings[2].value_le, "0000803f");
    assert_eq!(postings[2].chunks, vec![vec![0], vec![1]]);

    let manifest = pkg.read_manifest(1).unwrap();
    assert_eq!(
        manifest.secondary_indexes.as_deref(),
        Some(SECONDARY_INDEX_PRESENT)
    );
    assert!(manifest.secondary_indexes_sha256.is_some());

    publish_f32(&pkg, 2, "s2", "next", &[9.0, 8.0, 7.0, 6.0]);
    assert_eq!(fs::read(&index_path).unwrap(), first);
}

#[test]
fn high_cardinality_omits_the_index_file() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("wide.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    let n = VALUE_POSTINGS_MAX_CARDINALITY + 1;
    let vals: Vec<u8> = (0..n).map(|i| u8::try_from(i).unwrap()).collect();
    publish_u8(&pkg, 1, &vals);

    assert!(!pkg.generation_dir(1).join(SECONDARY_INDEX_FILE).is_file());
    let manifest = pkg.read_manifest(1).unwrap();
    assert_eq!(
        manifest.secondary_indexes.as_deref(),
        Some(SECONDARY_INDEX_OMITTED)
    );
    assert!(manifest.secondary_indexes_sha256.is_none());

    let report = verify_tet_package(&root).unwrap();
    assert!(report.ok, "{:?}", report.findings);
    assert!(report.findings.iter().any(|f| {
        f.check == "secondary_indexes"
            && f.ok
            && f.detail.as_deref() == Some(SECONDARY_INDEX_OMITTED)
    }));
}

#[test]
fn cardinality_at_the_cap_is_present() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("cap.tet.d")).unwrap();
    let vals: Vec<u8> = (0..VALUE_POSTINGS_MAX_CARDINALITY)
        .map(|i| u8::try_from(i).unwrap())
        .collect();
    publish_u8(&pkg, 1, &vals);
    let doc = read_secondary_index_file(&pkg.generation_dir(1).join(SECONDARY_INDEX_FILE)).unwrap();
    let SecondaryIndexEntry::ValuePostings { postings, .. } = &doc.indexes[0] else {
        panic!("expected value postings");
    };
    assert_eq!(postings.len(), VALUE_POSTINGS_MAX_CARDINALITY);
}

#[test]
fn tamper_and_missing_index_fail_verify() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("tamper.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    publish_f32(&pkg, 1, "s", "only", &[1.0, 2.0]);

    let report = verify_tet_package(&root).unwrap();
    assert!(report.ok, "{:?}", report.findings);
    assert!(report.findings.iter().any(|f| {
        f.check == "secondary_indexes"
            && f.ok
            && f.detail.as_deref() == Some(SECONDARY_INDEX_PRESENT)
    }));

    let index_path = pkg.generation_dir(1).join(SECONDARY_INDEX_FILE);
    let original = fs::read(&index_path).unwrap();
    let mut flipped = original.clone();
    let last = flipped.len() - 2;
    flipped[last] ^= 0x01;
    fs::write(&index_path, &flipped).unwrap();
    let tampered = verify_tet_package(&root).unwrap();
    assert!(!tampered.ok, "{:?}", tampered.findings);
    assert!(
        tampered
            .findings
            .iter()
            .any(|f| f.check == "secondary_indexes_sha256" && !f.ok)
    );

    fs::write(&index_path, &original).unwrap();
    fs::remove_file(&index_path).unwrap();
    let missing = verify_tet_package(&root).unwrap();
    assert!(!missing.ok);
    assert!(
        missing
            .findings
            .iter()
            .any(|f| f.check == "secondary_indexes" && !f.ok)
    );
}

#[test]
fn absent_manifest_field_is_advisory() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("old.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    publish_f32(&pkg, 1, "s", "only", &[1.0, 2.0]);
    let manifest_path = pkg.generation_dir(1).join("MANIFEST.json");
    let mut value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    value.as_object_mut().unwrap().remove("secondary_indexes");
    value
        .as_object_mut()
        .unwrap()
        .remove("secondary_indexes_sha256");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&value).unwrap() + "\n",
    )
    .unwrap();
    fs::remove_file(pkg.generation_dir(1).join(SECONDARY_INDEX_FILE)).unwrap();

    let advisory = verify_tet_package(&root).unwrap();
    assert!(advisory.ok, "{:?}", advisory.findings);
    assert!(advisory.findings.iter().any(|f| {
        f.check == "secondary_indexes"
            && f.severity == VerifySeverity::Warning
            && f.detail.as_deref().is_some_and(|d| d.contains("advisory"))
    }));
}

#[test]
fn deep_verify_rejects_postings_that_do_not_match_payloads() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("deep.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    publish_f32(&pkg, 1, "s", "only", &[1.0, 2.0]);

    let index_path = pkg.generation_dir(1).join(SECONDARY_INDEX_FILE);
    let mut doc = read_secondary_index_file(&index_path).unwrap();
    let SecondaryIndexEntry::ValuePostings { postings, .. } = &mut doc.indexes[0] else {
        panic!("expected value postings");
    };
    postings.pop();
    let bytes = crate::catalog::encode_secondary_index(&doc).unwrap();
    let sha = crate::catalog::zone::sha256_hex(&bytes);
    fs::write(&index_path, &bytes).unwrap();
    let manifest_path = pkg.generation_dir(1).join("MANIFEST.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["secondary_indexes_sha256"] = json!(sha);
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap() + "\n",
    )
    .unwrap();

    let quick = verify_tet_package(&root).unwrap();
    assert!(quick.ok, "{:?}", quick.findings);

    let deep = verify_tet_package_with_options(&root, VerifyOptions { deep_decode: true }).unwrap();
    assert!(!deep.ok, "{:?}", deep.findings);
    assert!(
        deep.findings
            .iter()
            .any(|f| f.check == "secondary_indexes_stale" && !f.ok)
    );
}

#[test]
fn store_publish_writes_index_object() {
    let store = MemoryStore::new();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("part.tet");
    let mut session = crate::catalog::TetWriterSession::create(&path);
    session
        .push_dataset(
            TetDatasetWrite::row_major("n", DATASET_DTYPE_TAG_V1.i32, &[2], &[2], {
                let mut buf = Vec::new();
                buf.extend_from_slice(&(-2_i32).to_le_bytes());
                buf.extend_from_slice(&7_i32.to_le_bytes());
                buf
            })
            .unwrap(),
        )
        .unwrap();
    session.commit().unwrap();
    let bytes = fs::read(&path).unwrap();
    stage_part(&store, "sess", "p", &bytes).unwrap();
    let manifest = publish_staged_parts(&store, 1, &["p".into()]).unwrap();
    assert_eq!(
        manifest.secondary_indexes.as_deref(),
        Some(SECONDARY_INDEX_PRESENT)
    );
    let index_bytes = store.read_all(&secondary_index_key(1)).unwrap();
    let doc: crate::catalog::GenerationSecondaryIndex =
        serde_json::from_slice(&index_bytes).unwrap();
    let SecondaryIndexEntry::ValuePostings { postings, .. } = &doc.indexes[0] else {
        panic!("expected value postings");
    };
    assert_eq!(postings[0].value_le, "07000000");
    assert_eq!(postings[1].value_le, "feffffff");

    let listed = store.list_prefix("staging/").unwrap();
    assert!(listed.iter().all(|k| !k.ends_with(SECONDARY_INDEX_FILE)));

    let doc_q = crate::query::parse_query_json(r#"{"dataset":"n"}"#).unwrap();
    crate::query::validate_query(&doc_q).unwrap();
    let resp = crate::query::execute_query_on_store_pin(
        &doc_q,
        &store,
        GenerationPin::Explicit(1),
        crate::query::ExecuteQueryOptions::plan_only(),
    )
    .unwrap();
    assert!(resp.read_plan.is_some());
}

#[test]
fn eq_posting_skips_chunks_zone_maps_keep() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("eq.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    // Three chunks whose ranges all contain 9, but only the middle chunk stores 9.
    publish_f32(&pkg, 1, "s", "only", &[0.0, 10.0, 0.0, 9.0, 1.0, 10.0]);
    let packed = pkg.packed_tet_path(1);
    let bytes = fs::read(&packed).unwrap();
    let query = r#"{"dataset":"a","filter":{"eq":9},"sum":[]}"#;

    let hit = execute_sum(&packed, &bytes, query);
    let plan = hit.read_plan.expect("plan");
    assert_eq!(plan.zone_filter_skipped_chunks, 0);
    assert_eq!(plan.index_filter_skipped_chunks, 2);
    assert_eq!(plan.chunk_count, 1);
    assert_eq!(plan.chunks[0].chunk_index, vec![1]);
    assert_eq!(hit.execution.expect("execution").operation_sum, Some(9.0));
    assert!(
        hit.message.contains("index filter skipped 2"),
        "{}",
        hit.message
    );

    fs::remove_file(pkg.generation_dir(1).join(SECONDARY_INDEX_FILE)).unwrap();
    let fallback = execute_sum(&packed, &bytes, query);
    let fallback_plan = fallback.read_plan.expect("plan");
    assert_eq!(fallback_plan.zone_filter_skipped_chunks, 0);
    assert_eq!(fallback_plan.index_filter_skipped_chunks, 0);
    assert_eq!(fallback_plan.chunk_count, 3);
    assert_eq!(
        fallback.execution.expect("execution").operation_sum,
        Some(9.0)
    );
}

fn execute_sum(packed: &std::path::Path, bytes: &[u8], query: &str) -> crate::query::QueryResponse {
    let doc = crate::query::parse_query_json(query).unwrap();
    crate::query::validate_query(&doc).unwrap();
    crate::query::execute_query_document(
        &doc,
        packed,
        bytes,
        crate::query::ExecuteQueryOptions::execute_no_preview(),
        None,
    )
    .unwrap()
}

fn publish_f32_with_sorted_time(pkg: &TetPackage, generation: u64, vals: &[f32]) {
    let labels: Vec<String> = (0..vals.len()).map(|i| format!("{}.0", i * 10)).collect();
    let mut ds = TetDatasetWrite::f32_row_major(
        "a",
        &[vals.len() as u64],
        &[2],
        f32_le(vals),
    )
    .unwrap();
    ds.dim_names = Some(vec!["time".to_owned()]);
    let mut coords = BTreeMap::new();
    coords.insert(
        "time".to_owned(),
        CoordAxisV1 {
            labels,
        },
    );
    ds.coords = Some(coords);
    let mut w = TetShardWriter::create(pkg, "s", "only").unwrap();
    w.push_dataset(ds).unwrap();
    w.seal().unwrap();
    crate::catalog::publish_parts(pkg, generation, &["only".to_owned()]).unwrap();
}

#[test]
fn publish_writes_sorted_axis_range_for_monotonic_coords() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("sorted.tet.d")).unwrap();
    // Four values → two chunks; high-ish cardinality still gets value postings (4 ≤ 64)
    // plus sorted_axis_range on time.
    publish_f32_with_sorted_time(&pkg, 1, &[1.0, 2.0, 3.0, 4.0]);

    let doc = read_secondary_index_file(&pkg.generation_dir(1).join(SECONDARY_INDEX_FILE)).unwrap();
    let sorted = doc
        .indexes
        .iter()
        .find(|e| matches!(e, SecondaryIndexEntry::SortedAxisRange { .. }))
        .expect("sorted_axis_range entry");
    let SecondaryIndexEntry::SortedAxisRange {
        dataset_id,
        axis,
        ascending,
        segments,
    } = sorted
    else {
        unreachable!();
    };
    assert_eq!(*dataset_id, 0);
    assert_eq!(*axis, 0);
    assert!(*ascending);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].chunks, vec![vec![0]]);
    assert_eq!(segments[1].chunks, vec![vec![1]]);

    let report = verify_tet_package_with_options(
        pkg.root(),
        VerifyOptions {
            deep_decode: true,
        },
    )
    .unwrap();
    assert!(report.ok, "{:?}", report.findings);
}

#[test]
fn unsorted_coords_omit_sorted_axis_kind() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("unsorted.tet.d")).unwrap();
    let mut ds =
        TetDatasetWrite::f32_row_major("a", &[4], &[2], f32_le(&[1.0, 2.0, 3.0, 4.0])).unwrap();
    ds.dim_names = Some(vec!["time".to_owned()]);
    let mut coords = BTreeMap::new();
    coords.insert(
        "time".to_owned(),
        CoordAxisV1 {
            labels: vec![
                "10.0".into(),
                "30.0".into(),
                "20.0".into(),
                "40.0".into(),
            ],
        },
    );
    ds.coords = Some(coords);
    let mut w = TetShardWriter::create(&pkg, "s", "only").unwrap();
    w.push_dataset(ds).unwrap();
    w.seal().unwrap();
    crate::catalog::publish_parts(&pkg, 1, &["only".to_owned()]).unwrap();

    let doc = read_secondary_index_file(&pkg.generation_dir(1).join(SECONDARY_INDEX_FILE)).unwrap();
    assert!(
        doc.indexes
            .iter()
            .all(|e| !matches!(e, SecondaryIndexEntry::SortedAxisRange { .. })),
        "unsorted coords must not emit sorted_axis_range"
    );
}
