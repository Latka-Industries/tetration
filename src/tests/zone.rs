//! Zone maps written at compact/publish (THI-473) and the plan field rename (THI-540).

use std::fs;

use serde_json::json;

use crate::catalog::store::ReadAt;
use crate::catalog::{
    DATASET_DTYPE_TAG_V1, GenerationPin, MemoryStore, TetDatasetWrite, TetPackage, TetShardWriter,
    ZONE_MAPS_FILE, ZONE_MAPS_REQUIRED, publish_staged_parts, read_zone_maps_file, stage_part,
    zone_maps_key,
};
use crate::query::types::CHUNK_TOUCH_POLICY;
use crate::query::{parse_query_json, plan_query_with_tet_mmap, validate_query};
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

#[test]
fn publish_writes_zone_maps_without_touching_staging() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("z.tet.d")).unwrap();
    publish_f32(&pkg, 1, "s", "only", &[1.0, 2.0]);

    let staging = pkg.root().join("staging");
    for entry in fs::read_dir(&staging).unwrap() {
        let session = entry.unwrap().path();
        assert!(
            !session.join(ZONE_MAPS_FILE).is_file(),
            "staging must not carry zone maps"
        );
    }

    let zone_path = pkg.generation_dir(1).join(ZONE_MAPS_FILE);
    let first = fs::read(&zone_path).unwrap();
    let doc = read_zone_maps_file(&zone_path).unwrap();
    assert_eq!(doc.generation, 1);
    assert_eq!(doc.chunks.len(), 1);
    assert_eq!(doc.chunks[0].null_count, 0);
    assert_eq!(doc.chunks[0].element_count, 2);
    assert_eq!(doc.chunks[0].min_le.as_deref(), Some("0000803f"));
    assert_eq!(doc.chunks[0].max_le.as_deref(), Some("00000040"));

    let manifest = pkg.read_manifest(1).unwrap();
    assert_eq!(manifest.zone_maps.as_deref(), Some(ZONE_MAPS_REQUIRED));
    assert!(manifest.zone_maps_sha256.is_some());

    publish_f32(&pkg, 2, "s2", "next", &[9.0, 8.0]);
    assert_eq!(fs::read(&zone_path).unwrap(), first);
    let g2 = read_zone_maps_file(&pkg.generation_dir(2).join(ZONE_MAPS_FILE)).unwrap();
    assert_ne!(g2.chunks[0].min_le, doc.chunks[0].min_le);
}

#[test]
fn nan_is_null_and_excluded_from_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("nan.tet.d")).unwrap();
    publish_f32(&pkg, 1, "s", "only", &[1.0, f32::NAN, 3.0, 3.0]);
    let doc = read_zone_maps_file(&pkg.generation_dir(1).join(ZONE_MAPS_FILE)).unwrap();
    // chunk shape 2 → two chunks: [1, NaN] and [3, 3]
    assert_eq!(doc.chunks[0].null_count, 1);
    assert_eq!(doc.chunks[0].min_le, doc.chunks[0].max_le);
    assert_eq!(doc.chunks[1].null_count, 0);
    assert_eq!(doc.chunks[1].min_le, doc.chunks[1].max_le);
}

#[test]
fn verify_required_vs_advisory() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("v.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    publish_f32(&pkg, 1, "s", "only", &[1.0, 2.0]);

    let report = verify_tet_package(&root).unwrap();
    assert!(report.ok, "{:?}", report.findings);
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "zone_maps" && f.ok)
    );

    let zone_path = pkg.generation_dir(1).join(ZONE_MAPS_FILE);
    fs::remove_file(&zone_path).unwrap();
    let missing = verify_tet_package(&root).unwrap();
    assert!(!missing.ok);
    assert!(
        missing
            .findings
            .iter()
            .any(|f| f.check == "zone_maps" && !f.ok)
    );

    // Restore a sealed document, then drop the required flag to simulate a pre-0.4 manifest.
    publish_f32(&pkg, 2, "s2", "b", &[4.0, 5.0]);
    let manifest_path = pkg.generation_dir(1).join("MANIFEST.json");
    // G1 is still missing ZONE.json and marked required — point the advisory case at G2.
    let g2_manifest = pkg.generation_dir(2).join("MANIFEST.json");
    let mut value: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&g2_manifest).unwrap()).unwrap();
    value.as_object_mut().unwrap().remove("zone_maps");
    value.as_object_mut().unwrap().remove("zone_maps_sha256");
    fs::write(
        &g2_manifest,
        serde_json::to_string_pretty(&value).unwrap() + "\n",
    )
    .unwrap();
    // G1 still fails required. Remove G1's required flag too so the package can show advisory.
    let mut g1: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    g1.as_object_mut().unwrap().remove("zone_maps");
    g1.as_object_mut().unwrap().remove("zone_maps_sha256");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&g1).unwrap() + "\n",
    )
    .unwrap();

    let advisory = verify_tet_package(&root).unwrap();
    assert!(advisory.ok, "{:?}", advisory.findings);
    assert!(advisory.findings.iter().any(|f| {
        f.check == "zone_maps"
            && f.severity == VerifySeverity::Warning
            && f.detail.as_deref().is_some_and(|d| d.contains("advisory"))
    }));
}

#[test]
fn deep_verify_rejects_stats_that_do_not_match_payloads() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("deep.tet.d");
    let pkg = TetPackage::create(&root).unwrap();
    publish_f32(&pkg, 1, "s", "only", &[1.0, 2.0]);

    let zone_path = pkg.generation_dir(1).join(ZONE_MAPS_FILE);
    let mut doc = read_zone_maps_file(&zone_path).unwrap();
    doc.chunks[0].min_le = Some("00000000".into());
    let bytes = crate::catalog::encode_zone_maps(&doc).unwrap();
    let sha = crate::catalog::zone::sha256_hex(&bytes);
    fs::write(&zone_path, &bytes).unwrap();
    let manifest_path = pkg.generation_dir(1).join("MANIFEST.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["zone_maps_sha256"] = json!(sha);
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
            .any(|f| f.check == "zone_maps_stale" && !f.ok)
    );
}

#[test]
fn read_plan_carries_zone_stats_and_renamed_count() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("plan.tet.d")).unwrap();
    publish_f32(&pkg, 1, "s", "only", &[1.0, 4.0]);
    let packed = pkg.packed_tet_path(1);
    let bytes = fs::read(&packed).unwrap();
    let doc = parse_query_json(r#"{"dataset":"a"}"#).unwrap();
    validate_query(&doc).unwrap();
    let path = packed.display().to_string();
    let resp = plan_query_with_tet_mmap(&doc, Some(&path), &bytes, None).unwrap();
    let plan = resp.read_plan.expect("plan");
    assert_eq!(plan.logical_element_count, 2);
    let zone = plan.chunks[0].zone.as_ref().expect("zone");
    assert_eq!(zone.min_le.as_deref(), Some("0000803f"));
    assert_eq!(zone.max_le.as_deref(), Some("00008040"));

    let wire = serde_json::to_value(&plan).unwrap();
    assert!(wire.get("logical_element_count").is_some());
    assert!(wire.get("logical_f32_element_count").is_none());
    let mut old = wire.clone();
    let n = old
        .as_object_mut()
        .unwrap()
        .remove("logical_element_count")
        .unwrap();
    old["logical_f32_element_count"] = n;
    let parsed: crate::query::ReadPlan = serde_json::from_value(old).unwrap();
    assert_eq!(parsed.logical_element_count, 2);
    assert_eq!(
        parsed.chunk_touch_policy,
        CHUNK_TOUCH_POLICY.dense_half_open_unit_step
    );
}

#[test]
fn store_publish_writes_zone_object() {
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
    assert_eq!(manifest.zone_maps.as_deref(), Some(ZONE_MAPS_REQUIRED));
    let zone_bytes = store.read_all(&zone_maps_key(1)).unwrap();
    let doc: crate::catalog::GenerationZoneMaps = serde_json::from_slice(&zone_bytes).unwrap();
    assert_eq!(doc.chunks[0].min_le.as_deref(), Some("feffffff"));
    assert_eq!(doc.chunks[0].max_le.as_deref(), Some("07000000"));
    assert_eq!(doc.chunks[0].null_count, 0);

    let listed = store.list_prefix("staging/").unwrap();
    assert!(listed.iter().all(|k| !k.ends_with(ZONE_MAPS_FILE)));

    let doc_q = parse_query_json(r#"{"dataset":"n"}"#).unwrap();
    validate_query(&doc_q).unwrap();
    let resp = crate::query::execute_query_on_store_pin(
        &doc_q,
        &store,
        GenerationPin::Explicit(1),
        crate::query::ExecuteQueryOptions::plan_only(),
    )
    .unwrap();
    assert!(
        resp.read_plan
            .as_ref()
            .and_then(|p| p.chunks.first())
            .and_then(|c| c.zone.as_ref())
            .is_some()
    );
}
