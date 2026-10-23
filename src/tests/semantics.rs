//! Declared semantics helpers (THI-583) + package SEMANTICS.json (THI-620).

use crate::catalog::{
    DatasetSemanticsDecl, PackageSemanticsV1, SEMANTICS_FILE, SEMANTICS_PRESENT, ScaffoldFormat,
    TetDatasetWrite, TetPackage, TetShardWriter, declared_attrs, get_dataset_attr, is_declared_key,
    metadata_has_declared_attrs, read_package_semantics, read_tet_summary_v1, scaffold_policy,
    set_dataset_attrs, write_package_semantics, write_recording_fixture,
};
use crate::layout::mmap_file_read;
use crate::query::{QueryDocument, plan_query_with_tet_mmap};
use crate::verify::verify_tet_package;

use super::fixture::write_multichunk_2x3_tiles;

#[test]
fn declared_key_catalog_includes_units_and_frame_rate() {
    assert!(is_declared_key("units"));
    assert!(is_declared_key("frame_rate"));
    assert!(!is_declared_key("long_name"));
}

#[test]
fn set_and_clear_attrs_roundtrip_and_surface_in_summary() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.tet");
    write_multichunk_2x3_tiles(&path, "a");

    set_dataset_attrs(
        &path,
        "a",
        &[("units", "K"), ("frame_rate", "30")],
        &[] as &[&str],
    )
    .unwrap();
    assert_eq!(
        get_dataset_attr(&path, "a", "units").unwrap().as_deref(),
        Some("K")
    );

    {
        let mmap = mmap_file_read(&path).unwrap();
        let summary = read_tet_summary_v1(&mmap).unwrap();
        assert!(metadata_has_declared_attrs(&summary.metadata));
        let attrs = declared_attrs(
            &summary
                .metadata
                .datasets
                .get("a")
                .expect("dataset meta")
                .attrs,
        );
        assert_eq!(attrs.get("units").map(String::as_str), Some("K"));
        assert_eq!(attrs.get("frame_rate").map(String::as_str), Some("30"));
        assert!(
            summary
                .history
                .iter()
                .any(|e| e.op == "semantics" && e.params.get("set").is_some())
        );
    }

    // Drop mmap before rewrite (Windows refuses writes with a user-mapped section open).
    set_dataset_attrs(&path, "a", &[] as &[(&str, &str)], &["frame_rate"]).unwrap();
    assert!(
        get_dataset_attr(&path, "a", "frame_rate")
            .unwrap()
            .is_none()
    );
    assert_eq!(
        get_dataset_attr(&path, "a", "units").unwrap().as_deref(),
        Some("K")
    );
}

#[test]
fn scaffold_empty_vs_declared() {
    let dir = tempfile::tempdir().unwrap();
    let bare = dir.path().join("bare.tet");
    write_multichunk_2x3_tiles(&bare, "temperature");
    {
        let mmap = mmap_file_read(&bare).unwrap();
        let summary = read_tet_summary_v1(&mmap).unwrap();
        let empty = scaffold_policy(&summary, ScaffoldFormat::Toml);
        assert!(empty.contains("No declared semantics attrs"));
        assert!(empty.contains("docs.tetration.sh/guides/semantics/"));
        assert!(!empty.contains("guides/semantics/examples"));
        assert!(!empty.contains("[profiles."));
    }

    set_dataset_attrs(
        &bare,
        "temperature",
        &[
            ("frame_rate", "30"),
            ("frame_rate_unit", "Hz"),
            ("semantics_profile", "media_v1"),
        ],
        &[] as &[&str],
    )
    .unwrap();
    let mmap = mmap_file_read(&bare).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let stub = scaffold_policy(&summary, ScaffoldFormat::Toml);
    assert!(stub.contains("[profiles.media_v1]"));
    assert!(!stub.contains("[[profiles]]"));
    assert!(!stub.contains("id = "));
    assert!(!stub.contains("[profiles.\""));
    assert!(stub.contains("index_to_seconds"));
    assert!(stub.contains("[units.Hz]"));

    let json = scaffold_policy(&summary, ScaffoldFormat::Json);
    assert!(json.contains("\"media_v1\""));
    assert!(!json.contains("\"id\""));
    assert!(json.contains("declared"));
}

#[test]
fn query_unchanged_with_or_without_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let doc: QueryDocument = serde_json::from_str(r#"{"dataset":"a","mean":[]}"#).unwrap();
    let before = {
        let mmap = mmap_file_read(&path).unwrap();
        plan_query_with_tet_mmap(&doc, None, &mmap, Some(0))
            .unwrap()
            .execution
            .unwrap()
            .operation_mean
            .unwrap()
    };

    set_dataset_attrs(
        &path,
        "a",
        &[("units", "K"), ("time_axis", "0")],
        &[] as &[&str],
    )
    .unwrap();
    let after = {
        let mmap = mmap_file_read(&path).unwrap();
        plan_query_with_tet_mmap(&doc, None, &mmap, Some(0))
            .unwrap()
            .execution
            .unwrap()
            .operation_mean
            .unwrap()
    };
    assert_eq!(before, after);
}

#[test]
fn recording_fixture_has_media_attrs_and_history() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("recording.tet");
    write_recording_fixture(&path).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    assert_eq!(summary.datasets[0].name, "frames");
    let attrs = &summary.metadata.datasets.get("frames").unwrap().attrs;
    assert_eq!(attrs.get("frame_rate").map(String::as_str), Some("30"));
    assert_eq!(attrs.get("time_axis").map(String::as_str), Some("0"));
    assert_eq!(
        attrs.get("semantics_profile").map(String::as_str),
        Some("media_v1")
    );
    assert!(summary.history.iter().any(|e| e.op == "semantics"));
}

#[test]
fn package_semantics_json_roundtrip_and_verify() {
    let dir = tempfile::tempdir().unwrap();
    let pkg = TetPackage::create(dir.path().join("sem.tet.d")).unwrap();
    let mut w = TetShardWriter::create(&pkg, "s", "only").unwrap();
    w.push_dataset(
        TetDatasetWrite::f32_row_major("a", &[2], &[2], {
            let mut b = Vec::new();
            for v in [1.0f32, 2.0] {
                b.extend_from_slice(&v.to_le_bytes());
            }
            b
        })
        .unwrap(),
    )
    .unwrap();
    w.seal().unwrap();
    crate::catalog::publish_parts(&pkg, 1, &["only".to_owned()]).unwrap();
    let packed = pkg.packed_tet_path(1);

    // Declare-only sidecar — do not rewrite sealed packed.tet (would break MANIFEST lens).
    let mut doc = PackageSemanticsV1::empty(1);
    let mut attrs = std::collections::BTreeMap::new();
    attrs.insert("units".into(), "K".into());
    attrs.insert("semantics_profile".into(), "cf_lite_v1".into());
    attrs.insert("standard_name".into(), "air_temperature".into());
    doc.datasets.insert(
        "a".into(),
        DatasetSemanticsDecl {
            semantics_profile: Some("cf_lite_v1".into()),
            attrs,
        },
    );
    let _sha = write_package_semantics(&pkg, 1, &doc, true).unwrap();

    let path = pkg.generation_dir(1).join(SEMANTICS_FILE);
    assert!(path.is_file());
    let manifest = pkg.read_manifest(1).unwrap();
    assert_eq!(manifest.semantics.as_deref(), Some(SEMANTICS_PRESENT));
    assert!(manifest.semantics_sha256.is_some());

    let again = read_package_semantics(&pkg, 1).unwrap().unwrap();
    assert_eq!(again, doc);

    let report = verify_tet_package(pkg.root()).unwrap();
    assert!(
        report
            .findings
            .iter()
            .any(|f| f.check == "semantics" && f.ok),
        "{:?}",
        report.findings
    );
    assert!(report.ok);

    // Fold answers still ignore SEMANTICS.json / footer attrs.
    let q: QueryDocument = serde_json::from_str(r#"{"dataset":"a","mean":[]}"#).unwrap();
    let mean = {
        let mmap = mmap_file_read(&packed).unwrap();
        plan_query_with_tet_mmap(&q, None, &mmap, Some(0))
            .unwrap()
            .execution
            .unwrap()
            .operation_mean
            .unwrap()
    };
    assert!((mean - 1.5).abs() < 1e-5);
}
