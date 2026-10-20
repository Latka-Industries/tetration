//! Declared semantics helpers (THI-583).

use crate::catalog::{
    ScaffoldFormat, declared_attrs, get_dataset_attr, is_declared_key, metadata_has_declared_attrs,
    read_tet_summary_v1, scaffold_policy, set_dataset_attrs, write_recording_fixture,
};
use crate::layout::mmap_file_read;
use crate::query::{QueryDocument, plan_query_with_tet_mmap};

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
            ("semantics_profile", "tetration.media.v1"),
        ],
        &[] as &[&str],
    )
    .unwrap();
    let mmap = mmap_file_read(&bare).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let stub = scaffold_policy(&summary, ScaffoldFormat::Toml);
    assert!(stub.contains("[profiles.\"tetration.media.v1\"]"));
    assert!(stub.contains("index_to_seconds"));
    assert!(stub.contains("[units.Hz]"));

    let json = scaffold_policy(&summary, ScaffoldFormat::Json);
    assert!(json.contains("tetration.media.v1"));
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
        Some("tetration.media.v1")
    );
    assert!(summary.history.iter().any(|e| e.op == "semantics"));
}
