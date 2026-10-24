//! Declared semantics helpers (THI-583) + package SEMANTICS.json (THI-620) + PROV history (THI-624).

use crate::catalog::{
    DatasetSemanticsDecl, FooterBlobV1, HistoryEvent, PackageSemanticsV1, SEMANTICS_FILE,
    SEMANTICS_PRESENT, ScaffoldFormat, ScriptedWizardIo, TetDatasetWrite, TetPackage,
    TetShardWriter, WizardOptions, apply_semantics_toml, declared_attrs, format_apply_report,
    get_dataset_attr, is_declared_key, metadata_has_declared_attrs, read_package_semantics,
    read_tet_summary_v1, run_semantics_wizard, scaffold_policy, set_dataset_attrs,
    write_footer_blob, write_package_semantics, write_recording_fixture,
    write_wizard_scratch_fixture,
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
            ("semantics_profile", "media.v1"),
        ],
        &[] as &[&str],
    )
    .unwrap();
    let mmap = mmap_file_read(&bare).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let stub = scaffold_policy(&summary, ScaffoldFormat::Toml);
    assert!(stub.contains("schema_version = 1"), "{stub}");
    assert!(stub.contains("[datasets.temperature]"), "{stub}");
    assert!(stub.contains("[datasets.temperature.attrs]"), "{stub}");
    assert!(stub.contains("[profiles.media.v1]"), "{stub}");
    assert!(stub.contains("semantics_profile = \"media.v1\""), "{stub}");
    assert!(stub.contains("# media_kind ="), "{stub}");
    assert!(stub.contains("# time_axis ="), "{stub}");
    assert!(!stub.contains("[[profiles]]"), "{stub}");
    assert!(!stub.contains("id = "), "{stub}");
    assert!(stub.contains("index_to_seconds"), "{stub}");
    assert!(stub.contains("[units.Hz]"), "{stub}");

    let json = scaffold_policy(&summary, ScaffoldFormat::Json);
    assert!(json.contains("\"schema_version\": 1"), "{json}");
    assert!(json.contains("\"media.v1\""), "{json}");
    assert!(json.contains("\"datasets\""), "{json}");
    assert!(!json.contains("\"declared\""), "{json}");
    assert!(!json.contains("\"id\""), "{json}");
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
        Some("media.v1")
    );
    assert!(summary.history.iter().any(|e| e.op == "semantics"));
    let stub = scaffold_policy(&summary, ScaffoldFormat::Toml);
    assert!(stub.contains("dim_names = [\"t\", \"y\", \"x\"]"), "{stub}");
    assert!(stub.contains("[profiles.media.v1]"), "{stub}");
    assert!(!stub.contains("# frame_rate ="), "{stub}");
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
            dim_names: Some(vec!["time".into()]),
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

#[test]
fn package_semantics_copies_dim_names_without_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dims.tet");
    write_multichunk_2x3_tiles(&path, "frames");
    let mut blob = FooterBlobV1::default();
    blob.metadata = Some(crate::catalog::TetMetadataV1 {
        file: None,
        datasets: std::collections::BTreeMap::from([(
            "frames".to_owned(),
            crate::catalog::DatasetMetadataV1 {
                attrs: std::collections::BTreeMap::new(),
                dim_names: Some(vec!["y".into(), "x".into()]),
                coords: None,
            },
        )]),
    });
    write_footer_blob(&path, &blob).unwrap();

    let doc = PackageSemanticsV1::from_packed_tet(&path, 4).unwrap();
    assert_eq!(
        doc.datasets["frames"].dim_names.as_deref(),
        Some(["y".to_owned(), "x".to_owned()].as_slice())
    );
    assert!(doc.datasets["frames"].attrs.is_empty());
    assert!(doc.datasets["frames"].semantics_profile.is_none());
}

#[test]
fn history_prov_fields_roundtrip_and_ignore_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prov.tet");
    write_multichunk_2x3_tiles(&path, "a");

    let mut blob = FooterBlobV1::default();
    blob.history
        .push(HistoryEvent::new("semantics", "test").with_prov(
            Some("tet-test"),
            Some("annotate"),
            Some("dataset:a"),
        ));
    write_footer_blob(&path, &blob).unwrap();

    let mmap = mmap_file_read(&path).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let ev = &summary.history[0];
    assert_eq!(ev.agent_ref(), Some("tet-test"));
    assert_eq!(ev.activity_ref(), Some("annotate"));
    assert_eq!(ev.entity_ref(), Some("dataset:a"));
    assert_eq!(ev.params.get("agent").map(String::as_str), Some("tet-test"));

    // Unknown keys ignored (forward compatible).
    let raw = serde_json::json!({
        "history": [{
            "op": "x",
            "source": "y",
            "at": "1",
            "agent": "a",
            "future_prov_key": true
        }]
    });
    let parsed: FooterBlobV1 = serde_json::from_value(raw).unwrap();
    assert_eq!(parsed.history[0].agent.as_deref(), Some("a"));
}

const APPLY_TOML: &str = r#"
schema_version = 1

[datasets.frames]
dim_names = ["y", "x"]

[datasets.frames.attrs]
semantics_profile = "media.v1"
media_kind = "video"
frame_rate = 30
frame_rate_unit = "Hz"
time_axis = 0
pixel_unit = "px"

[profiles.media.v1]
index_to_seconds = "index / frame_rate"
seconds_to_index = "floor(seconds * frame_rate)"

[units.Hz]
display = "Hz"
"#;

fn write_apply_toml(dir: &std::path::Path, body: &str) -> std::path::PathBuf {
    let path = dir.join("tet-semantics.toml");
    std::fs::write(&path, body).unwrap();
    path
}

#[test]
fn apply_stamps_dims_and_attrs_and_leaves_formulas_in_toml() {
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("clip.tet");
    write_multichunk_2x3_tiles(&tet, "frames");
    set_dataset_attrs(&tet, "frames", &[("units", "K")], &[] as &[&str]).unwrap();
    let toml = write_apply_toml(dir.path(), APPLY_TOML);
    let before = std::fs::read(&toml).unwrap();

    let report = apply_semantics_toml(&tet, &toml).unwrap();
    assert_eq!(std::fs::read(&toml).unwrap(), before);
    let text = format_apply_report(&tet, &report);
    assert!(text.contains("dim_names=[y, x]"), "{text}");
    assert!(text.contains("frame_rate"), "{text}");
    assert!(text.contains("profiles media.v1"), "{text}");
    assert!(text.contains("units Hz"), "{text}");
    assert!(!text.contains("index_to_seconds"), "{text}");

    let bytes = std::fs::read(&tet).unwrap();
    let summary = read_tet_summary_v1(&bytes).unwrap();
    let ds = summary.metadata.datasets.get("frames").unwrap();
    assert_eq!(
        ds.dim_names.as_deref(),
        Some(["y".to_owned(), "x".to_owned()].as_slice())
    );
    assert_eq!(ds.attrs.get("frame_rate").map(String::as_str), Some("30"));
    assert_eq!(ds.attrs.get("units").map(String::as_str), Some("K"));
    assert_eq!(
        ds.attrs.get("semantics_profile").map(String::as_str),
        Some("media.v1")
    );
    let lossy = String::from_utf8_lossy(&bytes);
    assert!(
        !lossy.contains("index_to_seconds"),
        "formula leaked into footer"
    );
    assert!(
        summary
            .history
            .iter()
            .any(|e| { e.op == "semantics" && e.params.get("via").is_some_and(|v| v == "apply") })
    );
}

#[test]
fn apply_normalizes_profile_alias_and_preserves_omitted_dim_names() {
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("clip.tet");
    write_multichunk_2x3_tiles(&tet, "frames");
    let first = write_apply_toml(dir.path(), APPLY_TOML);
    apply_semantics_toml(&tet, &first).unwrap();
    let second = r#"
schema_version = 1
[datasets.frames.attrs]
semantics_profile = "v1.media"
frame_rate = 24
"#;
    let toml = write_apply_toml(dir.path(), second);
    let report = apply_semantics_toml(&tet, &toml).unwrap();
    assert!(
        report.stamped[0]
            .profile_normalized
            .as_ref()
            .is_some_and(|(from, to)| from == "v1.media" && to == "media.v1")
    );
    let bytes = std::fs::read(&tet).unwrap();
    let summary = read_tet_summary_v1(&bytes).unwrap();
    let ds = summary.metadata.datasets.get("frames").unwrap();
    assert_eq!(
        ds.dim_names.as_deref(),
        Some(["y".to_owned(), "x".to_owned()].as_slice())
    );
    assert_eq!(ds.attrs.get("frame_rate").map(String::as_str), Some("24"));
    assert_eq!(ds.attrs.get("pixel_unit").map(String::as_str), Some("px"));
    assert_eq!(
        ds.attrs.get("semantics_profile").map(String::as_str),
        Some("media.v1")
    );
}

#[test]
fn apply_rejects_missing_dataset_rank_mismatch_coords_and_sealed_package() {
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("clip.tet");
    write_multichunk_2x3_tiles(&tet, "frames");
    let before = std::fs::read(&tet).unwrap();

    let missing = write_apply_toml(
        dir.path(),
        "schema_version = 1\n[datasets.nope.attrs]\nunits = \"K\"\n",
    );
    let err = apply_semantics_toml(&tet, &missing)
        .unwrap_err()
        .to_string();
    assert!(err.contains("not found"), "{err}");

    let rank = write_apply_toml(
        dir.path(),
        "schema_version = 1\n[datasets.frames]\ndim_names = [\"t\", \"y\", \"x\"]\n",
    );
    let err = apply_semantics_toml(&tet, &rank).unwrap_err().to_string();
    assert!(err.contains("rank"), "{err}");

    let axis = write_apply_toml(
        dir.path(),
        "schema_version = 1\n[datasets.frames.attrs]\ntime_axis = 9\n",
    );
    let err = apply_semantics_toml(&tet, &axis).unwrap_err().to_string();
    assert!(err.contains("time_axis"), "{err}");

    let coords = write_apply_toml(
        dir.path(),
        "schema_version = 1\n[datasets.frames]\ncoords = [\"t\"]\n",
    );
    let err = apply_semantics_toml(&tet, &coords).unwrap_err().to_string();
    assert!(err.contains("coords"), "{err}");
    assert_eq!(std::fs::read(&tet).unwrap(), before);

    let pkg = dir.path().join("clip.tet.d");
    std::fs::create_dir(&pkg).unwrap();
    std::fs::write(pkg.join("PROTOCOL"), b"{}\n").unwrap();
    let err = apply_semantics_toml(&pkg, &missing)
        .unwrap_err()
        .to_string();
    assert!(err.contains("sealed"), "{err}");
}

#[test]
fn wizard_scripted_stamps_dims_and_media_attrs() {
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("wizard.tet");
    write_wizard_scratch_fixture(&tet).unwrap();
    let toml_out = dir.path().join("tet-semantics.toml");

    // dims t/y/x; profile index 0 = media.v1; recommended attrs; time_axis 0; no extras; confirm
    let mut io = ScriptedWizardIo::new(vec![
        Some("t".into()),
        Some("y".into()),
        Some("x".into()),
        Some("0".into()), // select media.v1
        Some("video".into()),
        Some("30".into()),
        Some("Hz".into()),
        Some("0".into()), // select time_axis 0
        None,             // end extras
        None,             // confirm Y
    ]);
    let opts = WizardOptions {
        dataset: None,
        toml_out: toml_out.clone(),
        apply: true,
    };
    let result = run_semantics_wizard(&tet, &opts, &mut io).unwrap();
    assert_eq!(result.dataset, "frames");
    assert!(toml_out.is_file());
    let text = std::fs::read_to_string(&toml_out).unwrap();
    assert!(text.contains("dim_names = [\"t\", \"y\", \"x\"]"), "{text}");
    assert!(text.contains("semantics_profile = \"media.v1\""), "{text}");
    assert!(text.contains("[profiles.media.v1]"), "{text}");

    let mmap = mmap_file_read(&tet).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let meta = summary.metadata.datasets.get("frames").unwrap();
    assert_eq!(
        meta.dim_names.as_deref(),
        Some(["t".into(), "y".into(), "x".into()].as_slice())
    );
    assert_eq!(
        meta.attrs.get("semantics_profile").map(String::as_str),
        Some("media.v1")
    );
    assert_eq!(meta.attrs.get("frame_rate").map(String::as_str), Some("30"));
}

#[test]
fn wizard_empty_skips_attrs_without_prior() {
    let dir = tempfile::tempdir().unwrap();
    let tet = dir.path().join("wizard.tet");
    write_wizard_scratch_fixture(&tet).unwrap();
    let toml_out = dir.path().join("out.toml");
    // dims; profile 0; empty×4 recommended (skip — hints are not defaults); no extras; confirm
    let mut io = ScriptedWizardIo::new(vec![
        Some("t".into()),
        Some("y".into()),
        Some("x".into()),
        Some("0".into()),    // select media.v1
        None,                // media_kind skip
        None,                // frame_rate skip
        None,                // frame_rate_unit skip
        Some("skip".into()), // time_axis skip (select)
        None,                // end extras
        None,                // confirm Y
    ]);
    let opts = WizardOptions {
        dataset: None,
        toml_out,
        apply: true,
    };
    run_semantics_wizard(&tet, &opts, &mut io).unwrap();
    let mmap = mmap_file_read(&tet).unwrap();
    let summary = read_tet_summary_v1(&mmap).unwrap();
    let attrs = &summary.metadata.datasets.get("frames").unwrap().attrs;
    assert_eq!(
        attrs.get("semantics_profile").map(String::as_str),
        Some("media.v1")
    );
    assert!(!attrs.contains_key("media_kind"));
    assert!(!attrs.contains_key("frame_rate"));
}
