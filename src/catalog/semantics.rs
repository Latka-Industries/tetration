//! Declared science/media attrs helpers and caller-policy scaffold (0.5.1).
//!
//! Facts live in footer `metadata.datasets[<name>].attrs`. Interpretation lives in a
//! caller policy file outside the `.tet` — see `docs/semantics.md`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Serialize;

use crate::utils::dtype::ElementDtype;

use super::history::{FooterBlobV1, HistoryEvent, read_footer_blob_from_path, write_footer_blob};
use super::metadata::DatasetMetadataV1;
use super::{CatalogError, DatasetRecordV1, TetFileSummaryV1, TetMetadataV1, read_tet_summary_v1};

/// Recommended declared attr keys (`docs/semantics.md`). Unknown keys remain allowed.
pub const DECLARED_ATTR_KEYS: &[&str] = &[
    "units",
    "frame_rate",
    "frame_rate_unit",
    "sample_rate",
    "sample_rate_unit",
    "time_axis",
    "crs",
    "media_kind",
    "color_space",
    "pixel_unit",
    "start_time",
    "semantics_profile",
    "uncertainty_kind",
    "uncertainty_sibling",
    "standard_name",
];

/// Doc URL emitted by [`scaffold_policy`] when no declared attrs are present.
pub const SEMANTICS_GUIDE_URL: &str = "https://docs.tetration.sh/guides/semantics/";

/// Output format for [`scaffold_policy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaffoldFormat {
    /// `tet-semantics.toml` stub.
    Toml,
    /// `tet-semantics.json` stub.
    Json,
}

impl ScaffoldFormat {
    /// Parse `toml` / `json` (case-insensitive).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::InvalidWriteSpec`] for other values.
    pub fn parse(s: &str) -> Result<Self, CatalogError> {
        match s.trim().to_ascii_lowercase().as_str() {
            "toml" => Ok(Self::Toml),
            "json" => Ok(Self::Json),
            _ => Err(CatalogError::InvalidWriteSpec(
                "scaffold format must be toml or json",
            )),
        }
    }
}

/// True when `key` is in [`DECLARED_ATTR_KEYS`].
#[must_use]
pub fn is_declared_key(key: &str) -> bool {
    DECLARED_ATTR_KEYS.contains(&key)
}

/// Filter `attrs` to the recommended declared keys (stable key order).
#[must_use]
pub fn declared_attrs(attrs: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    attrs
        .iter()
        .filter(|(k, _)| is_declared_key(k))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

/// Declared attrs for one dataset from footer metadata (empty when absent).
#[must_use]
pub fn declared_attrs_for_dataset(
    metadata: &TetMetadataV1,
    dataset: &str,
) -> BTreeMap<String, String> {
    metadata
        .datasets
        .get(dataset)
        .map(|ds| declared_attrs(&ds.attrs))
        .unwrap_or_default()
}

/// Whether any dataset in `metadata` carries a recommended declared key.
#[must_use]
pub fn metadata_has_declared_attrs(metadata: &TetMetadataV1) -> bool {
    metadata
        .datasets
        .values()
        .any(|ds| ds.attrs.keys().any(|k| is_declared_key(k)))
}

/// Read one dataset attr from a `.tet` path (any key; `None` when missing).
///
/// # Errors
///
/// Returns [`CatalogError`] when the file cannot be read or the footer is malformed.
pub fn get_dataset_attr(
    path: &Path,
    dataset: &str,
    key: &str,
) -> Result<Option<String>, CatalogError> {
    let summary = read_summary_from_path(path)?;
    ensure_dataset_exists(&summary, dataset)?;
    Ok(summary
        .metadata
        .datasets
        .get(dataset)
        .and_then(|ds| ds.attrs.get(key).cloned()))
}

/// Declare and/or clear attrs on one dataset; appends a `semantics` history row.
///
/// Does not reinterpret values. Clears run before sets. Empty `set` + empty `clear` is a no-op.
///
/// # Errors
///
/// Returns [`CatalogError`] when the dataset is missing, validation fails, or I/O fails.
pub fn set_dataset_attrs<K, V, C>(
    path: &Path,
    dataset: &str,
    set: &[(K, V)],
    clear: &[C],
) -> Result<(), CatalogError>
where
    K: AsRef<str>,
    V: AsRef<str>,
    C: AsRef<str>,
{
    if set.is_empty() && clear.is_empty() {
        return Ok(());
    }

    let summary = read_summary_from_path(path)?;
    ensure_dataset_exists(&summary, dataset)?;

    let mut blob = read_footer_blob_from_path(path).unwrap_or_default();
    let meta = blob.metadata.get_or_insert_with(TetMetadataV1::default);
    let ds = meta.dataset_mut(dataset);

    let mut cleared = Vec::new();
    for key in clear {
        let key = key.as_ref();
        if ds.attrs.remove(key).is_some() {
            cleared.push(key.to_owned());
        }
    }

    let mut set_keys = Vec::new();
    for (key, value) in set {
        let key = key.as_ref();
        let value = value.as_ref();
        ds.attrs.insert(key.to_owned(), value.to_owned());
        set_keys.push(key.to_owned());
    }

    // Drop empty dataset entries that only existed for attrs we cleared.
    if DatasetMetadataV1::import_is_empty(&ds.attrs, ds.dim_names.as_ref(), ds.coords.as_ref()) {
        meta.datasets.remove(dataset);
    }
    if meta.file.is_none() && meta.datasets.is_empty() {
        blob.metadata = None;
    }

    let mut event = HistoryEvent::new("semantics", path.display().to_string());
    event.params.insert("dataset".into(), dataset.to_owned());
    if !set_keys.is_empty() {
        event.params.insert("set".into(), set_keys.join(","));
    }
    if !cleared.is_empty() {
        event.params.insert("clear".into(), cleared.join(","));
    }
    blob.history.push(event);

    write_footer_blob(path, &blob)
}

/// Emit a caller-policy stub from declared attrs (or catalog + doc links when none).
///
/// Never invents commented fake config when attrs are absent.
#[must_use]
pub fn scaffold_policy(summary: &TetFileSummaryV1, format: ScaffoldFormat) -> String {
    match format {
        ScaffoldFormat::Toml => scaffold_toml(summary),
        ScaffoldFormat::Json => scaffold_json(summary),
    }
}

/// Scaffold from a `.tet` path.
///
/// # Errors
///
/// Returns [`CatalogError`] when the file cannot be read.
pub fn scaffold_policy_from_path(
    path: &Path,
    format: ScaffoldFormat,
) -> Result<String, CatalogError> {
    let summary = read_summary_from_path(path)?;
    Ok(scaffold_policy(&summary, format))
}

fn read_summary_from_path(path: &Path) -> Result<TetFileSummaryV1, CatalogError> {
    let data = std::fs::read(path)?;
    read_tet_summary_v1(&data)
}

fn ensure_dataset_exists(summary: &TetFileSummaryV1, dataset: &str) -> Result<(), CatalogError> {
    if summary.datasets.iter().any(|d| d.name == dataset) {
        Ok(())
    } else {
        Err(CatalogError::InvalidWriteSpec(
            "dataset not found in catalog",
        ))
    }
}

fn dtype_label(tag: u32) -> &'static str {
    match ElementDtype::try_from_wire_tag(tag) {
        Some(ElementDtype::F32) => "f32",
        Some(ElementDtype::F64) => "f64",
        Some(ElementDtype::I32) => "i32",
        Some(ElementDtype::I64) => "i64",
        Some(ElementDtype::U8) => "u8",
        Some(ElementDtype::U16) => "u16",
        Some(ElementDtype::I16) => "i16",
        Some(ElementDtype::U32) => "u32",
        Some(ElementDtype::F16) => "f16",
        Some(ElementDtype::U64) => "u64",
        None => "?",
    }
}

fn catalog_lines(datasets: &[DatasetRecordV1]) -> Vec<String> {
    datasets
        .iter()
        .map(|d| format!("{}  {}  shape={:?}", d.name, dtype_label(d.dtype), d.shape))
        .collect()
}

fn collect_declared_by_dataset(
    summary: &TetFileSummaryV1,
) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for rec in &summary.datasets {
        let attrs = declared_attrs_for_dataset(&summary.metadata, &rec.name);
        if !attrs.is_empty() {
            out.insert(rec.name.clone(), attrs);
        }
    }
    out
}

fn scaffold_toml(summary: &TetFileSummaryV1) -> String {
    let declared = collect_declared_by_dataset(summary);
    let mut out = String::new();
    if declared.is_empty() {
        out.push_str("# No declared semantics attrs in this .tet.\n");
        out.push_str("# Catalog:\n");
        for line in catalog_lines(&summary.datasets) {
            out.push_str(&format!("#   {line}\n"));
        }
        out.push_str("#\n");
        out.push_str("# Docs:\n");
        out.push_str(&format!("#   {SEMANTICS_GUIDE_URL}\n"));
        return out;
    }

    out.push_str("# Generated by `tet semantics scaffold` from declared .tet attrs.\n");
    out.push_str("# Caller policy only — the crate does not load this file by default.\n\n");

    let profiles = collect_profile_ids(&declared);
    for profile in &profiles {
        // Identity is the table key (TOML-safe; dots → underscores). No spare `id` field.
        let key = profile_table_key(profile);
        out.push_str(&format!("[profiles.{key}]\n"));
        if profile_uses_rate(profile, &declared) {
            out.push_str("index_to_seconds = \"index / frame_rate\"\n");
            out.push_str("seconds_to_index = \"floor(seconds * frame_rate)\"\n");
        }
        out.push('\n');
    }

    let units = collect_unit_tokens(&declared);
    for unit in &units {
        out.push_str(&format!("[units.{unit}]\n"));
        out.push_str(&format!("display = \"{unit}\"\n\n"));
    }

    out.push_str("# Declared attrs (facts in the .tet):\n");
    for (ds, attrs) in &declared {
        for (k, v) in attrs {
            out.push_str(&format!("#   {ds}.{k} = {v:?}\n"));
        }
    }
    out
}

fn scaffold_json(summary: &TetFileSummaryV1) -> String {
    let declared = collect_declared_by_dataset(summary);
    if declared.is_empty() {
        #[derive(Serialize)]
        struct EmptyScaffold {
            note: &'static str,
            catalog: Vec<String>,
            docs: [&'static str; 1],
        }
        let body = EmptyScaffold {
            note: "No declared semantics attrs in this .tet.",
            catalog: catalog_lines(&summary.datasets),
            docs: [SEMANTICS_GUIDE_URL],
        };
        return serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".to_owned());
    }

    #[derive(Serialize)]
    struct PolicyScaffold {
        profiles: BTreeMap<String, BTreeMap<String, String>>,
        units: BTreeMap<String, BTreeMap<String, String>>,
        declared: BTreeMap<String, BTreeMap<String, String>>,
    }

    let mut profiles = BTreeMap::new();
    for profile in collect_profile_ids(&declared) {
        let key = profile_table_key(&profile);
        let mut entry = BTreeMap::new();
        if profile_uses_rate(&profile, &declared) {
            entry.insert(
                "index_to_seconds".to_owned(),
                "index / frame_rate".to_owned(),
            );
            entry.insert(
                "seconds_to_index".to_owned(),
                "floor(seconds * frame_rate)".to_owned(),
            );
        }
        profiles.insert(key, entry);
    }

    let mut units = BTreeMap::new();
    for unit in collect_unit_tokens(&declared) {
        units.insert(unit.clone(), BTreeMap::from([("display".to_owned(), unit)]));
    }

    let body = PolicyScaffold {
        profiles,
        units,
        declared,
    };
    serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".to_owned())
}

/// TOML/JSON map key for a profile id — dots become underscores (bare TOML keys).
fn profile_table_key(profile: &str) -> String {
    profile.replace('.', "_")
}

fn collect_profile_ids(declared: &BTreeMap<String, BTreeMap<String, String>>) -> BTreeSet<String> {
    let mut profiles = BTreeSet::new();
    for attrs in declared.values() {
        if let Some(p) = attrs.get("semantics_profile") {
            profiles.insert(p.clone());
            continue;
        }
        if attrs.contains_key("frame_rate") || attrs.get("media_kind").is_some_and(|k| k == "video")
        {
            profiles.insert("media_v1".to_owned());
        } else if attrs.contains_key("sample_rate")
            || attrs.get("media_kind").is_some_and(|k| k == "timeseries")
        {
            profiles.insert("timeseries_v1".to_owned());
        } else if attrs.contains_key("crs") || attrs.contains_key("standard_name") {
            // climate-as-attrs example only — doc profile id, not a runtime mode
            profiles.insert("climate_v1".to_owned());
        }
    }
    if profiles.is_empty() {
        profiles.insert("media_v1".to_owned());
    }
    profiles
}

fn profile_uses_rate(profile: &str, declared: &BTreeMap<String, BTreeMap<String, String>>) -> bool {
    if profile.contains("media") || profile.contains("timeseries") {
        return true;
    }
    declared
        .values()
        .any(|a| a.contains_key("frame_rate") || a.contains_key("sample_rate"))
}

fn collect_unit_tokens(declared: &BTreeMap<String, BTreeMap<String, String>>) -> BTreeSet<String> {
    let mut units = BTreeSet::new();
    for attrs in declared.values() {
        for key in ["units", "frame_rate_unit", "sample_rate_unit", "pixel_unit"] {
            if let Some(u) = attrs.get(key) {
                // TOML bare keys: keep simple tokens only.
                if u.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    units.insert(u.clone());
                }
            }
        }
    }
    units
}

/// Write a small recording fixture with media attrs + provenance (tests / tracked fixtures).
///
/// Shape `[6, 2, 2]` / chunk `[2, 2, 2]`, dataset `frames`, values `1..24` as `f32`.
///
/// # Errors
///
/// Returns [`CatalogError`] when write or footer update fails.
pub fn write_recording_fixture(path: &Path) -> Result<(), CatalogError> {
    use super::{
        CHUNK_PAYLOAD_CODEC_V1, DATASET_DTYPE_TAG_V1, RawArrayWrite, write_raw_array_file,
    };

    const SHAPE: [u64; 3] = [6, 2, 2];
    const CHUNK: [u64; 3] = [2, 2, 2];
    let mut data = vec![0u8; 6 * 2 * 2 * 4];
    for (slot, n) in data.chunks_exact_mut(4).zip(1_u32..=24) {
        slot.copy_from_slice(&(n as f32).to_le_bytes());
    }
    write_raw_array_file(
        path,
        &RawArrayWrite {
            name: "frames",
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &SHAPE,
            chunk_shape: &CHUNK,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &data,
            file_execution: None,
        },
    )?;

    let mut attrs = BTreeMap::new();
    attrs.insert("frame_rate".into(), "30".into());
    attrs.insert("frame_rate_unit".into(), "Hz".into());
    attrs.insert("time_axis".into(), "0".into());
    attrs.insert("media_kind".into(), "video".into());
    attrs.insert("semantics_profile".into(), "media_v1".into());
    attrs.insert("pixel_unit".into(), "px".into());

    let mut event = HistoryEvent::new("semantics", "fixtures/small/tet/recording.tet");
    event.params.insert("tool".into(), "tetration".into());
    event
        .params
        .insert("note".into(), "recording worked example".into());

    write_footer_blob(
        path,
        &FooterBlobV1 {
            history: vec![event],
            metadata: Some(TetMetadataV1 {
                file: None,
                datasets: BTreeMap::from([(
                    "frames".to_owned(),
                    DatasetMetadataV1 {
                        attrs,
                        dim_names: Some(vec!["t".into(), "y".into(), "x".into()]),
                        coords: None,
                    },
                )]),
            }),
            metadata_ref: None,
        },
    )?;
    Ok(())
}
