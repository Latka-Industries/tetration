//! Declared science/media attrs helpers and caller-policy scaffold (0.5.1).
//!
//! Facts live in footer `metadata.datasets[<name>].attrs`. Interpretation lives in a
//! caller policy file outside the `.tet` — see `docs/semantics.md`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use serde::Serialize;

use super::path_io::read_summary_from_path;
use super::profiles::{
    MEDIA_V1, default_rate_formulas, infer_profile_id, normalize_profile_id, rate_attr_for_profile,
    recommended_keys_for_profile,
};

use crate::catalog::history::{
    FooterBlobV1, HistoryEvent, read_footer_blob_from_path, write_footer_blob,
};
use crate::catalog::metadata::DatasetMetadataV1;
use crate::catalog::{CatalogError, DatasetRecordV1, TetFileSummaryV1, TetMetadataV1};
use crate::utils::dtype::dtype_wire_label;

/// Footer / declare key for the soft profile contract id.
pub const ATTR_SEMANTICS_PROFILE: &str = "semantics_profile";
/// Footer / declare key for the time axis index (stringified integer).
pub const ATTR_TIME_AXIS: &str = "time_axis";

/// Recommended declared attr keys (`docs/semantics.md`). Unknown keys remain allowed.
pub const DECLARED_ATTR_KEYS: &[&str] = &[
    "units",
    "frame_rate",
    "frame_rate_unit",
    "sample_rate",
    "sample_rate_unit",
    ATTR_TIME_AXIS,
    "crs",
    "media_kind",
    "color_space",
    "pixel_unit",
    "start_time",
    ATTR_SEMANTICS_PROFILE,
    "uncertainty_kind",
    "uncertainty_sibling",
    "standard_name",
];

/// Doc URL emitted by [`scaffold_policy`] when no declared attrs are present.
pub const SEMANTICS_GUIDE_URL: &str = "https://docs.tetration.sh/guides/semantics/";

/// Output format for [`scaffold_policy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "tetration-cli", derive(clap::ValueEnum))]
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

fn ensure_dataset_exists(summary: &TetFileSummaryV1, dataset: &str) -> Result<(), CatalogError> {
    if summary.datasets.iter().any(|d| d.name == dataset) {
        Ok(())
    } else {
        Err(CatalogError::InvalidWriteSpec(
            "dataset not found in catalog",
        ))
    }
}

pub(crate) fn dtype_label(tag: u32) -> &'static str {
    dtype_wire_label(tag)
}

fn catalog_lines(datasets: &[DatasetRecordV1]) -> Vec<String> {
    datasets
        .iter()
        .map(|d| format!("{}  {}  shape={:?}", d.name, dtype_label(d.dtype), d.shape))
        .collect()
}

/// Caller-policy tables copied into an annotated scaffold (formulas and units stay out of the footer).
#[derive(Debug, Clone, Default)]
pub struct ScaffoldPolicy {
    pub profiles: BTreeMap<String, BTreeMap<String, String>>,
    pub units: BTreeMap<String, BTreeMap<String, String>>,
}

/// Combined one-file TOML for `summary`.
///
/// `overlay` keeps formulas and units from an existing `--from` file. Missing recommended
/// attr keys are `# key =` comments. Does not invent a file when nothing is declared.
#[must_use]
pub fn scaffold_annotated_toml(
    summary: &TetFileSummaryV1,
    overlay: Option<&ScaffoldPolicy>,
) -> String {
    let rows = collect_declare_rows(summary);
    if rows.is_empty() {
        return empty_scaffold_toml(summary);
    }
    render_combined_toml(&rows, overlay)
}

fn empty_scaffold_toml(summary: &TetFileSummaryV1) -> String {
    let mut out = String::new();
    out.push_str("# No declared semantics attrs in this .tet.\n");
    out.push_str("# Catalog:\n");
    for line in catalog_lines(&summary.datasets) {
        let _ = writeln!(out, "#   {line}");
    }
    out.push_str("#\n");
    out.push_str("# Docs:\n");
    let _ = writeln!(out, "#   {SEMANTICS_GUIDE_URL}");
    out
}

fn scaffold_toml(summary: &TetFileSummaryV1) -> String {
    scaffold_annotated_toml(summary, None)
}

fn scaffold_json(summary: &TetFileSummaryV1) -> String {
    #[derive(Serialize)]
    struct EmptyScaffold {
        note: &'static str,
        catalog: Vec<String>,
        docs: [&'static str; 1],
    }
    #[derive(Serialize)]
    struct DatasetOut {
        #[serde(skip_serializing_if = "Option::is_none")]
        dim_names: Option<Vec<String>>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        attrs: BTreeMap<String, String>,
    }
    #[derive(Serialize)]
    struct CombinedScaffold {
        schema_version: u32,
        datasets: BTreeMap<String, DatasetOut>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        profiles: BTreeMap<String, BTreeMap<String, String>>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        units: BTreeMap<String, BTreeMap<String, String>>,
    }

    let rows = collect_declare_rows(summary);
    if rows.is_empty() {
        let body = EmptyScaffold {
            note: "No declared semantics attrs in this .tet.",
            catalog: catalog_lines(&summary.datasets),
            docs: [SEMANTICS_GUIDE_URL],
        };
        return serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".to_owned());
    }

    let mut datasets = BTreeMap::new();
    for row in &rows {
        datasets.insert(
            row.name.clone(),
            DatasetOut {
                dim_names: row.facts.dim_names.clone(),
                attrs: row.facts.attrs.clone(),
            },
        );
    }
    let profiles = generated_profiles(&rows);
    let units = generated_units(&rows);
    let body = CombinedScaffold {
        schema_version: 1,
        datasets,
        profiles,
        units,
    };
    serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".to_owned())
}

/// Shared `dim_names` + attrs facts (scaffold / apply / wizard). Open attr minting stays a map.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeclareFacts {
    /// Axis names in rank order. `None` means “do not touch” (apply) or absent (scaffold).
    pub dim_names: Option<Vec<String>>,
    /// Footer / TOML attrs (any key allowed).
    pub attrs: BTreeMap<String, String>,
}

struct DeclareRow {
    name: String,
    facts: DeclareFacts,
}

fn collect_declare_rows(summary: &TetFileSummaryV1) -> Vec<DeclareRow> {
    let mut rows = Vec::new();
    for rec in &summary.datasets {
        let meta = summary.metadata.datasets.get(&rec.name);
        let dim_names = meta
            .and_then(|ds| ds.dim_names.clone())
            .filter(|names| !names.is_empty());
        let attrs = meta.map(|ds| ds.attrs.clone()).unwrap_or_default();
        if dim_names.is_none() && attrs.is_empty() {
            continue;
        }
        rows.push(DeclareRow {
            name: rec.name.clone(),
            facts: DeclareFacts { dim_names, attrs },
        });
    }
    rows
}

fn render_combined_toml(rows: &[DeclareRow], overlay: Option<&ScaffoldPolicy>) -> String {
    let mut out = String::new();
    out.push_str("# Generated by `tet semantics scaffold` from declared .tet facts.\n");
    out.push_str(
        "# `tet semantics apply` stamps dim_names and attrs. Profiles and units stay here.\n\n",
    );
    out.push_str("schema_version = 1\n\n");
    for row in rows {
        render_dataset_toml(&mut out, row);
    }
    let profiles = overlay
        .filter(|p| !p.profiles.is_empty())
        .map_or_else(|| generated_profiles(rows), |p| p.profiles.clone());
    for (id, formulas) in &profiles {
        out.push_str(&dotted_table("profiles", id));
        out.push('\n');
        if formulas.is_empty() {
            out.push_str("# Policy formulas stay in this file (fold-inert).\n");
        }
        for (key, value) in formulas {
            out.push_str(key);
            out.push_str(" = ");
            out.push_str(&toml_string(value));
            out.push('\n');
        }
        out.push('\n');
    }
    let units = overlay
        .filter(|p| !p.units.is_empty())
        .map_or_else(|| generated_units(rows), |p| p.units.clone());
    for (unit, fields) in &units {
        out.push_str(&dotted_table("units", unit));
        out.push('\n');
        for (key, value) in fields {
            out.push_str(key);
            out.push_str(" = ");
            out.push_str(&toml_string(value));
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

fn render_dataset_toml(out: &mut String, row: &DeclareRow) {
    let missing = missing_recommended(row);
    render_dataset_declare_toml(out, &row.name, &row.facts, &missing);
}

/// Write `[datasets.<name>]` + optional attrs (and `# key =` comments for `missing`).
pub(crate) fn render_dataset_declare_toml(
    out: &mut String,
    name: &str,
    facts: &DeclareFacts,
    missing: &[&str],
) {
    let _ = writeln!(out, "[datasets.{}]", bare_or_quoted(name));
    if let Some(names) = &facts.dim_names {
        out.push_str("dim_names = [");
        for (i, dim) in names.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(&toml_string(dim));
        }
        out.push_str("]\n");
    }
    if !facts.attrs.is_empty() || !missing.is_empty() {
        let _ = writeln!(out, "[datasets.{}.attrs]", bare_or_quoted(name));
        for (key, value) in &facts.attrs {
            out.push_str(&bare_or_quoted(key));
            out.push_str(" = ");
            out.push_str(&toml_string(value));
            out.push('\n');
        }
        for key in missing {
            out.push_str("# ");
            out.push_str(key);
            out.push_str(" =\n");
        }
    }
    out.push('\n');
}

fn missing_recommended(row: &DeclareRow) -> Vec<&'static str> {
    let Some(profile) = profile_for_row(row) else {
        return Vec::new();
    };
    recommended_keys_for_profile(&profile)
        .iter()
        .copied()
        .filter(|key| !row.facts.attrs.contains_key(*key))
        .collect()
}

fn profile_for_row(row: &DeclareRow) -> Option<String> {
    if let Some(id) = row.facts.attrs.get(ATTR_SEMANTICS_PROFILE) {
        return Some(normalize_profile_id(id).to_owned());
    }
    infer_profile_id(&row.facts.attrs).map(str::to_owned)
}

fn generated_profiles(rows: &[DeclareRow]) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut profiles = BTreeMap::new();
    for id in collect_profile_ids(rows) {
        let formulas = rate_attr_for_profile(&id)
            .map(default_rate_formulas)
            .unwrap_or_default();
        profiles.insert(id, formulas);
    }
    profiles
}

fn generated_units(rows: &[DeclareRow]) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut units = BTreeMap::new();
    for unit in collect_unit_tokens(rows) {
        units.insert(unit.clone(), BTreeMap::from([("display".to_owned(), unit)]));
    }
    units
}

pub(crate) fn dotted_table(prefix: &str, id: &str) -> String {
    let body = id
        .split('.')
        .map(bare_or_quoted)
        .collect::<Vec<_>>()
        .join(".");
    format!("[{prefix}.{body}]")
}

pub(crate) fn bare_or_quoted(segment: &str) -> String {
    if !segment.is_empty()
        && segment
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        segment.to_owned()
    } else {
        toml_string(segment)
    }
}

pub(crate) fn toml_string(value: &str) -> String {
    format!("{value:?}")
}

fn collect_profile_ids(rows: &[DeclareRow]) -> BTreeSet<String> {
    let mut profiles = BTreeSet::new();
    for row in rows {
        if let Some(id) = profile_for_row(row) {
            profiles.insert(id);
        }
    }
    profiles
}

fn collect_unit_tokens(rows: &[DeclareRow]) -> BTreeSet<String> {
    let mut units = BTreeSet::new();
    for row in rows {
        for key in ["units", "frame_rate_unit", "sample_rate_unit", "pixel_unit"] {
            if let Some(unit) = row.facts.attrs.get(key)
                && unit.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            {
                units.insert(unit.clone());
            }
        }
    }
    units
}

/// Write bare `frames` `[6, 2, 2]` / chunk `[2, 2, 2]` (values `1..24` as `f32`) with **no** footer dims/attrs.
///
/// Intended for the metadata wizard scratch loop ([THI-638](https://linear.app/thicclatka/issue/THI-638)):
/// open → see shape → name axes interactively. Prefer `fixtures/scratch/wizard.tet` via
/// `mise semantics:scratch` rather than mutating tracked fixtures.
///
/// # Errors
///
/// Returns [`CatalogError`] when the raw write fails.
pub fn write_wizard_scratch_fixture(path: &Path) -> Result<(), CatalogError> {
    write_recording_raw(path)
}

/// Write a small recording fixture with media attrs + provenance (tests / tracked fixtures).
///
/// Shape `[6, 2, 2]` / chunk `[2, 2, 2]`, dataset `frames`, values `1..24` as `f32`.
///
/// # Errors
///
/// Returns [`CatalogError`] when write or footer update fails.
pub fn write_recording_fixture(path: &Path) -> Result<(), CatalogError> {
    write_recording_raw(path)?;

    let mut attrs = BTreeMap::new();
    attrs.insert("frame_rate".into(), "30".into());
    attrs.insert("frame_rate_unit".into(), "Hz".into());
    attrs.insert(ATTR_TIME_AXIS.into(), "0".into());
    attrs.insert("media_kind".into(), "video".into());
    attrs.insert(ATTR_SEMANTICS_PROFILE.into(), MEDIA_V1.id.into());
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

fn write_recording_raw(path: &Path) -> Result<(), CatalogError> {
    use crate::catalog::{
        CHUNK_PAYLOAD_CODEC_V1, DATASET_DTYPE_TAG_V1, RawArrayWrite, write_raw_array_file,
    };

    const SHAPE: [u64; 3] = [6, 2, 2];
    const CHUNK: [u64; 3] = [2, 2, 2];
    let mut data = vec![0u8; 6 * 2 * 2 * 4];
    for (slot, n) in data.as_chunks_mut::<4>().0.iter_mut().zip(1_u32..=24) {
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
    )
}
