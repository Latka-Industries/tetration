//! One-file semantics TOML: parse declare + policy, stamp declare into a mutable `.tet`.
//!
//! `tet semantics apply` writes `dim_names`, science/media attrs, and `semantics_profile`.
//! `[profiles.*]` formulas and `[units.*]` stay in the TOML (fold-inert). The source file
//! is never rewritten. Coords are rejected in this cut.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

use super::profiles::normalize_profile_id;
use crate::catalog::history::{HistoryEvent, read_footer_blob_from_path, write_footer_blob};
use crate::catalog::metadata::DatasetMetadataV1;
use crate::catalog::{CatalogError, TetFileSummaryV1, TetMetadataV1, read_tet_summary_v1};

/// On-disk schema for `tet-semantics.toml`.
pub const SEMANTICS_TOML_SCHEMA_VERSION: i64 = 1;

/// Parsed one-file semantics document.
///
/// `profiles` and `units` are retained so callers can report what was **not** stamped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticsDocument {
    pub schema_version: u32,
    pub datasets: BTreeMap<String, DatasetDeclare>,
    pub profiles: BTreeMap<String, BTreeMap<String, String>>,
    pub units: BTreeMap<String, BTreeMap<String, String>>,
}

/// Declare section for one dataset (`dim_names` and/or attrs).
///
/// Absent `dim_names` means “do not touch axis names”. An empty `attrs` map means no attr writes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DatasetDeclare {
    pub dim_names: Option<Vec<String>>,
    pub attrs: BTreeMap<String, String>,
}

/// One dataset that received a stamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StampedDataset {
    pub dataset: String,
    pub dim_names: Option<Vec<String>>,
    /// Attr keys written, sorted.
    pub attr_keys: Vec<String>,
    /// `(source, canonical)` when `semantics_profile` was an alias.
    pub profile_normalized: Option<(String, String)>,
}

/// What `apply` wrote versus what stayed in the TOML.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticsApplyReport {
    pub stamped: Vec<StampedDataset>,
    pub profiles_left: Vec<String>,
    pub units_left: Vec<String>,
}

/// Parse `tet-semantics.toml` text.
///
/// # Errors
///
/// TOML syntax, `schema_version`, attr types, or a `coords` key.
pub fn parse_semantics_toml(text: &str) -> Result<SemanticsDocument, CatalogError> {
    let root: toml::Value = toml::from_str(text)
        .map_err(|e| CatalogError::InvalidSemantics(format!("tet-semantics.toml parse: {e}")))?;
    let table = root.as_table().ok_or_else(|| {
        CatalogError::InvalidSemantics("tet-semantics.toml root must be a table".to_owned())
    })?;
    for key in table.keys() {
        if !matches!(
            key.as_str(),
            "schema_version" | "datasets" | "profiles" | "units"
        ) {
            return Err(CatalogError::InvalidSemantics(format!(
                "unknown top-level key {key:?}; expected schema_version, datasets, profiles, units"
            )));
        }
    }
    let schema_version = require_schema_version(table.get("schema_version"))?;
    let datasets = match table.get("datasets") {
        None => BTreeMap::new(),
        Some(v) => parse_datasets(v)?,
    };
    let profiles = match table.get("profiles") {
        None => BTreeMap::new(),
        Some(v) => parse_profiles(v)?,
    };
    let units = match table.get("units") {
        None => BTreeMap::new(),
        Some(v) => parse_units(v)?,
    };
    Ok(SemanticsDocument {
        schema_version,
        datasets,
        profiles,
        units,
    })
}

/// Stamp declare facts from `toml_path` onto a mutable `.tet`.
///
/// Does not modify `toml_path`. Does not write profiles or units into the footer.
///
/// # Errors
///
/// Sealed package targets, missing datasets, rank / `time_axis` mismatches, `coords`,
/// or I/O.
pub fn apply_semantics_toml(
    tet_path: &Path,
    toml_path: &Path,
) -> Result<SemanticsApplyReport, CatalogError> {
    ensure_mutable_tet(tet_path)?;
    let text = fs::read_to_string(toml_path)?;
    let doc = parse_semantics_toml(&text)?;
    let summary = read_summary_from_path(tet_path)?;
    let stamped = plan_stamps(&summary, &doc)?;
    if !stamped.is_empty() {
        write_stamps(tet_path, toml_path, &doc, &stamped)?;
    }
    Ok(SemanticsApplyReport {
        stamped,
        profiles_left: doc.profiles.keys().cloned().collect(),
        units_left: doc.units.keys().cloned().collect(),
    })
}

/// CLI / log text: what was stamped, and which profile / unit tables stayed in the TOML.
#[must_use]
pub fn format_apply_report(tet_path: &Path, report: &SemanticsApplyReport) -> String {
    let mut out = String::new();
    if report.stamped.is_empty() {
        let _ = writeln!(out, "semantics: stamped nothing on {}", tet_path.display());
    } else {
        let _ = writeln!(out, "semantics: stamped {}", tet_path.display());
        for row in &report.stamped {
            out.push_str("  dataset ");
            out.push_str(&row.dataset);
            out.push(':');
            if let Some(names) = &row.dim_names {
                out.push_str(" dim_names=[");
                out.push_str(&names.join(", "));
                out.push(']');
            }
            if !row.attr_keys.is_empty() {
                if row.dim_names.is_some() {
                    out.push(';');
                }
                out.push_str(" attrs ");
                out.push_str(&row.attr_keys.join(", "));
            }
            if let Some((from, to)) = &row.profile_normalized {
                let _ = write!(out, "; normalized semantics_profile {from} → {to}");
            }
            out.push('\n');
        }
    }
    out.push_str("semantics: left in TOML (not written to the footer): ");
    if report.profiles_left.is_empty() && report.units_left.is_empty() {
        out.push_str("none\n");
    } else {
        let mut parts = Vec::new();
        if !report.profiles_left.is_empty() {
            parts.push(format!("profiles {}", report.profiles_left.join(", ")));
        }
        if !report.units_left.is_empty() {
            parts.push(format!("units {}", report.units_left.join(", ")));
        }
        out.push_str(&parts.join("; "));
        out.push('\n');
    }
    out
}

fn require_schema_version(value: Option<&toml::Value>) -> Result<u32, CatalogError> {
    let Some(value) = value else {
        return Err(CatalogError::InvalidSemantics(
            "schema_version = 1 is required".to_owned(),
        ));
    };
    match value {
        toml::Value::Integer(v) if *v == SEMANTICS_TOML_SCHEMA_VERSION => Ok(1),
        toml::Value::Integer(v) => Err(CatalogError::InvalidSemantics(format!(
            "schema_version {v} is unsupported (want {SEMANTICS_TOML_SCHEMA_VERSION})"
        ))),
        _ => Err(CatalogError::InvalidSemantics(
            "schema_version must be the integer 1".to_owned(),
        )),
    }
}

fn parse_datasets(value: &toml::Value) -> Result<BTreeMap<String, DatasetDeclare>, CatalogError> {
    let table = expect_table(value, "datasets")?;
    let mut out = BTreeMap::new();
    for (name, body) in table {
        if name.is_empty() {
            return Err(CatalogError::InvalidSemantics(
                "dataset name must be non-empty".to_owned(),
            ));
        }
        out.insert(name.clone(), parse_dataset(name, body)?);
    }
    Ok(out)
}

fn parse_dataset(name: &str, value: &toml::Value) -> Result<DatasetDeclare, CatalogError> {
    let table = expect_table(value, &format!("datasets.{name}"))?;
    let mut decl = DatasetDeclare::default();
    for (key, body) in table {
        match key.as_str() {
            "dim_names" => decl.dim_names = Some(parse_dim_names(name, body)?),
            "attrs" => decl.attrs = parse_attrs(name, body)?,
            "coords" => return Err(coords_deferred(name)),
            other => {
                return Err(CatalogError::InvalidSemantics(format!(
                    "unknown field {other:?} on dataset {name}; this cut accepts dim_names and attrs"
                )));
            }
        }
    }
    Ok(decl)
}

fn coords_deferred(dataset: &str) -> CatalogError {
    CatalogError::InvalidSemantics(format!(
        "coords are deferred and are not applied in this cut; remove coords from dataset {dataset}"
    ))
}

fn parse_dim_names(dataset: &str, value: &toml::Value) -> Result<Vec<String>, CatalogError> {
    let Some(items) = value.as_array() else {
        return Err(CatalogError::InvalidSemantics(format!(
            "datasets.{dataset}.dim_names must be an array of strings"
        )));
    };
    let mut names = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let Some(name) = item.as_str() else {
            return Err(CatalogError::InvalidSemantics(format!(
                "datasets.{dataset}.dim_names[{i}] must be a string"
            )));
        };
        if name.is_empty() {
            return Err(CatalogError::InvalidSemantics(format!(
                "datasets.{dataset}.dim_names[{i}] must be non-empty"
            )));
        }
        names.push(name.to_owned());
    }
    Ok(names)
}

fn parse_attrs(
    dataset: &str,
    value: &toml::Value,
) -> Result<BTreeMap<String, String>, CatalogError> {
    let table = expect_table(value, &format!("datasets.{dataset}.attrs"))?;
    let mut attrs = BTreeMap::new();
    for (key, body) in table {
        if key == "coords" {
            return Err(coords_deferred(dataset));
        }
        if key.is_empty() {
            return Err(CatalogError::InvalidSemantics(format!(
                "datasets.{dataset} attr key must be non-empty"
            )));
        }
        let coerced = coerce_attr(dataset, key, body)?;
        attrs.insert(key.clone(), coerced);
    }
    Ok(attrs)
}

fn coerce_attr(dataset: &str, key: &str, value: &toml::Value) -> Result<String, CatalogError> {
    let where_ = format!("datasets.{dataset}.attrs.{key}");
    let text = match value {
        toml::Value::String(s) => s.clone(),
        toml::Value::Integer(i) => i.to_string(),
        toml::Value::Float(f) => {
            if !f.is_finite() {
                return Err(CatalogError::InvalidSemantics(format!(
                    "{where_} number must be finite"
                )));
            }
            format!("{f}")
        }
        toml::Value::Boolean(b) => b.to_string(),
        toml::Value::Array(_) | toml::Value::Table(_) => {
            return Err(CatalogError::InvalidSemantics(format!(
                "{where_} must be a string, number, or bool (arrays and tables are rejected)"
            )));
        }
        toml::Value::Datetime(_) => {
            return Err(CatalogError::InvalidSemantics(format!(
                "{where_} must be a string, number, or bool"
            )));
        }
    };
    if text.is_empty() {
        return Err(CatalogError::InvalidSemantics(format!(
            "{where_} must be non-empty"
        )));
    }
    Ok(text)
}

fn parse_profiles(
    value: &toml::Value,
) -> Result<BTreeMap<String, BTreeMap<String, String>>, CatalogError> {
    let table = expect_table(value, "profiles")?;
    let mut out = BTreeMap::new();
    walk_profiles(table, "", &mut out)?;
    Ok(out)
}

fn walk_profiles(
    table: &toml::map::Map<String, toml::Value>,
    prefix: &str,
    out: &mut BTreeMap<String, BTreeMap<String, String>>,
) -> Result<(), CatalogError> {
    for (key, value) in table {
        let id = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        let Some(inner) = value.as_table() else {
            return Err(CatalogError::InvalidSemantics(format!(
                "profiles.{id} must be a table"
            )));
        };
        if is_profile_leaf(inner) {
            out.insert(id.clone(), profile_formulas(&id, inner)?);
        } else {
            walk_profiles(inner, &id, out)?;
        }
    }
    Ok(())
}

fn is_profile_leaf(table: &toml::map::Map<String, toml::Value>) -> bool {
    table.values().all(|v| !matches!(v, toml::Value::Table(_)))
}

fn profile_formulas(
    id: &str,
    table: &toml::map::Map<String, toml::Value>,
) -> Result<BTreeMap<String, String>, CatalogError> {
    let mut formulas = BTreeMap::new();
    for (key, value) in table {
        let Some(text) = value.as_str() else {
            return Err(CatalogError::InvalidSemantics(format!(
                "profiles.{id}.{key} must be a string (policy formulas stay in the TOML)"
            )));
        };
        formulas.insert(key.clone(), text.to_owned());
    }
    Ok(formulas)
}

fn parse_units(
    value: &toml::Value,
) -> Result<BTreeMap<String, BTreeMap<String, String>>, CatalogError> {
    let table = expect_table(value, "units")?;
    let mut out = BTreeMap::new();
    for (name, body) in table {
        let inner = expect_table(body, &format!("units.{name}"))?;
        let mut fields = BTreeMap::new();
        for (key, field) in inner {
            let text = match field {
                toml::Value::String(s) => s.clone(),
                toml::Value::Integer(i) => i.to_string(),
                toml::Value::Float(f) if f.is_finite() => format!("{f}"),
                toml::Value::Boolean(b) => b.to_string(),
                _ => {
                    return Err(CatalogError::InvalidSemantics(format!(
                        "units.{name}.{key} must be a string, number, or bool"
                    )));
                }
            };
            fields.insert(key.clone(), text);
        }
        out.insert(name.clone(), fields);
    }
    Ok(out)
}

fn expect_table<'a>(
    value: &'a toml::Value,
    what: &str,
) -> Result<&'a toml::map::Map<String, toml::Value>, CatalogError> {
    value
        .as_table()
        .ok_or_else(|| CatalogError::InvalidSemantics(format!("{what} must be a table")))
}

fn ensure_mutable_tet(path: &Path) -> Result<(), CatalogError> {
    if path.is_dir() || path.join("PROTOCOL").is_file() || ancestor_is_package(path) {
        return Err(CatalogError::InvalidSemantics(
            "semantics apply targets a mutable .tet file; a sealed .tet.d generation is immutable — apply on the mutable .tet, then seal a new generation"
                .to_owned(),
        ));
    }
    Ok(())
}

fn ancestor_is_package(path: &Path) -> bool {
    let mut cur = path.parent();
    while let Some(dir) = cur {
        if dir.join("PROTOCOL").is_file() {
            return true;
        }
        cur = dir.parent();
    }
    false
}

fn read_summary_from_path(path: &Path) -> Result<TetFileSummaryV1, CatalogError> {
    let data = fs::read(path)?;
    read_tet_summary_v1(&data)
}

fn plan_stamps(
    summary: &TetFileSummaryV1,
    doc: &SemanticsDocument,
) -> Result<Vec<StampedDataset>, CatalogError> {
    let mut stamped = Vec::new();
    for (name, decl) in &doc.datasets {
        let rank = dataset_rank(summary, name)?;
        if let Some(names) = &decl.dim_names
            && names.len() != rank
        {
            return Err(CatalogError::InvalidSemantics(format!(
                "dataset {name}: dim_names has length {} but rank is {rank}",
                names.len()
            )));
        }
        if let Some(axis) = decl.attrs.get("time_axis") {
            check_time_axis(name, axis, rank)?;
        }
        if decl.dim_names.is_none() && decl.attrs.is_empty() {
            continue;
        }
        let (_, profile_normalized) = normalize_profile_attr(&decl.attrs);
        stamped.push(StampedDataset {
            dataset: name.clone(),
            dim_names: decl.dim_names.clone(),
            attr_keys: decl.attrs.keys().cloned().collect(),
            profile_normalized,
        });
    }
    Ok(stamped)
}

fn normalize_profile_attr(
    attrs: &BTreeMap<String, String>,
) -> (BTreeMap<String, String>, Option<(String, String)>) {
    let mut out = attrs.clone();
    let mut note = None;
    if let Some(raw) = out.get("semantics_profile").cloned() {
        let canonical = normalize_profile_id(&raw).to_owned();
        if canonical != raw {
            out.insert("semantics_profile".to_owned(), canonical.clone());
            note = Some((raw, canonical));
        }
    }
    (out, note)
}

fn dataset_rank(summary: &TetFileSummaryV1, name: &str) -> Result<usize, CatalogError> {
    summary
        .datasets
        .iter()
        .find(|d| d.name == name)
        .map(|d| d.shape.len())
        .ok_or_else(|| {
            CatalogError::InvalidSemantics(format!("dataset {name} not found in catalog"))
        })
}

fn check_time_axis(dataset: &str, raw: &str, rank: usize) -> Result<(), CatalogError> {
    let axis: i64 = raw.parse().map_err(|_| {
        CatalogError::InvalidSemantics(format!(
            "dataset {dataset}: time_axis {raw:?} must be an integer axis index"
        ))
    })?;
    if axis < 0 {
        return Err(CatalogError::InvalidSemantics(format!(
            "dataset {dataset}: time_axis {axis} is outside rank {rank}"
        )));
    }
    let axis_u = usize::try_from(axis).map_err(|_| {
        CatalogError::InvalidSemantics(format!(
            "dataset {dataset}: time_axis {axis} is outside rank {rank}"
        ))
    })?;
    if axis_u >= rank {
        return Err(CatalogError::InvalidSemantics(format!(
            "dataset {dataset}: time_axis {axis} is outside rank {rank}"
        )));
    }
    Ok(())
}

fn write_stamps(
    tet_path: &Path,
    toml_path: &Path,
    doc: &SemanticsDocument,
    stamped: &[StampedDataset],
) -> Result<(), CatalogError> {
    let mut blob = read_footer_blob_from_path(tet_path).unwrap_or_default();
    let meta = blob.metadata.get_or_insert_with(TetMetadataV1::default);
    for row in stamped {
        let decl = doc.datasets.get(&row.dataset).ok_or_else(|| {
            CatalogError::InvalidSemantics("internal: missing declare for stamp".to_owned())
        })?;
        let (attrs, _) = normalize_profile_attr(&decl.attrs);
        stamp_dataset(meta, row, decl, &attrs);
        blob.history.push(history_event(tet_path, toml_path, row));
    }
    if meta.file.is_none() && meta.datasets.is_empty() {
        blob.metadata = None;
    }
    write_footer_blob(tet_path, &blob)
}

fn stamp_dataset(
    meta: &mut TetMetadataV1,
    row: &StampedDataset,
    decl: &DatasetDeclare,
    attrs: &BTreeMap<String, String>,
) {
    let empty = {
        let ds = meta.dataset_mut(&row.dataset);
        if let Some(names) = &decl.dim_names {
            ds.dim_names = Some(names.clone());
        }
        for (key, value) in attrs {
            ds.attrs.insert(key.clone(), value.clone());
        }
        DatasetMetadataV1::import_is_empty(&ds.attrs, ds.dim_names.as_ref(), ds.coords.as_ref())
    };
    if empty {
        meta.datasets.remove(&row.dataset);
    }
}

fn history_event(tet_path: &Path, toml_path: &Path, row: &StampedDataset) -> HistoryEvent {
    let mut event = HistoryEvent::new("semantics", tet_path.display().to_string());
    event.params.insert("dataset".into(), row.dataset.clone());
    event.params.insert("via".into(), "apply".into());
    event
        .params
        .insert("from".into(), toml_path.display().to_string());
    if let Some(names) = &row.dim_names {
        event.params.insert("dim_names".into(), names.join(","));
    }
    if !row.attr_keys.is_empty() {
        event.params.insert("set".into(), row.attr_keys.join(","));
    }
    event
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANONICAL: &str = r#"
schema_version = 1

[datasets.frames]
dim_names = ["t", "y", "x"]

[datasets.frames.attrs]
semantics_profile = "media.v1"
media_kind = "video"
frame_rate = 30
frame_rate_unit = "Hz"
time_axis = 0
pixel_unit = "px"
flag = true

[profiles.media.v1]
index_to_seconds = "index / frame_rate"
seconds_to_index = "floor(seconds * frame_rate)"

[units.Hz]
display = "Hz"
"#;

    #[test]
    fn parses_canonical_one_file_and_coerces_scalars() {
        let doc = parse_semantics_toml(CANONICAL).unwrap();
        assert_eq!(doc.schema_version, 1);
        let frames = &doc.datasets["frames"];
        assert_eq!(
            frames.dim_names.as_deref(),
            Some(["t".to_owned(), "y".to_owned(), "x".to_owned()].as_slice())
        );
        assert_eq!(
            frames.attrs.get("frame_rate").map(String::as_str),
            Some("30")
        );
        assert_eq!(frames.attrs.get("time_axis").map(String::as_str), Some("0"));
        assert_eq!(frames.attrs.get("flag").map(String::as_str), Some("true"));
        assert_eq!(
            doc.profiles["media.v1"]
                .get("index_to_seconds")
                .map(String::as_str),
            Some("index / frame_rate")
        );
        assert_eq!(
            doc.units["Hz"].get("display").map(String::as_str),
            Some("Hz")
        );
    }

    #[test]
    fn rejects_coords_arrays_and_bad_schema() {
        let coords = "schema_version = 1\n[datasets.frames]\ncoords = [\"a\"]\n";
        let err = parse_semantics_toml(coords).unwrap_err().to_string();
        assert!(err.contains("coords"), "{err}");

        let array = "schema_version = 1\n[datasets.frames.attrs]\ntags = [\"a\"]\n";
        let err = parse_semantics_toml(array).unwrap_err().to_string();
        assert!(err.contains("arrays and tables"), "{err}");

        let ver = "schema_version = 2\n";
        let err = parse_semantics_toml(ver).unwrap_err().to_string();
        assert!(err.contains("unsupported"), "{err}");
    }
}
