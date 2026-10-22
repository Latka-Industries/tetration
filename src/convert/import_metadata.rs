//! Map foreign dataset attributes into footer `metadata` on convert.

use std::collections::BTreeMap;
use std::path::Path;

use crate::catalog::{
    CatalogError, CoordAxisV1, DatasetMetadataV1, EMBEDDED_STANDALONE_GENERATION, FileMetadataV1,
    FooterBlobV1, HistoryEvent, MetadataLimitsV1, TetMetadataV1, embed_indexes_in_tet,
    unix_timestamp_now, write_footer_blob,
};

use super::shared::ImportPlan;

/// CF / discovery attribute names imported first when present.
const PREFERRED_ATTRS: &[&str] = &[
    "units",
    "long_name",
    "standard_name",
    "description",
    "title",
    "scale_factor",
    "add_offset",
    "_FillValue",
];

/// Write convert history plus optional dataset metadata into the `THST` footer.
///
/// # Errors
///
/// Returns [`CatalogError`] when metadata validation or footer I/O fails.
pub fn finish_convert_footer(
    output: &Path,
    source: &str,
    plans: &[ImportPlan],
) -> Result<Vec<HistoryEvent>, CatalogError> {
    let metadata = build_convert_metadata(plans)?;
    let event = HistoryEvent::new("convert", source);
    write_footer_blob(
        output,
        &FooterBlobV1 {
            history: vec![event.clone()],
            metadata,
            metadata_ref: None,
        },
    )?;
    // Standalone convert output must carry ZONE/INDEX for pushdown (0.7.0).
    embed_indexes_in_tet(output, EMBEDDED_STANDALONE_GENERATION)?;
    Ok(vec![event])
}

/// Merge per-plan import fields into footer metadata (file tool stamp + dataset entries).
fn build_convert_metadata(plans: &[ImportPlan]) -> Result<Option<TetMetadataV1>, CatalogError> {
    let mut meta = TetMetadataV1 {
        file: Some(FileMetadataV1 {
            tool: Some("tet convert".to_owned()),
            library_version: Some(env!("CARGO_PKG_VERSION").to_owned()),
            created_at: Some(unix_timestamp_now()),
        }),
        ..TetMetadataV1::default()
    };
    for plan in plans {
        if DatasetMetadataV1::import_is_empty(
            &plan.import_attrs,
            plan.import_dim_names.as_ref(),
            plan.import_coords.as_ref(),
        ) {
            continue;
        }
        meta.dataset_mut(&plan.name).apply_import(
            &plan.import_attrs,
            plan.import_dim_names.as_ref(),
            plan.import_coords.as_ref(),
        );
    }
    meta.validate()?;
    Ok(Some(meta))
}

#[cfg(feature = "tetration-hdf5")]
pub(crate) fn hdf5_dataset_attrs(ds: &hdf5_metno::Dataset) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Ok(names) = ds.attr_names() else {
        return out;
    };
    import_named_attrs(&mut out, &names, |name| hdf5_attr_string(ds, name));
    out
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_attr_string(ds: &hdf5_metno::Dataset, name: &str) -> Option<String> {
    use hdf5_metno::types::{VarLenAscii, VarLenUnicode};

    let attr = ds.attr(name).ok()?;
    let reader = attr.as_reader();
    if let Ok(s) = reader.read_scalar::<VarLenUnicode>() {
        return non_empty_string(s.as_str().to_owned());
    }
    if let Ok(s) = reader.read_scalar::<VarLenAscii>() {
        return non_empty_string(s.as_str().to_owned());
    }
    if let Ok(dt) = attr.dtype()
        && let Ok(td) = dt.to_descriptor()
        && let Some(s) = hdf5_attr_string_fixed(&reader, &td)
    {
        return non_empty_string(s);
    }
    if let Ok(f) = reader.read_scalar::<f32>() {
        return Some(f.to_string());
    }
    if let Ok(f) = reader.read_scalar::<f64>() {
        return Some(f.to_string());
    }
    if let Ok(n) = reader.read_scalar::<i32>() {
        return Some(n.to_string());
    }
    if let Ok(n) = reader.read_scalar::<i64>() {
        return Some(n.to_string());
    }
    None
}

#[cfg(feature = "tetration-netcdf")]
pub(crate) fn netcdf_variable_attrs(var: &netcdf::Variable<'_>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let names: Vec<String> = var.attributes().map(|a| a.name().to_owned()).collect();
    import_named_attrs(&mut out, &names, |name| {
        var.attribute_value(name)
            .and_then(Result::ok)
            .and_then(nc_attr_value_to_string)
    });
    out
}

#[cfg(feature = "tetration-netcdf")]
pub(crate) fn netcdf_dim_names(var: &netcdf::Variable<'_>) -> Option<Vec<String>> {
    let names: Vec<String> = var
        .dimensions()
        .iter()
        .map(netcdf::Dimension::name)
        .filter(|n| !n.is_empty())
        .collect();
    if names.is_empty() {
        return None;
    }
    let limits = crate::catalog::MetadataLimitsV1::DEFAULT;
    if names.len() > limits.dim_names {
        return None;
    }
    for n in &names {
        if n.len() > limits.attr_string_bytes {
            return None;
        }
    }
    Some(names)
}

#[cfg(feature = "tetration-netcdf")]
fn nc_attr_value_to_string(value: netcdf::AttributeValue) -> Option<String> {
    use netcdf::AttributeValue;
    let s = match value {
        AttributeValue::Str(s) => s,
        AttributeValue::Strs(v) => v.into_iter().next()?,
        AttributeValue::Uchar(x) => return Some(x.to_string()),
        AttributeValue::Schar(x) => return Some(x.to_string()),
        AttributeValue::Ushort(x) => return Some(x.to_string()),
        AttributeValue::Short(x) => return Some(x.to_string()),
        AttributeValue::Uint(x) => return Some(x.to_string()),
        AttributeValue::Int(x) => return Some(x.to_string()),
        AttributeValue::Ulonglong(x) => return Some(x.to_string()),
        AttributeValue::Longlong(x) => return Some(x.to_string()),
        AttributeValue::Float(x) => return Some(x.to_string()),
        AttributeValue::Double(x) => return Some(x.to_string()),
        _ => return None,
    };
    non_empty_string(s)
}

fn import_named_attrs(
    out: &mut BTreeMap<String, String>,
    names: &[String],
    mut read: impl FnMut(&str) -> Option<String>,
) {
    for key in PREFERRED_ATTRS {
        if names.iter().any(|n| n == *key)
            && let Some(v) = read(key)
        {
            out.insert((*key).to_owned(), v);
        }
    }
    let limits = crate::catalog::MetadataLimitsV1::DEFAULT;
    for name in names {
        if out.len() >= limits.dataset_attrs {
            break;
        }
        if PREFERRED_ATTRS.contains(&name.as_str()) {
            continue;
        }
        if name.len() > limits.attr_string_bytes {
            continue;
        }
        if let Some(v) = read(name)
            && v.len() <= limits.attr_string_bytes
        {
            out.insert(name.clone(), v);
        }
    }
}

fn non_empty_string(s: String) -> Option<String> {
    if s.is_empty() { None } else { Some(s) }
}

/// Zarr v3 array `attributes` object from `zarr.json` → footer metadata attrs.
pub(crate) fn zarr_array_attrs(
    attributes: &serde_json::Map<String, serde_json::Value>,
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let names: Vec<String> = attributes.keys().cloned().collect();
    import_named_attrs(&mut out, &names, |name| {
        attributes.get(name).and_then(json_attr_value_to_string)
    });
    out
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_attr_string_fixed(
    reader: &hdf5_metno::Reader<'_>,
    td: &hdf5_metno::types::TypeDescriptor,
) -> Option<String> {
    use hdf5_metno::types::TypeDescriptor;
    match td {
        TypeDescriptor::FixedAscii(n) => try_hdf5_fixed_ascii(reader, *n),
        TypeDescriptor::FixedUnicode(n) => try_hdf5_fixed_unicode(reader, *n),
        _ => None,
    }
}

#[cfg(feature = "tetration-hdf5")]
macro_rules! try_hdf5_fixed_string {
    ($reader:expr, $cap:expr, $fixed:ident, $($n:literal),+ $(,)?) => {{
        use hdf5_metno::types::$fixed;
        $(
            if $cap == $n {
                if let Ok(v) = $reader.read_scalar::<$fixed<$n>>() {
                    return Some(v.as_str().to_owned());
                }
            }
        )+
        None
    }};
}

#[cfg(feature = "tetration-hdf5")]
macro_rules! try_hdf5_fixed_caps {
    ($reader:expr, $cap:expr, $fixed:ident) => {
        try_hdf5_fixed_string!(
            $reader, $cap, $fixed, 8, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 512,
            1024, 2048, 4096
        )
    };
}

#[cfg(feature = "tetration-hdf5")]
fn try_hdf5_fixed_ascii(reader: &hdf5_metno::Reader<'_>, cap: usize) -> Option<String> {
    try_hdf5_fixed_caps!(reader, cap, FixedAscii)
}

#[cfg(feature = "tetration-hdf5")]
fn try_hdf5_fixed_unicode(reader: &hdf5_metno::Reader<'_>, cap: usize) -> Option<String> {
    try_hdf5_fixed_caps!(reader, cap, FixedUnicode)
}

#[cfg(feature = "tetration-hdf5")]
pub(crate) fn enrich_hdf5_cf_coordinates(file: &hdf5_metno::File, plans: &mut [ImportPlan]) {
    for plan in plans {
        let Some(coord_list) = plan.import_attrs.get("coordinates") else {
            continue;
        };
        let mut coords = BTreeMap::new();
        for axis in coord_list.split_whitespace() {
            let Some(ds) = resolve_hdf5_coord_dataset(file, axis) else {
                continue;
            };
            if let Some(axis_coords) = hdf5_1d_coord_labels(&ds) {
                coords.insert(axis.to_owned(), axis_coords);
            }
        }
        if !coords.is_empty() {
            plan.import_coords = Some(coords);
        }
    }
}

#[cfg(feature = "tetration-hdf5")]
fn resolve_hdf5_coord_dataset(file: &hdf5_metno::File, axis: &str) -> Option<hdf5_metno::Dataset> {
    for path in [format!("coordinates/{axis}"), axis.to_owned()] {
        if let Ok(ds) = file.dataset(&path) {
            return Some(ds);
        }
    }
    None
}

#[cfg(feature = "tetration-hdf5")]
pub(crate) fn hdf5_1d_coord_labels(ds: &hdf5_metno::Dataset) -> Option<CoordAxisV1> {
    let shape = ds.shape();
    if shape.len() != 1 {
        return None;
    }
    let n = shape[0];
    if !coord_axis_len_ok(n) {
        return None;
    }
    let dtype = super::hdf5_shared::element_dtype_from_hdf5_dataset(ds)?;
    let labels = hdf5_read_1d_labels(ds, dtype)?;
    coord_axis_from_labels(labels, n)
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_read_1d_labels(
    ds: &hdf5_metno::Dataset,
    dtype: crate::utils::dtype::ElementDtype,
) -> Option<Vec<String>> {
    use crate::utils::dtype::ElementDtype;
    Some(match dtype {
        ElementDtype::F32 => ds
            .read_raw::<f32>()
            .ok()?
            .into_iter()
            .map(label_from_display)
            .collect(),
        ElementDtype::F64 => ds
            .read_raw::<f64>()
            .ok()?
            .into_iter()
            .map(label_from_display)
            .collect(),
        ElementDtype::I32 => ds
            .read_raw::<i32>()
            .ok()?
            .into_iter()
            .map(label_from_display)
            .collect(),
        ElementDtype::I64 => ds
            .read_raw::<i64>()
            .ok()?
            .into_iter()
            .map(label_from_display)
            .collect(),
        ElementDtype::U8 => ds
            .read_raw::<u8>()
            .ok()?
            .into_iter()
            .map(label_from_display)
            .collect(),
        ElementDtype::U16 => ds
            .read_raw::<u16>()
            .ok()?
            .into_iter()
            .map(label_from_display)
            .collect(),
        ElementDtype::I16 => ds
            .read_raw::<i16>()
            .ok()?
            .into_iter()
            .map(label_from_display)
            .collect(),
        ElementDtype::U32 => ds
            .read_raw::<u32>()
            .ok()?
            .into_iter()
            .map(label_from_display)
            .collect(),
        ElementDtype::U64 => ds
            .read_raw::<u64>()
            .ok()?
            .into_iter()
            .map(label_from_display)
            .collect(),
        ElementDtype::F16 => ds
            .read_raw::<u16>()
            .ok()?
            .into_iter()
            .map(|bits| label_from_display(f64::from(half::f16::from_bits(bits))))
            .collect(),
    })
}

#[cfg(feature = "tetration-netcdf")]
pub(crate) fn netcdf_self_import_coords(
    name: &str,
    var: &netcdf::Variable<'_>,
) -> Option<BTreeMap<String, CoordAxisV1>> {
    netcdf_inline_coord_labels(var).map(|c| coord_axis_map(name, c))
}

#[cfg(feature = "tetration-netcdf")]
pub(crate) fn netcdf_inline_coord_labels(var: &netcdf::Variable<'_>) -> Option<CoordAxisV1> {
    use netcdf::types::{FloatType, NcVariableType};
    if var.dimensions().len() != 1 {
        return None;
    }
    let n = var.dimensions()[0].len();
    if !coord_axis_len_ok(n) {
        return None;
    }
    let labels: Vec<String> = match var.vartype() {
        NcVariableType::Float(FloatType::F32) => (0..n)
            .filter_map(|i| var.get_value::<f32, _>(i).ok().map(label_from_display))
            .collect(),
        NcVariableType::Float(FloatType::F64) => (0..n)
            .filter_map(|i| var.get_value::<f64, _>(i).ok().map(label_from_display))
            .collect(),
        NcVariableType::Int(netcdf::types::IntType::I32) => (0..n)
            .filter_map(|i| var.get_value::<i32, _>(i).ok().map(label_from_display))
            .collect(),
        NcVariableType::Int(netcdf::types::IntType::I64) => (0..n)
            .filter_map(|i| var.get_value::<i64, _>(i).ok().map(label_from_display))
            .collect(),
        NcVariableType::Int(netcdf::types::IntType::U8) => (0..n)
            .filter_map(|i| var.get_value::<u8, _>(i).ok().map(label_from_display))
            .collect(),
        NcVariableType::Int(netcdf::types::IntType::U16) => (0..n)
            .filter_map(|i| var.get_value::<u16, _>(i).ok().map(label_from_display))
            .collect(),
        NcVariableType::Int(netcdf::types::IntType::I16) => (0..n)
            .filter_map(|i| var.get_value::<i16, _>(i).ok().map(label_from_display))
            .collect(),
        _ => return None,
    };
    coord_axis_from_labels(labels, n)
}

fn coord_axis_map(name: &str, axis: CoordAxisV1) -> BTreeMap<String, CoordAxisV1> {
    let mut m = BTreeMap::new();
    m.insert(name.to_owned(), axis);
    m
}

fn coord_axis_len_ok(n: usize) -> bool {
    let limits = MetadataLimitsV1::DEFAULT;
    n > 0 && n <= limits.coord_labels_per_axis
}

fn coord_axis_from_labels(labels: Vec<String>, expected_len: usize) -> Option<CoordAxisV1> {
    (labels.len() == expected_len).then_some(CoordAxisV1 { labels })
}

fn label_from_display(v: impl std::fmt::Display) -> String {
    trim_coord_label(&v.to_string())
}

fn trim_coord_label(s: &str) -> String {
    if s.len() > MetadataLimitsV1::DEFAULT.attr_string_bytes {
        s[..MetadataLimitsV1::DEFAULT.attr_string_bytes].to_owned()
    } else {
        s.to_owned()
    }
}

fn json_attr_value_to_string(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => non_empty_string(s.clone()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Null | serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            None
        }
    }
}
