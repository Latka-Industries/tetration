//! CF conventions (`scale_factor`, `add_offset`, `_FillValue`) applied at import.

use crate::utils::dtype::ElementDtype;

use super::ConvertError;

/// CF numeric packing decoded during tile read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CfTransform {
    pub scale: f64,
    pub offset: f64,
    pub fill_value: Option<f64>,
}

impl CfTransform {
    pub(crate) fn apply_tile_le(
        &self,
        dtype: ElementDtype,
        buf: &mut [u8],
    ) -> Result<(), ConvertError> {
        match dtype {
            ElementDtype::F32 => {
                apply_cf_f32(self, buf);
                Ok(())
            }
            ElementDtype::F64 => {
                apply_cf_f64(self, buf);
                Ok(())
            }
            _ => Err(ConvertError::UnsupportedDtype {
                name: "cf-packed variable".into(),
                detail: "CF decode supports f32/f64 storage only".into(),
            }),
        }
    }
}

fn apply_cf_f32(cf: &CfTransform, buf: &mut [u8]) {
    for chunk in buf.as_chunks_mut::<4>().0 {
        let stored = f32::from_le_bytes(*chunk);
        let decoded = decode_cf_f32(stored, cf);
        *chunk = decoded.to_le_bytes();
    }
}

fn apply_cf_f64(cf: &CfTransform, buf: &mut [u8]) {
    for chunk in buf.as_chunks_mut::<8>().0 {
        let stored = f64::from_le_bytes(*chunk);
        let decoded = decode_cf_f64(stored, cf);
        *chunk = decoded.to_le_bytes();
    }
}

fn decode_cf_f32(stored: f32, cf: &CfTransform) -> f32 {
    if cf_fill_matches_f32(stored, cf) {
        return f32::NAN;
    }
    (f64::from(stored) * cf.scale + cf.offset) as f32
}

fn decode_cf_f64(stored: f64, cf: &CfTransform) -> f64 {
    if cf_fill_matches_f64(stored, cf) {
        return f64::NAN;
    }
    stored * cf.scale + cf.offset
}

fn cf_fill_matches_f32(stored: f32, cf: &CfTransform) -> bool {
    cf.fill_value.is_some_and(|fill| {
        let fill_f32 = fill as f32;
        stored.to_bits() == fill_f32.to_bits()
            || (stored - fill_f32).abs() <= cf_fill_tolerance(fill) as f32
    })
}

fn cf_fill_matches_f64(stored: f64, cf: &CfTransform) -> bool {
    cf.fill_value.is_some_and(|fill| {
        if stored.is_nan() && fill.is_nan() {
            return true;
        }
        (stored - fill).abs() <= cf_fill_tolerance(fill)
    })
}

fn cf_fill_tolerance(fill: f64) -> f64 {
    (fill.abs() * 1.0e-6).max(1.0e-6)
}

#[cfg(feature = "tetration-hdf5")]
pub(crate) fn cf_from_hdf5(ds: &hdf5_metno::Dataset) -> Option<CfTransform> {
    let scale = hdf5_attr_f64(ds, "scale_factor");
    let offset = hdf5_attr_f64(ds, "add_offset");
    if scale.is_none() && offset.is_none() {
        return None;
    }
    Some(CfTransform {
        scale: scale.unwrap_or(1.0),
        offset: offset.unwrap_or(0.0),
        fill_value: hdf5_attr_f64(ds, "_FillValue"),
    })
}

#[cfg(feature = "tetration-hdf5")]
fn hdf5_attr_f64(ds: &hdf5_metno::Dataset, name: &str) -> Option<f64> {
    let attr = ds.attr(name).ok()?;
    let reader = attr.as_reader();
    reader
        .read_scalar::<f64>()
        .ok()
        .or_else(|| reader.read_scalar::<f32>().ok().map(f64::from))
        .or_else(|| reader.read_scalar::<i32>().ok().map(f64::from))
}

#[cfg(feature = "tetration-netcdf")]
pub(crate) fn cf_from_netcdf(var: &netcdf::Variable<'_>) -> Option<CfTransform> {
    let scale = nc_attr_f64(var, "scale_factor");
    let offset = nc_attr_f64(var, "add_offset");
    if scale.is_none() && offset.is_none() {
        return None;
    }
    let fill = nc_attr_f64(var, "_FillValue").or_else(|| nc_fill_value_f64(var));
    Some(CfTransform {
        scale: scale.unwrap_or(1.0),
        offset: offset.unwrap_or(0.0),
        fill_value: fill,
    })
}

#[cfg(feature = "tetration-netcdf")]
fn nc_attr_f64(var: &netcdf::Variable<'_>, name: &str) -> Option<f64> {
    match var.attribute_value(name) {
        Some(Ok(value)) => value.try_into().ok(),
        _ => None,
    }
}

#[cfg(feature = "tetration-netcdf")]
fn nc_fill_value_f64(var: &netcdf::Variable<'_>) -> Option<f64> {
    var.fill_value::<f32>()
        .ok()
        .flatten()
        .map(f64::from)
        .or_else(|| var.fill_value::<f64>().ok().flatten())
}
