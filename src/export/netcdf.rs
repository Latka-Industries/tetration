//! Export a `.tet` file to NetCDF-4 (dense numeric variables).

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use crate::catalog::{DatasetRecordV1, TetFileSummaryV1};
use crate::utils::dtype::ElementDtype;

use super::common::{
    ExportError, ExportProgress, ExportReport, chunks_for_dataset, dataset_basic_export_attrs,
    dataset_dim_names, densify_dataset, element_dtype, element_dtype_wire_label,
    ensure_missing_file, finish_report, open_summary, shape_as_usize,
};

/// Export a mapped `.tet` to a NetCDF-4 file (dense numeric variables).
///
/// Slash-separated catalog names become nested groups. Footer `dim_names` become
/// NetCDF dimensions when present; otherwise synthetic per-dataset dim names are used.
/// Basic string attrs are copied onto each variable.
///
/// # Errors
///
/// Returns [`ExportError`] when the file cannot be read, `output` exists, or a dtype
/// is unsupported for NetCDF (`f16`, `u32`, `u64`).
pub fn export_tet_to_netcdf(input: &Path, output: &Path) -> Result<ExportReport, ExportError> {
    export_tet_to_netcdf_with_progress(input, output, None::<fn(ExportProgress)>)
}

/// Like [`export_tet_to_netcdf`], invoking `progress` after each dataset is written.
///
/// # Errors
///
/// Same as [`export_tet_to_netcdf`].
pub fn export_tet_to_netcdf_with_progress(
    input: &Path,
    output: &Path,
    mut progress: Option<impl FnMut(ExportProgress)>,
) -> Result<ExportReport, ExportError> {
    let started = Instant::now();
    ensure_missing_file(output)?;
    let (mmap, summary) = open_summary(input)?;
    let mut file = netcdf::create(output).map_err(|e| ExportError::Netcdf(e.to_string()))?;
    file.add_attribute("tetration_export", env!("CARGO_PKG_VERSION"))
        .map_err(|e| ExportError::Netcdf(e.to_string()))?;

    let chunks_total = summary.chunks.len() as u64;
    let mut chunks_done = 0u64;
    let mut dim_lens: HashMap<(String, String), usize> = HashMap::new();

    for (ds_id, ds) in summary.datasets.iter().enumerate() {
        let wire_id = u64::try_from(ds_id).map_err(|_| {
            ExportError::Catalog(crate::catalog::CatalogError::TooLargeForPlatform {
                field: "dataset_id",
                value: ds_id as u64,
            })
        })?;
        let chunks = chunks_for_dataset(&summary, wire_id);
        export_dataset(&mut file, &mmap, ds, &chunks, &summary, &mut dim_lens)?;
        chunks_done += chunks.len() as u64;
        if let Some(ref mut cb) = progress {
            cb(ExportProgress {
                chunks_done,
                chunks_total,
                dataset: ds.name.clone(),
            });
        }
    }

    Ok(finish_report(input, output, &summary, chunks_done, started))
}

fn export_dataset(
    file: &mut netcdf::FileMut,
    mmap: &[u8],
    ds: &DatasetRecordV1,
    chunks: &[&crate::catalog::ChunkIndexEntryV1],
    summary: &TetFileSummaryV1,
    dim_lens: &mut HashMap<(String, String), usize>,
) -> Result<(), ExportError> {
    let elem = element_dtype(ds)?;
    ensure_netcdf_dtype(ds, elem)?;
    let dense = densify_dataset(mmap, ds, chunks)?;
    let shape = shape_as_usize(&ds.shape)?;
    ensure_parent_group(file, &ds.name)?;
    let group_key = parent_group_key(&ds.name);
    let dim_names = resolve_dim_names(ds, summary);
    ensure_dimensions(file, &group_key, &dim_names, &shape, dim_lens)?;
    let dim_refs: Vec<&str> = dim_names.iter().map(String::as_str).collect();
    write_typed_variable(file, &ds.name, elem, &dim_refs, &dense)?;
    write_variable_attrs(file, &ds.name, summary, elem)?;
    Ok(())
}

fn ensure_netcdf_dtype(ds: &DatasetRecordV1, elem: ElementDtype) -> Result<(), ExportError> {
    match elem {
        ElementDtype::F16 | ElementDtype::U32 | ElementDtype::U64 => {
            Err(ExportError::UnsupportedDtype {
                name: ds.name.clone(),
                detail: format!(
                    "{} is not supported for NetCDF export",
                    element_dtype_wire_label(elem)
                ),
            })
        }
        _ => Ok(()),
    }
}

fn parent_group_key(catalog_name: &str) -> String {
    match catalog_name.rsplit_once('/') {
        Some((parent, _)) => parent.to_owned(),
        None => String::new(),
    }
}

fn ensure_parent_group(file: &mut netcdf::FileMut, catalog_name: &str) -> Result<(), ExportError> {
    let Some((parent, _)) = catalog_name.rsplit_once('/') else {
        return Ok(());
    };
    if file
        .group(parent)
        .map_err(|e| ExportError::Netcdf(e.to_string()))?
        .is_some()
    {
        return Ok(());
    }
    file.add_group(parent)
        .map_err(|e| ExportError::Netcdf(e.to_string()))?;
    Ok(())
}

fn resolve_dim_names(ds: &DatasetRecordV1, summary: &TetFileSummaryV1) -> Vec<String> {
    if let Some(names) = dataset_dim_names(summary, &ds.name)
        && names.len() == ds.shape.len()
        && names.iter().all(|n| !n.is_empty())
    {
        return names;
    }
    let leaf = ds.name.rsplit('/').next().unwrap_or("data");
    (0..ds.shape.len())
        .map(|i| format!("{leaf}_d{i}"))
        .collect()
}

fn ensure_dimensions(
    file: &mut netcdf::FileMut,
    group_key: &str,
    dim_names: &[String],
    shape: &[usize],
    dim_lens: &mut HashMap<(String, String), usize>,
) -> Result<(), ExportError> {
    for (name, &len) in dim_names.iter().zip(shape.iter()) {
        let key = (group_key.to_owned(), name.clone());
        if let Some(&existing) = dim_lens.get(&key) {
            if existing != len {
                return Err(ExportError::Netcdf(format!(
                    "dimension `{name}` length conflict in group `{group_key}`: {existing} vs {len}"
                )));
            }
            continue;
        }
        let dim_path = if group_key.is_empty() {
            name.clone()
        } else {
            format!("{group_key}/{name}")
        };
        file.add_dimension(&dim_path, len)
            .map_err(|e| ExportError::Netcdf(e.to_string()))?;
        dim_lens.insert(key, len);
    }
    Ok(())
}

fn write_typed_variable(
    file: &mut netcdf::FileMut,
    name: &str,
    elem: ElementDtype,
    dims: &[&str],
    dense: &[u8],
) -> Result<(), ExportError> {
    match elem {
        ElementDtype::F32 => put_variable::<f32>(file, name, dims, dense),
        ElementDtype::F64 => put_variable::<f64>(file, name, dims, dense),
        ElementDtype::I32 => put_variable::<i32>(file, name, dims, dense),
        ElementDtype::I64 => put_variable::<i64>(file, name, dims, dense),
        ElementDtype::U8 => put_variable::<u8>(file, name, dims, dense),
        ElementDtype::U16 => put_variable::<u16>(file, name, dims, dense),
        ElementDtype::I16 => put_variable::<i16>(file, name, dims, dense),
        ElementDtype::F16 | ElementDtype::U32 | ElementDtype::U64 => {
            Err(ExportError::UnsupportedDtype {
                name: name.to_owned(),
                detail: format!(
                    "{} is not supported for NetCDF export",
                    element_dtype_wire_label(elem)
                ),
            })
        }
    }
}

fn put_variable<T: netcdf::NcTypeDescriptor + bytemuck::Pod>(
    file: &mut netcdf::FileMut,
    name: &str,
    dims: &[&str],
    dense: &[u8],
) -> Result<(), ExportError> {
    let values: &[T] = bytemuck::try_cast_slice(dense)
        .map_err(|e| ExportError::Netcdf(format!("dense buffer cast failed for `{name}`: {e}")))?;
    let mut var = file
        .add_variable::<T>(name, dims)
        .map_err(|e| ExportError::Netcdf(e.to_string()))?;
    var.put_values(values, ..)
        .map_err(|e| ExportError::Netcdf(e.to_string()))?;
    Ok(())
}

fn write_variable_attrs(
    file: &mut netcdf::FileMut,
    name: &str,
    summary: &TetFileSummaryV1,
    elem: ElementDtype,
) -> Result<(), ExportError> {
    let mut var = file
        .variable_mut(name)
        .ok_or_else(|| ExportError::Netcdf(format!("missing variable after create: `{name}`")))?;
    for (key, value) in dataset_basic_export_attrs(summary, name) {
        var.put_attribute(key.as_str(), value.as_str())
            .map_err(|e| ExportError::Netcdf(e.to_string()))?;
    }
    var.put_attribute("tetration_dtype", element_dtype_wire_label(elem))
        .map_err(|e| ExportError::Netcdf(e.to_string()))?;
    Ok(())
}
