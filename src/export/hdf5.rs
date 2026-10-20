//! Export a `.tet` file to HDF5 (dense numeric datasets).

use std::path::Path;
use std::time::Instant;

use hdf5_metno::File;

use crate::catalog::{DatasetRecordV1, TetFileSummaryV1};
use crate::utils::dtype::ElementDtype;

use super::common::{
    ExportError, ExportProgress, ExportReport, chunks_for_dataset, dataset_basic_export_attrs,
    dataset_dim_names, densify_dataset, element_dtype, element_dtype_wire_label,
    ensure_missing_file, finish_report, open_summary, shape_as_usize,
};

/// Export a mapped `.tet` to an HDF5 file (dense numeric datasets).
///
/// Slash-separated catalog names become nested groups. Footer attrs and `dim_names`
/// (when present) are written as string attributes.
///
/// # Errors
///
/// Returns [`ExportError`] when the file cannot be read, `output` exists, or a dtype
/// is unsupported for HDF5 (notably `f16`).
pub fn export_tet_to_hdf5(input: &Path, output: &Path) -> Result<ExportReport, ExportError> {
    export_tet_to_hdf5_with_progress(input, output, None::<fn(ExportProgress)>)
}

/// Like [`export_tet_to_hdf5`], invoking `progress` after each dataset is written.
///
/// # Errors
///
/// Same as [`export_tet_to_hdf5`].
pub fn export_tet_to_hdf5_with_progress(
    input: &Path,
    output: &Path,
    mut progress: Option<impl FnMut(ExportProgress)>,
) -> Result<ExportReport, ExportError> {
    let started = Instant::now();
    ensure_missing_file(output)?;
    let (mmap, summary) = open_summary(input)?;
    let file = File::create(output)?;
    file.new_attr_builder()
        .with_data(env!("CARGO_PKG_VERSION"))
        .create("tetration_export")?;

    let chunks_total = summary.chunks.len() as u64;
    let mut chunks_done = 0u64;

    for (ds_id, ds) in summary.datasets.iter().enumerate() {
        let wire_id = u64::try_from(ds_id).map_err(|_| {
            ExportError::Catalog(crate::catalog::CatalogError::TooLargeForPlatform {
                field: "dataset_id",
                value: ds_id as u64,
            })
        })?;
        let chunks = chunks_for_dataset(&summary, wire_id);
        export_dataset(&file, &mmap, ds, &chunks, &summary)?;
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
    file: &File,
    mmap: &[u8],
    ds: &DatasetRecordV1,
    chunks: &[&crate::catalog::ChunkIndexEntryV1],
    summary: &TetFileSummaryV1,
) -> Result<(), ExportError> {
    let elem = element_dtype(ds)?;
    if matches!(elem, ElementDtype::F16) {
        return Err(ExportError::UnsupportedDtype {
            name: ds.name.clone(),
            detail: "f16 is not supported for HDF5 export".into(),
        });
    }
    let dense = densify_dataset(mmap, ds, chunks)?;
    let shape = shape_as_usize(&ds.shape)?;
    let chunk = shape_as_usize(&ds.chunk_shape)?;
    let h5_ds = write_typed_dataset(file, &ds.name, elem, &shape, &chunk, &dense)?;
    write_string_attrs(&h5_ds, &dataset_basic_export_attrs(summary, &ds.name))?;
    if let Some(dims) = dataset_dim_names(summary, &ds.name)
        && dims.len() == ds.shape.len()
    {
        let joined = dims.join(",");
        h5_ds
            .new_attr_builder()
            .with_data(joined.as_str())
            .create("dim_names")?;
    }
    h5_ds
        .new_attr_builder()
        .with_data(element_dtype_wire_label(elem))
        .create("tetration_dtype")?;
    Ok(())
}

fn write_typed_dataset(
    file: &File,
    name: &str,
    elem: ElementDtype,
    shape: &[usize],
    chunk: &[usize],
    dense: &[u8],
) -> Result<hdf5_metno::Dataset, ExportError> {
    match elem {
        ElementDtype::F32 => create_and_write::<f32>(file, name, shape, chunk, dense),
        ElementDtype::F64 => create_and_write::<f64>(file, name, shape, chunk, dense),
        ElementDtype::I32 => create_and_write::<i32>(file, name, shape, chunk, dense),
        ElementDtype::I64 => create_and_write::<i64>(file, name, shape, chunk, dense),
        ElementDtype::U8 => create_and_write::<u8>(file, name, shape, chunk, dense),
        ElementDtype::U16 => create_and_write::<u16>(file, name, shape, chunk, dense),
        ElementDtype::I16 => create_and_write::<i16>(file, name, shape, chunk, dense),
        ElementDtype::U32 => create_and_write::<u32>(file, name, shape, chunk, dense),
        ElementDtype::U64 => create_and_write::<u64>(file, name, shape, chunk, dense),
        ElementDtype::F16 => Err(ExportError::UnsupportedDtype {
            name: name.to_owned(),
            detail: "f16 is not supported for HDF5 export".into(),
        }),
    }
}

fn create_and_write<T: hdf5_metno::H5Type + bytemuck::Pod>(
    file: &File,
    name: &str,
    shape: &[usize],
    chunk: &[usize],
    dense: &[u8],
) -> Result<hdf5_metno::Dataset, ExportError> {
    let values: &[T] = bytemuck::try_cast_slice(dense)
        .map_err(|e| ExportError::Hdf5(format!("dense buffer cast failed for `{name}`: {e}")))?;
    let builder = file.new_dataset::<T>();
    let builder = if chunk.iter().all(|&c| c > 0) && chunk.len() == shape.len() {
        builder.chunk(chunk)
    } else {
        builder
    };
    let ds = builder.shape(shape).create(name)?;
    ds.write_raw(values)?;
    Ok(ds)
}

fn write_string_attrs(
    ds: &hdf5_metno::Dataset,
    attrs: &std::collections::BTreeMap<String, String>,
) -> Result<(), ExportError> {
    for (key, value) in attrs {
        if key == "tetration_dtype" || key == "dim_names" {
            continue;
        }
        ds.new_attr_builder()
            .with_data(value.as_str())
            .create(key.as_str())?;
    }
    Ok(())
}
