//! HDF5 → `.tet` conversion (streaming, one chunk at a time).

use std::path::Path;
use std::time::Instant;

use hdf5_metno::File;

use super::cf::cf_from_hdf5;
use super::hdf5_shared::map_hdf5_element_dtype;
use super::import_metadata::{
    enrich_hdf5_cf_coordinates, finish_convert_footer, hdf5_dataset_attrs,
};
use super::parallel::H5ParallelSource;
use super::shared::{
    ImportPlan, ensure_non_empty, ensure_strict, import_chunk_shape, join_catalog_path, note_plan,
    write_plans_streaming,
};
use super::{ConvertError, ConvertProgress, ConvertReport, report};

/// Import all supported numeric datasets from an HDF5 file into one `.tet`.
///
/// Walks nested groups (`group/var` → catalog name `group/var`). Numeric dtypes from `u8`
/// through `f64` are imported. CF `scale_factor` / `add_offset` / `_FillValue` are decoded
/// per element at tile read.
///
/// # Errors
///
/// Returns [`ConvertError`] when the file cannot be read or no supported datasets are present.
pub fn convert_h5_to_tet(input: &Path, output: &Path) -> Result<ConvertReport, ConvertError> {
    convert_h5_to_tet_with_progress(input, output, 0, None::<fn(ConvertProgress)>)
}

/// Like [`convert_h5_to_tet`], invoking `progress` after each chunk payload is written.
///
/// `parallel_jobs`: chunk read workers (`0` = [`super::parallel::default_parallel_jobs`]).
///
/// # Errors
///
/// Returns [`ConvertError`] when the file cannot be read or no supported datasets are present.
pub fn convert_h5_to_tet_with_progress(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    convert_h5_inner(input, output, parallel_jobs, false, None, progress)
}

pub(crate) fn convert_h5_inner(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    chunk_bytes: Option<u64>,
    mut progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    let started = Instant::now();
    let file = File::open(input)?;
    let mut plans = Vec::new();
    let mut skipped = Vec::new();
    collect_h5_plans(&file, "", chunk_bytes, &mut plans, &mut skipped)?;
    enrich_hdf5_cf_coordinates(&file, &mut plans);

    ensure_non_empty(
        input,
        &plans.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
    )?;
    ensure_strict(input, &skipped, strict)?;

    let mut progress_bridge = |done: u64, total: u64, dataset: &str| {
        if let Some(ref mut cb) = progress {
            cb(ConvertProgress {
                chunks_done: done,
                chunks_total: total,
                dataset: dataset.to_owned(),
            });
        }
    };

    let parallel_jobs = super::parallel::resolve_parallel_jobs(parallel_jobs);
    let source = H5ParallelSource::new(input.to_path_buf(), plans.clone());
    write_plans_streaming(
        output,
        &plans,
        parallel_jobs,
        |job, buf| source.fill_tile(job, buf),
        Some(&mut progress_bridge as &mut dyn FnMut(u64, u64, &str)),
    )?;
    let history = finish_convert_footer(output, "h5", &plans)?;

    Ok(report(
        input,
        output,
        &plans,
        skipped,
        history,
        started.elapsed().as_secs_f64(),
    ))
}

fn collect_h5_plans(
    loc: &hdf5_metno::Group,
    prefix: &str,
    chunk_bytes: Option<u64>,
    plans: &mut Vec<ImportPlan>,
    skipped: &mut Vec<super::ConvertSkip>,
) -> Result<(), ConvertError> {
    let err = ConvertError::from;
    let mut names = loc.member_names().map_err(err)?;
    names.sort();
    for name in names {
        if loc.dataset(&name).is_ok() {
            let ds = loc.dataset(&name).map_err(err)?;
            let path = join_catalog_path(prefix, &name);
            note_plan(plan_dataset(&path, &ds, chunk_bytes), plans, skipped)?;
        } else if loc.group(&name).is_ok() {
            let grp = loc.group(&name).map_err(err)?;
            collect_h5_plans(
                &grp,
                &join_catalog_path(prefix, &name),
                chunk_bytes,
                plans,
                skipped,
            )?;
        }
    }
    Ok(())
}

fn plan_dataset(
    name: &str,
    ds: &hdf5_metno::Dataset,
    chunk_bytes: Option<u64>,
) -> Result<ImportPlan, ConvertError> {
    let dtype = map_hdf5_element_dtype(ds, name)?;
    let shape: Vec<u64> = ds
        .shape()
        .iter()
        .map(|&d| u64::try_from(d).unwrap_or(0))
        .collect();
    if shape.is_empty() || shape.contains(&0) {
        return Err(ConvertError::UnsupportedDtype {
            name: name.to_owned(),
            detail: "empty or scalar dataset".into(),
        });
    }
    let cf = cf_from_hdf5(ds);
    let source_chunks = if cf.is_some() {
        None
    } else {
        hdf5_chunk_shape(ds)
    };
    let chunk_shape = import_chunk_shape(&shape, dtype.elem_size(), source_chunks, chunk_bytes);
    Ok(
        ImportPlan::new(name.to_owned(), dtype, shape, chunk_shape, cf).with_import(
            hdf5_dataset_attrs(ds),
            None,
            None,
        ),
    )
}

fn hdf5_chunk_shape(ds: &hdf5_metno::Dataset) -> Option<Vec<usize>> {
    if !ds.is_chunked() {
        return None;
    }
    ds.chunk()
}
