//! Foreign format → `.tet` conversion.
//!
//! Importers: HDF5, `NetCDF`, Zarr v2 and v3, `NumPy` `.npy` / `.npz`, CSV / TSV, and Parquet
//! (`tetration-parquet`). `--chunk-mib` retile the output grid; omitting it keeps source
//! chunks, or a 64 MiB leading-axis tile for `NumPy`, CSV, and Parquet.
//!
//! With neither `tetration-hdf5` nor `tetration-netcdf` (e.g. lean `libtetration` for FFI),
//! HDF5/NetCDF-only helpers in this tree are unused; see the `allow` below.

#![cfg_attr(
    all(not(feature = "tetration-hdf5"), not(feature = "tetration-netcdf")),
    allow(dead_code, unused_imports)
)]

use std::path::Path;

use serde::Serialize;

use crate::catalog::CatalogError;

mod category;
mod cf;
mod column_buf;
mod csv;
#[cfg(feature = "tetration-hdf5")]
mod hdf5;
#[cfg(feature = "tetration-hdf5")]
mod hdf5_shared;
mod import_metadata;
#[cfg(feature = "tetration-netcdf")]
mod netcdf;
mod numpy;
mod parallel;
#[cfg(feature = "tetration-parquet")]
mod parquet;
mod s3;
mod shared;
mod sniff;
mod tile_io;
mod zarr;
mod zarr_io;

pub use parallel::{default_parallel_jobs, resolve_parallel_jobs};
pub use s3::{
    S3ConvertObject, convert_s3_object_to_tet, is_s3_convert_uri, parse_s3_convert_object,
};

#[cfg(test)]
pub(crate) use csv::convert_csv_with_chunk_bytes;
pub use csv::{convert_csv_to_tet, convert_csv_to_tet_with_progress};
#[cfg(feature = "tetration-hdf5")]
pub use hdf5::{convert_h5_to_tet, convert_h5_to_tet_with_progress};
#[cfg(feature = "tetration-netcdf")]
pub use netcdf::{convert_netcdf_to_tet, convert_netcdf_to_tet_with_progress};
#[cfg(test)]
pub(crate) use numpy::convert_numpy_with_chunk_bytes;
pub use numpy::{
    convert_npy_to_tet, convert_npy_to_tet_with_progress, convert_npz_to_tet,
    convert_npz_to_tet_with_progress,
};
#[cfg(feature = "tetration-parquet")]
pub use parquet::{convert_parquet_to_tet, convert_parquet_to_tet_with_progress};
pub use zarr::{
    convert_zarr_to_tet, convert_zarr_to_tet_with_progress, is_zarr_v2_directory,
    is_zarr_v3_directory,
};

/// Supported foreign input formats for [`convert_to_tet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertInputFormat {
    H5,
    Netcdf,
    Zarr,
    Npy,
    Npz,
    Csv,
    Parquet,
}

impl ConvertInputFormat {
    /// History footer source tag (`"h5"`, `"nc"`, `"zarr"`, `"npy"`, `"npz"`, `"csv"`, or `"parquet"`).
    ///
    /// Zarr v2 writes `"zarr2"` from the importer. This method still returns `"zarr"` for
    /// [`Self::Zarr`], which is the sniff tag shared by both store versions.
    #[must_use]
    pub const fn history_source(self) -> &'static str {
        match self {
            Self::H5 => "h5",
            Self::Netcdf => "nc",
            Self::Zarr => "zarr",
            Self::Npy => "npy",
            Self::Npz => "npz",
            Self::Csv => "csv",
            Self::Parquet => "parquet",
        }
    }
}

/// Detect importer from the input path (extension, then file signature).
///
/// See [`detect_convert_format`] and [`Hdf5ConvertInput`] / [`NetcdfConvertInput`].
pub use sniff::{
    ConvertCompressionSuffixes, CsvConvertInput, Hdf5ConvertInput, NetcdfConvertInput,
    NumpyConvertInput, ParquetConvertInput, ZarrConvertInput, detect_convert_format,
};

/// Convert a foreign array file to `.tet`, picking the importer from the input extension.
///
/// # Errors
///
/// Returns [`ConvertError`] when the extension is unsupported, the required feature is disabled,
/// or import fails.
pub fn convert_to_tet(input: &Path, output: &Path) -> Result<ConvertReport, ConvertError> {
    convert_to_tet_with_progress(input, output, 0, None::<fn(ConvertProgress)>)
}

/// Like [`convert_to_tet`], invoking `progress` after each chunk payload is written.
///
/// `parallel_jobs`: chunk read workers (`0` = [`default_parallel_jobs`]).
///
/// # Errors
///
/// Returns [`ConvertError`] when the extension is unsupported, the required feature is disabled,
/// or import fails.
pub fn convert_to_tet_with_progress(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    convert_to_tet_with_options(
        input,
        output,
        ConvertOptions {
            parallel_jobs,
            strict: false,
            chunk_mib: None,
        },
        progress,
    )
}

/// Like [`convert_to_tet_with_progress`], with [`ConvertOptions::strict`].
///
/// # Errors
///
/// Returns [`ConvertError::StrictSkips`] when `options.strict` is set and any dataset was skipped.
/// The output file is not written in that case.
pub fn convert_to_tet_with_options(
    input: &Path,
    output: &Path,
    options: ConvertOptions,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    let chunk_bytes = shared::chunk_target_bytes(options.chunk_mib)?;
    match detect_convert_format(input)? {
        ConvertInputFormat::H5 => convert_h5_dispatch(
            input,
            output,
            options.parallel_jobs,
            options.strict,
            chunk_bytes,
            progress,
        ),
        ConvertInputFormat::Netcdf => convert_netcdf_dispatch(
            input,
            output,
            options.parallel_jobs,
            options.strict,
            chunk_bytes,
            progress,
        ),
        ConvertInputFormat::Zarr => zarr::convert_zarr_inner(
            input,
            output,
            options.parallel_jobs,
            options.strict,
            chunk_bytes,
            progress,
        ),
        ConvertInputFormat::Npy => numpy::convert_numpy_inner(
            input,
            output,
            options.parallel_jobs,
            options.strict,
            ConvertInputFormat::Npy,
            chunk_bytes,
            progress,
        ),
        ConvertInputFormat::Npz => numpy::convert_numpy_inner(
            input,
            output,
            options.parallel_jobs,
            options.strict,
            ConvertInputFormat::Npz,
            chunk_bytes,
            progress,
        ),
        ConvertInputFormat::Csv => csv::convert_csv_inner(
            input,
            output,
            options.parallel_jobs,
            options.strict,
            chunk_bytes,
            progress,
        ),
        ConvertInputFormat::Parquet => convert_parquet_dispatch(
            input,
            output,
            options.parallel_jobs,
            options.strict,
            chunk_bytes,
            progress,
        ),
    }
}

#[cfg(feature = "tetration-parquet")]
fn convert_parquet_dispatch(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    chunk_bytes: Option<u64>,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    parquet::convert_parquet_inner(input, output, parallel_jobs, strict, chunk_bytes, progress)
}

#[cfg(not(feature = "tetration-parquet"))]
fn convert_parquet_dispatch(
    _input: &Path,
    _output: &Path,
    _parallel_jobs: usize,
    _strict: bool,
    _chunk_bytes: Option<u64>,
    _progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    Err(ConvertError::ConvertFeatureDisabled {
        format: ConvertInputFormat::Parquet,
        feature: "tetration-parquet",
    })
}

#[cfg(feature = "tetration-hdf5")]
fn convert_h5_dispatch(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    chunk_bytes: Option<u64>,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    hdf5::convert_h5_inner(input, output, parallel_jobs, strict, chunk_bytes, progress)
}

#[cfg(not(feature = "tetration-hdf5"))]
fn convert_h5_dispatch(
    _input: &Path,
    _output: &Path,
    _parallel_jobs: usize,
    _strict: bool,
    _chunk_bytes: Option<u64>,
    _progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    Err(ConvertError::ConvertFeatureDisabled {
        format: ConvertInputFormat::H5,
        feature: "tetration-hdf5",
    })
}

#[cfg(feature = "tetration-netcdf")]
fn convert_netcdf_dispatch(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    chunk_bytes: Option<u64>,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    netcdf::convert_netcdf_inner(input, output, parallel_jobs, strict, chunk_bytes, progress)
}

#[cfg(not(feature = "tetration-netcdf"))]
fn convert_netcdf_dispatch(
    _input: &Path,
    _output: &Path,
    _parallel_jobs: usize,
    _strict: bool,
    _chunk_bytes: Option<u64>,
    _progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    Err(ConvertError::ConvertFeatureDisabled {
        format: ConvertInputFormat::Netcdf,
        feature: "tetration-netcdf",
    })
}

/// Knobs for [`convert_to_tet_with_options`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConvertOptions {
    /// Chunk read workers (`0` = host default, cap 64).
    pub parallel_jobs: usize,
    /// Fail before writing when any dataset was skipped.
    pub strict: bool,
    /// Retile every dataset to about this many MiB along the leading axis.
    /// `None` keeps source chunking, or the 64 MiB default for `NumPy`, CSV, and Parquet.
    pub chunk_mib: Option<u32>,
}

/// Progress event emitted while streaming chunk payloads during convert.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ConvertProgress {
    /// Chunks whose payloads have been written.
    pub chunks_done: u64,
    /// Chunks in the planned output grid.
    pub chunks_total: u64,
    /// Catalog name of the dataset currently being written.
    pub dataset: String,
}

/// One dataset written during convert.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ConvertDatasetSummary {
    /// Catalog path (`group/var`).
    pub name: String,
    /// Rank of [`Self::dims`].
    pub ndim: usize,
    /// Per-axis lengths (same order as on-disk `.tet` / source array).
    pub dims: Vec<u64>,
}

/// A source array that was not imported.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ConvertSkip {
    /// Catalog path that would have been used (`group/var`).
    pub path: String,
    /// Why it was skipped (unsupported dtype, empty, scalar, …).
    pub reason: String,
}

/// Summary returned after a successful convert.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ConvertReport {
    /// Source path that was converted.
    pub input: String,
    /// `.tet` path that was written.
    pub output: String,
    /// Number of datasets written. Same as [`Self::dataset_names`] length.
    pub dataset_count: usize,
    /// Catalog names, in write order.
    pub dataset_names: Vec<String>,
    pub datasets: Vec<ConvertDatasetSummary>,
    /// Unsupported arrays left out of the `.tet` (empty when nothing was skipped).
    pub skipped: Vec<ConvertSkip>,
    /// History row written to the `.tet` footer (`HistoryEvent`: `op`, `source`, `at`, …).
    pub history: Vec<crate::catalog::HistoryEvent>,
    /// Wall-clock seconds for the full convert (plan + stream write + history footer).
    pub elapsed_secs: f64,
}

/// Convert pipeline failures.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error(transparent)]
    Catalog(#[from] CatalogError),
    #[error("no supported numeric datasets found in {path}")]
    NoDatasets { path: String },
    #[cfg(feature = "tetration-netcdf")]
    #[error("NetCDF open/read failed: {0}")]
    Netcdf(String),
    #[cfg(feature = "tetration-hdf5")]
    #[error("HDF5 open/read failed: {0}")]
    Hdf5(String),
    #[error("Zarr open/read failed: {0}")]
    Zarr(String),
    #[error("NumPy open/read failed: {0}")]
    Numpy(String),
    #[error("CSV open/read failed: {0}")]
    Csv(String),
    #[error("Parquet open/read failed: {0}")]
    Parquet(String),
    #[error("unsupported element type in variable `{name}`: {detail}")]
    UnsupportedDtype { name: String, detail: String },
    #[error("skipped {count} unsupported dataset(s) in {path} (--strict): {detail}")]
    StrictSkips {
        path: String,
        count: usize,
        detail: String,
    },
    #[error(
        "unsupported convert input `{path}`: extension `{ext}` (supported extensions: {h5_ext}; {nc_ext}; {zarr_ext}; {npy_ext}; .csv (comma), .tsv (tab); .parquet, .pq; or recognizable HDF5 / NetCDF-3 / NumPy / Parquet file signature / Zarr v2 or v3 directory store)"
    )]
    UnsupportedInputExtension {
        path: String,
        ext: String,
        h5_ext: &'static str,
        nc_ext: &'static str,
        zarr_ext: &'static str,
        npy_ext: &'static str,
    },
    #[error("--chunk-mib must be at least 1")]
    ChunkMibZero,
    #[error("{format:?} convert requires Cargo feature `{feature}`")]
    ConvertFeatureDisabled {
        format: ConvertInputFormat,
        feature: &'static str,
    },
    #[error("s3 convert failed: {0}")]
    S3(String),
    #[error("s3:// convert requires Cargo feature `tetration-s3`")]
    S3FeatureDisabled,
    #[error("s3:// convert requires environment variable `{0}`")]
    S3ConfigMissing(&'static str),
}

fn report(
    path_in: &Path,
    path_out: &Path,
    plans: &[shared::ImportPlan],
    skipped: Vec<ConvertSkip>,
    history: Vec<crate::catalog::HistoryEvent>,
    elapsed_secs: f64,
) -> ConvertReport {
    let datasets: Vec<ConvertDatasetSummary> = plans
        .iter()
        .map(|p| ConvertDatasetSummary {
            name: p.name.clone(),
            ndim: p.shape.len(),
            dims: p.shape.clone(),
        })
        .collect();
    let dataset_names: Vec<String> = datasets.iter().map(|d| d.name.clone()).collect();
    ConvertReport {
        input: path_in.display().to_string(),
        output: path_out.display().to_string(),
        dataset_count: datasets.len(),
        dataset_names,
        datasets,
        skipped,
        history,
        elapsed_secs,
    }
}
