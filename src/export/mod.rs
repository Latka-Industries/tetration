//! `.tet` → foreign interchange export (Zarr v3, HDF5, NetCDF).
//!
//! HDF5 / NetCDF writers are feature-gated (`tetration-hdf5` / `tetration-netcdf`),
//! matching the convert importers.

mod common;
#[cfg(feature = "tetration-hdf5")]
mod hdf5;
#[cfg(feature = "tetration-netcdf")]
mod netcdf;
mod zarr;

pub use common::{ExportError, ExportOutputFormat, ExportProgress, ExportReport};
pub use zarr::{export_tet_to_zarr, export_tet_to_zarr_with_progress};

#[cfg(feature = "tetration-hdf5")]
pub use hdf5::{export_tet_to_hdf5, export_tet_to_hdf5_with_progress};
#[cfg(feature = "tetration-netcdf")]
pub use netcdf::{export_tet_to_netcdf, export_tet_to_netcdf_with_progress};

use std::path::Path;

use common::ExportError as E;

/// Export a `.tet` file, picking the writer from the output path.
///
/// - Directory / `.zarr` → Zarr v3 directory store
/// - `.h5` / `.hdf5` / … → HDF5 (`tetration-hdf5`)
/// - `.nc` / `.netcdf` / … → NetCDF (`tetration-netcdf`)
///
/// # Errors
///
/// Returns [`ExportError`] when the output format is unsupported, a required feature is
/// disabled, or the writer fails.
pub fn export_tet(input: &Path, output: &Path) -> Result<ExportReport, ExportError> {
    export_tet_with_progress(input, output, None::<fn(ExportProgress)>)
}

/// Like [`export_tet`], invoking `progress` during the write.
///
/// # Errors
///
/// Same as [`export_tet`].
pub fn export_tet_with_progress(
    input: &Path,
    output: &Path,
    progress: Option<impl FnMut(ExportProgress)>,
) -> Result<ExportReport, ExportError> {
    let format = ExportOutputFormat::detect(output).ok_or_else(|| E::UnsupportedOutput {
        path: output.display().to_string(),
    })?;
    match format {
        ExportOutputFormat::Zarr => export_tet_to_zarr_with_progress(input, output, progress),
        ExportOutputFormat::H5 => export_hdf5_dispatch(input, output, progress),
        ExportOutputFormat::Netcdf => export_netcdf_dispatch(input, output, progress),
    }
}

#[cfg(feature = "tetration-hdf5")]
fn export_hdf5_dispatch(
    input: &Path,
    output: &Path,
    progress: Option<impl FnMut(ExportProgress)>,
) -> Result<ExportReport, ExportError> {
    export_tet_to_hdf5_with_progress(input, output, progress)
}

#[cfg(not(feature = "tetration-hdf5"))]
fn export_hdf5_dispatch(
    _input: &Path,
    _output: &Path,
    _progress: Option<impl FnMut(ExportProgress)>,
) -> Result<ExportReport, ExportError> {
    Err(E::ExportFeatureDisabled {
        format: "HDF5",
        feature: "tetration-hdf5",
    })
}

#[cfg(feature = "tetration-netcdf")]
fn export_netcdf_dispatch(
    input: &Path,
    output: &Path,
    progress: Option<impl FnMut(ExportProgress)>,
) -> Result<ExportReport, ExportError> {
    export_tet_to_netcdf_with_progress(input, output, progress)
}

#[cfg(not(feature = "tetration-netcdf"))]
fn export_netcdf_dispatch(
    _input: &Path,
    _output: &Path,
    _progress: Option<impl FnMut(ExportProgress)>,
) -> Result<ExportReport, ExportError> {
    Err(E::ExportFeatureDisabled {
        format: "NetCDF",
        feature: "tetration-netcdf",
    })
}
