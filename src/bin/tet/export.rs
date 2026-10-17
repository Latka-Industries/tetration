//! `tet export` (`.tet` → Zarr v3 / HDF5 / `NetCDF`).

use std::path::Path;

use tetration::export::export_tet_with_progress;

use super::util::cli_error;

pub(crate) fn run_export(input: &Path, output: &Path) -> Result<(), String> {
    let report = export_tet_with_progress(input, output, None::<fn(_)>).map_err(cli_error)?;
    eprintln!(
        "exported {} dataset(s), {} chunk(s) → {} ({:.2}s)",
        report.dataset_count, report.chunks_written, report.output, report.elapsed_secs
    );
    Ok(())
}
