//! `tet convert` (HDF5 / `NetCDF` / Zarr → `.tet`).

use std::path::Path;

use tetration::convert::{
    ConvertOptions, ConvertProgress, ConvertReport, convert_to_tet_with_options,
    detect_convert_format,
};

use crate::util::cli_error;

fn finish_convert_report(
    pb: &indicatif::ProgressBar,
    label: &str,
    report: &ConvertReport,
) -> Result<(), String> {
    pb.finish_with_message(format!("{label} done in {:.2}s", report.elapsed_secs));
    if !report.skipped.is_empty() {
        eprintln!("skipped {} dataset(s):", report.skipped.len());
        for skip in &report.skipped {
            eprintln!("  {}: {}", skip.path, skip.reason);
        }
    }
    let pretty = serde_json::to_string_pretty(report).map_err(cli_error)?;
    println!();
    println!("{pretty}");
    Ok(())
}

pub(crate) fn run_convert(
    input: &Path,
    output: &Path,
    jobs: usize,
    strict: bool,
) -> Result<(), String> {
    use indicatif::{ProgressBar, ProgressStyle};

    let format = detect_convert_format(input).map_err(cli_error)?;
    let label = match format {
        tetration::convert::ConvertInputFormat::H5 => "HDF5 convert",
        tetration::convert::ConvertInputFormat::Netcdf => "NetCDF convert",
        tetration::convert::ConvertInputFormat::Zarr => "Zarr convert",
    };
    let progress_prefix = match format {
        tetration::convert::ConvertInputFormat::H5 => "HDF5 → .tet",
        tetration::convert::ConvertInputFormat::Netcdf => "NetCDF → .tet",
        tetration::convert::ConvertInputFormat::Zarr => "Zarr → .tet",
    };

    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::with_template("{msg} [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
            .map_err(cli_error)?
            .progress_chars("=>-"),
    );
    pb.set_message(progress_prefix.to_owned());

    let progress = Some(|p: ConvertProgress| {
        if pb.length().unwrap_or(0) != p.chunks_total {
            pb.set_length(p.chunks_total);
        }
        pb.set_position(p.chunks_done);
        pb.set_message(format!("{progress_prefix} ({})", p.dataset));
    });

    let report = convert_to_tet_with_options(
        input,
        output,
        ConvertOptions {
            parallel_jobs: jobs,
            strict,
        },
        progress,
    )
    .map_err(cli_error)?;
    finish_convert_report(&pb, label, &report)
}
