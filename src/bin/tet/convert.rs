//! `tet convert` (HDF5, `NetCDF`, Zarr v2/v3, `NumPy`, CSV, Parquet → `.tet`).

use std::path::Path;

use tetration::cli_pretty::chunk_progress_style;
use tetration::convert::{
    ConvertOptions, ConvertProgress, ConvertReport, convert_s3_object_to_tet,
    convert_to_tet_with_options, detect_convert_format, is_s3_convert_uri,
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
    input: &str,
    output: &Path,
    jobs: usize,
    strict: bool,
    chunk_mib: Option<u32>,
) -> Result<(), String> {
    if is_s3_convert_uri(input) {
        return run_s3_convert(input, output, jobs, strict, chunk_mib);
    }

    let format = detect_convert_format(Path::new(input)).map_err(cli_error)?;
    let label = match format {
        tetration::convert::ConvertInputFormat::H5 => "HDF5 convert",
        tetration::convert::ConvertInputFormat::Netcdf => "NetCDF convert",
        tetration::convert::ConvertInputFormat::Zarr => "Zarr convert",
        tetration::convert::ConvertInputFormat::Npy
        | tetration::convert::ConvertInputFormat::Npz => "NumPy convert",
        tetration::convert::ConvertInputFormat::Csv => "CSV convert",
        tetration::convert::ConvertInputFormat::Parquet => "Parquet convert",
    };
    let progress_prefix = match format {
        tetration::convert::ConvertInputFormat::H5 => "HDF5 → .tet",
        tetration::convert::ConvertInputFormat::Netcdf => "NetCDF → .tet",
        tetration::convert::ConvertInputFormat::Zarr => "Zarr → .tet",
        tetration::convert::ConvertInputFormat::Npy
        | tetration::convert::ConvertInputFormat::Npz => "NumPy → .tet",
        tetration::convert::ConvertInputFormat::Csv => "CSV → .tet",
        tetration::convert::ConvertInputFormat::Parquet => "Parquet → .tet",
    };

    run_convert_with_bar(progress_prefix, label, |bar| {
        let prefix = progress_prefix.to_owned();
        convert_to_tet_with_options(
            Path::new(input),
            output,
            convert_options(jobs, strict, chunk_mib),
            Some(move |p| tick_bar(&bar, &prefix, &p)),
        )
    })
}

fn run_s3_convert(
    input: &str,
    output: &Path,
    jobs: usize,
    strict: bool,
    chunk_mib: Option<u32>,
) -> Result<(), String> {
    run_convert_with_bar("s3 → .tet", "s3 convert", |bar| {
        convert_s3_object_to_tet(
            input,
            output,
            convert_options(jobs, strict, chunk_mib),
            Some(move |p| tick_bar(&bar, "s3 → .tet", &p)),
        )
    })
}

fn convert_options(jobs: usize, strict: bool, chunk_mib: Option<u32>) -> ConvertOptions {
    ConvertOptions {
        parallel_jobs: jobs,
        strict,
        chunk_mib,
    }
}

fn run_convert_with_bar(
    progress_prefix: &str,
    label: &str,
    convert: impl FnOnce(
        indicatif::ProgressBar,
    ) -> Result<ConvertReport, tetration::convert::ConvertError>,
) -> Result<(), String> {
    use indicatif::ProgressBar;

    let pb = ProgressBar::new(0);
    pb.set_style(chunk_progress_style().map_err(cli_error)?);
    pb.set_message(progress_prefix.to_owned());
    let report = convert(pb.clone()).map_err(cli_error)?;
    finish_convert_report(&pb, label, &report)
}

fn tick_bar(pb: &indicatif::ProgressBar, prefix: &str, p: &ConvertProgress) {
    if pb.length().unwrap_or(0) != p.chunks_total {
        pb.set_length(p.chunks_total);
    }
    pb.set_position(p.chunks_done);
    pb.set_message(format!("{prefix} ({})", p.dataset));
}
