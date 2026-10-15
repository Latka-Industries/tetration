//! `tet verify` — layout v1 file or `.tet.d` package health check.

use std::path::PathBuf;

use tetration::repair::{format_repair_text, repair_from_verify_report};
use tetration::verify::{
    VerifyOptions, format_verify_json, format_verify_quiet, format_verify_text,
    looks_like_tet_package, verify_tet_file_with_options, verify_tet_package_with_options,
};

use crate::util::cli_error;

#[allow(clippy::struct_excessive_bools)]
pub(crate) struct VerifyRunOpts {
    pub path: PathBuf,
    pub json: bool,
    pub quiet: bool,
    pub repair: bool,
    pub deep: bool,
}

fn run_verify_once(
    path: &std::path::Path,
    verify_opts: VerifyOptions,
) -> Result<tetration::verify::TetVerifyReport, String> {
    if looks_like_tet_package(path) {
        verify_tet_package_with_options(path, verify_opts).map_err(cli_error)
    } else {
        verify_tet_file_with_options(path, verify_opts).map_err(cli_error)
    }
}

pub(crate) fn run_verify(opts: &VerifyRunOpts) -> Result<(), String> {
    let verify_opts = VerifyOptions {
        deep_decode: opts.deep,
    };
    let mut report = run_verify_once(&opts.path, verify_opts)?;
    if opts.repair && !report.ok {
        let repair_report =
            repair_from_verify_report(&opts.path, &report, false).map_err(cli_error)?;
        if !opts.quiet && !opts.json {
            eprintln!("{}", format_repair_text(&repair_report));
        }
        report = run_verify_once(&opts.path, verify_opts)?;
    }
    let out = if opts.json {
        format_verify_json(&report).map_err(cli_error)?
    } else if opts.quiet {
        format_verify_quiet(&report)
    } else {
        format_verify_text(&report)
    };
    print!("{out}");
    if report.ok {
        Ok(())
    } else {
        Err("verification failed".to_owned())
    }
}
