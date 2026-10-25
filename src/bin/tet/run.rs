//! `tet run` — thin wrapper over [`tetration::job`] (job file → ordered `tet` steps, fail-fast).

use std::io::IsTerminal;
use std::path::PathBuf;

use tetration::job::{RunJobOptions, StepReport, StepStatus, run_job_path_observed};

use crate::util::cli_error;

pub(crate) struct RunCmdOpts {
    pub job: PathBuf,
    pub dry_run: bool,
    pub root: Option<PathBuf>,
}

fn print_step(report: &StepReport) {
    let n = report.index + 1;
    match report.status {
        StepStatus::Planned => eprintln!("· step {n}: {}", report.description),
        StepStatus::Ok => eprintln!(
            "✓ step {n}: {} ({:.2}s)",
            report.description, report.elapsed_secs
        ),
    }
    if let Some(note) = &report.note {
        for line in note.lines() {
            eprintln!("    {line}");
        }
    }
    if let Some(out) = &report.stdout {
        print!("{out}");
        if !out.ends_with('\n') {
            println!();
        }
    }
}

pub(crate) fn run_run(opts: RunCmdOpts) -> Result<(), String> {
    let run_opts = RunJobOptions {
        root: opts.root,
        dry_run: opts.dry_run,
        allow_interactive: std::io::stdin().is_terminal(),
        record_query_history: false,
    };
    let report = run_job_path_observed(&opts.job, &run_opts, print_step).map_err(cli_error)?;
    if report.dry_run {
        eprintln!(
            "dry run ok: {} step(s) planned (root {})",
            report.steps.len(),
            report.root.display()
        );
    } else {
        eprintln!(
            "job ok: {} step(s) in {:.2}s",
            report.steps.len(),
            report.elapsed_secs
        );
    }
    Ok(())
}
