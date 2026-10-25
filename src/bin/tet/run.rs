//! `tet run` — thin wrapper over [`tetration::job`] (job file → ordered `tet` steps, fail-fast).

use std::io::IsTerminal;
use std::path::PathBuf;

use tetration::cli_pretty::{parse_kv_line, step_kv_table, step_note_line, step_ok, step_planned};
use tetration::job::{RunJobOptions, StepReport, StepStatus, run_job_path_observed};

use crate::util::cli_error;

pub(crate) struct RunCmdOpts {
    pub job: PathBuf,
    pub dry_run: bool,
    pub root: Option<PathBuf>,
}

/// Step status + notes on stderr. Quiet one-line results nest under the step;
/// multi-line dumps (tables / JSON) stay on stdout so scripts can still capture them.
fn print_step(report: &StepReport, total: usize) {
    let n = report.index + 1;
    let line = match report.status {
        StepStatus::Planned => step_planned(n, total, &report.description),
        StepStatus::Ok => step_ok(n, total, &report.description, report.elapsed_secs),
    };
    eprintln!("{line}");
    if let Some(note) = &report.note {
        for line in note.lines() {
            eprintln!("{}", step_note_line(line));
        }
    }
    if let Some(out) = &report.stdout {
        let trimmed = out.trim_end_matches('\n');
        let line_count = trimmed.lines().count();
        if line_count <= 1 && !trimmed.is_empty() {
            // Quiet `key=value …` line → light vertical kv table under the step.
            let pairs = parse_kv_line(trimmed);
            if pairs.len() >= 2 {
                eprint!("{}", step_kv_table(&pairs));
            } else {
                eprintln!("{}", step_note_line(trimmed));
            }
        } else if line_count > 1 {
            print!("{out}");
            if !out.ends_with('\n') {
                println!();
            }
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
