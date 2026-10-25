//! `tet run` — thin wrapper over [`tetration::job`] (job file → ordered `tet` steps, fail-fast).

use std::io::IsTerminal;
use std::path::PathBuf;

use tetration::cli_pretty::{parse_kv_line, step_kv_table, step_note_line, step_ok, step_planned};
use tetration::job::{
    JobError, JobReport, RunJobOptions, StepReport, StepStatus, run_job_path_observed,
};

use crate::util::cli_error;

pub(crate) struct RunCmdOpts {
    pub job: PathBuf,
    pub dry_run: bool,
    pub root: Option<PathBuf>,
    /// Final `JobReport` as pretty JSON on stdout; step dumps move to stderr.
    pub json: bool,
}

/// Step status + notes on stderr. Quiet one-line results nest under the step;
/// multi-line dumps (tables / JSON) stay on stdout so scripts can still capture them.
/// With `--json`, stdout is reserved for the final `JobReport`, so those dumps go to stderr.
fn print_step(report: &StepReport, total: usize, json: bool) {
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
            if json {
                eprint!("{out}");
                if !out.ends_with('\n') {
                    eprintln!();
                }
            } else {
                print!("{out}");
                if !out.ends_with('\n') {
                    println!();
                }
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
    let json = opts.json;
    let report = match run_job_path_observed(&opts.job, &run_opts, |r, total| {
        print_step(r, total, json);
    }) {
        Ok(report) => report,
        Err(err) => {
            if json {
                print_partial_report(&err, &opts.job, run_opts.root.as_deref(), run_opts.dry_run);
            }
            return Err(cli_error(err));
        }
    };
    if json {
        let text = serde_json::to_string_pretty(&report).map_err(cli_error)?;
        println!("{text}");
    }
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

/// `--json` on failure: stdout gets the steps that completed (a [`JobReport`] plus an `error`
/// object naming the failed step) so tooling can see partial results. Exit code stays 1.
/// Failures before step 1 (parse / pre-flight / I/O) print nothing on stdout.
fn print_partial_report(
    err: &JobError,
    job: &std::path::Path,
    root: Option<&std::path::Path>,
    dry_run: bool,
) {
    let JobError::Step {
        index,
        op,
        message,
        path,
        completed,
    } = err
    else {
        return;
    };
    let root = root.map_or_else(
        || {
            job.parent()
                .filter(|p| !p.as_os_str().is_empty())
                .map_or_else(|| PathBuf::from("."), PathBuf::from)
        },
        PathBuf::from,
    );
    let partial = JobReport {
        dry_run,
        root,
        elapsed_secs: completed.iter().map(|s| s.elapsed_secs).sum(),
        steps: completed.clone(),
    };
    let Ok(mut value) = serde_json::to_value(&partial) else {
        return;
    };
    if let Some(obj) = value.as_object_mut() {
        obj.insert(
            "error".to_owned(),
            serde_json::json!({
                "index": index,
                "op": op,
                "message": message,
                "path": path.as_ref().map(|p| p.display().to_string()),
            }),
        );
    }
    if let Ok(text) = serde_json::to_string_pretty(&value) {
        println!("{text}");
    }
}
