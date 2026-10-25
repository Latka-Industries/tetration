//! Job files: a TOML or JSON list of stock `tet` steps, run in order, fail-fast.
//!
//! `tet run job.toml` is a thin wrapper over [`run_job_path`]. The executor calls the same
//! public library APIs the CLI uses (convert, verify, query plan/execute, publish, …); it never
//! spawns `tet`. Steps are **not** a DAG: no conditionals, loops, retries, or "continue on
//! error". This is not a scheduler and not query-pipeline ETL (use the query `pipeline` wire for
//! chained transforms inside one query).
//!
//! ```toml
//! schema_version = 1
//!
//! [[step]]
//! op = "convert"
//! from = "raw/x.csv"
//! to = "work/x.tet"
//!
//! [[step]]
//! op = "verify"
//! path = "work/x.tet"
//! ```
//!
//! Relative paths resolve against [`RunJobOptions::root`] (default: the job file's directory).
//! The executor is quiet: it returns text each step would print as
//! [`StepReport::stdout`] / [`StepReport::note`] instead of printing, and takes no progress
//! callbacks. Query history is off unless [`RunJobOptions::record_query_history`] is set.
//! Full guide: [`docs/job.md`](../../docs/job.md).

mod exec;
mod schema;

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;

pub use schema::{
    AttrValue, CompactStep, ConvertStep, ExportStep, GcStep, IndexStep, InfoStep, IngestStep,
    JOB_SCHEMA_VERSION, Job, OPS, PackStep, PublishStep, QueryStep, RepairStep, SemanticsApplyStep,
    SemanticsLintStep, SemanticsPackageWriteStep, SemanticsScaffoldStep, SemanticsSetStep,
    SemanticsWizardStep, Step, VerifyStep,
};

/// How [`parse_job`] chooses a parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JobFormat {
    /// `.json` / leading `{` → JSON; otherwise TOML.
    #[default]
    Auto,
    Json,
    Toml,
}

/// Infer JSON vs TOML from an optional file path and the payload text.
#[must_use]
pub fn detect_job_format(path_hint: Option<&Path>, text: &str) -> JobFormat {
    match path_hint
        .and_then(|p| p.extension())
        .and_then(|e| e.to_str())
    {
        Some("json") => JobFormat::Json,
        Some("toml") => JobFormat::Toml,
        _ if text.trim_start().starts_with('{') => JobFormat::Json,
        _ => JobFormat::Toml,
    }
}

/// Job parse / validation / execution failures.
#[derive(Debug)]
pub enum JobError {
    /// Job file could not be read.
    Io { path: PathBuf, message: String },
    /// Not valid TOML / JSON, or does not match the schema.
    Parse(String),
    /// Schema is well-formed but the job cannot run (flag conflicts, bad query doc, …).
    Invalid(String),
    /// A step failed; later steps did not run.
    Step {
        /// Zero-based step index.
        index: usize,
        op: &'static str,
        message: String,
        /// Steps that finished before the failure.
        completed: Vec<StepReport>,
    },
}

impl fmt::Display for JobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "{}: {message}", path.display()),
            Self::Parse(m) | Self::Invalid(m) => write!(f, "invalid job: {m}"),
            Self::Step {
                index, op, message, ..
            } => write!(f, "step {} ({op}) failed: {message}", index + 1),
        }
    }
}

impl std::error::Error for JobError {}

/// Knobs for [`run_job`] / [`run_job_path`].
#[derive(Debug, Clone, Default)]
pub struct RunJobOptions {
    /// Resolve relative paths against this directory. [`run_job_path`] defaults to the job
    /// file's parent; [`run_job`] defaults to the current directory.
    pub root: Option<PathBuf>,
    /// Validate and describe every step; execute nothing.
    pub dry_run: bool,
    /// Allow `semantics.wizard` (the `tet` binary sets this when stdin is a TTY).
    pub allow_interactive: bool,
    /// Append query steps to the platform query history (default off).
    pub record_query_history: bool,
}

/// Outcome of one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    /// Dry run: validated, not executed.
    Planned,
    Ok,
}

/// Result of one step.
#[derive(Debug, Clone, Serialize)]
pub struct StepReport {
    /// Zero-based step index.
    pub index: usize,
    pub op: &'static str,
    pub status: StepStatus,
    /// One-line plan with resolved paths.
    pub description: String,
    pub elapsed_secs: f64,
    /// Text the matching `tet` command prints to stdout (query response, `info`, …).
    pub stdout: Option<String>,
    /// Text the matching `tet` command prints to stderr (progress / summary lines).
    pub note: Option<String>,
    /// Structured result where one exists (convert / export / verify report, query response).
    pub data: Option<serde_json::Value>,
}

/// Result of a whole job.
#[derive(Debug, Clone, Serialize)]
pub struct JobReport {
    pub dry_run: bool,
    pub root: PathBuf,
    pub steps: Vec<StepReport>,
    pub elapsed_secs: f64,
}

impl Job {
    /// Checks that need no filesystem: schema version, at least one step, per-step flag rules.
    ///
    /// # Errors
    ///
    /// [`JobError::Invalid`] naming the first offending step.
    pub fn validate(&self) -> Result<(), JobError> {
        if self.schema_version != JOB_SCHEMA_VERSION {
            return Err(JobError::Invalid(format!(
                "unsupported schema_version {} (expected {JOB_SCHEMA_VERSION})",
                self.schema_version
            )));
        }
        if self.steps.is_empty() {
            return Err(JobError::Invalid("job has no steps".into()));
        }
        for (i, step) in self.steps.iter().enumerate() {
            step.validate()
                .map_err(|m| JobError::Invalid(format!("step {} ({}): {m}", i + 1, step.op())))?;
        }
        Ok(())
    }
}

/// Parse and statically validate a job document.
///
/// Accepts `[[step]]` (TOML) or `"steps": [...]` (JSON); either key works in either format.
///
/// # Errors
///
/// [`JobError::Parse`] for syntax / schema errors (unknown `op`, unknown field, wrong type);
/// [`JobError::Invalid`] from [`Job::validate`].
pub fn parse_job(text: &str, format: JobFormat) -> Result<Job, JobError> {
    let format = match format {
        JobFormat::Auto => detect_job_format(None, text),
        other => other,
    };
    let value: serde_json::Value = match format {
        JobFormat::Json | JobFormat::Auto => {
            serde_json::from_str(text).map_err(|e| JobError::Parse(e.to_string()))?
        }
        JobFormat::Toml => toml::from_str(text).map_err(|e| JobError::Parse(e.to_string()))?,
    };
    let job = job_from_value(value)?;
    job.validate()?;
    Ok(job)
}

fn job_from_value(value: serde_json::Value) -> Result<Job, JobError> {
    let serde_json::Value::Object(mut top) = value else {
        return Err(JobError::Parse("job must be a table / object".into()));
    };
    let schema_version = top
        .remove("schema_version")
        .ok_or_else(|| JobError::Parse("missing `schema_version`".into()))?
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| JobError::Parse("`schema_version` must be a positive integer".into()))?;
    let step_values = match (top.remove("steps"), top.remove("step")) {
        (Some(_), Some(_)) => {
            return Err(JobError::Parse(
                "use either `steps` or `step`, not both".into(),
            ));
        }
        (Some(v), None) | (None, Some(v)) => v,
        (None, None) => return Err(JobError::Parse("missing `[[step]]` / `steps`".into())),
    };
    if let Some(key) = top.keys().next() {
        return Err(JobError::Parse(format!("unknown top-level key `{key}`")));
    }
    let serde_json::Value::Array(items) = step_values else {
        return Err(JobError::Parse("`step` / `steps` must be an array".into()));
    };
    let mut steps = Vec::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        steps.push(step_from_value(i, item)?);
    }
    Ok(Job {
        schema_version,
        steps,
    })
}

fn step_from_value(index: usize, item: serde_json::Value) -> Result<Step, JobError> {
    let n = index + 1;
    let op = item
        .get("op")
        .ok_or_else(|| JobError::Parse(format!("step {n}: missing `op`")))?
        .as_str()
        .ok_or_else(|| JobError::Parse(format!("step {n}: `op` must be a string")))?
        .to_owned();
    if !OPS.contains(&op.as_str()) {
        return Err(JobError::Parse(format!(
            "step {n}: unknown op {op:?}; expected one of: {}",
            OPS.join(", ")
        )));
    }
    serde_json::from_value(item).map_err(|e| JobError::Parse(format!("step {n} ({op}): {e}")))
}

/// Read, parse, and run a job file. `opts.root` defaults to the file's parent directory.
///
/// # Errors
///
/// [`JobError::Io`] when the file cannot be read, plus everything [`run_job`] returns.
pub fn run_job_path(path: &Path, opts: &RunJobOptions) -> Result<JobReport, JobError> {
    run_job_path_observed(path, opts, |_| {})
}

/// [`run_job_path`] with a callback after each finished step (the CLI prints through this).
///
/// # Errors
///
/// Same as [`run_job_path`].
pub fn run_job_path_observed(
    path: &Path,
    opts: &RunJobOptions,
    on_step: impl FnMut(&StepReport),
) -> Result<JobReport, JobError> {
    let text = std::fs::read_to_string(path).map_err(|e| JobError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    let job = parse_job(&text, detect_job_format(Some(path), &text))?;
    let mut opts = opts.clone();
    if opts.root.is_none() {
        let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
        opts.root = Some(parent.map_or_else(|| PathBuf::from("."), Path::to_path_buf));
    }
    run_job_observed(&job, &opts, on_step)
}

/// Run (or, with `opts.dry_run`, plan) a parsed job.
///
/// Pre-flight runs first and fails before any step executes: query documents are loaded and
/// validated, and `semantics.wizard` is refused unless `allow_interactive` is set. Then steps run
/// in order; the first failure stops the job.
///
/// # Errors
///
/// [`JobError::Invalid`] from pre-flight, or [`JobError::Step`] for the first failing step.
pub fn run_job(job: &Job, opts: &RunJobOptions) -> Result<JobReport, JobError> {
    run_job_observed(job, opts, |_| {})
}

/// [`run_job`] with a callback after each finished (or, in dry run, planned) step.
///
/// # Errors
///
/// Same as [`run_job`].
pub fn run_job_observed(
    job: &Job,
    opts: &RunJobOptions,
    mut on_step: impl FnMut(&StepReport),
) -> Result<JobReport, JobError> {
    job.validate()?;
    let root = opts.root.clone().unwrap_or_else(|| PathBuf::from("."));
    let prepared = exec::prepare(job, &root, opts)?;
    let started = Instant::now();
    let mut steps: Vec<StepReport> = Vec::with_capacity(job.steps.len());
    for (index, (step, docs)) in job.steps.iter().zip(&prepared).enumerate() {
        let description = exec::describe(step, &root);
        let step_started = Instant::now();
        let report = if opts.dry_run {
            StepReport {
                index,
                op: step.op(),
                status: StepStatus::Planned,
                description,
                elapsed_secs: 0.0,
                stdout: None,
                note: None,
                data: None,
            }
        } else {
            match exec::execute(step, &root, docs.as_ref(), opts) {
                Ok(out) => StepReport {
                    index,
                    op: step.op(),
                    status: StepStatus::Ok,
                    description,
                    elapsed_secs: step_started.elapsed().as_secs_f64(),
                    stdout: out.stdout,
                    note: out.note,
                    data: out.data,
                },
                Err(message) => {
                    return Err(JobError::Step {
                        index,
                        op: step.op(),
                        message,
                        completed: steps,
                    });
                }
            }
        };
        on_step(&report);
        steps.push(report);
    }
    Ok(JobReport {
        dry_run: opts.dry_run,
        root,
        steps,
        elapsed_secs: started.elapsed().as_secs_f64(),
    })
}
