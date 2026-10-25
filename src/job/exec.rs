//! Step execution: calls the same public library APIs as the `tet` CLI (no subprocess, no
//! progress bars, no printing).

use std::fmt::{Display, Write as _};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::catalog::{
    GcOptions, PackGenerationOptions, PublishPartsOptions, ScaffoldFormat, ScaffoldPolicy,
    StdWizardIo, TetPackage, WizardOptions, apply_semantics_toml, clear_embedded_indexes,
    dry_run_publish_parts_with_options, embed_indexes_in_tet, format_apply_report,
    format_lint_report, ingest_tet_file, lint_package_semantics, lint_tet_path, list_sealed_parts,
    open_or_create_package, parse_semantics_toml, part_id_from_path, publish_parts_with_options,
    read_package_semantics, read_tet_summary_v1, resolve_tet_read_path, run_semantics_wizard,
    scaffold_annotated_toml, scaffold_policy_from_path, set_dataset_attrs,
    write_package_semantics_from_packed,
};
use crate::convert::{
    ConvertOptions, ConvertProgress, convert_s3_object_to_tet, convert_to_tet_with_options,
    is_s3_convert_uri, resolve_parallel_jobs,
};
use crate::export::export_tet_with_progress;
use crate::layout::mmap_file_read;
use crate::query::cli::{
    DEFAULT_INFO_CHUNK_TABLE_LIMIT, InfoListFilter, InfoMetadataDisplay, InfoSection,
    InfoSectionRequest, QUERY_PREVIEW_DEFAULT, QueryOutputFormat, append_cli_query_history,
    format_info_json, format_info_quiet, format_info_text, format_query_response,
    format_query_stderr_hints, info_view_sections_from_flags,
};
use crate::query::{
    ExecutionDeviceHint, QueryDocument, QueryResponse, SpillPathAllowlist,
    detect_query_input_format, parse_query_json, parse_query_text, plan_query_empty,
    plan_query_with_tet_mmap_ex, validate_query,
};
use crate::repair::{
    RepairOptions, format_plan_json, format_plan_text, format_repair_json, format_repair_text,
    repair_from_verify_report, repair_plan, repair_tet_file,
};
use crate::verify::{
    VerifyOptions, format_verify_json, format_verify_quiet, format_verify_text,
    looks_like_tet_package, verify_tet_file, verify_tet_file_with_options,
    verify_tet_package_with_options,
};

use super::schema::{
    CompactStep, ConvertStep, ExportStep, GcStep, IndexStep, InfoStep, IngestStep, Job, PackStep,
    QueryStep, RepairStep, SemanticsApplyStep, SemanticsLintStep, SemanticsPackageWriteStep,
    SemanticsScaffoldStep, SemanticsSetStep, SemanticsWizardStep, Step, VerifyStep,
    effective_quiet,
};
use super::{JobError, RunJobOptions};

/// What one executed step produced.
#[derive(Default)]
pub(super) struct StepOutput {
    pub stdout: Option<String>,
    pub note: Option<String>,
    pub data: Option<serde_json::Value>,
}

impl StepOutput {
    fn note(text: impl Into<String>) -> Self {
        Self {
            note: Some(text.into()),
            ..Self::default()
        }
    }

    fn stdout(text: String) -> Self {
        Self {
            stdout: Some(text),
            ..Self::default()
        }
    }

    fn quiet_note(quiet: bool, text: impl Into<String>) -> Self {
        if quiet {
            Self::default()
        } else {
            Self::note(text)
        }
    }
}

/// Why a step failed: the message plus the resolved path the failing operation touched.
///
/// Becomes [`JobError::Step`] at the module boundary. Build it with [`fail`] / [`AtPath::at`]
/// (path attached once, at the call that knows it) or `?` from a plain `String` / `&str`.
#[derive(Debug)]
pub(super) struct StepFailure {
    pub message: String,
    pub path: Option<PathBuf>,
}

type StepResult<T> = Result<T, StepFailure>;

impl From<String> for StepFailure {
    fn from(message: String) -> Self {
        Self {
            message,
            path: None,
        }
    }
}

impl From<&str> for StepFailure {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

/// Failure at `path` (reported absolute). The path is dropped when `err` already spells it out.
fn fail(path: &Path, err: impl Display) -> StepFailure {
    let message = err.to_string();
    let named = message.contains(&path.display().to_string());
    StepFailure {
        message,
        path: (!named).then(|| std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf())),
    }
}

/// `result.at(&path)?` — attach the resolved path to a failing filesystem-ish operation.
trait AtPath<T> {
    fn at(self, path: &Path) -> StepResult<T>;
}

impl<T, E: Display> AtPath<T> for Result<T, E> {
    fn at(self, path: &Path) -> StepResult<T> {
        self.map_err(|e| fail(path, e))
    }
}

/// Failure reading `input`: name it only when it is missing (other errors may be about the
/// output side, where blaming the input would mislead).
fn fail_input(input: &Path, err: impl Display) -> StepFailure {
    if input.exists() {
        err.to_string().into()
    } else {
        fail(input, err)
    }
}

fn es(e: impl Display) -> String {
    e.to_string()
}

// ---------------------------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------------------------

/// `s3://…` (optionally `profile@s3://…`) targets are passed through untouched.
fn is_s3_target(text: &str) -> bool {
    let trimmed = text.trim();
    trimmed
        .split_once('@')
        .map_or(trimmed, |(_, after)| after)
        .starts_with("s3://")
}

fn abs(root: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        root.join(p)
    }
}

/// Like [`abs`], but leaves `s3://` targets alone.
fn abs_or_s3(root: &Path, p: &Path) -> PathBuf {
    if is_s3_target(&p.to_string_lossy()) {
        p.to_path_buf()
    } else {
        abs(root, p)
    }
}

fn ensure_parent(path: &Path) -> StepResult<()> {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() && !parent.exists() => {
            fs::create_dir_all(parent).at(parent)
        }
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------------------------
// Describe (dry-run plan lines)
// ---------------------------------------------------------------------------------------------

/// Prefer job-relative paths in plan lines (absolute / off-root / s3 unchanged).
fn display_path(root: &Path, path: &Path) -> String {
    let abs_path = abs_or_s3(root, path);
    if is_s3_target(&abs_path.to_string_lossy()) {
        return abs_path.display().to_string();
    }
    abs_path.strip_prefix(root).map_or_else(
        |_| abs_path.display().to_string(),
        |rel| rel.display().to_string(),
    )
}

pub(super) fn describe(step: &Step, root: &Path) -> String {
    let p = |path: &Path| display_path(root, path);
    match step {
        Step::Info(s) => format!("info {}", p(&s.path)),
        Step::Verify(s) => format!(
            "verify {}{}{}",
            p(&s.path),
            if s.deep { " --deep" } else { "" },
            if s.repair { " --repair" } else { "" }
        ),
        Step::Repair(s) if s.apply.is_empty() => format!("repair plan {}", p(&s.path)),
        Step::Repair(s) => format!(
            "repair {} apply [{}]{}",
            p(&s.path),
            s.apply.join(", "),
            if s.dry_run { " (dry-run)" } else { "" }
        ),
        Step::Query(s) => describe_query(s, root),
        Step::Convert(s) => {
            let from = if is_s3_convert_uri(&s.from) {
                s.from.clone()
            } else {
                p(Path::new(&s.from))
            };
            format!("convert {from} → {}", p(&s.to))
        }
        Step::Export(s) => format!("export {} → {}", p(&s.from), p(&s.to)),
        Step::Index(s) => format!(
            "index {}{}",
            p(&s.path),
            if s.clear { " --clear" } else { "" }
        ),
        Step::SemanticsApply(s) => {
            format!("semantics apply {} --from {}", p(&s.path), p(&s.from))
        }
        Step::SemanticsScaffold(s) => format!("semantics scaffold {}", p(&s.path)),
        Step::SemanticsSet(s) => format!(
            "semantics set {} dataset={} (+{} -{})",
            p(&s.path),
            s.dataset,
            s.set.len(),
            s.clear.len()
        ),
        Step::SemanticsWizard(s) => format!("semantics wizard {} (interactive)", p(&s.path)),
        Step::SemanticsLint(s) => format!("semantics lint {}", p(&s.path)),
        Step::SemanticsPackageWrite(s) => format!("semantics package-write {}", p(&s.path)),
        Step::Ingest(s) => format!("ingest {} file(s) → {}", s.from.len(), p(&s.to)),
        Step::Publish(s) => format!(
            "publish {}{}",
            p(&s.package),
            if s.abort {
                " (abort)"
            } else if s.dry_run {
                " (dry-run)"
            } else {
                ""
            }
        ),
        Step::Compact(s) => format!("compact {}", p(&s.package)),
        Step::Pack(s) => format!("pack {} → {}", p(&s.from), p(&s.to)),
        Step::Gc(s) => format!(
            "gc {} ({})",
            p(&s.target),
            if s.apply { "apply" } else { "dry-run" }
        ),
    }
}

fn describe_query(s: &QueryStep, root: &Path) -> String {
    let source = match (&s.doc, &s.query) {
        (Some(doc), _) => display_path(root, doc),
        (None, Some(q)) => q
            .get("dataset")
            .and_then(|v| v.as_str())
            .map_or_else(|| "inline".to_owned(), |d| format!("dataset={d}")),
        (None, None) => "query".to_owned(),
    };
    match &s.tet {
        None => format!("query {source} (plan)"),
        Some(t) => format!(
            "query {source} @ {}{}",
            display_path(root, t),
            if s.execute { " -x" } else { "" }
        ),
    }
}

// ---------------------------------------------------------------------------------------------
// Pre-flight
// ---------------------------------------------------------------------------------------------

/// Validate everything that can be checked before step 1 runs. Returns, per step, the parsed
/// query document for `query` steps.
pub(super) fn prepare(
    job: &Job,
    root: &Path,
    opts: &RunJobOptions,
) -> Result<Vec<Option<QueryDocument>>, JobError> {
    job.steps
        .iter()
        .enumerate()
        .map(|(i, step)| {
            prepare_step(step, root, opts)
                .map_err(|m| JobError::Invalid(format!("step {} ({}): {m}", i + 1, step.op())))
        })
        .collect()
}

fn prepare_step(
    step: &Step,
    root: &Path,
    opts: &RunJobOptions,
) -> Result<Option<QueryDocument>, String> {
    match step {
        Step::Query(s) => load_query_doc(s, root).map(Some),
        Step::SemanticsWizard(_) if !opts.allow_interactive => Err(
            "semantics.wizard is interactive; run `tet run` from a terminal (stdin TTY) or \
             remove the step"
                .into(),
        ),
        _ => Ok(None),
    }
}

fn load_query_doc(s: &QueryStep, root: &Path) -> Result<QueryDocument, String> {
    let mut doc = match (&s.doc, &s.query) {
        (Some(path), None) => {
            let path = abs(root, path);
            let text = fs::read_to_string(&path)
                .map_err(|e| format!("query doc {}: {e}", path.display()))?;
            let hint = path.to_string_lossy();
            let format = detect_query_input_format(Some(&hint), &text);
            parse_query_text(text.trim(), format)
                .map_err(|e| format!("query doc {}: {e}", path.display()))?
        }
        (None, Some(value)) => {
            let text = serde_json::to_string(value).map_err(es)?;
            parse_query_json(&text).map_err(|e| format!("inline query: {e}"))?
        }
        _ => return Err("requires either `doc` or inline `query`".into()),
    };
    if let Some(token) = &s.device {
        let parsed = ExecutionDeviceHint::parse(token).map_err(es)?;
        doc.execution.get_or_insert_with(Default::default).device = Some(parsed);
    }
    validate_query(&doc).map_err(es)?;
    Ok(doc)
}

// ---------------------------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------------------------

pub(super) fn execute(
    step: &Step,
    root: &Path,
    doc: Option<&QueryDocument>,
    opts: &RunJobOptions,
) -> StepResult<StepOutput> {
    match step {
        Step::Info(s) => run_info(s, root),
        Step::Verify(s) => run_verify(s, root),
        Step::Repair(s) => run_repair(s, root),
        Step::Query(s) => {
            let doc = doc.ok_or("internal: query document was not prepared")?;
            run_query(s, doc, root, opts)
        }
        Step::Convert(s) => run_convert(s, root),
        Step::Export(s) => run_export(s, root),
        Step::Index(s) => run_index(s, root),
        Step::SemanticsApply(s) => run_semantics_apply(s, root),
        Step::SemanticsScaffold(s) => run_semantics_scaffold(s, root),
        Step::SemanticsSet(s) => run_semantics_set(s, root),
        Step::SemanticsWizard(s) => run_semantics_wizard_step(s, root),
        Step::SemanticsLint(s) => run_semantics_lint(s, root),
        Step::SemanticsPackageWrite(s) => run_semantics_package_write(s, root),
        Step::Ingest(s) => run_ingest(s, root),
        Step::Publish(s) => run_publish(&PublishArgs {
            package: abs(root, &s.package),
            generation: s.generation,
            parts: s.parts.clone(),
            abort: s.abort,
            dry_run: s.dry_run,
            replace: s.replace.clone(),
            quiet: s.quiet,
        }),
        Step::Compact(s) => run_compact(s, root),
        Step::Pack(s) => run_pack(s, root),
        Step::Gc(s) => run_gc(s, root),
    }
}

// ---------------------------------------------------------------------------------------------
// info / verify / repair
// ---------------------------------------------------------------------------------------------

fn run_info(s: &InfoStep, root: &Path) -> StepResult<StepOutput> {
    let path = abs(root, &s.path);
    let resolved = resolve_tet_read_path(&path).at(&path)?;
    let mmap = mmap_file_read(&resolved).at(&resolved)?;
    let file_len = u64::try_from(mmap.len())
        .map_err(|_| format!("file size {} exceeds u64::MAX", mmap.len()))?;
    let summary = read_tet_summary_v1(&mmap).at(&resolved)?;
    let filter = InfoListFilter {
        dataset: s.dataset.clone(),
        grep: s.grep.clone(),
    };
    let filter = (!filter.is_empty()).then_some(filter);
    let path_ref = Some(path.as_path());
    let out = if s.json {
        format_info_json(path_ref, file_len, &summary, filter.as_ref())?
    } else if effective_quiet(s.quiet, false) {
        format_info_quiet(path_ref, file_len, &summary, filter.as_ref())
    } else {
        let mut req = InfoSectionRequest::default();
        if s.all {
            req = req.with_all();
        }
        for (on, section) in [
            (s.layout, InfoSection::Layout),
            (s.execution, InfoSection::Execution),
            (s.datasets, InfoSection::Datasets),
            (s.chunks, InfoSection::Chunks),
            (s.history, InfoSection::History),
        ] {
            if on {
                req = req.enable(section);
            }
        }
        format_info_text(
            path_ref,
            file_len,
            &summary,
            filter.as_ref(),
            info_view_sections_from_flags(req),
            s.limit.unwrap_or(DEFAULT_INFO_CHUNK_TABLE_LIMIT),
            if s.metadata {
                InfoMetadataDisplay::Verbose
            } else {
                InfoMetadataDisplay::WhenPresent
            },
        )
    };
    Ok(StepOutput::stdout(out))
}

fn verify_once(path: &Path, opts: VerifyOptions) -> StepResult<crate::verify::TetVerifyReport> {
    if looks_like_tet_package(path) {
        verify_tet_package_with_options(path, opts).at(path)
    } else {
        verify_tet_file_with_options(path, opts).at(path)
    }
}

fn run_verify(s: &VerifyStep, root: &Path) -> StepResult<StepOutput> {
    let path = abs(root, &s.path);
    let verify_opts = VerifyOptions {
        deep_decode: s.deep,
    };
    let mut report = verify_once(&path, verify_opts)?;
    let mut note = None;
    if s.repair && !report.ok {
        let repaired = repair_from_verify_report(&path, &report, false).at(&path)?;
        note = Some(format_repair_text(&repaired));
        report = verify_once(&path, verify_opts)?;
    }
    let out = if s.json {
        format_verify_json(&report).map_err(es)?
    } else if effective_quiet(s.quiet, false) {
        format_verify_quiet(&report)
    } else {
        format_verify_text(&report)
    };
    if !report.ok {
        return Err(format!("verification failed\n{}", out.trim_end()).into());
    }
    Ok(StepOutput {
        stdout: Some(out),
        note,
        data: serde_json::to_value(&report).ok(),
    })
}

fn run_repair(s: &RepairStep, root: &Path) -> StepResult<StepOutput> {
    let path = abs(root, &s.path);
    if s.apply.is_empty() {
        let verify = verify_tet_file(&path).at(&path)?;
        let plan = repair_plan(&path, &verify);
        let out = if s.json {
            format_plan_json(&plan).map_err(es)?
        } else {
            format_plan_text(&plan)
        };
        return Ok(StepOutput::stdout(out));
    }
    let repair_opts = RepairOptions {
        dry_run: s.dry_run,
        apply: s.apply.clone(),
        plan_codes: Vec::new(),
    };
    let report = repair_tet_file(&path, &repair_opts).at(&path)?;
    let out = if s.json {
        format_repair_json(&report).map_err(es)?
    } else {
        format_repair_text(&report)
    };
    if !report.dry_run && report.verify_after_ok == Some(false) {
        return Err(fail(&path, "repair finished but verification still failed"));
    }
    Ok(StepOutput::stdout(out))
}

// ---------------------------------------------------------------------------------------------
// query
// ---------------------------------------------------------------------------------------------

fn run_query(
    s: &QueryStep,
    doc: &QueryDocument,
    root: &Path,
    opts: &RunJobOptions,
) -> StepResult<StepOutput> {
    let stdout_format = if effective_quiet(s.quiet, s.format.is_some()) {
        QueryOutputFormat::Quiet
    } else {
        s.format.unwrap_or_default()
    };
    let preview = resolve_preview(s.execute, stdout_format, s.preview)?;
    let tet = s.tet.as_ref().map(|t| abs_or_s3(root, t));
    let spill_allow: Vec<PathBuf> = s.spill_allow.iter().map(|p| abs(root, p)).collect();
    let response = if let Some(uri) = tet.as_ref().and_then(|t| s3_package_uri(t)) {
        query_s3_package(&uri, doc, s.execute, preview, &spill_allow)?
    } else {
        query_local_tet(doc, tet.as_ref(), s.execute, preview, &spill_allow)?
    };
    let out = format_query_response(&response, stdout_format)?;
    let mut note = format_query_stderr_hints(&response);
    if opts.record_query_history {
        let tet_display = tet.as_ref().map(|p| p.display().to_string());
        if let Err(e) = append_cli_query_history(doc, tet_display.as_deref(), s.execute) {
            let line = format!("warning: query history not saved: {e}");
            note = Some(note.map_or_else(|| line.clone(), |n| format!("{n}\n{line}")));
        }
    }
    Ok(StepOutput {
        stdout: Some(out),
        note,
        data: serde_json::to_value(&response).ok(),
    })
}

fn resolve_preview(
    execute: bool,
    stdout: QueryOutputFormat,
    explicit: Option<usize>,
) -> Result<Option<usize>, String> {
    if !execute {
        return if explicit.is_some() {
            Err("`preview` requires `execute = true`".into())
        } else {
            Ok(None)
        };
    }
    Ok(Some(explicit.unwrap_or(match stdout {
        QueryOutputFormat::Full | QueryOutputFormat::Json => QUERY_PREVIEW_DEFAULT,
        QueryOutputFormat::Stats
        | QueryOutputFormat::Plan
        | QueryOutputFormat::Quiet
        | QueryOutputFormat::Table
        | QueryOutputFormat::TableVerbose => 0,
    })))
}

fn s3_package_uri(path: &Path) -> Option<String> {
    let text = path.to_string_lossy();
    is_s3_target(&text).then(|| text.trim().to_owned())
}

fn query_local_tet(
    doc: &QueryDocument,
    tet: Option<&PathBuf>,
    execute: bool,
    preview: Option<usize>,
    spill_allow: &[PathBuf],
) -> StepResult<QueryResponse> {
    let resolved_tet = tet
        .map(|path| resolve_tet_read_path(path).at(path))
        .transpose()?;
    let mut spill_owned = None;
    if execute {
        let path = resolved_tet.as_ref().ok_or("`execute` requires `tet`")?;
        spill_owned = Some(
            SpillPathAllowlist::default_for_tet(path, spill_allow.iter().cloned()).map_err(es)?,
        );
    }
    if let (Some(orig), Some(resolved)) = (tet, resolved_tet.as_ref()) {
        // Plan against the sealed packed.tet but echo the user-facing path as `tet_file`.
        let path_for_io = resolved.display().to_string();
        let mmap = mmap_file_read(resolved).at(resolved)?;
        let mut response = plan_query_with_tet_mmap_ex(
            doc,
            Some(path_for_io.as_str()),
            &mmap,
            preview,
            spill_owned.as_ref(),
        )
        .map_err(es)?;
        response.tet_file = Some(orig.display().to_string());
        return Ok(response);
    }
    Ok(plan_query_empty(doc))
}

fn query_s3_package(
    uri: &str,
    doc: &QueryDocument,
    execute: bool,
    preview: Option<usize>,
    spill_allow: &[PathBuf],
) -> Result<QueryResponse, String> {
    #[cfg(not(feature = "tetration-s3"))]
    {
        let _ = (uri, doc, execute, preview, spill_allow);
        Err("s3:// query requires Cargo feature `tetration-s3` and TETRATION_S3_ENDPOINT".into())
    }
    #[cfg(feature = "tetration-s3")]
    {
        use crate::catalog::open_s3_package;
        use crate::query::{ExecuteQueryOptions, query_opened_s3_package};

        let opened = open_s3_package(uri).map_err(es)?;
        let spill_owned = if execute && !spill_allow.is_empty() {
            let base = std::env::current_dir().map_err(es)?;
            Some(SpillPathAllowlist::from_roots(
                base,
                spill_allow.iter().cloned(),
            ))
        } else {
            None
        };
        query_opened_s3_package(
            doc,
            &opened,
            execute,
            ExecuteQueryOptions {
                preview_max: preview,
            },
            spill_owned.as_ref(),
            Some(uri),
        )
        .map_err(es)
    }
}

// ---------------------------------------------------------------------------------------------
// convert / export / index
// ---------------------------------------------------------------------------------------------

fn run_convert(s: &ConvertStep, root: &Path) -> StepResult<StepOutput> {
    let out = abs(root, &s.to);
    ensure_parent(&out)?;
    let options = ConvertOptions {
        parallel_jobs: s.jobs,
        strict: s.strict,
        chunk_mib: s.chunk_mib,
    };
    let report = if is_s3_convert_uri(&s.from) {
        convert_s3_object_to_tet(&s.from, &out, options, None::<fn(ConvertProgress)>).map_err(es)?
    } else {
        let input = abs(root, Path::new(&s.from));
        convert_to_tet_with_options(&input, &out, options, None::<fn(ConvertProgress)>)
            .map_err(|e| fail_input(&input, e))?
    };
    let mut note = format!("{} dataset(s)", report.dataset_count);
    for skip in &report.skipped {
        let _ = write!(note, "\nskipped {}: {}", skip.path, skip.reason);
    }
    Ok(StepOutput {
        note: Some(note),
        data: serde_json::to_value(&report).ok(),
        ..StepOutput::default()
    })
}

fn run_export(s: &ExportStep, root: &Path) -> StepResult<StepOutput> {
    let input = abs(root, &s.from);
    let out = abs(root, &s.to);
    ensure_parent(&out)?;
    let report =
        export_tet_with_progress(&input, &out, None::<fn(_)>).map_err(|e| fail_input(&input, e))?;
    Ok(StepOutput {
        note: Some(format!(
            "exported {} dataset(s), {} chunk(s) → {} ({:.2}s)",
            report.dataset_count, report.chunks_written, report.output, report.elapsed_secs
        )),
        data: serde_json::to_value(&report).ok(),
        ..StepOutput::default()
    })
}

fn run_index(s: &IndexStep, root: &Path) -> StepResult<StepOutput> {
    let path = abs(root, &s.path);
    if s.clear {
        clear_embedded_indexes(&path).at(&path)?;
        return Ok(StepOutput::quiet_note(
            s.quiet,
            format!("cleared embedded ZONE/INDEX → {}", path.display()),
        ));
    }
    let embedded =
        embed_indexes_in_tet(&path, crate::catalog::EMBEDDED_STANDALONE_GENERATION).at(&path)?;
    let index = if embedded.secondary_index.bytes.is_some() {
        "INDEX present"
    } else {
        "INDEX omitted"
    };
    let sha = if embedded.packed_sha256 == crate::catalog::STANDALONE_PACKED_SHA256_PLACEHOLDER {
        "standalone".to_owned()
    } else {
        embedded.packed_sha256[..12.min(embedded.packed_sha256.len())].to_owned()
    };
    Ok(StepOutput::quiet_note(
        s.quiet,
        format!(
            "embedded ZONE + {index} → {} (packed_sha256={sha})",
            path.display()
        ),
    ))
}

// ---------------------------------------------------------------------------------------------
// semantics
// ---------------------------------------------------------------------------------------------

fn same_path(a: &Path, b: &Path) -> bool {
    a == b || matches!((a.canonicalize(), b.canonicalize()), (Ok(x), Ok(y)) if x == y)
}

fn write_text(out: &Path, text: &str) -> StepResult<()> {
    ensure_parent(out)?;
    fs::write(out, text).at(out)
}

fn run_semantics_apply(s: &SemanticsApplyStep, root: &Path) -> StepResult<StepOutput> {
    let path = abs(root, &s.path);
    let from = abs(root, &s.from);
    let output = s.output.as_ref().map(|o| abs(root, o));
    if let Some(out) = &output
        && same_path(out, &from)
    {
        return Err(
            "`output` must not be the `from` file (the source TOML is never rewritten)".into(),
        );
    }
    let report = apply_semantics_toml(&path, &from).map_err(|e| fail_input(&from, e))?;
    let note = format_apply_report(&path, &report);
    let mut step_out = StepOutput::note(note.trim_end());
    if s.annotate {
        let toml_text = fs::read_to_string(&from).at(&from)?;
        let doc = parse_semantics_toml(&toml_text).at(&from)?;
        let mmap = mmap_file_read(&path).at(&path)?;
        let summary = read_tet_summary_v1(&mmap).at(&path)?;
        let overlay = ScaffoldPolicy {
            profiles: doc.profiles,
            units: doc.units,
        };
        let text = scaffold_annotated_toml(
            &summary,
            Some(&overlay).filter(|p| !p.profiles.is_empty() || !p.units.is_empty()),
        );
        if let Some(out) = output {
            write_text(&out, &text)?;
        } else {
            step_out.stdout = Some(text);
        }
    }
    Ok(step_out)
}

fn run_semantics_scaffold(s: &SemanticsScaffoldStep, root: &Path) -> StepResult<StepOutput> {
    let path = abs(root, &s.path);
    let format = s.format.unwrap_or(ScaffoldFormat::Toml);
    let text = scaffold_policy_from_path(&path, format).at(&path)?;
    if let Some(out) = &s.output {
        write_text(&abs(root, out), &text)?;
        return Ok(StepOutput::default());
    }
    Ok(StepOutput::stdout(text))
}

fn run_semantics_set(s: &SemanticsSetStep, root: &Path) -> StepResult<StepOutput> {
    let path = abs(root, &s.path);
    let pairs: Vec<(String, String)> = s
        .set
        .iter()
        .map(|(k, v)| (k.clone(), v.to_attr_string()))
        .collect();
    set_dataset_attrs(&path, &s.dataset, &pairs, &s.clear).at(&path)?;
    Ok(StepOutput::note(format!(
        "semantics: updated dataset {:?} on {}",
        s.dataset,
        path.display()
    )))
}

fn run_semantics_wizard_step(s: &SemanticsWizardStep, root: &Path) -> StepResult<StepOutput> {
    let path = abs(root, &s.path);
    let mut opts = WizardOptions::beside_tet(&path);
    opts.dataset.clone_from(&s.dataset);
    opts.apply = !s.no_apply;
    if let Some(out) = &s.output {
        opts.toml_out = abs(root, out);
    }
    let mut io = StdWizardIo::new();
    let result = run_semantics_wizard(&path, &opts, &mut io).at(&path)?;
    Ok(StepOutput::note(format!(
        "wizard: done dataset {:?} → {}",
        result.dataset,
        result.toml_path.display()
    )))
}

fn resolve_generation(package: &TetPackage, generation: Option<u64>) -> Result<u64, String> {
    match generation {
        Some(g) => Ok(g),
        None => package
            .current()
            .map_err(es)?
            .ok_or_else(|| "package has no CURRENT generation; pass `generation`".to_owned()),
    }
}

fn run_semantics_lint(s: &SemanticsLintStep, root: &Path) -> StepResult<StepOutput> {
    let path = abs(root, &s.path);
    let report = if looks_like_tet_package(&path) {
        let package = TetPackage::open(&path).at(&path)?;
        let g = resolve_generation(&package, s.generation)?;
        if let Some(doc) = read_package_semantics(&package, g).at(&path)? {
            lint_package_semantics(&doc)
        } else {
            lint_tet_path(&package.packed_tet_path(g)).at(&path)?
        }
    } else {
        lint_tet_path(&path).at(&path)?
    };
    Ok(StepOutput::stdout(format_lint_report(&report)))
}

fn run_semantics_package_write(
    s: &SemanticsPackageWriteStep,
    root: &Path,
) -> StepResult<StepOutput> {
    let path = abs(root, &s.path);
    let package = TetPackage::open(&path).at(&path)?;
    let g = resolve_generation(&package, s.generation)?;
    let (doc, sha) = write_package_semantics_from_packed(&package, g, !s.no_manifest).at(&path)?;
    Ok(StepOutput::note(format!(
        "semantics: wrote generations/{g:08}/SEMANTICS.json ({} dataset(s), sha256={sha})",
        doc.datasets.len()
    )))
}

// ---------------------------------------------------------------------------------------------
// ingest / publish / compact / pack / gc
// ---------------------------------------------------------------------------------------------

fn run_ingest(s: &IngestStep, root: &Path) -> StepResult<StepOutput> {
    let inputs: Vec<PathBuf> = s.from.iter().map(|p| abs(root, p)).collect();
    for input in &inputs {
        if !input.is_file() {
            return Err(fail(input, "input not found"));
        }
    }
    let out = abs(root, &s.to);
    let pkg = open_or_create_package(&out).at(&out)?;
    let jobs = resolve_parallel_jobs(s.jobs);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .map_err(es)?;
    let counter = AtomicUsize::new(0);
    let sealed: StepResult<Vec<_>> = pool.install(|| {
        inputs
            .par_iter()
            .map(|input| {
                let base = part_id_from_path(input);
                let n = counter.fetch_add(1, Ordering::Relaxed);
                let part_id = if n == 0 && inputs.len() == 1 {
                    base
                } else {
                    format!("{base}-{n}")
                };
                let session_id = format!("ingest-{part_id}");
                let meta = ingest_tet_file(&pkg, input, &session_id, &part_id).at(input)?;
                Ok((input.clone(), meta))
            })
            .collect()
    });
    let sealed = sealed?;
    let mut note = format!(
        "ingested {} part(s) into {} (jobs={jobs})",
        sealed.len(),
        out.display()
    );
    for (src, meta) in &sealed {
        let _ = write!(
            note,
            "\n  {} → part_id={} datasets={}",
            src.display(),
            meta.part_id,
            meta.claims.len()
        );
    }
    Ok(StepOutput::quiet_note(s.quiet, note))
}

struct PublishArgs {
    package: PathBuf,
    generation: Option<u64>,
    parts: Option<Vec<String>>,
    abort: bool,
    dry_run: bool,
    replace: Vec<String>,
    quiet: bool,
}

fn run_compact(s: &CompactStep, root: &Path) -> StepResult<StepOutput> {
    let mut out = run_publish(&PublishArgs {
        package: abs(root, &s.package),
        generation: s.generation,
        parts: s.parts.clone(),
        abort: false,
        dry_run: false,
        replace: s.replace.clone(),
        quiet: s.quiet,
    })?;
    if !s.quiet {
        let line = "compact merges parts during publish (no separate visibility step)";
        out.note = Some(format!("note: {line}\n{}", out.note.unwrap_or_default()));
    }
    Ok(out)
}

fn run_publish(opts: &PublishArgs) -> StepResult<StepOutput> {
    let at = opts.package.as_path();
    let pkg = TetPackage::open(at).at(at)?;
    if opts.abort {
        let generation = opts.generation.ok_or("`abort` requires `generation`")?;
        let current_before = pkg.current().at(at)?;
        pkg.abort_publish(generation).at(at)?;
        return Ok(StepOutput::quiet_note(
            opts.quiet,
            format!(
                "aborted publish generation={generation} → {} (CURRENT={})",
                opts.package.display(),
                current_before.map_or_else(|| "none".into(), |g| g.to_string())
            ),
        ));
    }
    let generation = match opts.generation {
        Some(g) => g,
        None => pkg.next_generation().at(at)?,
    };
    let part_ids = match &opts.parts {
        Some(ids) if !ids.is_empty() => ids.clone(),
        _ => {
            let sealed = list_sealed_parts(&pkg).at(at)?;
            if sealed.is_empty() {
                return Err("no sealed staging parts to publish (run ingest first)".into());
            }
            sealed.into_iter().map(|(_, m)| m.part_id).collect()
        }
    };
    let options = PublishPartsOptions {
        replace: opts.replace.clone(),
    };
    if opts.dry_run {
        let validation =
            dry_run_publish_parts_with_options(&pkg, generation, &part_ids, &options).at(at)?;
        return Ok(StepOutput::quiet_note(
            opts.quiet,
            format!(
                "dry-run ok: generation={} parts={} → {} (CURRENT unchanged)",
                validation.generation,
                validation.part_ids.len(),
                opts.package.display()
            ),
        ));
    }
    let manifest = publish_parts_with_options(&pkg, generation, &part_ids, &options).at(at)?;
    Ok(StepOutput::quiet_note(
        opts.quiet,
        format!(
            "published generation={} parts={} packed_byte_len={} → {}\n  CURRENT → {}",
            manifest.generation,
            manifest.part_ids.len(),
            manifest.packed_byte_len,
            opts.package.display(),
            manifest.generation
        ),
    ))
}

fn run_pack(s: &PackStep, root: &Path) -> StepResult<StepOutput> {
    let package = abs(root, &s.from);
    let output = abs(root, &s.to);
    let pkg = TetPackage::open(&package).at(&package)?;
    let generation = match s.generation {
        Some(g) => g,
        None => pkg
            .current()
            .at(&package)?
            .ok_or_else(|| fail(&package, "package has no CURRENT generation"))?,
    };
    ensure_parent(&output)?;
    let pack_opts = PackGenerationOptions {
        remove_package: s.remove_package,
    };
    pkg.pack_generation_with(generation, &output, &pack_opts)
        .at(&package)?;
    let note = if s.remove_package {
        format!(
            "packed generation={generation} → {} (removed package {})",
            output.display(),
            package.display()
        )
    } else {
        format!("packed generation={generation} → {}", output.display())
    };
    Ok(StepOutput::quiet_note(s.quiet, note))
}

fn run_gc(s: &GcStep, root: &Path) -> StepResult<StepOutput> {
    let target = abs_or_s3(root, &s.target);
    let text = target.to_string_lossy().into_owned();
    if is_s3_target(&text) {
        return Ok(run_gc_s3(s, text.trim())?);
    }
    if !s.protect_generation.is_empty() {
        return Err(
            "`protect_generation` is for s3:// targets; local gc keeps CURRENT and CURRENT-1 only"
                .into(),
        );
    }
    let pkg = TetPackage::open(&target).at(&target)?;
    let plan = pkg.gc(&GcOptions { apply: s.apply }).at(&target)?;
    let mode = if s.apply { "applied" } else { "dry-run" };
    if plan.is_empty() {
        return Ok(StepOutput::quiet_note(
            s.quiet,
            format!("gc {mode}: nothing to remove → {}", target.display()),
        ));
    }
    let mut note = format!(
        "gc {mode}: staging_sessions={} generations={:?} → {}",
        plan.staging_sessions.len(),
        plan.generations,
        target.display()
    );
    for session in &plan.staging_sessions {
        let _ = write!(note, "\n  staging {}", session.display());
    }
    for &g in &plan.generations {
        let _ = write!(note, "\n  generation {g}");
    }
    Ok(StepOutput::quiet_note(s.quiet, note))
}

fn run_gc_s3(s: &GcStep, uri: &str) -> Result<StepOutput, String> {
    #[cfg(not(feature = "tetration-s3"))]
    {
        let _ = (s, uri);
        Err("s3:// gc requires Cargo feature `tetration-s3` and TETRATION_S3_ENDPOINT".into())
    }
    #[cfg(feature = "tetration-s3")]
    {
        use crate::catalog::{
            GenerationPin, PruneOptions, PruneRefuseReason, PruneTarget, S3Config, S3Store,
            parse_s3_package_uri, prune_store,
        };

        let location = parse_s3_package_uri(uri).map_err(es)?;
        let mut pinned = s.protect_generation.clone();
        if let GenerationPin::Explicit(generation) = location.generation {
            pinned.push(generation);
        }
        let config = S3Config::from_env(location.bucket, location.prefix).map_err(es)?;
        let store = S3Store::connect(config).map_err(es)?;
        let report = prune_store(
            &store,
            &PruneOptions {
                apply: s.apply,
                pinned_generations: pinned,
            },
        )
        .map_err(es)?;
        let mode = if s.apply { "applied" } else { "dry-run" };
        let mut note = format!(
            "gc {mode}: staging_sessions={} generations={:?} refused={} → {uri}",
            report.staging_sessions.len(),
            report.generations,
            report.refused.len(),
        );
        for session in &report.staging_sessions {
            let _ = write!(note, "\n  staging {session}");
        }
        for &generation in &report.generations {
            let _ = write!(note, "\n  generation {generation}");
        }
        for refusal in &report.refused {
            let why = match refusal.reason {
                PruneRefuseReason::OpenPin => "open pin",
                PruneRefuseReason::ActivePublishClaim => "active publish claim",
            };
            match &refusal.target {
                PruneTarget::Generation(generation) => {
                    let _ = write!(note, "\n  refused generation {generation} ({why})");
                }
                PruneTarget::StagingSession(session) => {
                    let _ = write!(note, "\n  refused {session} ({why})");
                }
            }
        }
        Ok(StepOutput::quiet_note(s.quiet, note))
    }
}
