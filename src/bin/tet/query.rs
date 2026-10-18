//! `tet query` execution.

#![allow(deprecated)]

use std::path::PathBuf;

use tetration::layout::mmap_file_read;
use tetration::query::{
    ExecutionDeviceHint, QueryDocument, QueryOutputFormat, QueryResponse, SpillPathAllowlist,
    append_cli_query_history, format_query_response, format_query_stderr_hints, plan_query_empty,
    plan_query_with_tet_mmap_ex, validate_query,
};

use crate::package_ops::resolve_tet_read_path;
use crate::util::cli_error;

/// Default preview cap when `--preview` is omitted.
pub(crate) const QUERY_PREVIEW_DEFAULT: usize = 64;

pub(crate) struct QueryRunOpts {
    pub doc: QueryDocument,
    pub tet: Option<PathBuf>,
    pub execute: bool,
    pub stdout: QueryOutputFormat,
    pub preview: Option<usize>,
    pub spill_allow: Vec<PathBuf>,
    pub device: Option<String>,
    pub record_history: bool,
}

/// Apply CLI `--device` over any `execution.device` in the query document.
pub(crate) fn merge_cli_device(
    doc: &mut QueryDocument,
    device: Option<String>,
) -> Result<(), String> {
    let Some(token) = device else {
        return Ok(());
    };
    let parsed = ExecutionDeviceHint::parse(&token).map_err(|e| e.to_string())?;
    let execution = doc.execution.get_or_insert_with(Default::default);
    execution.device = Some(parsed);
    Ok(())
}

pub(crate) fn resolve_execute_preview_limit(
    execute: bool,
    stdout: QueryOutputFormat,
    explicit: Option<usize>,
) -> Result<Option<usize>, String> {
    if !execute {
        if explicit.is_some() {
            return Err(cli_error("--preview requires -x / --execute"));
        }
        return Ok(None);
    }
    Ok(Some(explicit.unwrap_or(match stdout {
        QueryOutputFormat::Full | QueryOutputFormat::Json => QUERY_PREVIEW_DEFAULT,
        QueryOutputFormat::Stats
        | QueryOutputFormat::Plan
        | QueryOutputFormat::Quiet
        | QueryOutputFormat::Table => 0,
    })))
}

pub(crate) fn run_query(opts: QueryRunOpts) -> Result<(), String> {
    let QueryRunOpts {
        doc,
        tet,
        execute,
        stdout,
        preview,
        spill_allow,
        device,
        record_history,
    } = opts;
    let mut doc = doc;
    merge_cli_device(&mut doc, device).map_err(cli_error)?;
    validate_query(&doc).map_err(cli_error)?;
    let preview = resolve_execute_preview_limit(execute, stdout, preview).map_err(cli_error)?;
    let response = if let Some(uri) = tet.as_ref().and_then(|path| s3_package_uri(path)) {
        query_s3_package(&uri, &doc, execute, preview, &spill_allow)?
    } else {
        query_local_tet(&doc, tet.as_ref(), execute, preview, &spill_allow)?
    };
    let out = format_query_response(&response, stdout).map_err(cli_error)?;
    println!("{out}");
    if let Some(hint) = format_query_stderr_hints(&response) {
        eprint!("{hint}");
    }
    if record_history {
        let tet_display = tet.as_ref().map(|p| p.display().to_string());
        if let Err(e) = append_cli_query_history(&doc, tet_display.as_deref(), execute) {
            eprintln!("warning: query history not saved: {e}");
        }
    }
    Ok(())
}

fn s3_package_uri(path: &std::path::Path) -> Option<String> {
    let text = path.to_string_lossy();
    let trimmed = text.trim();
    let rest = trimmed.split_once('@').map_or(trimmed, |(_, after)| after);
    if rest.starts_with("s3://") {
        Some(trimmed.to_string())
    } else {
        None
    }
}

fn query_local_tet(
    doc: &QueryDocument,
    tet: Option<&PathBuf>,
    execute: bool,
    preview: Option<usize>,
    spill_allow: &[PathBuf],
) -> Result<QueryResponse, String> {
    let resolved_tet = tet.map(|path| resolve_tet_read_path(path)).transpose()?;
    let mut spill_owned = None;
    if execute {
        let path = resolved_tet.as_ref().ok_or_else(|| {
            cli_error("execute requires -t / --tet (or a saved history row with a .tet path)")
        })?;
        spill_owned = Some(
            SpillPathAllowlist::default_for_tet(path, spill_allow.iter().cloned())
                .map_err(cli_error)?,
        );
    }
    if let (Some(orig), Some(resolved)) = (tet, resolved_tet.as_ref()) {
        let path_display = orig.display().to_string();
        let mmap = mmap_file_read(resolved).map_err(cli_error)?;
        return plan_query_with_tet_mmap_ex(
            doc,
            Some(path_display.as_str()),
            &mmap,
            preview,
            spill_owned.as_ref(),
        )
        .map_err(cli_error);
    }
    if execute {
        return Err(cli_error("execute requires -t / --tet"));
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
        Err(cli_error(
            "s3:// query requires Cargo feature `tetration-s3` and TETRATION_S3_ENDPOINT",
        ))
    }
    #[cfg(feature = "tetration-s3")]
    {
        use tetration::catalog::open_s3_package;
        use tetration::query::{ExecuteQueryOptions, query_opened_s3_package};

        let opened = open_s3_package(uri).map_err(cli_error)?;
        let spill_owned = if execute && !spill_allow.is_empty() {
            let base = std::env::current_dir().map_err(cli_error)?;
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
        .map_err(cli_error)
    }
}
