//! `tet qhist` — platform query history (list, run).

#![allow(deprecated)]

use std::path::PathBuf;

use tetration::query::{
    HistoryListFilter, HistorySettings, clear_cli_query_history, format_history_list_json,
    format_history_list_text, get_cli_query_history_entry, parse_history_execute_filter,
};

use crate::args::{QhistCmd, QueryStdoutFormat};
use crate::query::{QueryRunOpts, run_query};
use crate::util::{cli_error, resolve_stdout};

pub(crate) fn build_history_filter(
    dataset: Option<String>,
    tet: Option<String>,
    mode: Option<String>,
    grep: Option<String>,
) -> Result<Option<HistoryListFilter>, String> {
    let mode = mode
        .map(|s| parse_history_execute_filter(&s))
        .transpose()
        .map_err(cli_error)?;
    let filter = HistoryListFilter {
        dataset,
        tet,
        mode,
        grep,
    };
    if filter.is_empty() {
        Ok(None)
    } else {
        Ok(Some(filter))
    }
}

fn run_qhist_list(
    limit: usize,
    all: bool,
    json: bool,
    filter_ref: Option<&HistoryListFilter>,
) -> Result<(), String> {
    let settings = HistorySettings::from_env();
    let path = settings.path();
    let entries = settings.list(limit, all, filter_ref).map_err(cli_error)?;
    let path_ref = path.as_deref();
    let out = if json {
        format_history_list_json(&entries, path_ref, &settings, filter_ref).map_err(cli_error)?
    } else {
        format_history_list_text(&entries, path_ref, &settings, filter_ref)
    };
    print!("{out}");
    Ok(())
}

struct QhistReplayOpts {
    entry: tetration::query::CliQueryHistoryEntry,
    tet: Option<PathBuf>,
    force_execute: bool,
    plan: bool,
    format: QueryStdoutFormat,
    quiet: bool,
    preview: Option<usize>,
    spill_allow: Vec<PathBuf>,
}

fn run_qhist_replay(opts: QhistReplayOpts) -> Result<(), String> {
    let stdout = resolve_stdout(opts.quiet, opts.format);
    let execute = if opts.plan {
        false
    } else if opts.force_execute {
        true
    } else {
        opts.entry.execute
    };
    let tet = opts.tet.or_else(|| opts.entry.tet.map(PathBuf::from));
    run_query(QueryRunOpts {
        doc: opts.entry.query,
        tet,
        execute,
        stdout,
        preview: opts.preview,
        spill_allow: opts.spill_allow,
        device: None,
        record_history: true,
    })
}

pub(crate) fn run_qhist(cmd: Option<QhistCmd>, clear: bool) -> Result<(), String> {
    if clear {
        clear_cli_query_history().map_err(cli_error)?;
        eprintln!("query history cleared");
        return Ok(());
    }
    match cmd {
        None => {
            let limit = HistorySettings::from_env().cli_query_max;
            run_qhist_list(limit, false, false, None)
        }
        Some(QhistCmd::List {
            limit,
            all,
            json,
            dataset,
            tet,
            mode,
            grep,
        }) => {
            let filter = build_history_filter(dataset, tet, mode, grep)?;
            run_qhist_list(limit, all, json, filter.as_ref())
        }
        Some(QhistCmd::Run {
            index,
            tet,
            execute,
            plan,
            format,
            quiet,
            preview,
            spill_allow,
            dataset,
            tet_filter,
            mode,
            grep,
        }) => {
            let filter = build_history_filter(dataset, tet_filter, mode, grep)?;
            let entry = get_cli_query_history_entry(index, filter.as_ref()).map_err(cli_error)?;
            run_qhist_replay(QhistReplayOpts {
                entry,
                tet,
                force_execute: execute,
                plan,
                format,
                quiet,
                preview,
                spill_allow,
            })
        }
    }
}
