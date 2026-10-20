//! `tet semantics` — declare attrs / scaffold caller policy.

use std::fs;
use std::path::Path;

use tetration::catalog::{ScaffoldFormat, scaffold_policy_from_path, set_dataset_attrs};

use crate::args::SemanticsCmd;
use crate::util::cli_error;

pub(crate) fn run_semantics(cmd: SemanticsCmd) -> Result<(), String> {
    match cmd {
        SemanticsCmd::Scaffold {
            path,
            output,
            format,
        } => run_scaffold(&path, output.as_deref(), &format),
        SemanticsCmd::Set {
            path,
            dataset,
            set,
            clear,
        } => run_set(&path, &dataset, &set, &clear),
    }
}

fn run_scaffold(path: &Path, output: Option<&Path>, format: &str) -> Result<(), String> {
    let fmt = ScaffoldFormat::parse(format).map_err(cli_error)?;
    let text = scaffold_policy_from_path(path, fmt).map_err(cli_error)?;
    if let Some(out) = output {
        if let Some(parent) = out.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(cli_error)?;
        }
        fs::write(out, text).map_err(cli_error)?;
    } else {
        print!("{text}");
    }
    Ok(())
}

fn run_set(path: &Path, dataset: &str, set: &[String], clear: &[String]) -> Result<(), String> {
    if set.is_empty() && clear.is_empty() {
        return Err(cli_error(
            "semantics set requires at least one --set KEY=VAL or --clear KEY",
        ));
    }
    let mut pairs = Vec::with_capacity(set.len());
    for s in set {
        let (k, v) = parse_set_pair(s)?;
        pairs.push((k, v));
    }
    set_dataset_attrs(path, dataset, &pairs, clear).map_err(cli_error)?;
    eprintln!(
        "semantics: updated dataset {dataset:?} on {}",
        path.display()
    );
    Ok(())
}

fn parse_set_pair(raw: &str) -> Result<(String, String), String> {
    let (k, v) = raw
        .split_once('=')
        .ok_or_else(|| cli_error(format!("--set expects KEY=VAL, got {raw:?}")))?;
    if k.is_empty() || v.is_empty() {
        return Err(cli_error(format!(
            "--set KEY and VAL must be non-empty, got {raw:?}"
        )));
    }
    Ok((k.to_owned(), v.to_owned()))
}
