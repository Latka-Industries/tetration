//! `tet semantics` — declare attrs / scaffold caller policy / package SEMANTICS.json.

use std::fs;
use std::path::Path;

use tetration::catalog::{
    ScaffoldFormat, TetPackage, apply_semantics_toml, encode_package_semantics,
    format_apply_report, format_lint_report, lint_package_semantics, lint_tet_path,
    read_package_semantics, scaffold_policy_from_path, set_dataset_attrs,
    write_package_semantics_from_packed,
};
use tetration::verify::looks_like_tet_package;

use crate::args::SemanticsCmd;
use crate::util::cli_error;

pub(crate) fn run_semantics(cmd: SemanticsCmd) -> Result<(), String> {
    match cmd {
        SemanticsCmd::Apply { path, from } => run_apply(&path, &from),
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
        SemanticsCmd::PackageWrite {
            path,
            generation,
            no_manifest,
        } => run_package_write(&path, generation, !no_manifest),
        SemanticsCmd::PackageShow { path, generation } => run_package_show(&path, generation),
        SemanticsCmd::Lint { path, generation } => run_lint(&path, generation),
    }
}

fn run_apply(path: &Path, from: &Path) -> Result<(), String> {
    let report = apply_semantics_toml(path, from).map_err(cli_error)?;
    eprint!("{}", format_apply_report(path, &report));
    Ok(())
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

fn resolve_generation(package: &TetPackage, generation: Option<u64>) -> Result<u64, String> {
    match generation {
        Some(g) => Ok(g),
        None => package
            .current()
            .map_err(cli_error)?
            .ok_or_else(|| cli_error("package has no CURRENT generation; pass --generation")),
    }
}

fn run_package_write(
    path: &Path,
    generation: Option<u64>,
    update_manifest: bool,
) -> Result<(), String> {
    let package = TetPackage::open(path).map_err(cli_error)?;
    let g = resolve_generation(&package, generation)?;
    let (doc, sha) =
        write_package_semantics_from_packed(&package, g, update_manifest).map_err(cli_error)?;
    eprintln!(
        "semantics: wrote generations/{g:08}/SEMANTICS.json ({} dataset(s), sha256={sha})",
        doc.datasets.len()
    );
    Ok(())
}

fn run_package_show(path: &Path, generation: Option<u64>) -> Result<(), String> {
    let package = TetPackage::open(path).map_err(cli_error)?;
    let g = resolve_generation(&package, generation)?;
    match read_package_semantics(&package, g).map_err(cli_error)? {
        None => Err(cli_error(format!(
            "generations/{g:08}/SEMANTICS.json not present"
        ))),
        Some(doc) => {
            let (bytes, _) = encode_package_semantics(&doc).map_err(cli_error)?;
            print!("{}", String::from_utf8_lossy(&bytes));
            Ok(())
        }
    }
}

fn run_lint(path: &Path, generation: Option<u64>) -> Result<(), String> {
    let report = if looks_like_tet_package(path) {
        let package = TetPackage::open(path).map_err(cli_error)?;
        let g = resolve_generation(&package, generation)?;
        if let Some(doc) = read_package_semantics(&package, g).map_err(cli_error)? {
            lint_package_semantics(&doc)
        } else {
            let packed = package.packed_tet_path(g);
            lint_tet_path(&packed).map_err(cli_error)?
        }
    } else {
        lint_tet_path(path).map_err(cli_error)?
    };
    print!("{}", format_lint_report(&report));
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
