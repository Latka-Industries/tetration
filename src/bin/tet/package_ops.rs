//! `tet ingest` / `tet publish` / `tet compact` / `tet pack` / `tet gc` for `.tet.d` packages.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use tetration::catalog::{
    TetPackage, ingest_tet_file, list_sealed_parts, part_id_from_path, publish_parts,
};
use tetration::verify::looks_like_tet_package;

use crate::util::cli_error;

/// Open an existing package, or create when `create` is set / path is missing.
pub(crate) fn open_or_create_package(path: &Path, create: bool) -> Result<TetPackage, String> {
    if looks_like_tet_package(path) {
        return TetPackage::open(path).map_err(cli_error);
    }
    if path.exists() {
        return Err(cli_error(format!(
            "{} exists but is not a .tet.d package (missing PROTOCOL)",
            path.display()
        )));
    }
    if create || !path.exists() {
        return TetPackage::create(path).map_err(cli_error);
    }
    Err(cli_error(format!(
        "package not found: {} (pass --create)",
        path.display()
    )))
}

/// Resolve `-t` / info path: package → sealed `packed.tet`, else the path itself.
pub(crate) fn resolve_tet_read_path(path: &Path) -> Result<PathBuf, String> {
    if !looks_like_tet_package(path) {
        return Ok(path.to_path_buf());
    }
    let pkg = TetPackage::open(path).map_err(cli_error)?;
    let pin = pkg.pin_current().map_err(cli_error)?;
    Ok(pin.packed_tet_path().to_path_buf())
}

pub(crate) struct IngestOpts {
    pub inputs: Vec<PathBuf>,
    pub out: PathBuf,
    pub jobs: usize,
    pub create: bool,
    pub quiet: bool,
}

pub(crate) fn run_ingest(opts: &IngestOpts) -> Result<(), String> {
    if opts.inputs.is_empty() {
        return Err(cli_error("ingest requires at least one input .tet"));
    }
    for input in &opts.inputs {
        if !input.is_file() {
            return Err(cli_error(format!("input not found: {}", input.display())));
        }
    }

    let pkg = open_or_create_package(&opts.out, opts.create)?;
    let jobs = resolve_jobs(opts.jobs);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .map_err(cli_error)?;

    let counter = AtomicUsize::new(0);
    let results: Result<Vec<_>, String> = pool.install(|| {
        opts.inputs
            .par_iter()
            .map(|input| {
                let base = part_id_from_path(input);
                let n = counter.fetch_add(1, Ordering::Relaxed);
                let part_id = if n == 0 && opts.inputs.len() == 1 {
                    base.clone()
                } else {
                    format!("{base}-{n}")
                };
                let session_id = format!("ingest-{part_id}");
                let meta =
                    ingest_tet_file(&pkg, input, &session_id, &part_id).map_err(cli_error)?;
                Ok((input.clone(), meta))
            })
            .collect()
    });
    let sealed = results?;

    if !opts.quiet {
        println!(
            "ingested {} part(s) into {} (jobs={jobs})",
            sealed.len(),
            opts.out.display()
        );
        for (src, meta) in &sealed {
            println!(
                "  {} → part_id={} datasets={}",
                src.display(),
                meta.part_id,
                meta.claims.len()
            );
        }
    }
    Ok(())
}

pub(crate) struct PublishOpts {
    pub package: PathBuf,
    pub generation: Option<u64>,
    pub parts: Option<Vec<String>>,
    pub quiet: bool,
}

pub(crate) fn run_publish(opts: &PublishOpts) -> Result<(), String> {
    let pkg = TetPackage::open(&opts.package).map_err(cli_error)?;
    let generation = match opts.generation {
        Some(g) => g,
        None => pkg.next_generation().map_err(cli_error)?,
    };
    let part_ids = match &opts.parts {
        Some(ids) if !ids.is_empty() => ids.clone(),
        _ => {
            let sealed = list_sealed_parts(&pkg).map_err(cli_error)?;
            if sealed.is_empty() {
                return Err(cli_error(
                    "no sealed staging parts to publish (run tet ingest first)",
                ));
            }
            sealed.into_iter().map(|(_, m)| m.part_id).collect()
        }
    };

    let manifest = publish_parts(&pkg, generation, &part_ids).map_err(cli_error)?;
    if !opts.quiet {
        println!(
            "published generation={} parts={} packed_byte_len={} → {}",
            manifest.generation,
            manifest.part_ids.len(),
            manifest.packed_byte_len,
            opts.package.display()
        );
        println!("  CURRENT → {}", manifest.generation);
    }
    Ok(())
}

/// `tet compact` — compact happens inside publish (decision I); forwards to [`run_publish`].
pub(crate) fn run_compact(opts: &PublishOpts) -> Result<(), String> {
    if !opts.quiet {
        eprintln!("note: compact merges parts during publish (no separate visibility step)");
    }
    run_publish(opts)
}

pub(crate) struct PackOpts {
    pub package: PathBuf,
    pub output: PathBuf,
    pub generation: Option<u64>,
    pub remove_package: bool,
    pub quiet: bool,
}

pub(crate) fn run_pack(opts: &PackOpts) -> Result<(), String> {
    let pkg = TetPackage::open(&opts.package).map_err(cli_error)?;
    let generation = match opts.generation {
        Some(g) => g,
        None => pkg
            .current()
            .map_err(cli_error)?
            .ok_or_else(|| cli_error("package has no CURRENT generation"))?,
    };
    if let Some(parent) = opts.output.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        fs::create_dir_all(parent).map_err(cli_error)?;
    }
    let pack_opts = tetration::catalog::PackGenerationOptions {
        remove_package: opts.remove_package,
    };
    pkg.pack_generation_with(generation, &opts.output, &pack_opts)
        .map_err(cli_error)?;
    if !opts.quiet {
        if opts.remove_package {
            println!(
                "packed generation={} → {} (removed package {})",
                generation,
                opts.output.display(),
                opts.package.display()
            );
        } else {
            println!(
                "packed generation={} → {}",
                generation,
                opts.output.display()
            );
        }
    }
    Ok(())
}

pub(crate) struct GcOpts {
    pub package: PathBuf,
    pub apply: bool,
    pub quiet: bool,
}

pub(crate) fn run_gc(opts: &GcOpts) -> Result<(), String> {
    let pkg = TetPackage::open(&opts.package).map_err(cli_error)?;
    let plan = pkg
        .gc(&tetration::catalog::GcOptions { apply: opts.apply })
        .map_err(cli_error)?;
    if opts.quiet {
        return Ok(());
    }
    let mode = if opts.apply { "applied" } else { "dry-run" };
    if plan.is_empty() {
        println!("gc {mode}: nothing to remove → {}", opts.package.display());
        return Ok(());
    }
    println!(
        "gc {mode}: staging_sessions={} generations={:?} → {}",
        plan.staging_sessions.len(),
        plan.generations,
        opts.package.display()
    );
    for session in &plan.staging_sessions {
        println!("  staging {}", session.display());
    }
    for &g in &plan.generations {
        println!("  generation {g}");
    }
    Ok(())
}

fn resolve_jobs(jobs: usize) -> usize {
    if jobs == 0 {
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(64)
    } else {
        jobs.clamp(1, 64)
    }
}
