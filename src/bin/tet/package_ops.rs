//! `tet ingest` / `tet publish` / `tet compact` / `tet pack` / `tet gc` for `.tet.d` packages.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use tetration::catalog::{
    PublishPartsOptions, TetPackage, dry_run_publish_parts_with_options, ingest_tet_file,
    list_sealed_parts, open_or_create_package, part_id_from_path, publish_parts_with_options,
};
use tetration::convert::resolve_parallel_jobs;

use crate::util::cli_error;

pub(crate) struct IngestOpts {
    pub inputs: Vec<PathBuf>,
    pub out: PathBuf,
    pub jobs: usize,
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

    let pkg = open_or_create_package(&opts.out).map_err(cli_error)?;
    let jobs = resolve_parallel_jobs(opts.jobs);
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
    pub abort: bool,
    pub dry_run: bool,
    pub replace: Vec<String>,
    pub quiet: bool,
}

pub(crate) fn run_publish(opts: &PublishOpts) -> Result<(), String> {
    let pkg = TetPackage::open(&opts.package).map_err(cli_error)?;
    if opts.abort && opts.dry_run {
        return Err(cli_error(
            "tet publish: --abort and --dry-run cannot be combined",
        ));
    }
    if opts.abort {
        let generation = opts
            .generation
            .ok_or_else(|| cli_error("tet publish --abort requires --generation G"))?;
        let current_before = pkg.current().map_err(cli_error)?;
        pkg.abort_publish(generation).map_err(cli_error)?;
        if !opts.quiet {
            println!(
                "aborted publish generation={} → {} (CURRENT={})",
                generation,
                opts.package.display(),
                current_before.map_or_else(|| "none".into(), |g| g.to_string())
            );
        }
        return Ok(());
    }
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

    let options = PublishPartsOptions {
        replace: opts.replace.clone(),
    };
    if opts.dry_run {
        let validation = dry_run_publish_parts_with_options(&pkg, generation, &part_ids, &options)
            .map_err(cli_error)?;
        if !opts.quiet {
            println!(
                "dry-run ok: generation={} parts={} → {} (CURRENT unchanged)",
                validation.generation,
                validation.part_ids.len(),
                opts.package.display()
            );
        }
        return Ok(());
    }

    let manifest =
        publish_parts_with_options(&pkg, generation, &part_ids, &options).map_err(cli_error)?;
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
    pub target: PathBuf,
    pub apply: bool,
    pub protect_generation: Vec<u64>,
    pub quiet: bool,
}

pub(crate) fn run_gc(opts: &GcOpts) -> Result<(), String> {
    let text = opts.target.to_string_lossy();
    let trimmed = text.trim();
    let rest = trimmed.split_once('@').map_or(trimmed, |(_, after)| after);
    if rest.starts_with("s3://") {
        return run_gc_s3(opts, trimmed);
    }
    if !opts.protect_generation.is_empty() {
        return Err(cli_error(
            "--protect-generation is for s3:// targets; local tet gc keeps CURRENT and CURRENT-1 only",
        ));
    }
    let pkg = TetPackage::open(&opts.target).map_err(cli_error)?;
    let plan = pkg
        .gc(&tetration::catalog::GcOptions { apply: opts.apply })
        .map_err(cli_error)?;
    if opts.quiet {
        return Ok(());
    }
    let mode = if opts.apply { "applied" } else { "dry-run" };
    if plan.is_empty() {
        println!("gc {mode}: nothing to remove → {}", opts.target.display());
        return Ok(());
    }
    println!(
        "gc {mode}: staging_sessions={} generations={:?} → {}",
        plan.staging_sessions.len(),
        plan.generations,
        opts.target.display()
    );
    for session in &plan.staging_sessions {
        println!("  staging {}", session.display());
    }
    for &g in &plan.generations {
        println!("  generation {g}");
    }
    Ok(())
}

fn run_gc_s3(opts: &GcOpts, uri: &str) -> Result<(), String> {
    #[cfg(not(feature = "tetration-s3"))]
    {
        let _ = (opts, uri);
        Err(cli_error(
            "s3:// gc requires Cargo feature `tetration-s3` and TETRATION_S3_ENDPOINT",
        ))
    }
    #[cfg(feature = "tetration-s3")]
    {
        use tetration::catalog::{
            GenerationPin, PruneOptions, S3Config, S3Store, parse_s3_package_uri, prune_store,
        };

        let location = parse_s3_package_uri(uri).map_err(cli_error)?;
        let mut pinned = opts.protect_generation.clone();
        if let GenerationPin::Explicit(generation) = location.generation {
            pinned.push(generation);
        }
        let config = S3Config::from_env(location.bucket, location.prefix).map_err(cli_error)?;
        let store = S3Store::connect(config).map_err(cli_error)?;
        if !opts.quiet {
            eprintln!(
                "note: s3 gc cannot see open pins in other processes. Pass --protect-generation G (or ?generation=G). CURRENT, CURRENT-1, and a live publish claim are kept. CURRENT is not changed."
            );
        }
        let report = prune_store(
            &store,
            &PruneOptions {
                apply: opts.apply,
                pinned_generations: pinned,
            },
        )
        .map_err(cli_error)?;
        if !opts.quiet {
            print_store_gc(uri, opts.apply, &report);
        }
        Ok(())
    }
}

#[cfg(feature = "tetration-s3")]
fn print_store_gc(target: &str, apply: bool, report: &tetration::catalog::PruneReport) {
    use tetration::catalog::{PruneRefuseReason, PruneTarget};

    let mode = if apply { "applied" } else { "dry-run" };
    if report.is_empty() && report.refused.is_empty() {
        println!("gc {mode}: nothing to remove → {target}");
        return;
    }
    println!(
        "gc {mode}: staging_sessions={} generations={:?} refused={} → {target}",
        report.staging_sessions.len(),
        report.generations,
        report.refused.len(),
    );
    for session in &report.staging_sessions {
        println!("  staging {session}");
    }
    for &generation in &report.generations {
        println!("  generation {generation}");
    }
    for refusal in &report.refused {
        let why = match refusal.reason {
            PruneRefuseReason::OpenPin => "open pin",
            PruneRefuseReason::ActivePublishClaim => "active publish claim",
        };
        match &refusal.target {
            PruneTarget::Generation(generation) => {
                println!("  refused generation {generation} ({why})");
            }
            PruneTarget::StagingSession(session) => {
                println!("  refused {session} ({why})");
            }
        }
    }
}
