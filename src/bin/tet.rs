//! `tet` — CLI for JSON queries and foreign-format conversion.
//!
//! Command bodies live in [`src/bin/tet/`](tet/) (separate source files). Cargo only
//! builds `tet.rs` as the binary; siblings under `src/bin/` would become extra binaries.

#[path = "tet/args.rs"]
mod args;
#[path = "tet/convert.rs"]
mod convert;
#[path = "tet/export.rs"]
mod export;
#[path = "tet/info.rs"]
mod info;
#[path = "tet/package_ops.rs"]
mod package_ops;
#[path = "tet/qhist.rs"]
mod qhist;
#[path = "tet/query.rs"]
mod query;
#[path = "tet/repair.rs"]
mod repair;
#[path = "tet/util.rs"]
mod util;
#[path = "tet/verify.rs"]
mod verify;

use std::process::ExitCode;

use clap::Parser;
use tetration::query::parse_query_text;

use args::{Cli, Commands};
use convert::run_convert;
use export::run_export;
use info::{InfoRunOpts, run_info};
use package_ops::{
    IngestOpts, PackOpts, PublishOpts, run_compact, run_ingest, run_pack, run_publish,
};
use qhist::run_qhist;
use query::{QueryRunOpts, run_query};
use repair::{RepairRunOpts, run_repair};
use util::{cli_error, query_input_format, read_query_payload, resolve_stdout};
use verify::{VerifyRunOpts, run_verify};

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        cmd @ (Commands::Info { .. }
        | Commands::Query { .. }
        | Commands::Qhist(_)
        | Commands::Verify { .. }
        | Commands::Repair { .. }) => run_read(cmd),
        cmd => run_write(cmd),
    }
}

fn run_read(cmd: Commands) -> Result<(), String> {
    match cmd {
        Commands::Info {
            path,
            json,
            quiet,
            all,
            layout,
            execution,
            datasets,
            chunks,
            show_footer_history,
            limit,
            dataset,
            grep,
            metadata,
        } => run_info(InfoRunOpts {
            path,
            json,
            quiet,
            all,
            layout,
            execution,
            datasets,
            chunks,
            show_footer_history,
            limit,
            dataset,
            grep,
            metadata,
        }),
        Commands::Query {
            query,
            tet,
            execute,
            format,
            quiet,
            preview,
            spill_allow,
            device,
        } => run_query_cmd(QueryCli {
            query,
            tet,
            execute,
            format,
            quiet,
            preview,
            spill_allow,
            device,
        }),
        Commands::Qhist(args) => run_qhist(args.cmd, args.clear),
        Commands::Verify {
            path,
            json,
            quiet,
            repair,
            deep,
        } => run_verify(&VerifyRunOpts {
            path,
            json,
            quiet,
            repair,
            deep,
        }),
        Commands::Repair {
            path,
            json,
            dry_run,
            apply,
        } => run_repair(&RepairRunOpts {
            path,
            json,
            dry_run,
            apply,
        }),
        Commands::Export { .. }
        | Commands::Convert { .. }
        | Commands::Ingest { .. }
        | Commands::Publish { .. }
        | Commands::Compact { .. }
        | Commands::Pack { .. } => unreachable!("read commands only"),
    }
}

struct QueryCli {
    query: Option<String>,
    tet: Option<std::path::PathBuf>,
    execute: bool,
    format: args::QueryStdoutFormat,
    quiet: bool,
    preview: Option<usize>,
    spill_allow: Vec<std::path::PathBuf>,
    device: Option<String>,
}

fn run_query_cmd(q: QueryCli) -> Result<(), String> {
    let stdout = resolve_stdout(q.quiet, q.format);
    let (raw, path_hint) = read_query_payload(q.query.as_deref()).map_err(cli_error)?;
    let trimmed = raw.trim();
    let format = query_input_format(path_hint.as_deref(), trimmed);
    let doc = parse_query_text(trimmed, format).map_err(cli_error)?;
    run_query(QueryRunOpts {
        doc,
        tet: q.tet,
        execute: q.execute,
        stdout,
        preview: q.preview,
        spill_allow: q.spill_allow,
        device: q.device,
        record_history: true,
    })
}

fn run_write(cmd: Commands) -> Result<(), String> {
    match cmd {
        Commands::Export { input, output } => run_export(&input, &output),
        Commands::Convert {
            input,
            output,
            jobs,
            strict,
            chunk_mib,
        } => run_convert(&input, &output, jobs, strict, chunk_mib),
        Commands::Ingest {
            inputs,
            out,
            jobs,
            create,
            quiet,
        } => run_ingest(&IngestOpts {
            inputs,
            out,
            jobs,
            create,
            quiet,
        }),
        Commands::Publish {
            package,
            generation,
            parts,
            quiet,
        } => run_publish(&PublishOpts {
            package,
            generation,
            parts,
            quiet,
        }),
        Commands::Compact {
            package,
            generation,
            parts,
            quiet,
        } => run_compact(&PublishOpts {
            package,
            generation,
            parts,
            quiet,
        }),
        Commands::Pack {
            package,
            output,
            generation,
            quiet,
        } => run_pack(&PackOpts {
            package,
            output,
            generation,
            quiet,
        }),
        Commands::Info { .. }
        | Commands::Query { .. }
        | Commands::Qhist(_)
        | Commands::Verify { .. }
        | Commands::Repair { .. } => unreachable!("write commands only"),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
