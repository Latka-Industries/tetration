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
        } => {
            let stdout = resolve_stdout(quiet, format);
            let (raw, path_hint) = read_query_payload(query.as_deref()).map_err(cli_error)?;
            let trimmed = raw.trim();
            let format = query_input_format(path_hint.as_deref(), trimmed);
            let doc = parse_query_text(trimmed, format).map_err(cli_error)?;
            run_query(QueryRunOpts {
                doc,
                tet,
                execute,
                stdout,
                preview,
                spill_allow,
                device,
                record_history: true,
            })
        }
        Commands::Qhist(args) => run_qhist(args.cmd, args.clear),
        Commands::Verify {
            path,
            json,
            quiet,
            repair,
            deep,
        } => {
            let opts = VerifyRunOpts {
                path,
                json,
                quiet,
                repair,
                deep,
            };
            run_verify(&opts)
        }
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
        Commands::Export { input, output } => run_export(&input, &output),
        Commands::Convert {
            input,
            output,
            jobs,
        } => run_convert(&input, &output, jobs),
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
