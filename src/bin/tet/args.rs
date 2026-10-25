//! Clap argument definitions for `tet`.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};
use tetration::catalog::ScaffoldFormat;
use tetration::query::cli::{HistorySettings, QueryOutputFormat};

#[derive(Parser)]
#[command(
    name = "tet",
    version,
    about = "Tetration CLI: JSON queries and format conversion"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Summarize a `.tet` file or `.tet.d` package (default: dataset table).
    /// Use `--json` for full dump; `--metadata` for coordinate label previews.
    Info {
        /// Path to `.tet` file or `.tet.d` package (package uses CURRENT `packed.tet`).
        path: PathBuf,
        /// Full pretty JSON (superblock, catalog, chunks, history).
        #[arg(long)]
        json: bool,
        /// One-line summary on stdout.
        #[arg(short = 'q', long, conflicts_with = "json")]
        quiet: bool,
        /// All text sections (layout, execution, datasets, chunks, history).
        #[arg(long)]
        all: bool,
        /// Superblock / layout fields.
        #[arg(long)]
        layout: bool,
        /// Per-file execution defaults from the chunk index header.
        #[arg(long)]
        execution: bool,
        /// Dataset catalog table (default when no section flags).
        #[arg(long)]
        datasets: bool,
        /// Chunk index table (`-n` limits rows; `0` = all).
        #[arg(long)]
        chunks: bool,
        /// Convert / provenance footer events (`tet info --history`; not `tet qhist`).
        #[arg(long = "history")]
        show_footer_history: bool,
        /// Max chunk rows when `--chunks` or `--all` (default 32; `0` = all).
        #[arg(short = 'n', long, value_name = "N")]
        limit: Option<usize>,
        /// Case-insensitive substring on dataset name.
        #[arg(long)]
        dataset: Option<String>,
        /// Case-insensitive substring on dataset name, dtype, or footer metadata.
        #[arg(long)]
        grep: Option<String>,
        /// Verbose footer metadata under dataset rows (coordinate label previews).
        #[arg(long)]
        metadata: bool,
    },
    /// Run a JSON query (plan, execute, or both).
    ///
    /// Typical flows:
    ///   tet query q.json -t data.tet
    ///   tet query q.json -t data.tet -x -q
    ///   tet query '{"dataset":"f32","mean":[]}' -t data.tet -x
    #[command(
        visible_alias = "q",
        after_help = "QUERY: path to .json / .toml, inline JSON or TOML, or `-` for stdin; omit QUERY to read stdin. \
                      Leading `{` selects JSON; otherwise TOML. Extension `.json` / `.toml` overrides. \
                      -x decodes chunks (requires -t). -q is --format quiet; else default full. \
                      A package query pins the generation it opened and does not follow a newer publish \
                      (docs/query_engine.md#pin--consistency). \
                      -t also accepts s3://bucket/prefix (feature tetration-s3; TETRATION_S3_ENDPOINT)."
    )]
    Query {
        /// Query document: `.json` / `.toml` path, inline JSON/TOML, or `-` for stdin.
        #[arg(value_name = "QUERY")]
        query: Option<String>,
        /// `.tet` file, `.tet.d` package, or `s3://bucket/prefix` package (`?generation=N` pins that generation).
        /// An `s3://` package needs `--features tetration-s3` and `TETRATION_S3_ENDPOINT`.
        /// A package pins the generation it opened and does not follow a newer publish.
        #[arg(short = 't', long, value_name = "PATH")]
        tet: Option<PathBuf>,
        /// Decode planned chunks and attach `execution` (requires `-t`).
        #[arg(short = 'x', long, requires = "tet")]
        execute: bool,
        /// stdout: full, json, stats, plan, quiet, table, table-verbose. Default: full.
        #[arg(long, value_enum, default_value_t = QueryStdoutFormat::Full, conflicts_with = "quiet")]
        format: QueryStdoutFormat,
        /// Shorthand for `--format quiet` (one-line stdout).
        #[arg(short = 'q', long, conflicts_with = "format")]
        quiet: bool,
        /// Sample values in JSON when executing (all dtypes). Default: 64 (full/json), 0 (quiet/stats).
        #[arg(long, visible_alias = "preview-f32", value_name = "N")]
        preview: Option<usize>,
        /// Extra spill directory roots (repeatable; needs `-x` and `-t`).
        #[arg(long, requires = "execute", requires = "tet")]
        spill_allow: Vec<PathBuf>,
        /// Device for tier-A/B reductions: `cpu`, `auto`, `metal`, `cuda`, `cuda:N` (overrides query `execution.device`).
        #[arg(long, value_name = "DEVICE", requires = "execute")]
        device: Option<String>,
    },
    /// Verify a `.tet` file or `.tet.d` package (layout / protocol health).
    Verify {
        /// Path to `.tet` file or `.tet.d` package directory.
        path: PathBuf,
        /// Pretty JSON report (automation / CI).
        #[arg(long)]
        json: bool,
        /// One-line summary on stdout.
        #[arg(short = 'q', long, conflicts_with = "json")]
        quiet: bool,
        /// After verify, apply safe in-place repairs (see `tet repair`).
        #[arg(long)]
        repair: bool,
        /// Full verify: decode every chunk payload (`tet verify` alone is a quick scan of the first 128 chunks on large files).
        #[arg(long)]
        deep: bool,
    },
    /// Plan or apply in-place repairs (default: plan from verify recommendations).
    Repair {
        /// Path to `.tet` file.
        path: PathBuf,
        /// Pretty JSON (plan or repair report).
        #[arg(long)]
        json: bool,
        /// Recommendation codes to apply (repeatable).
        #[arg(long = "apply", value_name = "CODE")]
        apply: Vec<String>,
        /// Show what would change without writing the file (with `--apply`).
        #[arg(long)]
        dry_run: bool,
    },
    /// Export a `.tet` file to Zarr v3, HDF5, or `NetCDF` (format from output path).
    Export {
        /// Source `.tet` file.
        input: PathBuf,
        /// Destination: Zarr directory (empty/missing), `.h5`/`.hdf5`, or `.nc`/`.netcdf`.
        output: PathBuf,
    },
    /// Convert HDF5, `NetCDF`, `Zarr`, `NumPy`, CSV, or Parquet into `.tet`.
    ///
    /// A single-file `s3://bucket/key` downloads the whole object, then uses the same importer.
    /// A Zarr prefix (`s3://bucket/store` or `s3://bucket/store.zarr`) fetches each chunk as it
    /// is written (`tetration-s3`, `TETRATION_S3_ENDPOINT`).
    Convert {
        /// Source array file, Zarr directory, `s3://bucket/object.ext`, or `s3://bucket/zarr-prefix`.
        input: String,
        /// Destination `.tet` file.
        output: PathBuf,
        /// Parallel chunk read workers (`0` = host `available_parallelism`, capped at 64).
        #[arg(long = "jobs", default_value_t = 0)]
        jobs: usize,
        /// Fail when any source array is skipped (unsupported dtype, empty, or scalar).
        #[arg(long)]
        strict: bool,
        /// Retile every dataset to about N MiB along the leading axis.
        ///
        /// Omit to keep source chunking, or the 64 MiB default for `NumPy`, CSV, and Parquet.
        #[arg(long, value_name = "N")]
        chunk_mib: Option<u32>,
    },
    /// Build or refresh embedded ZONE/INDEX spills on a sealed `.tet` (`history_version = 2`).
    ///
    /// Convert and writer commit already embed automatically (0.7.0). Use this to re-index an
    /// older bare file, or `--clear` to drop embeds for a no-pushdown baseline.
    Index {
        /// Sealed `.tet` file.
        path: PathBuf,
        /// Drop embedded ZONE/INDEX (rewrite `history_version = 1` footer; keep history/metadata).
        #[arg(long)]
        clear: bool,
        #[arg(short = 'q', long)]
        quiet: bool,
    },
    /// Ingest one or more `.tet` files into a `.tet.d` package as sealed staging parts.
    Ingest {
        /// Source `.tet` files (each becomes one staging part).
        #[arg(required = true, num_args = 1..)]
        inputs: Vec<PathBuf>,
        /// Destination package directory (created if missing).
        #[arg(long = "out", short = 'o', value_name = "PKG.tet.d")]
        out: PathBuf,
        /// Parallel ingest workers (`0` = host `available_parallelism`, capped at 64).
        #[arg(long = "jobs", default_value_t = 0)]
        jobs: usize,
        /// No-op: a missing package is always created; an existing non-package path always fails.
        #[arg(long)]
        create: bool,
        /// Suppress progress lines.
        #[arg(short = 'q', long)]
        quiet: bool,
    },
    /// Publish sealed staging parts into a new generation (`CURRENT` flip).
    ///
    /// Compact/merge of parts happens during this step (see also `tet compact`).
    /// Pass `--abort --generation G` to drop a stuck `publishing/<G>/` lock.
    /// Pass `--dry-run` for claim/schema checks only (no seal, no `CURRENT` flip).
    Publish {
        /// Package directory (`.tet.d`).
        package: PathBuf,
        /// Generation id (default: CURRENT+1, or 1 when empty). Required with `--abort`.
        #[arg(long, value_name = "G")]
        generation: Option<u64>,
        /// Explicit part ids (comma-separated). Default: all sealed staging parts.
        #[arg(long, value_delimiter = ',', value_name = "ID")]
        parts: Option<Vec<String>>,
        /// Drop `publishing/<G>/` for a stuck publish (requires `--generation`; refuses CURRENT).
        #[arg(long)]
        abort: bool,
        /// Claim / schema checks only; do not seal or flip `CURRENT`.
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// Allow dtype/shape change for this dataset name vs sealed CURRENT (repeatable; G+1 only).
        #[arg(long = "replace", value_name = "DATASET")]
        replace: Vec<String>,
        #[arg(short = 'q', long)]
        quiet: bool,
    },
    /// Alias for `tet publish` — compact merges parts during publish (decision I).
    Compact {
        /// Package directory (`.tet.d`).
        package: PathBuf,
        #[arg(long, value_name = "G")]
        generation: Option<u64>,
        #[arg(long, value_delimiter = ',', value_name = "ID")]
        parts: Option<Vec<String>>,
        #[arg(long = "replace", value_name = "DATASET")]
        replace: Vec<String>,
        #[arg(short = 'q', long)]
        quiet: bool,
    },
    /// Copy a sealed generation's `packed.tet` to a standalone `.tet`.
    Pack {
        /// Package directory (`.tet.d`).
        package: PathBuf,
        /// Destination `.tet` path.
        #[arg(long = "out", short = 'o', value_name = "OUT.tet")]
        output: PathBuf,
        /// Generation to pack (default: CURRENT).
        #[arg(long, value_name = "G")]
        generation: Option<u64>,
        /// After a verified copy, delete the `.tet.d` package (opt-in; close pinned readers first).
        #[arg(long = "remove-package")]
        remove_package: bool,
        #[arg(short = 'q', long)]
        quiet: bool,
    },
    /// Drop leftover staging and sealed generations older than CURRENT / CURRENT-1.
    ///
    /// A local `.tet.d` uses package GC. An `s3://bucket/prefix` target (feature
    /// `tetration-s3`, credentials `TETRATION_S3_*`) uses the same retention on
    /// `PackageStore`. Dry-run by default; pass `--apply` to delete.
    ///
    /// `s3://` gc cannot see open pins in other processes. Pass `--protect-generation G`
    /// for each (or `?generation=G` on the URI). CURRENT, CURRENT-1, and a live
    /// `publishing/<G>` claim are kept either way. Prune does not change CURRENT.
    Gc {
        /// Local `.tet.d` package, or `s3://bucket/prefix` (`?generation=G` protects G).
        target: PathBuf,
        /// Actually delete (default: print the plan only).
        #[arg(long)]
        apply: bool,
        /// Generation an open reader still needs. Repeatable. `s3://` targets only.
        #[arg(long = "protect-generation", value_name = "G")]
        protect_generation: Vec<u64>,
        #[arg(short = 'q', long)]
        quiet: bool,
    },
    /// Recent `tet query` log (platform cache; not in `.tet` footer). Default: `list`.
    Qhist(QhistArgs),
    /// Declared science/media attrs and caller-policy stubs (not a plugin host).
    Semantics(SemanticsArgs),
    /// Run a job file: ordered `tet` steps from TOML or JSON, fail-fast.
    ///
    /// Not a scheduler or DAG (no conditionals, loops, or retries) and not query-pipeline ETL.
    /// Relative paths resolve against `--root` (default: the job file's directory).
    /// See docs/job.md.
    Run {
        /// Job file (`.toml` or `.json`; `schema_version = 1`, `[[step]]` / `"steps"`).
        #[arg(value_name = "JOB")]
        job: PathBuf,
        /// Validate the job and print the plan; execute nothing.
        #[arg(long = "dry-run")]
        dry_run: bool,
        /// Directory that relative paths in the job resolve against.
        #[arg(long, value_name = "DIR")]
        root: Option<PathBuf>,
        /// Print the full `JobReport` as pretty JSON on stdout (progress stays on stderr).
        ///
        /// Each `steps[].data` carries the step's structured result. On a step failure the
        /// exit code is 1 and stdout holds the steps that completed plus an `error` object.
        #[arg(long)]
        json: bool,
    },
    /// Run one distributed worker job (`WorkerJobV1` JSON → `WorkerPartialV1` JSON).
    ///
    /// Localhost harness for the thin distributed protocol (0.7.1). Reads job JSON from
    /// `--job PATH` or stdin; writes the partial to stdout. Not a cluster scheduler.
    Worker {
        /// Path to `WorkerJobV1` JSON (`-` or omit = stdin).
        #[arg(long, value_name = "PATH")]
        job: Option<PathBuf>,
    },
}

/// `tet semantics` — declare footer attrs / scaffold `tet-semantics.toml|json`.
#[derive(Args)]
#[command(
    name = "semantics",
    about = "Declare science/media attrs, apply tet-semantics.toml, or scaffold caller policy"
)]
pub struct SemanticsArgs {
    #[command(subcommand)]
    pub cmd: SemanticsCmd,
}

#[derive(Subcommand)]
pub enum SemanticsCmd {
    /// Stamp `dim_names` and attrs from a one-file `tet-semantics.toml`.
    ///
    /// Policy formulas and `[units.*]` stay in the TOML. Does not rewrite `--from`.
    /// Sealed `.tet.d` generations are immutable.
    Apply {
        /// Path to a mutable `.tet` file.
        path: PathBuf,
        /// One-file semantics TOML (`schema_version`, datasets, profiles, units).
        #[arg(long, value_name = "FILE")]
        from: PathBuf,
        /// Print (or `-o`) an annotated TOML with live values and missing recommended keys
        /// as comments. Does not rewrite `--from`.
        #[arg(long)]
        annotate: bool,
        /// Write the annotated TOML here. Requires `--annotate`. Must not be `--from`.
        #[arg(short = 'o', long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
    /// Emit caller-policy stubs from declared attrs (or catalog + doc links when none).
    Scaffold {
        /// Path to `.tet` file.
        path: PathBuf,
        /// Write to this path instead of stdout.
        #[arg(short = 'o', long = "output", value_name = "FILE")]
        output: Option<PathBuf>,
        /// `toml` (default) or `json`.
        #[arg(long, value_enum, default_value_t = ScaffoldFormat::Toml)]
        format: ScaffoldFormat,
    },
    /// Declare or clear dataset attrs in the history footer (does not reinterpret values).
    Set {
        /// Path to `.tet` file.
        path: PathBuf,
        /// Dataset catalog name.
        #[arg(long)]
        dataset: String,
        /// Set `KEY=VAL` (repeatable).
        #[arg(long = "set", value_name = "KEY=VAL")]
        set: Vec<String>,
        /// Remove attr `KEY` (repeatable).
        #[arg(long = "clear", value_name = "KEY")]
        clear: Vec<String>,
    },
    /// Write declare-only `SEMANTICS.json` for a sealed package generation (from packed footer).
    PackageWrite {
        /// Path to `.tet.d` package root.
        path: PathBuf,
        /// Generation id (default: CURRENT).
        #[arg(long)]
        generation: Option<u64>,
        /// Skip stamping MANIFEST `semantics` / `semantics_sha256`.
        #[arg(long)]
        no_manifest: bool,
    },
    /// Print `SEMANTICS.json` for a sealed package generation (if present).
    PackageShow {
        /// Path to `.tet.d` package root.
        path: PathBuf,
        /// Generation id (default: CURRENT).
        #[arg(long)]
        generation: Option<u64>,
    },
    /// Advisory lint: claimed `semantics_profile` vs recommended attrs (does not refuse open).
    Lint {
        /// Path to `.tet` file, or `.tet.d` package (lints CURRENT `SEMANTICS.json` when present,
        /// else packed footer).
        path: PathBuf,
        /// Package generation id (default: CURRENT). Only used for `.tet.d`.
        #[arg(long)]
        generation: Option<u64>,
    },
    /// Interactive prompts: name dims, pick profile / attrs, confirm, write TOML + apply.
    ///
    /// Empty keeps the last selection (shown in brackets); `-` clears/skips.
    /// `n` on confirm redoes the prompts with that last selection. Sealed `.tet.d` is refused.
    Wizard {
        /// Path to a mutable `.tet` file.
        path: PathBuf,
        /// Dataset catalog name (default: sole dataset, or prompt).
        #[arg(long)]
        dataset: Option<String>,
        /// Write the combined TOML here (default: `tet-semantics.toml` beside the `.tet`).
        #[arg(short = 'o', long, value_name = "FILE")]
        output: Option<PathBuf>,
        /// Write TOML only; do not stamp the `.tet`.
        #[arg(long)]
        no_apply: bool,
    },
}

/// `tet qhist` — platform query history (not `tet info --history` footer).
#[derive(Args)]
#[command(
    name = "qhist",
    visible_alias = "hist",
    about = "Recent tet query log (platform cache; not in .tet footer). Default: list"
)]
pub struct QhistArgs {
    #[command(subcommand)]
    pub cmd: Option<QhistCmd>,
    /// Remove the query history file (`list --clear` or bare `tet qhist --clear`).
    #[arg(long, global = true)]
    pub clear: bool,
}

#[derive(Subcommand)]
pub enum QhistCmd {
    /// List recent queries (default). Use `--all` for every retained row.
    List {
        /// Max rows to print (ignored when `--all` is set).
        #[arg(short = 'n', long, default_value_t = HistorySettings::default().cli_query_max)]
        limit: usize,
        /// Print all retained rows (up to `TET_QUERY_HISTORY_MAX` on disk).
        #[arg(long)]
        all: bool,
        /// Pretty JSON (full entries); default is a compact table.
        #[arg(long)]
        json: bool,
        /// Case-insensitive substring on dataset name.
        #[arg(long)]
        dataset: Option<String>,
        /// Case-insensitive substring on saved `.tet` path.
        #[arg(long)]
        tet: Option<String>,
        /// `x` / `execute` or `p` / `plan` (matches list `mode` column).
        #[arg(long, value_name = "x|p")]
        mode: Option<String>,
        /// Search dataset, tet path, and operation label.
        #[arg(long)]
        grep: Option<String>,
    },
    /// Re-run a saved query (`N`: 1 = newest; indices match filtered `list`).
    Run {
        /// History index (1 = newest).
        #[arg(value_name = "N")]
        index: usize,
        /// Override `.tet` from the saved row.
        #[arg(short = 't', long, value_name = "PATH")]
        tet: Option<PathBuf>,
        /// Force execute (needs `-t` on the row or this flag).
        #[arg(short = 'x', long, conflicts_with = "plan")]
        execute: bool,
        /// Plan only (ignore saved execute).
        #[arg(long, conflicts_with = "execute")]
        plan: bool,
        #[arg(long, value_enum, default_value_t = QueryStdoutFormat::Full, conflicts_with = "quiet")]
        format: QueryStdoutFormat,
        #[arg(short = 'q', long, conflicts_with = "format")]
        quiet: bool,
        #[arg(long, visible_alias = "preview-f32", value_name = "N")]
        preview: Option<usize>,
        #[arg(long)]
        spill_allow: Vec<PathBuf>,
        /// Same filters as `qhist list` (with positional `N`).
        #[arg(long)]
        dataset: Option<String>,
        #[arg(long, value_name = "PATH")]
        tet_filter: Option<String>,
        #[arg(long, value_name = "x|p")]
        mode: Option<String>,
        #[arg(long)]
        grep: Option<String>,
    },
}

/// How `tet query` prints success on stdout (errors always go to stderr).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum QueryStdoutFormat {
    /// Pretty JSON of the full `QueryResponse` (default).
    #[default]
    Full,
    /// Compact one-line JSON (scripts, `jq`).
    Json,
    /// Slim JSON: plan summary + aggregates, no chunk list or preview arrays.
    Stats,
    /// Slim JSON: catalog + `read_plan` summary only (no chunks, no execution).
    Plan,
    /// One human-readable line (`dataset=… op=… mean=…`).
    Quiet,
    /// Compact tables: overview + result (+ preview). Receipt-style.
    Table,
    /// Verbose tables: overview + `read_plan` + execution + result (short fields, no wide rows).
    #[value(name = "table-verbose", alias = "tablev")]
    TableVerbose,
}

impl From<QueryStdoutFormat> for QueryOutputFormat {
    fn from(f: QueryStdoutFormat) -> Self {
        match f {
            QueryStdoutFormat::Full => Self::Full,
            QueryStdoutFormat::Json => Self::Json,
            QueryStdoutFormat::Stats => Self::Stats,
            QueryStdoutFormat::Plan => Self::Plan,
            QueryStdoutFormat::Quiet => Self::Quiet,
            QueryStdoutFormat::Table => Self::Table,
            QueryStdoutFormat::TableVerbose => Self::TableVerbose,
        }
    }
}
