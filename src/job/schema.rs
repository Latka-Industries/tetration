//! Job file schema (`schema_version = 1`): ordered steps, one closed `op` tag per step.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Deserializer};

use crate::catalog::ScaffoldFormat;
use crate::query::cli::QueryOutputFormat;

/// Job schema version accepted by this crate.
pub const JOB_SCHEMA_VERSION: u32 = 1;

/// Parsed job: schema version plus ordered steps (run fail-fast).
#[derive(Debug, Clone, PartialEq)]
pub struct Job {
    pub schema_version: u32,
    pub steps: Vec<Step>,
}

/// One job step. The `op` string selects the variant; fields mirror `tet` CLI flags.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "op")]
pub enum Step {
    #[serde(rename = "info")]
    Info(InfoStep),
    #[serde(rename = "verify")]
    Verify(VerifyStep),
    #[serde(rename = "repair")]
    Repair(RepairStep),
    #[serde(rename = "query")]
    Query(QueryStep),
    #[serde(rename = "convert")]
    Convert(ConvertStep),
    #[serde(rename = "export")]
    Export(ExportStep),
    #[serde(rename = "index")]
    Index(IndexStep),
    #[serde(rename = "semantics.apply")]
    SemanticsApply(SemanticsApplyStep),
    #[serde(rename = "semantics.scaffold")]
    SemanticsScaffold(SemanticsScaffoldStep),
    #[serde(rename = "semantics.set")]
    SemanticsSet(SemanticsSetStep),
    #[serde(rename = "semantics.wizard")]
    SemanticsWizard(SemanticsWizardStep),
    #[serde(rename = "semantics.lint")]
    SemanticsLint(SemanticsLintStep),
    #[serde(rename = "semantics.package-write")]
    SemanticsPackageWrite(SemanticsPackageWriteStep),
    #[serde(rename = "ingest")]
    Ingest(IngestStep),
    #[serde(rename = "publish")]
    Publish(PublishStep),
    #[serde(rename = "compact")]
    Compact(CompactStep),
    #[serde(rename = "pack")]
    Pack(PackStep),
    #[serde(rename = "gc")]
    Gc(GcStep),
}

/// Every `op` string accepted in v1 (closed set).
pub const OPS: &[&str] = &[
    "info",
    "verify",
    "repair",
    "query",
    "convert",
    "export",
    "index",
    "semantics.apply",
    "semantics.scaffold",
    "semantics.set",
    "semantics.wizard",
    "semantics.lint",
    "semantics.package-write",
    "ingest",
    "publish",
    "compact",
    "pack",
    "gc",
];

/// `tet info <path>`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)] // mirrors `tet info` flags 1:1
pub struct InfoStep {
    pub path: PathBuf,
    #[serde(default)]
    pub json: bool,
    /// One-line summary. Omitted → quiet unless `json`; `false` restores the full text.
    #[serde(default)]
    pub quiet: Option<bool>,
    #[serde(default)]
    pub all: bool,
    #[serde(default)]
    pub layout: bool,
    #[serde(default)]
    pub execution: bool,
    #[serde(default)]
    pub datasets: bool,
    #[serde(default)]
    pub chunks: bool,
    /// `--history` (convert / provenance footer events).
    #[serde(default)]
    pub history: bool,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub dataset: Option<String>,
    #[serde(default)]
    pub grep: Option<String>,
    #[serde(default)]
    pub metadata: bool,
}

/// `tet verify <path>`. A failed verification fails the step.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(clippy::struct_excessive_bools)] // mirrors `tet verify` flags 1:1
pub struct VerifyStep {
    pub path: PathBuf,
    #[serde(default)]
    pub json: bool,
    /// One-line summary. Omitted → quiet unless `json`; `false` restores the full text.
    #[serde(default)]
    pub quiet: Option<bool>,
    #[serde(default)]
    pub repair: bool,
    #[serde(default)]
    pub deep: bool,
}

/// `tet repair <path>` (plan only unless `apply` is non-empty).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RepairStep {
    pub path: PathBuf,
    #[serde(default)]
    pub json: bool,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub apply: Vec<String>,
}

/// `tet query`: either `doc = "path"` or inline `query = { … }` (same schema as a query file).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryStep {
    /// `-t`: `.tet`, `.tet.d`, or `s3://` package.
    #[serde(default)]
    pub tet: Option<PathBuf>,
    /// Path to a `.json` / `.toml` query document.
    #[serde(default)]
    pub doc: Option<PathBuf>,
    /// Inline query document.
    #[serde(default)]
    pub query: Option<serde_json::Value>,
    /// `-x`: decode planned chunks (requires `tet`).
    #[serde(default)]
    pub execute: bool,
    /// `--format` (`full`, `json`, `stats`, `plan`, `quiet`, `table`, `table-verbose`).
    #[serde(default, deserialize_with = "de_output_format")]
    pub format: Option<QueryOutputFormat>,
    /// Shorthand for `format = "quiet"`. Omitted → quiet unless `format` is set; `false` restores
    /// the full text.
    #[serde(default)]
    pub quiet: Option<bool>,
    #[serde(default)]
    pub preview: Option<usize>,
    #[serde(default)]
    pub spill_allow: Vec<PathBuf>,
    #[serde(default)]
    pub device: Option<String>,
}

/// `tet convert <from> <to>`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConvertStep {
    /// Source file, Zarr directory, or `s3://` object / prefix.
    pub from: String,
    pub to: PathBuf,
    #[serde(default)]
    pub jobs: usize,
    #[serde(default)]
    pub strict: bool,
    #[serde(default)]
    pub chunk_mib: Option<u32>,
}

/// `tet export <from> <to>`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExportStep {
    pub from: PathBuf,
    pub to: PathBuf,
}

/// `tet index <path>`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexStep {
    pub path: PathBuf,
    #[serde(default)]
    pub clear: bool,
    #[serde(default)]
    pub quiet: bool,
}

/// `tet semantics apply <path> --from FILE`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticsApplyStep {
    pub path: PathBuf,
    pub from: PathBuf,
    #[serde(default)]
    pub annotate: bool,
    #[serde(default)]
    pub output: Option<PathBuf>,
}

/// `tet semantics scaffold <path>`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticsScaffoldStep {
    pub path: PathBuf,
    #[serde(default)]
    pub output: Option<PathBuf>,
    /// `toml` (default) or `json`.
    #[serde(default, deserialize_with = "de_scaffold_format")]
    pub format: Option<ScaffoldFormat>,
}

/// A scalar attr value (`set = { frame_rate = 30 }`); stringified like `--set KEY=VAL`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum AttrValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl AttrValue {
    /// The `VAL` half of `--set KEY=VAL`.
    #[must_use]
    pub fn to_attr_string(&self) -> String {
        match self {
            Self::Str(s) => s.clone(),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => f.to_string(),
            Self::Bool(b) => b.to_string(),
        }
    }
}

/// `tet semantics set <path> --dataset D --set K=V --clear K`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticsSetStep {
    pub path: PathBuf,
    pub dataset: String,
    #[serde(default)]
    pub set: BTreeMap<String, AttrValue>,
    #[serde(default)]
    pub clear: Vec<String>,
}

/// `tet semantics wizard <path>` (interactive; refused unless the runner allows it).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticsWizardStep {
    pub path: PathBuf,
    #[serde(default)]
    pub dataset: Option<String>,
    #[serde(default)]
    pub output: Option<PathBuf>,
    #[serde(default)]
    pub no_apply: bool,
}

/// `tet semantics lint <path>`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticsLintStep {
    pub path: PathBuf,
    #[serde(default)]
    pub generation: Option<u64>,
}

/// `tet semantics package-write <path>`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticsPackageWriteStep {
    pub path: PathBuf,
    #[serde(default)]
    pub generation: Option<u64>,
    #[serde(default)]
    pub no_manifest: bool,
}

/// `tet ingest <from…> --out <to>`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngestStep {
    /// One `.tet` path or a list of them.
    #[serde(deserialize_with = "de_one_or_many")]
    pub from: Vec<PathBuf>,
    /// Destination `.tet.d` package (created if missing).
    pub to: PathBuf,
    #[serde(default)]
    pub jobs: usize,
    #[serde(default)]
    pub create: bool,
    #[serde(default)]
    pub quiet: bool,
}

/// `tet publish <package>`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublishStep {
    pub package: PathBuf,
    #[serde(default)]
    pub generation: Option<u64>,
    #[serde(default)]
    pub parts: Option<Vec<String>>,
    #[serde(default)]
    pub abort: bool,
    #[serde(default)]
    pub dry_run: bool,
    #[serde(default)]
    pub replace: Vec<String>,
    #[serde(default)]
    pub quiet: bool,
}

/// `tet compact <package>` (alias of publish; no `abort` / `dry_run`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactStep {
    pub package: PathBuf,
    #[serde(default)]
    pub generation: Option<u64>,
    #[serde(default)]
    pub parts: Option<Vec<String>>,
    #[serde(default)]
    pub replace: Vec<String>,
    #[serde(default)]
    pub quiet: bool,
}

/// `tet pack <from> --out <to>`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackStep {
    /// `.tet.d` package.
    pub from: PathBuf,
    /// Destination standalone `.tet`.
    pub to: PathBuf,
    #[serde(default)]
    pub generation: Option<u64>,
    #[serde(default)]
    pub remove_package: bool,
    #[serde(default)]
    pub quiet: bool,
}

/// `tet gc <target>` (dry-run unless `apply`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GcStep {
    pub target: PathBuf,
    #[serde(default)]
    pub apply: bool,
    #[serde(default)]
    pub protect_generation: Vec<u64>,
    #[serde(default)]
    pub quiet: bool,
}

fn de_output_format<'de, D>(d: D) -> Result<Option<QueryOutputFormat>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = Option::<String>::deserialize(d)?;
    raw.map(|s| s.parse::<QueryOutputFormat>())
        .transpose()
        .map_err(serde::de::Error::custom)
}

fn de_scaffold_format<'de, D>(d: D) -> Result<Option<ScaffoldFormat>, D::Error>
where
    D: Deserializer<'de>,
{
    let raw = Option::<String>::deserialize(d)?;
    raw.map(|s| ScaffoldFormat::parse(&s))
        .transpose()
        .map_err(serde::de::Error::custom)
}

fn de_one_or_many<'de, D>(d: D) -> Result<Vec<PathBuf>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(PathBuf),
        Many(Vec<PathBuf>),
    }
    Ok(match OneOrMany::deserialize(d)? {
        OneOrMany::One(p) => vec![p],
        OneOrMany::Many(v) => v,
    })
}

/// Resolve a step's optional `quiet`. Only `info` / `verify` / `query` default to quiet
/// (`None` → `true`); `forced_verbose` is set when `json` / an explicit `format` already chose
/// the output shape, which always wins over quiet.
pub(super) fn effective_quiet(quiet: Option<bool>, forced_verbose: bool) -> bool {
    if forced_verbose {
        false
    } else {
        quiet.unwrap_or(true)
    }
}

impl Step {
    /// The `op` tag for this step.
    #[must_use]
    pub const fn op(&self) -> &'static str {
        match self {
            Self::Info(_) => "info",
            Self::Verify(_) => "verify",
            Self::Repair(_) => "repair",
            Self::Query(_) => "query",
            Self::Convert(_) => "convert",
            Self::Export(_) => "export",
            Self::Index(_) => "index",
            Self::SemanticsApply(_) => "semantics.apply",
            Self::SemanticsScaffold(_) => "semantics.scaffold",
            Self::SemanticsSet(_) => "semantics.set",
            Self::SemanticsWizard(_) => "semantics.wizard",
            Self::SemanticsLint(_) => "semantics.lint",
            Self::SemanticsPackageWrite(_) => "semantics.package-write",
            Self::Ingest(_) => "ingest",
            Self::Publish(_) => "publish",
            Self::Compact(_) => "compact",
            Self::Pack(_) => "pack",
            Self::Gc(_) => "gc",
        }
    }

    /// Static checks that need no filesystem (flag conflicts, required combinations).
    ///
    /// # Errors
    ///
    /// Returns a message naming the offending field(s).
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Info(s) if s.json && s.quiet == Some(true) => {
                Err("`json` and `quiet` conflict".into())
            }
            Self::Verify(s) if s.json && s.quiet == Some(true) => {
                Err("`json` and `quiet` conflict".into())
            }
            Self::Query(s) => s.validate(),
            Self::SemanticsApply(s) => {
                if s.output.is_some() && !s.annotate {
                    return Err("`output` requires `annotate = true`".into());
                }
                if s.output.as_ref().is_some_and(|o| *o == s.from) {
                    return Err("`output` must not be the `from` file".into());
                }
                Ok(())
            }
            Self::SemanticsSet(s) if s.set.is_empty() && s.clear.is_empty() => {
                Err("requires at least one `set` entry or `clear` key".into())
            }
            Self::SemanticsSet(s) => {
                if s.set
                    .iter()
                    .any(|(k, v)| k.is_empty() || v.to_attr_string().is_empty())
                {
                    return Err("`set` keys and values must be non-empty".into());
                }
                Ok(())
            }
            Self::Ingest(s) if s.from.is_empty() => {
                Err("`from` needs at least one .tet path".into())
            }
            Self::Publish(s) => {
                if s.abort && s.dry_run {
                    return Err("`abort` and `dry_run` cannot be combined".into());
                }
                if s.abort && s.generation.is_none() {
                    return Err("`abort` requires `generation`".into());
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl QueryStep {
    fn validate(&self) -> Result<(), String> {
        match (&self.doc, &self.query) {
            (Some(_), Some(_)) => {
                return Err("`doc` and `query` are mutually exclusive".into());
            }
            (None, None) => return Err("requires either `doc = \"path\"` or inline `query`".into()),
            _ => {}
        }
        if self.format.is_some() && self.quiet == Some(true) {
            return Err("`format` and `quiet` conflict".into());
        }
        if self.execute && self.tet.is_none() {
            return Err("`execute` requires `tet`".into());
        }
        if !self.execute {
            if self.preview.is_some() {
                return Err("`preview` requires `execute = true`".into());
            }
            if !self.spill_allow.is_empty() {
                return Err("`spill_allow` requires `execute = true`".into());
            }
            if self.device.is_some() {
                return Err("`device` requires `execute = true`".into());
            }
        }
        Ok(())
    }
}
