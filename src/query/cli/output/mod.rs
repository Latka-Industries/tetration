//! CLI presentation for [`QueryResponse`] (full JSON, compact JSON, stats, quiet).

mod format_num;
mod hints;
mod plan;
mod quiet;
mod stats;
mod table;

use crate::query::types::QueryResponse;

/// How `tet query` formats stdout (errors stay on stderr).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueryOutputFormat {
    /// Pretty-printed full [`QueryResponse`] (default, backward compatible).
    #[default]
    Full,
    /// Compact single-line JSON of the full response.
    Json,
    /// Slim JSON: plan summary + aggregates, no chunk rows or preview arrays.
    Stats,
    /// Slim JSON: catalog + `read_plan` only (no chunk rows, no execution block).
    Plan,
    /// One human-readable line on stdout.
    Quiet,
    /// Compact human tables: overview + result (+ preview). No plan/execution dump.
    Table,
    /// Verbose human tables: overview + read_plan + execution + result (short fields).
    TableVerbose,
}

impl std::str::FromStr for QueryOutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "full" => Ok(Self::Full),
            "json" => Ok(Self::Json),
            "stats" => Ok(Self::Stats),
            "plan" => Ok(Self::Plan),
            "quiet" => Ok(Self::Quiet),
            "table" => Ok(Self::Table),
            "table-verbose" | "table_verbose" | "tablev" => Ok(Self::TableVerbose),
            other => Err(format!(
                "unknown output format {other:?}; expected full, json, stats, plan, quiet, table, or table-verbose"
            )),
        }
    }
}

/// Format a query response for CLI stdout.
///
/// # Errors
///
/// Returns an error when [`QueryOutputFormat::Quiet`] cannot summarize the response (e.g. missing
/// aggregate fields after `--execute`).
pub fn format_query_response(
    response: &QueryResponse,
    format: QueryOutputFormat,
) -> Result<String, String> {
    match format {
        QueryOutputFormat::Full => {
            serde_json::to_string_pretty(response).map_err(|e| e.to_string())
        }
        QueryOutputFormat::Json => serde_json::to_string(response).map_err(|e| e.to_string()),
        QueryOutputFormat::Stats => stats::format_stats_json(response),
        QueryOutputFormat::Plan => plan::format_plan_json(response),
        QueryOutputFormat::Quiet => quiet::format_quiet_line(response),
        QueryOutputFormat::Table => table::format_table_text(response, table::TableDetail::Compact),
        QueryOutputFormat::TableVerbose => {
            table::format_table_text(response, table::TableDetail::Verbose)
        }
    }
}

/// Optional stderr text after a successful query (e.g. catalog miss).
#[must_use]
pub fn format_query_stderr_hints(response: &QueryResponse) -> Option<String> {
    hints::format_catalog_miss_hint(response)
}
