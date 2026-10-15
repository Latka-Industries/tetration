//! In-place `.tet` repairs (invoked by `tet repair` or `tet verify --repair`).
//!
//! **Design:** [`crate::verify`] is read-only by default. It attaches suggested
//! `tet repair <path> …` commands to recommendations. Mutations happen only here.
//!
//! ```text
//! tet verify data.tet              → report + repair command hints
//! tet repair data.tet --dry-run    → plan (default)
//! tet repair data.tet --apply footer_invalid
//! tet verify data.tet --repair     → verify, then apply safe repairable fixes
//! ```

mod actions;
mod format;
mod plan;

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::catalog::CatalogError;
use crate::verify::TetVerifyReport;

pub use format::{format_plan_json, format_plan_text, format_repair_json, format_repair_text};
pub use plan::{RepairAction, RepairPlan, repair_plan_from_verify};

/// Options for [`repair_tet_file`].
#[derive(Debug, Clone, Default)]
pub struct RepairOptions {
    /// When true, report planned actions without writing the file.
    pub dry_run: bool,
    /// Recommendation / repair codes to apply (e.g. `footer_invalid`). Empty = plan only.
    pub apply: Vec<String>,
    /// When set with `apply` empty, only consider these codes from the verify report.
    pub plan_codes: Vec<String>,
}

/// Outcome of one repair action.
#[derive(Debug, Clone, Serialize)]
pub struct RepairActionResult {
    pub code: String,
    pub applied: bool,
    pub dry_run: bool,
    pub message: String,
}

/// Full repair run result.
#[derive(Debug, Clone, Serialize)]
pub struct TetRepairReport {
    pub path: PathBuf,
    pub dry_run: bool,
    pub actions: Vec<RepairActionResult>,
    pub verify_after_ok: Option<bool>,
}

/// Build a `tet repair …` command line for scripts (default includes `--dry-run`).
#[must_use]
pub fn repair_command_line(path: &Path, code: &str, dry_run: bool) -> String {
    let p = path.display();
    if dry_run {
        format!("tet repair {p} --apply {code} --dry-run")
    } else {
        format!("tet repair {p} --apply {code}")
    }
}

/// Whether this recommendation code has an in-place repair implementation.
#[must_use]
pub fn is_repairable_code(code: &str) -> bool {
    matches!(code, "footer_invalid" | "quarantine_publish")
}

/// Suggested CLI command for a recommendation code, if repairable.
#[must_use]
pub fn repair_command_for_code(path: &Path, code: &str) -> Option<String> {
    if is_repairable_code(code) {
        Some(repair_command_line(path, code, true))
    } else {
        None
    }
}

/// Attach `tet repair <path> …` to verify recommendations when a fix exists.
pub fn enrich_verify_recommendations(path: &Path, report: &mut TetVerifyReport) {
    for rec in &mut report.recommendations {
        let Some(cmd) = repair_command_for_code(path, &rec.code) else {
            continue;
        };
        match &mut rec.fix {
            Some(fix) => fix.command = Some(cmd),
            None => {
                rec.fix = Some(crate::verify::VerifyFixHint {
                    summary: "Run repair (dry-run first).".to_owned(),
                    command: Some(cmd),
                });
            }
        }
    }
}

/// Plan repairs from a verify report (no I/O beyond what verify already did).
#[must_use]
pub fn repair_plan(path: &Path, verify: &TetVerifyReport) -> RepairPlan {
    repair_plan_from_verify(path, verify)
}

/// Run repairs on a `.tet` file (re-verify at the end when not dry-run).
///
/// # Errors
///
/// I/O or catalog errors from repair actions.
pub fn repair_tet_file(
    path: &Path,
    options: &RepairOptions,
) -> Result<TetRepairReport, CatalogError> {
    let verify = verify_path(path)?;
    let plan = repair_plan_from_verify(path, &verify);
    let mut actions = Vec::new();

    let codes: Vec<String> = if options.apply.is_empty() {
        if options.plan_codes.is_empty() {
            plan.actions
                .iter()
                .filter(|a| a.repairable)
                .map(|a| a.code.clone())
                .collect()
        } else {
            options.plan_codes.clone()
        }
    } else {
        options.apply.clone()
    };

    for code in codes {
        let action = plan
            .actions
            .iter()
            .find(|a| a.code == code)
            .cloned()
            .unwrap_or(RepairAction {
                code: code.clone(),
                repairable: is_repairable_code(&code),
                summary: "unknown repair code".to_owned(),
            });
        let result = if action.repairable {
            actions::apply_repair_code(path, &code, options.dry_run)?
        } else {
            RepairActionResult {
                code: code.clone(),
                applied: false,
                dry_run: options.dry_run,
                message: "no in-place repair for this code; re-convert or rewrite required"
                    .to_owned(),
            }
        };
        actions.push(result);
    }

    let verify_after_ok = if options.dry_run {
        None
    } else {
        Some(verify_path(path)?.ok)
    };

    Ok(TetRepairReport {
        path: path.to_path_buf(),
        dry_run: options.dry_run,
        actions,
        verify_after_ok,
    })
}

fn verify_path(path: &Path) -> Result<TetVerifyReport, CatalogError> {
    if crate::verify::looks_like_tet_package(path) {
        crate::verify::verify_tet_package(path)
    } else {
        crate::verify::verify_tet_file(path)
    }
}

/// Apply repairable fixes from a verify report (used by `tet verify --repair`).
///
/// # Errors
///
/// I/O or catalog errors from repair actions.
pub fn repair_from_verify_report(
    path: &Path,
    verify: &TetVerifyReport,
    dry_run: bool,
) -> Result<TetRepairReport, CatalogError> {
    let plan = repair_plan_from_verify(path, verify);
    let mut actions = Vec::new();
    for action in plan.actions.iter().filter(|a| a.repairable) {
        actions.push(actions::apply_repair_code(path, &action.code, dry_run)?);
    }
    let verify_after_ok = if dry_run {
        None
    } else {
        Some(verify_path(path)?.ok)
    };
    Ok(TetRepairReport {
        path: path.to_path_buf(),
        dry_run,
        actions,
        verify_after_ok,
    })
}
