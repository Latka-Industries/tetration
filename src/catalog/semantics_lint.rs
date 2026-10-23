//! Optional semantics policy linter (0.7.3 / THI-621).
//!
//! Warns when a dataset claims a documented profile id but is missing recommended
//! attrs (or when history looks thin under `prov_log_v1`). Does **not** refuse open
//! or change fold answers. Unknown profile ids produce an informational finding only.

use std::collections::BTreeMap;
use std::path::Path;

use super::history::HistoryEvent;
use super::package_semantics::PackageSemanticsV1;
use super::semantics::{declared_attrs, is_declared_key};
use super::{CatalogError, TetFileSummaryV1, TetMetadataV1, read_tet_summary_v1};
use crate::layout::mmap_file_read;

/// Severity of a linter finding (advisory only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticsLintSeverity {
    /// Missing recommended attr / history field for a claimed profile.
    Warning,
    /// Unknown profile id or soft note.
    Info,
}

/// One advisory finding from [`lint_declared_semantics`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticsLintFinding {
    pub severity: SemanticsLintSeverity,
    /// Dataset catalog name, or `"*"` for file-wide notes.
    pub dataset: String,
    pub profile: Option<String>,
    pub code: String,
    pub message: String,
}

/// Lint report (always “ok” for open/query — findings are advisory).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SemanticsLintReport {
    pub findings: Vec<SemanticsLintFinding>,
}

impl SemanticsLintReport {
    /// `true` when no [`SemanticsLintSeverity::Warning`] findings.
    #[must_use]
    pub fn ok(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|f| f.severity == SemanticsLintSeverity::Warning)
    }
}

/// Normalize legacy / alias profile ids to registry ids.
#[must_use]
pub fn normalize_profile_id(id: &str) -> &str {
    match id {
        "climate_v1" => "cf_lite_v1",
        other => other,
    }
}

/// Recommended declared attr keys for a documented profile (empty → unknown).
#[must_use]
pub fn recommended_keys_for_profile(profile: &str) -> &'static [&'static str] {
    match normalize_profile_id(profile) {
        "media_v1" => &[
            "semantics_profile",
            "media_kind",
            "frame_rate",
            "frame_rate_unit",
            "time_axis",
        ],
        "timeseries_v1" => &[
            "semantics_profile",
            "media_kind",
            "sample_rate",
            "sample_rate_unit",
            "time_axis",
            "units",
        ],
        "cf_lite_v1" => &["semantics_profile", "units", "standard_name", "crs"],
        "prov_log_v1" => &["semantics_profile"],
        _ => &[],
    }
}

/// Recommended `history[].params` keys when claiming `prov_log_v1`.
pub const PROV_LOG_RECOMMENDED_PARAMS: &[&str] = &["agent", "activity"];

/// Lint footer metadata (+ optional history) for profile contract gaps.
#[must_use]
pub fn lint_declared_semantics(
    metadata: &TetMetadataV1,
    history: &[HistoryEvent],
) -> SemanticsLintReport {
    let mut findings = Vec::new();
    for (name, ds) in &metadata.datasets {
        let attrs = declared_attrs(&ds.attrs);
        let profile = attrs
            .get("semantics_profile")
            .cloned()
            .or_else(|| ds.attrs.get("semantics_profile").cloned());
        let Some(profile) = profile.filter(|p| !p.is_empty()) else {
            continue;
        };
        let canonical = normalize_profile_id(&profile).to_owned();
        if profile != canonical {
            findings.push(SemanticsLintFinding {
                severity: SemanticsLintSeverity::Info,
                dataset: name.clone(),
                profile: Some(profile.clone()),
                code: "profile_alias".into(),
                message: format!("legacy profile id {profile:?} → prefer {canonical:?}"),
            });
        }
        let recommended = recommended_keys_for_profile(&canonical);
        if recommended.is_empty() {
            findings.push(SemanticsLintFinding {
                severity: SemanticsLintSeverity::Info,
                dataset: name.clone(),
                profile: Some(canonical.clone()),
                code: "unknown_profile".into(),
                message: format!(
                    "unknown semantics_profile {canonical:?} (open minting; no registry keys)"
                ),
            });
            continue;
        }
        for key in recommended {
            if *key == "semantics_profile" {
                continue;
            }
            let present = attrs.contains_key(*key) || ds.attrs.contains_key(*key);
            if !present {
                findings.push(SemanticsLintFinding {
                    severity: SemanticsLintSeverity::Warning,
                    dataset: name.clone(),
                    profile: Some(canonical.clone()),
                    code: "missing_attr".into(),
                    message: format!("claims {canonical} but missing recommended attr {key:?}"),
                });
            }
        }
        if canonical == "prov_log_v1" {
            lint_prov_log_history(name, history, &mut findings);
        }
    }
    SemanticsLintReport { findings }
}

fn lint_prov_log_history(
    dataset: &str,
    history: &[HistoryEvent],
    findings: &mut Vec<SemanticsLintFinding>,
) {
    if history.is_empty() {
        findings.push(SemanticsLintFinding {
            severity: SemanticsLintSeverity::Warning,
            dataset: dataset.to_owned(),
            profile: Some("prov_log_v1".into()),
            code: "empty_history".into(),
            message: "claims prov_log_v1 but history is empty".into(),
        });
        return;
    }
    let mut any_agent = false;
    let mut any_activity = false;
    for ev in history {
        if ev.agent_ref().is_some() {
            any_agent = true;
        }
        if ev.activity_ref().is_some() {
            any_activity = true;
        }
    }
    if !any_agent {
        findings.push(SemanticsLintFinding {
            severity: SemanticsLintSeverity::Warning,
            dataset: dataset.to_owned(),
            profile: Some("prov_log_v1".into()),
            code: "missing_prov_param".into(),
            message: "claims prov_log_v1 but no history row has agent / params.agent".into(),
        });
    }
    if !any_activity {
        findings.push(SemanticsLintFinding {
            severity: SemanticsLintSeverity::Warning,
            dataset: dataset.to_owned(),
            profile: Some("prov_log_v1".into()),
            code: "missing_prov_param".into(),
            message: "claims prov_log_v1 but no history row has activity / params.activity".into(),
        });
    }
}

/// Lint a file summary (attrs + history).
#[must_use]
pub fn lint_summary(summary: &TetFileSummaryV1) -> SemanticsLintReport {
    lint_declared_semantics(&summary.metadata, &summary.history)
}

/// Lint a `.tet` path.
///
/// # Errors
///
/// Catalog / I/O failures.
pub fn lint_tet_path(path: &Path) -> Result<SemanticsLintReport, CatalogError> {
    let mmap = mmap_file_read(path)?;
    let summary = read_tet_summary_v1(&mmap)?;
    Ok(lint_summary(&summary))
}

/// Lint a package `SEMANTICS.json` document (declare-only overlay).
#[must_use]
pub fn lint_package_semantics(doc: &PackageSemanticsV1) -> SemanticsLintReport {
    let mut meta = TetMetadataV1::default();
    for (name, decl) in &doc.datasets {
        let mut attrs = BTreeMap::new();
        for (k, v) in &decl.attrs {
            if is_declared_key(k) {
                attrs.insert(k.clone(), v.clone());
            }
        }
        if let Some(p) = &decl.semantics_profile {
            attrs.insert("semantics_profile".into(), p.clone());
        }
        meta.datasets.insert(
            name.clone(),
            super::metadata::DatasetMetadataV1 {
                attrs,
                ..Default::default()
            },
        );
    }
    lint_declared_semantics(&meta, &[])
}

/// Format a report for CLI / logs.
#[must_use]
pub fn format_lint_report(report: &SemanticsLintReport) -> String {
    use std::fmt::Write as _;
    if report.findings.is_empty() {
        return "semantics lint: ok (no findings)\n".to_owned();
    }
    let mut out = String::new();
    for f in &report.findings {
        let sev = match f.severity {
            SemanticsLintSeverity::Warning => "warning",
            SemanticsLintSeverity::Info => "info",
        };
        let profile = f.profile.as_deref().unwrap_or("-");
        let _ = writeln!(
            out,
            "{sev}: [{}] dataset={} profile={profile}: {}",
            f.code, f.dataset, f.message
        );
    }
    if report.ok() {
        out.push_str("semantics lint: ok (info only)\n");
    } else {
        out.push_str("semantics lint: warnings present (advisory; open/query still valid)\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::metadata::DatasetMetadataV1;

    #[test]
    fn warns_when_cf_lite_missing_units() {
        let mut meta = TetMetadataV1::default();
        meta.datasets.insert(
            "tas".into(),
            DatasetMetadataV1 {
                attrs: BTreeMap::from([("semantics_profile".into(), "cf_lite_v1".into())]),
                ..Default::default()
            },
        );
        let report = lint_declared_semantics(&meta, &[]);
        assert!(!report.ok());
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == "missing_attr" && f.message.contains("units"))
        );
    }

    #[test]
    fn climate_v1_alias_info() {
        let mut meta = TetMetadataV1::default();
        meta.datasets.insert(
            "tas".into(),
            DatasetMetadataV1 {
                attrs: BTreeMap::from([
                    ("semantics_profile".into(), "climate_v1".into()),
                    ("units".into(), "K".into()),
                    ("standard_name".into(), "air_temperature".into()),
                    ("crs".into(), "EPSG:4326".into()),
                ]),
                ..Default::default()
            },
        );
        let report = lint_declared_semantics(&meta, &[]);
        assert!(report.ok());
        assert!(report.findings.iter().any(|f| f.code == "profile_alias"));
    }
}
