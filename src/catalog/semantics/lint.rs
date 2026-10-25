//! Optional semantics policy linter (0.7.3 / THI-621).
//!
//! Warns when a dataset claims a documented profile id but is missing recommended
//! attrs (or when history looks thin under `prov_log.v1`). Does **not** refuse open
//! or change fold answers. Unknown profile ids produce an informational finding only.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;

use super::declare::{ATTR_SEMANTICS_PROFILE, declared_attrs, is_declared_key};
use super::profiles::{PROV_LOG_V1, normalize_profile_id, recommended_keys_for_profile};
use crate::catalog::history::HistoryEvent;
use crate::catalog::package_semantics::PackageSemanticsV1;
use crate::catalog::{CatalogError, TetFileSummaryV1, TetMetadataV1, read_tet_summary_v1};
use crate::layout::mmap_file_read;

/// Severity of a linter finding (advisory only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticsLintSeverity {
    /// Missing recommended attr / history field for a claimed profile.
    Warning,
    /// Unknown profile id or soft note.
    Info,
}

/// Stable machine-readable lint code (advisory only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticsLintCode {
    /// Legacy / alias profile id; prefer the canonical spelling.
    ProfileAlias,
    /// Open-minted id with no documented recommended keys.
    UnknownProfile,
    /// Claimed documented profile is missing a recommended attr.
    MissingAttr,
    /// `prov_log.v1` with an empty history.
    EmptyHistory,
    /// `prov_log.v1` missing agent or activity on history rows.
    MissingProvParam,
}

impl SemanticsLintCode {
    /// Wire / CLI token.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProfileAlias => "profile_alias",
            Self::UnknownProfile => "unknown_profile",
            Self::MissingAttr => "missing_attr",
            Self::EmptyHistory => "empty_history",
            Self::MissingProvParam => "missing_prov_param",
        }
    }
}

impl std::fmt::Display for SemanticsLintCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One advisory finding from [`lint_declared_semantics`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SemanticsLintFinding {
    pub severity: SemanticsLintSeverity,
    /// Dataset catalog name, or `"*"` for file-wide notes.
    pub dataset: String,
    pub profile: Option<String>,
    pub code: SemanticsLintCode,
    pub message: String,
}

/// Lint report (always “ok” for open/query — findings are advisory).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
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

/// Lint footer metadata (+ optional history) for profile contract gaps.
#[must_use]
pub fn lint_declared_semantics(
    metadata: &TetMetadataV1,
    history: &[HistoryEvent],
) -> SemanticsLintReport {
    let mut findings = Vec::new();
    for (name, ds) in &metadata.datasets {
        let attrs = declared_attrs(&ds.attrs);
        let Some(profile) = attrs
            .get(ATTR_SEMANTICS_PROFILE)
            .cloned()
            .filter(|p| !p.is_empty())
        else {
            continue;
        };
        let canonical = normalize_profile_id(&profile).to_owned();
        if profile != canonical {
            findings.push(SemanticsLintFinding {
                severity: SemanticsLintSeverity::Info,
                dataset: name.clone(),
                profile: Some(profile.clone()),
                code: SemanticsLintCode::ProfileAlias,
                message: format!("legacy profile id {profile:?} → prefer {canonical:?}"),
            });
        }
        let recommended = recommended_keys_for_profile(&canonical);
        if recommended.is_empty() {
            findings.push(SemanticsLintFinding {
                severity: SemanticsLintSeverity::Info,
                dataset: name.clone(),
                profile: Some(canonical.clone()),
                code: SemanticsLintCode::UnknownProfile,
                message: format!(
                    "unknown semantics_profile {canonical:?} (open minting; no registry keys)"
                ),
            });
            continue;
        }
        for key in recommended {
            if *key == ATTR_SEMANTICS_PROFILE {
                continue;
            }
            let present = attrs.contains_key(*key) || ds.attrs.contains_key(*key);
            if !present {
                findings.push(SemanticsLintFinding {
                    severity: SemanticsLintSeverity::Warning,
                    dataset: name.clone(),
                    profile: Some(canonical.clone()),
                    code: SemanticsLintCode::MissingAttr,
                    message: format!("claims {canonical} but missing recommended attr {key:?}"),
                });
            }
        }
        if canonical == PROV_LOG_V1.id {
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
            profile: Some(PROV_LOG_V1.id.into()),
            code: SemanticsLintCode::EmptyHistory,
            message: format!("claims {} but history is empty", PROV_LOG_V1.id),
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
            profile: Some(PROV_LOG_V1.id.into()),
            code: SemanticsLintCode::MissingProvParam,
            message: format!(
                "claims {} but no history row has agent / params.agent",
                PROV_LOG_V1.id
            ),
        });
    }
    if !any_activity {
        findings.push(SemanticsLintFinding {
            severity: SemanticsLintSeverity::Warning,
            dataset: dataset.to_owned(),
            profile: Some(PROV_LOG_V1.id.into()),
            code: SemanticsLintCode::MissingProvParam,
            message: format!(
                "claims {} but no history row has activity / params.activity",
                PROV_LOG_V1.id
            ),
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
            attrs.insert(ATTR_SEMANTICS_PROFILE.into(), p.clone());
        }
        meta.datasets.insert(
            name.clone(),
            crate::catalog::metadata::DatasetMetadataV1 {
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
                attrs: BTreeMap::from([("semantics_profile".into(), "cf_lite.v1".into())]),
                ..Default::default()
            },
        );
        let report = lint_declared_semantics(&meta, &[]);
        assert!(!report.ok());
        assert!(
            report.findings.iter().any(|f| {
                f.code == SemanticsLintCode::MissingAttr && f.message.contains("units")
            })
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
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == SemanticsLintCode::ProfileAlias)
        );
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.message.contains("cf_lite.v1"))
        );
    }

    #[test]
    fn media_dot_v1_with_recommended_keys_is_clean() {
        let mut meta = TetMetadataV1::default();
        meta.datasets.insert(
            "frames".into(),
            DatasetMetadataV1 {
                attrs: BTreeMap::from([
                    ("semantics_profile".into(), "media.v1".into()),
                    ("media_kind".into(), "video".into()),
                    ("frame_rate".into(), "30".into()),
                    ("frame_rate_unit".into(), "Hz".into()),
                    ("time_axis".into(), "0".into()),
                ]),
                ..Default::default()
            },
        );
        let report = lint_declared_semantics(&meta, &[]);
        assert!(report.ok(), "{:?}", report.findings);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn v1_media_alias_prefers_canonical_id() {
        let mut meta = TetMetadataV1::default();
        meta.datasets.insert(
            "frames".into(),
            DatasetMetadataV1 {
                attrs: BTreeMap::from([
                    ("semantics_profile".into(), "v1.media".into()),
                    ("media_kind".into(), "video".into()),
                    ("frame_rate".into(), "30".into()),
                    ("frame_rate_unit".into(), "Hz".into()),
                    ("time_axis".into(), "0".into()),
                ]),
                ..Default::default()
            },
        );
        let report = lint_declared_semantics(&meta, &[]);
        assert!(report.ok());
        assert!(report.findings.iter().any(|f| {
            f.code == SemanticsLintCode::ProfileAlias && f.message.contains("media.v1")
        }));
    }

    #[test]
    fn unknown_profile_is_advisory_only() {
        let mut meta = TetMetadataV1::default();
        meta.datasets.insert(
            "a".into(),
            DatasetMetadataV1 {
                attrs: BTreeMap::from([("semantics_profile".into(), "lab.custom".into())]),
                ..Default::default()
            },
        );
        let report = lint_declared_semantics(&meta, &[]);
        assert!(report.ok());
        assert!(
            report
                .findings
                .iter()
                .any(|f| f.code == SemanticsLintCode::UnknownProfile)
        );
    }
}
