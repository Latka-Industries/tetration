//! Map verification findings to actionable recommendations.

use super::report::{VerifyFinding, VerifyFixHint, VerifyRecommendation, VerifySeverity};

/// Build deduplicated recommendations from the current finding set.
#[must_use]
pub fn recommendations_for_findings(findings: &[VerifyFinding]) -> Vec<VerifyRecommendation> {
    let mut out = Vec::new();
    for f in findings {
        if f.ok {
            continue;
        }
        let Some(rec) = recommendation_for_finding(f) else {
            continue;
        };
        if out
            .iter()
            .any(|r: &VerifyRecommendation| r.code == rec.code)
        {
            continue;
        }
        out.push(rec);
    }
    out
}

fn rec(
    code: &str,
    message: impl Into<String>,
    summary: impl Into<String>,
    command: Option<&str>,
) -> VerifyRecommendation {
    VerifyRecommendation {
        code: code.to_owned(),
        message: message.into(),
        fix: Some(VerifyFixHint {
            summary: summary.into(),
            command: command.map(str::to_owned),
        }),
    }
}

fn recommendation_for_finding(f: &VerifyFinding) -> Option<VerifyRecommendation> {
    recommendation_for_file(f).or_else(|| recommendation_for_package(f))
}

fn footer_invalid() -> VerifyRecommendation {
    rec(
        "footer_invalid",
        "History/metadata footer is present but malformed.",
        "Strip the invalid footer (truncate after chunk payloads).",
        None,
    )
}

fn recommendation_for_file(f: &VerifyFinding) -> Option<VerifyRecommendation> {
    let detail = f.detail.as_deref().unwrap_or("");
    let warn = f.severity == VerifySeverity::Warning;
    match f.check.as_str() {
        "parse" if detail.contains("footer") => Some(footer_invalid()),
        "footer" => Some(footer_invalid()),
        "parse" | "superblock" | "dataset_directory" | "chunk_index" => Some(rec(
            "invalid_layout",
            "File layout is invalid or truncated; it cannot be read reliably.",
            "Re-create the `.tet` from the source array (convert or writer session).",
            Some("tet convert <source> <output.tet>"),
        )),
        "chunk_payloads" | "payload_bounds" => Some(rec(
            "payload_out_of_bounds",
            "A chunk index entry points outside the file payload region.",
            "Restore from backup or re-export; do not append to a truncated file.",
            None,
        )),
        "chunk_dataset_ids" => Some(rec(
            "bad_dataset_id",
            "Chunk index references a non-existent dataset id.",
            "Re-write the file; the catalog index is internally inconsistent.",
            None,
        )),
        "chunk_decode" => Some(rec(
            "chunk_decode_failed",
            format!("Chunk payload could not be decoded: {detail}"),
            "Re-convert from the original HDF5/NetCDF/Zarr source.",
            Some("tet convert <source> <output.tet>"),
        )),
        "dataset_count" => Some(rec(
            "dataset_count_mismatch",
            "Superblock dataset count does not match the catalog blob.",
            "Re-create the file; superblock and catalog disagree.",
            None,
        )),
        "payload_order" if warn => Some(rec(
            "non_contiguous_payloads",
            "Chunk payloads are not stored in contiguous read-plan order.",
            "Optional: re-convert for sequential layout; full-file linear scan may be slower.",
            None,
        )),
        "chunk_decode_skipped" if warn => Some(rec(
            "run_deep_verify",
            "Not all chunks were decode-checked (large file).",
            "Re-run with `tet verify --deep` or VerifyOptions { deep_decode: true }.",
            None,
        )),
        "chunk_coords" => Some(rec(
            "duplicate_chunk_coords",
            "Packed image has duplicate dataset chunk coordinates.",
            "Republish without overlapping shard claims.",
            None,
        )),
        _ => None,
    }
}

fn recommendation_for_package(f: &VerifyFinding) -> Option<VerifyRecommendation> {
    let warn = f.severity == VerifySeverity::Warning;
    match f.check.as_str() {
        "current_packed"
        | "current_manifest"
        | "manifest_generation"
        | "manifest_protocol"
        | "manifest_byte_len"
        | "manifest_sha256"
        | "packed_layout"
        | "protocol"
        | "current" => Some(rec(
            "package_corrupt",
            "Package PROTOCOL / CURRENT / generation manifest is inconsistent.",
            "Republish from sealed staging parts, or restore generations/ from backup.",
            None,
        )),
        "publishing_stale" => Some(rec(
            "quarantine_publish",
            "A stale publish lock is blocking retry for that generation.",
            "Quarantine the stale publishing/<G>/ directory, then republish.",
            None,
        )),
        "publishing_incomplete" => Some(rec(
            "abort_or_finish_publish",
            "An incomplete publish lock is present; CURRENT is unchanged.",
            "Wait for the committer, abort_publish, or quarantine if stale.",
            None,
        )),
        "staging" if warn => Some(rec(
            "staging_gc_optional",
            "Staging sessions remain after publish (allowed by protocol).",
            "Optional GC of sealed staging once CURRENT is healthy.",
            None,
        )),
        _ => None,
    }
}
