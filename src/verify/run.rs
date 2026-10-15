//! Orchestrate layout, chunk, and footer verification.

use std::path::Path;

use crate::catalog::{CatalogError, TetFileSummaryV1, read_tet_summary_v1};

use super::chunks::{
    check_chunk_dataset_ids, check_chunk_decode, check_duplicate_chunk_coords,
    check_duplicate_payload_offsets, check_payload_order,
};
use super::datasets::check_dataset_tensor_bytes;
use super::footer::{check_dataset_count, check_footer, check_footer_metadata_limits};
use super::options::VerifyOptions;
use super::report::{TetVerifyReport, VerifyFinding, VerifySummary, ok_finding};

/// Verify a mapped `.tet` byte slice.
#[must_use]
pub fn verify_tet_bytes(data: &[u8], path: Option<&Path>, opts: VerifyOptions) -> TetVerifyReport {
    let file_len = u64::try_from(data.len()).unwrap_or(u64::MAX);
    let path_s = path.map(|p| p.display().to_string());

    let summary = match read_tet_summary_v1(data) {
        Ok(s) => s,
        Err(e) => {
            return TetVerifyReport::from_fatal(path_s, file_len, "parse", e.to_string());
        }
    };

    let findings = match verify_chunks_and_datasets(&summary, path_s.clone(), file_len) {
        Ok(f) => f,
        Err(report) => return report,
    };
    let (findings, deep_decode) = verify_decode_pass(data, &summary.chunks, findings, opts);
    verify_footer_pass(data, &summary, findings, path_s, file_len, deep_decode)
}

fn initial_findings() -> Vec<VerifyFinding> {
    vec![
        ok_finding("superblock", None),
        ok_finding("dataset_directory", None),
        ok_finding("chunk_index", None),
        ok_finding(
            "chunk_payloads",
            Some("index spans validated during parse".to_owned()),
        ),
    ]
}

fn verify_chunks_and_datasets(
    summary: &TetFileSummaryV1,
    path_s: Option<String>,
    file_len: u64,
) -> Result<Vec<VerifyFinding>, TetVerifyReport> {
    let mut findings = initial_findings();

    if let Some(f) = check_chunk_dataset_ids(&summary.datasets, &summary.chunks) {
        findings.push(f);
        return Err(TetVerifyReport::incomplete(path_s, file_len, findings));
    }
    findings.push(ok_finding("chunk_dataset_ids", None));

    findings.extend(check_dataset_tensor_bytes(
        &summary.datasets,
        &summary.chunks,
    ));
    if findings
        .iter()
        .any(|f| f.check == "dataset_tensor_bytes" && !f.ok)
    {
        return Err(TetVerifyReport::incomplete(path_s, file_len, findings));
    }

    if let Some(f) = check_duplicate_payload_offsets(&summary.chunks) {
        findings.push(f);
    }
    if let Some(f) = check_duplicate_chunk_coords(&summary.chunks) {
        findings.push(f);
    } else {
        findings.push(ok_finding("chunk_coords", None));
    }
    if let Some(f) = check_payload_order(&summary.chunks) {
        findings.push(f);
    }

    Ok(findings)
}

fn verify_decode_pass(
    data: &[u8],
    chunks: &[crate::catalog::ChunkIndexEntryV1],
    mut findings: Vec<VerifyFinding>,
    opts: VerifyOptions,
) -> (Vec<VerifyFinding>, bool) {
    let (decode_findings, deep_decode) = check_chunk_decode(data, chunks, opts.deep_decode);
    findings.extend(decode_findings);
    (findings, deep_decode)
}

fn verify_footer_pass(
    data: &[u8],
    summary: &TetFileSummaryV1,
    mut findings: Vec<VerifyFinding>,
    path_s: Option<String>,
    file_len: u64,
    deep_decode: bool,
) -> TetVerifyReport {
    if let Some(f) = check_dataset_count(summary) {
        findings.push(f);
        return TetVerifyReport::incomplete(path_s, file_len, findings);
    }
    findings.push(ok_finding("dataset_count", None));

    match check_footer(data, summary) {
        Ok(f) => findings.push(f),
        Err(f) => {
            findings.push(f);
            return TetVerifyReport::incomplete(path_s, file_len, findings);
        }
    }

    let meta_finding = check_footer_metadata_limits(summary);
    if !meta_finding.ok {
        findings.push(meta_finding);
        return TetVerifyReport::incomplete(path_s, file_len, findings);
    }
    findings.push(meta_finding);

    let has_metadata = !summary.metadata.datasets.is_empty() || summary.metadata.file.is_some();
    let history_footer =
        summary.superblock.flags & crate::layout::SUPERBLOCK_FLAG_HISTORY_FOOTER != 0;

    TetVerifyReport {
        ok: true,
        path: path_s,
        file_len,
        findings,
        recommendations: Vec::new(),
        summary: Some(VerifySummary {
            layout_version: summary.superblock.layout_version,
            dataset_count: summary.datasets.len(),
            chunk_count: summary.chunks.len(),
            history_events: summary.history.len(),
            has_metadata,
            history_footer,
            deep_chunk_decode: deep_decode,
        }),
    }
    .finalize()
}

/// Verify a `.tet` file on disk (reads entire file into memory).
///
/// # Errors
///
/// Returns [`CatalogError::Io`] when the file cannot be read.
pub fn verify_tet_file(path: &Path) -> Result<TetVerifyReport, CatalogError> {
    verify_tet_file_with_options(path, VerifyOptions::default())
}

/// Verify with explicit [`VerifyOptions`] (e.g. [`VerifyOptions::deep_decode`]).
///
/// # Errors
///
/// Returns [`CatalogError::Io`] when the file cannot be read.
pub fn verify_tet_file_with_options(
    path: &Path,
    opts: VerifyOptions,
) -> Result<TetVerifyReport, CatalogError> {
    let data = std::fs::read(path)?;
    Ok(verify_tet_bytes(&data, Some(path), opts))
}
