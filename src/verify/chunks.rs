//! Chunk index and payload integrity checks.

use crate::catalog::{CHUNK_PAYLOAD_CODEC_V1, ChunkIndexEntryV1, DatasetRecordV1};

use super::report::{VerifyFinding, err_finding, ok_finding, warn_finding};

/// Max chunks to fully decode in one verify pass (larger files get a warning + sample).
pub const DEEP_DECODE_MAX_CHUNKS: usize = 128;

pub(crate) fn check_chunk_dataset_ids(
    datasets: &[DatasetRecordV1],
    chunks: &[ChunkIndexEntryV1],
) -> Option<VerifyFinding> {
    let n = datasets.len() as u64;
    for (i, ch) in chunks.iter().enumerate() {
        if ch.dataset_id >= n {
            return Some(err_finding(
                "chunk_dataset_ids",
                format!(
                    "chunk {i}: dataset_id {} >= dataset_count {n}",
                    ch.dataset_id
                ),
            ));
        }
    }
    None
}

pub(crate) fn check_payload_order(chunks: &[ChunkIndexEntryV1]) -> Option<VerifyFinding> {
    for w in chunks.windows(2) {
        let a = &w[0];
        let b = &w[1];
        if a.payload_offset.saturating_add(a.stored_byte_len) != b.payload_offset {
            return Some(warn_finding(
                "payload_order",
                "chunk payloads are not contiguous in index order (linear scan may not apply)"
                    .to_owned(),
            ));
        }
    }
    None
}

pub(crate) fn check_duplicate_payload_offsets(
    chunks: &[ChunkIndexEntryV1],
) -> Option<VerifyFinding> {
    let mut seen = std::collections::BTreeSet::new();
    for (i, ch) in chunks.iter().enumerate() {
        if !seen.insert(ch.payload_offset) {
            return Some(err_finding(
                "payload_offsets",
                format!("duplicate payload_offset at chunk index {i}"),
            ));
        }
    }
    None
}

/// Reject duplicate `(dataset_id, chunk_index)` pairs (torn / overlapping shard merge).
pub(crate) fn check_duplicate_chunk_coords(chunks: &[ChunkIndexEntryV1]) -> Option<VerifyFinding> {
    let mut seen = std::collections::BTreeSet::new();
    for (i, ch) in chunks.iter().enumerate() {
        let key = (ch.dataset_id, ch.chunk_index);
        if !seen.insert(key) {
            return Some(err_finding(
                "chunk_coords",
                format!(
                    "duplicate chunk coords at index {i} (dataset_id={}, coords={:?})",
                    ch.dataset_id, ch.chunk_index
                ),
            ));
        }
    }
    None
}

/// Decode chunk payloads to confirm stored bytes match codec and lengths.
pub(crate) fn check_chunk_decode(
    data: &[u8],
    chunks: &[ChunkIndexEntryV1],
    deep_decode: bool,
) -> (Vec<VerifyFinding>, bool) {
    let mut findings = Vec::new();
    let deep = deep_decode || chunks.len() <= DEEP_DECODE_MAX_CHUNKS;
    let to_check: Vec<usize> = if deep {
        (0..chunks.len()).collect()
    } else {
        findings.push(warn_finding(
            "chunk_decode_skipped",
            format!(
                "decode-check skipped for chunks {}..{} (limit {DEEP_DECODE_MAX_CHUNKS}); re-run with --deep or VerifyOptions::deep_decode",
                DEEP_DECODE_MAX_CHUNKS,
                chunks.len()
            ),
        ));
        (0..DEEP_DECODE_MAX_CHUNKS.min(chunks.len())).collect()
    };

    let mut decode_failures = 0_u32;
    let checked = to_check.len();
    for i in &to_check {
        let ch = &chunks[*i];
        let start = usize::try_from(ch.payload_offset).unwrap_or(usize::MAX);
        let end = start.saturating_add(usize::try_from(ch.stored_byte_len).unwrap_or(usize::MAX));
        if end > data.len() {
            findings.push(err_finding(
                "chunk_decode",
                format!("chunk {i}: payload slice out of file bounds"),
            ));
            decode_failures += 1;
            continue;
        }
        let stored = &data[start..end];
        if let Err(e) = CHUNK_PAYLOAD_CODEC_V1.decode_tile_payload(
            stored,
            ch.raw_byte_len,
            ch.stored_byte_len,
            ch.codec,
        ) {
            findings.push(err_finding("chunk_decode", format!("chunk {i}: {e}")));
            decode_failures += 1;
        }
    }

    if decode_failures == 0 {
        let detail = if deep {
            Some(format!("decoded {} chunk(s)", chunks.len()))
        } else {
            Some(format!("decoded {checked} of {} chunk(s)", chunks.len()))
        };
        findings.push(ok_finding("chunk_decode", detail));
    }
    (findings, deep)
}
