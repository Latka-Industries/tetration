//! Per-dataset chunk grid and logical tensor byte-length cross-checks.

use crate::catalog::tile::{chunk_grid_counts, tile_raw_byte_len, total_chunk_count};
use crate::catalog::{ChunkIndexEntryV1, DatasetRecordV1};
use crate::utils::dtype::ElementDtype;

use super::report::{VerifyFinding, err_finding, ok_finding};

pub(crate) fn check_dataset_tensor_bytes(
    datasets: &[DatasetRecordV1],
    chunks: &[ChunkIndexEntryV1],
) -> Vec<VerifyFinding> {
    let per_ds = chunk_indices_by_dataset(datasets.len(), chunks);
    let mut findings = Vec::new();
    for (ds_id, ds) in datasets.iter().enumerate() {
        findings.extend(check_one_dataset_tensor_bytes(
            ds_id,
            ds,
            &per_ds[ds_id],
            chunks,
        ));
    }
    findings
}

fn chunk_indices_by_dataset(dataset_count: usize, chunks: &[ChunkIndexEntryV1]) -> Vec<Vec<usize>> {
    let mut per_ds = vec![Vec::new(); dataset_count];
    for (i, ch) in chunks.iter().enumerate() {
        let id = usize::try_from(ch.dataset_id).unwrap_or(usize::MAX);
        if id < dataset_count {
            per_ds[id].push(i);
        }
    }
    per_ds
}

fn check_one_dataset_tensor_bytes(
    ds_id: usize,
    ds: &DatasetRecordV1,
    chunk_indices: &[usize],
    chunks: &[ChunkIndexEntryV1],
) -> Vec<VerifyFinding> {
    let ndim = ds.shape.len();
    let Some(elem) = ElementDtype::try_from_wire_tag(ds.dtype) else {
        return vec![err_finding(
            "dataset_tensor_bytes",
            format!(
                "dataset {ds_id} ({:?}): unsupported dtype tag {}",
                ds.name, ds.dtype
            ),
        )];
    };
    let Some(expected_total) = elem.tensor_bytes_for_shape(&ds.shape) else {
        return vec![err_finding(
            "dataset_tensor_bytes",
            format!(
                "dataset {ds_id} ({:?}): logical tensor byte length overflow",
                ds.name
            ),
        )];
    };
    let counts = match chunk_grid_counts(&ds.shape, &ds.chunk_shape) {
        Ok(c) => c,
        Err(e) => {
            return vec![err_finding(
                "dataset_tensor_bytes",
                format!("dataset {ds_id} ({:?}): {e}", ds.name),
            )];
        }
    };
    let expected_chunks = match total_chunk_count(&counts) {
        Ok(n) => n,
        Err(e) => {
            return vec![err_finding(
                "dataset_tensor_bytes",
                format!("dataset {ds_id} ({:?}): {e}", ds.name),
            )];
        }
    };
    if u64::try_from(chunk_indices.len()).ok() != Some(expected_chunks) {
        return vec![err_finding(
            "dataset_tensor_bytes",
            format!(
                "dataset {ds_id} ({:?}): chunk count {} != expected grid {}",
                ds.name,
                chunk_indices.len(),
                expected_chunks
            ),
        )];
    }

    let mut sum_raw: u64 = 0;
    for &ci in chunk_indices {
        let ch = &chunks[ci];
        let coord = &ch.chunk_index[..ndim];
        let expected_tile =
            match tile_raw_byte_len(&ds.shape, &ds.chunk_shape, coord, ndim, elem.elem_size()) {
                Ok(n) => n,
                Err(e) => {
                    return vec![err_finding(
                        "dataset_tensor_bytes",
                        format!("dataset {ds_id} chunk {ci}: {e}"),
                    )];
                }
            };
        if ch.raw_byte_len != expected_tile {
            return vec![err_finding(
                "dataset_tensor_bytes",
                format!(
                    "dataset {ds_id} chunk {ci}: raw_byte_len {} != tile {}",
                    ch.raw_byte_len, expected_tile
                ),
            )];
        }
        sum_raw = sum_raw.saturating_add(ch.raw_byte_len);
    }

    if sum_raw == expected_total {
        vec![ok_finding(
            "dataset_tensor_bytes",
            Some(format!(
                "dataset {ds_id} ({:?}): {expected_chunks} chunk(s), {expected_total} B",
                ds.name
            )),
        )]
    } else {
        vec![err_finding(
            "dataset_tensor_bytes",
            format!(
                "dataset {ds_id} ({:?}): sum(raw_byte_len)={sum_raw} != logical tensor {expected_total}",
                ds.name
            ),
        )]
    }
}
