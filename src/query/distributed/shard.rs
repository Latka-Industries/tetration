//! Split a [`ReadPlan`] into disjoint chunk shards (THI-576).

use serde::{Deserialize, Serialize};

use crate::query::types::ReadPlan;

/// How to partition `plan.chunks` indices across workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShardPartition {
    /// Contiguous `chunks[start..end)` ranges (ceil-split). Prefer for I/O locality.
    #[default]
    ContiguousRanges,
    /// Interleaved indices `shard`, `shard + n`, … (same pattern as multi-GPU).
    Interleaved,
}

/// One worker's share of parent `plan.chunks` indices.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ShardSpec {
    pub shard_id: usize,
    pub n_shards: usize,
    /// Indices into the coordinator's `ReadPlan.chunks`.
    pub chunk_indices: Vec<usize>,
}

/// Partition `0..n_chunks` into `n_shards` disjoint specs (coverage exact, order preserved).
///
/// Empty plans yield `n_shards` empty specs. `n_shards == 0` returns an empty vec.
#[must_use]
pub fn plan_chunk_shards(
    n_chunks: usize,
    n_shards: usize,
    partition: ShardPartition,
) -> Vec<ShardSpec> {
    if n_shards == 0 {
        return Vec::new();
    }
    match partition {
        ShardPartition::ContiguousRanges => contiguous_shards(n_chunks, n_shards),
        ShardPartition::Interleaved => (0..n_shards)
            .map(|shard_id| ShardSpec {
                shard_id,
                n_shards,
                chunk_indices: interleaved_chunk_indices(shard_id, n_shards, n_chunks),
            })
            .collect(),
    }
}

fn contiguous_shards(n_chunks: usize, n_shards: usize) -> Vec<ShardSpec> {
    let base = n_chunks / n_shards;
    let rem = n_chunks % n_shards;
    let mut start = 0usize;
    (0..n_shards)
        .map(|shard_id| {
            let len = base + usize::from(shard_id < rem);
            let end = start + len;
            let chunk_indices = (start..end).collect();
            start = end;
            ShardSpec {
                shard_id,
                n_shards,
                chunk_indices,
            }
        })
        .collect()
}

/// Indices `shard`, `shard + n_shards`, … covering `0..total` exactly once across shards.
#[must_use]
pub fn interleaved_chunk_indices(shard: usize, n_shards: usize, total: usize) -> Vec<usize> {
    debug_assert!(n_shards == 0 || shard < n_shards);
    if n_shards == 0 {
        return Vec::new();
    }
    (shard..total).step_by(n_shards).collect()
}

/// Clone geometry; subset `chunks` per `spec`; recompute `chunk_count` + `total_stored_bytes`.
///
/// Selection geometry and `logical_element_count` stay identical on every shard so
/// argmin/argmax logical indices remain global. `cell_filter` is cloned (not on JSON wire).
/// Zone/index skip counters are copied from the parent (coordinator log stats).
#[must_use]
pub fn split_read_plan(plan: &ReadPlan, spec: &ShardSpec) -> ReadPlan {
    let chunks: Vec<_> = spec
        .chunk_indices
        .iter()
        .map(|&i| plan.chunks[i].clone())
        .collect();
    let total_stored_bytes = chunks.iter().map(|c| c.stored_byte_len).sum();
    ReadPlan {
        chunk_touch_policy: plan.chunk_touch_policy,
        chunk_count: chunks.len(),
        zone_filter_skipped_chunks: plan.zone_filter_skipped_chunks,
        index_filter_skipped_chunks: plan.index_filter_skipped_chunks,
        total_stored_bytes,
        chunks,
        dataset_shape: plan.dataset_shape.clone(),
        chunk_shape: plan.chunk_shape.clone(),
        selection_box_start: plan.selection_box_start.clone(),
        selection_box_stop_exclusive: plan.selection_box_stop_exclusive.clone(),
        selection_step: plan.selection_step.clone(),
        logical_selection_shape: plan.logical_selection_shape.clone(),
        logical_element_count: plan.logical_element_count,
        cell_filter: plan.cell_filter,
        label_eq: plan.label_eq.clone(),
        label_group_remap: plan.label_group_remap.clone(),
        dataset_fill: plan.dataset_fill,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_covers_exactly_once() {
        let specs = plan_chunk_shards(10, 3, ShardPartition::ContiguousRanges);
        assert_eq!(specs.len(), 3);
        let mut all: Vec<usize> = specs.iter().flat_map(|s| s.chunk_indices.clone()).collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        assert_eq!(specs[0].chunk_indices, vec![0, 1, 2, 3]);
        assert_eq!(specs[1].chunk_indices, vec![4, 5, 6]);
        assert_eq!(specs[2].chunk_indices, vec![7, 8, 9]);
    }

    #[test]
    fn interleaved_covers_exactly_once() {
        let specs = plan_chunk_shards(10, 3, ShardPartition::Interleaved);
        let mut all: Vec<usize> = specs.iter().flat_map(|s| s.chunk_indices.clone()).collect();
        all.sort_unstable();
        assert_eq!(all, (0..10).collect::<Vec<_>>());
        assert_eq!(specs[0].chunk_indices, vec![0, 3, 6, 9]);
        assert_eq!(specs[1].chunk_indices, vec![1, 4, 7]);
        assert_eq!(specs[2].chunk_indices, vec![2, 5, 8]);
    }
}
