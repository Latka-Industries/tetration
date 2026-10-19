//! Chunk read plan types produced after resolving a query selection against a catalog.

use serde::{Deserialize, Serialize};

use super::CellPredicate;
use crate::catalog::ChunkZoneStat;

/// Per-chunk file regions needed to satisfy the current plan (on-disk payload bytes; may be zstd-compressed).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PlannedChunkIo {
    pub chunk_index: Vec<u64>,
    pub payload_offset: u64,
    pub stored_byte_len: u64,
    pub raw_byte_len: u64,
    pub codec: u32,
    /// Min/max/null from the sealed generation's zone map, when the open file has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zone: Option<ChunkZoneStat>,
}

fn chunk_touch_policy_static(s: &str) -> Result<&'static str, String> {
    if s == CHUNK_TOUCH_POLICY.dense_half_open_unit_step {
        Ok(CHUNK_TOUCH_POLICY.dense_half_open_unit_step)
    } else if s == CHUNK_TOUCH_POLICY.strided_half_open {
        Ok(CHUNK_TOUCH_POLICY.strided_half_open)
    } else {
        Err(format!("unknown chunk_touch_policy {s:?}"))
    }
}

/// Chunk-level read plan: which tiles touch the resolved global selection box.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct ReadPlan {
    /// How `selection` maps to chunk coordinates in this build.
    pub chunk_touch_policy: &'static str,
    pub chunk_count: usize,
    /// Chunks dropped by value-filter zone pushdown (0 when no `filter` or no zone stats).
    ///
    /// Geometric touch count is `chunk_count + zone_filter_skipped_chunks`.
    #[serde(default)]
    pub zone_filter_skipped_chunks: usize,
    pub total_stored_bytes: u64,
    pub chunks: Vec<PlannedChunkIo>,
    /// Dataset global shape (same rank as selection).
    pub dataset_shape: Vec<u64>,
    pub chunk_shape: Vec<u64>,
    /// Half-open global index box used for chunk intersection (`stop` exclusive).
    pub selection_box_start: Vec<u64>,
    pub selection_box_stop_exclusive: Vec<u64>,
    /// Per-axis step (≥ 1) for strided selection.
    pub selection_step: Vec<u64>,
    /// Extents of the strided selection grid (row-major order matches materialized `f32`).
    pub logical_selection_shape: Vec<u64>,
    /// Element count of the logical row-major selection. Every dtype, not only `f32` (0.4.0).
    ///
    /// JSON accepts `logical_f32_element_count` for this release only.
    pub logical_element_count: usize,
    /// Execute-path cell filter. Not part of the JSON plan wire (`serde` skips it).
    ///
    /// Zone skip still uses `filter` while planning. Folds read this copy so a kept
    /// chunk can drop non-matching cells without a second wire key.
    #[serde(skip)]
    pub(crate) cell_filter: Option<CellPredicate>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
struct ReadPlanWire {
    chunk_touch_policy: String,
    chunk_count: usize,
    #[serde(default)]
    zone_filter_skipped_chunks: usize,
    total_stored_bytes: u64,
    chunks: Vec<PlannedChunkIo>,
    dataset_shape: Vec<u64>,
    chunk_shape: Vec<u64>,
    selection_box_start: Vec<u64>,
    selection_box_stop_exclusive: Vec<u64>,
    selection_step: Vec<u64>,
    logical_selection_shape: Vec<u64>,
    #[serde(alias = "logical_f32_element_count")]
    logical_element_count: usize,
}

impl<'de> Deserialize<'de> for ReadPlan {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = ReadPlanWire::deserialize(deserializer)?;
        let chunk_touch_policy = chunk_touch_policy_static(&wire.chunk_touch_policy)
            .map_err(serde::de::Error::custom)?;
        Ok(Self {
            chunk_touch_policy,
            chunk_count: wire.chunk_count,
            zone_filter_skipped_chunks: wire.zone_filter_skipped_chunks,
            total_stored_bytes: wire.total_stored_bytes,
            chunks: wire.chunks,
            dataset_shape: wire.dataset_shape,
            chunk_shape: wire.chunk_shape,
            selection_box_start: wire.selection_box_start,
            selection_box_stop_exclusive: wire.selection_box_stop_exclusive,
            selection_step: wire.selection_step,
            logical_selection_shape: wire.logical_selection_shape,
            logical_element_count: wire.logical_element_count,
            cell_filter: None,
        })
    }
}

/// Stable string tokens for [`ReadPlan::chunk_touch_policy`] (JSON wire compatibility).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkTouchPolicy {
    /// Half-open intervals with effective step 1 on every axis (JSON `step` omitted or 1).
    pub dense_half_open_unit_step: &'static str,
    /// Per-axis JSON `step` applied when deciding which chunks are touched.
    pub strided_half_open: &'static str,
}

/// Singleton tokens for chunk-touch policy strings on the wire.
pub const CHUNK_TOUCH_POLICY: ChunkTouchPolicy = ChunkTouchPolicy {
    dense_half_open_unit_step: "dense_half_open_intervals_per_axis_json_step_ignored_for_chunk_touch_list",
    strided_half_open: "strided_half_open_per_axis_json_step_applied_to_chunk_touch_list",
};
