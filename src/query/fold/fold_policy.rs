//! Fold I/O policy: parallel vs sequential chunk visits for streaming reductions.

use crate::query::engine::budget::ExecutionBudget;
use crate::query::fold::linear_scan;
use crate::query::types::{ExecutionHints, ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;

/// Share of [`ExecutionBudget::host_available_ram_bytes`] treated as in-core page-cache headroom.
pub const IN_CORE_IO_HEADROOM_BPS: u64 = 8500;

/// Whether the logical selection is likely served from page cache vs storage-bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoRegime {
    InCore,
    OutOfCore,
    Unknown,
}

impl IoRegime {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InCore => "in_core",
            Self::OutOfCore => "out_of_core",
            Self::Unknown => "unknown",
        }
    }
}

/// Resolved streaming-fold chunk visit strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FoldIoPolicy {
    pub parallel: bool,
    pub io_regime: IoRegime,
    /// Visit chunks in ascending `payload_offset` order (sequential fold on full scans).
    pub sequential_io: bool,
    /// Optional Rayon worker cap for parallel fold; default uses the global pool size.
    pub fold_workers: Option<usize>,
    /// Sequential byte-stream fold over one contiguous raw payload span (out-of-core full scans).
    pub linear_scan: bool,
}

impl FoldIoPolicy {
    /// Dense unit-step selection over the full dataset shape (full-tensor scan).
    #[must_use]
    pub fn is_dense_unit_step_full_selection(plan: &ReadPlan) -> bool {
        if plan.selection_step.iter().any(|&s| s != 1) {
            return false;
        }
        if plan.selection_box_start.iter().any(|&s| s != 0) {
            return false;
        }
        plan.selection_box_stop_exclusive == plan.dataset_shape
    }

    /// In-core: parallel chunks. Out-of-core full dense raw scan with contiguous payloads: linear scan.
    /// Set `execution.fold_parallel: false` for offset-ordered sequential chunk visits.
    pub fn resolve(
        plan: &ReadPlan,
        budget: &ExecutionBudget,
        hints: Option<&ExecutionHints>,
        dtype: ElementDtype,
    ) -> Result<Self, TetError> {
        let logical_bytes = budget.logical_element_bytes(dtype, plan.logical_element_count)?;
        let io_regime = resolve_io_regime(budget, logical_bytes);
        let multi_chunk = plan.chunks.len() > 1;
        let full_dense = Self::is_dense_unit_step_full_selection(plan);
        let contiguous_raw =
            linear_scan::detect_contiguous_raw_span(plan, dtype.elem_size()).is_some();

        let linear_scan = io_regime == IoRegime::OutOfCore
            && full_dense
            && contiguous_raw
            && hints.and_then(|h| h.fold_parallel) != Some(true);

        let parallel = if linear_scan {
            false
        } else {
            match hints.and_then(|h| h.fold_parallel) {
                Some(true) => multi_chunk,
                Some(false) => false,
                None if !multi_chunk => false,
                None => true,
            }
        };

        let sequential_io = !parallel && !linear_scan && multi_chunk && full_dense;

        Ok(Self {
            parallel,
            io_regime,
            sequential_io,
            fold_workers: None,
            linear_scan,
        })
    }

    /// True when chunk visits should follow ascending `payload_offset` order.
    ///
    /// Covers both `sequential_io` (multi-chunk sequential fold) and `linear_scan`
    /// (out-of-core contiguous raw span). Callers that only care about visit order —
    /// not which sequential mode — should use this instead of or-ing the flags.
    #[must_use]
    pub const fn prefer_ordered_chunk_io(self) -> bool {
        self.sequential_io || self.linear_scan
    }
}

#[must_use]
pub fn resolve_io_regime(budget: &ExecutionBudget, logical_bytes: u64) -> IoRegime {
    match budget.host_available_ram_bytes {
        Some(avail) => {
            let headroom = avail.saturating_mul(IN_CORE_IO_HEADROOM_BPS) / 10_000;
            if logical_bytes > headroom {
                IoRegime::OutOfCore
            } else {
                IoRegime::InCore
            }
        }
        None => IoRegime::Unknown,
    }
}

/// Chunk indices for sequential fold; sorted by on-disk offset when `sequential_io`.
#[must_use]
pub fn chunk_indices_for_fold(plan: &ReadPlan, sequential_io: bool) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..plan.chunks.len()).collect();
    if sequential_io && plan.chunks.len() > 1 {
        indices.sort_by_key(|&i| plan.chunks[i].payload_offset);
    }
    indices
}
