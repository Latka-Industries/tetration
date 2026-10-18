//! Execution memory budget: when to stream, spill, or hold buffers in RAM.

use crate::catalog::FileExecutionSettingsV1;
use crate::query::types::{ExecutionHints, ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;
use crate::utils::host_memory;

/// Fallback cap when host RAM cannot be detected and no fixed budget is set (256 MiB).
pub const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// How the engine chose to execute against a read plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStrategy {
    /// Scalar / partial reductions fold over chunks without a full logical tensor buffer.
    StreamingFold,
    /// Preview or spill uses a capped in-memory slice only.
    CappedInMemory,
    /// Full logical tensor written to a caller path via file mmap (disk-backed, not a giant `Vec`).
    MmapSpill,
    /// Tier-C op: full logical selection held in RAM for the operation.
    InMemoryMaterialize,
    /// Tier-C op: full logical selection written to an engine temp file, then removed.
    TempSpillMaterialize,
    /// Transform pass-2: dense output held in RAM.
    TransformRam,
    /// Transform pass-2: dense output written to a spill file (caller path or cache temp).
    TransformSpill,
    /// Transform pass-2: `.tet` sidecar draft in cache, published beside the source file.
    TransformSidecar,
}

impl MemoryStrategy {
    /// Stable `snake_case` token written to execution preview JSON.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StreamingFold => "streaming_fold",
            Self::CappedInMemory => "capped_in_memory",
            Self::MmapSpill => "mmap_spill",
            Self::InMemoryMaterialize => "in_memory_materialize",
            Self::TempSpillMaterialize => "temp_spill_materialize",
            Self::TransformRam => "transform_ram",
            Self::TransformSpill => "transform_spill",
            Self::TransformSidecar => "transform_sidecar",
        }
    }
}

/// Host memory budget for query execution (does not limit mmap spill **file** size).
#[derive(Debug, Clone, Copy)]
pub struct ExecutionBudget {
    pub memory_budget_bytes: u64,
    /// Host RAM available when the budget was resolved, if detectable.
    pub host_available_ram_bytes: Option<u64>,
    /// Percent of host RAM used when the budget was derived from a percent (basis points).
    pub memory_budget_percent_bps: u16,
}

impl Default for ExecutionBudget {
    fn default() -> Self {
        Self::resolve(&FileExecutionSettingsV1::default_engine(), None)
    }
}

impl ExecutionBudget {
    /// Resolve budget: query JSON overrides `.tet` header overrides default percent of host RAM.
    #[must_use]
    pub fn resolve(file: &FileExecutionSettingsV1, hints: Option<&ExecutionHints>) -> Self {
        let host = host_memory::available_memory_bytes();

        if let Some(bytes) = hints.and_then(|h| h.memory_budget_bytes) {
            return Self {
                memory_budget_bytes: bytes.max(1),
                host_available_ram_bytes: host,
                memory_budget_percent_bps: file.effective_percent_bps(),
            };
        }
        if file.memory_budget_bytes > 0 {
            return Self {
                memory_budget_bytes: u64::from(file.memory_budget_bytes).max(1),
                host_available_ram_bytes: host,
                memory_budget_percent_bps: file.effective_percent_bps(),
            };
        }

        let bps = hints
            .and_then(|h| h.memory_budget_percent_bps)
            .filter(|&p| p > 0)
            .unwrap_or_else(|| file.effective_percent_bps());

        let memory_budget_bytes = if let Some(avail) = host {
            avail.saturating_mul(u64::from(bps)) / 10_000
        } else {
            DEFAULT_MEMORY_BUDGET_BYTES
        };

        Self {
            memory_budget_bytes: memory_budget_bytes.max(1),
            host_available_ram_bytes: host,
            memory_budget_percent_bps: bps,
        }
    }

    /// Resolve budget from query JSON hints only (no per-file TIDX overrides).
    #[must_use]
    pub fn from_hints(hints: Option<&ExecutionHints>) -> Self {
        Self::resolve(&FileExecutionSettingsV1::default_engine(), hints)
    }

    /// # Errors
    ///
    /// Returns [`TetError::Validation`] when the element count or byte product overflows.
    pub fn logical_element_bytes(
        &self,
        dtype: ElementDtype,
        element_count: usize,
    ) -> Result<u64, TetError> {
        let count = u64::try_from(element_count)
            .map_err(|_| TetError::Validation("logical element count overflow".into()))?;
        dtype
            .bytes_from_elem_count(count)
            .ok_or_else(|| TetError::Validation("logical element byte size overflow".into()))
    }

    /// # Errors
    ///
    /// Returns [`TetError::Validation`] when the element count or byte product overflows.
    pub fn logical_f32_bytes(&self, element_count: usize) -> Result<u64, TetError> {
        self.logical_element_bytes(ElementDtype::F32, element_count)
    }

    /// # Errors
    ///
    /// Propagates errors from [`Self::logical_element_bytes`].
    pub fn exceeds_budget(
        &self,
        dtype: ElementDtype,
        element_count: usize,
    ) -> Result<bool, TetError> {
        Ok(self.logical_element_bytes(dtype, element_count)? > self.memory_budget_bytes)
    }

    /// # Errors
    ///
    /// Propagates errors from [`Self::logical_f32_bytes`].
    pub fn exceeds_budget_f32(&self, element_count: usize) -> Result<bool, TetError> {
        self.exceeds_budget(ElementDtype::F32, element_count)
    }

    /// # Errors
    ///
    /// Propagates errors from [`Self::exceeds_budget`].
    pub fn full_tensor_exceeds_budget(
        &self,
        plan: &ReadPlan,
        dtype: ElementDtype,
    ) -> Result<bool, TetError> {
        self.exceeds_budget(dtype, plan.logical_element_count)
    }

    /// Whether a dense in-memory logical buffer for `plan` would exceed the RAM budget (f32 bytes).
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::exceeds_budget_f32`].
    pub fn full_tensor_exceeds_budget_f32(&self, plan: &ReadPlan) -> Result<bool, TetError> {
        self.full_tensor_exceeds_budget(plan, ElementDtype::F32)
    }
}
