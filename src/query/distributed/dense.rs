//! Dense shard partials for distributed materialize / transform (THI-617).
//!
//! Workers emit sparse `(logical_index, value)` contributions. The coordinator
//! assembles a full logical buffer — not via [`super::FoldPartialState`].

use serde::{Deserialize, Serialize};

use crate::query::decode::chunk_decode::visit_planned_chunk;
use crate::query::types::{ReadPlan, TetError, fold_keeps_element};
use crate::utils::dtype::ElementDtype;

/// Sparse dense-shard contribution (protocol payload on [`super::WorkerPartialV1::dense`]).
///
/// `indices[i]` is a row-major logical index into the parent selection; `values_f32[i]`
/// is the value written there (NaN when a cell/label filter masks the element).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct DenseShardPartial {
    /// Wire dtype (`ElementDtype` as `u32`).
    pub dtype: u32,
    /// Logical indices written by this shard (same length as [`Self::values_f32`]).
    pub indices: Vec<u64>,
    /// f32 values parallel to [`Self::indices`] (0.7.2 harness: f32 only).
    pub values_f32: Vec<f32>,
    pub bytes_read: u64,
}

impl DenseShardPartial {
    /// Empty contribution.
    #[must_use]
    pub fn empty_f32() -> Self {
        Self {
            dtype: ElementDtype::F32.wire_tag(),
            indices: Vec::new(),
            values_f32: Vec::new(),
            bytes_read: 0,
        }
    }

    /// Bytes read across this shard's chunk visits.
    #[must_use]
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }
}

/// Decode one shard plan into a sparse f32 dense partial (hole-tolerant; no completeness check).
///
/// # Errors
///
/// Unsupported dtype, visit/decode failures, or index overflow.
pub fn materialize_shard_dense_f32(
    mmap: &[u8],
    plan: &ReadPlan,
) -> Result<DenseShardPartial, TetError> {
    let mut indices = Vec::new();
    let mut values_f32 = Vec::new();
    let mut bytes_read = 0u64;
    for c in &plan.chunks {
        let n = visit_planned_chunk(mmap, plan, c, |li, v| {
            let out_v = if fold_keeps_element(plan, li, f64::from(v)) {
                v
            } else {
                f32::NAN
            };
            let idx = u64::try_from(li)
                .map_err(|_| TetError::Validation("logical index does not fit u64".into()))?;
            indices.push(idx);
            values_f32.push(out_v);
            Ok(())
        })?;
        bytes_read = bytes_read
            .checked_add(n)
            .ok_or_else(|| TetError::Validation("bytes_read overflow".into()))?;
    }
    Ok(DenseShardPartial {
        dtype: ElementDtype::F32.wire_tag(),
        indices,
        values_f32,
        bytes_read,
    })
}

/// Compact a full logical f32 buffer into a sparse partial using shard chunk coverage.
///
/// Used after transform pass-2 (buffer already rewritten; holes remain NaN).
///
/// # Errors
///
/// Visit/decode failures or index overflow.
pub fn compact_buffer_to_dense_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    buffer: &[f32],
    bytes_read: u64,
) -> Result<DenseShardPartial, TetError> {
    let n = plan.logical_element_count;
    if buffer.len() < n {
        return Err(TetError::Validation(format!(
            "dense buffer length {} < logical_element_count {n}",
            buffer.len()
        )));
    }
    let mut indices = Vec::new();
    let mut values_f32 = Vec::new();
    for c in &plan.chunks {
        // Re-walk coverage only (values come from `buffer`, not the re-decoded sample).
        visit_planned_chunk(mmap, plan, c, |li, _v| {
            let idx = u64::try_from(li)
                .map_err(|_| TetError::Validation("logical index does not fit u64".into()))?;
            indices.push(idx);
            values_f32.push(buffer[li]);
            Ok(())
        })?;
    }
    Ok(DenseShardPartial {
        dtype: ElementDtype::F32.wire_tag(),
        indices,
        values_f32,
        bytes_read,
    })
}

/// Assemble sparse f32 shards into a full logical buffer.
///
/// # Errors
///
/// Overlapping writes, dtype mismatch, index out of range, or incomplete coverage when
/// `allow_holes` is false.
pub fn assemble_dense_f32(
    partials: &[DenseShardPartial],
    logical_element_count: usize,
    allow_holes: bool,
) -> Result<(Vec<f32>, u64), TetError> {
    let mut out = vec![f32::NAN; logical_element_count];
    let mut filled = vec![false; logical_element_count];
    let mut bytes_read = 0u64;
    for part in partials {
        if part.dtype != ElementDtype::F32.wire_tag() {
            return Err(TetError::Validation(format!(
                "dense shard dtype {} is not f32 (0.7.2 harness)",
                part.dtype
            )));
        }
        if part.indices.len() != part.values_f32.len() {
            return Err(TetError::Validation(
                "dense shard indices/values length mismatch".into(),
            ));
        }
        bytes_read = bytes_read
            .checked_add(part.bytes_read)
            .ok_or_else(|| TetError::Validation("bytes_read overflow assembling dense".into()))?;
        for (&idx, &v) in part.indices.iter().zip(&part.values_f32) {
            let i = usize::try_from(idx)
                .map_err(|_| TetError::Validation("dense shard index does not fit usize".into()))?;
            if i >= logical_element_count {
                return Err(TetError::Validation(format!(
                    "dense shard index {i} >= logical_element_count {logical_element_count}"
                )));
            }
            if filled[i] {
                return Err(TetError::Validation(format!(
                    "dense shard overlap at logical index {i}"
                )));
            }
            out[i] = v;
            filled[i] = true;
        }
    }
    if !allow_holes {
        let holes = filled.iter().filter(|&&f| !f).count();
        if holes > 0 {
            return Err(TetError::Validation(format!(
                "assembled dense selection has {holes} unset elements (chunk coverage incomplete)"
            )));
        }
    }
    Ok((out, bytes_read))
}
