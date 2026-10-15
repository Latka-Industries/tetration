//! Logical row-major linear index helpers shared by materialization and reductions.
//!
//! Last axis is contiguous in memory (C-order). Used by scatter/fold paths and by
//! THI-377 write-invariant tests to cross-check chunk-local offsets.

use crate::query::types::TetError;

/// Row-major linear index of `logical_coords` inside `shape` (last axis fastest).
///
/// # Errors
///
/// Returns [`TetError::Validation`] on rank mismatch, out-of-range coords, or overflow.
pub(crate) fn linear_rm_index(logical_coords: &[usize], shape: &[u64]) -> Result<usize, TetError> {
    if logical_coords.len() != shape.len() {
        return Err(TetError::Validation(
            "internal: logical coordinate rank mismatch".into(),
        ));
    }
    let mut idx = 0usize;
    let mut stride = 1usize;
    for d in (0..shape.len()).rev() {
        let sd = usize::try_from(shape[d]).map_err(|_| {
            TetError::Validation(format!(
                "logical shape extent {} is too large for this host",
                shape[d]
            ))
        })?;
        let c = logical_coords[d];
        if c >= sd {
            return Err(TetError::Validation(format!(
                "internal: logical coordinate {c} out of range for axis {d} (extent {sd})"
            )));
        }
        idx = idx
            .checked_add(
                c.checked_mul(stride)
                    .ok_or_else(|| TetError::Validation("linear index overflow".into()))?,
            )
            .ok_or_else(|| TetError::Validation("linear index overflow".into()))?;
        stride = stride
            .checked_mul(sd)
            .ok_or_else(|| TetError::Validation("linear index stride overflow".into()))?;
    }
    Ok(idx)
}

/// Inverse of [`linear_rm_index`]: multi-dimensional coords for linear index `li`.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when an extent does not fit `usize`, is zero, or
/// conversion overflows.
pub(crate) fn coords_from_linear_row_major(
    mut li: usize,
    shape: &[u64],
) -> Result<Vec<usize>, TetError> {
    let mut coords = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        let sd = usize::try_from(shape[d]).map_err(|_| {
            TetError::Validation(format!(
                "shape extent {} is too large for this host",
                shape[d]
            ))
        })?;
        if sd == 0 {
            return Err(TetError::Validation(
                "internal: zero extent in shape".into(),
            ));
        }
        coords[d] = li % sd;
        li /= sd;
    }
    Ok(coords)
}
