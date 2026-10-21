//! Fast paths when a chunk tile lies fully inside a unit-step selection box.

use crate::catalog::MAX_NDIM;
use crate::query::decode::indexing::linear_rm_index;
use crate::query::types::{ReadPlan, TetError};

/// True when every selection axis uses step `1` (dense, no striding).
#[must_use]
pub(crate) fn selection_is_unit_step(plan: &ReadPlan) -> bool {
    plan.selection_step.iter().all(|&s| s == 1)
}

/// Logical row-major index for a global coordinate under a unit-step selection.
pub(crate) fn logical_index_unit_step(
    plan: &ReadPlan,
    global: &[u64],
    ndim: usize,
) -> Result<usize, TetError> {
    let mut lc = [0usize; MAX_NDIM];
    for d in 0..ndim {
        let q = global[d] - plan.selection_box_start[d];
        lc[d] = usize::try_from(q).map_err(|_| {
            TetError::Validation(format!(
                "logical coordinate on axis {d} does not fit usize on this host"
            ))
        })?;
    }
    linear_rm_index(&lc[..ndim], &plan.logical_selection_shape)
}

/// True when a tile extent is a single contiguous block in row-major order of `sel_shape`.
///
/// Innermost axes may be full-width; at most one partial axis; every slower axis must have
/// extent `1`. That is exactly when payload order matches `li_base .. li_base + nelem`.
#[must_use]
pub(crate) fn tile_is_rm_contiguous(sel_shape: &[u64], tile: &[u64]) -> bool {
    if sel_shape.len() != tile.len() || sel_shape.is_empty() {
        return false;
    }
    let ndim = sel_shape.len();
    let mut d = ndim;
    while d > 0 {
        d -= 1;
        if tile[d] != sel_shape[d] {
            break;
        }
        if d == 0 {
            return true; // tile spans the full selection
        }
    }
    // `d` is the innermost non-full axis (or 0 if axis 0 is partial / we broke there).
    if tile[d] != sel_shape[d] {
        for &t in tile.iter().take(d) {
            if t != 1 {
                return false;
            }
        }
    }
    true
}

/// When the tile at `chunk_coord` is fully inside the selection box, step is `1`, and the tile
/// is row-major-contiguous in the selection, return the logical index of the tile's first element.
///
/// Callers may then treat payload order as `li_base + k` (slab SIMD / dense visit). Row-chunked
/// multi-dim tiles (full trailing axes) qualify; partial trailing tiles do not.
pub(crate) fn dense_tile_logical_base(
    plan: &ReadPlan,
    chunk_coord: &[u64],
    tile: &[u64],
) -> Result<Option<usize>, TetError> {
    if !selection_is_unit_step(plan) {
        return Ok(None);
    }
    let ndim = plan.dataset_shape.len();
    if chunk_coord.len() != ndim || tile.len() != ndim {
        return Ok(None);
    }
    if !tile_is_rm_contiguous(&plan.logical_selection_shape, tile) {
        return Ok(None);
    }
    for d in 0..ndim {
        let g0 = chunk_coord[d].saturating_mul(plan.chunk_shape[d]);
        let g1 = g0.saturating_add(tile[d]);
        if g0 < plan.selection_box_start[d] || g1 > plan.selection_box_stop_exclusive[d] {
            return Ok(None);
        }
    }
    let mut g0 = [0u64; MAX_NDIM];
    for d in 0..ndim {
        g0[d] = chunk_coord[d] * plan.chunk_shape[d];
    }
    Ok(Some(logical_index_unit_step(plan, &g0[..ndim], ndim)?))
}
