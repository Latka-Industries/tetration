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

/// When the tile at `chunk_coord` is fully inside the selection box and step is `1`, return the
/// logical row-major index of the tile's first element.
///
/// Valid at any rank: a fully contained tile is contiguous in payload order and in logical
/// indices (`li_base .. li_base + nelem`), so fold/visit can take the slab fast path.
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
