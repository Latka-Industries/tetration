//! Windowed reduction geometry along one axis (bins / rolling).

use std::collections::BTreeSet;

use crate::query::decode::indexing::linear_rm_index;
use crate::query::plan::read_plan::shape_product_usize;
use crate::query::types::{ReadPlan, TetError, WindowMode, WindowSpec};

/// Layout for a windowed partial-axis reduction.
#[derive(Debug, Clone)]
pub(crate) struct WindowAxisLayout {
    pub window_axis: usize,
    pub size: u64,
    pub mode: WindowMode,
    pub window_count: usize,
    /// Axes fully reduced within each window (always includes `window_axis`).
    pub reduce_set: BTreeSet<usize>,
    pub out_shape: Vec<u64>,
    pub out_len: usize,
}

/// Build window layout from a resolved [`WindowSpec`] and reduction axis labels.
pub(crate) fn window_axis_layout(
    plan: &ReadPlan,
    window: &WindowSpec,
    axis_labels: &[String],
) -> Result<WindowAxisLayout, TetError> {
    let shape = &plan.logical_selection_shape;
    let ndim = shape.len();
    let window_axis = window.axis.parse::<usize>().map_err(|_| {
        TetError::Validation(format!(
            "invalid window axis {:?} after name resolution",
            window.axis
        ))
    })?;
    if window_axis >= ndim {
        return Err(TetError::Validation(format!(
            "window axis {window_axis} out of range for rank {ndim}"
        )));
    }
    if window.size == 0 {
        return Err(TetError::Validation("`window.size` must be >= 1".into()));
    }
    let extent = shape[window_axis];
    if window.size > extent {
        return Err(TetError::Validation(format!(
            "`window.size` ({}) exceeds axis extent ({extent})",
            window.size
        )));
    }
    let window_count = match window.mode {
        WindowMode::Bins => {
            let n = extent.div_ceil(window.size);
            usize::try_from(n)
                .map_err(|_| TetError::Validation("window count overflows usize".into()))?
        }
        WindowMode::Rolling => {
            let n = extent - window.size + 1;
            usize::try_from(n)
                .map_err(|_| TetError::Validation("window count overflows usize".into()))?
        }
    };
    if window_count == 0 {
        return Err(TetError::Validation(
            "window produces zero output positions".into(),
        ));
    }

    let mut reduce_set = BTreeSet::new();
    for label in axis_labels {
        let v: usize = label
            .parse()
            .map_err(|_| TetError::Validation(format!("invalid operation axis index {label:?}")))?;
        if v >= ndim {
            return Err(TetError::Validation(format!(
                "operation axis index {v} out of range for rank {ndim}"
            )));
        }
        if !reduce_set.insert(v) {
            return Err(TetError::Validation(format!(
                "duplicate operation axis index {v}"
            )));
        }
    }
    if !reduce_set.contains(&window_axis) {
        return Err(TetError::Validation(
            "internal: window layout requires the windowed axis in the reduction set".into(),
        ));
    }

    // Window axis becomes `window_count`; other reduced axes drop; kept axes retain extent.
    // When every axis is reduced, the output is the 1-D list of window results.
    let mut out_shape = Vec::new();
    if reduce_set.len() == ndim {
        out_shape.push(window_count as u64);
    } else {
        for (d, &e) in shape.iter().enumerate() {
            if d == window_axis {
                out_shape.push(window_count as u64);
            } else if !reduce_set.contains(&d) {
                out_shape.push(e);
            }
        }
    }
    let out_len = shape_product_usize(&out_shape)?;
    Ok(WindowAxisLayout {
        window_axis,
        size: window.size,
        mode: window.mode,
        window_count,
        reduce_set,
        out_shape,
        out_len,
    })
}

/// Output cell index for one bins/rolling window id and input coordinates.
pub(crate) fn window_reduced_index(
    coords: &[usize],
    window_idx: usize,
    layout: &WindowAxisLayout,
) -> Result<usize, TetError> {
    if layout.reduce_set.len() == coords.len() {
        return Ok(window_idx);
    }
    let mut out_c = Vec::with_capacity(layout.out_shape.len());
    for (d, &c) in coords.iter().enumerate() {
        if d == layout.window_axis {
            out_c.push(window_idx);
        } else if !layout.reduce_set.contains(&d) {
            out_c.push(c);
        }
    }
    linear_rm_index(&out_c, &layout.out_shape)
}

/// Window indices that include `coords[window_axis]` for this layout.
pub(crate) fn windows_covering(
    coords: &[usize],
    layout: &WindowAxisLayout,
) -> Result<Vec<usize>, TetError> {
    let c = *coords
        .get(layout.window_axis)
        .ok_or_else(|| TetError::Validation("coordinate rank mismatch for window fold".into()))?;
    let c = u64::try_from(c).map_err(|_| TetError::Validation("axis coord overflow".into()))?;
    match layout.mode {
        WindowMode::Bins => {
            let w = c / layout.size;
            let w = usize::try_from(w)
                .map_err(|_| TetError::Validation("window index overflow".into()))?;
            if w >= layout.window_count {
                return Err(TetError::Validation(format!(
                    "bins window index {w} out of range ({})",
                    layout.window_count
                )));
            }
            Ok(vec![w])
        }
        WindowMode::Rolling => {
            // Cell at c belongs to every window w with w <= c <= w+size-1.
            let size = layout.size;
            let first = c.saturating_sub(size - 1);
            let last = c.min(u64::try_from(layout.window_count.saturating_sub(1)).unwrap_or(0));
            let mut out = Vec::new();
            let mut w = first;
            while w <= last {
                let wi = usize::try_from(w)
                    .map_err(|_| TetError::Validation("window index overflow".into()))?;
                if wi < layout.window_count {
                    out.push(wi);
                }
                if w == u64::MAX {
                    break;
                }
                w += 1;
            }
            Ok(out)
        }
    }
}
