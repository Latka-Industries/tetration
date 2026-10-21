//! Windowed reduction geometry along one axis (bins / rolling).

use std::collections::BTreeSet;

use crate::query::decode::indexing::linear_rm_index;
use crate::query::fold::partial_geometry;
use crate::query::plan::read_plan::shape_product_usize;
use crate::query::types::{ReadPlan, TetError, WindowAlign, WindowMode, WindowPad, WindowSpec};

/// Layout for a windowed partial-axis reduction.
#[derive(Debug, Clone)]
pub(crate) struct WindowAxisLayout {
    pub window_axis: usize,
    pub size: u64,
    pub mode: WindowMode,
    pub pad: WindowPad,
    pub step: u64,
    pub align: WindowAlign,
    pub min_count: u64,
    pub window_count: usize,
    /// First window start along the axis (may be negative under rolling + pad).
    pub first_start: i64,
    /// Product of extents of reduced axes other than the window axis.
    pub other_reduce_product: u64,
    /// Axes fully reduced within each window (always includes `window_axis`).
    pub reduce_set: BTreeSet<usize>,
    pub out_shape: Vec<u64>,
    pub out_len: usize,
    /// Index of the window dimension inside [`Self::out_shape`].
    pub out_window_dim: usize,
    /// Axis extent (selection shape on the windowed axis).
    pub extent: u64,
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
    window.validate_cross_fields()?;
    let extent = shape[window_axis];
    if window.size > extent {
        return Err(TetError::Validation(format!(
            "`window.size` ({}) exceeds axis extent ({extent})",
            window.size
        )));
    }
    let step = window.effective_step();

    let (first_start, window_count) = match window.mode {
        WindowMode::Bins => {
            let n = extent.div_ceil(window.size);
            let count = usize::try_from(n)
                .map_err(|_| TetError::Validation("window count overflows usize".into()))?;
            (0_i64, count)
        }
        WindowMode::Rolling => {
            rolling_geometry(extent, window.size, step, window.pad, window.align)?
        }
    };
    if window_count == 0 {
        return Err(TetError::Validation(
            "window produces zero output positions".into(),
        ));
    }

    let reduce_set: BTreeSet<usize> = partial_geometry::parse_axis_indices(axis_labels, ndim)?
        .into_iter()
        .collect();
    if !reduce_set.contains(&window_axis) {
        return Err(TetError::Validation(
            "internal: window layout requires the windowed axis in the reduction set".into(),
        ));
    }

    let mut other_reduce_product = 1_u64;
    for (d, &e) in shape.iter().enumerate() {
        if d != window_axis && reduce_set.contains(&d) {
            other_reduce_product = other_reduce_product.checked_mul(e).ok_or_else(|| {
                TetError::Validation("window other-reduce product overflows".into())
            })?;
        }
    }

    // Window axis becomes `window_count`; other reduced axes drop; kept axes retain extent.
    // When every axis is reduced, the output is the 1-D list of window results.
    let mut out_shape = Vec::new();
    let out_window_dim;
    if reduce_set.len() == ndim {
        out_shape.push(window_count as u64);
        out_window_dim = 0;
    } else {
        out_window_dim = {
            let mut dim = 0usize;
            for d in 0..window_axis {
                if !reduce_set.contains(&d) {
                    dim += 1;
                }
            }
            dim
        };
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
        pad: window.pad,
        step,
        align: window.align,
        min_count: window.min_count,
        window_count,
        first_start,
        other_reduce_product,
        reduce_set,
        out_shape,
        out_len,
        out_window_dim,
        extent,
    })
}

fn rolling_geometry(
    extent: u64,
    size: u64,
    step: u64,
    pad: WindowPad,
    align: WindowAlign,
) -> Result<(i64, usize), TetError> {
    let extent_i = i64::try_from(extent)
        .map_err(|_| TetError::Validation("window axis extent overflows i64".into()))?;
    let size_i = i64::try_from(size)
        .map_err(|_| TetError::Validation("window.size overflows i64".into()))?;
    let step_i = i64::try_from(step)
        .map_err(|_| TetError::Validation("window.step overflows i64".into()))?;
    let (first, last) = match (pad, align) {
        (WindowPad::None, _) => {
            // Fully in-bounds starts only: 0, step, … while start + size - 1 < extent.
            let last = extent_i - size_i;
            if last < 0 {
                return Err(TetError::Validation(
                    "window produces zero output positions".into(),
                ));
            }
            (0_i64, last)
        }
        (_, WindowAlign::Start) => {
            // Left overhang: first start = 1 - size; last fully in-bounds start.
            let first = 1 - size_i;
            let last = extent_i - size_i;
            (first, last)
        }
        (_, WindowAlign::End) => {
            // Right overhang: first start = 0; last start = extent - 1.
            let last = extent_i - 1;
            (0_i64, last)
        }
    };
    if last < first {
        return Err(TetError::Validation(
            "window produces zero output positions".into(),
        ));
    }
    let span = last - first;
    let n = span / step_i + 1;
    let count = usize::try_from(n)
        .map_err(|_| TetError::Validation("window count overflows usize".into()))?;
    Ok((first, count))
}

/// Axis start index (may be negative) for window `w`.
#[must_use]
pub(crate) fn window_start(layout: &WindowAxisLayout, w: usize) -> i64 {
    match layout.mode {
        WindowMode::Bins => bins_window_start(layout, w),
        WindowMode::Rolling => {
            layout.first_start
                + i64::try_from(w).unwrap_or(0) * i64::try_from(layout.step).unwrap_or(1)
        }
    }
}

fn bins_window_start(layout: &WindowAxisLayout, w: usize) -> i64 {
    let w = u64::try_from(w).unwrap_or(0);
    match layout.align {
        WindowAlign::Start => i64::try_from(w.saturating_mul(layout.size)).unwrap_or(0),
        WindowAlign::End => {
            // Short remainder at the start: last window ends at `extent`.
            let rem = layout.extent % layout.size;
            let offset = if rem == 0 { 0 } else { rem };
            // Window 0 starts at 0 with length `offset` (or `size` when rem==0).
            // Subsequent windows are full `size` chunks.
            if w == 0 {
                0
            } else {
                let start = offset + (w - 1) * layout.size;
                i64::try_from(start).unwrap_or(0)
            }
        }
    }
}

/// In-bounds axis length for window `w` (before pad).
#[must_use]
pub(crate) fn window_real_axis_len(layout: &WindowAxisLayout, w: usize) -> u64 {
    let start = window_start(layout, w);
    let end_exclusive = start + i64::try_from(layout.size).unwrap_or(0);
    let lo = start.max(0);
    let hi = end_exclusive.min(i64::try_from(layout.extent).unwrap_or(0));
    if hi <= lo {
        0
    } else {
        u64::try_from(hi - lo).unwrap_or(0)
    }
}

/// Pad slots to inject along the window axis for window `w` (0 when pad is none).
#[must_use]
pub(crate) fn window_pad_axis_slots(layout: &WindowAxisLayout, w: usize) -> u64 {
    if layout.pad == WindowPad::None {
        return 0;
    }
    layout.size.saturating_sub(window_real_axis_len(layout, w))
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
            let w = bins_window_for_coord(layout, c)?;
            Ok(vec![w])
        }
        WindowMode::Rolling => {
            let c_i =
                i64::try_from(c).map_err(|_| TetError::Validation("axis coord overflow".into()))?;
            let size_i = i64::try_from(layout.size)
                .map_err(|_| TetError::Validation("window.size overflows i64".into()))?;
            // Cell c belongs to window w when start(w) <= c < start(w)+size.
            let mut out = Vec::new();
            for w in 0..layout.window_count {
                let start = window_start(layout, w);
                if start > c_i {
                    break;
                }
                if c_i < start + size_i {
                    out.push(w);
                }
            }
            Ok(out)
        }
    }
}

fn bins_window_for_coord(layout: &WindowAxisLayout, c: u64) -> Result<usize, TetError> {
    let w = match layout.align {
        WindowAlign::Start => c / layout.size,
        WindowAlign::End => {
            let rem = layout.extent % layout.size;
            if rem == 0 {
                c / layout.size
            } else if c < rem {
                0
            } else {
                1 + (c - rem) / layout.size
            }
        }
    };
    let w = usize::try_from(w).map_err(|_| TetError::Validation("window index overflow".into()))?;
    if w >= layout.window_count {
        return Err(TetError::Validation(format!(
            "bins window index {w} out of range ({})",
            layout.window_count
        )));
    }
    Ok(w)
}

/// Fiber index within a window for argmin/argmax (0..size-1, including pad slots).
pub(crate) fn window_fiber_index(
    axis_c: u64,
    window_idx: usize,
    layout: &WindowAxisLayout,
) -> Result<u64, TetError> {
    let start = window_start(layout, window_idx);
    let c_i =
        i64::try_from(axis_c).map_err(|_| TetError::Validation("axis coord overflow".into()))?;
    let fiber = c_i - start;
    if fiber < 0 {
        return Err(TetError::Validation(
            "internal: axis coord before window start".into(),
        ));
    }
    u64::try_from(fiber).map_err(|_| TetError::Validation("window fiber overflow".into()))
}
