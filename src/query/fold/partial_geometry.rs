//! Shared partial-axis reduction geometry (output shape, index mapping).

use std::collections::BTreeSet;

use crate::query::decode::indexing::{coords_from_linear_row_major, linear_rm_index};
use crate::query::plan::read_plan::shape_product_usize;
use crate::query::types::{ReadPlan, TetError};

#[derive(Debug, Clone)]
pub(crate) struct PartialAxisLayout {
    pub axis_indices: Vec<usize>,
    pub axis_set: BTreeSet<usize>,
    pub out_shape: Vec<u64>,
    pub out_len: usize,
    /// When set (label-value `group_by`), kept-axis coords remap through these maps.
    kept_remap: Option<Vec<Vec<usize>>>,
}

impl PartialAxisLayout {
    /// Map full-rank coords to a linear index in the reduced output.
    pub(crate) fn reduced_index(&self, coords: &[usize]) -> Result<usize, TetError> {
        let mut out_c = Vec::with_capacity(self.out_shape.len());
        let mut kept_i = 0usize;
        for (d, &c) in coords.iter().enumerate() {
            if self.axis_set.contains(&d) {
                continue;
            }
            let mapped = match &self.kept_remap {
                None => c,
                Some(maps) => {
                    let map = maps.get(kept_i).ok_or_else(|| {
                        TetError::Validation(
                            "internal: label group remap missing a kept axis".into(),
                        )
                    })?;
                    *map.get(c).ok_or_else(|| {
                        TetError::Validation(format!(
                            "internal: label group remap index {c} out of range (len {})",
                            map.len()
                        ))
                    })?
                }
            };
            out_c.push(mapped);
            kept_i += 1;
        }
        linear_rm_index(&out_c, &self.out_shape)
    }
}

pub(crate) fn parse_axis_indices(labels: &[String], ndim: usize) -> Result<Vec<usize>, TetError> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for label in labels {
        let v: usize = label
            .parse()
            .map_err(|_| TetError::Validation(format!("invalid operation axis index {label:?}")))?;
        if v >= ndim {
            return Err(TetError::Validation(format!(
                "operation axis index {v} out of range for rank {ndim}"
            )));
        }
        if !seen.insert(v) {
            return Err(TetError::Validation(format!(
                "duplicate operation axis index {v}"
            )));
        }
        out.push(v);
    }
    out.sort_unstable();
    Ok(out)
}

pub(crate) fn out_shape(shape: &[u64], axis_indices: &[usize]) -> Vec<u64> {
    let axis_set: BTreeSet<usize> = axis_indices.iter().copied().collect();
    shape
        .iter()
        .enumerate()
        .filter(|(d, _)| !axis_set.contains(d))
        .map(|(_, &e)| e)
        .collect()
}

pub(crate) fn partial_axis_layout(
    plan: &ReadPlan,
    axis_labels: &[String],
) -> Result<PartialAxisLayout, TetError> {
    let shape = &plan.logical_selection_shape;
    let ndim = shape.len();
    let axis_indices = parse_axis_indices(axis_labels, ndim)?;
    if axis_indices.is_empty() {
        return Err(TetError::Validation(
            "internal: partial layout requires non-empty axes".into(),
        ));
    }
    if axis_indices.len() == ndim {
        return Err(TetError::Validation(
            "operation axes list reduces every dimension; use \"axes\": [] for a scalar reduction"
                .into(),
        ));
    }
    let axis_set: BTreeSet<usize> = axis_indices.iter().copied().collect();
    let (out_shape, kept_remap) = match &plan.label_group_remap {
        None => (out_shape(shape, &axis_indices), None),
        Some(remap) => {
            let kept_dims: Vec<usize> = (0..ndim).filter(|d| !axis_set.contains(d)).collect();
            if remap.maps.len() != kept_dims.len() || remap.extents.len() != kept_dims.len() {
                return Err(TetError::Validation(format!(
                    "internal: label group remap has {} maps for {} kept axes",
                    remap.maps.len(),
                    kept_dims.len()
                )));
            }
            for (i, &d) in kept_dims.iter().enumerate() {
                let extent = usize::try_from(shape[d]).map_err(|_| {
                    TetError::Validation("selection extent does not fit usize".into())
                })?;
                if remap.maps[i].len() != extent {
                    return Err(TetError::Validation(format!(
                        "internal: label group remap for axis {d} has len {}, expected {extent}",
                        remap.maps[i].len()
                    )));
                }
            }
            (remap.extents.clone(), Some(remap.maps.clone()))
        }
    };
    let out_len = shape_product_usize(&out_shape)?;
    Ok(PartialAxisLayout {
        axis_indices,
        axis_set,
        out_shape,
        out_len,
        kept_remap,
    })
}

pub(crate) fn fiber_linear_index(
    coords: &[usize],
    axis_indices: &[usize],
    shape: &[u64],
) -> Result<usize, TetError> {
    let rshape: Vec<u64> = axis_indices.iter().map(|&d| shape[d]).collect();
    let rc: Vec<usize> = axis_indices.iter().map(|&d| coords[d]).collect();
    linear_rm_index(&rc, &rshape)
}

/// Map a logical row-major index to reduced output cell and fiber index (partial-axis fold).
pub(crate) fn reduced_cell_index(
    li: usize,
    shape: &[u64],
    layout: &PartialAxisLayout,
) -> Result<(usize, u64), TetError> {
    let coords = coords_from_linear_row_major(li, shape)?;
    let oi = layout.reduced_index(&coords)?;
    let fi = fiber_linear_index(&coords, &layout.axis_indices, shape)? as u64;
    Ok((oi, fi))
}

/// Reduced output cell index for partial-axis transforms, or `0` when `layout` is `None` (global stats).
pub(crate) fn reduced_cell_index_or_global(
    li: usize,
    shape: &[u64],
    layout: Option<&PartialAxisLayout>,
) -> Result<usize, TetError> {
    if let Some(layout) = layout {
        let (oi, _) = reduced_cell_index(li, shape, layout)?;
        Ok(oi)
    } else {
        Ok(0)
    }
}
