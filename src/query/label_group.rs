//! Label-value `group_by` remap (collapse repeated footer coordinate labels).

use std::collections::HashMap;

use crate::catalog::{CoordAxisV1, DatasetMetadataV1};
use crate::query::types::{GroupBySpec, LabelGroupRemap, ReadPlan, TetError};

/// Build a kept-axis remap when `group_by` uses `{ "by": "labels", ... }`.
///
/// # Errors
///
/// Missing coords, length mismatch with the dataset axis, or selection that does not
/// line up with the logical selection shape — all fail closed (no index fall-through).
pub(crate) fn build_label_group_remap(
    group_by: &GroupBySpec,
    meta: Option<&DatasetMetadataV1>,
    plan: &ReadPlan,
) -> Result<Option<LabelGroupRemap>, TetError> {
    let GroupBySpec::Labels { axes } = group_by else {
        return Ok(None);
    };
    let meta = meta.ok_or_else(|| {
        TetError::Validation(
            "`group_by` with `by: labels` requires footer coordinate labels on the dataset".into(),
        )
    })?;

    let mut group_dims = Vec::with_capacity(axes.len());
    for key in axes {
        let idx: usize = key.parse().map_err(|_| {
            TetError::Validation(format!(
                "invalid group_by axis {key:?} after name resolution"
            ))
        })?;
        group_dims.push(idx);
    }
    group_dims.sort_unstable();

    let ndim = plan.logical_selection_shape.len();
    let mut maps = Vec::with_capacity(group_dims.len());
    let mut extents = Vec::with_capacity(group_dims.len());
    for &d in &group_dims {
        if d >= ndim {
            return Err(TetError::Validation(format!(
                "group_by axis {d} out of range for rank {ndim}"
            )));
        }
        let coord = coord_axis_for_dimension(meta, d).ok_or_else(|| {
            TetError::Validation(format!(
                "`group_by` with `by: labels` requires coordinate labels on axis {d}"
            ))
        })?;
        let full_extent = plan
            .dataset_shape
            .get(d)
            .copied()
            .ok_or_else(|| TetError::Validation(format!("dataset shape missing axis {d}")))?;
        let full_extent_usize = usize::try_from(full_extent)
            .map_err(|_| TetError::Validation("dataset axis extent does not fit usize".into()))?;
        if coord.labels.len() != full_extent_usize {
            return Err(TetError::Validation(format!(
                "coordinate labels on axis {d} have length {}, expected {full_extent_usize}",
                coord.labels.len()
            )));
        }
        let start = plan.selection_box_start[d];
        let stop_excl = plan.selection_box_stop_exclusive[d];
        let axis_step = plan.selection_step[d].max(1);
        let selected = selected_labels(&coord.labels, start, stop_excl, axis_step)?;
        let expected = usize::try_from(plan.logical_selection_shape[d])
            .map_err(|_| TetError::Validation("selection extent does not fit usize".into()))?;
        if selected.len() != expected {
            return Err(TetError::Validation(format!(
                "coordinate label selection on axis {d} has length {}, expected {expected}",
                selected.len()
            )));
        }
        let (map, n_groups) = index_to_group(&selected);
        maps.push(map);
        extents.push(
            u64::try_from(n_groups)
                .map_err(|_| TetError::Validation("group count does not fit u64".into()))?,
        );
    }
    Ok(Some(LabelGroupRemap { maps, extents }))
}

fn selected_labels(
    full: &[String],
    start: u64,
    stop_excl: u64,
    axis_step: u64,
) -> Result<Vec<String>, TetError> {
    if start > stop_excl {
        return Err(TetError::Validation(
            "selection start is after stop for label group_by".into(),
        ));
    }
    let mut out = Vec::new();
    let mut i = start;
    while i < stop_excl {
        let idx = usize::try_from(i)
            .map_err(|_| TetError::Validation("selection index does not fit usize".into()))?;
        let label = full.get(idx).ok_or_else(|| {
            TetError::Validation(format!(
                "selection index {i} out of range for coordinate labels (len {})",
                full.len()
            ))
        })?;
        out.push(label.clone());
        i = i.checked_add(axis_step).ok_or_else(|| {
            TetError::Validation("selection step overflow while sampling labels".into())
        })?;
    }
    Ok(out)
}

fn index_to_group(labels: &[String]) -> (Vec<usize>, usize) {
    let mut first_seen: HashMap<&str, usize> = HashMap::new();
    let mut map = Vec::with_capacity(labels.len());
    for lab in labels {
        let next_id = first_seen.len();
        let id = *first_seen.entry(lab.as_str()).or_insert(next_id);
        map.push(id);
    }
    (map, first_seen.len())
}

fn coord_axis_for_dimension(meta: &DatasetMetadataV1, d: usize) -> Option<&CoordAxisV1> {
    let coords = meta.coords.as_ref()?;
    if let Some(names) = &meta.dim_names
        && let Some(name) = names.get(d)
        && let Some(c) = coords.get(name)
    {
        return Some(c);
    }
    coords.get(d.to_string().as_str())
}
