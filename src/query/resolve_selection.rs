//! Resolve `selection` coordinate labels to numeric slice bounds using footer metadata.

use crate::catalog::{CoordAxisV1, DatasetMetadataV1};
use crate::query::types::{AxisSlice, QueryDocument, TetError};

/// Replace `start_label` / `stop_label` on each selection axis with `start` / `stop` indices.
///
/// # Errors
///
/// [`TetError::Validation`] when labels are unknown, coords are missing, or bounds conflict.
pub(crate) fn resolve_query_document_selection(
    doc: &mut QueryDocument,
    dataset_meta: Option<&DatasetMetadataV1>,
    shape: &[u64],
) -> Result<(), TetError> {
    let Some(selection) = doc.selection.as_mut() else {
        return Ok(());
    };
    if !selection
        .iter()
        .any(|s| s.start_label.is_some() || s.stop_label.is_some())
    {
        return Ok(());
    }
    let meta = dataset_meta.ok_or_else(|| {
        TetError::Validation(
            "selection coordinate labels require footer metadata for this dataset".into(),
        )
    })?;
    for (d, sl) in selection.iter_mut().enumerate() {
        resolve_axis_slice_labels(d, sl, meta, shape[d])?;
    }
    Ok(())
}

fn resolve_axis_slice_labels(
    axis: usize,
    sl: &mut AxisSlice,
    meta: &DatasetMetadataV1,
    extent: u64,
) -> Result<(), TetError> {
    let has_start_label = sl.start_label.is_some();
    let has_stop_label = sl.stop_label.is_some();
    if !has_start_label && !has_stop_label {
        return Ok(());
    }
    if sl.start.is_some() && has_start_label {
        return Err(TetError::Validation(format!(
            "selection[{axis}]: use either `start` or `start_label`, not both"
        )));
    }
    if sl.stop.is_some() && has_stop_label {
        return Err(TetError::Validation(format!(
            "selection[{axis}]: use either `stop` or `stop_label`, not both"
        )));
    }
    let coord = coord_axis_for_dimension(meta, axis, extent).ok_or_else(|| {
        TetError::Validation(format!(
            "selection[{axis}] coordinate labels require `coords` for this axis in footer metadata"
        ))
    })?;
    if let Some(label) = sl.start_label.take() {
        let idx = label_to_index(&coord, &label)?;
        sl.start = Some(idx);
    }
    if let Some(label) = sl.stop_label.take() {
        let idx = label_to_index(&coord, &label)?;
        sl.stop = Some(idx);
    }
    let start = sl.start.unwrap_or(0);
    let stop = sl.stop.unwrap_or(extent);
    if start >= extent {
        return Err(TetError::Validation(format!(
            "selection[{axis}].start must be < shape[{axis}] ({extent}), got {start}"
        )));
    }
    if stop > extent {
        return Err(TetError::Validation(format!(
            "selection[{axis}].stop must be <= shape[{axis}] ({extent}), got {stop}"
        )));
    }
    if start >= stop {
        return Err(TetError::Validation(format!(
            "selection[{axis}]: require start < stop (got {start} >= {stop})"
        )));
    }
    Ok(())
}

fn coord_axis_for_dimension(
    meta: &DatasetMetadataV1,
    d: usize,
    extent: u64,
) -> Option<CoordAxisV1> {
    let coords = meta.coords.as_ref()?;
    if let Some(names) = &meta.dim_names
        && let Some(name) = names.get(d)
        && let Some(c) = coords.get(name)
    {
        return Some(c.clone());
    }
    if let Some(c) = coords.get(d.to_string().as_str()) {
        return Some(c.clone());
    }
    // HDF5 CF import: coords keyed by axis name without dim_names.
    let extent = usize::try_from(extent).ok()?;
    let mut matches = coords.values().filter(|c| c.labels.len() == extent);
    let first = matches.next()?;
    matches.next().is_none().then(|| first.clone())
}

fn label_to_index(coord: &CoordAxisV1, label: &str) -> Result<u64, TetError> {
    let pos = coord
        .labels
        .iter()
        .position(|l| l == label)
        .ok_or_else(|| {
            TetError::Validation(format!(
                "unknown coordinate label {label:?} (axis has {} labels)",
                coord.labels.len()
            ))
        })?;
    u64::try_from(pos).map_err(|_| {
        TetError::Validation("coordinate label index does not fit u64 on this host".into())
    })
}
