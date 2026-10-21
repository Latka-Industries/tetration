//! Multi-dataset alignment checks (`align` wire key).
//!
//! Compares named axes between two catalog datasets in one sealed package: extents,
//! dimension names, and coordinate labels. Does not read cell payloads or produce a join.

use crate::catalog::{DatasetMetadataV1, DatasetRecordV1, TetFileSummaryV1};
use crate::query::resolve_axes::resolve_one_axis_label;
use crate::query::types::{AlignSpec, AlignmentAxisResult, AlignmentResult, TetError};

/// Compare `left` (`dataset`) against `align.with` on the listed axes.
///
/// # Errors
///
/// [`TetError::Validation`] when the right dataset is missing or an axis token cannot resolve.
pub(crate) fn check_alignment(
    summary: &TetFileSummaryV1,
    left_name: &str,
    left_rec: &DatasetRecordV1,
    left_meta: Option<&DatasetMetadataV1>,
    align: &AlignSpec,
) -> Result<AlignmentResult, TetError> {
    let right_name = align.with.as_str();
    let right_rec = summary
        .datasets
        .iter()
        .find(|d| d.name == right_name)
        .ok_or_else(|| {
            TetError::Validation(format!(
                "`align.with` dataset {right_name:?} not found in catalog"
            ))
        })?;
    let right_meta = summary.metadata.datasets.get(right_name);

    let left_dims = left_meta.and_then(|m| m.dim_names.as_deref());
    let right_dims = right_meta.and_then(|m| m.dim_names.as_deref());
    let left_ndim = left_rec.shape.len();
    let right_ndim = right_rec.shape.len();

    let mut axes = Vec::with_capacity(align.axes.len());
    let mut all_ok = true;
    for wire_axis in &align.axes {
        let left_idx_s = resolve_one_axis_label(wire_axis, left_dims, left_ndim)?;
        let right_idx_s = resolve_one_axis_label(wire_axis, right_dims, right_ndim)?;
        let left_idx: usize = left_idx_s
            .parse()
            .map_err(|_| TetError::Validation(format!("invalid axis index {left_idx_s:?}")))?;
        let right_idx: usize = right_idx_s
            .parse()
            .map_err(|_| TetError::Validation(format!("invalid axis index {right_idx_s:?}")))?;

        let extent_left = left_rec.shape[left_idx];
        let extent_right = right_rec.shape[right_idx];
        let mut ok = extent_left == extent_right;
        let mut reason: Option<String> = None;
        if !ok {
            reason = Some(format!("extent mismatch ({extent_left} vs {extent_right})"));
        }

        let name_left = left_dims.and_then(|n| n.get(left_idx)).cloned();
        let name_right = right_dims.and_then(|n| n.get(right_idx)).cloned();
        if left_dims.is_some() || right_dims.is_some() {
            match (&name_left, &name_right) {
                (Some(a), Some(b)) if a == b => {}
                (Some(a), Some(b)) => {
                    ok = false;
                    reason = Some(format!("dim_names mismatch (`{a}` vs `{b}`)"));
                }
                (None, Some(_)) | (Some(_), None) => {
                    ok = false;
                    reason = Some("dim_names present on only one dataset".into());
                }
                (None, None) => {}
            }
        }

        let coord_key_left = name_left
            .as_deref()
            .unwrap_or(left_idx_s.as_str())
            .to_owned();
        let coord_key_right = name_right
            .as_deref()
            .unwrap_or(right_idx_s.as_str())
            .to_owned();
        let labels_left = coord_labels(left_meta, &coord_key_left);
        let labels_right = coord_labels(right_meta, &coord_key_right);
        match (&labels_left, &labels_right) {
            (None, None) => {}
            (Some(a), Some(b)) if a == b => {}
            (Some(_), Some(_)) => {
                ok = false;
                reason = Some("coordinate labels mismatch".into());
            }
            (None, Some(_)) | (Some(_), None) => {
                ok = false;
                reason = Some("coordinate labels present on only one dataset".into());
            }
        }

        if !ok {
            all_ok = false;
        }
        axes.push(AlignmentAxisResult {
            axis: wire_axis.clone(),
            ok,
            extent_left,
            extent_right,
            dim_name_left: name_left,
            dim_name_right: name_right,
            labels_ok: labels_left == labels_right,
            reason,
        });
    }

    Ok(AlignmentResult {
        aligned: all_ok,
        left: left_name.to_owned(),
        with: right_name.to_owned(),
        axes,
    })
}

fn coord_labels<'a>(meta: Option<&'a DatasetMetadataV1>, key: &str) -> Option<&'a [String]> {
    meta?.coords.as_ref()?.get(key).map(|c| c.labels.as_slice())
}
