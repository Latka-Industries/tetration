//! Resolve operation axis labels (dimension names → decimal indices) using footer metadata.

use std::collections::BTreeMap;

use crate::catalog::DatasetMetadataV1;
use crate::query::types::{Operation, QueryDocument, TetError};

/// True when `label` is a non-negative decimal axis index (`"0"`, `"12"`, …).
#[must_use]
pub(crate) fn is_decimal_axis_label(label: &str) -> bool {
    !label.is_empty() && label.chars().all(|c| c.is_ascii_digit())
}

/// True when `label` is a dimension **name** token (Phase 9); resolved against `dim_names` at plan time.
#[must_use]
pub(crate) fn is_dimension_name_label(label: &str) -> bool {
    let Some(first) = label.chars().next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// Parse-time check: axis label is either a decimal index or a dimension name pending resolution.
pub(crate) fn validate_axis_label_token(label: &str) -> Result<(), TetError> {
    if is_decimal_axis_label(label) || is_dimension_name_label(label) {
        return Ok(());
    }
    Err(TetError::Validation(format!(
        "invalid axis label {label:?} (use a non-negative decimal index or a dimension name)"
    )))
}

/// Replace dimension names in [`QueryDocument::operation`] with decimal indices using footer `dim_names`.
///
/// # Errors
///
/// [`TetError::Validation`] when a name is unknown, `dim_names` is missing, or rank does not match.
pub(crate) fn resolve_query_document_axes(
    doc: &mut QueryDocument,
    dataset_meta: Option<&DatasetMetadataV1>,
    ndim: usize,
) -> Result<(), TetError> {
    let dim_names = dataset_meta.and_then(|m| m.dim_names.as_deref());
    if let Some(keys) = doc.group_by.as_mut() {
        for label in keys.iter_mut() {
            *label = resolve_one_axis_label(label, dim_names, ndim)?;
        }
    }
    let Some(op) = doc.operation.as_mut() else {
        return Ok(());
    };
    resolve_operation_axes(op, dim_names, ndim)?;
    if let Some(attrs) = dataset_meta.map(|m| &m.attrs) {
        resolve_null_count_fill(op, attrs)?;
    } else {
        resolve_null_count_fill(op, &BTreeMap::new())?;
    }
    apply_group_by_partition(doc, ndim)?;
    Ok(())
}

/// Reduction axes are the complement of `group_by` when omitted, and must partition rank otherwise.
fn apply_group_by_partition(doc: &mut QueryDocument, ndim: usize) -> Result<(), TetError> {
    let Some(keys) = doc.group_by.as_ref() else {
        return Ok(());
    };
    let Some(op) = doc.operation.as_mut() else {
        return Err(TetError::Validation(
            "`group_by` requires a reduction (for example `\"sum\": []`)".into(),
        ));
    };
    let group_idx = axis_index_set(keys)?;
    if op.axes().is_empty() {
        let reduce: Vec<String> = (0..ndim)
            .filter(|i| !group_idx.contains(i))
            .map(|i| i.to_string())
            .collect();
        if reduce.is_empty() {
            return Err(TetError::Validation(
                "`group_by` must leave at least one axis for the reduction".into(),
            ));
        }
        *op.axes_mut() = reduce;
        return Ok(());
    }
    let reduce_idx = axis_index_set(op.axes())?;
    if group_idx.intersection(&reduce_idx).next().is_some() {
        return Err(TetError::Validation(
            "`group_by` dimensions must be disjoint from the reduction axes".into(),
        ));
    }
    if group_idx.len() + reduce_idx.len() != ndim {
        return Err(TetError::Validation(
            "`group_by` and the reduction axes must together cover every dimension".into(),
        ));
    }
    Ok(())
}

fn axis_index_set(labels: &[String]) -> Result<std::collections::BTreeSet<usize>, TetError> {
    let mut set = std::collections::BTreeSet::new();
    for label in labels {
        let idx = label.parse::<usize>().map_err(|_| {
            TetError::Validation(format!(
                "invalid axis index {label:?} after name resolution"
            ))
        })?;
        if !set.insert(idx) {
            return Err(TetError::Validation(format!(
                "`group_by` resolves dimension `{label}` more than once"
            )));
        }
    }
    Ok(set)
}

/// Fill [`Operation::NullCount::fill`] from query JSON or dataset attrs (`_FillValue`, …).
pub(crate) fn resolve_null_count_fill(
    op: &mut Operation,
    attrs: &BTreeMap<String, String>,
) -> Result<(), TetError> {
    let Operation::NullCount { fill, .. } = op else {
        return Ok(());
    };
    if fill.is_some() {
        return Ok(());
    }
    for key in ["_FillValue", "missing_value", "fill_value"] {
        if let Some(raw) = attrs.get(key) {
            *fill = Some(parse_fill_attr(raw)?);
            return Ok(());
        }
    }
    Err(TetError::Validation(
        "null_count requires `fill` in the query or dataset attr (_FillValue, missing_value, fill_value)".into(),
    ))
}

/// Dataset missing-value from `_FillValue`, `missing_value`, or `fill_value`.
///
/// Invalid text is ignored (zone maps also skip a fill they cannot parse).
#[must_use]
pub(crate) fn fill_from_attrs(attrs: &BTreeMap<String, String>) -> Option<f64> {
    for key in ["_FillValue", "missing_value", "fill_value"] {
        if let Some(raw) = attrs.get(key) {
            return parse_fill_attr(raw).ok();
        }
    }
    None
}

fn parse_fill_attr(raw: &str) -> Result<f64, TetError> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("nan") {
        return Ok(f64::NAN);
    }
    trimmed
        .parse::<f64>()
        .map_err(|_| TetError::Validation(format!("invalid fill value {raw:?}")))
}

fn resolve_operation_axes(
    op: &mut Operation,
    dim_names: Option<&[String]>,
    ndim: usize,
) -> Result<(), TetError> {
    for label in op.axes_mut() {
        *label = resolve_one_axis_label(label, dim_names, ndim)?;
    }
    Ok(())
}

fn resolve_one_axis_label(
    label: &str,
    dim_names: Option<&[String]>,
    ndim: usize,
) -> Result<String, TetError> {
    if is_decimal_axis_label(label) {
        let v: usize = label
            .parse()
            .map_err(|_| TetError::Validation(format!("invalid axis index {label:?}")))?;
        if v >= ndim {
            return Err(TetError::Validation(format!(
                "operation axis index {v} out of range for rank {ndim}"
            )));
        }
        return Ok(label.to_owned());
    }
    if !is_dimension_name_label(label) {
        return Err(TetError::Validation(format!(
            "invalid axis label {label:?}"
        )));
    }
    let names = dim_names.ok_or_else(|| {
        TetError::Validation(format!(
            "dimension name `{label}` requires footer metadata `dim_names` for this dataset"
        ))
    })?;
    if names.len() != ndim {
        return Err(TetError::Validation(format!(
            "metadata dim_names length {} does not match dataset rank {ndim}",
            names.len()
        )));
    }
    let mut matches = names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.as_str() == label);
    let (idx, _) = matches.next().ok_or_else(|| {
        TetError::Validation(format!(
            "unknown dimension name `{label}` (dim_names: {})",
            names.join(", ")
        ))
    })?;
    if matches.next().is_some() {
        return Err(TetError::Validation(format!(
            "ambiguous dimension name `{label}` (duplicate dim_names entries)"
        )));
    }
    Ok(idx.to_string())
}
