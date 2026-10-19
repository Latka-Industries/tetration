//! Parse and validate query documents (flat JSON wire; TOML front-end).
//!
//! Security boundaries and deployment guidance: see `docs/query_engine.md` (section
//! “JSON security (input and output)”).

use crate::catalog::MAX_NDIM;

use super::document_wire::parse_query_value;
use super::types::{
    AxisSlice, Operation, QueryDocument, TetError, ValueCompareOp, ValueFilter, ValuePredicate,
    WriteHints, WriteTarget,
};

/// How [`parse_query_text`] chooses a parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueryInputFormat {
    /// `.json` / `{…}` → JSON; `.toml` / otherwise → TOML.
    #[default]
    Auto,
    Json,
    Toml,
}

/// Input limits enforced by [`parse_query_json`] and [`validate_query`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryLimits {
    /// Maximum query JSON payload size accepted by [`parse_query_json`].
    pub max_json_bytes: usize,
    /// Maximum composite nesting depth (arrays/objects) in query JSON.
    pub max_json_depth: usize,
    /// Maximum length of the `dataset` name string (bytes).
    pub max_dataset_name_len: usize,
    /// Maximum per-axis operation label length (decimal indices are short).
    pub max_operation_axis_label_len: usize,
    /// Maximum number of atomic predicates in a top-level `filter` object.
    pub max_filter_predicates: usize,
    /// Maximum number of dimension keys in a top-level `group_by` array.
    pub max_group_by_keys: usize,
    /// Maximum number of steps in a top-level `pipeline` array.
    pub max_pipeline_steps: usize,
}

impl QueryLimits {
    /// Default limits used by the query engine.
    pub const DEFAULT: Self = Self {
        max_json_bytes: 1 << 20,
        max_json_depth: 64,
        max_dataset_name_len: 1024,
        max_operation_axis_label_len: 32,
        max_filter_predicates: 8,
        max_group_by_keys: 8,
        max_pipeline_steps: 8,
    };
}

/// Composite nesting depth of a parsed JSON value (objects/arrays only).
fn json_composite_depth(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Array(items) => {
            1 + items.iter().map(json_composite_depth).max().unwrap_or(0)
        }
        serde_json::Value::Object(map) => {
            1 + map.values().map(json_composite_depth).max().unwrap_or(0)
        }
        _ => 0,
    }
}

pub(crate) fn check_query_payload_size(text: &str) -> Result<(), TetError> {
    let limits = QueryLimits::DEFAULT;
    if text.len() > limits.max_json_bytes {
        return Err(TetError::Validation(format!(
            "query document exceeds maximum size ({} bytes, limit {})",
            text.len(),
            limits.max_json_bytes
        )));
    }
    Ok(())
}

pub(crate) fn check_query_value_depth(value: &serde_json::Value) -> Result<(), TetError> {
    let limits = QueryLimits::DEFAULT;
    let depth = json_composite_depth(value);
    if depth > limits.max_json_depth {
        return Err(TetError::Validation(format!(
            "query document nesting depth {depth} exceeds maximum {}",
            limits.max_json_depth
        )));
    }
    Ok(())
}

/// Infer JSON vs TOML from an optional file path and payload text.
#[must_use]
pub fn detect_query_input_format(path_hint: Option<&str>, text: &str) -> QueryInputFormat {
    if let Some(path) = path_hint {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        return match ext {
            "json" => QueryInputFormat::Json,
            "toml" => QueryInputFormat::Toml,
            _ => detect_query_input_format(None, text),
        };
    }
    let trimmed = text.trim_start();
    if trimmed.starts_with('{') {
        QueryInputFormat::Json
    } else {
        QueryInputFormat::Toml
    }
}

/// Parse a query document from JSON or TOML.
///
/// # Errors
///
/// Same as [`parse_query_json`] / [`super::document_toml::parse_query_toml`].
pub fn parse_query_text(text: &str, format: QueryInputFormat) -> Result<QueryDocument, TetError> {
    match format {
        QueryInputFormat::Json => parse_query_json(text),
        QueryInputFormat::Toml => super::document_toml::parse_query_toml(text),
        QueryInputFormat::Auto => {
            if text.trim_start().starts_with('{') {
                parse_query_json(text)
            } else {
                super::document_toml::parse_query_toml(text)
            }
        }
    }
}

/// Parse a JSON query document from `text`.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when `text` exceeds [`QueryLimits::DEFAULT`].
/// Returns [`TetError::InvalidJson`] when `text` is not valid JSON or does not deserialize into a
/// [`QueryDocument`] (including unknown object keys).
pub fn parse_query_json(text: &str) -> Result<QueryDocument, TetError> {
    check_query_payload_size(text)?;
    let value: serde_json::Value = serde_json::from_str(text)?;
    check_query_value_depth(&value)?;
    parse_query_value(&value)
}

/// JSON-only checks: `step != 0`, numeric bounds, and no mixed label/numeric endpoints.
pub(super) fn validate_axis_slice_json(i: usize, sl: &AxisSlice) -> Result<(), TetError> {
    if let Some(step) = sl.step
        && step == 0
    {
        return Err(TetError::Validation(format!(
            "selection[{i}].step must be >= 1, got 0"
        )));
    }
    if sl.start.is_some() && sl.start_label.is_some() {
        return Err(TetError::Validation(format!(
            "selection[{i}]: use either `start` or `start_label`, not both"
        )));
    }
    if sl.stop.is_some() && sl.stop_label.is_some() {
        return Err(TetError::Validation(format!(
            "selection[{i}]: use either `stop` or `stop_label`, not both"
        )));
    }
    match (sl.start, sl.stop) {
        (Some(a), Some(b)) if a >= b => Err(TetError::Validation(format!(
            "selection[{i}]: start must be < stop when both set (got {a} >= {b})"
        ))),
        _ => Ok(()),
    }
}

/// Validate a parsed query document.
///
/// # Errors
///
/// Returns [`TetError::Validation`] when required fields or slice semantics are invalid.
pub fn validate_query(doc: &QueryDocument) -> Result<(), TetError> {
    let limits = QueryLimits::DEFAULT;
    let name = doc.dataset.trim();
    if name.is_empty() {
        return Err(TetError::Validation(
            "`dataset` must be a non-empty string".into(),
        ));
    }
    if doc.dataset.len() > limits.max_dataset_name_len {
        return Err(TetError::Validation(format!(
            "`dataset` name exceeds maximum length ({} > {})",
            doc.dataset.len(),
            limits.max_dataset_name_len
        )));
    }
    if let Some(axes) = &doc.selection {
        if axes.len() > MAX_NDIM {
            return Err(TetError::Validation(format!(
                "selection rank {} exceeds maximum {MAX_NDIM}",
                axes.len()
            )));
        }
        for (i, sl) in axes.iter().enumerate() {
            validate_axis_slice_json(i, sl)?;
        }
    }
    if let Some(filter) = &doc.filter {
        validate_value_filter(filter, &limits)?;
    }
    if let Some(keys) = &doc.group_by {
        validate_group_by(keys, doc.operation.as_ref(), &limits)?;
    }
    if let Some(op) = &doc.operation {
        validate_operation_axes_v1(op)?;
        validate_operation_params(op)?;
        if op.requires_transform() {
            if doc.output.is_some() {
                return Err(TetError::Validation(
                    "transform operations use top-level `write`, not `spill`".into(),
                ));
            }
            if let Some(w) = &doc.write {
                validate_write_hints(w)?;
            }
        } else if doc.write.is_some() {
            return Err(TetError::Validation(
                "`write` is only valid with `transform`".into(),
            ));
        }
    } else if doc.write.is_some() {
        return Err(TetError::Validation(
            "`write` requires a transform operation".into(),
        ));
    }
    if doc.output.is_some() && doc.write.is_some() {
        return Err(TetError::Validation(
            "query document must not set both `spill` and `write`".into(),
        ));
    }
    Ok(())
}

/// Validate a parsed [`ValueFilter`] (non-empty, finite compare literals, closed ops).
pub(super) fn validate_value_filter(
    filter: &ValueFilter,
    limits: &QueryLimits,
) -> Result<(), TetError> {
    if filter.predicates.is_empty() {
        return Err(TetError::Validation(
            "`filter` must include at least one predicate".into(),
        ));
    }
    if filter.predicates.len() > limits.max_filter_predicates {
        return Err(TetError::Validation(format!(
            "`filter` has {} predicates; maximum is {}",
            filter.predicates.len(),
            limits.max_filter_predicates
        )));
    }
    let mut saw_is_null = false;
    let mut saw_is_not_null = false;
    let mut lower: Option<(ValueCompareOp, f64)> = None;
    let mut upper: Option<(ValueCompareOp, f64)> = None;
    for pred in &filter.predicates {
        match pred {
            ValuePredicate::Compare { op, value } => {
                if !value.is_finite() {
                    return Err(TetError::Validation(format!(
                        "`filter.{}` must be a finite number",
                        op.as_wire_str()
                    )));
                }
                match op {
                    ValueCompareOp::Gt | ValueCompareOp::Ge => {
                        if lower.is_some() {
                            return Err(TetError::Validation(
                                "`filter` accepts at most one lower bound (`gt` or `ge`)".into(),
                            ));
                        }
                        lower = Some((*op, *value));
                    }
                    ValueCompareOp::Lt | ValueCompareOp::Le => {
                        if upper.is_some() {
                            return Err(TetError::Validation(
                                "`filter` accepts at most one upper bound (`lt` or `le`)".into(),
                            ));
                        }
                        upper = Some((*op, *value));
                    }
                    ValueCompareOp::Eq | ValueCompareOp::Ne => {}
                }
            }
            ValuePredicate::IsNull => {
                if saw_is_null {
                    return Err(TetError::Validation(
                        "`filter.is_null` must appear at most once".into(),
                    ));
                }
                saw_is_null = true;
            }
            ValuePredicate::IsNotNull => {
                if saw_is_not_null {
                    return Err(TetError::Validation(
                        "`filter.is_not_null` must appear at most once".into(),
                    ));
                }
                saw_is_not_null = true;
            }
        }
    }
    if saw_is_null && saw_is_not_null {
        return Err(TetError::Validation(
            "`filter` cannot set both `is_null` and `is_not_null`".into(),
        ));
    }
    if saw_is_null
        && filter
            .predicates
            .iter()
            .any(|p| matches!(p, ValuePredicate::Compare { .. }))
    {
        return Err(TetError::Validation(
            "`filter.is_null` cannot be combined with numeric comparisons".into(),
        ));
    }
    if let (Some((lo_op, lo)), Some((hi_op, hi))) = (lower, upper) {
        let empty = match (lo_op, hi_op) {
            (ValueCompareOp::Gt, ValueCompareOp::Lt) | (ValueCompareOp::Gt, ValueCompareOp::Le) => {
                lo >= hi
            }
            (ValueCompareOp::Ge, ValueCompareOp::Lt) => lo >= hi,
            (ValueCompareOp::Ge, ValueCompareOp::Le) => lo > hi,
            _ => false,
        };
        if empty {
            return Err(TetError::Validation(format!(
                "`filter` range is empty (`{}`: {lo}, `{}`: {hi})",
                lo_op.as_wire_str(),
                hi_op.as_wire_str()
            )));
        }
    }
    Ok(())
}

fn validate_group_by(
    keys: &[String],
    op: Option<&Operation>,
    limits: &QueryLimits,
) -> Result<(), TetError> {
    if keys.is_empty() {
        return Err(TetError::Validation(
            "`group_by` must list at least one dimension".into(),
        ));
    }
    if keys.len() > limits.max_group_by_keys {
        return Err(TetError::Validation(format!(
            "`group_by` has {} keys; maximum is {}",
            keys.len(),
            limits.max_group_by_keys
        )));
    }
    let Some(op) = op else {
        return Err(TetError::Validation(
            "`group_by` requires a reduction (for example `\"sum\": []`)".into(),
        ));
    };
    if op.requires_transform()
        || matches!(
            op,
            Operation::Covariance { .. } | Operation::Correlation { .. }
        )
    {
        return Err(TetError::Validation(
            "`group_by` is not valid with transform, covariance, or correlation".into(),
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    for key in keys {
        validate_operation_axis_token(key)?;
        if !seen.insert(key.as_str()) {
            return Err(TetError::Validation(format!(
                "`group_by` lists dimension `{key}` more than once"
            )));
        }
    }
    Ok(())
}

fn validate_write_hints(w: &WriteHints) -> Result<(), TetError> {
    match w.target {
        WriteTarget::Switch | WriteTarget::Ram => {
            if w.path.is_some() {
                return Err(TetError::Validation(
                    "`write.path` is only valid with `spill` or `sidecar` targets".into(),
                ));
            }
            if w.timestamp.is_some() {
                return Err(TetError::Validation(
                    "`write.timestamp` is only valid with `sidecar` target".into(),
                ));
            }
        }
        WriteTarget::Spill => {
            if w.timestamp.is_some() {
                return Err(TetError::Validation(
                    "`write.timestamp` is only valid with `sidecar` target".into(),
                ));
            }
        }
        WriteTarget::Sidecar => {}
    }
    Ok(())
}

fn validate_operation_params(op: &Operation) -> Result<(), TetError> {
    match op {
        Operation::Quantile { q, .. } if !(0.0..=1.0).contains(q) => Err(TetError::Validation(
            format!("quantile q must be in [0.0, 1.0], got {q}"),
        )),
        Operation::Histogram { bins, .. } if *bins == 0 => {
            Err(TetError::Validation("histogram bins must be >= 1".into()))
        }
        Operation::Histogram { bins, .. } if *bins > 4096 => Err(TetError::Validation(format!(
            "histogram bins exceeds maximum 4096 (got {bins})"
        ))),
        Operation::Histogram { min, max, .. } => match (min, max) {
            (Some(a), Some(b)) if !a.is_finite() || !b.is_finite() => Err(TetError::Validation(
                "histogram min/max must be finite".into(),
            )),
            (Some(a), Some(b)) if a >= b => Err(TetError::Validation(format!(
                "histogram min must be < max (got {a} >= {b})"
            ))),
            (Some(_), None) | (None, Some(_)) => Err(TetError::Validation(
                "histogram requires both `min` and `max` when either is set".into(),
            )),
            _ => Ok(()),
        },
        Operation::Covariance { axes } | Operation::Correlation { axes } => {
            crate::query::materialize::covariance::require_single_observation_axis(axes)
        }
        _ => Ok(()),
    }
}

fn validate_operation_axis_token(s: &str) -> Result<(), TetError> {
    let max_label = QueryLimits::DEFAULT.max_operation_axis_label_len;
    if s.is_empty() {
        return Err(TetError::Validation(
            "operation axis label must not be empty".into(),
        ));
    }
    if s.len() > max_label {
        return Err(TetError::Validation(format!(
            "operation axis label exceeds maximum length ({} > {max_label})",
            s.len()
        )));
    }
    super::resolve_axes::validate_axis_label_token(s)
}

fn validate_operation_axes_v1(op: &Operation) -> Result<(), TetError> {
    let axes = op.axes();
    if axes.len() > MAX_NDIM {
        return Err(TetError::Validation(format!(
            "operation has {} axes; maximum is {MAX_NDIM}",
            axes.len()
        )));
    }
    for a in axes {
        validate_operation_axis_token(a)?;
    }
    Ok(())
}
