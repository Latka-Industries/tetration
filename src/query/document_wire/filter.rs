//! Parse `filter` predicates from the flat wire format.

use serde_json::Value;

use super::super::document::QueryLimits;
use super::super::types::{TetError, ValueCompareOp, ValueFilter, ValuePredicate};
use super::keys::{FILTER_COMPARE_KEYS, is_filter_key};

pub(super) fn parse_filter(v: &Value) -> Result<ValueFilter, TetError> {
    let obj = v
        .as_object()
        .ok_or_else(|| TetError::Validation("`filter` must be a JSON object".into()))?;
    if obj.is_empty() {
        return Err(TetError::Validation(
            "`filter` must include at least one predicate".into(),
        ));
    }
    if obj.len() > QueryLimits::DEFAULT.max_filter_predicates {
        return Err(TetError::Validation(format!(
            "`filter` has {} keys; maximum is {}",
            obj.len(),
            QueryLimits::DEFAULT.max_filter_predicates
        )));
    }

    let mut predicates = Vec::with_capacity(obj.len());
    // Stable parse order so AND lists are deterministic across JSON key order.
    for key in FILTER_COMPARE_KEYS {
        if let Some(val) = obj.get(*key) {
            let op = compare_op_from_key(key);
            let value = parse_filter_number(key, val)?;
            predicates.push(ValuePredicate::Compare { op, value });
        }
    }
    if let Some(val) = obj.get("is_null") {
        parse_filter_true_flag("is_null", val)?;
        predicates.push(ValuePredicate::IsNull);
    }
    if let Some(val) = obj.get("is_not_null") {
        parse_filter_true_flag("is_not_null", val)?;
        predicates.push(ValuePredicate::IsNotNull);
    }
    for key in obj.keys() {
        if !is_filter_key(key) {
            return Err(TetError::Validation(format!(
                "unknown field `filter.{key}` (expected eq, ne, lt, le, gt, ge, is_null, or is_not_null)"
            )));
        }
    }
    Ok(ValueFilter { predicates })
}

fn compare_op_from_key(key: &str) -> ValueCompareOp {
    match key {
        "eq" => ValueCompareOp::Eq,
        "ne" => ValueCompareOp::Ne,
        "lt" => ValueCompareOp::Lt,
        "le" => ValueCompareOp::Le,
        "gt" => ValueCompareOp::Gt,
        "ge" => ValueCompareOp::Ge,
        _ => unreachable!("compare_op_from_key: {key}"),
    }
}

fn parse_filter_number(key: &str, v: &Value) -> Result<f64, TetError> {
    let n = v
        .as_f64()
        .ok_or_else(|| TetError::Validation(format!("`filter.{key}` must be a finite number")))?;
    if !n.is_finite() {
        return Err(TetError::Validation(format!(
            "`filter.{key}` must be a finite number"
        )));
    }
    Ok(n)
}

fn parse_filter_true_flag(key: &str, v: &Value) -> Result<(), TetError> {
    match v.as_bool() {
        Some(true) => Ok(()),
        Some(false) => Err(TetError::Validation(format!(
            "`filter.{key}` must be true (omit the key or use the opposite predicate)"
        ))),
        None => Err(TetError::Validation(format!(
            "`filter.{key}` must be a boolean"
        ))),
    }
}
