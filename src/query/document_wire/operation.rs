//! Parse reduction / parametric operation keys from the flat wire.

use serde_json::{Map, Value};

use super::super::document::QueryLimits;
use super::super::types::{Operation, TetError, TransformMethod};

pub(super) fn parse_operation(name: &str, v: &Value) -> Result<Operation, TetError> {
    match name {
        "quantile" => {
            let (axes, obj) = parse_parametric_op_object("quantile", v)?;
            let q = obj
                .get("q")
                .and_then(Value::as_f64)
                .ok_or_else(|| TetError::Validation("`quantile.q` is required".into()))?;
            Ok(Operation::Quantile { axes, q })
        }
        "histogram" => {
            let (axes, obj) = parse_parametric_op_object("histogram", v)?;
            let bins = obj
                .get("bins")
                .and_then(Value::as_u64)
                .and_then(|u| u32::try_from(u).ok())
                .ok_or_else(|| TetError::Validation("`histogram.bins` is required".into()))?;
            let min = obj.get("min").and_then(Value::as_f64);
            let max = obj.get("max").and_then(Value::as_f64);
            Ok(Operation::Histogram {
                axes,
                bins,
                min,
                max,
            })
        }
        "nan_count" => {
            let axes = parse_axis_spec(v)?;
            Ok(Operation::NanCount { axes })
        }
        "inf_count" => {
            let axes = parse_axis_spec(v)?;
            Ok(Operation::InfCount { axes })
        }
        "null_count" => {
            if v.is_object() {
                let (axes, obj) = parse_parametric_op_object("null_count", v)?;
                let fill = obj.get("fill").and_then(Value::as_f64);
                Ok(Operation::NullCount { axes, fill })
            } else {
                let axes = parse_axis_spec(v)?;
                Ok(Operation::NullCount { axes, fill: None })
            }
        }
        "covariance" => {
            let axes = parse_axis_spec(v)?;
            Ok(Operation::Covariance { axes })
        }
        "correlation" => {
            let axes = parse_axis_spec(v)?;
            Ok(Operation::Correlation { axes })
        }
        "transform" => {
            let (axes, obj) = parse_parametric_op_object("transform", v)?;
            let method = obj
                .get("method")
                .and_then(Value::as_str)
                .ok_or_else(|| TetError::Validation("`transform.method` is required".into()))?;
            let method = TransformMethod::parse(method)?;
            Ok(Operation::Transform { method, axes })
        }
        other => {
            let axes = parse_axis_spec(v)?;
            Ok(operation_from_axes(other, axes))
        }
    }
}

fn parse_parametric_op_object(
    op: &str,
    v: &Value,
) -> Result<(Vec<String>, Map<String, Value>), TetError> {
    let obj = v.as_object().ok_or_else(|| {
        TetError::Validation(format!(
            "`{op}` must be a JSON object (e.g. {{ \"q\": 0.95, \"axis\": 0 }})"
        ))
    })?;
    let axes = if let Some(axis) = obj.get("axis") {
        parse_axis_spec(axis)?
    } else if let Some(axes) = obj.get("axes") {
        parse_axis_spec(axes)?
    } else {
        Vec::new()
    };
    Ok((axes, obj.clone()))
}

fn operation_from_axes(name: &str, axes: Vec<String>) -> Operation {
    match name {
        "sum" => Operation::Sum { axes },
        "mean" => Operation::Mean { axes },
        "min" => Operation::Min { axes },
        "max" => Operation::Max { axes },
        "count" => Operation::Count { axes },
        "var" => Operation::Var { axes },
        "std" => Operation::Std { axes },
        "product" => Operation::Product { axes },
        "norm_l1" => Operation::NormL1 { axes },
        "norm_l2" => Operation::NormL2 { axes },
        "all_finite" => Operation::AllFinite { axes },
        "any_nan" => Operation::AnyNan { axes },
        "any_inf" => Operation::AnyInf { axes },
        "arg_min" => Operation::ArgMin { axes },
        "arg_max" => Operation::ArgMax { axes },
        "median" => Operation::Median { axes },
        "nan_mean" => Operation::NanMean { axes },
        "nan_std" => Operation::NanStd { axes },
        _ => unreachable!("operation_from_axes: {name}"),
    }
}

fn parse_axis_spec(v: &Value) -> Result<Vec<String>, TetError> {
    match v {
        Value::Null => Err(TetError::Validation("axis spec must not be null".into())),
        Value::Number(n) => {
            let idx = parse_axis_index_number(n)?;
            Ok(vec![idx])
        }
        Value::String(s) => Ok(vec![validate_axis_token(s)?]),
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(parse_axis_spec_item(item)?);
            }
            Ok(out)
        }
        Value::Object(map) => {
            if let Some(axis) = map.get("axis") {
                return parse_axis_spec(axis);
            }
            if let Some(axes) = map.get("axes") {
                return parse_axis_spec(axes);
            }
            Err(TetError::Validation(
                "axis object must include `axis` or `axes`".into(),
            ))
        }
        Value::Bool(_) => Err(TetError::Validation(
            "axis spec must be a number, string, or array".into(),
        )),
    }
}

pub(super) fn parse_axis_spec_item(v: &Value) -> Result<String, TetError> {
    match v {
        Value::Number(n) => parse_axis_index_number(n),
        Value::String(s) => validate_axis_token(s),
        _ => Err(TetError::Validation(
            "axis list entries must be numbers or strings".into(),
        )),
    }
}

fn parse_axis_index_number(n: &serde_json::Number) -> Result<String, TetError> {
    let u = n
        .as_u64()
        .ok_or_else(|| TetError::Validation("axis index must be a non-negative integer".into()))?;
    Ok(u.to_string())
}

fn validate_axis_token(s: &str) -> Result<String, TetError> {
    if s.is_empty() {
        return Err(TetError::Validation("axis index must not be empty".into()));
    }
    if s.len() > QueryLimits::DEFAULT.max_operation_axis_label_len {
        return Err(TetError::Validation(format!(
            "axis label exceeds maximum length ({})",
            QueryLimits::DEFAULT.max_operation_axis_label_len
        )));
    }
    super::super::resolve_axes::validate_axis_label_token(s)?;
    Ok(s.to_owned())
}
