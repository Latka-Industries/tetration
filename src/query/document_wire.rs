//! Flat JSON query document wire format (v1).
//!
//! Top-level reduction keys (`mean`, `sum`, …) replace nested `operation`. Export spill uses
//! top-level `"spill": "path"` (not nested `output`). Axis specs accept `0`, `[0, 1]`, or `[]`
//! (scalar). Parametric ops use `{ "q": …, "axis": … }`.

use serde::de::{self, Deserialize, Deserializer};
use serde::ser::{Serialize, SerializeMap, Serializer};
use serde_json::{Map, Value};

use super::document::QueryLimits;
use super::types::{
    AxisSlice, ExecutionDeviceHint, ExecutionHints, Operation, OutputHint, OutputHints,
    QueryDocument, TetError, TransformMethod, ValueCompareOp, ValueFilter, ValuePredicate,
    WriteHints, WriteTarget,
};

const OP_KEYS: &[&str] = &[
    "sum",
    "mean",
    "min",
    "max",
    "count",
    "var",
    "std",
    "product",
    "norm_l1",
    "norm_l2",
    "all_finite",
    "any_nan",
    "any_inf",
    "arg_min",
    "arg_max",
    "median",
    "quantile",
    "histogram",
    "nan_count",
    "inf_count",
    "null_count",
    "covariance",
    "correlation",
    "nan_mean",
    "nan_std",
    "transform",
];

const RESERVED_KEYS: &[&str] = &[
    "layout_version",
    "dataset",
    "selection",
    "filter",
    "group_by",
    "spill",
    "write",
    "execution",
];

const FILTER_COMPARE_KEYS: &[&str] = &["eq", "ne", "lt", "le", "gt", "ge"];

/// Parse a JSON value into a [`QueryDocument`].
///
/// # Errors
///
/// [`TetError::Validation`] or [`TetError::InvalidJson`].
pub fn parse_query_value(value_ref: &Value) -> Result<QueryDocument, TetError> {
    let obj = value_ref
        .as_object()
        .ok_or_else(|| TetError::Validation("query document must be a JSON object".into()))?;

    if obj.contains_key("operation") {
        return Err(TetError::Validation(
            "nested `operation` is not supported; use a top-level op key (e.g. `\"mean\": 0`)"
                .into(),
        ));
    }

    if obj.contains_key("output") {
        return Err(TetError::Validation(
            "nested `output` is not supported; use top-level `\"spill\": \"path\"`".into(),
        ));
    }

    for key in obj.keys() {
        if !is_allowed_key(key) {
            return Err(TetError::Validation(format!("unknown field `{key}`")));
        }
    }

    let dataset = obj
        .get("dataset")
        .and_then(Value::as_str)
        .ok_or_else(|| TetError::Validation("`dataset` must be a string".into()))?
        .to_owned();

    let layout_version = match obj.get("layout_version") {
        None => None,
        Some(v) => Some(parse_u32_field("layout_version", v)?),
    };

    let selection = match obj.get("selection") {
        None => None,
        Some(v) => Some(parse_selection(v)?),
    };

    let filter = match obj.get("filter") {
        None => None,
        Some(v) => Some(parse_filter(v)?),
    };

    let group_by = match obj.get("group_by") {
        None => None,
        Some(v) => Some(parse_group_by(v)?),
    };

    let output = match obj.get("spill") {
        None => None,
        Some(v) => Some(parse_spill(v)?),
    };

    let write = match obj.get("write") {
        None => None,
        Some(v) => Some(parse_write(v)?),
    };

    let execution = match obj.get("execution") {
        None => None,
        Some(v) => Some(parse_execution(v)?),
    };

    let mut operation = None;
    for key in OP_KEYS {
        if let Some(v) = obj.get(*key) {
            if operation.is_some() {
                return Err(TetError::Validation(
                    "query document must include at most one reduction key".into(),
                ));
            }
            operation = Some(parse_operation(key, v)?);
        }
    }

    Ok(QueryDocument {
        layout_version,
        dataset,
        selection,
        filter,
        group_by,
        operation,
        output,
        write,
        execution,
    })
}

fn is_allowed_key(key: &str) -> bool {
    RESERVED_KEYS.contains(&key) || OP_KEYS.contains(&key)
}

fn parse_group_by(v: &Value) -> Result<Vec<String>, TetError> {
    let arr = v.as_array().ok_or_else(|| {
        TetError::Validation("`group_by` must be an array of dimension names or indices".into())
    })?;
    if arr.is_empty() {
        return Err(TetError::Validation(
            "`group_by` must list at least one dimension".into(),
        ));
    }
    if arr.len() > QueryLimits::DEFAULT.max_group_by_keys {
        return Err(TetError::Validation(format!(
            "`group_by` has {} keys; maximum is {}",
            arr.len(),
            QueryLimits::DEFAULT.max_group_by_keys
        )));
    }
    arr.iter().map(parse_axis_spec_item).collect()
}

fn parse_spill(v: &Value) -> Result<OutputHints, TetError> {
    let handle = v.as_str().ok_or_else(|| {
        TetError::Validation(
            "`spill` must be a string path (relative to the `.tet` parent or absolute)".into(),
        )
    })?;
    if handle.is_empty() {
        return Err(TetError::Validation(
            "`spill` path must not be empty".into(),
        ));
    }
    if handle.len() > QueryLimits::DEFAULT.max_dataset_name_len {
        return Err(TetError::Validation(format!(
            "`spill` path exceeds maximum length ({} > {})",
            handle.len(),
            QueryLimits::DEFAULT.max_dataset_name_len
        )));
    }
    Ok(OutputHints {
        preferred: Some(OutputHint::SpillArray {
            handle: handle.to_owned(),
        }),
    })
}

fn parse_write(v: &Value) -> Result<WriteHints, TetError> {
    match v {
        Value::String(s) => Ok(WriteHints {
            target: parse_write_target_token(s)?,
            path: None,
            timestamp: None,
        }),
        Value::Object(obj) => {
            for key in obj.keys() {
                if !matches!(key.as_str(), "target" | "path" | "timestamp") {
                    return Err(TetError::Validation(format!(
                        "unknown field `write.{key}`"
                    )));
                }
            }
            let target = match obj.get("target") {
                None => WriteTarget::Switch,
                Some(v) => parse_write_target_value(v)?,
            };
            let path = match obj.get("path") {
                None => None,
                Some(v) => Some(parse_write_path(v)?),
            };
            let timestamp = match obj.get("timestamp") {
                None => None,
                Some(v) => Some(parse_write_timestamp(v)?),
            };
            Ok(WriteHints {
                target,
                path,
                timestamp,
            })
        }
        _ => Err(TetError::Validation(
            "`write` must be a target string (`switch`, `spill`, `sidecar`, `ram`) or an object with `target` / `path`".into(),
        )),
    }
}

fn parse_write_target_value(v: &Value) -> Result<WriteTarget, TetError> {
    let s = v
        .as_str()
        .ok_or_else(|| TetError::Validation("`write.target` must be a string".into()))?;
    parse_write_target_token(s)
}

fn parse_write_target_token(s: &str) -> Result<WriteTarget, TetError> {
    match s {
        "switch" => Ok(WriteTarget::Switch),
        "spill" => Ok(WriteTarget::Spill),
        "sidecar" => Ok(WriteTarget::Sidecar),
        "ram" => Ok(WriteTarget::Ram),
        _ => Err(TetError::Validation(format!(
            "unknown write target `{s}` (expected switch, spill, sidecar, or ram)"
        ))),
    }
}

fn parse_write_path(v: &Value) -> Result<String, TetError> {
    let handle = v
        .as_str()
        .ok_or_else(|| TetError::Validation("`write.path` must be a string".into()))?;
    if handle.is_empty() {
        return Err(TetError::Validation(
            "`write.path` must not be empty".into(),
        ));
    }
    if handle.len() > QueryLimits::DEFAULT.max_dataset_name_len {
        return Err(TetError::Validation(format!(
            "`write.path` exceeds maximum length ({} > {})",
            handle.len(),
            QueryLimits::DEFAULT.max_dataset_name_len
        )));
    }
    Ok(handle.to_owned())
}

fn parse_write_timestamp(v: &Value) -> Result<bool, TetError> {
    v.as_bool()
        .ok_or_else(|| TetError::Validation("`write.timestamp` must be a boolean".into()))
}

fn parse_selection(v: &Value) -> Result<Vec<AxisSlice>, TetError> {
    let slices: Vec<AxisSlice> = serde_json::from_value(v.clone())?;
    Ok(slices)
}

fn parse_filter(v: &Value) -> Result<ValueFilter, TetError> {
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

fn is_filter_key(key: &str) -> bool {
    FILTER_COMPARE_KEYS.contains(&key) || matches!(key, "is_null" | "is_not_null")
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

fn parse_u32_field(name: &str, v: &Value) -> Result<u32, TetError> {
    match v {
        Value::Number(n) => n
            .as_u64()
            .and_then(|u| u32::try_from(u).ok())
            .ok_or_else(|| {
                TetError::Validation(format!("`{name}` must be a non-negative integer"))
            }),
        _ => Err(TetError::Validation(format!("`{name}` must be a number"))),
    }
}

fn parse_execution(v: &Value) -> Result<ExecutionHints, TetError> {
    let obj = v
        .as_object()
        .ok_or_else(|| TetError::Validation("`execution` must be a JSON object".into()))?;

    for key in obj.keys() {
        if !matches!(
            key.as_str(),
            "memory_budget_bytes"
                | "memory_budget_percent"
                | "memory_budget_percent_bps"
                | "fold_parallel"
                | "device"
        ) {
            return Err(TetError::Validation(format!(
                "unknown field `execution.{key}`"
            )));
        }
    }

    let memory_budget_bytes = match obj.get("memory_budget_bytes") {
        None => None,
        Some(v) => Some(parse_u64_field("execution.memory_budget_bytes", v)?),
    };

    let mut memory_budget_percent_bps = None;
    if let Some(v) = obj.get("memory_budget_percent_bps") {
        memory_budget_percent_bps = Some(parse_percent_bps_field(
            "execution.memory_budget_percent_bps",
            v,
        )?);
    }
    if let Some(v) = obj.get("memory_budget_percent") {
        if memory_budget_percent_bps.is_some() {
            return Err(TetError::Validation(
                "use either `execution.memory_budget_percent` or `execution.memory_budget_percent_bps`, not both".into(),
            ));
        }
        memory_budget_percent_bps = Some(percent_to_bps(parse_percent_field(
            "execution.memory_budget_percent",
            v,
        )?)?);
    }

    let fold_parallel = match obj.get("fold_parallel") {
        None => None,
        Some(v) => Some(v.as_bool().ok_or_else(|| {
            TetError::Validation("`execution.fold_parallel` must be a boolean".into())
        })?),
    };

    let device = match obj.get("device") {
        None => None,
        Some(v) => {
            let s = v.as_str().ok_or_else(|| {
                TetError::Validation("`execution.device` must be a string".into())
            })?;
            Some(ExecutionDeviceHint::parse(s)?)
        }
    };

    Ok(ExecutionHints {
        memory_budget_bytes,
        memory_budget_percent_bps,
        fold_parallel,
        device,
    })
}

fn parse_u64_field(name: &str, v: &Value) -> Result<u64, TetError> {
    match v {
        Value::Number(n) => n.as_u64().ok_or_else(|| {
            TetError::Validation(format!("`{name}` must be a non-negative integer"))
        }),
        _ => Err(TetError::Validation(format!("`{name}` must be a number"))),
    }
}

fn parse_percent_field(name: &str, v: &Value) -> Result<f64, TetError> {
    let pct = match v {
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
    .ok_or_else(|| TetError::Validation(format!("`{name}` must be a number")))?;
    if !(0.0..=100.0).contains(&pct) {
        return Err(TetError::Validation(format!(
            "`{name}` must be in [0, 100], got {pct}"
        )));
    }
    Ok(pct)
}

fn parse_percent_bps_field(name: &str, v: &Value) -> Result<u16, TetError> {
    let bps = parse_u64_field(name, v)?;
    let bps =
        u16::try_from(bps).map_err(|_| TetError::Validation(format!("`{name}` exceeds u16")))?;
    if bps > 10_000 {
        return Err(TetError::Validation(format!(
            "`{name}` must be <= 10000 (100%)"
        )));
    }
    Ok(bps)
}

fn percent_to_bps(percent: f64) -> Result<u16, TetError> {
    let bps = (percent * 100.0).round();
    let bps = u16::try_from(bps as u64)
        .map_err(|_| TetError::Validation("memory_budget_percent out of range".into()))?;
    if bps > 10_000 {
        return Err(TetError::Validation(
            "memory_budget_percent must be <= 100".into(),
        ));
    }
    Ok(bps)
}

fn parse_operation(name: &str, v: &Value) -> Result<Operation, TetError> {
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

fn parse_axis_spec_item(v: &Value) -> Result<String, TetError> {
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
    super::resolve_axes::validate_axis_label_token(s)?;
    Ok(s.to_owned())
}

// --- Serialize (flat wire) ---

struct QueryDocumentWire<'a> {
    doc: &'a QueryDocument,
}

impl<'a> From<&'a QueryDocument> for QueryDocumentWire<'a> {
    fn from(doc: &'a QueryDocument) -> Self {
        Self { doc }
    }
}

impl Serialize for QueryDocumentWire<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let doc = self.doc;
        let mut map = serializer.serialize_map(None)?;
        if let Some(v) = doc.layout_version {
            map.serialize_entry("layout_version", &v)?;
        }
        map.serialize_entry("dataset", &doc.dataset)?;
        if let Some(sel) = &doc.selection {
            map.serialize_entry("selection", sel)?;
        }
        if let Some(filter) = &doc.filter {
            serialize_filter(&mut map, filter)?;
        }
        if let Some(keys) = &doc.group_by {
            map.serialize_entry("group_by", keys)?;
        }
        if let Some(op) = &doc.operation {
            serialize_operation(&mut map, op)?;
        }
        if let Some(out) = &doc.output {
            serialize_output(&mut map, out)?;
        }
        if let Some(w) = &doc.write {
            serialize_write(&mut map, w)?;
        }
        if let Some(ex) = &doc.execution {
            serialize_execution(&mut map, ex)?;
        }
        map.end()
    }
}

fn serialize_filter<S>(map: &mut S, filter: &ValueFilter) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    let mut obj = serde_json::Map::new();
    for pred in &filter.predicates {
        match pred {
            ValuePredicate::Compare { op, value } => {
                obj.insert(op.as_wire_str().to_owned(), serde_json::json!(value));
            }
            ValuePredicate::IsNull => {
                obj.insert("is_null".to_owned(), serde_json::Value::Bool(true));
            }
            ValuePredicate::IsNotNull => {
                obj.insert("is_not_null".to_owned(), serde_json::Value::Bool(true));
            }
        }
    }
    map.serialize_entry("filter", &serde_json::Value::Object(obj))?;
    Ok(())
}

fn serialize_output<S>(map: &mut S, out: &OutputHints) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    match &out.preferred {
        None | Some(OutputHint::InlineJson) => {}
        Some(OutputHint::SpillArray { handle }) => {
            map.serialize_entry("spill", handle)?;
        }
    }
    Ok(())
}

fn serialize_write<S>(map: &mut S, write: &WriteHints) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    if write.path.is_none() && write.timestamp.is_none() && write.target == WriteTarget::Switch {
        return Ok(());
    }
    if write.path.is_none() && write.timestamp.is_none() {
        map.serialize_entry("write", write.target.as_wire_str())?;
        return Ok(());
    }
    let mut obj = serde_json::Map::new();
    obj.insert(
        "target".to_owned(),
        serde_json::Value::String(write.target.as_wire_str().to_owned()),
    );
    if let Some(path) = &write.path {
        obj.insert("path".to_owned(), serde_json::Value::String(path.clone()));
    }
    if let Some(timestamp) = write.timestamp {
        obj.insert("timestamp".to_owned(), serde_json::Value::Bool(timestamp));
    }
    map.serialize_entry("write", &serde_json::Value::Object(obj))?;
    Ok(())
}

fn serialize_axes_op<S>(map: &mut S, key: &str, axes: &[String]) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    map.serialize_entry(key, &AxisSpecWire::from_axes(axes))
}

fn serialize_operation<S>(map: &mut S, op: &Operation) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    match op {
        Operation::Quantile { axes, q } => {
            let wire = ParametricOpWire {
                axes: AxisSpecWire::from_axes(axes),
                extra: vec![("q", serde_json::json!(q))],
            };
            map.serialize_entry("quantile", &wire)?;
        }
        Operation::Histogram {
            axes,
            bins,
            min,
            max,
        } => {
            let mut extra = vec![("bins", serde_json::json!(bins))];
            if let Some(v) = min {
                extra.push(("min", serde_json::json!(v)));
            }
            if let Some(v) = max {
                extra.push(("max", serde_json::json!(v)));
            }
            let wire = ParametricOpWire {
                axes: AxisSpecWire::from_axes(axes),
                extra,
            };
            map.serialize_entry("histogram", &wire)?;
        }
        Operation::NullCount { axes, fill } => {
            let mut extra = Vec::new();
            if let Some(v) = fill {
                extra.push(("fill", serde_json::json!(v)));
            }
            let wire = ParametricOpWire {
                axes: AxisSpecWire::from_axes(axes),
                extra,
            };
            map.serialize_entry("null_count", &wire)?;
        }
        Operation::Transform { method, axes } => {
            let wire = ParametricOpWire {
                axes: AxisSpecWire::from_axes(axes),
                extra: vec![("method", serde_json::json!(method.as_str()))],
            };
            map.serialize_entry("transform", &wire)?;
        }
        _ => serialize_axes_op(map, op.wire_key(), op.axes())?,
    }
    Ok(())
}

fn serialize_execution<S>(map: &mut S, ex: &ExecutionHints) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    let mut obj = serde_json::Map::new();
    if let Some(bytes) = ex.memory_budget_bytes {
        obj.insert("memory_budget_bytes".to_owned(), serde_json::json!(bytes));
    }
    if let Some(bps) = ex.memory_budget_percent_bps {
        let percent = f64::from(bps) / 100.0;
        obj.insert(
            "memory_budget_percent".to_owned(),
            serde_json::json!(percent),
        );
    }
    if let Some(par) = ex.fold_parallel {
        obj.insert("fold_parallel".to_owned(), serde_json::json!(par));
    }
    if let Some(dev) = ex.device {
        obj.insert("device".to_owned(), serde_json::json!(dev.to_token()));
    }
    if !obj.is_empty() {
        map.serialize_entry("execution", &obj)?;
    }
    Ok(())
}

struct AxisSpecWire {
    axes: Vec<String>,
}

impl AxisSpecWire {
    fn from_axes(axes: &[String]) -> Self {
        Self {
            axes: axes.to_vec(),
        }
    }
}

impl Serialize for AxisSpecWire {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.axes.is_empty() {
            return serializer.collect_seq(std::iter::empty::<u8>());
        }
        if self.axes.len() == 1
            && let Ok(idx) = self.axes[0].parse::<u64>()
        {
            return serializer.serialize_u64(idx);
        }
        let indices: Vec<u64> = self
            .axes
            .iter()
            .map(|s| s.parse::<u64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(serde::ser::Error::custom)?;
        serializer.collect_seq(indices.iter())
    }
}

struct ParametricOpWire {
    axes: AxisSpecWire,
    extra: Vec<(&'static str, Value)>,
}

impl Serialize for QueryDocument {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        QueryDocumentWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for QueryDocument {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        parse_query_value(&value).map_err(de::Error::custom)
    }
}

impl Serialize for ParametricOpWire {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        for (k, v) in &self.extra {
            map.serialize_entry(*k, v)?;
        }
        if !self.axes.axes.is_empty() {
            if self.axes.axes.len() == 1 {
                if let Ok(idx) = self.axes.axes[0].parse::<u64>() {
                    map.serialize_entry("axis", &idx)?;
                } else {
                    map.serialize_entry("axis", &self.axes.axes[0])?;
                }
            } else {
                map.serialize_entry("axes", &self.axes)?;
            }
        }
        map.end()
    }
}
