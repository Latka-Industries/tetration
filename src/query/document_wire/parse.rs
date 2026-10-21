//! Top-level flat JSON query document parse.

use serde_json::Value;

use super::super::types::{QueryDocument, TetError};
use super::fields::{
    parse_align, parse_execution, parse_group_by, parse_selection, parse_spill, parse_u32_field,
    parse_window, parse_write,
};
use super::filter::parse_filter;
use super::keys::{OP_KEYS, is_allowed_key, is_flat_analytic_key};
use super::operation::parse_operation;
use super::pipeline::{lower_pipeline, parse_pipeline};

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

    let execution = match obj.get("execution") {
        None => None,
        Some(v) => Some(parse_execution(v)?),
    };

    if obj.contains_key("pipeline") {
        for key in obj.keys() {
            if is_flat_analytic_key(key) {
                return Err(TetError::Validation(format!(
                    "`pipeline` cannot be combined with top-level `{key}`"
                )));
            }
        }
        let pipeline = parse_pipeline(obj.get("pipeline").ok_or_else(|| {
            TetError::Validation("`pipeline` key disappeared during parse".into())
        })?)?;
        let lowered = lower_pipeline(&pipeline);
        return Ok(QueryDocument {
            layout_version,
            dataset,
            selection: lowered.selection,
            filter: lowered.filter,
            group_by: lowered.group_by,
            window: lowered.window,
            align: None,
            operation: lowered.operation,
            output: lowered.output,
            write: lowered.write,
            execution,
            pipeline: Some(pipeline),
        });
    }

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

    let window = match obj.get("window") {
        None => None,
        Some(v) => Some(parse_window(v)?),
    };

    let align = match obj.get("align") {
        None => None,
        Some(v) => Some(parse_align(v)?),
    };

    let output = match obj.get("spill") {
        None => None,
        Some(v) => Some(parse_spill(v)?),
    };

    let write = match obj.get("write") {
        None => None,
        Some(v) => Some(parse_write(v)?),
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
        window,
        align,
        operation,
        output,
        write,
        execution,
        pipeline: None,
    })
}
