//! Parse selection, `group_by`, window, spill, write, and execution fields.

use serde_json::Value;

use super::super::document::QueryLimits;
use super::super::types::{
    AlignSpec, AxisSlice, ExecutionDeviceHint, ExecutionHints, GroupBySpec, OutputHint,
    OutputHints, TetError, WindowAlign, WindowMode, WindowPad, WindowSpec, WriteHints, WriteTarget,
};
use super::operation::parse_axis_spec_item;

pub(super) fn parse_align(v: &Value) -> Result<AlignSpec, TetError> {
    let obj = v
        .as_object()
        .ok_or_else(|| TetError::Validation("`align` must be a JSON object".into()))?;
    for key in obj.keys() {
        if !matches!(key.as_str(), "with" | "axes") {
            return Err(TetError::Validation(format!("unknown field `align.{key}`")));
        }
    }
    let with = match obj.get("with") {
        None => {
            return Err(TetError::Validation(
                "`align.with` is required (second dataset name)".into(),
            ));
        }
        Some(v) => {
            let s = v
                .as_str()
                .ok_or_else(|| TetError::Validation("`align.with` must be a string".into()))?;
            if s.trim().is_empty() {
                return Err(TetError::Validation(
                    "`align.with` must be a non-empty string".into(),
                ));
            }
            if s.len() > QueryLimits::DEFAULT.max_dataset_name_len {
                return Err(TetError::Validation(format!(
                    "`align.with` name exceeds maximum length ({} > {})",
                    s.len(),
                    QueryLimits::DEFAULT.max_dataset_name_len
                )));
            }
            s.to_owned()
        }
    };
    let axes = match obj.get("axes") {
        None => {
            return Err(TetError::Validation(
                "`align.axes` is required (non-empty array of dimension names or indices)".into(),
            ));
        }
        Some(v) => {
            let arr = v.as_array().ok_or_else(|| {
                TetError::Validation(
                    "`align.axes` must be an array of dimension names or indices".into(),
                )
            })?;
            if arr.is_empty() {
                return Err(TetError::Validation(
                    "`align.axes` must list at least one axis".into(),
                ));
            }
            if arr.len() > crate::catalog::MAX_NDIM {
                return Err(TetError::Validation(format!(
                    "`align.axes` has {} entries; maximum is {}",
                    arr.len(),
                    crate::catalog::MAX_NDIM
                )));
            }
            arr.iter()
                .map(parse_axis_spec_item)
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    Ok(AlignSpec { with, axes })
}

pub(super) fn parse_group_by(v: &Value) -> Result<GroupBySpec, TetError> {
    if let Some(arr) = v.as_array() {
        return Ok(GroupBySpec::Axes(parse_group_by_axes(arr, "`group_by`")?));
    }
    let obj = v.as_object().ok_or_else(|| {
        TetError::Validation(
            "`group_by` must be an array of dimension names/indices or `{ \"by\": \"labels\", \"axes\": [...] }`"
                .into(),
        )
    })?;
    for key in obj.keys() {
        if !matches!(key.as_str(), "by" | "axes") {
            return Err(TetError::Validation(format!(
                "unknown field `group_by.{key}`"
            )));
        }
    }
    let by = match obj.get("by") {
        None => {
            return Err(TetError::Validation(
                "`group_by.by` is required (`labels`)".into(),
            ));
        }
        Some(v) => v
            .as_str()
            .ok_or_else(|| TetError::Validation("`group_by.by` must be a string".into()))?,
    };
    if by != "labels" {
        return Err(TetError::Validation(format!(
            "unknown group_by.by `{by}` (expected labels)"
        )));
    }
    let axes = match obj.get("axes") {
        None => {
            return Err(TetError::Validation(
                "`group_by.axes` is required when `by` is `labels`".into(),
            ));
        }
        Some(v) => {
            let arr = v.as_array().ok_or_else(|| {
                TetError::Validation(
                    "`group_by.axes` must be an array of dimension names or indices".into(),
                )
            })?;
            parse_group_by_axes(arr, "`group_by.axes`")?
        }
    };
    Ok(GroupBySpec::Labels { axes })
}

fn parse_group_by_axes(arr: &[Value], label: &str) -> Result<Vec<String>, TetError> {
    if arr.is_empty() {
        return Err(TetError::Validation(format!(
            "{label} must list at least one dimension"
        )));
    }
    if arr.len() > QueryLimits::DEFAULT.max_group_by_keys {
        return Err(TetError::Validation(format!(
            "{label} has {} keys; maximum is {}",
            arr.len(),
            QueryLimits::DEFAULT.max_group_by_keys
        )));
    }
    arr.iter().map(parse_axis_spec_item).collect()
}

pub(super) fn parse_window(v: &Value) -> Result<WindowSpec, TetError> {
    let obj = v
        .as_object()
        .ok_or_else(|| TetError::Validation("`window` must be a JSON object".into()))?;
    for key in obj.keys() {
        if !matches!(
            key.as_str(),
            "axis" | "size" | "mode" | "pad" | "step" | "align" | "min_count"
        ) {
            return Err(TetError::Validation(format!(
                "unknown field `window.{key}`"
            )));
        }
    }
    let axis = match obj.get("axis") {
        None => {
            return Err(TetError::Validation(
                "`window.axis` is required (dimension name or index)".into(),
            ));
        }
        Some(v) => parse_axis_spec_item(v)?,
    };
    let size = match obj.get("size") {
        None => {
            return Err(TetError::Validation(
                "`window.size` is required (positive integer)".into(),
            ));
        }
        Some(v) => parse_u64_field("window.size", v)?,
    };
    if size == 0 {
        return Err(TetError::Validation("`window.size` must be >= 1".into()));
    }
    if size > QueryLimits::DEFAULT.max_window_size {
        return Err(TetError::Validation(format!(
            "`window.size` exceeds maximum ({size} > {})",
            QueryLimits::DEFAULT.max_window_size
        )));
    }
    let mode = match obj.get("mode") {
        None => {
            return Err(TetError::Validation(
                "`window.mode` is required (`bins` or `rolling`)".into(),
            ));
        }
        Some(v) => {
            let s = v
                .as_str()
                .ok_or_else(|| TetError::Validation("`window.mode` must be a string".into()))?;
            WindowMode::parse(s)?
        }
    };
    let pad = match obj.get("pad") {
        None => WindowPad::None,
        Some(v) => {
            let s = v
                .as_str()
                .ok_or_else(|| TetError::Validation("`window.pad` must be a string".into()))?;
            WindowPad::parse(s)?
        }
    };
    let step = match obj.get("step") {
        None => None,
        Some(v) => {
            let s = parse_u64_field("window.step", v)?;
            if s == 0 {
                return Err(TetError::Validation("`window.step` must be >= 1".into()));
            }
            Some(s)
        }
    };
    if mode == WindowMode::Bins && step.is_some() {
        return Err(TetError::Validation(
            "`window.step` is only valid with mode \"rolling\"".into(),
        ));
    }
    let align = match obj.get("align") {
        None => WindowAlign::Start,
        Some(v) => {
            let s = v
                .as_str()
                .ok_or_else(|| TetError::Validation("`window.align` must be a string".into()))?;
            WindowAlign::parse(s)?
        }
    };
    if mode == WindowMode::Rolling && pad == WindowPad::None && align == WindowAlign::End {
        return Err(TetError::Validation(
            "`window.align` \"end\" with mode \"rolling\" requires pad other than \"none\"".into(),
        ));
    }
    let min_count = match obj.get("min_count") {
        None => 1,
        Some(v) => {
            let c = parse_u64_field("window.min_count", v)?;
            if c == 0 {
                return Err(TetError::Validation(
                    "`window.min_count` must be >= 1".into(),
                ));
            }
            if c > size {
                return Err(TetError::Validation(format!(
                    "`window.min_count` ({c}) exceeds window.size ({size})"
                )));
            }
            c
        }
    };
    Ok(WindowSpec {
        axis,
        size,
        mode,
        pad,
        step,
        align,
        min_count,
    })
}

pub(super) fn parse_spill(v: &Value) -> Result<OutputHints, TetError> {
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

pub(super) fn parse_write(v: &Value) -> Result<WriteHints, TetError> {
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

pub(super) fn parse_selection(v: &Value) -> Result<Vec<AxisSlice>, TetError> {
    let slices: Vec<AxisSlice> = serde_json::from_value(v.clone())?;
    Ok(slices)
}

pub(super) fn parse_u32_field(name: &str, v: &Value) -> Result<u32, TetError> {
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

pub(super) fn parse_execution(v: &Value) -> Result<ExecutionHints, TetError> {
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
