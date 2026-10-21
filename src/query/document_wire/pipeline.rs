//! Pipeline step parsing and lowering to flat analytic fields.

use serde_json::{Map, Value};

use super::super::document::QueryLimits;
use super::super::types::{
    AxisSlice, Operation, OutputHint, OutputHints, PipelineStep, TetError, ValueFilter, WindowSpec,
    WriteHints,
};
use super::fields::{parse_group_by, parse_selection, parse_spill, parse_window, parse_write};
use super::filter::parse_filter;
use super::keys::OP_KEYS;
use super::operation::parse_operation;

pub(super) struct LoweredPipeline {
    pub(super) selection: Option<Vec<AxisSlice>>,
    pub(super) filter: Option<ValueFilter>,
    pub(super) group_by: Option<Vec<String>>,
    pub(super) window: Option<WindowSpec>,
    pub(super) operation: Option<Operation>,
    pub(super) output: Option<OutputHints>,
    pub(super) write: Option<WriteHints>,
}

pub(super) fn lower_pipeline(steps: &[PipelineStep]) -> LoweredPipeline {
    let mut lowered = LoweredPipeline {
        selection: None,
        filter: None,
        group_by: None,
        window: None,
        operation: None,
        output: None,
        write: None,
    };
    for step in steps {
        match step {
            PipelineStep::Select(selection) => lowered.selection = Some(selection.clone()),
            PipelineStep::Filter(filter) => lowered.filter = Some(filter.clone()),
            PipelineStep::Reduce {
                group_by,
                window,
                operation,
            } => {
                lowered.group_by.clone_from(group_by);
                lowered.window.clone_from(window);
                lowered.operation = Some(operation.clone());
            }
            PipelineStep::Spill(path) => {
                lowered.output = Some(OutputHints {
                    preferred: Some(OutputHint::SpillArray {
                        handle: path.clone(),
                    }),
                });
            }
            PipelineStep::Write(write) => lowered.write = Some(write.clone()),
        }
    }
    lowered
}

pub(super) fn parse_pipeline(v: &Value) -> Result<Vec<PipelineStep>, TetError> {
    let arr = v.as_array().ok_or_else(|| {
        TetError::Validation("`pipeline` must be an array of step objects".into())
    })?;
    if arr.is_empty() {
        return Err(TetError::Validation(
            "`pipeline` must include at least one step".into(),
        ));
    }
    if arr.len() > QueryLimits::DEFAULT.max_pipeline_steps {
        return Err(TetError::Validation(format!(
            "`pipeline` has {} steps; maximum is {}",
            arr.len(),
            QueryLimits::DEFAULT.max_pipeline_steps
        )));
    }
    let mut steps = Vec::with_capacity(arr.len());
    let mut last_rank: u8 = 0;
    let mut saw = [false; 4];
    for (i, step) in arr.iter().enumerate() {
        let parsed = parse_pipeline_step(i, step)?;
        let rank = pipeline_step_rank(&parsed);
        if saw[usize::from(rank)] || rank < last_rank {
            return Err(TetError::Validation(format!(
                "`pipeline[{i}]` is out of order or repeats a stage (select, filter, reduce, then spill or write)"
            )));
        }
        saw[usize::from(rank)] = true;
        last_rank = rank;
        steps.push(parsed);
    }
    Ok(steps)
}

fn pipeline_step_rank(step: &PipelineStep) -> u8 {
    match step {
        PipelineStep::Select(_) => 0,
        PipelineStep::Filter(_) => 1,
        PipelineStep::Reduce { .. } => 2,
        PipelineStep::Spill(_) | PipelineStep::Write(_) => 3,
    }
}

fn parse_pipeline_step(index: usize, v: &Value) -> Result<PipelineStep, TetError> {
    let obj = v.as_object().ok_or_else(|| {
        TetError::Validation(format!("`pipeline[{index}]` must be a JSON object"))
    })?;
    if obj.is_empty() {
        return Err(TetError::Validation(format!(
            "`pipeline[{index}]` must name a step"
        )));
    }
    if obj.contains_key("select") {
        require_step_keys(index, obj, &["select"])?;
        return Ok(PipelineStep::Select(parse_selection(&obj["select"])?));
    }
    if obj.contains_key("filter") {
        require_step_keys(index, obj, &["filter"])?;
        return Ok(PipelineStep::Filter(parse_filter(&obj["filter"])?));
    }
    if obj.contains_key("spill") {
        require_step_keys(index, obj, &["spill"])?;
        let hints = parse_spill(&obj["spill"])?;
        let Some(OutputHint::SpillArray { handle }) = hints.preferred else {
            return Err(TetError::Validation(format!(
                "`pipeline[{index}].spill` must be a path"
            )));
        };
        return Ok(PipelineStep::Spill(handle));
    }
    if obj.contains_key("write") {
        require_step_keys(index, obj, &["write"])?;
        return Ok(PipelineStep::Write(parse_write(&obj["write"])?));
    }
    if obj.contains_key("group_by")
        || obj.contains_key("window")
        || obj.keys().any(|key| OP_KEYS.contains(&key.as_str()))
    {
        let mut group_by = None;
        let mut window = None;
        let mut operation = None;
        for (key, val) in obj {
            if key == "group_by" {
                group_by = Some(parse_group_by(val)?);
            } else if key == "window" {
                window = Some(parse_window(val)?);
            } else if OP_KEYS.contains(&key.as_str()) {
                if operation.is_some() {
                    return Err(TetError::Validation(format!(
                        "`pipeline[{index}]` must include at most one reduction key"
                    )));
                }
                operation = Some(parse_operation(key, val)?);
            } else {
                return Err(TetError::Validation(format!(
                    "unknown pipeline step `{key}`"
                )));
            }
        }
        let Some(operation) = operation else {
            return Err(TetError::Validation(format!(
                "`pipeline[{index}]` group_by/window requires a reduction"
            )));
        };
        if group_by.is_some() && window.is_some() {
            return Err(TetError::Validation(format!(
                "`pipeline[{index}]` cannot combine `group_by` and `window`"
            )));
        }
        return Ok(PipelineStep::Reduce {
            group_by,
            window,
            operation,
        });
    }
    let key = obj.keys().next().unwrap_or(&String::new()).clone();
    Err(TetError::Validation(format!(
        "unknown pipeline step `{key}`"
    )))
}

fn require_step_keys(
    index: usize,
    obj: &Map<String, Value>,
    allowed: &[&str],
) -> Result<(), TetError> {
    for key in obj.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(TetError::Validation(format!(
                "unknown field `pipeline[{index}].{key}`"
            )));
        }
    }
    Ok(())
}
