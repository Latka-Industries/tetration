//! Flat-wire serialization for [`QueryDocument`](crate::query::types::QueryDocument).

use serde::ser::{Serialize, SerializeMap, Serializer};
use serde_json::Value;

use super::super::types::{
    AlignSpec, ExecutionHints, GroupBySpec, LabelFilterSpec, Operation, OutputHint, OutputHints,
    PipelineStep, QueryDocument, ValueFilter, ValuePredicate, WindowSpec, WriteHints, WriteTarget,
};

pub(super) struct QueryDocumentWire<'a> {
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
        if let Some(steps) = &doc.pipeline {
            serialize_pipeline(&mut map, steps)?;
        } else if doc.selection.is_some()
            || doc.filter.is_some()
            || doc.label_filter.is_some()
            || doc.group_by.is_some()
            || doc.window.is_some()
            || doc.align.is_some()
            || doc.operation.is_some()
            || doc.output.is_some()
            || doc.write.is_some()
        {
            serialize_flat_analytics(&mut map, doc)?;
        }
        if let Some(ex) = &doc.execution {
            serialize_execution(&mut map, ex)?;
        }
        map.end()
    }
}

fn serialize_flat_analytics<S>(map: &mut S, doc: &QueryDocument) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    if let Some(sel) = &doc.selection {
        map.serialize_entry("selection", sel)?;
    }
    if let Some(filter) = &doc.filter {
        serialize_filter(map, filter)?;
    }
    if let Some(label_filter) = &doc.label_filter {
        serialize_label_filter(map, label_filter)?;
    }
    if let Some(group_by) = &doc.group_by {
        serialize_group_by(map, group_by)?;
    }
    if let Some(window) = &doc.window {
        serialize_window(map, window)?;
    }
    if let Some(align) = &doc.align {
        serialize_align(map, align)?;
    }
    if let Some(op) = &doc.operation {
        serialize_operation(map, op)?;
    }
    if let Some(out) = &doc.output {
        serialize_output(map, out)?;
    }
    if let Some(w) = &doc.write {
        serialize_write(map, w)?;
    }
    Ok(())
}

fn serialize_pipeline<S>(map: &mut S, steps: &[PipelineStep]) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    let mut values = Vec::with_capacity(steps.len());
    for step in steps {
        values.push(pipeline_step_value(step).map_err(serde::ser::Error::custom)?);
    }
    map.serialize_entry("pipeline", &values)?;
    Ok(())
}

fn pipeline_step_value(step: &PipelineStep) -> Result<Value, serde_json::Error> {
    match step {
        PipelineStep::Select(selection) => Ok(serde_json::json!({ "select": selection })),
        step => {
            let mut scratch = QueryDocument::analytics_scratch();
            match step {
                PipelineStep::Select(_) => unreachable!("handled above"),
                PipelineStep::Filter(filter) => scratch.filter = Some(filter.clone()),
                PipelineStep::Reduce {
                    group_by,
                    window,
                    operation,
                } => {
                    scratch.group_by.clone_from(group_by);
                    scratch.window.clone_from(window);
                    scratch.operation = Some(operation.clone());
                }
                PipelineStep::Spill(path) => {
                    scratch.output = Some(OutputHints {
                        preferred: Some(OutputHint::SpillArray {
                            handle: path.clone(),
                        }),
                    });
                }
                PipelineStep::Write(write) => scratch.write = Some(write.clone()),
            }
            analytics_fragment_json(&scratch)
        }
    }
}

/// Serialize selection / filter / op / spill / write without dataset / execution / pipeline.
fn analytics_fragment_json(doc: &QueryDocument) -> Result<Value, serde_json::Error> {
    match serde_json::to_value(doc)? {
        Value::Object(mut obj) => {
            obj.remove("dataset");
            obj.remove("layout_version");
            obj.remove("execution");
            obj.remove("pipeline");
            Ok(Value::Object(obj))
        }
        other => Ok(other),
    }
}

fn serialize_window<S>(map: &mut S, window: &WindowSpec) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    let mut obj = serde_json::Map::new();
    if let Ok(idx) = window.axis.parse::<u64>() {
        obj.insert("axis".to_owned(), serde_json::json!(idx));
    } else {
        obj.insert(
            "axis".to_owned(),
            serde_json::Value::String(window.axis.clone()),
        );
    }
    obj.insert("size".to_owned(), serde_json::json!(window.size));
    obj.insert(
        "mode".to_owned(),
        serde_json::Value::String(window.mode.as_wire_str().to_owned()),
    );
    if window.pad != crate::query::types::WindowPad::None {
        obj.insert(
            "pad".to_owned(),
            serde_json::Value::String(window.pad.as_wire_str().to_owned()),
        );
    }
    if let Some(step) = window.step {
        obj.insert("step".to_owned(), serde_json::json!(step));
    }
    if window.align != crate::query::types::WindowAlign::Start {
        obj.insert(
            "align".to_owned(),
            serde_json::Value::String(window.align.as_wire_str().to_owned()),
        );
    }
    if window.min_count != 1 {
        obj.insert("min_count".to_owned(), serde_json::json!(window.min_count));
    }
    map.serialize_entry("window", &serde_json::Value::Object(obj))?;
    Ok(())
}

fn serialize_align<S>(map: &mut S, align: &AlignSpec) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    let mut obj = serde_json::Map::new();
    obj.insert(
        "with".to_owned(),
        serde_json::Value::String(align.with.clone()),
    );
    let axes: Vec<serde_json::Value> = align
        .axes
        .iter()
        .map(|a| {
            if let Ok(idx) = a.parse::<u64>() {
                serde_json::json!(idx)
            } else {
                serde_json::Value::String(a.clone())
            }
        })
        .collect();
    obj.insert("axes".to_owned(), serde_json::Value::Array(axes));
    map.serialize_entry("align", &serde_json::Value::Object(obj))?;
    Ok(())
}

fn serialize_group_by<S>(map: &mut S, group_by: &GroupBySpec) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    match group_by {
        GroupBySpec::Axes(keys) => {
            map.serialize_entry("group_by", keys)?;
        }
        GroupBySpec::Labels { axes } => {
            let mut obj = serde_json::Map::new();
            obj.insert(
                "by".to_owned(),
                serde_json::Value::String("labels".to_owned()),
            );
            let axes_v: Vec<serde_json::Value> = axes
                .iter()
                .map(|a| {
                    if let Ok(idx) = a.parse::<u64>() {
                        serde_json::json!(idx)
                    } else {
                        serde_json::Value::String(a.clone())
                    }
                })
                .collect();
            obj.insert("axes".to_owned(), serde_json::Value::Array(axes_v));
            map.serialize_entry("group_by", &serde_json::Value::Object(obj))?;
        }
    }
    Ok(())
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

fn serialize_label_filter<S>(map: &mut S, label_filter: &LabelFilterSpec) -> Result<(), S::Error>
where
    S: SerializeMap,
{
    let mut obj = serde_json::Map::new();
    obj.insert(
        "axis".to_owned(),
        serde_json::Value::String(label_filter.axis.clone()),
    );
    obj.insert(
        "eq".to_owned(),
        serde_json::Value::String(label_filter.eq.clone()),
    );
    map.serialize_entry("label_filter", &serde_json::Value::Object(obj))?;
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
