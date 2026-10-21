//! JSON query document wire types (`QueryDocument`, selection slices, operations).
//!
//! Parsed by [`crate::query::parse_query_json`] and validated by [`crate::query::validate_query`].
//! See `docs/query_engine.md` for the full schema.

use serde::{Deserialize, Serialize};

use super::error::TetError;

// `QueryDocument` JSON wire format: see `document_wire.rs` (flat op keys, `mean: 0`, `spill: "…"`, …).

/// Per-axis slice: `start` inclusive, `stop` exclusive, `step` ≥ 1 when present.
///
/// Coordinate labels (`start_label` / `stop_label`) resolve to indices at plan time when footer
/// `coords` exist (axis key = `dim_names[d]` or decimal `"d"`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AxisSlice {
    pub start: Option<u64>,
    pub stop: Option<u64>,
    pub step: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_label: Option<String>,
}

/// Reduction or aggregate over the logical selection.
///
/// Each variant carries `axes`: decimal dimension indices (`"0"`, `"1"`, …) to reduce along.
/// An empty `axes` list reduces all elements to a scalar. Tier-A/B ops stream over chunks;
/// tier-C ops (`median`, `quantile`, `histogram`) may materialize the full logical tensor.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Operation {
    Sum {
        axes: Vec<String>,
    },
    Mean {
        axes: Vec<String>,
    },
    Min {
        axes: Vec<String>,
    },
    Max {
        axes: Vec<String>,
    },
    Count {
        axes: Vec<String>,
    },
    Var {
        axes: Vec<String>,
    },
    Std {
        axes: Vec<String>,
    },
    Product {
        axes: Vec<String>,
    },
    NormL1 {
        axes: Vec<String>,
    },
    NormL2 {
        axes: Vec<String>,
    },
    AllFinite {
        axes: Vec<String>,
    },
    AnyNan {
        axes: Vec<String>,
    },
    /// True when any element is ±infinity (float/`f16`; integers contribute false).
    AnyInf {
        axes: Vec<String>,
    },
    ArgMin {
        axes: Vec<String>,
    },
    ArgMax {
        axes: Vec<String>,
    },
    Median {
        axes: Vec<String>,
    },
    Quantile {
        axes: Vec<String>,
        q: f64,
    },
    Histogram {
        axes: Vec<String>,
        bins: u32,
        /// When both set, bin edges span `[min, max]`; otherwise data min/max.
        min: Option<f64>,
        max: Option<f64>,
    },
    /// Count of NaN elements (float/`f16` wire tags; integers contribute 0).
    NanCount {
        axes: Vec<String>,
    },
    /// Count of ±infinity elements (float/`f16`; integers contribute 0).
    InfCount {
        axes: Vec<String>,
    },
    /// Count of elements equal to `fill` (resolved from query or dataset attrs at plan time).
    NullCount {
        axes: Vec<String>,
        fill: Option<f64>,
    },
    /// Population covariance matrix (variables × variables) with samples along `axes[0]`.
    Covariance {
        axes: Vec<String>,
    },
    /// Pearson correlation matrix with samples along `axes[0]`.
    Correlation {
        axes: Vec<String>,
    },
    /// Mean over finite elements (NaN-skipping), same axis semantics as [`Self::Mean`].
    NanMean {
        axes: Vec<String>,
    },
    /// Population std over finite elements (NaN-skipping), `ddof = 0`.
    NanStd {
        axes: Vec<String>,
    },
    /// Two-pass shape-preserving transform (`transform` wire key); see [`TransformMethod`].
    /// Empty `axes` applies one global stat set; non-empty `axes` fold per reduced cell.
    Transform {
        method: TransformMethod,
        axes: Vec<String>,
    },
}

use super::transform_method::TransformMethod;

macro_rules! operation_axes_match {
    ($op:expr) => {
        match $op {
            Operation::Sum { axes }
            | Operation::Mean { axes }
            | Operation::Min { axes }
            | Operation::Max { axes }
            | Operation::Count { axes }
            | Operation::Var { axes }
            | Operation::Std { axes }
            | Operation::Product { axes }
            | Operation::NormL1 { axes }
            | Operation::NormL2 { axes }
            | Operation::AllFinite { axes }
            | Operation::AnyNan { axes }
            | Operation::AnyInf { axes }
            | Operation::ArgMin { axes }
            | Operation::ArgMax { axes }
            | Operation::Median { axes }
            | Operation::Quantile { axes, .. }
            | Operation::Histogram { axes, .. }
            | Operation::NanCount { axes }
            | Operation::InfCount { axes }
            | Operation::NullCount { axes, .. }
            | Operation::Covariance { axes }
            | Operation::Correlation { axes }
            | Operation::NanMean { axes }
            | Operation::NanStd { axes }
            | Operation::Transform { axes, .. } => axes,
        }
    };
}

impl Operation {
    /// Per-axis decimal dimension indices for this operation.
    #[must_use]
    pub fn axes(&self) -> &[String] {
        operation_axes_match!(self)
    }

    /// Mutable axis list for plan-time name → index resolution.
    pub(crate) fn axes_mut(&mut self) -> &mut Vec<String> {
        operation_axes_match!(self)
    }

    /// Top-level JSON wire key (`"mean"`, `"arg_min"`, …).
    #[must_use]
    pub fn wire_key(&self) -> &'static str {
        match self {
            Self::Sum { .. } => "sum",
            Self::Mean { .. } => "mean",
            Self::Min { .. } => "min",
            Self::Max { .. } => "max",
            Self::Count { .. } => "count",
            Self::Var { .. } => "var",
            Self::Std { .. } => "std",
            Self::Product { .. } => "product",
            Self::NormL1 { .. } => "norm_l1",
            Self::NormL2 { .. } => "norm_l2",
            Self::AllFinite { .. } => "all_finite",
            Self::AnyNan { .. } => "any_nan",
            Self::AnyInf { .. } => "any_inf",
            Self::NanCount { .. } => "nan_count",
            Self::InfCount { .. } => "inf_count",
            Self::NullCount { .. } => "null_count",
            Self::ArgMin { .. } => "arg_min",
            Self::ArgMax { .. } => "arg_max",
            Self::Median { .. } => "median",
            Self::Quantile { .. } => "quantile",
            Self::Histogram { .. } => "histogram",
            Self::Covariance { .. } => "covariance",
            Self::Correlation { .. } => "correlation",
            Self::NanMean { .. } => "nan_mean",
            Self::NanStd { .. } => "nan_std",
            Self::Transform { .. } => "transform",
        }
    }

    /// Tier-C ops that require a full logical materialize (not streaming fold).
    #[must_use]
    pub fn requires_materialize(&self) -> bool {
        matches!(
            self,
            Self::Median { .. }
                | Self::Quantile { .. }
                | Self::Histogram { .. }
                | Self::Covariance { .. }
                | Self::Correlation { .. }
        )
    }

    /// Two-pass element-wise transforms (pass-1 fold stats, pass-2 rewrite).
    #[must_use]
    pub fn requires_transform(&self) -> bool {
        matches!(self, Self::Transform { .. })
    }

    /// Transform method when [`Self::requires_transform`] is true.
    #[must_use]
    pub fn transform_method(&self) -> Option<TransformMethod> {
        match self {
            Self::Transform { method, .. } => Some(*method),
            _ => None,
        }
    }
}

/// Where a transform operation should write its dense output tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WriteTarget {
    /// RAM when the selection fits the memory budget; otherwise a cache spill file.
    #[default]
    Switch,
    /// Always spill to `path` or an engine temp file under platform cache.
    Spill,
    /// Publish a `.tet` sidecar beside the source file (explicit only).
    Sidecar,
    /// Dense in-process buffer (errors when the selection exceeds the memory budget).
    Ram,
}

impl WriteTarget {
    /// Stable wire token for JSON `write` fields.
    #[must_use]
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Switch => "switch",
            Self::Spill => "spill",
            Self::Sidecar => "sidecar",
            Self::Ram => "ram",
        }
    }
}

/// Output routing for [`Operation::Transform`] (`write` wire key).
#[derive(Debug, Clone, Default)]
pub struct WriteHints {
    pub target: WriteTarget,
    /// Override spill/sidecar path (relative to `.tet` parent when relative).
    pub path: Option<String>,
    /// For [`WriteTarget::Sidecar`] auto filenames: append UTC timestamp (default **true**).
    pub timestamp: Option<bool>,
}

/// Caller preference for where large query results should land.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum OutputHint {
    /// Keep results inline in the JSON response.
    InlineJson,
    /// Spill a dense array to a caller-managed file identified by `handle`.
    SpillArray { handle: String },
}

/// Optional output routing hints on a query document.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct OutputHints {
    #[serde(default)]
    pub preferred: Option<OutputHint>,
}

/// Comparison operator for a numeric cell-value predicate (`filter` wire key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueCompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl ValueCompareOp {
    /// Stable wire key (`eq`, `gt`, …).
    #[must_use]
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
        }
    }
}

/// One atomic value predicate on cells of the queried dataset.
///
/// Closed set only — no arbitrary expressions or SQL. Zone-map pushdown (later)
/// uses these against per-chunk min/max/null stats.
#[derive(Debug, Clone, PartialEq)]
pub enum ValuePredicate {
    /// Compare every cell (and chunk bounds) to a finite numeric literal.
    Compare { op: ValueCompareOp, value: f64 },
    /// Match fill/NaN nulls (`ZONE.json` `null_count`).
    IsNull,
    /// Match non-null cells.
    IsNotNull,
}

/// Conjunction of [`ValuePredicate`]s (`filter` wire key). AND semantics.
///
/// Wire shape is a single object of closed keys, e.g. `{ "ge": 0, "lt": 40 }` or
/// `{ "is_null": true }`. Slice-by-label stays on `selection`; this filters **cell values**.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ValueFilter {
    pub predicates: Vec<ValuePredicate>,
}

/// Sliding or tumbling window along one named axis (`window` wire key).
///
/// A window is a reduction over a span of one axis — not a new dataset, join, or SQL `OVER`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowSpec {
    /// Dimension name or decimal index (resolved to a decimal index at plan time).
    pub axis: String,
    /// Positive window length in logical indices along [`Self::axis`].
    pub size: u64,
    /// How consecutive windows are placed along the axis.
    pub mode: WindowMode,
}

/// Window placement along the windowed axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowMode {
    /// Non-overlapping spans of `size` (last window may be shorter when extent is not divisible).
    Tumbling,
    /// Overlapping spans of `size` advancing by one index each step.
    Sliding,
}

impl WindowMode {
    /// Stable wire token (`tumbling` / `sliding`).
    #[must_use]
    pub const fn as_wire_str(self) -> &'static str {
        match self {
            Self::Tumbling => "tumbling",
            Self::Sliding => "sliding",
        }
    }

    /// Parse a closed mode token.
    ///
    /// # Errors
    ///
    /// [`TetError::Validation`] when the token is not `tumbling` or `sliding`.
    pub fn parse(token: &str) -> Result<Self, TetError> {
        match token {
            "tumbling" => Ok(Self::Tumbling),
            "sliding" => Ok(Self::Sliding),
            _ => Err(TetError::Validation(format!(
                "unknown window mode `{token}` (expected tumbling or sliding)"
            ))),
        }
    }
}

/// Top-level JSON query document accepted by the engine and `tet query`.
#[derive(Debug, Clone)]
pub struct QueryDocument {
    /// Optional layout version hint (currently informational).
    pub layout_version: Option<u32>,
    /// Dataset name as stored in the `.tet` catalog.
    pub dataset: String,
    /// Per-axis half-open slices; omitted means the full dataset extent on each axis.
    pub selection: Option<Vec<AxisSlice>>,
    /// Optional cell-value filter (`filter` wire key); see [`ValueFilter`].
    pub filter: Option<ValueFilter>,
    /// Dimensions that identify each group (`group_by` wire key).
    ///
    /// Names or decimal indices. The reduction runs inside each group (streaming fold for
    /// tier-A/B). Together with the reduction axes they must partition the dataset rank.
    pub group_by: Option<Vec<String>>,
    /// Optional windowed reduction along one axis (`window` wire key); see [`WindowSpec`].
    ///
    /// Mutually exclusive with [`Self::group_by`]. Empty reduction axes mean reduce only within
    /// each window along the windowed axis (other axes are kept).
    pub window: Option<WindowSpec>,
    /// Optional reduction over the logical selection.
    pub operation: Option<Operation>,
    /// Optional export spill (`spill` wire key); not used with transform ops.
    pub output: Option<OutputHints>,
    /// Transform output routing (`write` wire key); see [`WriteHints`] and [`TransformMethod`].
    pub write: Option<WriteHints>,
    /// Host-side memory budget overrides for execution.
    pub execution: Option<ExecutionHints>,
    /// Ordered steps when the document was written as a `pipeline`.
    ///
    /// Lowered into `selection`, `filter`, `group_by`, `window`, `operation`, `spill`, and `write`
    /// at parse time. Absent on flat documents.
    pub pipeline: Option<Vec<PipelineStep>>,
}

/// One closed stage in a query `pipeline`.
///
/// Order is fixed: `select`, then `filter`, then one reduction (`group_by` or `window` optional),
/// then `spill` or `write`. Repeats and unknown stages are rejected.
#[derive(Debug, Clone)]
pub enum PipelineStep {
    /// Axis slices (`select` wire key).
    Select(Vec<AxisSlice>),
    /// Cell-value filter (`filter` wire key).
    Filter(ValueFilter),
    /// One reduction, optionally partitioned by `group_by` or windowed along one axis.
    Reduce {
        group_by: Option<Vec<String>>,
        window: Option<WindowSpec>,
        operation: Operation,
    },
    /// Export spill path (`spill` wire key).
    Spill(String),
    /// Transform output routing (`write` wire key).
    Write(WriteHints),
}

/// Host-side execution limits (JSON query document).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ExecutionHints {
    /// Max anonymous RAM for a dense in-memory logical `f32` buffer; overrides `.tet` header and percent defaults.
    #[serde(default)]
    pub memory_budget_bytes: Option<u64>,
    /// Share of host available RAM in basis points (10000 = 100%); overrides `.tet` header percent when no fixed bytes.
    #[serde(default)]
    pub memory_budget_percent_bps: Option<u16>,
    /// When `Some(true)`, force parallel streaming fold; `Some(false)` force sequential; `None` = auto from RAM vs selection size.
    #[serde(default)]
    pub fold_parallel: Option<bool>,
    /// Device routing for tier-A/B reductions: `cpu`, `auto`, `cuda`, or `cuda:N` (Phase 10; wire via [`crate::query::document_wire`]).
    #[serde(default, skip)]
    pub device: Option<ExecutionDeviceHint>,
}

/// Parsed `execution.device` or CLI `--device` token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionDeviceHint {
    Cpu,
    Auto,
    Metal,
    Cuda(usize),
    CudaMulti,
    Rocm(usize),
    RocmMulti,
}

impl ExecutionDeviceHint {
    /// Parse `cpu`, `auto`, `metal`, `cuda`, or `cuda:N`.
    ///
    /// # Errors
    ///
    /// Returns [`TetError::Validation`] when the token is unknown or malformed.
    pub fn parse(token: &str) -> Result<Self, TetError> {
        let t = token.trim();
        if t.is_empty() {
            return Err(TetError::Validation(
                "device token must not be empty".into(),
            ));
        }
        if t.eq_ignore_ascii_case("cpu") {
            return Ok(Self::Cpu);
        }
        if t.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        if t.eq_ignore_ascii_case("metal") {
            return Ok(Self::Metal);
        }
        if t.eq_ignore_ascii_case("cuda") {
            return Ok(Self::Cuda(0));
        }
        if t.eq_ignore_ascii_case("cuda:multi") {
            return Ok(Self::CudaMulti);
        }
        if let Some(rest) = t.strip_prefix("cuda:") {
            let idx = rest.parse::<usize>().map_err(|_| {
                TetError::Validation(format!(
                    "invalid device `{token}` (expected cuda:N with non-negative N)"
                ))
            })?;
            return Ok(Self::Cuda(idx));
        }
        if t.eq_ignore_ascii_case("rocm") {
            return Ok(Self::Rocm(0));
        }
        if t.eq_ignore_ascii_case("rocm:multi") {
            return Ok(Self::RocmMulti);
        }
        if let Some(rest) = t.strip_prefix("rocm:") {
            let idx = rest.parse::<usize>().map_err(|_| {
                TetError::Validation(format!(
                    "invalid device `{token}` (expected rocm:N with non-negative N)"
                ))
            })?;
            return Ok(Self::Rocm(idx));
        }
        Err(TetError::Validation(format!(
            "unknown device `{token}` (expected cpu, auto, metal, cuda[:N| :multi], or rocm[:N| :multi])"
        )))
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Auto => "auto",
            Self::Metal => "metal",
            Self::Cuda(0) => "cuda:0",
            Self::CudaMulti => "cuda:multi",
            Self::Rocm(0) => "rocm:0",
            Self::RocmMulti => "rocm:multi",
            Self::Cuda(n) => {
                let _ = n;
                "cuda"
            }
            Self::Rocm(n) => {
                let _ = n;
                "rocm"
            }
        }
    }

    /// Wire/JSON token (includes index for `cuda:N`).
    #[must_use]
    pub fn to_token(self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::Auto => "auto".to_string(),
            Self::Metal => "metal".to_string(),
            Self::Cuda(0) => "cuda".to_string(),
            Self::Cuda(n) => format!("cuda:{n}"),
            Self::CudaMulti => "cuda:multi".to_string(),
            Self::Rocm(0) => "rocm".to_string(),
            Self::Rocm(n) => format!("rocm:{n}"),
            Self::RocmMulti => "rocm:multi".to_string(),
        }
    }
}
