use super::ReductionKind;
use crate::query::types::OperationPreviewFields;

/// Wire twin of [`ArgIndexAccum`].
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct ArgIndexAccumWire {
    pub have: bool,
    pub best_val: f64,
    pub best_idx: u64,
}

/// Index of min/max in logical row-major order (scalar) or within reduced axes (partial).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ArgIndexAccum {
    have: bool,
    best_val: f64,
    best_idx: u64,
}

impl ArgIndexAccum {
    pub fn is_empty(&self) -> bool {
        !self.have
    }

    pub fn push(&mut self, idx: u64, v: f32, kind: ReductionKind) {
        self.push_f64(idx, f64::from(v), kind);
    }

    pub fn push_f64(&mut self, idx: u64, v: f64, kind: ReductionKind) {
        if !self.have {
            self.best_val = v;
            self.best_idx = idx;
            self.have = true;
            return;
        }
        match kind {
            ReductionKind::ArgMin if v < self.best_val => {
                self.best_val = v;
                self.best_idx = idx;
            }
            ReductionKind::ArgMax if v > self.best_val => {
                self.best_val = v;
                self.best_idx = idx;
            }
            _ => {}
        }
    }

    pub fn index(&self) -> u64 {
        self.best_idx
    }

    /// Wire snapshot for distributed merge.
    #[must_use]
    pub fn to_wire(self) -> ArgIndexAccumWire {
        ArgIndexAccumWire {
            have: self.have,
            best_val: self.best_val,
            best_idx: self.best_idx,
        }
    }

    /// Restore from a wire snapshot.
    #[must_use]
    pub fn from_wire(w: &ArgIndexAccumWire) -> Self {
        Self {
            have: w.have,
            best_val: w.best_val,
            best_idx: w.best_idx,
        }
    }

    /// Combine partial argmin/argmax state from disjoint chunk visits.
    pub fn merge_from(&mut self, other: &Self, kind: ReductionKind) {
        if !other.have {
            return;
        }
        if !self.have {
            *self = *other;
            return;
        }
        match kind {
            ReductionKind::ArgMin if other.best_val < self.best_val => *self = *other,
            ReductionKind::ArgMax if other.best_val > self.best_val => *self = *other,
            _ => {}
        }
    }

    pub fn finish_scalar(self, kind: ReductionKind, element_count: usize) -> ScalarReductionResult {
        let idx = self.best_idx;
        let mut argmin_index = None;
        let mut argmax_index = None;
        match kind {
            ReductionKind::ArgMin => argmin_index = Some(idx),
            ReductionKind::ArgMax => argmax_index = Some(idx),
            _ => {}
        }
        ScalarReductionResult {
            element_count,
            argmin_index,
            argmax_index,
            ..ScalarReductionResult::default_fields(element_count)
        }
    }
}

/// Scalar reduction outputs (one field set per kind).
#[derive(Debug, Clone)]
pub(crate) struct ScalarReductionResult {
    pub element_count: usize,
    pub sum_scalar: Option<f64>,
    pub mean_scalar: Option<f64>,
    pub min_scalar: Option<f64>,
    pub max_scalar: Option<f64>,
    pub var_scalar: Option<f64>,
    pub std_scalar: Option<f64>,
    pub nan_mean_scalar: Option<f64>,
    pub nan_std_scalar: Option<f64>,
    pub product_scalar: Option<f64>,
    pub norm_l1_scalar: Option<f64>,
    pub norm_l2_scalar: Option<f64>,
    pub all_finite_scalar: Option<bool>,
    pub any_nan_scalar: Option<bool>,
    pub any_inf_scalar: Option<bool>,
    pub nan_count_scalar: Option<f64>,
    pub inf_count_scalar: Option<f64>,
    pub null_count_scalar: Option<f64>,
    pub argmin_index: Option<u64>,
    pub argmax_index: Option<u64>,
}

impl ScalarReductionResult {
    /// Combine partial results from disjoint chunk device (or host) reduces.
    #[allow(dead_code)]
    pub(crate) fn merge_partial(&mut self, part: &Self, kind: ReductionKind) {
        if part.element_count == 0 {
            return;
        }
        self.element_count += part.element_count;
        match kind {
            ReductionKind::Sum | ReductionKind::Mean => {
                let acc = self.sum_scalar.get_or_insert(0.0);
                *acc += part.sum_scalar.unwrap_or(0.0);
            }
            ReductionKind::Min => {
                let p = part.min_scalar.expect("partial min");
                match self.min_scalar {
                    Some(m) => self.min_scalar = Some(m.min(p)),
                    None => self.min_scalar = Some(p),
                }
            }
            ReductionKind::Max => {
                let p = part.max_scalar.expect("partial max");
                match self.max_scalar {
                    Some(m) => self.max_scalar = Some(m.max(p)),
                    None => self.max_scalar = Some(p),
                }
            }
            ReductionKind::Var | ReductionKind::Std => {
                unreachable!("var/std use ValueAccum streaming, not ScalarReductionResult merge")
            }
            _ => {}
        }
    }

    /// Set derived fields after all chunk partials are merged.
    #[allow(dead_code)]
    pub(crate) fn finalize_merged(self, kind: ReductionKind) -> Self {
        let mut out = self;
        match kind {
            ReductionKind::Mean if out.element_count > 0 => {
                let sum = out.sum_scalar.unwrap_or(0.0);
                out.mean_scalar = Some(sum / out.element_count as f64);
            }
            ReductionKind::Count => {
                out.sum_scalar = None;
                out.mean_scalar = None;
            }
            _ => {}
        }
        out
    }

    pub(crate) fn default_fields(element_count: usize) -> Self {
        Self {
            element_count,
            sum_scalar: None,
            mean_scalar: None,
            min_scalar: None,
            max_scalar: None,
            var_scalar: None,
            std_scalar: None,
            nan_mean_scalar: None,
            nan_std_scalar: None,
            product_scalar: None,
            norm_l1_scalar: None,
            norm_l2_scalar: None,
            all_finite_scalar: None,
            any_nan_scalar: None,
            any_inf_scalar: None,
            nan_count_scalar: None,
            inf_count_scalar: None,
            null_count_scalar: None,
            argmin_index: None,
            argmax_index: None,
        }
    }
}

impl From<ScalarReductionResult> for OperationPreviewFields {
    fn from(r: ScalarReductionResult) -> Self {
        Self {
            element_count: Some(r.element_count),
            sum: r.sum_scalar,
            mean: r.mean_scalar,
            min: r.min_scalar,
            max: r.max_scalar,
            var: r.var_scalar,
            std: r.std_scalar,
            nan_mean: r.nan_mean_scalar,
            nan_std: r.nan_std_scalar,
            product: r.product_scalar,
            norm_l1: r.norm_l1_scalar,
            norm_l2: r.norm_l2_scalar,
            all_finite: r.all_finite_scalar,
            any_nan: r.any_nan_scalar,
            any_inf: r.any_inf_scalar,
            nan_count: r.nan_count_scalar,
            inf_count: r.inf_count_scalar,
            null_count: r.null_count_scalar,
            argmin_index: r.argmin_index,
            argmax_index: r.argmax_index,
            ..Self::default()
        }
    }
}
