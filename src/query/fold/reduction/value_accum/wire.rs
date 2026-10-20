//! Serializable snapshot of [`super::ValueAccum`] for distributed merge (THI-575).

use super::super::WelfordAccum;
use super::ValueAccum;

/// Wire twin of [`ValueAccum`] (sufficient stats for associative tier-A/B merge).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub struct ValueAccumWire {
    pub count: usize,
    pub finite_count: usize,
    pub sum: f64,
    pub welford_count_f: f64,
    pub welford_mean: f64,
    pub welford_m2: f64,
    pub product: f64,
    pub norm_l1: f64,
    pub norm_l2_sq: f64,
    pub all_finite: bool,
    pub any_nan: bool,
    pub any_inf: bool,
    pub min: f64,
    pub max: f64,
    pub have_min_max: bool,
    pub match_count: usize,
}

impl ValueAccum {
    /// Snapshot sufficient stats for wire / merge across workers.
    #[must_use]
    pub fn to_wire(&self) -> ValueAccumWire {
        ValueAccumWire {
            count: self.count,
            finite_count: self.finite_count,
            sum: self.sum,
            welford_count_f: self.welford.count_f(),
            welford_mean: self.welford.mean(),
            welford_m2: self.welford.m2(),
            product: self.product,
            norm_l1: self.norm_l1,
            norm_l2_sq: self.norm_l2_sq,
            all_finite: self.all_finite,
            any_nan: self.any_nan,
            any_inf: self.any_inf,
            min: self.min,
            max: self.max,
            have_min_max: self.have_min_max,
            match_count: self.match_count,
        }
    }

    /// Restore from a wire snapshot.
    #[must_use]
    pub fn from_wire(w: &ValueAccumWire) -> Self {
        Self {
            count: w.count,
            finite_count: w.finite_count,
            sum: w.sum,
            welford: WelfordAccum::from_parts(w.welford_count_f, w.welford_mean, w.welford_m2),
            product: w.product,
            norm_l1: w.norm_l1,
            norm_l2_sq: w.norm_l2_sq,
            all_finite: w.all_finite,
            any_nan: w.any_nan,
            any_inf: w.any_inf,
            min: w.min,
            max: w.max,
            have_min_max: w.have_min_max,
            match_count: w.match_count,
        }
    }
}
