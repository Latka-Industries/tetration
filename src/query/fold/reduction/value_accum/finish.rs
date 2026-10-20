use super::super::{ReductionKind, ScalarReductionResult};
use super::ValueAccum;

impl ValueAccum {
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn finish_f64(&self, kind: ReductionKind) -> f64 {
        match kind {
            ReductionKind::Sum => self.sum,
            ReductionKind::Mean => {
                if self.count == 0 {
                    0.0
                } else {
                    self.sum / self.count as f64
                }
            }
            ReductionKind::NanMean => {
                if self.finite_count == 0 {
                    f64::NAN
                } else {
                    self.sum / self.finite_count as f64
                }
            }
            ReductionKind::Min => self.min,
            ReductionKind::Max => self.max,
            ReductionKind::Count => self.count as f64,
            ReductionKind::Var => self.welford.population_variance(),
            ReductionKind::Std => self.welford.population_std(),
            ReductionKind::NanStd => {
                if self.finite_count == 0 {
                    f64::NAN
                } else {
                    self.welford.population_std()
                }
            }
            ReductionKind::Product => self.product,
            ReductionKind::NormL1 => self.norm_l1,
            ReductionKind::NormL2 => self.norm_l2_sq.sqrt(),
            ReductionKind::AllFinite => f64::from(u8::from(self.all_finite)),
            ReductionKind::AnyNan => f64::from(u8::from(self.any_nan)),
            ReductionKind::AnyInf => f64::from(u8::from(self.any_inf)),
            ReductionKind::NanCount | ReductionKind::InfCount | ReductionKind::NullCount { .. } => {
                self.match_count as f64
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    #[must_use]
    pub fn finish_bool(&self, kind: ReductionKind) -> bool {
        match kind {
            ReductionKind::AllFinite => self.all_finite,
            ReductionKind::AnyNan => self.any_nan,
            ReductionKind::AnyInf => self.any_inf,
            ReductionKind::NanCount | ReductionKind::InfCount | ReductionKind::NullCount { .. } => {
                self.match_count > 0
            }
            _ => self.finish_f64(kind) > 0.5,
        }
    }

    pub fn finish_scalar(self, kind: ReductionKind) -> ScalarReductionResult {
        let mut sum_scalar = None;
        let mut mean_scalar = None;
        let mut min_scalar = None;
        let mut max_scalar = None;
        let mut var_scalar = None;
        let mut std_scalar = None;
        let mut nan_mean_scalar = None;
        let mut nan_std_scalar = None;
        let mut product_scalar = None;
        let mut norm_l1_scalar = None;
        let mut norm_l2_scalar = None;
        let mut all_finite_scalar = None;
        let mut any_nan_scalar = None;
        let mut any_inf_scalar = None;
        let mut nan_count_scalar = None;
        let mut inf_count_scalar = None;
        let mut null_count_scalar = None;
        match kind {
            ReductionKind::Sum => sum_scalar = Some(self.sum),
            ReductionKind::Mean => {
                mean_scalar = Some(if self.count == 0 {
                    0.0
                } else {
                    self.sum / self.count as f64
                });
            }
            ReductionKind::Min => min_scalar = Some(self.min),
            ReductionKind::Max => max_scalar = Some(self.max),
            ReductionKind::Var => var_scalar = Some(self.welford.population_variance()),
            ReductionKind::Std => std_scalar = Some(self.welford.population_std()),
            ReductionKind::NanMean => nan_mean_scalar = Some(self.finish_f64(kind)),
            ReductionKind::NanStd => nan_std_scalar = Some(self.finish_f64(kind)),
            ReductionKind::Product => product_scalar = Some(self.product),
            ReductionKind::NormL1 => norm_l1_scalar = Some(self.norm_l1),
            ReductionKind::NormL2 => norm_l2_scalar = Some(self.norm_l2_sq.sqrt()),
            ReductionKind::AllFinite => all_finite_scalar = Some(self.all_finite),
            ReductionKind::AnyNan => any_nan_scalar = Some(self.any_nan),
            ReductionKind::AnyInf => any_inf_scalar = Some(self.any_inf),
            ReductionKind::NanCount => nan_count_scalar = Some(self.match_count as f64),
            ReductionKind::InfCount => inf_count_scalar = Some(self.match_count as f64),
            ReductionKind::NullCount { .. } => null_count_scalar = Some(self.match_count as f64),
            ReductionKind::Count | ReductionKind::ArgMin | ReductionKind::ArgMax => {}
        }
        ScalarReductionResult {
            element_count: self.count,
            sum_scalar,
            mean_scalar,
            min_scalar,
            max_scalar,
            var_scalar,
            std_scalar,
            nan_mean_scalar,
            nan_std_scalar,
            product_scalar,
            norm_l1_scalar,
            norm_l2_scalar,
            all_finite_scalar,
            any_nan_scalar,
            any_inf_scalar,
            nan_count_scalar,
            inf_count_scalar,
            null_count_scalar,
            argmin_index: None,
            argmax_index: None,
        }
    }
}
