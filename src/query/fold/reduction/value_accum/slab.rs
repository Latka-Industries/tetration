use super::super::ReductionKind;
use super::ValueAccum;
use crate::query::fold::variance_simd;

impl ValueAccum {
    /// Accumulate every little-endian `f32` in `raw` for a scalar fold (no per-element callbacks).
    pub fn push_f32_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 4, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[f32] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::f32_sum(vals);
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::f32_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::f32_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::Product => {
                self.count += vals.len();
                for &v in vals {
                    self.product *= f64::from(v);
                }
            }
            ReductionKind::NormL1 => {
                self.count += vals.len();
                for &v in vals {
                    self.norm_l1 += f64::from(v).abs();
                }
            }
            ReductionKind::NormL2 => {
                self.count += vals.len();
                for &v in vals {
                    let vd = f64::from(v);
                    self.norm_l2_sq += vd * vd;
                }
            }
            ReductionKind::AllFinite => self.push_all_finite_slice(vals, f32::is_finite),
            ReductionKind::AnyNan => self.push_any_nan_slice(vals, f32::is_nan),
            ReductionKind::AnyInf => self.push_any_inf_slice(vals, f32::is_infinite),
            ReductionKind::NanCount => {
                self.push_match_count_slice(vals.len(), vals.iter().filter(|v| v.is_nan()).count());
            }
            ReductionKind::InfCount => self.push_match_count_slice(
                vals.len(),
                vals.iter().filter(|v| v.is_infinite()).count(),
            ),
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::NanMean => {
                self.push_nan_mean_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::NanStd => {
                self.push_nan_std_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Like [`Self::push_f32_le_bytes`] but promotes each value to `f64` first.
    pub fn push_f64_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 8, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[f64] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::f64_sum(vals);
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::f64_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.push_min_max_f64_slice(vals);
            }
            ReductionKind::Product => {
                self.count += vals.len();
                for &v in vals {
                    self.product *= v;
                }
            }
            ReductionKind::NormL1 => {
                self.count += vals.len();
                for &v in vals {
                    self.norm_l1 += v.abs();
                }
            }
            ReductionKind::NormL2 => {
                self.count += vals.len();
                for &v in vals {
                    self.norm_l2_sq += v * v;
                }
            }
            ReductionKind::AllFinite => self.push_all_finite_slice(vals, f64::is_finite),
            ReductionKind::AnyNan => self.push_any_nan_slice(vals, f64::is_nan),
            ReductionKind::AnyInf => self.push_any_inf_slice(vals, f64::is_infinite),
            ReductionKind::NanCount => {
                self.push_match_count_slice(vals.len(), vals.iter().filter(|v| v.is_nan()).count());
            }
            ReductionKind::InfCount => self.push_match_count_slice(
                vals.len(),
                vals.iter().filter(|v| v.is_infinite()).count(),
            ),
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().copied(), fill);
            }
            ReductionKind::NanMean => {
                self.push_nan_mean_values(vals.iter().copied());
            }
            ReductionKind::NanStd => {
                self.push_nan_std_values(vals.iter().copied());
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `i32` LE slab fold (values promoted to `f64`).
    pub fn push_i32_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 4, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[i32] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::i32_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::i32_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::i32_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan
            | ReductionKind::AnyInf => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::NanMean => {
                self.push_nan_mean_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::NanStd => {
                self.push_nan_std_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `u8` LE slab fold (values promoted to `f64`).
    pub fn push_u8_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        if raw.is_empty() {
            return;
        }
        let vals = raw;
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::u8_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::u8_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::u8_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan
            | ReductionKind::AnyInf => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::NanMean => {
                self.push_nan_mean_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::NanStd => {
                self.push_nan_std_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `i64` LE slab fold (values promoted to `f64`).
    pub fn push_i64_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 8, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[i64] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::i64_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::i64_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::i64_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| v as f64), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan
            | ReductionKind::AnyInf => {
                for &v in vals {
                    self.push_f64(v as f64);
                }
            }
            ReductionKind::NanMean => {
                self.push_nan_mean_values(vals.iter().map(|&v| v as f64));
            }
            ReductionKind::NanStd => {
                self.push_nan_std_values(vals.iter().map(|&v| v as f64));
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `u32` LE slab fold (values promoted to `f64`).
    pub fn push_u32_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 4, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[u32] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::u32_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::u32_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::u32_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan
            | ReductionKind::AnyInf => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::NanMean => {
                self.push_nan_mean_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::NanStd => {
                self.push_nan_std_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `u64` LE slab fold (values promoted to `f64`).
    pub fn push_u64_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 8, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[u64] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::u64_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::u64_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::u64_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| v as f64), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan
            | ReductionKind::AnyInf => {
                for &v in vals {
                    self.push_f64(v as f64);
                }
            }
            ReductionKind::NanMean => {
                self.push_nan_mean_values(vals.iter().map(|&v| v as f64));
            }
            ReductionKind::NanStd => {
                self.push_nan_std_values(vals.iter().map(|&v| v as f64));
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `i16` LE slab fold (values promoted to `f64`).
    pub fn push_i16_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 2, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[i16] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::i16_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::i16_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::i16_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan
            | ReductionKind::AnyInf => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::NanMean => {
                self.push_nan_mean_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::NanStd => {
                self.push_nan_std_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `u16` LE slab fold (values promoted to `f64`).
    pub fn push_u16_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 2, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[u16] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count | ReductionKind::NanCount | ReductionKind::InfCount => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::u16_sum_sumsq(vals).0;
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::u16_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::u16_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|&v| f64::from(v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan
            | ReductionKind::AnyInf => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::NanMean => {
                self.push_nan_mean_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::NanStd => {
                self.push_nan_std_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }

    /// Bulk `f16` LE slab fold (values promoted to `f64`).
    pub fn push_f16_le_bytes(&mut self, raw: &[u8], kind: ReductionKind) {
        debug_assert_eq!(raw.len() % 2, 0);
        if raw.is_empty() {
            return;
        }
        let vals: &[half::f16] = bytemuck::cast_slice(raw);
        match kind {
            ReductionKind::Count => {
                self.count += vals.len();
            }
            ReductionKind::Sum | ReductionKind::Mean => {
                self.count += vals.len();
                self.sum += variance_simd::f16_sum(vals);
            }
            ReductionKind::Var | ReductionKind::Std => {
                self.count += vals.len();
                let (slab_sum, slab_sumsq) = variance_simd::f16_sum_sumsq(vals);
                self.sum += slab_sum;
                self.welford
                    .merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
            }
            ReductionKind::Min | ReductionKind::Max => {
                self.count += vals.len();
                let (slab_min, slab_max) = variance_simd::f16_min_max(vals);
                self.merge_slab_min_max(slab_min, slab_max);
            }
            ReductionKind::NanCount => self.push_match_count_slice(
                vals.len(),
                vals.iter().filter(|v| f64::from(**v).is_nan()).count(),
            ),
            ReductionKind::InfCount => self.push_match_count_slice(
                vals.len(),
                vals.iter().filter(|v| f64::from(**v).is_infinite()).count(),
            ),
            ReductionKind::NullCount { fill } => {
                self.push_null_count_values(vals.len(), vals.iter().map(|v| f64::from(*v)), fill);
            }
            ReductionKind::Product
            | ReductionKind::NormL1
            | ReductionKind::NormL2
            | ReductionKind::AllFinite
            | ReductionKind::AnyNan
            | ReductionKind::AnyInf => {
                for &v in vals {
                    self.push_f64(f64::from(v));
                }
            }
            ReductionKind::NanMean => {
                self.push_nan_mean_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::NanStd => {
                self.push_nan_std_values(vals.iter().map(|&v| f64::from(v)));
            }
            ReductionKind::ArgMin | ReductionKind::ArgMax => {
                unreachable!("argmin/argmax use ArgIndexAccum")
            }
        }
    }
}
