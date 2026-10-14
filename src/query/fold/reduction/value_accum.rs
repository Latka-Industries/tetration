use super::scalar::ScalarReductionResult;
use super::{ReductionKind, WelfordAccum};
use crate::query::fold::variance_simd;

/// Single-pass accumulator for one scalar or partial-reduction cell.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct ValueAccum {
    count: usize,
    finite_count: usize,
    sum: f64,
    welford: WelfordAccum,
    product: f64,
    norm_l1: f64,
    norm_l2_sq: f64,
    all_finite: bool,
    any_nan: bool,
    any_inf: bool,
    min: f64,
    max: f64,
    have_min_max: bool,
    /// NaN / inf / null-match tally for [`ReductionKind::NanCount`], [`ReductionKind::InfCount`], and [`ReductionKind::NullCount`].
    match_count: usize,
}

impl Default for ValueAccum {
    fn default() -> Self {
        Self {
            count: 0,
            finite_count: 0,
            sum: 0.0,
            welford: WelfordAccum::default(),
            product: 1.0,
            norm_l1: 0.0,
            norm_l2_sq: 0.0,
            all_finite: true,
            any_nan: false,
            any_inf: false,
            min: 0.0,
            max: 0.0,
            have_min_max: false,
            match_count: 0,
        }
    }
}

impl ValueAccum {
    fn values_equal_fill(v: f64, fill: f64) -> bool {
        if fill.is_nan() {
            v.is_nan()
        } else {
            v.to_bits() == fill.to_bits()
        }
    }

    fn merge_slab_min_max(&mut self, slab_min: f64, slab_max: f64) {
        if self.have_min_max {
            self.min = self.min.min(slab_min);
            self.max = self.max.max(slab_max);
        } else {
            self.min = slab_min;
            self.max = slab_max;
            self.have_min_max = true;
        }
    }

    fn push_null_count_values(
        &mut self,
        len: usize,
        values: impl IntoIterator<Item = f64>,
        fill: f64,
    ) {
        self.count += len;
        for v in values {
            if Self::values_equal_fill(v, fill) {
                self.match_count += 1;
            }
        }
    }

    fn push_match_count_slice(&mut self, len: usize, matches: usize) {
        self.count += len;
        self.match_count += matches;
    }

    pub(crate) fn push_nan_f64(&mut self, v: f64) {
        self.count += 1;
        if v.is_nan() {
            self.match_count += 1;
        }
    }

    pub(crate) fn push_null_f64(&mut self, v: f64, fill: f64) {
        self.count += 1;
        if Self::values_equal_fill(v, fill) {
            self.match_count += 1;
        }
    }

    pub(crate) fn push_inf_f64(&mut self, v: f64) {
        self.count += 1;
        if v.is_infinite() {
            self.match_count += 1;
        }
    }

    pub(crate) fn push_nan_mean_f64(&mut self, v: f64) {
        self.push_nan_mean_values(std::iter::once(v));
    }

    pub(crate) fn push_nan_std_f64(&mut self, v: f64) {
        self.push_nan_std_values(std::iter::once(v));
    }

    fn push_all_finite_slice<V>(&mut self, vals: &[V], is_finite: impl Fn(V) -> bool)
    where
        V: Copy,
    {
        self.count += vals.len();
        if !self.all_finite {
            return;
        }
        for &v in vals {
            if !is_finite(v) {
                self.all_finite = false;
                return;
            }
        }
    }

    fn push_any_nan_slice<V>(&mut self, vals: &[V], is_nan: impl Fn(V) -> bool)
    where
        V: Copy,
    {
        if self.any_nan {
            self.count += vals.len();
            return;
        }
        for &v in vals {
            self.count += 1;
            if is_nan(v) {
                self.any_nan = true;
                return;
            }
        }
    }

    fn push_any_inf_slice<V>(&mut self, vals: &[V], is_infinite: impl Fn(V) -> bool)
    where
        V: Copy,
    {
        if self.any_inf {
            self.count += vals.len();
            return;
        }
        for &v in vals {
            self.count += 1;
            if is_infinite(v) {
                self.any_inf = true;
                return;
            }
        }
    }

    fn push_min_max_f64_slice(&mut self, vals: &[f64]) {
        self.count += vals.len();
        for &v in vals {
            if self.have_min_max {
                self.min = self.min.min(v);
                self.max = self.max.max(v);
            } else {
                self.min = v;
                self.max = v;
                self.have_min_max = true;
            }
        }
    }

    fn push_nan_mean_values(&mut self, values: impl IntoIterator<Item = f64>) {
        for v in values {
            self.count += 1;
            if !v.is_nan() {
                self.finite_count += 1;
                self.sum += v;
            }
        }
    }

    fn push_nan_std_values(&mut self, values: impl IntoIterator<Item = f64>) {
        for v in values {
            self.count += 1;
            if !v.is_nan() {
                self.finite_count += 1;
                self.welford.push(v);
            }
        }
    }
}

impl ValueAccum {
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn push(&mut self, v: f32) {
        self.push_f64(f64::from(v));
    }

    pub fn push_f64(&mut self, v: f64) {
        self.count += 1;
        self.sum += v;
        self.welford.push(v);
        self.product *= v;
        self.norm_l1 += v.abs();
        self.norm_l2_sq += v * v;
        self.all_finite &= v.is_finite();
        self.any_nan |= v.is_nan();
        self.any_inf |= v.is_infinite();
        if self.have_min_max {
            self.min = self.min.min(v);
            self.max = self.max.max(v);
        } else {
            self.min = v;
            self.max = v;
            self.have_min_max = true;
        }
    }

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
                self.sum += variance_simd::f32_sum_sumsq(vals).0;
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
                self.sum += variance_simd::f64_sum_sumsq(vals).0;
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
                self.sum += variance_simd::f16_sum_sumsq(vals).0;
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

    /// Combine partial accumulators from disjoint chunk visits.
    pub fn merge_from(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        self.count += other.count;
        self.finite_count += other.finite_count;
        self.sum += other.sum;
        self.welford.merge_from(&other.welford);
        self.product *= other.product;
        self.norm_l1 += other.norm_l1;
        self.norm_l2_sq += other.norm_l2_sq;
        self.all_finite &= other.all_finite;
        self.any_nan |= other.any_nan;
        self.any_inf |= other.any_inf;
        self.match_count += other.match_count;
        if other.have_min_max {
            if self.have_min_max {
                self.min = self.min.min(other.min);
                self.max = self.max.max(other.max);
            } else {
                self.min = other.min;
                self.max = other.max;
                self.have_min_max = true;
            }
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
