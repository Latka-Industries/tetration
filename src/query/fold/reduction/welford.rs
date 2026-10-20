use crate::query::fold::variance_simd;

/// Online mean / variance (population, `ddof = 0`).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct WelfordAccum {
    count_f: f64,
    mean: f64,
    m2: f64,
}

impl WelfordAccum {
    #[must_use]
    pub(crate) fn from_parts(count_f: f64, mean: f64, m2: f64) -> Self {
        Self { count_f, mean, m2 }
    }

    #[must_use]
    pub(crate) fn count_f(&self) -> f64 {
        self.count_f
    }

    #[must_use]
    pub(crate) fn mean(&self) -> f64 {
        self.mean
    }

    #[must_use]
    pub(crate) fn m2(&self) -> f64 {
        self.m2
    }

    pub fn push(&mut self, x: f64) {
        self.count_f += 1.0;
        let delta = x - self.mean;
        self.mean += delta / self.count_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Population variance (`ddof = 0`); `0.0` for a single sample.
    #[must_use]
    pub fn population_variance(&self) -> f64 {
        if self.count_f <= 1.0 {
            0.0
        } else {
            self.m2 / self.count_f
        }
    }

    #[must_use]
    pub fn population_std(&self) -> f64 {
        self.population_variance().sqrt()
    }

    /// Merge another online accumulator (Chan parallel algorithm).
    pub fn merge_from(&mut self, other: &Self) {
        if other.count_f == 0.0 {
            return;
        }
        if self.count_f == 0.0 {
            *self = *other;
            return;
        }
        let n_a = self.count_f;
        let n_b = other.count_f;
        let n = n_a + n_b;
        let delta = other.mean - self.mean;
        self.mean += delta * (n_b / n);
        self.m2 += other.m2 + delta * delta * n_a * n_b / n;
        self.count_f = n;
    }

    /// Merge population variance stats from sum and sum-of-squares over `count` values.
    pub(crate) fn merge_sum_sumsq(&mut self, count: f64, sum: f64, sumsq: f64) {
        if count == 0.0 {
            return;
        }
        let mean = sum / count;
        let m2 = sumsq - count * mean * mean;
        self.merge_from(&Self {
            count_f: count,
            mean,
            m2,
        });
    }

    /// Merge population variance stats for a contiguous `f32` slice (slab / chunk bulk path).
    #[allow(dead_code)]
    pub fn merge_f32_slice(&mut self, vals: &[f32]) {
        if vals.is_empty() {
            return;
        }
        let (slab_sum, slab_sumsq) = variance_simd::f32_sum_sumsq(vals);
        self.merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
    }

    /// Like [`Self::merge_f32_slice`] for an `f64` slice.
    #[allow(dead_code)]
    pub fn merge_f64_slice(&mut self, vals: &[f64]) {
        if vals.is_empty() {
            return;
        }
        let (slab_sum, slab_sumsq) = variance_simd::f64_sum_sumsq(vals);
        self.merge_sum_sumsq(vals.len() as f64, slab_sum, slab_sumsq);
    }
}
