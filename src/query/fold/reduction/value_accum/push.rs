use super::ValueAccum;

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
}
