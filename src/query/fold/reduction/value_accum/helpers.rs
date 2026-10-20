use super::ValueAccum;

impl ValueAccum {
    pub(super) fn values_equal_fill(v: f64, fill: f64) -> bool {
        if fill.is_nan() {
            v.is_nan()
        } else {
            v.to_bits() == fill.to_bits()
        }
    }

    pub(super) fn merge_slab_min_max(&mut self, slab_min: f64, slab_max: f64) {
        if self.have_min_max {
            self.min = self.min.min(slab_min);
            self.max = self.max.max(slab_max);
        } else {
            self.min = slab_min;
            self.max = slab_max;
            self.have_min_max = true;
        }
    }

    pub(super) fn push_null_count_values(
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

    pub(super) fn push_match_count_slice(&mut self, len: usize, matches: usize) {
        self.count += len;
        self.match_count += matches;
    }

    pub(super) fn push_all_finite_slice<V>(&mut self, vals: &[V], is_finite: impl Fn(V) -> bool)
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

    pub(super) fn push_any_nan_slice<V>(&mut self, vals: &[V], is_nan: impl Fn(V) -> bool)
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

    pub(super) fn push_any_inf_slice<V>(&mut self, vals: &[V], is_infinite: impl Fn(V) -> bool)
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

    pub(super) fn push_min_max_f64_slice(&mut self, vals: &[f64]) {
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

    pub(super) fn push_nan_mean_values(&mut self, values: impl IntoIterator<Item = f64>) {
        for v in values {
            self.count += 1;
            if !v.is_nan() {
                self.finite_count += 1;
                self.sum += v;
            }
        }
    }

    pub(super) fn push_nan_std_values(&mut self, values: impl IntoIterator<Item = f64>) {
        for v in values {
            self.count += 1;
            if !v.is_nan() {
                self.finite_count += 1;
                self.welford.push(v);
            }
        }
    }
}
