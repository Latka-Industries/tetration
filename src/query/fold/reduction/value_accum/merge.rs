use super::ValueAccum;

impl ValueAccum {
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
}
