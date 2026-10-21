//! Div-by-zero and invalid-shift warnings recorded during transform pass 2.

/// Cap of logical indices listed in the execution response (total count is unbounded).
pub(crate) const MAX_LISTED_DIV_BY_ZERO_INDICES: usize = 256;

/// Warnings emitted while applying a transform (pass 2).
#[derive(Debug, Clone, Default)]
pub(crate) struct TransformWarnings {
    pub div_by_zero_indices: Vec<u64>,
    pub div_by_zero_count: u64,
}

impl TransformWarnings {
    /// Increment the total count and append `logical_index` when under [`MAX_LISTED_DIV_BY_ZERO_INDICES`].
    pub(crate) fn record_div_by_zero(&mut self, logical_index: u64) {
        self.div_by_zero_count += 1;
        if self.div_by_zero_indices.len() < MAX_LISTED_DIV_BY_ZERO_INDICES {
            self.div_by_zero_indices.push(logical_index);
        }
    }

    /// Merge per-chunk warning bags (parallel pass-2).
    pub(crate) fn merge_from(&mut self, other: &Self) {
        self.div_by_zero_count = self
            .div_by_zero_count
            .saturating_add(other.div_by_zero_count);
        for &idx in &other.div_by_zero_indices {
            if self.div_by_zero_indices.len() >= MAX_LISTED_DIV_BY_ZERO_INDICES {
                break;
            }
            self.div_by_zero_indices.push(idx);
        }
    }
}
