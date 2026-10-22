//! Execute-path cell predicate for tier-A/B folds (THI-563).
//!
//! Same closed `filter` as zone-map chunk skip. This layer decides which **decoded
//! cells** update fold state. It is not serialized on [`super::ReadPlan`].
//!
//! Numeric policy matches the fold accumulators:
//!
//! * Compares use the `f64` the fold would push (`f64::from` for `f32`/`f16`/`i32` and
//!   narrower integers, `as f64` for `i64`/`u64`). Integers above 2^53 therefore compare
//!   as that promoted value, the same way `sum` / `mean` see them.
//! * A cell is null when it is NaN or equal to the dataset fill (`_FillValue`,
//!   `missing_value`, or `fill_value`), using the same bit-equality as `null_count`.
//!   Floats without a fill still treat NaN as null, matching zone maps.
//! * Null cells fail numeric compares except `ne` (null ≠ a finite literal), matching
//!   zone-map `ne` pushdown. `is_null` / `is_not_null` use that same null test.
//! * Predicates are AND-ed. An empty predicate list keeps every cell.

use super::{ValueCompareOp, ValueFilter, ValuePredicate};

const MAX_SLOTS: usize = 8;

#[derive(Clone, Copy, Debug)]
enum CellSlot {
    Unused,
    Compare(ValueCompareOp, f64),
    IsNull,
    IsNotNull,
}

/// Copied closed filter plus the dataset fill used for null tests.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CellPredicate {
    slots: [CellSlot; MAX_SLOTS],
    len: u8,
    fill: f64,
    has_fill: bool,
}

impl CellPredicate {
    /// Build a predicate from a validated [`ValueFilter`].
    ///
    /// `fill` is the dataset missing-value, when one is present. More than eight
    /// predicates cannot occur after [`crate::query::QueryLimits`] validation; extras
    /// make the predicate match nothing rather than dropping a term.
    #[must_use]
    pub(crate) fn from_filter(filter: &ValueFilter, fill: Option<f64>) -> Self {
        let mut slots = [CellSlot::Unused; MAX_SLOTS];
        let mut len = 0u8;
        for pred in &filter.predicates {
            if usize::from(len) == MAX_SLOTS {
                return Self {
                    slots,
                    len: 0,
                    fill: fill.unwrap_or(0.0),
                    has_fill: false,
                }
                .reject_all();
            }
            slots[usize::from(len)] = match pred {
                ValuePredicate::Compare { op, value } => CellSlot::Compare(*op, *value),
                ValuePredicate::IsNull => CellSlot::IsNull,
                ValuePredicate::IsNotNull => CellSlot::IsNotNull,
            };
            len += 1;
        }
        Self {
            slots,
            len,
            fill: fill.unwrap_or(0.0),
            has_fill: fill.is_some(),
        }
    }

    fn reject_all(mut self) -> Self {
        self.slots[0] = CellSlot::Compare(ValueCompareOp::Eq, f64::NAN);
        self.len = 1;
        self
    }

    /// True when this cell should update fold state.
    #[must_use]
    pub(crate) fn matches(self, value: f64) -> bool {
        let null = self.is_null_cell(value);
        for slot in self.slots.iter().take(usize::from(self.len)) {
            let ok = match *slot {
                CellSlot::Unused => true,
                CellSlot::IsNull => null,
                CellSlot::IsNotNull => !null,
                CellSlot::Compare(op, rhs) => compare_cell(null, op, value, rhs),
            };
            if !ok {
                return false;
            }
        }
        true
    }

    /// Dataset fill used for integer masks, when the query resolved one.
    #[must_use]
    pub(crate) fn fill_f64(self) -> Option<f64> {
        self.has_fill.then_some(self.fill)
    }

    fn is_null_cell(self, value: f64) -> bool {
        if value.is_nan() {
            return true;
        }
        self.has_fill && values_equal_fill(value, self.fill)
    }
}

/// Error when an integer dense materialize has `filter` but no dataset fill.
pub(crate) const INTEGER_FILTER_FILL_MSG: &str = "integer materialize under `filter` requires a dataset fill (_FillValue, missing_value, or fill_value)";

/// Keep `value` when no cell filter is attached, or when it matches.
#[inline]
#[must_use]
pub(crate) fn fold_keeps_cell(pred: Option<CellPredicate>, value: f64) -> bool {
    pred.is_none_or(|p| p.matches(value))
}

/// Keep a logical selection element when cell filter and optional label-eq both match.
#[inline]
#[must_use]
pub(crate) fn fold_keeps_element(plan: &super::ReadPlan, logical_index: usize, value: f64) -> bool {
    if !fold_keeps_cell(plan.cell_filter, value) {
        return false;
    }
    fold_keeps_label(plan, logical_index)
}

/// Keep when no `label_eq` is attached, or the global axis index matches the eq label.
#[inline]
#[must_use]
pub(crate) fn fold_keeps_label(plan: &super::ReadPlan, logical_index: usize) -> bool {
    let Some(pred) = plan.label_eq.as_ref() else {
        return true;
    };
    let Some(g) = global_axis_from_logical(plan, logical_index, pred.axis) else {
        return false;
    };
    pred.matching.get(g).copied().unwrap_or(false)
}

fn global_axis_from_logical(plan: &super::ReadPlan, li: usize, axis: usize) -> Option<usize> {
    let ndim = plan.logical_selection_shape.len();
    if axis >= ndim || plan.selection_box_start.len() != ndim || plan.selection_step.len() != ndim {
        return None;
    }
    let mut rem = li;
    let mut coords = vec![0usize; ndim];
    for d in (0..ndim).rev() {
        let extent = *plan.logical_selection_shape.get(d)?;
        let extent = usize::try_from(extent).ok()?;
        if extent == 0 {
            return None;
        }
        coords[d] = rem % extent;
        rem /= extent;
    }
    if rem != 0 {
        return None;
    }
    let start = *plan.selection_box_start.get(axis)?;
    let step = (*plan.selection_step.get(axis)?).max(1);
    let start = usize::try_from(start).ok()?;
    let step = usize::try_from(step).ok()?;
    let lc = *coords.get(axis)?;
    start.checked_add(lc.checked_mul(step)?)
}

fn values_equal_fill(value: f64, fill: f64) -> bool {
    if fill.is_nan() {
        value.is_nan()
    } else {
        value.to_bits() == fill.to_bits()
    }
}

fn compare_cell(null: bool, op: ValueCompareOp, value: f64, rhs: f64) -> bool {
    if null {
        return matches!(op, ValueCompareOp::Ne);
    }
    match op {
        // Bit equality matches fill handling and avoids clippy::float_cmp.
        ValueCompareOp::Eq => value.to_bits() == rhs.to_bits(),
        ValueCompareOp::Ne => value.to_bits() != rhs.to_bits(),
        ValueCompareOp::Lt => value < rhs,
        ValueCompareOp::Le => value <= rhs,
        ValueCompareOp::Gt => value > rhs,
        ValueCompareOp::Ge => value >= rhs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pred(preds: Vec<ValuePredicate>, fill: Option<f64>) -> CellPredicate {
        CellPredicate::from_filter(&ValueFilter { predicates: preds }, fill)
    }

    #[test]
    fn nan_follows_fold_compare_and_null_rules() {
        let ge = pred(
            vec![ValuePredicate::Compare {
                op: ValueCompareOp::Ge,
                value: 0.0,
            }],
            None,
        );
        assert!(!ge.matches(f64::NAN));
        assert!(ge.matches(0.0));
        assert!(!ge.matches(-1.0));

        let ne = pred(
            vec![ValuePredicate::Compare {
                op: ValueCompareOp::Ne,
                value: 0.0,
            }],
            None,
        );
        assert!(ne.matches(f64::NAN));
        assert!(!ne.matches(0.0));

        let is_null = pred(vec![ValuePredicate::IsNull], None);
        assert!(is_null.matches(f64::NAN));
        assert!(!is_null.matches(1.0));
    }

    #[test]
    fn fill_is_null_for_numeric_compares() {
        let ge = pred(
            vec![ValuePredicate::Compare {
                op: ValueCompareOp::Ge,
                value: 0.0,
            }],
            Some(-1.0),
        );
        assert!(!ge.matches(-1.0));
        assert!(ge.matches(4.0));

        let is_null = pred(vec![ValuePredicate::IsNull], Some(-1.0));
        assert!(is_null.matches(-1.0));
        assert!(is_null.matches(f64::NAN));
        assert!(!is_null.matches(4.0));
    }

    #[test]
    fn i64_promotion_matches_fold_cast() {
        let raw = (1i64 << 53) + 1;
        let promoted = raw as f64;
        let eq = pred(
            vec![ValuePredicate::Compare {
                op: ValueCompareOp::Eq,
                value: (1i64 << 53) as f64,
            }],
            None,
        );
        assert_eq!(promoted, (1i64 << 53) as f64);
        assert!(eq.matches(promoted));
    }
}
