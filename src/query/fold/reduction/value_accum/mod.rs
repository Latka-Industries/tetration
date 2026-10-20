//! Single-pass accumulator for one scalar or partial-reduction cell.

mod finish;
mod helpers;
mod merge;
mod push;
mod slab;
mod wire;

pub(crate) use wire::ValueAccumWire;

use super::WelfordAccum;

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
    /// NaN / inf / null-match tally for [`super::ReductionKind::NanCount`], [`super::ReductionKind::InfCount`], and [`super::ReductionKind::NullCount`].
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
