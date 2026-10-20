//! Shared reduction kinds and single-pass accumulators.

use crate::query::types::Operation;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ReductionKind {
    Sum,
    Mean,
    Min,
    Max,
    Count,
    Var,
    Std,
    Product,
    NormL1,
    NormL2,
    NanMean,
    NanStd,
    AllFinite,
    AnyNan,
    AnyInf,
    NanCount,
    InfCount,
    NullCount { fill: f64 },
    ArgMin,
    ArgMax,
}

impl From<&Operation> for ReductionKind {
    fn from(op: &Operation) -> Self {
        debug_assert!(!op.requires_materialize());
        match op {
            Operation::Sum { .. } => Self::Sum,
            Operation::Mean { .. } => Self::Mean,
            Operation::Min { .. } => Self::Min,
            Operation::Max { .. } => Self::Max,
            Operation::Count { .. } => Self::Count,
            Operation::Var { .. } => Self::Var,
            Operation::Std { .. } => Self::Std,
            Operation::NanMean { .. } => Self::NanMean,
            Operation::NanStd { .. } => Self::NanStd,
            Operation::Product { .. } => Self::Product,
            Operation::NormL1 { .. } => Self::NormL1,
            Operation::NormL2 { .. } => Self::NormL2,
            Operation::AllFinite { .. } => Self::AllFinite,
            Operation::AnyNan { .. } => Self::AnyNan,
            Operation::AnyInf { .. } => Self::AnyInf,
            Operation::NanCount { .. } => Self::NanCount,
            Operation::InfCount { .. } => Self::InfCount,
            Operation::NullCount { fill: Some(f), .. } => Self::NullCount { fill: *f },
            Operation::NullCount { fill: None, .. } => {
                unreachable!("null_count fill must be resolved before fold")
            }
            Operation::ArgMin { .. } => Self::ArgMin,
            Operation::ArgMax { .. } => Self::ArgMax,
            Operation::Median { .. }
            | Operation::Quantile { .. }
            | Operation::Histogram { .. }
            | Operation::Covariance { .. }
            | Operation::Correlation { .. }
            | Operation::Transform { .. } => {
                unreachable!("tier-C stats and transforms use dedicated execution paths")
            }
        }
    }
}

mod scalar;
mod value_accum;
mod welford;

#[allow(unused_imports)]
// `ScalarReductionResult` / `WelfordAccum` mainly used under `#[cfg(test)]` or GPU features
pub(crate) use scalar::{ArgIndexAccum, ArgIndexAccumWire, ScalarReductionResult};
pub(crate) use value_accum::{ValueAccum, ValueAccumWire};
pub(crate) use welford::WelfordAccum;
