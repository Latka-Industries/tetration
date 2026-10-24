//! Axes-only reduction kinds (flat wire keys without parametric payloads).

use super::document::Operation;

/// Closed set of axes-only operations (`sum`, `mean`, …).
///
/// Parametric ops (`quantile`, `histogram`, `null_count`, `transform`) stay on [`Operation`] only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimpleOp {
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
    AllFinite,
    AnyNan,
    AnyInf,
    ArgMin,
    ArgMax,
    Median,
    NanCount,
    InfCount,
    Covariance,
    Correlation,
    NanMean,
    NanStd,
}

impl SimpleOp {
    /// Wire keys for every [`SimpleOp`] variant (stable order matches historical `OP_KEYS` prefix).
    pub const WIRE_KEYS: &'static [&'static str] = &[
        "sum",
        "mean",
        "min",
        "max",
        "count",
        "var",
        "std",
        "product",
        "norm_l1",
        "norm_l2",
        "all_finite",
        "any_nan",
        "any_inf",
        "arg_min",
        "arg_max",
        "median",
        "nan_count",
        "inf_count",
        "covariance",
        "correlation",
        "nan_mean",
        "nan_std",
    ];

    /// Parametric top-level keys (not [`SimpleOp`]).
    pub const PARAMETRIC_WIRE_KEYS: &'static [&'static str] =
        &["quantile", "histogram", "null_count", "transform"];

    /// Parse an axes-only wire key.
    #[must_use]
    pub fn parse(token: &str) -> Option<Self> {
        Some(match token {
            "sum" => Self::Sum,
            "mean" => Self::Mean,
            "min" => Self::Min,
            "max" => Self::Max,
            "count" => Self::Count,
            "var" => Self::Var,
            "std" => Self::Std,
            "product" => Self::Product,
            "norm_l1" => Self::NormL1,
            "norm_l2" => Self::NormL2,
            "all_finite" => Self::AllFinite,
            "any_nan" => Self::AnyNan,
            "any_inf" => Self::AnyInf,
            "arg_min" => Self::ArgMin,
            "arg_max" => Self::ArgMax,
            "median" => Self::Median,
            "nan_count" => Self::NanCount,
            "inf_count" => Self::InfCount,
            "covariance" => Self::Covariance,
            "correlation" => Self::Correlation,
            "nan_mean" => Self::NanMean,
            "nan_std" => Self::NanStd,
            _ => return None,
        })
    }

    /// Stable flat-wire key.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Min => "min",
            Self::Max => "max",
            Self::Count => "count",
            Self::Var => "var",
            Self::Std => "std",
            Self::Product => "product",
            Self::NormL1 => "norm_l1",
            Self::NormL2 => "norm_l2",
            Self::AllFinite => "all_finite",
            Self::AnyNan => "any_nan",
            Self::AnyInf => "any_inf",
            Self::ArgMin => "arg_min",
            Self::ArgMax => "arg_max",
            Self::Median => "median",
            Self::NanCount => "nan_count",
            Self::InfCount => "inf_count",
            Self::Covariance => "covariance",
            Self::Correlation => "correlation",
            Self::NanMean => "nan_mean",
            Self::NanStd => "nan_std",
        }
    }

    /// Build the corresponding [`Operation`] with `axes`.
    #[must_use]
    pub fn with_axes(self, axes: Vec<String>) -> Operation {
        match self {
            Self::Sum => Operation::Sum { axes },
            Self::Mean => Operation::Mean { axes },
            Self::Min => Operation::Min { axes },
            Self::Max => Operation::Max { axes },
            Self::Count => Operation::Count { axes },
            Self::Var => Operation::Var { axes },
            Self::Std => Operation::Std { axes },
            Self::Product => Operation::Product { axes },
            Self::NormL1 => Operation::NormL1 { axes },
            Self::NormL2 => Operation::NormL2 { axes },
            Self::AllFinite => Operation::AllFinite { axes },
            Self::AnyNan => Operation::AnyNan { axes },
            Self::AnyInf => Operation::AnyInf { axes },
            Self::ArgMin => Operation::ArgMin { axes },
            Self::ArgMax => Operation::ArgMax { axes },
            Self::Median => Operation::Median { axes },
            Self::NanCount => Operation::NanCount { axes },
            Self::InfCount => Operation::InfCount { axes },
            Self::Covariance => Operation::Covariance { axes },
            Self::Correlation => Operation::Correlation { axes },
            Self::NanMean => Operation::NanMean { axes },
            Self::NanStd => Operation::NanStd { axes },
        }
    }
}
