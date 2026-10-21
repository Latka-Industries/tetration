//! Scalar and partial-axis reductions over decoded selections.

pub mod fold_policy;
pub mod linear_scan;
pub mod parallel;
pub mod partial;
pub mod partial_geometry;
pub mod reduction;
pub mod shared;
pub(crate) mod variance_simd;
pub(crate) mod window;
pub(crate) mod window_geometry;

pub(crate) use shared::FoldPlanOutcome;
