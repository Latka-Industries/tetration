//! Scalar and partial-axis reductions over decoded selections.
//!
//! [`fold_policy`] chooses parallel chunks, offset-ordered sequential I/O, or contiguous
//! byte-stream [`linear_scan`]. Linear scan skips per-element predicates — cell/`label_eq`
//! filters force the chunked path so position/cell keep logic runs.

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
