//! Single-pass sum / sum-of-squares and min / max over numeric slabs (`f32` SIMD when available).

#[macro_use]
mod util;
mod float;
mod integer;

// Re-exported for tests and bulk fold paths; not every symbol is used inside `float`/`integer` alone.
#[allow(unused_imports)]
pub(crate) use float::{
    f16_min_max, f16_sum, f16_sum_sumsq, f32_min_max, f32_sum, f32_sum_sumsq, f64_sum, f64_sum_sumsq,
    scalar_f32_min_max, scalar_f32_sum, scalar_f32_sum_sumsq,
};
#[allow(unused_imports)]
pub(crate) use integer::{
    i16_min_max, i16_sum_sumsq, i32_min_max, i32_sum_sumsq, i64_min_max, i64_sum_sumsq,
    scalar_i32_sum_sumsq, scalar_i64_sum_sumsq, scalar_u8_sum_sumsq, scalar_u16_sum_sumsq,
    scalar_u32_sum_sumsq, scalar_u64_sum_sumsq, u8_min_max, u8_sum_sumsq, u16_min_max,
    u16_sum_sumsq, u32_min_max, u32_sum_sumsq, u64_min_max, u64_sum_sumsq,
};
