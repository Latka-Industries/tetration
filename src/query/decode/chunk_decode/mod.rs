//! Chunk payload I/O: mmap slices, codec decode, and per-element visitation.

mod elem;
mod prepare;
mod scatter;
mod visit;

pub use prepare::planned_chunk_mmap_slices;

// Re-export the former flat-module `pub`/`pub(crate)` surface so call sites keep
// `chunk_decode::…` paths. Some symbols are only used inside this directory today.
#[allow(unused_imports)]
pub(crate) use prepare::{decode_planned_chunk_bytes, planned_chunk_stored_slice};

#[allow(unused_imports)]
pub(crate) use visit::{
    fold_planned_chunk_f32, fold_planned_chunk_f64, visit_planned_chunk, visit_planned_chunk_f16,
    visit_planned_chunk_f16_as_f64, visit_planned_chunk_f64, visit_planned_chunk_i16,
    visit_planned_chunk_i16_as_f64, visit_planned_chunk_i32, visit_planned_chunk_i32_as_f64,
    visit_planned_chunk_i64, visit_planned_chunk_i64_as_f64, visit_planned_chunk_u8,
    visit_planned_chunk_u8_as_f64, visit_planned_chunk_u16, visit_planned_chunk_u16_as_f64,
    visit_planned_chunk_u32, visit_planned_chunk_u32_as_f64, visit_planned_chunk_u64,
    visit_planned_chunk_u64_as_f64,
};

pub(crate) use scatter::{
    scatter_chunk_into_plan, scatter_chunk_into_plan_f16, scatter_chunk_into_plan_f64,
    scatter_chunk_into_plan_i16, scatter_chunk_into_plan_i32, scatter_chunk_into_plan_i64,
    scatter_chunk_into_plan_u8, scatter_chunk_into_plan_u16, scatter_chunk_into_plan_u32,
    scatter_chunk_into_plan_u64,
};
