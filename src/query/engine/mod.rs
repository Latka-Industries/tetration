//! Query run entrypoints and execution orchestration (budget, spill).

pub(crate) mod budget;
mod operations;
mod run;
pub(crate) mod spill_policy;
pub mod store;

pub use crate::query::decode::planned_chunk_mmap_slices;
pub use crate::query::materialize::int::{
    materialize_read_plan_i16_le, materialize_read_plan_i32_le, materialize_read_plan_i64_le,
    materialize_read_plan_u8_le, materialize_read_plan_u16_le, spill_read_plan_i16_le,
    spill_read_plan_i32_le, spill_read_plan_i64_le, spill_read_plan_u8_le, spill_read_plan_u16_le,
};
pub use crate::query::materialize::parallel::{
    materialize_read_plan_f32_le_into_parallel, materialize_read_plan_f32_le_parallel,
};
pub use crate::query::materialize::{
    MaterializeReadPlanF32IntoOutcome, materialize_read_plan_f32_le,
    materialize_read_plan_f32_le_into, materialize_read_plan_f64_le, spill_read_plan_f32_le,
};
pub use budget::{DEFAULT_MEMORY_BUDGET_BYTES, ExecutionBudget, MemoryStrategy};
pub use run::{
    PlannedRead, plan_query_empty, plan_query_with_tet_mmap, plan_query_with_tet_mmap_ex,
    plan_read_for_document,
};
pub use spill_policy::SpillPathAllowlist;
#[doc(hidden)]
pub use spill_policy::TempSpillFile;
