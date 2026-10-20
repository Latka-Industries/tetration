//! Selection geometry and chunk-level read plans.

pub mod fetch;
pub(crate) mod index_pushdown;
pub mod read_plan;
pub mod selection;
pub(crate) mod zone_pushdown;
