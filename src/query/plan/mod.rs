//! Selection geometry and chunk-level read plans.
//!
//! Zone-map and secondary-index pushdown live in [`zone_pushdown`] and [`index_pushdown`]
//! (`value_postings`, `sorted_axis_range`, `coord_label_postings`).

pub mod fetch;
pub(crate) mod index_pushdown;
pub mod read_plan;
pub mod selection;
pub(crate) mod zone_pushdown;
