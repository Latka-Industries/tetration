//! Mmap chunk decode and parallel scatter into logical buffers.
//!
//! `pub(crate)` so in-crate tests (e.g. write-invariant proptest) can use
//! [`indexing`] helpers without widening the public crate API.

pub mod chunk_decode;
pub mod dense_visit;
pub mod indexing;

pub use chunk_decode::planned_chunk_mmap_slices;
