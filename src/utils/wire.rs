//! Alignment and byte-span checks from [`argus-chunk`](https://docs.rs/argus-chunk).
//!
//! Fixed-width LE encode/decode uses [`argus::LeReader`] / [`argus::LeWriter`] at the
//! call site (same pattern as Tessera) — no local `put_u32_le` / `u64_le_at` helpers.

pub(crate) use argus::{
    SpanError, align8 as align8_u64, checked_subslice as checked_usize_subslice,
    checked_u64_byte_span, padding_to_align8,
};
