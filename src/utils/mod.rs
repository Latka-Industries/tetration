//! Shared low-level helpers used across layout, catalog, and related code.

pub mod dtype;
mod le_pod;
pub use le_pod::f32_le;
pub(crate) use le_pod::{f16_le, f64_le, i16_le, i32_le, i64_le, u8_le, u16_le, u32_le, u64_le};
pub mod fs_device;
pub mod host_memory;
pub(crate) mod wire;
pub(crate) mod zarr_dtype;
