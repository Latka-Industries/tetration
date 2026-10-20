//! ROCm / HIP path — same NVRTC kernels as CUDA when built with `tetration-rocm`.

#[cfg(feature = "tetration-rocm")]
pub(crate) use super::cuda::{reduce_f32_scalar, visible_device_count, vram_check};
