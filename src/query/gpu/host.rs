//! Shared host-side API for optional GPU scalar reductions.
//!
//! Backends (CUDA / ROCm / Metal) implement [`GpuScalarReduce`]. Streaming and
//! dense scalar folds dispatch through that trait instead of matching on each
//! backend at every call site.

use crate::query::fold::reduction::{ReductionKind, ScalarReductionResult};

/// Host entry points shared by CUDA, ROCm, and Metal scalar folds.
pub(crate) trait GpuScalarReduce {
    /// Refuse when `logical_bytes` would exceed the device buffer budget.
    fn vram_check(&self, logical_bytes: usize) -> Result<(), &'static str>;

    /// Dense `f32` scalar reduction on device (kind must be GPU-supported).
    fn reduce_f32_scalar(
        &self,
        vals: &[f32],
        kind: ReductionKind,
    ) -> Result<ScalarReductionResult, &'static str>;
}

/// CUDA or HIP device index (`tetration-gpu` / `tetration-rocm` share the cudarc path).
#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CudaDevice(pub usize);

#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
impl GpuScalarReduce for CudaDevice {
    fn vram_check(&self, logical_bytes: usize) -> Result<(), &'static str> {
        super::cuda::vram_check(self.0, logical_bytes)
    }

    fn reduce_f32_scalar(
        &self,
        vals: &[f32],
        kind: ReductionKind,
    ) -> Result<ScalarReductionResult, &'static str> {
        super::cuda::reduce_f32_scalar(vals, kind, self.0).map_err(|_| "gpu_runtime_error")
    }
}

#[cfg(feature = "tetration-rocm")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct RocmDevice(pub usize);

#[cfg(feature = "tetration-rocm")]
impl GpuScalarReduce for RocmDevice {
    fn vram_check(&self, logical_bytes: usize) -> Result<(), &'static str> {
        super::rocm::vram_check(self.0, logical_bytes)
    }

    fn reduce_f32_scalar(
        &self,
        vals: &[f32],
        kind: ReductionKind,
    ) -> Result<ScalarReductionResult, &'static str> {
        super::rocm::reduce_f32_scalar(vals, kind, self.0).map_err(|_| "gpu_runtime_error")
    }
}

#[cfg(all(feature = "tetration-metal", target_os = "macos"))]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MetalDevice;

#[cfg(all(feature = "tetration-metal", target_os = "macos"))]
impl GpuScalarReduce for MetalDevice {
    fn vram_check(&self, logical_bytes: usize) -> Result<(), &'static str> {
        super::metal::vram_check(logical_bytes)
    }

    fn reduce_f32_scalar(
        &self,
        vals: &[f32],
        kind: ReductionKind,
    ) -> Result<ScalarReductionResult, &'static str> {
        super::metal::reduce_f32_scalar(vals, kind).map_err(|_| "gpu_runtime_error")
    }
}
