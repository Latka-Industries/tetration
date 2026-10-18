//! Phase 10: optional GPU execution routing (CUDA / Metal scalar `f32` reductions).
//!
//! Decode and mmap stay on the host; [`DeviceRoute`] records what was requested vs what ran.

use crate::query::fold::reduction::ReductionKind;
use crate::query::types::{ExecutionDeviceHint, ExecutionHints, Operation, ReadPlan};
use crate::utils::dtype::ElementDtype;
#[cfg(any(feature = "tetration-gpu", feature = "tetration-metal"))]
use crate::utils::host_memory;

/// Minimum logical selection size for `device: auto` to consider GPU (when enabled).
pub const GPU_AUTO_MIN_LOGICAL_BYTES: u64 = 64 * 1024 * 1024;

/// Max share of reported host RAM for the dense host `f32` materialize buffer before GPU reduce.
pub const GPU_HOST_MATERIALIZE_RAM_FRACTION: f64 = 0.85;

/// When host RAM cannot be probed, refuse GPU materialize above this logical size.
pub const GPU_HOST_MATERIALIZE_UNKNOWN_HOST_CAP_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// True when `logical_bytes` exceeds the host materialize budget (GPU path must decode to a dense vec).
#[must_use]
pub fn host_materialize_exceeds(logical_bytes: u64, host_available: Option<u64>) -> bool {
    let Some(available) = host_available else {
        return logical_bytes > GPU_HOST_MATERIALIZE_UNKNOWN_HOST_CAP_BYTES;
    };
    let limit = ((available as f64) * GPU_HOST_MATERIALIZE_RAM_FRACTION) as u64;
    logical_bytes > limit
}

/// Whether the GPU dense path may allocate a full-selection host buffer (streaming fold when false).
#[must_use]
#[cfg(any(feature = "tetration-gpu", feature = "tetration-metal"))]
pub(crate) fn host_materialize_fits(logical_bytes: u64) -> bool {
    !host_materialize_exceeds(logical_bytes, host_memory::available_memory_bytes())
}

/// Resolved device path for one execution (written to `execution` preview JSON).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRoute {
    pub requested: Option<ExecutionDeviceHint>,
    pub used: &'static str,
    pub fallback_reason: Option<&'static str>,
    pub gpu_reduce: bool,
    /// Decode/GPU overlap (streaming pipeline).
    pub gpu_pipeline: bool,
    /// Chunk shards across multiple devices (`cuda:multi` / `rocm:multi`).
    pub gpu_multi: bool,
    /// CUDA/ROCm device index when [`Self::used`] is `"cuda"` / `"rocm"`.
    pub cuda_device: Option<usize>,
}

impl DeviceRoute {
    #[must_use]
    pub const fn cpu_only() -> Self {
        Self {
            requested: None,
            used: "cpu",
            fallback_reason: None,
            gpu_reduce: false,
            gpu_pipeline: false,
            gpu_multi: false,
            cuda_device: None,
        }
    }

    #[must_use]
    pub const fn cpu_fallback(
        requested: Option<ExecutionDeviceHint>,
        reason: &'static str,
    ) -> Self {
        Self {
            requested,
            used: "cpu",
            fallback_reason: Some(reason),
            gpu_reduce: false,
            gpu_pipeline: false,
            gpu_multi: false,
            cuda_device: None,
        }
    }
}

#[must_use]
pub fn cuda_backend_available() -> bool {
    cfg!(feature = "tetration-gpu")
}

#[must_use]
pub fn rocm_backend_available() -> bool {
    cfg!(feature = "tetration-rocm")
}

#[must_use]
pub fn driver_backend_available() -> bool {
    cuda_backend_available() || rocm_backend_available()
}

/// Apple Metal path (`tetration-metal`, macOS only).
#[must_use]
pub fn metal_backend_available() -> bool {
    cfg!(all(feature = "tetration-metal", target_os = "macos"))
}

#[must_use]
pub fn gpu_backend_available() -> bool {
    driver_backend_available() || metal_backend_available()
}

/// Pick host vs device reduction for this execute.
#[must_use]
pub fn resolve_device_route(
    hints: Option<&ExecutionHints>,
    plan: &ReadPlan,
    dtype: ElementDtype,
    operation: Option<&Operation>,
) -> DeviceRoute {
    let Some(requested) = hints.and_then(|h| h.device) else {
        return DeviceRoute::cpu_only();
    };

    if operation.is_none() {
        return DeviceRoute::cpu_fallback(Some(requested), "preview_or_spill_only");
    }

    if let Some(op) = operation
        && op.requires_materialize()
    {
        return DeviceRoute::cpu_fallback(Some(requested), "tier_c_materialize");
    }

    if matches!(requested, ExecutionDeviceHint::Cpu) {
        return DeviceRoute {
            requested: Some(requested),
            used: "cpu",
            fallback_reason: None,
            gpu_reduce: false,
            gpu_pipeline: false,
            gpu_multi: false,
            cuda_device: None,
        };
    }

    if !matches!(dtype, ElementDtype::F32 | ElementDtype::F16) {
        return DeviceRoute::cpu_fallback(Some(requested), "gpu_unsupported_dtype");
    }

    let scalar_kind = operation.and_then(|op| {
        if op.axes().is_empty() && !op.requires_materialize() {
            Some(ReductionKind::from(op))
        } else {
            None
        }
    });
    if scalar_kind.is_none_or(|k| !gpu_supports_scalar_kind(k)) {
        return DeviceRoute::cpu_fallback(Some(requested), "gpu_unsupported_op");
    }

    let logical_bytes = u64::try_from(plan.logical_element_count)
        .ok()
        .and_then(|n| dtype.bytes_from_elem_count(n));

    if matches!(requested, ExecutionDeviceHint::Auto) {
        let Some(bytes) = logical_bytes else {
            return DeviceRoute::cpu_fallback(Some(requested), "auto_unknown_logical_bytes");
        };
        if bytes < GPU_AUTO_MIN_LOGICAL_BYTES {
            return DeviceRoute::cpu_fallback(Some(requested), "auto_below_size_threshold");
        }
    }

    match requested {
        ExecutionDeviceHint::Metal => {
            if !metal_backend_available() {
                return DeviceRoute::cpu_fallback(Some(requested), "gpu_feature_disabled");
            }
            gpu_scalar_route(Some(requested), "metal", None, false)
        }
        ExecutionDeviceHint::Cuda(idx) => {
            if !cuda_backend_available() {
                return DeviceRoute::cpu_fallback(Some(requested), "gpu_feature_disabled");
            }
            gpu_scalar_route(Some(requested), "cuda", Some(idx), false)
        }
        ExecutionDeviceHint::CudaMulti => {
            if !cuda_backend_available() {
                return DeviceRoute::cpu_fallback(Some(requested), "gpu_feature_disabled");
            }
            gpu_scalar_route(Some(requested), "cuda:multi", None, true)
        }
        ExecutionDeviceHint::Rocm(idx) => {
            if !rocm_backend_available() {
                return DeviceRoute::cpu_fallback(Some(requested), "gpu_feature_disabled");
            }
            gpu_scalar_route(Some(requested), "rocm", Some(idx), false)
        }
        ExecutionDeviceHint::RocmMulti => {
            if !rocm_backend_available() {
                return DeviceRoute::cpu_fallback(Some(requested), "gpu_feature_disabled");
            }
            gpu_scalar_route(Some(requested), "rocm:multi", None, true)
        }
        ExecutionDeviceHint::Auto => auto_device_route(Some(requested), logical_bytes),
        ExecutionDeviceHint::Cpu => unreachable!("handled above"),
    }
}

/// `device: auto` — Metal on macOS when enabled, else CUDA when enabled.
fn auto_device_route(
    requested: Option<ExecutionDeviceHint>,
    _logical_bytes: Option<u64>,
) -> DeviceRoute {
    if metal_backend_available() {
        return gpu_scalar_route(requested, "metal", None, false);
    }
    if cuda_backend_available() {
        return gpu_scalar_route(requested, "cuda", Some(0), false);
    }
    if rocm_backend_available() {
        return gpu_scalar_route(requested, "rocm", Some(0), false);
    }
    DeviceRoute::cpu_fallback(requested, "gpu_feature_disabled")
}

fn gpu_scalar_route(
    requested: Option<ExecutionDeviceHint>,
    used: &'static str,
    cuda_device: Option<usize>,
    multi: bool,
) -> DeviceRoute {
    // Oversized selections use per-chunk streaming GPU fold (see `gpu/streaming_fold.rs`).
    DeviceRoute {
        requested,
        used,
        fallback_reason: None,
        gpu_reduce: true,
        gpu_pipeline: false,
        gpu_multi: multi,
        cuda_device,
    }
}

/// Tier-A/B scalar ops implemented on GPU for dense `f32` (population var/std).
#[must_use]
pub(crate) fn gpu_supports_scalar_kind(kind: ReductionKind) -> bool {
    matches!(
        kind,
        ReductionKind::Sum
            | ReductionKind::Mean
            | ReductionKind::Min
            | ReductionKind::Max
            | ReductionKind::Count
            | ReductionKind::Var
            | ReductionKind::Std
    )
}

pub(crate) fn attach_device_fields(
    preview: &mut crate::query::types::QueryExecutionPreview,
    route: DeviceRoute,
) {
    preview.device_requested = route.requested.map(ExecutionDeviceHint::to_token);
    preview.device_used = Some(route.used);
    preview.device_fallback_reason = route.fallback_reason.map(str::to_string);
    preview.device_gpu_reduce = Some(route.gpu_reduce);
    preview.device_gpu_pipeline = Some(route.gpu_pipeline);
    preview.device_gpu_multi = Some(route.gpu_multi);
}
