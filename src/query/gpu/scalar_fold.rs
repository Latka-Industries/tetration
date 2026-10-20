//! Host materialize + shared helpers for device scalar GPU folds (`f32` / promoted `f16`).

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-rocm",
    feature = "tetration-metal"
))]
use crate::query::device::DeviceRoute;
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-rocm",
    feature = "tetration-metal"
))]
use crate::query::dispatch;
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-rocm",
    feature = "tetration-metal"
))]
use crate::query::fold::reduction::ReductionKind;
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-rocm",
    feature = "tetration-metal"
))]
use crate::query::fold::{self, FoldPlanOutcome};
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-rocm",
    feature = "tetration-metal"
))]
use crate::query::materialize::{self, parallel};
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-rocm",
    feature = "tetration-metal"
))]
use crate::query::types::{ReadPlan, TetError};

#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
use super::host::CudaDevice;
#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-rocm",
    feature = "tetration-metal"
))]
use super::host::GpuScalarReduce;
#[cfg(all(feature = "tetration-metal", target_os = "macos"))]
use super::host::MetalDevice;
#[cfg(feature = "tetration-rocm")]
use super::host::RocmDevice;

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-rocm",
    feature = "tetration-metal"
))]
use super::streaming_fold::{StreamingFoldRequest, StreamingGpuBackend, try_streaming_f32_fold};

#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
pub(crate) fn try_scalar_f32_fold_cuda(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    route: DeviceRoute,
    f16_input: bool,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    if route.gpu_multi {
        return super::multi::try_multi_driver_fold(
            "cuda:multi",
            mmap,
            plan,
            max_preview,
            kind,
            route,
            f16_input,
        );
    }
    let device_index = route.cuda_device.unwrap_or(0);
    let n = plan.logical_element_count;
    let logical_bytes = u64::try_from(n.checked_mul(4).ok_or("gpu_logical_bytes_overflow")?)
        .map_err(|_| "gpu_logical_bytes_overflow")?;
    if !crate::query::device::host_materialize_fits(logical_bytes) {
        return try_streaming_f32_fold(StreamingFoldRequest {
            backend: StreamingGpuBackend::Cuda(device_index),
            used: "cuda",
            cuda_device: Some(device_index),
            mmap,
            plan,
            max_preview,
            kind,
            route,
            f16_input,
        });
    }
    let logical_bytes_usize =
        usize::try_from(logical_bytes).map_err(|_| "gpu_logical_bytes_overflow")?;
    let device = CudaDevice(device_index);
    if let Err(reason) = device.vram_check(logical_bytes_usize) {
        if reason == "gpu_vram_exceeded" {
            return try_streaming_f32_fold(StreamingFoldRequest {
                backend: StreamingGpuBackend::Cuda(device_index),
                used: "cuda",
                cuda_device: Some(device_index),
                mmap,
                plan,
                max_preview,
                kind,
                route,
                f16_input,
            });
        }
        return Err(reason);
    }

    let mut host = vec![0.0_f32; n];
    let total_bytes_read_from_disk = materialize_host_f32(mmap, plan, &mut host, f16_input)
        .map_err(|_| "gpu_host_decode_failed")?;

    let scalar = device.reduce_f32_scalar(&host, kind)?;

    build_gpu_fold_outcome(DenseGpuFoldResult {
        route,
        used: "cuda",
        cuda_device: Some(device_index),
        host,
        max_preview,
        n,
        total_bytes_read_from_disk,
        operation: scalar.into(),
        dense_materialize: true,
    })
}

#[cfg(feature = "tetration-rocm")]
pub(crate) fn try_scalar_f32_fold_rocm(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    route: DeviceRoute,
    f16_input: bool,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    if route.gpu_multi {
        return super::multi::try_multi_driver_fold(
            "rocm:multi",
            mmap,
            plan,
            max_preview,
            kind,
            route,
            f16_input,
        );
    }
    let device_index = route.cuda_device.unwrap_or(0);
    let n = plan.logical_element_count;
    let logical_bytes = u64::try_from(n.checked_mul(4).ok_or("gpu_logical_bytes_overflow")?)
        .map_err(|_| "gpu_logical_bytes_overflow")?;
    if !crate::query::device::host_materialize_fits(logical_bytes) {
        return try_streaming_f32_fold(StreamingFoldRequest {
            backend: StreamingGpuBackend::Rocm(device_index),
            used: "rocm",
            cuda_device: Some(device_index),
            mmap,
            plan,
            max_preview,
            kind,
            route,
            f16_input,
        });
    }
    let logical_bytes_usize =
        usize::try_from(logical_bytes).map_err(|_| "gpu_logical_bytes_overflow")?;
    let device = RocmDevice(device_index);
    if let Err(reason) = device.vram_check(logical_bytes_usize) {
        if reason == "gpu_vram_exceeded" {
            return try_streaming_f32_fold(StreamingFoldRequest {
                backend: StreamingGpuBackend::Rocm(device_index),
                used: "rocm",
                cuda_device: Some(device_index),
                mmap,
                plan,
                max_preview,
                kind,
                route,
                f16_input,
            });
        }
        return Err(reason);
    }

    let mut host = vec![0.0_f32; n];
    let total_bytes_read_from_disk = materialize_host_f32(mmap, plan, &mut host, f16_input)
        .map_err(|_| "gpu_host_decode_failed")?;

    let scalar = device.reduce_f32_scalar(&host, kind)?;

    build_gpu_fold_outcome(DenseGpuFoldResult {
        route,
        used: "rocm",
        cuda_device: Some(device_index),
        host,
        max_preview,
        n,
        total_bytes_read_from_disk,
        operation: scalar.into(),
        dense_materialize: true,
    })
}

#[cfg(all(feature = "tetration-metal", target_os = "macos"))]
pub(crate) fn try_scalar_f32_fold_metal(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    route: DeviceRoute,
    f16_input: bool,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    let n = plan.logical_element_count;
    let logical_bytes = u64::try_from(n.checked_mul(4).ok_or("gpu_logical_bytes_overflow")?)
        .map_err(|_| "gpu_logical_bytes_overflow")?;
    if !crate::query::device::host_materialize_fits(logical_bytes) {
        return try_streaming_f32_fold(StreamingFoldRequest {
            backend: StreamingGpuBackend::Metal,
            used: "metal",
            cuda_device: None,
            mmap,
            plan,
            max_preview,
            kind,
            route,
            f16_input,
        });
    }
    let logical_bytes_usize =
        usize::try_from(logical_bytes).map_err(|_| "gpu_logical_bytes_overflow")?;
    let device = MetalDevice;
    if let Err(reason) = device.vram_check(logical_bytes_usize) {
        if reason == "gpu_vram_exceeded" {
            return try_streaming_f32_fold(StreamingFoldRequest {
                backend: StreamingGpuBackend::Metal,
                used: "metal",
                cuda_device: None,
                mmap,
                plan,
                max_preview,
                kind,
                route,
                f16_input,
            });
        }
        return Err(reason);
    }

    let mut host = vec![0.0_f32; n];
    let total_bytes_read_from_disk = materialize_host_f32(mmap, plan, &mut host, f16_input)
        .map_err(|_| "gpu_host_decode_failed")?;

    let scalar = device.reduce_f32_scalar(&host, kind)?;

    build_gpu_fold_outcome(DenseGpuFoldResult {
        route,
        used: "metal",
        cuda_device: None,
        host,
        max_preview,
        n,
        total_bytes_read_from_disk,
        operation: scalar.into(),
        dense_materialize: true,
    })
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-rocm",
    feature = "tetration-metal"
))]
pub(crate) struct DenseGpuFoldResult {
    pub route: DeviceRoute,
    pub used: &'static str,
    pub cuda_device: Option<usize>,
    pub host: Vec<f32>,
    pub max_preview: usize,
    pub n: usize,
    pub total_bytes_read_from_disk: u64,
    pub operation: crate::query::types::OperationPreviewFields,
    pub dense_materialize: bool,
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-rocm",
    feature = "tetration-metal"
))]
fn build_gpu_fold_outcome(
    input: DenseGpuFoldResult,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    let DenseGpuFoldResult {
        route,
        used,
        cuda_device,
        host,
        max_preview,
        n,
        total_bytes_read_from_disk,
        operation,
        dense_materialize,
    } = input;
    let preview_cap = max_preview.min(n);
    let preview = if max_preview == 0 {
        Vec::new()
    } else {
        host[..preview_cap].to_vec()
    };
    fold::shared::validate_fold_preview(n > 0, &preview, preview_cap)
        .map_err(|_| "gpu_empty_selection")?;

    let outcome = fold::shared::build_fold_plan_outcome(
        preview,
        max_preview,
        n,
        total_bytes_read_from_disk,
        operation,
    );
    let ok_route = DeviceRoute {
        requested: route.requested,
        used,
        fallback_reason: None,
        gpu_reduce: true,
        gpu_pipeline: route.gpu_pipeline && !dense_materialize,
        gpu_multi: route.gpu_multi,
        cuda_device,
    };
    Ok((outcome, ok_route))
}

#[cfg(any(
    feature = "tetration-gpu",
    feature = "tetration-rocm",
    feature = "tetration-metal"
))]
fn materialize_host_f32(
    mmap: &[u8],
    plan: &ReadPlan,
    dst: &mut [f32],
    f16_input: bool,
) -> Result<u64, TetError> {
    if f16_input {
        let preview: &mut [f32] = &mut [];
        let mut offset = 0usize;
        for c in &plan.chunks {
            let (bytes, vals) = super::streaming_fold::collect_planned_chunk_values(
                mmap, plan, c, 0, preview, true,
            )?;
            let end = offset + vals.len();
            if end > dst.len() {
                return Err(TetError::Validation(
                    "f16 gpu materialize buffer too small".into(),
                ));
            }
            dst[offset..end].copy_from_slice(&vals);
            offset = end;
            let _ = bytes;
        }
        let bytes = plan.chunks.iter().map(|c| c.stored_byte_len);
        return dispatch::sum_chunk_read_bytes(bytes);
    }
    if plan.chunks.len() > 1 {
        parallel::materialize_read_plan_f32_le_into_parallel(mmap, plan, None, dst)?;
    } else {
        materialize::materialize_read_plan_f32_le_into(mmap, plan, None, dst)?;
    }
    let bytes = plan.chunks.iter().map(|c| c.stored_byte_len);
    dispatch::sum_chunk_read_bytes(bytes)
}
