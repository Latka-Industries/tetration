//! Multi-GPU chunk sharding (CUDA / ROCm): one partial fold per device, host merge.

#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
use rayon::prelude::*;

#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
use super::streaming_fold::{
    StreamingFoldRequest, StreamingGpuBackend, collect_planned_chunk_values,
    streaming_fold_partial_driver,
};
#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
use crate::query::device::DeviceRoute;
#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
use crate::query::fold::reduction::{ReductionKind, ScalarReductionResult};
#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
use crate::query::fold::{self, FoldPlanOutcome};
#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
use crate::query::types::ReadPlan;

#[cfg(feature = "tetration-gpu")]
use super::cuda;

#[cfg(feature = "tetration-rocm")]
use super::rocm;

/// Indices `shard`, `shard + n_shards`, … covering `0..total` exactly once across shards.
#[must_use]
pub(crate) fn shard_chunk_indices(shard: usize, n_shards: usize, total: usize) -> Vec<usize> {
    debug_assert!(shard < n_shards);
    (shard..total).step_by(n_shards).collect()
}

#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
pub(crate) fn visible_driver_device_count() -> usize {
    #[cfg(feature = "tetration-gpu")]
    {
        return cuda::visible_device_count();
    }
    #[cfg(all(feature = "tetration-rocm", not(feature = "tetration-gpu")))]
    {
        return rocm::visible_device_count();
    }
    #[allow(unreachable_code)]
    0
}

#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
pub(crate) fn try_multi_driver_fold(
    used: &'static str,
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    route: DeviceRoute,
    f16_input: bool,
) -> Result<(FoldPlanOutcome, DeviceRoute), &'static str> {
    let n_devices = visible_driver_device_count();
    if n_devices <= 1 {
        let device_index = 0;
        let backend = driver_backend(device_index)?;
        return super::streaming_fold::try_streaming_f32_fold(StreamingFoldRequest {
            backend,
            used,
            cuda_device: Some(device_index),
            mmap,
            plan,
            max_preview,
            kind,
            route,
            f16_input,
        });
    }

    let n = plan.logical_f32_element_count;
    let preview_cap = max_preview.min(n);
    let mut preview = vec![0.0_f32; preview_cap];
    let mut gpu_acc = ScalarReductionResult::default_fields(0);
    let mut total_bytes_read_from_disk = 0u64;

    let partials: Result<Vec<(ScalarReductionResult, u64)>, &'static str> = (0..n_devices)
        .into_par_iter()
        .map(|device_index| {
            let indices = shard_chunk_indices(device_index, n_devices, plan.chunks.len());
            if indices.is_empty() {
                return Ok((ScalarReductionResult::default_fields(0), 0));
            }
            let backend = driver_backend(device_index)?;
            streaming_fold_partial_driver(&backend, mmap, plan, kind, f16_input, &indices)
        })
        .collect();

    let partials = partials?;
    for (part, bytes) in partials {
        total_bytes_read_from_disk = total_bytes_read_from_disk
            .checked_add(bytes)
            .ok_or("gpu_host_decode_failed")?;
        if part.element_count > 0 {
            gpu_acc.merge_partial(&part, kind);
        }
    }

    if gpu_acc.element_count == 0 {
        return Err("gpu_empty_selection");
    }

    if preview_cap > 0 {
        fill_preview_head(mmap, plan, preview_cap, &mut preview, f16_input)
            .map_err(|_| "gpu_host_decode_failed")?;
    }

    fold::shared::validate_fold_preview(n > 0, &preview, preview_cap)
        .map_err(|_| "gpu_empty_selection")?;

    let operation = gpu_acc.finalize_merged(kind).into();
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
        gpu_pipeline: true,
        gpu_multi: true,
        cuda_device: None,
    };
    Ok((outcome, ok_route))
}

#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
fn driver_backend(device_index: usize) -> Result<StreamingGpuBackend, &'static str> {
    #[cfg(feature = "tetration-gpu")]
    {
        return Ok(StreamingGpuBackend::Cuda(device_index));
    }
    #[cfg(all(feature = "tetration-rocm", not(feature = "tetration-gpu")))]
    {
        return Ok(StreamingGpuBackend::Rocm(device_index));
    }
    #[allow(unreachable_code)]
    Err("gpu_feature_disabled")
}

#[cfg(any(feature = "tetration-gpu", feature = "tetration-rocm"))]
fn fill_preview_head(
    mmap: &[u8],
    plan: &ReadPlan,
    preview_cap: usize,
    preview: &mut [f32],
    f16_input: bool,
) -> Result<(), crate::query::types::TetError> {
    for c in plan.chunks.iter().take(4) {
        let (_bytes, vals) =
            collect_planned_chunk_values(mmap, plan, c, preview_cap, preview, f16_input)?;
        if vals.len() >= preview_cap {
            break;
        }
    }
    Ok(())
}
