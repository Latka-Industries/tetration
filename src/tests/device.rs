//! Phase 10 device routing (CPU scaffold until GPU kernels ship).

use super::fixture;
use crate::layout::mmap_file_read;
use crate::query::{
    ExecutionDeviceHint, GPU_HOST_MATERIALIZE_RAM_FRACTION, cuda_backend_available,
    host_materialize_exceeds, metal_backend_available, plan_query_with_tet_mmap,
    resolve_device_route, validate_query,
};
use crate::utils::dtype::ElementDtype;

#[test]
fn host_materialize_exceeds_when_buffer_larger_than_host_budget() {
    let gib = 1024_u64.pow(3);
    assert!(!host_materialize_exceeds(gib, Some(8 * gib)));
    let limit = ((8 * gib) as f64 * GPU_HOST_MATERIALIZE_RAM_FRACTION) as u64;
    assert!(host_materialize_exceeds(limit + 1, Some(8 * gib)));
    assert!(!host_materialize_exceeds(gib, None));
    assert!(host_materialize_exceeds(20 * gib, None));
}

#[test]
fn execution_device_hint_parse_tokens() {
    assert_eq!(
        ExecutionDeviceHint::parse("cpu").unwrap(),
        ExecutionDeviceHint::Cpu
    );
    assert_eq!(
        ExecutionDeviceHint::parse("auto").unwrap(),
        ExecutionDeviceHint::Auto
    );
    assert_eq!(
        ExecutionDeviceHint::parse("cuda").unwrap(),
        ExecutionDeviceHint::Cuda(0)
    );
    assert_eq!(
        ExecutionDeviceHint::parse("cuda:2").unwrap(),
        ExecutionDeviceHint::Cuda(2)
    );
    assert_eq!(
        ExecutionDeviceHint::parse("metal").unwrap(),
        ExecutionDeviceHint::Metal
    );
    assert_eq!(
        ExecutionDeviceHint::parse("cuda:multi").unwrap(),
        ExecutionDeviceHint::CudaMulti
    );
    assert_eq!(
        ExecutionDeviceHint::parse("rocm").unwrap(),
        ExecutionDeviceHint::Rocm(0)
    );
    assert_eq!(
        ExecutionDeviceHint::parse("rocm:multi").unwrap(),
        ExecutionDeviceHint::RocmMulti
    );
    assert_eq!(
        ExecutionDeviceHint::parse("rocm:2").unwrap(),
        ExecutionDeviceHint::Rocm(2)
    );
    assert_eq!(ExecutionDeviceHint::Cuda(0).to_token(), "cuda");
    assert_eq!(ExecutionDeviceHint::Cuda(2).to_token(), "cuda:2");
    assert_eq!(ExecutionDeviceHint::Cuda(0).as_static_str(), Some("cuda"));
    assert_eq!(ExecutionDeviceHint::Cuda(2).as_static_str(), None);
    assert_eq!(
        ExecutionDeviceHint::parse(&ExecutionDeviceHint::Cuda(2).to_token()).unwrap(),
        ExecutionDeviceHint::Cuda(2)
    );
}

#[test]
fn parse_query_json_execution_device() {
    let doc = crate::query::parse_query_json(
        r#"{"dataset":"a","mean":[],"execution":{"device":"auto"}}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    assert_eq!(
        doc.execution.as_ref().unwrap().device,
        Some(ExecutionDeviceHint::Auto)
    );
}

#[test]
fn resolve_device_auto_small_selection_stays_cpu() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("small.tet");
    fixture::write_multichunk_2x3_tiles(&path, "a");
    let doc = fixture::query_files::json("mean_a");
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let plan = response.read_plan.as_ref().unwrap();
    let route = resolve_device_route(
        Some(&crate::query::ExecutionHints {
            device: Some(ExecutionDeviceHint::Auto),
            ..Default::default()
        }),
        plan,
        ElementDtype::F32,
        doc.operation.as_ref(),
    );
    assert_eq!(route.used, "cpu");
    assert_eq!(route.fallback_reason, Some("auto_below_size_threshold"));
}

#[test]
fn resolve_device_cuda_falls_back_without_cuda_feature() {
    if cuda_backend_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.tet");
    fixture::write_verify_large_f32(&path, "a");
    let doc = crate::query::parse_query_json(
        r#"{"dataset":"a","mean":[],"execution":{"device":"cuda:0"}}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let ex = response.execution.as_ref().unwrap();
    assert_eq!(ex.device_requested.as_deref(), Some("cuda"));
    assert_eq!(ex.device_used, Some("cpu"));
    assert_eq!(
        ex.device_fallback_reason.as_deref(),
        Some("gpu_feature_disabled")
    );
    assert_eq!(ex.device_gpu_reduce, Some(false));
}

#[test]
fn resolve_device_metal_falls_back_without_metal_feature() {
    if metal_backend_available() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.tet");
    fixture::write_verify_large_f32(&path, "a");
    let doc = crate::query::parse_query_json(
        r#"{"dataset":"a","mean":[],"execution":{"device":"metal"}}"#,
    )
    .unwrap();
    validate_query(&doc).unwrap();
    let mmap = mmap_file_read(&path).unwrap();
    let response = plan_query_with_tet_mmap(&doc, None, &mmap, Some(0)).unwrap();
    let ex = response.execution.as_ref().unwrap();
    assert_eq!(ex.device_requested.as_deref(), Some("metal"));
    assert_eq!(ex.device_used, Some("cpu"));
    assert_eq!(
        ex.device_fallback_reason.as_deref(),
        Some("gpu_feature_disabled")
    );
}

#[cfg(all(feature = "tetration-metal", target_os = "macos"))]
#[test]
fn resolve_device_auto_large_prefers_metal_when_enabled() {
    use crate::query::GPU_AUTO_MIN_LOGICAL_BYTES;
    use crate::query::types::{ChunkTouchPolicy, ReadPlan};

    let doc = crate::query::parse_query_json(r#"{"dataset":"a","mean":[]}"#).unwrap();
    validate_query(&doc).unwrap();
    let elems = usize::try_from(GPU_AUTO_MIN_LOGICAL_BYTES / 4 + 1).unwrap();
    let plan = ReadPlan {
        chunk_touch_policy: ChunkTouchPolicy::DenseHalfOpenUnitStep,
        chunk_count: 0,
        zone_filter_skipped_chunks: 0,
        index_filter_skipped_chunks: 0,
        total_stored_bytes: 0,
        chunks: vec![],
        dataset_shape: vec![elems as u64],
        chunk_shape: vec![elems as u64],
        selection_box_start: vec![0],
        selection_box_stop_exclusive: vec![elems as u64],
        selection_step: vec![1],
        logical_selection_shape: vec![elems as u64],
        logical_element_count: elems,
        cell_filter: None,
        label_eq: None,
        label_group_remap: None,
        dataset_fill: None,
    };
    let route = resolve_device_route(
        Some(&crate::query::ExecutionHints {
            device: Some(ExecutionDeviceHint::Auto),
            ..Default::default()
        }),
        &plan,
        ElementDtype::F32,
        doc.operation.as_ref(),
    );
    assert_eq!(route.used, "metal");
    assert!(route.gpu_reduce);
}

#[test]
fn resolve_device_auto_huge_selection_still_routes_gpu_for_streaming_fold() {
    use crate::query::types::{ChunkTouchPolicy, ReadPlan};
    use crate::utils::host_memory;

    if !metal_backend_available() && !cuda_backend_available() {
        return;
    }

    let gib = 1024_u64.pow(3);
    let logical_bytes = 20 * gib;
    if !host_materialize_exceeds(logical_bytes, host_memory::available_memory_bytes()) {
        return;
    }

    let doc = crate::query::parse_query_json(r#"{"dataset":"a","mean":[]}"#).unwrap();
    let elems = (logical_bytes / 4) as usize;
    let plan = ReadPlan {
        chunk_touch_policy: ChunkTouchPolicy::DenseHalfOpenUnitStep,
        chunk_count: 0,
        zone_filter_skipped_chunks: 0,
        index_filter_skipped_chunks: 0,
        total_stored_bytes: 0,
        chunks: vec![],
        dataset_shape: vec![elems as u64],
        chunk_shape: vec![elems as u64],
        selection_box_start: vec![0],
        selection_box_stop_exclusive: vec![elems as u64],
        selection_step: vec![1],
        logical_selection_shape: vec![elems as u64],
        logical_element_count: elems,
        cell_filter: None,
        label_eq: None,
        label_group_remap: None,
        dataset_fill: None,
    };
    let route = resolve_device_route(
        Some(&crate::query::ExecutionHints {
            device: Some(ExecutionDeviceHint::Auto),
            ..Default::default()
        }),
        &plan,
        ElementDtype::F32,
        doc.operation.as_ref(),
    );
    assert!(route.gpu_reduce);
    assert!(
        route.used == "metal" || route.used == "cuda",
        "expected GPU backend, got {}",
        route.used
    );
    assert!(route.fallback_reason.is_none());
}
