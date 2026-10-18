//! Parallel chunk decode + scatter (Rayon).

use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use crate::query::types::{ReadPlan, TetError};

use super::int::{
    materialize_read_plan_i16_le_core, materialize_read_plan_i32_le_core,
    materialize_read_plan_i64_le_core, materialize_read_plan_u8_le_core,
    materialize_read_plan_u16_le_core,
};
use super::{
    MaterializeReadPlanF32IntoOutcome, materialize_read_plan_f32_le_core,
    materialize_read_plan_f32_le_into_core, materialize_read_plan_f64_le_core,
    validate_read_plan_geometry,
};
use crate::query::decode::chunk_decode::{
    scatter_chunk_into_plan, scatter_chunk_into_plan_f64, scatter_chunk_into_plan_i16,
    scatter_chunk_into_plan_i32, scatter_chunk_into_plan_i64, scatter_chunk_into_plan_u8,
    scatter_chunk_into_plan_u16,
};

pub(crate) fn materialize_scatter_fill_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f32],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    if plan.chunks.is_empty() {
        return Ok(0);
    }
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let total_bytes = AtomicU64::new(0);
    plan.chunks.par_iter().try_for_each(|c| {
        // SAFETY: Each planned chunk covers disjoint global coordinates; logical row-major
        // indices written per chunk do not overlap.
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut f32, out_len) };
        let n = scatter_chunk_into_plan(mmap, plan, c, out)?;
        total_bytes.fetch_add(n, Ordering::Relaxed);
        Ok::<(), TetError>(())
    })?;
    Ok(total_bytes.load(Ordering::Relaxed))
}

/// Like [`crate::query::materialize::materialize_read_plan_f32_le`], but decodes planned chunks in parallel.
///
/// # Errors
///
/// Same failure modes as [`crate::query::materialize::materialize_read_plan_f32_le`].
pub fn materialize_read_plan_f32_le_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f32>, bool, u64), TetError> {
    materialize_read_plan_f32_le_core(mmap, plan, max_elements, materialize_scatter_fill_parallel)
}

/// Like [`crate::query::materialize::materialize_read_plan_f32_le_into`], but decodes planned chunks in parallel.
///
/// # Errors
///
/// Same failure modes as [`crate::query::materialize::materialize_read_plan_f32_le_into`].
pub fn materialize_read_plan_f32_le_into_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    dst: &mut [f32],
) -> Result<MaterializeReadPlanF32IntoOutcome, TetError> {
    materialize_read_plan_f32_le_into_core(
        mmap,
        plan,
        max_elements,
        dst,
        materialize_scatter_fill_parallel,
    )
}

pub(crate) fn materialize_scatter_fill_parallel_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [f64],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    if plan.chunks.is_empty() {
        return Ok(0);
    }
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let total_bytes = AtomicU64::new(0);
    plan.chunks.par_iter().try_for_each(|c| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut f64, out_len) };
        let n = scatter_chunk_into_plan_f64(mmap, plan, c, out)?;
        total_bytes.fetch_add(n, Ordering::Relaxed);
        Ok::<(), TetError>(())
    })?;
    Ok(total_bytes.load(Ordering::Relaxed))
}

/// Like [`crate::query::materialize::materialize_read_plan_f64_le`], but decodes planned chunks in parallel.
pub fn materialize_read_plan_f64_le_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<f64>, bool, u64), TetError> {
    materialize_read_plan_f64_le_core(
        mmap,
        plan,
        max_elements,
        materialize_scatter_fill_parallel_f64,
    )
}

pub(crate) fn materialize_scatter_fill_parallel_i32(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [Option<i32>],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    if plan.chunks.is_empty() {
        return Ok(0);
    }
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let total_bytes = AtomicU64::new(0);
    plan.chunks.par_iter().try_for_each(|c| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut Option<i32>, out_len) };
        let n = scatter_chunk_into_plan_i32(mmap, plan, c, out)?;
        total_bytes.fetch_add(n, Ordering::Relaxed);
        Ok::<(), TetError>(())
    })?;
    Ok(total_bytes.load(Ordering::Relaxed))
}

pub(crate) fn materialize_scatter_fill_parallel_i64(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [Option<i64>],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    if plan.chunks.is_empty() {
        return Ok(0);
    }
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let total_bytes = AtomicU64::new(0);
    plan.chunks.par_iter().try_for_each(|c| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut Option<i64>, out_len) };
        let n = scatter_chunk_into_plan_i64(mmap, plan, c, out)?;
        total_bytes.fetch_add(n, Ordering::Relaxed);
        Ok::<(), TetError>(())
    })?;
    Ok(total_bytes.load(Ordering::Relaxed))
}

/// Like [`crate::query::materialize::materialize_read_plan_i32_le`], but decodes planned chunks in parallel.
///
/// # Errors
///
/// Same failure modes as [`crate::query::materialize::materialize_read_plan_i32_le`].
pub fn materialize_read_plan_i32_le_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<i32>, bool, u64), TetError> {
    materialize_read_plan_i32_le_core(
        mmap,
        plan,
        max_elements,
        materialize_scatter_fill_parallel_i32,
    )
}

/// Like [`crate::query::materialize::materialize_read_plan_i64_le`], but decodes planned chunks in parallel.
///
/// # Errors
///
/// Same failure modes as [`crate::query::materialize::materialize_read_plan_i64_le`].
pub fn materialize_read_plan_i64_le_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<i64>, bool, u64), TetError> {
    materialize_read_plan_i64_le_core(
        mmap,
        plan,
        max_elements,
        materialize_scatter_fill_parallel_i64,
    )
}

pub(crate) fn materialize_scatter_fill_parallel_u8(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [Option<u8>],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    if plan.chunks.is_empty() {
        return Ok(0);
    }
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let total_bytes = AtomicU64::new(0);
    plan.chunks.par_iter().try_for_each(|c| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut Option<u8>, out_len) };
        let n = scatter_chunk_into_plan_u8(mmap, plan, c, out)?;
        total_bytes.fetch_add(n, Ordering::Relaxed);
        Ok::<(), TetError>(())
    })?;
    Ok(total_bytes.load(Ordering::Relaxed))
}

/// Like [`crate::query::materialize::materialize_read_plan_u8_le`], but decodes planned chunks in parallel.
pub fn materialize_read_plan_u8_le_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<u8>, bool, u64), TetError> {
    materialize_read_plan_u8_le_core(
        mmap,
        plan,
        max_elements,
        materialize_scatter_fill_parallel_u8,
    )
}

pub(crate) fn materialize_scatter_fill_parallel_u16(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [Option<u16>],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    if plan.chunks.is_empty() {
        return Ok(0);
    }
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let total_bytes = AtomicU64::new(0);
    plan.chunks.par_iter().try_for_each(|c| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut Option<u16>, out_len) };
        let n = scatter_chunk_into_plan_u16(mmap, plan, c, out)?;
        total_bytes.fetch_add(n, Ordering::Relaxed);
        Ok::<(), TetError>(())
    })?;
    Ok(total_bytes.load(Ordering::Relaxed))
}

/// Like [`crate::query::materialize::materialize_read_plan_u16_le`], but decodes planned chunks in parallel.
pub fn materialize_read_plan_u16_le_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<u16>, bool, u64), TetError> {
    materialize_read_plan_u16_le_core(
        mmap,
        plan,
        max_elements,
        materialize_scatter_fill_parallel_u16,
    )
}

pub(crate) fn materialize_scatter_fill_parallel_i16(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [Option<i16>],
) -> Result<u64, TetError> {
    validate_read_plan_geometry(plan, out.len())?;
    if plan.chunks.is_empty() {
        return Ok(0);
    }
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let total_bytes = AtomicU64::new(0);
    plan.chunks.par_iter().try_for_each(|c| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut Option<i16>, out_len) };
        let n = scatter_chunk_into_plan_i16(mmap, plan, c, out)?;
        total_bytes.fetch_add(n, Ordering::Relaxed);
        Ok::<(), TetError>(())
    })?;
    Ok(total_bytes.load(Ordering::Relaxed))
}

/// Like [`crate::query::materialize::materialize_read_plan_i16_le`], but decodes planned chunks in parallel.
pub fn materialize_read_plan_i16_le_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<i16>, bool, u64), TetError> {
    materialize_read_plan_i16_le_core(
        mmap,
        plan,
        max_elements,
        materialize_scatter_fill_parallel_i16,
    )
}

pub(crate) fn materialize_scatter_fill_parallel_u32(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [Option<u32>],
) -> Result<u64, TetError> {
    use crate::query::decode::chunk_decode::scatter_chunk_into_plan_u32;

    validate_read_plan_geometry(plan, out.len())?;
    if plan.chunks.is_empty() {
        return Ok(0);
    }
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let total_bytes = AtomicU64::new(0);
    plan.chunks.par_iter().try_for_each(|c| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut Option<u32>, out_len) };
        let n = scatter_chunk_into_plan_u32(mmap, plan, c, out)?;
        total_bytes.fetch_add(n, Ordering::Relaxed);
        Ok::<(), TetError>(())
    })?;
    Ok(total_bytes.load(Ordering::Relaxed))
}

pub fn materialize_read_plan_u32_le_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<u32>, bool, u64), TetError> {
    crate::query::materialize::int::materialize_read_plan_u32_le_core(
        mmap,
        plan,
        max_elements,
        materialize_scatter_fill_parallel_u32,
    )
}

pub(crate) fn materialize_scatter_fill_parallel_u64(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [Option<u64>],
) -> Result<u64, TetError> {
    use crate::query::decode::chunk_decode::scatter_chunk_into_plan_u64;

    validate_read_plan_geometry(plan, out.len())?;
    if plan.chunks.is_empty() {
        return Ok(0);
    }
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let total_bytes = AtomicU64::new(0);
    plan.chunks.par_iter().try_for_each(|c| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut Option<u64>, out_len) };
        let n = scatter_chunk_into_plan_u64(mmap, plan, c, out)?;
        total_bytes.fetch_add(n, Ordering::Relaxed);
        Ok::<(), TetError>(())
    })?;
    Ok(total_bytes.load(Ordering::Relaxed))
}

pub fn materialize_read_plan_u64_le_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<u64>, bool, u64), TetError> {
    crate::query::materialize::int::materialize_read_plan_u64_le_core(
        mmap,
        plan,
        max_elements,
        materialize_scatter_fill_parallel_u64,
    )
}

pub(crate) fn materialize_scatter_fill_parallel_f16(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [half::f16],
) -> Result<u64, TetError> {
    use crate::query::decode::chunk_decode::scatter_chunk_into_plan_f16;

    validate_read_plan_geometry(plan, out.len())?;
    if plan.chunks.is_empty() {
        return Ok(0);
    }
    let out_addr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let total_bytes = AtomicU64::new(0);
    plan.chunks.par_iter().try_for_each(|c| {
        let out = unsafe { std::slice::from_raw_parts_mut(out_addr as *mut half::f16, out_len) };
        let n = scatter_chunk_into_plan_f16(mmap, plan, c, out)?;
        total_bytes.fetch_add(n, Ordering::Relaxed);
        Ok::<(), TetError>(())
    })?;
    Ok(total_bytes.load(Ordering::Relaxed))
}

pub fn materialize_read_plan_f16_le_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
) -> Result<(Vec<half::f16>, bool, u64), TetError> {
    super::float::materialize_read_plan_f16_le_core(
        mmap,
        plan,
        max_elements,
        materialize_scatter_fill_parallel_f16,
    )
}
