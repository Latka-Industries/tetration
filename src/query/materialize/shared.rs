//! Shared materialize helpers (preview mmap, dense POD spill, capped decode).

use std::path::Path;

use memmap2::{Mmap, MmapMut};

use crate::query::dispatch::accumulate_chunk_read_bytes;
use crate::query::types::{ReadPlan, TetError};

use super::validate::{validate_full_read_plan_buffer, validate_read_plan_geometry};

pub(crate) const UNSET_MATERIALIZED_MSG: &str =
    "materialized selection has unset elements (chunk payloads vs selection mismatch)";

pub(crate) fn mmap_spill(path: &Path) -> Result<Mmap, TetError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TetError::Validation(format!("temp spill read failed: {e}")))?;
    unsafe {
        Mmap::map(&file).map_err(|e| TetError::Validation(format!("temp spill mmap failed: {e}")))
    }
}

pub(crate) fn preview_from_backing_in_memory<T: Copy>(
    v: &[T],
    logical_len: usize,
    max: usize,
) -> (Vec<T>, bool) {
    let cap = max.min(logical_len);
    if cap == 0 {
        return (Vec::new(), logical_len > 0);
    }
    (v[..cap].to_vec(), logical_len > max)
}

pub(crate) fn preview_from_spill_file_pod<T: bytemuck::Pod + Copy>(
    path: &Path,
    cap: usize,
    logical_len: usize,
) -> Result<(Vec<T>, bool), TetError> {
    let mmap = mmap_spill(path)?;
    let slice: &[T] = bytemuck::cast_slice(&mmap);
    if slice.len() < cap {
        return Err(TetError::Validation(
            "temp spill shorter than logical selection".into(),
        ));
    }
    Ok((slice[..cap].to_vec(), logical_len > cap))
}

pub(crate) fn check_materialized_nan_slice<T: Copy>(
    out: &[T],
    is_unset: fn(T) -> bool,
) -> Result<(), TetError> {
    if out.iter().any(|&v| is_unset(v)) {
        return Err(TetError::Validation(UNSET_MATERIALIZED_MSG.into()));
    }
    Ok(())
}

pub(crate) fn materialize_read_plan_pod_le_core<T, F>(
    mmap: &[u8],
    plan: &ReadPlan,
    max_elements: Option<usize>,
    scatter_fill: F,
    unset: T,
    check_complete: fn(&[T]) -> Result<(), TetError>,
) -> Result<(Vec<T>, bool, u64), TetError>
where
    T: Copy,
    F: Fn(&[u8], &ReadPlan, &mut [T]) -> Result<u64, TetError>,
{
    if matches!(max_elements, Some(0)) {
        return Ok((Vec::new(), false, 0));
    }
    let n = plan.logical_element_count;
    let cap = max_elements.map(|m| m.min(n));
    let (buf_len, truncated) = match cap {
        Some(c) if c < n => (c, true),
        _ => (n, max_elements.is_some_and(|m| m < n)),
    };
    let mut out = vec![unset; buf_len];
    let total_bytes_read_from_disk = scatter_fill(mmap, plan, &mut out)?;
    check_complete(&out)?;
    Ok((out, truncated, total_bytes_read_from_disk))
}

pub(crate) fn scatter_fill_chunks<T, S>(
    mmap: &[u8],
    plan: &ReadPlan,
    out: &mut [T],
    scatter_chunk: S,
) -> Result<u64, TetError>
where
    S: Fn(
        &[u8],
        &ReadPlan,
        &crate::query::types::PlannedChunkIo,
        &mut [T],
    ) -> Result<u64, TetError>,
{
    validate_read_plan_geometry(plan, out.len())?;
    let mut total_bytes_read_from_disk: u64 = 0;
    for c in &plan.chunks {
        let n = scatter_chunk(mmap, plan, c, out)?;
        accumulate_chunk_read_bytes(&mut total_bytes_read_from_disk, n)?;
    }
    Ok(total_bytes_read_from_disk)
}

pub(crate) fn spill_byte_len_from_elem_count(
    elem_count: usize,
    bytes_from_elem_count: fn(u64) -> Option<u64>,
) -> Result<u64, TetError> {
    let n = u64::try_from(elem_count)
        .map_err(|_| TetError::Validation("logical element count overflow".into()))?;
    bytes_from_elem_count(n)
        .ok_or_else(|| TetError::Validation("spill byte length overflow".into()))
}

type MaterializedCheckFn<T> = fn(&[T]) -> Result<(), TetError>;

pub(crate) fn spill_read_plan_pod_le_impl<T, F>(
    mmap: &[u8],
    plan: &ReadPlan,
    path: &Path,
    byte_len: u64,
    scatter_fill: F,
    check_complete: Option<MaterializedCheckFn<T>>,
) -> Result<u64, TetError>
where
    T: bytemuck::Pod + Copy,
    F: Fn(&[u8], &ReadPlan, &mut [T]) -> Result<u64, TetError>,
{
    use std::fs::OpenOptions;
    use std::io::Write;

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(|e| TetError::Validation(format!("spill open failed: {e}")))?;
    file.set_len(byte_len)
        .map_err(|e| TetError::Validation(format!("spill set_len failed: {e}")))?;
    file.flush()
        .map_err(|e| TetError::Validation(format!("spill flush failed: {e}")))?;
    let mut out_mmap = unsafe {
        MmapMut::map_mut(&file)
            .map_err(|e| TetError::Validation(format!("spill mmap failed: {e}")))?
    };
    let out = bytemuck::cast_slice_mut(out_mmap.as_mut());
    validate_full_read_plan_buffer(plan, out.len())?;
    let total = scatter_fill(mmap, plan, out)?;
    if let Some(check) = check_complete {
        check(out)?;
    }
    out_mmap
        .flush()
        .map_err(|e| TetError::Validation(format!("spill mmap flush failed: {e}")))?;
    Ok(total)
}

pub(crate) fn materialize_into_vec_dispatch<T, F, G>(
    mmap: &[u8],
    plan: &ReadPlan,
    sequential: F,
    parallel: G,
) -> Result<(T, u64), TetError>
where
    F: Fn(&[u8], &ReadPlan, Option<usize>) -> Result<(T, bool, u64), TetError>,
    G: Fn(&[u8], &ReadPlan, Option<usize>) -> Result<(T, bool, u64), TetError>,
{
    let read = if plan.chunks.len() <= 1 {
        sequential(mmap, plan, None)
    } else {
        parallel(mmap, plan, None)
    };
    read.map(|(v, truncated, bytes)| {
        debug_assert!(!truncated);
        (v, bytes)
    })
}
