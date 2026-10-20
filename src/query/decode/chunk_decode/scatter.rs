//! Scatter planned chunk payloads into logical float / integer buffers.

use crate::query::types::{
    CellPredicate, INTEGER_FILTER_FILL_MSG, PlannedChunkIo, ReadPlan, TetError, fold_keeps_cell,
};

use super::elem::ChunkElem;
use super::visit::visit_planned_chunk_elem;

trait FloatScatter: ChunkElem {
    fn to_f64(self) -> f64;
    fn filtered_mask() -> Self;
}

impl FloatScatter for f32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn filtered_mask() -> Self {
        f32::NAN
    }
}

impl FloatScatter for f64 {
    fn to_f64(self) -> f64 {
        self
    }
    fn filtered_mask() -> Self {
        f64::NAN
    }
}

impl FloatScatter for half::f16 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn filtered_mask() -> Self {
        half::f16::NAN
    }
}

trait IntScatter: ChunkElem {
    fn to_f64(self) -> f64;
    fn from_fill(fill: f64) -> Result<Self, TetError>;
}

fn int_from_fill<T, F>(fill: f64, cast: F, roundtrip: impl Fn(T) -> f64) -> Result<T, TetError>
where
    T: Copy,
    F: FnOnce(f64) -> T,
{
    if !fill.is_finite() {
        return Err(TetError::Validation(
            "integer materialize fill must be a finite integer".into(),
        ));
    }
    let v = cast(fill);
    if roundtrip(v).to_bits() != fill.to_bits() {
        return Err(TetError::Validation(
            "integer materialize fill is outside this dtype".into(),
        ));
    }
    Ok(v)
}

impl IntScatter for i32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as i32, f64::from)
    }
}

impl IntScatter for i64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as i64, |v| v as f64)
    }
}

impl IntScatter for i16 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as i16, f64::from)
    }
}

impl IntScatter for u8 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as u8, f64::from)
    }
}

impl IntScatter for u16 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as u16, f64::from)
    }
}

impl IntScatter for u32 {
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as u32, f64::from)
    }
}

impl IntScatter for u64 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_fill(fill: f64) -> Result<Self, TetError> {
        int_from_fill(fill, |v| v as u64, |v| v as f64)
    }
}

fn integer_filter_mask<E: IntScatter>(pred: CellPredicate) -> Result<E, TetError> {
    let Some(fill) = pred.fill_f64() else {
        return Err(TetError::Validation(INTEGER_FILTER_FILL_MSG.into()));
    };
    E::from_fill(fill)
}

fn scatter_chunk_into<E>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [E],
) -> Result<u64, TetError>
where
    E: FloatScatter,
{
    visit_planned_chunk_elem::<E, _>(mmap, plan, c, |li, v| {
        if li < out.len() {
            out[li] = if fold_keeps_cell(plan.cell_filter, v.to_f64()) {
                v
            } else {
                E::filtered_mask()
            };
        }
        Ok(())
    })
}

fn scatter_chunk_into_option<E>(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<E>],
) -> Result<u64, TetError>
where
    E: IntScatter,
{
    let mask = plan.cell_filter.map(integer_filter_mask::<E>).transpose()?;
    visit_planned_chunk_elem::<E, _>(mmap, plan, c, |li, v| {
        if li < out.len() {
            out[li] = Some(match mask {
                Some(masked) if !fold_keeps_cell(plan.cell_filter, v.to_f64()) => masked,
                _ => v,
            });
        }
        Ok(())
    })
}

pub(crate) fn scatter_chunk_into_plan(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [f32],
) -> Result<u64, TetError> {
    scatter_chunk_into::<f32>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_f64(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [f64],
) -> Result<u64, TetError> {
    scatter_chunk_into::<f64>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_i32(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<i32>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<i32>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_i64(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<i64>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<i64>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_u8(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<u8>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<u8>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_u16(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<u16>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<u16>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_i16(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<i16>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<i16>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_u32(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<u32>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<u32>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_u64(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [Option<u64>],
) -> Result<u64, TetError> {
    scatter_chunk_into_option::<u64>(mmap, plan, c, out)
}

pub(crate) fn scatter_chunk_into_plan_f16(
    mmap: &[u8],
    plan: &ReadPlan,
    c: &PlannedChunkIo,
    out: &mut [half::f16],
) -> Result<u64, TetError> {
    scatter_chunk_into::<half::f16>(mmap, plan, c, out)
}
