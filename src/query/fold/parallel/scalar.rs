//! Parallel scalar (full-selection) fold over planned chunks.

use rayon::prelude::*;

use super::merge::{ScalarChunkWork, merge_scalar_chunks, sum_chunk_bytes};
use super::preview::{
    disjoint_preview_mut, write_disjoint_preview_f16, write_disjoint_preview_f32,
    write_disjoint_preview_f64, write_disjoint_preview_i16, write_disjoint_preview_i32,
    write_disjoint_preview_i64, write_disjoint_preview_u8, write_disjoint_preview_u16,
    write_disjoint_preview_u32, write_disjoint_preview_u64,
};
use super::workers::with_fold_workers;
use crate::query::decode;
use crate::query::fold::{reduction, shared};
use crate::query::materialize::int::IntVisit;
use crate::query::types::{OperationPreviewFields, ReadPlan, TetError, fold_keeps_element};
use crate::utils::dtype::ElementDtype;

#[derive(Copy, Clone)]
enum ScalarFoldVisit {
    F32,
    F64,
    F16,
    Int(IntVisit),
}

struct ScalarFoldChunkCtx<'a> {
    kind: reduction::ReductionKind,
    value: &'a mut reduction::ValueAccum,
    arg: &'a mut reduction::ArgIndexAccum,
    preview_addr: usize,
    preview_len: usize,
}

impl ScalarFoldVisit {
    fn fold_chunk(
        self,
        mmap: &[u8],
        plan: &ReadPlan,
        c: &crate::query::types::PlannedChunkIo,
        ctx: &mut ScalarFoldChunkCtx<'_>,
    ) -> Result<u64, TetError> {
        match self {
            Self::F32 => decode::chunk_decode::fold_planned_chunk_f32(
                mmap,
                plan,
                c,
                ctx.kind,
                ctx.value,
                ctx.arg,
                disjoint_preview_mut(ctx.preview_addr, ctx.preview_len),
            ),
            Self::F64 => decode::chunk_decode::fold_planned_chunk_f64(
                mmap,
                plan,
                c,
                ctx.kind,
                ctx.value,
                ctx.arg,
                disjoint_preview_mut(ctx.preview_addr, ctx.preview_len),
            ),
            Self::F16 | Self::Int(_) => self.visit_chunk(mmap, plan, c, |li, val| {
                self.write_preview(ctx.preview_addr, ctx.preview_len, li, val);
                if !fold_keeps_element(plan, li, val) {
                    return Ok(());
                }
                match ctx.kind {
                    reduction::ReductionKind::ArgMin | reduction::ReductionKind::ArgMax => {
                        self.push_arg(ctx.arg, li, val, ctx.kind);
                    }
                    reduction::ReductionKind::NanCount => ctx.value.push_nan_f64(val),
                    reduction::ReductionKind::InfCount => ctx.value.push_inf_f64(val),
                    reduction::ReductionKind::NullCount { fill } => {
                        ctx.value.push_null_f64(val, fill);
                    }
                    _ => self.push_value(ctx.value, val),
                }
                Ok(())
            }),
        }
    }

    fn visit_chunk<F>(
        self,
        mmap: &[u8],
        plan: &ReadPlan,
        c: &crate::query::types::PlannedChunkIo,
        mut visit: F,
    ) -> Result<u64, TetError>
    where
        F: FnMut(usize, f64) -> Result<(), TetError>,
    {
        match self {
            Self::F32 => decode::chunk_decode::visit_planned_chunk(mmap, plan, c, |li, v| {
                visit(li, f64::from(v))
            }),
            Self::F64 => decode::chunk_decode::visit_planned_chunk_f64(mmap, plan, c, visit),
            Self::F16 => decode::chunk_decode::visit_planned_chunk_f16(mmap, plan, c, |li, v| {
                visit(li, f64::from(v))
            }),
            Self::Int(v) => v.visit_chunk_as_f64(mmap, plan, c, visit),
        }
    }

    fn push_value(self, acc: &mut reduction::ValueAccum, v: f64) {
        match self {
            Self::F32 => acc.push(v as f32),
            Self::F64 | Self::F16 | Self::Int(_) => acc.push_f64(v),
        }
    }

    fn push_arg(
        self,
        acc: &mut reduction::ArgIndexAccum,
        li: usize,
        v: f64,
        kind: reduction::ReductionKind,
    ) {
        match self {
            Self::F32 => acc.push(li as u64, v as f32, kind),
            Self::F64 | Self::F16 | Self::Int(_) => acc.push_f64(li as u64, v, kind),
        }
    }

    fn write_preview(self, preview_addr: usize, preview_len: usize, li: usize, v: f64) {
        match self {
            Self::F32 => write_disjoint_preview_f32(preview_addr, preview_len, li, v as f32),
            Self::F64 => write_disjoint_preview_f64(preview_addr, preview_len, li, v),
            Self::F16 => write_disjoint_preview_f16(preview_addr, preview_len, li, v),
            Self::Int(IntVisit::I32) => {
                write_disjoint_preview_i32(preview_addr, preview_len, li, v);
            }
            Self::Int(IntVisit::I64) => {
                write_disjoint_preview_i64(preview_addr, preview_len, li, v);
            }
            Self::Int(IntVisit::U8) => {
                write_disjoint_preview_u8(preview_addr, preview_len, li, v);
            }
            Self::Int(IntVisit::U16) => {
                write_disjoint_preview_u16(preview_addr, preview_len, li, v);
            }
            Self::Int(IntVisit::I16) => {
                write_disjoint_preview_i16(preview_addr, preview_len, li, v);
            }
            Self::Int(IntVisit::U32) => {
                write_disjoint_preview_u32(preview_addr, preview_len, li, v);
            }
            Self::Int(IntVisit::U64) => {
                write_disjoint_preview_u64(preview_addr, preview_len, li, v);
            }
        }
    }

    fn finish_parallel_scalar_fold(
        self,
        previews: ParallelScalarFoldPreviews,
        max_preview: usize,
        n: usize,
        total_bytes_read_from_disk: u64,
        operation: OperationPreviewFields,
    ) -> Result<shared::FoldPlanOutcome, TetError> {
        let preview_cap = previews.preview_cap;
        match self {
            Self::F32 => {
                shared::validate_fold_preview(true, &previews.f32, preview_cap)?;
                Ok(shared::build_fold_plan_outcome(
                    previews.f32,
                    max_preview,
                    n,
                    total_bytes_read_from_disk,
                    operation,
                ))
            }
            Self::F64 => {
                shared::validate_fold_preview_f64(true, &previews.f64, preview_cap)?;
                Ok(shared::build_fold_plan_outcome_typed(
                    shared::FoldPreviewBuffer::F64(previews.f64),
                    max_preview,
                    n,
                    total_bytes_read_from_disk,
                    operation,
                ))
            }
            Self::F16 => {
                shared::validate_fold_preview_f16(true, &previews.f16, preview_cap)?;
                Ok(shared::build_fold_plan_outcome_typed(
                    shared::FoldPreviewBuffer::F16(previews.f16),
                    max_preview,
                    n,
                    total_bytes_read_from_disk,
                    operation,
                ))
            }
            Self::Int(IntVisit::I32) => Ok(shared::build_fold_plan_outcome_typed(
                shared::FoldPreviewBuffer::I32(previews.i32),
                max_preview,
                n,
                total_bytes_read_from_disk,
                operation,
            )),
            Self::Int(IntVisit::I64) => Ok(shared::build_fold_plan_outcome_typed(
                shared::FoldPreviewBuffer::I64(previews.i64),
                max_preview,
                n,
                total_bytes_read_from_disk,
                operation,
            )),
            Self::Int(IntVisit::U8) => Ok(shared::build_fold_plan_outcome_typed(
                shared::FoldPreviewBuffer::U8(previews.u8),
                max_preview,
                n,
                total_bytes_read_from_disk,
                operation,
            )),
            Self::Int(IntVisit::U16) => Ok(shared::build_fold_plan_outcome_typed(
                shared::FoldPreviewBuffer::U16(previews.u16),
                max_preview,
                n,
                total_bytes_read_from_disk,
                operation,
            )),
            Self::Int(IntVisit::I16) => Ok(shared::build_fold_plan_outcome_typed(
                shared::FoldPreviewBuffer::I16(previews.i16),
                max_preview,
                n,
                total_bytes_read_from_disk,
                operation,
            )),
            Self::Int(IntVisit::U32) => Ok(shared::build_fold_plan_outcome_typed(
                shared::FoldPreviewBuffer::U32(previews.u32),
                max_preview,
                n,
                total_bytes_read_from_disk,
                operation,
            )),
            Self::Int(IntVisit::U64) => Ok(shared::build_fold_plan_outcome_typed(
                shared::FoldPreviewBuffer::U64(previews.u64),
                max_preview,
                n,
                total_bytes_read_from_disk,
                operation,
            )),
        }
    }
}

struct ParallelScalarFoldPreviews {
    preview_cap: usize,
    f32: Vec<f32>,
    f64: Vec<f64>,
    f16: Vec<half::f16>,
    i32: Vec<i32>,
    i64: Vec<i64>,
    u8: Vec<u8>,
    u16: Vec<u16>,
    i16: Vec<i16>,
    u32: Vec<u32>,
    u64: Vec<u64>,
}

impl ParallelScalarFoldPreviews {
    fn new(preview_cap: usize, visit: ScalarFoldVisit) -> (Self, usize, usize) {
        let mut previews = Self {
            preview_cap,
            f32: vec![f32::NAN; preview_cap],
            f64: vec![f64::NAN; preview_cap],
            f16: vec![half::f16::NAN; preview_cap],
            i32: vec![0; preview_cap],
            i64: vec![0; preview_cap],
            u8: vec![0u8; preview_cap],
            u16: vec![0u16; preview_cap],
            i16: vec![0i16; preview_cap],
            u32: vec![0u32; preview_cap],
            u64: vec![0u64; preview_cap],
        };
        let (preview_addr, preview_len) = match visit {
            ScalarFoldVisit::F32 => (previews.f32.as_mut_ptr() as usize, previews.f32.len()),
            ScalarFoldVisit::F64 => (previews.f64.as_mut_ptr() as usize, previews.f64.len()),
            ScalarFoldVisit::F16 => (previews.f16.as_mut_ptr() as usize, previews.f16.len()),
            ScalarFoldVisit::Int(IntVisit::I32) => {
                (previews.i32.as_mut_ptr() as usize, previews.i32.len())
            }
            ScalarFoldVisit::Int(IntVisit::I64) => {
                (previews.i64.as_mut_ptr() as usize, previews.i64.len())
            }
            ScalarFoldVisit::Int(IntVisit::U8) => {
                (previews.u8.as_mut_ptr() as usize, previews.u8.len())
            }
            ScalarFoldVisit::Int(IntVisit::U16) => {
                (previews.u16.as_mut_ptr() as usize, previews.u16.len())
            }
            ScalarFoldVisit::Int(IntVisit::I16) => {
                (previews.i16.as_mut_ptr() as usize, previews.i16.len())
            }
            ScalarFoldVisit::Int(IntVisit::U32) => {
                (previews.u32.as_mut_ptr() as usize, previews.u32.len())
            }
            ScalarFoldVisit::Int(IntVisit::U64) => {
                (previews.u64.as_mut_ptr() as usize, previews.u64.len())
            }
        };
        (previews, preview_addr, preview_len)
    }
}

fn parallel_scalar_fold_chunks(
    mmap: &[u8],
    plan: &ReadPlan,
    kind: reduction::ReductionKind,
    visit: ScalarFoldVisit,
    preview_addr: usize,
    preview_len: usize,
    workers: Option<usize>,
) -> Result<Vec<ScalarChunkWork>, TetError> {
    with_fold_workers(workers, || {
        plan.chunks
            .par_iter()
            .map(|c| {
                let mut value = reduction::ValueAccum::default();
                let mut arg = reduction::ArgIndexAccum::default();
                let mut ctx = ScalarFoldChunkCtx {
                    kind,
                    value: &mut value,
                    arg: &mut arg,
                    preview_addr,
                    preview_len,
                };
                let bytes = visit.fold_chunk(mmap, plan, c, &mut ctx)?;
                Ok(ScalarChunkWork { bytes, value, arg })
            })
            .collect::<Result<Vec<_>, TetError>>()
    })
}

fn fold_read_plan_scalar_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    visit: ScalarFoldVisit,
    workers: Option<usize>,
) -> Result<shared::FoldPlanOutcome, TetError> {
    let n = plan.logical_element_count;
    let preview_cap = max_preview.min(n);
    let (previews, preview_addr, preview_len) = ParallelScalarFoldPreviews::new(preview_cap, visit);
    let parts =
        parallel_scalar_fold_chunks(mmap, plan, kind, visit, preview_addr, preview_len, workers)?;
    let total_bytes_read_from_disk = sum_chunk_bytes(parts.iter().map(|p| p.bytes))?;
    let operation = merge_scalar_chunks(
        &parts,
        kind,
        n,
        plan.cell_filter.is_some() || plan.label_eq.is_some(),
    )?;
    visit.finish_parallel_scalar_fold(
        previews,
        max_preview,
        n,
        total_bytes_read_from_disk,
        operation,
    )
}

/// Parallel scalar fold over `f32` planned chunks.
pub(crate) fn fold_read_plan_scalar_operation_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_f32: usize,
    kind: reduction::ReductionKind,
    workers: Option<usize>,
) -> Result<shared::FoldPlanOutcome, TetError> {
    fold_read_plan_scalar_parallel(mmap, plan, max_f32, kind, ScalarFoldVisit::F32, workers)
}

/// Parallel scalar fold over `f64` planned chunks.
pub(crate) fn fold_read_plan_scalar_operation_f64_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    workers: Option<usize>,
) -> Result<shared::FoldPlanOutcome, TetError> {
    fold_read_plan_scalar_parallel(mmap, plan, max_preview, kind, ScalarFoldVisit::F64, workers)
}

/// Parallel scalar fold over `f16` planned chunks.
pub(crate) fn fold_read_plan_scalar_operation_f16_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    workers: Option<usize>,
) -> Result<shared::FoldPlanOutcome, TetError> {
    fold_read_plan_scalar_parallel(mmap, plan, max_preview, kind, ScalarFoldVisit::F16, workers)
}

/// Parallel scalar fold for `i32` / `i64` (promoted to `f64` accumulators).
pub(crate) fn fold_read_plan_scalar_operation_int_parallel(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: reduction::ReductionKind,
    dtype: ElementDtype,
    workers: Option<usize>,
) -> Result<shared::FoldPlanOutcome, TetError> {
    let visit = match dtype {
        ElementDtype::I32 => ScalarFoldVisit::Int(IntVisit::I32),
        ElementDtype::I64 => ScalarFoldVisit::Int(IntVisit::I64),
        ElementDtype::U8 => ScalarFoldVisit::Int(IntVisit::U8),
        ElementDtype::U16 => ScalarFoldVisit::Int(IntVisit::U16),
        ElementDtype::I16 => ScalarFoldVisit::Int(IntVisit::I16),
        ElementDtype::U32 => ScalarFoldVisit::Int(IntVisit::U32),
        ElementDtype::U64 => ScalarFoldVisit::Int(IntVisit::U64),
        _ => {
            return Err(TetError::Validation(
                "integer fold requires i32, i64, u8, u16, i16, u32, or u64 dtype".into(),
            ));
        }
    };
    fold_read_plan_scalar_parallel(mmap, plan, max_preview, kind, visit, workers)
}
