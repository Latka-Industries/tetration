//! Full logical selection (RAM vs temp spill) and preview helpers.

use std::path::Path;

use crate::query::engine::budget::{ExecutionBudget, MemoryStrategy};
use crate::query::engine::spill_policy::{SpillPathAllowlist, TempSpillFile};
use crate::query::types::{ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;

use super::f32::{
    materialize_into_vec, preview_from_materialized_f32, preview_from_spill_file_f32,
};
use super::float::{
    materialize_into_vec_f16, materialize_into_vec_f64, preview_from_materialized_f16,
    preview_from_materialized_f64, preview_from_spill_file_f16, preview_from_spill_file_f64,
};
use super::int;
use super::logical::MaterializedLogical;
use super::types::{DecodePreviewBundle, LogicalF16Backing, LogicalF32Backing, LogicalF64Backing};

/// Decode the full logical selection, choosing RAM vs temp spill from the memory budget.
pub(crate) fn materialize_logical_selection(
    mmap: &[u8],
    plan: &ReadPlan,
    budget: &ExecutionBudget,
    allowlist: &SpillPathAllowlist,
    dtype: ElementDtype,
) -> Result<MaterializedLogical, TetError> {
    if budget.full_tensor_exceeds_budget(plan, dtype)? {
        materialize_logical_spill(mmap, plan, allowlist, dtype)
    } else {
        materialize_logical_in_memory(mmap, plan, dtype)
    }
}

fn materialize_logical_spill(
    mmap: &[u8],
    plan: &ReadPlan,
    allowlist: &SpillPathAllowlist,
    dtype: ElementDtype,
) -> Result<MaterializedLogical, TetError> {
    let temp = TempSpillFile::create(allowlist)?;
    let bytes = crate::query::dispatch::spill_full_selection(mmap, plan, temp.path(), dtype)?;
    let strategy = MemoryStrategy::TempSpillMaterialize;
    Ok(match dtype {
        ElementDtype::F32 => MaterializedLogical::F32 {
            backing: LogicalF32Backing::TempSpill(temp),
            total_bytes_read_from_disk: bytes,
            strategy,
        },
        ElementDtype::F64 => MaterializedLogical::F64 {
            backing: LogicalF64Backing::TempSpill(temp),
            total_bytes_read_from_disk: bytes,
            strategy,
        },
        ElementDtype::I32 => MaterializedLogical::I32 {
            backing: int::LogicalI32Backing::TempSpill(temp),
            total_bytes_read_from_disk: bytes,
            strategy,
        },
        ElementDtype::I64 => MaterializedLogical::I64 {
            backing: int::LogicalI64Backing::TempSpill(temp),
            total_bytes_read_from_disk: bytes,
            strategy,
        },
        ElementDtype::U8 => MaterializedLogical::U8 {
            backing: int::LogicalU8Backing::TempSpill(temp),
            total_bytes_read_from_disk: bytes,
            strategy,
        },
        ElementDtype::U16 => MaterializedLogical::U16 {
            backing: int::LogicalU16Backing::TempSpill(temp),
            total_bytes_read_from_disk: bytes,
            strategy,
        },
        ElementDtype::I16 => MaterializedLogical::I16 {
            backing: int::LogicalI16Backing::TempSpill(temp),
            total_bytes_read_from_disk: bytes,
            strategy,
        },
        ElementDtype::U32 => MaterializedLogical::U32 {
            backing: int::LogicalU32Backing::TempSpill(temp),
            total_bytes_read_from_disk: bytes,
            strategy,
        },
        ElementDtype::U64 => MaterializedLogical::U64 {
            backing: int::LogicalU64Backing::TempSpill(temp),
            total_bytes_read_from_disk: bytes,
            strategy,
        },
        ElementDtype::F16 => MaterializedLogical::F16 {
            backing: LogicalF16Backing::TempSpill(temp),
            total_bytes_read_from_disk: bytes,
            strategy,
        },
    })
}

fn materialize_logical_in_memory(
    mmap: &[u8],
    plan: &ReadPlan,
    dtype: ElementDtype,
) -> Result<MaterializedLogical, TetError> {
    let strategy = MemoryStrategy::InMemoryMaterialize;
    Ok(match dtype {
        ElementDtype::F32 => {
            let (vec, bytes) = materialize_into_vec(mmap, plan)?;
            MaterializedLogical::F32 {
                backing: LogicalF32Backing::InMemory(vec),
                total_bytes_read_from_disk: bytes,
                strategy,
            }
        }
        ElementDtype::F64 => {
            let (vec, bytes) = materialize_into_vec_f64(mmap, plan)?;
            MaterializedLogical::F64 {
                backing: LogicalF64Backing::InMemory(vec),
                total_bytes_read_from_disk: bytes,
                strategy,
            }
        }
        ElementDtype::I32 => {
            let (vec, bytes) = int::materialize_into_vec_i32(mmap, plan)?;
            MaterializedLogical::I32 {
                backing: int::LogicalI32Backing::InMemory(vec),
                total_bytes_read_from_disk: bytes,
                strategy,
            }
        }
        ElementDtype::I64 => {
            let (vec, bytes) = int::materialize_into_vec_i64(mmap, plan)?;
            MaterializedLogical::I64 {
                backing: int::LogicalI64Backing::InMemory(vec),
                total_bytes_read_from_disk: bytes,
                strategy,
            }
        }
        ElementDtype::U8 => {
            let (vec, bytes) = int::materialize_into_vec_u8(mmap, plan)?;
            MaterializedLogical::U8 {
                backing: int::LogicalU8Backing::InMemory(vec),
                total_bytes_read_from_disk: bytes,
                strategy,
            }
        }
        ElementDtype::U16 => {
            let (vec, bytes) = int::materialize_into_vec_u16(mmap, plan)?;
            MaterializedLogical::U16 {
                backing: int::LogicalU16Backing::InMemory(vec),
                total_bytes_read_from_disk: bytes,
                strategy,
            }
        }
        ElementDtype::I16 => {
            let (vec, bytes) = int::materialize_into_vec_i16(mmap, plan)?;
            MaterializedLogical::I16 {
                backing: int::LogicalI16Backing::InMemory(vec),
                total_bytes_read_from_disk: bytes,
                strategy,
            }
        }
        ElementDtype::U32 => {
            let (vec, bytes) = int::materialize_into_vec_u32(mmap, plan)?;
            MaterializedLogical::U32 {
                backing: int::LogicalU32Backing::InMemory(vec),
                total_bytes_read_from_disk: bytes,
                strategy,
            }
        }
        ElementDtype::U64 => {
            let (vec, bytes) = int::materialize_into_vec_u64(mmap, plan)?;
            MaterializedLogical::U64 {
                backing: int::LogicalU64Backing::InMemory(vec),
                total_bytes_read_from_disk: bytes,
                strategy,
            }
        }
        ElementDtype::F16 => {
            let (vec, bytes) = materialize_into_vec_f16(mmap, plan)?;
            MaterializedLogical::F16 {
                backing: LogicalF16Backing::InMemory(vec),
                total_bytes_read_from_disk: bytes,
                strategy,
            }
        }
    })
}

/// First `max` logical values without a second full decode (from RAM or spill file).
pub(crate) fn preview_from_materialized(
    materialized: &MaterializedLogical,
    logical_len: usize,
    max: usize,
) -> Result<DecodePreviewBundle, TetError> {
    let cap = max.min(logical_len);
    if cap == 0 {
        return Ok(DecodePreviewBundle::all_truncated(logical_len > 0));
    }
    match materialized {
        MaterializedLogical::F32 { backing, .. } => {
            let (p, t) = preview_from_materialized_f32(backing, logical_len, max)?;
            Ok(DecodePreviewBundle::f32_preview(p, t))
        }
        MaterializedLogical::F64 { backing, .. } => {
            let (p, t) = preview_from_materialized_f64(backing, logical_len, max)?;
            Ok(DecodePreviewBundle::f64_preview(p, t))
        }
        MaterializedLogical::I32 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_i32(backing, logical_len, max)?;
            Ok(DecodePreviewBundle::i32_preview(p, t))
        }
        MaterializedLogical::I64 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_i64(backing, logical_len, max)?;
            Ok(DecodePreviewBundle::i64_preview(p, t))
        }
        MaterializedLogical::U8 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_u8(backing, logical_len, max)?;
            Ok(DecodePreviewBundle::u8_preview(p, t))
        }
        MaterializedLogical::U16 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_u16(backing, logical_len, max)?;
            Ok(DecodePreviewBundle::u16_preview(p, t))
        }
        MaterializedLogical::I16 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_i16(backing, logical_len, max)?;
            Ok(DecodePreviewBundle::i16_preview(p, t))
        }
        MaterializedLogical::U32 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_u32(backing, logical_len, max)?;
            Ok(DecodePreviewBundle::u32_preview(p, t))
        }
        MaterializedLogical::U64 { backing, .. } => {
            let (p, t) = int::preview_from_materialized_u64(backing, logical_len, max)?;
            Ok(DecodePreviewBundle::u64_preview(p, t))
        }
        MaterializedLogical::F16 { backing, .. } => {
            let (p, t) = preview_from_materialized_f16(backing, logical_len, max)?;
            Ok(DecodePreviewBundle::f16_preview(p, t))
        }
    }
}

/// Preview from an export spill file (single decode pass for spill + preview).
pub(crate) fn preview_from_spill_export_file(
    path: &Path,
    logical_len: usize,
    max: usize,
    dtype: ElementDtype,
) -> Result<DecodePreviewBundle, TetError> {
    let cap = max.min(logical_len);
    match dtype {
        ElementDtype::F32 => {
            let (p, t) = preview_from_spill_file_f32(path, cap, logical_len)?;
            Ok(DecodePreviewBundle::f32_preview(p, t))
        }
        ElementDtype::F64 => {
            let (p, t) = preview_from_spill_file_f64(path, cap, logical_len)?;
            Ok(DecodePreviewBundle::f64_preview(p, t))
        }
        ElementDtype::I32 => {
            let (p, t) = int::preview_from_spill_file_i32(path, cap, logical_len)?;
            Ok(DecodePreviewBundle::i32_preview(p, t))
        }
        ElementDtype::I64 => {
            let (p, t) = int::preview_from_spill_file_i64(path, cap, logical_len)?;
            Ok(DecodePreviewBundle::i64_preview(p, t))
        }
        ElementDtype::U8 => {
            let (p, t) = int::preview_from_spill_file_u8(path, cap, logical_len)?;
            Ok(DecodePreviewBundle::u8_preview(p, t))
        }
        ElementDtype::U16 => {
            let (p, t) = int::preview_from_spill_file_u16(path, cap, logical_len)?;
            Ok(DecodePreviewBundle::u16_preview(p, t))
        }
        ElementDtype::I16 => {
            let (p, t) = int::preview_from_spill_file_i16(path, cap, logical_len)?;
            Ok(DecodePreviewBundle::i16_preview(p, t))
        }
        ElementDtype::U32 => {
            let (p, t) = int::preview_from_spill_file_u32(path, cap, logical_len)?;
            Ok(DecodePreviewBundle::u32_preview(p, t))
        }
        ElementDtype::U64 => {
            let (p, t) = int::preview_from_spill_file_u64(path, cap, logical_len)?;
            Ok(DecodePreviewBundle::u64_preview(p, t))
        }
        ElementDtype::F16 => {
            let (p, t) = preview_from_spill_file_f16(path, cap, logical_len)?;
            Ok(DecodePreviewBundle::f16_preview(p, t))
        }
    }
}
