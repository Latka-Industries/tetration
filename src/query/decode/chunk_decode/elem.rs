//! Wire element types for chunk decode / visit / scatter.

use crate::query::fold::reduction::{ArgIndexAccum, ReductionKind, ValueAccum};
use crate::utils::{f16_le, f32_le, f64_le, i16_le, i32_le, i64_le, u8_le, u16_le, u32_le, u64_le};

/// Wire element type for chunk decode / visit / scatter.
pub(super) trait ChunkElem: Copy {
    const ELEM_SIZE: usize;
    fn read_at(raw: &[u8], index: usize) -> Self;
}

impl ChunkElem for f32 {
    const ELEM_SIZE: usize = 4;
    fn read_at(raw: &[u8], index: usize) -> Self {
        f32_le::read_f32_le_at(raw, index)
    }
}

impl ChunkElem for f64 {
    const ELEM_SIZE: usize = 8;
    fn read_at(raw: &[u8], index: usize) -> Self {
        f64_le::read_f64_le_at(raw, index)
    }
}

impl ChunkElem for i32 {
    const ELEM_SIZE: usize = 4;
    fn read_at(raw: &[u8], index: usize) -> Self {
        i32_le::read_i32_le_at(raw, index)
    }
}

impl ChunkElem for i64 {
    const ELEM_SIZE: usize = 8;
    fn read_at(raw: &[u8], index: usize) -> Self {
        i64_le::read_i64_le_at(raw, index)
    }
}

impl ChunkElem for u8 {
    const ELEM_SIZE: usize = 1;
    fn read_at(raw: &[u8], index: usize) -> Self {
        u8_le::read_u8_le_at(raw, index)
    }
}

impl ChunkElem for u16 {
    const ELEM_SIZE: usize = 2;
    fn read_at(raw: &[u8], index: usize) -> Self {
        u16_le::read_u16_le_at(raw, index)
    }
}

impl ChunkElem for i16 {
    const ELEM_SIZE: usize = 2;
    fn read_at(raw: &[u8], index: usize) -> Self {
        i16_le::read_i16_le_at(raw, index)
    }
}

impl ChunkElem for u32 {
    const ELEM_SIZE: usize = 4;
    fn read_at(raw: &[u8], index: usize) -> Self {
        u32_le::read_u32_le_at(raw, index)
    }
}

impl ChunkElem for u64 {
    const ELEM_SIZE: usize = 8;
    fn read_at(raw: &[u8], index: usize) -> Self {
        u64_le::read_u64_le_at(raw, index)
    }
}

impl ChunkElem for half::f16 {
    const ELEM_SIZE: usize = 2;
    fn read_at(raw: &[u8], index: usize) -> Self {
        f16_le::read_f16_le_at(raw, index)
    }
}

/// Scalar fold hooks for floating chunk payloads (`f32` / `f64`).
pub(super) trait FoldChunkElem: ChunkElem {
    fn as_f64(v: Self) -> f64;
    fn push_le_bytes(value: &mut ValueAccum, raw: &[u8], kind: ReductionKind);
    fn push_value(value: &mut ValueAccum, v: Self);
    fn push_arg(arg: &mut ArgIndexAccum, li: u64, v: Self, kind: ReductionKind);
}

impl FoldChunkElem for f32 {
    fn as_f64(v: Self) -> f64 {
        f64::from(v)
    }
    fn push_le_bytes(value: &mut ValueAccum, raw: &[u8], kind: ReductionKind) {
        value.push_f32_le_bytes(raw, kind);
    }
    fn push_value(value: &mut ValueAccum, v: Self) {
        value.push(v);
    }
    fn push_arg(arg: &mut ArgIndexAccum, li: u64, v: Self, kind: ReductionKind) {
        arg.push(li, v, kind);
    }
}

impl FoldChunkElem for f64 {
    fn as_f64(v: Self) -> f64 {
        v
    }
    fn push_le_bytes(value: &mut ValueAccum, raw: &[u8], kind: ReductionKind) {
        value.push_f64_le_bytes(raw, kind);
    }
    fn push_value(value: &mut ValueAccum, v: Self) {
        value.push_f64(v);
    }
    fn push_arg(arg: &mut ArgIndexAccum, li: u64, v: Self, kind: ReductionKind) {
        arg.push_f64(li, v, kind);
    }
}
