//! Sequential byte-stream scalar fold over one contiguous raw payload span (hyperslab path).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::catalog::CHUNK_PAYLOAD_CODEC_V1;
use crate::query::fold::reduction::{ReductionKind, ValueAccum};
use crate::query::fold::shared::{
    FoldPreviewBuffer, build_fold_plan_outcome_typed, validate_fold_preview,
    validate_fold_preview_f64,
};
use crate::query::types::{ReadPlan, TetError};
use crate::utils::dtype::ElementDtype;
use crate::utils::wire;

/// Raw payload bytes per sequential scan window (64 MiB — matches bench slab size).
pub const SCAN_WINDOW_RAW_BYTES: usize = 64 * 1024 * 1024;

/// Contiguous raw byte range covering the full logical selection in on-disk order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContiguousRawSpan {
    pub start: usize,
    pub len: usize,
}

/// Tier-A/B scalar ops supported without per-element logical indices.
#[must_use]
pub fn supports_scalar_kind(kind: ReductionKind) -> bool {
    !matches!(kind, ReductionKind::ArgMin | ReductionKind::ArgMax)
}

/// Full dense scan with raw codec and adjacent payloads in logical plan order.
#[must_use]
pub fn detect_contiguous_raw_span(plan: &ReadPlan, elem_size: usize) -> Option<ContiguousRawSpan> {
    if plan.chunks.is_empty() {
        return None;
    }
    let expected_raw = plan.logical_element_count.checked_mul(elem_size)? as u64;
    if expected_raw == 0 {
        return None;
    }
    for c in &plan.chunks {
        if c.codec != CHUNK_PAYLOAD_CODEC_V1.raw || c.stored_byte_len != c.raw_byte_len {
            return None;
        }
    }
    for w in plan.chunks.windows(2) {
        let end = w[0]
            .payload_offset
            .checked_add(w[0].raw_byte_len)
            .filter(|&end| end == w[1].payload_offset)?;
        let _ = end;
    }
    let total_raw: u64 = plan.chunks.iter().map(|c| c.raw_byte_len).sum();
    if total_raw != expected_raw {
        return None;
    }
    let start = usize::try_from(plan.chunks[0].payload_offset).ok()?;
    let len = usize::try_from(total_raw).ok()?;
    Some(ContiguousRawSpan { start, len })
}

fn span_slice(mmap: &[u8], span: ContiguousRawSpan) -> Result<&[u8], TetError> {
    let range = wire::checked_usize_subslice(span.start, span.len, mmap.len()).map_err(|e| {
        TetError::Validation(format!("contiguous payload span out of mmap bounds: {e:?}"))
    })?;
    Ok(&mmap[range])
}

fn fill_f32_preview(raw: &[u8], preview: &mut [f32], global_offset_elems: usize) -> bool {
    let vals: &[f32] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn fill_f64_preview(raw: &[u8], preview: &mut [f64], global_offset_elems: usize) -> bool {
    let vals: &[f64] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn fill_i32_preview(raw: &[u8], preview: &mut [i32], global_offset_elems: usize) -> bool {
    let vals: &[i32] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn fill_u16_preview(raw: &[u8], preview: &mut [u16], global_offset_elems: usize) -> bool {
    let vals: &[u16] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn fill_i16_preview(raw: &[u8], preview: &mut [i16], global_offset_elems: usize) -> bool {
    let vals: &[i16] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn fill_u8_preview(raw: &[u8], preview: &mut [u8], global_offset_elems: usize) -> bool {
    let vals: &[u8] = raw;
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn fill_i64_preview(raw: &[u8], preview: &mut [i64], global_offset_elems: usize) -> bool {
    let vals: &[i64] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn fold_window_f32(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [f32],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 4, 0);
    acc.push_f32_le_bytes(window, kind);
    if preview.is_empty() {
        false
    } else {
        fill_f32_preview(window, preview, global_offset_elems)
    }
}

fn fold_window_f64(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [f64],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 8, 0);
    acc.push_f64_le_bytes(window, kind);
    if preview.is_empty() {
        false
    } else {
        fill_f64_preview(window, preview, global_offset_elems)
    }
}

fn fold_window_i32(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [i32],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 4, 0);
    if matches!(kind, ReductionKind::Count) {
        acc.push_f32_le_bytes(window, kind);
    } else {
        acc.push_i32_le_bytes(window, kind);
    }
    if preview.is_empty() {
        false
    } else {
        fill_i32_preview(window, preview, global_offset_elems)
    }
}

fn fold_window_i64(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [i64],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 8, 0);
    if matches!(kind, ReductionKind::Count) {
        acc.push_f64_le_bytes(window, kind);
    } else {
        acc.push_i64_le_bytes(window, kind);
    }
    if preview.is_empty() {
        false
    } else {
        fill_i64_preview(window, preview, global_offset_elems)
    }
}

fn fold_span_windows(
    raw: &[u8],
    elem_size: usize,
    mut fold_window: impl FnMut(&mut ValueAccum, &[u8], usize) -> bool,
) -> Result<(ValueAccum, bool), TetError> {
    let mut acc = ValueAccum::default();
    let mut offset = 0usize;
    let mut global_elems = 0usize;
    let mut saw_preview = false;
    while offset < raw.len() {
        let end = (offset + SCAN_WINDOW_RAW_BYTES).min(raw.len());
        let window = &raw[offset..end];
        if !window.len().is_multiple_of(elem_size) {
            return Err(TetError::Validation(
                "contiguous payload span length is not a multiple of element size".into(),
            ));
        }
        saw_preview |= fold_window(&mut acc, window, global_elems);
        offset = end;
        global_elems += window.len() / elem_size;
    }
    Ok((acc, saw_preview))
}

/// Sequential `read` from `path` (one seek + streaming reads) — avoids cold mmap page faults.
fn fold_span_file(
    path: &Path,
    span: ContiguousRawSpan,
    elem_size: usize,
    mut fold_window: impl FnMut(&mut ValueAccum, &[u8], usize) -> bool,
) -> Result<(ValueAccum, bool), TetError> {
    let mut file = File::open(path).map_err(|e| {
        TetError::Validation(format!(
            "linear scan open failed for {}: {e}",
            path.display()
        ))
    })?;
    file.seek(SeekFrom::Start(span.start as u64))
        .map_err(|e| TetError::Validation(format!("linear scan seek failed: {e}")))?;

    let win_cap = SCAN_WINDOW_RAW_BYTES.max(elem_size);
    let mut buf = vec![0u8; win_cap];
    let mut acc = ValueAccum::default();
    let mut offset = 0usize;
    let mut global_elems = 0usize;
    let mut saw_preview = false;

    while offset < span.len {
        let win_len = SCAN_WINDOW_RAW_BYTES.min(span.len - offset);
        let window = &mut buf[..win_len];
        file.read_exact(window)
            .map_err(|e| TetError::Validation(format!("linear scan read failed: {e}")))?;
        if !window.len().is_multiple_of(elem_size) {
            return Err(TetError::Validation(
                "contiguous payload span length is not a multiple of element size".into(),
            ));
        }
        saw_preview |= fold_window(&mut acc, window, global_elems);
        offset += win_len;
        global_elems += window.len() / elem_size;
    }
    Ok((acc, saw_preview))
}

fn require_nonempty(acc: &ValueAccum) -> Result<(), TetError> {
    if acc.is_empty() {
        return Err(TetError::Validation(
            "operation requires at least one decoded value from the read plan".into(),
        ));
    }
    Ok(())
}

fn require_int_preview(preview_cap: usize, saw_preview: bool) -> Result<(), TetError> {
    if preview_cap > 0 && !saw_preview {
        return Err(TetError::Validation(
            "materialized selection has unset preview elements".into(),
        ));
    }
    Ok(())
}

fn fold_span_source(
    mmap: &[u8],
    tet_path: Option<&Path>,
    span: ContiguousRawSpan,
    elem_size: usize,
    mut fold_window: impl FnMut(&mut ValueAccum, &[u8], usize) -> bool,
) -> Result<(ValueAccum, bool), TetError> {
    if let Some(path) = tet_path {
        fold_span_file(path, span, elem_size, &mut fold_window)
    } else {
        let raw = span_slice(mmap, span)?;
        fold_span_windows(raw, elem_size, fold_window)
    }
}

struct LinearFoldParams<'a> {
    mmap: &'a [u8],
    tet_path: Option<&'a Path>,
    span: ContiguousRawSpan,
    elem_size: usize,
    kind: ReductionKind,
    max_preview: usize,
    preview_cap: usize,
    n: usize,
    total_bytes: u64,
}

fn linear_fold_f32(
    p: &LinearFoldParams<'_>,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let mut preview = vec![f32::NAN; p.preview_cap];
    let (acc, _) = fold_span_source(p.mmap, p.tet_path, p.span, p.elem_size, |acc, window, g| {
        fold_window_f32(acc, window, p.kind, &mut preview, g)
    })?;
    require_nonempty(&acc)?;
    validate_fold_preview(true, &preview, p.preview_cap)?;
    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::F32(if p.max_preview == 0 {
            Vec::new()
        } else {
            preview
        }),
        p.max_preview,
        p.n,
        p.total_bytes,
        acc.finish_scalar(p.kind).into(),
    ))
}

fn linear_fold_f64(
    p: &LinearFoldParams<'_>,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let mut preview = vec![f64::NAN; p.preview_cap];
    let (acc, _) = fold_span_source(p.mmap, p.tet_path, p.span, p.elem_size, |acc, window, g| {
        fold_window_f64(acc, window, p.kind, &mut preview, g)
    })?;
    require_nonempty(&acc)?;
    validate_fold_preview_f64(true, &preview, p.preview_cap)?;
    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::F64(if p.max_preview == 0 {
            Vec::new()
        } else {
            preview
        }),
        p.max_preview,
        p.n,
        p.total_bytes,
        acc.finish_scalar(p.kind).into(),
    ))
}

fn linear_fold_i32(
    p: &LinearFoldParams<'_>,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let mut preview = vec![0i32; p.preview_cap];
    let (acc, saw_preview) =
        fold_span_source(p.mmap, p.tet_path, p.span, p.elem_size, |acc, window, g| {
            fold_window_i32(acc, window, p.kind, &mut preview, g)
        })?;
    require_nonempty(&acc)?;
    require_int_preview(p.preview_cap, saw_preview || p.preview_cap == 0)?;
    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::I32(if p.max_preview == 0 {
            Vec::new()
        } else {
            preview
        }),
        p.max_preview,
        p.n,
        p.total_bytes,
        acc.finish_scalar(p.kind).into(),
    ))
}

fn fold_window_u8(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [u8],
    global_offset_elems: usize,
) -> bool {
    if matches!(kind, ReductionKind::Count) {
        acc.push_f32_le_bytes(window, kind);
    } else {
        acc.push_u8_le_bytes(window, kind);
    }
    if preview.is_empty() {
        false
    } else {
        fill_u8_preview(window, preview, global_offset_elems)
    }
}

fn linear_fold_u8(
    p: &LinearFoldParams<'_>,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let mut preview = vec![0u8; p.preview_cap];
    let (acc, saw_preview) =
        fold_span_source(p.mmap, p.tet_path, p.span, p.elem_size, |acc, window, g| {
            fold_window_u8(acc, window, p.kind, &mut preview, g)
        })?;
    require_nonempty(&acc)?;
    require_int_preview(p.preview_cap, saw_preview || p.preview_cap == 0)?;
    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::U8(if p.max_preview == 0 {
            Vec::new()
        } else {
            preview
        }),
        p.max_preview,
        p.n,
        p.total_bytes,
        acc.finish_scalar(p.kind).into(),
    ))
}

fn fold_window_u16(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [u16],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 2, 0);
    if matches!(kind, ReductionKind::Count) {
        acc.push_f32_le_bytes(window, kind);
    } else {
        acc.push_u16_le_bytes(window, kind);
    }
    if preview.is_empty() {
        false
    } else {
        fill_u16_preview(window, preview, global_offset_elems)
    }
}

fn fold_window_i16(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [i16],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 2, 0);
    if matches!(kind, ReductionKind::Count) {
        acc.push_f32_le_bytes(window, kind);
    } else {
        acc.push_i16_le_bytes(window, kind);
    }
    if preview.is_empty() {
        false
    } else {
        fill_i16_preview(window, preview, global_offset_elems)
    }
}

fn linear_fold_u16(
    p: &LinearFoldParams<'_>,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let mut preview = vec![0u16; p.preview_cap];
    let (acc, saw_preview) =
        fold_span_source(p.mmap, p.tet_path, p.span, p.elem_size, |acc, window, g| {
            fold_window_u16(acc, window, p.kind, &mut preview, g)
        })?;
    require_nonempty(&acc)?;
    require_int_preview(p.preview_cap, saw_preview || p.preview_cap == 0)?;
    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::U16(if p.max_preview == 0 {
            Vec::new()
        } else {
            preview
        }),
        p.max_preview,
        p.n,
        p.total_bytes,
        acc.finish_scalar(p.kind).into(),
    ))
}

fn linear_fold_i16(
    p: &LinearFoldParams<'_>,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let mut preview = vec![0i16; p.preview_cap];
    let (acc, saw_preview) =
        fold_span_source(p.mmap, p.tet_path, p.span, p.elem_size, |acc, window, g| {
            fold_window_i16(acc, window, p.kind, &mut preview, g)
        })?;
    require_nonempty(&acc)?;
    require_int_preview(p.preview_cap, saw_preview || p.preview_cap == 0)?;
    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::I16(if p.max_preview == 0 {
            Vec::new()
        } else {
            preview
        }),
        p.max_preview,
        p.n,
        p.total_bytes,
        acc.finish_scalar(p.kind).into(),
    ))
}

fn linear_fold_i64(
    p: &LinearFoldParams<'_>,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let mut preview = vec![0i64; p.preview_cap];
    let (acc, saw_preview) =
        fold_span_source(p.mmap, p.tet_path, p.span, p.elem_size, |acc, window, g| {
            fold_window_i64(acc, window, p.kind, &mut preview, g)
        })?;
    require_nonempty(&acc)?;
    require_int_preview(p.preview_cap, saw_preview || p.preview_cap == 0)?;
    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::I64(if p.max_preview == 0 {
            Vec::new()
        } else {
            preview
        }),
        p.max_preview,
        p.n,
        p.total_bytes,
        acc.finish_scalar(p.kind).into(),
    ))
}

pub(crate) fn fold_read_plan_scalar_linear(
    mmap: &[u8],
    plan: &ReadPlan,
    max_preview: usize,
    kind: ReductionKind,
    dtype: ElementDtype,
    tet_path: Option<&Path>,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let elem_size = dtype.elem_size();
    let span = detect_contiguous_raw_span(plan, elem_size).ok_or_else(|| {
        TetError::Validation("linear scan requires contiguous raw payload span".into())
    })?;
    let n = plan.logical_element_count;
    let p = LinearFoldParams {
        mmap,
        tet_path,
        span,
        elem_size,
        kind,
        max_preview,
        preview_cap: max_preview.min(n),
        n,
        total_bytes: u64::try_from(span.len).unwrap_or(u64::MAX),
    };
    match dtype {
        ElementDtype::F32 => linear_fold_f32(&p),
        ElementDtype::F64 => linear_fold_f64(&p),
        ElementDtype::I32 => linear_fold_i32(&p),
        ElementDtype::I64 => linear_fold_i64(&p),
        ElementDtype::U8 => linear_fold_u8(&p),
        ElementDtype::U16 => linear_fold_u16(&p),
        ElementDtype::I16 => linear_fold_i16(&p),
        ElementDtype::U32 => linear_fold_u32(&p),
        ElementDtype::U64 => linear_fold_u64(&p),
        ElementDtype::F16 => linear_fold_f16(&p),
    }
}

fn linear_fold_u32(
    p: &LinearFoldParams<'_>,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let mut preview = vec![0u32; p.preview_cap];
    let (acc, saw_preview) =
        fold_span_source(p.mmap, p.tet_path, p.span, p.elem_size, |acc, window, g| {
            fold_window_u32(acc, window, p.kind, &mut preview, g)
        })?;
    require_nonempty(&acc)?;
    require_int_preview(p.preview_cap, saw_preview || p.preview_cap == 0)?;
    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::U32(if p.max_preview == 0 {
            Vec::new()
        } else {
            preview
        }),
        p.max_preview,
        p.n,
        p.total_bytes,
        acc.finish_scalar(p.kind).into(),
    ))
}

fn fold_window_u32(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [u32],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 4, 0);
    if matches!(kind, ReductionKind::Count) {
        acc.push_f32_le_bytes(window, kind);
    } else {
        acc.push_u32_le_bytes(window, kind);
    }
    if preview.is_empty() {
        false
    } else {
        fill_u32_preview(window, preview, global_offset_elems)
    }
}

fn fill_u32_preview(raw: &[u8], preview: &mut [u32], global_offset_elems: usize) -> bool {
    let vals: &[u32] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn linear_fold_u64(
    p: &LinearFoldParams<'_>,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let mut preview = vec![0u64; p.preview_cap];
    let (acc, saw_preview) =
        fold_span_source(p.mmap, p.tet_path, p.span, p.elem_size, |acc, window, g| {
            fold_window_u64(acc, window, p.kind, &mut preview, g)
        })?;
    require_nonempty(&acc)?;
    require_int_preview(p.preview_cap, saw_preview || p.preview_cap == 0)?;
    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::U64(if p.max_preview == 0 {
            Vec::new()
        } else {
            preview
        }),
        p.max_preview,
        p.n,
        p.total_bytes,
        acc.finish_scalar(p.kind).into(),
    ))
}

fn fold_window_u64(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [u64],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 8, 0);
    if matches!(kind, ReductionKind::Count) {
        acc.push_f64_le_bytes(window, kind);
    } else {
        acc.push_u64_le_bytes(window, kind);
    }
    if preview.is_empty() {
        false
    } else {
        fill_u64_preview(window, preview, global_offset_elems)
    }
}

fn fill_u64_preview(raw: &[u8], preview: &mut [u64], global_offset_elems: usize) -> bool {
    let vals: &[u64] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}

fn linear_fold_f16(
    p: &LinearFoldParams<'_>,
) -> Result<crate::query::fold::FoldPlanOutcome, TetError> {
    let mut preview = vec![half::f16::NAN; p.preview_cap];
    let (acc, _) = fold_span_source(p.mmap, p.tet_path, p.span, p.elem_size, |acc, window, g| {
        fold_window_f16(acc, window, p.kind, &mut preview, g)
    })?;
    require_nonempty(&acc)?;
    crate::query::fold::shared::validate_fold_preview_f16(true, &preview, p.preview_cap)?;
    Ok(build_fold_plan_outcome_typed(
        FoldPreviewBuffer::F16(if p.max_preview == 0 {
            Vec::new()
        } else {
            preview
        }),
        p.max_preview,
        p.n,
        p.total_bytes,
        acc.finish_scalar(p.kind).into(),
    ))
}

fn fold_window_f16(
    acc: &mut ValueAccum,
    window: &[u8],
    kind: ReductionKind,
    preview: &mut [half::f16],
    global_offset_elems: usize,
) -> bool {
    debug_assert_eq!(window.len() % 2, 0);
    acc.push_f16_le_bytes(window, kind);
    if preview.is_empty() {
        false
    } else {
        fill_f16_preview(window, preview, global_offset_elems)
    }
}

fn fill_f16_preview(raw: &[u8], preview: &mut [half::f16], global_offset_elems: usize) -> bool {
    let vals: &[half::f16] = bytemuck::cast_slice(raw);
    let mut wrote = false;
    for (k, &v) in vals.iter().enumerate() {
        let li = global_offset_elems + k;
        if li < preview.len() {
            preview[li] = v;
            wrote = true;
        } else {
            break;
        }
    }
    wrote
}
