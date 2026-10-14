//! Disjoint parallel-fold preview buffers (raw pointer slices).

/// Typed preview slice backed by a disjoint parallel-fold buffer (or empty when capped).
pub(crate) fn disjoint_preview_mut<'a, T>(preview_addr: usize, preview_len: usize) -> &'a mut [T] {
    if preview_len == 0 {
        &mut []
    } else {
        // SAFETY: planned chunks write disjoint logical indices.
        unsafe { std::slice::from_raw_parts_mut(preview_addr as *mut T, preview_len) }
    }
}

fn write_disjoint_preview<T: Copy>(preview_addr: usize, preview_len: usize, li: usize, v: T) {
    if li < preview_len {
        disjoint_preview_mut(preview_addr, preview_len)[li] = v;
    }
}

pub(crate) fn write_disjoint_preview_f32(
    preview_addr: usize,
    preview_len: usize,
    li: usize,
    v: f32,
) {
    write_disjoint_preview(preview_addr, preview_len, li, v);
}

pub(crate) fn write_disjoint_preview_f64(
    preview_addr: usize,
    preview_len: usize,
    li: usize,
    v: f64,
) {
    write_disjoint_preview(preview_addr, preview_len, li, v);
}

pub(crate) fn write_disjoint_preview_i32(
    preview_addr: usize,
    preview_len: usize,
    li: usize,
    v: f64,
) {
    write_disjoint_preview(preview_addr, preview_len, li, v as i32);
}

pub(crate) fn write_disjoint_preview_i64(
    preview_addr: usize,
    preview_len: usize,
    li: usize,
    v: f64,
) {
    write_disjoint_preview(preview_addr, preview_len, li, v as i64);
}

pub(crate) fn write_disjoint_preview_u8(
    preview_addr: usize,
    preview_len: usize,
    li: usize,
    v: f64,
) {
    write_disjoint_preview(preview_addr, preview_len, li, v.clamp(0., 255.) as u8);
}

pub(crate) fn write_disjoint_preview_u16(
    preview_addr: usize,
    preview_len: usize,
    li: usize,
    v: f64,
) {
    write_disjoint_preview(preview_addr, preview_len, li, v.clamp(0., 65535.) as u16);
}

pub(crate) fn write_disjoint_preview_i16(
    preview_addr: usize,
    preview_len: usize,
    li: usize,
    v: f64,
) {
    write_disjoint_preview(
        preview_addr,
        preview_len,
        li,
        v.clamp(f64::from(i16::MIN), f64::from(i16::MAX)) as i16,
    );
}

pub(crate) fn write_disjoint_preview_u32(
    preview_addr: usize,
    preview_len: usize,
    li: usize,
    v: f64,
) {
    write_disjoint_preview(
        preview_addr,
        preview_len,
        li,
        v.clamp(0., f64::from(u32::MAX)) as u32,
    );
}

pub(crate) fn write_disjoint_preview_u64(
    preview_addr: usize,
    preview_len: usize,
    li: usize,
    v: f64,
) {
    write_disjoint_preview(preview_addr, preview_len, li, v.max(0.) as u64);
}

pub(crate) fn write_disjoint_preview_f16(
    preview_addr: usize,
    preview_len: usize,
    li: usize,
    v: f64,
) {
    write_disjoint_preview(preview_addr, preview_len, li, half::f16::from_f64(v));
}
