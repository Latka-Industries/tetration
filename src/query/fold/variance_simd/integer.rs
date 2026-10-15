#[inline]
pub(crate) fn scalar_i32_sum_sumsq(vals: &[i32]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod i32_arch {
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn i32_sum_sumsq_sse2(vals: &[i32]) -> (f64, f64) {
        use crate::query::fold::variance_simd::util::x86;
        with_sum_sumsq_loop!(
            vals,
            lanes: 4,
            simd: |ptr, i, sum, sumsq| {
                // SAFETY: `*i` is chunk-aligned within `vals`.
                unsafe { x86::accum_i32x4(sum, sumsq, ptr.add(*i)) };
                *i += 4;
            },
            tail_load: |p: *const i32| f64::from(*p)
        )
    }
}

#[cfg(target_arch = "aarch64")]
mod i32_aarch64 {
    use std::arch::aarch64::{
        vaddvq_f64, vcvt_f64_f32, vcvtq_f32_s32, vget_high_f32, vget_low_f32, vld1q_s32, vmulq_f64,
    };

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn i32_sum_sumsq_neon(vals: &[i32]) -> (f64, f64) {
        with_sum_sumsq_loop!(
            vals,
            lanes: 4,
            simd: |ptr, i, sum, sumsq| {
                // SAFETY: `*i` is chunk-aligned within `vals`.
                let v = unsafe { vld1q_s32(ptr.add(*i)) };
                let vf = vcvtq_f32_s32(v);
                let lo = vcvt_f64_f32(vget_low_f32(vf));
                let hi = vcvt_f64_f32(vget_high_f32(vf));
                *sum += vaddvq_f64(lo) + vaddvq_f64(hi);
                let sq_lo = vmulq_f64(lo, lo);
                let sq_hi = vmulq_f64(hi, hi);
                *sumsq += vaddvq_f64(sq_lo) + vaddvq_f64(sq_hi);
                *i += 4;
            },
            tail_load: |p: *const i32| f64::from(*p)
        )
    }
}

/// Sum and sum-of-squares for `i32` slabs promoted to `f64`.
#[must_use]
pub(crate) fn i32_sum_sumsq(vals: &[i32]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { i32_aarch64::i32_sum_sumsq_neon(vals) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { i32_arch::i32_sum_sumsq_sse2(vals) };
        }
    }
    scalar_i32_sum_sumsq(vals)
}

#[inline]
pub(crate) fn scalar_u8_sum_sumsq(vals: &[u8]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod u8_arch {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    #[inline]
    pub(super) unsafe fn accum_epi32_pd(sum: &mut f64, sumsq: &mut f64, reg: __m128i) {
        // SAFETY: caller enables SSE2 via `#[target_feature]`.
        unsafe {
            let lo_pd = _mm_cvtepi32_pd(reg);
            let hi_pd = _mm_cvtepi32_pd(_mm_shuffle_epi32(reg, 0xEE));
            let mut lo_arr = [0.0f64; 2];
            let mut hi_arr = [0.0f64; 2];
            _mm_storeu_pd(lo_arr.as_mut_ptr(), lo_pd);
            _mm_storeu_pd(hi_arr.as_mut_ptr(), hi_pd);
            *sum += lo_arr[0] + lo_arr[1] + hi_arr[0] + hi_arr[1];
            *sumsq += lo_arr[0] * lo_arr[0]
                + lo_arr[1] * lo_arr[1]
                + hi_arr[0] * hi_arr[0]
                + hi_arr[1] * hi_arr[1];
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn u8_sum_sumsq_sse2(vals: &[u8]) -> (f64, f64) {
        with_sum_sumsq_loop!(
            vals,
            lanes: 16,
            simd: |ptr, i, sum, sumsq| {
                // SAFETY: `*i` is chunk-aligned within `vals`.
                unsafe {
                    let zero = _mm_setzero_si128();
                    let v = _mm_loadu_si128(ptr.add(*i).cast::<__m128i>());
                    let lo16 = _mm_unpacklo_epi8(v, zero);
                    let hi16 = _mm_unpackhi_epi8(v, zero);
                    accum_epi32_pd(sum, sumsq, _mm_unpacklo_epi16(lo16, zero));
                    accum_epi32_pd(sum, sumsq, _mm_unpackhi_epi16(lo16, zero));
                    accum_epi32_pd(sum, sumsq, _mm_unpacklo_epi16(hi16, zero));
                    accum_epi32_pd(sum, sumsq, _mm_unpackhi_epi16(hi16, zero));
                }
                *i += 16;
            },
            tail_load: |p: *const u8| f64::from(*p)
        )
    }
}

/// Sum and sum-of-squares for `u8` slabs promoted to `f64`.
#[must_use]
pub(crate) fn u8_sum_sumsq(vals: &[u8]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { u8_arch::u8_sum_sumsq_sse2(vals) };
        }
    }
    scalar_u8_sum_sumsq(vals)
}

#[inline]
pub(crate) fn i32_min_max(vals: &[i32]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn u8_min_max(vals: &[u8]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn scalar_i64_sum_sumsq(vals: &[i64]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = v as f64;
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod i64_arch {
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn i64_sum_sumsq_sse2(vals: &[i64]) -> (f64, f64) {
        use crate::query::fold::variance_simd::util::x86;
        with_sum_sumsq_loop!(
            vals,
            lanes: 2,
            simd: |ptr, i, sum, sumsq| {
                // SAFETY: `*i` is chunk-aligned within `vals`.
                unsafe { x86::accum_i64x2(sum, sumsq, ptr.add(*i)) };
                *i += 2;
            },
            tail_load: |p: *const i64| *p as f64
        )
    }
}

/// Sum and sum-of-squares for `i64` slabs promoted to `f64`.
#[must_use]
pub(crate) fn i64_sum_sumsq(vals: &[i64]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { i64_arch::i64_sum_sumsq_sse2(vals) };
        }
    }
    scalar_i64_sum_sumsq(vals)
}

#[inline]
pub(crate) fn scalar_u32_sum_sumsq(vals: &[u32]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod u32_arch {
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn accum_u32_pair(sum: &mut f64, sumsq: &mut f64, pair: u64) {
        let x0 = (pair & 0xFFFF_FFFF) as f64;
        let x1 = (pair >> 32) as f64;
        *sum += x0 + x1;
        *sumsq += x0 * x0 + x1 * x1;
    }

    /// SSE2-only: `_mm_cvtepu32_pd` is AVX-512VL and SIGILLs on typical CI hosts.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn u32_sum_sumsq_sse2(vals: &[u32]) -> (f64, f64) {
        with_sum_sumsq_loop!(
            vals,
            lanes: 4,
            simd: |ptr, i, sum, sumsq| {
                // SAFETY: `*i` is chunk-aligned within `vals`.
                unsafe {
                    let v = _mm_loadu_si128(ptr.add(*i).cast::<__m128i>());
                    accum_u32_pair(sum, sumsq, _mm_cvtsi128_si64(v) as u64);
                    accum_u32_pair(
                        sum,
                        sumsq,
                        _mm_cvtsi128_si64(_mm_unpackhi_epi64(v, v)) as u64,
                    );
                }
                *i += 4;
            },
            tail_load: |p: *const u32| f64::from(*p)
        )
    }
}

/// Sum and sum-of-squares for `u32` slabs promoted to `f64`.
#[must_use]
pub(crate) fn u32_sum_sumsq(vals: &[u32]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { u32_arch::u32_sum_sumsq_sse2(vals) };
        }
    }
    scalar_u32_sum_sumsq(vals)
}

#[inline]
pub(crate) fn scalar_u64_sum_sumsq(vals: &[u64]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = v as f64;
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod u64_arch {
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn u64_sum_sumsq_sse2(vals: &[u64]) -> (f64, f64) {
        use crate::query::fold::variance_simd::util::x86;
        with_sum_sumsq_loop!(
            vals,
            lanes: 2,
            simd: |ptr, i, sum, sumsq| {
                // SAFETY: `*i` is chunk-aligned within `vals`; `u64` lanes match `i64` load width.
                unsafe { x86::accum_i64x2(sum, sumsq, ptr.add(*i).cast::<i64>()) };
                *i += 2;
            },
            tail_load: |p: *const u64| *p as f64
        )
    }
}

/// Sum and sum-of-squares for `u64` slabs promoted to `f64`.
#[must_use]
pub(crate) fn u64_sum_sumsq(vals: &[u64]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { u64_arch::u64_sum_sumsq_sse2(vals) };
        }
    }
    scalar_u64_sum_sumsq(vals)
}

#[inline]
pub(crate) fn scalar_u16_sum_sumsq(vals: &[u16]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "x86_64")]
mod u16_arch {
    use super::u8_arch::accum_epi32_pd;
    #[allow(clippy::wildcard_imports)]
    use std::arch::x86_64::*;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn u16_sum_sumsq_sse2(vals: &[u16]) -> (f64, f64) {
        with_sum_sumsq_loop!(
            vals,
            lanes: 8,
            simd: |ptr, i, sum, sumsq| {
                // SAFETY: `*i` is chunk-aligned within `vals`.
                unsafe {
                    let zero = _mm_setzero_si128();
                    let v = _mm_loadu_si128(ptr.add(*i).cast::<__m128i>());
                    accum_epi32_pd(sum, sumsq, _mm_unpacklo_epi16(v, zero));
                    accum_epi32_pd(sum, sumsq, _mm_unpackhi_epi16(v, zero));
                }
                *i += 8;
            },
            tail_load: |p: *const u16| f64::from(*p)
        )
    }
}

/// Sum and sum-of-squares for `u16` slabs promoted to `f64`.
#[must_use]
pub(crate) fn u16_sum_sumsq(vals: &[u16]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { u16_arch::u16_sum_sumsq_sse2(vals) };
        }
    }
    scalar_u16_sum_sumsq(vals)
}

#[inline]
pub(crate) fn i64_min_max(vals: &[i64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = v as f64;
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn u32_min_max(vals: &[u32]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn u64_min_max(vals: &[u64]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = v as f64;
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn i16_sum_sumsq(vals: &[i16]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[inline]
pub(crate) fn i16_min_max(vals: &[i16]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[inline]
pub(crate) fn u16_min_max(vals: &[u16]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}
