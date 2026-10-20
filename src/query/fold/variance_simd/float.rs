#[inline]
pub(crate) fn scalar_f32_sum_sumsq(vals: &[f32]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        let x = f64::from(v);
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[cfg(target_arch = "aarch64")]
mod arch {
    use std::arch::aarch64::{
        vaddvq_f64, vcvt_f64_f32, vget_high_f32, vget_low_f32, vld1q_f32, vmulq_f64,
    };

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn f32_sum_sumsq(vals: &[f32]) -> (f64, f64) {
        with_sum_sumsq_loop!(
            vals,
            lanes: 4,
            simd: |ptr, i, sum, sumsq| {
                // SAFETY: `*i` is chunk-aligned within `vals`.
                let v = unsafe { vld1q_f32(ptr.add(*i)) };
                let lo = vcvt_f64_f32(vget_low_f32(v));
                let hi = vcvt_f64_f32(vget_high_f32(v));
                *sum += vaddvq_f64(lo) + vaddvq_f64(hi);
                let sq_lo = vmulq_f64(lo, lo);
                let sq_hi = vmulq_f64(hi, hi);
                *sumsq += vaddvq_f64(sq_lo) + vaddvq_f64(sq_hi);
                *i += 4;
            },
            tail_load: |p: *const f32| f64::from(*p)
        )
    }
}

#[cfg(target_arch = "x86_64")]
mod arch {
    use crate::query::fold::variance_simd::util::x86;

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn f32_sum_sumsq_sse2(vals: &[f32]) -> (f64, f64) {
        with_sum_sumsq_loop!(
            vals,
            lanes: 4,
            simd: |ptr, i, sum, sumsq| {
                // SAFETY: `*i` is chunk-aligned within `vals`.
                unsafe { x86::accum_f32x4(sum, sumsq, ptr.add(*i)) };
                *i += 4;
            },
            tail_load: |p: *const f32| f64::from(*p)
        )
    }
}

/// Sum and sum-of-squares for population variance (`f64` accumulators).
#[must_use]
pub(crate) fn f32_sum_sumsq(vals: &[f32]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { arch::f32_sum_sumsq(vals) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { arch::f32_sum_sumsq_sse2(vals) };
        }
    }
    scalar_f32_sum_sumsq(vals)
}

#[inline]
pub(crate) fn scalar_f32_min_max(vals: &[f32]) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in vals {
        let vd = f64::from(v);
        min = min.min(vd);
        max = max.max(vd);
    }
    (min, max)
}

#[cfg(target_arch = "aarch64")]
mod minmax_arch {
    use std::arch::aarch64::{
        vdupq_n_f32, vget_high_f32, vget_lane_f32, vget_low_f32, vld1q_f32, vmaxq_f32, vminq_f32,
        vpmax_f32, vpmin_f32,
    };

    #[inline]
    unsafe fn horizontal_max_f32(v: std::arch::aarch64::float32x4_t) -> f32 {
        // SAFETY: caller enables NEON via `#[target_feature]`.
        unsafe {
            let pair = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
            let one = vpmax_f32(pair, pair);
            vget_lane_f32(one, 0)
        }
    }

    #[inline]
    unsafe fn horizontal_min_f32(v: std::arch::aarch64::float32x4_t) -> f32 {
        // SAFETY: caller enables NEON via `#[target_feature]`.
        unsafe {
            let pair = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
            let one = vpmin_f32(pair, pair);
            vget_lane_f32(one, 0)
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn f32_min_max(vals: &[f32]) -> (f64, f64) {
        with_f32_min_max_loop!(
            vals,
            lanes: 4,
            init: (vdupq_n_f32(f32::INFINITY), vdupq_n_f32(f32::NEG_INFINITY)),
            simd: |ptr, i, min_v, max_v| {
                // SAFETY: `*i` is chunk-aligned within `vals`.
                let v = unsafe { vld1q_f32(ptr.add(*i)) };
                *min_v = vminq_f32(*min_v, v);
                *max_v = vmaxq_f32(*max_v, v);
                *i += 4;
            },
            finalize: |min_vec, max_vec| -> (min_f, max_f) {
                (
                    unsafe { horizontal_min_f32(min_vec) },
                    unsafe { horizontal_max_f32(max_vec) },
                )
            }
        )
    }
}

#[cfg(target_arch = "x86_64")]
mod minmax_arch {
    use std::arch::x86_64::{
        __m128, _mm_cvtss_f32, _mm_loadu_ps, _mm_max_ps, _mm_max_ss, _mm_min_ps, _mm_min_ss,
        _mm_movehl_ps, _mm_set1_ps, _mm_shuffle_ps,
    };

    #[inline]
    unsafe fn horizontal_min_f32(v: __m128) -> f32 {
        // SAFETY: caller enables SSE2 via `#[target_feature]`.
        unsafe {
            let shuf = _mm_movehl_ps(v, v);
            let mins = _mm_min_ps(v, shuf);
            let shuf2 = _mm_shuffle_ps(mins, mins, 0x01);
            _mm_cvtss_f32(_mm_min_ss(mins, shuf2))
        }
    }

    #[inline]
    unsafe fn horizontal_max_f32(v: __m128) -> f32 {
        // SAFETY: caller enables SSE2 via `#[target_feature]`.
        unsafe {
            let shuf = _mm_movehl_ps(v, v);
            let maxs = _mm_max_ps(v, shuf);
            let shuf2 = _mm_shuffle_ps(maxs, maxs, 0x01);
            _mm_cvtss_f32(_mm_max_ss(maxs, shuf2))
        }
    }

    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn f32_min_max_sse2(vals: &[f32]) -> (f64, f64) {
        with_f32_min_max_loop!(
            vals,
            lanes: 4,
            init: (_mm_set1_ps(f32::INFINITY), _mm_set1_ps(f32::NEG_INFINITY)),
            simd: |ptr, i, min_v, max_v| {
                // SAFETY: `*i` is chunk-aligned within `vals`.
                let v = unsafe { _mm_loadu_ps(ptr.add(*i)) };
                *min_v = _mm_min_ps(*min_v, v);
                *max_v = _mm_max_ps(*max_v, v);
                *i += 4;
            },
            finalize: |min_vec, max_vec| -> (min_f, max_f) {
                (
                    unsafe { horizontal_min_f32(min_vec) },
                    unsafe { horizontal_max_f32(max_vec) },
                )
            }
        )
    }
}

/// Min and max over an `f32` slab (`f64` accumulators, matches scalar fold promotion).
#[must_use]
pub(crate) fn f32_min_max(vals: &[f32]) -> (f64, f64) {
    if vals.is_empty() {
        return (f64::INFINITY, f64::NEG_INFINITY);
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { minmax_arch::f32_min_max(vals) };
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { minmax_arch::f32_min_max_sse2(vals) };
        }
    }
    scalar_f32_min_max(vals)
}

/// Sum and sum-of-squares for population variance over an `f64` slice.
#[must_use]
pub(crate) fn f64_sum_sumsq(vals: &[f64]) -> (f64, f64) {
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    for &v in vals {
        sum += v;
        sumsq += v * v;
    }
    (sum, sumsq)
}

const F16_SIMD_CHUNK: usize = 8;

/// Sum and sum-of-squares for `f16` slabs (via `f32` SIMD chunks).
#[must_use]
pub(crate) fn f16_sum_sumsq(vals: &[half::f16]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let mut buf = [0.0f32; F16_SIMD_CHUNK];
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut i = 0usize;
    while i + F16_SIMD_CHUNK <= vals.len() {
        for (j, slot) in buf.iter_mut().enumerate() {
            *slot = f32::from(vals[i + j]);
        }
        let (s, sq) = f32_sum_sumsq(&buf);
        sum += s;
        sumsq += sq;
        i += F16_SIMD_CHUNK;
    }
    while i < vals.len() {
        let x = f64::from(f32::from(vals[i]));
        sum += x;
        sumsq += x * x;
        i += 1;
    }
    (sum, sumsq)
}

/// Min and max for `f16` slabs (via `f32` SIMD chunks).
#[must_use]
pub(crate) fn f16_min_max(vals: &[half::f16]) -> (f64, f64) {
    if vals.is_empty() {
        return (f64::INFINITY, f64::NEG_INFINITY);
    }
    let mut buf = [0.0f32; F16_SIMD_CHUNK];
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut i = 0usize;
    while i + F16_SIMD_CHUNK <= vals.len() {
        for (j, slot) in buf.iter_mut().enumerate() {
            *slot = f32::from(vals[i + j]);
        }
        let (slab_min, slab_max) = f32_min_max(&buf);
        min = min.min(slab_min);
        max = max.max(slab_max);
        i += F16_SIMD_CHUNK;
    }
    while i < vals.len() {
        let vd = f64::from(f32::from(vals[i]));
        min = min.min(vd);
        max = max.max(vd);
        i += 1;
    }
    (min, max)
}
