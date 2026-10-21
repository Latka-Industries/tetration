//! Shared scaffolding for SIMD sum/sumsq (and related) slab loops.

#[inline]
pub(super) const fn simd_aligned_end(len: usize, lanes: usize) -> usize {
    len - (len % lanes)
}

/// Prologue, SIMD main loop, and scalar tail for population `sum` only over a slice.
macro_rules! with_sum_loop {
    (
        $vals:expr,
        lanes: $lanes:expr,
        simd: |$ptr:ident, $i:ident, $sum:ident| $body:expr,
        tail_load: $tail_load:expr
    ) => {{
        let ptr = $vals.as_ptr();
        let len = $vals.len();
        let mut sum = 0.0f64;
        let mut i = 0usize;
        let simd_end = $crate::query::fold::variance_simd::util::simd_aligned_end(len, $lanes);
        while i < simd_end {
            let $ptr = ptr;
            let $i = &mut i;
            let $sum = &mut sum;
            $body;
        }
        while i < len {
            // SAFETY: `i` is in `0..len` for the same slice as `ptr`.
            let x: f64 = unsafe { ($tail_load)(ptr.add(i)) };
            sum += x;
            i += 1;
        }
        sum
    }};
}

/// Prologue, SIMD main loop, and scalar tail for population `sum` / `sumsq` over a slice.
macro_rules! with_sum_sumsq_loop {
    (
        $vals:expr,
        lanes: $lanes:expr,
        simd: |$ptr:ident, $i:ident, $sum:ident, $sumsq:ident| $body:expr,
        tail_load: $tail_load:expr
    ) => {{
        let ptr = $vals.as_ptr();
        let len = $vals.len();
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut i = 0usize;
        let simd_end = $crate::query::fold::variance_simd::util::simd_aligned_end(len, $lanes);
        while i < simd_end {
            let $ptr = ptr;
            let $i = &mut i;
            let $sum = &mut sum;
            let $sumsq = &mut sumsq;
            $body;
        }
        while i < len {
            // SAFETY: `i` is in `0..len` for the same slice as `ptr`.
            let x: f64 = unsafe { ($tail_load)(ptr.add(i)) };
            sum += x;
            sumsq += x * x;
            i += 1;
        }
        (sum, sumsq)
    }};
}

/// Aligned SIMD main loop + scalar tail for min/max over `f32` (horizontal reduce after loop).
macro_rules! with_f32_min_max_loop {
    (
        $vals:expr,
        lanes: $lanes:expr,
        init: ($min_init:expr, $max_init:expr),
        simd: |$ptr:ident, $i:ident, $min_v:ident, $max_v:ident| $body:expr,
        finalize: |$min_vec:ident, $max_vec:ident| -> ($min_f:ident, $max_f:ident) $horiz:block
    ) => {{
        let ptr = $vals.as_ptr();
        let len = $vals.len();
        let mut min_v = $min_init;
        let mut max_v = $max_init;
        let mut i = 0usize;
        let simd_end = $crate::query::fold::variance_simd::util::simd_aligned_end(len, $lanes);
        while i < simd_end {
            let $ptr = ptr;
            let $i = &mut i;
            let $min_v = &mut min_v;
            let $max_v = &mut max_v;
            $body;
        }
        let (mut $min_f, mut $max_f) = {
            let $min_vec = min_v;
            let $max_vec = max_v;
            $horiz
        };
        while i < len {
            let x = unsafe { *ptr.add(i) };
            $min_f = $min_f.min(x);
            $max_f = $max_f.max(x);
            i += 1;
        }
        (f64::from($min_f), f64::from($max_f))
    }};
}

#[cfg(target_arch = "x86_64")]
pub(super) mod x86 {
    use std::arch::x86_64::{
        __m128i, _mm_cvtepi32_pd, _mm_cvtps_pd, _mm_cvtsi128_si64, _mm_loadu_ps, _mm_loadu_si128,
        _mm_movehl_ps, _mm_shuffle_epi32, _mm_storeu_pd, _mm_unpackhi_epi64,
    };

    #[inline]
    fn accum_f64_quad(sum: &mut f64, sumsq: &mut f64, lo: [f64; 2], hi: [f64; 2]) {
        *sum += lo[0] + lo[1] + hi[0] + hi[1];
        *sumsq += lo[0] * lo[0] + lo[1] * lo[1] + hi[0] * hi[0] + hi[1] * hi[1];
    }

    #[inline]
    fn accum_f64_pair(sum: &mut f64, sumsq: &mut f64, lo: f64, hi: f64) {
        *sum += lo + hi;
        *sumsq += lo * lo + hi * hi;
    }

    /// Four `f32` lanes → `f64` sum accumulator (SSE2).
    #[inline]
    pub unsafe fn accum_f32x4_sum(sum: &mut f64, ptr: *const f32) {
        // SAFETY: caller aligns `ptr` to a 4-lane chunk inside the slice.
        let (lo_arr, hi_arr) = unsafe {
            let v = _mm_loadu_ps(ptr);
            let lo = _mm_cvtps_pd(v);
            let hi = _mm_cvtps_pd(_mm_movehl_ps(v, v));
            let mut lo_arr = [0.0f64; 2];
            let mut hi_arr = [0.0f64; 2];
            _mm_storeu_pd(lo_arr.as_mut_ptr(), lo);
            _mm_storeu_pd(hi_arr.as_mut_ptr(), hi);
            (lo_arr, hi_arr)
        };
        *sum += lo_arr[0] + lo_arr[1] + hi_arr[0] + hi_arr[1];
    }

    /// Four `f32` lanes → `f64` sum/sumsq accumulators (SSE2).
    #[inline]
    pub unsafe fn accum_f32x4(sum: &mut f64, sumsq: &mut f64, ptr: *const f32) {
        // SAFETY: caller aligns `ptr` to a 4-lane chunk inside the slice.
        let (lo_arr, hi_arr) = unsafe {
            let v = _mm_loadu_ps(ptr);
            let lo = _mm_cvtps_pd(v);
            let hi = _mm_cvtps_pd(_mm_movehl_ps(v, v));
            let mut lo_arr = [0.0f64; 2];
            let mut hi_arr = [0.0f64; 2];
            _mm_storeu_pd(lo_arr.as_mut_ptr(), lo);
            _mm_storeu_pd(hi_arr.as_mut_ptr(), hi);
            (lo_arr, hi_arr)
        };
        accum_f64_quad(sum, sumsq, lo_arr, hi_arr);
    }

    /// Four `i32` lanes promoted to `f64` sum/sumsq (SSE2).
    #[inline]
    pub unsafe fn accum_i32x4(sum: &mut f64, sumsq: &mut f64, ptr: *const i32) {
        let (lo_arr, hi_arr) = unsafe {
            let v = _mm_loadu_si128(ptr.cast::<__m128i>());
            let lo = _mm_cvtepi32_pd(v);
            let hi = _mm_cvtepi32_pd(_mm_shuffle_epi32(v, 0xEE));
            let mut lo_arr = [0.0f64; 2];
            let mut hi_arr = [0.0f64; 2];
            _mm_storeu_pd(lo_arr.as_mut_ptr(), lo);
            _mm_storeu_pd(hi_arr.as_mut_ptr(), hi);
            (lo_arr, hi_arr)
        };
        accum_f64_quad(sum, sumsq, lo_arr, hi_arr);
    }

    /// Two `i64` / `u64` lanes as `f64` (SSE2).
    #[inline]
    pub unsafe fn accum_i64x2(sum: &mut f64, sumsq: &mut f64, ptr: *const i64) {
        // SAFETY: caller aligns `ptr` to a 2-lane chunk inside the slice.
        unsafe {
            let v = _mm_loadu_si128(ptr.cast::<__m128i>());
            let lo = _mm_cvtsi128_si64(v) as f64;
            let hi = _mm_cvtsi128_si64(_mm_unpackhi_epi64(v, v)) as f64;
            accum_f64_pair(sum, sumsq, lo, hi);
        }
    }
}
