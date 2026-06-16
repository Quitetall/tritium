//! The two ternary mpGEMM kernels and their runtime dispatch.
//!
//! Both kernels compute the identical contraction
//!
//! ```text
//! out[m, n] = scale[n] · Σ_k act[m, k] · trit[n, k]
//! ```
//!
//! over already-unpacked [`Trit`]s (`{-1, 0, +1}`), expressed in the reference's
//! **add / subtract / skip** form — never a literal multiply by the trit.
//!
//! - [`scalar_mpgemm`] delegates to [`tritium_core::reference_mpgemm`], so it is
//!   correct by construction and serves as both the fallback and the ground truth
//!   the AVX2 path is graded against.
//! - [`avx2_mpgemm`] is an `#[target_feature(enable = "avx2")]` kernel using
//!   `core::arch::x86_64` intrinsics. It vectorises the trit decode and the
//!   per-element add/sub/skip into a signed contribution vector, then folds that
//!   contribution **sequentially in f32 in k-order** — reproducing the reference's
//!   single-accumulator rounding bit-for-bit (any reordered or re-widened
//!   reduction drifts by ~1e-4 at K=512, the conformance floor for near-zero
//!   outputs). It therefore matches the scalar/reference output exactly.
//!
//! [`dispatch_mpgemm`] picks the AVX2 path at runtime when the host advertises
//! `avx2`, and the scalar path otherwise, parallelising over the independent `M`
//! rows with `rayon`.

use rayon::prelude::*;
use tritium_core::{GemmShape, Trit, reference_mpgemm};

use tritium_spec::BackendError;

/// Run the ternary mpGEMM with the best kernel the host supports, parallelising
/// over the independent `M` rows.
///
/// `weights` are the already-unpacked `[N, K]` trits (output-major). The shape
/// and buffer lengths are validated up front; on mismatch the scalar reference's
/// typed error is surfaced as [`BackendError::Core`].
///
/// The `M` rows of a mpGEMM are mutually independent (each reads its own `act`
/// row and writes its own `out` row, sharing only the read-only weights/scales),
/// so the work splits cleanly across `rayon`'s thread pool. Each task runs the
/// same single-threaded kernel on a disjoint chunk of rows, so the numeric result
/// is identical to the serial path regardless of thread count — the per-row
/// accumulation order is unchanged.
///
/// # Errors
/// [`BackendError::Core`] if a buffer length disagrees with `shape`.
pub(crate) fn dispatch_mpgemm(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    let GemmShape { m, n, k } = shape;
    // Validate here so the per-chunk shapes below are exact and the kernels never
    // index out of bounds.
    if act.len() != m * k || weights.len() != n * k || scales.len() != n || out.len() != m * n {
        // Reuse the reference's typed `ShapeMismatch` for the precise lengths.
        return reference_mpgemm(act, weights, scales, shape, out).map_err(BackendError::Core);
    }
    if m == 0 || n == 0 {
        return Ok(());
    }

    // Split the M rows into chunks and run one kernel invocation per chunk in
    // parallel. `par_chunks_mut(k|n)` yields disjoint, non-overlapping slices, so
    // there is no aliasing between tasks.
    let chunk_rows = row_chunk(m, n, k);
    out.par_chunks_mut(chunk_rows * n)
        .zip(act.par_chunks(chunk_rows * k))
        .try_for_each(|(out_chunk, act_chunk)| {
            let rows = out_chunk.len() / n;
            let chunk_shape = GemmShape::new(rows, n, k);
            run_chunk(act_chunk, weights, scales, chunk_shape, out_chunk)
        })
}

/// Rows per parallel task. A coarse heuristic: keep at least a few thousand
/// add/sub/skip ops per task so the rayon overhead is amortised, and never split
/// finer than one row.
fn row_chunk(m: usize, n: usize, k: usize) -> usize {
    // Target ~64K inner ops per chunk; clamp to [1, m].
    const TARGET_OPS: usize = 1 << 16;
    let per_row = n.saturating_mul(k).max(1);
    (TARGET_OPS / per_row).clamp(1, m)
}

/// Run the best available kernel on one chunk of `M` rows (serial within the
/// chunk; the chunk itself is one rayon task).
fn run_chunk(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    #[cfg(target_arch = "x86_64")]
    {
        // Prefer the widest kernel the host can execute. AVX-512 (with the BW +
        // VL subsets, needed for the byte-lane trit compares that emit a mask from
        // a 128-bit operand) is selected first where present; otherwise AVX2;
        // otherwise scalar. The AVX-512 kernel compiles on every x86-64 target but
        // only runs on an AVX-512 host, so on the AVX2 build box this branch is
        // inert and the AVX2 path below runs. All three share the reference's
        // k-order f32 fold, so the choice is bit-neutral.
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
        {
            // SAFETY: `avx512_mpgemm` requires the `avx512f` + `avx512bw` +
            // `avx512vl` target features, all just confirmed present by the runtime
            // checks above. All buffer lengths were validated by `dispatch_mpgemm`
            // and are re-validated inside the call before any intrinsic touches
            // memory.
            return unsafe {
                crate::simd::avx512::avx512_mpgemm(act, weights, scales, shape, out)
            };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2_mpgemm` requires the `avx2` target feature, which the
            // runtime check immediately above has just confirmed is present on
            // this CPU. All buffer lengths were validated by `dispatch_mpgemm` and
            // are re-validated inside the call before any intrinsic touches memory.
            return unsafe { avx2_mpgemm(act, weights, scales, shape, out) };
        }
    }
    // Baseline NEON is mandatory on aarch64; select it unconditionally there.
    // NEON shares the x86 kernels' per-element k-order f32 fold, so its output is
    // bit-identical to the scalar reference (and to the AVX2/AVX-512 paths on
    // x86). Written as the function tail (not an early `return`) so the trailing
    // scalar fallback is only compiled where no SIMD kernel applies — no
    // unreachable-code warning.
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `neon_mpgemm` requires the `neon` target feature. On aarch64,
        // baseline NEON (`Advanced SIMD`) is part of the mandatory architecture —
        // it is always present, so the precondition holds unconditionally on this
        // `#[cfg(target_arch = "aarch64")]` path. All buffer lengths were
        // validated by `dispatch_mpgemm` and are re-validated inside the call.
        unsafe { crate::simd::neon::neon_mpgemm(act, weights, scales, shape, out) }
    }
    // Universal terminal fallback: the bit-exact scalar reference. Reached on
    // x86-64 with neither AVX-512 nor AVX2 (very old CPUs / restricted CPUID), or
    // any architecture with no SIMD kernel. Kept deliberately bit-exact (not the
    // tolerance-only T-MAC LUT) so cross-host output stays byte-reproducible for
    // golden-file / deterministic-checkpoint callers; the LUT
    // (`crate::simd::lut::lut_mpgemm`) is implemented and tested but its
    // production wiring is the per-ISA SIMD gather, deferred to a later step.
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_mpgemm(act, weights, scales, shape, out)
    }
}

/// The scalar ground-truth kernel: delegate to [`tritium_core::reference_mpgemm`].
///
/// # Errors
/// [`BackendError::Core`] if a buffer length disagrees with `shape`.
pub(crate) fn scalar_mpgemm(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    reference_mpgemm(act, weights, scales, shape, out).map_err(BackendError::Core)
}

/// AVX2 ternary mpGEMM. Same contraction as [`scalar_mpgemm`], vectorised over
/// the `K` dimension eight `f32` lanes at a time.
///
/// For each output it vectorises the trit decode and the per-element
/// add/sub/skip — masking the activation to `+a` on `+1` lanes, `-a` on `-1`
/// lanes, `0` on `0` lanes (no multiply by the trit) — into a signed contribution
/// vector, then folds that vector **sequentially in f32 in k-order**. That fold
/// reproduces the reference's single-`f32`-accumulator rounding bit-for-bit:
/// `acc + 0.0 == acc` (skip) and `acc + (-a) == acc - a` (sub) are exact in IEEE
/// round-to-nearest, so the result is identical to [`scalar_mpgemm`], not merely
/// within tolerance. (A reordered or f64-widened SIMD reduction would be *more*
/// accurate than the reference yet drift ~1e-4 from it at K=512 — past the
/// conformance floor for near-zero outputs.) The `K` tail that does not fill a
/// lane is folded with the same scalar add/sub/skip loop, in k-order.
///
/// # Safety
/// The caller must ensure the `avx2` target feature is available on the host
/// (checked via `is_x86_feature_detected!("avx2")`). Calling this on a CPU
/// without AVX2 is undefined behaviour.
///
/// # Errors
/// [`BackendError::Core`] if a buffer length disagrees with `shape`; in that case
/// no intrinsic runs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_mpgemm(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    use core::arch::x86_64::{
        _mm256_blendv_ps, _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_cvtepi8_epi32,
        _mm256_loadu_ps, _mm256_set1_epi32, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
    };

    let GemmShape { m, n, k } = shape;
    // Validate up front so no intrinsic ever reads out of bounds. This mirrors
    // the reference's checks; on mismatch we return before touching SIMD.
    if act.len() != m * k || weights.len() != n * k || scales.len() != n || out.len() != m * n {
        // Defer to the scalar reference purely for its typed error; it will
        // detect the same inconsistency and return `ShapeMismatch`.
        return scalar_mpgemm(act, weights, scales, shape, out);
    }

    // Eight f32 lanes per AVX2 register.
    const LANES: usize = 8;
    let k_simd = k - (k % LANES);

    // Trits are `#[repr(transparent)]` over i8, so a `&[Trit]` is bit-identical
    // to a `&[i8]` whose elements are all in `{-1, 0, 1}`. Reinterpret for the
    // byte loads the intrinsics need.
    // SAFETY: `Trit` is `#[repr(transparent)]` over `i8` and has the same size
    // and alignment; the slice length is unchanged, so the new slice covers
    // exactly the same valid, initialised, immutably-borrowed bytes. Every value
    // is in `{-1,0,1}`, a subset of valid `i8`.
    let weights_i8: &[i8] =
        unsafe { core::slice::from_raw_parts(weights.as_ptr().cast::<i8>(), weights.len()) };

    // The reference walks `k` in one sequential **f32** accumulator. Over K up to
    // 512 that single accumulator's own rounding (partial sums reach ~150 before
    // cancelling down to ~0.2) is itself ~1e-4 — right at the conformance floor.
    // Any SIMD reduction that reorders or re-widens those adds produces a
    // *different* ~1e-4 rounding and drifts past the tolerance. So this kernel
    // uses SIMD only to decode the trits and form the signed per-element
    // contribution (`+a` / `-a` / `0`, the add/sub/skip), then accumulates that
    // contribution **sequentially in f32 in k-order**, reproducing the reference's
    // accumulation bit-for-bit. `acc + 0.0 == acc`, `acc + (-a) == acc - a`, so the
    // signed-contribution fold is identical to the reference's branchy add/sub/skip.
    //
    // Inside a `#[target_feature(enable = "avx2")]` function the AVX2 intrinsics
    // are callable without an `unsafe` block; `unsafe` is confined to the
    // intrinsics that dereference a raw pointer.
    let mut signed_buf = vec![0.0f32; k_simd];
    for mi in 0..m {
        let arow = &act[mi * k..mi * k + k];
        for ni in 0..n {
            let wrow = &weights_i8[ni * k..ni * k + k];
            let zero = _mm256_setzero_ps();

            // Vectorised pass: fill `signed_buf[ki]` with the add/sub/skip value.
            let mut ki = 0;
            while ki < k_simd {
                // SAFETY: `ki + LANES <= k_simd <= k = arow.len()`, so the eight
                // f32 starting at `arow.as_ptr().add(ki)` are in bounds and
                // initialised. `loadu` permits any alignment.
                let a = unsafe { _mm256_loadu_ps(arow.as_ptr().add(ki)) };

                // Widen 8 packed i8 trits to 8 i32 lanes.
                // SAFETY: `ki + LANES <= k = wrow.len()`, so the 8 bytes at
                // `wrow.as_ptr().add(ki)` are in bounds and initialised.
                // `_mm_loadl_epi64` reads exactly those 8 bytes; the pointer cast
                // is between integer element types of compatible size with no
                // alignment requirement (loadl is an unaligned 64-bit load).
                let lo = unsafe {
                    core::arch::x86_64::_mm_loadl_epi64(
                        wrow.as_ptr().add(ki).cast::<core::arch::x86_64::__m128i>(),
                    )
                };
                let t = _mm256_cvtepi8_epi32(lo);

                // pos_mask = (trit == 1), neg_mask = (trit == -1), as i32 lanes,
                // reinterpreted as float masks for blendv.
                let pos_maskf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(t, _mm256_set1_epi32(1)));
                let neg_maskf = _mm256_castsi256_ps(_mm256_cmpeq_epi32(t, _mm256_set1_epi32(-1)));

                // Signed contribution per lane: `+a` where the trit is +1, `-a`
                // where it is -1, `0` where it is 0 (the skip). Two blends, no
                // multiply: select `+a` on the +1 lanes, then overwrite the -1
                // lanes with `-a`. The two masks are mutually exclusive.
                let neg_a = _mm256_sub_ps(zero, a);
                let plus = _mm256_blendv_ps(zero, a, pos_maskf);
                let signed = _mm256_blendv_ps(plus, neg_a, neg_maskf);

                // SAFETY: `ki + LANES <= k_simd = signed_buf.len()`, so the 8-wide
                // store is in bounds; `storeu` has no alignment requirement.
                unsafe { _mm256_storeu_ps(signed_buf.as_mut_ptr().add(ki), signed) };

                ki += LANES;
            }

            // Sequential f32 fold over the signed contributions, in k-order —
            // bit-identical to the reference's accumulator.
            let mut acc = 0.0f32;
            for &s in &signed_buf {
                acc += s;
            }
            // Tail in the same k-order, add/sub/skip in f32.
            for kt in k_simd..k {
                match wrow[kt] {
                    1 => acc += arow[kt],
                    -1 => acc -= arow[kt],
                    _ => {}
                }
            }

            out[mi * n + ni] = acc * scales[ni];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_core::Trit;

    fn trits(vals: &[i8]) -> Vec<Trit> {
        vals.iter().map(|&v| Trit::from_i8(v).unwrap()).collect()
    }

    #[test]
    fn scalar_matches_reference_hand_case() {
        // 1 row, 1 output, k=3: [+1,-1,+1] · [10,3,5] = 12, scaled by 2 -> 24.
        let act = [10.0, 3.0, 5.0];
        let w = trits(&[1, -1, 1]);
        let mut out = [0.0; 1];
        scalar_mpgemm(&act, &w, &[2.0], GemmShape::new(1, 1, 3), &mut out).unwrap();
        assert_eq!(out[0], 24.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("avx2 not detected on this host — skipping AVX2 kernel unit test");
            return;
        }
        let mut s = 0x1234_5678u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // Cover a ragged tail (16 + 3) and the block-aligned K=512 where the
        // reference's f32 cancellation is worst — the kernel must reproduce the
        // reference accumulation bit-for-bit, not merely within 1e-4.
        for &(m, n, k) in &[(3usize, 5usize, 19usize), (2, 4, 512)] {
            let act: Vec<f32> = (0..m * k)
                .map(|_| (next() % 20000) as f32 / 100.0 - 100.0)
                .collect();
            let w = trits(
                &(0..n * k)
                    .map(|_| (next() % 3) as i8 - 1)
                    .collect::<Vec<_>>(),
            );
            let scales: Vec<f32> = (0..n).map(|_| (next() % 50) as f32 / 25.0 + 0.1).collect();
            let shape = GemmShape::new(m, n, k);

            let mut want = vec![0.0f32; m * n];
            scalar_mpgemm(&act, &w, &scales, shape, &mut want).unwrap();
            let mut got = vec![0.0f32; m * n];
            // SAFETY: avx2 was just confirmed available by the guard above.
            unsafe { avx2_mpgemm(&act, &w, &scales, shape, &mut got).unwrap() };

            for (g, e) in got.iter().zip(&want) {
                assert_eq!(
                    g.to_bits(),
                    e.to_bits(),
                    "got {g}, want {e} (shape {shape:?})"
                );
            }
        }
    }
}
