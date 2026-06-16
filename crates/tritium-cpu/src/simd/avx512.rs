//! AVX-512 ternary mpGEMM (v0.30, WF-C).
//!
//! A 512-bit-wide sibling of the AVX2 kernel in [`crate::kernel`]: it processes
//! the `K` dimension **sixteen** `f32` lanes at a time (vs. AVX2's eight),
//! decoding the trits and forming the signed add/sub/skip contribution with
//! `__mmask16` mask registers, then folding that contribution **sequentially in
//! `f32` in k-order** so the result is bit-identical to the scalar reference —
//! exactly the discipline the AVX2 kernel follows.
//!
//! ## Why the fold is still scalar-in-k-order
//!
//! The reference walks `k` in a single `f32` accumulator. Over `K` up to 512 its
//! own rounding (partial sums reach ~150 before cancelling to ~0.2) is itself
//! ~1e-4 — right at the conformance floor. Any SIMD reduction that *reorders* or
//! *re-widens* those adds produces a different ~1e-4 rounding and drifts past the
//! tolerance. So this kernel uses AVX-512 only to **decode** the trits and
//! **form** the per-element signed contribution (`+a` / `-a` / `0`); the fold of
//! those contributions is sequential `f32` in k-order. `acc + 0.0 == acc` (skip)
//! and `acc + (-a) == acc - a` (sub) are exact in IEEE round-to-nearest, so the
//! signed-contribution fold reproduces the reference's branchy add/sub/skip
//! bit-for-bit — clearing the cross-ISA parity gate against AVX2 and scalar.
//!
//! ## Availability
//!
//! Compiled on every `x86_64` target (the intrinsics are stable in
//! `core::arch::x86_64`), but only *executed* when the host advertises `avx512f`
//! at runtime — [`crate::kernel::dispatch_mpgemm`] gates the call behind
//! `is_x86_feature_detected!("avx512f")`. On a host without AVX-512 (e.g. this
//! sm_89 / AVX2 build box) the branch is never taken and the AVX2 path runs; the
//! kernel still compiles, so the cross-ISA conformance lane can exercise it on an
//! AVX-512 host with no source change.
//!
//! VNNI (`vpdpbusd`) / AMX-int8 acceleration of the int8-activation path is a
//! further v0.30 step layered on the same decode; this kernel implements the
//! `f32`-activation contraction that the parity gate covers.

use core::arch::x86_64::{
    _mm512_loadu_ps, _mm512_mask_sub_ps, _mm512_maskz_mov_ps, _mm512_setzero_ps, _mm512_storeu_ps,
    _mm_cmpgt_epi8_mask, _mm_loadu_si128, _mm_set1_epi8,
};

use tritium_core::{GemmShape, Trit};
use tritium_spec::BackendError;

use crate::kernel::scalar_mpgemm;

/// AVX-512 ternary mpGEMM. Same contraction as the scalar reference, vectorised
/// over the `K` dimension sixteen `f32` lanes at a time.
///
/// For each output it decodes sixteen packed `i8` trits per step, builds two
/// `__mmask16`s (`trit > 0` and `trit < 0`), and forms the signed contribution
/// `+a` / `-a` / `0` with a zero-masked move plus a masked subtract — no multiply
/// by the trit — then folds that contribution **sequentially in `f32` in
/// k-order**. The `K` tail that does not fill a 16-lane register is folded with
/// the same scalar add/sub/skip, in k-order. The result is bit-identical to
/// [`crate::kernel::scalar_mpgemm`] (see the module note), not merely within the
/// `1e-4` tolerance.
///
/// # Safety
/// The caller must ensure the `avx512f`, `avx512bw` and `avx512vl` target
/// features are available on the host (checked via
/// `is_x86_feature_detected!`). The byte-lane trit compares
/// (`_mm_cmpgt_epi8_mask`) need `avx512bw` for the per-byte compare and
/// `avx512vl` to emit a mask from a 128-bit operand; the 512-bit f32 selects need
/// `avx512f`. Calling this on a CPU without all three is undefined behaviour.
///
/// # Errors
/// [`BackendError::Core`] if a buffer length disagrees with `shape`; in that case
/// no intrinsic runs (the call defers to the scalar reference for its typed
/// error before touching SIMD).
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn avx512_mpgemm(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    let GemmShape { m, n, k } = shape;
    // Validate up front so no intrinsic ever reads out of bounds. This mirrors
    // the AVX2 kernel; on mismatch we return before touching SIMD.
    if act.len() != m * k || weights.len() != n * k || scales.len() != n || out.len() != m * n {
        // Defer to the scalar reference purely for its typed `ShapeMismatch`.
        return scalar_mpgemm(act, weights, scales, shape, out);
    }

    // Sixteen f32 lanes per AVX-512 register.
    const LANES: usize = 16;
    let k_simd = k - (k % LANES);

    // `Trit` is `#[repr(transparent)]` over `i8`, so a `&[Trit]` is bit-identical
    // to a `&[i8]` whose elements are all in `{-1, 0, 1}`. Reinterpret for the
    // byte loads the integer intrinsics need.
    // SAFETY: `Trit` is `#[repr(transparent)]` over `i8` with the same size and
    // alignment; the slice length is unchanged, so the new slice covers exactly
    // the same valid, initialised, immutably-borrowed bytes, every one in
    // `{-1,0,1}` ⊂ valid `i8`.
    let weights_i8: &[i8] =
        unsafe { core::slice::from_raw_parts(weights.as_ptr().cast::<i8>(), weights.len()) };

    // Scratch for the signed per-element contribution of one row's SIMD body.
    // Filled by the vector pass, then folded sequentially in k-order — exactly
    // as the AVX2 kernel does, so the two ISAs share one accumulation order.
    let mut signed_buf = vec![0.0f32; k_simd];

    // Inside a `#[target_feature(enable = …)]` function the AVX-512 intrinsics are
    // callable without an `unsafe` block; `unsafe` is confined to the intrinsics
    // that dereference a raw pointer (the loads/stores). `unsafe_op_in_unsafe_fn`
    // is `deny`, so each such intrinsic carries its own `unsafe` + `// SAFETY:`.
    for mi in 0..m {
        let arow = &act[mi * k..mi * k + k];
        for ni in 0..n {
            let wrow = &weights_i8[ni * k..ni * k + k];

            // Vectorised pass: fill `signed_buf[ki]` with the add/sub/skip value.
            let mut ki = 0;
            while ki < k_simd {
                // Sixteen activations.
                // SAFETY: `ki + LANES <= k_simd <= k = arow.len()`, so the sixteen
                // f32 starting at `arow.as_ptr().add(ki)` are in bounds and
                // initialised. `loadu` permits any alignment.
                let a = unsafe { _mm512_loadu_ps(arow.as_ptr().add(ki)) };

                // Sixteen packed i8 trits.
                // SAFETY: `ki + LANES <= k = wrow.len()`, so the sixteen bytes at
                // `wrow.as_ptr().add(ki)` are in bounds and initialised.
                // `_mm_loadu_si128` reads exactly those 16 bytes with no alignment
                // requirement; the pointer cast is to the 128-bit integer vector
                // type the intrinsic expects.
                let t = unsafe {
                    _mm_loadu_si128(wrow.as_ptr().add(ki).cast::<core::arch::x86_64::__m128i>())
                };

                // Mask lanes by sign. Trits are in `{-1,0,1}`, so:
                //   pos = (trit == 1)  ≡ (trit > 0)
                //   neg = (trit == -1) ≡ (trit < 0) ≡ (0 > trit)
                // `_mm_cmpgt_epi8_mask(a, b)` yields `a > b` per lane as a mask.
                let zero_i8 = _mm_set1_epi8(0);
                let pos_mask = _mm_cmpgt_epi8_mask(t, zero_i8);
                let neg_mask = _mm_cmpgt_epi8_mask(zero_i8, t);
                // `pos_mask` and `neg_mask` are mutually exclusive; the remaining
                // (zero-trit) lanes stay 0 via the zero-masked move below — that is
                // the skip, no separate `== 0` mask needed.

                // Signed contribution per lane, no multiply: start `+a` on the +1
                // lanes (0 elsewhere), then overwrite the -1 lanes with `-a` via a
                // masked subtract from zero. Two steps, mutually-exclusive masks.
                //
                // Step 1: `plus = pos ? a : 0`  (zero-masked move keeps `+a` only
                // on the +1 lanes; 0 on every other lane, the skip).
                let plus = _mm512_maskz_mov_ps(pos_mask, a);
                // Step 2: on the -1 lanes, replace with `0 - a = -a`. The masked
                // subtract writes `0 - a` where `neg_mask` is set and leaves
                // `plus` untouched elsewhere — so +1 lanes keep `+a`, 0 lanes keep
                // `0`, -1 lanes become `-a`.
                let signed = _mm512_mask_sub_ps(plus, neg_mask, _mm512_setzero_ps(), a);

                // SAFETY: `ki + LANES <= k_simd = signed_buf.len()`, so the
                // 16-wide store is in bounds; `storeu` has no alignment need.
                unsafe { _mm512_storeu_ps(signed_buf.as_mut_ptr().add(ki), signed) };

                ki += LANES;
            }

            // Sequential f32 fold over the signed contributions, in k-order —
            // bit-identical to the reference's single accumulator.
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

    /// AVX-512 vs scalar parity — only runs where the CPU actually has AVX-512.
    /// On the AVX2 build box this is a documented no-op (the kernel is compiled
    /// but never executed); on an AVX-512 host it asserts bit-for-bit parity,
    /// the cross-ISA gate.
    #[test]
    fn avx512_matches_scalar_when_available() {
        if !is_x86_feature_detected!("avx512f")
            || !is_x86_feature_detected!("avx512bw")
            || !is_x86_feature_detected!("avx512vl")
        {
            eprintln!(
                "avx512f/bw/vl not detected on this host — skipping AVX-512 kernel unit test"
            );
            return;
        }
        let mut s = 0x0BAD_F00Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // Ragged tail (K=19 → 1×16-lane vector + 3 scalar tail) and block-aligned
        // K=512 where the reference's f32
        // cancellation is worst; bit-for-bit parity is required, not just 1e-4.
        for &(m, n, k) in &[(3usize, 5usize, 19usize), (2, 4, 512), (1, 1, 16)] {
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
            // SAFETY: avx512f+bw were just confirmed available by the guard above.
            unsafe { avx512_mpgemm(&act, &w, &scales, shape, &mut got).unwrap() };

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
