//! ARM NEON ternary mpGEMM (v0.30, WF-C).
//!
//! An `aarch64` sibling of the AVX2 / AVX-512 kernels in [`crate::kernel`] /
//! [`super::avx512`]: it processes the `K` dimension **four** `f32` lanes at a
//! time (`float32x4_t`), decoding the trits and forming the signed add/sub/skip
//! contribution with NEON compare + select (`vcgtq` / `vbslq`), then folding that
//! contribution **sequentially in `f32` in k-order** so the result is
//! bit-identical to the scalar reference — exactly the discipline the x86 kernels
//! follow. This shared accumulation order is what makes AVX2 == AVX-512 == NEON
//! == scalar bit-for-bit, the cross-ISA parity gate.
//!
//! ## Why the fold is still scalar-in-k-order
//!
//! The reference walks `k` in a single `f32` accumulator. Over `K` up to 512 its
//! own rounding is ~1e-4 — right at the conformance floor — so any SIMD reduction
//! that reorders or re-widens those adds drifts past tolerance. NEON is used only
//! to **decode** the trits and **form** the per-element signed contribution
//! (`+a` / `-a` / `0`); the fold is sequential `f32` in k-order. `acc + 0.0 ==
//! acc` (skip) and `acc + (-a) == acc - a` (sub) are exact in IEEE
//! round-to-nearest, so the fold reproduces the reference's add/sub/skip
//! bit-for-bit.
//!
//! ## Availability
//!
//! `#[cfg(target_arch = "aarch64")]` — this module is compiled **only** on
//! aarch64 and is absent from the x86-64 build (the sm_89 / AVX2 box this lands
//! on), so it cannot affect that build. On aarch64 it is selected by
//! [`crate::kernel::dispatch_mpgemm`]; the baseline NEON (`Advanced SIMD`) used
//! here is mandatory on aarch64, so no feature detection is needed for this
//! `f32`-activation path. The SDOT/UDOT dot-product extension for the
//! int8-activation path is a further v0.30 step gated by
//! `std::arch::is_aarch64_feature_detected!("dotprod")`, layered on this same
//! decode.

use core::arch::aarch64::{
    float32x4_t, vbslq_f32, vcgtq_s32, vcltq_s32, vdupq_n_f32, vdupq_n_s32, vget_low_s16,
    vld1q_f32, vmovl_s8, vmovl_s16, vnegq_f32, vst1q_f32,
};

use tritium_core::{GemmShape, Trit};
use tritium_spec::BackendError;

use crate::kernel::scalar_mpgemm;

/// NEON ternary mpGEMM. Same contraction as the scalar reference, vectorised over
/// the `K` dimension four `f32` lanes at a time.
///
/// For each output it widens four packed `i8` trits to four `i32` lanes, builds
/// `pos` / `neg` masks by comparing against zero, and forms the signed
/// contribution `+a` / `-a` / `0` with two `vbslq_f32` selects — no multiply by
/// the trit — then folds that contribution **sequentially in `f32` in k-order**.
/// The `K` tail that does not fill a 4-lane register is folded with the same
/// scalar add/sub/skip, in k-order. The result is bit-identical to
/// [`crate::kernel::scalar_mpgemm`] (see the module note), not merely within the
/// `1e-4` tolerance — so the NEON output matches the x86 kernels exactly.
///
/// # Errors
/// [`BackendError::Core`] if a buffer length disagrees with `shape`; in that case
/// no intrinsic runs (the call defers to the scalar reference for its typed error
/// before touching SIMD).
///
/// # Safety
/// `#[target_feature(enable = "neon")]` — the function is `unsafe` to *call*
/// because Rust requires every `#[target_feature]` fn to be, but on aarch64
/// baseline NEON (`Advanced SIMD`) is architecturally mandatory, so the feature
/// is always present and the caller need only assert that (see the SAFETY note at
/// the call site in [`crate::kernel`]). This mirrors the AVX2 kernel's structure;
/// each raw-pointer load/store inside additionally carries its own `// SAFETY:`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neon_mpgemm(
    act: &[f32],
    weights: &[Trit],
    scales: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), BackendError> {
    let GemmShape { m, n, k } = shape;
    // Validate up front so no intrinsic ever reads out of bounds. This mirrors
    // the x86 kernels; on mismatch we return before touching SIMD.
    if act.len() != m * k || weights.len() != n * k || scales.len() != n || out.len() != m * n {
        // Defer to the scalar reference purely for its typed `ShapeMismatch`.
        return scalar_mpgemm(act, weights, scales, shape, out);
    }

    // Four f32 lanes per NEON register.
    const LANES: usize = 4;
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

    // Scratch for the signed per-element contribution of one row's SIMD body,
    // folded sequentially in k-order — the shared accumulation order.
    let mut signed_buf = vec![0.0f32; k_simd];

    for mi in 0..m {
        let arow = &act[mi * k..mi * k + k];
        for ni in 0..n {
            let wrow = &weights_i8[ni * k..ni * k + k];

            // Vectorised pass: fill `signed_buf[ki]` with the add/sub/skip value.
            let mut ki = 0;
            while ki < k_simd {
                // Four activations.
                // SAFETY: `ki + LANES <= k_simd <= k = arow.len()`, so the four
                // f32 at `arow.as_ptr().add(ki)` are in bounds and initialised;
                // `vld1q_f32` is an unaligned load.
                let a: float32x4_t = unsafe { vld1q_f32(arow.as_ptr().add(ki)) };

                // Widen four packed i8 trits to four i32 lanes. Load 8 bytes
                // (using only the low 4), widen i8→i16→i32, and keep the low four.
                // SAFETY: `ki + LANES <= k = wrow.len()`. We read four bytes via a
                // direct copy into an `[i8; 8]` scratch (the upper four are unused
                // padding), so no out-of-bounds read occurs even though the widen
                // intrinsics consume an 8-lane `int8x8_t`.
                let mut bytes = [0i8; 8];
                bytes[..LANES].copy_from_slice(&wrow[ki..ki + LANES]);
                // SAFETY: `bytes` is a fully-initialised `[i8; 8]`; `vld1_s8`
                // reads exactly those eight bytes with no alignment requirement.
                let t8 = unsafe { core::arch::aarch64::vld1_s8(bytes.as_ptr()) };
                let t16 = vmovl_s8(t8); // i8x8 → i16x8
                let t32 = vmovl_s16(vget_low_s16(t16)); // low i16x4 → i32x4

                // pos = (trit > 0), neg = (trit < 0); trits are in {-1,0,1}.
                let zero_i32 = vdupq_n_s32(0);
                let pos_mask = vcgtq_s32(t32, zero_i32); // u32x4 all-ones / zero
                let neg_mask = vcltq_s32(t32, zero_i32);

                // Signed contribution per lane, no multiply: select `+a` on the
                // +1 lanes (else 0), then overwrite the -1 lanes with `-a`. Two
                // bit-selects, mutually-exclusive masks; the 0-trit lanes keep 0
                // (the skip).
                let zero_f = vdupq_n_f32(0.0);
                let neg_a = vnegq_f32(a);
                // `vbslq_f32(mask, on_true, on_false)` selects per-bit.
                // `vcgtq_s32` / `vcltq_s32` already yield `uint32x4_t`, the mask
                // lane type `vbslq_f32(mask, on_true, on_false)` expects.
                let plus = vbslq_f32(pos_mask, a, zero_f);
                let signed = vbslq_f32(neg_mask, neg_a, plus);

                // SAFETY: `ki + LANES <= k_simd = signed_buf.len()`, so the 4-wide
                // store is in bounds; `vst1q_f32` has no alignment requirement.
                unsafe { vst1q_f32(signed_buf.as_mut_ptr().add(ki), signed) };

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

    /// NEON vs scalar parity — bit-for-bit, the cross-ISA gate. Runs on any
    /// aarch64 host (baseline NEON is mandatory there); this module is `cfg`'d
    /// out entirely on x86-64.
    #[test]
    fn neon_matches_scalar() {
        let mut s = 0x0FACE_u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // Ragged tail (K=7 → 1×4-lane vector + 3 scalar tail) and block-aligned
        // K=512 where the reference's f32
        // cancellation is worst; bit-for-bit parity is required.
        for &(m, n, k) in &[(3usize, 5usize, 7usize), (2, 4, 512), (1, 1, 4)] {
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
            // SAFETY: NEON (`Advanced SIMD`) is mandatory on every aarch64 target,
            // and this test only compiles under `#[cfg(target_arch = "aarch64")]`,
            // so the `neon` target feature is guaranteed present.
            unsafe { neon_mpgemm(&act, &w, &scales, shape, &mut got).unwrap() };

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
