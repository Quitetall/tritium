//! Residual ternary plane expansion — the heart of SALT (ADR 0001 §1).
//!
//! A weight group `W` is approximated by a sum of `T` ternary planes,
//! `W ≈ Σ_p s_p · t_p` with every `t_p ∈ {-1, 0, +1}`, fit **greedily**: plane
//! `p` AbsMean-quantizes the residual left by planes `1..p`. The expansion is
//! prefix-stable — the first `T` planes never change when you ask for more — so
//! `residual_expand(w, 1)` is *exactly* flat BitNet b1.58 AbsMean, the `T = 1`
//! special case (ADR 0006 regression gate).

use tritium_core::{Trit, absmean};

/// One ternary plane over a weight group: a single non-negative `scale` and the
/// per-weight trits. Dequantizes element-wise to `scale · trit`.
#[derive(Clone, Debug, PartialEq)]
pub struct Plane {
    /// Per-plane AbsMean scale (`mean(|residual|)`); always `≥ 0`.
    pub scale: f32,
    /// Ternary codes, one per weight in the group.
    pub trits: Vec<Trit>,
}

/// A stack of residual planes. `planes.len()` is the *realized* plane count `T`
/// (`0..=T_max`); an empty stack reconstructs to all-zero (a fully pruned tile).
///
/// The contract is `plane_count() ≤ t_requested`, **not** equality:
/// [`residual_expand`] stops early once the residual collapses to zero, so
/// consumers (the allocator, the format sidecar) must read the realized count
/// rather than assume the requested one.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct PlaneStack {
    /// Planes in fit order: `planes[0]` is the dense base, the rest are residual.
    pub planes: Vec<Plane>,
}

impl PlaneStack {
    /// Realized plane count `T`.
    #[inline]
    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }

    /// Dequantize the whole stack: `Σ_p scale_p · trit_p`, element-wise, summed
    /// in plane order (base first) to a freshly zeroed `group_len` buffer.
    pub fn reconstruct(&self, group_len: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; group_len];
        for plane in &self.planes {
            debug_assert_eq!(plane.trits.len(), group_len, "plane width ≠ group_len");
            for (o, code) in out.iter_mut().zip(&plane.trits) {
                *o += plane.scale * code.to_f32();
            }
        }
        out
    }
}

/// Flat BitNet b1.58 AbsMean quantization of one weight group:
/// `scale = mean(|w|)`, `trit = round(w / scale)` clamped to `{-1, 0, +1}`.
///
/// This is the `T = 1` SALT special case and the BitNet regression oracle: the
/// scale is the shared [`tritium_core::absmean`] contract, and the rounding
/// threshold (`|w| < scale/2 → 0`) is BitNet's. A degenerate (all-zero) group
/// has `scale = 0` and quantizes to all-[`Trit::ZERO`].
pub fn absmean_ternary(w: &[f32]) -> Plane {
    let scale = absmean(w);
    let trits = w.iter().map(|&x| quantize_one(x, scale)).collect();
    Plane { scale, trits }
}

/// Quantize `w` to a single ternary plane at a **given** `scale` instead of the group's
/// own AbsMean: `trit = round(w / scale)` clamped to `{-1, 0, +1}`, `Plane::scale = scale`.
///
/// The per-tensor base plane ([`BaseScaleScope::Tensor`](crate::BaseScaleScope)) uses this so every
/// 256-block of the base shares ONE scale — the deployed BitNet b1.58 per-tensor AbsMean,
/// which a per-block fit (each block its own AbsMean) does not reproduce. `scale == 0`
/// prunes to all-[`Trit::ZERO`], matching [`absmean_ternary`].
pub fn ternary_at_scale(w: &[f32], scale: f32) -> Plane {
    let trits = w.iter().map(|&x| quantize_one(x, scale)).collect();
    Plane { scale, trits }
}

/// `round(x / scale)` clamped to `{-1, 0, +1}`. `scale == 0` (degenerate group)
/// prunes to [`Trit::ZERO`]. The clamp guarantees the result is in range, so the
/// [`Trit`] construction cannot fail.
#[inline]
fn quantize_one(x: f32, scale: f32) -> Trit {
    if scale == 0.0 {
        return Trit::ZERO;
    }
    let q = (x / scale).round().clamp(-1.0, 1.0) as i8;
    Trit::from_i8(q).expect("clamp keeps q in {-1, 0, 1}")
}

/// Greedily expand `w` into at most `t` residual ternary planes (ADR 0001 §1).
///
/// Each plane AbsMean-quantizes the residual left by the previous planes, then
/// subtracts its contribution. Stops early when the residual collapses to exactly
/// zero (`scale == 0`) — no information left to fit, so trailing all-zero planes
/// are never emitted. `t = 0` yields an empty stack; `t = 1` yields exactly a
/// single [`absmean_ternary`] plane.
pub fn residual_expand(w: &[f32], t: usize) -> PlaneStack {
    let mut residual = w.to_vec();
    let mut planes = Vec::with_capacity(t);
    for _ in 0..t {
        let plane = absmean_ternary(&residual);
        if plane.scale == 0.0 {
            break;
        }
        for (r, code) in residual.iter_mut().zip(&plane.trits) {
            *r -= plane.scale * code.to_f32();
        }
        planes.push(plane);
    }
    PlaneStack { planes }
}

/// Sum-of-squared reconstruction error between `w` and its plane-stack
/// approximation, accumulated in `f64`. Non-increasing as planes are added —
/// the monotonicity the allocator's marginal-gain table relies on (ADR 0006).
pub fn recon_error(w: &[f32], stack: &PlaneStack) -> f64 {
    let approx = stack.reconstruct(w.len());
    w.iter()
        .zip(&approx)
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ── Gate (ADR 0006): `T = 1` reduces *exactly* to flat AbsMean. ──────────
    // The SALT path at one plane must be bit-identical to the BitNet b1.58
    // oracle — same scale (the shared `absmean` contract) and same trits.
    #[test]
    fn t1_equals_flat_absmean_golden() {
        let w = [0.0, 0.3, -0.9, 1.7, -2.4, 0.05, 0.8, -0.51];
        let flat = absmean_ternary(&w);
        let salt = residual_expand(&w, 1);
        assert_eq!(salt.planes.len(), 1, "one non-degenerate plane");
        assert_eq!(salt.planes[0], flat, "T=1 plane must equal flat AbsMean");
        // And the scale is the shared contract, bit-for-bit.
        assert_eq!(flat.scale.to_bits(), absmean(&w).to_bits());
    }

    // A degenerate (all-zero) group: flat AbsMean is all-ZERO at scale 0, SALT
    // short-circuits to an empty stack — but both reconstruct to the same zeros.
    #[test]
    fn t1_zero_group_reconstruction_matches() {
        let w = [0.0f32; 16];
        let flat = absmean_ternary(&w);
        let salt = residual_expand(&w, 1);
        assert_eq!(flat.scale, 0.0);
        assert!(flat.trits.iter().all(|t| t.is_zero()));
        assert!(salt.planes.is_empty(), "no information → no planes");
        let from_flat = PlaneStack { planes: vec![flat] }.reconstruct(w.len());
        let from_salt = salt.reconstruct(w.len());
        assert_eq!(from_flat, from_salt);
        assert!(from_salt.iter().all(|&x| x == 0.0));
    }

    // A single weight is fit exactly by one plane (scale = |w|, trit = sign).
    #[test]
    fn single_weight_exact_at_t1() {
        for &x in &[3.5f32, -0.001, 42.0, -7.25] {
            let stack = residual_expand(&[x], 1);
            let recon = stack.reconstruct(1);
            assert_eq!(recon[0].to_bits(), x.to_bits(), "x={x} not reproduced");
        }
    }

    proptest! {
        // ── Gate (ADR 0006): reconstruction error is monotonic in T. ─────────
        // err(T+1) ≤ err(T) for every T — the property the rate-distortion
        // allocator's marginal-gain table assumes (Δerr ≥ 0).
        #[test]
        fn recon_error_monotonic_in_t(
            w in prop::collection::vec(-10.0f32..10.0, 1..192)
        ) {
            let mut prev = f64::INFINITY;
            for t in 0..=5usize {
                let err = recon_error(&w, &residual_expand(&w, t));
                prop_assert!(
                    err <= prev + 1e-9,
                    "err grew at T={t}: {err} > {prev}"
                );
                prev = err;
            }
        }

        // Prefix stability: the planes of expand(w, T) are a prefix of
        // expand(w, T+1) — greedy never revisits an earlier plane.
        #[test]
        fn expansion_is_prefix_stable(
            w in prop::collection::vec(-10.0f32..10.0, 1..128),
            t in 0..5usize,
        ) {
            let short = residual_expand(&w, t);
            let long = residual_expand(&w, t + 1);
            prop_assert!(long.planes.len() >= short.planes.len());
            for (a, b) in short.planes.iter().zip(&long.planes) {
                prop_assert_eq!(a, b);
            }
        }

        // ── Gate (ADR 0006): determinism — same input ⇒ byte-identical out. ──
        #[test]
        fn expansion_is_deterministic(
            w in prop::collection::vec(-10.0f32..10.0, 1..128),
            t in 0..4usize,
        ) {
            let a = residual_expand(&w, t);
            let b = residual_expand(&w, t);
            prop_assert_eq!(a.planes.len(), b.planes.len());
            for (pa, pb) in a.planes.iter().zip(&b.planes) {
                prop_assert_eq!(pa.scale.to_bits(), pb.scale.to_bits());
                prop_assert_eq!(&pa.trits, &pb.trits);
            }
        }

        // T=1 equals flat AbsMean for arbitrary non-degenerate groups (the gate,
        // generalized): identical reconstruction, bit-for-bit.
        #[test]
        fn t1_matches_flat_absmean_recon(
            w in prop::collection::vec(-10.0f32..10.0, 1..128)
        ) {
            let flat = PlaneStack { planes: vec![absmean_ternary(&w)] };
            let salt = residual_expand(&w, 1);
            let rf = flat.reconstruct(w.len());
            let rs = salt.reconstruct(w.len());
            for (a, b) in rf.iter().zip(&rs) {
                prop_assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
}
