//! Stochastic rounding to bfloat16 — for holding optimizer **master** weights in half the VRAM
//! without stalling training (Lever 5, the storage half of the 8-bit-optimizer work).
//!
//! bf16 keeps f32's 8 exponent bits but only 7 mantissa bits, so it *is* the high 16 bits of an f32.
//! An AdamW weight update smaller than the master's bf16 ULP rounds away entirely under
//! round-to-nearest, and training stalls once the learning rate shrinks below that floor. **Stochastic
//! rounding** (Gupta et al. 2015) rounds up with probability equal to the discarded fraction, so a
//! sub-ULP update survives *in expectation* across steps: `E[to_f32(sr(x))] = x` within the bf16
//! exponent range. This module is the CPU reference the device bf16-master mirror rounds to bit-for-bit
//! (same dither → same code).

/// Reinterpret a bf16 bit pattern (in the low 16 bits) as `f32`. bf16 is exactly the high 16 bits of
/// an IEEE-754 f32, so widening is a left shift — lossless and `libm`-free.
#[must_use]
pub fn to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Round-to-nearest-even `f32` → bf16 — the deterministic baseline (matches `torch`'s `bfloat16` cast
/// and `half::bf16::from_f32`). Non-finite inputs keep their sign/exponent (and a NaN stays a NaN).
#[must_use]
pub fn from_f32_nearest(x: f32) -> u16 {
    let bits = x.to_bits();
    if !x.is_finite() {
        // Preserve inf; for NaN force a mantissa bit so the truncated value stays NaN.
        let hi = (bits >> 16) as u16;
        return if x.is_nan() { hi | 0x0040 } else { hi };
    }
    // Add the round-to-nearest-even bias to the discarded low 16 bits, then truncate. The `+lsb`
    // term breaks ties toward even (the standard bf16 RNE trick).
    let lsb = (bits >> 16) & 1;
    let bias = 0x7fff + lsb;
    (bits.wrapping_add(bias) >> 16) as u16
}

/// Stochastic round `f32` → bf16 with a 16-bit uniform dither `rand16 ∈ [0, 2¹⁶)`. The low 16 bits of
/// the f32 are the fraction being discarded; adding the dither and truncating rounds up with exactly
/// that fraction's probability, so the rounding is unbiased. Non-finite inputs fall back to
/// [`from_f32_nearest`] (dithering an inf/NaN is meaningless).
#[must_use]
pub fn from_f32_stochastic(x: f32, rand16: u16) -> u16 {
    if !x.is_finite() {
        return from_f32_nearest(x);
    }
    // The carry out of the low 16 bits IS the round-up; truncation then keeps the high 16 bits.
    (x.to_bits().wrapping_add(u32::from(rand16)) >> 16) as u16
}

/// Deterministic 16-bit dither from a seed and flat index (xorshift64 idiom, forced non-zero) — the
/// same generator family as the FSQ stochastic STE, so the device mirror can reproduce the exact
/// dither stream from `(seed, index)`.
#[must_use]
pub fn dither16(seed: u64, idx: usize) -> u16 {
    let mut s = (seed ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    (s >> 32) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// bf16-representable values round-trip exactly; a generic f32 round-trips within one bf16 ULP.
    #[test]
    fn roundtrip_is_within_one_ulp() {
        for &x in &[0.0f32, 1.0, -1.0, 0.5, 2.0, 1.5, -256.0] {
            assert_eq!(to_f32(from_f32_nearest(x)), x, "bf16-exact value {x}");
        }
        let x = 1.001_953_1_f32; // not bf16-representable
        let ulp = 2f32.powi(-7); // bf16 has 7 mantissa bits ⇒ ULP near 1.0
        assert!(
            (to_f32(from_f32_nearest(x)) - x).abs() <= ulp,
            "nearest within a ULP"
        );
    }

    /// Stochastic rounding is unbiased: averaging many dithered roundings of an off-grid value
    /// converges to the value itself (not to either bf16 neighbour). This is the property that lets a
    /// sub-ULP update survive in expectation.
    #[test]
    fn stochastic_rounding_is_unbiased() {
        // x sits a quarter of a bf16 ULP above 1.0 (low 16 bits = 0x4000 ⇒ fraction 0.25).
        let x = f32::from_bits(0x3f80_4000);
        let down = to_f32(from_f32_nearest(1.0)); // 1.0
        let up = to_f32(0x3f80u16 + 1); // next bf16 above 1.0
        assert!(
            down < x && x < up,
            "x must be strictly between bf16 neighbours"
        );
        let n = 200_000usize;
        let mut acc = 0.0f64;
        let mut ups = 0usize;
        for i in 0..n {
            let bits = from_f32_stochastic(x, dither16(0xB16F, i));
            let r = to_f32(bits);
            assert!(r == down || r == up, "SR must land on a neighbour: {r}");
            acc += f64::from(r);
            ups += usize::from(r == up);
        }
        let mean = (acc / n as f64) as f32;
        assert!((mean - x).abs() < 1e-4, "SR mean {mean} must track x {x}");
        // Fraction 0.25 above 1.0 ⇒ ~25% of draws round up.
        let up_frac = ups as f32 / n as f32;
        assert!(
            (up_frac - 0.25).abs() < 0.02,
            "round-up fraction {up_frac} ≈ 0.25"
        );
    }

    /// The load-bearing claim: with a bf16 master, sub-ULP updates stall under nearest rounding but
    /// descend under stochastic rounding. Simulate one weight far from a target with a learning rate
    /// whose per-step step is below the bf16 ULP at that magnitude.
    #[test]
    fn stochastic_master_descends_where_nearest_stalls() {
        let target = 100.0f32; // ULP of bf16 near 100 is 2^-7·64 = 0.5 — a coarse grid
        let step = 0.1f32; // each update is < 0.5 ⇒ nearest rounding erases it
        let run = |stochastic: bool| {
            let mut w = 90.0f32;
            let mut bits = from_f32_nearest(w);
            for i in 0..2_000usize {
                w = to_f32(bits);
                if (w - target).abs() < 1e-3 {
                    break;
                }
                let updated = w + step * (target - w).signum();
                bits = if stochastic {
                    from_f32_stochastic(updated, dither16(0x5EED, i))
                } else {
                    from_f32_nearest(updated)
                };
            }
            to_f32(bits)
        };
        let nearest = run(false);
        let stochastic = run(true);
        assert!(
            (nearest - 90.0).abs() < 0.5,
            "nearest rounding must stall near the start: {nearest}"
        );
        assert!(
            (stochastic - target).abs() < 2.0,
            "stochastic rounding must climb to the target: {stochastic}"
        );
    }
}
