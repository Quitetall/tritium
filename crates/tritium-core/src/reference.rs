//! The correctness ground truth.

use crate::{ConvShape, GemmShape, Trit, error::TritError};

/// Reference mixed-precision GEMM: ternary weights × `f32` activations, with a
/// per-output-channel scale.
///
/// ```text
/// out[m, n] = scale[n] · Σ_k  act[m, k] · w[n, k]
/// ```
///
/// Layout — all row-major:
/// - `act`     — `[M, K]` activations.
/// - `weights` — `[N, K]` ternary weights (output-major).
/// - `scale_n` — `[N]` per-output-channel scales.
/// - `out`     — `[M, N]`, overwritten.
///
/// This is the slow, obviously-correct implementation every backend kernel is
/// measured against. It deliberately expresses ternary matmul in its essential
/// **add / subtract / skip** form (no multiply) — that is the contract a kernel
/// optimizes, here written for clarity rather than speed.
///
/// # Errors
/// [`TritError::ShapeMismatch`] if any buffer length disagrees with `shape`.
pub fn reference_mpgemm(
    act: &[f32],
    weights: &[Trit],
    scale_n: &[f32],
    shape: GemmShape,
    out: &mut [f32],
) -> Result<(), TritError> {
    check(act.len(), shape.m * shape.k)?;
    check(weights.len(), shape.n * shape.k)?;
    check(scale_n.len(), shape.n)?;
    check(out.len(), shape.m * shape.n)?;

    let GemmShape { m, n, k } = shape;
    for mi in 0..m {
        let arow = &act[mi * k..mi * k + k];
        for ni in 0..n {
            let wrow = &weights[ni * k..ni * k + k];
            let mut acc = 0.0f32;
            for ki in 0..k {
                // Ternary: accumulate by sign, skip zeros. No multiply.
                match wrow[ki].get() {
                    1 => acc += arow[ki],
                    -1 => acc -= arow[ki],
                    _ => {}
                }
            }
            out[mi * n + ni] = acc * scale_n[ni];
        }
    }
    Ok(())
}

#[inline]
fn check(got: usize, expected: usize) -> Result<(), TritError> {
    if got == expected {
        Ok(())
    } else {
        Err(TritError::ShapeMismatch { expected, got })
    }
}

/// Reference ternary 1-D convolution — the multiply-free (add / subtract / skip) oracle every backend
/// (CPU / CUDA / MCU) is measured against, and the numeric truth the codec conformance vectors freeze.
///
/// ```text
/// out[b, co, l] = scale[co] · Σ_{ci',kk}  x[b, g·(C_in/groups)+ci', l·stride + kk·dilation − pad_left] · w[co, ci'·K + kk]
/// ```
/// with out-of-range (padding) taps skipped and group `g = co / (C_out/groups)`. Accumulation runs in
/// the pinned order `ci' → kk` — the same order as the training im2col column index `ci'·K + kk` — so it
/// is bit-reproducible across every backend (the ADR 0018 reduction-order discipline), and bit-identical
/// to the training `conv1d` forward at ternary weights (float `×{−1,0,1}` equals add/sub/skip exactly).
///
/// Layout — all row-major: `x` `[B, C_in, L_in]`; `weights` `[C_out, (C_in/groups)·K]` ternary;
/// `scale_n` `[C_out]`; `out` `[B, C_out, L_out]`, overwritten.
///
/// # Errors
/// [`TritError::ShapeMismatch`] on any buffer-length or geometry disagreement (bad groups divisibility,
/// or a kernel wider than the padded input).
pub fn reference_conv1d(
    x: &[f32],
    weights: &[Trit],
    scale_n: &[f32],
    shape: ConvShape,
    out: &mut [f32],
) -> Result<(), TritError> {
    let l_out = shape.l_out();
    if !shape.buffers_fit(x.len(), weights.len(), scale_n.len(), out.len()) {
        return Err(TritError::ShapeMismatch {
            expected: shape.batch * shape.c_out * l_out,
            got: out.len(),
        });
    }
    let (c_in_pg, n_g, k_g, k) = (shape.c_in_pg(), shape.n_g(), shape.k_g(), shape.k);
    for b in 0..shape.batch {
        for g in 0..shape.groups {
            for n in 0..n_g {
                let co = g * n_g + n;
                let wrow = &weights[co * k_g..co * k_g + k_g];
                let s = scale_n[co];
                for l in 0..l_out {
                    let mut acc = 0.0f32;
                    for ci_local in 0..c_in_pg {
                        let ci = g * c_in_pg + ci_local;
                        let xbase = (b * shape.c_in + ci) * shape.l_in;
                        for kk in 0..k {
                            let p = l as isize * shape.stride as isize
                                + kk as isize * shape.dilation as isize
                                - shape.pad_left as isize;
                            if p >= 0 && (p as usize) < shape.l_in {
                                let xv = x[xbase + p as usize];
                                match wrow[ci_local * k + kk].get() {
                                    1 => acc += xv,
                                    -1 => acc -= xv,
                                    _ => {}
                                }
                            }
                        }
                    }
                    out[(b * shape.c_out + co) * l_out + l] = acc * s;
                }
            }
        }
    }
    Ok(())
}

/// Reference FSQ (finite scalar quantization) — the byte-exact **clamp** deploy grid, the oracle for the
/// codec latent quantizer. Per element (channel `c = i / len`, `L = levels[c]`):
///
/// ```text
/// b = clamp(x, -1, 1);  code = round_half_away((b+1)/2·(L−1)) ∈ [0, L−1];  q = code/(L−1)·2 − 1
/// ```
///
/// The round is an explicit non-negative truncation (`(v + 0.5) as i32`), never `libm`'s `rintf`
/// (half-to-even) — so the grid is bit-identical across CPU / CUDA / MCU (and matches the training
/// `fsq` clamp path). The `tanh` bound is training-only and deliberately not part of this deploy oracle.
///
/// # Errors
/// [`TritError::ShapeMismatch`] if `x.len() != channels·len`, `levels.len() != channels`,
/// `out.len() != x.len()`, or any level `L < 2`.
pub fn reference_fsq(
    x: &[f32],
    levels: &[u32],
    channels: usize,
    len: usize,
    out: &mut [f32],
) -> Result<(), TritError> {
    check(x.len(), channels * len)?;
    check(levels.len(), channels)?;
    check(out.len(), channels * len)?;
    for (i, (&xi, o)) in x.iter().zip(out.iter_mut()).enumerate() {
        let l = levels[i / len];
        if l < 2 {
            return Err(TritError::ShapeMismatch {
                expected: 2,
                got: l as usize,
            });
        }
        let lm1 = (l - 1) as f32;
        let b = xi.clamp(-1.0, 1.0);
        // (b+1)/2·(L−1) ≥ 0, so truncating (v + 0.5) is round-half-away-from-zero without libm.
        let code = (((b + 1.0) * 0.5 * lm1 + 0.5) as i32).clamp(0, (l - 1) as i32);
        *o = code as f32 / lm1 * 2.0 - 1.0;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Trit;

    fn trits(vals: &[i8]) -> alloc::vec::Vec<Trit> {
        vals.iter().map(|&v| Trit::from_i8(v).unwrap()).collect()
    }

    #[test]
    fn zero_weights_give_zero_output() {
        // 2 rows, 1 output, k=2: act is [M,K]=4, weights are [N,K]=2.
        let act = [1.0, 2.0, 3.0, 4.0];
        let w = trits(&[0, 0]);
        let mut out = [9.0; 2];
        reference_mpgemm(&act, &w, &[1.0], GemmShape::new(2, 1, 2), &mut out).unwrap();
        assert_eq!(out, [0.0, 0.0]);
    }

    #[test]
    fn signs_add_and_subtract() {
        // 1 row, 1 output, k=3. w = [+1, -1, +1], act = [10, 3, 5] -> 10 - 3 + 5 = 12.
        let act = [10.0, 3.0, 5.0];
        let w = trits(&[1, -1, 1]);
        let mut out = [0.0; 1];
        reference_mpgemm(&act, &w, &[2.0], GemmShape::new(1, 1, 3), &mut out).unwrap();
        assert_eq!(out[0], 24.0); // (12) * scale 2.0
    }

    #[test]
    fn shape_mismatch_is_caught() {
        let act = [1.0, 2.0];
        let w = trits(&[1, 1, 1]); // wrong: n*k = 3 but shape wants 1*2
        let mut out = [0.0; 1];
        let err =
            reference_mpgemm(&act, &w, &[1.0], GemmShape::new(1, 1, 2), &mut out).unwrap_err();
        assert!(matches!(err, TritError::ShapeMismatch { .. }));
    }
}

// Property: the multiply-free (add/sub/skip) reference must equal a plain f32
// matmul that treats each trit as a float coefficient. If these ever diverge,
// the "no multiply" decomposition is wrong.
#[cfg(test)]
mod prop {
    use super::*;
    use crate::Trit;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn add_sub_skip_equals_float_matmul(
            m in 1usize..4,
            n in 1usize..4,
            k in 1usize..8,
            seed in any::<u64>(),
        ) {
            // Deterministic pseudo-random fill from the seed (no rand dep).
            let mut s = seed | 1;
            let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };

            let act: alloc::vec::Vec<f32> =
                (0..m * k).map(|_| ((next() % 200) as f32 - 100.0) / 10.0).collect();
            let w: alloc::vec::Vec<Trit> =
                (0..n * k).map(|_| Trit::from_sign((next() % 3) as i8 - 1)).collect();
            let scale: alloc::vec::Vec<f32> =
                (0..n).map(|_| ((next() % 50) as f32 + 1.0) / 25.0).collect();

            let shape = GemmShape::new(m, n, k);
            let mut got = alloc::vec![0.0f32; m * n];
            reference_mpgemm(&act, &w, &scale, shape, &mut got).unwrap();

            // Independent float reference.
            for mi in 0..m {
                for ni in 0..n {
                    let mut acc = 0.0f32;
                    for ki in 0..k {
                        acc += act[mi * k + ki] * w[ni * k + ki].to_f32();
                    }
                    let want = acc * scale[ni];
                    prop_assert!((got[mi * n + ni] - want).abs() < 1e-4);
                }
            }
        }
    }
}
