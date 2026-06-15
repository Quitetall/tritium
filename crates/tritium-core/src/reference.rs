//! The correctness ground truth.

use crate::{GemmShape, Trit, error::TritError};

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
