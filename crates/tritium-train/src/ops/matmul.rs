//! Ternary matmul forward + backward for the autograd tape.
//!
//! Forward:  Y[m,n] = s[n] · Σ_k A[m,k]·W[n,k]   (A:[M,K], W:[N,K], s:[N]).
//! Backward: gA[m,k] = Σ_n gY[m,n]·s[n]·W[n,k]
//!           gs[n]   = Σ_m gY[m,n]·(Σ_k A[m,k]·W[n,k])   (= gY·P, P the unscaled contraction)
//!           gW[n,k] = s[n]·Σ_m gY[m,n]·A[m,k]           (straight-through-estimated to Wf upstream)
//!
//! `W` is a real `f32` weight matrix: ternary `{-1,0,1}` in the QAT value path, or the
//! continuous STE *surrogate* in the gradient-checked autograd path. The forward is a
//! plain float contraction so it stays differentiable in `W` — consistent with [`vjp`]
//! for *any* real `W`. At ternary `W` it matches `tritium_core::reference_mpgemm`, the
//! bit-exact inference kernel (anchored by `real_forward_matches_reference_at_ternary`);
//! we do not route the autograd forward through that kernel because its add/sub/skip form
//! sign-collapses non-ternary inputs (`Trit::from_sign`), which would make the weight path
//! piecewise-constant and silently zero the finite-difference gradient w.r.t. `Wf`.

/// Forward: `Y[m,n] = s[n] · Σ_k A[m,k]·W[n,k]` (real contraction over `W`).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn forward(
    act: &[f32],
    weights: &[f32],
    scale: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += act[mi * k + ki] * weights[ni * k + ki];
            }
            out[mi * n + ni] = scale[ni] * acc;
        }
    }
    out
}

/// vjp returning `[gA, gW, gs]` (same shapes as `act`, `weights`, `scale`).
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn vjp(
    act: &[f32],
    weights: &[f32],
    scale: &[f32],
    m: usize,
    n: usize,
    k: usize,
    grad_out: &[f32],
) -> Vec<Vec<f32>> {
    let mut g_a = vec![0.0f32; m * k];
    let mut g_w = vec![0.0f32; n * k];
    let mut g_s = vec![0.0f32; n];

    for mi in 0..m {
        for ni in 0..n {
            let gy = grad_out[mi * n + ni];
            let s = scale[ni];
            // unscaled contraction P[m,n] = Σ_k A[m,k]·W[n,k] for gs.
            let mut p = 0.0f32;
            for ki in 0..k {
                let a = act[mi * k + ki];
                let w = weights[ni * k + ki];
                p += a * w;
                g_a[mi * k + ki] += gy * s * w;
                g_w[ni * k + ki] += gy * s * a;
            }
            g_s[ni] += gy * p;
        }
    }
    vec![g_a, g_w, g_s]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_core::{GemmShape, Trit, reference_mpgemm};

    #[test]
    fn forward_matches_reference_signs() {
        // 1 row, 1 out, k=3: w=[+1,-1,+1], a=[10,3,5] -> 12, scale 2 -> 24.
        let y = forward(&[10.0, 3.0, 5.0], &[1.0, -1.0, 1.0], &[2.0], 1, 1, 3);
        assert_eq!(y, vec![24.0]);
    }

    #[test]
    fn real_forward_matches_reference_at_ternary() {
        // At ternary weights the float contraction equals the bit-exact inference kernel
        // (same tolerance the core's own add/sub/skip-vs-float property test uses).
        let (m, n, k) = (2, 3, 4);
        let act = [0.5f32, -1.2, 0.3, 2.0, -0.7, 0.9, 1.1, -0.4];
        let w = [
            1.0f32, -1.0, 0.0, 1.0, //
            0.0, 1.0, -1.0, -1.0, //
            1.0, 0.0, 0.0, -1.0,
        ];
        let scale = [0.7f32, 1.3, 0.5];
        let got = forward(&act, &w, &scale, m, n, k);
        let trits: Vec<Trit> = w.iter().map(|&x| Trit::from_sign(x as i8)).collect();
        let mut want = vec![0.0f32; m * n];
        reference_mpgemm(&act, &trits, &scale, GemmShape::new(m, n, k), &mut want).unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "got {g} want {w}");
        }
    }
}
