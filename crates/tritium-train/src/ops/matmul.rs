//! Ternary matmul forward (via the bit-exact reference) + backward.
//!
//! Forward:  Y[m,n] = s[n] · Σ_k A[m,k]·T[n,k]   (A:[M,K], T:[N,K] in {-1,0,1}, s:[N]).
//! Backward: gA[m,k] = Σ_n gY[m,n]·s[n]·T[n,k]
//!           gs[n]   = Σ_m gY[m,n]·(Σ_k A[m,k]·T[n,k])      (= gY·P, P unscaled contraction)
//!           gT[n,k] = s[n]·Σ_m gY[m,n]·A[m,k]              (STE'd to Wf upstream)

use tritium_core::{GemmShape, Trit, reference_mpgemm};

fn to_trits(t: &[f32]) -> Vec<Trit> {
    t.iter().map(|&x| Trit::from_sign(x as i8)).collect()
}

/// Forward via `tritium_core::reference_mpgemm` (bit-exact; deterministic k-order).
#[must_use]
pub fn forward(
    act: &[f32],
    trits: &[f32],
    scale: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    let w = to_trits(trits);
    reference_mpgemm(act, &w, scale, GemmShape::new(m, n, k), &mut out)
        .expect("shapes constructed consistently");
    out
}

/// vjp returning `[gA, gT, gs]` (same shapes as `act`, `trits`, `scale`).
#[must_use]
pub fn vjp(
    act: &[f32],
    trits: &[f32],
    scale: &[f32],
    m: usize,
    n: usize,
    k: usize,
    grad_out: &[f32],
) -> Vec<Vec<f32>> {
    let mut g_a = vec![0.0f32; m * k];
    let mut g_t = vec![0.0f32; n * k];
    let mut g_s = vec![0.0f32; n];

    for mi in 0..m {
        for ni in 0..n {
            let gy = grad_out[mi * n + ni];
            let s = scale[ni];
            // unscaled contraction P[m,n] = Σ_k A[m,k]·T[n,k] for gs.
            let mut p = 0.0f32;
            for ki in 0..k {
                let a = act[mi * k + ki];
                let t = trits[ni * k + ki];
                p += a * t;
                g_a[mi * k + ki] += gy * s * t;
                g_t[ni * k + ki] += gy * s * a;
            }
            g_s[ni] += gy * p;
        }
    }
    vec![g_a, g_t, g_s]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_matches_reference_signs() {
        // 1 row, 1 out, k=3: w=[+1,-1,+1], a=[10,3,5] -> 12, scale 2 -> 24.
        let y = forward(&[10.0, 3.0, 5.0], &[1.0, -1.0, 1.0], &[2.0], 1, 1, 3);
        assert_eq!(y, vec![24.0]);
    }
}
