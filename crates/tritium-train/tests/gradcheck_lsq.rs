//! LSQ (learned step-size) backward validation (ADR 0030 Tier 1).
//!
//! The **weight** gradient is a standard round-clamp STE, so it is finite-difference-checked against the
//! surrogate `clamp(Wf/α)·α` (Gate C). The **α** gradient is the LSQ step-size *estimator* — it uses the
//! rounded value and is provably not the gradient of any smooth surrogate, so it cannot be
//! finite-differenced; we validate it by its closed form and by confirming it is a descent direction on
//! reconstruction MSE.

use tritium_train::gradcheck::{GradCheckCfg, check_op};
use tritium_train::ops::ste;

fn seeded(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            lo + (s % 1000) as f32 / 1000.0 * (hi - lo)
        })
        .collect()
}

const ROWS: usize = 3;
const COLS: usize = 5;

#[test]
fn lsq_weight_grad_matches_surrogate_finite_difference() {
    // Hand-placed so every |Wf/α| sits clearly in-band (<0.9) or clearly saturated (>1.1) — never
    // within h of the clamp kink at |v|=1 (α = 1 per row, so v = Wf).
    #[rustfmt::skip]
    let wf: Vec<f32> = vec![
         0.2, -0.4,  1.5, -1.8,  0.1,
        -0.3,  0.6, -1.4,  1.9, -0.2,
         0.5, -0.5,  1.6, -1.7,  0.3,
    ];
    let alpha = vec![1.0f32; ROWS];
    let inputs = vec![wf, alpha];
    check_op(
        |ins| ste::lsq_surrogate(ins[0], ins[1], ROWS, COLS),
        |ins, g| ste::lsq_vjp(ins[0], ins[1], ROWS, COLS, g),
        &inputs,
        &[0], // wrt Wf only; the α gradient is the LSQ estimator, checked below
        GradCheckCfg::default(),
    )
    .expect("LSQ weight vjp must equal the surrogate's central finite difference");
}

#[test]
fn lsq_alpha_grad_matches_closed_form() {
    let wf = seeded(7, ROWS * COLS, -2.5, 2.5);
    let alpha = seeded(8, ROWS, 0.5, 1.5);
    let grad = seeded(9, ROWS * COLS, -1.0, 1.0);
    let g_alpha = ste::lsq_vjp(&wf, &alpha, ROWS, COLS, &grad)[1].clone();
    // Recompute the LSQ step-size estimator independently.
    let scale = 1.0 / (COLS as f32).sqrt();
    for r in 0..ROWS {
        let a = alpha[r];
        let mut want = 0.0f32;
        for c in 0..COLS {
            let i = r * COLS + c;
            let v = wf[i] / a;
            want += grad[i]
                * if v.abs() < 1.0 {
                    v.round() - v
                } else {
                    v.signum()
                };
        }
        want *= scale;
        assert!(
            (g_alpha[r] - want).abs() < 1e-5,
            "row {r}: gAlpha {} vs closed form {want}",
            g_alpha[r]
        );
    }
}

#[test]
fn lsq_alpha_descent_reduces_reconstruction_mse() {
    // Starting α off the optimum (1.5× AbsMean), gradient descent on the LSQ α estimator must reduce
    // the reconstruction MSE ‖q(Wf,α) − Wf‖² — i.e. the estimator is a valid descent direction.
    let (rows, cols) = (4usize, 32usize);
    let wf = seeded(11, rows * cols, -1.5, 1.5);
    let base = ste::absmean_scale_per_row(&wf, rows, cols);
    let mut alpha: Vec<f32> = base.iter().map(|&a| a * 1.5).collect();

    let mse = |a: &[f32]| -> f32 {
        let q = ste::lsq_forward(&wf, a, rows, cols);
        q.iter().zip(&wf).map(|(&qi, &wi)| (qi - wi).powi(2)).sum()
    };
    let l0 = mse(&alpha);
    let lr = 0.02f32;
    for _ in 0..80 {
        let q = ste::lsq_forward(&wf, &alpha, rows, cols);
        let grad: Vec<f32> = q.iter().zip(&wf).map(|(&qi, &wi)| qi - wi).collect(); // dMSE/dq
        let g_alpha = ste::lsq_vjp(&wf, &alpha, rows, cols, &grad)[1].clone();
        for r in 0..rows {
            alpha[r] = (alpha[r] - lr * g_alpha[r]).max(1e-6);
        }
    }
    let l1 = mse(&alpha);
    assert!(
        l1 < l0,
        "LSQ α descent must reduce reconstruction MSE: {l0} -> {l1}"
    );
}
