//! Gate C (ADR 0007): STE + ternary-matmul backward vs central finite difference.

use tritium_train::gradcheck::{GradCheckCfg, check_op};
use tritium_train::ops::{dense, matmul, ste};

// Canonical fixture: M=3, N=4, K=5. Weights straddle the clamp band so the STE
// mask (pass-through vs saturated) is exercised in both states.
const M: usize = 3;
const N: usize = 4;
const K: usize = 5;

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

#[test]
fn ste_quantize_grad_matches_surrogate_finite_difference() {
    // The STE backward is, by definition, the exact gradient of the *surrogate*
    // `clamp(Wf/s_q, -1, 1)` — NOT of the real forward `round(clamp(...))`, whose
    // true derivative is 0 a.e. (round is piecewise-constant) and so cannot be
    // finite-difference-checked. We therefore finite-difference the surrogate.
    //
    // [N=4, K=5] hand-placed magnitudes so every |w/s_q| sits clearly in-band
    // (<0.9) or clearly saturated (>1.1) — never within `h` of the clamp kink at
    // |w/s_q|=1, which would make the central difference straddle it. Row 3 is
    // all-zero (s_q==0, the degenerate stop-gradient branch).
    #[rustfmt::skip]
    let wf: Vec<f32> = vec![
         0.2, -0.4,  1.5, -1.8,  0.1, // s_q=0.80: in-band {0,1,4}, saturated {2,3}
        -0.3,  0.6, -1.4,  1.9, -0.2, // s_q=0.88: in-band {0,1,4}, saturated {2,3}
         0.5, -0.5,  1.6, -1.7,  0.3, // s_q=0.92: in-band {0,1,4}, saturated {2,3}
         0.0,  0.0,  0.0,  0.0,  0.0, // s_q=0.00: degenerate row (all grads zero)
    ];
    let s_q = ste::absmean_scale_per_row(&wf, N, K); // constant of the forward
    let inputs = vec![wf, s_q];
    check_op(
        |ins| ste::quantize_surrogate(ins[0], ins[1], N, K),
        |ins, g| ste::quantize_vjp(ins[0], ins[1], N, K, g),
        &inputs,
        &[0], // wrt Wf only; s_q is stop-gradient
        GradCheckCfg::default(),
    )
    .expect("STE vjp must equal the surrogate's central finite difference");
}

#[test]
fn quantize_forward_rounds_into_band() {
    // Covers the real QAT forward's round() path, which the gradient check (above)
    // deliberately cannot: round(clamp(Wf/s_q)) must yield trits in {-1,0,+1}, and
    // a degenerate row (s_q==0) must be all-zero.
    #[rustfmt::skip]
    let wf: Vec<f32> = vec![
        0.2, -0.3,  1.5, -1.8,  0.1, // s_q=0.78 -> [0, 0, 1, -1, 0]
        0.0,  0.0,  0.0,  0.0,  0.0, // s_q=0.00 -> all zero
    ];
    let s_q = ste::absmean_scale_per_row(&wf, 2, K);
    let t = ste::quantize_forward(&wf, &s_q, 2, K);
    assert!(t.iter().all(|&x| x == -1.0 || x == 0.0 || x == 1.0));
    assert_eq!(t, vec![0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn ternary_matmul_grad_wrt_act_and_scale() {
    let act = seeded(2, M * K, -3.0, 3.0);
    // Pre-quantized trits as f32 in {-1,0,1}; matmul treats them as fixed here
    // (the Wf path is checked by the STE test + the composed toy-layer test later).
    let trits = seeded(3, N * K, -1.49, 1.49)
        .iter()
        .map(|x| x.round())
        .collect::<Vec<_>>();
    let scale = seeded(4, N, 0.1, 2.0);
    let inputs = vec![act, trits, scale];
    check_op(
        |ins| matmul::forward(ins[0], ins[1], ins[2], M, N, K),
        |ins, g| matmul::vjp(ins[0], ins[1], ins[2], M, N, K, g),
        &inputs,
        &[0, 2], // wrt act and scale; trits are the STE intermediate, checked separately
        GradCheckCfg::default(),
    )
    .expect("ternary matmul wrt act and scale must match finite difference");
}

#[test]
fn dense_matmul_grad_wrt_x_and_w() {
    // The plain dense matmul (the LoRA building block): Y[m,n] = Σ_k X[m,k]·W[n,k].
    // Differentiable in both inputs, so check both against central finite difference.
    let x = seeded(5, M * K, -2.0, 2.0);
    let w = seeded(6, N * K, -2.0, 2.0);
    let inputs = vec![x, w];
    check_op(
        |ins| dense::forward(ins[0], ins[1], M, N, K),
        |ins, g| dense::vjp(ins[0], ins[1], M, N, K, g),
        &inputs,
        &[0, 1],
        GradCheckCfg::default(),
    )
    .expect("dense matmul wrt X and W must match finite difference");
}
