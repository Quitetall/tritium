//! HESTIA differentiable-ternarization backward vs central finite difference (ADR 0035 WS-C1).
//!
//! Unlike the STE/LSQ checks — whose clamp-kinked surrogates force hand-placed inputs — the HESTIA
//! forward is smooth everywhere in both `Wf` and `τ`, so the finite-difference check needs no kink
//! placement and covers both gradients at arbitrary points: the Gate-C strengthening ADR 0035
//! records. The τ-limits, odd symmetry, and the zero-scale/stop-gradient conventions are pinned
//! alongside.

use tritium_train::gradcheck::{GradCheckCfg, check_op};
use tritium_train::ops::{hestia, ste};
use tritium_train::tape::Tape;

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
fn hestia_grads_match_finite_difference() {
    // Random inputs anywhere on the line are valid — no kink placement (contrast gradcheck_lsq).
    // wrt = [Wf, τ]; the scale (input 1) is stop-gradient by design, pinned separately below.
    let wf = seeded(21, ROWS * COLS, -2.0, 2.0);
    let s = seeded(22, ROWS, 0.5, 1.5);
    let tau = vec![0.7f32];
    let inputs = vec![wf, s, tau];
    check_op(
        |ins| hestia::hestia_forward(ins[0], ins[1], ins[2], ROWS, COLS),
        |ins, g| hestia::hestia_vjp(ins[0], ins[1], ins[2], ROWS, COLS, g),
        &inputs,
        &[0, 2],
        GradCheckCfg::default(),
    )
    .expect("HESTIA vjp must equal the forward's central finite difference");
}

#[test]
fn tau_to_zero_converges_to_hard_quantize() {
    // At τ = 1e-3 the softmax is effectively one-hot on the nearest trit; with s = 1 (out is the
    // trit itself) and every z kept > 0.1 away from the ±0.5 decision boundaries, the relaxation
    // matches the hard round-clamp forward to float noise.
    #[rustfmt::skip]
    let wf: Vec<f32> = vec![
         0.2,  -0.35,  0.75, -0.9,   1.4,
        -1.8,   0.05,  0.62, -0.15,  0.99,
         1.15, -0.7,   0.3,  -2.4,   0.0,
    ];
    let s = vec![1.0f32; ROWS];
    let tau = vec![1e-3f32];
    let soft = hestia::hestia_forward(&wf, &s, &tau, ROWS, COLS);
    let hard = ste::quantize_forward(&wf, &s, ROWS, COLS);
    for (i, (&a, &b)) in soft.iter().zip(&hard).enumerate() {
        assert!((a - b).abs() < 1e-4, "elem {i}: soft {a} vs hard {b}");
    }
}

#[test]
fn tau_large_output_approaches_zero() {
    // τ → ∞ ⇒ uniform π ⇒ E = π₊₁ − π₋₁ → 0 (leading term 4z/(3τ)).
    let wf = seeded(31, 12, -2.0, 2.0);
    let s = seeded(32, 4, 0.5, 1.5);
    let tau = vec![1e5f32];
    let out = hestia::hestia_forward(&wf, &s, &tau, 4, 3);
    for (i, &o) in out.iter().enumerate() {
        assert!(o.abs() < 1e-3, "elem {i}: |{o}| must vanish at large τ");
    }
}

#[test]
fn forward_is_odd_in_wf() {
    // Negating z swaps the ±1 softmax lanes bit-exactly (the partition sum is ordered to be
    // swap-invariant), so out(−Wf) == −out(Wf) holds exactly, not just approximately.
    let wf = seeded(41, 12, -2.0, 2.0);
    let neg: Vec<f32> = wf.iter().map(|w| -w).collect();
    let s = seeded(42, 4, 0.5, 1.5);
    let tau = vec![0.6f32];
    let plus = hestia::hestia_forward(&wf, &s, &tau, 4, 3);
    let minus = hestia::hestia_forward(&neg, &s, &tau, 4, 3);
    for (i, (&p, &m)) in plus.iter().zip(&minus).enumerate() {
        assert_eq!(p, -m, "elem {i}: out(−Wf) must equal −out(Wf)");
    }
}

#[test]
fn zero_scale_row_outputs_and_grads_zero() {
    let (rows, cols) = (2usize, 4usize);
    let wf = seeded(51, rows * cols, -2.0, 2.0);
    let s = vec![1.0f32, 0.0];
    let tau = vec![0.5f32];
    let out = hestia::hestia_forward(&wf, &s, &tau, rows, cols);
    assert!(
        out[cols..].iter().all(|&o| o == 0.0),
        "zero-scale row must output 0"
    );
    assert!(
        out[..cols].iter().any(|&o| o != 0.0),
        "live row must be nontrivial"
    );
    let g = vec![1.0f32; rows * cols];
    let grads = hestia::hestia_vjp(&wf, &s, &tau, rows, cols, &g);
    assert!(
        grads[0][cols..].iter().all(|&v| v == 0.0),
        "zero-scale row: zero Wf grad"
    );
    assert!(grads[1].iter().all(|&v| v == 0.0), "scale is stop-gradient");
    // τ accumulates over live rows only: all-zero s kills the τ grad entirely.
    let dead = hestia::hestia_vjp(&wf, &[0.0, 0.0], &tau, rows, cols, &g);
    assert_eq!(dead[2][0], 0.0);
}

#[test]
fn unrepresentable_temperature_fails_closed_without_nan() {
    let weight = [0.25f32, -0.75, 1.0, -1.5];
    let scale = [0.5f32, 1.25];
    let tau = [hestia::MIN_DIFFERENTIABLE_TAU * 0.5];
    let grad_output = [1.0f32; 4];
    assert_eq!(
        hestia::hestia_forward(&weight, &scale, &tau, 2, 2),
        [0.0; 4]
    );
    assert_eq!(
        hestia::hestia_vjp(&weight, &scale, &tau, 2, 2, &grad_output),
        [vec![0.0; 4], vec![0.0; 2], vec![0.0]]
    );
}

#[test]
fn exact_temperature_floor_has_finite_exact_vjp() {
    let weight = [1.0f32];
    let scale = [1.0f32];
    let tau = [hestia::MIN_DIFFERENTIABLE_TAU];
    let grad_output = [1.0f32];
    assert_eq!(hestia::hestia_forward(&weight, &scale, &tau, 1, 1), [1.0]);
    assert_eq!(
        hestia::hestia_vjp(&weight, &scale, &tau, 1, 1, &grad_output),
        [vec![0.0], vec![0.0], vec![0.0]]
    );
}

#[test]
fn tape_hestia_relax_smoke() {
    // Record → backward through MSE: grads carry the input shapes, the scale grad is all-zero
    // (stop-gradient), and both Wf and τ receive nonzero gradients.
    let (rows, cols) = (2usize, 3usize);
    let mut tape = Tape::new();
    let wf = tape.leaf(vec![0.3, -0.8, 1.2, -0.2, 0.6, -1.5]);
    let s = tape.leaf(vec![0.9, 1.1]);
    let tau = tape.leaf(vec![0.7]);
    let q = tape.hestia_relax(wf, s, tau, rows, cols);
    let target = tape.leaf(vec![0.0f32; rows * cols]);
    let loss = tape.mse(q, target);
    let grads = tape.backward(loss);
    assert_eq!(grads[wf].len(), rows * cols);
    assert_eq!(grads[s].len(), rows);
    assert_eq!(grads[tau].len(), 1);
    assert!(grads[s].iter().all(|&v| v == 0.0), "gS must be all-zero");
    assert!(
        grads[wf].iter().any(|&v| v != 0.0),
        "Wf must receive gradient"
    );
    assert!(grads[tau][0] != 0.0, "τ must receive gradient");
}
