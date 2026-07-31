//! Tequila (deadzone bias / leaky STE) and Sherry (cosine-annealed fp residual) validation.
//!
//! **Tequila.** The plain STE gives a saturated weight (`|Wf/s| >= 1`) *exactly zero* gradient, so it can
//! never recover. Tequila leaks a fraction `leak` through that region. The leaky surrogate
//! `clamp(x) + leak·(x − clamp(x))` and [`ste::quantize_vjp_leaky`] are an exact forward/backward pair,
//! so Gate C finite-differences them directly.
//!
//! **Sherry.** A forward-only blend `(1−α)·Ŵ + α·Wf` annealed to 0, so training starts near the smooth fp
//! landscape and ends fully ternary. Its endpoints are exact identities, which is what these gates pin.

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

/// Gate C for the leaky STE at several leaks, including 0 (the hard mask) and 1 (fully transparent).
/// The fixture deliberately spans well past `±s` so a large fraction of the weights are saturated —
/// otherwise the leak term is never exercised.
#[test]
fn tequila_leaky_ste_grad_matches_finite_difference() {
    const ROWS: usize = 3;
    const COLS: usize = 8;
    for leak in [0.0f32, 0.05, 0.25, 1.0] {
        let wf = seeded(21, ROWS * COLS, -3.0, 3.0); // |Wf| up to 3× the scale ⇒ many saturated
        let s_q = vec![1.0f32, 0.8, 1.3];
        let inputs = vec![wf, s_q];
        check_op(
            |ins| ste::quantize_surrogate_leaky(ins[0], ins[1], ROWS, COLS, leak),
            |ins, g| ste::quantize_vjp_leaky(ins[0], ins[1], ROWS, COLS, g, leak),
            &inputs,
            &[0], // wrt Wf (the scale is stop-gradient, as in quantize_vjp)
            GradCheckCfg::default(),
        )
        .unwrap_or_else(|e| panic!("leaky STE grad mismatch at leak={leak}: {e:?}"));
    }
}

/// `leak = 0` must reproduce the committed hard-mask estimators bit-for-bit — the new code paths are a
/// strict generalization, so no existing behaviour moves.
#[test]
fn tequila_leak_zero_is_the_committed_estimator() {
    const ROWS: usize = 4;
    const COLS: usize = 6;
    let wf = seeded(22, ROWS * COLS, -2.5, 2.5);
    let s_q = vec![1.0f32, 0.5, 1.5, 0.9];
    let g = seeded(23, ROWS * COLS, -1.0, 1.0);

    let base = ste::quantize_vjp(&wf, &s_q, ROWS, COLS, &g);
    let leaky = ste::quantize_vjp_leaky(&wf, &s_q, ROWS, COLS, &g, 0.0);
    assert_eq!(
        base, leaky,
        "quantize_vjp_leaky(leak=0) must equal quantize_vjp"
    );
    let surro = ste::quantize_surrogate(&wf, &s_q, ROWS, COLS);
    let surro_leaky = ste::quantize_surrogate_leaky(&wf, &s_q, ROWS, COLS, 0.0);
    assert_eq!(
        surro, surro_leaky,
        "leak=0 surrogate must equal the clamp surrogate"
    );

    let alpha = vec![1.0f32, 0.5, 1.5, 0.9];
    let lsq_base = ste::lsq_vjp(&wf, &alpha, ROWS, COLS, &g);
    let lsq_leaky = ste::lsq_vjp_leaky(&wf, &alpha, ROWS, COLS, &g, 0.0);
    assert_eq!(
        lsq_base, lsq_leaky,
        "lsq_vjp_leaky(leak=0) must equal lsq_vjp"
    );
}

/// The point of Tequila: a saturated weight goes from a *dead* gradient to a live one, scaled by `leak`.
/// (The SALT estimator is already fully transparent, so only these masked paths change — see the module
/// note in `ste.rs`.)
#[test]
fn tequila_revives_saturated_weights() {
    const ROWS: usize = 1;
    const COLS: usize = 3;
    let s_q = vec![1.0f32];
    // element 0 in-band, elements 1 and 2 saturated (+/-)
    let wf = vec![0.4f32, 2.0, -2.0];
    let g = vec![1.0f32, 1.0, 1.0];

    let dead = ste::quantize_vjp(&wf, &s_q, ROWS, COLS, &g).remove(0);
    assert_eq!(dead[1], 0.0, "plain STE kills the saturated weight");
    assert_eq!(dead[2], 0.0);

    let leak = 0.1f32;
    let alive = ste::quantize_vjp_leaky(&wf, &s_q, ROWS, COLS, &g, leak).remove(0);
    assert_eq!(alive[0], dead[0], "in-band gradient is untouched");
    assert!(
        (alive[1] - leak).abs() < 1e-6,
        "saturated weight now gets leak·g: {}",
        alive[1]
    );
    assert!((alive[2] - leak).abs() < 1e-6);

    // The LamQuant/LSQ path behaves the same way.
    let alpha = vec![1.0f32];
    let lsq_dead = ste::lsq_vjp(&wf, &alpha, ROWS, COLS, &g).remove(0);
    let lsq_alive = ste::lsq_vjp_leaky(&wf, &alpha, ROWS, COLS, &g, leak).remove(0);
    assert_eq!(lsq_dead[1], 0.0);
    assert!((lsq_alive[1] - leak).abs() < 1e-6);
}

/// Sherry's endpoints are exact: `α=0` is the untouched SALT reconstruction, `α=1` is the fp weight.
/// Anything in between interpolates, so a finished anneal leaves a purely ternary model.
#[test]
fn sherry_endpoints_are_exact_identities() {
    const ROWS: usize = 3;
    const COLS: usize = 8;
    const T: usize = 2;
    let wf = seeded(24, ROWS * COLS, -1.5, 1.5);

    let pure = ste::salt_quantize_forward(&wf, ROWS, COLS, T);
    let at_zero = ste::salt_quantize_forward_sherry(&wf, ROWS, COLS, T, 0.0);
    assert_eq!(
        at_zero, pure,
        "α=0 must be bit-identical to salt_quantize_forward"
    );

    let at_one = ste::salt_quantize_forward_sherry(&wf, ROWS, COLS, T, 1.0);
    for (a, b) in at_one.iter().zip(&wf) {
        assert!(
            (a - b).abs() < 1e-6,
            "α=1 must return the fp weight: {a} vs {b}"
        );
    }

    // Mid-anneal sits strictly between the two, and is closer to fp than the pure reconstruction is.
    let mid = ste::salt_quantize_forward_sherry(&wf, ROWS, COLS, T, 0.5);
    let err = |v: &[f32]| -> f32 { v.iter().zip(&wf).map(|(a, b)| (a - b) * (a - b)).sum() };
    assert!(
        err(&mid) < err(&pure),
        "blending fp in must reduce reconstruction error: {} vs {}",
        err(&mid),
        err(&pure)
    );
}

/// The anneal starts at `start`, decays monotonically, and is exactly 0 from `total` onward — so the
/// model is guaranteed fully ternary by the end of training.
#[test]
fn sherry_alpha_anneals_monotonically_to_zero() {
    let (start, total) = (0.5f32, 1000u64);
    assert!(
        (ste::sherry_alpha(start, 0, total) - start).abs() < 1e-6,
        "starts at `start`"
    );
    assert_eq!(
        ste::sherry_alpha(start, total, total),
        0.0,
        "hits 0 at total"
    );
    assert_eq!(
        ste::sherry_alpha(start, total + 500, total),
        0.0,
        "stays 0 past total"
    );
    let mut prev = f32::INFINITY;
    for step in (0..=total).step_by(50) {
        let a = ste::sherry_alpha(start, step, total);
        assert!(a <= prev + 1e-6, "must decay monotonically: {a} > {prev}");
        assert!(
            (0.0..=start + 1e-6).contains(&a),
            "stays in [0, start]: {a}"
        );
        prev = a;
    }
    assert_eq!(
        ste::sherry_alpha(start, 10, 0),
        0.0,
        "total=0 is a no-op anneal"
    );
}
