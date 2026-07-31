//! Iterative Ternary Fitting (ITF) for SALT planes — the PT²-LLM technique, specialized to a plane.
//!
//! The greedy fit takes `s = AbsMean(residual)`, which is a heuristic, not the error-minimizing scale.
//! For fixed trits the optimal scalar is the projection `<r,t>/<t,t>`; for fixed scale the optimal
//! ternary assignment is `clamp(round(r/s))`. Both half-steps exactly minimize `||r − s·t||²`, so
//! alternating them cannot increase the error — which is what these gates pin, on real weight-shaped
//! data as well as adversarial ones.

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

fn recon_sse(recon: &[f32], wf: &[f32]) -> f64 {
    recon
        .iter()
        .zip(wf)
        .map(|(&a, &b)| f64::from(a - b) * f64::from(a - b))
        .sum()
}

/// `iters = 0` must reproduce the committed greedy expansion **bit-for-bit** — ITF is opt-in and
/// cannot perturb any existing result (including the in-flight distillation runs).
#[test]
fn itf_zero_iters_is_the_greedy_fit() {
    for &(rows, cols) in &[(4usize, 16usize), (7, 65), (3, 256)] {
        let wf = seeded(0x117 ^ rows as u64, rows * cols, -2.0, 2.0);
        for t in 1..=3 {
            let greedy = ste::salt_quantize_forward(&wf, rows, cols, t);
            let itf0 = ste::salt_quantize_forward_itf(&wf, rows, cols, t, 0);
            assert_eq!(
                greedy, itf0,
                "iters=0 must equal the greedy fit ({rows}x{cols}, T={t})"
            );
        }
    }
}

/// The load-bearing property: ITF never reconstructs worse than AbsMean, at any plane count, on any
/// of these fixtures. Guaranteed by construction (each alternation half-step is an exact minimization,
/// and candidates are accepted only on strict improvement) — this gate holds it to that.
#[test]
fn itf_never_worsens_reconstruction() {
    for &(rows, cols) in &[(4usize, 16usize), (8, 64), (3, 257)] {
        for seed in [0x5A17u64, 0xBEEF, 0x1234] {
            let wf = seeded(seed ^ cols as u64, rows * cols, -1.5, 1.5);
            for t in 1..=3 {
                let greedy = ste::salt_quantize_forward(&wf, rows, cols, t);
                let e_greedy = recon_sse(&greedy, &wf);
                for iters in [1usize, 3, 8] {
                    let itf = ste::salt_quantize_forward_itf(&wf, rows, cols, t, iters);
                    let e_itf = recon_sse(&itf, &wf);
                    assert!(
                        e_itf <= e_greedy * (1.0 + 1e-6),
                        "ITF worsened SSE ({rows}x{cols}, T={t}, iters={iters}): {e_itf:.6e} > {e_greedy:.6e}"
                    );
                }
            }
        }
    }
}

/// And it should actually *help*, not just tie — on a single plane, where the scale carries all the
/// approximation, the least-squares scale beats AbsMean by a clear margin.
#[test]
fn itf_measurably_improves_the_single_plane_fit() {
    let (rows, cols) = (16usize, 128usize);
    let wf = seeded(0xF17, rows * cols, -1.0, 1.0);
    let greedy = recon_sse(&ste::salt_quantize_forward(&wf, rows, cols, 1), &wf);
    let itf = recon_sse(&ste::salt_quantize_forward_itf(&wf, rows, cols, 1, 5), &wf);
    let gain = (greedy - itf) / greedy;
    assert!(
        gain > 0.005,
        "expected a real single-plane gain, got {:.4}% (greedy {greedy:.6e} → itf {itf:.6e})",
        gain * 100.0
    );
    println!(
        "ITF single-plane SSE gain: {:.2}% ({greedy:.4e} → {itf:.4e})",
        gain * 100.0
    );

    // Report the gain across plane counts: the residual shrinks with T, so the question is whether
    // the least-squares scale still helps once several planes are stacked.
    for t in 1..=3 {
        let g = recon_sse(&ste::salt_quantize_forward(&wf, rows, cols, t), &wf);
        let i = recon_sse(&ste::salt_quantize_forward_itf(&wf, rows, cols, t, 5), &wf);
        println!(
            "  T={t}: SSE {g:.4e} → {i:.4e}  ({:.2}% better)",
            (g - i) / g * 100.0
        );
    }
}

/// Degenerate inputs must not produce NaN/inf or panic: an all-zero row (no scale), a single huge
/// outlier (all other trits round to 0), and a row where every weight is identical.
#[test]
fn itf_handles_degenerate_rows() {
    let (rows, cols) = (4usize, 8usize);
    let mut wf = seeded(0xD1E, rows * cols, -1.0, 1.0);
    wf[..cols].fill(0.0); // row 0: all zero
    wf[cols] = 1e6; // row 1: one huge outlier
    for c in 1..cols {
        wf[cols + c] = 1e-9;
    }
    for c in 0..cols {
        wf[2 * cols + c] = 0.5; // row 2: constant
    }
    for t in 1..=3 {
        for iters in [0usize, 1, 5] {
            let out = ste::salt_quantize_forward_itf(&wf, rows, cols, t, iters);
            assert!(
                out.iter().all(|v| v.is_finite()),
                "non-finite output (T={t}, iters={iters})"
            );
            assert!(
                out[..cols].iter().all(|&v| v == 0.0),
                "an all-zero row must reconstruct to zero"
            );
        }
    }
}
