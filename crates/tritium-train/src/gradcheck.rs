//! Finite-difference gradient check — the engine behind ADR 0007 Gate C.
//!
//! For an op `forward: inputs -> output` and its `vjp: (inputs, grad_out) -> grads`,
//! pick a fixed random cotangent `r` over the output, scalarize `g(x) = Σ r·forward(x)`,
//! and compare each analytic grad `vjp(inputs, r)` to the central finite difference
//! `(g(x+h·e_i) − g(x−h·e_i)) / 2h`. Within tolerance ⇒ the backward is correct.

use tritium_testkit::Tolerance;

/// Why a gradient check failed.
#[derive(Clone, Debug, PartialEq)]
pub struct GradCheckFail {
    /// Which input (index into the op's input list) disagreed.
    pub input: usize,
    /// Flat element index within that input.
    pub index: usize,
    /// The analytic gradient the `vjp` returned.
    pub analytic: f32,
    /// The central finite-difference estimate.
    pub numeric: f32,
}

/// Measured error envelope for one successful finite-difference check.
#[derive(Clone, Debug, PartialEq)]
pub struct GradCheckReport {
    /// Input ordinals checked, in caller-supplied order.
    pub checked_inputs: Vec<usize>,
    /// Total scalar partial derivatives compared.
    pub checked_elements: usize,
    /// Largest `|analytic-numeric| / max(1, |numeric|)` observed.
    pub max_relative_error: f32,
}

/// Gradient-check configuration.
#[derive(Clone, Copy, Debug)]
pub struct GradCheckCfg {
    /// Finite-difference step. `1e-3` is the f32 sweet spot.
    pub h: f32,
    /// Grading rule (Gate C bar: `relative = 2e-3`, not bit-exact). Note
    /// `Tolerance` floors the comparison denominator at 1.0, so for sub-unit
    /// gradients (every value in the v0.50 fixtures) this is effectively an
    /// absolute 2e-3 bound; it only becomes truly relative once `|grad| > 1`.
    pub tol: Tolerance,
    /// Seed for the random output cotangent (xorshift64 idiom).
    pub seed: u64,
}

impl Default for GradCheckCfg {
    fn default() -> Self {
        GradCheckCfg {
            h: 1e-3,
            tol: Tolerance::relative(2e-3),
            seed: 0xC0FFEE,
        }
    }
}

/// xorshift64 — the repo's dependency-free PRNG idiom (seed forced non-zero).
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// A fixed random cotangent in `[-1, 1)` of length `n`.
fn cotangent(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| (xorshift(&mut s) % 2000) as f32 / 1000.0 - 1.0)
        .collect()
}

/// Compare an op's analytic `vjp` to a central finite difference.
///
/// `forward` maps the input buffers to one output buffer; `vjp` maps `(inputs, grad_out)`
/// to one grad buffer per input (same shapes as the inputs). `wrt` lists the input
/// indices to check (skip constants such as the quantizer scale). Returns the first
/// disagreement, or `Ok(())` if all checked grads are within `cfg.tol`.
pub fn check_op<F, V>(
    forward: F,
    vjp: V,
    inputs: &[Vec<f32>],
    wrt: &[usize],
    cfg: GradCheckCfg,
) -> Result<(), GradCheckFail>
where
    F: Fn(&[&[f32]]) -> Vec<f32>,
    V: Fn(&[&[f32]], &[f32]) -> Vec<Vec<f32>>,
{
    check_op_report(forward, vjp, inputs, wrt, cfg).map(|_| ())
}

/// Compare an op's analytic VJP and return its measured passing envelope.
///
/// This executes the same check as [`check_op`] while retaining the maximum
/// normalized error needed by source-bound release receipts.
pub fn check_op_report<F, V>(
    forward: F,
    vjp: V,
    inputs: &[Vec<f32>],
    wrt: &[usize],
    cfg: GradCheckCfg,
) -> Result<GradCheckReport, GradCheckFail>
where
    F: Fn(&[&[f32]]) -> Vec<f32>,
    V: Fn(&[&[f32]], &[f32]) -> Vec<Vec<f32>>,
{
    let refs: Vec<&[f32]> = inputs.iter().map(Vec::as_slice).collect();
    let out_len = forward(&refs).len();
    let r = cotangent(cfg.seed, out_len);

    let scalar =
        |ins: &[&[f32]]| -> f32 { forward(ins).iter().zip(&r).map(|(&y, &ri)| y * ri).sum() };

    let analytic = vjp(&refs, &r);
    let mut checked_elements = 0usize;
    let mut max_relative_error = 0.0f32;

    for &wi in wrt {
        let n = inputs[wi].len();
        for i in 0..n {
            let mut plus = inputs.to_vec();
            let mut minus = inputs.to_vec();
            plus[wi][i] += cfg.h;
            minus[wi][i] -= cfg.h;
            let pr: Vec<&[f32]> = plus.iter().map(Vec::as_slice).collect();
            let mr: Vec<&[f32]> = minus.iter().map(Vec::as_slice).collect();
            let numeric = (scalar(&pr) - scalar(&mr)) / (2.0 * cfg.h);
            let a = analytic[wi][i];
            if !a.is_finite() || !numeric.is_finite() || !cfg.tol.accepts(a, numeric) {
                return Err(GradCheckFail {
                    input: wi,
                    index: i,
                    analytic: a,
                    numeric,
                });
            }
            max_relative_error =
                max_relative_error.max((a - numeric).abs() / numeric.abs().max(1.0));
            checked_elements += 1;
        }
    }
    Ok(GradCheckReport {
        checked_inputs: wrt.to_vec(),
        checked_elements,
        max_relative_error,
    })
}

/// Execute ADR-0035's canonical HESTIA Gate-C fixture.
///
/// Gate C checks every weight derivative plus the scalar temperature
/// derivative. Per-row scale remains stop-gradient by contract.
pub fn hestia_gate_c_report() -> Result<GradCheckReport, GradCheckFail> {
    const ROWS: usize = 3;
    const COLS: usize = 5;
    let inputs = vec![
        seeded(21, ROWS * COLS, -2.0, 2.0),
        seeded(22, ROWS, 0.5, 1.5),
        vec![0.7],
    ];
    check_op_report(
        |ins| crate::ops::hestia::hestia_forward(ins[0], ins[1], ins[2], ROWS, COLS),
        |ins, grad| crate::ops::hestia::hestia_vjp(ins[0], ins[1], ins[2], ROWS, COLS, grad),
        &inputs,
        &[0, 2],
        GradCheckCfg::default(),
    )
}

fn seeded(seed: u64, count: usize, low: f32, high: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            low + (state % 1000) as f32 / 1000.0 * (high - low)
        })
        .collect()
}
