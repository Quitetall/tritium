//! Gate C (ADR 0007): FSQ straight-through backward vs central finite difference.
//!
//! The FSQ backward is, by the STE definition, the exact gradient of a smooth **surrogate** (the bound
//! for hard/stochastic, the annealed soft-round for soft) — not of the rounded forward, whose true
//! derivative is 0 a.e. We therefore finite-difference the surrogate. Clamp-bound inputs are placed off
//! the `|x|=1` kink; the soft surrogate uses small `L` so its `cos(2πz)` curvature stays inside the
//! central-difference truncation budget.

use tritium_train::gradcheck::{GradCheckCfg, check_op};
use tritium_train::ops::fsq::{self, FsqBound, FsqCfg};

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

fn cfg(channels: usize, len: usize, levels: &[u32], bound: FsqBound) -> FsqCfg {
    FsqCfg {
        channels,
        len,
        levels: levels.to_vec(),
        bound,
    }
}

#[test]
fn fsq_hard_clamp_grad_matches_surrogate() {
    // Hard STE, Clamp bound: gX = grad·1[|x|<1]. Inputs in (−0.85,0.85) so every element is clearly
    // in-band (never within h of the clamp kink), plus a couple clearly saturated to exercise the mask.
    let c = cfg(3, 4, &[2, 5, 8], FsqBound::Clamp);
    let mut x = seeded(1, 12, -0.85, 0.85);
    x[0] = 1.6; // saturated → zero grad
    x[11] = -1.7;
    check_op(
        |ins| fsq::surrogate(ins[0], &c),
        |ins, g| fsq::vjp_hard(ins[0], &c, g),
        &[x],
        &[0],
        GradCheckCfg::default(),
    )
    .expect("FSQ hard/clamp vjp must match the bound's finite difference");
}

#[test]
fn fsq_hard_tanh_grad_matches_surrogate() {
    // Tanh bound is smooth everywhere; gX = grad·(1−tanh²x). Moderate range so tanh' is well-scaled.
    let c = cfg(2, 5, &[4, 16], FsqBound::Tanh);
    let x = seeded(2, 10, -1.6, 1.6);
    check_op(
        |ins| fsq::surrogate(ins[0], &c),
        |ins, g| fsq::vjp_hard(ins[0], &c, g),
        &[x],
        &[0],
        GradCheckCfg::default(),
    )
    .expect("FSQ hard/tanh vjp must match the bound's finite difference");
}

#[test]
fn fsq_soft_tanh_grad_matches_surrogate() {
    // Annealed soft-round, Tanh bound. Small L keeps the cos(2πz) curvature inside the central-diff
    // truncation budget; alpha=0.5 is mid-anneal.
    let alpha = 0.5f32;
    let c = cfg(2, 4, &[2, 3], FsqBound::Tanh);
    let x = seeded(3, 8, -1.2, 1.2);
    check_op(
        |ins| fsq::surrogate_soft(ins[0], &c, alpha),
        |ins, g| fsq::vjp_soft(ins[0], &c, alpha, g),
        &[x],
        &[0],
        GradCheckCfg::default(),
    )
    .expect("FSQ soft/tanh vjp must match the soft surrogate's finite difference");
}

#[test]
fn fsq_soft_clamp_grad_matches_surrogate() {
    let alpha = 0.4f32;
    let c = cfg(2, 4, &[2, 3], FsqBound::Clamp);
    let x = seeded(4, 8, -0.8, 0.8); // off the clamp kink
    check_op(
        |ins| fsq::surrogate_soft(ins[0], &c, alpha),
        |ins, g| fsq::vjp_soft(ins[0], &c, alpha, g),
        &[x],
        &[0],
        GradCheckCfg::default(),
    )
    .expect("FSQ soft/clamp vjp must match the soft surrogate's finite difference");
}
