//! Gate C (ADR 0008 / plan 0011): the new transformer ops' backward vs central finite
//! difference — rmsnorm and row-wise softmax. (RoPE + the composed attention land next.)

use tritium_train::gradcheck::{GradCheckCfg, check_op};
use tritium_train::ops::{norm, rope, softmax};

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
fn rmsnorm_grad_wrt_x_and_w() {
    const ROWS: usize = 3;
    const COLS: usize = 5;
    const EPS: f32 = 1e-5;
    // Rows kept clearly nonzero so mean_sq + eps is well off 0 (inv smooth).
    let x = seeded(1, ROWS * COLS, -2.0, 2.0);
    let w = seeded(2, COLS, 0.3, 1.7);
    let inputs = vec![x, w];
    check_op(
        |ins| norm::forward(ins[0], ins[1], ROWS, COLS, EPS),
        |ins, g| norm::vjp(ins[0], ins[1], ROWS, COLS, EPS, g),
        &inputs,
        &[0, 1],
        GradCheckCfg::default(),
    )
    .expect("rmsnorm vjp must match central finite difference");
}

#[test]
fn softmax_grad_wrt_x() {
    const ROWS: usize = 4;
    const COLS: usize = 6;
    let x = seeded(3, ROWS * COLS, -3.0, 3.0);
    let inputs = vec![x];
    check_op(
        |ins| softmax::forward(ins[0], ROWS, COLS),
        |ins, g| softmax::vjp(ins[0], ROWS, COLS, g),
        &inputs,
        &[0],
        GradCheckCfg::default(),
    )
    .expect("softmax vjp must match central finite difference");
}

#[test]
fn rope_grad_wrt_x() {
    // [n_token=3, n_head=2, head_dim=4] flat = 24. RoPE is a position-parameterised
    // orthogonal rotation; positions/theta are data, only x is differentiated.
    const N_HEAD: usize = 2;
    const HEAD_DIM: usize = 4;
    const THETA: f32 = 10_000.0;
    let positions = [0usize, 1, 2];
    let x = seeded(4, positions.len() * N_HEAD * HEAD_DIM, -2.0, 2.0);
    let inputs = vec![x];
    check_op(
        |ins| rope::forward(ins[0], &positions, N_HEAD, HEAD_DIM, THETA),
        |_ins, g| rope::vjp(&positions, N_HEAD, HEAD_DIM, THETA, g),
        &inputs,
        &[0],
        GradCheckCfg::default(),
    )
    .expect("rope vjp must match central finite difference");
}
