//! Gate C (ADR 0007): every trainable CPU op's backward vs central finite difference.
//! Bias, squared-ReLU, MSE, softmax-cross-entropy, and element-wise add/mul.

use tritium_train::gradcheck::{GradCheckCfg, check_op};
use tritium_train::ops::{act, bias, elementwise, loss};

// Smooth deterministic fixture in [lo, hi); kept off any op kink by the callers.
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
fn bias_grad_wrt_x_and_b() {
    const ROWS: usize = 3;
    const COLS: usize = 4;
    let x = seeded(1, ROWS * COLS, -2.0, 2.0);
    let b = seeded(2, COLS, -1.0, 1.0);
    let inputs = vec![x, b];
    check_op(
        |ins| bias::forward(ins[0], ins[1], ROWS, COLS),
        |ins, g| bias::vjp(ins[0], ins[1], ROWS, COLS, g),
        &inputs,
        &[0, 1],
        GradCheckCfg::default(),
    )
    .expect("bias grad wrt x and b must match finite difference");
}

#[test]
fn relu2_grad_wrt_x() {
    // Values spread across both sides of 0; relu² is C¹ so the kink at 0 is benign.
    let x = seeded(3, 16, -2.0, 2.0);
    let inputs = vec![x];
    check_op(
        |ins| act::relu2_forward(ins[0]),
        |ins, g| act::relu2_vjp(ins[0], g),
        &inputs,
        &[0],
        GradCheckCfg::default(),
    )
    .expect("relu² grad wrt x must match finite difference");
}

#[test]
fn silu_grad_wrt_x() {
    // SiLU is C^∞; spread values across both sides of 0.
    let x = seeded(5, 16, -3.0, 3.0);
    let inputs = vec![x];
    check_op(
        |ins| act::silu_forward(ins[0]),
        |ins, g| act::silu_vjp(ins[0], g),
        &inputs,
        &[0],
        GradCheckCfg::default(),
    )
    .expect("silu grad wrt x must match finite difference");
}

#[test]
fn mse_grad_wrt_pred() {
    let pred = seeded(4, 12, -2.0, 2.0);
    let target = seeded(5, 12, -1.0, 1.0); // constant of the forward
    let inputs = vec![pred, target];
    check_op(
        |ins| loss::mse_forward(ins[0], ins[1]),
        |ins, g| loss::mse_vjp(ins[0], ins[1], g),
        &inputs,
        &[0], // wrt pred only
        GradCheckCfg::default(),
    )
    .expect("mse grad wrt pred must match finite difference");
}

#[test]
fn softmax_xent_grad_wrt_logits() {
    const ROWS: usize = 2;
    const COLS: usize = 3;
    let logits = seeded(6, ROWS * COLS, -2.0, 2.0);
    // Per-row one-hot target (a valid distribution, Σ_c = 1).
    let target = vec![
        1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0,
    ];
    let inputs = vec![logits, target];
    check_op(
        |ins| loss::softmax_xent_forward(ins[0], ins[1], ROWS, COLS),
        |ins, g| loss::softmax_xent_vjp(ins[0], ins[1], ROWS, COLS, g),
        &inputs,
        &[0], // wrt logits only
        GradCheckCfg::default(),
    )
    .expect("softmax-xent grad wrt logits must match finite difference");
}

#[test]
fn elementwise_add_grad() {
    let a = seeded(7, 10, -2.0, 2.0);
    let b = seeded(8, 10, -2.0, 2.0);
    let inputs = vec![a, b];
    check_op(
        |ins| elementwise::add_forward(ins[0], ins[1]),
        |ins, g| elementwise::add_vjp(ins[0], ins[1], g),
        &inputs,
        &[0, 1],
        GradCheckCfg::default(),
    )
    .expect("elementwise add grad must match finite difference");
}

#[test]
fn elementwise_mul_grad() {
    let a = seeded(9, 10, -2.0, 2.0);
    let b = seeded(10, 10, -2.0, 2.0);
    let inputs = vec![a, b];
    check_op(
        |ins| elementwise::mul_forward(ins[0], ins[1]),
        |ins, g| elementwise::mul_vjp(ins[0], ins[1], g),
        &inputs,
        &[0, 1],
        GradCheckCfg::default(),
    )
    .expect("elementwise mul grad must match finite difference");
}
