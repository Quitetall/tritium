//! Gate C (ADR 0007): ternary Conv1d backward vs central finite difference, across every geometry
//! the codec uses (pointwise, depthwise, grouped, dilated, strided, padded, even+odd kernels).
//!
//! `conv1d::forward` is a smooth scale-folded contraction (im2col → ternary matmul → col2im), so it is
//! differentiable everywhere in `x`, `w`, and `scale` — no clamp kink to dodge (the STE round/clamp is
//! upstream and checked by `gradcheck_ste_matmul.rs`). We therefore finite-difference all three inputs.

use tritium_train::gradcheck::{GradCheckCfg, check_op};
use tritium_train::ops::conv1d::{self, Conv1dCfg};

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

/// Finite-difference the forward against the vjp for one geometry, w.r.t. x, w, and scale.
fn check_cfg(name: &str, cfg: Conv1dCfg, seed: u64) {
    assert!(cfg.l_out() > 0, "{name}: degenerate geometry");
    let x = seeded(seed, cfg.batch * cfg.c_in * cfg.l_in, -1.5, 1.5);
    let w = seeded(seed + 1, cfg.c_out * cfg.k_g(), -1.5, 1.5);
    let scale = seeded(seed + 2, cfg.c_out, 0.3, 1.5);
    let inputs = vec![x, w, scale];
    // conv1d is multilinear (linear in x, in w, and in scale separately), so the central difference is
    // exact in real arithmetic — the only error is f32 cancellation, which grows with the number of
    // outputs summed into the scalarized loss and with the overlap count per gX cell (dense k7 sums 7
    // taps). A larger step (1e-2 vs the 1e-3 default) reduces that cancellation with zero truncation
    // penalty on a linear function; the 2e-3 grade bar is unchanged.
    let cfg_gc = GradCheckCfg {
        h: 1e-2,
        ..GradCheckCfg::default()
    };
    check_op(
        |ins| conv1d::forward(ins[0], ins[1], ins[2], &cfg),
        |ins, g| conv1d::vjp(ins[0], ins[1], ins[2], &cfg, g),
        &inputs,
        &[0, 1, 2], // x, w, scale
        cfg_gc,
    )
    .unwrap_or_else(|e| panic!("conv1d gradcheck failed for {name} ({cfg:?}): {e:?}"));
}

fn base() -> Conv1dCfg {
    Conv1dCfg {
        batch: 2,
        c_in: 4,
        c_out: 6,
        l_in: 12,
        k: 3,
        stride: 1,
        dilation: 1,
        pad_left: 0,
        pad_right: 0,
        groups: 1,
    }
}

#[test]
fn conv1d_gradcheck_dense_k3() {
    check_cfg("dense k3", base(), 11);
}

#[test]
fn conv1d_gradcheck_pointwise() {
    check_cfg("pointwise", Conv1dCfg { k: 1, ..base() }, 12);
}

#[test]
fn conv1d_gradcheck_depthwise() {
    // groups == C_in == C_out ⇒ depthwise.
    check_cfg(
        "depthwise",
        Conv1dCfg {
            c_in: 6,
            c_out: 6,
            groups: 6,
            k: 5,
            ..base()
        },
        13,
    );
}

#[test]
fn conv1d_gradcheck_grouped() {
    check_cfg(
        "grouped-2",
        Conv1dCfg {
            groups: 2,
            k: 3,
            ..base()
        },
        14,
    );
}

#[test]
fn conv1d_gradcheck_dilated() {
    check_cfg(
        "dilated-2",
        Conv1dCfg {
            dilation: 2,
            k: 3,
            pad_left: 2,
            pad_right: 2,
            ..base()
        },
        15,
    );
}

#[test]
fn conv1d_gradcheck_strided_padded() {
    check_cfg(
        "stride2-pad1",
        Conv1dCfg {
            stride: 2,
            pad_left: 1,
            pad_right: 1,
            k: 3,
            ..base()
        },
        16,
    );
}

#[test]
fn conv1d_gradcheck_even_kernel_asym_pad() {
    // Even K with asymmetric "same"-style padding — the reason pad_left/pad_right are separate.
    check_cfg(
        "k4-pad-asym",
        Conv1dCfg {
            k: 4,
            pad_left: 2,
            pad_right: 1,
            ..base()
        },
        17,
    );
}

#[test]
fn conv1d_gradcheck_k7_causal() {
    // Wide odd kernel, causal left-pad (the codec's kernel-7 stages).
    check_cfg(
        "k7-causal",
        Conv1dCfg {
            k: 7,
            pad_left: 6,
            pad_right: 0,
            ..base()
        },
        18,
    );
}
