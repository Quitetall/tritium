//! Portable ternary Conv2d forward and VJP gates.

use tritium_train::Tape;
use tritium_train::gradcheck::{GradCheckCfg, check_op};
use tritium_train::ops::conv2d::{self, Conv2dCfg, Conv2dError};
use tritium_train::ops::loss;

fn seeded(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            lo + (state % 1000) as f32 / 1000.0 * (hi - lo)
        })
        .collect()
}

fn reference(x: &[f32], w: &[f32], scale: &[f32], cfg: &Conv2dCfg) -> Vec<f32> {
    let (h_out, w_out) = cfg.output_hw();
    let c_in_pg = cfg.c_in_per_group();
    let c_out_pg = cfg.c_out_per_group();
    let mut output = vec![0.0; cfg.batch * cfg.c_out * h_out * w_out];
    for batch in 0..cfg.batch {
        for group in 0..cfg.groups {
            for co_local in 0..c_out_pg {
                let co = group * c_out_pg + co_local;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0;
                        for ci_local in 0..c_in_pg {
                            let ci = group * c_in_pg + ci_local;
                            for kh in 0..cfg.kernel_h {
                                for kw in 0..cfg.kernel_w {
                                    let ih = oh as isize * cfg.stride_h as isize
                                        + kh as isize * cfg.dilation_h as isize
                                        - cfg.pad_top as isize;
                                    let iw = ow as isize * cfg.stride_w as isize
                                        + kw as isize * cfg.dilation_w as isize
                                        - cfg.pad_left as isize;
                                    if ih >= 0
                                        && iw >= 0
                                        && (ih as usize) < cfg.input_h
                                        && (iw as usize) < cfg.input_w
                                    {
                                        let x_index = ((batch * cfg.c_in + ci) * cfg.input_h
                                            + ih as usize)
                                            * cfg.input_w
                                            + iw as usize;
                                        let weight_col =
                                            (ci_local * cfg.kernel_h + kh) * cfg.kernel_w + kw;
                                        sum += x[x_index]
                                            * w[co * cfg.kernel_elements_per_output() + weight_col];
                                    }
                                }
                            }
                        }
                        let output_index = ((batch * cfg.c_out + co) * h_out + oh) * w_out + ow;
                        output[output_index] = sum * scale[co];
                    }
                }
            }
        }
    }
    output
}

fn check_cfg(name: &str, cfg: Conv2dCfg, seed: u64) {
    let (h_out, w_out) = cfg.output_hw();
    assert!(h_out > 0 && w_out > 0, "{name}: degenerate output");
    let x = seeded(seed, cfg.input_elements(), -1.0, 1.0);
    let weight = seeded(seed + 1, cfg.weight_elements(), -1.0, 1.0);
    let scale = seeded(seed + 2, cfg.c_out, 0.25, 1.25);
    assert_eq!(
        conv2d::forward(&x, &weight, &scale, &cfg),
        reference(&x, &weight, &scale, &cfg),
        "{name}: forward mismatch"
    );
    check_op(
        |inputs| conv2d::forward(inputs[0], inputs[1], inputs[2], &cfg),
        |inputs, grad| conv2d::vjp(inputs[0], inputs[1], inputs[2], &cfg, grad),
        &[x, weight, scale],
        &[0, 1, 2],
        GradCheckCfg {
            h: 1e-2,
            ..GradCheckCfg::default()
        },
    )
    .unwrap_or_else(|error| panic!("conv2d gradcheck failed for {name}: {error:?}"));
}

fn dense() -> Conv2dCfg {
    Conv2dCfg {
        batch: 1,
        c_in: 2,
        c_out: 3,
        input_h: 4,
        input_w: 5,
        kernel_h: 3,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        pad_top: 1,
        pad_bottom: 1,
        pad_left: 0,
        pad_right: 1,
        groups: 1,
    }
}

#[test]
fn dense_asymmetric_conv2d_matches_reference_and_gradcheck() {
    check_cfg("dense-asymmetric", dense(), 31);
}

#[test]
fn grouped_strided_dilated_conv2d_matches_reference_and_gradcheck() {
    check_cfg(
        "grouped-strided-dilated",
        Conv2dCfg {
            batch: 1,
            c_in: 4,
            c_out: 4,
            input_h: 5,
            input_w: 6,
            kernel_h: 2,
            kernel_w: 3,
            stride_h: 2,
            stride_w: 2,
            dilation_h: 2,
            dilation_w: 1,
            pad_top: 1,
            pad_bottom: 1,
            pad_left: 1,
            pad_right: 0,
            groups: 2,
        },
        32,
    );
}

#[test]
fn depthwise_pointwise_conv2d_matches_reference_and_gradcheck() {
    check_cfg(
        "depthwise-pointwise",
        Conv2dCfg {
            batch: 2,
            c_in: 3,
            c_out: 3,
            input_h: 3,
            input_w: 4,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            pad_top: 0,
            pad_bottom: 0,
            pad_left: 0,
            pad_right: 0,
            groups: 3,
        },
        33,
    );
}

#[test]
fn spatial_tile_boundary_bounds_scratch_and_matches_reference_and_gradcheck() {
    let cfg = Conv2dCfg {
        batch: 1,
        c_in: 1,
        c_out: 1,
        input_h: 8,
        input_w: 9,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        pad_top: 1,
        pad_bottom: 1,
        pad_left: 1,
        pad_right: 1,
        groups: 1,
    };
    assert_eq!(cfg.output_elements(), 72);
    assert_eq!(cfg.max_scratch_elements(), 32 * (9 + 1));
    check_cfg("spatial-tile-boundary", cfg, 34);
}

#[test]
fn tape_conv2d_routes_all_three_gradients() {
    let cfg = dense();
    let x = seeded(41, cfg.input_elements(), -0.5, 0.5);
    let weight = seeded(42, cfg.weight_elements(), -0.5, 0.5);
    let scale = seeded(43, cfg.c_out, 0.5, 1.0);
    let target = vec![0.0; cfg.output_elements()];
    let direct_output = conv2d::forward(&x, &weight, &scale, &cfg);
    let grad_output = loss::mse_vjp(&direct_output, &target, &[1.0]).remove(0);
    let direct_gradients = conv2d::vjp(&x, &weight, &scale, &cfg, &grad_output);
    let mut tape = Tape::new();
    let x_id = tape.leaf(x);
    let weight_id = tape.leaf(weight);
    let scale_id = tape.leaf(scale);
    let target_id = tape.leaf(target);
    let output = tape.conv2d(x_id, weight_id, scale_id, cfg);
    let loss = tape.mse(output, target_id);
    let gradients = tape.backward(loss);
    assert_eq!(gradients[x_id], direct_gradients[0]);
    assert_eq!(gradients[weight_id], direct_gradients[1]);
    assert_eq!(gradients[scale_id], direct_gradients[2]);
}

#[test]
fn fallible_conv2d_rejects_geometry_and_buffers_before_recording() {
    let cfg = dense();
    let input = vec![0.0; cfg.input_elements() - 1];
    let weight = vec![0.0; cfg.weight_elements()];
    let scale = vec![1.0; cfg.c_out];
    assert!(matches!(
        conv2d::try_forward(&input, &weight, &scale, &cfg),
        Err(Conv2dError::BufferLength {
            buffer: "input",
            ..
        })
    ));

    let malformed = Conv2dCfg { groups: 0, ..cfg };
    assert_eq!(malformed.output_hw(), (0, 0));
    assert!(matches!(
        conv2d::try_forward(&[], &[], &[], &malformed),
        Err(Conv2dError::InvalidGeometry(_))
    ));

    let mut tape = Tape::new();
    let input_id = tape.leaf(input);
    let weight_id = tape.leaf(weight);
    let scale_id = tape.leaf(scale);
    assert!(matches!(
        tape.try_conv2d(input_id, weight_id, scale_id, cfg),
        Err(Conv2dError::BufferLength {
            buffer: "input",
            ..
        })
    ));
}
