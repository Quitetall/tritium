//! The `tritium-core` conformance oracles (`reference_conv1d` / `reference_fsq`) must be **bit-identical**
//! to the training ops — the multiply-free ternary conv equals `conv1d::forward` at ternary weights, and
//! the clamp-grid FSQ equals `fsq::forward` (clamp bound, hard STE). This equality is the foundation of
//! the cross-backend codec conformance (ADR 0030 Tier 4): every backend matches the core oracle, and the
//! oracle matches what the trainer produced.

use tritium_core::{ConvShape, Trit, reference_conv1d, reference_fsq};
use tritium_train::ops::conv1d::{self, Conv1dCfg};
use tritium_train::ops::fsq::{self, FsqBound, FsqCfg, FsqSte};

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

/// A ternary weight vector as `f32` in `{-1,0,+1}` (what the training path carries) alongside the same
/// values as [`Trit`] (what the core oracle takes).
fn ternary(seed: u64, n: usize) -> (Vec<f32>, Vec<Trit>) {
    let raw = seeded(seed, n, -1.5, 1.5);
    let f: Vec<f32> = raw
        .iter()
        .map(|&v| {
            if v < -0.5 {
                -1.0
            } else if v > 0.5 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let t: Vec<Trit> = f.iter().map(|&v| Trit::from_sign(v as i8)).collect();
    (f, t)
}

fn shape_of(c: &Conv1dCfg) -> ConvShape {
    ConvShape {
        batch: c.batch,
        c_in: c.c_in,
        c_out: c.c_out,
        l_in: c.l_in,
        k: c.k,
        stride: c.stride,
        dilation: c.dilation,
        pad_left: c.pad_left,
        pad_right: c.pad_right,
        groups: c.groups,
    }
}

#[test]
fn reference_conv1d_matches_training_forward_at_ternary() {
    let base = Conv1dCfg {
        batch: 2,
        c_in: 6,
        c_out: 6,
        l_in: 14,
        k: 3,
        stride: 1,
        dilation: 1,
        pad_left: 1,
        pad_right: 1,
        groups: 1,
    };
    let cfgs = [
        ("dense", base),
        (
            "pointwise",
            Conv1dCfg {
                k: 1,
                pad_left: 0,
                pad_right: 0,
                ..base
            },
        ),
        (
            "depthwise",
            Conv1dCfg {
                groups: 6,
                k: 5,
                pad_left: 2,
                pad_right: 2,
                ..base
            },
        ),
        ("grouped", Conv1dCfg { groups: 3, ..base }),
        (
            "dilated",
            Conv1dCfg {
                dilation: 2,
                pad_left: 2,
                pad_right: 2,
                ..base
            },
        ),
        ("stride2", Conv1dCfg { stride: 2, ..base }),
        (
            "k7-causal",
            Conv1dCfg {
                k: 7,
                pad_left: 6,
                pad_right: 0,
                ..base
            },
        ),
    ];
    for (name, cfg) in cfgs {
        let x = seeded(0x100, cfg.batch * cfg.c_in * cfg.l_in, -2.0, 2.0);
        let (wf, wt) = ternary(0x200, cfg.c_out * cfg.k_g());
        let scale = seeded(0x300, cfg.c_out, 0.2, 1.7);

        let y_train = conv1d::forward(&x, &wf, &scale, &cfg);
        let mut y_ref = vec![0.0f32; y_train.len()];
        reference_conv1d(&x, &wt, &scale, shape_of(&cfg), &mut y_ref).unwrap();

        assert_eq!(y_train, y_ref, "conv oracle != training forward for {name}");
    }
}

#[test]
fn reference_fsq_matches_training_forward_clamp_hard() {
    let (channels, len) = (4usize, 9usize);
    let levels = vec![2u32, 3, 5, 8];
    let x = seeded(0x400, channels * len, -1.4, 1.4); // spans in-band + saturated
    let cfg = FsqCfg {
        channels,
        len,
        levels: levels.clone(),
        bound: FsqBound::Clamp,
    };
    let q_train = fsq::forward(&x, &cfg, FsqSte::Hard);
    let mut q_ref = vec![0.0f32; x.len()];
    reference_fsq(&x, &levels, channels, len, &mut q_ref).unwrap();
    assert_eq!(q_train, q_ref, "FSQ clamp oracle != training forward");
}
