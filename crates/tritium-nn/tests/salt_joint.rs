//! **Greedy residual expansion vs exact joint assignment across all T planes.**
//!
//! SALT fits plane 1, subtracts it, fits plane 2 on what's left. That is matching pursuit: earlier
//! planes' TRITS are never revisited in light of later ones. ITF refines each plane's *scale* but
//! never re-decides its trits, so the whole stack is greedy by construction — the one structural
//! limitation of the additive-residual formulation that is not a bug and not a tuning knob.
//!
//! `tritium_quantize::fit_joint_ternary` is the alternative, and it has never run: an OA-EM
//! alternation whose E-step assigns ALL planes' trits for one weight jointly by exact enumeration of
//! `3^P` states (3, 9, or 27 candidates), and whose M-step solves every plane's scale together under
//! a ridge-conditioned system.
//!
//! Unlike GPTQ — which `salt_gptq.rs` showed is a *substitute* for our joint-group fitter and loses
//! to the salience fold — this attacks a genuinely different axis: not which curvature the fit is
//! weighted by, but whether the plane assignment is greedy or joint. It should compose.
//!
//! `#[ignore]`d; run:
//! ```text
//! TRITIUM_CORPUS=<corpus.json> cargo test -p tritium-nn --release \
//!   --test salt_joint -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold, perplexity_windowed};
use tritium_nn::ModelRunner;
use tritium_quantize::{JointFitConfig, JointFitMetric, fit_joint_ternary};
use tritium_train::ops::ste::{self, RotationPolicy};

const EVAL_WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 8;
const CALIB_SEQ: usize = 512;
const GROUP: usize = 128;
const ITERS: usize = 5;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn corpus() -> (Vec<u32>, Vec<u32>) {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_string()
    });
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("corpus json")).expect("parse");
    let ids = |k: &str| -> Vec<u32> {
        j[k].as_array()
            .expect(k)
            .iter()
            .map(|v| v.as_u64().expect("id") as u32)
            .collect()
    };
    (ids("train_ids"), ids("eval_ids"))
}

/// Joint fit of one scale group, with the same optional Hadamard the greedy path uses.
///
/// Rotation is applied around the fit exactly as `ste::fit_group` does — into the rotated basis,
/// fit, back — so this isolates GREEDY vs JOINT and nothing else.
fn joint_group(bs: &[f32], t: usize, rotate: bool, cfg: JointFitConfig) -> Vec<f32> {
    let fit_in = |v: &[f32]| -> Option<Vec<f32>> {
        fit_joint_ternary(v, JointFitMetric::Identity, cfg)
            .ok()
            .map(|f| f.reconstruction)
    };
    let rotatable = rotate && bs.len().is_power_of_two() && bs.len() > 1;
    if rotatable {
        let mut buf = bs.to_vec();
        ste::fast_hadamard(&mut buf);
        match fit_in(&buf) {
            Some(q) => {
                let mut back = q;
                ste::fast_hadamard(&mut back);
                back
            }
            // A failed fit must not silently become zeros; fall back to the greedy path.
            None => ste::salt_quantize_forward_grouped(
                bs,
                1,
                bs.len(),
                t,
                GROUP,
                ITERS,
                RotationPolicy::Always,
            ),
        }
    } else {
        fit_in(bs).unwrap_or_else(|| {
            ste::salt_quantize_forward_grouped(
                bs,
                1,
                bs.len(),
                t,
                GROUP,
                ITERS,
                RotationPolicy::Never,
            )
        })
    }
}

/// Whole-tensor joint fit, choosing rotation per group by which fits better — the same `Auto`
/// decision rule the greedy path uses, so the comparison stays like-for-like.
fn joint(w: &[f32], rows: usize, cols: usize, t: usize, cfg: JointFitConfig) -> Vec<f32> {
    let sse = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x - y) * f64::from(x - y))
            .sum()
    };
    let mut out = vec![0.0f32; w.len()];
    for r in 0..rows {
        let src = &w[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (bs, bd) in src.chunks(GROUP).zip(dst.chunks_mut(GROUP)) {
            let plain = joint_group(bs, t, false, cfg);
            let rot = joint_group(bs, t, true, cfg);
            let pick = if sse(&rot, bs) < sse(&plain, bs) {
                rot
            } else {
                plain
            };
            bd.copy_from_slice(&pick);
        }
    }
    out
}

#[test]
#[ignore = "slow joint-vs-greedy sweep; needs SmolLM2-135M; run explicitly"]
fn joint_plane_assignment_vs_greedy_residual() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);

    // The fold is the strongest known configuration, so measure joint-vs-greedy ON TOP of it —
    // otherwise a win here might just be recovering what the fold already recovers.
    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp,
            &arch,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp, arch) = {
        let (folded, farch) = fold(&fp, &shapes, &arch, &calib, 0.75);
        (folded, farch)
    };
    println!("fp {ppl_fp:.3} ppl | salience fold alpha=0.75 applied to both arms\n");
    println!(
        "{:<34} {:>8} {:>13} {:>10}",
        "configuration", "bpw", "ppl", "× fp"
    );
    println!("{}", "-".repeat(70));

    for t in 1..=3usize {
        let bpw = ste::ternary_bits_per_weight(t, GROUP) + 1.0 / GROUP as f64;

        let greedy: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| {
                ste::salt_quantize_forward_grouped(w, n, k, t, GROUP, ITERS, RotationPolicy::Auto)
            })
            .collect();
        let p_greedy = perplexity_windowed(&greedy, &arch, &eval, EVAL_WINDOW);
        println!(
            "{:<34} {bpw:>8.2} {p_greedy:>13.3} {:>9.2}×",
            format!("T={t} greedy residual +ITF"),
            p_greedy / ppl_fp
        );

        let cfg = JointFitConfig {
            planes: t,
            ..JointFitConfig::default()
        };
        let jt: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| joint(w, n, k, t, cfg))
            .collect();
        let p_joint = perplexity_windowed(&jt, &arch, &eval, EVAL_WINDOW);
        println!(
            "{:<34} {bpw:>8.2} {p_joint:>13.3} {:>9.2}×   ({:+.1}% vs greedy)",
            format!("T={t} joint 3^{t} assignment"),
            p_joint / ppl_fp,
            (p_joint / p_greedy - 1.0) * 100.0
        );
    }
    println!(
        "\nBoth arms share the salience fold, g128 scale groups, and the same per-group Auto \
         rotation rule. The only difference is greedy residual expansion versus exact joint \
         assignment over all T planes."
    );
}
