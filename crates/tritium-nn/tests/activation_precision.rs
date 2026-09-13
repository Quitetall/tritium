//! **What does the model cost when the activations are quantized too — and how far can that go?**
//!
//! Every SALT perplexity this repo publishes is weight-quantized only; the tape consumes fp32
//! activations. The shipping runtime int8-quantizes per token before each projection, which costs
//! +2.76% at `T=4` and, as a fraction of the excess over fp, **doubles** the gap (2.77% → 5.60%).
//! Nobody has measured any other activation precision, because there was no forward pass that
//! could express one. [`forward_aq`] is that pass; this sweeps it.
//!
//! Three questions in one grid, because they are the same question at different precisions:
//!
//! 1. **Is per-row absmax the defect?** The A8 headroom measurement found **5.41 dB** available on
//!    the FFN intermediate from per-group scales — 57% of a SALT plane, at zero bit cost. `int8/g128`
//!    against `int8/row` converts that bound into perplexity.
//! 2. **How cheap can activations get?** `int4` and ternary planes say where the cliff is.
//! 3. **What happens when EVERYTHING is ternary?** Weights already are. Ternary activations make the
//!    GEMM add/subtract only — no multiplies anywhere in the projection path. That is a **compute**
//!    result, not a compression one: activations are transient, so ternary buys no bytes.
//!
//! # What this harness does and does not cover
//!
//! It quantizes the inputs to q/k/v, gate/up, down, and the tied head. It does **not** quantize
//! `o_proj`'s input, which lives inside `attention()`. So one of seven projections per layer keeps
//! fp32 activations and every number here is an **upper bound** on a real deployment at the same
//! setting. A "fully ternary" claim from this file would be false, and is not made.
//!
//! Weights are held at a fixed `T` throughout so the only moving part is activation precision.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test activation_precision -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold, score_window};
use tritium_nn::ModelRunner;
use tritium_nn::calibrate::forward_aq;
use tritium_train::Tape;
use tritium_train::ops::ste::{self, RotationPolicy};
use tritium_train::tape::ValueId;

const EVAL_WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 4;
const CALIB_SEQ: usize = 512;
const WEIGHT_GROUP: usize = 128;
const GRID: usize = 16;
const ACT_GROUP: usize = 128;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR")
            .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m")),
    )
}

fn corpus() -> (Vec<u32>, Vec<u32>) {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_owned()
    });
    let text = std::fs::read_to_string(&path).expect("corpus");
    let v: serde_json::Value = serde_json::from_str(&text).expect("corpus json");
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .expect(k)
            .iter()
            .map(|x| x.as_u64().expect("id") as u32)
            .collect()
    };
    (ids("train_ids"), ids("eval_ids"))
}

/// Symmetric uniform quantization of one contiguous span to `levels` positive steps.
///
/// `levels = 127` is int8, `7` is int4. The step comes from the span's own absmax, so the *span* is
/// the granularity knob: a whole row reproduces the shipping per-token quantizer, a 128-wide slice
/// is the per-group variant the headroom measurement argued for.
fn quant_span(span: &mut [f32], levels: f32) {
    let gamma = span.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if gamma <= 0.0 {
        return;
    }
    let s = levels / gamma;
    for v in span.iter_mut() {
        // Round half to even, matching the reference activation quantizer's rounding mode.
        let scaled = (*v * s).round_ties_even().clamp(-levels - 1.0, levels);
        *v = scaled / s;
    }
}

/// How activations are quantized at each projection input.
#[derive(Clone, Copy)]
enum Mode {
    /// fp32 — what every published perplexity in this repo actually used.
    None,
    /// Uniform, one absmax per token. `levels = 127` is the shipping A8 path.
    PerRow(f32),
    /// Uniform, one absmax per `ACT_GROUP` span — the headroom measurement's proposal.
    PerGroup(f32),
    /// Additive ternary planes with the geometric ladder, i.e. the weight-side representation
    /// applied to activations. `T` planes ⇒ `T * 1.58` bits per value, multiply-free.
    Ternary(usize),
}

impl Mode {
    fn label(self) -> String {
        match self {
            Mode::None => "fp32 (published basis)".to_owned(),
            Mode::PerRow(127.0) => "int8 per row (SHIPPING)".to_owned(),
            Mode::PerRow(l) => format!("int{} per row", (l + 1.0).log2() as u32 + 1),
            Mode::PerGroup(127.0) => format!("int8 per group g{ACT_GROUP}"),
            Mode::PerGroup(l) => {
                format!("int{} per group g{ACT_GROUP}", (l + 1.0).log2() as u32 + 1)
            }
            Mode::Ternary(t) => format!("TERNARY {t} plane(s), {:.2} bits", t as f64 * 1.585),
        }
    }

    fn apply(self, v: &mut [f32], seq: usize, cols: usize) {
        match self {
            Mode::None => {}
            Mode::PerRow(levels) => {
                for r in 0..seq {
                    quant_span(&mut v[r * cols..r * cols + cols], levels);
                }
            }
            Mode::PerGroup(levels) => {
                for r in 0..seq {
                    for g in v[r * cols..r * cols + cols].chunks_mut(ACT_GROUP) {
                        quant_span(g, levels);
                    }
                }
            }
            Mode::Ternary(t) => {
                // The ladder fitter operates on a [rows, cols] matrix with per-group scales — the
                // same code path the weights use. Rotation is Never: the Hadamard is a weight-side
                // transform folded into activations at deploy time, and applying it here as well
                // would double-count it.
                let q = ste::salt_quantize_forward_grouped_geometric(
                    v,
                    seq,
                    cols,
                    t,
                    ACT_GROUP,
                    GRID,
                    RotationPolicy::Never,
                );
                v.copy_from_slice(&q);
            }
        }
    }
}

fn perplexity_aq(
    weights: &[Vec<f32>],
    a: &common::Arch,
    eval: &[u32],
    window: usize,
    mode: Mode,
) -> f64 {
    let mut nll = 0.0f64;
    let mut scored = 0usize;
    for chunk in eval.chunks(window) {
        if chunk.len() < 2 {
            continue;
        }
        let mut t = Tape::new();
        let wids: Vec<ValueId> = weights.iter().map(|w| t.leaf(w.clone())).collect();
        let out = forward_aq(&mut t, &wids, a, chunk, &mut |_kind, _li, v, seq, cols| {
            mode.apply(v, seq, cols);
        });
        let logits = t.value(out).to_vec();
        score_window(&logits, chunk, a.vocab, &mut nll, &mut scored);
    }
    assert!(scored > 0, "held-out set scored no positions");
    (nll / scored as f64).exp()
}

#[test]
#[ignore = "activation-precision sweep; needs SmolLM2-135M; run explicitly"]
fn activation_precision_sweep() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let t_w = env_usize("TRITIUM_AP_TW", 3);
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();

    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp,
            &arch,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (folded, arch) = fold(&fp, &shapes, &arch, &calib, 0.75);

    // fp weights, fp activations — the reference every ratio is taken against.
    let ppl_fp = perplexity_aq(&folded, &arch, &eval, EVAL_WINDOW, Mode::None);

    // Weights at a fixed T for every row of the table, so activation precision is the only variable.
    let qw: Vec<Vec<f32>> = folded
        .iter()
        .zip(&shapes)
        .map(|(w, &(r, c))| {
            ste::salt_quantize_forward_grouped_geometric(
                w,
                r,
                c,
                t_w,
                WEIGHT_GROUP,
                GRID,
                RotationPolicy::Always,
            )
        })
        .collect();

    println!(
        "SmolLM2-135M | fp {ppl_fp:.3} | fold α=0.75 | weights: ladder T={t_w}, g{WEIGHT_GROUP}\n\
         Activations quantized at q/k/v, gate/up, down and the tied head.\n\
         o_proj's input is NOT quantized (it lives inside attention()), so these are an UPPER\n\
         BOUND on a real deployment at the same setting — 1 of 7 projections is unpenalised.\n"
    );
    println!(
        "{:<34} {:>11} {:>10} {:>14}",
        "activation precision", "ppl", "× fp", "vs A-fp32"
    );
    println!("{}", "-".repeat(74));

    let mut baseline = f64::NAN;
    for mode in [
        Mode::None,
        Mode::PerRow(127.0),
        Mode::PerGroup(127.0),
        Mode::PerRow(7.0),
        Mode::PerGroup(7.0),
        Mode::Ternary(4),
        Mode::Ternary(3),
        Mode::Ternary(2),
        Mode::Ternary(1),
    ] {
        let ppl = perplexity_aq(&qw, &arch, &eval, EVAL_WINDOW, mode);
        if baseline.is_nan() {
            baseline = ppl;
        }
        println!(
            "{:<34} {ppl:>11.3} {:>9.3}× {:>13}",
            mode.label(),
            ppl / ppl_fp,
            if (ppl - baseline).abs() < 1e-9 {
                "—".to_owned()
            } else {
                format!("{:+.2}%", 100.0 * (ppl - baseline) / baseline)
            }
        );
    }

    println!(
        "\nA-fp32 is the basis every published SALT number was measured on; int8-per-row is what\n\
         the runtime actually executes. The gap between those two rows is the tax the published\n\
         claims do not carry. int8-per-group is the headroom measurement's proposal (5.41 dB\n\
         available on the FFN intermediate). The ternary rows are the multiply-free limit: at\n\
         T planes an activation costs T×1.58 bits, but activations are transient — this buys\n\
         COMPUTE, not bytes."
    );
}
