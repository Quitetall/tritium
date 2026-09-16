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
//! It quantizes **every** projection input: q/k/v, `o`, gate/up, down, and the tied head.
//! `o_proj`'s input used to be exempt because it lived inside `attention()`, which made every
//! number here an upper bound with one of seven projections per layer unpenalised.
//! `nn::attention_heads` splits that projection out, so the caveat is retired.
//!
//! What is still **not** covered: the attention interior itself — Q·Kᵀ, the softmax, and P·V run in
//! fp32 regardless. So "fully ternary" remains false of this harness, and is not claimed.
//!
//! Weights are held at a fixed `T` throughout so the only moving part is activation precision.
//!
//! # Measured 2026-09-16 — SmolLM2-135M, weights ladder T=3/g128, fp 22.675
//!
//! ```text
//! activation precision                       ppl       x fp      vs A-fp32
//! fp32 (published basis)                  24.278     1.071x             —
//! int8 per row (SHIPPING)                 24.542     1.082x        +1.09%
//! int8 per group g128                     24.356     1.074x        +0.32%
//! int4 per row                          1327.664    58.551x     +5368.51%
//! int4 per group g128                     74.884     3.302x      +208.44%
//! TERNARY 6 plane(s), 9.51 bits           24.289     1.071x        +0.04%
//! TERNARY 5 plane(s), 7.92 bits           24.355     1.074x        +0.32%
//! TERNARY 4 plane(s), 6.34 bits           25.126     1.108x        +3.49%
//! TERNARY 3 plane(s), 4.75 bits           33.470     1.476x       +37.86%
//! TERNARY 2 plane(s), 3.17 bits          350.623    15.463x     +1344.18%
//! TERNARY 1 plane(s), 1.58 bits       377273.282 16638.066x  +1553850.66%
//! per tap [4,4,4,4,4] (control)           25.126     1.108x        +3.49%
//! per tap [3,4,5,4,4]                     25.220     1.112x        +3.88%
//! per tap [3,3,6,4,4]                     25.821     1.139x        +6.35%
//! per tap [4,4,6,3,3]                     29.961     1.321x       +23.40%
//! per tap [5,5,2,4,4] (anti-control)      36.388     1.605x       +49.88%
//! ```
//!
//! **Ternary x5 costs 7.92 bits — less than int8 — and beats the shipping int8 quantizer 3.4x**
//! (+0.32% against +1.09%), matching int8-per-group exactly while needing no multiplies. That is
//! the L2 result and it is the one to act on. x6 at 9.51 bits is near-lossless (+0.04%) but no
//! longer cheaper than a byte.
//!
//! Ternary also dominates integers at sub-byte width by a wide margin: x4 at 6.34 bits is +3.49%
//! where int4-per-group at 4 bits is +208% — 60x better. The additive ladder transfers to
//! activations; a flat 4-bit grid does not.
//!
//! **Per-tap allocation loses, 4 arms out of 4.** Every arm spends 20 planes, so each is
//! bit-neutral against the uniform control — which it reproduces to the digit (25.126), proving the
//! plumbing changes nothing by itself.
//!
//! Two things make this a stronger negative than the weight-side campaign it echoes. First, the
//! headroom signal is demonstrably REAL and correctly signed: the anti-control, which starves
//! `down_in` (5.41 dB of measured headroom) to feed the taps with the least, is by far the worst
//! arm at +49.88%. Second, the standing excuse for the weight-side failure was proxy over-fitting
//! across 1.14M decisions from one noisy scalar each. **This problem has five decision units and an
//! exactly-measured budget, and allocation still loses.** That explanation does not survive.
//!
//! What is visible instead is the 9x asymmetry: removing a plane costs about 9x what adding one
//! saves, so `[4,4,6,3,3]` — which takes two planes from `o` and the head — pays +23.40% to buy
//! two planes on the tap that most wants them. The opportunity is self-extinguishing.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test activation_precision -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold, score_window};
use tritium_nn::ModelRunner;
use tritium_nn::calibrate::{Tap, forward_aq};
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
    /// Ternary planes allocated **per tap**: `[attn_in, ffn_in, down_in, o_proj_in, head]`.
    ///
    /// The A8 headroom measurement found the taps are not alike — 5.41 dB available on the FFN
    /// intermediate against ~1.7 dB elsewhere, a 3.2x spread — and `down_in` is the one activation
    /// in the block that is NOT post-RMSNorm. With only five decision units, allocation here is
    /// nothing like the 211-tensor weight-side problem that has failed every test: the search space
    /// is small enough to enumerate, and the budget accounting is exact.
    TernaryPerTap([usize; 5]),
}

/// Ternary bits per activation value at `T` planes. `log2(3) = 1.58496`.
fn ternary_bits(t: usize) -> f64 {
    t as f64 * 3.0f64.log2()
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
            Mode::Ternary(t) => format!("TERNARY {t} plane(s), {:.2} bits", ternary_bits(t)),
            Mode::TernaryPerTap(planes) => format!(
                "TERNARY per tap {planes:?}, {:.2} bits avg",
                planes.iter().map(|&t| ternary_bits(t)).sum::<f64>() / planes.len() as f64
            ),
        }
    }

    /// Planes this mode spends at one tap, for the per-tap arm; `None` for the uniform modes.
    fn planes_at(self, kind: Tap) -> Option<usize> {
        match self {
            Mode::TernaryPerTap(planes) => Some(
                planes[match kind {
                    Tap::AttnIn => 0,
                    Tap::FfnIn => 1,
                    Tap::DownIn => 2,
                    Tap::OProjIn => 3,
                    Tap::Head => 4,
                }],
            ),
            _ => None,
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
            // Resolved to a plane count by `perplexity_aq` before it gets here.
            Mode::TernaryPerTap(_) => unreachable!("per-tap modes are resolved by the caller"),
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
        let out = forward_aq(
            &mut t,
            &wids,
            a,
            chunk,
            &mut |kind, _li, v, seq, cols| match mode.planes_at(kind) {
                Some(t) => Mode::Ternary(t).apply(v, seq, cols),
                None => mode.apply(v, seq, cols),
            },
        );
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
         Every projection input is quantized: q/k/v, o, gate/up, down and the tied head.\n\
         o_proj's used to be exempt (it lived inside attention()); `attention_heads` splits it\n\
         out, so the old 'upper bound, 1 of 7 unpenalised' caveat no longer applies.\n"
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
        // L2. T=5 is 7.92 bits, CHEAPER than int8, and nobody has measured it. T=6 brackets it.
        Mode::Ternary(6),
        Mode::Ternary(5),
        Mode::Ternary(4),
        Mode::Ternary(3),
        Mode::Ternary(2),
        Mode::Ternary(1),
        // L3. Five decision units, order [attn_in, ffn_in, down_in, o_proj_in, head]. Every arm
        // below sums to 20 planes, so each is bit-neutral against uniform T=4 and the comparison is
        // a pure allocation question. `down_in` is the tap the headroom measurement says is starved.
        //
        // The degenerate control comes first: an all-4 allocation must reproduce uniform T=4
        // exactly, which is what proves the per-tap plumbing is not itself changing the answer.
        Mode::TernaryPerTap([4, 4, 4, 4, 4]),
        Mode::TernaryPerTap([3, 4, 5, 4, 4]),
        Mode::TernaryPerTap([3, 3, 6, 4, 4]),
        Mode::TernaryPerTap([4, 4, 6, 3, 3]),
        // The anti-control: spend the extra planes where the measured headroom is LOWEST. If this
        // ties the arms above, the per-tap signal is noise and the measurement says so out loud.
        Mode::TernaryPerTap([5, 5, 2, 4, 4]),
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
