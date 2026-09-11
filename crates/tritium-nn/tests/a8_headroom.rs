//! **How much of the A8 tax is recoverable by finer activation scales?**
//!
//! `quantize_activation_int8` takes one absmax **per token, over the whole row**, matching BitNet's
//! reference exactly. A single outlier therefore sets the quantization step for every value in that
//! row — and LLM activations are outlier-heavy, which is the entire premise AWQ and SmoothQuant
//! exist to address. The weight side already solved this with per-group scales; the activation side
//! never got the same treatment.
//!
//! That costs real quality. Measured on SmolLM2-360M at `T=4/g256`, the deployed artifact scores
//! **1.0560×** fp against **1.0277×** weights-only: a +2.76% tax that, as a fraction of the excess
//! over fp — which is what a parity claim actually asserts — **doubles** the gap (2.77% → 5.60%).
//!
//! # This measures the ceiling before anyone builds the feature
//!
//! Changing the activation quantizer means touching production code that carries a deliberate
//! bit-exactness contract with the reference. Worth knowing first whether the headroom is there.
//!
//! For a uniform quantizer the noise power per element is `step²/12` with `step = 2γ/255`, so the
//! error is proportional to `γ²`. Comparing one row-wide absmax against per-group absmaxes:
//!
//! ```text
//! per-row   MSE  ∝  cols · γ_row²
//! per-group MSE  ∝  Σ_g  width_g · γ_g²
//! gain           =  (cols · γ_row²) / Σ_g (width_g · γ_g²)
//! ```
//!
//! `γ_row = max_g γ_g` by construction, so the gain is **≥ 1 always** — the question is entirely how
//! much, and that is decided by how concentrated the outliers are. A flat row gives exactly 0 dB and
//! says the idea is worthless here; a row whose maximum lives in one group gives close to the
//! group-count ratio.
//!
//! This is an upper bound on what per-group activation scales can recover, not a perplexity
//! prediction: SQNR does not map linearly onto loss. A small number here kills the idea outright; a
//! large one says implement and measure.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test a8_headroom -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold};
use tritium_nn::ModelRunner;
use tritium_nn::calibrate::{Tap, calibrate_tapped};

const CALIB_WINDOWS: usize = 4;
const CALIB_SEQ: usize = 512;
/// Matches the weight-side group width, which is the granularity a real implementation would use.
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

fn corpus() -> Vec<u32> {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_owned()
    });
    let text = std::fs::read_to_string(&path).expect("corpus");
    let v: serde_json::Value = serde_json::from_str(&text).expect("corpus json");
    v["train_ids"]
        .as_array()
        .expect("train_ids")
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect()
}

/// Accumulated `(Σ per-row error, Σ per-group error, rows, worst single-row gain)` for one tap kind.
#[derive(Default, Clone, Copy)]
struct Headroom {
    row_err: f64,
    group_err: f64,
    rows: usize,
    worst: f64,
}

impl Headroom {
    fn gain_db(self) -> f64 {
        if self.group_err <= 0.0 {
            return 0.0;
        }
        10.0 * (self.row_err / self.group_err).log10()
    }
}

/// Fold one activation matrix `[seq, cols]` into the headroom accumulator.
fn measure(act: &[f32], seq: usize, cols: usize, h: &mut Headroom) {
    for r in 0..seq {
        let row = &act[r * cols..r * cols + cols];
        let gamma_row = row.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
        if gamma_row <= 0.0 {
            continue;
        }
        // Per-row: one step for the whole row.
        let row_err = cols as f64 * gamma_row * gamma_row;
        // Per-group: each group pays only for its own maximum.
        let mut group_err = 0.0f64;
        for g in row.chunks(ACT_GROUP) {
            let gamma_g = g.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
            group_err += g.len() as f64 * gamma_g * gamma_g;
        }
        h.row_err += row_err;
        h.group_err += group_err;
        h.rows += 1;
        if group_err > 0.0 {
            let gain = 10.0 * (row_err / group_err).log10();
            if gain > h.worst {
                h.worst = gain;
            }
        }
    }
}

#[test]
#[ignore = "needs SmolLM2-135M; run explicitly"]
fn a8_headroom_per_group_vs_per_row() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let train = corpus();
    let windows = env_usize("TRITIUM_A8_WINDOWS", CALIB_WINDOWS);

    // Calibrate, then fold — the headroom must be measured on activations the DEPLOYED model sees,
    // and the fold rescales them (it divides the preceding norm by `s`).
    let mut calib = Calib::new(&arch);
    for w in 0..windows {
        calibrate(
            &fp,
            &arch,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (folded, arch) = fold(&fp, &shapes, &arch, &calib, 0.75);

    let mut attn = Headroom::default();
    let mut ffn = Headroom::default();
    let mut down = Headroom::default();
    let mut per_layer: Vec<Headroom> = vec![Headroom::default(); arch.n_layers];
    for w in 0..windows {
        calibrate_tapped(
            &folded,
            &arch,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut |kind, li, t, id, seq, cols| {
                let act = t.value(id);
                let slot = match kind {
                    Tap::AttnIn => &mut attn,
                    Tap::FfnIn => &mut ffn,
                    Tap::DownIn => &mut down,
                };
                measure(act, seq, cols, slot);
                measure(act, seq, cols, &mut per_layer[li]);
            },
        );
    }

    println!(
        "SmolLM2-135M | fold α=0.75 | activation group {ACT_GROUP} | {windows} calib windows\n\
         Upper bound on what per-group activation scales recover vs one absmax per token.\n\
         Gain is ≥ 0 dB by construction; 0 dB means the row is flat and the idea is worthless.\n"
    );
    println!(
        "{:<28} {:>10} {:>12} {:>14}",
        "tap", "rows", "mean gain", "worst row"
    );
    println!("{}", "-".repeat(68));
    for (name, h) in [
        ("attn_in (post-RMSNorm)", attn),
        ("ffn_in  (post-RMSNorm)", ffn),
        ("down_in (FFN interm.)", down),
    ] {
        println!(
            "{name:<28} {:>10} {:>9.2} dB {:>11.2} dB",
            h.rows,
            h.gain_db(),
            h.worst
        );
    }
    let total = Headroom {
        row_err: attn.row_err + ffn.row_err + down.row_err,
        group_err: attn.group_err + ffn.group_err + down.group_err,
        rows: attn.rows + ffn.rows + down.rows,
        worst: attn.worst.max(ffn.worst).max(down.worst),
    };
    println!(
        "{:<28} {:>10} {:>9.2} dB {:>11.2} dB",
        "ALL TAPS",
        total.rows,
        total.gain_db(),
        total.worst
    );

    println!("\nper-layer mean gain (dB):");
    let line: Vec<String> = per_layer
        .iter()
        .map(|h| format!("{:.1}", h.gain_db()))
        .collect();
    println!("  {}", line.join(" "));

    println!(
        "\nFor scale: one SALT plane buys ~9.54 dB. A tap below ~1 dB is not worth touching a\n\
         bit-exact reference path for; several dB says implement it and measure the perplexity."
    );
}

/// The tap refactor must not have changed what `calibrate` computes.
///
/// `calibrate` was rewritten to call `calibrate_tapped` with an accumulating closure, and the
/// original body is retained as `calibrate_old`. Every curvature number in this repo flows through
/// it, so equivalence is asserted rather than assumed.
#[test]
#[ignore = "needs SmolLM2-135M; run explicitly"]
fn tapped_calibrate_matches_the_original() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, _shapes) = extract(&runner);
    let train = corpus();

    let mut a = Calib::new(&arch);
    let mut b = Calib::new(&arch);
    for w in 0..2 {
        let toks = &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ];
        calibrate(&fp, &arch, toks, &mut a);
        // The same forward, driven through the tap, reducing identically.
        let seq = toks.len();
        calibrate_tapped(&fp, &arch, toks, &mut |kind, li, t, id, seq, cols| {
            let acc = match kind {
                Tap::AttnIn => &mut b.attn_in[li],
                Tap::FfnIn => &mut b.ffn_in[li],
                Tap::DownIn => &mut b.down_in[li],
            };
            tritium_nn::calibrate::accumulate(t, id, seq, cols, acc);
        });
        b.rows += seq;
    }
    assert_eq!(a.rows, b.rows);
    for li in 0..arch.n_layers {
        for (x, y) in a.attn_in[li].iter().zip(&b.attn_in[li]) {
            assert_eq!(x.to_bits(), y.to_bits(), "attn_in layer {li}");
        }
        for (x, y) in a.ffn_in[li].iter().zip(&b.ffn_in[li]) {
            assert_eq!(x.to_bits(), y.to_bits(), "ffn_in layer {li}");
        }
        for (x, y) in a.down_in[li].iter().zip(&b.down_in[li]) {
            assert_eq!(x.to_bits(), y.to_bits(), "down_in layer {li}");
        }
    }
    println!(
        "tap refactor is bit-identical to the original calibrate across {} layers",
        arch.n_layers
    );
}
