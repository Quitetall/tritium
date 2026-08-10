//! **The comparison that decides whether ternary is interesting**: bits-per-weight vs perplexity
//! against integer quantization, measured in the *same* harness so there are no eval confounds.
//!
//! Beating Tritium's own naive PTQ proves nothing. Beating **fp16** proves less than it sounds —
//! nobody ships fp16 weights. The baseline that matters is ordinary group-wise integer
//! quantization, because that is what a deployed model is actually stored as.
//!
//! - **RTN** (round-to-nearest, group-wise asymmetric int-`b`): the standard baseline GPTQ/AWQ
//!   improve upon. `bpw = b + 32/group` (f16 scale + f16 zero per group).
//! - **SALT ladder** `T` planes: balanced-ternary geometric ladder, one f16 anchor per group plus a
//!   rotation bit, packed B3 (5 trits/byte) — `1.625` bpw per plane.
//! - **SALT ITF** `T` planes: the previous free-scale fitter, kept as the "old fitter" control.
//!
//! # Fairness: the fold goes on BOTH sides
//!
//! Every published ladder number is measured under the AWQ-style salience fold (α = 0.75). Scoring
//! a folded ladder against *unfolded* RTN would compare our best configuration against the
//! baseline's weakest, and the fold — not ternary — could be doing the work. So RTN is run **both
//! ways**: unfolded (plain RTN) and folded (activation-aware, i.e. AWQ-like). The headline
//! comparison is folded-vs-folded; the unfolded RTN row is kept so the fold's own contribution is
//! visible rather than hidden inside our column.
//!
//! `#[ignore]`d; run:
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//! TRITIUM_MODEL_DIR=$HOME/.cache/tritium-models/smollm2-360m \
//!   cargo test -p tritium-nn --release --test baseline_rtn_vs_salt -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, Evaluator, calibrate, extract, fold};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_nn::ModelRunner;
use tritium_train::ops::ste::{self, RotationPolicy};

const EVAL_WINDOW: usize = 512;
/// Default group size for both RTN scales and SALT groups — 128 is the standard GPTQ/AWQ
/// reporting width. Override with `TRITIUM_GROUP`; unset reproduces every published number.
const GROUP_DEFAULT: usize = 128;
/// ITF alternations for the legacy free-scale control.
const ITERS: usize = 5;
/// Δ candidates per group for the ladder.
const GRID: usize = 16;
/// Default salience-fold strength — the value every published SALT number was measured under.
/// Override with `TRITIUM_FOLD_ALPHA`. The optimum is known to shift DOWN with model size
/// (0.75 -> 0.50 observed), so a fixed value is very likely wrong away from 135M.
const FOLD_ALPHA_DEFAULT: f64 = 0.75;

/// Group size for this run. Both the RTN scales and the SALT groups use it, so a sweep moves the
/// two baselines together and the comparison stays like-for-like.
fn group() -> usize {
    std::env::var("TRITIUM_GROUP")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&g: &usize| g >= 8)
        .unwrap_or(GROUP_DEFAULT)
}

/// Salience-fold alpha for this run.
fn fold_alpha() -> f64 {
    std::env::var("TRITIUM_FOLD_ALPHA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(FOLD_ALPHA_DEFAULT)
}
const CALIB_WINDOWS: usize = 8;
const CALIB_SEQ: usize = 512;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = std::env::var("TRITIUM_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m"));
    PathBuf::from(dir)
}

/// `(train_ids, eval_ids)` — training ids feed calibration, eval ids score perplexity.
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

/// Group-wise **asymmetric** round-to-nearest integer quantization — the ordinary `int-b` weight
/// quantizer. Each contiguous run of `group()` weights within a row gets its own `(scale, zero)`, the
/// weight is mapped to `[0, 2^b - 1]`, and the returned buffer is the dequantized reconstruction.
fn rtn_quantize(w: &[f32], rows: usize, cols: usize, bits: u32) -> Vec<f32> {
    let levels = ((1u32 << bits) - 1) as f32;
    let mut out = vec![0.0f32; w.len()];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (g_src, g_dst) in row.chunks(group()).zip(dst.chunks_mut(group())) {
            let lo = g_src.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = g_src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = (hi - lo) / levels;
            if scale <= 0.0 || scale.is_nan() {
                g_dst.fill(lo); // constant group: exactly representable
                continue;
            }
            // Asymmetric: the zero-point lands `lo` on integer 0.
            let zero = (-lo / scale).round();
            for (d, &s) in g_dst.iter_mut().zip(g_src) {
                let q = ((s / scale).round() + zero).clamp(0.0, levels);
                *d = (q - zero) * scale;
            }
        }
    }
    out
}

/// bpw for group-wise int-`b`: the codes plus one f16 scale and one f16 zero per group.
fn rtn_bpw(bits: u32) -> f64 {
    f64::from(bits) + 32.0 / group() as f64
}

/// bpw for `T` ladder planes: B3-packed trits, one f16 anchor per group, plus the rotation bit.
fn ladder_bpw(t: usize) -> f64 {
    ste::ternary_bits_per_weight_geometric(t, group(), SaltV2Codec::B3, group())
        + 1.0 / group() as f64
}

/// bpw for `T` free-scale (ITF) planes: B3-packed trits, one f16 scale **per plane** per group.
fn itf_bpw(t: usize) -> f64 {
    ste::ternary_bits_per_weight_codec(t, group(), SaltV2Codec::B3, group()) + 1.0 / group() as f64
}

/// One scored configuration.
struct Row {
    name: String,
    bpw: f64,
    ppl: f64,
    /// `true` for the ternary arms — the ones whose multiply-free execution is the real claim.
    ternary: bool,
}

#[test]
#[ignore = "slow PTQ baseline comparison; set TRITIUM_MODEL_DIR + TRITIUM_CORPUS; run explicitly"]
fn salt_vs_rtn_quality_per_byte() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() && !dir.join("model.safetensors.index.json").exists()
    {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();

    let ev = Evaluator::from_env();
    let g = group();
    let alpha = fold_alpha();

    // The fp reference is scored on the ORIGINAL weights, before the fold reparameterizes them.
    let ppl_fp = ev.ppl(&fp, &arch, &eval, EVAL_WINDOW);

    // Calibrate once; the folded pair feeds every folded arm so the fold is identical across them.
    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp,
            &arch,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp_folded, arch_folded) = fold(&fp, &shapes, &arch, &calib, alpha);

    println!(
        "{} | fp {ppl_fp:.3} ppl | {} held-out tokens | g{g} | B3 bpw | fold α={alpha} | eval={}\n\
         RTN is scored BOTH unfolded and folded; folded-vs-folded is the honest comparison, and \
         the unfolded row shows what the fold itself is worth.\n",
        dir.file_name().unwrap_or_default().to_string_lossy(),
        eval.len(),
        ev.label(),
    );
    println!(
        "{:<30} {:>7}  {:>11}  {:>9}",
        "method", "bpw", "ppl", "× fp"
    );
    println!("{}", "-".repeat(62));

    let mut rows: Vec<Row> = Vec::new();
    let mut score = |name: String, bpw: f64, ppl: f64, ternary: bool| {
        println!("{name:<30} {bpw:>7.3}  {ppl:>11.3}  {:>8.3}×", ppl / ppl_fp);
        rows.push(Row {
            name,
            bpw,
            ppl,
            ternary,
        });
    };

    // ── Baseline: ordinary integer quantization, unfolded then folded ────────────────────────────
    // int5/int6 are NOT optional padding: at g128 they land at 5.25 and 6.25 bpw, straddling the
    // ladder's T=3 (5.008) and T=4 (6.633). Without them the "best RTN at or below budget" rule
    // silently compares those arms against int4 while they spend 18%/56% more bytes, which flatters
    // ternary. int6 is the row that can dominate T=4 outright.
    for bits in [2u32, 3, 4, 5, 6, 8] {
        let q: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| rtn_quantize(w, n, k, bits))
            .collect();
        let ppl = ev.ppl(&q, &arch, &eval, EVAL_WINDOW);
        score(format!("RTN int{bits} g{g}"), rtn_bpw(bits), ppl, false);
    }
    for bits in [2u32, 3, 4, 5, 6, 8] {
        let q: Vec<Vec<f32>> = fp_folded
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| rtn_quantize(w, n, k, bits))
            .collect();
        let ppl = ev.ppl(&q, &arch_folded, &eval, EVAL_WINDOW);
        score(
            format!("RTN int{bits} g{g} +fold"),
            rtn_bpw(bits),
            ppl,
            false,
        );
    }

    // ── SALT: the shipped ladder, plus the legacy free-scale fitter as a control ─────────────────
    for t in 1..=4 {
        let q: Vec<Vec<f32>> = fp_folded
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| {
                ste::salt_quantize_forward_grouped_geometric(
                    w,
                    n,
                    k,
                    t,
                    group(),
                    GRID,
                    RotationPolicy::Always,
                )
            })
            .collect();
        let ppl = ev.ppl(&q, &arch_folded, &eval, EVAL_WINDOW);
        score(format!("SALT ladder T={t} +fold"), ladder_bpw(t), ppl, true);
    }
    for t in 1..=3 {
        let q: Vec<Vec<f32>> = fp_folded
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| {
                ste::salt_quantize_forward_grouped(w, n, k, t, group(), ITERS, RotationPolicy::Auto)
            })
            .collect();
        let ppl = ev.ppl(&q, &arch_folded, &eval, EVAL_WINDOW);
        score(format!("SALT ITF T={t} +fold"), itf_bpw(t), ppl, true);
    }

    // ── Verdict: at matched-or-fewer bits, does ternary beat integer? ────────────────────────────
    println!("\n--- quality-per-byte verdict (vs FOLDED RTN — like against like) ---");
    let rivals: Vec<&Row> = rows
        .iter()
        .filter(|r| r.name.starts_with("RTN") && r.name.ends_with("+fold"))
        .collect();
    let mut ternary_wins = 0usize;
    let mut ternary_total = 0usize;
    for r in rows.iter().filter(|r| r.ternary) {
        ternary_total += 1;
        // The best RTN setting that costs no more than this ternary setting.
        let rival = rivals
            .iter()
            .filter(|x| x.bpw <= r.bpw + 1e-9)
            .min_by(|a, b| a.ppl.partial_cmp(&b.ppl).expect("finite ppl"));
        match rival {
            Some(x) if x.ppl < r.ppl => println!(
                "  {} ({:.3} bpw, {:.2} ppl) LOSES to {} ({:.3} bpw, {:.2} ppl)",
                r.name, r.bpw, r.ppl, x.name, x.bpw, x.ppl
            ),
            Some(x) => {
                ternary_wins += 1;
                println!(
                    "  {} ({:.3} bpw, {:.2} ppl) beats {} ({:.3} bpw, {:.2} ppl)",
                    r.name, r.bpw, r.ppl, x.name, x.bpw, x.ppl
                );
            }
            None => println!(
                "  {} ({:.3} bpw) — no RTN setting at or below this budget",
                r.name, r.bpw
            ),
        }
    }
    println!("\n  ternary arms winning their budget: {ternary_wins}/{ternary_total}");

    // The stricter test. "Beats the best RTN at or below my budget" lets a ternary arm spend more
    // bytes than its rival and still be called a win. Pareto DOMINATION is the claim that survives
    // scrutiny: no cheaper-or-equal point is also better-or-equal. Report both directions, because
    // an arm being dominated is the result that would sink the thesis.
    println!("\n--- strict Pareto check (dominate = no worse on BOTH bpw and ppl) ---");
    let folded_rtn: Vec<&Row> = rows
        .iter()
        .filter(|r| r.name.starts_with("RTN") && r.name.ends_with("+fold"))
        .collect();
    let eps = 1e-9;
    for r in rows.iter().filter(|r| r.ternary) {
        let dominated_by: Vec<&str> = folded_rtn
            .iter()
            .filter(|x| x.bpw <= r.bpw + eps && x.ppl <= r.ppl + eps)
            .map(|x| x.name.as_str())
            .collect();
        let dominates: Vec<&str> = folded_rtn
            .iter()
            .filter(|x| r.bpw <= x.bpw + eps && r.ppl <= x.ppl + eps)
            .map(|x| x.name.as_str())
            .collect();
        if !dominated_by.is_empty() {
            println!("  {} is DOMINATED by {}", r.name, dominated_by.join(", "));
        } else if !dominates.is_empty() {
            println!("  {} DOMINATES {}", r.name, dominates.join(", "));
        } else {
            println!(
                "  {} — on the Pareto front (no domination either way)",
                r.name
            );
        }
    }

    println!(
        "\nNOTE: pure PTQ — no distillation. Uniform plane counts, not per-group allocation \
         (measured negative: lower weight SSE, worse ppl).\n\
         If ternary LOSES on quality-per-byte here, the honest claim is not \"ternary is smaller\" \
         but \"ternary matches integer quantization per byte AND executes multiply-free\" — a \
         speed/energy argument this harness does not measure."
    );
}
