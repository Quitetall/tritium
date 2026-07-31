//! **Why is SALT's PTQ catastrophic?** An ablation separating the candidate causes, after the first
//! external baseline showed plain RTN int4 (4.25 bpw, no training) reaching 1.54× fp while SALT ITF
//! T=3 (6.19 bpw) sat at 7822× fp.
//!
//! Two suspects, both design choices rather than bugs:
//!
//! 1. **The tied token embedding is ternarized.** `extract()` puts `token_embd` at index 0 and the
//!    model ties it to the LM head, so every experiment so far ternarized the *output* layer — 28M of
//!    135M parameters (21%), and the one place errors corrupt logits directly. Most ternary work keeps
//!    embeddings/head at higher precision; the SOTA survey flags "fully ternary incl. embeddings and
//!    LM head" as a *notable* property of Ternary Bonsai, i.e. not the default.
//! 2. **Scale granularity.** SALT fits one AbsMean scale per output ROW — 576 weights (attention) or
//!    1536 (FFN) — while its own deployed TQ2_0 format uses 256-trit blocks and the RTN baseline uses
//!    group 128. SALT is quantizing 2.25–6× coarser than the format it ships into.
//!
//! bpw is accounted honestly per configuration: ternary planes cost `2 + 16/G` bits per weight per
//! plane, an int8-kept embedding costs `8 + 32/128`, and the reported figure is the parameter-weighted
//! average over the whole model.
//!
//! `#[ignore]`d; run:
//! ```text
//! TRITIUM_CORPUS=<corpus.json> cargo test -p tritium-nn --release \
//!   --test salt_ablation_ppl -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{extract, perplexity_windowed};
use tritium_nn::ModelRunner;
use tritium_train::ops::ste;

const EVAL_WINDOW: usize = 512;
const RTN_GROUP: usize = 128;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

fn eval_ids() -> Vec<u32> {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_string()
    });
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("corpus json")).expect("parse");
    j["eval_ids"]
        .as_array()
        .expect("eval_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect()
}

/// Group-wise asymmetric RTN (the same quantizer as the baseline test).
fn rtn(w: &[f32], rows: usize, cols: usize, bits: u32) -> Vec<f32> {
    let levels = ((1u32 << bits) - 1) as f32;
    let mut out = vec![0.0f32; w.len()];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (gs, gd) in row.chunks(RTN_GROUP).zip(dst.chunks_mut(RTN_GROUP)) {
            let lo = gs.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = gs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = (hi - lo) / levels;
            if scale <= 0.0 || scale.is_nan() {
                gd.fill(lo);
                continue;
            }
            let zero = (-lo / scale).round();
            for (d, &s) in gd.iter_mut().zip(gs) {
                *d = (((s / scale).round() + zero).clamp(0.0, levels) - zero) * scale;
            }
        }
    }
    out
}

/// SALT with an explicit scale **group size**: each row is split into `g`-wide blocks (ragged tail
/// allowed) and each block is fitted independently, instead of one scale for the whole row.
fn salt_grouped(w: &[f32], rows: usize, cols: usize, t: usize, g: usize, iters: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w.len()];
    for r in 0..rows {
        let src = &w[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (bs, bd) in src.chunks(g).zip(dst.chunks_mut(g)) {
            let q = ste::salt_quantize_forward_itf(bs, 1, bs.len(), t, iters);
            bd.copy_from_slice(&q);
        }
    }
    out
}

/// Ternary bits per weight at group size `g`: 2 bits per trit plus one f16 scale per group, per plane.
fn ternary_bpw(t: usize, g: usize) -> f64 {
    t as f64 * (2.0 + 16.0 / g as f64)
}

#[test]
#[ignore = "slow SALT ablation; needs SmolLM2-135M; run explicitly"]
fn why_is_salt_ptq_catastrophic() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let iters = 5usize;
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let eval = eval_ids();
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);

    let embed_n = fp[0].len();
    let total: usize = fp.iter().map(Vec::len).sum();
    println!(
        "fp {ppl_fp:.3} ppl | tied embedding is {embed_n} of {total} weights ({:.1}% of the model)\n",
        embed_n as f64 / total as f64 * 100.0
    );
    println!(
        "{:<44} {:>8} {:>13} {:>9}",
        "configuration", "bpw", "ppl", "× fp"
    );
    println!("{}", "-".repeat(78));

    let report = |label: String, bpw: f64, q: &[Vec<f32>]| {
        let ppl = perplexity_windowed(q, &arch, &eval, EVAL_WINDOW);
        println!("{label:<44} {bpw:>8.2} {ppl:>13.3} {:>8.2}×", ppl / ppl_fp);
    };

    // Reference: the baseline that currently wins.
    let rtn4: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(w, &(n, k))| rtn(w, n, k, 4))
        .collect();
    report(
        "RTN int4 g128 (baseline)".into(),
        4.0 + 32.0 / RTN_GROUP as f64,
        &rtn4,
    );

    // Weighted bpw when the embedding is kept int8 and the rest is ternary at group `g`.
    let mixed_bpw = |t: usize, g: usize| -> f64 {
        let emb_bits = (8.0 + 32.0 / RTN_GROUP as f64) * embed_n as f64;
        let rest_bits = ternary_bpw(t, g) * (total - embed_n) as f64;
        (emb_bits + rest_bits) / total as f64
    };

    for t in [2usize, 3] {
        // (a) as-is: everything ternary, per-row scales — what every experiment has done.
        let per_row: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| ste::salt_quantize_forward_itf(w, n, k, t, iters))
            .collect();
        report(
            format!("SALT ITF T={t}, all ternary, per-row scale"),
            ternary_bpw(t, 576), // ~the row width; the exact figure barely moves
            &per_row,
        );

        // (b) granularity only: same coverage, 128-wide scales.
        let g128: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| salt_grouped(w, n, k, t, 128, iters))
            .collect();
        report(
            format!("SALT ITF T={t}, all ternary, g128 scale"),
            ternary_bpw(t, 128),
            &g128,
        );

        // (c) embedding kept int8, rest ternary per-row.
        let mut emb_row = per_row.clone();
        emb_row[0] = rtn(&fp[0], shapes[0].0, shapes[0].1, 8);
        report(
            format!("SALT ITF T={t}, int8 embed, per-row scale"),
            mixed_bpw(t, 576),
            &emb_row,
        );

        // (d) both fixes: embedding int8 + 128-wide ternary scales.
        let mut emb_g128 = g128.clone();
        emb_g128[0] = rtn(&fp[0], shapes[0].0, shapes[0].1, 8);
        report(
            format!("SALT ITF T={t}, int8 embed, g128 scale"),
            mixed_bpw(t, 128),
            &emb_g128,
        );
    }

    println!(
        "\nRead: (b) isolates scale granularity, (c) isolates ternarizing the tied embedding/LM head, \
         (d) applies both. Compare against RTN int4 at matched-or-fewer bits — that is the bar SALT \
         has to clear before any SOTA claim."
    );
}
