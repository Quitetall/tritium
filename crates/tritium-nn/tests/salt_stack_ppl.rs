//! **The full SALT fitter stack on the real model**: Hadamard rotation (auto, per group) + finer scale
//! groups (256 = the deployed TQ2_0 block, 128 = PTQ convention) + iterative ternary fitting.
//!
//! Each improvement was validated in isolation on synthetic fixtures; this measures them stacked, on
//! SmolLM2-135M against held-out WikiText-2, with honest bpw. RTN int4 is carried as the external
//! reference — noting that it needs multipliers at inference while every SALT row here stays
//! multiply-free, so bits are a storage axis rather than the figure of merit.
//!
//! `#[ignore]`d; run:
//! ```text
//! TRITIUM_CORPUS=<corpus.json> cargo test -p tritium-nn --release \
//!   --test salt_stack_ppl -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{extract, perplexity_windowed};
use tritium_nn::ModelRunner;
use tritium_train::ops::ste::{self, RotationPolicy};

const EVAL_WINDOW: usize = 512;
const ITERS: usize = 5;

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

fn rtn4(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    const G: usize = 128;
    let levels = 15.0f32;
    let mut out = vec![0.0f32; w.len()];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (gs, gd) in row.chunks(G).zip(dst.chunks_mut(G)) {
            let lo = gs.iter().copied().fold(f32::INFINITY, f32::min);
            let hi = gs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let s = (hi - lo) / levels;
            if !(s > 0.0) {
                gd.fill(lo);
                continue;
            }
            let z = (-lo / s).round();
            for (d, &v) in gd.iter_mut().zip(gs) {
                *d = (((v / s).round() + z).clamp(0.0, levels) - z) * s;
            }
        }
    }
    out
}

#[test]
#[ignore = "slow full-stack PTQ sweep; needs SmolLM2-135M; run explicitly"]
fn salt_full_stack_quality() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let eval = eval_ids();
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    println!("fp reference {ppl_fp:.3} ppl\n");
    println!("{:<40} {:>8} {:>14} {:>11}", "configuration", "bpw", "ppl", "× fp");
    println!("{}", "-".repeat(78));

    let rtn: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(w, &(n, k))| rtn4(w, n, k))
        .collect();
    let p = perplexity_windowed(&rtn, &arch, &eval, EVAL_WINDOW);
    println!(
        "{:<40} {:>8.2} {:>14.3} {:>10.2}×   (needs multipliers)",
        "RTN int4 g128 [reference]", 4.25, p, p / ppl_fp
    );

    for t in 1..=3usize {
        // The original: per-row scales, greedy AbsMean, no rotation.
        let base: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| ste::salt_quantize_forward(w, n, k, t))
            .collect();
        let p = perplexity_windowed(&base, &arch, &eval, EVAL_WINDOW);
        println!(
            "{:<40} {:>8.2} {:>14.3} {:>10.2}×",
            format!("T={t} per-row, greedy [original]"),
            ste::ternary_bits_per_weight(t, 576),
            p,
            p / ppl_fp
        );

        for group in [256usize, 128] {
            for rotation in [RotationPolicy::Never, RotationPolicy::Auto] {
                let q: Vec<Vec<f32>> = fp
                    .iter()
                    .zip(&shapes)
                    .map(|(w, &(n, k))| {
                        ste::salt_quantize_forward_grouped(w, n, k, t, group, ITERS, rotation)
                    })
                    .collect();
                let p = perplexity_windowed(&q, &arch, &eval, EVAL_WINDOW);
                // Auto stores one bit per group to record the rotation choice.
                let bpw = ste::ternary_bits_per_weight(t, group)
                    + if rotation == RotationPolicy::Auto {
                        1.0 / group as f64
                    } else {
                        0.0
                    };
                let tag = if rotation == RotationPolicy::Auto {
                    "+ITF +Hadamard(auto)"
                } else {
                    "+ITF"
                };
                println!(
                    "{:<40} {:>8.2} {:>14.3} {:>10.2}×",
                    format!("T={t} g{group} {tag}"),
                    bpw,
                    p,
                    p / ppl_fp
                );
            }
        }
    }
    println!(
        "\nEvery SALT row is multiply-free at inference; the RTN reference is not. Compare within \
         the SALT rows to see what the fitter stack buys, and against RTN only with that caveat."
    );
}
