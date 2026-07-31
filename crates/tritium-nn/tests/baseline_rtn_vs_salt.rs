//! **The comparison SALT has never had**: a bits-per-weight vs perplexity curve against a real,
//! external baseline, measured in the *same* harness so there are no eval confounds.
//!
//! Every SALT number so far has been compared only against Tritium's own naive PTQ (greedy AbsMean,
//! no error feedback). Beating that by 22 000× is not evidence of anything external — it is a weak
//! baseline. The honest question is whether SALT's quality-per-byte beats ordinary integer
//! quantization, so this scores both on one axis:
//!
//! - **RTN** (round-to-nearest, group-wise asymmetric int-`b`): the standard baseline that GPTQ/AWQ
//!   improve upon. `bpw = b + 32/group` (f16 scale + f16 zero per group).
//! - **SALT** `T` planes: `bpw = T · 2.0625` packed as TQ2_0 (66 B per 256 trits).
//!
//! Both are pure PTQ (no training, no calibration), so this is an apples-to-apples fitter comparison.
//! ITF is included because it is SALT's best available fitter.
//!
//! `#[ignore]`d; run:
//! ```text
//! TRITIUM_CORPUS=<corpus.json> cargo test -p tritium-nn --release \
//!   --test baseline_rtn_vs_salt -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{extract, perplexity_windowed};
use tritium_nn::ModelRunner;
use tritium_train::ops::ste;

const EVAL_WINDOW: usize = 512;
/// Group size for RTN scales — 128 is the standard used by GPTQ/AWQ reporting.
const GROUP: usize = 128;

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

/// Group-wise **asymmetric** round-to-nearest integer quantization — the ordinary `int-b` weight
/// quantizer. Each contiguous run of `GROUP` weights within a row gets its own `(scale, zero)`, the
/// weight is mapped to `[0, 2^b - 1]`, and the returned buffer is the dequantized reconstruction.
fn rtn_quantize(w: &[f32], rows: usize, cols: usize, bits: u32) -> Vec<f32> {
    let levels = ((1u32 << bits) - 1) as f32;
    let mut out = vec![0.0f32; w.len()];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let dst = &mut out[r * cols..(r + 1) * cols];
        for (g_src, g_dst) in row.chunks(GROUP).zip(dst.chunks_mut(GROUP)) {
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
    f64::from(bits) + 32.0 / GROUP as f64
}

/// bpw for `T` SALT planes packed as TQ2_0 (66 bytes per 256 trits).
fn salt_bpw(t: usize) -> f64 {
    t as f64 * (66.0 * 8.0 / 256.0)
}

#[test]
#[ignore = "slow PTQ baseline comparison; needs SmolLM2-135M; run explicitly"]
fn salt_vs_rtn_quality_per_byte() {
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
    println!(
        "fp reference: {ppl_fp:.3} ppl ({} held-out tokens, {GROUP}-wide RTN groups)\n",
        eval.len()
    );
    println!(
        "{:<22} {:>7}  {:>14}  {:>10}",
        "method", "bpw", "ppl", "× fp"
    );
    println!("{}", "-".repeat(58));

    let mut rows: Vec<(String, f64, f64)> = Vec::new();

    // Baseline: ordinary integer quantization at 2/3/4/8 bits.
    for bits in [2u32, 3, 4, 8] {
        let q: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(n, k))| rtn_quantize(w, n, k, bits))
            .collect();
        let ppl = perplexity_windowed(&q, &arch, &eval, EVAL_WINDOW);
        let bpw = rtn_bpw(bits);
        println!(
            "{:<22} {:>7.2}  {:>14.3}  {:>9.2}×",
            format!("RTN int{bits} g{GROUP}"),
            bpw,
            ppl,
            ppl / ppl_fp
        );
        rows.push((format!("RTN int{bits}"), bpw, ppl));
    }

    // SALT, both fitters, at every plane count.
    for t in 1..=3 {
        for (label, q) in [
            (
                "SALT greedy",
                fp.iter()
                    .zip(&shapes)
                    .map(|(w, &(n, k))| ste::salt_quantize_forward(w, n, k, t))
                    .collect::<Vec<_>>(),
            ),
            (
                "SALT ITF",
                fp.iter()
                    .zip(&shapes)
                    .map(|(w, &(n, k))| ste::salt_quantize_forward_itf(w, n, k, t, 5))
                    .collect::<Vec<_>>(),
            ),
        ] {
            let ppl = perplexity_windowed(&q, &arch, &eval, EVAL_WINDOW);
            let bpw = salt_bpw(t);
            println!(
                "{:<22} {:>7.2}  {:>14.3}  {:>9.2}×",
                format!("{label} T={t}"),
                bpw,
                ppl,
                ppl / ppl_fp
            );
            rows.push((format!("{label} T={t}"), bpw, ppl));
        }
    }

    // The verdict: at matched-or-fewer bits, does anything SALT produces beat RTN?
    println!("\n--- quality-per-byte verdict ---");
    let rtn: Vec<_> = rows
        .iter()
        .filter(|(n, ..)| n.starts_with("RTN"))
        .cloned()
        .collect();
    for (name, bpw, ppl) in rows.iter().filter(|(n, ..)| n.starts_with("SALT")) {
        // The cheapest RTN setting that is no more expensive than this SALT setting.
        let rival = rtn
            .iter()
            .filter(|(_, b, _)| *b <= *bpw + 1e-9)
            .min_by(|a, b| a.2.partial_cmp(&b.2).expect("finite ppl"));
        match rival {
            Some((rn, rb, rp)) if *rp < *ppl => println!(
                "  {name} ({bpw:.2} bpw, {ppl:.1} ppl) LOSES to {rn} ({rb:.2} bpw, {rp:.1} ppl)"
            ),
            Some((rn, rb, rp)) => println!(
                "  {name} ({bpw:.2} bpw, {ppl:.1} ppl) beats {rn} ({rb:.2} bpw, {rp:.1} ppl)"
            ),
            None => println!("  {name} ({bpw:.2} bpw) — no RTN setting at or below this budget"),
        }
    }
    println!(
        "\nNOTE: pure PTQ, no distillation and no calibration — this measures the FITTER, not the \
         full SALT pipeline. Uniform plane counts, not SALT V2's curvature allocation."
    );
}
