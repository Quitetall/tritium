//! **E5 — allocate by MEASURED marginal loss, not by a weight-space proxy** (ADR 0041).
//!
//! Every sensitivity signal tested so far lives in weight space, and every one lost to uniform:
//! raw SSE, curvature-weighted SSE, at four fold strengths and three granularities — 21 arms, zero
//! wins. That leaves one question open, and it is the one that decides whether the negative result
//! is about *the proxy* or about *the problem*:
//!
//! > Is allocation losing because `H_g` is the wrong sensitivity, or because the loss is not
//! > separable across tensors at all?
//!
//! A proxy cannot answer it. This measures the real thing: for each tensor, drop **that tensor
//! alone** to `T_ref - 1` with everything else at `T_ref`, and evaluate held-out perplexity. The
//! rise is the true marginal cost of taking a plane from that tensor — no Hessian, no diagonal
//! approximation, no separability assumption in the *measurement*.
//!
//! Then allocate with those measurements as the sensitivity and compare at matched bits.
//!
//! * **If measured-sensitivity allocation also loses**, the line closes and the negative result
//!   gets much stronger: it says the loss is not separable, not that the proxy was bad. No choice
//!   of `H_g` rescues allocation, because the best possible `H_g` was just tried.
//! * **If it wins**, the proxy was the whole problem, and the granularity/floor results already in
//!   hand say where to look.
//!
//! # Cost, and why this is resumable
//!
//! One held-out evaluation per tensor — 211 for SmolLM2-135M (1 embedding + 30 layers x 7 slots).
//! The per-tensor quantize is cheap because only the changed tensor is re-quantized; the evaluation
//! dominates. Deltas are cached to JSON keyed by tensor index, so an interrupted sweep resumes
//! instead of restarting, and the cache is the artifact worth keeping even if the allocation
//! conclusion changes.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test salt_alloc_e5 -- --ignored --nocapture
//! ```
//!
//! `TRITIUM_E5_LIMIT=N` stops after N tensors (rate probe). `TRITIUM_E5_EVAL_TOKENS=N` measures the
//! marginal deltas on the first N held-out tokens; the final arms are ALWAYS scored on the full
//! split, because a subsample is fine for *ranking* tensors and not fine for reporting perplexity
//! (the first 8,192 tokens read ~22% high against all 32,768).

mod common;

use std::path::PathBuf;

use common::{Arch, Calib, calibrate, extract, fold, perplexity_windowed, smooth_scales};
use tritium_nn::ModelRunner;
use tritium_quantize::{AllocConfig, GroupCurve, allocate_with_curves};
use tritium_train::ops::ste::{self, RotationPolicy};

const EVAL_WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 8;
const CALIB_SEQ: usize = 512;
const GROUP: usize = 128;
const GRID: usize = 16;
const B3_BITS_PER_TRIT: f64 = 1.625;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn fold_alpha() -> f64 {
    std::env::var("TRITIUM_E5_ALPHA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.75)
}

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR")
            .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m")),
    )
}

fn cache_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(
        std::env::var("TRITIUM_E5_CACHE")
            .unwrap_or_else(|_| format!("{home}/.cache/tritium-corpora/e5_tensor_deltas.json")),
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

/// Post-fold effective input curvature per column — identical to the `salt_alloc_ppl` definition,
/// kept here so the two harnesses can be compared arm-for-arm without one importing the other.
fn column_curvature(a: &Arch, c: &Calib, shapes: &[(usize, usize)], alpha: f64) -> Vec<Vec<f64>> {
    let eff = |acc: &[f64], rows: usize| -> Vec<f64> {
        let s = smooth_scales(acc, rows, alpha);
        acc.iter()
            .zip(&s)
            .map(|(&d, &sj)| {
                let sj = f64::from(sj).max(1e-12);
                (d / rows as f64) / (sj * sj)
            })
            .collect()
    };
    let mut out: Vec<Vec<f64>> = shapes.iter().map(|_| Vec::new()).collect();
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        let attn = eff(&c.attn_in[li], c.rows);
        let ffn = eff(&c.ffn_in[li], c.rows);
        let down = eff(&c.down_in[li], c.rows);
        for k in 0..3 {
            out[base + k].clone_from(&attn);
        }
        out[base + 4].clone_from(&ffn);
        out[base + 5].clone_from(&ffn);
        out[base + 6].clone_from(&down);
    }
    let (sum, cnt) = out
        .iter()
        .flatten()
        .fold((0.0f64, 0usize), |(s, n), &v| (s + v, n + 1));
    let mean = if cnt > 0 { sum / cnt as f64 } else { 1.0 };
    for (o, &(_, k)) in out.iter_mut().zip(shapes) {
        if o.is_empty() {
            *o = vec![mean; k];
        }
    }
    out
}

fn ladder_curves(
    w: &[f32],
    rows: usize,
    cols: usize,
    curvature: &[f64],
    t_max: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<usize>) {
    let per_row = cols.div_ceil(GROUP);
    let n_groups = rows * per_row;
    let mut curves = vec![vec![0.0f64; t_max + 1]; n_groups];
    let mut sens = vec![0.0f64; n_groups];
    let mut sizes = vec![0usize; n_groups];
    for r in 0..rows {
        for b in 0..per_row {
            let lo = r * cols + b * GROUP;
            let hi = (lo + GROUP).min((r + 1) * cols);
            let g = r * per_row + b;
            sizes[g] = hi - lo;
            curves[g][0] = w[lo..hi].iter().map(|&v| f64::from(v) * f64::from(v)).sum();
            let c0 = b * GROUP;
            let c1 = (c0 + GROUP).min(cols);
            sens[g] = curvature[c0..c1].iter().sum::<f64>() / (c1 - c0) as f64;
        }
    }
    #[allow(clippy::needless_range_loop)]
    for t in 1..=t_max {
        let q = ste::salt_quantize_forward_grouped_geometric(
            w,
            rows,
            cols,
            t,
            GROUP,
            GRID,
            RotationPolicy::Always,
        );
        for r in 0..rows {
            for b in 0..per_row {
                let lo = r * cols + b * GROUP;
                let hi = (lo + GROUP).min((r + 1) * cols);
                curves[r * per_row + b][t] = q[lo..hi]
                    .iter()
                    .zip(&w[lo..hi])
                    .map(|(&a, &x)| f64::from(a - x) * f64::from(a - x))
                    .sum();
            }
        }
    }
    (curves, sens, sizes)
}

fn alloc_bpw(counts: &[u8], sizes: &[usize], plane_bits: f64) -> f64 {
    let total: usize = sizes.iter().sum();
    let trits: f64 = counts
        .iter()
        .zip(sizes)
        .map(|(&t, &s)| f64::from(t) * s as f64 * B3_BITS_PER_TRIT)
        .sum();
    (trits + (16.0 + 1.0 + plane_bits) * sizes.len() as f64) / total as f64
}

/// Quantize one tensor at a uniform plane count.
fn quantize_at(w: &[f32], rows: usize, cols: usize, t: usize) -> Vec<f32> {
    ste::salt_quantize_forward_grouped_geometric(w, rows, cols, t, GROUP, GRID, RotationPolicy::Always)
}

#[test]
#[ignore = "one held-out eval per tensor (211 for 135M); resumable; run explicitly"]
fn measured_marginal_loss_allocation_vs_uniform() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let t_ref = env_usize("TRITIUM_E5_T", 3);
    let t_max = env_usize("TRITIUM_E5_TMAX", 6).max(t_ref);
    let plane_bits = ((t_max + 1) as f64).log2().ceil();
    let limit = env_usize("TRITIUM_E5_LIMIT", usize::MAX);

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();

    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(&fp, &arch, &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ], &mut calib);
    }
    let alpha = fold_alpha();
    let curvature = column_curvature(&arch, &calib, &shapes, alpha);
    let (fp, arch) = fold(&fp, &shapes, &arch, &calib, alpha);
    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);

    // Subsample only for the 211 marginal probes; the reported arms use the full split.
    let probe_tokens = env_usize("TRITIUM_E5_EVAL_TOKENS", eval.len()).min(eval.len());
    let probe_eval = &eval[..probe_tokens];

    let n_tensors = fp.len();
    println!(
        "SmolLM2-135M | fp {ppl_fp:.3} | fold α={alpha} | g{GROUP} | ladder (always rot)\n\
         E5: {n_tensors} tensors, drop each alone to T={} with the rest at T={t_ref}\n\
         marginal probes on {probe_tokens} tokens; arms scored on all {} tokens\n",
        t_ref - 1,
        eval.len()
    );

    // Baseline: every tensor at T_ref.
    let uniform: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(w, &(n, k))| quantize_at(w, n, k, t_ref))
        .collect();
    let ppl_u_probe = perplexity_windowed(&uniform, &arch, probe_eval, EVAL_WINDOW);
    let ppl_u_full = if probe_tokens == eval.len() {
        ppl_u_probe
    } else {
        perplexity_windowed(&uniform, &arch, &eval, EVAL_WINDOW)
    };
    println!("uniform T={t_ref}: probe ppl {ppl_u_probe:.4} | full ppl {ppl_u_full:.4} ({:.3}× fp)\n", ppl_u_full / ppl_fp);

    // The tensor at T_ref-1, precomputed once per tensor and swapped in alone.
    let mut deltas: Vec<f64> = load_cache(n_tensors, probe_tokens, t_ref);
    let start = std::time::Instant::now();
    let mut measured = 0usize;
    for i in 0..n_tensors.min(limit) {
        if deltas[i].is_finite() {
            continue;
        }
        let (n, k) = shapes[i];
        let mut probe = uniform.clone();
        probe[i] = quantize_at(&fp[i], n, k, t_ref - 1);
        let ppl = perplexity_windowed(&probe, &arch, probe_eval, EVAL_WINDOW);
        deltas[i] = ppl - ppl_u_probe;
        measured += 1;
        let rate = start.elapsed().as_secs_f64() / measured as f64;
        println!(
            "  [{i:>3}/{n_tensors}] Δppl {:+.5}  ({:.1}s/tensor, ~{:.0} min left)",
            deltas[i],
            rate,
            rate * (n_tensors.saturating_sub(i + 1)) as f64 / 60.0
        );
        save_cache(&deltas, probe_tokens, t_ref);
    }
    if limit < n_tensors {
        println!("\nTRITIUM_E5_LIMIT={limit} — stopping before the allocation arm.");
        return;
    }

    // Allocate with the MEASURED marginal cost as the sensitivity, at tensor granularity (one
    // measurement per tensor is exactly one sensitivity per tensor).
    let mut all_curves = Vec::with_capacity(n_tensors);
    let mut all_sizes = Vec::with_capacity(n_tensors);
    for ((w, &(rows, cols)), curv) in fp.iter().zip(&shapes).zip(&curvature) {
        let (c, _s, z) = ladder_curves(w, rows, cols, curv, t_max);
        all_curves.push(c);
        all_sizes.push(z);
    }
    let flat_sizes: Vec<usize> = all_sizes.iter().flatten().copied().collect();
    let total_weights: usize = flat_sizes.iter().sum();

    let mut unit_curve = vec![vec![0.0f64; t_max + 1]; n_tensors];
    let mut unit_weights = vec![0usize; n_tensors];
    for (i, (cs, zs)) in all_curves.iter().zip(&all_sizes).enumerate() {
        for (c, &z) in cs.iter().zip(zs) {
            for t in 0..=t_max {
                unit_curve[i][t] += c[t];
            }
            unit_weights[i] += z;
        }
    }

    println!("\n{:<28} {:>9} {:>12} {:>11} {:>9}", "arm", "bpw", "mean T", "ppl", "× fp");
    println!("{}", "-".repeat(76));
    println!(
        "{:<28} {:>9.3} {:>12.3} {ppl_u_full:>11.3} {:>8.3}×",
        format!("uniform T={t_ref}"),
        alloc_bpw(&vec![t_ref as u8; flat_sizes.len()], &flat_sizes, plane_bits),
        t_ref as f64,
        ppl_u_full / ppl_fp
    );

    // Negative deltas mean removing a plane HELPED that tensor; clamp at 0 so they are treated as
    // "no evidence of sensitivity" rather than as evidence to starve the tensor further.
    let helped = deltas.iter().filter(|d| **d < 0.0).count();
    let sens: Vec<f64> = deltas.iter().map(|d| d.max(0.0)).collect();
    let curved: Vec<GroupCurve<'_>> = (0..n_tensors)
        .map(|i| GroupCurve {
            curve: &unit_curve[i],
            weights: unit_weights[i],
            sensitivity: sens[i],
        })
        .collect();
    let cfg = AllocConfig::from_bpw(
        tritium_quantize::TRIT_BITS * t_ref as f64,
        total_weights,
        env_usize("TRITIUM_E5_TMIN", 1),
        t_max,
    );
    let alloc = allocate_with_curves(&curved, &cfg).expect("allocate");
    let mut counts_flat: Vec<u8> = Vec::with_capacity(flat_sizes.len());
    for (i, zs) in all_sizes.iter().enumerate() {
        let t = u8::try_from(alloc.plane_counts[i]).expect("plane count fits u8");
        counts_flat.extend(std::iter::repeat_n(t, zs.len()));
    }
    let q: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .enumerate()
        .map(|(i, (w, &(n, k)))| quantize_at(w, n, k, alloc.plane_counts[i]))
        .collect();
    let ppl_m = perplexity_windowed(&q, &arch, &eval, EVAL_WINDOW);
    let mean_t: f64 = counts_flat
        .iter()
        .zip(&flat_sizes)
        .map(|(&t, &s)| f64::from(t) * s as f64)
        .sum::<f64>()
        / total_weights as f64;
    println!(
        "{:<28} {:>9.3} {:>12.3} {ppl_m:>11.3} {:>8.3}×   ({:+.2}% vs uniform)",
        "allocated, MEASURED Δppl",
        alloc_bpw(&counts_flat, &flat_sizes, plane_bits),
        mean_t,
        ppl_m / ppl_fp,
        100.0 * (ppl_m - ppl_u_full) / ppl_u_full
    );
    println!("     per-tensor T: {:?}", alloc.plane_counts);
    println!(
        "     {helped}/{n_tensors} tensors IMPROVED when a plane was removed (clamped to 0 sensitivity)"
    );
    println!(
        "\nIf this arm also loses, the proxy was never the problem: the best obtainable sensitivity\n\
         is a direct measurement of the objective, and allocation still cannot beat uniform. That\n\
         would say the loss is not separable across tensors — no choice of H_g rescues it."
    );
}

fn load_cache(n: usize, probe_tokens: usize, t_ref: usize) -> Vec<f64> {
    let path = cache_path();
    let fresh = vec![f64::NAN; n];
    let Ok(text) = std::fs::read_to_string(&path) else {
        return fresh;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return fresh;
    };
    // A cache measured under different settings is not reusable; start over rather than mix bases.
    if v["probe_tokens"].as_u64() != Some(probe_tokens as u64)
        || v["t_ref"].as_u64() != Some(t_ref as u64)
        || v["deltas"].as_array().map(Vec::len) != Some(n)
    {
        eprintln!("e5 cache at {} does not match this configuration — ignoring", path.display());
        return fresh;
    }
    let got: Vec<f64> = v["deltas"]
        .as_array()
        .expect("deltas")
        .iter()
        .map(|x| x.as_f64().unwrap_or(f64::NAN))
        .collect();
    let done = got.iter().filter(|d| d.is_finite()).count();
    eprintln!("e5 cache: {done}/{n} tensors already measured");
    got
}

fn save_cache(deltas: &[f64], probe_tokens: usize, t_ref: usize) {
    let path = cache_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let body = serde_json::json!({
        "probe_tokens": probe_tokens,
        "t_ref": t_ref,
        "deltas": deltas.iter().map(|d| if d.is_finite() { serde_json::json!(d) } else { serde_json::Value::Null }).collect::<Vec<_>>(),
    });
    let _ = std::fs::write(&path, serde_json::to_string(&body).unwrap_or_default());
}
