//! **Is the quadratic model valid at the size of a ternary error? (H1, first-principles plan)**
//!
//! Every allocator this repo has tried — Gram, curvature, exact task-loss gradients, rank
//! allocation with an exact σ² price — has lost to uniform, and on a five-unit exactly-budgeted
//! problem too, so proxy over-fitting is not the explanation. The candidate left standing is the
//! model under all of them: `ΔL ≈ gᵀe + ½ eᵀHe`. A ternary error is not small (it is O(1) relative
//! to the weight at T=1–2), and a second-order expansion has no reason to hold there. The O-P paper
//! disclaims its own theory below ~2 bits for the same reason.
//!
//! This measures the model directly. For each tensor `W` with quantization error `E = Q(W) − W`, the
//! loss is evaluated exactly along the ray `W + εE` at `ε ∈ {±¼, ±1}` (everything else fp). That
//! gives, per tensor:
//!
//! - `ΔL(1)`: the real cost of quantizing this tensor;
//! - the **exact local quadratic along the ray**, `16·even(¼)` with `even(ε) = (ΔL(ε)+ΔL(−ε))/2`,
//!   plus the first-order term `gᵀE` from one backward. This is the BEST any quadratic proxy
//!   (Gram, K-FAC, Fisher) can do: they all approximate this curvature. If it fails at `ε = 1`,
//!   every quadratic allocator fails with it, and the failure is the model, not the approximation;
//! - `R = even(1) / (16·even(¼))`, the non-quadraticity of the ray (`1` = exactly quadratic);
//! - the two proxies actually used: the input-Gram (GPTQ) objective `tr(E·G·Eᵀ)` and an empirical
//!   diagonal Fisher `Σ F⊙E²` (window-gradient squares; coarse, reported for ranking only).
//!
//! Then Spearman rank correlation of each prediction against `ΔL(1)` across tensors, per `T`, and one
//! additivity check (all selected tensors at once vs the sum of their singles) as a first H2 read.
//!
//! Kill/promote (plan): T2 is promoted if the on-ray-quadratic rank correlation falls below 0.5 at
//! `T ≤ 2`; killed if it holds (then allocation failed for another reason).
//!
//! # Measured 2026-09-23 — SmolLM2-135M, 42 tensors, 4×512 WikiText-2 eval tokens
//!
//! ```text
//!                                        T=2        T=3
//! R = even(1)/(16·even(¼)), median     1.00       1.01    (IQR [1.00,1.01] / [1.00,1.02])
//! Spearman ΔL(+1) vs local quadratic   +0.998     +1.000
//! joint ΔL / Σ single ΔL                1.018      1.030
//! Spearman even part vs Gram           +0.756     +0.824
//! Spearman even part vs diagFisher     +0.863     +0.806
//! Spearman ΔL(+1) vs even part         +0.484     +0.288
//! Σ|odd| / Σ|even|                      0.61       1.82
//! tensors with ΔL(+1) < 0              10/42      16/42
//! ```
//!
//! **T2 ("beyond quadratic") is killed.** At ternary error sizes the loss along every quantization
//! ray is a quadratic to within a few percent, and the per-tensor costs add (joint/Σ 1.02-1.03). The
//! quadratic proxies rank the curvature part well (Gram +0.76/+0.82, diagonal Fisher +0.86/+0.81).
//!
//! **What they do NOT rank is the measured cost**, because per tensor it is dominated by the first-
//! order term `gᵀE`: at T=3 the odd part is 1.8x the even part in absolute sum, 16 of 42 tensors
//! *improve* the loss when quantized alone, and measured ΔL correlates with the curvature at only
//! +0.29. `first_order_term_replicates_across_disjoint_text` shows that term is the SAMPLE, not the
//! model — `gᵀE` from three disjoint 2k-token sets correlates at -0.46..+0.38 with mixed signs, and
//! its sum flips sign:
//!
//! ```text
//! Pearson of per-tensor gᵀE     T=2      T=3
//! eval 0-3 vs eval 4-7        -0.120   +0.384
//! eval 0-3 vs train 8-11      +0.243   +0.061
//! eval 4-7 vs train 8-11      +0.086   -0.455
//! ```
//!
//! So a per-unit loss measured on a finite sample carries a first-order noise term as large as the
//! signal. That noise shrinks only as 1/sqrt(tokens), while a unit's curvature shrinks linearly
//! with its size, so per-unit SNR gets WORSE at finer granularity. Any allocator scored or driven
//! by measured per-unit loss (the exact task-loss-gradient allocator that lost to random) is
//! dominated by it, and a reallocation's gain (second order) competes with an evaluation noise
//! that is first order. The stable objective is the curvature (even) part, which the cheap proxies
//! already track. Bounds: one model, 42 of 211 tensors, 2k-token samples; the replication of the
//! EVEN part across samples is assumed (a sum of PSD terms), not measured here.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test quadratic_validity -- --ignored --nocapture
//! ```
//! Knobs: `TRITIUM_H1_LAYERS` (comma list, default `0,6,12,18,24,29`), `TRITIUM_H1_TS` (default
//! `1,2,3`), `TRITIUM_H1_WINDOWS` (eval windows of 512, default 4).

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{Calib, GramSet, calibrate, extract, fold, forward, score_window};
use tritium_nn::ModelRunner;
use tritium_train::Tape;
use tritium_train::ops::ste::{self, RotationPolicy};
use tritium_train::tape::ValueId;

const WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 4;
const GROUP: usize = 128;
const GRID: usize = 16;
const ROLES: [&str; 7] = ["q", "k", "v", "o", "gate", "up", "down"];

fn env_list(key: &str, default: &str) -> Vec<usize> {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .map(|s| s.trim().parse().expect("usize list"))
        .collect()
}

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR")
            .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m")),
    )
}

fn corpus() -> (Vec<u32>, Vec<u32>) {
    let path = std::env::var("TRITIUM_CORPUS").expect("set TRITIUM_CORPUS (WikiText-2 basis)");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("corpus")).expect("json");
    let ids = |k: &str| -> Vec<u32> {
        v[k].as_array()
            .expect(k)
            .iter()
            .map(|x| x.as_u64().expect("id") as u32)
            .collect()
    };
    (ids("train_ids"), ids("eval_ids"))
}

/// Mean next-token NLL over the eval windows.
fn mean_nll(weights: &[Vec<f32>], a: &common::Arch, windows: &[&[u32]]) -> f64 {
    let (mut nll, mut scored) = (0.0f64, 0usize);
    for w in windows {
        let mut t = Tape::new();
        let wids: Vec<ValueId> = weights.iter().map(|x| t.leaf(x.clone())).collect();
        let out = forward(&mut t, &wids, a, w);
        score_window(t.value(out), w, a.vocab, &mut nll, &mut scored);
    }
    nll / scored as f64
}

/// Gradient of the mean NLL (the quantity [`mean_nll`] returns) w.r.t. the selected weights, plus
/// the window-level squared gradients for the empirical diagonal Fisher.
fn gradients(
    weights: &[Vec<f32>],
    a: &common::Arch,
    windows: &[&[u32]],
    selected: &[usize],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut grad: Vec<Vec<f64>> = selected
        .iter()
        .map(|&i| vec![0.0; weights[i].len()])
        .collect();
    let mut fisher = grad.clone();
    for w in windows {
        let seq = w.len();
        let mut t = Tape::new();
        let wids: Vec<ValueId> = weights.iter().map(|x| t.leaf(x.clone())).collect();
        let logits = forward(&mut t, &wids, a, w);
        let mut target = vec![0.0f32; seq * a.vocab];
        for p in 0..seq - 1 {
            target[p * a.vocab + w[p + 1] as usize] = 1.0;
        }
        let tg = t.leaf(target);
        let loss = t.softmax_xent(logits, tg, seq, a.vocab);
        let g = t.backward(loss);
        // softmax_xent averages over `seq` rows, the last of which has no target; rescale to the
        // mean over the `seq − 1` scored positions, then average the windows.
        let scale = seq as f64 / (seq - 1) as f64 / windows.len() as f64;
        for (slot, &i) in selected.iter().enumerate() {
            for (j, &v) in g[wids[i]].iter().enumerate() {
                let v = f64::from(v) * scale;
                grad[slot][j] += v;
                fisher[slot][j] += v * v * windows.len() as f64;
            }
        }
    }
    (grad, fisher)
}

/// `tr(E·G·Eᵀ)` for `E` of shape `[rows, k]` and a `k×k` Gram.
fn gram_objective(e: &[f64], rows: usize, k: usize, g: &[f64]) -> f64 {
    let mut total = 0.0;
    let mut tmp = vec![0.0f64; k];
    for r in 0..rows {
        let er = &e[r * k..(r + 1) * k];
        for (i, t) in tmp.iter_mut().enumerate() {
            *t = g[i * k..(i + 1) * k]
                .iter()
                .zip(er)
                .map(|(a, b)| a * b)
                .sum();
        }
        total += er.iter().zip(&tmp).map(|(a, b)| a * b).sum::<f64>();
    }
    total
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    for (rank, &i) in idx.iter().enumerate() {
        r[i] = rank as f64;
    }
    r
}

/// Spearman rank correlation over the pairs where both values are finite.
fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let (x, y): (Vec<f64>, Vec<f64>) = x
        .iter()
        .zip(y)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .unzip();
    let (rx, ry) = (ranks(&x), ranks(&y));
    let n = rx.len() as f64;
    let mean = (n - 1.0) / 2.0;
    let cov: f64 = rx
        .iter()
        .zip(&ry)
        .map(|(a, b)| (a - mean) * (b - mean))
        .sum();
    let var: f64 = rx.iter().map(|a| (a - mean) * (a - mean)).sum();
    cov / var
}

struct Row {
    label: String,
    actual: f64,
    minus: f64,
    first_order: f64,
    local_quadratic: f64,
    r: f64,
    gram: f64,
    fisher: f64,
}

#[test]
#[ignore = "H1 quadratic-validity sweep; needs SmolLM2-135M; run explicitly"]
fn quadratic_model_validity_along_the_quantization_ray() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let layers = env_list("TRITIUM_H1_LAYERS", "0,6,12,18,24,29");
    let ts = env_list("TRITIUM_H1_TS", "1,2,3");
    let n_windows = env_list("TRITIUM_H1_WINDOWS", "4")[0];

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(&fp, &arch, &train[w * WINDOW..(w + 1) * WINDOW], &mut calib);
    }
    let (folded, arch) = fold(&fp, &shapes, &arch, &calib, 0.75);

    // Input Grams on the folded model, from the same calibration tokens.
    let mut grams = GramSet::new(&arch);
    for w in 0..CALIB_WINDOWS {
        grams.accumulate_forward(&folded, &arch, &train[w * WINDOW..(w + 1) * WINDOW]);
    }
    let finish =
        |v: Vec<common::Gram>| -> Vec<Vec<f64>> { v.into_iter().map(|g| g.finish()).collect() };
    let (g_attn, g_ffn, g_down) = (finish(grams.attn), finish(grams.ffn), finish(grams.down));

    let windows: Vec<&[u32]> = eval.chunks_exact(WINDOW).take(n_windows).collect();
    let selected: Vec<usize> = layers
        .iter()
        .flat_map(|&l| (0..7).map(move |k| 1 + 7 * l + k))
        .collect();

    let started = Instant::now();
    let base = mean_nll(&folded, &arch, &windows);
    let (grad, fisher) = gradients(&folded, &arch, &windows, &selected);
    println!(
        "SmolLM2-135M | fold α=0.75 | ladder g{GROUP} group-Hadamard | {} tensors (layers {layers:?}) | \
         {n_windows}×{WINDOW} eval tokens | fp NLL {base:.5} ({:.0}s incl. gradients)\n",
        selected.len(),
        started.elapsed().as_secs_f64()
    );

    let delta_at = |i: usize, e: &[f32], eps: f32| -> f64 {
        let mut w = folded.clone();
        for (x, d) in w[i].iter_mut().zip(e) {
            *x += eps * d;
        }
        mean_nll(&w, &arch, &windows) - base
    };

    for &t in &ts {
        let started = Instant::now();
        let mut rows = Vec::new();
        let mut joint = folded.clone();
        for (slot, &i) in selected.iter().enumerate() {
            let (n, k) = shapes[i];
            let q = ste::salt_quantize_forward_grouped_geometric(
                &folded[i],
                n,
                k,
                t,
                GROUP,
                GRID,
                RotationPolicy::Always,
            );
            let e: Vec<f32> = q.iter().zip(&folded[i]).map(|(a, b)| a - b).collect();
            joint[i].clone_from(&q);
            let (p1, m1, pq, mq) = (
                delta_at(i, &e, 1.0),
                delta_at(i, &e, -1.0),
                delta_at(i, &e, 0.25),
                delta_at(i, &e, -0.25),
            );
            let even_q = 0.5 * (pq + mq);
            let even_1 = 0.5 * (p1 + m1);
            let first_order: f64 = grad[slot]
                .iter()
                .zip(&e)
                .map(|(g, &d)| g * f64::from(d))
                .sum();
            let e64: Vec<f64> = e.iter().map(|&d| f64::from(d)).collect();
            let layer = (i - 1) / 7;
            let role = (i - 1) % 7;
            let gram = match role {
                0..=2 => gram_objective(&e64, n, k, &g_attn[layer]),
                4 | 5 => gram_objective(&e64, n, k, &g_ffn[layer]),
                6 => gram_objective(&e64, n, k, &g_down[layer]),
                _ => f64::NAN, // o_proj: no tap (GQA concat), as in the fold
            };
            let fisher: f64 = fisher[slot].iter().zip(&e64).map(|(f, d)| f * d * d).sum();
            rows.push(Row {
                label: format!("L{layer:02} {}", ROLES[role]),
                actual: p1,
                minus: m1,
                first_order,
                local_quadratic: first_order + 16.0 * even_q,
                r: even_1 / (16.0 * even_q),
                gram,
                fisher,
            });
        }
        let joint_delta = mean_nll(&joint, &arch, &windows) - base;
        let sum_single: f64 = rows.iter().map(|r| r.actual).sum();

        println!("T={t}  ({:.0}s)", started.elapsed().as_secs_f64());
        println!(
            "{:<9} {:>10} {:>10} {:>10} {:>10} {:>8} {:>11} {:>11}",
            "tensor", "ΔL(+1)", "ΔL(−1)", "gᵀE", "loc.quad", "R", "Gram", "diagFisher"
        );
        for r in &rows {
            println!(
                "{:<9} {:>+10.5} {:>+10.5} {:>+10.5} {:>+10.5} {:>8.2} {:>11.4e} {:>11.4e}",
                r.label, r.actual, r.minus, r.first_order, r.local_quadratic, r.r, r.gram, r.fisher
            );
        }
        let col = |f: fn(&Row) -> f64| rows.iter().map(f).collect::<Vec<f64>>();
        let actual = col(|r| r.actual);
        let mut rs: Vec<f64> = col(|r| r.r).into_iter().filter(|v| v.is_finite()).collect();
        rs.sort_by(f64::total_cmp);
        let ratio: Vec<f64> = rows.iter().map(|r| r.actual / r.local_quadratic).collect();
        let mut sorted_ratio: Vec<f64> = ratio.iter().copied().filter(|v| v.is_finite()).collect();
        sorted_ratio.sort_by(f64::total_cmp);
        println!(
            "Spearman vs ΔL(+1): local quadratic {:+.3} | first-order {:+.3} | Gram {:+.3} (no o_proj) | diagFisher {:+.3}",
            spearman(&actual, &col(|r| r.local_quadratic)),
            spearman(&actual, &col(|r| r.first_order)),
            spearman(&actual, &col(|r| r.gram)),
            spearman(&actual, &col(|r| r.fisher)),
        );
        println!(
            "R = even(1)/(16·even(¼)): median {:.2}, IQR [{:.2}, {:.2}] | actual/loc.quad: median {:.2}, IQR [{:.2}, {:.2}]",
            rs[rs.len() / 2],
            rs[rs.len() / 4],
            rs[3 * rs.len() / 4],
            sorted_ratio[sorted_ratio.len() / 2],
            sorted_ratio[sorted_ratio.len() / 4],
            sorted_ratio[3 * sorted_ratio.len() / 4],
        );
        println!(
            "additivity: joint ΔL {joint_delta:+.5} vs Σ singles {sum_single:+.5} (ratio {:.3})\n",
            joint_delta / sum_single
        );
    }
}

/// Pearson correlation.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let cov: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let vx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    let vy: f64 = y.iter().map(|b| (b - my) * (b - my)).sum();
    cov / (vx * vy).sqrt()
}

/// **Is the linear term real, or the sample? (H1b)**
///
/// H1 found the loss along each quantization ray exactly quadratic and nearly additive, but the
/// per-tensor cost dominated by the first-order term `gᵀE`, which sums to ~0 across tensors. That
/// term is `E` projected on the gradient of the *evaluation sample's* loss. If it is sampling noise,
/// any per-unit measured ΔL — and every allocator ranked by one — is chasing noise. If it
/// replicates on disjoint text, the fp model is measurably off-stationary on this distribution and
/// the quantizer can steer `E` against it.
///
/// `gᵀE` for the same 42 tensors under three disjoint gradient sources: eval windows 0-3, eval
/// windows 4-7, train windows 8-11 (calibration uses train 0-3).
#[test]
#[ignore = "H1b linear-term replication; needs SmolLM2-135M; run explicitly"]
fn first_order_term_replicates_across_disjoint_text() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let layers = env_list("TRITIUM_H1_LAYERS", "0,6,12,18,24,29");
    let ts = env_list("TRITIUM_H1_TS", "2,3");
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(&fp, &arch, &train[w * WINDOW..(w + 1) * WINDOW], &mut calib);
    }
    let (folded, arch) = fold(&fp, &shapes, &arch, &calib, 0.75);
    let selected: Vec<usize> = layers
        .iter()
        .flat_map(|&l| (0..7).map(move |k| 1 + 7 * l + k))
        .collect();
    let windows = |ids: &[u32], from: usize| -> Vec<Vec<u32>> {
        (from..from + 4)
            .map(|w| ids[w * WINDOW..(w + 1) * WINDOW].to_vec())
            .collect()
    };
    let sources = [
        ("eval 0-3", windows(&eval, 0)),
        ("eval 4-7", windows(&eval, 4)),
        ("train 8-11", windows(&train, 8)),
    ];
    let grads: Vec<Vec<Vec<f64>>> = sources
        .iter()
        .map(|(_, w)| {
            let refs: Vec<&[u32]> = w.iter().map(Vec::as_slice).collect();
            gradients(&folded, &arch, &refs, &selected).0
        })
        .collect();
    for &t in &ts {
        let proj: Vec<Vec<f64>> = grads
            .iter()
            .map(|g| {
                selected
                    .iter()
                    .enumerate()
                    .map(|(slot, &i)| {
                        let (n, k) = shapes[i];
                        let q = ste::salt_quantize_forward_grouped_geometric(
                            &folded[i],
                            n,
                            k,
                            t,
                            GROUP,
                            GRID,
                            RotationPolicy::Always,
                        );
                        g[slot]
                            .iter()
                            .zip(q.iter().zip(&folded[i]))
                            .map(|(g, (a, b))| g * f64::from(a - b))
                            .sum()
                    })
                    .collect()
            })
            .collect();
        println!("T={t}: gᵀE per tensor under three disjoint gradient sources");
        for a in 0..3 {
            for b in a + 1..3 {
                println!(
                    "  {:<10} vs {:<10}  Pearson {:+.3}  Spearman {:+.3}",
                    sources[a].0,
                    sources[b].0,
                    pearson(&proj[a], &proj[b]),
                    spearman(&proj[a], &proj[b])
                );
            }
        }
        for (s, p) in sources.iter().zip(&proj) {
            println!(
                "  {:<10} Σ gᵀE {:+.5}   Σ|gᵀE| {:.5}",
                s.0,
                p.iter().sum::<f64>(),
                p.iter().map(|v| v.abs()).sum::<f64>()
            );
        }
    }
}
