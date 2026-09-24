//! **How much is left in the gauge? (H3, first-principles plan)**
//!
//! Every quality win this repo has recorded came from an exact symmetry of the network: the AWQ
//! fold (a per-channel scale moved into a norm), the Hadamard rotation, the 1/3 ladder spacing.
//! H1 then showed the loss along a quantization ray is quadratic and additive and the Gram proxy
//! ranks its curvature well. That makes the gauge question well posed: among the transforms that
//! leave the fp function unchanged, which one makes the ternary lattice fit best?
//!
//! Objective A (a model that runs 100% ternary arithmetic) restricts the admissible family to
//! transforms the runtime can apply without multiplies: ±1 signs, permutations, Walsh–Hadamard
//! blocks (add/subtract only), and channel scales absorbed into a norm. A dense learned rotation is
//! outside that family; a random dense orthogonal is included as the CONTROL that asks whether
//! Hadamard is special or any rotation would do.
//!
//! Every arm quantizes the same folded weights with the geometric ladder at `g128` scales, fp
//! activations, and scores held-out WikiText-2 perplexity through the tape. Arms:
//!
//! - `H128` — the shipping gauge: group Hadamard, no signs (the baseline);
//! - `H128 + random signs` — one seeded ±1 vector per tensor;
//! - `H128 + best-of-K signs` — per tensor, the sign vector minimising the Gram objective
//!   `tr(E·G·Eᵀ)` over `K` seeds (a search over the sign gauge with the proxy H1 validated);
//! - `H{64,256}` — block size as a gauge dimension (256 only where the width allows);
//! - `dense random orthogonal 128` — the control;
//! - `no rotation` — the floor.
//!
//! Kill/promote (plan): T1 is promoted if an admissible gauge beats `H128` by more than 3% ppl at
//! T=2. The proxy sum `Σ tr(E·G·Eᵀ)` is printed beside each arm, so the ranking can also be
//! checked against H1's prediction that the curvature proxy orders aggregate outcomes.
//!
//! # Measured 2026-09-23 — SmolLM2-135M, full WikiText-2 eval (32,768 tokens), fp 22.675
//!
//! ```text
//! gauge                            T=2 ppl   vs H128   Σ tr(EGEᵀ)    T=3 ppl   vs H128   Σ tr(EGEᵀ)
//! H128 (shipping)                   46.405    +0.00%    3.185e4       24.278    +0.00%    4.630e3
//! no rotation                     1098.742 +2267.74%    6.949e4       29.400   +21.10%    9.498e3
//! dense random orthogonal 128       53.920   +16.19%    3.558e4       25.100    +3.39%    4.447e3
//! H128 + random signs (seed 0)      45.506    -1.94%    3.098e4       24.800    +2.15%    4.514e3
//! H128 + best-of-16 signs           44.943    -3.15%    3.009e4       24.619    +1.40%    4.176e3
//! H64                               47.854    +3.12%    3.235e4       24.452    +0.72%    4.621e3
//! H256 (where the width allows)     46.598    +0.42%    3.314e4       24.348    +0.29%    4.371e3
//!
//! noise floor: random signs, seeds 0-3
//! T=2  45.506 46.014 45.531 46.336   mean 45.85  SD 0.9%
//! T=3  24.800 24.509 24.460 24.550   mean 24.58  SD 0.6%
//! ```
//!
//! **The Hadamard harvests nearly all of the gauge.** Against a random dense rotation it is worth
//! 16% at T=2 and 3.4% at T=3, so the STRUCTURE matters, not just incoherence. That is the robust
//! result. Everything above it is within about 2 SD of gauge-to-gauge noise, and the sign depends
//! on T:
//! - at T=2, plain H128 is worse than all 4 random signings (+1.2% vs their mean) and the
//!   proxy-selected best-of-16 beats all 4 (-2.0% vs mean, -3.15% vs H128);
//! - at T=3, plain H128 BEATS all 4 random signings (-1.2%), and proxy selection buys nothing
//!   (best-of-16 lands on the random mean although it cuts Σ tr(EGEᵀ) by 10%).
//!
//! The plan's T1 bar (an admissible gauge beating H128 by more than 3% at T=2) is met by 0.15 points
//! and reverses at T=3, so it is recorded as NOT robustly met. The aggregate Gram proxy orders the
//! T=2 arms but not the T=3 arms. Gauge noise of 0.6-0.9% (1 SD) means any single-draw comparison
//! below ~2% between quantizer configurations is unresolved.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test gauge_search -- --ignored --nocapture
//! ```
//! Knobs: `TRITIUM_H3_TS` (default `2,3`), `TRITIUM_H3_K` (sign seeds, default 16),
//! `TRITIUM_H3_EVAL_WINDOWS` (512-token windows, default all 64), `TRITIUM_H3_ARMS` (labels),
//! `TRITIUM_H3_SEED` (re-draws the random gauges, default 0).

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{Arch, Calib, GramSet, calibrate, extract, fold, perplexity_windowed};
use tritium_nn::ModelRunner;
use tritium_train::ops::ste::{self, RotationPolicy};

const WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 4;
const GROUP: usize = 128;
const GRID: usize = 16;

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

/// SplitMix64: a deterministic, dependency-free stream for sign vectors and Gaussian draws.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn sign(&mut self) -> f32 {
        if self.next() & 1 == 0 { 1.0 } else { -1.0 }
    }
    fn gaussian(&mut self) -> f64 {
        // Box–Muller on two open-interval uniforms.
        let u = ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let v = ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// A per-tensor input transform applied to every `block`-wide column slice of every row.
/// Orthonormal, so the stored tensor is `W·Bᵀ` and the runtime feeds it `B·x`.
enum Gauge {
    None,
    /// `H·D` on each block, `D` the given signs across the full input width.
    SignedHadamard {
        block: usize,
        signs: Vec<f32>,
    },
    /// A dense orthogonal `block × block` matrix (row-major), the same for every block.
    Dense {
        block: usize,
        q: Vec<f64>,
    },
}

impl Gauge {
    fn forward(&self, row: &mut [f32]) {
        match self {
            Gauge::None => {}
            Gauge::SignedHadamard { block, signs } => {
                for (v, s) in row.iter_mut().zip(signs) {
                    *v *= s;
                }
                hadamard_blocks(row, *block);
            }
            Gauge::Dense { block, q } => {
                for b in row.chunks_exact_mut(*block) {
                    let x: Vec<f64> = b.iter().map(|&v| f64::from(v)).collect();
                    for (i, out) in b.iter_mut().enumerate() {
                        *out = q[i * block..(i + 1) * block]
                            .iter()
                            .zip(&x)
                            .map(|(a, b)| a * b)
                            .sum::<f64>() as f32;
                    }
                }
            }
        }
    }

    fn inverse(&self, row: &mut [f32]) {
        match self {
            Gauge::None => {}
            Gauge::SignedHadamard { block, signs } => {
                hadamard_blocks(row, *block);
                for (v, s) in row.iter_mut().zip(signs) {
                    *v *= s;
                }
            }
            Gauge::Dense { block, q } => {
                for b in row.chunks_exact_mut(*block) {
                    let x: Vec<f64> = b.iter().map(|&v| f64::from(v)).collect();
                    for (j, out) in b.iter_mut().enumerate() {
                        *out = (0..*block).map(|i| q[i * block + j] * x[i]).sum::<f64>() as f32;
                    }
                }
            }
        }
    }
}

/// Normalized Hadamard on each `block`-wide slice. A ragged tail is rotated too when its length is
/// a power of two — exactly what `RotationPolicy::Always` does (576 = 4 × 128 + 64 on SmolLM2).
fn hadamard_blocks(row: &mut [f32], block: usize) {
    for b in row.chunks_mut(block) {
        if b.len().is_power_of_two() && b.len() > 1 {
            ste::fast_hadamard(b);
        }
    }
}

/// Quantize `w [rows, cols]` in `gauge`'s basis with the g128 ladder, returning the
/// reconstruction in the ORIGINAL basis (what the fp tape consumes).
fn quantize_in(w: &[f32], rows: usize, cols: usize, t: usize, gauge: &Gauge) -> Vec<f32> {
    let mut rotated = w.to_vec();
    for r in rotated.chunks_exact_mut(cols) {
        gauge.forward(r);
    }
    let mut q = ste::salt_quantize_forward_grouped_geometric(
        &rotated,
        rows,
        cols,
        t,
        GROUP,
        GRID,
        RotationPolicy::Never,
    );
    for r in q.chunks_exact_mut(cols) {
        gauge.inverse(r);
    }
    q
}

/// `tr(E·G·Eᵀ)`, or the plain Frobenius norm when there is no Gram (o_proj).
fn gram_objective(e: &[f32], cols: usize, g: Option<&[f64]>) -> f64 {
    let Some(g) = g else {
        return e.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
    };
    let mut total = 0.0;
    let mut tmp = vec![0.0f64; cols];
    for er in e.chunks_exact(cols) {
        for (i, t) in tmp.iter_mut().enumerate() {
            *t = g[i * cols..(i + 1) * cols]
                .iter()
                .zip(er)
                .map(|(a, &b)| a * f64::from(b))
                .sum();
        }
        total += er
            .iter()
            .zip(&tmp)
            .map(|(&a, b)| f64::from(a) * b)
            .sum::<f64>();
    }
    total
}

/// Gram-Schmidt of a Gaussian matrix: a Haar-ish random orthogonal `n × n`.
fn random_orthogonal(n: usize, rng: &mut Rng) -> Vec<f64> {
    let mut q = vec![0.0f64; n * n];
    for i in 0..n {
        let mut v: Vec<f64> = (0..n).map(|_| rng.gaussian()).collect();
        for j in 0..i {
            let d: f64 = (0..n).map(|k| v[k] * q[j * n + k]).sum();
            for k in 0..n {
                v[k] -= d * q[j * n + k];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        for k in 0..n {
            q[i * n + k] = v[k] / norm;
        }
    }
    q
}

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    NoRotation,
    Hadamard(usize),
    RandomSigns,
    BestOfKSigns,
    DenseRandom,
}

impl Arm {
    fn label(self, k: usize) -> String {
        match self {
            Arm::NoRotation => "no rotation".into(),
            Arm::Hadamard(b) => format!("H{b}"),
            Arm::RandomSigns => "H128 + random signs".into(),
            Arm::BestOfKSigns => format!("H128 + best-of-{k} signs"),
            Arm::DenseRandom => "dense random orthogonal 128".into(),
        }
    }
}

#[test]
#[ignore = "H3 gauge search; needs SmolLM2-135M; run explicitly"]
fn admissible_gauges_against_the_hadamard_baseline() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let ts = env_list("TRITIUM_H3_TS", "2,3");
    let k = env_list("TRITIUM_H3_K", "16")[0];
    // Re-draws every random gauge; `random signs` across seeds is the noise floor between gauges
    // that are equal in expectation.
    let seed = env_list("TRITIUM_H3_SEED", "0")[0] as u64;
    let n_eval = env_list("TRITIUM_H3_EVAL_WINDOWS", "64")[0];
    let only: Option<Vec<String>> = std::env::var("TRITIUM_H3_ARMS")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().to_owned()).collect());

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(&fp, &arch, &train[w * WINDOW..(w + 1) * WINDOW], &mut calib);
    }
    let (folded, arch): (Vec<Vec<f32>>, Arch) = fold(&fp, &shapes, &arch, &calib, 0.75);
    let mut grams = GramSet::new(&arch);
    for w in 0..CALIB_WINDOWS {
        grams.accumulate_forward(&folded, &arch, &train[w * WINDOW..(w + 1) * WINDOW]);
    }
    let finish =
        |v: Vec<common::Gram>| -> Vec<Vec<f64>> { v.into_iter().map(|g| g.finish()).collect() };
    let (g_attn, g_ffn, g_down) = (finish(grams.attn), finish(grams.ffn), finish(grams.down));
    let gram_for = |i: usize| -> Option<&[f64]> {
        if i == 0 {
            return None; // tied embed/head: no single input Gram
        }
        let (layer, role) = ((i - 1) / 7, (i - 1) % 7);
        match role {
            0..=2 => Some(&g_attn[layer]),
            4 | 5 => Some(&g_ffn[layer]),
            6 => Some(&g_down[layer]),
            _ => None,
        }
    };
    // The H128 arm must BE the shipping quantizer, or "vs H128" compares against a strawman.
    {
        let i = 1; // layer 0 q_proj, cols = 576 (ragged 64-wide tail)
        let (rows, cols) = shapes[i];
        let ours = quantize_in(
            &folded[i],
            rows,
            cols,
            2,
            &Gauge::SignedHadamard {
                block: 128,
                signs: vec![1.0; cols],
            },
        );
        let shipping = ste::salt_quantize_forward_grouped_geometric(
            &folded[i],
            rows,
            cols,
            2,
            GROUP,
            GRID,
            RotationPolicy::Always,
        );
        let worst = ours
            .iter()
            .zip(&shipping)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= 1e-6,
            "H128 gauge differs from RotationPolicy::Always by {worst}"
        );
    }
    let eval = &eval[..(n_eval * WINDOW).min(eval.len())];
    let ppl_fp = perplexity_windowed(&folded, &arch, eval, WINDOW);
    println!(
        "SmolLM2-135M | fold α=0.75 | ladder g{GROUP} | fp activations | {} eval tokens | fp {ppl_fp:.3}\n",
        eval.len()
    );

    let arms = [
        Arm::Hadamard(128),
        Arm::NoRotation,
        Arm::DenseRandom,
        Arm::RandomSigns,
        Arm::BestOfKSigns,
        Arm::Hadamard(64),
        Arm::Hadamard(256),
    ];
    for &t in &ts {
        println!(
            "T={t}\n{:<30} {:>10} {:>9} {:>11} {:>14}",
            "gauge", "ppl", "× fp", "vs H128", "Σ tr(EGEᵀ)"
        );
        let mut baseline = f64::NAN;
        for arm in arms {
            let label = arm.label(k);
            if only
                .as_ref()
                .is_some_and(|o| !o.iter().any(|l| *l == label))
            {
                continue;
            }
            let started = Instant::now();
            let mut rng = Rng(0x7157_1A7E ^ t as u64 ^ seed.wrapping_mul(0x9E37_79B9));
            let mut proxy = 0.0f64;
            let q: Vec<Vec<f32>> = folded
                .iter()
                .zip(&shapes)
                .enumerate()
                .map(|(i, (w, &(rows, cols)))| {
                    let g = gram_for(i);
                    let score = |gauge: &Gauge| -> (Vec<f32>, f64) {
                        let q = quantize_in(w, rows, cols, t, gauge);
                        let e: Vec<f32> = q.iter().zip(w).map(|(a, b)| a - b).collect();
                        let obj = gram_objective(&e, cols, g);
                        (q, obj)
                    };
                    let block_fits = |b: usize| cols % b == 0;
                    let (q, obj) = match arm {
                        Arm::NoRotation => score(&Gauge::None),
                        // A width the block does not divide falls back to H128 for that tensor.
                        Arm::Hadamard(b) => {
                            let block = if block_fits(b) { b } else { 128 };
                            score(&Gauge::SignedHadamard {
                                block,
                                signs: vec![1.0; cols],
                            })
                        }
                        Arm::RandomSigns => score(&Gauge::SignedHadamard {
                            block: 128,
                            signs: (0..cols).map(|_| rng.sign()).collect(),
                        }),
                        Arm::BestOfKSigns => (0..k)
                            .map(|_| {
                                score(&Gauge::SignedHadamard {
                                    block: 128,
                                    signs: (0..cols).map(|_| rng.sign()).collect(),
                                })
                            })
                            .min_by(|a, b| a.1.total_cmp(&b.1))
                            .expect("k >= 1"),
                        Arm::DenseRandom => score(&Gauge::Dense {
                            block: 128,
                            q: random_orthogonal(128, &mut rng),
                        }),
                    };
                    proxy += obj;
                    q
                })
                .collect();
            let ppl = perplexity_windowed(&q, &arch, eval, WINDOW);
            if baseline.is_nan() {
                baseline = ppl;
            }
            println!(
                "{label:<30} {ppl:>10.3} {:>8.3}× {:>+10.2}% {proxy:>14.5e}   ({:.0}s)",
                ppl / ppl_fp,
                100.0 * (ppl - baseline) / baseline,
                started.elapsed().as_secs_f64()
            );
        }
        println!();
    }
}
