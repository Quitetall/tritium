//! **Does the Golay-coded trit-plane gain transfer to a second model? (H8 transfer check)**
//!
//! H8 measured, on SmolLM2-135M, that constraining the finest trit plane of every 12-weight block to
//! an extended ternary Golay codeword (`T + ½` trits a weight, still an ordinary ternary plane)
//! beats every uncoded structure at the same bits. This repeats the plain (calibration-free)
//! comparison on Qwen3.5-0.8B-Base — a different family, a hybrid DeltaNet/attention stack, wider
//! layers — through the loader's tensor-transform seam: every projection is quantized in the g128
//! group-Hadamard basis; embeddings, norms and DeltaNet scalars stay fp; activations are fp.
//!
//! Qwen's widths (1024, 2048, 3584) are not multiples of 12. The last `cols mod 12` columns of each
//! row use the uncoded ladder at `T + 1` planes with their own fp16 scale, and the reported bits per
//! weight include that tail exactly (~0.2% of weights).
//!
//! Arms at each `T`: the ladder at `T` and `T + 1`, Golay `T + ½`, time-sharing (alternate rows at
//! `T` / `T + 1`, the only uncoded ternary way to reach `T + ½`), and the best single uniform scalar
//! at that rate (5 / 15 levels, not trit-storable). Each scored with the antithetic split
//! (`even` = curvature cost with the first-order sample term cancelled; H5).
//!
//! # Measured 2026-10-09 — Qwen3.5-0.8B-Base, 4 × 1024 WikiText-2 test tokens, fp ppl 14.547
//!
//! ```text
//! structure                 bits/w       ppl     even (curvature)    odd
//! ladder T=1                 1.710  20158.907      +6.04488       +1.18917
//! Golay-coded 1+½            2.518     61.642      +1.40464       +0.03934
//! time-share 1/2             2.502    506.329      +3.07834       +0.47149
//! uniform 5-level scalar     2.447    200.353      +2.53937       +0.08335
//! ladder T=2                 3.295     21.532      +0.40670       -0.01451
//! Golay-coded 2+½            4.103     16.242      +0.10020       +0.01003
//! time-share 2/3             4.087     18.222      +0.20067       +0.02462
//! uniform 15-level scalar    4.032     16.524      +0.12763       -0.00018
//! ladder T=3                 4.880     15.190      +0.03979       +0.00349
//! ```
//!
//! **The gain transfers, and is larger than on SmolLM2.** Against the ternary fallback (time-share)
//! the code is 8.2× better in ppl at ~2.5 bpw and −10.9% at ~4.1 bpw, halving the curvature cost at
//! both; against the best single scalar it is 3.3× / −1.7% in ppl and −45% / −22% in curvature
//! (SmolLM2: −21% / −17%). Golay 2+½ at 4.10 bpw is within 7% of ladder T=3 at 4.88 bpw. Plain
//! nearest-point fits only — no calibration on this model.
//!
//! ```text
//! TRITIUM_QWEN35_DIR=/mnt/4tb/models/qwen3.5-0.8b-base-tritium \
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k_qwen35.json \
//!   cargo test -p tritium-nn --release --test qwen35_coded_planes -- --ignored --nocapture
//! ```
//! Knobs: `TRITIUM_H8Q_TS` (default `1,2`), `TRITIUM_H8Q_WINDOW` (default 1024),
//! `TRITIUM_H8Q_WINDOWS` (default 4).

mod common;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use common::golay::{BLOCK, GRID, GROUP, golay_codewords, golay_row, hadamard_blocks};
use rayon::prelude::*;
use tritium_nn::Qwen35HfSource;
use tritium_train::ops::ste::{self, RotationPolicy};

fn env_list(key: &str, default: &str) -> Vec<usize> {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_owned())
        .split(',')
        .map(|s| s.trim().parse().expect("usize list"))
        .collect()
}

fn eval_ids() -> Vec<u32> {
    let path =
        std::env::var("TRITIUM_CORPUS").expect("set TRITIUM_CORPUS to a Qwen-tokenized corpus");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("corpus")).expect("json");
    v["eval_ids"]
        .as_array()
        .expect("eval_ids")
        .iter()
        .map(|x| x.as_u64().expect("id") as u32)
        .collect()
}

/// Every projection matrix — the H4 "all projections" set.
fn is_projection(name: &str) -> bool {
    name.ends_with("_proj.weight")
        || name.ends_with("in_proj_qkv.weight")
        || name.ends_with("in_proj_a.weight")
        || name.ends_with("in_proj_b.weight")
        || name.ends_with("in_proj_z.weight")
}

#[derive(Clone, Copy)]
enum Arm {
    Ladder(usize),
    Golay(usize),
    TimeShare(usize),
    Uniform(usize),
}

impl Arm {
    fn label(self) -> String {
        match self {
            Arm::Ladder(t) => format!("ladder T={t}"),
            Arm::Golay(t) => format!("Golay-coded {t}+½"),
            Arm::TimeShare(t) => format!("time-share {t}/{}", t + 1),
            Arm::Uniform(l) => format!("uniform {l}-level scalar"),
        }
    }
}

const LOG3: f64 = 1.584_962_500_721_156;

/// Quantize one rotated row in place; returns the bits it costs.
fn quantize_row(y: &mut [f32], arm: Arm, row_index: usize, code: &[[i8; BLOCK]]) -> f64 {
    let cols = y.len();
    let groups = cols.div_ceil(GROUP) as f64;
    let ladder = |v: &mut [f32], t: usize| {
        let q = ste::salt_quantize_forward_grouped_geometric(
            v,
            1,
            v.len(),
            t,
            GROUP,
            GRID,
            RotationPolicy::Never,
        );
        v.copy_from_slice(&q);
    };
    match arm {
        Arm::Ladder(t) => {
            ladder(y, t);
            cols as f64 * t as f64 * LOG3 + 16.0 * groups
        }
        Arm::TimeShare(t) => {
            let t = t + row_index % 2;
            ladder(y, t);
            cols as f64 * t as f64 * LOG3 + 16.0 * groups
        }
        Arm::Uniform(levels) => {
            let k = ((levels - 1) / 2) as f32;
            for g in y.chunks_mut(GROUP) {
                let peak = g.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let mut best = (f32::INFINITY, 0.0f32);
                for i in 0..36 {
                    let d = peak / k * (0.35 + 0.02 * i as f32);
                    if d <= 0.0 {
                        break;
                    }
                    let e: f32 = g
                        .iter()
                        .map(|v| (v - (v / d).round().clamp(-k, k) * d).powi(2))
                        .sum();
                    if e < best.0 {
                        best = (e, d);
                    }
                }
                let d = best.1;
                for v in g.iter_mut() {
                    *v = if d > 0.0 {
                        (*v / d).round().clamp(-k, k) * d
                    } else {
                        0.0
                    };
                }
            }
            cols as f64 * (levels as f64).log2() + 16.0 * groups
        }
        Arm::Golay(t) => {
            let coded = cols / BLOCK * BLOCK;
            let rec = golay_row(&y[..coded], t, code);
            y[..coded].copy_from_slice(&rec);
            let mut bits =
                coded as f64 * (t as f64 + 0.5) * LOG3 + 16.0 * coded.div_ceil(GROUP) as f64;
            if coded < cols {
                // Uncoded tail: ladder at T+1 planes with its own scale.
                ladder(&mut y[coded..], t + 1);
                bits += (cols - coded) as f64 * (t + 1) as f64 * LOG3 + 16.0;
            }
            bits
        }
    }
}

/// Quantized projections of one arm, kept between the `+E` and `−E` loads so each tensor is
/// fitted once: name → (Q(W), bits).
type QCache = Mutex<HashMap<String, (Vec<f32>, f64)>>;

/// Mean next-token NLL with every projection replaced by `Q(W)` (or by `2W − Q(W)` when
/// `antithetic`); returns (mean NLL, quantized weights, bits).
fn mean_nll(
    dir: &std::path::Path,
    arm: Option<Arm>,
    antithetic: bool,
    windows: &[&[u32]],
    code: &[[i8; BLOCK]],
    cache: &QCache,
) -> (f64, usize, f64) {
    let tally = Mutex::new((0usize, 0.0f64));
    let transform = |name: &str, shape: &[usize], values: &mut [f32]| {
        let Some(arm) = arm else { return };
        if !is_projection(name) {
            return;
        }
        let [rows, cols] = shape else { return };
        let (rows, cols) = (*rows, *cols);
        let cached = cache.lock().expect("cache").get(name).cloned();
        let (q, bits) = cached.unwrap_or_else(|| {
            let mut q = values.to_vec();
            let bits: f64 = q
                .par_chunks_mut(cols)
                .enumerate()
                .map(|(r, row)| {
                    hadamard_blocks(row);
                    let b = quantize_row(row, arm, r, code);
                    hadamard_blocks(row); // normalized H is its own inverse
                    b
                })
                .sum();
            cache
                .lock()
                .expect("cache")
                .insert(name.to_owned(), (q.clone(), bits));
            (q, bits)
        });
        for (v, qv) in values.iter_mut().zip(&q) {
            *v = if antithetic { 2.0 * *v - qv } else { *qv };
        }
        let mut t = tally.lock().expect("tally");
        t.0 += rows * cols;
        t.1 += bits;
    };
    let model = Qwen35HfSource::open(dir)
        .expect("open")
        .verify_semantic_identity()
        .expect("verify")
        .load_language_transformed(Box::new(tritium_cpu::CpuBackend::new()), &transform)
        .expect("load");
    let runner = model.runner();
    let (mut total, mut count) = (0.0f64, 0usize);
    for window in windows {
        let mut cache = runner.new_cache(window.len()).expect("cache");
        let out = runner.forward(window, &mut cache).expect("forward");
        let nll: Vec<f64> = (0..window.len() - 1)
            .into_par_iter()
            .map(|pos| {
                let logits = runner.logits_for_row(&out, pos).expect("row logits");
                let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let lse = f64::from(max)
                    + logits
                        .iter()
                        .map(|&l| f64::from(l - max).exp())
                        .sum::<f64>()
                        .ln();
                lse - f64::from(logits[window[pos + 1] as usize])
            })
            .collect();
        total += nll.iter().sum::<f64>();
        count += nll.len();
    }
    let (params, bits) = *tally.lock().expect("tally");
    (total / count as f64, params, bits)
}

#[test]
#[ignore = "H8 transfer: needs Qwen3.5-0.8B-Base and a Qwen-tokenized corpus; run explicitly"]
fn golay_coded_planes_transfer_to_qwen35() {
    let dir = PathBuf::from(
        std::env::var("TRITIUM_QWEN35_DIR")
            .unwrap_or_else(|_| "/mnt/4tb/models/qwen3.5-0.8b-base-tritium".into()),
    );
    if !dir.join("config.json").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let ts = env_list("TRITIUM_H8Q_TS", "1,2");
    let window = env_list("TRITIUM_H8Q_WINDOW", "1024")[0];
    let n_windows = env_list("TRITIUM_H8Q_WINDOWS", "4")[0];
    let ids = eval_ids();
    let windows: Vec<&[u32]> = ids.chunks_exact(window).take(n_windows).collect();
    assert_eq!(windows.len(), n_windows, "corpus too short");
    let code = golay_codewords();

    let started = Instant::now();
    let (fp, _, _) = mean_nll(&dir, None, false, &windows, &code, &QCache::default());
    println!(
        "Qwen3.5-0.8B-Base | all projections, g{GROUP} group-Hadamard | fp activations | \
         {n_windows} x {window} WikiText-2 test tokens | fp ppl {:.3} ({:.0}s)\n",
        fp.exp(),
        started.elapsed().as_secs_f64()
    );
    println!(
        "{:<24} {:>7} {:>9} {:>10} {:>10} {:>10} {:>10}",
        "structure", "bits/w", "ppl", "ΔL(+E)", "ΔL(−E)", "even", "odd"
    );
    let mut arms = Vec::new();
    for &t in &ts {
        arms.extend([
            Arm::Ladder(t),
            Arm::Golay(t),
            Arm::TimeShare(t),
            Arm::Uniform(if t == 1 { 5 } else { 15 }),
        ]);
        if !ts.contains(&(t + 1)) {
            arms.push(Arm::Ladder(t + 1));
        }
    }
    for arm in arms {
        let started = Instant::now();
        let cache = QCache::default();
        let (plus_nll, params, bits) = mean_nll(&dir, Some(arm), false, &windows, &code, &cache);
        let (minus_nll, _, _) = mean_nll(&dir, Some(arm), true, &windows, &code, &cache);
        let (plus, minus) = (plus_nll - fp, minus_nll - fp);
        println!(
            "{:<24} {:>7.3} {:>9.3} {:>+10.5} {:>+10.5} {:>+10.5} {:>+10.5}   ({:.0}s)",
            arm.label(),
            bits / params as f64,
            plus_nll.exp(),
            plus,
            minus,
            0.5 * (plus + minus),
            0.5 * (plus - minus),
            started.elapsed().as_secs_f64()
        );
    }
}
