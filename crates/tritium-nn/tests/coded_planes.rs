//! **Is the ternary plane the best additive structure? Coded trit-planes (H8, first-principles).**
//!
//! A SALT ladder with `T` planes is a uniform scalar quantizer with `3^T` levels per group. The
//! structure audit (`docs/plans/0056`) measured, on SmolLM2's real Hadamard-rotated weights, that
//! the level placement is already right — free per-plane scales and Lloyd codebooks gain ≤ 0.2 dB —
//! and that the rotated weights are Gaussian to within 0.03 dB. What a scalar structure cannot
//! have is the granular (cell-shape) gain of a vector quantizer, up to 1.53 dB.
//!
//! Ternary structure can reach it without giving up multiply-free arithmetic. Construction A over
//! `F₃`: constrain the FINEST trit plane of every 12-weight block to be a codeword of the extended
//! ternary Golay code `[12, 6, 6]`. The reconstruction is `Δ·(3k + c)` with `k` the usual `T` coarse
//! planes and `c ∈ C`, i.e. an ordinary `(T+1)`-plane ternary tensor whose last plane happens to be a
//! codeword — so the kernel is unchanged, one more plane of add/subtract. Storage is `T + ½` trits a
//! weight (6 message trits per 12-block; the parity trits are mod-3 sums of them). The lattice
//! `C + 3ℤ¹²` has normalized second moment 0.0705 (0.73 dB over ℤ, better than E₈'s 0.65).
//!
//! Decoding is exact: per block and per codeword the coarse digits round independently, so the
//! nearest point is a minimum over 729 candidates. Blocks run along the row and may straddle g128
//! scale groups; each coordinate uses its own group's step (a diagonal-metric decode), and the steps
//! are refit by least squares after the first decode.
//!
//! Arms, all in the g128 group-Hadamard basis on the folded model, fp activations: the ladder at
//! `T`, Golay `T+½`, and the only uncoded way to reach `T+½` trits — time-sharing, alternate rows at
//! `T` and `T+1`. Scored with the antithetic split (`even` = curvature cost, sample term cancelled).
//!
//! # Measured 2026-09-24 — SmolLM2-135M, full WikiText-2 eval (32,768 tokens), fp 22.675
//!
//! ```text
//! structure               bits/w       ppl       even (curvature)   SNR dB
//! ladder T=1               1.710  22506.406      +8.09769            8.40
//! uniform 5-level scalar   2.447    219.137      +2.28345           12.18
//! Golay-coded 1+½          2.502    129.231      +1.79460           13.10
//! time-share 1/2           2.502   2321.279      +5.64457           10.81
//! ladder T=2               3.295     46.405      +0.71623           16.52
//! uniform 15-level scalar  4.032     28.396      +0.23924           20.76
//! Golay-coded 2+½          4.087     27.362      +0.19744           21.38
//! time-share 2/3           4.087     32.683      +0.41632           19.01
//! ladder T=3               4.880     24.278      +0.07746           25.11
//! ```
//!
//! **Coding the finest plane beats every uncoded structure at the same rate.** Against the ternary
//! fallback (time-sharing) it is 18x better in perplexity at 2.50 bpw and -16% at 4.09 bpw (curvature
//! cost -68% / -53%). Against the fairest uncoded rival — the best single scalar at that rate, which
//! is not trit-plane storable — it is -41% / -3.6% perplexity and -21% / -17% curvature cost while
//! spending 0.055 bpw more; net of that premium the gain is ~0.3-0.6 dB, which is what a lattice with
//! 0.73 dB of granular gain should deliver. The finest plane stays a ternary plane, so the kernel is
//! unchanged (one more add/subtract plane); only storage changes (6 message trits per 12 weights).
//! Plain nearest-point decoding only — no GPTQ/BlockLDLQ yet. The odd (sample) term is small at this
//! eval size (|odd| ≤ 0.05 nats except the destroyed T=1 arms).
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test coded_planes -- --ignored --nocapture
//! ```
//! Knobs: `TRITIUM_H8_TS` (coarse planes for the coded arms, default `1,2`),
//! `TRITIUM_H8_EVAL_WINDOWS` (default 64), `TRITIUM_H8_ARMS=uniform` (single-scalar comparators).

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{Calib, calibrate, extract, fold, perplexity_windowed};
use rayon::prelude::*;
use tritium_nn::ModelRunner;
use tritium_train::ops::ste::{self, RotationPolicy};

const WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 4;
const GROUP: usize = 128;
const GRID: usize = 16;
const BLOCK: usize = 12;

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

/// The 729 codewords of the extended ternary Golay code, balanced digits `{-1, 0, 1}`.
///
/// Generator `[I₆ | A]` with `A` the standard symmetric circulant-bordered form.
fn golay_codewords() -> Vec<[i8; BLOCK]> {
    const A: [[u8; 6]; 6] = [
        [0, 1, 1, 1, 1, 1],
        [1, 0, 1, 2, 2, 1],
        [1, 1, 0, 1, 2, 2],
        [1, 2, 1, 0, 1, 2],
        [1, 2, 2, 1, 0, 1],
        [1, 1, 2, 2, 1, 0],
    ];
    let mut out = Vec::with_capacity(729);
    for m in 0..729u32 {
        let msg: Vec<u8> = (0..6).map(|i| ((m / 3u32.pow(i)) % 3) as u8).collect();
        let mut word = [0i8; BLOCK];
        for i in 0..6 {
            word[i] = msg[i] as i8;
        }
        for j in 0..6 {
            let s: u32 = (0..6).map(|i| u32::from(msg[i]) * u32::from(A[i][j])).sum();
            word[6 + j] = (s % 3) as i8;
        }
        for d in &mut word {
            if *d == 2 {
                *d = -1;
            }
        }
        out.push(word);
    }
    out
}

fn kmax(t: usize) -> i64 {
    (3i64.pow(t as u32) - 1) / 2
}

fn hadamard_blocks(row: &mut [f32]) {
    for b in row.chunks_mut(GROUP) {
        if b.len().is_power_of_two() && b.len() > 1 {
            ste::fast_hadamard(b);
        }
    }
}

/// Nearest point of the scaled lattice for every 12-block of `y`: codes in step units, and the
/// squared error. Per codeword the coarse digits round independently, so the minimum over the 729
/// codewords is exact (up to the coarse range clamp).
fn golay_decode(y: &[f32], steps: &[f64], k: f64, code: &[[i8; BLOCK]]) -> (Vec<i64>, f64) {
    let cols = y.len();
    let mut units = vec![0i64; cols];
    let mut total = 0.0;
    for b0 in (0..cols).step_by(BLOCK) {
        let mut best = (f64::INFINITY, 0usize);
        for (ci, c) in code.iter().enumerate() {
            let mut cost = 0.0;
            for i in 0..BLOCK {
                let d = steps[(b0 + i) / GROUP];
                let x = f64::from(y[b0 + i]);
                let e = if d > 0.0 {
                    let kk = ((x / d - f64::from(c[i])) / 3.0).round().clamp(-k, k);
                    x - d * (3.0 * kk + f64::from(c[i]))
                } else {
                    x
                };
                cost += e * e;
            }
            if cost < best.0 {
                best = (cost, ci);
            }
        }
        total += best.0;
        let c = &code[best.1];
        for i in 0..BLOCK {
            let d = steps[(b0 + i) / GROUP];
            units[b0 + i] = if d > 0.0 {
                let kk = ((f64::from(y[b0 + i]) / d - f64::from(c[i])) / 3.0)
                    .round()
                    .clamp(-k, k);
                3 * kk as i64 + i64::from(c[i])
            } else {
                0
            };
        }
    }
    (units, total)
}

/// Golay-coded `T + ½` quantization of one rotated row: returns the reconstruction.
///
/// Steps start from the best uniform step of the `T+1`-plane grid the lattice sits on, scaled by a
/// per-row multiplier searched over a small grid (the lattice is sparser than that grid, so its
/// best step is larger), then refit per group by least squares given the codes.
fn golay_row(y: &[f32], t: usize, code: &[[i8; BLOCK]]) -> Vec<f32> {
    let cols = y.len();
    assert!(
        cols.is_multiple_of(BLOCK),
        "row width {cols} is not a whole number of 12-blocks"
    );
    let k = kmax(t) as f64;
    let base: Vec<f64> = y
        .chunks(GROUP)
        .map(|g| f64::from(ste::ladder_step(g, t + 1, GRID)))
        .collect();
    let (mut steps, mut units) = [1.0, 1.15, 1.3, 1.45, 1.6]
        .iter()
        .map(|&f| {
            let steps: Vec<f64> = base.iter().map(|d| d * f).collect();
            let (units, err) = golay_decode(y, &steps, k, code);
            (err, steps, units)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, s, u)| (s, u))
        .expect("non-empty grid");
    for _ in 0..2 {
        let mut refit = steps.clone();
        for (g, step) in refit.iter_mut().enumerate() {
            let (lo, hi) = (g * GROUP, ((g + 1) * GROUP).min(cols));
            let num: f64 = (lo..hi).map(|j| f64::from(y[j]) * units[j] as f64).sum();
            let den: f64 = (lo..hi).map(|j| (units[j] as f64).powi(2)).sum();
            if den > 0.0 && num > 0.0 {
                *step = num / den;
            }
        }
        let (u, _) = golay_decode(y, &refit, k, code);
        steps = refit;
        units = u;
    }
    units
        .iter()
        .enumerate()
        .map(|(j, &r)| (steps[j / GROUP] * r as f64) as f32)
        .collect()
}

#[derive(Clone, Copy)]
enum Arm {
    Ladder(usize),
    Golay(usize),
    TimeShare(usize),
    /// A uniform scalar quantizer with `L` levels (odd), packed at `log2 L` bits: the best uncoded
    /// SINGLE scalar at a fractional ternary rate (5 levels ≈ 1+½ trits, 15 ≈ 2+½).
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
    /// Stored bits per weight: trits plus one fp16 step per 128-group.
    fn bits(self) -> f64 {
        let payload = match self {
            Arm::Ladder(t) => t as f64 * 3f64.log2(),
            Arm::Golay(t) | Arm::TimeShare(t) => (t as f64 + 0.5) * 3f64.log2(),
            Arm::Uniform(l) => (l as f64).log2(),
        };
        payload + 16.0 / GROUP as f64
    }
}

fn quantize(w: &[f32], rows: usize, cols: usize, arm: Arm, code: &[[i8; BLOCK]]) -> Vec<f32> {
    let ladder = |t: usize, x: &[f32], r: usize| {
        ste::salt_quantize_forward_grouped_geometric(
            x,
            r,
            cols,
            t,
            GROUP,
            GRID,
            RotationPolicy::Always,
        )
    };
    match arm {
        Arm::Ladder(t) => ladder(t, w, rows),
        Arm::TimeShare(t) => {
            let mut out = vec![0.0f32; w.len()];
            out.par_chunks_mut(cols)
                .zip(w.par_chunks(cols))
                .enumerate()
                .for_each(|(r, (o, x))| o.copy_from_slice(&ladder(t + r % 2, x, 1)));
            out
        }
        Arm::Uniform(levels) => {
            let k = ((levels - 1) / 2) as f32;
            let mut out = vec![0.0f32; w.len()];
            out.par_chunks_mut(cols)
                .zip(w.par_chunks(cols))
                .for_each(|(o, x)| {
                    let mut y = x.to_vec();
                    hadamard_blocks(&mut y);
                    for (og, g) in o.chunks_mut(GROUP).zip(y.chunks(GROUP)) {
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
                        for (q, v) in og.iter_mut().zip(g) {
                            *q = if d > 0.0 {
                                (v / d).round().clamp(-k, k) * d
                            } else {
                                0.0
                            };
                        }
                    }
                    hadamard_blocks(o);
                });
            out
        }
        Arm::Golay(t) => {
            let mut out = vec![0.0f32; w.len()];
            out.par_chunks_mut(cols)
                .zip(w.par_chunks(cols))
                .for_each(|(o, x)| {
                    let mut y = x.to_vec();
                    hadamard_blocks(&mut y);
                    let mut rec = golay_row(&y, t, code);
                    hadamard_blocks(&mut rec); // normalized H is its own inverse
                    o.copy_from_slice(&rec);
                });
            out
        }
    }
}

#[test]
fn the_code_is_the_extended_ternary_golay_code() {
    let code = golay_codewords();
    assert_eq!(code.len(), 729);
    let mut weights = [0usize; BLOCK + 1];
    for w in &code {
        weights[w.iter().filter(|&&d| d != 0).count()] += 1;
    }
    // Weight enumerator of the [12, 6, 6] extended ternary Golay code.
    assert_eq!(weights[0], 1);
    assert_eq!(weights[6], 264);
    assert_eq!(weights[9], 440);
    assert_eq!(weights[12], 24);
}

#[test]
#[ignore = "H8 coded trit-planes; needs SmolLM2-135M; run explicitly"]
fn golay_coded_planes_against_the_ladder() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let ts: Vec<usize> = std::env::var("TRITIUM_H8_TS")
        .unwrap_or_else(|_| "1,2".into())
        .split(',')
        .map(|s| s.trim().parse().expect("T"))
        .collect();
    let n_eval: usize = std::env::var("TRITIUM_H8_EVAL_WINDOWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let code = golay_codewords();
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    let eval = &eval[..(n_eval * WINDOW).min(eval.len())];
    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(&fp, &arch, &train[w * WINDOW..(w + 1) * WINDOW], &mut calib);
    }
    let (folded, arch) = fold(&fp, &shapes, &arch, &calib, 0.75);
    let nll = |w: &[Vec<f32>]| perplexity_windowed(w, &arch, eval, WINDOW).ln();
    let base = nll(&folded);
    println!(
        "SmolLM2-135M | fold α=0.75 | g{GROUP} group-Hadamard | fp activations | {} eval tokens | fp ppl {:.3}\n",
        eval.len(),
        base.exp()
    );
    println!(
        "{:<22} {:>7} {:>9} {:>10} {:>10} {:>10} {:>10}  {:>11}",
        "structure", "bits/w", "ppl", "ΔL(+E)", "ΔL(−E)", "even", "odd", "SNR dB"
    );
    let mut arms = Vec::new();
    for &t in &ts {
        if arms
            .last()
            .is_none_or(|a| !matches!(a, Arm::Ladder(x) if *x == t))
        {
            arms.push(Arm::Ladder(t));
        }
        arms.extend([Arm::Golay(t), Arm::TimeShare(t), Arm::Ladder(t + 1)]);
    }
    // `TRITIUM_H8_ARMS=uniform` runs only the single-scalar comparators at the Golay rates.
    if std::env::var("TRITIUM_H8_ARMS").is_ok_and(|v| v == "uniform") {
        arms = ts
            .iter()
            .map(|&t| Arm::Uniform(if t == 1 { 5 } else { 15 }))
            .collect();
    }
    for arm in arms {
        let started = Instant::now();
        let q: Vec<Vec<f32>> = folded
            .iter()
            .zip(&shapes)
            .map(|(w, &(rows, cols))| quantize(w, rows, cols, arm, &code))
            .collect();
        let (mut sig, mut err) = (0.0f64, 0.0f64);
        for (w, qv) in folded.iter().zip(&q) {
            for (a, b) in w.iter().zip(qv) {
                sig += f64::from(*a) * f64::from(*a);
                err += f64::from(a - b) * f64::from(a - b);
            }
        }
        let anti: Vec<Vec<f32>> = folded
            .iter()
            .zip(&q)
            .map(|(w, qv)| w.iter().zip(qv).map(|(a, b)| 2.0 * a - b).collect())
            .collect();
        let (plus, minus) = (nll(&q) - base, nll(&anti) - base);
        println!(
            "{:<22} {:>7.3} {:>9.3} {:>+10.5} {:>+10.5} {:>+10.5} {:>+10.5}  {:>11.2}   ({:.0}s)",
            arm.label(),
            arm.bits(),
            (base + plus).exp(),
            plus,
            minus,
            0.5 * (plus + minus),
            0.5 * (plus - minus),
            10.0 * (sig / err).log10(),
            started.elapsed().as_secs_f64()
        );
    }
}
