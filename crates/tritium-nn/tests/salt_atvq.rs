//! **Additive ternary vector quantization — a different rotation per plane.**
//!
//! SALT's `T` planes all live in one Hadamard basis, so the reachable reconstructions form a cubic
//! grid: with the 1/3 ladder, `T` planes are exactly a uniform scalar quantizer with `3^T` levels.
//! That caps the rate–distortion at the **scalar** bound. Vector quantizers (AQLM's codebooks,
//! QuIP#'s lattices, QTIP's trellises) beat it, but decode through lookups and multiplies.
//!
//! This keeps decode multiply-free and escapes the cube. Plane `p` gets its own fixed rotation
//! `R_p = H·D_p` — the group's Walsh–Hadamard transform after a plane-specific sign pattern `D_p`:
//!
//! ```text
//! Ŵ_g = Σ_p  R_p · (s_p · t_p)        t_p ternary, s_p one scale per plane per group
//! ```
//!
//! Decode is still additions and subtractions: a sign flip, a fast Hadamard, a ternary matmul. What
//! the extra rotations buy is **decorrelation between planes**. In one shared basis the residual left
//! by plane `p` is cube-aligned, which is why greedy free-scale fitting wasted most of each plane's
//! rate before the ladder replaced it. In a fresh basis the same residual looks close to Gaussian,
//! so every plane gets a well-conditioned target.
//!
//! **Resident memory is identical to SALT at the same `T`**: `T` planes of trits and one scale per
//! plane per group. The only runtime cost is `T` Hadamards per group instead of one.
//!
//! # Prediction, recorded before the first run
//!
//! An optimal ternary fit of a Gaussian removes about **7.2 dB** per plane. The ladder's uniform grid
//! does better than that at high resolution. So: **tie at T=1** (identical construction), **close at
//! T=2**, **the ladder wins at T≥3**. A win, if there is one, is at low `T` — exactly the regime where
//! SALT is weakest.
//!
//! # What this measures
//!
//! Weight-space relative Frobenius error on every tensor of the folded model, `g128`, both methods in
//! their rotated form, both charged the same scale overhead (one f16 per plane per group). This is
//! the screen; perplexity is the verdict, and only worth running where the screen shows a gap.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test salt_atvq -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold};
use rayon::prelude::*;
use tritium_nn::ModelRunner;
use tritium_train::ops::ste::{self, RotationPolicy, fast_hadamard, group_is_rotatable};

const CALIB_WINDOWS: usize = 4;
const CALIB_SEQ: usize = 512;
const GROUP: usize = 128;
const GRID: usize = 16;
const REFINE_SWEEPS: usize = 6;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR")
            .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m")),
    )
}

fn corpus_train() -> Vec<u32> {
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

/// Plane `p`'s sign pattern over an `n`-wide group. Plane 0 has none, so `R_0` is exactly the
/// Hadamard SALT already uses and `T=1` is the same construction in both methods.
fn signs(p: usize, n: usize) -> Vec<f32> {
    if p == 0 {
        return vec![1.0; n];
    }
    let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ (p as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if s & 1 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// `y ← R_pᵀ·x = D_p·H·x`.
fn analyze(x: &[f32], d: &[f32], out: &mut [f32]) {
    out.copy_from_slice(x);
    if group_is_rotatable(out.len()) {
        fast_hadamard(out);
        for (o, &s) in out.iter_mut().zip(d) {
            *o *= s;
        }
    }
}

/// `x ← R_p·y = H·D_p·y`.
fn synthesize(y: &[f32], d: &[f32], out: &mut [f32]) {
    if group_is_rotatable(out.len()) {
        for ((o, &v), &s) in out.iter_mut().zip(y).zip(d) {
            *o = v * s;
        }
        fast_hadamard(out);
    } else {
        out.copy_from_slice(y);
    }
}

/// Exact minimum-SSE ternary fit `y ≈ s·t`: keep the `m` largest-magnitude entries with the sign of
/// `y`, scale `s` = their mean magnitude, `m` chosen to maximize `(Σ top-m |y|)² / m`.
fn ternary_fit(y: &[f32], out: &mut [f32]) {
    let mut mag: Vec<(f32, usize)> = y.iter().enumerate().map(|(i, &v)| (v.abs(), i)).collect();
    mag.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut best_m, mut best_gain, mut acc) = (0usize, 0.0f64, 0.0f64);
    for (m, &(v, _)) in mag.iter().enumerate() {
        acc += f64::from(v);
        let gain = acc * acc / (m + 1) as f64;
        if gain > best_gain {
            best_gain = gain;
            best_m = m + 1;
        }
    }
    out.fill(0.0);
    if best_m == 0 {
        return;
    }
    let s = (mag[..best_m]
        .iter()
        .map(|&(v, _)| f64::from(v))
        .sum::<f64>()
        / best_m as f64) as f32;
    for &(_, i) in &mag[..best_m] {
        out[i] = s * y[i].signum();
    }
}

/// Fit one group with `t` differently-rotated ternary planes. Greedy, then alternating refinement:
/// each sweep removes one plane, refits it exactly against what the others leave, and puts it back —
/// every step can only lower the error.
fn atvq_group(w: &[f32], t: usize, sweeps: usize) -> Vec<f32> {
    let n = w.len();
    let ds: Vec<Vec<f32>> = (0..t).map(|p| signs(p, n)).collect();
    let mut contrib = vec![vec![0.0f32; n]; t];
    let mut resid = w.to_vec();
    let (mut y, mut fit) = (vec![0.0f32; n], vec![0.0f32; n]);
    let mut step = |p: usize, resid: &mut Vec<f32>, contrib: &mut Vec<Vec<f32>>| {
        for (r, &c) in resid.iter_mut().zip(&contrib[p]) {
            *r += c;
        }
        analyze(resid, &ds[p], &mut y);
        ternary_fit(&y, &mut fit);
        synthesize(&fit, &ds[p], &mut contrib[p]);
        for (r, &c) in resid.iter_mut().zip(&contrib[p]) {
            *r -= c;
        }
    };
    for p in 0..t {
        step(p, &mut resid, &mut contrib);
    }
    for _ in 0..sweeps {
        for p in 0..t {
            step(p, &mut resid, &mut contrib);
        }
    }
    let mut out = vec![0.0f32; n];
    for c in &contrib {
        for (o, &v) in out.iter_mut().zip(c) {
            *o += v;
        }
    }
    out
}

fn atvq_tensor(w: &[f32], cols: usize, t: usize, sweeps: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w.len()];
    out.par_chunks_mut(cols)
        .zip(w.par_chunks(cols))
        .for_each(|(o, row)| {
            for (og, g) in o.chunks_mut(GROUP).zip(row.chunks(GROUP)) {
                og.copy_from_slice(&atvq_group(g, t, sweeps));
            }
        });
    out
}

/// Sum of squared error and of squared weight across the model.
fn frob(fp: &[Vec<f32>], q: &[Vec<f32>]) -> f64 {
    let (mut se, mut sw) = (0.0f64, 0.0f64);
    for (a, b) in fp.iter().zip(q) {
        for (&x, &y) in a.iter().zip(b) {
            se += f64::from(x - y) * f64::from(x - y);
            sw += f64::from(x) * f64::from(x);
        }
    }
    (se / sw).sqrt()
}

/// The optimal ternary fit must never lose to any other ternary vector, and one plane with no sign
/// pattern must reproduce a plain Hadamard-domain ternary fit — the degenerate control.
#[test]
fn ternary_fit_is_optimal_and_one_plane_is_the_shared_basis() {
    let mut s = 0x1234_5678u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / 8_388_608.0) - 1.0
    };
    let y: Vec<f32> = (0..16).map(|_| next()).collect();
    let mut fit = vec![0.0f32; 16];
    ternary_fit(&y, &mut fit);
    let sse = |f: &[f32]| y.iter().zip(f).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
    let best = sse(&fit);
    // Exhaustive over supports by magnitude order is what the fit claims; check against brute force
    // over every sign-consistent support size with the optimal scale.
    for m in 0..=16usize {
        let mut idx: Vec<usize> = (0..16).collect();
        idx.sort_by(|&a, &b| y[b].abs().total_cmp(&y[a].abs()));
        let mut f = vec![0.0f32; 16];
        if m > 0 {
            let sc = idx[..m].iter().map(|&i| y[i].abs()).sum::<f32>() / m as f32;
            for &i in &idx[..m] {
                f[i] = sc * y[i].signum();
            }
        }
        assert!(
            best <= sse(&f) + 1e-5,
            "support {m} beats the claimed optimum"
        );
    }

    let w: Vec<f32> = (0..128).map(|_| next()).collect();
    let one = atvq_group(&w, 1, 0);
    let mut y2 = w.clone();
    fast_hadamard(&mut y2);
    let mut f2 = vec![0.0f32; 128];
    ternary_fit(&y2, &mut f2);
    fast_hadamard(&mut f2);
    for (a, b) in one.iter().zip(&f2) {
        assert!(
            (a - b).abs() < 1e-5,
            "one plane is not the shared-basis ternary fit"
        );
    }
    // Refinement can only lower the error.
    let sse_w = |q: &[f32]| w.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
    assert!(sse_w(&atvq_group(&w, 3, 6)) <= sse_w(&atvq_group(&w, 3, 0)) + 1e-5);
}

#[test]
#[ignore = "needs SmolLM2-135M; fits every tensor at T=1..4 with both methods"]
fn per_plane_rotations_against_the_ladder_in_weight_space() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch0, fp0, shapes) = extract(&runner);
    let train = corpus_train();
    let mut calib = Calib::new(&arch0);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp0,
            &arch0,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp, _arch) = fold(&fp0, &shapes, &arch0, &calib, 0.75);

    println!(
        "SmolLM2-135M folded (α=0.75), all {} tensors, g{GROUP}, same scale overhead in every arm\n",
        fp.len()
    );
    println!(
        "{:<4} {:>14} {:>14} {:>14} {:>12} {:>12}",
        "T", "ladder", "ATVQ greedy", "ATVQ refined", "dB vs ladder", "dB/plane"
    );
    println!("{}", "-".repeat(76));
    for t in 1..=4usize {
        let ladder: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(r, c))| {
                ste::salt_quantize_forward_grouped_geometric(
                    w,
                    r,
                    c,
                    t,
                    GROUP,
                    GRID,
                    RotationPolicy::Always,
                )
            })
            .collect();
        let greedy: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(_, c))| atvq_tensor(w, c, t, 0))
            .collect();
        let refined: Vec<Vec<f32>> = fp
            .iter()
            .zip(&shapes)
            .map(|(w, &(_, c))| atvq_tensor(w, c, t, REFINE_SWEEPS))
            .collect();
        let (el, eg, er) = (frob(&fp, &ladder), frob(&fp, &greedy), frob(&fp, &refined));
        let db = |e: f64| -20.0 * e.log10();
        println!(
            "{t:<4} {el:>14.5} {eg:>14.5} {er:>14.5} {:>+11.2} {:>12.2}",
            db(er) - db(el),
            db(er) / t as f64
        );
    }
    println!(
        "\nPositive dB means per-plane rotations reconstruct better than the ladder at the same\n\
         resident bits. Prediction: tie at T=1, close at T=2, ladder ahead at T≥3."
    );
}
