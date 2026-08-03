//! **How much is left in the encoding, and where?**
//!
//! With the balanced-ternary ladder there are no residuals any more. The fit is
//! `k = clamp(round(w/Δ))` followed by base-3 digit extraction — the planes are a **positional
//! numeral system for a scalar quantizer's index**, not a residual expansion. So "encode the
//! residuals better" is no longer the right question. The right question is: given `T` planes, which
//! is `3^T` reconstruction levels, how close is our level placement to the best possible?
//!
//! Three fitters at the same `T`, so the level BUDGET is identical and only the placement differs:
//!
//! | arm | levels | what it optimises |
//! |---|---|---|
//! | ladder | `3^T` uniform, one free parameter (`Δ`) | SSE over a deterministic `Δ` grid |
//! | joint free-scale | `≤3^T`, `T` free parameters | `fit_joint_ternary`: OA-EM over all `T` scales |
//! | **optimal scalar** | `3^T` **fully free** | the exact scalar optimum, by DP — `3^T` unconstrained levels |
//!
//! The optimal scalar quantizer is the oracle. It is **not representable** as `Σ_p s_p·trit_p` (it needs `3^T` free
//! levels where an additive ternary stack has only `T` free parameters), so it cannot ship — it
//! exists to bound what *any* `T`-plane scheme could reach. If the ladder is close to it, level
//! placement is a solved problem. If the oracle is far ahead, the additive family is leaving real
//! distortion on the table — and since a free codebook cannot be decoded multiply-free, that gap is
//! the PRICE OF MULTIPLY-FREE rather than a fitter defect to go fix.
//!
//! MEASURED: the gap grows with `T` — 1.05x at T=1, 1.49x at T=2, 3.21x at T=3 — and the free-scale
//! joint fitter, which searches the whole `T`-parameter family, only narrows it to 2.85x. So no
//! cleverer scale rule closes it.
//!
//! This also closes a hole I left: the ladder has never been compared head-to-head against the
//! free-scale joint fitter on the same corpus. The published 1.079× joint number and the ladder's
//! 1.090× were measured in different runs against different fp baselines, so they are not
//! comparable, and I should not have implied they were.
//!
//! **Weight-space SSE is the right proxy *for this specific question*** — it is literally the
//! objective all three fitters minimise, so comparing them on it is not a proxy-gap violation. It
//! would be one the moment any of this is used to predict perplexity, which it is not.
//!
//! Run:
//! ```text
//! cargo test -p tritium-nn --release --test salt_encoding_headroom -- --ignored --nocapture
//! ```

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold};
use tritium_nn::ModelRunner;
use tritium_quantize::{JointFitConfig, JointFitMetric, fit_joint_ternary};
use tritium_train::ops::ste::{self, RotationPolicy};

const GROUP: usize = 128;
const CALIB_WINDOWS: usize = 8;
const CALIB_SEQ: usize = 512;
const GRID: usize = 16;
const FOLD_ALPHA: f64 = 0.75;
/// Groups sampled per tensor. The joint fitter is `3^T` enumeration × EM restarts per group, so a
/// full sweep is hours; a stride-sampled few thousand groups pins the ratios to well under a percent.
const SAMPLE_STRIDE: usize = 97;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = std::env::var("TRITIUM_MODEL_DIR")
        .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m"));
    PathBuf::from(dir)
}

fn corpus_train() -> Vec<u32> {
    let path = std::env::var("TRITIUM_CORPUS").unwrap_or_else(|_| {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tools/reference/heldout_corpus.json"
        )
        .to_string()
    });
    let j: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("corpus json")).expect("parse");
    j["train_ids"]
        .as_array()
        .expect("train_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect()
}

fn sse(recon: &[f32], target: &[f32]) -> f64 {
    recon
        .iter()
        .zip(target)
        .map(|(&a, &b)| f64::from(a - b) * f64::from(a - b))
        .sum()
}

/// The **exact** minimum-SSE `levels`-point scalar quantizer for this group, by dynamic programming.
///
/// This is used as an ORACLE — a bound the shippable fitters are measured against — so it must be
/// exact, not a heuristic. Lloyd iteration is not good enough: at 27 levels on a 128-sample group it
/// reliably sticks in a local optimum *worse than both real fitters*, which produces a table where
/// the bound is beaten by the things it bounds. (That happened; the assertion in the caller now
/// catches it.)
///
/// 1-D k-means is exactly solvable because an optimal cluster is a contiguous interval of the sorted
/// samples. With prefix sums giving interval cost in O(1):
///
/// ```text
/// D[k][j] = min over i of  D[k-1][i-1] + cost(i..j)
/// cost(i..j) = Σx² − (Σx)²/count      (SSE about the interval mean)
/// ```
///
/// O(K·n²) = 27·128² ≈ 442k operations per group — a few tens of microseconds, and no local optima.
/// `levels ≥ n` returns exactly 0 (one centroid per sample), the correct bound and a self-check.
fn optimal_scalar_quantizer_sse(bs: &[f32], levels: usize) -> f64 {
    let n = bs.len();
    if n == 0 || levels == 0 {
        return 0.0;
    }
    let mut x: Vec<f64> = bs.iter().map(|&v| f64::from(v)).collect();
    x.sort_by(f64::total_cmp);
    if levels >= n {
        return 0.0;
    }

    // 1-indexed prefix sums of x and x².
    let mut s1 = vec![0.0f64; n + 1];
    let mut s2 = vec![0.0f64; n + 1];
    for i in 0..n {
        s1[i + 1] = s1[i] + x[i];
        s2[i + 1] = s2[i] + x[i] * x[i];
    }
    // SSE of the sorted interval [i..=j] (1-indexed) about its own mean.
    let cost = |i: usize, j: usize| -> f64 {
        let cnt = (j - i + 1) as f64;
        let s = s1[j] - s1[i - 1];
        let q = s2[j] - s2[i - 1];
        (q - s * s / cnt).max(0.0)
    };

    let mut prev = vec![f64::INFINITY; n + 1];
    prev[0] = 0.0;
    let mut cur = vec![f64::INFINITY; n + 1];
    for _k in 1..=levels {
        cur[0] = f64::INFINITY;
        for (j, slot) in cur.iter_mut().enumerate().skip(1) {
            let mut best = f64::INFINITY;
            for i in 1..=j {
                let head = prev[i - 1];
                if head.is_finite() {
                    let c = head + cost(i, j);
                    if c < best {
                        best = c;
                    }
                }
            }
            *slot = best;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[n]
}

/// Shannon entropy (bits/symbol) of a base-3-encoded plane stack for one group.
fn symbol_entropy(planes: &[Vec<i8>]) -> f64 {
    let n = planes[0].len();
    let mut hist: HashMap<u32, u64> = HashMap::new();
    for i in 0..n {
        let mut sym = 0u32;
        let mut radix = 1u32;
        for p in planes {
            sym += u32::try_from(p[i] + 1).expect("trit") * radix;
            radix *= 3;
        }
        *hist.entry(sym).or_default() += 1;
    }
    hist.values()
        .map(|&c| {
            let p = c as f64 / n as f64;
            -p * p.log2()
        })
        .sum()
}

#[test]
#[ignore = "needs SmolLM2-135M; joint fitter is slow; run explicitly"]
fn level_placement_headroom_against_the_scalar_optimum() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let train = corpus_train();
    let mut calib = Calib::new(&arch);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp,
            &arch,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp, _arch) = fold(&fp, &shapes, &arch, &calib, FOLD_ALPHA);

    println!(
        "SmolLM2-135M | fold α={FOLD_ALPHA} | g{GROUP} | every group Hadamard-rotated (the basis all\n\
         three fitters are compared in) | stride-{SAMPLE_STRIDE} group sample\n\n\
         The optimal scalar quantizer is an ORACLE: 3^T fully free levels. It cannot be written as Σ s_p·trit_p, so it\n\
         cannot ship — it bounds what any T-plane scheme could reach.\n"
    );
    println!(
        "{:<4} {:>8} {:>13} {:>13} {:>13} {:>10} {:>10}",
        "T", "levels", "ladder SSE", "joint SSE", "optimal SSE", "ladder/opt", "joint/opt"
    );
    println!("{}", "-".repeat(78));

    for t in 1..=3usize {
        let levels = 3usize.pow(t as u32);
        let cfg = JointFitConfig {
            planes: t,
            ..JointFitConfig::default()
        };
        let (mut s_ladder, mut s_joint, mut s_lm) = (0.0f64, 0.0f64, 0.0f64);
        let (mut h_ladder, mut h_joint) = (0.0f64, 0.0f64);
        let mut groups = 0usize;

        for (w, &(rows, cols)) in fp.iter().zip(&shapes) {
            let per_row = cols.div_ceil(GROUP);
            for g in (0..rows * per_row).step_by(SAMPLE_STRIDE) {
                let (r, b) = (g / per_row, g % per_row);
                let lo = r * cols + b * GROUP;
                let hi = (lo + GROUP).min((r + 1) * cols);
                if hi - lo != GROUP {
                    continue; // ragged tail: never rotated, so not comparable in this basis
                }
                // Compare all three in the ROTATED basis — that is where the ladder is designed to
                // work, and rotation is orthogonal so SSE is preserved either way.
                let mut bs = w[lo..hi].to_vec();
                ste::fast_hadamard(&mut bs);

                let fit =
                    ste::geometric_ladder_fit(&bs, 1, GROUP, t, GROUP, GRID, RotationPolicy::Never);
                let (s0, planes) = &fit[0];
                let mut recon = vec![0.0f32; GROUP];
                for (p, plane) in planes.iter().enumerate() {
                    let s = s0 / 3.0f32.powi(p as i32);
                    for (o, &tr) in recon.iter_mut().zip(plane) {
                        *o += s * f32::from(tr);
                    }
                }
                s_ladder += sse(&recon, &bs);
                h_ladder += symbol_entropy(planes);

                if let Ok(j) = fit_joint_ternary(&bs, JointFitMetric::Identity, cfg) {
                    s_joint += sse(&j.reconstruction, &bs);
                    h_joint += symbol_entropy(&j.trits);
                } else {
                    s_joint += sse(&recon, &bs); // failed fit falls back, same as the sweep does
                    h_joint += symbol_entropy(planes);
                }

                s_lm += optimal_scalar_quantizer_sse(&bs, levels);
                groups += 1;
            }
        }

        // The oracle has `3^T` FREE levels; both fitters are restricted to `3^T` levels generated by
        // `T` parameters. So the oracle cannot lose. If it does, Lloyd has stuck in a local optimum
        // and every ratio in this table is meaningless — fail rather than print it. (This fired on
        // the first version, which never re-seeded empty cells.)
        assert!(
            s_lm <= s_ladder * 1.001 && s_lm <= s_joint * 1.001,
            "T={t}: optimal-quantizer oracle ({s_lm:.4e}) lost to ladder ({s_ladder:.4e}) or joint \
             ({s_joint:.4e}) — the oracle is stuck, not the fitters winning"
        );
        println!(
            "{t:<4} {levels:>8} {s_ladder:>13.4e} {s_joint:>13.4e} {s_lm:>13.4e} {:>9.3}x {:>9.3}x",
            s_ladder / s_lm,
            s_joint / s_lm,
        );
        println!(
            "     symbol entropy: ladder {:.3} bits, joint {:.3} bits (of {:.3} dense) over {groups} groups",
            h_ladder / groups as f64,
            h_joint / groups as f64,
            t as f64 * 3.0f64.log2(),
        );
    }

    println!(
        "\nWhat the ratios mean. They GROW with T (≈1.05x at T=1, ≈1.5x at T=2, ≈3x at T=3), so the\n\
         additive-ternary family falls further behind free scalar quantization the more planes it is\n\
         given. Three free scales simply cannot place 27 levels well: the reachable levels are\n\
         subset-sums, which is a rigid 3-parameter family, and no cleverer SCALE RULE escapes it —\n\
         the joint free-scale fitter searches that whole family and still sits at ~2.85x.\n\n\
         That gap is the price of multiply-free decode, and it is worth naming as such. A free\n\
         codebook needs `Σ_k act[k]·c[idx[k]]` — a real multiply per element — whereas\n\
         `Σ_p s_p·(Σ_k act[k]·t_p[k])` keeps the inner loop add/sub/skip and pays only T multiplies\n\
         per output. You cannot have both; this table prices the trade at ~5 dB at T=3.\n\n\
         Note also the entropy column against the SSE column: the joint fitter buys ~13% lower SSE\n\
         for ~0.32 more bits/weight of symbol entropy. At ~6 dB/bit that is a losing trade, so the\n\
         uniform ladder is ahead on RATE-DISTORTION even where it is behind on SSE — the\n\
         entropy-constrained-scalar-quantization result showing up in real weights."
    );
}
