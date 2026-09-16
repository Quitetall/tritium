//! **Lever 2 — make the rate axis continuous.**
//!
//! A plane is a fixed **9× step** applied to all 128 weights of a group at once. That coarseness is
//! why allocation is closed: ~50 arms across four campaigns, four signals up to exact task-loss
//! gradients, and uniform won every time, because there is nothing finer than a plane to spend.
//!
//! And a plane cannot be improved on *as a plane*. Each buys **9.54 dB**, which is
//! `log2(3) × 6.02 dB/bit` — the information-theoretic maximum for 1.585 bits. The ladder is sitting
//! exactly on the scalar rate–distortion bound.
//!
//! So the rate axis has to stop being quantized. A rank-`r` floating-point correction
//! `Ŵ = Q_T(W) + A·B` costs `r(n+k)` values for **any integer `r`**, which is a continuous knob, and
//! it can put precision into a *subspace* rather than into every coordinate of a group.
//!
//! # The bet, stated so it can lose
//!
//! Byte-matched against one more plane:
//!
//! ```text
//! one plane      n·k · 2.0625 bits      (TQ2_0: 2 bits/trit + one f16 scale per 256)
//! rank r in f16  r(n+k) · 16 bits
//! ⇒  r = n·k·2.0625 / (16(n+k))
//! ```
//!
//! For a square 1536×1536 that is **r ≈ 99** — 6.4% of full rank. If the residual after `T` planes
//! were white, rank 99 of 1536 would capture 6.4% of its energy, i.e. **0.29 dB**, against the
//! plane's 9.54 dB. Low rank would lose by a factor of thirty.
//!
//! It only wins if the residual is *concentrated in the directions the activations actually excite*.
//! That is the same anisotropy the AWQ fold exploits, so the premise is not unreasonable — but it is
//! a premise, and this measures it instead of assuming it.
//!
//! # The fit is the exact activation-weighted low-rank approximation
//!
//! Minimize `‖(E − AB)·X‖²` for the residual `E = W − Q_T(W)`, not `‖E − AB‖²`. With `H = E[x xᵀ]`
//! and `H = L·Lᵀ`, that is `‖(E − AB)·L‖_F`, whose optimum is the rank-`r` truncation of `E·L`.
//!
//! Randomized range finding gives it without forming an SVD of a 1536-wide matrix, and it collapses
//! neatly: with `Q = orth(E·L·Ω)`,
//!
//! ```text
//! A·B = Q·(Qᵀ·E·L)·L⁻¹ = Q·Qᵀ·E
//! ```
//!
//! so `L⁻¹` never has to be formed. The metric steers *which* subspace is chosen; the projection
//! onto it is plain.
//!
//! # Arms
//!
//! All three move the same 210 projections and leave the tied embedding at `T` — it has no Gram
//! (see `salt_gptq`), and moving it in one arm but not another would break the byte match.
//!
//! - `T` — the shipping baseline.
//! - `T` + rank-`r` — the same bits as `T+1`, spent as a subspace correction.
//! - `T+1` — the same bits spent as a plane. **This is the arm to beat.**
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test salt_lowrank -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;

use common::{Calib, calibrate, extract, fold, perplexity_windowed};
use rayon::prelude::*;
use tritium_nn::ModelRunner;
use tritium_nn::calibrate::{Tap, forward_aq};
use tritium_train::Tape;
use tritium_train::ops::ste::{self, RotationPolicy};
use tritium_train::tape::ValueId;

const EVAL_WINDOW: usize = 512;
const CALIB_WINDOWS: usize = 4;
const CALIB_SEQ: usize = 512;
const GROUP: usize = 128;
const GRID: usize = 16;
/// 48 × 256 = 12,288 tokens, 8× the 1536-wide FFN intermediate. The first GPTQ run used 1,024 —
/// fewer samples than that Gram has dimensions, so it was rank-deficient by construction. The Gram
/// here only STEERS which subspace the correction spans (the correction itself is a plain
/// projection and cannot overfit the way error compensation can), but a steering metric estimated
/// from too few samples still points at the wrong subspace.
const GRAM_WINDOWS: usize = 48;
const GRAM_SEQ: usize = 256;
const DAMP: f64 = 0.01;
/// TQ2_0 charges 2 bits per trit plus one f16 scale per 256-trit block.
const PLANE_BITS_PER_WEIGHT: f64 = 2.0 + 16.0 / 256.0;
/// The correction is stored in f16, the same precision as every other scale in the container.
const CORRECTION_BITS: f64 = 16.0;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR")
            .unwrap_or_else(|_| format!("{home}/.cache/tritium-models/smollm2-135m")),
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

fn accumulate_gram(h: &mut [f64], k: usize, act: &[f32], seq: usize) {
    h.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        for r in 0..seq {
            let x = &act[r * k..(r + 1) * k];
            let xi = f64::from(x[i]);
            if xi == 0.0 {
                continue;
            }
            for (j, hij) in row.iter_mut().enumerate().skip(i) {
                *hij += xi * f64::from(x[j]);
            }
        }
    });
}

fn mirror_and_scale(h: &mut [f64], k: usize, rows: usize) {
    for i in 0..k {
        for j in 0..i {
            h[i * k + j] = h[j * k + i];
        }
    }
    let n = rows.max(1) as f64;
    for v in h.iter_mut() {
        *v /= n;
    }
}

/// Lower Cholesky of `H + λI`, `λ = damp·mean(diag H)`. Returns `None` if still not PD.
fn damped_cholesky(h: &[f64], k: usize, damp: f64) -> Option<Vec<f32>> {
    let mean_diag: f64 = (0..k).map(|i| h[i * k + i]).sum::<f64>() / k as f64;
    let lambda = damp * mean_diag.max(1e-12);
    let mut l = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = h[i * k + j] + if i == j { lambda } else { 0.0 };
            for p in 0..j {
                sum -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                if sum <= 0.0 || sum.is_nan() {
                    return None;
                }
                l[i * k + i] = sum.sqrt();
            } else {
                l[i * k + j] = sum / l[j * k + j];
            }
        }
    }
    Some(l.iter().map(|&v| v as f32).collect())
}

/// Orthonormalize the `r` columns of `y` (`n × r`, row-major) in place by modified Gram–Schmidt.
/// Returns the number of columns that survived; a rank-deficient column is dropped to zero.
fn orthonormalize(y: &mut [f32], n: usize, r: usize) -> usize {
    let mut kept = 0usize;
    for c in 0..r {
        // Project out everything already kept.
        for p in 0..kept {
            let dot: f32 = (0..n).map(|i| y[i * r + c] * y[i * r + p]).sum();
            for i in 0..n {
                y[i * r + c] -= dot * y[i * r + p];
            }
        }
        let norm: f32 = (0..n)
            .map(|i| y[i * r + c] * y[i * r + c])
            .sum::<f32>()
            .sqrt();
        if norm > 1e-12 {
            for i in 0..n {
                let v = y[i * r + c] / norm;
                y[i * r + kept] = v;
            }
            if kept != c {
                for i in 0..n {
                    y[i * r + c] = 0.0;
                }
            }
            kept += 1;
        } else {
            for i in 0..n {
                y[i * r + c] = 0.0;
            }
        }
    }
    kept
}

/// The rank-`r` activation-weighted correction to `residual`, as a dense `n × k` matrix.
///
/// `Q = orth(E·L·Ω)` and the correction is `Q·Qᵀ·E` — see the module docs for why `L⁻¹` cancels.
fn lowrank_correction(residual: &[f32], n: usize, k: usize, r: usize, l: &[f32]) -> Vec<f32> {
    // Ω, k×r, deterministic so the arm is reproducible.
    let mut s = 0xDEAD_BEEF_1234_5678u64;
    let mut omega = vec![0.0f32; k * r];
    for v in omega.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = ((s >> 40) as f32 / 8_388_608.0) - 1.0;
    }
    // LΩ, k×r. `l` is lower triangular, so row i touches only columns 0..=i.
    let mut lo = vec![0.0f32; k * r];
    lo.par_chunks_mut(r).enumerate().for_each(|(i, out)| {
        for p in 0..=i {
            let lip = l[i * k + p];
            if lip == 0.0 {
                continue;
            }
            for c in 0..r {
                out[c] += lip * omega[p * r + c];
            }
        }
    });
    // Y = E·(LΩ), n×r.
    let mut y = vec![0.0f32; n * r];
    y.par_chunks_mut(r)
        .zip(residual.par_chunks(k))
        .for_each(|(out, e)| {
            for (p, &ep) in e.iter().enumerate() {
                if ep == 0.0 {
                    continue;
                }
                for c in 0..r {
                    out[c] += ep * lo[p * r + c];
                }
            }
        });
    let kept = orthonormalize(&mut y, n, r);
    if kept == 0 {
        return vec![0.0f32; n * k];
    }
    // B = Qᵀ·E, kept×k.
    let mut b = vec![0.0f32; kept * k];
    for (i, e) in residual.chunks(k).enumerate() {
        for c in 0..kept {
            let q = y[i * r + c];
            if q == 0.0 {
                continue;
            }
            let brow = &mut b[c * k..(c + 1) * k];
            for (j, bj) in brow.iter_mut().enumerate() {
                *bj += q * e[j];
            }
        }
    }
    // Q·B, n×k.
    let mut out = vec![0.0f32; n * k];
    out.par_chunks_mut(k).enumerate().for_each(|(i, o)| {
        for c in 0..kept {
            let q = y[i * r + c];
            if q == 0.0 {
                continue;
            }
            let brow = &b[c * k..(c + 1) * k];
            for (j, oj) in o.iter_mut().enumerate() {
                *oj += q * brow[j];
            }
        }
    });
    out
}

fn fit(w: &[f32], rows: usize, cols: usize, t: usize) -> Vec<f32> {
    ste::salt_quantize_forward_grouped_geometric(
        w,
        rows,
        cols,
        t,
        GROUP,
        GRID,
        RotationPolicy::Always,
    )
}

#[test]
#[ignore = "needs SmolLM2-135M; per-tap Grams plus a randomized range finder over every projection"]
fn low_rank_correction_against_one_more_plane() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let t_ref = env_usize("TRITIUM_LR_T", 3);
    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch0, fp0, shapes) = extract(&runner);
    let (train, eval) = corpus();

    let mut calib = Calib::new(&arch0);
    for w in 0..CALIB_WINDOWS {
        calibrate(
            &fp0,
            &arch0,
            &train[w * CALIB_SEQ..(w + 1) * CALIB_SEQ],
            &mut calib,
        );
    }
    let (fp, arch) = fold(&fp0, &shapes, &arch0, &calib, 0.75);

    let n_layers = arch.n_layers;
    let q_width = arch.n_head * arch.head_dim;
    let mut g_attn: Vec<Vec<f64>> = vec![vec![0.0; arch.n_embd * arch.n_embd]; n_layers];
    let mut g_ffn: Vec<Vec<f64>> = vec![vec![0.0; arch.n_embd * arch.n_embd]; n_layers];
    let mut g_down: Vec<Vec<f64>> = vec![vec![0.0; arch.ff * arch.ff]; n_layers];
    let mut g_o: Vec<Vec<f64>> = vec![vec![0.0; q_width * q_width]; n_layers];
    let mut seen = 0usize;
    println!("collecting per-tap Grams over {GRAM_WINDOWS} × {GRAM_SEQ} tokens…");
    for wnd in 0..GRAM_WINDOWS {
        let toks = &train[wnd * GRAM_SEQ..(wnd + 1) * GRAM_SEQ];
        let mut t = Tape::new();
        let wids: Vec<ValueId> = fp.iter().map(|w| t.leaf(w.clone())).collect();
        forward_aq(
            &mut t,
            &wids,
            &arch,
            toks,
            &mut |kind, li, v, seq, cols| match kind {
                Tap::AttnIn => accumulate_gram(&mut g_attn[li], cols, v, seq),
                Tap::FfnIn => accumulate_gram(&mut g_ffn[li], cols, v, seq),
                Tap::DownIn => accumulate_gram(&mut g_down[li], cols, v, seq),
                Tap::OProjIn => accumulate_gram(&mut g_o[li], cols, v, seq),
                Tap::Head => {}
            },
        );
        seen += toks.len();
    }
    for li in 0..n_layers {
        mirror_and_scale(&mut g_attn[li], arch.n_embd, seen);
        mirror_and_scale(&mut g_ffn[li], arch.n_embd, seen);
        mirror_and_scale(&mut g_down[li], arch.ff, seen);
        mirror_and_scale(&mut g_o[li], q_width, seen);
    }

    let base: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(w, &(r, c))| fit(w, r, c, t_ref))
        .collect();
    // The arm to beat: the same bytes spent as a plane, on the same 210 projections.
    let mut plus_plane = base.clone();
    let mut lowrank = base.clone();

    let mut bits_plane = 0.0f64;
    let mut bits_rank = 0.0f64;
    let mut captured = 0.0f64;
    let mut residual_energy = 0.0f64;
    let mut ranks: Vec<usize> = Vec::new();
    println!("fitting corrections…");
    for li in 0..n_layers {
        let b = 1 + 7 * li;
        for (slot, gram) in [
            (0usize, &g_attn[li]),
            (1, &g_attn[li]),
            (2, &g_attn[li]),
            (3, &g_o[li]),
            (4, &g_ffn[li]),
            (5, &g_ffn[li]),
            (6, &g_down[li]),
        ] {
            let i = b + slot;
            let (n, k) = shapes[i];
            plus_plane[i] = fit(&fp[i], n, k, t_ref + 1);
            bits_plane += (n * k) as f64 * PLANE_BITS_PER_WEIGHT;

            // Byte-matched rank: what one plane's bits buy as f16 factors.
            let r = ((n * k) as f64 * PLANE_BITS_PER_WEIGHT / (CORRECTION_BITS * (n + k) as f64))
                .floor() as usize;
            let r = r.max(1).min(n.min(k));
            ranks.push(r);
            bits_rank += (r * (n + k)) as f64 * CORRECTION_BITS;

            let residual: Vec<f32> = fp[i].iter().zip(&base[i]).map(|(&a, &q)| a - q).collect();
            let Some(l) = damped_cholesky(gram, k, DAMP) else {
                continue;
            };
            let corr = lowrank_correction(&residual, n, k, r, &l);
            for (o, (&q, &c)) in lowrank[i].iter_mut().zip(base[i].iter().zip(&corr)) {
                *o = q + c;
            }
            // How much of the residual the correction actually captured, in plain energy.
            for (&e, &c) in residual.iter().zip(&corr) {
                residual_energy += f64::from(e) * f64::from(e);
                captured += f64::from(c) * f64::from(c);
            }
        }
        if li % 10 == 0 {
            println!("  layer {li}/{n_layers}");
        }
    }

    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    let ppl_base = perplexity_windowed(&base, &arch, &eval, EVAL_WINDOW);
    let ppl_plane = perplexity_windowed(&plus_plane, &arch, &eval, EVAL_WINDOW);
    let ppl_lr = perplexity_windowed(&lowrank, &arch, &eval, EVAL_WINDOW);

    let mean_rank = ranks.iter().sum::<usize>() as f64 / ranks.len() as f64;
    println!(
        "\nSmolLM2-135M | WikiText-2 {} held-out | fold α=0.75 | g{GROUP} | rotation always\n\
         210 projections corrected; the tied embedding stays at T={t_ref} in every arm.\n\
         mean byte-matched rank {mean_rank:.0}; correction captured {:.1}% of the residual energy\n\
         bit check: one plane {:.3e} bits vs rank factors {:.3e} bits ({:+.2}%)\n",
        100.0 * captured / residual_energy.max(1e-30),
        bits_plane,
        bits_rank,
        100.0 * (bits_rank - bits_plane) / bits_plane,
        eval.len()
    );
    println!("{:<40} {:>11} {:>10} {:>12}", "arm", "ppl", "× fp", "vs T");
    println!("{}", "-".repeat(78));
    for (label, ppl) in [
        ("fp master", ppl_fp),
        (&format!("T={t_ref} (shipping baseline)"), ppl_base),
        (&format!("T={t_ref} + rank-r f16 correction"), ppl_lr),
        (
            &format!("T={} — the same bytes as a plane", t_ref + 1),
            ppl_plane,
        ),
    ] {
        println!(
            "{label:<40} {ppl:>11.4} {:>9.4}× {:>11}",
            ppl / ppl_fp,
            if (ppl - ppl_base).abs() < 1e-12 {
                "—".to_owned()
            } else {
                format!("{:+.2}%", 100.0 * (ppl - ppl_base) / ppl_base)
            }
        );
    }
    println!(
        "\nlow-rank vs one more plane at matched bytes: {:+.2}%",
        100.0 * (ppl_lr - ppl_plane) / ppl_plane
    );

    assert!(
        (bits_rank - bits_plane).abs() / bits_plane < 0.02,
        "the two arms are not byte-matched: plane {bits_plane:.3e} vs rank {bits_rank:.3e}. An \
         arm that spends more cannot be compared"
    );
    assert!(
        ppl_lr.is_finite() && ppl_lr > 0.0,
        "the low-rank arm did not produce a usable model"
    );
    // The correction is a strict addition to the SAME base, so it cannot be worse than the base
    // unless the fit is wrong — a rank-r projection of the residual can only remove error in the
    // subspace it spans.
    assert!(
        ppl_lr <= ppl_base * 1.02,
        "adding a low-rank correction to the T={t_ref} base made it WORSE ({ppl_base:.4} → \
         {ppl_lr:.4}). The correction is Q·Qᵀ·E with Q orthonormal, which is a projection of the \
         residual, so this means the range finder or the Gram basis is wrong"
    );
}
