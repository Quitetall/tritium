//! **Rank allocation — does allocation work once the step is continuous?**
//!
//! Plane allocation lost every arm across ~50 attempts and five signals, up to exact task-loss
//! gradients. The explanation offered was structural: a plane is a fixed 9× step over all 128
//! weights in a group, and uniform `T` already equalizes relative precision, so nothing finer than a
//! plane exists to redistribute.
//!
//! That explanation makes a prediction this file tests. **Rank is not a coarse step.** A rank-`r`
//! correction on a tensor can take any `r`, and each added rank removes a known, diminishing
//! amount of error. If coarseness was the reason allocation failed, allocating rank should work. If
//! it loses here too, the explanation is wrong.
//!
//! # Priced in resident bytes, not file bytes
//!
//! `TSLJ` is decoded at load, so the file's entropy-coded size says nothing about memory. Every cost
//! here is what sits in RAM: a rank costs `(n + k)·16` bits of f16 factors, and the plane it is
//! compared against costs its real in-memory TQ2_0 rows, **padding included** —
//! `rows · ⌈k/256⌉ · 66` bytes.
//!
//! # The value of a rank is exact, not a proxy over it
//!
//! The optimal rank-`r` correction in the activation metric is the top-`r` truncation of `E·L`
//! (`E` the residual after `T` planes, `H + λI = L·Lᵀ`). Its `i`-th singular value squared is
//! *exactly* the output error the `i`-th rank removes, and singular values are sorted, so each
//! tensor's marginal value is monotone and greedy water-filling by value per bit is optimal for this
//! objective. What is still a modelling choice is how errors in different tensors compare, so two
//! pricings are run:
//!
//! - **absolute** — `σᵢ²` as is.
//! - **relative** — `σᵢ² / ‖W·L‖²`, output error as a fraction of the tensor's own output energy.
//!   The plane campaign found absolute pricing promotes large-norm tensors and starves small ones,
//!   while RMSNorm makes relative precision the meaningful quantity; this arm encodes that lesson.
//!
//! # Arms, at 0.1, 0.25 and 1.0 of a plane's resident bytes
//!
//! - **uniform rank** — the budget spread in proportion to each tensor's plane cost. This, not the
//!   `T` baseline, is what allocation has to beat.
//! - **allocated (absolute)** and **allocated (relative)**.
//! - **anti-allocated** at 0.25 — cheapest-value ranks first, the sign check.
//! - `T=4` — the same bytes as one plane, the reference for the 1.0 budget.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test salt_rank_alloc -- --ignored --nocapture
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
/// 12,288 tokens — where `salt_gptq` stopped being under-sampled.
const GRAM_WINDOWS: usize = 48;
const GRAM_SEQ: usize = 256;
const DAMP: f64 = 0.01;
/// Largest rank computed for any one tensor.
const RANK_MAX: usize = 192;
const OVERSAMPLE: usize = 8;
const TQ2_0_BLOCK_BITS: f64 = 66.0 * 8.0;
const FACTOR_BITS: f64 = 16.0;

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

/// `H + λI` and its lower Cholesky factor, both f64.
fn damped(h: &[f64], k: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let mean: f64 = (0..k).map(|i| h[i * k + i]).sum::<f64>() / k as f64;
    let lambda = DAMP * mean.max(1e-12);
    let mut hd = h.to_vec();
    for i in 0..k {
        hd[i * k + i] += lambda;
    }
    let mut l = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = hd[i * k + j];
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
    Some((hd, l))
}

/// Modified Gram–Schmidt on the `r` columns of `y` (`n × r`); returns surviving columns, packed left.
fn orthonormalize(y: &mut [f64], n: usize, r: usize) -> usize {
    let mut kept = 0usize;
    for c in 0..r {
        for p in 0..kept {
            let dot: f64 = (0..n).map(|i| y[i * r + c] * y[i * r + p]).sum();
            for i in 0..n {
                y[i * r + c] -= dot * y[i * r + p];
            }
        }
        let norm: f64 = (0..n)
            .map(|i| y[i * r + c] * y[i * r + c])
            .sum::<f64>()
            .sqrt();
        if norm > 1e-10 {
            for i in 0..n {
                y[i * r + kept] = y[i * r + c] / norm;
            }
            kept += 1;
        }
    }
    kept
}

/// Cyclic Jacobi eigendecomposition of a symmetric `m × m` matrix. Returns (eigenvalues, eigenvectors
/// column-major in `v[i*m + j]` = component i of vector j), sorted descending.
fn jacobi_eigen(a: &mut [f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0f64; m * m];
    for i in 0..m {
        v[i * m + i] = 1.0;
    }
    for _sweep in 0..60 {
        let off: f64 = (0..m)
            .flat_map(|i| (0..m).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * m + j] * a[i * m + j])
            .sum();
        let scale: f64 = (0..m)
            .map(|i| a[i * m + i] * a[i * m + i])
            .sum::<f64>()
            .max(1e-300);
        if off <= 1e-22 * scale {
            break;
        }
        for p in 0..m {
            for q in (p + 1)..m {
                let apq = a[p * m + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * m + q] - a[p * m + p]) / (2.0 * apq);
                let t = theta.signum().max(0.0).mul_add(2.0, -1.0)
                    / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..m {
                    let akp = a[k * m + p];
                    let akq = a[k * m + q];
                    a[k * m + p] = c * akp - s * akq;
                    a[k * m + q] = s * akp + c * akq;
                }
                for k in 0..m {
                    let apk = a[p * m + k];
                    let aqk = a[q * m + k];
                    a[p * m + k] = c * apk - s * aqk;
                    a[q * m + k] = s * apk + c * aqk;
                }
                for k in 0..m {
                    let vkp = v[k * m + p];
                    let vkq = v[k * m + q];
                    v[k * m + p] = c * vkp - s * vkq;
                    v[k * m + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&x, &y| a[y * m + y].total_cmp(&a[x * m + x]));
    let vals: Vec<f64> = order.iter().map(|&j| a[j * m + j]).collect();
    let mut vecs = vec![0.0f64; m * m];
    for (newj, &j) in order.iter().enumerate() {
        for i in 0..m {
            vecs[i * m + newj] = v[i * m + j];
        }
    }
    (vals, vecs)
}

/// Everything needed to apply any rank `≤ r_max` correction to one tensor, and to price it.
struct Spectrum {
    /// `σᵢ²` in the activation metric, descending — the exact output error rank `i` removes.
    gains: Vec<f64>,
    /// `U` (`n × r_max`): top left singular vectors of `E·L`.
    u: Vec<f32>,
    /// `P = Uᵀ·E` (`r_max × k`). The rank-`r` correction is `U[:, ..r] · P[..r, :]`.
    p: Vec<f32>,
    r_max: usize,
    /// `‖W·L‖²` — the tensor's own output energy, for relative pricing.
    output_energy: f64,
}

fn spectrum(w: &[f32], base: &[f32], n: usize, k: usize, gram: &[f64]) -> Option<Spectrum> {
    let (hd, l) = damped(gram, k)?;
    let target = RANK_MAX.min(n).min(k);
    let r = (target + OVERSAMPLE).min(n).min(k);
    let e: Vec<f64> = w
        .iter()
        .zip(base)
        .map(|(&a, &b)| f64::from(a - b))
        .collect();

    // Range finder: Y = E·L·Ω.
    let mut s = 0x0123_4567_89AB_CDEFu64 ^ (n as u64 * 31 + k as u64);
    let omega: Vec<f64> = (0..k * r)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        })
        .collect();
    let mut lo = vec![0.0f64; k * r];
    lo.par_chunks_mut(r).enumerate().for_each(|(i, out)| {
        for p in 0..=i {
            let lip = l[i * k + p];
            if lip != 0.0 {
                for c in 0..r {
                    out[c] += lip * omega[p * r + c];
                }
            }
        }
    });
    let mut y = vec![0.0f64; n * r];
    y.par_chunks_mut(r)
        .zip(e.par_chunks(k))
        .for_each(|(out, er)| {
            for (p, &ep) in er.iter().enumerate() {
                if ep != 0.0 {
                    for c in 0..r {
                        out[c] += ep * lo[p * r + c];
                    }
                }
            }
        });
    let kept = orthonormalize(&mut y, n, r);
    if kept == 0 {
        return None;
    }

    // C = Qᵀ·E (kept × k), G = C·Hd·Cᵀ (kept × kept). Eigenpairs of G are σ² and the rotation of Q
    // onto the left singular vectors of E·L.
    let mut c = vec![0.0f64; kept * k];
    c.par_chunks_mut(k).enumerate().for_each(|(col, crow)| {
        for i in 0..n {
            let q = y[i * r + col];
            if q != 0.0 {
                for (j, cj) in crow.iter_mut().enumerate() {
                    *cj += q * e[i * k + j];
                }
            }
        }
    });
    let mut ch = vec![0.0f64; kept * k];
    ch.par_chunks_mut(k).enumerate().for_each(|(a, out)| {
        let crow = &c[a * k..(a + 1) * k];
        for (j, o) in out.iter_mut().enumerate() {
            let hrow = &hd[j * k..(j + 1) * k];
            *o = crow.iter().zip(hrow).map(|(x, y)| x * y).sum();
        }
    });
    let mut g = vec![0.0f64; kept * kept];
    for a in 0..kept {
        for b in 0..kept {
            g[a * kept + b] = ch[a * k..(a + 1) * k]
                .iter()
                .zip(&c[b * k..(b + 1) * k])
                .map(|(x, y)| x * y)
                .sum();
        }
    }
    let (vals, vecs) = jacobi_eigen(&mut g, kept);
    let r_max = target.min(kept);

    // U = Q·V, P = Vᵀ·C, truncated to r_max.
    let mut u = vec![0.0f32; n * r_max];
    for i in 0..n {
        for j in 0..r_max {
            let mut acc = 0.0f64;
            for a in 0..kept {
                acc += y[i * r + a] * vecs[a * kept + j];
            }
            u[i * r_max + j] = acc as f32;
        }
    }
    let mut p = vec![0.0f32; r_max * k];
    for j in 0..r_max {
        for col in 0..k {
            let mut acc = 0.0f64;
            for a in 0..kept {
                acc += vecs[a * kept + j] * c[a * k + col];
            }
            p[j * k + col] = acc as f32;
        }
    }
    let gains: Vec<f64> = vals.iter().take(r_max).map(|&v| v.max(0.0)).collect();

    // ‖W·L‖² = tr(W·Hd·Wᵀ).
    let output_energy: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let wr = &w[i * k..(i + 1) * k];
            let mut acc = 0.0f64;
            for a in 0..k {
                let wa = f64::from(wr[a]);
                if wa == 0.0 {
                    continue;
                }
                let hrow = &hd[a * k..(a + 1) * k];
                acc += wa
                    * wr.iter()
                        .zip(hrow)
                        .map(|(&x, &h)| f64::from(x) * h)
                        .sum::<f64>();
            }
            acc
        })
        .sum();

    Some(Spectrum {
        gains,
        u,
        p,
        r_max,
        output_energy,
    })
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

/// Resident bits of one TQ2_0 plane for a tensor, padding included.
fn plane_bits(n: usize, k: usize) -> f64 {
    (n * k.div_ceil(256)) as f64 * TQ2_0_BLOCK_BITS
}

#[test]
#[ignore = "needs SmolLM2-135M; per-tap Grams, a spectrum per projection, and ~12 evaluations"]
fn rank_allocation_against_uniform_rank_and_a_plane() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let t_ref = 3usize;
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

    // Spectra for the 210 projections. The tied embedding has no Gram and stays at T in every arm.
    println!("computing spectra (rank ≤ {RANK_MAX})…");
    let mut targets: Vec<usize> = Vec::new();
    let mut spectra: Vec<Spectrum> = Vec::new();
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
            if let Some(sp) = spectrum(&fp[i], &base[i], n, k, gram) {
                targets.push(i);
                spectra.push(sp);
            }
        }
        if li % 10 == 0 {
            println!("  layer {li}/{n_layers}");
        }
    }
    drop((g_attn, g_ffn, g_down, g_o));

    let cost = |i: usize| -> f64 {
        let (n, k) = shapes[targets[i]];
        (n + k) as f64 * FACTOR_BITS
    };
    let total_plane_bits: f64 = targets
        .iter()
        .map(|&t| plane_bits(shapes[t].0, shapes[t].1))
        .sum();

    let apply = |ranks: &[usize]| -> Vec<Vec<f32>> {
        let mut out = base.clone();
        for (s, (&t, sp)) in targets.iter().zip(&spectra).enumerate() {
            let r = ranks[s].min(sp.r_max);
            if r == 0 {
                continue;
            }
            let (n, k) = shapes[t];
            out[t].par_chunks_mut(k).enumerate().for_each(|(i, row)| {
                for j in 0..r {
                    let uij = sp.u[i * sp.r_max + j];
                    if uij == 0.0 {
                        continue;
                    }
                    let prow = &sp.p[j * k..(j + 1) * k];
                    for (o, &pv) in row.iter_mut().zip(prow) {
                        *o += uij * pv;
                    }
                }
            });
            let _ = n;
        }
        out
    };
    let bits_of = |ranks: &[usize]| -> f64 {
        ranks
            .iter()
            .enumerate()
            .map(|(s, &r)| r.min(spectra[s].r_max) as f64 * cost(s))
            .sum()
    };

    // Uniform: each tensor's share of the budget in proportion to its own plane cost.
    let uniform = |frac: f64| -> Vec<usize> {
        targets
            .iter()
            .enumerate()
            .map(|(s, &t)| {
                let (n, k) = shapes[t];
                ((frac * plane_bits(n, k) / cost(s)).floor() as usize).min(spectra[s].r_max)
            })
            .collect()
    };
    // Greedy water-fill by value per bit. Each tensor's gains are sorted descending, so taking the
    // best next rank globally is optimal for the priced objective.
    let allocate = |budget: f64, relative: bool, anti: bool| -> Vec<usize> {
        let value = |s: usize, j: usize| -> f64 {
            let g = spectra[s].gains[j];
            let g = if relative {
                g / spectra[s].output_energy.max(1e-30)
            } else {
                g
            };
            g / cost(s)
        };
        let mut ranks = vec![0usize; spectra.len()];
        let mut spent = 0.0f64;
        if anti {
            // Cheapest value first: take each tensor's WORST ranks. Sorting all ranks ascending by
            // value and walking the list is the mirror image of water-filling.
            let mut all: Vec<(f64, usize)> = (0..spectra.len())
                .flat_map(|s| (0..spectra[s].r_max).map(move |j| (s, j)))
                .map(|(s, j)| (value(s, j), s))
                .collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (_, s) in all {
                if spent + cost(s) > budget {
                    continue;
                }
                if ranks[s] < spectra[s].r_max {
                    ranks[s] += 1;
                    spent += cost(s);
                }
            }
            return ranks;
        }
        loop {
            let mut best: Option<(f64, usize)> = None;
            for s in 0..spectra.len() {
                let j = ranks[s];
                if j >= spectra[s].r_max || spent + cost(s) > budget {
                    continue;
                }
                let v = value(s, j);
                if best.is_none_or(|(bv, _)| v > bv) {
                    best = Some((v, s));
                }
            }
            match best {
                Some((_, s)) => {
                    ranks[s] += 1;
                    spent += cost(s);
                }
                None => break,
            }
        }
        ranks
    };

    let ppl_fp = perplexity_windowed(&fp, &arch, &eval, EVAL_WINDOW);
    let ppl_base = perplexity_windowed(&base, &arch, &eval, EVAL_WINDOW);
    let plus_plane: Vec<Vec<f32>> = {
        let mut q = base.clone();
        for &t in &targets {
            q[t] = fit(&fp[t], shapes[t].0, shapes[t].1, t_ref + 1);
        }
        q
    };
    let ppl_plane = perplexity_windowed(&plus_plane, &arch, &eval, EVAL_WINDOW);

    println!(
        "\nSmolLM2-135M | WikiText-2 {} held-out | fold α=0.75 | g{GROUP} | rotation | T={t_ref}\n\
         {} projections; one plane of resident TQ2_0 (padding included) = {:.3e} bits\n",
        eval.len(),
        targets.len(),
        total_plane_bits
    );
    println!(
        "{:<38} {:>6} {:>10} {:>9} {:>11} {:>11}",
        "arm", "budget", "ppl", "vs T", "vs uniform", "ranks≠0"
    );
    println!("{}", "-".repeat(92));
    let pct = |a: f64, b: f64| format!("{:+.2}%", 100.0 * (a - b) / b);
    println!(
        "{:<38} {:>6} {ppl_fp:>10.4} {:>9}",
        "fp master",
        "—",
        pct(ppl_fp, ppl_base)
    );
    println!(
        "{:<38} {:>6} {ppl_base:>10.4} {:>9}",
        format!("T={t_ref} (baseline)"),
        "0",
        "—"
    );
    println!(
        "{:<38} {:>6} {ppl_plane:>10.4} {:>9}",
        format!("T={} (one plane, reference)", t_ref + 1),
        "1.00",
        pct(ppl_plane, ppl_base)
    );

    // DEGENERATE CONTROL: zero budget must reproduce the baseline exactly.
    let zero = allocate(0.0, false, false);
    assert!(zero.iter().all(|&r| r == 0), "a zero budget allocated rank");
    let ppl_zero = perplexity_windowed(&apply(&zero), &arch, &eval, EVAL_WINDOW);
    assert!(
        (ppl_zero - ppl_base).abs() / ppl_base < 1e-9,
        "zero budget scored {ppl_zero} against the baseline's {ppl_base}"
    );

    let mut summary: Vec<(f64, f64, f64, f64)> = Vec::new();
    for frac in [0.1f64, 0.25, 1.0] {
        let budget = frac * total_plane_bits;
        let ru = uniform(frac);
        let ppl_u = perplexity_windowed(&apply(&ru), &arch, &eval, EVAL_WINDOW);
        let nz = |r: &[usize]| r.iter().filter(|&&x| x > 0).count();
        println!(
            "{:<38} {frac:>6.2} {ppl_u:>10.4} {:>9} {:>11} {:>11}",
            "uniform rank",
            pct(ppl_u, ppl_base),
            "—",
            nz(&ru)
        );
        let mut row = (frac, ppl_u, f64::NAN, f64::NAN);
        for (label, relative) in [
            ("allocated (absolute)", false),
            ("allocated (relative)", true),
        ] {
            let ra = allocate(budget, relative, false);
            // Allocation may not spend more than uniform did.
            assert!(
                bits_of(&ra) <= budget * 1.000_001,
                "{label} at {frac} spent {} of {budget}",
                bits_of(&ra)
            );
            let ppl_a = perplexity_windowed(&apply(&ra), &arch, &eval, EVAL_WINDOW);
            println!(
                "{:<38} {frac:>6.2} {ppl_a:>10.4} {:>9} {:>11} {:>11}",
                label,
                pct(ppl_a, ppl_base),
                pct(ppl_a, ppl_u),
                nz(&ra)
            );
            if relative {
                row.3 = ppl_a;
            } else {
                row.2 = ppl_a;
            }
        }
        if (frac - 0.25).abs() < 1e-9 {
            let rn = allocate(budget, false, true);
            let ppl_n = perplexity_windowed(&apply(&rn), &arch, &eval, EVAL_WINDOW);
            println!(
                "{:<38} {frac:>6.2} {ppl_n:>10.4} {:>9} {:>11} {:>11}",
                "anti-allocated (sign check)",
                pct(ppl_n, ppl_base),
                pct(ppl_n, ppl_u),
                nz(&rn)
            );
        }
        summary.push(row);
    }

    println!("\nper-bit return (loss removed per plane-equivalent), relative to one plane:");
    let plane_gain = ppl_base - ppl_plane;
    for (frac, u, a, r) in &summary {
        let eff = |p: f64| (ppl_base - p) / (frac * plane_gain);
        println!(
            "  budget {frac:.2}: uniform {:.2}×, absolute {:.2}×, relative {:.2}× a plane's return per bit",
            eff(*u),
            eff(*a),
            eff(*r)
        );
    }
    println!(
        "\n>1× means rank beats a plane per resident bit at that budget. Allocation WORKS if the\n\
         allocated arms beat uniform rank at the same budget; it is the coarseness explanation that\n\
         is on trial, not just the allocator."
    );
}

/// The allocator prices a rank by `σᵢ²`. That is only valid if applying ranks `0..r` removes exactly
/// `Σ_{i<r} σᵢ²` of activation-weighted error — checked here against a direct computation of
/// `tr((E − C)·Hd·(E − C)ᵀ)`, along with the eigensolver itself.
#[test]
fn a_ranks_price_is_exactly_the_error_it_removes() {
    // k > RANK_MAX + OVERSAMPLE, so the range finder is genuinely truncated — the case that matters.
    // The price must stay exact anyway: for ANY orthonormal U the removed error is
    // tr(Uᵀ·E·Hd·Eᵀ·U), and U = Q·V diagonalizes that to Σλ. Only optimality is approximate.
    let (n, k) = (260usize, 240usize);
    assert!(k > RANK_MAX + OVERSAMPLE);
    let mut s = 0xFEED_F00Du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f64 / (1u64 << 24) as f64) - 0.5
    };
    let w: Vec<f32> = (0..n * k).map(|_| next() as f32).collect();
    let base = vec![0.0f32; n * k];
    // Anisotropic Gram: X with decaying column scales, H = XᵀX / m.
    let m = 600;
    let x: Vec<f64> = (0..m * k)
        .map(|idx| next() * (1.0 / (1.0 + (idx % k) as f64)))
        .collect();
    let mut h = vec![0.0f64; k * k];
    for r in 0..m {
        for a in 0..k {
            for b in 0..k {
                h[a * k + b] += x[r * k + a] * x[r * k + b] / m as f64;
            }
        }
    }

    // Eigensolver: A = V·Λ·Vᵀ.
    let mut sym = h.clone();
    let (vals, vecs) = jacobi_eigen(&mut sym, k);
    for a in 0..k {
        for b in 0..k {
            let rec: f64 = (0..k)
                .map(|j| vecs[a * k + j] * vals[j] * vecs[b * k + j])
                .sum();
            assert!(
                (rec - h[a * k + b]).abs() < 1e-9,
                "Jacobi does not reconstruct H"
            );
        }
    }
    assert!(
        vals.windows(2).all(|p| p[0] >= p[1]),
        "eigenvalues not descending"
    );

    let sp = spectrum(&w, &base, n, k, &h).expect("spectrum");
    let (hd, _) = damped(&h, k).unwrap();
    let err = |corr: &[f64]| -> f64 {
        let d: Vec<f64> = w
            .iter()
            .zip(corr)
            .map(|(&a, &c)| f64::from(a) - c)
            .collect();
        (0..n)
            .map(|i| {
                let row = &d[i * k..(i + 1) * k];
                (0..k)
                    .map(|a| row[a] * (0..k).map(|b| hd[a * k + b] * row[b]).sum::<f64>())
                    .sum::<f64>()
            })
            .sum()
    };
    let total = err(&vec![0.0; n * k]);
    assert_eq!(
        sp.r_max, RANK_MAX,
        "range finder should be truncated at RANK_MAX"
    );
    for r in [1usize, 3, 8, sp.r_max] {
        let mut corr = vec![0.0f64; n * k];
        for i in 0..n {
            for j in 0..r {
                for c in 0..k {
                    corr[i * k + c] +=
                        f64::from(sp.u[i * sp.r_max + j]) * f64::from(sp.p[j * k + c]);
                }
            }
        }
        let predicted = total - sp.gains[..r].iter().sum::<f64>();
        let actual = err(&corr);
        assert!(
            (predicted - actual).abs() <= 1e-4 * total,
            "rank {r}: gains predict residual {predicted:.6e}, actual {actual:.6e}"
        );
    }
}
