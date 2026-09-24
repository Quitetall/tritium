//! **Does output-side curvature help GPTQ? (H6, first-principles plan)**
//!
//! H1 showed the cost of a ternary error is a stable quadratic form and that the cheap proxies rank
//! it (Gram/Fisher ~0.8), while the MEASURED per-unit cost is swamped by a first-order sample term.
//! So the one objective worth optimizing harder is the curvature itself. Shipping GPTQ uses only its
//! input half: it minimises `tr(E·A·Eᵀ)` with `A = E[x·xᵀ]`, treating every output row as equally
//! and independently important. The Kronecker (K-FAC) curvature is `A ⊗ G` with `G = E[δ·δᵀ]` over
//! the output gradient, and a non-diagonal `G` couples the rows' rounding decisions.
//! [`tritium_nn::salt_fit::fit_tensor_kronecker`] rounds on that lattice (nearest-plane over
//! `chol(A⁻¹) ⊗ chol(G⁻¹)`); with `G = I` it is shipping GPTQ bit for bit.
//!
//! `G` is the TRUE Fisher: labels are sampled from the fp model's own softmax, not taken from the
//! text, so it is a curvature and not a gradient-noise estimate.
//!
//! Scoring uses the antithetic split H1 suggested. Besides `Q = W + E`, each arm is also scored at
//! `W − E`: `even = (ΔL(+E) + ΔL(−E))/2` is the curvature cost with the first-order sample term
//! cancelled, `odd` is that sample term. Differences in `even` are what a method changes; `odd` is
//! what the evaluation set happens to reward.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test kronecker_gptq -- --ignored --nocapture
//! ```
//! Knobs: `TRITIUM_H6_TS` (default `2,3`), `TRITIUM_H6_CALIB` (512-token windows for A and G,
//! default 32), `TRITIUM_H6_ROW_DECAYS` (default `1,0.5`), `TRITIUM_H6_EVAL_WINDOWS` (default 64).

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{Arch, Calib, calibrate, extract, fold, perplexity_windowed};
use rayon::prelude::*;
use tritium_nn::ModelRunner;
use tritium_nn::salt_fit::{
    ActivationAwareConfig, OutputMetric, TapGrams, auto_decay, fit_tensor, fit_tensor_kronecker,
    fit_to_dense,
};
use tritium_train::Tape;
use tritium_train::ops::ste::{self, RotationPolicy};
use tritium_train::tape::ValueId;

const WINDOW: usize = 512;
const FOLD_WINDOWS: usize = 4;
const GROUP: usize = 128;
const GRID: usize = 16;

fn env_str(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_owned())
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

/// `calibrate::forward`, recording the output value of every projection in `extract` order
/// (per layer q, k, v, o, gate, up, down; q/k/v before RoPE). Asserted equal to the library
/// forward below, so the gradients read at these ids are the model's.
fn forward_with_outputs(
    t: &mut Tape,
    wids: &[ValueId],
    a: &Arch,
    tokens: &[u32],
) -> (ValueId, Vec<[ValueId; 7]>) {
    let seq = tokens.len();
    let qd = a.n_head * a.head_dim;
    let mut outs = Vec::with_capacity(a.n_layers);
    let mut hidden = t.embed_gather(wids[0], tokens, a.vocab, a.n_embd);
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        let an = t.leaf(a.attn_norms[li].clone());
        let xn = t.rmsnorm(hidden, an, seq, a.n_embd, a.eps);
        // attention_heads, unrolled so the q/k/v projection outputs are reachable.
        let kvd = a.n_head_kv * a.head_dim;
        let q = t.dense_matmul(xn, wids[base], seq, qd, a.n_embd);
        let k = t.dense_matmul(xn, wids[base + 1], seq, kvd, a.n_embd);
        let v = t.dense_matmul(xn, wids[base + 2], seq, kvd, a.n_embd);
        let cat = heads_from(t, q, k, v, seq, a);
        let o = t.dense_matmul(cat, wids[base + 3], seq, a.n_embd, qd);
        hidden = t.add(hidden, o);
        let fnw = t.leaf(a.ffn_norms[li].clone());
        let hn = t.rmsnorm(hidden, fnw, seq, a.n_embd, a.eps);
        let g = t.dense_matmul(hn, wids[base + 4], seq, a.ff, a.n_embd);
        let u = t.dense_matmul(hn, wids[base + 5], seq, a.ff, a.n_embd);
        let ga = t.silu(g);
        let gated = t.mul(ga, u);
        let down = t.dense_matmul(gated, wids[base + 6], seq, a.n_embd, a.ff);
        hidden = t.add(hidden, down);
        outs.push([q, k, v, o, g, u, down]);
    }
    let onw = t.leaf(a.out_norm.clone());
    let fnorm = t.rmsnorm(hidden, onw, seq, a.n_embd, a.eps);
    (t.dense_matmul(fnorm, wids[0], seq, a.vocab, a.n_embd), outs)
}

/// The body of `tritium_train::nn::attention_heads` after its three projections.
fn heads_from(t: &mut Tape, q: ValueId, k: ValueId, v: ValueId, seq: usize, a: &Arch) -> ValueId {
    let (qd, kvd) = (a.n_head * a.head_dim, a.n_head_kv * a.head_dim);
    let group = a.n_head / a.n_head_kv;
    let scale = 1.0 / (a.head_dim as f32).sqrt();
    let pos: Vec<usize> = (0..seq).collect();
    let q = t.rope(q, pos.clone(), a.n_head, a.head_dim, a.theta);
    let k = t.rope(k, pos, a.n_head_kv, a.head_dim, a.theta);
    let mut head_outs = Vec::with_capacity(a.n_head);
    for h in 0..a.n_head {
        let kv = h / group;
        let qh = t.slice_cols(q, seq, qd, h * a.head_dim, a.head_dim);
        let kh = t.slice_cols(k, seq, kvd, kv * a.head_dim, a.head_dim);
        let vh = t.slice_cols(v, seq, kvd, kv * a.head_dim, a.head_dim);
        let scores = t.dense_matmul(qh, kh, seq, seq, a.head_dim);
        let scores = t.scale_const(scores, scale);
        let scores = t.causal_mask(scores, seq, seq);
        let p = t.softmax(scores, seq, seq);
        let vt = t.transpose(vh, seq, a.head_dim);
        head_outs.push(t.dense_matmul(p, vt, seq, a.head_dim, seq));
    }
    t.concat_cols(&head_outs, seq, &vec![a.head_dim; a.n_head])
}

/// `G += δᵀδ` over the rows of a `[seq, n]` gradient block (upper triangle; mirrored in `finish`).
fn accumulate_outer(g: &mut [f64], n: usize, delta: &[f32], seq: usize) {
    g.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for r in 0..seq {
            let d = &delta[r * n..(r + 1) * n];
            let di = f64::from(d[i]);
            if di == 0.0 {
                continue;
            }
            for (j, gij) in row.iter_mut().enumerate().skip(i) {
                *gij += di * f64::from(d[j]);
            }
        }
    });
}

fn finish_outer(g: &mut [f64], n: usize, samples: usize) {
    for i in 0..n {
        for j in 0..i {
            g[i * n + j] = g[j * n + i];
        }
    }
    for v in g.iter_mut() {
        *v /= samples.max(1) as f64;
    }
}

/// Output-side Fisher `G = E[δ·δᵀ]` for every projection, labels sampled from the model.
fn output_fishers(weights: &[Vec<f32>], a: &Arch, windows: &[&[u32]], seed: u64) -> Vec<Vec<f64>> {
    // Output width of each projection, in `extract` order.
    let widths = |slot: usize| -> usize {
        match slot {
            0 => a.n_head * a.head_dim,
            1 | 2 => a.n_head_kv * a.head_dim,
            4 | 5 => a.ff,
            _ => a.n_embd, // o, down
        }
    };
    let mut gs: Vec<Vec<f64>> = (0..a.n_layers * 7)
        .map(|x| vec![0.0; widths(x % 7) * widths(x % 7)])
        .collect();
    let mut state = seed | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut samples = 0usize;
    for w in windows {
        let seq = w.len();
        let mut t = Tape::new();
        let wids: Vec<ValueId> = weights.iter().map(|x| t.leaf(x.clone())).collect();
        let (logits, outs) = forward_with_outputs(&mut t, &wids, a, w);
        let values = t.value(logits).to_vec();
        let mut target = vec![0.0f32; seq * a.vocab];
        for p in 0..seq {
            let row = &values[p * a.vocab..(p + 1) * a.vocab];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z: f64 = row.iter().map(|&l| f64::from(l - m).exp()).sum();
            let mut u = uniform() * z;
            let mut pick = a.vocab - 1;
            for (c, &l) in row.iter().enumerate() {
                u -= f64::from(l - m).exp();
                if u <= 0.0 {
                    pick = c;
                    break;
                }
            }
            target[p * a.vocab + pick] = 1.0;
        }
        let tg = t.leaf(target);
        let loss = t.softmax_xent(logits, tg, seq, a.vocab);
        let grads = t.backward(loss);
        for (li, ids) in outs.iter().enumerate() {
            for (slot, &id) in ids.iter().enumerate() {
                accumulate_outer(&mut gs[li * 7 + slot], widths(slot), &grads[id], seq);
            }
        }
        samples += seq;
    }
    for (x, g) in gs.iter_mut().enumerate() {
        finish_outer(g, widths(x % 7), samples);
    }
    gs
}

#[derive(Clone, Copy)]
enum Arm {
    Plain,
    Gptq,
    Kronecker(f64),
}

impl Arm {
    fn label(self) -> String {
        match self {
            Arm::Plain => "ladder (RTN)".into(),
            Arm::Gptq => "GPTQ input-only, auto decay".into(),
            Arm::Kronecker(d) => format!("Kronecker GPTQ, λ_G={d}"),
        }
    }
}

#[test]
#[ignore = "H6 Kronecker GPTQ; needs SmolLM2-135M; run explicitly"]
fn output_side_curvature_against_input_only_gptq() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let ts: Vec<usize> = env_str("TRITIUM_H6_TS", "2,3")
        .split(',')
        .map(|s| s.trim().parse().expect("T"))
        .collect();
    let calib_windows: usize = env_str("TRITIUM_H6_CALIB", "32").parse().expect("calib");
    let row_decays: Vec<f64> = env_str("TRITIUM_H6_ROW_DECAYS", "1,0.5")
        .split(',')
        .map(|s| s.trim().parse().expect("decay"))
        .collect();
    let n_eval: usize = env_str("TRITIUM_H6_EVAL_WINDOWS", "64")
        .parse()
        .expect("eval");

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch, fp, shapes) = extract(&runner);
    let (train, eval) = corpus();
    let eval = &eval[..(n_eval * WINDOW).min(eval.len())];
    let mut calib = Calib::new(&arch);
    for w in 0..FOLD_WINDOWS {
        calibrate(&fp, &arch, &train[w * WINDOW..(w + 1) * WINDOW], &mut calib);
    }
    let (folded, arch) = fold(&fp, &shapes, &arch, &calib, 0.75);

    // The recorded forward must be the library forward, or G is some other model's curvature.
    {
        let tokens = &train[..64];
        let mut t1 = Tape::new();
        let w1: Vec<ValueId> = folded.iter().map(|x| t1.leaf(x.clone())).collect();
        let lib = common::forward(&mut t1, &w1, &arch, tokens);
        let mut t2 = Tape::new();
        let w2: Vec<ValueId> = folded.iter().map(|x| t2.leaf(x.clone())).collect();
        let (ours, _) = forward_with_outputs(&mut t2, &w2, &arch, tokens);
        assert_eq!(
            t1.value(lib),
            t2.value(ours),
            "recorded forward drifted from the library"
        );
    }

    let started = Instant::now();
    let windows: Vec<&[u32]> = (0..calib_windows)
        .map(|w| &train[w * WINDOW..(w + 1) * WINDOW])
        .collect();
    let grams = TapGrams::collect(&folded, &arch, &windows);
    let fishers = output_fishers(&folded, &arch, &windows, 0xF15E_A5ED);
    let tokens = calib_windows * WINDOW;
    let nll = |w: &[Vec<f32>]| perplexity_windowed(w, &arch, eval, WINDOW).ln();
    let base = nll(&folded);
    println!(
        "SmolLM2-135M | fold α=0.75 | ladder g{GROUP} rotated | A and G from {tokens} tokens (G: sampled-label Fisher) \
         | {} eval tokens | fp NLL {base:.5} | curvature {:.0}s\n",
        eval.len(),
        started.elapsed().as_secs_f64()
    );

    let mut arms = vec![Arm::Plain, Arm::Gptq];
    arms.extend(row_decays.iter().map(|&d| Arm::Kronecker(d)));
    for &t in &ts {
        println!("T={t}");
        println!(
            "{:<32} {:>9} {:>10} {:>10} {:>10} {:>10}",
            "arm", "ppl", "ΔL(+E)", "ΔL(−E)", "even", "odd"
        );
        for &arm in &arms {
            let started = Instant::now();
            let q: Vec<Vec<f32>> = folded
                .iter()
                .zip(&shapes)
                .enumerate()
                .map(|(i, (w, &(rows, cols)))| {
                    let ladder = || {
                        ste::salt_quantize_forward_grouped_geometric(
                            w,
                            rows,
                            cols,
                            t,
                            GROUP,
                            GRID,
                            RotationPolicy::Always,
                        )
                    };
                    // The tied embed/head has no single input Gram: every arm keeps it on the ladder.
                    if i == 0 || matches!(arm, Arm::Plain) {
                        return ladder();
                    }
                    let (li, slot) = ((i - 1) / 7, (i - 1) % 7);
                    let cfg = ActivationAwareConfig {
                        planes: t,
                        group: GROUP,
                        grid: GRID,
                        damp: 0.01,
                        search_sweeps: 0,
                        refit_scale: false,
                        rotate: true,
                        decay: auto_decay(tokens, cols),
                        decay_ramp: false,
                    };
                    let a = grams.for_slot(li, slot);
                    let fits = match arm {
                        Arm::Gptq => fit_tensor(w, rows, cols, a, &cfg),
                        Arm::Kronecker(d) => fit_tensor_kronecker(
                            w,
                            rows,
                            cols,
                            a,
                            &OutputMetric {
                                gram: &fishers[li * 7 + slot],
                                damp: 0.01,
                                decay: d,
                            },
                            &cfg,
                        ),
                        Arm::Plain => unreachable!(),
                    };
                    fits.map_or_else(ladder, |f| fit_to_dense(&f, rows, cols, GROUP, true))
                })
                .collect();
            let anti: Vec<Vec<f32>> = folded
                .iter()
                .zip(&q)
                .map(|(w, qv)| w.iter().zip(qv).map(|(a, b)| 2.0 * a - b).collect())
                .collect();
            let (plus, minus) = (nll(&q) - base, nll(&anti) - base);
            println!(
                "{:<32} {:>9.3} {:>+10.5} {:>+10.5} {:>+10.5} {:>+10.5}   ({:.0}s)",
                arm.label(),
                (base + plus).exp(),
                plus,
                minus,
                0.5 * (plus + minus),
                0.5 * (plus - minus),
                started.elapsed().as_secs_f64()
            );
        }
        println!();
    }
}
