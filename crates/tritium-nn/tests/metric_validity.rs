//! **Does WikiText perplexity rank quantizers the way the model's own behaviour does? (H5)**
//!
//! Every method comparison in this repo is teacher-forced WikiText-2 perplexity. H1 showed that a
//! finite evaluation sample carries a first-order term as large as a per-tensor quantization cost,
//! and H3 put the configuration-to-configuration noise at 0.6-0.9% ppl. This asks whether the
//! metric itself ranks methods the way the things we care about do.
//!
//! One set of quantized models per `T`, scored six ways:
//!
//! - `ppl` — the repo's metric: teacher-forced WikiText-2 eval perplexity;
//! - `even` — the antithetic curvature cost `(ΔL(+E) + ΔL(−E))/2` on the same text (H1's first-order
//!   sample term cancelled);
//! - `KL wiki` — `KL(fp ‖ q)` over the full next-token distribution on WikiText (every token, not
//!   only the observed one);
//! - `KL self` — the same KL on text the fp model sampled itself (temperature 1) — on-policy, the
//!   distribution the quantized model will actually be run on;
//! - `late` — excess NLL at positions ≥ 512 of 1024-token windows;
//! - `agree` — free-running fidelity: tokens of greedy continuation that match fp before the first
//!   divergence, mean over prompts (KV-cached runners).
//!
//! The model set contains ordered pairs whose true order is known in advance (RTN vs GPTQ, 2k vs
//! 16k calibration tokens) and one NULL pair — two random sign gauges that are equal in expectation
//! — so each metric's noise can be read off it. It also contains the self-trace calibration arm
//! (ScaleQ/AYOT): GPTQ calibrated on fp-sampled text instead of WikiText train.
//!
//! ```text
//! TRITIUM_CORPUS=$HOME/.cache/tritium-corpora/wikitext2_400k_32k.json \
//!   cargo test -p tritium-nn --release --test metric_validity -- --ignored --nocapture
//! ```
//! Knobs: `TRITIUM_H5_TS` (default `3,2`).

mod common;

use std::path::PathBuf;
use std::time::Instant;

use common::{Arch, Calib, calibrate, extract, fold, forward, perplexity_windowed};
use rayon::prelude::*;
use tritium_nn::salt_fit::{ActivationAwareConfig, TapGrams, auto_decay, fit_tensor, fit_to_dense};
use tritium_nn::{DenseLinear, Mlp, ModelRunner, ModelWeights, Projection, TokenEmbedding};
use tritium_train::Tape;
use tritium_train::ops::ste::{self, RotationPolicy};
use tritium_train::tape::ValueId;

const WINDOW: usize = 512;
const FOLD_WINDOWS: usize = 4;
const GROUP: usize = 128;
const GRID: usize = 16;
const EVAL_WINDOWS: usize = 64;
const KL_WINDOWS: usize = 16;
const LATE_WINDOW: usize = 1024;
const LATE_WINDOWS: usize = 8;
const SELF_SEQS: usize = 16;
const PROMPTS: usize = 32;
const PROMPT_LEN: usize = 64;
const CONTINUATION: usize = 64;

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

struct Rng(u64);

impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = f64::from(m)
        + row
            .iter()
            .map(|&l| f64::from(l - m).exp())
            .sum::<f64>()
            .ln();
    row.iter().map(|&l| f64::from(l) - lse).collect()
}

fn logits(weights: &[Vec<f32>], a: &Arch, tokens: &[u32]) -> Vec<f32> {
    let mut t = Tape::new();
    let wids: Vec<ValueId> = weights.iter().map(|w| t.leaf(w.clone())).collect();
    let out = forward(&mut t, &wids, a, tokens);
    t.value(out).to_vec()
}

/// A quantized model: weights in the folded basis plus the (folded) norms they pair with.
struct Model {
    label: String,
    weights: Vec<Vec<f32>>,
    arch: Arch,
}

/// A KV-cached runner for `model`: the fp checkpoint's structure with every projection, the tied
/// table and the two per-layer norms replaced by the model's (folded) values.
fn runner_for(dir: &std::path::Path, model: &Model) -> ModelRunner {
    let (config, _spec, mut w): (_, _, ModelWeights) = ModelWeights::load_hf(dir).expect("load_hf");
    let a = &model.arch;
    w.token_embd =
        TokenEmbedding::from_dense(model.weights[0].clone(), a.vocab, a.n_embd).expect("embedding");
    let dense = |i: usize, n: usize, k: usize| {
        Projection::Dense(DenseLinear::new_exact(model.weights[i].clone(), n, k).expect("dense"))
    };
    let (qd, kvd) = (a.n_head * a.head_dim, a.n_head_kv * a.head_dim);
    for (li, layer) in w.layers.iter_mut().enumerate() {
        let base = 1 + 7 * li;
        layer.attn_norm.clone_from(&a.attn_norms[li]);
        layer.ffn_norm.clone_from(&a.ffn_norms[li]);
        layer.q_proj = dense(base, qd, a.n_embd);
        layer.k_proj = dense(base + 1, kvd, a.n_embd);
        layer.v_proj = dense(base + 2, kvd, a.n_embd);
        layer.o_proj = dense(base + 3, a.n_embd, qd);
        let Mlp::SwiGlu(mlp) = &mut layer.mlp else {
            panic!("SmolLM2 is SwiGLU")
        };
        mlp.gate = dense(base + 4, a.ff, a.n_embd);
        mlp.up = dense(base + 5, a.ff, a.n_embd);
        mlp.down = dense(base + 6, a.n_embd, a.ff);
    }
    ModelRunner::from_weights(config, w, Box::new(tritium_cpu::CpuBackend::new()))
}

fn greedy(runner: &mut ModelRunner, prompt: &[u32], n: usize) -> Vec<u32> {
    runner.generate(prompt, n, u32::MAX).expect("generate")
}

/// Temperature-1 samples from the fp runner, each continuing a WikiText-train prompt.
fn sample_self_text(
    runner: &mut ModelRunner,
    prompts: &[&[u32]],
    len: usize,
    seed: u64,
) -> Vec<Vec<u32>> {
    let mut rng = Rng(seed);
    prompts
        .iter()
        .map(|prompt| {
            runner.reset();
            let mut seq = prompt.to_vec();
            let positions: Vec<usize> = (0..seq.len()).collect();
            let mut last = runner.forward(&seq, &positions).expect("prefill");
            while seq.len() < len {
                let lp = log_softmax(&last);
                let mut u = rng.uniform();
                let mut pick = lp.len() - 1;
                for (c, &l) in lp.iter().enumerate() {
                    u -= l.exp();
                    if u <= 0.0 {
                        pick = c;
                        break;
                    }
                }
                let next = pick as u32;
                let pos = seq.len();
                seq.push(next);
                last = runner.forward(&[next], &[pos]).expect("decode");
            }
            seq
        })
        .collect()
}

/// Mean `KL(p ‖ q)` over every position of `windows`, `p` from `reference`.
fn mean_kl(reference: &Model, model: &Model, windows: &[&[u32]]) -> f64 {
    let (mut total, mut n) = (0.0f64, 0usize);
    for w in windows {
        let lp = logits(&reference.weights, &reference.arch, w);
        let lq = logits(&model.weights, &model.arch, w);
        let vocab = reference.arch.vocab;
        let per: Vec<f64> = (0..w.len())
            .into_par_iter()
            .map(|p| {
                let a = log_softmax(&lp[p * vocab..(p + 1) * vocab]);
                let b = log_softmax(&lq[p * vocab..(p + 1) * vocab]);
                a.iter()
                    .zip(&b)
                    .map(|(x, y)| x.exp() * (x - y))
                    .sum::<f64>()
            })
            .collect();
        total += per.iter().sum::<f64>();
        n += per.len();
    }
    total / n as f64
}

/// Mean next-token NLL at positions `>= from` of each window.
fn nll_from(model: &Model, windows: &[&[u32]], from: usize) -> f64 {
    let (mut total, mut n) = (0.0f64, 0usize);
    for w in windows {
        let l = logits(&model.weights, &model.arch, w);
        let vocab = model.arch.vocab;
        for p in from..w.len() - 1 {
            total -= log_softmax(&l[p * vocab..(p + 1) * vocab])[w[p + 1] as usize];
            n += 1;
        }
    }
    total / n as f64
}

fn ladder_all(w: &[Vec<f32>], shapes: &[(usize, usize)], t: usize) -> Vec<Vec<f32>> {
    w.iter()
        .zip(shapes)
        .map(|(v, &(r, c))| {
            ste::salt_quantize_forward_grouped_geometric(
                v,
                r,
                c,
                t,
                GROUP,
                GRID,
                RotationPolicy::Always,
            )
        })
        .collect()
}

/// RTN in a signed-Hadamard gauge: `H·D` per 128-block, one seeded sign vector per tensor.
fn signed_ladder_all(
    w: &[Vec<f32>],
    shapes: &[(usize, usize)],
    t: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = Rng(seed);
    w.iter()
        .zip(shapes)
        .map(|(v, &(rows, cols))| {
            let signs: Vec<f32> = (0..cols)
                .map(|_| if rng.uniform() < 0.5 { -1.0 } else { 1.0 })
                .collect();
            let hb = |row: &mut [f32]| {
                for b in row.chunks_mut(GROUP) {
                    if b.len().is_power_of_two() && b.len() > 1 {
                        ste::fast_hadamard(b);
                    }
                }
            };
            let mut rot = v.clone();
            for row in rot.chunks_exact_mut(cols) {
                for (x, s) in row.iter_mut().zip(&signs) {
                    *x *= s;
                }
                hb(row);
            }
            let mut q = ste::salt_quantize_forward_grouped_geometric(
                &rot,
                rows,
                cols,
                t,
                GROUP,
                GRID,
                RotationPolicy::Never,
            );
            for row in q.chunks_exact_mut(cols) {
                hb(row);
                for (x, s) in row.iter_mut().zip(&signs) {
                    *x *= s;
                }
            }
            q
        })
        .collect()
}

fn gptq_all(
    w: &[Vec<f32>],
    shapes: &[(usize, usize)],
    t: usize,
    grams: &TapGrams,
    tokens: usize,
) -> Vec<Vec<f32>> {
    w.iter()
        .zip(shapes)
        .enumerate()
        .map(|(i, (v, &(rows, cols)))| {
            let ladder = || {
                ste::salt_quantize_forward_grouped_geometric(
                    v,
                    rows,
                    cols,
                    t,
                    GROUP,
                    GRID,
                    RotationPolicy::Always,
                )
            };
            if i == 0 {
                return ladder();
            }
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
            let a = grams.for_slot((i - 1) / 7, (i - 1) % 7);
            fit_tensor(v, rows, cols, a, &cfg)
                .map_or_else(ladder, |f| fit_to_dense(&f, rows, cols, GROUP, true))
        })
        .collect()
}

/// Kendall's τ-b between two score vectors (lower = better in both).
fn kendall(x: &[f64], y: &[f64]) -> f64 {
    let (mut c, mut d) = (0i64, 0i64);
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            let s = (x[i] - x[j]).signum() * (y[i] - y[j]).signum();
            if s > 0.0 {
                c += 1;
            } else if s < 0.0 {
                d += 1;
            }
        }
    }
    (c - d) as f64 / (c + d).max(1) as f64
}

#[test]
#[ignore = "H5 metric validity; needs SmolLM2-135M; run explicitly"]
fn perplexity_against_on_policy_metrics() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let ts: Vec<usize> = std::env::var("TRITIUM_H5_TS")
        .unwrap_or_else(|_| "3,2".into())
        .split(',')
        .map(|s| s.trim().parse().expect("T"))
        .collect();
    let started = Instant::now();
    let mut fp_runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let (arch0, fp, shapes) = extract(&fp_runner);
    let (train, eval) = corpus();
    let mut calib = Calib::new(&arch0);
    for w in 0..FOLD_WINDOWS {
        calibrate(
            &fp,
            &arch0,
            &train[w * WINDOW..(w + 1) * WINDOW],
            &mut calib,
        );
    }
    let (folded, arch) = fold(&fp, &shapes, &arch0, &calib, 0.75);
    let (folded_half, arch_half) = fold(&fp, &shapes, &arch0, &calib, 0.5);
    let reference = Model {
        label: "fp".into(),
        weights: folded.clone(),
        arch: arch.clone(),
    };

    // Evaluation material.
    let eval_windows: Vec<&[u32]> = eval.chunks_exact(WINDOW).take(EVAL_WINDOWS).collect();
    let eval_tokens: Vec<u32> = eval_windows.concat();
    let kl_windows: Vec<&[u32]> = eval_windows[..KL_WINDOWS].to_vec();
    let late_windows: Vec<&[u32]> = eval.chunks_exact(LATE_WINDOW).take(LATE_WINDOWS).collect();
    let self_prompts: Vec<&[u32]> = (0..SELF_SEQS)
        .map(|i| &train[300_000 + i * 997..300_000 + i * 997 + 32])
        .collect();
    let self_text = sample_self_text(&mut fp_runner, &self_prompts, WINDOW, 0x5E1F);
    let self_windows: Vec<&[u32]> = self_text.iter().map(Vec::as_slice).collect();
    let prompts: Vec<&[u32]> = (0..PROMPTS)
        .map(|i| &eval[i * 1000..i * 1000 + PROMPT_LEN])
        .collect();
    let fp_greedy: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| greedy(&mut fp_runner, p, CONTINUATION))
        .collect();

    // Calibration Grams: WikiText 16k, WikiText 2k, and 16k of fp-sampled text (self-trace).
    let wiki16: Vec<&[u32]> = (0..32)
        .map(|w| &train[w * WINDOW..(w + 1) * WINDOW])
        .collect();
    let wiki2: Vec<&[u32]> = wiki16[..4].to_vec();
    let trace_prompts: Vec<&[u32]> = (0..32)
        .map(|i| &train[200_000 + i * 1009..200_000 + i * 1009 + 32])
        .collect();
    let trace_text = sample_self_text(&mut fp_runner, &trace_prompts, WINDOW, 0x7EACE);
    let trace: Vec<&[u32]> = trace_text.iter().map(Vec::as_slice).collect();
    let g16 = TapGrams::collect(&folded, &arch, &wiki16);
    let g2 = TapGrams::collect(&folded, &arch, &wiki2);
    let gself = TapGrams::collect(&folded, &arch, &trace);

    let fp_nll = perplexity_windowed(&folded, &arch, &eval_tokens, WINDOW).ln();
    let fp_late = nll_from(&reference, &late_windows, 512);
    println!(
        "SmolLM2-135M | fold α=0.75 (one arm α=0.5) | ladder g{GROUP} rotated | fp ppl {:.3} | setup {:.0}s\n",
        fp_nll.exp(),
        started.elapsed().as_secs_f64()
    );

    // Plumbing control: the folded fp model rebuilt through `runner_for` must continue exactly as
    // the fp runner does, or `agree` measures the rebuild rather than the quantizer.
    {
        let mut rebuilt = runner_for(&dir, &reference);
        let agree: f64 = prompts
            .iter()
            .zip(&fp_greedy)
            .map(|(p, want)| {
                let got = greedy(&mut rebuilt, p, CONTINUATION);
                got.iter().zip(want).take_while(|(a, b)| a == b).count() as f64
            })
            .sum::<f64>()
            / PROMPTS as f64;
        println!(
            "control: folded fp rebuilt through runner_for agrees {agree:.2}/{CONTINUATION} tokens\n"
        );
        assert!(
            agree >= 0.9 * CONTINUATION as f64,
            "runner_for does not reproduce the fp model ({agree:.2}/{CONTINUATION})"
        );
    }

    for &t in &ts {
        let models = vec![
            Model {
                label: "RTN".into(),
                weights: ladder_all(&folded, &shapes, t),
                arch: arch.clone(),
            },
            Model {
                label: "RTN fold α=0.5".into(),
                weights: ladder_all(&folded_half, &shapes, t),
                arch: arch_half.clone(),
            },
            Model {
                label: "GPTQ wiki 16k".into(),
                weights: gptq_all(&folded, &shapes, t, &g16, 16_384),
                arch: arch.clone(),
            },
            Model {
                label: "GPTQ wiki 2k".into(),
                weights: gptq_all(&folded, &shapes, t, &g2, 2_048),
                arch: arch.clone(),
            },
            Model {
                label: "GPTQ self-trace 16k".into(),
                weights: gptq_all(&folded, &shapes, t, &gself, 16_384),
                arch: arch.clone(),
            },
            Model {
                label: "RTN signs A (null)".into(),
                weights: signed_ladder_all(&folded, &shapes, t, 0xA),
                arch: arch.clone(),
            },
            Model {
                label: "RTN signs B (null)".into(),
                weights: signed_ladder_all(&folded, &shapes, t, 0xB),
                arch: arch.clone(),
            },
        ];
        println!("T={t}");
        println!(
            "{:<22} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7}",
            "model", "ppl", "even", "KL wiki", "KL self", "late", "agree"
        );
        let mut table: Vec<[f64; 6]> = Vec::new();
        for m in &models {
            let started = Instant::now();
            let base_fp = if m.label.contains("0.5") {
                &folded_half
            } else {
                &folded
            };
            let plus = perplexity_windowed(&m.weights, &m.arch, &eval_tokens, WINDOW).ln() - fp_nll;
            let anti: Vec<Vec<f32>> = base_fp
                .iter()
                .zip(&m.weights)
                .map(|(w, q)| w.iter().zip(q).map(|(a, b)| 2.0 * a - b).collect())
                .collect();
            let minus = perplexity_windowed(&anti, &m.arch, &eval_tokens, WINDOW).ln() - fp_nll;
            let kl_wiki = mean_kl(&reference, m, &kl_windows);
            let kl_self = mean_kl(&reference, m, &self_windows);
            let late = nll_from(m, &late_windows, 512) - fp_late;
            let mut runner = runner_for(&dir, m);
            let agree = prompts
                .iter()
                .zip(&fp_greedy)
                .map(|(p, want)| {
                    let got = greedy(&mut runner, p, CONTINUATION);
                    got.iter().zip(want).take_while(|(a, b)| a == b).count() as f64
                })
                .sum::<f64>()
                / PROMPTS as f64;
            let row = [
                (fp_nll + plus).exp(),
                0.5 * (plus + minus),
                kl_wiki,
                kl_self,
                late,
                agree,
            ];
            println!(
                "{:<22} {:>9.3} {:>+9.5} {:>9.5} {:>9.5} {:>+9.5} {:>7.2}   ({:.0}s)",
                m.label,
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                started.elapsed().as_secs_f64()
            );
            table.push(row);
        }
        let col = |c: usize| -> Vec<f64> {
            table
                .iter()
                .map(|r| if c == 5 { -r[c] } else { r[c] }) // more agreement is better
                .collect()
        };
        let names = ["ppl", "even", "KL wiki", "KL self", "late", "agree"];
        println!("Kendall τ vs ppl ranking:");
        for c in 1..6 {
            println!("  {:<8} {:+.3}", names[c], kendall(&col(0), &col(c)));
        }
        println!("Kendall τ vs KL self (on-policy) ranking:");
        for c in [0usize, 1, 2, 4, 5] {
            println!("  {:<8} {:+.3}", names[c], kendall(&col(3), &col(c)));
        }
        println!();
    }
}
