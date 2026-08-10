//! Real-model end-to-end SALT distillation on SmolLM2-135M (plan 0040 step 4) — the payoff.
//!
//! 0038 step 5 showed on this exact model that ternary PTQ is catastrophic (ppl 24→3.3M) and a
//! LOCAL layerwise heal cannot rescue it. 0038b proved END-TO-END distillation defeats depth on a
//! toy (98% recovery). 0040 steps 1-3 built + validated the real-model differentiable forward
//! (bit-exact vs ModelRunner). This test closes the loop: hold every 2D weight as an fp32 latent,
//! SALT-quantize it in that validated forward (STE), and distill ALL latents jointly against the fp
//! teacher's soft logits with AdamW — then show teacher-forced ppl recovers vs PTQ on the real
//! 30-layer model. `#[ignore]`d (slow, needs SmolLM2); run:
//!
//! ```text
//! cargo test -p tritium-nn --release --test salt_distill_real -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use tritium_nn::{Mlp, ModelRunner, Projection};
use tritium_train::nn::attention;
use tritium_train::ops::ste;
use tritium_train::{AdamState, AdamW, Muon, MuonState, Optimizer, Tape, ValueId};

/// Per-leaf optimizer: AdamW everywhere, or the hybrid Muon (2D hidden weights) + AdamW (embedding).
enum Opt {
    Adam(AdamW, AdamState),
    Muon(Muon, MuonState),
}

const T: usize = 2; // SALT planes (matches 0038 step 5's PTQ config)
const STEPS: u64 = 8; // smoke default; the real run overrides via TRITIUM_DISTILL_STEPS
const LR: f32 = 2e-3;

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}

/// The committed coherent-text prompt (real token ids → sensible fp ppl), truncated for speed.
fn eval_tokens() -> Vec<u32> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tools/reference/smollm2_ref.json"
    );
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).expect("ref json")).expect("parse");
    json["prompt_ids"]
        .as_array()
        .expect("prompt_ids")
        .iter()
        .take(12)
        .map(|v| v.as_u64().expect("id") as u32)
        .collect()
}

fn dense(p: &Projection) -> (Vec<f32>, usize, usize) {
    match p {
        Projection::Dense(d) => (d.weights.clone(), d.n_out, d.k_in),
        Projection::Salt(_)
        | Projection::HostSaltV2(_)
        | Projection::Ternary(_)
        | Projection::Q2(_) => {
            panic!("from_hf builds Dense projections")
        }
        #[cfg(feature = "cuda")]
        Projection::SaltV2(_) => panic!("from_hf must not build resident SALT V2 projections"),
    }
}

/// Extracted 1D norms + dims for the tape forward (the parts held fp, not trained).
struct Arch {
    attn_norms: Vec<Vec<f32>>,
    ffn_norms: Vec<Vec<f32>>,
    out_norm: Vec<f32>,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    ff: usize,
    vocab: usize,
    eps: f32,
    theta: f32,
    n_layers: usize,
}

/// Build the model forward on the tape from a flat weight-id list (index 0 = tied embed/head, then
/// per layer q,k,v,o,gate,up,down). Returns `[seq, vocab]` logits.
#[allow(clippy::too_many_arguments)]
fn forward(t: &mut Tape, wids: &[ValueId], a: &Arch, tokens: &[u32]) -> ValueId {
    let seq = tokens.len();
    let mut hidden = t.embed_gather(wids[0], tokens, a.vocab, a.n_embd);
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        let an = t.leaf(a.attn_norms[li].clone());
        let xn = t.rmsnorm(hidden, an, seq, a.n_embd, a.eps);
        let attn = attention(
            t,
            xn,
            wids[base],
            wids[base + 1],
            wids[base + 2],
            wids[base + 3],
            seq,
            a.n_embd,
            a.n_head,
            a.n_head_kv,
            a.head_dim,
            a.theta,
        );
        hidden = t.add(hidden, attn);
        let fnw = t.leaf(a.ffn_norms[li].clone());
        let hn = t.rmsnorm(hidden, fnw, seq, a.n_embd, a.eps);
        let g = t.dense_matmul(hn, wids[base + 4], seq, a.ff, a.n_embd);
        let u = t.dense_matmul(hn, wids[base + 5], seq, a.ff, a.n_embd);
        let ga = t.silu(g);
        let gated = t.mul(ga, u);
        let down = t.dense_matmul(gated, wids[base + 6], seq, a.n_embd, a.ff);
        hidden = t.add(hidden, down);
    }
    let onw = t.leaf(a.out_norm.clone());
    let fnorm = t.rmsnorm(hidden, onw, seq, a.n_embd, a.eps);
    t.dense_matmul(fnorm, wids[0], seq, a.vocab, a.n_embd) // tied head
}

/// Teacher-forced perplexity from `[seq, vocab]` logits over `tokens`.
fn perplexity(logits: &[f32], tokens: &[u32], vocab: usize) -> f64 {
    let seq = tokens.len();
    let mut nll = 0.0f64;
    for tpos in 0..seq - 1 {
        let row = &logits[tpos * vocab..tpos * vocab + vocab];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = m + row
            .iter()
            .map(|&x| (f64::from(x) - m).exp())
            .sum::<f64>()
            .ln();
        nll += lse - f64::from(row[tokens[tpos + 1] as usize]);
    }
    (nll / (seq - 1) as f64).exp()
}

/// Forward once with a fixed weight set (leaves), returning `[seq, vocab]` logits values.
fn logits_of(weights: &[Vec<f32>], a: &Arch, tokens: &[u32]) -> Vec<f32> {
    let mut t = Tape::new();
    let wids: Vec<ValueId> = weights.iter().map(|w| t.leaf(w.clone())).collect();
    let out = forward(&mut t, &wids, a, tokens);
    t.value(out).to_vec()
}

#[test]
#[ignore = "slow real-model distillation; needs SmolLM2-135M; run explicitly"]
fn salt_distillation_recovers_smollm2_perplexity() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let steps: u64 = std::env::var("TRITIUM_DISTILL_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(STEPS);
    // SALT plane count; T=1 is a "lesser" single-plane ternary STE (plain BitNet b1.58 QAT),
    // T≥2 is SALT residual expansion. Override to compare (plan 0041 / SALT-vs-STE study).
    let tp: usize = std::env::var("TRITIUM_DISTILL_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(T);

    let runner =
        ModelRunner::from_hf(&dir, Box::new(tritium_cpu::CpuBackend::new())).expect("from_hf");
    let cfg = &runner.config;
    let w = &runner.weights;
    assert!(w.lm_head.is_none(), "assumes tied lm-head");
    let a = Arch {
        attn_norms: w.layers.iter().map(|b| b.attn_norm.clone()).collect(),
        ffn_norms: w.layers.iter().map(|b| b.ffn_norm.clone()).collect(),
        out_norm: w.output_norm.clone(),
        n_embd: cfg.n_embd as usize,
        n_head: cfg.n_head as usize,
        n_head_kv: cfg.n_head_kv as usize,
        head_dim: cfg.head_dim() as usize,
        ff: match &w.layers[0].mlp {
            Mlp::SwiGlu(m) => dense(&m.gate).1,
            Mlp::Relu2(_) => panic!("SwiGLU"),
        },
        vocab: w.vocab,
        eps: cfg.rms_eps,
        theta: cfg.rope_theta,
        n_layers: w.layers.len(),
    };

    // Flat fp weights (index 0 = tied token_embd, then per layer q,k,v,o,gate,up,down) + shapes.
    let mut fp: Vec<Vec<f32>> = vec![
        w.token_embd
            .as_dense()
            .expect("fp teacher requires dense token embedding")
            .to_vec(),
    ];
    let mut shapes: Vec<(usize, usize)> = vec![(w.vocab, a.n_embd)];
    for b in w.layers.iter() {
        let (gate, up, down) = match &b.mlp {
            Mlp::SwiGlu(m) => (&m.gate, &m.up, &m.down),
            Mlp::Relu2(_) => unreachable!(),
        };
        for p in [&b.q_proj, &b.k_proj, &b.v_proj, &b.o_proj, gate, up, down] {
            let (wv, n, k) = dense(p);
            fp.push(wv);
            shapes.push((n, k));
        }
    }

    let eval = eval_tokens();

    // fp teacher: logits + soft targets + baseline ppl.
    let teacher_logits = logits_of(&fp, &a, &eval);
    let ppl_fp = perplexity(&teacher_logits, &eval, a.vocab);
    let seq = eval.len();
    let teacher_probs: Vec<f32> = {
        // per-row softmax of the teacher logits
        let mut p = teacher_logits.clone();
        for row in p.chunks_mut(a.vocab) {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0.0f32;
            for v in row.iter_mut() {
                *v = (*v - m).exp();
                s += *v;
            }
            for v in row.iter_mut() {
                *v /= s;
            }
        }
        p
    };

    // PTQ baseline (SALT-quantize all latents, no training).
    let ptq: Vec<Vec<f32>> = fp
        .iter()
        .zip(&shapes)
        .map(|(wf, &(n, k))| ste::salt_quantize_forward(wf, n, k, tp))
        .collect();
    let ppl_ptq = perplexity(&logits_of(&ptq, &a, &eval), &eval, a.vocab);

    // Distill: latents start at fp; SALT-STE in the forward; the optimizer steps all latents against
    // the teacher's soft logits (softmax_xent = KL gradient). TRITIUM_DISTILL_OPT=muon uses the
    // hybrid Muon (2D hidden weights) + AdamW (the tied embedding) recipe — half the optimizer state.
    let use_muon = std::env::var("TRITIUM_DISTILL_OPT")
        .map(|s| s == "muon")
        .unwrap_or(false);
    let muon_lr: f32 = std::env::var("TRITIUM_MUON_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.02);
    let mut lat = fp.clone();
    let mut opts: Vec<Opt> = (0..lat.len())
        .map(|i| {
            if use_muon && i > 0 {
                let (n, k) = shapes[i];
                let m = Muon::new(muon_lr, n, k);
                let s = m.init_state(lat[i].len());
                Opt::Muon(m, s)
            } else {
                let ad = AdamW::new(LR);
                let s = ad.init_state(lat[i].len());
                Opt::Adam(ad, s)
            }
        })
        .collect();
    let (mut first, mut last) = (f32::NAN, f32::NAN);
    for step in 1..=steps {
        let mut t = Tape::new();
        let mut leaf_ids = Vec::with_capacity(lat.len());
        let mut ste_ids = Vec::with_capacity(lat.len());
        for (i, wv) in lat.iter().enumerate() {
            let l = t.leaf(wv.clone());
            let (n, k) = shapes[i];
            ste_ids.push(t.salt_ste(l, n, k, tp));
            leaf_ids.push(l);
        }
        let logits = forward(&mut t, &ste_ids, &a, &eval);
        let tg = t.leaf(teacher_probs.clone());
        let l = t.softmax_xent(logits, tg, seq, a.vocab);
        let lv = t.value(l)[0];
        if step == 1 {
            first = lv;
        }
        last = lv;
        let grads = t.backward(l);
        for i in 0..lat.len() {
            match &mut opts[i] {
                Opt::Adam(o, s) => o.step(step, &mut lat[i], &grads[leaf_ids[i]], s),
                Opt::Muon(o, s) => o.step(step, &mut lat[i], &grads[leaf_ids[i]], s),
            }
        }
        if step % 10 == 0 || step == 1 {
            eprintln!("  step {step}/{steps}  xent {lv:.4}");
        }
    }

    let distilled: Vec<Vec<f32>> = lat
        .iter()
        .zip(&shapes)
        .map(|(wf, &(n, k))| ste::salt_quantize_forward(wf, n, k, tp))
        .collect();
    let ppl_distilled = perplexity(&logits_of(&distilled, &a, &eval), &eval, a.vocab);

    println!(
        "0040 step4 SmolLM2 real distillation (T={tp}, {steps} steps, opt={}, IN-SAMPLE {seq}-tok calib): \
         fp ppl {ppl_fp:.3} | PTQ ppl {ppl_ptq:.3e} | distilled ppl {ppl_distilled:.3e}  \
         (surrogate xent {first:.4}→{last:.4}). Distilled ppl is {:.1}× lower than PTQ — end-to-end \
         distillation recovers the real 30-layer model where a local heal (0038 step 5) could not. \
         (distilled ≤ fp is in-sample overfit to this one sequence; held-out generalization = a real \
         corpus, 0041 scale.)",
        if use_muon {
            format!("muon+adam(mlr={muon_lr})")
        } else {
            "adamw".to_string()
        },
        ppl_ptq / ppl_distilled
    );

    assert!(ppl_ptq > ppl_fp, "PTQ must degrade ppl");
    assert!(last < first, "distillation surrogate must decrease");
    assert!(
        ppl_distilled < ppl_ptq,
        "end-to-end distillation must recover real-model ppl vs PTQ (the thing local heal could not): \
         distilled {ppl_distilled:.3e} vs PTQ {ppl_ptq:.3e}"
    );
}
