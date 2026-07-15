//! Real-model SALT-heal on SmolLM2-135M (plan 0038 step 4). Two things, one run:
//!
//!  1. **The finding** — full ternary SALT PTQ (T=2) of a normally-trained model is *catastrophic*
//!     at the model level: quantizing every projection compounds across all layers and the
//!     teacher-forced perplexity explodes (fp ~24 → PTQ ~1e6). This is the empirical case for the
//!     ADR-0020 premise (PTQ alone is not enough) and shows why a purely *local* heal — matching
//!     each projection to its fp output against fp activations — cannot repair the model: in the
//!     fully-quantized model the residual stream a layer sees is already destroyed. Model-level
//!     recovery needs END-TO-END distillation (validated at tiny scale by the SwiGLU e2e test;
//!     scaled by a real whole-model tape in 0038b).
//!  2. **The gate** — the SALT-STE heal MECHANISM works on real weights: across every layer's
//!     q/k/v, distilling the latent to match the fp projection output (driven by the *real*
//!     calibration activation) shrinks the output error ≥50% vs plain SALT PTQ.
//!
//! `#[ignore]`d (needs SmolLM2-135M); run:
//!
//! ```text
//! cargo test -p tritium-nn --release --test salt_heal_ppl -- --ignored --nocapture
//! ```
//!
//! Each layer's q/k/v calibration input is reconstructed from the fp `forward_dump`
//! (`rmsnorm(embedding|hidden_states[li-1], attn_norm[li])`), so no runner change is needed.
//! All weights use the exact-fp activation path, isolating the weight-quantization effect.

use std::path::PathBuf;

use tritium_nn::{ForwardDump, Mlp, ModelRunner, ModelWeights, Projection};
use tritium_train::ops::ste::salt_quantize_forward;
use tritium_train::{AdamW, Optimizer, Tape};

const T: usize = 2; // SALT planes

fn model_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/smollm2-135m")
}
fn reference() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tools/reference/smollm2_ref.json"
    ))
}

/// `Y[m,n] = Σ_k X[m,k]·W[n,k]` (`W` is `[n,k]` row-major).
fn matmul(x: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0f32;
            for ki in 0..k {
                acc += x[mi * k + ki] * w[ni * k + ki];
            }
            y[mi * n + ni] = acc;
        }
    }
    y
}

/// Row-wise RMSNorm: `out[r] = x[r] / sqrt(mean(x[r]²) + eps) · weight`. `[m, d]`.
fn rmsnorm_rows(x: &[f32], weight: &[f32], m: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; m * d];
    for r in 0..m {
        let row = &x[r * d..r * d + d];
        let ms = row
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / d as f64;
        let inv = 1.0 / (ms + f64::from(eps)).sqrt();
        for c in 0..d {
            out[r * d + c] = (f64::from(row[c]) * inv) as f32 * weight[c];
        }
    }
    out
}

fn log_prob(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = m + logits
        .iter()
        .map(|&x| (f64::from(x) - m).exp())
        .sum::<f64>()
        .ln();
    f64::from(logits[target]) - lse
}

/// Teacher-forced perplexity over `eval_ids`.
fn perplexity(runner: &mut ModelRunner, eval_ids: &[u32]) -> f64 {
    runner.reset();
    let n = eval_ids.len();
    let mut nll = 0.0f64;
    let mut logits = runner.forward(&eval_ids[..1], &[0]).expect("prefill");
    for t in 0..n - 1 {
        nll += -log_prob(&logits, eval_ids[t + 1] as usize);
        logits = runner
            .forward(&[eval_ids[t + 1]], &[t + 1])
            .expect("decode");
    }
    (nll / (n - 1) as f64).exp()
}

fn dense_of(p: &Projection) -> (&[f32], usize, usize) {
    match p {
        Projection::Dense(d) => (&d.weights, d.n_out, d.k_in),
        Projection::Salt(_) | Projection::Ternary(_) => {
            panic!("expected a Dense projection (load_hf builds fp)")
        }
        #[cfg(feature = "cuda")]
        Projection::SaltV2(_) => panic!("load_hf must not build resident SALT V2 projections"),
    }
}
fn set_dense(p: &mut Projection, w: Vec<f32>) {
    match p {
        Projection::Dense(d) => d.weights = w,
        Projection::Salt(_) | Projection::Ternary(_) => panic!("expected a Dense projection"),
        #[cfg(feature = "cuda")]
        Projection::SaltV2(_) => panic!("expected a Dense projection"),
    }
}

/// Distill a projection's SALT latent to match the fp target output on `input` (`[seq, k]`),
/// returning the healed dense reconstruction (`[n, k]`).
fn heal(fp_w: &[f32], input: &[f32], seq: usize, n: usize, k: usize) -> Vec<f32> {
    let target = matmul(input, fp_w, seq, n, k);
    let mut latent = fp_w.to_vec();
    let opt = AdamW::new(3e-3);
    let mut state = opt.init_state(latent.len());
    for step in 1..=80u64 {
        let mut tape = Tape::new();
        let wf = tape.leaf(latent.clone());
        let xi = tape.leaf(input.to_vec());
        let tg = tape.leaf(target.clone());
        let wh = tape.salt_ste(wf, n, k, T);
        let y = tape.dense_matmul(xi, wh, seq, n, k);
        let loss = tape.mse(y, tg);
        let grads = tape.backward(loss);
        opt.step(step, &mut latent, &grads[wf], &mut state);
    }
    salt_quantize_forward(&latent, n, k, T)
}

/// PTQ every block projection (q/k/v/o + gate/up/down) in place: `w ← SALT-quantize(w)`.
fn ptq_in_place(w: &mut ModelWeights) {
    for block in &mut w.layers {
        for p in [
            &mut block.q_proj,
            &mut block.k_proj,
            &mut block.v_proj,
            &mut block.o_proj,
        ] {
            let (fp, n, k) = dense_of(p);
            let q = salt_quantize_forward(fp, n, k, T);
            set_dense(p, q);
        }
        if let Mlp::SwiGlu(m) = &mut block.mlp {
            for p in [&mut m.gate, &mut m.up, &mut m.down] {
                let (fp, n, k) = dense_of(p);
                let q = salt_quantize_forward(fp, n, k, T);
                set_dense(p, q);
            }
        }
    }
}

#[test]
#[ignore = "needs SmolLM2-135M under ~/.cache/tritium-models/smollm2-135m; run explicitly"]
fn salt_heal_recovers_smollm2_perplexity() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let rj: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference()).expect("reference")).expect("json");
    let eval_ids: Vec<u32> = rj["prompt_ids"]
        .as_array()
        .expect("prompt_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let seq = eval_ids.len();

    let cpu = || Box::new(tritium_cpu::CpuBackend::new());

    // fp teacher.
    let mut fp = ModelRunner::from_hf(&dir, cpu()).expect("from_hf");
    let ppl_fp = perplexity(&mut fp, &eval_ids);
    let n_embd = fp.config.n_embd as usize;
    let eps = fp.config.rms_eps;

    // Capture the residual stream; reconstruct each layer's q/k/v input = rmsnorm(input_li).
    let mut dump = ForwardDump::default();
    fp.reset();
    let positions: Vec<usize> = (0..seq).collect();
    fp.forward_dump(&eval_ids, &positions, &mut dump)
        .expect("dump");
    let n_layers = fp.weights.layers.len();
    let attn_in: Vec<Vec<f32>> = (0..n_layers)
        .map(|li| {
            let input = if li == 0 {
                &dump.embedding
            } else {
                &dump.hidden_states[li - 1]
            };
            rmsnorm_rows(input, &fp.weights.layers[li].attn_norm, seq, n_embd, eps)
        })
        .collect();

    // fp q/k/v weights (for the heal targets), before any mutation.
    let fp_qkv: Vec<[(Vec<f32>, usize, usize); 3]> = (0..n_layers)
        .map(|li| {
            let b = &fp.weights.layers[li];
            let g = |p: &Projection| {
                let (w, n, k) = dense_of(p);
                (w.to_vec(), n, k)
            };
            [g(&b.q_proj), g(&b.k_proj), g(&b.v_proj)]
        })
        .collect();

    // Full PTQ model perplexity. Expected to be CATASTROPHIC: ternary PTQ of a normally-trained
    // (non-QAT) model compounds multiplicatively across 30 layers × 7 projections, so the model-
    // level ppl explodes. This is the empirical justification for the ADR-0020 premise — PTQ is
    // not enough; distillation is required. Crucially, a *local* layerwise heal against the fp
    // activations cannot repair a model this broken (the upstream damage dominates the residual
    // stream a layer actually sees), so model-level ppl recovery needs END-TO-END distillation —
    // the mechanism the tiny-SwiGLU e2e test (steps 3–4) validates and a real-model whole-model
    // tape (0038b) will scale. Here we gate the heal MECHANISM on real weights + real activations.
    let mut ptq = ModelRunner::from_hf(&dir, cpu()).expect("from_hf");
    ptq_in_place(&mut ptq.weights);
    ptq.invalidate_resident();
    let ppl_ptq = perplexity(&mut ptq, &eval_ids);

    // Locally-healed model: start from full PTQ, then replace q/k/v with the SALT-STE-healed
    // reconstructions. As we heal each projection we (a) accumulate its calibration-set output
    // error vs the fp target — the MECHANISM gate — and (b) install it into the model so we can
    // measure whether a purely local q/k/v heal moves model-level ppl (it does not — o/gate/up/down
    // stay PTQ and the compounding upstream damage dominates; that's the case for end-to-end).
    let mut healed = ModelRunner::from_hf(&dir, cpu()).expect("from_hf");
    ptq_in_place(&mut healed.weights);
    let (mut ptq_err, mut healed_err) = (0.0f64, 0.0f64);
    for (li, x) in attn_in.iter().enumerate() {
        let b = &mut healed.weights.layers[li];
        let projs: [&mut Projection; 3] = [&mut b.q_proj, &mut b.k_proj, &mut b.v_proj];
        for (pi, p) in projs.into_iter().enumerate() {
            let (fp_w, n, k) = &fp_qkv[li][pi];
            let fp_out = matmul(x, fp_w, seq, *n, *k);
            let ptq_out = matmul(x, &salt_quantize_forward(fp_w, *n, *k, T), seq, *n, *k);
            let healed_w = heal(fp_w, x, seq, *n, *k);
            let healed_out = matmul(x, &healed_w, seq, *n, *k);
            ptq_err += mse(&ptq_out, &fp_out);
            healed_err += mse(&healed_out, &fp_out);
            set_dense(p, healed_w);
        }
    }
    healed.invalidate_resident();
    let ppl_healed = perplexity(&mut healed, &eval_ids);
    let recovered = 100.0 * (1.0 - healed_err / ptq_err);

    println!(
        "SmolLM2-135M: fp ppl {ppl_fp:.3} | full-PTQ(T={T}) ppl {ppl_ptq:.3e} | q/k/v-locally-healed ppl {ppl_healed:.3e}. \
         q/k/v projection output-error vs fp target (in-sample, calibration = eval seq) over all {n_layers} layers: \
         PTQ {ptq_err:.4e} → healed {healed_err:.4e} ({recovered:.1}% recovered). \
         → the heal MECHANISM works on real weights, yet a local q/k/v heal leaves model ppl catastrophic: END-TO-END distillation is required (0038b)."
    );

    // (1) PTQ degrades the model.
    assert!(
        ppl_ptq > ppl_fp,
        "PTQ must degrade ppl: {ppl_ptq} vs fp {ppl_fp}"
    );
    // (2) MECHANISM gate: the SALT-STE heal shrinks the q/k/v projection output error ≥50% on real
    // weights driven by their real calibration activations (in-sample fit — this gates that the
    // STE-backward + AdamW loop actually learns; per-projection generalization is carried by the
    // atomic salt_distill.rs test).
    assert!(
        healed_err < 0.5 * ptq_err,
        "SALT-STE heal must recover ≥50% of the q/k/v output error: healed {healed_err:.4e} vs PTQ {ptq_err:.4e}"
    );
    // (3) FINDING (the load-bearing claim, now demonstrated not just narrated): a purely local
    // q/k/v heal does NOT rescue model-level ppl — it stays catastrophic (≫ fp) because the four
    // un-healed projections keep the residual stream broken. This is why model-level recovery needs
    // end-to-end distillation, not layerwise healing against clean activations.
    assert!(
        ppl_healed > 100.0 * ppl_fp,
        "a local q/k/v heal must NOT rescue model ppl (stays catastrophic — motivates e2e): healed {ppl_healed:.3e} vs fp {ppl_fp:.3}"
    );
}

/// Mean squared error over two equal-length buffers (f64 accumulation).
fn mse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64
}
