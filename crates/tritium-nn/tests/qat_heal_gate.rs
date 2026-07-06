//! v0.5.0 capstone gate (ADR 0007, plan 0010): the QAT machinery (AdamW + STE + tape)
//! drives a real BitNet-2b4t layer's ternary **distillation loss** down by **≥90%** on
//! real model activations, end-to-end through the heal bridge — a convergence smoke.
//!
//! What is gated is *distillation-loss convergence*, NOT a full-model perplexity recovery.
//! BitNet's bf16 master is the QAT *latent* weight (it runs as garbage densely — see
//! `salt_accuracy.rs`) and is already per-tensor-QAT-optimal, so naive ternary of it ≈ the
//! deployed model (no meaningful "vs fp16" or "naive-quant" gap), and layerwise distillation
//! from a short eval slice is underdetermined for 2560-wide layers — full-model PPL recovery
//! is therefore deliberately reported as *context*, not gated (a v0.60 full-backprop item;
//! see plan 0010 + memory `bitnet-2b4t-qat-recovery-gotchas`).
//!
//! Shape of the test: install per-tensor-ternary(master) ("good", ≈ deployed) → under-bit a
//! slice of early-layer q/k/v to 1-bit (PPL context, exercises the swap+PPL bridge) → QAT
//! each projection from the latent master toward the good per-tensor-ternary output and
//! assert the aggregate (and per-projection) distillation loss drops ≥90%, every step finite.
//! 1-bit is stored as ternary trits in {-1,+1} so the model stays all-ternary (fast resident PPL).
//!
//! GPU + on-disk-model gated; self-skips otherwise.
#![cfg(feature = "cuda")]

use std::path::Path;

use tritium_core::Trit;
use tritium_format::SafeTensors;
use tritium_nn::{ForwardDump, ModelRunner};
use tritium_train::ops::{dense, ste};
use tritium_train::{AdamW, Optimizer, Tape};

use tritium_cpu as _;
use tritium_cuda as _;
use tritium_runtime as _;

const GGUF_PATH: &str =
    "/home/brianklam/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf";
const BF16_PATH: &str = "/home/brianklam/.cache/tritium-models/bitnet-2b4t-bf16/model.safetensors";
const REF_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/reference/bitnet_accept.json"
);

/// Number of early transformer layers whose q/k/v are under-bitted + healed.
const SLICE_LAYERS: usize = 6;
/// Activation rows used to train each layerwise heal (subset keeps the CPU tape fast).
const M_TRAIN: usize = 32;
/// QAT optimizer steps per projection.
const QAT_STEPS: u64 = 60;

#[derive(serde::Deserialize)]
struct Reference {
    eval_ids: Vec<u32>,
}

fn maybe_load() -> Option<(Vec<u32>, Vec<u8>, Vec<u8>)> {
    for (p, what) in [
        (GGUF_PATH, "gguf model"),
        (BF16_PATH, "bf16 master"),
        (REF_PATH, "reference json"),
    ] {
        if !Path::new(p).exists() {
            eprintln!("skipping qat_heal_gate: {what} absent ({p})");
            return None;
        }
    }
    let reference: Reference =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse ref");
    Some((
        reference.eval_ids,
        std::fs::read(GGUF_PATH).expect("read gguf"),
        std::fs::read(BF16_PATH).expect("read bf16"),
    ))
}

fn load_cuda(bytes: &[u8]) -> Option<ModelRunner> {
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")
        .map(|e| e.init)?;
    let backend = match init() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping qat_heal_gate: cuda init failed ({e}); no device?");
            return None;
        }
    };
    let file = tritium_format::read_gguf(bytes).expect("parse gguf");
    Some(ModelRunner::load(&file, bytes, backend).expect("load model"))
}

fn log_prob_of(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)) as f64;
    let sum: f64 = logits.iter().map(|&l| ((l as f64) - max).exp()).sum();
    (logits[target] as f64) - max - sum.ln()
}

fn perplexity(runner: &mut ModelRunner, eval_ids: &[u32]) -> f64 {
    runner.reset();
    let n = eval_ids.len();
    let mut neg_log_sum = 0.0f64;
    let mut count = 0usize;
    let mut logits = runner.forward(&eval_ids[..1], &[0]).expect("prefill");
    for t in 0..n - 1 {
        neg_log_sum -= log_prob_of(&logits, eval_ids[t + 1] as usize);
        count += 1;
        if t + 1 < n - 1 {
            logits = runner
                .forward(&[eval_ids[t + 1]], &[t + 1])
                .expect("decode");
        }
    }
    (neg_log_sum / count as f64).exp()
}

/// RMSNorm each `[e]` row of `x` (`[rows, e]`): `out = x / sqrt(mean(x²)+eps) · w`.
/// Same formula as `tritium_nn::ops::rmsnorm` with a plain sequential sum — NOT
/// bit-identical since ADR 0018 (canonical tree order); fine here, every assertion
/// in this gate is loss/perplexity-threshold based, not bit-based.
fn rmsnorm_rows(x: &[f32], w: &[f32], eps: f32, e: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (xr, or) in x.chunks_exact(e).zip(out.chunks_exact_mut(e)) {
        let mean_sq = xr.iter().map(|v| v * v).sum::<f32>() / e as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for ((o, &xi), &wi) in or.iter_mut().zip(xr).zip(w) {
            *o = xi * inv * wi;
        }
    }
    out
}

fn to_trits(vals: &[f32]) -> Vec<Trit> {
    vals.iter()
        .map(|&x| {
            if x > 0.5 {
                Trit::POS
            } else if x < -0.5 {
                Trit::NEG
            } else {
                Trit::ZERO
            }
        })
        .collect()
}

fn dequant(trits: &[f32], scale: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n * k];
    for ni in 0..n {
        for ki in 0..k {
            w[ni * k + ki] = scale[ni] * trits[ni * k + ki];
        }
    }
    w
}

/// 1-bit (sign-only) trits: `sign(w) ∈ {-1,+1}`, no zero level. Strictly coarser than
/// ternary — the induced under-bitting degradation. The scale is supplied separately.
fn onebit(master: &[f32]) -> Vec<f32> {
    master
        .iter()
        .map(|&w| if w >= 0.0 { 1.0 } else { -1.0 })
        .collect()
}

/// Per-TENSOR absmean scale, broadcast to `n` rows. BitNet b1.58 is QAT-trained against a
/// single per-tensor absmean ternary (see `salt_accuracy.rs`); a finer per-row scale
/// reconstructs the unusable latent master → garbage. The heal must use this granularity.
fn per_tensor_scale(w: &[f32], n: usize) -> Vec<f32> {
    let pt = w.iter().map(|v| v.abs()).sum::<f32>() / w.len() as f32;
    vec![pt; n]
}

/// One slice projection to heal: its dims, master weights, full + train inputs, and the
/// good (per-tensor-ternary) distillation target on the train inputs.
struct Slice {
    layer: usize,
    which: &'static str, // "q" | "k" | "v"
    n: usize,
    k: usize,
    master: Vec<f32>,
    x_train: Vec<f32>,      // [m, k]
    good_trits: Vec<f32>,   // per-tensor ternary of master (the deployed-equivalent)
    good_scale: Vec<f32>,   // [n]
    target_train: Vec<f32>, // [m, n] = good layer output on x_train
}

/// Install a projection's ternary weight + per-row scale, then drop the resident decoder.
fn install(runner: &mut ModelRunner, layer: usize, which: &str, trits: &[Trit], scale: Vec<f32>) {
    let block = &mut runner.weights.layers[layer];
    let proj = match which {
        "q" => &mut block.q_proj,
        "k" => &mut block.k_proj,
        "v" => &mut block.v_proj,
        _ => unreachable!(),
    };
    proj.as_ternary_mut()
        .expect("ternary proj")
        .replace_weights(runner.backend.as_ref(), trits, scale)
        .expect("replace_weights");
    runner.invalidate_resident();
}

#[test]
fn qat_distillation_converges_on_real_model_slice() {
    let Some((eval_ids, gguf, bf16)) = maybe_load() else {
        return;
    };
    assert!(eval_ids.len() >= 2, "need ≥2 eval tokens");
    let Some(mut runner) = load_cuda(&gguf) else {
        return;
    };
    let seq = eval_ids.len();
    let e = runner.config.n_embd as usize;
    let eps = runner.config.rms_eps;

    // Capture the residual stream over the eval slice (one host prefill).
    let mut dump = ForwardDump::default();
    let positions: Vec<usize> = (0..seq).collect();
    runner.reset();
    runner
        .forward_dump(&eval_ids, &positions, &mut dump)
        .expect("forward_dump");
    runner.reset();

    // bf16 master weights.
    let st = SafeTensors::parse(&bf16).expect("parse safetensors");

    // Assemble the slice: q/k/v of the first SLICE_LAYERS layers.
    let mut slices: Vec<Slice> = Vec::new();
    for layer in 0..SLICE_LAYERS.min(runner.weights.layers.len()) {
        // Input to this layer's q/k/v = rmsnorm(residual_in, attn_norm).
        let resid_in: &[f32] = if layer == 0 {
            &dump.embedding
        } else {
            &dump.hidden_states[layer - 1]
        };
        let attn_norm = runner.weights.layers[layer].attn_norm.clone();
        let x_full = rmsnorm_rows(resid_in, &attn_norm, eps, e);
        let m = M_TRAIN.min(seq);
        let x_train = x_full[..m * e].to_vec();

        for which in ["q", "k", "v"] {
            let n = match which {
                "q" => runner.weights.layers[layer].q_proj.n_out(),
                "k" => runner.weights.layers[layer].k_proj.n_out(),
                _ => runner.weights.layers[layer].v_proj.n_out(),
            };
            let k = e;
            let name = format!("model.layers.{layer}.self_attn.{which}_proj.weight");
            let master = st
                .tensor_f32(&name)
                .unwrap_or_else(|err| panic!("{name}: {err}"));
            assert_eq!(master.len(), n * k, "{name} shape");
            // Good reference = per-tensor-absmean ternary of the master (≈ deployed I2_S).
            let pt = master.iter().map(|v| v.abs()).sum::<f32>() / master.len() as f32;
            let good_scale = vec![pt; n];
            let good_trits = ste::quantize_forward(&master, &good_scale, n, k);
            let good_dense = dequant(&good_trits, &good_scale, n, k);
            let target_train = dense::forward(&x_train, &good_dense, m, n, k);
            slices.push(Slice {
                layer,
                which,
                n,
                k,
                master,
                x_train: x_train.clone(),
                good_trits,
                good_scale,
                target_train,
            });
        }
    }
    eprintln!(
        "slice: {} projections across {} layers (q/k/v); eval tokens {seq}",
        slices.len(),
        SLICE_LAYERS
    );

    // --- Good: install per-tensor-ternary (deployed-equivalent) for the whole slice. ---
    for s in &slices {
        install(
            &mut runner,
            s.layer,
            s.which,
            &to_trits(&s.good_trits),
            s.good_scale.clone(),
        );
    }
    let ppl_good = perplexity(&mut runner, &eval_ids);

    // --- Degraded: under-bit the whole slice to 1-bit (sign-only, per-tensor scale). ---
    for s in &slices {
        let s_pt = per_tensor_scale(&s.master, s.n);
        let ob = onebit(&s.master);
        install(&mut runner, s.layer, s.which, &to_trits(&ob), s_pt);
    }
    let ppl_degraded = perplexity(&mut runner, &eval_ids);
    let gap = ppl_degraded - ppl_good;
    eprintln!("PPL good={ppl_good:.4} degraded(1-bit)={ppl_degraded:.4} gap={gap:.4}");
    assert!(
        gap > 0.02 * ppl_good,
        "induced 1-bit gap too small ({gap:.4} on {ppl_good:.4}); widen SLICE_LAYERS"
    );

    // --- Heal: STE-ternary QAT per projection, distilling to the good output. ---
    let opt = AdamW {
        lr: 3e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let mut total_first = 0.0f64;
    let mut total_last = 0.0f64;
    let mut min_proj_recovery = f64::INFINITY;
    for s in &slices {
        let (n, k) = (s.n, s.k);
        let m = s.x_train.len() / k;
        // Heal from the latent master (the standard QAT init): its values are spread, so
        // the STE mask `1[|Wf/s_q|<1]` lets gradients flow. Initializing at the 1-bit
        // values (±absmean) would sit at |Wf/s_q|≈1 — the clamp boundary — and zero every
        // gradient, freezing the optimizer.
        let mut wf = s.master.clone();
        let mut state = opt.init_state(wf.len());
        let mut first = 0.0f32;
        let mut last = 0.0f32;
        for step in 1..=QAT_STEPS {
            // Per-TENSOR scale (BitNet's training granularity), recomputed from the
            // current latent and broadcast to all rows.
            let s_q = per_tensor_scale(&wf, n);
            let mut tape = Tape::new();
            let wf_id = tape.leaf(wf.clone());
            let sq_id = tape.leaf(s_q.clone());
            let x_id = tape.leaf(s.x_train.clone());
            let tg_id = tape.leaf(s.target_train.clone());
            let t_id = tape.ste_surrogate(wf_id, sq_id, n, k);
            let y_id = tape.matmul(x_id, t_id, sq_id, m, n, k);
            let loss_id = tape.mse(y_id, tg_id);
            let loss = tape.value(loss_id)[0];
            let grads = tape.backward(loss_id);
            opt.step(step, &mut wf, &grads[wf_id], &mut state);
            assert!(
                loss.is_finite() && wf.iter().all(|v| v.is_finite()),
                "non-finite at {step}"
            );
            if step == 1 {
                first = loss;
            }
            last = loss;
        }
        total_first += first as f64;
        total_last += last as f64;
        // Per-projection convergence: every projection must individually reduce its
        // distillation loss, so a single stagnating projection can't be masked by the
        // magnitude-weighted aggregate.
        assert!(
            last < first && first > 0.0,
            "{}:{} distillation loss did not decrease ({first:.4} -> {last:.4})",
            s.layer,
            s.which
        );
        min_proj_recovery = min_proj_recovery.min(((first - last) / first) as f64);
        // Re-quantize the healed latent at the same per-tensor granularity and install.
        let s_q = per_tensor_scale(&wf, n);
        let healed = ste::quantize_forward(&wf, &s_q, n, k);
        install(&mut runner, s.layer, s.which, &to_trits(&healed), s_q);
    }
    let ppl_healed = perplexity(&mut runner, &eval_ids);
    let ppl_recovery = (ppl_degraded - ppl_healed) / gap;

    // THE GATE — what is actually certified: the QAT machinery (AdamW + STE + tape) drives
    // a real BitNet layer's ternary distillation loss down by ≥90% on real model
    // activations, end-to-end through the heal bridge. This is *distillation-loss
    // convergence*, NOT a full-model perplexity recovery: BitNet's latent master is already
    // per-tensor-QAT-optimal and the eval slice is too short to constrain 2560-wide layers,
    // so full-model PPL recovery via layerwise distillation is underdetermined (plan 0010).
    // The 1-bit degradation + PPL numbers below are reported as *context* (they show the
    // model's under-bitting sensitivity and that the bridge swaps weights + scores PPL
    // correctly), not as the gated metric.
    assert!(
        total_first > 0.0,
        "aggregate step-1 distillation loss is zero — the gate divisor is undefined"
    );
    let distill_convergence = (total_first - total_last) / total_first;
    eprintln!(
        "GATE: QAT distillation-loss convergence = {:.1}% aggregate (min per-projection {:.1}%); loss {total_first:.4} -> {total_last:.4}",
        distill_convergence * 100.0,
        min_proj_recovery * 100.0
    );
    eprintln!(
        "context (not gated): PPL good={ppl_good:.4} 1-bit-degraded={ppl_degraded:.4} (real under-bitting gap); \
         post-heal full-model PPL={ppl_healed:.4} / recovery {:.1}% — underdetermined from a {seq}-row eval slice on {}-wide layers (a v0.60 full-backprop item)",
        ppl_recovery * 100.0,
        e
    );

    assert!(
        distill_convergence >= 0.90,
        "QAT distillation-loss convergence {:.1}% < 90% (aggregate loss {total_first:.4} -> {total_last:.4})",
        distill_convergence * 100.0
    );
}
