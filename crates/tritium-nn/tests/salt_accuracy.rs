//! SALT accuracy-vs-bpw curve (ADR 0006, v0.4.0 P4) — gated, run explicitly.
//!
//! Quantizes the **bf16 BitNet master** (`microsoft/bitnet-b1.58-2B-4T-bf16`) to
//! SALT at a sweep of bits-per-weight budgets, dequantizes each projection to dense
//! fp32, and runs **teacher-forced perplexity** through the host forward
//! ([`DenseLinear`] projections on the int8 A8 activation path — the same activation
//! quantization the deployed ternary model uses, so only the *weight* quantization
//! varies). Reports the perplexity-vs-bpw curve.
//!
//! Reference points: the `fp` row is the upper bound (continuous weights), and
//! SALT at the budget floor (`≈1.585` bpw, all `T=1`) is flat-AbsMean BitNet — the
//! deployed ternary checkpoint scores **1.4028** perplexity on this eval set.
//!
//! Expensive (loads a 4.5 GB model, quantizes ~2.4B params per budget, runs a CPU
//! forward), so it is `#[ignore]`d and skips cleanly when the model is absent. Run:
//!
//! ```text
//! cargo test -p tritium-nn --release --test salt_accuracy -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use tritium_format::{SafeTensors, dequant_salt_row};
use tritium_nn::{
    DenseLinear, ForwardDump, Mlp, ModelConfig, ModelRunner, ModelWeights, Projection, Relu2Mlp,
    TransformerBlock,
};
use tritium_quantize::{BaseScaleScope, QuantConfig, Sensitivity, quantize_tensor};

/// Eval tokens to score. The reference set is 262; the CPU host forward over a
/// 2.4B-param fp model is memory-bandwidth-bound (~11 GB streamed/token), so a
/// short prefix keeps the directional curve tractable. The high-resolution curve
/// is properly a GPU job (the deferred resident-SALT decode path).
const EVAL_LEN: usize = 24;

/// Weight quantization for one model build.
#[derive(Clone, Copy, Debug)]
enum Mode {
    /// Continuous bf16 → f32 (the upper bound).
    Fp,
    /// SALT at a target average bits-per-weight.
    Salt { bpw: f64 },
}

fn bf16_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".cache/tritium-models/bitnet-2b4t-bf16/model.safetensors")
}

fn reference_path() -> &'static str {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tools/reference/bitnet_accept.json"
    )
}

/// The BitNet-2B4T geometry (from the bf16 `config.json`).
fn config() -> ModelConfig {
    ModelConfig {
        arch: "bitnet".to_owned(),
        n_layers: 30,
        n_embd: 2560,
        n_head: 20,
        n_head_kv: 5,
        head_dim: 128,
        n_ff: 6912,
        n_ctx: 4096,
        rope_theta: 500_000.0,
        rms_eps: 1e-5,
    }
}

/// Build one projection: read the bf16 weight, optionally SALT-quantize-then-dequant,
/// and wrap as a dense fp32 projection.
fn proj(st: &SafeTensors, name: &str, n_out: usize, k_in: usize, mode: Mode) -> Projection {
    let w = st
        .tensor_f32(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(w.len(), n_out * k_in, "{name} shape");
    let dense = match mode {
        Mode::Fp => w,
        Mode::Salt { bpw } => {
            let cfg = QuantConfig {
                budget_bpw: bpw,
                t_min: 1,
                t_max: 3,
                sensitivity: Sensitivity::Uniform,
                // BitNet b1.58 is QAT-trained against a single per-tensor absmean ternary, so
                // the SALT base plane must match that granularity (per-256-block reconstructs
                // the latent master too faithfully → garbage). T=1 ⇒ the deployed I2_S.
                scale_group: BaseScaleScope::Tensor,
            };
            let qt = quantize_tensor(&w, n_out, k_in, &cfg).expect("quantize_tensor");
            let mut dq = vec![0.0f32; n_out * k_in];
            for (r, row) in qt.salt_rows.iter().enumerate() {
                let wr = dequant_salt_row(row).expect("dequant_salt_row");
                dq[r * k_in..r * k_in + k_in].copy_from_slice(&wr);
            }
            dq
        }
    };
    Projection::Dense(DenseLinear::new(dense, n_out, k_in).expect("DenseLinear"))
}

/// Assemble `ModelWeights` from the bf16 safetensors under the given quantization.
fn build_weights(st: &SafeTensors, cfg: &ModelConfig, mode: Mode) -> ModelWeights {
    let n_embd = cfg.n_embd as usize;
    let head_dim = cfg.head_dim() as usize;
    let q_width = cfg.n_head as usize * head_dim;
    let kv_width = cfg.n_head_kv as usize * head_dim;
    let n_ff = cfg.n_ff as usize;

    let token_embd = st
        .tensor_f32("model.embed_tokens.weight")
        .expect("embed_tokens");
    let vocab = st.shape("model.embed_tokens.weight").expect("embed shape")[0];
    let output_norm = st.tensor_f32("model.norm.weight").expect("norm");

    let layers = (0..cfg.n_layers as usize)
        .map(|li| {
            let nm = |s: &str| format!("model.layers.{li}.{s}");
            TransformerBlock {
                attn_norm: st.tensor_f32(&nm("input_layernorm.weight")).unwrap(),
                q_proj: proj(st, &nm("self_attn.q_proj.weight"), q_width, n_embd, mode),
                k_proj: proj(st, &nm("self_attn.k_proj.weight"), kv_width, n_embd, mode),
                v_proj: proj(st, &nm("self_attn.v_proj.weight"), kv_width, n_embd, mode),
                o_proj: proj(st, &nm("self_attn.o_proj.weight"), n_embd, q_width, mode),
                attn_sub_norm: st
                    .tensor_f32(&nm("self_attn.attn_sub_norm.weight"))
                    .unwrap(),
                q_bias: Vec::new(),
                k_bias: Vec::new(),
                v_bias: Vec::new(),
                q_norm: Vec::new(),
                k_norm: Vec::new(),
                ffn_norm: st
                    .tensor_f32(&nm("post_attention_layernorm.weight"))
                    .unwrap(),
                mlp: Mlp::Relu2(Relu2Mlp {
                    gate: proj(st, &nm("mlp.gate_proj.weight"), n_ff, n_embd, mode),
                    up: proj(st, &nm("mlp.up_proj.weight"), n_ff, n_embd, mode),
                    down: proj(st, &nm("mlp.down_proj.weight"), n_embd, n_ff, mode),
                    ffn_sub_norm: st.tensor_f32(&nm("mlp.ffn_sub_norm.weight")).unwrap(),
                    rms_eps: cfg.rms_eps,
                }),
            }
        })
        .collect();

    ModelWeights {
        token_embd,
        vocab,
        n_embd,
        layers,
        output_norm,
        lm_head: None,
    }
}

/// Log-softmax of `logits` at `target` (numerically stable).
fn log_prob(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = m + logits
        .iter()
        .map(|&x| (x as f64 - m).exp())
        .sum::<f64>()
        .ln();
    logits[target] as f64 - lse
}

/// Teacher-forced perplexity over `eval_ids`: prefill token 0, then at each position
/// read the next true token's log-prob and step forward with it.
fn perplexity(runner: &mut ModelRunner, eval_ids: &[u32]) -> f64 {
    runner.reset();
    let n = eval_ids.len();
    let mut nll = 0.0f64;
    let mut logits = runner.forward(&eval_ids[..1], &[0]).expect("prefill");
    for t in 0..n - 1 {
        let target = eval_ids[t + 1] as usize;
        nll += -log_prob(&logits, target);
        logits = runner
            .forward(&[eval_ids[t + 1]], &[t + 1])
            .expect("decode");
    }
    (nll / (n - 1) as f64).exp()
}

#[test]
#[ignore = "loads the 4.5GB bf16 master + runs a CPU forward per budget; run explicitly"]
fn salt_accuracy_curve() {
    let path = bf16_path();
    if !path.exists() {
        eprintln!("skipping: {} absent", path.display());
        return;
    }
    let ref_raw = match std::fs::read(reference_path()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: reference json: {e}");
            return;
        }
    };
    let ref_json: serde_json::Value = serde_json::from_slice(&ref_raw).expect("parse reference");
    let all_ids: Vec<u32> = ref_json["eval_ids"]
        .as_array()
        .expect("eval_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let eval_ids = &all_ids[..EVAL_LEN.min(all_ids.len())];

    let bytes = std::fs::read(&path).expect("read bf16");
    let st = SafeTensors::parse(&bytes).expect("parse safetensors");
    let cfg = config();

    let modes = [
        Mode::Fp,
        Mode::Salt { bpw: 1.585 },
        Mode::Salt { bpw: 2.0 },
        Mode::Salt { bpw: 2.6 },
        Mode::Salt { bpw: 3.0 },
    ];

    println!(
        "\nSALT accuracy-vs-bpw curve ({} eval tokens):",
        eval_ids.len()
    );
    println!("  reference: deployed ternary I2_S = 1.4028 (full 262-tok set); see");
    println!("  `gguf_eval_perplexity` for the deployed score on THIS prefix.\n");
    println!("  {:<16} {:>10}", "mode", "perplexity");

    let mut fp_ppl = f64::NAN;
    let mut salt_floor = f64::NAN;
    for mode in modes {
        let weights = build_weights(&st, &cfg, mode);
        let backend = Box::new(tritium_cpu::CpuBackend::new());
        let mut runner = ModelRunner::from_weights(cfg.clone(), weights, backend);
        let ppl = perplexity(&mut runner, eval_ids);
        let label = match mode {
            Mode::Fp => "fp (bf16)".to_owned(),
            Mode::Salt { bpw } => format!("salt {bpw:.3}bpw"),
        };
        println!("  {label:<16} {ppl:>10.4}");
        match mode {
            Mode::Fp => fp_ppl = ppl,
            Mode::Salt { bpw } if (bpw - tritium_quantize::TRIT_BITS).abs() < 1e-3 => {
                salt_floor = ppl
            }
            Mode::Salt { .. } => {}
        }
        // `runner` (and its ~11 GB of weights) drops here before the next build.
    }

    // For a QAT-ternary master (BitNet b1.58) the curve INVERTS the usual shape: the bf16
    // `fp` row is the *latent* weight, not a usable forward, so it scores garbage; the
    // per-tensor SALT floor (`budget = log2 3`, `BaseScaleScope::Tensor`) is the deployed I2_S
    // ternary and the curve's optimum, and residual planes (higher bpw) regress back toward
    // the unusable master. So the gate is: the floor is a *working* model that crushes the
    // raw-master fp. (Per-tensor base reproduces the GGUF weights to f16; `gguf_eval_perplexity`
    // is the deployed score on this same prefix.)
    assert!(fp_ppl.is_finite(), "fp perplexity {fp_ppl} not finite");
    assert!(
        salt_floor.is_finite() && salt_floor < 50.0,
        "per-tensor SALT floor {salt_floor} — expected a working model (deployed I2_S ≈ 5 on this prefix), not the ~8000 a per-block base produces"
    );
    assert!(
        salt_floor * 100.0 < fp_ppl,
        "per-tensor SALT floor {salt_floor} must crush the raw latent master fp {fp_ppl}"
    );
}

const GGUF_PATH: &str =
    "/home/brianklam/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf";

/// Reference: the deployed GGUF I2_S perplexity on the SAME `EVAL_LEN` tokens the curve
/// uses (the committed 1.4028 is over the full 262-token set; a short prefix scores
/// higher). `salt@1.585` with [`BaseScaleScope::Tensor`] must match this — that's the gate
/// that the per-tensor SALT base reproduces the deployed ternary. Cheap (one model, one
/// forward), unlike the full curve.
#[test]
#[ignore = "loads the GGUF + one CPU forward; run explicitly"]
fn gguf_eval_perplexity() {
    if !std::path::Path::new(GGUF_PATH).exists() {
        eprintln!("skipping: {GGUF_PATH} absent");
        return;
    }
    let ref_raw = match std::fs::read(reference_path()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: reference json: {e}");
            return;
        }
    };
    let ref_json: serde_json::Value = serde_json::from_slice(&ref_raw).expect("parse reference");
    let all_ids: Vec<u32> = ref_json["eval_ids"]
        .as_array()
        .expect("eval_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let eval_ids = &all_ids[..EVAL_LEN.min(all_ids.len())];

    let gbytes = std::fs::read(GGUF_PATH).expect("read gguf");
    let mut gg = ModelRunner::load_cpu(&gbytes).expect("load gguf cpu");
    let ppl = perplexity(&mut gg, eval_ids);
    println!(
        "\nGGUF I2_S perplexity on {} eval tokens = {ppl:.4}",
        eval_ids.len()
    );
}

/// L2 norm + max-abs of a stage tensor.
fn stats(v: &[f32]) -> (f64, f32) {
    let l2 = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    let maxabs = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    (l2, maxabs)
}

/// DIAGNOSTIC (curve debug, a04dfa7): the fp curve produces garbage perplexity. Run the
/// HF-fp model and the known-good GGUF-ternary model through the **same** CPU host forward
/// on one token, dumping each stage's L2/max so the first divergence localizes the bug.
/// They use different weights (fp vs ternary) so small per-stage drift is expected; an
/// order-of-magnitude gap (or a NaN/explosion) marks the broken stage.
#[test]
#[ignore = "diagnostic: HF-fp vs GGUF-ternary per-stage dump on one token"]
fn salt_fp_vs_gguf_stage_dump() {
    let path = bf16_path();
    if !path.exists() {
        eprintln!("skipping: {} absent", path.display());
        return;
    }
    if !std::path::Path::new(GGUF_PATH).exists() {
        eprintln!("skipping: {GGUF_PATH} absent");
        return;
    }
    let cfg = config();
    let tok = 128000u32; // BOS; probed at position 0

    let bytes = std::fs::read(&path).expect("read bf16");
    let st = SafeTensors::parse(&bytes).expect("parse safetensors");
    let weights = build_weights(&st, &cfg, Mode::Fp);
    let mut hf = ModelRunner::from_weights(
        cfg.clone(),
        weights,
        Box::new(tritium_cpu::CpuBackend::new()),
    );

    let gbytes = std::fs::read(GGUF_PATH).expect("read gguf");
    let mut gg = ModelRunner::load_cpu(&gbytes).expect("load gguf cpu");

    let mut dh = ForwardDump::default();
    hf.reset();
    hf.forward_dump(&[tok], &[0], &mut dh).expect("hf dump");
    let mut dg = ForwardDump::default();
    gg.reset();
    gg.forward_dump(&[tok], &[0], &mut dg).expect("gguf dump");

    let pr = |name: &str, a: &[f32], b: &[f32]| {
        let (la, ma) = stats(a);
        let (lb, mb) = stats(b);
        println!("  {name:<20} hf[l2={la:>11.3} max={ma:>9.3}]  gguf[l2={lb:>11.3} max={mb:>9.3}]");
    };
    println!("\nstage dump (token {tok}, pos 0) — HF-fp vs GGUF-ternary:");
    pr("embedding", &dh.embedding, &dg.embedding);
    pr(
        "layer0_attn_norm",
        &dh.layer0_attn_norm,
        &dg.layer0_attn_norm,
    );
    pr("layer0_attn_out", &dh.layer0_attn_out, &dg.layer0_attn_out);
    let nl = dh.hidden_states.len();
    for (i, (a, b)) in dh.hidden_states.iter().zip(&dg.hidden_states).enumerate() {
        if i < 3 || i + 1 == nl {
            pr(&format!("hidden[{i}]"), a, b);
        }
    }
    pr("final_norm", &dh.final_norm, &dg.final_norm);
    let amax = |v: &[f32]| {
        v.iter().enumerate().fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) },
        )
    };
    println!(
        "  hf logits argmax={:?}   gguf logits argmax={:?}",
        amax(&dh.logits),
        amax(&dg.logits)
    );

    // --- decompose layer-0 attention (the first divergence) --- //
    // At pos 0 attention is trivial (attn_out == v expanded), so the q_width-3× gap is in
    // v_proj / attn_sub_norm / o_proj. Compare the norm WEIGHTS (input-independent) and the
    // v_proj output on the shared (matching) normed input.
    let hb = &hf.weights.layers[0];
    let gb = &gg.weights.layers[0];
    println!("\nlayer-0 norm weights (hf vs gguf):");
    pr("attn_norm.w", &hb.attn_norm, &gb.attn_norm);
    pr("attn_sub_norm.w", &hb.attn_sub_norm, &gb.attn_sub_norm);
    pr("ffn_norm.w", &hb.ffn_norm, &gb.ffn_norm);
    pr(
        "ffn_sub_norm.w",
        &hb.mlp.as_relu2().unwrap().ffn_sub_norm,
        &gb.mlp.as_relu2().unwrap().ffn_sub_norm,
    );
    pr(
        "output_norm.w",
        &hf.weights.output_norm,
        &gg.weights.output_norm,
    );

    // v_proj on the shared matching input.
    let backend = tritium_cpu::CpuBackend::new();
    let n_embd = cfg.n_embd as usize;
    let kv_width = cfg.n_head_kv as usize * cfg.head_dim() as usize;
    let inp = &dg.layer0_attn_norm[..n_embd]; // identical for both (matched above)
    let mut vh = vec![0.0f32; kv_width];
    let mut vg = vec![0.0f32; kv_width];
    hb.v_proj.forward(&backend, inp, 1, &mut vh).expect("hf v");
    gb.v_proj.forward(&backend, inp, 1, &mut vg).expect("gg v");
    pr("v_proj(shared inp)", &vh, &vg);

    // Compare HF master absmean (the implied weight_scale) vs the GGUF I2_S per-tensor
    // scale, per projection. If the ratio is ~constant across tensors → a global scale
    // convention bug; if it tracks absmean → the fp path is just missing the ×scale.
    let absmean = |w: &[f32]| w.iter().map(|x| f64::from(x.abs())).sum::<f64>() / w.len() as f64;
    let hf_w = |p: &Projection| match p {
        Projection::Dense(d) => d.weights.clone(),
        Projection::Salt(_) | Projection::Ternary(_) => unreachable!("hf is dense"),
    };
    let gg_scale = |p: &Projection| f64::from(p.as_ternary().expect("gguf ternary").scales[0]);
    println!("\nlayer-0 weight scales (hf absmean vs gguf I2_S scale):");
    for (name, hp, gp) in [
        ("q_proj", &hb.q_proj, &gb.q_proj),
        ("k_proj", &hb.k_proj, &gb.k_proj),
        ("v_proj", &hb.v_proj, &gb.v_proj),
        ("o_proj", &hb.o_proj, &gb.o_proj),
        (
            "gate",
            &hb.mlp.as_relu2().unwrap().gate,
            &gb.mlp.as_relu2().unwrap().gate,
        ),
        (
            "down",
            &hb.mlp.as_relu2().unwrap().down,
            &gb.mlp.as_relu2().unwrap().down,
        ),
    ] {
        let am = absmean(&hf_w(hp));
        let sc = gg_scale(gp);
        println!(
            "  {name:<8} hf_absmean={am:>12.6}  gguf_scale={sc:>12.6}  ratio={:>8.3}",
            am / sc
        );
    }

    // Decisive: does SALT@1.585 (all-T=1 == flat absmean == BitNet ternary) reproduce the
    // GGUF v_proj? If yes, the forward is fine and only raw-master "fp" is ill-defined; if
    // it also blows up, SALT T=1 != the BitNet ternary the GGUF uses.
    let salt_v = proj(
        &st,
        "model.layers.0.self_attn.v_proj.weight",
        kv_width,
        n_embd,
        Mode::Salt { bpw: 1.585 },
    );
    let mut vs = vec![0.0f32; kv_width];
    salt_v.forward(&backend, inp, 1, &mut vs).expect("salt v");
    pr("v_proj salt@1.585", &vs, &vg); // vg is the GGUF v_proj output (reference)

    // Confirm the fix direction: a PER-TENSOR absmean ternary of the raw master (one scale
    // for the whole tensor, round-clamp to {-1,0,1}) should reproduce the GGUF I2_S output.
    let w_v = hf_w(&hb.v_proj);
    let am_v = absmean(&w_v) as f32;
    let w_pt: Vec<f32> = w_v
        .iter()
        .map(|&x| am_v * (x / am_v).round().clamp(-1.0, 1.0))
        .collect();
    let pt = Projection::Dense(DenseLinear::new(w_pt, kv_width, n_embd).unwrap());
    let mut vp = vec![0.0f32; kv_width];
    pt.forward(&backend, inp, 1, &mut vp).expect("per-tensor v");
    pr("v_proj per-tensor", &vp, &vg); // should match gguf (~105)
}
