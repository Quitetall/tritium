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
    DenseLinear, ModelConfig, ModelRunner, ModelWeights, Projection, Relu2Mlp, TransformerBlock,
};
use tritium_quantize::{QuantConfig, Sensitivity, quantize_tensor};

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
        n_ff: 6912,
        n_ctx: 4096,
        rope_theta: 500_000.0,
        rms_eps: 1e-5,
    }
}

/// Build one projection: read the bf16 weight, optionally SALT-quantize-then-dequant,
/// and wrap as a dense fp32 projection.
fn proj(st: &SafeTensors, name: &str, n_out: usize, k_in: usize, mode: Mode) -> Projection {
    let w = st.tensor_f32(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(w.len(), n_out * k_in, "{name} shape");
    let dense = match mode {
        Mode::Fp => w,
        Mode::Salt { bpw } => {
            let cfg = QuantConfig {
                budget_bpw: bpw,
                t_min: 1,
                t_max: 3,
                sensitivity: Sensitivity::Uniform,
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

    let token_embd = st.tensor_f32("model.embed_tokens.weight").expect("embed_tokens");
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
                attn_sub_norm: st.tensor_f32(&nm("self_attn.attn_sub_norm.weight")).unwrap(),
                ffn_norm: st.tensor_f32(&nm("post_attention_layernorm.weight")).unwrap(),
                mlp: Relu2Mlp {
                    gate: proj(st, &nm("mlp.gate_proj.weight"), n_ff, n_embd, mode),
                    up: proj(st, &nm("mlp.up_proj.weight"), n_ff, n_embd, mode),
                    down: proj(st, &nm("mlp.down_proj.weight"), n_embd, n_ff, mode),
                    ffn_sub_norm: st.tensor_f32(&nm("mlp.ffn_sub_norm.weight")).unwrap(),
                    rms_eps: cfg.rms_eps,
                },
            }
        })
        .collect();

    ModelWeights {
        token_embd,
        vocab,
        n_embd,
        layers,
        output_norm,
    }
}

/// Log-softmax of `logits` at `target` (numerically stable).
fn log_prob(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = m + logits.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln();
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
        logits = runner.forward(&[eval_ids[t + 1]], &[t + 1]).expect("decode");
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

    println!("\nSALT accuracy-vs-bpw curve ({} eval tokens):", eval_ids.len());
    println!("  reference: deployed ternary I2_S = 1.4028 perplexity\n");
    println!("  {:<16} {:>10}", "mode", "perplexity");

    let mut fp_ppl = f64::NAN;
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
        if let Mode::Fp = mode {
            fp_ppl = ppl;
        }
        // `runner` (and its ~11 GB of weights) drops here before the next build.
    }

    // Sanity: the fp upper bound is a finite, reasonable perplexity.
    assert!(fp_ppl.is_finite() && fp_ppl > 1.0 && fp_ppl < 100.0, "fp perplexity {fp_ppl}");
}
