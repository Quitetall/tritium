//! Conformance gates for the general inference engine (plan 0035 / 0037 / ADR 0020 keystone):
//! standard-transformer fp models loaded via [`ModelRunner::from_hf`] run a CPU forward that is
//! **greedy-token-exact** vs a `transformers` reference, with a last-position logit
//! **rel-err < 1e-3**.
//!
//! - SmolLM2-135M — Llama-arch (SwiGLU/GQA/RoPE/RMSNorm, tied). Plan 0035.
//! - Qwen2.5-0.5B — adds **QKV bias**. Plan 0037.
//! - Qwen3-0.6B — adds **QK-norm** + an explicit **head_dim** (≠ n_embd/n_head). Plan 0037.
//!
//! `#[ignore]`d (need the models downloaded); skip cleanly when absent. Regenerate a reference
//! with `python3 tools/gen_hf_logits.py <model_dir> <out.json>`. Run:
//!
//! ```text
//! cargo test -p tritium-nn --release --test hf_inference -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use tritium_nn::ModelRunner;

fn cache(subdir: &str) -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    PathBuf::from(home)
        .join(".cache/tritium-models")
        .join(subdir)
}

fn reference(file: &str) -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tools/reference"
    ))
    .join(file)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

/// Load `<cache>/<model_subdir>` via `from_hf`, teacher-force the reference prompt, and assert
/// greedy token-exactness + last-row logit rel-err < 1e-3. Skips (passing) if the model or its
/// reference is absent.
fn assert_conforms(label: &str, model_subdir: &str, ref_file: &str) {
    let dir = cache(model_subdir);
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping {label}: {} absent", dir.display());
        return;
    }
    let ref_raw = match std::fs::read(reference(ref_file)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping {label}: reference: {e}");
            return;
        }
    };
    let rj: serde_json::Value = serde_json::from_slice(&ref_raw).expect("parse reference");
    let ids: Vec<u32> = rj["prompt_ids"]
        .as_array()
        .expect("prompt_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let ref_argmax: Vec<usize> = rj["next_argmax_per_pos"]
        .as_array()
        .expect("argmax")
        .iter()
        .map(|v| v.as_u64().expect("argmax id") as usize)
        .collect();
    let ref_last: Vec<f32> = rj["logits_last_row"]
        .as_array()
        .expect("logits")
        .iter()
        .map(|v| v.as_f64().expect("logit") as f32)
        .collect();

    let backend = Box::new(tritium_cpu::CpuBackend::new());
    let mut runner = ModelRunner::from_hf(&dir, backend).expect("from_hf");
    runner.reset();

    // Teacher-force: prefill token 0, then step each true token; `got[t]` predicts token `t+1`,
    // matching HF `logits[t]`.
    let n = ids.len();
    let mut got = Vec::with_capacity(n);
    let mut logits = runner.forward(&ids[..1], &[0]).expect("prefill");
    got.push(argmax(&logits));
    for (t, &tok) in ids.iter().enumerate().skip(1) {
        logits = runner.forward(&[tok], &[t]).expect("decode");
        got.push(argmax(&logits));
    }

    let matched = got
        .iter()
        .zip(&ref_argmax)
        .take_while(|(a, b)| a == b)
        .count();
    let num: f64 = ref_last
        .iter()
        .zip(&logits)
        .map(|(&r, &g)| {
            let d = f64::from(r) - f64::from(g);
            d * d
        })
        .sum();
    let den: f64 = ref_last.iter().map(|&r| f64::from(r) * f64::from(r)).sum();
    let rel = (num / den).sqrt();
    println!("{label} conformance: greedy match {matched}/{n}, last-row logit rel-err {rel:.3e}");

    assert_eq!(
        got, ref_argmax,
        "{label}: greedy tokens must match transformers"
    );
    assert!(
        rel < 1e-3,
        "{label}: last-row logit rel-err {rel:.3e} exceeds 1e-3"
    );
}

#[test]
#[ignore = "needs SmolLM2-135M under ~/.cache/tritium-models/smollm2-135m; run explicitly"]
fn smollm2_greedy_matches_transformers() {
    assert_conforms("SmolLM2-135M", "smollm2-135m", "smollm2_ref.json");
}

#[test]
#[ignore = "needs Qwen2.5-0.5B under ~/.cache/tritium-models/qwen2.5-0.5b; run explicitly"]
fn qwen25_greedy_matches_transformers() {
    assert_conforms(
        "Qwen2.5-0.5B (QKV-bias)",
        "qwen2.5-0.5b",
        "qwen25_0.5b_ref.json",
    );
}

#[test]
#[ignore = "needs Qwen3-0.6B under ~/.cache/tritium-models/qwen3-0.6b; run explicitly"]
fn qwen3_greedy_matches_transformers() {
    assert_conforms("Qwen3-0.6B (QK-norm)", "qwen3-0.6b", "qwen3_0.6b_ref.json");
}
