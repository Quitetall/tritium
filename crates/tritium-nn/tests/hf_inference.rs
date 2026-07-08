//! Conformance gate for the general inference engine (plan 0035 / ADR 0020 keystone):
//! a **standard-transformer fp** model (SmolLM2-135M, Llama-arch: SwiGLU/GQA/RoPE/RMSNorm,
//! tied) loaded via [`ModelRunner::from_hf`] runs a CPU forward that is **greedy-token-exact**
//! vs a `transformers` reference, with a last-position logit **rel-err < 1e-3**.
//!
//! `#[ignore]`d (needs the model downloaded + is a real forward); skips cleanly when absent.
//! Regenerate the reference with `tools/gen_hf_logits.py`. Run:
//!
//! ```text
//! cargo test -p tritium-nn --release --test hf_inference -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use tritium_nn::ModelRunner;

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

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

#[test]
#[ignore = "needs SmolLM2-135M under ~/.cache/tritium-models/smollm2-135m; run explicitly"]
fn smollm2_greedy_matches_transformers() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: {} absent", dir.display());
        return;
    }
    let ref_raw = match std::fs::read(reference()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: reference: {e}");
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

    // Teacher-force the prompt: prefill token 0, then step each true token; collect the
    // greedy argmax at every position (`got[t]` predicts token `t+1`, matching HF `out[t]`).
    let n = ids.len();
    let mut got = Vec::with_capacity(n);
    let mut logits = runner.forward(&ids[..1], &[0]).expect("prefill");
    got.push(argmax(&logits));
    for (t, &tok) in ids.iter().enumerate().skip(1) {
        logits = runner.forward(&[tok], &[t]).expect("decode");
        got.push(argmax(&logits));
    }
    // `logits` is now the last-position row.

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
    println!(
        "SmolLM2-135M conformance: greedy match {matched}/{n}, last-row logit rel-err {rel:.3e}"
    );

    assert_eq!(
        got, ref_argmax,
        "greedy tokens must match transformers exactly"
    );
    assert!(rel < 1e-3, "last-row logit rel-err {rel:.3e} exceeds 1e-3");
}
