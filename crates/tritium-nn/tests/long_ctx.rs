//! Long-context decode bench (model + GPU gated, run explicitly):
//!
//! ```text
//! cargo test -p tritium-nn --features cuda --release --test long_ctx -- --ignored --nocapture
//! TRITIUM_KV_F16=1 cargo test -p tritium-nn --features cuda --release --test long_ctx -- --ignored --nocapture
//! ```
//!
//! Prefills near the context limit and times decode there — the regime where
//! attention is KV-bandwidth-bound and the ADR 0020 f16 rung should pay
//! (short-context decode is latency-bound and gains nothing).

#![cfg(feature = "cuda")]

use std::path::Path;

const GGUF_PATH: &str =
    "/home/brianklam/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf";
const REF_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/reference/bitnet_accept.json"
);

#[test]
#[ignore = "long-context bench: run explicitly with --ignored --nocapture"]
fn cuda_long_ctx_decode_bench() {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("skipping: {GGUF_PATH} absent (gated real-model bench)");
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse reference");
    let base: Vec<u32> = reference["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let bytes = std::fs::read(GGUF_PATH).expect("read gguf");
    let file = tritium_format::read_gguf(&bytes).expect("parse gguf");
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")
        .expect("cuda backend")
        .init;
    let backend = match init() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: cuda backend failed to init ({e})");
            return;
        }
    };
    let mut runner = tritium_nn::ModelRunner::load(&file, &bytes, backend).expect("load model");

    // Fill most of the context (leave room for the decode tail).
    let n_ctx = runner.config.n_ctx as usize;
    let decode_steps = 64usize;
    let target = n_ctx - decode_steps - 8;
    let prompt: Vec<u32> = base.iter().cycle().take(target).copied().collect();
    let positions: Vec<usize> = (0..prompt.len()).collect();

    let t0 = std::time::Instant::now();
    let logits = runner.forward(&prompt, &positions).expect("prefill");
    let t_prefill = t0.elapsed();

    // Warm the decode graph, then time the long-context tail.
    let mut next = tritium_nn::sample_greedy(&logits).expect("token");
    let mut pos = prompt.len();
    let logits = runner.forward(&[next], &[pos]).expect("warm step");
    next = tritium_nn::sample_greedy(&logits).expect("token");
    pos += 1;

    let t0 = std::time::Instant::now();
    for _ in 0..decode_steps {
        let logits = runner.forward(&[next], &[pos]).expect("decode");
        next = tritium_nn::sample_greedy(&logits).expect("token");
        pos += 1;
    }
    let dt = t0.elapsed();
    let kv = std::env::var("TRITIUM_KV_F16").unwrap_or_else(|_| "0".into());
    println!(
        "long-ctx (KV_F16={kv}): prefill {} tok in {t_prefill:.2?} ({:.0} tok/s) | decode @ctx≈{} {decode_steps} tok in {dt:.2?} ({:.1} tok/s)",
        prompt.len(),
        prompt.len() as f64 / t_prefill.as_secs_f64(),
        prompt.len(),
        decode_steps as f64 / dt.as_secs_f64(),
    );
}
