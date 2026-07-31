//! Prompt-lookup speculative decoding gates (model + GPU gated, `cuda` feature).
//!
//! Losslessness: the spec-lookup stream must equal the plain greedy stream
//! token-for-token (every emitted token is the target's own argmax — the
//! BASTION verifier only ever commits those). Also prints both wall times; the
//! committed reference continuation is highly repetitive, so the lookup
//! drafter should land multi-token commits and beat plain decode.

#![cfg(feature = "cuda")]

use std::path::Path;

use tritium_serve::{GenRequest, Generator, RunnerGenerator, Sampling};

use tritium_cpu as _;
use tritium_cuda as _;

/// Model cache root: override via `TRITIUM_MODEL_DIR`; default `~/.cache/tritium-models`; tests skip cleanly when absent.
static GGUF_PATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    let dir = std::env::var("TRITIUM_MODEL_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/tritium-models",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    format!("{dir}/bitnet-2b4t-gguf/ggml-model-i2_s.gguf")
});
const REF_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/reference/bitnet_accept.json"
);

fn load_runner(bytes: &[u8]) -> Option<tritium_nn::ModelRunner> {
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == "cuda")
        .map(|e| e.init)?;
    let backend = match init() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: cuda backend failed to init ({e})");
            return None;
        }
    };
    let file = tritium_format::read_gguf(bytes).expect("parse gguf");
    Some(tritium_nn::ModelRunner::load(&file, bytes, backend).expect("load model"))
}

fn collect(generator: &mut dyn Generator, req: &GenRequest) -> (Vec<u32>, std::time::Duration) {
    let mut out = Vec::new();
    let t0 = std::time::Instant::now();
    generator
        .generate(req, &mut |step| {
            out.push(step.token);
            true
        })
        .expect("generate");
    (out, t0.elapsed())
}

/// temp→0 gate for the SAMPLING accept rule: TopK{k:1} makes p̃ collapse to
/// the argmax candidate at probability 1, so the whole sampled machinery
/// (tree_verify_logits → host accept walk → tree_commit) becomes
/// deterministic and must reproduce the plain greedy stream token-for-token.
#[test]
fn cuda_spec_sampled_topk1_matches_plain_greedy() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse reference");
    let prompt: Vec<u32> = reference["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");

    let greedy_req = GenRequest {
        prompt_tokens: prompt.clone(),
        max_new: 128,
        sampling: Sampling::Greedy,
        stop_eos: false,
        logprobs: None,
    };
    let sampled_req = GenRequest {
        prompt_tokens: prompt,
        max_new: 128,
        sampling: Sampling::TopK {
            k: 1,
            temp: 1.0,
            seed: 42,
        },
        stop_eos: false,
        logprobs: None,
    };

    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut plain = RunnerGenerator::new(runner, u32::MAX);
    let (want, _) = collect(&mut plain, &greedy_req);

    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut spec = RunnerGenerator::new(runner, u32::MAX).with_spec_lookup(true);
    let (got, t) = collect(&mut spec, &sampled_req);
    println!("spec-sampled k=1: {} tok in {t:.2?}", got.len());
    assert_eq!(
        got, want,
        "spec-sampled TopK{{k:1}} must equal plain greedy"
    );

    // Same seed twice → identical stream (the spec path is deterministic).
    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut spec2 = RunnerGenerator::new(runner, u32::MAX).with_spec_lookup(true);
    let (got2, _) = collect(&mut spec2, &sampled_req);
    assert_eq!(got2, got, "same-seed spec-sampled runs must be identical");
}

#[test]
fn cuda_spec_lookup_matches_plain_greedy() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse reference");
    let prompt: Vec<u32> = reference["token_ids"]
        .as_array()
        .expect("token_ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");

    let req = GenRequest {
        prompt_tokens: prompt.clone(),
        max_new: 224,
        sampling: Sampling::Greedy,
        stop_eos: false,
        logprobs: None,
    };
    // Warmup request: builds the CUDA graph + JIT outside the timed runs.
    let warm = GenRequest {
        prompt_tokens: prompt,
        max_new: 4,
        sampling: Sampling::Greedy,
        stop_eos: false,
        logprobs: None,
    };

    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut plain = RunnerGenerator::new(runner, u32::MAX);
    let _ = collect(&mut plain, &warm);
    let (want, t_plain) = collect(&mut plain, &req);

    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut spec = RunnerGenerator::new(runner, u32::MAX).with_spec_lookup(true);
    let _ = collect(&mut spec, &warm);
    let (got, t_spec) = collect(&mut spec, &req);

    println!(
        "spec-lookup: plain {} tok in {t_plain:.2?} ({:.1} tok/s) | spec {} tok in {t_spec:.2?} ({:.1} tok/s) | speedup {:.2}x",
        want.len(),
        want.len() as f64 / t_plain.as_secs_f64(),
        got.len(),
        got.len() as f64 / t_spec.as_secs_f64(),
        t_plain.as_secs_f64() / t_spec.as_secs_f64(),
    );
    assert_eq!(
        got, want,
        "spec-lookup stream must equal plain greedy (lossless)"
    );
}
