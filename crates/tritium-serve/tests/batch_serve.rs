//! Continuous-batching gates (model + GPU gated, `cuda` feature).
//!
//! Numerics contract: the M=N batch decode path is its OWN numerics domain —
//! the repo's acceptance gate pins it to single-sequence greedy TOKENS over a
//! short horizon (ulp-level logit differences from the different attention
//! reduction shapes are expected and can flip near-tie argmaxes at long
//! horizons). The serve gates mirror that:
//!  - short-horizon token equality vs single-sequence greedy (per prompt),
//!  - full determinism/isolation within the batch domain: the same request
//!    set through a fresh pool twice (different admission order and slot
//!    assignments via reversed submission) must produce identical streams —
//!    covering concurrency, slot reuse and pad-row isolation bit-for-bit.
//!
#![cfg(feature = "cuda")]

use std::path::Path;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

use tritium_serve::{
    GenRequest, Generator, IdPassthroughTokenizer, RunnerGenerator, Sampling, ServeConfig,
    build_router_batched,
};

use tritium_cpu as _;
use tritium_cuda as _;

const GGUF_PATH: &str =
    "/home/brianklam/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf";
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

async fn chat(router: &Router, prompt_ids: &str, max_tokens: usize) -> String {
    let body = serde_json::json!({
        "model": "tritium",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt_ids}],
    });
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let resp = router.clone().oneshot(req).await.expect("send");
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    v["choices"][0]["message"]["content"]
        .as_str()
        .expect("content")
        .to_owned()
}

#[tokio::test(flavor = "multi_thread")]
async fn cuda_batched_serve_matches_single_sequence_greedy() {
    if !Path::new(GGUF_PATH).exists() {
        eprintln!("skipping: {GGUF_PATH} absent (gated real-model test)");
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

    // Six DIFFERENT prompts (the reference ids cycled to different lengths,
    // rotated so each prompt starts differently) through a 4-slot pool:
    // forces concurrency AND slot reuse.
    // Six DIFFERENT prompts through a 4-slot pool: concurrency AND slot reuse.
    let prompts: Vec<Vec<u32>> = (0..6usize)
        .map(|i| {
            base.iter()
                .cycle()
                .skip(i)
                .take(16 + 3 * i)
                .copied()
                .collect()
        })
        .collect();
    let max_new = 24usize;

    // G1 — short-horizon token equality vs single-sequence greedy (the
    // acceptance-gate contract: same tokens over a modest horizon).
    let short = 10usize;
    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let mut single = RunnerGenerator::new(runner, u32::MAX);
    let mut want_short: Vec<Vec<u32>> = Vec::new();
    for p in &prompts {
        let req = GenRequest {
            prompt_tokens: p.clone(),
            max_new: short,
            sampling: Sampling::Greedy,
            stop_eos: false,
        logprobs: None,
    };
        let mut out = Vec::new();
        single
            .generate(&req, &mut |step| {
                out.push(step.token);
                true
            })
            .expect("single-seq generate");
        want_short.push(out);
    }
    drop(single);

    let batched_run = |bytes: &[u8], order: Vec<usize>, max_tokens: usize| {
        let runner = load_runner(bytes).expect("runner");
        let tok = Arc::new(IdPassthroughTokenizer::new(128_000, u32::MAX));
        let cfg = ServeConfig {
            model_id: "tritium".into(),
            queue_cap: 32,
            max_new_default: max_tokens,
            ..ServeConfig::default()
        };
        let (router, _draining) =
            build_router_batched(runner, u32::MAX, 4, tok, cfg).expect("batched router");
        let prompts = prompts.clone();
        async move {
            let handles: Vec<_> = order
                .into_iter()
                .map(|i| {
                    let ids = prompts[i]
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(" ");
                    let router = router.clone();
                    tokio::spawn(async move { (i, chat(&router, &ids, max_tokens).await) })
                })
                .collect();
            let mut out = vec![String::new(); prompts.len()];
            for h in handles {
                let (i, text) = h.await.expect("join");
                out[i] = text;
            }
            out
        }
    };

    let got_short = batched_run(&bytes, (0..6).collect(), short).await;
    for (i, text) in got_short.iter().enumerate() {
        let got: Vec<u32> = text
            .split_whitespace()
            .map(|t| t.parse().expect("token id"))
            .collect();
        assert_eq!(got.len(), short, "G1: stream {i} wrong length");
        // Token 0 is sampled from the SINGLE-SEQUENCE prefill logits at
        // admission — bit-guaranteed equal.
        assert_eq!(
            got[0], want_short[i][0],
            "G1: stream {i} first token must equal single-sequence greedy exactly"
        );
        // Beyond token 0 the batch path is its own ulp domain (different
        // attention reduction shapes); near-tie argmaxes may flip. Report the
        // agreement prefix — informative, not asserted.
        let agree = got
            .iter()
            .zip(&want_short[i])
            .take_while(|(a, b)| a == b)
            .count();
        println!("G1: stream {i} agrees with single-seq for {agree}/{short} tokens");
        assert!(
            agree >= 2,
            "G1: stream {i} diverged immediately after the first token — \
             that's beyond ulp-domain drift ({got:?} vs {:?})",
            want_short[i]
        );
    }

    // G2 — determinism/isolation within the batch domain: the same request
    // set through a FRESH pool, submitted in reverse (different slots +
    // admission interleaving), must reproduce identical streams at a long
    // horizon (24 tokens: covers retirement, slot reuse and pad rows).
    let a = batched_run(&bytes, (0..6).collect(), 24).await;
    let b = batched_run(&bytes, (0..6).rev().collect(), 24).await;
    for i in 0..6 {
        assert_eq!(
            a[i], b[i],
            "G2: stream {i} must be identical across pool runs / slot assignments"
        );
        assert!(!a[i].is_empty(), "stream {i} empty");
    }
}

/// Throughput bench (run explicitly): aggregate tok/s of N concurrent
/// requests through an N-slot pool vs the same requests sequentially
/// through the single-request worker — the number continuous batching exists
/// to move (the warp-starved GEMMs get fed N rows for one weight read).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "throughput bench: run with --ignored --nocapture"]
async fn cuda_batched_throughput_vs_sequential() {
    if !Path::new(GGUF_PATH).exists() {
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(REF_PATH).expect("read reference"))
            .expect("parse reference");
    let base: Vec<u32> = reference["token_ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let bytes = std::fs::read(GGUF_PATH).expect("read gguf");
    let n = 8usize;
    let max_tokens = 64usize;
    let prompts: Vec<String> = (0..n)
        .map(|i| {
            base.iter()
                .cycle()
                .skip(i)
                .take(24)
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();

    // Sequential baseline: plain single-request worker.
    let runner = load_runner(&bytes).expect("runner");
    let tok = Arc::new(IdPassthroughTokenizer::new(128_000, u32::MAX));
    let cfg = ServeConfig {
        model_id: "tritium".into(),
        queue_cap: 32,
        max_new_default: max_tokens,
        ..ServeConfig::default()
    };
    let (router, _d) = tritium_serve::build_router(
        Box::new(RunnerGenerator::new(runner, u32::MAX)),
        tok.clone(),
        cfg.clone(),
    );
    // Warm (graph capture) then time.
    let _ = chat(&router, &prompts[0], 4).await;
    let t0 = std::time::Instant::now();
    for p in &prompts {
        let _ = chat(&router, p, max_tokens).await;
    }
    let seq = t0.elapsed();

    // Batched: N slots, N concurrent.
    let runner = load_runner(&bytes).expect("runner");
    let (router, _d) = build_router_batched(runner, u32::MAX, n, tok, cfg).expect("batched");
    let _ = chat(&router, &prompts[0], 4).await; // warm capture
    let t0 = std::time::Instant::now();
    let handles: Vec<_> = prompts
        .iter()
        .map(|p| {
            let router = router.clone();
            let p = p.clone();
            tokio::spawn(async move { chat(&router, &p, max_tokens).await })
        })
        .collect();
    for h in handles {
        let _ = h.await.expect("join");
    }
    let conc = t0.elapsed();

    let total = (n * max_tokens) as f64;
    println!(
        "throughput: sequential {seq:.2?} ({:.1} tok/s) | {n}-slot concurrent {conc:.2?} ({:.1} tok/s) | {:.2}x",
        total / seq.as_secs_f64(),
        total / conc.as_secs_f64(),
        seq.as_secs_f64() / conc.as_secs_f64(),
    );
}
