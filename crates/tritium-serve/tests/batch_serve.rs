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
    build_router_batched, build_router_batched_with_draft,
};

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

/// ADR 0021 drafter for the I0 spec-coexistence gate; the test skips cleanly
/// when absent (or when its vocab does not match the target's — the same
/// startup check `--draft-model` enforces).
static DRAFT_PATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    format!(
        "{}/blut/data/drafter-8L768-s3.gguf",
        std::env::var("HOME").unwrap_or_default()
    )
});

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
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
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
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");

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

async fn tree_post(
    router: &Router,
    path: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let req = Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let resp = router.clone().oneshot(req).await.expect("send");
    let status = resp.status();
    let bytes = resp.into_body().collect().await.expect("body").to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, v)
}

/// C4 gate — BASTION tree sessions coexist with the batch pool.
///
/// (a) Losslessness: open + two chained verifies on the batched server must
///     return EXACTLY what the single-worker server returns (same model, same
///     tree ops on the single-sequence KV — the mode must not change one
///     committed token).
/// (b) Coexistence: the same session ops succeed WHILE a chat stream decodes
///     in a slot, return the same tokens, and the chat stream is unaffected.
/// (c) The single-worker contract carries over: a chat ADMISSION closes the
///     session — the next verify gets 409 Conflict.
#[tokio::test(flavor = "multi_thread")]
async fn cuda_batched_tree_session_coexists() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
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
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");
    let session_prompt: Vec<u32> = base.iter().cycle().take(16).copied().collect();
    let drafts: Vec<u32> = base.iter().cycle().skip(2).take(2).copied().collect();

    // Two chained verify rounds against whatever router: open → verify
    // [root, d1, d2] chain → verify again rooted at the last committed token.
    // Returns (pending_token, committed_round1, committed_round2).
    let session_rounds = |router: Router, prompt: Vec<u32>, drafts: Vec<u32>| async move {
        let (st, v) = tree_post(
            &router,
            "/v1/tree/session",
            serde_json::json!({ "prompt_tokens": prompt }),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "session open failed: {v}");
        let root = v["pending_token"].as_u64().expect("pending_token") as u32;
        let mut rounds = Vec::new();
        let mut cur_root = root;
        for _ in 0..2 {
            let tokens = vec![cur_root, drafts[0], drafts[1]];
            let (st, v) = tree_post(
                &router,
                "/v1/tree/verify",
                serde_json::json!({ "tokens": tokens, "parents": [-1, 0, 1] }),
            )
            .await;
            assert_eq!(st, StatusCode::OK, "verify failed: {v}");
            let committed: Vec<u32> = v["committed"]
                .as_array()
                .expect("committed")
                .iter()
                .map(|t| t.as_u64().expect("token") as u32)
                .collect();
            assert!(!committed.is_empty(), "verify must commit >= 1 token");
            cur_root = *committed.last().expect("non-empty");
            rounds.push(committed);
        }
        (root, rounds)
    };

    // (a) Single-worker reference.
    let (single_root, single_rounds) = {
        let runner = load_runner(&bytes).expect("runner");
        let tok = Arc::new(IdPassthroughTokenizer::new(128_000, u32::MAX));
        let cfg = ServeConfig {
            model_id: "tritium".into(),
            queue_cap: 8,
            max_new_default: 32,
            ..ServeConfig::default()
        };
        let (router, _d) =
            tritium_serve::build_router(Box::new(RunnerGenerator::new(runner, u32::MAX)), tok, cfg);
        session_rounds(router, session_prompt.clone(), drafts.clone()).await
    };

    // Batched server, 2 slots.
    let runner = load_runner(&bytes).expect("runner");
    let tok = Arc::new(IdPassthroughTokenizer::new(128_000, u32::MAX));
    let cfg = ServeConfig {
        model_id: "tritium".into(),
        queue_cap: 8,
        max_new_default: 128,
        ..ServeConfig::default()
    };
    let (router, _draining) =
        build_router_batched(runner, u32::MAX, 2, tok, cfg).expect("batched router");

    // Warm the batch graph off the clock.
    let _ = chat(&router, "128000 791", 2).await;

    // (a) Idle-pool batched session == single-worker session, token for token.
    let (b_root, b_rounds) =
        session_rounds(router.clone(), session_prompt.clone(), drafts.clone()).await;
    assert_eq!(
        b_root, single_root,
        "C4a: batched root != single-worker root"
    );
    assert_eq!(
        b_rounds, single_rounds,
        "C4a: batched committed tokens != single-worker (mode must be lossless)"
    );

    // (b) Same session ops WHILE a chat stream decodes in the other slot.
    let a_ids = base
        .iter()
        .cycle()
        .skip(1)
        .take(16)
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let (a_handle, a_times) = spawn_stream(&router, &a_ids, 96);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while a_times.lock().expect("lock").len() < 2 {
        assert!(std::time::Instant::now() < deadline, "chat never streamed");
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let (c_root, c_rounds) =
        session_rounds(router.clone(), session_prompt.clone(), drafts.clone()).await;
    assert_eq!(c_root, single_root, "C4b: root changed under coexistence");
    assert_eq!(
        c_rounds, single_rounds,
        "C4b: committed tokens changed while a chat stream was live"
    );
    let a_before = a_times.lock().expect("lock").len();
    a_handle.abort();
    assert!(
        a_before >= 2,
        "C4b: chat stream stalled out during tree traffic"
    );

    // (a2) Force an ACCEPT (junk drafts degenerate to L=1 plain steps):
    // round 1 told us the argmax after the root is single_rounds[0][0] —
    // draft exactly that. Both modes must commit ≥2 tokens (accepted draft
    // + bonus) and agree exactly. NB an accepted CHAIN is compaction-free
    // (node == k along the path); KV row-moving promotion is covered by the
    // kernel-level tree gates, not this HTTP-level one.
    let informed = vec![single_root, single_rounds[0][0], drafts[1]];
    let accept_on = |router: Router, prompt: Vec<u32>, tokens: Vec<u32>| async move {
        let (st, v) = tree_post(
            &router,
            "/v1/tree/session",
            serde_json::json!({ "prompt_tokens": prompt }),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "accept-round open failed: {v}");
        let (st, v) = tree_post(
            &router,
            "/v1/tree/verify",
            serde_json::json!({ "tokens": tokens, "parents": [-1, 0, 1] }),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "accept-round verify failed: {v}");
        v["committed"]
            .as_array()
            .expect("committed")
            .iter()
            .map(|t| t.as_u64().expect("token") as u32)
            .collect::<Vec<u32>>()
    };
    let single_accept = {
        let runner = load_runner(&bytes).expect("runner");
        let tok = Arc::new(IdPassthroughTokenizer::new(128_000, u32::MAX));
        let cfg = ServeConfig {
            model_id: "tritium".into(),
            queue_cap: 8,
            max_new_default: 32,
            ..ServeConfig::default()
        };
        let (r, _d) =
            tritium_serve::build_router(Box::new(RunnerGenerator::new(runner, u32::MAX)), tok, cfg);
        accept_on(r, session_prompt.clone(), informed.clone()).await
    };
    assert!(
        single_accept.len() >= 2,
        "informed draft must be accepted (committed {single_accept:?})"
    );
    let batched_accept = accept_on(router.clone(), session_prompt.clone(), informed).await;
    assert_eq!(
        batched_accept, single_accept,
        "C4a2: accepted-path commit differs across modes"
    );

    // (c) A chat ADMISSION closes the session: open, run a chat to Done,
    // then verify must 409.
    let (st, v) = tree_post(
        &router,
        "/v1/tree/session",
        serde_json::json!({ "prompt_tokens": session_prompt }),
    )
    .await;
    assert_eq!(st, StatusCode::OK, "re-open failed: {v}");
    let root = v["pending_token"].as_u64().expect("pending_token") as u32;
    let _ = chat(&router, &a_ids, 4).await;
    let (st, v) = tree_post(
        &router,
        "/v1/tree/verify",
        serde_json::json!({ "tokens": [root, drafts[0]], "parents": [-1, 0] }),
    )
    .await;
    assert_eq!(
        st,
        StatusCode::CONFLICT,
        "C4c: verify after a chat admission must 409 (got {st}: {v})"
    );
    println!(
        "C4: batched tree session == single-worker (root {single_root}, {} + {} committed; \
         accepted-path round {} committed); coexists with a live chat stream; chat \
         admission closes it (409)",
        single_rounds[0].len(),
        single_rounds[1].len(),
        single_accept.len(),
    );
}

/// C3 gate — paged KV streams equal dense streams exactly. Paging is
/// bit-exact by construction (same values, different addresses; gated at the
/// kernel level by `cuda_batch_paged_matches_dense_bit_exact`), so the same
/// request set through a paged pool must produce IDENTICAL text. The pool is
/// deliberately SCARCER than the slots (3 pages, 4 slots, 1 page per
/// request) so a 4th concurrent admission finds a free SLOT but no free
/// PAGE — reserve fails → the job parks and is retried after a retirement.
/// Parking delays a stream; it must never change one. (A 4-page pool would
/// never park: every retirement frees slot and page together, so a free
/// slot would imply a free page — review finding on the first version.)
#[tokio::test(flavor = "multi_thread")]
async fn cuda_batched_paged_streams_equal_dense() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
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
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");
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
    let max_tokens = 24usize;

    let run = |bytes: &[u8], kv_pool_tokens: Option<usize>| {
        let runner = load_runner(bytes).expect("runner");
        let tok = Arc::new(IdPassthroughTokenizer::new(128_000, u32::MAX));
        let cfg = ServeConfig {
            model_id: "tritium".into(),
            queue_cap: 32,
            max_new_default: max_tokens,
            kv_pool_tokens,
            ..ServeConfig::default()
        };
        let (router, _draining) =
            build_router_batched(runner, u32::MAX, 4, tok, cfg).expect("batched router");
        let prompts = prompts.clone();
        async move {
            let handles: Vec<_> = (0..prompts.len())
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

    let dense = run(&bytes, None).await;
    // 3 pages × 256 tokens for 4 slots: each request needs 1 page (max
    // prompt 31 + 24 ≤ 256), so at most 3 slots decode concurrently and the
    // next admission PARKS on page exhaustion until a retirement.
    let paged = run(&bytes, Some(768)).await;
    for i in 0..prompts.len() {
        assert!(!dense[i].is_empty(), "dense stream {i} empty");
        assert_eq!(
            dense[i], paged[i],
            "G3: paged stream {i} must equal the dense stream exactly"
        );
    }
    println!("G3: 6 paged streams identical to dense (3-page/4-slot pool — parking exercised)");
}

/// Spawn a streaming chat request; each emitted token's arrival time is pushed
/// into the returned shared vec (live — the caller polls it mid-stream). SSE
/// events may split across body frames, so a rolling buffer reassembles them.
fn spawn_stream(
    router: &Router,
    ids: &str,
    max_tokens: usize,
) -> (
    tokio::task::JoinHandle<()>,
    Arc<std::sync::Mutex<Vec<std::time::Instant>>>,
) {
    let body = serde_json::json!({
        "model": "tritium",
        "max_tokens": max_tokens,
        "stream": true,
        "messages": [{"role": "user", "content": ids}],
    });
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let router = router.clone();
    let times = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = times.clone();
    let handle = tokio::spawn(async move {
        let resp = router.oneshot(req).await.expect("send");
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body = resp.into_body();
        let mut buf = String::new();
        while let Some(frame) = body.frame().await {
            let Ok(frame) = frame else { break };
            let Some(data) = frame.data_ref() else {
                continue;
            };
            buf.push_str(&String::from_utf8_lossy(data));
            while let Some(pos) = buf.find("\n\n") {
                let event: String = buf.drain(..pos + 2).collect();
                let Some(json_str) = event.trim().strip_prefix("data: ") else {
                    continue;
                };
                if json_str.trim() == "[DONE]" {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) else {
                    continue;
                };
                if v["choices"][0]["delta"]["content"]
                    .as_str()
                    .is_some_and(|s| !s.trim().is_empty())
                {
                    sink.lock()
                        .expect("times lock")
                        .push(std::time::Instant::now());
                }
            }
        }
    });
    (handle, times)
}

/// C1 gate — chunked prefill keeps live slots streaming during admission.
///
/// A short request (A) decodes steadily in one slot; then a 2048-token-prompt
/// request (B) is admitted into the other. With monolithic admission A stalls
/// for B's entire prefill (~0–1 tokens in that window); with chunked prefill
/// A gets a decode step between every chunk (~prompt/chunk tokens). The gate
/// asserts the interleaving structurally (≥4 A-tokens inside B's admission
/// window) and prints A's max inter-token gap for the log. Set
/// `TRITIUM_PREFILL_CHUNK` huge to reproduce the "before" behavior.
#[tokio::test(flavor = "multi_thread")]
async fn cuda_batched_admission_interleaves_live_slot() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
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
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");
    let Some(runner) = load_runner(&bytes) else {
        return;
    };
    let tok = Arc::new(IdPassthroughTokenizer::new(128_000, u32::MAX));
    let cfg = ServeConfig {
        model_id: "tritium".into(),
        queue_cap: 8,
        max_new_default: 256,
        ..ServeConfig::default()
    };
    let (router, _draining) =
        build_router_batched(runner, u32::MAX, 2, tok, cfg).expect("batched router");

    let join_ids = |n: usize| {
        base.iter()
            .cycle()
            .take(n)
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(" ")
    };
    // Warm: graph capture + first prefill paths off the clock.
    let _ = chat(&router, &join_ids(8), 2).await;

    // A: short prompt, long budget — must outlive B's admission.
    let (a_handle, a_times) = spawn_stream(&router, &join_ids(16), 256);
    let wait_deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while a_times.lock().expect("lock").len() < 4 {
        assert!(
            std::time::Instant::now() < wait_deadline,
            "A never started streaming"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    // B: 2048-token prompt into the free slot.
    let t_submit = std::time::Instant::now();
    let (b_handle, b_times) = spawn_stream(&router, &join_ids(2048), 4);
    let b_first = loop {
        if let Some(&t) = b_times.lock().expect("lock").first() {
            break t;
        }
        assert!(
            std::time::Instant::now() < wait_deadline,
            "B never produced a token"
        );
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    };

    // A tokens that landed inside B's admission window, and A's max gap there.
    let a_snapshot: Vec<std::time::Instant> = a_times.lock().expect("lock").clone();
    let in_window: Vec<std::time::Instant> = a_snapshot
        .iter()
        .copied()
        .filter(|&t| t > t_submit && t < b_first)
        .collect();
    let mut max_gap = std::time::Duration::ZERO;
    let mut prev = t_submit;
    for &t in &in_window {
        max_gap = max_gap.max(t - prev);
        prev = t;
    }
    max_gap = max_gap.max(b_first - prev);
    println!(
        "C1: B admission {:?} (2048-token prompt); A tokens inside the window: {}; \
         A max inter-token gap in window: {:?}",
        b_first - t_submit,
        in_window.len(),
        max_gap,
    );
    assert!(
        in_window.len() >= 4,
        "C1: live slot starved during admission — {} tokens in a {:?} window \
         (monolithic-prefill behavior)",
        in_window.len(),
        b_first - t_submit,
    );

    a_handle.abort(); // measurement done; disconnecting A also exercises retire-on-close
    let _ = b_handle.await;
}

/// Spawn a streaming chat request, accumulating every SSE delta's content
/// into the returned shared String (whitespace-separated token ids under the
/// passthrough tokenizer — parse with `split_whitespace`). Same SSE
/// reassembly as [`spawn_stream`], collecting text instead of arrival times.
fn spawn_token_stream(
    router: &Router,
    ids: &str,
    max_tokens: usize,
) -> (tokio::task::JoinHandle<()>, Arc<std::sync::Mutex<String>>) {
    let body = serde_json::json!({
        "model": "tritium",
        "max_tokens": max_tokens,
        "stream": true,
        "messages": [{"role": "user", "content": ids}],
    });
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let router = router.clone();
    let text = Arc::new(std::sync::Mutex::new(String::new()));
    let sink = text.clone();
    let handle = tokio::spawn(async move {
        let resp = router.oneshot(req).await.expect("send");
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body = resp.into_body();
        let mut buf = String::new();
        while let Some(frame) = body.frame().await {
            let Ok(frame) = frame else { break };
            let Some(data) = frame.data_ref() else {
                continue;
            };
            buf.push_str(&String::from_utf8_lossy(data));
            while let Some(pos) = buf.find("\n\n") {
                let event: String = buf.drain(..pos + 2).collect();
                let Some(json_str) = event.trim().strip_prefix("data: ") else {
                    continue;
                };
                if json_str.trim() == "[DONE]" {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) else {
                    continue;
                };
                if let Some(s) = v["choices"][0]["delta"]["content"].as_str() {
                    sink.lock().expect("text lock").push_str(s);
                }
            }
        }
    });
    (handle, text)
}

/// I0 gate (ADR 0032 L3) — speculative decoding coexists with batch slots
/// under the "spec-when-solo, migrate-on-admission" contract.
///
/// Phase 1 (solo losslessness): one greedy request on a 4-slot batched server
/// WITH a drafter must stream tokens IDENTICAL to the single-worker spec
/// server on the same model + drafter. Both run the same single-sequence
/// prefill + `tree_verify_greedy` ops — this is NOT the batch-decode ulp
/// domain, so exact equality is the contract (spec is lossless: only target
/// argmaxes are committed).
///
/// Phase 2 (migration): start a spec request, then admit a second request
/// mid-stream. The spec sequence must migrate into a batch slot (continuation
/// admission = its full history — the boundary token is bit-guaranteed by the
/// single-sequence prefill), both requests must complete, and the first
/// stream must STILL equal the single-worker reference.
#[tokio::test(flavor = "multi_thread")]
async fn cuda_batched_spec_solo_matches_single_worker() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return;
    }
    if !Path::new(&*DRAFT_PATH).exists() {
        eprintln!("skipping: {} absent (gated drafter test)", *DRAFT_PATH);
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
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");
    let dbytes = std::fs::read(&*DRAFT_PATH).expect("read draft gguf");

    let Some(target) = load_runner(&bytes) else {
        return;
    };
    let draft = load_runner(&dbytes).expect("draft runner");
    if draft.weights.vocab != target.weights.vocab {
        eprintln!(
            "skipping: drafter vocab {} != target vocab {} (--draft-model requires a \
             shared tokenizer; this drafter is for another target)",
            draft.weights.vocab, target.weights.vocab
        );
        return;
    }

    let prompt: Vec<u32> = base.iter().cycle().take(16).copied().collect();
    let horizon = 64usize;
    let ids = prompt
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ");

    // Single-worker spec reference: the exact loop the batched I0 path
    // mirrors (generator.rs generate_spec_lookup), driven directly.
    let mut single = RunnerGenerator::new(target, u32::MAX).with_draft_model(draft);
    let req = GenRequest {
        prompt_tokens: prompt.clone(),
        max_new: horizon,
        sampling: Sampling::Greedy,
        stop_eos: false,
        logprobs: None,
    };
    let mut want: Vec<u32> = Vec::new();
    single
        .generate(&req, &mut |step| {
            want.push(step.token);
            true
        })
        .expect("single-worker spec generate");
    assert_eq!(want.len(), horizon, "reference stream wrong length");
    drop(single);

    // Batched server (4 slots) WITH the drafter attached.
    let runner = load_runner(&bytes).expect("runner");
    let draft = load_runner(&dbytes).expect("draft runner");
    let tok = Arc::new(IdPassthroughTokenizer::new(128_000, u32::MAX));
    let cfg = ServeConfig {
        model_id: "tritium".into(),
        queue_cap: 32,
        max_new_default: horizon,
        ..ServeConfig::default()
    };
    let (router, _draining) = build_router_batched_with_draft(runner, draft, u32::MAX, 4, tok, cfg)
        .expect("batched router");

    // Phase 1 — solo spec on the batched server == single-worker spec,
    // token for token.
    let text = chat(&router, &ids, horizon).await;
    let got: Vec<u32> = text
        .split_whitespace()
        .map(|t| t.parse().expect("token id"))
        .collect();
    assert_eq!(
        got, want,
        "I0 phase 1: solo batched spec stream != single-worker spec stream \
         (spec must be lossless in batched mode)"
    );

    // Phase 2 — start a spec request, admit a second request mid-stream.
    let (a_handle, a_text) = spawn_token_stream(&router, &ids, horizon);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        let n = a_text.lock().expect("lock").split_whitespace().count();
        if n >= 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "spec stream never started"
        );
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    // B triggers the migration: A's continuation prefills FIRST (FIFO with
    // no re-emits), then B admits into another slot; both decode lockstep.
    // Teeth: A must still have a healthy tail of its stream left when B is
    // fired, or this phase silently degrades to "B after A finished" and
    // exercises zero migration code. ≥8 remaining tokens is >25ms of decode
    // on any GPU this suite runs on — the local HTTP round-trip wins that
    // race; a violation means the gate went vacuous, not that serving broke.
    let a_before_b = a_text.lock().expect("lock").split_whitespace().count();
    assert!(
        a_before_b + 8 <= horizon,
        "I0 phase 2 lost the migration race before it started: A had \
         {a_before_b}/{horizon} tokens when B was sent — gate is vacuous"
    );
    let b_ids = base
        .iter()
        .cycle()
        .skip(3)
        .take(12)
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let b_text = chat(&router, &b_ids, 8).await;
    let b: Vec<u32> = b_text
        .split_whitespace()
        .map(|t| t.parse().expect("token id"))
        .collect();
    assert_eq!(b.len(), 8, "I0 phase 2: admitted request must complete");
    a_handle.await.expect("join spec stream");
    let a: Vec<u32> = a_text
        .lock()
        .expect("lock")
        .split_whitespace()
        .map(|t| t.parse().expect("token id"))
        .collect();
    let migrated_at = a.iter().zip(&want).take_while(|(x, y)| x == y).count();
    assert_eq!(a.len(), horizon, "I0 phase 2: spec stream wrong length");
    assert_eq!(
        a, want,
        "I0 phase 2: migration must preserve exact greedy (agreed for \
         {migrated_at}/{horizon} tokens)"
    );
    println!(
        "I0: solo spec == single worker ({horizon} tokens); migration mid-stream \
         preserved the stream exactly (B admitted and completed, {} tokens)",
        b.len()
    );
}

/// Multi-slot spec gate (ADR 0032 L3 serve wiring of I1+I4) — N concurrent
/// greedy requests on a `--batch-slots 3 --draft-model` worker decode via
/// batched draft → grouped multi-slot tree verify → per-slot commit rounds,
/// and EVERY stream is token-identical to the single-worker spec path run
/// sequentially (the I0 gate's comparison pattern, extended to N=3). This is
/// NOT the batch-decode ulp domain: adoption copies the single-sequence
/// prefill bit-exactly, slot tree-verifies are engine-gated identical to the
/// single-sequence verify (I2/I4), and drafts never affect committed tokens
/// (losslessness) — so exact equality is the contract.
///
/// Phase A (concurrent): three different prompts submitted together — the
/// first admission takes the I0 solo path, the second triggers migration,
/// and the pool then runs multi-slot rounds; all three streams must equal
/// their references exactly. Teeth against a silent lockstep fallback: the
/// lockstep path never bumps the spec counters, so most of the 3×horizon
/// tokens must have been committed by spec verifies.
///
/// Phase B (staggered admission): A+B mid-flight in multi-spec, C arrives —
/// C admits into the spec pool through the chunked prefill (spec rounds
/// coexist with admissions) and all three streams stay exact.
///
/// Phase C (EOS mid-round): a server whose eos is a token known to appear
/// mid-stream in reference 0. Spec is lossless, so the greedy stream is
/// eos-independent and `stop_eos` only truncates it — the affected stream
/// must stop exactly at the first eos occurrence (emitted, then retired
/// mid-round) while the survivors run to their full horizon.
#[tokio::test(flavor = "multi_thread")]
async fn cuda_batched_spec_multi_matches_single_worker() {
    if !Path::new(&*GGUF_PATH).exists() {
        eprintln!("skipping: {} absent (gated real-model test)", *GGUF_PATH);
        return;
    }
    if !Path::new(&*DRAFT_PATH).exists() {
        eprintln!("skipping: {} absent (gated drafter test)", *DRAFT_PATH);
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
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");
    let dbytes = std::fs::read(&*DRAFT_PATH).expect("read draft gguf");

    let Some(target) = load_runner(&bytes) else {
        return;
    };
    let draft = load_runner(&dbytes).expect("draft runner");
    if draft.weights.vocab != target.weights.vocab {
        eprintln!(
            "skipping: drafter vocab {} != target vocab {} (--draft-model requires a \
             shared tokenizer; this drafter is for another target)",
            draft.weights.vocab, target.weights.vocab
        );
        return;
    }

    let horizon = 48usize;
    // Three DIFFERENT prompts (lengths 16/19/22, distinct rotations).
    let prompts: Vec<Vec<u32>> = (0..3usize)
        .map(|i| {
            base.iter()
                .cycle()
                .skip(2 * i)
                .take(16 + 3 * i)
                .copied()
                .collect()
        })
        .collect();
    let ids: Vec<String> = prompts
        .iter()
        .map(|p| p.iter().map(u32::to_string).collect::<Vec<_>>().join(" "))
        .collect();

    // Single-worker spec references, run SEQUENTIALLY on one generator (the
    // production single-worker pattern — generate resets per request).
    let mut single = RunnerGenerator::new(target, u32::MAX).with_draft_model(draft);
    let mut want: Vec<Vec<u32>> = Vec::new();
    for p in &prompts {
        let req = GenRequest {
            prompt_tokens: p.clone(),
            max_new: horizon,
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
            .expect("single-worker spec generate");
        assert_eq!(out.len(), horizon, "reference stream wrong length");
        want.push(out);
    }
    drop(single);

    let parse = |text: &str| -> Vec<u32> {
        text.split_whitespace()
            .map(|t| t.parse().expect("token id"))
            .collect()
    };

    // Batched server: 3 slots WITH the drafter.
    let runner = load_runner(&bytes).expect("runner");
    let draft = load_runner(&dbytes).expect("draft runner");
    let tok = Arc::new(IdPassthroughTokenizer::new(128_000, u32::MAX));
    let cfg = ServeConfig {
        model_id: "tritium".into(),
        queue_cap: 32,
        max_new_default: horizon,
        ..ServeConfig::default()
    };
    let (router, _draining) =
        build_router_batched_with_draft(runner, draft, u32::MAX, 3, tok.clone(), cfg.clone())
            .expect("batched router");

    // Warm graph captures off the clock.
    let _ = chat(&router, "128000 791", 2).await;

    // Phase A — three concurrent greedy streams, all exact.
    let spec_before =
        tritium_serve::generator::SPEC_COMMITTED.load(std::sync::atomic::Ordering::Relaxed);
    let verifies_before =
        tritium_serve::generator::SPEC_VERIFIES.load(std::sync::atomic::Ordering::Relaxed);
    let handles: Vec<_> = (0..3)
        .map(|i| {
            let router = router.clone();
            let ids = ids[i].clone();
            tokio::spawn(async move { (i, chat(&router, &ids, horizon).await) })
        })
        .collect();
    for h in handles {
        let (i, text) = h.await.expect("join");
        assert_eq!(
            parse(&text),
            want[i],
            "phase A: concurrent stream {i} != single-worker spec stream \
             (multi-slot spec must be lossless)"
        );
    }
    let spec_delta = tritium_serve::generator::SPEC_COMMITTED
        .load(std::sync::atomic::Ordering::Relaxed)
        - spec_before;
    assert!(
        spec_delta >= (3 * horizon as u64) / 2,
        "phase A teeth: only {spec_delta} of {} tokens were committed by spec \
         verifies — the pool silently fell back to lockstep",
        3 * horizon
    );
    // Acceptance teeth: streams stay EXACT even with a corrupted drafter KV
    // (spec is lossless — bad drafts just reject wholesale), so pin the
    // tok/verify ratio too. The fixture achieves 2.80 tok/verify
    // (2026-08-03 run: 140 committed / 50 verifies); the floor is
    // max(half of that, 1.5) = 1.5 — a collapse toward ~1 means the
    // drafter rows are junk.
    let verify_delta = tritium_serve::generator::SPEC_VERIFIES
        .load(std::sync::atomic::Ordering::Relaxed)
        - verifies_before;
    let tok_per_verify = spec_delta as f64 / verify_delta.max(1) as f64;
    assert!(
        tok_per_verify >= 1.5,
        "phase A teeth: tok/verify collapsed to {tok_per_verify:.2} \
         ({spec_delta} committed / {verify_delta} verifies) — drafter-KV \
         state is likely corrupt (drafts rejecting wholesale)"
    );

    // Phase B — staggered admission: A+B mid-flight, C arrives.
    let (a_handle, a_text) = spawn_token_stream(&router, &ids[0], horizon);
    let (b_handle, b_text) = spawn_token_stream(&router, &ids[1], horizon);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        let na = a_text.lock().expect("lock").split_whitespace().count();
        let nb = b_text.lock().expect("lock").split_whitespace().count();
        if na >= 4 && nb >= 4 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "phase B: streams never started"
        );
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    // Teeth (the I0 vacuousness guard): both streams must still have a
    // healthy tail left when C is fired, or this phase degrades to
    // "C after A/B finished" and exercises no mid-flight admission.
    let na = a_text.lock().expect("lock").split_whitespace().count();
    let nb = b_text.lock().expect("lock").split_whitespace().count();
    assert!(
        na + 8 <= horizon && nb + 8 <= horizon,
        "phase B lost the admission race before it started (A {na}/{horizon}, \
         B {nb}/{horizon} when C was sent) — gate is vacuous"
    );
    let c_text = chat(&router, &ids[2], horizon).await;
    assert_eq!(
        parse(&c_text),
        want[2],
        "phase B: stream C (admitted mid-multi-spec) != single-worker spec stream"
    );
    a_handle.await.expect("join A");
    b_handle.await.expect("join B");
    assert_eq!(
        parse(&a_text.lock().expect("lock")),
        want[0],
        "phase B: stream A changed across C's admission"
    );
    assert_eq!(
        parse(&b_text.lock().expect("lock")),
        want[1],
        "phase B: stream B changed across C's admission"
    );
    drop(router);

    // Phase C — EOS mid-round: eos = a token whose FIRST occurrence in
    // reference 0 is mid-stream (a token repeated from position 0 would
    // truncate at the adoption first-token and exercise no mid-round
    // retirement). Spec losslessness makes the greedy stream eos-independent;
    // stop_eos truncates it at the first occurrence (which IS emitted — the
    // tokenizer eos stays u32::MAX so the detok surfaces it).
    let eos_tok = (8..horizon - 8)
        .rev()
        .map(|j| want[0][j])
        .find(|t| {
            let first = want[0].iter().position(|x| x == t).expect("t is in want");
            (8..horizon - 8).contains(&first)
        })
        .expect("reference stream has no token first-appearing mid-stream");
    let expects: Vec<Vec<u32>> = want
        .iter()
        .map(|w| {
            w.iter()
                .position(|&t| t == eos_tok)
                .map_or_else(|| w.clone(), |j| w[..=j].to_vec())
        })
        .collect();
    assert!(
        expects[0].len() >= 8 && expects[0].len() < horizon,
        "phase C teeth: the EOS must fire mid-stream (fires at {})",
        expects[0].len()
    );
    let runner = load_runner(&bytes).expect("runner");
    let draft = load_runner(&dbytes).expect("draft runner");
    let (router, _draining) = build_router_batched_with_draft(runner, draft, eos_tok, 3, tok, cfg)
        .expect("batched router (eos)");
    let handles: Vec<_> = (0..3)
        .map(|i| {
            let router = router.clone();
            let ids = ids[i].clone();
            tokio::spawn(async move { (i, chat(&router, &ids, horizon).await) })
        })
        .collect();
    for h in handles {
        let (i, text) = h.await.expect("join");
        assert_eq!(
            parse(&text),
            expects[i],
            "phase C: stream {i} must truncate exactly at the first eos \
             occurrence (EOS-mid-round retirement)"
        );
    }
    println!(
        "multi-spec: 3 concurrent == single worker ({horizon} tokens each, \
         {spec_delta} spec-committed over {verify_delta} verifies = \
         {tok_per_verify:.2} tok/verify); staggered admission exact; EOS \
         mid-round truncated stream 0 at {} tokens while the others ran to \
         {horizon}",
        expects[0].len(),
    );
}

/// Throughput bench (run explicitly): aggregate tok/s of N concurrent
/// requests through an N-slot pool vs the same requests sequentially
/// through the single-request worker — the number continuous batching exists
/// to move (the warp-starved GEMMs get fed N rows for one weight read).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "throughput bench: run with --ignored --nocapture"]
async fn cuda_batched_throughput_vs_sequential() {
    if !Path::new(&*GGUF_PATH).exists() {
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
    let bytes = std::fs::read(&*GGUF_PATH).expect("read gguf");
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
