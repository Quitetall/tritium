//! OpenAI-wire contract tests (model-free, via `MockGenerator` + `tower::oneshot`).
//! These are the ADR 0010 / v0.80 serve gate: schema, SSE framing, stream==buffered,
//! finish_reason, stop strings, concurrency, backpressure, graceful shutdown.
//!
//! Run with `cargo test -p tritium-serve --features serve`.
#![cfg(feature = "serve")]

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;
use tritium_nn::Tokenizer;
use tritium_serve::{
    FinishReason, GenError, GenRequest, Generator, IdPassthroughTokenizer, MockGenerator,
    ServeConfig, Step, build_router,
};

/// A generator that always fails (for the backend-error / panic-resilience tests).
struct ErrGen;
impl Generator for ErrGen {
    fn generate(
        &mut self,
        _req: &GenRequest,
        _on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        Err(GenError::Backend("boom".to_owned()))
    }
    fn n_ctx(&self) -> usize {
        4096
    }
    fn vocab(&self) -> usize {
        128_256
    }
}

/// A generator that panics (proves the worker isolates panics + stays alive).
struct PanicGen;
impl Generator for PanicGen {
    fn generate(
        &mut self,
        _req: &GenRequest,
        _on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        panic!("kaboom in generate");
    }
    fn n_ctx(&self) -> usize {
        4096
    }
    fn vocab(&self) -> usize {
        128_256
    }
}

fn shared_tok() -> Arc<dyn Tokenizer + Send + Sync> {
    Arc::new(IdPassthroughTokenizer::default())
}

fn router_with(mock: MockGenerator, cfg: ServeConfig) -> (Router, Arc<AtomicBool>) {
    build_router(Box::new(mock), shared_tok(), cfg)
}

fn mock_router(script: Vec<u32>, end_reason: FinishReason) -> (Router, Arc<AtomicBool>) {
    router_with(
        MockGenerator {
            end_reason,
            ..MockGenerator::new(script)
        },
        ServeConfig::default(),
    )
}

fn chat(body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn send(router: &Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let resp = router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let body = resp
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes()
        .to_vec();
    (status, body)
}

/// Split an SSE body into its `data:` payloads (keep-alive comment lines skipped).
fn parse_sse(bytes: &[u8]) -> Vec<String> {
    let text = String::from_utf8_lossy(bytes);
    text.split("\n\n")
        .filter_map(|block| block.strip_prefix("data: ").map(str::to_owned))
        .collect()
}

/// The chunks before the terminal `[DONE]`, parsed as JSON.
fn sse_chunks(events: &[String]) -> Vec<Value> {
    events[..events.len() - 1]
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}

#[tokio::test]
async fn nonstream_schema_roundtrip() {
    let (router, _) = mock_router(vec![10, 11, 12], FinishReason::Stop);
    let (status, body) = send(
        &router,
        chat(json!({"model":"tritium","messages":[{"role":"user","content":"1 2"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "10 11 12");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], 2);
    assert_eq!(v["usage"]["completion_tokens"], 3);
    assert_eq!(v["usage"]["total_tokens"], 5);
}

#[tokio::test]
async fn sse_framing_and_done() {
    let (router, _) = mock_router(vec![10, 11, 12], FinishReason::Stop);
    let (status, body) = send(
        &router,
        chat(json!({"model":"tritium","stream":true,"messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let events = parse_sse(&body);
    assert_eq!(
        events.last().unwrap(),
        "[DONE]",
        "stream must end with [DONE]"
    );
    let chunks = sse_chunks(&events);
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(chunks[0]["choices"][0]["finish_reason"].is_null());
    let terminals: Vec<_> = chunks
        .iter()
        .filter(|c| !c["choices"][0]["finish_reason"].is_null())
        .collect();
    assert_eq!(terminals.len(), 1, "exactly one terminal chunk");
    assert_eq!(terminals[0]["choices"][0]["finish_reason"], "stop");
    // ids stable across chunks
    let id0 = chunks[0]["id"].as_str().unwrap();
    assert!(chunks.iter().all(|c| c["id"] == id0));
}

#[tokio::test]
async fn stream_concat_equals_nonstream() {
    let script = vec![10, 11, 12, 13];
    let (r1, _) = mock_router(script.clone(), FinishReason::Stop);
    let (_, nbody) = send(
        &r1,
        chat(json!({"model":"tritium","messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    let nv: Value = serde_json::from_slice(&nbody).unwrap();
    let nonstream = nv["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned();

    let (r2, _) = mock_router(script, FinishReason::Stop);
    let (_, sbody) = send(
        &r2,
        chat(json!({"model":"tritium","stream":true,"messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    let mut concat = String::new();
    for c in sse_chunks(&parse_sse(&sbody)) {
        if let Some(s) = c["choices"][0]["delta"]["content"].as_str() {
            concat.push_str(s);
        }
    }
    assert_eq!(concat, nonstream);
}

#[tokio::test]
async fn finish_reason_length_when_truncated() {
    let (router, _) = mock_router(vec![1, 2, 3, 4, 5], FinishReason::Stop);
    let (_, body) = send(
        &router,
        chat(json!({"model":"tritium","max_tokens":2,"messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn stop_string_truncates_nonstream() {
    // decode([10,11,12]) = "10 11 12"; stop "11" truncates to "10 ".
    let (router, _) = mock_router(vec![10, 11, 12], FinishReason::Length);
    let (_, body) = send(
        &router,
        chat(json!({"model":"tritium","stop":"11","messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "10 ");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn stop_string_truncates_stream() {
    let (router, _) = mock_router(vec![10, 11, 12], FinishReason::Length);
    let (_, body) = send(
        &router,
        chat(json!({"model":"tritium","stream":true,"stop":"11","messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    let events = parse_sse(&body);
    let chunks = sse_chunks(&events);
    let mut concat = String::new();
    for c in &chunks {
        if let Some(s) = c["choices"][0]["delta"]["content"].as_str() {
            concat.push_str(s);
        }
    }
    assert_eq!(concat, "10 ", "stop string truncates streamed content");
    let terminal = chunks
        .iter()
        .find(|c| !c["choices"][0]["finish_reason"].is_null())
        .unwrap();
    assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn models_and_health_drain() {
    let (router, draining) = mock_router(vec![1], FinishReason::Stop);
    let (s, body) = send(
        &router,
        Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "tritium");

    let (ok, _) = send(
        &router,
        Request::builder()
            .uri("/healthz")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(ok, StatusCode::OK);
    draining.store(true, Ordering::Relaxed);
    let (drained, _) = send(
        &router,
        Request::builder()
            .uri("/healthz")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(drained, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn invalid_requests_rejected() {
    let (router, _) = mock_router(vec![1], FinishReason::Stop);
    let (s, b) = send(&router, chat(json!({"model":"tritium","messages":[]}))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    let v: Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(v["error"]["param"], "messages");

    let (s2, _) = send(
        &router,
        chat(
            json!({"model":"tritium","temperature":9.0,"messages":[{"role":"user","content":"1"}]}),
        ),
    )
    .await;
    assert_eq!(s2, StatusCode::BAD_REQUEST);

    let (s3, _) = send(
        &router,
        chat(json!({"model":"does-not-exist","messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    assert_eq!(s3, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn concurrent_requests_complete_and_unique() {
    let (router, _) = mock_router(vec![10, 11, 12], FinishReason::Stop);
    let mut handles = Vec::new();
    for _ in 0..8 {
        let r = router.clone();
        handles.push(tokio::spawn(async move {
            send(
                &r,
                chat(json!({"model":"tritium","messages":[{"role":"user","content":"1"}]})),
            )
            .await
        }));
    }
    let mut ids = HashSet::new();
    for h in handles {
        let (status, body) = h.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "10 11 12");
        ids.insert(v["id"].as_str().unwrap().to_owned());
    }
    assert_eq!(ids.len(), 8, "each concurrent response has a unique id");
}

#[tokio::test]
async fn backpressure_429_when_full() {
    // A slow mock holds the single worker; with queue_cap=1, a burst overflows -> 429.
    let mock = MockGenerator {
        step_delay_ms: 300,
        ..MockGenerator::new(vec![1, 2, 3, 4])
    };
    let (router, _) = router_with(
        mock,
        ServeConfig {
            queue_cap: 1,
            ..ServeConfig::default()
        },
    );
    let mut handles = Vec::new();
    for _ in 0..6 {
        let r = router.clone();
        handles.push(tokio::spawn(async move {
            send(
                &r,
                chat(json!({"model":"tritium","messages":[{"role":"user","content":"1"}]})),
            )
            .await
            .0
        }));
    }
    let mut statuses = Vec::new();
    for h in handles {
        statuses.push(h.await.unwrap());
    }
    let rejected = statuses
        .iter()
        .filter(|s| **s == StatusCode::TOO_MANY_REQUESTS)
        .count();
    assert!(
        rejected >= 1,
        "expected at least one 429 under backpressure, got {statuses:?}"
    );
}

#[tokio::test]
async fn rejects_bad_params() {
    let (router, _) = mock_router(vec![1], FinishReason::Stop);
    // top_p out of range
    let (s1, _) = send(
        &router,
        chat(json!({"model":"tritium","top_p":1.5,"temperature":0.7,"messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    assert_eq!(s1, StatusCode::BAD_REQUEST, "top_p > 1 -> 400");
    // empty stop string
    let (s2, _) = send(
        &router,
        chat(json!({"model":"tritium","stop":"","messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    assert_eq!(s2, StatusCode::BAD_REQUEST, "empty stop -> 400");
    // max_tokens 0
    let (s3, _) = send(
        &router,
        chat(json!({"model":"tritium","max_tokens":0,"messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    assert_eq!(s3, StatusCode::BAD_REQUEST, "max_tokens 0 -> 400");
    // empty content -> empty prompt after tokenization
    let (s4, _) = send(
        &router,
        chat(json!({"model":"tritium","messages":[{"role":"user","content":"  "}]})),
    )
    .await;
    assert_eq!(s4, StatusCode::BAD_REQUEST, "empty prompt -> 400");
}

#[tokio::test]
async fn stream_backend_error_signals_error_not_clean_stop() {
    let tok: Arc<dyn Tokenizer + Send + Sync> = Arc::new(IdPassthroughTokenizer::default());
    let (router, _) = build_router(Box::new(ErrGen), tok, ServeConfig::default());
    let (status, body) = send(
        &router,
        chat(json!({"model":"tritium","stream":true,"messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK); // stream headers already committed
    let events = parse_sse(&body);
    assert_eq!(events.last().unwrap(), "[DONE]");
    let chunks = sse_chunks(&events);
    let terminal = chunks
        .iter()
        .find(|c| !c["choices"][0]["finish_reason"].is_null())
        .expect("a terminal chunk");
    assert_eq!(
        terminal["choices"][0]["finish_reason"], "error",
        "a backend error must surface a distinct finish_reason, not a clean stop"
    );
}

#[tokio::test]
async fn worker_survives_panic_and_stays_healthy() {
    let tok: Arc<dyn Tokenizer + Send + Sync> = Arc::new(IdPassthroughTokenizer::default());
    let (router, _) = build_router(Box::new(PanicGen), tok, ServeConfig::default());
    // First request: the generator panics; the worker catches it -> 500, not a hang.
    let (s1, _) = send(
        &router,
        chat(json!({"model":"tritium","messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    assert_eq!(s1, StatusCode::INTERNAL_SERVER_ERROR);
    // /healthz still ok — the panic did not zombify the worker.
    let (h, _) = send(
        &router,
        Request::builder()
            .uri("/healthz")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(h, StatusCode::OK, "worker survived the panic; /healthz ok");
    // A second request is still served (worker alive), not wedged.
    let (s2, _) = send(
        &router,
        chat(json!({"model":"tritium","messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    assert_eq!(s2, StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn graceful_shutdown_midstream_is_wellformed() {
    let mock = MockGenerator {
        step_delay_ms: 100,
        ..MockGenerator::new(vec![10, 11, 12, 13, 14, 15, 16, 17])
    };
    let (router, draining) = router_with(mock, ServeConfig::default());
    let r = router.clone();
    let handle = tokio::spawn(async move {
        send(
            &r,
            chat(
                json!({"model":"tritium","stream":true,"messages":[{"role":"user","content":"1"}]}),
            ),
        )
        .await
    });
    // Let a couple tokens flow, then drain mid-stream.
    tokio::time::sleep(Duration::from_millis(250)).await;
    draining.store(true, Ordering::Relaxed);

    let (status, body) = handle.await.unwrap();
    assert_eq!(status, StatusCode::OK);
    let events = parse_sse(&body);
    assert_eq!(
        events.last().unwrap(),
        "[DONE]",
        "interrupted stream still terminates with [DONE]"
    );
    let chunks = sse_chunks(&events);
    let terminals = chunks
        .iter()
        .filter(|c| !c["choices"][0]["finish_reason"].is_null())
        .count();
    assert_eq!(
        terminals, 1,
        "interrupted stream still has exactly one terminal chunk"
    );
}
