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
    AdmissionPolicy, FinishReason, GenError, GenRequest, Generator, IdPassthroughTokenizer,
    MockGenerator, PrincipalRateLimit, RequestLimits, ServeConfig, Step, build_router,
    build_router_governed, build_router_with_limits,
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
    // Stop-string hits CANCEL generation (review parity fix): usage counts
    // tokens up to the match (2: "10", "11"), not the whole scripted budget,
    // and agrees with what the streamed path reports for the same request.
    assert_eq!(v["usage"]["completion_tokens"], 2);
}

/// OpenAI parity: stream_options without stream:true is a 400.
#[tokio::test]
async fn stream_options_requires_stream() {
    let (router, _) = mock_router(vec![1], FinishReason::Stop);
    let (status, _) = send(
        &router,
        chat(
            json!({"model":"tritium","stream_options":{"include_usage":true},
                    "messages":[{"role":"user","content":"1"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
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
async fn models_liveness_and_readiness_split_during_drain() {
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

    let (ok, body) = send(
        &router,
        Request::builder()
            .uri("/healthz")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(ok, StatusCode::OK);
    let health: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(health["worker_alive"], true);
    assert_eq!(health["draining"], false);
    let (ready, body) = send(
        &router,
        Request::builder()
            .uri("/readyz")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(ready, StatusCode::OK);
    let readiness: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(readiness["status"], "ready");

    draining.store(true, Ordering::Relaxed);
    let (alive, body) = send(
        &router,
        Request::builder()
            .uri("/healthz")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(alive, StatusCode::OK);
    let health: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(health["worker_alive"], true);
    assert_eq!(health["draining"], true);
    let (drained, body) = send(
        &router,
        Request::builder()
            .uri("/readyz")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(drained, StatusCode::SERVICE_UNAVAILABLE);
    let readiness: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(readiness["status"], "not_ready");
    assert_eq!(readiness["draining"], true);
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
async fn request_resource_limits_reject_before_queue_admission() {
    let (router, _) = build_router_with_limits(
        Box::new(MockGenerator::new(vec![1])),
        shared_tok(),
        ServeConfig::default(),
        RequestLimits {
            max_messages: 1,
            max_prompt_bytes: 8,
            max_prompt_tokens: 2,
            max_new_tokens: 2,
            max_total_tokens: 3,
        },
    );

    let cases = [
        (
            json!({"model":"tritium","messages":[
                {"role":"user","content":"1"},
                {"role":"user","content":"2"}
            ]}),
            "messages",
        ),
        (
            json!({"model":"tritium","messages":[
                {"role":"user","content":"12345"}
            ]}),
            "messages",
        ),
        (
            json!({"model":"tritium","messages":[
                {"role":"u","content":"1 2 3"}
            ]}),
            "messages",
        ),
        (
            json!({"model":"tritium","max_tokens":3,"messages":[
                {"role":"u","content":"1"}
            ]}),
            "max_tokens",
        ),
        (
            json!({"model":"tritium","max_tokens":2,"messages":[
                {"role":"u","content":"1 2"}
            ]}),
            "max_tokens",
        ),
    ];

    for (body, expected_param) in cases {
        let (status, response) = send(&router, chat(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let value: Value = serde_json::from_slice(&response).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["param"], expected_param);
    }
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

// ───────────────────── BASTION tree-verify surface (ADR 0014) ─────────────────────

/// A session-capable test generator: scripted pending token + committed
/// streams, plus the same open/invalidate semantics `RunnerGenerator` has —
/// lets the contract suite pin every tree status code without a model.
#[derive(Debug)]
struct TreeMock {
    session_open: bool,
    pending: u32,
    committed: Vec<u32>,
    /// When set, `tree_verify` returns this instead of `committed`.
    verify_error: Option<tritium_serve::TreeOpError>,
}

impl Generator for TreeMock {
    fn generate(
        &mut self,
        _req: &GenRequest,
        on_step: &mut dyn FnMut(Step) -> bool,
    ) -> Result<(), GenError> {
        // A generation invalidates the session (mirrors RunnerGenerator).
        self.session_open = false;
        let _ = on_step(Step {
            token: 7,
            finished: true,
            finish_reason: Some(FinishReason::Stop),
            logprobs: None,
        });
        Ok(())
    }
    fn n_ctx(&self) -> usize {
        4096
    }
    fn vocab(&self) -> usize {
        128_256
    }
    fn open_tree_session(&mut self, _prompt: &[u32]) -> Result<u32, tritium_serve::TreeOpError> {
        self.session_open = true;
        Ok(self.pending)
    }
    fn tree_verify(
        &mut self,
        _tokens: &[u32],
        _parents: &[i32],
    ) -> Result<Vec<u32>, tritium_serve::TreeOpError> {
        if !self.session_open {
            return Err(tritium_serve::TreeOpError::Conflict(
                "no open tree session".to_owned(),
            ));
        }
        match &self.verify_error {
            Some(e) => Err(e.clone()),
            None => Ok(self.committed.clone()),
        }
    }
}

async fn post_json(
    router: &mut Router,
    path: &str,
    body: serde_json::Value,
) -> (u16, serde_json::Value) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    let resp = tower::ServiceExt::oneshot(router.clone(), req)
        .await
        .expect("oneshot");
    let status = resp.status().as_u16();
    let bytes = http_body_util::BodyExt::collect(resp.into_body())
        .await
        .expect("body")
        .to_bytes();
    let v = serde_json::from_slice(&bytes).unwrap_or(serde_json::json!(null));
    (status, v)
}

#[tokio::test]
async fn tree_endpoints_map_statuses_by_variant() {
    // Default MockGenerator refuses tree ops → 501 at session-open.
    let (mut router, _) = mock_router(vec![1], FinishReason::Stop);
    let (st, _v) = post_json(
        &mut router,
        "/v1/tree/session",
        serde_json::json!({"prompt_tokens": [1, 2]}),
    )
    .await;
    assert_eq!(st, 501, "default generator must refuse with 501");

    // Session-capable mock: 200 open, 200 verify, 409 after a chat completion.
    let tree = TreeMock {
        session_open: false,
        pending: 42,
        committed: vec![43, 44],
        verify_error: None,
    };
    let (mut router, _) = build_router(Box::new(tree), shared_tok(), ServeConfig::default());
    let (st, v) = post_json(
        &mut router,
        "/v1/tree/session",
        serde_json::json!({"prompt_tokens": [1, 2]}),
    )
    .await;
    assert_eq!((st, v["pending_token"].as_u64()), (200, Some(42)));
    let (st, v) = post_json(
        &mut router,
        "/v1/tree/verify",
        serde_json::json!({"tokens": [42, 43], "parents": [-1, 0]}),
    )
    .await;
    assert_eq!(st, 200);
    assert_eq!(v["committed"], serde_json::json!([43, 44]));

    // 409 without a session (fresh router → session never opened).
    let tree = TreeMock {
        session_open: false,
        pending: 42,
        committed: vec![],
        verify_error: None,
    };
    let (mut router, _) = build_router(Box::new(tree), shared_tok(), ServeConfig::default());
    let (st, _v) = post_json(
        &mut router,
        "/v1/tree/verify",
        serde_json::json!({"tokens": [1], "parents": [-1]}),
    )
    .await;
    assert_eq!(st, 409);

    // 400 (BadRequest) and 500 (Internal) map by variant, never by string.
    for (err, want) in [
        (
            tritium_serve::TreeOpError::BadRequest("parents[1]=5 is not topological".into()),
            400u16,
        ),
        // The trap the string-sniffing version fell into: an INTERNAL error
        // whose message contains "not supported" must stay a 500, not 501.
        (
            tritium_serve::TreeOpError::Internal("driver: operation not supported".into()),
            500,
        ),
    ] {
        let tree = TreeMock {
            session_open: false,
            pending: 42,
            committed: vec![],
            verify_error: Some(err),
        };
        let (mut router, _) = build_router(Box::new(tree), shared_tok(), ServeConfig::default());
        let (st, _v) = post_json(
            &mut router,
            "/v1/tree/session",
            serde_json::json!({"prompt_tokens": [1]}),
        )
        .await;
        assert_eq!(st, 200);
        let (st, _v) = post_json(
            &mut router,
            "/v1/tree/verify",
            serde_json::json!({"tokens": [1], "parents": [-1]}),
        )
        .await;
        assert_eq!(st, want);
    }
}

/// Bearer auth: with `auth_token` set, requests without (or with a wrong)
/// token are 401; the right token passes. (P1 network-exposure hardening.)
#[tokio::test]
async fn bearer_auth_enforced_when_configured() {
    let (router, _d) = router_with(
        MockGenerator::new(vec![10, 11]),
        ServeConfig {
            auth_token: Some("sekrit".into()),
            ..ServeConfig::default()
        },
    );
    // No token → 401.
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({"model":"tritium","messages":[{"role":"user","content":"1"}]})
                .to_string(),
        ))
        .unwrap();
    let (status, _) = send(&router, req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    // Wrong token → 401.
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer nope")
        .body(Body::from(
            serde_json::json!({"model":"tritium","messages":[{"role":"user","content":"1"}]})
                .to_string(),
        ))
        .unwrap();
    let (status, _) = send(&router, req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    // Right token → 200.
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer sekrit")
        .body(Body::from(
            serde_json::json!({"model":"tritium","messages":[{"role":"user","content":"1"}]})
                .to_string(),
        ))
        .unwrap();
    let (status, _) = send(&router, req).await;
    assert_eq!(status, StatusCode::OK);
    // Health is also behind auth when configured (uniform surface).
    let req = Request::get("/healthz").body(Body::empty()).unwrap();
    let (status, _) = send(&router, req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let req = Request::get("/readyz").body(Body::empty()).unwrap();
    let (status, _) = send(&router, req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

/// Rotating keys are separate bounded principals: exhausting one bucket does
/// not affect another, probe routes do not consume generation credit, and a
/// rejection carries stable OpenAI/Retry-After/metric semantics.
#[tokio::test]
async fn rotating_auth_has_per_principal_rate_buckets() {
    let (router, _d) = build_router_governed(
        Box::new(MockGenerator::new(vec![10])),
        shared_tok(),
        ServeConfig::default(),
        RequestLimits::default(),
        AdmissionPolicy {
            bearer_tokens: vec!["old-key".into(), "new-key".into()],
            rate_limit: Some(PrincipalRateLimit {
                requests_per_minute: 1,
                burst: 1,
            }),
        },
    )
    .expect("valid governed router");

    let request = |token: &'static str| {
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {token}"))
            .body(Body::from(
                json!({"model":"tritium","messages":[{"role":"user","content":"1"}]}).to_string(),
            ))
            .unwrap()
    };
    assert_eq!(send(&router, request("old-key")).await.0, StatusCode::OK);

    let response = router.clone().oneshot(request("old-key")).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["retry-after"], "60");
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["error"]["type"], "rate_limit_exceeded");

    // Probe access is authenticated but never charged against generation.
    let health = Request::get("/healthz")
        .header("authorization", "Bearer old-key")
        .body(Body::empty())
        .unwrap();
    assert_eq!(send(&router, health).await.0, StatusCode::OK);

    // The rotating replacement key has its own fixed bucket.
    assert_eq!(send(&router, request("new-key")).await.0, StatusCode::OK);
    let metrics = Request::get("/metrics")
        .header("authorization", "Bearer new-key")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(&router, metrics).await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(body).unwrap();
    assert!(text.contains("tritium_rate_rejections_total 1\n"), "{text}");
}

/// Body limit: an over-2MiB request body is rejected, not buffered.
#[tokio::test]
async fn oversized_body_rejected() {
    let (router, _d) = router_with(MockGenerator::new(vec![10]), ServeConfig::default());
    let big = "9 ".repeat(2 * 1024 * 1024); // > 2 MiB of token text
    let req = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(format!(
            r#"{{"model":"tritium","messages":[{{"role":"user","content":"{big}"}}]}}"#
        )))
        .unwrap();
    let (status, _) = send(&router, req).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
}

/// Non-streaming requests are bounded by the request timeout: the handler
/// awaits the full aggregation, so a generation slower than the deadline
/// surfaces as 408.
#[tokio::test]
async fn nonstream_timeout_408() {
    let mock = MockGenerator {
        step_delay_ms: 400, // 4 tokens x 400ms = 1.6s > the 1s deadline
        ..MockGenerator::new(vec![1, 2, 3, 4])
    };
    let (router, _d) = router_with(
        mock,
        ServeConfig {
            request_timeout_secs: 1,
            ..ServeConfig::default()
        },
    );
    let (status, body) = send(
        &router,
        chat(json!({"model":"tritium","messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::REQUEST_TIMEOUT,
        "slow non-streaming -> 408"
    );
    let body: Value = serde_json::from_slice(&body).expect("OpenAI timeout envelope");
    assert_eq!(body["error"]["type"], "request_timeout_error");
    assert_eq!(body["error"]["code"], "request_timeout");
}

/// Streaming has an absolute deadline inside the lazy SSE body. Expiry emits
/// a typed OpenAI error event, terminates framing, and drops the receiver so
/// the worker observes cancellation.
#[tokio::test]
async fn sse_deadline_cancels_generation() {
    let mock = MockGenerator {
        step_delay_ms: 400, // total 1.6s of generation vs a 1s deadline
        ..MockGenerator::new(vec![1, 2, 3, 4])
    };
    let (router, _d) = router_with(
        mock,
        ServeConfig {
            request_timeout_secs: 1,
            ..ServeConfig::default()
        },
    );
    let t0 = std::time::Instant::now();
    let (status, body) = send(
        &router,
        chat(json!({"model":"tritium","stream":true,"messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let events = parse_sse(&body);
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let chunks = sse_chunks(&events);
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let error = chunks.last().expect("timeout error event");
    assert_eq!(error["error"]["type"], "request_timeout_error");
    assert_eq!(error["error"]["code"], "request_timeout");
    assert!(
        t0.elapsed() >= Duration::from_millis(900),
        "deadline fired too early: {:?}",
        t0.elapsed()
    );
    assert!(
        t0.elapsed() < Duration::from_millis(1500),
        "stream outlived the configured deadline: {:?}",
        t0.elapsed()
    );

    let req = Request::get("/metrics").body(Body::empty()).unwrap();
    let (status, body) = send(&router, req).await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(body).unwrap();
    assert!(text.contains("tritium_stream_timeouts_total 1\n"), "{text}");
}

/// Chat-template rendering: the RoleEot template must reproduce the official
/// transformers template ("{Role}: {content}<|eot_id|>" per message + the
/// "Assistant: " generation prompt); Concat stays the id-passthrough join.
#[test]
fn chat_template_render() {
    use tritium_serve::ChatTemplate;
    let msgs = [
        ("system", "Be terse."),
        ("user", " What is 2+2? "),
        ("assistant", "4"),
        ("user", "And 3+3?"),
    ];
    let rendered = ChatTemplate::RoleEot.render(msgs.iter().map(|&(r, c)| (r, c)));
    assert_eq!(
        rendered,
        "System: Be terse.<|eot_id|>User: What is 2+2?<|eot_id|>\
         Assistant: 4<|eot_id|>User: And 3+3?<|eot_id|>Assistant: "
    );
    let concat = ChatTemplate::Concat.render(msgs.iter().map(|&(r, c)| (r, c)));
    assert_eq!(concat, "Be terse.\n What is 2+2? \n4\nAnd 3+3?");
}

/// /metrics: Prometheus text exposition — counters move with traffic, the
/// queue gauge and worker liveness render, and the endpoint sits behind the
/// same auth as everything else (uniform surface).
#[tokio::test]
async fn metrics_exposition() {
    let (router, _d) = mock_router(vec![10, 11, 12], FinishReason::Stop);
    // Two requests -> 2 accepted, 6 tokens.
    for _ in 0..2 {
        let (status, _) = send(
            &router,
            chat(json!({"model":"tritium","messages":[{"role":"user","content":"1"}]})),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }
    let req = Request::get("/metrics").body(Body::empty()).unwrap();
    let (status, body) = send(&router, req).await;
    assert_eq!(status, StatusCode::OK);
    let text = String::from_utf8(body).unwrap();
    assert!(text.contains("tritium_chat_requests_total 2\n"), "{text}");
    assert!(text.contains("tritium_tokens_out_total 6\n"), "{text}");
    assert!(
        text.contains("tritium_queue_rejections_total 0\n"),
        "{text}"
    );
    assert!(text.contains("tritium_rate_rejections_total 0\n"), "{text}");
    assert!(text.contains("tritium_stream_timeouts_total 0\n"), "{text}");
    assert!(text.contains("tritium_worker_alive 1\n"), "{text}");
    assert!(text.contains("# TYPE tritium_queue_depth gauge"), "{text}");

    // Auth uniformity: with a token configured, /metrics 401s like the rest.
    let (router, _d) = router_with(
        MockGenerator::new(vec![1]),
        ServeConfig {
            auth_token: Some("sekrit".into()),
            ..ServeConfig::default()
        },
    );
    let req = Request::get("/metrics").body(Body::empty()).unwrap();
    let (status, _) = send(&router, req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

/// stream_options.include_usage: a final pre-[DONE] chunk with empty choices
/// and usage matching the non-streaming accounting; absent when not asked.
#[tokio::test]
async fn stream_usage_chunk() {
    let (router, _) = mock_router(vec![10, 11, 12], FinishReason::Stop);
    let (_, body) = send(
        &router,
        chat(json!({"model":"tritium","stream":true,
                    "stream_options":{"include_usage":true},
                    "messages":[{"role":"user","content":"1 2"}]})),
    )
    .await;
    let events = parse_sse(&body);
    let chunks = sse_chunks(&events);
    let last = chunks.last().unwrap();
    assert!(last["choices"].as_array().unwrap().is_empty(), "{last}");
    assert_eq!(last["usage"]["prompt_tokens"], 2);
    assert_eq!(last["usage"]["completion_tokens"], 3);
    assert_eq!(last["usage"]["total_tokens"], 5);
    // Every earlier chunk omits usage entirely.
    assert!(
        chunks[..chunks.len() - 1]
            .iter()
            .all(|c| c["usage"].is_null())
    );

    // Without stream_options: no usage chunk, terminal chunk is last.
    let (router, _) = mock_router(vec![10, 11, 12], FinishReason::Stop);
    let (_, body) = send(
        &router,
        chat(json!({"model":"tritium","stream":true,"messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    let chunks = sse_chunks(&parse_sse(&body));
    assert!(chunks.iter().all(|c| c["usage"].is_null()));
    assert!(!chunks.last().unwrap()["choices"][0]["finish_reason"].is_null());
}

/// logprobs: OpenAI shape on both paths — sampled token's record per
/// completion token with top-k alternatives; absent when not requested;
/// top_logprobs without logprobs is a 400.
#[tokio::test]
async fn logprobs_shapes() {
    // Non-stream: 3 tokens, k=2 alternatives each (mock synthesizes -0.1 /
    // -1.0 / -2.0).
    let (router, _) = mock_router(vec![10, 11, 12], FinishReason::Stop);
    let (status, body) = send(
        &router,
        chat(json!({"model":"tritium","logprobs":true,"top_logprobs":2,
                    "messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: Value = serde_json::from_slice(&body).unwrap();
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 3);
    assert_eq!(content[0]["token"], "10");
    assert!((content[0]["logprob"].as_f64().unwrap() + 0.1).abs() < 1e-6);
    assert_eq!(content[0]["top_logprobs"].as_array().unwrap().len(), 2);
    assert_eq!(content[0]["bytes"], json!([49, 48])); // b"10"

    // Stream: each content chunk carries its token's record.
    let (router, _) = mock_router(vec![10, 11], FinishReason::Stop);
    let (_, body) = send(
        &router,
        chat(
            json!({"model":"tritium","stream":true,"logprobs":true,"top_logprobs":1,
                    "messages":[{"role":"user","content":"1"}]}),
        ),
    )
    .await;
    let chunks = sse_chunks(&parse_sse(&body));
    let with_lp: Vec<_> = chunks
        .iter()
        .filter(|c| !c["choices"][0]["logprobs"].is_null())
        .collect();
    assert_eq!(with_lp.len(), 2, "one logprobs record per content chunk");
    assert_eq!(
        with_lp[0]["choices"][0]["logprobs"]["content"][0]["token"],
        "10"
    );

    // Not requested -> absent entirely.
    let (router, _) = mock_router(vec![10], FinishReason::Stop);
    let (_, body) = send(
        &router,
        chat(json!({"model":"tritium","messages":[{"role":"user","content":"1"}]})),
    )
    .await;
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert!(v["choices"][0]["logprobs"].is_null());

    // top_logprobs without logprobs -> 400.
    let (router, _) = mock_router(vec![10], FinishReason::Stop);
    let (status, _) = send(
        &router,
        chat(
            json!({"model":"tritium","top_logprobs":3,"messages":[{"role":"user","content":"1"}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// Review finding: streamed logprob rows must never be dropped — with a stop
/// string active (StopMatcher holds text back) the row count must still
/// equal the non-streaming path's for the same generation.
#[tokio::test]
async fn stream_logprobs_survive_stop_holdback() {
    let req = json!({"model":"tritium","logprobs":true,"top_logprobs":1,
                     "stop":"11","messages":[{"role":"user","content":"1"}]});
    // Non-stream reference: rows for tokens up to the stop hit.
    let (router, _) = mock_router(vec![10, 11, 12], FinishReason::Stop);
    let (_, body) = send(&router, chat(req.clone())).await;
    let v: Value = serde_json::from_slice(&body).unwrap();
    let want_rows = v["choices"][0]["logprobs"]["content"]
        .as_array()
        .unwrap()
        .len();

    // Stream: sum rows across all chunks (incl. any empty-content carrier).
    let mut sreq = req;
    sreq["stream"] = json!(true);
    let (router, _) = mock_router(vec![10, 11, 12], FinishReason::Stop);
    let (_, body) = send(&router, chat(sreq)).await;
    let got_rows: usize = sse_chunks(&parse_sse(&body))
        .iter()
        .filter_map(|c| c["choices"][0]["logprobs"]["content"].as_array())
        .map(Vec::len)
        .sum();
    assert_eq!(
        got_rows, want_rows,
        "streamed logprob rows must match non-streamed for the same request"
    );
    assert!(
        want_rows >= 2,
        "sanity: stop at second token yields >= 2 rows"
    );
}
