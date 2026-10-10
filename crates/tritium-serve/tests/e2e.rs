//! Real-model end-to-end smoke test (manual, gated). Mirrors tritium-nn's
//! acceptance gating: compile with `--features e2e` AND set `TRITIUM_SERVE_E2E=1`
//! + `TRITIUM_MODEL_PATH=<gguf>` or `TRITIUM_CONVERTED_PATH=<directory>`.
//! Explicit selection without opt-in fails; default CI leaves these ignored.
#![cfg(feature = "e2e")]

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;
use tritium_nn::Tokenizer;
use tritium_serve::{IdPassthroughTokenizer, RunnerGenerator, ServeConfig, build_router};

// Register the CPU backend that `ModelRunner::load_cpu` resolves from the registry.
use tritium_cpu as _;

fn require_real_model_run() {
    assert_eq!(
        std::env::var("TRITIUM_SERVE_E2E").as_deref(),
        Ok("1"),
        "explicitly selected real-model tests require TRITIUM_SERVE_E2E=1; missing inputs are not a pass"
    );
}

#[tokio::test]
#[ignore = "real model: set TRITIUM_SERVE_E2E=1 + TRITIUM_MODEL_PATH=<gguf>"]
async fn serve_e2e_token_id_roundtrip() {
    require_real_model_run();
    let path =
        std::env::var("TRITIUM_MODEL_PATH").expect("TRITIUM_MODEL_PATH must point at a GGUF");
    let bytes = std::fs::read(&path).expect("read model");
    let runner = tritium_nn::ModelRunner::load_cpu(&bytes).expect("load cpu runner");
    let eos = 128_001;
    let generator = Box::new(RunnerGenerator::new(runner, eos));
    let tok: Arc<dyn Tokenizer + Send + Sync> = Arc::new(IdPassthroughTokenizer::new(128_000, eos));
    let (router, _) = build_router(generator, tok, {
        let mut c = ServeConfig::default();
        c.max_new_default = 16;
        c
    });

    // A short token-ID prompt (the v0.80 passthrough tokenizer takes integer IDs).
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model":"tritium","max_tokens":8,
                   "messages":[{"role":"user","content":"1 2 3 4"}]})
            .to_string(),
        ))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert!(v["usage"]["completion_tokens"].as_u64().unwrap() >= 1);
}

#[tokio::test]
#[ignore = "real converted model: set TRITIUM_SERVE_E2E=1 + TRITIUM_CONVERTED_PATH=<directory>"]
async fn serve_e2e_converted_directory() {
    require_real_model_run();
    let dir = std::env::var("TRITIUM_CONVERTED_PATH")
        .expect("TRITIUM_CONVERTED_PATH must point at a tritium convert directory");
    let dir = std::path::Path::new(&dir);
    let bundle = dir.join("model.tslb");
    let runner =
        tritium_nn::ModelRunner::from_salt(dir, &bundle, Box::new(tritium_cpu::CpuBackend::new()))
            .expect("load converted model");
    let tokenizer = tritium_nn::HfJsonTokenizer::from_files(
        &dir.join("tokenizer.json"),
        &dir.join("tokenizer_config.json"),
    )
    .expect("load converted tokenizer");
    let eos = tritium_nn::Tokenizer::eos(&tokenizer);
    let generator = Box::new(RunnerGenerator::new(runner, eos));
    let tok: Arc<dyn Tokenizer + Send + Sync> = Arc::new(tokenizer);
    let (router, _) = build_router(generator, tok, {
        let mut c = ServeConfig::default();
        c.max_new_default = 8;
        c
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model":"tritium","max_tokens":4,
                   "messages":[{"role":"user","content":"The capital of France is"}]})
            .to_string(),
        ))
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert!(v["usage"]["completion_tokens"].as_u64().unwrap() >= 1);
}

fn converted_router() -> Router {
    let dir = std::env::var("TRITIUM_CONVERTED_PATH")
        .expect("TRITIUM_CONVERTED_PATH must point at a tritium convert directory");
    let dir = std::path::Path::new(&dir);
    let runner = tritium_nn::ModelRunner::from_salt(
        dir,
        &dir.join("model.tslb"),
        Box::new(tritium_cpu::CpuBackend::new()),
    )
    .expect("load actual converted CPU model");
    let tokenizer = tritium_nn::HfJsonTokenizer::from_files(
        &dir.join("tokenizer.json"),
        &dir.join("tokenizer_config.json"),
    )
    .expect("load actual converted tokenizer");
    let generator = RunnerGenerator::new(runner, Tokenizer::eos(&tokenizer));
    build_router(
        Box::new(generator),
        Arc::new(tokenizer),
        ServeConfig::default(),
    )
    .0
}

fn converted_request(stream: bool, max_tokens: usize) -> Request<Body> {
    let mut request = json!({
        "model": "tritium", "temperature": 0, "max_tokens": max_tokens,
        "stream": stream,
        "messages": [{"role": "user", "content": "The capital of France is"}]
    });
    if stream {
        request["stream_options"] = json!({"include_usage": true});
    }
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(request.to_string()))
        .unwrap()
}

async fn converted_completion(router: &Router) -> Value {
    tokio::time::timeout(Duration::from_secs(60), async {
        let response = router
            .clone()
            .oneshot(converted_request(false, 4))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes()).unwrap()
    })
    .await
    .expect("bounded native completion")
}

async fn converted_metrics(router: &Router) -> String {
    let response = router
        .clone()
        .oneshot(Request::get("/metrics").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap()
}

async fn wait_native_phase(router: &Router, phase: &str) -> String {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let metrics = converted_metrics(router).await;
            if metrics.contains(&format!("tritium_worker_phase{{phase=\"{phase}\"}} 1\n")) {
                return metrics;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("native worker did not reach {phase} within execution bound"))
}

/// Actual SALT weights + HF tokenizer + native CPU forwards, not MockGenerator.
/// This exercises the legacy compatibility router in-process; it does not
/// qualify schema-v3 startup, socket transport, paged KV, quality or performance.
#[tokio::test]
#[ignore = "real converted model: set TRITIUM_SERVE_E2E=1 + TRITIUM_CONVERTED_PATH=<directory>"]
async fn serve_e2e_converted_stream_disconnect_recovery() {
    require_real_model_run();
    let router = converted_router();
    let reference = converted_completion(&router).await;
    assert!(reference["usage"]["completion_tokens"].as_u64().unwrap() > 0);

    let response = router
        .clone()
        .oneshot(converted_request(true, 4))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let bytes = tokio::time::timeout(Duration::from_secs(60), response.into_body().collect())
        .await
        .expect("bounded actual SSE completion")
        .unwrap()
        .to_bytes();
    let text = std::str::from_utf8(&bytes).unwrap();
    let events: Vec<_> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect();
    assert_eq!(events.last().copied(), Some("[DONE]"));
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|event| serde_json::from_str(event).unwrap())
        .collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let streamed: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(
        streamed,
        reference["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
    );
    assert_eq!(chunks.last().unwrap()["usage"], reference["usage"]);
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["choices"][0]["finish_reason"]
                == reference["choices"][0]["finish_reason"])
    );

    for phase in ["prefill", "decode"] {
        let response = router
            .clone()
            .oneshot(converted_request(true, 256))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body();
        if phase == "decode" {
            tokio::time::timeout(Duration::from_secs(30), async {
                loop {
                    let frame = body
                        .frame()
                        .await
                        .expect("SSE stays open until actual content")
                        .expect("valid native SSE frame");
                    if let Ok(data) = frame.into_data()
                        && std::str::from_utf8(&data).unwrap().contains("\"content\"")
                    {
                        break;
                    }
                }
            })
            .await
            .expect("actual native content before decode disconnect");
        }
        // Observe real worker phase before dropping, not merely queue entry.
        // This gauge does not pinpoint an individual native checkpoint.
        wait_native_phase(&router, phase).await;
        let start = Instant::now();
        drop(body);
        let metrics = wait_native_phase(&router, "idle").await;
        let idle_elapsed = start.elapsed();
        assert!(metrics.contains("tritium_queue_depth 0\n"));
        assert!(metrics.contains("tritium_worker_alive 1\n"));
        assert!(metrics.contains("tritium_backend_faults_total 0\n"));
        let recovery_start = Instant::now();
        let recovered = converted_completion(&router).await;
        assert_eq!(
            recovered["choices"], reference["choices"],
            "{phase} recovery output"
        );
        assert_eq!(
            recovered["usage"], reference["usage"],
            "{phase} recovery accounting"
        );
        println!(
            "native converted {phase} disconnect: idle in {idle_elapsed:?}, recovery in {:?}",
            recovery_start.elapsed()
        );
    }
    let metrics = wait_native_phase(&router, "idle").await;
    assert!(metrics.contains("tritium_stream_disconnects_total 2\n"));
    // Shut down the actual worker rather than leaving model retirement to
    // process exit. Sender ownership ends when the last router is dropped.
    drop(router);
}
