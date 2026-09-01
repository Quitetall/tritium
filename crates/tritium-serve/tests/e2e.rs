//! Real-model end-to-end smoke test (manual, gated). Mirrors tritium-nn's
//! acceptance gating: compile with `--features e2e` AND set `TRITIUM_SERVE_E2E=1`
//! + `TRITIUM_MODEL_PATH=<gguf>`. Never runs in default CI.
#![cfg(feature = "e2e")]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;
use tritium_nn::Tokenizer;
use tritium_serve::{IdPassthroughTokenizer, RunnerGenerator, ServeConfig, build_router};

// Register the CPU backend that `ModelRunner::load_cpu` resolves from the registry.
use tritium_cpu as _;

#[tokio::test]
#[ignore = "real model: set TRITIUM_SERVE_E2E=1 + TRITIUM_MODEL_PATH=<gguf>"]
async fn serve_e2e_token_id_roundtrip() {
    if std::env::var("TRITIUM_SERVE_E2E").as_deref() != Ok("1") {
        eprintln!("serve_e2e: TRITIUM_SERVE_E2E != 1 — skipping");
        return;
    }
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
