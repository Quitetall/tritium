//! Real tiny standard/hybrid graphs through public load, generator and HTTP/SSE.
#![cfg(feature = "serve")]
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tower::ServiceExt;
use tritium_evidence::{EvidenceRecorder, MODEL_LOADED_REGISTRATION, TENSOR_UPLOADED_REGISTRATION};
use tritium_nn::Tokenizer;
use tritium_schema::*;
use tritium_serve::{
    AdmissionPolicy, FinishReason, GenRequest, Generator, RequestLimits, RunnerGenerator, Sampling,
    ServeConfig, Step, TreeOpError, build_router_governed,
};

#[path = "portable_fixture/mod.rs"]
mod fixture;
use fixture::package;
fn loaded(bytes: &[u8], budget: &tritium_runtime::TensorPayloadBudget) -> tritium_nn::LoadedModel {
    let mut evidence = EvidenceRecorder::new(
        "serve-public-fixture",
        [TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION],
    )
    .unwrap();
    tritium_nn::load(
        tritium_nn::ModelSource::Trit {
            bytes,
            level: "test",
        },
        Box::new(tritium_cpu::CpuBackend::new()),
        TensorUploadPolicy::default(),
        budget,
        tritium_runtime::TensorUploadContext {
            recorder: &mut evidence,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap()
}
fn request(sampling: Sampling) -> GenRequest {
    GenRequest {
        prompt_tokens: vec![1, 2],
        max_new: 4,
        sampling,
        stop_eos: false,
        logprobs: Some(2),
    }
}
fn collect(generator: &mut dyn Generator, req: &GenRequest) -> Vec<Step> {
    let mut result = Vec::new();
    generator
        .generate(req, &mut |step| {
            result.push(step);
            true
        })
        .unwrap();
    result
}
#[test]
fn common_generators_stream_real_standard_hybrid_and_reset_cancelled_requests() {
    for hybrid in [false, true] {
        for additive in [false, true] {
            let bytes = package(hybrid, additive);
            let budget = tritium_runtime::TensorPayloadBudget::new(1_000_000);
            let model = loaded(&bytes, &budget);
            let mut generator = RunnerGenerator::from_runner(model.runner, model.tokenizer.eos());
            assert_eq!((generator.n_ctx(), generator.vocab()), (32, 7));
            assert_eq!(
                generator.cancellation_capabilities().tree_verify,
                tritium_serve::CancellationCheckpoint::NotEnabled
            );
            assert!(matches!(
                generator.open_tree_session(&[1]),
                Err(TreeOpError::Unsupported(_))
            ));
            assert!(
                generator
                    .tree_verify_cancellable(&[], &[], &|| true)
                    .unwrap()
                    .is_none()
            );
            for sampling in [
                Sampling::Greedy,
                Sampling::TopK {
                    k: 3,
                    temp: 0.8,
                    seed: 7,
                },
                Sampling::TopP {
                    p: 0.9,
                    temp: 0.8,
                    seed: 7,
                },
            ] {
                let req = request(sampling);
                let first = collect(&mut generator, &req);
                let again = collect(&mut generator, &req);
                assert_eq!(
                    first.iter().map(|s| s.token).collect::<Vec<_>>(),
                    again.iter().map(|s| s.token).collect::<Vec<_>>()
                );
                assert_eq!(first.len(), 4);
                assert_eq!(
                    first.last().unwrap().finish_reason,
                    Some(FinishReason::Length)
                );
                assert!(first.iter().all(|s| {
                    s.logprobs
                        .as_ref()
                        .unwrap()
                        .iter()
                        .all(|(_, p)| p.is_finite())
                }));
                let cancel = AtomicBool::new(false);
                let mut steps = 0;
                generator
                    .generate_cancellable(
                        &req,
                        &mut |_| {
                            steps += 1;
                            cancel.store(true, Ordering::SeqCst);
                            true
                        },
                        &|| cancel.load(Ordering::SeqCst),
                    )
                    .unwrap();
                assert_eq!(steps, 1);
                generator.generate(&req, &mut |_| false).unwrap();
                let after = collect(&mut generator, &req);
                assert_eq!(
                    first.iter().map(|s| s.token).collect::<Vec<_>>(),
                    after.iter().map(|s| s.token).collect::<Vec<_>>()
                );
            }
            let mut invalid = request(Sampling::Greedy);
            invalid.prompt_tokens = vec![7];
            assert!(
                generator
                    .generate(&invalid, &mut |_| panic!("invalid prompt published"))
                    .is_err()
            );
            generator
                .generate_cancellable(
                    &invalid,
                    &mut |_| panic!("cancelled prompt published"),
                    &|| true,
                )
                .unwrap();
            let mut overflow = request(Sampling::Greedy);
            overflow.prompt_tokens = vec![1; 33];
            assert!(
                generator
                    .generate(&overflow, &mut |_| panic!("overflow published"))
                    .is_err()
            );
            let mut limited = request(Sampling::Greedy);
            limited.prompt_tokens = vec![1; 31];
            let steps = collect(&mut generator, &limited);
            assert_eq!(steps.len(), 1);
            assert_eq!(steps[0].finish_reason, Some(FinishReason::Length));
            limited.prompt_tokens.push(1);
            assert!(collect(&mut generator, &limited).is_empty());
            limited.prompt_tokens.clear();
            assert!(
                generator
                    .generate(&limited, &mut |_| panic!("empty prompt published"))
                    .is_err()
            );
            let mut eos_model = loaded(&bytes, &budget);
            let eos =
                tritium_nn::sample_greedy(&eos_model.runner.prefill(&[1, 2]).unwrap()).unwrap();
            let mut eos_generator = RunnerGenerator::from_runner(eos_model.runner, eos);
            let mut stopping = request(Sampling::Greedy);
            stopping.stop_eos = true;
            let stopped = collect(&mut eos_generator, &stopping);
            assert_eq!(stopped.len(), 1);
            assert_eq!(stopped[0].token, eos);
            assert_eq!(stopped[0].finish_reason, Some(FinishReason::Stop));
            drop(eos_generator);
            drop(generator);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn common_runner_cannot_silently_accept_legacy_model_drafting() {
    let bytes = package(false, true);
    let budget = tritium_runtime::TensorPayloadBudget::new(1_000_000);
    let model = loaded(&bytes, &budget);
    let mut evidence =
        EvidenceRecorder::new("legacy-draft-fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let draft = tritium_nn::ModelRunner::from_source_admitted(
        tritium_nn::ModelSource::Trit {
            bytes: &bytes,
            level: "test",
        },
        Box::new(tritium_cpu::CpuBackend::new()),
        TensorUploadPolicy::default(),
        &budget,
        tritium_runtime::TensorUploadContext {
            recorder: &mut evidence,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    let mut generator = RunnerGenerator::from_runner(model.runner, 3).with_draft_model(draft);
    let error = generator
        .generate(&request(Sampling::Greedy), &mut |_| {
            panic!("unavailable draft published")
        })
        .unwrap_err();
    assert!(error.to_string().contains("model draft is not enabled"));
    drop(generator);
    assert_eq!(budget.used_bytes(), 0);
}

#[tokio::test]
async fn common_loaded_graphs_serve_buffered_and_sse_without_production_authority() {
    for hybrid in [false, true] {
        for additive in [false, true] {
            let bytes = package(hybrid, additive);
            let budget = tritium_runtime::TensorPayloadBudget::new(1_000_000);
            let model = loaded(&bytes, &budget);
            let eos = model.tokenizer.eos();
            let mut expected_generator =
                RunnerGenerator::from_runner(loaded(&bytes, &budget).runner, eos);
            let mut req = request(Sampling::Greedy);
            req.stop_eos = true;
            req.logprobs = None;
            req.max_new = 3;
            let expected = collect(&mut expected_generator, &req)
                .iter()
                .map(|s| s.token)
                .filter(|&t| t != eos)
                .collect::<Vec<_>>();
            let expected_text = model.tokenizer.decode(&expected).unwrap();
            drop(expected_generator);
            let (router, draining) = build_router_governed(
                Box::new(RunnerGenerator::from_runner(model.runner, eos)),
                Arc::new(model.tokenizer),
                ServeConfig::default(),
                RequestLimits::default(),
                AdmissionPolicy::default(),
            )
            .unwrap();
            let health = router
                .clone()
                .oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
                .await
                .unwrap();
            let health: Value =
                serde_json::from_slice(&health.into_body().collect().await.unwrap().to_bytes())
                    .unwrap();
            assert_eq!(health["production_artifact"], false);
            assert!(health["startup_receipt"].is_null());
            for stream in [false, true] {
                let response=router.clone().oneshot(Request::post("/v1/chat/completions").header("content-type","application/json")
                .body(Body::from(json!({"model":"tritium","messages":[{"role":"user","content":"hello partner"}],
                    "max_tokens":3,"temperature":0,"stream":stream}).to_string())).unwrap()).await.unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let body = response.into_body().collect().await.unwrap().to_bytes();
                if stream {
                    let text = std::str::from_utf8(&body).unwrap();
                    assert!(text.contains("data: [DONE]"));
                    let content = text
                        .lines()
                        .filter_map(|line| line.strip_prefix("data: "))
                        .filter(|line| *line != "[DONE]")
                        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                        .filter_map(|v| {
                            v["choices"][0]["delta"]["content"]
                                .as_str()
                                .map(str::to_owned)
                        })
                        .collect::<String>();
                    assert_eq!(content, expected_text);
                } else {
                    let value: Value = serde_json::from_slice(&body).unwrap();
                    assert_eq!(value["choices"][0]["message"]["content"], expected_text);
                }
            }
            draining.store(true, Ordering::Release);
            drop(router);
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                while budget.used_bytes() != 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("worker shutdown releases loaded graph payloads");
        }
    }
}
