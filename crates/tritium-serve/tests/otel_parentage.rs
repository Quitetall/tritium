#![cfg(feature = "serve")]

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt as _;
use tritium_nn::Tokenizer;
use tritium_serve::{IdPassthroughTokenizer, MockGenerator, ServeConfig, build_router};

#[derive(Clone, Debug, Default)]
struct CollectSpans(Arc<Mutex<Vec<SpanData>>>);

impl SpanExporter for CollectSpans {
    async fn export(&self, mut batch: Vec<SpanData>) -> OTelSdkResult {
        self.0.lock().unwrap().append(&mut batch);
        Ok(())
    }
}

#[test]
fn request_span_exports_with_the_remote_w3c_parent() {
    let exporter = CollectSpans::default();
    let provider = SdkTracerProvider::builder()
        .with_simple_exporter(exporter.clone())
        .build();
    let subscriber = tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("tritium-serve-test")));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let tok: Arc<dyn Tokenizer + Send + Sync> = Arc::new(IdPassthroughTokenizer::default());
    tracing::subscriber::set_global_default(subscriber)
        .expect("install test tracing subscriber before serving");

    let response_context = runtime.block_on(async {
        let (router, _) = build_router(
            Box::new(MockGenerator::new(vec![1])),
            tok,
            ServeConfig::default(),
        );
        let response = router
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header(
                        "traceparent",
                        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                    )
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"tritium","messages":[{"role":"user","content":"1"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response_context = response
            .headers()
            .get("traceparent")
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();
        let _body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        response_context
    });
    provider.force_flush().unwrap();
    let spans = exporter.0.lock().unwrap();
    let request_span = spans
        .iter()
        .find(|span| span.name == "http.server.request")
        .unwrap();
    assert_eq!(
        request_span.span_context.trace_id().to_string(),
        "4bf92f3577b34da6a3ce929d0e0e4736"
    );
    assert_eq!(request_span.parent_span_id.to_string(), "00f067aa0ba902b7");
    assert!(request_span.parent_span_is_remote);
    for name in ["model.queue", "model.prefill", "model.decode"] {
        let span = spans
            .iter()
            .find(|span| span.name == name)
            .unwrap_or_else(|| {
                panic!(
                    "missing {name}; exported spans: {:?}",
                    spans
                        .iter()
                        .map(|span| span.name.to_string())
                        .collect::<Vec<_>>()
                )
            });
        assert_eq!(
            span.parent_span_id,
            request_span.span_context.span_id(),
            "{name} should be a child of the HTTP request"
        );
    }
    assert_eq!(
        &response_context[36..52],
        request_span.span_context.span_id().to_string()
    );
}
