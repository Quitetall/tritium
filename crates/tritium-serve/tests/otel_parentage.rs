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

    let response_context = tracing::subscriber::with_default(subscriber, || {
        runtime.block_on(async {
            let (router, _) = build_router(
                Box::new(MockGenerator::new(Vec::new())),
                tok,
                ServeConfig::default(),
            );
            let response = router
                .oneshot(
                    Request::get("/healthz")
                        .header(
                            "traceparent",
                            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                        )
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            response
                .headers()
                .get("traceparent")
                .unwrap()
                .to_str()
                .unwrap()
                .to_owned()
        })
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
    assert_eq!(
        &response_context[36..52],
        request_span.span_context.span_id().to_string()
    );
}
