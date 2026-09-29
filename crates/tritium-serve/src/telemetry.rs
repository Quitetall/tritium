//! Optional OpenTelemetry trace export for the server binary.

use std::time::Duration;

use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::SpanExporter;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{Sampler, SdkTracerProvider};
use tracing_subscriber::layer::SubscriberExt as _;

const FLUSH_TIMEOUT: Duration = Duration::from_secs(5);

/// Owns the provider so shutdown flush is bounded and occurs on every exit path.
pub(crate) struct TelemetryGuard {
    provider: SdkTracerProvider,
}

impl Drop for TelemetryGuard {
    fn drop(&mut self) {
        if let Err(error) = self.provider.shutdown_with_timeout(FLUSH_TIMEOUT) {
            eprintln!("tritium-serve: OpenTelemetry shutdown/flush failed: {error}");
        }
    }
}

/// Initialize OTLP HTTP tracing when a standard endpoint variable opts it in.
///
/// The SDK uses a parent-based always-on sampler: roots are sampled and valid
/// incoming W3C sampling decisions are preserved for all child spans.
pub(crate) fn initialize() -> Result<Option<TelemetryGuard>, Box<dyn std::error::Error>> {
    let configured = [
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    ]
    .into_iter()
    .any(|name| std::env::var_os(name).is_some_and(|value| !value.is_empty()));
    if !configured {
        return Ok(None);
    }

    let exporter = SpanExporter::builder().with_http().build()?;
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_sampler(Sampler::ParentBased(Box::new(Sampler::AlwaysOn)))
        .with_resource(
            Resource::builder()
                .with_service_name("tritium-serve")
                .build(),
        )
        .build();
    global::set_text_map_propagator(TraceContextPropagator::new());
    let tracer = provider.tracer("tritium-serve");
    let subscriber =
        tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|error| format!("install OpenTelemetry tracing subscriber: {error}"))?;
    global::set_tracer_provider(provider.clone());

    Ok(Some(TelemetryGuard { provider }))
}
