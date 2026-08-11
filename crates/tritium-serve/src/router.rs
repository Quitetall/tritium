//! The axum router: OpenAI `/v1/chat/completions` (non-stream + SSE), `/v1/models`,
//! `/healthz` liveness and `/readyz` traffic readiness, with backpressure and a
//! drain flag for graceful shutdown.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::mpsc;
use tritium_nn::Tokenizer;

use crate::admission::{Admission, AdmissionDecision, AdmissionPolicy};
use crate::dto::{
    ApiError, ChatChunk, ChatCompletion, ChatMessage, ChatRequest, Choice, ModelEntry, ModelList,
    StopField, Usage,
};
use crate::generator::TreeOpError;
use crate::generator::{FinishReason, GenRequest, Generator, Sampling};
use crate::sse::{
    IncrementalDetok, StopMatcher, content_chunk, error_chunk, role_chunk, terminal_chunk,
};
use crate::startup::{AdmittedGeneratorV1, ProductionReadiness, prepare_production_generator};
use crate::worker::{
    GenEvent, Job, PHASE_DECODE, PHASE_IDLE, PHASE_PREFILL, WorkerSignals, WorkerTelemetry,
    spawn_worker,
};

/// How `/v1/chat/completions` renders `messages` into the prompt string
/// handed to the tokenizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplate {
    /// Join message contents with newlines (the id-passthrough MVP: clients
    /// send integer token ids, roles carry no wire meaning).
    #[default]
    Concat,
    /// The official BitNet/LLaMA-3-family template the `transformers`
    /// reference uses: `{Role}: {content}<|eot_id|>` per message, then the
    /// `Assistant: ` generation prompt. BOS comes from the tokenizer's
    /// encode. Requires a real BPE tokenizer (the special tokens must map
    /// to their control ids).
    RoleEot,
    /// Qwen chat format: one `<|im_start|>role` block per message, terminated
    /// by `<|im_end|>`, followed by assistant generation prefix.
    QwenIm,
}

impl ChatTemplate {
    /// Render `(role, content)` messages into the prompt string.
    pub fn render<'a>(self, messages: impl Iterator<Item = (&'a str, &'a str)>) -> String {
        match self {
            ChatTemplate::Concat => messages
                .map(|(_, content)| content)
                .collect::<Vec<_>>()
                .join("\n"),
            ChatTemplate::RoleEot => {
                let mut out = String::new();
                for (role, content) in messages {
                    // Jinja `capitalize`: first char uppercased, REST
                    // lowercased ("USER" -> "User", matching transformers).
                    let mut cs = role.chars();
                    if let Some(c) = cs.next() {
                        out.extend(c.to_uppercase());
                        out.push_str(&cs.as_str().to_lowercase());
                    }
                    out.push_str(": ");
                    out.push_str(content.trim());
                    out.push_str("<|eot_id|>");
                }
                out.push_str("Assistant: ");
                out
            }
            ChatTemplate::QwenIm => {
                let mut out = String::new();
                for (role, content) in messages {
                    out.push_str("<|im_start|>");
                    out.push_str(role);
                    out.push('\n');
                    out.push_str(content.trim());
                    out.push_str("<|im_end|>\n");
                }
                out.push_str("<|im_start|>assistant\n");
                out
            }
        }
    }
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// The single served model id (matched against the request `model`).
    pub model_id: String,
    /// Bounded job-queue capacity (backpressure threshold).
    pub queue_cap: usize,
    /// `max_tokens` used when the request omits it.
    pub max_new_default: usize,
    /// Per-request lifetime budget. For non-streaming requests the service
    /// timeout bounds body handling, queue wait and generation. Streaming
    /// responses additionally enforce the same budget inside the lazy SSE
    /// body, starting at queue admission; expiry emits a typed error event and
    /// cancels generation. `0` disables both deadlines.
    pub request_timeout_secs: u64,
    /// Global in-flight request cap (DoS bound on handler memory/FDs).
    /// 0 disables.
    pub max_concurrent_requests: usize,
    /// When set, every request must carry `Authorization: Bearer <token>`.
    /// Required by `main` when binding beyond loopback.
    pub auth_token: Option<String>,
    /// Paged-KV pool size in TOKENS for the batched worker (ADR 0025,
    /// `--kv-pool-tokens`). `None` = dense per-slot arenas. KV VRAM scales
    /// with the pool instead of `slots × n_ctx`; admissions reserve
    /// `prompt + max_tokens` and queue when the pool is exhausted.
    pub kv_pool_tokens: Option<usize>,
    /// How chat messages render into the prompt (see [`ChatTemplate`]).
    pub chat_template: ChatTemplate,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            model_id: "tritium".to_owned(),
            queue_cap: 32,
            max_new_default: 256,
            request_timeout_secs: 600,
            max_concurrent_requests: 64,
            auth_token: None,
            kv_pool_tokens: None,
            chat_template: ChatTemplate::default(),
        }
    }
}

/// Pre-admission resource ceilings for one chat-completion request.
///
/// This is passed separately from [`ServeConfig`] so adding v1.1 governance did
/// not break downstream exhaustive literals of the existing public config.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestLimits {
    /// Maximum number of chat messages accepted before prompt rendering.
    pub max_messages: usize,
    /// Maximum aggregate UTF-8 bytes across message roles and contents.
    pub max_prompt_bytes: usize,
    /// Maximum tokenized prompt length accepted before queue admission.
    pub max_prompt_tokens: usize,
    /// Maximum requested completion length. Larger explicit values are
    /// rejected rather than silently clamped.
    pub max_new_tokens: usize,
    /// Maximum combined prompt and requested completion token budget.
    pub max_total_tokens: usize,
}

impl Default for RequestLimits {
    fn default() -> Self {
        Self {
            max_messages: 128,
            max_prompt_bytes: 1024 * 1024,
            max_prompt_tokens: 128 * 1024,
            max_new_tokens: 4096,
            max_total_tokens: 128 * 1024,
        }
    }
}

/// Serve-level counters for `/metrics` (Prometheus text format, no deps).
/// Counters are monotone; gauges are computed at scrape time.
const REQUEST_DURATION_BUCKET_US: [u64; 8] = [
    1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000,
];
const GENERATION_DURATION_BUCKET_US: [u64; 8] = [
    1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000,
];
const TTFT_BUCKET_US: [u64; 8] = [
    1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000,
];

#[derive(Debug, Default)]
pub(crate) struct Metrics {
    /// Chat completions accepted into the queue (streaming + non-streaming).
    pub(crate) chat_requests: AtomicU64,
    /// 429s returned because the job queue was full.
    pub(crate) queue_rejections: AtomicU64,
    /// 429s returned by per-principal admission control.
    pub(crate) rate_rejections: AtomicU64,
    /// Completion tokens emitted to clients across all requests.
    pub(crate) tokens_out: AtomicU64,
    /// Prompt tokens accepted into the generation queue.
    pub(crate) tokens_in: AtomicU64,
    /// Streaming generations cancelled after exceeding their lifetime budget.
    pub(crate) stream_timeouts: AtomicU64,
    /// SSE bodies dropped before a terminal event, causing cooperative cancellation.
    pub(crate) stream_disconnects: AtomicU64,
    /// HTTP requests currently inside authenticated router middleware.
    pub(crate) requests_inflight: AtomicU64,
    /// Accepted chat generations awaiting or receiving model events.
    pub(crate) generations_active: AtomicU64,
    /// Fixed-cardinality request-duration histogram buckets.
    pub(crate) request_duration_buckets: [AtomicU64; REQUEST_DURATION_BUCKET_US.len()],
    /// Sum of observed request durations in microseconds.
    pub(crate) request_duration_sum_us: AtomicU64,
    /// Count of observed request durations.
    pub(crate) request_duration_count: AtomicU64,
    /// Fixed-cardinality end-to-end generation-duration histogram buckets.
    pub(crate) generation_duration_buckets: [AtomicU64; GENERATION_DURATION_BUCKET_US.len()],
    /// Sum of observed generation durations in microseconds.
    pub(crate) generation_duration_sum_us: AtomicU64,
    /// Count of observed generation durations.
    pub(crate) generation_duration_count: AtomicU64,
    /// Fixed-cardinality time-to-first-token histogram buckets.
    pub(crate) ttft_buckets: [AtomicU64; TTFT_BUCKET_US.len()],
    /// Sum of observed time-to-first-token durations in microseconds.
    pub(crate) ttft_sum_us: AtomicU64,
    /// Count of observed time-to-first-token observations.
    pub(crate) ttft_count: AtomicU64,
}

impl Metrics {
    fn observe_request(&self, elapsed: Duration) {
        observe_histogram(
            elapsed,
            &self.request_duration_buckets,
            &self.request_duration_sum_us,
            &self.request_duration_count,
            REQUEST_DURATION_BUCKET_US,
        );
    }

    fn observe_generation(&self, elapsed: Duration) {
        observe_histogram(
            elapsed,
            &self.generation_duration_buckets,
            &self.generation_duration_sum_us,
            &self.generation_duration_count,
            GENERATION_DURATION_BUCKET_US,
        );
    }

    fn observe_ttft(&self, elapsed: Duration) {
        observe_histogram(
            elapsed,
            &self.ttft_buckets,
            &self.ttft_sum_us,
            &self.ttft_count,
            TTFT_BUCKET_US,
        );
    }
}

fn observe_histogram(
    elapsed: Duration,
    buckets: &[AtomicU64; 8],
    sum_us: &AtomicU64,
    count: &AtomicU64,
    ceilings_us: [u64; 8],
) {
    let micros = elapsed.as_micros().min(u128::from(u64::MAX)) as u64;
    sum_us.fetch_add(micros, Ordering::Relaxed);
    count.fetch_add(1, Ordering::Relaxed);
    for (bucket, ceiling) in buckets.iter().zip(ceilings_us) {
        if micros <= ceiling {
            bucket.fetch_add(1, Ordering::Relaxed);
        }
    }
}

const HISTOGRAM_BUCKET_LABELS: [&str; 8] =
    ["0.001", "0.005", "0.01", "0.05", "0.1", "0.5", "1", "5"];

fn render_histogram(
    name: &str,
    help: &str,
    buckets: &[AtomicU64; 8],
    sum_us: &AtomicU64,
    count: &AtomicU64,
) -> String {
    let mut text = format!("# HELP {name} {help}\n# TYPE {name} histogram\n");
    for (label, bucket) in HISTOGRAM_BUCKET_LABELS.iter().zip(buckets) {
        text.push_str(&format!(
            "{name}_bucket{{le=\"{label}\"}} {}\n",
            bucket.load(Ordering::Relaxed)
        ));
    }
    let total = count.load(Ordering::Relaxed);
    text.push_str(&format!(
        "{name}_bucket{{le=\"+Inf\"}} {total}\n{name}_sum {}\n{name}_count {total}\n",
        sum_us.load(Ordering::Relaxed) as f64 / 1_000_000.0
    ));
    text
}

struct RequestMetricsGuard {
    metrics: Arc<Metrics>,
    started: Instant,
}

impl RequestMetricsGuard {
    fn new(metrics: Arc<Metrics>) -> Self {
        metrics.requests_inflight.fetch_add(1, Ordering::Relaxed);
        Self {
            metrics,
            started: Instant::now(),
        }
    }
}

impl Drop for RequestMetricsGuard {
    fn drop(&mut self) {
        self.metrics
            .requests_inflight
            .fetch_sub(1, Ordering::Relaxed);
        self.metrics.observe_request(self.started.elapsed());
    }
}

struct GenerationMetricsGuard {
    metrics: Arc<Metrics>,
    started: Instant,
    first_token_seen: bool,
}

impl GenerationMetricsGuard {
    fn with_start(metrics: Arc<Metrics>, started: Instant) -> Self {
        metrics.generations_active.fetch_add(1, Ordering::Relaxed);
        Self {
            metrics,
            started,
            first_token_seen: false,
        }
    }

    fn observe_first_token(&mut self) {
        if !self.first_token_seen {
            self.first_token_seen = true;
            self.metrics.observe_ttft(self.started.elapsed());
        }
    }
}

impl Drop for GenerationMetricsGuard {
    fn drop(&mut self) {
        self.metrics
            .generations_active
            .fetch_sub(1, Ordering::Relaxed);
        self.metrics.observe_generation(self.started.elapsed());
    }
}

struct StreamDisconnectGuard {
    metrics: Arc<Metrics>,
    completed: bool,
}

impl Drop for StreamDisconnectGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.metrics
                .stream_disconnects
                .fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<Job>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    model_id: Arc<str>,
    runtime: RuntimeState,
    max_new_default: usize,
    max_messages: usize,
    max_prompt_bytes: usize,
    max_prompt_tokens: usize,
    max_new_tokens: usize,
    max_total_tokens: usize,
    request_timeout: Option<Duration>,
    chat_template: ChatTemplate,
    metrics: Arc<Metrics>,
}

#[derive(Clone)]
struct RuntimeState {
    draining: Arc<AtomicBool>,
    worker_alive: Arc<AtomicBool>,
    phase: Arc<AtomicU8>,
    backend_faulted: Arc<AtomicBool>,
    backend_faults: Arc<AtomicU64>,
    telemetry: Arc<WorkerTelemetry>,
    production: Option<ProductionReadiness>,
}

/// Build the router. Returns it plus the drain flag — set the flag (then run
/// axum's graceful shutdown) to stop accepting new work and close in-flight
/// streams cleanly.
pub fn build_router(
    generator: Box<dyn Generator>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
) -> (Router, Arc<AtomicBool>) {
    build_router_with_limits(generator, tok, cfg, RequestLimits::default())
}

/// Build the router with explicit pre-admission request ceilings.
pub fn build_router_with_limits(
    generator: Box<dyn Generator>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
    limits: RequestLimits,
) -> (Router, Arc<AtomicBool>) {
    let draining = Arc::new(AtomicBool::new(false));
    let worker_alive = Arc::new(AtomicBool::new(true));
    let phase = Arc::new(AtomicU8::new(PHASE_IDLE));
    let backend_faulted = Arc::new(AtomicBool::new(false));
    let backend_faults = Arc::new(AtomicU64::new(0));
    let telemetry = Arc::new(WorkerTelemetry::default());
    let jobs = spawn_worker(
        generator,
        WorkerSignals {
            draining: draining.clone(),
            worker_alive: worker_alive.clone(),
            phase: phase.clone(),
            backend_faulted: backend_faulted.clone(),
            backend_faults: backend_faults.clone(),
            telemetry: telemetry.clone(),
            latch_backend_faults: false,
        },
        cfg.queue_cap,
    );
    let admission = Arc::new(Admission::legacy(cfg.auth_token.as_deref()));
    build_router_inner(
        jobs,
        tok,
        cfg,
        limits,
        admission,
        RuntimeState {
            draining,
            worker_alive,
            phase,
            backend_faulted,
            backend_faults,
            telemetry,
            production: None,
        },
    )
}

/// Build the router with explicit request ceilings, rotating bearer keys and
/// fixed-cardinality per-principal admission control.
///
/// Policy errors are returned before the worker is spawned or a listener can
/// be bound.
pub fn build_router_governed(
    generator: Box<dyn Generator>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
    limits: RequestLimits,
    policy: AdmissionPolicy,
) -> std::io::Result<(Router, Arc<AtomicBool>)> {
    let admission = Arc::new(Admission::new(cfg.auth_token.as_deref(), policy)?);
    let draining = Arc::new(AtomicBool::new(false));
    let worker_alive = Arc::new(AtomicBool::new(true));
    let phase = Arc::new(AtomicU8::new(PHASE_IDLE));
    let backend_faulted = Arc::new(AtomicBool::new(false));
    let backend_faults = Arc::new(AtomicU64::new(0));
    let telemetry = Arc::new(WorkerTelemetry::default());
    let jobs = spawn_worker(
        generator,
        WorkerSignals {
            draining: draining.clone(),
            worker_alive: worker_alive.clone(),
            phase: phase.clone(),
            backend_faulted: backend_faulted.clone(),
            backend_faults: backend_faults.clone(),
            telemetry: telemetry.clone(),
            latch_backend_faults: false,
        },
        cfg.queue_cap,
    );
    Ok(build_router_inner(
        jobs,
        tok,
        cfg,
        limits,
        admission,
        RuntimeState {
            draining,
            worker_alive,
            phase,
            backend_faulted,
            backend_faults,
            telemetry,
            production: None,
        },
    ))
}

/// Build a production router only after strict schema-v3 load admission and a
/// synchronous deterministic one-token self-test. No worker or listener-facing
/// router exists when policy validation or self-test fails.
pub fn build_router_production(
    admitted: AdmittedGeneratorV1,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
    limits: RequestLimits,
    policy: AdmissionPolicy,
) -> std::io::Result<(Router, Arc<AtomicBool>, ProductionReadiness)> {
    let admission = Arc::new(Admission::new(cfg.auth_token.as_deref(), policy)?);
    let (generator, production) = prepare_production_generator(admitted)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let draining = Arc::new(AtomicBool::new(false));
    let worker_alive = Arc::new(AtomicBool::new(true));
    let phase = Arc::new(AtomicU8::new(PHASE_IDLE));
    let backend_faulted = Arc::new(AtomicBool::new(false));
    let backend_faults = Arc::new(AtomicU64::new(0));
    let telemetry = Arc::new(WorkerTelemetry::default());
    let jobs = spawn_worker(
        generator,
        WorkerSignals {
            draining: draining.clone(),
            worker_alive: worker_alive.clone(),
            phase: phase.clone(),
            backend_faulted: backend_faulted.clone(),
            backend_faults: backend_faults.clone(),
            telemetry: telemetry.clone(),
            latch_backend_faults: true,
        },
        cfg.queue_cap,
    );
    let (router, _) = build_router_inner(
        jobs,
        tok,
        cfg,
        limits,
        admission,
        RuntimeState {
            draining: draining.clone(),
            worker_alive,
            phase,
            backend_faulted,
            backend_faults,
            telemetry,
            production: Some(production.clone()),
        },
    );
    Ok((router, draining, production))
}

/// Continuous-batching router (`--batch-slots > 1`): spawns the batched
/// worker (`batch::run_batched`) on its own thread — a fixed slot pool over
/// the M=N decode graph — and routes the same job queue + SSE plumbing at it.
/// Tree/spec endpoints answer 501 in this mode (the pool owns the model).
#[cfg(feature = "cuda")]
pub fn build_router_batched(
    runner: tritium_nn::ModelRunner,
    eos: u32,
    slots: usize,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
) -> std::io::Result<(Router, Arc<AtomicBool>)> {
    build_router_batched_with_limits(runner, eos, slots, tok, cfg, RequestLimits::default())
}

/// Build the continuous-batching router with explicit request ceilings.
#[cfg(feature = "cuda")]
pub fn build_router_batched_with_limits(
    runner: tritium_nn::ModelRunner,
    eos: u32,
    slots: usize,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
    limits: RequestLimits,
) -> std::io::Result<(Router, Arc<AtomicBool>)> {
    let admission = Arc::new(Admission::legacy(cfg.auth_token.as_deref()));
    build_router_batched_inner(runner, None, eos, slots, tok, cfg, limits, admission)
}

/// Build the continuous-batching router with an attached ADR 0021 draft
/// model: solo greedy requests decode speculatively and migrate into a batch
/// slot on the next admission ("spec-when-solo, migrate-on-admission",
/// ADR 0032 L3 I0 — see `batch.rs`).
#[cfg(feature = "cuda")]
pub fn build_router_batched_with_draft(
    runner: tritium_nn::ModelRunner,
    draft: tritium_nn::ModelRunner,
    eos: u32,
    slots: usize,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
) -> std::io::Result<(Router, Arc<AtomicBool>)> {
    let admission = Arc::new(Admission::legacy(cfg.auth_token.as_deref()));
    build_router_batched_inner(
        runner,
        Some(draft),
        eos,
        slots,
        tok,
        cfg,
        RequestLimits::default(),
        admission,
    )
}

/// Build the continuous-batching router with rotating bearer keys and
/// fixed-cardinality per-principal admission control. `draft` optionally
/// attaches the ADR 0021 drafter for the I0 solo-spec path.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)] // the governed worker's full wiring
pub fn build_router_batched_governed(
    runner: tritium_nn::ModelRunner,
    draft: Option<tritium_nn::ModelRunner>,
    eos: u32,
    slots: usize,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
    limits: RequestLimits,
    policy: AdmissionPolicy,
) -> std::io::Result<(Router, Arc<AtomicBool>)> {
    let admission = Arc::new(Admission::new(cfg.auth_token.as_deref(), policy)?);
    build_router_batched_inner(runner, draft, eos, slots, tok, cfg, limits, admission)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)] // the batched worker's full wiring
fn build_router_batched_inner(
    runner: tritium_nn::ModelRunner,
    draft: Option<tritium_nn::ModelRunner>,
    eos: u32,
    slots: usize,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
    limits: RequestLimits,
    admission: Arc<Admission>,
) -> std::io::Result<(Router, Arc<AtomicBool>)> {
    use std::sync::atomic::Ordering;
    if slots == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "--batch-slots must be >= 1",
        ));
    }
    let (jobs_tx, jobs_rx) = tokio::sync::mpsc::channel(cfg.queue_cap);
    let worker_alive = Arc::new(AtomicBool::new(true));
    let draining = Arc::new(AtomicBool::new(false));
    let phase = Arc::new(AtomicU8::new(PHASE_IDLE));
    let backend_faulted = Arc::new(AtomicBool::new(false));
    let backend_faults = Arc::new(AtomicU64::new(0));
    let telemetry = Arc::new(WorkerTelemetry::default());
    let alive = worker_alive.clone();
    let drain_flag = draining.clone();
    let worker_phase = phase.clone();
    let worker_telemetry = telemetry.clone();
    let pool_tokens = cfg.kv_pool_tokens;
    std::thread::Builder::new()
        .name("tritium-serve-batch".into())
        .spawn(move || {
            // Drop guard: liveness flips false even if the loop panics
            // (mirrors the single worker's AliveGuard).
            struct Guard(Arc<AtomicBool>);
            impl Drop for Guard {
                fn drop(&mut self) {
                    self.0.store(false, Ordering::SeqCst);
                }
            }
            let _guard = Guard(alive);
            crate::batch::run_batched(
                runner,
                draft,
                eos,
                slots,
                pool_tokens,
                jobs_rx,
                drain_flag,
                worker_phase,
                worker_telemetry,
            );
        })?;
    Ok(build_router_inner(
        jobs_tx,
        tok,
        cfg,
        limits,
        admission,
        RuntimeState {
            draining,
            worker_alive,
            phase,
            backend_faulted,
            backend_faults,
            telemetry,
            production: None,
        },
    ))
}

fn build_router_inner(
    jobs: tokio::sync::mpsc::Sender<crate::worker::Job>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
    limits: RequestLimits,
    admission: Arc<Admission>,
    runtime: RuntimeState,
) -> (Router, Arc<AtomicBool>) {
    let metrics_state = Arc::new(Metrics::default());
    let state = AppState {
        jobs,
        tok,
        model_id: Arc::from(cfg.model_id.as_str()),
        runtime: runtime.clone(),
        max_new_default: cfg.max_new_default,
        max_messages: limits.max_messages,
        max_prompt_bytes: limits.max_prompt_bytes,
        max_prompt_tokens: limits.max_prompt_tokens,
        max_new_tokens: limits.max_new_tokens,
        max_total_tokens: limits.max_total_tokens,
        request_timeout: (cfg.request_timeout_secs > 0)
            .then(|| Duration::from_secs(cfg.request_timeout_secs)),
        chat_template: cfg.chat_template,
        metrics: metrics_state.clone(),
    };
    let mut router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/healthz", get(health))
        .route("/readyz", get(readiness))
        .route("/metrics", get(metrics))
        .route("/v1/tree/session", post(tree_session))
        .route("/v1/tree/verify", post(tree_verify))
        .with_state(state)
        // Explicit request-body cap (axum's default, stated rather than
        // implied — threat-model DoS bound).
        .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024));
    if cfg.request_timeout_secs > 0 {
        // Bound body extraction, handler work, queue wait and buffered
        // generation. Streaming has a second deadline inside its lazy body.
        // A local middleware is used so timeout failures retain the OpenAI
        // error envelope instead of tower-http's empty 408 response.
        let timeout = Duration::from_secs(cfg.request_timeout_secs);
        router = router.layer(axum::middleware::from_fn(
            move |req: axum::extract::Request, next: axum::middleware::Next| async move {
                match tokio::time::timeout(timeout, next.run(req)).await {
                    Ok(response) => response,
                    Err(_) => {
                        (StatusCode::REQUEST_TIMEOUT, Json(request_timeout_error())).into_response()
                    }
                }
            },
        ));
    }
    if cfg.max_concurrent_requests > 0 {
        // Global in-flight cap: bounds handler memory/FD growth under
        // connection floods (threat-model slowloris item; the accept loop
        // itself remains unbounded — documented residual).
        router = router.layer(tower::limit::ConcurrencyLimitLayer::new(
            cfg.max_concurrent_requests,
        ));
    }
    // Authentication and admission share one bounded principal resolution.
    // The middleware covers every endpoint for uniform auth, but charges only
    // routes that can enqueue model work; health and metrics remain probeable
    // by an authenticated operator even when the generation bucket is empty.
    router = router.layer(axum::middleware::from_fn(
        move |req: axum::extract::Request, next: axum::middleware::Next| {
            let admission = admission.clone();
            let metrics = metrics_state.clone();
            async move {
                let _request_metrics = RequestMetricsGuard::new(metrics.clone());
                let presented = req
                    .headers()
                    .get(axum::http::header::AUTHORIZATION)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.strip_prefix("Bearer "));
                let Some(principal) = admission.authenticate(presented) else {
                    let mut response = api_error(
                        StatusCode::UNAUTHORIZED,
                        "invalid_request_error",
                        "missing or invalid bearer token",
                        None,
                    );
                    response.headers_mut().insert(
                        axum::http::header::WWW_AUTHENTICATE,
                        axum::http::HeaderValue::from_static("Bearer"),
                    );
                    return response;
                };

                let governed = req.method() == axum::http::Method::POST
                    && matches!(
                        req.uri().path(),
                        "/v1/chat/completions" | "/v1/tree/session" | "/v1/tree/verify"
                    );
                if governed
                    && let AdmissionDecision::Reject { retry_after_secs } =
                        admission.admit(principal)
                {
                    metrics.rate_rejections.fetch_add(1, Ordering::Relaxed);
                    let mut response = api_error(
                        StatusCode::TOO_MANY_REQUESTS,
                        "rate_limit_exceeded",
                        "principal request rate exceeded; retry later",
                        None,
                    );
                    if let Ok(value) = HeaderValue::from_str(&retry_after_secs.to_string()) {
                        response.headers_mut().insert(header::RETRY_AFTER, value);
                    }
                    return response;
                }
                next.run(req).await
            }
        },
    ));
    (router, runtime.draining)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn make_id() -> String {
    format!(
        "chatcmpl-{:016x}",
        ID_COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn api_error(
    status: StatusCode,
    kind: &str,
    msg: impl Into<String>,
    param: Option<&str>,
) -> Response {
    (status, Json(ApiError::new(kind, msg, param))).into_response()
}

fn request_timeout_error() -> ApiError {
    let mut error = ApiError::new(
        "request_timeout_error",
        "request exceeded the configured lifetime",
        None,
    );
    error.error.code = Some("request_timeout".to_owned());
    error
}

fn request_ready(state: &AppState) -> bool {
    state.runtime.worker_alive.load(Ordering::Relaxed)
        && !state.runtime.backend_faulted.load(Ordering::Acquire)
        && !state.runtime.draining.load(Ordering::Relaxed)
        && state
            .runtime
            .production
            .as_ref()
            .is_none_or(ProductionReadiness::is_serving)
}

fn json_rejection(rejection: JsonRejection) -> Response {
    let (status, message) = match rejection.status() {
        StatusCode::PAYLOAD_TOO_LARGE => (
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body exceeds configured byte limit",
        ),
        StatusCode::UNSUPPORTED_MEDIA_TYPE => (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "content-type must be application/json",
        ),
        _ => (
            StatusCode::BAD_REQUEST,
            "request body must be valid application/json",
        ),
    };
    api_error(status, "invalid_request_error", message, None)
}

/// Lower the OpenAI sampling fields to the internal [`Sampling`]. `temperature
/// <= 0` is deterministic greedy; otherwise top-p if given, else greedy (we do
/// not invent a default top_p).
fn lower_sampling(req: &ChatRequest) -> Sampling {
    let seed = req.seed.unwrap_or(0x7A1C_0DE5);
    if req.temperature <= 0.0 {
        Sampling::Greedy
    } else if let Some(p) = req.top_p {
        Sampling::TopP {
            p,
            temp: req.temperature,
            seed,
        }
    } else {
        Sampling::Greedy
    }
}

async fn chat_completions(
    State(st): State<AppState>,
    request: Result<Json<ChatRequest>, JsonRejection>,
) -> Response {
    let req = match request {
        Ok(Json(req)) => req,
        Err(rejection) => return json_rejection(rejection),
    };
    if st.runtime.draining.load(Ordering::Relaxed) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "server is draining",
            None,
        );
    }
    if !request_ready(&st) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "server is not ready",
            None,
        );
    }
    if req.top_logprobs.is_some() && !req.logprobs {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "top_logprobs requires logprobs: true",
            Some("top_logprobs"),
        );
    }
    if req.top_logprobs.is_some_and(|k| k > 20) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "top_logprobs must be in [0, 20]",
            Some("top_logprobs"),
        );
    }
    if !req.stream && req.stream_options.is_some() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "stream_options is only allowed when stream is true",
            Some("stream_options"),
        );
    }
    if req.messages.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "messages must not be empty",
            Some("messages"),
        );
    }
    if req.messages.len() > st.max_messages {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("messages supports at most {} entries", st.max_messages),
            Some("messages"),
        );
    }
    let prompt_bytes = req.messages.iter().try_fold(0_usize, |total, message| {
        total
            .checked_add(message.role.len())?
            .checked_add(message.content.len())
    });
    if prompt_bytes.is_none_or(|bytes| bytes > st.max_prompt_bytes) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "message roles and contents exceed the {} byte prompt limit",
                st.max_prompt_bytes
            ),
            Some("messages"),
        );
    }
    if !(0.0..=2.0).contains(&req.temperature) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "temperature must be in [0, 2]",
            Some("temperature"),
        );
    }
    if req.model.as_str() != &*st.model_id {
        return api_error(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("model {:?} not found", req.model),
            Some("model"),
        );
    }
    let stops = req
        .stop
        .clone()
        .map(StopField::into_vec)
        .unwrap_or_default();
    if stops.len() > 4 {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "stop supports at most 4 sequences",
            Some("stop"),
        );
    }
    if stops.iter().any(String::is_empty) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "stop sequences must be non-empty",
            Some("stop"),
        );
    }
    if let Some(p) = req.top_p
        && !(p.is_finite() && p > 0.0 && p <= 1.0)
    {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "top_p must be in (0, 1]",
            Some("top_p"),
        );
    }
    if req.max_tokens == Some(0) {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "max_tokens must be >= 1",
            Some("max_tokens"),
        );
    }

    let prompt_text = st.chat_template.render(
        req.messages
            .iter()
            .map(|m| (m.role.as_str(), m.content.as_str())),
    );
    let prompt_tokens = match st.tok.encode(&prompt_text) {
        Ok(t) => t,
        Err(e) => {
            return api_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                e.to_string(),
                Some("messages"),
            );
        }
    };
    if prompt_tokens.is_empty() {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "prompt is empty after tokenization",
            Some("messages"),
        );
    }
    let prompt_len = prompt_tokens.len();
    let max_new = req.max_tokens.map_or(st.max_new_default, |m| m as usize);
    if prompt_len > st.max_prompt_tokens {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "tokenized prompt exceeds the {} token limit",
                st.max_prompt_tokens
            ),
            Some("messages"),
        );
    }
    if max_new > st.max_new_tokens {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "max_tokens exceeds the {} token completion limit",
                st.max_new_tokens
            ),
            Some("max_tokens"),
        );
    }
    if prompt_len
        .checked_add(max_new)
        .is_none_or(|tokens| tokens > st.max_total_tokens)
    {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "prompt plus max_tokens exceeds the {} token request limit",
                st.max_total_tokens
            ),
            Some("max_tokens"),
        );
    }
    let logprobs_k = req.logprobs.then(|| req.top_logprobs.unwrap_or(0) as usize);
    let gen_req = GenRequest {
        prompt_tokens,
        max_new,
        logprobs: logprobs_k,
        sampling: lower_sampling(&req),
        stop_eos: true,
    };

    let (tx, rx) = mpsc::channel::<GenEvent>(64);
    // Start latency accounting before queue admission so generation duration
    // includes bounded queue wait, not only response-body consumption.
    let generation_started = Instant::now();
    match st.jobs.try_send(Job::Generate {
        req: gen_req,
        accepted_at: Some(generation_started),
        tx,
    }) {
        Ok(()) => {
            st.metrics.chat_requests.fetch_add(1, Ordering::Relaxed);
            st.metrics
                .tokens_in
                .fetch_add(prompt_len as u64, Ordering::Relaxed);
        }
        Err(mpsc::error::TrySendError::Full(_)) => {
            st.metrics.queue_rejections.fetch_add(1, Ordering::Relaxed);
            let mut r = api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_exceeded",
                "server is at capacity; retry shortly",
                None,
            );
            r.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
            return r;
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            return api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "server is shutting down",
                None,
            );
        }
    }

    if req.stream {
        let include_usage = req.stream_options.is_some_and(|o| o.include_usage);
        stream_response(
            rx,
            st.tok.clone(),
            req.model,
            stops,
            st.metrics.clone(),
            generation_started,
            include_usage.then_some(prompt_len),
            st.request_timeout,
        )
    } else {
        nonstream_response(
            rx,
            st.tok.clone(),
            req.model,
            prompt_len,
            stops,
            st.metrics.clone(),
            generation_started,
        )
        .await
    }
}

async fn nonstream_response(
    mut rx: mpsc::Receiver<GenEvent>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    model: String,
    prompt_len: usize,
    stops: Vec<String>,
    metrics: Arc<Metrics>,
    generation_started: Instant,
) -> Response {
    let mut generation_metrics =
        GenerationMetricsGuard::with_start(metrics.clone(), generation_started);
    let eos = tok.eos();
    let detok_ref = tok.clone();
    // Mirror the streaming loop: incremental detok + stop-matcher, BREAK on a
    // stop hit (dropping `rx` cancels the worker's generation) — so stop
    // strings no longer burn decode budget past the match, and
    // usage.completion_tokens agrees with the streamed accounting.
    let mut detok = IncrementalDetok::new(tok);
    let mut matcher = StopMatcher::new(stops);
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut logprob_rows: Vec<Vec<(u32, f32)>> = Vec::new();
    let mut fr = FinishReason::Stop;
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Token(t, lp) => {
                if t != eos {
                    generation_metrics.observe_first_token();
                    completion_tokens += 1;
                    metrics.tokens_out.fetch_add(1, Ordering::Relaxed);
                    if let Some(lp) = lp {
                        logprob_rows.push(lp);
                    }
                }
                let piece = detok.push(t);
                if !piece.is_empty() {
                    let (emit, hit) = matcher.feed(&piece);
                    text.push_str(&emit);
                    if hit {
                        fr = FinishReason::Stop;
                        break;
                    }
                }
            }
            GenEvent::Done(reason) => {
                fr = reason;
                text.push_str(&matcher.flush());
                break;
            }
            GenEvent::Error(e) => {
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e, None);
            }
        }
    }
    let completion = ChatCompletion {
        id: make_id(),
        object: "chat.completion",
        created: now_secs(),
        model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_owned(),
                content: text,
            },
            finish_reason: fr.as_str().to_owned(),
            logprobs: (!logprob_rows.is_empty())
                .then(|| render_logprobs(&logprob_rows, detok_ref.as_ref())),
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens,
            total_tokens: prompt_len + completion_tokens,
        },
    };
    Json(completion).into_response()
}

/// Lower `(token, logprob)` rows into the OpenAI logprobs shape. Row layout:
/// sampled token first, then the top-k alternatives.
fn render_logprobs(
    rows: &[Vec<(u32, f32)>],
    tok: &(dyn Tokenizer + Send + Sync),
) -> crate::dto::ChoiceLogprobs {
    let piece = |id: u32| tok.decode(&[id]).unwrap_or_default();
    crate::dto::ChoiceLogprobs {
        content: rows
            .iter()
            .filter_map(|row| {
                let (t, lp) = *row.first()?;
                let text = piece(t);
                Some(crate::dto::TokenLogprob {
                    bytes: text.as_bytes().to_vec(),
                    token: text,
                    logprob: lp,
                    top_logprobs: row[1..]
                        .iter()
                        .map(|&(alt, alp)| {
                            let atext = piece(alt);
                            crate::dto::TopLogprob {
                                bytes: atext.as_bytes().to_vec(),
                                token: atext,
                                logprob: alp,
                            }
                        })
                        .collect(),
                })
            })
            .collect(),
    }
}

fn sse_data(chunk: &ChatChunk) -> Event {
    Event::default().data(serde_json::to_string(chunk).unwrap_or_default())
}

#[allow(clippy::too_many_arguments)] // wire contract keeps stream controls explicit
fn stream_response(
    mut rx: mpsc::Receiver<GenEvent>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    model: String,
    stops: Vec<String>,
    metrics: Arc<Metrics>,
    generation_started: Instant,
    // `Some(prompt_tokens)` when the client asked for
    // `stream_options.include_usage`: emit the final usage chunk.
    usage_prompt_len: Option<usize>,
    // Absolute lifetime measured from queue admission. Unlike tower's
    // service-future timeout, this remains active after SSE headers resolve.
    timeout: Option<Duration>,
) -> Response {
    let id = make_id();
    let created = now_secs();
    let detok_eos = tok.eos();
    let stream_tok = tok.clone();
    let deadline = timeout.map(|budget| {
        let now = tokio::time::Instant::now();
        // A pathological public config must fail closed rather than turning
        // an overflowing lifetime into an unbounded stream.
        now.checked_add(budget).unwrap_or(now)
    });
    let generation_metrics =
        GenerationMetricsGuard::with_start(metrics.clone(), generation_started);
    let stream = async_stream::stream! {
        let mut generation_metrics = generation_metrics;
        let mut disconnect = StreamDisconnectGuard {
            metrics: metrics.clone(),
            completed: false,
        };
        // 1. role-first chunk
        yield Ok::<Event, std::convert::Infallible>(sse_data(&role_chunk(&id, created, &model)));

        let mut detok = IncrementalDetok::new(tok);
        let mut matcher = StopMatcher::new(stops);
        let mut completion_tokens = 0usize;
        let mut pending_rows: Vec<Vec<(u32, f32)>> = Vec::new();
        let mut finish = FinishReason::Stop;
        let mut stopped_by_string = false;
        let mut errored = false;
        let mut timed_out = false;

        loop {
            let next = match deadline {
                Some(deadline) => match tokio::time::timeout_at(deadline, rx.recv()).await {
                    Ok(event) => event,
                    Err(_) => {
                        timed_out = true;
                        metrics.stream_timeouts.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                },
                None => rx.recv().await,
            };
            let Some(ev) = next else { break };
            match ev {
                GenEvent::Token(t, lp) => {
                    // Count like the non-stream path: eos terminates, it is
                    // not an emitted completion token.
                    if t != detok_eos {
                        generation_metrics.observe_first_token();
                        metrics.tokens_out.fetch_add(1, Ordering::Relaxed);
                        completion_tokens += 1;
                        // Buffer the row: text may be held back (StopMatcher
                        // partial match, mid-codepoint byte-level token), so
                        // records ride the NEXT chunk that actually emits —
                        // no row is ever dropped (review finding).
                        if let Some(lp) = lp {
                            pending_rows.push(lp);
                        }
                    }
                    let text = detok.push(t);
                    if !text.is_empty() {
                        let (emit, hit) = matcher.feed(&text);
                        if !emit.is_empty() {
                            let mut chunk = content_chunk(&id, created, &model, &emit);
                            if !pending_rows.is_empty() {
                                chunk.choices[0].logprobs = Some(render_logprobs(
                                    &pending_rows,
                                    stream_tok.as_ref(),
                                ));
                                pending_rows.clear();
                            }
                            yield Ok(sse_data(&chunk));
                        }
                        if hit {
                            stopped_by_string = true;
                            finish = FinishReason::Stop;
                            break;
                        }
                    }
                }
                GenEvent::Done(fr) => { finish = fr; break; }
                // Surface a backend error distinctly (finish_reason "error"), not a
                // clean "stop" — a streaming client must be able to detect failure.
                GenEvent::Error(_) => { errored = true; break; }
            }
        }

        if timed_out {
            let error = request_timeout_error();
            yield Ok(Event::default().data(serde_json::to_string(&error).unwrap_or_default()));
        } else if errored {
            yield Ok(sse_data(&error_chunk(&id, created, &model)));
        } else {
            if !stopped_by_string {
                let tail = matcher.flush();
                if !tail.is_empty() {
                    let mut chunk = content_chunk(&id, created, &model, &tail);
                    if !pending_rows.is_empty() {
                        chunk.choices[0].logprobs =
                            Some(render_logprobs(&pending_rows, stream_tok.as_ref()));
                        pending_rows.clear();
                    }
                    yield Ok(sse_data(&chunk));
                }
            }
            // Rows still pending (empty flush tail, or a stop-string hit
            // whose matched token never emitted): carry them on an
            // empty-content chunk so streamed rows == non-streamed rows.
            if !pending_rows.is_empty() {
                let mut chunk = content_chunk(&id, created, &model, "");
                chunk.choices[0].logprobs =
                    Some(render_logprobs(&pending_rows, stream_tok.as_ref()));
                yield Ok(sse_data(&chunk));
            }
            yield Ok(sse_data(&terminal_chunk(&id, created, &model, finish)));
            // OpenAI stream_options.include_usage: one final chunk with
            // empty choices carrying the token accounting.
            if let Some(prompt_tokens) = usage_prompt_len {
                yield Ok(sse_data(&crate::sse::usage_chunk(
                    &id,
                    created,
                    &model,
                    prompt_tokens,
                    completion_tokens,
                )));
            }
        }
        disconnect.completed = true;
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn models(State(st): State<AppState>) -> Response {
    Json(ModelList {
        object: "list",
        data: vec![ModelEntry {
            id: st.model_id.to_string(),
            object: "model",
            created: now_secs(),
            owned_by: "tritium",
        }],
    })
    .into_response()
}

async fn health(State(st): State<AppState>) -> Response {
    let queue_depth = st.jobs.max_capacity() - st.jobs.capacity();
    if !st.runtime.worker_alive.load(Ordering::Relaxed) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "unhealthy",
                "detail": "decode worker stopped",
                "model": &*st.model_id,
            })),
        )
            .into_response();
    }
    if st.runtime.backend_faulted.load(Ordering::Acquire) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "unhealthy",
                "detail": "backend fault latched",
                "model": &*st.model_id,
                "worker_alive": true,
            })),
        )
            .into_response();
    }
    Json(serde_json::json!({
        "status": "ok",
        "model": &*st.model_id,
        "worker_alive": true,
        "draining": st.runtime.draining.load(Ordering::Relaxed),
        "queue_depth": queue_depth,
        // RFC 0001: disclose the numerics domain this process serves. The
        // tier/rungs are parsed once at model build from these env vars
        // (loud-reject on typos — a process with an invalid value never gets
        // this far). `kernel_tier` is the RESOLVED tier, mirroring
        // `kernel_tier_from_env` (unset and "" both parse to exact), not a
        // raw env echo. NOTE: `fast` is a per-route claim — verifies outside
        // the fused contract (head_dim > 128, paged/slots arenas, eager
        // fallback) still run exact numerics in a fast-tier process.
        "kernel_tier": match std::env::var("TRITIUM_KERNEL_TIER").as_deref() {
            Ok("fast") => "fast",
            _ => "exact",
        },
        "kv_dtype": std::env::var("TRITIUM_KV")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| {
                // Legacy alias honored by kv_dtype_from_env.
                if std::env::var("TRITIUM_KV_F16").is_ok_and(|v| v == "1") {
                    "f16".into()
                } else {
                    "f32".into()
                }
            }),
        "lm_head": std::env::var("TRITIUM_LM_HEAD").unwrap_or_else(|_| "f16".into()),
    }))
    .into_response()
}

async fn readiness(State(st): State<AppState>) -> Response {
    let worker_alive = st.runtime.worker_alive.load(Ordering::Relaxed);
    let draining = st.runtime.draining.load(Ordering::Relaxed);
    let backend_faulted = st.runtime.backend_faulted.load(Ordering::Acquire);
    let queue_depth = st.jobs.max_capacity() - st.jobs.capacity();
    let artifact_ready = st
        .runtime
        .production
        .as_ref()
        .is_none_or(|state| state.is_serving());
    let ready = worker_alive && !draining && !backend_faulted && artifact_ready;
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(serde_json::json!({
            "status": if ready { "ready" } else { "not_ready" },
            "model": &*st.model_id,
            "worker_alive": worker_alive,
            "draining": draining,
            "backend_faulted": backend_faulted,
            "queue_depth": queue_depth,
            "production_artifact": st.runtime.production.is_some(),
            "artifact_ready": artifact_ready,
            "release_gate": if st.runtime.production.is_some() {
                "production_artifact_admitted"
            } else {
                "legacy_compatibility"
            },
            "startup_receipt": st.runtime.production.as_ref().map(ProductionReadiness::receipt),
        })),
    )
        .into_response()
}

/// Prometheus text exposition (behind the same auth as everything else).
/// Gauges are scrape-time reads; counters live in [`Metrics`].
async fn metrics(State(st): State<AppState>) -> Response {
    let queue_depth = st.jobs.max_capacity() - st.jobs.capacity();
    let phase = st.runtime.phase.load(Ordering::Acquire);
    let request_buckets = st
        .metrics
        .request_duration_buckets
        .iter()
        .map(|bucket| bucket.load(Ordering::Relaxed));
    let request_buckets: Vec<u64> = request_buckets.collect();
    let generation_buckets: Vec<u64> = st
        .metrics
        .generation_duration_buckets
        .iter()
        .map(|bucket| bucket.load(Ordering::Relaxed))
        .collect();
    let ttft_buckets: Vec<u64> = st
        .metrics
        .ttft_buckets
        .iter()
        .map(|bucket| bucket.load(Ordering::Relaxed))
        .collect();
    let worker = &st.runtime.telemetry;
    let queue_wait_histogram = render_histogram(
        "tritium_queue_wait_seconds",
        "Time from accepted request to decode-worker admission.",
        &worker.queue_wait_buckets,
        &worker.queue_wait_sum_us,
        &worker.queue_wait_count,
    );
    let prefill_histogram = render_histogram(
        "tritium_prefill_duration_seconds",
        "Decode-worker model prefill duration.",
        &worker.prefill_buckets,
        &worker.prefill_sum_us,
        &worker.prefill_count,
    );
    let decode_histogram = render_histogram(
        "tritium_decode_duration_seconds",
        "Decode-worker token-generation duration after first token callback.",
        &worker.decode_buckets,
        &worker.decode_sum_us,
        &worker.decode_count,
    );
    let body = format!(
        "# HELP tritium_chat_requests_total Chat completions accepted into the queue.\n\
         # TYPE tritium_chat_requests_total counter\n\
         tritium_chat_requests_total {}\n\
         # HELP tritium_queue_rejections_total Requests 429'd because the job queue was full.\n\
         # TYPE tritium_queue_rejections_total counter\n\
         tritium_queue_rejections_total {}\n\
         # HELP tritium_rate_rejections_total Requests 429'd by per-principal admission control.\n\
         # TYPE tritium_rate_rejections_total counter\n\
         tritium_rate_rejections_total {}\n\
         # HELP tritium_tokens_out_total Completion tokens emitted to clients.\n\
         # TYPE tritium_tokens_out_total counter\n\
         tritium_tokens_out_total {}\n\
         # HELP tritium_tokens_in_total Prompt tokens accepted into the generation queue.\n\
         # TYPE tritium_tokens_in_total counter\n\
         tritium_tokens_in_total {}\n\
         # HELP tritium_stream_timeouts_total Streaming generations cancelled at their lifetime deadline.\n\
         # TYPE tritium_stream_timeouts_total counter\n\
         tritium_stream_timeouts_total {}\n\
         # HELP tritium_stream_disconnects_total SSE bodies dropped before terminal completion.\n\
         # TYPE tritium_stream_disconnects_total counter\n\
         tritium_stream_disconnects_total {}\n\
         # HELP tritium_requests_inflight HTTP requests inside router middleware.\n\
         # TYPE tritium_requests_inflight gauge\n\
         tritium_requests_inflight {}\n\
         # HELP tritium_generations_active Accepted chat generations awaiting or receiving model events.\n\
         # TYPE tritium_generations_active gauge\n\
         tritium_generations_active {}\n\
         # HELP tritium_request_duration_seconds HTTP request duration, fixed buckets.\n\
         # TYPE tritium_request_duration_seconds histogram\n\
         tritium_request_duration_seconds_bucket{{le=\"0.001\"}} {}\n\
         tritium_request_duration_seconds_bucket{{le=\"0.005\"}} {}\n\
         tritium_request_duration_seconds_bucket{{le=\"0.01\"}} {}\n\
         tritium_request_duration_seconds_bucket{{le=\"0.05\"}} {}\n\
         tritium_request_duration_seconds_bucket{{le=\"0.1\"}} {}\n\
         tritium_request_duration_seconds_bucket{{le=\"0.5\"}} {}\n\
         tritium_request_duration_seconds_bucket{{le=\"1\"}} {}\n\
         tritium_request_duration_seconds_bucket{{le=\"5\"}} {}\n\
         tritium_request_duration_seconds_bucket{{le=\"+Inf\"}} {}\n\
         tritium_request_duration_seconds_sum {}\n\
         tritium_request_duration_seconds_count {}\n\
         # HELP tritium_generation_duration_seconds End-to-end accepted generation duration, including queue wait.\n\
         # TYPE tritium_generation_duration_seconds histogram\n\
         tritium_generation_duration_seconds_bucket{{le=\"0.001\"}} {}\n\
         tritium_generation_duration_seconds_bucket{{le=\"0.005\"}} {}\n\
         tritium_generation_duration_seconds_bucket{{le=\"0.01\"}} {}\n\
         tritium_generation_duration_seconds_bucket{{le=\"0.05\"}} {}\n\
         tritium_generation_duration_seconds_bucket{{le=\"0.1\"}} {}\n\
         tritium_generation_duration_seconds_bucket{{le=\"0.5\"}} {}\n\
         tritium_generation_duration_seconds_bucket{{le=\"1\"}} {}\n\
         tritium_generation_duration_seconds_bucket{{le=\"5\"}} {}\n\
         tritium_generation_duration_seconds_bucket{{le=\"+Inf\"}} {}\n\
         tritium_generation_duration_seconds_sum {}\n\
         tritium_generation_duration_seconds_count {}\n\
         # HELP tritium_time_to_first_token_seconds Time from generation admission to first emitted token.\n\
         # TYPE tritium_time_to_first_token_seconds histogram\n\
         tritium_time_to_first_token_seconds_bucket{{le=\"0.001\"}} {}\n\
         tritium_time_to_first_token_seconds_bucket{{le=\"0.005\"}} {}\n\
         tritium_time_to_first_token_seconds_bucket{{le=\"0.01\"}} {}\n\
         tritium_time_to_first_token_seconds_bucket{{le=\"0.05\"}} {}\n\
         tritium_time_to_first_token_seconds_bucket{{le=\"0.1\"}} {}\n\
         tritium_time_to_first_token_seconds_bucket{{le=\"0.5\"}} {}\n\
         tritium_time_to_first_token_seconds_bucket{{le=\"1\"}} {}\n\
         tritium_time_to_first_token_seconds_bucket{{le=\"5\"}} {}\n\
         tritium_time_to_first_token_seconds_bucket{{le=\"+Inf\"}} {}\n\
         tritium_time_to_first_token_seconds_sum {}\n\
         tritium_time_to_first_token_seconds_count {}\n\
         {}{}{}\n\
         # HELP tritium_queue_depth Jobs waiting in the decode queue.\n\
         # TYPE tritium_queue_depth gauge\n\
         tritium_queue_depth {}\n\
         # HELP tritium_worker_alive Decode worker liveness (1 = alive).\n\
         # TYPE tritium_worker_alive gauge\n\
         tritium_worker_alive {}\n\
         # HELP tritium_backend_faults_total Backend faults observed by the decode worker.\n\
         # TYPE tritium_backend_faults_total counter\n\
         tritium_backend_faults_total {}\n\
         # HELP tritium_backend_faulted Latched production backend fault state.\n\
         # TYPE tritium_backend_faulted gauge\n\
         tritium_backend_faulted {}\n\
         # HELP tritium_worker_phase Current decode-worker phase as one-hot fixed-cardinality gauges.\n\
         # TYPE tritium_worker_phase gauge\n\
         tritium_worker_phase{{phase=\"idle\"}} {}\n\
         tritium_worker_phase{{phase=\"prefill\"}} {}\n\
         tritium_worker_phase{{phase=\"decode\"}} {}\n\
         # HELP tritium_spec_verifies_total Spec-decode tree verifies.\n\
         # TYPE tritium_spec_verifies_total counter\n\
         tritium_spec_verifies_total {}\n\
         # HELP tritium_spec_committed_total Tokens committed by spec verifies.\n\
         # TYPE tritium_spec_committed_total counter\n\
         tritium_spec_committed_total {}\n\
         # HELP tritium_spec_suppressed_plain_total Plain-decode tokens committed while adaptive spec suppression was engaged (TRITIUM_SPEC_ADAPTIVE).\n\
         # TYPE tritium_spec_suppressed_plain_total counter\n\
         tritium_spec_suppressed_plain_total {}\n\
         # HELP tritium_spec_floor Adaptive-spec breakeven floor last applied by the governor (the fixed fallback until the cost model warms).\n\
         # TYPE tritium_spec_floor gauge\n\
         tritium_spec_floor{{path=\"solo\"}} {}\n\
         tritium_spec_floor{{path=\"batched\"}} {}\n\
         # HELP tritium_spec_cost_us Measured spec-decode phase wall-cost EWMA in microseconds (0 until the first sample).\n\
         # TYPE tritium_spec_cost_us gauge\n\
         tritium_spec_cost_us{{phase=\"verify\"}} {}\n\
         tritium_spec_cost_us{{phase=\"plain\"}} {}\n\
         tritium_spec_cost_us{{phase=\"draft_token\"}} {}\n\
         tritium_spec_cost_us{{phase=\"draft_resync\"}} {}\n\
         tritium_spec_cost_us{{phase=\"verify_round\"}} {}\n\
         tritium_spec_cost_us{{phase=\"lockstep\"}} {}\n",
        st.metrics.chat_requests.load(Ordering::Relaxed),
        st.metrics.queue_rejections.load(Ordering::Relaxed),
        st.metrics.rate_rejections.load(Ordering::Relaxed),
        st.metrics.tokens_out.load(Ordering::Relaxed),
        st.metrics.tokens_in.load(Ordering::Relaxed),
        st.metrics.stream_timeouts.load(Ordering::Relaxed),
        st.metrics.stream_disconnects.load(Ordering::Relaxed),
        st.metrics.requests_inflight.load(Ordering::Relaxed),
        st.metrics.generations_active.load(Ordering::Relaxed),
        request_buckets[0],
        request_buckets[1],
        request_buckets[2],
        request_buckets[3],
        request_buckets[4],
        request_buckets[5],
        request_buckets[6],
        request_buckets[7],
        st.metrics.request_duration_count.load(Ordering::Relaxed),
        st.metrics.request_duration_sum_us.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        st.metrics.request_duration_count.load(Ordering::Relaxed),
        generation_buckets[0],
        generation_buckets[1],
        generation_buckets[2],
        generation_buckets[3],
        generation_buckets[4],
        generation_buckets[5],
        generation_buckets[6],
        generation_buckets[7],
        st.metrics.generation_duration_count.load(Ordering::Relaxed),
        st.metrics
            .generation_duration_sum_us
            .load(Ordering::Relaxed) as f64
            / 1_000_000.0,
        st.metrics.generation_duration_count.load(Ordering::Relaxed),
        ttft_buckets[0],
        ttft_buckets[1],
        ttft_buckets[2],
        ttft_buckets[3],
        ttft_buckets[4],
        ttft_buckets[5],
        ttft_buckets[6],
        ttft_buckets[7],
        st.metrics.ttft_count.load(Ordering::Relaxed),
        st.metrics.ttft_sum_us.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        st.metrics.ttft_count.load(Ordering::Relaxed),
        queue_wait_histogram,
        prefill_histogram,
        decode_histogram,
        queue_depth,
        u8::from(st.runtime.worker_alive.load(Ordering::Relaxed)),
        st.runtime.backend_faults.load(Ordering::Relaxed),
        u8::from(st.runtime.backend_faulted.load(Ordering::Acquire)),
        u8::from(phase == PHASE_IDLE),
        u8::from(phase == PHASE_PREFILL),
        u8::from(phase == PHASE_DECODE),
        crate::generator::SPEC_VERIFIES.load(Ordering::Relaxed),
        crate::generator::SPEC_COMMITTED.load(Ordering::Relaxed),
        crate::generator::SPEC_SUPPRESSED_PLAIN.load(Ordering::Relaxed),
        f64::from_bits(crate::generator::SPEC_FLOOR_SOLO.load(Ordering::Relaxed)),
        f64::from_bits(crate::generator::SPEC_FLOOR_BATCHED.load(Ordering::Relaxed)),
        crate::generator::SPEC_COST.verify.mean_us().unwrap_or(0.0),
        crate::generator::SPEC_COST.plain.mean_us().unwrap_or(0.0),
        crate::generator::SPEC_COST
            .draft_tok
            .mean_us()
            .unwrap_or(0.0),
        // Probe-time drafter re-sync (telemetry only — never in the floors;
        // see SpecCostModel::draft_resync).
        crate::generator::SPEC_COST
            .draft_resync
            .mean_us()
            .unwrap_or(0.0),
        crate::generator::SPEC_COST
            .verify_round
            .mean_us()
            .unwrap_or(0.0),
        crate::generator::SPEC_COST
            .lockstep
            .mean_us()
            .unwrap_or(0.0),
    );
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body).into_response()
}

// ───────────────────── BASTION tree-verify surface (ADR 0014) ─────────────────────
//
// The stateful spec-decode boundary an external orchestrator (e.g. LAMU driving a
// block-diffusion drafter) uses:
//
//   POST /v1/tree/session  {"prompt_tokens":[...]}      → {"pending_token":t1}
//   POST /v1/tree/verify   {"tokens":[t1,d..],"parents":[-1,..]}
//                                                        → {"committed":[...]}
//
// One session at a time (the worker owns one model); a chat completion invalidates
// it — in batched mode (`--batch-slots > 1`, C4) the same contract reads: a chat
// ADMISSION closes the session while already-running chat streams coexist with
// it. Node 0 of every verify tree must be the current pending token; the new
// pending token is the last committed element. Backends without the CUDA
// device-resident decoder answer 501.

#[derive(serde::Deserialize)]
struct TreeSessionRequest {
    prompt_tokens: Vec<u32>,
}

#[derive(serde::Deserialize)]
struct TreeVerifyRequest {
    tokens: Vec<u32>,
    parents: Vec<i32>,
}

async fn tree_session(
    State(st): State<AppState>,
    request: Result<Json<TreeSessionRequest>, JsonRejection>,
) -> Response {
    let req = match request {
        Ok(Json(req)) => req,
        Err(rejection) => return json_rejection(rejection),
    };
    if st.runtime.draining.load(Ordering::Relaxed) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "draining",
            "server is draining",
            None,
        );
    }
    if !request_ready(&st) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "server is not ready",
            None,
        );
    }
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    if let Err(e) = st.jobs.try_send(Job::OpenTreeSession {
        prompt: req.prompt_tokens,
        resp: resp_tx,
    }) {
        return tree_queue_error(e);
    }
    match resp_rx.await {
        Ok(Ok(pending)) => Json(serde_json::json!({ "pending_token": pending })).into_response(),
        Ok(Err(e)) => tree_error(&e),
        Err(_) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "worker dropped the request",
            None,
        ),
    }
}

async fn tree_verify(
    State(st): State<AppState>,
    request: Result<Json<TreeVerifyRequest>, JsonRejection>,
) -> Response {
    let req = match request {
        Ok(Json(req)) => req,
        Err(rejection) => return json_rejection(rejection),
    };
    if st.runtime.draining.load(Ordering::Relaxed) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "draining",
            "server is draining",
            None,
        );
    }
    if !request_ready(&st) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "server is not ready",
            None,
        );
    }
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    if let Err(e) = st.jobs.try_send(Job::TreeVerify {
        tokens: req.tokens,
        parents: req.parents,
        resp: resp_tx,
    }) {
        return tree_queue_error(e);
    }
    match resp_rx.await {
        Ok(Ok(committed)) => Json(serde_json::json!({ "committed": committed })).into_response(),
        Ok(Err(e)) => tree_error(&e),
        Err(_) => api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "worker dropped the request",
            None,
        ),
    }
}

/// Queue submission failures, matching the chat path's semantics: Full → 429
/// with Retry-After; Closed (dead/shutting-down worker) → 503, so tree
/// clients don't spin forever on a dead server.
fn tree_queue_error(e: mpsc::error::TrySendError<Job>) -> Response {
    match e {
        mpsc::error::TrySendError::Full(_) => {
            let mut r = api_error(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_exceeded",
                "decode queue full; retry shortly",
                None,
            );
            r.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
            r
        }
        mpsc::error::TrySendError::Closed(_) => api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "shutting_down",
            "server is shutting down",
            None,
        ),
    }
}

/// Map worker-side tree errors to HTTP by VARIANT (no string sniffing —
/// a CUDA driver message containing "not supported" stays a 500).
fn tree_error(e: &TreeOpError) -> Response {
    let (code, kind) = match e {
        TreeOpError::Draining(_) => (StatusCode::SERVICE_UNAVAILABLE, "draining"),
        TreeOpError::Unsupported(_) => (StatusCode::NOT_IMPLEMENTED, "tree_unsupported"),
        TreeOpError::Conflict(_) => (StatusCode::CONFLICT, "tree_session_closed"),
        TreeOpError::BadRequest(_) => (StatusCode::BAD_REQUEST, "tree_bad_request"),
        TreeOpError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
    };
    api_error(code, kind, e.to_string(), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    struct BackendFail;

    impl Generator for BackendFail {
        fn generate(
            &mut self,
            _req: &GenRequest,
            _on_step: &mut dyn FnMut(crate::generator::Step) -> bool,
        ) -> Result<(), crate::generator::GenError> {
            Err(crate::generator::GenError::Backend(
                "device lost".to_owned(),
            ))
        }

        fn n_ctx(&self) -> usize {
            4096
        }

        fn vocab(&self) -> usize {
            128_256
        }
    }

    fn latching_router() -> Router {
        let cfg = ServeConfig::default();
        let draining = Arc::new(AtomicBool::new(false));
        let worker_alive = Arc::new(AtomicBool::new(true));
        let phase = Arc::new(AtomicU8::new(PHASE_IDLE));
        let backend_faulted = Arc::new(AtomicBool::new(false));
        let backend_faults = Arc::new(AtomicU64::new(0));
        let telemetry = Arc::new(WorkerTelemetry::default());
        let jobs = spawn_worker(
            Box::new(BackendFail),
            WorkerSignals {
                draining: draining.clone(),
                worker_alive: worker_alive.clone(),
                phase: phase.clone(),
                backend_faulted: backend_faulted.clone(),
                backend_faults: backend_faults.clone(),
                telemetry: telemetry.clone(),
                latch_backend_faults: true,
            },
            cfg.queue_cap,
        );
        build_router_inner(
            jobs,
            Arc::new(crate::IdPassthroughTokenizer::default()),
            cfg,
            RequestLimits::default(),
            Arc::new(Admission::legacy(None)),
            RuntimeState {
                draining,
                worker_alive,
                phase,
                backend_faulted,
                backend_faults,
                telemetry,
                production: None,
            },
        )
        .0
    }

    fn chat_request() -> Request<Body> {
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"tritium","messages":[{"role":"user","content":"1"}]}"#,
            ))
            .unwrap()
    }

    #[test]
    fn qwen_template_preserves_roles_and_adds_generation_prefix() {
        let rendered =
            ChatTemplate::QwenIm.render([("system", " be safe "), ("user", " hello ")].into_iter());
        assert_eq!(
            rendered,
            "<|im_start|>system\nbe safe<|im_end|>\n\
             <|im_start|>user\nhello<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[tokio::test]
    async fn production_backend_latch_fails_health_readiness_and_new_work() {
        let router = latching_router();
        assert_eq!(
            router
                .clone()
                .oneshot(chat_request())
                .await
                .unwrap()
                .status(),
            StatusCode::INTERNAL_SERVER_ERROR,
        );
        for path in ["/healthz", "/readyz", "/v1/chat/completions"] {
            let request = if path == "/v1/chat/completions" {
                chat_request()
            } else {
                Request::get(path).body(Body::empty()).unwrap()
            };
            assert_eq!(
                router.clone().oneshot(request).await.unwrap().status(),
                StatusCode::SERVICE_UNAVAILABLE,
                "{path} must fail closed after a production backend fault",
            );
        }
    }
}
