//! The axum router: OpenAI `/v1/chat/completions` (non-stream + SSE), `/v1/models`,
//! `/healthz`, with backpressure and a drain flag for graceful shutdown.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::mpsc;
use tritium_nn::Tokenizer;

use crate::dto::{
    ApiError, ChatChunk, ChatCompletion, ChatMessage, ChatRequest, Choice, ModelEntry, ModelList,
    StopField, Usage,
};
use crate::generator::TreeOpError;
use crate::generator::{FinishReason, GenRequest, Generator, Sampling};
use crate::sse::{
    IncrementalDetok, StopMatcher, content_chunk, error_chunk, role_chunk, terminal_chunk,
};
use crate::worker::{GenEvent, Job, spawn_worker};

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
}

impl ChatTemplate {
    /// Render `(role, content)` messages into the prompt string.
    pub fn render<'a>(
        self,
        messages: impl Iterator<Item = (&'a str, &'a str)>,
    ) -> String {
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
    /// Per-request service-future budget. For NON-STREAMING requests this
    /// bounds queue wait + the whole generation. For streaming requests the
    /// service future resolves at headers (the SSE body is lazy), so the
    /// timeout bounds essentially nothing — streaming slow-clients are
    /// instead bounded by the 64-event channel + try_send cancellation.
    /// 0 disables.
    pub request_timeout_secs: u64,
    /// Global in-flight request cap (DoS bound on handler memory/FDs).
    /// 0 disables.
    pub max_concurrent_requests: usize,
    /// When set, every request must carry `Authorization: Bearer <token>`.
    /// Required by `main` when binding beyond loopback.
    pub auth_token: Option<String>,
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
            chat_template: ChatTemplate::default(),
        }
    }
}

/// Serve-level counters for `/metrics` (Prometheus text format, no deps).
/// Counters are monotone; gauges are computed at scrape time.
#[derive(Debug, Default)]
pub(crate) struct Metrics {
    /// Chat completions accepted into the queue (streaming + non-streaming).
    pub(crate) chat_requests: AtomicU64,
    /// 429s returned because the job queue was full.
    pub(crate) queue_rejections: AtomicU64,
    /// Completion tokens emitted to clients across all requests.
    pub(crate) tokens_out: AtomicU64,
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<Job>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    model_id: Arc<str>,
    draining: Arc<AtomicBool>,
    worker_alive: Arc<AtomicBool>,
    max_new_default: usize,
    chat_template: ChatTemplate,
    metrics: Arc<Metrics>,
}

/// Build the router. Returns it plus the drain flag — set the flag (then run
/// axum's graceful shutdown) to stop accepting new work and close in-flight
/// streams cleanly.
pub fn build_router(
    generator: Box<dyn Generator>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
) -> (Router, Arc<AtomicBool>) {
    let draining = Arc::new(AtomicBool::new(false));
    let worker_alive = Arc::new(AtomicBool::new(true));
    let jobs = spawn_worker(
        generator,
        draining.clone(),
        worker_alive.clone(),
        cfg.queue_cap,
    );
    build_router_inner(jobs, tok, cfg, draining, worker_alive)
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
    let alive = worker_alive.clone();
    let drain_flag = draining.clone();
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
            crate::batch::run_batched(runner, eos, slots, jobs_rx, drain_flag);
        })?;
    Ok(build_router_inner(
        jobs_tx,
        tok,
        cfg,
        draining,
        worker_alive,
    ))
}

fn build_router_inner(
    jobs: tokio::sync::mpsc::Sender<crate::worker::Job>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    cfg: ServeConfig,
    draining: Arc<AtomicBool>,
    worker_alive: Arc<AtomicBool>,
) -> (Router, Arc<AtomicBool>) {
    let state = AppState {
        jobs,
        tok,
        model_id: Arc::from(cfg.model_id.as_str()),
        draining: draining.clone(),
        worker_alive,
        max_new_default: cfg.max_new_default,
        chat_template: cfg.chat_template,
        metrics: Arc::new(Metrics::default()),
    };
    let auth_token: Option<Arc<str>> = cfg.auth_token.as_deref().map(Arc::from);
    let mut router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/healthz", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/tree/session", post(tree_session))
        .route("/v1/tree/verify", post(tree_verify))
        .with_state(state)
        // Explicit request-body cap (axum's default, stated rather than
        // implied — threat-model DoS bound).
        .layer(axum::extract::DefaultBodyLimit::max(2 * 1024 * 1024));
    if cfg.request_timeout_secs > 0 {
        // Times the SERVICE FUTURE only: bounds non-streaming requests
        // end-to-end; streaming resolves at headers (lazy SSE body), so this
        // bounds nothing there — see `request_timeout_secs` docs.
        router = router.layer(tower_http::timeout::TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            std::time::Duration::from_secs(cfg.request_timeout_secs),
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
    if let Some(token) = auth_token {
        router = router.layer(axum::middleware::from_fn(
            move |req: axum::extract::Request, next: axum::middleware::Next| {
                let token = token.clone();
                async move {
                    let ok = req
                        .headers()
                        .get(axum::http::header::AUTHORIZATION)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.strip_prefix("Bearer "))
                        .is_some_and(|t| {
                            // Constant-time-ish compare (length + bytes).
                            t.len() == token.len()
                                && t.bytes()
                                    .zip(token.bytes())
                                    .fold(0u8, |acc, (a, b)| acc | (a ^ b))
                                    == 0
                        });
                    if ok {
                        next.run(req).await
                    } else {
                        let mut resp = api_error(
                            StatusCode::UNAUTHORIZED,
                            "invalid_request_error",
                            "missing or invalid bearer token",
                            None,
                        )
                        .into_response();
                        resp.headers_mut().insert(
                            axum::http::header::WWW_AUTHENTICATE,
                            axum::http::HeaderValue::from_static("Bearer"),
                        );
                        resp
                    }
                }
            },
        ));
    }
    (router, draining)
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

async fn chat_completions(State(st): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    if st.draining.load(Ordering::Relaxed) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "server is draining",
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
    let logprobs_k = req
        .logprobs
        .then(|| req.top_logprobs.unwrap_or(0) as usize);
    let gen_req = GenRequest {
        prompt_tokens,
        max_new,
        logprobs: logprobs_k,
        sampling: lower_sampling(&req),
        stop_eos: true,
    };

    let (tx, rx) = mpsc::channel::<GenEvent>(64);
    match st.jobs.try_send(Job::Generate { req: gen_req, tx }) {
        Ok(()) => {
            st.metrics.chat_requests.fetch_add(1, Ordering::Relaxed);
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
            include_usage.then_some(prompt_len),
        )
    } else {
        nonstream_response(rx, st.tok.clone(), req.model, prompt_len, stops, st.metrics.clone())
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
) -> Response {
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

fn stream_response(
    mut rx: mpsc::Receiver<GenEvent>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    model: String,
    stops: Vec<String>,
    metrics: Arc<Metrics>,
    // `Some(prompt_tokens)` when the client asked for
    // `stream_options.include_usage`: emit the final usage chunk.
    usage_prompt_len: Option<usize>,
) -> Response {
    let id = make_id();
    let created = now_secs();
    let detok_eos = tok.eos();
    let stream_tok = tok.clone();
    let stream = async_stream::stream! {
        // 1. role-first chunk
        yield Ok::<Event, std::convert::Infallible>(sse_data(&role_chunk(&id, created, &model)));

        let mut detok = IncrementalDetok::new(tok);
        let mut matcher = StopMatcher::new(stops);
        let mut completion_tokens = 0usize;
        let mut finish = FinishReason::Stop;
        let mut stopped_by_string = false;
        let mut errored = false;

        while let Some(ev) = rx.recv().await {
            match ev {
                GenEvent::Token(t, lp) => {
                    // Count like the non-stream path: eos terminates, it is
                    // not an emitted completion token.
                    if t != detok_eos {
                        metrics.tokens_out.fetch_add(1, Ordering::Relaxed);
                        completion_tokens += 1;
                    }
                    let text = detok.push(t);
                    if !text.is_empty() {
                        let (emit, hit) = matcher.feed(&text);
                        if !emit.is_empty() {
                            let mut chunk = content_chunk(&id, created, &model, &emit);
                            if t != detok_eos
                                && let Some(lp) = lp
                            {
                                chunk.choices[0].logprobs =
                                    Some(render_logprobs(&[lp], stream_tok.as_ref()));
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

        if errored {
            yield Ok(sse_data(&error_chunk(&id, created, &model)));
        } else {
            if !stopped_by_string {
                let tail = matcher.flush();
                if !tail.is_empty() {
                    yield Ok(sse_data(&content_chunk(&id, created, &model, &tail)));
                }
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
    if !st.worker_alive.load(Ordering::Relaxed) {
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
    if st.draining.load(Ordering::Relaxed) {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "status": "draining", "model": &*st.model_id })),
        )
            .into_response()
    } else {
        Json(serde_json::json!({
            "status": "ok",
            "model": &*st.model_id,
            "queue_depth": queue_depth,
        }))
        .into_response()
    }
}

/// Prometheus text exposition (behind the same auth as everything else).
/// Gauges are scrape-time reads; counters live in [`Metrics`].
async fn metrics(State(st): State<AppState>) -> Response {
    let queue_depth = st.jobs.max_capacity() - st.jobs.capacity();
    let body = format!(
        "# HELP tritium_chat_requests_total Chat completions accepted into the queue.\n\
         # TYPE tritium_chat_requests_total counter\n\
         tritium_chat_requests_total {}\n\
         # HELP tritium_queue_rejections_total Requests 429'd because the job queue was full.\n\
         # TYPE tritium_queue_rejections_total counter\n\
         tritium_queue_rejections_total {}\n\
         # HELP tritium_tokens_out_total Completion tokens emitted to clients.\n\
         # TYPE tritium_tokens_out_total counter\n\
         tritium_tokens_out_total {}\n\
         # HELP tritium_queue_depth Jobs waiting in the decode queue.\n\
         # TYPE tritium_queue_depth gauge\n\
         tritium_queue_depth {}\n\
         # HELP tritium_worker_alive Decode worker liveness (1 = alive).\n\
         # TYPE tritium_worker_alive gauge\n\
         tritium_worker_alive {}\n",
        st.metrics.chat_requests.load(Ordering::Relaxed),
        st.metrics.queue_rejections.load(Ordering::Relaxed),
        st.metrics.tokens_out.load(Ordering::Relaxed),
        queue_depth,
        u8::from(st.worker_alive.load(Ordering::Relaxed)),
    );
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        body,
    )
        .into_response()
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

async fn tree_session(State(st): State<AppState>, Json(req): Json<TreeSessionRequest>) -> Response {
    if st.draining.load(std::sync::atomic::Ordering::Relaxed) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "draining",
            "server is draining",
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

async fn tree_verify(State(st): State<AppState>, Json(req): Json<TreeVerifyRequest>) -> Response {
    if st.draining.load(std::sync::atomic::Ordering::Relaxed) {
        return api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "draining",
            "server is draining",
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
        TreeOpError::Unsupported(_) => (StatusCode::NOT_IMPLEMENTED, "tree_unsupported"),
        TreeOpError::Conflict(_) => (StatusCode::CONFLICT, "tree_session_closed"),
        TreeOpError::BadRequest(_) => (StatusCode::BAD_REQUEST, "tree_bad_request"),
        TreeOpError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
    };
    api_error(code, kind, e.to_string(), None)
}
