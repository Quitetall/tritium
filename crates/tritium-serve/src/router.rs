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
use crate::generator::{FinishReason, GenRequest, Generator, Sampling};
use crate::sse::{
    IncrementalDetok, StopMatcher, content_chunk, error_chunk, role_chunk, terminal_chunk,
};
use crate::worker::{GenEvent, Job, spawn_worker};

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// The single served model id (matched against the request `model`).
    pub model_id: String,
    /// Bounded job-queue capacity (backpressure threshold).
    pub queue_cap: usize,
    /// `max_tokens` used when the request omits it.
    pub max_new_default: usize,
}

impl Default for ServeConfig {
    fn default() -> Self {
        Self {
            model_id: "tritium".to_owned(),
            queue_cap: 32,
            max_new_default: 256,
        }
    }
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<Job>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    model_id: Arc<str>,
    draining: Arc<AtomicBool>,
    worker_alive: Arc<AtomicBool>,
    max_new_default: usize,
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
    let state = AppState {
        jobs,
        tok,
        model_id: Arc::from(cfg.model_id.as_str()),
        draining: draining.clone(),
        worker_alive,
        max_new_default: cfg.max_new_default,
    };
    let router = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/healthz", get(health))
        .route("/v1/tree/session", post(tree_session))
        .route("/v1/tree/verify", post(tree_verify))
        .with_state(state);
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

    // MVP prompt build: join message contents (the LLaMA-3 chat template ships
    // with the real BPE tokenizer; the id-passthrough wants integer token IDs).
    let prompt_text = req
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
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
    let gen_req = GenRequest {
        prompt_tokens,
        max_new,
        sampling: lower_sampling(&req),
        stop_eos: true,
    };

    let (tx, rx) = mpsc::channel::<GenEvent>(64);
    match st.jobs.try_send(Job::Generate { req: gen_req, tx }) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(_)) => {
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
        stream_response(rx, st.tok.clone(), req.model, stops)
    } else {
        nonstream_response(rx, st.tok.clone(), req.model, prompt_len, stops).await
    }
}

async fn nonstream_response(
    mut rx: mpsc::Receiver<GenEvent>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    model: String,
    prompt_len: usize,
    stops: Vec<String>,
) -> Response {
    let eos = tok.eos();
    let mut tokens: Vec<u32> = Vec::new();
    let mut finish = FinishReason::Stop;
    while let Some(ev) = rx.recv().await {
        match ev {
            GenEvent::Token(t) => {
                if t != eos {
                    tokens.push(t);
                }
            }
            GenEvent::Done(fr) => {
                finish = fr;
                break;
            }
            GenEvent::Error(e) => {
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e, None);
            }
        }
    }
    let mut text = tok.decode(&tokens).unwrap_or_default();
    let mut fr = finish;
    // Truncate at the EARLIEST stop match across all sequences (order-independent —
    // matches the streaming StopMatcher's `.min()` semantics).
    if let Some(pos) = stops.iter().filter_map(|s| text.find(s.as_str())).min() {
        text.truncate(pos);
        fr = FinishReason::Stop;
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
        }],
        usage: Usage {
            prompt_tokens: prompt_len,
            completion_tokens: tokens.len(),
            total_tokens: prompt_len + tokens.len(),
        },
    };
    Json(completion).into_response()
}

fn sse_data(chunk: &ChatChunk) -> Event {
    Event::default().data(serde_json::to_string(chunk).unwrap_or_default())
}

fn stream_response(
    mut rx: mpsc::Receiver<GenEvent>,
    tok: Arc<dyn Tokenizer + Send + Sync>,
    model: String,
    stops: Vec<String>,
) -> Response {
    let id = make_id();
    let created = now_secs();
    let stream = async_stream::stream! {
        // 1. role-first chunk
        yield Ok::<Event, std::convert::Infallible>(sse_data(&role_chunk(&id, created, &model)));

        let mut detok = IncrementalDetok::new(tok);
        let mut matcher = StopMatcher::new(stops);
        let mut finish = FinishReason::Stop;
        let mut stopped_by_string = false;
        let mut errored = false;

        while let Some(ev) = rx.recv().await {
            match ev {
                GenEvent::Token(t) => {
                    let text = detok.push(t);
                    if !text.is_empty() {
                        let (emit, hit) = matcher.feed(&text);
                        if !emit.is_empty() {
                            yield Ok(sse_data(&content_chunk(&id, created, &model, &emit)));
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
    if !st.worker_alive.load(Ordering::Relaxed) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "status": "unhealthy", "detail": "decode worker stopped" })),
        )
            .into_response();
    }
    if st.draining.load(Ordering::Relaxed) {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({ "status": "draining" })),
        )
            .into_response()
    } else {
        Json(serde_json::json!({ "status": "ok" })).into_response()
    }
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
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "draining", "server is draining", None);
    }
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    if st
        .jobs
        .try_send(Job::OpenTreeSession {
            prompt: req.prompt_tokens,
            resp: resp_tx,
        })
        .is_err()
    {
        return api_error(StatusCode::TOO_MANY_REQUESTS, "rate_limit_exceeded", "decode queue full; retry shortly", None);
    }
    match resp_rx.await {
        Ok(Ok(pending)) => Json(serde_json::json!({ "pending_token": pending })).into_response(),
        Ok(Err(e)) => tree_error(&e),
        Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", "worker dropped the request", None),
    }
}

async fn tree_verify(State(st): State<AppState>, Json(req): Json<TreeVerifyRequest>) -> Response {
    if st.draining.load(std::sync::atomic::Ordering::Relaxed) {
        return api_error(StatusCode::SERVICE_UNAVAILABLE, "draining", "server is draining", None);
    }
    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    if st
        .jobs
        .try_send(Job::TreeVerify {
            tokens: req.tokens,
            parents: req.parents,
            resp: resp_tx,
        })
        .is_err()
    {
        return api_error(StatusCode::TOO_MANY_REQUESTS, "rate_limit_exceeded", "decode queue full; retry shortly", None);
    }
    match resp_rx.await {
        Ok(Ok(committed)) => Json(serde_json::json!({ "committed": committed })).into_response(),
        Ok(Err(e)) => tree_error(&e),
        Err(_) => api_error(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", "worker dropped the request", None),
    }
}

/// Map worker-side tree errors: capability refusals → 501, everything else 400
/// (malformed trees) — the strings originate from `Generator`'s defaults and
/// `tree_verify_greedy`'s validation.
fn tree_error(msg: &str) -> Response {
    let code = if msg.contains("not supported") || msg.contains("needs the") {
        StatusCode::NOT_IMPLEMENTED
    } else if msg.contains("no open tree session") {
        StatusCode::CONFLICT
    } else {
        StatusCode::BAD_REQUEST
    };
    api_error(code, "tree_verify_error", msg, None)
}
