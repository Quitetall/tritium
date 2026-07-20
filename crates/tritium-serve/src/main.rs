//! The `tritium-serve` binary: load a GGUF model on a registry backend and serve
//! it over the OpenAI HTTP/SSE protocol. Only built with `--features serve`.
//!
//! Usage: `tritium-serve --model <path.gguf> [--backend cpu|cuda] [--port 8080]
//! [--model-id tritium] [--max-new 256] [--eos 128001]`. `--backend cuda` needs
//! the `cuda` cargo feature (links tritium-cuda). Text input requires integer
//! token IDs (the v0.80 id-passthrough tokenizer); inject a real BPE for prose.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tritium_serve::{IdPassthroughTokenizer, RequestLimits, RunnerGenerator, ServeConfig};

// Force-link the backends so their `linkme` registrations populate the runtime
// registry consulted below.
use tritium_cpu as _;
#[cfg(feature = "cuda")]
use tritium_cuda as _;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_path: Option<String> = None;
    let mut backend_name = "cpu".to_owned();
    let mut spec: Option<String> = None;
    let mut batch_slots: usize = 1;
    let mut host = "127.0.0.1".to_owned();
    let mut port: u16 = 8080;
    let mut model_id = "tritium".to_owned();
    let mut max_new: usize = 256;
    let mut max_messages: usize = 128;
    let mut max_prompt_bytes: usize = 1024 * 1024;
    let mut max_prompt_tokens: usize = 128 * 1024;
    let mut max_new_tokens: usize = 4096;
    let mut max_total_tokens: usize = 128 * 1024;
    let mut eos: u32 = 128_001;
    let mut raw_tokens = false;
    let mut draft_model: Option<String> = None;
    let mut kv_pool_tokens: Option<usize> = None;

    // Parse a required value for `name`, erroring (not silently defaulting) on a
    // missing or malformed value.
    fn val<T: std::str::FromStr>(
        v: Option<String>,
        name: &str,
    ) -> Result<T, Box<dyn std::error::Error>> {
        let s = v.ok_or_else(|| format!("{name} requires a value"))?;
        s.parse::<T>()
            .map_err(|_| format!("{name}: invalid value {s:?}").into())
    }

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model_path = Some(val::<String>(args.next(), "--model")?),
            "--backend" => backend_name = val::<String>(args.next(), "--backend")?,
            "--spec" => spec = Some(val::<String>(args.next(), "--spec")?),
            "--batch-slots" => batch_slots = val::<usize>(args.next(), "--batch-slots")?,
            "--host" => host = val::<String>(args.next(), "--host")?,
            "--port" => port = val(args.next(), "--port")?,
            "--model-id" => model_id = val::<String>(args.next(), "--model-id")?,
            "--max-new" => max_new = val(args.next(), "--max-new")?,
            "--max-messages" => max_messages = val(args.next(), "--max-messages")?,
            "--max-prompt-bytes" => {
                max_prompt_bytes = val(args.next(), "--max-prompt-bytes")?;
            }
            "--max-prompt-tokens" => {
                max_prompt_tokens = val(args.next(), "--max-prompt-tokens")?;
            }
            "--max-completion-tokens" => {
                max_new_tokens = val(args.next(), "--max-completion-tokens")?;
            }
            "--max-total-tokens" => {
                max_total_tokens = val(args.next(), "--max-total-tokens")?;
            }
            "--eos" => eos = val(args.next(), "--eos")?,
            "--raw-tokens" => raw_tokens = true,
            "--draft-model" => draft_model = Some(val(args.next(), "--draft-model")?),
            "--kv-pool-tokens" => {
                let t: usize = val(args.next(), "--kv-pool-tokens")?;
                if t == 0 {
                    return Err("--kv-pool-tokens must be >= 1".into());
                }
                kv_pool_tokens = Some(t);
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: tritium-serve --model <gguf> [--backend cpu|cuda] [--spec lookup] \
                     [--batch-slots N] [--host 127.0.0.1] [--port 8080] [--model-id tritium] \
                     [--max-new 256] [--max-messages 128] [--max-prompt-bytes 1048576] \
                     [--max-prompt-tokens 131072] [--max-completion-tokens 4096] \
                     [--max-total-tokens 131072] [--eos 128001] [--raw-tokens] \
                     [--draft-model <gguf>] [--kv-pool-tokens N]  (non-loopback \
                     --host requires TRITIUM_AUTH_TOKEN)"
                );
                return Ok(());
            }
            other => eprintln!("tritium-serve: ignoring unknown arg {other:?}"),
        }
    }

    let model_path = model_path.ok_or("missing --model <path-to-gguf>")?;
    eprintln!("tritium-serve: loading {model_path} on the `{backend_name}` backend...");
    let bytes = std::fs::read(&model_path)?;
    // Resolve the named backend from the runtime registry (the same owned-init
    // pattern the acceptance tests use). `cpu` is always linked; `cuda` needs
    // the `cuda` cargo feature and a working device.
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|e| e.name == backend_name)
        .map(|e| e.init)
        .ok_or_else(|| {
            format!(
                "backend `{backend_name}` is not in the registry (linked backends: {}); \
                 for cuda, build with `--features cuda`",
                tritium_runtime::BACKENDS
                    .iter()
                    .map(|e| e.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
    let backend = init().map_err(|e| format!("backend `{backend_name}` failed to init: {e}"))?;
    let file = tritium_format::read_gguf(&bytes)?;
    let runner = tritium_nn::ModelRunner::load(&file, &bytes, backend)?;
    let spec_lookup = match spec.as_deref() {
        None => false,
        Some("lookup") => true,
        Some(other) => return Err(format!("--spec: unknown mode {other:?} (try `lookup`)").into()),
    };
    // Real text mode by default: rebuild the byte-level BPE the GGUF embeds
    // (vocab + merges + control tokens) and render chat via the official
    // template. `--raw-tokens` (or a GGUF without a gpt2 tokenizer) falls back
    // to the id-passthrough MVP: whitespace-separated integer token ids.
    let mut chat_template = tritium_serve::ChatTemplate::Concat;
    let tok: Arc<dyn tritium_nn::Tokenizer + Send + Sync> = if raw_tokens {
        eprintln!("tritium-serve: --raw-tokens — id-passthrough tokenizer");
        Arc::new(IdPassthroughTokenizer::new(128_000, eos))
    } else {
        match tritium_nn::GgufBpeTokenizer::from_gguf(&file) {
            Ok(t) => {
                eos = tritium_nn::Tokenizer::eos(&t);
                chat_template = tritium_serve::ChatTemplate::RoleEot;
                eprintln!(
                    "tritium-serve: GGUF-embedded BPE tokenizer (eos {eos}); chat template: role-eot"
                );
                Arc::new(t)
            }
            Err(e) => {
                eprintln!(
                    "tritium-serve: no usable embedded tokenizer ({e}); falling back to \
                     id-passthrough (send integer token ids)"
                );
                Arc::new(IdPassthroughTokenizer::new(128_000, eos))
            }
        }
    };
    // Binding beyond loopback requires a bearer token (TRITIUM_AUTH_TOKEN):
    // the server is otherwise an unauthenticated code-adjacent surface.
    let host_ip: std::net::IpAddr = host.parse().map_err(|e| format!("--host {host:?}: {e}"))?;
    let auth_token = std::env::var("TRITIUM_AUTH_TOKEN")
        .ok()
        .filter(|t| !t.is_empty());
    if !host_ip.is_loopback() {
        if auth_token.is_none() {
            return Err(format!(
                "--host {host} binds beyond loopback; set TRITIUM_AUTH_TOKEN to require \
                 `Authorization: Bearer <token>` on every request (refusing to serve \
                 an open endpoint)"
            )
            .into());
        }
        eprintln!(
            "tritium-serve: WARNING — binding {host} (non-loopback). Bearer auth is \
             enforced; requests time out after 600s to first byte; body limit 2 MiB."
        );
    }
    if kv_pool_tokens.is_some() && batch_slots <= 1 {
        return Err(
            "--kv-pool-tokens requires --batch-slots > 1 (paged KV is the \
                    batched worker's pool)"
                .into(),
        );
    }
    if [
        max_new,
        max_messages,
        max_prompt_bytes,
        max_prompt_tokens,
        max_new_tokens,
        max_total_tokens,
    ]
    .contains(&0)
    {
        return Err("all request and prompt limits must be >= 1".into());
    }
    if max_new > max_new_tokens {
        return Err("--max-new must not exceed --max-completion-tokens".into());
    }
    if max_new > max_total_tokens {
        return Err("--max-new must not exceed --max-total-tokens".into());
    }
    let cfg = ServeConfig {
        model_id,
        queue_cap: 32,
        max_new_default: max_new,
        auth_token,
        kv_pool_tokens,
        chat_template,
        ..ServeConfig::default()
    };
    let request_limits = RequestLimits {
        max_messages,
        max_prompt_bytes,
        max_prompt_tokens,
        max_new_tokens,
        max_total_tokens,
    };
    // ADR 0021 model drafter: a SECOND runner in-process on the same device
    // (its own KV + decode graphs). Load failures are loud — a missing or
    // unloadable draft must never silently degrade to lookup drafting. A
    // device-OOM here also fails at startup, not mid-request.
    let draft_runner = match &draft_model {
        None => None,
        Some(path) => {
            // The spec path needs the CUDA resident decoder; a draft that can
            // never be consulted must not load silently (stated contract).
            if backend_name != "cuda" {
                return Err(format!(
                    "--draft-model requires --backend cuda (spec decoding runs on the \
                     resident decoder); got --backend {backend_name}"
                )
                .into());
            }
            eprintln!("tritium-serve: loading draft model {path} ...");
            let dbytes = std::fs::read(path).map_err(|e| format!("--draft-model {path}: {e}"))?;
            let dfile = tritium_format::read_gguf(&dbytes)
                .map_err(|e| format!("--draft-model {path}: {e}"))?;
            let dbackend = init().map_err(|e| format!("--draft-model backend init: {e}"))?;
            let d = tritium_nn::ModelRunner::load(&dfile, &dbytes, dbackend)
                .map_err(|e| format!("--draft-model {path}: {e}"))?;
            // A vocab mismatch degrades to silent per-call draft failures —
            // reject at startup instead.
            if d.weights.vocab != runner.weights.vocab {
                return Err(format!(
                    "--draft-model vocab {} != target vocab {} (the drafter must share \
                     the target's tokenizer, ADR 0021)",
                    d.weights.vocab, runner.weights.vocab
                )
                .into());
            }
            Some(d)
        }
    };
    #[cfg(feature = "cuda")]
    let (router, draining) = if batch_slots > 1 {
        // Continuous batching: a dedicated worker owns the runner + a fixed
        // slot pool; requests stream through the same job queue + SSE plumbing.
        if spec_lookup || draft_runner.is_some() {
            return Err(
                "--spec lookup / --draft-model and --batch-slots > 1 are mutually \
                 exclusive (the IN-PROCESS spec loop drives a single sequence; \
                 batched spec decoding is served to EXTERNAL drafters via the \
                 /v1/tree endpoints, which DO work with --batch-slots — C4)"
                    .into(),
            );
        }
        tritium_serve::build_router_batched_with_limits(
            runner,
            eos,
            batch_slots,
            tok,
            cfg,
            request_limits,
        )?
    } else {
        let mut generator = RunnerGenerator::new(runner, eos).with_spec_lookup(spec_lookup);
        if let Some(d) = draft_runner {
            generator = generator.with_draft_model(d);
        }
        tritium_serve::build_router_with_limits(Box::new(generator), tok, cfg, request_limits)
    };
    #[cfg(not(feature = "cuda"))]
    let (router, draining) = {
        if batch_slots > 1 {
            return Err("--batch-slots > 1 requires the cuda feature".into());
        }
        let mut generator = RunnerGenerator::new(runner, eos).with_spec_lookup(spec_lookup);
        if let Some(d) = draft_runner {
            generator = generator.with_draft_model(d);
        }
        tritium_serve::build_router_with_limits(Box::new(generator), tok, cfg, request_limits)
    };

    let addr = std::net::SocketAddr::new(host_ip, port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("tritium-serve: listening on http://{addr}/v1 (Ctrl-C to drain + stop)");
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal(draining))
        .await?;
    Ok(())
}

/// Resolve when Ctrl-C (or SIGTERM on unix) arrives, flagging `draining` so
/// in-flight SSE streams close cleanly and new requests get 503.
async fn shutdown_signal(draining: Arc<AtomicBool>) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {}
        () = term => {}
    }
    eprintln!("tritium-serve: draining...");
    draining.store(true, Ordering::Relaxed);
}
