//! The `tritium-serve` binary: load a GGUF model on a registry backend and serve
//! it over the OpenAI HTTP/SSE protocol. Only built with `--features serve`.
//!
//! Usage: `tritium-serve --model <path.gguf> [--backend cpu|cuda] [--port 8080]
//! [--model-id tritium] [--max-new 256] [--eos 128001]`. `--backend cuda` needs
//! the `cuda` cargo feature (links tritium-cuda). Text input requires integer
//! token IDs (the v0.80 id-passthrough tokenizer); inject a real BPE for prose.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tritium_serve::{IdPassthroughTokenizer, RunnerGenerator, ServeConfig, build_router};

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
    let mut eos: u32 = 128_001;
    let mut raw_tokens = false;
    let mut draft_model: Option<String> = None;

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
            "--eos" => eos = val(args.next(), "--eos")?,
            "--raw-tokens" => raw_tokens = true,
            "--draft-model" => draft_model = Some(val(args.next(), "--draft-model")?),
            "-h" | "--help" => {
                eprintln!(
                    "usage: tritium-serve --model <gguf> [--backend cpu|cuda] [--spec lookup] \
                     [--batch-slots N] [--host 127.0.0.1] [--port 8080] [--model-id tritium] \
                     [--max-new 256] [--eos 128001] [--raw-tokens] \
                     [--draft-model <gguf>]  (non-loopback --host requires \
                     TRITIUM_AUTH_TOKEN)"
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
    let cfg = ServeConfig {
        model_id,
        queue_cap: 32,
        max_new_default: max_new,
        auth_token,
        chat_template,
        ..ServeConfig::default()
    };
    // ADR 0021 model drafter: a SECOND runner in-process on the same device
    // (its own KV + decode graphs). Load failures are loud — a missing or
    // unloadable draft must never silently degrade to lookup drafting. A
    // device-OOM here also fails at startup, not mid-request.
    let draft_runner = match &draft_model {
        None => None,
        Some(path) => {
            eprintln!("tritium-serve: loading draft model {path} ...");
            let dbytes = std::fs::read(path).map_err(|e| format!("--draft-model {path}: {e}"))?;
            let dfile = tritium_format::read_gguf(&dbytes)
                .map_err(|e| format!("--draft-model {path}: {e}"))?;
            let dbackend = init().map_err(|e| format!("--draft-model backend init: {e}"))?;
            Some(
                tritium_nn::ModelRunner::load(&dfile, &dbytes, dbackend)
                    .map_err(|e| format!("--draft-model {path}: {e}"))?,
            )
        }
    };
    #[cfg(feature = "cuda")]
    let (router, draining) = if batch_slots > 1 {
        // Continuous batching: a dedicated worker owns the runner + a fixed
        // slot pool; requests stream through the same job queue + SSE plumbing.
        if spec_lookup || draft_runner.is_some() {
            return Err(
                "--spec lookup / --draft-model and --batch-slots > 1 are mutually \
                        exclusive (the spec loop owns the single-sequence KV)"
                    .into(),
            );
        }
        tritium_serve::build_router_batched(runner, eos, batch_slots, tok, cfg)?
    } else {
        let mut generator = RunnerGenerator::new(runner, eos).with_spec_lookup(spec_lookup);
        if let Some(d) = draft_runner {
            generator = generator.with_draft_model(d);
        }
        build_router(Box::new(generator), tok, cfg)
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
        build_router(Box::new(generator), tok, cfg)
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
