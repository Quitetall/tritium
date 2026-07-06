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
    let mut port: u16 = 8080;
    let mut model_id = "tritium".to_owned();
    let mut max_new: usize = 256;
    let mut eos: u32 = 128_001;

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
            "--port" => port = val(args.next(), "--port")?,
            "--model-id" => model_id = val::<String>(args.next(), "--model-id")?,
            "--max-new" => max_new = val(args.next(), "--max-new")?,
            "--eos" => eos = val(args.next(), "--eos")?,
            "-h" | "--help" => {
                eprintln!(
                    "usage: tritium-serve --model <gguf> [--backend cpu|cuda] [--port 8080] \
                     [--model-id tritium] [--max-new 256] [--eos 128001]"
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
    let generator = Box::new(RunnerGenerator::new(runner, eos));
    let tok = Arc::new(IdPassthroughTokenizer::new(128_000, eos));
    let cfg = ServeConfig {
        model_id,
        queue_cap: 32,
        max_new_default: max_new,
    };
    let (router, draining) = build_router(generator, tok, cfg);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
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
