//! The `tritium-serve` binary: load a strict schema-v3 Qwen bundle or an
//! explicitly legacy GGUF model and serve OpenAI HTTP/SSE.
//!
//! Production usage: `tritium-serve --bundle <schema-v3-dir> --profile
//! compact-v1 [--backend cpu|cuda]`. `--model <path.gguf>` retains compatibility
//! serving but cannot satisfy production readiness.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tritium_serve::{
    AdmissionPolicy, IdPassthroughTokenizer, PrincipalRateLimit, RequestLimits, RunnerGenerator,
    ServeConfig,
};

// Force-link the backends so their `linkme` registrations populate the runtime
// registry consulted below.
use tritium_cpu as _;
#[cfg(feature = "cuda")]
use tritium_cuda as _;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_path: Option<String> = None;
    let mut bundle_path: Option<String> = None;
    let mut profile = "compact-v1".to_owned();
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
    let mut rate_limit_rpm: u32 = 120;
    let mut rate_limit_burst: u32 = 8;
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
            "--bundle" => bundle_path = Some(val::<String>(args.next(), "--bundle")?),
            "--profile" => profile = val::<String>(args.next(), "--profile")?,
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
            "--rate-limit-rpm" => {
                rate_limit_rpm = val(args.next(), "--rate-limit-rpm")?;
            }
            "--rate-limit-burst" => {
                rate_limit_burst = val(args.next(), "--rate-limit-burst")?;
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
                    "usage: tritium-serve (--bundle <schema-v3-dir> [--profile compact-v1] | \
                     --model <legacy.gguf>) [--backend cpu|cuda] [--spec lookup] \
                     [--batch-slots N] [--host 127.0.0.1] [--port 8080] [--model-id tritium] \
                     [--max-new 256] [--max-messages 128] [--max-prompt-bytes 1048576] \
                     [--max-prompt-tokens 131072] [--max-completion-tokens 4096] \
                     [--max-total-tokens 131072] [--rate-limit-rpm 120] \
                     [--rate-limit-burst 8] [--eos 128001] [--raw-tokens] \
                     [--draft-model <gguf>] [--kv-pool-tokens N]  (non-loopback \
                     --host requires TRITIUM_AUTH_TOKEN or TRITIUM_AUTH_TOKENS)"
                );
                return Ok(());
            }
            other => return Err(format!("unknown argument {other:?}").into()),
        }
    }

    if model_path.is_some() == bundle_path.is_some() {
        return Err(
            "provide exactly one of --bundle <schema-v3-dir> or --model <legacy.gguf>".into(),
        );
    }
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
    let spec_lookup = match spec.as_deref() {
        None => false,
        Some("lookup") => true,
        Some(other) => return Err(format!("--spec: unknown mode {other:?} (try `lookup`)").into()),
    };
    enum LoadedModel {
        Legacy(Box<tritium_nn::ModelRunner>),
        Production {
            model: Box<tritium_nn::Qwen35SaltV2LanguageMtpModel>,
            physical_device: String,
        },
    }
    let (loaded, tok, chat_template): (
        LoadedModel,
        Arc<dyn tritium_nn::Tokenizer + Send + Sync>,
        tritium_serve::ChatTemplate,
    ) = if let Some(bundle) = bundle_path {
        if raw_tokens
            || spec_lookup
            || draft_model.is_some()
            || batch_slots != 1
            || kv_pool_tokens.is_some()
        {
            return Err("--bundle forbids --raw-tokens, --spec, --draft-model, --batch-slots != 1, and --kv-pool-tokens".into());
        }
        let bundle = std::path::Path::new(&bundle);
        let backend =
            init().map_err(|error| format!("backend `{backend_name}` failed to init: {error}"))?;
        let physical_device = backend.device_id().to_owned();
        eprintln!(
            "tritium-serve: loading strict bundle {} profile {profile} on `{backend_name}` ({physical_device})...",
            bundle.display()
        );
        let model = tritium_nn::Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
            bundle, &profile, backend,
        )?;
        let (model, tokenizer_json, tokenizer_config_json) = model.into_serving_assets();
        let tokenizer =
            tritium_nn::HfJsonTokenizer::from_bytes(&tokenizer_json, &tokenizer_config_json)?;
        eos = tritium_nn::Tokenizer::eos(&tokenizer);
        (
            LoadedModel::Production {
                model: Box::new(model),
                physical_device,
            },
            Arc::new(tokenizer),
            tritium_serve::ChatTemplate::QwenIm,
        )
    } else {
        let model_path = model_path.expect("exactly-one validation established legacy path");
        eprintln!("tritium-serve: loading legacy {model_path} on `{backend_name}`...");
        let bytes = std::fs::read(&model_path)?;
        let backend =
            init().map_err(|error| format!("backend `{backend_name}` failed to init: {error}"))?;
        let file = tritium_format::read_gguf(&bytes)?;
        let runner = tritium_nn::ModelRunner::load(&file, &bytes, backend)?;
        let (tokenizer, template): (Arc<dyn tritium_nn::Tokenizer + Send + Sync>, _) = if raw_tokens
        {
            (
                Arc::new(IdPassthroughTokenizer::new(128_000, eos)),
                tritium_serve::ChatTemplate::Concat,
            )
        } else {
            match tritium_nn::GgufBpeTokenizer::from_gguf(&file) {
                Ok(tokenizer) => {
                    eos = tritium_nn::Tokenizer::eos(&tokenizer);
                    (Arc::new(tokenizer), tritium_serve::ChatTemplate::RoleEot)
                }
                Err(error) => {
                    eprintln!(
                        "tritium-serve: legacy tokenizer unavailable ({error}); using raw token IDs"
                    );
                    (
                        Arc::new(IdPassthroughTokenizer::new(128_000, eos)),
                        tritium_serve::ChatTemplate::Concat,
                    )
                }
            }
        };
        (LoadedModel::Legacy(Box::new(runner)), tokenizer, template)
    };
    // Binding beyond loopback requires at least one bearer key. The singular
    // variable preserves compatibility; the plural comma-separated variable
    // supports bounded rotation without a restart-time authentication gap.
    let host_ip: std::net::IpAddr = host.parse().map_err(|e| format!("--host {host:?}: {e}"))?;
    let mut auth_tokens = Vec::new();
    if let Ok(token) = std::env::var("TRITIUM_AUTH_TOKEN")
        && !token.is_empty()
    {
        auth_tokens.push(token);
    }
    if let Ok(tokens) = std::env::var("TRITIUM_AUTH_TOKENS")
        && !tokens.is_empty()
    {
        for token in tokens.split(',') {
            if token.is_empty() {
                return Err("TRITIUM_AUTH_TOKENS contains an empty comma-separated entry".into());
            }
            auth_tokens.push(token.to_owned());
        }
    }
    if !host_ip.is_loopback() {
        if auth_tokens.is_empty() {
            return Err(format!(
                "--host {host} binds beyond loopback; set TRITIUM_AUTH_TOKEN or \
                 TRITIUM_AUTH_TOKENS to require \
                 `Authorization: Bearer <token>` on every request (refusing to serve \
                 an open endpoint)"
            )
            .into());
        }
        eprintln!(
            "tritium-serve: WARNING — binding {host} (non-loopback). Bearer auth is \
             enforced; requests and streams time out after 600s; body limit 2 MiB."
        );
    }
    if rate_limit_rpm > 0 && rate_limit_burst == 0 {
        return Err("--rate-limit-burst must be >= 1 when rate limiting is enabled".into());
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
        // The v1.1 governed constructor owns rotating authentication. Keep the
        // legacy single-key field empty to make precedence unambiguous.
        auth_token: None,
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
    let admission = AdmissionPolicy {
        bearer_tokens: auth_tokens,
        rate_limit: (rate_limit_rpm > 0).then_some(PrincipalRateLimit {
            requests_per_minute: rate_limit_rpm,
            burst: rate_limit_burst,
        }),
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
            let target_vocab = match &loaded {
                LoadedModel::Legacy(runner) => runner.weights.vocab,
                LoadedModel::Production { .. } => unreachable!("bundle rejected draft model"),
            };
            if d.weights.vocab != target_vocab {
                return Err(format!(
                    "--draft-model vocab {} != target vocab {} (the drafter must share \
                     the target's tokenizer, ADR 0021)",
                    d.weights.vocab, target_vocab
                )
                .into());
            }
            Some(d)
        }
    };
    let (router, draining) = match loaded {
        LoadedModel::Production {
            model,
            physical_device,
        } => {
            let source_identity = option_env!("TRITIUM_SOURCE_ID")
                .ok_or("production binary lacks TRITIUM_SOURCE_ID")?;
            let source_revision = source_identity
                .strip_prefix("source-git:")
                .filter(|revision| {
                    revision.len() == 40
                        && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
                        && revision.bytes().all(|byte| !byte.is_ascii_uppercase())
                })
                .ok_or("production binary source identity is not one clean Git revision")?;
            let build_id = format!(
                "tritium-serve:{}:{source_revision}",
                env!("CARGO_PKG_VERSION")
            );
            let admitted = tritium_serve::admit_qwen36_salt_v3(
                *model,
                eos,
                source_revision,
                &build_id,
                &backend_name,
                &backend_name,
                &physical_device,
            )?;
            let (router, draining, _) = tritium_serve::build_router_production(
                admitted,
                tok,
                cfg,
                request_limits,
                admission,
            )?;
            (router, draining)
        }
        LoadedModel::Legacy(runner) => {
            #[cfg(feature = "cuda")]
            let result = if batch_slots > 1 {
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
                tritium_serve::build_router_batched_governed(
                    *runner,
                    eos,
                    batch_slots,
                    tok,
                    cfg,
                    request_limits,
                    admission,
                )?
            } else {
                let mut generator =
                    RunnerGenerator::new(*runner, eos).with_spec_lookup(spec_lookup);
                if let Some(d) = draft_runner {
                    generator = generator.with_draft_model(d);
                }
                tritium_serve::build_router_governed(
                    Box::new(generator),
                    tok,
                    cfg,
                    request_limits,
                    admission,
                )?
            };
            #[cfg(not(feature = "cuda"))]
            let result = {
                if batch_slots > 1 {
                    return Err("--batch-slots > 1 requires the cuda feature".into());
                }
                let mut generator =
                    RunnerGenerator::new(*runner, eos).with_spec_lookup(spec_lookup);
                if let Some(draft) = draft_runner {
                    generator = generator.with_draft_model(draft);
                }
                tritium_serve::build_router_governed(
                    Box::new(generator),
                    tok,
                    cfg,
                    request_limits,
                    admission,
                )?
            };
            result
        }
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
