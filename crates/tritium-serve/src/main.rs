//! The `tritium-serve` binary: load a strict schema-v3 Qwen bundle or an
//! explicitly legacy GGUF model and serve OpenAI HTTP/SSE.
//!
//! Production usage: `tritium-serve --bundle <schema-v3-dir> --profile
//! compact-v1 [--backend cpu|cuda]`. `--model <path.gguf>` retains compatibility
//! serving but cannot satisfy production readiness.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;
use tritium_serve::{
    AdmissionPolicy, IdPassthroughTokenizer, PrincipalRateLimit, RequestLimits, RunnerGenerator,
    ServeConfig,
};

// Force-link the backends so their `linkme` registrations populate the runtime
// registry consulted below.
use tritium_cpu as _;
#[cfg(feature = "cuda")]
use tritium_cuda as _;

const DESTRUCTIVE_CUDA_LOSS_ENV: &str = "TRITIUM_DESTRUCTIVE_CUDA_LOSS_QUALIFICATION";

/// JSON/`TRITIUM_*` launch overlay. Secrets intentionally stay environment-only.
/// `deny_unknown_fields` makes deployment typos fail before model/device init.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct FileConfig {
    model: Option<String>,
    bundle: Option<String>,
    profile: Option<String>,
    backend: Option<String>,
    spec: Option<String>,
    batch_slots: Option<usize>,
    queue_cap: Option<usize>,
    host: Option<String>,
    port: Option<u16>,
    admin_host: Option<String>,
    admin_port: Option<u16>,
    model_id: Option<String>,
    max_new: Option<usize>,
    max_messages: Option<usize>,
    max_prompt_bytes: Option<usize>,
    max_prompt_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    max_total_tokens: Option<usize>,
    rate_limit_rpm: Option<u32>,
    rate_limit_burst: Option<u32>,
    eos: Option<u32>,
    raw_tokens: Option<bool>,
    draft_model: Option<String>,
    kv_pool_tokens: Option<usize>,
}

#[derive(Debug)]
struct LaunchConfig {
    model_path: Option<String>,
    bundle_path: Option<String>,
    profile: String,
    backend_name: String,
    spec: Option<String>,
    batch_slots: usize,
    queue_cap: usize,
    host: String,
    port: u16,
    admin_host: String,
    admin_port: Option<u16>,
    model_id: String,
    max_new: usize,
    max_messages: usize,
    max_prompt_bytes: usize,
    max_prompt_tokens: usize,
    max_new_tokens: usize,
    max_total_tokens: usize,
    rate_limit_rpm: u32,
    rate_limit_burst: u32,
    eos: u32,
    raw_tokens: bool,
    draft_model: Option<String>,
    kv_pool_tokens: Option<usize>,
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            bundle_path: None,
            profile: "compact-v1".to_owned(),
            backend_name: "cpu".to_owned(),
            spec: None,
            batch_slots: 1,
            queue_cap: 32,
            host: "127.0.0.1".to_owned(),
            port: 8080,
            admin_host: "127.0.0.1".to_owned(),
            admin_port: None,
            model_id: "tritium".to_owned(),
            max_new: 256,
            max_messages: 128,
            max_prompt_bytes: 1024 * 1024,
            max_prompt_tokens: 128 * 1024,
            max_new_tokens: 4096,
            max_total_tokens: 128 * 1024,
            rate_limit_rpm: 120,
            rate_limit_burst: 8,
            eos: 128_001,
            raw_tokens: false,
            draft_model: None,
            kv_pool_tokens: None,
        }
    }
}

fn config_path(args: &[String]) -> Result<Option<String>, String> {
    let mut from_args = None;
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--config" {
            if from_args.is_some() {
                return Err("--config may be specified only once".into());
            }
            let value = args
                .get(index + 1)
                .ok_or_else(|| "--config requires a value".to_owned())?;
            from_args = Some(value.clone());
            index += 1;
        }
        index += 1;
    }
    Ok(from_args.or_else(|| std::env::var("TRITIUM_CONFIG").ok()))
}

fn read_config(args: &[String]) -> Result<FileConfig, Box<dyn std::error::Error>> {
    let Some(path) = config_path(args)? else {
        return Ok(FileConfig::default());
    };
    let bytes = std::fs::read(&path).map_err(|error| format!("--config {path:?}: {error}"))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("--config {path:?}: invalid JSON/config: {error}").into())
}

fn parse_env<T: std::str::FromStr>(name: &str) -> Result<Option<T>, Box<dyn std::error::Error>>
where
    T::Err: std::fmt::Display,
{
    let Some(value) = std::env::var_os(name) else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| format!("{name} is not valid UTF-8"))?;
    value
        .parse::<T>()
        .map(Some)
        .map_err(|error| format!("{name}: invalid value {value:?}: {error}").into())
}

fn parse_env_bool(name: &str) -> Result<Option<bool>, Box<dyn std::error::Error>> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| format!("{name} is not valid UTF-8"))?;
    match value.as_str() {
        "1" | "true" | "TRUE" => Ok(Some(true)),
        "0" | "false" | "FALSE" => Ok(Some(false)),
        _ => Err(format!("{name}: expected one of 1/0/true/false, got {value:?}").into()),
    }
}

fn apply_file_config(config: FileConfig, launch: &mut LaunchConfig) {
    if let Some(value) = config.model {
        launch.model_path = Some(value);
    }
    if let Some(value) = config.bundle {
        launch.bundle_path = Some(value);
    }
    if let Some(value) = config.profile {
        launch.profile = value;
    }
    if let Some(value) = config.backend {
        launch.backend_name = value;
    }
    if let Some(value) = config.spec {
        launch.spec = Some(value);
    }
    if let Some(value) = config.batch_slots {
        launch.batch_slots = value;
    }
    if let Some(value) = config.queue_cap {
        launch.queue_cap = value;
    }
    if let Some(value) = config.host {
        launch.host = value;
    }
    if let Some(value) = config.port {
        launch.port = value;
    }
    if let Some(value) = config.admin_host {
        launch.admin_host = value;
    }
    if let Some(value) = config.admin_port {
        launch.admin_port = Some(value);
    }
    if let Some(value) = config.model_id {
        launch.model_id = value;
    }
    if let Some(value) = config.max_new {
        launch.max_new = value;
    }
    if let Some(value) = config.max_messages {
        launch.max_messages = value;
    }
    if let Some(value) = config.max_prompt_bytes {
        launch.max_prompt_bytes = value;
    }
    if let Some(value) = config.max_prompt_tokens {
        launch.max_prompt_tokens = value;
    }
    if let Some(value) = config.max_completion_tokens {
        launch.max_new_tokens = value;
    }
    if let Some(value) = config.max_total_tokens {
        launch.max_total_tokens = value;
    }
    if let Some(value) = config.rate_limit_rpm {
        launch.rate_limit_rpm = value;
    }
    if let Some(value) = config.rate_limit_burst {
        launch.rate_limit_burst = value;
    }
    if let Some(value) = config.eos {
        launch.eos = value;
    }
    if let Some(value) = config.raw_tokens {
        launch.raw_tokens = value;
    }
    if let Some(value) = config.draft_model {
        launch.draft_model = Some(value);
    }
    if let Some(value) = config.kv_pool_tokens {
        launch.kv_pool_tokens = Some(value);
    }
}

fn apply_env_config(launch: &mut LaunchConfig) -> Result<(), Box<dyn std::error::Error>> {
    macro_rules! env_string {
        ($name:literal, $target:expr) => {
            if let Some(value) = std::env::var_os($name) {
                $target = Some(
                    value
                        .into_string()
                        .map_err(|_| format!("{} is not valid UTF-8", $name))?,
                );
            }
        };
    }
    macro_rules! env_value {
        ($name:literal, $target:expr) => {
            if let Some(value) = parse_env($name)? {
                $target = value;
            }
        };
    }
    env_string!("TRITIUM_MODEL", launch.model_path);
    env_string!("TRITIUM_BUNDLE", launch.bundle_path);
    if let Some(value) = std::env::var_os("TRITIUM_PROFILE") {
        launch.profile = value
            .into_string()
            .map_err(|_| "TRITIUM_PROFILE is not valid UTF-8")?;
    }
    if let Some(value) = std::env::var_os("TRITIUM_BACKEND") {
        launch.backend_name = value
            .into_string()
            .map_err(|_| "TRITIUM_BACKEND is not valid UTF-8")?;
    }
    env_string!("TRITIUM_SPEC", launch.spec);
    env_value!("TRITIUM_BATCH_SLOTS", launch.batch_slots);
    env_value!("TRITIUM_QUEUE_CAP", launch.queue_cap);
    if let Some(value) = std::env::var_os("TRITIUM_HOST") {
        launch.host = value
            .into_string()
            .map_err(|_| "TRITIUM_HOST is not valid UTF-8")?;
    }
    env_value!("TRITIUM_PORT", launch.port);
    if let Some(value) = std::env::var_os("TRITIUM_ADMIN_HOST") {
        launch.admin_host = value
            .into_string()
            .map_err(|_| "TRITIUM_ADMIN_HOST is not valid UTF-8")?;
    }
    if let Some(value) = parse_env("TRITIUM_ADMIN_PORT")? {
        launch.admin_port = Some(value);
    }
    if let Some(value) = std::env::var_os("TRITIUM_MODEL_ID") {
        launch.model_id = value
            .into_string()
            .map_err(|_| "TRITIUM_MODEL_ID is not valid UTF-8")?;
    }
    env_value!("TRITIUM_MAX_NEW", launch.max_new);
    env_value!("TRITIUM_MAX_MESSAGES", launch.max_messages);
    env_value!("TRITIUM_MAX_PROMPT_BYTES", launch.max_prompt_bytes);
    env_value!("TRITIUM_MAX_PROMPT_TOKENS", launch.max_prompt_tokens);
    env_value!("TRITIUM_MAX_COMPLETION_TOKENS", launch.max_new_tokens);
    env_value!("TRITIUM_MAX_TOTAL_TOKENS", launch.max_total_tokens);
    env_value!("TRITIUM_RATE_LIMIT_RPM", launch.rate_limit_rpm);
    env_value!("TRITIUM_RATE_LIMIT_BURST", launch.rate_limit_burst);
    env_value!("TRITIUM_EOS", launch.eos);
    if let Some(value) = parse_env_bool("TRITIUM_RAW_TOKENS")? {
        launch.raw_tokens = value;
    }
    env_string!("TRITIUM_DRAFT_MODEL", launch.draft_model);
    if let Some(value) = parse_env("TRITIUM_KV_POOL_TOKENS")? {
        launch.kv_pool_tokens = Some(value);
    }
    Ok(())
}

fn destructive_cuda_loss_enabled(
    value: Option<&std::ffi::OsStr>,
    backend: &str,
) -> Result<bool, String> {
    match value {
        None => Ok(false),
        Some(value) if value == "1" && backend == "cuda" => Ok(true),
        Some(value) if value == "1" => Err(format!(
            "{DESTRUCTIVE_CUDA_LOSS_ENV}=1 requires --backend cuda"
        )),
        Some(_) => Err(format!(
            "{DESTRUCTIVE_CUDA_LOSS_ENV} must be unset or exactly `1`"
        )),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration precedence is deliberately visible and deterministic:
    // built-in defaults < strict JSON file < `TRITIUM_*` environment < CLI.
    // Parse overlays before model/backend initialization so malformed deploy
    // configuration fails without allocating device or artifact state.
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let mut launch = LaunchConfig::default();
    apply_file_config(read_config(&raw_args)?, &mut launch);
    apply_env_config(&mut launch)?;
    let LaunchConfig {
        mut model_path,
        mut bundle_path,
        mut profile,
        mut backend_name,
        mut spec,
        mut batch_slots,
        mut queue_cap,
        mut host,
        mut port,
        mut admin_host,
        mut admin_port,
        mut model_id,
        mut max_new,
        mut max_messages,
        mut max_prompt_bytes,
        mut max_prompt_tokens,
        mut max_new_tokens,
        mut max_total_tokens,
        mut rate_limit_rpm,
        mut rate_limit_burst,
        mut eos,
        mut raw_tokens,
        mut draft_model,
        mut kv_pool_tokens,
    } = launch;

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

    let mut args = raw_args.into_iter();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => {
                // Already loaded in the pre-scan. Consume its value so it is
                // not mistaken for a model/backend argument below.
                let _ = val::<String>(args.next(), "--config")?;
            }
            "--model" => model_path = Some(val::<String>(args.next(), "--model")?),
            "--bundle" => bundle_path = Some(val::<String>(args.next(), "--bundle")?),
            "--profile" => profile = val::<String>(args.next(), "--profile")?,
            "--backend" => backend_name = val::<String>(args.next(), "--backend")?,
            "--spec" => spec = Some(val::<String>(args.next(), "--spec")?),
            "--batch-slots" => batch_slots = val::<usize>(args.next(), "--batch-slots")?,
            "--queue-cap" => queue_cap = val::<usize>(args.next(), "--queue-cap")?,
            "--host" => host = val::<String>(args.next(), "--host")?,
            "--port" => port = val(args.next(), "--port")?,
            "--admin-host" => admin_host = val::<String>(args.next(), "--admin-host")?,
            "--admin-port" => admin_port = Some(val(args.next(), "--admin-port")?),
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
                     [--config <strict-json>] \
                     [--batch-slots N] [--queue-cap 32] [--host 127.0.0.1] [--port 8080] \
                     [--admin-host 127.0.0.1] [--admin-port N] \
                     [--model-id tritium] \
                     [--max-new 256] [--max-messages 128] [--max-prompt-bytes 1048576] \
                     [--max-prompt-tokens 131072] [--max-completion-tokens 4096] \
                     [--max-total-tokens 131072] [--rate-limit-rpm 120] \
                     [--rate-limit-burst 8] [--eos 128001] [--raw-tokens] \
                     [--draft-model <gguf>] [--kv-pool-tokens N]  (precedence: defaults < \
                     config < TRITIUM_* < CLI; non-loopback --host requires \
                     TRITIUM_AUTH_TOKEN or TRITIUM_AUTH_TOKENS)"
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
    if queue_cap == 0 {
        return Err("--queue-cap must be >= 1".into());
    }
    // Reject configuration before backend initialization or multi-gigabyte
    // artifact loading. Startup qualification depends on malformed limits
    // failing fast, without touching model/device state.
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
    let destructive_cuda_loss = destructive_cuda_loss_enabled(
        std::env::var_os(DESTRUCTIVE_CUDA_LOSS_ENV).as_deref(),
        &backend_name,
    )?;
    #[cfg(any(not(unix), not(feature = "device-loss-qualification")))]
    if destructive_cuda_loss {
        return Err(format!(
            "{DESTRUCTIVE_CUDA_LOSS_ENV}=1 requires Unix and the `device-loss-qualification` feature"
        )
        .into());
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
        let physical_device = backend.physical_device_id().to_owned();
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
    let admin_ip: std::net::IpAddr = admin_host
        .parse()
        .map_err(|e| format!("--admin-host {admin_host:?}: {e}"))?;
    if admin_port.is_some_and(|port| port == 0) {
        return Err("--admin-port must be >= 1 when enabled".into());
    }
    if admin_port.is_some() && !admin_ip.is_loopback() {
        return Err(format!(
            "--admin-host {admin_host} must be loopback when --admin-port is enabled"
        )
        .into());
    }
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
    let cfg = ServeConfig {
        model_id,
        queue_cap,
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
                if spec_lookup {
                    return Err(
                        "--spec lookup and --batch-slots > 1 are mutually exclusive \
                 (the IN-PROCESS lookup loop drives a single sequence; batched \
                 spec decoding is served to EXTERNAL drafters via the /v1/tree \
                 endpoints, which DO work with --batch-slots — C4)"
                            .into(),
                    );
                }
                // I0 (ADR 0032 L3): --draft-model coexists with --batch-slots
                // under the "spec-when-solo, migrate-on-admission" contract —
                // a solo greedy request decodes speculatively on the
                // single-sequence KV; any admission migrates it into a batch
                // slot and everyone proceeds under the lockstep contract.
                tritium_serve::build_router_batched_governed(
                    *runner,
                    draft_runner,
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
    #[cfg(all(feature = "device-loss-qualification", unix))]
    let destructive_signal = if destructive_cuda_loss {
        Some(
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined2()).map_err(
                |error| format!("install destructive CUDA qualification SIGUSR2 handler: {error}"),
            )?,
        )
    } else {
        None
    };
    let listener = tokio::net::TcpListener::bind(addr).await?;
    #[cfg(all(feature = "device-loss-qualification", unix))]
    if let Some(mut signal) = destructive_signal {
        tokio::spawn(async move {
            if signal.recv().await.is_some() {
                let armed = tritium_cuda::request_destructive_context_loss_for_qualification();
                eprintln!(
                    "tritium-serve: destructive CUDA context-loss qualification SIGUSR2 received; armed={armed}"
                );
            }
        });
    }
    eprintln!("tritium-serve: listening on http://{addr}/v1 (Ctrl-C to drain + stop)");
    let public_server =
        axum::serve(listener, router).with_graceful_shutdown(shutdown_signal(draining.clone()));
    if let Some(admin_port) = admin_port {
        let admin_addr = std::net::SocketAddr::new(admin_ip, admin_port);
        let admin_listener = tokio::net::TcpListener::bind(admin_addr).await?;
        eprintln!(
            "tritium-serve: admin drain listener on http://{admin_addr}/drain (loopback-only)"
        );
        let admin_server = axum::serve(
            admin_listener,
            tritium_serve::build_admin_router(draining.clone()),
        )
        .with_graceful_shutdown(shutdown_signal(draining));
        tokio::try_join!(public_server, admin_server)?;
    } else {
        public_server.await?;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destructive_cuda_loss_gate_is_exact_and_cuda_only() {
        assert_eq!(destructive_cuda_loss_enabled(None, "cpu"), Ok(false));
        assert_eq!(
            destructive_cuda_loss_enabled(Some(std::ffi::OsStr::new("1")), "cuda"),
            Ok(true)
        );
        assert_eq!(
            destructive_cuda_loss_enabled(Some(std::ffi::OsStr::new("1")), "cpu"),
            Err(format!(
                "{DESTRUCTIVE_CUDA_LOSS_ENV}=1 requires --backend cuda"
            ))
        );
        assert_eq!(
            destructive_cuda_loss_enabled(Some(std::ffi::OsStr::new("true")), "cuda"),
            Err(format!(
                "{DESTRUCTIVE_CUDA_LOSS_ENV} must be unset or exactly `1`"
            ))
        );
    }

    #[test]
    fn config_path_rejects_duplicate_cli_flags() {
        let args = vec![
            "--config".to_owned(),
            "one.json".to_owned(),
            "--config".to_owned(),
            "two.json".to_owned(),
        ];
        assert_eq!(
            config_path(&args),
            Err("--config may be specified only once".to_owned())
        );
    }

    #[test]
    fn config_file_rejects_unknown_keys_before_startup() {
        let path = std::env::temp_dir().join(format!(
            "tritium-serve-config-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, br#"{"queue-cap":4,"unknown-key":true}"#).unwrap();
        let args = vec!["--config".to_owned(), path.to_string_lossy().into_owned()];
        let error = read_config(&args).unwrap_err().to_string();
        let _ = std::fs::remove_file(&path);
        assert!(error.contains("unknown field `unknown-key`"), "{error}");
    }
}
