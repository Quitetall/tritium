//! `tritium generate`: load a model and greedily decode tokens.
//!
//! The subcommand is deterministic and offline-reproducible: generation is greedy, and the output
//! is the list of newly generated token IDs printed one per line (plus a JSON array for easy
//! machine consumption). The model is loaded through [`tritium_nn::ModelRunner`], which selects a
//! backend via the runtime registry.
//!
//! # Two kinds of model, because a converter that produces nothing runnable is half a feature
//!
//! `--model` accepts either a `.gguf` file or a **directory written by `tritium convert`** (ADR
//! 0038 WS-4). Until this existed, `convert` wrote an artifact that only the library could load:
//! `generate` was GGUF-only and `report` takes fp masters, so the adoption spine stopped one step
//! short of running anything.
//!
//! A converted directory is recognised by the presence of `model.tslb` beside `config.json`, and
//! loaded with [`ModelRunner::from_salt`]. Dispatching on *content* rather than on the `--model`
//! spelling means a mistyped path fails as "not a model" instead of being silently treated as the
//! other kind.
//!
//! Explicit original-captured `.trit` generation additionally admits the supplied B stage and
//! mandatory capture/transform/conversion logs before using the common CPU runner. It does not
//! infer ancestry, a model level, a payload budget or an EOS policy from nearby files.
//!
//! # Prompts
//!
//! `--tokens <file.json>` remains the reproducible path: exact ids in, exact ids out, no tokenizer
//! in the loop. `--prompt "..."` is the ergonomic one, and it uses the **model's own** tokenizer —
//! `tokenizer.json` from a converted directory (which `convert` copies for exactly this reason) or
//! the BPE embedded in a GGUF. Ids are only meaningful relative to the vocabulary that produced
//! them, so there is no default tokenizer to fall back on; if none is available, the command says
//! so rather than emitting confident nonsense.
//!
//! Like the rest of the CLI, every failure flows through [`anyhow::Result`]: a missing model, an
//! unreadable token file, malformed JSON, or a model that is missing weights all yield a clean
//! message and a non-zero exit — never a panic.

use std::path::{Path, PathBuf};

use crate::stage_input::{DEFAULT_MAX_JSON_BYTES, StageArgs, read_bounded, required};
use anyhow::{Context as _, bail};
use clap::Args;
use tritium_nn::{GgufBpeTokenizer, HfJsonTokenizer, ModelRunner, Runner, Tokenizer};

// Private bridge for legacy GGUF/SALT commands until their loader migration.
enum CliRunner {
    Legacy(Box<ModelRunner>),
    Shared(Box<Runner>),
    Bounded(Box<Runner>),
}
impl From<ModelRunner> for CliRunner {
    fn from(runner: ModelRunner) -> Self {
        Self::Legacy(Box::new(runner))
    }
}
impl From<Runner> for CliRunner {
    fn from(runner: Runner) -> Self {
        Self::Shared(Box::new(runner))
    }
}
impl CliRunner {
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: u32,
    ) -> Result<Vec<u32>, tritium_nn::NnError> {
        match self {
            Self::Legacy(runner) => runner.generate(prompt, max_new, eos),
            Self::Shared(runner) => runner.generate(prompt, max_new, eos),
            Self::Bounded(runner) => runner.generate_bounded(prompt, max_new, eos),
        }
    }
}

/// Explicit scalar-host QTEA continuation; shares the request with serving.
#[cfg(feature = "host-qtea")]
pub(crate) fn run_qtea(
    path: &Path,
    backend_name: &str,
    token_file: Option<&Path>,
    text: Option<&str>,
    max_new: usize,
    greedy: bool,
    eos: u32,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        backend_name == "cpu",
        "--qtea requires --backend cpu; no GPU or resident fallback"
    );
    anyhow::ensure!(
        greedy,
        "--greedy=false: sampling is not implemented in this subcommand"
    );
    let config = tritium_nn::QteaLoadConfig::read(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    anyhow::ensure!(
        max_new > 0 && max_new <= config.stream_limits().max_context,
        "--qtea --max-new must be positive and fit admitted context"
    );
    let ids = token_file
        .map(|path| -> anyhow::Result<Vec<u32>> {
            let bytes = read_bounded(path, DEFAULT_MAX_JSON_BYTES)?;
            serde_json::from_slice(&bytes).context("QTEA prompt token IDs")
        })
        .transpose()?;
    let prompt = match (&ids, text) {
        (Some(ids), None) => Prompt::Ids(ids),
        (None, Some(text)) => Prompt::Text(text),
        _ => bail!("QTEA generation requires exactly one of --tokens or --prompt"),
    };
    let model_path = config.path().to_path_buf();
    let mut recorder = tritium_evidence::EvidenceRecorder::new(
        "qtea-generation-load",
        [tritium_evidence::MODEL_LOADED_REGISTRATION],
    )?;
    let loaded = config
        .load(tritium_runtime::TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        })
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    eprintln!(
        "QTEA scalar-host candidate, not production-qualified: {}",
        serde_json::to_string(loaded.receipt.observation())?
    );
    recorder.seal()?;
    decode_and_print(
        CliRunner::Bounded(Box::new(loaded.runner)),
        Some(Box::new(loaded.tokenizer)),
        &model_path,
        &prompt,
        max_new,
        eos,
    )
}

#[cfg(not(feature = "host-qtea"))]
pub(crate) fn run_qtea(
    _path: &Path,
    _backend_name: &str,
    _token_file: Option<&Path>,
    _text: Option<&str>,
    _max_new: usize,
    _greedy: bool,
    _eos: u32,
) -> anyhow::Result<()> {
    bail!("--qtea requires build feature host-qtea")
}

/// Explicit typed generation; same load JSON and common streaming path as serving.
pub(crate) fn run_typed(
    path: &Path,
    backend_name: &str,
    token_file: Option<&Path>,
    text: Option<&str>,
    max_new: usize,
    greedy: bool,
    eos: u32,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        greedy,
        "--greedy=false: sampling is not implemented in this subcommand"
    );
    let config = tritium_nn::TypedLoadConfig::read(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    config
        .require_encoded_streaming()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let limits = config
        .stream_limits()
        .context("typed generation requires stream_limits")?;
    anyhow::ensure!(
        max_new <= limits.max_context,
        "--max-new exceeds admitted context"
    );
    let ids = token_file
        .map(|path| -> anyhow::Result<Vec<u32>> {
            let bytes = read_bounded(path, DEFAULT_MAX_JSON_BYTES)?;
            let ids: Vec<u32> = serde_json::from_slice(&bytes).context("typed prompt token IDs")?;
            anyhow::ensure!(
                ids.len()
                    .checked_add(max_new)
                    .is_some_and(|n| n <= limits.max_context),
                "prompt and --max-new exceed admitted context"
            );
            Ok(ids)
        })
        .transpose()?;
    let prompt = match (&ids, text) {
        (Some(ids), None) => Prompt::Ids(ids),
        (None, Some(text)) => Prompt::Text(text),
        _ => bail!("typed generation requires exactly one of --tokens or --prompt"),
    };
    let model_path = config.path().to_path_buf();
    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|b| b.name == backend_name)
        .map(|b| b.init)
        .with_context(|| format!("backend `{backend_name}` is not in the registry"))?;
    let backend =
        init().map_err(|e| anyhow::anyhow!("backend `{backend_name}` failed to init: {e}"))?;
    let mut recorder = tritium_evidence::EvidenceRecorder::new(
        "typed-generation-load",
        [
            tritium_evidence::TENSOR_UPLOADED_REGISTRATION,
            tritium_evidence::MODEL_LOADED_REGISTRATION,
        ],
    )?;
    let context = tritium_runtime::TensorUploadContext {
        recorder: &mut recorder,
        span: "load",
        parent: None,
        logical_time: 0,
    };
    let loaded = config
        .load(backend, context)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    anyhow::ensure!(
        loaded.runner.resource_admission().is_some(),
        "typed generation missing mandatory resource admission"
    );
    eprintln!(
        "typed candidate, not production-qualified: execution={:?}, resource_admission={:?}",
        loaded.runner.execution(),
        loaded.runner.resource_admission()
    );
    recorder.seal()?;
    decode_and_print(
        CliRunner::Bounded(Box::new(loaded.runner)),
        Some(Box::new(loaded.tokenizer)),
        &model_path,
        &prompt,
        max_new,
        eos,
    )
}

#[derive(Args, Debug)]
pub(crate) struct JoinedArgs {
    #[command(flatten)]
    stage: StageArgs,
    /// Complete canonical actual C conversion JSONL log; included in aggregate evidence bound.
    #[arg(long, requires = "original_captured_stage")]
    conversion_evidence: Option<PathBuf>,
    /// Explicit complete model level; no prefix/default inference.
    #[arg(long, requires = "original_captured_stage")]
    level: Option<String>,
    /// Positive backend tensor payload ceiling; not total process or KV/tokenizer RSS.
    #[arg(long, requires = "original_captured_stage")]
    max_payload_bytes: Option<u64>,
    /// Positive bound on each limits JSON file (default 16MiB).
    #[arg(long, requires = "original_captured_stage")]
    max_limits_bytes: Option<u64>,
    /// Require native tensor execution; default allows the shared explicit emulated tier.
    #[arg(long, requires = "original_captured_stage")]
    native_only: bool,
}
impl JoinedArgs {
    pub(crate) fn enabled(&self) -> bool {
        self.stage.enabled()
    }
}

pub(crate) fn run_joined(
    model_path: &Path,
    args: &JoinedArgs,
    prompt: &Prompt<'_>,
    max_new: usize,
    greedy: bool,
    eos: u32,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        greedy,
        "--greedy=false: sampling is not implemented in this subcommand"
    );
    let level = args
        .level
        .as_deref()
        .context("joined generation requires --level")?;
    anyhow::ensure!(!level.is_empty(), "--level must not be empty");
    let maximum = args
        .max_payload_bytes
        .context("joined generation requires --max-payload-bytes")?;
    anyhow::ensure!(maximum > 0, "--max-payload-bytes must be positive");
    // All cold-admission buffers leave scope before decoding. Backend payload
    // leases and the real tokenizer are owned by the returned runtime objects.
    let (runner, tokenizer) = {
        let mut inputs = args
            .stage
            .read(args.max_limits_bytes.unwrap_or(DEFAULT_MAX_JSON_BYTES))?;
        let fit = inputs.read_conversion_log(required(&args.conversion_evidence)?)?;
        let bytes = read_bounded(model_path, inputs.conversion_limits.max_package_bytes)?;
        let stage = inputs.open()?;
        let joined = tritium_quantize::engine::JoinedSalienceModel::open_original_captured(
            &bytes,
            &stage,
            &inputs.capture,
            &fit,
            inputs.conversion_limits,
        )?;
        let source = tritium_nn::ModelSource::JoinedSalience {
            model: &joined,
            level,
        };
        let budget = tritium_runtime::TensorPayloadBudget::new(maximum);
        let policy = tritium_schema::TensorUploadPolicy {
            native_only: args.native_only,
            max_payload_bytes: Some(maximum),
        };
        let mut recorder = tritium_evidence::EvidenceRecorder::new(
            "joined-generation-load",
            [
                tritium_evidence::TENSOR_UPLOADED_REGISTRATION,
                tritium_evidence::MODEL_LOADED_REGISTRATION,
            ],
        )?;
        let context = tritium_runtime::TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        };
        let backend = Box::new(tritium_cpu::CpuBackend::new());
        let result = match prompt {
            Prompt::Ids(_) => (
                CliRunner::from(ModelRunner::from_source_admitted(
                    source, backend, policy, &budget, context,
                )?),
                None,
            ),
            Prompt::Text(_) => {
                let loaded = tritium_nn::load(source, backend, policy, &budget, context)?;
                (
                    CliRunner::from(loaded.runner),
                    Some(Box::new(loaded.tokenizer) as Box<dyn Tokenizer>),
                )
            }
        };
        recorder.seal()?;
        result
    };
    decode_and_print(runner, tokenizer, model_path, prompt, max_new, eos)
}

/// Parse a JSON file of input token IDs.
///
/// The file must contain a JSON array of non-negative integers, e.g. `[1, 128000,
/// 9906]`. Each value must fit in a `u32` (token IDs always do).
///
/// # Errors
/// Returns an [`anyhow::Error`] if the file cannot be read, is not valid JSON, is
/// not an array of integers, or holds a value that does not fit in `u32`.
pub(crate) fn read_token_file(path: &Path) -> anyhow::Result<Vec<u32>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read token file `{}`", path.display()))?;
    let raw: Vec<i64> = serde_json::from_str(&text).with_context(|| {
        format!(
            "failed to parse `{}` as a JSON array of ints",
            path.display()
        )
    })?;
    raw.into_iter()
        .map(|v| {
            u32::try_from(v)
                .with_context(|| format!("token id {v} is out of range for u32 (0..=4294967295)"))
        })
        .collect()
}

/// Format generated token IDs as a report: a JSON array line plus one ID per line.
///
/// Pure so it can be unit-tested without running a model.
#[must_use]
pub(crate) fn render_output(tokens: &[u32]) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    // Machine-readable line first (a valid JSON array), then human-readable list.
    let json: Vec<String> = tokens.iter().map(u32::to_string).collect();
    let _ = writeln!(out, "[{}]", json.join(", "));
    for t in tokens {
        let _ = writeln!(out, "{t}");
    }
    out
}

/// Load `model_path`, greedily generate up to `max_new` tokens continuing
/// `tokens`, and print the resulting IDs.
///
/// The explicit `eos` token is passed to [`ModelRunner::generate`]; generation
/// also stops at `max_new`. The legacy CLI supplies its historical EOS default
/// when omitted. `greedy = false` rejects because sampling is unsupported.
///
/// # Errors
/// Returns an [`anyhow::Error`] if the model file cannot be read or parsed, no
/// backend is available, the model is missing weights, or generation fails. The
/// caller maps this to a non-zero exit; nothing here panics on bad input.
pub(crate) fn run(
    model_path: &Path,
    prompt: &Prompt<'_>,
    max_new: usize,
    greedy: bool,
    eos: u32,
) -> anyhow::Result<()> {
    if !greedy {
        // A knob that silently does nothing is worse than no knob: refuse.
        anyhow::bail!(
            "--greedy=false: sampling is not implemented in this subcommand — use \
             tritium-serve's OpenAI API (temperature/top-k) for sampled decoding"
        );
    }

    let (runner, tokenizer) = load(model_path)?;
    decode_and_print(runner, tokenizer, model_path, prompt, max_new, eos)
}

fn decode_and_print(
    runner: impl Into<CliRunner>,
    tokenizer: Option<Box<dyn Tokenizer>>,
    model_path: &Path,
    prompt: &Prompt<'_>,
    max_new: usize,
    eos: u32,
) -> anyhow::Result<()> {
    let mut runner = runner.into();
    let tokens = match prompt {
        Prompt::Ids(ids) => (*ids).to_vec(),
        Prompt::Text(text) => {
            let Some(tokenizer) = tokenizer.as_ref() else {
                bail!(
                    "--prompt needs a tokenizer, and `{}` does not carry one. A `tritium convert` \
                     directory has tokenizer.json (copied from the source model); a GGUF must embed \
                     its BPE. Use --tokens <ids.json> instead, or convert from a source snapshot \
                     that includes tokenizer.json.",
                    model_path.display()
                );
            };
            tokenizer
                .encode(text)
                .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?
        }
    };
    if tokens.is_empty() {
        bail!("the prompt is empty — nothing to condition generation on");
    }

    let generated = runner
        .generate(&tokens, max_new, eos)
        .context("generation failed")?;

    // Text out only when text went in: `--tokens` promises exact ids in, exact ids out, and a
    // decoded string there would be a different contract than the reproducible path advertises.
    if let (Prompt::Text(_), Some(tokenizer)) = (prompt, tokenizer.as_ref())
        && let Ok(text) = tokenizer.decode(&generated)
    {
        println!("{text}");
    }
    print!("{}", render_output(&generated));
    Ok(())
}

/// What to condition generation on.
pub(crate) enum Prompt<'a> {
    /// Exact token ids from `--tokens`. No tokenizer is consulted, so this path is reproducible
    /// across tokenizer versions.
    Ids(&'a [u32]),
    /// Raw text from `--prompt`, tokenized with the model's own tokenizer.
    Text(&'a str),
}

/// Load a model, plus its tokenizer when it carries one.
///
/// Dispatches on **content**: a directory holding `model.tslb` is a `tritium convert` output and
/// loads through [`ModelRunner::from_salt`]; anything else is read as a GGUF file. A path that is
/// neither fails as "not a model" rather than being silently misread as the other kind.
fn load(model_path: &Path) -> anyhow::Result<(ModelRunner, Option<Box<dyn Tokenizer>>)> {
    if model_path.is_dir() {
        let bundle = model_path.join("model.tslb");
        if !bundle.exists() {
            bail!(
                "`{}` is a directory but has no model.tslb, so it is not a `tritium convert` \
                 output. Pass a converted directory or a .gguf file.",
                model_path.display()
            );
        }
        let runner = ModelRunner::from_salt(
            model_path,
            &bundle,
            Box::new(tritium_cpu::CpuBackend::new()),
        )
        .with_context(|| format!("failed to load converted model `{}`", model_path.display()))?;

        // `convert` copies these; a source snapshot without them still converts fine, so their
        // absence is a missing capability rather than a broken model.
        let tokenizer_json = model_path.join("tokenizer.json");
        let tokenizer = if tokenizer_json.exists() {
            HfJsonTokenizer::from_files(&tokenizer_json, &model_path.join("tokenizer_config.json"))
                .ok()
                .map(|t| Box::new(t) as Box<dyn Tokenizer>)
        } else {
            None
        };
        return Ok((runner, tokenizer));
    }

    let bytes = std::fs::read(model_path)
        .with_context(|| format!("failed to read model `{}`", model_path.display()))?;
    let runner = ModelRunner::load_cpu(&bytes)
        .with_context(|| format!("failed to load model `{}`", model_path.display()))?;
    let tokenizer = tritium_format::read_gguf(&bytes)
        .ok()
        .and_then(|f| GgufBpeTokenizer::from_gguf(&f).ok())
        .map(|t| Box::new(t) as Box<dyn Tokenizer>);
    Ok((runner, tokenizer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runner_bridge_does_not_embed_backend_resident_state() {
        // Neither graph owner should be copied into every CLI bridge value.
        assert!(std::mem::size_of::<CliRunner>() <= 2 * std::mem::size_of::<usize>());
    }

    #[test]
    fn render_output_is_json_then_lines() {
        let report = render_output(&[1, 2, 3]);
        assert!(report.starts_with("[1, 2, 3]\n"), "{report}");
        assert!(report.contains("\n1\n2\n3\n"), "{report}");
    }

    #[test]
    fn render_output_empty_is_empty_array() {
        let report = render_output(&[]);
        assert_eq!(report, "[]\n");
    }

    #[test]
    fn read_token_file_parses_array() {
        let mut tmp = std::env::temp_dir();
        tmp.push(format!("tritium-cli-tokens-{}.json", std::process::id()));
        std::fs::write(&tmp, b"[1, 128000, 9906]").expect("write temp");
        let toks = read_token_file(&tmp).expect("parse tokens");
        let _ = std::fs::remove_file(&tmp);
        assert_eq!(toks, vec![1, 128_000, 9906]);
    }

    #[test]
    fn read_token_file_rejects_negative() {
        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "tritium-cli-tokens-neg-{}.json",
            std::process::id()
        ));
        std::fs::write(&tmp, b"[1, -5]").expect("write temp");
        let err = read_token_file(&tmp).expect_err("negative must error");
        let _ = std::fs::remove_file(&tmp);
        assert!(format!("{err:#}").contains("out of range"), "{err:#}");
    }

    #[test]
    fn read_token_file_rejects_non_json() {
        let mut tmp = std::env::temp_dir();
        tmp.push(format!(
            "tritium-cli-tokens-bad-{}.json",
            std::process::id()
        ));
        std::fs::write(&tmp, b"not json").expect("write temp");
        let err = read_token_file(&tmp).expect_err("bad json must error");
        let _ = std::fs::remove_file(&tmp);
        assert!(format!("{err:#}").contains("failed to parse"), "{err:#}");
    }

    #[test]
    fn run_on_missing_model_errors_cleanly() {
        let err = run(
            Path::new("/nonexistent/model.gguf"),
            &Prompt::Ids(&[1, 2, 3]),
            4,
            true,
            128_001,
        )
        .expect_err("missing model must error");
        assert!(
            format!("{err:#}").contains("failed to read model"),
            "{err:#}"
        );
    }

    /// A path that exists but is a directory without `model.tslb` must be rejected as "not a
    /// converted model", not fall through to `std::fs::read` and surface a bare EISDIR.
    #[test]
    fn run_on_a_directory_without_a_bundle_names_the_missing_file() {
        let dir = std::env::temp_dir().join(format!("tritium-gen-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let err = run(&dir, &Prompt::Ids(&[1, 2, 3]), 4, true, 128_001)
            .expect_err("a directory with no bundle must error");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(format!("{err:#}").contains("model.tslb"), "{err:#}");
    }
}
