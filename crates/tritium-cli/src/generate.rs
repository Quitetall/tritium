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

use std::path::Path;

use anyhow::{Context as _, bail};
use tritium_nn::{GgufBpeTokenizer, HfJsonTokenizer, ModelRunner, Tokenizer};

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
/// The `eos` token (defaults to the BitNet/LLaMA-3 end-of-text id when `None`) is
/// passed to [`ModelRunner::generate`]; generation also stops at `max_new`. When
/// `greedy` is `false` the function still decodes greedily for v0.20 (sampling is a
/// later wave) but records the intent in a note on stderr so the flag is honest.
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

    let (mut runner, tokenizer) = load(model_path)?;

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
