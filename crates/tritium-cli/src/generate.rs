//! `tritium generate`: load a GGUF model and greedily decode tokens.
//!
//! The subcommand is deterministic and offline-reproducible: input token IDs come
//! from a JSON file (`--tokens`), generation is greedy, and the output is the list
//! of newly generated token IDs printed one per line (plus a JSON array for easy
//! machine consumption). The model is loaded through
//! [`tritium_nn::ModelRunner`], which selects a backend via the runtime registry.
//!
//! Like the rest of the CLI, every failure flows through [`anyhow::Result`]: a
//! missing model, an unreadable token file, malformed JSON, or a model that is
//! missing weights all yield a clean message and a non-zero exit — never a panic.

use std::path::Path;

use anyhow::Context as _;
use tritium_nn::ModelRunner;

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
    tokens: &[u32],
    max_new: usize,
    greedy: bool,
    eos: u32,
) -> anyhow::Result<()> {
    if !greedy {
        eprintln!(
            "note: only greedy decoding is implemented in v0.20; \
             `--greedy=false` still decodes greedily"
        );
    }

    let bytes = std::fs::read(model_path)
        .with_context(|| format!("failed to read model `{}`", model_path.display()))?;
    let mut runner = ModelRunner::load_cpu(&bytes)
        .with_context(|| format!("failed to load model `{}`", model_path.display()))?;

    let generated = runner
        .generate(tokens, max_new, eos)
        .context("generation failed")?;

    print!("{}", render_output(&generated));
    Ok(())
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
            &[1, 2, 3],
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
}
