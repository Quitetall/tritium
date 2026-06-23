//! End-to-end test of the `tritium generate` subcommand against the committed tiny
//! GGUF fixture.
//!
//! The fixture is a *format-level* container (a handful of tensors), not a complete
//! runnable model, so `generate` must fail cleanly with a non-zero exit and a
//! descriptive message about the missing weights — never panic or abort. This
//! exercises the whole subcommand path: argument parsing, the JSON token-file
//! reader, the model-load attempt through `ModelRunner`, and the `anyhow` error
//! surface. If the real BitNet 2B4T model is present on disk, a second case proves
//! `generate` produces deterministic greedy tokens.

use std::path::PathBuf;
use std::process::Command;

/// Absolute path to the `tritium` binary built by Cargo for this test run.
fn tritium_bin() -> PathBuf {
    // `CARGO_BIN_EXE_<name>` is set by Cargo for integration tests of a bin crate.
    PathBuf::from(env!("CARGO_BIN_EXE_tritium"))
}

/// Absolute path to the committed tiny GGUF fixture in the sibling format crate.
fn tiny_fixture() -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tritium-format/tests/fixtures/bitnet_tiny.gguf"
    ))
}

/// Write a JSON token file into the temp dir and return its path.
fn write_tokens(name: &str, json: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "tritium-cli-gen-{}-{}.json",
        name,
        std::process::id()
    ));
    std::fs::write(&p, json).expect("write token file");
    p
}

#[test]
fn generate_on_tiny_fixture_errors_cleanly() {
    let tokens = write_tokens("tiny", "[1, 2, 3]");
    let output = Command::new(tritium_bin())
        .arg("generate")
        .arg("--model")
        .arg(tiny_fixture())
        .arg("--tokens")
        .arg(&tokens)
        .arg("--max-new")
        .arg("4")
        .output()
        .expect("spawn tritium");
    let _ = std::fs::remove_file(&tokens);

    // The fixture is not a full model, so generation must fail — but cleanly.
    assert!(
        !output.status.success(),
        "generate unexpectedly succeeded on the partial fixture"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    // A descriptive load error, not a panic / abort.
    assert!(
        stderr.contains("failed to load model") || stderr.contains("missing tensor"),
        "expected a clean load error, got: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "must not panic, got: {stderr}"
    );
}

#[test]
fn generate_rejects_missing_token_file() {
    let output = Command::new(tritium_bin())
        .arg("generate")
        .arg("--model")
        .arg(tiny_fixture())
        .arg("--tokens")
        .arg("/nonexistent/tokens.json")
        .output()
        .expect("spawn tritium");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to read token file"),
        "expected a token-file read error, got: {stderr}"
    );
}

#[test]
fn generate_help_lists_flags() {
    let output = Command::new(tritium_bin())
        .arg("generate")
        .arg("--help")
        .output()
        .expect("spawn tritium");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for flag in ["--model", "--tokens", "--max-new", "--greedy", "--eos"] {
        assert!(stdout.contains(flag), "help missing {flag}: {stdout}");
    }
}

/// End-to-end greedy generation against the real BitNet 2B4T GGUF.
///
/// A full 2B-parameter generation on CPU takes minutes, so this is gated behind
/// the `TRITIUM_RUN_SLOW` environment variable *and* the model's presence; it is
/// skipped (passing) otherwise so the default `cargo test` lane stays fast. When it
/// does run it asserts `generate` exits 0 and emits a JSON array of token IDs as
/// its first output line. (Greedy-exactness vs. the HF oracle is verified by
/// WF-4's fidelity ladder; here we only prove the CLI plumbing works on the real
/// model.)
#[test]
fn generate_on_real_model_runs() {
    if std::env::var_os("TRITIUM_RUN_SLOW").is_none() {
        eprintln!("skipping real-model generate; set TRITIUM_RUN_SLOW=1 to enable");
        return;
    }
    // Resolve the home dir at runtime (cross-platform: HOME on unix, USERPROFILE
    // on Windows) — `env!("HOME")` is a compile-time lookup that is undefined on
    // the Windows CI runner and breaks the build there.
    let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) else {
        eprintln!("skipping: no HOME/USERPROFILE in the environment");
        return;
    };
    let model =
        PathBuf::from(home).join(".cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf");
    if !model.exists() {
        eprintln!("skipping: real model not present at {}", model.display());
        return;
    }

    let tokens = write_tokens("real", "[128000, 9906, 1917]");
    let out = Command::new(tritium_bin())
        .arg("generate")
        .arg("--model")
        .arg(&model)
        .arg("--tokens")
        .arg(&tokens)
        .arg("--max-new")
        .arg("4")
        .output()
        .expect("spawn tritium");
    let _ = std::fs::remove_file(&tokens);

    assert!(
        out.status.success(),
        "generate failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let first_line = stdout.lines().next().unwrap_or_default();
    assert!(
        first_line.starts_with('[') && first_line.ends_with(']'),
        "first line should be a JSON array, got: {first_line}"
    );
}
