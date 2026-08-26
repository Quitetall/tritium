//! **Can a converted model actually be run from the CLI?** (ADR 0038 WS-4)
//!
//! `tritium convert` writes a ternary model directory. Until WS-4 nothing in the CLI consumed one:
//! `generate` was GGUF-only and `report` takes fp masters, so the adoption spine — acquire, convert,
//! run — stopped one step short of running anything. The artifact was library-loadable and
//! CLI-unrunnable, which is a converter that produces nothing a user can use.
//!
//! The cheap half of this file needs no model at all: the dispatch is on *content*, and its refusal
//! messages are the part a user hits first when they point `--model` somewhere wrong.
//!
//! The end-to-end half is `#[ignore]`d and needs a real fp master:
//!
//! ```text
//! TRITIUM_MODEL_DIR=$HOME/.cache/tritium-models/smollm2-135m \
//!   cargo test -p tritium-cli --release --test generate_converted_dir -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

fn tritium_bin() -> PathBuf {
    let mut p = std::env::current_exe().expect("test exe");
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.join("tritium")
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(tritium_bin())
        .args(args)
        .output()
        .expect("run tritium")
}

/// A directory that is not a converted model must say so. Before the content dispatch existed this
/// path went to `std::fs::read`, which fails with a bare EISDIR — true but useless.
#[test]
fn a_directory_without_a_bundle_is_refused_by_name() {
    let dir = std::env::temp_dir().join(format!("tritium-gen-empty-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("mkdir");
    let out = run(&[
        "generate",
        "--model",
        dir.to_str().unwrap(),
        "--prompt",
        "hello",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "should have failed: {stderr}");
    assert!(
        stderr.contains("model.tslb"),
        "the error should name what is missing, got: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `--tokens` and `--prompt` are different contracts — exact ids in/out versus tokenizer in the
/// loop — so taking both would leave the reproducibility guarantee ambiguous.
#[test]
fn tokens_and_prompt_are_mutually_exclusive() {
    let out = run(&[
        "generate",
        "--model",
        "/nonexistent.gguf",
        "--tokens",
        "/nonexistent.json",
        "--prompt",
        "hello",
    ]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(
        stderr.contains("cannot be used with") || stderr.contains("conflict"),
        "clap should reject both at once, got: {stderr}"
    );
}

/// Neither one is also an error: there is nothing to condition generation on.
#[test]
fn one_of_tokens_or_prompt_is_required() {
    let out = run(&["generate", "--model", "/nonexistent.gguf"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success());
    assert!(
        stderr.contains("required") || stderr.contains("--prompt"),
        "clap should require one of them, got: {stderr}"
    );
}

fn convert(model: &Path, out: &Path) {
    let _ = std::fs::remove_dir_all(out);
    let status = Command::new(tritium_bin())
        .args([
            "convert",
            "--model",
            model.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--planes",
            "4",
            "--group",
            "256",
            "--fold-alpha",
            "0",
        ])
        .status()
        .expect("run tritium convert");
    assert!(status.success(), "convert failed");
}

/// The whole spine in one test: convert a real model, then generate from the directory with a text
/// prompt. This is the claim WS-4 exists to make true.
#[test]
#[ignore = "needs a real fp master; converts a whole model"]
fn convert_then_generate_from_the_directory() {
    let model = PathBuf::from(
        std::env::var("TRITIUM_MODEL_DIR").expect("set TRITIUM_MODEL_DIR to an fp model directory"),
    );
    let out = std::env::temp_dir().join(format!("tritium-gen-e2e-{}", std::process::id()));
    convert(&model, &out);

    let result = run(&[
        "generate",
        "--model",
        out.to_str().unwrap(),
        "--prompt",
        "The capital of France is",
        "--max-new",
        "8",
    ]);
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    println!("--- stdout ---\n{stdout}");
    assert!(result.status.success(), "generate failed: {stderr}");

    // `render_output` always emits a JSON array line; ids prove the model ran, and the decoded text
    // line above it proves the tokenizer round-tripped.
    let ids_line = stdout
        .lines()
        .find(|l| l.starts_with('[') && l.ends_with(']'))
        .unwrap_or_else(|| panic!("no JSON id array in output:\n{stdout}"));
    let ids: Vec<u32> = serde_json::from_str(ids_line).expect("parse id array");
    assert!(!ids.is_empty(), "generated nothing");
    assert!(ids.len() <= 8, "respected --max-new, got {}", ids.len());

    // Ids must be inside the model's vocabulary. A tokenizer/model mismatch would otherwise show up
    // only as garbled text, which is easy to squint past.
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("config.json")).expect("read config"))
            .expect("parse config");
    let vocab = config["vocab_size"].as_u64().expect("vocab_size") as u32;
    for id in &ids {
        assert!(*id < vocab, "token id {id} outside vocab {vocab}");
    }

    let _ = std::fs::remove_dir_all(&out);
}
