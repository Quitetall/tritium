//! CLI contract tests for model-source selection.
#![cfg(feature = "serve")]

use std::process::Command;

fn server_bin() -> &'static str {
    env!("CARGO_BIN_EXE_tritium-serve")
}

#[test]
fn help_documents_converted_source() {
    let output = Command::new(server_bin())
        .arg("--help")
        .output()
        .expect("run tritium-serve --help");
    assert!(output.status.success());
    let help = String::from_utf8_lossy(&output.stdout);
    assert!(help.contains("--converted <dir>"), "{help}");
    assert!(help.contains("TRITIUM_CONVERTED"), "{help}");
}

#[test]
fn model_sources_are_mutually_exclusive() {
    let output = Command::new(server_bin())
        .args(["--model", "model.gguf", "--converted", "converted"])
        .output()
        .expect("run conflicting model-source options");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("provide exactly one of"),
        "unexpected stderr: {stderr}"
    );
}
