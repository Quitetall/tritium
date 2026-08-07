#![cfg(feature = "cuda")]

use std::{path::PathBuf, process::Command};

fn tritium_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tritium"))
}

#[test]
fn hestia_gate_c_command_is_discoverable() {
    let output = Command::new(tritium_bin())
        .args(["salt", "seal-hestia-gate-c", "--help"])
        .output()
        .expect("spawn Tritium CLI");

    assert!(
        output.status.success(),
        "help failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--release"));
    assert!(stdout.contains("--source-revision"));
    assert!(stdout.contains("--output"));
    assert!(stdout.contains("--cuda-device"));
}

#[test]
fn hestia_gate_c_rejects_invalid_revision_before_creating_output() {
    let output_path = std::env::temp_dir().join(format!(
        "tritium-hestia-invalid-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let output = Command::new(tritium_bin())
        .args([
            "salt",
            "seal-hestia-gate-c",
            "--release",
            "1.1.0-rc.0",
            "--source-revision",
            "not-a-revision",
            "--output",
        ])
        .arg(&output_path)
        .args(["--cuda-device", "0"])
        .output()
        .expect("spawn Tritium CLI");

    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("source revision must be 40 lowercase hexadecimal characters")
    );
    assert!(!output_path.exists());
}
