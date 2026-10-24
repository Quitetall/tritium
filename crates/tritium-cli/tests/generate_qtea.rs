//! Explicit-source command refusals; real native-fit protocol replay is separate.
use std::process::Command;
fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tritium"));
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("TRITIUM_") {
            command.env_remove(name);
        }
    }
    command.args(["--evidence", "off", "generate"]);
    command
}
#[test]
fn qtea_requires_explicit_eos_and_excludes_other_sources() {
    for args in [
        vec!["--qtea", "/absent.json", "--prompt", "Testing"],
        vec![
            "--qtea",
            "/absent.json",
            "--typed",
            "/other.json",
            "--prompt",
            "Testing",
            "--eos",
            "3",
        ],
        vec![
            "--qtea",
            "/absent.json",
            "--model",
            "/other.trit",
            "--prompt",
            "Testing",
            "--eos",
            "3",
        ],
    ] {
        let output = command().args(args).output().unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("error:"));
    }
}
#[test]
fn qtea_unsupported_build_or_backend_refuses_before_source_read() {
    let output = command()
        .args([
            "--qtea",
            "/absent.json",
            "--prompt",
            "Testing",
            "--eos",
            "3",
            "--backend",
            "cuda",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains(if cfg!(feature = "host-qtea") {
            "--qtea requires --backend cpu"
        } else {
            "--qtea requires build feature host-qtea"
        }),
        "{error}"
    );
    assert!(!error.contains("No such file"), "{error}");
}
