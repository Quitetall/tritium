//! CLI contract tests for model-source selection.
#![cfg(feature = "serve")]

use std::process::Command;
#[path = "portable_fixture/mod.rs"]
mod fixture;

struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new(label: &str) -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "serve-typed-{label}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn clean_command() -> Command {
    let mut command = Command::new(server_bin());
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("TRITIUM_") {
            command.env_remove(name);
        }
    }
    command
}
fn typed_config(path: &std::path::Path) -> serde_json::Value {
    serde_json::json!({"path":path,"level":"test","max_source_bytes":1_000_000,
        "max_manifest_bytes":128_000,"max_directory_bytes":128_000,"max_config_bytes":16_000,
        "max_tensor_bytes":16_000,"max_decoded_tensor_bytes":16_000,"max_payload_bytes":1_000_000,"native_only":false})
}

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

#[test]
fn typed_source_rejects_bad_launch_and_limits_before_backend_init() {
    let root = Scratch::new("reject");
    let load_path = root.0.join("load.json");
    for case in 0..11 {
        let mut config = typed_config(&root.0.join("missing.trit"));
        match case {
            0 => config["max_payload_bytes"] = serde_json::json!(0),
            1 => {
                config.as_object_mut().unwrap().remove("level");
            }
            2 => config["unknown"] = serde_json::json!(true),
            3 => config["level"] = serde_json::json!(""),
            4 => config["path"] = serde_json::json!("relative.trit"),
            _ => {}
        }
        std::fs::write(&load_path, config.to_string()).unwrap();
        if case == 5 {
            std::fs::write(&load_path, vec![b' '; 16 * 1024 + 1]).unwrap();
        }
        let mut command = clean_command();
        command
            .arg("--typed")
            .arg(&load_path)
            .args(["--backend", "not-linked"]);
        match case {
            6 => {
                command.args(["--host", "0.0.0.0"]);
            }
            7 => {
                command.arg("--raw-tokens");
            }
            8 => {
                command.args(["--batch-slots", "2"]);
            }
            9 => {
                command.args(["--spec", "lookup"]);
            }
            10 => {
                command.args(["--model", "legacy.gguf"]);
            }
            _ => {}
        }
        let output = command.output().unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            !error.contains("is not in the registry"),
            "case {case}: {error}"
        );
        assert!(!error.contains("loading legacy"), "case {case}: {error}");
    }
    let output = clean_command().arg("--help").output().unwrap();
    assert!(String::from_utf8_lossy(&output.stdout).contains("TRITIUM_TYPED"));
}

struct Server(std::process::Child);
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn http(port: u16, method: &str, path: &str, body: &str) -> std::io::Result<String> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        std::time::Duration::from_millis(200),
    )?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(3)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(3)))?;
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}
#[test]
fn actual_typed_binary_loads_real_graph_and_serves_chat_on_loopback() {
    for hybrid in [false, true] {
        for additive in [false, true] {
            let root = Scratch::new("binary");
            let model_path = root.0.join("model.trit");
            let load_path = root.0.join("load.json");
            std::fs::write(&model_path, fixture::package(hybrid, additive)).unwrap();
            std::fs::write(&load_path, typed_config(&model_path).to_string()).unwrap();
            let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let port = listener.local_addr().unwrap().port();
            drop(listener);
            let mut server = Server(
                clean_command()
                    .arg("--typed")
                    .arg(&load_path)
                    .args([
                        "--backend",
                        "cpu",
                        "--port",
                        &port.to_string(),
                        "--rate-limit-rpm",
                        "0",
                    ])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .unwrap(),
            );
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let ready = loop {
                assert!(
                    server.0.try_wait().unwrap().is_none(),
                    "typed server exited before ready"
                );
                if let Ok(response) = http(port, "GET", "/readyz", "") {
                    break response;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "typed server did not start within 5s"
                );
                std::thread::sleep(std::time::Duration::from_millis(20));
            };
            assert!(ready.starts_with("HTTP/1.1 200"), "{ready}");
            assert!(ready.contains("\"production_artifact\":false"));
            assert!(ready.contains("\"startup_receipt\":null"));
            let body=serde_json::json!({"model":"tritium","messages":[{"role":"user","content":"hello partner"}],
            "max_tokens":3,"temperature":0,"stream":false}).to_string();
            let response = http(port, "POST", "/v1/chat/completions", &body).unwrap();
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
            assert!(response.contains("\"choices\""));
            assert!(response.contains("\"finish_reason\""));
            eprintln!(
                "actual typed binary hybrid={hybrid} additive={additive}: loopback chat returned HTTP 200; unqualified"
            );
            drop(server);
        }
    }
}
