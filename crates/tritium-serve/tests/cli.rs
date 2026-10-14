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
    for case in 0..19 {
        let mut config = typed_config(&root.0.join("missing.trit"));
        match case {
            0 => config["max_payload_bytes"] = serde_json::json!(0),
            1 => {
                config.as_object_mut().unwrap().remove("level");
            }
            2 => config["unknown"] = serde_json::json!(true),
            3 => config["level"] = serde_json::json!(""),
            4 => config["path"] = serde_json::json!("relative.trit"),
            11 => {
                config["execution"] = serde_json::json!({"kind":"resident_device_f32","max_device_payload_bytes":1_000_000})
            }
            12 => {
                config["execution"] =
                    serde_json::json!({"kind":"resident_device_f32","max_device_payload_bytes":0});
                config["stream_limits"] = stream_config();
            }
            13 => {
                config["stream_limits"] = stream_config();
                config["stream_limits"]["max_context"] = 0.into();
            }
            14 => {
                config["stream_limits"] = stream_config();
                config["stream_limits"]["max_segment_tokens"] = 33.into();
            }
            15 => config["stream_limits"] = serde_json::json!({"max_context":32}),
            16 => config["execution"] = serde_json::json!({"kind":"unknown"}),
            17 => {
                config["execution"] =
                    serde_json::json!({"kind":"host","max_device_payload_bytes":1})
            }
            18 => config["execution"] = serde_json::Value::Null,
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
fn stream_config() -> serde_json::Value {
    serde_json::json!({"max_context":32,"max_segment_tokens":2,"max_host_cache_payload_bytes":1_000_000})
}
fn streamed_resources() -> serde_json::Value {
    serde_json::json!({
        "max_source_workspace_bytes":65_536,
        "max_host_workspace_bytes":1_000_000,
        "max_device_workspace_bytes":0,
        "max_retained_host_bytes":1_000_000
    })
}

#[test]
fn streamed_resources_reject_bad_contract_before_backend_initialization() {
    let root = Scratch::new("streamed-contract");
    let path = root.0.join("load.json");
    for case in 0..10 {
        let mut config = typed_config(&root.0.join("missing.trit"));
        config["stream_limits"] = stream_config();
        config["streamed_resources"] = streamed_resources();
        match case {
            0 => {
                config.as_object_mut().unwrap().remove("stream_limits");
            }
            1 => {
                config["execution"] = serde_json::json!({"kind":"resident_device_f32","max_device_payload_bytes":1_000_000})
            }
            2 => config["streamed_resources"]["max_source_workspace_bytes"] = 0.into(),
            3 => config["streamed_resources"]["max_host_workspace_bytes"] = 0.into(),
            4 => config["streamed_resources"]["max_retained_host_bytes"] = 0.into(),
            5 => {
                config["streamed_resources"]
                    .as_object_mut()
                    .unwrap()
                    .remove("max_device_workspace_bytes");
            }
            6 => config["streamed_resources"]["unknown"] = true.into(),
            7 => config["streamed_resources"]["max_device_workspace_bytes"] = (-1).into(),
            8 => config["streamed_resources"]["max_host_workspace_bytes"] = "unlimited".into(),
            9 => config["streamed_resources"] = true.into(),
            _ => unreachable!(),
        }
        std::fs::write(&path, config.to_string()).unwrap();
        let output = clean_command()
            .arg("--typed")
            .arg(&path)
            .args(["--backend", "not-linked"])
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "case {case}");
        assert!(
            !error.contains("is not in the registry"),
            "case {case}: {error}"
        );
        assert!(!error.contains("No such file"), "case {case}: {error}");
    }
}

#[test]
fn actual_streamed_binary_refuses_each_covered_budget_without_starting_listener() {
    let root = Scratch::new("streamed-budget");
    let model = root.0.join("model.trit");
    let path = root.0.join("load.json");
    std::fs::write(&model, fixture::package(true, true)).unwrap();
    for field in [
        "max_payload_bytes",
        "max_source_workspace_bytes",
        "max_host_workspace_bytes",
        "max_retained_host_bytes",
    ] {
        let mut config = typed_config(&model);
        config["stream_limits"] = stream_config();
        config["streamed_resources"] = streamed_resources();
        if field == "max_payload_bytes" {
            config[field] = 1.into();
        } else {
            config["streamed_resources"][field] = 1.into();
        }
        std::fs::write(&path, config.to_string()).unwrap();
        let output = clean_command()
            .arg("--typed")
            .arg(&path)
            .args(["--backend", "cpu"])
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{field}");
        assert!(
            error.contains("model resource admission"),
            "{field}: {error}"
        );
        assert!(
            !error.contains("typed runtime diagnostics"),
            "{field}: {error}"
        );
        assert!(!error.contains("listening"), "{field}: {error}");
    }
}
#[test]
fn unsupported_resident_selection_rejects_before_model_access_without_host_fallback() {
    let root = Scratch::new("resident-unsupported");
    let path = root.0.join("load.json");
    let mut config = typed_config(&root.0.join("missing.trit"));
    config["stream_limits"] = stream_config();
    config["execution"] =
        serde_json::json!({"kind":"resident_device_f32","max_device_payload_bytes":1_000_000});
    std::fs::write(&path, config.to_string()).unwrap();
    let output = clean_command()
        .arg("--typed")
        .arg(&path)
        .args(["--backend", "cpu"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(
        error.contains("resident execution selection on cpu"),
        "{error}"
    );
    assert!(!error.contains("No such file"), "{error}");
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
    check_actual_typed_binary(false, false, false, false);
}
#[test]
fn actual_bounded_typed_binary_prefills_multiple_segments_and_reports_admission() {
    check_actual_typed_binary(false, true, false, false);
}
#[test]
fn actual_encoded_streamed_typed_binary_serves_chat_and_reports_logical_resources() {
    check_actual_typed_binary(false, true, true, false);
}
#[cfg(feature = "cuda")]
#[test]
fn physical_encoded_streamed_typed_binary_serves_cuda_chat_without_resident_fallback() {
    tritium_cuda::CudaBackend::new(0).expect("physical encoded serving test requires CUDA");
    check_actual_typed_binary(false, true, true, true);
}
#[cfg(feature = "cuda")]
#[test]
fn actual_resident_typed_binary_serves_gpu_chat_and_sse_with_effective_diagnostics() {
    match tritium_cuda::CudaBackend::new(0) {
        Ok(_) => check_actual_typed_binary(true, true, false, true),
        Err(e) if std::env::var("TRITIUM_REQUIRE_CUDA").as_deref() == Ok("1") => {
            panic!("required CUDA: {e}")
        }
        Err(e) => eprintln!("SKIP resident typed binary: {e}"),
    }
}
fn check_actual_typed_binary(resident: bool, bounded: bool, streamed: bool, cuda: bool) {
    for hybrid in [false, true] {
        for additive in [false, true] {
            let root = Scratch::new("binary");
            let model_path = root.0.join("model.trit");
            let load_path = root.0.join("load.json");
            std::fs::write(&model_path, fixture::package(hybrid, additive)).unwrap();
            let mut config = typed_config(&model_path);
            if bounded {
                config["stream_limits"] = stream_config();
                config["stream_limits"]["max_context"] = 16.into();
            }
            if resident {
                config["execution"] = serde_json::json!({"kind":"resident_device_f32","max_device_payload_bytes":1_000_000});
            }
            if streamed {
                config["streamed_resources"] = streamed_resources();
                if cuda {
                    config["streamed_resources"]["max_device_workspace_bytes"] = 4.into();
                }
            }
            std::fs::write(&load_path, config.to_string()).unwrap();
            let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let port = listener.local_addr().unwrap().port();
            drop(listener);
            let mut server = Server(
                clean_command()
                    .arg("--typed")
                    .arg(&load_path)
                    .args([
                        "--backend",
                        if cuda { "cuda" } else { "cpu" },
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
            let ready_json: serde_json::Value =
                serde_json::from_str(ready.split_once("\r\n\r\n").unwrap().1).unwrap();
            let info = &ready_json["runtime_info"];
            assert_eq!(
                info["execution"],
                if resident {
                    "resident_device_f32"
                } else {
                    "host"
                }
            );
            assert_eq!(info["max_context"], if bounded { 16 } else { 32 });
            assert_eq!(info["max_segment_tokens"], if bounded { 2 } else { 32 });
            assert_eq!(info["resident_payload_bytes"].as_u64().is_some(), resident);
            let admission = &info["resource_admission"];
            if streamed {
                assert!(admission["payload_bytes"].as_u64().unwrap() > 0);
                assert_eq!(admission["source_workspace_bytes"], 65_536);
                assert!(admission["host_workspace_bytes"].as_u64().unwrap() > 0);
                assert_eq!(
                    admission["device_workspace_bytes"],
                    if cuda && additive { 4 } else { 0 }
                );
                assert!(admission["retained_host_bytes"].as_u64().unwrap() > 0);
                assert_eq!(admission.as_object().unwrap().len(), 5);
            } else {
                assert!(
                    admission.is_null(),
                    "compatibility must not invent admission: {info}"
                );
            }
            let body=serde_json::json!({"model":"tritium","messages":[{"role":"user","content":"hello partner"}],
            "max_tokens":3,"temperature":0,"stream":false}).to_string();
            let response = http(port, "POST", "/v1/chat/completions", &body).unwrap();
            assert!(response.starts_with("HTTP/1.1 200"), "{response}");
            assert!(response.contains("\"choices\""));
            assert!(response.contains("\"finish_reason\""));
            let streaming: serde_json::Value = serde_json::from_str(&body).unwrap();
            let mut streaming = streaming;
            streaming["stream"] = true.into();
            let sse = http(port, "POST", "/v1/chat/completions", &streaming.to_string()).unwrap();
            assert!(sse.starts_with("HTTP/1.1 200"), "{sse}");
            assert!(sse.contains("data: [DONE]"), "{sse}");
            assert!(
                sse.contains("\"choices\"") && !sse.contains("\"error\""),
                "{sse}"
            );
            if bounded {
                let too_long = serde_json::json!({"model":"tritium","messages":[{"role":"user","content":(["hello";20].join(" "))}],"max_tokens":1,"temperature":0,"stream":false});
                let rejected =
                    http(port, "POST", "/v1/chat/completions", &too_long.to_string()).unwrap();
                assert!(
                    rejected.starts_with("HTTP/1.1 400")
                        && rejected.contains("admitted model context 16"),
                    "{rejected}"
                );
                let retry = http(port, "POST", "/v1/chat/completions", &body).unwrap();
                assert!(
                    retry.starts_with("HTTP/1.1 200") && retry.contains("\"choices\""),
                    "{retry}"
                );
                assert!(
                    http(port, "GET", "/readyz", "")
                        .unwrap()
                        .starts_with("HTTP/1.1 200")
                );
            }
            eprintln!(
                "actual typed binary resident={resident} bounded={bounded} streamed={streamed} cuda={cuda} hybrid={hybrid} additive={additive}: loopback chat/SSE returned HTTP 200; toy unqualified"
            );
            drop(server);
        }
    }
}
