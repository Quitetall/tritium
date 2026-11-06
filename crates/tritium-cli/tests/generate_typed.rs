//! Authored miniature graphs; no official-model/quality qualification.
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    process::Command,
};
#[path = "portable_fixture/mod.rs"]
mod fixture;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "generate-typed-{}-{}",
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
fn config(path: &Path) -> Value {
    json!({"path":path,"level":"test","max_source_bytes":1_000_000,
        "max_manifest_bytes":128_000,"max_directory_bytes":128_000,"max_config_bytes":16_000,
        "max_tensor_bytes":16_000,"max_decoded_tensor_bytes":16_000,"max_payload_bytes":1_000_000,"native_only":false,
        "stream_limits":{"max_context":16,"max_segment_tokens":2,"max_host_cache_payload_bytes":1_000_000},
        "streamed_resources":{"max_source_workspace_bytes":65_536,"max_host_workspace_bytes":1_000_000,"max_device_workspace_bytes":0,"max_retained_host_bytes":1_000_000}})
}
fn command(path: &Path, backend: &str) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_tritium"));
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("TRITIUM_") {
            cmd.env_remove(name);
        }
    }
    cmd.args(["--evidence", "off", "generate", "--typed"])
        .arg(path)
        .args(["--backend", backend, "--eos", "4294967295"]);
    cmd
}
fn load(config: &Value, root: &Scratch) -> tritium_nn::LoadedModel {
    let path = root.0.join("oracle.json");
    std::fs::write(&path, config.to_string()).unwrap();
    let mut recorder = tritium_evidence::EvidenceRecorder::new(
        "typed-test",
        [
            tritium_evidence::TENSOR_UPLOADED_REGISTRATION,
            tritium_evidence::MODEL_LOADED_REGISTRATION,
        ],
    )
    .unwrap();
    tritium_nn::TypedLoadConfig::read(&path)
        .unwrap()
        .load(
            Box::new(tritium_cpu::CpuBackend::new()),
            tritium_runtime::TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        )
        .unwrap()
}
fn tokens(output: &std::process::Output) -> Vec<u32> {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|l| serde_json::from_str(l).ok())
        .expect("token array")
}
fn check_binary(backend: &str, resident: bool) {
    for hybrid in [false, true] {
        for additive in [false, true] {
            let root = Scratch::new();
            let model = root.0.join("model.trit");
            let path = root.0.join("load.json");
            let ids = root.0.join("ids.json");
            std::fs::write(&model, fixture::package(hybrid, additive)).unwrap();
            std::fs::write(&ids, "[1,2,1,2,1,2]").unwrap();
            let mut request = config(&model);
            if backend == "cuda" {
                request["streamed_resources"]["max_device_workspace_bytes"] = 4.into();
            }
            if resident {
                request["execution"] =
                    json!({"kind":"resident_device_f32","max_device_payload_bytes":1_000_000});
            }
            std::fs::write(&path, request.to_string()).unwrap();
            let mut compatibility = request.clone();
            compatibility.as_object_mut().unwrap().remove("execution");
            compatibility
                .as_object_mut()
                .unwrap()
                .remove("streamed_resources");
            compatibility["stream_limits"]["max_segment_tokens"] = 16.into();
            let mut oracle = load(&compatibility, &root);
            for max_new in [0, 3, 10] {
                let expected = oracle
                    .runner
                    .generate(&[1, 2, 1, 2, 1, 2], max_new, u32::MAX)
                    .unwrap();
                let output = command(&path, backend)
                    .arg("--tokens")
                    .arg(&ids)
                    .args(["--max-new", &max_new.to_string()])
                    .output()
                    .unwrap();
                assert_eq!(
                    tokens(&output),
                    expected,
                    "{backend} hybrid={hybrid} additive={additive} max_new={max_new}"
                );
                assert!(
                    String::from_utf8_lossy(&output.stderr).contains("resource_admission=Some")
                );
                if max_new == 3 {
                    let text = command(&path, backend)
                        .args([
                            "--prompt",
                            "hello partner hello partner hello partner",
                            "--max-new",
                            "3",
                        ])
                        .output()
                        .unwrap();
                    assert_eq!(tokens(&text), expected);
                }
            }
            eprintln!(
                "typed generation {backend} hybrid={hybrid} additive={additive}: authored fixture only"
            );
        }
    }
}
#[test]
fn typed_cpu_binary_matches_decoded_oracle_across_segments_and_exact_context_boundary() {
    check_binary("cpu", false);
}
#[cfg(feature = "cuda")]
#[test]
fn physical_typed_cuda_binary_matches_small_cpu_oracle() {
    tritium_cuda::CudaBackend::new(0).expect("physical typed generation requires CUDA");
    check_binary("cuda", false);
}
#[cfg(feature = "cuda")]
#[test]
fn physical_typed_resident_binary_streams_weights_without_host_fallback() {
    tritium_cuda::CudaBackend::new(0).expect("requires physical CUDA");
    check_binary("cuda", true);
}
#[test]
fn typed_configuration_and_selection_refusals_emit_no_token_report() {
    let root = Scratch::new();
    let path = root.0.join("load.json");
    for case in 0..8 {
        let mut request = config(&root.0.join("missing.trit"));
        match case {
            0 => {
                request
                    .as_object_mut()
                    .unwrap()
                    .remove("streamed_resources");
            }
            1 => {
                request.as_object_mut().unwrap().remove("stream_limits");
            }
            2 => request["streamed_resources"]["max_retained_host_bytes"] = 0.into(),
            3 => request["unknown"] = true.into(),
            4 => {
                request["execution"] =
                    json!({"kind":"resident_device_f32","max_device_payload_bytes":1_000_001})
            }
            _ => {}
        }
        std::fs::write(&path, request.to_string()).unwrap();
        let mut cmd = command(&path, "not-linked");
        cmd.args([
            "--prompt",
            "hello",
            "--max-new",
            if case == 7 {
                "18446744073709551615"
            } else {
                "1"
            },
        ]);
        match case {
            5 => {
                cmd.args(["--model", "legacy.gguf"]);
            }
            6 => {
                cmd.args(["--greedy", "false"]);
            }
            _ => {}
        }
        let result = cmd.output().unwrap();
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success(), "case{case}");
        assert!(result.stdout.is_empty());
        assert!(
            !error.contains("is not in the registry"),
            "case{case}:{error}"
        );
    }
    let no_eos = Command::new(env!("CARGO_BIN_EXE_tritium"))
        .args(["generate", "--typed"])
        .arg(&path)
        .args(["--prompt", "hello"])
        .output()
        .unwrap();
    assert!(!no_eos.status.success());
    assert!(String::from_utf8_lossy(&no_eos.stderr).contains("--eos"));
}

#[test]
fn typed_token_file_is_bounded_before_backend_initialization() {
    let root = Scratch::new();
    let path = root.0.join("load.json");
    std::fs::write(&path, config(&root.0.join("missing.trit")).to_string()).unwrap();
    let ids = root.0.join("oversized.json");
    std::fs::File::create(&ids)
        .unwrap()
        .set_len(16 * 1024 * 1024 + 1)
        .unwrap();
    let result = command(&path, "not-linked")
        .arg("--tokens")
        .arg(&ids)
        .args(["--max-new", "1"])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&result.stderr).contains("is not in the registry"));
}

#[test]
fn typed_binary_budget_refusals_emit_no_token_report() {
    let root = Scratch::new();
    let model = root.0.join("model.trit");
    let path = root.0.join("load.json");
    std::fs::write(&model, fixture::package(true, true)).unwrap();
    for field in [
        "max_payload_bytes",
        "max_source_workspace_bytes",
        "max_host_workspace_bytes",
        "max_retained_host_bytes",
    ] {
        let mut request = config(&model);
        if field == "max_payload_bytes" {
            request[field] = 1.into();
        } else {
            request["streamed_resources"][field] = 1.into();
        }
        std::fs::write(&path, request.to_string()).unwrap();
        let result = command(&path, "cpu")
            .args(["--prompt", "hello", "--max-new", "1"])
            .output()
            .unwrap();
        assert!(!result.status.success(), "{field}");
        assert!(result.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&result.stderr).contains("model resource admission"),
            "{field}:{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
#[test]
fn bounded_runner_validates_before_mutation_and_preserves_eos_zero_and_retry_policy() {
    for hybrid in [false, true] {
        let root = Scratch::new();
        let model = root.0.join("model.trit");
        std::fs::write(&model, fixture::package(hybrid, true)).unwrap();
        let mut loaded = load(&config(&model), &root);
        let runner = &mut loaded.runner;
        assert!(
            runner
                .prefill_segmented_cancellable(&[7], &|| true)
                .unwrap()
                .is_none()
        );
        assert_eq!(runner.cache_len(), 0);
        assert_eq!(
            runner.generate_bounded(&[1, 2], 0, u32::MAX).unwrap(),
            Vec::<u32>::new()
        );
        assert_eq!(runner.cache_len(), 0);
        let expected = runner
            .generate_bounded(&[1, 2, 1, 2, 1, 2], 3, u32::MAX)
            .unwrap();
        assert_eq!(expected.len(), 3);
        assert_eq!(
            runner.cache_len(),
            8,
            "unused final token must not be forwarded"
        );
        let prefix = runner.cache_len();
        assert!(runner.generate_bounded(&[7], 1, u32::MAX).is_err());
        assert_eq!(runner.cache_len(), prefix);
        assert!(runner.generate_bounded(&[1], usize::MAX, u32::MAX).is_err());
        assert_eq!(runner.cache_len(), prefix);
        assert!(
            runner
                .prefill_segmented_cancellable(&[1], &|| false)
                .is_err()
        );
        assert_eq!(runner.cache_len(), prefix);
        assert!(
            runner
                .generate_bounded(&[1, 2, 1, 2, 1, 2], 3, expected[0])
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            runner
                .generate_bounded(&[1, 2, 1, 2, 1, 2], 3, u32::MAX)
                .unwrap(),
            expected
        );
    }
}
