//! Pure inspection on authored tiny models, not physical/backend qualification.
use serde_json::Value;
use std::{path::PathBuf, process::Command};
use tritium_spec::TernaryBackend;
#[path = "portable_fixture/mod.rs"]
mod fixture;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "inspect-resource-{}-{}",
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

fn standard_tied(additive: bool) -> Vec<u8> {
    use tritium_format::{TritBlob, TritBlobKind};
    use tritium_schema::{BlobId, BlobReference, TensorStorage};
    let bytes = fixture::package(false, additive);
    let model = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
    let mut manifest = model.manifest().clone();
    manifest.levels[0]
        .tensors
        .retain(|t| t.name != "lm_head.weight");
    let mut blobs = Vec::new();
    for asset in &mut manifest.assets {
        let mut bytes = model.asset(&asset.name).unwrap().to_vec();
        if asset.name == "config.json" {
            let mut config: Value = serde_json::from_slice(&bytes).unwrap();
            config["tie_word_embeddings"] = true.into();
            bytes = serde_json::to_vec(&config).unwrap();
        }
        asset.blob = BlobReference {
            id: BlobId::from_bytes(*blake3::hash(&bytes).as_bytes()),
            shard: None,
        };
        blobs.push((TritBlobKind::Asset(asset.name.clone()), bytes));
    }
    for tensor in &manifest.levels[0].tensors {
        let (_, bytes) = model.tensor("test", &tensor.name).unwrap();
        let kind = match tensor.storage {
            TensorStorage::Additive { .. } => TritBlobKind::AdditiveTensor,
            TensorStorage::Dense { .. } => TritBlobKind::DenseTensor(tritium_core::DType::F32),
            _ => panic!("fixture storage"),
        };
        blobs.push((kind, bytes.to_vec()));
    }
    let borrowed = blobs
        .iter()
        .map(|(kind, bytes)| TritBlob {
            kind: kind.clone(),
            bytes,
        })
        .collect::<Vec<_>>();
    tritium_format::write_trit_package(
        &tritium_format::trit_model::encode_model_manifest(&manifest).unwrap(),
        &borrowed,
    )
    .unwrap()
}

fn command(path: &std::path::Path, backend: Option<&str>) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tritium"));
    command
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("TRITIUM_CHECK_ONLY", "1")
        .args(["--evidence", "off", "inspect-trit", "--model"])
        .arg(path)
        .args([
            "--level",
            "test",
            "--max-context",
            "16",
            "--max-segment-tokens",
            "2",
            "--max-host-cache-payload-bytes",
            "1000000",
        ]);
    if let Some(backend) = backend {
        command.args(["--backend", backend]);
    }
    command
}

fn check_inspection(backend: &str) {
    for hybrid in [false, true] {
        for additive in [false, true] {
            for tied in [false, true] {
                if hybrid && tied {
                    continue;
                }
                let root = Scratch::new();
                let path = root.0.join("model.trit");
                let bytes = if tied {
                    standard_tied(additive)
                } else {
                    fixture::package(hybrid, additive)
                };
                std::fs::write(&path, &bytes).unwrap();
                let mut planned = 0;
                let mut native_allowed = true;
                let expected = tritium_nn::inspect_trit_file_streamed(
                    &path,
                    "test",
                    tritium_format::TritPackageReadLimits {
                        max_source_bytes: 1_000_000,
                        max_manifest_bytes: 128_000,
                        max_directory_bytes: 128_000,
                    },
                    tritium_nn::Qwen35TritReadLimits {
                        max_config_bytes: 16_000,
                        max_tensor_bytes: 16_000,
                        max_decoded_tensor_bytes: 16_000,
                    },
                    tritium_nn::StreamLimits {
                        max_context: 16,
                        max_segment_tokens: 2,
                        max_host_cache_payload_bytes: 1_000_000,
                    },
                    tritium_schema::TensorUploadPolicy::default(),
                    |metadata, encoding| {
                        planned += 1;
                        let plan = match backend {
                            "cpu" => tritium_cpu::CpuBackend.tensor_stream_plan(metadata, encoding),
                            #[cfg(feature = "cuda")]
                            "cuda" => {
                                tritium_cuda::CudaBackend::plan_tensor_stream(metadata, encoding)
                            }
                            _ => panic!("test backend"),
                        }?;
                        if let Some(plan) = plan {
                            native_allowed &= tritium_spec::validate_tensor_upload_policy(
                                plan.upload.caps,
                                tritium_schema::TensorUploadPolicy {
                                    native_only: true,
                                    max_payload_bytes: None,
                                },
                            )
                            .is_ok();
                        }
                        Ok(plan)
                    },
                )
                .unwrap();
                if !hybrid {
                    assert_eq!(planned, if tied { 8 } else { 9 });
                }
                let output = command(&path, Some(backend)).output().unwrap();
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
                let report: Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(report["status"], "resource-planned-not-runtime-admitted");
                assert_eq!(report["weights_uploaded"], false);
                assert_eq!(report["backend_initialized"], false);
                assert_eq!(report["backend"], backend);
                assert_eq!(report["backend_payload_bytes"], expected.payload_bytes());
                for (field, cost) in [
                    ("payload_bytes", expected.payload_bytes()),
                    ("source_workspace_bytes", expected.source_workspace_bytes()),
                    ("host_workspace_bytes", expected.host_workspace_bytes()),
                    ("device_workspace_bytes", expected.device_workspace_bytes()),
                    ("retained_host_bytes", expected.retained_host_bytes()),
                ] {
                    assert_eq!(report["resource_plan"][field], cost);
                }
                assert_eq!(std::fs::read(&path).unwrap(), bytes);
                let source_only = command(&path, None).output().unwrap();
                assert!(source_only.status.success());
                let source_only: Value = serde_json::from_slice(&source_only.stdout).unwrap();
                assert_eq!(source_only["resource_plan"], Value::Null);
                assert_eq!(source_only["backend_payload_bytes"], Value::Null);
                let refused = command(&path, Some(backend))
                    .args(["--max-output-bytes", "16"])
                    .output()
                    .unwrap();
                assert!(!refused.status.success());
                assert!(refused.stdout.is_empty());
                let refused = command(&path, Some(backend))
                    .args(["--max-tensor-payload-bytes", "1"])
                    .output()
                    .unwrap();
                assert!(!refused.status.success());
                assert!(refused.stdout.is_empty());
                let native = command(&path, Some(backend))
                    .arg("--native-only")
                    .output()
                    .unwrap();
                assert_eq!(
                    native.status.success(),
                    native_allowed,
                    "{}",
                    String::from_utf8_lossy(&native.stderr)
                );
                if !native_allowed {
                    assert!(native.stdout.is_empty());
                }
                if backend == "cpu" {
                    let ledgers: [tritium_runtime::TensorPayloadBudget; 5] =
                        std::array::from_fn(|_| {
                            tritium_runtime::TensorPayloadBudget::new(1_000_000)
                        });
                    let mut recorder = tritium_evidence::EvidenceRecorder::new(
                        "inspect-parity",
                        [
                            tritium_evidence::TENSOR_UPLOADED_REGISTRATION,
                            tritium_evidence::MODEL_LOADED_REGISTRATION,
                        ],
                    )
                    .unwrap();
                    let loaded = tritium_nn::load_streamed_with_resources(
                        tritium_nn::ModelSource::TritFile {
                            path: &path,
                            level: "test",
                            receiving: tritium_format::TritPackageReadLimits {
                                max_source_bytes: 1_000_000,
                                max_manifest_bytes: 128_000,
                                max_directory_bytes: 128_000,
                            },
                            limits: tritium_nn::Qwen35TritReadLimits {
                                max_config_bytes: 16_000,
                                max_tensor_bytes: 16_000,
                                max_decoded_tensor_bytes: 16_000,
                            },
                        },
                        Box::new(tritium_cpu::CpuBackend),
                        tritium_schema::TensorUploadPolicy::default(),
                        tritium_nn::ModelLoadResources {
                            payload: &ledgers[0],
                            source_workspace: &ledgers[1],
                            host_workspace: &ledgers[2],
                            device_workspace: &ledgers[3],
                            retained_host: &ledgers[4],
                        },
                        tritium_nn::StreamLimits {
                            max_context: 16,
                            max_segment_tokens: 2,
                            max_host_cache_payload_bytes: 1_000_000,
                        },
                        tritium_runtime::TensorUploadContext {
                            recorder: &mut recorder,
                            span: "load",
                            parent: None,
                            logical_time: 0,
                        },
                    )
                    .unwrap();
                    let actual = loaded.runner.resource_admission().unwrap();
                    assert_eq!(
                        (
                            actual.payload_bytes,
                            actual.source_workspace_bytes,
                            actual.host_workspace_bytes,
                            actual.device_workspace_bytes,
                            actual.retained_host_bytes
                        ),
                        (
                            expected.payload_bytes(),
                            expected.source_workspace_bytes(),
                            expected.host_workspace_bytes(),
                            expected.device_workspace_bytes(),
                            expected.retained_host_bytes()
                        )
                    );
                    drop(loaded);
                    assert!(ledgers.iter().all(|b| b.used_bytes() == 0));
                }
            }
        }
    }
}

#[test]
fn cpu_inspection_and_actual_loader_share_standard_tied_and_hybrid_resource_costs() {
    check_inspection("cpu");
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_inspection_without_visible_devices_is_pure_structural_planning() {
    check_inspection("cuda");
}

#[test]
fn backend_inspection_requires_complete_stream_policy_and_linked_pure_planner() {
    let missing = "/does-not-exist.trit";
    let output = Command::new(env!("CARGO_BIN_EXE_tritium"))
        .args([
            "inspect-trit",
            "--model",
            missing,
            "--level",
            "test",
            "--backend",
            "cpu",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("--max-context"));
    #[cfg(not(feature = "cuda"))]
    {
        let output = command(std::path::Path::new(missing), Some("cuda"))
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("requires a build with --features cuda")
        );
    }
}
