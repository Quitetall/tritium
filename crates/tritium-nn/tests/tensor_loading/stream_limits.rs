//! Real public loader and runner admission, not physical-memory qualification.
use super::*;
use tritium_evidence::{EvidenceEvent, MODEL_LOADED_REGISTRATION};
use tritium_nn::{ModelSource, StreamLimits};

struct HeadProbe {
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    fail_at: Option<usize>,
    cancel_at: Option<usize>,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
impl TernaryBackend for HeadProbe {
    fn device_id(&self) -> &str {
        "cpu-prefill-head-probe"
    }
    fn capabilities(&self) -> tritium_spec::DeviceCaps {
        tritium_cpu::CpuBackend::new().capabilities()
    }
    fn tensor_caps(
        &self,
        tensor: TensorView<'_>,
    ) -> Result<Option<tritium_spec::TensorCaps>, tritium_spec::BackendError> {
        tritium_cpu::CpuBackend::new().tensor_caps(tensor)
    }
    fn upload_tensor(
        &self,
        tensor: TensorView<'_>,
    ) -> Result<Box<dyn tritium_spec::DeviceBuffer>, tritium_spec::BackendError> {
        tritium_cpu::CpuBackend::new().upload_tensor(tensor)
    }
    fn upload_weights(
        &self,
        packed: &[u8],
        shape: tritium_core::GemmShape,
        format: tritium_core::TernaryFormat,
    ) -> Result<Box<dyn tritium_spec::DeviceBuffer>, tritium_spec::BackendError> {
        tritium_cpu::CpuBackend::new().upload_weights(packed, shape, format)
    }
    fn mpgemm(&self, p: tritium_spec::MpGemm<'_>) -> Result<(), tritium_spec::BackendError> {
        tritium_cpu::CpuBackend::new().mpgemm(p)
    }
    fn embed_rows(
        &self,
        tensor: &dyn tritium_spec::DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), tritium_spec::BackendError> {
        tritium_cpu::CpuBackend::new().embed_rows(tensor, ids, out)
    }
    fn matmul(&self, p: tritium_spec::TensorMatmul<'_>) -> Result<(), tritium_spec::BackendError> {
        use std::sync::atomic::Ordering;
        // This fixture has vocab10, hidden8, query12, KV6 and FF16: only
        // the untied final head produces ten values at batch1.
        let head = p.batch == 1 && p.out.len() == 10;
        let call = if head {
            self.calls.fetch_add(1, Ordering::SeqCst) + 1
        } else {
            0
        };
        if head && self.fail_at == Some(call) {
            return Err(tritium_spec::BackendError::Backend(
                "injected prefill head failure".into(),
            ));
        }
        tritium_cpu::CpuBackend::new().matmul(p)?;
        if head && self.cancel_at == Some(call) {
            self.cancelled.store(true, Ordering::SeqCst);
        }
        Ok(())
    }
}

#[test]
fn segmented_prefill_head_projection_count_and_transaction_contract() {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    let (_hf, package) = fixture_with_tie(false);
    let bytes = package.bytes();
    let policy = StreamLimits {
        max_context: 6,
        max_segment_tokens: 2,
        max_host_cache_payload_bytes: 288,
    };
    for (fail_at, cancel_at) in [
        (None, None),
        (Some(1), None),
        (Some(2), None),
        (None, Some(1)),
        (None, Some(2)),
    ] {
        let calls = Arc::new(AtomicUsize::new(0));
        let cancelled = Arc::new(AtomicBool::new(false));
        let budget = TensorPayloadBudget::new(100_000);
        let source = ModelSource::Trit {
            bytes: &bytes,
            level: "standard",
        };
        let mut loaded = bounded(
            source,
            Box::new(HeadProbe {
                calls: calls.clone(),
                fail_at,
                cancel_at,
                cancelled: cancelled.clone(),
            }),
            &budget,
            policy,
            &mut record(),
        )
        .unwrap();
        let mut oracle = bounded(
            source,
            Box::new(tritium_cpu::CpuBackend::new()),
            &budget,
            policy,
            &mut record(),
        )
        .unwrap();
        oracle.runner.forward(&[1, 2], &[0, 1]).unwrap();
        let expected = oracle.runner.forward(&[3, 4], &[2, 3]).unwrap();
        assert!(
            loaded
                .runner
                .prefill_segmented_cancellable(&[u32::MAX], &|| true)
                .unwrap()
                .is_none()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let result = loaded
            .runner
            .prefill_segmented_cancellable(&[1, 2, 3, 4], &|| cancelled.load(Ordering::SeqCst));
        if fail_at.is_some() {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("injected prefill head failure")
            );
        } else if cancel_at.is_some() {
            assert!(result.unwrap().is_none());
        } else {
            assert_eq!(result.unwrap().unwrap(), expected);
        }
        let boundary = fail_at.or(cancel_at);
        let prefix = boundary.map_or(4, |n| 2 * (n - 1));
        assert_eq!(loaded.runner.cache_len(), prefix);
        // Final-row logits are discarded by the outer loop, but head errors
        // and cancellation still guard each segment's publication. Deferring
        // those checks is a contract change, not a local dead-code removal.
        assert_eq!(calls.load(Ordering::SeqCst), boundary.unwrap_or(2));
        cancelled.store(false, Ordering::SeqCst);
        if prefix == 0 {
            loaded.runner.forward(&[1, 2], &[0, 1]).unwrap();
        }
        if prefix != 4 {
            assert_eq!(loaded.runner.forward(&[3, 4], &[2, 3]).unwrap(), expected);
        }
        for token in [5, 6] {
            assert_eq!(
                loaded.runner.step(token).unwrap(),
                oracle.runner.step(token).unwrap()
            );
        }
        drop(loaded);
        drop(oracle);
        assert_eq!(budget.used_bytes(), 0);
    }
}

pub(super) fn tokenizer() -> Vec<u8> {
    let vocabulary = (0..10).map(|i| (format!("t{i}"), i as u32)).collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("t0".into())
        .build()
        .unwrap();
    tokenizers::Tokenizer::new(model)
        .to_string(false)
        .unwrap()
        .into_bytes()
}
fn record() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "bounded-standard",
        [TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION],
    )
    .unwrap()
}
fn fixture() -> (HfFixture, TritFixture) {
    fixture_with_tie(true)
}
fn fixture_with_tie(tied: bool) -> (HfFixture, TritFixture) {
    let hf = HfFixture::new("qwen3", tied, 6, None);
    let mut package = TritFixture::new(&hf, None);
    let tokenizer = tokenizer();
    std::fs::write(hf.dir.join("tokenizer.json"), &tokenizer).unwrap();
    let tokenizer_config = br#"{"bos_token":"t8","eos_token":"t9"}"#;
    std::fs::write(hf.dir.join("tokenizer_config.json"), tokenizer_config).unwrap();
    package.asset("tokenizer.json", tokenizer);
    package.asset("tokenizer_config.json", tokenizer_config.to_vec());
    (hf, package)
}
fn limits() -> StreamLimits {
    StreamLimits {
        max_context: 4,
        max_segment_tokens: 2,
        max_host_cache_payload_bytes: 192,
    }
}
fn bounded(
    source: ModelSource<'_>,
    backend: Box<dyn TernaryBackend>,
    budget: &TensorPayloadBudget,
    limits: StreamLimits,
    record: &mut EvidenceRecorder,
) -> Result<tritium_nn::LoadedModel, tritium_nn::NnError> {
    tritium_nn::load_with_limits(
        source,
        backend,
        TensorUploadPolicy::default(),
        budget,
        limits,
        TensorUploadContext {
            recorder: record,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
}
fn sources<'a>(
    hf: &'a HfFixture,
    bytes: &'a [u8],
    path: &'a std::path::Path,
) -> [ModelSource<'a>; 3] {
    [
        ModelSource::HuggingFace(&hf.dir),
        ModelSource::Trit {
            bytes,
            level: "standard",
        },
        ModelSource::TritFile {
            path,
            level: "standard",
            receiving: tritium_format::TritPackageReadLimits {
                max_source_bytes: 1_000_000,
                max_manifest_bytes: 100_000,
                max_directory_bytes: 100_000,
            },
            limits: tritium_nn::Qwen35TritReadLimits {
                max_config_bytes: 100_000,
                max_tensor_bytes: 1_000_000,
                max_decoded_tensor_bytes: 1_000_000,
            },
        },
    ]
}

#[test]
fn invalid_stream_policy_rejects_all_standard_sources_before_upload() {
    let (hf, package) = fixture();
    let bytes = package.bytes();
    let path = hf.dir.join("bounded.trit");
    std::fs::write(&path, &bytes).unwrap();
    for source in sources(&hf, &bytes, &path) {
        for bad in [
            StreamLimits {
                max_context: 0,
                ..limits()
            },
            StreamLimits {
                max_segment_tokens: 0,
                ..limits()
            },
            StreamLimits {
                max_host_cache_payload_bytes: 0,
                ..limits()
            },
            StreamLimits {
                max_segment_tokens: 5,
                ..limits()
            },
            StreamLimits {
                max_context: 17,
                max_host_cache_payload_bytes: u64::MAX,
                ..limits()
            },
            StreamLimits {
                max_host_cache_payload_bytes: 191,
                ..limits()
            },
            StreamLimits {
                max_context: usize::MAX,
                max_host_cache_payload_bytes: u64::MAX,
                ..limits()
            },
        ] {
            let (backend, uploads) = upload_probe();
            let budget = TensorPayloadBudget::new(100_000);
            let mut evidence = record();
            let error = bounded(source, backend, &budget, bad, &mut evidence)
                .err()
                .expect("bad policy must fail");
            assert!(
                matches!(error, tritium_nn::NnError::ResourceExhausted(_)),
                "{error:?}"
            );
            assert_eq!(uploads.load(std::sync::atomic::Ordering::SeqCst), 0);
            assert_eq!(budget.used_bytes(), 0);
            assert!(evidence.events().is_empty());
        }
    }
}

#[test]
fn bounded_standard_stream_preserves_logits_and_enforces_real_cursor_limits() {
    let (hf, package) = fixture();
    let bytes = package.bytes();
    let path = hf.dir.join("bounded.trit");
    std::fs::write(&path, &bytes).unwrap();
    for source in sources(&hf, &bytes, &path) {
        let budget = TensorPayloadBudget::new(100_000);
        let mut oracle = tritium_nn::load(
            source,
            Box::new(tritium_cpu::CpuBackend::new()),
            TensorUploadPolicy::default(),
            &budget,
            TensorUploadContext {
                recorder: &mut record(),
                span: "load",
                parent: None,
                logical_time: 0,
            },
        )
        .unwrap();
        let mut loaded = bounded(
            source,
            Box::new(tritium_cpu::CpuBackend::new()),
            &budget,
            limits(),
            &mut record(),
        )
        .unwrap();
        assert!(oracle.runner.cache_admission().is_none());
        let admission = loaded.runner.cache_admission().unwrap();
        assert_eq!(admission.limits(), limits());
        assert_eq!(admission.attention_kv_bytes(), 192); // 2 * context4 * KV-head1 * actual-head-dim6 * f32.
        assert_eq!(admission.delta_cache_bytes(), 0);
        assert_eq!(admission.logical_host_cache_payload_bytes(), 192);
        assert_eq!(loaded.runner.max_context(), 4);
        assert_eq!(loaded.runner.architecture().max_context(), 16);
        assert_eq!(
            loaded.runner.prefill(&[1, 3]).unwrap(),
            oracle.runner.prefill(&[1, 3]).unwrap()
        );
        assert!(loaded.runner.forward(&[2, 4, 5], &[2, 3, 4]).is_err());
        assert_eq!(loaded.runner.cache_len(), 2);
        for token in [4, 5] {
            assert_eq!(
                loaded.runner.step(token).unwrap(),
                oracle.runner.step(token).unwrap()
            );
        }
        if let tritium_nn::ArchitectureSpec::Standard { config, .. } = &mut loaded.architecture {
            config.n_ctx = u32::MAX;
        }
        assert!(loaded.runner.step(6).is_err());
        assert_eq!(loaded.runner.cache_len(), 4);
        loaded.runner.reset();
        assert_eq!(loaded.runner.cache_admission(), Some(admission));
        assert!(loaded.runner.prefill(&[1, 2, 3]).is_err());
        assert_eq!(loaded.runner.cache_len(), 0);
        assert!(
            loaded
                .runner
                .forward_cancellable(&[u32::MAX], &[usize::MAX], &|| true)
                .unwrap()
                .is_none()
        );
        assert_eq!(loaded.runner.cache_len(), 0);
        oracle.runner.reset();
        assert_eq!(
            loaded.runner.prefill(&[1, 3]).unwrap(),
            oracle.runner.prefill(&[1, 3]).unwrap()
        );
        drop(loaded);
        drop(oracle);
        assert_eq!(budget.used_bytes(), 0);
    }
}

#[test]
fn standard_retained_resources_count_tied_embedding_once_and_refuse_other_carriers() {
    for tied in [true, false] {
        let (hf, package) = fixture_with_tie(tied);
        let bytes = package.bytes();
        let path = hf.dir.join("resource-standard.trit");
        std::fs::write(&path, &bytes).unwrap();
        for source in sources(&hf, &bytes, &path) {
            for streamed in [false, true] {
                let total = TensorPayloadBudget::new(1_000_000);
                let resources = tritium_nn::ModelLoadResources {
                    payload: &total,
                    source_workspace: &total,
                    host_workspace: &total,
                    device_workspace: &total,
                    retained_host: &total,
                };
                let mut evidence = record();
                let load = if streamed {
                    tritium_nn::load_streamed_with_resources
                } else {
                    tritium_nn::load_with_resources
                };
                let result = load(
                    source,
                    Box::new(tritium_cpu::CpuBackend::new()),
                    TensorUploadPolicy::default(),
                    resources,
                    limits(),
                    TensorUploadContext {
                        recorder: &mut evidence,
                        span: "load",
                        parent: None,
                        logical_time: 0,
                    },
                );
                if matches!(source, ModelSource::HuggingFace(_)) {
                    assert!(
                        result
                            .err()
                            .unwrap()
                            .to_string()
                            .contains("requires a retained typed source")
                    );
                    assert!(evidence.events().is_empty());
                } else {
                    let mut loaded = result.unwrap();
                    let admission = loaded.runner.resource_admission().unwrap();
                    let uploads = evidence
                        .events()
                        .iter()
                        .filter(|e| e.schema == tritium_evidence::TensorUploaded::SCHEMA_ID)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        uploads
                            .iter()
                            .filter(|e| e.payload["tensor"] == "lm_head.weight")
                            .count(),
                        usize::from(!tied)
                    );
                    assert_eq!(
                        admission.payload_bytes,
                        uploads
                            .iter()
                            .map(|e| e.payload["caps"]["payload_bytes"].as_u64().unwrap())
                            .sum::<u64>()
                    );
                    assert_eq!(
                        total.used_bytes(),
                        admission.payload_bytes + admission.retained_host_bytes
                    );
                    let oracle_budget = TensorPayloadBudget::new(1_000_000);
                    let mut oracle = bounded(
                        source,
                        Box::new(tritium_cpu::CpuBackend::new()),
                        &oracle_budget,
                        limits(),
                        &mut record(),
                    )
                    .unwrap();
                    assert_eq!(
                        loaded.runner.prefill(&[1, 2]).unwrap(),
                        oracle.runner.prefill(&[1, 2]).unwrap()
                    );
                    drop(loaded);
                }
                assert_eq!(total.used_bytes(), 0);
            }
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn bounded_actual_cuda_stream_matches_cpu_and_rejects_context_overrun() {
    let (hf, package) = fixture();
    let bytes = package.bytes();
    let source = ModelSource::Trit {
        bytes: &bytes,
        level: "standard",
    };
    let budget = TensorPayloadBudget::new(100_000);
    let mut gpu = bounded(
        source,
        Box::new(
            tritium_cuda::CudaBackend::new(0)
                .expect("bounded CUDA stream check requires physical GPU"),
        ),
        &budget,
        limits(),
        &mut record(),
    )
    .unwrap();
    let mut cpu = bounded(
        source,
        Box::new(tritium_cpu::CpuBackend::new()),
        &budget,
        limits(),
        &mut record(),
    )
    .unwrap();
    assert_eq!(gpu.runner.cache_admission(), cpu.runner.cache_admission());
    assert_eq!(
        gpu.runner.prefill(&[1, 3]).unwrap(),
        cpu.runner.prefill(&[1, 3]).unwrap()
    );
    for token in [4, 5] {
        assert_eq!(
            gpu.runner.step(token).unwrap(),
            cpu.runner.step(token).unwrap()
        );
    }
    assert!(gpu.runner.step(6).is_err());
    assert_eq!(gpu.runner.cache_len(), 4);
    gpu.runner.reset();
    assert!(gpu.runner.prefill(&[1, 2, 3]).is_err());
    assert_eq!(gpu.runner.cache_len(), 0);
    drop(gpu);
    drop(cpu);
    drop(hf);
    assert_eq!(budget.used_bytes(), 0);
}
