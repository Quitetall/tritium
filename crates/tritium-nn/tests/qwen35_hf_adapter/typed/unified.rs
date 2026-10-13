//! Actual public loader/runner tests. Small CPU/reference fixtures, not Qwen quality.
use super::*;
use std::sync::atomic::AtomicBool;
use tritium_evidence::{
    EvidenceEvent, MODEL_LOADED_REGISTRATION, ModelLoadReceiptView, verify_jsonl_with_registry,
};
use tritium_nn::{LayerKind, ModelSource, Tokenizer};

#[path = "unified/stream_limits.rs"]
mod stream_limits;

#[cfg(feature = "cuda")]
#[path = "unified/resident.rs"]
mod resident;

#[test]
fn common_owned_graph_matches_legacy_typed_hybrid_logits_bit_for_bit() {
    for uniform in [false, true] {
        let bytes = text_package(uniform, false).bytes();
        for reference in [false, true] {
            let backend = || -> Box<dyn TernaryBackend> {
                if reference {
                    Box::new(tritium_testkit::ReferenceBackend::new())
                } else {
                    Box::new(tritium_cpu::CpuBackend::new())
                }
            };
            let budget = TensorPayloadBudget::new(1_000_000);
            let mut loaded = load(
                ModelSource::Trit {
                    bytes: &bytes,
                    level: "typed",
                },
                backend(),
                &budget,
                &mut recorder(),
            )
            .unwrap();
            let legacy_backend = backend();
            let mut evidence = recorder();
            let mut reader = TritModelReader::open(Cursor::new(&bytes), receiving()).unwrap();
            let (config, weights, mtp_weights) = {
                let mut loader = TensorLoader::new(
                    legacy_backend.as_ref(),
                    TensorUploadPolicy::default(),
                    &budget,
                    TensorUploadContext {
                        recorder: &mut evidence,
                        span: "load",
                        parent: None,
                        logical_time: 0,
                    },
                );
                Qwen35TextWeights::load_trit_admitted(&mut reader, "typed", &mut loader, limits())
                    .unwrap()
            };
            let legacy = Qwen35TextRunner::new(&config.text, weights, legacy_backend).unwrap();
            let mtp = UnverifiedQwen35Mtp::new(&legacy, mtp_weights).unwrap();
            assert_eq!(loaded.runner.mtp_status(), Some(mtp.status()));
            let mut cache = legacy.new_cache(32).unwrap();
            for tokens in [&[1, 2][..], &[4][..], &[5][..]] {
                let at = loaded.runner.cache_len();
                let positions: Vec<_> = (at..at + tokens.len()).collect();
                let actual = loaded.runner.forward(tokens, &positions).unwrap();
                let expected = legacy.forward(tokens, &mut cache).unwrap();
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected
                        .last_logits()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "uniform={uniform} reference={reference}"
                );
                assert_eq!(loaded.runner.cache_len(), cache.len());
            }
            loaded.runner.reset();
            cache.reset();
            assert_eq!(
                loaded.runner.prefill(&[1, 2]).unwrap(),
                legacy.forward(&[1, 2], &mut cache).unwrap().last_logits()
            );
            drop(loaded);
            drop(legacy);
            drop(mtp);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

fn asset(package: &mut Package, name: &str, bytes: Vec<u8>) {
    let blob = BlobReference {
        id: BlobId::from_bytes(*blake3::hash(&bytes).as_bytes()),
        shard: None,
    };
    if let Some(a) = package.manifest.assets.iter_mut().find(|a| a.name == name) {
        a.blob = blob;
    } else {
        package.manifest.assets.push(ManifestAsset {
            name: name.into(),
            blob,
        });
    }
    package
        .blobs
        .push((TritBlobKind::Asset(name.into()), bytes));
}
fn text_package(uniform: bool, bad_vocab: bool) -> Package {
    let mut package = Package::new(tensors(), uniform);
    let mut vocab = ["[UNK]", "hello", "partner", "[EOS]", "[BOS]", "t5", "t6"]
        .into_iter()
        .enumerate()
        .map(|(id, word)| (word.to_owned(), id as u32))
        .collect::<Vec<_>>();
    if bad_vocab {
        vocab.push(("invalid".into(), 7));
    }
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    asset(
        &mut package,
        "tokenizer.json",
        tokenizer.to_string(false).unwrap().into_bytes(),
    );
    asset(
        &mut package,
        "tokenizer_config.json",
        br#"{"bos_token":"[BOS]","eos_token":"[EOS]"}"#.to_vec(),
    );
    package
}
pub(super) fn add_text_assets(source: &mut InMemoryWeightSource) {
    let package = text_package(false, false);
    for (kind, bytes) in package.blobs {
        if let TritBlobKind::Asset(name) = kind {
            if name == "tokenizer.json" && bytes == b"{}" {
                continue;
            }
            source.assets.insert(name, bytes);
        }
    }
}
pub(super) fn check_imported_text_file(path: &Path, label: &str) {
    let root = Scratch::new(&format!("public-{label}"));
    write_fixture_tensors(&root.0, tensors());
    let oracle = Qwen35HfSource::open(&root.0)
        .unwrap()
        .verify_semantic_identity()
        .unwrap()
        .load_language_mtp(Box::new(tritium_cpu::CpuBackend::new()))
        .unwrap();
    for cpu in [true, false] {
        let budget = TensorPayloadBudget::new(1_000_000);
        let backend: Box<dyn TernaryBackend> = if cpu {
            Box::new(tritium_cpu::CpuBackend::new())
        } else {
            Box::new(tritium_testkit::ReferenceBackend::new())
        };
        let mut loaded = load(
            ModelSource::TritFile {
                path,
                level: "imported",
                receiving: receiving(),
                limits: limits(),
            },
            backend,
            &budget,
            &mut recorder(),
        )
        .unwrap();
        let mut cache = oracle.runner().new_cache(32).unwrap();
        let mut error = close(
            &loaded.runner.prefill(&[1, 2]).unwrap(),
            oracle
                .runner()
                .forward(&[1, 2], &mut cache)
                .unwrap()
                .last_logits(),
        );
        for token in [4, 5] {
            error = error.max(close(
                &loaded.runner.step(token).unwrap(),
                oracle
                    .runner()
                    .forward(&[token], &mut cache)
                    .unwrap()
                    .last_logits(),
            ));
        }
        eprintln!("public actual {label} cpu={cpu}: max_abs_error={error}");
        drop(loaded);
        assert_eq!(budget.used_bytes(), 0);
    }
}
fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "unified-text-load",
        [TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION],
    )
    .unwrap()
}
fn load(
    source: ModelSource<'_>,
    backend: Box<dyn TernaryBackend>,
    budget: &TensorPayloadBudget,
    evidence: &mut EvidenceRecorder,
) -> Result<tritium_nn::LoadedModel, tritium_nn::NnError> {
    tritium_nn::load(
        source,
        backend,
        TensorUploadPolicy::default(),
        budget,
        TensorUploadContext {
            recorder: evidence,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
}
fn file_source(path: &Path) -> ModelSource<'_> {
    ModelSource::TritFile {
        path,
        level: "typed",
        receiving: receiving(),
        limits: limits(),
    }
}
fn close(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    let mut error = 0.0_f32;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        error = error.max((a - b).abs());
    }
    assert!(error <= 1e-5, "public runner max error: {error}");
    error
}

#[test]
fn public_loader_unifies_file_and_borrowed_hybrid_prefill_decode_generation() {
    fn assert_send<T: Send>() {}
    assert_send::<tritium_nn::Runner>();
    for uniform in [false, true] {
        for cpu in [true, false] {
            for file in [false, true] {
                let root = Scratch::new(&format!("public-{uniform}-{cpu}-{file}"));
                write_fixture_tensors(&root.0, tensors());
                let oracle = Qwen35HfSource::open(&root.0)
                    .unwrap()
                    .verify_semantic_identity()
                    .unwrap()
                    .load_language_mtp(Box::new(tritium_cpu::CpuBackend::new()))
                    .unwrap();
                let bytes = text_package(uniform, false).bytes();
                let source = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
                let identity = LoadedSourceIdentity::Trit {
                    package_id: PackageId::from_bytes(*source.package_id().as_bytes()),
                    model_id: source.model_id("typed").unwrap(),
                    level: "typed".into(),
                };
                let path = root.0.join("model.trit");
                std::fs::write(&path, &bytes).unwrap();
                let budget = TensorPayloadBudget::new(1_000_000);
                let mut evidence = recorder();
                let backend: Box<dyn TernaryBackend> = if cpu {
                    Box::new(tritium_cpu::CpuBackend::new())
                } else {
                    Box::new(tritium_testkit::ReferenceBackend::new())
                };
                let device = backend.physical_device_id().to_owned();
                let mut loaded = load(
                    if file {
                        file_source(&path)
                    } else {
                        ModelSource::Trit {
                            bytes: &bytes,
                            level: "typed",
                        }
                    },
                    backend,
                    &budget,
                    &mut evidence,
                )
                .unwrap();
                // Only uploaded graph handles are needed after admitted load.
                std::fs::remove_file(&path).unwrap();
                assert_eq!(loaded.tokenizer.encode("hello partner").unwrap(), [1, 2]);
                assert_eq!(loaded.tokenizer.decode(&[1, 2]).unwrap(), "hello partner");
                assert_eq!((loaded.tokenizer.bos(), loaded.tokenizer.eos()), (4, 3));
                assert_eq!(loaded.runner.architecture().name(), "qwen3_5");
                assert_eq!(loaded.runner.architecture().vocab_size(), 7);
                assert_eq!(loaded.runner.architecture().max_context(), 32);
                assert_eq!(
                    loaded
                        .runner
                        .architecture()
                        .layer_kinds()
                        .collect::<Vec<_>>(),
                    [LayerKind::DeltaNet, LayerKind::Attention]
                );
                assert_eq!(
                    loaded.runner.mtp_status(),
                    Some(tritium_nn::Qwen35MtpStatus::Unverified)
                );
                assert_eq!(loaded.receipt.observation().source, identity);
                assert_eq!(loaded.receipt.observation().physical_device, device);
                assert_eq!(loaded.receipt.observation().layer_count, 2);
                assert_eq!(Some(loaded.receipt.envelope()), evidence.events().last());
                let mut cache = oracle.runner().new_cache(32).unwrap();
                assert!(loaded.runner.step(1).is_err());
                let mut error = close(
                    &loaded.runner.prefill(&[1, 2, 4]).unwrap(),
                    oracle
                        .runner()
                        .forward(&[1, 2, 4], &mut cache)
                        .unwrap()
                        .last_logits(),
                );
                for token in [5, 6] {
                    error = error.max(close(
                        &loaded.runner.step(token).unwrap(),
                        oracle
                            .runner()
                            .forward(&[token], &mut cache)
                            .unwrap()
                            .last_logits(),
                    ));
                    assert_eq!(loaded.runner.cache_len(), cache.len());
                }
                assert!(loaded.runner.prefill(&[1]).is_err());
                assert!(loaded.runner.forward(&[1], &[0]).is_err());
                assert!(loaded.runner.step(7).is_err());
                assert!(
                    loaded
                        .runner
                        .forward(&[1; 28], &(5..33).collect::<Vec<_>>())
                        .is_err()
                );
                assert!(
                    loaded
                        .runner
                        .forward_cancellable(&[], &[], &|| true)
                        .unwrap()
                        .is_none()
                );
                assert_eq!(loaded.runner.cache_len(), 5);
                close(
                    &loaded.runner.step(1).unwrap(),
                    oracle
                        .runner()
                        .forward(&[1], &mut cache)
                        .unwrap()
                        .last_logits(),
                );
                for max_new in [0, 1, 4] {
                    cache.reset();
                    let mut logits = oracle
                        .runner()
                        .forward(&[1, 2], &mut cache)
                        .unwrap()
                        .last_logits()
                        .to_vec();
                    let mut expected = Vec::new();
                    for _ in 0..max_new {
                        let next = tritium_nn::sample_greedy(&logits).unwrap();
                        if next == 3 {
                            break;
                        }
                        expected.push(next);
                        logits = oracle
                            .runner()
                            .forward(&[next], &mut cache)
                            .unwrap()
                            .last_logits()
                            .to_vec();
                    }
                    assert_eq!(
                        loaded.runner.generate(&[1, 2], max_new, 3).unwrap(),
                        expected
                    );
                    assert_eq!(loaded.runner.cache_len(), cache.len());
                }
                assert!(loaded.runner.generate(&[], 4, 3).unwrap().is_empty());
                assert_eq!(loaded.runner.cache_len(), 0);
                let verified = verify_jsonl_with_registry(
                    &evidence.to_jsonl().unwrap(),
                    [TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION],
                )
                .unwrap();
                assert_eq!(
                    ModelLoadReceiptView::fold(&verified).unwrap(),
                    [loaded.receipt.clone()]
                );
                assert!(budget.used_bytes() > 0);
                eprintln!(
                    "public hybrid uniform={uniform} cpu={cpu} file={file}: max_abs_error={error}"
                );
                drop(loaded);
                assert_eq!(budget.used_bytes(), 0);
            }
        }
    }
}

struct ExecutionProbe {
    work: Arc<AtomicUsize>,
    fail: Arc<AtomicBool>,
    fail_at: Option<Arc<AtomicUsize>>,
}
impl TernaryBackend for ExecutionProbe {
    fn device_id(&self) -> &str {
        "cpu-public-execution-probe"
    }
    fn capabilities(&self) -> tritium_spec::DeviceCaps {
        tritium_cpu::CpuBackend::new().capabilities()
    }
    fn tensor_caps(
        &self,
        t: TensorView<'_>,
    ) -> Result<Option<tritium_spec::TensorCaps>, BackendError> {
        tritium_cpu::CpuBackend::new().tensor_caps(t)
    }
    fn upload_tensor(&self, t: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        tritium_cpu::CpuBackend::new().upload_tensor(t)
    }
    fn matmul(&self, p: tritium_spec::TensorMatmul<'_>) -> Result<(), BackendError> {
        if self.fail.swap(false, Ordering::SeqCst)
            || self
                .fail_at
                .as_ref()
                .is_some_and(|at| at.load(Ordering::SeqCst) == self.work.load(Ordering::SeqCst) + 1)
        {
            return Err(BackendError::Backend("injected decode fault".into()));
        }
        tritium_cpu::CpuBackend::new().matmul(p)?;
        self.work.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn embed_rows(
        &self,
        t: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        tritium_cpu::CpuBackend::new().embed_rows(t, ids, out)
    }
    fn upload_weights(
        &self,
        p: &[u8],
        s: tritium_core::GemmShape,
        f: tritium_core::TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        tritium_cpu::CpuBackend::new().upload_weights(p, s, f)
    }
    fn mpgemm(&self, p: tritium_spec::MpGemm<'_>) -> Result<(), BackendError> {
        tritium_cpu::CpuBackend::new().mpgemm(p)
    }
}
#[test]
fn public_hybrid_execution_failure_and_mid_forward_cancel_preserve_prefix() {
    let bytes = text_package(true, false).bytes();
    let budget = TensorPayloadBudget::new(1_000_000);
    let work = Arc::new(AtomicUsize::new(0));
    let fail = Arc::new(AtomicBool::new(false));
    let mut evidence = recorder();
    let mut loaded = load(
        ModelSource::Trit {
            bytes: &bytes,
            level: "typed",
        },
        Box::new(ExecutionProbe {
            work: work.clone(),
            fail: fail.clone(),
            fail_at: None,
        }),
        &budget,
        &mut evidence,
    )
    .unwrap();
    let mut oracle = load(
        ModelSource::Trit {
            bytes: &bytes,
            level: "typed",
        },
        Box::new(tritium_cpu::CpuBackend::new()),
        &budget,
        &mut recorder(),
    )
    .unwrap();
    close(
        &loaded.runner.prefill(&[1, 2]).unwrap(),
        &oracle.runner.prefill(&[1, 2]).unwrap(),
    );
    fail.store(true, Ordering::SeqCst);
    assert!(
        loaded
            .runner
            .step(4)
            .unwrap_err()
            .to_string()
            .contains("injected decode fault")
    );
    assert_eq!(loaded.runner.cache_len(), 2);
    let before = work.load(Ordering::SeqCst);
    assert!(
        loaded
            .runner
            .forward_cancellable(&[4], &[2], &|| work.load(Ordering::SeqCst) > before)
            .unwrap()
            .is_none()
    );
    assert!(work.load(Ordering::SeqCst) > before);
    assert_eq!(loaded.runner.cache_len(), 2);
    close(
        &loaded.runner.step(4).unwrap(),
        &oracle.runner.step(4).unwrap(),
    );
    close(
        &loaded.runner.step(5).unwrap(),
        &oracle.runner.step(5).unwrap(),
    );
    loaded.runner.reset();
    oracle.runner.reset();
    close(
        &loaded.runner.prefill(&[6, 1]).unwrap(),
        &oracle.runner.prefill(&[6, 1]).unwrap(),
    );
    drop(loaded);
    drop(oracle);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn language_head_failure_rolls_back_every_mixer_before_exact_retry() {
    let bytes = text_package(true, false).bytes();
    let source = ModelSource::Trit {
        bytes: &bytes,
        level: "typed",
    };
    let budget = TensorPayloadBudget::new(1_000_000);
    let work = Arc::new(AtomicUsize::new(0));
    let failure = Arc::new(AtomicUsize::new(usize::MAX));
    let mut loaded = load(
        source,
        Box::new(ExecutionProbe {
            work: work.clone(),
            fail: Arc::new(AtomicBool::new(false)),
            fail_at: Some(failure.clone()),
        }),
        &budget,
        &mut recorder(),
    )
    .unwrap();
    let mut oracle = load(
        source,
        Box::new(tritium_cpu::CpuBackend::new()),
        &budget,
        &mut recorder(),
    )
    .unwrap();
    assert_eq!(
        loaded.runner.prefill(&[1, 2]).unwrap(),
        oracle.runner.prefill(&[1, 2]).unwrap()
    );
    let calls = work.load(Ordering::SeqCst);
    assert!(calls > 1);
    // Same projection schedule for each segment: the last matrix is the head,
    // after GDN has staged and the attention layer has appended its KV rows.
    failure.store(calls * 2, Ordering::SeqCst);
    assert!(
        loaded
            .runner
            .step(4)
            .unwrap_err()
            .to_string()
            .contains("injected decode fault")
    );
    assert_eq!(work.load(Ordering::SeqCst), calls * 2 - 1);
    assert_eq!(loaded.runner.cache_len(), 2);
    failure.store(usize::MAX, Ordering::SeqCst);
    for token in [4, 5] {
        assert_eq!(
            loaded.runner.step(token).unwrap(),
            oracle.runner.step(token).unwrap()
        );
    }
    loaded.runner.reset();
    oracle.runner.reset();
    assert_eq!(
        loaded.runner.prefill(&[1, 2]).unwrap(),
        oracle.runner.prefill(&[1, 2]).unwrap()
    );
    drop(loaded);
    drop(oracle);
    assert_eq!(budget.used_bytes(), 0);
}

#[test]
fn public_file_admission_faults_publish_no_model_and_release_payloads() {
    let root = Scratch::new("public-rejection");
    let path = root.0.join("model.trit");
    for case in 0..12 {
        let mut package = text_package(true, case == 0);
        match case {
            1 => package
                .manifest
                .assets
                .retain(|a| a.name != "tokenizer_config.json"),
            2 => asset(&mut package, "tokenizer.json", b"{}".to_vec()),
            3 => {
                package.manifest.levels[0]
                    .tensors
                    .retain(|t| !t.name.starts_with("mtp."));
            }
            _ => {}
        }
        std::fs::write(&path, package.bytes()).unwrap();
        let mut caps = limits();
        match case {
            4 => caps.max_config_bytes = 0,
            5 => caps.max_config_bytes = 1,
            6 => caps.max_tensor_bytes = 1,
            7 => caps.max_decoded_tensor_bytes = 1,
            _ => {}
        }
        let count = Arc::new(AtomicUsize::new(0));
        let budget = TensorPayloadBudget::new(1_000_000);
        let mut evidence = if case == 8 {
            EvidenceRecorder::new("missing-model-registration", [TENSOR_UPLOADED_REGISTRATION])
                .unwrap()
        } else if case == 9 {
            EvidenceRecorder::new("missing-upload-registration", [MODEL_LOADED_REGISTRATION])
                .unwrap()
        } else {
            recorder()
        };
        let result = load(
            ModelSource::TritFile {
                path: &path,
                level: "typed",
                receiving: receiving(),
                limits: caps,
            },
            Box::new(Probe {
                count: count.clone(),
                fail_at: (case == 10).then_some(3),
                mutate: (case == 11).then(|| path.clone()),
            }),
            &budget,
            &mut evidence,
        );
        assert!(result.is_err(), "rejection case {case}");
        assert_eq!(budget.used_bytes(), 0, "rejection case {case}");
        if case < 10 {
            assert_eq!(count.load(Ordering::SeqCst), 0, "preupload case {case}");
        } else {
            assert!(count.load(Ordering::SeqCst) > 0);
        }
        assert!(
            !evidence
                .events()
                .iter()
                .any(|e| e.schema == ModelLoaded::SCHEMA_ID),
            "case {case}"
        );
    }
}
