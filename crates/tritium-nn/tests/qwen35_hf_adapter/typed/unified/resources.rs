//! Public retained loader admission, CPU toy semantics rather than physical fit.
use super::*;
use tritium_nn::{ModelLoadResources, StreamLimits};

fn stream() -> StreamLimits {
    StreamLimits {
        max_context: 4,
        max_segment_tokens: 2,
        max_host_cache_payload_bytes: 448,
    }
}
fn guarded(
    source: ModelSource<'_>,
    backend: Box<dyn TernaryBackend>,
    ledgers: &[TensorPayloadBudget; 5],
    evidence: &mut EvidenceRecorder,
) -> Result<tritium_nn::LoadedModel, tritium_nn::NnError> {
    guarded_mode(source, backend, ledgers, evidence, false)
}
fn guarded_mode(
    source: ModelSource<'_>,
    backend: Box<dyn TernaryBackend>,
    ledgers: &[TensorPayloadBudget; 5],
    evidence: &mut EvidenceRecorder,
    streamed: bool,
) -> Result<tritium_nn::LoadedModel, tritium_nn::NnError> {
    let load = if streamed {
        tritium_nn::load_streamed_with_resources
    } else {
        tritium_nn::load_with_resources
    };
    load(
        source,
        backend,
        TensorUploadPolicy::default(),
        ModelLoadResources {
            payload: &ledgers[0],
            source_workspace: &ledgers[1],
            host_workspace: &ledgers[2],
            device_workspace: &ledgers[3],
            retained_host: &ledgers[4],
        },
        stream(),
        TensorUploadContext {
            recorder: evidence,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
}
fn ledgers() -> [TensorPayloadBudget; 5] {
    std::array::from_fn(|_| TensorPayloadBudget::new(1_000_000))
}
fn empty(ledgers: &[TensorPayloadBudget; 5]) {
    assert!(ledgers.iter().all(|b| b.used_bytes() == 0));
}

#[derive(Debug)]
struct PlanningProbe {
    inner: Probe,
    calls: AtomicUsize,
    changed_after: Option<usize>,
    resident: bool,
    guards: [TensorPayloadBudget; 5],
}
impl TernaryBackend for PlanningProbe {
    fn device_id(&self) -> &str {
        self.inner.device_id()
    }
    fn capabilities(&self) -> tritium_spec::DeviceCaps {
        self.inner.capabilities()
    }
    fn resident_decoder_requested(&self) -> bool {
        self.resident
    }
    fn tensor_upload_plan(
        &self,
        metadata: tritium_spec::TensorMetadata<'_>,
    ) -> Result<Option<tritium_spec::TensorUploadPlan>, BackendError> {
        let mut plan = self.inner.tensor_upload_plan(metadata)?.unwrap();
        plan.host_workspace_bytes = 17;
        plan.device_workspace_bytes = 23;
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.changed_after.is_some_and(|n| call >= n) {
            plan.caps.payload_bytes += 1;
        }
        Ok(Some(plan))
    }
    fn tensor_caps(
        &self,
        tensor: TensorView<'_>,
    ) -> Result<Option<tritium_spec::TensorCaps>, BackendError> {
        self.inner.tensor_caps(tensor)
    }
    fn upload_tensor(&self, tensor: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        // These are logical synthetic workspace declarations, not GPU allocation.
        assert!(self.guards[0].used_bytes() > 0);
        assert!(self.guards[1].used_bytes() > 0);
        assert_eq!(self.guards[2].used_bytes(), 17);
        assert_eq!(self.guards[3].used_bytes(), 23);
        assert!(self.guards[4].used_bytes() > 448);
        self.inner.upload_tensor(tensor)
    }
    fn upload_weights(
        &self,
        packed: &[u8],
        shape: tritium_core::GemmShape,
        format: tritium_core::TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.inner.upload_weights(packed, shape, format)
    }
    fn mpgemm(&self, p: tritium_spec::MpGemm<'_>) -> Result<(), BackendError> {
        self.inner.mpgemm(p)
    }
}
fn probe(
    budgets: &[TensorPayloadBudget; 5],
    count: &Arc<AtomicUsize>,
    changed_after: Option<usize>,
    resident: bool,
) -> Box<dyn TernaryBackend> {
    Box::new(PlanningProbe {
        inner: Probe {
            count: count.clone(),
            fail_at: None,
            mutate: None,
        },
        calls: AtomicUsize::new(0),
        changed_after,
        resident,
        guards: budgets.clone(),
    })
}

#[test]
fn host_and_device_workspace_are_admitted_before_upload_and_held_through_assembly() {
    let bytes = text_package(true, false).bytes();
    let source = ModelSource::Trit {
        bytes: &bytes,
        level: "typed",
    };
    for constrained in [None, Some(2), Some(3)] {
        let mut budgets = ledgers();
        if let Some(index) = constrained {
            budgets[index] = TensorPayloadBudget::new(0);
        }
        let count = Arc::new(AtomicUsize::new(0));
        let mut evidence = recorder();
        let result = guarded(
            source,
            probe(&budgets, &count, None, false),
            &budgets,
            &mut evidence,
        );
        if constrained.is_some() {
            assert!(result.is_err());
            assert_eq!(count.load(Ordering::SeqCst), 0);
            assert!(evidence.events().is_empty());
        } else {
            let loaded = result.unwrap();
            let admission = loaded.runner.resource_admission().unwrap();
            assert_eq!(
                (
                    admission.host_workspace_bytes,
                    admission.device_workspace_bytes
                ),
                (17, 23)
            );
            assert!(budgets[1..4].iter().all(|b| b.used_bytes() == 0));
            drop(loaded);
        }
        empty(&budgets);
    }
}

#[test]
fn changed_predecode_plan_and_unknown_resident_resources_refuse_without_upload() {
    let bytes = text_package(true, false).bytes();
    let matrix_count = qwen35_language_mtp_tensor_schema(
        &Qwen35CheckpointConfig::from_hf_config(&config_json())
            .unwrap()
            .text,
    )
    .unwrap()
    .iter()
    .filter(|s| s.role == Qwen35TensorSchemaRole::Matrix)
    .count();
    for resident in [false, true] {
        let budgets = ledgers();
        let count = Arc::new(AtomicUsize::new(0));
        let mut evidence = recorder();
        let result = guarded(
            ModelSource::Trit {
                bytes: &bytes,
                level: "typed",
            },
            probe(&budgets, &count, Some(matrix_count), resident),
            &budgets,
            &mut evidence,
        );
        let error = result.err().unwrap().to_string();
        assert!(
            error.contains(if resident {
                "resident payload envelope is unknown"
            } else {
                "upload plan changed after aggregate admission"
            }),
            "{error}"
        );
        assert_eq!(count.load(Ordering::SeqCst), 0);
        assert!(evidence.events().is_empty());
        empty(&budgets);
    }
}

#[cfg(feature = "cuda")]
fn hybrid_structural_payload(runner: &tritium_nn::Runner, matrix_payload_bytes: usize) -> usize {
    use tritium_spec::*;
    let tritium_nn::ArchitectureSpec::Hybrid(config) = runner.architecture() else {
        panic!("hybrid fixture")
    };
    let c = &config.text;
    let hidden = c.hidden_size as usize;
    let recurrence = DeltaNetGeometry {
        value_heads: c.delta_net.num_value_heads as usize,
        key_head_dim: c.delta_net.key_head_dim as usize,
        value_head_dim: c.delta_net.value_head_dim as usize,
        group_size: (c.delta_net.num_value_heads / c.delta_net.num_key_heads) as usize,
    };
    let delta = DeltaMixerGeometry {
        recurrence,
        conv_kernel: c.delta_net.conv_kernel_dim as usize,
        rms_eps: c.rms_norm_eps as f32,
    };
    let convolution = delta.widths().unwrap().0 * delta.conv_kernel;
    let layers = c
        .layer_types
        .iter()
        .map(|kind| DecoderAllocationLayer {
            input_norm: hidden,
            post_norm: hidden,
            activation: DecoderActivation::SwiGlu,
            ffn_sub_norm: 0,
            ffn_sub_eps: c.rms_norm_eps as f32,
            mixer: match kind {
                tritium_nn::Qwen35LayerType::FullAttention => DecoderAllocationMixer::Attention {
                    gated: true,
                    vectors: [
                        0,
                        0,
                        0,
                        0,
                        c.full_attention.head_dim as usize,
                        c.full_attention.head_dim as usize,
                    ],
                },
                tritium_nn::Qwen35LayerType::DeltaNet => DecoderAllocationMixer::DeltaNet {
                    geometry: delta,
                    weights: DeltaMixerWeightLengths {
                        convolution,
                        norm: recurrence.value_head_dim,
                        dt_bias: recurrence.value_heads,
                        a_log: recurrence.value_heads,
                    },
                },
            },
        })
        .collect::<Vec<_>>();
    let bounds = stream();
    tritium_cuda::CudaBackend::plan_decoder_metadata(
        DecoderAllocationMetadata {
            geometry: DecoderGeometry {
                hidden,
                intermediate: c.intermediate_size as usize,
                query_heads: c.full_attention.num_heads as usize,
                kv_heads: c.full_attention.num_key_value_heads as usize,
                head_dim: c.full_attention.head_dim as usize,
                vocabulary: c.vocab_size as usize,
                rms_eps: c.rms_norm_eps as f32,
                rope_theta: c.rope.theta as f32,
            },
            kind: DecoderAllocationKind::Hybrid,
            semantics: DecoderSemantics {
                zero_centered_norm: true,
                rotary_dim: c.rope.rotary_dim as usize,
                qwen_rope_f32: true,
            },
            final_norm: hidden,
            layers: &layers,
            matrix_payload_bytes,
            retained_matrix_shapes: &[],
            rotation_workspace_width: 0,
        },
        DecoderLimits {
            max_context: bounds.max_context,
            max_segment_tokens: bounds.max_segment_tokens,
        },
    )
    .unwrap()
    .payload_bytes()
}

#[cfg(feature = "cuda")]
#[test]
fn physical_streamed_resident_envelopes_cover_the_complete_runner_lifetime() {
    use tritium_spec::DecoderExecution;
    const LIMIT: usize = 1_000_000;
    let mut cases = 0;
    for uniform in [false, true] {
        let bytes = text_package(uniform, false).bytes();
        let source = ModelSource::Trit {
            bytes: &bytes,
            level: "typed",
        };
        let budgets = ledgers();
        let mut evidence = recorder();
        let mut loaded = guarded_mode(
            source,
            Box::new(
                tritium_cuda::CudaBackend::new(0)
                    .expect("requires physical CUDA")
                    .with_resident_decoder(LIMIT),
            ),
            &budgets,
            &mut evidence,
            true,
        )
        .unwrap();
        let admission = loaded.runner.resource_admission().unwrap();
        assert_eq!(admission.resident_device_payload_limit, Some(LIMIT as u64));
        assert_eq!(admission.reserved_payload_bytes(), LIMIT as u64);
        assert_eq!(budgets[0].used_bytes(), LIMIT as u64);
        assert!(budgets[1..4].iter().all(|b| b.used_bytes() == 0));
        assert_eq!(
            loaded.runner.execution(),
            DecoderExecution::ResidentDeviceF32
        );
        let actual = loaded.runner.resident_payload_bytes().unwrap();
        assert_eq!(
            admission
                .resident_device_payload_plan
                .unwrap()
                .payload_bytes(),
            actual
        );
        assert_eq!(
            admission
                .resident_device_payload_plan
                .unwrap()
                .matrix_payload_bytes(),
            admission.payload_bytes as usize
        );
        assert_eq!(
            actual,
            hybrid_structural_payload(
                &loaded.runner,
                usize::try_from(admission.payload_bytes).unwrap()
            )
        );
        assert!(actual > admission.payload_bytes as usize && actual <= LIMIT);
        let uploaded = evidence
            .events()
            .iter()
            .filter(|e| e.schema == tritium_evidence::TensorUploaded::SCHEMA_ID)
            .map(|e| e.payload["caps"]["payload_bytes"].as_u64().unwrap())
            .sum::<u64>();
        assert_eq!(uploaded, admission.payload_bytes);
        let mut cpu = guarded_mode(
            source,
            Box::new(tritium_cpu::CpuBackend::new()),
            &ledgers(),
            &mut recorder(),
            true,
        )
        .unwrap();
        assert_eq!(
            cpu.runner
                .resource_admission()
                .unwrap()
                .resident_device_payload_limit,
            None
        );
        for tokens in [&[1, 2][..], &[3][..]] {
            let at = loaded.runner.cache_len();
            let positions = (at..at + tokens.len()).collect::<Vec<_>>();
            let got = loaded.runner.forward(tokens, &positions).unwrap();
            let expected = cpu.runner.forward(tokens, &positions).unwrap();
            for (a, b) in got.iter().zip(&expected) {
                assert!((a - b).abs() <= 2e-4 + 2e-4 * b.abs(), "{a} != {b}");
            }
        }
        loaded.runner.reset();
        cpu.runner.reset();
        assert!(
            loaded
                .runner
                .prefill(&[1, 2])
                .unwrap()
                .iter()
                .zip(cpu.runner.prefill(&[1, 2]).unwrap())
                .all(|(a, b)| (a - b).abs() <= 2e-4 + 2e-4 * b.abs())
        );
        // A second model cannot borrow this reserved envelope while the first runs.
        let mut refused = recorder();
        assert!(
            guarded_mode(
                source,
                Box::new(
                    tritium_cuda::CudaBackend::new(0)
                        .unwrap()
                        .with_resident_decoder(LIMIT)
                ),
                &budgets,
                &mut refused,
                true
            )
            .is_err()
        );
        assert!(refused.events().is_empty());
        assert_eq!(budgets[0].used_bytes(), LIMIT as u64);
        let runner = loaded.runner;
        assert_eq!(budgets[0].used_bytes(), LIMIT as u64);
        drop(runner);
        empty(&budgets);
        // Parent one byte short and a matrix-only ceiling both reject before upload.
        for cap in [LIMIT, admission.payload_bytes as usize, 1, 0] {
            let mut rejected_budgets = ledgers();
            if cap == LIMIT {
                rejected_budgets[0] = TensorPayloadBudget::new(LIMIT as u64 - 1);
            }
            let mut rejected = recorder();
            let result = guarded_mode(
                source,
                Box::new(
                    tritium_cuda::CudaBackend::new(0)
                        .unwrap()
                        .with_resident_decoder(cap),
                ),
                &rejected_budgets,
                &mut rejected,
                true,
            );
            assert!(result.is_err());
            assert!(rejected.events().is_empty());
            assert!(
                !rejected
                    .events()
                    .iter()
                    .any(|e| e.schema == ModelLoaded::SCHEMA_ID)
            );
            empty(&rejected_budgets);
        }
        cases += 1;
    }
    eprintln!("PHYSICAL_STREAMED_RESIDENT_ENVELOPES={cases}");
}

#[test]
fn aggregate_payload_source_and_retained_host_refuse_before_upload_and_rollback() {
    let root = Scratch::new("resource-refusals");
    let bytes = text_package(true, false).bytes();
    let path = root.0.join("model.trit");
    std::fs::write(&path, &bytes).unwrap();
    for source in [
        ModelSource::Trit {
            bytes: &bytes,
            level: "typed",
        },
        file_source(&path),
    ] {
        for constrained in [0, 1, 4] {
            let mut budgets = ledgers();
            budgets[constrained] = TensorPayloadBudget::new(0);
            let count = Arc::new(AtomicUsize::new(0));
            let mut evidence = recorder();
            let result = guarded(
                source,
                Box::new(Probe {
                    count: count.clone(),
                    fail_at: None,
                    mutate: None,
                }),
                &budgets,
                &mut evidence,
            );
            assert!(result.is_err(), "domain {constrained}");
            assert_eq!(count.load(Ordering::SeqCst), 0);
            assert!(evidence.events().is_empty());
            empty(&budgets);
        }
    }
}

#[test]
fn retained_admission_matches_uploads_roles_and_oracle_and_follows_runner_ownership() {
    let root = Scratch::new("resource-success");
    for uniform in [false, true] {
        let bytes = text_package(uniform, false).bytes();
        let path = root.0.join("model.trit");
        std::fs::write(&path, &bytes).unwrap();
        let plan =
            tritium_nn::inspect_trit_file(&path, "typed", receiving(), limits(), Some(stream()))
                .unwrap();
        let config = Qwen35CheckpointConfig::from_hf_config(&config_json()).unwrap();
        let schema = qwen35_language_mtp_tensor_schema(&config.text).unwrap();
        let preserved = schema
            .iter()
            .filter(|s| s.role == Qwen35TensorSchemaRole::Preserved)
            .map(|s| s.shape.iter().product::<usize>() as u64 * 4)
            .sum::<u64>();
        for source in [
            ModelSource::Trit {
                bytes: &bytes,
                level: "typed",
            },
            file_source(&path),
        ] {
            for streamed in [false, true] {
                let budgets = ledgers();
                let mut evidence = recorder();
                let mut loaded = guarded_mode(
                    source,
                    Box::new(tritium_cpu::CpuBackend::new()),
                    &budgets,
                    &mut evidence,
                    streamed,
                )
                .unwrap();
                let admission = loaded.runner.resource_admission().unwrap();
                assert_eq!(
                    admission.source_workspace_bytes,
                    if streamed {
                        65536
                    } else {
                        plan.maximum_tensor_working_payload_bytes()
                    }
                );
                assert_eq!(admission.retained_host_bytes, preserved + 448);
                let actual = evidence
                    .events()
                    .iter()
                    .filter(|e| e.schema == tritium_evidence::TensorUploaded::SCHEMA_ID)
                    .map(|e| e.payload["caps"]["payload_bytes"].as_u64().unwrap())
                    .sum::<u64>();
                assert_eq!(admission.payload_bytes, actual);
                assert_eq!(budgets[0].used_bytes(), actual);
                assert_eq!(budgets[4].used_bytes(), preserved + 448);
                assert!(budgets[1..4].iter().all(|b| b.used_bytes() == 0));
                let oracle_budget = TensorPayloadBudget::new(1_000_000);
                let mut oracle = load(
                    source,
                    Box::new(tritium_cpu::CpuBackend::new()),
                    &oracle_budget,
                    &mut recorder(),
                )
                .unwrap();
                assert!(oracle.runner.resource_admission().is_none());
                assert_eq!(
                    loaded.runner.prefill(&[1, 2]).unwrap(),
                    oracle.runner.prefill(&[1, 2]).unwrap()
                );
                assert_eq!(
                    loaded.runner.step(3).unwrap(),
                    oracle.runner.step(3).unwrap()
                );
                // Moving the runner out of LoadedModel must not release host custody.
                let runner = loaded.runner;
                assert_eq!(budgets[4].used_bytes(), preserved + 448);
                drop(runner);
                empty(&budgets);
            }
        }
    }
}

#[test]
fn unknown_backend_planning_is_not_zero_or_a_compatibility_fallback() {
    let bytes = text_package(false, false).bytes();
    let budgets = ledgers();
    let mut evidence = recorder();
    let error = guarded(
        ModelSource::Trit {
            bytes: &bytes,
            level: "typed",
        },
        Box::new(tritium_testkit::ReferenceBackend::new()),
        &budgets,
        &mut evidence,
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("backend upload plan is unknown"));
    assert!(evidence.events().is_empty());
    empty(&budgets);
}

#[test]
fn late_upload_or_retained_source_fault_releases_all_domains_without_model_loaded() {
    for mutate in [false, true] {
        let root = Scratch::new("resource-rollback");
        let path = root.0.join("model.trit");
        std::fs::write(&path, text_package(true, false).bytes()).unwrap();
        let budgets = ledgers();
        let count = Arc::new(AtomicUsize::new(0));
        let mut evidence = recorder();
        let result = guarded(
            file_source(&path),
            Box::new(Probe {
                count: count.clone(),
                fail_at: (!mutate).then_some(3),
                mutate: mutate.then_some(path.clone()),
            }),
            &budgets,
            &mut evidence,
        );
        assert!(result.is_err());
        assert!(count.load(Ordering::SeqCst) > 0);
        assert!(
            !evidence
                .events()
                .iter()
                .any(|e| e.schema == ModelLoaded::SCHEMA_ID)
        );
        empty(&budgets);
    }
}

#[test]
fn shared_ledger_charges_simultaneous_domains_without_double_charging_uploads() {
    let bytes = text_package(true, false).bytes();
    let generous = ledgers();
    let mut first = guarded(
        ModelSource::Trit {
            bytes: &bytes,
            level: "typed",
        },
        Box::new(tritium_cpu::CpuBackend::new()),
        &generous,
        &mut recorder(),
    )
    .unwrap();
    let admitted = first.runner.resource_admission().unwrap();
    let peak =
        admitted.payload_bytes + admitted.source_workspace_bytes + admitted.retained_host_bytes;
    let total = TensorPayloadBudget::new(peak);
    let shared = std::array::from_fn(|_| total.clone());
    let mut second = guarded(
        ModelSource::Trit {
            bytes: &bytes,
            level: "typed",
        },
        Box::new(tritium_cpu::CpuBackend::new()),
        &shared,
        &mut recorder(),
    )
    .unwrap();
    assert_eq!(
        total.used_bytes(),
        admitted.payload_bytes + admitted.retained_host_bytes
    );
    assert_eq!(
        first.runner.prefill(&[2]).unwrap(),
        second.runner.prefill(&[2]).unwrap()
    );
    drop(second);
    assert_eq!(total.used_bytes(), 0);
    let too_small = TensorPayloadBudget::new(peak - 1);
    let shared = std::array::from_fn(|_| too_small.clone());
    assert!(
        guarded(
            ModelSource::Trit {
                bytes: &bytes,
                level: "typed"
            },
            Box::new(tritium_cpu::CpuBackend::new()),
            &shared,
            &mut recorder()
        )
        .is_err()
    );
    assert_eq!(too_small.used_bytes(), 0);
}

#[test]
fn per_tile_pruning_coding_and_signed_basis_share_the_same_aggregate_loader() {
    for codec in [PlaneCodec::D2, PlaneCodec::B3] {
        let mut package = text_package(true, false);
        let original = package.bytes();
        let reader = tritium_format::trit_model::read_trit_model(&original).unwrap();
        for meta in &mut package.manifest.levels[0].tensors {
            let TensorStorage::Additive { layout, planes } = &mut meta.storage else {
                continue;
            };
            let (old, payload) = reader.tensor("typed", &meta.name).unwrap();
            let decoded =
                tritium_format::trit_model::decode_additive_payload(old, payload).unwrap();
            layout.allocation = PlaneAllocation::PerTile;
            layout.codec = codec;
            layout.basis = Basis::SignedRht {
                block: 2,
                seed: 7,
                domain: 9,
            };
            let rows = layout.rows as usize;
            let cols = layout.cols as usize;
            assert!(cols < layout.group as usize && cols < layout.tile as usize);
            let counts = (0..rows).map(|r| (r % 4) as u8).collect::<Vec<_>>();
            let mut trits = Vec::new();
            let mut scales = Vec::new();
            for (r, &count) in counts.iter().enumerate() {
                for p in 0..count as usize {
                    trits.extend_from_slice(
                        &decoded.trits()[(p * rows + r) * cols..(p * rows + r + 1) * cols],
                    );
                    scales.push(decoded.scales()[p * rows + r]);
                }
            }
            *planes = PlaneCounts::PerTile(counts.clone());
            let tensor =
                tritium_format::AdditiveTensor::new_per_tile(*layout, counts, trits, scales)
                    .unwrap();
            let payload = tensor.encode_payload().unwrap();
            meta.blob = BlobReference {
                id: BlobId::from_bytes(*blake3::hash(&payload).as_bytes()),
                shard: None,
            };
            if !package.blobs.iter().any(|(_, bytes)| bytes == &payload) {
                package.blobs.push((TritBlobKind::AdditiveTensor, payload));
            }
        }
        let bytes = package.bytes();
        let source = ModelSource::Trit {
            bytes: &bytes,
            level: "typed",
        };
        let budgets = ledgers();
        let mut loaded = guarded_mode(
            source,
            Box::new(tritium_cpu::CpuBackend::new()),
            &budgets,
            &mut recorder(),
            true,
        )
        .unwrap();
        let oracle_budget = TensorPayloadBudget::new(1_000_000);
        let mut oracle = load(
            source,
            Box::new(tritium_cpu::CpuBackend::new()),
            &oracle_budget,
            &mut recorder(),
        )
        .unwrap();
        assert_eq!(
            loaded.runner.prefill(&[1, 2]).unwrap(),
            oracle.runner.prefill(&[1, 2]).unwrap()
        );
        assert_eq!(
            loaded.runner.step(3).unwrap(),
            oracle.runner.step(3).unwrap()
        );
        drop(loaded);
        empty(&budgets);
    }
}

#[test]
fn streamed_large_embeddings_load_below_the_full_decoding_workspace_cap() {
    let root = Scratch::new("stream-working-cap");
    let small = text_package(true, false).bytes();
    let assets = tritium_format::trit_model::read_trit_model(&small).unwrap();
    let original_vocab = Qwen35CheckpointConfig::from_hf_config(&config_json())
        .unwrap()
        .text
        .vocab_size as usize;
    let expanded = tensors()
        .into_iter()
        .map(|t| {
            if t.shape == [original_vocab, H] {
                let mut large = tensor(t.name, &[32768, H], 1);
                for v in &mut large.values {
                    *v = v.clamp(-13. / 32., 13. / 32.);
                }
                large
            } else {
                t
            }
        })
        .collect();
    let mut package = Package::new(expanded, true);
    let mut config: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    config["text_config"]["vocab_size"] = 32768.into();
    asset(
        &mut package,
        "config.json",
        serde_json::to_vec(&config).unwrap(),
    );
    asset(
        &mut package,
        "tokenizer.json",
        assets.asset("tokenizer.json").unwrap().to_vec(),
    );
    asset(
        &mut package,
        "tokenizer_config.json",
        assets.asset("tokenizer_config.json").unwrap().to_vec(),
    );
    let path = root.0.join("model.trit");
    std::fs::write(&path, package.bytes()).unwrap();
    let source = ModelSource::TritFile {
        path: &path,
        level: "typed",
        receiving: TritPackageReadLimits {
            max_source_bytes: 4_000_000,
            ..receiving()
        },
        limits: Qwen35TritReadLimits {
            max_tensor_bytes: 2_000_000,
            max_decoded_tensor_bytes: 2_000_000,
            ..limits()
        },
    };
    let plan = tritium_nn::inspect_trit_file(
        &path,
        "typed",
        TritPackageReadLimits {
            max_source_bytes: 4_000_000,
            ..receiving()
        },
        Qwen35TritReadLimits {
            max_tensor_bytes: 2_000_000,
            max_decoded_tensor_bytes: 2_000_000,
            ..limits()
        },
        Some(stream()),
    )
    .unwrap();
    assert!(plan.maximum_tensor_working_payload_bytes() > 10 * 65536);
    let mut budgets = ledgers();
    budgets[1] = TensorPayloadBudget::new(65536);
    let mut refused = recorder();
    assert!(
        guarded(
            source,
            Box::new(tritium_cpu::CpuBackend::new()),
            &budgets,
            &mut refused
        )
        .is_err()
    );
    assert!(refused.events().is_empty());
    empty(&budgets);
    let mut actual = guarded_mode(
        source,
        Box::new(tritium_cpu::CpuBackend::new()),
        &budgets,
        &mut recorder(),
        true,
    )
    .unwrap();
    let admission = actual.runner.resource_admission().unwrap();
    assert_eq!(admission.source_workspace_bytes, 65536);
    assert_eq!(admission.host_workspace_bytes, 65536);
    assert_eq!(budgets[1].used_bytes(), 0);
    let oracle_budget = TensorPayloadBudget::new(2_000_000);
    let mut oracle = load(
        source,
        Box::new(tritium_cpu::CpuBackend::new()),
        &oracle_budget,
        &mut recorder(),
    )
    .unwrap();
    assert_eq!(
        actual.runner.prefill(&[1, 2]).unwrap(),
        oracle.runner.prefill(&[1, 2]).unwrap()
    );
    assert_eq!(
        actual.runner.step(3).unwrap(),
        oracle.runner.step(3).unwrap()
    );
    drop(actual);
    empty(&budgets);
}

#[test]
fn mandatory_streaming_support_and_all_positive_storage_domains_refuse_without_fallback() {
    let bytes = text_package(true, false).bytes();
    let source = ModelSource::Trit {
        bytes: &bytes,
        level: "typed",
    };
    for index in [0, 1, 2, 4] {
        let mut budgets = ledgers();
        budgets[index] = TensorPayloadBudget::new(0);
        let mut evidence = recorder();
        assert!(
            guarded_mode(
                source,
                Box::new(tritium_cpu::CpuBackend::new()),
                &budgets,
                &mut evidence,
                true
            )
            .is_err()
        );
        assert!(evidence.events().is_empty());
        empty(&budgets);
    }
    let budgets = ledgers();
    let count = Arc::new(AtomicUsize::new(0));
    let mut evidence = recorder();
    let error = guarded_mode(
        source,
        Box::new(Probe {
            count: count.clone(),
            fail_at: None,
            mutate: None,
        }),
        &budgets,
        &mut evidence,
        true,
    )
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("backend streaming plan is unknown")
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(evidence.events().is_empty());
    empty(&budgets);
}

#[derive(Debug)]
struct StreamFaultProbe(Probe);
impl TernaryBackend for StreamFaultProbe {
    fn device_id(&self) -> &str {
        "stream-model-fault-probe"
    }
    fn capabilities(&self) -> tritium_spec::DeviceCaps {
        self.0.capabilities()
    }
    fn tensor_stream_plan(
        &self,
        m: tritium_spec::TensorMetadata<'_>,
        e: tritium_spec::TensorStreamEncoding,
    ) -> Result<Option<tritium_spec::TensorStreamPlan>, BackendError> {
        tritium_cpu::CpuBackend::new().tensor_stream_plan(m, e)
    }
    fn upload_tensor_stream(
        &self,
        source: &dyn tritium_spec::TensorStreamSource,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.0.before_upload()?;
        tritium_cpu::CpuBackend::new().upload_tensor_stream(source)
    }
    fn upload_tensor(&self, _: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        panic!("decoded fallback forbidden")
    }
    fn upload_weights(
        &self,
        p: &[u8],
        s: tritium_core::GemmShape,
        f: tritium_core::TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.0.upload_weights(p, s, f)
    }
    fn mpgemm(&self, p: tritium_spec::MpGemm<'_>) -> Result<(), BackendError> {
        self.0.mpgemm(p)
    }
}
#[test]
fn streamed_partial_models_and_terminal_source_mutations_release_without_model_loaded() {
    for mutate in [false, true] {
        let root = Scratch::new("stream-model-rollback");
        let path = root.0.join("model.trit");
        std::fs::write(&path, text_package(true, false).bytes()).unwrap();
        let count = Arc::new(AtomicUsize::new(0));
        let budgets = ledgers();
        let mut evidence = recorder();
        let result = guarded_mode(
            file_source(&path),
            Box::new(StreamFaultProbe(Probe {
                count: count.clone(),
                fail_at: (!mutate).then_some(3),
                mutate: mutate.then_some(path.clone()),
            })),
            &budgets,
            &mut evidence,
            true,
        );
        assert!(result.is_err());
        assert!(count.load(Ordering::SeqCst) > 0);
        assert!(
            !evidence
                .events()
                .iter()
                .any(|e| e.schema == ModelLoaded::SCHEMA_ID)
        );
        empty(&budgets);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn physical_streamed_cuda_hybrid_public_loader_matches_cpu_without_resident_or_decode_fallback() {
    let root = Scratch::new("cuda-stream-model");
    for uniform in [false, true] {
        let path = root.0.join("model.trit");
        std::fs::write(&path, text_package(uniform, false).bytes()).unwrap();
        let cpu_budgets = ledgers();
        let gpu_budgets = ledgers();
        let source = file_source(&path);
        let mut cpu = guarded_mode(
            source,
            Box::new(tritium_cpu::CpuBackend::new()),
            &cpu_budgets,
            &mut recorder(),
            true,
        )
        .unwrap();
        let backend = tritium_cuda::CudaBackend::new(0)
            .expect("public streamed model test requires physical CUDA");
        let mut gpu = guarded_mode(
            source,
            Box::new(backend),
            &gpu_budgets,
            &mut recorder(),
            true,
        )
        .unwrap();
        assert_eq!(gpu.runner.execution(), tritium_spec::DecoderExecution::Host);
        assert_eq!(gpu.runner.mtp_status(), Some(Qwen35MtpStatus::Unverified));
        for token in [None, Some(3)] {
            let (a, b) = if let Some(token) = token {
                (
                    gpu.runner.step(token).unwrap(),
                    cpu.runner.step(token).unwrap(),
                )
            } else {
                (
                    gpu.runner.prefill(&[1, 2]).unwrap(),
                    cpu.runner.prefill(&[1, 2]).unwrap(),
                )
            };
            for (a, b) in a.into_iter().zip(b) {
                assert!(
                    a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-4 + 2e-4 * b.abs(),
                    "GPU={a} CPU={b}"
                );
            }
        }
        drop(cpu);
        drop(gpu);
        empty(&cpu_budgets);
        empty(&gpu_budgets);
    }
}
