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
    tritium_nn::load_with_resources(
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
                "resident graph resource planning is unavailable"
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
            let budgets = ledgers();
            let mut evidence = recorder();
            let mut loaded = guarded(
                source,
                Box::new(tritium_cpu::CpuBackend::new()),
                &budgets,
                &mut evidence,
            )
            .unwrap();
            let admission = loaded.runner.resource_admission().unwrap();
            assert_eq!(
                admission.source_workspace_bytes,
                plan.maximum_tensor_working_payload_bytes()
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
        let mut loaded = guarded(
            source,
            Box::new(tritium_cpu::CpuBackend::new()),
            &budgets,
            &mut recorder(),
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
