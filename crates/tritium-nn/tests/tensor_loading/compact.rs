//! Candidate complete compact model execution, not pretrained-model qualification.

use super::*;
use tritium_format::AdditiveTensor;
use tritium_schema::{AdmittedLaw, BlobId, PlaneCounts, TensorStorage};

#[derive(Clone, Copy)]
struct Profile {
    codec: PlaneCodec,
    basis: Basis,
    law: AdmittedLaw,
    group: u16,
    pruned: bool,
    width: usize,
    ff: usize,
}

fn fixture(arch: &str, tied: bool, profile: Profile) -> TritFixture {
    let hf = HfFixture::with_dimensions(
        arch,
        tied,
        if arch == "qwen3" { 6 } else { 4 },
        profile.width,
        profile.ff,
        None,
    );
    let mut fixture = TritFixture::new(&hf, None);
    for meta in &mut fixture.manifest.levels[0].tensors {
        let TensorStorage::Dense { shape, .. } = &meta.storage else {
            continue;
        };
        if shape.len() != 2 {
            continue;
        }
        let layout = AdditiveLayout {
            rows: shape[0],
            cols: shape[1],
            tile: 256,
            group: profile.group,
            max_planes: profile.law.max_planes,
            allocation: PlaneAllocation::PerTile,
            codec: profile.codec,
            law: profile.law.law,
            basis: profile.basis,
            transport: Transport::Raw,
        };
        let counts: Vec<_> = [1_u8, 3, 2, 0, 3, 0]
            .into_iter()
            .cycle()
            .take(shape[0] as usize * (shape[1] as usize).div_ceil(256))
            .map(|n| {
                if profile.pruned {
                    0
                } else {
                    n.min(profile.law.max_planes)
                }
            })
            .collect();
        let plan = tritium_core::PerTilePlan::new(layout, &counts).unwrap();
        let trits = [Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS]
            .into_iter()
            .cycle()
            .take(plan.trit_count())
            .collect();
        let scales = [0.0625, 0.125, 0.25]
            .into_iter()
            .cycle()
            .take(plan.scale_count())
            .collect();
        let tensor = AdditiveTensor::new_per_tile(layout, counts.clone(), trits, scales).unwrap();
        let bytes = tensor.encode_payload().unwrap();
        meta.storage = TensorStorage::Additive {
            layout,
            planes: PlaneCounts::PerTile(counts),
        };
        meta.blob.id = BlobId::from_bytes(*blake3::hash(&bytes).as_bytes());
        fixture
            .blobs
            .push((tritium_format::TritBlobKind::AdditiveTensor, bytes));
    }
    // Retain only manifest-referenced assets/vectors/compact blobs. No hidden
    // dense master travels inside these complete compact package fixtures.
    let referenced: std::collections::BTreeSet<_> = fixture
        .manifest
        .assets
        .iter()
        .map(|asset| *asset.blob.id.as_bytes())
        .chain(
            fixture.manifest.levels[0]
                .tensors
                .iter()
                .map(|tensor| *tensor.blob.id.as_bytes()),
        )
        .collect();
    fixture
        .blobs
        .retain(|(_, bytes)| referenced.contains(blake3::hash(bytes).as_bytes()));
    fixture
}

fn basic() -> Profile {
    Profile {
        codec: PlaneCodec::D2,
        basis: Basis::Identity,
        law: ADMITTED_LAWS[0],
        group: 32,
        pruned: false,
        width: 8,
        ff: 16,
    }
}

#[test]
fn shared_source_accepts_complete_compact_packages() {
    let fixture = fixture("qwen3", true, basic());
    let bytes = fixture.bytes();
    tritium_format::trit_model::read_trit_model(&bytes).unwrap();
    let budget = TensorPayloadBudget::new(u64::MAX);
    let mut recorder =
        EvidenceRecorder::new("compact-load", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let runner = ModelRunner::from_source_admitted(
        tritium_nn::ModelSource::Trit {
            bytes: &bytes,
            level: "standard",
        },
        Box::new(tritium_cpu::CpuBackend::new()),
        TensorUploadPolicy {
            native_only: true,
            max_payload_bytes: None,
        },
        &budget,
        TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .expect("compact candidate package must use shared matrix decoder");
    assert!(budget.used_bytes() > 0);
    drop(runner);
    assert_eq!(budget.used_bytes(), 0);
}

// Derive expectations from actual admitted caps, not wire size or a guessed
// dense compression ratio. Norm/bias vectors are host-side and not this ledger.
fn payloads(fixture: &TritFixture, backend: &dyn TernaryBackend) -> Vec<(String, u64)> {
    let bytes = fixture.bytes();
    let model = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
    model.validate_complete_payload_inventory().unwrap();
    fixture.manifest.levels[0]
        .tensors
        .iter()
        .filter_map(|meta| {
            if !matches!(meta.storage, TensorStorage::Additive { .. }) {
                return None;
            }
            let (_, bytes) = model.tensor("standard", &meta.name).unwrap();
            let tensor = tritium_format::trit_model::decode_additive_payload(meta, bytes).unwrap();
            let caps = backend
                .tensor_caps(TensorView::Additive(tensor.view()))
                .unwrap()
                .unwrap();
            assert_eq!(caps.execution, TensorExecution::Native);
            Some((meta.name.clone(), caps.payload_bytes))
        })
        .collect()
}

fn execute(fixture: &TritFixture, backend: Box<dyn TernaryBackend>) -> Vec<Vec<u32>> {
    let expected = payloads(fixture, &*backend);
    let budget = TensorPayloadBudget::new(expected.iter().map(|(_, n)| n).sum());
    let bytes = fixture.bytes();
    let mut oracle = fixture.dense_oracle();
    let mut recorder =
        EvidenceRecorder::new("compact-execute", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let mut runner = ModelRunner::from_source_admitted(
        tritium_nn::ModelSource::Trit {
            bytes: &bytes,
            level: "standard",
        },
        backend,
        TensorUploadPolicy {
            native_only: true,
            max_payload_bytes: None,
        },
        &budget,
        TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    assert_eq!(
        budget.used_bytes(),
        expected.iter().map(|(_, n)| n).sum::<u64>()
    );
    assert!(runner.weights.token_embd.as_dense().is_none());
    let mut observed = recorder
        .events()
        .iter()
        .map(|event| {
            assert_eq!(event.payload["caps"]["execution"], "Native");
            (
                event.payload["tensor"].as_str().unwrap().to_owned(),
                event.payload["caps"]["payload_bytes"].as_u64().unwrap(),
            )
        })
        .collect::<Vec<_>>();
    observed.sort();
    let mut expected = expected;
    expected.sort();
    assert_eq!(observed, expected);
    // Device handles own their data; the complete borrowed source is gone before
    // either multi-token prefill or single-token cached decode executes.
    drop(bytes);
    let mut trace = Vec::new();
    for (tokens, positions) in [
        (&[1, 3, 2][..], &[0, 1, 2][..]),
        (&[4][..], &[3][..]),
        (&[5][..], &[4][..]),
    ] {
        let actual = runner.forward(tokens, positions).unwrap();
        let expected = oracle.forward(tokens, positions).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(&expected) {
            // Same unchanged numerical oracle tolerance as Uniform packages.
            assert!(
                a.is_finite() && e.is_finite() && (a - e).abs() <= 1e-5 * e.abs().max(1.),
                "{a} != {e}"
            );
        }
        trace.push(actual.iter().map(|v| v.to_bits()).collect());
    }
    verify_jsonl_with_registry(
        &recorder.to_jsonl().unwrap(),
        [TENSOR_UPLOADED_REGISTRATION],
    )
    .unwrap();
    drop(runner);
    assert_eq!(budget.used_bytes(), 0);
    trace
}

#[test]
fn compact_complete_models_execute_every_admitted_profile_without_source_lifetime() {
    let mut cases = 0;
    for &law in ADMITTED_LAWS {
        for group in [32, 128] {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                for basis in [
                    Basis::Identity,
                    Basis::Hadamard { block: 4 },
                    Basis::SignedRht {
                        block: 4,
                        seed: 17,
                        domain: 29,
                    },
                ] {
                    for pruned in [false, true] {
                        for tied in [false, true] {
                            let profile = Profile {
                                law,
                                group,
                                codec,
                                basis,
                                pruned,
                                ..basic()
                            };
                            let fixture =
                                fixture(["llama", "qwen2", "qwen3"][cases % 3], tied, profile);
                            let cpu = execute(&fixture, Box::new(tritium_cpu::CpuBackend::new()));
                            let reference = execute(
                                &fixture,
                                Box::new(tritium_runtime::ReferenceBackend::new()),
                            );
                            assert_eq!(cpu, reference, "compact profile {cases}");
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 288);
}

#[test]
fn compact_row_aligned_ragged_multitile_models_execute_packed_and_reference() {
    // S34 and rotations require four-aligned coefficients. Identity D2/B3
    // separately cover odd tails, so those restrictions are not hidden by padding.
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        for &law in ADMITTED_LAWS {
            for group in [32, 128] {
                let basis = if codec == PlaneCodec::S34 {
                    Basis::Hadamard { block: 4 }
                } else {
                    Basis::Identity
                };
                let (width, ff) = if codec == PlaneCodec::S34 {
                    (260, 516)
                } else {
                    (257, 513)
                };
                let fixture = fixture(
                    "qwen3",
                    law == ADMITTED_LAWS[0],
                    Profile {
                        codec,
                        law,
                        group,
                        basis,
                        width,
                        ff,
                        ..basic()
                    },
                );
                assert_eq!(
                    execute(&fixture, Box::new(tritium_cpu::CpuBackend::new())),
                    execute(&fixture, Box::new(tritium_runtime::ReferenceBackend::new()))
                );
            }
        }
    }
}

#[test]
fn compact_failed_assembly_releases_partial_payloads_without_success_receipts() {
    let fixture = fixture("qwen3", false, basic());
    let bytes = fixture.bytes();
    let expected = payloads(&fixture, &tritium_cpu::CpuBackend::new());
    let total: u64 = expected.iter().map(|(_, n)| n).sum();
    let smallest = expected.iter().map(|(_, n)| n).min().unwrap();
    for (limit, per_tensor, registered) in [
        (total - 1, None, true),
        (total, Some(smallest - 1), true),
        (total, None, false),
    ] {
        let budget = TensorPayloadBudget::new(limit);
        let (backend, uploads) = upload_probe();
        let mut recorder = EvidenceRecorder::new(
            "compact-reject",
            if registered {
                vec![TENSOR_UPLOADED_REGISTRATION]
            } else {
                vec![]
            },
        )
        .unwrap();
        let result = ModelRunner::from_source_admitted(
            tritium_nn::ModelSource::Trit {
                bytes: &bytes,
                level: "standard",
            },
            backend,
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: per_tensor,
            },
            &budget,
            TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        );
        assert!(result.is_err());
        assert_eq!(budget.used_bytes(), 0);
        if registered && per_tensor.is_none() {
            assert!(!recorder.events().is_empty());
            assert!(recorder.events().len() < expected.len());
            assert_eq!(
                uploads.load(std::sync::atomic::Ordering::SeqCst),
                recorder.events().len()
            );
            verify_jsonl_with_registry(
                &recorder.to_jsonl().unwrap(),
                [TENSOR_UPLOADED_REGISTRATION],
            )
            .unwrap();
        } else {
            // Both per-tensor rejection and missing evidence registration are
            // preflight failures in the production model loader, not late upload.
            assert!(recorder.events().is_empty());
            assert_eq!(uploads.load(std::sync::atomic::Ordering::SeqCst), 0);
        }
    }
}

#[test]
fn compact_rehashed_invalid_payload_and_wrong_counts_reject_before_backend_upload() {
    for corrupt_counts in [false, true] {
        let mut fixture = fixture("qwen3", true, basic());
        let meta = fixture.manifest.levels[0]
            .tensors
            .iter_mut()
            .find(|meta| matches!(meta.storage, TensorStorage::Additive { .. }))
            .unwrap();
        if corrupt_counts {
            let TensorStorage::Additive {
                planes: PlaneCounts::PerTile(counts),
                ..
            } = &mut meta.storage
            else {
                unreachable!()
            };
            counts.fill(0); // Structurally valid metadata, inconsistent physical payload.
        } else {
            let (_, payload) = fixture
                .blobs
                .iter_mut()
                .find(|(_, bytes)| blake3::hash(bytes).as_bytes() == meta.blob.id.as_bytes())
                .unwrap();
            payload[0] |= 0b11; // Invalid D2 code, deliberately rehashed into manifest.
            meta.blob.id = BlobId::from_bytes(*blake3::hash(payload).as_bytes());
        }
        let bytes = fixture.bytes();
        let (backend, uploads) = upload_probe();
        let budget = TensorPayloadBudget::new(u64::MAX);
        let mut recorder =
            EvidenceRecorder::new("compact-invalid", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
        assert!(
            ModelRunner::from_source_admitted(
                tritium_nn::ModelSource::Trit {
                    bytes: &bytes,
                    level: "standard"
                },
                backend,
                TensorUploadPolicy::default(),
                &budget,
                TensorUploadContext {
                    recorder: &mut recorder,
                    span: "load",
                    parent: None,
                    logical_time: 0
                }
            )
            .is_err()
        );
        assert_eq!(budget.used_bytes(), 0);
        assert!(recorder.events().is_empty());
        assert_eq!(uploads.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
}

#[cfg(feature = "tokenizer")]
#[test]
fn compact_text_loader_uses_real_assets_and_shared_generation() {
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        let mut fixture = fixture(
            "qwen3",
            true,
            Profile {
                codec,
                basis: Basis::SignedRht {
                    block: 4,
                    seed: 17,
                    domain: 29,
                },
                ..basic()
            },
        );
        let (tokenizer, config) = tokenizer_assets(false);
        fixture.asset("tokenizer.json", tokenizer);
        fixture.asset("tokenizer_config.json", config);
        let bytes = fixture.bytes();
        check_text_load(
            tritium_nn::ModelSource::Trit {
                bytes: &bytes,
                level: "standard",
            },
            fixture.dense_oracle(),
            Box::new(tritium_cpu::CpuBackend::new()),
        );
        check_text_load(
            tritium_nn::ModelSource::Trit {
                bytes: &bytes,
                level: "standard",
            },
            fixture.dense_oracle(),
            Box::new(tritium_runtime::ReferenceBackend::new()),
        );
    }
}

#[cfg(feature = "tokenizer")]
#[test]
fn compact_text_load_rejects_assets_registration_and_partial_budget_without_model_success() {
    use tritium_evidence::{EvidenceEvent, MODEL_LOADED_REGISTRATION, ModelLoaded};
    for bad in ["tokenizer", "vocab", "registration", "budget"] {
        let mut fixture = fixture("qwen3", false, basic());
        let (mut tokenizer, config) = tokenizer_assets(bad == "vocab");
        if bad == "tokenizer" {
            tokenizer = b"{}".to_vec();
        }
        fixture.asset("tokenizer.json", tokenizer);
        fixture.asset("tokenizer_config.json", config);
        let total = payloads(&fixture, &tritium_cpu::CpuBackend::new())
            .iter()
            .map(|(_, n)| n)
            .sum::<u64>();
        let bytes = fixture.bytes();
        let (backend, uploads) = upload_probe();
        let budget = TensorPayloadBudget::new(if bad == "budget" { total - 1 } else { total });
        let registry = if bad == "registration" {
            vec![TENSOR_UPLOADED_REGISTRATION]
        } else {
            vec![TENSOR_UPLOADED_REGISTRATION, MODEL_LOADED_REGISTRATION]
        };
        let mut recorder = EvidenceRecorder::new("compact-text-reject", registry.clone()).unwrap();
        assert!(
            tritium_nn::load(
                tritium_nn::ModelSource::Trit {
                    bytes: &bytes,
                    level: "standard"
                },
                backend,
                TensorUploadPolicy::default(),
                &budget,
                TensorUploadContext {
                    recorder: &mut recorder,
                    span: "load",
                    parent: None,
                    logical_time: 0
                }
            )
            .is_err(),
            "{bad}"
        );
        assert_eq!(budget.used_bytes(), 0, "{bad}");
        assert!(
            recorder
                .events()
                .iter()
                .all(|event| event.schema != ModelLoaded::SCHEMA_ID),
            "{bad}"
        );
        if bad == "budget" {
            assert!(!recorder.events().is_empty());
            assert_eq!(
                uploads.load(std::sync::atomic::Ordering::SeqCst),
                recorder.events().len()
            );
            verify_jsonl_with_registry(&recorder.to_jsonl().unwrap(), registry).unwrap();
        } else {
            assert!(recorder.events().is_empty(), "{bad}");
            assert_eq!(
                uploads.load(std::sync::atomic::Ordering::SeqCst),
                0,
                "{bad}"
            );
        }
    }
}
