//! Engine-produced joined C crosses the existing model-loading/forward seam.
//! Toy standard-transformer software evidence, not real-model qualification.
#[path = "joined/original.rs"]
mod original;
#[path = "joined/staged.rs"]
mod staged;
use super::*;
use std::collections::BTreeMap;
use tritium_evidence::{
    DIRECT_LEVELS_FITTED_REGISTRATION, QUANTIZATION_COMPLETED_REGISTRATION,
    QUANTIZATION_STARTED_REGISTRATION, SALIENCE_TRANSFORM_COMPLETED_REGISTRATION,
    SALIENCE_TRANSFORM_STARTED_REGISTRATION, VerifiedLog, verify_jsonl,
};
use tritium_quantize::engine::{
    DeclaredSalienceStatistics, Engine, InMemoryWeightSource, JoinedSalienceModel,
    MemoryTensorSink, RawTensor, SourceTransform, StagedSalienceSource, WeightSource,
};
use tritium_schema::{
    DirectFitLevel, DirectFitLimits, DirectFitMetric, DirectFitPlan, DirectFitProfile,
    JointGroupFitOptions, QuantAction, QuantLimits, QuantPlan, QuantTensorPlan,
    SalienceConversionLimits, SalienceInverseAxis, SalienceNumerics, SalienceStageLimits,
    SalienceStatisticsKind, SalienceTransformLimits, SalienceTransformOperation,
    SalienceTransformPlan, TensorStorage,
};

struct JoinedFixture {
    b: Vec<u8>,
    c: Vec<u8>,
    transform: VerifiedLog,
    fit: VerifiedLog,
}

fn stage_limits() -> SalienceStageLimits {
    SalienceStageLimits {
        max_package_bytes: 1_000_000,
        max_manifest_bytes: 256_000,
        max_tensor_bytes: 64_000,
        max_asset_bytes: 16_000,
        max_statistics_bytes: 256,
    }
}

fn conversion_limits() -> SalienceConversionLimits {
    SalienceConversionLimits {
        max_package_bytes: 1_000_000,
        max_manifest_bytes: 256_000,
    }
}

fn produced(input: &TritFixture, additive: bool) -> JoinedFixture {
    produced_with_alpha(input, additive, 0.75)
}

fn raw_fixture_source(input: &TritFixture) -> InMemoryWeightSource {
    let bytes = input.bytes();
    let model = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
    InMemoryWeightSource {
        tensors: model.manifest().levels[0]
            .tensors
            .iter()
            .map(|meta| {
                let TensorStorage::Dense { shape, precision } = &meta.storage else {
                    panic!("original fixture must be dense")
                };
                (
                    meta.name.clone(),
                    RawTensor {
                        shape: shape.clone(),
                        precision: *precision,
                        bytes: model.tensor("standard", &meta.name).unwrap().1.into(),
                    },
                )
            })
            .collect(),
        assets: model
            .manifest()
            .assets
            .iter()
            .map(|asset| (asset.name.clone(), model.asset(&asset.name).unwrap().into()))
            .collect(),
        grams: BTreeMap::new(),
    }
}

fn produced_with_alpha(input: &TritFixture, additive: bool, alpha: f64) -> JoinedFixture {
    let mut a = raw_fixture_source(input);
    let dense_plan = |source: &dyn WeightSource| QuantPlan {
        architecture: input.manifest.architecture.clone(),
        tensors: source
            .inventory()
            .unwrap()
            .into_iter()
            .map(|source| QuantTensorPlan {
                source,
                action: QuantAction::KeepDense,
            })
            .collect(),
        assets: source.asset_names(),
        levels: vec!["standard".into()],
        limits: QuantLimits {
            max_source_bytes: 64_000,
            max_asset_bytes: 16_000,
            max_retained_blob_bytes: 64_000,
            max_package_bytes: 1_000_000,
        },
    };
    let stats: BTreeMap<String, _> = [("attention", 8), ("feedforward", 8), ("down", 16)]
        .into_iter()
        .map(|(name, n)| {
            (
                name.into(),
                DeclaredSalienceStatistics {
                    moments: (1..=n).map(|i| f64::from(i).powi(2)).collect(),
                    rows: 2,
                },
            )
        })
        .collect();
    let operations = [
        (
            "attention",
            vec![
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ],
            "input_layernorm.weight",
            SalienceInverseAxis::NormChannels,
        ),
        (
            "feedforward",
            vec!["mlp.gate_proj.weight", "mlp.up_proj.weight"],
            "post_attention_layernorm.weight",
            SalienceInverseAxis::NormChannels,
        ),
        (
            "down",
            vec!["mlp.down_proj.weight"],
            "mlp.up_proj.weight",
            SalienceInverseAxis::LinearOutputRows,
        ),
    ]
    .into_iter()
    .map(
        |(name, projections, inverse, inverse_axis)| SalienceTransformOperation {
            name: name.into(),
            projections: projections
                .into_iter()
                .map(|p| format!("model.layers.0.{p}"))
                .collect(),
            inverse: format!("model.layers.0.{inverse}"),
            inverse_axis,
            moments: stats[name].binding().unwrap(),
            alpha_bits: alpha.to_le_bytes(),
        },
    )
    .collect();
    let request = SalienceTransformPlan {
        architecture: input.manifest.architecture.clone(),
        original_source: Engine::observe_source(&dense_plan(&a), &mut a).unwrap(),
        tensors: a.inventory().unwrap(),
        assets: a.asset_names(),
        operations,
        numerics: SalienceNumerics::LegacyF64ScaleF32V1,
        statistics_kind: SalienceStatisticsKind::Declared,
        limits: SalienceTransformLimits {
            max_source_bytes: 64_000,
            max_asset_bytes: 16_000,
            max_derived_bytes: 64_000,
            max_statistics_bytes: 256,
            max_pair_working_bytes: 64_000,
        },
    };
    let recorded = SourceTransform::run_recorded(
        &request,
        &mut a,
        &stats,
        EvidenceRecorder::new(
            "nn-joined-transform",
            [
                SALIENCE_TRANSFORM_STARTED_REGISTRATION,
                SALIENCE_TRANSFORM_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let b = recorded
        .encode_stage_package(&stats, stage_limits())
        .unwrap();
    let transform = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    drop(recorded);
    let mut stage = StagedSalienceSource::open(&b, &transform, stage_limits()).unwrap();
    let mut plan = dense_plan(&stage);
    if additive {
        for tensor in &mut plan.tensors {
            if let [rows, cols] = tensor.source.shape.as_slice() {
                let count = rows * cols;
                tensor.action = QuantAction::Fit {
                    plan: Box::new(DirectFitPlan {
                        layout: AdditiveLayout {
                            rows: *rows,
                            cols: *cols,
                            tile: 256,
                            group: 32,
                            max_planes: 3,
                            allocation: PlaneAllocation::Uniform,
                            codec: PlaneCodec::B3,
                            law: ADMITTED_LAWS[1].law,
                            basis: Basis::Identity,
                            transport: Transport::Raw,
                        },
                        profile: DirectFitProfile::JointIndependentGroups(JointGroupFitOptions {
                            max_iterations: 4,
                            ridge: 1e-8,
                            em_restarts: 1,
                            ridge_condition_limit: 1e6,
                            softened_relay: false,
                            modulated_relay: false,
                        }),
                        levels: vec![DirectFitLevel {
                            name: "standard".into(),
                            planes: 3,
                        }],
                        limits: DirectFitLimits {
                            max_weights: count,
                            max_metric_entries: cols * cols,
                            max_output_trits: 3 * count,
                        },
                    }),
                    metric: DirectFitMetric::Identity,
                };
            }
        }
    }
    let mut sink = MemoryTensorSink::default();
    Engine::run_salience(
        &plan,
        &mut stage,
        &mut sink,
        EvidenceRecorder::new(
            "nn-joined-fit",
            [
                QUANTIZATION_STARTED_REGISTRATION,
                DIRECT_LEVELS_FITTED_REGISTRATION,
                QUANTIZATION_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap(),
        conversion_limits(),
    )
    .unwrap();
    let (c, seal) = sink.artifact().unwrap();
    let fit = verify_jsonl(seal.jsonl()).unwrap();
    JoinedSalienceModel::open(c, &stage, &fit, conversion_limits())
        .unwrap()
        .verify_transform(&stage, &mut a)
        .unwrap();
    JoinedFixture {
        b,
        c: c.into(),
        transform,
        fit,
    }
}

fn oracle_fixture(joined: &JoinedSalienceModel<'_>, c: &[u8]) -> TritFixture {
    // Only the numerical oracle materializes a legacy HF representation of the
    // exact stored C values. Production loading uses the mandatory original C.
    let package = tritium_format::read_trit_package(c).unwrap();
    TritFixture {
        manifest: joined.model().manifest().clone(),
        blobs: package
            .blobs()
            .iter()
            .map(|entry| (entry.kind.clone(), package.blob(entry).unwrap().into()))
            .collect(),
    }
}

#[test]
fn joined_runtime_executes_after_stage_logs_and_c_bytes_are_dropped() {
    for additive in [false, true] {
        for reference in [false, true] {
            let hf = HfFixture::new("qwen3", true, 6, None);
            let JoinedFixture {
                b,
                c,
                transform,
                fit,
            } = produced(&TritFixture::new(&hf, None), additive);
            let stage = StagedSalienceSource::open(&b, &transform, stage_limits()).unwrap();
            let joined = JoinedSalienceModel::open(&c, &stage, &fit, conversion_limits()).unwrap();
            drop(stage);
            drop(b);
            drop(transform);
            drop(fit);
            let mut oracle = oracle_fixture(&joined, &c).dense_oracle();
            let budget = TensorPayloadBudget::new(100_000);
            let mut recorder =
                EvidenceRecorder::new("nn-joined-runtime", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
            let backend: Box<dyn TernaryBackend> = if reference {
                Box::new(tritium_testkit::ReferenceBackend::new())
            } else {
                Box::new(tritium_cpu::CpuBackend::new())
            };
            let mut runner = ModelRunner::from_source_admitted(
                tritium_nn::ModelSource::JoinedSalience {
                    model: &joined,
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
            assert!(budget.used_bytes() > 0);
            assert_eq!(recorder.events().len(), hf.upload_count);
            drop(joined);
            drop(c);
            for (tokens, positions) in [
                (&[1, 3, 2][..], &[0, 1, 2][..]),
                (&[4][..], &[3][..]),
                (&[5][..], &[4][..]),
            ] {
                let expected = oracle.forward(tokens, positions).unwrap();
                let actual = runner.forward(tokens, positions).unwrap();
                if additive {
                    for (a, e) in actual.iter().zip(expected) {
                        assert!(
                            a.is_finite()
                                && e.is_finite()
                                && (a - e).abs() <= 1e-5 * e.abs().max(1.),
                            "{a} != {e}"
                        );
                    }
                } else {
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
            verify_jsonl_with_registry(
                &recorder.to_jsonl().unwrap(),
                [TENSOR_UPLOADED_REGISTRATION],
            )
            .unwrap();
            drop(runner);
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[test]
fn joined_runtime_rejects_legacy_source_missing_levels_geometry_and_payload_budget() {
    for malformed_config in [false, true] {
        let hf = HfFixture::new("qwen3", true, 6, None);
        let mut input = TritFixture::new(&hf, None);
        if malformed_config {
            let mut config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(hf.dir.join("config.json")).unwrap())
                    .unwrap();
            config["hidden_size"] = 16.into();
            input.asset("config.json", serde_json::to_vec(&config).unwrap());
        }
        let JoinedFixture {
            b,
            c,
            transform,
            fit,
        } = produced(&input, true);
        let stage = StagedSalienceSource::open(&b, &transform, stage_limits()).unwrap();
        let joined = JoinedSalienceModel::open(&c, &stage, &fit, conversion_limits()).unwrap();
        for source in [
            tritium_nn::ModelSource::Trit {
                bytes: &c,
                level: "standard",
            },
            tritium_nn::ModelSource::JoinedSalience {
                model: &joined,
                level: "absent",
            },
        ]
        .into_iter()
        .chain(
            malformed_config.then_some(tritium_nn::ModelSource::JoinedSalience {
                model: &joined,
                level: "standard",
            }),
        ) {
            let (backend, uploads) = upload_probe();
            let budget = TensorPayloadBudget::new(100_000);
            let mut recorder =
                EvidenceRecorder::new("nn-joined-reject", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
            assert!(
                ModelRunner::from_source_admitted(
                    source,
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
            assert_eq!(uploads.load(std::sync::atomic::Ordering::SeqCst), 0);
            assert!(recorder.events().is_empty());
            assert_eq!(budget.used_bytes(), 0);
        }
        if !malformed_config {
            let budget = TensorPayloadBudget::new(1);
            let mut recorder =
                EvidenceRecorder::new("nn-joined-budget", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
            assert!(
                ModelRunner::from_source_admitted(
                    tritium_nn::ModelSource::JoinedSalience {
                        model: &joined,
                        level: "standard"
                    },
                    Box::new(tritium_cpu::CpuBackend::new()),
                    TensorUploadPolicy::default(),
                    &budget,
                    TensorUploadContext {
                        recorder: &mut recorder,
                        span: "load",
                        parent: None,
                        logical_time: 0
                    },
                )
                .is_err()
            );
            assert_eq!(budget.used_bytes(), 0);
        }
    }
}

#[cfg(feature = "tokenizer")]
#[test]
fn joined_runtime_uses_actual_tokenizer_and_c_load_identity() {
    for additive in [false, true] {
        let hf = HfFixture::new("qwen3", true, 6, None);
        let mut input = TritFixture::new(&hf, None);
        let (tokenizer, config) = tokenizer_assets(false);
        input.asset("tokenizer.json", tokenizer);
        input.asset("tokenizer_config.json", config);
        let JoinedFixture {
            b,
            c,
            transform,
            fit,
        } = produced(&input, additive);
        let stage = StagedSalienceSource::open(&b, &transform, stage_limits()).unwrap();
        let joined = JoinedSalienceModel::open(&c, &stage, &fit, conversion_limits()).unwrap();
        drop(stage);
        drop(b);
        drop(transform);
        drop(fit);
        let oracle = oracle_fixture(&joined, &c).dense_oracle();
        check_text_load(
            tritium_nn::ModelSource::JoinedSalience {
                model: &joined,
                level: "standard",
            },
            oracle,
            Box::new(tritium_cpu::CpuBackend::new()),
        );
    }
}
