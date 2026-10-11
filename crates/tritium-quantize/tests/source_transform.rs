//! Exercise the real A->B source interface and existing B->C fitter/cache seams.
use std::{collections::BTreeMap, io::Cursor};
use tritium_evidence::{
    CALIBRATED_LEVELS_FITTED_REGISTRATION, CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
    CALIBRATED_QUANTIZATION_STARTED_REGISTRATION, CalibratedQuantizationReceiptView,
    DIRECT_LEVELS_FITTED_REGISTRATION, EvidenceRecorder, QUANTIZATION_COMPLETED_REGISTRATION,
    QUANTIZATION_STARTED_REGISTRATION, QuantizationReceiptView, verify_jsonl,
};
use tritium_format::trit_model::{decode_dense_f32, read_trit_model};
use tritium_quantize::engine::{
    CalibrationBatch, CalibrationCacheBuilder, CalibrationCollector, CalibrationInput,
    CalibrationOutputBatch, DeclaredSalienceStatistics, Engine, InMemoryCalibrationSource,
    InMemoryWeightSource, MemoryTensorSink, RawTensor, SafeTensorsWeightSource, SalienceFoldError,
    SaliencePair, SalienceSourceError, SourceTransform, WeightSource, verify_package_with_fit_log,
};
use tritium_schema::*;

fn raw(shape: &[u64], precision: ScalePrecision, values: &[f32]) -> RawTensor {
    RawTensor {
        shape: shape.into(),
        precision,
        bytes: values
            .iter()
            .flat_map(|&v| match precision {
                ScalePrecision::F32 => v.to_le_bytes().to_vec(),
                ScalePrecision::F16 => half::f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                ScalePrecision::Bf16 => half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                _ => unreachable!(),
            })
            .collect(),
    }
}
fn fixture(mixed: bool) -> InMemoryWeightSource {
    let values: Vec<_> = (0..32).map(|i| (i as f32 - 16.0) / 17.0).collect();
    let up: Vec<_> = (0..1024).map(|i| (i as f32 % 31.0 - 15.0) / 19.0).collect();
    InMemoryWeightSource {
        tensors: BTreeMap::from([
            (
                "attn_norm".into(),
                raw(&[32], ScalePrecision::F32, &[1.0; 32]),
            ),
            ("down".into(), raw(&[1, 32], ScalePrecision::F32, &values)),
            (
                "embedding".into(),
                raw(
                    &[1, 2],
                    if mixed {
                        ScalePrecision::Bf16
                    } else {
                        ScalePrecision::F32
                    },
                    &[-0.0, 1.0],
                ),
            ),
            (
                "ff_norm".into(),
                raw(
                    &[32],
                    if mixed {
                        ScalePrecision::Bf16
                    } else {
                        ScalePrecision::F32
                    },
                    &[1.0; 32],
                ),
            ),
            (
                "q".into(),
                raw(
                    &[1, 32],
                    if mixed {
                        ScalePrecision::F16
                    } else {
                        ScalePrecision::F32
                    },
                    &values,
                ),
            ),
            ("up".into(), raw(&[32, 32], ScalePrecision::F32, &up)),
        ]),
        assets: BTreeMap::from([
            ("config.json".into(), b"{ \"exact\": true }\n".to_vec()),
            ("tokenizer.json".into(), b"{\"fixture\":1}".to_vec()),
        ]),
        grams: BTreeMap::from([("q".into(), vec![1.0; 1024])]),
    }
}
fn dense_plan(source: &dyn WeightSource) -> QuantPlan {
    QuantPlan {
        architecture: "fixture".into(),
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
        levels: vec!["l1".into()],
        limits: QuantLimits {
            max_source_bytes: 20_000,
            max_asset_bytes: 1024,
            max_retained_blob_bytes: 20_000,
            max_package_bytes: 100_000,
        },
    }
}
fn plan(
    source: &mut InMemoryWeightSource,
) -> (
    SalienceTransformPlan,
    BTreeMap<String, DeclaredSalienceStatistics>,
) {
    let stats: BTreeMap<String, _> = ["attn", "ff", "down"]
        .into_iter()
        .enumerate()
        .map(|(offset, name)| {
            (
                name.into(),
                DeclaredSalienceStatistics {
                    moments: (0..32).map(|i| ((i + offset + 1) as f64).powi(2)).collect(),
                    rows: 2,
                },
            )
        })
        .collect();
    let operations = [
        ("attn", "q", "attn_norm", SalienceInverseAxis::NormChannels),
        ("ff", "up", "ff_norm", SalienceInverseAxis::NormChannels),
        ("down", "down", "up", SalienceInverseAxis::LinearOutputRows),
    ]
    .into_iter()
    .map(
        |(name, projection, inverse, inverse_axis)| SalienceTransformOperation {
            name: name.into(),
            projections: vec![projection.into()],
            inverse: inverse.into(),
            inverse_axis,
            moments: stats[name].binding().unwrap(),
            alpha_bits: 0.75f64.to_le_bytes(),
        },
    )
    .collect();
    let original_source = Engine::observe_source(&dense_plan(source), source).unwrap();
    (
        SalienceTransformPlan {
            architecture: "fixture".into(),
            original_source,
            tensors: source.inventory().unwrap(),
            assets: source.asset_names(),
            operations,
            numerics: SalienceNumerics::LegacyF64ScaleF32V1,
            statistics_kind: SalienceStatisticsKind::Declared,
            limits: SalienceTransformLimits {
                max_source_bytes: 20_000,
                max_asset_bytes: 1024,
                max_derived_bytes: 20_000,
                max_statistics_bytes: 768,
                max_pair_working_bytes: 20_000,
            },
        },
        stats,
    )
}
fn unchanged(source: &InMemoryWeightSource, before: &InMemoryWeightSource) {
    assert_eq!(source.inventory().unwrap(), before.inventory().unwrap());
    assert_eq!(source.assets, before.assets);
    assert_eq!(source.grams, before.grams);
    for (name, tensor) in &before.tensors {
        assert_eq!(source.tensors[name].bytes, tensor.bytes);
    }
}
fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "derived-fit-fixture",
        [
            DIRECT_LEVELS_FITTED_REGISTRATION,
            QUANTIZATION_STARTED_REGISTRATION,
            QUANTIZATION_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap()
}

fn direct() -> DirectFitPlan {
    DirectFitPlan {
        layout: AdditiveLayout {
            rows: 1,
            cols: 32,
            tile: 256,
            group: 32,
            max_planes: 1,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::B3,
            law: ScaleLaw {
                anchor: ScaleAnchor::Group,
                relation: PlaneRelation::Free,
                precision: ScalePrecision::F32,
            },
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
            name: "l1".into(),
            planes: 1,
        }],
        limits: DirectFitLimits {
            max_weights: 32,
            max_metric_entries: 1024,
            max_output_trits: 32,
        },
    }
}

fn calibrated(cache: &tritium_quantize::engine::CalibrationCache) -> CalibratedFitPlan {
    let mut plan = CalibratedFitPlan {
        direct: direct(),
        cache: cache.binding().clone(),
        partition: CalibrationPartition::Full,
        damping: 0.0,
        max_numeric_bytes: 1,
    };
    plan.max_numeric_bytes =
        CalibrationCollector::required_numeric_bytes(&plan.calibration_plan().unwrap()).unwrap()
            + cache.binding().payload_bytes;
    plan
}

#[test]
fn complete_mixed_precision_source_has_actual_identities_and_ordered_shared_outputs() {
    let mut source = fixture(true);
    let (plan, stats) = plan(&mut source);
    let before = source.clone();
    let mut derived = SourceTransform::run(&plan, &mut source, &stats).unwrap();
    unchanged(&source, &before);
    let observation = derived.observation().clone();
    assert_eq!(observation.original_source, plan.original_source);
    assert_ne!(observation.original_source, observation.derived_source);
    assert_eq!(
        observation.plan.statistics_kind,
        SalienceStatisticsKind::Declared
    );
    assert_eq!(
        observation.derived_source,
        Engine::observe_source(&dense_plan(&derived), &mut derived).unwrap()
    );
    assert_eq!(
        observation.original_source,
        tritium_evidence::original_source_id(
            "fixture",
            &observation.original_tensors,
            &observation.assets
        )
        .unwrap()
    );
    assert_eq!(
        observation.derived_source,
        tritium_evidence::original_source_id(
            "fixture",
            &observation.derived_tensors,
            &observation.assets
        )
        .unwrap()
    );
    assert_eq!(
        derived.tensor("embedding", 100).unwrap().as_ref(),
        before.tensors["embedding"].bytes
    );
    for (name, bytes) in &before.assets {
        assert_eq!(derived.asset(name, 1024).unwrap().as_ref(), bytes);
    }
    assert!(
        derived.input_gram("q", 1024).is_err(),
        "A's raw Gram cannot silently follow the transformed model"
    );

    // Compare with the shared production pair, not a duplicated fold formula.
    let mut expected: BTreeMap<String, Vec<f32>> = before
        .tensors
        .iter()
        .map(|(name, t)| {
            (
                name.clone(),
                decode_dense_f32(name, &t.shape, t.precision, &t.bytes).unwrap(),
            )
        })
        .collect();
    for (op, actual) in plan.operations.iter().zip(&observation.operations) {
        let mut inverse = expected.remove(&op.inverse).unwrap();
        let mut projection = expected.remove(&op.projections[0]).unwrap();
        let scales = SaliencePair {
            projections: &mut [&mut projection],
            inverse: &mut inverse,
            inverse_columns: if op.inverse_axis == SalienceInverseAxis::NormChannels {
                1
            } else {
                32
            },
        }
        .apply(&stats[&op.name].moments, 2, 0.75)
        .unwrap();
        assert_eq!(actual.operation, op.name);
        assert_eq!(
            actual.scale_bits,
            scales.iter().map(|v| v.to_le_bytes()).collect::<Vec<_>>()
        );
        expected.insert(op.inverse.clone(), inverse);
        expected.insert(op.projections[0].clone(), projection);
    }
    for tensor in derived.inventory().unwrap() {
        if tensor.name != "embedding" {
            assert_eq!(tensor.precision, ScalePrecision::F32);
            let bytes: Vec<_> = expected[&tensor.name]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            assert_eq!(
                derived.tensor(&tensor.name, 20_000).unwrap().as_ref(),
                bytes
            );
        }
    }
}

#[test]
fn alpha_zero_can_have_equal_a_b_identity_but_touched_half_precision_still_widens() {
    for mixed in [false, true] {
        let mut source = fixture(mixed);
        let (mut plan, stats) = plan(&mut source);
        for op in &mut plan.operations {
            op.alpha_bits = (-0.0f64).to_le_bytes();
        }
        let derived = SourceTransform::run(&plan, &mut source, &stats).unwrap();
        assert_eq!(
            derived.observation().original_source == derived.observation().derived_source,
            !mixed
        );
        assert_eq!(derived.observation().operations.len(), 3);
        for op in &derived.observation().operations {
            assert_eq!(op.scale_bits, vec![1.0f32.to_le_bytes(); 32]);
        }
    }
}

#[test]
fn wrong_original_inventory_content_and_asset_bindings_fail_without_mutation() {
    for case in 0..5 {
        let mut source = fixture(true);
        let (mut plan, stats) = plan(&mut source);
        match case {
            0 => plan.original_source = OriginalSourceId::from_bytes([9; 32]),
            1 => source.tensors.get_mut("q").unwrap().bytes[..2]
                .copy_from_slice(&half::f16::from_f32(2.0).to_bits().to_le_bytes()),
            2 => {
                source.assets.get_mut("config.json").unwrap().push(b' ');
            }
            3 => {
                source
                    .tensors
                    .insert("extra".into(), raw(&[1], ScalePrecision::F32, &[1.0]));
            }
            4 => {
                source.assets.remove("tokenizer.json");
            }
            _ => unreachable!(),
        }
        let before = source.clone();
        assert!(
            SourceTransform::run(&plan, &mut source, &stats).is_err(),
            "case {case}"
        );
        unchanged(&source, &before);
    }
}

#[test]
fn wrong_moment_bytes_rows_counts_and_missing_extra_statistics_are_rejected() {
    for case in 0..6 {
        let mut source = fixture(true);
        let (plan, mut stats) = plan(&mut source);
        match case {
            0 => stats.get_mut("attn").unwrap().moments[0] = 2.0,
            1 => stats.get_mut("attn").unwrap().rows += 1,
            2 => {
                stats.get_mut("attn").unwrap().moments.pop();
            }
            3 => {
                stats.remove("attn");
            }
            4 => {
                stats.insert("extra".into(), stats["attn"].clone());
            }
            5 => stats.get_mut("attn").unwrap().moments[0] = f64::NAN,
            _ => unreachable!(),
        }
        let before = source.clone();
        assert!(
            matches!(
                SourceTransform::run(&plan, &mut source, &stats),
                Err(SalienceSourceError::Statistics)
            ),
            "case {case}"
        );
        unchanged(&source, &before);
    }
}

#[test]
fn aliases_geometry_duplicate_names_and_all_logical_limits_reject() {
    for case in 0..11 {
        let mut source = fixture(true);
        let (mut plan, stats) = plan(&mut source);
        match case {
            0 => plan.operations[0].projections.push("q".into()),
            1 => plan.operations[2].projections.push("up".into()),
            2 => plan.operations[0].inverse = "absent".into(),
            3 => plan.operations[0].inverse_axis = SalienceInverseAxis::LinearOutputRows,
            4 => plan.operations[1].name = plan.operations[0].name.clone(),
            5 => plan.limits.max_source_bytes = 1,
            6 => plan.limits.max_derived_bytes = 1,
            7 => plan.limits.max_statistics_bytes = 767,
            8 => plan.limits.max_pair_working_bytes = 1,
            9 => plan.limits.max_asset_bytes = 1,
            10 => plan.operations[0].alpha_bits = f64::INFINITY.to_le_bytes(),
            _ => unreachable!(),
        }
        let before = source.clone();
        assert!(
            SourceTransform::run(&plan, &mut source, &stats).is_err(),
            "case {case}"
        );
        unchanged(&source, &before);
    }
}

#[test]
fn late_pair_overflow_discards_whole_derived_candidate_and_preserves_a() {
    let mut source = fixture(false);
    source.tensors.get_mut("down").unwrap().bytes[31 * 4..32 * 4]
        .copy_from_slice(&f32::MAX.to_le_bytes());
    let (plan, stats) = plan(&mut source);
    let before = source.clone();
    assert!(matches!(
        SourceTransform::run(&plan, &mut source, &stats),
        Err(SalienceSourceError::Fold(
            SalienceFoldError::NonFiniteOutput
        ))
    ));
    unchanged(&source, &before);
}

#[test]
fn actual_b_is_consumed_by_guarded_fitter_and_keep_dense_means_b_not_a() {
    let mut source = fixture(true);
    let (transform, stats) = plan(&mut source);
    let mut derived = SourceTransform::run(&transform, &mut source, &stats).unwrap();
    let b = derived.observation().derived_source;
    let norm_b = derived.tensor("attn_norm", 128).unwrap().into_owned();
    assert_ne!(norm_b, source.tensors["attn_norm"].bytes);
    let mut fit = dense_plan(&derived);
    fit.tensors
        .iter_mut()
        .find(|t| t.source.name == "q")
        .unwrap()
        .action = QuantAction::Fit {
        plan: Box::new(direct()),
        metric: DirectFitMetric::Identity,
    };
    let mut wrong_sink = MemoryTensorSink::default();
    assert!(
        Engine::run_with_source_guard(
            &fit,
            &mut derived,
            &mut wrong_sink,
            recorder(),
            transform.original_source
        )
        .is_err()
    );
    assert!(wrong_sink.artifact().is_none());
    let mut sink = MemoryTensorSink::default();
    let result =
        Engine::run_with_source_guard(&fit, &mut derived, &mut sink, recorder(), b).unwrap();
    assert_eq!(result.source_id, b);
    let (package, evidence) = sink.artifact().unwrap();
    let replay = verify_jsonl(evidence.jsonl()).unwrap();
    verify_package_with_fit_log(package, &replay).unwrap();
    assert_eq!(
        QuantizationReceiptView::fold(&replay).unwrap()[0].source_id(),
        b
    );
    let model = read_trit_model(package).unwrap();
    assert_eq!(model.tensor("l1", "attn_norm").unwrap().1, norm_b);
    assert!(matches!(
        model.tensor("l1", "q").unwrap().0.storage,
        TensorStorage::Additive { .. }
    ));
    // This is B->C fitting evidence, not a full transformed-source release receipt.
}

#[test]
fn actual_cache_builder_binds_b_identity_and_does_not_reuse_a_cache_identity() {
    let mut source = fixture(true);
    let (transform, stats) = plan(&mut source);
    let derived = SourceTransform::run(&transform, &mut source, &stats).unwrap();
    let build = |model| {
        let request = CalibrationCacheRequest {
            tensor: derived
                .inventory()
                .unwrap()
                .into_iter()
                .find(|t| t.name == "q")
                .unwrap(),
            model,
            consumer: CalibrationConsumer::Matmul,
            output: CalibrationOutput::None,
            sample_start: 0,
            limits: CalibrationCacheLimits {
                max_payload_bytes: 1024,
                max_samples: 2,
                max_windows: 1,
            },
        };
        let mut builder = CalibrationCacheBuilder::new(
            request,
            b"tokenizer-fixture",
            b"declared-fixture-context",
        )
        .unwrap();
        builder
            .capture(
                &[1, 2],
                CalibrationBatch {
                    samples: 2,
                    input: CalibrationInput::Activations(&[1.0; 64]),
                    output: CalibrationOutputBatch::None,
                    weights: None,
                    mask: None,
                },
            )
            .unwrap();
        builder.finish().unwrap()
    };
    let b_cache = build(derived.observation().derived_source);
    let a_cache = build(transform.original_source);
    assert_ne!(a_cache.id(), b_cache.id());
    assert_eq!(
        b_cache.binding().source.model.0,
        *derived.observation().derived_source.as_bytes()
    );
    let reduced = b_cache
        .collect(
            CalibrationPartition::Full,
            0.0,
            calibrated(&b_cache).max_numeric_bytes,
        )
        .unwrap();
    assert_eq!(reduced.factors().inputs.len(), 1);
}

#[test]
fn calibrated_engine_consumes_b_and_rejects_a_cache_or_a_source_declaration() {
    let mut source = fixture(true);
    let (transform, stats) = plan(&mut source);
    let mut derived = SourceTransform::run(&transform, &mut source, &stats).unwrap();
    let b = derived.observation().derived_source;
    let tensor = derived
        .inventory()
        .unwrap()
        .into_iter()
        .find(|t| t.name == "q")
        .unwrap();
    let build = |model| {
        let mut builder = CalibrationCacheBuilder::new(
            CalibrationCacheRequest {
                tensor: tensor.clone(),
                model,
                consumer: CalibrationConsumer::Matmul,
                output: CalibrationOutput::None,
                sample_start: 0,
                limits: CalibrationCacheLimits {
                    max_payload_bytes: 10_000,
                    max_samples: 32,
                    max_windows: 1,
                },
            },
            b"tokenizer",
            b"declared-fixture-context",
        )
        .unwrap();
        let input: Vec<_> = (0..1024).map(|i| f32::from(i / 32 == i % 32)).collect();
        builder
            .capture(
                &(0..32).collect::<Vec<u32>>(),
                CalibrationBatch {
                    samples: 32,
                    input: CalibrationInput::Activations(&input),
                    output: CalibrationOutputBatch::None,
                    weights: None,
                    mask: None,
                },
            )
            .unwrap();
        builder.finish().unwrap()
    };
    let make_plan = |cache: &tritium_quantize::engine::CalibrationCache, model| {
        let dense = dense_plan(&derived);
        CalibratedQuantPlan {
            architecture: dense.architecture,
            original_source: model,
            tensors: dense
                .tensors
                .into_iter()
                .map(|t| CalibratedQuantTensorPlan {
                    action: if t.source.name == "q" {
                        CalibratedQuantAction::CalibratedFit {
                            plan: Box::new(calibrated(cache)),
                        }
                    } else {
                        CalibratedQuantAction::KeepDense
                    },
                    source: t.source,
                })
                .collect(),
            assets: dense.assets,
            levels: dense.levels,
            limits: dense.limits,
            max_cache_payload_bytes: 10_000,
            max_cache_package_bytes: 100_000,
        }
    };
    let a_cache = build(transform.original_source);
    let wrong_cache_plan = make_plan(&a_cache, b);
    assert!(
        wrong_cache_plan.validate().is_err(),
        "A cache cannot be declared a B cache"
    );
    let wrong_source_plan = make_plan(&a_cache, transform.original_source);
    let b_cache = build(b);
    let fit = make_plan(&b_cache, b);
    let rec = || {
        EvidenceRecorder::new(
            "derived-calibrated-fit",
            [
                CALIBRATED_LEVELS_FITTED_REGISTRATION,
                CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
                CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap()
    };
    let mut a_caches = InMemoryCalibrationSource::new([a_cache]).unwrap();
    let mut rejected = MemoryTensorSink::default();
    assert!(
        Engine::run_calibrated(
            &wrong_source_plan,
            &mut derived,
            &mut a_caches,
            &mut rejected,
            rec()
        )
        .is_err()
    );
    assert!(rejected.artifact().is_none());
    let mut b_caches = InMemoryCalibrationSource::new([b_cache]).unwrap();
    let mut sink = MemoryTensorSink::default();
    let result =
        Engine::run_calibrated(&fit, &mut derived, &mut b_caches, &mut sink, rec()).unwrap();
    assert_eq!(result.source_id, b);
    let (package, sealed) = sink.artifact().unwrap();
    let log = verify_jsonl(sealed.jsonl()).unwrap();
    verify_package_with_fit_log(package, &log).unwrap();
    assert_eq!(
        CalibratedQuantizationReceiptView::fold(&log).unwrap()[0].source_id(),
        b
    );
}

#[test]
fn typed_json_round_trips_exact_bits_and_rejects_unknown_fields_and_policies() {
    let mut source = fixture(true);
    let (mut plan, stats) = plan(&mut source);
    plan.operations[0].alpha_bits = (-0.0f64).to_le_bytes();
    let value = serde_json::to_value(&plan).unwrap();
    assert_eq!(
        serde_json::from_value::<SalienceTransformPlan>(value.clone()).unwrap(),
        plan
    );
    for location in ["", "/limits", "/operations/0", "/operations/0/moments"] {
        let mut bad = value.clone();
        bad.pointer_mut(location)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("ignored_context".into(), serde_json::json!(true));
        assert!(
            serde_json::from_value::<SalienceTransformPlan>(bad).is_err(),
            "{location}"
        );
    }
    let mut bad = value.clone();
    bad["numerics"] = serde_json::json!("UnknownV2");
    assert!(serde_json::from_value::<SalienceTransformPlan>(bad).is_err());
    let mut bad = value;
    bad["statistics_kind"] = serde_json::json!("Captured");
    assert!(serde_json::from_value::<SalienceTransformPlan>(bad).is_err());
    let derived = SourceTransform::run(&plan, &mut source, &stats).unwrap();
    let bytes = serde_json::to_vec(derived.observation()).unwrap();
    assert_eq!(
        serde_json::from_slice::<SalienceTransformObservation>(&bytes).unwrap(),
        *derived.observation()
    );
}

#[test]
fn seekable_safetensors_adapter_produces_the_same_complete_derived_source() {
    let mut memory = fixture(true);
    let (plan, stats) = plan(&mut memory);
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, tensor) in &memory.tensors {
        let start = data.len();
        data.extend_from_slice(&tensor.bytes);
        let dtype = match tensor.precision {
            ScalePrecision::F16 => "F16",
            ScalePrecision::Bf16 => "BF16",
            ScalePrecision::F32 => "F32",
            _ => unreachable!(),
        };
        header.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": tensor.shape, "data_offsets": [start, data.len()]}));
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    let mut seek =
        SafeTensorsWeightSource::new([Cursor::new(bytes)], memory.assets.clone(), BTreeMap::new())
            .unwrap();
    let mut expected = SourceTransform::run(&plan, &mut memory, &stats).unwrap();
    let mut actual = SourceTransform::run(&plan, &mut seek, &stats).unwrap();
    assert_eq!(actual.observation(), expected.observation());
    for tensor in actual.inventory().unwrap() {
        assert_eq!(
            actual.tensor(&tensor.name, 20_000).unwrap(),
            expected.tensor(&tensor.name, 20_000).unwrap()
        );
    }
}
