//! Actual cache/reducer/controller/log interfaces, no copied numerical oracle.
use tritium_evidence::{
    CALIBRATED_LEVELS_FITTED_REGISTRATION, CalibratedFitReceiptView, DirectFitReceiptView,
    EvidenceError, EvidenceEvent, EvidenceRecorder, original_source_id, verify_jsonl,
};
use tritium_format::trit_model::additive_semantic_digest;
use tritium_quantize::engine::{
    CalibrationBatch, CalibrationCache, CalibrationCacheBuilder, CalibrationCollector,
    CalibrationInput, CalibrationOutputBatch, RecordedCalibratedFit, RecordedCalibratedFitError,
    fit_calibrated_levels, fit_calibrated_levels_recorded,
};
use tritium_schema::*;

#[derive(Clone, Copy)]
enum Change {
    None,
    Source,
    Input,
    Gradient,
    Tokens,
    Tokenizer,
    Objective,
    ZeroGradient,
}

fn fixture(
    output: CalibrationOutput,
    consumer: CalibrationConsumer,
    change: Change,
) -> (Vec<f32>, CalibrationCache) {
    let tensor = QuantSourceTensor {
        name: "layer.weight".into(),
        shape: vec![2, 35],
        precision: ScalePrecision::F32,
    };
    let mut weights: Vec<f32> = (0..70)
        .map(|i| ((i * 29 % 101) as f32 - 50.) / 37.)
        .collect();
    if matches!(change, Change::Source) {
        weights[0] += 0.1;
    }
    let raw: Vec<u8> = weights
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect();
    let model = original_source_id(
        "recorded-calibrated-fixture",
        &[QuantSourceBinding {
            source: tensor.clone(),
            content: BlobId::from_bytes(*blake3::hash(&raw).as_bytes()),
        }],
        &[],
    )
    .unwrap();
    let mut input = vec![0.; 35 * 35];
    for i in 0..35 {
        input[i * 35 + i] = 1.;
    }
    if matches!(change, Change::Input) {
        input[0] = 2.;
    }
    let mut gradients: Vec<f32> = (0..35).flat_map(|_| [0.25, 2.]).collect();
    if matches!(change, Change::Gradient) {
        gradients[0] = 0.5;
    }
    if matches!(change, Change::ZeroGradient) {
        gradients.fill(0.);
    }
    let indices: Vec<usize> = (0..35).map(|i| i % 2).collect();
    let factors: Vec<f32> = indices
        .iter()
        .enumerate()
        .map(|(i, row)| gradients[i * 2 + row])
        .collect();
    let mut tokens: Vec<u32> = (0..35).collect();
    if matches!(change, Change::Tokens) {
        tokens[0] = 999;
    }
    let tokenizer: &[u8] = if matches!(change, Change::Tokenizer) {
        b"tokenizer v2"
    } else {
        b"tokenizer v1"
    };
    let objective: &[u8] = if matches!(change, Change::Objective) {
        b"objective v2"
    } else {
        b"objective v1"
    };
    let mut builder = CalibrationCacheBuilder::new(
        CalibrationCacheRequest {
            tensor,
            model,
            consumer,
            output,
            sample_start: 5,
            limits: CalibrationCacheLimits {
                max_payload_bytes: 100_000,
                max_samples: 64,
                max_windows: 4,
            },
        },
        tokenizer,
        objective,
    )
    .unwrap();
    for (start, end) in [(0, 16), (16, 35)] {
        builder
            .capture(
                &tokens[start..end],
                CalibrationBatch {
                    samples: end - start,
                    input: if consumer == CalibrationConsumer::Matmul {
                        CalibrationInput::Activations(&input[start * 35..end * 35])
                    } else {
                        CalibrationInput::GatherIdentity
                    },
                    output: match output {
                        CalibrationOutput::None => CalibrationOutputBatch::None,
                        CalibrationOutput::DenseFisher => {
                            CalibrationOutputBatch::Dense(&gradients[start * 2..end * 2])
                        }
                        _ => CalibrationOutputBatch::Indexed {
                            indices: &indices[start..end],
                            factors: &factors[start..end],
                        },
                    },
                    weights: None,
                    mask: None,
                },
            )
            .unwrap();
    }
    (weights, builder.finish().unwrap())
}
fn plan(cache: &CalibrationCache, partition: CalibrationPartition) -> CalibratedFitPlan {
    let mut plan = CalibratedFitPlan {
        direct: DirectFitPlan {
            layout: AdditiveLayout {
                rows: 2,
                cols: 35,
                tile: ADDITIVE_TILE_SIZE,
                group: 32,
                max_planes: 3,
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
                max_iterations: 8,
                ridge: 1e-8,
                em_restarts: 2,
                ridge_condition_limit: 1e6,
                softened_relay: false,
                modulated_relay: false,
            }),
            levels: [3, 1, 2]
                .map(|planes| DirectFitLevel {
                    name: format!("level-{planes}"),
                    planes,
                })
                .into(),
            limits: DirectFitLimits {
                max_weights: 70,
                max_metric_entries: 1225,
                max_output_trits: 420,
            },
        },
        cache: cache.binding().clone(),
        partition,
        damping: if cache.binding().request.output == CalibrationOutput::None {
            0.
        } else {
            0.01
        },
        max_numeric_bytes: 1,
    };
    // The existing bound admits retained reducers, not per-batch group packing.
    // Actual complete cache payload bytes conservatively cover this fixture's
    // packing scratch without copying the collector's workspace formula.
    plan.max_numeric_bytes =
        CalibrationCollector::required_numeric_bytes(&plan.calibration_plan().unwrap())
            .unwrap()
            .checked_add(cache.binding().payload_bytes)
            .unwrap();
    plan
}
fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new("calibrated-fit", [CALIBRATED_LEVELS_FITTED_REGISTRATION]).unwrap()
}
fn record(
    plan: &CalibratedFitPlan,
    weights: &[f32],
    cache: &CalibrationCache,
) -> (RecordedCalibratedFit, EvidenceRecorder) {
    let mut log = recorder();
    let fit =
        fit_calibrated_levels_recorded(plan, weights, cache, &mut log, "fit", None, 0).unwrap();
    (fit, log)
}

#[test]
fn actual_cache_factors_outputs_and_sealed_replay_match_the_numerical_consumer() {
    for (output, consumer) in [
        (CalibrationOutput::None, CalibrationConsumer::Matmul),
        (CalibrationOutput::DenseFisher, CalibrationConsumer::Matmul),
        (
            CalibrationOutput::IndexedFisher,
            CalibrationConsumer::Matmul,
        ),
        (
            CalibrationOutput::IndexedFisher,
            CalibrationConsumer::GatherIdentity,
        ),
    ] {
        let (weights, cache) = fixture(output, consumer, Change::None);
        let before = weights.clone();
        let cache_before = cache.binding().clone();
        for partition in [
            CalibrationPartition::Full,
            CalibrationPartition::Groups { columns: 16 },
        ] {
            let plan = plan(&cache, partition);
            let factors = cache
                .collect(partition, plan.damping, plan.max_numeric_bytes)
                .unwrap();
            let expected = fit_calibrated_levels(
                &plan.direct,
                &plan.cache.request.tensor,
                plan.cache.request.model,
                &weights,
                &factors,
            )
            .unwrap();
            let (actual, log) = record(&plan, &weights, &cache);
            let observation = actual.observation.observation();
            assert_eq!(observation.plan, plan);
            assert_eq!(&observation.factors, factors.factors());
            assert_eq!(actual.levels, expected);
            for (level, output) in actual.levels.iter().zip(&observation.outputs) {
                assert_eq!(output.level, level.level);
                assert_eq!(output.planes, level.tensor.plane_count());
                assert_eq!(
                    output.content,
                    additive_semantic_digest(&level.tensor).unwrap()
                );
            }
            let verified = verify_jsonl(&log.to_jsonl().unwrap()).unwrap();
            assert_eq!(
                CalibratedFitReceiptView::fold(&verified).unwrap(),
                [actual.observation]
            );
            assert!(DirectFitReceiptView::fold(&verified).unwrap().is_empty());
            let (_, replay) = record(&plan, &weights, &cache);
            assert_eq!(replay.to_jsonl().unwrap(), log.to_jsonl().unwrap());
            assert_eq!(replay.root_digest().unwrap(), log.root_digest().unwrap());
            assert!(verified.events[0].payload.get("verdict").is_none());
            assert!(verified.events[0].payload.get("weights").is_none());
        }
        assert_eq!(weights, before);
        assert_eq!(cache.binding(), &cache_before);
    }
}

#[test]
fn actual_source_corpus_gradient_tokens_recipe_and_metric_changes_alter_evidence() {
    let (weights, cache) = fixture(
        CalibrationOutput::DenseFisher,
        CalibrationConsumer::Matmul,
        Change::None,
    );
    let base_plan = plan(&cache, CalibrationPartition::Full);
    let (base, log) = record(&base_plan, &weights, &cache);
    for change in [
        Change::Source,
        Change::Input,
        Change::Gradient,
        Change::Tokens,
        Change::Tokenizer,
        Change::Objective,
    ] {
        let (weights, changed_cache) = fixture(
            CalibrationOutput::DenseFisher,
            CalibrationConsumer::Matmul,
            change,
        );
        let (_, changed) = record(
            &plan(&changed_cache, CalibrationPartition::Full),
            &weights,
            &changed_cache,
        );
        assert_ne!(changed.root_digest().unwrap(), log.root_digest().unwrap());
        assert_ne!(changed_cache.id(), cache.id());
        let mut rejected = recorder();
        assert!(matches!(
            fit_calibrated_levels_recorded(
                &base_plan,
                &weights,
                &changed_cache,
                &mut rejected,
                "fit",
                None,
                0
            ),
            Err(RecordedCalibratedFitError::CacheBinding)
        ));
        assert!(rejected.events().is_empty());
    }
    let mut changed = base_plan.clone();
    changed.damping = 0.02;
    let (_, damping) = record(&changed, &weights, &cache);
    assert_ne!(log.root_digest().unwrap(), damping.root_digest().unwrap());
    changed.partition = CalibrationPartition::Groups { columns: 16 };
    let (_, grouped) = record(&changed, &weights, &cache);
    assert_ne!(
        grouped.root_digest().unwrap(),
        damping.root_digest().unwrap()
    );
    let mut signed = weights.clone();
    signed[0] = -0.0;
    let (source, _) = record(&base_plan, &signed, &cache);
    assert_ne!(
        base.observation.observation().source_f32,
        source.observation.observation().source_f32
    );
}

#[test]
fn failures_have_no_success_event_and_no_fits() {
    let (weights, cache) = fixture(
        CalibrationOutput::DenseFisher,
        CalibrationConsumer::Matmul,
        Change::None,
    );
    let plan = plan(&cache, CalibrationPartition::Full);
    let mut unregistered = EvidenceRecorder::new("calibrated-fit", []).unwrap();
    assert!(matches!(
        fit_calibrated_levels_recorded(&plan, &[], &cache, &mut unregistered, "fit", None, 0),
        Err(RecordedCalibratedFitError::Evidence(
            EvidenceError::UnregisteredSchema(_)
        ))
    ));
    for bad in [vec![], vec![f32::NAN; 70]] {
        let mut log = recorder();
        assert!(matches!(
            fit_calibrated_levels_recorded(&plan, &bad, &cache, &mut log, "fit", None, 0),
            Err(RecordedCalibratedFitError::Admission(_))
        ));
        assert!(log.events().is_empty());
    }
    for corruption in 0..8 {
        let mut bad = plan.clone();
        match corruption {
            0 => bad.direct.limits.max_weights = 69,
            1 => bad.direct.limits.max_metric_entries = 1224,
            2 => bad.direct.limits.max_output_trits = 419,
            3 => bad.max_numeric_bytes = 1,
            4 => bad.partition = CalibrationPartition::Groups { columns: 0 },
            5 => bad.damping = f64::NAN,
            6 => bad.cache.request.tensor.name = "wrong.weight".into(),
            _ => bad.direct.layout.rows = 1,
        }
        let mut log = recorder();
        assert!(
            fit_calibrated_levels_recorded(&bad, &weights, &cache, &mut log, "fit", None, 0)
                .is_err()
        );
        assert!(log.events().is_empty());
    }
    let (_, mut log) = record(&plan, &weights, &cache);
    let before = log.to_jsonl().unwrap();
    for (span, parent, time) in [
        ("", None, 1),
        ("fit", Some("different-parent"), 1),
        ("fit", None, 1 << 53),
    ] {
        assert!(
            fit_calibrated_levels_recorded(&plan, &weights, &cache, &mut log, span, parent, time)
                .is_err()
        );
        assert_eq!(before, log.to_jsonl().unwrap());
    }
}

#[test]
fn zero_fisher_keeps_the_existing_explicit_damping_contract() {
    let (weights, cache) = fixture(
        CalibrationOutput::DenseFisher,
        CalibrationConsumer::Matmul,
        Change::ZeroGradient,
    );
    let mut plan = plan(&cache, CalibrationPartition::Full);
    plan.damping = 0.;
    let mut log = recorder();
    assert!(matches!(
        fit_calibrated_levels_recorded(&plan, &weights, &cache, &mut log, "fit", None, 0),
        Err(RecordedCalibratedFitError::Fit(_))
    ));
    assert!(log.events().is_empty());
    plan.damping = 0.01;
    record(&plan, &weights, &cache);
}

#[test]
fn typed_fold_rejects_cross_field_swaps_even_when_generic_chains_are_valid() {
    let (weights, cache) = fixture(
        CalibrationOutput::DenseFisher,
        CalibrationConsumer::Matmul,
        Change::None,
    );
    let (fit, log) = record(
        &plan(&cache, CalibrationPartition::Groups { columns: 16 }),
        &weights,
        &cache,
    );
    let mut wrong_root = verify_jsonl(&log.to_jsonl().unwrap()).unwrap();
    wrong_root.root_digest = "00".repeat(32);
    assert!(matches!(
        CalibratedFitReceiptView::fold(&wrong_root),
        Err(EvidenceError::RootMismatch)
    ));
    for corruption in 0..17 {
        let mut observation = fit.observation.observation().clone();
        match corruption {
            0 => observation.factors.plan.source.cache.0[0] ^= 1,
            1 => observation.factors.plan.source.tokens.0[0] ^= 1,
            2 => observation.factors.plan.tensor.name = "swapped".into(),
            3 => observation.factors.plan.damping += 1.,
            4 => observation.factors.plan.partition = CalibrationPartition::Full,
            5 => {
                observation.factors.inputs.pop();
            }
            6 => observation.factors.inputs[2].dimension += 1,
            7 => observation.factors.inputs[1].selection.0[0] ^= 1,
            8 => observation.factors.inputs[1].samples -= 1,
            9 => observation.factors.output.as_mut().unwrap().dimension += 1,
            10 => observation.factors.output.as_mut().unwrap().total_weight += 1.,
            11 => observation.factors.output = None,
            12 => {
                observation.outputs.pop();
            }
            13 => observation.outputs[0].planes = 1,
            14 => observation.outputs[0].level = "unrequested".into(),
            15 => observation.rayon_threads = 0,
            _ => observation.plan.cache.request.model = OriginalSourceId::from_bytes([42; 32]),
        }
        let mut malformed = recorder();
        assert!(
            malformed
                .emit_calibrated_levels_fitted("fit", None, 0, &observation)
                .is_err()
        );
        assert!(malformed.events().is_empty());
        malformed.emit("fit", None, 0, &observation).unwrap();
        let verified = verify_jsonl(&malformed.to_jsonl().unwrap()).unwrap();
        assert!(matches!(
            CalibratedFitReceiptView::fold(&verified),
            Err(EvidenceError::InvalidPayload { .. })
        ));
    }
}

#[test]
fn strict_payload_version_and_actual_rayon_pool_are_observed() {
    let (weights, cache) = fixture(
        CalibrationOutput::None,
        CalibrationConsumer::Matmul,
        Change::None,
    );
    let plan = plan(&cache, CalibrationPartition::Full);
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| record(&plan, &weights, &cache))
    };
    let (one, log) = run(1);
    let (two, _) = run(2);
    assert_eq!(one.observation.observation().rayon_threads, 1);
    assert_eq!(two.observation.observation().rayon_threads, 2);
    assert_eq!(
        one.observation.observation().outputs,
        two.observation.observation().outputs
    );
    #[derive(serde::Serialize)]
    struct Payload(serde_json::Value);
    impl EvidenceEvent for Payload {
        const SCHEMA_ID: &'static str = "tritium.quantize.calibrated_levels_fitted";
        const VERSION: u16 = 1;
    }
    let mut payload = serde_json::to_value(one.observation.observation()).unwrap();
    payload["invented_release_approval"] = true.into();
    let mut malformed = recorder();
    malformed.emit("fit", None, 0, &Payload(payload)).unwrap();
    assert!(
        CalibratedFitReceiptView::fold(&verify_jsonl(&malformed.to_jsonl().unwrap()).unwrap())
            .is_err()
    );
    #[derive(serde::Serialize)]
    struct V2<'a>(&'a CalibratedLevelsFitted);
    impl EvidenceEvent for V2<'_> {
        const SCHEMA_ID: &'static str = "tritium.quantize.calibrated_levels_fitted";
        const VERSION: u16 = 2;
    }
    let mut unsupported = EvidenceRecorder::new(
        "calibrated-fit",
        [tritium_evidence::EventRegistration {
            schema: V2::SCHEMA_ID,
            version: 2,
        }],
    )
    .unwrap();
    unsupported
        .emit("fit", None, 0, &V2(one.observation.observation()))
        .unwrap();
    assert!(matches!(
        CalibratedFitReceiptView::fold(&verify_jsonl(&unsupported.to_jsonl().unwrap()).unwrap()),
        Err(EvidenceError::UnregisteredSchema(_))
    ));
    assert_eq!(
        CalibratedFitReceiptView::fold(&verify_jsonl(&log.to_jsonl().unwrap()).unwrap()).unwrap(),
        [one.observation]
    );
}
