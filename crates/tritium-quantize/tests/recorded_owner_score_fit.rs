//! Actual immutable mixed-use cache -> same fitter -> typed sealed-log replay.
use std::{collections::BTreeMap, sync::Arc};
use tritium_evidence::*;
use tritium_format::trit_model::additive_semantic_digest;
use tritium_quantize::engine::*;
use tritium_schema::*;

fn weights() -> Vec<f32> {
    (0..70)
        .map(|i| ((i * 17 % 43) as f32 - 21.) / 13.)
        .collect()
}
fn fixture(window: i64, objective: &[u8]) -> OwnerScoreCache {
    let tensor = QuantSourceTensor {
        name: "weight".into(),
        shape: vec![2, 35],
        precision: ScalePrecision::F32,
    };
    let raw: Vec<_> = weights().iter().flat_map(|v| v.to_le_bytes()).collect();
    let blob = BlobId::from_bytes(*blake3::hash(&raw).as_bytes());
    let asset = BlobId::from_bytes(*blake3::hash(b"{}").as_bytes());
    let source = original_source_id(
        "recorded-owner",
        &[QuantSourceBinding {
            source: tensor.clone(),
            content: blob,
        }],
        &[
            NamedAssetDigest {
                name: "config.json".into(),
                content: asset,
            },
            NamedAssetDigest {
                name: "tokenizer.json".into(),
                content: asset,
            },
        ],
    )
    .unwrap();
    let inputs: Vec<_> = (0..3 * 35)
        .map(|i| ((i + 1 + window as usize) % 7) as f32 / 9.)
        .collect();
    let adjoints = [0.2, -0.5, 0.6, 0.3, -0.1, 0.7];
    let indices = [0_i64, 1, 0];
    let gathered: Vec<_> = (0..3 * 35)
        .map(|i| ((i + 3 + window as usize) % 11) as f32 / 8.)
        .collect();
    let uses = [
        OwnerScoreUse::Matmul {
            samples: 3,
            inputs: &inputs,
            adjoints: &adjoints,
        },
        OwnerScoreUse::Gather {
            indices: &indices,
            adjoints: &gathered,
        },
    ];
    let context = window.to_le_bytes();
    OwnerScoreCache::new(
        OwnerScoreCacheInput {
            source,
            tensor,
            tokenizer: b"{}",
            objective,
            meta: OwnerScoreWindowMeta {
                context: vec![OwnerContextMeta {
                    name: "input_ids".into(),
                    dtype: OwnerContextDtype::I64,
                    shape: vec![1],
                    conjugated: false,
                    negated: false,
                }],
                invocations: vec![
                    OwnerInvocationMeta {
                        ordinal: 0,
                        paths: vec!["head".into()],
                        role: OwnerScoreRole::Matmul,
                        samples: 3,
                    },
                    OwnerInvocationMeta {
                        ordinal: 1,
                        paths: vec!["embed".into()],
                        role: OwnerScoreRole::Gather,
                        samples: 3,
                    },
                ],
            },
            context: &[&context],
            observations: Arc::new(
                OwnerScoreWindow::new(2, 35, &uses, 65536)
                    .unwrap()
                    .snapshot()
                    .unwrap(),
            ),
        },
        OwnerScoreCacheLimits {
            max_payload_bytes: 65536,
            max_context_bytes: 8192,
            max_context_tensors: 8,
            max_invocations: 8,
            max_samples: 32,
        },
    )
    .unwrap()
}
fn plan(caches: &[&OwnerScoreCache]) -> OwnerScoreFitPlan {
    OwnerScoreFitPlan {
        direct: DirectFitPlan {
            layout: AdditiveLayout {
                rows: 2,
                cols: 35,
                tile: 256,
                group: 32,
                max_planes: 3,
                allocation: PlaneAllocation::Uniform,
                codec: PlaneCodec::B3,
                law: ScaleLaw {
                    anchor: ScaleAnchor::Group,
                    precision: ScalePrecision::F32,
                    relation: PlaneRelation::Free,
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
            levels: vec![
                DirectFitLevel {
                    name: "three".into(),
                    planes: 3,
                },
                DirectFitLevel {
                    name: "one".into(),
                    planes: 1,
                },
            ],
            limits: DirectFitLimits {
                max_weights: 70,
                max_metric_entries: 1225,
                max_output_trits: 280,
            },
        },
        windows: caches
            .iter()
            .enumerate()
            .map(|(index, cache)| WeightedOwnerCache {
                binding: cache.binding().clone(),
                weight: index as f64 + 1.,
            })
            .collect(),
        damping: 0.1,
        limits: OwnerScoreMetricLimits {
            max_windows: 8,
            max_payload_bytes: 200_000,
            max_row_bytes: 1024,
            max_reduction_entries: 100_000,
        },
        max_row_bindings: 2,
    }
}
fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new("owner-fit", [OWNER_SCORE_LEVELS_FITTED_REGISTRATION]).unwrap()
}
fn record(
    plan: &OwnerScoreFitPlan,
    caches: &[&OwnerScoreCache],
) -> (RecordedOwnerScoreFit, EvidenceRecorder) {
    let mut log = recorder();
    let fit = fit_owner_score_levels_recorded(plan, &weights(), caches, &mut log, "fit", None, 0)
        .unwrap();
    (fit, log)
}

#[test]
fn actual_recorded_fit_rows_outputs_and_sealed_replay_match_common_consumer() {
    let a = fixture(0, b"scalar mean");
    let b = fixture(1, b"scalar mean");
    let refs = [&a, &b];
    let plan = plan(&refs);
    let windows: Vec<_> = plan.windows.iter().map(|w| w.window()).collect();
    let metrics = OwnerScoreMetrics::new(&refs, &windows, plan.damping, plan.limits).unwrap();
    let expected = fit_owner_score_levels(
        &plan.direct,
        metrics.tensor(),
        metrics.source(),
        &weights(),
        &metrics,
    )
    .unwrap();
    let (actual, log) = record(&plan, &refs);
    assert_eq!(actual.levels, expected);
    let observation = actual.observation.observation();
    assert_eq!(observation.plan, plan);
    assert_eq!(observation.stream, metrics.stream_id());
    for row in &observation.rows {
        assert_eq!(
            row.content.as_bytes(),
            &metrics.row_metric(row.row as usize).unwrap().digest()
        );
    }
    for (output, fitted) in observation.outputs.iter().zip(&actual.levels) {
        assert_eq!(
            output.content,
            additive_semantic_digest(&fitted.tensor).unwrap()
        );
    }
    let text = log.to_jsonl().unwrap();
    let verified = verify_jsonl(&text).unwrap();
    assert_eq!(
        OwnerScoreFitReceiptView::fold(&verified).unwrap(),
        vec![actual.observation]
    );
    assert!(DirectFitReceiptView::fold(&verified).unwrap().is_empty());
    assert!(
        CalibratedFitReceiptView::fold(&verified)
            .unwrap()
            .is_empty()
    );
    let (_, again) = record(&plan, &refs);
    assert_eq!(again.to_jsonl().unwrap(), text);
    if let Some(path) = std::env::var_os("TRITIUM_OWNER_FIT_ORACLE_DIR") {
        let root = std::path::Path::new(&path);
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(root.join("evidence.jsonl"), text).unwrap();
        std::fs::write(root.join("cache-0.trit"), a.encode_package(65536).unwrap()).unwrap();
        std::fs::write(root.join("cache-1.trit"), b.encode_package(65536).unwrap()).unwrap();
    }
}

#[test]
fn actual_cache_package_replay_records_identical_fit_observations() {
    let a = fixture(0, b"scalar mean");
    let b = fixture(1, b"scalar mean");
    let bytes = b.encode_package(65536).unwrap();
    let replay = OwnerScoreCache::decode_package(&bytes, b.binding(), b.limits(), 65536).unwrap();
    let refs = [&a, &b];
    let expected = plan(&refs);
    let (before, log) = record(&expected, &refs);
    let (after, other) = record(&expected, &[&a, &replay]);
    assert_eq!(before.levels, after.levels);
    assert_eq!(before.observation, after.observation);
    assert_eq!(log.to_jsonl().unwrap(), other.to_jsonl().unwrap());
}

#[test]
fn recorded_profiles_keep_the_same_row_metrics_and_common_fitted_outputs() {
    let a = fixture(0, b"scalar mean");
    let b = fixture(1, b"scalar mean");
    let refs = [&a, &b];
    let base = plan(&refs);
    let mut previous = None;
    let fitter = JointGroupFitOptions {
        max_iterations: 4,
        ridge: 1e-8,
        em_restarts: 1,
        ridge_condition_limit: 1e6,
        softened_relay: false,
        modulated_relay: false,
    };
    for profile in [
        base.direct.profile,
        DirectFitProfile::JointFeedbackGroups(JointFeedbackFitOptions {
            fitter,
            damp: 0.01,
            precision: WorkPrecision::F32Accum64,
            decay: 0.6,
            decay_ramp: true,
        }),
        DirectFitProfile::JointFeedbackGroups(JointFeedbackFitOptions {
            fitter,
            damp: 0.01,
            precision: WorkPrecision::F64,
            decay: 0.6,
            decay_ramp: true,
        }),
        DirectFitProfile::ActivationAwareLadder(LadderFitOptions {
            grid: 8,
            damp: 0.01,
            search_sweeps: 1,
            scale_refit: LadderScaleRefit::PostPass,
            decay: 0.6,
            decay_ramp: true,
        }),
    ] {
        let mut p = base.clone();
        p.direct.profile = profile;
        if matches!(profile, DirectFitProfile::ActivationAwareLadder(_)) {
            p.direct.layout.law.precision = ScalePrecision::F16;
            p.direct.layout.law.relation = PlaneRelation::Tied { num: 1, den: 3 };
        }
        let windows: Vec<_> = p.windows.iter().map(|w| w.window()).collect();
        let metrics = OwnerScoreMetrics::new(&refs, &windows, p.damping, p.limits).unwrap();
        let expected = fit_owner_score_levels(
            &p.direct,
            metrics.tensor(),
            metrics.source(),
            &weights(),
            &metrics,
        )
        .unwrap();
        let (actual, log) = record(&p, &refs);
        assert_eq!(actual.levels, expected);
        let rows = &actual.observation.observation().rows;
        if let Some(previous) = &previous {
            assert_eq!(rows, previous);
        } else {
            previous = Some(rows.clone());
        }
        assert_eq!(
            OwnerScoreFitReceiptView::fold(&verify_jsonl(&log.to_jsonl().unwrap()).unwrap())
                .unwrap(),
            vec![actual.observation]
        );
    }
}

#[test]
fn real_window_recipe_weight_damping_and_consumed_source_changes_alter_evidence() {
    let a = fixture(0, b"scalar mean");
    let b = fixture(1, b"scalar mean");
    let refs = [&a, &b];
    let base = plan(&refs);
    let (before, _) = record(&base, &refs);
    let changed = fixture(2, b"scalar mean");
    let replaced = [&a, &changed];
    let (after, _) = record(&plan(&replaced), &replaced);
    assert_ne!(
        before.observation.observation().stream,
        after.observation.observation().stream
    );
    let recipe0 = fixture(0, b"another loss");
    let recipe1 = fixture(1, b"another loss");
    let recipes = [&recipe0, &recipe1];
    assert_ne!(
        before.observation.observation().stream,
        record(&plan(&recipes), &recipes)
            .0
            .observation
            .observation()
            .stream
    );
    for field in 0..3 {
        let mut changed = base.clone();
        match field {
            0 => changed.windows[0].weight = 3.,
            1 => changed.damping = 0.2,
            _ => changed.windows.reverse(),
        }
        let reordered: Vec<_> = if field == 2 {
            vec![&b, &a]
        } else {
            refs.to_vec()
        };
        assert_ne!(
            before.observation.observation().stream,
            record(&changed, &reordered)
                .0
                .observation
                .observation()
                .stream
        );
    }
    // The runtime records consumed f32 values but does not independently prove
    // declared raw-source custody; that requires complete model-engine admission.
    let mut changed = weights();
    changed[0] += 0.1;
    let mut log = recorder();
    let after =
        fit_owner_score_levels_recorded(&base, &changed, &refs, &mut log, "fit", None, 0).unwrap();
    assert_ne!(
        before.observation.observation().source_f32,
        after.observation.observation().source_f32
    );
}

#[test]
fn binding_configuration_budget_numerics_and_recorder_failure_emit_no_success() {
    let a = fixture(0, b"scalar mean");
    let b = fixture(1, b"scalar mean");
    let refs = [&a, &b];
    let base = plan(&refs);
    for field in 0..10 {
        let mut p = base.clone();
        let mut w = weights();
        let mut log = recorder();
        match field {
            0 => p.windows[0].binding.samples += 1,
            1 => p.windows[0].weight = -1.,
            2 => p.damping = f64::NAN,
            3 => p.max_row_bindings = 1,
            4 => p.limits.max_reduction_entries = 1,
            5 => p.direct.limits.max_weights = 69,
            6 => w[0] = f32::INFINITY,
            7 => {
                p.windows.truncate(1);
                p.windows[0].weight = f64::MAX;
            }
            8 => log = EvidenceRecorder::new("missing-registration", []).unwrap(),
            _ => p.windows[0].binding.content.objective = BlobId::from_bytes([99; 32]),
        }
        let selected: Vec<_> = if field == 7 { vec![&a] } else { refs.to_vec() };
        assert!(
            fit_owner_score_levels_recorded(&p, &w, &selected, &mut log, "fit", None, 0).is_err(),
            "field {field}"
        );
        assert!(log.events().is_empty(), "field {field}");
    }
}

#[test]
fn typed_fold_rejects_valid_hash_chains_with_inconsistent_content_and_rows() {
    let a = fixture(0, b"scalar mean");
    let refs = [&a];
    let p = plan(&refs);
    let (fit, _) = record(&p, &refs);
    let base = fit.observation.observation();
    for field in 0..14 {
        let mut obs = base.clone();
        match field {
            0 => obs.plan.windows[0].binding.content.objective = BlobId::from_bytes([99; 32]),
            1 => obs.plan.windows[0].weight = 2.,
            2 => obs.plan.damping = 0.2,
            3 => obs.stream = OwnerMetricStreamId::from_bytes([99; 32]),
            4 => {
                obs.rows.swap(0, 1);
            }
            5 => obs.rows[0].row = 1,
            6 => {
                obs.rows.pop();
            }
            7 => obs.rows[1].content = obs.rows[0].content,
            8 => obs.rows_content = OwnerMetricRowsId::from_bytes([99; 32]),
            9 => obs.source_f32 = FitInputDigest([0; 32]),
            10 => obs.outputs[0].level = "foreign".into(),
            11 => obs.rayon_threads = 0,
            12 => obs.outputs[0].planes = 0,
            _ => obs.plan.windows[0].binding.samples += 1,
        }
        let mut typed = recorder();
        assert!(
            typed
                .emit_owner_score_levels_fitted("fit", None, 0, &obs)
                .is_err(),
            "field {field}"
        );
        assert!(typed.events().is_empty());
        let mut generic = recorder();
        generic.emit("fit", None, 0, &obs).unwrap();
        let log = verify_jsonl(&generic.to_jsonl().unwrap()).unwrap();
        assert!(
            OwnerScoreFitReceiptView::fold(&log).is_err(),
            "field {field}"
        );
    }
}

#[test]
fn unknown_fields_versions_and_mutated_seals_fail_closed() {
    let a = fixture(0, b"scalar mean");
    let refs = [&a];
    let (fit, log) = record(&plan(&refs), &refs);
    #[derive(serde::Serialize)]
    struct Unknown(serde_json::Value);
    impl EvidenceEvent for Unknown {
        const SCHEMA_ID: &'static str = "tritium.quantize.owner_score_levels_fitted";
        const VERSION: u16 = 1;
    }
    for nested in [false, true] {
        let mut payload = serde_json::to_value(fit.observation.observation()).unwrap();
        if nested {
            payload["plan"]["limits"]["release_approved"] = true.into();
        } else {
            payload["release_approved"] = true.into();
        }
        let mut forged = recorder();
        forged.emit("fit", None, 0, &Unknown(payload)).unwrap();
        assert!(
            OwnerScoreFitReceiptView::fold(&verify_jsonl(&forged.to_jsonl().unwrap()).unwrap())
                .is_err()
        );
    }
    #[derive(serde::Serialize)]
    struct V2<'a>(&'a OwnerScoreLevelsFitted);
    impl EvidenceEvent for V2<'_> {
        const SCHEMA_ID: &'static str = "tritium.quantize.owner_score_levels_fitted";
        const VERSION: u16 = 2;
    }
    let mut unsupported = EvidenceRecorder::new(
        "v2",
        [EventRegistration {
            schema: V2::SCHEMA_ID,
            version: 2,
        }],
    )
    .unwrap();
    unsupported
        .emit("fit", None, 0, &V2(fit.observation.observation()))
        .unwrap();
    assert!(
        OwnerScoreFitReceiptView::fold(&verify_jsonl(&unsupported.to_jsonl().unwrap()).unwrap())
            .is_err()
    );
    let mut altered = verify_jsonl(&log.to_jsonl().unwrap()).unwrap();
    altered.events[0].payload["rayon_threads"] = 99.into();
    assert!(OwnerScoreFitReceiptView::fold(&altered).is_err());
}

#[test]
fn actual_rayon_pool_is_recorded_without_changing_numerical_outputs() {
    let a = fixture(0, b"scalar mean");
    let refs = [&a];
    let p = plan(&refs);
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| record(&p, &refs).0)
    };
    let one = run(1);
    let two = run(2);
    assert_eq!(one.observation.observation().rayon_threads, 1);
    assert_eq!(two.observation.observation().rayon_threads, 2);
    assert_eq!(one.levels, two.levels);
    assert_eq!(
        one.observation.observation().rows,
        two.observation.observation().rows
    );
}

#[test]
fn wire_and_ancestry_admission_precedes_numerical_work_and_append() {
    let a = fixture(0, b"scalar mean");
    let refs = [&a];
    let mut p = plan(&refs);
    p.direct.limits.max_weights = (1_u64 << 53) + 1;
    let mut invalid = weights();
    invalid[0] = f32::NAN;
    let mut log = recorder();
    assert!(matches!(
        fit_owner_score_levels_recorded(&p, &invalid, &refs, &mut log, "fit", None, 0),
        Err(RecordedOwnerScoreFitError::Evidence(_))
    ));
    assert!(log.events().is_empty());
    let p = plan(&refs);
    assert!(fit_owner_score_levels_recorded(&p, &weights(), &refs, &mut log, "", None, 0).is_err());
    assert!(log.events().is_empty());
    fit_owner_score_levels_recorded(&p, &weights(), &refs, &mut log, "fit", None, 10).unwrap();
    let before = log.to_jsonl().unwrap();
    assert!(
        fit_owner_score_levels_recorded(
            &p,
            &weights(),
            &refs,
            &mut log,
            "fit",
            Some("different-parent"),
            10
        )
        .is_err()
    );
    assert_eq!(before, log.to_jsonl().unwrap());
    assert!(
        fit_owner_score_levels_recorded(&p, &weights(), &refs, &mut log, "fit", None, 9).is_err()
    );
    assert_eq!(before, log.to_jsonl().unwrap());
}

#[test]
fn existing_plain_and_calibrated_model_lifecycles_reject_owner_fit_children() {
    let a = fixture(0, b"scalar mean");
    let refs = [&a];
    let p = plan(&refs);
    let tensor = p.windows[0].binding.content.tensor.clone();
    let request = QuantPlan {
        architecture: "recorded-owner".into(),
        tensors: vec![QuantTensorPlan {
            source: tensor.clone(),
            action: QuantAction::Fit {
                plan: Box::new(p.direct.clone()),
                metric: DirectFitMetric::Identity,
            },
        }],
        assets: vec!["config.json".into(), "tokenizer.json".into()],
        levels: vec!["one".into(), "three".into()],
        limits: QuantLimits {
            max_source_bytes: 4096,
            max_asset_bytes: 8192,
            max_retained_blob_bytes: 65536,
            max_package_bytes: 100_000,
        },
    };
    let source = InMemoryWeightSource {
        tensors: BTreeMap::from([(
            "weight".into(),
            RawTensor {
                shape: tensor.shape.clone(),
                precision: tensor.precision,
                bytes: weights().iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        )]),
        assets: BTreeMap::from([
            ("config.json".into(), b"{}".to_vec()),
            ("tokenizer.json".into(), b"{}".to_vec()),
        ]),
        grams: BTreeMap::new(),
    };
    let registrations = [
        QUANTIZATION_STARTED_REGISTRATION,
        DIRECT_LEVELS_FITTED_REGISTRATION,
        QUANTIZATION_COMPLETED_REGISTRATION,
        OWNER_SCORE_LEVELS_FITTED_REGISTRATION,
        CALIBRATED_LEVELS_FITTED_REGISTRATION,
        CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
        CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
    ];
    for calibrated in [false, true] {
        let mut source = source.clone();
        let result = if calibrated {
            let mut builder = CalibrationCacheBuilder::new(
                CalibrationCacheRequest {
                    tensor: tensor.clone(),
                    model: a.binding().content.source,
                    consumer: CalibrationConsumer::Matmul,
                    output: CalibrationOutput::DenseFisher,
                    sample_start: 0,
                    limits: CalibrationCacheLimits {
                        max_payload_bytes: 65536,
                        max_samples: 8,
                        max_windows: 2,
                    },
                },
                b"{}",
                b"scalar mean",
            )
            .unwrap();
            builder
                .capture(
                    &[0, 1],
                    CalibrationBatch {
                        samples: 2,
                        input: CalibrationInput::Activations(&vec![1.; 70]),
                        output: CalibrationOutputBatch::Dense(&[1., 2., 3., 4.]),
                        weights: None,
                        mask: None,
                    },
                )
                .unwrap();
            let cache = builder.finish().unwrap();
            let mut fit = CalibratedFitPlan {
                direct: p.direct.clone(),
                cache: cache.binding().clone(),
                partition: CalibrationPartition::Full,
                damping: 0.01,
                max_numeric_bytes: 1_000_000,
            };
            fit.max_numeric_bytes =
                CalibrationCollector::required_numeric_bytes(&fit.calibration_plan().unwrap())
                    .unwrap()
                    + cache.binding().payload_bytes;
            let cal = CalibratedQuantPlan {
                architecture: request.architecture.clone(),
                original_source: a.binding().content.source,
                tensors: vec![CalibratedQuantTensorPlan {
                    source: tensor.clone(),
                    action: CalibratedQuantAction::CalibratedFit {
                        plan: Box::new(fit),
                    },
                }],
                assets: request.assets.clone(),
                levels: request.levels.clone(),
                limits: request.limits,
                max_cache_payload_bytes: 65536,
                max_cache_package_bytes: 100_000,
            };
            Engine::run_calibrated(
                &cal,
                &mut source,
                &mut InMemoryCalibrationSource::new([cache]).unwrap(),
                &mut MemoryTensorSink::default(),
                EvidenceRecorder::new("foreign-fit", registrations).unwrap(),
            )
            .unwrap()
        } else {
            Engine::run(
                &request,
                &mut source,
                &mut MemoryTensorSink::default(),
                EvidenceRecorder::new("foreign-fit", registrations).unwrap(),
            )
            .unwrap()
        };
        let valid = verify_jsonl(result.evidence.jsonl()).unwrap();
        if calibrated {
            assert_eq!(
                CalibratedQuantizationReceiptView::fold(&valid)
                    .unwrap()
                    .len(),
                1
            );
        } else {
            assert_eq!(QuantizationReceiptView::fold(&valid).unwrap().len(), 1);
        }
        let mut log = EvidenceRecorder::new("foreign-fit", registrations).unwrap();
        let mut root = String::new();
        for event in valid.events {
            if event.schema == QuantizationStarted::SCHEMA_ID {
                root = event.span.clone();
                let start: QuantizationStarted = serde_json::from_value(event.payload).unwrap();
                log.emit(&event.span, None, event.t, &start).unwrap();
            } else if event.schema == CalibratedQuantizationStarted::SCHEMA_ID {
                root = event.span.clone();
                let start: CalibratedQuantizationStarted =
                    serde_json::from_value(event.payload).unwrap();
                log.emit(&event.span, None, event.t, &start).unwrap();
            } else if event.schema == DirectLevelsFitted::SCHEMA_ID {
                let fit: DirectLevelsFitted = serde_json::from_value(event.payload).unwrap();
                log.emit(&event.span, event.parent.as_deref(), event.t, &fit)
                    .unwrap();
            } else if event.schema == CalibratedLevelsFitted::SCHEMA_ID {
                let fit: CalibratedLevelsFitted = serde_json::from_value(event.payload).unwrap();
                log.emit(&event.span, event.parent.as_deref(), event.t, &fit)
                    .unwrap();
            } else if event.schema == QuantizationCompleted::SCHEMA_ID
                || event.schema == CalibratedQuantizationCompleted::SCHEMA_ID
            {
                fit_owner_score_levels_recorded(
                    &p,
                    &weights(),
                    &refs,
                    &mut log,
                    "foreign-owner",
                    Some(&root),
                    event.t,
                )
                .unwrap();
                if calibrated {
                    let complete: CalibratedQuantizationCompleted =
                        serde_json::from_value(event.payload).unwrap();
                    log.emit(&event.span, None, event.t, &complete).unwrap();
                } else {
                    let complete: QuantizationCompleted =
                        serde_json::from_value(event.payload).unwrap();
                    log.emit(&event.span, None, event.t, &complete).unwrap();
                }
            }
        }
        let forged = verify_jsonl(&log.to_jsonl().unwrap()).unwrap();
        assert_eq!(OwnerScoreFitReceiptView::fold(&forged).unwrap().len(), 1);
        let error = if calibrated {
            CalibratedQuantizationReceiptView::fold(&forged).unwrap_err()
        } else {
            QuantizationReceiptView::fold(&forged).unwrap_err()
        };
        assert!(error.to_string().contains("separate model lifecycle"));
    }
}
