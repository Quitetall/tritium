//! Actual text -> B graph caches -> existing calibrated joined C -> runtime.
use super::*;
use tritium_nn::Tokenizer;
use tritium_nn::calibrate::StagedTextCaptureLimits;
use tritium_quantize::engine::{CalibrationCache, InMemoryCalibrationSource};
use tritium_schema::{
    CalibratedFitPlan, CalibratedQuantAction, CalibratedQuantPlan, CalibratedQuantTensorPlan,
    CalibrationPartition,
};

#[path = "text/captured.rs"]
mod captured;

fn text_limits() -> StagedTextCaptureLimits {
    StagedTextCaptureLimits {
        max_text_bytes: 1024,
        max_tokenizer_config_bytes: 16_000,
    }
}

fn text_input(hf: &HfFixture) -> TritFixture {
    let mut input = TritFixture::new(hf, None);
    let (tokenizer, config) = tokenizer_assets(false);
    input.asset("tokenizer.json", tokenizer);
    input.asset("tokenizer_config.json", config);
    input
}

#[test]
fn recorded_text_capture_cold_replays_and_checks_actual_raw_text_encoding() {
    use super::recorded::{recorder, rehashed};
    use tritium_evidence::CalibrationCaptureView;
    use tritium_nn::calibrate::replay_staged_capture;
    use tritium_schema::CalibrationCaptureInput;
    let hf = HfFixture::new("qwen3", true, 6, None);
    let fixture = produced(&text_input(&hf), false);
    let mut stage =
        StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
    let model = StagedCalibrationModel::new(&mut stage, limits()).unwrap();
    let actual = model
        .capture_next_token_text_recorded(
            &mut stage,
            &["hello partner hello", "partner hello"],
            text_limits(),
            capture_limits(),
            recorder(),
        )
        .unwrap();
    let view = CalibrationCaptureView::fold(actual.evidence.log())
        .unwrap()
        .remove(0);
    assert!(
        matches!(&view.started().plan.input,CalibrationCaptureInput::RawText {windows} if windows==&["hello partner hello","partner hello"])
    );
    assert!(view.started().plan.tokenizer_config.is_some());
    let receiving = view.started().plan.limits;
    drop(model);
    drop(stage);
    let log = tritium_evidence::verify_jsonl(actual.evidence.jsonl()).unwrap();
    let mut stage =
        StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
    let replay = replay_staged_capture(&log, &mut stage, receiving).unwrap();
    assert_eq!(
        replay.iter().map(|c| c.id()).collect::<Vec<_>>(),
        actual.caches.iter().map(|c| c.id()).collect::<Vec<_>>()
    );
    for bad in [
        tritium_schema::CalibrationCaptureLimits {
            max_text_bytes: receiving.max_text_bytes - 1,
            ..receiving
        },
        tritium_schema::CalibrationCaptureLimits {
            max_tokenizer_config_bytes: receiving.max_tokenizer_config_bytes - 1,
            ..receiving
        },
    ] {
        assert!(replay_staged_capture(&log, &mut stage, bad).is_err());
    }
    let mut changed = view.started().clone();
    let CalibrationCaptureInput::RawText { windows } = &mut changed.plan.input else {
        panic!()
    };
    windows[0] = "partner hello partner".into();
    let forged = rehashed(&changed, &serde_json::to_value(view.completed()).unwrap());
    // Structure alone cannot certify text->tokens. Actual shared runtime encode
    // reruns from this changed text and rejects its old encoded cache claim.
    assert_eq!(CalibrationCaptureView::fold(forged.log()).unwrap().len(), 1);
    assert!(replay_staged_capture(forged.log(), &mut stage, receiving).is_err());
    if let Some(path) = std::env::var_os("TRITIUM_CAPTURE_ORACLE_OUTPUT") {
        std::fs::write(path, actual.evidence.jsonl()).unwrap();
    }
}

#[test]
fn actual_text_capture_matches_shared_runtime_tokenizer_and_token_id_capture() {
    let hf = HfFixture::new("qwen3", true, 6, None);
    let fixture = produced(&text_input(&hf), false);
    let mut stage =
        StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
    let model = StagedCalibrationModel::new(&mut stage, limits()).unwrap();
    let texts = ["hello partner hello", "partner hello"];
    let caches = model
        .capture_next_token_text(&mut stage, &texts, text_limits(), capture_limits())
        .unwrap();
    let (tokenizer, config) = tokenizer_assets(false);
    let tokenizer = tritium_nn::HfJsonTokenizer::from_bytes(&tokenizer, &config).unwrap();
    let windows: Vec<_> = texts.iter().map(|s| tokenizer.encode(s).unwrap()).collect();
    assert_eq!(windows, [vec![1, 2, 1], vec![2, 1]]); // no hidden BOS/EOS
    let windows: Vec<_> = windows.iter().map(Vec::as_slice).collect();
    let token_caches = model
        .capture_next_token(&mut stage, &windows, capture_limits())
        .unwrap();
    for (actual, expected) in caches.iter().zip(token_caches) {
        assert_eq!(actual.id(), expected.id());
        assert_eq!(actual.binding(), expected.binding());
    }
    let required = texts.iter().map(|s| s.len() as u64).sum::<u64>();
    assert!(
        model
            .capture_next_token_text(
                &mut stage,
                &texts,
                StagedTextCaptureLimits {
                    max_text_bytes: required,
                    max_tokenizer_config_bytes: config.len() as u64
                },
                capture_limits()
            )
            .is_ok()
    );
    for bad in [
        StagedTextCaptureLimits {
            max_text_bytes: required - 1,
            ..text_limits()
        },
        StagedTextCaptureLimits {
            max_text_bytes: 0,
            ..text_limits()
        },
        StagedTextCaptureLimits {
            max_tokenizer_config_bytes: config.len() as u64 - 1,
            ..text_limits()
        },
        StagedTextCaptureLimits {
            max_tokenizer_config_bytes: 0,
            ..text_limits()
        },
    ] {
        assert!(
            model
                .capture_next_token_text(&mut stage, &texts, bad, capture_limits())
                .is_err()
        );
    }
    for bad in ["", "hello", "hello hello hello hello hello hello"] {
        assert!(
            model
                .capture_next_token_text(&mut stage, &[bad], text_limits(), capture_limits())
                .is_err()
        );
    }
    assert!(
        model
            .capture_next_token_text(&mut stage, &[], text_limits(), capture_limits())
            .is_err()
    );
    let foreign = produced(&TritFixture::new(&hf, None), false);
    let mut foreign_stage =
        StagedSalienceSource::open(&foreign.b, &foreign.transform, stage_limits()).unwrap();
    assert!(
        model
            .capture_next_token_text(&mut foreign_stage, &texts, text_limits(), capture_limits())
            .is_err()
    );
}

#[test]
fn text_capture_rejects_missing_bad_unused_vocab_and_window_mutation_assets() {
    let hf = HfFixture::new("llama", true, 4, None);
    for bad in [
        "missing",
        "json",
        "config",
        "vocab",
        "truncation",
        "padding",
    ] {
        let mut input = text_input(&hf);
        if bad == "missing" {
            input
                .manifest
                .assets
                .retain(|a| a.name != "tokenizer_config.json");
            input.blobs.retain(|(kind, _)| !matches!(kind, tritium_format::TritBlobKind::Asset(name) if name == "tokenizer_config.json"));
        }
        if bad == "json" {
            input.asset("tokenizer.json", b"not JSON".to_vec());
        }
        if bad == "config" {
            input.asset("tokenizer_config.json", b"{}".to_vec());
        }
        if bad == "vocab" {
            input.asset("tokenizer.json", tokenizer_assets(true).0);
        }
        if matches!(bad, "truncation" | "padding") {
            let mut tokenizer =
                tokenizers::Tokenizer::from_bytes(&tokenizer_assets(false).0).unwrap();
            if bad == "truncation" {
                tokenizer
                    .with_truncation(Some(tokenizers::TruncationParams {
                        max_length: 2,
                        ..Default::default()
                    }))
                    .unwrap();
            } else {
                tokenizer.with_padding(Some(tokenizers::PaddingParams {
                    strategy: tokenizers::PaddingStrategy::Fixed(4),
                    ..Default::default()
                }));
            }
            input.asset(
                "tokenizer.json",
                tokenizer.to_string(false).unwrap().into_bytes(),
            );
        }
        let fixture = produced(&input, false);
        let mut stage =
            StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
        let model = StagedCalibrationModel::new(&mut stage, limits()).unwrap();
        assert!(
            model
                .capture_next_token_text(
                    &mut stage,
                    &["hello partner hello"],
                    text_limits(),
                    capture_limits()
                )
                .is_err(),
            "{bad}"
        );
    }
}

fn calibrated_plan(
    stage: &StagedSalienceSource<'_>,
    caches: &[CalibrationCache],
) -> CalibratedQuantPlan {
    CalibratedQuantPlan {
        architecture: stage.manifest().observation.plan.architecture.clone(),
        original_source: stage.source_id(),
        tensors: stage
            .inventory()
            .unwrap()
            .into_iter()
            .map(|source| {
                let action = if let Some(cache) =
                    caches.iter().find(|c| c.binding().request.tensor == source)
                {
                    let [rows, cols] = source.shape.as_slice() else {
                        panic!("matrix cache required")
                    };
                    CalibratedQuantAction::CalibratedFit {
                        plan: Box::new(CalibratedFitPlan {
                            direct: DirectFitPlan {
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
                                profile: DirectFitProfile::JointIndependentGroups(
                                    JointGroupFitOptions {
                                        max_iterations: 4,
                                        ridge: 1e-8,
                                        em_restarts: 1,
                                        ridge_condition_limit: 1e6,
                                        softened_relay: false,
                                        modulated_relay: false,
                                    },
                                ),
                                levels: vec![DirectFitLevel {
                                    name: "standard".into(),
                                    planes: 3,
                                }],
                                limits: DirectFitLimits {
                                    max_weights: rows * cols,
                                    max_metric_entries: cols * cols,
                                    max_output_trits: 3 * rows * cols,
                                },
                            },
                            cache: cache.binding().clone(),
                            partition: CalibrationPartition::Full,
                            damping: 0.125,
                            max_numeric_bytes: 8_000_000,
                        }),
                    }
                } else {
                    CalibratedQuantAction::KeepDense
                };
                CalibratedQuantTensorPlan { source, action }
            })
            .collect(),
        assets: stage.asset_names(),
        levels: vec!["standard".into()],
        limits: QuantLimits {
            max_source_bytes: 64_000,
            max_asset_bytes: 16_000,
            max_retained_blob_bytes: 64_000,
            max_package_bytes: 1_000_000,
        },
        max_cache_payload_bytes: caches.iter().map(|c| c.binding().payload_bytes).sum(),
        max_cache_package_bytes: 1_000_000,
    }
}

fn calibrated_recorder() -> EvidenceRecorder {
    use tritium_evidence::{
        CALIBRATED_LEVELS_FITTED_REGISTRATION, CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
        CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
    };
    EvidenceRecorder::new(
        "actual-b-capture-fit",
        [
            CALIBRATED_LEVELS_FITTED_REGISTRATION,
            CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
            CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap()
}

#[test]
fn actual_b_caches_fit_joined_c_execute_and_reject_actual_a_cache_swaps() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        let hf = HfFixture::new("qwen3", true, 6, None);
        let mut input = text_input(&hf);
        if precision != ScalePrecision::F32 {
            input.dense_precision(precision);
        }
        let fixture = produced(&input, false);
        let mut stage =
            StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
        let model = StagedCalibrationModel::new(&mut stage, limits()).unwrap();
        let texts = ["hello partner hello", "partner hello"];
        let caches = model
            .capture_next_token_text(&mut stage, &texts, text_limits(), capture_limits())
            .unwrap();
        // Keep tied embedding/head dense until both gather and head are admitted.
        // Never pretend the head-only cache captures the tied gather consumer.
        let caches: Vec<_> = caches
            .into_iter()
            .filter(|c| c.binding().request.tensor.name != "model.embed_tokens.weight")
            .collect();
        assert_eq!(caches.len(), 7);
        let plan = calibrated_plan(&stage, &caches);
        plan.validate().unwrap();
        let mut provider = InMemoryCalibrationSource::new(caches).unwrap();
        let mut sink = MemoryTensorSink::default();
        Engine::run_calibrated_salience(
            &plan,
            &mut stage,
            &mut provider,
            &mut sink,
            calibrated_recorder(),
            conversion_limits(),
        )
        .unwrap();
        let (c, seal) = sink.artifact().unwrap();
        let fit = verify_jsonl(seal.jsonl()).unwrap();
        let joined = JoinedSalienceModel::open(c, &stage, &fit, conversion_limits()).unwrap();
        assert_eq!(joined.manifest().derived_source, model.source_id());
        assert_eq!(
            joined.manifest().original_source,
            stage.manifest().observation.original_source
        );
        assert_eq!(
            joined.model().manifest().levels[0]
                .tensors
                .iter()
                .filter(|t| matches!(t.storage, TensorStorage::Additive { .. }))
                .count(),
            7
        );
        for tensor in stage
            .inventory()
            .unwrap()
            .into_iter()
            .filter(|t| t.shape.len() == 1 || t.name == "model.embed_tokens.weight")
        {
            assert_eq!(
                joined.model().tensor("standard", &tensor.name).unwrap().1,
                stage.tensor(&tensor.name, 64_000).unwrap().as_ref()
            );
        }
        for reference in [false, true] {
            let backend: Box<dyn TernaryBackend> = if reference {
                Box::new(tritium_testkit::ReferenceBackend::new())
            } else {
                Box::new(tritium_cpu::CpuBackend::new())
            };
            let mut oracle = oracle_fixture(&joined, c).dense_oracle();
            let budget = TensorPayloadBudget::new(100_000);
            let mut recorder =
                EvidenceRecorder::new("captured-c-runtime", [TENSOR_UPLOADED_REGISTRATION])
                    .unwrap();
            let mut runner = ModelRunner::from_source_admitted(
                tritium_nn::ModelSource::JoinedSalience {
                    model: &joined,
                    level: "standard",
                },
                backend,
                TensorUploadPolicy::default(),
                &budget,
                TensorUploadContext {
                    recorder: &mut recorder,
                    span: "load",
                    parent: None,
                    logical_time: 0,
                },
            )
            .unwrap();
            for (tokens, positions) in [(&[1, 2][..], &[0, 1][..]), (&[1][..], &[2][..])] {
                let actual = runner.forward(tokens, positions).unwrap();
                let expected = oracle.forward(tokens, positions).unwrap();
                assert!(
                    actual
                        .iter()
                        .zip(expected)
                        .all(|(a, b)| (a - b).abs() < 2e-5)
                );
            }
            drop(runner);
            assert_eq!(budget.used_bytes(), 0);
        }
        // Actual alpha-zero stage, not a relabeled B cache. For F32 its complete
        // byte identity is exactly A; lowered inputs are also truly different B.
        let original = produced_with_alpha(&input, false, 0.0);
        let mut a_stage =
            StagedSalienceSource::open(&original.b, &original.transform, stage_limits()).unwrap();
        if precision == ScalePrecision::F32 {
            assert_eq!(
                a_stage.source_id(),
                a_stage.manifest().observation.original_source
            );
        }
        assert_ne!(a_stage.source_id(), stage.source_id());
        let a_model = StagedCalibrationModel::new(&mut a_stage, limits()).unwrap();
        let a_caches = a_model
            .capture_next_token_text(&mut a_stage, &texts, text_limits(), capture_limits())
            .unwrap();
        let mut a_provider = InMemoryCalibrationSource::new(
            a_caches
                .into_iter()
                .filter(|c| c.binding().request.tensor.name != "model.embed_tokens.weight"),
        )
        .unwrap();
        let mut failed = MemoryTensorSink::default();
        assert!(
            Engine::run_calibrated_salience(
                &plan,
                &mut stage,
                &mut a_provider,
                &mut failed,
                calibrated_recorder(),
                conversion_limits()
            )
            .is_err()
        );
        assert!(failed.artifact().is_none());
        let mut foreign = MemoryTensorSink::default();
        assert!(
            Engine::run_calibrated_salience(
                &plan,
                &mut a_stage,
                &mut provider,
                &mut foreign,
                calibrated_recorder(),
                conversion_limits()
            )
            .is_err()
        );
        assert!(foreign.artifact().is_none());
        assert!(JoinedSalienceModel::open(c, &a_stage, &fit, conversion_limits()).is_err());
    }
}
