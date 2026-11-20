//! Actual common engine/cache/sink calls, not a copied model/numerical pipeline.
use super::*;
use std::io::Write;
use tritium_evidence::{
    CALIBRATED_LEVELS_FITTED_REGISTRATION, CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
    CALIBRATED_QUANTIZATION_STARTED_REGISTRATION, CalibratedQuantizationReceiptView, EvidenceEvent,
};
use tritium_quantize::engine::{
    CalibrationBatch, CalibrationCache, CalibrationCacheBuilder, CalibrationCollector,
    CalibrationInput, CalibrationOutputBatch, InMemoryCalibrationSource, TritCalibrationSource,
    fit_calibrated_levels,
};

fn cal_recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "engine-run",
        [
            DIRECT_LEVELS_FITTED_REGISTRATION,
            CALIBRATED_LEVELS_FITTED_REGISTRATION,
            CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
            CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap()
}
fn fixture(
    precision: ScalePrecision,
    partition: CalibrationPartition,
) -> (InMemoryWeightSource, CalibratedQuantPlan, CalibrationCache) {
    fixture_context(
        precision,
        partition,
        CalibrationConsumer::Matmul,
        CalibrationOutput::DenseFisher,
    )
}

fn fixture_context(
    precision: ScalePrecision,
    partition: CalibrationPartition,
    consumer: CalibrationConsumer,
    output: CalibrationOutput,
) -> (InMemoryWeightSource, CalibratedQuantPlan, CalibrationCache) {
    let mut raw = source(precision);
    // A genuinely mixed plain/calibrated/dense request through the common pipeline.
    raw.tensors
        .insert("plain".into(), raw.tensors["weight"].clone());
    let request = plan(&raw);
    let model = Engine::observe_source(&request, &mut raw).unwrap();
    let original = request
        .tensors
        .iter()
        .find(|t| t.source.name == "weight")
        .unwrap()
        .source
        .clone();
    let mut input = vec![0.; 32 * 32];
    for i in 0..32 {
        input[i * 32 + i] = 1.;
    }
    let mut builder = CalibrationCacheBuilder::new(
        CalibrationCacheRequest {
            tensor: original,
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
        &raw.assets["tokenizer.json"],
        b"declared fixture objective",
    )
    .unwrap();
    builder
        .capture(
            &(0..32).collect::<Vec<u32>>(),
            CalibrationBatch {
                samples: 32,
                input: if consumer == CalibrationConsumer::Matmul {
                    CalibrationInput::Activations(&input)
                } else {
                    CalibrationInput::GatherIdentity
                },
                output: match output {
                    CalibrationOutput::None => CalibrationOutputBatch::None,
                    CalibrationOutput::DenseFisher => CalibrationOutputBatch::Dense(&[0.25; 32]),
                    _ => CalibrationOutputBatch::Indexed {
                        indices: &[0; 32],
                        factors: &[0.25; 32],
                    },
                },
                weights: None,
                mask: None,
            },
        )
        .unwrap();
    let cache = builder.finish().unwrap();
    let mut fit = CalibratedFitPlan {
        direct: direct(),
        cache: cache.binding().clone(),
        partition,
        damping: if output == CalibrationOutput::None {
            0.0
        } else {
            0.01
        },
        max_numeric_bytes: 1,
    };
    fit.max_numeric_bytes =
        CalibrationCollector::required_numeric_bytes(&fit.calibration_plan().unwrap()).unwrap()
            + cache.binding().payload_bytes;
    let request = CalibratedQuantPlan {
        architecture: request.architecture,
        original_source: model,
        tensors: request
            .tensors
            .into_iter()
            .map(|t| {
                let action = if t.source.name == "weight" {
                    CalibratedQuantAction::CalibratedFit {
                        plan: Box::new(fit.clone()),
                    }
                } else {
                    match t.action {
                        QuantAction::KeepDense => CalibratedQuantAction::KeepDense,
                        QuantAction::Fit { plan, metric } => {
                            CalibratedQuantAction::Fit { plan, metric }
                        }
                        _ => unreachable!(),
                    }
                };
                CalibratedQuantTensorPlan {
                    source: t.source,
                    action,
                }
            })
            .collect(),
        assets: request.assets,
        levels: request.levels,
        limits: request.limits,
        max_cache_payload_bytes: 100_000,
        max_cache_package_bytes: 100_000,
    };
    (raw, request, cache)
}

#[test]
fn both_cache_sources_create_byte_identical_checked_mixed_models() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        for partition in [
            CalibrationPartition::Full,
            CalibrationPartition::Groups { columns: 13 },
        ] {
            for (consumer, output) in [
                (CalibrationConsumer::Matmul, CalibrationOutput::None),
                (CalibrationConsumer::Matmul, CalibrationOutput::DenseFisher),
                (
                    CalibrationConsumer::Matmul,
                    CalibrationOutput::IndexedFisher,
                ),
                (
                    CalibrationConsumer::GatherIdentity,
                    CalibrationOutput::IndexedFisher,
                ),
            ] {
                let (mut raw, request, cache) =
                    fixture_context(precision, partition, consumer, output);
                let original = raw.clone();
                let factor = cache
                    .collect(
                        partition,
                        if output == CalibrationOutput::None {
                            0.0
                        } else {
                            0.01
                        },
                        request
                            .tensors
                            .iter()
                            .find_map(|t| match &t.action {
                                CalibratedQuantAction::CalibratedFit { plan } => {
                                    Some(plan.max_numeric_bytes)
                                }
                                _ => None,
                            })
                            .unwrap(),
                    )
                    .unwrap();
                let decoded = tritium_format::trit_model::decode_dense_f32(
                    "weight",
                    &[1, 32],
                    precision,
                    &raw.tensors["weight"].bytes,
                )
                .unwrap();
                let expected = fit_calibrated_levels(
                    &direct(),
                    &request.tensors.last().unwrap().source,
                    request.original_source,
                    &decoded,
                    &factor,
                )
                .unwrap();
                let encoded = cache.encode_package(100_000).unwrap();
                let mut memory = InMemoryCalibrationSource::new([cache]).unwrap();
                let mut sink = MemoryTensorSink::default();
                let result = Engine::run_calibrated(
                    &request,
                    &mut raw,
                    &mut memory,
                    &mut sink,
                    cal_recorder(),
                )
                .unwrap();
                assert_eq!(result.source_id, request.original_source);
                let (package, sealed) = sink.artifact().unwrap();
                verify_package_with_fit_log(package, sealed.log()).unwrap();
                let model = read_trit_model(package).unwrap();
                for fit in &expected {
                    assert_eq!(
                        model.tensor_digest(&fit.level, "weight").unwrap(),
                        tritium_format::trit_model::additive_semantic_digest(&fit.tensor).unwrap()
                    );
                }
                assert_eq!(
                    model.tensor("l1", "norm").unwrap().1,
                    original.tensors["norm"].bytes
                );
                let replay = verify_jsonl(sealed.jsonl()).unwrap();
                assert!(QuantizationReceiptView::fold(&replay).unwrap().is_empty());
                let views = CalibratedQuantizationReceiptView::fold(&replay).unwrap();
                assert_eq!(views.len(), 1);
                assert_eq!(views[0].fits().len(), 1);
                assert_eq!(views[0].calibrated_fits().len(), 1);
                assert_eq!(views[0].manifest(), model.manifest());
                assert!(
                    sealed
                        .jsonl()
                        .find("tritium.quantize.quantization_completed")
                        .is_none()
                );
                let path = std::env::temp_dir().join(format!(
                    "calibrated-engine-{}-{:?}-{:?}.trit",
                    std::process::id(),
                    precision,
                    partition
                ));
                let mut file = std::fs::OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&path)
                    .unwrap();
                use std::io::Write;
                file.write_all(&encoded).unwrap();
                drop(file);
                let mut files =
                    TritCalibrationSource::new([("weight".into(), path.clone())]).unwrap();
                let mut disk_sink = MemoryTensorSink::default();
                let converted = Engine::run_calibrated(
                    &request,
                    &mut raw,
                    &mut files,
                    &mut disk_sink,
                    cal_recorder(),
                );
                std::fs::remove_file(path).unwrap();
                converted.unwrap();
                let (disk_package, disk_log) = disk_sink.artifact().unwrap();
                assert_eq!(disk_package, package);
                assert_eq!(disk_log.jsonl(), sealed.jsonl());
            }
        }
    }
}

#[test]
fn actual_cache_file_errors_and_duplicate_inventory_never_publish() {
    let (raw, request, cache) = fixture(ScalePrecision::F32, CalibrationPartition::Full);
    let encoded = cache.encode_package(100_000).unwrap();
    let duplicate = fixture(ScalePrecision::F32, CalibrationPartition::Full).2;
    assert!(InMemoryCalibrationSource::new([cache, duplicate]).is_err());
    let path = std::env::temp_dir().join(format!(
        "calibrated-cache-fault-{}.trit",
        std::process::id()
    ));
    assert!(
        TritCalibrationSource::new([
            ("weight".into(), path.clone()),
            ("weight".into(), path.clone())
        ])
        .is_err()
    );
    for corruption in 0..6 {
        let mut request = request.clone();
        let mut bytes = encoded.clone();
        match corruption {
            0 => request.max_cache_package_bytes = bytes.len() as u64 - 1,
            1 => {
                bytes.pop();
            }
            2 => bytes[256] ^= 1,
            3 => bytes.extend_from_slice(b"unadmitted trailing bytes"),
            _ => (),
        }
        if corruption != 4 {
            std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)
                .unwrap()
                .write_all(&bytes)
                .unwrap();
        }
        let name = if corruption == 5 {
            "unrequested"
        } else {
            "weight"
        };
        let mut files = TritCalibrationSource::new([(name.into(), path.clone())]).unwrap();
        let mut sink = MemoryTensorSink::default();
        let result = Engine::run_calibrated(
            &request,
            &mut raw.clone(),
            &mut files,
            &mut sink,
            cal_recorder(),
        );
        if corruption != 4 {
            std::fs::remove_file(&path).unwrap();
        }
        assert!(result.is_err(), "file corruption {corruption}");
        assert!(sink.artifact().is_none());
    }
}

#[test]
fn actual_source_cache_inventory_and_bounds_fail_without_publication() {
    for corruption in 0..8 {
        let (mut raw, mut request, cache) =
            fixture(ScalePrecision::F32, CalibrationPartition::Full);
        let mut caches = InMemoryCalibrationSource::new([cache]).unwrap();
        match corruption {
            0 => raw.tensors.get_mut("norm").unwrap().bytes[4..8]
                .copy_from_slice(&2f32.to_le_bytes()),
            1 => raw.assets.get_mut("config.json").unwrap().push(b' '),
            2 => caches = InMemoryCalibrationSource::default(),
            3 => request.max_cache_payload_bytes = 1,
            4 => request.original_source = OriginalSourceId::from_bytes([42; 32]),
            5 => request.tensors.pop().map(|_| ()).unwrap(),
            6 => request.limits.max_package_bytes = 1,
            _ => {
                if let CalibratedQuantAction::CalibratedFit { plan } =
                    &mut request.tensors.last_mut().unwrap().action
                {
                    plan.cache.objective = BlobId::from_bytes([42; 32]);
                }
            }
        }
        let mut sink = MemoryTensorSink::default();
        assert!(
            Engine::run_calibrated(&request, &mut raw, &mut caches, &mut sink, cal_recorder())
                .is_err()
        );
        assert!(sink.artifact().is_none());
    }
}

#[test]
fn replay_rejects_source_and_fit_class_swaps_even_with_rehashed_chains() {
    let (mut raw, request, cache) = fixture(ScalePrecision::F32, CalibrationPartition::Full);
    let mut source = InMemoryCalibrationSource::new([cache]).unwrap();
    let mut sink = MemoryTensorSink::default();
    let result =
        Engine::run_calibrated(&request, &mut raw, &mut source, &mut sink, cal_recorder()).unwrap();
    for corruption in 0..6 {
        let mut log = cal_recorder();
        for event in &result.evidence.log().events {
            if event.schema == CalibratedQuantizationStarted::SCHEMA_ID {
                let mut start: CalibratedQuantizationStarted =
                    serde_json::from_value(event.payload.clone()).unwrap();
                if corruption == 0 {
                    start.plan.original_source = OriginalSourceId::from_bytes([42; 32]);
                }
                log.emit(&event.span, event.parent.as_deref(), event.t, &start)
                    .unwrap();
            } else if event.schema == CalibratedLevelsFitted::SCHEMA_ID {
                let mut fit: CalibratedLevelsFitted =
                    serde_json::from_value(event.payload.clone()).unwrap();
                if corruption == 1 {
                    fit.plan.damping = 0.02;
                    fit.factors.plan.damping = 0.02;
                }
                if corruption != 2 {
                    log.emit(&event.span, event.parent.as_deref(), event.t, &fit)
                        .unwrap();
                }
            } else if event.schema == DirectLevelsFitted::SCHEMA_ID {
                let fit: DirectLevelsFitted =
                    serde_json::from_value(event.payload.clone()).unwrap();
                log.emit(&event.span, event.parent.as_deref(), event.t, &fit)
                    .unwrap();
            } else {
                let mut completed: CalibratedQuantizationCompleted =
                    serde_json::from_value(event.payload.clone()).unwrap();
                match corruption {
                    3 => completed.original_source = OriginalSourceId::from_bytes([42; 32]),
                    4 => completed.output.sources[0].content = BlobId::from_bytes([42; 32]),
                    5 => completed.output.levels[0]
                        .tensors
                        .pop()
                        .map(|_| ())
                        .unwrap(),
                    _ => (),
                }
                log.emit(&event.span, event.parent.as_deref(), event.t, &completed)
                    .unwrap();
            }
        }
        let forged = log.seal().unwrap();
        assert!(CalibratedQuantizationReceiptView::fold(forged.log()).is_err());
        assert!(verify_package_with_fit_log(sink.artifact().unwrap().0, forged.log()).is_err());
    }
}

#[test]
fn lifecycle_replay_rejects_future_versions_unknown_fields_and_wrong_roots() {
    let (mut raw, request, cache) = fixture(ScalePrecision::F32, CalibrationPartition::Full);
    let mut caches = InMemoryCalibrationSource::new([cache]).unwrap();
    let result = Engine::run_calibrated(
        &request,
        &mut raw,
        &mut caches,
        &mut MemoryTensorSink::default(),
        cal_recorder(),
    )
    .unwrap();
    let views = CalibratedQuantizationReceiptView::fold(result.evidence.log()).unwrap();
    let mut wrong_root = result.evidence.log().clone();
    wrong_root.root_digest = "00".repeat(32);
    assert!(CalibratedQuantizationReceiptView::fold(&wrong_root).is_err());

    #[derive(serde::Serialize)]
    #[serde(transparent)]
    struct FutureStart<'a>(&'a CalibratedQuantizationStarted);
    impl EvidenceEvent for FutureStart<'_> {
        const SCHEMA_ID: &'static str = CalibratedQuantizationStarted::SCHEMA_ID;
        const VERSION: u16 = 2;
    }
    #[derive(serde::Serialize)]
    #[serde(transparent)]
    struct FutureCompletion<'a>(&'a CalibratedQuantizationCompleted);
    impl EvidenceEvent for FutureCompletion<'_> {
        const SCHEMA_ID: &'static str = CalibratedQuantizationCompleted::SCHEMA_ID;
        const VERSION: u16 = 2;
    }
    for completion in [false, true] {
        let registry = tritium_evidence::EventRegistration {
            schema: if completion {
                FutureCompletion::SCHEMA_ID
            } else {
                FutureStart::SCHEMA_ID
            },
            version: 2,
        };
        let mut log = EvidenceRecorder::new("future", [registry]).unwrap();
        if completion {
            log.emit("quantize", None, 0, &FutureCompletion(views[0].completed()))
                .unwrap();
        } else {
            log.emit("quantize", None, 0, &FutureStart(views[0].started()))
                .unwrap();
        }
        assert!(matches!(
            CalibratedQuantizationReceiptView::fold(log.seal().unwrap().log()),
            Err(tritium_evidence::EvidenceError::UnregisteredSchema(_))
        ));
    }
    #[derive(serde::Serialize)]
    #[serde(transparent)]
    struct Payload(serde_json::Value);
    impl EvidenceEvent for Payload {
        const SCHEMA_ID: &'static str = CalibratedQuantizationStarted::SCHEMA_ID;
        const VERSION: u16 = 1;
    }
    let mut payload = serde_json::to_value(views[0].started()).unwrap();
    payload["invented_release_approval"] = true.into();
    let mut log = cal_recorder();
    log.emit("quantize", None, 0, &Payload(payload)).unwrap();
    assert!(CalibratedQuantizationReceiptView::fold(log.seal().unwrap().log()).is_err());
    let mut partial = cal_recorder();
    partial
        .emit("quantize", None, 0, views[0].started())
        .unwrap();
    assert!(
        CalibratedQuantizationReceiptView::fold(partial.seal().unwrap().log())
            .unwrap()
            .is_empty()
    );
}
