//! Real shared model engine, distinct cache adapters and complete package replay.
use super::*;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use tritium_evidence::*;
use tritium_quantize::engine::*;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "tritium-owner-model-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "owner-model",
        [
            OWNER_QUANTIZATION_STARTED_REGISTRATION,
            OWNER_QUANTIZATION_COMPLETED_REGISTRATION,
            OWNER_SCORE_LEVELS_FITTED_REGISTRATION,
            DIRECT_LEVELS_FITTED_REGISTRATION,
            CALIBRATED_LEVELS_FITTED_REGISTRATION,
        ],
    )
    .unwrap()
}
fn owner_cache(
    tensor: &QuantSourceTensor,
    model: OriginalSourceId,
    window: i64,
) -> OwnerScoreCache {
    let inputs: Vec<f32> = (0..64)
        .map(|i| ((i + 3) % 7) as f32 / 7. + window as f32 * 0.02)
        .collect();
    let gathered: Vec<f32> = (0..64)
        .map(|i| ((i + window as usize) % 5) as f32 * 0.1)
        .collect();
    let uses = [
        OwnerScoreUse::Matmul {
            samples: 2,
            inputs: &inputs,
            adjoints: &[0.25, -0.5],
        },
        OwnerScoreUse::Gather {
            indices: &[0, 0],
            adjoints: &gathered,
        },
    ];
    let tokens = [0_u8; 16];
    let context = window.to_le_bytes();
    OwnerScoreCache::new(
        OwnerScoreCacheInput {
            source: model,
            tensor: tensor.clone(),
            tokenizer: b"{}",
            objective: b"owner fixture scalar",
            meta: OwnerScoreWindowMeta {
                context: vec![
                    OwnerContextMeta {
                        name: "input_ids".into(),
                        dtype: OwnerContextDtype::I64,
                        shape: vec![2],
                        conjugated: false,
                        negated: false,
                    },
                    OwnerContextMeta {
                        name: "window_id".into(),
                        dtype: OwnerContextDtype::I64,
                        shape: vec![1],
                        conjugated: false,
                        negated: false,
                    },
                ],
                invocations: vec![
                    OwnerInvocationMeta {
                        ordinal: 0,
                        paths: vec!["head".into()],
                        role: OwnerScoreRole::Matmul,
                        samples: 2,
                    },
                    OwnerInvocationMeta {
                        ordinal: 1,
                        paths: vec!["embed".into()],
                        role: OwnerScoreRole::Gather,
                        samples: 2,
                    },
                ],
            },
            context: &[&tokens, &context],
            observations: Arc::new(
                OwnerScoreWindow::new(1, 32, &uses, 100_000)
                    .unwrap()
                    .snapshot()
                    .unwrap(),
            ),
        },
        OwnerScoreCacheLimits {
            max_payload_bytes: 100_000,
            max_context_bytes: 1024,
            max_context_tensors: 8,
            max_invocations: 8,
            max_samples: 32,
        },
    )
    .unwrap()
}
fn fixture(
    precision: ScalePrecision,
) -> (
    InMemoryWeightSource,
    OwnerQuantPlan,
    Vec<OwnerScoreCache>,
    CalibrationCache,
) {
    let mut raw = source(precision);
    raw.tensors
        .insert("calibrated".into(), raw.tensors["weight"].clone());
    raw.tensors
        .insert("plain".into(), raw.tensors["weight"].clone());
    let mut plain = plan(&raw);
    plain.limits.max_retained_blob_bytes = 100_000;
    let model = Engine::observe_source(&plain, &mut raw).unwrap();
    let tensor = plain
        .tensors
        .iter()
        .find(|t| t.source.name == "weight")
        .unwrap()
        .source
        .clone();
    let caches = vec![
        owner_cache(&tensor, model, 0),
        owner_cache(&tensor, model, 1),
    ];
    let owner = OwnerScoreFitPlan {
        direct: direct(),
        windows: vec![
            WeightedOwnerCache {
                binding: caches[0].binding().clone(),
                weight: 1.,
            },
            WeightedOwnerCache {
                binding: caches[1].binding().clone(),
                weight: 2.,
            },
            WeightedOwnerCache {
                binding: caches[0].binding().clone(),
                weight: 0.5,
            },
        ],
        damping: 0.1,
        limits: OwnerScoreMetricLimits {
            max_windows: 8,
            max_payload_bytes: 200_000,
            max_row_bytes: 1024,
            max_reduction_entries: 50_000,
        },
        max_row_bindings: 1,
    };
    let cal_tensor = plain
        .tensors
        .iter()
        .find(|t| t.source.name == "calibrated")
        .unwrap()
        .source
        .clone();
    let mut builder = CalibrationCacheBuilder::new(
        CalibrationCacheRequest {
            tensor: cal_tensor,
            model,
            consumer: CalibrationConsumer::Matmul,
            output: CalibrationOutput::DenseFisher,
            sample_start: 0,
            limits: CalibrationCacheLimits {
                max_payload_bytes: 100_000,
                max_samples: 64,
                max_windows: 4,
            },
        },
        b"{}",
        b"legacy fixture",
    )
    .unwrap();
    let mut diagonal = vec![0.; 1024];
    for i in 0..32 {
        diagonal[i * 32 + i] = 1.;
    }
    builder
        .capture(
            &(0..32).collect::<Vec<_>>(),
            CalibrationBatch {
                samples: 32,
                input: CalibrationInput::Activations(&diagonal),
                output: CalibrationOutputBatch::Dense(&[0.25; 32]),
                weights: None,
                mask: None,
            },
        )
        .unwrap();
    let cal = builder.finish().unwrap();
    let mut fit = CalibratedFitPlan {
        direct: direct(),
        cache: cal.binding().clone(),
        partition: CalibrationPartition::Full,
        damping: 0.01,
        max_numeric_bytes: 1,
    };
    fit.max_numeric_bytes =
        CalibrationCollector::required_numeric_bytes(&fit.calibration_plan().unwrap()).unwrap()
            + cal.binding().payload_bytes;
    let request = OwnerQuantPlan {
        architecture: plain.architecture,
        original_source: model,
        tensors: plain
            .tensors
            .into_iter()
            .map(|t| OwnerQuantTensorPlan {
                source: t.source.clone(),
                action: match t.source.name.as_str() {
                    "weight" => OwnerQuantAction::OwnerFit {
                        plan: Box::new(owner.clone()),
                    },
                    "calibrated" => OwnerQuantAction::CalibratedFit {
                        plan: Box::new(fit.clone()),
                    },
                    "plain" => OwnerQuantAction::Fit {
                        plan: Box::new(direct()),
                        metric: DirectFitMetric::Identity,
                    },
                    _ => OwnerQuantAction::KeepDense,
                },
            })
            .collect(),
        assets: plain.assets,
        levels: plain.levels,
        limits: plain.limits,
        max_cache_payload_bytes: 500_000,
        max_cache_package_bytes: 200_000,
    };
    request.validate().unwrap();
    (raw, request, caches, cal)
}
fn owner_plan(plan: &OwnerQuantPlan) -> &OwnerScoreFitPlan {
    plan.tensors
        .iter()
        .find_map(|t| {
            if let OwnerQuantAction::OwnerFit { plan } = &t.action {
                Some(plan.as_ref())
            } else {
                None
            }
        })
        .unwrap()
}
fn file_source(caches: &[OwnerScoreCache], dir: &std::path::Path) -> TritOwnerCalibrationSource {
    let mut paths = BTreeMap::new();
    for (i, cache) in caches.iter().enumerate() {
        let path = dir.join(format!("{i}.trit"));
        std::fs::write(&path, cache.encode_package(200_000).unwrap()).unwrap();
        paths.insert(*cache.id().as_bytes(), path);
    }
    TritOwnerCalibrationSource::new(BTreeMap::from([("weight".into(), paths)])).unwrap()
}

#[test]
fn actual_mixed_owner_models_preserve_raw_bits_fit_outputs_and_both_cache_adapters() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        let (mut raw, request, caches, cal) = fixture(precision);
        let originals = raw.clone();
        let expected = owner_plan(&request);
        let refs: Vec<_> = caches.iter().collect();
        let windows: Vec<_> = expected.windows.iter().map(|w| w.window()).collect();
        let ordered = [refs[0], refs[1], refs[0]];
        let metrics =
            OwnerScoreMetrics::new(&ordered, &windows, expected.damping, expected.limits).unwrap();
        let original = &raw.tensors["weight"];
        let values = tritium_format::trit_model::decode_dense_f32(
            "weight",
            &original.shape,
            precision,
            &original.bytes,
        )
        .unwrap();
        let expected = fit_owner_score_levels(
            &expected.direct,
            metrics.tensor(),
            metrics.source(),
            &values,
            &metrics,
        )
        .unwrap();
        let scratch = Scratch::new();
        let mut files = file_source(&caches, &scratch.0);
        let (_, _, _, other_cal) = fixture(precision);
        let mut legacy = InMemoryCalibrationSource::new([cal]).unwrap();
        let mut owners = InMemoryOwnerCalibrationSource::new([("weight".into(), caches)]).unwrap();
        {
            let stream = owners.stream(owner_plan(&request), 200_000).unwrap();
            assert!(std::ptr::eq(stream[0], stream[2]));
        }
        let mut sink = MemoryTensorSink::default();
        let result = Engine::run_owner(
            &request,
            &mut raw,
            &mut owners,
            Some(&mut legacy),
            &mut sink,
            recorder(),
        )
        .unwrap();
        assert_eq!(result.source_id, request.original_source);
        let (bytes, evidence) = sink.artifact().unwrap();
        verify_package_with_fit_log(bytes, evidence.log()).unwrap();
        let view = OwnerQuantizationReceiptView::fold(evidence.log()).unwrap();
        let view = &view[0];
        assert_eq!(view.owner_fits().len(), 1);
        assert_eq!(view.calibrated_fits().len(), 1);
        assert_eq!(view.fits().len(), 1);
        assert_eq!(view.source_id(), request.original_source);
        assert!(
            QuantizationReceiptView::fold(evidence.log())
                .unwrap()
                .is_empty()
        );
        assert!(
            CalibratedQuantizationReceiptView::fold(evidence.log())
                .unwrap()
                .is_empty()
        );
        let model = read_trit_model(bytes).unwrap();
        for level in &request.levels {
            assert_eq!(
                model.tensor(level, "norm").unwrap().1,
                originals.tensors["norm"].bytes
            );
        }
        for level in &expected {
            assert_eq!(
                model.tensor_digest(&level.level, "weight").unwrap(),
                tritium_format::trit_model::additive_semantic_digest(&level.tensor).unwrap()
            );
        }
        let mut seek_sink = MemoryTensorSink::default();
        let mut source = originals.clone();
        let mut legacy = InMemoryCalibrationSource::new([other_cal]).unwrap();
        let other = Engine::run_owner(
            &request,
            &mut source,
            &mut files,
            Some(&mut legacy),
            &mut seek_sink,
            recorder(),
        )
        .unwrap();
        assert_eq!(result.package_id, other.package_id);
        assert_eq!(result.evidence, other.evidence);
        assert_eq!(bytes, seek_sink.artifact().unwrap().0);
        if let Some(path) = std::env::var_os("TRITIUM_OWNER_MODEL_ORACLE_DIR") {
            let dir = std::path::Path::new(&path).join(format!("{precision:?}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.trit"), bytes).unwrap();
            std::fs::write(dir.join("evidence.jsonl"), evidence.jsonl()).unwrap();
        }
    }
}

#[test]
fn owner_source_context_and_limits_fail_before_publication_without_fallback() {
    for change in 0..9 {
        let (mut raw, mut request, caches, cal) = fixture(ScalePrecision::F32);
        let mut sink = MemoryTensorSink::default();
        let mut owners = InMemoryOwnerCalibrationSource::new([("weight".into(), caches)]).unwrap();
        let mut legacy = InMemoryCalibrationSource::new([cal]).unwrap();
        match change {
            0 => raw.tensors.get_mut("plain").unwrap().bytes[0] ^= 1,
            1 => {
                raw.assets.insert("config.json".into(), b"{ }".to_vec());
            }
            2 => request.max_cache_payload_bytes = 1,
            3 => request.original_source = OriginalSourceId::from_bytes([99; 32]),
            4 => owners = InMemoryOwnerCalibrationSource::default(),
            5 => legacy = InMemoryCalibrationSource::default(),
            6 => request.limits.max_source_bytes = 1,
            7 => {
                if let OwnerQuantAction::OwnerFit { plan } = &mut request
                    .tensors
                    .iter_mut()
                    .find(|t| t.source.name == "weight")
                    .unwrap()
                    .action
                {
                    plan.windows[0].binding.content.objective = BlobId::from_bytes([99; 32]);
                }
            }
            _ => request.levels.push("extra".into()),
        }
        assert!(
            Engine::run_owner(
                &request,
                &mut raw,
                &mut owners,
                Some(&mut legacy),
                &mut sink,
                recorder()
            )
            .is_err(),
            "change {change}"
        );
        assert!(sink.artifact().is_none());
    }
}

#[test]
fn file_owner_cache_bounds_corruption_inventory_and_no_clobber_are_enforced() {
    for change in 0..4 {
        let (mut raw, mut request, caches, cal) = fixture(ScalePrecision::F32);
        let scratch = Scratch::new();
        let mut files = file_source(&caches, &scratch.0);
        match change {
            0 => request.max_cache_package_bytes = 1,
            1 => {
                let path = scratch.0.join("1.trit");
                let mut bytes = std::fs::read(&path).unwrap();
                let i = bytes.len() - 1;
                bytes[i] ^= 1;
                std::fs::write(path, bytes).unwrap();
            }
            2 => {
                std::fs::remove_file(scratch.0.join("0.trit")).unwrap();
            }
            _ => files = TritOwnerCalibrationSource::new(BTreeMap::new()).unwrap(),
        }
        let mut sink = MemoryTensorSink::default();
        let mut legacy = InMemoryCalibrationSource::new([cal]).unwrap();
        assert!(
            Engine::run_owner(
                &request,
                &mut raw,
                &mut files,
                Some(&mut legacy),
                &mut sink,
                recorder()
            )
            .is_err()
        );
        assert!(sink.artifact().is_none());
    }
    let (mut raw, request, caches, cal) = fixture(ScalePrecision::F32);
    let scratch = Scratch::new();
    let mut files = file_source(&caches, &scratch.0);
    let destination = scratch.0.join("output");
    let mut sink = DirectoryTensorSink::new(&destination);
    let mut legacy = InMemoryCalibrationSource::new([cal]).unwrap();
    Engine::run_owner(
        &request,
        &mut raw,
        &mut files,
        Some(&mut legacy),
        &mut sink,
        recorder(),
    )
    .unwrap();
    let before = std::fs::read(destination.join("model.trit")).unwrap();
    assert!(
        Engine::run_owner(
            &request,
            &mut raw,
            &mut files,
            Some(&mut legacy),
            &mut sink,
            recorder()
        )
        .is_err()
    );
    assert_eq!(
        before,
        std::fs::read(destination.join("model.trit")).unwrap()
    );
}

#[test]
fn mismatched_owner_log_and_package_cannot_be_rebound() {
    let (mut raw, request, caches, cal) = fixture(ScalePrecision::F32);
    let mut owners = InMemoryOwnerCalibrationSource::new([("weight".into(), caches)]).unwrap();
    let mut legacy = InMemoryCalibrationSource::new([cal]).unwrap();
    let mut a = MemoryTensorSink::default();
    Engine::run_owner(
        &request,
        &mut raw,
        &mut owners,
        Some(&mut legacy),
        &mut a,
        recorder(),
    )
    .unwrap();
    let mut changed = request.clone();
    if let OwnerQuantAction::OwnerFit { plan } = &mut changed
        .tensors
        .iter_mut()
        .find(|t| t.source.name == "weight")
        .unwrap()
        .action
    {
        plan.damping = 0.2;
    }
    let mut b = MemoryTensorSink::default();
    Engine::run_owner(
        &changed,
        &mut raw,
        &mut owners,
        Some(&mut legacy),
        &mut b,
        recorder(),
    )
    .unwrap();
    assert!(
        verify_package_with_fit_log(b.artifact().unwrap().0, a.artifact().unwrap().1.log())
            .is_err()
    );
    let mut altered = a.artifact().unwrap().1.log().clone();
    altered.events[0].payload["release_approved"] = true.into();
    assert!(OwnerQuantizationReceiptView::fold(&altered).is_err());
}

#[test]
fn complete_owner_replay_rejects_resealed_inventory_source_and_fit_swaps() {
    let (mut raw, request, caches, cal) = fixture(ScalePrecision::F32);
    let mut owners = InMemoryOwnerCalibrationSource::new([("weight".into(), caches)]).unwrap();
    let mut legacy = InMemoryCalibrationSource::new([cal]).unwrap();
    let mut sink = MemoryTensorSink::default();
    let result = Engine::run_owner(
        &request,
        &mut raw,
        &mut owners,
        Some(&mut legacy),
        &mut sink,
        recorder(),
    )
    .unwrap();
    for change in 0..13 {
        let mut replay = recorder();
        for event in &result.evidence.log().events {
            let parent = event.parent.as_deref();
            match event.schema.as_str() {
                OwnerQuantizationStarted::SCHEMA_ID => {
                    let mut value: OwnerQuantizationStarted =
                        serde_json::from_value(event.payload.clone()).unwrap();
                    match change {
                        0 => value.plan.original_source = OriginalSourceId::from_bytes([42; 32]),
                        1 => {
                            let tensor = value
                                .plan
                                .tensors
                                .iter_mut()
                                .find(|t| t.source.name == "weight")
                                .unwrap();
                            tensor.action = OwnerQuantAction::KeepDense;
                        }
                        _ => (),
                    }
                    replay.emit(&event.span, parent, event.t, &value).unwrap();
                }
                OwnerScoreLevelsFitted::SCHEMA_ID => {
                    let value: OwnerScoreLevelsFitted =
                        serde_json::from_value(event.payload.clone()).unwrap();
                    if change != 2 {
                        replay
                            .emit(
                                &event.span,
                                if change == 3 { None } else { parent },
                                event.t,
                                &value,
                            )
                            .unwrap();
                    }
                    if change == 4 {
                        replay
                            .emit("duplicate-owner", parent, event.t, &value)
                            .unwrap();
                    }
                }
                CalibratedLevelsFitted::SCHEMA_ID => {
                    let value: CalibratedLevelsFitted =
                        serde_json::from_value(event.payload.clone()).unwrap();
                    if change != 5 {
                        replay.emit(&event.span, parent, event.t, &value).unwrap();
                    }
                }
                DirectLevelsFitted::SCHEMA_ID => {
                    let value: DirectLevelsFitted =
                        serde_json::from_value(event.payload.clone()).unwrap();
                    if change != 6 {
                        replay.emit(&event.span, parent, event.t, &value).unwrap();
                    }
                }
                OwnerQuantizationCompleted::SCHEMA_ID => {
                    let mut value: OwnerQuantizationCompleted =
                        serde_json::from_value(event.payload.clone()).unwrap();
                    match change {
                        7 => value.original_source = OriginalSourceId::from_bytes([42; 32]),
                        8 => value.output.sources[0].content = BlobId::from_bytes([42; 32]),
                        9 => {
                            value.output.levels[0].tensors.pop();
                        }
                        10 => {
                            value.output.assets.pop();
                        }
                        11 => {
                            value.output.levels[0]
                                .tensors
                                .iter_mut()
                                .find(|t| t.name == "norm")
                                .unwrap()
                                .blob
                                .id = BlobId::from_bytes([42; 32]);
                        }
                        12 => {
                            value.output.levels[0].tensors[0].blob.id = BlobId::from_bytes([42; 32])
                        }
                        _ => (),
                    }
                    replay.emit(&event.span, parent, event.t, &value).unwrap();
                }
                _ => panic!("unexpected fixture schema"),
            }
        }
        let forged = replay.seal().unwrap();
        // The sealed chain itself is valid; typed model agreement is not.
        verify_jsonl(forged.jsonl()).unwrap();
        if change == 12 {
            // A declared encoded fit-blob hash cannot be derived from the
            // recorded semantic digest alone. Actual package replay must reject it.
            OwnerQuantizationReceiptView::fold(forged.log()).unwrap();
        } else {
            assert!(
                OwnerQuantizationReceiptView::fold(forged.log()).is_err(),
                "change {change}"
            );
        }
        assert!(verify_package_with_fit_log(sink.artifact().unwrap().0, forged.log()).is_err());
    }
}

#[test]
fn owner_lifecycle_versions_unknown_fields_and_partial_stages_remain_fail_closed() {
    let (mut raw, request, caches, cal) = fixture(ScalePrecision::F32);
    let mut owners = InMemoryOwnerCalibrationSource::new([("weight".into(), caches)]).unwrap();
    let mut legacy = InMemoryCalibrationSource::new([cal]).unwrap();
    let result = Engine::run_owner(
        &request,
        &mut raw,
        &mut owners,
        Some(&mut legacy),
        &mut MemoryTensorSink::default(),
        recorder(),
    )
    .unwrap();
    let views = OwnerQuantizationReceiptView::fold(result.evidence.log()).unwrap();
    #[derive(serde::Serialize)]
    #[serde(transparent)]
    struct FutureStart<'a>(&'a OwnerQuantizationStarted);
    impl EvidenceEvent for FutureStart<'_> {
        const SCHEMA_ID: &'static str = OwnerQuantizationStarted::SCHEMA_ID;
        const VERSION: u16 = 2;
    }
    #[derive(serde::Serialize)]
    #[serde(transparent)]
    struct FutureCompleted<'a>(&'a OwnerQuantizationCompleted);
    impl EvidenceEvent for FutureCompleted<'_> {
        const SCHEMA_ID: &'static str = OwnerQuantizationCompleted::SCHEMA_ID;
        const VERSION: u16 = 2;
    }
    for complete in [false, true] {
        let mut log = EvidenceRecorder::new(
            "future",
            [EventRegistration {
                schema: if complete {
                    FutureCompleted::SCHEMA_ID
                } else {
                    FutureStart::SCHEMA_ID
                },
                version: 2,
            }],
        )
        .unwrap();
        if complete {
            log.emit("quantize", None, 0, &FutureCompleted(views[0].completed()))
                .unwrap();
        } else {
            log.emit("quantize", None, 0, &FutureStart(views[0].started()))
                .unwrap();
        }
        assert!(matches!(
            OwnerQuantizationReceiptView::fold(log.seal().unwrap().log()),
            Err(EvidenceError::UnregisteredSchema(_))
        ));
    }
    #[derive(serde::Serialize)]
    #[serde(transparent)]
    struct Unknown(serde_json::Value);
    impl EvidenceEvent for Unknown {
        const SCHEMA_ID: &'static str = OwnerQuantizationStarted::SCHEMA_ID;
        const VERSION: u16 = 1;
    }
    let mut payload = serde_json::to_value(views[0].started()).unwrap();
    payload["release_approved"] = true.into();
    let mut log = recorder();
    log.emit("quantize", None, 0, &Unknown(payload)).unwrap();
    assert!(OwnerQuantizationReceiptView::fold(log.seal().unwrap().log()).is_err());
    let mut partial = recorder();
    partial
        .emit("quantize", None, 0, views[0].started())
        .unwrap();
    assert!(
        OwnerQuantizationReceiptView::fold(partial.seal().unwrap().log())
            .unwrap()
            .is_empty()
    );
}
