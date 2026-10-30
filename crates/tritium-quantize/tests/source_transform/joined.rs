//! Actual joined production/admission; not capture or empirical qualification.
use super::*;
use ciborium::value::{CanonicalValue, Value};
use tritium_evidence::VerifiedLog;
use tritium_format::trit_salience_conversion::{
    decode_salience_conversion_manifest, encode_salience_conversion_manifest,
};
use tritium_format::{TritBlobKind, read_trit_package};
use tritium_quantize::engine::{JoinedSalienceModel, StagedSalienceSource};

fn limits() -> SalienceConversionLimits {
    SalienceConversionLimits {
        max_package_bytes: 100_000,
        max_manifest_bytes: 50_000,
    }
}

fn fitting(stage: &StagedSalienceSource<'_>) -> QuantPlan {
    let mut request = dense_plan(stage);
    request
        .tensors
        .iter_mut()
        .find(|t| t.source.name == "q")
        .unwrap()
        .action = QuantAction::Fit {
        plan: Box::new(direct()),
        metric: DirectFitMetric::Identity,
    };
    request
}

fn fit(stage: &mut StagedSalienceSource<'_>, dense: bool) -> (Vec<u8>, VerifiedLog) {
    let request = if dense {
        dense_plan(stage)
    } else {
        fitting(stage)
    };
    let mut sink = MemoryTensorSink::default();
    let output = Engine::run_salience(&request, stage, &mut sink, recorder(), limits()).unwrap();
    assert_eq!(output.source_id, stage.source_id());
    let (bytes, seal) = sink.artifact().unwrap();
    assert_eq!(
        output.package_id.as_bytes(),
        read_trit_package(bytes).unwrap().package_id().as_bytes()
    );
    let log = verify_jsonl(seal.jsonl()).unwrap();
    let joined = JoinedSalienceModel::open(bytes, stage, &log, limits()).unwrap();
    assert_eq!(
        output.model_ids["l1"],
        joined.model().model_id("l1").unwrap()
    );
    (bytes.into(), log)
}

#[test]
fn cold_stage_plain_and_dense_only_conversions_use_actual_mandatory_roots() {
    let (mut a, recorded, stats) = stage::recorded();
    let bytes = recorded
        .encode_stage_package(&stats, stage::limits())
        .unwrap();
    let transform = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    drop(recorded);
    let mut stage = StagedSalienceSource::open(&bytes, &transform, stage::limits()).unwrap();
    let norm_b = stage.tensor("attn_norm", 128).unwrap().into_owned();
    for dense in [false, true] {
        let (c, log) = fit(&mut stage, dense);
        let joined = JoinedSalienceModel::open(&c, &stage, &log, limits()).unwrap();
        let carrier = joined.manifest();
        assert_eq!(carrier.stage_package, stage.package_id());
        assert_eq!(
            carrier.original_source,
            stage.manifest().observation.original_source
        );
        assert_eq!(carrier.derived_source, stage.source_id());
        assert_eq!(carrier.transform, stage.manifest().transform);
        assert_eq!(
            &carrier.conversion,
            QuantizationReceiptView::fold(&log).unwrap()[0].reference()
        );
        assert_eq!(joined.model().tensor("l1", "attn_norm").unwrap().1, norm_b);
        assert_eq!(
            matches!(
                joined.model().tensor("l1", "q").unwrap().0.storage,
                TensorStorage::Additive { .. }
            ),
            !dense
        );
        assert_eq!(
            joined.model().asset("config.json").unwrap(),
            a.assets["config.json"]
        );
        joined.verify_transform(&stage, &mut a).unwrap();
        assert!(
            read_trit_model(&c).is_err(),
            "legacy reader must not unwrap mandatory lineage"
        );
        assert!(verify_package_with_fit_log(&c, &log).is_err());
    }
}

#[test]
fn identity_transform_a_equals_b_is_valid_and_dense_only_has_a_real_conversion_root() {
    let mut a = fixture(false);
    let (mut request, stats) = plan(&mut a);
    for op in &mut request.operations {
        op.alpha_bits = 0.0f64.to_le_bytes();
    }
    let recorded =
        SourceTransform::run_recorded(&request, &mut a, &stats, transform_recorder()).unwrap();
    let bytes = recorded
        .encode_stage_package(&stats, stage::limits())
        .unwrap();
    let transform = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    drop(recorded);
    let mut stage = StagedSalienceSource::open(&bytes, &transform, stage::limits()).unwrap();
    let (c, log) = fit(&mut stage, true);
    let joined = JoinedSalienceModel::open(&c, &stage, &log, limits()).unwrap();
    assert_eq!(
        joined.manifest().original_source,
        joined.manifest().derived_source
    );
    assert_eq!(log.events.len(), 2);
    joined.verify_transform(&stage, &mut a).unwrap();
}

#[test]
fn rehashed_carrier_substitutions_cannot_invent_stage_source_or_evidence() {
    let (_, recorded, stats) = stage::recorded();
    let bytes = recorded
        .encode_stage_package(&stats, stage::limits())
        .unwrap();
    let transform = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let mut stage = StagedSalienceSource::open(&bytes, &transform, stage::limits()).unwrap();
    let (c, log) = fit(&mut stage, false);
    let original = JoinedSalienceModel::open(&c, &stage, &log, limits())
        .unwrap()
        .manifest()
        .clone();
    for case in 0..9 {
        let mut changed = original.clone();
        match case {
            0 => changed.stage_package = PackageId::from_bytes([1; 32]),
            1 => changed.original_source = OriginalSourceId::from_bytes([1; 32]),
            2 => changed.derived_source = OriginalSourceId::from_bytes([1; 32]),
            3 => changed.transform.run = "foreign".into(),
            4 => changed.transform.span = "foreign".into(),
            5 => changed.transform.root = [1; 32],
            6 => changed.conversion.run = "foreign".into(),
            7 => changed.conversion.span = "foreign".into(),
            8 => changed.conversion.root = [1; 32],
            _ => unreachable!(),
        }
        let metadata = encode_salience_conversion_manifest(&changed, 50_000).unwrap();
        let malicious = stage::repack(&metadata, &stage::entries(&c));
        assert!(read_trit_package(&malicious).is_ok());
        assert!(
            JoinedSalienceModel::open(&malicious, &stage, &log, limits()).is_err(),
            "case {case}"
        );
    }
}

#[test]
fn valid_hashes_do_not_admit_missing_extra_wrong_kind_or_changed_payloads() {
    let (_, recorded, stats) = stage::recorded();
    let bytes = recorded
        .encode_stage_package(&stats, stage::limits())
        .unwrap();
    let transform = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let mut stage = StagedSalienceSource::open(&bytes, &transform, stage::limits()).unwrap();
    let (c, log) = fit(&mut stage, false);
    let metadata = read_trit_package(&c).unwrap().manifest();
    for case in 0..4 {
        let mut entries = stage::entries(&c);
        match case {
            0 => {
                entries.remove(0);
            }
            1 => entries.push((TritBlobKind::Asset("unbound.json".into()), b"{}".into())),
            2 => {
                entries
                    .iter_mut()
                    .find(|(k, _)| matches!(k, TritBlobKind::AdditiveTensor))
                    .unwrap()
                    .0 = TritBlobKind::DenseTensor(tritium_core::DType::F32)
            }
            3 => {
                entries
                    .iter_mut()
                    .find(|(k, _)| matches!(k, TritBlobKind::AdditiveTensor))
                    .unwrap()
                    .1[0] ^= 1
            }
            _ => unreachable!(),
        }
        let malicious = stage::repack(metadata, &entries);
        assert!(read_trit_package(&malicious).is_ok());
        assert!(
            JoinedSalienceModel::open(&malicious, &stage, &log, limits()).is_err(),
            "case {case}"
        );
    }
}

#[test]
fn strict_joined_codec_rejects_unknown_profile_version_fields_order_and_caps() {
    let (_, recorded, stats) = stage::recorded();
    let bytes = recorded
        .encode_stage_package(&stats, stage::limits())
        .unwrap();
    let transform = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let mut stage = StagedSalienceSource::open(&bytes, &transform, stage::limits()).unwrap();
    let (c, log) = fit(&mut stage, true);
    let metadata = read_trit_package(&c).unwrap().manifest();
    let original: Value = ciborium::de::from_reader(metadata).unwrap();
    for case in 0..4 {
        let mut changed = original.clone();
        let Value::Array(tuple) = &mut changed else {
            panic!("tuple")
        };
        if case == 0 {
            tuple[0] = Value::Text("tritium.salience-conversion.v2".into());
        } else if case == 3 {
            tuple.push(Value::Null);
        } else {
            let Value::Map(fields) = &mut tuple[1] else {
                panic!("manifest")
            };
            if case == 1 {
                fields.push((Value::Text("unexpected".into()), Value::Bool(true)));
                fields.sort_by_cached_key(|(key, _)| CanonicalValue::from(key.clone()));
            } else {
                fields.reverse();
            }
        }
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&changed, &mut bytes).unwrap();
        assert!(
            decode_salience_conversion_manifest(&bytes, 50_000).is_err(),
            "case {case}"
        );
    }
    let mut manifest = JoinedSalienceModel::open(&c, &stage, &log, limits())
        .unwrap()
        .manifest()
        .clone();
    manifest.schema.minor = 1;
    assert!(encode_salience_conversion_manifest(&manifest, 50_000).is_err());
    assert!(decode_salience_conversion_manifest(metadata, metadata.len() - 1).is_err());
    for caps in [
        SalienceConversionLimits {
            max_package_bytes: 1,
            ..limits()
        },
        SalienceConversionLimits {
            max_manifest_bytes: 1,
            ..limits()
        },
        SalienceConversionLimits {
            max_package_bytes: 0,
            ..limits()
        },
        SalienceConversionLimits {
            max_manifest_bytes: 0,
            ..limits()
        },
    ] {
        assert!(JoinedSalienceModel::open(&c, &stage, &log, caps).is_err());
        let mut sink = MemoryTensorSink::default();
        assert!(
            Engine::run_salience(&dense_plan(&stage), &mut stage, &mut sink, recorder(), caps)
                .is_err()
        );
        assert!(sink.artifact().is_none());
    }
}

#[test]
fn foreign_stage_even_with_identical_b_bytes_and_foreign_or_unfinished_fit_logs_reject() {
    let (_, recorded, stats) = stage::recorded();
    let bytes = recorded
        .encode_stage_package(&stats, stage::limits())
        .unwrap();
    let transform = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let mut stage = StagedSalienceSource::open(&bytes, &transform, stage::limits()).unwrap();
    let (c, log) = fit(&mut stage, true);
    let (_, other, _) = stage::recorded();
    let mut foreign = EvidenceRecorder::new(
        "other-transform",
        [
            SALIENCE_TRANSFORM_STARTED_REGISTRATION,
            SALIENCE_TRANSFORM_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap();
    foreign
        .emit(
            "other-span",
            None,
            0,
            &SalienceTransformStarted {
                plan: other.source.observation().plan.clone(),
            },
        )
        .unwrap();
    foreign
        .emit(
            "other-span",
            None,
            1,
            &SalienceTransformCompleted {
                observation: other.source.observation().clone(),
            },
        )
        .unwrap();
    let foreign_claim = foreign.seal().unwrap();
    // Same B contents, different real transform span/root and physical B package.
    let mut manifest = stage.manifest().clone();
    let other_log = verify_jsonl(foreign_claim.jsonl()).unwrap();
    manifest.transform = SalienceTransformView::fold(&other_log).unwrap()[0]
        .reference()
        .clone();
    let metadata =
        tritium_format::trit_salience::encode_salience_stage_manifest(&manifest, 50_000).unwrap();
    let b2 = stage::repack(&metadata, &stage::entries(&bytes));
    let other_stage = StagedSalienceSource::open(&b2, &other_log, stage::limits()).unwrap();
    assert_eq!(stage.source_id(), other_stage.source_id());
    assert_ne!(stage.package_id(), other_stage.package_id());
    assert!(JoinedSalienceModel::open(&c, &other_stage, &log, limits()).is_err());
    let joined = JoinedSalienceModel::open(&c, &stage, &log, limits()).unwrap();
    let mut a = fixture(true);
    assert!(joined.verify_transform(&other_stage, &mut a).is_err());
    assert!(JoinedSalienceModel::open(&c, &stage, &transform, limits()).is_err());
    let mut incomplete = recorder();
    let start: QuantizationStarted = serde_json::from_value(log.events[0].payload.clone()).unwrap();
    incomplete.emit("quantize", None, 0, &start).unwrap();
    let incomplete = incomplete.seal().unwrap();
    assert!(JoinedSalienceModel::open(&c, &stage, incomplete.log(), limits()).is_err());
    let mut other_sink = MemoryTensorSink::default();
    let foreign_recorder = EvidenceRecorder::new(
        "other-conversion",
        [
            DIRECT_LEVELS_FITTED_REGISTRATION,
            QUANTIZATION_STARTED_REGISTRATION,
            QUANTIZATION_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap();
    Engine::run_salience(
        &dense_plan(&stage),
        &mut stage,
        &mut other_sink,
        foreign_recorder,
        limits(),
    )
    .unwrap();
    assert!(
        JoinedSalienceModel::open(&c, &stage, other_sink.artifact().unwrap().1.log(), limits())
            .is_err()
    );
}

#[derive(serde::Serialize)]
struct Unrelated {
    note: &'static str,
}
impl EvidenceEvent for Unrelated {
    const SCHEMA_ID: &'static str = "tritium.fixture.unrelated";
    const VERSION: u16 = 1;
}

#[test]
fn valid_resealed_log_with_matching_root_cannot_hide_unknown_or_unconsumed_events() {
    let (_, recorded, stats) = stage::recorded();
    let bytes = recorded
        .encode_stage_package(&stats, stage::limits())
        .unwrap();
    let transform = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    let mut stage = StagedSalienceSource::open(&bytes, &transform, stage::limits()).unwrap();
    let (c, log) = fit(&mut stage, true);
    let original = JoinedSalienceModel::open(&c, &stage, &log, limits())
        .unwrap()
        .manifest()
        .clone();
    for unknown in [false, true] {
        let mut extra = EvidenceRecorder::new(
            &log.events[0].run,
            [
                QUANTIZATION_STARTED_REGISTRATION,
                QUANTIZATION_COMPLETED_REGISTRATION,
                EventRegistration {
                    schema: Unrelated::SCHEMA_ID,
                    version: 1,
                },
            ],
        )
        .unwrap();
        let start: QuantizationStarted =
            serde_json::from_value(log.events[0].payload.clone()).unwrap();
        let completed: QuantizationCompleted =
            serde_json::from_value(log.events[1].payload.clone()).unwrap();
        extra.emit("quantize", None, 0, &start).unwrap();
        extra.emit("quantize", None, 1, &completed).unwrap();
        if unknown {
            extra
                .emit(
                    "extra",
                    None,
                    0,
                    &Unrelated {
                        note: "not consumed",
                    },
                )
                .unwrap();
        } else {
            extra.emit("extra", None, 0, &start).unwrap();
        }
        let extra = extra.seal().unwrap();
        let views = QuantizationReceiptView::fold(extra.log()).unwrap();
        assert_eq!(
            views.len(),
            1,
            "ordinary view intentionally exposes only complete conversions"
        );
        let mut carrier = original.clone();
        carrier.conversion = views[0].reference().clone();
        let metadata = encode_salience_conversion_manifest(&carrier, 50_000).unwrap();
        let malicious = stage::repack(&metadata, &stage::entries(&c));
        assert!(read_trit_package(&malicious).is_ok());
        assert!(JoinedSalienceModel::open(&malicious, &stage, extra.log(), limits()).is_err());
    }
}

#[test]
fn actual_calibrated_stage_conversion_reuses_b_cache_and_rejects_a_declarations() {
    let (_, recorded, stats) = stage::recorded();
    let bytes = recorded
        .encode_stage_package(&stats, stage::limits())
        .unwrap();
    let transform = verify_jsonl(recorded.evidence.jsonl()).unwrap();
    drop(recorded);
    let mut stage = StagedSalienceSource::open(&bytes, &transform, stage::limits()).unwrap();
    let b = stage.source_id();
    let tensor = stage
        .inventory()
        .unwrap()
        .into_iter()
        .find(|t| t.name == "q")
        .unwrap();
    let mut builder = CalibrationCacheBuilder::new(
        CalibrationCacheRequest {
            tensor,
            model: b,
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
    let cache = builder.finish().unwrap();
    let dense = dense_plan(&stage);
    let request = CalibratedQuantPlan {
        architecture: dense.architecture,
        original_source: b,
        tensors: dense
            .tensors
            .into_iter()
            .map(|t| CalibratedQuantTensorPlan {
                action: if t.source.name == "q" {
                    CalibratedQuantAction::CalibratedFit {
                        plan: Box::new(calibrated(&cache)),
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
    };
    let rec = || {
        EvidenceRecorder::new(
            "joined-calibrated",
            [
                CALIBRATED_LEVELS_FITTED_REGISTRATION,
                CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
                CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap()
    };
    let mut caches = InMemoryCalibrationSource::new([cache]).unwrap();
    let mut wrong = request.clone();
    wrong.original_source = stage.manifest().observation.original_source;
    let mut sink = MemoryTensorSink::default();
    assert!(
        Engine::run_calibrated_salience(
            &wrong,
            &mut stage,
            &mut caches,
            &mut sink,
            rec(),
            limits()
        )
        .is_err()
    );
    assert!(sink.artifact().is_none());
    let result = Engine::run_calibrated_salience(
        &request,
        &mut stage,
        &mut caches,
        &mut sink,
        rec(),
        limits(),
    )
    .unwrap();
    let (c, seal) = sink.artifact().unwrap();
    let log = verify_jsonl(seal.jsonl()).unwrap();
    let joined = JoinedSalienceModel::open(c, &stage, &log, limits()).unwrap();
    assert_eq!(result.source_id, b);
    assert_eq!(
        &joined.manifest().conversion,
        CalibratedQuantizationReceiptView::fold(&log).unwrap()[0].reference()
    );
    let mut a = fixture(true);
    joined.verify_transform(&stage, &mut a).unwrap();
    assert!(read_trit_model(c).is_err());
}
