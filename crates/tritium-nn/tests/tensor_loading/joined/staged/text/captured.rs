//! Actual mandatory capture root crosses the one fitter/loader/runtime seam.
use super::super::recorded::{recorder, rehashed};
use super::*;
use tritium_evidence::CalibrationCaptureView;
use tritium_format::trit_salience_conversion::{
    decode_captured_salience_conversion_manifest, decode_salience_conversion_manifest,
    encode_captured_salience_conversion_manifest,
};
use tritium_format::{TritBlob, read_trit_package, write_trit_package};
use tritium_nn::calibrate::replay_staged_capture;

fn projection_caches(caches: Vec<CalibrationCache>) -> Vec<CalibrationCache> {
    caches
        .into_iter()
        .filter(|c| c.binding().request.tensor.name != "model.embed_tokens.weight")
        .collect()
}

#[test]
fn actual_captured_conversion_cold_opens_and_runs_without_stage_or_capture_text() {
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
        let captured = model
            .capture_next_token_text_recorded(
                &mut stage,
                &["hello partner hello", "partner hello"],
                text_limits(),
                capture_limits(),
                recorder(),
            )
            .unwrap();
        let capture_view = CalibrationCaptureView::fold(captured.evidence.log())
            .unwrap()
            .remove(0);
        // Actual cold replay, not mere structural capture admission.
        drop(model);
        let caches = projection_caches(
            replay_staged_capture(
                captured.evidence.log(),
                &mut stage,
                capture_view.started().plan.limits,
            )
            .unwrap(),
        );
        let plan = calibrated_plan(&stage, &caches);
        let mut provider = InMemoryCalibrationSource::new(caches).unwrap();
        let mut sink = MemoryTensorSink::default();
        let result = Engine::run_captured_salience(
            &plan,
            &mut stage,
            captured.evidence.log(),
            &mut provider,
            &mut sink,
            calibrated_recorder(),
            conversion_limits(),
        )
        .unwrap();
        let (c, seal) = sink.artifact().unwrap();
        let c = c.to_vec();
        let fit = verify_jsonl(seal.jsonl()).unwrap();
        let capture = verify_jsonl(captured.evidence.jsonl()).unwrap();
        drop(captured);
        drop(stage);
        drop(sink);
        let mut stage =
            StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
        let joined =
            JoinedSalienceModel::open_captured(&c, &stage, &capture, &fit, conversion_limits())
                .unwrap();
        assert_eq!(joined.capture_reference(), Some(capture_view.reference()));
        assert_eq!(joined.manifest().derived_source, result.source_id);
        assert!(JoinedSalienceModel::open(&c, &stage, &fit, conversion_limits()).is_err());
        assert!(tritium_format::trit_model::read_trit_model(&c).is_err());
        let package = read_trit_package(&c).unwrap();
        assert!(decode_salience_conversion_manifest(package.manifest(), 256_000).is_err());
        let carrier =
            decode_captured_salience_conversion_manifest(package.manifest(), 256_000).unwrap();
        assert_eq!(&carrier.capture, capture_view.reference());
        assert_eq!(
            encode_captured_salience_conversion_manifest(&carrier, 256_000).unwrap(),
            package.manifest()
        );
        assert!(
            decode_captured_salience_conversion_manifest(
                package.manifest(),
                package.manifest().len() - 1
            )
            .is_err()
        );
        assert!(
            JoinedSalienceModel::open_captured(
                &c,
                &stage,
                &capture,
                &fit,
                SalienceConversionLimits {
                    max_package_bytes: c.len() as u64 - 1,
                    ..conversion_limits()
                }
            )
            .is_err()
        );
        assert!(
            JoinedSalienceModel::open_captured(
                &fixture.c,
                &stage,
                &capture,
                &fixture.fit,
                conversion_limits()
            )
            .is_err()
        );
        let mut changed = carrier;
        changed.capture.root = [17; 32];
        let manifest = encode_captured_salience_conversion_manifest(&changed, 256_000).unwrap();
        let blobs: Vec<_> = package
            .blobs()
            .iter()
            .map(|b| TritBlob {
                kind: b.kind.clone(),
                bytes: package.blob(b).unwrap(),
            })
            .collect();
        let rehashed = write_trit_package(&manifest, &blobs).unwrap();
        assert!(read_trit_package(&rehashed).is_ok());
        assert!(
            JoinedSalienceModel::open_captured(
                &rehashed,
                &stage,
                &capture,
                &fit,
                conversion_limits()
            )
            .is_err()
        );
        let foreign = model_capture(&mut stage, &["partner hello partner", "hello partner"]);
        assert!(
            JoinedSalienceModel::open_captured(
                &c,
                &stage,
                foreign.evidence.log(),
                &fit,
                conversion_limits()
            )
            .is_err()
        );
        let foreign_stage_fixture = produced_with_alpha(&input, false, 0.0);
        let foreign_stage = StagedSalienceSource::open(
            &foreign_stage_fixture.b,
            &foreign_stage_fixture.transform,
            stage_limits(),
        )
        .unwrap();
        assert!(
            JoinedSalienceModel::open_captured(
                &c,
                &foreign_stage,
                &capture,
                &fit,
                conversion_limits()
            )
            .is_err()
        );
        drop(stage);
        drop(capture);
        drop(fit);
        drop(capture_view);
        drop(foreign);
        // C proof-bearing view retains only C metadata/reference. Runtime uses
        // the existing selector and may outlive B, logs and their private text.
        for reference in [false, true] {
            let backend: Box<dyn TernaryBackend> = if reference {
                Box::new(tritium_testkit::ReferenceBackend::new())
            } else {
                Box::new(tritium_cpu::CpuBackend::new())
            };
            let mut oracle = oracle_fixture(&joined, &c).dense_oracle();
            let budget = TensorPayloadBudget::new(100_000);
            let mut uploads =
                EvidenceRecorder::new("captured-runtime", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
            let mut runner = ModelRunner::from_source_admitted(
                tritium_nn::ModelSource::JoinedSalience {
                    model: &joined,
                    level: "standard",
                },
                backend,
                TensorUploadPolicy::default(),
                &budget,
                TensorUploadContext {
                    recorder: &mut uploads,
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
    }
}

fn model_capture(
    stage: &mut StagedSalienceSource<'_>,
    texts: &[&str],
) -> tritium_nn::calibrate::RecordedStagedCapture {
    StagedCalibrationModel::new(stage, limits())
        .unwrap()
        .capture_next_token_text_recorded(stage, texts, text_limits(), capture_limits(), recorder())
        .unwrap()
}

#[test]
fn captured_conversion_rejects_missing_foreign_or_uncalibrated_evidence_before_publication() {
    let hf = HfFixture::new("llama", true, 4, None);
    let fixture = produced(&text_input(&hf), false);
    let mut stage =
        StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
    let actual = model_capture(&mut stage, &["hello partner hello", "partner hello"]);
    let view = CalibrationCaptureView::fold(actual.evidence.log())
        .unwrap()
        .remove(0);
    let receiver = view.started().plan.limits;
    let caches = projection_caches(
        replay_staged_capture(actual.evidence.log(), &mut stage, receiver).unwrap(),
    );
    let plan = calibrated_plan(&stage, &caches);
    let foreign = model_capture(&mut stage, &["partner hello partner", "hello partner"]);
    let mut incomplete = recorder();
    incomplete.emit("capture", None, 0, view.started()).unwrap();
    let incomplete = incomplete.seal().unwrap();
    let mut missing = view.started().clone();
    let mut missing_output = view.completed().clone();
    missing.plan.tensors.remove(0);
    missing_output.caches.remove(0);
    let missing = rehashed(&missing, &serde_json::to_value(missing_output).unwrap());
    assert_eq!(
        CalibrationCaptureView::fold(missing.log()).unwrap().len(),
        1
    );
    for log in [foreign.evidence.log(), incomplete.log(), missing.log()] {
        let mut provider = InMemoryCalibrationSource::new(projection_caches(
            replay_staged_capture(actual.evidence.log(), &mut stage, receiver).unwrap(),
        ))
        .unwrap();
        let mut sink = MemoryTensorSink::default();
        assert!(
            Engine::run_captured_salience(
                &plan,
                &mut stage,
                log,
                &mut provider,
                &mut sink,
                calibrated_recorder(),
                conversion_limits()
            )
            .is_err()
        );
        assert!(sink.artifact().is_none());
    }
    for plain in [false, true] {
        let mut bad = plan.clone();
        let action = bad
            .tensors
            .iter_mut()
            .find(|t| matches!(t.action, CalibratedQuantAction::CalibratedFit { .. }))
            .unwrap();
        let CalibratedQuantAction::CalibratedFit { plan: fit } = &mut action.action else {
            panic!()
        };
        if plain {
            action.action = CalibratedQuantAction::Fit {
                plan: Box::new(fit.direct.clone()),
                metric: DirectFitMetric::Identity,
            };
        } else {
            fit.cache.source.cache = tritium_schema::FitInputDigest([22; 32]);
        }
        bad.validate().unwrap();
        let mut provider = InMemoryCalibrationSource::new(projection_caches(
            replay_staged_capture(actual.evidence.log(), &mut stage, receiver).unwrap(),
        ))
        .unwrap();
        let mut sink = MemoryTensorSink::default();
        assert!(
            Engine::run_captured_salience(
                &bad,
                &mut stage,
                actual.evidence.log(),
                &mut provider,
                &mut sink,
                calibrated_recorder(),
                conversion_limits()
            )
            .is_err()
        );
        assert!(sink.artifact().is_none());
    }
}
