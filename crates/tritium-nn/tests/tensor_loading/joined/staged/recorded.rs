//! Actual controller/sealed log/cold source replay, including valid-chain lies.
use super::*;
use tritium_evidence::{
    CALIBRATION_CAPTURE_COMPLETED_REGISTRATION, CALIBRATION_CAPTURE_STARTED_REGISTRATION,
    CalibrationCaptureView, EvidenceEvent, EvidenceRecorder, SealedEvidence,
};
use tritium_nn::calibrate::replay_staged_capture;
use tritium_schema::{CalibrationCaptureCompleted, CalibrationCaptureStarted, FitInputDigest};

pub(super) fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "actual-capture",
        [
            CALIBRATION_CAPTURE_STARTED_REGISTRATION,
            CALIBRATION_CAPTURE_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap()
}

pub(super) fn rehashed(
    start: &CalibrationCaptureStarted,
    output: &serde_json::Value,
) -> SealedEvidence {
    #[derive(serde::Serialize)]
    struct Raw(serde_json::Value);
    impl EvidenceEvent for Raw {
        const SCHEMA_ID: &'static str = CalibrationCaptureCompleted::SCHEMA_ID;
        const VERSION: u16 = 1;
    }
    let mut recorder = recorder();
    recorder.emit("capture", None, 0, start).unwrap();
    recorder
        .emit("capture", None, 1, &Raw(output.clone()))
        .unwrap();
    recorder.seal().unwrap()
}

#[test]
fn recorded_token_capture_cold_replays_actual_b_and_rejects_valid_chain_fabrication() {
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        let hf = HfFixture::new("qwen3", true, 6, None);
        let mut input = TritFixture::new(&hf, None);
        if precision != ScalePrecision::F32 {
            input.dense_precision(precision);
        }
        let fixture = produced(&input, false);
        let mut stage =
            StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
        let model = StagedCalibrationModel::new(&mut stage, limits()).unwrap();
        let actual = model
            .capture_next_token_recorded(
                &mut stage,
                &[&[1, 4, 2], &[3, 7]],
                capture_limits(),
                recorder(),
            )
            .unwrap();
        let views = CalibrationCaptureView::fold(actual.evidence.log()).unwrap();
        assert_eq!(views.len(), 1);
        let view = &views[0];
        assert_eq!(view.started().plan.source, stage.source_id());
        assert_eq!(view.started().plan.stage_package, stage.package_id());
        assert_eq!(view.completed().windows, [vec![1, 4, 2], vec![3, 7]]);
        assert_eq!(view.completed().caches.len(), model.arch().n_layers * 7 + 1);
        let receiver = view.started().plan.limits;
        let ids: Vec<_> = actual.caches.iter().map(|c| c.id()).collect();
        drop(model);
        drop(stage);
        let log = tritium_evidence::verify_jsonl(actual.evidence.jsonl()).unwrap();
        let mut stage =
            StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
        let replay = replay_staged_capture(&log, &mut stage, receiver).unwrap();
        assert_eq!(replay.iter().map(|c| c.id()).collect::<Vec<_>>(), ids);
        for (original, replayed) in actual.caches.iter().zip(&replay) {
            assert_eq!(original.binding(), replayed.binding());
            assert_eq!(
                original.encode_package(1_000_000).unwrap(),
                replayed.encode_package(1_000_000).unwrap()
            );
        }
        for bad in [
            tritium_schema::CalibrationCaptureLimits {
                max_dense_bytes: receiver.max_dense_bytes - 1,
                ..receiver
            },
            tritium_schema::CalibrationCaptureLimits {
                max_config_bytes: receiver.max_config_bytes - 1,
                ..receiver
            },
            tritium_schema::CalibrationCaptureLimits {
                max_window_tokens: receiver.max_window_tokens - 1,
                ..receiver
            },
            tritium_schema::CalibrationCaptureLimits {
                max_total_payload_bytes: receiver.max_total_payload_bytes - 1,
                ..receiver
            },
            tritium_schema::CalibrationCaptureLimits {
                per_cache: tritium_schema::CalibrationCacheLimits {
                    max_payload_bytes: receiver.per_cache.max_payload_bytes - 1,
                    ..receiver.per_cache
                },
                ..receiver
            },
        ] {
            assert!(replay_staged_capture(&log, &mut stage, bad).is_err());
        }
        let mut output = view.completed().clone();
        output.caches[0].source.cache = FitInputDigest([77; 32]);
        let forged = rehashed(view.started(), &serde_json::to_value(&output).unwrap());
        // Complete structural context is valid; the forged gradient/cache claim
        // is caught only by actual B forward/backward/cache reconstruction.
        assert_eq!(CalibrationCaptureView::fold(forged.log()).unwrap().len(), 1);
        assert!(replay_staged_capture(forged.log(), &mut stage, receiver).is_err());
        for malformed in [
            {
                let mut v = serde_json::to_value(view.completed()).unwrap();
                v["unknown"] = true.into();
                v
            },
            {
                let mut c = view.completed().clone();
                c.caches.pop();
                serde_json::to_value(c).unwrap()
            },
            {
                let mut c = view.completed().clone();
                c.caches.swap(0, 1);
                serde_json::to_value(c).unwrap()
            },
            {
                let mut c = view.completed().clone();
                c.windows[0][2] = 3;
                serde_json::to_value(c).unwrap()
            },
        ] {
            let forged = rehashed(view.started(), &malformed);
            assert!(CalibrationCaptureView::fold(forged.log()).is_err());
        }
        let mut foreign_input = TritFixture::new(&hf, None);
        foreign_input.asset("tokenizer.json", b"foreign tokenizer bytes".to_vec());
        let foreign = produced(&foreign_input, false);
        let mut foreign_stage =
            StagedSalienceSource::open(&foreign.b, &foreign.transform, stage_limits()).unwrap();
        assert_ne!(
            foreign_stage.package_id(),
            view.started().plan.stage_package
        );
        assert!(replay_staged_capture(&log, &mut foreign_stage, receiver).is_err());
    }
}

#[test]
fn recorded_capture_requires_registration_and_complete_root_lifecycle() {
    let hf = HfFixture::new("llama", true, 4, None);
    let fixture = produced(&TritFixture::new(&hf, None), false);
    let mut stage =
        StagedSalienceSource::open(&fixture.b, &fixture.transform, stage_limits()).unwrap();
    let model = StagedCalibrationModel::new(&mut stage, limits()).unwrap();
    let actual = model
        .capture_next_token_recorded(&mut stage, &[&[1, 2]], capture_limits(), recorder())
        .unwrap();
    let view = CalibrationCaptureView::fold(actual.evidence.log())
        .unwrap()
        .remove(0);
    let missing =
        EvidenceRecorder::new("missing", [CALIBRATION_CAPTURE_STARTED_REGISTRATION]).unwrap();
    assert!(
        model
            .capture_next_token_recorded(&mut stage, &[&[1, 2]], capture_limits(), missing)
            .is_err()
    );
    let mut incomplete = recorder();
    incomplete.emit("capture", None, 0, view.started()).unwrap();
    let incomplete = incomplete.seal().unwrap();
    assert!(
        CalibrationCaptureView::fold(incomplete.log())
            .unwrap()
            .is_empty()
    );
    assert!(
        replay_staged_capture(incomplete.log(), &mut stage, view.started().plan.limits).is_err()
    );
    let mut orphan = recorder();
    orphan.emit("capture", None, 0, view.completed()).unwrap();
    assert!(CalibrationCaptureView::fold(orphan.seal().unwrap().log()).is_err());
    assert!(
        model
            .capture_next_token_recorded(&mut stage, &[&[1]], capture_limits(), recorder())
            .is_err()
    );
}
