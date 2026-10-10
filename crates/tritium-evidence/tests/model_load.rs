use tritium_evidence::{
    EvidenceError, EvidenceEvent, EvidenceRecorder, MODEL_LOADED_REGISTRATION,
    ModelLoadReceiptView, ModelLoaded, verify_jsonl_with_registry,
};
use tritium_schema::{LoadedSourceIdentity, ModelId, PackageId};

fn observation() -> ModelLoaded {
    ModelLoaded {
        source: LoadedSourceIdentity::Trit {
            package_id: PackageId::from_bytes([1; 32]),
            model_id: ModelId::from_bytes([2; 32]),
            level: "standard".into(),
        },
        architecture: "llama".into(),
        backend: "controlled-test".into(),
        physical_device: "unknown-test-device".into(),
        vocab_size: 10,
        layer_count: 2,
    }
}

#[test]
fn recorded_load_view_matches_verified_replay_without_a_verdict() {
    let mut recorder = EvidenceRecorder::new("load-test", [MODEL_LOADED_REGISTRATION]).unwrap();
    let view = recorder
        .emit_model_loaded("load", None, 0, &observation())
        .unwrap();
    assert_eq!(view.observation(), &observation());
    assert_eq!(view.envelope().seq, 0);
    assert_eq!(view.envelope().run, "load-test");
    assert!(view.envelope().payload.get("verdict").is_none());
    let log =
        verify_jsonl_with_registry(&recorder.to_jsonl().unwrap(), [MODEL_LOADED_REGISTRATION])
            .unwrap();
    assert_eq!(ModelLoadReceiptView::fold(&log).unwrap(), [view]);
}

#[test]
fn load_view_refuses_modified_verified_log_and_root() {
    let mut recorder = EvidenceRecorder::new("load-test", [MODEL_LOADED_REGISTRATION]).unwrap();
    recorder
        .emit_model_loaded("load", None, 0, &observation())
        .unwrap();
    let log =
        verify_jsonl_with_registry(&recorder.to_jsonl().unwrap(), [MODEL_LOADED_REGISTRATION])
            .unwrap();
    let mut modified = log.clone();
    modified.root_digest = "0".repeat(64);
    assert_eq!(
        ModelLoadReceiptView::fold(&modified),
        Err(EvidenceError::RootMismatch)
    );
    let mut modified = log;
    modified.events[0].payload["architecture"] = "qwen3".into();
    assert!(matches!(
        ModelLoadReceiptView::fold(&modified),
        Err(EvidenceError::DigestMismatch { .. })
    ));
}

#[test]
fn load_view_strictly_decodes_registered_payloads() {
    #[derive(serde::Serialize)]
    struct Hostile {
        #[serde(flatten)]
        event: ModelLoaded,
        unauthorized_verdict: bool,
    }
    impl EvidenceEvent for Hostile {
        const SCHEMA_ID: &'static str = ModelLoaded::SCHEMA_ID;
        const VERSION: u16 = 1;
    }
    let mut recorder = EvidenceRecorder::new("hostile", [MODEL_LOADED_REGISTRATION]).unwrap();
    recorder
        .emit(
            "load",
            None,
            0,
            &Hostile {
                event: observation(),
                unauthorized_verdict: true,
            },
        )
        .unwrap();
    let log =
        verify_jsonl_with_registry(&recorder.to_jsonl().unwrap(), [MODEL_LOADED_REGISTRATION])
            .unwrap();
    assert!(matches!(
        ModelLoadReceiptView::fold(&log),
        Err(EvidenceError::InvalidPayload { .. })
    ));
}

#[test]
fn failed_recording_does_not_advance_the_load_event_chain() {
    let mut recorder = EvidenceRecorder::new("unregistered", []).unwrap();
    assert!(matches!(
        recorder.emit_model_loaded("load", None, 0, &observation()),
        Err(EvidenceError::UnregisteredSchema(_))
    ));
    assert!(recorder.events().is_empty());
    let mut recorder = EvidenceRecorder::new("registered", [MODEL_LOADED_REGISTRATION]).unwrap();
    assert!(
        recorder
            .emit_model_loaded("load", None, 1 << 53, &observation())
            .is_err()
    );
    assert_eq!(
        recorder
            .emit_model_loaded("load", None, 0, &observation())
            .unwrap()
            .envelope()
            .seq,
        0
    );
}

#[test]
fn load_view_rejects_future_versions_and_wrong_nested_identity_fields() {
    #[derive(serde::Serialize)]
    #[serde(transparent)]
    struct Future(ModelLoaded);
    impl EvidenceEvent for Future {
        const SCHEMA_ID: &'static str = ModelLoaded::SCHEMA_ID;
        const VERSION: u16 = 2;
    }
    let registry = tritium_evidence::EventRegistration {
        schema: ModelLoaded::SCHEMA_ID,
        version: 2,
    };
    let mut recorder = EvidenceRecorder::new("future", [registry]).unwrap();
    recorder
        .emit("load", None, 0, &Future(observation()))
        .unwrap();
    let log = verify_jsonl_with_registry(&recorder.to_jsonl().unwrap(), [registry]).unwrap();
    assert!(matches!(
        ModelLoadReceiptView::fold(&log),
        Err(EvidenceError::UnregisteredSchema(_))
    ));

    #[derive(serde::Serialize)]
    #[serde(transparent)]
    struct Wrong(serde_json::Value);
    impl EvidenceEvent for Wrong {
        const SCHEMA_ID: &'static str = ModelLoaded::SCHEMA_ID;
        const VERSION: u16 = 1;
    }
    let mut payload = serde_json::to_value(observation()).unwrap();
    payload["source"]["Trit"]["legacy_id"] = "not-a-checked-identity".into();
    let mut recorder = EvidenceRecorder::new("wrong-source", [MODEL_LOADED_REGISTRATION]).unwrap();
    recorder.emit("load", None, 0, &Wrong(payload)).unwrap();
    let log =
        verify_jsonl_with_registry(&recorder.to_jsonl().unwrap(), [MODEL_LOADED_REGISTRATION])
            .unwrap();
    assert!(matches!(
        ModelLoadReceiptView::fold(&log),
        Err(EvidenceError::InvalidPayload { .. })
    ));
}

#[test]
fn upload_only_log_has_no_invented_load_receipt() {
    use tritium_evidence::{TENSOR_UPLOADED_REGISTRATION, TensorUploaded};
    let mut recorder =
        EvidenceRecorder::new("upload-only", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    recorder
        .emit(
            "upload",
            None,
            0,
            &TensorUploaded {
                tensor: "controlled".into(),
                backend: "controlled-test".into(),
                physical_device: "unknown".into(),
                caps: tritium_schema::TensorCaps {
                    execution: tritium_schema::TensorExecution::Native,
                    payload_bytes: 4,
                },
                decoded_source_bytes: 4,
            },
        )
        .unwrap();
    let log = verify_jsonl_with_registry(
        &recorder.to_jsonl().unwrap(),
        [TENSOR_UPLOADED_REGISTRATION],
    )
    .unwrap();
    assert!(ModelLoadReceiptView::fold(&log).unwrap().is_empty());
}
