//! Synthetic structural lifecycle tests; not executed salience or model qualification.
use tritium_evidence::*;
use tritium_schema::*;

fn fixture() -> (
    OriginalSalienceCapturePlan,
    OriginalSalienceCaptureCompleted,
) {
    let digest = |bytes: &[u8]| BlobId::from_bytes(*blake3::hash(bytes).as_bytes());
    let tensors = [
        ("down", vec![2, 3]),
        ("norm", vec![2]),
        ("q", vec![2, 2]),
        ("up", vec![3, 2]),
    ]
    .into_iter()
    .map(|(name, shape)| QuantSourceBinding {
        source: QuantSourceTensor {
            name: name.into(),
            shape,
            precision: ScalePrecision::F32,
        },
        content: digest(name.as_bytes()),
    })
    .collect::<Vec<_>>();
    let assets = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        .into_iter()
        .map(|name| NamedAssetDigest {
            name: name.into(),
            content: digest(name.as_bytes()),
        })
        .collect::<Vec<_>>();
    let plan = OriginalSalienceCapturePlan {
        architecture: "fixture".into(),
        source: original_source_id("fixture", &tensors, &assets).unwrap(),
        tensors,
        assets,
        profile: OriginalSalienceCaptureProfile::AllTokenF64SecondMomentV1,
        pairs: vec![
            OriginalSalienceCapturePair {
                name: "attn".into(),
                projections: vec!["q".into()],
                inverse: "norm".into(),
                inverse_axis: SalienceInverseAxis::NormChannels,
                channels: 2,
            },
            OriginalSalienceCapturePair {
                name: "down".into(),
                projections: vec!["down".into()],
                inverse: "up".into(),
                inverse_axis: SalienceInverseAxis::LinearOutputRows,
                channels: 3,
            },
        ],
        input: OriginalSalienceCaptureInput::Tokens {
            windows: vec![vec![0], vec![1, 2]],
        },
        limits: OriginalSalienceCaptureLimits {
            max_source_bytes: 128,
            max_asset_bytes: 128,
            max_config_bytes: 128,
            max_dense_bytes: 128,
            max_window_tokens: 2,
            max_windows: 2,
            max_total_tokens: 3,
            max_moment_bytes: 40,
            max_text_bytes: 0,
            max_tokenizer_config_bytes: 0,
        },
    };
    let completed = OriginalSalienceCaptureCompleted {
        windows: vec![vec![0], vec![1, 2]],
        outputs: plan
            .pairs
            .iter()
            .map(|pair| OriginalSalienceCaptureOutput {
                operation: pair.name.clone(),
                moments: SalienceMomentBinding {
                    content: digest(pair.name.as_bytes()),
                    channels: pair.channels,
                    rows: 3,
                },
            })
            .collect(),
    };
    (plan, completed)
}
fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new(
        "original-salience-structural",
        [
            ORIGINAL_SALIENCE_CAPTURE_STARTED_REGISTRATION,
            ORIGINAL_SALIENCE_CAPTURE_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap()
}
fn record(
    plan: OriginalSalienceCapturePlan,
    completed: OriginalSalienceCaptureCompleted,
) -> SealedEvidence {
    let mut r = recorder();
    r.emit(
        "salience",
        None,
        0,
        &OriginalSalienceCaptureStarted { plan },
    )
    .unwrap();
    r.emit("salience", None, 1, &completed).unwrap();
    r.seal().unwrap()
}

#[test]
fn original_salience_all_tokens_and_exact_canonical_lifecycle() {
    let (plan, completed) = fixture();
    assert!(plan.validate().is_ok());
    assert!(completed.agrees_with(&plan));
    let log = record(plan.clone(), completed.clone());
    let cold = verify_jsonl(log.jsonl()).unwrap();
    let views = OriginalSalienceCaptureView::fold(&cold).unwrap();
    assert_eq!(views.len(), 1);
    assert_eq!(views[0].started().plan, plan);
    assert_eq!(*views[0].completed(), completed);
    assert_eq!(
        blake3::Hash::from_bytes(views[0].reference().root)
            .to_hex()
            .as_str(),
        cold.root_digest
    );
    assert!(CalibrationCaptureView::fold(&cold).unwrap().is_empty());
    assert!(SalienceTransformView::fold(&cold).unwrap().is_empty());
    assert_eq!(record(plan, completed).jsonl(), log.jsonl());
    if let Some(path) = std::env::var_os("TRITIUM_ORIGINAL_SALIENCE_STRUCTURAL_OUTPUT") {
        std::fs::write(path, log.jsonl()).unwrap();
    }
}

#[test]
fn original_salience_rejects_counts_order_caps_and_final_token_omission() {
    let (plan, completed) = fixture();
    let mut wrong = Vec::new();
    let mut c = completed.clone();
    c.outputs.reverse();
    wrong.push(c);
    let mut c = completed.clone();
    c.outputs[0].moments.rows = 2;
    wrong.push(c);
    let mut c = completed.clone();
    c.outputs[0].moments.channels = 3;
    wrong.push(c);
    let mut c = completed.clone();
    c.outputs[0].operation = "foreign".into();
    wrong.push(c);
    let mut c = completed.clone();
    c.outputs[0].moments.content = BlobId::from_bytes([0; 32]);
    wrong.push(c);
    let mut c = completed.clone();
    c.windows[1].pop();
    wrong.push(c);
    let mut c = completed.clone();
    c.windows.reverse();
    wrong.push(c);
    let mut c = completed.clone();
    c.windows[0].clear();
    wrong.push(c);
    let mut c = completed.clone();
    c.outputs.pop();
    wrong.push(c);
    let mut c = completed.clone();
    c.outputs.push(c.outputs[0].clone());
    wrong.push(c);
    for c in wrong {
        assert!(!c.agrees_with(&plan));
        assert!(OriginalSalienceCaptureView::fold(record(plan.clone(), c).log()).is_err());
    }
    for (which, cap) in [(0, 71), (1, 71), (2, 1), (3, 1), (4, 2), (5, 39)] {
        let mut p = plan.clone();
        match which {
            0 => p.limits.max_source_bytes = cap,
            1 => p.limits.max_dense_bytes = cap,
            2 => p.limits.max_window_tokens = cap,
            3 => p.limits.max_windows = cap,
            4 => p.limits.max_total_tokens = cap,
            _ => p.limits.max_moment_bytes = cap,
        }
        assert!(p.validate().is_err());
        assert!(OriginalSalienceCaptureView::fold(record(p, completed.clone()).log()).is_err());
    }
}

#[test]
fn original_salience_complete_source_and_pair_geometry_are_not_asserted_ids() {
    let (plan, completed) = fixture();
    for which in 0..10 {
        let mut p = plan.clone();
        match which {
            0 => p.source = OriginalSourceId::from_bytes([7; 32]),
            1 => p.tensors[0].content = BlobId::from_bytes([8; 32]),
            2 => p.assets[0].content = BlobId::from_bytes([9; 32]),
            3 => p.tensors.reverse(),
            4 => p.assets.push(p.assets[0].clone()),
            5 => p.pairs[0].projections.push("q".into()),
            6 => p.pairs[0].inverse = "q".into(),
            7 => p.pairs[1].inverse_axis = SalienceInverseAxis::NormChannels,
            8 => p.pairs[1].channels = (1u64 << 53) - 1,
            _ => p.pairs[0].projections[0] = "foreign".into(),
        }
        assert!(OriginalSalienceCaptureView::fold(record(p, completed.clone()).log()).is_err());
    }
    let mut overflowing = plan.clone();
    overflowing.pairs[1].channels = u64::MAX;
    assert!(overflowing.validate().is_err());
    // Rehashed moment claims are structurally coherent but unproven without actual replay.
    let mut lie = completed.clone();
    lie.outputs[0].moments.content = BlobId::from_bytes([7; 32]);
    assert_eq!(
        OriginalSalienceCaptureView::fold(record(plan, lie).log())
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn original_salience_text_preserves_raw_context_without_claiming_encoding_truth() {
    let (mut plan, completed) = fixture();
    plan.input = OriginalSalienceCaptureInput::RawText {
        windows: vec!["a".into(), "b c".into()],
    };
    plan.limits.max_text_bytes = 4;
    plan.limits.max_tokenizer_config_bytes = 128;
    assert!(completed.agrees_with(&plan));
    let original = record(plan.clone(), completed.clone());
    if let OriginalSalienceCaptureInput::RawText { windows } = &mut plan.input {
        windows[0] = "A".into();
    }
    let changed = record(plan.clone(), completed.clone());
    assert_ne!(original.log().root_digest, changed.log().root_digest);
    assert_eq!(
        OriginalSalienceCaptureView::fold(changed.log())
            .unwrap()
            .len(),
        1
    );
    plan.limits.max_text_bytes = 3;
    assert!(plan.validate().is_err());
    plan.limits.max_text_bytes = 4;
    plan.assets.pop();
    assert!(plan.validate().is_err());
}

#[test]
fn original_salience_every_receiving_limit_is_explicit() {
    let (plan, _) = fixture();
    let mut recorded = plan.limits;
    recorded.max_text_bytes = 4;
    recorded.max_tokenizer_config_bytes = 128;
    assert!(recorded.admitted_by(&recorded));
    for which in 0..10 {
        let mut receiver = recorded;
        match which {
            0 => receiver.max_source_bytes -= 1,
            1 => receiver.max_asset_bytes -= 1,
            2 => receiver.max_config_bytes -= 1,
            3 => receiver.max_dense_bytes -= 1,
            4 => receiver.max_window_tokens -= 1,
            5 => receiver.max_windows -= 1,
            6 => receiver.max_total_tokens -= 1,
            7 => receiver.max_moment_bytes -= 1,
            8 => receiver.max_text_bytes -= 1,
            _ => receiver.max_tokenizer_config_bytes -= 1,
        }
        assert!(!recorded.admitted_by(&receiver));
    }
    for which in 0..8 {
        let mut p = plan.clone();
        match which {
            0 => p.limits.max_source_bytes = 0,
            1 => p.limits.max_asset_bytes = 0,
            2 => p.limits.max_config_bytes = 0,
            3 => p.limits.max_dense_bytes = 0,
            4 => p.limits.max_window_tokens = 0,
            5 => p.limits.max_windows = 0,
            6 => p.limits.max_total_tokens = 0,
            _ => p.limits.max_moment_bytes = 0,
        }
        assert!(p.validate().is_err());
    }
}

#[derive(serde::Serialize)]
#[serde(transparent)]
struct RawStart(serde_json::Value);
impl EvidenceEvent for RawStart {
    const SCHEMA_ID: &'static str = OriginalSalienceCaptureStarted::SCHEMA_ID;
    const VERSION: u16 = 1;
}
#[derive(serde::Serialize)]
#[serde(transparent)]
struct FutureStart(serde_json::Value);
impl EvidenceEvent for FutureStart {
    const SCHEMA_ID: &'static str = OriginalSalienceCaptureStarted::SCHEMA_ID;
    const VERSION: u16 = 2;
}
#[derive(serde::Serialize)]
#[serde(transparent)]
struct RawCompletion(serde_json::Value);
impl EvidenceEvent for RawCompletion {
    const SCHEMA_ID: &'static str = OriginalSalienceCaptureCompleted::SCHEMA_ID;
    const VERSION: u16 = 1;
}
#[derive(serde::Serialize)]
#[serde(transparent)]
struct FutureCompletion(serde_json::Value);
impl EvidenceEvent for FutureCompletion {
    const SCHEMA_ID: &'static str = OriginalSalienceCaptureCompleted::SCHEMA_ID;
    const VERSION: u16 = 2;
}

#[test]
fn original_salience_strict_schema_versions_and_nested_fields() {
    let (plan, completed) = fixture();
    let base = serde_json::to_value(OriginalSalienceCaptureStarted { plan }).unwrap();
    for path in [
        vec![],
        vec!["plan"],
        vec!["plan", "limits"],
        vec!["plan", "input"],
    ] {
        let mut payload = base.clone();
        let mut at = &mut payload;
        for key in path {
            at = &mut at[key];
        }
        at.as_object_mut()
            .unwrap()
            .insert("unknown".into(), true.into());
        let mut r = recorder();
        r.emit("salience", None, 0, &RawStart(payload)).unwrap();
        r.emit("salience", None, 1, &completed).unwrap();
        assert!(OriginalSalienceCaptureView::fold(r.seal().unwrap().log()).is_err());
    }
    let mut r = EvidenceRecorder::new(
        "future",
        [
            EventRegistration {
                schema: FutureStart::SCHEMA_ID,
                version: FutureStart::VERSION,
            },
            ORIGINAL_SALIENCE_CAPTURE_COMPLETED_REGISTRATION,
        ],
    )
    .unwrap();
    r.emit("salience", None, 0, &FutureStart(base)).unwrap();
    r.emit("salience", None, 1, &completed).unwrap();
    assert!(OriginalSalienceCaptureView::fold(r.seal().unwrap().log()).is_err());
}

#[test]
fn original_salience_completion_fields_and_versions_are_strict() {
    let (plan, completed) = fixture();
    let base = serde_json::to_value(&completed).unwrap();
    for which in 0..3 {
        let mut payload = base.clone();
        let at = match which {
            0 => &mut payload,
            1 => &mut payload["outputs"][0],
            _ => &mut payload["outputs"][0]["moments"],
        };
        at.as_object_mut()
            .unwrap()
            .insert("unknown".into(), true.into());
        let mut r = recorder();
        r.emit(
            "salience",
            None,
            0,
            &OriginalSalienceCaptureStarted { plan: plan.clone() },
        )
        .unwrap();
        r.emit("salience", None, 1, &RawCompletion(payload))
            .unwrap();
        assert!(OriginalSalienceCaptureView::fold(r.seal().unwrap().log()).is_err());
    }
    let mut r = EvidenceRecorder::new(
        "future-completion",
        [
            ORIGINAL_SALIENCE_CAPTURE_STARTED_REGISTRATION,
            EventRegistration {
                schema: FutureCompletion::SCHEMA_ID,
                version: FutureCompletion::VERSION,
            },
        ],
    )
    .unwrap();
    r.emit(
        "salience",
        None,
        0,
        &OriginalSalienceCaptureStarted { plan },
    )
    .unwrap();
    r.emit("salience", None, 1, &FutureCompletion(base))
        .unwrap();
    assert!(OriginalSalienceCaptureView::fold(r.seal().unwrap().log()).is_err());
}

#[test]
fn original_salience_incomplete_or_foreign_lifecycle_never_completes() {
    let (plan, completed) = fixture();
    let mut r = recorder();
    r.emit(
        "salience",
        None,
        0,
        &OriginalSalienceCaptureStarted { plan: plan.clone() },
    )
    .unwrap();
    assert!(
        OriginalSalienceCaptureView::fold(r.seal().unwrap().log())
            .unwrap()
            .is_empty()
    );
    let mut r = recorder();
    r.emit("salience", None, 0, &completed).unwrap();
    assert!(OriginalSalienceCaptureView::fold(r.seal().unwrap().log()).is_err());
    let mut r = recorder();
    r.emit(
        "salience",
        None,
        0,
        &OriginalSalienceCaptureStarted { plan: plan.clone() },
    )
    .unwrap();
    r.emit("foreign", None, 1, &completed).unwrap();
    assert!(OriginalSalienceCaptureView::fold(r.seal().unwrap().log()).is_err());
    let mut r = recorder();
    r.emit(
        "salience",
        None,
        0,
        &OriginalSalienceCaptureStarted { plan: plan.clone() },
    )
    .unwrap();
    r.emit("salience", None, 1, &completed).unwrap();
    r.emit("salience", None, 2, &completed).unwrap();
    assert!(OriginalSalienceCaptureView::fold(r.seal().unwrap().log()).is_err());
    let mut r = recorder();
    r.emit(
        "salience",
        Some("unobserved-parent"),
        0,
        &OriginalSalienceCaptureStarted { plan },
    )
    .unwrap();
    assert!(r.seal().is_err());
}
