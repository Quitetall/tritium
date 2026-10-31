use tritium_evidence::{
    DIRECT_LEVELS_FITTED_REGISTRATION, DirectFitReceiptView, EvidenceError, EvidenceRecorder,
    verify_jsonl_with_registry,
};
use tritium_format::trit_model::additive_semantic_digest;
use tritium_quantize::engine::{
    RecordedDirectFitError, fit_direct_levels, fit_direct_levels_recorded,
};
use tritium_schema::{
    ADDITIVE_TILE_SIZE, AdditiveLayout, Basis, DirectFitLevel, DirectFitLimits, DirectFitMetric,
    DirectFitPlan, DirectFitProfile, JointGroupFitOptions, PlaneAllocation, PlaneCodec,
    PlaneRelation, ScaleAnchor, ScaleLaw, ScalePrecision, Transport,
};

fn plan() -> DirectFitPlan {
    DirectFitPlan {
        layout: AdditiveLayout {
            rows: 1,
            cols: 32,
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
            max_weights: 32,
            max_metric_entries: 1024,
            max_output_trits: 192,
        },
    }
}

fn weights() -> Vec<f32> {
    (0..32).map(|i| (i as f32 - 16.0) / 16.0).collect()
}

fn recorder() -> EvidenceRecorder {
    EvidenceRecorder::new("fit-run", [DIRECT_LEVELS_FITTED_REGISTRATION]).unwrap()
}

fn record(
    plan: &DirectFitPlan,
    weights: &[f32],
    gram: Option<&[f64]>,
) -> (
    tritium_quantize::engine::RecordedDirectFit,
    EvidenceRecorder,
) {
    let mut recorder = recorder();
    let fit = fit_direct_levels_recorded(
        "layer.weight",
        plan,
        weights,
        gram,
        &mut recorder,
        "fit",
        None,
        0,
    )
    .unwrap();
    (fit, recorder)
}

#[test]
fn actual_controller_outputs_are_bound_and_replay_identically() {
    let plan = plan();
    let source = weights();
    let before = source.clone();
    let plain = fit_direct_levels(&plan, &source, None).unwrap();
    let (recorded, log) = record(&plan, &source, None);
    let observation = recorded.observation.observation();
    assert_eq!(source, before);
    assert_eq!(observation.tensor, "layer.weight");
    assert_eq!(observation.plan, plan);
    assert_eq!(observation.metric, DirectFitMetric::Identity);
    // Independent Python blake3 + struct.pack preimage calculation, ADR0057.
    assert_eq!(
        blake3::Hash::from_bytes(observation.source_f32.0)
            .to_hex()
            .as_str(),
        "3ff0391bc453893f85734b0e7c309cc62ec26282404a3b7e3c5f5627aa8634f9"
    );
    assert_eq!(
        observation.rayon_threads as usize,
        rayon::current_num_threads()
    );
    for ((expected, actual), observed) in
        plain.iter().zip(&recorded.levels).zip(&observation.outputs)
    {
        assert_eq!(
            expected.tensor.encode_payload().unwrap(),
            actual.tensor.encode_payload().unwrap()
        );
        assert_eq!(observed.level, actual.level);
        assert_eq!(observed.planes, actual.tensor.plane_count());
        assert_eq!(
            observed.content,
            additive_semantic_digest(&actual.tensor).unwrap()
        );
    }
    let verified = verify_jsonl_with_registry(
        &log.to_jsonl().unwrap(),
        [DIRECT_LEVELS_FITTED_REGISTRATION],
    )
    .unwrap();
    assert_eq!(
        DirectFitReceiptView::fold(&verified).unwrap(),
        [recorded.observation]
    );
    let (_, replay) = record(&plan, &source, None);
    assert_eq!(replay.to_jsonl().unwrap(), log.to_jsonl().unwrap());
    assert_eq!(replay.root_digest().unwrap(), log.root_digest().unwrap());
    assert!(verified.events[0].payload.get("verdict").is_none());
    assert!(verified.events[0].payload.get("weights").is_none());
}

#[test]
fn actual_source_metric_plan_and_name_changes_are_observable() {
    let mut plan = plan();
    let mut source = weights();
    let (base, log) = record(&plan, &source, None);
    source[16] = -0.0; // Equal numeric value, distinct exact original source bits.
    let (signed_zero, changed) = record(&plan, &source, None);
    assert_ne!(
        base.observation.observation().source_f32,
        signed_zero.observation.observation().source_f32
    );
    assert_ne!(log.root_digest().unwrap(), changed.root_digest().unwrap());

    let mut gram = vec![0.0; 1024];
    for i in 0..32 {
        gram[i * 32 + i] = 1.0;
    }
    let original = gram.clone();
    let (metric, _) = record(&plan, &source, Some(&gram));
    assert_eq!(gram, original);
    let DirectFitMetric::InputGramF64 { content } = metric.observation.observation().metric else {
        panic!("original Gram must be recorded")
    };
    assert_eq!(
        blake3::Hash::from_bytes(content.0).to_hex().as_str(),
        "bdc2a4a884e1edbd44eadf7818f984d19cffafad220f16d3ec84c812bf8ef575"
    );
    assert_ne!(
        signed_zero.observation.observation().metric,
        metric.observation.observation().metric
    );
    // Binding original bits is separate from mathematical metric equivalence.
    gram[1] = -0.0;
    let (changed_metric, _) = record(&plan, &source, Some(&gram));
    assert_ne!(
        metric.observation.observation().metric,
        changed_metric.observation.observation().metric
    );
    plan.levels.swap(0, 1);
    let (reordered, changed) = record(&plan, &source, None);
    assert_ne!(
        signed_zero.observation.observation().plan,
        reordered.observation.observation().plan
    );
    assert_ne!(changed.root_digest().unwrap(), log.root_digest().unwrap());
    let mut named = recorder();
    let fit = fit_direct_levels_recorded(
        "other.weight",
        &plan,
        &source,
        None,
        &mut named,
        "fit",
        None,
        0,
    )
    .unwrap();
    assert_ne!(
        fit.observation.observation().tensor,
        reordered.observation.observation().tensor
    );
}

#[test]
fn rejection_does_not_record_success_or_return_fits() {
    let plan = plan();
    let source = weights();
    let mut unregistered = EvidenceRecorder::new("fit-run", []).unwrap();
    // Even a bad source is not inspected/fitted before rejecting the recorder.
    let error = fit_direct_levels_recorded(
        "weight",
        &plan,
        &[],
        None,
        &mut unregistered,
        "fit",
        None,
        0,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        RecordedDirectFitError::Evidence(EvidenceError::UnregisteredSchema(_))
    ));
    assert!(unregistered.events().is_empty());
    for (name, span, time) in [
        ("", "fit", 0),
        ("weight", "", 0),
        ("weight", "fit", 1 << 53),
        ("weight\n", "fit", 0),
    ] {
        let mut log = recorder();
        assert!(
            fit_direct_levels_recorded(name, &plan, &source, None, &mut log, span, None, time)
                .is_err()
        );
        assert!(log.events().is_empty());
    }
    for bad in [vec![0.0; 31], vec![f32::NAN; 32]] {
        let mut log = recorder();
        assert!(matches!(
            fit_direct_levels_recorded("weight", &plan, &bad, None, &mut log, "fit", None, 0),
            Err(RecordedDirectFitError::Fit(_))
        ));
        assert!(log.events().is_empty());
    }
    let (_, mut log) = record(&plan, &source, None);
    let before = log.to_jsonl().unwrap();
    assert!(
        fit_direct_levels_recorded(
            "weight",
            &plan,
            &source,
            None,
            &mut log,
            "fit",
            Some("new-parent"),
            1
        )
        .is_err()
    );
    assert_eq!(log.to_jsonl().unwrap(), before);
}

#[test]
fn folds_reject_root_tampering_and_structural_lies_even_with_valid_chains() {
    let (fit, recorder) = record(&plan(), &weights(), None);
    let verified = verify_jsonl_with_registry(
        &recorder.to_jsonl().unwrap(),
        [DIRECT_LEVELS_FITTED_REGISTRATION],
    )
    .unwrap();
    let mut wrong_root = verified.clone();
    wrong_root.root_digest = "00".repeat(32);
    assert!(matches!(
        DirectFitReceiptView::fold(&wrong_root),
        Err(EvidenceError::RootMismatch)
    ));
    let mut tampered = verified;
    tampered.events[0].payload["tensor"] = "tampered".into();
    assert!(DirectFitReceiptView::fold(&tampered).is_err());
    for corruption in 0..5 {
        let mut observation = fit.observation.observation().clone();
        match corruption {
            0 => {
                observation.outputs.pop();
            }
            1 => {
                observation.outputs[0].planes = 1;
            }
            2 => {
                observation.plan.levels[1].name = observation.plan.levels[0].name.clone();
                observation.outputs[1].level = observation.plan.levels[0].name.clone();
            }
            3 => {
                observation.rayon_threads = 0;
            }
            _ => {
                observation.outputs[0].level = "unrequested".into();
            }
        }
        let mut malformed = self::recorder();
        assert!(
            malformed
                .emit_direct_levels_fitted("fit", None, 0, &observation)
                .is_err()
        );
        assert!(malformed.events().is_empty());
        // Generic emit can hash a declaration; typed replay must still reject it.
        malformed.emit("fit", None, 0, &observation).unwrap();
        let verified = verify_jsonl_with_registry(
            &malformed.to_jsonl().unwrap(),
            [DIRECT_LEVELS_FITTED_REGISTRATION],
        )
        .unwrap();
        assert!(matches!(
            DirectFitReceiptView::fold(&verified),
            Err(EvidenceError::InvalidPayload { .. })
        ));
    }
}

#[test]
fn actual_rayon_environment_is_recorded_not_assumed_or_pinned() {
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| record(&plan(), &weights(), None))
    };
    let (one, one_log) = run(1);
    let (two, two_log) = run(2);
    assert_eq!(one.observation.observation().rayon_threads, 1);
    assert_eq!(two.observation.observation().rayon_threads, 2);
    assert_ne!(
        one_log.root_digest().unwrap(),
        two_log.root_digest().unwrap()
    );
    assert_eq!(
        one.observation.observation().outputs,
        two.observation.observation().outputs
    );
}

#[test]
fn replay_rejects_unknown_fields_and_unsupported_versions() {
    use tritium_evidence::{EventRegistration, EvidenceEvent, verify_jsonl};
    #[derive(serde::Serialize)]
    struct Raw(serde_json::Value);
    impl EvidenceEvent for Raw {
        const SCHEMA_ID: &'static str = "tritium.quantize.direct_levels_fitted";
        const VERSION: u16 = 1;
    }
    #[derive(serde::Serialize)]
    struct Future(serde_json::Value);
    impl EvidenceEvent for Future {
        const SCHEMA_ID: &'static str = Raw::SCHEMA_ID;
        const VERSION: u16 = 2;
    }
    let (fit, _) = record(&plan(), &weights(), None);
    let payload = serde_json::to_value(fit.observation.observation()).unwrap();
    for field in ["release_approved", "fake_model_id"] {
        let mut forged = payload.clone();
        forged[field] = true.into();
        let mut log = recorder();
        log.emit("fit", None, 0, &Raw(forged)).unwrap();
        let verified = verify_jsonl(&log.to_jsonl().unwrap()).unwrap();
        assert!(matches!(
            DirectFitReceiptView::fold(&verified),
            Err(EvidenceError::InvalidPayload { .. })
        ));
    }
    let mut future = EvidenceRecorder::new(
        "fit-run",
        [EventRegistration {
            schema: Future::SCHEMA_ID,
            version: Future::VERSION,
        }],
    )
    .unwrap();
    future.emit("fit", None, 0, &Future(payload)).unwrap();
    let verified = verify_jsonl(&future.to_jsonl().unwrap()).unwrap();
    assert!(matches!(
        DirectFitReceiptView::fold(&verified),
        Err(EvidenceError::UnregisteredSchema(_))
    ));
}
