use tritium_salt::{
    ContentId, FRONTIER_PARETO_RECEIPT_SCHEMA_V1, FRONTIER_SOLVER_ABI_V1,
    FRONTIER_STAGE_RECEIPT_SCHEMA_V1, FrontierObjectiveDirection, FrontierObjectiveSpec,
    FrontierObjectiveValue, FrontierParetoCandidate, FrontierParetoReceipt, FrontierPlanError,
    FrontierProfile, FrontierProfileId, FrontierSelection, FrontierStage, FrontierStageOutcome,
    FrontierStageReceipt, ResourceVector, SolverDescriptor, SolverFamily, SolverId, SolverTrust,
};

fn digest(label: &str) -> ContentId {
    ContentId::of_bytes(label.as_bytes())
}

fn resources(artifact_bytes: u64, resident_bytes: u64, fitting_millis: u64) -> ResourceVector {
    ResourceVector::new(
        4_096,
        2_048,
        8_192,
        artifact_bytes,
        resident_bytes,
        64,
        fitting_millis,
        Some(250),
    )
}

fn solver() -> SolverDescriptor {
    SolverDescriptor::new(
        SolverId::new("salt.v3").unwrap(),
        SolverFamily::Salt,
        SolverTrust::Registered,
        FRONTIER_SOLVER_ABI_V1,
    )
    .unwrap()
}

fn profile(auto_select: bool) -> FrontierProfile {
    FrontierProfile::new(
        FrontierProfileId::new("research.default").unwrap(),
        tritium_salt::FrontierOrdering::Search,
        vec![SolverId::new("salt.v3").unwrap()],
        SolverTrust::Registered,
        resources(100, 200, 1_000),
        None,
        auto_select,
    )
    .unwrap()
}

#[test]
fn stage_receipt_round_trips_and_binds_terminal_resource_evidence() {
    let receipt = FrontierStageReceipt::new(
        digest("run"),
        digest("source"),
        3,
        FrontierStage::Fit,
        FrontierProfileId::new("research.default").unwrap(),
        solver(),
        digest("input"),
        Some(digest("output")),
        resources(80, 150, 900),
        resources(85, 160, 950),
        resources(100, 200, 1_000),
        FrontierStageOutcome::Completed,
        None,
    )
    .unwrap();

    assert_eq!(receipt.stage_index(), 3);
    assert_eq!(receipt.stage(), FrontierStage::Fit);
    assert_eq!(receipt.output_id(), Some(digest("output")));
    assert_eq!(receipt.measured().artifact_bytes(), 85);

    let encoded = serde_json::to_string_pretty(&receipt).unwrap() + "\n";
    assert_eq!(
        encoded,
        include_str!("fixtures/frontier-stage-receipt-v1.json")
    );
    assert_eq!(
        serde_json::from_str::<FrontierStageReceipt>(&encoded).unwrap(),
        receipt
    );
    let value = serde_json::to_value(&receipt).unwrap();
    assert_eq!(value["schema"], FRONTIER_STAGE_RECEIPT_SCHEMA_V1);
}

#[test]
fn stage_receipt_refuses_false_success_and_incoherent_terminal_outcomes() {
    let common = || {
        (
            digest("run"),
            digest("source"),
            FrontierProfileId::new("research.default").unwrap(),
            solver(),
            digest("input"),
        )
    };
    let (run, source, profile, solver, input) = common();
    assert!(matches!(
        FrontierStageReceipt::new(
            run,
            source,
            0,
            FrontierStage::Pack,
            profile,
            solver,
            input,
            None,
            resources(80, 150, 900),
            resources(85, 160, 950),
            resources(100, 200, 1_000),
            FrontierStageOutcome::Completed,
            None,
        ),
        Err(FrontierPlanError::IncoherentStageOutcome { .. })
    ));

    let (run, source, profile, solver, input) = common();
    assert!(
        FrontierStageReceipt::new(
            run,
            source,
            0,
            FrontierStage::Pack,
            profile,
            solver,
            input,
            Some(digest("partial output")),
            resources(80, 150, 900),
            resources(101, 201, 1_001),
            resources(100, 200, 1_000),
            FrontierStageOutcome::BudgetExceeded,
            Some("measured budget exceeded".into()),
        )
        .is_ok()
    );

    let (run, source, profile, solver, input) = common();
    assert!(matches!(
        FrontierStageReceipt::new(
            run,
            source,
            0,
            FrontierStage::Pack,
            profile,
            solver,
            input,
            None,
            resources(80, 150, 900),
            resources(85, 160, 950),
            resources(100, 200, 1_000),
            FrontierStageOutcome::BudgetExceeded,
            Some("claimed violation".into()),
        ),
        Err(FrontierPlanError::IncoherentStageOutcome { .. })
    ));

    let (run, source, profile, solver, input) = common();
    assert!(matches!(
        FrontierStageReceipt::new(
            run,
            source,
            0,
            FrontierStage::Pack,
            profile,
            solver,
            input,
            Some(digest("impossible output")),
            resources(80, 150, 900),
            resources(85, 160, 950),
            resources(100, 200, 1_000),
            FrontierStageOutcome::Failed,
            Some("driver failed".into()),
        ),
        Err(FrontierPlanError::IncoherentStageOutcome { .. })
    ));

    let (run, source, profile, solver, input) = common();
    let missing_latency = ResourceVector::new(4_096, 2_048, 8_192, 85, 160, 64, 950, None);
    assert!(matches!(
        FrontierStageReceipt::new(
            run,
            source,
            0,
            FrontierStage::Pack,
            profile,
            solver,
            input,
            Some(digest("output")),
            resources(80, 150, 900),
            missing_latency,
            resources(100, 200, 1_000),
            FrontierStageOutcome::Completed,
            None,
        ),
        Err(FrontierPlanError::IncompleteStageMeasurement { .. })
    ));
}

fn objectives() -> Vec<FrontierObjectiveSpec> {
    vec![
        FrontierObjectiveSpec::new(
            "artifact-bytes",
            FrontierObjectiveDirection::Minimize,
            "bytes",
        )
        .unwrap(),
        FrontierObjectiveSpec::new(
            "quality-retention-ppm",
            FrontierObjectiveDirection::Maximize,
            "ppm",
        )
        .unwrap(),
    ]
}

fn candidate(
    artifact: &str,
    receipt: &str,
    artifact_bytes: i64,
    quality_ppm: i64,
) -> FrontierParetoCandidate {
    FrontierParetoCandidate::new(
        digest(artifact),
        digest(receipt),
        vec![
            FrontierObjectiveValue::new("artifact-bytes", artifact_bytes).unwrap(),
            FrontierObjectiveValue::new("quality-retention-ppm", quality_ppm).unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn pareto_receipt_round_trips_nondominated_frontier_and_explicit_selection() {
    let mut candidates = vec![
        candidate("artifact-a", "receipt-a", 80, 950_000),
        candidate("artifact-b", "receipt-b", 95, 980_000),
    ];
    candidates.sort_by_key(|candidate| *candidate.artifact_id().as_bytes());
    let selected = candidates[1].artifact_id();
    let receipt = FrontierParetoReceipt::new(
        digest("run"),
        digest("source"),
        &profile(true),
        objectives(),
        candidates,
        FrontierSelection::Automatic {
            artifact_id: selected,
            policy_id: digest("selection policy"),
        },
    )
    .unwrap();

    assert_eq!(receipt.selected_artifact_id(), Some(selected));
    assert_eq!(receipt.candidates().len(), 2);
    let encoded = serde_json::to_string_pretty(&receipt).unwrap() + "\n";
    assert_eq!(
        encoded,
        include_str!("fixtures/frontier-pareto-receipt-v1.json")
    );
    assert_eq!(
        serde_json::from_str::<FrontierParetoReceipt>(&encoded).unwrap(),
        receipt
    );
    let value = serde_json::to_value(&receipt).unwrap();
    assert_eq!(value["schema"], FRONTIER_PARETO_RECEIPT_SCHEMA_V1);
}

#[test]
fn pareto_receipt_refuses_dominated_candidates_bad_order_and_unauthorized_auto_selection() {
    let dominated = vec![
        candidate("artifact-a", "receipt-a", 80, 950_000),
        candidate("artifact-b", "receipt-b", 90, 940_000),
    ];
    assert!(matches!(
        FrontierParetoReceipt::new(
            digest("run"),
            digest("source"),
            &profile(true),
            objectives(),
            dominated,
            FrontierSelection::Pending,
        ),
        Err(FrontierPlanError::DominatedParetoCandidate { .. })
    ));

    let unsorted_objectives = vec![objectives()[1].clone(), objectives()[0].clone()];
    assert!(matches!(
        FrontierParetoReceipt::new(
            digest("run"),
            digest("source"),
            &profile(true),
            unsorted_objectives,
            vec![candidate("artifact-a", "receipt-a", 80, 950_000)],
            FrontierSelection::Pending,
        ),
        Err(FrontierPlanError::NonCanonicalObjectiveOrder { .. })
    ));

    let only = candidate("artifact-a", "receipt-a", 80, 950_000);
    assert!(matches!(
        FrontierParetoReceipt::new(
            digest("run"),
            digest("source"),
            &profile(false),
            objectives(),
            vec![only.clone()],
            FrontierSelection::Automatic {
                artifact_id: only.artifact_id(),
                policy_id: digest("selection policy"),
            },
        ),
        Err(FrontierPlanError::AutomaticSelectionForbidden { .. })
    ));
}

#[test]
fn receipt_readers_refuse_corrupted_derived_fields_and_unknown_data() {
    let mut stage: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/frontier-stage-receipt-v1.json")).unwrap();
    stage["schema"] = "tritium.frontier-stage-receipt.v999".into();
    assert!(serde_json::from_value::<FrontierStageReceipt>(stage).is_err());

    let mut pareto: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/frontier-pareto-receipt-v1.json")).unwrap();
    pareto["auto_select"] = false.into();
    assert!(serde_json::from_value::<FrontierParetoReceipt>(pareto).is_err());

    let mut pareto: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/frontier-pareto-receipt-v1.json")).unwrap();
    pareto["future_field"] = true.into();
    assert!(serde_json::from_value::<FrontierParetoReceipt>(pareto).is_err());
}
