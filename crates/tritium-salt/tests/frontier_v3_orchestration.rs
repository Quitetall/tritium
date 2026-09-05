use std::sync::Arc;

use tritium_salt::{
    FRONTIER_SOLVER_ABI_V1, FrontierOrdering, FrontierPlanError, FrontierProfile,
    FrontierProfileId, FrontierResourceEstimate, FrontierRunReceipt, FrontierSolver,
    FrontierSolverError, FrontierStage, FrontierStageOutcome, FrontierStageReceipt,
    FrontierStageRequest, ResourceVector, SolverDescriptor, SolverFamily, SolverId, SolverRegistry,
    SolverRequest, SolverTrust,
};

fn id(byte: u8) -> tritium_salt::ContentId {
    tritium_salt::ContentId::from_digest([byte; 32])
}

fn resources() -> ResourceVector {
    ResourceVector::new(100, 20, 300, 40, 50, 60, 700, Some(80))
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

#[derive(Debug)]
struct FixedSolver {
    descriptor: SolverDescriptor,
}

impl FrontierSolver for FixedSolver {
    fn descriptor(&self) -> &SolverDescriptor {
        &self.descriptor
    }

    fn estimate(
        &self,
        _request: &SolverRequest,
    ) -> Result<FrontierResourceEstimate, FrontierSolverError> {
        Ok(FrontierResourceEstimate::new(resources(), id(91), id(92)).unwrap())
    }
}

fn profile() -> FrontierProfile {
    FrontierProfile::new(
        FrontierProfileId::new("salt.reference.v1").unwrap(),
        FrontierOrdering::Fixed,
        vec![SolverId::new("salt.v3").unwrap()],
        SolverTrust::Registered,
        resources(),
        None,
        false,
    )
    .unwrap()
}

fn request(stage_index: u32, stage: FrontierStage, input_id: u8) -> FrontierStageRequest {
    let profile = profile();
    let mut registry = SolverRegistry::new();
    registry
        .register(Arc::new(FixedSolver {
            descriptor: solver(),
        }))
        .unwrap();
    let shape = profile.request(16, 32).unwrap();
    let plan = registry
        .plan(
            &SolverId::new("salt.v3").unwrap(),
            profile.minimum_trust(),
            &shape,
        )
        .unwrap();
    FrontierStageRequest::new(
        id(1),
        id(2),
        stage_index,
        stage,
        &profile,
        &plan,
        id(input_id),
    )
    .unwrap()
}

fn completed(request: &FrontierStageRequest, output_id: u8) -> FrontierStageReceipt {
    FrontierStageReceipt::new(
        request.run_id(),
        request.source_id(),
        request.stage_index(),
        request.stage(),
        request.profile_id().clone(),
        request.solver().clone(),
        request.input_id(),
        Some(id(output_id)),
        request.estimate(),
        resources(),
        request.budget(),
        FrontierStageOutcome::Completed,
        None,
    )
    .unwrap()
}

#[test]
fn stage_request_round_trips_and_has_stable_content_identity() {
    let request = request(0, FrontierStage::AdmitSource, 2);
    let encoded = serde_json::to_string_pretty(&request).unwrap();
    let decoded: FrontierStageRequest = serde_json::from_str(&encoded).unwrap();

    assert_eq!(decoded, request);
    assert_eq!(decoded.content_id(), request.content_id());
    assert_eq!(
        serde_json::from_str::<FrontierStageRequest>(include_str!(
            "fixtures/frontier-stage-request-v1.json"
        ))
        .unwrap(),
        request
    );
}

#[test]
fn serialized_stage_request_cannot_bypass_profile_admission() {
    let mut wrong_solver = serde_json::to_value(request(0, FrontierStage::Fit, 2)).unwrap();
    wrong_solver["solver"]["id"] = serde_json::json!("qtea.v1");
    wrong_solver["solver"]["family"] = serde_json::json!("qtea-salient-residual");
    assert!(serde_json::from_value::<FrontierStageRequest>(wrong_solver).is_err());

    let mut wrong_budget = serde_json::to_value(request(0, FrontierStage::Fit, 2)).unwrap();
    wrong_budget["request"]["budget"]["host_ram_bytes"] = serde_json::json!(101);
    assert!(serde_json::from_value::<FrontierStageRequest>(wrong_budget).is_err());

    let mut unknown = serde_json::to_value(request(0, FrontierStage::Fit, 2)).unwrap();
    unknown["untrusted"] = serde_json::json!(true);
    assert!(serde_json::from_value::<FrontierStageRequest>(unknown).is_err());
}

#[test]
fn stage_request_refuses_solver_or_budget_outside_profile() {
    let wrong_solver = SolverDescriptor::new(
        SolverId::new("qtea.v1").unwrap(),
        SolverFamily::QteaSalientResidual,
        SolverTrust::Registered,
        FRONTIER_SOLVER_ABI_V1,
    )
    .unwrap();
    let mut wrong_registry = SolverRegistry::new();
    wrong_registry
        .register(Arc::new(FixedSolver {
            descriptor: wrong_solver,
        }))
        .unwrap();
    let profile = profile();
    let wrong_plan = wrong_registry
        .plan(
            &SolverId::new("qtea.v1").unwrap(),
            SolverTrust::Registered,
            &profile.request(16, 32).unwrap(),
        )
        .unwrap();
    assert!(matches!(
        FrontierStageRequest::new(
            id(1),
            id(2),
            0,
            FrontierStage::Fit,
            &profile,
            &wrong_plan,
            id(3),
        ),
        Err(FrontierPlanError::ProfileSolverMismatch { .. })
    ));

    let mut changed = resources();
    changed = ResourceVector::new(
        changed.host_ram_bytes() + 1,
        changed.vram_bytes(),
        changed.disk_bytes(),
        changed.artifact_bytes(),
        changed.resident_bytes(),
        changed.transient_bytes(),
        changed.fitting_millis(),
        changed.runtime_latency_micros(),
    );
    let changed_request = SolverRequest::new(16, 32, changed).unwrap();
    let mut registry = SolverRegistry::new();
    registry
        .register(Arc::new(FixedSolver {
            descriptor: solver(),
        }))
        .unwrap();
    let changed_plan = registry
        .plan(
            &SolverId::new("salt.v3").unwrap(),
            SolverTrust::Registered,
            &changed_request,
        )
        .unwrap();
    assert!(matches!(
        FrontierStageRequest::new(
            id(1),
            id(2),
            0,
            FrontierStage::Fit,
            &profile,
            &changed_plan,
            id(3),
        ),
        Err(FrontierPlanError::ProfileBudgetMismatch { .. })
    ));
}

#[test]
fn request_accepts_only_exactly_bound_terminal_receipt() {
    let request = request(2, FrontierStage::Fit, 3);
    request.validate_receipt(&completed(&request, 4)).unwrap();

    let wrong = FrontierStageReceipt::new(
        request.run_id(),
        request.source_id(),
        3,
        request.stage(),
        request.profile_id().clone(),
        request.solver().clone(),
        request.input_id(),
        Some(id(4)),
        request.estimate(),
        resources(),
        request.budget(),
        FrontierStageOutcome::Completed,
        None,
    )
    .unwrap();
    assert!(matches!(
        request.validate_receipt(&wrong),
        Err(FrontierPlanError::StageReceiptMismatch {
            field: "stage_index"
        })
    ));
}

#[test]
fn run_receipt_validates_chain_and_builds_next_resumable_request() {
    let admit = request(0, FrontierStage::AdmitSource, 2);
    let admit_receipt = completed(&admit, 3);
    let fit = admit.next(FrontierStage::Fit, &admit_receipt).unwrap();
    assert_eq!(fit.stage_index(), 1);
    assert_eq!(fit.input_id(), id(3));
    let fit_receipt = completed(&fit, 4);
    let fit_receipt_id = fit_receipt.content_id();
    let fit_receipt_round_trip: FrontierStageReceipt =
        serde_json::from_str(&serde_json::to_string(&fit_receipt).unwrap()).unwrap();
    assert_eq!(fit_receipt_round_trip.content_id(), fit_receipt_id);

    let run = FrontierRunReceipt::new(&profile(), vec![admit_receipt, fit_receipt]).unwrap();
    assert_eq!(run.last_output_id(), Some(id(4)));
    assert_eq!(run.next_stage_index(), 2);
    assert!(!run.is_terminal());
    let run_id = run.content_id();
    let round_trip =
        serde_json::from_str::<FrontierRunReceipt>(&serde_json::to_string(&run).unwrap()).unwrap();
    assert_eq!(round_trip.content_id(), run_id);
    assert_eq!(round_trip, run);
}

#[test]
fn run_receipt_refuses_gaps_identity_drift_and_continuation_after_failure() {
    let first = request(0, FrontierStage::AdmitSource, 2);
    let first_done = completed(&first, 3);

    let gap = request(2, FrontierStage::Fit, 3);
    assert!(matches!(
        FrontierRunReceipt::new(&profile(), vec![first_done.clone(), completed(&gap, 4)]),
        Err(FrontierPlanError::NonContiguousStageIndex { .. })
    ));

    let wrong_input = request(1, FrontierStage::Fit, 9);
    assert!(matches!(
        FrontierRunReceipt::new(
            &profile(),
            vec![first_done.clone(), completed(&wrong_input, 4)]
        ),
        Err(FrontierPlanError::BrokenStageInputChain { .. })
    ));

    let failed = FrontierStageReceipt::new(
        first.run_id(),
        first.source_id(),
        1,
        FrontierStage::Fit,
        first.profile_id().clone(),
        first.solver().clone(),
        id(3),
        None,
        first.estimate(),
        resources(),
        first.budget(),
        FrontierStageOutcome::Failed,
        Some("worker exited 17".to_owned()),
    )
    .unwrap();
    let after_failure = request(2, FrontierStage::Pack, 3);
    assert!(matches!(
        FrontierRunReceipt::new(
            &profile(),
            vec![first_done, failed, completed(&after_failure, 5)]
        ),
        Err(FrontierPlanError::StageAfterTerminalOutcome { .. })
    ));
}

#[test]
fn serialized_run_receipt_revalidates_chain_and_profile_snapshot() {
    let first = request(0, FrontierStage::AdmitSource, 2);
    let first_done = completed(&first, 3);
    let second = first.next(FrontierStage::Fit, &first_done).unwrap();
    let run = FrontierRunReceipt::new(&profile(), vec![first_done, completed(&second, 4)]).unwrap();

    let mut broken_chain = serde_json::to_value(&run).unwrap();
    broken_chain["receipts"][1]["input_id"] =
        serde_json::json!("tsc1_0909090909090909090909090909090909090909090909090909090909090909");
    assert!(serde_json::from_value::<FrontierRunReceipt>(broken_chain).is_err());

    let mut changed_profile = serde_json::to_value(&run).unwrap();
    changed_profile["profile"]["id"] = serde_json::json!("salt.changed.v1");
    assert!(serde_json::from_value::<FrontierRunReceipt>(changed_profile).is_err());
}

#[test]
fn failed_receipt_is_terminal_and_cannot_seed_next_request() {
    let request = request(0, FrontierStage::Fit, 2);
    let failed = FrontierStageReceipt::new(
        request.run_id(),
        request.source_id(),
        request.stage_index(),
        request.stage(),
        request.profile_id().clone(),
        request.solver().clone(),
        request.input_id(),
        None,
        request.estimate(),
        resources(),
        request.budget(),
        FrontierStageOutcome::Failed,
        Some("deterministic failure".to_owned()),
    )
    .unwrap();
    let run = FrontierRunReceipt::new(&profile(), vec![failed.clone()]).unwrap();
    assert!(run.is_terminal());
    assert_eq!(run.last_output_id(), None);
    assert!(matches!(
        request.next(FrontierStage::Pack, &failed),
        Err(FrontierPlanError::CannotAdvanceTerminalStage { .. })
    ));
}
