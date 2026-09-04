use std::sync::Arc;

use tritium_salt::{
    FRONTIER_PROFILE_SCHEMA_V1, FRONTIER_SOLVER_ABI_V1, FrontierOrdering, FrontierPlanError,
    FrontierProfile, FrontierProfileId, FrontierSolver, FrontierSolverError, ResourceDimension,
    ResourceVector, SolverDescriptor, SolverFamily, SolverId, SolverRegistry, SolverRequest,
    SolverTrust,
};

#[derive(Debug)]
struct FixedEstimateSolver {
    descriptor: SolverDescriptor,
    estimate: ResourceVector,
}

impl FrontierSolver for FixedEstimateSolver {
    fn descriptor(&self) -> &SolverDescriptor {
        &self.descriptor
    }

    fn estimate(&self, _request: &SolverRequest) -> Result<ResourceVector, FrontierSolverError> {
        Ok(self.estimate)
    }
}

fn solver(
    id: &str,
    trust: SolverTrust,
    abi_version: u16,
    estimate: ResourceVector,
) -> Arc<dyn FrontierSolver> {
    Arc::new(FixedEstimateSolver {
        descriptor: SolverDescriptor::new(
            SolverId::new(id).expect("valid test solver id"),
            SolverFamily::Salt,
            trust,
            abi_version,
        )
        .expect("valid descriptor"),
        estimate,
    })
}

#[test]
fn solver_ids_are_canonical_and_bounded() {
    assert_eq!(SolverId::new("salt.v3").unwrap().as_str(), "salt.v3");
    for invalid in ["", "SALT", ".salt", "salt/v3", "salt v3"] {
        assert!(SolverId::new(invalid).is_err(), "accepted {invalid:?}");
    }
    assert!(SolverId::new("x".repeat(129)).is_err());
}

#[test]
fn registry_is_deterministic_and_refuses_duplicate_or_wrong_abi() {
    let tiny = ResourceVector::new(1, 0, 1, 1, 1, 1, 1, None);
    let mut registry = SolverRegistry::new();
    registry
        .register(solver(
            "salt.v3",
            SolverTrust::Registered,
            FRONTIER_SOLVER_ABI_V1,
            tiny,
        ))
        .unwrap();
    registry
        .register(solver(
            "externd.v1",
            SolverTrust::Experimental,
            FRONTIER_SOLVER_ABI_V1,
            tiny,
        ))
        .unwrap();

    assert_eq!(registry.ids(), ["externd.v1", "salt.v3"]);
    assert!(matches!(
        registry.register(solver(
            "salt.v3",
            SolverTrust::Certified,
            FRONTIER_SOLVER_ABI_V1,
            tiny,
        )),
        Err(FrontierPlanError::DuplicateSolver { .. })
    ));
    assert!(matches!(
        SolverDescriptor::new(
            SolverId::new("future.v1").unwrap(),
            SolverFamily::Custom,
            SolverTrust::Experimental,
            FRONTIER_SOLVER_ABI_V1 + 1,
        ),
        Err(FrontierPlanError::UnsupportedSolverAbi { .. })
    ));
}

#[test]
fn planning_refuses_unknown_and_under_trusted_solvers() {
    let mut registry = SolverRegistry::new();
    registry
        .register(solver(
            "qtea.v1",
            SolverTrust::Experimental,
            FRONTIER_SOLVER_ABI_V1,
            ResourceVector::new(1, 0, 1, 1, 1, 1, 1, None),
        ))
        .unwrap();
    let request =
        SolverRequest::new(256, 256, ResourceVector::new(2, 0, 2, 2, 2, 2, 2, None)).unwrap();

    assert!(matches!(
        registry.plan(
            &SolverId::new("missing.v1").unwrap(),
            SolverTrust::Experimental,
            &request
        ),
        Err(FrontierPlanError::UnknownSolver { .. })
    ));
    assert!(matches!(
        registry.plan(
            &SolverId::new("qtea.v1").unwrap(),
            SolverTrust::Registered,
            &request
        ),
        Err(FrontierPlanError::InsufficientTrust { .. })
    ));
}

#[test]
fn hard_budget_refusal_reports_every_exceeded_dimension() {
    let mut registry = SolverRegistry::new();
    registry
        .register(solver(
            "twla.v1",
            SolverTrust::Registered,
            FRONTIER_SOLVER_ABI_V1,
            ResourceVector::new(11, 13, 17, 19, 23, 29, 31, Some(37)),
        ))
        .unwrap();
    let request = SolverRequest::new(
        64,
        64,
        ResourceVector::new(10, 12, 17, 18, 23, 28, 30, Some(36)),
    )
    .unwrap();

    let error = registry
        .plan(
            &SolverId::new("twla.v1").unwrap(),
            SolverTrust::Registered,
            &request,
        )
        .unwrap_err();
    let FrontierPlanError::BudgetExceeded { violations, .. } = error else {
        panic!("unexpected error: {error:?}");
    };
    assert_eq!(
        violations
            .iter()
            .map(|violation| violation.dimension())
            .collect::<Vec<_>>(),
        vec![
            ResourceDimension::HostRamBytes,
            ResourceDimension::VramBytes,
            ResourceDimension::ArtifactBytes,
            ResourceDimension::TransientBytes,
            ResourceDimension::FittingMillis,
            ResourceDimension::RuntimeLatencyMicros,
        ]
    );
    assert!(
        violations
            .iter()
            .all(|item| item.required() > item.available())
    );
}

#[test]
fn admitted_plan_binds_solver_request_and_exact_estimate() {
    let estimate = ResourceVector::new(8, 0, 13, 5, 7, 3, 21, None);
    let mut registry = SolverRegistry::new();
    registry
        .register(solver(
            "salt.v3",
            SolverTrust::Certified,
            FRONTIER_SOLVER_ABI_V1,
            estimate,
        ))
        .unwrap();
    let request =
        SolverRequest::new(32, 64, ResourceVector::new(8, 0, 13, 5, 7, 3, 21, None)).unwrap();

    let plan = registry
        .plan(
            &SolverId::new("salt.v3").unwrap(),
            SolverTrust::Registered,
            &request,
        )
        .unwrap();
    assert_eq!(plan.solver_id().as_str(), "salt.v3");
    assert_eq!(plan.request(), &request);
    assert_eq!(plan.estimate(), estimate);
}

#[test]
fn serialized_contracts_cannot_bypass_constructor_validation() {
    assert!(serde_json::from_str::<SolverId>(r#""SALT""#).is_err());
    let invalid_descriptor = format!(
        r#"{{"id":"salt.v3","family":"salt","trust":"registered","abi_version":{}}}"#,
        FRONTIER_SOLVER_ABI_V1 + 1
    );
    assert!(serde_json::from_str::<SolverDescriptor>(&invalid_descriptor).is_err());
    assert!(serde_json::from_str::<SolverRequest>(
        r#"{"rows":0,"columns":4,"budget":{"host_ram_bytes":1,"vram_bytes":0,"disk_bytes":1,"artifact_bytes":1,"resident_bytes":1,"transient_bytes":1,"fitting_millis":1,"runtime_latency_micros":null}}"#
    )
    .is_err());
}

#[test]
fn constrained_dimension_requires_an_estimate() {
    let mut registry = SolverRegistry::new();
    registry
        .register(solver(
            "salt.v3",
            SolverTrust::Registered,
            FRONTIER_SOLVER_ABI_V1,
            ResourceVector::new(1, 0, 1, 1, 1, 1, 1, None),
        ))
        .unwrap();
    let request =
        SolverRequest::new(8, 8, ResourceVector::new(1, 0, 1, 1, 1, 1, 1, Some(10))).unwrap();

    assert!(matches!(
        registry.plan(
            &SolverId::new("salt.v3").unwrap(),
            SolverTrust::Registered,
            &request,
        ),
        Err(FrontierPlanError::IncompleteEstimate {
            dimension: ResourceDimension::RuntimeLatencyMicros,
            ..
        })
    ));
}

#[test]
fn profile_contract_round_trips_with_stable_schema() {
    let profile = FrontierProfile::new(
        FrontierProfileId::new("research.default").unwrap(),
        FrontierOrdering::Search,
        vec![
            SolverId::new("salt.v3").unwrap(),
            SolverId::new("qtea.v1").unwrap(),
        ],
        SolverTrust::Experimental,
        ResourceVector::new(1024, 512, 2048, 768, 896, 128, 60_000, None),
        Some(FrontierProfileId::new("safe.salt-only").unwrap()),
        false,
    )
    .unwrap();

    let value = serde_json::to_value(&profile).unwrap();
    assert_eq!(value["schema"], FRONTIER_PROFILE_SCHEMA_V1);
    assert_eq!(value["ordering"], "search");
    assert_eq!(value["fallback_profile"], "safe.salt-only");
    assert_eq!(value["auto_select"], false);
    assert!(value.get("elements").is_none());
    assert_eq!(
        serde_json::from_value::<FrontierProfile>(value).unwrap(),
        profile
    );
}

#[test]
fn profile_refuses_bad_schema_empty_duplicate_and_recursive_fallback() {
    let budget = ResourceVector::new(1, 0, 1, 1, 1, 1, 1, None);
    let profile_id = FrontierProfileId::new("research.default").unwrap();
    assert!(matches!(
        FrontierProfile::new(
            profile_id.clone(),
            FrontierOrdering::Search,
            Vec::new(),
            SolverTrust::Experimental,
            budget,
            None,
            false,
        ),
        Err(FrontierPlanError::EmptySolverSet { .. })
    ));
    assert!(matches!(
        FrontierProfile::new(
            profile_id.clone(),
            FrontierOrdering::Fixed,
            vec![
                SolverId::new("salt.v3").unwrap(),
                SolverId::new("salt.v3").unwrap(),
            ],
            SolverTrust::Registered,
            budget,
            None,
            false,
        ),
        Err(FrontierPlanError::DuplicateProfileSolver { .. })
    ));
    assert!(matches!(
        FrontierProfile::new(
            profile_id.clone(),
            FrontierOrdering::Search,
            vec![SolverId::new("salt.v3").unwrap()],
            SolverTrust::Experimental,
            budget,
            Some(profile_id),
            false,
        ),
        Err(FrontierPlanError::RecursiveFallback { .. })
    ));

    let bad_schema = r#"{"schema":"tritium.frontier-profile.v999","id":"research.default","ordering":"search","solver_ids":["salt.v3"],"minimum_trust":"experimental","budget":{"host_ram_bytes":1,"vram_bytes":0,"disk_bytes":1,"artifact_bytes":1,"resident_bytes":1,"transient_bytes":1,"fitting_millis":1,"runtime_latency_micros":null},"fallback_profile":null,"auto_select":false}"#;
    assert!(serde_json::from_str::<FrontierProfile>(bad_schema).is_err());
}

#[test]
fn profile_validation_refuses_missing_or_under_trusted_members() {
    let budget = ResourceVector::new(4, 0, 4, 4, 4, 4, 4, None);
    let mut registry = SolverRegistry::new();
    registry
        .register(solver(
            "salt.v3",
            SolverTrust::Experimental,
            FRONTIER_SOLVER_ABI_V1,
            budget,
        ))
        .unwrap();
    let registered = FrontierProfile::new(
        FrontierProfileId::new("release.fixed").unwrap(),
        FrontierOrdering::Fixed,
        vec![SolverId::new("salt.v3").unwrap()],
        SolverTrust::Registered,
        budget,
        None,
        false,
    )
    .unwrap();
    assert!(matches!(
        registry.validate_profile(&registered),
        Err(FrontierPlanError::InsufficientTrust { .. })
    ));

    let missing = FrontierProfile::new(
        FrontierProfileId::new("research.missing").unwrap(),
        FrontierOrdering::Search,
        vec![SolverId::new("missing.v1").unwrap()],
        SolverTrust::Experimental,
        budget,
        None,
        true,
    )
    .unwrap();
    assert!(matches!(
        registry.validate_profile(&missing),
        Err(FrontierPlanError::UnknownSolver { .. })
    ));
}
