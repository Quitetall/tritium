use std::sync::Arc;
use tritium_format::ModelId;
use tritium_quantize::{
    ActivationCacheBuilder, ActivationCacheSpec, ActivationChunk, ActivationDType,
    ActivationDigest, CurvatureArtifact, CurvatureSourceId, SaltV2Config,
    SaltV2RestartableTensorMasterFitInput, SaltV2TensorFitInput, SaltV2TensorMasterFitInput,
    fit_salt_v2_restartable_tensor_master, fit_salt_v2_tensor_master,
    plan_salt_v2_restartable_tensor_master, plan_salt_v2_tensor_master,
};

use tritium_salt::{
    BuiltinSolverBlocker, BuiltinSolverStatus, ContentId, FRONTIER_SALT_V2_REFERENCE_SOLVER_ID,
    FrontierOrdering, FrontierProfile, FrontierProfileId, FrontierResourceEstimator,
    FrontierSolverError, FrontierStage, FrontierStageRequest, ResourceVector,
    SaltV2FrontierFitError, SaltV2ReferenceAdapter, SolverFamily, SolverId, SolverRegistry,
    SolverRequest, SolverTrust, builtin_solver_capabilities,
};

#[derive(Debug)]
struct ExactResources(ResourceVector);

impl FrontierResourceEstimator for ExactResources {
    fn estimate(&self, _request: &SolverRequest) -> Result<ResourceVector, FrontierSolverError> {
        Ok(self.0)
    }
}

fn tensor_fixture() -> (Vec<f32>, Vec<f32>, tritium_quantize::ActivationCache) {
    let activation_spec = ActivationCacheSpec::new(
        0,
        "weight.input",
        1,
        128,
        ActivationDType::Float32,
        ActivationDigest::from_bytes([7; 32]),
        1,
    )
    .unwrap();
    let chunk =
        ActivationChunk::new(&activation_spec, 0, 1, vec![1.0; 128], vec![true], vec![1]).unwrap();
    let mut builder = ActivationCacheBuilder::new(activation_spec);
    builder.ingest(chunk).unwrap();
    let cache = builder.finalize().unwrap();
    let weights = (0..256)
        .map(|index| ((index % 17) as f32 - 8.0) / 11.0)
        .collect();
    (weights, vec![1.0; 256], cache)
}

#[test]
fn builtin_catalog_advertises_only_real_executable_solver() {
    let capabilities = builtin_solver_capabilities().unwrap();
    assert_eq!(capabilities.len(), 8);
    assert_eq!(
        capabilities[0].id().as_str(),
        FRONTIER_SALT_V2_REFERENCE_SOLVER_ID
    );
    assert_eq!(capabilities[0].family(), SolverFamily::Salt);
    assert_eq!(
        capabilities[0].status(),
        BuiltinSolverStatus::Available {
            trust: SolverTrust::Registered
        }
    );
    assert!(capabilities[0].descriptor().is_some());

    let expected_unavailable = [
        "qtea-salient-residual.v1",
        "externd.v1",
        "twla.v1",
        "twn.v1",
        "ttq.v1",
        "sparse-ternary.v1",
        "folded-nine-level.v1",
    ];
    for (capability, expected_id) in capabilities[1..].iter().zip(expected_unavailable) {
        assert_eq!(capability.id().as_str(), expected_id);
        assert_eq!(
            capability.status(),
            BuiltinSolverStatus::Unavailable {
                blocker: BuiltinSolverBlocker::FrontierIntegrationMissing
            }
        );
        assert!(capability.descriptor().is_none());
    }
}

#[test]
fn salt_reference_adapter_is_bit_identical_to_canonical_fitter() {
    let (weights, diagonal, cache) = tensor_fixture();
    let model_id = ModelId::from_digest([3; 32]);
    let source = CurvatureSourceId::new(
        *model_id.as_bytes(),
        cache.digest().into_bytes(),
        cache.spec().source_digest().into_bytes(),
    )
    .unwrap();
    let tensor = SaltV2TensorFitInput {
        name: "model.layers.0.mlp.down_proj.weight",
        weights: &weights,
        rows: 2,
        cols: 128,
        curvature: CurvatureArtifact::diagonal_fisher(source, [5; 32], &diagonal),
    };
    let input = SaltV2TensorMasterFitInput {
        tensor,
        activations: &cache,
        source_model_id: model_id,
        tensor_index: 0,
        source_tensor_digest: [9; 32],
    };
    let config = SaltV2Config::default();
    let adapter = SaltV2ReferenceAdapter::new().unwrap();

    assert_eq!(adapter.descriptor().family(), SolverFamily::Salt);
    assert_eq!(adapter.descriptor().trust(), SolverTrust::Registered);
    assert_eq!(
        adapter.plan_tensor(input, &config).unwrap(),
        plan_salt_v2_tensor_master(input, &config).unwrap()
    );

    let mut direct = Vec::new();
    let direct_result = fit_salt_v2_tensor_master(input, &config, &mut direct).unwrap();
    let budget = ResourceVector::new(
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        None,
    );
    let estimate = ResourceVector::new(1024, 0, 1024, 1024, 1024, 1024, 10, None);
    let solver_id = SolverId::new(FRONTIER_SALT_V2_REFERENCE_SOLVER_ID).unwrap();
    let profile = FrontierProfile::new(
        FrontierProfileId::new("test.salt-admitted").unwrap(),
        FrontierOrdering::Fixed,
        vec![solver_id.clone()],
        SolverTrust::Registered,
        budget,
        None,
        false,
    )
    .unwrap();
    let solver = Arc::new(adapter.with_resource_estimator(Arc::new(ExactResources(estimate))));
    let mut registry = SolverRegistry::new();
    registry.register(solver.clone()).unwrap();
    let solver_request = profile.request(2, 128).unwrap();
    let plan = registry
        .plan(&solver_id, SolverTrust::Registered, &solver_request)
        .unwrap();
    let planned_spec = adapter.plan_tensor(input, &config).unwrap();
    let input_id = ContentId::of_bytes(&planned_spec.canonical_bytes().unwrap());
    let request = FrontierStageRequest::new(
        ContentId::of_bytes(b"run"),
        ContentId::of_bytes(b"source"),
        0,
        FrontierStage::Fit,
        &profile,
        &plan,
        input_id,
    )
    .unwrap();

    let mut adapted = Vec::new();
    let adapted_result = solver
        .fit_admitted_tensor(&plan, &request, input, &config, &mut adapted)
        .unwrap();
    assert_eq!(adapted, direct);
    assert_eq!(adapted_result.fit_result(), &direct_result);
    assert_eq!(
        adapted_result.fit_result().spec().evidence().recipe_id,
        config.master_recipe_id()
    );
    assert_eq!(adapted_result.stage_request_id(), request.content_id());
    let receipt = adapted_result
        .completed_receipt(&request, estimate)
        .unwrap();
    assert_eq!(receipt.output_id(), Some(adapted_result.output_id()));
    request.validate_receipt(&receipt).unwrap();

    let restartable = SaltV2RestartableTensorMasterFitInput {
        tensor,
        source_model_id: model_id,
        tensor_index: 0,
        source_tensor_digest: [9; 32],
    };
    assert_eq!(
        adapter
            .plan_restartable_tensor(restartable, &config)
            .unwrap(),
        plan_salt_v2_restartable_tensor_master(restartable, &config).unwrap()
    );
    let mut direct_restart = Vec::new();
    let direct_restart_result =
        fit_salt_v2_restartable_tensor_master(restartable, &config, &mut direct_restart).unwrap();
    let mut adapted_restart = Vec::new();
    let adapted_restart_result = solver
        .fit_admitted_restartable_tensor(
            &plan,
            &request,
            restartable,
            &config,
            &mut adapted_restart,
        )
        .unwrap();
    assert_eq!(adapted_restart, direct_restart);
    assert_eq!(adapted_restart_result.fit_result(), &direct_restart_result);

    let wrong_stage = FrontierStageRequest::new(
        ContentId::of_bytes(b"run"),
        ContentId::of_bytes(b"source"),
        0,
        FrontierStage::Pack,
        &profile,
        &plan,
        input_id,
    )
    .unwrap();
    let mut rejected = Vec::new();
    assert!(matches!(
        solver.fit_admitted_tensor(&plan, &wrong_stage, input, &config, &mut rejected),
        Err(SaltV2FrontierFitError::WrongStage {
            found: FrontierStage::Pack
        })
    ));
    assert!(rejected.is_empty());

    let changed_config = SaltV2Config {
        coordinate_sweeps: config.coordinate_sweeps + 1,
        ..config
    };
    assert!(matches!(
        solver.fit_admitted_tensor(&plan, &request, input, &changed_config, &mut rejected),
        Err(SaltV2FrontierFitError::InputIdentityMismatch)
    ));
    assert!(rejected.is_empty());

    let alternate_estimate = ResourceVector::new(2048, 0, 1024, 1024, 1024, 1024, 10, None);
    let alternate_solver =
        Arc::new(adapter.with_resource_estimator(Arc::new(ExactResources(alternate_estimate))));
    let mut alternate_registry = SolverRegistry::new();
    alternate_registry.register(alternate_solver).unwrap();
    let alternate_plan = alternate_registry
        .plan(&solver_id, SolverTrust::Registered, &solver_request)
        .unwrap();
    assert!(matches!(
        solver.fit_admitted_tensor(&alternate_plan, &request, input, &config, &mut rejected),
        Err(SaltV2FrontierFitError::Admission(
            tritium_salt::FrontierPlanError::StageRequestPlanMismatch { field: "estimate" }
        ))
    ));
    assert!(rejected.is_empty());
}

#[test]
fn salt_adapter_registers_only_with_explicit_resource_estimator() {
    let resources = ResourceVector::new(11, 0, 13, 17, 19, 23, 29, None);
    let adapter = SaltV2ReferenceAdapter::new().unwrap();
    let mut registry = SolverRegistry::new();
    registry
        .register(Arc::new(
            adapter.with_resource_estimator(Arc::new(ExactResources(resources))),
        ))
        .unwrap();
    let request =
        SolverRequest::new(2, 128, ResourceVector::new(11, 0, 13, 17, 19, 23, 29, None)).unwrap();
    let plan = registry
        .plan(
            &tritium_salt::SolverId::new(FRONTIER_SALT_V2_REFERENCE_SOLVER_ID).unwrap(),
            SolverTrust::Registered,
            &request,
        )
        .unwrap();
    assert_eq!(plan.estimate(), resources);
    assert_eq!(plan.descriptor(), adapter.descriptor());
}
