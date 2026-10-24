//! Real native execution with synthetic planning-only fixtures, not hardware evidence.
use std::{
    io::{self, Write},
    sync::Arc,
};
use tritium_format::qtea::QteaLimits;
use tritium_quantize::qtea_reference::{TensorFit, TensorFitConfig};
use tritium_quantize::{CurvatureSourceId, InputGram, InputGramAccumulator};
use tritium_salt::*;

struct Fixture {
    weights: Vec<f32>,
    gram: InputGram,
}
impl Fixture {
    fn new() -> Self {
        let mut builder = InputGramAccumulator::new_bound(
            4,
            CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap(),
            0,
        )
        .unwrap();
        let mut x = [0.; 16];
        for c in 0..4 {
            x[c * 4 + c] = 1.;
        }
        builder.accumulate_batch(&x, 4, None, None).unwrap();
        Self {
            weights: [-1., 1., -2., 2.].repeat(4),
            gram: builder.finish().unwrap(),
        }
    }
    fn work(&self) -> QteaTensorWorkPlan<'_> {
        self.work_limited(4096)
    }
    fn work_limited(&self, max_tensor_bytes: usize) -> QteaTensorWorkPlan<'_> {
        QteaTensorWorkPlan::new(
            TensorFit::plan(
                &self.weights,
                4,
                4,
                &self.gram,
                TensorFitConfig {
                    group_size: 2,
                    salient_ratio: 0.25,
                    ..Default::default()
                },
                TensorFit::required_work_bytes(4, 4).unwrap(),
            )
            .unwrap(),
            QteaTensorOrigin {
                source_id: ContentId::of_bytes(b"synthetic campaign"),
                owner: "model.weight",
                tensor_index: 0,
                source_tensor_digest: [9; 32],
            },
            QteaLimits {
                max_tensor_bytes,
                max_validation_bytes: 4,
            },
            usize::MAX,
        )
        .unwrap()
    }
}

#[test]
fn raising_operational_capacity_requires_sufficient_admitted_resources() {
    let f = Fixture::new();
    let old = f.work();
    let (plan, request) = setup(&old, |_| {});
    let larger = f.work_limited(4097);
    assert_eq!(old.input_id(), larger.input_id());
    assert_eq!(old.byte_ledger(), larger.byte_ledger());
    assert!(matches!(
        larger.admit(&plan, &request, plan.content_id()),
        Err(QteaFrontierWorkError::ResourceUnderestimate {
            dimension: ResourceDimension::HostRamBytes,
            ..
        })
    ));
}

#[test]
fn unsupported_abi_refuses_before_a_descriptor_or_token_exists() {
    let descriptor = SolverDescriptor::new(
        SolverId::new("qtea-salient-residual.v1").unwrap(),
        SolverFamily::QteaSalientResidual,
        SolverTrust::Experimental,
        2,
    );
    assert!(matches!(
        descriptor,
        Err(FrontierPlanError::UnsupportedSolverAbi { found: 2, .. })
    ));
}

#[derive(Debug)]
struct PlanningOnly {
    descriptor: SolverDescriptor,
    estimate: FrontierResourceEstimate,
}
impl FrontierSolver for PlanningOnly {
    fn descriptor(&self) -> &SolverDescriptor {
        &self.descriptor
    }
    fn estimate(&self, _: &SolverRequest) -> Result<FrontierResourceEstimate, FrontierSolverError> {
        Ok(self.estimate)
    }
}
struct Setup {
    descriptor: SolverDescriptor,
    rows: u64,
    columns: u64,
    resources: ResourceVector,
    budget: ResourceVector,
    machine: ContentId,
    evidence: ContentId,
    source: ContentId,
    input: ContentId,
    stage: FrontierStage,
}
fn vector(host: u64, transient: u64, artifact: u64) -> ResourceVector {
    ResourceVector::new(host, 0, 0, artifact, 0, transient, 0, None)
}
fn setup(
    work: &QteaTensorWorkPlan<'_>,
    update: impl FnOnce(&mut Setup),
) -> (AdmittedSolverPlan, FrontierStageRequest) {
    let m = work.memory();
    let meta = work.metadata();
    let mut s = Setup {
        descriptor: SolverDescriptor::new(
            SolverId::new("qtea-salient-residual.v1").unwrap(),
            SolverFamily::QteaSalientResidual,
            SolverTrust::Experimental,
            FRONTIER_SOLVER_ABI_V1,
        )
        .unwrap(),
        rows: u64::try_from(meta.rows).unwrap(),
        columns: u64::try_from(meta.columns).unwrap(),
        resources: vector(
            m.total_logical_bytes as u64,
            (m.total_logical_bytes - m.borrowed_input_bytes) as u64,
            work.byte_ledger().total_bytes as u64,
        ),
        budget: ResourceVector::new(
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            None,
        ),
        machine: ContentId::of_bytes(b"synthetic machine label, not inventory"),
        evidence: ContentId::of_bytes(b"synthetic estimate, not measurements"),
        source: work.origin().source_id,
        input: work.input_id(),
        stage: FrontierStage::Fit,
    };
    update(&mut s);
    let id = s.descriptor.id().clone();
    let profile = FrontierProfile::new(
        FrontierProfileId::new("test.qtea-admission").unwrap(),
        FrontierOrdering::Fixed,
        vec![id.clone()],
        SolverTrust::Experimental,
        s.budget,
        None,
        false,
    )
    .unwrap();
    let mut registry = SolverRegistry::new();
    registry
        .register(Arc::new(PlanningOnly {
            descriptor: s.descriptor,
            estimate: FrontierResourceEstimate::new(s.resources, s.machine, s.evidence).unwrap(),
        }))
        .unwrap();
    let plan = registry
        .plan(
            &id,
            SolverTrust::Experimental,
            &profile.request(s.rows, s.columns).unwrap(),
        )
        .unwrap();
    let request = FrontierStageRequest::new(
        ContentId::of_bytes(b"synthetic run"),
        s.source,
        0,
        s.stage,
        &profile,
        &plan,
        s.input,
    )
    .unwrap();
    (plan, request)
}

#[derive(Default)]
struct Probe {
    calls: usize,
    bytes: Vec<u8>,
    fail: bool,
}
impl Write for Probe {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        if self.fail && !self.bytes.is_empty() {
            return Err(io::ErrorKind::BrokenPipe.into());
        }
        let n = if self.fail { b.len().min(7) } else { b.len() };
        self.bytes.extend_from_slice(&b[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn actual_admitted_writer_preserves_bytes_and_binds_exact_selected_plan() {
    let f = Fixture::new();
    let work = f.work();
    let input = work.input_id();
    let (plan, request) = setup(&work, |_| {});
    let admitted = work.admit(&plan, &request, plan.content_id()).unwrap();
    assert_eq!(admitted.plan_id(), plan.content_id());
    assert_eq!(admitted.stage_request_id(), request.content_id());
    let mut sink = Probe::default();
    let result = admitted.fit_to(&mut sink).unwrap();
    let mut direct = Vec::new();
    let direct_result = f.work().fit_to(&mut direct).unwrap();
    assert_eq!(sink.bytes, direct);
    assert_eq!(result.work_result(), &direct_result);
    assert_eq!(result.resource_estimate(), plan.resource_estimate());
    let mut framed = b"tritium frontier admitted qtea tensor output v1\0".to_vec();
    for id in [
        request.content_id(),
        plan.content_id(),
        plan.resource_estimate().machine_id(),
        plan.resource_estimate().evidence_id(),
        input,
        ContentId::of_bytes(&sink.bytes),
    ] {
        framed.extend_from_slice(id.as_bytes());
    }
    assert_eq!(result.output_id(), ContentId::of_bytes(&framed));
    assert_ne!(result.output_id(), result.work_result().tensor_id());
    // Synthetic caller observations exercise receipt framing, not physical RAM.
    let measured = vector(1, 0, sink.bytes.len() as u64);
    let receipt = result.completed_receipt(&request, measured).unwrap();
    request.validate_receipt(&receipt).unwrap();
    assert_eq!(receipt.output_id(), Some(result.output_id()));
    assert_eq!(receipt.solver().trust(), SolverTrust::Experimental);
    assert!(matches!(
        result.completed_receipt(&request, vector(1, 0, 0)),
        Err(QteaFrontierWorkError::ResourceUnderestimate {
            dimension: ResourceDimension::ArtifactBytes,
            ..
        })
    ));
    let other = FrontierStageRequest::new(
        ContentId::of_bytes(b"other run"),
        request.source_id(),
        0,
        FrontierStage::Fit,
        request.profile(),
        &plan,
        request.input_id(),
    )
    .unwrap();
    assert!(matches!(
        result.completed_receipt(&other, measured),
        Err(QteaFrontierWorkError::Admission(_))
    ));
    let qtea = builtin_solver_capabilities()
        .unwrap()
        .into_iter()
        .find(|c| c.family() == SolverFamily::QteaSalientResidual)
        .unwrap();
    assert!(matches!(
        qtea.status(),
        BuiltinSolverStatus::Unavailable { .. }
    ));
    assert!(qtea.descriptor().is_none());
}

#[test]
fn wrong_descriptor_stage_source_shape_and_input_never_reach_writer() {
    let f = Fixture::new();
    for case in 0..8 {
        let work = f.work();
        let (plan, request) = setup(&work, |s| match case {
            0 => {
                s.descriptor = SolverDescriptor::new(
                    SolverId::new("alternate.qtea.v1").unwrap(),
                    SolverFamily::QteaSalientResidual,
                    SolverTrust::Experimental,
                    1,
                )
                .unwrap()
            }
            1 => {
                s.descriptor = SolverDescriptor::new(
                    SolverId::new("qtea-salient-residual.v1").unwrap(),
                    SolverFamily::Salt,
                    SolverTrust::Experimental,
                    1,
                )
                .unwrap()
            }
            2 => {
                s.descriptor = SolverDescriptor::new(
                    SolverId::new("qtea-salient-residual.v1").unwrap(),
                    SolverFamily::QteaSalientResidual,
                    SolverTrust::Registered,
                    1,
                )
                .unwrap()
            }
            3 => {
                s.descriptor = SolverDescriptor::new(
                    SolverId::new("qtea-salient-residual.v1").unwrap(),
                    SolverFamily::QteaSalientResidual,
                    SolverTrust::Certified,
                    1,
                )
                .unwrap()
            }
            4 => s.stage = FrontierStage::Pack,
            5 => s.source = ContentId::of_bytes(b"other campaign"),
            6 => s.rows = 8,
            7 => s.input = ContentId::of_bytes(b"other owner/numerical input"),
            _ => unreachable!(),
        });
        let mut sink = Probe::default();
        let error = work
            .admit(&plan, &request, plan.content_id())
            .and_then(|w| w.fit_to(&mut sink))
            .unwrap_err();
        match case {
            0..=3 => assert_eq!(error, QteaFrontierWorkError::SolverMismatch),
            4 => assert!(matches!(
                error,
                QteaFrontierWorkError::WrongStage {
                    found: FrontierStage::Pack
                }
            )),
            5 => assert_eq!(error, QteaFrontierWorkError::SourceMismatch),
            6 => assert_eq!(error, QteaFrontierWorkError::ShapeMismatch),
            7 => assert_eq!(error, QteaFrontierWorkError::InputIdentityMismatch),
            _ => unreachable!(),
        }
        assert_eq!(sink.calls, 0);
    }
}

#[test]
fn exact_selection_detects_machine_or_evidence_substitution_missing_from_v1_request() {
    let f = Fixture::new();
    let original = f.work();
    let (a, ra) = setup(&original, |_| {});
    for evidence in [false, true] {
        let work = f.work();
        let (b, rb) = setup(&work, |s| {
            if evidence {
                s.evidence = ContentId::of_bytes(b"different synthetic evidence");
            } else {
                s.machine = ContentId::of_bytes(b"different synthetic machine");
            }
        });
        assert_eq!(ra.content_id(), rb.content_id()); // Documented V1 wire limitation.
        assert_ne!(a.content_id(), b.content_id());
        assert_eq!(
            work.admit(&b, &ra, a.content_id()).unwrap_err(),
            QteaFrontierWorkError::PlanIdentityMismatch
        );
        let mut one = Vec::new();
        let mut two = Vec::new();
        let x = f
            .work()
            .admit(&a, &ra, a.content_id())
            .unwrap()
            .fit_to(&mut one)
            .unwrap();
        let y = f
            .work()
            .admit(&b, &rb, b.content_id())
            .unwrap()
            .fit_to(&mut two)
            .unwrap();
        assert_eq!(one, two);
        assert_eq!(x.work_result().tensor_id(), y.work_result().tensor_id());
        assert_ne!(x.output_id(), y.output_id());
    }
}

#[test]
fn known_logical_scope_floors_refuse_underestimates_not_invented_measurements() {
    let f = Fixture::new();
    for dimension in [
        ResourceDimension::HostRamBytes,
        ResourceDimension::TransientBytes,
        ResourceDimension::ArtifactBytes,
    ] {
        let work = f.work();
        let (plan, request) = setup(&work, |s| {
            let e = s.resources;
            s.resources = vector(
                e.host_ram_bytes() - u64::from(dimension == ResourceDimension::HostRamBytes),
                e.transient_bytes() - u64::from(dimension == ResourceDimension::TransientBytes),
                e.artifact_bytes() - u64::from(dimension == ResourceDimension::ArtifactBytes),
            );
        });
        let mut sink = Probe::default();
        assert!(
            matches!(work.admit(&plan,&request,plan.content_id()).and_then(|w|w.fit_to(&mut sink)),
            Err(QteaFrontierWorkError::ResourceUnderestimate {dimension:d,..}) if d==dimension)
        );
        assert_eq!(sink.calls, 0);
    }
    let work = f.work();
    let (plan, request) = setup(&work, |_| {});
    assert_eq!(plan.estimate().disk_bytes(), 0);
    assert_eq!(plan.estimate().vram_bytes(), 0);
    assert_eq!(plan.estimate().resident_bytes(), 0);
    assert_eq!(plan.estimate().fitting_millis(), 0);
    assert!(work.admit(&plan, &request, plan.content_id()).is_ok());
}

#[test]
fn actual_input_mutation_and_portable_estimate_mutation_refuse() {
    let mut f = Fixture::new();
    let (plan, request) = {
        let old = f.work();
        setup(&old, |_| {})
    };
    f.weights[0] = -1.01;
    assert_eq!(
        f.work()
            .admit(&plan, &request, plan.content_id())
            .unwrap_err(),
        QteaFrontierWorkError::InputIdentityMismatch
    );
    let f = Fixture::new();
    let work = f.work();
    let (plan, request) = setup(&work, |_| {});
    let mut wire = serde_json::to_value(&request).unwrap();
    wire["estimate"]["host_ram_bytes"] = serde_json::json!(plan.estimate().host_ram_bytes() + 1);
    let altered: FrontierStageRequest = serde_json::from_value(wire).unwrap();
    assert!(matches!(
        work.admit(&plan, &altered, plan.content_id()),
        Err(QteaFrontierWorkError::Admission(_))
    ));
}

#[test]
fn partial_write_returns_no_result_and_receipt_budget_rules_stay_active() {
    let f = Fixture::new();
    let work = f.work();
    let (plan, request) = setup(&work, |_| {});
    let mut failed = Probe {
        fail: true,
        ..Default::default()
    };
    assert!(matches!(
        work.admit(&plan, &request, plan.content_id())
            .unwrap()
            .fit_to(&mut failed),
        Err(QteaFrontierWorkError::Work(_))
    ));
    assert_eq!(failed.bytes.len(), 7);
    let work = f.work();
    let (plan, request) = setup(&work, |s| s.budget = s.resources);
    let result = work
        .admit(&plan, &request, plan.content_id())
        .unwrap()
        .fit_to(&mut Vec::new())
        .unwrap();
    let e = plan.estimate();
    assert!(matches!(
        result.completed_receipt(
            &request,
            vector(
                e.host_ram_bytes() + 1,
                e.transient_bytes(),
                e.artifact_bytes()
            )
        ),
        Err(QteaFrontierWorkError::Admission(_))
    ));
}
