//! Actual native adapter; estimator fixtures are synthetic, never hardware proof.
use std::{
    io::{self, Write},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use tritium_format::qtea::QteaLimits;
use tritium_quantize::qtea_reference::{TensorFit, TensorFitConfig};
use tritium_quantize::{CurvatureSourceId, InputGram, InputGramAccumulator};
use tritium_salt::*;

fn fixture() -> (Vec<f32>, InputGram) {
    let mut b = InputGramAccumulator::new_bound(
        4,
        CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap(),
        0,
    )
    .unwrap();
    let mut x = [0.; 16];
    for c in 0..4 {
        x[c * 4 + c] = 1.;
    }
    b.accumulate_batch(&x, 4, None, None).unwrap();
    ([-1., 1., -2., 2.].repeat(4), b.finish().unwrap())
}
fn work<'a>(weights: &'a [f32], gram: &'a InputGram) -> QteaTensorWorkPlan<'a> {
    QteaTensorWorkPlan::new(
        TensorFit::plan(
            weights,
            4,
            4,
            gram,
            TensorFitConfig {
                group_size: 2,
                ..Default::default()
            },
            TensorFit::required_work_bytes(4, 4).unwrap(),
        )
        .unwrap(),
        QteaTensorOrigin {
            source_id: ContentId::of_bytes(b"synthetic source"),
            owner: "model.weight",
            tensor_index: 0,
            source_tensor_digest: [9; 32],
        },
        QteaLimits {
            max_tensor_bytes: 4096,
            max_validation_bytes: 4,
        },
        usize::MAX,
    )
    .unwrap()
}
#[derive(Debug)]
struct Estimator {
    calls: AtomicUsize,
    seen: Mutex<Option<SolverRequest>>,
    fail: AtomicBool,
    value: FrontierResourceEstimate,
}
impl Estimator {
    fn new(w: &QteaTensorWorkPlan<'_>) -> Self {
        let m = w.memory();
        Self {
            calls: AtomicUsize::new(0),
            seen: Mutex::new(None),
            fail: AtomicBool::new(false),
            value: FrontierResourceEstimate::new(
                ResourceVector::new(
                    m.total_logical_bytes as u64,
                    0,
                    0,
                    w.byte_ledger().total_bytes as u64,
                    0,
                    (m.total_logical_bytes - m.borrowed_input_bytes) as u64,
                    0,
                    None,
                ),
                ContentId::of_bytes(b"synthetic machine label"),
                ContentId::of_bytes(b"synthetic estimator label"),
            )
            .unwrap(),
        }
    }
}
impl FrontierResourceEstimator for Estimator {
    fn estimate(&self, r: &SolverRequest) -> Result<FrontierResourceEstimate, FrontierSolverError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        *self.seen.lock().unwrap() = Some(r.clone());
        if self.fail.load(Ordering::SeqCst) {
            Err(FrontierSolverError::new("synthetic estimate refusal"))
        } else {
            Ok(self.value)
        }
    }
}
fn profile(id: SolverId, minimum: SolverTrust) -> FrontierProfile {
    FrontierProfile::new(
        FrontierProfileId::new("test.qtea-builtin").unwrap(),
        FrontierOrdering::Fixed,
        vec![id],
        minimum,
        ResourceVector::new(
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            None,
        ),
        None,
        false,
    )
    .unwrap()
}

#[test]
fn real_factory_registers_and_executes_without_reestimating_or_changing_bytes() {
    let (weights, gram) = fixture();
    let w = work(&weights, &gram);
    let adapter = QteaReferenceAdapter::new().unwrap();
    assert_eq!(
        adapter.descriptor().id().as_str(),
        FRONTIER_QTEA_REFERENCE_SOLVER_ID
    );
    assert_eq!(adapter.descriptor().trust(), SolverTrust::Experimental);
    let e = Arc::new(Estimator::new(&w));
    let solver = Arc::new(adapter.with_resource_estimator(e.clone()));
    assert_eq!(solver.adapter(), &adapter);
    assert_eq!(solver.descriptor(), adapter.descriptor());
    let mut registry = SolverRegistry::new();
    assert!(registry.ids().is_empty());
    registry.register(solver.clone()).unwrap();
    let p = profile(adapter.descriptor().id().clone(), SolverTrust::Experimental);
    let request = p.request(4, 4).unwrap();
    let plan = registry
        .plan(
            adapter.descriptor().id(),
            SolverTrust::Experimental,
            &request,
        )
        .unwrap();
    assert_eq!(e.calls.load(Ordering::SeqCst), 1);
    assert_eq!(*e.seen.lock().unwrap(), Some(request));
    assert_eq!(plan.resource_estimate(), e.value);
    let stage = FrontierStageRequest::new(
        ContentId::of_bytes(b"synthetic run"),
        w.origin().source_id,
        0,
        FrontierStage::Fit,
        &p,
        &plan,
        w.input_id(),
    )
    .unwrap();
    e.fail.store(true, Ordering::SeqCst); // An executor must use the admitted snapshot, not replan.
    let mut bytes = Vec::new();
    let actual = solver
        .fit_admitted_tensor(w, &plan, &stage, plan.content_id(), &mut bytes)
        .unwrap();
    assert_eq!(e.calls.load(Ordering::SeqCst), 1);
    let mut direct = Vec::new();
    let expected = work(&weights, &gram).fit_to(&mut direct).unwrap();
    assert_eq!(bytes, direct);
    assert_eq!(actual.work_result(), &expected);
    assert_eq!(actual.resource_estimate(), e.value);
}

#[test]
fn higher_trust_and_failed_estimation_do_not_start_native_work() {
    let (weights, gram) = fixture();
    let w = work(&weights, &gram);
    let adapter = QteaReferenceAdapter::new().unwrap();
    let e = Arc::new(Estimator::new(&w));
    let solver = Arc::new(adapter.with_resource_estimator(e.clone()));
    let mut registry = SolverRegistry::new();
    registry.register(solver).unwrap();
    for minimum in [SolverTrust::Registered, SolverTrust::Certified] {
        let p = profile(adapter.descriptor().id().clone(), minimum);
        assert!(matches!(
            registry.validate_profile(&p),
            Err(FrontierPlanError::InsufficientTrust { .. })
        ));
        assert!(matches!(
            registry.plan(
                adapter.descriptor().id(),
                minimum,
                &p.request(4, 4).unwrap()
            ),
            Err(FrontierPlanError::InsufficientTrust { .. })
        ));
    }
    assert_eq!(e.calls.load(Ordering::SeqCst), 0);
    e.fail.store(true, Ordering::SeqCst);
    let p = profile(adapter.descriptor().id().clone(), SolverTrust::Experimental);
    assert!(matches!(
        registry.plan(
            adapter.descriptor().id(),
            SolverTrust::Experimental,
            &p.request(4, 4).unwrap()
        ),
        Err(FrontierPlanError::SolverEstimate { .. })
    ));
    assert_eq!(e.calls.load(Ordering::SeqCst), 1);
}

#[derive(Default)]
struct RefusingSink {
    calls: usize,
}
impl Write for RefusingSink {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        self.calls += 1;
        Err(io::ErrorKind::BrokenPipe.into())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn factory_preserves_exact_plan_and_sink_failure_checks() {
    let (weights, gram) = fixture();
    let w = work(&weights, &gram);
    let adapter = QteaReferenceAdapter::new().unwrap();
    let e = Arc::new(Estimator::new(&w));
    let solver = Arc::new(adapter.with_resource_estimator(e));
    let mut registry = SolverRegistry::new();
    registry.register(solver.clone()).unwrap();
    let p = profile(adapter.descriptor().id().clone(), SolverTrust::Experimental);
    let plan = registry
        .plan(
            adapter.descriptor().id(),
            SolverTrust::Experimental,
            &p.request(4, 4).unwrap(),
        )
        .unwrap();
    let stage = FrontierStageRequest::new(
        ContentId::of_bytes(b"synthetic run"),
        w.origin().source_id,
        0,
        FrontierStage::Fit,
        &p,
        &plan,
        w.input_id(),
    )
    .unwrap();
    let mut sink = RefusingSink::default();
    assert_eq!(
        solver
            .fit_admitted_tensor(
                w,
                &plan,
                &stage,
                ContentId::of_bytes(b"wrong selection"),
                &mut sink
            )
            .unwrap_err(),
        QteaFrontierWorkError::PlanIdentityMismatch
    );
    assert_eq!(sink.calls, 0);
    assert!(matches!(
        solver.fit_admitted_tensor(
            work(&weights, &gram),
            &plan,
            &stage,
            plan.content_id(),
            &mut sink
        ),
        Err(QteaFrontierWorkError::Work(_))
    ));
    assert_eq!(sink.calls, 1);
}

#[test]
fn real_adapter_refuses_a_different_solver_before_touching_sink() {
    let (weights, gram) = fixture();
    let w = work(&weights, &gram);
    let e = Arc::new(Estimator::new(&w));
    let qtea = QteaReferenceAdapter::new()
        .unwrap()
        .with_resource_estimator(e.clone());
    let salt = Arc::new(
        SaltV2ReferenceAdapter::new()
            .unwrap()
            .with_resource_estimator(e),
    );
    let id = salt.adapter().descriptor().id().clone();
    let mut registry = SolverRegistry::new();
    registry.register(salt).unwrap();
    let p = profile(id.clone(), SolverTrust::Experimental);
    let plan = registry
        .plan(&id, SolverTrust::Experimental, &p.request(4, 4).unwrap())
        .unwrap();
    let stage = FrontierStageRequest::new(
        ContentId::of_bytes(b"synthetic run"),
        w.origin().source_id,
        0,
        FrontierStage::Fit,
        &p,
        &plan,
        w.input_id(),
    )
    .unwrap();
    let mut sink = RefusingSink::default();
    assert_eq!(
        qtea.fit_admitted_tensor(w, &plan, &stage, plan.content_id(), &mut sink)
            .unwrap_err(),
        QteaFrontierWorkError::SolverMismatch
    );
    assert_eq!(sink.calls, 0);
}
