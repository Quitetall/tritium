//! Exact experimental admission for real owner-bound QTEA work.
pub(crate) fn qtea_output_id(
    stage_request_id: ContentId,
    plan_id: ContentId,
    estimate: FrontierResourceEstimate,
    input_id: ContentId,
    tensor_id: ContentId,
) -> ContentId {
    let mut hash = blake3::Hasher::new_derive_key(CONTENT_ID_CONTEXT);
    hash.update(b"tritium frontier admitted qtea tensor output v1\0");
    for id in [
        stage_request_id,
        plan_id,
        estimate.machine_id(),
        estimate.evidence_id(),
        input_id,
        tensor_id,
    ] {
        hash.update(id.as_bytes());
    }
    ContentId::from_digest(*hash.finalize().as_bytes())
}
// CPU reference adapter and exact experimental admission; never qualification.

use std::{error::Error, fmt, io::Write, sync::Arc};

use super::{
    AdmittedSolverPlan, FRONTIER_SOLVER_ABI_V1, FrontierPlanError, FrontierResourceEstimate,
    FrontierResourceEstimator, FrontierSolver, FrontierSolverError, FrontierStage,
    FrontierStageOutcome, FrontierStageReceipt, FrontierStageRequest, ResourceDimension,
    ResourceVector, SolverDescriptor, SolverFamily, SolverId, SolverRequest, SolverTrust,
};
use crate::{
    CONTENT_ID_CONTEXT, ContentId, QteaTensorWorkError, QteaTensorWorkPlan, QteaTensorWorkResult,
};

/// Existing reserved identity, now backed by an Experimental CPU tensor fitter.
pub const FRONTIER_QTEA_REFERENCE_SOLVER_ID: &str = "qtea-salient-residual.v1";

/// Real native scalar-f32/SPD-only tensor adapter; not a model-runtime promise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QteaReferenceAdapter {
    descriptor: SolverDescriptor,
}

impl QteaReferenceAdapter {
    /// Construct the fixed Experimental descriptor; no trust promotion is exposed.
    ///
    /// # Errors
    /// Refuses invalid source-controlled identity or ABI metadata.
    pub fn new() -> Result<Self, FrontierPlanError> {
        Ok(Self {
            descriptor: SolverDescriptor::new(
                SolverId::new(FRONTIER_QTEA_REFERENCE_SOLVER_ID)?,
                SolverFamily::QteaSalientResidual,
                SolverTrust::Experimental,
                FRONTIER_SOLVER_ABI_V1,
            )?,
        })
    }
    /// Exact source-controlled implementation identity and Experimental maturity.
    pub const fn descriptor(&self) -> &SolverDescriptor {
        &self.descriptor
    }
    /// Pair with caller-owned complete resource evidence; no invented measurements.
    pub fn with_resource_estimator(
        &self,
        estimator: Arc<dyn FrontierResourceEstimator>,
    ) -> QteaFrontierSolver {
        QteaFrontierSolver {
            adapter: self.clone(),
            estimator,
        }
    }
}

/// Installable experimental planning implementation plus actual admitted execution.
#[derive(Debug)]
pub struct QteaFrontierSolver {
    adapter: QteaReferenceAdapter,
    estimator: Arc<dyn FrontierResourceEstimator>,
}
impl QteaFrontierSolver {
    /// Mathematical adapter paired with this explicit estimator.
    pub const fn adapter(&self) -> &QteaReferenceAdapter {
        &self.adapter
    }
    /// Execute only the exact selected admission through the real native writer.
    /// Does not call the estimator again, replan, publish or qualify a model.
    ///
    /// # Errors
    /// Refuses another descriptor before delegating all exact-plan/work/resource
    /// and native/codec/sink checks. A failed write yields no completed result.
    pub fn fit_admitted_tensor(
        &self,
        work: QteaTensorWorkPlan<'_>,
        plan: &AdmittedSolverPlan,
        request: &FrontierStageRequest,
        expected_plan_id: ContentId,
        sink: &mut impl Write,
    ) -> Result<QteaAdmittedTensorFitResult, QteaFrontierWorkError> {
        if plan.descriptor() != self.adapter.descriptor() {
            return Err(QteaFrontierWorkError::SolverMismatch);
        }
        work.admit(plan, request, expected_plan_id)?.fit_to(sink)
    }
}
impl FrontierSolver for QteaFrontierSolver {
    fn descriptor(&self) -> &SolverDescriptor {
        self.adapter.descriptor()
    }
    fn estimate(
        &self,
        request: &SolverRequest,
    ) -> Result<FrontierResourceEstimate, FrontierSolverError> {
        self.estimator.estimate(request)
    }
}

/// Immutable selected plan/request identities and exact inputs, not completed execution.
#[derive(Debug)]
pub struct QteaAdmittedTensorWork<'a> {
    work: QteaTensorWorkPlan<'a>,
    plan_id: ContentId,
    stage_request_id: ContentId,
    estimate: FrontierResourceEstimate,
}

impl<'a> QteaTensorWorkPlan<'a> {
    /// Check an existing experimental registry admission before native execution.
    ///
    /// V1 requests do not bind machine/evidence IDs. `expected_plan_id` must come
    /// from the caller's selected admission record, not be inferred from a received
    /// request. It prevents substitution, but does not authenticate that record.
    /// This method itself creates no descriptor or installer; use the explicit
    /// [`QteaReferenceAdapter`] with a caller estimator for builtin planning.
    ///
    /// # Errors
    /// Refuses wrong plan/descriptor/trust/stage/source/input/shape, byte-scope
    /// underestimate or overflow before fitting or touching a sink.
    pub fn admit(
        self,
        plan: &AdmittedSolverPlan,
        request: &FrontierStageRequest,
        expected_plan_id: ContentId,
    ) -> Result<QteaAdmittedTensorWork<'a>, QteaFrontierWorkError> {
        let plan_id = plan.content_id();
        if plan_id != expected_plan_id {
            return Err(QteaFrontierWorkError::PlanIdentityMismatch);
        }
        request.validate_plan(plan)?;
        let descriptor = plan.descriptor();
        if descriptor.id().as_str() != FRONTIER_QTEA_REFERENCE_SOLVER_ID
            || descriptor.family() != SolverFamily::QteaSalientResidual
            || descriptor.abi_version() != FRONTIER_SOLVER_ABI_V1
            || descriptor.trust() != SolverTrust::Experimental
        {
            return Err(QteaFrontierWorkError::SolverMismatch);
        }
        if request.stage() != FrontierStage::Fit {
            return Err(QteaFrontierWorkError::WrongStage {
                found: request.stage(),
            });
        }
        if request.source_id() != self.origin().source_id {
            return Err(QteaFrontierWorkError::SourceMismatch);
        }
        let metadata = self.metadata();
        let rows = u64::try_from(metadata.rows).map_err(|_| QteaFrontierWorkError::SizeOverflow)?;
        let columns =
            u64::try_from(metadata.columns).map_err(|_| QteaFrontierWorkError::SizeOverflow)?;
        if request.solver_request().rows() != rows || request.solver_request().columns() != columns
        {
            return Err(QteaFrontierWorkError::ShapeMismatch);
        }
        if request.input_id() != self.input_id() {
            return Err(QteaFrontierWorkError::InputIdentityMismatch);
        }
        let memory = self.memory();
        let transient = memory
            .total_logical_bytes
            .checked_sub(memory.borrowed_input_bytes)
            .ok_or(QteaFrontierWorkError::SizeOverflow)?;
        let estimate = plan.estimate();
        for (dimension, required, provided) in [
            (
                ResourceDimension::HostRamBytes,
                memory.total_logical_bytes,
                estimate.host_ram_bytes(),
            ),
            (
                ResourceDimension::TransientBytes,
                transient,
                estimate.transient_bytes(),
            ),
            (
                ResourceDimension::ArtifactBytes,
                self.byte_ledger().total_bytes,
                estimate.artifact_bytes(),
            ),
        ] {
            require_bytes(dimension, required, provided)?;
        }
        Ok(QteaAdmittedTensorWork {
            work: self,
            plan_id,
            stage_request_id: request.content_id(),
            estimate: plan.resource_estimate(),
        })
    }
}

fn require_bytes(
    dimension: ResourceDimension,
    required: usize,
    provided: u64,
) -> Result<(), QteaFrontierWorkError> {
    let required = u64::try_from(required).map_err(|_| QteaFrontierWorkError::SizeOverflow)?;
    if provided < required {
        return Err(QteaFrontierWorkError::ResourceUnderestimate {
            dimension,
            required,
            provided,
        });
    }
    Ok(())
}

impl QteaAdmittedTensorWork<'_> {
    /// Exact selected admission, including machine and estimate evidence identity.
    pub const fn plan_id(&self) -> ContentId {
        self.plan_id
    }
    /// Content identity of the exact portable request retained through execution.
    pub const fn stage_request_id(&self) -> ContentId {
        self.stage_request_id
    }

    /// Execute the already admitted native writer into a private unpublished sink.
    ///
    /// Generic Write cannot enforce sink privacy or durability. A failed/partial
    /// write returns no completed result. No resource measurements are invented.
    ///
    /// # Errors
    /// Propagates actual native/codec/writer failures without fallback or receipt.
    pub fn fit_to(
        self,
        sink: &mut impl Write,
    ) -> Result<QteaAdmittedTensorFitResult, QteaFrontierWorkError> {
        let work = self.work.fit_to(sink)?;
        let plan_id = self.plan_id;
        let stage_request_id = self.stage_request_id;
        let estimate = self.estimate;
        let output_id = qtea_output_id(
            stage_request_id,
            plan_id,
            estimate,
            work.input_id(),
            work.tensor_id(),
        );
        Ok(QteaAdmittedTensorFitResult {
            work,
            plan_id,
            stage_request_id,
            estimate,
            output_id,
        })
    }
}

/// Completed experimental write bound to actual selected admission; not qualification.
#[derive(Clone, Debug, PartialEq)]
pub struct QteaAdmittedTensorFitResult {
    work: QteaTensorWorkResult,
    plan_id: ContentId,
    stage_request_id: ContentId,
    estimate: FrontierResourceEstimate,
    output_id: ContentId,
}

impl QteaAdmittedTensorFitResult {
    /// Actual underlying tensor bytes/metadata identity and owner-bound input ID.
    pub const fn work_result(&self) -> &QteaTensorWorkResult {
        &self.work
    }
    /// Exact selected admission, not inferred from a portable V1 request.
    pub const fn plan_id(&self) -> ContentId {
        self.plan_id
    }
    /// Exact original portable request ID.
    pub const fn stage_request_id(&self) -> ContentId {
        self.stage_request_id
    }
    /// Request/plan/machine/evidence/input/file-bound stage output ID.
    pub const fn output_id(&self) -> ContentId {
        self.output_id
    }
    /// Caller-supplied admitted estimate and machine/evidence IDs, not authentication.
    pub const fn resource_estimate(&self) -> FrontierResourceEstimate {
        self.estimate
    }

    /// Construct a completed Experimental Fit receipt from explicit observations.
    ///
    /// Does not measure/approve resource use, source custody or physical memory.
    /// Measured artifact bytes cannot underreport the known raw serialized output;
    /// measured RAM need not equal the conservative logical reservation bound.
    ///
    /// # Errors
    /// Refuses a different request, artifact underreport or existing receipt/budget
    /// invariant. No receipt is returned on failure.
    pub fn completed_receipt(
        &self,
        request: &FrontierStageRequest,
        measured: ResourceVector,
    ) -> Result<FrontierStageReceipt, QteaFrontierWorkError> {
        if request.content_id() != self.stage_request_id {
            return Err(QteaFrontierWorkError::Admission(
                FrontierPlanError::StageRequestPlanMismatch {
                    field: "request_id",
                },
            ));
        }
        require_bytes(
            ResourceDimension::ArtifactBytes,
            self.work.byte_ledger().total_bytes,
            measured.artifact_bytes(),
        )?;
        let receipt = FrontierStageReceipt::new(
            request.run_id(),
            request.source_id(),
            request.stage_index(),
            request.stage(),
            request.profile_id().clone(),
            request.solver().clone(),
            request.input_id(),
            Some(self.output_id),
            request.estimate(),
            measured,
            request.budget(),
            FrontierStageOutcome::Completed,
            None,
        )?;
        request.validate_receipt(&receipt)?;
        Ok(receipt)
    }
}

/// Exact experimental admission, known byte-scope or native work refusal.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum QteaFrontierWorkError {
    /// Existing portable request/receipt/registry admission invariant failed.
    Admission(FrontierPlanError),
    /// Caller selection ID differs from the actual opaque plan.
    PlanIdentityMismatch,
    /// Wrong reserved ID, family, ABI or non-Experimental trust claim.
    SolverMismatch,
    /// Only Fit can execute native tensor work.
    WrongStage {
        /// Rejected stage.
        found: FrontierStage,
    },
    /// Campaign source differs from owner-bound work origin.
    SourceMismatch,
    /// Numerical geometry differs from admitted shape.
    ShapeMismatch,
    /// Owner/storage/numerical input identity differs from request.
    InputIdentityMismatch,
    /// Logical scope cannot be represented in the resource vector.
    SizeOverflow,
    /// Admitted estimate or reported artifact length underreports known scope.
    ResourceUnderestimate {
        /// Rejected byte dimension.
        dimension: ResourceDimension,
        /// Known declared-scope requirement.
        required: u64,
        /// Supplied byte value.
        provided: u64,
    },
    /// Native fit/export/codec/sink failure; no successful result.
    Work(QteaTensorWorkError),
}
impl fmt::Display for QteaFrontierWorkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTEA frontier work refused: {self:?}")
    }
}
impl Error for QteaFrontierWorkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Admission(e) => Some(e),
            Self::Work(e) => Some(e),
            _ => None,
        }
    }
}
impl From<FrontierPlanError> for QteaFrontierWorkError {
    fn from(e: FrontierPlanError) -> Self {
        Self::Admission(e)
    }
}
impl From<QteaTensorWorkError> for QteaFrontierWorkError {
    fn from(e: QteaTensorWorkError) -> Self {
        Self::Work(e)
    }
}
