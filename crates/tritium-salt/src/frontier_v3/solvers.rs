//! Built-in solver capability catalog and concrete native adapters.

use std::{error::Error, fmt, io::Write, sync::Arc};

use tritium_format::salt_v2_master::SaltV2MasterTensorSpec;
use tritium_quantize::{
    SaltV2Config, SaltV2Error, SaltV2RestartableTensorMasterFitInput, SaltV2TensorMasterFitInput,
    SaltV2TensorMasterFitResult, fit_salt_v2_restartable_tensor_master, fit_salt_v2_tensor_master,
    plan_salt_v2_restartable_tensor_master, plan_salt_v2_tensor_master,
};

use super::{
    AdmittedSolverPlan, FRONTIER_SOLVER_ABI_V1, FrontierPlanError, FrontierResourceEstimate,
    FrontierSolver, FrontierSolverError, FrontierStage, FrontierStageOutcome, FrontierStageReceipt,
    FrontierStageRequest, ResourceVector, SolverDescriptor, SolverFamily, SolverId, SolverRequest,
    SolverTrust,
};
use crate::ContentId;

/// Stable identity of native bounded-memory SALT V2 CPU reference fitter.
pub const FRONTIER_SALT_V2_REFERENCE_SOLVER_ID: &str = "salt.v2.cpu-reference.v1";

/// Exact reason built-in family cannot be installed or executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BuiltinSolverBlocker {
    /// Complete frontier fitter, artifact writer, and conformance evidence are missing.
    FrontierIntegrationMissing,
}

/// Truthful execution status of one built-in solver identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BuiltinSolverStatus {
    /// Native implementation exists at stated trust tier.
    Available {
        /// Review and evidence maturity for exact implementation.
        trust: SolverTrust,
    },
    /// Identity is reserved for planned work but cannot execute.
    Unavailable {
        /// Missing evidence or implementation preventing execution.
        blocker: BuiltinSolverBlocker,
    },
}

/// Fail-closed capability record for one built-in method family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltinSolverCapability {
    id: SolverId,
    family: SolverFamily,
    status: BuiltinSolverStatus,
    descriptor: Option<SolverDescriptor>,
}

impl BuiltinSolverCapability {
    fn available(
        id: &'static str,
        family: SolverFamily,
        trust: SolverTrust,
    ) -> Result<Self, FrontierPlanError> {
        let id = SolverId::new(id)?;
        let descriptor = SolverDescriptor::new(id.clone(), family, trust, FRONTIER_SOLVER_ABI_V1)?;
        Ok(Self {
            id,
            family,
            status: BuiltinSolverStatus::Available { trust },
            descriptor: Some(descriptor),
        })
    }

    fn unavailable(id: &'static str, family: SolverFamily) -> Result<Self, FrontierPlanError> {
        Ok(Self {
            id: SolverId::new(id)?,
            family,
            status: BuiltinSolverStatus::Unavailable {
                blocker: BuiltinSolverBlocker::FrontierIntegrationMissing,
            },
            descriptor: None,
        })
    }

    /// Stable candidate identity, whether available or not.
    pub const fn id(&self) -> &SolverId {
        &self.id
    }

    /// Mathematical method family.
    pub const fn family(&self) -> SolverFamily {
        self.family
    }

    /// Current implementation availability and trust.
    pub const fn status(&self) -> BuiltinSolverStatus {
        self.status
    }

    /// Installable descriptor only when native implementation exists.
    pub const fn descriptor(&self) -> Option<&SolverDescriptor> {
        self.descriptor.as_ref()
    }
}

/// Return deterministic built-in capability catalog.
///
/// Planned identities never produce installable descriptors. Custom solver
/// families remain caller-registered and therefore do not appear here.
///
/// # Errors
/// Fails if any source-controlled built-in identity violates identifier or ABI
/// invariants. Such failure indicates corrupted program metadata.
pub fn builtin_solver_capabilities() -> Result<Vec<BuiltinSolverCapability>, FrontierPlanError> {
    Ok(vec![
        BuiltinSolverCapability::available(
            FRONTIER_SALT_V2_REFERENCE_SOLVER_ID,
            SolverFamily::Salt,
            SolverTrust::Registered,
        )?,
        BuiltinSolverCapability::unavailable(
            "qtea-salient-residual.v1",
            SolverFamily::QteaSalientResidual,
        )?,
        BuiltinSolverCapability::unavailable("externd.v1", SolverFamily::ExTernD)?,
        BuiltinSolverCapability::unavailable("twla.v1", SolverFamily::Twla)?,
        BuiltinSolverCapability::unavailable("twn.v1", SolverFamily::Twn)?,
        BuiltinSolverCapability::unavailable("ttq.v1", SolverFamily::Ttq)?,
        BuiltinSolverCapability::unavailable("sparse-ternary.v1", SolverFamily::SparseTernary)?,
        BuiltinSolverCapability::unavailable(
            "folded-nine-level.v1",
            SolverFamily::FoldedNineLevel,
        )?,
    ])
}

/// Typed adapter exposing existing canonical SALT V2 reference fitter as one
/// concrete frontier-family implementation.
///
/// Adapter delegates without changing recipe, bytes, receipt, or errors. It
/// deliberately does not implement [`super::FrontierSolver`]: that trait
/// requires complete machine-specific resource estimates, which fitting logic
/// cannot honestly invent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2ReferenceAdapter {
    descriptor: SolverDescriptor,
}

impl SaltV2ReferenceAdapter {
    /// Construct adapter using source-controlled stable identity.
    ///
    /// # Errors
    /// Fails only if built-in identity or ABI metadata is invalid.
    pub fn new() -> Result<Self, FrontierPlanError> {
        Ok(Self {
            descriptor: SolverDescriptor::new(
                SolverId::new(FRONTIER_SALT_V2_REFERENCE_SOLVER_ID)?,
                SolverFamily::Salt,
                SolverTrust::Registered,
                FRONTIER_SOLVER_ABI_V1,
            )?,
        })
    }

    /// Stable identity and registered trust of native implementation.
    pub const fn descriptor(&self) -> &SolverDescriptor {
        &self.descriptor
    }

    /// Attach caller-owned machine/resource estimation, making this adapter
    /// installable in planning registry without fabricating hardware claims.
    pub fn with_resource_estimator(
        &self,
        estimator: Arc<dyn FrontierResourceEstimator>,
    ) -> SaltV2FrontierSolver {
        SaltV2FrontierSolver {
            adapter: self.clone(),
            estimator,
        }
    }

    /// Plan one ordinary tensor using canonical SALT V2 validation.
    ///
    /// # Errors
    /// Returns canonical SALT V2 planning failure unchanged.
    pub fn plan_tensor(
        &self,
        input: SaltV2TensorMasterFitInput<'_>,
        config: &SaltV2Config,
    ) -> Result<SaltV2MasterTensorSpec, SaltV2Error> {
        plan_salt_v2_tensor_master(input, config)
    }

    /// Plan one tensor from verified reopened curvature evidence.
    ///
    /// # Errors
    /// Returns canonical SALT V2 restart planning failure unchanged.
    pub fn plan_restartable_tensor(
        &self,
        input: SaltV2RestartableTensorMasterFitInput<'_>,
        config: &SaltV2Config,
    ) -> Result<SaltV2MasterTensorSpec, SaltV2Error> {
        plan_salt_v2_restartable_tensor_master(input, config)
    }
}

/// Machine-specific resource estimation supplied separately from mathematical
/// solver implementation.
pub trait FrontierResourceEstimator: std::fmt::Debug + Send + Sync {
    /// Estimate every dimension and bind supporting machine/evidence identity.
    ///
    /// # Errors
    /// Returns evidence or hardware-model failure without starting fitting.
    fn estimate(
        &self,
        request: &SolverRequest,
    ) -> Result<FrontierResourceEstimate, FrontierSolverError>;
}

/// SALT reference adapter plus explicit resource estimator, suitable for
/// [`super::SolverRegistry`] planning admission.
#[derive(Debug)]
pub struct SaltV2FrontierSolver {
    adapter: SaltV2ReferenceAdapter,
    estimator: Arc<dyn FrontierResourceEstimator>,
}

impl SaltV2FrontierSolver {
    /// Concrete fitting adapter paired with this planning object.
    pub const fn adapter(&self) -> &SaltV2ReferenceAdapter {
        &self.adapter
    }

    /// Fit one ordinary tensor only after registry and portable-request admission.
    ///
    /// `request.input_id` must equal [`Self::tensor_input_id`]. Every admission
    /// validation completes before the sink is touched. A post-fit metadata
    /// invariant can still fail after writing a complete, unpublished stream.
    ///
    /// # Errors
    /// Rejects a forged or mismatched admission/request/input binding, then
    /// propagates canonical SALT V2 fitting failures.
    pub fn fit_admitted_tensor<W: Write>(
        &self,
        plan: &AdmittedSolverPlan,
        request: &FrontierStageRequest,
        input: SaltV2TensorMasterFitInput<'_>,
        config: &SaltV2Config,
        sink: W,
    ) -> Result<SaltV2AdmittedTensorFitResult, SaltV2FrontierFitError> {
        let spec = self.adapter.plan_tensor(input, config)?;
        self.validate_fit_request(plan, request, &spec)?;
        // Canonical fitter plans internally. Keep pre-plan for admission, then
        // compare returned metadata so future fitter drift fails closed.
        let result = fit_salt_v2_tensor_master(input, config, sink)?;
        if result.spec() != &spec {
            return Err(SaltV2FrontierFitError::OutputMetadataMismatch);
        }
        Ok(SaltV2AdmittedTensorFitResult::new(plan, request, result))
    }

    /// Fit one restartable tensor only after exact admission and input binding.
    ///
    /// # Errors
    /// Rejects a forged or mismatched admission/request/input binding, then
    /// propagates canonical SALT V2 restart or fitting failures.
    pub fn fit_admitted_restartable_tensor<W: Write>(
        &self,
        plan: &AdmittedSolverPlan,
        request: &FrontierStageRequest,
        input: SaltV2RestartableTensorMasterFitInput<'_>,
        config: &SaltV2Config,
        sink: W,
    ) -> Result<SaltV2AdmittedTensorFitResult, SaltV2FrontierFitError> {
        let spec = self.adapter.plan_restartable_tensor(input, config)?;
        self.validate_fit_request(plan, request, &spec)?;
        // See ordinary path: duplicate planning separates admission from the
        // canonical fitter while its API does not accept a precomputed spec.
        let result = fit_salt_v2_restartable_tensor_master(input, config, sink)?;
        if result.spec() != &spec {
            return Err(SaltV2FrontierFitError::OutputMetadataMismatch);
        }
        Ok(SaltV2AdmittedTensorFitResult::new(plan, request, result))
    }

    fn validate_fit_request(
        &self,
        plan: &AdmittedSolverPlan,
        request: &FrontierStageRequest,
        spec: &SaltV2MasterTensorSpec,
    ) -> Result<(), SaltV2FrontierFitError> {
        request.validate_plan(plan)?;
        if plan.descriptor() != self.adapter.descriptor() {
            return Err(SaltV2FrontierFitError::SolverMismatch);
        }
        if request.stage() != FrontierStage::Fit {
            return Err(SaltV2FrontierFitError::WrongStage {
                found: request.stage(),
            });
        }
        let [rows, columns] = spec.shape() else {
            return Err(SaltV2FrontierFitError::ShapeMismatch);
        };
        if request.solver_request().rows() != *rows
            || request.solver_request().columns() != *columns
        {
            return Err(SaltV2FrontierFitError::ShapeMismatch);
        }
        if request.input_id() != self.tensor_input_id(plan, spec)? {
            return Err(SaltV2FrontierFitError::InputIdentityMismatch);
        }
        Ok(())
    }

    /// Bind opaque admission identity to canonical planned tensor metadata.
    ///
    /// # Errors
    /// Rejects a plan for another solver or invalid canonical metadata.
    pub fn tensor_input_id(
        &self,
        plan: &AdmittedSolverPlan,
        spec: &SaltV2MasterTensorSpec,
    ) -> Result<ContentId, SaltV2FrontierFitError> {
        if plan.descriptor() != self.adapter.descriptor() {
            return Err(SaltV2FrontierFitError::SolverMismatch);
        }
        let metadata = spec.canonical_bytes().map_err(SaltV2Error::from)?;
        let mut identity = Vec::with_capacity(80 + metadata.len());
        identity.extend_from_slice(b"tritium frontier admitted salt tensor input v1\0");
        identity.extend_from_slice(plan.content_id().as_bytes());
        identity.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        identity.extend_from_slice(&metadata);
        Ok(ContentId::of_bytes(&identity))
    }
}

/// Completed canonical tensor fit bound to one admitted frontier request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2AdmittedTensorFitResult {
    stage_request_id: ContentId,
    resource_estimate: FrontierResourceEstimate,
    output_id: ContentId,
    result: SaltV2TensorMasterFitResult,
}

impl SaltV2AdmittedTensorFitResult {
    fn new(
        plan: &AdmittedSolverPlan,
        request: &FrontierStageRequest,
        result: SaltV2TensorMasterFitResult,
    ) -> Self {
        let resource_estimate = plan.resource_estimate();
        let mut identity = Vec::with_capacity(160);
        identity.extend_from_slice(b"tritium frontier admitted salt tensor output v1\0");
        identity.extend_from_slice(request.content_id().as_bytes());
        identity.extend_from_slice(plan.content_id().as_bytes());
        identity.extend_from_slice(resource_estimate.machine_id().as_bytes());
        identity.extend_from_slice(resource_estimate.evidence_id().as_bytes());
        identity.extend_from_slice(&result.receipt().tensor_master_id());
        let output_id = ContentId::of_bytes(&identity);
        Self {
            stage_request_id: request.content_id(),
            resource_estimate,
            output_id,
            result,
        }
    }

    /// Exact portable request identity admitted before fitting.
    pub const fn stage_request_id(&self) -> ContentId {
        self.stage_request_id
    }

    /// Frontier content identity wrapping the canonical tensor-master identity.
    pub const fn output_id(&self) -> ContentId {
        self.output_id
    }

    /// Exact machine inventory identity supporting admitted estimate.
    pub const fn machine_id(&self) -> ContentId {
        self.resource_estimate.machine_id()
    }

    /// Exact evidence identity supporting admitted estimate.
    pub const fn estimate_evidence_id(&self) -> ContentId {
        self.resource_estimate.evidence_id()
    }

    /// Canonical SALT V2 metadata and payload receipt.
    pub const fn fit_result(&self) -> &SaltV2TensorMasterFitResult {
        &self.result
    }

    /// Produce a completed Fit-stage receipt after caller measures resources.
    ///
    /// # Errors
    /// Rejects a different stage request or incoherent measured budget claim.
    pub fn completed_receipt(
        &self,
        request: &FrontierStageRequest,
        measured: ResourceVector,
    ) -> Result<FrontierStageReceipt, FrontierPlanError> {
        if request.content_id() != self.stage_request_id {
            return Err(FrontierPlanError::StageRequestPlanMismatch {
                field: "request_id",
            });
        }
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

/// Admission or canonical-fit failure from frontier SALT execution.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum SaltV2FrontierFitError {
    /// Portable request differs from opaque registry admission.
    Admission(FrontierPlanError),
    /// Admission names another concrete solver implementation.
    SolverMismatch,
    /// Request attempts to execute a non-Fit pipeline stage.
    WrongStage {
        /// Rejected stage.
        found: FrontierStage,
    },
    /// Admitted matrix shape differs from canonical tensor metadata.
    ShapeMismatch,
    /// Request input identity differs from canonical planned metadata.
    InputIdentityMismatch,
    /// Canonical fitter returned metadata differing from its admitted plan.
    OutputMetadataMismatch,
    /// Canonical SALT V2 planning, fitting, or sink failure.
    Fit(SaltV2Error),
}

impl fmt::Display for SaltV2FrontierFitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Admission(source) => write!(formatter, "frontier fit admission failed: {source}"),
            Self::SolverMismatch => formatter.write_str("frontier fit solver does not match plan"),
            Self::WrongStage { found } => {
                write!(
                    formatter,
                    "frontier fit request has non-fit stage {found:?}"
                )
            }
            Self::ShapeMismatch => {
                formatter.write_str("frontier fit tensor shape does not match admitted request")
            }
            Self::InputIdentityMismatch => {
                formatter.write_str("frontier fit input does not match canonical planned metadata")
            }
            Self::OutputMetadataMismatch => {
                formatter.write_str("frontier fit output metadata differs from admitted plan")
            }
            Self::Fit(source) => write!(formatter, "frontier SALT V2 fit failed: {source}"),
        }
    }
}

impl Error for SaltV2FrontierFitError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Admission(source) => Some(source),
            Self::Fit(source) => Some(source),
            _ => None,
        }
    }
}

impl From<FrontierPlanError> for SaltV2FrontierFitError {
    fn from(value: FrontierPlanError) -> Self {
        Self::Admission(value)
    }
}

impl From<SaltV2Error> for SaltV2FrontierFitError {
    fn from(value: SaltV2Error) -> Self {
        Self::Fit(value)
    }
}

impl FrontierSolver for SaltV2FrontierSolver {
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
