//! Built-in solver capability catalog and concrete native adapters.

use std::io::Write;

use tritium_format::salt_v2_master::SaltV2MasterTensorSpec;
use tritium_quantize::{
    SaltV2Config, SaltV2Error, SaltV2RestartableTensorMasterFitInput, SaltV2TensorMasterFitInput,
    SaltV2TensorMasterFitResult, fit_salt_v2_restartable_tensor_master, fit_salt_v2_tensor_master,
    plan_salt_v2_restartable_tensor_master, plan_salt_v2_tensor_master,
};

use super::{
    FRONTIER_SOLVER_ABI_V1, FrontierPlanError, SolverDescriptor, SolverFamily, SolverId,
    SolverTrust,
};

/// Stable identity of native bounded-memory SALT V2 CPU reference fitter.
pub const FRONTIER_SALT_V2_REFERENCE_SOLVER_ID: &str = "salt.v2.cpu-reference.v1";

/// Exact reason built-in family cannot be installed or executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BuiltinSolverBlocker {
    /// Native implementation and conformance evidence do not exist yet.
    NativeImplementationMissing,
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
                blocker: BuiltinSolverBlocker::NativeImplementationMissing,
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

    /// Fit one ordinary tensor into canonical rate-free Pmax bytes.
    ///
    /// # Errors
    /// Returns canonical SALT V2 planning, fitting, or sink failure unchanged.
    pub fn fit_tensor<W: Write>(
        &self,
        input: SaltV2TensorMasterFitInput<'_>,
        config: &SaltV2Config,
        sink: W,
    ) -> Result<SaltV2TensorMasterFitResult, SaltV2Error> {
        fit_salt_v2_tensor_master(input, config, sink)
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

    /// Fit one tensor from verified reopened curvature evidence.
    ///
    /// # Errors
    /// Returns canonical SALT V2 restart, fitting, or sink failure unchanged.
    pub fn fit_restartable_tensor<W: Write>(
        &self,
        input: SaltV2RestartableTensorMasterFitInput<'_>,
        config: &SaltV2Config,
        sink: W,
    ) -> Result<SaltV2TensorMasterFitResult, SaltV2Error> {
        fit_salt_v2_restartable_tensor_master(input, config, sink)
    }
}
