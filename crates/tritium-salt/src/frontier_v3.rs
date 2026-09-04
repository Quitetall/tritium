//! Experimental SALT V3 solver-planning contracts.
//!
//! This module admits a solver only after its stable identity, ABI, trust tier,
//! and complete resource estimate have been checked. It deliberately contains
//! no automatic fallback: callers must name and submit any fallback as a new
//! planning request.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    sync::Arc,
};

use serde::{Deserialize, Serialize};

mod artifact;

pub use artifact::{
    ArtifactByteLedger, ArtifactClaim, ByteBreakdown, FRONTIER_ARTIFACT_SCHEMA_V1,
    FrontierArtifactError, FrontierArtifactManifest, FrontierTensorArtifact, TensorRepresentation,
};

/// Object-safe solver planning ABI supported by this release.
pub const FRONTIER_SOLVER_ABI_V1: u16 = 1;

/// Stable serialized schema for experimental V3 profiles.
pub const FRONTIER_PROFILE_SCHEMA_V1: &str = "tritium.frontier-profile.v1";

/// Stable, canonical solver identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct SolverId(String);

impl SolverId {
    /// Parse a lowercase ASCII identifier containing letters, digits, `.`, `_`, or `-`.
    pub fn new(value: impl Into<String>) -> Result<Self, FrontierPlanError> {
        let value = value.into();
        if !valid_identifier(&value) {
            return Err(FrontierPlanError::InvalidSolverId { value });
        }
        Ok(Self(value))
    }

    /// Canonical string form used by profiles and receipts.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SolverId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl TryFrom<String> for SolverId {
    type Error = FrontierPlanError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<SolverId> for String {
    fn from(value: SolverId) -> Self {
        value.0
    }
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.as_bytes()[0].is_ascii_lowercase()
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')
        })
}

/// Mathematical family implemented by a solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum SolverFamily {
    /// Sensitivity-allocated additive ternary planes.
    Salt,
    /// Salience-aware ternary residual compensation.
    QteaSalientResidual,
    /// Expanded-rank two-sided ternary factorization.
    ExTernD,
    /// Activation-aware rotation and mixed-precision method family.
    Twla,
    /// Ternary Weight Networks baseline family.
    Twn,
    /// Trained Ternary Quantization baseline family.
    Ttq,
    /// Sparse-ternary hybrid family.
    SparseTernary,
    /// Ratio-three folded nine-level execution family.
    FoldedNineLevel,
    /// Registered external family whose semantics live in its versioned recipe.
    Custom,
}

/// Review and evidence maturity attached to one exact solver version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SolverTrust {
    /// Research-only implementation with no registry evidence requirement.
    Experimental,
    /// Registered implementation with provenance and contract tests.
    Registered,
    /// Independently qualified implementation for certified profiles.
    Certified,
}

/// Stable, canonical V3 profile identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct FrontierProfileId(String);

impl FrontierProfileId {
    /// Parse a lowercase ASCII profile identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, FrontierPlanError> {
        let value = value.into();
        if !valid_identifier(&value) {
            return Err(FrontierPlanError::InvalidProfileId { value });
        }
        Ok(Self(value))
    }

    /// Canonical string form used by manifests and receipts.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FrontierProfileId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl TryFrom<String> for FrontierProfileId {
    type Error = FrontierPlanError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<FrontierProfileId> for String {
    fn from(value: FrontierProfileId) -> Self {
        value.0
    }
}

/// Whether a profile searches candidate order or executes one frozen order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontierOrdering {
    /// Search candidate composition and order with complete trial receipts.
    Search,
    /// Execute solver IDs in their declared order without searching alternatives.
    Fixed,
}

/// Immutable identity and maturity metadata for one solver implementation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SolverDescriptorWire")]
pub struct SolverDescriptor {
    id: SolverId,
    family: SolverFamily,
    trust: SolverTrust,
    abi_version: u16,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SolverDescriptorWire {
    id: SolverId,
    family: SolverFamily,
    trust: SolverTrust,
    abi_version: u16,
}

impl TryFrom<SolverDescriptorWire> for SolverDescriptor {
    type Error = FrontierPlanError;

    fn try_from(value: SolverDescriptorWire) -> Result<Self, Self::Error> {
        Self::new(value.id, value.family, value.trust, value.abi_version)
    }
}

impl SolverDescriptor {
    /// Construct a descriptor for the currently supported object-safe ABI.
    pub fn new(
        id: SolverId,
        family: SolverFamily,
        trust: SolverTrust,
        abi_version: u16,
    ) -> Result<Self, FrontierPlanError> {
        if abi_version != FRONTIER_SOLVER_ABI_V1 {
            return Err(FrontierPlanError::UnsupportedSolverAbi {
                solver_id: id,
                found: abi_version,
                supported: FRONTIER_SOLVER_ABI_V1,
            });
        }
        Ok(Self {
            id,
            family,
            trust,
            abi_version,
        })
    }

    /// Stable solver identity.
    pub fn id(&self) -> &SolverId {
        &self.id
    }

    /// Mathematical method family.
    pub const fn family(&self) -> SolverFamily {
        self.family
    }

    /// Current trust tier.
    pub const fn trust(&self) -> SolverTrust {
        self.trust
    }

    /// Object-safe solver ABI version.
    pub const fn abi_version(&self) -> u16 {
        self.abi_version
    }
}

/// One hard-budget or estimated-resource dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ResourceDimension {
    /// Peak host memory.
    HostRamBytes,
    /// Peak accelerator memory.
    VramBytes,
    /// Durable and temporary disk footprint.
    DiskBytes,
    /// Final serialized artifact bytes.
    ArtifactBytes,
    /// Steady-state resident artifact bytes.
    ResidentBytes,
    /// Peak transient bytes beyond steady-state residency.
    TransientBytes,
    /// End-to-end fitting wall time.
    FittingMillis,
    /// Runtime latency when relevant to the requested profile.
    RuntimeLatencyMicros,
}

/// Complete hard budget or solver resource estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceVector {
    host_ram_bytes: u64,
    vram_bytes: u64,
    disk_bytes: u64,
    artifact_bytes: u64,
    resident_bytes: u64,
    transient_bytes: u64,
    fitting_millis: u64,
    runtime_latency_micros: Option<u64>,
}

impl ResourceVector {
    /// Construct an exact vector. Zero explicitly means no capacity or no use.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        host_ram_bytes: u64,
        vram_bytes: u64,
        disk_bytes: u64,
        artifact_bytes: u64,
        resident_bytes: u64,
        transient_bytes: u64,
        fitting_millis: u64,
        runtime_latency_micros: Option<u64>,
    ) -> Self {
        Self {
            host_ram_bytes,
            vram_bytes,
            disk_bytes,
            artifact_bytes,
            resident_bytes,
            transient_bytes,
            fitting_millis,
            runtime_latency_micros,
        }
    }

    /// Peak host memory bytes.
    pub const fn host_ram_bytes(self) -> u64 {
        self.host_ram_bytes
    }

    /// Peak accelerator memory bytes.
    pub const fn vram_bytes(self) -> u64 {
        self.vram_bytes
    }

    /// Total disk bytes.
    pub const fn disk_bytes(self) -> u64 {
        self.disk_bytes
    }

    /// Final artifact bytes.
    pub const fn artifact_bytes(self) -> u64 {
        self.artifact_bytes
    }

    /// Steady-state resident bytes.
    pub const fn resident_bytes(self) -> u64 {
        self.resident_bytes
    }

    /// Peak transient bytes.
    pub const fn transient_bytes(self) -> u64 {
        self.transient_bytes
    }

    /// Fitting wall time in milliseconds.
    pub const fn fitting_millis(self) -> u64 {
        self.fitting_millis
    }

    /// Runtime latency bound or estimate when applicable.
    pub const fn runtime_latency_micros(self) -> Option<u64> {
        self.runtime_latency_micros
    }

    fn violations(self, required: Self) -> Vec<ResourceViolation> {
        let mut violations = Vec::new();
        compare(
            &mut violations,
            ResourceDimension::HostRamBytes,
            required.host_ram_bytes,
            self.host_ram_bytes,
        );
        compare(
            &mut violations,
            ResourceDimension::VramBytes,
            required.vram_bytes,
            self.vram_bytes,
        );
        compare(
            &mut violations,
            ResourceDimension::DiskBytes,
            required.disk_bytes,
            self.disk_bytes,
        );
        compare(
            &mut violations,
            ResourceDimension::ArtifactBytes,
            required.artifact_bytes,
            self.artifact_bytes,
        );
        compare(
            &mut violations,
            ResourceDimension::ResidentBytes,
            required.resident_bytes,
            self.resident_bytes,
        );
        compare(
            &mut violations,
            ResourceDimension::TransientBytes,
            required.transient_bytes,
            self.transient_bytes,
        );
        compare(
            &mut violations,
            ResourceDimension::FittingMillis,
            required.fitting_millis,
            self.fitting_millis,
        );
        if let (Some(required), Some(available)) =
            (required.runtime_latency_micros, self.runtime_latency_micros)
        {
            compare(
                &mut violations,
                ResourceDimension::RuntimeLatencyMicros,
                required,
                available,
            );
        }
        violations
    }
}

fn compare(
    violations: &mut Vec<ResourceViolation>,
    dimension: ResourceDimension,
    required: u64,
    available: u64,
) {
    if required > available {
        violations.push(ResourceViolation {
            dimension,
            required,
            available,
        });
    }
}

/// Exact reason one resource estimate exceeded its hard ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceViolation {
    dimension: ResourceDimension,
    required: u64,
    available: u64,
}

impl ResourceViolation {
    /// Exceeded resource dimension.
    pub const fn dimension(&self) -> ResourceDimension {
        self.dimension
    }

    /// Solver-estimated requirement.
    pub const fn required(&self) -> u64 {
        self.required
    }

    /// Caller-declared hard ceiling.
    pub const fn available(&self) -> u64 {
        self.available
    }
}

/// Versioned V3 policy selecting candidate solvers and hard resource ceilings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "FrontierProfileWire", into = "FrontierProfileWire")]
pub struct FrontierProfile {
    id: FrontierProfileId,
    ordering: FrontierOrdering,
    solver_ids: Vec<SolverId>,
    minimum_trust: SolverTrust,
    budget: ResourceVector,
    fallback_profile: Option<FrontierProfileId>,
    auto_select: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrontierProfileWire {
    schema: String,
    id: FrontierProfileId,
    ordering: FrontierOrdering,
    solver_ids: Vec<SolverId>,
    minimum_trust: SolverTrust,
    budget: ResourceVector,
    fallback_profile: Option<FrontierProfileId>,
    auto_select: bool,
}

impl FrontierProfile {
    /// Construct a profile, rejecting empty or ambiguous solver policy.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: FrontierProfileId,
        ordering: FrontierOrdering,
        solver_ids: Vec<SolverId>,
        minimum_trust: SolverTrust,
        budget: ResourceVector,
        fallback_profile: Option<FrontierProfileId>,
        auto_select: bool,
    ) -> Result<Self, FrontierPlanError> {
        if solver_ids.is_empty() {
            return Err(FrontierPlanError::EmptySolverSet { profile_id: id });
        }
        let mut unique = BTreeSet::new();
        for solver_id in &solver_ids {
            if !unique.insert(solver_id) {
                return Err(FrontierPlanError::DuplicateProfileSolver {
                    profile_id: id,
                    solver_id: solver_id.clone(),
                });
            }
        }
        if fallback_profile.as_ref() == Some(&id) {
            return Err(FrontierPlanError::SelfReferentialFallback { profile_id: id });
        }
        Ok(Self {
            id,
            ordering,
            solver_ids,
            minimum_trust,
            budget,
            fallback_profile,
            auto_select,
        })
    }

    /// Stable profile identity.
    pub const fn id(&self) -> &FrontierProfileId {
        &self.id
    }

    /// Search or fixed-order execution policy.
    pub const fn ordering(&self) -> FrontierOrdering {
        self.ordering
    }

    /// Candidate solvers, preserving declared order.
    pub fn solver_ids(&self) -> &[SolverId] {
        &self.solver_ids
    }

    /// Minimum admitted maturity for every candidate.
    pub const fn minimum_trust(&self) -> SolverTrust {
        self.minimum_trust
    }

    /// Complete hard resource ceilings.
    pub const fn budget(&self) -> ResourceVector {
        self.budget
    }

    /// Explicit alternate profile identity, if caller supplied one.
    ///
    /// [`SolverRegistry`] does not resolve profile identities or follow this
    /// reference. A profile catalog must resolve and validate it when the
    /// caller explicitly submits that alternate profile as a new request.
    pub const fn fallback_profile(&self) -> Option<&FrontierProfileId> {
        self.fallback_profile.as_ref()
    }

    /// Whether validated Pareto output may be selected automatically.
    pub const fn auto_select(&self) -> bool {
        self.auto_select
    }

    /// Bind this profile's hard budget to one checked matrix shape.
    pub fn request(&self, rows: u64, columns: u64) -> Result<SolverRequest, FrontierPlanError> {
        SolverRequest::new(rows, columns, self.budget)
    }
}

impl TryFrom<FrontierProfileWire> for FrontierProfile {
    type Error = FrontierPlanError;

    fn try_from(value: FrontierProfileWire) -> Result<Self, Self::Error> {
        if value.schema != FRONTIER_PROFILE_SCHEMA_V1 {
            return Err(FrontierPlanError::UnsupportedProfileSchema {
                found: value.schema,
                supported: FRONTIER_PROFILE_SCHEMA_V1,
            });
        }
        Self::new(
            value.id,
            value.ordering,
            value.solver_ids,
            value.minimum_trust,
            value.budget,
            value.fallback_profile,
            value.auto_select,
        )
    }
}

impl From<FrontierProfile> for FrontierProfileWire {
    fn from(value: FrontierProfile) -> Self {
        Self {
            schema: FRONTIER_PROFILE_SCHEMA_V1.to_owned(),
            id: value.id,
            ordering: value.ordering,
            solver_ids: value.solver_ids,
            minimum_trust: value.minimum_trust,
            budget: value.budget,
            fallback_profile: value.fallback_profile,
            auto_select: value.auto_select,
        }
    }
}

/// Shape and hard resource ceilings supplied to one planning attempt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "SolverRequestWire", into = "SolverRequestWire")]
pub struct SolverRequest {
    rows: u64,
    columns: u64,
    elements: u64,
    budget: ResourceVector,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SolverRequestWire {
    rows: u64,
    columns: u64,
    budget: ResourceVector,
}

impl TryFrom<SolverRequestWire> for SolverRequest {
    type Error = FrontierPlanError;

    fn try_from(value: SolverRequestWire) -> Result<Self, Self::Error> {
        Self::new(value.rows, value.columns, value.budget)
    }
}

impl From<SolverRequest> for SolverRequestWire {
    fn from(value: SolverRequest) -> Self {
        Self {
            rows: value.rows,
            columns: value.columns,
            budget: value.budget,
        }
    }
}

impl SolverRequest {
    /// Construct a request, rejecting empty or overflowing tensor shapes.
    pub fn new(rows: u64, columns: u64, budget: ResourceVector) -> Result<Self, FrontierPlanError> {
        let Some(elements) = rows.checked_mul(columns) else {
            return Err(FrontierPlanError::InvalidShape { rows, columns });
        };
        if elements == 0 {
            return Err(FrontierPlanError::InvalidShape { rows, columns });
        }
        Ok(Self {
            rows,
            columns,
            elements,
            budget,
        })
    }

    /// Matrix rows.
    pub const fn rows(&self) -> u64 {
        self.rows
    }

    /// Matrix columns.
    pub const fn columns(&self) -> u64 {
        self.columns
    }

    /// Checked product of rows and columns.
    pub const fn elements(&self) -> u64 {
        self.elements
    }

    /// Caller-declared hard resource ceilings.
    pub const fn budget(&self) -> ResourceVector {
        self.budget
    }
}

/// Solver-side planning failure before hard-budget admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontierSolverError {
    message: String,
}

impl FrontierSolverError {
    /// Construct a solver failure with a non-empty diagnostic.
    pub fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            message: if message.is_empty() {
                "solver planning failed".to_owned()
            } else {
                message
            },
        }
    }
}

impl fmt::Display for FrontierSolverError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for FrontierSolverError {}

/// Object-safe planning seam implemented by native or registered solvers.
pub trait FrontierSolver: fmt::Debug + Send + Sync {
    /// Stable descriptor for this exact implementation.
    fn descriptor(&self) -> &SolverDescriptor;

    /// Estimate complete resources without mutating artifacts or starting fitting.
    fn estimate(&self, request: &SolverRequest) -> Result<ResourceVector, FrontierSolverError>;
}

/// Successful plan admitted against caller budget and trust requirements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmittedSolverPlan {
    descriptor: SolverDescriptor,
    request: SolverRequest,
    estimate: ResourceVector,
}

impl AdmittedSolverPlan {
    /// Solver selected for this exact attempt.
    pub fn solver_id(&self) -> &SolverId {
        self.descriptor.id()
    }

    /// Full admitted solver descriptor.
    pub const fn descriptor(&self) -> &SolverDescriptor {
        &self.descriptor
    }

    /// Shape and hard budget bound into this plan.
    pub const fn request(&self) -> &SolverRequest {
        &self.request
    }

    /// Exact estimate checked against the hard budget.
    pub const fn estimate(&self) -> ResourceVector {
        self.estimate
    }
}

/// Deterministic registry of explicitly installed solver implementations.
#[derive(Debug, Default)]
pub struct SolverRegistry {
    solvers: BTreeMap<SolverId, Arc<dyn FrontierSolver>>,
}

impl SolverRegistry {
    /// Construct an empty registry. No built-in capability is advertised implicitly.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one exact solver implementation, refusing identity replacement.
    pub fn register(&mut self, solver: Arc<dyn FrontierSolver>) -> Result<(), FrontierPlanError> {
        let descriptor = solver.descriptor();
        if descriptor.abi_version() != FRONTIER_SOLVER_ABI_V1 {
            return Err(FrontierPlanError::UnsupportedSolverAbi {
                solver_id: descriptor.id().clone(),
                found: descriptor.abi_version(),
                supported: FRONTIER_SOLVER_ABI_V1,
            });
        }
        if self.solvers.contains_key(descriptor.id()) {
            return Err(FrontierPlanError::DuplicateSolver {
                solver_id: descriptor.id().clone(),
            });
        }
        self.solvers.insert(descriptor.id().clone(), solver);
        Ok(())
    }

    /// Registered solver IDs in deterministic lexical order.
    pub fn ids(&self) -> Vec<&str> {
        self.solvers.keys().map(SolverId::as_str).collect()
    }

    /// Validate that every profile member is installed at the required trust tier.
    pub fn validate_profile(&self, profile: &FrontierProfile) -> Result<(), FrontierPlanError> {
        for solver_id in profile.solver_ids() {
            let solver =
                self.solvers
                    .get(solver_id)
                    .ok_or_else(|| FrontierPlanError::UnknownSolver {
                        solver_id: solver_id.clone(),
                    })?;
            let actual = solver.descriptor().trust();
            if actual < profile.minimum_trust() {
                return Err(FrontierPlanError::InsufficientTrust {
                    solver_id: solver_id.clone(),
                    actual,
                    required: profile.minimum_trust(),
                });
            }
        }
        Ok(())
    }

    /// Plan one explicitly named solver with no automatic fallback.
    pub fn plan(
        &self,
        solver_id: &SolverId,
        minimum_trust: SolverTrust,
        request: &SolverRequest,
    ) -> Result<AdmittedSolverPlan, FrontierPlanError> {
        let solver =
            self.solvers
                .get(solver_id)
                .ok_or_else(|| FrontierPlanError::UnknownSolver {
                    solver_id: solver_id.clone(),
                })?;
        let descriptor = solver.descriptor();
        if descriptor.trust() < minimum_trust {
            return Err(FrontierPlanError::InsufficientTrust {
                solver_id: solver_id.clone(),
                actual: descriptor.trust(),
                required: minimum_trust,
            });
        }
        let estimate =
            solver
                .estimate(request)
                .map_err(|source| FrontierPlanError::SolverEstimate {
                    solver_id: solver_id.clone(),
                    source,
                })?;
        if request.budget().runtime_latency_micros().is_some()
            && estimate.runtime_latency_micros().is_none()
        {
            return Err(FrontierPlanError::IncompleteEstimate {
                solver_id: solver_id.clone(),
                dimension: ResourceDimension::RuntimeLatencyMicros,
            });
        }
        let violations = request.budget().violations(estimate);
        if !violations.is_empty() {
            return Err(FrontierPlanError::BudgetExceeded {
                solver_id: solver_id.clone(),
                violations,
            });
        }
        Ok(AdmittedSolverPlan {
            descriptor: descriptor.clone(),
            request: request.clone(),
            estimate,
        })
    }
}

/// Fail-closed V3 registry and resource-admission error.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrontierPlanError {
    /// Solver identity is empty, non-canonical, or too long.
    InvalidSolverId {
        /// Rejected input.
        value: String,
    },
    /// Profile identity is empty, non-canonical, or too long.
    InvalidProfileId {
        /// Rejected input.
        value: String,
    },
    /// Serialized profile uses an unsupported schema.
    UnsupportedProfileSchema {
        /// Schema found on input.
        found: String,
        /// Schema supported by this reader.
        supported: &'static str,
    },
    /// Profile declares no candidate solver.
    EmptySolverSet {
        /// Invalid profile identity.
        profile_id: FrontierProfileId,
    },
    /// Profile repeats one solver identity.
    DuplicateProfileSolver {
        /// Invalid profile identity.
        profile_id: FrontierProfileId,
        /// Repeated solver identity.
        solver_id: SolverId,
    },
    /// Profile names itself as its direct fallback.
    SelfReferentialFallback {
        /// Invalid profile identity.
        profile_id: FrontierProfileId,
    },
    /// Solver descriptor targets an unsupported object-safe ABI.
    UnsupportedSolverAbi {
        /// Exact solver identity.
        solver_id: SolverId,
        /// Requested ABI version.
        found: u16,
        /// Supported ABI version.
        supported: u16,
    },
    /// Registry already contains this exact identity.
    DuplicateSolver {
        /// Duplicate identity.
        solver_id: SolverId,
    },
    /// Requested solver is not explicitly installed.
    UnknownSolver {
        /// Missing identity.
        solver_id: SolverId,
    },
    /// Solver maturity is below caller policy.
    InsufficientTrust {
        /// Exact solver identity.
        solver_id: SolverId,
        /// Registered maturity.
        actual: SolverTrust,
        /// Caller-required maturity.
        required: SolverTrust,
    },
    /// Tensor shape is empty or its element count overflows.
    InvalidShape {
        /// Requested rows.
        rows: u64,
        /// Requested columns.
        columns: u64,
    },
    /// Solver could not produce a complete estimate.
    SolverEstimate {
        /// Exact solver identity.
        solver_id: SolverId,
        /// Solver diagnostic.
        source: FrontierSolverError,
    },
    /// Solver omitted a dimension constrained by the caller.
    IncompleteEstimate {
        /// Exact solver identity.
        solver_id: SolverId,
        /// Missing required dimension.
        dimension: ResourceDimension,
    },
    /// Estimate exceeded one or more hard resource ceilings.
    BudgetExceeded {
        /// Exact solver identity.
        solver_id: SolverId,
        /// Every exceeded dimension in stable schema order.
        violations: Vec<ResourceViolation>,
    },
}

impl fmt::Display for FrontierPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSolverId { value } => write!(formatter, "invalid solver id {value:?}"),
            Self::InvalidProfileId { value } => write!(formatter, "invalid profile id {value:?}"),
            Self::UnsupportedProfileSchema { found, supported } => write!(
                formatter,
                "unsupported frontier profile schema {found:?}; supported schema is {supported}"
            ),
            Self::EmptySolverSet { profile_id } => {
                write!(formatter, "frontier profile {profile_id} has no solvers")
            }
            Self::DuplicateProfileSolver {
                profile_id,
                solver_id,
            } => write!(
                formatter,
                "frontier profile {profile_id} repeats solver {solver_id}"
            ),
            Self::SelfReferentialFallback { profile_id } => write!(
                formatter,
                "frontier profile {profile_id} cannot fall back to itself"
            ),
            Self::UnsupportedSolverAbi {
                solver_id,
                found,
                supported,
            } => write!(
                formatter,
                "solver {solver_id} uses ABI {found}; supported ABI is {supported}"
            ),
            Self::DuplicateSolver { solver_id } => {
                write!(formatter, "solver {solver_id} is already registered")
            }
            Self::UnknownSolver { solver_id } => {
                write!(formatter, "solver {solver_id} is not registered")
            }
            Self::InsufficientTrust {
                solver_id,
                actual,
                required,
            } => write!(
                formatter,
                "solver {solver_id} trust {actual:?} is below required {required:?}"
            ),
            Self::InvalidShape { rows, columns } => {
                write!(formatter, "invalid solver matrix shape {rows}x{columns}")
            }
            Self::SolverEstimate { solver_id, source } => {
                write!(formatter, "solver {solver_id} estimate failed: {source}")
            }
            Self::IncompleteEstimate {
                solver_id,
                dimension,
            } => write!(
                formatter,
                "solver {solver_id} omitted required estimate dimension {dimension:?}"
            ),
            Self::BudgetExceeded {
                solver_id,
                violations,
            } => write!(
                formatter,
                "solver {solver_id} exceeds {} hard resource budget(s)",
                violations.len()
            ),
        }
    }
}

impl Error for FrontierPlanError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SolverEstimate { source, .. } => Some(source),
            _ => None,
        }
    }
}
