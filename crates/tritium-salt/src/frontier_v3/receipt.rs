//! Terminal stage evidence and Pareto-selection contracts for SALT V3.

use serde::{Deserialize, Serialize};

use super::{
    FrontierPlanError, FrontierProfile, FrontierProfileId, ResourceDimension, ResourceVector,
    SolverDescriptor, artifact::content_id_text, valid_identifier,
};
use crate::ContentId;

/// Stable serialized schema emitted by current V3 stage-receipt writers.
pub const FRONTIER_STAGE_RECEIPT_SCHEMA_V1: &str = "tritium.frontier-stage-receipt.v1";

/// Stable serialized schema emitted by current V3 Pareto-receipt writers.
pub const FRONTIER_PARETO_RECEIPT_SCHEMA_V1: &str = "tritium.frontier-pareto-receipt.v1";

/// Pipeline stage bound by one terminal receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum FrontierStage {
    /// Validate source identity and model contract.
    AdmitSource,
    /// Capture calibration or sensitivity evidence.
    CaptureEvidence,
    /// Fit one exact solver recipe.
    Fit,
    /// Encode persistent artifact bytes.
    Pack,
    /// Load and validate runtime representation.
    Load,
    /// Measure quality, runtime, or physical properties.
    Evaluate,
}

/// Terminal state supported by a stage receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontierStageOutcome {
    /// Output exists and measured resources remain within bound budget.
    Completed,
    /// Stage ran, but measured resources exceeded at least one bound.
    BudgetExceeded,
    /// Stage failed without a usable output.
    Failed,
}

/// Immutable evidence for one terminal solver stage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    try_from = "FrontierStageReceiptWire",
    into = "FrontierStageReceiptWire"
)]
pub struct FrontierStageReceipt {
    run_id: ContentId,
    source_id: ContentId,
    stage_index: u32,
    stage: FrontierStage,
    profile_id: FrontierProfileId,
    solver: SolverDescriptor,
    input_id: ContentId,
    output_id: Option<ContentId>,
    estimate: ResourceVector,
    measured: ResourceVector,
    budget: ResourceVector,
    outcome: FrontierStageOutcome,
    diagnostic: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrontierStageReceiptWire {
    schema: String,
    #[serde(with = "content_id_text")]
    run_id: ContentId,
    #[serde(with = "content_id_text")]
    source_id: ContentId,
    stage_index: u32,
    stage: FrontierStage,
    profile_id: FrontierProfileId,
    solver: SolverDescriptor,
    #[serde(with = "content_id_text")]
    input_id: ContentId,
    #[serde(default, with = "optional_content_id_text")]
    output_id: Option<ContentId>,
    estimate: ResourceVector,
    measured: ResourceVector,
    budget: ResourceVector,
    outcome: FrontierStageOutcome,
    diagnostic: Option<String>,
}

impl FrontierStageReceipt {
    /// Construct terminal evidence, rejecting false success and false budget claims.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: ContentId,
        source_id: ContentId,
        stage_index: u32,
        stage: FrontierStage,
        profile_id: FrontierProfileId,
        solver: SolverDescriptor,
        input_id: ContentId,
        output_id: Option<ContentId>,
        estimate: ResourceVector,
        measured: ResourceVector,
        budget: ResourceVector,
        outcome: FrontierStageOutcome,
        diagnostic: Option<String>,
    ) -> Result<Self, FrontierPlanError> {
        require_digest("run_id", run_id)?;
        require_digest("source_id", source_id)?;
        require_digest("input_id", input_id)?;
        if let Some(output_id) = output_id {
            require_digest("output_id", output_id)?;
        }
        if diagnostic.as_ref().is_some_and(|value| !valid_text(value)) {
            return Err(FrontierPlanError::InvalidReceiptText {
                field: "diagnostic",
            });
        }
        if budget.runtime_latency_micros().is_some() && estimate.runtime_latency_micros().is_none()
        {
            return Err(FrontierPlanError::IncompleteEstimate {
                solver_id: solver.id().clone(),
                dimension: ResourceDimension::RuntimeLatencyMicros,
            });
        }
        if budget.runtime_latency_micros().is_some() && measured.runtime_latency_micros().is_none()
        {
            return Err(FrontierPlanError::IncompleteStageMeasurement {
                dimension: ResourceDimension::RuntimeLatencyMicros,
            });
        }
        let estimate_violations = budget.violations(estimate);
        if !estimate_violations.is_empty() {
            return Err(FrontierPlanError::UnadmittedStageEstimate {
                violations: estimate_violations,
            });
        }
        let measured_exceeded = !budget.violations(measured).is_empty();
        let coherent = match outcome {
            FrontierStageOutcome::Completed => {
                output_id.is_some() && diagnostic.is_none() && !measured_exceeded
            }
            FrontierStageOutcome::BudgetExceeded => diagnostic.is_some() && measured_exceeded,
            FrontierStageOutcome::Failed => output_id.is_none() && diagnostic.is_some(),
        };
        if !coherent {
            return Err(FrontierPlanError::IncoherentStageOutcome { outcome });
        }
        Ok(Self {
            run_id,
            source_id,
            stage_index,
            stage,
            profile_id,
            solver,
            input_id,
            output_id,
            estimate,
            measured,
            budget,
            outcome,
            diagnostic,
        })
    }

    /// Content identity of canonical JSON for lineage and Pareto binding.
    pub fn content_id(&self) -> ContentId {
        let bytes = serde_json::to_vec(self).expect("frontier stage receipt serializes");
        ContentId::of_bytes(&bytes)
    }

    /// Campaign or fitting-run identity.
    pub const fn run_id(&self) -> ContentId {
        self.run_id
    }

    /// Exact dense source identity shared by all stages in this run.
    pub const fn source_id(&self) -> ContentId {
        self.source_id
    }

    /// Monotonic stage index assigned by caller.
    pub const fn stage_index(&self) -> u32 {
        self.stage_index
    }

    /// Stage represented by this receipt.
    pub const fn stage(&self) -> FrontierStage {
        self.stage
    }

    /// Profile governing stage admission.
    pub const fn profile_id(&self) -> &FrontierProfileId {
        &self.profile_id
    }

    /// Exact solver version used by stage.
    pub const fn solver(&self) -> &SolverDescriptor {
        &self.solver
    }

    /// Input evidence or artifact identity.
    pub const fn input_id(&self) -> ContentId {
        self.input_id
    }

    /// Output identity, absent on failed stages.
    pub const fn output_id(&self) -> Option<ContentId> {
        self.output_id
    }

    /// Resource vector admitted before execution.
    pub const fn estimate(&self) -> ResourceVector {
        self.estimate
    }

    /// Resource vector measured during execution.
    pub const fn measured(&self) -> ResourceVector {
        self.measured
    }

    /// Hard budget bound into this receipt.
    pub const fn budget(&self) -> ResourceVector {
        self.budget
    }

    /// Terminal stage outcome.
    pub const fn outcome(&self) -> FrontierStageOutcome {
        self.outcome
    }

    /// Bounded failure or violation diagnostic.
    pub fn diagnostic(&self) -> Option<&str> {
        self.diagnostic.as_deref()
    }
}

impl TryFrom<FrontierStageReceiptWire> for FrontierStageReceipt {
    type Error = FrontierPlanError;

    fn try_from(value: FrontierStageReceiptWire) -> Result<Self, Self::Error> {
        if value.schema != FRONTIER_STAGE_RECEIPT_SCHEMA_V1 {
            return Err(FrontierPlanError::UnsupportedReceiptSchema {
                kind: "stage",
                found: value.schema,
                supported: FRONTIER_STAGE_RECEIPT_SCHEMA_V1,
            });
        }
        Self::new(
            value.run_id,
            value.source_id,
            value.stage_index,
            value.stage,
            value.profile_id,
            value.solver,
            value.input_id,
            value.output_id,
            value.estimate,
            value.measured,
            value.budget,
            value.outcome,
            value.diagnostic,
        )
    }
}

impl From<FrontierStageReceipt> for FrontierStageReceiptWire {
    fn from(value: FrontierStageReceipt) -> Self {
        Self {
            schema: FRONTIER_STAGE_RECEIPT_SCHEMA_V1.to_owned(),
            run_id: value.run_id,
            source_id: value.source_id,
            stage_index: value.stage_index,
            stage: value.stage,
            profile_id: value.profile_id,
            solver: value.solver,
            input_id: value.input_id,
            output_id: value.output_id,
            estimate: value.estimate,
            measured: value.measured,
            budget: value.budget,
            outcome: value.outcome,
            diagnostic: value.diagnostic,
        }
    }
}

/// Optimization direction for one exact integer objective.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FrontierObjectiveDirection {
    /// Smaller values dominate larger values.
    Minimize,
    /// Larger values dominate smaller values.
    Maximize,
}

/// Canonical Pareto objective identity, direction, and unit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    try_from = "FrontierObjectiveSpecWire",
    into = "FrontierObjectiveSpecWire"
)]
pub struct FrontierObjectiveSpec {
    id: String,
    direction: FrontierObjectiveDirection,
    unit: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrontierObjectiveSpecWire {
    id: String,
    direction: FrontierObjectiveDirection,
    unit: String,
}

impl FrontierObjectiveSpec {
    /// Construct a bounded objective definition.
    pub fn new(
        id: impl Into<String>,
        direction: FrontierObjectiveDirection,
        unit: impl Into<String>,
    ) -> Result<Self, FrontierPlanError> {
        let id = id.into();
        if !valid_identifier(&id) {
            return Err(FrontierPlanError::InvalidReceiptText {
                field: "objective id",
            });
        }
        let unit = unit.into();
        if !valid_short_text(&unit) {
            return Err(FrontierPlanError::InvalidReceiptText {
                field: "objective unit",
            });
        }
        Ok(Self {
            id,
            direction,
            unit,
        })
    }

    /// Stable objective identity.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Optimization direction.
    pub const fn direction(&self) -> FrontierObjectiveDirection {
        self.direction
    }

    /// Exact unit for integer values.
    pub fn unit(&self) -> &str {
        &self.unit
    }
}

impl TryFrom<FrontierObjectiveSpecWire> for FrontierObjectiveSpec {
    type Error = FrontierPlanError;

    fn try_from(value: FrontierObjectiveSpecWire) -> Result<Self, Self::Error> {
        Self::new(value.id, value.direction, value.unit)
    }
}

impl From<FrontierObjectiveSpec> for FrontierObjectiveSpecWire {
    fn from(value: FrontierObjectiveSpec) -> Self {
        Self {
            id: value.id,
            direction: value.direction,
            unit: value.unit,
        }
    }
}

/// One candidate's exact fixed-point or integer objective value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    try_from = "FrontierObjectiveValueWire",
    into = "FrontierObjectiveValueWire"
)]
pub struct FrontierObjectiveValue {
    objective_id: String,
    value: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrontierObjectiveValueWire {
    objective_id: String,
    value: i64,
}

impl FrontierObjectiveValue {
    /// Construct an exact objective value.
    pub fn new(objective_id: impl Into<String>, value: i64) -> Result<Self, FrontierPlanError> {
        let objective_id = objective_id.into();
        if !valid_identifier(&objective_id) {
            return Err(FrontierPlanError::InvalidReceiptText {
                field: "objective value id",
            });
        }
        Ok(Self {
            objective_id,
            value,
        })
    }

    /// Bound objective identity.
    pub fn objective_id(&self) -> &str {
        &self.objective_id
    }

    /// Exact signed integer value in objective unit.
    pub const fn value(&self) -> i64 {
        self.value
    }
}

impl TryFrom<FrontierObjectiveValueWire> for FrontierObjectiveValue {
    type Error = FrontierPlanError;

    fn try_from(value: FrontierObjectiveValueWire) -> Result<Self, Self::Error> {
        Self::new(value.objective_id, value.value)
    }
}

impl From<FrontierObjectiveValue> for FrontierObjectiveValueWire {
    fn from(value: FrontierObjectiveValue) -> Self {
        Self {
            objective_id: value.objective_id,
            value: value.value,
        }
    }
}

/// One independently evidenced candidate on a declared Pareto frontier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    try_from = "FrontierParetoCandidateWire",
    into = "FrontierParetoCandidateWire"
)]
pub struct FrontierParetoCandidate {
    artifact_id: ContentId,
    stage_receipt_id: ContentId,
    objectives: Vec<FrontierObjectiveValue>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrontierParetoCandidateWire {
    #[serde(with = "content_id_text")]
    artifact_id: ContentId,
    #[serde(with = "content_id_text")]
    stage_receipt_id: ContentId,
    objectives: Vec<FrontierObjectiveValue>,
}

impl FrontierParetoCandidate {
    /// Construct a candidate with strict lexical objective order.
    pub fn new(
        artifact_id: ContentId,
        stage_receipt_id: ContentId,
        objectives: Vec<FrontierObjectiveValue>,
    ) -> Result<Self, FrontierPlanError> {
        require_digest("artifact_id", artifact_id)?;
        require_digest("stage_receipt_id", stage_receipt_id)?;
        if objectives.is_empty() {
            return Err(FrontierPlanError::EmptyObjectiveValues);
        }
        validate_objective_order(objectives.iter().map(FrontierObjectiveValue::objective_id))?;
        Ok(Self {
            artifact_id,
            stage_receipt_id,
            objectives,
        })
    }

    /// Candidate artifact identity.
    pub const fn artifact_id(&self) -> ContentId {
        self.artifact_id
    }

    /// Identity of terminal evaluation-stage receipt.
    pub const fn stage_receipt_id(&self) -> ContentId {
        self.stage_receipt_id
    }

    /// Objective values in canonical definition order.
    pub fn objectives(&self) -> &[FrontierObjectiveValue] {
        &self.objectives
    }
}

impl TryFrom<FrontierParetoCandidateWire> for FrontierParetoCandidate {
    type Error = FrontierPlanError;

    fn try_from(value: FrontierParetoCandidateWire) -> Result<Self, Self::Error> {
        Self::new(value.artifact_id, value.stage_receipt_id, value.objectives)
    }
}

impl From<FrontierParetoCandidate> for FrontierParetoCandidateWire {
    fn from(value: FrontierParetoCandidate) -> Self {
        Self {
            artifact_id: value.artifact_id,
            stage_receipt_id: value.stage_receipt_id,
            objectives: value.objectives,
        }
    }
}

/// Explicit selection state for a validated Pareto frontier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "kebab-case", deny_unknown_fields)]
pub enum FrontierSelection {
    /// No candidate has been selected.
    Pending,
    /// External decision selected one candidate and supplied its evidence identity.
    Manual {
        /// Selected candidate artifact.
        #[serde(with = "content_id_text")]
        artifact_id: ContentId,
        /// Content identity of external decision evidence.
        #[serde(with = "content_id_text")]
        decision_id: ContentId,
    },
    /// Profile-authorized deterministic policy selected one candidate.
    Automatic {
        /// Selected candidate artifact.
        #[serde(with = "content_id_text")]
        artifact_id: ContentId,
        /// Exact selection policy identity.
        #[serde(with = "content_id_text")]
        policy_id: ContentId,
    },
}

/// Canonical receipt containing only nondominated candidates and explicit selection state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    try_from = "FrontierParetoReceiptWire",
    into = "FrontierParetoReceiptWire"
)]
pub struct FrontierParetoReceipt {
    run_id: ContentId,
    source_id: ContentId,
    profile_id: FrontierProfileId,
    auto_select: bool,
    objectives: Vec<FrontierObjectiveSpec>,
    candidates: Vec<FrontierParetoCandidate>,
    selection: FrontierSelection,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrontierParetoReceiptWire {
    schema: String,
    #[serde(with = "content_id_text")]
    run_id: ContentId,
    #[serde(with = "content_id_text")]
    source_id: ContentId,
    profile_id: FrontierProfileId,
    auto_select: bool,
    objectives: Vec<FrontierObjectiveSpec>,
    candidates: Vec<FrontierParetoCandidate>,
    selection: FrontierSelection,
}

impl FrontierParetoReceipt {
    /// Construct a profile-bound, nondominated frontier receipt.
    pub fn new(
        run_id: ContentId,
        source_id: ContentId,
        profile: &FrontierProfile,
        objectives: Vec<FrontierObjectiveSpec>,
        candidates: Vec<FrontierParetoCandidate>,
        selection: FrontierSelection,
    ) -> Result<Self, FrontierPlanError> {
        Self::from_parts(
            run_id,
            source_id,
            profile.id().clone(),
            profile.auto_select(),
            objectives,
            candidates,
            selection,
        )
    }

    fn from_parts(
        run_id: ContentId,
        source_id: ContentId,
        profile_id: FrontierProfileId,
        auto_select: bool,
        objectives: Vec<FrontierObjectiveSpec>,
        candidates: Vec<FrontierParetoCandidate>,
        selection: FrontierSelection,
    ) -> Result<Self, FrontierPlanError> {
        require_digest("run_id", run_id)?;
        require_digest("source_id", source_id)?;
        if objectives.is_empty() {
            return Err(FrontierPlanError::EmptyObjectiveSet);
        }
        validate_objective_order(objectives.iter().map(FrontierObjectiveSpec::id))?;
        if candidates.is_empty() {
            return Err(FrontierPlanError::EmptyParetoCandidateSet);
        }
        for pair in candidates.windows(2) {
            if pair[0].artifact_id().as_bytes() >= pair[1].artifact_id().as_bytes() {
                return Err(FrontierPlanError::NonCanonicalParetoCandidateOrder {
                    previous: pair[0].artifact_id(),
                    current: pair[1].artifact_id(),
                });
            }
        }
        let expected: Vec<&str> = objectives.iter().map(FrontierObjectiveSpec::id).collect();
        for candidate in &candidates {
            let actual: Vec<&str> = candidate
                .objectives()
                .iter()
                .map(FrontierObjectiveValue::objective_id)
                .collect();
            if actual != expected {
                return Err(FrontierPlanError::ObjectiveSetMismatch {
                    artifact_id: candidate.artifact_id(),
                });
            }
        }
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            for (other_index, other) in candidates.iter().enumerate() {
                if candidate_index != other_index && dominates(other, candidate, &objectives) {
                    return Err(FrontierPlanError::DominatedParetoCandidate {
                        dominated: candidate.artifact_id(),
                        dominator: other.artifact_id(),
                    });
                }
            }
        }
        let selected = match &selection {
            FrontierSelection::Pending => None,
            FrontierSelection::Manual {
                artifact_id,
                decision_id,
            } => {
                require_digest("decision_id", *decision_id)?;
                Some(*artifact_id)
            }
            FrontierSelection::Automatic {
                artifact_id,
                policy_id,
            } => {
                if !auto_select {
                    return Err(FrontierPlanError::AutomaticSelectionForbidden { profile_id });
                }
                require_digest("policy_id", *policy_id)?;
                Some(*artifact_id)
            }
        };
        if let Some(selected) = selected
            && !candidates
                .iter()
                .any(|candidate| candidate.artifact_id() == selected)
        {
            return Err(FrontierPlanError::UnknownSelectedCandidate {
                artifact_id: selected,
            });
        }
        Ok(Self {
            run_id,
            source_id,
            profile_id,
            auto_select,
            objectives,
            candidates,
            selection,
        })
    }

    /// Campaign or fitting-run identity.
    pub const fn run_id(&self) -> ContentId {
        self.run_id
    }

    /// Exact dense source identity.
    pub const fn source_id(&self) -> ContentId {
        self.source_id
    }

    /// Profile governing this frontier.
    pub const fn profile_id(&self) -> &FrontierProfileId {
        &self.profile_id
    }

    /// Whether bound profile permits automatic selection.
    pub const fn auto_select(&self) -> bool {
        self.auto_select
    }

    /// Canonically ordered objective definitions.
    pub fn objectives(&self) -> &[FrontierObjectiveSpec] {
        &self.objectives
    }

    /// Canonically ordered, nondominated candidates.
    pub fn candidates(&self) -> &[FrontierParetoCandidate] {
        &self.candidates
    }

    /// Explicit selection state.
    pub const fn selection(&self) -> &FrontierSelection {
        &self.selection
    }

    /// Selected artifact identity, if selection is terminal.
    pub const fn selected_artifact_id(&self) -> Option<ContentId> {
        match self.selection {
            FrontierSelection::Pending => None,
            FrontierSelection::Manual { artifact_id, .. }
            | FrontierSelection::Automatic { artifact_id, .. } => Some(artifact_id),
        }
    }
}

impl TryFrom<FrontierParetoReceiptWire> for FrontierParetoReceipt {
    type Error = FrontierPlanError;

    fn try_from(value: FrontierParetoReceiptWire) -> Result<Self, Self::Error> {
        if value.schema != FRONTIER_PARETO_RECEIPT_SCHEMA_V1 {
            return Err(FrontierPlanError::UnsupportedReceiptSchema {
                kind: "Pareto",
                found: value.schema,
                supported: FRONTIER_PARETO_RECEIPT_SCHEMA_V1,
            });
        }
        Self::from_parts(
            value.run_id,
            value.source_id,
            value.profile_id,
            value.auto_select,
            value.objectives,
            value.candidates,
            value.selection,
        )
    }
}

impl From<FrontierParetoReceipt> for FrontierParetoReceiptWire {
    fn from(value: FrontierParetoReceipt) -> Self {
        Self {
            schema: FRONTIER_PARETO_RECEIPT_SCHEMA_V1.to_owned(),
            run_id: value.run_id,
            source_id: value.source_id,
            profile_id: value.profile_id,
            auto_select: value.auto_select,
            objectives: value.objectives,
            candidates: value.candidates,
            selection: value.selection,
        }
    }
}

fn dominates(
    left: &FrontierParetoCandidate,
    right: &FrontierParetoCandidate,
    objectives: &[FrontierObjectiveSpec],
) -> bool {
    let mut strictly_better = false;
    for ((left, right), objective) in left
        .objectives()
        .iter()
        .zip(right.objectives())
        .zip(objectives)
    {
        let no_worse = match objective.direction() {
            FrontierObjectiveDirection::Minimize => left.value() <= right.value(),
            FrontierObjectiveDirection::Maximize => left.value() >= right.value(),
        };
        if !no_worse {
            return false;
        }
        strictly_better |= left.value() != right.value();
    }
    strictly_better
}

fn validate_objective_order<'a>(
    values: impl Iterator<Item = &'a str>,
) -> Result<(), FrontierPlanError> {
    let mut previous: Option<&str> = None;
    for current in values {
        if let Some(previous) = previous
            && previous >= current
        {
            return Err(FrontierPlanError::NonCanonicalObjectiveOrder {
                previous: previous.to_owned(),
                current: current.to_owned(),
            });
        }
        previous = Some(current);
    }
    Ok(())
}

pub(super) fn require_digest(field: &'static str, id: ContentId) -> Result<(), FrontierPlanError> {
    if id.as_bytes() == &[0_u8; 32] {
        return Err(FrontierPlanError::ZeroReceiptDigest { field });
    }
    Ok(())
}

fn valid_text(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 4096
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn valid_short_text(value: &str) -> bool {
    valid_text(value) && value.len() <= 64
}

mod optional_content_id_text {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::content_id_text;
    use crate::ContentId;

    pub(super) fn serialize<S>(id: &Option<ContentId>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match id {
            Some(id) => serializer.serialize_some(&ContentIdRef(id)),
            None => serializer.serialize_none(),
        }
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Option<ContentId>, D::Error>
    where
        D: Deserializer<'de>,
    {
        Option::<ContentIdOwned>::deserialize(deserializer).map(|value| value.map(|value| value.0))
    }

    struct ContentIdRef<'a>(&'a ContentId);

    impl Serialize for ContentIdRef<'_> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            content_id_text::serialize(self.0, serializer)
        }
    }

    struct ContentIdOwned(ContentId);

    impl<'de> Deserialize<'de> for ContentIdOwned {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            content_id_text::deserialize(deserializer).map(Self)
        }
    }
}
