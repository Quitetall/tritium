//! Portable orchestration handoff for SALT V3 stages.
//!
//! Tritium owns numeric and evidence contracts. An external orchestrator such
//! as BLUT may carry these JSON values as opaque, content-addressed artifacts,
//! but this crate never depends on that orchestrator or its execution model.

use serde::{Deserialize, Serialize};

use super::{
    AdmittedSolverPlan, FrontierPlanError, FrontierProfile, FrontierProfileId, FrontierStage,
    FrontierStageOutcome, FrontierStageReceipt, ResourceDimension, ResourceVector,
    SolverDescriptor, SolverRequest, artifact::content_id_text, receipt::require_digest,
};
use crate::ContentId;

/// Stable serialized schema for one admitted external stage invocation.
pub const FRONTIER_STAGE_REQUEST_SCHEMA_V1: &str = "tritium.frontier-stage-request.v1";

/// Stable serialized schema for one validated resumable run chain.
pub const FRONTIER_RUN_RECEIPT_SCHEMA_V1: &str = "tritium.frontier-run-receipt.v1";

/// Content-addressable request handed to one native or external stage worker.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    try_from = "FrontierStageRequestWire",
    into = "FrontierStageRequestWire"
)]
pub struct FrontierStageRequest {
    run_id: ContentId,
    source_id: ContentId,
    stage_index: u32,
    stage: FrontierStage,
    profile: FrontierProfile,
    solver: SolverDescriptor,
    input_id: ContentId,
    request: SolverRequest,
    estimate: ResourceVector,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrontierStageRequestWire {
    schema: String,
    #[serde(with = "content_id_text")]
    run_id: ContentId,
    #[serde(with = "content_id_text")]
    source_id: ContentId,
    stage_index: u32,
    stage: FrontierStage,
    profile: FrontierProfile,
    solver: SolverDescriptor,
    #[serde(with = "content_id_text")]
    input_id: ContentId,
    request: SolverRequest,
    estimate: ResourceVector,
}

impl FrontierStageRequest {
    /// Bind an already admitted plan into a portable stage invocation.
    pub fn new(
        run_id: ContentId,
        source_id: ContentId,
        stage_index: u32,
        stage: FrontierStage,
        profile: &FrontierProfile,
        plan: &AdmittedSolverPlan,
        input_id: ContentId,
    ) -> Result<Self, FrontierPlanError> {
        Self::from_parts(
            run_id,
            source_id,
            stage_index,
            stage,
            profile.clone(),
            plan.descriptor().clone(),
            input_id,
            plan.request().clone(),
            plan.estimate(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        run_id: ContentId,
        source_id: ContentId,
        stage_index: u32,
        stage: FrontierStage,
        profile: FrontierProfile,
        solver: SolverDescriptor,
        input_id: ContentId,
        request: SolverRequest,
        estimate: ResourceVector,
    ) -> Result<Self, FrontierPlanError> {
        require_digest("run_id", run_id)?;
        require_digest("source_id", source_id)?;
        require_digest("input_id", input_id)?;
        if !profile.solver_ids().contains(solver.id()) || solver.trust() < profile.minimum_trust() {
            return Err(FrontierPlanError::ProfileSolverMismatch {
                profile_id: profile.id().clone(),
                solver_id: solver.id().clone(),
            });
        }
        if request.budget() != profile.budget() {
            return Err(FrontierPlanError::ProfileBudgetMismatch {
                profile_id: profile.id().clone(),
            });
        }
        let violations = request.budget().violations(estimate);
        if !violations.is_empty() {
            return Err(FrontierPlanError::UnadmittedStageEstimate { violations });
        }
        if request.budget().runtime_latency_micros().is_some()
            && estimate.runtime_latency_micros().is_none()
        {
            return Err(FrontierPlanError::IncompleteEstimate {
                solver_id: solver.id().clone(),
                dimension: ResourceDimension::RuntimeLatencyMicros,
            });
        }
        Ok(Self {
            run_id,
            source_id,
            stage_index,
            stage,
            profile,
            solver,
            input_id,
            request,
            estimate,
        })
    }

    /// Content identity of canonical JSON for caching and durable resume.
    pub fn content_id(&self) -> ContentId {
        let bytes = serde_json::to_vec(self).expect("frontier stage request serializes");
        ContentId::of_bytes(&bytes)
    }

    /// Validate one worker receipt against every bound request field.
    pub fn validate_receipt(
        &self,
        receipt: &FrontierStageReceipt,
    ) -> Result<(), FrontierPlanError> {
        macro_rules! require_equal {
            ($field:literal, $left:expr, $right:expr) => {
                if $left != $right {
                    return Err(FrontierPlanError::StageReceiptMismatch { field: $field });
                }
            };
        }
        require_equal!("run_id", self.run_id, receipt.run_id());
        require_equal!("source_id", self.source_id, receipt.source_id());
        require_equal!("stage_index", self.stage_index, receipt.stage_index());
        require_equal!("stage", self.stage, receipt.stage());
        require_equal!("profile_id", self.profile.id(), receipt.profile_id());
        require_equal!("solver", &self.solver, receipt.solver());
        require_equal!("input_id", self.input_id, receipt.input_id());
        require_equal!("estimate", self.estimate, receipt.estimate());
        require_equal!("budget", self.request.budget(), receipt.budget());
        Ok(())
    }

    /// Build next request from one exact completed receipt.
    ///
    /// This convenience path preserves the admitted matrix shape, budget, and
    /// estimate. A stage needing a different admission must obtain a new
    /// [`AdmittedSolverPlan`] and call [`Self::new`] instead.
    pub fn next(
        &self,
        stage: FrontierStage,
        receipt: &FrontierStageReceipt,
    ) -> Result<Self, FrontierPlanError> {
        self.validate_receipt(receipt)?;
        if receipt.outcome() != FrontierStageOutcome::Completed
            || stage_rank(self.stage) >= stage_rank(stage)
        {
            return Err(FrontierPlanError::CannotAdvanceTerminalStage {
                stage_index: self.stage_index,
                outcome: receipt.outcome(),
            });
        }
        let output_id = receipt
            .output_id()
            .ok_or(FrontierPlanError::StageReceiptMismatch { field: "output_id" })?;
        require_digest("output_id", output_id)?;
        let stage_index = self
            .stage_index
            .checked_add(1)
            .ok_or(FrontierPlanError::StageIndexOverflow)?;
        Self::from_parts(
            self.run_id,
            self.source_id,
            stage_index,
            stage,
            self.profile.clone(),
            self.solver.clone(),
            output_id,
            self.request.clone(),
            self.estimate,
        )
    }

    /// Run identity.
    pub const fn run_id(&self) -> ContentId {
        self.run_id
    }

    /// Dense source identity.
    pub const fn source_id(&self) -> ContentId {
        self.source_id
    }

    /// Zero-based stage sequence index.
    pub const fn stage_index(&self) -> u32 {
        self.stage_index
    }

    /// Requested stage.
    pub const fn stage(&self) -> FrontierStage {
        self.stage
    }

    /// Complete immutable profile snapshot.
    pub const fn profile(&self) -> &FrontierProfile {
        &self.profile
    }

    /// Profile identity.
    pub const fn profile_id(&self) -> &FrontierProfileId {
        self.profile.id()
    }

    /// Exact solver implementation.
    pub const fn solver(&self) -> &SolverDescriptor {
        &self.solver
    }

    /// Input artifact or evidence identity.
    pub const fn input_id(&self) -> ContentId {
        self.input_id
    }

    /// Matrix shape and admitted hard budget.
    pub const fn solver_request(&self) -> &SolverRequest {
        &self.request
    }

    /// Resource estimate admitted before dispatch.
    pub const fn estimate(&self) -> ResourceVector {
        self.estimate
    }

    /// Hard resource budget inherited from bound profile.
    pub const fn budget(&self) -> ResourceVector {
        self.request.budget()
    }
}

impl TryFrom<FrontierStageRequestWire> for FrontierStageRequest {
    type Error = FrontierPlanError;

    fn try_from(value: FrontierStageRequestWire) -> Result<Self, Self::Error> {
        if value.schema != FRONTIER_STAGE_REQUEST_SCHEMA_V1 {
            return Err(FrontierPlanError::UnsupportedReceiptSchema {
                kind: "stage request",
                found: value.schema,
                supported: FRONTIER_STAGE_REQUEST_SCHEMA_V1,
            });
        }
        Self::from_parts(
            value.run_id,
            value.source_id,
            value.stage_index,
            value.stage,
            value.profile,
            value.solver,
            value.input_id,
            value.request,
            value.estimate,
        )
    }
}

impl From<FrontierStageRequest> for FrontierStageRequestWire {
    fn from(value: FrontierStageRequest) -> Self {
        Self {
            schema: FRONTIER_STAGE_REQUEST_SCHEMA_V1.to_owned(),
            run_id: value.run_id,
            source_id: value.source_id,
            stage_index: value.stage_index,
            stage: value.stage,
            profile: value.profile,
            solver: value.solver,
            input_id: value.input_id,
            request: value.request,
            estimate: value.estimate,
        }
    }
}

/// Validated ordered stage history carried across orchestrator retries/resume.
///
/// One run binds one immutable profile snapshot and one solver. Explicit
/// profile fallback or solver substitution starts a new run, matching the
/// registry rule that fallback is never automatic.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "FrontierRunReceiptWire", into = "FrontierRunReceiptWire")]
pub struct FrontierRunReceipt {
    profile: FrontierProfile,
    receipts: Vec<FrontierStageReceipt>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrontierRunReceiptWire {
    schema: String,
    profile: FrontierProfile,
    receipts: Vec<FrontierStageReceipt>,
}

impl FrontierRunReceipt {
    /// Validate a nonempty, contiguous, identity-stable stage chain.
    pub fn new(
        profile: &FrontierProfile,
        receipts: Vec<FrontierStageReceipt>,
    ) -> Result<Self, FrontierPlanError> {
        if receipts.is_empty() {
            return Err(FrontierPlanError::EmptyRunReceipt);
        }
        let first = &receipts[0];
        for (position, receipt) in receipts.iter().enumerate() {
            let expected_index =
                u32::try_from(position).map_err(|_| FrontierPlanError::StageIndexOverflow)?;
            if receipt.stage_index() != expected_index {
                return Err(FrontierPlanError::NonContiguousStageIndex {
                    expected: expected_index,
                    found: receipt.stage_index(),
                });
            }
            for (field, matches) in [
                ("run_id", receipt.run_id() == first.run_id()),
                ("source_id", receipt.source_id() == first.source_id()),
                ("profile_id", receipt.profile_id() == profile.id()),
                ("solver", receipt.solver() == first.solver()),
                ("budget", receipt.budget() == profile.budget()),
            ] {
                if !matches {
                    return Err(FrontierPlanError::RunReceiptIdentityDrift {
                        stage_index: receipt.stage_index(),
                        field,
                    });
                }
            }
            if !profile.solver_ids().contains(receipt.solver().id())
                || receipt.solver().trust() < profile.minimum_trust()
            {
                return Err(FrontierPlanError::ProfileSolverMismatch {
                    profile_id: profile.id().clone(),
                    solver_id: receipt.solver().id().clone(),
                });
            }
            if let Some(previous) = position.checked_sub(1).map(|index| &receipts[index]) {
                if previous.outcome() != FrontierStageOutcome::Completed {
                    return Err(FrontierPlanError::StageAfterTerminalOutcome {
                        terminal_index: previous.stage_index(),
                    });
                }
                if previous.stage() == FrontierStage::Evaluate
                    || stage_rank(previous.stage()) >= stage_rank(receipt.stage())
                {
                    return Err(FrontierPlanError::NonMonotonicStageOrder {
                        previous: previous.stage(),
                        current: receipt.stage(),
                    });
                }
                if previous.output_id() != Some(receipt.input_id()) {
                    return Err(FrontierPlanError::BrokenStageInputChain {
                        stage_index: receipt.stage_index(),
                    });
                }
            }
        }
        Ok(Self {
            profile: profile.clone(),
            receipts,
        })
    }

    /// Immutable profile snapshot governing every stage.
    pub const fn profile(&self) -> &FrontierProfile {
        &self.profile
    }

    /// Canonical stage receipts.
    pub fn receipts(&self) -> &[FrontierStageReceipt] {
        &self.receipts
    }

    /// Next zero-based stage index for resume.
    pub fn next_stage_index(&self) -> u32 {
        u32::try_from(self.receipts.len()).expect("validated receipt count fits u32")
    }

    /// Last durable output, absent when last stage failed.
    pub fn last_output_id(&self) -> Option<ContentId> {
        self.receipts
            .last()
            .and_then(FrontierStageReceipt::output_id)
    }

    /// Whether this chain may not advance.
    pub fn is_terminal(&self) -> bool {
        self.receipts.last().is_some_and(|receipt| {
            receipt.outcome() != FrontierStageOutcome::Completed
                || receipt.stage() == FrontierStage::Evaluate
        })
    }

    /// Content identity of canonical JSON for registry and lineage binding.
    pub fn content_id(&self) -> ContentId {
        let bytes = serde_json::to_vec(self).expect("frontier run receipt serializes");
        ContentId::of_bytes(&bytes)
    }
}

impl TryFrom<FrontierRunReceiptWire> for FrontierRunReceipt {
    type Error = FrontierPlanError;

    fn try_from(value: FrontierRunReceiptWire) -> Result<Self, Self::Error> {
        if value.schema != FRONTIER_RUN_RECEIPT_SCHEMA_V1 {
            return Err(FrontierPlanError::UnsupportedReceiptSchema {
                kind: "run",
                found: value.schema,
                supported: FRONTIER_RUN_RECEIPT_SCHEMA_V1,
            });
        }
        Self::new(&value.profile, value.receipts)
    }
}

impl From<FrontierRunReceipt> for FrontierRunReceiptWire {
    fn from(value: FrontierRunReceipt) -> Self {
        Self {
            schema: FRONTIER_RUN_RECEIPT_SCHEMA_V1.to_owned(),
            profile: value.profile,
            receipts: value.receipts,
        }
    }
}

const fn stage_rank(stage: FrontierStage) -> u8 {
    match stage {
        FrontierStage::AdmitSource => 0,
        FrontierStage::CaptureEvidence => 1,
        FrontierStage::Fit => 2,
        FrontierStage::Pack => 3,
        FrontierStage::Load => 4,
        FrontierStage::Evaluate => 5,
    }
}
