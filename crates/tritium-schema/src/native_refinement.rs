//! Native recovery declarations/observations, not numerical or source-truth proof.
use crate::{
    BlobId, DigestBytes, EvidenceReference, ModelId, ModelManifest, NativeRefinementPlanId,
    NativeRefinementStateId, OriginalSourceId, PackageId, TensorStorage,
};
use alloc::{collections::BTreeSet, string::String, vec::Vec};

/// Exact recorded ancestry of the one explicitly selected parent level.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeRefinementParent {
    /// Exact parent file identity, not just its model values.
    pub package: PackageId,
    /// Explicit parent level; no default or prefix inference.
    pub level: String,
    /// Actual semantic identity of that selected level.
    pub model: ModelId,
    /// Parent's recorded original observation, not official source authentication.
    pub source: OriginalSourceId,
    /// Original admitted parent run/span/root, never a resealed projection.
    pub evidence: EvidenceReference,
}
impl NativeRefinementParent {
    /// Validate names only; actual package/log admission is a separate obligation.
    pub fn validate(&self) -> Result<(), NativeRefinementError> {
        name(&self.level)?;
        name(&self.evidence.run)?;
        name(&self.evidence.span)
    }
}

/// Distinct schema1.1 refined ancestry; not Direct or a quality verdict.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeRefinementAncestry {
    /// Exact parent binding.
    pub parent: NativeRefinementParent,
    /// Separate native request identity, not a legacy PV recipe digest.
    pub plan: NativeRefinementPlanId,
    /// Actual recovery log reference; external admission is still required.
    pub evidence: EvidenceReference,
}
impl NativeRefinementAncestry {
    /// Structural names only, not verification of the referenced recovery.
    pub fn validate(&self) -> Result<(), NativeRefinementError> {
        self.parent.validate()?;
        name(&self.evidence.run)?;
        name(&self.evidence.span)
    }
}

/// Typed native AdamW settings; equations belong to the common training engine.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeAdamWRecipe {
    /// Finite positive learning rate.
    pub lr: f32,
    /// First-moment decay in [0,1), excluding negative zero.
    pub beta1: f32,
    /// Second-moment decay in [0,1), excluding negative zero.
    pub beta2: f32,
    /// Finite positive epsilon outside the square root.
    pub eps: f32,
    /// Finite nonnegative decoupled decay with lr*decay <=1.
    pub weight_decay: f32,
}
impl NativeAdamWRecipe {
    /// Check numeric configuration, not implementation/backend admission.
    pub fn validate(&self) -> Result<(), NativeRefinementError> {
        if !self.lr.is_finite()
            || self.lr <= 0.0
            || !self.eps.is_finite()
            || self.eps <= 0.0
            || [self.beta1, self.beta2]
                .iter()
                .any(|v| !v.is_finite() || !(0.0..1.0).contains(v) || v.is_sign_negative())
            || !self.weight_decay.is_finite()
            || self.weight_decay.is_sign_negative()
            || self.lr * self.weight_decay > 1.0
        {
            return Err(NativeRefinementError::Recipe);
        }
        Ok(())
    }
}

/// Declared common recovery update, not the legacy Python diagonal refiner.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub enum NativeRecoveryRecipe {
    /// Codes/allocation remain frozen; only actual scales are updated.
    ScaleOnly {
        /// Actual continuous update settings.
        optimizer: NativeAdamWRecipe,
    },
    /// Alternating actual-scale and bounded logical-code updates.
    AlternatingPv {
        /// Continuous-scale optimizer.
        continuous_optimizer: NativeAdamWRecipe,
        /// Code-proposal optimizer.
        code_optimizer: NativeAdamWRecipe,
        /// Positive fraction <=1 of logical units considered each code update.
        max_code_change_fraction: f32,
        /// Optional finite positive relative code movement bound.
        max_relative_code_change: Option<f32>,
    },
}
impl NativeRecoveryRecipe {
    /// Validate declared parameters; native execution remains separately admitted.
    pub fn validate(&self) -> Result<(), NativeRefinementError> {
        match self {
            Self::ScaleOnly { optimizer } => optimizer.validate(),
            Self::AlternatingPv {
                continuous_optimizer,
                code_optimizer,
                max_code_change_fraction,
                max_relative_code_change,
            } => {
                continuous_optimizer.validate()?;
                code_optimizer.validate()?;
                if !max_code_change_fraction.is_finite()
                    || !(0.0..=1.0).contains(max_code_change_fraction)
                    || *max_code_change_fraction == 0.0
                    || max_relative_code_change.is_some_and(|v| !v.is_finite() || v <= 0.0)
                {
                    return Err(NativeRefinementError::Recipe);
                }
                Ok(())
            }
        }
    }
}

/// Ordered declared raw-content window; its identity excludes role and weight.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeRefinementWindow {
    /// Canonical content identity; not proof of semantic dataset decontamination.
    pub content: DigestBytes,
    /// Finite positive declared objective weight.
    pub weight: f64,
}

/// Logical limits, not complete process/device memory measurements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeRefinementLimits {
    /// Retained selected-owner state payload ceiling.
    pub max_state_bytes: u64,
    /// Native update workspace ceiling.
    pub max_workspace_bytes: u64,
    /// Physical bytes of one actual recovery checkpoint.
    pub max_checkpoint_bytes: u64,
    /// Complete child package ceiling.
    pub max_package_bytes: u64,
}

/// Complete native single-level recovery request; no implicit numerical defaults.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeRefinementPlan {
    /// Actual parent binding.
    pub parent: NativeRefinementParent,
    /// Explicit name of the one complete child level.
    pub child_level: String,
    /// Sorted canonical selected owner names, not aliases or blob-dedup groups.
    pub owners: Vec<String>,
    /// Declared exact native recovery parameters.
    pub recipe: NativeRecoveryRecipe,
    /// Declared scalar objective content; not immutable callback authentication.
    pub objective: DigestBytes,
    /// Optional declared teacher original source, never silently inferred.
    pub teacher_source: Option<OriginalSourceId>,
    /// One explicitly ordered window per requested update; repetitions are allowed.
    pub training: Vec<NativeRefinementWindow>,
    /// Ordered validation windows, disjoint from training content identities.
    pub validation: Vec<NativeRefinementWindow>,
    /// Explicit logical resource bounds.
    pub limits: NativeRefinementLimits,
}
impl NativeRefinementPlan {
    /// Structural request checks, not data/teacher truth or numerical admission.
    pub fn validate(&self) -> Result<(), NativeRefinementError> {
        self.parent.validate()?;
        name(&self.child_level)?;
        self.recipe.validate()?;
        if self.owners.is_empty()
            || self.owners.len() > 1_000_000
            || self.owners.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(NativeRefinementError::Owners);
        }
        for owner in &self.owners {
            name(owner)?;
        }
        for windows in [&self.training, &self.validation] {
            if windows.is_empty()
                || windows.len() > 1_000_000
                || windows
                    .iter()
                    .any(|w| !w.weight.is_finite() || w.weight <= 0.0)
            {
                return Err(NativeRefinementError::Data);
            }
        }
        let training: BTreeSet<_> = self.training.iter().map(|w| w.content).collect();
        if self
            .validation
            .iter()
            .any(|w| training.contains(&w.content))
        {
            return Err(NativeRefinementError::Data);
        }
        let l = self.limits;
        if [
            l.max_state_bytes,
            l.max_workspace_bytes,
            l.max_checkpoint_bytes,
            l.max_package_bytes,
        ]
        .iter()
        .any(|&n| n == 0 || n > ((1u64 << 53) - 1))
        {
            return Err(NativeRefinementError::Limits);
        }
        Ok(())
    }
    /// Bind the expected package observation and actual selected-level inventory.
    /// This validates metadata agreement; the caller must admit original bytes/logs.
    pub fn validate_parent(
        &self,
        manifest: &ModelManifest,
        parent: &NativeRefinementParent,
    ) -> Result<(), NativeRefinementError> {
        self.validate()?;
        manifest
            .validate()
            .map_err(|_| NativeRefinementError::Parent)?;
        if &self.parent != parent {
            return Err(NativeRefinementError::Parent);
        }
        let level = manifest
            .levels
            .iter()
            .find(|l| l.name == parent.level)
            .ok_or(NativeRefinementError::Parent)?;
        for name in &self.owners {
            let index = level
                .tensors
                .binary_search_by(|t| t.name.as_str().cmp(name.as_str()))
                .map_err(|_| NativeRefinementError::Owners)?;
            let tensor = &level.tensors[index];
            if !matches!(tensor.storage, TensorStorage::Additive { .. }) {
                return Err(NativeRefinementError::Owners);
            }
        }
        Ok(())
    }
}

/// Recorded actual execution scope, not independently authenticated hardware.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeRefinementExecution {
    /// Actual backend identifier.
    pub backend: String,
    /// Actual machine/device observation.
    pub device: String,
    /// Actual producing source revision identity.
    pub source_revision: String,
}
impl NativeRefinementExecution {
    /// Check recorded scope names, not their independent authenticity.
    pub fn validate(&self) -> Result<(), NativeRefinementError> {
        name(&self.backend)?;
        name(&self.device)?;
        name(&self.source_revision)
    }
}

/// Distinct native recovery start, not a PTQ conversion start.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeRefinementStarted {
    /// Exact original request.
    pub plan: NativeRefinementPlan,
    /// Recorded actual execution scope.
    pub execution: NativeRefinementExecution,
}

/// Sparse durable-state observation; does not require a checkpoint every update.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeRefinementCheckpointed {
    /// Exact native request identity.
    pub plan: NativeRefinementPlanId,
    /// Strictly advancing update cursor into the declared ordered training list.
    pub completed_updates: u64,
    /// Actual native state identity, not a legacy checkpoint digest.
    pub state: NativeRefinementStateId,
    /// BLAKE3 of exact serialized checkpoint bytes.
    pub checkpoint: BlobId,
    /// Actual serialized checkpoint length.
    pub checkpoint_bytes: u64,
    /// Logical retained selected-owner state payload bytes, not RSS.
    pub state_payload_bytes: u64,
    /// Complete sorted selected-owner stored-byte bindings. Layouts/count vectors
    /// are inherited from the parent, not copied into every checkpoint event.
    pub owners: Vec<NativeRefinementTensorOutput>,
}

/// Compact output binding; exact storage interpretation is inherited from parent.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeRefinementTensorOutput {
    /// Canonical tensor owner name.
    pub name: String,
    /// Exact canonical stored-byte identity, not a semantic model ID.
    pub content: BlobId,
}

/// Ordered observed objective values, with no acceptance/quality verdict.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeRefinementValidation {
    /// Declared content actually observed; raw-data verification remains separate.
    pub content: DigestBytes,
    /// Finite parent objective observation.
    pub parent_loss: f64,
    /// Finite child objective observation; not automatically an improvement.
    pub child_loss: f64,
}

/// Completed recovery observations, not successful publication or qualification.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeRefinementCompleted {
    /// Exact native request identity.
    pub plan: NativeRefinementPlanId,
    /// Must name the exact final durable checkpoint.
    pub checkpoint: BlobId,
    /// Complete child stored-byte bindings; no repeated count vectors or
    /// caller-supplied replacement layout/precision/basis interpretations.
    pub tensors: Vec<NativeRefinementTensorOutput>,
    /// Complete ordered finite observations for the requested validation windows.
    pub validation: Vec<NativeRefinementValidation>,
}

/// Recorded numerical replay inputs/output, not objective execution or acceptance.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct NativeScaleReplayObserved {
    /// Exact frozen native request identity.
    pub plan: NativeRefinementPlanId,
    /// Complete gradient journal BlobId, including its trailer checksum.
    pub journal: BlobId,
    /// Exact complete journal bytes.
    pub journal_bytes: u64,
    /// Exact replayed TNSR BlobId.
    pub checkpoint: BlobId,
    /// Exact replayed TNSR length.
    pub checkpoint_bytes: u64,
    /// Complete declared update cursor.
    pub completed_updates: u64,
}

/// Structural native request error; unavailable execution is a separate result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NativeRefinementError {
    /// Invalid declared name/reference.
    Name,
    /// Invalid numerical recipe declaration.
    Recipe,
    /// Empty, unordered, repeated or non-additive selected owners.
    Owners,
    /// Empty/invalid windows or overlapping train/validation identities.
    Data,
    /// Invalid logical limits.
    Limits,
    /// Parent observation or inventory mismatch.
    Parent,
}
fn name(value: &str) -> Result<(), NativeRefinementError> {
    crate::model::name(value, false).map_err(|_| NativeRefinementError::Name)
}
impl core::fmt::Display for NativeRefinementError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "native refinement: {self:?}")
    }
}
#[cfg(feature = "std")]
impl std::error::Error for NativeRefinementError {}
