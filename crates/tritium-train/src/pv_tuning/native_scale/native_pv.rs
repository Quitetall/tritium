//! Numerical alternating P/V; not source admission, replay, or publication.
use super::*;
use std::{cmp::Ordering, collections::BinaryHeap};
use tritium_core::project_additive_code_unit;
use tritium_schema::NativeAdamWRecipe;

mod checkpoint;
mod paged;
pub use paged::{
    NativePvPagedError, NativePvPagedLimits, NativePvPagedSession, NativePvPagedSizeLedger,
};
mod consumers;
#[cfg(feature = "native-cpu")]
mod cpu;
#[cfg(feature = "native-cpu")]
pub use cpu::{NativeCpuPvError, NativeCpuPvJournalStageError, NativeCpuPvSession};
mod verification;
pub use verification::{NativePvChildVerification, VerifiedNativePvChild, verify_native_pv_child};
mod journal;
pub use journal::{
    NativePvJournalError, NativePvJournalLimits, NativePvJournalResume, NativePvJournalSizeLedger,
    NativePvJournalSummary, NativePvJournalWriter, NativePvPagedJournalWriter,
    NativePvPagedRecordingError, NativePvPagedRecordingLimits, NativePvPagedRecordingSizeLedger,
    NativePvPagedUpdate, NativePvResumeBundleError, NativePvResumeBundleLimits,
    NativePvResumeBundleReader, NativePvResumeTail, PublishedNativePvResumeBundle,
    VerifiedNativePvReplay, publish_native_pv_resume_bundle, replay_native_pv_journal,
    replay_native_pv_journal_with_state, resume_native_pv_journal, stage_native_pv_resume_bundle,
};

/// Logical buffer lengths; excludes metadata, allocator slack and external graphs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativePvSizeLedger {
    /// Hard codes/index/scales and separate scale/code Adam moments.
    pub state_payload_bytes: u64,
    /// Conservative private numerical candidate and scratch payload ceiling.
    pub update_workspace_bytes: u64,
    /// Exact TNPV serialized length including checksum.
    pub checkpoint_bytes: u64,
}

/// Per-owner numerical projection observations, not quality evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativePvOwnerStep {
    /// Present logical units selected by proposal movement, including zero codes.
    pub selected_units: usize,
    /// Selected units with accepted changed coefficients.
    pub changed_units: usize,
    /// Improving units refused by this owner's relative movement budget.
    pub trust_limited_units: usize,
    /// Sum of selected-unit squared proposal errors before projection.
    pub surrogate_before: f64,
    /// Sum of selected-unit squared proposal errors after accepted projection.
    pub surrogate_after: f64,
    /// Accepted movement divided by the post-P norm (with epsilon floor).
    pub relative_change: f64,
}

/// Numerical observations for a committed ordered training window.
#[derive(Clone, Debug, PartialEq)]
pub struct NativePvStepStats {
    /// One-based completed window cursor.
    pub completed_updates: u64,
    /// Stored scales changed by P, across all owners.
    pub changed_scales: usize,
    /// V observations in frozen owner order.
    pub owners: Vec<NativePvOwnerStep>,
}

#[derive(Debug)]
struct PvOwner {
    weight: AdditiveTensor,
    scale: AdamState,
    code: AdamState,
}

#[derive(Clone, Copy, Debug)]
struct PvUpdater {
    continuous: AdamW,
    code_optimizer: AdamW,
    fraction: f32,
    trust: Option<f32>,
}
impl PvUpdater {
    fn from_plan(plan: &NativeRefinementPlan) -> Result<Self, PvTuningError> {
        plan.validate().map_err(|e| {
            PvTuningError::invalid_config(format!("invalid native PV request: {e:?}"))
        })?;
        let NativeRecoveryRecipe::AlternatingPv {
            continuous_optimizer,
            code_optimizer,
            max_code_change_fraction,
            max_relative_code_change,
        } = plan.recipe
        else {
            return Err(PvTuningError::invalid_config(
                "native PV requires AlternatingPv",
            ));
        };
        Ok(Self {
            continuous: adam(continuous_optimizer),
            code_optimizer: adam(code_optimizer),
            fraction: max_code_change_fraction,
            trust: max_relative_code_change,
        })
    }

    fn candidate(
        &self,
        owner: &PvOwner,
        scales: &[f32],
        weights: &[f32],
        window_weight: f64,
        next: u64,
    ) -> Result<(PvOwner, usize, NativePvOwnerStep), PvTuningError> {
        // Both adapters preflight their complete supplied owner adjoints before
        // this private update; do not add a second model-sized scan here.
        let (changed, continuous) = prepare_scale_candidate(
            &owner.weight,
            &owner.scale,
            self.continuous,
            next,
            window_weight,
            scales,
        )?;
        let mut weight = AdditiveTensor::from_view(owner.weight.view()).map_err(invalid_tensor)?;
        weight
            .replace_scales(continuous.values)
            .map_err(invalid_tensor)?;
        let mut candidate = PvOwner {
            weight,
            scale: continuous.state,
            code: AdamState {
                m: copied(&owner.code.m)?,
                v: copied(&owner.code.v)?,
            },
        };
        let stats = self.project(&mut candidate, weights, window_weight, next)?;
        Ok((candidate, changed, stats))
    }
}

fn check_gradients(owner: &PvOwner, scales: &[f32], weights: &[f32]) -> Result<(), PvTuningError> {
    if scales.len() != owner.weight.scales().len()
        || weights.len() != owner.code.m.len()
        || scales.iter().chain(weights).any(|value| !value.is_finite())
    {
        return Err(PvTuningError::step(
            "native PV gradient shape/values mismatch",
        ));
    }
    Ok(())
}

fn initial_owner(weight: AdditiveTensor) -> Result<PvOwner, PvTuningError> {
    let scales = weight.scales().len();
    let active = active_count(&weight)?;
    Ok(PvOwner {
        weight,
        scale: AdamState {
            m: zeroed(scales)?,
            v: zeroed(scales)?,
        },
        code: AdamState {
            m: zeroed(active)?,
            v: zeroed(active)?,
        },
    })
}

/// Exact native CPU P/V reference with bounded compact optimizer state.
///
/// No persistent dense weight master is retained. Code moments follow present
/// base-plane coordinates; absent tiles remain absent. Actual parent custody,
/// gradient/objective authenticity, journals and child publication are separate.
#[derive(Debug)]
pub struct NativePvRefinementSession {
    plan: NativeRefinementPlan,
    plan_id: NativeRefinementPlanId,
    parent: [u8; 32],
    continuous: AdamW,
    code_optimizer: AdamW,
    fraction: f32,
    trust: Option<f32>,
    owners: Vec<PvOwner>,
    completed: u64,
    ledger: NativePvSizeLedger,
}

/// Borrowed fully checked candidate. Cannot escape its staging callback.
pub struct NativePvCandidate<'a> {
    session: &'a NativePvRefinementSession,
    owners: &'a [PvOwner],
    stats: &'a NativePvStepStats,
}
impl std::fmt::Debug for NativePvCandidate<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativePvCandidate")
            .field("plan", &self.session.plan_id)
            .field("proposed_update", &self.stats.completed_updates)
            .field("owners", &self.owners.len())
            .finish_non_exhaustive()
    }
}
impl NativePvCandidate<'_> {
    /// Exact frozen request, not an evidence or model identity.
    #[must_use]
    pub fn plan_id(&self) -> NativeRefinementPlanId {
        self.session.plan_id
    }
    /// Proposed complete numerical observations; not yet committed.
    #[must_use]
    pub fn stats(&self) -> &NativePvStepStats {
        self.stats
    }
    /// Immutable hard candidate tensors in frozen owner order.
    pub fn owners(&self) -> impl ExactSizeIterator<Item = (&str, &AdditiveTensor)> {
        self.session
            .plan
            .owners
            .iter()
            .zip(self.owners)
            .map(|(name, owner)| (name.as_str(), &owner.weight))
    }
}

/// Numerical rejection versus the caller's original staging failure.
#[derive(Debug, Eq, PartialEq)]
pub enum NativePvStageError<E> {
    /// No state was committed because numerical/preflight validation failed.
    Update(PvTuningError),
    /// No state was committed because the caller refused staging.
    Stage(E),
}
impl<E> From<PvTuningError> for NativePvStageError<E> {
    fn from(error: PvTuningError) -> Self {
        Self::Update(error)
    }
}
impl<E: std::fmt::Display> std::fmt::Display for NativePvStageError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Update(error) => error.fmt(f),
            Self::Stage(error) => write!(f, "native PV staging failed: {error}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for NativePvStageError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Update(error) => Some(error),
            Self::Stage(error) => Some(error),
        }
    }
}

#[derive(Clone, Copy)]
struct Rank {
    unit: usize,
    offset: usize,
    magnitude: f64,
}

// Better entries compare smaller, so the heap root is the worst retained unit.
// The logical-unit tie break is the same one used by the original full sort.
impl Ord for Rank {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .magnitude
            .total_cmp(&self.magnitude)
            .then(self.unit.cmp(&other.unit))
    }
}
impl PartialOrd for Rank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Rank {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Rank {}

fn selected_count(units: usize, fraction: f32) -> usize {
    ((units as f64 * f64::from(fraction)).ceil() as usize).min(units)
}

impl NativePvRefinementSession {
    /// Validate frozen request, exact native parents and logical caps before moments.
    pub fn new(
        plan: NativeRefinementPlan,
        parents: Vec<AdditiveTensor>,
    ) -> Result<Self, PvTuningError> {
        admit(plan, &parents)?.with_parents(parents)
    }
    fn with_parents(mut self, parents: Vec<AdditiveTensor>) -> Result<Self, PvTuningError> {
        for weight in parents {
            self.owners.push(initial_owner(weight)?);
        }
        Ok(self)
    }
    /// Frozen plan including the explicitly selected parent level.
    #[must_use]
    pub fn plan(&self) -> &NativeRefinementPlan {
        &self.plan
    }
    /// Exact request identity.
    #[must_use]
    pub const fn plan_id(&self) -> NativeRefinementPlanId {
        self.plan_id
    }
    /// Number of fully committed training windows.
    #[must_use]
    pub const fn completed_updates(&self) -> u64 {
        self.completed
    }
    /// Logical sizes admitted before state allocation; not physical memory.
    #[must_use]
    pub const fn size_ledger(&self) -> NativePvSizeLedger {
        self.ledger
    }
    /// Current hard owner tensors in frozen order.
    pub fn weights(&self) -> impl ExactSizeIterator<Item = &AdditiveTensor> {
        self.owners.iter().map(|owner| &owner.weight)
    }
    pub(super) fn weight_at(&self, index: usize) -> &AdditiveTensor {
        &self.owners[index].weight
    }
    pub(super) fn parent_fingerprint(&self) -> [u8; 32] {
        self.parent
    }
    pub(super) fn parent_hasher(plan: NativeRefinementPlanId) -> blake3::Hasher {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"tritium.native-pv.parent.v1\0");
        hasher.update(plan.as_bytes());
        hasher
    }

    /// Update all owners atomically using two independent unweighted adjoints.
    ///
    /// Scales use actual stored-scale order. Weight adjoints use present base-plane
    /// coefficient order in stored basis, not original coordinates or per-plane
    /// trit derivatives. Each window weight is applied exactly once to both arrays.
    pub fn step(
        &mut self,
        scales: &[&[f32]],
        weights: &[&[f32]],
    ) -> Result<NativePvStepStats, PvTuningError> {
        match self.step_staged(scales, weights, |_| Ok::<_, std::convert::Infallible>(())) {
            Ok((stats, ())) => Ok(stats),
            Err(NativePvStageError::Update(error)) => Err(error),
            Err(NativePvStageError::Stage(never)) => match never {},
        }
    }

    /// Stage all checked hard candidates before committing any owner or cursor.
    /// Error or unwind preserves all old numerical state. Arbitrary external
    /// effects and staged graph resources are the caller's responsibility.
    ///
    /// ```compile_fail
    /// use tritium_train::pv_tuning::NativePvRefinementSession;
    /// fn escape(s: &mut NativePvRefinementSession, g: &[&[f32]]) {
    ///     let _ = s.step_staged(g, g, |candidate| Ok::<_, ()>(candidate));
    /// }
    /// ```
    pub fn step_staged<T, E>(
        &mut self,
        scales: &[&[f32]],
        weights: &[&[f32]],
        stage: impl for<'a> FnOnce(NativePvCandidate<'a>) -> Result<T, E>,
    ) -> Result<(NativePvStepStats, T), NativePvStageError<E>> {
        let window = self
            .plan
            .training
            .get(self.completed as usize)
            .ok_or_else(|| PvTuningError::step("native PV schedule is exhausted"))?;
        if scales.len() != self.owners.len() || weights.len() != self.owners.len() {
            return Err(PvTuningError::step("native PV gradient owner inventory mismatch").into());
        }
        for ((owner, scale), weight) in self.owners.iter().zip(scales).zip(weights) {
            check_gradients(owner, scale, weight)?;
        }
        let next = self.completed + 1;
        let mut candidates = reserved(self.owners.len())?;
        let mut stats = NativePvStepStats {
            completed_updates: next,
            changed_scales: 0,
            owners: reserved(self.owners.len())?,
        };
        let updater = PvUpdater {
            continuous: self.continuous,
            code_optimizer: self.code_optimizer,
            fraction: self.fraction,
            trust: self.trust,
        };
        for ((owner, scale), gradient) in self.owners.iter().zip(scales).zip(weights) {
            let (candidate, changed, observation) =
                updater.candidate(owner, scale, gradient, window.weight, next)?;
            stats.changed_scales += changed;
            stats.owners.push(observation);
            candidates.push(candidate);
        }
        let staged = stage(NativePvCandidate {
            session: self,
            owners: &candidates,
            stats: &stats,
        })
        .map_err(NativePvStageError::Stage)?;
        self.owners = candidates;
        self.completed = next;
        Ok((stats, staged))
    }
}

impl PvUpdater {
    fn project(
        &self,
        owner: &mut PvOwner,
        gradient: &[f32],
        window_weight: f64,
        step: u64,
    ) -> Result<NativePvOwnerStep, PvTuningError> {
        let active = gradient.len();
        if active == 0 {
            return Ok(NativePvOwnerStep {
                selected_units: 0,
                changed_units: 0,
                trust_limited_units: 0,
                surrogate_before: 0.0,
                surrogate_after: 0.0,
                relative_change: 0.0,
            });
        }
        let width = unit_width(&owner.weight);
        let mut proposal = reserved(active)?;
        let cols = owner.weight.layout().cols as usize;
        let mut row = zeroed(cols)?;
        for r in 0..owner.weight.layout().rows as usize {
            owner
                .weight
                .view()
                .dequant_row_into(r, &mut row)
                .map_err(|e| PvTuningError::step(format!("native PV decode: {e:?}")))?;
            for tile in 0..cols.div_ceil(256) {
                if present(&owner.weight, r, tile) {
                    proposal.extend_from_slice(&row[tile * 256..((tile + 1) * 256).min(cols)]);
                }
            }
        }
        debug_assert_eq!(proposal.len(), active);
        if proposal.iter().any(|value| !value.is_finite()) {
            return Err(PvTuningError::step("native PV decoded weight overflowed"));
        }
        let norm = proposal
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .max(f64::from(f32::EPSILON));
        for &value in gradient {
            let value = (f64::from(value) * window_weight) as f32;
            if !value.is_finite() {
                return Err(PvTuningError::step(
                    "native PV weighted gradient overflowed",
                ));
            }
        }
        self.code_optimizer
            .step_mapped(step, &mut proposal, gradient, &mut owner.code, |value| {
                (f64::from(value) * window_weight) as f32
            });
        check_optimizer_moments(&owner.code, "native PV code")?;
        check_corrected_moments(self.code_optimizer, step, &owner.code)
            .map_err(PvTuningError::step)?;
        if proposal.iter().any(|value| !value.is_finite()) {
            return Err(PvTuningError::step("native PV proposal overflowed"));
        }
        let keep = selected_count(active / width, self.fraction);
        let mut ranked = BinaryHeap::from(reserved(keep)?);
        let mut offset = 0;
        for r in 0..owner.weight.layout().rows as usize {
            owner
                .weight
                .view()
                .dequant_row_into(r, &mut row)
                .map_err(|e| PvTuningError::step(format!("native PV decode: {e:?}")))?;
            for tile in 0..cols.div_ceil(256) {
                if !present(&owner.weight, r, tile) {
                    continue;
                }
                for col in (tile * 256..((tile + 1) * 256).min(cols)).step_by(width) {
                    let rank = Rank {
                        unit: (r * cols + col) / width,
                        offset,
                        magnitude: row[col..col + width]
                            .iter()
                            .zip(&proposal[offset..offset + width])
                            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                            .sum(),
                    };
                    if ranked.len() < keep {
                        ranked.push(rank);
                    } else if let Some(mut worst) = ranked.peek_mut()
                        && rank < *worst
                    {
                        *worst = rank;
                    }
                    offset += width;
                }
            }
        }
        debug_assert_eq!(offset, active);
        drop(row);
        let ranked = ranked.into_sorted_vec();
        let limit = self.trust.map(|ratio| f64::from(ratio).powi(2) * norm);
        let mut accepted = 0.0;
        let mut stats = NativePvOwnerStep {
            selected_units: keep,
            changed_units: 0,
            trust_limited_units: 0,
            surrogate_before: 0.0,
            surrogate_after: 0.0,
            relative_change: 0.0,
        };
        let mut trits = copied(owner.weight.trits())?;
        for rank in ranked {
            let result = project_additive_code_unit(
                owner.weight.view(),
                rank.unit,
                &proposal[rank.offset..rank.offset + width],
                limit.map(|limit| (limit - accepted).max(0.0)),
                &mut trits,
            )
            .map_err(|error| PvTuningError::step(format!("native PV projection: {error:?}")))?;
            stats.changed_units += usize::from(result.changed);
            stats.trust_limited_units += usize::from(result.trust_limited);
            stats.surrogate_before += result.surrogate_before;
            stats.surrogate_after += result.surrogate_after;
            accepted += result.squared_change;
        }
        owner.weight.replace_trits(trits).map_err(invalid_tensor)?;
        stats.relative_change = (accepted / norm).sqrt();
        Ok(stats)
    }
}

fn invalid_tensor(error: impl std::fmt::Debug) -> PvTuningError {
    PvTuningError::invalid_weight(format!("native PV tensor: {error:?}"))
}
fn adam(recipe: NativeAdamWRecipe) -> AdamW {
    AdamW {
        lr: recipe.lr,
        beta1: recipe.beta1,
        beta2: recipe.beta2,
        eps: recipe.eps,
        weight_decay: recipe.weight_decay,
    }
}
fn active_count(weight: &AdditiveTensor) -> Result<usize, PvTuningError> {
    if weight.plane_count() == 0 {
        Ok(0)
    } else {
        Ok(weight.view().plane_trits(0).map_err(invalid_tensor)?.len())
    }
}
fn unit_width(weight: &AdditiveTensor) -> usize {
    if weight.layout().codec == PlaneCodec::S34 {
        4
    } else {
        1
    }
}
fn present(weight: &AdditiveTensor, row: usize, tile: usize) -> bool {
    weight
        .tile_counts()
        .is_none_or(|counts| counts[row * (weight.layout().cols as usize).div_ceil(256) + tile] > 0)
}
fn admit(
    plan: NativeRefinementPlan,
    parents: &[AdditiveTensor],
) -> Result<NativePvRefinementSession, PvTuningError> {
    let updater = PvUpdater::from_plan(&plan)?;
    if parents.len() != plan.owners.len() {
        return Err(PvTuningError::invalid_weight(
            "native PV parent owner inventory mismatch",
        ));
    }
    let mut ledger = NativePvSizeLedger {
        state_payload_bytes: 0,
        update_workspace_bytes: 0,
        checkpoint_bytes: checkpoint::FIXED_BYTES,
    };
    for weight in parents {
        add_sizes(&mut ledger, owner_sizes(weight, updater.fraction)?)?;
    }
    check_sizes(&plan, ledger)?;
    let plan_id = native_refinement_plan_id(&plan)
        .map_err(|e| PvTuningError::invalid_config(e.to_string()))?;
    let mut hasher = NativePvRefinementSession::parent_hasher(plan_id);
    for weight in parents {
        hash_parent_tensor(&mut hasher, weight)?;
    }
    Ok(NativePvRefinementSession {
        plan,
        plan_id,
        parent: *hasher.finalize().as_bytes(),
        continuous: updater.continuous,
        code_optimizer: updater.code_optimizer,
        fraction: updater.fraction,
        trust: updater.trust,
        owners: reserved(parents.len())?,
        completed: 0,
        ledger,
    })
}

fn owner_sizes(
    weight: &AdditiveTensor,
    fraction: f32,
) -> Result<NativePvSizeLedger, PvTuningError> {
    let t = weight.trits().len() as u64;
    let s = weight.scales().len() as u64;
    let a = active_count(weight)? as u64;
    let c = weight.layout().cols;
    let i = weight
        .view()
        .auxiliary_buffer_len_bytes()
        .map_err(invalid_tensor)? as u64;
    if unit_width(weight) == 4 && !c.is_multiple_of(4) {
        return Err(PvTuningError::invalid_weight(
            "native PV S34 columns must be divisible by4",
        ));
    }
    let sizes = || -> Option<(u64, u64, u64)> {
        let state = t
            .checked_add(i)?
            .checked_add(s.checked_mul(12)?)?
            .checked_add(a.checked_mul(8)?)?;
        let workspace = t
            .checked_mul(2)?
            .checked_add(i)?
            .checked_add(s.checked_mul(20)?)?
            .checked_add(a.checked_mul(12)?)?
            .checked_add(
                (selected_count((a / unit_width(weight) as u64).try_into().ok()?, fraction) as u64)
                    .checked_mul(std::mem::size_of::<Rank>() as u64)?,
            )?
            .checked_add(c.checked_mul(4)?)?;
        let checkpoint = 24u64
            .checked_add(t)?
            .checked_add(s.checked_mul(8 + scale_width(weight.layout().law.precision).ok()?)?)?
            .checked_add(a.checked_mul(8)?)?;
        Some((state, workspace, checkpoint))
    };
    let (state, workspace, checkpoint) = sizes().ok_or_else(size_overflow)?;
    Ok(NativePvSizeLedger {
        state_payload_bytes: state,
        update_workspace_bytes: workspace,
        checkpoint_bytes: checkpoint,
    })
}

fn add_sizes(
    ledger: &mut NativePvSizeLedger,
    owner: NativePvSizeLedger,
) -> Result<(), PvTuningError> {
    ledger.state_payload_bytes = ledger
        .state_payload_bytes
        .checked_add(owner.state_payload_bytes)
        .ok_or_else(size_overflow)?;
    ledger.update_workspace_bytes = ledger
        .update_workspace_bytes
        .checked_add(owner.update_workspace_bytes)
        .ok_or_else(size_overflow)?;
    ledger.checkpoint_bytes = ledger
        .checkpoint_bytes
        .checked_add(owner.checkpoint_bytes)
        .ok_or_else(size_overflow)?;
    Ok(())
}

fn check_sizes(
    plan: &NativeRefinementPlan,
    ledger: NativePvSizeLedger,
) -> Result<(), PvTuningError> {
    if ledger.state_payload_bytes > plan.limits.max_state_bytes
        || ledger.update_workspace_bytes > plan.limits.max_workspace_bytes
        || ledger.checkpoint_bytes > plan.limits.max_checkpoint_bytes
    {
        return Err(PvTuningError::invalid_config(
            "native PV state/workspace/checkpoint limit exceeded",
        ));
    }
    Ok(())
}
