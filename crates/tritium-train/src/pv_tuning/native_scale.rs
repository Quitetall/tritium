//! Exact native scale state; numerical execution is not parent/release admission.
use half::{bf16, f16};
use tritium_evidence::native_refinement_plan_id;
use tritium_format::{AdditiveTensor, salt_v2::SaltV2Codec};
use tritium_schema::{
    NativeRecoveryRecipe, NativeRefinementPlan, NativeRefinementPlanId, PlaneCodec, ScalePrecision,
};

use super::{PvTuningError, continuous::check_optimizer_moments};
use crate::optim::{AdamState, AdamW, Optimizer};

mod checkpoint;
mod native_pv;
#[cfg(feature = "native-cpu")]
pub use native_pv::{NativeCpuPvError, NativeCpuPvJournalStageError, NativeCpuPvSession};
pub use native_pv::{
    NativePvCandidate, NativePvOwnerStep, NativePvRefinementSession, NativePvResumeBundleError,
    NativePvResumeBundleLimits, NativePvResumeBundleReader, NativePvSizeLedger, NativePvStageError,
    NativePvStepStats, PublishedNativePvResumeBundle, publish_native_pv_resume_bundle,
    stage_native_pv_resume_bundle,
};
pub use native_pv::{
    NativePvJournalError, NativePvJournalLimits, NativePvJournalResume, NativePvJournalSizeLedger,
    NativePvJournalSummary, NativePvJournalWriter, NativePvResumeTail, VerifiedNativePvReplay,
    replay_native_pv_journal, replay_native_pv_journal_with_state, resume_native_pv_journal,
};
pub use native_pv::{
    NativePvPagedError, NativePvPagedLimits, NativePvPagedSession, NativePvPagedSizeLedger,
};
pub use native_pv::{
    NativePvPagedJournalReplay, NativePvPagedJournalResume, NativePvPagedReplayError,
    NativePvPagedReplayLimits, replay_native_pv_paged_journal, resume_native_pv_paged_journal,
};
pub use native_pv::{
    NativePvPagedJournalWriter, NativePvPagedRecordingError, NativePvPagedRecordingLimits,
    NativePvPagedRecordingSizeLedger, NativePvPagedUpdate,
};
pub use native_pv::{
    NativePvPagedResumeBundleError, NativePvPagedResumeBundleLimits,
    NativePvPagedResumeBundleReader, PublishedNativePvPagedResumeBundle,
    publish_native_pv_paged_resume_bundle, stage_native_pv_paged_resume_bundle,
};
mod consumers;
#[cfg(feature = "native-cpu")]
mod cpu;
mod identity;
mod journal;
pub use journal::{
    NativeScaleResumeBundleError, NativeScaleResumeBundleLimits, NativeScaleResumeBundleReader,
    PublishedNativeScaleResumeBundle, publish_native_scale_resume_bundle,
    stage_native_scale_resume_bundle,
};
mod publication;
pub use native_pv::{NativePvChildVerification, VerifiedNativePvChild, verify_native_pv_child};
pub use publication::{
    NATIVE_PV_CHECKPOINT_FILE, NATIVE_PV_JOURNAL_FILE, NativeChildPublicationError,
    NativePvChildError, NativePvChildGuardError, NativePvChildLimits, NativePvChildSizeLedger,
    NativePvChildSource, PublishedNativePvChild, inspect_native_pv_child, publish_native_pv_child,
};
pub use verification::{NativePvParentReadLimits, decode_native_pv_parents};
mod transaction;
mod verification;
pub use consumers::{NativePvContribution, NativeScaleContribution};
#[cfg(feature = "native-cpu")]
pub use cpu::{NativeCpuScaleError, NativeCpuScaleJournalStageError, NativeCpuScaleSession};
pub use journal::{
    NativeScaleJournalError, NativeScaleJournalLimits, NativeScaleJournalResume,
    NativeScaleJournalSizeLedger, NativeScaleJournalSummary, NativeScaleJournalWriter,
    NativeScaleResumeTail, VerifiedNativeScaleReplay, replay_native_scale_journal,
    replay_native_scale_journal_with_state, resume_native_scale_journal,
};
pub use publication::{
    NATIVE_SCALE_CHECKPOINT_FILE, NATIVE_SCALE_JOURNAL_FILE, NativeScaleChildError,
    NativeScaleChildGuardError, NativeScaleChildLimits, NativeScaleChildSizeLedger,
    NativeScaleChildSource, PublishedNativeScaleChild, inspect_native_scale_child,
    publish_native_scale_child,
};
pub use transaction::{NativeScaleStageError, NativeScaleStep};
pub use verification::{
    NativeRefinementOwnerReadLimits, NativeRefinementOwnerSource, NativeRefinementOwnerSourceError,
};
pub use verification::{
    NativeScaleChildVerification, NativeScaleParentReadLimits, VerifiedNativeScaleChild,
    decode_native_scale_parents, verify_native_scale_child,
};

struct ScaleCandidate {
    values: Vec<f32>,
    state: AdamState,
}
struct PreparedScaleStep {
    next: u64,
    changed: usize,
    candidates: Vec<ScaleCandidate>,
}

/// Logical selected-owner buffers; not graph memory, allocator capacity or RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeScaleSizeLedger {
    /// Decoded ternary coefficients, widened exact scales, compact index and Adam moments.
    pub state_payload_bytes: u64,
    /// Conservative update buffers: candidate scales/moments plus weighted gradients.
    /// Excludes caller gradients, metadata and external activation/autograd state.
    pub update_workspace_bytes: u64,
    /// Exact serialized checkpoint length, including header and checksum.
    pub checkpoint_bytes: u64,
}

#[derive(Debug)]
struct Owner {
    weight: AdditiveTensor,
    state: AdamState,
}

/// CPU scale-only recovery over one explicit level's complete selected-owner set.
///
/// The caller supplies admitted parent tensors in the frozen plan's owner order.
/// This numerical object does not authenticate their package/source or objective.
/// Codes, allocation, basis, law and transport remain frozen. There is no dense
/// weight master and no code optimizer. Alternating-PV requests are rejected.
#[derive(Debug)]
pub struct NativeScaleRefinementSession {
    plan: NativeRefinementPlan,
    plan_id: NativeRefinementPlanId,
    parent: [u8; 32],
    optimizer: AdamW,
    owners: Vec<Owner>,
    completed_updates: u64,
    ledger: NativeScaleSizeLedger,
}

struct Admission {
    plan_id: NativeRefinementPlanId,
    parent: [u8; 32],
    optimizer: AdamW,
    ledger: NativeScaleSizeLedger,
}

impl NativeScaleRefinementSession {
    /// Start with zeroed moments and the exact supplied parent representation.
    /// Logical limits and codecs are checked before optimizer buffers are allocated.
    pub fn new(
        plan: NativeRefinementPlan,
        parents: Vec<AdditiveTensor>,
    ) -> Result<Self, PvTuningError> {
        let admission = admit(&plan, &parents)?;
        let mut owners = reserved(parents.len())?;
        for weight in parents {
            let len = weight.scales().len();
            owners.push(Owner {
                weight,
                state: AdamState {
                    m: zeroed(len)?,
                    v: zeroed(len)?,
                },
            });
        }
        Ok(Self::from_admitted(plan, admission, owners, 0))
    }

    fn from_admitted(
        plan: NativeRefinementPlan,
        admission: Admission,
        owners: Vec<Owner>,
        completed_updates: u64,
    ) -> Self {
        Self {
            plan,
            plan_id: admission.plan_id,
            parent: admission.parent,
            optimizer: admission.optimizer,
            owners,
            completed_updates,
            ledger: admission.ledger,
        }
    }

    /// Frozen native request, including the one explicitly selected parent level.
    #[must_use]
    pub fn plan(&self) -> &NativeRefinementPlan {
        &self.plan
    }

    /// Identity of the exact native request, not a legacy PV recipe digest.
    #[must_use]
    pub const fn plan_id(&self) -> NativeRefinementPlanId {
        self.plan_id
    }

    /// Number of fully committed ordered training windows.
    #[must_use]
    pub const fn completed_updates(&self) -> u64 {
        self.completed_updates
    }

    /// Exact logical buffer/checkpoint ledger admitted at construction.
    #[must_use]
    pub const fn size_ledger(&self) -> NativeScaleSizeLedger {
        self.ledger
    }

    /// Borrow current selected tensors in the frozen plan's owner order.
    pub fn weights(&self) -> impl ExactSizeIterator<Item = &AdditiveTensor> {
        self.owners.iter().map(|owner| &owner.weight)
    }

    /// Apply one window's **unweighted stored-scale gradients** for every owner.
    ///
    /// The session applies that window's declared weight, then common AdamW,
    /// nonnegative projection and one exact storage-precision rounding. These are
    /// not logical dense-weight gradients; owner/basis/scale adjoints belong to
    /// the caller. All owners, moments and the cursor commit together or not at all.
    pub fn step(&mut self, gradients: &[&[f32]]) -> Result<usize, PvTuningError> {
        match self.step_staged(gradients, |_| Ok::<_, std::convert::Infallible>(())) {
            Ok((changed, ())) => Ok(changed),
            Err(NativeScaleStageError::Update(error)) => Err(error),
            Err(NativeScaleStageError::Stage(never)) => match never {},
        }
    }

    fn prepare_step(&self, gradients: &[&[f32]]) -> Result<PreparedScaleStep, PvTuningError> {
        let window = self
            .plan
            .training
            .get(self.completed_updates as usize)
            .ok_or_else(|| PvTuningError::step("native training schedule is exhausted"))?;
        if gradients.len() != self.owners.len()
            || gradients.iter().zip(&self.owners).any(|(g, owner)| {
                g.len() != owner.weight.scales().len() || g.iter().any(|v| !v.is_finite())
            })
        {
            return Err(PvTuningError::step(
                "native stored-scale gradient inventory/values mismatch",
            ));
        }
        let next = self.completed_updates + 1; // bounded by the validated schedule
        let mut candidates = reserved(self.owners.len())?;
        let mut changed = 0;
        for (owner, gradient) in self.owners.iter().zip(gradients) {
            let candidate = prepare_scale_candidate(
                &owner.weight,
                &owner.state,
                self.optimizer,
                next,
                window.weight,
                gradient,
            )?;
            changed += candidate.0;
            candidates.push(candidate.1);
        }
        Ok(PreparedScaleStep {
            next,
            changed,
            candidates,
        })
    }

    fn commit_step(&mut self, prepared: PreparedScaleStep) {
        for (owner, ScaleCandidate { values, state }) in
            self.owners.iter_mut().zip(prepared.candidates)
        {
            // The exact candidate has already passed validation; no allocation or
            // externally fallible operation remains in this commit phase.
            owner
                .weight
                .replace_scales(values)
                .expect("prevalidated native scales");
            owner.state = state;
        }
        self.completed_updates = prepared.next;
    }
}

fn admit(
    plan: &NativeRefinementPlan,
    parents: &[AdditiveTensor],
) -> Result<Admission, PvTuningError> {
    plan.validate()
        .map_err(|e| PvTuningError::invalid_config(format!("invalid native request: {e:?}")))?;
    let NativeRecoveryRecipe::ScaleOnly { optimizer } = plan.recipe else {
        return Err(PvTuningError::invalid_config(
            "native scale session requires ScaleOnly; alternating PV is not implemented",
        ));
    };
    if parents.len() != plan.owners.len() {
        return Err(PvTuningError::invalid_weight(
            "native parent owner count mismatch",
        ));
    }
    let ledger = size_ledger(parents)?;
    if ledger.state_payload_bytes > plan.limits.max_state_bytes
        || ledger.update_workspace_bytes > plan.limits.max_workspace_bytes
        || ledger.checkpoint_bytes > plan.limits.max_checkpoint_bytes
    {
        return Err(PvTuningError::invalid_config(
            "native state/workspace/checkpoint limit exceeded",
        ));
    }
    let plan_id = native_refinement_plan_id(plan)
        .map_err(|e| PvTuningError::invalid_config(e.to_string()))?;
    let mut hasher = parent_hasher(plan_id);
    for weight in parents {
        hash_parent_tensor(&mut hasher, weight)?;
    }
    Ok(Admission {
        plan_id,
        parent: *hasher.finalize().as_bytes(),
        ledger,
        optimizer: AdamW {
            lr: optimizer.lr,
            beta1: optimizer.beta1,
            beta2: optimizer.beta2,
            eps: optimizer.eps,
            weight_decay: optimizer.weight_decay,
        },
    })
}

fn parent_hasher(plan_id: NativeRefinementPlanId) -> blake3::Hasher {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"tritium.native-scale.parent.v1\0");
    hasher.update(plan_id.as_bytes());
    hasher
}
fn hash_parent_tensor(
    hasher: &mut blake3::Hasher,
    weight: &AdditiveTensor,
) -> Result<(), PvTuningError> {
    // Reuse canonical plane codecs with fixed 20-trit (LCM of 4 and 5)
    // chunks: no encoded model-size temporary and no S34 padding shortcut.
    let codec = match weight.layout().codec {
        PlaneCodec::D2 => SaltV2Codec::D2,
        PlaneCodec::B3 => SaltV2Codec::B3,
        PlaneCodec::S34 => SaltV2Codec::S34,
        _ => {
            return Err(PvTuningError::invalid_weight(
                "unsupported native plane codec",
            ));
        }
    };
    let view = weight.view();
    for plane in 0..view.plane_count() {
        for trits in view
            .plane_trits(plane)
            .map_err(|e| PvTuningError::invalid_weight(format!("{e:?}")))?
            .chunks(20)
        {
            let len = codec
                .ledger(trits.len())
                .map_err(|e| PvTuningError::invalid_weight(e.to_string()))?
                .physical_bytes;
            codec
                .pack_into(trits, &mut [0; 5][..len])
                .map_err(|e| PvTuningError::invalid_weight(e.to_string()))?;
        }
    }
    identity::hash_layout(hasher, weight.layout())?;
    hasher.update(&[weight.plane_count()]);
    frame(hasher, weight.tile_counts().unwrap_or_default());
    hasher.update(&(weight.trits().len() as u64).to_le_bytes());
    for chunk in weight.trits().chunks(256) {
        let mut bytes = [0; 256];
        for (byte, trit) in bytes.iter_mut().zip(chunk) {
            *byte = trit.get() as u8;
        }
        hasher.update(&bytes[..chunk.len()]);
    }
    hasher.update(&(weight.scales().len() as u64).to_le_bytes());
    for scale in weight.scales() {
        hasher.update(&scale.to_bits().to_le_bytes());
    }
    Ok(())
}

fn size_ledger(parents: &[AdditiveTensor]) -> Result<NativeScaleSizeLedger, PvTuningError> {
    let mut ledger = NativeScaleSizeLedger {
        state_payload_bytes: 0,
        update_workspace_bytes: 0,
        checkpoint_bytes: checkpoint::FIXED_BYTES,
    };
    for weight in parents {
        let scales = weight.scales().len() as u64;
        let auxiliary = weight
            .view()
            .auxiliary_buffer_len_bytes()
            .map_err(|e| PvTuningError::invalid_weight(format!("{e:?}")))?
            as u64;
        let payload = (weight.trits().len() as u64)
            .checked_add(auxiliary)
            .and_then(|v| v.checked_add(scales.checked_mul(12)?))
            .ok_or_else(size_overflow)?;
        ledger.state_payload_bytes = ledger
            .state_payload_bytes
            .checked_add(payload)
            .ok_or_else(size_overflow)?;
        ledger.update_workspace_bytes = ledger
            .update_workspace_bytes
            .checked_add(scales.checked_mul(16).ok_or_else(size_overflow)?)
            .ok_or_else(size_overflow)?;
        let width = scale_width(weight.layout().law.precision)?;
        ledger.checkpoint_bytes = ledger
            .checkpoint_bytes
            .checked_add(8)
            .and_then(|v| v.checked_add(scales.checked_mul(8 + width)?))
            .ok_or_else(size_overflow)?;
    }
    Ok(ledger)
}

fn frame(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}
fn size_overflow() -> PvTuningError {
    PvTuningError::invalid_config("native state size overflow")
}
fn scale_width(precision: ScalePrecision) -> Result<u64, PvTuningError> {
    match precision {
        ScalePrecision::F16 | ScalePrecision::Bf16 => Ok(2),
        ScalePrecision::F32 => Ok(4),
        _ => Err(PvTuningError::invalid_weight(
            "unsupported native scale precision",
        )),
    }
}
fn round_scale(value: f32, precision: ScalePrecision) -> Result<f32, PvTuningError> {
    let value = match precision {
        ScalePrecision::F16 => f16::from_f32(value).to_f32(),
        ScalePrecision::Bf16 => bf16::from_f32(value).to_f32(),
        ScalePrecision::F32 => value,
        _ => return Err(PvTuningError::step("unsupported native scale precision")),
    };
    if !value.is_finite() {
        return Err(PvTuningError::step(
            "native scale update overflowed stored precision",
        ));
    }
    Ok(value)
}
fn reserved<T>(len: usize) -> Result<Vec<T>, PvTuningError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| PvTuningError::step("native buffer allocation failed"))?;
    Ok(values)
}
fn copied<T: Copy>(values: &[T]) -> Result<Vec<T>, PvTuningError> {
    let mut result = reserved(values.len())?;
    result.extend_from_slice(values);
    Ok(result)
}
fn zeroed(len: usize) -> Result<Vec<f32>, PvTuningError> {
    let mut values = reserved(len)?;
    values.resize(len, 0.0);
    Ok(values)
}

fn check_corrected_moments(
    optimizer: AdamW,
    step: u64,
    state: &AdamState,
) -> Result<(), &'static str> {
    // Same f32 bias-correction denominators as common AdamW. A finite stored v
    // can still overflow v/bc2, silently turning the adaptive update into zero.
    // Guard the native transaction without changing the shared optimizer law
    // or reinterpreting legacy PV checkpoints/failure policy.
    let exponent = i32::try_from(step).unwrap_or(i32::MAX);
    let bc1 = 1.0 - optimizer.beta1.powi(exponent);
    let bc2 = 1.0 - optimizer.beta2.powi(exponent);
    if state.m.iter().any(|m| !(m / bc1).is_finite())
        || state.v.iter().any(|v| !(v / bc2).is_finite())
    {
        return Err("native bias-corrected optimizer moment became non-finite");
    }
    Ok(())
}

// One native continuous law for scale-only and alternating P/V sessions.
fn prepare_scale_candidate(
    weight: &AdditiveTensor,
    state: &AdamState,
    optimizer: AdamW,
    next: u64,
    window_weight: f64,
    gradient: &[f32],
) -> Result<(usize, ScaleCandidate), PvTuningError> {
    let mut weighted = reserved(gradient.len())?;
    for &value in gradient.iter() {
        let value = (f64::from(value) * window_weight) as f32;
        if !value.is_finite() {
            return Err(PvTuningError::step(
                "weighted native scale gradient overflowed f32",
            ));
        }
        weighted.push(value);
    }
    let mut values = copied(weight.scales())?;
    let mut state = AdamState {
        m: copied(&state.m)?,
        v: copied(&state.v)?,
    };
    optimizer.step(next, &mut values, &weighted, &mut state);
    check_optimizer_moments(&state, "native scale")?;
    check_corrected_moments(optimizer, next, &state).map_err(PvTuningError::step)?;
    for value in &mut values {
        if !value.is_finite() {
            return Err(PvTuningError::step(
                "native optimizer produced a non-finite scale",
            ));
        }
        *value = round_scale(value.max(0.0), weight.layout().law.precision)?;
    }
    weight
        .validate_scales(&values)
        .map_err(|e| PvTuningError::step(format!("native scale update is invalid: {e:?}")))?;
    let changed = values
        .iter()
        .zip(weight.scales())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    Ok((changed, ScaleCandidate { values, state }))
}
