use super::{NativeScaleRefinementSession, ScaleCandidate};
use crate::pv_tuning::PvTuningError;
use tritium_schema::NativeRefinementPlanId;

/// Borrowed fully checked next-scale values; not committed state or a receipt.
/// References cannot outlive the staging callback. No optimizer moments/codes
/// are exposed or copied, and owners follow the frozen request order.
pub struct NativeScaleStep<'a> {
    pub(super) names: &'a [String],
    pub(super) candidates: &'a [ScaleCandidate],
    pub(super) next: u64,
    pub(super) plan: NativeRefinementPlanId,
}
impl std::fmt::Debug for NativeScaleStep<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeScaleStep")
            .field("plan", &self.plan)
            .field("proposed_update", &self.next)
            .field("owners", &self.names.len())
            .finish_non_exhaustive()
    }
}
impl NativeScaleStep<'_> {
    /// Native request being proposed, not a legacy optimizer recipe identity.
    #[must_use]
    pub const fn plan_id(&self) -> NativeRefinementPlanId {
        self.plan
    }
    /// Cursor that becomes committed only if staging succeeds.
    #[must_use]
    pub const fn proposed_update(&self) -> u64 {
        self.next
    }
    /// Exact proposed stored scales for every canonical owner, in plan order.
    pub fn owners(&self) -> impl ExactSizeIterator<Item = (&str, &[f32])> {
        self.names
            .iter()
            .zip(self.candidates)
            .map(|(name, candidate)| (name.as_str(), candidate.values.as_slice()))
    }
}

/// Distinguish common numerical/preflight failure from unchanged caller staging failure.
#[derive(Debug, PartialEq, Eq)]
pub enum NativeScaleStageError<E> {
    /// The common update/reduction failed before any state commit.
    Update(PvTuningError),
    /// Staging failed; the caller's original error is preserved.
    Stage(E),
}
impl<E> From<PvTuningError> for NativeScaleStageError<E> {
    fn from(value: PvTuningError) -> Self {
        Self::Update(value)
    }
}
impl<E: std::fmt::Display> std::fmt::Display for NativeScaleStageError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Update(e) => e.fmt(f),
            Self::Stage(e) => write!(f, "native scale staging failed: {e}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for NativeScaleStageError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Update(e) => Some(e),
            Self::Stage(e) => Some(e),
        }
    }
}

impl NativeScaleRefinementSession {
    /// Stage resources for every fully checked candidate before committing state.
    ///
    /// Error or callback unwinding preserves scales, moments and cursor. The
    /// callback must keep resources private and return their owner as T; arbitrary
    /// external effects are not rolled back. Staging resource/metadata/graph costs
    /// are the caller's obligation, not complete admission by the update ledger.
    /// No coefficient array is cloned. The numerical recipe/checkpoint is unchanged.
    ///
    /// ```compile_fail
    /// use tritium_train::pv_tuning::NativeScaleRefinementSession;
    /// fn escape(s:&mut NativeScaleRefinementSession,g:&[&[f32]]) {
    ///     let _ = s.step_staged(g, |candidate| Ok::<_,()>(candidate));
    /// }
    /// ```
    pub fn step_staged<T, E>(
        &mut self,
        gradients: &[&[f32]],
        stage: impl for<'a> FnOnce(NativeScaleStep<'a>) -> Result<T, E>,
    ) -> Result<(usize, T), NativeScaleStageError<E>> {
        let prepared = self.prepare_step(gradients)?;
        let staged = stage(NativeScaleStep {
            names: &self.plan.owners,
            candidates: &prepared.candidates,
            next: prepared.next,
            plan: self.plan_id,
        })
        .map_err(NativeScaleStageError::Stage)?;
        let changed = prepared.changed;
        self.commit_step(prepared);
        Ok((changed, staged))
    }
}
