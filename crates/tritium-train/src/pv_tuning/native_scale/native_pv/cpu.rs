//! Actual packed consumers/versions around the common P/V transaction.
use super::{
    NativePvCandidate, NativePvContribution, NativePvJournalError, NativePvJournalWriter,
    NativePvRefinementSession, NativePvStageError, NativePvStepStats, consumers::ConsumerGradients,
};
use std::{
    io::{Seek, Write},
    sync::Arc,
};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_runtime::TensorPayloadBudget;
use tritium_spec::{BackendError, DeviceBuffer};

/// Common numerical rejection versus the original packed/backend failure.
pub type NativeCpuPvError = NativePvStageError<BackendError>;

/// Original packed or journal staging failure before common/handle commit.
#[derive(Debug)]
#[non_exhaustive]
pub enum NativeCpuPvJournalStageError {
    /// Packed adjoint/version/budget failure; journal was not advanced.
    Packed(BackendError),
    /// Journal failure; all staged new versions are dropped.
    Journal(NativePvJournalError),
}
impl std::fmt::Display for NativeCpuPvJournalStageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Packed(e) => e.fmt(f),
            Self::Journal(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for NativeCpuPvJournalStageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Packed(e) => Some(e),
            Self::Journal(e) => Some(e),
        }
    }
}

/// Common P/V state and all immutable CPU versions committed together.
///
/// Owns decoded reference state plus separately admitted packed roots, not a
/// production GPU/paged optimizer. Does not authenticate source, lineage, graph,
/// data or teacher, publish a child, enable Python P/V or qualify model quality.
pub struct NativeCpuPvSession {
    session: NativePvRefinementSession,
    handles: Vec<Arc<dyn DeviceBuffer>>,
    budget: TensorPayloadBudget,
}
impl std::fmt::Debug for NativeCpuPvSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCpuPvSession")
            .field("plan", &self.session.plan_id())
            .field("completed_updates", &self.session.completed_updates())
            .field("owners", &self.handles.len())
            .finish_non_exhaustive()
    }
}
impl NativeCpuPvSession {
    /// Bind exact CURRENT codes/scales/geometry and canonical owner names/order,
    /// then privately admit complete packed versions. Resumed common state needs
    /// matching current hard handles, not original parent codes. Initial packed
    /// roots need budget room; later steps need both old/new roots until commit.
    /// Refusal drops consumed inputs/staged resources, never external model files.
    pub fn bind(
        session: NativePvRefinementSession,
        owners: Vec<(String, Arc<dyn DeviceBuffer>)>,
        budget: TensorPayloadBudget,
    ) -> Result<Self, NativeCpuPvError> {
        if owners.len() != session.plan.owners.len()
            || owners
                .iter()
                .zip(&session.plan.owners)
                .any(|((name, _), expected)| name != expected)
        {
            return Err(binding_error("CPU PV owner names/order mismatch"));
        }
        for ((_, parent), weight) in owners.iter().zip(session.weights()) {
            if !CpuBackend::new()
                .additive_equals_view(&**parent, weight.view())
                .map_err(NativePvStageError::Stage)?
            {
                return Err(binding_error(
                    "CPU PV packed state differs from current common state",
                ));
            }
        }
        let handles = snapshots(owners.iter().map(|(_, h)| h), session.weights(), &budget)
            .map_err(NativePvStageError::Stage)?;
        Ok(Self {
            session,
            handles,
            budget,
        })
    }

    /// Read committed common state/checkpoints without an independent update.
    #[must_use]
    pub fn session(&self) -> &NativePvRefinementSession {
        &self.session
    }

    /// Current named immutable versions; old graph Arcs retain old admission.
    pub fn handles(&self) -> impl ExactSizeIterator<Item = (&str, &Arc<dyn DeviceBuffer>)> {
        self.session
            .plan
            .owners
            .iter()
            .zip(&self.handles)
            .map(|(n, h)| (n.as_str(), h))
    }

    /// Stage every packed owner inside the common candidate transaction, then
    /// commit all codes/scales/moments/cursor/handles under this exclusive borrow.
    /// Any error or unwind leaves numerical state and old versions unchanged.
    pub fn step(
        &mut self,
        scales: &[&[f32]],
        weights: &[&[f32]],
    ) -> Result<NativePvStepStats, NativeCpuPvError> {
        let handles = &self.handles;
        let budget = &self.budget;
        let (stats, staged) = self.session.step_staged(scales, weights, |candidate| {
            stage(handles, budget, &candidate)
        })?;
        self.handles = staged;
        Ok(stats)
    }

    /// Stage ALL packed versions before writing/flushing TNPJ, then commit the
    /// common state and complete current set. Snapshot refusal writes nothing.
    /// Journal failure/unwind drops staged versions, preserves old state, and
    /// follows the existing writer poisoning rules. Flush is not fsync/durability.
    pub fn step_recorded<W: Write + Seek>(
        &mut self,
        scales: &[&[f32]],
        weights: &[&[f32]],
        journal: &mut NativePvJournalWriter<W>,
    ) -> Result<NativePvStepStats, NativePvStageError<NativeCpuPvJournalStageError>> {
        journal
            .check(&self.session)
            .map_err(|e| NativePvStageError::Stage(NativeCpuPvJournalStageError::Journal(e)))?;
        let ordinal = self.session.completed;
        let window = self
            .session
            .plan
            .training
            .get(ordinal as usize)
            .map(|w| w.content);
        let handles = &self.handles;
        let budget = &self.budget;
        let (stats, staged) = self.session.step_staged(scales, weights, |candidate| {
            let staged =
                stage(handles, budget, &candidate).map_err(NativeCpuPvJournalStageError::Packed)?;
            journal
                .append(ordinal, window.expect("admitted schedule"), scales, weights)
                .map_err(NativeCpuPvJournalStageError::Journal)?;
            Ok::<_, NativeCpuPvJournalStageError>(staged)
        })?;
        self.handles = staged;
        Ok(stats)
    }

    /// Reduce actual packed Matmul/Embed scale and present-weight adjoints through
    /// the shared bounded reducer, then atomically update all owners. Window
    /// weight is applied once by the common optimizer; empty uses are zero input.
    pub fn step_from_consumers(
        &mut self,
        contributions: &[&[NativePvContribution<'_>]],
    ) -> Result<NativePvStepStats, NativeCpuPvError> {
        let gradients = self.gradients(contributions)?;
        let adjoints = gradients.borrow(&self.session)?;
        self.step(&adjoints.scales, &adjoints.weights)
    }

    /// Actual packed consumer reduction plus all-owner packed/journal commit.
    /// Preflight, adjoint or snapshot failure cannot advance the journal.
    pub fn step_from_consumers_recorded<W: Write + Seek>(
        &mut self,
        contributions: &[&[NativePvContribution<'_>]],
        journal: &mut NativePvJournalWriter<W>,
    ) -> Result<NativePvStepStats, NativePvStageError<NativeCpuPvJournalStageError>> {
        journal
            .check(&self.session)
            .map_err(|e| NativePvStageError::Stage(NativeCpuPvJournalStageError::Journal(e)))?;
        let gradients = self.gradients(contributions).map_err(|e| match e {
            NativePvStageError::Update(e) => NativePvStageError::Update(e),
            NativePvStageError::Stage(e) => {
                NativePvStageError::Stage(NativeCpuPvJournalStageError::Packed(e))
            }
        })?;
        let adjoints = gradients.borrow(&self.session)?;
        self.step_recorded(&adjoints.scales, &adjoints.weights, journal)
    }

    fn gradients(
        &self,
        contributions: &[&[NativePvContribution<'_>]],
    ) -> Result<ConsumerGradients, NativeCpuPvError> {
        self.session.consumer_gradients(
            contributions,
            |index, _, contribution, row, scales, weights| {
                let cpu = CpuBackend::new();
                let handle = &*self.handles[index];
                match contribution {
                    NativePvContribution::Matmul {
                        activations,
                        output_gradients,
                        batch,
                    } => {
                        cpu.accumulate_additive_matmul_scale_vjp(
                            handle,
                            activations,
                            output_gradients,
                            batch,
                            row,
                            scales,
                        )?;
                        cpu.accumulate_additive_matmul_weight_vjp(
                            handle,
                            activations,
                            output_gradients,
                            batch,
                            row,
                            weights,
                        )
                    }
                    NativePvContribution::Embed {
                        ids,
                        output_gradients,
                    } => {
                        cpu.accumulate_additive_embed_scale_vjp(
                            handle,
                            ids,
                            output_gradients,
                            row,
                            scales,
                        )?;
                        cpu.accumulate_additive_embed_weight_vjp(
                            handle,
                            ids,
                            output_gradients,
                            row,
                            weights,
                        )
                    }
                }
            },
        )
    }
}

fn snapshots<'a>(
    parents: impl ExactSizeIterator<Item = &'a Arc<dyn DeviceBuffer>>,
    weights: impl ExactSizeIterator<Item = &'a AdditiveTensor>,
    budget: &TensorPayloadBudget,
) -> Result<Vec<Arc<dyn DeviceBuffer>>, BackendError> {
    if parents.len() != weights.len() {
        return Err(BackendError::InvalidInput(
            "CPU PV candidate owner inventory mismatch".into(),
        ));
    }
    let mut result = super::super::cpu::handle_storage(parents.len())?;
    for (parent, weight) in parents.zip(weights) {
        result.push(CpuBackend::new().snapshot_additive_weights(parent, weight.view(), budget)?);
    }
    Ok(result)
}
fn stage(
    handles: &[Arc<dyn DeviceBuffer>],
    budget: &TensorPayloadBudget,
    candidate: &NativePvCandidate<'_>,
) -> Result<Vec<Arc<dyn DeviceBuffer>>, BackendError> {
    snapshots(
        handles.iter(),
        candidate.owners().map(|(_, weight)| weight),
        budget,
    )
}
fn binding_error(message: &str) -> NativeCpuPvError {
    NativePvStageError::Stage(BackendError::InvalidInput(message.into()))
}
