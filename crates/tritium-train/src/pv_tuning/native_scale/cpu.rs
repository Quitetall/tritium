//! Opt-in owned numerical CPU binding; not model/source or release admission.
use super::{
    NativeScaleContribution, NativeScaleRefinementSession, NativeScaleStageError, reserved,
};
use std::sync::Arc;
use tritium_cpu::CpuBackend;
use tritium_runtime::TensorPayloadBudget;
use tritium_spec::{BackendError, DeviceBuffer};

/// Numeric binding/preflight/update or packed snapshot/gradient staging failure.
pub type NativeCpuScaleError = NativeScaleStageError<BackendError>;

/// Common scale state and a complete immutable CPU handle set committed together.
///
/// Still owns decoded reference state, not production packed/out-of-core optimizer
/// storage. Does not authenticate graph/data/source/objective, emit completion
/// evidence or publish a refined child. Native public Python training is separate.
pub struct NativeCpuScaleSession {
    session: NativeScaleRefinementSession,
    handles: Vec<Arc<dyn DeviceBuffer>>,
    budget: TensorPayloadBudget,
}
impl std::fmt::Debug for NativeCpuScaleSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCpuScaleSession")
            .field("plan", &self.session.plan_id())
            .field("completed_updates", &self.session.completed_updates())
            .field("owners", &self.handles.len())
            .finish_non_exhaustive()
    }
}
impl NativeCpuScaleSession {
    /// Consume common state and named handles in exact selected-owner order.
    /// Privately pins scale versions and checks actual packed data against state
    /// before returning the binding. Initial scale payloads need budget admission;
    /// later steps need temporary room for both old/new versions. Failed binding
    /// drops these consumed inputs/staged resources, not external model files.
    pub fn bind(
        session: NativeScaleRefinementSession,
        owners: Vec<(String, Arc<dyn DeviceBuffer>)>,
        budget: TensorPayloadBudget,
    ) -> Result<Self, NativeCpuScaleError> {
        if owners.len() != session.plan.owners.len()
            || owners
                .iter()
                .zip(&session.plan.owners)
                .any(|((name, _), expected)| name != expected)
        {
            return Err(stage_error("CPU binding owner names/order mismatch"));
        }
        let mut handles = handle_storage(owners.len()).map_err(NativeScaleStageError::Stage)?;
        for ((_, parent), weight) in owners.into_iter().zip(session.weights()) {
            let (snapshot, _) = CpuBackend::new()
                .snapshot_additive_scales(&parent, weight.scales(), &budget)
                .map_err(NativeScaleStageError::Stage)?;
            if !CpuBackend::new()
                .additive_equals_view(&*snapshot, weight.view())
                .map_err(NativeScaleStageError::Stage)?
            {
                return Err(stage_error(
                    "CPU binding packed state differs from common owner state",
                ));
            }
            handles.push(snapshot);
        }
        Ok(Self {
            session,
            handles,
            budget,
        })
    }

    /// Read committed state/checkpoint without permitting an independent update.
    #[must_use]
    pub fn session(&self) -> &NativeScaleRefinementSession {
        &self.session
    }

    /// Cloneable immutable versions in canonical owner order; old graphs may
    /// retain them, continuing to hold their payload admission until final drop.
    pub fn handles(&self) -> impl ExactSizeIterator<Item = (&str, &Arc<dyn DeviceBuffer>)> {
        self.session
            .plan
            .owners
            .iter()
            .zip(&self.handles)
            .map(|(name, handle)| (name.as_str(), handle))
    }

    /// Stage every next handle before committing common scales/moments/cursor
    /// and replacing the complete current set. Any error leaves both unchanged.
    pub fn step(&mut self, gradients: &[&[f32]]) -> Result<usize, NativeCpuScaleError> {
        let handles = &self.handles;
        let budget = &self.budget;
        let (changed, staged) = self.session.step_staged(gradients, |candidate| {
            let mut staged = handle_storage(handles.len())?;
            for ((_, scales), parent) in candidate.owners().zip(handles) {
                staged.push(
                    CpuBackend::new()
                        .snapshot_additive_scales(parent, scales, budget)?
                        .0,
                );
            }
            Ok::<_, BackendError>(staged)
        })?;
        self.handles = staged;
        Ok(changed)
    }

    /// Reduce actual packed Matmul/Embed cotangents through the shared reducer,
    /// then atomically update the complete bound owner set. Window weight is
    /// applied once by the common engine; empty owner uses mean zero gradients.
    pub fn step_from_consumers(
        &mut self,
        contributions: &[&[NativeScaleContribution<'_>]],
    ) -> Result<usize, NativeCpuScaleError> {
        let handles = &self.handles;
        let gradients = self.session.consumer_gradients(
            contributions,
            |index, _, contribution, row, out| {
                let cpu = CpuBackend::new();
                let handle = &*handles[index];
                match contribution {
                    NativeScaleContribution::Matmul {
                        activations,
                        output_gradients,
                        batch,
                    } => cpu.accumulate_additive_matmul_scale_vjp(
                        handle,
                        activations,
                        output_gradients,
                        batch,
                        row,
                        out,
                    ),
                    NativeScaleContribution::Embed {
                        ids,
                        output_gradients,
                    } => cpu.accumulate_additive_embed_scale_vjp(
                        handle,
                        ids,
                        output_gradients,
                        row,
                        out,
                    ),
                }
            },
        )?;
        let mut borrowed = reserved(gradients.len())?;
        borrowed.extend(gradients.iter().map(Vec::as_slice));
        self.step(&borrowed)
    }
}

fn stage_error(message: &str) -> NativeCpuScaleError {
    NativeScaleStageError::Stage(BackendError::InvalidInput(message.into()))
}
fn handle_storage(len: usize) -> Result<Vec<Arc<dyn DeviceBuffer>>, BackendError> {
    let requested = len
        .checked_mul(std::mem::size_of::<Arc<dyn DeviceBuffer>>())
        .ok_or_else(|| BackendError::InvalidInput("CPU handle metadata size overflow".into()))?;
    let mut handles = Vec::new();
    handles
        .try_reserve_exact(len)
        .map_err(|_| BackendError::OutOfMemory { requested })?;
    Ok(handles)
}
