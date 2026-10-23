use super::{
    NativePvContribution, NativePvJournalError, NativePvJournalWriter, NativePvRefinementSession,
    NativePvStageError, NativePvStepStats, NativeScaleStageError, PvTuningError, reserved,
};
use std::io::{Seek, Write};
use tritium_core::{
    AdditiveView, accumulate_embed_scale_vjp, accumulate_embed_weight_vjp,
    accumulate_ternary_matmul_scale_vjp, accumulate_ternary_matmul_weight_vjp,
};

pub(super) struct ConsumerGradients(Vec<Vec<f32>>);
pub(super) struct BorrowedAdjoints<'a> {
    pub(super) scales: Vec<&'a [f32]>,
    pub(super) weights: Vec<&'a [f32]>,
}
impl ConsumerGradients {
    pub(super) fn borrow<'a>(
        &'a self,
        session: &NativePvRefinementSession,
    ) -> Result<BorrowedAdjoints<'a>, PvTuningError> {
        let mut scales = reserved(self.0.len())?;
        let mut weights = reserved(self.0.len())?;
        for (owner, values) in session.owners.iter().zip(&self.0) {
            let (s, w) = values.split_at(owner.weight.scales().len());
            scales.push(s);
            weights.push(w);
        }
        Ok(BorrowedAdjoints { scales, weights })
    }
}

impl NativePvRefinementSession {
    /// Reduce ordered Matmul/Embed uses into separate stored-scale and present
    /// stored-weight adjoints, then perform the existing atomic P/V step.
    ///
    /// Uses shared f64 accumulation, narrowed once per owner; the common optimizer
    /// applies window weight once. Empty owner uses still receive optimizer/decay.
    /// Before input scans/allocation, checks conservative workspace
    /// max(4G+8M+4C,4G+L): total combined coordinates G, maximum owner count M,
    /// maximum columns C, existing update workspace L. Excludes metadata, caller
    /// inputs, external graphs and separately admitted packed versions. This is
    /// numerical input, not objective/data/teacher or complete-graph custody.
    pub fn step_from_consumers(
        &mut self,
        contributions: &[&[NativePvContribution<'_>]],
    ) -> Result<NativePvStepStats, PvTuningError> {
        let gradients = self.decoded_gradients(contributions)?;
        let adjoints = gradients.borrow(self)?;
        self.step(&adjoints.scales, &adjoints.weights)
    }

    /// Same consumer reduction, recording the exact unweighted adjoints through
    /// TNPJ before the common atomic commit. Numerical refusal writes nothing;
    /// write/flush failure poisons the writer and preserves common state.
    /// Generic flush is not fsync, durable recovery or objective authentication.
    pub fn step_from_consumers_recorded<W: Write + Seek>(
        &mut self,
        contributions: &[&[NativePvContribution<'_>]],
        journal: &mut NativePvJournalWriter<W>,
    ) -> Result<NativePvStepStats, NativePvStageError<NativePvJournalError>> {
        journal.check(self).map_err(NativePvStageError::Stage)?;
        let gradients = self.decoded_gradients(contributions)?;
        let adjoints = gradients.borrow(self)?;
        self.step_recorded(&adjoints.scales, &adjoints.weights, journal)
    }

    pub(super) fn consumer_gradients<E>(
        &self,
        contributions: &[&[NativePvContribution<'_>]],
        consume: impl FnMut(
            usize,
            AdditiveView<'_>,
            NativePvContribution<'_>,
            &mut [f32],
            &mut [f64],
            &mut [f64],
        ) -> Result<(), E>,
    ) -> Result<ConsumerGradients, NativePvStageError<E>> {
        super::super::consumers::reduce(
            &self.plan,
            self.completed,
            self.ledger.update_workspace_bytes,
            true,
            |index| self.owners[index].weight.view(),
            contributions,
            consume,
        )
        .map(ConsumerGradients)
        .map_err(|e| match e {
            NativeScaleStageError::Update(e) => NativePvStageError::Update(e),
            NativeScaleStageError::Stage(e) => NativePvStageError::Stage(e),
        })
    }

    fn decoded_gradients(
        &self,
        contributions: &[&[NativePvContribution<'_>]],
    ) -> Result<ConsumerGradients, PvTuningError> {
        self.consumer_gradients(
            contributions,
            |_, view, contribution, row, scales, weights| {
                let result = match contribution {
                    NativePvContribution::Matmul {
                        activations,
                        output_gradients,
                        batch,
                    } => accumulate_ternary_matmul_scale_vjp(
                        activations,
                        &view.bind_matmul(),
                        output_gradients,
                        batch,
                        row,
                        scales,
                    )
                    .and_then(|()| {
                        accumulate_ternary_matmul_weight_vjp(
                            activations,
                            &view.bind_matmul(),
                            output_gradients,
                            batch,
                            row,
                            weights,
                        )
                    }),
                    NativePvContribution::Embed {
                        ids,
                        output_gradients,
                    } => accumulate_embed_scale_vjp(
                        &view.bind_gather(),
                        ids,
                        output_gradients,
                        row,
                        scales,
                    )
                    .and_then(|()| {
                        accumulate_embed_weight_vjp(
                            &view.bind_gather(),
                            ids,
                            output_gradients,
                            row,
                            weights,
                        )
                    }),
                };
                result.map_err(|e| {
                    PvTuningError::step(format!("native PV adjoint rejected consumer: {e:?}"))
                })
            },
        )
        .map_err(|e| match e {
            NativePvStageError::Update(e) | NativePvStageError::Stage(e) => e,
        })
    }
}
