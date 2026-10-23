use super::{NativeScaleRefinementSession, NativeScaleStageError, PvTuningError, reserved, zeroed};
use tritium_core::{AdditiveView, accumulate_embed_scale_vjp, accumulate_ternary_matmul_scale_vjp};
use tritium_schema::NativeRefinementPlan;

/// One explicit native consumer contribution to a selected owner's scale gradient.
///
/// Lists are supplied in owner-plan order, then consumer occurrence order. This
/// is numerical input, not proof of an authenticated graph or complete use list.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum NativeScaleContribution<'a> {
    /// A Linear/matmul consumer in original activation coordinates.
    Matmul {
        /// Row-major `batch * cols` f32 activations, before the stored basis.
        activations: &'a [f32],
        /// Row-major `batch * rows` output cotangents, unweighted by the window.
        output_gradients: &'a [f32],
        /// Explicit batch size, including zero.
        batch: usize,
    },
    /// An embedding/gather consumer in original output coordinates.
    Embed {
        /// Occurrence-ordered row indices; duplicates contribute repeatedly.
        ids: &'a [usize],
        /// Row-major `ids.len() * cols` cotangents, unweighted by the window.
        output_gradients: &'a [f32],
    },
}

/// P/V's original-coordinate consumer inputs, shared with scale-only recovery.
/// The reducer derives distinct stored-scale and present-weight adjoints; this
/// alias does not reinterpret scale gradients or authenticate a complete graph.
pub type NativePvContribution<'a> = NativeScaleContribution<'a>;

impl NativeScaleRefinementSession {
    /// Reduce tied/repeated consumer cotangents and perform one atomic scale step.
    ///
    /// Supply one ordered consumer slice per selected owner. The shared core
    /// accumulates actual stored-scale gradients in f64, narrowed once to f32;
    /// [`Self::step`] applies the declared window weight exactly once. An empty
    /// owner list means zero gradient, **not** a skipped optimizer/decay update.
    /// No logical dense-weight gradient or second optimizer is constructed.
    ///
    /// Checks the schedule, every shape/index/finite input, and the conservative
    /// workspace peak `max(4S+8M+4C,20S)` before native scratch allocation: total
    /// scales S, largest owner scale count M and largest column count C. Includes
    /// the reusable basis row, but excludes metadata/caller inputs/graph memory.
    /// The direct-step ledger and TNSR checkpoint format remain unchanged.
    pub fn step_from_consumers(
        &mut self,
        contributions: &[&[NativeScaleContribution<'_>]],
    ) -> Result<usize, PvTuningError> {
        let gradients = self
            .consumer_gradients(contributions, |_, view, contribution, scratch, out| {
                match contribution {
                    NativeScaleContribution::Matmul {
                        activations,
                        output_gradients,
                        batch,
                    } => accumulate_ternary_matmul_scale_vjp(
                        activations,
                        &view.bind_matmul(),
                        output_gradients,
                        batch,
                        scratch,
                        out,
                    ),
                    NativeScaleContribution::Embed {
                        ids,
                        output_gradients,
                    } => accumulate_embed_scale_vjp(
                        &view.bind_gather(),
                        ids,
                        output_gradients,
                        scratch,
                        out,
                    ),
                }
                .map_err(|e| {
                    PvTuningError::step(format!("native scale adjoint rejected consumer: {e:?}"))
                })
            })
            .map_err(|e| match e {
                NativeScaleStageError::Update(e) | NativeScaleStageError::Stage(e) => e,
            })?;
        let mut borrowed = reserved(gradients.len())?;
        borrowed.extend(gradients.iter().map(Vec::as_slice));
        self.step(&borrowed)
    }

    pub(super) fn consumer_gradients<E>(
        &self,
        contributions: &[&[NativeScaleContribution<'_>]],
        mut consume: impl FnMut(
            usize,
            AdditiveView<'_>,
            NativeScaleContribution<'_>,
            &mut [f32],
            &mut [f64],
        ) -> Result<(), E>,
    ) -> Result<Vec<Vec<f32>>, NativeScaleStageError<E>> {
        reduce(
            &self.plan,
            self.completed_updates,
            self.ledger.update_workspace_bytes,
            false,
            |index| self.owners[index].weight.view(),
            contributions,
            |index, view, contribution, row, scales, _| {
                consume(index, view, contribution, row, scales)
            },
        )
    }
}

// One checked reducer for both profiles and decoded/packed readers. The
// callback computes actual adjoints; flattened owner vectors are private, not
// per-plane gradients. Metadata and caller inputs remain outside logical caps.
pub(super) fn reduce<'a, E>(
    plan: &NativeRefinementPlan,
    completed: u64,
    update_workspace: u64,
    include_weights: bool,
    view_at: impl Fn(usize) -> AdditiveView<'a>,
    contributions: &[&[NativeScaleContribution<'_>]],
    mut consume: impl FnMut(
        usize,
        AdditiveView<'a>,
        NativeScaleContribution<'_>,
        &mut [f32],
        &mut [f64],
        &mut [f64],
    ) -> Result<(), E>,
) -> Result<Vec<Vec<f32>>, NativeScaleStageError<E>> {
    if completed >= plan.training.len() as u64 {
        return Err(PvTuningError::step("native training schedule is exhausted").into());
    }
    if contributions.len() != plan.owners.len() {
        return Err(PvTuningError::step("native consumer owner count mismatch").into());
    }
    let combined = |view: AdditiveView<'_>| -> Result<usize, PvTuningError> {
        let weights = if include_weights && view.plane_count() != 0 {
            view.plane_trits(0).map_err(|_| workspace_overflow())?.len()
        } else {
            0
        };
        view.scales()
            .len()
            .checked_add(weights)
            .ok_or_else(workspace_overflow)
    };
    let mut total = 0u64;
    let mut max_gradients = 0usize;
    let mut max_cols = 0usize;
    for index in 0..plan.owners.len() {
        let view = view_at(index);
        let count = combined(view)?;
        max_gradients = max_gradients.max(count);
        max_cols = max_cols.max(view.layout().cols as usize);
        total = total
            .checked_add(count as u64)
            .ok_or_else(workspace_overflow)?;
    }
    let retained = total.checked_mul(4).ok_or_else(workspace_overflow)?;
    let accumulation = retained
        .checked_add(
            (max_gradients as u64)
                .checked_mul(8)
                .ok_or_else(workspace_overflow)?,
        )
        .and_then(|n| n.checked_add((max_cols as u64).checked_mul(4)?))
        .ok_or_else(workspace_overflow)?;
    let peak = accumulation.max(
        retained
            .checked_add(update_workspace)
            .ok_or_else(workspace_overflow)?,
    );
    if peak > plan.limits.max_workspace_bytes {
        return Err(PvTuningError::step(format!(
            "native consumer workspace requires {peak} bytes, limit is {}",
            plan.limits.max_workspace_bytes
        ))
        .into());
    }
    // All input scans precede every native gradient/scratch allocation.
    for (index, uses) in contributions.iter().enumerate() {
        let view = view_at(index);
        for contribution in *uses {
            contribution.validate(view.layout().rows as usize, view.layout().cols as usize)?;
        }
    }
    let mut row = zeroed(max_cols)?;
    let mut accumulated = reserved(max_gradients)?;
    accumulated.resize(max_gradients, 0.0f64);
    let mut gradients = reserved(plan.owners.len())?;
    for (index, uses) in contributions.iter().enumerate() {
        let view = view_at(index);
        let out = &mut accumulated[..combined(view)?];
        out.fill(0.0);
        let (scales, weights) = out.split_at_mut(view.scales().len());
        let scratch = &mut row[..view.layout().cols as usize];
        for contribution in *uses {
            consume(index, view, *contribution, scratch, scales, weights)
                .map_err(NativeScaleStageError::Stage)?;
        }
        let mut values = reserved(out.len())?;
        for &value in out.iter() {
            let value = value as f32;
            if !value.is_finite() {
                return Err(PvTuningError::step(if include_weights {
                    "native accumulated gradient overflowed f32"
                } else {
                    "native accumulated scale gradient overflowed f32"
                })
                .into());
            }
            values.push(value);
        }
        gradients.push(values);
    }
    Ok(gradients)
}

impl NativeScaleContribution<'_> {
    fn validate(self, rows: usize, cols: usize) -> Result<(), PvTuningError> {
        let shape_error = || PvTuningError::step("native consumer shape/index mismatch");
        let (inputs, gradients) = match self {
            Self::Matmul {
                activations,
                output_gradients,
                batch,
            } => {
                if batch.checked_mul(cols) != Some(activations.len())
                    || batch.checked_mul(rows) != Some(output_gradients.len())
                {
                    return Err(shape_error());
                }
                (activations, output_gradients)
            }
            Self::Embed {
                ids,
                output_gradients,
            } => {
                if ids.len().checked_mul(cols) != Some(output_gradients.len())
                    || ids.iter().any(|&id| id >= rows)
                {
                    return Err(shape_error());
                }
                (&[][..], output_gradients)
            }
        };
        if inputs.iter().chain(gradients).any(|v| !v.is_finite()) {
            return Err(PvTuningError::step(
                "native consumer contains a non-finite activation/cotangent",
            ));
        }
        Ok(())
    }
}

fn workspace_overflow() -> PvTuningError {
    PvTuningError::step("native consumer workspace size overflow")
}
