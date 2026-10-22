use super::{NativeScaleRefinementSession, PvTuningError, reserved, zeroed};
use tritium_core::{accumulate_embed_scale_vjp, accumulate_ternary_matmul_scale_vjp};

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
        if self.completed_updates >= self.plan.training.len() as u64 {
            return Err(PvTuningError::step("native training schedule is exhausted"));
        }
        if contributions.len() != self.owners.len() {
            return Err(PvTuningError::step("native consumer owner count mismatch"));
        }
        let mut total = 0u64;
        let mut max_scales = 0usize;
        let mut max_cols = 0usize;
        // Reject known buffer-budget failures without scanning activation data.
        for owner in &self.owners {
            let cols = owner.weight.layout().cols as usize;
            let scales = owner.weight.scales().len();
            max_scales = max_scales.max(scales);
            max_cols = max_cols.max(cols);
            total = total
                .checked_add(scales as u64)
                .ok_or_else(workspace_overflow)?;
        }
        let accumulation = total
            .checked_mul(4)
            .and_then(|n| n.checked_add((max_scales as u64).checked_mul(8)?))
            .and_then(|n| n.checked_add((max_cols as u64).checked_mul(4)?))
            .ok_or_else(workspace_overflow)?;
        let peak = accumulation.max(total.checked_mul(20).ok_or_else(workspace_overflow)?);
        if peak > self.plan.limits.max_workspace_bytes {
            return Err(PvTuningError::step(format!(
                "native consumer workspace requires {peak} bytes, limit is {}",
                self.plan.limits.max_workspace_bytes
            )));
        }
        for (owner, uses) in self.owners.iter().zip(contributions) {
            for contribution in *uses {
                contribution.validate(
                    owner.weight.layout().rows as usize,
                    owner.weight.layout().cols as usize,
                )?;
            }
        }
        let mut row = zeroed(max_cols)?;
        let mut accumulated = reserved(max_scales)?;
        accumulated.resize(max_scales, 0.0f64);
        let mut gradients = reserved(self.owners.len())?;
        for (owner, uses) in self.owners.iter().zip(contributions) {
            let view = owner.weight.view();
            let out = &mut accumulated[..view.scales().len()];
            out.fill(0.0);
            let scratch = &mut row[..view.layout().cols as usize];
            for contribution in *uses {
                match contribution {
                    NativeScaleContribution::Matmul {
                        activations,
                        output_gradients,
                        batch,
                    } => accumulate_ternary_matmul_scale_vjp(
                        activations,
                        &view.bind_matmul(),
                        output_gradients,
                        *batch,
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
                })?;
            }
            let mut values = reserved(out.len())?;
            for &value in out.iter() {
                let value = value as f32;
                if !value.is_finite() {
                    return Err(PvTuningError::step(
                        "native accumulated scale gradient overflowed f32",
                    ));
                }
                values.push(value);
            }
            gradients.push(values);
        }
        drop(row);
        drop(accumulated);
        let mut borrowed = reserved(gradients.len())?;
        borrowed.extend(gradients.iter().map(Vec::as_slice));
        self.step(&borrowed)
    }
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
