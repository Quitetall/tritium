use crate::optim::Optimizer;

use super::projection::project_units;
use super::selection::selected_units;
use super::{PvStepReceipt, PvTernaryWeight, PvTuningConfig, PvTuningError, PvTuningSession};

impl PvTuningSession {
    /// Start from a validated deployed parent. No dense master is retained.
    ///
    /// # Errors
    /// Returns a validation error if parent invariants no longer hold.
    pub fn new(parent: PvTernaryWeight, config: PvTuningConfig) -> Result<Self, PvTuningError> {
        parent.validate()?;
        let parent_digest = parent.digest();
        let scale_state = config
            .continuous_optimizer
            .init_state(parent.total_scale_count());
        let code_state = config.code_optimizer.init_state(parent.len());
        Ok(Self {
            parent_digest,
            config,
            weight: parent,
            scale_state,
            code_state,
            completed_step: 0,
            blockwise: None,
        })
    }

    /// Current exact deployed representation.
    #[must_use]
    pub const fn weight(&self) -> &PvTernaryWeight {
        &self.weight
    }

    /// Number of successfully committed alternating steps.
    #[must_use]
    pub const fn completed_step(&self) -> u64 {
        self.completed_step
    }

    /// Apply one transactional P/V step at exact next one-based index.
    ///
    /// P differentiates actual group scales. V uses dequantized-weight gradient to
    /// form Adam proposal, selects largest bounded proposal units, then projects them
    /// onto deployed code space. This reference clones deployed representation and
    /// optimizer state before mutation; device/block adapters use scoped journals at
    /// model scale.
    ///
    /// # Errors
    /// Invalid input or update fails without mutating representation or optimizer state.
    pub fn step(
        &mut self,
        gradient: &[f32],
        optimizer_step: u64,
    ) -> Result<PvStepReceipt, PvTuningError> {
        let mut next = self.clone();
        let receipt = next.step_inner(gradient, optimizer_step)?;
        *self = next;
        Ok(receipt)
    }

    fn step_inner(
        &mut self,
        gradient: &[f32],
        optimizer_step: u64,
    ) -> Result<PvStepReceipt, PvTuningError> {
        if self.blockwise.is_some() {
            return Err(PvTuningError::step(
                "cannot run a whole-gradient step while a blockwise step is active",
            ));
        }
        let expected_step = self
            .completed_step
            .checked_add(1)
            .ok_or_else(|| PvTuningError::step("step counter overflow"))?;
        if optimizer_step != expected_step {
            return Err(PvTuningError::step(format!(
                "expected optimizer step {expected_step}, got {optimizer_step}"
            )));
        }
        if gradient.len() != self.weight.len() {
            return Err(PvTuningError::step("gradient length mismatch"));
        }
        if gradient.iter().any(|value| !value.is_finite()) {
            return Err(PvTuningError::step("gradient contains a non-finite value"));
        }

        let changed_scales = self.p_step(gradient, optimizer_step)?;
        let decoded = self.weight.decode();
        let mut proposal = decoded.clone();
        self.config.code_optimizer.step(
            optimizer_step,
            &mut proposal,
            gradient,
            &mut self.code_state,
        );
        if proposal.iter().any(|value| !value.is_finite()) {
            return Err(PvTuningError::step(
                "code optimizer produced a non-finite proposal",
            ));
        }

        let units = selected_units(
            self.weight.structure,
            &decoded,
            &proposal,
            self.config.max_code_change_fraction,
        );
        let (projection, v_surrogate_before, v_surrogate_after) = project_units(
            &mut self.weight,
            self.config.max_relative_code_change,
            &units,
            |index, _decoded| proposal[index],
        );
        self.weight.validate()?;
        if v_surrogate_after > v_surrogate_before {
            return Err(PvTuningError::step(
                "V projection increased its discrete surrogate",
            ));
        }
        self.completed_step = optimizer_step;
        Ok(PvStepReceipt {
            optimizer_step,
            selected_code_units: units.len(),
            changed_code_units: projection.changed_units,
            trust_limited_code_units: projection.trust_limited_units,
            changed_scales,
            v_surrogate_before,
            v_surrogate_after,
            relative_code_change: projection.relative_change,
            representation_digest: self.weight.digest(),
        })
    }
}
