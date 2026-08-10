use crate::optim::AdamW;

use super::PvTuningError;

/// Frozen optimizer and bounded-subspace recipe for one PV session.
#[derive(Clone, Copy, Debug)]
pub struct PvTuningConfig {
    pub(super) continuous_optimizer: AdamW,
    pub(super) code_optimizer: AdamW,
    pub(super) max_code_change_fraction: f32,
    pub(super) max_relative_code_change: Option<f32>,
}

impl PvTuningConfig {
    /// Start a named recipe builder with distinct continuous and code optimizers.
    #[must_use]
    pub const fn builder(
        continuous_optimizer: AdamW,
        code_optimizer: AdamW,
    ) -> PvTuningConfigBuilder {
        PvTuningConfigBuilder {
            continuous_optimizer,
            code_optimizer,
            max_code_change_fraction: None,
            max_relative_code_change: None,
        }
    }

    fn validated(
        continuous_optimizer: AdamW,
        code_optimizer: AdamW,
        max_code_change_fraction: f32,
        max_relative_code_change: Option<f32>,
    ) -> Result<Self, PvTuningError> {
        validate_adam("continuous optimizer", continuous_optimizer)?;
        validate_adam("code optimizer", code_optimizer)?;
        if !max_code_change_fraction.is_finite()
            || !(0.0..=1.0).contains(&max_code_change_fraction)
            || max_code_change_fraction == 0.0
        {
            return Err(PvTuningError::invalid_config(
                "max_code_change_fraction must be finite and in (0, 1]",
            ));
        }
        if matches!(max_relative_code_change, Some(value) if !value.is_finite() || value <= 0.0) {
            return Err(PvTuningError::invalid_config(
                "max_relative_code_change must be finite and positive",
            ));
        }
        Ok(Self {
            continuous_optimizer,
            code_optimizer,
            max_code_change_fraction,
            max_relative_code_change,
        })
    }

    /// Stable recipe identity used by model-level plans and checkpoints.
    #[must_use]
    pub fn recipe_digest(self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"tritium.pv-tuning-plan.v1\0");
        hash_adam(&mut hasher, self.continuous_optimizer);
        hash_adam(&mut hasher, self.code_optimizer);
        hasher.update(&self.max_code_change_fraction.to_bits().to_le_bytes());
        match self.max_relative_code_change {
            Some(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_bits().to_le_bytes());
            }
            None => {
                hasher.update(&[0]);
            }
        }
        *hasher.finalize().as_bytes()
    }

    pub(super) fn digest(self) -> [u8; 32] {
        self.recipe_digest()
    }
}

/// Named construction path for a frozen [`PvTuningConfig`].
#[derive(Clone, Copy, Debug)]
pub struct PvTuningConfigBuilder {
    continuous_optimizer: AdamW,
    code_optimizer: AdamW,
    max_code_change_fraction: Option<f32>,
    max_relative_code_change: Option<f32>,
}

impl PvTuningConfigBuilder {
    /// Set fraction of scalar/S34 units considered each V step.
    #[must_use]
    pub const fn max_code_change_fraction(mut self, fraction: f32) -> Self {
        self.max_code_change_fraction = Some(fraction);
        self
    }

    /// Set cumulative L2 code-movement trust ratio.
    #[must_use]
    pub const fn max_relative_code_change(mut self, ratio: f32) -> Self {
        self.max_relative_code_change = Some(ratio);
        self
    }

    /// Validate and freeze recipe identity.
    ///
    /// # Errors
    /// Missing fraction or invalid optimizer/bound values return
    /// [`PvTuningError::InvalidConfig`].
    pub fn build(self) -> Result<PvTuningConfig, PvTuningError> {
        let fraction = self.max_code_change_fraction.ok_or_else(|| {
            PvTuningError::invalid_config("max_code_change_fraction must be specified")
        })?;
        PvTuningConfig::validated(
            self.continuous_optimizer,
            self.code_optimizer,
            fraction,
            self.max_relative_code_change,
        )
    }
}

fn validate_adam(label: &str, optimizer: AdamW) -> Result<(), PvTuningError> {
    if !optimizer.lr.is_finite() || optimizer.lr <= 0.0 {
        return Err(PvTuningError::invalid_config(format!(
            "{label} learning rate must be finite and positive"
        )));
    }
    if !optimizer.beta1.is_finite()
        || !optimizer.beta2.is_finite()
        || !(0.0..1.0).contains(&optimizer.beta1)
        || !(0.0..1.0).contains(&optimizer.beta2)
        || (optimizer.beta1 == 0.0 && optimizer.beta1.is_sign_negative())
        || (optimizer.beta2 == 0.0 && optimizer.beta2.is_sign_negative())
    {
        return Err(PvTuningError::invalid_config(format!(
            "{label} betas must be finite and in [0, 1)"
        )));
    }
    if !optimizer.eps.is_finite() || optimizer.eps <= 0.0 {
        return Err(PvTuningError::invalid_config(format!(
            "{label} epsilon must be finite and positive"
        )));
    }
    if !optimizer.weight_decay.is_finite()
        || optimizer.weight_decay < 0.0
        || (optimizer.weight_decay == 0.0 && optimizer.weight_decay.is_sign_negative())
        || optimizer.lr * optimizer.weight_decay > 1.0
    {
        return Err(PvTuningError::invalid_config(format!(
            "{label} weight decay must be finite, nonnegative, and non-sign-flipping"
        )));
    }
    Ok(())
}

fn hash_adam(hasher: &mut blake3::Hasher, optimizer: AdamW) {
    for value in [
        optimizer.lr,
        optimizer.beta1,
        optimizer.beta2,
        optimizer.eps,
        optimizer.weight_decay,
    ] {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}
