//! Deterministic recovery policy for SALT V2 additive ternarization.
//!
//! This module decides *when* recovery objectives and gates are active. It does
//! not implement gradients, teacher-cache storage, or packed inference. Keeping
//! those mechanics outside the policy makes the training receipt replayable and
//! lets callers reject invalid schedules before allocating accelerator time.

/// The two fixed SALT V2 recovery phases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryPhase {
    /// Differentiable recovery over the first 80% of optimizer steps.
    Soft,
    /// Hard-discrete recovery over the final 20% of optimizer steps.
    Hard,
}

/// A validated, exact 80/20 recovery schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecoverySchedule {
    total_steps: u64,
    hard_start_step: u64,
}

impl RecoverySchedule {
    /// Build a schedule whose step count admits an exact 80/20 split.
    ///
    /// # Errors
    /// Returns [`RecoveryError::InvalidSchedule`] when `total_steps` is zero or
    /// is not divisible by five. Rejecting rather than rounding keeps receipts
    /// replayable across implementations.
    pub fn new(total_steps: u64) -> Result<Self, RecoveryError> {
        if total_steps == 0 || !total_steps.is_multiple_of(5) {
            return Err(RecoveryError::InvalidSchedule(
                "total_steps must be positive and divisible by five",
            ));
        }
        Ok(Self {
            total_steps,
            hard_start_step: total_steps / 5 * 4,
        })
    }

    /// Total number of zero-based steps in the schedule.
    #[must_use]
    pub const fn total_steps(self) -> u64 {
        self.total_steps
    }

    /// First zero-based step in the hard tail.
    #[must_use]
    pub const fn hard_start_step(self) -> u64 {
        self.hard_start_step
    }

    /// Phase for a zero-based optimizer step.
    ///
    /// # Errors
    /// Returns [`RecoveryError::StepOutOfRange`] outside this schedule.
    pub const fn phase_at(self, step: u64) -> Result<RecoveryPhase, RecoveryError> {
        if step >= self.total_steps {
            return Err(RecoveryError::StepOutOfRange {
                step,
                total_steps: self.total_steps,
            });
        }
        if step < self.hard_start_step {
            Ok(RecoveryPhase::Soft)
        } else {
            Ok(RecoveryPhase::Hard)
        }
    }
}

/// Validation-window definition for conditional cached-logit distillation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlateauConfig {
    earliest_step: u64,
    validation_window: usize,
    maximum_interval_improvement: f64,
}

impl PlateauConfig {
    /// Define a plateau as a validation window where every adjacent held-out
    /// NLL improvement is no larger than `maximum_interval_improvement`.
    ///
    /// Regressions count as no improvement. This intentionally enables the
    /// stronger cached-logit target when CE-only recovery has stalled *or*
    /// worsened, rather than silently spending the rest of the campaign on CE.
    ///
    /// # Errors
    /// Rejects windows shorter than two or non-finite/negative thresholds.
    pub fn new(
        earliest_step: u64,
        validation_window: usize,
        maximum_interval_improvement: f64,
    ) -> Result<Self, RecoveryError> {
        if validation_window < 2 {
            return Err(RecoveryError::InvalidConfiguration(
                "plateau validation_window must be at least two",
            ));
        }
        if !maximum_interval_improvement.is_finite() || maximum_interval_improvement < 0.0 {
            return Err(RecoveryError::InvalidConfiguration(
                "plateau improvement threshold must be finite and non-negative",
            ));
        }
        Ok(Self {
            earliest_step,
            validation_window,
            maximum_interval_improvement,
        })
    }
}

/// Optional hidden-state cosine recovery target on explicitly selected layers.
#[derive(Clone, Debug, PartialEq)]
pub struct HiddenCosineTerm {
    layers: Vec<u32>,
    weight: f64,
}

impl HiddenCosineTerm {
    /// Build a selected-layer hidden-cosine term.
    ///
    /// # Errors
    /// Rejects an empty layer set, duplicate layers, or a non-positive/non-finite
    /// weight. Layer order remains caller-defined and is copied into receipts.
    pub fn new(layers: Vec<u32>, weight: f64) -> Result<Self, RecoveryError> {
        if layers.is_empty() {
            return Err(RecoveryError::InvalidConfiguration(
                "hidden-cosine layers cannot be empty",
            ));
        }
        if !weight.is_finite() || weight <= 0.0 {
            return Err(RecoveryError::InvalidConfiguration(
                "hidden-cosine weight must be finite and positive",
            ));
        }
        if layers
            .iter()
            .enumerate()
            .any(|(index, layer)| layers[..index].contains(layer))
        {
            return Err(RecoveryError::InvalidConfiguration(
                "hidden-cosine layers must be unique",
            ));
        }
        Ok(Self { layers, weight })
    }

    /// Selected transformer layer indices.
    #[must_use]
    pub fn layers(&self) -> &[u32] {
        &self.layers
    }

    /// Loss coefficient applied to the selected hidden-cosine term.
    #[must_use]
    pub const fn weight(&self) -> f64 {
        self.weight
    }
}

/// Monotone linear allowance for temporary full-precision bypass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BypassSchedule {
    total_steps: u64,
    initial_fraction: f64,
    zero_at_step: u64,
}

impl BypassSchedule {
    /// Linearly decay the maximum bypass fraction to exactly zero.
    ///
    /// # Errors
    /// Rejects a fraction outside `[0, 1]`, a non-finite fraction, or a zero
    /// point outside the zero-based recovery schedule.
    pub fn linear(
        schedule: RecoverySchedule,
        initial_fraction: f64,
        zero_at_step: u64,
    ) -> Result<Self, RecoveryError> {
        if !initial_fraction.is_finite() || !(0.0..=1.0).contains(&initial_fraction) {
            return Err(RecoveryError::InvalidConfiguration(
                "initial bypass fraction must be finite and in [0, 1]",
            ));
        }
        if zero_at_step == 0 || zero_at_step >= schedule.total_steps() {
            return Err(RecoveryError::InvalidConfiguration(
                "bypass zero_at_step must be inside the recovery schedule and after step zero",
            ));
        }
        Ok(Self {
            total_steps: schedule.total_steps(),
            initial_fraction,
            zero_at_step,
        })
    }

    /// Maximum allowed bypass fraction for this step.
    ///
    /// # Errors
    /// Returns [`RecoveryError::StepOutOfRange`] outside this schedule.
    pub fn allowance_at(self, step: u64) -> Result<f64, RecoveryError> {
        if step >= self.total_steps {
            return Err(RecoveryError::StepOutOfRange {
                step,
                total_steps: self.total_steps,
            });
        }
        if step >= self.zero_at_step {
            return Ok(0.0);
        }
        let remaining = self.zero_at_step - step;
        Ok(self.initial_fraction * remaining as f64 / self.zero_at_step as f64)
    }
}

/// Active token-level recovery objective.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LossMode {
    /// Ground-truth cross entropy only.
    CrossEntropyOnly,
    /// Cross entropy plus KL against a precomputed teacher-logit cache.
    CrossEntropyWithCachedLogitKd {
        /// KL coefficient; CE retains coefficient `1 - kd_weight`.
        kd_weight: f64,
    },
}

/// Complete policy directive for one optimizer step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveryDirective {
    phase: RecoveryPhase,
    loss_mode: LossMode,
    hidden_cosine_weight: Option<f64>,
    pv_polish: bool,
    maximum_bypass_fraction: f64,
}

impl RecoveryDirective {
    /// Current soft/hard phase.
    #[must_use]
    pub const fn phase(self) -> RecoveryPhase {
        self.phase
    }

    /// Active CE/KD objective.
    #[must_use]
    pub const fn loss_mode(self) -> LossMode {
        self.loss_mode
    }

    /// Optional selected-layer hidden-cosine coefficient.
    #[must_use]
    pub const fn hidden_cosine_weight(self) -> Option<f64> {
        self.hidden_cosine_weight
    }

    /// Whether PV hard-discrete polishing is enabled.
    #[must_use]
    pub const fn pv_polish(self) -> bool {
        self.pv_polish
    }

    /// Maximum temporary full-precision bypass fraction.
    #[must_use]
    pub const fn maximum_bypass_fraction(self) -> f64 {
        self.maximum_bypass_fraction
    }
}

/// Stateful recovery-policy evaluator.
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveryPolicy {
    schedule: RecoverySchedule,
    plateau: PlateauConfig,
    cached_logit_kd_weight: f64,
    hidden_cosine: Option<HiddenCosineTerm>,
    bypass: BypassSchedule,
    recent_validations: Vec<(u64, f64)>,
    last_validation_step: Option<u64>,
    kd_active: bool,
}

impl RecoveryPolicy {
    /// Validate and create a CE-first recovery policy.
    ///
    /// # Errors
    /// Rejects cross-schedule bypasses, an out-of-range plateau start, or a KD
    /// coefficient outside `(0, 1]`.
    pub fn new(
        schedule: RecoverySchedule,
        plateau: PlateauConfig,
        cached_logit_kd_weight: f64,
        hidden_cosine: Option<HiddenCosineTerm>,
        bypass: BypassSchedule,
    ) -> Result<Self, RecoveryError> {
        if plateau.earliest_step >= schedule.total_steps() {
            return Err(RecoveryError::InvalidConfiguration(
                "plateau earliest_step must be inside the recovery schedule",
            ));
        }
        if !(cached_logit_kd_weight.is_finite()
            && 0.0 < cached_logit_kd_weight
            && cached_logit_kd_weight <= 1.0)
        {
            return Err(RecoveryError::InvalidConfiguration(
                "cached-logit KD weight must be finite and in (0, 1]",
            ));
        }
        if bypass.total_steps != schedule.total_steps() {
            return Err(RecoveryError::InvalidConfiguration(
                "bypass and phase schedules must have the same total_steps",
            ));
        }
        let validation_window = u64::try_from(plateau.validation_window).map_err(|_| {
            RecoveryError::InvalidConfiguration(
                "plateau validation_window must fit the recovery schedule domain",
            )
        })?;
        if validation_window > schedule.total_steps() {
            return Err(RecoveryError::InvalidConfiguration(
                "plateau validation_window cannot exceed total recovery steps",
            ));
        }
        let mut recent_validations = Vec::new();
        recent_validations
            .try_reserve_exact(plateau.validation_window)
            .map_err(|_| RecoveryError::AllocationFailed("plateau validation window"))?;
        Ok(Self {
            schedule,
            plateau,
            cached_logit_kd_weight,
            hidden_cosine,
            bypass,
            recent_validations,
            last_validation_step: None,
            kd_active: false,
        })
    }

    /// Observe held-out NLL at a monotonically increasing validation step.
    ///
    /// Cached-logit KD becomes active permanently once the configured window is
    /// full and no adjacent interval beats the plateau threshold.
    ///
    /// # Errors
    /// Rejects non-finite NLL, out-of-range steps, or non-increasing validation
    /// step numbers.
    pub fn observe_held_out_nll(&mut self, step: u64, nll: f64) -> Result<(), RecoveryError> {
        self.schedule.phase_at(step)?;
        if !nll.is_finite() {
            return Err(RecoveryError::NonFiniteMetric("held-out NLL"));
        }
        if let Some(previous) = self.last_validation_step
            && step <= previous
        {
            return Err(RecoveryError::ObservationOrder {
                previous_step: previous,
                current_step: step,
            });
        }
        self.last_validation_step = Some(step);
        if step < self.plateau.earliest_step {
            return Ok(());
        }
        self.recent_validations.push((step, nll));
        if self.recent_validations.len() > self.plateau.validation_window {
            self.recent_validations.remove(0);
        }
        if self.recent_validations.len() == self.plateau.validation_window
            && self
                .recent_validations
                .windows(2)
                .all(|pair| pair[0].1 - pair[1].1 <= self.plateau.maximum_interval_improvement)
        {
            self.kd_active = true;
        }
        Ok(())
    }

    /// Return the validated policy directive for one zero-based step.
    ///
    /// # Errors
    /// Returns [`RecoveryError::StepOutOfRange`] outside the schedule.
    pub fn directive(&self, step: u64) -> Result<RecoveryDirective, RecoveryError> {
        let phase = self.schedule.phase_at(step)?;
        let maximum_bypass_fraction = if phase == RecoveryPhase::Hard {
            0.0
        } else {
            self.bypass.allowance_at(step)?
        };
        Ok(RecoveryDirective {
            phase,
            loss_mode: if self.kd_active {
                LossMode::CrossEntropyWithCachedLogitKd {
                    kd_weight: self.cached_logit_kd_weight,
                }
            } else {
                LossMode::CrossEntropyOnly
            },
            hidden_cosine_weight: self.hidden_cosine.as_ref().map(HiddenCosineTerm::weight),
            pv_polish: phase == RecoveryPhase::Hard,
            maximum_bypass_fraction,
        })
    }

    /// Selected hidden-cosine layers, if configured.
    #[must_use]
    pub fn hidden_cosine_layers(&self) -> Option<&[u32]> {
        self.hidden_cosine.as_ref().map(HiddenCosineTerm::layers)
    }
}

/// Accounting and measurements supplied after one optimizer step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepObservation {
    step: u64,
    tokens_processed: u64,
    gpu_seconds: f64,
    bypass_fraction: f64,
    held_out_nll: Option<f64>,
}

impl StepObservation {
    /// Describe one completed zero-based optimizer step.
    #[must_use]
    pub const fn new(
        step: u64,
        tokens_processed: u64,
        gpu_seconds: f64,
        bypass_fraction: f64,
        held_out_nll: Option<f64>,
    ) -> Self {
        Self {
            step,
            tokens_processed,
            gpu_seconds,
            bypass_fraction,
            held_out_nll,
        }
    }
}

/// Audit marker returned for each accepted bypass observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BypassUsageFlag {
    /// This optimizer step did not use the temporary full-precision path.
    Unused,
    /// This optimizer step used a non-zero temporary full-precision path.
    Used,
}

/// Final paired measurements needed to close a recovery receipt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FinalRecoveryMetrics {
    soft_nll: f64,
    hard_nll: f64,
    code_churn: f64,
    packed_minus_training_nll: f64,
}

impl FinalRecoveryMetrics {
    /// Validate final soft, hard, code-stability, and pack/reload measurements.
    ///
    /// `code_churn` is the fraction of codes changed by hard-tail polishing.
    /// `packed_minus_training_nll` is signed, so a negative value is retained.
    ///
    /// # Errors
    /// Rejects non-finite values, negative NLL, code churn outside `[0, 1]`,
    /// or a hard-minus-soft subtraction that overflows to infinity.
    pub fn new(
        soft_nll: f64,
        hard_nll: f64,
        code_churn: f64,
        packed_minus_training_nll: f64,
    ) -> Result<Self, RecoveryError> {
        if !soft_nll.is_finite() || soft_nll < 0.0 {
            return Err(RecoveryError::InvalidMetric("soft NLL"));
        }
        if !hard_nll.is_finite() || hard_nll < 0.0 {
            return Err(RecoveryError::InvalidMetric("hard NLL"));
        }
        if !code_churn.is_finite() || !(0.0..=1.0).contains(&code_churn) {
            return Err(RecoveryError::InvalidMetric("code churn"));
        }
        if !packed_minus_training_nll.is_finite() {
            return Err(RecoveryError::InvalidMetric("packed-minus-training NLL"));
        }
        if !(hard_nll - soft_nll).is_finite() {
            return Err(RecoveryError::InvalidMetric("hard-minus-soft NLL"));
        }
        Ok(Self {
            soft_nll,
            hard_nll,
            code_churn,
            packed_minus_training_nll,
        })
    }
}

/// Immutable closeout record for one complete recovery run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveryReceipt {
    tokens_processed: u64,
    gpu_seconds: f64,
    soft_nll: f64,
    hard_nll: f64,
    hard_minus_soft_nll: f64,
    code_churn: f64,
    packed_minus_training_nll: f64,
    scheduled_steps: u64,
    bypass_steps_used: u64,
    exported_with_zero_bypass: bool,
}

impl RecoveryReceipt {
    /// Total examples' tokens processed by the recovery run.
    #[must_use]
    pub const fn tokens_processed(self) -> u64 {
        self.tokens_processed
    }

    /// Accelerator-seconds summed across the run.
    #[must_use]
    pub const fn gpu_seconds(self) -> f64 {
        self.gpu_seconds
    }

    /// Held-out NLL under the differentiable training path.
    #[must_use]
    pub const fn soft_nll(self) -> f64 {
        self.soft_nll
    }

    /// Held-out NLL after hard-discrete PV polishing.
    #[must_use]
    pub const fn hard_nll(self) -> f64 {
        self.hard_nll
    }

    /// Signed `hard_nll - soft_nll` gap.
    #[must_use]
    pub const fn hard_minus_soft_nll(self) -> f64 {
        self.hard_minus_soft_nll
    }

    /// Fraction of ternary codes changed by hard-tail polishing.
    #[must_use]
    pub const fn code_churn(self) -> f64 {
        self.code_churn
    }

    /// Signed NLL delta after packing, reload, and inference-path evaluation.
    #[must_use]
    pub const fn packed_minus_training_nll(self) -> f64 {
        self.packed_minus_training_nll
    }

    /// Number of scheduled steps with any non-zero full-precision bypass.
    #[must_use]
    pub const fn bypass_steps_used(self) -> u64 {
        self.bypass_steps_used
    }

    /// Whether bypass was used during more than 1% of scheduled steps.
    #[must_use]
    pub const fn bypass_usage_flagged(self) -> bool {
        self.bypass_steps_used > self.scheduled_steps / 100
    }

    /// Whether the export gate observed exactly zero final bypass.
    #[must_use]
    pub const fn exported_with_zero_bypass(self) -> bool {
        self.exported_with_zero_bypass
    }
}

/// Stateful step accounting around a [`RecoveryPolicy`].
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveryRun {
    policy: RecoveryPolicy,
    completed_steps: u64,
    tokens_processed: u64,
    gpu_seconds: f64,
    previous_bypass_fraction: Option<f64>,
    bypass_steps_used: u64,
}

impl RecoveryRun {
    /// Start a zero-accounted recovery run.
    #[must_use]
    pub const fn new(policy: RecoveryPolicy) -> Self {
        Self {
            policy,
            completed_steps: 0,
            tokens_processed: 0,
            gpu_seconds: 0.0,
            previous_bypass_fraction: None,
            bypass_steps_used: 0,
        }
    }

    /// Borrow the live objective policy.
    #[must_use]
    pub const fn policy(&self) -> &RecoveryPolicy {
        &self.policy
    }

    /// Return the directive that must be used for the next step.
    ///
    /// # Errors
    /// Returns [`RecoveryError::StepOutOfRange`] after the run is complete.
    pub fn next_directive(&self) -> Result<RecoveryDirective, RecoveryError> {
        self.policy.directive(self.completed_steps)
    }

    /// Validate and record one complete optimizer step.
    ///
    /// Steps must be exhaustive and ordered. Actual bypass must remain below the
    /// planned allowance and must never increase, including across omitted-zero
    /// intervals (which cannot be omitted because step indices are exhaustive).
    ///
    /// # Errors
    /// Fails closed on invalid metrics, missing/reordered steps, counter overflow,
    /// bypass growth, or bypass above the planned allowance.
    pub fn record_step(
        &mut self,
        observation: StepObservation,
    ) -> Result<BypassUsageFlag, RecoveryError> {
        if observation.step != self.completed_steps {
            return Err(RecoveryError::StepOrder {
                expected_step: self.completed_steps,
                actual_step: observation.step,
            });
        }
        if observation.tokens_processed == 0 {
            return Err(RecoveryError::InvalidMetric("tokens processed per step"));
        }
        if !observation.gpu_seconds.is_finite() || observation.gpu_seconds <= 0.0 {
            return Err(RecoveryError::InvalidMetric("GPU seconds per step"));
        }
        if !observation.bypass_fraction.is_finite()
            || !(0.0..=1.0).contains(&observation.bypass_fraction)
        {
            return Err(RecoveryError::InvalidMetric("bypass fraction"));
        }
        if observation
            .held_out_nll
            .is_some_and(|nll| !nll.is_finite() || nll < 0.0)
        {
            return Err(RecoveryError::InvalidMetric("held-out NLL"));
        }

        let allowance = self
            .policy
            .directive(observation.step)?
            .maximum_bypass_fraction();
        if observation.bypass_fraction > allowance {
            return Err(RecoveryError::BypassAllowanceExceeded {
                step: observation.step,
            });
        }
        if self
            .previous_bypass_fraction
            .is_some_and(|previous| observation.bypass_fraction > previous)
        {
            return Err(RecoveryError::BypassIncreased {
                step: observation.step,
            });
        }

        let tokens_processed = self
            .tokens_processed
            .checked_add(observation.tokens_processed)
            .ok_or(RecoveryError::ArithmeticOverflow("tokens processed"))?;
        let gpu_seconds = self.gpu_seconds + observation.gpu_seconds;
        if !gpu_seconds.is_finite() {
            return Err(RecoveryError::ArithmeticOverflow("GPU seconds"));
        }
        let used = observation.bypass_fraction > 0.0;
        let bypass_steps_used = if used {
            self.bypass_steps_used
                .checked_add(1)
                .ok_or(RecoveryError::ArithmeticOverflow("bypass-use steps"))?
        } else {
            self.bypass_steps_used
        };

        if let Some(nll) = observation.held_out_nll {
            self.policy.observe_held_out_nll(observation.step, nll)?;
        }
        self.tokens_processed = tokens_processed;
        self.gpu_seconds = gpu_seconds;
        self.previous_bypass_fraction = Some(observation.bypass_fraction);
        self.bypass_steps_used = bypass_steps_used;
        self.completed_steps += 1;

        Ok(if used {
            BypassUsageFlag::Used
        } else {
            BypassUsageFlag::Unused
        })
    }

    /// Close a fully observed run into an immutable receipt.
    ///
    /// # Errors
    /// Refuses incomplete runs and any run whose final actual bypass is not
    /// exactly zero.
    pub fn export_receipt(
        &self,
        metrics: FinalRecoveryMetrics,
    ) -> Result<RecoveryReceipt, RecoveryError> {
        let total_steps = self.policy.schedule.total_steps();
        if self.completed_steps != total_steps {
            return Err(RecoveryError::IncompleteRun {
                completed_steps: self.completed_steps,
                total_steps,
            });
        }
        if self.previous_bypass_fraction != Some(0.0) {
            return Err(RecoveryError::NonZeroBypassAtExport);
        }
        Ok(RecoveryReceipt {
            tokens_processed: self.tokens_processed,
            gpu_seconds: self.gpu_seconds,
            soft_nll: metrics.soft_nll,
            hard_nll: metrics.hard_nll,
            hard_minus_soft_nll: metrics.hard_nll - metrics.soft_nll,
            code_churn: metrics.code_churn,
            packed_minus_training_nll: metrics.packed_minus_training_nll,
            scheduled_steps: total_steps,
            bypass_steps_used: self.bypass_steps_used,
            exported_with_zero_bypass: true,
        })
    }
}

/// Model scale with a frozen SALT V2 short-refinement token budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryModelRung {
    /// Model-ladder rungs 1 and 2 (SmolLM2-135M and SmolLM2-1.7B).
    Pilot,
    /// The frozen Qwen3-8B proof rung.
    Qwen8B,
    /// The frozen Qwen3-32B confirmation rung.
    Qwen32B,
}

impl RecoveryModelRung {
    /// Exact Stage 5 token cap for a recovery track.
    ///
    /// PTQ is conversion rather than short refinement and therefore has no
    /// refinement-token cap in this policy.
    #[must_use]
    pub const fn token_cap(self, track: RecoveryTrack) -> Option<u64> {
        match (self, track) {
            (_, RecoveryTrack::Ptq) => None,
            (Self::Pilot, RecoveryTrack::ScaleOnly) => Some(8_000_000),
            (Self::Pilot, RecoveryTrack::Pv) => Some(32_000_000),
            (Self::Qwen8B, RecoveryTrack::ScaleOnly) => Some(32_000_000),
            (Self::Qwen8B, RecoveryTrack::Pv) => Some(256_000_000),
            (Self::Qwen32B, RecoveryTrack::ScaleOnly) => Some(64_000_000),
            (Self::Qwen32B, RecoveryTrack::Pv) => Some(512_000_000),
        }
    }
}

/// Separately labeled SALT V2 conversion and refinement tracks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryTrack {
    /// Weight-only post-training quantization.
    Ptq,
    /// Teacher-KL scale refinement with ternary codes frozen.
    ScaleOnly,
    /// Short PV hard-trit/scale alternating refinement.
    Pv,
}

/// Frozen Stage 5 learning-curve evaluation fraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryEvaluationCheckpoint {
    /// One eighth of the track's token cap.
    OneEighth,
    /// One quarter of the track's token cap.
    OneQuarter,
    /// One half of the track's token cap.
    OneHalf,
    /// The full token cap.
    Full,
}

impl RecoveryEvaluationCheckpoint {
    /// Every checkpoint in required order.
    pub const ALL: [Self; 4] = [Self::OneEighth, Self::OneQuarter, Self::OneHalf, Self::Full];

    /// Exact cumulative token count for this checkpoint and cap.
    #[must_use]
    pub const fn tokens(self, token_cap: u64) -> u64 {
        match self {
            Self::OneEighth => token_cap / 8,
            Self::OneQuarter => token_cap / 4,
            Self::OneHalf => token_cap / 2,
            Self::Full => token_cap,
        }
    }
}

/// Activation-precision rung in the frozen A16 -> A8 -> A4 proof order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryActivationRung {
    /// Sixteen-bit activation proof; the only campaign root.
    A16,
    /// Eight-bit activation proof, authorized by a complete A16 chain.
    A8,
    /// Four-bit activation proof, authorized by a complete A8 chain.
    A4,
}

/// Non-zero content digest used to bind campaign and receipt evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecoveryEvidenceDigest([u8; 32]);

impl RecoveryEvidenceDigest {
    /// Validate a content digest.
    ///
    /// # Errors
    /// Rejects the all-zero sentinel, which cannot identify durable evidence.
    pub fn new(bytes: [u8; 32]) -> Result<Self, RecoveryError> {
        if bytes == [0; 32] {
            return Err(RecoveryError::InvalidEvidenceDigest);
        }
        Ok(Self(bytes))
    }

    /// Digest bytes for serialization or content-addressed reopen checks.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Exact source-model identity for frozen recovery authorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoverySourceModel {
    /// SmolLM2-135M smoke rung.
    SmolLm2OneThirtyFiveMillion,
    /// SmolLM2-1.7B pilot-selection rung.
    SmolLm2OnePointSevenBillion,
    /// Frozen Qwen3-8B proof rung.
    Qwen3EightBillion,
    /// Frozen Qwen3-32B confirmation rung.
    Qwen3ThirtyTwoBillion,
}

impl RecoverySourceModel {
    /// Token-budget rung for this exact source model.
    #[must_use]
    pub const fn model_rung(self) -> RecoveryModelRung {
        match self {
            Self::SmolLm2OneThirtyFiveMillion | Self::SmolLm2OnePointSevenBillion => {
                RecoveryModelRung::Pilot
            }
            Self::Qwen3EightBillion => RecoveryModelRung::Qwen8B,
            Self::Qwen3ThirtyTwoBillion => RecoveryModelRung::Qwen32B,
        }
    }
}

/// Nonzero semantic digest of exact source checkpoint/config/tokenizer identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecoverySourceModelId([u8; 32]);

impl RecoverySourceModelId {
    /// Validate exact source identity.
    ///
    /// # Errors
    /// Rejects the all-zero missing-provenance sentinel.
    pub fn new(bytes: [u8; 32]) -> Result<Self, RecoveryError> {
        if bytes == [0; 32] {
            return Err(RecoveryError::InvalidSourceModelId);
        }
        Ok(Self(bytes))
    }

    /// Exact identity bytes.
    #[must_use]
    pub const fn as_bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Preregistered quality gate required before dependent spend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveryPromotionGate {
    minimum_validation_quality: f64,
    gate_digest: RecoveryEvidenceDigest,
}

impl RecoveryPromotionGate {
    /// Construct a digest-bound larger-is-better quality gate.
    ///
    /// # Errors
    /// Rejects a non-finite threshold.
    pub fn new(
        minimum_validation_quality: f64,
        gate_digest: RecoveryEvidenceDigest,
    ) -> Result<Self, RecoveryError> {
        if !minimum_validation_quality.is_finite() {
            return Err(RecoveryError::NonFiniteMetric(
                "campaign promotion threshold",
            ));
        }
        Ok(Self {
            minimum_validation_quality: canonical_f64(minimum_validation_quality),
            gate_digest,
        })
    }

    /// Frozen minimum validation quality.
    #[must_use]
    pub const fn minimum_validation_quality(self) -> f64 {
        self.minimum_validation_quality
    }

    /// Digest of full preregistered gate definition.
    #[must_use]
    pub const fn gate_digest(self) -> RecoveryEvidenceDigest {
        self.gate_digest
    }
}

const RECOVERY_PROMOTION_EVIDENCE_MAGIC: [u8; 8] = *b"TRPEVD\0\0";
const RECOVERY_PROMOTION_EVIDENCE_VERSION: u16 = 1;
const RECOVERY_PROMOTION_EVIDENCE_CONTENT_BYTES: usize = 159;
const RECOVERY_PROMOTION_EVIDENCE_BYTES: usize = RECOVERY_PROMOTION_EVIDENCE_CONTENT_BYTES + 32;
const RECOVERY_PROMOTION_EVIDENCE_HASH_CONTEXT: &str = "tritium recovery promotion evidence v1";

/// Independently persisted evidence used to evaluate a promotion gate.
///
/// Instances are created from a terminal [`RecoveryCampaignRun`], not from a
/// caller-supplied digest. The durable encoding binds every campaign context
/// field needed to distinguish otherwise identical quality measurements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveryPromotionEvidence {
    source_model: RecoverySourceModel,
    source_model_id: RecoverySourceModelId,
    activation_rung: RecoveryActivationRung,
    track: RecoveryTrack,
    campaign_digest: RecoveryEvidenceDigest,
    promotion_gate: RecoveryPromotionGate,
    selected_checkpoint: RecoverySelectedCheckpoint,
    validation_quality: f64,
    artifact_digest: RecoveryEvidenceDigest,
    evidence_digest: RecoveryEvidenceDigest,
}

impl RecoveryPromotionEvidence {
    /// Serialize this evidence into its canonical, versioned durable encoding.
    ///
    /// # Errors
    /// Rejects inconsistent internal evidence or an allocation failure.
    pub fn to_bytes(self) -> Result<Vec<u8>, RecoveryError> {
        self.validate_digest()?;
        let mut bytes = self.canonical_content_bytes()?;
        bytes
            .try_reserve_exact(32)
            .map_err(|_| RecoveryError::AllocationFailed("promotion evidence encoding"))?;
        bytes.extend_from_slice(&self.evidence_digest.as_bytes());
        Ok(bytes)
    }

    /// Reopen and strictly verify canonical promotion-evidence bytes.
    ///
    /// # Errors
    /// Rejects truncation, trailing bytes, unknown versions, non-canonical
    /// encodings, invalid fields, or a content-digest mismatch.
    pub fn reopen(bytes: &[u8]) -> Result<Self, RecoveryError> {
        if bytes.len() < RECOVERY_PROMOTION_EVIDENCE_MAGIC.len() + core::mem::size_of::<u16>() {
            return Err(RecoveryError::InvalidPromotionEvidence("truncated header"));
        }
        if bytes[..RECOVERY_PROMOTION_EVIDENCE_MAGIC.len()] != RECOVERY_PROMOTION_EVIDENCE_MAGIC {
            return Err(RecoveryError::InvalidPromotionEvidence("invalid magic"));
        }
        let version = u16::from_le_bytes(
            bytes[RECOVERY_PROMOTION_EVIDENCE_MAGIC.len()
                ..RECOVERY_PROMOTION_EVIDENCE_MAGIC.len() + 2]
                .try_into()
                .expect("fixed version slice"),
        );
        if version != RECOVERY_PROMOTION_EVIDENCE_VERSION {
            return Err(RecoveryError::UnsupportedPromotionEvidenceVersion(version));
        }
        if bytes.len() < RECOVERY_PROMOTION_EVIDENCE_BYTES {
            return Err(RecoveryError::InvalidPromotionEvidence(
                "truncated evidence",
            ));
        }
        if bytes.len() > RECOVERY_PROMOTION_EVIDENCE_BYTES {
            return Err(RecoveryError::InvalidPromotionEvidence(
                "trailing evidence bytes",
            ));
        }

        let mut cursor = RecoveryPromotionEvidenceCursor::new(bytes);
        cursor.read_exact::<8>()?;
        cursor.read_exact::<2>()?;
        let source_model = decode_source_model(cursor.read_u8()?)
            .map_err(|_| RecoveryError::InvalidPromotionEvidence("invalid source-model tag"))?;
        let source_model_id = RecoverySourceModelId::new(cursor.read_exact::<32>()?)?;
        let activation_rung = decode_activation_rung(cursor.read_u8()?)
            .map_err(|_| RecoveryError::InvalidPromotionEvidence("invalid activation-rung tag"))?;
        let track = decode_track(cursor.read_u8()?)
            .map_err(|_| RecoveryError::InvalidPromotionEvidence("invalid track tag"))?;
        let campaign_digest = RecoveryEvidenceDigest::new(cursor.read_exact::<32>()?)?;
        let promotion_gate = RecoveryPromotionGate::new(
            cursor.read_f64()?,
            RecoveryEvidenceDigest::new(cursor.read_exact::<32>()?)?,
        )?;
        let selected_checkpoint = decode_selected_checkpoint(cursor.read_u8()?, cursor.read_u8()?)
            .map_err(|_| {
                RecoveryError::InvalidPromotionEvidence("invalid selected-checkpoint tag")
            })?;
        let validation_quality = cursor.read_f64()?;
        let artifact_digest = RecoveryEvidenceDigest::new(cursor.read_exact::<32>()?)?;
        let evidence_digest = RecoveryEvidenceDigest::new(cursor.read_exact::<32>()?)?;
        if !cursor.is_empty() {
            return Err(RecoveryError::InvalidPromotionEvidence(
                "trailing evidence bytes",
            ));
        }

        let evidence = Self {
            source_model,
            source_model_id,
            activation_rung,
            track,
            campaign_digest,
            promotion_gate,
            selected_checkpoint,
            validation_quality,
            artifact_digest,
            evidence_digest,
        };
        evidence.validate_digest()?;
        let canonical = evidence.canonical_content_bytes()?;
        if canonical.as_slice() != &bytes[..RECOVERY_PROMOTION_EVIDENCE_CONTENT_BYTES] {
            return Err(RecoveryError::InvalidPromotionEvidence(
                "non-canonical evidence encoding",
            ));
        }
        Ok(evidence)
    }

    fn from_run(
        run: &RecoveryCampaignRun,
        best: RecoveryCampaignBest,
    ) -> Result<Self, RecoveryError> {
        if !best.validation_quality.is_finite() {
            return Err(RecoveryError::NonFiniteMetric("campaign promotion quality"));
        }
        let mut evidence = Self {
            source_model: run.source_model,
            source_model_id: run.source_model_id,
            activation_rung: run.activation_rung,
            track: run.track,
            campaign_digest: run.campaign_digest,
            promotion_gate: run.promotion_gate,
            selected_checkpoint: best.checkpoint,
            validation_quality: canonical_f64(best.validation_quality),
            artifact_digest: best.artifact_digest,
            evidence_digest: RecoveryEvidenceDigest([0; 32]),
        };
        evidence.evidence_digest =
            promotion_evidence_content_digest(&evidence.canonical_content_bytes()?)?;
        Ok(evidence)
    }

    /// Content digest of gate-evaluation evidence.
    #[must_use]
    pub const fn evidence_digest(self) -> RecoveryEvidenceDigest {
        self.evidence_digest
    }

    fn canonical_content_bytes(self) -> Result<Vec<u8>, RecoveryError> {
        let mut out = Vec::new();
        out.try_reserve_exact(RECOVERY_PROMOTION_EVIDENCE_CONTENT_BYTES)
            .map_err(|_| RecoveryError::AllocationFailed("promotion evidence encoding"))?;
        out.extend_from_slice(&RECOVERY_PROMOTION_EVIDENCE_MAGIC);
        out.extend_from_slice(&RECOVERY_PROMOTION_EVIDENCE_VERSION.to_le_bytes());
        out.push(encode_source_model(self.source_model));
        out.extend_from_slice(&self.source_model_id.as_bytes());
        out.push(encode_activation_rung(self.activation_rung));
        out.push(encode_track(self.track));
        out.extend_from_slice(&self.campaign_digest.as_bytes());
        out.extend_from_slice(
            &canonical_f64_bits(self.promotion_gate.minimum_validation_quality).to_le_bytes(),
        );
        out.extend_from_slice(&self.promotion_gate.gate_digest.as_bytes());
        let (checkpoint, detail) = encode_selected_checkpoint(self.selected_checkpoint);
        out.push(checkpoint);
        out.push(detail);
        out.extend_from_slice(&canonical_f64_bits(self.validation_quality).to_le_bytes());
        out.extend_from_slice(&self.artifact_digest.as_bytes());
        debug_assert_eq!(out.len(), RECOVERY_PROMOTION_EVIDENCE_CONTENT_BYTES);
        Ok(out)
    }

    fn validate_digest(self) -> Result<(), RecoveryError> {
        if !self.promotion_gate.minimum_validation_quality.is_finite()
            || !self.validation_quality.is_finite()
        {
            return Err(RecoveryError::InvalidPromotionEvidence("non-finite metric"));
        }
        if self.promotion_gate.minimum_validation_quality.to_bits()
            != canonical_f64_bits(self.promotion_gate.minimum_validation_quality)
            || self.validation_quality.to_bits() != canonical_f64_bits(self.validation_quality)
        {
            return Err(RecoveryError::InvalidPromotionEvidence(
                "non-canonical metric",
            ));
        }
        let expected = promotion_evidence_content_digest(&self.canonical_content_bytes()?)?;
        if expected != self.evidence_digest {
            return Err(RecoveryError::CampaignPromotionEvidenceDigestMismatch);
        }
        Ok(())
    }

    fn same_semantics(self, other: Self) -> bool {
        self.source_model == other.source_model
            && self.source_model_id == other.source_model_id
            && self.activation_rung == other.activation_rung
            && self.track == other.track
            && self.campaign_digest == other.campaign_digest
            && self.promotion_gate.gate_digest == other.promotion_gate.gate_digest
            && canonical_f64_bits(self.promotion_gate.minimum_validation_quality)
                == canonical_f64_bits(other.promotion_gate.minimum_validation_quality)
            && self.selected_checkpoint == other.selected_checkpoint
            && canonical_f64_bits(self.validation_quality)
                == canonical_f64_bits(other.validation_quality)
            && self.artifact_digest == other.artifact_digest
    }
}

/// Explicit result of applying the preregistered promotion gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryPromotionOutcome {
    /// Selected checkpoint passed and may authorize dependent work.
    Passed,
    /// Run completed but selected checkpoint did not pass.
    Failed,
}

/// Complete immutable authorization request for one recovery track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveryCampaignPlan {
    source_model: RecoverySourceModel,
    source_model_id: RecoverySourceModelId,
    activation_rung: RecoveryActivationRung,
    track: RecoveryTrack,
    campaign_digest: RecoveryEvidenceDigest,
    promotion_gate: RecoveryPromotionGate,
}

impl RecoveryCampaignPlan {
    /// Construct a source-bound frozen campaign track.
    #[must_use]
    pub const fn new(
        source_model: RecoverySourceModel,
        source_model_id: RecoverySourceModelId,
        activation_rung: RecoveryActivationRung,
        track: RecoveryTrack,
        campaign_digest: RecoveryEvidenceDigest,
        promotion_gate: RecoveryPromotionGate,
    ) -> Self {
        Self {
            source_model,
            source_model_id,
            activation_rung,
            track,
            campaign_digest,
            promotion_gate,
        }
    }

    /// Exact source checkpoint family.
    #[must_use]
    pub const fn source_model(self) -> RecoverySourceModel {
        self.source_model
    }

    /// Exact semantic source identity.
    #[must_use]
    pub const fn source_model_id(self) -> RecoverySourceModelId {
        self.source_model_id
    }
}

/// Best checkpoint retained by a frozen recovery-track receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoverySelectedCheckpoint {
    /// The completed PTQ conversion itself.
    Ptq,
    /// The predecessor remains better than every evaluated refinement point.
    Predecessor,
    /// A refinement evaluation replaced the predecessor.
    Evaluation(RecoveryEvaluationCheckpoint),
}

/// Terminal reason recorded by a frozen recovery-track receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryCampaignTermination {
    /// The non-token-budgeted PTQ conversion completed.
    PtqComplete,
    /// Three consecutive frozen evaluations failed to improve the best point.
    ThreeEvaluationsWithoutImprovement,
    /// The full refinement token cap was evaluated.
    TokenCapReached,
}

/// Result of accepting one frozen campaign measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryCampaignDecision {
    /// The next frozen evaluation checkpoint remains authorized.
    Continue,
    /// The track is terminal for the enclosed reason.
    Complete(RecoveryCampaignTermination),
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RecoveryCampaignBest {
    checkpoint: RecoverySelectedCheckpoint,
    validation_quality: f64,
    artifact_digest: RecoveryEvidenceDigest,
}

const RECOVERY_RECEIPT_MAGIC: [u8; 8] = *b"TRRCPT\0\0";
const RECOVERY_RECEIPT_VERSION: u16 = 2;
const RECOVERY_RECEIPT_CONTENT_BYTES: usize = 285;
const RECOVERY_RECEIPT_BYTES: usize = RECOVERY_RECEIPT_CONTENT_BYTES + 32;
const RECOVERY_RECEIPT_HASH_CONTEXT: &str = "tritium recovery campaign receipt v2";

/// Immutable authorization and closeout evidence for one frozen recovery track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveryCampaignReceipt {
    source_model: RecoverySourceModel,
    source_model_id: RecoverySourceModelId,
    model_rung: RecoveryModelRung,
    activation_rung: RecoveryActivationRung,
    track: RecoveryTrack,
    campaign_digest: RecoveryEvidenceDigest,
    predecessor_receipt_digest: Option<RecoveryEvidenceDigest>,
    receipt_digest: RecoveryEvidenceDigest,
    token_cap: Option<u64>,
    tokens_consumed: u64,
    evaluations_completed: u8,
    best: RecoveryCampaignBest,
    termination: RecoveryCampaignTermination,
    promotion_gate: RecoveryPromotionGate,
    promotion_evidence: RecoveryPromotionEvidence,
    promotion_outcome: RecoveryPromotionOutcome,
}

impl RecoveryCampaignReceipt {
    /// Serialize this receipt into its canonical, versioned durable encoding.
    ///
    /// The final 32 bytes contain a BLAKE3 digest over every preceding semantic
    /// field. The digest field is deliberately excluded from its own preimage.
    ///
    /// # Errors
    /// Rejects an internally inconsistent receipt or an allocation failure.
    pub fn to_bytes(self) -> Result<Vec<u8>, RecoveryError> {
        self.validate_semantics()?;
        let mut bytes = self.canonical_content_bytes()?;
        let expected = receipt_content_digest(&bytes)?;
        if expected != self.receipt_digest {
            return Err(RecoveryError::CampaignEvidenceDigestMismatch);
        }
        bytes
            .try_reserve_exact(32)
            .map_err(|_| RecoveryError::AllocationFailed("campaign receipt encoding"))?;
        bytes.extend_from_slice(&self.receipt_digest.as_bytes());
        Ok(bytes)
    }

    /// Reopen and strictly verify a canonical durable receipt.
    ///
    /// # Errors
    /// Rejects truncation, trailing bytes, unknown versions, non-canonical
    /// encodings, invalid semantic combinations, or a content-digest mismatch.
    pub fn reopen(bytes: &[u8]) -> Result<Self, RecoveryError> {
        if bytes.len() < RECOVERY_RECEIPT_MAGIC.len() + core::mem::size_of::<u16>() {
            return Err(RecoveryError::InvalidCampaignReceipt("truncated header"));
        }
        if bytes[..RECOVERY_RECEIPT_MAGIC.len()] != RECOVERY_RECEIPT_MAGIC {
            return Err(RecoveryError::InvalidCampaignReceipt("invalid magic"));
        }
        let version = u16::from_le_bytes(
            bytes[RECOVERY_RECEIPT_MAGIC.len()..RECOVERY_RECEIPT_MAGIC.len() + 2]
                .try_into()
                .expect("fixed version slice"),
        );
        if version != RECOVERY_RECEIPT_VERSION {
            return Err(RecoveryError::UnsupportedCampaignReceiptVersion(version));
        }
        if bytes.len() < RECOVERY_RECEIPT_BYTES {
            return Err(RecoveryError::InvalidCampaignReceipt("truncated receipt"));
        }
        if bytes.len() > RECOVERY_RECEIPT_BYTES {
            return Err(RecoveryError::InvalidCampaignReceipt(
                "trailing receipt bytes",
            ));
        }

        let mut cursor = RecoveryReceiptCursor::new(bytes);
        cursor.read_exact::<8>()?;
        cursor.read_exact::<2>()?;
        let source_model = decode_source_model(cursor.read_u8()?)?;
        let source_model_id = RecoverySourceModelId::new(cursor.read_exact::<32>()?)?;
        let model_rung = decode_model_rung(cursor.read_u8()?)?;
        let activation_rung = decode_activation_rung(cursor.read_u8()?)?;
        let track = decode_track(cursor.read_u8()?)?;
        let campaign_digest = RecoveryEvidenceDigest::new(cursor.read_exact::<32>()?)?;
        let predecessor_present = cursor.read_u8()?;
        let predecessor_bytes = cursor.read_exact::<32>()?;
        let predecessor_receipt_digest = match predecessor_present {
            0 if predecessor_bytes == [0; 32] => None,
            1 => Some(RecoveryEvidenceDigest::new(predecessor_bytes)?),
            0 => {
                return Err(RecoveryError::InvalidCampaignReceipt(
                    "non-canonical absent predecessor",
                ));
            }
            _ => {
                return Err(RecoveryError::InvalidCampaignReceipt(
                    "invalid predecessor tag",
                ));
            }
        };
        let token_cap_present = cursor.read_u8()?;
        let token_cap_value = cursor.read_u64()?;
        let token_cap = match token_cap_present {
            0 if token_cap_value == 0 => None,
            1 => Some(token_cap_value),
            0 => {
                return Err(RecoveryError::InvalidCampaignReceipt(
                    "non-canonical absent token cap",
                ));
            }
            _ => {
                return Err(RecoveryError::InvalidCampaignReceipt(
                    "invalid token-cap tag",
                ));
            }
        };
        let tokens_consumed = cursor.read_u64()?;
        let evaluations_completed = cursor.read_u8()?;
        let checkpoint_tag = cursor.read_u8()?;
        let checkpoint_detail = cursor.read_u8()?;
        let checkpoint = decode_selected_checkpoint(checkpoint_tag, checkpoint_detail)?;
        let validation_quality = cursor.read_f64()?;
        let artifact_digest = RecoveryEvidenceDigest::new(cursor.read_exact::<32>()?)?;
        let termination = decode_termination(cursor.read_u8()?)?;
        let promotion_gate = RecoveryPromotionGate::new(
            cursor.read_f64()?,
            RecoveryEvidenceDigest::new(cursor.read_exact::<32>()?)?,
        )?;
        let promotion_evidence = RecoveryPromotionEvidence {
            source_model,
            source_model_id,
            activation_rung,
            track,
            campaign_digest,
            promotion_gate,
            selected_checkpoint: checkpoint,
            validation_quality: cursor.read_f64()?,
            artifact_digest: RecoveryEvidenceDigest::new(cursor.read_exact::<32>()?)?,
            evidence_digest: RecoveryEvidenceDigest::new(cursor.read_exact::<32>()?)?,
        };
        let promotion_outcome = decode_promotion_outcome(cursor.read_u8()?)?;
        let receipt_digest = RecoveryEvidenceDigest::new(cursor.read_exact::<32>()?)?;
        if !cursor.is_empty() {
            return Err(RecoveryError::InvalidCampaignReceipt(
                "trailing receipt bytes",
            ));
        }

        let receipt = Self {
            source_model,
            source_model_id,
            model_rung,
            activation_rung,
            track,
            campaign_digest,
            predecessor_receipt_digest,
            receipt_digest,
            token_cap,
            tokens_consumed,
            evaluations_completed,
            best: RecoveryCampaignBest {
                checkpoint,
                validation_quality,
                artifact_digest,
            },
            termination,
            promotion_gate,
            promotion_evidence,
            promotion_outcome,
        };
        let canonical = receipt.canonical_content_bytes()?;
        if canonical.as_slice() != &bytes[..RECOVERY_RECEIPT_CONTENT_BYTES] {
            return Err(RecoveryError::InvalidCampaignReceipt(
                "non-canonical receipt encoding",
            ));
        }
        if receipt_content_digest(&canonical)? != receipt.receipt_digest {
            return Err(RecoveryError::CampaignEvidenceDigestMismatch);
        }
        receipt.validate_semantics()?;
        Ok(receipt)
    }

    /// Exact source model family and scale.
    #[must_use]
    pub const fn source_model(self) -> RecoverySourceModel {
        self.source_model
    }

    /// Exact semantic source checkpoint identity.
    #[must_use]
    pub const fn source_model_id(self) -> RecoverySourceModelId {
        self.source_model_id
    }

    /// Model rung this evidence authorizes.
    #[must_use]
    pub const fn model_rung(self) -> RecoveryModelRung {
        self.model_rung
    }

    /// Activation rung this evidence authorizes.
    #[must_use]
    pub const fn activation_rung(self) -> RecoveryActivationRung {
        self.activation_rung
    }

    /// Separately labeled conversion or refinement track.
    #[must_use]
    pub const fn track(self) -> RecoveryTrack {
        self.track
    }

    /// Frozen campaign identity shared by every predecessor in the chain.
    #[must_use]
    pub const fn campaign_digest(self) -> RecoveryEvidenceDigest {
        self.campaign_digest
    }

    /// Receipt digest of the exact predecessor used for authorization.
    #[must_use]
    pub const fn predecessor_receipt_digest(self) -> Option<RecoveryEvidenceDigest> {
        self.predecessor_receipt_digest
    }

    /// Content digest that must match when this receipt is reopened.
    #[must_use]
    pub const fn receipt_digest(self) -> RecoveryEvidenceDigest {
        self.receipt_digest
    }

    /// Exact token cap, or `None` for PTQ.
    #[must_use]
    pub const fn token_cap(self) -> Option<u64> {
        self.token_cap
    }

    /// Cumulative refinement tokens consumed by the terminal point.
    #[must_use]
    pub const fn tokens_consumed(self) -> u64 {
        self.tokens_consumed
    }

    /// Number of accepted frozen refinement evaluations.
    #[must_use]
    pub const fn evaluations_completed(self) -> u8 {
        self.evaluations_completed
    }

    /// Best retained PTQ, predecessor, or refinement checkpoint.
    #[must_use]
    pub const fn selected_checkpoint(self) -> RecoverySelectedCheckpoint {
        self.best.checkpoint
    }

    /// Frozen validation aggregate for the selected checkpoint.
    ///
    /// The campaign contract defines larger values as better. Callers that use
    /// losses must negate or otherwise preregister their aggregate before use.
    #[must_use]
    pub const fn selected_validation_quality(self) -> f64 {
        self.best.validation_quality
    }

    /// Artifact digest for the selected checkpoint.
    #[must_use]
    pub const fn selected_artifact_digest(self) -> RecoveryEvidenceDigest {
        self.best.artifact_digest
    }

    /// Why this track stopped.
    #[must_use]
    pub const fn termination(self) -> RecoveryCampaignTermination {
        self.termination
    }

    /// Explicit preregistered promotion result.
    #[must_use]
    pub const fn promotion_outcome(self) -> RecoveryPromotionOutcome {
        self.promotion_outcome
    }

    /// Digest of independently persisted promotion evidence.
    #[must_use]
    pub const fn promotion_evidence_digest(self) -> RecoveryEvidenceDigest {
        self.promotion_evidence.evidence_digest
    }

    /// Digest of preregistered gate definition applied to this receipt.
    #[must_use]
    pub const fn promotion_gate_digest(self) -> RecoveryEvidenceDigest {
        self.promotion_gate.gate_digest
    }

    fn canonical_content_bytes(self) -> Result<Vec<u8>, RecoveryError> {
        let mut out = Vec::new();
        out.try_reserve_exact(RECOVERY_RECEIPT_CONTENT_BYTES)
            .map_err(|_| RecoveryError::AllocationFailed("campaign receipt encoding"))?;
        out.extend_from_slice(&RECOVERY_RECEIPT_MAGIC);
        out.extend_from_slice(&RECOVERY_RECEIPT_VERSION.to_le_bytes());
        out.push(encode_source_model(self.source_model));
        out.extend_from_slice(&self.source_model_id.as_bytes());
        out.push(encode_model_rung(self.model_rung));
        out.push(encode_activation_rung(self.activation_rung));
        out.push(encode_track(self.track));
        out.extend_from_slice(&self.campaign_digest.as_bytes());
        match self.predecessor_receipt_digest {
            Some(digest) => {
                out.push(1);
                out.extend_from_slice(&digest.as_bytes());
            }
            None => {
                out.push(0);
                out.extend_from_slice(&[0; 32]);
            }
        }
        match self.token_cap {
            Some(token_cap) => {
                out.push(1);
                out.extend_from_slice(&token_cap.to_le_bytes());
            }
            None => {
                out.push(0);
                out.extend_from_slice(&0_u64.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.tokens_consumed.to_le_bytes());
        out.push(self.evaluations_completed);
        let (checkpoint, detail) = encode_selected_checkpoint(self.best.checkpoint);
        out.push(checkpoint);
        out.push(detail);
        out.extend_from_slice(&canonical_f64_bits(self.best.validation_quality).to_le_bytes());
        out.extend_from_slice(&self.best.artifact_digest.as_bytes());
        out.push(encode_termination(self.termination));
        out.extend_from_slice(
            &canonical_f64_bits(self.promotion_gate.minimum_validation_quality).to_le_bytes(),
        );
        out.extend_from_slice(&self.promotion_gate.gate_digest.as_bytes());
        out.extend_from_slice(
            &canonical_f64_bits(self.promotion_evidence.validation_quality).to_le_bytes(),
        );
        out.extend_from_slice(&self.promotion_evidence.artifact_digest.as_bytes());
        out.extend_from_slice(&self.promotion_evidence.evidence_digest.as_bytes());
        out.push(encode_promotion_outcome(self.promotion_outcome));
        debug_assert_eq!(out.len(), RECOVERY_RECEIPT_CONTENT_BYTES);
        Ok(out)
    }

    fn validate_semantics(self) -> Result<(), RecoveryError> {
        if self.source_model.model_rung() != self.model_rung {
            return Err(RecoveryError::InvalidCampaignReceipt("model-rung mismatch"));
        }
        if self.token_cap != self.model_rung.token_cap(self.track) {
            return Err(RecoveryError::InvalidCampaignReceipt("token-cap mismatch"));
        }
        let predecessor_required =
            required_campaign_predecessor(self.activation_rung, self.track).is_some();
        if predecessor_required != self.predecessor_receipt_digest.is_some() {
            return Err(RecoveryError::InvalidCampaignReceipt(
                "predecessor presence mismatch",
            ));
        }
        if !self.best.validation_quality.is_finite()
            || !self.promotion_gate.minimum_validation_quality.is_finite()
            || !self.promotion_evidence.validation_quality.is_finite()
        {
            return Err(RecoveryError::InvalidCampaignReceipt("non-finite metric"));
        }
        let expected_promotion_evidence = RecoveryPromotionEvidence {
            source_model: self.source_model,
            source_model_id: self.source_model_id,
            activation_rung: self.activation_rung,
            track: self.track,
            campaign_digest: self.campaign_digest,
            promotion_gate: self.promotion_gate,
            selected_checkpoint: self.best.checkpoint,
            validation_quality: self.best.validation_quality,
            artifact_digest: self.best.artifact_digest,
            evidence_digest: self.promotion_evidence.evidence_digest,
        };
        if !self
            .promotion_evidence
            .same_semantics(expected_promotion_evidence)
        {
            return Err(RecoveryError::CampaignPromotionEvidenceMismatch);
        }
        self.promotion_evidence.validate_digest()?;
        let expected_outcome = if self.promotion_evidence.validation_quality
            >= self.promotion_gate.minimum_validation_quality
        {
            RecoveryPromotionOutcome::Passed
        } else {
            RecoveryPromotionOutcome::Failed
        };
        if self.promotion_outcome != expected_outcome {
            return Err(RecoveryError::InvalidCampaignReceipt(
                "promotion outcome mismatch",
            ));
        }
        match self.track {
            RecoveryTrack::Ptq => {
                if self.token_cap.is_some()
                    || self.tokens_consumed != 0
                    || self.evaluations_completed != 0
                    || self.best.checkpoint != RecoverySelectedCheckpoint::Ptq
                    || self.termination != RecoveryCampaignTermination::PtqComplete
                {
                    return Err(RecoveryError::InvalidCampaignReceipt(
                        "invalid PTQ closeout",
                    ));
                }
            }
            RecoveryTrack::ScaleOnly | RecoveryTrack::Pv => {
                let cap = self
                    .token_cap
                    .ok_or(RecoveryError::InvalidCampaignReceipt("missing token cap"))?;
                let completed = usize::from(self.evaluations_completed);
                if !(1..=RecoveryEvaluationCheckpoint::ALL.len()).contains(&completed) {
                    return Err(RecoveryError::InvalidCampaignReceipt(
                        "invalid evaluation count",
                    ));
                }
                let expected_tokens = RecoveryEvaluationCheckpoint::ALL[completed - 1].tokens(cap);
                if self.tokens_consumed != expected_tokens
                    || self.best.checkpoint == RecoverySelectedCheckpoint::Ptq
                {
                    return Err(RecoveryError::InvalidCampaignReceipt(
                        "invalid refinement closeout",
                    ));
                }
                if let RecoverySelectedCheckpoint::Evaluation(checkpoint) = self.best.checkpoint {
                    let selected = RecoveryEvaluationCheckpoint::ALL
                        .iter()
                        .position(|candidate| *candidate == checkpoint)
                        .expect("checkpoint belongs to frozen list");
                    if selected >= completed {
                        return Err(RecoveryError::InvalidCampaignReceipt(
                            "selected unevaluated checkpoint",
                        ));
                    }
                }
                if !valid_refinement_terminal(
                    self.termination,
                    self.evaluations_completed,
                    self.best.checkpoint,
                ) {
                    return Err(RecoveryError::InvalidCampaignReceipt(
                        "impossible refinement terminal history",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Reopened predecessor receipt whose durable digest has been checked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveryPredecessorEvidence {
    receipt: RecoveryCampaignReceipt,
}

impl RecoveryPredecessorEvidence {
    /// Reopen a canonical receipt and independently persisted promotion evidence.
    ///
    /// # Errors
    /// Rejects either malformed artifact and compares every campaign context,
    /// selected checkpoint, quality, artifact, gate, and content digest field
    /// duplicated across the two durable encodings.
    pub fn new(
        receipt_bytes: &[u8],
        promotion_evidence_bytes: &[u8],
    ) -> Result<Self, RecoveryError> {
        let receipt = RecoveryCampaignReceipt::reopen(receipt_bytes)?;
        let promotion_evidence = RecoveryPromotionEvidence::reopen(promotion_evidence_bytes)?;
        if !receipt
            .promotion_evidence
            .same_semantics(promotion_evidence)
        {
            return Err(RecoveryError::CampaignPromotionEvidenceMismatch);
        }
        if promotion_evidence.evidence_digest != receipt.promotion_evidence.evidence_digest {
            return Err(RecoveryError::CampaignPromotionEvidenceDigestMismatch);
        }
        Ok(Self { receipt })
    }
}

fn receipt_content_digest(bytes: &[u8]) -> Result<RecoveryEvidenceDigest, RecoveryError> {
    let mut hasher = blake3::Hasher::new_derive_key(RECOVERY_RECEIPT_HASH_CONTEXT);
    hasher.update(bytes);
    RecoveryEvidenceDigest::new(*hasher.finalize().as_bytes())
}

fn promotion_evidence_content_digest(
    bytes: &[u8],
) -> Result<RecoveryEvidenceDigest, RecoveryError> {
    let mut hasher = blake3::Hasher::new_derive_key(RECOVERY_PROMOTION_EVIDENCE_HASH_CONTEXT);
    hasher.update(bytes);
    RecoveryEvidenceDigest::new(*hasher.finalize().as_bytes())
}

const fn canonical_f64_bits(value: f64) -> u64 {
    if value == 0.0 { 0 } else { value.to_bits() }
}

fn canonical_f64(value: f64) -> f64 {
    f64::from_bits(canonical_f64_bits(value))
}

const fn valid_refinement_terminal(
    termination: RecoveryCampaignTermination,
    evaluations_completed: u8,
    selected_checkpoint: RecoverySelectedCheckpoint,
) -> bool {
    matches!(
        (termination, evaluations_completed, selected_checkpoint),
        (
            RecoveryCampaignTermination::ThreeEvaluationsWithoutImprovement,
            3,
            RecoverySelectedCheckpoint::Predecessor
        ) | (
            RecoveryCampaignTermination::ThreeEvaluationsWithoutImprovement,
            4,
            RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneEighth)
        ) | (
            RecoveryCampaignTermination::TokenCapReached,
            4,
            RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneQuarter)
        ) | (
            RecoveryCampaignTermination::TokenCapReached,
            4,
            RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneHalf)
        ) | (
            RecoveryCampaignTermination::TokenCapReached,
            4,
            RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::Full)
        )
    )
}

const fn encode_source_model(value: RecoverySourceModel) -> u8 {
    match value {
        RecoverySourceModel::SmolLm2OneThirtyFiveMillion => 0,
        RecoverySourceModel::SmolLm2OnePointSevenBillion => 1,
        RecoverySourceModel::Qwen3EightBillion => 2,
        RecoverySourceModel::Qwen3ThirtyTwoBillion => 3,
    }
}

const fn encode_model_rung(value: RecoveryModelRung) -> u8 {
    match value {
        RecoveryModelRung::Pilot => 0,
        RecoveryModelRung::Qwen8B => 1,
        RecoveryModelRung::Qwen32B => 2,
    }
}

const fn encode_activation_rung(value: RecoveryActivationRung) -> u8 {
    match value {
        RecoveryActivationRung::A16 => 0,
        RecoveryActivationRung::A8 => 1,
        RecoveryActivationRung::A4 => 2,
    }
}

const fn encode_track(value: RecoveryTrack) -> u8 {
    match value {
        RecoveryTrack::Ptq => 0,
        RecoveryTrack::ScaleOnly => 1,
        RecoveryTrack::Pv => 2,
    }
}

const fn encode_selected_checkpoint(value: RecoverySelectedCheckpoint) -> (u8, u8) {
    match value {
        RecoverySelectedCheckpoint::Ptq => (0, 0),
        RecoverySelectedCheckpoint::Predecessor => (1, 0),
        RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneEighth) => (2, 0),
        RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneQuarter) => (2, 1),
        RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneHalf) => (2, 2),
        RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::Full) => (2, 3),
    }
}

const fn encode_termination(value: RecoveryCampaignTermination) -> u8 {
    match value {
        RecoveryCampaignTermination::PtqComplete => 0,
        RecoveryCampaignTermination::ThreeEvaluationsWithoutImprovement => 1,
        RecoveryCampaignTermination::TokenCapReached => 2,
    }
}

const fn encode_promotion_outcome(value: RecoveryPromotionOutcome) -> u8 {
    match value {
        RecoveryPromotionOutcome::Passed => 0,
        RecoveryPromotionOutcome::Failed => 1,
    }
}

fn decode_source_model(value: u8) -> Result<RecoverySourceModel, RecoveryError> {
    match value {
        0 => Ok(RecoverySourceModel::SmolLm2OneThirtyFiveMillion),
        1 => Ok(RecoverySourceModel::SmolLm2OnePointSevenBillion),
        2 => Ok(RecoverySourceModel::Qwen3EightBillion),
        3 => Ok(RecoverySourceModel::Qwen3ThirtyTwoBillion),
        _ => Err(RecoveryError::InvalidCampaignReceipt(
            "invalid source-model tag",
        )),
    }
}

fn decode_model_rung(value: u8) -> Result<RecoveryModelRung, RecoveryError> {
    match value {
        0 => Ok(RecoveryModelRung::Pilot),
        1 => Ok(RecoveryModelRung::Qwen8B),
        2 => Ok(RecoveryModelRung::Qwen32B),
        _ => Err(RecoveryError::InvalidCampaignReceipt(
            "invalid model-rung tag",
        )),
    }
}

fn decode_activation_rung(value: u8) -> Result<RecoveryActivationRung, RecoveryError> {
    match value {
        0 => Ok(RecoveryActivationRung::A16),
        1 => Ok(RecoveryActivationRung::A8),
        2 => Ok(RecoveryActivationRung::A4),
        _ => Err(RecoveryError::InvalidCampaignReceipt(
            "invalid activation-rung tag",
        )),
    }
}

fn decode_track(value: u8) -> Result<RecoveryTrack, RecoveryError> {
    match value {
        0 => Ok(RecoveryTrack::Ptq),
        1 => Ok(RecoveryTrack::ScaleOnly),
        2 => Ok(RecoveryTrack::Pv),
        _ => Err(RecoveryError::InvalidCampaignReceipt("invalid track tag")),
    }
}

fn decode_selected_checkpoint(
    checkpoint: u8,
    detail: u8,
) -> Result<RecoverySelectedCheckpoint, RecoveryError> {
    match (checkpoint, detail) {
        (0, 0) => Ok(RecoverySelectedCheckpoint::Ptq),
        (1, 0) => Ok(RecoverySelectedCheckpoint::Predecessor),
        (2, 0) => Ok(RecoverySelectedCheckpoint::Evaluation(
            RecoveryEvaluationCheckpoint::OneEighth,
        )),
        (2, 1) => Ok(RecoverySelectedCheckpoint::Evaluation(
            RecoveryEvaluationCheckpoint::OneQuarter,
        )),
        (2, 2) => Ok(RecoverySelectedCheckpoint::Evaluation(
            RecoveryEvaluationCheckpoint::OneHalf,
        )),
        (2, 3) => Ok(RecoverySelectedCheckpoint::Evaluation(
            RecoveryEvaluationCheckpoint::Full,
        )),
        _ => Err(RecoveryError::InvalidCampaignReceipt(
            "invalid selected-checkpoint tag",
        )),
    }
}

fn decode_termination(value: u8) -> Result<RecoveryCampaignTermination, RecoveryError> {
    match value {
        0 => Ok(RecoveryCampaignTermination::PtqComplete),
        1 => Ok(RecoveryCampaignTermination::ThreeEvaluationsWithoutImprovement),
        2 => Ok(RecoveryCampaignTermination::TokenCapReached),
        _ => Err(RecoveryError::InvalidCampaignReceipt(
            "invalid termination tag",
        )),
    }
}

fn decode_promotion_outcome(value: u8) -> Result<RecoveryPromotionOutcome, RecoveryError> {
    match value {
        0 => Ok(RecoveryPromotionOutcome::Passed),
        1 => Ok(RecoveryPromotionOutcome::Failed),
        _ => Err(RecoveryError::InvalidCampaignReceipt(
            "invalid promotion-outcome tag",
        )),
    }
}

struct RecoveryPromotionEvidenceCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> RecoveryPromotionEvidenceCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N], RecoveryError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or(RecoveryError::InvalidPromotionEvidence(
                "evidence length overflow",
            ))?;
        let slice =
            self.bytes
                .get(self.offset..end)
                .ok_or(RecoveryError::InvalidPromotionEvidence(
                    "truncated evidence",
                ))?;
        self.offset = end;
        Ok(slice.try_into().expect("fixed-size evidence slice"))
    }

    fn read_u8(&mut self) -> Result<u8, RecoveryError> {
        Ok(self.read_exact::<1>()?[0])
    }

    fn read_u64(&mut self) -> Result<u64, RecoveryError> {
        Ok(u64::from_le_bytes(self.read_exact::<8>()?))
    }

    fn read_f64(&mut self) -> Result<f64, RecoveryError> {
        let bits = self.read_u64()?;
        let value = f64::from_bits(bits);
        if !value.is_finite() || canonical_f64_bits(value) != bits {
            return Err(RecoveryError::InvalidPromotionEvidence(
                "non-canonical evidence metric",
            ));
        }
        Ok(value)
    }

    const fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

struct RecoveryReceiptCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> RecoveryReceiptCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N], RecoveryError> {
        let end = self
            .offset
            .checked_add(N)
            .ok_or(RecoveryError::InvalidCampaignReceipt(
                "receipt length overflow",
            ))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or(RecoveryError::InvalidCampaignReceipt("truncated receipt"))?;
        self.offset = end;
        Ok(slice.try_into().expect("fixed-size receipt slice"))
    }

    fn read_u8(&mut self) -> Result<u8, RecoveryError> {
        Ok(self.read_exact::<1>()?[0])
    }

    fn read_u64(&mut self) -> Result<u64, RecoveryError> {
        Ok(u64::from_le_bytes(self.read_exact::<8>()?))
    }

    fn read_f64(&mut self) -> Result<f64, RecoveryError> {
        let bits = self.read_u64()?;
        let value = f64::from_bits(bits);
        if !value.is_finite() || canonical_f64_bits(value) != bits {
            return Err(RecoveryError::InvalidCampaignReceipt(
                "non-canonical receipt metric",
            ));
        }
        Ok(value)
    }

    const fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// Stateful authorization and evaluation control for one frozen campaign track.
#[derive(Clone, Debug, PartialEq)]
pub struct RecoveryCampaignRun {
    source_model: RecoverySourceModel,
    source_model_id: RecoverySourceModelId,
    model_rung: RecoveryModelRung,
    activation_rung: RecoveryActivationRung,
    track: RecoveryTrack,
    campaign_digest: RecoveryEvidenceDigest,
    predecessor_receipt_digest: Option<RecoveryEvidenceDigest>,
    token_cap: Option<u64>,
    tokens_consumed: u64,
    next_evaluation: usize,
    consecutive_no_improvement: u8,
    best: Option<RecoveryCampaignBest>,
    termination: Option<RecoveryCampaignTermination>,
    promotion_gate: RecoveryPromotionGate,
}

impl RecoveryCampaignRun {
    /// Start a track only when its exact predecessor has been reopened.
    ///
    /// The chain is A16/PTQ -> A16/ScaleOnly -> A16/PV -> A8/PTQ and repeats
    /// through A8/PV -> A4/PTQ -> A4/ScaleOnly -> A4/PV. This serializes both
    /// the activation proof order and the PTQ-before-refinement spend gate.
    ///
    /// # Errors
    /// Fails closed on a missing, unexpected, mislabeled, failed-gate,
    /// cross-model, cross-campaign, or digest-mismatched predecessor.
    pub fn start(
        plan: RecoveryCampaignPlan,
        predecessor: Option<RecoveryPredecessorEvidence>,
    ) -> Result<Self, RecoveryError> {
        let model_rung = plan.source_model.model_rung();
        let required = required_campaign_predecessor(plan.activation_rung, plan.track);
        let receipt = match (required, predecessor) {
            (None, None) => None,
            (None, Some(_)) => return Err(RecoveryError::UnexpectedCampaignPredecessor),
            (Some((activation, predecessor_track)), None) => {
                return Err(RecoveryError::CampaignPredecessorRequired {
                    activation,
                    track: predecessor_track,
                });
            }
            (Some((expected_activation, expected_track)), Some(evidence)) => {
                let receipt = evidence.receipt;
                if receipt.activation_rung != expected_activation || receipt.track != expected_track
                {
                    return Err(RecoveryError::CampaignPredecessorLabelMismatch {
                        expected_activation,
                        expected_track,
                        actual_activation: receipt.activation_rung,
                        actual_track: receipt.track,
                    });
                }
                if receipt.model_rung != model_rung {
                    return Err(RecoveryError::CampaignModelRungMismatch {
                        expected: model_rung,
                        actual: receipt.model_rung,
                    });
                }
                if receipt.source_model != plan.source_model
                    || receipt.source_model_id != plan.source_model_id
                {
                    return Err(RecoveryError::CampaignSourceModelMismatch);
                }
                if receipt.campaign_digest != plan.campaign_digest {
                    return Err(RecoveryError::CampaignDigestMismatch);
                }
                if receipt.promotion_outcome != RecoveryPromotionOutcome::Passed {
                    return Err(RecoveryError::CampaignPredecessorGateNotPassed);
                }
                Some(receipt)
            }
        };

        let token_cap = model_rung.token_cap(plan.track);
        let best = if plan.track == RecoveryTrack::Ptq {
            None
        } else {
            receipt.map(|predecessor| RecoveryCampaignBest {
                checkpoint: RecoverySelectedCheckpoint::Predecessor,
                validation_quality: predecessor.best.validation_quality,
                artifact_digest: predecessor.best.artifact_digest,
            })
        };
        Ok(Self {
            source_model: plan.source_model,
            source_model_id: plan.source_model_id,
            model_rung,
            activation_rung: plan.activation_rung,
            track: plan.track,
            campaign_digest: plan.campaign_digest,
            predecessor_receipt_digest: receipt.map(|receipt| receipt.receipt_digest),
            token_cap,
            tokens_consumed: 0,
            next_evaluation: 0,
            consecutive_no_improvement: 0,
            best,
            termination: None,
            promotion_gate: plan.promotion_gate,
        })
    }

    /// Record terminal PTQ evidence before any paid refinement can start.
    ///
    /// `validation_quality` is the preregistered aggregate with larger values
    /// defined as better.
    ///
    /// # Errors
    /// Rejects non-PTQ tracks, non-finite quality, or a terminal run.
    pub fn record_ptq(
        &mut self,
        validation_quality: f64,
        artifact_digest: RecoveryEvidenceDigest,
    ) -> Result<RecoveryCampaignDecision, RecoveryError> {
        if self.termination.is_some() {
            return Err(RecoveryError::CampaignEvaluationAfterTerminal);
        }
        if self.track != RecoveryTrack::Ptq {
            return Err(RecoveryError::CampaignTrackRequiresEvaluations);
        }
        if !validation_quality.is_finite() {
            return Err(RecoveryError::NonFiniteMetric(
                "campaign validation quality",
            ));
        }
        let termination = RecoveryCampaignTermination::PtqComplete;
        self.best = Some(RecoveryCampaignBest {
            checkpoint: RecoverySelectedCheckpoint::Ptq,
            validation_quality: canonical_f64(validation_quality),
            artifact_digest,
        });
        self.termination = Some(termination);
        Ok(RecoveryCampaignDecision::Complete(termination))
    }

    /// Record the next exact 1/8, 1/4, 1/2, or full-cap evaluation.
    ///
    /// Only a strictly larger preregistered validation aggregate counts as an
    /// improvement. The best earlier checkpoint is retained, and three
    /// consecutive non-improvements terminate the track.
    ///
    /// # Errors
    /// Rejects PTQ, non-finite quality, out-of-order or over-cap tokens, and
    /// observations after a terminal decision.
    pub fn record_evaluation(
        &mut self,
        cumulative_tokens: u64,
        validation_quality: f64,
        artifact_digest: RecoveryEvidenceDigest,
    ) -> Result<RecoveryCampaignDecision, RecoveryError> {
        if self.termination.is_some() {
            return Err(RecoveryError::CampaignEvaluationAfterTerminal);
        }
        let token_cap = self
            .token_cap
            .ok_or(RecoveryError::CampaignTrackHasNoTokenSchedule)?;
        if !validation_quality.is_finite() {
            return Err(RecoveryError::NonFiniteMetric(
                "campaign validation quality",
            ));
        }
        if cumulative_tokens > token_cap {
            return Err(RecoveryError::CampaignTokenCapExceeded {
                token_cap,
                actual_tokens: cumulative_tokens,
            });
        }
        let checkpoint = RecoveryEvaluationCheckpoint::ALL
            .get(self.next_evaluation)
            .copied()
            .ok_or(RecoveryError::CampaignEvaluationAfterTerminal)?;
        let expected_tokens = checkpoint.tokens(token_cap);
        if cumulative_tokens != expected_tokens {
            return Err(RecoveryError::CampaignEvaluationOrder {
                expected_tokens,
                actual_tokens: cumulative_tokens,
            });
        }

        let prior_best = self
            .best
            .ok_or(RecoveryError::CampaignMissingBestCheckpoint)?;
        let validation_quality = canonical_f64(validation_quality);
        if validation_quality > prior_best.validation_quality {
            self.best = Some(RecoveryCampaignBest {
                checkpoint: RecoverySelectedCheckpoint::Evaluation(checkpoint),
                validation_quality,
                artifact_digest,
            });
            self.consecutive_no_improvement = 0;
        } else {
            self.consecutive_no_improvement += 1;
        }
        self.tokens_consumed = cumulative_tokens;
        self.next_evaluation += 1;

        let termination = if self.consecutive_no_improvement == 3 {
            Some(RecoveryCampaignTermination::ThreeEvaluationsWithoutImprovement)
        } else if checkpoint == RecoveryEvaluationCheckpoint::Full {
            Some(RecoveryCampaignTermination::TokenCapReached)
        } else {
            None
        };
        self.termination = termination;
        Ok(
            termination.map_or(RecoveryCampaignDecision::Continue, |reason| {
                RecoveryCampaignDecision::Complete(reason)
            }),
        )
    }

    /// Prepare canonical, self-digesting promotion evidence for persistence.
    ///
    /// The returned value must be serialized, durably persisted, independently
    /// reopened, and supplied as bytes to [`Self::finish`].
    ///
    /// # Errors
    /// Rejects a run without a terminal decision or retained checkpoint.
    pub fn prepare_promotion_evidence(&self) -> Result<RecoveryPromotionEvidence, RecoveryError> {
        self.termination.ok_or(RecoveryError::CampaignNotTerminal)?;
        let best = self
            .best
            .ok_or(RecoveryError::CampaignMissingBestCheckpoint)?;
        RecoveryPromotionEvidence::from_run(self, best)
    }

    /// Close a terminal run using independently reopened promotion evidence.
    ///
    /// # Errors
    /// Rejects a non-terminal run, malformed evidence, or evidence bound to a
    /// different campaign context, gate, checkpoint, quality, or artifact.
    pub fn finish(
        self,
        promotion_evidence_bytes: &[u8],
    ) -> Result<RecoveryCampaignReceipt, RecoveryError> {
        let termination = self.termination.ok_or(RecoveryError::CampaignNotTerminal)?;
        let best = self
            .best
            .ok_or(RecoveryError::CampaignMissingBestCheckpoint)?;
        let evaluations_completed = u8::try_from(self.next_evaluation)
            .map_err(|_| RecoveryError::ArithmeticOverflow("campaign evaluations"))?;
        let expected_promotion_evidence = self.prepare_promotion_evidence()?;
        let promotion_evidence = RecoveryPromotionEvidence::reopen(promotion_evidence_bytes)?;
        if !promotion_evidence.same_semantics(expected_promotion_evidence) {
            return Err(RecoveryError::CampaignPromotionEvidenceMismatch);
        }
        if promotion_evidence.evidence_digest != expected_promotion_evidence.evidence_digest {
            return Err(RecoveryError::CampaignPromotionEvidenceDigestMismatch);
        }
        let promotion_outcome = if promotion_evidence.validation_quality
            >= self.promotion_gate.minimum_validation_quality
        {
            RecoveryPromotionOutcome::Passed
        } else {
            RecoveryPromotionOutcome::Failed
        };
        let mut receipt = RecoveryCampaignReceipt {
            source_model: self.source_model,
            source_model_id: self.source_model_id,
            model_rung: self.model_rung,
            activation_rung: self.activation_rung,
            track: self.track,
            campaign_digest: self.campaign_digest,
            predecessor_receipt_digest: self.predecessor_receipt_digest,
            receipt_digest: RecoveryEvidenceDigest([0; 32]),
            token_cap: self.token_cap,
            tokens_consumed: self.tokens_consumed,
            evaluations_completed,
            best,
            termination,
            promotion_gate: self.promotion_gate,
            promotion_evidence,
            promotion_outcome,
        };
        receipt.validate_semantics()?;
        receipt.receipt_digest = receipt_content_digest(&receipt.canonical_content_bytes()?)?;
        Ok(receipt)
    }
}

const fn required_campaign_predecessor(
    activation: RecoveryActivationRung,
    track: RecoveryTrack,
) -> Option<(RecoveryActivationRung, RecoveryTrack)> {
    match (activation, track) {
        (RecoveryActivationRung::A16, RecoveryTrack::Ptq) => None,
        (activation, RecoveryTrack::ScaleOnly) => Some((activation, RecoveryTrack::Ptq)),
        (activation, RecoveryTrack::Pv) => Some((activation, RecoveryTrack::ScaleOnly)),
        (RecoveryActivationRung::A8, RecoveryTrack::Ptq) => {
            Some((RecoveryActivationRung::A16, RecoveryTrack::Pv))
        }
        (RecoveryActivationRung::A4, RecoveryTrack::Ptq) => {
            Some((RecoveryActivationRung::A8, RecoveryTrack::Pv))
        }
    }
}

/// Generic token-count substrate for experiments outside the frozen Stage 5
/// recovery campaign.
///
/// Frozen SALT V2 recovery authorization uses
/// [`RecoveryEvaluationCheckpoint`] instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PromotionCheckpoint {
    /// First small-campaign gate.
    Tokens100M,
    /// Intermediate recovery gate.
    Tokens500M,
    /// One-billion-token scale gate.
    Tokens1B,
    /// Full ten-billion-token campaign gate.
    Tokens10B,
}

impl PromotionCheckpoint {
    /// Every checkpoint in ascending token order.
    pub const ALL: [Self; 4] = [
        Self::Tokens100M,
        Self::Tokens500M,
        Self::Tokens1B,
        Self::Tokens10B,
    ];

    /// Exact cumulative-token threshold represented by this checkpoint.
    #[must_use]
    pub const fn tokens(self) -> u64 {
        match self {
            Self::Tokens100M => 100_000_000,
            Self::Tokens500M => 500_000_000,
            Self::Tokens1B => 1_000_000_000,
            Self::Tokens10B => 10_000_000_000,
        }
    }

    /// Checkpoints newly crossed between two cumulative token counters.
    ///
    /// A checkpoint equal to `previous_tokens` has already been handled and is
    /// not returned. A checkpoint equal to `current_tokens` is returned.
    ///
    /// # Errors
    /// Rejects a decreasing cumulative token counter.
    pub fn crossed(previous_tokens: u64, current_tokens: u64) -> Result<Vec<Self>, RecoveryError> {
        if current_tokens < previous_tokens {
            return Err(RecoveryError::TokenOrder {
                previous_tokens,
                current_tokens,
            });
        }
        Ok(Self::ALL
            .into_iter()
            .filter(|checkpoint| {
                checkpoint.tokens() > previous_tokens && checkpoint.tokens() <= current_tokens
            })
            .collect())
    }
}

/// Quality evidence evaluated independently at each promotion checkpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromotionEvidence {
    held_out_nll_regression: f64,
    hard_minus_soft_nll: f64,
    code_churn: f64,
    packed_minus_training_nll: f64,
}

impl PromotionEvidence {
    /// Validate measured promotion evidence.
    ///
    /// Gaps and regression are signed; their promotion limits are applied to
    /// regression directly and to the two implementation gaps by magnitude.
    ///
    /// # Errors
    /// Rejects non-finite evidence or code churn outside `[0, 1]`.
    pub fn new(
        held_out_nll_regression: f64,
        hard_minus_soft_nll: f64,
        code_churn: f64,
        packed_minus_training_nll: f64,
    ) -> Result<Self, RecoveryError> {
        if !held_out_nll_regression.is_finite() {
            return Err(RecoveryError::InvalidMetric("held-out NLL regression"));
        }
        if !hard_minus_soft_nll.is_finite() {
            return Err(RecoveryError::InvalidMetric("hard-minus-soft NLL"));
        }
        if !code_churn.is_finite() || !(0.0..=1.0).contains(&code_churn) {
            return Err(RecoveryError::InvalidMetric("code churn"));
        }
        if !packed_minus_training_nll.is_finite() {
            return Err(RecoveryError::InvalidMetric("packed-minus-training NLL"));
        }
        Ok(Self {
            held_out_nll_regression,
            hard_minus_soft_nll,
            code_churn,
            packed_minus_training_nll,
        })
    }
}

/// Deterministic gate thresholds shared by all token checkpoints.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromotionGate {
    maximum_nll_regression: f64,
    maximum_hard_soft_gap: f64,
    maximum_code_churn: f64,
    maximum_packed_training_gap: f64,
}

impl PromotionGate {
    /// Construct a promotion gate from non-negative finite limits.
    ///
    /// # Errors
    /// Rejects negative/non-finite limits or a code-churn limit above one.
    pub fn new(
        maximum_nll_regression: f64,
        maximum_hard_soft_gap: f64,
        maximum_code_churn: f64,
        maximum_packed_training_gap: f64,
    ) -> Result<Self, RecoveryError> {
        if !maximum_nll_regression.is_finite() || maximum_nll_regression < 0.0 {
            return Err(RecoveryError::InvalidConfiguration(
                "maximum NLL regression must be finite and non-negative",
            ));
        }
        if !maximum_hard_soft_gap.is_finite() || maximum_hard_soft_gap < 0.0 {
            return Err(RecoveryError::InvalidConfiguration(
                "maximum hard-soft gap must be finite and non-negative",
            ));
        }
        if !maximum_code_churn.is_finite() || !(0.0..=1.0).contains(&maximum_code_churn) {
            return Err(RecoveryError::InvalidConfiguration(
                "maximum code churn must be finite and in [0, 1]",
            ));
        }
        if !maximum_packed_training_gap.is_finite() || maximum_packed_training_gap < 0.0 {
            return Err(RecoveryError::InvalidConfiguration(
                "maximum packed-training gap must be finite and non-negative",
            ));
        }
        Ok(Self {
            maximum_nll_regression,
            maximum_hard_soft_gap,
            maximum_code_churn,
            maximum_packed_training_gap,
        })
    }

    /// Apply this gate to immutable evidence.
    #[must_use]
    pub fn decide(self, evidence: PromotionEvidence) -> PromotionDecision {
        if evidence.held_out_nll_regression <= self.maximum_nll_regression
            && evidence.hard_minus_soft_nll.abs() <= self.maximum_hard_soft_gap
            && evidence.code_churn <= self.maximum_code_churn
            && evidence.packed_minus_training_nll.abs() <= self.maximum_packed_training_gap
        {
            PromotionDecision::Promote
        } else {
            PromotionDecision::Hold
        }
    }
}

/// Token-checkpoint campaign decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromotionDecision {
    /// Evidence meets every preregistered threshold.
    Promote,
    /// Keep the campaign at its current scale and retain the evidence.
    Hold,
}

/// One cumulative held-out evaluation for early-stop accounting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarlyStopPoint {
    tokens_processed: u64,
    gpu_seconds: u64,
    held_out_nll: f64,
}

impl EarlyStopPoint {
    /// Validate a cumulative evaluation point.
    ///
    /// # Errors
    /// Rejects zero tokens or non-finite/negative held-out NLL.
    pub fn new(
        tokens_processed: u64,
        gpu_seconds: u64,
        held_out_nll: f64,
    ) -> Result<Self, RecoveryError> {
        if tokens_processed == 0 {
            return Err(RecoveryError::InvalidMetric("cumulative tokens"));
        }
        if !held_out_nll.is_finite() || held_out_nll < 0.0 {
            return Err(RecoveryError::InvalidMetric("held-out NLL"));
        }
        Ok(Self {
            tokens_processed,
            gpu_seconds,
            held_out_nll,
        })
    }
}

/// Generic early-stop substrate based on marginal held-out NLL improvement per
/// GPU-hour.
///
/// This is not the frozen Stage 5 rule, which stops after three consecutive
/// evaluations without improvement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarlyStopGate {
    minimum_window_gpu_hours: f64,
    minimum_nll_improvement_per_gpu_hour: f64,
}

impl EarlyStopGate {
    /// Define the minimum measurement window and useful improvement rate.
    ///
    /// # Errors
    /// Rejects a non-positive/non-finite window or a negative/non-finite rate.
    pub fn new(
        minimum_window_gpu_hours: f64,
        minimum_nll_improvement_per_gpu_hour: f64,
    ) -> Result<Self, RecoveryError> {
        if !minimum_window_gpu_hours.is_finite() || minimum_window_gpu_hours <= 0.0 {
            return Err(RecoveryError::InvalidConfiguration(
                "minimum early-stop window must be finite and positive",
            ));
        }
        if !minimum_nll_improvement_per_gpu_hour.is_finite()
            || minimum_nll_improvement_per_gpu_hour < 0.0
        {
            return Err(RecoveryError::InvalidConfiguration(
                "minimum NLL improvement rate must be finite and non-negative",
            ));
        }
        Ok(Self {
            minimum_window_gpu_hours,
            minimum_nll_improvement_per_gpu_hour,
        })
    }

    /// Decide whether another training interval is justified.
    ///
    /// The gate always continues until the configured GPU-hour window is fully
    /// observed. At or beyond it, equality with the minimum rate continues;
    /// only a strictly smaller rate stops.
    ///
    /// # Errors
    /// Rejects non-increasing token or GPU-second counters and an unrepresentable
    /// floating-point improvement rate.
    pub fn decide(
        self,
        previous: EarlyStopPoint,
        current: EarlyStopPoint,
    ) -> Result<EarlyStopDecision, RecoveryError> {
        if current.tokens_processed <= previous.tokens_processed
            || current.gpu_seconds <= previous.gpu_seconds
        {
            return Err(RecoveryError::EvaluationOrder);
        }
        let delta_gpu_hours = (current.gpu_seconds - previous.gpu_seconds) as f64 / 3_600.0;
        if delta_gpu_hours < self.minimum_window_gpu_hours {
            return Ok(EarlyStopDecision::Continue);
        }
        let improvement_rate = (previous.held_out_nll - current.held_out_nll) / delta_gpu_hours;
        if !improvement_rate.is_finite() {
            return Err(RecoveryError::InvalidMetric("NLL improvement per GPU-hour"));
        }
        if improvement_rate < self.minimum_nll_improvement_per_gpu_hour {
            Ok(EarlyStopDecision::Stop)
        } else {
            Ok(EarlyStopDecision::Continue)
        }
    }
}

/// Marginal compute-efficiency decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EarlyStopDecision {
    /// The minimum window is incomplete or improvement remains worthwhile.
    Continue,
    /// Marginal held-out NLL improvement is below the preregistered rate.
    Stop,
}

/// A fail-closed SALT V2 recovery-policy error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryError {
    /// A configuration cannot be interpreted without guessing.
    InvalidSchedule(&'static str),
    /// A policy parameter violates its documented domain.
    InvalidConfiguration(&'static str),
    /// A measured value was NaN or infinite.
    NonFiniteMetric(&'static str),
    /// A measured value was finite but outside its valid domain.
    InvalidMetric(&'static str),
    /// A zero-based optimizer step lies outside the validated schedule.
    StepOutOfRange {
        /// Supplied step.
        step: u64,
        /// Exclusive schedule limit.
        total_steps: u64,
    },
    /// Stateful observations must be supplied in strictly increasing step order.
    ObservationOrder {
        /// Previously accepted observation step.
        previous_step: u64,
        /// Rejected observation step.
        current_step: u64,
    },
    /// Step accounting must cover the schedule exactly once and in order.
    StepOrder {
        /// Required next step.
        expected_step: u64,
        /// Supplied step.
        actual_step: u64,
    },
    /// Actual bypass was larger than the validated schedule allows.
    BypassAllowanceExceeded {
        /// Rejected zero-based step.
        step: u64,
    },
    /// Actual bypass increased relative to the prior observed step.
    BypassIncreased {
        /// Rejected zero-based step.
        step: u64,
    },
    /// Export was attempted before every scheduled step was accounted for.
    IncompleteRun {
        /// Number of accepted steps.
        completed_steps: u64,
        /// Required number of steps.
        total_steps: u64,
    },
    /// The final actual full-precision bypass was not exactly zero.
    NonZeroBypassAtExport,
    /// An accounting counter cannot represent the accumulated value.
    ArithmeticOverflow(&'static str),
    /// A bounded policy allocation could not be reserved.
    AllocationFailed(&'static str),
    /// Cumulative token accounting moved backwards.
    TokenOrder {
        /// Prior cumulative token counter.
        previous_tokens: u64,
        /// New cumulative token counter.
        current_tokens: u64,
    },
    /// Early-stop points did not advance both tokens and GPU seconds.
    EvaluationOrder,
    /// A content digest used the reserved all-zero sentinel.
    InvalidEvidenceDigest,
    /// A durable campaign receipt is malformed or non-canonical.
    InvalidCampaignReceipt(&'static str),
    /// A durable campaign receipt uses an unknown encoding version.
    UnsupportedCampaignReceiptVersion(u16),
    /// Durable promotion evidence is malformed or non-canonical.
    InvalidPromotionEvidence(&'static str),
    /// Durable promotion evidence uses an unknown encoding version.
    UnsupportedPromotionEvidenceVersion(u16),
    /// Exact source-model identity used the reserved all-zero sentinel.
    InvalidSourceModelId,
    /// The next frozen recovery track requires this exact predecessor label.
    CampaignPredecessorRequired {
        /// Required activation rung.
        activation: RecoveryActivationRung,
        /// Required predecessor track.
        track: RecoveryTrack,
    },
    /// The A16/PTQ root was supplied a predecessor.
    UnexpectedCampaignPredecessor,
    /// A predecessor exists but belongs to the wrong point in the frozen chain.
    CampaignPredecessorLabelMismatch {
        /// Required predecessor activation rung.
        expected_activation: RecoveryActivationRung,
        /// Required predecessor track.
        expected_track: RecoveryTrack,
        /// Supplied predecessor activation rung.
        actual_activation: RecoveryActivationRung,
        /// Supplied predecessor track.
        actual_track: RecoveryTrack,
    },
    /// A predecessor belongs to a different model-scale proof.
    CampaignModelRungMismatch {
        /// Model rung being authorized.
        expected: RecoveryModelRung,
        /// Model rung recorded by the predecessor.
        actual: RecoveryModelRung,
    },
    /// A predecessor belongs to a different exact source checkpoint/model.
    CampaignSourceModelMismatch,
    /// A predecessor belongs to a different frozen campaign identity.
    CampaignDigestMismatch,
    /// Reopened predecessor evidence did not match its recorded digest.
    CampaignEvidenceDigestMismatch,
    /// Reopened promotion evidence did not match its recorded digest.
    CampaignPromotionEvidenceDigestMismatch,
    /// Selected checkpoint and supplied promotion measurement do not match.
    CampaignPromotionEvidenceMismatch,
    /// Predecessor completed but did not pass its preregistered promotion gate.
    CampaignPredecessorGateNotPassed,
    /// A PTQ track was sent to the refinement evaluation API.
    CampaignTrackHasNoTokenSchedule,
    /// A refinement track was sent to the PTQ completion API.
    CampaignTrackRequiresEvaluations,
    /// A frozen checkpoint was skipped, repeated, or supplied early.
    CampaignEvaluationOrder {
        /// Required cumulative token count.
        expected_tokens: u64,
        /// Supplied cumulative token count.
        actual_tokens: u64,
    },
    /// Cumulative tokens exceeded the exact refinement cap.
    CampaignTokenCapExceeded {
        /// Track token cap.
        token_cap: u64,
        /// Supplied cumulative token count.
        actual_tokens: u64,
    },
    /// An observation was supplied after the track became terminal.
    CampaignEvaluationAfterTerminal,
    /// Receipt export was attempted before the track became terminal.
    CampaignNotTerminal,
    /// Internal campaign state had no predecessor or accepted PTQ checkpoint.
    CampaignMissingBestCheckpoint,
}

impl core::fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidSchedule(reason) => write!(f, "invalid recovery schedule: {reason}"),
            Self::InvalidConfiguration(reason) => {
                write!(f, "invalid recovery configuration: {reason}")
            }
            Self::NonFiniteMetric(metric) => write!(f, "{metric} must be finite"),
            Self::InvalidMetric(metric) => write!(f, "invalid recovery metric: {metric}"),
            Self::StepOutOfRange { step, total_steps } => write!(
                f,
                "recovery step {step} is outside zero-based schedule of {total_steps} steps"
            ),
            Self::ObservationOrder {
                previous_step,
                current_step,
            } => write!(
                f,
                "observation step {current_step} does not follow prior step {previous_step}"
            ),
            Self::StepOrder {
                expected_step,
                actual_step,
            } => write!(
                f,
                "recovery step {actual_step} supplied; expected exhaustive step {expected_step}"
            ),
            Self::BypassAllowanceExceeded { step } => {
                write!(f, "full-precision bypass exceeds allowance at step {step}")
            }
            Self::BypassIncreased { step } => {
                write!(f, "full-precision bypass increased at step {step}")
            }
            Self::IncompleteRun {
                completed_steps,
                total_steps,
            } => write!(
                f,
                "cannot export incomplete recovery run: {completed_steps}/{total_steps} steps"
            ),
            Self::NonZeroBypassAtExport => {
                f.write_str("cannot export while full-precision bypass is non-zero")
            }
            Self::ArithmeticOverflow(counter) => {
                write!(f, "recovery accounting overflow: {counter}")
            }
            Self::AllocationFailed(allocation) => {
                write!(f, "recovery allocation failed: {allocation}")
            }
            Self::TokenOrder {
                previous_tokens,
                current_tokens,
            } => write!(
                f,
                "cumulative tokens decreased from {previous_tokens} to {current_tokens}"
            ),
            Self::EvaluationOrder => f.write_str(
                "early-stop points must increase both cumulative tokens and GPU seconds",
            ),
            Self::InvalidEvidenceDigest => {
                f.write_str("recovery evidence digest cannot be all zero")
            }
            Self::InvalidCampaignReceipt(reason) => {
                write!(f, "invalid recovery campaign receipt: {reason}")
            }
            Self::UnsupportedCampaignReceiptVersion(version) => {
                write!(f, "unsupported recovery campaign receipt version {version}")
            }
            Self::InvalidPromotionEvidence(reason) => {
                write!(f, "invalid recovery promotion evidence: {reason}")
            }
            Self::UnsupportedPromotionEvidenceVersion(version) => {
                write!(
                    f,
                    "unsupported recovery promotion evidence version {version}"
                )
            }
            Self::InvalidSourceModelId => {
                f.write_str("recovery source-model identity cannot be all zero")
            }
            Self::CampaignPredecessorRequired { activation, track } => write!(
                f,
                "recovery campaign requires predecessor {activation:?}/{track:?}"
            ),
            Self::UnexpectedCampaignPredecessor => {
                f.write_str("A16/PTQ is the recovery campaign root and accepts no predecessor")
            }
            Self::CampaignPredecessorLabelMismatch {
                expected_activation,
                expected_track,
                actual_activation,
                actual_track,
            } => write!(
                f,
                "recovery predecessor {actual_activation:?}/{actual_track:?} does not match required {expected_activation:?}/{expected_track:?}"
            ),
            Self::CampaignModelRungMismatch { expected, actual } => write!(
                f,
                "recovery predecessor model rung {actual:?} does not match {expected:?}"
            ),
            Self::CampaignSourceModelMismatch => {
                f.write_str("recovery predecessor belongs to a different exact source model")
            }
            Self::CampaignDigestMismatch => {
                f.write_str("recovery predecessor belongs to a different campaign digest")
            }
            Self::CampaignEvidenceDigestMismatch => {
                f.write_str("reopened recovery predecessor digest does not match its receipt")
            }
            Self::CampaignPromotionEvidenceDigestMismatch => f.write_str(
                "reopened recovery promotion evidence digest does not match its receipt",
            ),
            Self::CampaignPromotionEvidenceMismatch => {
                f.write_str("recovery promotion evidence does not match the selected checkpoint")
            }
            Self::CampaignPredecessorGateNotPassed => {
                f.write_str("recovery predecessor completed but did not pass its promotion gate")
            }
            Self::CampaignTrackHasNoTokenSchedule => {
                f.write_str("PTQ has no frozen refinement-token schedule")
            }
            Self::CampaignTrackRequiresEvaluations => {
                f.write_str("scale-only and PV tracks require frozen checkpoint evaluations")
            }
            Self::CampaignEvaluationOrder {
                expected_tokens,
                actual_tokens,
            } => write!(
                f,
                "campaign evaluation supplied at {actual_tokens} tokens; expected {expected_tokens}"
            ),
            Self::CampaignTokenCapExceeded {
                token_cap,
                actual_tokens,
            } => write!(
                f,
                "campaign evaluation at {actual_tokens} tokens exceeds cap {token_cap}"
            ),
            Self::CampaignEvaluationAfterTerminal => {
                f.write_str("recovery campaign track is already terminal")
            }
            Self::CampaignNotTerminal => {
                f.write_str("cannot export a non-terminal recovery campaign receipt")
            }
            Self::CampaignMissingBestCheckpoint => {
                f.write_str("recovery campaign has no retained checkpoint")
            }
        }
    }
}

impl std::error::Error for RecoveryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_switches_from_soft_to_hard_at_exactly_eighty_percent() {
        let schedule = RecoverySchedule::new(100).expect("valid schedule");

        assert_eq!(schedule.phase_at(0), Ok(RecoveryPhase::Soft));
        assert_eq!(schedule.phase_at(79), Ok(RecoveryPhase::Soft));
        assert_eq!(schedule.phase_at(80), Ok(RecoveryPhase::Hard));
        assert_eq!(schedule.phase_at(99), Ok(RecoveryPhase::Hard));
        assert_eq!(
            schedule.phase_at(100),
            Err(RecoveryError::StepOutOfRange {
                step: 100,
                total_steps: 100,
            })
        );
    }

    #[test]
    fn cached_logit_kd_activates_only_after_configured_plateau() {
        let schedule = RecoverySchedule::new(100).expect("schedule");
        let plateau = PlateauConfig::new(10, 3, 0.02).expect("plateau config");
        let bypass = BypassSchedule::linear(schedule, 0.05, 99).expect("bypass schedule");
        let hidden = HiddenCosineTerm::new(vec![3, 11], 0.05).expect("hidden term");
        let mut policy =
            RecoveryPolicy::new(schedule, plateau, 0.4, Some(hidden), bypass).expect("policy");

        assert_eq!(
            policy.directive(10).expect("directive").loss_mode(),
            LossMode::CrossEntropyOnly
        );
        assert!(!policy.directive(79).expect("soft directive").pv_polish());
        policy
            .observe_held_out_nll(10, 2.00)
            .expect("first validation");
        policy
            .observe_held_out_nll(20, 1.99)
            .expect("second validation");
        assert_eq!(
            policy.directive(20).expect("directive").loss_mode(),
            LossMode::CrossEntropyOnly
        );
        policy
            .observe_held_out_nll(30, 1.98)
            .expect("plateau validation");

        assert_eq!(
            policy.directive(30).expect("directive").loss_mode(),
            LossMode::CrossEntropyWithCachedLogitKd { kd_weight: 0.4 }
        );
        let directive = policy.directive(80).expect("hard-tail directive");
        assert_eq!(directive.phase(), RecoveryPhase::Hard);
        assert!(directive.pv_polish());
        assert_eq!(directive.hidden_cosine_weight(), Some(0.05));
        assert_eq!(policy.hidden_cosine_layers(), Some(&[3_u32, 11][..]));
    }

    #[test]
    fn export_requires_zero_bypass_and_receipts_flagged_usage() {
        let schedule = RecoverySchedule::new(5).expect("schedule");
        let plateau = PlateauConfig::new(0, 2, 0.01).expect("plateau");
        let bypass = BypassSchedule::linear(schedule, 0.02, 4).expect("bypass");
        let policy = RecoveryPolicy::new(schedule, plateau, 0.5, None, bypass).expect("policy");
        let mut run = RecoveryRun::new(policy);

        for (step, bypass_fraction) in [(0, 0.02), (1, 0.014), (2, 0.009), (3, 0.004)] {
            run.record_step(StepObservation::new(step, 10, 1.5, bypass_fraction, None))
                .expect("valid soft step");
        }
        let metrics = FinalRecoveryMetrics::new(1.0, 1.01, 0.03, 0.002).expect("metrics");
        assert_eq!(
            run.export_receipt(metrics),
            Err(RecoveryError::IncompleteRun {
                completed_steps: 4,
                total_steps: 5,
            })
        );
        assert_eq!(
            run.record_step(StepObservation::new(4, 10, 1.5, 0.001, None)),
            Err(RecoveryError::BypassAllowanceExceeded { step: 4 })
        );
        run.record_step(StepObservation::new(4, 10, 1.5, 0.0, None))
            .expect("zero-bypass final step");

        let receipt = run.export_receipt(metrics).expect("export receipt");
        assert_eq!(receipt.tokens_processed(), 50);
        assert_eq!(receipt.gpu_seconds(), 7.5);
        assert!((receipt.hard_minus_soft_nll() - 0.01).abs() < 1e-12);
        assert_eq!(receipt.code_churn(), 0.03);
        assert_eq!(receipt.packed_minus_training_nll(), 0.002);
        assert_eq!(receipt.bypass_steps_used(), 4);
        assert!(receipt.bypass_usage_flagged());
        assert!(receipt.exported_with_zero_bypass());
    }

    #[test]
    fn hard_tail_forces_bypass_allowance_to_zero() {
        let schedule = RecoverySchedule::new(100).expect("schedule");
        let plateau = PlateauConfig::new(0, 2, 0.01).expect("plateau");
        let bypass = BypassSchedule::linear(schedule, 0.05, 99).expect("bypass");
        let policy = RecoveryPolicy::new(schedule, plateau, 0.5, None, bypass).expect("policy");

        assert!(
            policy
                .directive(79)
                .expect("last soft step")
                .maximum_bypass_fraction()
                > 0.0
        );
        assert_eq!(
            policy
                .directive(80)
                .expect("first hard step")
                .maximum_bypass_fraction(),
            0.0
        );
        assert_eq!(
            policy
                .directive(99)
                .expect("last hard step")
                .maximum_bypass_fraction(),
            0.0
        );
    }

    #[test]
    fn bypass_audit_counts_steps_with_any_use_not_per_step_fraction() {
        fn run_with_used_steps(used_steps: u64) -> RecoveryReceipt {
            let schedule = RecoverySchedule::new(100).expect("schedule");
            let plateau = PlateauConfig::new(0, 2, 0.01).expect("plateau");
            let bypass = BypassSchedule::linear(schedule, 0.005, 79).expect("bypass");
            let policy = RecoveryPolicy::new(schedule, plateau, 0.5, None, bypass).expect("policy");
            let mut run = RecoveryRun::new(policy);
            for step in 0..100 {
                let bypass_fraction = if step < used_steps { 0.001 } else { 0.0 };
                run.record_step(StepObservation::new(step, 1, 1.0, bypass_fraction, None))
                    .expect("scheduled step");
            }
            run.export_receipt(FinalRecoveryMetrics::new(1.0, 1.0, 0.0, 0.0).expect("metrics"))
                .expect("receipt")
        }

        let exactly_one_percent = run_with_used_steps(1);
        assert_eq!(exactly_one_percent.bypass_steps_used(), 1);
        assert!(!exactly_one_percent.bypass_usage_flagged());

        let more_than_one_percent = run_with_used_steps(2);
        assert_eq!(more_than_one_percent.bypass_steps_used(), 2);
        assert!(more_than_one_percent.bypass_usage_flagged());
    }

    #[test]
    fn promotion_milestones_and_quality_decisions_are_deterministic() {
        assert_eq!(
            PromotionCheckpoint::crossed(99_999_999, 1_000_000_000).expect("ordered tokens"),
            vec![
                PromotionCheckpoint::Tokens100M,
                PromotionCheckpoint::Tokens500M,
                PromotionCheckpoint::Tokens1B,
            ]
        );
        assert_eq!(PromotionCheckpoint::Tokens10B.tokens(), 10_000_000_000);

        let gate = PromotionGate::new(0.01, 0.005, 0.05, 0.002).expect("gate");
        let passing = PromotionEvidence::new(0.009, 0.004, 0.04, 0.001).expect("evidence");
        let failing = PromotionEvidence::new(0.011, 0.004, 0.04, 0.001).expect("evidence");

        assert_eq!(gate.decide(passing), PromotionDecision::Promote);
        assert_eq!(gate.decide(passing), PromotionDecision::Promote);
        assert_eq!(gate.decide(failing), PromotionDecision::Hold);
    }

    #[test]
    fn early_stop_uses_marginal_held_out_nll_improvement_per_gpu_hour() {
        let gate = EarlyStopGate::new(1.0, 0.01).expect("gate");
        let start = EarlyStopPoint::new(100_000_000, 0, 2.0).expect("start");
        let too_early = EarlyStopPoint::new(110_000_000, 1_800, 1.99).expect("point");
        let stalled = EarlyStopPoint::new(120_000_000, 3_600, 1.995).expect("point");
        let improving = EarlyStopPoint::new(120_000_000, 3_600, 1.98).expect("point");

        assert_eq!(
            gate.decide(start, too_early).expect("ordered points"),
            EarlyStopDecision::Continue
        );
        assert_eq!(
            gate.decide(start, stalled).expect("ordered points"),
            EarlyStopDecision::Stop
        );
        assert_eq!(
            gate.decide(start, stalled).expect("deterministic replay"),
            EarlyStopDecision::Stop
        );
        assert_eq!(
            gate.decide(start, improving).expect("ordered points"),
            EarlyStopDecision::Continue
        );
    }

    #[test]
    fn invalid_schedules_and_non_monotone_bypass_fail_closed() {
        assert!(RecoverySchedule::new(0).is_err());
        assert!(RecoverySchedule::new(99).is_err());
        assert!(PlateauConfig::new(0, 1, 0.0).is_err());
        assert!(PlateauConfig::new(0, 2, f64::NAN).is_err());
        assert!(HiddenCosineTerm::new(vec![7, 7], 0.1).is_err());

        let schedule = RecoverySchedule::new(5).expect("schedule");
        let oversized_plateau =
            PlateauConfig::new(0, usize::MAX, 0.01).expect("shape is cross-validated by policy");
        let oversized_bypass = BypassSchedule::linear(schedule, 0.1, 4).expect("bypass");
        assert!(matches!(
            RecoveryPolicy::new(schedule, oversized_plateau, 0.5, None, oversized_bypass),
            Err(RecoveryError::InvalidConfiguration(
                "plateau validation_window cannot exceed total recovery steps"
            )) | Err(RecoveryError::InvalidConfiguration(
                "plateau validation_window must fit the recovery schedule domain"
            ))
        ));

        let plateau = PlateauConfig::new(0, 2, 0.01).expect("plateau");
        let bypass = BypassSchedule::linear(schedule, 0.1, 4).expect("bypass");
        let policy = RecoveryPolicy::new(schedule, plateau, 0.5, None, bypass).expect("policy");
        let mut run = RecoveryRun::new(policy);
        run.record_step(StepObservation::new(0, 10, 1.0, 0.05, None))
            .expect("first step");
        assert_eq!(
            run.record_step(StepObservation::new(1, 10, 1.0, 0.06, None)),
            Err(RecoveryError::BypassIncreased { step: 1 })
        );
    }

    #[test]
    fn frozen_campaign_caps_and_evaluation_checkpoints_are_exact() {
        for (rung, scale_cap, pv_cap) in [
            (RecoveryModelRung::Pilot, 8_000_000, 32_000_000),
            (RecoveryModelRung::Qwen8B, 32_000_000, 256_000_000),
            (RecoveryModelRung::Qwen32B, 64_000_000, 512_000_000),
        ] {
            assert_eq!(rung.token_cap(RecoveryTrack::Ptq), None);
            assert_eq!(rung.token_cap(RecoveryTrack::ScaleOnly), Some(scale_cap));
            assert_eq!(rung.token_cap(RecoveryTrack::Pv), Some(pv_cap));
            assert_eq!(
                RecoveryEvaluationCheckpoint::ALL.map(|checkpoint| checkpoint.tokens(pv_cap)),
                [pv_cap / 8, pv_cap / 4, pv_cap / 2, pv_cap]
            );
        }
    }

    fn campaign_digest(byte: u8) -> RecoveryEvidenceDigest {
        RecoveryEvidenceDigest::new([byte; 32]).expect("non-zero digest")
    }

    fn source_model(model: RecoveryModelRung) -> RecoverySourceModel {
        match model {
            RecoveryModelRung::Pilot => RecoverySourceModel::SmolLm2OneThirtyFiveMillion,
            RecoveryModelRung::Qwen8B => RecoverySourceModel::Qwen3EightBillion,
            RecoveryModelRung::Qwen32B => RecoverySourceModel::Qwen3ThirtyTwoBillion,
        }
    }

    fn source_id(model: RecoveryModelRung) -> RecoverySourceModelId {
        let seed = match model {
            RecoveryModelRung::Pilot => 201,
            RecoveryModelRung::Qwen8B => 202,
            RecoveryModelRung::Qwen32B => 203,
        };
        RecoverySourceModelId::new([seed; 32]).expect("source identity")
    }

    fn campaign_plan(
        model: RecoveryModelRung,
        activation: RecoveryActivationRung,
        track: RecoveryTrack,
        campaign: RecoveryEvidenceDigest,
    ) -> RecoveryCampaignPlan {
        RecoveryCampaignPlan::new(
            source_model(model),
            source_id(model),
            activation,
            track,
            campaign,
            RecoveryPromotionGate::new(0.0, campaign).expect("promotion gate"),
        )
    }

    fn predecessor_evidence(receipt: &RecoveryCampaignReceipt) -> RecoveryPredecessorEvidence {
        let promotion_evidence_bytes = receipt
            .promotion_evidence
            .to_bytes()
            .expect("serialize promotion evidence");
        RecoveryPredecessorEvidence::new(
            &receipt.to_bytes().expect("serialize predecessor"),
            &promotion_evidence_bytes,
        )
        .expect("reopened predecessor")
    }

    fn finish_campaign(run: RecoveryCampaignRun) -> Result<RecoveryCampaignReceipt, RecoveryError> {
        let promotion_evidence = run.prepare_promotion_evidence()?;
        let promotion_evidence_bytes = promotion_evidence.to_bytes()?;
        run.finish(&promotion_evidence_bytes)
    }

    fn resign_promotion_evidence(mut evidence: RecoveryPromotionEvidence) -> Vec<u8> {
        evidence.evidence_digest = promotion_evidence_content_digest(
            &evidence
                .canonical_content_bytes()
                .expect("canonical promotion evidence"),
        )
        .expect("promotion evidence digest");
        evidence.to_bytes().expect("serialize promotion evidence")
    }

    fn resign_receipt(mut receipt: RecoveryCampaignReceipt) -> Vec<u8> {
        let canonical = receipt
            .canonical_content_bytes()
            .expect("canonical forged receipt");
        receipt.receipt_digest = receipt_content_digest(&canonical).expect("receipt digest");
        let mut bytes = canonical;
        bytes.extend_from_slice(&receipt.receipt_digest.as_bytes());
        bytes
    }

    fn complete_campaign_ptq(
        model: RecoveryModelRung,
        activation: RecoveryActivationRung,
        campaign: RecoveryEvidenceDigest,
        predecessor: Option<&RecoveryCampaignReceipt>,
        digest_seed: u8,
    ) -> RecoveryCampaignReceipt {
        let predecessor = predecessor.map(predecessor_evidence);
        let mut run = RecoveryCampaignRun::start(
            campaign_plan(model, activation, RecoveryTrack::Ptq, campaign),
            predecessor,
        )
        .expect("authorized PTQ");
        let artifact = campaign_digest(digest_seed);
        run.record_ptq(1.0, artifact).expect("PTQ measurement");
        finish_campaign(run).expect("PTQ receipt")
    }

    fn complete_campaign_refinement(
        model: RecoveryModelRung,
        activation: RecoveryActivationRung,
        track: RecoveryTrack,
        campaign: RecoveryEvidenceDigest,
        predecessor: &RecoveryCampaignReceipt,
        digest_seed: u8,
    ) -> RecoveryCampaignReceipt {
        let evidence = predecessor_evidence(predecessor);
        let mut run = RecoveryCampaignRun::start(
            campaign_plan(model, activation, track, campaign),
            Some(evidence),
        )
        .expect("authorized refinement");
        let cap = model.token_cap(track).expect("refinement cap");
        for (index, checkpoint) in RecoveryEvaluationCheckpoint::ALL.into_iter().enumerate() {
            let decision = run
                .record_evaluation(
                    checkpoint.tokens(cap),
                    predecessor.selected_validation_quality() + index as f64 + 1.0,
                    campaign_digest(digest_seed + index as u8),
                )
                .expect("ordered evaluation");
            if checkpoint == RecoveryEvaluationCheckpoint::Full {
                assert_eq!(
                    decision,
                    RecoveryCampaignDecision::Complete(
                        RecoveryCampaignTermination::TokenCapReached
                    )
                );
            } else {
                assert_eq!(decision, RecoveryCampaignDecision::Continue);
            }
        }
        let quality = predecessor.selected_validation_quality() + 4.0;
        let artifact = campaign_digest(digest_seed + 3);
        assert_eq!(
            quality,
            run.best.expect("best checkpoint").validation_quality
        );
        assert_eq!(artifact, run.best.expect("best checkpoint").artifact_digest);
        finish_campaign(run).expect("refinement receipt")
    }

    #[test]
    fn campaign_authorization_enforces_ptq_then_scale_then_pv() {
        let campaign = campaign_digest(1);
        let mut ptq = RecoveryCampaignRun::start(
            campaign_plan(
                RecoveryModelRung::Pilot,
                RecoveryActivationRung::A16,
                RecoveryTrack::Ptq,
                campaign,
            ),
            None,
        )
        .expect("A16 PTQ is the only root");
        let ptq_artifact = campaign_digest(2);
        assert_eq!(
            ptq.record_ptq(0.8, ptq_artifact),
            Ok(RecoveryCampaignDecision::Complete(
                RecoveryCampaignTermination::PtqComplete
            ))
        );
        let ptq_receipt = finish_campaign(ptq).expect("PTQ receipt");
        let ptq_evidence = predecessor_evidence(&ptq_receipt);

        assert!(matches!(
            RecoveryCampaignRun::start(
                campaign_plan(
                    RecoveryModelRung::Pilot,
                    RecoveryActivationRung::A16,
                    RecoveryTrack::ScaleOnly,
                    campaign,
                ),
                None,
            ),
            Err(RecoveryError::CampaignPredecessorRequired {
                activation: RecoveryActivationRung::A16,
                track: RecoveryTrack::Ptq,
            })
        ));
        RecoveryCampaignRun::start(
            campaign_plan(
                RecoveryModelRung::Pilot,
                RecoveryActivationRung::A16,
                RecoveryTrack::ScaleOnly,
                campaign,
            ),
            Some(ptq_evidence),
        )
        .expect("same-rung PTQ authorizes scale-only");

        assert!(matches!(
            RecoveryCampaignRun::start(
                campaign_plan(
                    RecoveryModelRung::Pilot,
                    RecoveryActivationRung::A16,
                    RecoveryTrack::Pv,
                    campaign,
                ),
                Some(ptq_evidence),
            ),
            Err(RecoveryError::CampaignPredecessorLabelMismatch {
                expected_activation: RecoveryActivationRung::A16,
                expected_track: RecoveryTrack::ScaleOnly,
                actual_activation: RecoveryActivationRung::A16,
                actual_track: RecoveryTrack::Ptq,
            })
        ));
        assert!(matches!(
            RecoveryCampaignRun::start(
                campaign_plan(
                    RecoveryModelRung::Pilot,
                    RecoveryActivationRung::A8,
                    RecoveryTrack::Ptq,
                    campaign,
                ),
                Some(ptq_evidence),
            ),
            Err(RecoveryError::CampaignPredecessorLabelMismatch {
                expected_activation: RecoveryActivationRung::A16,
                expected_track: RecoveryTrack::Pv,
                actual_activation: RecoveryActivationRung::A16,
                actual_track: RecoveryTrack::Ptq,
            })
        ));
    }

    #[test]
    fn complete_activation_chain_authorizes_a16_then_a8_then_a4() {
        let campaign = campaign_digest(40);
        let a16_ptq = complete_campaign_ptq(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A16,
            campaign,
            None,
            41,
        );
        let a16_scale = complete_campaign_refinement(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A16,
            RecoveryTrack::ScaleOnly,
            campaign,
            &a16_ptq,
            43,
        );
        let a16_pv = complete_campaign_refinement(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A16,
            RecoveryTrack::Pv,
            campaign,
            &a16_scale,
            48,
        );
        let a8_ptq = complete_campaign_ptq(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A8,
            campaign,
            Some(&a16_pv),
            53,
        );
        let a8_scale = complete_campaign_refinement(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A8,
            RecoveryTrack::ScaleOnly,
            campaign,
            &a8_ptq,
            55,
        );
        let a8_pv = complete_campaign_refinement(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A8,
            RecoveryTrack::Pv,
            campaign,
            &a8_scale,
            60,
        );
        let a4_ptq = complete_campaign_ptq(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A4,
            campaign,
            Some(&a8_pv),
            65,
        );
        let a4_scale = complete_campaign_refinement(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A4,
            RecoveryTrack::ScaleOnly,
            campaign,
            &a4_ptq,
            67,
        );
        let a4_pv = complete_campaign_refinement(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A4,
            RecoveryTrack::Pv,
            campaign,
            &a4_scale,
            72,
        );

        assert_eq!(a4_pv.activation_rung(), RecoveryActivationRung::A4);
        assert_eq!(a4_pv.track(), RecoveryTrack::Pv);
        assert_eq!(
            a4_pv.predecessor_receipt_digest(),
            Some(a4_scale.receipt_digest())
        );
        assert_eq!(a4_pv.tokens_consumed(), 32_000_000);
        assert_eq!(a4_pv.evaluations_completed(), 4);
    }

    #[test]
    fn campaign_stops_after_three_non_improvements_and_retains_best_prior_point() {
        let campaign = campaign_digest(10);
        let baseline_artifact = campaign_digest(11);
        let mut ptq = RecoveryCampaignRun::start(
            campaign_plan(
                RecoveryModelRung::Pilot,
                RecoveryActivationRung::A16,
                RecoveryTrack::Ptq,
                campaign,
            ),
            None,
        )
        .expect("root PTQ");
        ptq.record_ptq(0.8, baseline_artifact).expect("PTQ result");
        let ptq_receipt = finish_campaign(ptq).expect("PTQ receipt");
        let predecessor = predecessor_evidence(&ptq_receipt);
        let mut scale = RecoveryCampaignRun::start(
            campaign_plan(
                RecoveryModelRung::Pilot,
                RecoveryActivationRung::A16,
                RecoveryTrack::ScaleOnly,
                campaign,
            ),
            Some(predecessor),
        )
        .expect("authorized scale-only run");

        assert_eq!(
            scale.record_evaluation(1_000_000, 0.8, campaign_digest(13)),
            Ok(RecoveryCampaignDecision::Continue)
        );
        assert_eq!(
            scale.record_evaluation(2_000_000, 0.79, campaign_digest(14)),
            Ok(RecoveryCampaignDecision::Continue)
        );
        assert_eq!(
            scale.record_evaluation(4_000_000, 0.7, campaign_digest(15)),
            Ok(RecoveryCampaignDecision::Complete(
                RecoveryCampaignTermination::ThreeEvaluationsWithoutImprovement
            ))
        );
        assert_eq!(
            scale.record_evaluation(8_000_000, 0.9, campaign_digest(16)),
            Err(RecoveryError::CampaignEvaluationAfterTerminal)
        );

        let receipt = finish_campaign(scale).expect("early-stop receipt");
        assert_eq!(receipt.token_cap(), Some(8_000_000));
        assert_eq!(receipt.tokens_consumed(), 4_000_000);
        assert_eq!(receipt.evaluations_completed(), 3);
        assert_eq!(
            receipt.termination(),
            RecoveryCampaignTermination::ThreeEvaluationsWithoutImprovement
        );
        assert_eq!(
            receipt.selected_checkpoint(),
            RecoverySelectedCheckpoint::Predecessor
        );
        assert_eq!(receipt.selected_validation_quality(), 0.8);
        assert_eq!(receipt.selected_artifact_digest(), baseline_artifact);
        assert_eq!(
            receipt.predecessor_receipt_digest(),
            Some(ptq_receipt.receipt_digest())
        );
    }

    #[test]
    fn campaign_rejects_malformed_cross_rung_and_cross_digest_evidence() {
        assert_eq!(
            RecoveryEvidenceDigest::new([0; 32]),
            Err(RecoveryError::InvalidEvidenceDigest)
        );

        let campaign = campaign_digest(20);
        let mut ptq = RecoveryCampaignRun::start(
            campaign_plan(
                RecoveryModelRung::Pilot,
                RecoveryActivationRung::A16,
                RecoveryTrack::Ptq,
                campaign,
            ),
            None,
        )
        .expect("root PTQ");
        assert_eq!(
            ptq.record_ptq(f64::NAN, campaign_digest(21)),
            Err(RecoveryError::NonFiniteMetric(
                "campaign validation quality"
            ))
        );
        assert_eq!(
            finish_campaign(ptq.clone()),
            Err(RecoveryError::CampaignNotTerminal)
        );
        ptq.record_ptq(1.0, campaign_digest(21)).expect("valid PTQ");
        let receipt = finish_campaign(ptq).expect("PTQ receipt");

        let mut corrupted_receipt = receipt.to_bytes().expect("serialize receipt");
        let digest_byte = corrupted_receipt
            .last_mut()
            .expect("receipt includes content digest");
        *digest_byte ^= 1;
        let promotion_evidence_bytes = receipt
            .promotion_evidence
            .to_bytes()
            .expect("serialize promotion evidence");
        assert_eq!(
            RecoveryPredecessorEvidence::new(&corrupted_receipt, &promotion_evidence_bytes),
            Err(RecoveryError::CampaignEvidenceDigestMismatch)
        );
        let predecessor = predecessor_evidence(&receipt);
        assert!(matches!(
            RecoveryCampaignRun::start(
                campaign_plan(
                    RecoveryModelRung::Qwen8B,
                    RecoveryActivationRung::A16,
                    RecoveryTrack::ScaleOnly,
                    campaign,
                ),
                Some(predecessor),
            ),
            Err(RecoveryError::CampaignModelRungMismatch {
                expected: RecoveryModelRung::Qwen8B,
                actual: RecoveryModelRung::Pilot,
            })
        ));
        assert_eq!(
            RecoveryCampaignRun::start(
                campaign_plan(
                    RecoveryModelRung::Pilot,
                    RecoveryActivationRung::A16,
                    RecoveryTrack::ScaleOnly,
                    campaign_digest(24),
                ),
                Some(predecessor),
            ),
            Err(RecoveryError::CampaignDigestMismatch)
        );
        assert_eq!(
            RecoveryCampaignRun::start(
                campaign_plan(
                    RecoveryModelRung::Pilot,
                    RecoveryActivationRung::A16,
                    RecoveryTrack::Ptq,
                    campaign,
                ),
                Some(predecessor),
            ),
            Err(RecoveryError::UnexpectedCampaignPredecessor)
        );
    }

    #[test]
    fn campaign_receipt_reopen_rejects_mutation_truncation_trailing_version_and_digest_cycles() {
        let campaign = campaign_digest(101);
        let receipt = complete_campaign_ptq(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A16,
            campaign,
            None,
            102,
        );
        let bytes = receipt.to_bytes().expect("canonical receipt");
        assert_eq!(bytes.len(), RECOVERY_RECEIPT_BYTES);
        assert_eq!(
            RecoveryCampaignReceipt::reopen(&bytes).expect("verified reopen"),
            receipt
        );

        let mut mutated_content = bytes.clone();
        mutated_content[11] ^= 1;
        assert_eq!(
            RecoveryCampaignReceipt::reopen(&mutated_content),
            Err(RecoveryError::CampaignEvidenceDigestMismatch)
        );

        assert_eq!(
            RecoveryCampaignReceipt::reopen(&bytes[..bytes.len() - 1]),
            Err(RecoveryError::InvalidCampaignReceipt("truncated receipt"))
        );

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            RecoveryCampaignReceipt::reopen(&trailing),
            Err(RecoveryError::InvalidCampaignReceipt(
                "trailing receipt bytes"
            ))
        );

        let mut future_version = bytes.clone();
        future_version[8..10].copy_from_slice(&3_u16.to_le_bytes());
        assert_eq!(
            RecoveryCampaignReceipt::reopen(&future_version),
            Err(RecoveryError::UnsupportedCampaignReceiptVersion(3))
        );

        let mut legacy_caller_digest_version = bytes.clone();
        legacy_caller_digest_version[8..10].copy_from_slice(&1_u16.to_le_bytes());
        assert_eq!(
            RecoveryCampaignReceipt::reopen(&legacy_caller_digest_version),
            Err(RecoveryError::UnsupportedCampaignReceiptVersion(1))
        );

        let mut caller_chosen_digest = bytes;
        caller_chosen_digest[RECOVERY_RECEIPT_CONTENT_BYTES..].fill(0x5a);
        assert_eq!(
            RecoveryCampaignReceipt::reopen(&caller_chosen_digest),
            Err(RecoveryError::CampaignEvidenceDigestMismatch)
        );
    }

    #[test]
    fn promotion_evidence_roundtrips_and_rejects_corruption_and_signed_zero() {
        let campaign = campaign_digest(110);
        let gate = RecoveryPromotionGate::new(-0.0, campaign_digest(111)).expect("zero gate");
        assert_eq!(
            gate.minimum_validation_quality().to_bits(),
            0.0_f64.to_bits()
        );
        let plan = RecoveryCampaignPlan::new(
            RecoverySourceModel::SmolLm2OneThirtyFiveMillion,
            source_id(RecoveryModelRung::Pilot),
            RecoveryActivationRung::A16,
            RecoveryTrack::Ptq,
            campaign,
            gate,
        );
        let mut run = RecoveryCampaignRun::start(plan, None).expect("root PTQ");
        run.record_ptq(-0.0, campaign_digest(112))
            .expect("terminal PTQ");
        let evidence = run
            .prepare_promotion_evidence()
            .expect("promotion evidence");
        assert_eq!(evidence.validation_quality.to_bits(), 0.0_f64.to_bits());
        let bytes = evidence.to_bytes().expect("canonical evidence");
        assert_eq!(bytes.len(), RECOVERY_PROMOTION_EVIDENCE_BYTES);
        assert_eq!(
            RecoveryPromotionEvidence::reopen(&bytes).expect("verified evidence"),
            evidence
        );

        let receipt = run.clone().finish(&bytes).expect("receipt");
        assert_eq!(
            receipt.selected_validation_quality().to_bits(),
            0.0_f64.to_bits()
        );
        RecoveryPredecessorEvidence::new(&receipt.to_bytes().expect("receipt bytes"), &bytes)
            .expect("independently reopened evidence");

        let mut mutated = bytes.clone();
        mutated[11] ^= 1;
        assert_eq!(
            RecoveryPromotionEvidence::reopen(&mutated),
            Err(RecoveryError::CampaignPromotionEvidenceDigestMismatch)
        );
        assert_eq!(
            run.finish(&mutated),
            Err(RecoveryError::CampaignPromotionEvidenceDigestMismatch)
        );
        assert_eq!(
            RecoveryPromotionEvidence::reopen(&bytes[..bytes.len() - 1]),
            Err(RecoveryError::InvalidPromotionEvidence(
                "truncated evidence"
            ))
        );
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(
            RecoveryPromotionEvidence::reopen(&trailing),
            Err(RecoveryError::InvalidPromotionEvidence(
                "trailing evidence bytes"
            ))
        );
        let mut future_version = bytes.clone();
        future_version[8..10].copy_from_slice(&2_u16.to_le_bytes());
        assert_eq!(
            RecoveryPromotionEvidence::reopen(&future_version),
            Err(RecoveryError::UnsupportedPromotionEvidenceVersion(2))
        );
        let mut noncanonical_zero = bytes.clone();
        const GATE_THRESHOLD_OFFSET: usize = 77;
        noncanonical_zero[GATE_THRESHOLD_OFFSET..GATE_THRESHOLD_OFFSET + 8]
            .copy_from_slice(&(-0.0_f64).to_bits().to_le_bytes());
        assert_eq!(
            RecoveryPromotionEvidence::reopen(&noncanonical_zero),
            Err(RecoveryError::InvalidPromotionEvidence(
                "non-canonical evidence metric"
            ))
        );
        let mut caller_digest = bytes;
        caller_digest[RECOVERY_PROMOTION_EVIDENCE_CONTENT_BYTES..].fill(0x5a);
        assert_eq!(
            RecoveryPromotionEvidence::reopen(&caller_digest),
            Err(RecoveryError::CampaignPromotionEvidenceDigestMismatch)
        );
    }

    #[test]
    fn finish_and_predecessor_reject_every_resigned_context_forgery() {
        let campaign = campaign_digest(120);
        let plan = campaign_plan(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A16,
            RecoveryTrack::Ptq,
            campaign,
        );
        let mut run = RecoveryCampaignRun::start(plan, None).expect("root PTQ");
        run.record_ptq(0.8, campaign_digest(121))
            .expect("terminal PTQ");
        let original = run
            .prepare_promotion_evidence()
            .expect("promotion evidence");

        let mut forgeries = Vec::new();
        let mut forged = original;
        forged.source_model = RecoverySourceModel::SmolLm2OnePointSevenBillion;
        forgeries.push(resign_promotion_evidence(forged));
        let mut forged = original;
        forged.source_model_id = RecoverySourceModelId::new([122; 32]).expect("other source");
        forgeries.push(resign_promotion_evidence(forged));
        let mut forged = original;
        forged.activation_rung = RecoveryActivationRung::A8;
        forgeries.push(resign_promotion_evidence(forged));
        let mut forged = original;
        forged.track = RecoveryTrack::ScaleOnly;
        forgeries.push(resign_promotion_evidence(forged));
        let mut forged = original;
        forged.campaign_digest = campaign_digest(123);
        forgeries.push(resign_promotion_evidence(forged));
        let mut forged = original;
        forged.promotion_gate =
            RecoveryPromotionGate::new(0.25, original.promotion_gate.gate_digest)
                .expect("other threshold");
        forgeries.push(resign_promotion_evidence(forged));
        let mut forged = original;
        forged.promotion_gate = RecoveryPromotionGate::new(
            original.promotion_gate.minimum_validation_quality,
            campaign_digest(124),
        )
        .expect("other gate");
        forgeries.push(resign_promotion_evidence(forged));
        let mut forged = original;
        forged.selected_checkpoint = RecoverySelectedCheckpoint::Predecessor;
        forgeries.push(resign_promotion_evidence(forged));
        let mut forged = original;
        forged.validation_quality = 0.9;
        forgeries.push(resign_promotion_evidence(forged));
        let mut forged = original;
        forged.artifact_digest = campaign_digest(125);
        forgeries.push(resign_promotion_evidence(forged));

        for (index, forged_bytes) in forgeries.iter().enumerate() {
            assert_eq!(
                run.clone().finish(forged_bytes),
                Err(RecoveryError::CampaignPromotionEvidenceMismatch),
                "finish accepted context forgery {index}"
            );
        }

        let original_bytes = original.to_bytes().expect("original evidence");
        let receipt = run.finish(&original_bytes).expect("receipt");
        let receipt_bytes = receipt.to_bytes().expect("receipt bytes");
        for (index, forged_bytes) in forgeries.iter().enumerate() {
            assert_eq!(
                RecoveryPredecessorEvidence::new(&receipt_bytes, forged_bytes),
                Err(RecoveryError::CampaignPromotionEvidenceMismatch),
                "predecessor accepted context forgery {index}"
            );
        }
    }

    #[test]
    fn refinement_terminal_semantic_matrix_is_exact_and_enforced_on_reopen() {
        let checkpoints = [
            RecoverySelectedCheckpoint::Ptq,
            RecoverySelectedCheckpoint::Predecessor,
            RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneEighth),
            RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneQuarter),
            RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneHalf),
            RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::Full),
        ];
        let terminations = [
            RecoveryCampaignTermination::PtqComplete,
            RecoveryCampaignTermination::ThreeEvaluationsWithoutImprovement,
            RecoveryCampaignTermination::TokenCapReached,
        ];
        let allowed = [
            (
                RecoveryCampaignTermination::ThreeEvaluationsWithoutImprovement,
                3,
                RecoverySelectedCheckpoint::Predecessor,
            ),
            (
                RecoveryCampaignTermination::ThreeEvaluationsWithoutImprovement,
                4,
                RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneEighth),
            ),
            (
                RecoveryCampaignTermination::TokenCapReached,
                4,
                RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneQuarter),
            ),
            (
                RecoveryCampaignTermination::TokenCapReached,
                4,
                RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::OneHalf),
            ),
            (
                RecoveryCampaignTermination::TokenCapReached,
                4,
                RecoverySelectedCheckpoint::Evaluation(RecoveryEvaluationCheckpoint::Full),
            ),
        ];
        for termination in terminations {
            for evaluations in 0..=5 {
                for checkpoint in checkpoints {
                    assert_eq!(
                        valid_refinement_terminal(termination, evaluations, checkpoint),
                        allowed.contains(&(termination, evaluations, checkpoint)),
                        "unexpected terminal tuple {termination:?}/{evaluations}/{checkpoint:?}"
                    );
                }
            }
        }

        let campaign = campaign_digest(130);
        let ptq = complete_campaign_ptq(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A16,
            campaign,
            None,
            131,
        );
        let mut impossible = complete_campaign_refinement(
            RecoveryModelRung::Pilot,
            RecoveryActivationRung::A16,
            RecoveryTrack::ScaleOnly,
            campaign,
            &ptq,
            132,
        );
        impossible.termination = RecoveryCampaignTermination::ThreeEvaluationsWithoutImprovement;
        let forged_bytes = resign_receipt(impossible);
        assert_eq!(
            RecoveryCampaignReceipt::reopen(&forged_bytes),
            Err(RecoveryError::InvalidCampaignReceipt(
                "impossible refinement terminal history"
            ))
        );
    }

    #[test]
    fn campaign_rejects_out_of_order_and_over_cap_evaluations_without_mutation() {
        let campaign = campaign_digest(30);
        let mut ptq = RecoveryCampaignRun::start(
            campaign_plan(
                RecoveryModelRung::Pilot,
                RecoveryActivationRung::A16,
                RecoveryTrack::Ptq,
                campaign,
            ),
            None,
        )
        .expect("root PTQ");
        assert_eq!(
            ptq.record_evaluation(1, 1.0, campaign_digest(31)),
            Err(RecoveryError::CampaignTrackHasNoTokenSchedule)
        );
        ptq.record_ptq(1.0, campaign_digest(31)).expect("PTQ");
        let receipt = finish_campaign(ptq).expect("receipt");
        let predecessor = predecessor_evidence(&receipt);
        let mut scale = RecoveryCampaignRun::start(
            campaign_plan(
                RecoveryModelRung::Pilot,
                RecoveryActivationRung::A16,
                RecoveryTrack::ScaleOnly,
                campaign,
            ),
            Some(predecessor),
        )
        .expect("scale-only");

        assert_eq!(
            scale.record_evaluation(2_000_000, 1.1, campaign_digest(33)),
            Err(RecoveryError::CampaignEvaluationOrder {
                expected_tokens: 1_000_000,
                actual_tokens: 2_000_000,
            })
        );
        assert_eq!(
            scale.record_evaluation(8_000_001, 1.1, campaign_digest(33)),
            Err(RecoveryError::CampaignTokenCapExceeded {
                token_cap: 8_000_000,
                actual_tokens: 8_000_001,
            })
        );
        assert_eq!(
            scale.record_evaluation(1_000_000, f64::INFINITY, campaign_digest(33)),
            Err(RecoveryError::NonFiniteMetric(
                "campaign validation quality"
            ))
        );
        assert_eq!(
            scale.record_evaluation(1_000_000, 1.1, campaign_digest(33)),
            Ok(RecoveryCampaignDecision::Continue)
        );
        assert_eq!(
            scale.record_ptq(1.2, campaign_digest(34)),
            Err(RecoveryError::CampaignTrackRequiresEvaluations)
        );
    }

    #[test]
    fn terminal_bad_quality_receipt_cannot_authorize_the_next_track() {
        let campaign = campaign_digest(80);
        let source_id = RecoverySourceModelId::new([81; 32]).expect("source ID");
        let gate = RecoveryPromotionGate::new(0.9, campaign_digest(82)).expect("gate");
        let ptq_plan = RecoveryCampaignPlan::new(
            RecoverySourceModel::SmolLm2OneThirtyFiveMillion,
            source_id,
            RecoveryActivationRung::A16,
            RecoveryTrack::Ptq,
            campaign,
            gate,
        );
        let mut ptq = RecoveryCampaignRun::start(ptq_plan, None).expect("root PTQ");
        let artifact = campaign_digest(83);
        ptq.record_ptq(0.8, artifact).expect("terminal PTQ");
        let promotion_evidence_bytes = ptq
            .prepare_promotion_evidence()
            .expect("promotion evidence")
            .to_bytes()
            .expect("serialize promotion evidence");
        let receipt = ptq
            .finish(&promotion_evidence_bytes)
            .expect("failed-gate receipt remains durable");
        assert_eq!(
            receipt.promotion_outcome(),
            RecoveryPromotionOutcome::Failed
        );
        let predecessor = RecoveryPredecessorEvidence::new(
            &receipt.to_bytes().expect("serialize failed receipt"),
            &promotion_evidence_bytes,
        )
        .expect("reopened failed receipt");
        let scale_plan = RecoveryCampaignPlan::new(
            RecoverySourceModel::SmolLm2OneThirtyFiveMillion,
            source_id,
            RecoveryActivationRung::A16,
            RecoveryTrack::ScaleOnly,
            campaign,
            gate,
        );
        assert_eq!(
            RecoveryCampaignRun::start(scale_plan, Some(predecessor)),
            Err(RecoveryError::CampaignPredecessorGateNotPassed)
        );
    }

    #[test]
    fn campaign_binding_rejects_missing_tampered_and_same_rung_cross_model_identity() {
        assert_eq!(
            RecoverySourceModelId::new([0; 32]),
            Err(RecoveryError::InvalidSourceModelId)
        );
        let campaign = campaign_digest(90);
        let source_id = RecoverySourceModelId::new([91; 32]).expect("source ID");
        let gate = RecoveryPromotionGate::new(0.5, campaign_digest(92)).expect("gate");
        let plan = RecoveryCampaignPlan::new(
            RecoverySourceModel::SmolLm2OneThirtyFiveMillion,
            source_id,
            RecoveryActivationRung::A16,
            RecoveryTrack::Ptq,
            campaign,
            gate,
        );
        let mut run = RecoveryCampaignRun::start(plan, None).expect("root");
        let artifact = campaign_digest(93);
        run.record_ptq(0.8, artifact).expect("PTQ");
        let mut wrong_artifact_run = RecoveryCampaignRun::start(plan, None).expect("forged root");
        wrong_artifact_run
            .record_ptq(0.8, campaign_digest(95))
            .expect("forged PTQ");
        let wrong_artifact_evidence = wrong_artifact_run
            .prepare_promotion_evidence()
            .expect("forged promotion evidence")
            .to_bytes()
            .expect("serialize forged promotion evidence");
        assert_eq!(
            run.clone().finish(&wrong_artifact_evidence),
            Err(RecoveryError::CampaignPromotionEvidenceMismatch)
        );
        let promotion_evidence_bytes = run
            .prepare_promotion_evidence()
            .expect("promotion evidence")
            .to_bytes()
            .expect("serialize promotion evidence");
        let receipt = run
            .finish(&promotion_evidence_bytes)
            .expect("passed receipt");
        assert_eq!(
            receipt.promotion_outcome(),
            RecoveryPromotionOutcome::Passed
        );
        assert_eq!(
            RecoveryPredecessorEvidence::new(
                &receipt.to_bytes().expect("serialize receipt"),
                &wrong_artifact_evidence,
            ),
            Err(RecoveryError::CampaignPromotionEvidenceMismatch)
        );
        let predecessor = predecessor_evidence(&receipt);
        let different_pilot = RecoveryCampaignPlan::new(
            RecoverySourceModel::SmolLm2OnePointSevenBillion,
            source_id,
            RecoveryActivationRung::A16,
            RecoveryTrack::ScaleOnly,
            campaign,
            gate,
        );
        assert_eq!(
            RecoveryCampaignRun::start(different_pilot, Some(predecessor)),
            Err(RecoveryError::CampaignSourceModelMismatch)
        );
        let different_checkpoint = RecoveryCampaignPlan::new(
            RecoverySourceModel::SmolLm2OneThirtyFiveMillion,
            RecoverySourceModelId::new([98; 32]).expect("other source"),
            RecoveryActivationRung::A16,
            RecoveryTrack::ScaleOnly,
            campaign,
            gate,
        );
        assert_eq!(
            RecoveryCampaignRun::start(different_checkpoint, Some(predecessor)),
            Err(RecoveryError::CampaignSourceModelMismatch)
        );
    }
}
