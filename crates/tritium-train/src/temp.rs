//! Deterministic HESTIA temperature and sensitivity primitives.
//!
//! Plan 0054 freezes an exponential temperature schedule and reuses source-bound
//! S2KF curvature instead of running a second Hessian estimator. This module is
//! pure: it owns no tensors, devices, training state, or evidence files.

/// Failure to construct or evaluate a HESTIA temperature policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TemperatureError {
    /// A schedule or sensitivity parameter lies outside its declared domain.
    InvalidConfiguration(&'static str),
    /// A zero-based optimizer step lies outside the recovery schedule.
    StepOutOfRange {
        /// Supplied step.
        step: u64,
        /// Exclusive recovery-schedule limit.
        total_steps: u64,
    },
    /// A tensor ordinal does not exist in the bound sensitivity profile.
    TensorOutOfRange {
        /// Supplied tensor ordinal.
        tensor: usize,
        /// Number of bound tensor scores.
        tensor_count: usize,
    },
}

impl core::fmt::Display for TemperatureError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfiguration(reason) => {
                write!(formatter, "invalid temperature configuration: {reason}")
            }
            Self::StepOutOfRange { step, total_steps } => write!(
                formatter,
                "temperature step {step} is outside zero-based schedule of {total_steps} steps"
            ),
            Self::TensorOutOfRange {
                tensor,
                tensor_count,
            } => write!(
                formatter,
                "temperature tensor {tensor} is outside profile of {tensor_count} tensors"
            ),
        }
    }
}

impl std::error::Error for TemperatureError {}

/// Exponential decay from `tau_initial` to `tau_floor` over a fixed step span.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempSchedule {
    tau_initial: f64,
    tau_floor: f64,
    total_steps: u64,
    log_ratio: f64,
}

impl TempSchedule {
    /// Build a finite positive monotone exponential schedule.
    ///
    /// `tau(0) == tau_initial`; `tau(total_steps) == tau_floor`; later steps
    /// remain at the floor.
    ///
    /// # Errors
    /// Rejects non-finite/non-positive temperatures, an initial temperature
    /// below the floor, or a zero decay span.
    pub fn new(
        tau_initial: f64,
        tau_floor: f64,
        total_steps: u64,
    ) -> Result<Self, TemperatureError> {
        if !tau_initial.is_finite() || tau_initial <= 0.0 {
            return Err(TemperatureError::InvalidConfiguration(
                "initial temperature must be finite and positive",
            ));
        }
        if !tau_floor.is_finite() || tau_floor <= 0.0 {
            return Err(TemperatureError::InvalidConfiguration(
                "temperature floor must be finite and positive",
            ));
        }
        if tau_initial < tau_floor {
            return Err(TemperatureError::InvalidConfiguration(
                "initial temperature must not be below the floor",
            ));
        }
        if total_steps == 0 {
            return Err(TemperatureError::InvalidConfiguration(
                "temperature decay span must be positive",
            ));
        }
        Ok(Self {
            tau_initial,
            tau_floor,
            total_steps,
            // Subtract logarithms instead of taking `ln(floor / initial)`: the
            // direct ratio can underflow even when both endpoints are finite.
            log_ratio: tau_floor.ln() - tau_initial.ln(),
        })
    }

    /// Temperature at a zero-based step, clamped to the floor after decay.
    #[must_use]
    pub fn tau(self, step: u64) -> f64 {
        if step == 0 {
            return self.tau_initial;
        }
        if step >= self.total_steps {
            return self.tau_floor;
        }
        let progress = step as f64 / self.total_steps as f64;
        self.tau_initial * (self.log_ratio * progress).exp()
    }

    /// Initial base temperature.
    #[must_use]
    pub const fn tau_initial(self) -> f64 {
        self.tau_initial
    }

    /// Terminal base temperature.
    #[must_use]
    pub const fn tau_floor(self) -> f64 {
        self.tau_floor
    }

    /// Step at which this base schedule first reaches its floor.
    #[must_use]
    pub const fn floor_step(self) -> u64 {
        self.total_steps
    }
}

/// Ordered per-tensor HESTIA sensitivity scores in `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct HestiaSensitivityProfile {
    scores: Vec<f64>,
}

impl HestiaSensitivityProfile {
    /// Transform positive trace proxies with HESTIA's standardized sigmoid.
    ///
    /// For trace proxy `h_i`, this computes
    /// `sigmoid(gain * (ln(h_i) - mean(ln(h))) / (stddev(ln(h)) + epsilon))`.
    /// Population standard deviation is used because the supplied list is the
    /// complete converted tensor population, not a sample.
    ///
    /// # Errors
    /// Rejects an empty population, non-positive/non-finite trace proxies,
    /// non-positive/non-finite gain or epsilon, and non-finite arithmetic.
    pub fn standardized_sigmoid(
        trace_proxies: &[f64],
        gain: f64,
        epsilon: f64,
    ) -> Result<Self, TemperatureError> {
        if trace_proxies.is_empty() {
            return Err(TemperatureError::InvalidConfiguration(
                "sensitivity trace population cannot be empty",
            ));
        }
        if !gain.is_finite() || gain <= 0.0 {
            return Err(TemperatureError::InvalidConfiguration(
                "sensitivity gain must be finite and positive",
            ));
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(TemperatureError::InvalidConfiguration(
                "sensitivity epsilon must be finite and positive",
            ));
        }

        let mut logs = Vec::new();
        logs.try_reserve_exact(trace_proxies.len()).map_err(|_| {
            TemperatureError::InvalidConfiguration("sensitivity population is too large")
        })?;
        for &trace in trace_proxies {
            if !trace.is_finite() || trace <= 0.0 {
                return Err(TemperatureError::InvalidConfiguration(
                    "sensitivity traces must be finite and positive",
                ));
            }
            logs.push(trace.ln());
        }
        let mean = logs.iter().sum::<f64>() / logs.len() as f64;
        let variance = logs
            .iter()
            .map(|value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f64>()
            / logs.len() as f64;
        let denominator = variance.sqrt() + epsilon;
        if !mean.is_finite() || !denominator.is_finite() || denominator <= 0.0 {
            return Err(TemperatureError::InvalidConfiguration(
                "sensitivity normalization is not finite",
            ));
        }

        let mut scores = Vec::new();
        scores.try_reserve_exact(logs.len()).map_err(|_| {
            TemperatureError::InvalidConfiguration("sensitivity population is too large")
        })?;
        for value in logs {
            let standardized = gain * (value - mean) / denominator;
            if !standardized.is_finite() {
                return Err(TemperatureError::InvalidConfiguration(
                    "standardized sensitivity is not finite",
                ));
            }
            let score = if standardized >= 0.0 {
                1.0 / (1.0 + (-standardized).exp())
            } else {
                let exponential = standardized.exp();
                exponential / (1.0 + exponential)
            };
            scores.push(score);
        }
        Ok(Self { scores })
    }

    /// Ordered tensor sensitivity scores.
    #[must_use]
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    pub(crate) fn score(&self, tensor: usize) -> Result<f64, TemperatureError> {
        self.scores
            .get(tensor)
            .copied()
            .ok_or(TemperatureError::TensorOutOfRange {
                tensor,
                tensor_count: self.scores.len(),
            })
    }
}
