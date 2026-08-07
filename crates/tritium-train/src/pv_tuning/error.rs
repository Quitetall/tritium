use core::fmt;

/// Validation, step, or checkpoint failure. Step failures leave session unchanged.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PvTuningError {
    /// Recipe is invalid.
    InvalidConfig(String),
    /// Deployed representation is invalid.
    InvalidWeight(String),
    /// Gradient or step sequencing is invalid.
    Step(String),
    /// Checkpoint is corrupt, stale, or identity-mismatched.
    Checkpoint(String),
}

impl PvTuningError {
    pub(super) fn invalid_config(message: impl Into<String>) -> Self {
        Self::InvalidConfig(message.into())
    }

    pub(super) fn invalid_weight(message: impl Into<String>) -> Self {
        Self::InvalidWeight(message.into())
    }

    pub(super) fn step(message: impl Into<String>) -> Self {
        Self::Step(message.into())
    }

    pub(super) fn checkpoint(message: impl Into<String>) -> Self {
        Self::Checkpoint(message.into())
    }
}

impl fmt::Display for PvTuningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid PV config: {message}"),
            Self::InvalidWeight(message) => write!(f, "invalid PV weight: {message}"),
            Self::Step(message) => write!(f, "PV step failed: {message}"),
            Self::Checkpoint(message) => write!(f, "PV checkpoint failed: {message}"),
        }
    }
}

impl std::error::Error for PvTuningError {}
