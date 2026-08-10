use tritium_spec::BackendError;
use tritium_train::{PvTuningError, RecoveryError};

use crate::training::TrainingAdapterError;

/// Failure from hard-PV construction, execution, or resume.
#[derive(Debug)]
#[non_exhaustive]
pub enum DevicePvRecoveryError {
    InvalidInput(String),
    Backend(String),
    Pv(String),
    Checkpoint(String),
    Campaign(String),
}

impl core::fmt::Display for DevicePvRecoveryError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput(reason) => write!(formatter, "invalid device PV input: {reason}"),
            Self::Backend(reason) => write!(formatter, "device PV backend error: {reason}"),
            Self::Pv(reason) => write!(formatter, "device PV update error: {reason}"),
            Self::Checkpoint(reason) => write!(formatter, "device PV checkpoint error: {reason}"),
            Self::Campaign(reason) => write!(formatter, "device PV campaign error: {reason}"),
        }
    }
}

impl std::error::Error for DevicePvRecoveryError {}

impl From<BackendError> for DevicePvRecoveryError {
    fn from(error: BackendError) -> Self {
        Self::Backend(error.to_string())
    }
}

impl From<TrainingAdapterError> for DevicePvRecoveryError {
    fn from(error: TrainingAdapterError) -> Self {
        Self::Backend(error.to_string())
    }
}

impl From<PvTuningError> for DevicePvRecoveryError {
    fn from(error: PvTuningError) -> Self {
        Self::Pv(error.to_string())
    }
}

impl From<RecoveryError> for DevicePvRecoveryError {
    fn from(error: RecoveryError) -> Self {
        Self::Campaign(error.to_string())
    }
}

impl From<tritium_format::FormatError> for DevicePvRecoveryError {
    fn from(error: tritium_format::FormatError) -> Self {
        Self::InvalidInput(error.to_string())
    }
}
