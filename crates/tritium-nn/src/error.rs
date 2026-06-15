//! Error type for the inference layer.

use core::fmt;

/// Errors from nn ops and model loading.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum NnError {
    /// Operand/output buffer lengths disagree.
    Shape {
        /// Length that was required.
        expected: usize,
        /// Length supplied.
        got: usize,
    },
    /// A required GGUF metadata key was absent or the wrong type.
    MissingMetadata(String),
    /// A backend call failed; the message is the stringified `BackendError`.
    Backend(String),
}

impl fmt::Display for NnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NnError::Shape { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            NnError::MissingMetadata(key) => {
                write!(f, "missing or mistyped GGUF metadata: {key}")
            }
            NnError::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for NnError {}

impl From<tritium_spec::BackendError> for NnError {
    fn from(e: tritium_spec::BackendError) -> Self {
        NnError::Backend(e.to_string())
    }
}
