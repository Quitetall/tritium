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
    /// A required HuggingFace `config.json` key was absent or the wrong type; the
    /// `String` is the key (or a short reason).
    MissingConfig(String),
    /// A backend call failed; the message is the stringified `BackendError`.
    Backend(String),
    /// A tensor used a ggml type-id this layer cannot consume (the `u32` is the
    /// offending ggml type-id, e.g. an unexpected quantization scheme).
    UnsupportedTensorType(u32),
    /// A weight tensor the model loader requires was not present in the GGUF
    /// file; the `String` is the missing tensor name.
    MissingTensor(String),
    /// No execution backend satisfying the model's needs was available from the
    /// runtime registry (e.g. ternary mpGEMM is unsupported on every device).
    BackendUnavailable,
    /// Tokenization or detokenization failed; the `String` is a human-readable
    /// message from the tokenizer implementation.
    Tokenizer(String),
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
            NnError::MissingConfig(key) => {
                write!(f, "missing or mistyped HF config key: {key}")
            }
            NnError::Backend(msg) => write!(f, "backend error: {msg}"),
            NnError::UnsupportedTensorType(t) => {
                write!(f, "unsupported ggml tensor type-id: {t}")
            }
            NnError::MissingTensor(name) => write!(f, "missing tensor: {name}"),
            NnError::BackendUnavailable => {
                write!(f, "no execution backend available for this model")
            }
            NnError::Tokenizer(msg) => write!(f, "tokenizer error: {msg}"),
        }
    }
}

impl std::error::Error for NnError {}

impl From<tritium_spec::BackendError> for NnError {
    fn from(e: tritium_spec::BackendError) -> Self {
        NnError::Backend(e.to_string())
    }
}
