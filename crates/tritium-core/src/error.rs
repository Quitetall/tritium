//! Error type for the foundation crate. Hand-rolled (no `thiserror`) to keep
//! `tritium-core` dependency-free and `no_std`.

/// Errors from ternary type construction and reference math.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum TritError {
    /// A value outside `{-1, 0, +1}` was offered to [`crate::Trit`].
    OutOfRange(i32),
    /// Operand/output buffer lengths disagree with the [`crate::GemmShape`].
    ShapeMismatch {
        /// What the shape implies.
        expected: usize,
        /// What the buffer actually has.
        got: usize,
    },
}

impl core::fmt::Display for TritError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TritError::OutOfRange(v) => {
                write!(f, "value {v} out of ternary range {{-1, 0, 1}}")
            }
            TritError::ShapeMismatch { expected, got } => {
                write!(f, "buffer length mismatch: expected {expected}, got {got}")
            }
        }
    }
}

// `core::error::Error` is stable since Rust 1.81 (MSRV here is 1.89), so the
// Error impl is unconditional — no `std` feature gate.
impl core::error::Error for TritError {}
