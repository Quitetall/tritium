//! Checked native QTEA arithmetic, not a registered solver or artifact codec.
//!
//! Targets the MIT author implementation at revision
//! `e39bd8c0297e3427daeb404427c74cb6e202b683`, not a mixture with paper Eq.5.
//! Scalar f32 reductions are left-to-right; Torch parallel reductions may differ.
//! Input/intermediate overflow and degenerate initial groups fail explicitly.
//! [`TensorFit`] adds a complete source-bound-Gram/SPD-only CPU feedback sweep.
//! No model converter, compressed codec/backend handle, GPU implementation,
//! frontier registration or empirical qualification is implied.

mod column;
mod factor;
mod group;
mod saliency;
mod sparse;
mod tensor;

pub use column::RefinedColumn;
pub use group::GroupParameters;
pub use saliency::SalientRanking;
pub use sparse::SparseResidualColumn;
pub use tensor::{
    TENSOR_FIT_ID, TensorFit, TensorFitConfig, TensorFitPlan, TensorResidual,
    numerical_input_digest,
};

/// Numerical-policy identity, deliberately distinct from a frontier solver ID.
pub const ARITHMETIC_ID: &str = "qtea.cpu-arithmetic.f32.v1";

/// Checked arithmetic refusal; no partially constructed result is returned.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ArithmeticError {
    /// Empty, mismatched, overflowing or unsupported geometry.
    InvalidShape,
    /// Nonpositive/nonfinite coefficient or more than64 fitting iterations.
    InvalidConfig,
    /// Input is NaN or infinite at the stated flat index.
    NonFiniteInput {
        /// Index in the borrowed input slice.
        index: usize,
    },
    /// Diagonal-only saliency input is negative, NaN or infinite.
    InvalidDiagonal {
        /// Input-column index in the supplied diagonal.
        column: usize,
    },
    /// Full tensor fitting requires a source-bound Gram.
    UnboundGram,
    /// Gram is asymmetric, indefinite, numerically singular or needs a forbidden pivot.
    InvalidGram,
    /// Initial group row has no entries outside its dead zone.
    DegenerateRow {
        /// Output-row index in the group.
        row: usize,
    },
    /// Finite input produced an invalid/overflowing intermediate.
    NonFiniteArithmetic,
    /// New live array-element bytes exceed the caller's budget.
    ResourceLimit {
        /// Required logical array-element bytes (not allocator/RSS).
        required: usize,
        /// Caller's maximum logical array-element bytes.
        limit: usize,
    },
    /// Fallible vector reservation failed after logical-budget admission.
    AllocationFailed,
}

impl std::fmt::Display for ArithmeticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QTEA arithmetic refused: {self:?}")
    }
}

impl std::error::Error for ArithmeticError {}

fn budget(required: Option<usize>, limit: usize) -> Result<(), ArithmeticError> {
    let required = required.ok_or(ArithmeticError::InvalidShape)?;
    if required > limit {
        return Err(ArithmeticError::ResourceLimit { required, limit });
    }
    Ok(())
}

fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, ArithmeticError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|_| ArithmeticError::AllocationFailed)?;
    output.resize(len, value);
    Ok(output)
}

fn finite(value: f32) -> Result<f32, ArithmeticError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ArithmeticError::NonFiniteArithmetic)
    }
}

fn check_input(input: &[f32]) -> Result<(), ArithmeticError> {
    if let Some(index) = input.iter().position(|value| !value.is_finite()) {
        return Err(ArithmeticError::NonFiniteInput { index });
    }
    Ok(())
}

fn check_iterations(iterations: u32) -> Result<(), ArithmeticError> {
    if iterations > 64 {
        return Err(ArithmeticError::InvalidConfig);
    }
    Ok(())
}
