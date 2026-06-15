//! Numerically-stable row-wise softmax (the attention-score normalizer).

use crate::error::NnError;

/// In-place row-wise softmax over a `[rows, row_len]` row-major buffer.
///
/// Each contiguous run of `row_len` elements is replaced by its softmax,
/// computed with the standard max-subtraction trick for numerical stability.
/// A fully-masked row (all `-inf`) is handled per the convention fixed in WF-2.
///
/// # Errors
/// [`NnError::Shape`] if `row_len == 0` or `x.len()` is not a multiple of
/// `row_len`.
pub fn softmax_rows(x: &mut [f32], row_len: usize) -> Result<(), NnError> {
    let _ = (x, row_len);
    todo!("WF-2: numerically-stable row-wise softmax")
}
