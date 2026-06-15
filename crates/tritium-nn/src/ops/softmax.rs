//! Numerically-stable row-wise softmax (the attention-score normalizer).

use crate::error::NnError;

/// In-place row-wise softmax over a `[rows, row_len]` row-major buffer.
///
/// Each contiguous run of `row_len` elements is replaced by its softmax,
/// computed with the standard max-subtraction trick for numerical stability:
/// `out_i = exp(x_i - m) / sum_j exp(x_j - m)` where `m = max_j x_j`.
///
/// Masked entries are encoded as `f32::NEG_INFINITY`; after subtracting a finite
/// row max they map to `exp(-inf) = 0`, contributing nothing to either the
/// numerator or the denominator. A fully-masked row (every element
/// `-inf`, so `m = -inf`) follows the torch convention: the `0/0` ratio yields
/// `NaN` for every element, matching the committed golden.
///
/// # Errors
/// [`NnError::Shape`] if `row_len == 0` or `x.len()` is not a multiple of
/// `row_len`.
pub fn softmax_rows(x: &mut [f32], row_len: usize) -> Result<(), NnError> {
    if row_len == 0 || !x.len().is_multiple_of(row_len) {
        return Err(NnError::Shape {
            expected: row_len,
            got: x.len(),
        });
    }

    for row in x.chunks_mut(row_len) {
        // Row maximum (NaN-ignoring via `>`); for an all-`-inf` row this stays
        // `-inf`, which is what drives the torch-matching NaN convention below.
        let mut max = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > max {
                max = v;
            }
        }

        // exp(x - max); a `-inf` max makes `x - max` be `NaN` (e.g. -inf - -inf),
        // so a fully-masked row produces all-NaN, then a NaN sum, then NaN/NaN
        // per element — exactly torch's degenerate-softmax output.
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            let e = (*v - max).exp();
            *v = e;
            sum += e;
        }

        let inv = 1.0f32 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }

    Ok(())
}
