//! Scalar packed QTEA forward, not SIMD/GPU or complete-model qualification.
//!
//! Operates directly on checked format bytes, with no offline fitter dependency
//! or dense weight buffer. Staging is caller-owned and makes output refusal atomic.

use tritium_format::qtea::{QteaError, QteaTensorView};

/// Checked CPU execution refusal. Scratch may be dirty; output is unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum QteaCpuError {
    /// Input/output/staging counts do not match checked batch geometry.
    Shape,
    /// A caller input is nonfinite.
    NonFiniteInput {
        /// Flat input index, without printing input values.
        index: usize,
    },
    /// Finite operands overflowed a product or sequential reduction.
    ArithmeticOverflow,
    /// Checked format accessor refused an index.
    Format(QteaError),
}
impl core::fmt::Display for QteaCpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "QTEA CPU refused: {self:?}")
    }
}
impl std::error::Error for QteaCpuError {}
impl From<QteaError> for QteaCpuError {
    fn from(error: QteaError) -> Self {
        Self::Format(error)
    }
}

/// Execute `[batch,columns] * tensor^T` into exact `[batch,rows]` output.
///
/// Visits batch, output row, then original column in ascending order, using f32
/// `(alpha*v)*trit+beta`, optional selected residual, input product and reduction.
/// No reassociation, heap allocation or dense weight expansion. `staging` must
/// have the exact output length and is the only execution workspace. Batch0 is
/// valid. No packed SIMD speedup or physical-memory receipt is implied.
///
/// # Errors
/// Refuses shape/count overflow, nonfinite inputs and arithmetic overflow before
/// changing output. Staging can be partially written on a late failure.
pub fn matmul_into(
    tensor: QteaTensorView<'_>,
    inputs: &[f32],
    batch: usize,
    output: &mut [f32],
    staging: &mut [f32],
) -> Result<(), QteaCpuError> {
    let meta = tensor.metadata();
    let (rows, columns) = (meta.rows, meta.columns);
    if batch.checked_mul(columns) != Some(inputs.len())
        || batch.checked_mul(rows) != Some(output.len())
        || staging.len() != output.len()
    {
        return Err(QteaCpuError::Shape);
    }
    if let Some(index) = inputs.iter().position(|x| !x.is_finite()) {
        return Err(QteaCpuError::NonFiniteInput { index });
    }
    for b in 0..batch {
        for row in 0..rows {
            let mut residuals = tensor.residuals();
            let mut residual = residuals.next();
            let (mut group, mut alpha, mut beta) = (usize::MAX, 0.0, 0.0);
            let mut sum = 0.0_f32;
            for column in 0..columns {
                let current = column / meta.recipe.group_size;
                if current != group {
                    (alpha, beta) = tensor.group_parameters(current, row)?;
                    group = current;
                }
                let mut weight = (alpha * tensor.column_multiplier(column)?)
                    * tensor.trit(row, column)?.to_f32()
                    + beta;
                if let Some(record) = residual.filter(|record| record.column() == column) {
                    if let Some(value) = record.value_at_row(row)? {
                        weight += value;
                    }
                    residual = residuals.next();
                }
                let product = inputs[b * columns + column] * weight;
                if !product.is_finite() {
                    return Err(QteaCpuError::ArithmeticOverflow);
                }
                sum += product;
                if !sum.is_finite() {
                    return Err(QteaCpuError::ArithmeticOverflow);
                }
            }
            staging[b * rows + row] = sum;
        }
    }
    output.copy_from_slice(staging);
    Ok(())
}
