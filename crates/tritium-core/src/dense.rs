//! Allocation-free dense reference operations for semantic tensor execution.

/// Failure validating or consuming a dense semantic tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DenseError {
    /// A dimension product does not fit the target's address space.
    SizeOverflow,
    /// The weight buffer does not match `rows * cols`.
    WeightShape,
    /// Activation, scratch or output lengths do not match the operation.
    OperationShape,
    /// At least one gathered row is outside the matrix.
    RowOutOfRange,
}

/// Validated borrowed row-major dense f32 matrix.
///
/// Zero-width and zero-row matrices are legal. All operation geometry and IDs
/// are validated before writing outputs or scratch. No heap allocation occurs.
#[derive(Clone, Copy, Debug)]
pub struct DenseView<'a> {
    rows: usize,
    cols: usize,
    values: &'a [f32],
}

impl<'a> DenseView<'a> {
    /// Validate a dense matrix without copying its values.
    pub fn new(rows: usize, cols: usize, values: &'a [f32]) -> Result<Self, DenseError> {
        let expected = rows.checked_mul(cols).ok_or(DenseError::SizeOverflow)?;
        if values.len() != expected {
            return Err(DenseError::WeightShape);
        }
        Ok(Self { rows, cols, values })
    }

    /// Compute `[batch, cols] * weights^T`, preserving scalar f32 dot order.
    ///
    /// Identity-basis activations are also copied into caller-owned scratch,
    /// matching the semantic tensor matmul contract.
    pub fn matmul(
        &self,
        act: &[f32],
        batch: usize,
        scratch: &mut [f32],
        out: &mut [f32],
    ) -> Result<(), DenseError> {
        let inputs = batch
            .checked_mul(self.cols)
            .ok_or(DenseError::SizeOverflow)?;
        let outputs = batch
            .checked_mul(self.rows)
            .ok_or(DenseError::SizeOverflow)?;
        if act.len() != inputs || scratch.len() != inputs || out.len() != outputs {
            return Err(DenseError::OperationShape);
        }
        scratch.copy_from_slice(act);
        if self.rows == 0 {
            return Ok(());
        }
        for batch in 0..batch {
            for row in 0..self.rows {
                let weights = &self.values[row * self.cols..(row + 1) * self.cols];
                let activations = &act[batch * self.cols..(batch + 1) * self.cols];
                out[batch * self.rows + row] = activations
                    .iter()
                    .zip(weights)
                    .map(|(activation, weight)| activation * weight)
                    .sum();
            }
        }
        Ok(())
    }

    /// Copy logical rows in ID order, including duplicates and empty gathers.
    pub fn embed(&self, ids: &[usize], out: &mut [f32]) -> Result<(), DenseError> {
        let expected = ids
            .len()
            .checked_mul(self.cols)
            .ok_or(DenseError::SizeOverflow)?;
        if out.len() != expected {
            return Err(DenseError::OperationShape);
        }
        if ids.iter().any(|&row| row >= self.rows) {
            return Err(DenseError::RowOutOfRange);
        }
        if self.cols != 0 {
            for (&row, output) in ids.iter().zip(out.chunks_exact_mut(self.cols)) {
                output.copy_from_slice(&self.values[row * self.cols..(row + 1) * self.cols]);
            }
        }
        Ok(())
    }
}
