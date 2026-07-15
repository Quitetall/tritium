//! Dense-or-packed token table shared by embedding gather and a tied LM head.

use core::mem::size_of;

use rayon::prelude::*;
use tritium_format::PackedSaltRow;

use crate::error::NnError;
use crate::layers::packed_salt::PackedSaltMatrix;

#[derive(Clone, Debug)]
enum Storage {
    Dense(Vec<f32>),
    Salt(PackedSaltMatrix),
}

/// One token table used for both input embedding gather and exact tied-head logits.
///
/// Dense fp/BitNet models retain their original fp32 table. SALT models retain one
/// packed additive matrix; gather and unembedding execute against that same storage,
/// so tying never creates a second `vocab × hidden` allocation.
#[derive(Clone, Debug)]
pub struct TokenEmbedding {
    rows: usize,
    cols: usize,
    storage: Storage,
}

impl TokenEmbedding {
    /// Build a dense fp32 token table.
    ///
    /// # Errors
    /// [`NnError::Shape`] if dimensions are zero, overflow, or disagree with `values`.
    pub fn from_dense(values: Vec<f32>, rows: usize, cols: usize) -> Result<Self, NnError> {
        let expected = rows.checked_mul(cols).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: values.len(),
        })?;
        if rows == 0 || cols == 0 || values.len() != expected {
            return Err(NnError::Shape {
                expected: expected.max(1),
                got: values.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            storage: Storage::Dense(values),
        })
    }

    /// Build a token table from validated packed SALT rows.
    ///
    /// # Errors
    /// [`NnError::Shape`] if matrix geometry disagrees with the rows, or
    /// [`NnError::Backend`] if a packed plane contains a non-finite scale.
    pub fn from_packed_salt(
        rows_data: Vec<PackedSaltRow>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, NnError> {
        Ok(Self {
            rows,
            cols,
            storage: Storage::Salt(PackedSaltMatrix::new(rows_data, rows, cols)?),
        })
    }

    /// Vocabulary rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Hidden columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Borrow dense values when this is an fp32 table.
    ///
    /// Packed SALT storage deliberately returns `None`; callers must use gather or
    /// exact unembedding instead of silently materializing a dense copy.
    #[must_use]
    pub fn as_dense(&self) -> Option<&[f32]> {
        match &self.storage {
            Storage::Dense(values) => Some(values),
            Storage::Salt(_) => None,
        }
    }

    /// Whether this table retains packed additive SALT rows.
    #[must_use]
    pub const fn is_packed_salt(&self) -> bool {
        matches!(self.storage, Storage::Salt(_))
    }

    /// Number of sparse residual planes in the packed table, or zero for dense fp32.
    #[must_use]
    pub const fn sparse_plane_count(&self) -> usize {
        match &self.storage {
            Storage::Dense(_) => 0,
            Storage::Salt(matrix) => matrix.sparse_plane_count(),
        }
    }

    /// Retained payload and metadata bytes, excluding this wrapper.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match &self.storage {
            Storage::Dense(values) => values.capacity().saturating_mul(size_of::<f32>()),
            Storage::Salt(matrix) => matrix.resident_bytes(),
        }
    }

    /// Gather token rows into `out`, which must have `tokens.len() × cols` elements.
    ///
    /// # Errors
    /// [`NnError::Shape`] for a wrong output length or [`NnError::MissingTensor`]
    /// for an out-of-vocabulary token.
    pub fn gather(&self, tokens: &[u32], out: &mut [f32]) -> Result<(), NnError> {
        match &self.storage {
            Storage::Salt(matrix) => matrix.gather(tokens, out),
            Storage::Dense(values) => {
                let expected = tokens.len().checked_mul(self.cols).ok_or(NnError::Shape {
                    expected: usize::MAX,
                    got: out.len(),
                })?;
                if out.len() != expected {
                    return Err(NnError::Shape {
                        expected,
                        got: out.len(),
                    });
                }
                out.par_chunks_mut(self.cols)
                    .zip(tokens.par_iter())
                    .try_for_each(|(dst, &token)| {
                        let row = token as usize;
                        if row >= self.rows {
                            return Err(NnError::MissingTensor(format!("token_embd row {row}")));
                        }
                        let src = values
                            .get(row * self.cols..(row + 1) * self.cols)
                            .ok_or_else(|| {
                                NnError::MissingTensor(format!("token_embd row {row}"))
                            })?;
                        dst.copy_from_slice(src);
                        Ok::<(), NnError>(())
                    })
            }
        }
    }

    /// Compute exact tied-head logits in global-K accumulation order.
    ///
    /// Packed weights are reconstructed one 256-element block at a time. No A8
    /// activation quantization is applied on this path.
    ///
    /// # Errors
    /// [`NnError::Shape`] if `hidden` or `logits` has the wrong length.
    pub fn unembed_exact(&self, hidden: &[f32], logits: &mut [f32]) -> Result<(), NnError> {
        if hidden.len() != self.cols {
            return Err(NnError::Shape {
                expected: self.cols,
                got: hidden.len(),
            });
        }
        if logits.len() != self.rows {
            return Err(NnError::Shape {
                expected: self.rows,
                got: logits.len(),
            });
        }
        match &self.storage {
            Storage::Salt(matrix) => matrix.project_exact(hidden, logits),
            Storage::Dense(values) => {
                logits.par_iter_mut().enumerate().for_each(|(row, slot)| {
                    let weights = &values[row * self.cols..(row + 1) * self.cols];
                    let mut acc = 0.0f32;
                    for col in 0..self.cols {
                        acc += hidden[col] * weights[col];
                    }
                    *slot = acc;
                });
                Ok(())
            }
        }
    }
}
