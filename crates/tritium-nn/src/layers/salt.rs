//! Packed additive SALT projection.
//!
//! [`SaltLinear`] retains dense and sparse additive planes and reconstructs one
//! 256-weight block at a time during the A8 contraction. It matches
//! [`DenseLinear`](super::DenseLinear) bit-for-bit without retaining an `N × K`
//! fp32 matrix.

use tritium_format::{PackedSaltRow, SaltRow};

use crate::error::NnError;
use crate::layers::packed_salt::PackedSaltMatrix;
use crate::ops::quantize_activation_int8;

/// A bias-free additive ternary projection backed by packed SALT rows.
#[derive(Clone, Debug)]
pub struct SaltLinear {
    matrix: PackedSaltMatrix,
}

impl SaltLinear {
    pub(crate) fn from_packed_matrix(matrix: PackedSaltMatrix) -> Self {
        Self { matrix }
    }

    /// Build a projection from one packed SALT row per output channel.
    ///
    /// Validation consumes fixed one-block scratch; no dense row or matrix is materialized.
    ///
    /// # Errors
    /// [`NnError::Shape`] if dimensions or packed row geometry disagree, or
    /// [`NnError::Backend`] if a TQ2_0 plane is malformed.
    pub fn new(rows: Vec<SaltRow>, n_out: usize, k_in: usize) -> Result<Self, NnError> {
        if n_out == 0 || rows.len() != n_out {
            return Err(NnError::Shape {
                expected: n_out.max(1),
                got: rows.len(),
            });
        }
        if k_in == 0 || u32::try_from(k_in).is_err() {
            return Err(NnError::Shape {
                expected: if k_in == 0 { 1 } else { u32::MAX as usize },
                got: k_in,
            });
        }
        if let Some(row) = rows.iter().find(|row| row.k != k_in) {
            return Err(NnError::Shape {
                expected: k_in,
                got: row.k,
            });
        }
        let rows = rows
            .into_iter()
            .map(PackedSaltRow::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| NnError::Backend(error.to_string()))?;
        Self::from_packed_rows(rows, n_out, k_in)
    }

    /// Build directly from validated dense/sparse SALT rows without expanding residuals.
    ///
    /// # Errors
    /// [`NnError::Shape`] if matrix geometry disagrees with the rows, or
    /// [`NnError::Backend`] if a packed plane contains a non-finite scale.
    pub fn from_packed_rows(
        rows: Vec<PackedSaltRow>,
        n_out: usize,
        k_in: usize,
    ) -> Result<Self, NnError> {
        Ok(Self {
            matrix: PackedSaltMatrix::new(rows, n_out, k_in)?,
        })
    }

    /// Packed plane payload bytes retained by this projection.
    #[must_use]
    pub fn packed_bytes(&self) -> usize {
        self.matrix.packed_bytes()
    }

    /// Total retained arena and row/plane metadata bytes, excluding the struct itself.
    /// Cloned projections share arenas, so summing this value across clones double-counts them.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.matrix.resident_bytes()
    }

    /// Number of residual planes retained in sparse form.
    #[must_use]
    pub const fn sparse_plane_count(&self) -> usize {
        self.matrix.sparse_plane_count()
    }

    /// Output feature count (`N`).
    #[must_use]
    pub const fn n_out(&self) -> usize {
        self.matrix.n_out()
    }

    /// Input feature count (`K`).
    #[must_use]
    pub const fn k_in(&self) -> usize {
        self.matrix.k_in()
    }

    /// A8 forward matching `salt_rows_to_dense → DenseLinear::new` bit-for-bit.
    ///
    /// Packed rows remain resident. Each worker uses fixed 256-element reconstruction
    /// scratch, preserving plane-order weight reconstruction and global `K` dot order.
    ///
    /// # Errors
    /// [`NnError::Shape`] on operand/output mismatch, or [`NnError::Backend`] if a
    /// retained packed block cannot be decoded.
    pub fn forward(&self, act: &[f32], m: usize, out: &mut [f32]) -> Result<(), NnError> {
        let act_len = m.checked_mul(self.k_in()).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: act.len(),
        })?;
        if act.len() != act_len {
            return Err(NnError::Shape {
                expected: act_len,
                got: act.len(),
            });
        }
        let out_len = m.checked_mul(self.n_out()).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: out.len(),
        })?;
        if out.len() != out_len {
            return Err(NnError::Shape {
                expected: out_len,
                got: out.len(),
            });
        }

        let mut q_act = vec![0.0f32; act_len];
        let mut act_scale = vec![0.0f32; m];
        quantize_activation_int8(act, m, self.k_in(), &mut q_act, &mut act_scale)?;

        self.matrix.project_rows(&q_act, m, out)?;
        for (row, scale) in out.chunks_mut(self.n_out()).zip(act_scale) {
            for value in row {
                *value *= scale;
            }
        }
        Ok(())
    }
}
