//! Packed additive SALT projection.
//!
//! [`SaltLinear`] retains TQ2_0 planes and reconstructs one 256-weight block at a
//! time during the A8 contraction. It matches [`DenseLinear`](super::DenseLinear)
//! bit-for-bit without retaining an `N × K` fp32 matrix.

use half::f16;
use rayon::prelude::*;
use tritium_core::Trit;
use tritium_format::{QK_K, SaltRow, TQ2_0_BLOCK_BYTES, unpack_tq2_0_block};

use crate::error::NnError;
use crate::ops::quantize_activation_int8;

/// A bias-free additive ternary projection backed by packed SALT rows.
#[derive(Clone, Debug)]
pub struct SaltLinear {
    n_out: usize,
    k_in: usize,
    rows: Vec<SaltRow>,
    packed_bytes: usize,
}

impl SaltLinear {
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
        if k_in == 0 {
            return Err(NnError::Shape {
                expected: 1,
                got: 0,
            });
        }
        if u32::try_from(k_in).is_err() {
            return Err(NnError::Shape {
                expected: u32::MAX as usize,
                got: k_in,
            });
        }
        let mut packed_bytes = 0usize;
        for row in &rows {
            if row.k != k_in {
                return Err(NnError::Shape {
                    expected: k_in,
                    got: row.k,
                });
            }
            packed_bytes =
                packed_bytes
                    .checked_add(validate_packed_row(row)?)
                    .ok_or(NnError::Shape {
                        expected: usize::MAX,
                        got: packed_bytes,
                    })?;
        }
        Ok(Self {
            n_out,
            k_in,
            rows,
            packed_bytes,
        })
    }

    /// Packed plane payload bytes retained by this projection.
    #[must_use]
    pub const fn packed_bytes(&self) -> usize {
        self.packed_bytes
    }

    /// Output feature count (`N`).
    #[must_use]
    pub const fn n_out(&self) -> usize {
        self.n_out
    }

    /// Input feature count (`K`).
    #[must_use]
    pub const fn k_in(&self) -> usize {
        self.k_in
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
        let act_len = m.checked_mul(self.k_in).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: act.len(),
        })?;
        if act.len() != act_len {
            return Err(NnError::Shape {
                expected: act_len,
                got: act.len(),
            });
        }
        let out_len = m.checked_mul(self.n_out).ok_or(NnError::Shape {
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
        quantize_activation_int8(act, m, self.k_in, &mut q_act, &mut act_scale)?;

        out.par_iter_mut()
            .enumerate()
            .try_for_each(|(output_index, slot)| {
                let activation_row = output_index / self.n_out;
                let output_channel = output_index % self.n_out;
                let q = &q_act[activation_row * self.k_in..(activation_row + 1) * self.k_in];
                *slot = packed_row_dot(q, &self.rows[output_channel])? * act_scale[activation_row];
                Ok::<(), NnError>(())
            })
    }
}

fn validate_packed_row(row: &SaltRow) -> Result<usize, NnError> {
    if row.planes.len() > u8::MAX as usize {
        return Err(NnError::Backend(format!(
            "SALT row has {} planes; maximum is {}",
            row.planes.len(),
            u8::MAX
        )));
    }
    let blocks = row.k.div_ceil(QK_K);
    let plane_bytes = blocks
        .checked_mul(TQ2_0_BLOCK_BYTES)
        .ok_or(NnError::Shape {
            expected: usize::MAX,
            got: row.k,
        })?;
    let mut trits = [Trit::ZERO; QK_K];
    let mut scale = f16::ZERO;
    for plane in &row.planes {
        if plane.len() != plane_bytes {
            return Err(NnError::Backend(format!(
                "malformed SALT plane: expected {plane_bytes} bytes, got {}",
                plane.len()
            )));
        }
        for block in plane.chunks_exact(TQ2_0_BLOCK_BYTES) {
            unpack_tq2_0_block(block, &mut trits, &mut scale)
                .map_err(|error| NnError::Backend(error.to_string()))?;
        }
    }
    row.planes
        .len()
        .checked_mul(plane_bytes)
        .ok_or(NnError::Shape {
            expected: usize::MAX,
            got: row.planes.len(),
        })
}

fn packed_row_dot(act: &[f32], row: &SaltRow) -> Result<f32, NnError> {
    let mut acc = 0.0f32;
    let mut trits = [Trit::ZERO; QK_K];
    let mut weight = [0.0f32; QK_K];
    let blocks = row.k.div_ceil(QK_K);
    for block_index in 0..blocks {
        weight.fill(0.0);
        for plane in &row.planes {
            let start = block_index * TQ2_0_BLOCK_BYTES;
            let block = &plane[start..start + TQ2_0_BLOCK_BYTES];
            let mut scale = f16::ZERO;
            unpack_tq2_0_block(block, &mut trits, &mut scale)
                .map_err(|error| NnError::Backend(error.to_string()))?;
            let scale = scale.to_f32();
            for (combined, trit) in weight.iter_mut().zip(&trits) {
                *combined += scale * trit.to_f32();
            }
        }
        let start = block_index * QK_K;
        let end = (start + QK_K).min(row.k);
        for index in start..end {
            acc += act[index] * weight[index - start];
        }
    }
    Ok(acc)
}
