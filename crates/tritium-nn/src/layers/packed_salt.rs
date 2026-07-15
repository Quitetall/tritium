//! Shared packed SALT matrix storage for projections and tied token tables.

use core::mem::size_of;

use half::f16;
use rayon::prelude::*;
use tritium_core::Trit;
use tritium_format::{PackedSaltRow, PlaneRepr, QK_K, TQ2_0_BLOCK_BYTES, unpack_tq2_0_block};

use crate::error::NnError;

const SPARSE_SIGN_BIT: u32 = 1 << 31;
const SPARSE_INDEX_MASK: u32 = !SPARSE_SIGN_BIT;

#[derive(Clone, Copy, Debug)]
struct RowMeta {
    plane_start: usize,
    plane_len: usize,
}

#[derive(Clone, Copy, Debug)]
enum PlaneMeta {
    Dense {
        byte_offset: usize,
    },
    Sparse {
        scale_offset: usize,
        entry_offset: usize,
        entry_len: usize,
    },
}

/// Validated SALT rows flattened into allocation-efficient dense/sparse arenas.
#[derive(Clone, Debug)]
pub(crate) struct PackedSaltMatrix {
    n_out: usize,
    k_in: usize,
    rows: Vec<RowMeta>,
    planes: Vec<PlaneMeta>,
    dense_bytes: Vec<u8>,
    sparse_scales: Vec<f16>,
    sparse_entries: Vec<u32>,
    sparse_plane_count: usize,
}

impl PackedSaltMatrix {
    pub(crate) fn new(
        rows: Vec<PackedSaltRow>,
        n_out: usize,
        k_in: usize,
    ) -> Result<Self, NnError> {
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

        let mut plane_capacity = 0usize;
        let mut dense_capacity = 0usize;
        let mut scale_capacity = 0usize;
        let mut entry_capacity = 0usize;
        let mut sparse_plane_count = 0usize;
        for row in &rows {
            if row.k() != k_in {
                return Err(NnError::Shape {
                    expected: k_in,
                    got: row.k(),
                });
            }
            plane_capacity =
                plane_capacity
                    .checked_add(row.plane_count())
                    .ok_or(NnError::Shape {
                        expected: usize::MAX,
                        got: plane_capacity,
                    })?;
            for plane in row.planes() {
                match plane {
                    PlaneRepr::Dense(bytes) => {
                        if bytes.chunks_exact(TQ2_0_BLOCK_BYTES).any(|block| {
                            !f16::from_bits(u16::from_le_bytes([
                                block[TQ2_0_BLOCK_BYTES - 2],
                                block[TQ2_0_BLOCK_BYTES - 1],
                            ]))
                            .is_finite()
                        }) {
                            return Err(NnError::Backend(
                                "SALT dense plane contains a non-finite scale".to_owned(),
                            ));
                        }
                        dense_capacity =
                            dense_capacity
                                .checked_add(bytes.len())
                                .ok_or(NnError::Shape {
                                    expected: usize::MAX,
                                    got: dense_capacity,
                                })?;
                    }
                    PlaneRepr::Sparse(sparse) => {
                        if sparse.scales.iter().any(|scale| !scale.is_finite()) {
                            return Err(NnError::Backend(
                                "SALT sparse plane contains a non-finite scale".to_owned(),
                            ));
                        }
                        scale_capacity = scale_capacity.checked_add(sparse.scales.len()).ok_or(
                            NnError::Shape {
                                expected: usize::MAX,
                                got: scale_capacity,
                            },
                        )?;
                        entry_capacity =
                            entry_capacity
                                .checked_add(sparse.idx.len())
                                .ok_or(NnError::Shape {
                                    expected: usize::MAX,
                                    got: entry_capacity,
                                })?;
                        sparse_plane_count =
                            sparse_plane_count.checked_add(1).ok_or(NnError::Shape {
                                expected: usize::MAX,
                                got: sparse_plane_count,
                            })?;
                    }
                }
            }
        }

        let mut row_meta = Vec::with_capacity(n_out);
        let mut planes = Vec::with_capacity(plane_capacity);
        let mut dense_bytes = Vec::with_capacity(dense_capacity);
        let mut sparse_scales = Vec::with_capacity(scale_capacity);
        let mut sparse_entries = Vec::with_capacity(entry_capacity);

        for row in rows {
            let plane_start = planes.len();
            for plane in row.planes() {
                match plane {
                    PlaneRepr::Dense(bytes) => {
                        let byte_offset = dense_bytes.len();
                        dense_bytes.extend_from_slice(bytes);
                        planes.push(PlaneMeta::Dense { byte_offset });
                    }
                    PlaneRepr::Sparse(sparse) => {
                        let scale_offset = sparse_scales.len();
                        let entry_offset = sparse_entries.len();
                        sparse_scales.extend_from_slice(&sparse.scales);
                        for (&index, &sign) in sparse.idx.iter().zip(&sparse.sign) {
                            let entry = if sign < 0 {
                                index | SPARSE_SIGN_BIT
                            } else {
                                index
                            };
                            sparse_entries.push(entry);
                        }
                        planes.push(PlaneMeta::Sparse {
                            scale_offset,
                            entry_offset,
                            entry_len: sparse.idx.len(),
                        });
                    }
                }
            }
            row_meta.push(RowMeta {
                plane_start,
                plane_len: planes.len() - plane_start,
            });
        }

        dense_bytes.shrink_to_fit();
        sparse_scales.shrink_to_fit();
        sparse_entries.shrink_to_fit();
        planes.shrink_to_fit();
        row_meta.shrink_to_fit();
        Ok(Self {
            n_out,
            k_in,
            rows: row_meta,
            planes,
            dense_bytes,
            sparse_scales,
            sparse_entries,
            sparse_plane_count,
        })
    }

    pub(crate) const fn n_out(&self) -> usize {
        self.n_out
    }

    pub(crate) const fn k_in(&self) -> usize {
        self.k_in
    }

    pub(crate) const fn sparse_plane_count(&self) -> usize {
        self.sparse_plane_count
    }

    pub(crate) fn packed_bytes(&self) -> usize {
        self.dense_bytes
            .len()
            .saturating_add(self.sparse_scales.len().saturating_mul(size_of::<f16>()))
            .saturating_add(self.sparse_entries.len().saturating_mul(size_of::<u32>()))
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.dense_bytes
            .capacity()
            .saturating_add(
                self.sparse_scales
                    .capacity()
                    .saturating_mul(size_of::<f16>()),
            )
            .saturating_add(
                self.sparse_entries
                    .capacity()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(self.rows.capacity().saturating_mul(size_of::<RowMeta>()))
            .saturating_add(
                self.planes
                    .capacity()
                    .saturating_mul(size_of::<PlaneMeta>()),
            )
    }

    pub(crate) fn gather(&self, tokens: &[u32], out: &mut [f32]) -> Result<(), NnError> {
        let expected = tokens.len().checked_mul(self.k_in).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: out.len(),
        })?;
        if out.len() != expected {
            return Err(NnError::Shape {
                expected,
                got: out.len(),
            });
        }
        out.par_chunks_mut(self.k_in)
            .zip(tokens.par_iter())
            .try_for_each(|(dst, &token)| {
                let row = token as usize;
                if row >= self.n_out {
                    return Err(NnError::MissingTensor(format!("token_embd row {row}")));
                }
                self.dequant_row(row, dst)
            })
    }

    pub(crate) fn project_exact(&self, act: &[f32], out: &mut [f32]) -> Result<(), NnError> {
        self.project_rows(act, 1, out)
    }

    pub(crate) fn project_rows(
        &self,
        act: &[f32],
        m: usize,
        out: &mut [f32],
    ) -> Result<(), NnError> {
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
        out.par_iter_mut()
            .enumerate()
            .try_for_each(|(output_index, slot)| {
                let activation_row = output_index / self.n_out;
                let output_channel = output_index % self.n_out;
                let act = &act[activation_row * self.k_in..(activation_row + 1) * self.k_in];
                *slot = self.row_dot(act, output_channel)?;
                Ok::<(), NnError>(())
            })
    }

    fn dequant_row(&self, row: usize, out: &mut [f32]) -> Result<(), NnError> {
        debug_assert_eq!(out.len(), self.k_in);
        let blocks = self.k_in.div_ceil(QK_K);
        let mut trits = [Trit::ZERO; QK_K];
        let mut weight = [0.0f32; QK_K];
        for block in 0..blocks {
            self.reconstruct_block(row, block, &mut trits, &mut weight)?;
            let start = block * QK_K;
            let end = (start + QK_K).min(self.k_in);
            out[start..end].copy_from_slice(&weight[..end - start]);
        }
        Ok(())
    }

    fn row_dot(&self, act: &[f32], row: usize) -> Result<f32, NnError> {
        let mut acc = 0.0f32;
        let blocks = self.k_in.div_ceil(QK_K);
        let mut trits = [Trit::ZERO; QK_K];
        let mut weight = [0.0f32; QK_K];
        for block in 0..blocks {
            self.reconstruct_block(row, block, &mut trits, &mut weight)?;
            let start = block * QK_K;
            let end = (start + QK_K).min(self.k_in);
            for index in start..end {
                acc += act[index] * weight[index - start];
            }
        }
        Ok(acc)
    }

    fn reconstruct_block(
        &self,
        row: usize,
        block: usize,
        trits: &mut [Trit; QK_K],
        weight: &mut [f32; QK_K],
    ) -> Result<(), NnError> {
        weight.fill(0.0);
        let row = self.rows[row];
        let start_col = block * QK_K;
        let logical_len = (self.k_in - start_col).min(QK_K);
        for plane in &self.planes[row.plane_start..row.plane_start + row.plane_len] {
            match *plane {
                PlaneMeta::Dense { byte_offset } => {
                    let start = byte_offset + block * TQ2_0_BLOCK_BYTES;
                    let bytes = &self.dense_bytes[start..start + TQ2_0_BLOCK_BYTES];
                    let mut scale = f16::ZERO;
                    unpack_tq2_0_block(bytes, trits, &mut scale)
                        .map_err(|error| NnError::Backend(error.to_string()))?;
                    let scale = scale.to_f32();
                    for index in 0..logical_len {
                        weight[index] += scale * trits[index].to_f32();
                    }
                }
                PlaneMeta::Sparse {
                    scale_offset,
                    entry_offset,
                    entry_len,
                } => {
                    let scale = self.sparse_scales[scale_offset + block].to_f32();
                    let entries = &self.sparse_entries[entry_offset..entry_offset + entry_len];
                    let end_col = start_col + logical_len;
                    let first = entries
                        .partition_point(|entry| (*entry & SPARSE_INDEX_MASK) < start_col as u32);
                    let last = entries
                        .partition_point(|entry| (*entry & SPARSE_INDEX_MASK) < end_col as u32);
                    for &entry in &entries[first..last] {
                        let index = (entry & SPARSE_INDEX_MASK) as usize - start_col;
                        let sign = if entry & SPARSE_SIGN_BIT == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        weight[index] += scale * sign;
                    }
                }
            }
        }
        Ok(())
    }
}
