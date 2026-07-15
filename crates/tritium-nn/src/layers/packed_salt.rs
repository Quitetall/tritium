//! Shared packed SALT matrix storage for projections and tied token tables.

use core::mem::size_of;
use std::sync::Arc;

use half::f16;
use rayon::prelude::*;
use tritium_core::Trit;
use tritium_format::{
    PackedSaltRow, PackedSaltRowRef, PackedSaltStorageRequirements, PlaneRepr, QK_K,
    TQ2_0_BLOCK_BYTES, unpack_tq2_0_block,
};

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
    sparse_plane_count: usize,
    storage: Arc<PackedSaltStorage>,
}

#[derive(Debug)]
struct PackedSaltStorage {
    rows: Vec<RowMeta>,
    planes: Vec<PlaneMeta>,
    dense_bytes: Vec<u8>,
    sparse_scales: Vec<f16>,
    sparse_entries: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MatrixRequirements {
    rows: usize,
    planes: usize,
    dense_bytes: usize,
    sparse_scales: usize,
    sparse_entries: usize,
    sparse_planes: usize,
}

impl MatrixRequirements {
    fn from_streamed(requirements: PackedSaltStorageRequirements) -> Self {
        Self {
            rows: requirements.rows(),
            planes: requirements.planes(),
            dense_bytes: requirements.dense_bytes(),
            sparse_scales: requirements.sparse_scales(),
            sparse_entries: requirements.sparse_entries(),
            sparse_planes: requirements.sparse_planes(),
        }
    }

    fn from_owned_rows(rows: &[PackedSaltRow]) -> Result<Self, NnError> {
        let mut requirements = Self {
            rows: rows.len(),
            planes: 0,
            dense_bytes: 0,
            sparse_scales: 0,
            sparse_entries: 0,
            sparse_planes: 0,
        };
        for row in rows {
            requirements.planes =
                checked_count_add(requirements.planes, row.plane_count(), "SALT plane count")?;
            for plane in row.planes() {
                match plane {
                    PlaneRepr::Dense(bytes) => {
                        requirements.dense_bytes = checked_count_add(
                            requirements.dense_bytes,
                            bytes.len(),
                            "SALT dense bytes",
                        )?;
                    }
                    PlaneRepr::Sparse(sparse) => {
                        requirements.sparse_scales = checked_count_add(
                            requirements.sparse_scales,
                            sparse.scales.len(),
                            "SALT sparse scales",
                        )?;
                        requirements.sparse_entries = checked_count_add(
                            requirements.sparse_entries,
                            sparse.idx.len(),
                            "SALT sparse entries",
                        )?;
                        requirements.sparse_planes =
                            checked_count_add(requirements.sparse_planes, 1, "SALT sparse planes")?;
                    }
                }
            }
        }
        Ok(requirements)
    }
}

pub(crate) struct PackedSaltMatrixBuilder {
    n_out: usize,
    k_in: usize,
    expected: MatrixRequirements,
    rows: Vec<RowMeta>,
    planes: Vec<PlaneMeta>,
    dense_bytes: Vec<u8>,
    sparse_scales: Vec<f16>,
    sparse_entries: Vec<u32>,
    sparse_plane_count: usize,
}

impl PackedSaltMatrixBuilder {
    pub(crate) fn from_streamed(
        n_out: usize,
        k_in: usize,
        requirements: PackedSaltStorageRequirements,
    ) -> Result<Self, NnError> {
        Self::new(n_out, k_in, MatrixRequirements::from_streamed(requirements))
    }

    fn new(n_out: usize, k_in: usize, expected: MatrixRequirements) -> Result<Self, NnError> {
        if n_out == 0 || expected.rows != n_out {
            return Err(NnError::Shape {
                expected: n_out.max(1),
                got: expected.rows,
            });
        }
        if k_in == 0 || u32::try_from(k_in).is_err() {
            return Err(NnError::Shape {
                expected: if k_in == 0 { 1 } else { u32::MAX as usize },
                got: k_in,
            });
        }
        if expected.sparse_planes > expected.planes {
            return Err(NnError::Backend(
                "SALT sparse-plane requirement exceeds total planes".to_owned(),
            ));
        }

        let mut rows = Vec::new();
        let mut planes = Vec::new();
        let mut dense_bytes = Vec::new();
        let mut sparse_scales = Vec::new();
        let mut sparse_entries = Vec::new();
        try_reserve_exact(&mut rows, expected.rows, "SALT row metadata")?;
        try_reserve_exact(&mut planes, expected.planes, "SALT plane metadata")?;
        try_reserve_exact(&mut dense_bytes, expected.dense_bytes, "SALT dense arena")?;
        try_reserve_exact(
            &mut sparse_scales,
            expected.sparse_scales,
            "SALT sparse-scale arena",
        )?;
        try_reserve_exact(
            &mut sparse_entries,
            expected.sparse_entries,
            "SALT sparse-entry arena",
        )?;
        Ok(Self {
            n_out,
            k_in,
            expected,
            rows,
            planes,
            dense_bytes,
            sparse_scales,
            sparse_entries,
            sparse_plane_count: 0,
        })
    }

    pub(crate) fn push_ref(&mut self, row: PackedSaltRowRef<'_>) -> Result<(), NnError> {
        if row.k() != self.k_in {
            return Err(NnError::Shape {
                expected: self.k_in,
                got: row.k(),
            });
        }
        if self.rows.len() == self.expected.rows {
            return Err(NnError::Shape {
                expected: self.expected.rows,
                got: self.rows.len() + 1,
            });
        }

        let mut add_dense = 0usize;
        let mut add_scales = 0usize;
        let mut add_entries = 0usize;
        let mut add_sparse_planes = 0usize;
        for plane in row.planes() {
            if let Some(bytes) = plane.dense_bytes() {
                validate_dense_scales(bytes)?;
                add_dense = checked_count_add(add_dense, bytes.len(), "SALT dense bytes")?;
            } else if let Some(sparse) = plane.sparse() {
                if sparse.scales().any(|scale| !scale.is_finite()) {
                    return Err(NnError::Backend(
                        "SALT sparse plane contains a non-finite scale".to_owned(),
                    ));
                }
                add_scales =
                    checked_count_add(add_scales, sparse.scale_count(), "SALT sparse scales")?;
                add_entries =
                    checked_count_add(add_entries, sparse.entry_count(), "SALT sparse entries")?;
                add_sparse_planes = checked_count_add(add_sparse_planes, 1, "SALT sparse planes")?;
            }
        }
        self.preflight(
            row.plane_count(),
            add_dense,
            add_scales,
            add_entries,
            add_sparse_planes,
        )?;

        let plane_start = self.planes.len();
        for plane in row.planes() {
            if let Some(bytes) = plane.dense_bytes() {
                let byte_offset = self.dense_bytes.len();
                self.dense_bytes.extend_from_slice(bytes);
                self.planes.push(PlaneMeta::Dense { byte_offset });
            } else if let Some(sparse) = plane.sparse() {
                let scale_offset = self.sparse_scales.len();
                let entry_offset = self.sparse_entries.len();
                self.sparse_scales.extend(sparse.scales());
                self.sparse_entries.extend(sparse.encoded_entries());
                self.planes.push(PlaneMeta::Sparse {
                    scale_offset,
                    entry_offset,
                    entry_len: sparse.entry_count(),
                });
            }
        }
        self.sparse_plane_count += add_sparse_planes;
        self.rows.push(RowMeta {
            plane_start,
            plane_len: row.plane_count(),
        });
        Ok(())
    }

    fn push_owned(&mut self, row: &PackedSaltRow) -> Result<(), NnError> {
        if row.k() != self.k_in {
            return Err(NnError::Shape {
                expected: self.k_in,
                got: row.k(),
            });
        }
        if self.rows.len() == self.expected.rows {
            return Err(NnError::Shape {
                expected: self.expected.rows,
                got: self.rows.len() + 1,
            });
        }
        let mut add_dense = 0usize;
        let mut add_scales = 0usize;
        let mut add_entries = 0usize;
        let mut add_sparse_planes = 0usize;
        for plane in row.planes() {
            match plane {
                PlaneRepr::Dense(bytes) => {
                    validate_dense_scales(bytes)?;
                    add_dense = checked_count_add(add_dense, bytes.len(), "SALT dense bytes")?;
                }
                PlaneRepr::Sparse(sparse) => {
                    if sparse.scales.iter().any(|scale| !scale.is_finite()) {
                        return Err(NnError::Backend(
                            "SALT sparse plane contains a non-finite scale".to_owned(),
                        ));
                    }
                    add_scales =
                        checked_count_add(add_scales, sparse.scales.len(), "SALT sparse scales")?;
                    add_entries =
                        checked_count_add(add_entries, sparse.idx.len(), "SALT sparse entries")?;
                    add_sparse_planes =
                        checked_count_add(add_sparse_planes, 1, "SALT sparse planes")?;
                }
            }
        }
        self.preflight(
            row.plane_count(),
            add_dense,
            add_scales,
            add_entries,
            add_sparse_planes,
        )?;

        let plane_start = self.planes.len();
        for plane in row.planes() {
            match plane {
                PlaneRepr::Dense(bytes) => {
                    let byte_offset = self.dense_bytes.len();
                    self.dense_bytes.extend_from_slice(bytes);
                    self.planes.push(PlaneMeta::Dense { byte_offset });
                }
                PlaneRepr::Sparse(sparse) => {
                    let scale_offset = self.sparse_scales.len();
                    let entry_offset = self.sparse_entries.len();
                    self.sparse_scales.extend_from_slice(&sparse.scales);
                    self.sparse_entries
                        .extend(sparse.idx.iter().zip(&sparse.sign).map(|(&index, &sign)| {
                            if sign < 0 {
                                index | SPARSE_SIGN_BIT
                            } else {
                                index
                            }
                        }));
                    self.planes.push(PlaneMeta::Sparse {
                        scale_offset,
                        entry_offset,
                        entry_len: sparse.idx.len(),
                    });
                }
            }
        }
        self.sparse_plane_count += add_sparse_planes;
        self.rows.push(RowMeta {
            plane_start,
            plane_len: row.plane_count(),
        });
        Ok(())
    }

    fn preflight(
        &self,
        planes: usize,
        dense_bytes: usize,
        sparse_scales: usize,
        sparse_entries: usize,
        sparse_planes: usize,
    ) -> Result<(), NnError> {
        require_within(
            self.planes.len(),
            planes,
            self.expected.planes,
            "SALT plane metadata",
        )?;
        require_within(
            self.dense_bytes.len(),
            dense_bytes,
            self.expected.dense_bytes,
            "SALT dense arena",
        )?;
        require_within(
            self.sparse_scales.len(),
            sparse_scales,
            self.expected.sparse_scales,
            "SALT sparse-scale arena",
        )?;
        require_within(
            self.sparse_entries.len(),
            sparse_entries,
            self.expected.sparse_entries,
            "SALT sparse-entry arena",
        )?;
        require_within(
            self.sparse_plane_count,
            sparse_planes,
            self.expected.sparse_planes,
            "SALT sparse-plane count",
        )
    }

    pub(crate) fn finish(self) -> Result<PackedSaltMatrix, NnError> {
        let actual = MatrixRequirements {
            rows: self.rows.len(),
            planes: self.planes.len(),
            dense_bytes: self.dense_bytes.len(),
            sparse_scales: self.sparse_scales.len(),
            sparse_entries: self.sparse_entries.len(),
            sparse_planes: self.sparse_plane_count,
        };
        if actual != self.expected {
            return Err(NnError::Backend(format!(
                "SALT packed storage requirements changed: expected {:?}, got {:?}",
                self.expected, actual
            )));
        }
        Ok(PackedSaltMatrix {
            n_out: self.n_out,
            k_in: self.k_in,
            sparse_plane_count: self.sparse_plane_count,
            storage: Arc::new(PackedSaltStorage {
                rows: self.rows,
                planes: self.planes,
                dense_bytes: self.dense_bytes,
                sparse_scales: self.sparse_scales,
                sparse_entries: self.sparse_entries,
            }),
        })
    }
}

fn try_reserve_exact<T>(target: &mut Vec<T>, elements: usize, label: &str) -> Result<(), NnError> {
    target.try_reserve_exact(elements).map_err(|_| {
        NnError::Backend(format!(
            "allocate {label}: {} bytes",
            elements.saturating_mul(size_of::<T>())
        ))
    })
}

fn checked_count_add(current: usize, additional: usize, label: &str) -> Result<usize, NnError> {
    current
        .checked_add(additional)
        .ok_or_else(|| NnError::Backend(format!("{label} overflows usize")))
}

fn require_within(
    current: usize,
    additional: usize,
    expected: usize,
    label: &str,
) -> Result<(), NnError> {
    let actual = checked_count_add(current, additional, label)?;
    if actual > expected {
        Err(NnError::Backend(format!(
            "{label} exceeds validated requirement: {actual} > {expected}"
        )))
    } else {
        Ok(())
    }
}

fn validate_dense_scales(bytes: &[u8]) -> Result<(), NnError> {
    if bytes.chunks_exact(TQ2_0_BLOCK_BYTES).any(|block| {
        !f16::from_bits(u16::from_le_bytes([
            block[TQ2_0_BLOCK_BYTES - 2],
            block[TQ2_0_BLOCK_BYTES - 1],
        ]))
        .is_finite()
    }) {
        Err(NnError::Backend(
            "SALT dense plane contains a non-finite scale".to_owned(),
        ))
    } else {
        Ok(())
    }
}

impl PackedSaltMatrix {
    pub(crate) fn new(
        rows: Vec<PackedSaltRow>,
        n_out: usize,
        k_in: usize,
    ) -> Result<Self, NnError> {
        let requirements = MatrixRequirements::from_owned_rows(&rows)?;
        let mut builder = PackedSaltMatrixBuilder::new(n_out, k_in, requirements)?;
        for row in &rows {
            builder.push_owned(row)?;
        }
        builder.finish()
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
        self.storage
            .dense_bytes
            .len()
            .saturating_add(
                self.storage
                    .sparse_scales
                    .len()
                    .saturating_mul(size_of::<f16>()),
            )
            .saturating_add(
                self.storage
                    .sparse_entries
                    .len()
                    .saturating_mul(size_of::<u32>()),
            )
    }

    /// Allocated bytes in the shared backing arenas.
    ///
    /// Clones share these arenas, so summing this value across clones double-counts storage.
    pub(crate) fn resident_bytes(&self) -> usize {
        self.storage
            .dense_bytes
            .capacity()
            .saturating_add(
                self.storage
                    .sparse_scales
                    .capacity()
                    .saturating_mul(size_of::<f16>()),
            )
            .saturating_add(
                self.storage
                    .sparse_entries
                    .capacity()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(
                self.storage
                    .rows
                    .capacity()
                    .saturating_mul(size_of::<RowMeta>()),
            )
            .saturating_add(
                self.storage
                    .planes
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
        let row = self.storage.rows[row];
        let start_col = block * QK_K;
        let logical_len = (self.k_in - start_col).min(QK_K);
        for plane in &self.storage.planes[row.plane_start..row.plane_start + row.plane_len] {
            match *plane {
                PlaneMeta::Dense { byte_offset } => {
                    let start = byte_offset + block * TQ2_0_BLOCK_BYTES;
                    let bytes = &self.storage.dense_bytes[start..start + TQ2_0_BLOCK_BYTES];
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
                    let scale = self.storage.sparse_scales[scale_offset + block].to_f32();
                    let entries =
                        &self.storage.sparse_entries[entry_offset..entry_offset + entry_len];
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use half::f16;
    use tritium_core::Trit;
    use tritium_format::{
        DEFAULT_SPARSE_RESIDUAL_DENSITY, PackedSaltRowRef, SaltBundleIndex, SaltBundleReader,
        SaltRow, num_blocks, pack_salt_row, pack_tq2_0_row, write_progressive_salt_bundle,
    };

    use super::*;

    fn plane(k: usize, seed: usize, stride: Option<usize>) -> Vec<u8> {
        let trits = (0..k)
            .map(|index| {
                let value = match stride {
                    Some(stride) if !index.is_multiple_of(stride) => 0,
                    _ => ((index + seed) % 3) as i8 - 1,
                };
                Trit::from_i8(value).unwrap()
            })
            .collect::<Vec<_>>();
        let scales = vec![f16::from_f32(0.25); num_blocks(k)];
        let mut bytes = vec![0; num_blocks(k) * TQ2_0_BLOCK_BYTES];
        pack_tq2_0_row(&trits, &scales, &mut bytes).unwrap();
        bytes
    }

    #[test]
    fn streamed_builder_matches_owned_and_clone_shares_arenas() {
        let rows = vec![
            SaltRow {
                k: 269,
                planes: vec![plane(269, 1, None), plane(269, 2, Some(64))],
            },
            SaltRow {
                k: 269,
                planes: Vec::new(),
            },
        ];
        let bytes = write_progressive_salt_bundle(
            &[("w", rows.as_slice())],
            DEFAULT_SPARSE_RESIDUAL_DENSITY,
        )
        .unwrap();
        let owned_rows = SaltBundleIndex::new(&bytes)
            .unwrap()
            .tensor("w")
            .unwrap()
            .decode_packed()
            .unwrap()
            .salt_rows;
        let owned = PackedSaltMatrix::new(owned_rows, 2, 269).unwrap();

        let mut reader = SaltBundleReader::new_strict(Cursor::new(bytes)).unwrap();
        let requirements = reader.tensor_info("w").unwrap().storage_requirements();
        let mut builder = PackedSaltMatrixBuilder::from_streamed(2, 269, requirements).unwrap();
        let pointers = (
            builder.rows.as_ptr(),
            builder.planes.as_ptr(),
            builder.dense_bytes.as_ptr(),
            builder.sparse_scales.as_ptr(),
            builder.sparse_entries.as_ptr(),
        );
        reader
            .visit_packed_tensor("w", |row| builder.push_ref(row).unwrap())
            .unwrap();
        assert_eq!(
            pointers,
            (
                builder.rows.as_ptr(),
                builder.planes.as_ptr(),
                builder.dense_bytes.as_ptr(),
                builder.sparse_scales.as_ptr(),
                builder.sparse_entries.as_ptr(),
            )
        );
        let streamed = builder.finish().unwrap();

        assert_eq!(streamed.packed_bytes(), owned.packed_bytes());
        assert_eq!(streamed.sparse_plane_count(), owned.sparse_plane_count());
        let act = (0..269)
            .map(|index| (index as f32 * 0.013).sin())
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; 2];
        let mut actual = vec![0.0; 2];
        owned.project_exact(&act, &mut expected).unwrap();
        streamed.project_exact(&act, &mut actual).unwrap();
        assert_eq!(actual, expected);

        let cloned = streamed.clone();
        assert!(Arc::ptr_eq(&streamed.storage, &cloned.storage));
    }

    #[test]
    fn builder_preflight_rejects_nonfinite_scale_without_mutation() {
        let row = SaltRow {
            k: 256,
            planes: vec![plane(256, 1, None)],
        };
        let mut encoded = pack_salt_row(&row).unwrap();
        encoded[10 + 64..10 + 66].copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
        let row = PackedSaltRowRef::parse(&encoded).unwrap();
        let expected = MatrixRequirements {
            rows: 1,
            planes: 1,
            dense_bytes: TQ2_0_BLOCK_BYTES,
            sparse_scales: 0,
            sparse_entries: 0,
            sparse_planes: 0,
        };
        let mut builder = PackedSaltMatrixBuilder::new(1, 256, expected).unwrap();
        let pointers = (
            builder.rows.as_ptr(),
            builder.planes.as_ptr(),
            builder.dense_bytes.as_ptr(),
            builder.sparse_scales.as_ptr(),
            builder.sparse_entries.as_ptr(),
        );
        assert!(builder.push_ref(row).is_err());
        assert!(builder.rows.is_empty());
        assert!(builder.planes.is_empty());
        assert!(builder.dense_bytes.is_empty());
        assert_eq!(
            pointers,
            (
                builder.rows.as_ptr(),
                builder.planes.as_ptr(),
                builder.dense_bytes.as_ptr(),
                builder.sparse_scales.as_ptr(),
                builder.sparse_entries.as_ptr(),
            )
        );
    }
}
