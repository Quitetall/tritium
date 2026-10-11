//! Reference operations over decoded additive-ternary tensor views.
//!
//! Plane payload decoding belongs to `tritium-format`. This module defines the
//! no-allocation semantic view and the slow add/subtract/skip reference path
//! shared by backend conformance tests.

use tritium_schema::{
    AdditiveLayout, Basis, LayoutError, PlaneAllocation, PlaneRelation, ScaleAnchor,
};

use crate::{PerTileIndex, Trit};

/// Consumer marker: rotate activations before additive matrix multiplication.
#[derive(Clone, Copy, Debug)]
pub enum Matmul {}

/// Consumer marker: undo the stored basis before returning embedding rows.
#[derive(Clone, Copy, Debug)]
pub enum Gather {}

/// A validated tensor bound to its consumer's basis semantics.
///
/// Construct through [`AdditiveView::bind_matmul`] or
/// [`AdditiveView::bind_gather`]. The tensor and marker are private: a gather
/// binding cannot be passed to the matmul reference, or vice versa. No storage
/// is copied or allocated when binding.
///
/// ```compile_fail
/// use tritium_core::{AdditiveView, reference_embed};
/// fn wrong_consumer(view: AdditiveView<'_>) {
///     reference_embed(&view.bind_matmul(), &[0], &mut [0.0; 4]).unwrap();
/// }
/// ```
///
/// ```compile_fail
/// use tritium_core::{AdditiveView, reference_ternary_matmul};
/// fn wrong_consumer(view: AdditiveView<'_>) {
///     reference_ternary_matmul(&[0.0; 4], &view.bind_gather(), 1,
///         &mut [0.0; 4], &mut [0.0; 1]).unwrap();
/// }
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BasisBound<T, C> {
    tensor: T,
    consumer: core::marker::PhantomData<C>,
}

/// Failure while validating or executing a decoded additive tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AdditiveError {
    /// The layout failed its schema-level validation.
    InvalidLayout(LayoutError),
    /// The selected constructor or consumer does not support this allocation.
    UnsupportedAllocation,
    /// The actual plane count is zero or exceeds the tensor/law maximum.
    InvalidPlaneCount,
    /// The row-major tile count array disagrees with tensor geometry.
    TileCountMismatch {
        /// Expected number of tile counts.
        expected: usize,
        /// Supplied number of tile counts.
        got: usize,
    },
    /// A tile count exceeds its layout/law maximum.
    InvalidTilePlaneCount {
        /// Row-major tile index.
        tile: usize,
        /// Supplied tile count.
        count: u8,
        /// Maximum permitted by this layout.
        max: u8,
    },
    /// Caller-owned index storage has the wrong length.
    IndexStorageLength {
        /// Required row span count.
        expected: usize,
        /// Supplied row span count.
        got: usize,
    },
    /// The decoded ternary buffer length disagrees with tensor geometry.
    TritCountMismatch {
        /// Expected number of ternary coefficients.
        expected: usize,
        /// Supplied number of ternary coefficients.
        got: usize,
    },
    /// The decoded scale buffer length disagrees with the scale law.
    ScaleCountMismatch {
        /// Expected number of scales.
        expected: usize,
        /// Supplied number of scales.
        got: usize,
    },
    /// A scale is negative, NaN, or infinite, or a tied scale overflows.
    InvalidScale,
    /// A dimension product cannot be represented on this target.
    SizeOverflow,
    /// A requested matrix row is outside the tensor.
    RowOutOfRange,
    /// The output buffer length disagrees with tensor geometry.
    OutputLength {
        /// Expected number of output values.
        expected: usize,
        /// Supplied number of output values.
        got: usize,
    },
    /// The input/output/scratch lengths disagree with the matmul dimensions.
    MatmulShape,
    /// The basis is valid schema data but its reference implementation is not
    /// available yet.
    UnsupportedBasis,
}

/// Borrowed decoded additive ternary matrix and scale data.
///
/// Ternary coefficients are plane-major, then row-major within each plane.
/// For `Free` laws, scales are plane-major and follow the selected anchor; for
/// `Tied` laws, only the base-plane scale set is stored. Scales have already
/// been widened losslessly to `f32` from the precision named in the layout.
/// Per-tile views omit absent tiles and bind their compact order explicitly
/// through [`Self::new_per_tile`]; uniform construction never infers counts.
#[derive(Clone, Copy, Debug)]
pub struct AdditiveView<'a> {
    allocation: Allocation<'a>,
    trits: &'a [Trit],
    scales: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
enum Allocation<'a> {
    Uniform {
        layout: AdditiveLayout,
        planes: usize,
        scale_units: usize,
    },
    PerTile(PerTileIndex<'a>),
}

impl<'a> AdditiveView<'a> {
    /// Validate and construct a view over decoded, plane-major tensor data.
    pub fn new(
        layout: AdditiveLayout,
        plane_count: u8,
        trits: &'a [Trit],
        scales: &'a [f32],
    ) -> Result<Self, AdditiveError> {
        layout.validate().map_err(AdditiveError::InvalidLayout)?;
        if layout.allocation != PlaneAllocation::Uniform {
            return Err(AdditiveError::UnsupportedAllocation);
        }
        if plane_count == 0 || plane_count > layout.max_planes {
            return Err(AdditiveError::InvalidPlaneCount);
        }

        let rows = usize::try_from(layout.rows).map_err(|_| AdditiveError::SizeOverflow)?;
        let cols = usize::try_from(layout.cols).map_err(|_| AdditiveError::SizeOverflow)?;
        let coefficients = rows.checked_mul(cols).ok_or(AdditiveError::SizeOverflow)?;
        let expected_trits = coefficients
            .checked_mul(usize::from(plane_count))
            .ok_or(AdditiveError::SizeOverflow)?;
        if trits.len() != expected_trits {
            return Err(AdditiveError::TritCountMismatch {
                expected: expected_trits,
                got: trits.len(),
            });
        }

        let scale_units = match layout.law.anchor {
            ScaleAnchor::Group => {
                let groups_per_row = cols.div_ceil(usize::from(layout.group));
                rows.checked_mul(groups_per_row)
                    .ok_or(AdditiveError::SizeOverflow)?
            }
            ScaleAnchor::Row | ScaleAnchor::Channel => rows,
            ScaleAnchor::Tensor => 1,
            _ => return Err(AdditiveError::InvalidLayout(LayoutError::UnsupportedLaw)),
        };
        let expected_scales = match layout.law.relation {
            PlaneRelation::Free => scale_units
                .checked_mul(usize::from(plane_count))
                .ok_or(AdditiveError::SizeOverflow)?,
            PlaneRelation::Tied { .. } => scale_units,
            _ => return Err(AdditiveError::InvalidLayout(LayoutError::UnsupportedLaw)),
        };
        if scales.len() != expected_scales {
            return Err(AdditiveError::ScaleCountMismatch {
                expected: expected_scales,
                got: scales.len(),
            });
        }
        validate_scales(layout, usize::from(plane_count), scales)?;

        Ok(Self {
            allocation: Allocation::Uniform {
                layout,
                planes: usize::from(plane_count),
                scale_units,
            },
            trits,
            scales,
        })
    }

    /// Bind a checked per-tile index to its exact compact coefficients/scales.
    ///
    /// Borrows all storage and allocates nothing. An all-pruned index binds
    /// empty payloads with zero present planes, while retaining normal shape and
    /// basis validation. This does not imply native backend support.
    pub fn new_per_tile(
        index: PerTileIndex<'a>,
        trits: &'a [Trit],
        scales: &'a [f32],
    ) -> Result<Self, AdditiveError> {
        if trits.len() != index.trit_count() {
            return Err(AdditiveError::TritCountMismatch {
                expected: index.trit_count(),
                got: trits.len(),
            });
        }
        if scales.len() != index.scale_count() {
            return Err(AdditiveError::ScaleCountMismatch {
                expected: index.scale_count(),
                got: scales.len(),
            });
        }
        validate_scales(index.layout, index.plane_count(), scales)?;
        Ok(Self {
            allocation: Allocation::PerTile(index),
            trits,
            scales,
        })
    }

    /// Validated row-major counts, or `None` for uniform allocation.
    #[must_use]
    pub fn tile_counts(&self) -> Option<&'a [u8]> {
        match self.allocation {
            Allocation::Uniform { .. } => None,
            Allocation::PerTile(index) => Some(index.counts()),
        }
    }

    /// Coefficients for one present plane in canonical payload order.
    pub fn plane_trits(&self, plane: usize) -> Result<&'a [Trit], AdditiveError> {
        if plane >= self.plane_count() {
            return Err(AdditiveError::InvalidPlaneCount);
        }
        let range = if let Allocation::PerTile(index) = self.allocation {
            index.plane_range(plane)
        } else {
            let coefficients = self.layout().rows as usize * self.layout().cols as usize;
            plane * coefficients..(plane + 1) * coefficients
        };
        Ok(&self.trits[range])
    }

    /// Return the layout carried by this validated view.
    #[must_use]
    pub const fn layout(&self) -> AdditiveLayout {
        match self.allocation {
            Allocation::Uniform { layout, .. } => layout,
            Allocation::PerTile(index) => index.layout,
        }
    }

    /// Number of present planes; the actual maximum count for per-tile views.
    /// May be zero for an all-pruned per-tile tensor.
    #[must_use]
    pub const fn plane_count(&self) -> usize {
        match self.allocation {
            Allocation::Uniform { planes, .. } => planes,
            Allocation::PerTile(index) => index.plane_count(),
        }
    }

    /// Return the validated plane-major ternary coefficients.
    #[must_use]
    pub const fn trits(&self) -> &'a [Trit] {
        self.trits
    }

    /// Return the stored scale values in their canonical plane/unit order.
    #[must_use]
    pub const fn scales(&self) -> &'a [f32] {
        self.scales
    }

    /// Bind this tensor to matmul, whose activations use the forward basis.
    #[must_use]
    pub const fn bind_matmul(self) -> BasisBound<Self, Matmul> {
        BasisBound {
            tensor: self,
            consumer: core::marker::PhantomData,
        }
    }

    /// Bind this tensor to gather, whose decoded rows use the inverse basis.
    #[must_use]
    pub const fn bind_gather(self) -> BasisBound<Self, Gather> {
        BasisBound {
            tensor: self,
            consumer: core::marker::PhantomData,
        }
    }

    /// Decode one output row into `out`, using the declared additive scale law.
    ///
    /// This returns the stored-basis row, not an original-coordinate embedding.
    /// Embedding consumers must use [`Self::bind_gather`] and [`reference_embed`].
    pub fn dequant_row_into(&self, row: usize, out: &mut [f32]) -> Result<(), AdditiveError> {
        let rows = usize::try_from(self.layout().rows).map_err(|_| AdditiveError::SizeOverflow)?;
        let cols = usize::try_from(self.layout().cols).map_err(|_| AdditiveError::SizeOverflow)?;
        if row >= rows {
            return Err(AdditiveError::RowOutOfRange);
        }
        if out.len() != cols {
            return Err(AdditiveError::OutputLength {
                expected: cols,
                got: out.len(),
            });
        }
        out.fill(0.0);
        if let Allocation::PerTile(index) = self.allocation {
            let group = usize::from(self.layout().group);
            for plane in 0..self.plane_count() {
                for tile in index.row_tiles(plane, row) {
                    for offset in 0..tile.len {
                        let scale_index = tile.scale
                            + if self.layout().law.anchor == ScaleAnchor::Group {
                                offset / group
                            } else {
                                0
                            };
                        out[tile.col + offset] += self.trits[tile.trit + offset].to_f32()
                            * self.stored_scale(plane, scale_index)?;
                    }
                }
            }
            return Ok(());
        }
        for plane in 0..self.plane_count() {
            let trit_start = plane * rows * cols + row * cols;
            for (col, value) in out.iter_mut().enumerate() {
                let scale = self.scale(plane, self.scale_unit(row, col)?)?;
                *value += self.trits[trit_start + col].to_f32() * scale;
            }
        }
        Ok(())
    }

    fn scale_unit(&self, row: usize, col: usize) -> Result<usize, AdditiveError> {
        match self.layout().law.anchor {
            ScaleAnchor::Group => {
                let cols = self.layout().cols as usize;
                let groups_per_row = cols.div_ceil(usize::from(self.layout().group));
                Ok(row * groups_per_row + col / usize::from(self.layout().group))
            }
            ScaleAnchor::Row | ScaleAnchor::Channel => Ok(row),
            ScaleAnchor::Tensor => Ok(0),
            _ => Err(AdditiveError::InvalidLayout(LayoutError::UnsupportedLaw)),
        }
    }

    fn scale(&self, plane: usize, unit: usize) -> Result<f32, AdditiveError> {
        let Allocation::Uniform { scale_units, .. } = self.allocation else {
            unreachable!("uniform scale lookup on a compact view")
        };
        let stored = if self.layout().law.relation == PlaneRelation::Free {
            plane * scale_units + unit
        } else {
            unit
        };
        self.stored_scale(plane, stored)
    }

    fn stored_scale(&self, plane: usize, stored: usize) -> Result<f32, AdditiveError> {
        match self.layout().law.relation {
            PlaneRelation::Free => Ok(self.scales[stored]),
            PlaneRelation::Tied { num, den } => {
                let ratio = num as f32 / den as f32;
                let mut scale = self.scales[stored];
                for _ in 0..plane {
                    scale *= ratio;
                }
                if scale.is_finite() {
                    Ok(scale)
                } else {
                    Err(AdditiveError::InvalidScale)
                }
            }
            _ => Err(AdditiveError::InvalidLayout(LayoutError::UnsupportedLaw)),
        }
    }

    fn dot_row(&self, row: usize, activation: &[f32]) -> Result<f32, AdditiveError> {
        if let Allocation::PerTile(index) = self.allocation {
            return self.dot_compact_row(index, row, activation);
        }
        let cols = self.layout().cols as usize;
        let rows = self.layout().rows as usize;
        let group = usize::from(self.layout().group);
        let group_anchor = self.layout().law.anchor == ScaleAnchor::Group;
        let groups = if group_anchor {
            cols.div_ceil(group)
        } else {
            1
        };
        let mut sum = 0.0f32;
        for plane in 0..self.plane_count() {
            let plane_start = plane * rows * cols + row * cols;
            for unit in 0..groups {
                let start = if group_anchor { unit * group } else { 0 };
                let end = if group_anchor {
                    (start + group).min(cols)
                } else {
                    cols
                };
                let mut partial = 0.0f32;
                for (col, activation_value) in activation.iter().enumerate().take(end).skip(start) {
                    match self.trits[plane_start + col].get() {
                        1 => partial += activation_value,
                        -1 => partial -= activation_value,
                        _ => {}
                    }
                }
                sum += partial * self.scale(plane, self.scale_unit(row, start)?)?;
            }
        }
        Ok(sum)
    }

    fn dot_compact_row(
        &self,
        index: PerTileIndex<'_>,
        row: usize,
        activation: &[f32],
    ) -> Result<f32, AdditiveError> {
        let group = usize::from(self.layout().group);
        let tensor_anchor = self.layout().law.anchor == ScaleAnchor::Tensor;
        let mut sum = 0.0f32;
        for plane in 0..self.plane_count() {
            let mut tensor_partial = 0.0f32;
            for tile in index.row_tiles(plane, row) {
                let width = if tensor_anchor { tile.len } else { group };
                for start in (0..tile.len).step_by(width) {
                    let end = (start + width).min(tile.len);
                    let mut partial = if tensor_anchor { tensor_partial } else { 0.0 };
                    for offset in start..end {
                        match self.trits[tile.trit + offset].get() {
                            1 => partial += activation[tile.col + offset],
                            -1 => partial -= activation[tile.col + offset],
                            _ => {}
                        }
                    }
                    if tensor_anchor {
                        tensor_partial = partial;
                    } else {
                        sum += partial * self.stored_scale(plane, tile.scale + start / group)?;
                    }
                }
            }
            if tensor_anchor {
                sum += tensor_partial * self.stored_scale(plane, plane)?;
            }
        }
        Ok(sum)
    }
}

fn validate_scales(
    layout: AdditiveLayout,
    planes: usize,
    scales: &[f32],
) -> Result<(), AdditiveError> {
    if scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale < 0.0)
    {
        return Err(AdditiveError::InvalidScale);
    }
    if let PlaneRelation::Tied { num, den } = layout.law.relation {
        let ratio = num as f32 / den as f32;
        for &base_scale in scales {
            let mut plane_scale = base_scale;
            for _ in 1..planes {
                plane_scale *= ratio;
                if !plane_scale.is_finite() {
                    return Err(AdditiveError::InvalidScale);
                }
            }
        }
    }
    Ok(())
}

/// Apply the declared forward basis to one activation vector in place.
///
/// Hadamard blocks are normalized and use the same butterfly order as the
/// existing training transform. SignedRht uses SplitMix64-derived signs with
/// the signed diagonal applied before the normalized Hadamard transform.
pub fn apply_basis(values: &mut [f32], basis: Basis) -> Result<(), AdditiveError> {
    match basis {
        Basis::Identity => Ok(()),
        Basis::Hadamard { block } => apply_hadamard_blocks(values, block),
        Basis::SignedRht {
            block,
            seed,
            domain,
        } => {
            validate_basis_blocks(values.len(), block)?;
            apply_signs(values, seed, domain);
            apply_hadamard_blocks(values, block)
        }
        _ => Err(AdditiveError::UnsupportedBasis),
    }
}

/// Undo the declared input-axis basis on a decoded embedding row.
///
/// SignedRht's forward transform is `H D`, so its inverse is `D H`:
/// normalized Hadamard first, then the same deterministic signs. Reapplying
/// the forward transform is incorrect in general. Invalid block geometry is
/// rejected before mutating `values`.
pub fn apply_inverse_basis(values: &mut [f32], basis: Basis) -> Result<(), AdditiveError> {
    match basis {
        Basis::Identity => Ok(()),
        Basis::Hadamard { block } => apply_hadamard_blocks(values, block),
        Basis::SignedRht {
            block,
            seed,
            domain,
        } => {
            apply_hadamard_blocks(values, block)?;
            apply_signs(values, seed, domain);
            Ok(())
        }
        _ => Err(AdditiveError::UnsupportedBasis),
    }
}

fn apply_signs(values: &mut [f32], seed: u64, domain: u64) {
    for (index, value) in values.iter_mut().enumerate() {
        if signed_rht_negative(seed, domain, index as u64) {
            *value = -*value;
        }
    }
}

/// Gather original-coordinate embedding rows from a gather-bound tensor.
///
/// `out` contains exactly `ids.len() * cols` values, preserving ID order and
/// duplicates. All IDs and output geometry are checked before any output is
/// written. Empty gathers are valid. This path allocates no scratch memory.
pub fn reference_embed(
    weights: &BasisBound<AdditiveView<'_>, Gather>,
    ids: &[usize],
    out: &mut [f32],
) -> Result<(), AdditiveError> {
    let weights = &weights.tensor;
    let rows = usize::try_from(weights.layout().rows).map_err(|_| AdditiveError::SizeOverflow)?;
    let cols = usize::try_from(weights.layout().cols).map_err(|_| AdditiveError::SizeOverflow)?;
    let expected = ids
        .len()
        .checked_mul(cols)
        .ok_or(AdditiveError::SizeOverflow)?;
    if out.len() != expected {
        return Err(AdditiveError::OutputLength {
            expected,
            got: out.len(),
        });
    }
    if ids.iter().any(|&row| row >= rows) {
        return Err(AdditiveError::RowOutOfRange);
    }
    for (&row, output) in ids.iter().zip(out.chunks_exact_mut(cols)) {
        weights.dequant_row_into(row, output)?;
        apply_inverse_basis(output, weights.layout().basis)?;
    }
    Ok(())
}

/// Reference additive-ternary matrix multiplication over a validated view.
///
/// Activations are copied into caller-provided scratch, transformed by the
/// tensor basis, then accumulated plane-major and group-major in stable order.
/// The reference uses only add/subtract/skip for ternary dot products.
pub fn reference_ternary_matmul(
    activations: &[f32],
    weights: &BasisBound<AdditiveView<'_>, Matmul>,
    batch: usize,
    transformed_activations: &mut [f32],
    out: &mut [f32],
) -> Result<(), AdditiveError> {
    let weights = &weights.tensor;
    let rows = usize::try_from(weights.layout().rows).map_err(|_| AdditiveError::SizeOverflow)?;
    let cols = usize::try_from(weights.layout().cols).map_err(|_| AdditiveError::SizeOverflow)?;
    let activation_count = batch.checked_mul(cols).ok_or(AdditiveError::SizeOverflow)?;
    let output_count = batch.checked_mul(rows).ok_or(AdditiveError::SizeOverflow)?;
    if activations.len() != activation_count
        || transformed_activations.len() != activation_count
        || out.len() != output_count
    {
        return Err(AdditiveError::MatmulShape);
    }
    transformed_activations.copy_from_slice(activations);
    for activation in transformed_activations.chunks_exact_mut(cols) {
        apply_basis(activation, weights.layout().basis)?;
    }
    for batch_row in 0..batch {
        let activation = &transformed_activations[batch_row * cols..(batch_row + 1) * cols];
        for output_row in 0..rows {
            out[batch_row * rows + output_row] = weights.dot_row(output_row, activation)?;
        }
    }
    Ok(())
}

fn validate_basis_blocks(len: usize, block: u32) -> Result<usize, AdditiveError> {
    if !block.is_power_of_two() {
        return Err(AdditiveError::InvalidLayout(LayoutError::InvalidBasisBlock));
    }
    let block = usize::try_from(block).map_err(|_| AdditiveError::SizeOverflow)?;
    if !len.is_multiple_of(block) {
        return Err(AdditiveError::InvalidLayout(
            LayoutError::BasisBlockDoesNotDivideColumns,
        ));
    }
    Ok(block)
}

fn apply_hadamard_blocks(values: &mut [f32], block: u32) -> Result<(), AdditiveError> {
    let block = validate_basis_blocks(values.len(), block)?;
    for vector in values.chunks_exact_mut(block) {
        hadamard_vector(vector);
    }
    Ok(())
}

/// Whether the legacy group-local rotation rule transforms this group width.
///
/// A ragged tail is transformed only when its own length is a power of two and
/// greater than one. This does not admit ragged typed basis blocks: [`apply_basis`]
/// still requires its block width to divide the entire input axis.
#[must_use]
pub const fn group_is_rotatable(len: usize) -> bool {
    len.is_power_of_two() && len > 1
}

/// In-place normalized, self-inverse fast Walsh–Hadamard transform.
///
/// Uses the same butterflies and normalization as the typed basis path.
///
/// # Panics
/// Panics if the slice length is not a power of two (including an empty slice).
pub fn fast_hadamard(values: &mut [f32]) {
    let n = values.len();
    assert!(
        n.is_power_of_two(),
        "Hadamard needs a power-of-two length, got {n}"
    );
    hadamard_vector(values);
}

fn hadamard_vector(vector: &mut [f32]) {
    let block = vector.len();
    let mut width = 1usize;
    while width < block {
        for start in (0..block).step_by(width * 2) {
            for index in start..start + width {
                let (left, right) = (vector[index], vector[index + width]);
                vector[index] = left + right;
                vector[index + width] = left - right;
            }
        }
        width *= 2;
    }
    let normalization = hadamard_normalization(block.trailing_zeros());
    for value in vector {
        *value *= normalization;
    }
}

fn hadamard_normalization(log2_block: u32) -> f32 {
    if log2_block.is_multiple_of(2) {
        let exponent = 127 - log2_block / 2;
        f32::from_bits(exponent << 23)
    } else {
        let exponent = 126 - log2_block / 2;
        f32::from_bits((exponent << 23) | 0x0035_04f3)
    }
}

fn signed_rht_negative(seed: u64, domain: u64, index: u64) -> bool {
    // Domain tag is the ASCII token "TRITIUM1" as a big-endian integer.
    let domain_seed = splitmix64(seed ^ 0x5452_4954_4955_4d31);
    let indexed_seed = splitmix64(domain_seed ^ domain);
    // SplitMix64 finalizer keyed by a golden-ratio counter gives a stable sign
    // bit for every input coordinate without stateful/thread-order dependence.
    splitmix64(indexed_seed ^ index.wrapping_mul(0x9e37_79b9_7f4a_7c15)) & 1 != 0
}

fn splitmix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    #[test]
    fn normalization_retains_shipping_sqrt_reciprocal_bits() {
        for log2 in 0..usize::BITS {
            let len = 1usize << log2;
            let original = 1.0 / (len as f32).sqrt();
            assert_eq!(
                super::hadamard_normalization(log2).to_bits(),
                original.to_bits()
            );
        }
    }

    #[test]
    fn legacy_group_rotation_keeps_its_tail_rule() {
        for len in 0usize..=513 {
            assert_eq!(
                super::group_is_rotatable(len),
                len > 1 && len.is_power_of_two()
            );
        }
    }

    #[test]
    #[should_panic(expected = "Hadamard needs a power-of-two length, got 0")]
    fn fast_hadamard_rejects_empty_input() {
        super::fast_hadamard(&mut []);
    }

    #[test]
    #[should_panic(expected = "Hadamard needs a power-of-two length, got 3")]
    fn fast_hadamard_rejects_non_power_of_two_input() {
        super::fast_hadamard(&mut [1.0; 3]);
    }

    use super::{AdditiveError, AdditiveView, apply_basis, reference_ternary_matmul};
    use crate::Trit;
    use tritium_schema::{
        AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
        ScalePrecision, Transport,
    };

    fn layout(basis: Basis, law: ScaleLaw) -> AdditiveLayout {
        AdditiveLayout {
            rows: 1,
            cols: 4,
            tile: 256,
            group: 32,
            max_planes: 3,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::D2,
            law,
            basis,
            transport: Transport::Raw,
        }
    }

    fn free_group_law() -> ScaleLaw {
        ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F32,
        }
    }

    #[test]
    fn dequant_row_reconstructs_free_additive_planes() {
        let trits = [
            Trit::POS,
            Trit::NEG,
            Trit::ZERO,
            Trit::POS,
            Trit::ZERO,
            Trit::POS,
            Trit::NEG,
            Trit::POS,
        ];
        let scales = [2.0, 0.5];
        let view = AdditiveView::new(
            layout(Basis::Identity, free_group_law()),
            2,
            &trits,
            &scales,
        )
        .expect("valid additive view");
        let mut row = [0.0; 4];
        view.dequant_row_into(0, &mut row).expect("valid row");
        assert_eq!(row, [2.0, -1.5, -0.5, 2.5]);
    }

    #[test]
    fn matmul_matches_add_subtract_skip_reference() {
        let trits = [Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS];
        let scales = [2.0];
        let view = AdditiveView::new(
            layout(Basis::Identity, free_group_law()),
            1,
            &trits,
            &scales,
        )
        .expect("valid additive view");
        let activations = [1.0, 2.0, 3.0, 4.0];
        let mut transformed = [0.0; 4];
        let mut out = [0.0; 1];
        reference_ternary_matmul(
            &activations,
            &view.bind_matmul(),
            1,
            &mut transformed,
            &mut out,
        )
        .expect("valid reference matmul");
        assert_eq!(out, [6.0]);
    }

    #[test]
    fn normalized_hadamard_is_self_inverse() {
        let mut values = [1.0, 2.0, 3.0, 4.0];
        apply_basis(&mut values, Basis::Hadamard { block: 4 }).expect("forward basis");
        assert_eq!(values, [5.0, -1.0, -2.0, 0.0]);
        apply_basis(&mut values, Basis::Hadamard { block: 4 }).expect("inverse basis");
        for (got, expected) in values.into_iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((got - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn signed_rht_is_deterministic_and_domain_separated() {
        let mut first = [1.0, 2.0, 3.0, 4.0];
        let mut replay = first;
        let mut other_domain = first;
        let basis = Basis::SignedRht {
            block: 4,
            seed: 7,
            domain: 9,
        };
        apply_basis(&mut first, basis).expect("signed RHT");
        apply_basis(&mut replay, basis).expect("replay signed RHT");
        apply_basis(
            &mut other_domain,
            Basis::SignedRht {
                block: 4,
                seed: 7,
                domain: 10,
            },
        )
        .expect("domain-separated signed RHT");
        assert_eq!(first, [-4.0, 2.0, 3.0, 1.0]);
        assert_eq!(other_domain, [0.0, -2.0, -1.0, 5.0]);
        assert_eq!(first, replay);
    }

    #[test]
    fn tied_scale_law_expands_the_declared_geometric_ratio() {
        let law = ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Tied { num: 1, den: 3 },
            precision: ScalePrecision::F16,
        };
        let trits = [
            Trit::POS,
            Trit::POS,
            Trit::POS,
            Trit::ZERO,
            Trit::POS,
            Trit::POS,
            Trit::POS,
            Trit::ZERO,
            Trit::POS,
            Trit::POS,
            Trit::POS,
            Trit::ZERO,
        ];
        let scales = [3.0];
        let view = AdditiveView::new(layout(Basis::Identity, law), 3, &trits, &scales)
            .expect("valid tied scale view");
        let mut row = [0.0; 4];
        view.dequant_row_into(0, &mut row).expect("valid row");
        assert_eq!(row, [4.333_333_5, 4.333_333_5, 4.333_333_5, 0.0]);
    }

    #[test]
    fn view_rejects_nonfinite_scales() {
        let trits = [Trit::ZERO; 4];
        let scales = [f32::INFINITY];
        assert!(matches!(
            AdditiveView::new(
                layout(Basis::Identity, free_group_law()),
                1,
                &trits,
                &scales
            ),
            Err(AdditiveError::InvalidScale)
        ));
    }
}
