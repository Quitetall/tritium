//! Reference operations over decoded additive-ternary tensor views.
//!
//! Plane payload decoding belongs to `tritium-format`. This module defines the
//! no-allocation semantic view and the slow add/subtract/skip reference path
//! shared by backend conformance tests.

use tritium_schema::{
    AdditiveLayout, Basis, LayoutError, PlaneAllocation, PlaneRelation, ScaleAnchor,
};

use crate::Trit;

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
    /// The selected reference path does not yet support per-tile plane counts.
    UnsupportedAllocation,
    /// The actual plane count is zero or exceeds the tensor/law maximum.
    InvalidPlaneCount,
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
#[derive(Clone, Copy, Debug)]
pub struct AdditiveView<'a> {
    layout: AdditiveLayout,
    plane_count: usize,
    scale_units: usize,
    trits: &'a [Trit],
    scales: &'a [f32],
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
                for _ in 1..usize::from(plane_count) {
                    plane_scale *= ratio;
                    if !plane_scale.is_finite() {
                        return Err(AdditiveError::InvalidScale);
                    }
                }
            }
        }

        Ok(Self {
            layout,
            plane_count: usize::from(plane_count),
            scale_units,
            trits,
            scales,
        })
    }

    /// Return the layout carried by this validated view.
    #[must_use]
    pub const fn layout(&self) -> AdditiveLayout {
        self.layout
    }

    /// Number of stored planes present in this view.
    #[must_use]
    pub const fn plane_count(&self) -> usize {
        self.plane_count
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
        let rows = usize::try_from(self.layout.rows).map_err(|_| AdditiveError::SizeOverflow)?;
        let cols = usize::try_from(self.layout.cols).map_err(|_| AdditiveError::SizeOverflow)?;
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
        for plane in 0..self.plane_count {
            let trit_start = plane * rows * cols + row * cols;
            for (col, value) in out.iter_mut().enumerate() {
                let scale = self.scale(plane, self.scale_unit(row, col)?)?;
                *value += self.trits[trit_start + col].to_f32() * scale;
            }
        }
        Ok(())
    }

    fn scale_unit(&self, row: usize, col: usize) -> Result<usize, AdditiveError> {
        match self.layout.law.anchor {
            ScaleAnchor::Group => {
                let cols = self.layout.cols as usize;
                let groups_per_row = cols.div_ceil(usize::from(self.layout.group));
                Ok(row * groups_per_row + col / usize::from(self.layout.group))
            }
            ScaleAnchor::Row | ScaleAnchor::Channel => Ok(row),
            ScaleAnchor::Tensor => Ok(0),
            _ => Err(AdditiveError::InvalidLayout(LayoutError::UnsupportedLaw)),
        }
    }

    fn scale(&self, plane: usize, unit: usize) -> Result<f32, AdditiveError> {
        match self.layout.law.relation {
            PlaneRelation::Free => Ok(self.scales[plane * self.scale_units + unit]),
            PlaneRelation::Tied { num, den } => {
                let ratio = num as f32 / den as f32;
                let mut scale = self.scales[unit];
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
        let cols = self.layout.cols as usize;
        let rows = self.layout.rows as usize;
        let group = usize::from(self.layout.group);
        let group_anchor = self.layout.law.anchor == ScaleAnchor::Group;
        let groups = if group_anchor {
            cols.div_ceil(group)
        } else {
            1
        };
        let mut sum = 0.0f32;
        for plane in 0..self.plane_count {
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
    let rows = usize::try_from(weights.layout.rows).map_err(|_| AdditiveError::SizeOverflow)?;
    let cols = usize::try_from(weights.layout.cols).map_err(|_| AdditiveError::SizeOverflow)?;
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
        apply_inverse_basis(output, weights.layout.basis)?;
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
    let rows = usize::try_from(weights.layout.rows).map_err(|_| AdditiveError::SizeOverflow)?;
    let cols = usize::try_from(weights.layout.cols).map_err(|_| AdditiveError::SizeOverflow)?;
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
        apply_basis(activation, weights.layout.basis)?;
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
    Ok(())
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
