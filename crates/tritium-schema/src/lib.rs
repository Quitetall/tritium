//! Canonical, dependency-free schema vocabulary for Tritium artifacts.
//!
//! This crate deliberately separates semantic values from their wire
//! encodings. Encodings, law admission limits, and generated projections are
//! introduced by later ADR 0044 phases; callers must not infer wire tags from
//! Rust discriminants.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

/// Precision used to store additive plane scales.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ScalePrecision {
    /// IEEE binary16.
    F16,
    /// Brain floating-point 16-bit.
    Bf16,
    /// IEEE binary32.
    F32,
}

/// Grouping axis to which an additive scale is anchored.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ScaleAnchor {
    /// One scale per declared group.
    Group,
    /// One scale per matrix row.
    Row,
    /// One scale for the complete tensor.
    Tensor,
    /// One scale per output channel.
    Channel,
}

/// Relationship between scales of successive additive planes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PlaneRelation {
    /// Every plane has an independently fitted scale.
    Free,
    /// Plane `p` has scale `s0 * (num / den)^p`.
    Tied {
        /// Numerator of the positive scale ratio.
        num: u32,
        /// Denominator of the positive scale ratio.
        den: u32,
    },
}

/// Scale law represented as orthogonal anchor, relation, and storage precision.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ScaleLaw {
    /// Scale grouping axis.
    pub anchor: ScaleAnchor,
    /// Cross-plane scale relationship.
    pub relation: PlaneRelation,
    /// Scale storage precision.
    pub precision: ScalePrecision,
}

impl ScaleLaw {
    /// Returns whether a tied law has a valid, positive ratio.
    #[must_use]
    pub const fn has_valid_ratio(self) -> bool {
        match self.relation {
            PlaneRelation::Free => true,
            PlaneRelation::Tied { num, den } => num > 0 && den > 0,
        }
    }
}

/// Input-axis transform associated with an additive tensor.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Basis {
    /// No transform.
    Identity,
    /// Normalized, self-inverse Walsh–Hadamard transform.
    Hadamard {
        /// Power-of-two transform block width.
        block: u32,
    },
    /// Deterministic signed Walsh–Hadamard transform.
    SignedRht {
        /// Power-of-two transform block width.
        block: u32,
        /// Seed used to derive signs.
        seed: u64,
        /// Stable domain identifier for sign derivation.
        domain: u32,
    },
}

/// Compact code used for each additive ternary plane.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PlaneCodec {
    /// Two-bit ternary packing.
    D2,
    /// Three-bit ternary packing.
    B3,
    /// Structured sparse 3-of-4 encoding.
    S34,
}

/// Optional entropy transport applied outside the tensor's plane codec.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Transport {
    /// Store codec bytes directly.
    Raw,
    /// Joint Huffman coding with the specified block width.
    JointHuffman {
        /// Number of symbols grouped into one entropy-coding block.
        block: u32,
    },
}

/// Uniform or per-tile plane-count allocation policy.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PlaneAllocation {
    /// Every tile uses the tensor's maximum plane count.
    Uniform,
    /// Each tile records its own plane count.
    PerTile,
}

/// Semantic layout parameters for an additive tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AdditiveLayout {
    /// Number of matrix rows.
    pub rows: u64,
    /// Number of coefficients per row.
    pub cols: u64,
    /// Scale group width.
    pub group: u16,
    /// Maximum plane count represented by this tensor.
    pub max_planes: u8,
    /// Plane count allocation mode.
    pub allocation: PlaneAllocation,
    /// Per-plane codec.
    pub codec: PlaneCodec,
    /// Scale law.
    pub law: ScaleLaw,
    /// Input-axis basis.
    pub basis: Basis,
    /// Outer entropy transport.
    pub transport: Transport,
}

/// Structural validation failure for an [`AdditiveLayout`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutError {
    /// Rows or columns are zero.
    EmptyDimensions,
    /// Group width is outside the currently specified aligned-tile set.
    UnsupportedGroup,
    /// The maximum plane count is zero.
    EmptyPlaneStack,
    /// A tied scale law has a zero numerator or denominator.
    InvalidScaleRatio,
    /// A basis block is not a nonzero power of two.
    InvalidBasisBlock,
    /// A basis block does not divide the row width.
    BasisBlockDoesNotDivideColumns,
    /// The entropy transport block width is zero.
    EmptyTransportBlock,
}

impl AdditiveLayout {
    /// Validate geometry and local invariants that do not depend on law admission.
    ///
    /// This intentionally does not decide whether `law` is admitted or whether
    /// `max_planes` is within that law's cap; those checks require the
    /// evidence-backed `ADMITTED_LAWS` registry introduced in the next schema
    /// step.
    pub fn validate(&self) -> Result<(), LayoutError> {
        if self.rows == 0 || self.cols == 0 {
            return Err(LayoutError::EmptyDimensions);
        }
        if !matches!(self.group, 32 | 64 | 128 | 256) {
            return Err(LayoutError::UnsupportedGroup);
        }
        if self.max_planes == 0 {
            return Err(LayoutError::EmptyPlaneStack);
        }
        if !self.law.has_valid_ratio() {
            return Err(LayoutError::InvalidScaleRatio);
        }
        let basis_block = match self.basis {
            Basis::Identity => None,
            Basis::Hadamard { block } | Basis::SignedRht { block, .. } => Some(block),
        };
        if let Some(block) = basis_block {
            if !block.is_power_of_two() {
                return Err(LayoutError::InvalidBasisBlock);
            }
            if !self.cols.is_multiple_of(u64::from(block)) {
                return Err(LayoutError::BasisBlockDoesNotDivideColumns);
            }
        }
        if let Transport::JointHuffman { block: 0 } = self.transport {
            return Err(LayoutError::EmptyTransportBlock);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdditiveLayout, Basis, LayoutError, PlaneAllocation, PlaneCodec, PlaneRelation,
        ScaleAnchor, ScaleLaw, ScalePrecision, Transport,
    };

    fn valid_layout() -> AdditiveLayout {
        AdditiveLayout {
            rows: 2,
            cols: 256,
            group: 128,
            max_planes: 3,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::D2,
            law: ScaleLaw {
                anchor: ScaleAnchor::Group,
                relation: PlaneRelation::Free,
                precision: ScalePrecision::F16,
            },
            basis: Basis::Hadamard { block: 128 },
            transport: Transport::Raw,
        }
    }

    #[test]
    fn tied_law_requires_positive_ratio_components() {
        let law = |num, den| ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Tied { num, den },
            precision: ScalePrecision::F16,
        };

        assert!(law(1, 3).has_valid_ratio());
        assert!(!law(0, 3).has_valid_ratio());
        assert!(!law(1, 0).has_valid_ratio());
    }

    #[test]
    fn layout_accepts_valid_structural_geometry() {
        assert_eq!(valid_layout().validate(), Ok(()));
    }

    #[test]
    fn layout_rejects_basis_block_that_does_not_divide_columns() {
        let mut layout = valid_layout();
        layout.basis = Basis::Hadamard { block: 64 };
        layout.cols = 96;

        assert_eq!(
            layout.validate(),
            Err(LayoutError::BasisBlockDoesNotDivideColumns)
        );
    }

    #[test]
    fn layout_rejects_invalid_ratio_and_transport_block() {
        let mut layout = valid_layout();
        layout.law.relation = PlaneRelation::Tied { num: 1, den: 0 };
        assert_eq!(layout.validate(), Err(LayoutError::InvalidScaleRatio));

        layout.law.relation = PlaneRelation::Free;
        layout.transport = Transport::JointHuffman { block: 0 };
        assert_eq!(layout.validate(), Err(LayoutError::EmptyTransportBlock));
    }
}
