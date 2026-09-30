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

#[cfg(test)]
mod tests {
    use super::{PlaneRelation, ScaleAnchor, ScaleLaw, ScalePrecision};

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
}
