//! Canonical, dependency-free schema vocabulary for Tritium artifacts.
//!
//! This crate deliberately separates semantic values from their wire
//! encodings. Encodings, law admission limits, and generated projections are
//! introduced by later ADR 0044 phases; callers must not infer wire tags from
//! Rust discriminants.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

/// Precision used to store additive plane scales.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
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

/// One admitted scale law and the largest plane stack supported for it.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AdmittedLaw {
    /// The exact law tuple admitted by this schema version.
    pub law: ScaleLaw,
    /// Per-law maximum plane count; there is no shared/global plane cap.
    pub max_planes: u8,
}

/// Initial fail-closed law registry from ADR 0044 D2.
///
/// The three-plane caps follow the current serialized SALT/training limits
/// (`SALT_V2_MAX_PLANES` and `training_salt::MAX_PLANES`). Tensor-level I2_S
/// import is the existing single-plane representation. A new law or a larger
/// cap requires its own reference implementation and frozen conformance
/// vectors before it is added here.
pub const ADMITTED_LAWS: &[AdmittedLaw] = &[
    AdmittedLaw {
        law: ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F16,
        },
        max_planes: 3,
    },
    AdmittedLaw {
        law: ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F32,
        },
        max_planes: 3,
    },
    AdmittedLaw {
        law: ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Tied { num: 1, den: 3 },
            precision: ScalePrecision::F16,
        },
        max_planes: 3,
    },
    AdmittedLaw {
        law: ScaleLaw {
            anchor: ScaleAnchor::Tensor,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F32,
        },
        max_planes: 1,
    },
];

/// Find the admission record for an exact law tuple.
#[must_use]
pub fn admitted_law(law: ScaleLaw) -> Option<&'static AdmittedLaw> {
    ADMITTED_LAWS.iter().find(|admitted| admitted.law == law)
}

/// Input-axis transform associated with an additive tensor.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
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
        domain: u64,
    },
}

/// Compact code used for each additive ternary plane.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PlaneCodec {
    /// Two-bit ternary packing.
    D2,
    /// Base-3 packing of five ternary coefficients per byte.
    B3,
    /// Structured sparse 3-of-4 encoding.
    S34,
}

/// Optional entropy transport applied outside the tensor's plane codec.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
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
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PlaneAllocation {
    /// Every tile uses the tensor's maximum plane count.
    Uniform,
    /// Each tile records its own plane count.
    PerTile,
}

/// Canonical coefficient tile width required by ADR 0044 D4.
pub const ADDITIVE_TILE_SIZE: u16 = 256;

/// Semantic layout parameters for an additive tensor.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AdditiveLayout {
    /// Number of matrix rows.
    pub rows: u64,
    /// Number of coefficients per row.
    pub cols: u64,
    /// Canonical tile width. Currently fixed at [`ADDITIVE_TILE_SIZE`].
    pub tile: u16,
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

/// Stable identifier for a schema family and its version.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SchemaId {
    /// Major version; incompatible schemas require a new major version.
    pub major: u16,
    /// Minor version; additive compatible schema evolution increments this.
    pub minor: u16,
}

macro_rules! digest_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
        pub struct $name([u8; 32]);

        impl $name {
            /// Construct from a BLAKE3-sized digest byte array.
            #[must_use]
            pub const fn from_bytes(bytes: [u8; 32]) -> Self {
                Self(bytes)
            }

            /// Borrow the digest bytes.
            #[must_use]
            pub const fn as_bytes(&self) -> &[u8; 32] {
                &self.0
            }
        }
    };
}

digest_id!(ModelId, "Content identity for a Tritium model manifest.");
digest_id!(PackageId, "Content identity for a Tritium package.");
digest_id!(BlobId, "Content identity for stored tensor or asset bytes.");
digest_id!(
    SemanticTensorDigest,
    "Identity for a tensor's semantic values independent of its container."
);

/// Standardized reason category for an unresolved claim.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum UnknownReason {
    /// The claim has not yet been evaluated.
    NotEvaluated,
    /// Required evidence is unavailable.
    MissingEvidence,
    /// Required hardware or runtime is unavailable.
    MissingCapability,
    /// Independent review is still required.
    AwaitingIndependentReview,
    /// The producer supplied a reason not represented by a standard code.
    Other,
}

/// Outcome attached to a measured or verified claim.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Verdict {
    /// The claim passed its declared check.
    Pass,
    /// The claim failed its declared check.
    Fail,
    /// No pass/fail conclusion can be drawn; never aggregates to pass.
    Unknown {
        /// Stable reason code; free-form details belong in the evidence event.
        reason: UnknownReason,
    },
}

/// Shared metadata envelope for one typed evidence payload.
///
/// Payload schemas remain event-specific; this type supplies the common
/// version, identity, span, logical-time, and digest fields.
#[cfg(feature = "alloc")]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceEnvelope<P> {
    /// Registered event schema identifier, such as `tritium.quantize.fit`.
    pub schema: alloc::string::String,
    /// Version of this event schema.
    pub v: u16,
    /// Stable run identifier.
    pub run: alloc::string::String,
    /// Monotonic sequence number within this span, starting at zero.
    pub seq: u64,
    /// Stable span identifier.
    pub span: alloc::string::String,
    /// Parent span, if this span is nested.
    pub parent: Option<alloc::string::String>,
    /// Logical time within the run; wall-clock timing is intentionally separate.
    pub t: u64,
    /// Typed, event-specific payload.
    pub payload: P,
    /// Lowercase hexadecimal BLAKE3 digest of this event and its span predecessor.
    pub digest: alloc::string::String,
}

/// Structural validation failure for an [`AdditiveLayout`].
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutError {
    /// Rows or columns are zero.
    EmptyDimensions,
    /// Group width is outside the currently specified aligned-tile set.
    UnsupportedGroup,
    /// Tile width is not the canonical 256 coefficients.
    UnsupportedTileSize,
    /// The maximum plane count is zero.
    EmptyPlaneStack,
    /// A tied scale law has a zero numerator or denominator.
    InvalidScaleRatio,
    /// This schema version does not admit the requested scale law.
    UnsupportedLaw,
    /// The requested maximum plane count exceeds this law's own cap.
    PlaneCountExceedsLaw,
    /// A basis block is not a nonzero power of two.
    InvalidBasisBlock,
    /// A basis block does not divide the row width.
    BasisBlockDoesNotDivideColumns,
    /// The entropy transport block width is zero.
    EmptyTransportBlock,
}

impl AdditiveLayout {
    /// Validate geometry, basis, transport and the exact admitted scale law.
    ///
    /// The plane count is checked against this law's own admission record, not
    /// a shared global cap. Structural validation is not backend conformance
    /// or empirical model-quality qualification.
    pub fn validate(&self) -> Result<(), LayoutError> {
        if self.rows == 0 || self.cols == 0 {
            return Err(LayoutError::EmptyDimensions);
        }
        if !matches!(self.group, 32 | 64 | 128 | 256) {
            return Err(LayoutError::UnsupportedGroup);
        }
        if self.tile != ADDITIVE_TILE_SIZE {
            return Err(LayoutError::UnsupportedTileSize);
        }
        if self.max_planes == 0 {
            return Err(LayoutError::EmptyPlaneStack);
        }
        if !self.law.has_valid_ratio() {
            return Err(LayoutError::InvalidScaleRatio);
        }
        let Some(admitted) = admitted_law(self.law) else {
            return Err(LayoutError::UnsupportedLaw);
        };
        if self.max_planes > admitted.max_planes {
            return Err(LayoutError::PlaneCountExceedsLaw);
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
        ADDITIVE_TILE_SIZE, ADMITTED_LAWS, AdditiveLayout, Basis, LayoutError, PlaneAllocation,
        PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw, ScalePrecision, Transport, admitted_law,
    };

    fn valid_layout() -> AdditiveLayout {
        AdditiveLayout {
            rows: 2,
            cols: 256,
            tile: ADDITIVE_TILE_SIZE,
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

    #[test]
    fn layout_rejects_noncanonical_tile_width() {
        let mut layout = valid_layout();
        layout.tile = 128;
        assert_eq!(layout.validate(), Err(LayoutError::UnsupportedTileSize));
    }

    #[test]
    fn admission_is_exact_and_uses_per_law_plane_caps() {
        assert_eq!(ADMITTED_LAWS.len(), 4);
        for admitted in ADMITTED_LAWS {
            assert_eq!(admitted_law(admitted.law), Some(admitted));
        }

        let mut layout = valid_layout();
        layout.max_planes = 4;
        assert_eq!(layout.validate(), Err(LayoutError::PlaneCountExceedsLaw));

        layout.law.precision = ScalePrecision::Bf16;
        assert_eq!(layout.validate(), Err(LayoutError::UnsupportedLaw));
    }
}
