//! Candidate semantic identity metadata (ADR 0054); not a released hash profile.

use crate::{Basis, BlobId, PlaneCounts, ScaleLaw, ScalePrecision, SemanticTensorDigest};
use alloc::{string::String, vec::Vec};

/// Codec/transport-independent candidate tensor metadata hashed before data.
///
/// This describes the hash preimage, not a caller-supplied qualification claim.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TensorSemanticMeta {
    /// Exact raw stored floating-point payload interpretation.
    Dense {
        /// Logical row-major dimensions, excluding tensor name and package offset.
        shape: Vec<u64>,
        /// Stored precision; precision conversion is not identity equivalence.
        precision: ScalePrecision,
    },
    /// Logical trits and exact stored scale bits, independent of physical codec.
    Additive {
        /// Logical matrix rows.
        rows: u64,
        /// Logical matrix columns.
        cols: u64,
        /// Logical coefficient tile width.
        tile: u16,
        /// Logical scale group width.
        group: u16,
        /// Actual plane allocation; unused maximum capacity is not content.
        planes: PlaneCounts,
        /// Scale anchor, relation and exact stored precision.
        law: ScaleLaw,
        /// Input-axis basis, including block and deterministic sign identity.
        basis: Basis,
    },
}

/// Exact named asset-content binding in the candidate model preimage.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamedAssetDigest {
    /// Asset basename; names affect model semantics.
    pub name: String,
    /// Exact stored asset-content digest, not a shard/package identity.
    pub content: BlobId,
}

/// Named logical tensor binding in the candidate model preimage.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamedTensorDigest {
    /// Logical architecture tensor name.
    pub name: String,
    /// Logical tensor digest, never its packed blob identity.
    pub content: SemanticTensorDigest,
}

/// Per-level model content preimage; sorted inventories are derived by the reader.
///
/// Excludes level labels, fit/evidence annotations, other levels, legacy aliases,
/// shard paths and physical packaging. Content identity is not fit provenance,
/// tokenizer/architecture admission or release qualification. Profile is proposed.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSemanticMeta {
    /// Exact manifest architecture identifier.
    pub architecture: String,
    /// Every named asset content digest, sorted by name.
    pub assets: Vec<NamedAssetDigest>,
    /// Every named tensor semantic digest in one complete level, sorted by name.
    pub tensors: Vec<NamedTensorDigest>,
}
