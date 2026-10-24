//! Candidate text-load observation vocabulary (ADR 0054, still proposed).

use crate::{ModelId, PackageId, UnknownReason};
use alloc::string::String;

/// Initial source identity bound by the actual shared text loader.
///
/// Does not resolve legacy IDs or verify fitting provenance/quality. Paths are
/// not identities and are deliberately absent from this observation.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoadedSourceIdentity {
    /// Checked experimental typed package and explicit complete level.
    Trit {
        /// Exact physical package bytes, not logical model content.
        package_id: PackageId,
        /// Candidate logical identity calculated by the validated reader.
        model_id: ModelId,
        /// Explicit selected level label; labels are not ModelId preimage data.
        level: String,
    },
    /// HF execution is supported, but its semantic content hashing is not yet.
    HuggingFace {
        /// Explicit unavailable-identity reason; never a synthetic hash/path ID.
        reason: UnknownReason,
    },
    /// Checked residual-bearing QTEA carrier, not a typed SALT level or semantic ModelId.
    Qtea {
        /// Exact physical carrier bytes, including tokenizer assets.
        package_id: PackageId,
        /// Producer's source-model assertion; not authenticated original custody.
        declared_source_model: [u8; 32],
    },
}

/// A successful shared text-load observation, not a qualification verdict.
///
/// Emitted only after tokenizer, weights, coverage/vocabulary and runner
/// construction succeed. It describes the initial load, not future mutable
/// weights/runner state, full physical memory, model quality or release approval.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelLoaded {
    /// Checked package binding, or explicitly unavailable HF identity.
    pub source: LoadedSourceIdentity,
    /// Architecture admitted by the shared model builder.
    pub architecture: String,
    /// Logical execution adapter label, not a capability inferred from its name.
    pub backend: String,
    /// Backend-reported device label; may be unknown rather than a GPU UUID.
    pub physical_device: String,
    /// Loaded embedding vocabulary rows, after tokenizer ID validation.
    pub vocab_size: u64,
    /// Loaded standard transformer layer count.
    pub layer_count: u64,
}
