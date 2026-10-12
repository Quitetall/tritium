//! Prepared legacy migration observations; no fit/publication verdict is implied.
use crate::{ModelManifest, NamedAssetDigest, OriginalSourceId, QuantSourceBinding};
use alloc::{string::String, vec::Vec};

/// Actual prepared Imported manifest and observed original companion bindings.
/// The legacy package and companion raw source occupy separate identity domains.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyModelImportPrepared {
    /// Exact original namespaced legacy container ID, not a new package ID.
    pub legacy_id: String,
    /// Actual raw dense/asset source identity; excludes the legacy ternary source.
    pub companion_source: OriginalSourceId,
    /// Complete original companion interpretations and actual raw-byte hashes.
    pub companion_tensors: Vec<QuantSourceBinding>,
    /// Complete original companion asset hashes.
    pub companion_assets: Vec<NamedAssetDigest>,
    /// Exact prepared manifest, without fabricated Direct fit/evidence lineage.
    pub manifest: ModelManifest,
}
