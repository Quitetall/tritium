//! Prepared legacy migration observations; no fit/publication verdict is implied.
use crate::{DigestBytes, ModelManifest, NamedAssetDigest, OriginalSourceId, QuantSourceBinding};
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

/// Compact preparation bound to actual full canonical manifest bytes.
/// No embedded count vectors or fit/publication/source-truth verdict is implied.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacyModelImportBoundPrepared {
    /// Actual original namespaced legacy container identity.
    pub legacy_id: String,
    /// Actual observed raw dense/asset source identity.
    pub companion_source: OriginalSourceId,
    /// Complete companion interpretations and raw-byte content identities.
    pub companion_tensors: Vec<QuantSourceBinding>,
    /// Complete companion asset content identities.
    pub companion_assets: Vec<NamedAssetDigest>,
    /// Actual encoded canonical manifest byte length, not a sizing bound.
    pub manifest_bytes: u64,
    /// Plain BLAKE3 of the exact canonical manifest CBOR, not a package/model ID.
    pub manifest_digest: DigestBytes,
}

/// Canonical legacy schema-v3 HF asset names and encoded byte ceilings.
/// This catalog describes the old bundle, not the new model format.
pub const LEGACY_SALT_V2_HF_ASSETS: [(&str, u64); 8] = [
    ("chat_template.jinja", 1_048_576),
    ("config.json", 1_048_576),
    ("configuration.json", 65_536),
    ("generation_config.json", 1_048_576),
    ("merges.txt", 8_388_608),
    ("tokenizer.json", 33_554_432),
    ("tokenizer_config.json", 1_048_576),
    ("vocab.json", 16_777_216),
];

/// Original provisional schema-v3 bundle descriptor, shared with the legacy loader.
/// Identity strings retain their legacy domains; deserialization authenticates nothing.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacySaltV2BundleDescriptor {
    /// Original schema version (only 3 is admitted).
    pub schema_version: u8,
    /// Exact legacy bundle kind.
    pub artifact_kind: String,
    /// Original provisional completeness status, never promoted by import.
    pub complete_model: bool,
    /// Original SALT codec name.
    pub packing: String,
    /// Original completion identity.
    pub completion_id: String,
    /// Original campaign identity.
    pub campaign_id: String,
    /// Original admission identity.
    pub admission_id: String,
    /// Original selection identity.
    pub selection_id: String,
    /// Original claimed model identity, not independent authentication.
    pub source_model_id: String,
    /// Original source revision.
    pub source_revision: String,
    /// Original official-identity status.
    pub source_identity_status: String,
    /// Original authentication claim, not verified by migration.
    pub official_payload_authenticated: bool,
    /// Exact preserved-container descriptor.
    pub preserved: LegacySaltV2PreservedDescriptor,
    /// Canonical complete HF asset catalog.
    pub hf_assets: Vec<LegacySaltV2AssetDescriptor>,
    /// Both legacy quality profile descriptors.
    pub profiles: LegacySaltV2ProfileSet,
}

/// Original preserved safetensors container ledger.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacySaltV2PreservedDescriptor {
    /// Canonical preserved.safetensors basename.
    pub file: String,
    /// Original namespaced container identity.
    pub package_id: String,
    /// Complete preserved tensor count.
    pub tensors: u64,
    /// Exact raw data section size.
    pub payload_bytes: u64,
    /// Exact encoded container size.
    pub serialized_bytes: u64,
}

/// Original HF asset physical identity and size.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacySaltV2AssetDescriptor {
    /// Canonical asset basename.
    pub file: String,
    /// Original namespaced container identity.
    pub package_id: String,
    /// Exact encoded file size.
    pub bytes: u64,
}

/// Both named legacy profiles; selecting one does not admit the other payload.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacySaltV2ProfileSet {
    /// Original compact profile.
    #[cfg_attr(feature = "serde", serde(rename = "compact-v1"))]
    pub compact: LegacySaltV2ProfileDescriptor,
    /// Original near-lossless profile.
    #[cfg_attr(feature = "serde", serde(rename = "near-lossless-v1"))]
    pub near_lossless: LegacySaltV2ProfileDescriptor,
}

/// Original SALT V2 profile physical ledger.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LegacySaltV2ProfileDescriptor {
    /// Canonical profile basename.
    pub file: String,
    /// Original namespaced container identity.
    pub package_id: String,
    /// Exact encoded container size.
    pub serialized_bytes: u64,
    /// Original indexed resident-byte ledger, not measured process RSS.
    pub resident_bytes: u64,
}

impl LegacySaltV2BundleDescriptor {
    /// Check the existing pinned provisional bundle contract, not file contents.
    /// # Errors
    /// Unknown version/kind/revision, empty status/identity or noncanonical ledgers/paths.
    pub fn validate(&self, pinned_revision: &str) -> Result<(), &'static str> {
        if self.schema_version != 3
            || self.artifact_kind != "qwen3.6-language-mtp-salt-v2-hf-bundle"
            || self.complete_model
            || self.source_revision != pinned_revision
            || !matches!(self.packing.as_str(), "d2" | "b3" | "s34")
        {
            return Err("not the pinned provisional schema-v3 SALT V2 bundle");
        }
        if [
            &self.completion_id,
            &self.campaign_id,
            &self.admission_id,
            &self.selection_id,
            &self.source_model_id,
            &self.source_identity_status,
        ]
        .iter()
        .any(|v| v.is_empty())
        {
            return Err("empty original bundle identity/status");
        }
        if self.preserved.file != "preserved.safetensors"
            || self.preserved.package_id.is_empty()
            || self.preserved.tensors == 0
            || self.preserved.payload_bytes == 0
            || self.preserved.serialized_bytes == 0
            || self.preserved.serialized_bytes > 64 * 1024 * 1024
        {
            return Err("noncanonical preserved descriptor");
        }
        for (profile, file) in [
            (&self.profiles.compact, "compact.tsalt2"),
            (&self.profiles.near_lossless, "near-lossless.tsalt2"),
        ] {
            if profile.file != file
                || profile.package_id.is_empty()
                || profile.serialized_bytes == 0
                || profile.resident_bytes == 0
            {
                return Err("noncanonical profile descriptor");
            }
        }
        if self.hf_assets.len() != LEGACY_SALT_V2_HF_ASSETS.len() {
            return Err("incomplete legacy HF asset catalog");
        }
        for (asset, (name, cap)) in self.hf_assets.iter().zip(LEGACY_SALT_V2_HF_ASSETS) {
            if asset.file != name
                || asset.package_id.is_empty()
                || asset.bytes == 0
                || asset.bytes > cap
            {
                return Err("noncanonical legacy HF asset descriptor");
            }
        }
        Ok(())
    }
}
