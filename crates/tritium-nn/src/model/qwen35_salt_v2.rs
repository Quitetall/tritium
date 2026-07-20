//! Seek-backed Qwen3.5-family assembly from a SALT V2 matrix package and exact
//! BF16 preserved-tensor companion.

use core::mem::size_of;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Cursor, Read};
use std::path::Path;
use std::sync::Arc;

use serde::Deserialize;
use tritium_format::{
    PackageHasher, SafeTensorsReader,
    salt_v2::SaltV2Codec,
    salt_v2_package::{SaltV2PackageReader, SaltV2Transform},
};
use tritium_spec::TernaryBackend;

use crate::NnError;
use crate::QWEN36_27B_REVISION;
use crate::layers::{HostSaltV2Linear, Projection, TokenEmbedding};
use crate::model::qwen35_hf::{
    Qwen35HfTensorSource, TensorRole, TensorSpec, language_schema, load_language_weights,
    load_mtp_weights, mtp_schema,
};
use crate::model::{Qwen35TextRunner, UnverifiedQwen35Mtp};
use crate::qwen35_config::Qwen35CheckpointConfig;

const CONFIG_FILE: &str = "config.json";
const PRESERVED_FILE: &str = "preserved.safetensors";
const MANIFEST_FILE: &str = "tritium.json";
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_PRESERVED_BYTES: u64 = 64 * 1024 * 1024;
const ARTIFACT_KIND: &str = "qwen3.6-language-mtp-salt-v2-hf-bundle";
const HF_ASSET_SPECS: [(&str, u64); 8] = [
    ("chat_template.jinja", 1_048_576),
    (CONFIG_FILE, MAX_CONFIG_BYTES),
    ("configuration.json", 65_536),
    ("generation_config.json", 1_048_576),
    ("merges.txt", 8_388_608),
    ("tokenizer.json", 33_554_432),
    ("tokenizer_config.json", 1_048_576),
    ("vocab.json", 16_777_216),
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleManifest {
    schema_version: u8,
    artifact_kind: String,
    complete_model: bool,
    packing: String,
    completion_id: String,
    campaign_id: String,
    admission_id: String,
    selection_id: String,
    source_model_id: String,
    source_revision: String,
    source_identity_status: String,
    official_payload_authenticated: bool,
    preserved: PreservedManifest,
    hf_assets: Vec<HfAssetManifest>,
    profiles: ProfileManifestSet,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreservedManifest {
    file: String,
    package_id: String,
    tensors: u64,
    payload_bytes: u64,
    serialized_bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct HfAssetManifest {
    file: String,
    package_id: String,
    bytes: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileManifestSet {
    #[serde(rename = "compact-v1")]
    compact: ProfileManifest,
    #[serde(rename = "near-lossless-v1")]
    near_lossless: ProfileManifest,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileManifest {
    file: String,
    package_id: String,
    serialized_bytes: u64,
    resident_bytes: u64,
}

/// Exact coverage and physical identity consumed by a packed Qwen load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35SaltV2LoadReceipt {
    manifest_package_id: String,
    profile: String,
    source_revision: String,
    admission_id: String,
    source_model_id: String,
    source_identity_status: String,
    official_payload_authenticated: bool,
    config_package_id: String,
    package_id: String,
    preserved_package_id: String,
    codec: SaltV2Codec,
    matrix_tensors: usize,
    preserved_tensors: usize,
    serialized_bytes: u64,
    preserved_serialized_bytes: u64,
    manifest_bytes: u64,
    hf_asset_bytes: u64,
    loaded_bundle_bytes: u64,
    salt_resident_bytes: u64,
    preserved_fp32_bytes: u64,
    resident_bytes: u64,
}

impl Qwen35SaltV2LoadReceipt {
    #[must_use]
    pub fn manifest_package_id(&self) -> &str {
        &self.manifest_package_id
    }

    #[must_use]
    pub fn profile(&self) -> &str {
        &self.profile
    }

    #[must_use]
    pub fn source_revision(&self) -> &str {
        &self.source_revision
    }

    #[must_use]
    pub fn admission_id(&self) -> &str {
        &self.admission_id
    }

    #[must_use]
    pub fn source_model_id(&self) -> &str {
        &self.source_model_id
    }

    #[must_use]
    pub fn source_identity_status(&self) -> &str {
        &self.source_identity_status
    }

    #[must_use]
    pub const fn official_payload_authenticated(&self) -> bool {
        self.official_payload_authenticated
    }

    /// Exact-byte identity of the pinned execution configuration.
    #[must_use]
    pub fn config_package_id(&self) -> &str {
        &self.config_package_id
    }

    #[must_use]
    pub fn package_id(&self) -> &str {
        &self.package_id
    }

    /// Exact-byte identity of the preserved safetensors companion.
    #[must_use]
    pub fn preserved_package_id(&self) -> &str {
        &self.preserved_package_id
    }

    #[must_use]
    pub const fn codec(&self) -> SaltV2Codec {
        self.codec
    }

    #[must_use]
    pub const fn matrix_tensors(&self) -> usize {
        self.matrix_tensors
    }

    #[must_use]
    pub const fn preserved_tensors(&self) -> usize {
        self.preserved_tensors
    }

    #[must_use]
    pub const fn serialized_bytes(&self) -> u64 {
        self.serialized_bytes
    }

    /// Exact serialized size of the preserved safetensors companion.
    #[must_use]
    pub const fn preserved_serialized_bytes(&self) -> u64 {
        self.preserved_serialized_bytes
    }

    #[must_use]
    pub const fn manifest_bytes(&self) -> u64 {
        self.manifest_bytes
    }

    #[must_use]
    pub const fn hf_asset_bytes(&self) -> u64 {
        self.hf_asset_bytes
    }

    /// Manifest, selected profile, preserved tensors, and all HF assets consumed.
    #[must_use]
    pub const fn loaded_bundle_bytes(&self) -> u64 {
        self.loaded_bundle_bytes
    }

    /// Descriptor-free SALT payload, scale, map, and rank-prefix bytes.
    #[must_use]
    pub const fn salt_resident_bytes(&self) -> u64 {
        self.salt_resident_bytes
    }

    /// Logical fp32 bytes retained after exact BF16 widening.
    #[must_use]
    pub const fn preserved_fp32_bytes(&self) -> u64 {
        self.preserved_fp32_bytes
    }

    /// Total tracked steady weight bytes across SALT and preserved fp32 tensors.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

/// Packed language runner plus structurally loaded MTP graph.
///
/// Language execution is available through [`Self::runner`]. MTP deliberately
/// remains unverified until it passes the pinned serving-oracle promotion gate.
#[allow(missing_debug_implementations)]
pub struct Qwen35SaltV2LanguageMtpModel {
    config: Qwen35CheckpointConfig,
    runner: Qwen35TextRunner,
    mtp: UnverifiedQwen35Mtp,
    receipt: Qwen35SaltV2LoadReceipt,
}

impl Qwen35SaltV2LanguageMtpModel {
    /// Load one profile package from an exported bundle directory.
    ///
    /// The caller selects `compact-v1` or `near-lossless-v1`; all filenames,
    /// identities, physical ledgers, and source revision come from the strict
    /// schema-v3 manifest. Vision remains outside this language-plus-MTP artifact.
    ///
    /// # Errors
    /// Returns [`NnError`] for non-regular files, malformed configuration or
    /// formats, incomplete/overlapping tensor coverage, wrong shapes or dtypes,
    /// transformed matrices, source mutation, allocation failure, or backend
    /// construction failure.
    pub fn load_bundle_profile(
        bundle_dir: &Path,
        profile: &str,
        backend: Box<dyn TernaryBackend>,
    ) -> Result<Self, NnError> {
        if !matches!(profile, "compact-v1" | "near-lossless-v1") {
            return Err(NnError::InvalidArtifact(
                "profile must be `compact-v1` or `near-lossless-v1`".into(),
            ));
        }
        let manifest_bytes = read_regular(&bundle_dir.join(MANIFEST_FILE), MAX_MANIFEST_BYTES)?;
        let manifest_package_id = hash_bytes(&manifest_bytes);
        let manifest: BundleManifest =
            serde_json::from_slice(&manifest_bytes).map_err(|error| {
                NnError::InvalidArtifact(format!("parse strict schema-v3 manifest: {error}"))
            })?;
        validate_manifest(&manifest)?;
        let profile_manifest = match profile {
            "compact-v1" => &manifest.profiles.compact,
            "near-lossless-v1" => &manifest.profiles.near_lossless,
            _ => unreachable!(),
        };
        let (config_bytes, config_package_id, hf_asset_bytes) =
            verify_hf_assets(bundle_dir, &manifest.hf_assets)?;
        let config_text = std::str::from_utf8(&config_bytes)
            .map_err(|_| NnError::MissingConfig("config.json is not UTF-8".into()))?;
        let config = Qwen35CheckpointConfig::from_hf_config(config_text)?;
        config.validate_pinned_qwen36_27b(&manifest.source_revision)?;
        let package_path = bundle_dir.join(&profile_manifest.file);
        let package_file = open_regular(&package_path, "SALT V2 profile")?;
        let preserved_bytes = read_regular(
            &bundle_dir.join(&manifest.preserved.file),
            MAX_PRESERVED_BYTES,
        )?;
        let preserved_serialized_bytes = u64::try_from(preserved_bytes.len())
            .map_err(|_| NnError::ResourceExhausted("preserved size exceeds u64".into()))?;
        if preserved_serialized_bytes != manifest.preserved.serialized_bytes {
            return Err(NnError::Provenance(
                "preserved serialized-byte ledger differs from bundle manifest".into(),
            ));
        }
        let preserved_package_id = hash_bytes(&preserved_bytes);
        if preserved_package_id != manifest.preserved.package_id {
            return Err(NnError::Provenance(
                "preserved tensor identity differs from bundle manifest".into(),
            ));
        }
        let package = SaltV2PackageReader::new_strict(package_file)
            .map_err(|error| NnError::InvalidArtifact(format!("open SALT V2 profile: {error}")))?;
        let package_ledger = package.ledger();
        let runtime_ledger = package
            .indexed_runtime_ledger()
            .map_err(|error| NnError::InvalidArtifact(error.to_string()))?;
        if package.package_id().to_string() != profile_manifest.package_id
            || package_ledger.total_bytes != profile_manifest.serialized_bytes
            || runtime_ledger.steady_resident_bytes() != profile_manifest.resident_bytes
            || codec_name(package.codec()) != manifest.packing
        {
            return Err(NnError::Provenance(
                "SALT V2 profile identity, codec, or physical ledger differs from manifest".into(),
            ));
        }
        let preserved_payload_bytes = safetensors_payload_bytes(&preserved_bytes)?;
        if preserved_payload_bytes != manifest.preserved.payload_bytes {
            return Err(NnError::Provenance(
                "preserved payload ledger differs from bundle manifest".into(),
            ));
        }
        let preserved = SafeTensorsReader::new(Cursor::new(preserved_bytes.into_boxed_slice()))
            .map_err(|error| {
                NnError::InvalidArtifact(format!("open preserved safetensors: {error}"))
            })?;
        if u64::try_from(preserved.len()).ok() != Some(manifest.preserved.tensors) {
            return Err(NnError::Provenance(
                "preserved tensor count differs from bundle manifest".into(),
            ));
        }
        let source = SaltV2QwenSource {
            package: RefCell::new(package),
            preserved: RefCell::new(preserved),
        };
        let schema = combined_schema(&config)?;
        let (matrix_tensors, preserved_tensors, preserved_fp32_bytes) =
            validate_coverage(&source, &schema)?;
        let codec = source.package.borrow().codec();
        let package_id = source.package.borrow().package_id().to_string();

        let language = load_language_weights(&source, &config.text)?;
        let mtp_weights = load_mtp_weights(&source, &config.text)?;
        source
            .package
            .borrow_mut()
            .verify_unchanged()
            .map_err(|error| NnError::Provenance(format!("SALT V2 profile changed: {error}")))?;
        let runner = Qwen35TextRunner::new(&config.text, language, backend)?;
        let mtp = UnverifiedQwen35Mtp::new(&runner, mtp_weights)?;
        let salt_resident_bytes = runtime_ledger.steady_resident_bytes();
        let resident_bytes = salt_resident_bytes
            .checked_add(preserved_fp32_bytes)
            .ok_or_else(|| {
                NnError::ResourceExhausted("tracked Qwen weight bytes overflow".into())
            })?;
        let manifest_bytes = u64::try_from(manifest_bytes.len())
            .map_err(|_| NnError::ResourceExhausted("manifest size exceeds u64".into()))?;
        let loaded_bundle_bytes = manifest_bytes
            .checked_add(hf_asset_bytes)
            .and_then(|bytes| bytes.checked_add(package_ledger.total_bytes))
            .and_then(|bytes| bytes.checked_add(preserved_serialized_bytes))
            .ok_or_else(|| {
                NnError::ResourceExhausted("loaded bundle byte ledger overflow".into())
            })?;
        Ok(Self {
            config,
            runner,
            mtp,
            receipt: Qwen35SaltV2LoadReceipt {
                manifest_package_id,
                profile: profile.to_owned(),
                source_revision: manifest.source_revision,
                admission_id: manifest.admission_id,
                source_model_id: manifest.source_model_id,
                source_identity_status: manifest.source_identity_status,
                official_payload_authenticated: manifest.official_payload_authenticated,
                config_package_id,
                package_id,
                preserved_package_id,
                codec,
                matrix_tensors,
                preserved_tensors,
                serialized_bytes: package_ledger.total_bytes,
                preserved_serialized_bytes,
                manifest_bytes,
                hf_asset_bytes,
                loaded_bundle_bytes,
                salt_resident_bytes,
                preserved_fp32_bytes,
                resident_bytes,
            },
        })
    }

    #[must_use]
    pub const fn config(&self) -> &Qwen35CheckpointConfig {
        &self.config
    }

    #[must_use]
    pub const fn runner(&self) -> &Qwen35TextRunner {
        &self.runner
    }

    #[must_use]
    pub const fn mtp(&self) -> &UnverifiedQwen35Mtp {
        &self.mtp
    }

    #[must_use]
    pub const fn receipt(&self) -> &Qwen35SaltV2LoadReceipt {
        &self.receipt
    }
}

struct SaltV2QwenSource {
    package: RefCell<SaltV2PackageReader<File>>,
    preserved: RefCell<SafeTensorsReader<Cursor<Box<[u8]>>>>,
}

impl Qwen35HfTensorSource for SaltV2QwenSource {
    fn tensor_f32_exact(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, NnError> {
        let mut preserved = self
            .preserved
            .try_borrow_mut()
            .map_err(|_| NnError::Backend("reentrant preserved tensor read".into()))?;
        let shape = preserved
            .shape(name)
            .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
        if shape != expected {
            return Err(NnError::Shape {
                expected: expected.iter().copied().product(),
                got: shape.iter().copied().product(),
            });
        }
        if preserved.dtype(name) != Some("BF16") {
            return Err(NnError::InvalidArtifact(format!(
                "preserved tensor `{name}` is not exact BF16"
            )));
        }
        preserved
            .tensor_f32(name)
            .map_err(|error| NnError::InvalidArtifact(format!("read `{name}`: {error}")))
    }

    fn projection_exact(
        &self,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Projection, NnError> {
        let mut package = self
            .package
            .try_borrow_mut()
            .map_err(|_| NnError::Backend("reentrant SALT V2 matrix read".into()))?;
        let matrix = HostSaltV2Linear::from_reader(&mut package, name)?;
        require_matrix_geometry(name, &matrix, rows, columns)?;
        Ok(Projection::HostSaltV2(Arc::new(matrix)))
    }

    fn token_embedding_exact(
        &self,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<TokenEmbedding, NnError> {
        let mut package = self
            .package
            .try_borrow_mut()
            .map_err(|_| NnError::Backend("reentrant SALT V2 token-table read".into()))?;
        let matrix = Arc::new(HostSaltV2Linear::from_reader(&mut package, name)?);
        require_matrix_geometry(name, &matrix, rows, columns)?;
        TokenEmbedding::from_host_salt_v2(matrix)
    }
}

fn require_matrix_geometry(
    name: &str,
    matrix: &HostSaltV2Linear,
    rows: usize,
    columns: usize,
) -> Result<(), NnError> {
    if matrix.rows() != rows || matrix.columns() != columns {
        return Err(NnError::InvalidArtifact(format!(
            "SALT V2 matrix `{name}` has geometry {}x{}, expected {rows}x{columns}",
            matrix.rows(),
            matrix.columns()
        )));
    }
    Ok(())
}

fn combined_schema(
    config: &Qwen35CheckpointConfig,
) -> Result<BTreeMap<String, TensorSpec>, NnError> {
    let mut schema = language_schema(&config.text)?;
    for (name, spec) in mtp_schema(&config.text)? {
        if schema.insert(name.clone(), spec).is_some() {
            return Err(NnError::InvalidArtifact(format!(
                "duplicate language/MTP schema tensor `{name}`"
            )));
        }
    }
    Ok(schema)
}

fn validate_coverage(
    source: &SaltV2QwenSource,
    schema: &BTreeMap<String, TensorSpec>,
) -> Result<(usize, usize, u64), NnError> {
    let package = source.package.borrow();
    let preserved = source.preserved.borrow();
    let matrix_names = package.tensor_names().collect::<BTreeSet<_>>();
    let preserved_names = preserved.names().collect::<BTreeSet<_>>();
    let mut preserved_fp32_bytes = 0u64;
    for (name, spec) in schema {
        match spec.role {
            TensorRole::Matrix => {
                if preserved_names.contains(name.as_str()) || !matrix_names.contains(name.as_str())
                {
                    return Err(NnError::InvalidArtifact(format!(
                        "matrix `{name}` must occur exactly once in the SALT V2 profile"
                    )));
                }
                let info = package
                    .tensor_info(name)
                    .ok_or_else(|| NnError::MissingTensor(name.clone()))?;
                let expected = spec
                    .shape
                    .iter()
                    .copied()
                    .map(|dim| u64::try_from(dim).unwrap_or(u64::MAX))
                    .collect::<Vec<_>>();
                if info.dims() != expected || info.transform() != SaltV2Transform::None {
                    return Err(NnError::InvalidArtifact(format!(
                        "matrix `{name}` shape or transform differs from Qwen schema"
                    )));
                }
            }
            TensorRole::Preserved => {
                if matrix_names.contains(name.as_str())
                    || !preserved_names.contains(name.as_str())
                    || preserved.dtype(name) != Some("BF16")
                    || preserved.shape(name) != Some(spec.shape.as_slice())
                {
                    return Err(NnError::InvalidArtifact(format!(
                        "preserved tensor `{name}` must occur exactly once as exact BF16"
                    )));
                }
                let elements = spec.shape.iter().try_fold(1u64, |product, dimension| {
                    product.checked_mul(u64::try_from(*dimension).ok()?)
                });
                let bytes = elements
                    .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
                    .ok_or_else(|| {
                        NnError::ResourceExhausted(format!(
                            "preserved tensor `{name}` fp32 byte count overflows"
                        ))
                    })?;
                preserved_fp32_bytes =
                    preserved_fp32_bytes.checked_add(bytes).ok_or_else(|| {
                        NnError::ResourceExhausted("preserved Qwen fp32 bytes overflow".into())
                    })?;
            }
        }
    }
    if let Some(extra) = matrix_names.iter().find(|name| {
        !schema
            .get(**name)
            .is_some_and(|spec| spec.role == TensorRole::Matrix)
    }) {
        return Err(NnError::InvalidArtifact(format!(
            "unexpected SALT V2 matrix `{extra}`"
        )));
    }
    if let Some(extra) = preserved_names.iter().find(|name| {
        !schema
            .get(**name)
            .is_some_and(|spec| spec.role == TensorRole::Preserved)
    }) {
        return Err(NnError::InvalidArtifact(format!(
            "unexpected preserved tensor `{extra}`"
        )));
    }
    Ok((
        matrix_names.len(),
        preserved_names.len(),
        preserved_fp32_bytes,
    ))
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = PackageHasher::new();
    hasher.update(bytes);
    hasher.finalize().to_string()
}

fn validate_manifest(manifest: &BundleManifest) -> Result<(), NnError> {
    if manifest.schema_version != 3
        || manifest.artifact_kind != ARTIFACT_KIND
        || manifest.complete_model
        || manifest.source_revision != QWEN36_27B_REVISION
        || !matches!(manifest.packing.as_str(), "d2" | "b3" | "s34")
    {
        return Err(NnError::InvalidArtifact(
            "manifest is not the pinned schema-v3 Qwen3.6 language/MTP bundle".into(),
        ));
    }
    for (field, value) in [
        ("completion_id", manifest.completion_id.as_str()),
        ("campaign_id", manifest.campaign_id.as_str()),
        ("admission_id", manifest.admission_id.as_str()),
        ("selection_id", manifest.selection_id.as_str()),
        ("source_model_id", manifest.source_model_id.as_str()),
        (
            "source_identity_status",
            manifest.source_identity_status.as_str(),
        ),
    ] {
        if value.is_empty() {
            return Err(NnError::InvalidArtifact(format!(
                "manifest `{field}` must be non-empty"
            )));
        }
    }
    if manifest.preserved.file != PRESERVED_FILE
        || manifest.preserved.package_id.is_empty()
        || manifest.preserved.tensors == 0
        || manifest.preserved.payload_bytes == 0
        || manifest.preserved.serialized_bytes == 0
        || manifest.preserved.serialized_bytes > MAX_PRESERVED_BYTES
    {
        return Err(NnError::InvalidArtifact(
            "manifest preserved-tensor descriptor is not canonical".into(),
        ));
    }
    validate_profile_manifest(&manifest.profiles.compact, "compact.tsalt2")?;
    validate_profile_manifest(&manifest.profiles.near_lossless, "near-lossless.tsalt2")?;
    Ok(())
}

fn validate_profile_manifest(profile: &ProfileManifest, file: &str) -> Result<(), NnError> {
    if profile.file != file
        || profile.package_id.is_empty()
        || profile.serialized_bytes == 0
        || profile.resident_bytes == 0
    {
        return Err(NnError::InvalidArtifact(format!(
            "manifest profile `{file}` is not canonical"
        )));
    }
    Ok(())
}

fn verify_hf_assets(
    bundle_dir: &Path,
    assets: &[HfAssetManifest],
) -> Result<(Vec<u8>, String, u64), NnError> {
    if assets.len() != HF_ASSET_SPECS.len() {
        return Err(NnError::InvalidArtifact(
            "manifest HF asset catalog is incomplete".into(),
        ));
    }
    let mut config = None;
    let mut total_bytes = 0u64;
    for ((expected_file, max_bytes), asset) in HF_ASSET_SPECS.iter().zip(assets) {
        if asset.file != *expected_file
            || asset.package_id.is_empty()
            || asset.bytes == 0
            || asset.bytes > *max_bytes
        {
            return Err(NnError::InvalidArtifact(format!(
                "manifest HF asset `{expected_file}` is not canonical"
            )));
        }
        let bytes = read_regular(&bundle_dir.join(expected_file), *max_bytes)?;
        if u64::try_from(bytes.len()).ok() != Some(asset.bytes)
            || hash_bytes(&bytes) != asset.package_id
        {
            return Err(NnError::Provenance(format!(
                "HF asset `{expected_file}` differs from manifest"
            )));
        }
        total_bytes = total_bytes
            .checked_add(asset.bytes)
            .ok_or_else(|| NnError::ResourceExhausted("HF asset byte ledger overflow".into()))?;
        if *expected_file == CONFIG_FILE {
            config = Some((bytes, asset.package_id.clone()));
        }
    }
    let (config, package_id) = config
        .ok_or_else(|| NnError::InvalidArtifact("manifest has no config.json asset".into()))?;
    Ok((config, package_id, total_bytes))
}

fn codec_name(codec: SaltV2Codec) -> &'static str {
    match codec {
        SaltV2Codec::D2 => "d2",
        SaltV2Codec::B3 => "b3",
        SaltV2Codec::S34 => "s34",
        _ => "unknown",
    }
}

fn safetensors_payload_bytes(bytes: &[u8]) -> Result<u64, NnError> {
    let prefix: [u8; 8] = bytes
        .get(..8)
        .and_then(|value| value.try_into().ok())
        .ok_or_else(|| NnError::InvalidArtifact("preserved safetensors is truncated".into()))?;
    let data_start = u64::from_le_bytes(prefix)
        .checked_add(8)
        .ok_or_else(|| NnError::InvalidArtifact("safetensors header length overflows".into()))?;
    u64::try_from(bytes.len())
        .ok()
        .and_then(|total| total.checked_sub(data_start))
        .ok_or_else(|| NnError::InvalidArtifact("safetensors header exceeds file".into()))
}

fn open_regular(path: &Path, description: &str) -> Result<File, NnError> {
    let before = fs::symlink_metadata(path).map_err(|error| {
        NnError::InvalidArtifact(format!("inspect {description} {}: {error}", path.display()))
    })?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(NnError::InvalidArtifact(format!(
            "{description} must be an ordinary non-symlink file"
        )));
    }
    let file = File::open(path).map_err(|error| {
        NnError::InvalidArtifact(format!("open {description} {}: {error}", path.display()))
    })?;
    let opened = file.metadata().map_err(|error| {
        NnError::InvalidArtifact(format!("inspect opened {description}: {error}"))
    })?;
    if !opened.is_file() || !same_file(&before, &opened) {
        return Err(NnError::Provenance(format!(
            "{description} changed while it was opened"
        )));
    }
    Ok(file)
}

fn read_regular(path: &Path, max_bytes: u64) -> Result<Vec<u8>, NnError> {
    let mut file = open_regular(path, "bundle file")?;
    let length = file
        .metadata()
        .map_err(|error| NnError::InvalidArtifact(format!("inspect {}: {error}", path.display())))?
        .len();
    if length == 0 || length > max_bytes {
        return Err(NnError::InvalidArtifact(format!(
            "bundle file {} has invalid length {length} (limit {max_bytes})",
            path.display()
        )));
    }
    let length = usize::try_from(length)
        .map_err(|_| NnError::ResourceExhausted("bundle file exceeds host usize".into()))?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|_| {
        NnError::ResourceExhausted(format!("allocate {} bundle bytes", path.display()))
    })?;
    file.read_to_end(&mut bytes)
        .map_err(|error| NnError::InvalidArtifact(format!("read {}: {error}", path.display())))?;
    if bytes.len() != length {
        return Err(NnError::Provenance(format!(
            "bundle file {} changed while it was read",
            path.display()
        )));
    }
    Ok(bytes)
}

#[cfg(unix)]
fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.len() == right.len()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
}

#[cfg(not(unix))]
fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.len() == right.len()
        && left.modified().ok() == right.modified().ok()
        && left.is_file() == right.is_file()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use half::f16;
    use tritium_format::{
        PackageId,
        salt_v2_package::{
            SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, write_salt_v2_package,
        },
    };

    use super::*;

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TestFiles {
        directory: std::path::PathBuf,
    }

    impl TestFiles {
        fn new() -> Self {
            let directory = std::env::temp_dir().join(format!(
                "tritium-qwen-salt-source-{}-{}",
                std::process::id(),
                TEST_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&directory).unwrap();
            Self { directory }
        }

        fn source(&self) -> SaltV2QwenSource {
            let plane = SaltV2Plane::new(vec![1, 0, -1, 1], vec![f16::from_f32(0.5)]).unwrap();
            let tensor = SaltV2Tensor::new(
                "matrix",
                vec![2, 2],
                vec![SaltV2Tile::new(vec![plane]).unwrap()],
            )
            .unwrap();
            let package = SaltV2Package::new(SaltV2Codec::B3, vec![tensor]).unwrap();
            let encoded = write_salt_v2_package(&package).unwrap();
            let package_path = self.directory.join("profile.tsalt2");
            fs::write(&package_path, encoded.bytes).unwrap();
            let preserved_path = self.directory.join("preserved.safetensors");
            fs::write(&preserved_path, safetensors()).unwrap();
            SaltV2QwenSource {
                package: RefCell::new(
                    SaltV2PackageReader::new_strict(File::open(package_path).unwrap()).unwrap(),
                ),
                preserved: RefCell::new(
                    SafeTensorsReader::new(Cursor::new(
                        fs::read(preserved_path).unwrap().into_boxed_slice(),
                    ))
                    .unwrap(),
                ),
            }
        }
    }

    impl Drop for TestFiles {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.directory).unwrap();
        }
    }

    fn safetensors() -> Vec<u8> {
        let mut header = br#"{"__metadata__":{"format":"pt"},"norm":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&0x3f80_u16.to_le_bytes());
        bytes.extend_from_slice(&0xc000_u16.to_le_bytes());
        bytes
    }

    fn schema() -> BTreeMap<String, TensorSpec> {
        BTreeMap::from([
            (
                "matrix".to_owned(),
                TensorSpec {
                    shape: vec![2, 2],
                    role: TensorRole::Matrix,
                },
            ),
            (
                "norm".to_owned(),
                TensorSpec {
                    shape: vec![2],
                    role: TensorRole::Preserved,
                },
            ),
        ])
    }

    #[test]
    fn split_source_enforces_coverage_and_builds_packed_values() {
        let files = TestFiles::new();
        let source = files.source();
        assert_eq!(validate_coverage(&source, &schema()).unwrap(), (1, 1, 8));

        let projection = source.projection_exact("matrix", 2, 2).unwrap();
        assert!(matches!(projection, Projection::HostSaltV2(_)));
        let embedding = source.token_embedding_exact("matrix", 2, 2).unwrap();
        assert!(embedding.is_packed_salt());
        assert_eq!(source.tensor_f32_exact("norm", &[2]).unwrap(), [1.0, -2.0]);
        source.package.borrow_mut().verify_unchanged().unwrap();
    }

    #[test]
    fn split_source_rejects_role_and_shape_drift_before_assembly() {
        let files = TestFiles::new();
        let source = files.source();
        let mut wrong_role = schema();
        wrong_role.get_mut("matrix").unwrap().role = TensorRole::Preserved;
        assert!(matches!(
            validate_coverage(&source, &wrong_role),
            Err(NnError::InvalidArtifact(_))
        ));

        let mut wrong_shape = schema();
        wrong_shape.get_mut("matrix").unwrap().shape = vec![1, 4];
        assert!(matches!(
            validate_coverage(&source, &wrong_shape),
            Err(NnError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn preserved_reader_rejects_shape_and_dtype_substitution() {
        let files = TestFiles::new();
        let source = files.source();
        assert!(matches!(
            source.tensor_f32_exact("norm", &[1, 2]),
            Err(NnError::Shape { .. })
        ));
        assert!(matches!(
            source.tensor_f32_exact("missing", &[2]),
            Err(NnError::MissingTensor(_))
        ));
    }

    #[test]
    fn complete_bundle_loads_language_and_mtp_without_dense_matrix_shadows() {
        let files = TestFiles::new();
        let config_json = family_config_json();
        fs::write(files.directory.join(CONFIG_FILE), &config_json).unwrap();
        let config = Qwen35CheckpointConfig::from_hf_config(&config_json).unwrap();
        let schema = combined_schema(&config).unwrap();

        let mut matrices = Vec::new();
        let mut preserved = BTreeMap::new();
        for (name, spec) in &schema {
            match spec.role {
                TensorRole::Matrix => matrices.push(zero_matrix(name, &spec.shape)),
                TensorRole::Preserved => {
                    preserved.insert(name.clone(), spec.shape.clone());
                }
            }
        }
        let matrix_count = matrices.len();
        let encoded =
            write_salt_v2_package(&SaltV2Package::new(SaltV2Codec::B3, matrices).unwrap()).unwrap();
        let profile_id = PackageId::from_package_bytes(&encoded.bytes).to_string();
        fs::write(files.directory.join("compact.tsalt2"), encoded.bytes).unwrap();
        let preserved_bytes = zero_bf16_safetensors(&preserved);
        let preserved_id = PackageId::from_package_bytes(&preserved_bytes).to_string();
        fs::write(files.directory.join(PRESERVED_FILE), preserved_bytes).unwrap();

        let package = SaltV2PackageReader::new_strict(
            File::open(files.directory.join("compact.tsalt2")).unwrap(),
        )
        .unwrap();
        assert_eq!(package.package_id().to_string(), profile_id);
        let preserved_reader = SafeTensorsReader::new(Cursor::new(
            fs::read(files.directory.join(PRESERVED_FILE))
                .unwrap()
                .into_boxed_slice(),
        ))
        .unwrap();
        let source = SaltV2QwenSource {
            package: RefCell::new(package),
            preserved: RefCell::new(preserved_reader),
        };
        let (loaded_matrices, loaded_preserved, preserved_fp32_bytes) =
            validate_coverage(&source, &schema).unwrap();
        assert_eq!(loaded_matrices, matrix_count);
        assert_eq!(loaded_preserved, schema.len() - matrix_count);
        assert!(preserved_fp32_bytes > 0);
        assert_eq!(
            hash_bytes(&fs::read(files.directory.join(PRESERVED_FILE)).unwrap()),
            preserved_id
        );

        let language = load_language_weights(&source, &config.text).unwrap();
        let mtp_weights = load_mtp_weights(&source, &config.text).unwrap();
        source.package.borrow_mut().verify_unchanged().unwrap();
        let runner = Qwen35TextRunner::new(
            &config.text,
            language,
            Box::new(tritium_cpu::CpuBackend::new()),
        )
        .unwrap();
        let _mtp = UnverifiedQwen35Mtp::new(&runner, mtp_weights).unwrap();
        let mut cache = runner.new_cache(4).unwrap();
        let output = runner.forward(&[1, 2], &mut cache).unwrap();
        assert_eq!(output.last_logits(), &[0.0; 7]);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn bundle_loader_rejects_parent_traversal_before_opening_files() {
        let files = TestFiles::new();
        assert!(matches!(
            Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
                &files.directory,
                "..",
                Box::new(tritium_cpu::CpuBackend::new()),
            ),
            Err(NnError::InvalidArtifact(_))
        ));
    }

    #[test]
    fn bundle_loader_rejects_duplicate_manifest_fields_before_assets() {
        let files = TestFiles::new();
        fs::write(
            files.directory.join(MANIFEST_FILE),
            br#"{"schema_version":3,"schema_version":3}"#,
        )
        .unwrap();
        let error = Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
            &files.directory,
            "compact-v1",
            Box::new(tritium_cpu::CpuBackend::new()),
        )
        .err()
        .expect("duplicate manifest must fail");
        assert!(matches!(error, NnError::InvalidArtifact(_)));
        assert!(error.to_string().contains("duplicate field"));
    }

    fn zero_matrix(name: &str, shape: &[usize]) -> SaltV2Tensor {
        let coefficients = shape.iter().product::<usize>();
        let mut tiles = Vec::new();
        let mut remaining = coefficients;
        while remaining != 0 {
            let logical_len = remaining.min(256);
            let plane = SaltV2Plane::new(
                vec![0; logical_len],
                vec![f16::ZERO; logical_len.div_ceil(128)],
            )
            .unwrap();
            tiles.push(SaltV2Tile::new(vec![plane]).unwrap());
            remaining -= logical_len;
        }
        SaltV2Tensor::new(
            name,
            shape.iter().map(|dimension| *dimension as u64).collect(),
            tiles,
        )
        .unwrap()
    }

    fn zero_bf16_safetensors(tensors: &BTreeMap<String, Vec<usize>>) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let mut offset = 0usize;
        for (name, shape) in tensors {
            let bytes = shape.iter().product::<usize>() * size_of::<u16>();
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut encoded_header = serde_json::to_vec(&header).unwrap();
        while !encoded_header.len().is_multiple_of(8) {
            encoded_header.push(b' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(encoded_header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&encoded_header);
        bytes.resize(bytes.len() + offset, 0);
        bytes
    }

    fn family_config_json() -> String {
        serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "language_model_only": false,
            "model_type": "qwen3_5",
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attn_output_gate": true,
                "dtype": "bfloat16",
                "full_attention_interval": 2,
                "head_dim": 4,
                "hidden_act": "silu",
                "hidden_size": 4,
                "intermediate_size": 6,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 2,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 2,
                "linear_value_head_dim": 2,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 32,
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 2,
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "output_gate_type": "swish",
                "partial_rotary_factor": 0.5,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [1, 0, 0],
                    "partial_rotary_factor": 0.5,
                    "rope_theta": 10000.0,
                    "rope_type": "default"
                },
                "tie_word_embeddings": false,
                "use_cache": true,
                "vocab_size": 7
            },
            "tie_word_embeddings": false,
            "vision_config": {"model_type": "qwen3_5"}
        })
        .to_string()
    }
}
