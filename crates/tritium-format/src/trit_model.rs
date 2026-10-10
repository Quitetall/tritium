//! Strict typed model references over the existing `.trit` envelope (ADR 0044).
//! This proposed profile proves metadata/payload consistency, not model execution,
//! external evidence, owner acceptance of candidate identities, quality or release admission.

use std::collections::BTreeMap;

use ciborium::value::Value;
use half::{bf16, f16};
use tritium_core::DType;
use tritium_schema::{
    BlobId, Level, LevelFit, ModelId, ModelManifest, ModelManifestError, ModelSemanticMeta,
    NamedAssetDigest, NamedTensorDigest, PlaneCounts, ScalePrecision, SemanticTensorDigest,
    TensorMeta, TensorSemanticMeta, TensorStorage, Transport,
};

use crate::trit_package::{
    canonicalize_value, encode_cbor_value, parse_canonical_manifest, validate_canonical_manifest,
};
use crate::{
    AdditiveTensor, AdditiveTensorError, TritBlobKind, TritPackage, TritPackageError,
    read_trit_package,
};

/// Typed metadata/payload failure; unsupported capabilities are explicit.
#[derive(Debug)]
#[non_exhaustive]
pub enum TritModelError {
    /// Outer package or canonical-CBOR envelope failed validation.
    Package(TritPackageError),
    /// Required typed fields, variants or strict field names failed decoding.
    Schema(String),
    /// Metadata failed shared structural validation.
    Manifest(ModelManifestError),
    /// Lists are not in the canonical typed order, or typed bytes have aliases.
    NonCanonical,
    /// Named payload is absent or has the wrong directory kind/dtype/name.
    MissingOrWrongBlob(String),
    /// Named level or tensor is absent.
    MissingBinding(String),
    /// Dense byte length or finite element validation failed.
    DensePayload(String),
    /// Canonical additive bytes/scales/geometry failed validation.
    Additive(AdditiveTensorError),
    /// Declared exact prefix is not a prefix of the parent logical weights.
    InvalidPrefix(String),
    /// Profile data is representable but not supported by this payload reader.
    UnsupportedFeature(&'static str),
}

impl core::fmt::Display for TritModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "trit model: {self:?}")
    }
}
impl std::error::Error for TritModelError {}
impl From<TritPackageError> for TritModelError {
    fn from(error: TritPackageError) -> Self {
        Self::Package(error)
    }
}
impl From<ModelManifestError> for TritModelError {
    fn from(error: ModelManifestError) -> Self {
        Self::Manifest(error)
    }
}

/// Derive canonical CBOR from the Rust schema, normalizing named inventory order.
///
/// Duplicate names are rejected, not collapsed. No handwritten model encoder or
/// parallel manifest vocabulary is used. This profile has no floating metadata.
///
/// # Errors
/// Structural, serde or canonical-envelope errors.
pub fn encode_model_manifest(manifest: &ModelManifest) -> Result<Vec<u8>, TritModelError> {
    let mut canonical = manifest.clone();
    canonical.assets.sort_by(|a, b| a.name.cmp(&b.name));
    canonical.levels.sort_by(|a, b| a.name.cmp(&b.name));
    canonical.shards.sort_by(|a, b| a.name.cmp(&b.name));
    canonical.legacy_ids.sort_by(|a, b| a.id.cmp(&b.id));
    for level in &mut canonical.levels {
        level.tensors.sort_by(|a, b| a.name.cmp(&b.name));
        if let LevelFit::Direct { evidence } = &mut level.fit {
            evidence.sort_by(|a, b| (&a.run, &a.span, &a.root).cmp(&(&b.run, &b.span, &b.root)));
        }
    }
    canonical.validate()?;
    let mut value =
        Value::serialized(&canonical).map_err(|e| TritModelError::Schema(e.to_string()))?;
    canonicalize_value(&mut value, 0)?;
    let bytes = encode_cbor_value(&value)?;
    validate_canonical_manifest(&bytes)?;
    Ok(bytes)
}

/// Decode only the canonical typed encoding; unknown/duplicate fields fail closed.
///
/// # Errors
/// Unknown versions/tags/fields, invalid structure or noncanonical byte/order data.
pub fn decode_model_manifest(bytes: &[u8]) -> Result<ModelManifest, TritModelError> {
    let value = parse_canonical_manifest(bytes)?;
    let manifest: ModelManifest = value
        .deserialized()
        .map_err(|e| TritModelError::Schema(e.to_string()))?;
    manifest.validate()?;
    if encode_model_manifest(&manifest)? != bytes {
        return Err(TritModelError::NonCanonical);
    }
    Ok(manifest)
}

/// Borrowed model package with checked named references and payload interpretation.
///
/// Candidate logical identities are bound during validation, not inferred model
/// execution/provenance/qualification. Source bytes remain borrowed. Per-binding
/// digests are retained; additive validation temporarily decodes one tensor
/// (two for prefix checks), not an entire retained dense model.
#[derive(Debug)]
pub struct TritModel<'a> {
    package: TritPackage<'a>,
    manifest: ModelManifest,
    index: BTreeMap<[u8; 32], Vec<usize>>,
    tensor_digests: Vec<Vec<SemanticTensorDigest>>,
    model_ids: Vec<ModelId>,
}

impl<'a> TritModel<'a> {
    /// Immutable validated metadata; successful parsing is not model qualification.
    #[must_use]
    pub const fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Exact stored package identity, never substituted for a semantic model id.
    #[must_use]
    pub fn package_id(&self) -> crate::PackageId {
        self.package.package_id()
    }

    /// Candidate per-level content identity, independent of physical packaging.
    ///
    /// ADR 0054 remains proposed: matching IDs do not verify fit provenance,
    /// legacy identity resolution, architecture/tokenizer execution or release.
    /// # Errors
    /// Absent level.
    pub fn model_id(&self, level: &str) -> Result<ModelId, TritModelError> {
        Ok(self.model_ids[self.level_index(level)?])
    }

    /// Candidate logical tensor digest bound during validated payload reading.
    ///
    /// Excludes physical codec/transport, not basis/law/geometry/exact scale bits.
    /// # Errors
    /// Absent level/tensor.
    pub fn tensor_digest(
        &self,
        level: &str,
        name: &str,
    ) -> Result<SemanticTensorDigest, TritModelError> {
        let index = self.level_index(level)?;
        let tensor = self.manifest.levels[index]
            .tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .map_err(|_| TritModelError::MissingBinding(name.into()))?;
        Ok(self.tensor_digests[index][tensor])
    }

    /// Borrow a tensor's exact stored bytes from a named complete level.
    ///
    /// # Errors
    /// Absent level/tensor or incompatible directory binding.
    pub fn tensor(
        &self,
        level: &str,
        name: &str,
    ) -> Result<(&TensorMeta, &'a [u8]), TritModelError> {
        let level = self.level(level)?;
        let index = level
            .tensors
            .binary_search_by(|t| t.name.as_str().cmp(name))
            .map_err(|_| TritModelError::MissingBinding(name.into()))?;
        let meta = &level.tensors[index];
        Ok((meta, self.tensor_payload(meta)?))
    }

    /// Borrow an asset by its declared safe basename, without filesystem extraction.
    ///
    /// # Errors
    /// Absent asset or mismatched kind/name/reference.
    pub fn asset(&self, name: &str) -> Result<&'a [u8], TritModelError> {
        let index = self
            .manifest
            .assets
            .binary_search_by(|a| a.name.as_str().cmp(name))
            .map_err(|_| TritModelError::MissingBinding(name.into()))?;
        let asset = &self.manifest.assets[index];
        self.payload(
            asset.blob.id,
            &TritBlobKind::Asset(asset.name.clone()),
            name,
        )
    }

    fn level(&self, name: &str) -> Result<&Level, TritModelError> {
        Ok(&self.manifest.levels[self.level_index(name)?])
    }

    fn level_index(&self, name: &str) -> Result<usize, TritModelError> {
        self.manifest
            .levels
            .binary_search_by(|l| l.name.as_str().cmp(name))
            .map_err(|_| TritModelError::MissingBinding(name.into()))
    }

    fn payload(
        &self,
        id: BlobId,
        kind: &TritBlobKind,
        name: &str,
    ) -> Result<&'a [u8], TritModelError> {
        let index = self
            .index
            .get(id.as_bytes())
            .and_then(|indices| {
                indices
                    .iter()
                    .copied()
                    .find(|&i| &self.package.blobs()[i].kind == kind)
            })
            .ok_or_else(|| TritModelError::MissingOrWrongBlob(name.into()))?;
        Ok(self.package.indexed_blob(index)?)
    }

    fn tensor_payload(&self, meta: &TensorMeta) -> Result<&'a [u8], TritModelError> {
        let kind = match &meta.storage {
            TensorStorage::Dense { precision, .. } => TritBlobKind::DenseTensor(match precision {
                ScalePrecision::F16 => DType::F16,
                ScalePrecision::Bf16 => DType::BF16,
                ScalePrecision::F32 => DType::F32,
                _ => return Err(TritModelError::UnsupportedFeature("dense precision")),
            }),
            TensorStorage::Additive { .. } => TritBlobKind::AdditiveTensor,
            _ => return Err(TritModelError::UnsupportedFeature("tensor storage")),
        };
        self.payload(meta.blob.id, &kind, &meta.name)
    }
}

/// Validate all complete levels and their payloads before publishing a typed view.
///
/// The current reader explicitly rejects sharded/per-tile/entropy payloads rather
/// than treating structurally valid declarations as supported execution. Their
/// metadata is representable for later capability-gated reader work. Architecture
/// completeness, tokenizer semantics and external evidence admission are later gates.
///
/// # Errors
/// Package, typed metadata, binding, byte payload, prefix or unavailable-reader capability.
pub fn read_trit_model(bytes: &[u8]) -> Result<TritModel<'_>, TritModelError> {
    let package = read_trit_package(bytes)?;
    let manifest = decode_model_manifest(package.manifest())?;
    if !manifest.shards.is_empty() {
        return Err(TritModelError::UnsupportedFeature("sharded model payloads"));
    }
    let mut index = BTreeMap::<_, Vec<_>>::new();
    for (i, blob) in package.blobs().iter().enumerate() {
        index.entry(*blob.id.as_bytes()).or_default().push(i);
    }
    let mut model = TritModel {
        package,
        manifest,
        index,
        tensor_digests: Vec::new(),
        model_ids: Vec::new(),
    };
    for asset in &model.manifest.assets {
        model.asset(&asset.name)?;
    }
    for level in &model.manifest.levels {
        let mut digests = Vec::new();
        for tensor in &level.tensors {
            let bytes = model.tensor_payload(tensor)?;
            match &tensor.storage {
                TensorStorage::Dense { shape, precision } => {
                    validate_dense(&tensor.name, shape, *precision, bytes)?;
                    digests.push(dense_digest(shape, *precision, bytes)?);
                }
                TensorStorage::Additive { .. } => {
                    digests.push(additive_semantic_digest(&decode_additive(tensor, bytes)?)?);
                }
                _ => return Err(TritModelError::UnsupportedFeature("tensor storage")),
            }
        }
        model.tensor_digests.push(digests);
        if let LevelFit::ImportedPrefixOf { level: parent } = &level.fit {
            let parent = model.level(parent)?;
            for (child, parent) in level.tensors.iter().zip(&parent.tensors) {
                if !is_prefix(
                    child,
                    model.tensor_payload(child)?,
                    parent,
                    model.tensor_payload(parent)?,
                )? {
                    return Err(TritModelError::InvalidPrefix(level.name.clone()));
                }
            }
        }
    }
    model.model_ids = model
        .manifest
        .levels
        .iter()
        .zip(&model.tensor_digests)
        .map(|(level, digests)| model_digest(&model.manifest, level, digests))
        .collect::<Result<_, _>>()?;
    Ok(model)
}

fn metadata_hasher<T: serde::Serialize>(
    context: &str,
    metadata: &T,
) -> Result<blake3::Hasher, TritModelError> {
    let mut value =
        Value::serialized(metadata).map_err(|e| TritModelError::Schema(e.to_string()))?;
    canonicalize_value(&mut value, 0)?;
    let bytes = encode_cbor_value(&value)?;
    let mut hash = blake3::Hasher::new_derive_key(context);
    hash.update(&frame_length(bytes.len())?);
    hash.update(&bytes);
    Ok(hash)
}

fn frame_length(length: usize) -> Result<[u8; 8], TritModelError> {
    u64::try_from(length)
        .map(u64::to_le_bytes)
        .map_err(|_| TritModelError::Schema("semantic frame length exceeds u64".into()))
}

/// Candidate codec/transport-independent digest of validated logical content.
///
/// Hashes canonical Rust-derived metadata, plane-major trits and exact stored
/// scale bits with bounded staging. Does not materialize a dense tensor/preimage.
/// Unused maximum capacity is excluded. Profile remains proposed (ADR 0054).
/// # Errors
/// Metadata encoding or unsupported scale precision.
pub fn additive_semantic_digest(
    tensor: &AdditiveTensor,
) -> Result<SemanticTensorDigest, TritModelError> {
    let layout = tensor.layout();
    let metadata = TensorSemanticMeta::Additive {
        rows: layout.rows,
        cols: layout.cols,
        tile: layout.tile,
        group: layout.group,
        planes: PlaneCounts::Uniform(tensor.plane_count()),
        law: layout.law,
        basis: layout.basis,
    };
    let mut hash = metadata_hasher("tritium.trit.additive.v1", &metadata)?;
    hash.update(&frame_length(tensor.trits().len())?);
    let mut staging = [0u8; 1024];
    for chunk in tensor.trits().chunks(staging.len()) {
        for (i, trit) in chunk.iter().enumerate() {
            staging[i] = (trit.get() + 1) as u8;
        }
        hash.update(&staging[..chunk.len()]);
    }
    let width = match layout.law.precision {
        ScalePrecision::F16 | ScalePrecision::Bf16 => 2,
        ScalePrecision::F32 => 4,
        _ => {
            return Err(TritModelError::UnsupportedFeature(
                "semantic scale precision",
            ));
        }
    };
    let scale_bytes = tensor
        .scales()
        .len()
        .checked_mul(width)
        .ok_or_else(|| TritModelError::Schema("semantic scale length overflow".into()))?;
    hash.update(&frame_length(scale_bytes)?);
    for chunk in tensor.scales().chunks(staging.len() / width) {
        for (i, scale) in chunk.iter().enumerate() {
            let target = &mut staging[i * width..(i + 1) * width];
            match layout.law.precision {
                ScalePrecision::F16 => {
                    target.copy_from_slice(&f16::from_f32(*scale).to_bits().to_le_bytes())
                }
                ScalePrecision::Bf16 => {
                    target.copy_from_slice(&bf16::from_f32(*scale).to_bits().to_le_bytes())
                }
                ScalePrecision::F32 => target.copy_from_slice(&scale.to_bits().to_le_bytes()),
                _ => {
                    return Err(TritModelError::UnsupportedFeature(
                        "semantic scale precision",
                    ));
                }
            }
        }
        hash.update(&staging[..chunk.len() * width]);
    }
    Ok(SemanticTensorDigest::from_bytes(
        *hash.finalize().as_bytes(),
    ))
}

fn dense_digest(
    shape: &[u64],
    precision: ScalePrecision,
    bytes: &[u8],
) -> Result<SemanticTensorDigest, TritModelError> {
    let metadata = TensorSemanticMeta::Dense {
        shape: shape.to_vec(),
        precision,
    };
    let mut hash = metadata_hasher("tritium.trit.dense.v1", &metadata)?;
    hash.update(&frame_length(bytes.len())?);
    hash.update(bytes);
    Ok(SemanticTensorDigest::from_bytes(
        *hash.finalize().as_bytes(),
    ))
}

fn model_digest(
    manifest: &ModelManifest,
    level: &Level,
    digests: &[SemanticTensorDigest],
) -> Result<ModelId, TritModelError> {
    let metadata = ModelSemanticMeta {
        architecture: manifest.architecture.clone(),
        assets: manifest
            .assets
            .iter()
            .map(|a| NamedAssetDigest {
                name: a.name.clone(),
                content: a.blob.id,
            })
            .collect(),
        tensors: level
            .tensors
            .iter()
            .zip(digests)
            .map(|(t, &content)| NamedTensorDigest {
                name: t.name.clone(),
                content,
            })
            .collect(),
    };
    Ok(ModelId::from_bytes(
        *metadata_hasher("tritium.trit.model.v1", &metadata)?
            .finalize()
            .as_bytes(),
    ))
}

fn decode_additive(meta: &TensorMeta, bytes: &[u8]) -> Result<AdditiveTensor, TritModelError> {
    let TensorStorage::Additive { layout, planes } = &meta.storage else {
        return Err(TritModelError::UnsupportedFeature("not additive"));
    };
    if !matches!(layout.transport, Transport::Raw) {
        return Err(TritModelError::UnsupportedFeature("entropy model payloads"));
    }
    let PlaneCounts::Uniform(count) = planes else {
        return Err(TritModelError::UnsupportedFeature(
            "per-tile model payloads",
        ));
    };
    AdditiveTensor::decode_payload(*layout, *count, bytes).map_err(TritModelError::Additive)
}

fn validate_dense(
    name: &str,
    shape: &[u64],
    precision: ScalePrecision,
    bytes: &[u8],
) -> Result<(), TritModelError> {
    let width: u64 = match precision {
        ScalePrecision::F16 | ScalePrecision::Bf16 => 2,
        ScalePrecision::F32 => 4,
        _ => return Err(TritModelError::UnsupportedFeature("dense precision")),
    };
    let expected = shape
        .iter()
        .try_fold(width, |n, &d| n.checked_mul(d))
        .and_then(|n| usize::try_from(n).ok());
    if expected != Some(bytes.len()) {
        return Err(TritModelError::DensePayload(name.into()));
    }
    let finite = match precision {
        ScalePrecision::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| f16::from_bits(u16::from_le_bytes(*b)).is_finite()),
        ScalePrecision::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| bf16::from_bits(u16::from_le_bytes(*b)).is_finite()),
        ScalePrecision::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|b| f32::from_le_bytes(*b).is_finite()),
        _ => false,
    };
    if !finite {
        return Err(TritModelError::DensePayload(name.into()));
    }
    Ok(())
}

fn is_prefix(
    child: &TensorMeta,
    child_bytes: &[u8],
    parent: &TensorMeta,
    parent_bytes: &[u8],
) -> Result<bool, TritModelError> {
    match (&child.storage, &parent.storage) {
        (TensorStorage::Dense { .. }, TensorStorage::Dense { .. }) => {
            Ok(child.storage == parent.storage && child_bytes == parent_bytes)
        }
        (TensorStorage::Additive { layout: c, .. }, TensorStorage::Additive { layout: p, .. }) => {
            let mut normalized = *p;
            normalized.max_planes = c.max_planes;
            normalized.codec = c.codec;
            if normalized != *c || c.max_planes > p.max_planes {
                return Ok(false);
            }
            let child = decode_additive(child, child_bytes)?;
            let parent = decode_additive(parent, parent_bytes)?;
            Ok(parent.trits().starts_with(child.trits())
                && parent
                    .scales()
                    .iter()
                    .map(|s| s.to_bits())
                    .take(child.scales().len())
                    .eq(child.scales().iter().map(|s| s.to_bits())))
        }
        _ => Ok(false),
    }
}
