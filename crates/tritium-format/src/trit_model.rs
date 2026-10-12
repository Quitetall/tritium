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
    canonicalize_value, encode_borrowed_manifest, encode_cbor_value, parse_canonical_manifest,
    validate_canonical_manifest,
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
/// Already sorted inventories borrow the schema and reorder encoded map spans,
/// without cloning PerTile counts or constructing a Value per count. Unsorted
/// compatibility normalization still clones metadata; output capacity and typed
/// decoding remain separate from this encoder's temporary-storage behavior.
///
/// # Errors
/// Structural, serde or canonical-envelope errors.
pub fn encode_model_manifest(manifest: &ModelManifest) -> Result<Vec<u8>, TritModelError> {
    // Migration builds sorted inventories. Borrow them directly: a PerTile
    // count array must not become a cloned Vec or a Value for every integer.
    // Compatibility normalization of unsorted inputs stays on the old path.
    if manifest.validate().is_ok()
        && let Some(bytes) = encode_borrowed_manifest(manifest)?
    {
        return Ok(bytes);
    }
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

    /// Require every physical directory entry to be used by typed model metadata.
    /// Shared/deduplicated references remain valid. Ordinary compatibility readers
    /// need not require this; mandatory joined readers do.
    /// # Errors
    /// Missing typed binding or any unreferenced physical entry.
    pub fn validate_complete_payload_inventory(&self) -> Result<(), TritModelError> {
        let mut referenced = vec![false; self.package.blobs().len()];
        for asset in &self.manifest.assets {
            referenced[self.payload_index(
                asset.blob.id,
                &TritBlobKind::Asset(asset.name.clone()),
                &asset.name,
            )?] = true;
        }
        for level in &self.manifest.levels {
            for tensor in &level.tensors {
                referenced[self.tensor_index(tensor)?] = true;
            }
        }
        if referenced.contains(&false) {
            return Err(TritModelError::Schema("unreferenced model payload".into()));
        }
        Ok(())
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
        Ok(self
            .package
            .indexed_blob(self.payload_index(id, kind, name)?)?)
    }

    fn payload_index(
        &self,
        id: BlobId,
        kind: &TritBlobKind,
        name: &str,
    ) -> Result<usize, TritModelError> {
        self.index
            .get(id.as_bytes())
            .and_then(|indices| {
                indices
                    .iter()
                    .copied()
                    .find(|&i| &self.package.blobs()[i].kind == kind)
            })
            .ok_or_else(|| TritModelError::MissingOrWrongBlob(name.into()))
    }

    fn tensor_payload(&self, meta: &TensorMeta) -> Result<&'a [u8], TritModelError> {
        Ok(self.package.indexed_blob(self.tensor_index(meta)?)?)
    }

    fn tensor_index(&self, meta: &TensorMeta) -> Result<usize, TritModelError> {
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
        self.payload_index(meta.blob.id, &kind, &meta.name)
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
    bind_trit_model_payloads(package, manifest)
}

/// Bind a decoded typed manifest through the shared model payload validator.
/// Does not decode/admit the package's mandatory metadata profile or external logs.
/// Ordinary and joined readers must admit their profile before calling this seam.
/// The package remains borrowed; no encoded package or dense model clone is made.
/// # Errors
/// Invalid typed metadata, named payload/semantics/prefix or unsupported capability.
pub fn bind_trit_model_payloads<'a>(
    package: TritPackage<'a>,
    manifest: ModelManifest,
) -> Result<TritModel<'a>, TritModelError> {
    manifest.validate()?;
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
                    digests.push(additive_semantic_digest(&decode_additive_payload(
                        tensor, bytes,
                    )?)?);
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
/// Binds exact Uniform/PerTile counts, excluding unused capacity and derived
/// runtime indexes. Profile remains proposed (ADRs 0054 and 0066).
/// # Errors
/// Count reservation, metadata encoding or unsupported scale precision.
pub fn additive_semantic_digest(
    tensor: &AdditiveTensor,
) -> Result<SemanticTensorDigest, TritModelError> {
    let layout = tensor.layout();
    let planes = if let Some(counts) = tensor.tile_counts() {
        let mut owned = Vec::new();
        owned.try_reserve_exact(counts.len()).map_err(|_| {
            TritModelError::Additive(AdditiveTensorError::AllocationFailed {
                requested: counts.len(),
            })
        })?;
        owned.extend_from_slice(counts);
        PlaneCounts::PerTile(owned)
    } else {
        PlaneCounts::Uniform(tensor.plane_count())
    };
    let metadata = TensorSemanticMeta::Additive {
        rows: layout.rows,
        cols: layout.cols,
        tile: layout.tile,
        group: layout.group,
        planes,
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

/// Decode a declared raw additive payload with its Uniform or PerTile counts.
///
/// Reuses the same canonical decoder as typed model validation and prefix checks.
/// Reservations are fallible and no dense matrix is materialized. This validates
/// payload/layout/count structure, not blob, package, model or provenance identity;
/// a model loader must authenticate the complete source through [`read_trit_model`]
/// first. The metadata's count variant is never inferred from the physical bytes.
///
/// # Errors
/// Non-additive metadata, unsupported transport/allocation, invalid layout/counts,
/// malformed physical payload/codes/padding/scales or allocation failure.
pub fn decode_additive_payload(
    meta: &TensorMeta,
    bytes: &[u8],
) -> Result<AdditiveTensor, TritModelError> {
    let TensorStorage::Additive { layout, planes } = &meta.storage else {
        return Err(TritModelError::UnsupportedFeature("not additive"));
    };
    if !matches!(layout.transport, Transport::Raw) {
        return Err(TritModelError::UnsupportedFeature("entropy model payloads"));
    }
    match planes {
        PlaneCounts::Uniform(count) => AdditiveTensor::decode_payload(*layout, *count, bytes),
        PlaneCounts::PerTile(counts) => {
            AdditiveTensor::decode_per_tile_payload(*layout, counts, bytes)
        }
        _ => return Err(TritModelError::UnsupportedFeature("plane allocation")),
    }
    .map_err(TritModelError::Additive)
}

/// Validate original raw dense bytes without changing precision or bit patterns.
/// # Errors
/// Invalid shape, length, nonfinite values or unsupported precision.
pub fn validate_dense_payload(
    name: &str,
    shape: &[u64],
    precision: ScalePrecision,
    bytes: &[u8],
) -> Result<(), TritModelError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(TritModelError::DensePayload(name.into()));
    }
    validate_dense(name, shape, precision, bytes)
}

/// Widen the same validated raw payload for the production f32 fitter.
/// # Errors
/// Invalid payload or failed bounded output allocation.
pub fn decode_dense_f32(
    name: &str,
    shape: &[u64],
    precision: ScalePrecision,
    bytes: &[u8],
) -> Result<Vec<f32>, TritModelError> {
    validate_dense_payload(name, shape, precision, bytes)?;
    let width = if precision == ScalePrecision::F32 {
        4
    } else {
        2
    };
    let count = bytes.len() / width;
    let requested = count
        .checked_mul(4)
        .ok_or(TritPackageError::LengthOverflow)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| TritPackageError::AllocationFailed { requested })?;
    for bytes in bytes.chunks_exact(width) {
        values.push(match precision {
            ScalePrecision::F32 => f32::from_le_bytes(bytes.try_into().expect("four bytes")),
            ScalePrecision::F16 => {
                f16::from_bits(u16::from_le_bytes(bytes.try_into().expect("two bytes"))).to_f32()
            }
            ScalePrecision::Bf16 => {
                bf16::from_bits(u16::from_le_bytes(bytes.try_into().expect("two bytes"))).to_f32()
            }
            _ => return Err(TritModelError::UnsupportedFeature("dense precision")),
        });
    }
    Ok(values)
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
            // Cross-allocation prefixes use logical coordinates, not physical
            // offsets. Preserve every other prior layout/capacity constraint.
            normalized.allocation = c.allocation;
            if normalized != *c || c.max_planes > p.max_planes {
                return Ok(false);
            }
            let child = decode_additive_payload(child, child_bytes)?;
            let parent = decode_additive_payload(parent, parent_bytes)?;
            Ok(child.view().is_logical_prefix_of(parent.view()))
        }
        _ => Ok(false),
    }
}
