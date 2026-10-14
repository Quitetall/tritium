//! Strict typed model references over the existing `.trit` envelope (ADR 0044).
//! This proposed profile proves metadata/payload consistency, not model execution,
//! external evidence, owner acceptance of candidate identities, quality or release admission.

mod binding;
mod seek_reader;
mod sizing;
mod stage;

pub use seek_reader::{TritModelReadError, TritModelReader, TritModelVisitError, TritTensorStream};
pub use sizing::{imported_manifest_cbor_size_bounds, tensor_meta_cbor_size_bounds};
pub use stage::{TritModelStageError, stage_trit_model_from_spool};

use ciborium::value::Value;
use half::{bf16, f16};
use tritium_core::DType;
use tritium_schema::{
    BlobId, Level, LevelFit, ModelId, ModelManifest, ModelManifestError, ModelSemanticMeta,
    NamedAssetDigest, NamedTensorDigest, PlaneCounts, ScalePrecision, SemanticTensorDigest,
    TensorMeta, TensorSemanticMeta, TensorStorage, Transport,
};

use crate::trit_package::{
    canonicalize_value, encode_borrowed_manifest, encode_borrowed_metadata, encode_cbor_value,
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
impl From<AdditiveTensorError> for TritModelError {
    fn from(error: AdditiveTensorError) -> Self {
        Self::Additive(error)
    }
}
impl From<ModelManifestError> for TritModelError {
    fn from(error: ModelManifestError) -> Self {
        Self::Manifest(error)
    }
}

/// Exact encoded length of one storage fragment in a typed model manifest.
///
/// Uses the schema's actual serializer without allocating encoded bytes or
/// cloning counts. Canonical ordering only rearranges this primitive-key,
/// integer-only schema and cannot change its length. This is not structural
/// validation, a complete manifest estimate or payload/import qualification.
/// A manifest includes every storage fragment, so their sum is a lower bound.
/// # Errors
/// Serializer failure or checked byte-counter overflow.
pub fn tensor_storage_cbor_bytes(storage: &TensorStorage) -> Result<u64, TritModelError> {
    metadata_cbor_bytes(storage)
}

pub(crate) fn metadata_cbor_bytes<T: serde::Serialize>(
    metadata: &T,
) -> Result<u64, TritModelError> {
    let mut output = CountOutput(0);
    ciborium::ser::into_writer(metadata, &mut output)
        .map_err(|e| TritModelError::Schema(e.to_string()))?;
    Ok(output.0)
}

struct CountOutput(u64);
impl std::io::Write for CountOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| std::io::Error::other("metadata size overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod sizing_tests {
    use super::*;
    use tritium_schema::{
        AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
    };

    fn check(storage: TensorStorage) {
        let mut reference = Value::serialized(&storage).unwrap();
        canonicalize_value(&mut reference, 0).unwrap();
        let bytes = encode_cbor_value(&reference).unwrap();
        assert_eq!(
            tensor_storage_cbor_bytes(&storage).unwrap(),
            bytes.len() as u64
        );
    }

    #[test]
    fn storage_size_matches_reference_across_cbor_widths() {
        for n in [1, 23, 24, 255, 256, 65_535, 65_536] {
            for precision in [
                ScalePrecision::F16,
                ScalePrecision::Bf16,
                ScalePrecision::F32,
            ] {
                check(TensorStorage::Dense {
                    shape: vec![n, 3],
                    precision,
                });
            }
        }
        for len in [1, 23, 24, 255, 256, 65_535, 65_536] {
            let layout = AdditiveLayout {
                rows: len,
                cols: 256,
                tile: 256,
                group: 128,
                max_planes: 3,
                allocation: PlaneAllocation::PerTile,
                codec: PlaneCodec::B3,
                law: ScaleLaw {
                    anchor: ScaleAnchor::Group,
                    relation: PlaneRelation::Free,
                    precision: ScalePrecision::F16,
                },
                basis: Basis::Identity,
                transport: Transport::Raw,
            };
            check(TensorStorage::Additive {
                layout,
                planes: PlaneCounts::PerTile((0..len).map(|i| (i % 4) as u8).collect()),
            });
        }
    }

    #[test]
    fn checked_counter_rejects_overflow_without_mutation() {
        use std::io::Write;
        let mut output = CountOutput(u64::MAX);
        assert!(output.write(&[0]).is_err());
        assert_eq!(output.0, u64::MAX);
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
/// Ordinary metadata is deserialized directly into the schema after borrowed
/// envelope validation, without a generic Value per count. Returned metadata
/// owns its inventories; exact re-encoding temporarily retains encoded bytes.
/// This does not qualify complete-model or process memory admission.
///
/// # Errors
/// Unknown versions/tags/fields, invalid structure or noncanonical byte/order data.
pub fn decode_model_manifest(bytes: &[u8]) -> Result<ModelManifest, TritModelError> {
    validate_canonical_manifest(bytes)?;
    let manifest: ModelManifest =
        ciborium::de::from_reader(bytes).map_err(|e| TritModelError::Schema(e.to_string()))?;
    manifest.validate()?;
    if encode_model_manifest(&manifest)? != bytes {
        return Err(TritModelError::NonCanonical);
    }
    Ok(manifest)
}

/// Borrowed model with shared typed bindings and bounded payload validation.
/// Successful parsing is not execution, provenance or release qualification.
#[derive(Debug)]
pub struct TritModel<'a> {
    package: TritPackage<'a>,
    binding: binding::ModelBindings,
}
impl<'a> TritModel<'a> {
    /// Immutable validated metadata, not model qualification.
    #[must_use]
    pub const fn manifest(&self) -> &ModelManifest {
        &self.binding.manifest
    }
    /// Exact stored package identity, not a semantic ModelId.
    #[must_use]
    pub fn package_id(&self) -> crate::PackageId {
        self.package.package_id()
    }
    /// Candidate level identity; absent levels are rejected.
    pub fn model_id(&self, level: &str) -> Result<ModelId, TritModelError> {
        self.binding.model_id(level)
    }
    /// Candidate logical tensor digest; absent bindings are rejected.
    pub fn tensor_digest(
        &self,
        level: &str,
        name: &str,
    ) -> Result<SemanticTensorDigest, TritModelError> {
        self.binding.tensor_digest(level, name)
    }
    /// Borrow exact stored tensor bytes from a named complete level.
    pub fn tensor(
        &self,
        level: &str,
        name: &str,
    ) -> Result<(&TensorMeta, &'a [u8]), TritModelError> {
        let meta = self.binding.tensor_meta(level, name)?;
        let index = self.binding.tensor_index(meta, self.package.blobs())?;
        Ok((meta, self.package.indexed_blob(index)?))
    }
    /// Borrow a declared asset without filesystem extraction.
    pub fn asset(&self, name: &str) -> Result<&'a [u8], TritModelError> {
        let index = self.binding.asset_index(name, self.package.blobs())?;
        Ok(self.package.indexed_blob(index)?)
    }
    /// Require every physical directory record to be referenced by the model.
    pub fn validate_complete_payload_inventory(&self) -> Result<(), TritModelError> {
        self.binding
            .validate_complete_payload_inventory(self.package.blobs())
    }
}

/// Validate all complete levels and their payloads before publishing a typed view.
///
/// The current reader explicitly rejects sharded/entropy payloads rather
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
    let binding = binding::validate_bindings(manifest, &mut binding::BorrowedSource(&package))?;
    Ok(TritModel { package, binding })
}

fn metadata_hasher<T: serde::Serialize>(
    context: &str,
    metadata: &T,
) -> Result<blake3::Hasher, TritModelError> {
    let bytes = if let Some(bytes) = encode_borrowed_metadata(metadata, usize::MAX)? {
        bytes
    } else {
        let mut value =
            Value::serialized(metadata).map_err(|e| TritModelError::Schema(e.to_string()))?;
        canonicalize_value(&mut value, 0)?;
        encode_cbor_value(&value)?
    };
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
    validate_dense_length(name, shape, precision, bytes.len())?;
    if !dense_finite(precision, bytes) {
        return Err(TritModelError::DensePayload(name.into()));
    }
    Ok(())
}
fn validate_dense_length(
    name: &str,
    shape: &[u64],
    precision: ScalePrecision,
    length: usize,
) -> Result<(), TritModelError> {
    let width: u64 = match precision {
        ScalePrecision::F16 | ScalePrecision::Bf16 => 2,
        ScalePrecision::F32 => 4,
        _ => return Err(TritModelError::DensePayload(name.into())),
    };
    let expected = shape
        .iter()
        .try_fold(width, |n, &d| n.checked_mul(d))
        .and_then(|n| usize::try_from(n).ok());
    if expected != Some(length) {
        return Err(TritModelError::DensePayload(name.into()));
    }
    Ok(())
}
fn dense_finite(precision: ScalePrecision, bytes: &[u8]) -> bool {
    match precision {
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
    }
}
