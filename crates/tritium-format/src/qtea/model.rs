//! Named residual-bearing model profile over the existing `.trit` envelope.
//! Content agreement is not original-checkpoint or model-quality qualification.
use super::{QteaLimits, QteaTensor};
use crate::{PackageId, TritBlobKind, TritPackageReadLimits, TritPackageReader};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Seek},
};
use tritium_core::DType;
use tritium_schema::{BlobId, ScalePrecision};

/// Standard-transformer residual-bearing model manifest profile.
pub const QTEA_MODEL_PROFILE: &str = "tritium.qtea-model.v1";
/// Separate exact-shaped hybrid language/MTP profile; not admitted by standard readers.
pub const QTEA_HYBRID_MODEL_PROFILE: &str = "tritium.qtea-hybrid-model.v1";

/// One exact named physical owner. Matrices cannot use a dense fallback.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum QteaModelTensor {
    /// Complete canonical QTEA tensor bytes.
    Qtea {
        /// Exact stored blob identity.
        blob: BlobId,
        /// Logical output rows.
        rows: u64,
        /// Logical input columns.
        columns: u64,
    },
    /// Preserved norm/bias vector, not a matrix.
    Vector {
        /// Exact stored blob identity.
        blob: BlobId,
        /// Element count.
        length: u64,
        /// Original stored floating-point precision.
        precision: ScalePrecision,
    },
    /// Exact higher-rank non-matrix state, only in the explicitly hybrid profile.
    /// Architecture admission must still prove this is a preserved role, not a
    /// dense projection fallback. Rank is bounded to2..=4 by the wire reader.
    Preserved {
        /// Exact stored blob identity.
        blob: BlobId,
        /// Original unflattened positive dimensions.
        shape: Vec<u64>,
        /// Original finite floating-point precision.
        precision: ScalePrecision,
    },
}
impl QteaModelTensor {
    /// Exact original blob identity, not a source-owner attestation.
    pub const fn blob(&self) -> BlobId {
        match self {
            Self::Qtea { blob, .. } | Self::Vector { blob, .. } | Self::Preserved { blob, .. } => {
                *blob
            }
        }
    }
    fn kind(&self) -> Result<TritBlobKind, QteaModelError> {
        Ok(match self {
            Self::Qtea { .. } => TritBlobKind::QteaTensor,
            Self::Vector { precision, .. } | Self::Preserved { precision, .. } => {
                TritBlobKind::DenseTensor(match precision {
                    ScalePrecision::F16 => DType::F16,
                    ScalePrecision::Bf16 => DType::BF16,
                    ScalePrecision::F32 => DType::F32,
                    _ => return Err(invalid("unsupported preserved precision")),
                })
            }
        })
    }
}

/// Canonical named inventory. Tied owners are determined by the bound config.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QteaModelManifest {
    /// Must equal [`QTEA_MODEL_PROFILE`].
    pub profile: String,
    /// Must equal1.
    pub version: u32,
    /// Declared original source digest, not independent custody.
    pub source_model: [u8; 32],
    /// Exact `config.json` asset blob.
    pub config: BlobId,
    /// Other exact assets (for example tokenizers), excluding `config.json`.
    pub assets: BTreeMap<String, BlobId>,
    /// Canonical owner names, in lexical order.
    pub tensors: BTreeMap<String, QteaModelTensor>,
}

/// Separate logical receiver caps; metadata heap and physical RSS are not measured.
#[derive(Clone, Copy, Debug)]
pub struct QteaModelReadLimits {
    /// Maximum declared physical owner count.
    pub max_tensors: usize,
    /// Maximum additional asset count.
    pub max_assets: usize,
    /// Maximum bytes in any additional asset.
    pub max_asset_bytes: usize,
    /// Maximum config bytes.
    pub max_config_bytes: usize,
    /// Maximum elements in one preserved vector.
    pub max_vector_elements: usize,
    /// Existing per-matrix encoded and validation ceilings.
    pub tensor: QteaLimits,
}

/// Explicit hybrid receiver; existing vector caps retain their original domain.
#[derive(Clone, Copy, Debug)]
pub struct QteaHybridModelReadLimits {
    /// Common source-independent owner/asset/vector/QTEA ceilings.
    pub base: QteaModelReadLimits,
    /// Positive ceiling for elements in each higher-rank preserved tensor.
    /// Not an aggregate payload, decode-transient or RSS measurement.
    pub max_preserved_tensor_elements: usize,
}

/// Profile, format, read or allocation refusal. Never a qualification verdict.
#[derive(Debug)]
pub struct QteaModelError(String);
impl core::fmt::Display for QteaModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for QteaModelError {}
fn invalid(message: &str) -> QteaModelError {
    QteaModelError(message.into())
}
fn failure(error: impl core::fmt::Display) -> QteaModelError {
    QteaModelError(error.to_string())
}

fn preserved_elements(shape: &[u64]) -> Result<u64, QteaModelError> {
    if !(2..=4).contains(&shape.len()) || shape.contains(&0) {
        return Err(invalid(
            "preserved tensor requires positive rank2..=4 shape",
        ));
    }
    shape.iter().try_fold(1_u64, |n, d| {
        n.checked_mul(*d)
            .ok_or_else(|| invalid("preserved tensor shape overflow"))
    })
}
fn validate(manifest: &QteaModelManifest, hybrid: bool) -> Result<(), QteaModelError> {
    if manifest.profile
        != if hybrid {
            QTEA_HYBRID_MODEL_PROFILE
        } else {
            QTEA_MODEL_PROFILE
        }
        || manifest.version != 1
        || manifest.tensors.is_empty()
        || manifest.source_model == [0; 32]
    {
        return Err(invalid("unsupported or empty QTEA model profile"));
    }
    if manifest.assets.keys().any(|name| {
        name == "config.json"
            || name.is_empty()
            || name.len() > 1024
            || name.chars().any(char::is_control)
    }) {
        return Err(invalid("invalid or ambiguous QTEA asset name"));
    }
    for (name, tensor) in &manifest.tensors {
        if name.is_empty() || name.len() > 1024 || name.chars().any(char::is_control) {
            return Err(invalid("invalid QTEA owner name"));
        }
        match tensor {
            QteaModelTensor::Qtea { rows, columns, .. } if *rows == 0 || *columns == 0 => {
                return Err(invalid("empty QTEA geometry"));
            }
            QteaModelTensor::Vector { length: 0, .. } => {
                return Err(invalid("empty preserved vector"));
            }
            QteaModelTensor::Preserved { shape, .. } => {
                if !hybrid {
                    return Err(invalid(
                        "standard QTEA profile forbids higher-rank preserved state",
                    ));
                }
                preserved_elements(shape)?;
            }
            _ => {}
        }
        tensor.kind()?;
    }
    Ok(())
}

/// Encode only this canonical profile under an explicit encoded-metadata cap.
/// Does not validate supplied payloads, provenance or complete architecture coverage.
/// # Errors
/// Invalid profile/metadata, allocation or encoded cap failure.
pub fn encode_qtea_model_manifest(
    manifest: &QteaModelManifest,
    max_bytes: usize,
) -> Result<Vec<u8>, QteaModelError> {
    encode_profile(manifest, max_bytes, false)
}
/// Encode the separate hybrid inventory, retaining exact preserved state shapes.
/// Does not prove architecture coverage, numerical execution or source custody.
/// # Errors
/// Invalid profile/rank/shape/precision, allocation or encoded metadata cap.
pub fn encode_qtea_hybrid_model_manifest(
    manifest: &QteaModelManifest,
    max_bytes: usize,
) -> Result<Vec<u8>, QteaModelError> {
    encode_profile(manifest, max_bytes, true)
}
fn encode_profile(
    manifest: &QteaModelManifest,
    max_bytes: usize,
    hybrid: bool,
) -> Result<Vec<u8>, QteaModelError> {
    validate(manifest, hybrid)?;
    crate::trit_package::encode_borrowed_metadata(manifest, max_bytes)
        .map_err(failure)?
        .ok_or_else(|| invalid("unsupported QTEA metadata encoding"))
}

/// Retained source plus admitted named metadata. Payload numerics remain lazy.
/// Keep all returned work private until [`Self::verify`] succeeds; external
/// writers still require immutable source custody, not just a retained handle.
pub struct QteaModelReader<R> {
    package: TritPackageReader<R>,
    manifest: QteaModelManifest,
    indices: BTreeMap<String, usize>,
    config_index: usize,
    asset_indices: BTreeMap<String, usize>,
    limits: QteaModelReadLimits,
    preserved_cap: Option<usize>,
}
impl<R> core::fmt::Debug for QteaModelReader<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QteaModelReader")
            .field("tensors", &self.manifest.tensors.len())
            .finish_non_exhaustive()
    }
}

impl<R: Read + Seek> QteaModelReader<R> {
    /// Admit the outer source and exact named directory under explicit caps.
    /// Expected source agreement is not authentication of that assertion.
    /// # Errors
    /// Invalid/noncanonical metadata, wrong/missing/extra kinds, source mismatch,
    /// receiver limits, changed source or I/O. Numeric payload checks are lazy.
    pub fn open(
        source: R,
        receiving: TritPackageReadLimits,
        limits: QteaModelReadLimits,
        expected_source_model: [u8; 32],
    ) -> Result<Self, QteaModelError> {
        Self::open_profile(source, receiving, limits, expected_source_model, None)
    }
    /// Admit only the explicit hybrid profile under its separate preserved cap.
    /// # Errors
    /// Standard open errors, zero preserved cap or unsupported profile/rank/shape.
    pub fn open_hybrid(
        source: R,
        receiving: TritPackageReadLimits,
        limits: QteaHybridModelReadLimits,
        expected_source_model: [u8; 32],
    ) -> Result<Self, QteaModelError> {
        if limits.max_preserved_tensor_elements == 0 {
            return Err(invalid("hybrid preserved tensor cap must be positive"));
        }
        Self::open_profile(
            source,
            receiving,
            limits.base,
            expected_source_model,
            Some(limits.max_preserved_tensor_elements),
        )
    }
    fn open_profile(
        source: R,
        receiving: TritPackageReadLimits,
        limits: QteaModelReadLimits,
        expected_source_model: [u8; 32],
        preserved_cap: Option<usize>,
    ) -> Result<Self, QteaModelError> {
        let package = TritPackageReader::open(source, receiving).map_err(failure)?;
        let manifest: QteaModelManifest =
            ciborium::de::from_reader(package.manifest()).map_err(failure)?;
        validate(&manifest, preserved_cap.is_some())?;
        if encode_profile(
            &manifest,
            receiving.max_manifest_bytes,
            preserved_cap.is_some(),
        )? != package.manifest()
        {
            return Err(invalid("noncanonical QTEA manifest"));
        }
        if manifest.source_model != expected_source_model
            || manifest.tensors.len() > limits.max_tensors
            || manifest.assets.len() > limits.max_assets
        {
            return Err(invalid("QTEA source or owner-count admission mismatch"));
        }
        let mut used = BTreeSet::new();
        let mut directory = BTreeMap::<[u8; 32], Vec<usize>>::new();
        for (index, blob) in package.blobs().iter().enumerate() {
            directory
                .entry(*blob.id.as_bytes())
                .or_default()
                .push(index);
        }
        let find = |id: BlobId, kind: &TritBlobKind| {
            directory
                .get(id.as_bytes())
                .and_then(|indices| {
                    indices
                        .iter()
                        .copied()
                        .find(|&index| &package.blobs()[index].kind == kind)
                })
                .ok_or_else(|| invalid("missing or wrong-kind QTEA model blob"))
        };
        let config_index = find(manifest.config, &TritBlobKind::Asset("config.json".into()))?;
        if package.blobs()[config_index].length > limits.max_config_bytes as u64 {
            return Err(invalid("QTEA config cap exceeded"));
        }
        used.insert(config_index);
        let mut asset_indices = BTreeMap::new();
        for (name, id) in &manifest.assets {
            let index = find(*id, &TritBlobKind::Asset(name.clone()))?;
            if package.blobs()[index].length > limits.max_asset_bytes as u64 {
                return Err(invalid("QTEA asset cap exceeded"));
            }
            used.insert(index);
            asset_indices.insert(name.clone(), index);
        }
        let mut indices = BTreeMap::new();
        for (name, tensor) in &manifest.tensors {
            let index = find(tensor.blob(), &tensor.kind()?)?;
            let bytes = package.blobs()[index].length;
            match tensor {
                QteaModelTensor::Qtea { rows, columns, .. } => {
                    usize::try_from(*rows).map_err(failure)?;
                    usize::try_from(*columns).map_err(failure)?;
                    if bytes > limits.tensor.max_tensor_bytes as u64 {
                        return Err(invalid("QTEA tensor cap exceeded"));
                    }
                }
                QteaModelTensor::Vector {
                    length, precision, ..
                } => {
                    let width = if *precision == ScalePrecision::F32 {
                        4
                    } else {
                        2
                    };
                    if *length > limits.max_vector_elements as u64
                        || length.checked_mul(width) != Some(bytes)
                    {
                        return Err(invalid("QTEA vector cap or length mismatch"));
                    }
                }
                QteaModelTensor::Preserved {
                    shape, precision, ..
                } => {
                    let elements = preserved_elements(shape)?;
                    let cap = preserved_cap.ok_or_else(|| {
                        invalid("higher-rank state requires explicit hybrid receiver")
                    })?;
                    let width = if *precision == ScalePrecision::F32 {
                        4
                    } else {
                        2
                    };
                    if elements > cap as u64 || elements.checked_mul(width) != Some(bytes) {
                        return Err(invalid(
                            "preserved tensor cap or exact byte-length mismatch",
                        ));
                    }
                    usize::try_from(elements).map_err(failure)?;
                }
            }
            used.insert(index);
            indices.insert(name.clone(), index);
        }
        if used.len() != package.blobs().len() {
            return Err(invalid("unreferenced QTEA model blob"));
        }
        Ok(Self {
            package,
            manifest,
            indices,
            config_index,
            asset_indices,
            limits,
            preserved_cap,
        })
    }

    /// Exact admitted named metadata, not an executable-model verdict.
    pub fn manifest(&self) -> &QteaModelManifest {
        &self.manifest
    }
    /// Exact package bytes identity, distinct from a semantic/original model ID.
    pub fn package_id(&self) -> PackageId {
        self.package.package_id()
    }
    /// Encoded length of one admitted owner payload.
    /// # Errors
    /// Unknown owner name.
    pub fn tensor_bytes(&self, name: &str) -> Result<u64, QteaModelError> {
        let index = self
            .indices
            .get(name)
            .ok_or_else(|| invalid("unknown QTEA owner"))?;
        Ok(self.package.blobs()[*index].length)
    }
    fn bytes(&mut self, index: usize, cap: usize) -> Result<Vec<u8>, QteaModelError> {
        let count = usize::try_from(self.package.blobs()[index].length).map_err(failure)?;
        if count > cap {
            return Err(invalid("QTEA payload cap exceeded"));
        }
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(count).map_err(failure)?;
        if bytes.capacity() > cap {
            return Err(invalid("QTEA payload capacity exceeded"));
        }
        self.package
            .visit_blob(index, |chunk| {
                bytes.extend_from_slice(chunk);
                Ok::<_, std::convert::Infallible>(())
            })
            .map_err(failure)?;
        Ok(bytes)
    }
    /// Read the exact bound config asset, verifying bytes actually consumed.
    /// # Errors
    /// Changed source, I/O, allocation or config cap refusal.
    pub fn config(&mut self) -> Result<Vec<u8>, QteaModelError> {
        self.bytes(self.config_index, self.limits.max_config_bytes)
    }
    /// Read a bound additional asset, verifying bytes actually consumed.
    /// # Errors
    /// Unknown name, caps, changed source, allocation or I/O.
    pub fn asset(&mut self, name: &str) -> Result<Vec<u8>, QteaModelError> {
        let index = *self
            .asset_indices
            .get(name)
            .ok_or_else(|| invalid("unknown QTEA asset"))?;
        self.bytes(index, self.limits.max_asset_bytes)
    }
    /// Admitted encoded size of an additional asset, without materializing it.
    /// # Errors
    /// Unknown additional asset name (config is queried separately).
    pub fn asset_bytes(&self, name: &str) -> Result<u64, QteaModelError> {
        let index = *self
            .asset_indices
            .get(name)
            .ok_or_else(|| invalid("unknown QTEA asset"))?;
        Ok(self.package.blobs()[index].length)
    }
    /// Read actual packed bytes, checking geometry and declared source agreement.
    /// # Errors
    /// Wrong owner role, malformed wire, mismatch, caps, mutation or I/O.
    pub fn qtea(&mut self, name: &str) -> Result<QteaTensor, QteaModelError> {
        let Some(QteaModelTensor::Qtea { rows, columns, .. }) = self.manifest.tensors.get(name)
        else {
            return Err(invalid("owner is not QTEA"));
        };
        let (rows, columns) = (*rows, *columns);
        let index = self.indices[name];
        let bytes = self.bytes(index, self.limits.tensor.max_tensor_bytes)?;
        let tensor = QteaTensor::from_owned_bytes(bytes, self.limits.tensor).map_err(failure)?;
        let meta = tensor.view().metadata();
        if meta.rows as u64 != rows
            || meta.columns as u64 != columns
            || meta.bindings.source_model != self.manifest.source_model
        {
            return Err(invalid("QTEA payload geometry/source mismatch"));
        }
        Ok(tensor)
    }
    /// Decode one preserved finite floating-point vector, never a dense matrix.
    /// # Errors
    /// Wrong role, nonfinite/invalid bytes, cap/allocation, mutation or I/O.
    pub fn vector(&mut self, name: &str) -> Result<Vec<f32>, QteaModelError> {
        let Some(QteaModelTensor::Vector {
            length, precision, ..
        }) = self.manifest.tensors.get(name)
        else {
            return Err(invalid("owner is not a preserved vector"));
        };
        let (length, precision) = (*length, *precision);
        let cap = self
            .limits
            .max_vector_elements
            .checked_mul(4)
            .ok_or_else(|| invalid("vector cap overflow"))?;
        let bytes = self.bytes(self.indices[name], cap)?;
        crate::trit_model::decode_dense_f32(name, &[length], precision, &bytes).map_err(failure)
    }
    /// Decode finite exact-shaped non-matrix state from the hybrid profile only.
    /// # Errors
    /// Wrong role/profile, shape/byte/cap mismatch, nonfinite values, allocation or I/O.
    pub fn preserved(&mut self, name: &str) -> Result<Vec<f32>, QteaModelError> {
        let Some(QteaModelTensor::Preserved {
            shape, precision, ..
        }) = self.manifest.tensors.get(name)
        else {
            return Err(invalid("owner is not exact-shaped preserved state"));
        };
        let shape = shape.clone();
        let precision = *precision;
        let cap = self
            .preserved_cap
            .ok_or_else(|| invalid("hybrid receiver required"))?
            .checked_mul(4)
            .ok_or_else(|| invalid("preserved decode cap overflow"))?;
        let bytes = self.bytes(self.indices[name], cap)?;
        crate::trit_model::decode_dense_f32(name, &shape, precision, &bytes).map_err(failure)
    }
    /// Terminal whole-source recheck before publishing assembled weights.
    /// # Errors
    /// Changed source/length or actual read/seek failure.
    pub fn verify(&mut self) -> Result<(), QteaModelError> {
        self.package.verify().map_err(failure)
    }
}
