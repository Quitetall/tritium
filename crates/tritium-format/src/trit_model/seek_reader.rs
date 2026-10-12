//! Retained range adapter over the shared typed model validator.
use super::binding::{ModelBindings, ModelSource, validate_bindings};
use super::*;
use crate::additive_tensor::{PayloadSource, RANGE_BYTES};
use crate::{
    TritBlobInfo, TritBlobVisitError, TritPackageReadError, TritPackageReadLimits,
    TritPackageReader,
};
use std::io::{Read, Seek};

/// Original model refusal or retained package/source failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum TritModelReadError {
    /// Shared typed metadata, payload, binding, semantic or prefix refusal.
    Model(TritModelError),
    /// Package framing, digest, length, admission or actual I/O failure.
    Source(TritPackageReadError),
}
impl core::fmt::Display for TritModelReadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Model(error) => error.fmt(f),
            Self::Source(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for TritModelReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Source(error) => Some(error),
        }
    }
}
impl From<TritModelError> for TritModelReadError {
    fn from(error: TritModelError) -> Self {
        Self::Model(error)
    }
}
impl From<AdditiveTensorError> for TritModelReadError {
    fn from(error: AdditiveTensorError) -> Self {
        Self::Model(TritModelError::Additive(error))
    }
}
impl From<TritPackageReadError> for TritModelReadError {
    fn from(error: TritPackageReadError) -> Self {
        Self::Source(error)
    }
}

/// Model/source errors are distinct from an unchanged visitor error.
#[derive(Debug)]
#[non_exhaustive]
pub enum TritModelVisitError<E> {
    /// Named binding or retained source failure.
    Read(TritModelReadError),
    /// Original visitor error; no terminal success is returned.
    Visitor(E),
}
impl<E: core::fmt::Display> core::fmt::Display for TritModelVisitError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Read(error) => error.fmt(f),
            Self::Visitor(error) => write!(f, "trit model visitor: {error}"),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for TritModelVisitError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read(error) => Some(error),
            Self::Visitor(error) => Some(error),
        }
    }
}
impl<E> From<TritModelError> for TritModelVisitError<E> {
    fn from(error: TritModelError) -> Self {
        Self::Read(error.into())
    }
}
impl<E> From<TritBlobVisitError<E>> for TritModelVisitError<E> {
    fn from(error: TritBlobVisitError<E>) -> Self {
        match error {
            TritBlobVisitError::Source(error) => Self::Read(error.into()),
            TritBlobVisitError::Visitor(error) => Self::Visitor(error),
        }
    }
}

/// Typed model owning its admitted seekable source, not a complete source Vec.
///
/// Both reader adapters use one typed binding/payload/digest/prefix validator.
/// Admission uses two64KiB source caches and bounded codec/scale windows, then
/// discards caches. Retained metadata/index/digests still require separate heap
/// and aggregate admission. This is not architecture execution, authenticated fit
/// provenance, profile freeze, model quality, physical RSS or release clearance.
#[derive(Debug)]
pub struct TritModelReader<R> {
    package: TritPackageReader<R>,
    binding: ModelBindings,
}
impl<R: Read + Seek> TritModelReader<R> {
    /// Admit an ordinary typed model and verify complete source identity again.
    ///
    /// Owns the supplied handle, not its pathname. External concurrent writers
    /// are not made atomic snapshots; immutable source custody remains required.
    /// Shards/entropy/mandatory joined profiles are not silently admitted.
    /// # Errors
    /// Original shared model refusal, explicit caps, mutation or actual source I/O.
    pub fn open(source: R, limits: TritPackageReadLimits) -> Result<Self, TritModelReadError> {
        let mut package = TritPackageReader::open(source, limits)?;
        let manifest = decode_model_manifest(package.manifest())?;
        let binding = validate_bindings(manifest, &mut CachedSource::new(&mut package)?)?;
        package.verify()?;
        Ok(Self { package, binding })
    }
    /// Immutable typed metadata; successful parsing is not model qualification.
    #[must_use]
    pub const fn manifest(&self) -> &ModelManifest {
        &self.binding.manifest
    }
    /// Initial admitted exact-byte package identity, not current source state.
    #[must_use]
    pub fn package_id(&self) -> crate::PackageId {
        self.package.package_id()
    }
    /// Candidate codec-independent level identity, not authenticated fit provenance.
    /// # Errors
    /// Absent level.
    pub fn model_id(&self, level: &str) -> Result<ModelId, TritModelError> {
        self.binding.model_id(level)
    }
    /// Candidate logical tensor digest admitted during complete typed validation.
    /// # Errors
    /// Absent level/tensor.
    pub fn tensor_digest(
        &self,
        level: &str,
        name: &str,
    ) -> Result<SemanticTensorDigest, TritModelError> {
        self.binding.tensor_digest(level, name)
    }
    /// Borrow immutable metadata for an admitted named tensor.
    /// # Errors
    /// Absent level/tensor.
    pub fn tensor_meta(&self, level: &str, name: &str) -> Result<&TensorMeta, TritModelError> {
        self.binding.tensor_meta(level, name)
    }
    /// Exact admitted directory byte length, without reading or decoding payloads.
    /// This is an encoded ledger, not process memory or current source verification.
    /// # Errors
    /// Absent level/tensor or original binding failure.
    pub fn tensor_byte_len(&self, level: &str, name: &str) -> Result<u64, TritModelError> {
        let meta = self.binding.tensor_meta(level, name)?;
        let index = self.binding.tensor_index(meta, self.package.blobs())?;
        Ok(self.package.blobs()[index].length)
    }
    /// Exact admitted asset directory byte length; does not authenticate current bytes.
    /// # Errors
    /// Absent asset or original binding failure.
    pub fn asset_byte_len(&self, name: &str) -> Result<u64, TritModelError> {
        let index = self.binding.asset_index(name, self.package.blobs())?;
        Ok(self.package.blobs()[index].length)
    }
    /// Require every physical directory entry to be referenced by typed metadata.
    /// # Errors
    /// Missing typed binding or any unreferenced physical entry.
    pub fn validate_complete_payload_inventory(&self) -> Result<(), TritModelError> {
        self.binding
            .validate_complete_payload_inventory(self.package.blobs())
    }
    /// Stream raw tensor bytes through bounded chunks and recheck that blob's ID.
    /// Visitor effects are nontransactional; stage privately until success and
    /// complete terminal verify. Does not allocate a decoded or payload master.
    /// # Errors
    /// Absent binding, source mutation/fault or original typed visitor error.
    pub fn visit_tensor<E>(
        &mut self,
        level: &str,
        name: &str,
        visitor: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<(), TritModelVisitError<E>> {
        let meta = self.binding.tensor_meta(level, name)?;
        let index = self.binding.tensor_index(meta, self.package.blobs())?;
        self.package.visit_blob(index, visitor).map_err(Into::into)
    }
    /// Stream a declared asset without extracting any filesystem path.
    /// Caller must stage nontransactional effects until success and terminal verify.
    /// # Errors
    /// Absent binding, changed source or original typed visitor failure.
    pub fn visit_asset<E>(
        &mut self,
        name: &str,
        visitor: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<(), TritModelVisitError<E>> {
        let index = self.binding.asset_index(name, self.package.blobs())?;
        self.package.visit_blob(index, visitor).map_err(Into::into)
    }
    /// Recheck the full initial PackageId and exact source length.
    /// Required before caller terminal publication of privately staged work.
    /// # Errors
    /// Source mutation/truncation/growth or actual seek/read faults.
    pub fn verify(&mut self) -> Result<(), TritModelReadError> {
        self.package.verify().map_err(Into::into)
    }
}

struct Cache {
    blob: Option<usize>,
    start: usize,
    length: usize,
    bytes: Vec<u8>,
}
impl Cache {
    fn new() -> Result<Self, TritModelReadError> {
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(RANGE_BYTES).map_err(|_| {
            TritPackageReadError::Format(TritPackageError::AllocationFailed {
                requested: RANGE_BYTES,
            })
        })?;
        bytes.resize(RANGE_BYTES, 0);
        Ok(Self {
            blob: None,
            start: 0,
            length: 0,
            bytes,
        })
    }
}
struct CachedSource<'a, R> {
    package: &'a mut TritPackageReader<R>,
    cache: [Cache; 2],
    replace: usize,
}
impl<'a, R> CachedSource<'a, R> {
    fn new(package: &'a mut TritPackageReader<R>) -> Result<Self, TritModelReadError> {
        Ok(Self {
            package,
            cache: [Cache::new()?, Cache::new()?],
            replace: 0,
        })
    }
}
impl<R: Read + Seek> PayloadSource for CachedSource<'_, R> {
    type Error = TritModelReadError;
    fn read_range(
        &mut self,
        blob: usize,
        offset: usize,
        out: &mut [u8],
    ) -> Result<(), Self::Error> {
        let info = self
            .package
            .blobs()
            .get(blob)
            .ok_or(TritPackageReadError::Format(
                TritPackageError::InvalidDirectory,
            ))?;
        let length = usize::try_from(info.length)
            .map_err(|_| TritPackageReadError::Format(TritPackageError::LengthOverflow))?;
        let end = offset
            .checked_add(out.len())
            .ok_or(TritPackageReadError::Format(
                TritPackageError::LengthOverflow,
            ))?;
        if end > length {
            return Err(TritPackageReadError::Format(TritPackageError::InvalidDirectory).into());
        }
        let mut copied = 0;
        while copied < out.len() {
            let at = offset + copied;
            let hit = self.cache.iter().position(|cache| {
                cache.blob == Some(blob) && at >= cache.start && at - cache.start < cache.length
            });
            let slot = if let Some(slot) = hit {
                slot
            } else {
                let slot = self
                    .cache
                    .iter()
                    .position(|cache| cache.blob == Some(blob))
                    .unwrap_or(self.replace);
                let cache = &mut self.cache[slot];
                let start = at / RANGE_BYTES * RANGE_BYTES;
                let count = (length - start).min(RANGE_BYTES);
                self.package
                    .read_blob_range(blob, start, &mut cache.bytes[..count])?;
                cache.blob = Some(blob);
                cache.start = start;
                cache.length = count;
                self.replace = (slot + 1) % self.cache.len();
                slot
            };
            let cache = &self.cache[slot];
            let start = at - cache.start;
            let count = (out.len() - copied).min(cache.length - start);
            out[copied..copied + count].copy_from_slice(&cache.bytes[start..start + count]);
            copied += count;
        }
        Ok(())
    }
}
impl<R: Read + Seek> ModelSource for CachedSource<'_, R> {
    fn blobs(&self) -> &[TritBlobInfo] {
        self.package.blobs()
    }
}
