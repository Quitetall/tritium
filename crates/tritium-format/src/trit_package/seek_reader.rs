//! Retained bounded range adapter for the shared package validator.

use core::{convert::Infallible, fmt, ops::Range};
use std::io::{self, Read, Seek, SeekFrom};

use super::{
    FOOTER_BYTES, PackageHasher, PackageId, PackageRanges, TRIT_HEADER_BYTES, TritBlobInfo,
    TritPackageError, as_usize, decode_directory, read_layout, validate_blob_ranges,
    validate_canonical_manifest, validate_metadata_digests,
};

const CHUNK_BYTES: usize = 64 * 1024;

/// Explicit receiving caps, checked before metadata buffers or payload scans.
/// Encoded metadata caps do not bound parsed heap, caller storage or process RSS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TritPackageReadLimits {
    /// Maximum complete source length, including framing and alignment.
    pub max_source_bytes: u64,
    /// Maximum encoded manifest bytes; the existing global cap also applies.
    pub max_manifest_bytes: usize,
    /// Maximum encoded directory bytes; the existing global cap also applies.
    pub max_directory_bytes: usize,
}

/// Integrity, explicit admission or actual seek/read failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum TritPackageReadError {
    /// The common package format validator rejected the source.
    Format(TritPackageError),
    /// Complete source length exceeds the caller's explicit admission.
    SourceLimitExceeded {
        /// Actual complete source length.
        requested: u64,
        /// Caller-admitted maximum length.
        limit: u64,
    },
    /// Source length or exact package identity changed during verification.
    SourceChanged,
    /// The retained source failed an actual seek/read operation.
    Io {
        /// Bounded operation name, not source contents.
        operation: &'static str,
        /// Original source error.
        source: io::Error,
    },
}

impl fmt::Display for TritPackageReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(error) => error.fmt(f),
            Self::SourceLimitExceeded { requested, limit } => {
                write!(f, "trit source needs {requested} bytes, limit is {limit}")
            }
            Self::SourceChanged => f.write_str("trit source changed during verification"),
            Self::Io { operation, source } => write!(f, "trit source {operation}: {source}"),
        }
    }
}

impl std::error::Error for TritPackageReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<TritPackageError> for TritPackageReadError {
    fn from(error: TritPackageError) -> Self {
        Self::Format(error)
    }
}

/// Package/source failures are distinct from the visitor's original typed error.
#[derive(Debug)]
#[non_exhaustive]
pub enum TritBlobVisitError<E> {
    /// Source admission, framing, digest, index, length or I/O failure.
    Source(TritPackageReadError),
    /// Unchanged visitor failure; no terminal success is returned.
    Visitor(E),
}

impl<E: fmt::Display> fmt::Display for TritBlobVisitError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(error) => error.fmt(f),
            Self::Visitor(error) => write!(f, "trit blob visitor: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for TritBlobVisitError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::Visitor(error) => Some(error),
        }
    }
}

impl<E> From<TritPackageReadError> for TritBlobVisitError<E> {
    fn from(error: TritPackageReadError) -> Self {
        Self::Source(error)
    }
}

/// Strict package reader owning a retained File, Cursor or other seekable source.
///
/// Retains encoded manifest and parsed directory, never whole payloads. The
/// existing borrowed reader and this adapter share all envelope/range rules.
/// Payload scans use fixed64KiB scratch. Parsed metadata, caller source storage,
/// allocator overhead and page cache remain separate from these encoded caps.
/// This is package integrity, not typed-model or execution admission.
#[derive(Debug)]
pub struct TritPackageReader<R> {
    source: R,
    manifest: Vec<u8>,
    blobs: Vec<TritBlobInfo>,
    length: usize,
    id: PackageId,
}

impl<R: Read + Seek> TritPackageReader<R> {
    /// Admit the complete package under explicit source/metadata ceilings.
    ///
    /// Hashes the entire source before interpretation and again before returning
    /// success, preserving the existing exact PackageId domain. Header/footer,
    /// metadata digests, every blob and all padding are checked. Ownership pins
    /// the supplied handle, not its pathname. It does not create an atomic
    /// snapshot against an external concurrent writer; source custody is the
    /// caller's responsibility. Current source cursor is never an input offset.
    ///
    /// # Errors
    /// Explicit caps, common format checks, mutation, or actual seek/read faults.
    pub fn open(
        mut source: R,
        limits: TritPackageReadLimits,
    ) -> Result<Self, TritPackageReadError> {
        let length = source_length(&mut source)?;
        if length > limits.max_source_bytes {
            return Err(TritPackageReadError::SourceLimitExceeded {
                requested: length,
                limit: limits.max_source_bytes,
            });
        }
        let length = as_usize(length)?;
        if length < TRIT_HEADER_BYTES + FOOTER_BYTES {
            return Err(TritPackageError::Truncated.into());
        }
        let mut header = [0; TRIT_HEADER_BYTES];
        let mut footer = [0; FOOTER_BYTES];
        read_at(&mut source, 0, &mut header)?;
        read_at(&mut source, length - FOOTER_BYTES, &mut footer)?;
        let layout = read_layout(
            length,
            &header,
            &footer,
            limits.max_manifest_bytes,
            limits.max_directory_bytes,
        )?;
        let id = package_identity(&mut source, length)?;
        let mut check_header = [0; TRIT_HEADER_BYTES];
        let mut check_footer = [0; FOOTER_BYTES];
        read_at(&mut source, 0, &mut check_header)?;
        read_at(&mut source, length - FOOTER_BYTES, &mut check_footer)?;
        if header != check_header || footer != check_footer {
            return Err(TritPackageReadError::SourceChanged);
        }
        let manifest = read_buffer(&mut source, layout.manifest.clone())?;
        let directory = read_buffer(&mut source, layout.directory)?;
        validate_metadata_digests(&manifest, &directory, &footer)?;
        validate_canonical_manifest(&manifest)?;
        let blobs = decode_directory(&directory)?;
        validate_blob_ranges(&mut SeekRanges(&mut source), layout.manifest.start, &blobs)?;
        if package_identity(&mut source, length)? != id {
            return Err(TritPackageReadError::SourceChanged);
        }
        Ok(Self {
            source,
            manifest,
            blobs,
            length,
            id,
        })
    }

    /// Canonical encoded manifest admitted against the complete source.
    #[must_use]
    pub fn manifest(&self) -> &[u8] {
        &self.manifest
    }

    /// Admitted immutable directory, in the same order as the borrowed reader.
    #[must_use]
    pub fn blobs(&self) -> &[TritBlobInfo] {
        &self.blobs
    }

    /// Initial admitted exact-byte identity, not current mutable source state.
    #[must_use]
    pub const fn package_id(&self) -> PackageId {
        self.id
    }

    /// Recheck the full initial PackageId and exact source length.
    /// Required by callers before terminal publication of privately staged work.
    ///
    /// # Errors
    /// Source mutation, truncation/growth, or actual seek/read faults.
    pub fn verify(&mut self) -> Result<(), TritPackageReadError> {
        if package_identity(&mut self.source, self.length)? != self.id {
            return Err(TritPackageReadError::SourceChanged);
        }
        Ok(())
    }

    /// Stream one admitted blob through at most64KiB chunks and recheck its ID.
    ///
    /// Only a directory index is accepted; a caller cannot forge source ranges.
    /// Visitor effects are nontransactional: stage privately until digest and
    /// terminal length checks succeed. Empty blobs invoke no callbacks. This
    /// does not rehash unrelated package bytes on every tensor; `verify` is the
    /// complete-source terminal check. Chunks cannot be retained by the visitor.
    ///
    /// # Errors
    /// Invalid index, changed blob/length, actual I/O, or the unchanged visitor error.
    pub fn visit_blob<E>(
        &mut self,
        index: usize,
        mut visitor: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<(), TritBlobVisitError<E>> {
        let blob = self.blobs.get(index).ok_or(TritPackageReadError::Format(
            TritPackageError::InvalidDirectory,
        ))?;
        let start = as_usize(blob.offset).map_err(TritPackageReadError::from)?;
        let end = start
            .checked_add(as_usize(blob.length).map_err(TritPackageReadError::from)?)
            .ok_or(TritPackageReadError::Format(
                TritPackageError::LengthOverflow,
            ))?;
        check_length(&mut self.source, self.length)?;
        let mut digest = blake3::Hasher::new();
        visit_range(&mut self.source, start..end, |chunk| {
            digest.update(chunk);
            visitor(chunk)
        })?;
        check_length(&mut self.source, self.length)?;
        if digest.finalize().as_bytes() != blob.id.as_bytes() {
            return Err(
                TritPackageReadError::Format(TritPackageError::BlobDigestMismatch(blob.id)).into(),
            );
        }
        Ok(())
    }

    // Only typed validators in this crate can read an admitted blob subrange.
    // It does not create a new authenticated identity for a subrange. Enclosing
    // typed admission must verify the complete retained package before success.
    pub(crate) fn read_blob_range(
        &mut self,
        index: usize,
        offset: usize,
        bytes: &mut [u8],
    ) -> Result<(), TritPackageReadError> {
        let blob = self
            .blobs
            .get(index)
            .ok_or(TritPackageError::InvalidDirectory)?;
        let end = offset
            .checked_add(bytes.len())
            .ok_or(TritPackageError::LengthOverflow)?;
        if end > as_usize(blob.length)? {
            return Err(TritPackageError::InvalidDirectory.into());
        }
        let start = as_usize(blob.offset)?
            .checked_add(offset)
            .ok_or(TritPackageError::LengthOverflow)?;
        read_at(&mut self.source, start, bytes)
    }
}

fn source_io(operation: &'static str, source: io::Error) -> TritPackageReadError {
    TritPackageReadError::Io { operation, source }
}

fn source_length(source: &mut impl Seek) -> Result<u64, TritPackageReadError> {
    source
        .seek(SeekFrom::End(0))
        .map_err(|error| source_io("length", error))
}

fn check_length(source: &mut impl Seek, expected: usize) -> Result<(), TritPackageReadError> {
    if source_length(source)? != expected as u64 {
        return Err(TritPackageReadError::SourceChanged);
    }
    Ok(())
}

fn seek_to(source: &mut impl Seek, position: usize) -> Result<(), TritPackageReadError> {
    let actual = source
        .seek(SeekFrom::Start(position as u64))
        .map_err(|error| source_io("seek", error))?;
    if actual != position as u64 {
        return Err(source_io(
            "seek position",
            io::Error::new(
                io::ErrorKind::InvalidData,
                "source returned a different seek position",
            ),
        ));
    }
    Ok(())
}

fn read_at(
    source: &mut (impl Read + Seek),
    offset: usize,
    bytes: &mut [u8],
) -> Result<(), TritPackageReadError> {
    seek_to(source, offset)?;
    for chunk in bytes.chunks_mut(CHUNK_BYTES) {
        source
            .read_exact(chunk)
            .map_err(|error| source_io("read", error))?;
    }
    Ok(())
}

fn read_buffer(
    source: &mut (impl Read + Seek),
    range: Range<usize>,
) -> Result<Vec<u8>, TritPackageReadError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(range.len())
        .map_err(|_| TritPackageError::AllocationFailed {
            requested: range.len(),
        })?;
    bytes.resize(range.len(), 0);
    read_at(source, range.start, &mut bytes)?;
    Ok(bytes)
}

fn visit_range<E>(
    source: &mut (impl Read + Seek),
    range: Range<usize>,
    mut visit: impl FnMut(&[u8]) -> Result<(), E>,
) -> Result<(), TritBlobVisitError<E>> {
    seek_to(source, range.start)?;
    let mut remaining = range.len();
    let mut scratch = [0; CHUNK_BYTES];
    while remaining != 0 {
        let count = remaining.min(scratch.len());
        source
            .read_exact(&mut scratch[..count])
            .map_err(|error| source_io("read chunk", error))?;
        visit(&scratch[..count]).map_err(TritBlobVisitError::Visitor)?;
        remaining -= count;
    }
    Ok(())
}

fn read_error(error: TritBlobVisitError<Infallible>) -> TritPackageReadError {
    match error {
        TritBlobVisitError::Source(error) => error,
        TritBlobVisitError::Visitor(never) => match never {},
    }
}

fn package_identity(
    source: &mut (impl Read + Seek),
    length: usize,
) -> Result<PackageId, TritPackageReadError> {
    check_length(source, length)?;
    let mut hash = PackageHasher::new();
    visit_range(source, 0..length, |chunk| {
        hash.update(chunk);
        Ok::<_, Infallible>(())
    })
    .map_err(read_error)?;
    check_length(source, length)?;
    Ok(hash.finalize())
}

struct SeekRanges<'a, R>(&'a mut R);

impl<R: Read + Seek> PackageRanges for SeekRanges<'_, R> {
    type Error = TritPackageReadError;

    fn digest(&mut self, range: Range<usize>) -> Result<[u8; 32], Self::Error> {
        let mut digest = blake3::Hasher::new();
        visit_range(self.0, range, |chunk| {
            digest.update(chunk);
            Ok::<_, Infallible>(())
        })
        .map_err(read_error)?;
        Ok(*digest.finalize().as_bytes())
    }

    fn is_zero(&mut self, range: Range<usize>) -> Result<bool, Self::Error> {
        let mut zero = true;
        visit_range(self.0, range, |chunk| {
            zero &= chunk.iter().all(|byte| *byte == 0);
            Ok::<_, Infallible>(())
        })
        .map_err(read_error)?;
        Ok(zero)
    }
}
