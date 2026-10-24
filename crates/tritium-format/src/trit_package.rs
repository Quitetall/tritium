//! Reference codec for the versioned `.trit` package envelope.

use core::fmt;
use std::collections::BTreeMap;

use ciborium::value::{CanonicalValue, Value};
use tritium_core::DType;
use tritium_schema::BlobId;

use crate::artifact::{PackageHasher, PackageId};

mod cbor;
#[cfg(test)]
mod directory_tests;
mod seek_reader;
mod sizing;
pub use sizing::{
    EncodedSizeBounds, TritBlobSize, TritPackageSizeBounds, trit_package_size_bounds,
};

pub use seek_reader::{
    TritBlobVisitError, TritPackageReadError, TritPackageReadLimits, TritPackageReader,
};

/// `.trit` file magic.
pub const TRIT_MAGIC: [u8; 8] = *b"TRITIUM\0";
const TRIT_FOOTER_MAGIC: [u8; 8] = *b"TRITEND\0";
/// Fixed `.trit` header size from ADR 0044.
pub const TRIT_HEADER_BYTES: usize = 64;
const FOOTER_BYTES: usize = 72;
/// Required blob alignment from ADR 0044.
pub const TRIT_BLOB_ALIGNMENT: usize = 256;
/// Current `.trit` major version.
pub const TRIT_PACKAGE_MAJOR: u16 = 1;
const TRIT_PACKAGE_MINOR: u16 = 0;
const DIRECTORY_VERSION: u64 = 1;
const MAX_BLOBS: usize = 1_000_000;
const MAX_MANIFEST_BYTES: usize = 256 * 1024 * 1024;
const MAX_DIRECTORY_BYTES: usize = 256 * 1024 * 1024;

/// Semantic kind associated with one content-addressed blob.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TritBlobKind {
    /// Additive ternary tensor payload.
    AdditiveTensor,
    /// Complete residual-bearing QTEA V1 tensor, never additive SALT.
    QteaTensor,
    /// Dense tensor payload with its element dtype.
    DenseTensor(DType),
    /// Model asset such as config, tokenizer, or chat template.
    Asset(String),
}

/// A borrowed payload supplied to the `.trit` package writer.
#[derive(Clone, Debug)]
pub struct TritBlob<'a> {
    /// Semantic kind recorded in the package directory.
    pub kind: TritBlobKind,
    /// Exact stored blob bytes.
    pub bytes: &'a [u8],
}

/// A content-addressed blob range in a caller-owned seekable staging spool.
/// The writer checks the declared identity against actual bytes; metadata alone
/// is not trusted. Offsets are relative to the start of the source, not its
/// current cursor. Bytes outside referenced ranges are not package inputs.
#[derive(Clone, Debug)]
pub struct TritSpoolBlob {
    /// Semantic kind recorded in the package directory.
    pub kind: TritBlobKind,
    /// Expected BLAKE3 identity of the exact range.
    pub id: BlobId,
    /// Source byte offset.
    pub offset: u64,
    /// Exact source byte length.
    pub length: u64,
}

/// Exact canonical layout from known blob identities/lengths, not source-byte proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TritSpoolLayoutSize {
    /// Complete canonical container length including alignment and deduplication.
    pub package_bytes: u64,
    /// Exact supplied canonical manifest bytes.
    pub manifest_bytes: u64,
    /// Exact owning encoder's deduplicated typed directory bytes.
    pub directory_bytes: u64,
}

/// Inspect the owning spool encoder's exact metadata layout without source/output I/O.
/// Actual payload identities/ranges/collisions remain checked by the writer.
pub fn inspect_trit_spool_layout(
    manifest_cbor: &[u8],
    blobs: &[TritSpoolBlob],
) -> Result<TritSpoolLayoutSize, TritPackageError> {
    let plan = prepare_spool_layout(manifest_cbor, blobs, usize::MAX)?;
    Ok(TritSpoolLayoutSize {
        package_bytes: plan.size as u64,
        manifest_bytes: plan.manifest.len() as u64,
        directory_bytes: plan.directory.len() as u64,
    })
}

/// Indexed package entry for one blob kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TritBlobInfo {
    /// BLAKE3 content identity of the exact blob bytes.
    pub id: BlobId,
    /// Semantic kind recorded in the directory.
    pub kind: TritBlobKind,
    /// Byte offset from the start of the package.
    pub offset: u64,
    /// Exact stored byte length.
    pub length: u64,
}

/// A validated, borrowed `.trit` package view.
#[derive(Debug)]
pub struct TritPackage<'a> {
    bytes: &'a [u8],
    manifest: &'a [u8],
    blobs: Vec<TritBlobInfo>,
}

impl<'a> TritPackage<'a> {
    // Only original immutable directory entries reach this index-based seam.
    // read_trit_package already checked ranges and hashes; do not rehash every
    // shared tensor reference in every level.
    pub(crate) fn indexed_blob(&self, index: usize) -> Result<&'a [u8], TritPackageError> {
        let info = self
            .blobs
            .get(index)
            .ok_or(TritPackageError::InvalidDirectory)?;
        let start = as_usize(info.offset)?;
        let end = start
            .checked_add(as_usize(info.length)?)
            .ok_or(TritPackageError::LengthOverflow)?;
        self.bytes
            .get(start..end)
            .ok_or(TritPackageError::InvalidDirectory)
    }
    /// Borrow the canonical CBOR manifest bytes.
    #[must_use]
    pub const fn manifest(&self) -> &'a [u8] {
        self.manifest
    }

    /// Borrow indexed blob records in canonical directory order.
    #[must_use]
    pub fn blobs(&self) -> &[TritBlobInfo] {
        &self.blobs
    }

    /// Return the exact-byte package identity.
    #[must_use]
    pub fn package_id(&self) -> PackageId {
        PackageId::from_package_bytes(self.bytes)
    }

    /// Borrow and re-verify one blob by its directory entry.
    pub fn blob(&self, info: &TritBlobInfo) -> Result<&'a [u8], TritPackageError> {
        let start = usize::try_from(info.offset).map_err(|_| TritPackageError::LengthOverflow)?;
        let len = usize::try_from(info.length).map_err(|_| TritPackageError::LengthOverflow)?;
        let end = start
            .checked_add(len)
            .ok_or(TritPackageError::LengthOverflow)?;
        let bytes = self
            .bytes
            .get(start..end)
            .ok_or(TritPackageError::InvalidDirectory)?;
        if blake3::hash(bytes).as_bytes() != info.id.as_bytes() {
            return Err(TritPackageError::BlobDigestMismatch(info.id));
        }
        Ok(bytes)
    }
}

/// Failure writing or validating a `.trit` package.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TritPackageError {
    /// Input is shorter than the minimum package envelope.
    Truncated,
    /// Package header or footer magic is wrong.
    BadMagic,
    /// The package major/minor version is unsupported.
    UnsupportedVersion {
        /// Major version read from the header.
        major: u16,
        /// Minor version read from the header.
        minor: u16,
    },
    /// Required-feature bits include a feature unknown to this implementation.
    UnknownRequiredFeatures(u32),
    /// Reserved header bytes are not zero.
    NonZeroReserved,
    /// A metadata or blob length cannot be represented safely.
    LengthOverflow,
    /// Manifest or directory exceeds its explicit parser bound.
    MetadataLimitExceeded,
    /// Complete output exceeds the caller's bound before output allocation.
    PackageLimitExceeded {
        /// Computed complete package size, including alignment and metadata.
        requested: usize,
        /// Caller-admitted maximum bytes.
        limit: usize,
    },
    /// Manifest is not deterministic, definite-length CBOR in the supported profile.
    NonCanonicalManifest,
    /// Directory bytes are malformed or not in canonical form.
    InvalidDirectory,
    /// A declared metadata or blob digest does not match its bytes.
    DigestMismatch,
    /// A blob digest disagrees with the bytes at its declared offset.
    BlobDigestMismatch(BlobId),
    /// Two distinct payloads have the same content identity (cryptographic collision).
    BlobIdentityCollision,
    /// A fallible output allocation failed.
    AllocationFailed {
        /// Exact byte or element count requested by the failed reservation.
        requested: usize,
    },
    /// An asset name is empty.
    EmptyAssetName,
}

impl fmt::Display for TritPackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("trit package is truncated"),
            Self::BadMagic => f.write_str("trit package header or footer magic is invalid"),
            Self::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported trit package version {major}.{minor}")
            }
            Self::UnknownRequiredFeatures(bits) => {
                write!(f, "unknown required trit package feature bits {bits:#010x}")
            }
            Self::NonZeroReserved => f.write_str("trit package reserved bytes are nonzero"),
            Self::LengthOverflow => f.write_str("trit package length arithmetic overflowed"),
            Self::MetadataLimitExceeded => f.write_str("trit package metadata exceeds its limit"),
            Self::PackageLimitExceeded { requested, limit } => {
                write!(f, "trit package needs {requested} bytes, limit is {limit}")
            }
            Self::NonCanonicalManifest => {
                f.write_str("trit package manifest is not canonical CBOR")
            }
            Self::InvalidDirectory => f.write_str("trit package directory is invalid"),
            Self::DigestMismatch => f.write_str("trit package metadata digest mismatch"),
            Self::BlobDigestMismatch(id) => write!(f, "trit blob {id:?} digest mismatch"),
            Self::BlobIdentityCollision => {
                f.write_str("distinct trit blob payloads have the same content identity")
            }
            Self::AllocationFailed { requested } => {
                write!(f, "trit package allocation of {requested} bytes failed")
            }
            Self::EmptyAssetName => f.write_str("trit asset name must not be empty"),
        }
    }
}

impl std::error::Error for TritPackageError {}

/// Build a deterministic `.trit` package in memory.
///
/// Blob records are deduplicated by BLAKE3 content identity and aligned to 256
/// bytes. The manifest and footer-located directory use deterministic CBOR;
/// the directory and manifest bytes are independently hashed in the footer.
/// Manifest maps use canonical key order; floating-point values are not admitted
/// in this manifest profile (store tensor scales in typed blobs instead).
pub fn write_trit_package(
    manifest_cbor: &[u8],
    blobs: &[TritBlob<'_>],
) -> Result<Vec<u8>, TritPackageError> {
    write_trit_package_bounded(manifest_cbor, blobs, usize::MAX)
}

/// Build the same deterministic package with a complete output byte bound.
/// Metadata/index allocations retain their existing independent limits.
/// # Errors
/// Invalid input, overflow, allocation failure, or complete size above `max_bytes`.
pub fn write_trit_package_bounded(
    manifest_cbor: &[u8],
    blobs: &[TritBlob<'_>],
    max_bytes: usize,
) -> Result<Vec<u8>, TritPackageError> {
    let plan = prepare_write(manifest_cbor, blobs, max_bytes)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(plan.size)
        .map_err(|_| TritPackageError::AllocationFailed {
            requested: plan.size,
        })?;
    let result: Result<(), core::convert::Infallible> = plan.emit(
        &mut (),
        |bytes| {
            output.extend_from_slice(bytes);
            Ok(())
        },
        |index, _, chunk| chunk(blobs[index].bytes),
        |_| Ok(()),
    );
    match result {
        Ok(()) => Ok(output),
        Err(never) => match never {},
    }
}

/// Failure admitting or emitting a package to a caller-owned writer.
#[derive(Debug)]
#[non_exhaustive]
pub enum TritPackageWriteError {
    /// Invalid package input or bound; spool mutation may be detected after writes.
    Package(TritPackageError),
    /// Destination failed after zero or more bytes; no success identity is returned.
    Io(std::io::Error),
    /// Seekable staging source failed; pending destination bytes may remain.
    Source(std::io::Error),
}
impl fmt::Display for TritPackageWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Package(e) => e.fmt(f),
            Self::Io(e) => write!(f, "trit package destination failed: {e}"),
            Self::Source(e) => write!(f, "trit package staging source failed: {e}"),
        }
    }
}
impl std::error::Error for TritPackageWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Package(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::Source(e) => Some(e),
        }
    }
}

/// Emit the identical admitted package without retaining a complete output buffer.
///
/// Uses the same layout, canonical directory and chunk emitter as the Vec adapter,
/// and the existing incremental package hasher. All input/complete-size checks
/// finish before the first destination write. Borrowed inputs and metadata are
/// retained; this is not full RSS admission or an out-of-core source reader.
/// Writes at the current destination position, without seeking, flushing, syncing
/// or publishing. The caller must use an owned pending destination and handle
/// partial I/O failure before its no-clobber commit point.
/// # Errors
/// Package admission leaves the destination untouched. I/O failure may leave
/// partial output and never returns a successful PackageId.
pub fn write_trit_package_to(
    manifest_cbor: &[u8],
    blobs: &[TritBlob<'_>],
    max_bytes: usize,
    destination: &mut impl std::io::Write,
) -> Result<PackageId, TritPackageWriteError> {
    let plan =
        prepare_write(manifest_cbor, blobs, max_bytes).map_err(TritPackageWriteError::Package)?;
    let mut hash = PackageHasher::new();
    plan.emit(
        &mut (),
        |bytes| {
            destination.write_all(bytes)?;
            hash.update(bytes);
            Ok(())
        },
        |index, _, chunk| chunk(blobs[index].bytes),
        |_| Ok(()),
    )
    .map_err(TritPackageWriteError::Io)?;
    Ok(hash.finalize())
}

const SPOOL_CHUNK_BYTES: usize = 64 * 1024;

/// Write the identical canonical package from bounded seekable spool ranges.
///
/// Complete size/metadata/range admission and every source digest and duplicate
/// byte comparison finish before destination writes. Unique payloads are hashed
/// again while emitting, and all ranges are re-verified before the footer. Source
/// payload reads use at most two fixed 64 KiB buffers, independent of model size;
/// directory/manifest metadata is still retained under its existing limits.
///
/// The caller owns spool immutability, aggregate memory/disk admission and an
/// unpublished destination. This does not create an atomic source snapshot,
/// seek/flush/sync the destination or publish it. Source cursor position changes.
/// # Errors
/// Admission errors leave the destination untouched. Later source mutation or
/// source/destination I/O may leave partial bytes, never a success PackageId.
pub fn write_trit_package_from_spool(
    manifest_cbor: &[u8],
    blobs: &[TritSpoolBlob],
    source: &mut (impl std::io::Read + std::io::Seek),
    max_bytes: usize,
    destination: &mut impl std::io::Write,
) -> Result<PackageId, TritPackageWriteError> {
    let plan = prepare_spool_layout(manifest_cbor, blobs, max_bytes)
        .map_err(TritPackageWriteError::Package)?;
    let source_len = source
        .seek(std::io::SeekFrom::End(0))
        .map_err(TritPackageWriteError::Source)?;
    for blob in blobs {
        let end = blob
            .offset
            .checked_add(blob.length)
            .ok_or(TritPackageWriteError::Package(
                TritPackageError::LengthOverflow,
            ))?;
        if end > source_len {
            return Err(TritPackageWriteError::Package(TritPackageError::Truncated));
        }
    }
    // Allocate no blob-size buffer. Both buffers also serve collision comparisons.
    let mut state = (source, ([0u8; SPOOL_CHUNK_BYTES], [0u8; SPOOL_CHUNK_BYTES]));
    verify_spool(state.0, blobs, &mut state.1)?;
    let mut hash = PackageHasher::new();
    plan.emit(
        &mut state,
        |bytes| {
            destination
                .write_all(bytes)
                .map_err(TritPackageWriteError::Io)?;
            hash.update(bytes);
            Ok(())
        },
        |index, state, chunk| visit_spool_blob(state.0, &blobs[index], &mut state.1.0, chunk),
        |state| {
            // Include non-emitted aliases. A mutation there must not be hidden by dedup.
            verify_spool(state.0, blobs, &mut state.1)
        },
    )?;
    Ok(hash.finalize())
}

fn prepare_spool_layout<'a>(
    manifest_cbor: &'a [u8],
    blobs: &[TritSpoolBlob],
    max_bytes: usize,
) -> Result<PackageWritePlan<'a>, TritPackageError> {
    let entries = blob_metadata(manifest_cbor, blobs.len(), |index| {
        let blob = &blobs[index];
        Ok(BlobMetadata {
            id: blob.id,
            kind: blob.kind.clone(),
            length: as_usize(blob.length)?,
        })
    })?;
    prepare_layout(manifest_cbor, &entries, max_bytes)
}

fn seek_spool(source: &mut impl std::io::Seek, offset: u64) -> Result<(), TritPackageWriteError> {
    let actual = source
        .seek(std::io::SeekFrom::Start(offset))
        .map_err(TritPackageWriteError::Source)?;
    if actual != offset {
        return Err(TritPackageWriteError::Package(
            TritPackageError::InvalidDirectory,
        ));
    }
    Ok(())
}

fn visit_spool_blob(
    source: &mut (impl std::io::Read + std::io::Seek),
    blob: &TritSpoolBlob,
    scratch: &mut [u8; SPOOL_CHUNK_BYTES],
    mut chunk: impl FnMut(&[u8]) -> Result<(), TritPackageWriteError>,
) -> Result<(), TritPackageWriteError> {
    seek_spool(source, blob.offset)?;
    let mut remaining = blob.length;
    let mut hash = blake3::Hasher::new();
    while remaining != 0 {
        let count = remaining.min(SPOOL_CHUNK_BYTES as u64) as usize;
        let bytes = &mut scratch[..count];
        source
            .read_exact(bytes)
            .map_err(TritPackageWriteError::Source)?;
        hash.update(bytes);
        chunk(bytes)?;
        remaining -= count as u64;
    }
    if hash.finalize().as_bytes() != blob.id.as_bytes() {
        return Err(TritPackageWriteError::Package(
            TritPackageError::BlobDigestMismatch(blob.id),
        ));
    }
    Ok(())
}

fn verify_spool(
    source: &mut (impl std::io::Read + std::io::Seek),
    blobs: &[TritSpoolBlob],
    scratch: &mut ([u8; SPOOL_CHUNK_BYTES], [u8; SPOOL_CHUNK_BYTES]),
) -> Result<(), TritPackageWriteError> {
    let mut unique = BTreeMap::<[u8; 32], usize>::new();
    for (index, blob) in blobs.iter().enumerate() {
        visit_spool_blob(source, blob, &mut scratch.0, |_| Ok(()))?;
        if let Some(&other) = unique.get(blob.id.as_bytes()) {
            let other = &blobs[other];
            if other.length != blob.length {
                return Err(TritPackageWriteError::Package(
                    TritPackageError::BlobIdentityCollision,
                ));
            }
            if other.offset == blob.offset {
                continue;
            }
            let mut position = 0;
            while position < blob.length {
                let count = (blob.length - position).min(SPOOL_CHUNK_BYTES as u64) as usize;
                seek_spool(source, other.offset + position)?;
                source
                    .read_exact(&mut scratch.0[..count])
                    .map_err(TritPackageWriteError::Source)?;
                seek_spool(source, blob.offset + position)?;
                source
                    .read_exact(&mut scratch.1[..count])
                    .map_err(TritPackageWriteError::Source)?;
                if scratch.0[..count] != scratch.1[..count] {
                    return Err(TritPackageWriteError::Package(
                        TritPackageError::BlobIdentityCollision,
                    ));
                }
                position += count as u64;
            }
        } else {
            unique.insert(*blob.id.as_bytes(), index);
        }
    }
    Ok(())
}

struct BlobMetadata {
    id: BlobId,
    kind: TritBlobKind,
    length: usize,
}

struct PayloadPlan {
    padding: usize,
    index: usize,
}

struct PackageWritePlan<'a> {
    header: [u8; TRIT_HEADER_BYTES],
    payloads: Vec<PayloadPlan>,
    manifest: &'a [u8],
    directory: Vec<u8>,
    footer: [u8; FOOTER_BYTES],
    size: usize,
}
impl PackageWritePlan<'_> {
    fn emit<S, E, C: FnMut(&[u8]) -> Result<(), E>>(
        &self,
        source: &mut S,
        mut chunk: C,
        mut payload: impl FnMut(usize, &mut S, &mut C) -> Result<(), E>,
        finish: impl FnOnce(&mut S) -> Result<(), E>,
    ) -> Result<(), E> {
        chunk(&self.header)?;
        let padding = [0u8; TRIT_BLOB_ALIGNMENT];
        for entry in &self.payloads {
            chunk(&padding[..entry.padding])?;
            payload(entry.index, source, &mut chunk)?;
        }
        chunk(self.manifest)?;
        chunk(&self.directory)?;
        finish(source)?;
        chunk(&self.footer)
    }
}

fn prepare_write<'a>(
    manifest_cbor: &'a [u8],
    blobs: &[TritBlob<'a>],
    max_bytes: usize,
) -> Result<PackageWritePlan<'a>, TritPackageError> {
    let entries = blob_metadata(manifest_cbor, blobs.len(), |index| {
        let blob = &blobs[index];
        Ok(BlobMetadata {
            id: BlobId::from_bytes(*blake3::hash(blob.bytes).as_bytes()),
            kind: blob.kind.clone(),
            length: blob.bytes.len(),
        })
    })?;
    let mut unique = BTreeMap::<[u8; 32], &[u8]>::new();
    for (entry, blob) in entries.iter().zip(blobs) {
        if let Some(existing) = unique.insert(*entry.id.as_bytes(), blob.bytes)
            && existing != blob.bytes
        {
            return Err(TritPackageError::BlobIdentityCollision);
        }
    }
    prepare_layout(manifest_cbor, &entries, max_bytes)
}

fn blob_metadata(
    manifest_cbor: &[u8],
    count: usize,
    mut entry: impl FnMut(usize) -> Result<BlobMetadata, TritPackageError>,
) -> Result<Vec<BlobMetadata>, TritPackageError> {
    if manifest_cbor.len() > MAX_MANIFEST_BYTES || count > MAX_BLOBS {
        return Err(TritPackageError::MetadataLimitExceeded);
    }
    validate_canonical_manifest(manifest_cbor)?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(count)
        .map_err(|_| TritPackageError::AllocationFailed { requested: count })?;
    for index in 0..count {
        let blob = entry(index)?;
        if let TritBlobKind::Asset(name) = &blob.kind
            && name.is_empty()
        {
            return Err(TritPackageError::EmptyAssetName);
        }
        entries.push(blob);
    }
    Ok(entries)
}

fn prepare_layout<'a>(
    manifest_cbor: &'a [u8],
    blobs: &[BlobMetadata],
    max_bytes: usize,
) -> Result<PackageWritePlan<'a>, TritPackageError> {
    let mut unique = BTreeMap::<[u8; 32], (usize, usize)>::new();
    for (index, blob) in blobs.iter().enumerate() {
        let digest = *blob.id.as_bytes();
        if let Some((_, length)) = unique.get(&digest) {
            if *length != blob.length {
                return Err(TritPackageError::BlobIdentityCollision);
            }
        } else {
            unique.insert(digest, (index, blob.length));
        }
    }

    let mut offsets = BTreeMap::<[u8; 32], u64>::new();
    let mut cursor = TRIT_HEADER_BYTES;
    for (digest, (_, length)) in &unique {
        cursor = align_up(cursor)?;
        offsets.insert(
            *digest,
            u64::try_from(cursor).map_err(|_| TritPackageError::LengthOverflow)?,
        );
        cursor = cursor
            .checked_add(*length)
            .ok_or(TritPackageError::LengthOverflow)?;
    }
    let manifest_offset = cursor;
    cursor = cursor
        .checked_add(manifest_cbor.len())
        .ok_or(TritPackageError::LengthOverflow)?;
    let directory_offset = cursor;
    let mut directory_entries = Vec::new();
    directory_entries
        .try_reserve_exact(blobs.len())
        .map_err(|_| TritPackageError::AllocationFailed {
            requested: blobs.len(),
        })?;
    for blob in blobs {
        let id = *blob.id.as_bytes();
        directory_entries.push(TritBlobInfo {
            id: BlobId::from_bytes(id),
            kind: blob.kind.clone(),
            offset: offsets[&id],
            length: u64::try_from(blob.length).map_err(|_| TritPackageError::LengthOverflow)?,
        });
    }
    directory_entries.sort_by(|left, right| {
        left.id
            .as_bytes()
            .cmp(right.id.as_bytes())
            .then_with(|| kind_sort_key(&left.kind).cmp(&kind_sort_key(&right.kind)))
    });
    directory_entries.dedup_by(|right, left| right.id == left.id && right.kind == left.kind);
    let directory = encode_directory(&directory_entries)?;
    if directory.len() > MAX_DIRECTORY_BYTES {
        return Err(TritPackageError::MetadataLimitExceeded);
    }
    cursor = cursor
        .checked_add(directory.len())
        .and_then(|value| value.checked_add(FOOTER_BYTES))
        .ok_or(TritPackageError::LengthOverflow)?;

    if cursor > max_bytes {
        return Err(TritPackageError::PackageLimitExceeded {
            requested: cursor,
            limit: max_bytes,
        });
    }
    let mut payloads = Vec::new();
    payloads
        .try_reserve_exact(unique.len())
        .map_err(|_| TritPackageError::AllocationFailed {
            requested: unique.len(),
        })?;
    let mut position = TRIT_HEADER_BYTES;
    for (digest, (index, length)) in unique {
        let expected_offset =
            usize::try_from(offsets[&digest]).map_err(|_| TritPackageError::LengthOverflow)?;
        let pad = expected_offset
            .checked_sub(position)
            .ok_or(TritPackageError::InvalidDirectory)?;
        if pad >= TRIT_BLOB_ALIGNMENT {
            return Err(TritPackageError::InvalidDirectory);
        }
        payloads.push(PayloadPlan {
            padding: pad,
            index,
        });
        position = expected_offset
            .checked_add(length)
            .ok_or(TritPackageError::LengthOverflow)?;
    }
    if position != manifest_offset {
        return Err(TritPackageError::InvalidDirectory);
    }
    position = position
        .checked_add(manifest_cbor.len())
        .ok_or(TritPackageError::LengthOverflow)?;
    if position != directory_offset {
        return Err(TritPackageError::InvalidDirectory);
    }
    let mut header = [0u8; TRIT_HEADER_BYTES];
    header[..8].copy_from_slice(&TRIT_MAGIC);
    header[8..10].copy_from_slice(&TRIT_PACKAGE_MAJOR.to_le_bytes());
    header[10..12].copy_from_slice(&TRIT_PACKAGE_MINOR.to_le_bytes());
    write_u64(&mut header, 16, manifest_offset)?;
    write_u64(&mut header, 24, manifest_cbor.len())?;
    write_u64(&mut header, 32, directory_offset)?;
    write_u64(&mut header, 40, directory.len())?;
    let mut footer = [0u8; FOOTER_BYTES];
    footer[..8].copy_from_slice(&TRIT_FOOTER_MAGIC);
    footer[8..40].copy_from_slice(blake3::hash(manifest_cbor).as_bytes());
    footer[40..].copy_from_slice(blake3::hash(&directory).as_bytes());
    Ok(PackageWritePlan {
        header,
        payloads,
        manifest: manifest_cbor,
        directory,
        footer,
        size: cursor,
    })
}

/// Parse and fully validate a `.trit` package held in memory.
pub fn read_trit_package(bytes: &[u8]) -> Result<TritPackage<'_>, TritPackageError> {
    read_trit_package_with_manifest_limit(bytes, MAX_MANIFEST_BYTES)
}

/// Parse through the common validator with a receiving manifest byte ceiling.
/// Rejects the header's declared manifest size before canonical CBOR allocation.
/// The directory retains its global ceiling; this is not a full RSS bound.
/// # Errors
/// The same format/integrity errors as the default reader, or exceeded metadata caps.
pub fn read_trit_package_with_manifest_limit(
    bytes: &[u8],
    max_manifest_bytes: usize,
) -> Result<TritPackage<'_>, TritPackageError> {
    if bytes.len() < TRIT_HEADER_BYTES + FOOTER_BYTES {
        return Err(TritPackageError::Truncated);
    }
    let footer = &bytes[bytes.len() - FOOTER_BYTES..];
    let layout = read_layout(
        bytes.len(),
        &bytes[..TRIT_HEADER_BYTES],
        footer,
        max_manifest_bytes,
        MAX_DIRECTORY_BYTES,
    )?;
    let manifest = bytes
        .get(layout.manifest.clone())
        .ok_or(TritPackageError::Truncated)?;
    let directory = bytes
        .get(layout.directory.clone())
        .ok_or(TritPackageError::Truncated)?;
    validate_metadata_digests(manifest, directory, footer)?;
    validate_canonical_manifest(manifest)?;
    let blobs = decode_directory(directory)?;
    let mut ranges = bytes;
    validate_blob_ranges(&mut ranges, layout.manifest.start, &blobs)?;
    Ok(TritPackage {
        bytes,
        manifest,
        blobs,
    })
}

struct PackageReadLayout {
    manifest: core::ops::Range<usize>,
    directory: core::ops::Range<usize>,
}

fn read_layout(
    length: usize,
    header: &[u8],
    footer: &[u8],
    max_manifest_bytes: usize,
    max_directory_bytes: usize,
) -> Result<PackageReadLayout, TritPackageError> {
    if length < TRIT_HEADER_BYTES + FOOTER_BYTES
        || header.len() != TRIT_HEADER_BYTES
        || footer.len() != FOOTER_BYTES
    {
        return Err(TritPackageError::Truncated);
    }
    if header[..8] != TRIT_MAGIC || footer[..8] != TRIT_FOOTER_MAGIC {
        return Err(TritPackageError::BadMagic);
    }
    let major = read_u16(header, 8)?;
    let minor = read_u16(header, 10)?;
    if major != TRIT_PACKAGE_MAJOR || minor != TRIT_PACKAGE_MINOR {
        return Err(TritPackageError::UnsupportedVersion { major, minor });
    }
    let features = read_u32(header, 12)?;
    if features != 0 {
        return Err(TritPackageError::UnknownRequiredFeatures(features));
    }
    if header[48..TRIT_HEADER_BYTES].iter().any(|byte| *byte != 0) {
        return Err(TritPackageError::NonZeroReserved);
    }
    let manifest_offset = as_usize(read_u64(header, 16)?)?;
    let manifest_len = as_usize(read_u64(header, 24)?)?;
    let directory_offset = as_usize(read_u64(header, 32)?)?;
    let directory_len = as_usize(read_u64(header, 40)?)?;
    if manifest_len > MAX_MANIFEST_BYTES.min(max_manifest_bytes)
        || directory_len > MAX_DIRECTORY_BYTES.min(max_directory_bytes)
    {
        return Err(TritPackageError::MetadataLimitExceeded);
    }
    let manifest_end = manifest_offset
        .checked_add(manifest_len)
        .ok_or(TritPackageError::LengthOverflow)?;
    let directory_end = directory_offset
        .checked_add(directory_len)
        .ok_or(TritPackageError::LengthOverflow)?;
    let footer_offset = length - FOOTER_BYTES;
    if manifest_offset < TRIT_HEADER_BYTES
        || manifest_end != directory_offset
        || directory_end != footer_offset
    {
        return Err(TritPackageError::InvalidDirectory);
    }
    Ok(PackageReadLayout {
        manifest: manifest_offset..manifest_end,
        directory: directory_offset..directory_end,
    })
}

fn validate_metadata_digests(
    manifest: &[u8],
    directory: &[u8],
    footer: &[u8],
) -> Result<(), TritPackageError> {
    if blake3::hash(manifest).as_bytes() != &footer[8..40]
        || blake3::hash(directory).as_bytes() != &footer[40..]
    {
        return Err(TritPackageError::DigestMismatch);
    }
    Ok(())
}

// One invariant owner, with borrowed and seek-backed range adapters. Neither
// adapter defines independent directory, geometry or identity semantics.
trait PackageRanges {
    type Error: From<TritPackageError>;
    fn digest(&mut self, range: core::ops::Range<usize>) -> Result<[u8; 32], Self::Error>;
    fn is_zero(&mut self, range: core::ops::Range<usize>) -> Result<bool, Self::Error>;
}

impl PackageRanges for &[u8] {
    type Error = TritPackageError;

    fn digest(&mut self, range: core::ops::Range<usize>) -> Result<[u8; 32], Self::Error> {
        let payload = self.get(range).ok_or(TritPackageError::InvalidDirectory)?;
        Ok(*blake3::hash(payload).as_bytes())
    }

    fn is_zero(&mut self, range: core::ops::Range<usize>) -> Result<bool, Self::Error> {
        let padding = self.get(range).ok_or(TritPackageError::InvalidDirectory)?;
        Ok(padding.iter().all(|byte| *byte == 0))
    }
}

fn validate_blob_ranges<S: PackageRanges>(
    package: &mut S,
    manifest_offset: usize,
    blobs: &[TritBlobInfo],
) -> Result<(), S::Error> {
    let mut unique = BTreeMap::<[u8; 32], (usize, usize)>::new();
    for blob in blobs {
        let offset = as_usize(blob.offset)?;
        let length = as_usize(blob.length)?;
        let end = offset
            .checked_add(length)
            .ok_or(TritPackageError::LengthOverflow)?;
        if offset < TRIT_HEADER_BYTES || offset % TRIT_BLOB_ALIGNMENT != 0 || end > manifest_offset
        {
            return Err(TritPackageError::InvalidDirectory.into());
        }
        if let Some(previous) = unique.insert(*blob.id.as_bytes(), (offset, end))
            && previous != (offset, end)
        {
            return Err(TritPackageError::InvalidDirectory.into());
        }
        if &package.digest(offset..end)? != blob.id.as_bytes() {
            return Err(TritPackageError::BlobDigestMismatch(blob.id).into());
        }
    }
    let mut ranges: Vec<(usize, usize)> = unique.values().copied().collect();
    ranges.sort_unstable();
    let mut previous_end = TRIT_HEADER_BYTES;
    for (start, end) in ranges {
        if start < previous_end || !package.is_zero(previous_end..start)? {
            return Err(TritPackageError::InvalidDirectory.into());
        }
        previous_end = end;
    }
    if !package.is_zero(previous_end..manifest_offset)? {
        return Err(TritPackageError::InvalidDirectory.into());
    }
    Ok(())
}

fn encode_directory(blobs: &[TritBlobInfo]) -> Result<Vec<u8>, TritPackageError> {
    let entries = blobs
        .iter()
        .map(|blob| {
            let mut fields = directory_fields(&blob.kind, blob.offset, blob.length);
            fields.insert(0, Value::Bytes(blob.id.as_bytes().to_vec()));
            Value::Array(fields)
        })
        .collect();
    encode_cbor_value(&directory_value(entries))
}

fn directory_fields(kind: &TritBlobKind, offset: u64, length: u64) -> Vec<Value> {
    let (kind, aux) = encode_kind(kind);
    vec![
        Value::Integer(kind.into()),
        aux,
        Value::Integer(offset.into()),
        Value::Integer(length.into()),
    ]
}
fn directory_value(entries: Vec<Value>) -> Value {
    Value::Array(vec![
        Value::Integer(DIRECTORY_VERSION.into()),
        Value::Array(entries),
    ])
}

fn decode_directory(bytes: &[u8]) -> Result<Vec<TritBlobInfo>, TritPackageError> {
    if let Some(blobs) = decode_borrowed_directory(bytes)? {
        return Ok(blobs);
    }
    // Preserve the reference codec's original diagnostic precedence for
    // oversized, malformed or outside-fast-profile inputs. This optimization
    // does not claim complete adversarial metadata allocation admission.
    decode_reference_directory(bytes)
}

fn decode_borrowed_directory(bytes: &[u8]) -> Result<Option<Vec<TritBlobInfo>>, TritPackageError> {
    let Ok(mut entries) = cbor::Directory::new(bytes) else {
        return Ok(None);
    };
    if entries.len() > MAX_BLOBS {
        return Ok(None);
    }
    let mut blobs = Vec::new();
    blobs
        .try_reserve_exact(entries.len())
        .map_err(|_| TritPackageError::AllocationFailed {
            requested: entries.len(),
        })?;
    loop {
        let entry = match entries.next() {
            Ok(Some(entry)) => entry,
            Ok(None) => break,
            Err(_) => return Ok(None),
        };
        let kind = match (entry.kind, entry.aux) {
            (0, None) => TritBlobKind::AdditiveTensor,
            (3, None) => TritBlobKind::QteaTensor,
            (1, Some(name)) => match dtype_from_str(name) {
                Ok(dtype) => TritBlobKind::DenseTensor(dtype),
                Err(_) => return Ok(None),
            },
            (2, Some(name)) if !name.is_empty() => TritBlobKind::Asset(name.into()),
            _ => return Ok(None),
        };
        blobs.push(TritBlobInfo {
            id: BlobId::from_bytes(*entry.id),
            kind,
            offset: entry.offset,
            length: entry.length,
        });
    }
    if blobs
        .windows(2)
        .any(|pair| compare_blob_entries(&pair[0], &pair[1]).is_ge())
    {
        return Ok(None);
    }
    // Complete canonical preflight and the exact directory tuple/kind grammar
    // establish the reference re-encoding, without allocating those bytes.
    Ok(Some(blobs))
}

fn decode_reference_directory(bytes: &[u8]) -> Result<Vec<TritBlobInfo>, TritPackageError> {
    let value: Value =
        ciborium::de::from_reader(bytes).map_err(|_| TritPackageError::InvalidDirectory)?;
    if encode_cbor_value(&value)? != bytes {
        return Err(TritPackageError::InvalidDirectory);
    }
    let Value::Array(root) = value else {
        return Err(TritPackageError::InvalidDirectory);
    };
    if root.len() != 2 || integer(&root[0])? != DIRECTORY_VERSION {
        return Err(TritPackageError::InvalidDirectory);
    }
    let Value::Array(entries) = &root[1] else {
        return Err(TritPackageError::InvalidDirectory);
    };
    if entries.len() > MAX_BLOBS {
        return Err(TritPackageError::MetadataLimitExceeded);
    }
    let mut blobs = Vec::new();
    blobs
        .try_reserve_exact(entries.len())
        .map_err(|_| TritPackageError::AllocationFailed {
            requested: entries.len(),
        })?;
    for entry in entries {
        let Value::Array(fields) = entry else {
            return Err(TritPackageError::InvalidDirectory);
        };
        if fields.len() != 5 {
            return Err(TritPackageError::InvalidDirectory);
        }
        let Value::Bytes(id) = &fields[0] else {
            return Err(TritPackageError::InvalidDirectory);
        };
        let id: [u8; 32] = id
            .as_slice()
            .try_into()
            .map_err(|_| TritPackageError::InvalidDirectory)?;
        let kind_tag = integer(&fields[1])?;
        let kind = decode_kind(kind_tag, &fields[2])?;
        blobs.push(TritBlobInfo {
            id: BlobId::from_bytes(id),
            kind,
            offset: integer(&fields[3])?,
            length: integer(&fields[4])?,
        });
    }
    if blobs
        .windows(2)
        .any(|pair| compare_blob_entries(&pair[0], &pair[1]).is_ge())
    {
        return Err(TritPackageError::InvalidDirectory);
    }
    if encode_directory(&blobs)? != bytes {
        return Err(TritPackageError::InvalidDirectory);
    }
    Ok(blobs)
}

fn encode_kind(kind: &TritBlobKind) -> (u64, Value) {
    match kind {
        TritBlobKind::AdditiveTensor => (0, Value::Null),
        TritBlobKind::QteaTensor => (3, Value::Null),
        TritBlobKind::DenseTensor(dtype) => (1, Value::Text(dtype.to_string())),
        TritBlobKind::Asset(name) => (2, Value::Text(name.clone())),
    }
}

fn decode_kind(tag: u64, value: &Value) -> Result<TritBlobKind, TritPackageError> {
    match (tag, value) {
        (0, Value::Null) => Ok(TritBlobKind::AdditiveTensor),
        (3, Value::Null) => Ok(TritBlobKind::QteaTensor),
        (1, Value::Text(name)) => dtype_from_str(name).map(TritBlobKind::DenseTensor),
        (2, Value::Text(name)) if !name.is_empty() => Ok(TritBlobKind::Asset(name.clone())),
        _ => Err(TritPackageError::InvalidDirectory),
    }
}

fn dtype_from_str(name: &str) -> Result<DType, TritPackageError> {
    match name {
        "ternary" => Ok(DType::Ternary),
        "i4" => Ok(DType::I4),
        "i8" => Ok(DType::I8),
        "u8" => Ok(DType::U8),
        "f8e4m3" => Ok(DType::F8E4M3),
        "f8e5m2" => Ok(DType::F8E5M2),
        "f4e2m1" => Ok(DType::F4E2M1),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        "f32" => Ok(DType::F32),
        _ => Err(TritPackageError::InvalidDirectory),
    }
}

pub(crate) fn validate_canonical_manifest(bytes: &[u8]) -> Result<(), TritPackageError> {
    match cbor::validate(bytes) {
        Ok(()) => Ok(()),
        // Preserve the reference codec's tag and compound-key semantics. Typed
        // model metadata has neither; its count arrays take the borrowed path.
        Err(cbor::ScanError::Reference) => parse_canonical_manifest(bytes).map(|_| ()),
        Err(cbor::ScanError::Invalid) => Err(TritPackageError::NonCanonicalManifest),
    }
}

/// Serialize through the schema's derived serializer, without a Value per item.
/// Only checked map spans are retained during in-place canonical ordering.
pub(crate) fn encode_borrowed_manifest<T: serde::Serialize>(
    manifest: &T,
) -> Result<Option<Vec<u8>>, TritPackageError> {
    encode_borrowed_metadata(manifest, MAX_MANIFEST_BYTES)
}

// Semantic metadata historically has no manifest-size ceiling. Callers select
// the existing limit, without widening package or typed-manifest admission.
pub(crate) fn encode_borrowed_metadata<T: serde::Serialize>(
    metadata: &T,
    limit: usize,
) -> Result<Option<Vec<u8>>, TritPackageError> {
    let mut output = cbor::CappedOutput::with_limit(limit);
    if ciborium::ser::into_writer(metadata, &mut output).is_err() {
        return Err(if output.exceeded {
            TritPackageError::NonCanonicalManifest
        } else {
            TritPackageError::InvalidDirectory
        });
    }
    let canonical = if limit == MAX_MANIFEST_BYTES {
        cbor::canonicalize(&mut output.bytes)
    } else {
        cbor::canonicalize_with_limit(&mut output.bytes, limit)
    };
    match canonical {
        Ok(()) => {
            cbor::validate_with_limit(&output.bytes, limit)
                .map_err(|_| TritPackageError::NonCanonicalManifest)?;
            Ok(Some(output.bytes))
        }
        Err(cbor::ScanError::Reference) => Ok(None),
        Err(cbor::ScanError::Invalid) => Err(TritPackageError::NonCanonicalManifest),
    }
}

pub(crate) fn parse_canonical_manifest(bytes: &[u8]) -> Result<Value, TritPackageError> {
    if bytes.is_empty() || bytes.len() > MAX_MANIFEST_BYTES {
        return Err(TritPackageError::NonCanonicalManifest);
    }
    let mut value: Value =
        ciborium::de::from_reader(bytes).map_err(|_| TritPackageError::NonCanonicalManifest)?;
    canonicalize_value(&mut value, 0)?;
    if encode_cbor_value(&value)? != bytes {
        return Err(TritPackageError::NonCanonicalManifest);
    }
    Ok(value)
}

pub(crate) fn canonicalize_value(value: &mut Value, depth: usize) -> Result<(), TritPackageError> {
    if depth > 64 {
        return Err(TritPackageError::NonCanonicalManifest);
    }
    match value {
        Value::Array(items) => {
            for item in items {
                canonicalize_value(item, depth + 1)?;
            }
        }
        Value::Map(entries) => {
            for (key, item) in entries.iter_mut() {
                canonicalize_value(key, depth + 1)?;
                canonicalize_value(item, depth + 1)?;
            }
            entries.sort_by_cached_key(|entry| CanonicalValue::from(entry.0.clone()));
            if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                return Err(TritPackageError::NonCanonicalManifest);
            }
        }
        Value::Tag(_, inner) => canonicalize_value(inner, depth + 1)?,
        Value::Float(_) => return Err(TritPackageError::NonCanonicalManifest),
        _ => {}
    }
    Ok(())
}

pub(crate) fn encode_cbor_value(value: &Value) -> Result<Vec<u8>, TritPackageError> {
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(value, &mut bytes)
        .map_err(|_| TritPackageError::InvalidDirectory)?;
    Ok(bytes)
}

fn integer(value: &Value) -> Result<u64, TritPackageError> {
    value
        .as_integer()
        .and_then(|number| u64::try_from(number).ok())
        .ok_or(TritPackageError::InvalidDirectory)
}

fn kind_sort_key(kind: &TritBlobKind) -> (u8, String) {
    match kind {
        TritBlobKind::AdditiveTensor => (0, String::new()),
        TritBlobKind::QteaTensor => (3, String::new()),
        TritBlobKind::DenseTensor(dtype) => (1, dtype.to_string()),
        TritBlobKind::Asset(name) => (2, name.clone()),
    }
}

fn compare_blob_entries(left: &TritBlobInfo, right: &TritBlobInfo) -> core::cmp::Ordering {
    left.id
        .as_bytes()
        .cmp(right.id.as_bytes())
        .then_with(|| kind_sort_key(&left.kind).cmp(&kind_sort_key(&right.kind)))
}

fn align_up(value: usize) -> Result<usize, TritPackageError> {
    value
        .checked_add(TRIT_BLOB_ALIGNMENT - 1)
        .map(|value| value / TRIT_BLOB_ALIGNMENT * TRIT_BLOB_ALIGNMENT)
        .ok_or(TritPackageError::LengthOverflow)
}

fn write_u64(bytes: &mut [u8], offset: usize, value: usize) -> Result<(), TritPackageError> {
    let value = u64::try_from(value).map_err(|_| TritPackageError::LengthOverflow)?;
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, TritPackageError> {
    Ok(u16::from_le_bytes(
        bytes
            .get(offset..offset + 2)
            .ok_or(TritPackageError::Truncated)?
            .try_into()
            .map_err(|_| TritPackageError::Truncated)?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, TritPackageError> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or(TritPackageError::Truncated)?
            .try_into()
            .map_err(|_| TritPackageError::Truncated)?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, TritPackageError> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or(TritPackageError::Truncated)?
            .try_into()
            .map_err(|_| TritPackageError::Truncated)?,
    ))
}

fn as_usize(value: u64) -> Result<usize, TritPackageError> {
    usize::try_from(value).map_err(|_| TritPackageError::LengthOverflow)
}

#[cfg(test)]
mod tests {
    use super::{
        TRIT_BLOB_ALIGNMENT, TritBlob, TritBlobKind, TritPackageError, read_trit_package,
        validate_canonical_manifest, write_trit_package,
    };
    use crate::PackageId;
    use tritium_core::DType;

    const MANIFEST: &[u8] = &[0x82, 0x01, 0x02]; // canonical CBOR [1, 2]

    #[test]
    fn package_roundtrips_content_addressed_blobs_and_manifest() {
        let package = write_trit_package(
            MANIFEST,
            &[
                TritBlob {
                    kind: TritBlobKind::AdditiveTensor,
                    bytes: b"ternary payload",
                },
                TritBlob {
                    kind: TritBlobKind::DenseTensor(DType::F16),
                    bytes: b"dense payload",
                },
                TritBlob {
                    kind: TritBlobKind::Asset("tokenizer.json".into()),
                    bytes: b"{}",
                },
            ],
        )
        .expect("package writes");
        let parsed = read_trit_package(&package).expect("package reads");
        assert_eq!(parsed.manifest(), MANIFEST);
        assert_eq!(parsed.blobs().len(), 3);
        assert_eq!(
            parsed.package_id().as_bytes(),
            PackageId::from_package_bytes(&package).as_bytes()
        );
        for blob in parsed.blobs() {
            assert_eq!(blob.offset as usize % TRIT_BLOB_ALIGNMENT, 0);
            assert_eq!(
                parsed.blob(blob).expect("blob digest verifies").len() as u64,
                blob.length
            );
        }
        assert_eq!(
            write_trit_package(
                MANIFEST,
                &[
                    TritBlob {
                        kind: TritBlobKind::AdditiveTensor,
                        bytes: b"ternary payload",
                    },
                    TritBlob {
                        kind: TritBlobKind::DenseTensor(DType::F16),
                        bytes: b"dense payload",
                    },
                    TritBlob {
                        kind: TritBlobKind::Asset("tokenizer.json".into()),
                        bytes: b"{}",
                    },
                ],
            )
            .expect("deterministic repeat write"),
            package
        );
    }

    #[test]
    fn identical_payloads_share_aligned_storage() {
        let package = write_trit_package(
            MANIFEST,
            &[
                TritBlob {
                    kind: TritBlobKind::AdditiveTensor,
                    bytes: b"shared",
                },
                TritBlob {
                    kind: TritBlobKind::Asset("alias.bin".into()),
                    bytes: b"shared",
                },
                TritBlob {
                    kind: TritBlobKind::AdditiveTensor,
                    bytes: b"shared",
                },
            ],
        )
        .expect("package writes");
        let parsed = read_trit_package(&package).expect("package reads");
        assert_eq!(parsed.blobs().len(), 2);
        assert_eq!(parsed.blobs()[0].offset, parsed.blobs()[1].offset);
        assert_eq!(parsed.blobs()[0].id, parsed.blobs()[1].id);
    }

    #[test]
    fn reader_rejects_unknown_features_and_tampered_payload_or_manifest() {
        let package = write_trit_package(
            MANIFEST,
            &[TritBlob {
                kind: TritBlobKind::Asset("asset".into()),
                bytes: b"content",
            }],
        )
        .expect("package writes");
        let mut unknown_feature = package.clone();
        unknown_feature[12] = 1;
        assert!(matches!(
            read_trit_package(&unknown_feature),
            Err(TritPackageError::UnknownRequiredFeatures(1))
        ));

        let mut changed_manifest = package.clone();
        let manifest_offset = u64::from_le_bytes(
            changed_manifest[16..24]
                .try_into()
                .expect("header manifest offset"),
        ) as usize;
        changed_manifest[manifest_offset] ^= 1;
        assert!(matches!(
            read_trit_package(&changed_manifest),
            Err(TritPackageError::DigestMismatch)
        ));

        let mut changed_blob = package;
        let parsed = read_trit_package(&changed_blob).expect("valid package");
        let offset = usize::try_from(parsed.blobs()[0].offset).expect("host offset");
        changed_blob[offset] ^= 1;
        assert!(matches!(
            read_trit_package(&changed_blob),
            Err(TritPackageError::BlobDigestMismatch(_))
        ));
    }

    #[test]
    fn manifest_must_use_deterministic_supported_cbor() {
        assert!(validate_canonical_manifest(&[0x82, 0x01, 0x02]).is_ok());
        assert_eq!(
            validate_canonical_manifest(&[0x82, 0x18, 0x01, 0x02]),
            Err(TritPackageError::NonCanonicalManifest)
        );
        assert_eq!(
            validate_canonical_manifest(&[0xf9, 0x3c, 0x00]), // float16 1.0
            Err(TritPackageError::NonCanonicalManifest)
        );
        let canonical_map = [0xa2, 0x61, b'a', 0x01, 0x62, b'b', b'b', 0x02];
        assert!(validate_canonical_manifest(&canonical_map).is_ok());
        let reversed_map = [0xa2, 0x62, b'b', b'b', 0x02, 0x61, b'a', 0x01];
        assert_eq!(
            validate_canonical_manifest(&reversed_map),
            Err(TritPackageError::NonCanonicalManifest)
        );
        let duplicate_key = [0xa2, 0x61, b'a', 0x01, 0x61, b'a', 0x02];
        assert_eq!(
            validate_canonical_manifest(&duplicate_key),
            Err(TritPackageError::NonCanonicalManifest)
        );
    }
}
