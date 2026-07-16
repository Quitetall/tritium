//! Immutable, streaming, per-tensor work records for large synthesis campaigns.

use core::fmt;
use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use tritium_format::ModelId;

use crate::{CONTENT_ID_CONTEXT, ContentId};

const RECORD_MAGIC: [u8; 8] = *b"TSTWORK\0";
const RECORD_VERSION: u16 = 1;
const RECORD_FIXED_BYTES: u64 = 136;
const RECORD_FOOTER_BYTES: u64 = 32;
const RECEIPT_MAGIC: [u8; 8] = *b"TSTWREF\0";
const RECEIPT_VERSION: u8 = 1;
const RECEIPT_CHECKSUM_BYTES: usize = 32;
const RECEIPT_CHECKSUM_CONTEXT: &str = "tritium tensor work receipt checksum v1";
const PAYLOAD_DIGEST_CONTEXT: &str = "tritium tensor work payload v1";
const OBJECT_DIRECTORY: &str = "objects";
const TEMP_DIRECTORY: &str = ".tmp";
const RECORD_EXTENSION: &str = "twr";
const MAX_NAME_BYTES: usize = 64 * 1024;
const MAX_RANK: usize = 32;
const MAX_SCHEMA_METADATA_BYTES: usize = 16 * 1024 * 1024;
const MAX_RECEIPT_BYTES: usize = MAX_SCHEMA_METADATA_BYTES + MAX_NAME_BYTES + 1024;
const MAX_STAGING_BYTES: usize = 64 * 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Complete immutable descriptor for one tensor work record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRecordInfo {
    schema_id: ContentId,
    source_model_id: ModelId,
    source_tensor_digest: [u8; 32],
    name: String,
    shape: Vec<u64>,
    schema_metadata: Vec<u8>,
    payload_bytes: u64,
}

impl TensorRecordInfo {
    /// Versioned schema identity interpreting the opaque payload.
    #[must_use]
    pub const fn schema_id(&self) -> ContentId {
        self.schema_id
    }

    /// Semantic identity of the source model.
    #[must_use]
    pub const fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Architecture-adapter digest of the source tensor's canonical logical bytes.
    #[must_use]
    pub const fn source_tensor_digest(&self) -> &[u8; 32] {
        &self.source_tensor_digest
    }

    /// Canonical source tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Logical tensor dimensions.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Schema-specific canonical metadata.
    #[must_use]
    pub fn schema_metadata(&self) -> &[u8] {
        &self.schema_metadata
    }

    /// Exact opaque payload length.
    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
}

/// Validated descriptor supplied to [`TensorWorkStore::put`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRecordSpec(TensorRecordInfo);

impl TensorRecordSpec {
    /// Construct a bounded, nonempty tensor record specification.
    ///
    /// # Errors
    /// Returns [`TensorWorkError::InvalidSpec`] for an empty or oversized name,
    /// invalid shape, oversized schema metadata, or zero payload length.
    pub fn new(
        schema_id: ContentId,
        source_model_id: ModelId,
        source_tensor_digest: [u8; 32],
        name: impl Into<String>,
        shape: Vec<u64>,
        schema_metadata: Vec<u8>,
        payload_bytes: u64,
    ) -> Result<Self, TensorWorkError> {
        let info = TensorRecordInfo {
            schema_id,
            source_model_id,
            source_tensor_digest,
            name: name.into(),
            shape,
            schema_metadata,
            payload_bytes,
        };
        validate_info(&info)?;
        exact_record_bytes(&info)?;
        Ok(Self(info))
    }

    /// Versioned payload schema identity.
    #[must_use]
    pub const fn schema_id(&self) -> ContentId {
        self.0.schema_id()
    }

    /// Semantic source model identity.
    #[must_use]
    pub const fn source_model_id(&self) -> ModelId {
        self.0.source_model_id()
    }

    /// Source tensor semantic digest.
    #[must_use]
    pub const fn source_tensor_digest(&self) -> &[u8; 32] {
        self.0.source_tensor_digest()
    }

    /// Canonical tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.0.name()
    }

    /// Logical dimensions.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        self.0.shape()
    }

    /// Canonical schema metadata.
    #[must_use]
    pub fn schema_metadata(&self) -> &[u8] {
        self.0.schema_metadata()
    }

    /// Required payload length.
    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.0.payload_bytes()
    }
}

/// Durable reference to one exact immutable tensor work record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRecordReceipt {
    record_id: ContentId,
    record_bytes: u64,
    payload_digest: [u8; 32],
    info: TensorRecordInfo,
}

impl TensorRecordReceipt {
    /// Content identity of the complete exact record bytes.
    #[must_use]
    pub const fn record_id(&self) -> ContentId {
        self.record_id
    }

    /// Complete exact record length, including framing and payload digest.
    #[must_use]
    pub const fn record_bytes(&self) -> u64 {
        self.record_bytes
    }

    /// Domain-separated digest of the opaque payload bytes.
    #[must_use]
    pub const fn payload_digest(&self) -> &[u8; 32] {
        &self.payload_digest
    }

    /// Descriptor bound into the exact record.
    #[must_use]
    pub const fn info(&self) -> &TensorRecordInfo {
        &self.info
    }

    /// Whether this receipt describes exactly the supplied specification.
    #[must_use]
    pub fn matches_spec(&self, spec: &TensorRecordSpec) -> bool {
        self.info == spec.0 && self.record_bytes == exact_record_bytes(&self.info).unwrap_or(0)
    }

    /// Encode a bounded canonical receipt suitable for a resumable manifest.
    ///
    /// # Errors
    /// Returns [`TensorWorkError`] if a checked record length overflows.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, TensorWorkError> {
        validate_info(&self.info)?;
        if self.record_bytes != exact_record_bytes(&self.info)? {
            return Err(TensorWorkError::ReceiptMismatch);
        }
        let mut output = Vec::new();
        output.extend_from_slice(&RECEIPT_MAGIC);
        output.push(RECEIPT_VERSION);
        output.extend_from_slice(self.record_id.as_bytes());
        output.extend_from_slice(&self.record_bytes.to_le_bytes());
        output.extend_from_slice(&self.payload_digest);
        encode_info(&mut output, &self.info)?;
        let mut hasher = blake3::Hasher::new_derive_key(RECEIPT_CHECKSUM_CONTEXT);
        hasher.update(&output);
        output.extend_from_slice(hasher.finalize().as_bytes());
        if output.len() > MAX_RECEIPT_BYTES {
            return Err(TensorWorkError::InvalidReceipt("receipt too large"));
        }
        Ok(output)
    }

    /// Decode only a checksum-valid, canonical, internally consistent receipt.
    ///
    /// # Errors
    /// Returns [`TensorWorkError`] for malformed, oversized, unsupported, or
    /// noncanonical bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, TensorWorkError> {
        if bytes.len() > MAX_RECEIPT_BYTES {
            return Err(TensorWorkError::InvalidReceipt("receipt too large"));
        }
        if bytes.len() < RECEIPT_MAGIC.len() + 1 + RECEIPT_CHECKSUM_BYTES {
            return Err(TensorWorkError::InvalidReceipt("truncated receipt"));
        }
        let checksum_offset = bytes.len() - RECEIPT_CHECKSUM_BYTES;
        let (payload, checksum) = bytes.split_at(checksum_offset);
        let mut hasher = blake3::Hasher::new_derive_key(RECEIPT_CHECKSUM_CONTEXT);
        hasher.update(payload);
        if hasher.finalize().as_bytes() != checksum {
            return Err(TensorWorkError::InvalidReceipt("checksum mismatch"));
        }
        let mut cursor = ReceiptCursor::new(payload);
        if cursor.take(RECEIPT_MAGIC.len())? != RECEIPT_MAGIC {
            return Err(TensorWorkError::InvalidReceipt("magic"));
        }
        if cursor.u8()? != RECEIPT_VERSION {
            return Err(TensorWorkError::InvalidReceipt("version"));
        }
        let record_id = ContentId::from_digest(cursor.digest()?);
        let record_bytes = cursor.u64()?;
        let payload_digest = cursor.digest()?;
        let info = decode_info(&mut cursor)?;
        if cursor.remaining() != 0 {
            return Err(TensorWorkError::InvalidReceipt("trailing bytes"));
        }
        validate_info(&info)?;
        if record_bytes != exact_record_bytes(&info)? {
            return Err(TensorWorkError::ReceiptMismatch);
        }
        let receipt = Self {
            record_id,
            record_bytes,
            payload_digest,
            info,
        };
        if receipt.canonical_bytes()? != bytes {
            return Err(TensorWorkError::InvalidReceipt("noncanonical receipt"));
        }
        Ok(receipt)
    }
}

/// Filesystem-backed immutable tensor-object store.
#[derive(Debug)]
pub struct TensorWorkStore {
    root: PathBuf,
    objects: PathBuf,
    temporary: PathBuf,
}

impl TensorWorkStore {
    /// Open or create one work store rooted at `root`.
    ///
    /// # Errors
    /// Returns [`TensorWorkError`] if a required path is a symlink, special
    /// file, or cannot be created and inspected.
    pub fn open(root: &Path) -> Result<Self, TensorWorkError> {
        let root = absolute_path(root)?;
        ensure_durable_directory(&root, "store root")?;
        let objects = root.join(OBJECT_DIRECTORY);
        ensure_durable_directory(&objects, "object directory")?;
        let temporary = root.join(TEMP_DIRECTORY);
        ensure_durable_directory(&temporary, "temporary directory")?;
        Ok(Self {
            root,
            objects,
            temporary,
        })
    }

    /// Store root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Directory containing content-addressed record-prefix directories.
    #[must_use]
    pub fn objects_dir(&self) -> &Path {
        &self.objects
    }

    #[cfg(unix)]
    pub(crate) fn temporary_dir(&self) -> &Path {
        &self.temporary
    }

    /// Deterministic content-addressed path for a record ID.
    #[must_use]
    pub fn record_path(&self, record_id: ContentId) -> PathBuf {
        let prefix = format!("{:02x}", record_id.as_bytes()[0]);
        self.objects
            .join(prefix)
            .join(format!("{record_id}.{RECORD_EXTENSION}"))
    }

    /// Remove crash-left temporary records while the caller holds exclusive store ownership.
    #[cfg(unix)]
    pub(crate) fn scavenge_temporary(&self) -> Result<u64, TensorWorkError> {
        let mut removed = 0_u64;
        let entries = fs::read_dir(&self.temporary)
            .map_err(|error| work_io("read temporary record directory", error))?;
        for entry in entries {
            let entry = entry.map_err(|error| work_io("read temporary record entry", error))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| work_io("inspect temporary record", error))?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(TensorWorkError::InvalidPath("temporary record"));
            }
            fs::remove_file(&path)
                .map_err(|error| work_io("remove stale temporary record", error))?;
            removed = removed
                .checked_add(1)
                .ok_or(TensorWorkError::LengthOverflow)?;
        }
        if removed != 0 {
            sync_directory(&self.temporary, "sync scavenged temporary directory")?;
        }
        Ok(removed)
    }

    /// Stream and immutably publish one exact tensor record.
    ///
    /// The producer may write in arbitrary chunk sizes but must write exactly
    /// the declared payload length. Producer failure, overrun, and short output
    /// leave no newly published record. Publication itself is atomic. A cleanup
    /// or durability error after the no-replace link succeeds may return an error
    /// with the exact record already visible; retrying the same put is idempotent.
    ///
    /// # Errors
    /// Returns [`TensorPutError::Producer`] without erasing the producer's typed
    /// error, or [`TensorPutError::Store`] for framing, length, I/O, publication,
    /// or existing-object validation failures.
    pub fn put<E>(
        &self,
        spec: &TensorRecordSpec,
        produce: impl FnOnce(&mut TensorPayloadWriter<'_>) -> Result<(), E>,
    ) -> Result<TensorRecordReceipt, TensorPutError<E>> {
        validate_info(&spec.0).map_err(TensorPutError::Store)?;
        let record_bytes = exact_record_bytes(&spec.0).map_err(TensorPutError::Store)?;
        let (temporary, file) =
            create_temporary_file(&self.temporary, "record.tmp").map_err(TensorPutError::Store)?;
        let result = self.write_temporary(file, &temporary, spec, record_bytes, produce);
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        result
    }

    fn write_temporary<E>(
        &self,
        file: File,
        temporary: &Path,
        spec: &TensorRecordSpec,
        record_bytes: u64,
        produce: impl FnOnce(&mut TensorPayloadWriter<'_>) -> Result<(), E>,
    ) -> Result<TensorRecordReceipt, TensorPutError<E>> {
        let mut sink = RecordSink::new(file);
        write_record_prefix(&mut sink, &spec.0, record_bytes).map_err(TensorPutError::Store)?;
        let mut payload_writer = TensorPayloadWriter::new(&mut sink, spec.payload_bytes());
        let produced = produce(&mut payload_writer);
        let actual = payload_writer.written;
        let overrun = payload_writer.overrun;
        let ignored_io_failure = payload_writer.io_failure;
        let payload_digest = *payload_writer.payload_hasher.finalize().as_bytes();
        drop(payload_writer);
        if let Err(error) = produced {
            return Err(TensorPutError::Producer(error));
        }
        if overrun {
            return Err(TensorPutError::Store(TensorWorkError::PayloadOverrun {
                expected: spec.payload_bytes(),
            }));
        }
        if let Some(kind) = ignored_io_failure {
            return Err(TensorPutError::Store(TensorWorkError::Io {
                operation: "write record payload",
                kind,
            }));
        }
        if actual != spec.payload_bytes() {
            return Err(TensorPutError::Store(
                TensorWorkError::PayloadLengthMismatch {
                    expected: spec.payload_bytes(),
                    actual,
                },
            ));
        }
        sink.write_all(&payload_digest)
            .map_err(|error| TensorPutError::Store(work_io("write record footer", error)))?;
        let (file, record_id, actual_record_bytes) = sink.finish();
        if actual_record_bytes != record_bytes {
            return Err(TensorPutError::Store(
                TensorWorkError::RecordLengthMismatch {
                    expected: record_bytes,
                    actual: actual_record_bytes,
                },
            ));
        }
        file.sync_all()
            .map_err(|error| TensorPutError::Store(work_io("sync temporary record", error)))?;
        let receipt = TensorRecordReceipt {
            record_id,
            record_bytes,
            payload_digest,
            info: spec.0.clone(),
        };
        self.publish(temporary, &receipt)
            .map_err(TensorPutError::Store)?;
        Ok(receipt)
    }

    fn publish(
        &self,
        temporary: &Path,
        receipt: &TensorRecordReceipt,
    ) -> Result<(), TensorWorkError> {
        let path = self.record_path(receipt.record_id);
        let parent = path
            .parent()
            .ok_or(TensorWorkError::InvalidPath("record parent"))?;
        ensure_durable_directory(parent, "record prefix directory")?;
        sync_directory(&self.objects, "sync object directory")?;
        match fs::hard_link(temporary, &path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let existing = open_and_verify(&path, Some(receipt))?;
                drop(existing);
            }
            Err(error) => return Err(work_io("publish tensor record", error)),
        }
        sync_directory(parent, "sync record directory")?;
        fs::remove_file(temporary).map_err(|error| work_io("remove temporary record", error))?;
        sync_directory(&self.temporary, "sync temporary directory")?;
        Ok(())
    }

    /// Open a record through its receipt and validate every exact byte before use.
    ///
    /// The returned reader retains the verified file handle; later path
    /// replacement cannot redirect payload visits.
    ///
    /// # Errors
    /// Returns [`TensorWorkError`] for missing, malformed, mutated, mismatched,
    /// symlinked, special, truncated, or trailing record bytes.
    pub fn open_verified(
        &self,
        receipt: &TensorRecordReceipt,
    ) -> Result<TensorRecordReader, TensorWorkError> {
        open_and_verify(&self.record_path(receipt.record_id), Some(receipt))
    }
}

/// Bounded payload writer handed to a [`TensorWorkStore::put`] producer.
#[derive(Debug)]
pub struct TensorPayloadWriter<'a> {
    sink: &'a mut RecordSink,
    payload_hasher: blake3::Hasher,
    expected: u64,
    written: u64,
    overrun: bool,
    io_failure: Option<io::ErrorKind>,
}

impl<'a> TensorPayloadWriter<'a> {
    fn new(sink: &'a mut RecordSink, expected: u64) -> Self {
        Self {
            sink,
            payload_hasher: blake3::Hasher::new_derive_key(PAYLOAD_DIGEST_CONTEXT),
            expected,
            written: 0,
            overrun: false,
            io_failure: None,
        }
    }

    /// Declared exact payload length.
    #[must_use]
    pub const fn expected_bytes(&self) -> u64 {
        self.expected
    }

    /// Payload bytes accepted so far.
    #[must_use]
    pub const fn written_bytes(&self) -> u64 {
        self.written
    }
}

impl Write for TensorPayloadWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.overrun {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tensor work payload already exceeded its declaration",
            ));
        }
        if self.io_failure.is_some() {
            return Err(io::Error::other("tensor work payload writer is poisoned"));
        }
        let length = u64::try_from(buffer.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "payload chunk exceeds u64")
        })?;
        let next = self.written.checked_add(length).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "payload length overflow")
        })?;
        if next > self.expected {
            self.overrun = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tensor work payload exceeds declared length",
            ));
        }
        if let Err(error) = self.sink.write_all(buffer) {
            self.io_failure = Some(error.kind());
            return Err(error);
        }
        self.payload_hasher.update(buffer);
        self.written = next;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.sink.file.flush()
    }
}

/// Same-handle reader for one fully validated tensor work record.
#[derive(Debug)]
pub struct TensorRecordReader {
    file: File,
    receipt: TensorRecordReceipt,
    payload_offset: u64,
}

impl TensorRecordReader {
    /// Descriptor validated from the exact record.
    #[must_use]
    pub const fn info(&self) -> &TensorRecordInfo {
        &self.receipt.info
    }

    /// Exact validated record identity.
    #[must_use]
    pub const fn record_id(&self) -> ContentId {
        self.receipt.record_id
    }

    /// Stream payload bytes in chunks no larger than the requested size or 64 KiB.
    ///
    /// The complete record header, descriptor, payload digest, exact content ID,
    /// and file length are rechecked through the retained handle. Callback side
    /// effects are nontransactional because final mutation detection necessarily
    /// occurs after the last callback.
    ///
    /// # Errors
    /// Returns [`TensorVisitError::Store`] for invalid chunk size or changed
    /// record bytes and [`TensorVisitError::Sink`] without erasing the callback's
    /// typed failure.
    pub fn try_visit_payload<E>(
        &mut self,
        max_chunk_bytes: usize,
        mut visit: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<(), TensorVisitError<E>> {
        if max_chunk_bytes == 0 {
            return Err(TensorVisitError::Store(TensorWorkError::InvalidChunkSize));
        }
        let before = self
            .file
            .metadata()
            .map_err(|error| TensorVisitError::Store(work_io("inspect tensor record", error)))?
            .len();
        if before != self.receipt.record_bytes {
            return Err(TensorVisitError::Store(
                TensorWorkError::RecordLengthMismatch {
                    expected: self.receipt.record_bytes,
                    actual: before,
                },
            ));
        }
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|error| TensorVisitError::Store(work_io("seek tensor record", error)))?;
        let mut reader = HashingReader::new(&mut self.file);
        let prefix = read_record_prefix(&mut reader).map_err(TensorVisitError::Store)?;
        if prefix.info != self.receipt.info
            || prefix.total_bytes != self.receipt.record_bytes
            || prefix.payload_offset != self.payload_offset
        {
            return Err(TensorVisitError::Store(TensorWorkError::ReceiptMismatch));
        }
        let chunk_bytes = max_chunk_bytes.min(MAX_STAGING_BYTES);
        let mut staging = Vec::new();
        staging
            .try_reserve_exact(chunk_bytes)
            .map_err(|_| TensorVisitError::Store(TensorWorkError::AllocationFailed))?;
        staging.resize(chunk_bytes, 0);
        let mut remaining = prefix.info.payload_bytes;
        let mut payload_hasher = blake3::Hasher::new_derive_key(PAYLOAD_DIGEST_CONTEXT);
        while remaining != 0 {
            let count = usize::try_from(remaining.min(chunk_bytes as u64))
                .map_err(|_| TensorVisitError::Store(TensorWorkError::LengthOverflow))?;
            reader
                .read_exact(&mut staging[..count])
                .map_err(TensorVisitError::Store)?;
            payload_hasher.update(&staging[..count]);
            visit(&staging[..count]).map_err(TensorVisitError::Sink)?;
            remaining -= count as u64;
        }
        let footer = reader.digest().map_err(TensorVisitError::Store)?;
        let actual_payload_digest = *payload_hasher.finalize().as_bytes();
        if footer != actual_payload_digest || footer != self.receipt.payload_digest {
            return Err(TensorVisitError::Store(
                TensorWorkError::PayloadDigestMismatch,
            ));
        }
        if reader.position != prefix.total_bytes {
            return Err(TensorVisitError::Store(
                TensorWorkError::RecordLengthMismatch {
                    expected: prefix.total_bytes,
                    actual: reader.position,
                },
            ));
        }
        let record_id = reader.finish();
        if record_id != self.receipt.record_id {
            return Err(TensorVisitError::Store(TensorWorkError::RecordIdMismatch));
        }
        let after = self
            .file
            .metadata()
            .map_err(|error| TensorVisitError::Store(work_io("reinspect tensor record", error)))?
            .len();
        if after != before {
            return Err(TensorVisitError::Store(
                TensorWorkError::RecordLengthMismatch {
                    expected: before,
                    actual: after,
                },
            ));
        }
        Ok(())
    }
}

/// Store or record-format failure.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorWorkError {
    /// Filesystem operation failed.
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Portable I/O error category.
        kind: io::ErrorKind,
    },
    /// A required path was a symlink, special file, or lacked a parent.
    InvalidPath(&'static str),
    /// A record specification violated a bounded invariant.
    InvalidSpec(&'static str),
    /// Checked length arithmetic overflowed.
    LengthOverflow,
    /// Fallible bounded allocation failed.
    AllocationFailed,
    /// Producer wrote fewer bytes than declared.
    PayloadLengthMismatch {
        /// Required payload bytes.
        expected: u64,
        /// Accepted payload bytes.
        actual: u64,
    },
    /// Producer attempted to write beyond its exact declaration.
    PayloadOverrun {
        /// Required exact payload bytes.
        expected: u64,
    },
    /// Complete record length disagreed with its canonical declaration.
    RecordLengthMismatch {
        /// Declared or receipt-bound bytes.
        expected: u64,
        /// Observed bytes.
        actual: u64,
    },
    /// Record magic was not recognized.
    BadMagic,
    /// Record version is unsupported.
    UnsupportedVersion(u16),
    /// Reserved framing bits were nonzero.
    NonzeroReserved,
    /// Record ended before a declared field or payload completed.
    Truncated,
    /// Record payload digest did not match exact payload bytes.
    PayloadDigestMismatch,
    /// Complete exact record bytes did not match the content-addressed ID.
    RecordIdMismatch,
    /// Record descriptor or lengths disagreed with its receipt.
    ReceiptMismatch,
    /// Canonical receipt bytes were malformed or unsupported.
    InvalidReceipt(&'static str),
    /// Visitor requested zero-byte chunks.
    InvalidChunkSize,
}

impl fmt::Display for TensorWorkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, kind } => write!(formatter, "{operation} failed: {kind}"),
            Self::InvalidPath(field) => write!(formatter, "invalid tensor work {field}"),
            Self::InvalidSpec(field) => write!(formatter, "invalid tensor record {field}"),
            Self::LengthOverflow => formatter.write_str("tensor record length overflow"),
            Self::AllocationFailed => formatter.write_str("tensor work allocation failed"),
            Self::PayloadLengthMismatch { expected, actual } => write!(
                formatter,
                "tensor payload length mismatch: expected {expected}, got {actual}"
            ),
            Self::PayloadOverrun { expected } => {
                write!(
                    formatter,
                    "tensor payload exceeded declared {expected} bytes"
                )
            }
            Self::RecordLengthMismatch { expected, actual } => write!(
                formatter,
                "tensor record length mismatch: expected {expected}, got {actual}"
            ),
            Self::BadMagic => formatter.write_str("invalid tensor record magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported tensor record version {version}")
            }
            Self::NonzeroReserved => formatter.write_str("nonzero tensor record reserved bits"),
            Self::Truncated => formatter.write_str("truncated tensor record"),
            Self::PayloadDigestMismatch => formatter.write_str("tensor payload digest mismatch"),
            Self::RecordIdMismatch => formatter.write_str("tensor record content ID mismatch"),
            Self::ReceiptMismatch => formatter.write_str("tensor record receipt mismatch"),
            Self::InvalidReceipt(field) => write!(formatter, "invalid tensor receipt {field}"),
            Self::InvalidChunkSize => {
                formatter.write_str("tensor visit chunk size must be nonzero")
            }
        }
    }
}

impl Error for TensorWorkError {}

/// Failure from [`TensorWorkStore::put`] retaining a typed producer error.
#[derive(Debug)]
pub enum TensorPutError<E> {
    /// Store, framing, length, I/O, or immutable-publication failure.
    Store(TensorWorkError),
    /// Caller-provided producer stopped before publication.
    Producer(E),
}

impl<E: fmt::Display> fmt::Display for TensorPutError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(formatter, "tensor store put failed: {error}"),
            Self::Producer(error) => write!(formatter, "tensor payload producer failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for TensorPutError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Producer(error) => Some(error),
        }
    }
}

/// Failure while streaming a verified record payload to a typed sink.
#[derive(Debug)]
pub enum TensorVisitError<E> {
    /// Store record changed or failed strict validation.
    Store(TensorWorkError),
    /// Caller-provided sink stopped the stream.
    Sink(E),
}

impl<E: fmt::Display> fmt::Display for TensorVisitError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Store(error) => write!(formatter, "tensor record visit failed: {error}"),
            Self::Sink(error) => write!(formatter, "tensor payload sink failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for TensorVisitError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Store(error) => Some(error),
            Self::Sink(error) => Some(error),
        }
    }
}

#[derive(Debug)]
struct RecordSink {
    file: File,
    hasher: blake3::Hasher,
    written: u64,
}

impl RecordSink {
    fn new(file: File) -> Self {
        Self {
            file,
            hasher: blake3::Hasher::new_derive_key(CONTENT_ID_CONTEXT),
            written: 0,
        }
    }

    fn write_all(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.file.write_all(bytes)?;
        self.hasher.update(bytes);
        self.written = self
            .written
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| io::Error::other("tensor record length overflow"))?;
        Ok(())
    }

    fn finish(self) -> (File, ContentId, u64) {
        (
            self.file,
            ContentId::from_digest(*self.hasher.finalize().as_bytes()),
            self.written,
        )
    }
}

#[derive(Debug)]
struct RecordPrefix {
    total_bytes: u64,
    payload_offset: u64,
    info: TensorRecordInfo,
}

#[derive(Debug)]
struct HashingReader<'a> {
    file: &'a mut File,
    hasher: blake3::Hasher,
    position: u64,
}

impl<'a> HashingReader<'a> {
    fn new(file: &'a mut File) -> Self {
        Self {
            file,
            hasher: blake3::Hasher::new_derive_key(CONTENT_ID_CONTEXT),
            position: 0,
        }
    }

    fn read_exact(&mut self, output: &mut [u8]) -> Result<(), TensorWorkError> {
        match self.file.read_exact(output) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                return Err(TensorWorkError::Truncated);
            }
            Err(error) => return Err(work_io("read tensor record", error)),
        }
        self.hasher.update(output);
        self.position = self
            .position
            .checked_add(output.len() as u64)
            .ok_or(TensorWorkError::LengthOverflow)?;
        Ok(())
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], TensorWorkError> {
        let mut bytes = [0; N];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u16(&mut self) -> Result<u16, TensorWorkError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, TensorWorkError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, TensorWorkError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn digest(&mut self) -> Result<[u8; 32], TensorWorkError> {
        self.array()
    }

    fn bounded_vec(&mut self, length: usize) -> Result<Vec<u8>, TensorWorkError> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|_| TensorWorkError::AllocationFailed)?;
        output.resize(length, 0);
        self.read_exact(&mut output)?;
        Ok(output)
    }

    fn finish(self) -> ContentId {
        ContentId::from_digest(*self.hasher.finalize().as_bytes())
    }
}

fn write_record_prefix(
    sink: &mut RecordSink,
    info: &TensorRecordInfo,
    record_bytes: u64,
) -> Result<(), TensorWorkError> {
    let name_len = u32::try_from(info.name.len()).map_err(|_| TensorWorkError::LengthOverflow)?;
    let rank = u16::try_from(info.shape.len()).map_err(|_| TensorWorkError::LengthOverflow)?;
    let metadata_len =
        u32::try_from(info.schema_metadata.len()).map_err(|_| TensorWorkError::LengthOverflow)?;
    for bytes in [
        RECORD_MAGIC.as_slice(),
        &RECORD_VERSION.to_le_bytes(),
        &0_u16.to_le_bytes(),
        &record_bytes.to_le_bytes(),
        &info.payload_bytes.to_le_bytes(),
        info.schema_id.as_bytes(),
        info.source_model_id.as_bytes(),
        &info.source_tensor_digest,
        &name_len.to_le_bytes(),
        &rank.to_le_bytes(),
        &0_u16.to_le_bytes(),
        &metadata_len.to_le_bytes(),
        info.name.as_bytes(),
    ] {
        sink.write_all(bytes)
            .map_err(|error| work_io("write tensor record header", error))?;
    }
    for &dimension in &info.shape {
        sink.write_all(&dimension.to_le_bytes())
            .map_err(|error| work_io("write tensor record shape", error))?;
    }
    sink.write_all(&info.schema_metadata)
        .map_err(|error| work_io("write tensor schema metadata", error))
}

fn read_record_prefix(reader: &mut HashingReader<'_>) -> Result<RecordPrefix, TensorWorkError> {
    if reader.array::<8>()? != RECORD_MAGIC {
        return Err(TensorWorkError::BadMagic);
    }
    let version = reader.u16()?;
    if version != RECORD_VERSION {
        return Err(TensorWorkError::UnsupportedVersion(version));
    }
    if reader.u16()? != 0 {
        return Err(TensorWorkError::NonzeroReserved);
    }
    let total_bytes = reader.u64()?;
    let payload_bytes = reader.u64()?;
    let schema_id = ContentId::from_digest(reader.digest()?);
    let source_model_id = ModelId::from_digest(reader.digest()?);
    let source_tensor_digest = reader.digest()?;
    let name_len = reader.u32()? as usize;
    let rank = reader.u16()? as usize;
    if reader.u16()? != 0 {
        return Err(TensorWorkError::NonzeroReserved);
    }
    let metadata_len = reader.u32()? as usize;
    if name_len == 0 || name_len > MAX_NAME_BYTES {
        return Err(TensorWorkError::InvalidSpec("name length"));
    }
    if rank == 0 || rank > MAX_RANK {
        return Err(TensorWorkError::InvalidSpec("rank"));
    }
    if metadata_len > MAX_SCHEMA_METADATA_BYTES {
        return Err(TensorWorkError::InvalidSpec("schema metadata length"));
    }
    let name = String::from_utf8(reader.bounded_vec(name_len)?)
        .map_err(|_| TensorWorkError::InvalidSpec("name UTF-8"))?;
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(rank)
        .map_err(|_| TensorWorkError::AllocationFailed)?;
    for _ in 0..rank {
        shape.push(reader.u64()?);
    }
    let schema_metadata = reader.bounded_vec(metadata_len)?;
    let info = TensorRecordInfo {
        schema_id,
        source_model_id,
        source_tensor_digest,
        name,
        shape,
        schema_metadata,
        payload_bytes,
    };
    validate_info(&info)?;
    if total_bytes != exact_record_bytes(&info)? {
        return Err(TensorWorkError::RecordLengthMismatch {
            expected: exact_record_bytes(&info)?,
            actual: total_bytes,
        });
    }
    Ok(RecordPrefix {
        total_bytes,
        payload_offset: reader.position,
        info,
    })
}

fn open_and_verify(
    path: &Path,
    expected: Option<&TensorRecordReceipt>,
) -> Result<TensorRecordReader, TensorWorkError> {
    let metadata_before =
        fs::symlink_metadata(path).map_err(|error| work_io("inspect tensor record", error))?;
    if metadata_before.file_type().is_symlink() || !metadata_before.is_file() {
        return Err(TensorWorkError::InvalidPath("record file"));
    }
    let mut file = File::open(path).map_err(|error| work_io("open tensor record", error))?;
    let opened_metadata = file
        .metadata()
        .map_err(|error| work_io("inspect opened tensor record", error))?;
    let metadata_after = fs::symlink_metadata(path)
        .map_err(|error| work_io("reinspect tensor record path", error))?;
    if metadata_after.file_type().is_symlink() || !metadata_after.is_file() {
        return Err(TensorWorkError::InvalidPath("record file"));
    }
    if !same_file_identity(&metadata_before, &opened_metadata)
        || !same_file_identity(&opened_metadata, &metadata_after)
    {
        return Err(TensorWorkError::InvalidPath(
            "record path changed during open",
        ));
    }
    let opened_bytes = opened_metadata.len();
    if opened_bytes != metadata_before.len() || opened_bytes != metadata_after.len() {
        return Err(TensorWorkError::RecordLengthMismatch {
            expected: metadata_before.len(),
            actual: opened_bytes,
        });
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| work_io("seek tensor record", error))?;
    let mut reader = HashingReader::new(&mut file);
    let prefix = read_record_prefix(&mut reader)?;
    if prefix.total_bytes != opened_bytes {
        return Err(TensorWorkError::RecordLengthMismatch {
            expected: prefix.total_bytes,
            actual: opened_bytes,
        });
    }
    let mut staging = [0_u8; MAX_STAGING_BYTES];
    let mut remaining = prefix.info.payload_bytes;
    let mut payload_hasher = blake3::Hasher::new_derive_key(PAYLOAD_DIGEST_CONTEXT);
    while remaining != 0 {
        let count = usize::try_from(remaining.min(MAX_STAGING_BYTES as u64))
            .map_err(|_| TensorWorkError::LengthOverflow)?;
        reader.read_exact(&mut staging[..count])?;
        payload_hasher.update(&staging[..count]);
        remaining -= count as u64;
    }
    let footer = reader.digest()?;
    let payload_digest = *payload_hasher.finalize().as_bytes();
    if footer != payload_digest {
        return Err(TensorWorkError::PayloadDigestMismatch);
    }
    if reader.position != prefix.total_bytes {
        return Err(TensorWorkError::RecordLengthMismatch {
            expected: prefix.total_bytes,
            actual: reader.position,
        });
    }
    let record_id = reader.finish();
    let receipt = TensorRecordReceipt {
        record_id,
        record_bytes: prefix.total_bytes,
        payload_digest,
        info: prefix.info,
    };
    if let Some(expected) = expected {
        if receipt.record_id != expected.record_id {
            return Err(TensorWorkError::RecordIdMismatch);
        }
        if &receipt != expected {
            return Err(TensorWorkError::ReceiptMismatch);
        }
    }
    Ok(TensorRecordReader {
        file,
        receipt,
        payload_offset: prefix.payload_offset,
    })
}

fn validate_info(info: &TensorRecordInfo) -> Result<(), TensorWorkError> {
    if info.name.is_empty() || info.name.len() > MAX_NAME_BYTES {
        return Err(TensorWorkError::InvalidSpec("name"));
    }
    if info.shape.is_empty() || info.shape.len() > MAX_RANK {
        return Err(TensorWorkError::InvalidSpec("shape rank"));
    }
    if info.shape.contains(&0) {
        return Err(TensorWorkError::InvalidSpec("zero shape dimension"));
    }
    if info.schema_metadata.len() > MAX_SCHEMA_METADATA_BYTES {
        return Err(TensorWorkError::InvalidSpec("schema metadata"));
    }
    if info.payload_bytes == 0 {
        return Err(TensorWorkError::InvalidSpec("zero payload"));
    }
    u32::try_from(info.name.len()).map_err(|_| TensorWorkError::LengthOverflow)?;
    u16::try_from(info.shape.len()).map_err(|_| TensorWorkError::LengthOverflow)?;
    u32::try_from(info.schema_metadata.len()).map_err(|_| TensorWorkError::LengthOverflow)?;
    Ok(())
}

fn exact_record_bytes(info: &TensorRecordInfo) -> Result<u64, TensorWorkError> {
    let shape_bytes = u64::try_from(info.shape.len())
        .map_err(|_| TensorWorkError::LengthOverflow)?
        .checked_mul(8)
        .ok_or(TensorWorkError::LengthOverflow)?;
    [
        RECORD_FIXED_BYTES,
        u64::try_from(info.name.len()).map_err(|_| TensorWorkError::LengthOverflow)?,
        shape_bytes,
        u64::try_from(info.schema_metadata.len()).map_err(|_| TensorWorkError::LengthOverflow)?,
        info.payload_bytes,
        RECORD_FOOTER_BYTES,
    ]
    .into_iter()
    .try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or(TensorWorkError::LengthOverflow)
    })
}

fn encode_info(output: &mut Vec<u8>, info: &TensorRecordInfo) -> Result<(), TensorWorkError> {
    let name_len = u32::try_from(info.name.len()).map_err(|_| TensorWorkError::LengthOverflow)?;
    let rank = u16::try_from(info.shape.len()).map_err(|_| TensorWorkError::LengthOverflow)?;
    let metadata_len =
        u32::try_from(info.schema_metadata.len()).map_err(|_| TensorWorkError::LengthOverflow)?;
    output.extend_from_slice(info.schema_id.as_bytes());
    output.extend_from_slice(info.source_model_id.as_bytes());
    output.extend_from_slice(&info.source_tensor_digest);
    output.extend_from_slice(&info.payload_bytes.to_le_bytes());
    output.extend_from_slice(&name_len.to_le_bytes());
    output.extend_from_slice(&rank.to_le_bytes());
    output.extend_from_slice(&0_u16.to_le_bytes());
    output.extend_from_slice(&metadata_len.to_le_bytes());
    output.extend_from_slice(info.name.as_bytes());
    for dimension in &info.shape {
        output.extend_from_slice(&dimension.to_le_bytes());
    }
    output.extend_from_slice(&info.schema_metadata);
    Ok(())
}

fn decode_info(cursor: &mut ReceiptCursor<'_>) -> Result<TensorRecordInfo, TensorWorkError> {
    let schema_id = ContentId::from_digest(cursor.digest()?);
    let source_model_id = ModelId::from_digest(cursor.digest()?);
    let source_tensor_digest = cursor.digest()?;
    let payload_bytes = cursor.u64()?;
    let name_len = cursor.u32()? as usize;
    let rank = cursor.u16()? as usize;
    if cursor.u16()? != 0 {
        return Err(TensorWorkError::InvalidReceipt("reserved bits"));
    }
    let metadata_len = cursor.u32()? as usize;
    if name_len == 0 || name_len > MAX_NAME_BYTES {
        return Err(TensorWorkError::InvalidReceipt("name length"));
    }
    if rank == 0 || rank > MAX_RANK {
        return Err(TensorWorkError::InvalidReceipt("rank"));
    }
    if metadata_len > MAX_SCHEMA_METADATA_BYTES {
        return Err(TensorWorkError::InvalidReceipt("metadata length"));
    }
    let name = std::str::from_utf8(cursor.take(name_len)?)
        .map_err(|_| TensorWorkError::InvalidReceipt("name UTF-8"))?
        .to_owned();
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(rank)
        .map_err(|_| TensorWorkError::AllocationFailed)?;
    for _ in 0..rank {
        shape.push(cursor.u64()?);
    }
    let schema_metadata = cursor.take(metadata_len)?.to_vec();
    Ok(TensorRecordInfo {
        schema_id,
        source_model_id,
        source_tensor_digest,
        name,
        shape,
        schema_metadata,
        payload_bytes,
    })
}

#[derive(Debug)]
struct ReceiptCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ReceiptCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], TensorWorkError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(TensorWorkError::LengthOverflow)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(TensorWorkError::InvalidReceipt("truncated"))?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, TensorWorkError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, TensorWorkError> {
        let mut bytes = [0; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, TensorWorkError> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, TensorWorkError> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn digest(&mut self) -> Result<[u8; 32], TensorWorkError> {
        let mut digest = [0; 32];
        digest.copy_from_slice(self.take(32)?);
        Ok(digest)
    }
}

pub(crate) fn absolute_path(path: &Path) -> Result<PathBuf, TensorWorkError> {
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(TensorWorkError::InvalidPath("store root traversal"));
    }
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        std::env::current_dir()
            .map(|current| current.join(path))
            .map_err(|error| work_io("resolve tensor work root", error))
    }
}

pub(crate) fn ensure_durable_directory(
    path: &Path,
    field: &'static str,
) -> Result<(), TensorWorkError> {
    if path.as_os_str().is_empty() {
        return Err(TensorWorkError::InvalidPath(field));
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        ensure_durable_directory(parent, "directory ancestor")?;
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(TensorWorkError::InvalidPath(field));
            }
            return Ok(());
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(work_io("inspect tensor work directory", error)),
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or(TensorWorkError::InvalidPath("directory parent"))?;
    match fs::create_dir(path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(work_io("create tensor work directory", error)),
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| work_io("inspect created tensor work directory", error))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(TensorWorkError::InvalidPath(field));
    }
    sync_directory(parent, "sync tensor work parent directory")
}

pub(crate) fn create_temporary_file(
    directory: &Path,
    prefix: &str,
) -> Result<(PathBuf, File), TensorWorkError> {
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    for _ in 0..128 {
        let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(
            "{prefix}.{}.{}.{}",
            std::process::id(),
            epoch,
            nonce
        ));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(work_io("create temporary record", error)),
        }
    }
    Err(TensorWorkError::Io {
        operation: "create unique temporary record",
        kind: io::ErrorKind::AlreadyExists,
    })
}

#[cfg(unix)]
fn same_file_identity(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file_identity(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.is_file() == right.is_file() && left.len() == right.len()
}

#[cfg(unix)]
fn sync_directory(path: &Path, operation: &'static str) -> Result<(), TensorWorkError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| work_io(operation, error))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path, _operation: &'static str) -> Result<(), TensorWorkError> {
    Ok(())
}

fn work_io(operation: &'static str, error: io::Error) -> TensorWorkError {
    TensorWorkError::Io {
        operation,
        kind: error.kind(),
    }
}
