//! Seek-backed, bounded-memory reader for model-safe TSLB SALT bundles.
//!
//! Construction performs one strict streaming validation pass over every row. It
//! retains only owned tensor metadata, exact packed-storage requirements, and a
//! payload digest. Named reads then reuse one row-sized buffer and expose borrowing
//! row views, allowing runtimes to fill final arenas without whole-file or per-row
//! owned intermediates.

use core::fmt;
use std::io::{ErrorKind, Read, Seek, SeekFrom};

use crate::{
    FormatError, PackedSaltRowRef, SALT_BUNDLE_MAGIC, SALT_BUNDLE_VERSION, SALT_HEADER_BYTES,
    SALT_PROGRESSIVE_VERSION, SALT_VERSION, TQ2_0_BLOCK_BYTES, num_blocks,
};

const MAX_TENSORS: u64 = 1_000_000;
const MAX_INDEX_BYTES: u64 = 100_000_000;
const MAX_TOTAL_NAME_BYTES: u64 = 100_000_000;
const MAX_TOTAL_ROWS: u64 = 16_000_000;
const MAX_TOTAL_PLANES: u64 = 64_000_000;
const MAX_PLANES_PER_ROW: u64 = 8;
const MAX_ENCODED_ROW_BYTES: usize = 16 * 1024 * 1024;
const IO_CHUNK_BYTES: usize = 64 * 1024;
const MIN_INDEX_ENTRY_BYTES: u64 = 2 + 4 + 4 + 8;
const PLANE_DESCRIPTOR_BYTES: usize = 5;

/// Errors from seek-backed TSLB indexing and row streaming.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltBundleReadError {
    /// Encoded SALT data violated the canonical format.
    Format(FormatError),
    /// Source seek or read failed.
    Io {
        /// Operation being attempted.
        context: String,
        /// Portable I/O error classification.
        kind: ErrorKind,
        /// Original error text.
        message: String,
    },
    /// A bounded metadata or row allocation failed.
    AllocationFailed {
        /// Bytes requested by the failed reservation.
        requested_bytes: usize,
    },
    /// An explicit parser resource limit was exceeded.
    LimitExceeded {
        /// Limited resource.
        resource: String,
        /// Maximum accepted value.
        limit: u64,
        /// Value declared or observed in the source.
        actual: u64,
    },
    /// Requested tensor name is absent.
    TensorNotFound(String),
    /// Tensor payload changed after strict construction-time validation.
    SourceChanged(String),
}

impl fmt::Display for SaltBundleReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(error) => write!(f, "SALT bundle: {error}"),
            Self::Io {
                context, message, ..
            } => write!(f, "SALT bundle {context}: {message}"),
            Self::AllocationFailed { requested_bytes } => {
                write!(
                    f,
                    "SALT bundle allocation of {requested_bytes} bytes failed"
                )
            }
            Self::LimitExceeded {
                resource,
                limit,
                actual,
            } => write!(f, "SALT bundle {resource} {actual} exceeds limit {limit}"),
            Self::TensorNotFound(name) => write!(f, "SALT bundle tensor `{name}` not found"),
            Self::SourceChanged(name) => {
                write!(f, "SALT bundle tensor `{name}` changed after validation")
            }
        }
    }
}

impl std::error::Error for SaltBundleReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FormatError> for SaltBundleReadError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

/// Exact final-arena element counts discovered during strict validation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackedSaltStorageRequirements {
    rows: usize,
    planes: usize,
    dense_bytes: usize,
    sparse_scales: usize,
    sparse_entries: usize,
    sparse_planes: usize,
}

impl PackedSaltStorageRequirements {
    /// Row-metadata elements required.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Plane-metadata elements required.
    #[must_use]
    pub const fn planes(self) -> usize {
        self.planes
    }

    /// Dense TQ2_0 payload bytes required.
    #[must_use]
    pub const fn dense_bytes(self) -> usize {
        self.dense_bytes
    }

    /// Sparse f16 scale elements required.
    #[must_use]
    pub const fn sparse_scales(self) -> usize {
        self.sparse_scales
    }

    /// Sparse signed-index elements required.
    #[must_use]
    pub const fn sparse_entries(self) -> usize {
        self.sparse_entries
    }

    /// Sparse plane count.
    #[must_use]
    pub const fn sparse_planes(self) -> usize {
        self.sparse_planes
    }

    pub(crate) fn try_add_row(&mut self, row: PackedSaltRowRef<'_>) -> Option<()> {
        let rows = self.rows.checked_add(1)?;
        let planes = self.planes.checked_add(row.plane_count())?;
        let mut dense_bytes = self.dense_bytes;
        let mut sparse_scales = self.sparse_scales;
        let mut sparse_entries = self.sparse_entries;
        let mut sparse_planes = self.sparse_planes;
        for plane in row.planes() {
            if let Some(bytes) = plane.dense_bytes() {
                dense_bytes = dense_bytes.checked_add(bytes.len())?;
            } else if let Some(sparse) = plane.sparse() {
                sparse_scales = sparse_scales.checked_add(sparse.scale_count())?;
                sparse_entries = sparse_entries.checked_add(sparse.entry_count())?;
                sparse_planes = sparse_planes.checked_add(1)?;
            }
        }
        self.rows = rows;
        self.planes = planes;
        self.dense_bytes = dense_bytes;
        self.sparse_scales = sparse_scales;
        self.sparse_entries = sparse_entries;
        self.sparse_planes = sparse_planes;
        Some(())
    }

    fn add_row(&mut self, row: PackedSaltRowRef<'_>) -> Result<(), SaltBundleReadError> {
        self.try_add_row(row)
            .ok_or_else(|| limit_error("packed storage requirements", u64::MAX, usize::MAX as u64))
    }
}

/// Indexed metadata for one tensor in a [`SaltBundleReader`].
#[derive(Clone, Debug)]
pub struct SaltBundleTensorInfo {
    rows: usize,
    k: usize,
    encoded_len: u64,
    data_offset: u64,
    requirements: PackedSaltStorageRequirements,
    digest: [u8; 32],
}

#[derive(Debug)]
struct IndexedTensor {
    name: String,
    info: SaltBundleTensorInfo,
}

impl SaltBundleTensorInfo {
    /// Matrix shape `(rows, k)`.
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.k)
    }

    /// Encoded payload length in bytes.
    #[must_use]
    pub const fn encoded_len(&self) -> u64 {
        self.encoded_len
    }

    /// Exact final packed-storage requirements.
    #[must_use]
    pub const fn storage_requirements(&self) -> PackedSaltStorageRequirements {
        self.requirements
    }
}

/// Strict seek-backed reader for model artifacts in canonical TSLB framing.
///
/// `new_strict` reads every payload once for validation but retains no payload bytes.
/// Total I/O is therefore not lazy; anonymous memory is bounded by metadata plus one
/// encoded row. Named tensor visits use absolute seeks and support arbitrary order.
/// Model-safety limits intentionally reject more than 8 planes per row or encoded rows
/// above 16 MiB even though the low-level TSLB framing can represent them;
/// [`SaltBundleReadError::LimitExceeded`] reports the precise policy boundary.
#[derive(Debug)]
pub struct SaltBundleReader<R> {
    source: Source<R>,
    tensors: Vec<IndexedTensor>,
}

impl<R: Read + Seek> SaltBundleReader<R> {
    /// Parse and strictly validate an entire TSLB source with bounded staging.
    ///
    /// # Errors
    /// Returns typed format, I/O, allocation, and resource-limit errors. Duplicate
    /// names, malformed unselected rows, truncation, and trailing bytes fail here.
    pub fn new_strict(reader: R) -> Result<Self, SaltBundleReadError> {
        let mut source = Source::new(reader)?;
        if source.array::<4>("read magic")? != SALT_BUNDLE_MAGIC {
            return Err(FormatError::SaltBadMagic.into());
        }
        let version = source.u8("read version")?;
        if version != SALT_BUNDLE_VERSION {
            return Err(FormatError::UnsupportedSaltVersion(version).into());
        }
        let _reserved = source.u8("read reserved byte")?;
        let tensor_count = u64::from(source.u32("read tensor count")?);
        enforce_limit("tensor count", tensor_count, MAX_TENSORS)?;
        let minimum_index_end = tensor_count
            .checked_mul(MIN_INDEX_ENTRY_BYTES)
            .and_then(|bytes| source.position().checked_add(bytes))
            .ok_or_else(|| limit_error("index bytes", u64::MAX, MAX_INDEX_BYTES))?;
        enforce_limit("index bytes", minimum_index_end, MAX_INDEX_BYTES)?;
        if minimum_index_end > source.len() {
            return Err(FormatError::WrongBlockLen {
                expected: usize::try_from(minimum_index_end).unwrap_or(usize::MAX),
                got: usize::try_from(source.len()).unwrap_or(usize::MAX),
            }
            .into());
        }

        let tensor_count_usize =
            usize::try_from(tensor_count).map_err(|_| SaltBundleReadError::LimitExceeded {
                resource: "tensor count".to_owned(),
                limit: usize::MAX as u64,
                actual: tensor_count,
            })?;
        let mut entries = Vec::new();
        entries.try_reserve_exact(tensor_count_usize).map_err(|_| {
            SaltBundleReadError::AllocationFailed {
                requested_bytes: tensor_count_usize.saturating_mul(size_of::<IndexedTensor>()),
            }
        })?;
        let mut total_name_bytes = 0u64;
        let mut total_rows = 0u64;
        for _ in 0..tensor_count_usize {
            let name_len = usize::from(source.u16("read tensor name length")?);
            total_name_bytes = total_name_bytes
                .checked_add(name_len as u64)
                .ok_or_else(|| {
                    limit_error("total tensor-name bytes", u64::MAX, MAX_TOTAL_NAME_BYTES)
                })?;
            enforce_limit(
                "total tensor-name bytes",
                total_name_bytes,
                MAX_TOTAL_NAME_BYTES,
            )?;
            let name = source.string(name_len, "read tensor name")?;
            let rows_u32 = source.u32("read tensor rows")?;
            let rows = rows_u32 as usize;
            total_rows = total_rows
                .checked_add(u64::from(rows_u32))
                .ok_or_else(|| limit_error("total row count", u64::MAX, MAX_TOTAL_ROWS))?;
            enforce_limit("total row count", total_rows, MAX_TOTAL_ROWS)?;
            let k = source.u32("read tensor width")? as usize;
            let data_len = source.u64("read tensor payload length")?;
            let minimum = u64::from(rows_u32)
                .checked_mul(SALT_HEADER_BYTES as u64)
                .ok_or_else(|| limit_error("tensor row framing", u64::MAX, data_len))?;
            if data_len < minimum {
                return Err(FormatError::WrongBlockLen {
                    expected: usize::try_from(minimum).unwrap_or(usize::MAX),
                    got: usize::try_from(data_len).unwrap_or(usize::MAX),
                }
                .into());
            }
            enforce_limit("index bytes", source.position(), MAX_INDEX_BYTES)?;
            entries.push(IndexedTensor {
                name,
                info: SaltBundleTensorInfo {
                    rows,
                    k,
                    encoded_len: data_len,
                    data_offset: 0,
                    requirements: PackedSaltStorageRequirements::default(),
                    digest: [0; 32],
                },
            });
        }

        let mut payload_offset = source.position();
        for entry in &mut entries {
            let payload_end = payload_offset
                .checked_add(entry.info.encoded_len)
                .ok_or(FormatError::SaltLengthOverflow(entry.info.encoded_len))?;
            if payload_end > source.len() {
                return Err(FormatError::WrongBlockLen {
                    expected: usize::try_from(payload_end).unwrap_or(usize::MAX),
                    got: usize::try_from(source.len()).unwrap_or(usize::MAX),
                }
                .into());
            }
            entry.info.data_offset = payload_offset;
            payload_offset = payload_end;
        }
        if payload_offset != source.len() {
            return Err(FormatError::WrongBlockLen {
                expected: usize::try_from(payload_offset).unwrap_or(usize::MAX),
                got: usize::try_from(source.len()).unwrap_or(usize::MAX),
            }
            .into());
        }

        entries.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        if let Some(pair) = entries.windows(2).find(|pair| pair[0].name == pair[1].name) {
            return Err(FormatError::SaltDuplicateTensor(pair[0].name.clone()).into());
        }

        let mut scratch = Vec::new();
        let mut total_planes = 0u64;
        for entry in &mut entries {
            let payload_end = entry
                .info
                .data_offset
                .checked_add(entry.info.encoded_len)
                .ok_or(FormatError::SaltLengthOverflow(entry.info.encoded_len))?;
            source.seek_abs(entry.info.data_offset, "seek tensor payload")?;
            let mut requirements = PackedSaltStorageRequirements::default();
            let mut digest = blake3::Hasher::new();
            for _ in 0..entry.info.rows {
                let row = read_row(&mut source, payload_end, &mut scratch)?;
                if row.k() != entry.info.k {
                    return Err(FormatError::WrongBlockLen {
                        expected: entry.info.k,
                        got: row.k(),
                    }
                    .into());
                }
                total_planes = total_planes
                    .checked_add(row.plane_count() as u64)
                    .ok_or_else(|| limit_error("total plane count", u64::MAX, MAX_TOTAL_PLANES))?;
                enforce_limit("total plane count", total_planes, MAX_TOTAL_PLANES)?;
                requirements.add_row(row)?;
                digest.update(row.encoded_bytes());
            }
            if source.position() != payload_end {
                return Err(FormatError::WrongBlockLen {
                    expected: usize::try_from(entry.info.encoded_len).unwrap_or(usize::MAX),
                    got: usize::try_from(source.position().saturating_sub(entry.info.data_offset))
                        .unwrap_or(usize::MAX),
                }
                .into());
            }
            entry.info.requirements = requirements;
            entry.info.digest = *digest.finalize().as_bytes();
        }
        Ok(Self {
            source,
            tensors: entries,
        })
    }

    /// Number of indexed tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the bundle contains no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Tensor names in lexical order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|entry| entry.name.as_str())
    }

    /// Metadata and exact arena requirements for a named tensor.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&SaltBundleTensorInfo> {
        self.find_tensor(name).map(|entry| &entry.info)
    }

    /// Visit one tensor's validated rows using one reusable encoded-row buffer.
    ///
    /// A complete visit verifies row geometry, exact payload length, and the
    /// construction-time payload digest. The visitor may be called before a late I/O or
    /// digest error is known, and this method cannot roll callback side effects back.
    /// Callers requiring transactional mutation must stage their sink and publish it only
    /// after this method returns `Ok(())`.
    ///
    /// # Errors
    /// Returns a typed error for a missing tensor, I/O failure, malformed row, or
    /// same-handle source mutation after strict construction.
    pub fn visit_packed_tensor(
        &mut self,
        name: &str,
        mut visitor: impl FnMut(PackedSaltRowRef<'_>),
    ) -> Result<(), SaltBundleReadError> {
        let info = self
            .find_tensor(name)
            .map(|entry| entry.info.clone())
            .ok_or_else(|| SaltBundleReadError::TensorNotFound(name.to_owned()))?;
        let payload_end = info
            .data_offset
            .checked_add(info.encoded_len)
            .ok_or(FormatError::SaltLengthOverflow(info.encoded_len))?;
        self.source
            .seek_abs(info.data_offset, "seek selected tensor")?;
        let mut scratch = Vec::new();
        let mut digest = blake3::Hasher::new();
        for _ in 0..info.rows {
            let row = read_row(&mut self.source, payload_end, &mut scratch)?;
            if row.k() != info.k {
                return Err(FormatError::WrongBlockLen {
                    expected: info.k,
                    got: row.k(),
                }
                .into());
            }
            digest.update(row.encoded_bytes());
            visitor(row);
        }
        if self.source.position() != payload_end || *digest.finalize().as_bytes() != info.digest {
            return Err(SaltBundleReadError::SourceChanged(name.to_owned()));
        }
        Ok(())
    }

    /// Recover the underlying reader after validation/use.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.source.inner
    }

    fn find_tensor(&self, name: &str) -> Option<&IndexedTensor> {
        self.tensors
            .binary_search_by(|entry| entry.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.tensors[index])
    }
}

fn enforce_limit(resource: &str, actual: u64, limit: u64) -> Result<(), SaltBundleReadError> {
    if actual > limit {
        Err(limit_error(resource, actual, limit))
    } else {
        Ok(())
    }
}

fn limit_error(resource: &str, actual: u64, limit: u64) -> SaltBundleReadError {
    SaltBundleReadError::LimitExceeded {
        resource: resource.to_owned(),
        limit,
        actual,
    }
}

fn read_row<'a, R: Read + Seek>(
    source: &mut Source<R>,
    tensor_end: u64,
    scratch: &'a mut Vec<u8>,
) -> Result<PackedSaltRowRef<'a>, SaltBundleReadError> {
    let row_start = source.position();
    let header_end = row_start
        .checked_add(SALT_HEADER_BYTES as u64)
        .ok_or(FormatError::SaltLengthOverflow(u64::MAX))?;
    if header_end > tensor_end {
        return Err(FormatError::WrongBlockLen {
            expected: SALT_HEADER_BYTES,
            got: usize::try_from(tensor_end.saturating_sub(row_start)).unwrap_or(usize::MAX),
        }
        .into());
    }
    let header = source.array::<SALT_HEADER_BYTES>("read SALT row header")?;
    if header[0..4] != crate::SALT_MAGIC {
        return Err(FormatError::SaltBadMagic.into());
    }
    let version = header[4];
    let planes = header[5] as usize;
    enforce_limit("planes per row", planes as u64, MAX_PLANES_PER_ROW)?;
    let k = u32::from_le_bytes(header[6..10].try_into().expect("ten-byte SALT header")) as usize;
    let plane_bytes =
        num_blocks(k)
            .checked_mul(TQ2_0_BLOCK_BYTES)
            .ok_or(FormatError::WrongBlockLen {
                expected: usize::MAX,
                got: k,
            })?;

    scratch.clear();
    scratch.try_reserve_exact(SALT_HEADER_BYTES).map_err(|_| {
        SaltBundleReadError::AllocationFailed {
            requested_bytes: SALT_HEADER_BYTES,
        }
    })?;
    scratch.extend_from_slice(&header);
    let encoded_len = match version {
        SALT_VERSION => SALT_HEADER_BYTES
            .checked_add(
                planes
                    .checked_mul(plane_bytes)
                    .ok_or(FormatError::WrongBlockLen {
                        expected: usize::MAX,
                        got: planes,
                    })?,
            )
            .ok_or(FormatError::WrongBlockLen {
                expected: usize::MAX,
                got: planes,
            })?,
        SALT_PROGRESSIVE_VERSION => {
            let descriptor_bytes =
                planes
                    .checked_mul(PLANE_DESCRIPTOR_BYTES)
                    .ok_or(FormatError::WrongBlockLen {
                        expected: usize::MAX,
                        got: planes,
                    })?;
            let prefix_len = SALT_HEADER_BYTES.checked_add(descriptor_bytes).ok_or(
                FormatError::WrongBlockLen {
                    expected: usize::MAX,
                    got: descriptor_bytes,
                },
            )?;
            reserve_and_resize(scratch, prefix_len)?;
            source.read_exact_chunks(
                &mut scratch[SALT_HEADER_BYTES..prefix_len],
                "read SALT plane descriptors",
            )?;
            let mut payload_bytes = 0usize;
            for descriptor in scratch[SALT_HEADER_BYTES..prefix_len]
                .as_chunks::<PLANE_DESCRIPTOR_BYTES>()
                .0
            {
                let len = u32::from_le_bytes(
                    descriptor[1..]
                        .try_into()
                        .expect("five-byte plane descriptor"),
                ) as usize;
                payload_bytes =
                    payload_bytes
                        .checked_add(len)
                        .ok_or(FormatError::WrongBlockLen {
                            expected: usize::MAX,
                            got: len,
                        })?;
            }
            prefix_len
                .checked_add(payload_bytes)
                .ok_or(FormatError::WrongBlockLen {
                    expected: usize::MAX,
                    got: payload_bytes,
                })?
        }
        other => return Err(FormatError::UnsupportedSaltVersion(other).into()),
    };
    enforce_limit(
        "encoded row bytes",
        encoded_len as u64,
        MAX_ENCODED_ROW_BYTES as u64,
    )?;
    let row_end = row_start
        .checked_add(encoded_len as u64)
        .ok_or(FormatError::SaltLengthOverflow(encoded_len as u64))?;
    if row_end > tensor_end {
        return Err(FormatError::WrongBlockLen {
            expected: encoded_len,
            got: usize::try_from(tensor_end.saturating_sub(row_start)).unwrap_or(usize::MAX),
        }
        .into());
    }
    let already_read = scratch.len();
    reserve_and_resize(scratch, encoded_len)?;
    source.read_exact_chunks(&mut scratch[already_read..], "read SALT row payload")?;
    Ok(PackedSaltRowRef::parse(scratch)?)
}

fn reserve_and_resize(buffer: &mut Vec<u8>, len: usize) -> Result<(), SaltBundleReadError> {
    if len > buffer.len() {
        buffer.try_reserve_exact(len - buffer.len()).map_err(|_| {
            SaltBundleReadError::AllocationFailed {
                requested_bytes: len,
            }
        })?;
        buffer.resize(len, 0);
    }
    Ok(())
}

#[derive(Debug)]
struct Source<R> {
    inner: R,
    len: u64,
    position: u64,
}

impl<R: Read + Seek> Source<R> {
    fn new(mut inner: R) -> Result<Self, SaltBundleReadError> {
        let len = inner
            .seek(SeekFrom::End(0))
            .map_err(|error| io_error("seek source end", error))?;
        let start = inner
            .seek(SeekFrom::Start(0))
            .map_err(|error| io_error("seek source start", error))?;
        if start != 0 {
            return Err(io_error(
                "seek source start",
                std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("seek returned position {start}, expected 0"),
                ),
            ));
        }
        Ok(Self {
            inner,
            len,
            position: 0,
        })
    }

    const fn len(&self) -> u64 {
        self.len
    }

    const fn position(&self) -> u64 {
        self.position
    }

    fn seek_abs(&mut self, position: u64, context: &str) -> Result<(), SaltBundleReadError> {
        if position > self.len {
            return Err(FormatError::WrongBlockLen {
                expected: usize::try_from(position).unwrap_or(usize::MAX),
                got: usize::try_from(self.len).unwrap_or(usize::MAX),
            }
            .into());
        }
        let actual = self
            .inner
            .seek(SeekFrom::Start(position))
            .map_err(|error| io_error(context, error))?;
        if actual != position {
            return Err(io_error(
                context,
                std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("seek returned position {actual}, expected {position}"),
                ),
            ));
        }
        self.position = actual;
        Ok(())
    }

    fn read_exact_chunks(
        &mut self,
        bytes: &mut [u8],
        context: &str,
    ) -> Result<(), SaltBundleReadError> {
        let end = self
            .position
            .checked_add(bytes.len() as u64)
            .ok_or(FormatError::SaltLengthOverflow(bytes.len() as u64))?;
        if end > self.len {
            return Err(FormatError::WrongBlockLen {
                expected: usize::try_from(end).unwrap_or(usize::MAX),
                got: usize::try_from(self.len).unwrap_or(usize::MAX),
            }
            .into());
        }
        for chunk in bytes.chunks_mut(IO_CHUNK_BYTES) {
            self.inner
                .read_exact(chunk)
                .map_err(|error| io_error(context, error))?;
            self.position += chunk.len() as u64;
        }
        Ok(())
    }

    fn array<const N: usize>(&mut self, context: &str) -> Result<[u8; N], SaltBundleReadError> {
        let mut bytes = [0u8; N];
        self.read_exact_chunks(&mut bytes, context)?;
        Ok(bytes)
    }

    fn u8(&mut self, context: &str) -> Result<u8, SaltBundleReadError> {
        Ok(self.array::<1>(context)?[0])
    }

    fn u16(&mut self, context: &str) -> Result<u16, SaltBundleReadError> {
        Ok(u16::from_le_bytes(self.array::<2>(context)?))
    }

    fn u32(&mut self, context: &str) -> Result<u32, SaltBundleReadError> {
        Ok(u32::from_le_bytes(self.array::<4>(context)?))
    }

    fn u64(&mut self, context: &str) -> Result<u64, SaltBundleReadError> {
        Ok(u64::from_le_bytes(self.array::<8>(context)?))
    }

    fn string(&mut self, len: usize, context: &str) -> Result<String, SaltBundleReadError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| SaltBundleReadError::AllocationFailed {
                requested_bytes: len,
            })?;
        bytes.resize(len, 0);
        self.read_exact_chunks(&mut bytes, context)?;
        String::from_utf8(bytes).map_err(|_| FormatError::SaltInvalidTensorName.into())
    }
}

fn io_error(context: &str, error: std::io::Error) -> SaltBundleReadError {
    SaltBundleReadError::Io {
        context: context.to_owned(),
        kind: error.kind(),
        message: error.to_string(),
    }
}

const fn size_of<T>() -> usize {
    core::mem::size_of::<T>()
}
