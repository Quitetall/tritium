//! Seek-backed, bounded-memory reader for legacy SALT-in-GGUF artifacts.
//!
//! The reader parses GGUF metadata and the tensor table directly from a `Read + Seek`
//! source, then strictly scans every private SALT tensor once. It retains only owned
//! tensor metadata, exact packed-storage requirements, and payload digests. Named
//! visits seek back to one tensor and expose borrowing row views through one reusable
//! row buffer.

use core::fmt;
use std::io::{ErrorKind, Read, Seek, SeekFrom};

use half::f16;

use crate::gguf::{MAX_METADATA_ARRAY_ELEMENTS, MAX_METADATA_DEPTH};
use crate::{
    DEFAULT_ALIGNMENT, FormatError, GGML_TYPE_TRITIUM_SALT, GgufError, PackedSaltRowRef,
    PackedSaltStorageRequirements, SALT_GGUF_FORMAT_KEY, SALT_GGUF_FORMAT_VALUE, SALT_HEADER_BYTES,
    SALT_PROGRESSIVE_VERSION, SALT_VERSION, TQ2_0_BLOCK_BYTES, num_blocks,
};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const MAX_TENSORS: u64 = 1_000_000;
const MAX_METADATA_ITEMS: u64 = 1_000_000;
const MAX_HEADER_BYTES: u64 = 100_000_000;
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TOTAL_NAME_BYTES: u64 = 100_000_000;
const MAX_TOTAL_ROWS: u64 = 16_000_000;
const MAX_TOTAL_PLANES: u64 = 64_000_000;
const MAX_PLANES_PER_ROW: u64 = 8;
const MAX_ENCODED_ROW_BYTES: usize = 16 * 1024 * 1024;
const MAX_DIMS: u32 = 8;
const IO_CHUNK_BYTES: usize = 64 * 1024;
const PLANE_DESCRIPTOR_BYTES: usize = 5;
const MIN_TENSOR_INFO_BYTES: u64 = 8 + 4 + 4 + 8;

/// Errors from strict seek-backed SALT-GGUF indexing and row streaming.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltGgufReadError {
    /// Encoded GGUF or SALT data violated the canonical format.
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
    /// Requested SALT tensor name is absent.
    TensorNotFound(String),
    /// Tensor payload changed after strict construction-time validation.
    SourceChanged(String),
    /// A dense or sparse plane used a negative-sign (including -0), NaN, or infinite f16 scale.
    InvalidScale(u16),
}

impl fmt::Display for SaltGgufReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(error) => write!(f, "SALT-GGUF: {error}"),
            Self::Io {
                context, message, ..
            } => write!(f, "SALT-GGUF {context}: {message}"),
            Self::AllocationFailed { requested_bytes } => {
                write!(f, "SALT-GGUF allocation of {requested_bytes} bytes failed")
            }
            Self::LimitExceeded {
                resource,
                limit,
                actual,
            } => write!(f, "SALT-GGUF {resource} {actual} exceeds limit {limit}"),
            Self::TensorNotFound(name) => write!(f, "SALT-GGUF tensor `{name}` not found"),
            Self::SourceChanged(name) => {
                write!(f, "SALT-GGUF tensor `{name}` changed after validation")
            }
            Self::InvalidScale(bits) => {
                write!(f, "SALT-GGUF contains invalid f16 scale {bits:#06x}")
            }
        }
    }
}

impl std::error::Error for SaltGgufReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FormatError> for SaltGgufReadError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

impl From<GgufError> for SaltGgufReadError {
    fn from(error: GgufError) -> Self {
        Self::Format(error.into())
    }
}

/// Indexed metadata for one private SALT tensor in a [`SaltGgufReader`].
#[derive(Clone, Debug)]
pub struct SaltGgufTensorInfo {
    rows: usize,
    k: usize,
    encoded_len: u64,
    data_offset: u64,
    requirements: PackedSaltStorageRequirements,
    digest: [u8; 32],
}

impl SaltGgufTensorInfo {
    /// Matrix shape `(rows, k)`.
    #[must_use]
    pub const fn shape(&self) -> (usize, usize) {
        (self.rows, self.k)
    }

    /// Exact encoded SALT payload length, excluding GGUF alignment padding.
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

#[derive(Debug)]
struct IndexedTensor {
    name: String,
    info: SaltGgufTensorInfo,
}

#[derive(Debug)]
struct TableTensor {
    name: String,
    relative_offset: u64,
    n_bytes: u64,
    rows: usize,
    k: usize,
    is_salt: bool,
    requirements: PackedSaltStorageRequirements,
    digest: [u8; 32],
}

/// Strict seek-backed reader for legacy SALT-in-GGUF artifacts.
///
/// Construction parses bounded metadata, validates the canonical tensor layout,
/// and reads every SALT row once without retaining payload bytes. Sized standard
/// tensors are layout-validated and ignored. Named visits use absolute seeks and
/// support arbitrary order.
#[derive(Debug)]
pub struct SaltGgufReader<R> {
    source: Source<R>,
    tensors: Vec<IndexedTensor>,
}

impl<R: Read + Seek> SaltGgufReader<R> {
    /// Parse and strictly validate a complete SALT-GGUF source.
    ///
    /// # Errors
    /// Returns typed format, I/O, allocation, and resource-limit errors. Duplicate
    /// names, malformed selected or unselected SALT rows, non-canonical offsets or
    /// padding, truncation, and trailing bytes fail during construction.
    pub fn new_strict(reader: R) -> Result<Self, SaltGgufReadError> {
        let mut source = Source::new(reader)?;
        parse_magic_and_version(&mut source)?;
        let tensor_count = source.u64("read tensor count")?;
        let metadata_count = source.u64("read metadata count")?;
        enforce_limit("tensor count", tensor_count, MAX_TENSORS)?;
        enforce_limit("metadata count", metadata_count, MAX_METADATA_ITEMS)?;

        let mut marker = None;
        let mut declared_alignment = None;
        let mut metadata_array_elements = 0u64;
        for _ in 0..metadata_count {
            let key = source.string("read metadata key")?;
            let value_type = source.u32("read metadata value type")?;
            if key == SALT_GGUF_FORMAT_KEY {
                marker = if value_type == 8 {
                    Some(source.string("read SALT-GGUF marker")?)
                } else {
                    skip_value(&mut source, value_type, 0, &mut metadata_array_elements)?;
                    None
                };
            } else if key == "general.alignment" {
                declared_alignment = Some(read_alignment_value(
                    &mut source,
                    value_type,
                    &mut metadata_array_elements,
                )?);
            } else {
                skip_value(&mut source, value_type, 0, &mut metadata_array_elements)?;
            }
            enforce_limit("header bytes", source.position(), MAX_HEADER_BYTES)?;
        }
        if marker.as_deref() != Some(SALT_GGUF_FORMAT_VALUE) {
            return Err(FormatError::SaltGgufBadFormat.into());
        }
        let alignment = match declared_alignment {
            None => DEFAULT_ALIGNMENT,
            Some(Some(value)) if value != 0 && value % 8 == 0 => value,
            Some(_) => return Err(GgufError::InvalidAlignment.into()),
        };

        let minimum_table_end = tensor_count
            .checked_mul(MIN_TENSOR_INFO_BYTES)
            .and_then(|bytes| source.position().checked_add(bytes))
            .ok_or(GgufError::DimsOverflow)?;
        enforce_limit("header bytes", minimum_table_end, MAX_HEADER_BYTES)?;
        if minimum_table_end > source.len() {
            return Err(GgufError::Truncated.into());
        }
        let tensor_metadata_bytes = tensor_count
            .checked_mul(size_of::<TableTensor>() as u64)
            .ok_or(GgufError::DimsOverflow)?;
        enforce_limit(
            "tensor metadata allocation bytes",
            tensor_metadata_bytes,
            MAX_HEADER_BYTES,
        )?;
        let tensor_count = usize::try_from(tensor_count)
            .map_err(|_| limit_error("tensor count", tensor_count, usize::MAX as u64))?;
        let mut table = Vec::new();
        try_reserve_exact(&mut table, tensor_count, size_of::<TableTensor>())?;
        let mut total_name_bytes = 0u64;
        let mut total_rows = 0u64;
        for _ in 0..tensor_count {
            let name = source.string("read tensor name")?;
            total_name_bytes =
                total_name_bytes
                    .checked_add(name.len() as u64)
                    .ok_or_else(|| {
                        limit_error("total tensor-name bytes", u64::MAX, MAX_TOTAL_NAME_BYTES)
                    })?;
            enforce_limit(
                "total tensor-name bytes",
                total_name_bytes,
                MAX_TOTAL_NAME_BYTES,
            )?;
            let dims_count = source.u32("read tensor rank")?;
            if dims_count > MAX_DIMS {
                return Err(GgufError::DimsOverflow.into());
            }
            let mut dims = Vec::new();
            try_reserve_exact(&mut dims, dims_count as usize, size_of::<u64>())?;
            let mut elements = 1u64;
            for _ in 0..dims_count {
                let dim = source.u64("read tensor dimension")?;
                elements = elements.checked_mul(dim).ok_or(GgufError::DimsOverflow)?;
                dims.push(dim);
            }
            let ggml_type = source.u32("read tensor type")?;
            let relative_offset = source.u64("read tensor offset")?;
            let is_salt = ggml_type == GGML_TYPE_TRITIUM_SALT;
            let (rows, k, n_bytes) = if is_salt {
                if dims.len() != 2 {
                    return Err(FormatError::SaltGgufBadFormat.into());
                }
                let k = usize::try_from(dims[0]).map_err(|_| FormatError::SaltGgufBadFormat)?;
                let rows = usize::try_from(dims[1]).map_err(|_| FormatError::SaltGgufBadFormat)?;
                total_rows = total_rows
                    .checked_add(dims[1])
                    .ok_or_else(|| limit_error("total row count", u64::MAX, MAX_TOTAL_ROWS))?;
                enforce_limit("total row count", total_rows, MAX_TOTAL_ROWS)?;
                (rows, k, 0)
            } else {
                let n_bytes = crate::gguf::tensor_n_bytes(ggml_type, &dims)?;
                if n_bytes == 0 {
                    return Err(FormatError::SaltGgufBadFormat.into());
                }
                (0, 0, n_bytes)
            };
            table.push(TableTensor {
                name,
                relative_offset,
                n_bytes,
                rows,
                k,
                is_salt,
                requirements: PackedSaltStorageRequirements::default(),
                digest: [0; 32],
            });
            enforce_limit("header bytes", source.position(), MAX_HEADER_BYTES)?;
        }

        reject_duplicate_names(&table)?;
        let header_end = source.position();
        let tensor_data_offset = align_up(header_end, alignment)?;
        enforce_limit("header bytes", tensor_data_offset, MAX_HEADER_BYTES)?;
        if tensor_data_offset > source.len() {
            return Err(GgufError::OffsetOutOfBounds.into());
        }
        source.read_zeroes_to(tensor_data_offset, "read GGUF header padding")?;
        let data_len = source.len() - tensor_data_offset;
        validate_offsets(&table, data_len, alignment)?;
        if table.is_empty() {
            if data_len != 0 {
                return Err(FormatError::SaltGgufBadFormat.into());
            }
            return Ok(Self {
                source,
                tensors: Vec::new(),
            });
        }

        let mut scratch = Vec::new();
        let mut total_planes = 0u64;
        for index in 0..table.len() {
            let start = tensor_data_offset
                .checked_add(table[index].relative_offset)
                .ok_or(GgufError::DimsOverflow)?;
            let relative_end = table
                .get(index + 1)
                .map_or(data_len, |tensor| tensor.relative_offset);
            let interval_end = tensor_data_offset
                .checked_add(relative_end)
                .ok_or(GgufError::DimsOverflow)?;
            source.seek_abs(start, "seek tensor payload")?;
            let used = if table[index].is_salt {
                let mut requirements = PackedSaltStorageRequirements::default();
                let mut digest = blake3::Hasher::new();
                for _ in 0..table[index].rows {
                    let row = read_row(&mut source, interval_end, &mut scratch)?;
                    if row.k() != table[index].k {
                        return Err(FormatError::WrongBlockLen {
                            expected: table[index].k,
                            got: row.k(),
                        }
                        .into());
                    }
                    validate_row_scales(row)?;
                    total_planes = total_planes
                        .checked_add(row.plane_count() as u64)
                        .ok_or_else(|| {
                            limit_error("total plane count", u64::MAX, MAX_TOTAL_PLANES)
                        })?;
                    enforce_limit("total plane count", total_planes, MAX_TOTAL_PLANES)?;
                    requirements.try_add_row(row).ok_or_else(|| {
                        limit_error("packed storage requirements", u64::MAX, usize::MAX as u64)
                    })?;
                    digest.update(row.encoded_bytes());
                }
                let used = source.position().saturating_sub(start);
                table[index].requirements = requirements;
                table[index].digest = *digest.finalize().as_bytes();
                table[index].n_bytes = used;
                used
            } else {
                let used = table[index].n_bytes;
                let payload_end = start.checked_add(used).ok_or(GgufError::DimsOverflow)?;
                if payload_end > interval_end {
                    return Err(GgufError::OffsetOutOfBounds.into());
                }
                source.seek_abs(payload_end, "seek sized tensor end")?;
                used
            };
            validate_payload_tail(&mut source, &table, index, used, interval_end, alignment)?;
        }

        let salt_count = table.iter().filter(|tensor| tensor.is_salt).count();
        let mut tensors = Vec::new();
        try_reserve_exact(&mut tensors, salt_count, size_of::<IndexedTensor>())?;
        for tensor in table.into_iter().filter(|tensor| tensor.is_salt) {
            tensors.push(IndexedTensor {
                name: tensor.name,
                info: SaltGgufTensorInfo {
                    rows: tensor.rows,
                    k: tensor.k,
                    encoded_len: tensor.n_bytes,
                    data_offset: tensor_data_offset + tensor.relative_offset,
                    requirements: tensor.requirements,
                    digest: tensor.digest,
                },
            });
        }
        tensors.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        Ok(Self { source, tensors })
    }

    /// Number of indexed private SALT tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the container has no private SALT tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Private SALT tensor names in lexical order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|entry| entry.name.as_str())
    }

    /// Metadata and exact arena requirements for a named private SALT tensor.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&SaltGgufTensorInfo> {
        self.find_tensor(name).map(|entry| &entry.info)
    }

    /// Visit one tensor's validated rows using one reusable encoded-row buffer.
    ///
    /// A successful visit rechecks row geometry, scales, exact payload length, and the
    /// construction-time digest. Callbacks can run before a late I/O or digest error;
    /// this method cannot roll their side effects back. Callers requiring transactional
    /// mutation must stage their sink and publish it only after `Ok(())`.
    ///
    /// # Errors
    /// Returns a typed error for a missing tensor, I/O failure, malformed row, or
    /// same-handle source mutation after strict construction.
    pub fn visit_packed_tensor(
        &mut self,
        name: &str,
        mut visitor: impl FnMut(PackedSaltRowRef<'_>),
    ) -> Result<(), SaltGgufReadError> {
        let info = self
            .find_tensor(name)
            .map(|entry| entry.info.clone())
            .ok_or_else(|| SaltGgufReadError::TensorNotFound(name.to_owned()))?;
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
            validate_row_scales(row)?;
            digest.update(row.encoded_bytes());
            visitor(row);
        }
        if self.source.position() != payload_end || *digest.finalize().as_bytes() != info.digest {
            return Err(SaltGgufReadError::SourceChanged(name.to_owned()));
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

fn validate_row_scales(row: PackedSaltRowRef<'_>) -> Result<(), SaltGgufReadError> {
    for plane in row.planes() {
        if let Some(bytes) = plane.dense_bytes() {
            for block in bytes.chunks_exact(TQ2_0_BLOCK_BYTES) {
                let bits = u16::from_le_bytes([
                    block[TQ2_0_BLOCK_BYTES - 2],
                    block[TQ2_0_BLOCK_BYTES - 1],
                ]);
                validate_scale(bits)?;
            }
        } else if let Some(sparse) = plane.sparse() {
            for scale in sparse.scales() {
                validate_scale(scale.to_bits())?;
            }
        }
    }
    Ok(())
}

fn validate_scale(bits: u16) -> Result<(), SaltGgufReadError> {
    let scale = f16::from_bits(bits);
    if !scale.is_finite() || scale.is_sign_negative() {
        Err(SaltGgufReadError::InvalidScale(bits))
    } else {
        Ok(())
    }
}

fn parse_magic_and_version<R: Read + Seek>(
    source: &mut Source<R>,
) -> Result<(), SaltGgufReadError> {
    if source.array::<4>("read GGUF magic")? != GGUF_MAGIC {
        return Err(GgufError::BadMagic.into());
    }
    let version = source.u32("read GGUF version")?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version).into());
    }
    Ok(())
}

fn read_alignment_value<R: Read + Seek>(
    source: &mut Source<R>,
    value_type: u32,
    metadata_array_elements: &mut u64,
) -> Result<Option<u64>, SaltGgufReadError> {
    if value_type == 4 {
        Ok(Some(u64::from(source.u32("read alignment")?)))
    } else {
        skip_value(source, value_type, 0, metadata_array_elements)?;
        Ok(None)
    }
}

fn skip_value<R: Read + Seek>(
    source: &mut Source<R>,
    value_type: u32,
    depth: u8,
    metadata_array_elements: &mut u64,
) -> Result<(), SaltGgufReadError> {
    match value_type {
        0 | 1 => {
            source.u8("read metadata scalar")?;
        }
        7 => {
            let value = source.u8("read metadata bool")?;
            if value > 1 {
                return Err(GgufError::InvalidBoolean(value).into());
            }
        }
        2 | 3 => {
            source.u16("read metadata scalar")?;
        }
        4..=6 => {
            source.u32("read metadata scalar")?;
        }
        8 => {
            source.discard_string("read metadata string")?;
        }
        9 => {
            if depth >= MAX_METADATA_DEPTH {
                return Err(limit_error(
                    "metadata nesting depth",
                    u64::from(depth) + 1,
                    u64::from(MAX_METADATA_DEPTH),
                ));
            }
            let child_type = source.u32("read metadata array type")?;
            let count = source.u64("read metadata array count")?;
            if child_type > 12 {
                return Err(GgufError::UnknownValueType(child_type).into());
            }
            *metadata_array_elements =
                metadata_array_elements.checked_add(count).ok_or_else(|| {
                    limit_error(
                        "metadata array elements",
                        u64::MAX,
                        MAX_METADATA_ARRAY_ELEMENTS,
                    )
                })?;
            enforce_limit(
                "metadata array elements",
                *metadata_array_elements,
                MAX_METADATA_ARRAY_ELEMENTS,
            )?;
            for _ in 0..count {
                skip_value(source, child_type, depth + 1, metadata_array_elements)?;
                enforce_limit("header bytes", source.position(), MAX_HEADER_BYTES)?;
            }
        }
        10..=12 => {
            source.u64("read metadata scalar")?;
        }
        other => return Err(GgufError::UnknownValueType(other).into()),
    }
    Ok(())
}

fn reject_duplicate_names(table: &[TableTensor]) -> Result<(), SaltGgufReadError> {
    let mut names = Vec::new();
    try_reserve_exact(&mut names, table.len(), size_of::<&str>())?;
    names.extend(table.iter().map(|tensor| tensor.name.as_str()));
    names.sort_unstable();
    if let Some(pair) = names.windows(2).find(|pair| pair[0] == pair[1]) {
        return Err(FormatError::SaltDuplicateTensor(pair[0].to_owned()).into());
    }
    Ok(())
}

fn validate_offsets(
    table: &[TableTensor],
    data_len: u64,
    alignment: u64,
) -> Result<(), SaltGgufReadError> {
    let mut previous = None;
    for tensor in table {
        if tensor.relative_offset % alignment != 0 || tensor.relative_offset > data_len {
            return Err(FormatError::SaltGgufBadFormat.into());
        }
        if let Some(previous) = previous {
            if tensor.relative_offset <= previous {
                return Err(FormatError::SaltGgufBadFormat.into());
            }
        } else if tensor.relative_offset != 0 {
            return Err(FormatError::SaltGgufBadFormat.into());
        }
        previous = Some(tensor.relative_offset);
    }
    Ok(())
}

fn validate_payload_tail<R: Read + Seek>(
    source: &mut Source<R>,
    table: &[TableTensor],
    index: usize,
    used: u64,
    interval_end: u64,
    alignment: u64,
) -> Result<(), SaltGgufReadError> {
    let tensor = &table[index];
    let payload_end = tensor
        .relative_offset
        .checked_add(used)
        .ok_or(FormatError::SaltGgufBadFormat)?;
    if index + 1 == table.len() {
        if source.position() != interval_end {
            return Err(FormatError::WrongBlockLen {
                expected: usize::try_from(source.position()).unwrap_or(usize::MAX),
                got: usize::try_from(interval_end).unwrap_or(usize::MAX),
            }
            .into());
        }
        return Ok(());
    }
    let expected_next = align_up(payload_end, alignment)?;
    if table[index + 1].relative_offset != expected_next {
        return Err(FormatError::SaltGgufBadFormat.into());
    }
    source.read_zeroes_to(interval_end, "read tensor alignment padding")
}

fn align_up(value: u64, alignment: u64) -> Result<u64, SaltGgufReadError> {
    let bumped = value
        .checked_add(alignment - 1)
        .ok_or(FormatError::SaltGgufBadFormat)?;
    Ok(bumped - bumped % alignment)
}

fn read_row<'a, R: Read + Seek>(
    source: &mut Source<R>,
    tensor_end: u64,
    scratch: &'a mut Vec<u8>,
) -> Result<PackedSaltRowRef<'a>, SaltGgufReadError> {
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
        SaltGgufReadError::AllocationFailed {
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
            for descriptor in
                scratch[SALT_HEADER_BYTES..prefix_len].chunks_exact(PLANE_DESCRIPTOR_BYTES)
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

fn reserve_and_resize(buffer: &mut Vec<u8>, len: usize) -> Result<(), SaltGgufReadError> {
    if len > buffer.len() {
        buffer.try_reserve_exact(len - buffer.len()).map_err(|_| {
            SaltGgufReadError::AllocationFailed {
                requested_bytes: len,
            }
        })?;
        buffer.resize(len, 0);
    }
    Ok(())
}

fn try_reserve_exact<T>(
    values: &mut Vec<T>,
    elements: usize,
    element_bytes: usize,
) -> Result<(), SaltGgufReadError> {
    values
        .try_reserve_exact(elements)
        .map_err(|_| SaltGgufReadError::AllocationFailed {
            requested_bytes: elements.saturating_mul(element_bytes),
        })
}

fn enforce_limit(resource: &str, actual: u64, limit: u64) -> Result<(), SaltGgufReadError> {
    if actual > limit {
        Err(limit_error(resource, actual, limit))
    } else {
        Ok(())
    }
}

fn limit_error(resource: &str, actual: u64, limit: u64) -> SaltGgufReadError {
    SaltGgufReadError::LimitExceeded {
        resource: resource.to_owned(),
        limit,
        actual,
    }
}

fn size_of<T>() -> usize {
    core::mem::size_of::<T>()
}

#[derive(Debug)]
struct Source<R> {
    inner: R,
    len: u64,
    position: u64,
}

impl<R: Read + Seek> Source<R> {
    fn new(mut inner: R) -> Result<Self, SaltGgufReadError> {
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

    fn seek_abs(&mut self, position: u64, context: &str) -> Result<(), SaltGgufReadError> {
        if position > self.len {
            return Err(GgufError::OffsetOutOfBounds.into());
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
    ) -> Result<(), SaltGgufReadError> {
        let end = self
            .position
            .checked_add(bytes.len() as u64)
            .ok_or(GgufError::DimsOverflow)?;
        if end > self.len {
            return Err(GgufError::Truncated.into());
        }
        for chunk in bytes.chunks_mut(IO_CHUNK_BYTES) {
            self.inner
                .read_exact(chunk)
                .map_err(|error| io_error(context, error))?;
            self.position += chunk.len() as u64;
        }
        Ok(())
    }

    fn read_zeroes_to(&mut self, end: u64, context: &str) -> Result<(), SaltGgufReadError> {
        if end < self.position || end > self.len {
            return Err(GgufError::OffsetOutOfBounds.into());
        }
        let mut scratch = [0u8; IO_CHUNK_BYTES];
        while self.position < end {
            let len = usize::try_from((end - self.position).min(IO_CHUNK_BYTES as u64))
                .expect("64 KiB chunk fits usize");
            self.read_exact_chunks(&mut scratch[..len], context)?;
            if scratch[..len].iter().any(|&byte| byte != 0) {
                return Err(FormatError::SaltGgufBadFormat.into());
            }
        }
        Ok(())
    }

    fn array<const N: usize>(&mut self, context: &str) -> Result<[u8; N], SaltGgufReadError> {
        let mut bytes = [0u8; N];
        self.read_exact_chunks(&mut bytes, context)?;
        Ok(bytes)
    }

    fn u8(&mut self, context: &str) -> Result<u8, SaltGgufReadError> {
        Ok(self.array::<1>(context)?[0])
    }

    fn u16(&mut self, context: &str) -> Result<u16, SaltGgufReadError> {
        Ok(u16::from_le_bytes(self.array::<2>(context)?))
    }

    fn u32(&mut self, context: &str) -> Result<u32, SaltGgufReadError> {
        Ok(u32::from_le_bytes(self.array::<4>(context)?))
    }

    fn u64(&mut self, context: &str) -> Result<u64, SaltGgufReadError> {
        Ok(u64::from_le_bytes(self.array::<8>(context)?))
    }

    fn string(&mut self, context: &str) -> Result<String, SaltGgufReadError> {
        let len = self.u64(context)?;
        enforce_limit("string bytes", len, MAX_STRING_BYTES)?;
        let len = usize::try_from(len)
            .map_err(|_| limit_error("string bytes", len, usize::MAX as u64))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| SaltGgufReadError::AllocationFailed {
                requested_bytes: len,
            })?;
        bytes.resize(len, 0);
        self.read_exact_chunks(&mut bytes, context)?;
        String::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8.into())
    }

    fn discard_string(&mut self, context: &str) -> Result<(), SaltGgufReadError> {
        let _ = self.string(context)?;
        Ok(())
    }
}

fn io_error(context: &str, error: std::io::Error) -> SaltGgufReadError {
    SaltGgufReadError::Io {
        context: context.to_owned(),
        kind: error.kind(),
        message: error.to_string(),
    }
}
