//! Strict seek-backed access to canonical SALT V2 packages.

use core::fmt;
use core::ops::ControlFlow;
use std::io::{ErrorKind, Read, Seek, SeekFrom};

use super::*;
use crate::{
    FormatError, PackageHasher, PackageId, Q2_0_BLOCK_BYTES, Q2_0_GROUP_SIZE, pack_q2_0_row,
    q2_0_num_blocks,
};

const MAX_TENSORS: u64 = 1_000_000;
const MAX_TOTAL_NAME_BYTES: u64 = 100_000_000;
const MAX_RANK: u64 = 4_096;
const MAX_TOTAL_DIMENSIONS: u64 = 16_000_000;
const MAX_PRESENCE_MAP_BYTES: u64 = 256 * 1024 * 1024;
const MAX_READ_CHUNK_BYTES: usize = 64 * 1024;
const MAX_BATCH_PAYLOAD_BYTES: usize = 64 * 1024;
const MAX_PLANES_PER_BATCH: usize = 1_024;

/// Errors from strict seek-backed SALT V2 package indexing and plane visits.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2PackageReadError {
    /// Encoded SALT V2 data violated the canonical package format.
    Format(SaltV2PackageError),
    /// Source seek or read failed.
    Io {
        /// Operation being attempted.
        context: String,
        /// Portable I/O error classification.
        kind: ErrorKind,
        /// Original error text.
        message: String,
    },
    /// A bounded parser allocation failed.
    AllocationFailed {
        /// Bytes requested by the failed reservation.
        requested_bytes: usize,
    },
    /// An explicit model-reader resource limit was exceeded.
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
    /// Tensor bytes changed after strict construction-time validation.
    SourceChanged(String),
    /// Exact package bytes changed after strict construction-time validation.
    PackageChanged,
}

impl fmt::Display for SaltV2PackageReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(error) => write!(f, "SALT V2 package: {error}"),
            Self::Io {
                context, message, ..
            } => write!(f, "SALT V2 package {context}: {message}"),
            Self::AllocationFailed { requested_bytes } => write!(
                f,
                "SALT V2 package allocation of {requested_bytes} bytes failed"
            ),
            Self::LimitExceeded {
                resource,
                limit,
                actual,
            } => write!(
                f,
                "SALT V2 package {resource} {actual} exceeds limit {limit}"
            ),
            Self::TensorNotFound(name) => write!(f, "SALT V2 tensor `{name}` not found"),
            Self::SourceChanged(name) => {
                write!(f, "SALT V2 tensor `{name}` changed after validation")
            }
            Self::PackageChanged => f.write_str("SALT V2 package bytes changed after validation"),
        }
    }
}

impl std::error::Error for SaltV2PackageReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SaltV2PackageError> for SaltV2PackageReadError {
    fn from(error: SaltV2PackageError) -> Self {
        Self::Format(error)
    }
}

/// Errors from exact CompactV1 one-plane SALT V2 to Q2_0 export.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompactQ2ExportError {
    /// Strict SALT V2 package access or integrity verification failed.
    Read(SaltV2PackageReadError),
    /// Compact Q2_0 export requires one additive plane in every allocation tile.
    IncompatiblePlaneCount {
        /// Offending allocation-tile index.
        tile_index: usize,
        /// Number of planes present in that tile.
        got: usize,
    },
    /// Compact Q2_0 export requires source scales grouped by 128 coefficients.
    IncompatibleScaleGroupSize {
        /// Source tensor scale-group width.
        got: usize,
    },
    /// Bare Q2_0 bytes cannot preserve SALT V2 transform metadata.
    IncompatibleTransform {
        /// Source tensor transform identity.
        got: SaltV2Transform,
    },
    /// Q2_0 blocks cannot cross row boundaries in a shaped tensor.
    IncompatibleRowWidth {
        /// Source tensor innermost dimension.
        got: usize,
    },
    /// Output length arithmetic overflowed.
    LengthOverflow,
    /// Final output storage could not be reserved.
    AllocationFailed {
        /// Exact Q2_0 tensor bytes requested.
        requested_bytes: usize,
    },
    /// Q2_0 packing rejected internally derived geometry or coefficients.
    Format(FormatError),
}

impl fmt::Display for CompactQ2ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read(error) => write!(f, "Compact Q2_0 export: {error}"),
            Self::IncompatiblePlaneCount { tile_index, got } => write!(
                f,
                "Compact Q2_0 export requires one plane per tile; tile {tile_index} has {got}"
            ),
            Self::IncompatibleScaleGroupSize { got } => write!(
                f,
                "Compact Q2_0 export requires SALT V2 G128 scales, got G{got}"
            ),
            Self::IncompatibleTransform { got } => write!(
                f,
                "Compact Q2_0 export cannot preserve SALT V2 transform {got:?}"
            ),
            Self::IncompatibleRowWidth { got } => write!(
                f,
                "Compact Q2_0 export requires row width divisible by {Q2_0_GROUP_SIZE}, got {got}"
            ),
            Self::LengthOverflow => f.write_str("Compact Q2_0 output length overflowed"),
            Self::AllocationFailed { requested_bytes } => write!(
                f,
                "Compact Q2_0 output allocation of {requested_bytes} bytes failed"
            ),
            Self::Format(error) => write!(f, "Compact Q2_0 packing: {error}"),
        }
    }
}

impl std::error::Error for CompactQ2ExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read(error) => Some(error),
            Self::Format(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SaltV2PackageReadError> for CompactQ2ExportError {
    fn from(error: SaltV2PackageReadError) -> Self {
        Self::Read(error)
    }
}

impl From<FormatError> for CompactQ2ExportError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

/// Borrowed canonical bytes and scales for one present additive plane.
///
/// The view is valid only for the duration of the visitor callback. Its packed
/// bytes use the package codec returned by [`SaltV2PackageReader::codec`].
#[derive(Clone, Copy, Debug)]
pub struct PackedSaltV2PlaneRef<'a> {
    tile_index: usize,
    plane_index: usize,
    plane_count: usize,
    logical_len: usize,
    packed: &'a [u8],
    scales: &'a [f16],
}

impl<'a> PackedSaltV2PlaneRef<'a> {
    /// Allocation-tile index within the selected tensor.
    #[must_use]
    pub const fn tile_index(self) -> usize {
        self.tile_index
    }

    /// Additive-plane index within the tile.
    #[must_use]
    pub const fn plane_index(self) -> usize {
        self.plane_index
    }

    /// Number of present additive planes in this tile.
    #[must_use]
    pub const fn plane_count(self) -> usize {
        self.plane_count
    }

    /// Logical ternary coefficient count in this plane.
    #[must_use]
    pub const fn logical_len(self) -> usize {
        self.logical_len
    }

    /// Canonical package-codec payload bytes.
    #[must_use]
    pub const fn packed_bytes(self) -> &'a [u8] {
        self.packed
    }

    /// Zero-point-free f16 scales, one per tensor-declared logical group.
    #[must_use]
    pub const fn scales(self) -> &'a [f16] {
        self.scales
    }
}

/// Owned metadata and exact final-arena requirements for one tensor.
#[derive(Clone, Debug)]
pub struct SaltV2TensorInfo {
    dims: Vec<u64>,
    logical_coefficients: usize,
    transform: SaltV2Transform,
    scale_group_size: usize,
    tile_count: usize,
    present_planes: usize,
    encoded_payload_bytes: u64,
    encoded_scale_bytes: u64,
    runtime_ledger: SaltV2IndexedRuntimeLedger,
    semantic_content_digest: [u8; 32],
}

impl SaltV2TensorInfo {
    /// Tensor dimensions in row-major semantic order.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Product of the tensor dimensions.
    #[must_use]
    pub const fn logical_coefficients(&self) -> usize {
        self.logical_coefficients
    }

    /// Transform identity required by this tensor.
    #[must_use]
    pub const fn transform(&self) -> SaltV2Transform {
        self.transform
    }

    /// Number of coefficients sharing each zero-point-free scale.
    #[must_use]
    pub const fn scale_group_size(&self) -> usize {
        self.scale_group_size
    }

    /// Number of 256-coefficient allocation tiles.
    #[must_use]
    pub const fn tile_count(&self) -> usize {
        self.tile_count
    }

    /// Number of physically present additive planes.
    #[must_use]
    pub const fn present_planes(&self) -> usize {
        self.present_planes
    }

    /// Canonical encoded ternary payload bytes.
    #[must_use]
    pub const fn encoded_payload_bytes(&self) -> u64 {
        self.encoded_payload_bytes
    }

    /// Canonical encoded f16 scale bytes.
    #[must_use]
    pub const fn encoded_scale_bytes(&self) -> u64 {
        self.encoded_scale_bytes
    }

    /// Exact requested bytes for the descriptor-free indexed runtime layout.
    #[must_use]
    pub const fn runtime_ledger(&self) -> SaltV2IndexedRuntimeLedger {
        self.runtime_ledger
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TensorSectionDigest {
    payload: [u8; 32],
    scales: [u8; 32],
}

#[derive(Clone, Debug)]
struct IndexedTensor {
    name: String,
    info: SaltV2TensorInfo,
    record_offset: u64,
    metadata_len: u64,
    payload_offset: u64,
    scales_offset: u64,
    full_tile_start: usize,
    full_tile_count: usize,
    ragged_plane_count: Option<usize>,
    metadata_digest: [u8; 32],
    section_digest: TensorSectionDigest,
}

#[derive(Debug)]
struct OwnedPresenceMap {
    embedded_value: u8,
    bytes: Vec<u8>,
    complete_bits: usize,
}

impl OwnedPresenceMap {
    fn new(
        embedded_value: u8,
        bytes: Vec<u8>,
        total_full_tiles: usize,
    ) -> Result<Self, SaltV2PackageError> {
        ValidatedPresenceMap::new(embedded_value, &bytes, total_full_tiles)?;
        let complete_bits = bytes
            .len()
            .checked_mul(8)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        Ok(Self {
            embedded_value,
            bytes,
            complete_bits,
        })
    }

    fn plane_count(&self, tensor: &IndexedTensor, tile_index: usize) -> usize {
        if tile_index < tensor.full_tile_count {
            let global_tile = tensor.full_tile_start + tile_index;
            let plane_two_bit = global_tile * 2;
            let plane_two = global_map_bit(
                &self.bytes,
                self.embedded_value,
                self.complete_bits,
                plane_two_bit,
            );
            let plane_three = global_map_bit(
                &self.bytes,
                self.embedded_value,
                self.complete_bits,
                plane_two_bit + 1,
            );
            1 + usize::from(plane_two) + usize::from(plane_three)
        } else {
            tensor
                .ragged_plane_count
                .expect("only a ragged final tile follows the full-tile prefix")
        }
    }

    fn physical_byte_range(&self, tensor: &IndexedTensor) -> core::ops::Range<usize> {
        let start_bit = tensor.full_tile_start * 2;
        let end_bit = (tensor.full_tile_start + tensor.full_tile_count) * 2;
        let physical_end = end_bit.min(self.complete_bits);
        if start_bit >= physical_end {
            return 0..0;
        }
        start_bit / 8..physical_end.div_ceil(8)
    }
}

/// Strict, bounded-staging reader for canonical SALT V2 packages.
///
/// Construction validates every tensor, including tensors never selected later,
/// while retaining only owned metadata, the package's compact two-bit presence
/// map, exact runtime requirements, and section digests. Validation and visits
/// stage at most 64 KiB of packed payload plus a bounded descriptor batch; no
/// semantic whole-tensor or whole-package ternary vectors are retained.
#[derive(Debug)]
pub struct SaltV2PackageReader<R> {
    source: Source<R>,
    package_id: PackageId,
    header: [u8; SALT_V2_PACKAGE_HEADER_BYTES],
    codec: SaltV2Codec,
    ledger: SaltV2PackageLedger,
    tensors: Vec<IndexedTensor>,
    encoded_order: Vec<usize>,
    map: OwnedPresenceMap,
    map_offset: u64,
}

impl<R: Read + Seek> SaltV2PackageReader<R> {
    /// Parse and strictly validate an entire SALT V2 source with bounded staging.
    ///
    /// The model-reader policy accepts at most one million tensors, rank 4096,
    /// 16 million dimensions in aggregate, 100 MB of tensor names, and a 256 MiB
    /// physical presence map. The last bound still covers more than 250 billion
    /// logical coefficients at one 256-coefficient allocation tile per map entry.
    ///
    /// # Errors
    /// Returns typed format, I/O, allocation, and resource-limit errors. Duplicate
    /// names, malformed unselected planes, truncation, and trailing bytes fail here.
    pub fn new_strict(reader: R) -> Result<Self, SaltV2PackageReadError> {
        let mut source = Source::new(reader)?;
        source.start_package_hash();
        let header = source.array::<SALT_V2_PACKAGE_HEADER_BYTES>("read package header")?;
        let mut header_cursor = Cursor::new(&header);
        if header_cursor.take(SALT_V2_PACKAGE_MAGIC.len())? != SALT_V2_PACKAGE_MAGIC {
            return Err(SaltV2PackageError::BadMagic.into());
        }
        let version = header_cursor.u16()?;
        if !matches!(
            version,
            SALT_V2_PACKAGE_VERSION | SALT_V2_PACKAGE_VERSION_SCALE_GEOMETRY
        ) {
            return Err(SaltV2PackageError::UnsupportedVersion(version).into());
        }
        let codec = codec_from_tag(header_cursor.u8()?)?;
        let flags = header_cursor.u8()?;
        if flags != 0 {
            return Err(SaltV2PackageError::NonZeroFlags(flags).into());
        }
        let packed_tensor_count = header_cursor.u32()?;
        let tensor_count_mask = (1u32 << SALT_V2_TENSOR_COUNT_BITS) - 1;
        let tensor_count = u64::from(packed_tensor_count & tensor_count_mask);
        let embedded_map_value = (packed_tensor_count >> SALT_V2_TENSOR_COUNT_BITS) as u8;
        let declared_total = header_cursor.u64()?;
        if declared_total != source.len() {
            return Err(SaltV2PackageError::WrongTotalLength {
                declared: declared_total,
                actual: usize::try_from(source.len()).unwrap_or(usize::MAX),
            }
            .into());
        }
        if tensor_count == 0 {
            return Err(SaltV2PackageError::EmptyPackage.into());
        }
        enforce_limit("tensor count", tensor_count, MAX_TENSORS)?;
        let minimum_headers = tensor_count
            .checked_mul(SALT_V2_TENSOR_HEADER_BYTES as u64)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        if minimum_headers > source.remaining() {
            return Err(SaltV2PackageError::Truncated {
                needed: usize::try_from(minimum_headers).unwrap_or(usize::MAX),
                remaining: usize::try_from(source.remaining()).unwrap_or(usize::MAX),
            }
            .into());
        }

        let tensor_count_usize = usize::try_from(tensor_count)
            .map_err(|_| limit_error("tensor count", tensor_count, usize::MAX as u64))?;
        let mut tensors = Vec::new();
        try_reserve_exact::<IndexedTensor>(&mut tensors, tensor_count_usize)?;
        let mut total_name_bytes = 0u64;
        let mut total_dimensions = 0u64;
        let mut total_tiles = 0usize;
        let mut total_full_tiles = 0usize;
        let mut ragged_tensor_count = 0usize;
        let mut ledger = SaltV2PackageLedger {
            headers_bytes: SALT_V2_PACKAGE_HEADER_BYTES as u64,
            ..SaltV2PackageLedger::default()
        };

        for _ in 0..tensor_count_usize {
            let record_offset = source.position();
            source.start_range_digest();
            let name_len = u64::from(source.u32("read tensor name length")?);
            let rank = u64::from(source.u32("read tensor rank")?);
            enforce_limit("tensor rank", rank, MAX_RANK)?;
            total_name_bytes = total_name_bytes.checked_add(name_len).ok_or_else(|| {
                limit_error("total tensor-name bytes", u64::MAX, MAX_TOTAL_NAME_BYTES)
            })?;
            enforce_limit(
                "total tensor-name bytes",
                total_name_bytes,
                MAX_TOTAL_NAME_BYTES,
            )?;
            total_dimensions = total_dimensions.checked_add(rank).ok_or_else(|| {
                limit_error("total tensor dimensions", u64::MAX, MAX_TOTAL_DIMENSIONS)
            })?;
            enforce_limit(
                "total tensor dimensions",
                total_dimensions,
                MAX_TOTAL_DIMENSIONS,
            )?;

            let declared_coefficients = source.u64("read logical coefficient count")?;
            let packed_declared_tiles = source.u64("read allocation tile count")?;
            let declared_payload = source.u64("read payload length")?;
            let declared_scales = source.u64("read scale length")?;
            let transform_bytes =
                source.array::<SALT_V2_TRANSFORM_METADATA_BYTES>("read transform metadata")?;
            let (transform, scale_group_size) =
                read_layout(&mut Cursor::new(&transform_bytes), version)?;

            let name_len_usize = usize::try_from(name_len)
                .map_err(|_| limit_error("tensor name bytes", name_len, usize::MAX as u64))?;
            let name = source.string(name_len_usize, "read tensor name")?;
            if name.is_empty() {
                return Err(SaltV2PackageError::EmptyTensorName.into());
            }
            let rank_usize = usize::try_from(rank)
                .map_err(|_| limit_error("tensor rank", rank, usize::MAX as u64))?;
            let mut dims = Vec::new();
            try_reserve_exact::<u64>(&mut dims, rank_usize)?;
            for _ in 0..rank_usize {
                dims.push(source.u64("read tensor dimension")?);
            }
            let logical_u64 = checked_dimension_product(&dims)?;
            require_declared(
                "logical coefficient count",
                declared_coefficients,
                logical_u64,
            )?;
            let logical_coefficients = usize::try_from(logical_u64)
                .map_err(|_| SaltV2PackageError::DimensionProductTooLarge(logical_u64))?;
            let tile_count = logical_coefficients.div_ceil(SALT_V2_ALLOCATION_TILE_SIZE);
            let tile_count_mask = (1u64 << SALT_V2_TILE_COUNT_BITS) - 1;
            let declared_tiles = packed_declared_tiles & tile_count_mask;
            let ragged_map_value = (packed_declared_tiles >> SALT_V2_TILE_COUNT_BITS) as u8;
            require_declared(
                "allocation tile count",
                declared_tiles,
                u64::try_from(tile_count).map_err(|_| SaltV2PackageError::LengthOverflow)?,
            )?;
            let full_tile_count = logical_coefficients / SALT_V2_ALLOCATION_TILE_SIZE;
            let ragged_plane_count =
                if logical_coefficients.is_multiple_of(SALT_V2_ALLOCATION_TILE_SIZE) {
                    if ragged_map_value != 0 {
                        return Err(SaltV2PackageError::NonCanonicalMapPadding.into());
                    }
                    None
                } else {
                    ragged_tensor_count = ragged_tensor_count
                        .checked_add(1)
                        .ok_or(SaltV2PackageError::LengthOverflow)?;
                    Some(map_value_plane_count(ragged_map_value, tile_count - 1)?)
                };
            let minimum = minimum_tensor_physical(codec, logical_coefficients, scale_group_size)?;
            require_at_least("payload bytes", declared_payload, minimum.payload_bytes)?;
            require_at_least("scale bytes", declared_scales, minimum.scales_bytes)?;

            total_tiles = total_tiles
                .checked_add(tile_count)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            let full_tile_start = total_full_tiles;
            total_full_tiles = total_full_tiles
                .checked_add(full_tile_count)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            let payload_offset = source.position();
            let metadata_digest = source.finish_range_digest();
            let scales_offset = payload_offset
                .checked_add(declared_payload)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            let sections_end = scales_offset
                .checked_add(declared_scales)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            if sections_end > source.len() {
                return Err(SaltV2PackageError::Truncated {
                    needed: usize::try_from(
                        declared_payload
                            .checked_add(declared_scales)
                            .ok_or(SaltV2PackageError::LengthOverflow)?,
                    )
                    .unwrap_or(usize::MAX),
                    remaining: usize::try_from(source.remaining()).unwrap_or(usize::MAX),
                }
                .into());
            }
            let payload_digest =
                source.digest_next(declared_payload, "read tensor payload for package identity")?;
            let scale_digest =
                source.digest_next(declared_scales, "read tensor scales for package identity")?;
            debug_assert_eq!(source.position(), sections_end);

            let dims_bytes = rank_usize
                .checked_mul(8)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            let tensor_headers = (SALT_V2_TENSOR_HEADER_BYTES - SALT_V2_TRANSFORM_METADATA_BYTES)
                .checked_add(name_len_usize)
                .and_then(|value| value.checked_add(dims_bytes))
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            checked_ledger_add(&mut ledger.headers_bytes, tensor_headers)?;
            checked_ledger_add(
                &mut ledger.transform_bytes,
                SALT_V2_TRANSFORM_METADATA_BYTES,
            )?;
            tensors.push(IndexedTensor {
                name,
                info: SaltV2TensorInfo {
                    dims,
                    logical_coefficients,
                    transform,
                    scale_group_size,
                    tile_count,
                    present_planes: 0,
                    encoded_payload_bytes: declared_payload,
                    encoded_scale_bytes: declared_scales,
                    runtime_ledger: SaltV2IndexedRuntimeLedger::zero(),
                    semantic_content_digest: [0; 32],
                },
                record_offset,
                metadata_len: payload_offset
                    .checked_sub(record_offset)
                    .ok_or(SaltV2PackageError::LengthOverflow)?,
                payload_offset,
                scales_offset,
                full_tile_start,
                full_tile_count,
                ragged_plane_count,
                metadata_digest,
                section_digest: TensorSectionDigest {
                    payload: payload_digest,
                    scales: scale_digest,
                },
            });
        }

        let expected_version = package_version_for_scale_groups(
            tensors.iter().map(|tensor| tensor.info.scale_group_size),
        );
        if version != expected_version {
            return Err(SaltV2PackageError::NonCanonicalVersion {
                declared: version,
                expected: expected_version,
            }
            .into());
        }

        let map_offset = source.position();
        let map_len = presence_map_len(total_full_tiles)?;
        enforce_limit(
            "presence-map bytes",
            u64::try_from(map_len).unwrap_or(u64::MAX),
            MAX_PRESENCE_MAP_BYTES,
        )?;
        let mut map_bytes = Vec::new();
        reserve_and_resize(&mut map_bytes, map_len)?;
        source.read_exact_chunks(&mut map_bytes, "read presence map")?;
        let map = OwnedPresenceMap::new(embedded_map_value, map_bytes, total_full_tiles)?;

        let expected_padding = ((SALT_V2_PACKAGE_ALIGNMENT as u64
            - source.position() % SALT_V2_PACKAGE_ALIGNMENT as u64)
            % SALT_V2_PACKAGE_ALIGNMENT as u64) as usize;
        if source.remaining() != expected_padding as u64 {
            return Err(SaltV2PackageError::UnexpectedTrailingData {
                remaining: usize::try_from(source.remaining()).unwrap_or(usize::MAX),
                expected_padding,
            }
            .into());
        }
        let padding = source.array_vec(expected_padding, "read package padding")?;
        if padding.iter().any(|byte| *byte != 0) {
            return Err(SaltV2PackageError::NonCanonicalFilePadding.into());
        }
        let package_id = source.finish_package_hash()?;

        let total_map_bits = total_tiles
            .checked_mul(2)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let full_map_bits = total_full_tiles
            .checked_mul(2)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let tensor_embedded_bits = ragged_tensor_count
            .checked_mul(2)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        ledger.maps_bytes =
            u64::try_from(map_len).map_err(|_| SaltV2PackageError::LengthOverflow)?;
        ledger.allocation_map_bits =
            u64::try_from(total_map_bits).map_err(|_| SaltV2PackageError::LengthOverflow)?;
        ledger.allocation_map_package_embedded_bits = (full_map_bits % 8) as u8;
        ledger.allocation_map_tensor_embedded_bits =
            u64::try_from(tensor_embedded_bits).map_err(|_| SaltV2PackageError::LengthOverflow)?;
        ledger.allocation_map_embedded_bits =
            u64::from(ledger.allocation_map_package_embedded_bits)
                .checked_add(ledger.allocation_map_tensor_embedded_bits)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
        ledger.allocation_tiles =
            u64::try_from(total_tiles).map_err(|_| SaltV2PackageError::LengthOverflow)?;
        ledger.allocation_capacity_coefficients = ledger
            .allocation_tiles
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE as u64)
            .ok_or(SaltV2PackageError::LengthOverflow)?;

        tensors.sort_unstable_by(|left, right| left.name.cmp(&right.name));
        if let Some(pair) = tensors.windows(2).find(|pair| pair[0].name == pair[1].name) {
            return Err(SaltV2PackageError::DuplicateTensorName(pair[0].name.clone()).into());
        }
        for tensor in &mut tensors {
            let physical = expected_tensor_physical(
                codec,
                tensor.info.logical_coefficients,
                TensorPlaneCounts::new(&map, tensor),
                tensor.info.scale_group_size(),
            )?;
            require_declared(
                "payload bytes",
                tensor.info.encoded_payload_bytes,
                physical.payload_bytes,
            )?;
            require_declared(
                "scale bytes",
                tensor.info.encoded_scale_bytes,
                physical.scales_bytes,
            )?;
            let present_planes =
                TensorPlaneCounts::new(&map, tensor).try_fold(0usize, |total, count| {
                    total
                        .checked_add(count)
                        .ok_or(SaltV2PackageError::LengthOverflow)
                })?;
            tensor.info.present_planes = present_planes;
            tensor.info.runtime_ledger = indexed_runtime_ledger(tensor, present_planes)?;
            let metadata_digest = hash_range(
                &mut source,
                tensor.record_offset,
                tensor.metadata_len,
                "hash tensor metadata",
            )?;
            if metadata_digest != tensor.metadata_digest {
                return Err(SaltV2PackageReadError::SourceChanged(tensor.name.clone()));
            }
            let mut semantic_hasher = SaltV2SemanticTensorHasher::new_content_only(
                &tensor.info.dims,
                tensor.info.logical_coefficients,
                tensor.info.transform,
                tensor.info.scale_group_size,
                tensor.info.tile_count,
            );
            let section_digest = scan_tensor(
                &mut source,
                codec,
                &map,
                tensor,
                Some(&mut semantic_hasher),
                |_| ControlFlow::Continue(()),
            )?
            .expect("construction-time semantic visitor is infallible");
            if section_digest != tensor.section_digest {
                return Err(SaltV2PackageReadError::SourceChanged(tensor.name.clone()));
            }
            tensor.info.semantic_content_digest = semantic_hasher.finalize_content_digest();
            checked_ledger_add(
                &mut ledger.payload_bytes,
                usize::try_from(physical.payload_bytes)
                    .map_err(|_| SaltV2PackageError::LengthOverflow)?,
            )?;
            checked_ledger_add(
                &mut ledger.scales_bytes,
                usize::try_from(physical.scales_bytes)
                    .map_err(|_| SaltV2PackageError::LengthOverflow)?,
            )?;
            ledger.codec_padding_trits = ledger
                .codec_padding_trits
                .checked_add(physical.codec_padding_trits)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            ledger.codec_padding_bits = ledger
                .codec_padding_bits
                .checked_add(physical.codec_padding_bits)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
        }

        let mut encoded_order = Vec::new();
        try_reserve_exact::<usize>(&mut encoded_order, tensors.len())?;
        encoded_order.extend(0..tensors.len());
        encoded_order.sort_unstable_by_key(|&index| tensors[index].record_offset);

        ledger.padding_bytes = expected_padding as u64;
        ledger.serialized_unpadded_bytes = ledger
            .headers_bytes
            .checked_add(ledger.transform_bytes)
            .and_then(|bytes| bytes.checked_add(ledger.maps_bytes))
            .and_then(|bytes| bytes.checked_add(ledger.payload_bytes))
            .and_then(|bytes| bytes.checked_add(ledger.scales_bytes))
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        ledger.total_bytes = ledger
            .serialized_unpadded_bytes
            .checked_add(ledger.padding_bytes)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        require_declared("total package bytes", declared_total, ledger.total_bytes)?;

        Ok(Self {
            source,
            package_id,
            header,
            codec,
            ledger,
            tensors,
            encoded_order,
            map,
            map_offset,
        })
    }

    /// Physical codec used by every visited plane.
    #[must_use]
    pub const fn codec(&self) -> SaltV2Codec {
        self.codec
    }

    /// Exact identity of the package bytes consumed by strict construction.
    ///
    /// The identity is computed during the same bounded, sequential pass that
    /// supplies parsed metadata and every retained mutation baseline.
    #[must_use]
    pub const fn package_id(&self) -> PackageId {
        self.package_id
    }

    /// Exact physical package ledger measured during strict validation.
    #[must_use]
    pub const fn ledger(&self) -> SaltV2PackageLedger {
        self.ledger
    }

    /// Exact indexed-runtime requirements summed across all package tensors.
    ///
    /// This preserves each tensor's independently allocated map tail and rank
    /// prefixes; it is therefore not interchangeable with package-file byte
    /// accounting from [`Self::ledger`].
    ///
    /// # Errors
    /// Returns a format error if an aggregate counter overflows.
    pub fn indexed_runtime_ledger(
        &self,
    ) -> Result<SaltV2IndexedRuntimeLedger, SaltV2PackageReadError> {
        self.tensors
            .iter()
            .try_fold(SaltV2IndexedRuntimeLedger::zero(), |total, tensor| {
                total.checked_add(tensor.info.runtime_ledger())
            })
            .map_err(Into::into)
    }

    /// Number of indexed tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the package contains no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Tensor names in lexical order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|tensor| tensor.name.as_str())
    }

    /// Tensor names in their physical package-record order.
    pub fn tensor_names_encoded_order(&self) -> impl Iterator<Item = &str> {
        self.encoded_order
            .iter()
            .map(|&index| self.tensors[index].name.as_str())
    }

    /// Metadata and exact final indexed-runtime requirements for a named tensor.
    #[must_use]
    pub fn tensor_info(&self, name: &str) -> Option<&SaltV2TensorInfo> {
        self.find_tensor(name).map(|tensor| &tensor.info)
    }

    /// Per-tile present additive-plane counts for a named tensor.
    ///
    /// The exact-size iterator reads the retained, strictly validated presence
    /// map without exposing its physical byte layout. Call
    /// [`Self::verify_unchanged`] after consuming package metadata and counts
    /// when the result will authorize publication or another durable action.
    ///
    /// # Errors
    /// Returns [`SaltV2PackageReadError::TensorNotFound`] when `name` is absent.
    pub fn tensor_plane_counts(
        &self,
        name: &str,
    ) -> Result<impl ExactSizeIterator<Item = usize> + '_, SaltV2PackageReadError> {
        let tensor = self
            .find_tensor(name)
            .ok_or_else(|| SaltV2PackageReadError::TensorNotFound(name.to_owned()))?;
        Ok(TensorPlaneCounts::new(&self.map, tensor))
    }

    /// Codec-independent semantic manifest entry for a named tensor.
    ///
    /// Its content digest is computed during the mandatory bounded validation
    /// scan. This constructs the owned entry from cached metadata and that digest,
    /// without materializing or rescanning tensor coefficients.
    #[must_use]
    pub fn semantic_tensor(&self, name: &str) -> Option<SemanticTensor> {
        let tensor = self.find_tensor(name)?;
        Some(
            SemanticTensor::from_digest(
                tensor.name.clone(),
                tensor.info.dims.clone(),
                tensor.info.semantic_content_digest,
            )
            .expect("strict construction already validated tensor name and shape"),
        )
    }

    /// Visit one tensor's canonical packed planes using bounded reusable staging.
    ///
    /// Static package/header/map metadata is verified before the first callback.
    /// Payload and scale digests are verified only after the complete visit, so
    /// callback side effects are not transactional. Callers must stage mutations
    /// and publish them only after this method returns `Ok(())`.
    ///
    /// # Errors
    /// Returns a typed error for a missing tensor, I/O failure, malformed source,
    /// or same-handle mutation after strict construction.
    pub fn visit_packed_tensor(
        &mut self,
        name: &str,
        mut visitor: impl FnMut(PackedSaltV2PlaneRef<'_>),
    ) -> Result<(), SaltV2PackageReadError> {
        let outcome = self.visit_packed_tensor_control(name, |plane| {
            visitor(plane);
            ControlFlow::Continue(())
        })?;
        debug_assert_eq!(outcome, PackedTensorVisitOutcome::Complete);
        Ok(())
    }

    pub(super) fn visit_packed_tensor_control(
        &mut self,
        name: &str,
        visitor: impl FnMut(PackedSaltV2PlaneRef<'_>) -> ControlFlow<()>,
    ) -> Result<PackedTensorVisitOutcome, SaltV2PackageReadError> {
        let tensor = self
            .find_tensor(name)
            .cloned()
            .ok_or_else(|| SaltV2PackageReadError::TensorNotFound(name.to_owned()))?;
        if !self.source.len_unchanged()? {
            return Err(SaltV2PackageReadError::SourceChanged(name.to_owned()));
        }
        if !range_matches(&mut self.source, 0, &self.header, "verify package header")? {
            return Err(SaltV2PackageReadError::SourceChanged(name.to_owned()));
        }
        let metadata_digest = hash_range(
            &mut self.source,
            tensor.record_offset,
            tensor.metadata_len,
            "verify tensor metadata",
        )?;
        if metadata_digest != tensor.metadata_digest {
            return Err(SaltV2PackageReadError::SourceChanged(name.to_owned()));
        }
        let map_range = self.map.physical_byte_range(&tensor);
        if !range_matches(
            &mut self.source,
            self.map_offset + map_range.start as u64,
            &self.map.bytes[map_range],
            "verify tensor presence map",
        )? {
            return Err(SaltV2PackageReadError::SourceChanged(name.to_owned()));
        }

        // Revalidation is deliberate: the digest is known only after callbacks,
        // so canonical decode and scale checks prevent malformed mutated bytes
        // from ever reaching a callback while the final digest catches valid
        // same-length mutations.
        let Some(digest) = (match scan_tensor(
            &mut self.source,
            self.codec,
            &self.map,
            &tensor,
            None,
            visitor,
        ) {
            Ok(digest) => digest,
            Err(SaltV2PackageReadError::Format(_)) => {
                return Err(SaltV2PackageReadError::SourceChanged(name.to_owned()));
            }
            Err(error) => return Err(error),
        }) else {
            return Ok(PackedTensorVisitOutcome::Aborted);
        };
        if digest != tensor.section_digest {
            return Err(SaltV2PackageReadError::SourceChanged(name.to_owned()));
        }
        Ok(PackedTensorVisitOutcome::Complete)
    }

    /// Recompute and verify the exact package identity through this reader handle.
    ///
    /// This terminal integrity check uses at most 64 KiB of scratch space and
    /// covers all package bytes, including tensors not visited by the caller.
    /// Call it after the final metadata/count/plane read and before publishing a
    /// result derived from those reads. Later source mutation is outside the
    /// guarantee of this method.
    ///
    /// # Errors
    /// Returns [`SaltV2PackageReadError::PackageChanged`] when the source length
    /// or exact package identity differs from strict construction. I/O and
    /// bounded-allocation failures remain typed.
    pub fn verify_unchanged(&mut self) -> Result<(), SaltV2PackageReadError> {
        if !self.source.len_unchanged()? {
            return Err(SaltV2PackageReadError::PackageChanged);
        }
        self.source.seek_abs(0, "verify package identity")?;
        let mut scratch = Vec::new();
        reserve_and_resize(
            &mut scratch,
            usize::try_from(self.source.len().min(MAX_READ_CHUNK_BYTES as u64))
                .unwrap_or(MAX_READ_CHUNK_BYTES),
        )?;
        let mut remaining = self.source.len();
        let mut hasher = PackageHasher::new();
        while remaining != 0 {
            let chunk_len = usize::try_from(remaining.min(MAX_READ_CHUNK_BYTES as u64))
                .expect("chunk length fits usize");
            self.source
                .read_exact_chunks(&mut scratch[..chunk_len], "verify package identity")?;
            hasher.update(&scratch[..chunk_len]);
            remaining -= chunk_len as u64;
        }
        if !self.source.len_unchanged()? || hasher.finalize() != self.package_id {
            return Err(SaltV2PackageReadError::PackageChanged);
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
            .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
            .ok()
            .map(|index| &self.tensors[index])
    }
}

/// Export one compatible uniform P=1, G128 SALT V2 tensor as llama.cpp Q2_0 g64 bytes.
///
/// Trits remain unchanged. Each complete G128 source scale is copied into its
/// two corresponding G64 Q2_0 blocks; a ragged final G128 group emits only the
/// blocks needed by its logical coefficients. Identity transform and row widths
/// divisible by 64 are required because bare Q2_0 bytes cannot carry SALT V2
/// transform metadata and Q2_0 blocks cannot cross tensor row boundaries. Package
/// and tensor integrity are reverified before the returned bytes publish.
///
/// # Errors
/// Fails closed when the tensor is absent, any allocation tile has other than one
/// plane, source scale geometry is not G128, transform identity is not `None`, row
/// width is not divisible by 64, source bytes changed, output storage cannot be
/// reserved, length arithmetic overflows, or Q2_0 packing fails.
pub fn export_compact_q2_0_tensor<R: Read + Seek>(
    reader: &mut SaltV2PackageReader<R>,
    name: &str,
) -> Result<Vec<u8>, CompactQ2ExportError> {
    let output_len = validate_compact_q2_0_tensor(reader, name)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| CompactQ2ExportError::AllocationFailed {
            requested_bytes: output_len,
        })?;
    match visit_compact_q2_0_tensor_without_package_verification(reader, name, |chunk| {
        output.extend_from_slice(chunk);
        Ok::<(), core::convert::Infallible>(())
    }) {
        Ok(()) => {}
        Err(CompactQ2VisitError::Export(error)) => return Err(error),
        Err(CompactQ2VisitError::Sink(never)) => match never {},
    }
    reader.verify_unchanged()?;
    Ok(output)
}

pub(super) fn validate_compact_q2_0_tensor<R: Read + Seek>(
    reader: &SaltV2PackageReader<R>,
    name: &str,
) -> Result<usize, CompactQ2ExportError> {
    let (logical_coefficients, scale_group_size, transform, row_width) = reader
        .tensor_info(name)
        .map(|info| {
            (
                info.logical_coefficients(),
                info.scale_group_size(),
                info.transform(),
                info.dims().last().copied(),
            )
        })
        .ok_or_else(|| {
            CompactQ2ExportError::Read(SaltV2PackageReadError::TensorNotFound(name.to_owned()))
        })?;
    if scale_group_size != SALT_V2_SCALE_GROUP_SIZE {
        return Err(CompactQ2ExportError::IncompatibleScaleGroupSize {
            got: scale_group_size,
        });
    }
    if transform != SaltV2Transform::None {
        return Err(CompactQ2ExportError::IncompatibleTransform { got: transform });
    }
    let row_width = usize::try_from(row_width.ok_or(CompactQ2ExportError::LengthOverflow)?)
        .map_err(|_| CompactQ2ExportError::LengthOverflow)?;
    if !row_width.is_multiple_of(Q2_0_GROUP_SIZE) {
        return Err(CompactQ2ExportError::IncompatibleRowWidth { got: row_width });
    }
    for (tile_index, plane_count) in reader.tensor_plane_counts(name)?.enumerate() {
        if plane_count != 1 {
            return Err(CompactQ2ExportError::IncompatiblePlaneCount {
                tile_index,
                got: plane_count,
            });
        }
    }
    q2_0_num_blocks(logical_coefficients)
        .checked_mul(Q2_0_BLOCK_BYTES)
        .ok_or(CompactQ2ExportError::LengthOverflow)
}

pub(super) enum CompactQ2VisitError<E> {
    Export(CompactQ2ExportError),
    Sink(E),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PackedTensorVisitOutcome {
    Complete,
    Aborted,
}

pub(super) fn visit_compact_q2_0_tensor_without_package_verification<R: Read + Seek, E>(
    reader: &mut SaltV2PackageReader<R>,
    name: &str,
    mut sink: impl FnMut(&[u8]) -> Result<(), E>,
) -> Result<(), CompactQ2VisitError<E>> {
    let output_len =
        validate_compact_q2_0_tensor(reader, name).map_err(CompactQ2VisitError::Export)?;
    let codec = reader.codec();
    let mut decoded = Vec::new();
    let mut callback_error = None;
    let mut output_bytes = 0usize;
    let mut packed = [0u8; Q2_0_BLOCK_BYTES * SALT_V2_ALLOCATION_TILE_SIZE / Q2_0_GROUP_SIZE];

    let visit_outcome = reader
        .visit_packed_tensor_control(name, |plane| {
            if callback_error.is_some() {
                return ControlFlow::Break(());
            }
            if let Err(error) = unpack_salt_v2_plane_into(
                codec,
                plane.packed_bytes(),
                plane.logical_len(),
                &mut decoded,
            ) {
                callback_error = Some(CompactQ2VisitError::Export(CompactQ2ExportError::Read(
                    error.into(),
                )));
                return ControlFlow::Break(());
            }
            let q2_blocks = q2_0_num_blocks(decoded.len());
            let mut q2_scales = [f16::ZERO; SALT_V2_ALLOCATION_TILE_SIZE / Q2_0_GROUP_SIZE];
            for (q2_index, scale) in q2_scales[..q2_blocks].iter_mut().enumerate() {
                *scale = plane.scales()[q2_index / 2];
            }
            let tile_bytes = match q2_blocks.checked_mul(Q2_0_BLOCK_BYTES) {
                Some(value) => value,
                None => {
                    callback_error = Some(CompactQ2VisitError::Export(
                        CompactQ2ExportError::LengthOverflow,
                    ));
                    return ControlFlow::Break(());
                }
            };
            let end = match output_bytes.checked_add(tile_bytes) {
                Some(value) if value <= output_len => value,
                _ => {
                    callback_error = Some(CompactQ2VisitError::Export(
                        CompactQ2ExportError::LengthOverflow,
                    ));
                    return ControlFlow::Break(());
                }
            };
            if let Err(error) =
                pack_q2_0_row(&decoded, &q2_scales[..q2_blocks], &mut packed[..tile_bytes])
            {
                callback_error = Some(CompactQ2VisitError::Export(error.into()));
                return ControlFlow::Break(());
            }
            if let Err(error) = sink(&packed[..tile_bytes]) {
                callback_error = Some(CompactQ2VisitError::Sink(error));
                return ControlFlow::Break(());
            }
            output_bytes = end;
            ControlFlow::Continue(())
        })
        .map_err(|error| CompactQ2VisitError::Export(error.into()))?;
    if let Some(error) = callback_error {
        return Err(error);
    }
    if visit_outcome != PackedTensorVisitOutcome::Complete {
        return Err(CompactQ2VisitError::Export(
            CompactQ2ExportError::LengthOverflow,
        ));
    }
    if output_bytes != output_len {
        return Err(CompactQ2VisitError::Export(
            CompactQ2ExportError::LengthOverflow,
        ));
    }
    Ok(())
}

struct TensorPlaneCounts<'a> {
    map: &'a OwnedPresenceMap,
    tensor: &'a IndexedTensor,
    next: usize,
}

impl<'a> TensorPlaneCounts<'a> {
    const fn new(map: &'a OwnedPresenceMap, tensor: &'a IndexedTensor) -> Self {
        Self {
            map,
            tensor,
            next: 0,
        }
    }
}

impl Iterator for TensorPlaneCounts<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.tensor.info.tile_count {
            return None;
        }
        let tile_index = self.next;
        self.next += 1;
        Some(self.map.plane_count(self.tensor, tile_index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.tensor.info.tile_count - self.next;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for TensorPlaneCounts<'_> {}

#[derive(Clone, Copy)]
struct PlaneDescriptor {
    tile_index: usize,
    plane_index: usize,
    plane_count: usize,
    logical_len: usize,
    packed_start: usize,
    packed_len: usize,
    scale_start: usize,
    scale_len: usize,
}

fn scan_tensor<R: Read + Seek>(
    source: &mut Source<R>,
    codec: SaltV2Codec,
    map: &OwnedPresenceMap,
    tensor: &IndexedTensor,
    mut semantic_hasher: Option<&mut SaltV2SemanticTensorHasher>,
    mut visitor: impl FnMut(PackedSaltV2PlaneRef<'_>) -> ControlFlow<()>,
) -> Result<Option<TensorSectionDigest>, SaltV2PackageReadError> {
    let mut descriptors = Vec::new();
    try_reserve_exact::<PlaneDescriptor>(&mut descriptors, MAX_PLANES_PER_BATCH)?;
    let mut payload = Vec::new();
    let mut scales = Vec::new();
    let mut trits = Vec::new();
    try_reserve_exact::<Trit>(&mut trits, SALT_V2_ALLOCATION_TILE_SIZE)?;
    let mut payload_hasher = blake3::Hasher::new();
    let mut scale_hasher = blake3::Hasher::new();
    let mut tile_index = 0usize;
    let mut plane_index = 0usize;
    let mut payload_position = tensor.payload_offset;
    let mut scale_position = tensor.scales_offset;

    while tile_index < tensor.info.tile_count {
        descriptors.clear();
        let mut packed_batch_len = 0usize;
        let mut scale_batch_len = 0usize;
        while tile_index < tensor.info.tile_count && descriptors.len() < MAX_PLANES_PER_BATCH {
            let consumed = tile_index
                .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            let logical_len =
                (tensor.info.logical_coefficients - consumed).min(SALT_V2_ALLOCATION_TILE_SIZE);
            let plane_count = map.plane_count(tensor, tile_index);
            let stored_trits = stored_trit_count(codec, logical_len)?;
            let packed_len = codec
                .ledger(stored_trits)
                .map_err(SaltV2PackageError::from)?
                .physical_bytes;
            if !descriptors.is_empty()
                && packed_batch_len
                    .checked_add(packed_len)
                    .ok_or(SaltV2PackageError::LengthOverflow)?
                    > MAX_BATCH_PAYLOAD_BYTES
            {
                break;
            }
            let scale_len = logical_len
                .div_ceil(tensor.info.scale_group_size())
                .checked_mul(2)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            descriptors.push(PlaneDescriptor {
                tile_index,
                plane_index,
                plane_count,
                logical_len,
                packed_start: packed_batch_len,
                packed_len,
                scale_start: scale_batch_len,
                scale_len,
            });
            packed_batch_len = packed_batch_len
                .checked_add(packed_len)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            scale_batch_len = scale_batch_len
                .checked_add(scale_len)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            plane_index += 1;
            if plane_index == plane_count {
                tile_index += 1;
                plane_index = 0;
            }
        }

        reserve_and_resize(&mut payload, packed_batch_len)?;
        source.seek_abs(payload_position, "seek tensor payload batch")?;
        source.read_exact_chunks(&mut payload, "read tensor payload batch")?;
        payload_hasher.update(&payload);
        payload_position = payload_position
            .checked_add(packed_batch_len as u64)
            .ok_or(SaltV2PackageError::LengthOverflow)?;

        reserve_and_resize(&mut scales, scale_batch_len)?;
        source.seek_abs(scale_position, "seek tensor scale batch")?;
        source.read_exact_chunks(&mut scales, "read tensor scale batch")?;
        scale_hasher.update(&scales);
        scale_position = scale_position
            .checked_add(scale_batch_len as u64)
            .ok_or(SaltV2PackageError::LengthOverflow)?;

        for descriptor in &descriptors {
            let packed_end = descriptor
                .packed_start
                .checked_add(descriptor.packed_len)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            let packed = &payload[descriptor.packed_start..packed_end];
            unpack_semantic_plane_into(codec, packed, descriptor.logical_len, &mut trits)?;
            let scale_end = descriptor
                .scale_start
                .checked_add(descriptor.scale_len)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            let scale_bytes = &scales[descriptor.scale_start..scale_end];
            let mut decoded_scales = [f16::ZERO; 4];
            for (index, bytes) in scale_bytes.chunks_exact(2).enumerate() {
                decoded_scales[index] = f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]]));
            }
            let scale_count = scale_bytes.len() / 2;
            validate_scales(
                &trits,
                &decoded_scales[..scale_count],
                tensor.info.scale_group_size(),
            )?;
            if let Some(hasher) = &mut semantic_hasher {
                if descriptor.plane_index == 0 {
                    hasher.update_tile(
                        descriptor.tile_index,
                        descriptor.logical_len,
                        descriptor.plane_count,
                    );
                }
                hasher.update_plane(
                    descriptor.plane_index,
                    &trits,
                    &decoded_scales[..scale_count],
                );
            }
            if visitor(PackedSaltV2PlaneRef {
                tile_index: descriptor.tile_index,
                plane_index: descriptor.plane_index,
                plane_count: descriptor.plane_count,
                logical_len: descriptor.logical_len,
                packed,
                scales: &decoded_scales[..scale_count],
            })
            .is_break()
            {
                return Ok(None);
            }
        }
    }

    let expected_payload_end = tensor
        .payload_offset
        .checked_add(tensor.info.encoded_payload_bytes)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    let expected_scale_end = tensor
        .scales_offset
        .checked_add(tensor.info.encoded_scale_bytes)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    if payload_position != expected_payload_end {
        return Err(SaltV2PackageError::DeclaredFieldMismatch {
            field: "payload bytes",
            declared: tensor.info.encoded_payload_bytes,
            expected: payload_position.saturating_sub(tensor.payload_offset),
        }
        .into());
    }
    if scale_position != expected_scale_end {
        return Err(SaltV2PackageError::DeclaredFieldMismatch {
            field: "scale bytes",
            declared: tensor.info.encoded_scale_bytes,
            expected: scale_position.saturating_sub(tensor.scales_offset),
        }
        .into());
    }
    Ok(Some(TensorSectionDigest {
        payload: *payload_hasher.finalize().as_bytes(),
        scales: *scale_hasher.finalize().as_bytes(),
    }))
}

fn validate_scales(
    trits: &[Trit],
    scales: &[f16],
    scale_group_size: usize,
) -> Result<(), SaltV2PackageError> {
    let expected = trits.len().div_ceil(scale_group_size);
    if scales.len() != expected {
        return Err(SaltV2PackageError::WrongScaleCount {
            expected,
            got: scales.len(),
        });
    }
    for (group_index, scale) in scales.iter().copied().enumerate() {
        let value = scale.to_f32();
        if !value.is_finite() {
            return Err(SaltV2PackageError::NonFiniteScale {
                group_index,
                bits: scale.to_bits(),
            });
        }
        if scale.to_bits() & 0x8000 != 0 {
            return Err(SaltV2PackageError::NegativeScale {
                group_index,
                bits: scale.to_bits(),
            });
        }
        let start = group_index * scale_group_size;
        let end = (start + scale_group_size).min(trits.len());
        if scale == f16::ZERO && trits[start..end].iter().any(|trit| !trit.is_zero()) {
            return Err(SaltV2PackageError::ZeroScaleForNonzeroGroup { group_index });
        }
    }
    Ok(())
}

fn indexed_runtime_ledger(
    tensor: &IndexedTensor,
    present_planes: usize,
) -> Result<SaltV2IndexedRuntimeLedger, SaltV2PackageError> {
    let allocation_tiles =
        u64::try_from(tensor.info.tile_count).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    let present_planes =
        u64::try_from(present_planes).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    let allocation_map_bits = allocation_tiles
        .checked_mul(2)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    let allocation_map_bytes = allocation_map_bits / 8;
    let allocation_map_embedded_bits = allocation_map_bits % 8;
    let rank_prefix_count =
        allocation_tiles.saturating_sub(1) / SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES as u64;
    let rank_prefix_bytes = rank_prefix_count
        .checked_mul(SALT_V2_INDEXED_RUNTIME_RANK_PREFIX_BYTES as u64)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    for value in [
        allocation_tiles,
        present_planes,
        tensor.info.encoded_payload_bytes,
        tensor.info.encoded_scale_bytes / 2,
    ] {
        u32::try_from(value).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    }
    let steady_resident_bytes = tensor
        .info
        .encoded_payload_bytes
        .checked_add(tensor.info.encoded_scale_bytes)
        .and_then(|bytes| bytes.checked_add(allocation_map_bytes))
        .and_then(|bytes| bytes.checked_add(rank_prefix_bytes))
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    Ok(SaltV2IndexedRuntimeLedger {
        payload_bytes: tensor.info.encoded_payload_bytes,
        scale_bytes: tensor.info.encoded_scale_bytes,
        allocation_map_bytes,
        rank_prefix_bytes,
        allocation_map_bits,
        allocation_map_embedded_bits,
        dense_shadow_bytes: 0,
        allocation_tiles,
        present_planes,
        steady_resident_bytes,
    })
}

fn hash_range<R: Read + Seek>(
    source: &mut Source<R>,
    offset: u64,
    len: u64,
    context: &str,
) -> Result<[u8; 32], SaltV2PackageReadError> {
    source.seek_abs(offset, context)?;
    let mut scratch = Vec::new();
    reserve_and_resize(
        &mut scratch,
        usize::try_from(len.min(MAX_READ_CHUNK_BYTES as u64)).unwrap_or(MAX_READ_CHUNK_BYTES),
    )?;
    let mut remaining = len;
    let mut hasher = blake3::Hasher::new();
    while remaining != 0 {
        let chunk_len = usize::try_from(remaining.min(MAX_READ_CHUNK_BYTES as u64))
            .expect("chunk length fits usize");
        source.read_exact_chunks(&mut scratch[..chunk_len], context)?;
        hasher.update(&scratch[..chunk_len]);
        remaining -= chunk_len as u64;
    }
    Ok(*hasher.finalize().as_bytes())
}

fn range_matches<R: Read + Seek>(
    source: &mut Source<R>,
    offset: u64,
    expected: &[u8],
    context: &str,
) -> Result<bool, SaltV2PackageReadError> {
    source.seek_abs(offset, context)?;
    let mut scratch = Vec::new();
    reserve_and_resize(&mut scratch, expected.len().min(MAX_READ_CHUNK_BYTES))?;
    for expected_chunk in expected.chunks(MAX_READ_CHUNK_BYTES) {
        source.read_exact_chunks(&mut scratch[..expected_chunk.len()], context)?;
        if &scratch[..expected_chunk.len()] != expected_chunk {
            return Ok(false);
        }
    }
    Ok(true)
}

fn enforce_limit(resource: &str, actual: u64, limit: u64) -> Result<(), SaltV2PackageReadError> {
    if actual > limit {
        Err(limit_error(resource, actual, limit))
    } else {
        Ok(())
    }
}

fn limit_error(resource: &str, actual: u64, limit: u64) -> SaltV2PackageReadError {
    SaltV2PackageReadError::LimitExceeded {
        resource: resource.to_owned(),
        limit,
        actual,
    }
}

fn try_reserve_exact<T>(
    buffer: &mut Vec<T>,
    additional: usize,
) -> Result<(), SaltV2PackageReadError> {
    buffer
        .try_reserve_exact(additional)
        .map_err(|_| SaltV2PackageReadError::AllocationFailed {
            requested_bytes: additional.saturating_mul(core::mem::size_of::<T>()),
        })
}

fn reserve_and_resize(buffer: &mut Vec<u8>, len: usize) -> Result<(), SaltV2PackageReadError> {
    if len > buffer.len() {
        buffer.try_reserve_exact(len - buffer.len()).map_err(|_| {
            SaltV2PackageReadError::AllocationFailed {
                requested_bytes: len,
            }
        })?;
    }
    buffer.resize(len, 0);
    Ok(())
}

#[derive(Debug)]
struct Source<R> {
    inner: R,
    len: u64,
    position: u64,
    package_hasher: Option<PackageHasher>,
    range_hasher: Option<blake3::Hasher>,
}

impl<R: Read + Seek> Source<R> {
    fn new(mut inner: R) -> Result<Self, SaltV2PackageReadError> {
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
            package_hasher: None,
            range_hasher: None,
        })
    }

    fn start_package_hash(&mut self) {
        debug_assert_eq!(self.position, 0);
        debug_assert!(self.package_hasher.is_none());
        self.package_hasher = Some(PackageHasher::new());
    }

    fn finish_package_hash(&mut self) -> Result<PackageId, SaltV2PackageReadError> {
        if self.position != self.len {
            return Err(SaltV2PackageError::Truncated {
                needed: usize::try_from(self.len).unwrap_or(usize::MAX),
                remaining: usize::try_from(self.remaining()).unwrap_or(usize::MAX),
            }
            .into());
        }
        Ok(self
            .package_hasher
            .take()
            .expect("package hash starts before the first package byte")
            .finalize())
    }

    fn start_range_digest(&mut self) {
        debug_assert!(self.range_hasher.is_none());
        self.range_hasher = Some(blake3::Hasher::new());
    }

    fn finish_range_digest(&mut self) -> [u8; 32] {
        *self
            .range_hasher
            .take()
            .expect("range digest starts before metadata")
            .finalize()
            .as_bytes()
    }

    fn digest_next(&mut self, len: u64, context: &str) -> Result<[u8; 32], SaltV2PackageReadError> {
        let mut scratch = Vec::new();
        reserve_and_resize(
            &mut scratch,
            usize::try_from(len.min(MAX_READ_CHUNK_BYTES as u64)).unwrap_or(MAX_READ_CHUNK_BYTES),
        )?;
        let mut remaining = len;
        let mut hasher = blake3::Hasher::new();
        while remaining != 0 {
            let chunk_len = usize::try_from(remaining.min(MAX_READ_CHUNK_BYTES as u64))
                .expect("chunk length fits usize");
            self.read_exact_chunks(&mut scratch[..chunk_len], context)?;
            hasher.update(&scratch[..chunk_len]);
            remaining -= chunk_len as u64;
        }
        Ok(*hasher.finalize().as_bytes())
    }

    const fn len(&self) -> u64 {
        self.len
    }

    const fn position(&self) -> u64 {
        self.position
    }

    const fn remaining(&self) -> u64 {
        self.len - self.position
    }

    fn len_unchanged(&mut self) -> Result<bool, SaltV2PackageReadError> {
        let actual = self
            .inner
            .seek(SeekFrom::End(0))
            .map_err(|error| io_error("verify source length", error))?;
        self.position = actual;
        Ok(actual == self.len)
    }

    fn seek_abs(&mut self, position: u64, context: &str) -> Result<(), SaltV2PackageReadError> {
        if position > self.len {
            return Err(SaltV2PackageError::Truncated {
                needed: usize::try_from(position).unwrap_or(usize::MAX),
                remaining: usize::try_from(self.len).unwrap_or(usize::MAX),
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
    ) -> Result<(), SaltV2PackageReadError> {
        let end = self
            .position
            .checked_add(bytes.len() as u64)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        if end > self.len {
            return Err(SaltV2PackageError::Truncated {
                needed: bytes.len(),
                remaining: usize::try_from(self.remaining()).unwrap_or(usize::MAX),
            }
            .into());
        }
        for chunk in bytes.chunks_mut(MAX_READ_CHUNK_BYTES) {
            self.inner
                .read_exact(chunk)
                .map_err(|error| io_error(context, error))?;
            self.position += chunk.len() as u64;
            if let Some(hasher) = &mut self.package_hasher {
                hasher.update(chunk);
            }
            if let Some(hasher) = &mut self.range_hasher {
                hasher.update(chunk);
            }
        }
        Ok(())
    }

    fn array<const N: usize>(&mut self, context: &str) -> Result<[u8; N], SaltV2PackageReadError> {
        let mut bytes = [0u8; N];
        self.read_exact_chunks(&mut bytes, context)?;
        Ok(bytes)
    }

    fn array_vec(&mut self, len: usize, context: &str) -> Result<Vec<u8>, SaltV2PackageReadError> {
        let mut bytes = Vec::new();
        reserve_and_resize(&mut bytes, len)?;
        self.read_exact_chunks(&mut bytes, context)?;
        Ok(bytes)
    }

    fn u32(&mut self, context: &str) -> Result<u32, SaltV2PackageReadError> {
        Ok(u32::from_le_bytes(self.array::<4>(context)?))
    }

    fn u64(&mut self, context: &str) -> Result<u64, SaltV2PackageReadError> {
        Ok(u64::from_le_bytes(self.array::<8>(context)?))
    }

    fn string(&mut self, len: usize, context: &str) -> Result<String, SaltV2PackageReadError> {
        let bytes = self.array_vec(len, context)?;
        String::from_utf8(bytes).map_err(|_| SaltV2PackageError::InvalidTensorName.into())
    }
}

fn io_error(context: &str, error: std::io::Error) -> SaltV2PackageReadError {
    SaltV2PackageReadError::Io {
        context: context.to_owned(),
        kind: error.kind(),
        message: error.to_string(),
    }
}
