//! Bounded-memory, seek-backed canonical SALT V2 package writing.

use core::fmt;
use std::{
    collections::BTreeSet,
    io::{self, Seek, SeekFrom, Write},
};

use super::{
    SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_MAX_PLANES, SALT_V2_MAX_TENSORS,
    SALT_V2_PACKAGE_ALIGNMENT, SALT_V2_PACKAGE_HEADER_BYTES, SALT_V2_PACKAGE_MAGIC,
    SALT_V2_PACKAGE_VERSION, SALT_V2_SCALE_GROUP_SIZE, SALT_V2_TENSOR_COUNT_BITS,
    SALT_V2_TENSOR_HEADER_BYTES, SALT_V2_TILE_COUNT_BITS, SALT_V2_TRANSFORM_METADATA_BYTES,
    SaltV2Codec, SaltV2PackageError, SaltV2PackageLedger, SaltV2Tile, SaltV2Transform,
    alignment_padding, checked_dimension_product, codec_tag, pack_salt_v2_plane,
    plane_count_map_value, set_global_map_bit, stored_trit_count,
};

/// Immutable tensor metadata used to plan a streamed package before payload I/O.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2StreamTensorSpec {
    name: String,
    dims: Vec<u64>,
    transform: SaltV2Transform,
}

impl SaltV2StreamTensorSpec {
    /// Validate one named, positive-shape package tensor.
    ///
    /// # Errors
    /// Returns the same metadata errors as [`super::SaltV2Tensor`].
    pub fn new(
        name: impl Into<String>,
        dims: Vec<u64>,
        transform: SaltV2Transform,
    ) -> Result<Self, SaltV2PackageError> {
        let name = name.into();
        if name.is_empty() {
            return Err(SaltV2PackageError::EmptyTensorName);
        }
        if name.len() > u32::MAX as usize {
            return Err(SaltV2PackageError::TensorNameTooLong { got: name.len() });
        }
        if dims.is_empty() {
            return Err(SaltV2PackageError::EmptyDimensions);
        }
        if dims.len() > u32::MAX as usize {
            return Err(SaltV2PackageError::TooManyDimensions { got: dims.len() });
        }
        let _ = checked_dimension_product(&dims)?;
        Ok(Self {
            name,
            dims,
            transform,
        })
    }

    /// Canonical package tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Row-major semantic dimensions.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Transform identity carried by the tensor.
    #[must_use]
    pub const fn transform(&self) -> SaltV2Transform {
        self.transform
    }
}

#[derive(Debug)]
struct PlannedTensor {
    spec: SaltV2StreamTensorSpec,
    logical_coefficients: usize,
    tile_count: usize,
    full_tile_count: usize,
    ragged_plane_count: Option<u8>,
    full_tile_start: usize,
    header_offset: u64,
    payload_offset: u64,
    payload_bytes: u64,
    scales_offset: u64,
    scales_bytes: u64,
}

/// Exact canonical layout derived from tensor metadata and a flat selected-count stream.
///
/// The plan stores two bits per full allocation tile plus `O(tensors)` metadata. It never
/// retains ternary payloads, scales, or a dense tensor-sized shadow.
pub struct SaltV2PackageStreamPlan {
    codec: SaltV2Codec,
    tensors: Vec<PlannedTensor>,
    full_map_bytes: Vec<u8>,
    embedded_map_value: u8,
    packed_tensor_count: u32,
    map_offset: u64,
    ledger: SaltV2PackageLedger,
    tile_count: usize,
}

impl fmt::Debug for SaltV2PackageStreamPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SaltV2PackageStreamPlan")
            .field("codec", &self.codec)
            .field("tensor_count", &self.tensors.len())
            .field("tile_count", &self.tile_count)
            .field("allocation_map_storage_bytes", &self.full_map_bytes.len())
            .field("ledger", &self.ledger)
            .finish_non_exhaustive()
    }
}

impl SaltV2PackageStreamPlan {
    /// Plan one canonical package by consuming exactly one selected count per shape-derived tile.
    ///
    /// `plane_counts` follows tensor order and then allocation-tile order. Counts must be in
    /// `1..=3`; a short or long stream fails closed.
    ///
    /// # Errors
    /// Rejects invalid/duplicate metadata, invalid counts, length overflow, allocation failure,
    /// or a count stream whose length differs from the exact shape-derived tile count.
    pub fn new(
        codec: SaltV2Codec,
        specs: Vec<SaltV2StreamTensorSpec>,
        plane_counts: impl IntoIterator<Item = u8>,
    ) -> Result<Self, SaltV2PackageError> {
        if specs.is_empty() {
            return Err(SaltV2PackageError::EmptyPackage);
        }
        if specs.len() > SALT_V2_MAX_TENSORS {
            return Err(SaltV2PackageError::TooManyTensors { got: specs.len() });
        }
        let mut names = BTreeSet::new();
        for spec in &specs {
            if !names.insert(spec.name.clone()) {
                return Err(SaltV2PackageError::DuplicateTensorName(spec.name.clone()));
            }
        }

        let mut total_tiles = 0usize;
        let mut total_full_tiles = 0usize;
        let mut ragged_tensors = 0usize;
        let mut geometry = Vec::new();
        geometry
            .try_reserve_exact(specs.len())
            .map_err(|_| SaltV2PackageError::AllocationFailed)?;
        for spec in specs {
            let logical_u64 = checked_dimension_product(&spec.dims)?;
            let logical_coefficients = usize::try_from(logical_u64)
                .map_err(|_| SaltV2PackageError::DimensionProductTooLarge(logical_u64))?;
            let tile_count = logical_coefficients.div_ceil(SALT_V2_ALLOCATION_TILE_SIZE);
            let full_tile_count = logical_coefficients / SALT_V2_ALLOCATION_TILE_SIZE;
            total_tiles = total_tiles
                .checked_add(tile_count)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            total_full_tiles = total_full_tiles
                .checked_add(full_tile_count)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            ragged_tensors = ragged_tensors
                .checked_add(usize::from(tile_count != full_tile_count))
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            geometry.push((spec, logical_coefficients, tile_count, full_tile_count));
        }

        let full_map_bits = total_full_tiles
            .checked_mul(2)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let complete_map_bits = full_map_bits / 8 * 8;
        let mut full_map_bytes = Vec::new();
        full_map_bytes
            .try_reserve_exact(full_map_bits / 8)
            .map_err(|_| SaltV2PackageError::AllocationFailed)?;
        full_map_bytes.resize(full_map_bits / 8, 0);
        let mut embedded_map_value = 0u8;
        let mut counts = plane_counts.into_iter();
        let mut consumed_counts = 0usize;
        let mut full_tile_ordinal = 0usize;
        let mut planned = Vec::new();
        planned
            .try_reserve_exact(geometry.len())
            .map_err(|_| SaltV2PackageError::AllocationFailed)?;
        let mut ledger = SaltV2PackageLedger {
            headers_bytes: SALT_V2_PACKAGE_HEADER_BYTES as u64,
            maps_bytes: u64::try_from(full_map_bytes.len())
                .map_err(|_| SaltV2PackageError::LengthOverflow)?,
            allocation_map_bits: u64::try_from(
                total_tiles
                    .checked_mul(2)
                    .ok_or(SaltV2PackageError::LengthOverflow)?,
            )
            .map_err(|_| SaltV2PackageError::LengthOverflow)?,
            allocation_map_package_embedded_bits: (full_map_bits % 8) as u8,
            allocation_map_tensor_embedded_bits: u64::try_from(
                ragged_tensors
                    .checked_mul(2)
                    .ok_or(SaltV2PackageError::LengthOverflow)?,
            )
            .map_err(|_| SaltV2PackageError::LengthOverflow)?,
            allocation_tiles: u64::try_from(total_tiles)
                .map_err(|_| SaltV2PackageError::LengthOverflow)?,
            allocation_capacity_coefficients: u64::try_from(total_tiles)
                .map_err(|_| SaltV2PackageError::LengthOverflow)?
                .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE as u64)
                .ok_or(SaltV2PackageError::LengthOverflow)?,
            ..SaltV2PackageLedger::default()
        };
        ledger.allocation_map_embedded_bits =
            u64::from(ledger.allocation_map_package_embedded_bits)
                .checked_add(ledger.allocation_map_tensor_embedded_bits)
                .ok_or(SaltV2PackageError::LengthOverflow)?;

        for (spec, logical_coefficients, tile_count, full_tile_count) in geometry {
            let tensor_full_start = full_tile_ordinal;
            let mut payload_bytes = 0u64;
            let mut scales_bytes = 0u64;
            let mut codec_padding_trits = 0u64;
            let mut codec_padding_bits = 0u64;
            let mut ragged_plane_count = None;
            for tile_index in 0..tile_count {
                let count = counts.next().ok_or(SaltV2PackageError::WrongTileCount {
                    expected: total_tiles,
                    got: consumed_counts,
                })?;
                consumed_counts = consumed_counts
                    .checked_add(1)
                    .ok_or(SaltV2PackageError::LengthOverflow)?;
                if !(1..=SALT_V2_MAX_PLANES as u8).contains(&count) {
                    return Err(SaltV2PackageError::InvalidPlaneCount {
                        got: usize::from(count),
                    });
                }
                let consumed = tile_index
                    .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
                    .ok_or(SaltV2PackageError::LengthOverflow)?;
                let logical_len =
                    (logical_coefficients - consumed).min(SALT_V2_ALLOCATION_TILE_SIZE);
                let stored_trits = stored_trit_count(codec, logical_len)?;
                let plane_ledger = codec.ledger(stored_trits)?;
                let count_u64 = u64::from(count);
                payload_bytes = payload_bytes
                    .checked_add(
                        u64::try_from(plane_ledger.physical_bytes)
                            .map_err(|_| SaltV2PackageError::LengthOverflow)?
                            .checked_mul(count_u64)
                            .ok_or(SaltV2PackageError::LengthOverflow)?,
                    )
                    .ok_or(SaltV2PackageError::LengthOverflow)?;
                scales_bytes = scales_bytes
                    .checked_add(
                        u64::try_from(logical_len.div_ceil(SALT_V2_SCALE_GROUP_SIZE))
                            .map_err(|_| SaltV2PackageError::LengthOverflow)?
                            .checked_mul(2)
                            .and_then(|bytes| bytes.checked_mul(count_u64))
                            .ok_or(SaltV2PackageError::LengthOverflow)?,
                    )
                    .ok_or(SaltV2PackageError::LengthOverflow)?;
                codec_padding_trits = codec_padding_trits
                    .checked_add(
                        u64::try_from(
                            stored_trits - logical_len + plane_ledger.canonical_padding_trits,
                        )
                        .map_err(|_| SaltV2PackageError::LengthOverflow)?
                        .checked_mul(count_u64)
                        .ok_or(SaltV2PackageError::LengthOverflow)?,
                    )
                    .ok_or(SaltV2PackageError::LengthOverflow)?;
                codec_padding_bits = codec_padding_bits
                    .checked_add(
                        u64::from(plane_ledger.canonical_padding_bits)
                            .checked_mul(count_u64)
                            .ok_or(SaltV2PackageError::LengthOverflow)?,
                    )
                    .ok_or(SaltV2PackageError::LengthOverflow)?;

                if tile_index < full_tile_count {
                    let map_value = plane_count_map_value(usize::from(count));
                    let bit = full_tile_ordinal
                        .checked_mul(2)
                        .ok_or(SaltV2PackageError::LengthOverflow)?;
                    if map_value & 0b01 != 0 {
                        set_global_map_bit(
                            &mut full_map_bytes,
                            &mut embedded_map_value,
                            complete_map_bits,
                            bit,
                        );
                    }
                    if map_value & 0b10 != 0 {
                        set_global_map_bit(
                            &mut full_map_bytes,
                            &mut embedded_map_value,
                            complete_map_bits,
                            bit + 1,
                        );
                    }
                    full_tile_ordinal += 1;
                } else {
                    ragged_plane_count = Some(count);
                }
            }
            ledger.headers_bytes = ledger
                .headers_bytes
                .checked_add(
                    u64::try_from(
                        (SALT_V2_TENSOR_HEADER_BYTES - SALT_V2_TRANSFORM_METADATA_BYTES)
                            .checked_add(spec.name.len())
                            .and_then(|bytes| bytes.checked_add(spec.dims.len().checked_mul(8)?))
                            .ok_or(SaltV2PackageError::LengthOverflow)?,
                    )
                    .map_err(|_| SaltV2PackageError::LengthOverflow)?,
                )
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            ledger.transform_bytes = ledger
                .transform_bytes
                .checked_add(SALT_V2_TRANSFORM_METADATA_BYTES as u64)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            ledger.payload_bytes = ledger
                .payload_bytes
                .checked_add(payload_bytes)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            ledger.scales_bytes = ledger
                .scales_bytes
                .checked_add(scales_bytes)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            ledger.codec_padding_trits = ledger
                .codec_padding_trits
                .checked_add(codec_padding_trits)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            ledger.codec_padding_bits = ledger
                .codec_padding_bits
                .checked_add(codec_padding_bits)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            planned.push(PlannedTensor {
                spec,
                logical_coefficients,
                tile_count,
                full_tile_count,
                ragged_plane_count,
                full_tile_start: tensor_full_start,
                header_offset: 0,
                payload_offset: 0,
                payload_bytes,
                scales_offset: 0,
                scales_bytes,
            });
        }
        if counts.next().is_some() {
            return Err(SaltV2PackageError::WrongTileCount {
                expected: total_tiles,
                got: total_tiles.saturating_add(1),
            });
        }
        debug_assert_eq!(full_tile_ordinal, total_full_tiles);

        let mut offset = SALT_V2_PACKAGE_HEADER_BYTES as u64;
        for tensor in &mut planned {
            tensor.header_offset = offset;
            let metadata_bytes = SALT_V2_TENSOR_HEADER_BYTES
                .checked_add(tensor.spec.name.len())
                .and_then(|bytes| bytes.checked_add(tensor.spec.dims.len().checked_mul(8)?))
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            tensor.payload_offset = offset
                .checked_add(
                    u64::try_from(metadata_bytes)
                        .map_err(|_| SaltV2PackageError::LengthOverflow)?,
                )
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            tensor.scales_offset = tensor
                .payload_offset
                .checked_add(tensor.payload_bytes)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            offset = tensor
                .scales_offset
                .checked_add(tensor.scales_bytes)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
        }
        let map_offset = offset;
        ledger.serialized_unpadded_bytes = ledger
            .headers_bytes
            .checked_add(ledger.transform_bytes)
            .and_then(|bytes| bytes.checked_add(ledger.maps_bytes))
            .and_then(|bytes| bytes.checked_add(ledger.payload_bytes))
            .and_then(|bytes| bytes.checked_add(ledger.scales_bytes))
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let raw_len = usize::try_from(ledger.serialized_unpadded_bytes)
            .map_err(|_| SaltV2PackageError::LengthOverflow)?;
        ledger.padding_bytes = u64::try_from(alignment_padding(raw_len))
            .map_err(|_| SaltV2PackageError::LengthOverflow)?;
        ledger.total_bytes = ledger
            .serialized_unpadded_bytes
            .checked_add(ledger.padding_bytes)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        debug_assert_eq!(offset + ledger.maps_bytes, ledger.serialized_unpadded_bytes);
        debug_assert_eq!(ledger.total_bytes % SALT_V2_PACKAGE_ALIGNMENT as u64, 0);

        let tensor_count =
            u32::try_from(planned.len()).map_err(|_| SaltV2PackageError::LengthOverflow)?;
        let packed_tensor_count =
            tensor_count | (u32::from(embedded_map_value) << SALT_V2_TENSOR_COUNT_BITS);
        Ok(Self {
            codec,
            tensors: planned,
            full_map_bytes,
            embedded_map_value,
            packed_tensor_count,
            map_offset,
            ledger,
            tile_count: total_tiles,
        })
    }

    /// Selected physical codec.
    #[must_use]
    pub const fn codec(&self) -> SaltV2Codec {
        self.codec
    }

    /// Exact final package component ledger.
    #[must_use]
    pub const fn ledger(&self) -> SaltV2PackageLedger {
        self.ledger
    }

    /// Number of semantic tensors in package order.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Number of allocation tiles across the package.
    #[must_use]
    pub const fn tile_count(&self) -> usize {
        self.tile_count
    }

    /// Bytes retained by the packed full-tile allocation map.
    #[must_use]
    pub fn allocation_map_storage_bytes(&self) -> usize {
        self.full_map_bytes.len()
    }

    fn full_plane_count(&self, tile: usize) -> usize {
        let complete_bits = self.full_map_bytes.len() * 8;
        let bit = tile * 2;
        let map_bit = |index: usize| {
            if index < complete_bits {
                self.full_map_bytes[index / 8] & (1 << (index % 8)) != 0
            } else {
                self.embedded_map_value & (1 << (index - complete_bits)) != 0
            }
        };
        1 + usize::from(map_bit(bit)) + usize::from(map_bit(bit + 1))
    }
}

/// Failure while initializing or incrementally filling a seek-backed package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SaltV2PackageStreamError {
    /// Semantic/package invariant failed.
    Package(SaltV2PackageError),
    /// Portable I/O category from the caller-owned destination.
    Io(io::ErrorKind),
    /// Destination already contained bytes; canonical truncation is not implicit.
    OutputNotEmpty {
        /// Existing destination length.
        bytes: u64,
    },
    /// More tiles were supplied than the immutable plan admits.
    TooManyTiles,
    /// Fewer tiles were supplied than the immutable plan requires.
    TooFewTiles {
        /// Required total.
        expected: usize,
        /// Successfully written total.
        actual: usize,
    },
    /// Supplied tile prefix width differed from the selected map.
    PlaneCountMismatch {
        /// Global package tile ordinal.
        tile: usize,
        /// Selected plane count.
        expected: usize,
        /// Supplied plane count.
        actual: usize,
    },
}

impl fmt::Display for SaltV2PackageStreamError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Package(error) => write!(formatter, "streamed SALT V2 package: {error}"),
            Self::Io(kind) => write!(formatter, "streamed SALT V2 package I/O failed: {kind:?}"),
            Self::OutputNotEmpty { bytes } => {
                write!(
                    formatter,
                    "streamed SALT V2 output already contains {bytes} bytes"
                )
            }
            Self::TooManyTiles => formatter.write_str("streamed SALT V2 source has extra tiles"),
            Self::TooFewTiles { expected, actual } => write!(
                formatter,
                "streamed SALT V2 source supplied {actual} tiles, expected {expected}"
            ),
            Self::PlaneCountMismatch {
                tile,
                expected,
                actual,
            } => write!(
                formatter,
                "streamed SALT V2 tile {tile} has {actual} planes, selected map requires {expected}"
            ),
        }
    }
}

impl std::error::Error for SaltV2PackageStreamError {}

impl From<SaltV2PackageError> for SaltV2PackageStreamError {
    fn from(error: SaltV2PackageError) -> Self {
        Self::Package(error)
    }
}

fn stream_io(error: io::Error) -> SaltV2PackageStreamError {
    SaltV2PackageStreamError::Io(error.kind())
}

/// Incremental canonical writer that keeps no model-sized payload in memory.
#[derive(Debug)]
pub struct SaltV2PackageStreamWriter<W> {
    output: W,
    plan: SaltV2PackageStreamPlan,
    tensor_index: usize,
    local_tile: usize,
    written_tiles: usize,
    payload_cursor: u64,
    scales_cursor: u64,
}

impl<W: Write + Seek> SaltV2PackageStreamWriter<W> {
    /// Initialize all immutable headers and the packed map in an empty destination.
    ///
    /// Payload and scale regions are subsequently filled by [`Self::push_tile`]. The writer
    /// seeks between those two canonical tensor sections, so only one tile is resident at once.
    pub fn new(
        mut output: W,
        plan: SaltV2PackageStreamPlan,
    ) -> Result<Self, SaltV2PackageStreamError> {
        let existing = output.seek(SeekFrom::End(0)).map_err(stream_io)?;
        if existing != 0 {
            return Err(SaltV2PackageStreamError::OutputNotEmpty { bytes: existing });
        }
        output.seek(SeekFrom::Start(0)).map_err(stream_io)?;
        output
            .write_all(&SALT_V2_PACKAGE_MAGIC)
            .map_err(stream_io)?;
        output
            .write_all(&SALT_V2_PACKAGE_VERSION.to_le_bytes())
            .map_err(stream_io)?;
        output
            .write_all(&[codec_tag(plan.codec), 0])
            .map_err(stream_io)?;
        output
            .write_all(&plan.packed_tensor_count.to_le_bytes())
            .map_err(stream_io)?;
        output
            .write_all(&plan.ledger.total_bytes.to_le_bytes())
            .map_err(stream_io)?;

        for tensor in &plan.tensors {
            output
                .seek(SeekFrom::Start(tensor.header_offset))
                .map_err(stream_io)?;
            let name_len = u32::try_from(tensor.spec.name.len())
                .map_err(|_| SaltV2PackageError::LengthOverflow)?;
            let rank = u32::try_from(tensor.spec.dims.len())
                .map_err(|_| SaltV2PackageError::LengthOverflow)?;
            let tile_count =
                u64::try_from(tensor.tile_count).map_err(|_| SaltV2PackageError::LengthOverflow)?;
            let packed_tile_count = tile_count
                | (u64::from(
                    tensor
                        .ragged_plane_count
                        .map_or(0, |count| plane_count_map_value(usize::from(count))),
                ) << SALT_V2_TILE_COUNT_BITS);
            output
                .write_all(&name_len.to_le_bytes())
                .map_err(stream_io)?;
            output.write_all(&rank.to_le_bytes()).map_err(stream_io)?;
            output
                .write_all(
                    &u64::try_from(tensor.logical_coefficients)
                        .map_err(|_| SaltV2PackageError::LengthOverflow)?
                        .to_le_bytes(),
                )
                .map_err(stream_io)?;
            output
                .write_all(&packed_tile_count.to_le_bytes())
                .map_err(stream_io)?;
            output
                .write_all(&tensor.payload_bytes.to_le_bytes())
                .map_err(stream_io)?;
            output
                .write_all(&tensor.scales_bytes.to_le_bytes())
                .map_err(stream_io)?;
            write_transform(&mut output, tensor.spec.transform)?;
            output
                .write_all(tensor.spec.name.as_bytes())
                .map_err(stream_io)?;
            for dimension in &tensor.spec.dims {
                output
                    .write_all(&dimension.to_le_bytes())
                    .map_err(stream_io)?;
            }
        }
        output
            .seek(SeekFrom::Start(plan.map_offset))
            .map_err(stream_io)?;
        output.write_all(&plan.full_map_bytes).map_err(stream_io)?;
        let padding = usize::try_from(plan.ledger.padding_bytes)
            .map_err(|_| SaltV2PackageError::LengthOverflow)?;
        output.write_all(&vec![0; padding]).map_err(stream_io)?;

        let (payload_cursor, scales_cursor) = plan.tensors.first().map_or((0, 0), |tensor| {
            (tensor.payload_offset, tensor.scales_offset)
        });
        Ok(Self {
            output,
            plan,
            tensor_index: 0,
            local_tile: 0,
            written_tiles: 0,
            payload_cursor,
            scales_cursor,
        })
    }

    /// Write the next selected semantic tile in package order.
    pub fn push_tile(&mut self, tile: &SaltV2Tile) -> Result<(), SaltV2PackageStreamError> {
        let tensor = self
            .plan
            .tensors
            .get(self.tensor_index)
            .ok_or(SaltV2PackageStreamError::TooManyTiles)?;
        let expected_planes = if self.local_tile < tensor.full_tile_count {
            self.plan
                .full_plane_count(tensor.full_tile_start + self.local_tile)
        } else {
            usize::from(
                tensor
                    .ragged_plane_count
                    .expect("only the final ragged tile is not full"),
            )
        };
        if tile.planes().len() != expected_planes {
            return Err(SaltV2PackageStreamError::PlaneCountMismatch {
                tile: self.written_tiles,
                expected: expected_planes,
                actual: tile.planes().len(),
            });
        }
        let consumed = self
            .local_tile
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let expected_len =
            (tensor.logical_coefficients - consumed).min(SALT_V2_ALLOCATION_TILE_SIZE);
        if tile.logical_len() != expected_len {
            return Err(SaltV2PackageError::WrongTileLength {
                tile_index: self.local_tile,
                expected: expected_len,
                got: tile.logical_len(),
            }
            .into());
        }

        for plane in tile.planes() {
            let packed = pack_salt_v2_plane(self.plan.codec, plane.trits())?;
            self.output
                .seek(SeekFrom::Start(self.payload_cursor))
                .map_err(stream_io)?;
            self.output.write_all(&packed).map_err(stream_io)?;
            self.payload_cursor = self
                .payload_cursor
                .checked_add(
                    u64::try_from(packed.len()).map_err(|_| SaltV2PackageError::LengthOverflow)?,
                )
                .ok_or(SaltV2PackageError::LengthOverflow)?;

            self.output
                .seek(SeekFrom::Start(self.scales_cursor))
                .map_err(stream_io)?;
            for scale in plane.scales() {
                self.output
                    .write_all(&scale.to_bits().to_le_bytes())
                    .map_err(stream_io)?;
            }
            self.scales_cursor = self
                .scales_cursor
                .checked_add(
                    u64::try_from(plane.scales().len())
                        .map_err(|_| SaltV2PackageError::LengthOverflow)?
                        .checked_mul(2)
                        .ok_or(SaltV2PackageError::LengthOverflow)?,
                )
                .ok_or(SaltV2PackageError::LengthOverflow)?;
        }
        self.local_tile += 1;
        self.written_tiles += 1;
        if self.local_tile == tensor.tile_count {
            if self.payload_cursor != tensor.payload_offset + tensor.payload_bytes
                || self.scales_cursor != tensor.scales_offset + tensor.scales_bytes
            {
                return Err(SaltV2PackageError::LengthOverflow.into());
            }
            self.tensor_index += 1;
            self.local_tile = 0;
            if let Some(next) = self.plan.tensors.get(self.tensor_index) {
                self.payload_cursor = next.payload_offset;
                self.scales_cursor = next.scales_offset;
            }
        }
        Ok(())
    }

    /// Require every planned tile, flush the destination, and return it with the exact ledger.
    pub fn finish(mut self) -> Result<(W, SaltV2PackageLedger), SaltV2PackageStreamError> {
        if self.written_tiles != self.plan.tile_count {
            return Err(SaltV2PackageStreamError::TooFewTiles {
                expected: self.plan.tile_count,
                actual: self.written_tiles,
            });
        }
        self.output.flush().map_err(stream_io)?;
        let actual = self.output.seek(SeekFrom::End(0)).map_err(stream_io)?;
        if actual != self.plan.ledger.total_bytes {
            return Err(SaltV2PackageError::DeclaredFieldMismatch {
                field: "streamed package total bytes",
                declared: actual,
                expected: self.plan.ledger.total_bytes,
            }
            .into());
        }
        Ok((self.output, self.plan.ledger))
    }
}

fn write_transform(
    output: &mut (impl Write + ?Sized),
    transform: SaltV2Transform,
) -> Result<(), SaltV2PackageStreamError> {
    match transform {
        SaltV2Transform::None => {
            output
                .write_all(&[0, 0, 0, 0, 0, 0, 0, 0])
                .map_err(stream_io)?;
            output.write_all(&0u64.to_le_bytes()).map_err(stream_io)?;
            output.write_all(&0u64.to_le_bytes()).map_err(stream_io)?;
        }
        SaltV2Transform::SignedRht { seed, domain } => {
            output
                .write_all(&[1, 0, 0, 0, 0, 0, 0, 0])
                .map_err(stream_io)?;
            output.write_all(&seed.to_le_bytes()).map_err(stream_io)?;
            output.write_all(&domain.to_le_bytes()).map_err(stream_io)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use crate::salt_v2_package::{
        SaltV2Package, SaltV2Plane, SaltV2Tensor, read_salt_v2_package, write_salt_v2_package,
    };
    use half::f16;

    fn plane(len: usize, phase: usize) -> SaltV2Plane {
        let values = (0..len)
            .map(|index| {
                if index % 4 == 0 {
                    0
                } else if (index + phase).is_multiple_of(2) {
                    1
                } else {
                    -1
                }
            })
            .collect();
        let scales = (0..len.div_ceil(SALT_V2_SCALE_GROUP_SIZE))
            .map(|group| f16::from_f32(0.5 + (group + phase) as f32 / 16.0))
            .collect();
        SaltV2Plane::new(values, scales).expect("plane")
    }

    fn tile(len: usize, count: usize, phase: usize) -> SaltV2Tile {
        SaltV2Tile::new((0..count).map(|index| plane(len, phase + index)).collect()).expect("tile")
    }

    fn fixture(codec: SaltV2Codec) -> SaltV2Package {
        SaltV2Package::new(
            codec,
            vec![
                SaltV2Tensor::new_with_transform(
                    "first",
                    vec![2, 256],
                    SaltV2Transform::SignedRht { seed: 7, domain: 9 },
                    vec![tile(256, 1, 0), tile(256, 3, 1)],
                )
                .expect("first"),
                SaltV2Tensor::new(
                    "ragged",
                    vec![599],
                    vec![tile(256, 2, 2), tile(256, 3, 3), tile(87, 1, 4)],
                )
                .expect("ragged"),
            ],
        )
        .expect("package")
    }

    #[test]
    fn seek_writer_is_byte_identical_to_canonical_encoder_for_every_codec() {
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let package = fixture(codec);
            let canonical = write_salt_v2_package(&package).expect("canonical");
            let specs = package
                .tensors()
                .iter()
                .map(|tensor| {
                    SaltV2StreamTensorSpec::new(
                        tensor.name(),
                        tensor.dims().to_vec(),
                        tensor.transform(),
                    )
                    .expect("spec")
                })
                .collect();
            let counts = package
                .tensors()
                .iter()
                .flat_map(|tensor| tensor.tiles().iter())
                .map(|tile| tile.planes().len() as u8);
            let plan = SaltV2PackageStreamPlan::new(codec, specs, counts).expect("plan");
            assert_eq!(plan.ledger(), canonical.ledger);
            assert_eq!(plan.allocation_map_storage_bytes(), 1);

            let mut writer =
                SaltV2PackageStreamWriter::new(Cursor::new(Vec::new()), plan).expect("writer");
            for tile in package.tensors().iter().flat_map(|tensor| tensor.tiles()) {
                writer.push_tile(tile).expect("push");
            }
            let (output, ledger) = writer.finish().expect("finish");
            assert_eq!(ledger, canonical.ledger);
            assert_eq!(output.into_inner(), canonical.bytes, "{codec:?}");
        }
    }

    #[test]
    fn plan_and_writer_fail_closed_on_count_or_tile_mismatch() {
        let spec =
            SaltV2StreamTensorSpec::new("one", vec![257], SaltV2Transform::None).expect("spec");
        assert!(matches!(
            SaltV2PackageStreamPlan::new(SaltV2Codec::D2, vec![spec.clone()], [1]),
            Err(SaltV2PackageError::WrongTileCount {
                expected: 2,
                got: 1
            })
        ));
        assert!(matches!(
            SaltV2PackageStreamPlan::new(SaltV2Codec::D2, vec![spec.clone()], [1, 1, 1]),
            Err(SaltV2PackageError::WrongTileCount {
                expected: 2,
                got: 3
            })
        ));

        let plan = SaltV2PackageStreamPlan::new(SaltV2Codec::D2, vec![spec], [1, 2]).expect("plan");
        let mut writer =
            SaltV2PackageStreamWriter::new(Cursor::new(Vec::new()), plan).expect("writer");
        assert!(matches!(
            writer.push_tile(&tile(256, 2, 0)),
            Err(SaltV2PackageStreamError::PlaneCountMismatch {
                tile: 0,
                expected: 1,
                actual: 2
            })
        ));
        assert!(matches!(
            writer.finish(),
            Err(SaltV2PackageStreamError::TooFewTiles {
                expected: 2,
                actual: 0
            })
        ));
    }

    #[test]
    fn streamed_package_remains_strictly_readable() {
        let package = fixture(SaltV2Codec::B3);
        let specs = package
            .tensors()
            .iter()
            .map(|tensor| {
                SaltV2StreamTensorSpec::new(
                    tensor.name(),
                    tensor.dims().to_vec(),
                    tensor.transform(),
                )
                .expect("spec")
            })
            .collect();
        let counts = package
            .tensors()
            .iter()
            .flat_map(|tensor| tensor.tiles())
            .map(|tile| tile.planes().len() as u8);
        let plan = SaltV2PackageStreamPlan::new(SaltV2Codec::B3, specs, counts).expect("plan");
        let mut writer =
            SaltV2PackageStreamWriter::new(Cursor::new(Vec::new()), plan).expect("writer");
        for tile in package.tensors().iter().flat_map(|tensor| tensor.tiles()) {
            writer.push_tile(tile).expect("push");
        }
        let (output, _) = writer.finish().expect("finish");
        let decoded = read_salt_v2_package(&output.into_inner()).expect("strict read");
        assert_eq!(decoded.package, package);
    }

    #[test]
    fn plan_storage_is_two_bits_per_full_tile_not_payload_sized() {
        let tile_count = 1usize << 16;
        let coefficients = tile_count * SALT_V2_ALLOCATION_TILE_SIZE;
        let spec = SaltV2StreamTensorSpec::new(
            "model-scale-shape",
            vec![coefficients as u64],
            SaltV2Transform::None,
        )
        .expect("spec");
        let plan = SaltV2PackageStreamPlan::new(
            SaltV2Codec::B3,
            vec![spec],
            core::iter::repeat_n(3, tile_count),
        )
        .expect("plan");

        assert_eq!(plan.tile_count(), tile_count);
        assert_eq!(plan.allocation_map_storage_bytes(), tile_count * 2 / 8);
        assert!(plan.ledger().payload_bytes > plan.allocation_map_storage_bytes() as u64 * 500);
        let debug = format!("{plan:?}");
        assert!(!debug.contains("255, 255"));
    }
}
