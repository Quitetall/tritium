//! Canonical SALT V2 tensor/package encoding.
//!
//! The package is deliberately semantic: a tensor is split into 256-coefficient
//! allocation macrotiles, every plane has one zero-point-free f16 scale per 128
//! coefficients, and the two optional planes are described by one package-global
//! two-bit stream for full allocation tiles. Complete map bytes are serialized
//! once; its terminal 0/2/4/6 bits use unused high bits of the mandatory package
//! tensor-count word. A ragged tensor's final two map bits use unused high bits of
//! its mandatory tile-count word. Both embedded classes are reported explicitly.
//! Only present planes occupy payload or scale bytes.

use core::fmt;
use std::collections::BTreeSet;

use half::f16;
use tritium_core::Trit;

use crate::salt_v2::{
    SaltV2Codec, SaltV2CodecError, pack_b3, pack_d2, pack_s34, unpack_b3_into, unpack_d2_into,
    unpack_s34_into,
};
use crate::{SemanticTensor, SemanticTensorHasher};

mod reader;

pub use reader::{
    PackedSaltV2PlaneRef, SaltV2PackageReadError, SaltV2PackageReader, SaltV2TensorInfo,
};

/// Number of coefficients sharing one SALT V2 scale.
pub const SALT_V2_SCALE_GROUP_SIZE: usize = 128;

/// Number of coefficients in one variable-plane allocation macrotile.
pub const SALT_V2_ALLOCATION_TILE_SIZE: usize = 256;

/// Maximum number of additive ternary planes in SALT V2.
pub const SALT_V2_MAX_PLANES: usize = 3;

/// Low bits of the package tensor-count word reserved for the actual count.
pub const SALT_V2_TENSOR_COUNT_BITS: u32 = 26;

/// High tensor-count-word bits available for the terminal global-map fragment.
pub const SALT_V2_EMBEDDED_MAP_CAPACITY_BITS: u32 = 32 - SALT_V2_TENSOR_COUNT_BITS;

/// Maximum tensor count representable alongside terminal map bits.
pub const SALT_V2_MAX_TENSORS: usize = (1usize << SALT_V2_TENSOR_COUNT_BITS) - 1;

/// Low bits of each tensor tile-count word reserved for the actual count.
pub const SALT_V2_TILE_COUNT_BITS: u32 = 62;

/// Canonical package magic for the SALT V2 semantic container.
pub const SALT_V2_PACKAGE_MAGIC: [u8; 8] = *b"TSLT2PKG";

/// Current SALT V2 semantic package version.
pub const SALT_V2_PACKAGE_VERSION: u16 = 1;

/// Bytes in the fixed package header.
pub const SALT_V2_PACKAGE_HEADER_BYTES: usize = 24;

/// Bytes in each fixed tensor header, excluding its name and dimensions.
pub const SALT_V2_TENSOR_HEADER_BYTES: usize = 64;

/// Bytes used by the fixed transform tag, reserved bytes, seed, and domain.
pub const SALT_V2_TRANSFORM_METADATA_BYTES: usize = 24;

/// Allocation tiles covered by one indexed-runtime plane-rank prefix.
pub const SALT_V2_INDEXED_RUNTIME_RANK_STRIDE_TILES: usize = 256;

/// Bytes in one indexed-runtime plane-rank prefix.
pub const SALT_V2_INDEXED_RUNTIME_RANK_PREFIX_BYTES: usize = core::mem::size_of::<u32>();

/// Canonical package alignment.
pub const SALT_V2_PACKAGE_ALIGNMENT: usize = 8;

/// Domain and encoding version for codec-independent SALT V2 tensor semantics.
const SALT_V2_SEMANTIC_TENSOR_DOMAIN: &[u8] = b"tritium.salt-v2.semantic-tensor.v1\0";

/// A validated, zero-point-free additive ternary plane for one allocation tile.
#[derive(Clone, Debug, PartialEq)]
pub struct SaltV2Plane {
    trits: Vec<Trit>,
    scales: Vec<f16>,
}

impl SaltV2Plane {
    /// Construct a plane from raw ternary values and one f16 scale per group128.
    ///
    /// # Errors
    /// Rejects values outside `{-1, 0, +1}`, an empty or overlarge plane, an
    /// inconsistent scale count, a non-finite or negative scale, and a zero
    /// scale whose group contains a nonzero trit.
    pub fn new(raw_trits: Vec<i8>, scales: Vec<f16>) -> Result<Self, SaltV2PackageError> {
        if raw_trits.is_empty() || raw_trits.len() > SALT_V2_ALLOCATION_TILE_SIZE {
            return Err(SaltV2PackageError::InvalidPlaneLength {
                got: raw_trits.len(),
            });
        }

        let mut trits = Vec::new();
        trits
            .try_reserve_exact(raw_trits.len())
            .map_err(|_| SaltV2PackageError::AllocationFailed)?;
        for (index, value) in raw_trits.into_iter().enumerate() {
            trits.push(
                Trit::from_i8(value)
                    .map_err(|_| SaltV2PackageError::NonCanonicalTrit { index, value })?,
            );
        }

        let expected_scales = trits.len().div_ceil(SALT_V2_SCALE_GROUP_SIZE);
        if scales.len() != expected_scales {
            return Err(SaltV2PackageError::WrongScaleCount {
                expected: expected_scales,
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
            let start = group_index * SALT_V2_SCALE_GROUP_SIZE;
            let end = (start + SALT_V2_SCALE_GROUP_SIZE).min(trits.len());
            if scale == f16::ZERO && trits[start..end].iter().any(|trit| !trit.is_zero()) {
                return Err(SaltV2PackageError::ZeroScaleForNonzeroGroup { group_index });
            }
        }

        Ok(Self { trits, scales })
    }

    /// Logical ternary coefficients in this plane.
    #[must_use]
    pub fn trits(&self) -> &[Trit] {
        &self.trits
    }

    /// Group128 f16 scales in this plane.
    #[must_use]
    pub fn scales(&self) -> &[f16] {
        &self.scales
    }
}

/// One 256-coefficient allocation tile with one to three nested planes.
#[derive(Clone, Debug, PartialEq)]
pub struct SaltV2Tile {
    planes: Vec<SaltV2Plane>,
}

impl SaltV2Tile {
    /// Construct a tile, requiring one to three equally sized planes.
    ///
    /// Plane zero is always present. Plane two cannot exist without plane one
    /// because planes are supplied as a dense prefix.
    ///
    /// # Errors
    /// Rejects an invalid plane count or inconsistent logical plane lengths.
    pub fn new(planes: Vec<SaltV2Plane>) -> Result<Self, SaltV2PackageError> {
        if !(1..=SALT_V2_MAX_PLANES).contains(&planes.len()) {
            return Err(SaltV2PackageError::InvalidPlaneCount { got: planes.len() });
        }
        let expected = planes[0].trits.len();
        if let Some((plane_index, plane)) = planes
            .iter()
            .enumerate()
            .find(|(_, plane)| plane.trits.len() != expected)
        {
            return Err(SaltV2PackageError::InconsistentPlaneLength {
                plane_index,
                expected,
                got: plane.trits.len(),
            });
        }
        Ok(Self { planes })
    }

    /// Number of logical coefficients in the tile.
    #[must_use]
    pub fn logical_len(&self) -> usize {
        self.planes[0].trits.len()
    }

    /// Nested additive planes, always a nonempty prefix of length one to three.
    #[must_use]
    pub fn planes(&self) -> &[SaltV2Plane] {
        &self.planes
    }
}

/// Zero-cost transform identity bound to a SALT V2 tensor.
///
/// This module records transform identity only. Applying a signed randomized
/// Hadamard transform belongs to the fitter/runtime contract.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2Transform {
    /// No transform is applied.
    #[default]
    None,
    /// Deterministic signed randomized Hadamard transform identity.
    SignedRht {
        /// Pseudorandom sign seed.
        seed: u64,
        /// Domain-separation identity for the transform site.
        domain: u64,
    },
}

/// Incremental canonical encoder for one codec-independent SALT V2 tensor.
///
/// The stream deliberately contains semantic geometry, allocation structure,
/// decoded trits, and exact scale bits. It does not contain package codec,
/// transport padding, presence-map representation, offsets, or record order.
/// Counts, indices, dimensions, and transform parameters are little-endian u64
/// values; tags and trits are single bytes. After the fixed domain come the
/// transform tag and any seed/domain parameters, rank and dimensions, logical
/// coefficient and tile counts, then each tile's index, length, and plane count.
/// Each ordered plane contributes its index, trit count, trits encoded as
/// `-1 => 0`, `0 => 1`, `+1 => 2`, scale count, and exact f16 scale bits.
struct SaltV2SemanticTensorHasher {
    inner: SemanticTensorHasher,
}

impl SaltV2SemanticTensorHasher {
    fn new(
        name: &str,
        dims: &[u64],
        logical_coefficients: usize,
        transform: SaltV2Transform,
        tile_count: usize,
    ) -> Self {
        Self::with_inner(
            SemanticTensorHasher::new(name, dims.to_vec()),
            dims,
            logical_coefficients,
            transform,
            tile_count,
        )
    }

    fn new_content_only(
        dims: &[u64],
        logical_coefficients: usize,
        transform: SaltV2Transform,
        tile_count: usize,
    ) -> Self {
        Self::with_inner(
            SemanticTensorHasher::new_content_only(),
            dims,
            logical_coefficients,
            transform,
            tile_count,
        )
    }

    fn with_inner(
        mut inner: SemanticTensorHasher,
        dims: &[u64],
        logical_coefficients: usize,
        transform: SaltV2Transform,
        tile_count: usize,
    ) -> Self {
        inner.update(SALT_V2_SEMANTIC_TENSOR_DOMAIN);
        match transform {
            SaltV2Transform::None => inner.update(&[0]),
            SaltV2Transform::SignedRht { seed, domain } => {
                inner.update(&[1]);
                inner.update(&seed.to_le_bytes());
                inner.update(&domain.to_le_bytes());
            }
        }
        inner.update(&(dims.len() as u64).to_le_bytes());
        for &dimension in dims {
            inner.update(&dimension.to_le_bytes());
        }
        inner.update(&(logical_coefficients as u64).to_le_bytes());
        inner.update(&(tile_count as u64).to_le_bytes());
        Self { inner }
    }

    fn update_tile(&mut self, tile_index: usize, logical_len: usize, plane_count: usize) {
        self.inner.update(&(tile_index as u64).to_le_bytes());
        self.inner.update(&(logical_len as u64).to_le_bytes());
        self.inner.update(&(plane_count as u64).to_le_bytes());
    }

    fn update_plane(&mut self, plane_index: usize, trits: &[Trit], scales: &[f16]) {
        self.inner.update(&(plane_index as u64).to_le_bytes());
        self.inner.update(&(trits.len() as u64).to_le_bytes());
        let mut canonical_trits = [0u8; SALT_V2_ALLOCATION_TILE_SIZE];
        for (encoded, trit) in canonical_trits.iter_mut().zip(trits) {
            *encoded = (trit.get() + 1) as u8;
        }
        self.inner.update(&canonical_trits[..trits.len()]);
        self.inner.update(&(scales.len() as u64).to_le_bytes());
        for scale in scales {
            self.inner.update(&scale.to_bits().to_le_bytes());
        }
    }

    fn finalize(self) -> SemanticTensor {
        self.inner
            .finalize()
            .expect("SALT V2 construction already validated tensor name and shape")
    }

    fn finalize_content_digest(self) -> [u8; 32] {
        self.inner.finalize_content_digest()
    }
}

/// A named SALT V2 tensor split into deterministic allocation macrotiles.
#[derive(Clone, Debug, PartialEq)]
pub struct SaltV2Tensor {
    name: String,
    dims: Vec<u64>,
    logical_coefficients: usize,
    transform: SaltV2Transform,
    tiles: Vec<SaltV2Tile>,
}

impl SaltV2Tensor {
    /// Construct a semantic tensor.
    ///
    /// Every non-final tile must contain exactly 256 coefficients; the final
    /// tile contains the remaining coefficients. The dimension product must be
    /// nonzero, fit both `u64` and `usize`, and equal the tile coefficient sum.
    ///
    /// # Errors
    /// Rejects an invalid name, shape, tile count, or tile length.
    pub fn new(
        name: impl Into<String>,
        dims: Vec<u64>,
        tiles: Vec<SaltV2Tile>,
    ) -> Result<Self, SaltV2PackageError> {
        Self::new_with_transform(name, dims, SaltV2Transform::None, tiles)
    }

    /// Construct a semantic tensor with an explicit transform identity.
    ///
    /// # Errors
    /// Rejects an invalid name, shape, tile count, or tile length.
    pub fn new_with_transform(
        name: impl Into<String>,
        dims: Vec<u64>,
        transform: SaltV2Transform,
        tiles: Vec<SaltV2Tile>,
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

        let logical_u64 = checked_dimension_product(&dims)?;
        let logical_coefficients = usize::try_from(logical_u64)
            .map_err(|_| SaltV2PackageError::DimensionProductTooLarge(logical_u64))?;
        let expected_tiles = logical_coefficients.div_ceil(SALT_V2_ALLOCATION_TILE_SIZE);
        if tiles.len() != expected_tiles {
            return Err(SaltV2PackageError::WrongTileCount {
                expected: expected_tiles,
                got: tiles.len(),
            });
        }
        for (tile_index, tile) in tiles.iter().enumerate() {
            let consumed = tile_index
                .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            let expected = (logical_coefficients - consumed).min(SALT_V2_ALLOCATION_TILE_SIZE);
            if tile.logical_len() != expected {
                return Err(SaltV2PackageError::WrongTileLength {
                    tile_index,
                    expected,
                    got: tile.logical_len(),
                });
            }
        }

        Ok(Self {
            name,
            dims,
            logical_coefficients,
            transform,
            tiles,
        })
    }

    /// Tensor name stored as UTF-8.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Tensor dimensions in row-major semantic order.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Product of the tensor dimensions.
    #[must_use]
    pub fn logical_coefficients(&self) -> usize {
        self.logical_coefficients
    }

    /// Transform identity required to interpret the tensor coefficients.
    #[must_use]
    pub fn transform(&self) -> SaltV2Transform {
        self.transform
    }

    /// Deterministic 256-coefficient allocation tiles.
    #[must_use]
    pub fn tiles(&self) -> &[SaltV2Tile] {
        &self.tiles
    }

    /// Build the codec-independent semantic manifest entry for this tensor.
    ///
    /// The content digest binds transform parameters, tensor geometry, ordered
    /// tile/plane structure, decoded trits, and exact f16 scale bits. Repacking
    /// the same tensor as D2, B3, or S34 therefore leaves this identity unchanged.
    #[must_use]
    pub fn semantic_tensor(&self) -> SemanticTensor {
        let mut hasher = SaltV2SemanticTensorHasher::new(
            &self.name,
            &self.dims,
            self.logical_coefficients,
            self.transform,
            self.tiles.len(),
        );
        for (tile_index, tile) in self.tiles.iter().enumerate() {
            hasher.update_tile(tile_index, tile.logical_len(), tile.planes.len());
            for (plane_index, plane) in tile.planes.iter().enumerate() {
                hasher.update_plane(plane_index, &plane.trits, &plane.scales);
            }
        }
        hasher.finalize()
    }
}

/// A semantic SALT V2 package using exactly one physical codec.
#[derive(Clone, Debug, PartialEq)]
pub struct SaltV2Package {
    codec: SaltV2Codec,
    tensors: Vec<SaltV2Tensor>,
}

impl SaltV2Package {
    /// Construct a package and reject duplicate tensor names.
    ///
    /// # Errors
    /// Rejects an empty package, too many tensors, duplicate names, or a tensor
    /// whose trits cannot be represented canonically by the selected codec.
    pub fn new(codec: SaltV2Codec, tensors: Vec<SaltV2Tensor>) -> Result<Self, SaltV2PackageError> {
        if tensors.is_empty() {
            return Err(SaltV2PackageError::EmptyPackage);
        }
        if tensors.len() > SALT_V2_MAX_TENSORS {
            return Err(SaltV2PackageError::TooManyTensors { got: tensors.len() });
        }
        let mut names = BTreeSet::new();
        for tensor in &tensors {
            if !names.insert(tensor.name.clone()) {
                return Err(SaltV2PackageError::DuplicateTensorName(tensor.name.clone()));
            }
            if codec == SaltV2Codec::S34 {
                for tile in &tensor.tiles {
                    for plane in &tile.planes {
                        canonical_s34_trits(&plane.trits)?;
                    }
                }
            }
        }
        Ok(Self { codec, tensors })
    }

    /// One codec used by every plane of every tensor in the package.
    #[must_use]
    pub fn codec(&self) -> SaltV2Codec {
        self.codec
    }

    /// Semantic tensors in canonical package order.
    #[must_use]
    pub fn tensors(&self) -> &[SaltV2Tensor] {
        &self.tensors
    }

    /// Derive a nested semantic prefix by slicing planes from this package.
    ///
    /// `requested_plane_counts` follows package tensor order and then tile
    /// order. Every requested count must be in `1..=available`. Trits and f16
    /// scales are cloned verbatim from the corresponding source-plane prefix;
    /// this API has no fitting or requantization path.
    ///
    /// # Errors
    /// Rejects a request with the wrong tensor/tile shape or a count that is
    /// zero or exceeds the source tile's available plane count.
    pub fn derive_prefix(
        &self,
        requested_plane_counts: &[Vec<usize>],
    ) -> Result<Self, SaltV2PackageError> {
        if requested_plane_counts.len() != self.tensors.len() {
            return Err(SaltV2PackageError::WrongPrefixTensorCount {
                expected: self.tensors.len(),
                got: requested_plane_counts.len(),
            });
        }

        let mut prefix_tensors = Vec::new();
        prefix_tensors
            .try_reserve_exact(self.tensors.len())
            .map_err(|_| SaltV2PackageError::AllocationFailed)?;
        for (tensor_index, (tensor, requested)) in
            self.tensors.iter().zip(requested_plane_counts).enumerate()
        {
            if requested.len() != tensor.tiles.len() {
                return Err(SaltV2PackageError::WrongPrefixTileCount {
                    tensor_index,
                    expected: tensor.tiles.len(),
                    got: requested.len(),
                });
            }
            let mut prefix_tiles = Vec::new();
            prefix_tiles
                .try_reserve_exact(tensor.tiles.len())
                .map_err(|_| SaltV2PackageError::AllocationFailed)?;
            for (tile_index, (tile, &plane_count)) in tensor.tiles.iter().zip(requested).enumerate()
            {
                if plane_count == 0 || plane_count > tile.planes.len() {
                    return Err(SaltV2PackageError::InvalidPrefixPlaneCount {
                        tensor_index,
                        tile_index,
                        requested: plane_count,
                        available: tile.planes.len(),
                    });
                }
                prefix_tiles.push(SaltV2Tile::new(tile.planes[..plane_count].to_vec())?);
            }
            prefix_tensors.push(SaltV2Tensor::new_with_transform(
                tensor.name.clone(),
                tensor.dims.clone(),
                tensor.transform,
                prefix_tiles,
            )?);
        }
        Self::new(self.codec, prefix_tensors)
    }
}

/// Exact physical component accounting for a SALT V2 package.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SaltV2PackageLedger {
    /// Package/tensor framing, names, and dimensions, excluding transform metadata.
    pub headers_bytes: u64,
    /// Fixed transform tag, reserved, seed, and domain bytes.
    pub transform_bytes: u64,
    /// Complete serialized bytes in the package-global allocation map.
    pub maps_bytes: u64,
    /// Exact logical allocation-map bits, always two per allocation tile.
    pub allocation_map_bits: u64,
    /// Allocation-map bits embedded in mandatory count words.
    ///
    /// These bits occupy no additional byte, but are reported so map storage is
    /// never hidden inside [`Self::headers_bytes`]. This is the sum of the
    /// package-level and tensor-level embedded-bit fields below.
    pub allocation_map_embedded_bits: u64,
    /// Terminal full-tile map bits embedded in the package tensor-count word.
    pub allocation_map_package_embedded_bits: u8,
    /// Ragged-final-tile map bits embedded in mandatory tensor tile-count words.
    pub allocation_map_tensor_embedded_bits: u64,
    /// Number of 256-coefficient allocation tiles across the whole package.
    pub allocation_tiles: u64,
    /// Whole-package denominator for the regular-macrotile metadata-rate bound.
    pub allocation_capacity_coefficients: u64,
    /// Encoded ternary payload bytes for planes that are actually present.
    pub payload_bytes: u64,
    /// Zero-point-free f16 group128 scale bytes for present planes.
    pub scales_bytes: u64,
    /// Canonical zero bytes used to align the whole package.
    pub padding_bytes: u64,
    /// Serialized bytes before terminal file-alignment padding.
    ///
    /// This is not a runtime-residency claim. Use
    /// [`SaltV2IndexedRuntimeLedger`] for the indexed execution layout.
    pub serialized_unpadded_bytes: u64,
    /// Exact physical file size.
    pub total_bytes: u64,
    /// Semantic-zero trit slots added to complete codec structures.
    pub codec_padding_trits: u64,
    /// Terminal zero bits added to complete codec payload bytes.
    pub codec_padding_bits: u64,
}

/// Exact requested bytes for the indexed SALT V2 runtime representation.
///
/// This is intentionally distinct from
/// [`SaltV2PackageLedger::serialized_unpadded_bytes`], which describes package
/// bytes rather than a runtime allocation. The indexed
/// execution layout retains codec payloads and scales plus the complete bytes of
/// a two-bit-per-tile plane-count map. Its terminal partial byte is carried in a
/// mandatory kernel scalar and consumes no device allocation. One u32 prefix per
/// 256-tile rank block bounds map scans while keeping regular-tile index metadata
/// below 0.01 bpw. No per-plane descriptors or dense weight shadow are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2IndexedRuntimeLedger {
    payload_bytes: u64,
    scale_bytes: u64,
    allocation_map_bytes: u64,
    rank_prefix_bytes: u64,
    allocation_map_bits: u64,
    allocation_map_embedded_bits: u64,
    dense_shadow_bytes: u64,
    allocation_tiles: u64,
    present_planes: u64,
    steady_resident_bytes: u64,
}

impl SaltV2IndexedRuntimeLedger {
    /// Plan the exact indexed-runtime layout for one validated tensor.
    ///
    /// # Errors
    /// Returns a package error if codec length accounting or any byte/count sum
    /// overflows.
    pub fn for_tensor(
        tensor: &SaltV2Tensor,
        codec: SaltV2Codec,
    ) -> Result<Self, SaltV2PackageError> {
        let allocation_tiles =
            u64::try_from(tensor.tiles.len()).map_err(|_| SaltV2PackageError::LengthOverflow)?;
        let mut payload_bytes = 0_u64;
        let mut scale_bytes = 0_u64;
        let mut present_planes = 0_u64;
        for tile in &tensor.tiles {
            present_planes = present_planes
                .checked_add(
                    u64::try_from(tile.planes.len())
                        .map_err(|_| SaltV2PackageError::LengthOverflow)?,
                )
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            for plane in &tile.planes {
                if codec == SaltV2Codec::S34 {
                    // Validate the structured code before publishing an executable plan.
                    let _ = pack_salt_v2_plane(codec, &plane.trits)?;
                }
                let stored_trits = stored_trit_count(codec, plane.trits.len())?;
                payload_bytes = payload_bytes
                    .checked_add(
                        u64::try_from(codec.ledger(stored_trits)?.physical_bytes)
                            .map_err(|_| SaltV2PackageError::LengthOverflow)?,
                    )
                    .ok_or(SaltV2PackageError::LengthOverflow)?;
                scale_bytes = scale_bytes
                    .checked_add(
                        u64::try_from(plane.scales.len())
                            .map_err(|_| SaltV2PackageError::LengthOverflow)?
                            .checked_mul(2)
                            .ok_or(SaltV2PackageError::LengthOverflow)?,
                    )
                    .ok_or(SaltV2PackageError::LengthOverflow)?;
            }
        }
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
            payload_bytes,
            scale_bytes / 2,
        ] {
            u32::try_from(value).map_err(|_| SaltV2PackageError::LengthOverflow)?;
        }
        let steady_resident_bytes = payload_bytes
            .checked_add(scale_bytes)
            .and_then(|bytes| bytes.checked_add(allocation_map_bytes))
            .and_then(|bytes| bytes.checked_add(rank_prefix_bytes))
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        Ok(Self {
            payload_bytes,
            scale_bytes,
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

    /// Plan the sum of independently indexed tensors in a package.
    ///
    /// # Errors
    /// Returns a package error if any tensor is unrepresentable or any aggregate
    /// counter overflows.
    pub fn for_package(package: &SaltV2Package) -> Result<Self, SaltV2PackageError> {
        package
            .tensors
            .iter()
            .try_fold(Self::zero(), |total, tensor| {
                total.checked_add(Self::for_tensor(tensor, package.codec)?)
            })
    }

    const fn zero() -> Self {
        Self {
            payload_bytes: 0,
            scale_bytes: 0,
            allocation_map_bytes: 0,
            rank_prefix_bytes: 0,
            allocation_map_bits: 0,
            allocation_map_embedded_bits: 0,
            dense_shadow_bytes: 0,
            allocation_tiles: 0,
            present_planes: 0,
            steady_resident_bytes: 0,
        }
    }

    fn checked_add(self, other: Self) -> Result<Self, SaltV2PackageError> {
        let add = |left: u64, right: u64| {
            left.checked_add(right)
                .ok_or(SaltV2PackageError::LengthOverflow)
        };
        Ok(Self {
            payload_bytes: add(self.payload_bytes, other.payload_bytes)?,
            scale_bytes: add(self.scale_bytes, other.scale_bytes)?,
            allocation_map_bytes: add(self.allocation_map_bytes, other.allocation_map_bytes)?,
            rank_prefix_bytes: add(self.rank_prefix_bytes, other.rank_prefix_bytes)?,
            allocation_map_bits: add(self.allocation_map_bits, other.allocation_map_bits)?,
            allocation_map_embedded_bits: add(
                self.allocation_map_embedded_bits,
                other.allocation_map_embedded_bits,
            )?,
            dense_shadow_bytes: add(self.dense_shadow_bytes, other.dense_shadow_bytes)?,
            allocation_tiles: add(self.allocation_tiles, other.allocation_tiles)?,
            present_planes: add(self.present_planes, other.present_planes)?,
            steady_resident_bytes: add(self.steady_resident_bytes, other.steady_resident_bytes)?,
        })
    }

    /// Encoded D2/B3/S34 payload bytes.
    pub const fn payload_bytes(self) -> u64 {
        self.payload_bytes
    }

    /// Group128 f16 scale bytes.
    pub const fn scale_bytes(self) -> u64 {
        self.scale_bytes
    }

    /// Complete allocated bytes of the two-bit plane-count map.
    pub const fn allocation_map_bytes(self) -> u64 {
        self.allocation_map_bytes
    }

    /// Coarse u32 plane-rank prefix bytes.
    pub const fn rank_prefix_bytes(self) -> u64 {
        self.rank_prefix_bytes
    }

    /// Logical two-bit allocation-map size, including the scalar-carried tail.
    pub const fn allocation_map_bits(self) -> u64 {
        self.allocation_map_bits
    }

    /// Terminal allocation-map bits carried in the runtime handle/kernel scalar.
    pub const fn allocation_map_embedded_bits(self) -> u64 {
        self.allocation_map_embedded_bits
    }

    /// Dense reconstructed weight shadow bytes, structurally zero.
    pub const fn dense_shadow_bytes(self) -> u64 {
        self.dense_shadow_bytes
    }

    /// Allocation-tile count.
    pub const fn allocation_tiles(self) -> u64 {
        self.allocation_tiles
    }

    /// Present-plane count.
    pub const fn present_planes(self) -> u64 {
        self.present_planes
    }

    /// Sum of payload, scales, allocated map bytes, and rank prefixes.
    pub const fn steady_resident_bytes(self) -> u64 {
        self.steady_resident_bytes
    }
}

/// Encoded package bytes and their exact physical ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedSaltV2Package {
    /// Canonical package bytes.
    pub bytes: Vec<u8>,
    /// Exact component accounting; `total_bytes == bytes.len()`.
    pub ledger: SaltV2PackageLedger,
}

/// Decoded semantic package and the ledger measured from its bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedSaltV2Package {
    /// Validated semantic package.
    pub package: SaltV2Package,
    /// Exact component accounting measured during canonical decoding.
    pub ledger: SaltV2PackageLedger,
}

/// Errors from SALT V2 semantic construction or canonical package I/O.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2PackageError {
    /// A raw coefficient was not one of `-1`, `0`, or `+1`.
    NonCanonicalTrit {
        /// Coefficient index within the plane.
        index: usize,
        /// Invalid raw value.
        value: i8,
    },
    /// A plane was empty or exceeded one allocation tile.
    InvalidPlaneLength {
        /// Supplied logical coefficient count.
        got: usize,
    },
    /// A plane did not contain exactly one scale per logical group128.
    WrongScaleCount {
        /// Canonical scale count.
        expected: usize,
        /// Supplied scale count.
        got: usize,
    },
    /// A scale was NaN or infinite.
    NonFiniteScale {
        /// Scale-group index.
        group_index: usize,
        /// Raw f16 bits.
        bits: u16,
    },
    /// A scale was negative, including negative zero.
    NegativeScale {
        /// Scale-group index.
        group_index: usize,
        /// Raw f16 bits.
        bits: u16,
    },
    /// A zero scale would erase a nonzero ternary group.
    ZeroScaleForNonzeroGroup {
        /// Scale-group index.
        group_index: usize,
    },
    /// A tile did not contain between one and three planes.
    InvalidPlaneCount {
        /// Supplied plane count.
        got: usize,
    },
    /// Planes in one tile had different logical lengths.
    InconsistentPlaneLength {
        /// Offending plane index.
        plane_index: usize,
        /// Base-plane logical length.
        expected: usize,
        /// Offending logical length.
        got: usize,
    },
    /// A tensor name was empty.
    EmptyTensorName,
    /// A tensor name did not fit the canonical `u32` field.
    TensorNameTooLong {
        /// Name length in bytes.
        got: usize,
    },
    /// A tensor shape had no dimensions.
    EmptyDimensions,
    /// A tensor shape contained a zero dimension.
    ZeroDimension {
        /// Dimension index.
        index: usize,
    },
    /// A tensor rank did not fit the canonical `u32` field.
    TooManyDimensions {
        /// Supplied rank.
        got: usize,
    },
    /// Multiplying tensor dimensions overflowed `u64`.
    DimensionProductOverflow,
    /// The dimension product could not fit the host `usize`.
    DimensionProductTooLarge(u64),
    /// The number of tiles did not match the shape-derived count.
    WrongTileCount {
        /// Shape-derived tile count.
        expected: usize,
        /// Supplied tile count.
        got: usize,
    },
    /// A non-final tile was not full, or the final tile had the wrong remainder.
    WrongTileLength {
        /// Allocation tile index.
        tile_index: usize,
        /// Shape-derived logical length.
        expected: usize,
        /// Supplied logical length.
        got: usize,
    },
    /// A package contained no tensors.
    EmptyPackage,
    /// A package tensor count did not fit the canonical 26-bit count field.
    TooManyTensors {
        /// Supplied tensor count.
        got: usize,
    },
    /// A tensor name occurred more than once.
    DuplicateTensorName(String),
    /// A semantic prefix request did not cover every source tensor.
    WrongPrefixTensorCount {
        /// Source tensor count.
        expected: usize,
        /// Request tensor count.
        got: usize,
    },
    /// A semantic prefix request did not cover every tile in one tensor.
    WrongPrefixTileCount {
        /// Tensor index in package order.
        tensor_index: usize,
        /// Source tile count.
        expected: usize,
        /// Request tile count.
        got: usize,
    },
    /// A semantic prefix requested zero planes or more planes than available.
    InvalidPrefixPlaneCount {
        /// Tensor index in package order.
        tensor_index: usize,
        /// Tile index in tensor order.
        tile_index: usize,
        /// Requested prefix length.
        requested: usize,
        /// Source plane count.
        available: usize,
    },
    /// Physical length arithmetic overflowed.
    LengthOverflow,
    /// A requested allocation could not be reserved.
    AllocationFailed,
    /// A selected physical ternary codec rejected input or bytes.
    Codec(SaltV2CodecError),
    /// Package magic did not match [`SALT_V2_PACKAGE_MAGIC`].
    BadMagic,
    /// Package version is not supported.
    UnsupportedVersion(u16),
    /// Package codec tag is not supported.
    UnsupportedCodec(u8),
    /// Tensor transform tag is not supported.
    UnsupportedTransformTag(u8),
    /// Reserved transform bytes or identity-only seed/domain fields were nonzero.
    NonCanonicalTransformMetadata,
    /// Reserved package flags were nonzero.
    NonZeroFlags(u8),
    /// The declared total length did not equal the supplied slice length.
    WrongTotalLength {
        /// Length stored in the package header.
        declared: u64,
        /// Actual slice length.
        actual: usize,
    },
    /// A length-delimited field ran past the input.
    Truncated {
        /// Bytes requested by the field.
        needed: usize,
        /// Bytes remaining in the input.
        remaining: usize,
    },
    /// A tensor name in the package was not UTF-8.
    InvalidTensorName,
    /// A redundant on-disk field disagreed with its canonical derived value.
    DeclaredFieldMismatch {
        /// Name of the inconsistent field.
        field: &'static str,
        /// Value stored in the package.
        declared: u64,
        /// Canonical value derived from semantic metadata.
        expected: u64,
    },
    /// A declared section was too short even for one mandatory plane per tile.
    DeclaredFieldBelowMinimum {
        /// Name of the undersized field.
        field: &'static str,
        /// Value stored in the package.
        declared: u64,
        /// Smallest possible value derived from tensor geometry and codec.
        minimum: u64,
    },
    /// A map set plane three without setting plane two.
    NonNestedPlaneMap {
        /// Allocation tile index.
        tile_index: usize,
    },
    /// Unused high bits beside the terminal embedded allocation-map bits were nonzero.
    NonCanonicalMapPadding,
    /// An S34 logical group could not retain exactly one zero without changing values.
    S34IncompatibleGroup {
        /// Four-trit group index within the plane.
        group_index: usize,
        /// Logical values present in that group.
        logical_trits: usize,
        /// Number of logical zeros present in that group.
        zero_count: usize,
    },
    /// S34 shape padding was structurally valid but not the canonical completion.
    NonCanonicalS34ShapePadding,
    /// Bytes remained after all declared tensor sections, beyond canonical alignment.
    UnexpectedTrailingData {
        /// Remaining bytes.
        remaining: usize,
        /// Canonical alignment bytes expected at this position.
        expected_padding: usize,
    },
    /// Canonical package-alignment bytes were not all zero.
    NonCanonicalFilePadding,
}

impl From<SaltV2CodecError> for SaltV2PackageError {
    fn from(value: SaltV2CodecError) -> Self {
        Self::Codec(value)
    }
}

impl fmt::Display for SaltV2PackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonCanonicalTrit { index, value } => {
                write!(f, "noncanonical trit {value} at plane index {index}")
            }
            Self::InvalidPlaneLength { got } => write!(
                f,
                "plane length must be in 1..={SALT_V2_ALLOCATION_TILE_SIZE}, got {got}"
            ),
            Self::WrongScaleCount { expected, got } => {
                write!(
                    f,
                    "wrong group128 scale count: expected {expected}, got {got}"
                )
            }
            Self::NonFiniteScale { group_index, bits } => {
                write!(f, "non-finite f16 scale {bits:#06x} in group {group_index}")
            }
            Self::NegativeScale { group_index, bits } => {
                write!(f, "negative f16 scale {bits:#06x} in group {group_index}")
            }
            Self::ZeroScaleForNonzeroGroup { group_index } => {
                write!(f, "zero scale for nonzero group {group_index}")
            }
            Self::InvalidPlaneCount { got } => {
                write!(f, "tile plane count must be in 1..=3, got {got}")
            }
            Self::InconsistentPlaneLength {
                plane_index,
                expected,
                got,
            } => write!(
                f,
                "plane {plane_index} length {got} differs from base length {expected}"
            ),
            Self::EmptyTensorName => write!(f, "tensor name is empty"),
            Self::TensorNameTooLong { got } => {
                write!(f, "tensor name length {got} exceeds u32")
            }
            Self::EmptyDimensions => write!(f, "tensor shape is empty"),
            Self::ZeroDimension { index } => write!(f, "tensor dimension {index} is zero"),
            Self::TooManyDimensions { got } => write!(f, "tensor rank {got} exceeds u32"),
            Self::DimensionProductOverflow => write!(f, "tensor dimension product overflows u64"),
            Self::DimensionProductTooLarge(value) => {
                write!(f, "tensor dimension product {value} exceeds host usize")
            }
            Self::WrongTileCount { expected, got } => {
                write!(f, "wrong tile count: expected {expected}, got {got}")
            }
            Self::WrongTileLength {
                tile_index,
                expected,
                got,
            } => write!(f, "tile {tile_index} has length {got}, expected {expected}"),
            Self::EmptyPackage => write!(f, "SALT V2 package contains no tensors"),
            Self::TooManyTensors { got } => write!(
                f,
                "tensor count {got} exceeds the {SALT_V2_TENSOR_COUNT_BITS}-bit package field"
            ),
            Self::DuplicateTensorName(name) => write!(f, "duplicate tensor name `{name}`"),
            Self::WrongPrefixTensorCount { expected, got } => write!(
                f,
                "prefix tensor count is {got}, source package requires {expected}"
            ),
            Self::WrongPrefixTileCount {
                tensor_index,
                expected,
                got,
            } => write!(
                f,
                "prefix tensor {tensor_index} has {got} tile requests, expected {expected}"
            ),
            Self::InvalidPrefixPlaneCount {
                tensor_index,
                tile_index,
                requested,
                available,
            } => write!(
                f,
                "prefix tensor {tensor_index} tile {tile_index} requests {requested} of {available} planes"
            ),
            Self::LengthOverflow => write!(f, "SALT V2 package length arithmetic overflow"),
            Self::AllocationFailed => write!(f, "SALT V2 package allocation failed"),
            Self::Codec(error) => write!(f, "SALT V2 codec error: {error}"),
            Self::BadMagic => write!(f, "SALT V2 package has bad magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported SALT V2 package version {version}")
            }
            Self::UnsupportedCodec(codec) => write!(f, "unsupported SALT V2 codec tag {codec}"),
            Self::UnsupportedTransformTag(tag) => {
                write!(f, "unsupported SALT V2 transform tag {tag}")
            }
            Self::NonCanonicalTransformMetadata => {
                write!(f, "SALT V2 transform metadata is not canonical")
            }
            Self::NonZeroFlags(flags) => {
                write!(f, "SALT V2 package reserved flags are {flags:#04x}")
            }
            Self::WrongTotalLength { declared, actual } => write!(
                f,
                "SALT V2 package declares {declared} bytes but received {actual}"
            ),
            Self::Truncated { needed, remaining } => write!(
                f,
                "truncated SALT V2 package: need {needed} bytes, {remaining} remain"
            ),
            Self::InvalidTensorName => write!(f, "SALT V2 tensor name is not UTF-8"),
            Self::DeclaredFieldMismatch {
                field,
                declared,
                expected,
            } => write!(
                f,
                "declared {field} is {declared}, canonical value is {expected}"
            ),
            Self::DeclaredFieldBelowMinimum {
                field,
                declared,
                minimum,
            } => write!(
                f,
                "declared {field} is {declared}, below geometry minimum {minimum}"
            ),
            Self::NonNestedPlaneMap { tile_index } => {
                write!(f, "tile {tile_index} has plane three without plane two")
            }
            Self::NonCanonicalMapPadding => {
                write!(f, "optional-plane map has nonzero padding bits")
            }
            Self::S34IncompatibleGroup {
                group_index,
                logical_trits,
                zero_count,
            } => write!(
                f,
                "S34 group {group_index} has {zero_count} zeros in {logical_trits} logical trits"
            ),
            Self::NonCanonicalS34ShapePadding => {
                write!(f, "S34 shape padding is not canonical")
            }
            Self::UnexpectedTrailingData {
                remaining,
                expected_padding,
            } => write!(
                f,
                "{remaining} bytes remain where {expected_padding} alignment bytes are canonical"
            ),
            Self::NonCanonicalFilePadding => {
                write!(f, "SALT V2 package alignment padding is not zero")
            }
        }
    }
}

impl std::error::Error for SaltV2PackageError {}

struct EncodedTensor {
    name: String,
    dims: Vec<u64>,
    logical_coefficients: u64,
    packed_tile_count: u64,
    transform: SaltV2Transform,
    payload: Vec<u8>,
    scales: Vec<u8>,
    codec_padding_trits: u64,
    codec_padding_bits: u64,
}

struct RawTensor<'a> {
    name: String,
    dims: Vec<u64>,
    logical_coefficients: usize,
    full_tile_count: usize,
    ragged_plane_count: Option<usize>,
    transform: SaltV2Transform,
    declared_payload: u64,
    declared_scales: u64,
    payload: &'a [u8],
    scales: &'a [u8],
}

/// Encode a semantic package into its one canonical byte representation.
///
/// # Errors
/// Returns a typed error on physical length overflow, allocation failure, or a
/// selected codec invariant violation.
pub fn write_salt_v2_package(
    package: &SaltV2Package,
) -> Result<EncodedSaltV2Package, SaltV2PackageError> {
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(package.tensors.len())
        .map_err(|_| SaltV2PackageError::AllocationFailed)?;
    let allocation_map = encode_presence_maps(&package.tensors)?;

    let mut ledger = SaltV2PackageLedger {
        headers_bytes: SALT_V2_PACKAGE_HEADER_BYTES as u64,
        maps_bytes: u64::try_from(allocation_map.bytes.len())
            .map_err(|_| SaltV2PackageError::LengthOverflow)?,
        allocation_map_bits: u64::try_from(allocation_map.logical_bits)
            .map_err(|_| SaltV2PackageError::LengthOverflow)?,
        allocation_map_embedded_bits: u64::from(allocation_map.package_embedded_bits)
            .checked_add(
                u64::try_from(allocation_map.tensor_embedded_bits)
                    .map_err(|_| SaltV2PackageError::LengthOverflow)?,
            )
            .ok_or(SaltV2PackageError::LengthOverflow)?,
        allocation_map_package_embedded_bits: allocation_map.package_embedded_bits,
        allocation_map_tensor_embedded_bits: u64::try_from(allocation_map.tensor_embedded_bits)
            .map_err(|_| SaltV2PackageError::LengthOverflow)?,
        allocation_tiles: u64::try_from(allocation_map.total_tiles)
            .map_err(|_| SaltV2PackageError::LengthOverflow)?,
        allocation_capacity_coefficients: u64::try_from(allocation_map.total_tiles)
            .map_err(|_| SaltV2PackageError::LengthOverflow)?
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE as u64)
            .ok_or(SaltV2PackageError::LengthOverflow)?,
        ..SaltV2PackageLedger::default()
    };
    debug_assert_eq!(
        ledger.maps_bytes * 8 + ledger.allocation_map_embedded_bits,
        ledger.allocation_map_bits
    );
    debug_assert_eq!(
        ledger.allocation_map_embedded_bits,
        u64::from(ledger.allocation_map_package_embedded_bits)
            + ledger.allocation_map_tensor_embedded_bits
    );
    debug_assert_eq!(ledger.allocation_map_bits, ledger.allocation_tiles * 2);
    for tensor in &package.tensors {
        let encoded = encode_tensor(tensor, package.codec)?;
        checked_ledger_add(
            &mut ledger.headers_bytes,
            (SALT_V2_TENSOR_HEADER_BYTES - SALT_V2_TRANSFORM_METADATA_BYTES)
                .checked_add(encoded.name.len())
                .and_then(|value| value.checked_add(encoded.dims.len().checked_mul(8)?))
                .ok_or(SaltV2PackageError::LengthOverflow)?,
        )?;
        checked_ledger_add(
            &mut ledger.transform_bytes,
            SALT_V2_TRANSFORM_METADATA_BYTES,
        )?;
        checked_ledger_add(&mut ledger.payload_bytes, encoded.payload.len())?;
        checked_ledger_add(&mut ledger.scales_bytes, encoded.scales.len())?;
        ledger.codec_padding_trits = ledger
            .codec_padding_trits
            .checked_add(encoded.codec_padding_trits)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        ledger.codec_padding_bits = ledger
            .codec_padding_bits
            .checked_add(encoded.codec_padding_bits)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        tensors.push(encoded);
    }

    let raw_bytes = ledger
        .headers_bytes
        .checked_add(ledger.transform_bytes)
        .and_then(|value| value.checked_add(ledger.maps_bytes))
        .and_then(|value| value.checked_add(ledger.payload_bytes))
        .and_then(|value| value.checked_add(ledger.scales_bytes))
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    ledger.serialized_unpadded_bytes = ledger
        .headers_bytes
        .checked_add(ledger.transform_bytes)
        .and_then(|value| value.checked_add(ledger.maps_bytes))
        .and_then(|value| value.checked_add(ledger.payload_bytes))
        .and_then(|value| value.checked_add(ledger.scales_bytes))
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    let raw_len = usize::try_from(raw_bytes).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    let padding = alignment_padding(raw_len);
    ledger.padding_bytes = padding as u64;
    ledger.total_bytes = raw_bytes
        .checked_add(padding as u64)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    let total_len =
        usize::try_from(ledger.total_bytes).map_err(|_| SaltV2PackageError::LengthOverflow)?;

    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(total_len)
        .map_err(|_| SaltV2PackageError::AllocationFailed)?;
    bytes.extend_from_slice(&SALT_V2_PACKAGE_MAGIC);
    push_u16(&mut bytes, SALT_V2_PACKAGE_VERSION);
    bytes.push(codec_tag(package.codec));
    bytes.push(0);
    push_u32(&mut bytes, allocation_map.packed_tensor_count);
    push_u64(&mut bytes, ledger.total_bytes);

    for tensor in tensors {
        push_u32(
            &mut bytes,
            u32::try_from(tensor.name.len()).map_err(|_| SaltV2PackageError::LengthOverflow)?,
        );
        push_u32(
            &mut bytes,
            u32::try_from(tensor.dims.len()).map_err(|_| SaltV2PackageError::LengthOverflow)?,
        );
        push_u64(&mut bytes, tensor.logical_coefficients);
        push_u64(&mut bytes, tensor.packed_tile_count);
        push_u64(
            &mut bytes,
            u64::try_from(tensor.payload.len()).map_err(|_| SaltV2PackageError::LengthOverflow)?,
        );
        push_u64(
            &mut bytes,
            u64::try_from(tensor.scales.len()).map_err(|_| SaltV2PackageError::LengthOverflow)?,
        );
        push_transform(&mut bytes, tensor.transform);
        bytes.extend_from_slice(tensor.name.as_bytes());
        for dim in tensor.dims {
            push_u64(&mut bytes, dim);
        }
        bytes.extend_from_slice(&tensor.payload);
        bytes.extend_from_slice(&tensor.scales);
    }
    bytes.extend_from_slice(&allocation_map.bytes);
    bytes.resize(total_len, 0);
    debug_assert_eq!(bytes.len() as u64, ledger.total_bytes);

    Ok(EncodedSaltV2Package { bytes, ledger })
}

/// Decode and canonically validate a complete SALT V2 package.
///
/// Decoding is exact-length: declared section sizes are checked against values
/// derived from the shape and maps, codec padding must be canonical, duplicate
/// names are rejected, and no undeclared trailing byte is accepted.
///
/// # Errors
/// Returns a typed error for malformed, truncated, noncanonical, overflowing,
/// duplicate, or trailing input.
pub fn read_salt_v2_package(bytes: &[u8]) -> Result<DecodedSaltV2Package, SaltV2PackageError> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(SALT_V2_PACKAGE_MAGIC.len())? != SALT_V2_PACKAGE_MAGIC {
        return Err(SaltV2PackageError::BadMagic);
    }
    let version = cursor.u16()?;
    if version != SALT_V2_PACKAGE_VERSION {
        return Err(SaltV2PackageError::UnsupportedVersion(version));
    }
    let codec = codec_from_tag(cursor.u8()?)?;
    let flags = cursor.u8()?;
    if flags != 0 {
        return Err(SaltV2PackageError::NonZeroFlags(flags));
    }
    let packed_tensor_count = cursor.u32()?;
    let tensor_count_mask = (1u32 << SALT_V2_TENSOR_COUNT_BITS) - 1;
    let tensor_count = (packed_tensor_count & tensor_count_mask) as usize;
    let embedded_map_value = (packed_tensor_count >> SALT_V2_TENSOR_COUNT_BITS) as u8;
    let declared_total = cursor.u64()?;
    if u64::try_from(bytes.len()).ok() != Some(declared_total) {
        return Err(SaltV2PackageError::WrongTotalLength {
            declared: declared_total,
            actual: bytes.len(),
        });
    }
    if tensor_count == 0 {
        return Err(SaltV2PackageError::EmptyPackage);
    }
    if tensor_count > cursor.remaining() / SALT_V2_TENSOR_HEADER_BYTES {
        return Err(SaltV2PackageError::Truncated {
            needed: tensor_count
                .checked_mul(SALT_V2_TENSOR_HEADER_BYTES)
                .ok_or(SaltV2PackageError::LengthOverflow)?,
            remaining: cursor.remaining(),
        });
    }

    let mut raw_tensors = Vec::new();
    raw_tensors
        .try_reserve_exact(tensor_count)
        .map_err(|_| SaltV2PackageError::AllocationFailed)?;
    let mut names = BTreeSet::new();
    let mut ledger = SaltV2PackageLedger {
        headers_bytes: SALT_V2_PACKAGE_HEADER_BYTES as u64,
        ..SaltV2PackageLedger::default()
    };

    let mut total_tiles = 0usize;
    let mut total_full_tiles = 0usize;
    let mut ragged_tensor_count = 0usize;
    for _ in 0..tensor_count {
        let name_len = cursor.u32()? as usize;
        let rank = cursor.u32()? as usize;
        let declared_coefficients = cursor.u64()?;
        let packed_declared_tiles = cursor.u64()?;
        let declared_payload = cursor.u64()?;
        let declared_scales = cursor.u64()?;
        let transform = read_transform(&mut cursor)?;

        let name_bytes = cursor.take(name_len)?;
        let name = core::str::from_utf8(name_bytes)
            .map_err(|_| SaltV2PackageError::InvalidTensorName)?
            .to_owned();
        if name.is_empty() {
            return Err(SaltV2PackageError::EmptyTensorName);
        }
        if !names.insert(name.clone()) {
            return Err(SaltV2PackageError::DuplicateTensorName(name));
        }

        let dims_bytes = rank
            .checked_mul(8)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let mut dims_cursor = Cursor::new(cursor.take(dims_bytes)?);
        let mut dims = Vec::new();
        dims.try_reserve_exact(rank)
            .map_err(|_| SaltV2PackageError::AllocationFailed)?;
        for _ in 0..rank {
            dims.push(dims_cursor.u64()?);
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
                    return Err(SaltV2PackageError::NonCanonicalMapPadding);
                }
                None
            } else {
                ragged_tensor_count = ragged_tensor_count
                    .checked_add(1)
                    .ok_or(SaltV2PackageError::LengthOverflow)?;
                Some(map_value_plane_count(ragged_map_value, tile_count - 1)?)
            };
        let payload_len =
            usize::try_from(declared_payload).map_err(|_| SaltV2PackageError::LengthOverflow)?;
        let scales_len =
            usize::try_from(declared_scales).map_err(|_| SaltV2PackageError::LengthOverflow)?;
        let declared_sections = payload_len
            .checked_add(scales_len)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        if declared_sections > cursor.remaining() {
            return Err(SaltV2PackageError::Truncated {
                needed: declared_sections,
                remaining: cursor.remaining(),
            });
        }
        let minimum = minimum_tensor_physical(codec, logical_coefficients)?;
        require_at_least("payload bytes", declared_payload, minimum.payload_bytes)?;
        require_at_least("scale bytes", declared_scales, minimum.scales_bytes)?;
        total_tiles = total_tiles
            .checked_add(tile_count)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        total_full_tiles = total_full_tiles
            .checked_add(full_tile_count)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let payload = cursor.take(payload_len)?;
        let scales = cursor.take(scales_len)?;

        let tensor_headers = (SALT_V2_TENSOR_HEADER_BYTES - SALT_V2_TRANSFORM_METADATA_BYTES)
            .checked_add(name_len)
            .and_then(|value| value.checked_add(dims_bytes))
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        checked_ledger_add(&mut ledger.headers_bytes, tensor_headers)?;
        checked_ledger_add(
            &mut ledger.transform_bytes,
            SALT_V2_TRANSFORM_METADATA_BYTES,
        )?;
        raw_tensors.push(RawTensor {
            name,
            dims,
            logical_coefficients,
            full_tile_count,
            ragged_plane_count,
            transform,
            declared_payload,
            declared_scales,
            payload,
            scales,
        });
    }

    let map_len = presence_map_len(total_full_tiles)?;
    let maps = cursor.take(map_len)?;
    let full_plane_counts = ValidatedPresenceMap::new(embedded_map_value, maps, total_full_tiles)?;

    let expected_padding = alignment_padding(cursor.position());
    if cursor.remaining() != expected_padding {
        return Err(SaltV2PackageError::UnexpectedTrailingData {
            remaining: cursor.remaining(),
            expected_padding,
        });
    }
    let padding = cursor.take(expected_padding)?;
    if padding.iter().any(|byte| *byte != 0) {
        return Err(SaltV2PackageError::NonCanonicalFilePadding);
    }
    if cursor.remaining() != 0 {
        return Err(SaltV2PackageError::UnexpectedTrailingData {
            remaining: cursor.remaining(),
            expected_padding: 0,
        });
    }

    let total_map_bits = total_tiles
        .checked_mul(2)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    let full_map_bits = total_full_tiles
        .checked_mul(2)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    let package_embedded_bits = full_map_bits % 8;
    let tensor_embedded_bits = ragged_tensor_count
        .checked_mul(2)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    ledger.maps_bytes = u64::try_from(map_len).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    ledger.allocation_map_bits =
        u64::try_from(total_map_bits).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    ledger.allocation_map_package_embedded_bits = package_embedded_bits as u8;
    ledger.allocation_map_tensor_embedded_bits =
        u64::try_from(tensor_embedded_bits).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    ledger.allocation_map_embedded_bits = u64::from(ledger.allocation_map_package_embedded_bits)
        .checked_add(ledger.allocation_map_tensor_embedded_bits)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    ledger.allocation_tiles =
        u64::try_from(total_tiles).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    ledger.allocation_capacity_coefficients = ledger
        .allocation_tiles
        .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE as u64)
        .ok_or(SaltV2PackageError::LengthOverflow)?;

    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(raw_tensors.len())
        .map_err(|_| SaltV2PackageError::AllocationFailed)?;
    debug_assert_eq!(
        ledger.maps_bytes * 8 + ledger.allocation_map_embedded_bits,
        ledger.allocation_map_bits
    );
    let mut plane_offset = 0usize;
    for raw in raw_tensors {
        let plane_start = plane_offset;
        let plane_end = plane_start
            .checked_add(raw.full_tile_count)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let plane_counts =
            || full_plane_counts.counts(plane_start..plane_end, raw.ragged_plane_count);
        plane_offset = plane_end;
        let physical = expected_tensor_physical(codec, raw.logical_coefficients, plane_counts())?;
        require_declared(
            "payload bytes",
            raw.declared_payload,
            physical.payload_bytes,
        )?;
        require_declared("scale bytes", raw.declared_scales, physical.scales_bytes)?;
        let tiles = decode_tensor_tiles(
            codec,
            raw.logical_coefficients,
            plane_counts(),
            raw.payload,
            raw.scales,
        )?;
        let tensor = SaltV2Tensor::new_with_transform(raw.name, raw.dims, raw.transform, tiles)?;
        checked_ledger_add(&mut ledger.payload_bytes, raw.payload.len())?;
        checked_ledger_add(&mut ledger.scales_bytes, raw.scales.len())?;
        ledger.codec_padding_trits = ledger
            .codec_padding_trits
            .checked_add(physical.codec_padding_trits)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        ledger.codec_padding_bits = ledger
            .codec_padding_bits
            .checked_add(physical.codec_padding_bits)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        tensors.push(tensor);
    }
    debug_assert_eq!(plane_offset, full_plane_counts.len());

    ledger.padding_bytes = expected_padding as u64;
    ledger.serialized_unpadded_bytes = ledger
        .headers_bytes
        .checked_add(ledger.transform_bytes)
        .and_then(|value| value.checked_add(ledger.maps_bytes))
        .and_then(|value| value.checked_add(ledger.payload_bytes))
        .and_then(|value| value.checked_add(ledger.scales_bytes))
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    ledger.total_bytes = ledger
        .headers_bytes
        .checked_add(ledger.transform_bytes)
        .and_then(|value| value.checked_add(ledger.maps_bytes))
        .and_then(|value| value.checked_add(ledger.payload_bytes))
        .and_then(|value| value.checked_add(ledger.scales_bytes))
        .and_then(|value| value.checked_add(ledger.padding_bytes))
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    require_declared("total package bytes", declared_total, ledger.total_bytes)?;

    Ok(DecodedSaltV2Package {
        package: SaltV2Package::new(codec, tensors)?,
        ledger,
    })
}

fn encode_tensor(
    tensor: &SaltV2Tensor,
    codec: SaltV2Codec,
) -> Result<EncodedTensor, SaltV2PackageError> {
    let tile_count =
        u64::try_from(tensor.tiles.len()).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    let tile_count_mask = (1u64 << SALT_V2_TILE_COUNT_BITS) - 1;
    if tile_count > tile_count_mask {
        return Err(SaltV2PackageError::LengthOverflow);
    }
    let ragged_map_value = if tensor
        .logical_coefficients
        .is_multiple_of(SALT_V2_ALLOCATION_TILE_SIZE)
    {
        0
    } else {
        plane_count_map_value(
            tensor
                .tiles
                .last()
                .expect("a positive tensor shape has at least one tile")
                .planes
                .len(),
        )
    };
    let packed_tile_count = tile_count | (u64::from(ragged_map_value) << SALT_V2_TILE_COUNT_BITS);
    let expected = expected_tensor_physical(
        codec,
        tensor.logical_coefficients,
        tensor.tiles.iter().map(|tile| tile.planes.len()),
    )?;
    let payload_len =
        usize::try_from(expected.payload_bytes).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    let scales_len =
        usize::try_from(expected.scales_bytes).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(payload_len)
        .map_err(|_| SaltV2PackageError::AllocationFailed)?;
    let mut scales = Vec::new();
    scales
        .try_reserve_exact(scales_len)
        .map_err(|_| SaltV2PackageError::AllocationFailed)?;

    let mut codec_padding_trits = 0u64;
    let mut codec_padding_bits = 0u64;
    for tile in &tensor.tiles {
        for plane in &tile.planes {
            let packed = pack_salt_v2_plane(codec, &plane.trits)?;
            let stored_trits = stored_trit_count(codec, plane.trits.len())?;
            let plane_ledger = codec.ledger(stored_trits)?;
            let shape_padding = stored_trits - plane.trits.len();
            codec_padding_trits = codec_padding_trits
                .checked_add(
                    u64::try_from(shape_padding + plane_ledger.canonical_padding_trits)
                        .map_err(|_| SaltV2PackageError::LengthOverflow)?,
                )
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            codec_padding_bits = codec_padding_bits
                .checked_add(u64::from(plane_ledger.canonical_padding_bits))
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            payload.extend_from_slice(&packed);
            for scale in &plane.scales {
                scales.extend_from_slice(&scale.to_bits().to_le_bytes());
            }
        }
    }
    debug_assert_eq!(payload.len(), payload_len);
    debug_assert_eq!(scales.len(), scales_len);
    debug_assert_eq!(codec_padding_trits, expected.codec_padding_trits);
    debug_assert_eq!(codec_padding_bits, expected.codec_padding_bits);

    Ok(EncodedTensor {
        name: tensor.name.clone(),
        dims: tensor.dims.clone(),
        logical_coefficients: u64::try_from(tensor.logical_coefficients)
            .map_err(|_| SaltV2PackageError::LengthOverflow)?,
        packed_tile_count,
        transform: tensor.transform,
        payload,
        scales,
        codec_padding_trits,
        codec_padding_bits,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct TensorPhysical {
    payload_bytes: u64,
    scales_bytes: u64,
    codec_padding_trits: u64,
    codec_padding_bits: u64,
}

/// Return the one-plane lower bound for a tensor without walking its tiles.
///
/// The package map may raise a tile from one plane to two or three, but the
/// semantic format always requires plane zero. Computing that compulsory
/// plane in closed form keeps malformed giant shapes from turning header
/// validation into an attacker-controlled tile-count loop.
fn minimum_tensor_physical(
    codec: SaltV2Codec,
    logical_coefficients: usize,
) -> Result<TensorPhysical, SaltV2PackageError> {
    fn checked_repeated_plane(
        codec: SaltV2Codec,
        logical_len: usize,
        repetitions: usize,
    ) -> Result<TensorPhysical, SaltV2PackageError> {
        if repetitions == 0 {
            return Ok(TensorPhysical::default());
        }

        let stored_trits = stored_trit_count(codec, logical_len)?;
        let plane_ledger = codec.ledger(stored_trits)?;
        let repetitions =
            u64::try_from(repetitions).map_err(|_| SaltV2PackageError::LengthOverflow)?;
        let repeated = |value: usize| {
            u64::try_from(value)
                .map_err(|_| SaltV2PackageError::LengthOverflow)?
                .checked_mul(repetitions)
                .ok_or(SaltV2PackageError::LengthOverflow)
        };
        let scale_bytes = logical_len
            .div_ceil(SALT_V2_SCALE_GROUP_SIZE)
            .checked_mul(2)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let padding_trits = stored_trits
            .checked_sub(logical_len)
            .and_then(|value| value.checked_add(plane_ledger.canonical_padding_trits))
            .ok_or(SaltV2PackageError::LengthOverflow)?;

        Ok(TensorPhysical {
            payload_bytes: repeated(plane_ledger.physical_bytes)?,
            scales_bytes: repeated(scale_bytes)?,
            codec_padding_trits: repeated(padding_trits)?,
            codec_padding_bits: u64::from(plane_ledger.canonical_padding_bits)
                .checked_mul(repetitions)
                .ok_or(SaltV2PackageError::LengthOverflow)?,
        })
    }

    let full_tiles = logical_coefficients / SALT_V2_ALLOCATION_TILE_SIZE;
    let tail = logical_coefficients % SALT_V2_ALLOCATION_TILE_SIZE;
    let full = checked_repeated_plane(codec, SALT_V2_ALLOCATION_TILE_SIZE, full_tiles)?;
    let tail = checked_repeated_plane(codec, tail, usize::from(tail != 0))?;

    Ok(TensorPhysical {
        payload_bytes: full
            .payload_bytes
            .checked_add(tail.payload_bytes)
            .ok_or(SaltV2PackageError::LengthOverflow)?,
        scales_bytes: full
            .scales_bytes
            .checked_add(tail.scales_bytes)
            .ok_or(SaltV2PackageError::LengthOverflow)?,
        codec_padding_trits: full
            .codec_padding_trits
            .checked_add(tail.codec_padding_trits)
            .ok_or(SaltV2PackageError::LengthOverflow)?,
        codec_padding_bits: full
            .codec_padding_bits
            .checked_add(tail.codec_padding_bits)
            .ok_or(SaltV2PackageError::LengthOverflow)?,
    })
}

fn expected_tensor_physical(
    codec: SaltV2Codec,
    logical_coefficients: usize,
    plane_counts: impl ExactSizeIterator<Item = usize>,
) -> Result<TensorPhysical, SaltV2PackageError> {
    let expected_tiles = logical_coefficients.div_ceil(SALT_V2_ALLOCATION_TILE_SIZE);
    let got = plane_counts.len();
    if got != expected_tiles {
        return Err(SaltV2PackageError::WrongTileCount {
            expected: expected_tiles,
            got,
        });
    }
    let mut physical = TensorPhysical::default();
    for (tile_index, plane_count) in plane_counts.enumerate() {
        if !(1..=SALT_V2_MAX_PLANES).contains(&plane_count) {
            return Err(SaltV2PackageError::InvalidPlaneCount { got: plane_count });
        }
        let consumed = tile_index
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let logical_len = (logical_coefficients - consumed).min(SALT_V2_ALLOCATION_TILE_SIZE);
        let stored_trits = stored_trit_count(codec, logical_len)?;
        let plane_ledger = codec.ledger(stored_trits)?;
        let planes_u64 =
            u64::try_from(plane_count).map_err(|_| SaltV2PackageError::LengthOverflow)?;
        physical.payload_bytes = physical
            .payload_bytes
            .checked_add(
                u64::try_from(plane_ledger.physical_bytes)
                    .map_err(|_| SaltV2PackageError::LengthOverflow)?
                    .checked_mul(planes_u64)
                    .ok_or(SaltV2PackageError::LengthOverflow)?,
            )
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let scale_count = logical_len.div_ceil(SALT_V2_SCALE_GROUP_SIZE);
        physical.scales_bytes = physical
            .scales_bytes
            .checked_add(
                u64::try_from(scale_count)
                    .map_err(|_| SaltV2PackageError::LengthOverflow)?
                    .checked_mul(2)
                    .and_then(|value| value.checked_mul(planes_u64))
                    .ok_or(SaltV2PackageError::LengthOverflow)?,
            )
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let shape_padding = stored_trits - logical_len;
        physical.codec_padding_trits = physical
            .codec_padding_trits
            .checked_add(
                u64::try_from(shape_padding + plane_ledger.canonical_padding_trits)
                    .map_err(|_| SaltV2PackageError::LengthOverflow)?
                    .checked_mul(planes_u64)
                    .ok_or(SaltV2PackageError::LengthOverflow)?,
            )
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        physical.codec_padding_bits = physical
            .codec_padding_bits
            .checked_add(
                u64::from(plane_ledger.canonical_padding_bits)
                    .checked_mul(planes_u64)
                    .ok_or(SaltV2PackageError::LengthOverflow)?,
            )
            .ok_or(SaltV2PackageError::LengthOverflow)?;
    }
    Ok(physical)
}

fn decode_tensor_tiles(
    codec: SaltV2Codec,
    logical_coefficients: usize,
    plane_counts: impl ExactSizeIterator<Item = usize>,
    payload: &[u8],
    scales: &[u8],
) -> Result<Vec<SaltV2Tile>, SaltV2PackageError> {
    let mut payload_cursor = Cursor::new(payload);
    let mut scale_cursor = Cursor::new(scales);
    let mut tiles = Vec::new();
    let tile_count = plane_counts.len();
    tiles
        .try_reserve_exact(tile_count)
        .map_err(|_| SaltV2PackageError::AllocationFailed)?;

    for (tile_index, plane_count) in plane_counts.enumerate() {
        let consumed = tile_index
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let logical_len = (logical_coefficients - consumed).min(SALT_V2_ALLOCATION_TILE_SIZE);
        let stored_trits = stored_trit_count(codec, logical_len)?;
        let packed_len = codec.ledger(stored_trits)?.physical_bytes;
        let scale_count = logical_len.div_ceil(SALT_V2_SCALE_GROUP_SIZE);
        let mut planes = Vec::new();
        planes
            .try_reserve_exact(plane_count)
            .map_err(|_| SaltV2PackageError::AllocationFailed)?;
        for _ in 0..plane_count {
            let packed = payload_cursor.take(packed_len)?;
            let trits = unpack_salt_v2_plane(codec, packed, logical_len)?;
            let scale_bytes = scale_cursor.take(
                scale_count
                    .checked_mul(2)
                    .ok_or(SaltV2PackageError::LengthOverflow)?,
            )?;
            let decoded_scales = scale_bytes
                .chunks_exact(2)
                .map(|bytes| f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
                .collect::<Vec<_>>();
            let raw_trits = trits.into_iter().map(Trit::get).collect();
            planes.push(SaltV2Plane::new(raw_trits, decoded_scales)?);
        }
        tiles.push(SaltV2Tile::new(planes)?);
    }
    if payload_cursor.remaining() != 0 {
        return Err(SaltV2PackageError::UnexpectedTrailingData {
            remaining: payload_cursor.remaining(),
            expected_padding: 0,
        });
    }
    if scale_cursor.remaining() != 0 {
        return Err(SaltV2PackageError::UnexpectedTrailingData {
            remaining: scale_cursor.remaining(),
            expected_padding: 0,
        });
    }
    Ok(tiles)
}

/// Pack one semantic SALT V2 plane using the package's canonical shape-padding rules.
///
/// This is the host packing authority used by package serialization and device
/// upload. In particular, S34 ragged tails are canonicalized exactly once here.
///
/// # Errors
/// Returns a package or codec error when `codec` cannot represent the plane
/// canonically.
pub fn pack_salt_v2_plane(
    codec: SaltV2Codec,
    trits: &[Trit],
) -> Result<Vec<u8>, SaltV2PackageError> {
    Ok(match codec {
        SaltV2Codec::D2 => pack_d2(trits)?,
        SaltV2Codec::B3 => pack_b3(trits)?,
        SaltV2Codec::S34 => pack_s34(&canonical_s34_trits(trits)?)?,
    })
}

/// Unpack one canonical package-codec plane to its logical ternary coefficients.
///
/// For canonical nonempty planes no longer than one allocation tile, this is the
/// semantic inverse of [`pack_salt_v2_plane`]. For S34, `logical_len` excludes
/// canonical shape padding; the decoder validates that padding before returning
/// exactly the logical coefficients.
///
/// # Errors
/// Returns [`SaltV2PackageError::InvalidPlaneLength`] when `logical_len` is zero
/// or exceeds one allocation tile, [`SaltV2PackageError::AllocationFailed`] when
/// decoded storage cannot be reserved, or a codec/package error when the physical
/// payload length, ternary codes, or S34 shape padding are non-canonical.
pub fn unpack_salt_v2_plane(
    codec: SaltV2Codec,
    packed: &[u8],
    logical_len: usize,
) -> Result<Vec<Trit>, SaltV2PackageError> {
    if logical_len == 0 || logical_len > SALT_V2_ALLOCATION_TILE_SIZE {
        return Err(SaltV2PackageError::InvalidPlaneLength { got: logical_len });
    }
    let stored_len = stored_trit_count(codec, logical_len)?;
    let mut decoded = Vec::new();
    decoded
        .try_reserve_exact(stored_len)
        .map_err(|_| SaltV2PackageError::AllocationFailed)?;
    unpack_semantic_plane_into(codec, packed, logical_len, &mut decoded)?;
    Ok(decoded)
}

fn unpack_semantic_plane_into(
    codec: SaltV2Codec,
    packed: &[u8],
    logical_len: usize,
    decoded: &mut Vec<Trit>,
) -> Result<(), SaltV2PackageError> {
    match codec {
        SaltV2Codec::D2 => unpack_d2_into(packed, logical_len, decoded)?,
        SaltV2Codec::B3 => unpack_b3_into(packed, logical_len, decoded)?,
        SaltV2Codec::S34 => {
            let stored_len = stored_trit_count(codec, logical_len)?;
            unpack_s34_into(packed, stored_len, decoded)?;
            validate_s34_shape_padding(decoded, logical_len)?;
            decoded.truncate(logical_len);
        }
    }
    Ok(())
}

fn validate_s34_shape_padding(
    decoded: &[Trit],
    logical_len: usize,
) -> Result<(), SaltV2PackageError> {
    if decoded.len() == logical_len {
        return Ok(());
    }
    let group_start = logical_len / 4 * 4;
    let logical_tail = &decoded[group_start..logical_len];
    let zero_count = logical_tail.iter().filter(|trit| trit.is_zero()).count();
    if zero_count > 1 {
        return Err(SaltV2PackageError::S34IncompatibleGroup {
            group_index: group_start / 4,
            logical_trits: logical_tail.len(),
            zero_count,
        });
    }
    let mut padding_start = logical_len;
    if zero_count == 0 {
        if decoded.get(padding_start) != Some(&Trit::ZERO) {
            return Err(SaltV2PackageError::NonCanonicalS34ShapePadding);
        }
        padding_start += 1;
    }
    if decoded[padding_start..]
        .iter()
        .any(|trit| *trit != Trit::NEG)
    {
        return Err(SaltV2PackageError::NonCanonicalS34ShapePadding);
    }
    Ok(())
}

fn canonical_s34_trits(trits: &[Trit]) -> Result<Vec<Trit>, SaltV2PackageError> {
    let mut canonical = Vec::new();
    let stored_len = trits
        .len()
        .div_ceil(4)
        .checked_mul(4)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    canonical
        .try_reserve_exact(stored_len)
        .map_err(|_| SaltV2PackageError::AllocationFailed)?;
    canonical.extend_from_slice(trits);

    let full_groups = trits.len() / 4;
    for (group_index, group) in trits[..full_groups * 4].chunks_exact(4).enumerate() {
        let zero_count = group.iter().filter(|trit| trit.is_zero()).count();
        if zero_count != 1 {
            return Err(SaltV2PackageError::S34IncompatibleGroup {
                group_index,
                logical_trits: 4,
                zero_count,
            });
        }
    }

    let tail = &trits[full_groups * 4..];
    if !tail.is_empty() {
        let zero_count = tail.iter().filter(|trit| trit.is_zero()).count();
        if zero_count > 1 {
            return Err(SaltV2PackageError::S34IncompatibleGroup {
                group_index: full_groups,
                logical_trits: tail.len(),
                zero_count,
            });
        }
        if zero_count == 0 {
            canonical.push(Trit::ZERO);
        }
        canonical.resize(stored_len, Trit::NEG);
    }
    Ok(canonical)
}

fn stored_trit_count(codec: SaltV2Codec, logical_len: usize) -> Result<usize, SaltV2PackageError> {
    if codec == SaltV2Codec::S34 {
        logical_len
            .div_ceil(4)
            .checked_mul(4)
            .ok_or(SaltV2PackageError::LengthOverflow)
    } else {
        Ok(logical_len)
    }
}

struct EncodedPresenceMap {
    bytes: Vec<u8>,
    package_embedded_bits: u8,
    tensor_embedded_bits: usize,
    logical_bits: usize,
    total_tiles: usize,
    packed_tensor_count: u32,
}

fn encode_presence_maps(
    tensors: &[SaltV2Tensor],
) -> Result<EncodedPresenceMap, SaltV2PackageError> {
    let total_tiles = tensors.iter().try_fold(0usize, |total, tensor| {
        total
            .checked_add(tensor.tiles.len())
            .ok_or(SaltV2PackageError::LengthOverflow)
    })?;
    let full_tiles = tensors.iter().try_fold(0usize, |total, tensor| {
        total
            .checked_add(tensor.logical_coefficients / SALT_V2_ALLOCATION_TILE_SIZE)
            .ok_or(SaltV2PackageError::LengthOverflow)
    })?;
    let ragged_tensor_count = tensors
        .iter()
        .filter(|tensor| {
            !tensor
                .logical_coefficients
                .is_multiple_of(SALT_V2_ALLOCATION_TILE_SIZE)
        })
        .count();
    let logical_bits = total_tiles
        .checked_mul(2)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    let full_tile_bits = full_tiles
        .checked_mul(2)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    let tensor_embedded_bits = ragged_tensor_count
        .checked_mul(2)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    let complete_bits = full_tile_bits / 8 * 8;
    let mut bytes = vec![0u8; full_tile_bits / 8];
    let mut embedded_value = 0u8;
    let mut global_tile = 0usize;
    for tensor in tensors {
        let tensor_full_tiles = tensor.logical_coefficients / SALT_V2_ALLOCATION_TILE_SIZE;
        for tile in tensor.tiles.iter().take(tensor_full_tiles) {
            let plane_two_bit = global_tile
                .checked_mul(2)
                .ok_or(SaltV2PackageError::LengthOverflow)?;
            if tile.planes.len() >= 2 {
                set_global_map_bit(
                    &mut bytes,
                    &mut embedded_value,
                    complete_bits,
                    plane_two_bit,
                );
            }
            if tile.planes.len() >= 3 {
                set_global_map_bit(
                    &mut bytes,
                    &mut embedded_value,
                    complete_bits,
                    plane_two_bit + 1,
                );
            }
            global_tile += 1;
        }
    }
    debug_assert_eq!(global_tile, full_tiles);
    let package_embedded_bits = (full_tile_bits % 8) as u8;
    debug_assert!(u32::from(package_embedded_bits) <= SALT_V2_EMBEDDED_MAP_CAPACITY_BITS);
    debug_assert_eq!(
        bytes.len() * 8 + usize::from(package_embedded_bits) + tensor_embedded_bits,
        logical_bits
    );
    let tensor_count =
        u32::try_from(tensors.len()).map_err(|_| SaltV2PackageError::LengthOverflow)?;
    debug_assert!(tensor_count <= SALT_V2_MAX_TENSORS as u32);
    let packed_tensor_count =
        tensor_count | (u32::from(embedded_value) << SALT_V2_TENSOR_COUNT_BITS);

    Ok(EncodedPresenceMap {
        bytes,
        package_embedded_bits,
        tensor_embedded_bits,
        logical_bits,
        total_tiles,
        packed_tensor_count,
    })
}

#[derive(Clone, Copy)]
struct ValidatedPresenceMap<'a> {
    embedded_value: u8,
    bytes: &'a [u8],
    total_tiles: usize,
    complete_bits: usize,
}

impl<'a> ValidatedPresenceMap<'a> {
    fn new(
        embedded_value: u8,
        bytes: &'a [u8],
        total_tiles: usize,
    ) -> Result<Self, SaltV2PackageError> {
        let total_bits = total_tiles
            .checked_mul(2)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let expected = total_bits / 8;
        if bytes.len() != expected {
            return Err(SaltV2PackageError::DeclaredFieldMismatch {
                field: "presence map bytes",
                declared: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
                expected: u64::try_from(expected).unwrap_or(u64::MAX),
            });
        }
        let embedded_bits = total_bits % 8;
        if u32::try_from(embedded_bits).expect("map remainder fits u32")
            < SALT_V2_EMBEDDED_MAP_CAPACITY_BITS
            && embedded_value >> embedded_bits != 0
        {
            return Err(SaltV2PackageError::NonCanonicalMapPadding);
        }
        let complete_bits = bytes
            .len()
            .checked_mul(8)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        let map = Self {
            embedded_value,
            bytes,
            total_tiles,
            complete_bits,
        };
        for tile_index in 0..total_tiles {
            let (plane_two, plane_three) = map.plane_bits(tile_index);
            if plane_three && !plane_two {
                return Err(SaltV2PackageError::NonNestedPlaneMap { tile_index });
            }
        }
        Ok(map)
    }

    const fn len(self) -> usize {
        self.total_tiles
    }

    fn counts(
        self,
        range: core::ops::Range<usize>,
        ragged_plane_count: Option<usize>,
    ) -> PresencePlaneCounts<'a> {
        debug_assert!(range.start <= range.end && range.end <= self.total_tiles);
        PresencePlaneCounts {
            map: self,
            next: range.start,
            end: range.end,
            ragged_plane_count,
        }
    }

    fn plane_bits(self, tile_index: usize) -> (bool, bool) {
        let plane_two_bit = tile_index * 2;
        (
            global_map_bit(
                self.bytes,
                self.embedded_value,
                self.complete_bits,
                plane_two_bit,
            ),
            global_map_bit(
                self.bytes,
                self.embedded_value,
                self.complete_bits,
                plane_two_bit + 1,
            ),
        )
    }

    fn plane_count(self, tile_index: usize) -> usize {
        let (plane_two, plane_three) = self.plane_bits(tile_index);
        1 + usize::from(plane_two) + usize::from(plane_three)
    }
}

struct PresencePlaneCounts<'a> {
    map: ValidatedPresenceMap<'a>,
    next: usize,
    end: usize,
    ragged_plane_count: Option<usize>,
}

impl Iterator for PresencePlaneCounts<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next != self.end {
            let tile_index = self.next;
            self.next += 1;
            return Some(self.map.plane_count(tile_index));
        }
        self.ragged_plane_count.take()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.next + usize::from(self.ragged_plane_count.is_some());
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PresencePlaneCounts<'_> {}

fn presence_map_len(tile_count: usize) -> Result<usize, SaltV2PackageError> {
    tile_count
        .checked_mul(2)
        .map(|bits| bits / 8)
        .ok_or(SaltV2PackageError::LengthOverflow)
}

fn plane_count_map_value(plane_count: usize) -> u8 {
    match plane_count {
        1 => 0b00,
        2 => 0b01,
        3 => 0b11,
        _ => unreachable!("validated tiles have one to three planes"),
    }
}

fn map_value_plane_count(map_value: u8, tile_index: usize) -> Result<usize, SaltV2PackageError> {
    let plane_two = map_value & 0b01 != 0;
    let plane_three = map_value & 0b10 != 0;
    if plane_three && !plane_two {
        return Err(SaltV2PackageError::NonNestedPlaneMap { tile_index });
    }
    Ok(1 + usize::from(plane_two) + usize::from(plane_three))
}

fn set_global_map_bit(
    bytes: &mut [u8],
    embedded_value: &mut u8,
    complete_bits: usize,
    bit_index: usize,
) {
    if bit_index < complete_bits {
        bytes[bit_index / 8] |= 1 << (bit_index % 8);
    } else {
        *embedded_value |= 1 << (bit_index - complete_bits);
    }
}

fn global_map_bit(
    bytes: &[u8],
    embedded_value: u8,
    complete_bits: usize,
    bit_index: usize,
) -> bool {
    if bit_index < complete_bits {
        bytes[bit_index / 8] & (1 << (bit_index % 8)) != 0
    } else {
        embedded_value & (1 << (bit_index - complete_bits)) != 0
    }
}

fn checked_dimension_product(dims: &[u64]) -> Result<u64, SaltV2PackageError> {
    if dims.is_empty() {
        return Err(SaltV2PackageError::EmptyDimensions);
    }
    let mut product = 1u64;
    for (index, dim) in dims.iter().copied().enumerate() {
        if dim == 0 {
            return Err(SaltV2PackageError::ZeroDimension { index });
        }
        product = product
            .checked_mul(dim)
            .ok_or(SaltV2PackageError::DimensionProductOverflow)?;
    }
    Ok(product)
}

fn codec_tag(codec: SaltV2Codec) -> u8 {
    match codec {
        SaltV2Codec::D2 => 1,
        SaltV2Codec::B3 => 2,
        SaltV2Codec::S34 => 3,
    }
}

fn codec_from_tag(tag: u8) -> Result<SaltV2Codec, SaltV2PackageError> {
    match tag {
        1 => Ok(SaltV2Codec::D2),
        2 => Ok(SaltV2Codec::B3),
        3 => Ok(SaltV2Codec::S34),
        other => Err(SaltV2PackageError::UnsupportedCodec(other)),
    }
}

fn checked_ledger_add(target: &mut u64, amount: usize) -> Result<(), SaltV2PackageError> {
    *target = target
        .checked_add(u64::try_from(amount).map_err(|_| SaltV2PackageError::LengthOverflow)?)
        .ok_or(SaltV2PackageError::LengthOverflow)?;
    Ok(())
}

fn require_declared(
    field: &'static str,
    declared: u64,
    expected: u64,
) -> Result<(), SaltV2PackageError> {
    if declared != expected {
        return Err(SaltV2PackageError::DeclaredFieldMismatch {
            field,
            declared,
            expected,
        });
    }
    Ok(())
}

fn require_at_least(
    field: &'static str,
    declared: u64,
    minimum: u64,
) -> Result<(), SaltV2PackageError> {
    if declared < minimum {
        return Err(SaltV2PackageError::DeclaredFieldBelowMinimum {
            field,
            declared,
            minimum,
        });
    }
    Ok(())
}

fn alignment_padding(len: usize) -> usize {
    (SALT_V2_PACKAGE_ALIGNMENT - len % SALT_V2_PACKAGE_ALIGNMENT) % SALT_V2_PACKAGE_ALIGNMENT
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_transform(bytes: &mut Vec<u8>, transform: SaltV2Transform) {
    let start = bytes.len();
    match transform {
        SaltV2Transform::None => {
            bytes.push(0);
            bytes.extend_from_slice(&[0; 7]);
            push_u64(bytes, 0);
            push_u64(bytes, 0);
        }
        SaltV2Transform::SignedRht { seed, domain } => {
            bytes.push(1);
            bytes.extend_from_slice(&[0; 7]);
            push_u64(bytes, seed);
            push_u64(bytes, domain);
        }
    }
    debug_assert_eq!(bytes.len() - start, SALT_V2_TRANSFORM_METADATA_BYTES);
}

fn read_transform(cursor: &mut Cursor<'_>) -> Result<SaltV2Transform, SaltV2PackageError> {
    let tag = cursor.u8()?;
    let reserved = cursor.take(7)?;
    let seed = cursor.u64()?;
    let domain = cursor.u64()?;
    if reserved.iter().any(|byte| *byte != 0) {
        return Err(SaltV2PackageError::NonCanonicalTransformMetadata);
    }
    match tag {
        0 if seed == 0 && domain == 0 => Ok(SaltV2Transform::None),
        0 => Err(SaltV2PackageError::NonCanonicalTransformMetadata),
        1 => Ok(SaltV2Transform::SignedRht { seed, domain }),
        other => Err(SaltV2PackageError::UnsupportedTransformTag(other)),
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SaltV2PackageError> {
        let end = self
            .position
            .checked_add(count)
            .ok_or(SaltV2PackageError::LengthOverflow)?;
        if end > self.bytes.len() {
            return Err(SaltV2PackageError::Truncated {
                needed: count,
                remaining: self.remaining(),
            });
        }
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, SaltV2PackageError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SaltV2PackageError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> Result<u32, SaltV2PackageError> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes(
            bytes.try_into().expect("cursor returned four bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, SaltV2PackageError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(
            bytes.try_into().expect("cursor returned eight bytes"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn structured_values(len: usize, sign_phase: usize) -> Vec<i8> {
        (0..len)
            .map(|index| {
                if index % 4 == 0 {
                    0
                } else if (index + sign_phase).is_multiple_of(2) {
                    1
                } else {
                    -1
                }
            })
            .collect()
    }

    fn plane(len: usize, sign_phase: usize) -> SaltV2Plane {
        let scale_count = len.div_ceil(SALT_V2_SCALE_GROUP_SIZE);
        let scales = (0..scale_count)
            .map(|group| f16::from_f32(0.5 + (group + sign_phase) as f32 / 16.0))
            .collect();
        SaltV2Plane::new(structured_values(len, sign_phase), scales).expect("valid test plane")
    }

    fn tile(len: usize, plane_count: usize, phase: usize) -> SaltV2Tile {
        SaltV2Tile::new(
            (0..plane_count)
                .map(|plane_index| plane(len, phase + plane_index))
                .collect(),
        )
        .expect("valid test tile")
    }

    fn ragged_tensor(name: &str) -> SaltV2Tensor {
        SaltV2Tensor::new(
            name,
            vec![599],
            vec![tile(256, 1, 0), tile(256, 3, 1), tile(87, 2, 2)],
        )
        .expect("valid ragged tensor")
    }

    fn ragged_package(codec: SaltV2Codec) -> SaltV2Package {
        SaltV2Package::new(codec, vec![ragged_tensor("ragged")]).expect("valid package")
    }

    fn one_tile_tensor(name: &str, len: usize) -> SaltV2Tensor {
        SaltV2Tensor::new(name, vec![len as u64], vec![tile(len, 1, 0)])
            .expect("valid one-tile tensor")
    }

    fn packed_tensor_count_offset() -> usize {
        12
    }

    fn tensor_sections_offset(name: &str, rank: usize) -> usize {
        SALT_V2_PACKAGE_HEADER_BYTES + SALT_V2_TENSOR_HEADER_BYTES + name.len() + rank * 8
    }

    #[test]
    fn ragged_plane_counts_store_only_present_d2_planes() {
        let package = ragged_package(SaltV2Codec::D2);
        let encoded = write_salt_v2_package(&package).expect("encode D2 package");

        // Independent worked example:
        // tile payloads = 1*ceil(256/4) + 3*ceil(256/4) + 2*ceil(87/4).
        assert_eq!(encoded.ledger.payload_bytes, 300);
        assert_eq!(encoded.ledger.scales_bytes, 20);
        assert_eq!(encoded.ledger.maps_bytes, 0);
        assert_eq!(encoded.ledger.allocation_map_bits, 6);
        assert_eq!(encoded.ledger.allocation_map_embedded_bits, 6);
        assert_eq!(encoded.ledger.allocation_map_package_embedded_bits, 4);
        assert_eq!(encoded.ledger.allocation_map_tensor_embedded_bits, 2);
        assert_eq!(encoded.ledger.allocation_tiles, 3);
        // The logical map is exactly two bits per tile. The complete full-tile
        // bytes are the only added map storage; the four terminal full-tile
        // bits and two ragged-tile bits occupy mandatory count words and are
        // still reported explicitly.
        assert_eq!(
            encoded.ledger.allocation_map_bits as f64
                / encoded.ledger.allocation_capacity_coefficients as f64,
            0.007_812_5
        );
        assert!(encoded.ledger.allocation_map_bits as f64 / 599.0 > 0.01);
        assert!(encoded.ledger.maps_bytes as f64 * 8.0 / 599.0 <= 0.007_812_5);
        // Padding every tile to max-P would consume 450 payload bytes.
        assert!(encoded.ledger.payload_bytes < 450);

        let decoded = read_salt_v2_package(&encoded.bytes).expect("decode D2 package");
        let plane_counts = decoded.package.tensors()[0]
            .tiles()
            .iter()
            .map(|tile| tile.planes().len())
            .collect::<Vec<_>>();
        assert_eq!(plane_counts, [1, 3, 2]);
        assert_eq!(decoded.package, package);
    }

    #[test]
    fn d2_and_b3_round_trip_a_ragged_final_tile() {
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3] {
            let package = ragged_package(codec);
            let encoded = write_salt_v2_package(&package).expect("encode ragged package");
            let decoded = read_salt_v2_package(&encoded.bytes).expect("decode ragged package");

            assert_eq!(decoded.package, package, "codec {codec:?}");
            assert_eq!(decoded.package.tensors()[0].tiles()[2].logical_len(), 87);
            assert_eq!(decoded.ledger, encoded.ledger);
        }
    }

    #[test]
    fn every_short_final_group_repack_is_canonical_and_byte_exact() {
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            for logical_len in 1usize..=SALT_V2_ALLOCATION_TILE_SIZE {
                for plane_count in 1usize..=SALT_V2_MAX_PLANES {
                    let tensor = SaltV2Tensor::new(
                        "tail",
                        vec![logical_len as u64],
                        vec![tile(logical_len, plane_count, logical_len)],
                    )
                    .expect("valid short tensor");
                    let package =
                        SaltV2Package::new(codec, vec![tensor]).expect("valid short package");
                    let encoded = write_salt_v2_package(&package).expect("encode short package");
                    let decoded =
                        read_salt_v2_package(&encoded.bytes).expect("decode short package");
                    let repacked =
                        write_salt_v2_package(&decoded.package).expect("repack short package");

                    let stored_len = if codec == SaltV2Codec::S34 {
                        logical_len.div_ceil(4) * 4
                    } else {
                        logical_len
                    };
                    let per_plane_payload = codec
                        .ledger(stored_len)
                        .expect("short payload ledger")
                        .physical_bytes as u64;
                    assert_eq!(
                        encoded.ledger.payload_bytes,
                        per_plane_payload * plane_count as u64,
                        "{codec:?}, len={logical_len}, P={plane_count}"
                    );
                    assert_eq!(
                        encoded.ledger.scales_bytes,
                        (logical_len.div_ceil(SALT_V2_SCALE_GROUP_SIZE) * 2 * plane_count) as u64
                    );
                    assert_eq!(encoded.ledger.allocation_map_bits, 2);
                    assert_eq!(encoded.ledger.maps_bytes, 0);
                    assert_eq!(decoded.ledger, encoded.ledger);
                    assert_eq!(repacked.ledger, encoded.ledger);
                    assert_eq!(repacked.bytes, encoded.bytes);
                }
            }
        }
    }

    #[test]
    fn every_codec_round_trips_with_an_exact_file_ledger() {
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let package = ragged_package(codec);
            let encoded = write_salt_v2_package(&package).expect("encode package");
            let decoded = read_salt_v2_package(&encoded.bytes).expect("decode package");

            assert_eq!(decoded.package, package, "codec {codec:?}");
            assert_eq!(decoded.ledger, encoded.ledger, "codec {codec:?}");
            assert_eq!(encoded.ledger.total_bytes, encoded.bytes.len() as u64);
            assert_eq!(
                encoded.ledger.headers_bytes
                    + encoded.ledger.transform_bytes
                    + encoded.ledger.maps_bytes
                    + encoded.ledger.payload_bytes
                    + encoded.ledger.scales_bytes
                    + encoded.ledger.padding_bytes,
                encoded.ledger.total_bytes
            );
            assert_eq!(
                encoded.ledger.headers_bytes
                    + encoded.ledger.transform_bytes
                    + encoded.ledger.maps_bytes
                    + encoded.ledger.payload_bytes
                    + encoded.ledger.scales_bytes,
                encoded.ledger.serialized_unpadded_bytes
            );
            assert_eq!(
                encoded.ledger.maps_bytes * 8 + encoded.ledger.allocation_map_embedded_bits,
                encoded.ledger.allocation_map_bits
            );
            assert_eq!(
                encoded.ledger.allocation_map_embedded_bits,
                u64::from(encoded.ledger.allocation_map_package_embedded_bits)
                    + encoded.ledger.allocation_map_tensor_embedded_bits
            );
            assert_eq!(
                encoded.ledger.allocation_map_bits,
                encoded.ledger.allocation_tiles * 2
            );
            assert!(encoded.ledger.serialized_unpadded_bytes < encoded.ledger.total_bytes);
        }
    }

    #[test]
    fn indexed_runtime_ledger_counts_compact_map_and_rank_prefixes() {
        let tensor = ragged_tensor("ragged");
        let ledger = SaltV2IndexedRuntimeLedger::for_tensor(&tensor, SaltV2Codec::D2)
            .expect("measure indexed runtime layout");

        // Independent worked example: the three tiles contain 1 + 3 + 2 planes.
        // D2 payload and G128 scales are established by the package fixture above.
        // Six map bits ride in the mandatory runtime scalar; no rank prefix is
        // needed before tile 256.
        assert_eq!(ledger.payload_bytes(), 300);
        assert_eq!(ledger.scale_bytes(), 20);
        assert_eq!(ledger.allocation_map_bytes(), 0);
        assert_eq!(ledger.allocation_map_bits(), 6);
        assert_eq!(ledger.allocation_map_embedded_bits(), 6);
        assert_eq!(ledger.rank_prefix_bytes(), 0);
        assert_eq!(ledger.dense_shadow_bytes(), 0);
        assert_eq!(ledger.steady_resident_bytes(), 320);
    }

    #[test]
    fn indexed_runtime_metadata_stays_below_point_zero_one_bpw() {
        for tile_count in [1usize, 4, 255, 256, 257, 512, 513] {
            let tensor = SaltV2Tensor::new(
                "bounded-runtime",
                vec![(tile_count * SALT_V2_ALLOCATION_TILE_SIZE) as u64],
                (0..tile_count)
                    .map(|index| tile(SALT_V2_ALLOCATION_TILE_SIZE, 1 + index % 3, index))
                    .collect(),
            )
            .expect("valid full-tile tensor");
            let ledger = SaltV2IndexedRuntimeLedger::for_tensor(&tensor, SaltV2Codec::D2)
                .expect("runtime ledger");
            let metadata_bits = ledger
                .allocation_map_bits()
                .checked_add(ledger.rank_prefix_bytes() * 8)
                .expect("metadata-bit count fits");
            assert!(
                metadata_bits * 100 <= (tile_count * SALT_V2_ALLOCATION_TILE_SIZE) as u64,
                "{tile_count} tiles: {metadata_bits} logical metadata bits"
            );
        }
    }

    #[test]
    fn s34_shape_padding_is_canonical_and_discarded() {
        let tensor = SaltV2Tensor::new(
            "tail",
            vec![3],
            vec![
                SaltV2Tile::new(vec![
                    SaltV2Plane::new(vec![0, 1, -1], vec![f16::ONE]).expect("valid plane"),
                ])
                .expect("valid tile"),
            ],
        )
        .expect("valid tensor");
        let package = SaltV2Package::new(SaltV2Codec::S34, vec![tensor]).expect("valid S34");
        let encoded = write_salt_v2_package(&package).expect("encode S34 tail");
        let decoded = read_salt_v2_package(&encoded.bytes).expect("decode S34 tail");

        assert_eq!(decoded.package, package);
        assert_eq!(decoded.package.tensors()[0].logical_coefficients(), 3);
        assert_eq!(decoded.ledger.codec_padding_trits, 1);

        // Replace the canonical negative shape pad with positive. It is still a
        // structurally valid 3:4 group, but it is not the one canonical byte form.
        let alternate =
            pack_s34(&[Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS]).expect("alternate S34 group");
        let mut noncanonical = encoded.bytes;
        let payload_offset = tensor_sections_offset("tail", 1);
        noncanonical[payload_offset] = alternate[0];
        assert_eq!(
            read_salt_v2_package(&noncanonical),
            Err(SaltV2PackageError::NonCanonicalS34ShapePadding)
        );
    }

    #[test]
    fn optional_plane_maps_stay_below_point_zero_one_bpw_on_full_tiles() {
        let tile_count = 128usize;
        let tensor = SaltV2Tensor::new(
            "full",
            vec![(tile_count * SALT_V2_ALLOCATION_TILE_SIZE) as u64],
            (0..tile_count)
                .map(|index| tile(SALT_V2_ALLOCATION_TILE_SIZE, 1 + index % 3, index))
                .collect(),
        )
        .expect("full-tile tensor");
        let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).expect("valid package");
        let encoded = write_salt_v2_package(&package).expect("encode full-tile maps");

        assert_eq!(encoded.ledger.maps_bytes, 32);
        assert_eq!(encoded.ledger.allocation_map_embedded_bits, 0);
        assert_eq!(encoded.ledger.allocation_map_package_embedded_bits, 0);
        assert_eq!(encoded.ledger.allocation_map_tensor_embedded_bits, 0);
        let map_bpw = encoded.ledger.allocation_map_bits as f64
            / encoded.ledger.allocation_capacity_coefficients as f64;
        assert_eq!(map_bpw, 0.007_812_5);
        assert!(map_bpw <= 0.01);
        assert_eq!(encoded.ledger.allocation_map_bits, (tile_count * 2) as u64);
    }

    #[test]
    fn one_to_many_tiny_tensors_report_map_bits_without_hiding_them_in_headers() {
        for tensor_count in 1..=64usize {
            let tensors = (0..tensor_count)
                .map(|index| {
                    let plane_count = 1 + index % 3;
                    SaltV2Tensor::new(
                        format!("tiny.{index}"),
                        vec![1],
                        vec![tile(1, plane_count, index)],
                    )
                    .expect("valid tiny tensor")
                })
                .collect();
            let package = SaltV2Package::new(SaltV2Codec::D2, tensors).expect("valid tiny package");
            let encoded = write_salt_v2_package(&package).expect("encode tiny package");

            assert_eq!(encoded.ledger.maps_bytes, 0, "{tensor_count} tensors");
            let actual_logical_bpw =
                encoded.ledger.allocation_map_bits as f64 / tensor_count as f64;
            assert_eq!(actual_logical_bpw, 2.0, "{tensor_count} tiny tensors");
            assert_eq!(
                encoded.ledger.allocation_map_embedded_bits,
                (tensor_count * 2) as u64,
                "{tensor_count} tensors"
            );
            assert_eq!(
                encoded.ledger.allocation_map_package_embedded_bits, 0,
                "{tensor_count} tensors"
            );
            assert_eq!(
                encoded.ledger.allocation_map_tensor_embedded_bits,
                (tensor_count * 2) as u64,
                "{tensor_count} tensors"
            );
            assert_eq!(
                encoded.ledger.maps_bytes * 8 + encoded.ledger.allocation_map_embedded_bits,
                (tensor_count * 2) as u64,
                "{tensor_count} tensors"
            );
            assert_eq!(
                encoded.ledger.maps_bytes as f64 * 8.0 / tensor_count as f64,
                0.0,
                "{tensor_count} tensors"
            );
            assert_eq!(
                read_salt_v2_package(&encoded.bytes)
                    .expect("decode tiny package")
                    .package,
                package
            );
        }
    }

    #[test]
    fn package_global_map_storage_is_exactly_two_bits_per_allocation_tile() {
        for tile_count in [1usize, 31, 32, 33, 34, 63, 64, 65, 129] {
            let tensor = SaltV2Tensor::new(
                "bounded",
                vec![(tile_count * SALT_V2_ALLOCATION_TILE_SIZE) as u64],
                (0..tile_count)
                    .map(|index| tile(SALT_V2_ALLOCATION_TILE_SIZE, 1 + index % 3, index))
                    .collect(),
            )
            .expect("valid bounded tensor");
            let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).expect("valid package");
            let encoded = write_salt_v2_package(&package).expect("encode bounded package");

            assert_eq!(
                encoded.ledger.maps_bytes * 8 + encoded.ledger.allocation_map_embedded_bits,
                (tile_count * 2) as u64,
                "{tile_count} tiles"
            );
            assert_eq!(
                encoded.ledger.allocation_map_package_embedded_bits,
                (tile_count * 2 % 8) as u8,
                "{tile_count} tiles"
            );
            assert_eq!(
                encoded.ledger.allocation_map_tensor_embedded_bits, 0,
                "{tile_count} tiles"
            );
            let logical_coefficients = tile_count * SALT_V2_ALLOCATION_TILE_SIZE;
            assert!(
                encoded.ledger.maps_bytes as f64 * 8.0 / logical_coefficients as f64 <= 0.007_812_5
            );
            let grid_bpw = encoded.ledger.allocation_map_bits as f64
                / encoded.ledger.allocation_capacity_coefficients as f64;
            assert_eq!(grid_bpw, 0.007_812_5);
        }
    }

    #[test]
    fn decoder_rejects_impossible_payload_before_scanning_presence_map() {
        let tensor = SaltV2Tensor::new(
            "bounded",
            vec![(4 * SALT_V2_ALLOCATION_TILE_SIZE) as u64],
            (0..4)
                .map(|index| tile(SALT_V2_ALLOCATION_TILE_SIZE, 1, index))
                .collect(),
        )
        .expect("valid tensor");
        let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).expect("valid package");
        let encoded = write_salt_v2_package(&package).expect("encode package");
        let mut malicious = encoded.bytes;

        // The payload-length word starts 24 bytes into the tensor header. Make
        // it impossible, and independently make the trailing map non-nested.
        let declared_payload_offset = SALT_V2_PACKAGE_HEADER_BYTES + 24;
        malicious[declared_payload_offset..declared_payload_offset + 8]
            .copy_from_slice(&0_u64.to_le_bytes());
        let map_offset = malicious.len()
            - encoded.ledger.padding_bytes as usize
            - encoded.ledger.maps_bytes as usize;
        malicious[map_offset] = 0b10;

        assert_eq!(
            read_salt_v2_package(&malicious),
            Err(SaltV2PackageError::DeclaredFieldBelowMinimum {
                field: "payload bytes",
                declared: 0,
                minimum: 256,
            })
        );
    }

    #[test]
    fn decoder_rejects_giant_claimed_geometry_before_any_tile_count_loop() {
        let declared_payload = 1_u64 << 62;
        let declared_scales = 1_u64 << 58;
        let mut malicious = Vec::new();
        malicious.extend_from_slice(&SALT_V2_PACKAGE_MAGIC);
        push_u16(&mut malicious, SALT_V2_PACKAGE_VERSION);
        malicious.push(codec_tag(SaltV2Codec::D2));
        malicious.push(0);
        push_u32(&mut malicious, 1);
        push_u64(&mut malicious, 0); // patched to the exact file length below

        push_u32(&mut malicious, 1); // one-byte name
        push_u32(&mut malicious, 1); // rank one
        push_u64(&mut malicious, u64::MAX);
        push_u64(&mut malicious, 1_u64 << 56); // ceil(u64::MAX / 256)
        push_u64(&mut malicious, declared_payload);
        push_u64(&mut malicious, declared_scales);
        push_transform(&mut malicious, SaltV2Transform::None);
        malicious.push(b'x');
        push_u64(&mut malicious, u64::MAX);

        let total = u64::try_from(malicious.len()).expect("test package length fits");
        malicious[16..24].copy_from_slice(&total.to_le_bytes());
        assert_eq!(malicious.len(), 97);
        assert_eq!(
            read_salt_v2_package(&malicious),
            Err(SaltV2PackageError::Truncated {
                needed: usize::try_from(declared_payload + declared_scales)
                    .expect("declared sections fit on the test target"),
                remaining: 0,
            })
        );
    }

    #[test]
    fn standalone_map_bytes_meet_the_bound_over_mixed_actual_coefficients() {
        let tiny =
            SaltV2Tensor::new("tiny", vec![1], vec![tile(1, 3, 0)]).expect("valid tiny tensor");
        let mixed = SaltV2Tensor::new(
            "mixed",
            vec![1_281],
            (0..5)
                .map(|index| tile(SALT_V2_ALLOCATION_TILE_SIZE, 1 + index % 3, index))
                .chain([tile(1, 3, 5)])
                .collect(),
        )
        .expect("valid mixed tensor");
        let full = SaltV2Tensor::new(
            "full",
            vec![(3 * SALT_V2_ALLOCATION_TILE_SIZE) as u64],
            (0..3)
                .map(|index| tile(SALT_V2_ALLOCATION_TILE_SIZE, 1 + index % 3, index + 8))
                .collect(),
        )
        .expect("valid full tensor");
        let package = SaltV2Package::new(SaltV2Codec::D2, vec![tiny, mixed, full])
            .expect("valid mixed package");
        let encoded = write_salt_v2_package(&package).expect("encode mixed package");

        let logical_coefficients = package
            .tensors()
            .iter()
            .map(SaltV2Tensor::logical_coefficients)
            .sum::<usize>();
        assert_eq!(logical_coefficients, 2_050);
        assert_eq!(encoded.ledger.maps_bytes, 2);
        assert_eq!(encoded.ledger.allocation_map_bits, 20);
        assert_eq!(encoded.ledger.allocation_map_package_embedded_bits, 0);
        assert_eq!(encoded.ledger.allocation_map_tensor_embedded_bits, 4);
        assert_eq!(encoded.ledger.allocation_map_embedded_bits, 4);
        assert!(
            encoded.ledger.maps_bytes as f64 * 8.0 / logical_coefficients as f64 <= 0.007_812_5
        );
        assert_eq!(
            read_salt_v2_package(&encoded.bytes)
                .expect("decode mixed package")
                .package,
            package
        );
    }

    #[test]
    fn package_header_selects_one_codec_for_every_tensor() {
        let package = SaltV2Package::new(
            SaltV2Codec::B3,
            vec![one_tile_tensor("a", 4), one_tile_tensor("b", 7)],
        )
        .expect("two-tensor package");
        let encoded = write_salt_v2_package(&package).expect("encode B3 package");

        assert_eq!(&encoded.bytes[..8], &SALT_V2_PACKAGE_MAGIC);
        assert_eq!(u16::from_le_bytes([encoded.bytes[8], encoded.bytes[9]]), 1);
        assert_eq!(encoded.bytes[10], 2);
        let decoded = read_salt_v2_package(&encoded.bytes).expect("decode B3 package");
        assert_eq!(decoded.package.codec(), SaltV2Codec::B3);
        assert_eq!(decoded.package.tensors().len(), 2);
    }

    #[test]
    fn semantic_construction_rejects_noncanonical_trits_and_scales() {
        assert_eq!(
            SaltV2Plane::new(vec![2], vec![f16::ONE]),
            Err(SaltV2PackageError::NonCanonicalTrit { index: 0, value: 2 })
        );
        assert_eq!(
            SaltV2Plane::new(vec![0], vec![f16::INFINITY]),
            Err(SaltV2PackageError::NonFiniteScale {
                group_index: 0,
                bits: f16::INFINITY.to_bits(),
            })
        );
        assert_eq!(
            SaltV2Plane::new(vec![0], vec![f16::from_bits(0x8000)]),
            Err(SaltV2PackageError::NegativeScale {
                group_index: 0,
                bits: 0x8000,
            })
        );
        assert_eq!(
            SaltV2Plane::new(vec![1], vec![f16::ZERO]),
            Err(SaltV2PackageError::ZeroScaleForNonzeroGroup { group_index: 0 })
        );
        assert_eq!(
            SaltV2Plane::new(vec![0; 129], vec![f16::ZERO]),
            Err(SaltV2PackageError::WrongScaleCount {
                expected: 2,
                got: 1,
            })
        );
        assert!(SaltV2Plane::new(vec![0; 128], vec![f16::ZERO]).is_ok());
    }

    #[test]
    fn semantic_construction_rejects_inconsistent_groups_and_tiles() {
        let short = plane(127, 0);
        let long = plane(128, 1);
        assert_eq!(
            SaltV2Tile::new(vec![short, long]),
            Err(SaltV2PackageError::InconsistentPlaneLength {
                plane_index: 1,
                expected: 127,
                got: 128,
            })
        );
        assert_eq!(
            SaltV2Tile::new(Vec::new()),
            Err(SaltV2PackageError::InvalidPlaneCount { got: 0 })
        );

        let undersized = tile(255, 1, 0);
        assert_eq!(
            SaltV2Tensor::new("bad", vec![256], vec![undersized]),
            Err(SaltV2PackageError::WrongTileLength {
                tile_index: 0,
                expected: 256,
                got: 255,
            })
        );
        assert_eq!(
            SaltV2Tensor::new("bad", vec![u64::MAX, 2], Vec::new()),
            Err(SaltV2PackageError::DimensionProductOverflow)
        );
    }

    #[test]
    fn s34_rejects_logical_groups_that_cannot_be_preserved() {
        let plane = SaltV2Plane::new(vec![0, 0, 1, -1], vec![f16::ONE]).expect("semantic plane");
        let tensor = SaltV2Tensor::new(
            "s34",
            vec![4],
            vec![SaltV2Tile::new(vec![plane]).expect("semantic tile")],
        )
        .expect("semantic tensor");

        assert_eq!(
            SaltV2Package::new(SaltV2Codec::S34, vec![tensor]),
            Err(SaltV2PackageError::S34IncompatibleGroup {
                group_index: 0,
                logical_trits: 4,
                zero_count: 2,
            })
        );
    }

    #[test]
    fn package_construction_and_decoding_reject_duplicate_names() {
        let tensor = one_tile_tensor("duplicate", 4);
        assert_eq!(
            SaltV2Package::new(SaltV2Codec::D2, vec![tensor.clone(), tensor]),
            Err(SaltV2PackageError::DuplicateTensorName(
                "duplicate".to_owned()
            ))
        );

        let package = SaltV2Package::new(SaltV2Codec::D2, vec![one_tile_tensor("duplicate", 4)])
            .expect("single tensor package");
        let encoded = write_salt_v2_package(&package).expect("encode one tensor");
        let raw_end = encoded.bytes.len() - encoded.ledger.padding_bytes as usize;
        let record = encoded.bytes[SALT_V2_PACKAGE_HEADER_BYTES..raw_end].to_vec();
        let mut duplicate = encoded.bytes[..SALT_V2_PACKAGE_HEADER_BYTES].to_vec();
        duplicate[12..16].copy_from_slice(&2u32.to_le_bytes());
        duplicate.extend_from_slice(&record);
        duplicate.extend_from_slice(&record);
        let padding = alignment_padding(duplicate.len());
        duplicate.resize(duplicate.len() + padding, 0);
        let duplicate_len = duplicate.len() as u64;
        duplicate[16..24].copy_from_slice(&duplicate_len.to_le_bytes());

        assert_eq!(
            read_salt_v2_package(&duplicate),
            Err(SaltV2PackageError::DuplicateTensorName(
                "duplicate".to_owned()
            ))
        );
    }

    #[test]
    fn malformed_and_trailing_packages_fail_closed() {
        let package = SaltV2Package::new(SaltV2Codec::D2, vec![one_tile_tensor("tensor", 4)])
            .expect("valid package");
        let encoded = write_salt_v2_package(&package).expect("encode package");

        let mut bad_flags = encoded.bytes.clone();
        bad_flags[11] = 1;
        assert_eq!(
            read_salt_v2_package(&bad_flags),
            Err(SaltV2PackageError::NonZeroFlags(1))
        );

        let mut bad_maps = encoded.bytes.clone();
        let tensor_count_offset = packed_tensor_count_offset();
        let packed_count = 1u32 | (0b10_0000u32 << SALT_V2_TENSOR_COUNT_BITS);
        bad_maps[tensor_count_offset..tensor_count_offset + 4]
            .copy_from_slice(&packed_count.to_le_bytes());
        assert_eq!(
            read_salt_v2_package(&bad_maps),
            Err(SaltV2PackageError::NonCanonicalMapPadding)
        );

        let mut nonnested = encoded.bytes.clone();
        let tile_count_offset = SALT_V2_PACKAGE_HEADER_BYTES + 16;
        let packed_tile_count = 1u64 | (0b10u64 << SALT_V2_TILE_COUNT_BITS);
        nonnested[tile_count_offset..tile_count_offset + 8]
            .copy_from_slice(&packed_tile_count.to_le_bytes());
        assert_eq!(
            read_salt_v2_package(&nonnested),
            Err(SaltV2PackageError::NonNestedPlaneMap { tile_index: 0 })
        );

        let mut invalid_d2 = encoded.bytes.clone();
        let payload_offset = tensor_sections_offset("tensor", 1);
        invalid_d2[payload_offset] |= 0b11;
        assert!(matches!(
            read_salt_v2_package(&invalid_d2),
            Err(SaltV2PackageError::Codec(
                SaltV2CodecError::InvalidD2Code { .. }
            ))
        ));

        let mut trailing = encoded.bytes;
        let declared = trailing.len() as u64;
        trailing.push(0);
        assert_eq!(
            read_salt_v2_package(&trailing),
            Err(SaltV2PackageError::WrongTotalLength {
                declared,
                actual: trailing.len(),
            })
        );
    }

    #[test]
    fn declared_section_lengths_are_derived_not_trusted() {
        let package = ragged_package(SaltV2Codec::B3);
        let encoded = write_salt_v2_package(&package).expect("encode package");
        let mut malformed = encoded.bytes;
        let payload_len_offset = SALT_V2_PACKAGE_HEADER_BYTES + 24;
        let declared = u64::from_le_bytes(
            malformed[payload_len_offset..payload_len_offset + 8]
                .try_into()
                .expect("eight payload-length bytes"),
        );
        malformed[payload_len_offset..payload_len_offset + 8]
            .copy_from_slice(&(declared + 1).to_le_bytes());

        assert_eq!(
            read_salt_v2_package(&malformed),
            Err(SaltV2PackageError::DeclaredFieldMismatch {
                field: "payload bytes",
                declared: declared + 1,
                expected: declared,
            })
        );
    }

    #[test]
    fn malformed_embedded_map_bits_and_tile_offsets_fail_closed() {
        let tile_count = 33usize;
        let tensor = SaltV2Tensor::new(
            "tailmaps",
            vec![(tile_count * SALT_V2_ALLOCATION_TILE_SIZE) as u64],
            (0..tile_count)
                .map(|index| tile(SALT_V2_ALLOCATION_TILE_SIZE, 1, index))
                .collect(),
        )
        .expect("valid map-tail tensor");
        let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).expect("valid package");
        let encoded = write_salt_v2_package(&package).expect("encode map-tail package");
        assert_eq!(encoded.ledger.maps_bytes, 8);
        assert_eq!(encoded.ledger.allocation_map_embedded_bits, 2);

        let mut bad_embedded_padding = encoded.bytes.clone();
        let tensor_count_offset = packed_tensor_count_offset();
        let packed_count = u32::from_le_bytes(
            bad_embedded_padding[tensor_count_offset..tensor_count_offset + 4]
                .try_into()
                .expect("packed tensor-count word"),
        );
        bad_embedded_padding[tensor_count_offset..tensor_count_offset + 4]
            .copy_from_slice(&(packed_count | (1u32 << 31)).to_le_bytes());
        assert_eq!(
            read_salt_v2_package(&bad_embedded_padding),
            Err(SaltV2PackageError::NonCanonicalMapPadding)
        );

        let tile_count_offset = SALT_V2_PACKAGE_HEADER_BYTES + 16;
        let mut illegal_ragged_bits = encoded.bytes.clone();
        illegal_ragged_bits[tile_count_offset..tile_count_offset + 8]
            .copy_from_slice(&(33u64 | (1u64 << SALT_V2_TILE_COUNT_BITS)).to_le_bytes());
        assert_eq!(
            read_salt_v2_package(&illegal_ragged_bits),
            Err(SaltV2PackageError::NonCanonicalMapPadding)
        );

        let mut shifted_offset = encoded.bytes;
        shifted_offset[tile_count_offset..tile_count_offset + 8]
            .copy_from_slice(&34u64.to_le_bytes());
        assert_eq!(
            read_salt_v2_package(&shifted_offset),
            Err(SaltV2PackageError::DeclaredFieldMismatch {
                field: "allocation tile count",
                declared: 34,
                expected: 33,
            })
        );
    }

    #[test]
    fn transform_identity_round_trips_and_malformed_metadata_fails_closed() {
        let transform = SaltV2Transform::SignedRht {
            seed: 0x1234_5678_9abc_def0,
            domain: 0x5341_4c54_5f52_4854,
        };
        let tensor =
            SaltV2Tensor::new_with_transform("rotated", vec![17], transform, vec![tile(17, 2, 0)])
                .expect("valid transformed tensor");
        let package = SaltV2Package::new(SaltV2Codec::B3, vec![tensor]).expect("valid package");
        let encoded = write_salt_v2_package(&package).expect("encode transformed package");
        let decoded = read_salt_v2_package(&encoded.bytes).expect("decode transformed package");

        assert_eq!(decoded.package.tensors()[0].transform(), transform);
        assert_eq!(decoded.package, package);
        assert_eq!(encoded.ledger.transform_bytes, 24);
        assert_eq!(encoded.ledger.headers_bytes, 24 + 40 + 7 + 8);
        assert_eq!(
            decoded.ledger.transform_bytes,
            encoded.ledger.transform_bytes
        );

        let transform_tag_offset = SALT_V2_PACKAGE_HEADER_BYTES + 40;
        let mut bad_tag = encoded.bytes.clone();
        bad_tag[transform_tag_offset] = 9;
        assert_eq!(
            read_salt_v2_package(&bad_tag),
            Err(SaltV2PackageError::UnsupportedTransformTag(9))
        );

        let none = SaltV2Package::new(SaltV2Codec::D2, vec![one_tile_tensor("identity", 4)])
            .expect("identity package");
        let identity = write_salt_v2_package(&none).expect("encode identity package");
        let mut noncanonical_none = identity.bytes;
        let transform_seed_offset = SALT_V2_PACKAGE_HEADER_BYTES + 48;
        noncanonical_none[transform_seed_offset] = 1;
        assert_eq!(
            read_salt_v2_package(&noncanonical_none),
            Err(SaltV2PackageError::NonCanonicalTransformMetadata)
        );
    }

    #[test]
    fn compact_prefix_slices_exact_near_lossless_planes_and_scales() {
        let transform = SaltV2Transform::SignedRht {
            seed: 77,
            domain: 19,
        };
        let near_tensor = SaltV2Tensor::new_with_transform(
            "near",
            vec![599],
            transform,
            vec![tile(256, 3, 0), tile(256, 3, 3), tile(87, 3, 6)],
        )
        .expect("near-lossless tensor");
        let near = SaltV2Package::new(SaltV2Codec::D2, vec![near_tensor]).expect("near package");
        let requested = vec![vec![1usize, 2, 3]];
        let compact = near
            .derive_prefix(&requested)
            .expect("derive semantic compact prefix");

        assert_eq!(compact.codec(), near.codec());
        assert_eq!(compact.tensors()[0].transform(), transform);
        for (tile_index, &plane_count) in requested[0].iter().enumerate() {
            assert_eq!(
                compact.tensors()[0].tiles()[tile_index].planes(),
                &near.tensors()[0].tiles()[tile_index].planes()[..plane_count]
            );
        }
        let encoded = write_salt_v2_package(&compact).expect("encode compact prefix");
        assert_eq!(
            read_salt_v2_package(&encoded.bytes)
                .expect("decode compact prefix")
                .package,
            compact
        );
    }

    #[test]
    fn compact_prefix_rejects_missing_tiles_and_nonprefix_plane_counts() {
        let near = SaltV2Package::new(
            SaltV2Codec::D2,
            vec![SaltV2Tensor::new("near", vec![256], vec![tile(256, 2, 0)]).expect("near tensor")],
        )
        .expect("near package");

        assert_eq!(
            near.derive_prefix(&[]),
            Err(SaltV2PackageError::WrongPrefixTensorCount {
                expected: 1,
                got: 0,
            })
        );
        assert_eq!(
            near.derive_prefix(&[Vec::new()]),
            Err(SaltV2PackageError::WrongPrefixTileCount {
                tensor_index: 0,
                expected: 1,
                got: 0,
            })
        );
        for requested in [0usize, 3] {
            assert_eq!(
                near.derive_prefix(&[vec![requested]]),
                Err(SaltV2PackageError::InvalidPrefixPlaneCount {
                    tensor_index: 0,
                    tile_index: 0,
                    requested,
                    available: 2,
                })
            );
        }
    }

    #[test]
    fn every_truncation_and_appended_byte_fails_closed_without_panicking() {
        let package = SaltV2Package::new(SaltV2Codec::B3, vec![one_tile_tensor("fuzzish", 19)])
            .expect("valid package");
        let encoded = write_salt_v2_package(&package).expect("encode package");

        for end in 0..encoded.bytes.len() {
            assert!(
                read_salt_v2_package(&encoded.bytes[..end]).is_err(),
                "truncated prefix {end} was accepted"
            );
        }
        for appended in u8::MIN..=u8::MAX {
            let mut malformed = encoded.bytes.clone();
            malformed.push(appended);
            assert!(
                read_salt_v2_package(&malformed).is_err(),
                "appended byte {appended:#04x} was accepted"
            );
        }

        let mut bad_padding = encoded.bytes;
        assert!(encoded.ledger.padding_bytes > 0);
        *bad_padding.last_mut().expect("nonempty package") = 1;
        assert_eq!(
            read_salt_v2_package(&bad_padding),
            Err(SaltV2PackageError::NonCanonicalFilePadding)
        );
    }
}
