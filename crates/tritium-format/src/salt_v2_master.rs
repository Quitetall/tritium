//! Canonical, bounded-memory work format for fitted SALT V2 tensor masters.

use core::fmt;
use std::io::{self, Write};

use half::f16;

use crate::{
    ModelId,
    salt_v2::{SaltV2Codec, SaltV2CodecError, pack_b3, unpack_b3},
    salt_v2_package::{
        SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_SCALE_GROUP_SIZE, SaltV2PackageError, SaltV2Plane,
        SaltV2Tile, pack_salt_v2_plane,
    },
};

/// Stable identity string for the tensor-master payload schema.
pub const SALT_V2_MASTER_TENSOR_SCHEMA: &[u8] = b"tritium salt v2 tensor master radix3 v1";

const METADATA_MAGIC: [u8; 8] = *b"TSV2MTR\0";
const METADATA_VERSION: u16 = 1;
const METADATA_CHECKSUM_BYTES: usize = 32;
const METADATA_CHECKSUM_CONTEXT: &str = "tritium salt v2 tensor master metadata v1";
const TENSOR_MASTER_ID_CONTEXT: &str = "tritium salt v2 tensor master identity v1";
const MAX_NAME_BYTES: usize = 64 * 1024;
const MAX_RANK: usize = 32;
const MAX_METADATA_BYTES: usize = 128 * 1024;

/// Fitting constraint carried by a codec-independent ordered master.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaltV2FitConstraint {
    /// Unstructured ternary planes deployable as D2 or B3.
    Dense,
    /// One-zero-per-four planes deployable as D2, B3, or S34.
    S34,
}

/// Recovery lineage represented by one tensor master.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaltV2MasterTrack {
    /// Post-training quantization without a parent master.
    Ptq,
    /// Fixed-trit scale-only recovery from a parent master.
    ScaleOnly,
    /// Smooth-to-hard PV/KL recovery from a parent master.
    PvKl,
}

/// Rate-free evidence and lineage bound to one tensor master.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2MasterEvidence {
    /// Digest of every fitting choice except rate and deployment codec.
    pub recipe_id: [u8; 32],
    /// Identity of the solver implementation that produced the hard planes.
    pub solver_id: [u8; 32],
    /// Canonical calibration activation-cache digest.
    pub activation_digest: [u8; 32],
    /// Exact curvature artifact digest for this tensor.
    pub curvature_digest: [u8; 32],
    /// Optional second-order feedback receipt digest.
    pub feedback_digest: Option<[u8; 32]>,
    /// PTQ or refined recovery lineage.
    pub track: SaltV2MasterTrack,
    /// Required parent for refined tracks and absent for PTQ.
    pub parent_master_id: Option<[u8; 32]>,
}

/// Fixed semantic geometry for one tensor master.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2MasterGeometry {
    /// Dense or structured ternary fitting constraint.
    pub constraint: SaltV2FitConstraint,
    /// Number of ordered planes fitted once and stored exactly once.
    pub max_planes: u8,
}

/// Validated identity and geometry for one canonical tensor-master stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SaltV2MasterTensorSpec {
    name: String,
    shape: Vec<u64>,
    logical_coefficients: usize,
    source_model_id: ModelId,
    source_tensor_digest: [u8; 32],
    widened_source_digest: [u8; 32],
    tensor_index: u64,
    evidence: SaltV2MasterEvidence,
    geometry: SaltV2MasterGeometry,
    tile_count: usize,
    payload_bytes: u64,
}

impl SaltV2MasterTensorSpec {
    /// Construct one bounded, rate-free tensor-master specification.
    ///
    /// `source_tensor_digest` identifies admitted source-precision semantics;
    /// `widened_source_digest` separately identifies the exact widened or
    /// transformed source tensor before feedback-adjusted group fitting.
    ///
    /// # Errors
    /// Rejects an invalid name, shape, plane count, lineage, or overflowing size.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        shape: Vec<u64>,
        source_model_id: ModelId,
        source_tensor_digest: [u8; 32],
        widened_source_digest: [u8; 32],
        tensor_index: u64,
        evidence: SaltV2MasterEvidence,
        geometry: SaltV2MasterGeometry,
    ) -> Result<Self, SaltV2MasterError> {
        let name = name.into();
        if name.is_empty() || name.len() > MAX_NAME_BYTES {
            return Err(SaltV2MasterError::InvalidName);
        }
        if shape.is_empty() || shape.len() > MAX_RANK || shape.contains(&0) {
            return Err(SaltV2MasterError::InvalidShape);
        }
        if !(2..=3).contains(&geometry.max_planes) {
            return Err(SaltV2MasterError::InvalidPlaneCount {
                got: geometry.max_planes,
            });
        }
        let valid_lineage = matches!(
            (evidence.track, evidence.parent_master_id),
            (SaltV2MasterTrack::Ptq, None)
                | (
                    SaltV2MasterTrack::ScaleOnly | SaltV2MasterTrack::PvKl,
                    Some(_)
                )
        );
        if !valid_lineage {
            return Err(SaltV2MasterError::InvalidLineage);
        }
        let coefficients = shape
            .iter()
            .try_fold(1u64, |product, dimension| product.checked_mul(*dimension));
        let coefficients = coefficients.ok_or(SaltV2MasterError::LengthOverflow)?;
        let logical_coefficients =
            usize::try_from(coefficients).map_err(|_| SaltV2MasterError::LengthOverflow)?;
        let tile_count = logical_coefficients.div_ceil(SALT_V2_ALLOCATION_TILE_SIZE);
        let payload_bytes = payload_bytes(logical_coefficients, geometry.max_planes)?;
        Ok(Self {
            name,
            shape,
            logical_coefficients,
            source_model_id,
            source_tensor_digest,
            widened_source_digest,
            tensor_index,
            evidence,
            geometry,
            tile_count,
            payload_bytes,
        })
    }

    /// Canonical tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Logical row-major tensor dimensions.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Checked product of the logical dimensions.
    #[must_use]
    pub fn logical_coefficients(&self) -> usize {
        self.logical_coefficients
    }

    /// Semantic source-model identity.
    #[must_use]
    pub const fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Admitted source-precision tensor identity.
    #[must_use]
    pub const fn source_tensor_digest(&self) -> &[u8; 32] {
        &self.source_tensor_digest
    }

    /// Exact widened or transformed source-tensor identity before feedback.
    #[must_use]
    pub const fn widened_source_digest(&self) -> &[u8; 32] {
        &self.widened_source_digest
    }

    /// Stable tensor ordinal in the ordered model master.
    #[must_use]
    pub const fn tensor_index(&self) -> u64 {
        self.tensor_index
    }

    /// Rate-free fitting evidence and recovery lineage.
    #[must_use]
    pub const fn evidence(&self) -> SaltV2MasterEvidence {
        self.evidence
    }

    /// Dense/S34 constraint and fitted Pmax.
    #[must_use]
    pub const fn geometry(&self) -> SaltV2MasterGeometry {
        self.geometry
    }

    /// Shape-derived number of allocation tiles.
    #[must_use]
    pub const fn tile_count(&self) -> usize {
        self.tile_count
    }

    /// Exact canonical payload bytes excluding the record envelope.
    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Encode the standalone canonical metadata bound ahead of payload bytes.
    ///
    /// # Errors
    /// Returns a length error if a canonical count cannot fit its field.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, SaltV2MasterError> {
        let mut output = Vec::new();
        output.extend_from_slice(&METADATA_MAGIC);
        output.extend_from_slice(&METADATA_VERSION.to_le_bytes());
        output.extend_from_slice(&0u16.to_le_bytes());
        output.extend_from_slice(&self.tensor_index.to_le_bytes());
        output.extend_from_slice(self.source_model_id.as_bytes());
        output.extend_from_slice(&self.source_tensor_digest);
        output.extend_from_slice(&self.widened_source_digest);
        output.extend_from_slice(&self.evidence.recipe_id);
        output.extend_from_slice(&self.evidence.solver_id);
        output.extend_from_slice(&self.evidence.activation_digest);
        output.extend_from_slice(&self.evidence.curvature_digest);
        encode_optional_digest(&mut output, self.evidence.feedback_digest);
        encode_optional_digest(&mut output, self.evidence.parent_master_id);
        output.extend_from_slice(&[
            constraint_tag(self.geometry.constraint),
            track_tag(self.evidence.track),
            self.geometry.max_planes,
            0,
        ]);
        let name_len =
            u32::try_from(self.name.len()).map_err(|_| SaltV2MasterError::LengthOverflow)?;
        let rank =
            u32::try_from(self.shape.len()).map_err(|_| SaltV2MasterError::LengthOverflow)?;
        output.extend_from_slice(&name_len.to_le_bytes());
        output.extend_from_slice(self.name.as_bytes());
        output.extend_from_slice(&rank.to_le_bytes());
        for dimension in &self.shape {
            output.extend_from_slice(&dimension.to_le_bytes());
        }
        output.extend_from_slice(&(self.logical_coefficients as u64).to_le_bytes());
        output.extend_from_slice(&(self.tile_count as u64).to_le_bytes());
        output.extend_from_slice(&self.payload_bytes.to_le_bytes());
        let mut hasher = blake3::Hasher::new_derive_key(METADATA_CHECKSUM_CONTEXT);
        hasher.update(&output);
        output.extend_from_slice(hasher.finalize().as_bytes());
        if output.len() > MAX_METADATA_BYTES {
            return Err(SaltV2MasterError::LengthOverflow);
        }
        Ok(output)
    }

    /// Decode only checksum-valid, canonical, internally consistent metadata.
    ///
    /// # Errors
    /// Rejects malformed, unsupported, noncanonical, or contradictory bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, SaltV2MasterError> {
        if bytes.len() > MAX_METADATA_BYTES {
            return Err(SaltV2MasterError::MalformedMetadata("too large"));
        }
        if bytes.len() < METADATA_CHECKSUM_BYTES {
            return Err(SaltV2MasterError::MalformedMetadata("truncated checksum"));
        }
        let checksum_offset = bytes.len() - METADATA_CHECKSUM_BYTES;
        let (payload, checksum) = bytes.split_at(checksum_offset);
        let mut hasher = blake3::Hasher::new_derive_key(METADATA_CHECKSUM_CONTEXT);
        hasher.update(payload);
        if hasher.finalize().as_bytes() != checksum {
            return Err(SaltV2MasterError::MalformedMetadata("checksum"));
        }
        let mut cursor = Cursor::new(payload);
        if cursor.take(METADATA_MAGIC.len())? != METADATA_MAGIC {
            return Err(SaltV2MasterError::MalformedMetadata("magic"));
        }
        if cursor.u16()? != METADATA_VERSION {
            return Err(SaltV2MasterError::MalformedMetadata("version"));
        }
        if cursor.u16()? != 0 {
            return Err(SaltV2MasterError::MalformedMetadata("flags"));
        }
        let tensor_index = cursor.u64()?;
        let source_model_id = ModelId::from_digest(cursor.digest()?);
        let source_tensor_digest = cursor.digest()?;
        let widened_source_digest = cursor.digest()?;
        let recipe_id = cursor.digest()?;
        let solver_id = cursor.digest()?;
        let activation_digest = cursor.digest()?;
        let curvature_digest = cursor.digest()?;
        let feedback_digest = cursor.optional_digest()?;
        let parent_master_id = cursor.optional_digest()?;
        let constraint = constraint_from_tag(cursor.u8()?)?;
        let track = track_from_tag(cursor.u8()?)?;
        let max_planes = cursor.u8()?;
        if cursor.u8()? != 0 {
            return Err(SaltV2MasterError::MalformedMetadata("reserved byte"));
        }
        let name_len = cursor.u32()? as usize;
        if name_len == 0 || name_len > MAX_NAME_BYTES {
            return Err(SaltV2MasterError::InvalidName);
        }
        let name = std::str::from_utf8(cursor.take(name_len)?)
            .map_err(|_| SaltV2MasterError::InvalidName)?
            .to_owned();
        let rank = cursor.u32()? as usize;
        if rank == 0 || rank > MAX_RANK {
            return Err(SaltV2MasterError::InvalidShape);
        }
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(rank)
            .map_err(|_| SaltV2MasterError::AllocationFailed)?;
        for _ in 0..rank {
            shape.push(cursor.u64()?);
        }
        let declared_coefficients = cursor.u64()?;
        let declared_tiles = cursor.u64()?;
        let declared_payload_bytes = cursor.u64()?;
        if cursor.remaining() != 0 {
            return Err(SaltV2MasterError::MalformedMetadata("trailing bytes"));
        }
        let spec = Self::new(
            name,
            shape,
            source_model_id,
            source_tensor_digest,
            widened_source_digest,
            tensor_index,
            SaltV2MasterEvidence {
                recipe_id,
                solver_id,
                activation_digest,
                curvature_digest,
                feedback_digest,
                track,
                parent_master_id,
            },
            SaltV2MasterGeometry {
                constraint,
                max_planes,
            },
        )?;
        if declared_coefficients != spec.logical_coefficients as u64
            || declared_tiles != spec.tile_count as u64
            || declared_payload_bytes != spec.payload_bytes
            || spec.canonical_bytes()? != bytes
        {
            return Err(SaltV2MasterError::MalformedMetadata("derived fields"));
        }
        Ok(spec)
    }

    fn tile_logical_len(&self, tile_index: usize) -> Result<usize, SaltV2MasterError> {
        if tile_index >= self.tile_count {
            return Err(SaltV2MasterError::TooManyTiles);
        }
        let consumed = tile_index
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
            .ok_or(SaltV2MasterError::LengthOverflow)?;
        Ok((self.logical_coefficients - consumed).min(SALT_V2_ALLOCATION_TILE_SIZE))
    }

    fn tile_frame_bytes(&self, tile_index: usize) -> Result<usize, SaltV2MasterError> {
        tile_frame_bytes(self.tile_logical_len(tile_index)?, self.geometry.max_planes)
    }
}

/// One validated prefix-loss point from the same ordered Pmax master.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SaltV2PrefixLoss {
    hessian: f64,
    frobenius: f64,
}

impl SaltV2PrefixLoss {
    /// Construct one finite, non-negative loss pair.
    ///
    /// # Errors
    /// Rejects NaN, infinity, negative values, and negative zero.
    pub fn new(hessian: f64, frobenius: f64) -> Result<Self, SaltV2MasterError> {
        if !hessian.is_finite()
            || hessian < 0.0
            || (hessian == 0.0 && hessian.is_sign_negative())
            || !frobenius.is_finite()
            || frobenius < 0.0
            || (frobenius == 0.0 && frobenius.is_sign_negative())
        {
            return Err(SaltV2MasterError::InvalidPrefixLoss);
        }
        Ok(Self { hessian, frobenius })
    }

    /// Curvature-weighted reconstruction loss.
    #[must_use]
    pub const fn hessian(self) -> f64 {
        self.hessian
    }

    /// Ordinary squared reconstruction loss.
    #[must_use]
    pub const fn frobenius(self) -> f64 {
        self.frobenius
    }
}

/// One decoded allocation tile: its complete prefix curve and Pmax planes.
#[derive(Clone, Debug, PartialEq)]
pub struct SaltV2MasterTile {
    admissible_planes: u8,
    losses: Vec<SaltV2PrefixLoss>,
    tile: SaltV2Tile,
}

impl SaltV2MasterTile {
    /// Largest prefix admitted to the allocator; Pmax planes remain stored.
    #[must_use]
    pub const fn admissible_planes(&self) -> u8 {
        self.admissible_planes
    }

    /// Ordered P1 through Pmax loss points.
    #[must_use]
    pub fn losses(&self) -> &[SaltV2PrefixLoss] {
        &self.losses
    }

    /// Ordered Pmax planes stored once; every lower artifact slices this prefix.
    #[must_use]
    pub fn planes(&self) -> &[SaltV2Plane] {
        self.tile.planes()
    }
}

/// Content receipt for one complete canonical tensor-master stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2MasterTensorReceipt {
    tensor_master_id: [u8; 32],
    payload_bytes: u64,
    tile_count: u64,
}

impl SaltV2MasterTensorReceipt {
    /// Portable identity over canonical specification and payload bytes.
    #[must_use]
    pub const fn tensor_master_id(self) -> [u8; 32] {
        self.tensor_master_id
    }

    /// Exact canonical payload length.
    #[must_use]
    pub const fn payload_bytes(self) -> u64 {
        self.payload_bytes
    }

    /// Exact number of allocation tiles.
    #[must_use]
    pub const fn tile_count(self) -> u64 {
        self.tile_count
    }
}

/// Streaming encoder for one canonical tensor-master payload.
#[derive(Debug)]
pub struct SaltV2MasterTensorEncoder<'a, W> {
    spec: &'a SaltV2MasterTensorSpec,
    sink: W,
    hasher: blake3::Hasher,
    tile_index: usize,
    written: u64,
    failed: bool,
}

impl<'a, W: Write> SaltV2MasterTensorEncoder<'a, W> {
    /// Start a stream bound to `spec` without allocating tensor-sized storage.
    ///
    /// # Errors
    /// Returns a metadata or allocation error while initializing the identity.
    pub fn new(spec: &'a SaltV2MasterTensorSpec, sink: W) -> Result<Self, SaltV2MasterError> {
        let mut hasher = blake3::Hasher::new_derive_key(TENSOR_MASTER_ID_CONTEXT);
        hasher.update(SALT_V2_MASTER_TENSOR_SCHEMA);
        let metadata = spec.canonical_bytes()?;
        hasher.update(&(metadata.len() as u64).to_le_bytes());
        hasher.update(&metadata);
        Ok(Self {
            spec,
            sink,
            hasher,
            tile_index: 0,
            written: 0,
            failed: false,
        })
    }

    /// Write the next tile's complete prefix curve and full Pmax planes.
    ///
    /// The radix-3 work transport is canonical but is not a deployment-codec
    /// choice. D2/B3 packaging later repacks these exact semantic prefixes.
    ///
    /// # Errors
    /// Rejects a wrong tile order/shape, invalid curve, incompatible S34 plane,
    /// allocation failure, arithmetic overflow, or sink failure.
    pub fn write_tile(
        &mut self,
        admissible_planes: u8,
        losses: &[SaltV2PrefixLoss],
        planes: &[SaltV2Plane],
    ) -> Result<(), SaltV2MasterError> {
        if self.failed {
            return Err(SaltV2MasterError::Poisoned);
        }
        let result = self.encode_tile(admissible_planes, losses, planes);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn encode_tile(
        &mut self,
        admissible_planes: u8,
        losses: &[SaltV2PrefixLoss],
        planes: &[SaltV2Plane],
    ) -> Result<(), SaltV2MasterError> {
        let logical_len = self.spec.tile_logical_len(self.tile_index)?;
        let plane_count = usize::from(self.spec.geometry.max_planes);
        if losses.len() != plane_count || planes.len() != plane_count {
            return Err(SaltV2MasterError::WrongPlaneCount {
                expected: self.spec.geometry.max_planes,
                got: u8::try_from(planes.len()).unwrap_or(u8::MAX),
            });
        }
        if admissible_planes == 0 || usize::from(admissible_planes) > plane_count {
            return Err(SaltV2MasterError::InvalidAdmissiblePrefix {
                got: admissible_planes,
            });
        }
        validate_prefix_curve(losses, admissible_planes)?;
        let expected = self.spec.tile_frame_bytes(self.tile_index)?;
        let mut frame = Vec::new();
        frame
            .try_reserve_exact(expected)
            .map_err(|_| SaltV2MasterError::AllocationFailed)?;
        frame.push(admissible_planes);
        for loss in losses {
            frame.extend_from_slice(&loss.hessian.to_bits().to_le_bytes());
            frame.extend_from_slice(&loss.frobenius.to_bits().to_le_bytes());
        }
        for plane in planes {
            if plane.trits().len() != logical_len {
                return Err(SaltV2MasterError::WrongTileLength {
                    tile_index: self.tile_index,
                    expected: logical_len,
                    got: plane.trits().len(),
                });
            }
            if self.spec.geometry.constraint == SaltV2FitConstraint::S34 {
                pack_salt_v2_plane(SaltV2Codec::S34, plane.trits())?;
            }
            let packed = pack_b3(plane.trits())?;
            frame.extend_from_slice(&packed);
            for scale in plane.scales() {
                frame.extend_from_slice(&scale.to_bits().to_le_bytes());
            }
        }
        if frame.len() != expected {
            return Err(SaltV2MasterError::LengthOverflow);
        }
        self.sink
            .write_all(&frame)
            .map_err(|error| master_io("write tensor-master tile", error))?;
        self.hasher.update(&frame);
        self.written = self
            .written
            .checked_add(frame.len() as u64)
            .ok_or(SaltV2MasterError::LengthOverflow)?;
        self.tile_index += 1;
        Ok(())
    }

    /// Finish only after every shape-derived tile was written exactly once.
    ///
    /// # Errors
    /// Rejects a poisoned, short, or otherwise length-inconsistent stream.
    pub fn finish(self) -> Result<SaltV2MasterTensorReceipt, SaltV2MasterError> {
        if self.failed {
            return Err(SaltV2MasterError::Poisoned);
        }
        if self.tile_index != self.spec.tile_count || self.written != self.spec.payload_bytes {
            return Err(SaltV2MasterError::PayloadLengthMismatch {
                expected: self.spec.payload_bytes,
                actual: self.written,
            });
        }
        Ok(SaltV2MasterTensorReceipt {
            tensor_master_id: *self.hasher.finalize().as_bytes(),
            payload_bytes: self.written,
            tile_count: self.tile_index as u64,
        })
    }
}

/// Bounded streaming decoder and canonical verifier for one tensor master.
#[derive(Debug)]
pub struct SaltV2MasterTensorDecoder<'a> {
    spec: &'a SaltV2MasterTensorSpec,
    hasher: blake3::Hasher,
    tile_index: usize,
    buffer: Vec<u8>,
    received: u64,
    failed: bool,
}

impl<'a> SaltV2MasterTensorDecoder<'a> {
    /// Start a decoder whose staging memory is bounded by one 256-weight tile.
    ///
    /// # Errors
    /// Returns a metadata or allocation error while initializing validation.
    pub fn new(spec: &'a SaltV2MasterTensorSpec) -> Result<Self, SaltV2MasterError> {
        let mut hasher = blake3::Hasher::new_derive_key(TENSOR_MASTER_ID_CONTEXT);
        hasher.update(SALT_V2_MASTER_TENSOR_SCHEMA);
        let metadata = spec.canonical_bytes()?;
        hasher.update(&(metadata.len() as u64).to_le_bytes());
        hasher.update(&metadata);
        let maximum_frame = spec.tile_frame_bytes(0)?;
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(maximum_frame)
            .map_err(|_| SaltV2MasterError::AllocationFailed)?;
        Ok(Self {
            spec,
            hasher,
            tile_index: 0,
            buffer,
            received: 0,
            failed: false,
        })
    }

    /// Consume arbitrary byte chunks and visit each fully verified tile once.
    ///
    /// # Errors
    /// Returns [`SaltV2MasterVisitError::Master`] for malformed or excessive
    /// bytes and [`SaltV2MasterVisitError::Visitor`] without erasing a callback
    /// failure. Any failure poisons this decoder.
    pub fn try_push<E>(
        &mut self,
        bytes: &[u8],
        visit: &mut impl FnMut(SaltV2MasterTile) -> Result<(), E>,
    ) -> Result<(), SaltV2MasterVisitError<E>> {
        if self.failed {
            return Err(SaltV2MasterVisitError::Master(SaltV2MasterError::Poisoned));
        }
        let result = self.decode_bytes(bytes, visit);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn decode_bytes<E>(
        &mut self,
        mut bytes: &[u8],
        visit: &mut impl FnMut(SaltV2MasterTile) -> Result<(), E>,
    ) -> Result<(), SaltV2MasterVisitError<E>> {
        let length = u64::try_from(bytes.len())
            .map_err(|_| SaltV2MasterVisitError::Master(SaltV2MasterError::LengthOverflow))?;
        let next = self
            .received
            .checked_add(length)
            .ok_or_else(|| SaltV2MasterVisitError::Master(SaltV2MasterError::LengthOverflow))?;
        if next > self.spec.payload_bytes {
            return Err(SaltV2MasterVisitError::Master(
                SaltV2MasterError::PayloadLengthMismatch {
                    expected: self.spec.payload_bytes,
                    actual: next,
                },
            ));
        }
        self.hasher.update(bytes);
        self.received = next;
        while !bytes.is_empty() {
            let frame_bytes = self
                .spec
                .tile_frame_bytes(self.tile_index)
                .map_err(SaltV2MasterVisitError::Master)?;
            let needed = frame_bytes - self.buffer.len();
            let count = needed.min(bytes.len());
            self.buffer.extend_from_slice(&bytes[..count]);
            bytes = &bytes[count..];
            if self.buffer.len() == frame_bytes {
                let tile = decode_tile(self.spec, self.tile_index, &self.buffer)
                    .map_err(SaltV2MasterVisitError::Master)?;
                visit(tile).map_err(SaltV2MasterVisitError::Visitor)?;
                self.buffer.clear();
                self.tile_index += 1;
            }
        }
        Ok(())
    }

    /// Finish only after the exact declared payload was consumed and verified.
    ///
    /// # Errors
    /// Rejects a poisoned, truncated, or otherwise incomplete stream.
    pub fn finish(self) -> Result<SaltV2MasterTensorReceipt, SaltV2MasterError> {
        if self.failed {
            return Err(SaltV2MasterError::Poisoned);
        }
        if self.received != self.spec.payload_bytes
            || self.tile_index != self.spec.tile_count
            || !self.buffer.is_empty()
        {
            return Err(SaltV2MasterError::PayloadLengthMismatch {
                expected: self.spec.payload_bytes,
                actual: self.received,
            });
        }
        Ok(SaltV2MasterTensorReceipt {
            tensor_master_id: *self.hasher.finalize().as_bytes(),
            payload_bytes: self.received,
            tile_count: self.tile_index as u64,
        })
    }
}

/// Typed callback failure while streaming a canonical tensor master.
#[derive(Debug)]
pub enum SaltV2MasterVisitError<E> {
    /// Canonical master framing or semantics failed.
    Master(SaltV2MasterError),
    /// The caller's tile visitor failed.
    Visitor(E),
}

impl<E: fmt::Display> fmt::Display for SaltV2MasterVisitError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Master(error) => write!(formatter, "tensor master failed: {error}"),
            Self::Visitor(error) => write!(formatter, "tensor-master visitor failed: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for SaltV2MasterVisitError<E> {}

/// Failure to describe, encode, or canonically reopen a tensor master.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2MasterError {
    /// Tensor name was empty or exceeded the schema bound.
    InvalidName,
    /// Tensor shape was empty, zero, overranked, or overflowing.
    InvalidShape,
    /// Pmax was outside the supported two-to-three plane range.
    InvalidPlaneCount {
        /// Rejected count.
        got: u8,
    },
    /// PTQ had a parent or a refined track omitted its parent.
    InvalidLineage,
    /// A prefix loss was negative (including signed zero), NaN, or infinite.
    InvalidPrefixLoss,
    /// The admitted prefix count was zero or exceeded Pmax.
    InvalidAdmissiblePrefix {
        /// Rejected prefix count.
        got: u8,
    },
    /// A later curvature loss exceeded its predecessor beyond fitter tolerance.
    NonMonotonePrefixLoss,
    /// A tile supplied a wrong number of ordered planes.
    WrongPlaneCount {
        /// Required Pmax.
        expected: u8,
        /// Supplied plane count.
        got: u8,
    },
    /// A plane contradicted shape-derived tile geometry.
    WrongTileLength {
        /// Tensor-local tile ordinal.
        tile_index: usize,
        /// Shape-derived logical length.
        expected: usize,
        /// Supplied logical length.
        got: usize,
    },
    /// More tile data was supplied than the shape permits.
    TooManyTiles,
    /// Payload length was short, excessive, or contradictory.
    PayloadLengthMismatch {
        /// Exact schema-derived payload bytes.
        expected: u64,
        /// Observed payload bytes.
        actual: u64,
    },
    /// Checked length arithmetic overflowed.
    LengthOverflow,
    /// A bounded allocation failed.
    AllocationFailed,
    /// Standalone metadata was malformed or noncanonical.
    MalformedMetadata(&'static str),
    /// A prior stream error makes reuse unsafe.
    Poisoned,
    /// Radix or structured codec validation failed.
    Codec(SaltV2CodecError),
    /// Plane/tile semantic validation failed.
    Package(SaltV2PackageError),
    /// Bounded output I/O failed.
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Portable I/O category.
        kind: io::ErrorKind,
    },
}

impl From<SaltV2CodecError> for SaltV2MasterError {
    fn from(value: SaltV2CodecError) -> Self {
        Self::Codec(value)
    }
}

impl From<SaltV2PackageError> for SaltV2MasterError {
    fn from(value: SaltV2PackageError) -> Self {
        Self::Package(value)
    }
}

impl fmt::Display for SaltV2MasterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName => formatter.write_str("invalid tensor-master name"),
            Self::InvalidShape => formatter.write_str("invalid tensor-master shape"),
            Self::InvalidPlaneCount { got } => write!(formatter, "invalid Pmax {got}"),
            Self::InvalidLineage => formatter.write_str("invalid tensor-master refinement lineage"),
            Self::InvalidPrefixLoss => formatter.write_str("invalid tensor-master prefix loss"),
            Self::InvalidAdmissiblePrefix { got } => {
                write!(formatter, "invalid admissible tensor-master prefix {got}")
            }
            Self::NonMonotonePrefixLoss => {
                formatter.write_str("non-monotone tensor-master curvature loss")
            }
            Self::WrongPlaneCount { expected, got } => {
                write!(
                    formatter,
                    "wrong tensor-master plane count: expected {expected}, got {got}"
                )
            }
            Self::WrongTileLength {
                tile_index,
                expected,
                got,
            } => write!(
                formatter,
                "tensor-master tile {tile_index} length {got}, expected {expected}"
            ),
            Self::TooManyTiles => formatter.write_str("too many tensor-master tiles"),
            Self::PayloadLengthMismatch { expected, actual } => write!(
                formatter,
                "tensor-master payload length {actual}, expected {expected}"
            ),
            Self::LengthOverflow => formatter.write_str("tensor-master length overflow"),
            Self::AllocationFailed => formatter.write_str("tensor-master allocation failed"),
            Self::MalformedMetadata(field) => {
                write!(formatter, "malformed tensor-master metadata: {field}")
            }
            Self::Poisoned => formatter.write_str("tensor-master stream is poisoned"),
            Self::Codec(error) => write!(formatter, "tensor-master codec failed: {error}"),
            Self::Package(error) => write!(formatter, "tensor-master plane failed: {error}"),
            Self::Io { operation, kind } => {
                write!(formatter, "tensor-master {operation} failed: {kind}")
            }
        }
    }
}

impl std::error::Error for SaltV2MasterError {}

fn payload_bytes(logical_coefficients: usize, max_planes: u8) -> Result<u64, SaltV2MasterError> {
    let tile_count = logical_coefficients.div_ceil(SALT_V2_ALLOCATION_TILE_SIZE);
    let mut total = 0u64;
    for tile_index in 0..tile_count {
        let consumed = tile_index
            .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
            .ok_or(SaltV2MasterError::LengthOverflow)?;
        let logical_len = (logical_coefficients - consumed).min(SALT_V2_ALLOCATION_TILE_SIZE);
        total = total
            .checked_add(tile_frame_bytes(logical_len, max_planes)? as u64)
            .ok_or(SaltV2MasterError::LengthOverflow)?;
    }
    Ok(total)
}

fn tile_frame_bytes(logical_len: usize, max_planes: u8) -> Result<usize, SaltV2MasterError> {
    let planes = usize::from(max_planes);
    let losses = planes
        .checked_mul(16)
        .ok_or(SaltV2MasterError::LengthOverflow)?;
    let trits = SaltV2Codec::B3.ledger(logical_len)?.physical_bytes;
    let scales = logical_len
        .div_ceil(SALT_V2_SCALE_GROUP_SIZE)
        .checked_mul(2)
        .ok_or(SaltV2MasterError::LengthOverflow)?;
    1usize
        .checked_add(losses)
        .ok_or(SaltV2MasterError::LengthOverflow)?
        .checked_add(
            trits
                .checked_add(scales)
                .and_then(|bytes| bytes.checked_mul(planes))
                .ok_or(SaltV2MasterError::LengthOverflow)?,
        )
        .ok_or(SaltV2MasterError::LengthOverflow)
}

fn validate_prefix_curve(
    losses: &[SaltV2PrefixLoss],
    admissible_planes: u8,
) -> Result<(), SaltV2MasterError> {
    for loss in losses {
        SaltV2PrefixLoss::new(loss.hessian, loss.frobenius)?;
    }
    let admissible = usize::from(admissible_planes);
    if admissible == 0 || admissible > losses.len() {
        return Err(SaltV2MasterError::InvalidAdmissiblePrefix {
            got: admissible_planes,
        });
    }
    for pair in losses[..admissible].windows(2) {
        let tolerance = 1e-12f64.max(pair[0].hessian.abs() * 1e-12);
        if pair[1].hessian > pair[0].hessian + tolerance {
            return Err(SaltV2MasterError::NonMonotonePrefixLoss);
        }
    }
    Ok(())
}

fn decode_tile(
    spec: &SaltV2MasterTensorSpec,
    tile_index: usize,
    bytes: &[u8],
) -> Result<SaltV2MasterTile, SaltV2MasterError> {
    let logical_len = spec.tile_logical_len(tile_index)?;
    if bytes.len() != spec.tile_frame_bytes(tile_index)? {
        return Err(SaltV2MasterError::PayloadLengthMismatch {
            expected: spec.tile_frame_bytes(tile_index)? as u64,
            actual: bytes.len() as u64,
        });
    }
    let plane_count = usize::from(spec.geometry.max_planes);
    let mut cursor = Cursor::new(bytes);
    let admissible_planes = cursor.u8()?;
    let mut losses = Vec::new();
    losses
        .try_reserve_exact(plane_count)
        .map_err(|_| SaltV2MasterError::AllocationFailed)?;
    for _ in 0..plane_count {
        let hessian = f64::from_bits(cursor.u64()?);
        let frobenius = f64::from_bits(cursor.u64()?);
        losses.push(SaltV2PrefixLoss::new(hessian, frobenius)?);
    }
    validate_prefix_curve(&losses, admissible_planes)?;
    let packed_len = SaltV2Codec::B3.ledger(logical_len)?.physical_bytes;
    let scale_count = logical_len.div_ceil(SALT_V2_SCALE_GROUP_SIZE);
    let mut planes = Vec::new();
    planes
        .try_reserve_exact(plane_count)
        .map_err(|_| SaltV2MasterError::AllocationFailed)?;
    for _ in 0..plane_count {
        let trits = unpack_b3(cursor.take(packed_len)?, logical_len)?;
        let mut scales = Vec::new();
        scales
            .try_reserve_exact(scale_count)
            .map_err(|_| SaltV2MasterError::AllocationFailed)?;
        for _ in 0..scale_count {
            scales.push(f16::from_bits(cursor.u16()?));
        }
        let raw_trits = trits.into_iter().map(|trit| trit.get()).collect();
        let plane = SaltV2Plane::new(raw_trits, scales)?;
        if spec.geometry.constraint == SaltV2FitConstraint::S34 {
            pack_salt_v2_plane(SaltV2Codec::S34, plane.trits())?;
        }
        planes.push(plane);
    }
    if cursor.remaining() != 0 {
        return Err(SaltV2MasterError::PayloadLengthMismatch {
            expected: bytes.len() as u64,
            actual: (bytes.len() - cursor.remaining()) as u64,
        });
    }
    Ok(SaltV2MasterTile {
        admissible_planes,
        losses,
        tile: SaltV2Tile::new(planes)?,
    })
}

fn encode_optional_digest(output: &mut Vec<u8>, digest: Option<[u8; 32]>) {
    match digest {
        Some(digest) => {
            output.push(1);
            output.extend_from_slice(&digest);
        }
        None => {
            output.push(0);
            output.extend_from_slice(&[0; 32]);
        }
    }
}

const fn constraint_tag(constraint: SaltV2FitConstraint) -> u8 {
    match constraint {
        SaltV2FitConstraint::Dense => 1,
        SaltV2FitConstraint::S34 => 2,
    }
}

fn constraint_from_tag(tag: u8) -> Result<SaltV2FitConstraint, SaltV2MasterError> {
    match tag {
        1 => Ok(SaltV2FitConstraint::Dense),
        2 => Ok(SaltV2FitConstraint::S34),
        _ => Err(SaltV2MasterError::MalformedMetadata("constraint")),
    }
}

const fn track_tag(track: SaltV2MasterTrack) -> u8 {
    match track {
        SaltV2MasterTrack::Ptq => 1,
        SaltV2MasterTrack::ScaleOnly => 2,
        SaltV2MasterTrack::PvKl => 3,
    }
}

fn track_from_tag(tag: u8) -> Result<SaltV2MasterTrack, SaltV2MasterError> {
    match tag {
        1 => Ok(SaltV2MasterTrack::Ptq),
        2 => Ok(SaltV2MasterTrack::ScaleOnly),
        3 => Ok(SaltV2MasterTrack::PvKl),
        _ => Err(SaltV2MasterError::MalformedMetadata("track")),
    }
}

fn master_io(operation: &'static str, error: io::Error) -> SaltV2MasterError {
    SaltV2MasterError::Io {
        operation,
        kind: error.kind(),
    }
}

#[derive(Debug)]
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SaltV2MasterError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(SaltV2MasterError::LengthOverflow)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(SaltV2MasterError::MalformedMetadata("truncated bytes"))?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, SaltV2MasterError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SaltV2MasterError> {
        let mut bytes = [0; 2];
        bytes.copy_from_slice(self.take(2)?);
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32, SaltV2MasterError> {
        let mut bytes = [0; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, SaltV2MasterError> {
        let mut bytes = [0; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn digest(&mut self) -> Result<[u8; 32], SaltV2MasterError> {
        let mut digest = [0; 32];
        digest.copy_from_slice(self.take(32)?);
        Ok(digest)
    }

    fn optional_digest(&mut self) -> Result<Option<[u8; 32]>, SaltV2MasterError> {
        let present = self.u8()?;
        let digest = self.digest()?;
        match present {
            0 if digest == [0; 32] => Ok(None),
            1 => Ok(Some(digest)),
            _ => Err(SaltV2MasterError::MalformedMetadata("optional digest")),
        }
    }
}

#[cfg(test)]
mod tests {
    use half::f16;
    use tritium_core::Trit;

    use super::*;
    use crate::{ModelId, salt_v2_package::SaltV2Plane};

    fn plane(len: usize, phase: usize, scale: f32) -> SaltV2Plane {
        let trits = (0..len)
            .map(|index| match (index + phase) % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            })
            .collect();
        SaltV2Plane::new(trits, vec![f16::from_f32(scale); len.div_ceil(128)]).unwrap()
    }

    fn s34_plane(len: usize, phase: usize, scale: f32) -> SaltV2Plane {
        let trits = (0..len)
            .map(|index| match (index + phase) % 4 {
                0 => 0,
                1 | 2 => 1,
                _ => -1,
            })
            .collect();
        SaltV2Plane::new(trits, vec![f16::from_f32(scale); len.div_ceil(128)]).unwrap()
    }

    fn spec() -> SaltV2MasterTensorSpec {
        SaltV2MasterTensorSpec::new(
            "model.layers.0.mlp.down_proj.weight",
            vec![2, 129],
            ModelId::from_digest([1; 32]),
            [2; 32],
            [3; 32],
            7,
            SaltV2MasterEvidence {
                recipe_id: [4; 32],
                solver_id: [5; 32],
                activation_digest: [6; 32],
                curvature_digest: [7; 32],
                feedback_digest: Some([8; 32]),
                track: SaltV2MasterTrack::Ptq,
                parent_master_id: None,
            },
            SaltV2MasterGeometry {
                constraint: SaltV2FitConstraint::Dense,
                max_planes: 3,
            },
        )
        .unwrap()
    }

    #[test]
    fn pmax_tensor_stream_round_trips_across_arbitrary_chunks() {
        let spec = spec();
        let mut bytes = Vec::new();
        let mut encoder = SaltV2MasterTensorEncoder::new(&spec, &mut bytes).unwrap();
        encoder
            .write_tile(
                3,
                &[
                    SaltV2PrefixLoss::new(9.0, 8.0).unwrap(),
                    SaltV2PrefixLoss::new(4.0, 3.0).unwrap(),
                    SaltV2PrefixLoss::new(1.0, 0.5).unwrap(),
                ],
                &[
                    plane(256, 0, 0.5),
                    plane(256, 1, 0.25),
                    plane(256, 2, 0.125),
                ],
            )
            .unwrap();
        encoder
            .write_tile(
                3,
                &[
                    SaltV2PrefixLoss::new(3.0, 2.0).unwrap(),
                    SaltV2PrefixLoss::new(2.0, 1.0).unwrap(),
                    SaltV2PrefixLoss::new(1.0, 0.25).unwrap(),
                ],
                &[plane(2, 0, 0.5), plane(2, 1, 0.25), plane(2, 2, 0.125)],
            )
            .unwrap();
        let encoded = encoder.finish().unwrap();

        assert_eq!(encoded.payload_bytes(), bytes.len() as u64);
        assert_eq!(bytes.len() as u64, spec.payload_bytes());
        let duplicated_raw_prefix_bytes = (256 + 256 * 2 + 256 * 3 + 2 + 2 * 2 + 2 * 3) as u64;
        assert!(encoded.payload_bytes() < duplicated_raw_prefix_bytes);

        let mut decoder = SaltV2MasterTensorDecoder::new(&spec).unwrap();
        let mut decoded = Vec::new();
        for chunk in bytes.chunks(7) {
            decoder
                .try_push(chunk, &mut |tile| {
                    decoded.push(tile);
                    Ok::<_, core::convert::Infallible>(())
                })
                .unwrap();
        }
        let reopened = decoder.finish().unwrap();
        assert_eq!(reopened, encoded);
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].planes()[0].trits()[0], Trit::NEG);
        assert_eq!(decoded[0].planes()[..2].len(), 2);
        assert_eq!(decoded[0].planes()[..3].len(), 3);

        let mut corrupt = bytes;
        let last = corrupt.len() - 1;
        corrupt[last] = 255;
        let mut decoder = SaltV2MasterTensorDecoder::new(&spec).unwrap();
        assert!(corrupt.chunks(11).any(|chunk| {
            decoder
                .try_push(chunk, &mut |_| Ok::<_, core::convert::Infallible>(()))
                .is_err()
        }));
    }

    #[test]
    fn metadata_lineage_and_canonical_bytes_fail_closed() {
        let spec = spec();
        let bytes = spec.canonical_bytes().unwrap();
        assert_eq!(
            SaltV2MasterTensorSpec::from_canonical_bytes(&bytes).unwrap(),
            spec
        );
        for end in 0..bytes.len() {
            assert!(SaltV2MasterTensorSpec::from_canonical_bytes(&bytes[..end]).is_err());
        }
        let mut corrupt = bytes;
        corrupt[16] ^= 1;
        assert!(SaltV2MasterTensorSpec::from_canonical_bytes(&corrupt).is_err());

        let mut evidence = spec.evidence();
        evidence.track = SaltV2MasterTrack::PvKl;
        assert!(matches!(
            SaltV2MasterTensorSpec::new(
                spec.name(),
                spec.shape().to_vec(),
                spec.source_model_id(),
                *spec.source_tensor_digest(),
                *spec.widened_source_digest(),
                spec.tensor_index(),
                evidence,
                spec.geometry(),
            ),
            Err(SaltV2MasterError::InvalidLineage)
        ));
        assert_eq!(
            SaltV2PrefixLoss::new(-0.0, 0.0),
            Err(SaltV2MasterError::InvalidPrefixLoss)
        );
        assert_eq!(
            SaltV2PrefixLoss::new(0.0, -0.0),
            Err(SaltV2MasterError::InvalidPrefixLoss)
        );
    }

    #[test]
    fn decoder_rejects_noncanonical_radix_and_never_buffers_more_than_one_tile() {
        let spec = spec();
        let mut bytes = Vec::new();
        let mut encoder = SaltV2MasterTensorEncoder::new(&spec, &mut bytes).unwrap();
        let losses = [
            SaltV2PrefixLoss::new(3.0, 3.0).unwrap(),
            SaltV2PrefixLoss::new(2.0, 2.0).unwrap(),
            SaltV2PrefixLoss::new(1.0, 1.0).unwrap(),
        ];
        encoder
            .write_tile(
                3,
                &losses,
                &[
                    plane(256, 0, 0.5),
                    plane(256, 1, 0.25),
                    plane(256, 2, 0.125),
                ],
            )
            .unwrap();
        encoder
            .write_tile(
                3,
                &losses,
                &[plane(2, 0, 0.5), plane(2, 1, 0.25), plane(2, 2, 0.125)],
            )
            .unwrap();
        encoder.finish().unwrap();

        let first_b3_byte = 1 + usize::from(spec.geometry().max_planes) * 16;
        bytes[first_b3_byte] = 243;
        let mut decoder = SaltV2MasterTensorDecoder::new(&spec).unwrap();
        assert!(matches!(
            decoder.try_push(&bytes, &mut |_| Ok::<_, core::convert::Infallible>(())),
            Err(SaltV2MasterVisitError::Master(SaltV2MasterError::Codec(
                SaltV2CodecError::InvalidB3Code { .. }
            )))
        ));
        assert!(decoder.buffer.capacity() <= spec.tile_frame_bytes(0).unwrap());

        let mut output = Vec::new();
        let mut encoder = SaltV2MasterTensorEncoder::new(&spec, &mut output).unwrap();
        assert!(matches!(
            encoder.write_tile(
                3,
                &[
                    SaltV2PrefixLoss::new(1.0, 1.0).unwrap(),
                    SaltV2PrefixLoss::new(2.0, 0.5).unwrap(),
                    SaltV2PrefixLoss::new(0.5, 0.25).unwrap(),
                ],
                &[
                    plane(256, 0, 0.5),
                    plane(256, 1, 0.25),
                    plane(256, 2, 0.125)
                ],
            ),
            Err(SaltV2MasterError::NonMonotonePrefixLoss)
        ));
    }

    #[test]
    fn s34_master_preserves_a_non_admitted_pmax_suffix() {
        let dense = spec();
        let spec = SaltV2MasterTensorSpec::new(
            dense.name(),
            vec![1, 256],
            dense.source_model_id(),
            *dense.source_tensor_digest(),
            *dense.widened_source_digest(),
            dense.tensor_index(),
            dense.evidence(),
            SaltV2MasterGeometry {
                constraint: SaltV2FitConstraint::S34,
                max_planes: 3,
            },
        )
        .unwrap();
        let losses = [
            SaltV2PrefixLoss::new(3.0, 3.0).unwrap(),
            SaltV2PrefixLoss::new(4.0, 2.0).unwrap(),
            SaltV2PrefixLoss::new(2.0, 1.0).unwrap(),
        ];
        let planes = [
            s34_plane(256, 0, 0.5),
            s34_plane(256, 1, 0.25),
            s34_plane(256, 2, 0.125),
        ];
        let mut bytes = Vec::new();
        let mut encoder = SaltV2MasterTensorEncoder::new(&spec, &mut bytes).unwrap();
        encoder.write_tile(1, &losses, &planes).unwrap();
        encoder.finish().unwrap();

        let mut decoder = SaltV2MasterTensorDecoder::new(&spec).unwrap();
        let mut decoded = Vec::new();
        decoder
            .try_push(&bytes, &mut |tile| {
                decoded.push(tile);
                Ok::<_, core::convert::Infallible>(())
            })
            .unwrap();
        decoder.finish().unwrap();
        assert_eq!(decoded[0].admissible_planes(), 1);
        assert_eq!(decoded[0].losses(), losses);
        assert_eq!(decoded[0].planes(), planes);

        bytes[0] = 0;
        let mut decoder = SaltV2MasterTensorDecoder::new(&spec).unwrap();
        assert!(matches!(
            decoder.try_push(&bytes, &mut |_| Ok::<_, core::convert::Infallible>(())),
            Err(SaltV2MasterVisitError::Master(
                SaltV2MasterError::InvalidAdmissiblePrefix { got: 0 }
            ))
        ));
    }
}
