//! Tile-wise canonical encoding behind the shared strict legacy importer.
use std::io::{Read, Seek, SeekFrom, Write};

use tritium_core::{PerTileIndex, PerTilePlan, PerTileRowSpan, Trit};
use tritium_schema::{AdditiveLayout, BlobId, PlaneCodec, PlaneCounts, TensorStorage};

use super::{
    E, Prepared, RowCursor, SaltTensorImportLimits, decode, import_bound, locate, prepare,
    prove_scales, reserved, view_error, walk_runs,
};
use crate::{
    AdditiveTensorError, TritBlobKind, TritSpoolBlob,
    salt_v2::{B3_TRITS_PER_BYTE, D2_TRITS_PER_BYTE, SaltV2Codec},
    salt_v2_package::{SALT_V2_MAX_PLANES, SaltV2PackageReader},
};

const CHUNK: usize = 64 * 1024;

/// Explicit logical admission for tile-wise encoded staging, not whole RSS.
#[derive(Clone, Copy, Debug)]
pub struct SaltTensorSpoolLimits {
    /// Actual encoded coefficient plus scale source bytes, excluding its container.
    pub max_source_bytes: u64,
    /// Total logical retained coefficient count (not an allocated master).
    pub max_trits: u64,
    /// Total logical retained scale count (not an allocated scale master).
    pub max_scales: u64,
    /// Conservative count/row-index/state buffers plus packing/decode/hash scratch.
    /// Excludes reader/caller, capacity slack, allocator overhead and page cache.
    pub max_working_bytes: u64,
    /// Complete canonical coefficient plus exact FP16-scale payload length.
    pub max_payload_bytes: u64,
}

/// Verified staged range and exact row-major counts for a caller's manifest.
/// Original dimensions, legacy identities and Imported lineage remain the caller's
/// obligation. This is not a model-migration receipt or publication permission.
#[derive(Debug)]
pub struct SaltTensorSpoolImport {
    /// Canonical content-addressed range in the caller-owned spool.
    pub blob: TritSpoolBlob,
    /// Actual preserved allocation; legacy V2 still requires positive tile counts.
    pub tile_counts: Vec<u8>,
}

/// Failure staging canonical bytes; a failed fill may leave an unpublished tail.
#[derive(Debug)]
#[non_exhaustive]
pub enum SaltTensorSpoolError {
    /// Shared source/layout/representability/limit/codec validation failed.
    Import(E),
    /// Caller-owned spool seek, read or write failed.
    Io(std::io::Error),
    /// Staged bytes differ from the canonical bytes emitted for a segment.
    DigestMismatch,
}
impl From<E> for SaltTensorSpoolError {
    fn from(error: E) -> Self {
        Self::Import(error)
    }
}
impl From<std::io::Error> for SaltTensorSpoolError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
impl core::fmt::Display for SaltTensorSpoolError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Import(e) => e.fmt(f),
            Self::Io(e) => write!(f, "legacy SALT encoded spool failed: {e}"),
            Self::DigestMismatch => f.write_str("legacy SALT encoded spool digest mismatch"),
        }
    }
}
impl std::error::Error for SaltTensorSpoolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Import(e) => Some(e),
            Self::Io(e) => Some(e),
            Self::DigestMismatch => None,
        }
    }
}
type Error = SaltTensorSpoolError;

/// Read-only metadata/count plan, not scale representability or import evidence.
/// These logical sizes exclude the reader, allocator/slack, page cache and RSS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltTensorSpoolPlan {
    /// Actual indexed coefficient/scale source payload.
    pub source_bytes: u64,
    /// Retained logical coefficient count, not a decoded allocation.
    pub trits: u64,
    /// Retained logical exact-FP16 scale count.
    pub scales: u64,
    /// Maximum active additive plane count.
    pub planes: usize,
    /// Canonical encoded coefficient plus scale payload.
    pub payload_bytes: u64,
    /// Conservative importer count/index/state/packing scratch.
    pub working_bytes: u64,
    /// Exact target TensorStorage CBOR fragment size; not a whole manifest size.
    pub storage_metadata_bytes: u64,
}

struct PayloadPlan {
    codec: SaltV2Codec,
    lengths: [usize; SALT_V2_MAX_PLANES],
    coefficient_bytes: usize,
    total: usize,
    working: usize,
}
fn payload_plan(
    plan: &PerTilePlan<'_>,
    layout: AdditiveLayout,
    limits: SaltTensorSpoolLimits,
) -> Result<PayloadPlan, Error> {
    import_bound("output trits", plan.trit_count(), limits.max_trits)?;
    import_bound("output scales", plan.scale_count(), limits.max_scales)?;
    let planes = plan.plane_count();
    let codec = match layout.codec {
        PlaneCodec::D2 => SaltV2Codec::D2,
        PlaneCodec::B3 => SaltV2Codec::B3,
        _ => return Err(E::Unsupported("lossless flattened V2 layout").into()),
    };
    let mut lengths = [0; SALT_V2_MAX_PLANES];
    let mut coefficient_bytes = 0usize;
    for (p, length) in lengths.iter_mut().enumerate().take(planes) {
        *length = codec
            .ledger(plan.plane_trit_count(p).map_err(view_error)?)
            .map_err(|e| E::Tensor(AdditiveTensorError::Codec(e)))?
            .physical_bytes;
        coefficient_bytes = coefficient_bytes.checked_add(*length).ok_or(E::Overflow)?;
    }
    let total = plan
        .scale_count()
        .checked_mul(2)
        .and_then(|n| n.checked_add(coefficient_bytes))
        .ok_or(E::Overflow)?;
    import_bound("encoded payload bytes", total, limits.max_payload_bytes)?;
    let working = planes
        .checked_mul(core::mem::size_of::<PlaneStream>() + 2 * CHUNK)
        .and_then(|n| n.checked_add(CHUNK + 256))
        .and_then(|n| n.checked_add(plan.buffer_len_bytes().ok()?))
        .ok_or(E::Overflow)?;
    import_bound("spool working bytes", working, limits.max_working_bytes)?;
    Ok(PayloadPlan {
        codec,
        lengths,
        coefficient_bytes,
        total,
        working,
    })
}
fn prepared<R: Read + Seek>(
    reader: &SaltV2PackageReader<R>,
    name: &str,
    layout: AdditiveLayout,
    limits: SaltTensorSpoolLimits,
) -> Result<Prepared, Error> {
    Ok(prepare(
        reader,
        name,
        layout,
        SaltTensorImportLimits {
            max_source_bytes: limits.max_source_bytes,
            max_trits: limits.max_trits,
            max_scales: limits.max_scales,
            max_owned_bytes: limits.max_working_bytes,
        },
    )?)
}

/// Inspect through the actual import metadata/count/codec planner without writes.
/// Does not visit payload scales or prove scale-bit representability. The caller
/// still owns source custody, actual import/replay and full resource qualification.
/// # Errors
/// The same layout/count/logical-cap refusals as actual spool staging.
pub fn inspect_salt_v2_tensor_spool<R: Read + Seek>(
    reader: &SaltV2PackageReader<R>,
    name: &str,
    layout: AdditiveLayout,
    limits: SaltTensorSpoolLimits,
) -> Result<SaltTensorSpoolPlan, Error> {
    let p = prepared(reader, name, layout, limits)?;
    let plan = PerTilePlan::new(layout, &p.counts).map_err(view_error)?;
    let payload = payload_plan(&plan, layout, limits)?;
    let info = reader.tensor_info(name).ok_or(E::Geometry)?;
    Ok(SaltTensorSpoolPlan {
        source_bytes: info
            .encoded_payload_bytes()
            .checked_add(info.encoded_scale_bytes())
            .ok_or(E::Overflow)?,
        trits: plan.trit_count() as u64,
        scales: plan.scale_count() as u64,
        planes: plan.plane_count(),
        payload_bytes: payload.total as u64,
        working_bytes: payload.working as u64,
        storage_metadata_bytes: crate::trit_model::tensor_storage_cbor_bytes(
            &TensorStorage::Additive {
                layout,
                planes: PlaneCounts::PerTile(p.counts),
            },
        )
        .map_err(|_| E::Overflow)?,
    })
}

/// Stage the owned importer's identical canonical payload without a decoded master.
///
/// Uses the same strict reader, count/scale proof, sealed geometry cursors and
/// codec. Supported layouts and refusals are identical to import_salt_v2_tensor.
/// Appends only at the current end of a caller-owned pending seek/read/write
/// spool; prior bytes are never overwritten. Source proof and explicit limits
/// precede writes. Fixed per-plane buffers carry packing tails across tiles and
/// pruned/ragged rows, preserving plane-major payload order and exact scale bits.
/// Selected-tensor integrity and all stream ledgers must pass, then the range is
/// re-read/hashed before returning. No flush/sync or publication is performed.
///
/// Caller must ensure spool immutability, clean up failed pending tails, verify
/// the whole legacy source unchanged once after all model reads, preserve model
/// metadata/assets/identity/lineage and re-open/verify output bits before commit.
/// The working cap is logical importer storage, not reader/caller/full RSS.
/// # Errors
/// Refusal before fill leaves destination bytes untouched (cursor may change).
/// Fill/source/codec/spool failure may leave partial tail, never a success blob.
pub fn import_salt_v2_tensor_to_spool<R: Read + Seek, S: Read + Write + Seek>(
    reader: &mut SaltV2PackageReader<R>,
    name: &str,
    layout: AdditiveLayout,
    limits: SaltTensorSpoolLimits,
    spool: &mut S,
) -> Result<SaltTensorSpoolImport, Error> {
    let Prepared {
        width,
        source_group,
        counts,
    } = prepared(reader, name, layout, limits)?;
    let plan = PerTilePlan::new(layout, &counts).map_err(view_error)?;
    let planes = plan.plane_count();
    let PayloadPlan {
        codec,
        lengths,
        coefficient_bytes,
        total,
        ..
    } = payload_plan(&plan, layout, limits)?;
    prove_scales(reader, name, width, usize::from(layout.group), source_group)?;
    let offset = spool.stream_position()?;
    if spool.seek(SeekFrom::End(0))? != offset {
        return Err(E::Unsupported("spool must append at its current end").into());
    }
    let end = offset
        .checked_add(u64::try_from(total).map_err(|_| E::Overflow)?)
        .ok_or(E::Overflow)?;
    let mut spans = reserved(plan.index_len(), PerTileRowSpan::default())?;
    let index = plan.build(&mut spans).map_err(view_error)?;
    let rows = usize::try_from(layout.rows).map_err(|_| E::Overflow)?;
    let mut streams = Vec::new();
    streams
        .try_reserve_exact(planes)
        .map_err(|_| E::Allocation)?;
    let mut position = offset;
    for (p, &length) in lengths.iter().enumerate().take(planes) {
        let (trit_start, scale_start) = plane_start(index, p, rows)?;
        let (trit_end, scale_end) = if p + 1 == planes {
            (plan.trit_count(), plan.scale_count())
        } else {
            plane_start(index, p + 1, rows)?
        };
        let scale_offset = offset
            .checked_add(coefficient_bytes as u64)
            .and_then(|n| n.checked_add((scale_start as u64).checked_mul(2)?))
            .ok_or(E::Overflow)?;
        streams.push(PlaneStream::new(
            codec,
            position,
            length,
            trit_start..trit_end,
            scale_offset,
            scale_start..scale_end,
        )?);
        position = position.checked_add(length as u64).ok_or(E::Overflow)?;
    }
    let mut decoded = Vec::new();
    decoded.try_reserve_exact(256).map_err(|_| E::Allocation)?;
    let mut cursors: [Option<RowCursor<'_>>; SALT_V2_MAX_PLANES] = core::array::from_fn(|_| None);
    let source_codec = reader.codec();
    let group = usize::from(layout.group);
    let mut fill: Result<(), Error> = Ok(());
    reader
        .visit_packed_tensor(name, |plane| {
            if fill.is_ok() {
                fill = (|| {
                    decode(source_codec, plane, &mut decoded)?;
                    walk_runs(plane, width, group, source_group, |run| {
                        let p = plane.plane_index();
                        let tile = locate(index, &mut cursors[p], p, run.row, run.column)?;
                        let columns = tile.columns();
                        if run.column + run.len > columns.end {
                            return Err(E::Geometry.into());
                        }
                        let within = run.column - columns.start;
                        let stream = streams.get_mut(p).ok_or(E::Geometry)?;
                        stream.push(
                            spool,
                            tile.trit_offset() + within,
                            &decoded[run.source..run.source + run.len],
                        )?;
                        let scale = tile.scale_offset() + within / group;
                        if within.is_multiple_of(group) {
                            stream.push_scale(spool, scale, run.bits)?;
                        } else if stream.next_scale != scale + 1 {
                            return Err(E::Geometry.into());
                        }
                        Ok(())
                    })
                })();
            }
        })
        .map_err(E::Read)?;
    fill?;
    for stream in &mut streams {
        stream.finish(spool)?;
    }
    let mut hash = blake3::Hasher::new();
    // Bind returned bytes to the canonical bytes emitted, not just whatever a
    // mutable spool happens to contain. Reuse already-admitted segment buffers.
    for stream in &mut streams {
        stream.coefficients.verify(spool, &mut hash)?;
    }
    for stream in &mut streams {
        stream.scales.verify(spool, &mut hash)?;
    }
    seek(spool, end)?;
    Ok(SaltTensorSpoolImport {
        blob: TritSpoolBlob {
            kind: TritBlobKind::AdditiveTensor,
            id: BlobId::from_bytes(*hash.finalize().as_bytes()),
            offset,
            length: total as u64,
        },
        tile_counts: counts,
    })
}

// Read plane prefixes through the shared sealed cursor, not another geometry table.
fn plane_start(index: PerTileIndex<'_>, plane: usize, rows: usize) -> Result<(usize, usize), E> {
    for row in 0..rows {
        if let Some(tile) = index.tiles_in_row(plane, row).map_err(view_error)?.next() {
            return Ok((tile.trit_offset(), tile.scale_offset()));
        }
    }
    Err(E::Geometry)
}

fn seek(spool: &mut impl Seek, offset: u64) -> Result<(), Error> {
    if spool.seek(SeekFrom::Start(offset))? != offset {
        return Err(E::Geometry.into());
    }
    Ok(())
}

struct Segment {
    offset: u64,
    expected: usize,
    written: usize,
    buffer: Vec<u8>,
    hash: blake3::Hasher,
}
impl Segment {
    fn new(offset: u64, expected: usize) -> Result<Self, E> {
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(CHUNK).map_err(|_| E::Allocation)?;
        Ok(Self {
            offset,
            expected,
            written: 0,
            buffer,
            hash: blake3::Hasher::new(),
        })
    }
    fn push(&mut self, spool: &mut (impl Write + Seek), mut bytes: &[u8]) -> Result<(), Error> {
        let produced = self
            .written
            .checked_add(self.buffer.len())
            .and_then(|n| n.checked_add(bytes.len()))
            .ok_or(E::Overflow)?;
        if produced > self.expected {
            return Err(E::Geometry.into());
        }
        self.hash.update(bytes);
        while !bytes.is_empty() {
            let count = bytes.len().min(CHUNK - self.buffer.len());
            self.buffer.extend_from_slice(&bytes[..count]);
            bytes = &bytes[count..];
            if self.buffer.len() == CHUNK {
                self.flush(spool)?;
            }
        }
        Ok(())
    }
    fn flush(&mut self, spool: &mut (impl Write + Seek)) -> Result<(), Error> {
        if !self.buffer.is_empty() {
            seek(
                spool,
                self.offset
                    .checked_add(self.written as u64)
                    .ok_or(E::Overflow)?,
            )?;
            spool.write_all(&self.buffer)?;
            self.written += self.buffer.len();
            self.buffer.clear();
        }
        Ok(())
    }
    fn finish(&mut self, spool: &mut (impl Write + Seek)) -> Result<(), Error> {
        self.flush(spool)?;
        if self.written != self.expected {
            return Err(E::Geometry.into());
        }
        Ok(())
    }
    fn verify(
        &mut self,
        spool: &mut (impl Read + Seek),
        hash: &mut blake3::Hasher,
    ) -> Result<(), Error> {
        seek(spool, self.offset)?;
        let mut actual = blake3::Hasher::new();
        let mut remaining = self.expected;
        while remaining != 0 {
            let count = remaining.min(CHUNK);
            self.buffer.resize(count, 0);
            spool.read_exact(&mut self.buffer)?;
            actual.update(&self.buffer);
            hash.update(&self.buffer);
            remaining -= count;
        }
        self.buffer.clear();
        if actual.finalize() != self.hash.finalize() {
            return Err(Error::DigestMismatch);
        }
        Ok(())
    }
}

struct PlaneStream {
    codec: SaltV2Codec,
    pending: [Trit; 260],
    pending_len: usize,
    encoded: [u8; 65],
    next_trit: usize,
    trit_end: usize,
    next_scale: usize,
    scale_end: usize,
    coefficients: Segment,
    scales: Segment,
}
impl PlaneStream {
    fn new(
        codec: SaltV2Codec,
        offset: u64,
        length: usize,
        trits: core::ops::Range<usize>,
        scale_offset: u64,
        scales: core::ops::Range<usize>,
    ) -> Result<Self, E> {
        Ok(Self {
            codec,
            pending: [Trit::ZERO; 260],
            pending_len: 0,
            encoded: [0; 65],
            next_trit: trits.start,
            trit_end: trits.end,
            next_scale: scales.start,
            scale_end: scales.end,
            coefficients: Segment::new(offset, length)?,
            scales: Segment::new(
                scale_offset,
                scales.len().checked_mul(2).ok_or(E::Overflow)?,
            )?,
        })
    }
    fn push(
        &mut self,
        spool: &mut (impl Write + Seek),
        offset: usize,
        trits: &[Trit],
    ) -> Result<(), Error> {
        if offset != self.next_trit || trits.len() > 256 {
            return Err(E::Geometry.into());
        }
        self.next_trit = self.next_trit.checked_add(trits.len()).ok_or(E::Overflow)?;
        if self.next_trit > self.trit_end {
            return Err(E::Geometry.into());
        }
        let count = self.pending_len + trits.len();
        self.pending[self.pending_len..count].copy_from_slice(trits);
        let word = match self.codec {
            SaltV2Codec::D2 => D2_TRITS_PER_BYTE,
            SaltV2Codec::B3 => B3_TRITS_PER_BYTE,
            _ => return Err(E::Unsupported("spooled output codec").into()),
        };
        let full = count / word * word;
        if full != 0 {
            self.codec
                .pack_into(&self.pending[..full], &mut self.encoded[..full / word])
                .map_err(|e| E::Tensor(AdditiveTensorError::Codec(e)))?;
            self.coefficients
                .push(spool, &self.encoded[..full / word])?;
        }
        self.pending.copy_within(full..count, 0);
        self.pending_len = count - full;
        Ok(())
    }
    fn push_scale(
        &mut self,
        spool: &mut (impl Write + Seek),
        offset: usize,
        bits: u16,
    ) -> Result<(), Error> {
        if offset != self.next_scale || self.next_scale == self.scale_end {
            return Err(E::Geometry.into());
        }
        self.scales.push(spool, &bits.to_le_bytes())?;
        self.next_scale += 1;
        Ok(())
    }
    fn finish(&mut self, spool: &mut (impl Write + Seek)) -> Result<(), Error> {
        if self.next_trit != self.trit_end || self.next_scale != self.scale_end {
            return Err(E::Geometry.into());
        }
        if self.pending_len != 0 {
            self.codec
                .pack_into(&self.pending[..self.pending_len], &mut self.encoded[..1])
                .map_err(|e| E::Tensor(AdditiveTensorError::Codec(e)))?;
            self.coefficients.push(spool, &self.encoded[..1])?;
            self.pending_len = 0;
        }
        self.coefficients.finish(spool)?;
        self.scales.finish(spool)
    }
}
