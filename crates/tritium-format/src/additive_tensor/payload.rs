//! Borrowed canonical payloads and the shared bounded range-validation plan.
use super::*;
use crate::salt_v2::{B3_TRITS_PER_BYTE, D2_TRITS_PER_BYTE, S34_TRITS_PER_GROUP};
use tritium_core::UniformPlan;
use tritium_schema::PlaneCounts;

mod prefix;

const SCRATCH_TRITS: usize = D2_TRITS_PER_BYTE * B3_TRITS_PER_BYTE * (8 * S34_TRITS_PER_GROUP);
pub(crate) const RANGE_BYTES: usize = 64 * 1024;

/// Internal range seam; source custody is owned by the enclosing package reader.
pub(crate) trait PayloadSource {
    type Error: From<AdditiveTensorError>;
    fn read_range(&mut self, blob: usize, offset: usize, out: &mut [u8])
    -> Result<(), Self::Error>;
}

pub(crate) struct SlicePayloads<'a, E, const N: usize> {
    slices: [&'a [u8]; N],
    error: core::marker::PhantomData<E>,
}
impl<'a, E, const N: usize> SlicePayloads<'a, E, N> {
    pub(crate) fn new(slices: [&'a [u8]; N]) -> Self {
        Self {
            slices,
            error: core::marker::PhantomData,
        }
    }
}
impl<E: From<AdditiveTensorError>, const N: usize> PayloadSource for SlicePayloads<'_, E, N> {
    type Error = E;
    fn read_range(&mut self, blob: usize, offset: usize, out: &mut [u8]) -> Result<(), E> {
        let end = offset
            .checked_add(out.len())
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        let bytes = self
            .slices
            .get(blob)
            .and_then(|bytes| bytes.get(offset..end))
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        out.copy_from_slice(bytes);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Allocation<'a> {
    Uniform(u8),
    PerTile(&'a [u8]),
}

#[derive(Debug)]
struct Plane {
    offset: usize,
    bytes: usize,
    trits: usize,
}

/// Shared geometry/codec/scale interpretation; retains no payload or master.
#[derive(Debug)]
pub(crate) struct PayloadPlan<'a> {
    layout: AdditiveLayout,
    allocation: Allocation<'a>,
    planes: Vec<Plane>,
    trit_count: usize,
    scale_count: usize,
    scale_offset: usize,
    length: usize,
}
impl<'a> PayloadPlan<'a> {
    pub(crate) fn new(
        layout: AdditiveLayout,
        planes: &'a PlaneCounts,
        length: usize,
    ) -> Result<Self, AdditiveTensorError> {
        let allocation = match planes {
            PlaneCounts::Uniform(count) => Allocation::Uniform(*count),
            PlaneCounts::PerTile(counts) => Allocation::PerTile(counts),
            _ => {
                return Err(AdditiveTensorError::InvalidView(
                    AdditiveError::UnsupportedAllocation,
                ));
            }
        };
        Self::with_allocation(layout, allocation, length)
    }

    fn with_allocation(
        layout: AdditiveLayout,
        allocation: Allocation<'a>,
        length: usize,
    ) -> Result<Self, AdditiveTensorError> {
        if !matches!(layout.transport, tritium_schema::Transport::Raw) {
            return Err(AdditiveTensorError::UnsupportedTransport);
        }
        let (count, trit_count, scale_count, compact) = match allocation {
            Allocation::Uniform(count) => {
                let plan = UniformPlan::new(layout, count).map_err(|error| match error {
                    AdditiveError::SizeOverflow => AdditiveTensorError::LengthOverflow,
                    other => AdditiveTensorError::InvalidView(other),
                })?;
                (
                    plan.plane_count(),
                    plan.trit_count(),
                    plan.scale_count(),
                    None,
                )
            }
            Allocation::PerTile(counts) => {
                let plan =
                    PerTilePlan::new(layout, counts).map_err(AdditiveTensorError::InvalidView)?;
                (
                    plan.plane_count(),
                    plan.trit_count(),
                    plan.scale_count(),
                    Some(plan),
                )
            }
        };
        let codec = salt_codec(layout.codec)?;
        let mut planes = reserved_payload(count)?;
        let mut offset = 0usize;
        for plane in 0..count {
            let trits = if let Some(plan) = compact {
                plan.plane_trit_count(plane)
                    .map_err(AdditiveTensorError::InvalidView)?
            } else {
                trit_count / count
            };
            let bytes = encoded_plane_len(codec, trits)?;
            planes.push(Plane {
                offset,
                bytes,
                trits,
            });
            offset = offset
                .checked_add(bytes)
                .ok_or(AdditiveTensorError::LengthOverflow)?;
        }
        let scale_len = scale_count
            .checked_mul(scale_width(layout.law.precision)?)
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        let expected = offset
            .checked_add(scale_len)
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        if length != expected {
            return Err(AdditiveTensorError::PayloadLength {
                expected,
                got: length,
            });
        }
        Ok(Self {
            layout,
            allocation,
            planes,
            trit_count,
            scale_count,
            scale_offset: offset,
            length,
        })
    }

    pub(crate) const fn trit_count(&self) -> usize {
        self.trit_count
    }
    pub(crate) fn scale_len(&self) -> usize {
        self.length - self.scale_offset
    }

    pub(crate) fn validate<S: PayloadSource>(
        &self,
        source: &mut S,
        blob: usize,
    ) -> Result<(), S::Error> {
        self.visit_trits(source, blob, |_| Ok(()))?;
        self.visit_scale_bytes(source, blob, |bytes| {
            validate_scale_chunk(self.layout, self.planes.len(), bytes).map_err(Into::into)
        })
    }

    pub(crate) fn visit_trits<S: PayloadSource>(
        &self,
        source: &mut S,
        blob: usize,
        mut visit: impl FnMut(&[Trit]) -> Result<(), S::Error>,
    ) -> Result<(), S::Error> {
        let codec = salt_codec(self.layout.codec)?;
        let mut scratch = reserved_payload(SCRATCH_TRITS)?;
        let mut packed = [0u8; SCRATCH_TRITS];
        for plane in &self.planes {
            let mut trit_offset = 0;
            let mut byte_offset = 0;
            while trit_offset < plane.trits {
                let trits = (plane.trits - trit_offset).min(SCRATCH_TRITS);
                let len = encoded_plane_len(codec, trits)?;
                source.read_range(blob, plane.offset + byte_offset, &mut packed[..len])?;
                decode_window(
                    codec,
                    &packed[..len],
                    trits,
                    &mut scratch,
                    trit_offset,
                    byte_offset,
                )?;
                visit(&scratch)?;
                trit_offset += trits;
                byte_offset += len;
            }
            debug_assert_eq!(byte_offset, plane.bytes);
        }
        Ok(())
    }

    pub(crate) fn visit_scale_bytes<S: PayloadSource>(
        &self,
        source: &mut S,
        blob: usize,
        mut visit: impl FnMut(&[u8]) -> Result<(), S::Error>,
    ) -> Result<(), S::Error> {
        let mut scratch = [0u8; RANGE_BYTES];
        let mut at = self.scale_offset;
        while at < self.length {
            let count = (self.length - at).min(scratch.len());
            source.read_range(blob, at, &mut scratch[..count])?;
            visit(&scratch[..count])?;
            at += count;
        }
        Ok(())
    }
}

fn validate_scale_chunk(
    layout: AdditiveLayout,
    planes: usize,
    bytes: &[u8],
) -> Result<(), AdditiveTensorError> {
    let width = scale_width(layout.law.precision)?;
    for bytes in bytes.chunks_exact(width) {
        let scale = match layout.law.precision {
            ScalePrecision::F16 => {
                f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
            }
            ScalePrecision::Bf16 => {
                bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
            }
            ScalePrecision::F32 => {
                f32::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            }
            _ => return Err(AdditiveTensorError::LengthOverflow),
        };
        if !scale.is_finite() || scale.is_sign_negative() {
            return Err(AdditiveTensorError::InvalidView(
                AdditiveError::InvalidScale,
            ));
        }
        if let PlaneRelation::Tied { num, den } = layout.law.relation {
            let ratio = num as f32 / den as f32;
            let mut derived = scale;
            for _ in 1..planes {
                derived *= ratio;
                if !derived.is_finite() {
                    return Err(AdditiveTensorError::InvalidView(
                        AdditiveError::InvalidScale,
                    ));
                }
            }
        }
    }
    Ok(())
}

fn decode_window(
    codec: SaltV2Codec,
    packed: &[u8],
    trits: usize,
    scratch: &mut Vec<Trit>,
    trit_offset: usize,
    byte_offset: usize,
) -> Result<(), AdditiveTensorError> {
    match codec {
        SaltV2Codec::D2 => unpack_d2_into(packed, trits, scratch),
        SaltV2Codec::B3 => unpack_b3_into(packed, trits, scratch),
        SaltV2Codec::S34 => unpack_s34_into(packed, trits, scratch),
    }
    .map_err(|error| AdditiveTensorError::Codec(shift_error(error, trit_offset, byte_offset)))
}

/// Validated borrowed canonical raw additive payload, not a decoded tensor master.
///
/// Counts and bytes stay borrowed. Shared plan validation and visits use fixed
/// codec scratch plus at most one range per plane, never a decoded master.
#[derive(Debug)]
pub struct AdditivePayload<'a> {
    plan: PayloadPlan<'a>,
    bytes: &'a [u8],
}
impl<'a> AdditivePayload<'a> {
    /// Validate bytes under externally authenticated layout and count metadata.
    pub fn new(
        layout: AdditiveLayout,
        planes: &'a PlaneCounts,
        bytes: &'a [u8],
    ) -> Result<Self, AdditiveTensorError> {
        let plan = PayloadPlan::new(layout, planes, bytes.len())?;
        Self::from_plan(plan, bytes)
    }
    pub(super) fn with_allocation(
        layout: AdditiveLayout,
        allocation: Allocation<'a>,
        bytes: &'a [u8],
    ) -> Result<Self, AdditiveTensorError> {
        let plan = PayloadPlan::with_allocation(layout, allocation, bytes.len())?;
        Self::from_plan(plan, bytes)
    }
    fn from_plan(plan: PayloadPlan<'a>, bytes: &'a [u8]) -> Result<Self, AdditiveTensorError> {
        plan.validate(
            &mut SlicePayloads::<AdditiveTensorError, 1>::new([bytes]),
            0,
        )?;
        Ok(Self { plan, bytes })
    }

    /// The externally bound, validated semantic layout.
    #[must_use]
    pub const fn layout(&self) -> AdditiveLayout {
        self.plan.layout
    }
    /// Stored planes, zero only for all-pruned compact payloads.
    #[must_use]
    pub fn plane_count(&self) -> usize {
        self.plan.planes.len()
    }
    /// Complete decoded coefficient count, without allocating coefficients.
    #[must_use]
    pub const fn trit_count(&self) -> usize {
        self.plan.trit_count
    }
    /// Exact stored little-endian scale bytes in canonical order.
    #[must_use]
    pub fn scale_bytes(&self) -> &'a [u8] {
        &self.bytes[self.plan.scale_offset..]
    }
    /// Borrow compact row-major counts; uniform allocation returns None.
    #[must_use]
    pub const fn tile_counts(&self) -> Option<&'a [u8]> {
        match self.plan.allocation {
            Allocation::Uniform(_) => None,
            Allocation::PerTile(counts) => Some(counts),
        }
    }
    /// Visit canonical plane-major coefficients in chunks of at most640 trits.
    pub fn visit_trits(&self, mut visit: impl FnMut(&[Trit])) -> Result<(), AdditiveTensorError> {
        self.plan.visit_trits(
            &mut SlicePayloads::<AdditiveTensorError, 1>::new([self.bytes]),
            0,
            |trits| {
                visit(trits);
                Ok(())
            },
        )
    }
    /// Explicitly allocate a decoded owned tensor using the same validated plan.
    pub fn to_owned(&self) -> Result<AdditiveTensor, AdditiveTensorError> {
        let mut trits = reserved_payload(self.plan.trit_count)?;
        self.visit_trits(|chunk| trits.extend_from_slice(chunk))?;
        let mut scales = reserved_payload(self.plan.scale_count)?;
        extend_scales(
            self.plan.layout.law.precision,
            self.scale_bytes(),
            &mut scales,
        )?;
        match self.plan.allocation {
            Allocation::Uniform(count) => {
                AdditiveTensor::new(self.plan.layout, count, trits, scales)
            }
            Allocation::PerTile(counts) => {
                AdditiveTensor::new_per_tile(self.plan.layout, copy_payload(counts)?, trits, scales)
            }
        }
    }
}

fn shift_error(error: SaltV2CodecError, trits: usize, bytes: usize) -> SaltV2CodecError {
    match error {
        SaltV2CodecError::InvalidD2Code { trit_index, code } => SaltV2CodecError::InvalidD2Code {
            trit_index: trits + trit_index,
            code,
        },
        SaltV2CodecError::InvalidB3Code { byte_index, code } => SaltV2CodecError::InvalidB3Code {
            byte_index: bytes + byte_index,
            code,
        },
        SaltV2CodecError::NonCanonicalTritPadding {
            codec,
            trit_index,
            code,
        } => SaltV2CodecError::NonCanonicalTritPadding {
            codec,
            trit_index: trits + trit_index,
            code,
        },
        SaltV2CodecError::NonCanonicalBitPadding {
            codec,
            byte_index,
            bits,
        } => SaltV2CodecError::NonCanonicalBitPadding {
            codec,
            byte_index: bytes + byte_index,
            bits,
        },
        other => other,
    }
}
