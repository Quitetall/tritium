//! Borrowed canonical payloads; geometry and codecs are shared with owned tensors.

use super::*;
use crate::salt_v2::{B3_TRITS_PER_BYTE, D2_TRITS_PER_BYTE, S34_TRITS_PER_GROUP};
use tritium_core::UniformPlan;
use tritium_schema::PlaneCounts;

mod prefix;

// Every nonterminal chunk ends at a byte boundary for all three codecs.
const SCRATCH_TRITS: usize = D2_TRITS_PER_BYTE * B3_TRITS_PER_BYTE * (8 * S34_TRITS_PER_GROUP);

#[derive(Clone, Copy, Debug)]
pub(super) enum Allocation<'a> {
    Uniform(u8),
    PerTile(&'a [u8]),
}

#[derive(Debug)]
struct Plane {
    offset: usize,
    bytes: usize,
    trits: usize,
}

/// Validated borrowed canonical raw additive payload, not a decoded tensor master.
///
/// Counts and bytes stay borrowed. Validation and visits use fixed codec scratch
/// plus at most one range per plane; no coefficient or scale master is allocated.
/// This establishes payload structure, not provenance or execution capability.
#[derive(Debug)]
pub struct AdditivePayload<'a> {
    layout: AdditiveLayout,
    allocation: Allocation<'a>,
    planes: Vec<Plane>,
    trit_count: usize,
    scale_count: usize,
    scale_offset: usize,
    bytes: &'a [u8],
}

impl<'a> AdditivePayload<'a> {
    /// Validate bytes under externally authenticated layout and count metadata.
    pub fn new(
        layout: AdditiveLayout,
        planes: &'a PlaneCounts,
        bytes: &'a [u8],
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
        Self::with_allocation(layout, allocation, bytes)
    }

    pub(super) fn with_allocation(
        layout: AdditiveLayout,
        allocation: Allocation<'a>,
        bytes: &'a [u8],
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
            let len = encoded_plane_len(codec, trits)?;
            planes.push(Plane {
                offset,
                bytes: len,
                trits,
            });
            offset = offset
                .checked_add(len)
                .ok_or(AdditiveTensorError::LengthOverflow)?;
        }
        let scale_len = scale_count
            .checked_mul(scale_width(layout.law.precision)?)
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        let expected = offset
            .checked_add(scale_len)
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        if bytes.len() != expected {
            return Err(AdditiveTensorError::PayloadLength {
                expected,
                got: bytes.len(),
            });
        }
        let view = Self {
            layout,
            allocation,
            planes,
            trit_count,
            scale_count,
            scale_offset: offset,
            bytes,
        };
        view.visit_trits(|_| {})?;
        view.validate_scales()?;
        Ok(view)
    }

    /// The externally bound, validated semantic layout.
    #[must_use]
    pub const fn layout(&self) -> AdditiveLayout {
        self.layout
    }

    /// Number of stored planes, zero only for all-pruned compact payloads.
    #[must_use]
    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }

    /// Complete decoded coefficient count, without allocating the coefficients.
    #[must_use]
    pub const fn trit_count(&self) -> usize {
        self.trit_count
    }

    /// Exact stored little-endian scale bytes in canonical order.
    #[must_use]
    pub fn scale_bytes(&self) -> &'a [u8] {
        &self.bytes[self.scale_offset..]
    }

    /// Borrow compact row-major counts; uniform allocation returns `None`.
    #[must_use]
    pub const fn tile_counts(&self) -> Option<&'a [u8]> {
        match self.allocation {
            Allocation::Uniform(_) => None,
            Allocation::PerTile(counts) => Some(counts),
        }
    }

    /// Visit validated logical coefficients in canonical plane-major order.
    ///
    /// Chunks contain at most 640 trits. The callback cannot retain scratch.
    pub fn visit_trits(&self, mut visit: impl FnMut(&[Trit])) -> Result<(), AdditiveTensorError> {
        let codec = salt_codec(self.layout.codec)?;
        let mut scratch = reserved_payload(SCRATCH_TRITS)?;
        for plane in &self.planes {
            let mut trit_offset = 0;
            let mut byte_offset = 0;
            while trit_offset < plane.trits {
                let trits = (plane.trits - trit_offset).min(SCRATCH_TRITS);
                let len = encoded_plane_len(codec, trits)?;
                let start = plane.offset + byte_offset;
                let packed = &self.bytes[start..start + len];
                let result = match codec {
                    SaltV2Codec::D2 => unpack_d2_into(packed, trits, &mut scratch),
                    SaltV2Codec::B3 => unpack_b3_into(packed, trits, &mut scratch),
                    SaltV2Codec::S34 => unpack_s34_into(packed, trits, &mut scratch),
                };
                result.map_err(|error| {
                    AdditiveTensorError::Codec(shift_error(error, trit_offset, byte_offset))
                })?;
                visit(&scratch);
                trit_offset += trits;
                byte_offset += len;
            }
            debug_assert_eq!(byte_offset, plane.bytes);
        }
        Ok(())
    }

    /// Explicitly allocate a decoded owned tensor using the same validated plan.
    pub fn to_owned(&self) -> Result<AdditiveTensor, AdditiveTensorError> {
        let mut trits = reserved_payload(self.trit_count)?;
        self.visit_trits(|chunk| trits.extend_from_slice(chunk))?;
        let mut scales = reserved_payload(self.scale_count)?;
        extend_scales(self.layout.law.precision, self.scale_bytes(), &mut scales)?;
        match self.allocation {
            Allocation::Uniform(count) => AdditiveTensor::new(self.layout, count, trits, scales),
            Allocation::PerTile(counts) => {
                AdditiveTensor::new_per_tile(self.layout, copy_payload(counts)?, trits, scales)
            }
        }
    }

    fn validate_scales(&self) -> Result<(), AdditiveTensorError> {
        let width = scale_width(self.layout.law.precision)?;
        for bytes in self.scale_bytes().chunks_exact(width) {
            let scale = match self.layout.law.precision {
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
            if let PlaneRelation::Tied { num, den } = self.layout.law.relation {
                let ratio = num as f32 / den as f32;
                let mut derived = scale;
                for _ in 1..self.planes.len() {
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
