//! Owned, validated additive-ternary tensors.

use half::{bf16, f16};
use tritium_core::{AdditiveError, AdditiveView, OwnedPerTileIndex, PerTilePlan, Trit};
use tritium_schema::{AdditiveLayout, PlaneCodec, PlaneRelation, ScaleAnchor, ScalePrecision};

use crate::salt_v2::{
    SaltV2Codec, SaltV2CodecError, unpack_b3_into, unpack_d2_into, unpack_s34_into,
};

mod payload;
pub use payload::{AdditivePayload, AdditiveStreamValidator};
pub(crate) use payload::{PayloadPlan, PayloadSource, RANGE_BYTES};

/// Failure constructing an owned [`AdditiveTensor`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum AdditiveTensorError {
    /// The semantic view rejected layout, plane, coefficient, or scale data.
    InvalidView(AdditiveError),
    /// A scale is not exactly representable in its declared storage precision.
    ScalePrecisionMismatch {
        /// Scale index that cannot be represented exactly.
        index: usize,
    },
    /// A checked tensor payload could not be sized or allocated.
    LengthOverflow,
    /// Encoded payload does not have the one canonical size for its metadata.
    PayloadLength {
        /// Canonical byte length implied by the tensor metadata.
        expected: usize,
        /// Actual byte length supplied.
        got: usize,
    },
    /// The declared codec rejected a noncanonical physical payload.
    Codec(SaltV2CodecError),
    /// A fallible payload allocation could not be reserved.
    AllocationFailed {
        /// Exact byte or element count requested by the failed reservation.
        requested: usize,
    },
    /// This tensor-level codec does not itself apply the outer entropy transport.
    UnsupportedTransport,
    /// Incremental validation previously failed; ignoring it cannot yield success.
    StreamPoisoned,
}

/// Owned semantic additive tensor used between fitters, package codecs, and backends.
#[derive(Clone, Debug, PartialEq)]
pub struct AdditiveTensor {
    layout: AdditiveLayout,
    plane_count: u8,
    trits: Vec<Trit>,
    scales: Vec<f32>,
    per_tile: Option<OwnedPerTileIndex>,
}

impl AdditiveTensor {
    /// Copy a validated semantic view with fallible buffer reservations.
    ///
    /// The owned constructor additionally enforces exact stored scale precision.
    /// Coefficients remain ternary; this does not materialize dense weights.
    pub fn from_view(view: AdditiveView<'_>) -> Result<Self, AdditiveTensorError> {
        Self::validate_view(view)?;
        let planes =
            u8::try_from(view.plane_count()).map_err(|_| AdditiveTensorError::LengthOverflow)?;
        let scale_bytes = core::mem::size_of_val(view.scales());
        view.trits()
            .len()
            .checked_add(scale_bytes)
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        Ok(Self {
            layout: view.layout(),
            plane_count: planes,
            trits: copy_payload(view.trits())?,
            scales: copy_payload(view.scales())?,
            per_tile: view
                .tile_counts()
                .map(|counts| {
                    OwnedPerTileIndex::new(view.layout(), copy_payload(counts)?)
                        .map_err(AdditiveTensorError::InvalidView)
                })
                .transpose()?,
        })
    }

    /// Validate and own a decoded additive tensor.
    pub fn new(
        layout: AdditiveLayout,
        plane_count: u8,
        trits: Vec<Trit>,
        scales: Vec<f32>,
    ) -> Result<Self, AdditiveTensorError> {
        let view = AdditiveView::new(layout, plane_count, &trits, &scales)
            .map_err(AdditiveTensorError::InvalidView)?;
        Self::validate_view(view)?;
        Ok(Self {
            layout,
            plane_count,
            trits,
            scales,
            per_tile: None,
        })
    }

    /// Own compact payloads and counts under the shared checked index geometry.
    ///
    /// No dense weights are materialized. All-pruned counts require empty
    /// coefficients/scales. Exact stored-scale precision is enforced as usual.
    pub fn new_per_tile(
        layout: AdditiveLayout,
        counts: Vec<u8>,
        trits: Vec<Trit>,
        scales: Vec<f32>,
    ) -> Result<Self, AdditiveTensorError> {
        let index =
            OwnedPerTileIndex::new(layout, counts).map_err(AdditiveTensorError::InvalidView)?;
        Self::from_per_tile_index(index, trits, scales)
    }

    // Transfers a previously admitted shared index; no duplicate index buffer
    // during streaming migration. Public constructors retain the same checks.
    pub(crate) fn from_per_tile_index(
        index: OwnedPerTileIndex,
        trits: Vec<Trit>,
        scales: Vec<f32>,
    ) -> Result<Self, AdditiveTensorError> {
        let view = AdditiveView::new_per_tile(index.as_index(), &trits, &scales)
            .map_err(AdditiveTensorError::InvalidView)?;
        Self::validate_view(view)?;
        let layout = view.layout();
        let plane_count =
            u8::try_from(view.plane_count()).map_err(|_| AdditiveTensorError::LengthOverflow)?;
        Ok(Self {
            layout,
            plane_count,
            trits,
            scales,
            per_tile: Some(index),
        })
    }

    /// Validate exact stored-scale precision without allocating or copying.
    ///
    /// Used by capability queries and before reservations in [`Self::from_view`].
    pub fn validate_view(view: AdditiveView<'_>) -> Result<(), AdditiveTensorError> {
        for (index, scale) in view.scales().iter().copied().enumerate() {
            if scale.to_bits() & 0x8000_0000 != 0 {
                return Err(AdditiveTensorError::InvalidView(
                    AdditiveError::InvalidScale,
                ));
            }
            let roundtrip = match view.layout().law.precision {
                ScalePrecision::F16 => f16::from_f32(scale).to_f32(),
                ScalePrecision::Bf16 => bf16::from_f32(scale).to_f32(),
                ScalePrecision::F32 => scale,
                _ => {
                    return Err(AdditiveTensorError::InvalidView(
                        AdditiveError::InvalidScale,
                    ));
                }
            };
            if roundtrip.to_bits() != scale.to_bits() {
                return Err(AdditiveTensorError::ScalePrecisionMismatch { index });
            }
        }
        Ok(())
    }

    /// Encode canonical tensor payload bytes (plane payloads followed by scales).
    ///
    /// Geometry, codec, and scale precision are bound by the surrounding
    /// manifest; this method encodes only the content-addressed tensor blob.
    pub fn encode_payload(&self) -> Result<Vec<u8>, AdditiveTensorError> {
        if !matches!(self.layout.transport, tritium_schema::Transport::Raw) {
            return Err(AdditiveTensorError::UnsupportedTransport);
        }
        let view = self.view();
        let codec = salt_codec(self.layout.codec)?;
        let mut encoded_planes = 0usize;
        for plane in 0..view.plane_count() {
            let coefficients = view
                .plane_trits(plane)
                .map_err(AdditiveTensorError::InvalidView)?
                .len();
            encoded_planes = encoded_planes
                .checked_add(encoded_plane_len(codec, coefficients)?)
                .ok_or(AdditiveTensorError::LengthOverflow)?;
        }
        let scale_bytes = self
            .scales
            .len()
            .checked_mul(scale_width(self.layout.law.precision)?)
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        let total = encoded_planes
            .checked_add(scale_bytes)
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(total)
            .map_err(|_| AdditiveTensorError::AllocationFailed { requested: total })?;
        payload.resize(encoded_planes, 0);
        let mut cursor = 0;
        for plane in 0..view.plane_count() {
            let trits = view
                .plane_trits(plane)
                .map_err(AdditiveTensorError::InvalidView)?;
            let len = encoded_plane_len(codec, trits.len())?;
            codec
                .pack_into(trits, &mut payload[cursor..cursor + len])
                .map_err(AdditiveTensorError::Codec)?;
            cursor += len;
        }
        match self.layout.law.precision {
            ScalePrecision::F16 => payload.extend(
                self.scales
                    .iter()
                    .flat_map(|scale| f16::from_f32(*scale).to_bits().to_le_bytes()),
            ),
            ScalePrecision::Bf16 => payload.extend(
                self.scales
                    .iter()
                    .flat_map(|scale| bf16::from_f32(*scale).to_bits().to_le_bytes()),
            ),
            ScalePrecision::F32 => payload.extend(
                self.scales
                    .iter()
                    .flat_map(|scale| scale.to_bits().to_le_bytes()),
            ),
            _ => return Err(AdditiveTensorError::LengthOverflow),
        }
        debug_assert_eq!(payload.len(), total);
        Ok(payload)
    }

    /// Decode canonical tensor payload bytes under manifest-bound metadata.
    pub fn decode_payload(
        layout: AdditiveLayout,
        plane_count: u8,
        payload: &[u8],
    ) -> Result<Self, AdditiveTensorError> {
        AdditivePayload::with_allocation(
            layout,
            payload::Allocation::Uniform(plane_count),
            payload,
        )?
        .to_owned()
    }

    /// Decode compact canonical plane payloads under externally bound counts.
    ///
    /// Count metadata is not embedded here; model packages must authenticate it.
    /// Checked preflight rejects wrong physical length before buffer/index
    /// reservations. Shared codecs enforce codes and terminal padding. Fixed
    /// codec scratch is reused; decoded owned storage is explicitly allocated.
    pub fn decode_per_tile_payload(
        layout: AdditiveLayout,
        counts: &[u8],
        payload: &[u8],
    ) -> Result<Self, AdditiveTensorError> {
        AdditivePayload::with_allocation(layout, payload::Allocation::PerTile(counts), payload)?
            .to_owned()
    }

    /// Return the tensor's validated semantic layout.
    #[must_use]
    pub const fn layout(&self) -> AdditiveLayout {
        self.layout
    }

    /// Number of stored planes.
    #[must_use]
    pub const fn plane_count(&self) -> u8 {
        self.plane_count
    }

    /// Borrow the owned plane-major coefficients.
    #[must_use]
    pub fn trits(&self) -> &[Trit] {
        &self.trits
    }

    /// Borrow widened scales in the canonical plane/unit order.
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Validate replacement stored scales without changing coefficients or geometry.
    ///
    /// This also checks derived tied-plane scales and exact storage precision.
    /// No coefficient/index buffers are copied. Failure leaves the tensor unchanged.
    pub fn validate_scales(&self, scales: &[f32]) -> Result<(), AdditiveTensorError> {
        let view = if let Some(index) = &self.per_tile {
            AdditiveView::new_per_tile(index.as_index(), &self.trits, scales)
        } else {
            AdditiveView::new(self.layout, self.plane_count, &self.trits, scales)
        }
        .map_err(AdditiveTensorError::InvalidView)?;
        Self::validate_view(view)
    }

    /// Replace actual stored scales transactionally, preserving all other state.
    ///
    /// Accepts only finite nonnegative values exactly representable by the
    /// declared precision. It never silently narrows a scale or changes a law.
    pub fn replace_scales(&mut self, scales: Vec<f32>) -> Result<(), AdditiveTensorError> {
        self.validate_scales(&scales)?;
        self.scales = scales;
        Ok(())
    }

    /// Borrow row-major compact counts, or `None` for uniform allocation.
    #[must_use]
    pub fn tile_counts(&self) -> Option<&[u8]> {
        self.per_tile.as_ref().map(OwnedPerTileIndex::tile_counts)
    }

    /// Create a zero-copy semantic view over this owned tensor.
    #[must_use]
    pub fn view(&self) -> AdditiveView<'_> {
        // Constructors and checked scale replacement preserve all invariants.
        let view = if let Some(index) = &self.per_tile {
            AdditiveView::new_per_tile(index.as_index(), &self.trits, &self.scales)
        } else {
            AdditiveView::new(self.layout, self.plane_count, &self.trits, &self.scales)
        };
        view.expect("AdditiveTensor invariants are maintained by checked mutations")
    }
}

fn extend_scales(
    precision: ScalePrecision,
    data: &[u8],
    scales: &mut Vec<f32>,
) -> Result<(), AdditiveTensorError> {
    match precision {
        ScalePrecision::F16 => scales.extend(
            data.as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| f16::from_bits(u16::from_le_bytes(*bytes)).to_f32()),
        ),
        ScalePrecision::Bf16 => scales.extend(
            data.as_chunks::<2>()
                .0
                .iter()
                .map(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes)).to_f32()),
        ),
        ScalePrecision::F32 => scales.extend(
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_bits(u32::from_le_bytes(*bytes))),
        ),
        _ => return Err(AdditiveTensorError::LengthOverflow),
    }
    Ok(())
}

fn reserved_payload<T>(count: usize) -> Result<Vec<T>, AdditiveTensorError> {
    let requested = count
        .checked_mul(core::mem::size_of::<T>())
        .ok_or(AdditiveTensorError::LengthOverflow)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| AdditiveTensorError::AllocationFailed { requested })?;
    Ok(values)
}

fn copy_payload<T: Copy>(values: &[T]) -> Result<Vec<T>, AdditiveTensorError> {
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(values.len())
        .map_err(|_| AdditiveTensorError::AllocationFailed {
            requested: core::mem::size_of_val(values),
        })?;
    owned.extend_from_slice(values);
    Ok(owned)
}

fn salt_codec(codec: PlaneCodec) -> Result<SaltV2Codec, AdditiveTensorError> {
    match codec {
        PlaneCodec::D2 => Ok(SaltV2Codec::D2),
        PlaneCodec::B3 => Ok(SaltV2Codec::B3),
        PlaneCodec::S34 => Ok(SaltV2Codec::S34),
        _ => Err(AdditiveTensorError::LengthOverflow),
    }
}

fn encoded_plane_len(
    codec: SaltV2Codec,
    coefficients: usize,
) -> Result<usize, AdditiveTensorError> {
    codec
        .ledger(coefficients)
        .map(|ledger| ledger.physical_bytes)
        .map_err(|error| match error {
            SaltV2CodecError::LengthOverflow { .. } => AdditiveTensorError::LengthOverflow,
            other => AdditiveTensorError::Codec(other),
        })
}

const fn scale_width(precision: ScalePrecision) -> Result<usize, AdditiveTensorError> {
    match precision {
        ScalePrecision::F16 | ScalePrecision::Bf16 => Ok(2),
        ScalePrecision::F32 => Ok(4),
        _ => Err(AdditiveTensorError::LengthOverflow),
    }
}

#[cfg(test)]
mod tests {
    use super::{AdditiveTensor, AdditiveTensorError};
    use tritium_core::{AdditiveError, Trit};
    use tritium_schema::{
        AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
        ScalePrecision, Transport,
    };

    fn layout(precision: ScalePrecision) -> AdditiveLayout {
        AdditiveLayout {
            rows: 1,
            cols: 4,
            tile: 256,
            group: 32,
            max_planes: 3,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::D2,
            law: ScaleLaw {
                anchor: ScaleAnchor::Group,
                relation: PlaneRelation::Free,
                precision,
            },
            basis: Basis::Identity,
            transport: Transport::Raw,
        }
    }

    #[test]
    fn owns_valid_data_and_exposes_a_borrowed_view() {
        let tensor = AdditiveTensor::new(
            layout(ScalePrecision::F16),
            1,
            vec![Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS],
            vec![0.5],
        )
        .expect("valid tensor");
        assert_eq!(tensor.view().plane_count(), 1);
        let mut row = [0.0; 4];
        tensor.view().dequant_row_into(0, &mut row).expect("row");
        assert_eq!(row, [0.5, -0.5, 0.0, 0.5]);
    }

    #[test]
    fn rejects_scales_not_representable_at_declared_precision() {
        let result = AdditiveTensor::new(
            layout(ScalePrecision::F16),
            1,
            vec![Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS],
            vec![0.1],
        );
        assert_eq!(
            result,
            Err(AdditiveTensorError::ScalePrecisionMismatch { index: 0 })
        );
    }

    #[test]
    fn physical_plane_codecs_roundtrip_as_deterministic_payloads() {
        for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
            let mut tensor_layout = layout(ScalePrecision::F16);
            tensor_layout.codec = codec;
            let tensor = AdditiveTensor::new(
                tensor_layout,
                1,
                vec![Trit::POS, Trit::ZERO, Trit::NEG, Trit::POS],
                vec![0.5],
            )
            .expect("valid tensor for each physical codec");
            let payload = tensor.encode_payload().expect("payload encode");
            assert_eq!(tensor.encode_payload().expect("repeat encode"), payload);
            let decoded =
                AdditiveTensor::decode_payload(tensor_layout, 1, &payload).expect("payload decode");
            assert_eq!(decoded, tensor);
        }
    }

    #[test]
    fn payload_decoder_rejects_trailing_bytes_and_noncanonical_scale_bits() {
        let tensor_layout = layout(ScalePrecision::F16);
        let tensor = AdditiveTensor::new(
            tensor_layout,
            1,
            vec![Trit::POS, Trit::ZERO, Trit::NEG, Trit::POS],
            vec![0.5],
        )
        .expect("valid tensor");
        let mut payload = tensor.encode_payload().expect("payload encode");
        payload.push(0);
        assert!(matches!(
            AdditiveTensor::decode_payload(tensor_layout, 1, &payload),
            Err(AdditiveTensorError::PayloadLength { .. })
        ));

        let mut payload = tensor.encode_payload().expect("payload encode");
        *payload.last_mut().expect("scale bytes") = 0x80;
        assert!(matches!(
            AdditiveTensor::decode_payload(tensor_layout, 1, &payload),
            Err(AdditiveTensorError::InvalidView(
                AdditiveError::InvalidScale
            ))
        ));
    }
}
