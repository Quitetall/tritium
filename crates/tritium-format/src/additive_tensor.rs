//! Owned, validated additive-ternary tensors.

use half::{bf16, f16};
use tritium_core::{AdditiveError, AdditiveView, Trit};
use tritium_schema::{AdditiveLayout, PlaneCodec, PlaneRelation, ScaleAnchor, ScalePrecision};

use crate::salt_v2::{
    SaltV2Codec, SaltV2CodecError, pack_b3, pack_d2, pack_s34, unpack_b3, unpack_d2, unpack_s34,
};

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
}

/// Owned semantic additive tensor used between fitters, package codecs, and backends.
#[derive(Clone, Debug, PartialEq)]
pub struct AdditiveTensor {
    layout: AdditiveLayout,
    plane_count: u8,
    trits: Vec<Trit>,
    scales: Vec<f32>,
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
        let coefficients = tensor_coefficients(self.layout)?;
        let codec = salt_codec(self.layout.codec)?;
        let plane_bytes = encoded_plane_len(codec, coefficients)?;
        let scale_bytes = self
            .scales
            .len()
            .checked_mul(scale_width(self.layout.law.precision)?)
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        let total = plane_bytes
            .checked_mul(usize::from(self.plane_count))
            .and_then(|bytes| bytes.checked_add(scale_bytes))
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(total)
            .map_err(|_| AdditiveTensorError::AllocationFailed { requested: total })?;
        for plane in self.trits.chunks_exact(coefficients) {
            let encoded = match codec {
                SaltV2Codec::D2 => pack_d2(plane),
                SaltV2Codec::B3 => pack_b3(plane),
                SaltV2Codec::S34 => pack_s34(plane),
            }
            .map_err(AdditiveTensorError::Codec)?;
            payload.extend_from_slice(&encoded);
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
        if !matches!(layout.transport, tritium_schema::Transport::Raw) {
            return Err(AdditiveTensorError::UnsupportedTransport);
        }
        layout.validate().map_err(|error| {
            AdditiveTensorError::InvalidView(AdditiveError::InvalidLayout(error))
        })?;
        if plane_count == 0 || plane_count > layout.max_planes {
            return Err(AdditiveTensorError::InvalidView(
                AdditiveError::InvalidPlaneCount,
            ));
        }
        let coefficients = tensor_coefficients(layout)?;
        let codec = salt_codec(layout.codec)?;
        let plane_bytes = encoded_plane_len(codec, coefficients)?;
        let scale_units = tensor_scale_units(layout)?;
        let scale_count = match layout.law.relation {
            PlaneRelation::Free => scale_units
                .checked_mul(usize::from(plane_count))
                .ok_or(AdditiveTensorError::LengthOverflow)?,
            PlaneRelation::Tied { .. } => scale_units,
            _ => return Err(AdditiveTensorError::LengthOverflow),
        };
        let scale_len = scale_count
            .checked_mul(scale_width(layout.law.precision)?)
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        let encoded_planes = plane_bytes
            .checked_mul(usize::from(plane_count))
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        let expected = encoded_planes
            .checked_add(scale_len)
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        if payload.len() != expected {
            return Err(AdditiveTensorError::PayloadLength {
                expected,
                got: payload.len(),
            });
        }
        let mut trits = Vec::new();
        let trit_count = coefficients
            .checked_mul(usize::from(plane_count))
            .ok_or(AdditiveTensorError::LengthOverflow)?;
        trits
            .try_reserve_exact(trit_count)
            .map_err(|_| AdditiveTensorError::AllocationFailed {
                requested: trit_count,
            })?;
        for plane in payload[..encoded_planes].chunks_exact(plane_bytes) {
            let decoded = match codec {
                SaltV2Codec::D2 => unpack_d2(plane, coefficients),
                SaltV2Codec::B3 => unpack_b3(plane, coefficients),
                SaltV2Codec::S34 => unpack_s34(plane, coefficients),
            }
            .map_err(AdditiveTensorError::Codec)?;
            trits.extend(decoded);
        }
        let scale_data = &payload[encoded_planes..];
        let mut scales = Vec::new();
        scales.try_reserve_exact(scale_count).map_err(|_| {
            AdditiveTensorError::AllocationFailed {
                requested: scale_count,
            }
        })?;
        match layout.law.precision {
            ScalePrecision::F16 => scales.extend(
                scale_data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|bytes| f16::from_bits(u16::from_le_bytes(*bytes)).to_f32()),
            ),
            ScalePrecision::Bf16 => scales.extend(
                scale_data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|bytes| bf16::from_bits(u16::from_le_bytes(*bytes)).to_f32()),
            ),
            ScalePrecision::F32 => scales.extend(
                scale_data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_bits(u32::from_le_bytes(*bytes))),
            ),
            _ => return Err(AdditiveTensorError::LengthOverflow),
        }
        Self::new(layout, plane_count, trits, scales)
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

    /// Create a zero-copy semantic view over this owned tensor.
    #[must_use]
    pub fn view(&self) -> AdditiveView<'_> {
        // Constructor established all invariants and fields are immutable.
        AdditiveView::new(self.layout, self.plane_count, &self.trits, &self.scales)
            .expect("AdditiveTensor invariants are immutable after construction")
    }
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

fn tensor_coefficients(layout: AdditiveLayout) -> Result<usize, AdditiveTensorError> {
    let rows = usize::try_from(layout.rows).map_err(|_| AdditiveTensorError::LengthOverflow)?;
    let cols = usize::try_from(layout.cols).map_err(|_| AdditiveTensorError::LengthOverflow)?;
    rows.checked_mul(cols)
        .ok_or(AdditiveTensorError::LengthOverflow)
}

fn tensor_scale_units(layout: AdditiveLayout) -> Result<usize, AdditiveTensorError> {
    let rows = usize::try_from(layout.rows).map_err(|_| AdditiveTensorError::LengthOverflow)?;
    let cols = usize::try_from(layout.cols).map_err(|_| AdditiveTensorError::LengthOverflow)?;
    match layout.law.anchor {
        ScaleAnchor::Group => rows
            .checked_mul(cols.div_ceil(usize::from(layout.group)))
            .ok_or(AdditiveTensorError::LengthOverflow),
        ScaleAnchor::Row | ScaleAnchor::Channel => Ok(rows),
        ScaleAnchor::Tensor => Ok(1),
        _ => Err(AdditiveTensorError::LengthOverflow),
    }
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
    match codec {
        SaltV2Codec::D2 => coefficients
            .checked_add(3)
            .map(|len| len / 4)
            .ok_or(AdditiveTensorError::LengthOverflow),
        SaltV2Codec::B3 => coefficients
            .checked_add(4)
            .map(|len| len / 5)
            .ok_or(AdditiveTensorError::LengthOverflow),
        SaltV2Codec::S34 if coefficients.is_multiple_of(4) => (coefficients / 4)
            .checked_mul(5)
            .map(|bits| bits.div_ceil(8))
            .ok_or(AdditiveTensorError::LengthOverflow),
        SaltV2Codec::S34 => Err(AdditiveTensorError::Codec(
            SaltV2CodecError::S34TritCountNotMultipleOfFour {
                logical_trits: coefficients,
            },
        )),
    }
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
