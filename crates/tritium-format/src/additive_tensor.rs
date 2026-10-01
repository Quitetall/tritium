//! Owned, validated additive-ternary tensors.

use half::{bf16, f16};
use tritium_core::{AdditiveError, AdditiveView, Trit};
use tritium_schema::{AdditiveLayout, ScalePrecision};

/// Failure constructing an owned [`AdditiveTensor`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum AdditiveTensorError {
    /// The semantic view rejected layout, plane, coefficient, or scale data.
    InvalidView(AdditiveError),
    /// A scale is not exactly representable in its declared storage precision.
    ScalePrecisionMismatch {
        /// Scale index that cannot be represented exactly.
        index: usize,
    },
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
    /// Validate and own a decoded additive tensor.
    pub fn new(
        layout: AdditiveLayout,
        plane_count: u8,
        trits: Vec<Trit>,
        scales: Vec<f32>,
    ) -> Result<Self, AdditiveTensorError> {
        AdditiveView::new(layout, plane_count, &trits, &scales)
            .map_err(AdditiveTensorError::InvalidView)?;
        for (index, scale) in scales.iter().copied().enumerate() {
            let roundtrip = match layout.law.precision {
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
        Ok(Self {
            layout,
            plane_count,
            trits,
            scales,
        })
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

#[cfg(test)]
mod tests {
    use super::{AdditiveTensor, AdditiveTensorError};
    use tritium_core::Trit;
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
}
