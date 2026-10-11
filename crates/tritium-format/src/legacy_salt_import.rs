//! Lossless logical SALT import; container identity/basis/provenance belong to its caller.
use half::f16;
use tritium_core::Trit;
use tritium_schema::{
    AdditiveLayout, Basis, LayoutError, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor,
    ScalePrecision, Transport, admitted_execution_group,
};

use crate::{
    AdditiveTensor, AdditiveTensorError, FormatError, QK_K, SaltRow, TQ2_0_BLOCK_BYTES,
    unpack_tq2_0_block,
};

/// Explicit logical input/output bounds; not process RSS or container admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltTensorImportLimits {
    /// Aggregate actual packed plane bytes, including canonical block padding.
    pub max_source_bytes: u64,
    /// Retained logical trit count across every plane.
    pub max_trits: u64,
    /// Retained widened scale count, including exact subgroup replication.
    pub max_scales: u64,
    /// Trit bytes plus f32 scale bytes; allocator overhead and inputs are excluded.
    pub max_owned_bytes: u64,
}

/// Failure to preserve the exact admitted logical legacy tensor.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum SaltTensorImportError {
    /// The declared shared layout is structurally invalid.
    Layout(LayoutError),
    /// No lossless mapping is admitted for this layout or row representation.
    Unsupported(&'static str),
    /// Row count, width, plane count or physical plane length disagrees.
    Geometry,
    /// Checked byte/count/index arithmetic cannot be represented.
    Overflow,
    /// An explicit logical resource cap was exceeded before output allocation.
    Limit {
        /// Resource whose count was rejected.
        resource: &'static str,
        /// Actual required count.
        required: u64,
        /// Caller-supplied maximum.
        maximum: u64,
    },
    /// The shared decoder rejected a physical block.
    Format(FormatError),
    /// Stored scale is negative, negative zero or nonfinite.
    Scale,
    /// Logical row padding contains nonzero trits.
    Padding,
    /// A fallible output reservation failed.
    Allocation,
    /// The owned semantic tensor rejected the preserved output.
    Tensor(AdditiveTensorError),
}
impl core::fmt::Display for SaltTensorImportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "legacy SALT tensor import rejected: {self:?}")
    }
}
impl std::error::Error for SaltTensorImportError {}

/// Preserve actual legacy logical trits and independently stored FP16 plane scales.
///
/// The caller must bind `layout.basis` to verified container metadata and retain
/// imported provenance/legacy identities. This function does not infer rotation,
/// create a model manifest, refit weights, authorize Direct provenance or qualify
/// an artifact. Canonical zero padding is removed; scales are replicated across
/// admitted subgroups without changing any bit. Inputs remain unchanged.
///
/// # Errors
/// Invalid/unadmitted layout, row geometry, limits, codes/scales/padding or allocation.
pub fn import_salt_rows(
    rows: &[SaltRow],
    layout: AdditiveLayout,
    limits: SaltTensorImportLimits,
) -> Result<AdditiveTensor, SaltTensorImportError> {
    use SaltTensorImportError as E;
    layout.validate().map_err(E::Layout)?;
    if layout.law.anchor != ScaleAnchor::Group
        || layout.law.relation != PlaneRelation::Free
        || layout.law.precision != ScalePrecision::F16
        || !admitted_execution_group(layout.law, layout.group)
        || layout.allocation != PlaneAllocation::Uniform
        || !matches!(layout.codec, PlaneCodec::D2 | PlaneCodec::B3)
        || layout.transport != Transport::Raw
        || !matches!(layout.basis, Basis::Identity | Basis::Hadamard { .. })
    {
        return Err(E::Unsupported("lossless legacy row layout"));
    }
    let height = usize::try_from(layout.rows).map_err(|_| E::Overflow)?;
    let width = usize::try_from(layout.cols).map_err(|_| E::Overflow)?;
    let first = rows.first().ok_or(E::Geometry)?;
    let planes = first.planes.len();
    if rows.len() != height || planes == 0 || planes > usize::from(layout.max_planes) {
        return Err(E::Geometry);
    }
    let blocks = width.div_ceil(QK_K);
    let plane_bytes = blocks.checked_mul(TQ2_0_BLOCK_BYTES).ok_or(E::Overflow)?;
    let coefficients = height.checked_mul(width).ok_or(E::Overflow)?;
    let trit_count = coefficients.checked_mul(planes).ok_or(E::Overflow)?;
    let group = usize::from(layout.group);
    let per_row = width.div_ceil(group);
    let scale_units = height.checked_mul(per_row).ok_or(E::Overflow)?;
    let scale_count = scale_units.checked_mul(planes).ok_or(E::Overflow)?;
    let source_bytes = height
        .checked_mul(planes)
        .and_then(|n| n.checked_mul(plane_bytes))
        .ok_or(E::Overflow)?;
    let owned_bytes = scale_count
        .checked_mul(core::mem::size_of::<f32>())
        .and_then(|n| n.checked_add(trit_count))
        .ok_or(E::Overflow)?;
    for (resource, required, maximum) in [
        ("source bytes", source_bytes, limits.max_source_bytes),
        ("output trits", trit_count, limits.max_trits),
        ("output scales", scale_count, limits.max_scales),
        ("owned payload bytes", owned_bytes, limits.max_owned_bytes),
    ] {
        let required = u64::try_from(required).map_err(|_| E::Overflow)?;
        if required > maximum {
            return Err(E::Limit {
                resource,
                required,
                maximum,
            });
        }
    }
    // Reject malformed input before allocating retained output. The second pass
    // uses the same decoder; there is no hand-maintained TQ2 unpacker here.
    for row in rows {
        if row.k != width || row.planes.len() != planes {
            return Err(E::Geometry);
        }
        for plane in &row.planes {
            if plane.len() != plane_bytes {
                return Err(E::Geometry);
            }
            for (index, block) in plane.as_chunks::<TQ2_0_BLOCK_BYTES>().0.iter().enumerate() {
                let (trits, _) = block_values(block)?;
                let count = (width - index * QK_K).min(QK_K);
                if trits[count..].iter().any(|&t| t != Trit::ZERO) {
                    return Err(E::Padding);
                }
            }
        }
    }
    let mut trits = Vec::new();
    trits
        .try_reserve_exact(trit_count)
        .map_err(|_| E::Allocation)?;
    trits.resize(trit_count, Trit::ZERO);
    let mut scales = Vec::new();
    scales
        .try_reserve_exact(scale_count)
        .map_err(|_| E::Allocation)?;
    scales.resize(scale_count, 0.0);
    for (row_index, row) in rows.iter().enumerate() {
        for (plane_index, plane) in row.planes.iter().enumerate() {
            for (block_index, block) in plane.as_chunks::<TQ2_0_BLOCK_BYTES>().0.iter().enumerate()
            {
                let (digits, scale) = block_values(block)?;
                let start = block_index * QK_K;
                let count = (width - start).min(QK_K);
                let offset = plane_index * coefficients + row_index * width + start;
                trits[offset..offset + count].copy_from_slice(&digits[..count]);
                for column in (start..start + count).step_by(group) {
                    scales[plane_index * scale_units + row_index * per_row + column / group] =
                        scale;
                }
            }
        }
    }
    AdditiveTensor::new(layout, planes as u8, trits, scales).map_err(E::Tensor)
}

fn block_values(block: &[u8]) -> Result<([Trit; QK_K], f32), SaltTensorImportError> {
    let mut trits = [Trit::ZERO; QK_K];
    let mut scale = f16::ZERO;
    unpack_tq2_0_block(block, &mut trits, &mut scale).map_err(SaltTensorImportError::Format)?;
    if !scale.is_finite() || scale.to_bits() & 0x8000 != 0 {
        return Err(SaltTensorImportError::Scale);
    }
    Ok((trits, scale.to_f32()))
}
