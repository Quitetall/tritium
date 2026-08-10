//! Backend-neutral compact SALT snapshots used by differentiable runtimes.

use half::f16;
use tritium_core::Trit;

use crate::{FormatError, QK_K, TQ2_0_BLOCK_BYTES, pack_tq2_0_block};

const TQ2_CODE_BYTES: usize = QK_K / 4;
const MAX_PLANES: usize = 3;

/// Constraint applied independently to each additive ternary plane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TernaryStructure {
    /// Every scalar code is independently chosen from `{-1, 0, 1}`.
    Dense,
    /// Every contiguous four-code row block has exactly one zero and three signs.
    S34,
}

impl TernaryStructure {
    /// Stable wire tag used by refinement checkpoints.
    #[must_use]
    pub const fn wire_tag(self) -> u8 {
        match self {
            Self::Dense => 0,
            Self::S34 => 1,
        }
    }

    /// Decode one stable checkpoint wire tag.
    #[must_use]
    pub const fn from_wire_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(Self::Dense),
            1 => Some(Self::S34),
            _ => None,
        }
    }
}

/// Borrowed semantic plane supplied to [`PackedTrainingSaltSnapshot::pack`].
#[derive(Clone, Copy, Debug)]
pub struct TrainingSaltPlane<'a> {
    trits: &'a [i8],
    scales: &'a [f16],
}

impl<'a> TrainingSaltPlane<'a> {
    /// Borrow canonical row-major trits and row-major group scales.
    #[must_use]
    pub const fn new(trits: &'a [i8], scales: &'a [f16]) -> Self {
        Self { trits, scales }
    }
}

/// Validated plane-major TQ2 codes plus external f32 group scales.
///
/// TQ2 block scales are deliberately omitted: training kernels address only
/// canonical 64-byte code payloads, while one external scale applies to each
/// declared row group. Backends receive this already-packed snapshot and never
/// recreate host packing rules.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedTrainingSaltSnapshot {
    codes: Vec<u8>,
    scales: Vec<f32>,
    rows: usize,
    cols: usize,
    planes: usize,
    group_size: usize,
    groups_per_row: usize,
    row_bytes: usize,
    structure: TernaryStructure,
}

impl PackedTrainingSaltSnapshot {
    /// Validate semantic planes and pack canonical TQ2 code payloads.
    ///
    /// # Errors
    /// [`FormatError::InvalidTrainingSalt`] rejects zero or overflowing
    /// geometry, invalid plane counts, noncanonical trits/scales, codes hidden
    /// behind zero scales, and malformed S34 blocks.
    pub fn pack(
        rows: usize,
        cols: usize,
        group_size: usize,
        structure: TernaryStructure,
        planes: &[TrainingSaltPlane<'_>],
    ) -> Result<Self, FormatError> {
        if rows == 0 || cols == 0 || group_size == 0 {
            return Err(invalid("rows, cols, and group size must be nonzero"));
        }
        if !(1..=MAX_PLANES).contains(&planes.len()) {
            return Err(invalid("plane count must be between one and three"));
        }
        if structure == TernaryStructure::S34
            && (!cols.is_multiple_of(4) || !group_size.is_multiple_of(4))
        {
            return Err(invalid(
                "S34 requires cols and group size divisible by four",
            ));
        }
        let elements = rows
            .checked_mul(cols)
            .ok_or_else(|| invalid("element count overflows usize"))?;
        let groups_per_row = cols.div_ceil(group_size);
        let scales_per_plane = rows
            .checked_mul(groups_per_row)
            .ok_or_else(|| invalid("scale count overflows usize"))?;
        let row_bytes = cols
            .div_ceil(QK_K)
            .checked_mul(TQ2_CODE_BYTES)
            .ok_or_else(|| invalid("row byte count overflows usize"))?;
        let code_bytes = planes
            .len()
            .checked_mul(rows)
            .and_then(|value| value.checked_mul(row_bytes))
            .ok_or_else(|| invalid("packed byte count overflows usize"))?;
        let scale_count = planes
            .len()
            .checked_mul(scales_per_plane)
            .ok_or_else(|| invalid("total scale count overflows usize"))?;
        for plane in planes {
            validate_plane(
                plane,
                rows,
                cols,
                group_size,
                groups_per_row,
                elements,
                scales_per_plane,
                structure,
            )?;
        }
        let mut codes = Vec::new();
        codes
            .try_reserve_exact(code_bytes)
            .map_err(|_| invalid("packed code allocation failed"))?;
        codes.resize(code_bytes, 0x55);
        let mut scales = Vec::new();
        scales
            .try_reserve_exact(scale_count)
            .map_err(|_| invalid("scale allocation failed"))?;
        let mut block = [Trit::ZERO; QK_K];
        let mut packed_block = [0_u8; TQ2_0_BLOCK_BYTES];

        for (plane_index, plane) in planes.iter().enumerate() {
            scales.extend(plane.scales.iter().map(|scale| f32::from(*scale)));
            for row in 0..rows {
                let row_start = row * cols;
                for block_index in 0..cols.div_ceil(QK_K) {
                    let start = row_start + block_index * QK_K;
                    let end = (start + QK_K).min(row_start + cols);
                    block.fill(Trit::ZERO);
                    for (target, &source) in block.iter_mut().zip(&plane.trits[start..end]) {
                        *target = Trit::from_i8(source)
                            .map_err(|_| invalid("plane contains a noncanonical trit"))?;
                    }
                    pack_tq2_0_block(&block, f16::ONE, &mut packed_block)?;
                    let offset =
                        ((plane_index * rows + row) * row_bytes) + block_index * TQ2_CODE_BYTES;
                    codes[offset..offset + TQ2_CODE_BYTES]
                        .copy_from_slice(&packed_block[..TQ2_CODE_BYTES]);
                }
            }
        }

        Ok(Self {
            codes,
            scales,
            rows,
            cols,
            planes: planes.len(),
            group_size,
            groups_per_row,
            row_bytes,
            structure,
        })
    }

    /// Plane-major TQ2 code bytes without embedded block scales.
    #[must_use]
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Plane-major, row-major external f32 group scales.
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Matrix row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Matrix column count.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Additive plane count.
    #[must_use]
    pub const fn planes(&self) -> usize {
        self.planes
    }

    /// Columns sharing one external scale.
    #[must_use]
    pub const fn group_size(&self) -> usize {
        self.group_size
    }

    /// Scale groups in each row.
    #[must_use]
    pub const fn groups_per_row(&self) -> usize {
        self.groups_per_row
    }

    /// TQ2 code bytes occupied by one row in one plane.
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Discrete code constraint carried by this snapshot.
    #[must_use]
    pub const fn structure(&self) -> TernaryStructure {
        self.structure
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_plane(
    plane: &TrainingSaltPlane<'_>,
    rows: usize,
    cols: usize,
    group_size: usize,
    groups_per_row: usize,
    elements: usize,
    scales_per_plane: usize,
    structure: TernaryStructure,
) -> Result<(), FormatError> {
    if plane.trits.len() != elements || plane.scales.len() != scales_per_plane {
        return Err(invalid("plane has wrong code or scale count"));
    }
    if plane.trits.iter().any(|trit| !matches!(trit, -1..=1)) {
        return Err(invalid("plane contains a noncanonical trit"));
    }
    for row in 0..rows {
        for group in 0..groups_per_row {
            let scale = f32::from(plane.scales[row * groups_per_row + group]);
            if !scale.is_finite() || scale < 0.0 || (scale == 0.0 && scale.is_sign_negative()) {
                return Err(invalid("plane contains an invalid scale"));
            }
            let start = row * cols + group * group_size;
            let end = (start + group_size).min((row + 1) * cols);
            if scale == 0.0 && plane.trits[start..end].iter().any(|&trit| trit != 0) {
                return Err(invalid("zero scale hides nonzero codes"));
            }
        }
        if structure == TernaryStructure::S34 {
            for block in plane.trits[row * cols..(row + 1) * cols].chunks_exact(4) {
                if block.iter().filter(|&&trit| trit == 0).count() != 1 {
                    return Err(invalid("plane violates S34 structure"));
                }
            }
        }
    }
    Ok(())
}

const fn invalid(reason: &'static str) -> FormatError {
    FormatError::InvalidTrainingSalt(reason)
}
