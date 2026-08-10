use half::f16;
pub use tritium_format::TernaryStructure as PvTernaryStructure;

use super::PvTuningError;

const MAX_PLANES: usize = 3;

pub(super) const fn unit_width(structure: PvTernaryStructure) -> usize {
    match structure {
        PvTernaryStructure::Dense => 1,
        PvTernaryStructure::S34 => 4,
    }
}

pub(super) fn unit_start(weight: &PvTernaryWeight, unit: usize) -> usize {
    match weight.structure {
        PvTernaryStructure::Dense => unit,
        PvTernaryStructure::S34 => {
            let blocks_per_row = weight.cols / unit_width(weight.structure);
            (unit / blocks_per_row) * weight.cols
                + (unit % blocks_per_row) * unit_width(weight.structure)
        }
    }
}

/// One additive ternary plane: canonical trits plus one f16 scale per row-group.
#[derive(Clone, Debug, PartialEq)]
pub struct PvTernaryPlane {
    pub(super) trits: Vec<i8>,
    pub(super) scales: Vec<f16>,
}

impl PvTernaryPlane {
    /// Construct owned plane. Geometry and value checks run in [`PvTernaryWeight::new`].
    #[must_use]
    pub fn new(trits: Vec<i8>, scales: Vec<f16>) -> Self {
        Self { trits, scales }
    }

    /// Canonical row-major trits.
    #[must_use]
    pub fn trits(&self) -> &[i8] {
        &self.trits
    }

    /// Row-major group scales.
    #[must_use]
    pub fn scales(&self) -> &[f16] {
        &self.scales
    }
}

/// Exact deployed additive ternary representation refined by a PV session.
#[derive(Clone, Debug, PartialEq)]
pub struct PvTernaryWeight {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) group_size: usize,
    pub(super) structure: PvTernaryStructure,
    pub(super) planes: Vec<PvTernaryPlane>,
}

impl PvTernaryWeight {
    /// Validate and construct a one-to-three-plane ternary weight.
    ///
    /// # Errors
    /// Returns [`PvTuningError::InvalidWeight`] for invalid geometry, values, or 3:4.
    pub fn new(
        rows: usize,
        cols: usize,
        group_size: usize,
        structure: PvTernaryStructure,
        planes: Vec<PvTernaryPlane>,
    ) -> Result<Self, PvTuningError> {
        let weight = Self {
            rows,
            cols,
            group_size,
            structure,
            planes,
        };
        weight.validate()?;
        Ok(weight)
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

    /// Columns sharing one scale in each row and plane.
    #[must_use]
    pub const fn group_size(&self) -> usize {
        self.group_size
    }

    /// Discrete code constraint.
    #[must_use]
    pub const fn structure(&self) -> PvTernaryStructure {
        self.structure
    }

    /// Additive planes in stable order.
    #[must_use]
    pub fn planes(&self) -> &[PvTernaryPlane] {
        &self.planes
    }

    /// Decode exact deployed representation into row-major f32 scratch.
    #[must_use]
    pub fn decode(&self) -> Vec<f32> {
        let mut decoded = vec![0.0; self.len()];
        let groups_per_row = self.groups_per_row();
        for plane in &self.planes {
            for row in 0..self.rows {
                for col in 0..self.cols {
                    let index = row * self.cols + col;
                    let scale_index = row * groups_per_row + col / self.group_size;
                    decoded[index] +=
                        f32::from(plane.scales[scale_index]) * f32::from(plane.trits[index]);
                }
            }
        }
        decoded
    }

    /// Digest over geometry and exact code/scale bits.
    #[must_use]
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"tritium.pv-ternary-weight.v1\0");
        hasher.update(&(self.rows as u64).to_le_bytes());
        hasher.update(&(self.cols as u64).to_le_bytes());
        hasher.update(&(self.group_size as u64).to_le_bytes());
        hasher.update(&[self.structure.wire_tag(), self.planes.len() as u8]);
        for plane in &self.planes {
            for &trit in &plane.trits {
                hasher.update(&trit.to_le_bytes());
            }
            for scale in &plane.scales {
                hasher.update(&scale.to_bits().to_le_bytes());
            }
        }
        *hasher.finalize().as_bytes()
    }

    pub(super) fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub(super) fn groups_per_row(&self) -> usize {
        self.cols.div_ceil(self.group_size)
    }

    pub(super) fn scale_count_per_plane(&self) -> usize {
        self.rows * self.groups_per_row()
    }

    pub(super) fn total_scale_count(&self) -> usize {
        self.scale_count_per_plane() * self.planes.len()
    }

    pub(super) fn decode_element(&self, index: usize) -> f32 {
        let row = index / self.cols;
        let col = index % self.cols;
        let scale_index = row * self.groups_per_row() + col / self.group_size;
        let mut decoded = 0.0;
        for plane in &self.planes {
            decoded += f32::from(plane.scales[scale_index]) * f32::from(plane.trits[index]);
        }
        decoded
    }

    pub(super) fn validate(&self) -> Result<(), PvTuningError> {
        if self.rows == 0 || self.cols == 0 || self.group_size == 0 {
            return Err(PvTuningError::invalid_weight(
                "rows, cols, and group_size must be nonzero",
            ));
        }
        let Some(len) = self.rows.checked_mul(self.cols) else {
            return Err(PvTuningError::invalid_weight("matrix size overflows usize"));
        };
        if !(1..=MAX_PLANES).contains(&self.planes.len()) {
            return Err(PvTuningError::invalid_weight(
                "plane count must be between one and three",
            ));
        }
        if self.structure == PvTernaryStructure::S34
            && (!self.cols.is_multiple_of(4) || !self.group_size.is_multiple_of(4))
        {
            return Err(PvTuningError::invalid_weight(
                "S34 requires cols and group_size divisible by four",
            ));
        }
        let groups_per_row = self.cols.div_ceil(self.group_size);
        let Some(scale_count) = self.rows.checked_mul(groups_per_row) else {
            return Err(PvTuningError::invalid_weight("scale count overflows usize"));
        };
        for (plane_index, plane) in self.planes.iter().enumerate() {
            validate_plane(self, plane, plane_index, len, scale_count, groups_per_row)?;
        }
        Ok(())
    }
}

fn validate_plane(
    weight: &PvTernaryWeight,
    plane: &PvTernaryPlane,
    plane_index: usize,
    len: usize,
    scale_count: usize,
    groups_per_row: usize,
) -> Result<(), PvTuningError> {
    if plane.trits.len() != len || plane.scales.len() != scale_count {
        return Err(PvTuningError::invalid_weight(format!(
            "plane {plane_index} has wrong code or scale count"
        )));
    }
    if plane.trits.iter().any(|trit| !matches!(trit, -1..=1)) {
        return Err(PvTuningError::invalid_weight(format!(
            "plane {plane_index} contains a noncanonical trit"
        )));
    }
    for (scale_index, scale) in plane.scales.iter().enumerate() {
        let value = f32::from(*scale);
        if !value.is_finite() || value < 0.0 || (value == 0.0 && value.is_sign_negative()) {
            return Err(PvTuningError::invalid_weight(format!(
                "plane {plane_index} contains an invalid scale"
            )));
        }
        let row = scale_index / groups_per_row;
        let group = scale_index % groups_per_row;
        let start = row * weight.cols + group * weight.group_size;
        let end = (start + weight.group_size).min((row + 1) * weight.cols);
        if value == 0.0 && plane.trits[start..end].iter().any(|&trit| trit != 0) {
            return Err(PvTuningError::invalid_weight(format!(
                "plane {plane_index} has nonzero codes behind a zero scale"
            )));
        }
    }
    if weight.structure == PvTernaryStructure::S34 {
        for row in 0..weight.rows {
            for block_col in (0..weight.cols).step_by(4) {
                let start = row * weight.cols + block_col;
                let zeros = plane.trits[start..start + 4]
                    .iter()
                    .filter(|&&trit| trit == 0)
                    .count();
                if zeros != 1 {
                    return Err(PvTuningError::invalid_weight(format!(
                        "plane {plane_index} violates S34 at row {row}, col {block_col}"
                    )));
                }
            }
        }
    }
    Ok(())
}
