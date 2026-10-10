//! Canonical plane storage with direct add/subtract/skip execution.
//! No decoded trit or dense weight shadow is retained or allocated during execution.

use half::f16;
use rayon::prelude::*;
use tritium_core::{AdditiveView, apply_basis, apply_inverse_basis};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_schema::{AdditiveLayout, PlaneCodec, PlaneRelation, ScaleAnchor, ScalePrecision};
use tritium_spec::{BackendError, TensorMatmul};

#[derive(Debug)]
pub(crate) struct PackedAdditive {
    layout: AdditiveLayout,
    planes: usize,
    plane_bytes: usize,
    scale_units: usize,
    packed: Vec<u8>,
}

fn invalid(message: impl core::fmt::Debug) -> BackendError {
    BackendError::InvalidInput(format!("CPU packed tensor: {message:?}"))
}

fn codec(layout: AdditiveLayout) -> Result<SaltV2Codec, BackendError> {
    match layout.codec {
        PlaneCodec::D2 => Ok(SaltV2Codec::D2),
        PlaneCodec::B3 => Ok(SaltV2Codec::B3),
        PlaneCodec::S34 => Ok(SaltV2Codec::S34),
        _ => Err(BackendError::UnsupportedTensor),
    }
}

impl PackedAdditive {
    fn scale_width(layout: AdditiveLayout) -> Result<usize, BackendError> {
        match layout.law.precision {
            ScalePrecision::F16 => Ok(2),
            ScalePrecision::F32 => Ok(4),
            _ => Err(BackendError::UnsupportedTensor),
        }
    }

    pub(super) fn payload_bytes(view: AdditiveView<'_>) -> Result<usize, BackendError> {
        let planes = view.plane_count();
        let coefficients = view.trits().len() / planes;
        let bytes = codec(view.layout())?
            .ledger(coefficients)
            .map_err(invalid)?
            .physical_bytes;
        if view.layout().codec == PlaneCodec::S34 {
            // Semantic views validate laws/basis/scales, not codec-specific sparsity.
            // Validate before a capability promise or any allocation is made.
            for group in view.trits().as_chunks::<4>().0 {
                if group.iter().filter(|trit| trit.is_zero()).count() != 1 {
                    return Err(invalid("S34 needs exactly one zero per four coefficients"));
                }
            }
        }
        let scales = view
            .scales()
            .len()
            .checked_mul(Self::scale_width(view.layout())?)
            .ok_or_else(|| invalid("scale size overflow"))?;
        bytes
            .checked_mul(planes)
            .and_then(|n| n.checked_add(scales))
            .ok_or_else(|| invalid("payload size overflow"))
    }

    pub(super) fn upload(view: AdditiveView<'_>) -> Result<Self, BackendError> {
        let total = Self::payload_bytes(view)?;
        let layout = view.layout();
        let planes = view.plane_count();
        let coefficients = view.trits().len() / planes;
        let plane_bytes = codec(layout)?
            .ledger(coefficients)
            .map_err(invalid)?
            .physical_bytes;
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(total)
            .map_err(|_| BackendError::OutOfMemory { requested: total })?;
        packed.resize(total, 0);
        let (plane_data, scale_data) = packed.split_at_mut(planes * plane_bytes);
        for (source, output) in view
            .trits()
            .chunks_exact(coefficients)
            .zip(plane_data.chunks_exact_mut(plane_bytes))
        {
            codec(layout)?.pack_into(source, output).map_err(invalid)?;
        }
        for (&scale, output) in view
            .scales()
            .iter()
            .zip(scale_data.chunks_exact_mut(Self::scale_width(layout)?))
        {
            match layout.law.precision {
                ScalePrecision::F16 => {
                    output.copy_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes())
                }
                ScalePrecision::F32 => output.copy_from_slice(&scale.to_bits().to_le_bytes()),
                _ => return Err(BackendError::UnsupportedTensor),
            }
        }
        let scale_units = if matches!(layout.law.relation, PlaneRelation::Free) {
            view.scales().len() / planes
        } else {
            view.scales().len()
        };
        Ok(Self {
            layout,
            planes,
            plane_bytes,
            scale_units,
            packed,
        })
    }

    pub(super) fn len_bytes(&self) -> usize {
        self.packed.len()
    }

    // Only canonical encoders construct these private bytes. Plane offsets reset
    // codec padding; rows do not reset codec units (ragged rows can cross bytes).
    fn trit(&self, plane: usize, coefficient: usize) -> i8 {
        let data = &self.packed[plane * self.plane_bytes..(plane + 1) * self.plane_bytes];
        match self.layout.codec {
            PlaneCodec::D2 => ((data[coefficient / 4] >> (2 * (coefficient % 4))) & 3) as i8 - 1,
            PlaneCodec::B3 => B3_DIGITS[usize::from(data[coefficient / 5])][coefficient % 5],
            PlaneCodec::S34 => {
                let bit = (coefficient / 4) * 5;
                let byte = bit / 8;
                let shift = bit % 8;
                let mut word = u16::from(data[byte]);
                if shift > 3 {
                    word |= u16::from(data[byte + 1]) << 8;
                }
                S34_DIGITS[usize::from((word >> shift) & 31)][coefficient % 4]
            }
            _ => unreachable!("private tensor has a validated codec"),
        }
    }

    fn scale(&self, plane: usize, row: usize, col: usize) -> f32 {
        let unit = match self.layout.law.anchor {
            ScaleAnchor::Group => {
                row * (self.layout.cols as usize).div_ceil(usize::from(self.layout.group))
                    + col / usize::from(self.layout.group)
            }
            ScaleAnchor::Tensor => 0,
            _ => unreachable!("upload accepts only currently admitted anchors"),
        };
        let index = if matches!(self.layout.law.relation, PlaneRelation::Free) {
            plane * self.scale_units + unit
        } else {
            unit
        };
        let scales = &self.packed[self.planes * self.plane_bytes..];
        let base = match self.layout.law.precision {
            ScalePrecision::F16 => f16::from_bits(u16::from_le_bytes(
                scales[index * 2..index * 2 + 2]
                    .try_into()
                    .expect("private scale geometry"),
            ))
            .to_f32(),
            ScalePrecision::F32 => f32::from_bits(u32::from_le_bytes(
                scales[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("private scale geometry"),
            )),
            _ => unreachable!("upload accepts only currently admitted precisions"),
        };
        match self.layout.law.relation {
            PlaneRelation::Free => base,
            PlaneRelation::Tied { num, den } => {
                let mut scale = base;
                let ratio = num as f32 / den as f32;
                for _ in 0..plane {
                    scale *= ratio;
                }
                scale
            }
            _ => unreachable!("upload accepts only currently admitted relations"),
        }
    }

    fn dot(&self, row: usize, act: &[f32]) -> f32 {
        let cols = self.layout.cols as usize;
        let group = if self.layout.law.anchor == ScaleAnchor::Group {
            usize::from(self.layout.group)
        } else {
            cols
        };
        let mut sum = 0.0;
        for plane in 0..self.planes {
            for (unit, activation) in act.chunks(group).enumerate() {
                let start = unit * group;
                let mut partial = 0.0;
                for (offset, value) in activation.iter().enumerate() {
                    match self.trit(plane, row * cols + start + offset) {
                        1 => partial += value,
                        -1 => partial -= value,
                        _ => {}
                    }
                }
                sum += partial * self.scale(plane, row, start);
            }
        }
        sum
    }

    pub(super) fn matmul(&self, p: TensorMatmul<'_>) -> Result<(), BackendError> {
        let rows = self.layout.rows as usize;
        let cols = self.layout.cols as usize;
        let inputs = p
            .batch
            .checked_mul(cols)
            .ok_or_else(|| invalid("input size overflow"))?;
        let outputs = p
            .batch
            .checked_mul(rows)
            .ok_or_else(|| invalid("output size overflow"))?;
        if p.act.len() != inputs || p.transformed_act.len() != inputs || p.out.len() != outputs {
            return Err(invalid("matmul shape mismatch"));
        }
        p.transformed_act.copy_from_slice(p.act);
        for activation in p.transformed_act.chunks_exact_mut(cols) {
            apply_basis(activation, self.layout.basis).map_err(invalid)?;
        }
        p.out
            .par_iter_mut()
            .with_min_len(32)
            .enumerate()
            .for_each(|(index, value)| {
                let batch = index / rows;
                *value = self.dot(
                    index % rows,
                    &p.transformed_act[batch * cols..(batch + 1) * cols],
                );
            });
        Ok(())
    }

    pub(super) fn embed(&self, ids: &[usize], out: &mut [f32]) -> Result<(), BackendError> {
        let cols = self.layout.cols as usize;
        let rows = self.layout.rows as usize;
        let count = ids
            .len()
            .checked_mul(cols)
            .ok_or_else(|| invalid("gather size overflow"))?;
        if out.len() != count || ids.iter().any(|&row| row >= rows) {
            return Err(invalid("gather shape or row out of range"));
        }
        for (&row, output) in ids.iter().zip(out.chunks_exact_mut(cols)) {
            output.fill(0.0);
            for plane in 0..self.planes {
                for (col, value) in output.iter_mut().enumerate() {
                    *value +=
                        f32::from(self.trit(plane, row * cols + col)) * self.scale(plane, row, col);
                }
            }
            apply_inverse_basis(output, self.layout.basis).map_err(invalid)?;
        }
        Ok(())
    }
}

const B3_DIGITS: [[i8; 5]; 243] = {
    let mut table = [[0; 5]; 243];
    let mut code = 0;
    while code < 243 {
        let mut n = code;
        let mut slot = 0;
        while slot < 5 {
            table[code][slot] = (n % 3) as i8 - 1;
            n /= 3;
            slot += 1;
        }
        code += 1;
    }
    table
};

const S34_DIGITS: [[i8; 4]; 32] = {
    let mut table = [[0; 4]; 32];
    let mut code = 0;
    while code < 32 {
        let zero = code & 3;
        let mut slot = 0;
        let mut sign = 0;
        while slot < 4 {
            if slot != zero {
                table[code][slot] = if code & (1 << (2 + sign)) == 0 { -1 } else { 1 };
                sign += 1;
            }
            slot += 1;
        }
        code += 1;
    }
    table
};
