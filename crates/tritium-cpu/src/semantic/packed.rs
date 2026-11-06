//! Canonical plane storage with direct add/subtract/skip execution.
//! No decoded trit or dense weight shadow is retained or allocated during execution.

use half::f16;
use rayon::prelude::*;
use tritium_core::{AdditiveView, OwnedPerTileIndex, apply_basis, apply_inverse_basis};
use tritium_format::salt_v2::SaltV2Codec;
use tritium_schema::{AdditiveLayout, PlaneCodec, PlaneRelation, ScaleAnchor, ScalePrecision};
use tritium_spec::{
    AdditiveMetadata, BackendError, TENSOR_STREAM_CHUNK_BYTES, TensorMatmul, TensorStreamSource,
};

#[derive(Debug)]
pub(crate) struct PackedAdditive {
    layout: AdditiveLayout,
    planes: usize,
    scale_offset: usize,
    allocation: Allocation,
    packed: Vec<u8>,
}

#[derive(Debug)]
enum Allocation {
    Uniform {
        plane_bytes: usize,
        scale_units: usize,
    },
    PerTile {
        index: OwnedPerTileIndex,
        starts: Vec<PlaneStart>,
    },
}

#[derive(Debug)]
struct PlaneStart {
    byte: usize,
    trit: usize,
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

    pub(super) fn validate_symbols(view: AdditiveView<'_>) -> Result<(), BackendError> {
        for plane in 0..view.plane_count() {
            let trits = view.plane_trits(plane).map_err(invalid)?;
            if view.layout().codec == PlaneCodec::S34 {
                // Check each plane separately: canonical padding resets there.
                for group in trits.as_chunks::<4>().0 {
                    if group.iter().filter(|trit| trit.is_zero()).count() != 1 {
                        return Err(invalid("S34 needs exactly one zero per four coefficients"));
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn payload_metadata(metadata: AdditiveMetadata<'_>) -> Result<usize, BackendError> {
        let mut total = 0usize;
        let codec = codec(metadata.layout())?;
        for plane in 0..metadata.plane_count() {
            let bytes = codec
                .ledger(metadata.plane_trit_count(plane)?)
                .map_err(invalid)?
                .physical_bytes;
            total = total
                .checked_add(bytes)
                .ok_or_else(|| invalid("plane size overflow"))?;
        }
        let scales = metadata
            .scale_count()
            .checked_mul(Self::scale_width(metadata.layout())?)
            .ok_or_else(|| invalid("scale size overflow"))?;
        if metadata.count_bytes().is_some() {
            let index_bytes = metadata.auxiliary_bytes()?;
            let starts = metadata
                .plane_count()
                .checked_mul(core::mem::size_of::<PlaneStart>())
                .ok_or_else(|| invalid("plane index size overflow"))?;
            total = total
                .checked_add(index_bytes)
                .and_then(|n| n.checked_add(starts))
                .ok_or_else(|| invalid("index size overflow"))?;
        }
        total
            .checked_add(scales)
            .ok_or_else(|| invalid("payload size overflow"))
    }

    pub(super) fn upload(view: AdditiveView<'_>) -> Result<Self, BackendError> {
        Self::validate_symbols(view)?;
        let mut tensor = Self::allocate(AdditiveMetadata::from_view(view))?;
        let layout = tensor.layout;
        let codec = codec(layout)?;
        let (plane_data, scale_data) = tensor.packed.split_at_mut(tensor.scale_offset);
        let mut byte = 0;
        for plane in 0..tensor.planes {
            let source = view.plane_trits(plane).map_err(invalid)?;
            let len = codec.ledger(source.len()).map_err(invalid)?.physical_bytes;
            codec
                .pack_into(source, &mut plane_data[byte..byte + len])
                .map_err(invalid)?;
            byte += len;
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
        Ok(tensor)
    }

    pub(super) fn stream_encoded_bytes(
        metadata: AdditiveMetadata<'_>,
    ) -> Result<usize, BackendError> {
        let mut bytes = metadata
            .scale_count()
            .checked_mul(Self::scale_width(metadata.layout())?)
            .ok_or_else(|| invalid("scale size overflow"))?;
        let codec = codec(metadata.layout())?;
        for p in 0..metadata.plane_count() {
            bytes = bytes
                .checked_add(
                    codec
                        .ledger(metadata.plane_trit_count(p)?)
                        .map_err(invalid)?
                        .physical_bytes,
                )
                .ok_or_else(|| invalid("encoded size overflow"))?;
        }
        Ok(bytes)
    }

    pub(super) fn upload_stream(
        metadata: AdditiveMetadata<'_>,
        source: &dyn TensorStreamSource,
    ) -> Result<Self, BackendError> {
        let mut tensor = Self::allocate(metadata)?;
        let mut at = 0usize;
        source.visit(&mut |chunk| {
            if chunk.len() > TENSOR_STREAM_CHUNK_BYTES {
                return Err(invalid("source window exceeds stream bound"));
            }
            let end = at
                .checked_add(chunk.len())
                .ok_or_else(|| invalid("stream length overflow"))?;
            let output = tensor
                .packed
                .get_mut(at..end)
                .ok_or_else(|| invalid("source exceeds encoded plan"))?;
            output.copy_from_slice(chunk);
            at = end;
            Ok(())
        })?;
        if at != tensor.packed.len() {
            return Err(invalid("source shorter than encoded plan"));
        }
        // CPU canonical plane/scale ordering is identical, including plane-reset
        // padding. Validate this final private native buffer with bounded codec
        // scratch; never decode a complete trit master or retain a second copy.
        if let Some(counts) = metadata.tile_counts() {
            tritium_format::AdditivePayload::new_per_tile(
                metadata.layout(),
                counts,
                &tensor.packed,
            )
            .map_err(invalid)?;
        } else {
            tritium_format::AdditivePayload::new_uniform(
                metadata.layout(),
                metadata.plane_count() as u8,
                &tensor.packed,
            )
            .map_err(invalid)?;
        }
        Ok(tensor)
    }

    fn allocate(metadata: AdditiveMetadata<'_>) -> Result<Self, BackendError> {
        let total = Self::payload_metadata(metadata)?;
        let layout = metadata.layout();
        let planes = metadata.plane_count();
        let codec = codec(layout)?;
        let mut scale_offset = 0usize;
        for plane in 0..planes {
            scale_offset += codec
                .ledger(metadata.plane_trit_count(plane)?)
                .map_err(invalid)?
                .physical_bytes;
        }
        let data_bytes = Self::stream_encoded_bytes(metadata)?;
        let allocation = if let Some(counts) = metadata.tile_counts() {
            let mut owned_counts = Vec::new();
            owned_counts
                .try_reserve_exact(counts.len())
                .map_err(|_| BackendError::OutOfMemory { requested: total })?;
            owned_counts.extend_from_slice(counts);
            let index =
                OwnedPerTileIndex::new(layout, owned_counts).map_err(|error| match error {
                    tritium_core::AdditiveError::AllocationFailed { requested } => {
                        BackendError::OutOfMemory { requested }
                    }
                    other => invalid(other),
                })?;
            let mut starts = Vec::new();
            starts
                .try_reserve_exact(planes)
                .map_err(|_| BackendError::OutOfMemory { requested: total })?;
            let mut byte = 0;
            let mut trit = 0;
            for plane in 0..planes {
                starts.push(PlaneStart { byte, trit });
                let len = metadata.plane_trit_count(plane)?;
                byte += codec.ledger(len).map_err(invalid)?.physical_bytes;
                trit += len;
            }
            Allocation::PerTile { index, starts }
        } else {
            // Only Uniform divides by its validated nonzero plane count.
            let plane_bytes = scale_offset / planes;
            let scale_units = if matches!(layout.law.relation, PlaneRelation::Free) {
                metadata.scale_count() / planes
            } else {
                metadata.scale_count()
            };
            Allocation::Uniform {
                plane_bytes,
                scale_units,
            }
        };
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(data_bytes)
            .map_err(|_| BackendError::OutOfMemory { requested: total })?;
        packed.resize(data_bytes, 0);
        Ok(Self {
            layout,
            planes,
            scale_offset,
            allocation,
            packed,
        })
    }

    pub(super) fn len_bytes(&self) -> usize {
        let auxiliary = match &self.allocation {
            Allocation::Uniform { .. } => 0,
            Allocation::PerTile { index, starts } => {
                index
                    .as_index()
                    .buffer_len_bytes()
                    .expect("checked owned index lengths")
                    + core::mem::size_of_val(starts.as_slice())
            }
        };
        self.packed.len() + auxiliary // The total was checked before any reservation.
    }

    // Only canonical encoders construct these private bytes. Plane offsets reset
    // codec padding; rows do not reset codec units (ragged rows can cross bytes).
    fn trit(&self, plane: usize, coefficient: usize) -> i8 {
        let (bytes, coefficient) = match &self.allocation {
            Allocation::Uniform { plane_bytes, .. } => {
                (plane * plane_bytes..(plane + 1) * plane_bytes, coefficient)
            }
            Allocation::PerTile { starts, .. } => {
                let start = &starts[plane];
                let end = starts
                    .get(plane + 1)
                    .map_or(self.scale_offset, |next| next.byte);
                (start.byte..end, coefficient - start.trit)
            }
        };
        let data = &self.packed[bytes];
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
        let Allocation::Uniform { scale_units, .. } = self.allocation else {
            unreachable!("uniform scale geometry only")
        };
        let index = if matches!(self.layout.law.relation, PlaneRelation::Free) {
            plane * scale_units + unit
        } else {
            unit
        };
        self.stored_scale(plane, index)
    }

    fn stored_scale(&self, plane: usize, index: usize) -> f32 {
        let scales = &self.packed[self.scale_offset..];
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
        if let Allocation::PerTile { index, .. } = &self.allocation {
            let group = usize::from(self.layout.group);
            let tensor_anchor = self.layout.law.anchor == ScaleAnchor::Tensor;
            let mut sum = 0.0;
            for plane in 0..self.planes {
                let mut tensor_partial = 0.0;
                for tile in index
                    .as_index()
                    .tiles_in_row(plane, row)
                    .expect("checked row/plane")
                {
                    let columns = tile.columns();
                    let len = columns.len();
                    let width = if tensor_anchor { len } else { group };
                    for start in (0..len).step_by(width) {
                        let end = (start + width).min(len);
                        let mut partial = if tensor_anchor { tensor_partial } else { 0.0 };
                        for offset in start..end {
                            match self.trit(plane, tile.trit_offset() + offset) {
                                1 => partial += act[columns.start + offset],
                                -1 => partial -= act[columns.start + offset],
                                _ => {}
                            }
                        }
                        if tensor_anchor {
                            tensor_partial = partial;
                        } else {
                            sum += partial
                                * self.stored_scale(plane, tile.scale_offset() + start / group);
                        }
                    }
                }
                if tensor_anchor {
                    sum += tensor_partial * self.stored_scale(plane, plane);
                }
            }
            return sum;
        }
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
                if let Allocation::PerTile { index, .. } = &self.allocation {
                    for tile in index
                        .as_index()
                        .tiles_in_row(plane, row)
                        .expect("checked row/plane")
                    {
                        for (offset, value) in output[tile.columns()].iter_mut().enumerate() {
                            let scale = if self.layout.law.anchor == ScaleAnchor::Tensor {
                                plane
                            } else {
                                tile.scale_offset() + offset / usize::from(self.layout.group)
                            };
                            *value += f32::from(self.trit(plane, tile.trit_offset() + offset))
                                * self.stored_scale(plane, scale);
                        }
                    }
                } else {
                    for (col, value) in output.iter_mut().enumerate() {
                        *value += f32::from(self.trit(plane, row * cols + col))
                            * self.scale(plane, row, col);
                    }
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
