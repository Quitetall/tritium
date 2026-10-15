//! Native canonical plane bytes. Private device indexing is fully accounted.

use super::*;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use half::f16;
use tritium_core::AdditiveView;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_schema::{PlaneCodec, PlaneRelation, ScaleAnchor, ScalePrecision};
use tritium_spec::{AdditiveMetadata, TENSOR_STREAM_CHUNK_BYTES, TensorStreamSource};

const HEADER: usize = 60;
#[cfg(test)]
mod kernel_tests;
#[cfg(test)]
mod stream_tests;

struct Plan {
    header: [u32; 15],
    bytes: usize,
    codec: SaltV2Codec,
}

pub(super) struct PackedAdditive {
    bytes: CudaSlice<u8>,
    ordered_warp_eligible: bool,
}

fn narrow(n: usize) -> Result<u32, BackendError> {
    u32::try_from(n).map_err(|_| invalid("CUDA packed semantic payload/index exceeds u32 ABI"))
}

impl Plan {
    fn ordered_warp_eligible(&self) -> bool {
        // Shape-local heuristic from paired physical diagnostics, not an
        // arithmetic tier or a performance guarantee. Large output grids keep
        // scalar rows: replacing them with 32x more threads regresses them.
        (512..=17408).contains(&self.header[1])
            && self.header[3] == 128
            && self.header[4] <= 1
            && self.header[6] == 1
            && self.header[7] == 0
            && self.header[9] > HEADER as u32
    }
    fn new(view: AdditiveView<'_>) -> Result<Self, BackendError> {
        AdditiveTensor::validate_view(view)
            .map_err(|e| invalid(&format!("CUDA packed semantic input: {e:?}")))?;
        let plan = Self::from_metadata(AdditiveMetadata::from_view(view))?;
        Self::validate_finite(view)?;
        Ok(plan)
    }

    fn from_metadata(metadata: AdditiveMetadata<'_>) -> Result<Self, BackendError> {
        let layout = metadata.layout();
        if !admitted_execution_group(layout.law, layout.group) {
            return Err(BackendError::UnsupportedTensor);
        }
        let codec = match layout.codec {
            PlaneCodec::D2 => SaltV2Codec::D2,
            PlaneCodec::B3 => SaltV2Codec::B3,
            PlaneCodec::S34 => SaltV2Codec::S34,
            _ => return Err(BackendError::UnsupportedTensor),
        };
        let width = match layout.law.precision {
            ScalePrecision::F16 => 2,
            ScalePrecision::F32 => 4,
            _ => return Err(BackendError::UnsupportedTensor),
        };
        let (free, ratio) = match layout.law.relation {
            PlaneRelation::Free => (true, 1.0f32),
            PlaneRelation::Tied { num, den } => (false, num as f32 / den as f32),
            _ => return Err(BackendError::UnsupportedTensor),
        };
        let tensor = match layout.law.anchor {
            ScaleAnchor::Tensor => true,
            ScaleAnchor::Group => false,
            _ => return Err(BackendError::UnsupportedTensor),
        };
        let rows = usize::try_from(layout.rows).map_err(|_| invalid("CUDA rows exceed usize"))?;
        let cols =
            usize::try_from(layout.cols).map_err(|_| invalid("CUDA columns exceed usize"))?;
        if rows != 0 && cols != 0 {
            CudaBackend::check_grad_launch_bounds(1, rows, cols)?;
        }
        let planes = metadata.plane_count();
        let compact = metadata.count_bytes().is_some();
        let mut encoded = 0usize;
        for plane in 0..planes {
            let trits = metadata.plane_trit_count(plane)?;
            narrow(trits)?;
            let size = codec
                .ledger(trits)
                .map_err(|_| invalid("CUDA plane physical size overflow"))?
                .physical_bytes;
            encoded = encoded
                .checked_add(size)
                .ok_or_else(|| invalid("CUDA plane size overflow"))?;
        }
        let scale_offset = HEADER
            .checked_add(encoded)
            .ok_or_else(|| invalid("CUDA scale offset overflow"))?;
        let scale_bytes = product(metadata.scale_count(), width)?;
        let index_offset = scale_offset
            .checked_add(scale_bytes)
            .ok_or_else(|| invalid("CUDA index offset overflow"))?;
        let row_slots = product(
            planes,
            rows.checked_add(1)
                .ok_or_else(|| invalid("CUDA row count overflow"))?,
        )?;
        let index_bytes = if compact {
            product(
                planes
                    .checked_add(product(2, row_slots)?)
                    .ok_or_else(|| invalid("CUDA index size overflow"))?,
                4,
            )?
        } else {
            0
        };
        let counts_offset = index_offset
            .checked_add(index_bytes)
            .ok_or_else(|| invalid("CUDA counts offset overflow"))?;
        let bytes = counts_offset
            .checked_add(metadata.count_bytes().unwrap_or(0))
            .ok_or_else(|| invalid("CUDA payload byte count overflow"))?;
        narrow(bytes)?;
        let units = if compact {
            0
        } else if free {
            metadata.scale_count().checked_div(planes).unwrap_or(0)
        } else {
            metadata.scale_count()
        };
        Ok(Self {
            codec,
            bytes,
            header: [
                narrow(rows)?,
                narrow(cols)?,
                narrow(planes)?,
                u32::from(layout.group),
                match codec {
                    SaltV2Codec::D2 => 0,
                    SaltV2Codec::B3 => 1,
                    SaltV2Codec::S34 => 2,
                    _ => return Err(BackendError::UnsupportedTensor),
                },
                narrow(width)?,
                u32::from(free),
                u32::from(tensor),
                ratio.to_bits(),
                narrow(scale_offset)?,
                narrow(index_offset)?,
                narrow(counts_offset)?,
                if compact {
                    u32::from(layout.tile)
                } else {
                    narrow(encoded.checked_div(planes).unwrap_or(0))?
                },
                u32::from(compact),
                narrow(units)?,
            ],
        })
    }

    fn allocate(&self, metadata: AdditiveMetadata<'_>) -> Result<Vec<u8>, BackendError> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.bytes)
            .map_err(|_| BackendError::OutOfMemory {
                requested: self.bytes,
            })?;
        bytes.resize(self.bytes, 0);
        for (i, value) in self.header.iter().enumerate() {
            bytes[4 * i..4 * i + 4].copy_from_slice(&value.to_le_bytes());
        }
        if let Some(counts) = metadata.tile_counts() {
            let rows = self.header[0] as usize;
            let cols = self.header[1] as usize;
            let planes = self.header[2] as usize;
            let group = self.header[3] as usize;
            let free = self.header[6] != 0;
            let tensor = self.header[7] != 0;
            let index = self.header[10] as usize;
            let slots = planes * (rows + 1);
            let tile_width = self.header[12] as usize;
            let tiles = cols.div_ceil(tile_width);
            let mut at = HEADER;
            let mut scale = 0;
            for plane in 0..planes {
                if !free {
                    scale = 0;
                }
                put_u32(&mut bytes, index + 4 * plane, at)?;
                let mut coefficient = 0;
                for row in 0..rows {
                    let slot = plane * (rows + 1) + row;
                    put_u32(&mut bytes, index + 4 * (planes + slot), coefficient)?;
                    put_u32(
                        &mut bytes,
                        index + 4 * (planes + slots + slot),
                        if tensor { plane } else { scale },
                    )?;
                    for tile in 0..tiles {
                        let len = (cols - tile * tile_width).min(tile_width);
                        let count = usize::from(counts[row * tiles + tile]);
                        if count > plane {
                            coefficient += len;
                        }
                        if !tensor && if free { count > plane } else { count > 0 } {
                            scale += len.div_ceil(group);
                        }
                    }
                }
                let trits = metadata.plane_trit_count(plane)?;
                if coefficient != trits {
                    return Err(invalid("CUDA compact coefficient index mismatch"));
                }
                let end = plane * (rows + 1) + rows;
                put_u32(&mut bytes, index + 4 * (planes + end), coefficient)?;
                put_u32(
                    &mut bytes,
                    index + 4 * (planes + slots + end),
                    if tensor { plane } else { scale },
                )?;
                at += self
                    .codec
                    .ledger(trits)
                    .map_err(|_| invalid("CUDA codec ledger failed"))?
                    .physical_bytes;
            }
            if !tensor && scale != metadata.scale_count() {
                return Err(invalid("CUDA compact scale index mismatch"));
            }
            bytes[self.header[11] as usize..].copy_from_slice(counts);
        }
        Ok(bytes)
    }

    // Suffix in physical order; only scalar prefix state, no full row-index Vec.
    // The legacy allocated oracle remains independent for exact byte checks.
    fn visit_suffix(
        &self,
        metadata: AdditiveMetadata<'_>,
        mut emit: impl FnMut(&[u8]) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        let Some(counts) = metadata.tile_counts() else {
            return Ok(());
        };
        let rows = self.header[0] as usize;
        let cols = self.header[1] as usize;
        let planes = self.header[2] as usize;
        let group = self.header[3] as usize;
        let free = self.header[6] != 0;
        let tensor = self.header[7] != 0;
        let tile_width = self.header[12] as usize;
        let tiles = cols.div_ceil(tile_width);
        let mut at = HEADER;
        for plane in 0..planes {
            emit(&narrow(at)?.to_le_bytes())?;
            at += self
                .codec
                .ledger(metadata.plane_trit_count(plane)?)
                .map_err(|_| invalid("CUDA codec ledger failed"))?
                .physical_bytes;
        }
        for scale_indices in [false, true] {
            let mut scale = 0usize;
            for plane in 0..planes {
                if !free {
                    scale = 0;
                }
                let mut coefficient = 0usize;
                for row in 0..=rows {
                    let value = if scale_indices {
                        if tensor { plane } else { scale }
                    } else {
                        coefficient
                    };
                    emit(&narrow(value)?.to_le_bytes())?;
                    if row == rows {
                        break;
                    }
                    for tile in 0..tiles {
                        let len = (cols - tile * tile_width).min(tile_width);
                        let count = usize::from(counts[row * tiles + tile]);
                        if count > plane {
                            coefficient += len;
                        }
                        if !tensor && if free { count > plane } else { count > 0 } {
                            scale += len.div_ceil(group);
                        }
                    }
                }
                if coefficient != metadata.plane_trit_count(plane)? {
                    return Err(invalid("CUDA compact coefficient index mismatch"));
                }
            }
            if !tensor && scale != metadata.scale_count() {
                return Err(invalid("CUDA compact scale index mismatch"));
            }
        }
        emit(counts)
    }

    fn validate_finite(view: AdditiveView<'_>) -> Result<(), BackendError> {
        let layout = view.layout();
        let (free, ratio) = match layout.law.relation {
            PlaneRelation::Free => (true, 1.0f32),
            PlaneRelation::Tied { num, den } => (false, num as f32 / den as f32),
            _ => return Err(BackendError::UnsupportedTensor),
        };
        let max_scale = view.scales().iter().copied().fold(0.0f32, f32::max);
        let mut bound = 0.0f32;
        let mut plane_bound = max_scale;
        for _ in 0..view.plane_count() {
            bound += plane_bound;
            if !free {
                plane_bound *= ratio;
            }
        }
        if !bound.is_finite() {
            let (rows, cols) = (layout.rows as usize, layout.cols as usize);
            let requested = product(cols, 4)?;
            let mut row = Vec::new();
            row.try_reserve_exact(cols)
                .map_err(|_| BackendError::OutOfMemory { requested })?;
            row.resize(cols, 0.0);
            for index in 0..rows {
                view.dequant_row_into(index, &mut row)
                    .map_err(|_| invalid("CUDA packed semantic row validation failed"))?;
                if row.iter().any(|value| !value.is_finite()) {
                    return Err(invalid("CUDA packed semantic decoded weight is nonfinite"));
                }
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn scale(&self, bytes: &[u8], plane: usize, index: usize) -> f32 {
        let at = self.header[9] as usize + index * self.header[5] as usize;
        let mut value = if self.header[5] == 2 {
            f16::from_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]])).to_f32()
        } else {
            f32::from_le_bytes(bytes[at..at + 4].try_into().expect("private planned scale"))
        };
        if self.header[6] == 0 {
            let ratio = f32::from_bits(self.header[8]);
            for _ in 0..plane {
                value *= ratio;
            }
        }
        value
    }
    #[cfg(test)]
    fn digit(&self, bytes: &[u8], start: usize, coefficient: usize) -> f32 {
        let data = &bytes[start..self.header[9] as usize];
        match self.codec {
            SaltV2Codec::D2 => {
                f32::from(((data[coefficient / 4] >> (2 * (coefficient % 4))) & 3) as i8 - 1)
            }
            SaltV2Codec::B3 => {
                let divisor = [1, 3, 9, 27, 81][coefficient % 5];
                f32::from(((data[coefficient / 5] / divisor) % 3) as i8 - 1)
            }
            SaltV2Codec::S34 => {
                let bit = coefficient / 4 * 5;
                let at = bit / 8;
                let shift = bit % 8;
                let mut word = u16::from(data[at]);
                if shift > 3 {
                    word |= u16::from(data[at + 1]) << 8;
                }
                let code = ((word >> shift) & 31) as usize;
                let lane = coefficient % 4;
                let zero = code & 3;
                if lane == zero {
                    0.
                } else {
                    let sign = lane - usize::from(lane > zero);
                    if code & (1 << (2 + sign)) == 0 {
                        -1.
                    } else {
                        1.
                    }
                }
            }
            _ => unreachable!("private planner admits only canonical codecs"),
        }
    }
    #[cfg(test)]
    fn validate_encoded_finite(&self, bytes: &[u8]) -> Result<(), BackendError> {
        let width = self.header[5] as usize;
        let scales = (self.header[10] - self.header[9]) as usize / width;
        let maximum = (0..scales)
            .map(|i| self.scale(bytes, 0, i))
            .fold(0., f32::max);
        let planes = self.header[2] as usize;
        let ratio = f32::from_bits(self.header[8]);
        let mut bound = 0.;
        let mut component = maximum;
        for _ in 0..planes {
            bound += component;
            if self.header[6] == 0 {
                component *= ratio;
            }
        }
        if bound.is_finite() {
            return Ok(());
        }
        self.visit_weights(bytes, |_, _, value| {
            if value.is_finite() {
                Ok(())
            } else {
                Err(invalid("CUDA packed semantic decoded weight is nonfinite"))
            }
        })
    }
    #[cfg(test)]
    fn visit_weights(
        &self,
        bytes: &[u8],
        mut visit: impl FnMut(usize, usize, f32) -> Result<(), BackendError>,
    ) -> Result<(), BackendError> {
        let planes = self.header[2] as usize;
        let rows = self.header[0] as usize;
        let cols = self.header[1] as usize;
        let group = self.header[3] as usize;
        let compact = self.header[13] != 0;
        let tensor = self.header[7] != 0;
        let free = self.header[6] != 0;
        let index = self.header[10] as usize;
        let slots = planes * (rows + 1);
        // Plane counts are u8. Fixed cursors avoid a complete decoded row/master.
        let mut cursors = [[0usize; 3]; u8::MAX as usize];
        for row in 0..rows {
            for (plane, cursor) in cursors[..planes].iter_mut().enumerate() {
                *cursor = if compact {
                    let slot = plane * (rows + 1) + row;
                    [
                        read_u32(bytes, index + 4 * plane),
                        read_u32(bytes, index + 4 * (planes + slot)),
                        read_u32(bytes, index + 4 * (planes + slots + slot)),
                    ]
                } else {
                    [
                        HEADER + plane * self.header[12] as usize,
                        row * cols,
                        if free {
                            plane * self.header[14] as usize
                        } else {
                            0
                        },
                    ]
                };
            }
            for col in 0..cols {
                let mut sum = 0.;
                let tile_width = if compact {
                    self.header[12] as usize
                } else {
                    cols
                };
                let tiles = cols.div_ceil(tile_width);
                let tile = col / tile_width;
                let offset = col % tile_width;
                let len = (cols - tile * tile_width).min(tile_width);
                let count = if compact {
                    usize::from(bytes[self.header[11] as usize + row * tiles + tile])
                } else {
                    planes
                };
                for (plane, cursor) in cursors[..planes].iter_mut().enumerate() {
                    if count > plane {
                        let stored = if compact {
                            cursor[2] + if tensor { 0 } else { offset / group }
                        } else {
                            cursor[2]
                                + if tensor {
                                    0
                                } else {
                                    row * cols.div_ceil(group) + col / group
                                }
                        };
                        sum += self.digit(bytes, cursor[0], cursor[1] + offset)
                            * self.scale(bytes, plane, stored);
                    }
                    if compact && offset + 1 == len {
                        if count > plane {
                            cursor[1] += len;
                        }
                        if !tensor && if free { count > plane } else { count > 0 } {
                            cursor[2] += len.div_ceil(group);
                        }
                    }
                }
                visit(row, col, sum)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
fn read_u32(bytes: &[u8], at: usize) -> usize {
    u32::from_le_bytes(
        bytes[at..at + 4]
            .try_into()
            .expect("private planned native index"),
    ) as usize
}

fn put_u32(bytes: &mut [u8], at: usize, value: usize) -> Result<(), BackendError> {
    bytes[at..at + 4].copy_from_slice(&narrow(value)?.to_le_bytes());
    Ok(())
}

impl PackedAdditive {
    pub(super) fn stream_metadata(
        metadata: AdditiveMetadata<'_>,
    ) -> Result<(usize, usize, usize), BackendError> {
        if !matches!(metadata.layout().transport, tritium_schema::Transport::Raw) {
            return Err(BackendError::UnsupportedTensor);
        }
        let plan = Plan::from_metadata(metadata)?;
        let workspace = 3 * TENSOR_STREAM_CHUNK_BYTES;
        Ok((plan.bytes, workspace, plan.header[10] as usize - HEADER))
    }
    #[cfg(test)]
    fn stage_stream(
        metadata: AdditiveMetadata<'_>,
        source: &dyn TensorStreamSource,
    ) -> Result<Vec<u8>, BackendError> {
        let (_, _, expected) = Self::stream_metadata(metadata)?;
        if source.encoded_bytes() != expected as u64 {
            return Err(invalid("CUDA stream encoded length mismatch"));
        }
        let plan = Plan::from_metadata(metadata)?;
        let mut validator = if let Some(counts) = metadata.tile_counts() {
            tritium_format::AdditiveStreamValidator::new_per_tile(
                metadata.layout(),
                counts,
                expected,
            )
        } else {
            tritium_format::AdditiveStreamValidator::new_uniform(
                metadata.layout(),
                metadata.plane_count() as u8,
                expected,
            )
        }
        .map_err(|e| invalid(&format!("CUDA canonical source: {e:?}")))?;
        let mut bytes = plan.allocate(metadata)?;
        let mut at = HEADER;
        let end = plan.header[10] as usize;
        source.visit(&mut |chunk| {
            if chunk.len() > TENSOR_STREAM_CHUNK_BYTES {
                return Err(invalid("CUDA source window exceeds stream bound"));
            }
            let next = at
                .checked_add(chunk.len())
                .ok_or_else(|| invalid("CUDA source size overflows"))?;
            if next > end {
                return Err(invalid("CUDA source exceeds stream plan"));
            }
            validator
                .push(chunk)
                .map_err(|e| invalid(&format!("CUDA canonical source: {e:?}")))?;
            bytes[at..next].copy_from_slice(chunk);
            at = next;
            Ok(())
        })?;
        if at != end {
            return Err(invalid("CUDA source shorter than stream plan"));
        }
        validator
            .finish()
            .map_err(|e| invalid(&format!("CUDA canonical source: {e:?}")))?;
        plan.validate_encoded_finite(&bytes)?;
        Ok(bytes)
    }
    pub(super) fn upload_stream(
        backend: &CudaBackend,
        metadata: AdditiveMetadata<'_>,
        source: &dyn TensorStreamSource,
    ) -> Result<Self, BackendError> {
        let (_, _, expected) = Self::stream_metadata(metadata)?;
        if source.encoded_bytes() != expected as u64 {
            return Err(invalid("CUDA stream encoded length mismatch"));
        }
        let plan = Plan::from_metadata(metadata)?;
        let mut validator = if let Some(counts) = metadata.tile_counts() {
            tritium_format::AdditiveStreamValidator::new_per_tile(
                metadata.layout(),
                counts,
                expected,
            )
        } else {
            tritium_format::AdditiveStreamValidator::new_uniform(
                metadata.layout(),
                metadata.plane_count() as u8,
                expected,
            )
        }
        .map_err(|e| invalid(&format!("CUDA canonical source: {e:?}")))?;
        let mut bytes = backend.stream.alloc_zeros::<u8>(plan.bytes).map_err(|e| {
            alloc_or_backend("allocate private CUDA stream payload", &e, plan.bytes)
        })?;
        {
            let mut writer = super::stream_copy::ByteWriter::new(backend, &mut bytes, 0)?;
            for word in plan.header {
                writer.push(&word.to_le_bytes())?;
            }
            super::stream_copy::visit(source, expected, |chunk| {
                validator
                    .push(chunk)
                    .map_err(|e| invalid(&format!("CUDA canonical source: {e:?}")))?;
                writer.push(chunk)
            })?;
            writer.finish(plan.header[10] as usize)?;
        }
        let maximum = validator
            .finish()
            .map_err(|e| invalid(&format!("CUDA canonical source: {e:?}")))?;
        {
            let mut writer =
                super::stream_copy::ByteWriter::new(backend, &mut bytes, plan.header[10] as usize)?;
            plan.visit_suffix(metadata, |part| writer.push(part))?;
            writer.finish(plan.bytes)?;
        }
        let result = Self {
            bytes,
            ordered_warp_eligible: plan.ordered_warp_eligible(),
        };
        result.validate_device_finite(backend, &plan, maximum)?;
        backend
            .stream
            .synchronize()
            .map_err(|e| driver_err("finish private CUDA stream upload", &e))?;
        backend
            .stream
            .context()
            .check_err()
            .map_err(|e| driver_err("check packed pinned cleanup", &e))?;
        Ok(result)
    }
    pub(super) fn plan_metadata(
        metadata: AdditiveMetadata<'_>,
    ) -> Result<(usize, usize), BackendError> {
        let plan = Plan::from_metadata(metadata)?;
        let row = product(metadata.layout().cols as usize, 4)?;
        Ok((plan.bytes, plan.bytes.max(row)))
    }
    pub(super) fn payload_bytes(view: AdditiveView<'_>) -> Result<usize, BackendError> {
        Ok(Plan::new(view)?.bytes)
    }

    pub(super) fn upload(
        backend: &CudaBackend,
        view: AdditiveView<'_>,
    ) -> Result<Self, BackendError> {
        let (bytes, ordered_warp_eligible) = Self::stage_view(view)?;
        Self::transfer(backend, &bytes, ordered_warp_eligible)
    }

    fn stage_view(view: AdditiveView<'_>) -> Result<(Vec<u8>, bool), BackendError> {
        let plan = Plan::new(view)?;
        let mut bytes = plan.allocate(AdditiveMetadata::from_view(view))?;
        let mut at = HEADER;
        let planes = plan.header[2] as usize;
        for plane in 0..planes {
            let trits = view
                .plane_trits(plane)
                .map_err(|_| invalid("CUDA plane geometry mismatch"))?;
            let len = plan
                .codec
                .ledger(trits.len())
                .map_err(|_| invalid("CUDA codec ledger failed"))?
                .physical_bytes;
            plan.codec
                .pack_into(trits, &mut bytes[at..at + len])
                .map_err(|_| invalid("CUDA canonical plane encoding failed"))?;
            at += len;
        }
        let width = plan.header[5] as usize;
        for &value in view.scales() {
            if width == 2 {
                bytes[at..at + 2].copy_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
            } else {
                bytes[at..at + 4].copy_from_slice(&value.to_bits().to_le_bytes());
            }
            at += width;
        }
        Ok((bytes, plan.ordered_warp_eligible()))
    }

    fn transfer(
        backend: &CudaBackend,
        bytes: &[u8],
        ordered_warp_eligible: bool,
    ) -> Result<Self, BackendError> {
        let bytes = backend
            .stream
            .clone_htod(bytes)
            .map_err(|e| alloc_or_backend("upload CUDA packed semantic tensor", &e, bytes.len()))?;
        Ok(Self {
            bytes,
            ordered_warp_eligible,
        })
    }

    pub(super) fn len_bytes(&self) -> usize {
        self.bytes.len()
    }

    fn validate_device_finite(
        &self,
        backend: &CudaBackend,
        plan: &Plan,
        maximum: f32,
    ) -> Result<(), BackendError> {
        let mut bound = 0.;
        let mut value = maximum;
        for _ in 0..plan.header[2] {
            bound += value;
            if plan.header[6] == 0 {
                value *= f32::from_bits(plan.header[8]);
            }
        }
        if bound.is_finite() {
            return Ok(());
        }
        let total = product(plan.header[0] as usize, plan.header[1] as usize)?;
        if total == 0 {
            return Ok(());
        }
        let mut flag = backend
            .stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| alloc_or_backend("allocate CUDA finite-check flag", &e, 4))?;
        let mut launch = backend
            .stream
            .launch_builder(&backend.func_semantic_additive_validate_finite);
        launch.arg(&self.bytes).arg(&mut flag);
        #[allow(unsafe_code)]
        // SAFETY: generated metadata and all canonical bytes are checked;
        // grid-stride reads use exact logical geometry and one private flag.
        unsafe { launch.launch(LaunchConfig::for_num_elems(narrow(total.min(65536))?)) }
            .map_err(|e| driver_err("validate CUDA stored finite weights", &e))?;
        drop(launch);
        let mut result = [0u32];
        backend
            .stream
            .memcpy_dtoh(&flag, &mut result)
            .map_err(|e| driver_err("read CUDA finite-check flag", &e))?;
        if result[0] != 0 {
            return Err(invalid("CUDA packed semantic decoded weight is nonfinite"));
        }
        Ok(())
    }

    pub(super) fn matmul(
        &self,
        backend: &CudaBackend,
        p: TensorMatmul<'_>,
        rows: usize,
        cols: usize,
        basis: Basis,
    ) -> Result<(), BackendError> {
        let inputs = product(p.batch, cols)?;
        let outputs = product(p.batch, rows)?;
        if p.act.len() != inputs || p.transformed_act.len() != inputs || p.out.len() != outputs {
            return Err(invalid("CUDA packed semantic matmul shape mismatch"));
        }
        if outputs != 0 && cols != 0 {
            CudaBackend::check_grad_launch_bounds(p.batch, rows, cols)?;
        }
        if outputs == 0 || cols == 0 {
            p.transformed_act.copy_from_slice(p.act);
            p.out.fill(0.0);
            return Ok(());
        }
        let batch = narrow(p.batch)?;
        let mut input = backend
            .stream
            .clone_htod(p.act)
            .map_err(|e| driver_err("upload packed semantic activations", &e))?;
        super::basis::apply(backend, &mut input, cols, basis, false)?;
        if basis == Basis::Identity {
            p.transformed_act.copy_from_slice(p.act);
        } else {
            // This explicit host-slice contract observes transformed scratch.
            // A future graph-resident caller can retain this same device input.
            backend
                .stream
                .memcpy_dtoh(&input, p.transformed_act)
                .map_err(|e| driver_err("read packed semantic basis scratch", &e))?;
        }
        let requested = product(outputs, 4)?;
        let mut output = backend
            .stream
            .alloc_zeros::<f32>(outputs)
            .map_err(|e| alloc_or_backend("allocate packed semantic output", &e, requested))?;
        self.launch(backend, &input, &mut output, batch, outputs)?;
        backend
            .stream
            .memcpy_dtoh(&output, p.out)
            .map_err(|e| driver_err("read packed semantic output", &e))
    }

    pub(super) fn launch(
        &self,
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        batch: u32,
        outputs: usize,
    ) -> Result<(), BackendError> {
        if outputs > output.len() {
            return Err(invalid("packed active output extent exceeds workspace"));
        }
        let cooperative = self.ordered_warp_eligible && (512..=8192).contains(&outputs);
        let (function, cfg) = if cooperative {
            (
                &backend.func_semantic_additive_matmul_ordered_warp,
                LaunchConfig {
                    grid_dim: (narrow(outputs.div_ceil(4))?, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                },
            )
        } else {
            (
                &backend.func_semantic_additive_matmul,
                LaunchConfig::for_num_elems(narrow(outputs)?),
            )
        };
        let mut launch = backend.stream.launch_builder(function);
        launch.arg(&self.bytes).arg(input).arg(output).arg(&batch);
        #[allow(unsafe_code)]
        // SAFETY: validated geometry bounds every access. A scalar thread or
        // complete warp owns each output; all private indexing was checked at
        // upload. Cooperative early exits are uniform within every warp.
        unsafe { launch.launch(cfg) }
            .map_err(|e| driver_err("launch packed semantic matmul", &e))?;
        Ok(())
    }

    pub(super) fn gather_device(
        &self,
        backend: &CudaBackend,
        ids: &CudaSlice<u32>,
        output: &mut CudaSlice<f32>,
        batch: usize,
        outputs: usize,
    ) -> Result<(), BackendError> {
        let selected = narrow(batch)?;
        if outputs > output.len() {
            return Err(invalid("packed active gather extent exceeds workspace"));
        }
        let cfg = LaunchConfig::for_num_elems(narrow(outputs)?);
        let mut launch = backend
            .stream
            .launch_builder(&backend.func_semantic_additive_gather);
        launch.arg(&self.bytes).arg(ids).arg(output).arg(&selected);
        #[allow(unsafe_code)]
        // SAFETY: the graph validates row IDs and allocation/ABI extents before launch.
        unsafe { launch.launch(cfg) }.map_err(|e| driver_err("resident packed embedding", &e))?;
        Ok(())
    }

    pub(super) fn embed(
        &self,
        backend: &CudaBackend,
        ids: &[usize],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        basis: Basis,
    ) -> Result<(), BackendError> {
        let elements = product(ids.len(), cols)?;
        if out.len() != elements || ids.iter().any(|&row| row >= rows) {
            return Err(invalid(
                "CUDA packed semantic gather shape or row ID mismatch",
            ));
        }
        if elements == 0 {
            return Ok(());
        }
        let selected = narrow(ids.len())?;
        let count = narrow(elements)?;
        let requested = product(ids.len(), 4)?;
        let mut host_ids = Vec::new();
        host_ids
            .try_reserve_exact(ids.len())
            .map_err(|_| BackendError::OutOfMemory { requested })?;
        for &id in ids {
            host_ids.push(narrow(id)?);
        }
        let ids = backend
            .stream
            .clone_htod(&host_ids)
            .map_err(|e| driver_err("upload packed gather IDs", &e))?;
        let mut output = backend
            .stream
            .alloc_zeros::<f32>(elements)
            .map_err(|e| alloc_or_backend("allocate packed gather output", &e, elements * 4))?;
        let cfg = LaunchConfig::for_num_elems(count);
        let mut launch = backend
            .stream
            .launch_builder(&backend.func_semantic_additive_gather);
        launch
            .arg(&self.bytes)
            .arg(&ids)
            .arg(&mut output)
            .arg(&selected);
        #[allow(unsafe_code)]
        // SAFETY: row IDs, output and private packed indexing are validated;
        // each thread writes one selected-row lane, never the weight payload.
        unsafe { launch.launch(cfg) }
            .map_err(|e| driver_err("launch packed semantic gather", &e))?;
        drop(launch);
        super::basis::apply(backend, &mut output, cols, basis, true)?;
        backend
            .stream
            .memcpy_dtoh(&output, out)
            .map_err(|e| driver_err("read packed gather output", &e))?;
        Ok(())
    }
}
