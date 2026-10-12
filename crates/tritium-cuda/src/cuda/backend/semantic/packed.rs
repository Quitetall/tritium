//! Native canonical plane bytes. Private device indexing is fully accounted.

use super::*;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use half::f16;
use tritium_core::AdditiveView;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_schema::{PlaneCodec, PlaneRelation, ScaleAnchor, ScalePrecision};

const HEADER: usize = 60;

struct Plan {
    header: [u32; 15],
    bytes: usize,
    codec: SaltV2Codec,
}

pub(super) struct PackedAdditive {
    bytes: CudaSlice<u8>,
}

fn narrow(n: usize) -> Result<u32, BackendError> {
    u32::try_from(n).map_err(|_| invalid("CUDA packed semantic payload/index exceeds u32 ABI"))
}

impl Plan {
    fn new(view: AdditiveView<'_>) -> Result<Self, BackendError> {
        AdditiveTensor::validate_view(view)
            .map_err(|e| invalid(&format!("CUDA packed semantic input: {e:?}")))?;
        let layout = view.layout();
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
        let planes = view.plane_count();
        // Retain the previous adapter's rejection of nonfinite decoded weights
        // without materializing a matrix. Most inputs prove this from a scale
        // upper bound; only overflow-risk inputs need one reusable row.
        let max_scale = view.scales().iter().copied().fold(0.0f32, f32::max);
        let mut bound = 0.0f32;
        let mut plane_bound = max_scale;
        for _ in 0..planes {
            bound += plane_bound;
            if !free {
                plane_bound *= ratio;
            }
        }
        if !bound.is_finite() {
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
        let compact = view.tile_counts().is_some();
        let mut encoded = 0usize;
        for plane in 0..planes {
            let trits = view
                .plane_trits(plane)
                .map_err(|_| invalid("CUDA plane geometry mismatch"))?;
            narrow(trits.len())?;
            let size = codec
                .ledger(trits.len())
                .map_err(|_| invalid("CUDA plane physical size overflow"))?
                .physical_bytes;
            encoded = encoded
                .checked_add(size)
                .ok_or_else(|| invalid("CUDA plane size overflow"))?;
        }
        let scale_offset = HEADER
            .checked_add(encoded)
            .ok_or_else(|| invalid("CUDA scale offset overflow"))?;
        let scale_bytes = product(view.scales().len(), width)?;
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
            .checked_add(view.tile_counts().map_or(0, |c| c.len()))
            .ok_or_else(|| invalid("CUDA payload byte count overflow"))?;
        narrow(bytes)?;
        let units = if compact {
            0
        } else if free {
            view.scales().len().checked_div(planes).unwrap_or(0)
        } else {
            view.scales().len()
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
}

fn put_u32(bytes: &mut [u8], at: usize, value: usize) -> Result<(), BackendError> {
    bytes[at..at + 4].copy_from_slice(&narrow(value)?.to_le_bytes());
    Ok(())
}

impl PackedAdditive {
    pub(super) fn payload_bytes(view: AdditiveView<'_>) -> Result<usize, BackendError> {
        Ok(Plan::new(view)?.bytes)
    }

    pub(super) fn upload(
        backend: &CudaBackend,
        view: AdditiveView<'_>,
    ) -> Result<Self, BackendError> {
        let plan = Plan::new(view)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(plan.bytes)
            .map_err(|_| BackendError::OutOfMemory {
                requested: plan.bytes,
            })?;
        bytes.resize(plan.bytes, 0);
        for (i, value) in plan.header.iter().enumerate() {
            bytes[4 * i..4 * i + 4].copy_from_slice(&value.to_le_bytes());
        }
        let mut at = HEADER;
        let rows = plan.header[0] as usize;
        let cols = plan.header[1] as usize;
        let planes = plan.header[2] as usize;
        let group = plan.header[3] as usize;
        let free = plan.header[6] != 0;
        let tensor = plan.header[7] != 0;
        let index = plan.header[10] as usize;
        let slots = planes * (rows + 1);
        let mut scale = 0usize;
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
            if let Some(counts) = view.tile_counts() {
                if !free {
                    scale = 0;
                }
                put_u32(&mut bytes, index + 4 * plane, at)?;
                let mut coefficient = 0usize;
                let tile_width = plan.header[12] as usize;
                let tiles = cols.div_ceil(tile_width);
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
                        if usize::from(counts[row * tiles + tile]) > plane {
                            coefficient += len;
                        }
                        if !tensor
                            && if free {
                                usize::from(counts[row * tiles + tile]) > plane
                            } else {
                                counts[row * tiles + tile] > 0
                            }
                        {
                            scale += len.div_ceil(group);
                        }
                    }
                }
                if coefficient != trits.len() {
                    return Err(invalid("CUDA compact coefficient index mismatch"));
                }
                let end = plane * (rows + 1) + rows;
                put_u32(&mut bytes, index + 4 * (planes + end), coefficient)?;
                put_u32(
                    &mut bytes,
                    index + 4 * (planes + slots + end),
                    if tensor { plane } else { scale },
                )?;
            }
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
        if let Some(counts) = view.tile_counts() {
            if !tensor && scale != view.scales().len() {
                return Err(invalid("CUDA compact scale index mismatch"));
            }
            bytes[plan.header[11] as usize..].copy_from_slice(counts);
        }
        let bytes = backend
            .stream
            .clone_htod(&bytes)
            .map_err(|e| alloc_or_backend("upload CUDA packed semantic tensor", &e, plan.bytes))?;
        Ok(Self { bytes })
    }

    pub(super) fn len_bytes(&self) -> usize {
        self.bytes.len()
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
        let cfg = LaunchConfig::for_num_elems(narrow(outputs)?);
        let mut launch = backend
            .stream
            .launch_builder(&backend.func_semantic_additive_matmul);
        launch
            .arg(&self.bytes)
            .arg(&input)
            .arg(&mut output)
            .arg(&batch);
        #[allow(unsafe_code)]
        // SAFETY: validated geometry bounds every access; one thread owns each
        // output and private canonical bytes/indexing were checked at upload.
        unsafe { launch.launch(cfg) }
            .map_err(|e| driver_err("launch packed semantic matmul", &e))?;
        backend
            .stream
            .memcpy_dtoh(&output, p.out)
            .map_err(|e| driver_err("read packed semantic output", &e))
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
