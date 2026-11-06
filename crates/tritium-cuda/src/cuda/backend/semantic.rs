//! Device-owned dense and native packed additive execution.
//!
//! Additive plane/scale bytes stay packed; no dense weight shadow is created.
//! Matmul transforms activations and gather undoes that basis using shared core.
//! Existing packed/resident kernels are unchanged. No host shadow is retained.

use super::{CudaBackend, alloc_or_backend, driver_err};
use core::any::Any;
use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};
use std::sync::Arc;
use tritium_core::{GemmShape, apply_basis, apply_inverse_basis};
use tritium_format::AdditiveTensor;
use tritium_spec::{
    BackendError, Basis, DeviceBuffer, TENSOR_STREAM_CHUNK_BYTES, TensorCaps, TensorExecution,
    TensorMatmul, TensorMetadata, TensorStreamEncoding, TensorStreamPlan, TensorStreamSource,
    TensorUploadPlan, TensorView, admitted_execution_group,
};
mod basis;
mod packed;
mod stream_copy;
pub(super) mod swiglu;
use packed::PackedAdditive;
pub(super) use swiglu::Workspace as SwigluWorkspace;

pub(super) struct CudaTensor {
    rows: usize,
    cols: usize,
    basis: Basis,
    values: Option<CudaSlice<f32>>,
    packed: Option<PackedAdditive>,
    // Retain the context and identify the exact owning backend stream. A device
    // ordinal/string alone cannot prove that a handle belongs to this adapter.
    owner: Arc<CudaStream>,
}

impl DeviceBuffer for CudaTensor {
    fn len_bytes(&self) -> usize {
        if let Some(packed) = &self.packed {
            return packed.len_bytes();
        }
        self.values
            .as_ref()
            .map_or(0, |values| values.len() * core::mem::size_of::<f32>())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}

fn product(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("CUDA semantic geometry overflows"))
}

impl CudaTensor {
    pub(super) fn stream_plan(
        metadata: TensorMetadata<'_>,
        encoding: TensorStreamEncoding,
    ) -> Result<Option<TensorStreamPlan>, BackendError> {
        let (bytes, workspace, encoded) = match (metadata, encoding) {
            (TensorMetadata::Additive(a), TensorStreamEncoding::CanonicalRawAdditive) => {
                match PackedAdditive::stream_metadata(a) {
                    Ok(p) => p,
                    Err(BackendError::UnsupportedTensor) => return Ok(None),
                    Err(e) => return Err(e),
                }
            }
            (
                TensorMetadata::Dense { rows, cols },
                TensorStreamEncoding::DenseLittleEndian(precision),
            ) => {
                if !matches!(
                    precision,
                    tritium_schema::ScalePrecision::F16
                        | tritium_schema::ScalePrecision::Bf16
                        | tritium_schema::ScalePrecision::F32
                ) {
                    return Ok(None);
                }
                let bytes = TensorMetadata::dense_bytes(rows, cols)?;
                if rows != 0 && cols != 0 {
                    CudaBackend::check_grad_launch_bounds(1, rows, cols)?;
                }
                let encoded =
                    tritium_format::trit_model::DensePayloadDecoder::byte_len(precision, bytes / 4)
                        .map_err(|e| invalid(&e.to_string()))?;
                let workspace = 6 * TENSOR_STREAM_CHUNK_BYTES;
                (bytes, workspace, encoded)
            }
            _ => return Ok(None),
        };
        Ok(Some(TensorStreamPlan {
            upload: TensorUploadPlan {
                caps: TensorCaps {
                    execution: TensorExecution::Native,
                    payload_bytes: bytes as u64,
                },
                host_workspace_bytes: workspace as u64,
                device_workspace_bytes: if matches!(metadata, TensorMetadata::Additive(_)) {
                    4
                } else {
                    0
                },
            },
            encoded_bytes: encoded as u64,
        }))
    }
    pub(super) fn upload_stream(
        backend: &CudaBackend,
        source: &dyn TensorStreamSource,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let metadata = source.metadata();
        let plan = Self::stream_plan(metadata, source.encoding())?
            .ok_or(BackendError::UnsupportedTensor)?;
        if source.encoded_bytes() != plan.encoded_bytes {
            return Err(invalid("CUDA stream encoded length differs from geometry"));
        }
        match (metadata, source.encoding()) {
            (TensorMetadata::Additive(a), TensorStreamEncoding::CanonicalRawAdditive) => {
                let layout = a.layout();
                Ok(Box::new(Self {
                    rows: layout.rows as usize,
                    cols: layout.cols as usize,
                    basis: layout.basis,
                    values: None,
                    packed: Some(PackedAdditive::upload_stream(backend, a, source)?),
                    owner: Arc::clone(&backend.stream),
                }))
            }
            (
                TensorMetadata::Dense { rows, cols },
                TensorStreamEncoding::DenseLittleEndian(precision),
            ) => {
                let device = stream_copy::dense(backend, source, rows, cols, precision)?;
                Ok(Box::new(Self {
                    rows,
                    cols,
                    basis: Basis::Identity,
                    values: device,
                    packed: None,
                    owner: Arc::clone(&backend.stream),
                }))
            }
            _ => Err(BackendError::UnsupportedTensor),
        }
    }
    pub(super) fn plan(
        metadata: TensorMetadata<'_>,
    ) -> Result<Option<TensorUploadPlan>, BackendError> {
        let (bytes, workspace) = match metadata {
            TensorMetadata::Dense { rows, cols } => {
                let bytes = TensorMetadata::dense_bytes(rows, cols)?;
                if rows != 0 && cols != 0 {
                    CudaBackend::check_grad_launch_bounds(1, rows, cols)?;
                }
                (bytes, 0)
            }
            TensorMetadata::Additive(metadata) => match PackedAdditive::plan_metadata(metadata) {
                Ok(plan) => plan,
                Err(BackendError::UnsupportedTensor) => return Ok(None),
                Err(error) => return Err(error),
            },
        };
        Ok(Some(TensorUploadPlan {
            caps: TensorCaps {
                execution: TensorExecution::Native,
                payload_bytes: bytes as u64,
            },
            host_workspace_bytes: workspace as u64,
            device_workspace_bytes: 0,
        }))
    }
    pub(super) fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Existing native matrix kernels and exact basis, on caller-owned storage.
    pub(super) fn project_device(
        &self,
        backend: &CudaBackend,
        input: &CudaSlice<f32>,
        scratch: Option<&mut CudaSlice<f32>>,
        ones: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), BackendError> {
        let count = product(batch, self.cols)?;
        if input.len() < count || output.len() < product(batch, self.rows)? {
            return Err(invalid("resident projection workspace geometry mismatch"));
        }
        let transformed = if matches!(self.basis, Basis::Identity) {
            input
        } else {
            let scratch = scratch.ok_or_else(|| invalid("resident basis scratch missing"))?;
            if scratch.len() < count {
                return Err(invalid("resident basis scratch too small"));
            }
            backend
                .stream
                .memcpy_dtod(&input.slice(..count), &mut scratch.slice_mut(..count))
                .map_err(|e| driver_err("stage resident projection basis", &e))?;
            basis::apply_prefix(backend, scratch, count, self.cols, self.basis, false)?;
            &*scratch
        };
        if let Some(packed) = &self.packed {
            packed.launch(
                backend,
                transformed,
                output,
                batch as u32,
                product(batch, self.rows)?,
            )
        } else {
            backend.matmul_forward_dev(
                transformed,
                self.values
                    .as_ref()
                    .ok_or_else(|| invalid("missing dense payload"))?,
                ones,
                GemmShape {
                    m: batch,
                    n: self.rows,
                    k: self.cols,
                },
                output,
            )
        }
    }
    pub(super) fn basis(&self) -> Basis {
        self.basis
    }

    pub(super) fn gather_device(
        &self,
        backend: &CudaBackend,
        ids: &CudaSlice<u32>,
        output: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), BackendError> {
        if ids.len() < batch
            || output.len() < product(batch, self.cols)?
            || !output.len().is_multiple_of(self.cols)
        {
            return Err(invalid("resident gather workspace geometry mismatch"));
        }
        if let Some(packed) = &self.packed {
            packed.gather_device(backend, ids, output, batch, product(batch, self.cols)?)?;
        } else {
            let cols = self.cols as u32;
            let count = (batch * self.cols) as u32;
            let mut launch = backend
                .stream
                .launch_builder(&backend.func_decoder_dense_gather);
            launch
                .arg(
                    self.values
                        .as_ref()
                        .ok_or_else(|| invalid("missing dense token table"))?,
                )
                .arg(ids)
                .arg(&mut *output)
                .arg(&cols)
                .arg(&count);
            #[allow(unsafe_code)]
            // SAFETY: graph/ID preflight and active workspace bounds prove all reads/writes.
            unsafe { launch.launch(cudarc::driver::LaunchConfig::for_num_elems(count)) }
                .map_err(|e| driver_err("resident dense embedding", &e))?;
        }
        basis::apply_prefix(
            backend,
            output,
            product(batch, self.cols)?,
            self.cols,
            self.basis,
            true,
        )
    }
    pub(super) fn caps(view: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
        let (rows, cols, execution) = match view {
            TensorView::Dense { rows, cols, .. } => {
                view.decoded_payload_bytes()?;
                (rows, cols, TensorExecution::Native)
            }
            TensorView::Additive(additive) => {
                let layout = additive.layout();
                if !admitted_execution_group(layout.law, layout.group) {
                    return Ok(None);
                }
                let bytes = match PackedAdditive::payload_bytes(additive) {
                    Ok(bytes) => bytes,
                    Err(BackendError::UnsupportedTensor) => return Ok(None),
                    Err(error) => return Err(error),
                };
                return Ok(Some(TensorCaps {
                    execution: TensorExecution::Native,
                    payload_bytes: bytes as u64,
                }));
            }
        };
        let payload_bytes = product(product(rows, cols)?, core::mem::size_of::<f32>())? as u64;
        if rows != 0 && cols != 0 {
            CudaBackend::check_grad_launch_bounds(1, rows, cols)?;
        }
        Ok(Some(TensorCaps {
            execution,
            payload_bytes,
        }))
    }

    pub(super) fn upload(
        backend: &CudaBackend,
        view: TensorView<'_>,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let caps = Self::caps(view)?.ok_or(BackendError::UnsupportedTensor)?;
        if let TensorView::Additive(additive) = view {
            let layout = additive.layout();
            return Ok(Box::new(Self {
                rows: layout.rows as usize,
                cols: layout.cols as usize,
                basis: layout.basis,
                values: None,
                packed: Some(PackedAdditive::upload(backend, additive)?),
                owner: Arc::clone(&backend.stream),
            }));
        }
        let TensorView::Dense { rows, cols, values } = view else {
            unreachable!("additive upload returns above")
        };
        let basis = Basis::Identity;
        let device = if values.is_empty() {
            None // Do not make a zero-sized driver allocation.
        } else {
            Some(backend.stream.clone_htod(values).map_err(|e| {
                alloc_or_backend(
                    "upload CUDA semantic dense",
                    &e,
                    caps.payload_bytes as usize,
                )
            })?)
        };
        Ok(Box::new(Self {
            rows,
            cols,
            basis,
            values: device,
            packed: None,
            owner: Arc::clone(&backend.stream),
        }))
    }

    pub(super) fn checked<'a>(
        backend: &CudaBackend,
        tensor: &'a dyn DeviceBuffer,
    ) -> Result<&'a Self, BackendError> {
        let tensor = tensor
            .as_any()
            .downcast_ref::<Self>()
            .ok_or_else(|| invalid("not a CUDA semantic tensor"))?;
        if !Arc::ptr_eq(&backend.stream, &tensor.owner) {
            return Err(invalid("CUDA semantic tensor belongs to another backend"));
        }
        Ok(tensor)
    }

    pub(super) fn matmul(backend: &CudaBackend, p: TensorMatmul<'_>) -> Result<(), BackendError> {
        let tensor = Self::checked(backend, p.tensor)?;
        if let Some(packed) = &tensor.packed {
            return packed.matmul(backend, p, tensor.rows, tensor.cols, tensor.basis);
        }
        let inputs = product(p.batch, tensor.cols)?;
        let outputs = product(p.batch, tensor.rows)?;
        if p.act.len() != inputs || p.transformed_act.len() != inputs || p.out.len() != outputs {
            return Err(invalid("CUDA semantic matmul shape mismatch"));
        }
        // Validate all host geometry before writes or device allocations. Empty
        // operations do not launch and need no int32 device index restriction.
        if outputs == 0 || tensor.cols == 0 {
            p.transformed_act.copy_from_slice(p.act);
            if tensor.cols != 0 {
                for act in p.transformed_act.chunks_exact_mut(tensor.cols) {
                    apply_basis(act, tensor.basis)
                        .map_err(|e| invalid(&format!("CUDA activation basis: {e:?}")))?;
                }
            }
            p.out.fill(0.);
            return Ok(());
        }
        CudaBackend::check_grad_launch_bounds(p.batch, tensor.rows, tensor.cols)?;
        let weights = tensor
            .values
            .as_ref()
            .ok_or_else(|| invalid("CUDA semantic dense payload is absent"))?;
        p.transformed_act.copy_from_slice(p.act);
        for act in p.transformed_act.chunks_exact_mut(tensor.cols) {
            apply_basis(act, tensor.basis)
                .map_err(|e| invalid(&format!("CUDA activation basis: {e:?}")))?;
        }
        let activations = backend.stream.clone_htod(p.transformed_act).map_err(|e| {
            alloc_or_backend(
                "upload CUDA semantic activations",
                &e,
                core::mem::size_of_val(p.act),
            )
        })?;
        let scale_bytes = product(tensor.rows, core::mem::size_of::<f32>())?;
        let mut scales = Vec::new();
        scales
            .try_reserve_exact(tensor.rows)
            .map_err(|_| BackendError::OutOfMemory {
                requested: scale_bytes,
            })?;
        scales.resize(tensor.rows, 1.);
        let scales = backend
            .stream
            .clone_htod(&scales)
            .map_err(|e| alloc_or_backend("upload CUDA semantic unit scales", &e, scale_bytes))?;
        let output_bytes = product(outputs, core::mem::size_of::<f32>())?;
        let mut output = backend
            .stream
            .alloc_zeros::<f32>(outputs)
            .map_err(|e| alloc_or_backend("allocate CUDA semantic output", &e, output_bytes))?;
        // Arbitrary f32 weights, unit scales: the existing --fmad=false kernel
        // is a plain GEMM, with no ternary clipping or STE in this path.
        backend.matmul_forward_dev(
            &activations,
            weights,
            &scales,
            GemmShape {
                m: p.batch,
                n: tensor.rows,
                k: tensor.cols,
            },
            &mut output,
        )?;
        backend
            .stream
            .memcpy_dtoh(&output, p.out)
            .map_err(|e| driver_err("download CUDA semantic output", &e))?;
        Ok(())
    }

    pub(super) fn embed(
        backend: &CudaBackend,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let tensor = Self::checked(backend, tensor)?;
        if let Some(packed) = &tensor.packed {
            return packed.embed(backend, ids, out, tensor.rows, tensor.cols, tensor.basis);
        }
        let expected = product(ids.len(), tensor.cols)?;
        if out.len() != expected || ids.iter().any(|&row| row >= tensor.rows) {
            return Err(invalid("CUDA semantic gather shape or row ID is invalid"));
        }
        if out.is_empty() {
            return Ok(());
        }
        let weights = tensor
            .values
            .as_ref()
            .ok_or_else(|| invalid("CUDA semantic dense payload is absent"))?;
        for (&row, output) in ids.iter().zip(out.chunks_exact_mut(tensor.cols)) {
            let start = row * tensor.cols; // Validated upload geometry and IDs.
            backend
                .stream
                .memcpy_dtoh(&weights.slice(start..start + tensor.cols), output)
                .map_err(|e| driver_err("download CUDA semantic row", &e))?;
            apply_inverse_basis(output, tensor.basis)
                .map_err(|e| invalid(&format!("CUDA embedding inverse basis: {e:?}")))?;
        }
        Ok(())
    }
}
