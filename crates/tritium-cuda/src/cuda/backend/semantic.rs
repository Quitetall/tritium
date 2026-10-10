//! Device-owned dense execution through the shared tensor interface.
//!
//! This is a prerequisite for additive emulation, not an emulated capability:
//! additive views remain unsupported here. Existing packed/resident kernels
//! and their dispatch are unchanged. No dense host shadow is retained.

use super::{CudaBackend, alloc_or_backend, driver_err};
use core::any::Any;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;
use tritium_core::GemmShape;
use tritium_spec::{
    BackendError, DeviceBuffer, TensorCaps, TensorExecution, TensorMatmul, TensorView,
};

pub(super) struct CudaTensor {
    rows: usize,
    cols: usize,
    values: Option<CudaSlice<f32>>,
    // Retain the context and identify the exact owning backend stream. A device
    // ordinal/string alone cannot prove that a handle belongs to this adapter.
    owner: Arc<CudaStream>,
}

impl DeviceBuffer for CudaTensor {
    fn len_bytes(&self) -> usize {
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
    pub(super) fn caps(view: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
        let TensorView::Dense { rows, cols, .. } = view else {
            return Ok(None);
        };
        let payload_bytes = view.decoded_payload_bytes()?;
        if rows != 0 && cols != 0 {
            CudaBackend::check_grad_launch_bounds(1, rows, cols)?;
        }
        Ok(Some(TensorCaps {
            execution: TensorExecution::Native,
            payload_bytes,
        }))
    }

    pub(super) fn upload(
        backend: &CudaBackend,
        view: TensorView<'_>,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let caps = Self::caps(view)?.ok_or(BackendError::UnsupportedTensor)?;
        let TensorView::Dense { rows, cols, values } = view else {
            return Err(BackendError::UnsupportedTensor);
        };
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
            values: device,
            owner: Arc::clone(&backend.stream),
        }))
    }

    fn checked<'a>(
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
        let inputs = product(p.batch, tensor.cols)?;
        let outputs = product(p.batch, tensor.rows)?;
        if p.act.len() != inputs || p.transformed_act.len() != inputs || p.out.len() != outputs {
            return Err(invalid("CUDA semantic matmul shape mismatch"));
        }
        // Validate all host geometry before writes or device allocations. Empty
        // operations do not launch and need no int32 device index restriction.
        if outputs == 0 || tensor.cols == 0 {
            p.transformed_act.copy_from_slice(p.act);
            p.out.fill(0.);
            return Ok(());
        }
        CudaBackend::check_grad_launch_bounds(p.batch, tensor.rows, tensor.cols)?;
        let weights = tensor
            .values
            .as_ref()
            .ok_or_else(|| invalid("CUDA semantic dense payload is absent"))?;
        let activations = backend.stream.clone_htod(p.act).map_err(|e| {
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
        p.transformed_act.copy_from_slice(p.act);
        Ok(())
    }

    pub(super) fn embed(
        backend: &CudaBackend,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        let tensor = Self::checked(backend, tensor)?;
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
        }
        Ok(())
    }
}
