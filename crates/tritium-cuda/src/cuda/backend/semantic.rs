//! Device-owned dense and explicitly emulated additive execution.
//!
//! Additive coefficients expand once at upload in their stored basis. Matmul
//! transforms activations and gather undoes that basis using the shared core.
//! Existing packed/resident kernels are unchanged. No host shadow is retained.

use super::{CudaBackend, alloc_or_backend, driver_err};
use core::any::Any;
use cudarc::driver::{CudaSlice, CudaStream};
use std::sync::Arc;
use tritium_core::{GemmShape, apply_basis, apply_inverse_basis};
use tritium_format::AdditiveTensor;
use tritium_spec::{
    BackendError, Basis, DeviceBuffer, TensorCaps, TensorExecution, TensorMatmul, TensorView,
    admitted_execution_group,
};

pub(super) struct CudaTensor {
    rows: usize,
    cols: usize,
    basis: Basis,
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
                AdditiveTensor::validate_view(additive)
                    .map_err(|e| invalid(&format!("CUDA additive capability input: {e:?}")))?;
                (
                    usize::try_from(layout.rows).map_err(|_| invalid("row count overflows"))?,
                    usize::try_from(layout.cols).map_err(|_| invalid("column count overflows"))?,
                    TensorExecution::Emulated,
                )
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
        // Keep dense inputs borrowed; only additive uploads need a temporary
        // decoded allocation. It is dropped after transfer, never retained.
        let mut decoded = Vec::new();
        let (rows, cols, basis, values) = match view {
            TensorView::Dense { rows, cols, values } => (rows, cols, Basis::Identity, values),
            TensorView::Additive(additive) => {
                let layout = additive.layout();
                let rows = layout.rows as usize; // Checked by caps above.
                let cols = layout.cols as usize;
                let count = product(rows, cols)?;
                decoded
                    .try_reserve_exact(count)
                    .map_err(|_| BackendError::OutOfMemory {
                        requested: caps.payload_bytes as usize,
                    })?;
                decoded.resize(count, 0.);
                for (row, output) in decoded.chunks_exact_mut(cols).enumerate() {
                    additive
                        .dequant_row_into(row, output)
                        .map_err(|e| invalid(&format!("decode CUDA additive row: {e:?}")))?;
                }
                if decoded.iter().any(|value| !value.is_finite()) {
                    return Err(invalid(
                        "CUDA additive expansion produced nonfinite weights",
                    ));
                }
                (rows, cols, layout.basis, decoded.as_slice())
            }
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
            basis,
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
