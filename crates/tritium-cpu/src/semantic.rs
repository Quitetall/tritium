//! Scalar semantic execution; legacy packed SIMD dispatch is unchanged.

use core::any::Any;
use tritium_core::{DenseView, reference_embed, reference_ternary_matmul};
use tritium_format::{AdditiveTensor, AdditiveTensorError};
use tritium_spec::{
    BackendError, DeviceBuffer, TensorCaps, TensorExecution, TensorMatmul, TensorView,
    admitted_execution_group,
};

#[derive(Debug)]
pub(crate) enum CpuTensor {
    Additive(AdditiveTensor),
    Dense {
        rows: usize,
        cols: usize,
        values: Vec<f32>,
    },
}

impl DeviceBuffer for CpuTensor {
    fn len_bytes(&self) -> usize {
        match self {
            Self::Additive(tensor) => {
                tensor.trits().len() + core::mem::size_of_val(tensor.scales())
            }
            Self::Dense { values, .. } => core::mem::size_of_val(values.as_slice()),
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl CpuTensor {
    pub(crate) fn caps(view: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
        if let TensorView::Additive(additive) = view {
            let layout = additive.layout();
            if !admitted_execution_group(layout.law, layout.group) {
                return Ok(None);
            }
            AdditiveTensor::validate_view(additive).map_err(|e| {
                BackendError::InvalidInput(format!("CPU additive capability input failed: {e:?}"))
            })?;
        }
        Ok(Some(TensorCaps {
            execution: TensorExecution::Native,
            payload_bytes: view.decoded_payload_bytes()?,
        }))
    }

    pub(crate) fn upload(view: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let tensor = match view {
            TensorView::Additive(view) => {
                Self::Additive(AdditiveTensor::from_view(view).map_err(|e| match e {
                    AdditiveTensorError::AllocationFailed { requested } => {
                        BackendError::OutOfMemory { requested }
                    }
                    other => {
                        BackendError::InvalidInput(format!("CPU additive upload failed: {other:?}"))
                    }
                })?)
            }
            TensorView::Dense { rows, cols, values } => {
                DenseView::new(rows, cols, values).map_err(|e| {
                    BackendError::InvalidInput(format!("CPU dense upload failed: {e:?}"))
                })?;
                let requested = values
                    .len()
                    .checked_mul(core::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        BackendError::InvalidInput("dense storage size overflows".into())
                    })?;
                let mut owned = Vec::new();
                owned
                    .try_reserve_exact(values.len())
                    .map_err(|_| BackendError::OutOfMemory { requested })?;
                owned.extend_from_slice(values);
                Self::Dense {
                    rows,
                    cols,
                    values: owned,
                }
            }
        };
        Ok(Box::new(tensor))
    }

    pub(crate) fn matmul(p: TensorMatmul<'_>) -> Result<(), BackendError> {
        let tensor = Self::checked(p.tensor)?;
        match tensor {
            Self::Additive(tensor) => reference_ternary_matmul(
                p.act,
                &tensor.view().bind_matmul(),
                p.batch,
                p.transformed_act,
                p.out,
            )
            .map_err(|e| BackendError::InvalidInput(format!("CPU additive matmul failed: {e:?}"))),
            Self::Dense { rows, cols, values } => DenseView::new(*rows, *cols, values)
                .and_then(|view| view.matmul(p.act, p.batch, p.transformed_act, p.out))
                .map_err(|e| BackendError::InvalidInput(format!("CPU dense matmul failed: {e:?}"))),
        }
    }

    pub(crate) fn embed(
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        match Self::checked(tensor)? {
            Self::Additive(tensor) => reference_embed(&tensor.view().bind_gather(), ids, out)
                .map_err(|e| {
                    BackendError::InvalidInput(format!("CPU additive gather failed: {e:?}"))
                }),
            Self::Dense { rows, cols, values } => DenseView::new(*rows, *cols, values)
                .and_then(|view| view.embed(ids, out))
                .map_err(|e| BackendError::InvalidInput(format!("CPU dense gather failed: {e:?}"))),
        }
    }

    fn checked(tensor: &dyn DeviceBuffer) -> Result<&Self, BackendError> {
        tensor
            .as_any()
            .downcast_ref::<Self>()
            .ok_or_else(|| BackendError::InvalidInput("not a CPU semantic tensor".into()))
    }
}
