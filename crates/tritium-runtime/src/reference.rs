//! Production semantic reference adapter over the one core/owned tensor path.

use core::any::Any;
use tritium_core::{AdditiveError, DenseView, reference_embed, reference_ternary_matmul};
use tritium_format::{AdditiveTensor, AdditiveTensorError};
use tritium_spec::{
    BackendError, DeviceBuffer, DeviceCaps, GemmShape, MpGemm, TensorCaps, TensorExecution,
    TensorMatmul, TensorView, TernaryBackend, TernaryFormat, admitted_execution_group,
};

#[derive(Debug)]
struct AdditiveBuffer(AdditiveTensor);

impl DeviceBuffer for AdditiveBuffer {
    fn len_bytes(&self) -> usize {
        self.0.trits().len()
            + core::mem::size_of_val(self.0.scales())
            + self
                .0
                .view()
                .auxiliary_buffer_len_bytes()
                .expect("checked owned index")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug)]
struct DenseBuffer {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl DeviceBuffer for DenseBuffer {
    fn len_bytes(&self) -> usize {
        core::mem::size_of_val(self.values.as_slice())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn invalid(error: impl core::fmt::Debug) -> BackendError {
    BackendError::InvalidInput(format!("reference semantic tensor: {error:?}"))
}

fn owned_error(error: AdditiveTensorError) -> BackendError {
    match error {
        AdditiveTensorError::AllocationFailed { requested }
        | AdditiveTensorError::InvalidView(AdditiveError::AllocationFailed { requested }) => {
            BackendError::OutOfMemory { requested }
        }
        other => invalid(other),
    }
}

/// Shared semantic reference backend, available without a hardware backend.
///
/// Additive tensors retain decoded trits, exact widened stored scales and their
/// compact index, then execute the core reference directly. Native describes
/// additive execution, not packing, speed or physical accelerator qualification.
/// Payload reporting counts logical owned buffers, not capacity/handle overhead
/// or full runtime memory. Legacy packed ops are unsupported; the old unit-block
/// TQ conformance adapter remains in testkit, never a production GGUF decoder.
#[derive(Debug, Default)]
pub struct ReferenceBackend;

impl ReferenceBackend {
    /// Construct the stateless production reference provider.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl TernaryBackend for ReferenceBackend {
    fn device_id(&self) -> &str {
        "reference"
    }
    fn capabilities(&self) -> DeviceCaps {
        DeviceCaps::new("reference", "shared core semantic reference")
    }

    fn tensor_caps(&self, tensor: TensorView<'_>) -> Result<Option<TensorCaps>, BackendError> {
        let mut payload_bytes = tensor.decoded_payload_bytes()?;
        if let TensorView::Additive(view) = tensor {
            if !admitted_execution_group(view.layout().law, view.layout().group) {
                return Ok(None);
            }
            AdditiveTensor::validate_view(view).map_err(invalid)?;
            let auxiliary = u64::try_from(view.auxiliary_buffer_len_bytes().map_err(invalid)?)
                .map_err(|_| invalid("auxiliary buffer size exceeds u64"))?;
            payload_bytes = payload_bytes
                .checked_add(auxiliary)
                .ok_or_else(|| invalid("owned payload size overflows"))?;
        }
        usize::try_from(payload_bytes).map_err(|_| invalid("owned payload exceeds usize"))?;
        Ok(Some(TensorCaps {
            execution: TensorExecution::Native,
            payload_bytes,
        }))
    }

    fn upload_tensor(&self, tensor: TensorView<'_>) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        let caps = self
            .tensor_caps(tensor)?
            .ok_or(BackendError::UnsupportedTensor)?;
        match tensor {
            TensorView::Additive(view) => Ok(Box::new(AdditiveBuffer(
                AdditiveTensor::from_view(view).map_err(owned_error)?,
            ))),
            TensorView::Dense { rows, cols, values } => {
                let mut owned = Vec::new();
                owned
                    .try_reserve_exact(values.len())
                    .map_err(|_| BackendError::OutOfMemory {
                        requested: caps.payload_bytes as usize,
                    })?;
                owned.extend_from_slice(values);
                Ok(Box::new(DenseBuffer {
                    rows,
                    cols,
                    values: owned,
                }))
            }
        }
    }

    fn matmul(&self, p: TensorMatmul<'_>) -> Result<(), BackendError> {
        if let Some(buffer) = p.tensor.as_any().downcast_ref::<AdditiveBuffer>() {
            return reference_ternary_matmul(
                p.act,
                &buffer.0.view().bind_matmul(),
                p.batch,
                p.transformed_act,
                p.out,
            )
            .map_err(invalid);
        }
        let buffer = p
            .tensor
            .as_any()
            .downcast_ref::<DenseBuffer>()
            .ok_or_else(|| invalid("not a reference semantic tensor"))?;
        DenseView::new(buffer.rows, buffer.cols, &buffer.values)
            .and_then(|view| view.matmul(p.act, p.batch, p.transformed_act, p.out))
            .map_err(invalid)
    }

    fn embed_rows(
        &self,
        tensor: &dyn DeviceBuffer,
        ids: &[usize],
        out: &mut [f32],
    ) -> Result<(), BackendError> {
        if let Some(buffer) = tensor.as_any().downcast_ref::<AdditiveBuffer>() {
            return reference_embed(&buffer.0.view().bind_gather(), ids, out).map_err(invalid);
        }
        let buffer = tensor
            .as_any()
            .downcast_ref::<DenseBuffer>()
            .ok_or_else(|| invalid("not a reference semantic tensor"))?;
        DenseView::new(buffer.rows, buffer.cols, &buffer.values)
            .and_then(|view| view.embed(ids, out))
            .map_err(invalid)
    }

    fn upload_weights(
        &self,
        _: &[u8],
        _: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        Err(BackendError::UnsupportedFormat(format))
    }
    fn mpgemm(&self, _: MpGemm<'_>) -> Result<(), BackendError> {
        Err(invalid(
            "legacy packed reference is conformance-only; use semantic tensors",
        ))
    }
}
