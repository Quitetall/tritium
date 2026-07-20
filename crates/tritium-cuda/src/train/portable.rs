//! Plan-0049 portable-training adapter backed by resident CUDA kernels.

use tritium_spec::{
    BackendError, TernaryBackend, TrainAttributeValueV1, TrainBackendError, TrainBackendV1,
    TrainBufferDataMutV1, TrainBufferDataRefV1, TrainCapabilitiesV1, TrainDTypeV1,
    TrainExecutionV1, TrainLimitsV1, TrainOperationErrorV1, TrainOutputV1, TrainReceiptV1,
    TrainRequestV1, TrainingOpManifestV1, train_output_digest_v1, train_request_digest_v1,
};

use super::DeviceTape;
use crate::CudaBackend;

const BACKEND_FAMILY: &str = "cuda.portable.v1";
const OPERATIONS: &[&str] = &["graph.dense_matmul"];
const LIMITS: TrainLimitsV1 = TrainLimitsV1 {
    max_rank: 4,
    max_elements: u32::MAX as u64,
    max_bytes: u32::MAX as u64,
};

/// Incremental CUDA implementation of the portable training contract.
///
/// `supported_operations` contains only operations proved against the frozen
/// vector corpus. No request is delegated to the CPU reference adapter.
#[derive(Debug)]
pub struct CudaTrainBackendV1 {
    backend: CudaBackend,
    backend_id: String,
    physical_device: String,
}

impl CudaTrainBackendV1 {
    /// Open one CUDA ordinal and bind receipts to its physical identity.
    pub fn new(ordinal: usize) -> Result<Self, BackendError> {
        let backend = CudaBackend::new(ordinal)?;
        let device_id = backend.device_id().to_owned();
        let device_name = backend.capabilities().device_name;
        Ok(Self {
            backend,
            backend_id: format!("{BACKEND_FAMILY}:{device_id}"),
            physical_device: format!("{device_id}:{device_name}"),
        })
    }

    fn execute_dense_matmul(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x", "weight"],
            TrainExecutionV1::Vjp => &["x", "weight", "grad_output"],
            _ => {
                return Err(invariant(
                    "dense matmul received an illegal execution phase",
                ));
            }
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["m", "n", "k"],
            "attributes",
        )?;
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_x", "grad_weight"],
            _ => unreachable!(),
        };
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            expected_outputs,
            "outputs",
        )?;

        let m = attribute_usize(request, "m")?;
        let n = attribute_usize(request, "n")?;
        let k = attribute_usize(request, "k")?;
        let mk = m.checked_mul(k).ok_or_else(shape_error)?;
        let nk = n.checked_mul(k).ok_or_else(shape_error)?;
        let mn = m.checked_mul(n).ok_or_else(shape_error)?;
        let x = input_f32(request, "x", &[m as u64, k as u64])?;
        let weight = input_f32(request, "weight", &[n as u64, k as u64])?;
        if x.len() != mk || weight.len() != nk {
            return Err(shape_error());
        }
        require_finite("x", x)?;
        require_finite("weight", weight)?;

        let mut tape = DeviceTape::new(&self.backend, n).map_err(cuda_error)?;
        let x_id = tape.leaf(x).map_err(cuda_error)?;
        let weight_id = tape.leaf(weight).map_err(cuda_error)?;
        let result_id = tape.matmul(x_id, weight_id, m, n, k).map_err(cuda_error)?;

        match request.execution {
            TrainExecutionV1::Forward => {
                let result = tape.value(result_id).map_err(cuda_error)?;
                output_f32(output, "result", &[m as u64, n as u64], mn)?.copy_from_slice(&result);
            }
            TrainExecutionV1::Vjp => {
                let grad_output = input_f32(request, "grad_output", &[m as u64, n as u64])?;
                require_finite("grad_output", grad_output)?;
                let seed = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
                let gradients = tape
                    .backward_retain(result_id, &seed, &[x_id, weight_id])
                    .map_err(cuda_error)?;
                let mut grad_x = vec![0.0; mk];
                let mut grad_weight = vec![0.0; nk];
                self.backend
                    .dev_download(
                        gradients.grads[x_id]
                            .as_ref()
                            .ok_or_else(|| invariant("missing retained x gradient"))?,
                        &mut grad_x,
                    )
                    .map_err(cuda_error)?;
                self.backend
                    .dev_download(
                        gradients.grads[weight_id]
                            .as_ref()
                            .ok_or_else(|| invariant("missing retained weight gradient"))?,
                        &mut grad_weight,
                    )
                    .map_err(cuda_error)?;
                output_f32(output, "grad_x", &[m as u64, k as u64], mk)?.copy_from_slice(&grad_x);
                output_f32(output, "grad_weight", &[n as u64, k as u64], nk)?
                    .copy_from_slice(&grad_weight);
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}

impl TrainBackendV1 for CudaTrainBackendV1 {
    fn capabilities(&self) -> TrainCapabilitiesV1 {
        TrainCapabilitiesV1 {
            backend_id: self.backend_id.clone(),
            manifest_digest: TrainingOpManifestV1::digest(),
            supported_operations: OPERATIONS
                .iter()
                .map(|operation| (*operation).to_owned())
                .collect(),
            dtypes: vec![TrainDTypeV1::F32],
            limits: LIMITS,
            device_resident: true,
        }
    }

    fn execute(
        &self,
        request: TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<TrainReceiptV1, TrainBackendError> {
        request.validate_with_limits(output, LIMITS)?;
        let input_digest = train_request_digest_v1(&request);
        match request.operation {
            "graph.dense_matmul" => self.execute_dense_matmul(&request, output)?,
            operation => {
                return Err(TrainBackendError::UnsupportedOperation(
                    operation.to_owned(),
                ));
            }
        }
        Ok(TrainReceiptV1 {
            backend_id: self.backend_id.clone(),
            backend_build: format!("tritium-cuda-{}", env!("CARGO_PKG_VERSION")),
            physical_device: Some(self.physical_device.clone()),
            manifest_digest: TrainingOpManifestV1::digest(),
            vector_digest: request.vector_digest,
            operation: request.operation.to_owned(),
            execution: request.execution,
            dtype: TrainDTypeV1::F32,
            limits: LIMITS,
            input_digest,
            output_digest: train_output_digest_v1(output),
            peak_resident_bytes: resident_bytes(&request, output)?,
            scratch_bytes: 0,
            host_transfers: 0,
            device_resident: true,
        })
    }
}

fn require_names<'a>(
    actual: impl Iterator<Item = &'a str>,
    expected: &[&str],
    namespace: &'static str,
) -> Result<(), TrainBackendError> {
    if !actual.eq(expected.iter().copied()) {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::Roles { namespace },
        ));
    }
    Ok(())
}

fn attribute_usize(request: &TrainRequestV1<'_>, name: &str) -> Result<usize, TrainBackendError> {
    let value = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or_else(|| roles("attributes"))?;
    let TrainAttributeValueV1::U64(value) = value.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "u64",
            },
        ));
    };
    usize::try_from(value).map_err(|_| shape_error())
}

fn input_f32<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
    shape: &[u64],
) -> Result<&'a [f32], TrainBackendError> {
    let buffer = request
        .inputs
        .iter()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| roles("inputs"))?;
    if buffer.shape != shape {
        return Err(shape_error());
    }
    match buffer.data {
        TrainBufferDataRefV1::F32(data) => Ok(data),
        ref data => Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::DType {
                name: name.to_owned(),
                expected: TrainDTypeV1::F32,
                got: match data {
                    TrainBufferDataRefV1::F32(_) => TrainDTypeV1::F32,
                    TrainBufferDataRefV1::U32(_) => TrainDTypeV1::U32,
                    TrainBufferDataRefV1::Bytes(_) => TrainDTypeV1::Bytes,
                },
            },
        )),
    }
}

fn output_f32<'a>(
    output: &'a mut TrainOutputV1<'_>,
    name: &str,
    shape: &[u64],
    len: usize,
) -> Result<&'a mut [f32], TrainBackendError> {
    let buffer = output
        .buffers
        .iter_mut()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| roles("outputs"))?;
    if buffer.shape != shape {
        return Err(shape_error());
    }
    match &mut buffer.data {
        TrainBufferDataMutV1::F32(data) if data.len() == len => Ok(data),
        TrainBufferDataMutV1::F32(_) => Err(shape_error()),
        TrainBufferDataMutV1::U32(_) => Err(dtype_error(name, TrainDTypeV1::U32)),
        TrainBufferDataMutV1::Bytes(_) => Err(dtype_error(name, TrainDTypeV1::Bytes)),
    }
}

fn require_finite(name: &str, values: &[f32]) -> Result<(), TrainBackendError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::NonFinite {
                name: name.to_owned(),
            },
        ));
    }
    Ok(())
}

fn resident_bytes(
    request: &TrainRequestV1<'_>,
    output: &TrainOutputV1<'_>,
) -> Result<u64, TrainBackendError> {
    request
        .inputs
        .iter()
        .map(|buffer| match buffer.data {
            TrainBufferDataRefV1::F32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataRefV1::U32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataRefV1::Bytes(data) => Some(data.len() as u64),
        })
        .chain(output.buffers.iter().map(|buffer| match &buffer.data {
            TrainBufferDataMutV1::F32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataMutV1::U32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataMutV1::Bytes(data) => Some(data.len() as u64),
        }))
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes.ok_or_else(shape_error)?)
                .ok_or_else(shape_error)
        })
}

fn roles(namespace: &'static str) -> TrainBackendError {
    TrainBackendError::InvalidOperation(TrainOperationErrorV1::Roles { namespace })
}

fn shape_error() -> TrainBackendError {
    TrainBackendError::InvalidOperation(TrainOperationErrorV1::Shape)
}

fn dtype_error(name: &str, got: TrainDTypeV1) -> TrainBackendError {
    TrainBackendError::InvalidOperation(TrainOperationErrorV1::DType {
        name: name.to_owned(),
        expected: TrainDTypeV1::F32,
        got,
    })
}

fn invariant(message: &str) -> TrainBackendError {
    TrainBackendError::Backend {
        code: "dispatch_invariant".to_owned(),
        message: message.to_owned(),
    }
}

fn cuda_error(error: BackendError) -> TrainBackendError {
    TrainBackendError::Backend {
        code: "cuda".to_owned(),
        message: error.to_string(),
    }
}
