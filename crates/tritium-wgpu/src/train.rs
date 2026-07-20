//! Portable-training lifecycle adapter for the native wgpu device.

use tritium_spec::{
    BackendError, TrainAttributeValueV1, TrainBackendError, TrainBackendV1, TrainBufferDataMutV1,
    TrainBufferDataRefV1, TrainCapabilitiesV1, TrainDTypeV1, TrainExecutionV1, TrainLimitsV1,
    TrainOperationErrorV1, TrainOutputV1, TrainReceiptV1, TrainRequestV1, TrainingOpManifestV1,
    train_output_digest_v1, train_request_digest_v1,
};

use crate::WgpuBackend;

const OPERATIONS: &[&str] = &[
    "graph.detach",
    "graph.scale_const",
    "graph.add",
    "graph.mul",
    "graph.relu2",
    "graph.silu",
    "graph.causal_mask",
    "graph.softmax",
    "lifecycle.checkpoint",
    "lifecycle.resume",
    "lifecycle.export",
    "lifecycle.reload",
];
const LIMITS: TrainLimitsV1 = TrainLimitsV1 {
    max_rank: 4,
    max_elements: i32::MAX as u64,
    max_bytes: u32::MAX as u64,
};

/// Native-wgpu implementation of the frozen portable-training seam.
///
/// The initial proved slice contains lifecycle operations only. These operate
/// on canonical host-visible checkpoint/artifact bytes and never invoke CPU
/// tensor execution. Tensor operations are advertised only as their WGSL
/// kernels pass the frozen corpus on an actual adapter.
#[derive(Debug)]
pub struct WgpuTrainBackendV1 {
    backend: WgpuBackend,
    physical_device: String,
}

impl WgpuTrainBackendV1 {
    /// Open the selected native wgpu adapter.
    ///
    /// # Errors
    /// Returns a backend error when no compatible native adapter is available.
    pub fn new() -> Result<Self, BackendError> {
        let backend = WgpuBackend::new()?;
        let physical_device = backend.physical_device().to_owned();
        Ok(Self {
            backend,
            physical_device,
        })
    }

    fn execute_pointwise(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_names, output_names): (&[&str], &[&str]) =
            match (request.operation, request.execution) {
                ("graph.detach", TrainExecutionV1::Forward) => (&["x"], &["result"]),
                ("graph.detach", TrainExecutionV1::Vjp) => (&["grad_output"], &["grad_x"]),
                ("graph.scale_const", TrainExecutionV1::Forward) => (&["x"], &["result"]),
                ("graph.scale_const", TrainExecutionV1::Vjp) => (&["grad_output"], &["grad_x"]),
                ("graph.add", TrainExecutionV1::Forward) => (&["left", "right"], &["result"]),
                ("graph.add", TrainExecutionV1::Vjp) => {
                    (&["grad_output"], &["grad_left", "grad_right"])
                }
                ("graph.mul", TrainExecutionV1::Forward) => (&["left", "right"], &["result"]),
                ("graph.mul", TrainExecutionV1::Vjp) => (
                    &["left", "right", "grad_output"],
                    &["grad_left", "grad_right"],
                ),
                ("graph.relu2" | "graph.silu", TrainExecutionV1::Forward) => (&["x"], &["result"]),
                ("graph.relu2" | "graph.silu", TrainExecutionV1::Vjp) => {
                    (&["x", "grad_output"], &["grad_x"])
                }
                ("graph.causal_mask", TrainExecutionV1::Forward) => (&["x"], &["result"]),
                ("graph.causal_mask", TrainExecutionV1::Vjp) => (&["grad_output"], &["grad_x"]),
                ("graph.softmax", TrainExecutionV1::Forward) => (&["x"], &["result"]),
                ("graph.softmax", TrainExecutionV1::Vjp) => (&["x", "grad_output"], &["grad_x"]),
                _ => return Err(invariant("pointwise operation received an illegal phase")),
            };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            input_names,
            "inputs",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            output_names,
            "outputs",
        )?;
        let expected_attributes: &[&str] = match request.operation {
            "graph.scale_const" => &["scale"],
            "graph.causal_mask" | "graph.softmax" => &["rows", "cols"],
            _ => &[],
        };
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            expected_attributes,
            "attributes",
        )?;
        let (shape, first) = input_f32(request, input_names[0])?;
        require_finite(input_names[0], first)?;
        let scalar = if request.operation == "graph.scale_const" {
            attribute_f32(request, "scale")?
        } else {
            0.0
        };
        let second_name = match (request.operation, request.execution) {
            ("graph.add" | "graph.mul", TrainExecutionV1::Forward) => Some("right"),
            ("graph.relu2" | "graph.silu", TrainExecutionV1::Vjp) => Some("grad_output"),
            ("graph.softmax", TrainExecutionV1::Vjp) => Some("grad_output"),
            _ => None,
        };
        let second = if let Some(second_name) = second_name {
            let (second_shape, second) = input_f32(request, second_name)?;
            if second_shape != shape {
                return Err(shape_error());
            }
            require_finite(second_name, second)?;
            second
        } else {
            first
        };
        let auxiliary = if matches!(request.operation, "graph.causal_mask" | "graph.softmax") {
            let rows = attribute_u64(request, "rows")?;
            let cols = attribute_u64(request, "cols")?;
            if rows == 0 || cols == 0 || rows > u32::MAX as u64 || cols > u32::MAX as u64 {
                return Err(shape_error());
            }
            if shape != [rows, cols] {
                return Err(shape_error());
            }
            cols as u32
        } else {
            0
        };
        let results = match (request.operation, request.execution) {
            ("graph.detach", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(first, second, 0, 0.0, auxiliary)]
            }
            ("graph.detach", TrainExecutionV1::Vjp) => {
                vec![self.backend.pointwise(first, second, 1, 0.0, auxiliary)]
            }
            ("graph.scale_const", _) => {
                vec![self.backend.pointwise(first, second, 2, scalar, auxiliary)]
            }
            ("graph.add", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(first, second, 3, 0.0, auxiliary)]
            }
            ("graph.add", TrainExecutionV1::Vjp) => vec![
                self.backend.pointwise(first, second, 0, 0.0, auxiliary),
                self.backend.pointwise(first, second, 0, 0.0, auxiliary),
            ],
            ("graph.mul", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(first, second, 4, 0.0, auxiliary)]
            }
            ("graph.mul", TrainExecutionV1::Vjp) => {
                let (left_shape, left) = input_f32(request, "left")?;
                let (right_shape, right) = input_f32(request, "right")?;
                let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
                if left_shape != right_shape || left_shape != gradient_shape {
                    return Err(shape_error());
                }
                require_finite("left", left)?;
                require_finite("right", right)?;
                require_finite("grad_output", gradient)?;
                vec![
                    self.backend.pointwise(gradient, right, 4, 0.0, auxiliary),
                    self.backend.pointwise(gradient, left, 4, 0.0, auxiliary),
                ]
            }
            ("graph.relu2", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(first, second, 5, 0.0, auxiliary)]
            }
            ("graph.relu2", TrainExecutionV1::Vjp) => {
                vec![self.backend.pointwise(first, second, 6, 0.0, auxiliary)]
            }
            ("graph.silu", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(first, second, 7, 0.0, auxiliary)]
            }
            ("graph.silu", TrainExecutionV1::Vjp) => {
                vec![self.backend.pointwise(first, second, 8, 0.0, auxiliary)]
            }
            ("graph.causal_mask", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(first, second, 9, 0.0, auxiliary)]
            }
            ("graph.causal_mask", TrainExecutionV1::Vjp) => {
                vec![self.backend.pointwise(first, second, 10, 0.0, auxiliary)]
            }
            ("graph.softmax", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(first, second, 11, 0.0, auxiliary)]
            }
            ("graph.softmax", TrainExecutionV1::Vjp) => {
                vec![self.backend.pointwise(first, second, 12, 0.0, auxiliary)]
            }
            _ => unreachable!(),
        };
        for (name, result) in output_names.iter().zip(results) {
            let result = result.map_err(wgpu_error)?;
            output_f32(output, name, shape, first.len())?.copy_from_slice(&result);
        }
        Ok(())
    }
}

impl TrainBackendV1 for WgpuTrainBackendV1 {
    fn capabilities(&self) -> TrainCapabilitiesV1 {
        TrainCapabilitiesV1 {
            backend_id: "wgpu.portable.v1:wgpu".to_owned(),
            manifest_digest: TrainingOpManifestV1::digest(),
            supported_operations: OPERATIONS
                .iter()
                .map(|operation| (*operation).to_owned())
                .collect(),
            dtypes: vec![TrainDTypeV1::F32, TrainDTypeV1::Bytes],
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
        if !OPERATIONS.contains(&request.operation) {
            return Err(TrainBackendError::UnsupportedOperation(
                request.operation.to_owned(),
            ));
        }
        let input_digest = train_request_digest_v1(&request);
        let lifecycle = request.operation.starts_with("lifecycle.");
        let scratch_bytes = if lifecycle {
            tritium_train::portable::execute_lifecycle_control_plane(&request, output)?;
            tritium_train::portable::lifecycle_control_plane_scratch_bytes(&request)?
        } else {
            self.execute_pointwise(&request, output)?;
            0
        };
        Ok(TrainReceiptV1 {
            backend_id: "wgpu.portable.v1:wgpu".to_owned(),
            backend_build: format!("tritium-wgpu-{}", env!("CARGO_PKG_VERSION")),
            physical_device: Some(self.physical_device.clone()),
            manifest_digest: TrainingOpManifestV1::digest(),
            vector_digest: request.vector_digest,
            operation: request.operation.to_owned(),
            execution: request.execution,
            dtype: if lifecycle {
                TrainDTypeV1::Bytes
            } else {
                TrainDTypeV1::F32
            },
            limits: LIMITS,
            input_digest,
            output_digest: train_output_digest_v1(output),
            peak_resident_bytes: resident_bytes(&request, output)?,
            scratch_bytes,
            host_transfers: 0,
            device_resident: true,
        })
    }
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

fn shape_error() -> TrainBackendError {
    TrainBackendError::InvalidOperation(TrainOperationErrorV1::Shape)
}

fn require_names<'a>(
    actual: impl Iterator<Item = &'a str>,
    expected: &[&str],
    namespace: &'static str,
) -> Result<(), TrainBackendError> {
    if actual.eq(expected.iter().copied()) {
        Ok(())
    } else {
        Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::Roles { namespace },
        ))
    }
}

fn input_f32<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<(&'a [u64], &'a [f32]), TrainBackendError> {
    let buffer = request
        .inputs
        .iter()
        .find(|buffer| buffer.name == name)
        .ok_or({
            TrainBackendError::InvalidOperation(TrainOperationErrorV1::Roles {
                namespace: "inputs",
            })
        })?;
    match buffer.data {
        TrainBufferDataRefV1::F32(data) => Ok((buffer.shape, data)),
        TrainBufferDataRefV1::U32(_) => Err(dtype_error(name, TrainDTypeV1::U32)),
        TrainBufferDataRefV1::Bytes(_) => Err(dtype_error(name, TrainDTypeV1::Bytes)),
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
        .ok_or({
            TrainBackendError::InvalidOperation(TrainOperationErrorV1::Roles {
                namespace: "outputs",
            })
        })?;
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

fn attribute_f32(request: &TrainRequestV1<'_>, name: &str) -> Result<f32, TrainBackendError> {
    let attribute = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or({
            TrainBackendError::InvalidOperation(TrainOperationErrorV1::Roles {
                namespace: "attributes",
            })
        })?;
    let TrainAttributeValueV1::F32(value) = attribute.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "f32",
            },
        ));
    };
    if !value.is_finite() {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::NonFinite {
                name: name.to_owned(),
            },
        ));
    }
    Ok(value)
}

fn attribute_u64(request: &TrainRequestV1<'_>, name: &str) -> Result<u64, TrainBackendError> {
    let attribute = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or({
            TrainBackendError::InvalidOperation(TrainOperationErrorV1::Roles {
                namespace: "attributes",
            })
        })?;
    let TrainAttributeValueV1::U64(value) = attribute.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "u64",
            },
        ));
    };
    Ok(value)
}

fn require_finite(name: &str, values: &[f32]) -> Result<(), TrainBackendError> {
    if values.iter().any(|value| !value.is_finite()) {
        Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::NonFinite {
                name: name.to_owned(),
            },
        ))
    } else {
        Ok(())
    }
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

fn wgpu_error(error: BackendError) -> TrainBackendError {
    TrainBackendError::Backend {
        code: "wgpu".to_owned(),
        message: error.to_string(),
    }
}
