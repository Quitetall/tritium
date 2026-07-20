//! CPU reference adapter for the plan-0049 portable-training seam.

use blake3::Hasher;
use tritium_spec::{
    TrainAttributeV1, TrainAttributeValueV1, TrainBackendError, TrainBackendV1,
    TrainBufferDataMutV1, TrainBufferDataRefV1, TrainCapabilitiesV1, TrainDTypeV1,
    TrainExecutionV1, TrainLimitsV1, TrainNamedBufferRefV1, TrainOperationErrorV1, TrainOutputV1,
    TrainReceiptV1, TrainRequestV1, TrainingOpManifestV1,
};

use crate::{Optimizer, Sgd};

const BACKEND_ID: &str = "cpu.reference.v1";
const CPU_LIMITS: TrainLimitsV1 = TrainLimitsV1 {
    max_rank: u32::MAX,
    max_elements: usize::MAX as u64,
    max_bytes: usize::MAX as u64,
};

#[derive(Clone, Copy)]
enum CpuOperation {
    Add,
    Sgd,
}

#[derive(Clone, Copy)]
struct CpuOperationEntry {
    id: &'static str,
    operation: CpuOperation,
}

const CPU_OPERATIONS: &[CpuOperationEntry] = &[
    CpuOperationEntry {
        id: "graph.add",
        operation: CpuOperation::Add,
    },
    CpuOperationEntry {
        id: "optimizer.sgd",
        operation: CpuOperation::Sgd,
    },
];

struct OperationSchema {
    inputs: &'static [&'static str],
    attributes: &'static [&'static str],
    outputs: &'static [&'static str],
}

const ADD_FORWARD: OperationSchema = OperationSchema {
    inputs: &["left", "right"],
    attributes: &[],
    outputs: &["result"],
};
const ADD_VJP: OperationSchema = OperationSchema {
    inputs: &["grad_output"],
    attributes: &[],
    outputs: &["grad_left", "grad_right"],
};
const SGD_STEP: OperationSchema = OperationSchema {
    inputs: &["parameter", "gradient"],
    attributes: &["step", "lr"],
    outputs: &["parameter"],
};

/// Honest partial CPU reference adapter for `TrainBackendV1`.
///
/// Capability coverage grows only when forward/VJP or state-transition vectors
/// pass through this exact seam. It is not v1-conforming until it advertises
/// every operation in [`TrainingOpManifestV1`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuTrainBackendV1;

impl CpuTrainBackendV1 {
    /// Construct the stateless CPU reference adapter.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl TrainBackendV1 for CpuTrainBackendV1 {
    fn capabilities(&self) -> TrainCapabilitiesV1 {
        TrainCapabilitiesV1 {
            backend_id: BACKEND_ID.to_owned(),
            manifest_digest: TrainingOpManifestV1::digest(),
            supported_operations: CPU_OPERATIONS
                .iter()
                .map(|operation| operation.id.to_owned())
                .collect(),
            dtypes: vec![TrainDTypeV1::F32],
            limits: CPU_LIMITS,
            device_resident: true,
        }
    }

    fn execute(
        &self,
        request: TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<TrainReceiptV1, TrainBackendError> {
        request.validate_with_limits(output, CPU_LIMITS)?;
        let input_digest = digest_request(&request);
        let operation = CPU_OPERATIONS
            .iter()
            .find(|entry| entry.id == request.operation)
            .ok_or_else(|| TrainBackendError::UnsupportedOperation(request.operation.to_owned()))?;
        match (operation.operation, request.execution) {
            (CpuOperation::Add, TrainExecutionV1::Forward) => add_forward(&request, output)?,
            (CpuOperation::Add, TrainExecutionV1::Vjp) => add_vjp(&request, output)?,
            (CpuOperation::Sgd, TrainExecutionV1::Step) => sgd_step(&request, output)?,
            _ => {
                return Err(TrainBackendError::Backend {
                    code: "dispatch_invariant".to_owned(),
                    message: "manifest phase validation disagrees with CPU registry".to_owned(),
                });
            }
        }
        Ok(TrainReceiptV1 {
            backend_id: BACKEND_ID.to_owned(),
            backend_build: backend_build_identity(),
            physical_device: None,
            manifest_digest: TrainingOpManifestV1::digest(),
            vector_digest: request.vector_digest,
            operation: request.operation.to_owned(),
            execution: request.execution,
            dtype: TrainDTypeV1::F32,
            limits: CPU_LIMITS,
            input_digest,
            output_digest: digest_output(output),
            peak_resident_bytes: resident_bytes(&request, output)?,
            scratch_bytes: 0,
            host_transfers: 0,
            device_resident: true,
        })
    }
}

fn add_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &ADD_FORWARD)?;
    let (left_shape, left) = input_f32(request, "left")?;
    let (right_shape, right) = input_f32(request, "right")?;
    let result = output.buffers.first().ok_or_else(|| role_error("output"))?;
    if left_shape != right_shape || left_shape != result.shape {
        return Err(shape_error());
    }
    reject_nonfinite("left", left)?;
    reject_nonfinite("right", right)?;
    let result = output_f32(output, "result")?;
    for ((result, &left), &right) in result.iter_mut().zip(left).zip(right) {
        *result = left + right;
    }
    Ok(())
}

fn add_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &ADD_VJP)?;
    let (shape, grad_output) = input_f32(request, "grad_output")?;
    reject_nonfinite("grad_output", grad_output)?;
    if output.buffers.iter().any(|buffer| buffer.shape != shape) {
        return Err(shape_error());
    }
    for buffer in output.buffers.iter() {
        if !matches!(&buffer.data, TrainBufferDataMutV1::F32(_)) {
            return Err(dtype_error(
                buffer.name,
                TrainDTypeV1::F32,
                mut_dtype(&buffer.data),
            ));
        }
    }
    for buffer in output.buffers.iter_mut() {
        if let TrainBufferDataMutV1::F32(data) = &mut buffer.data {
            data.copy_from_slice(grad_output);
        }
    }
    Ok(())
}

fn sgd_step(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &SGD_STEP)?;
    let (parameter_shape, parameter) = input_f32(request, "parameter")?;
    let (gradient_shape, gradient) = input_f32(request, "gradient")?;
    if parameter_shape != gradient_shape || output.buffers[0].shape != parameter_shape {
        return Err(shape_error());
    }
    reject_nonfinite("parameter", parameter)?;
    reject_nonfinite("gradient", gradient)?;
    let step = attribute_u64(request, "step")?;
    let lr = attribute_f32(request, "lr")?;
    if step == 0 {
        return Err(attribute_value("step", "one_based"));
    }
    if lr < 0.0 {
        return Err(attribute_value("lr", "nonnegative"));
    }
    let updated = output_f32(output, "parameter")?;
    updated.copy_from_slice(parameter);
    let optimizer = Sgd { lr };
    let mut state = optimizer.init_state(updated.len());
    optimizer.step(step, updated, gradient, &mut state);
    Ok(())
}

fn require_contract(
    request: &TrainRequestV1<'_>,
    output: &TrainOutputV1<'_>,
    schema: &OperationSchema,
) -> Result<(), TrainBackendError> {
    if !same_names(
        request.inputs.iter().map(|buffer| buffer.name),
        schema.inputs,
    ) {
        return Err(role_error("input"));
    }
    if !same_names(
        request.attributes.iter().map(|attribute| attribute.name),
        schema.attributes,
    ) {
        return Err(role_error("attribute"));
    }
    if !same_names(
        output.buffers.iter().map(|buffer| buffer.name),
        schema.outputs,
    ) {
        return Err(role_error("output"));
    }
    Ok(())
}

fn same_names<'a>(observed: impl Iterator<Item = &'a str>, expected: &[&str]) -> bool {
    observed.eq(expected.iter().copied())
}

fn input_f32<'a>(
    request: &TrainRequestV1<'a>,
    name: &str,
) -> Result<(&'a [u64], &'a [f32]), TrainBackendError> {
    let buffer = request
        .inputs
        .iter()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| role_error("input"))?;
    match buffer.data {
        TrainBufferDataRefV1::F32(data) => Ok((buffer.shape, data)),
        data => Err(dtype_error(name, TrainDTypeV1::F32, ref_dtype(data))),
    }
}

fn output_f32<'a>(
    output: &'a mut TrainOutputV1<'_>,
    name: &str,
) -> Result<&'a mut [f32], TrainBackendError> {
    let buffer = output
        .buffers
        .iter_mut()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| role_error("output"))?;
    match &mut buffer.data {
        TrainBufferDataMutV1::F32(data) => Ok(data),
        data => Err(dtype_error(name, TrainDTypeV1::F32, mut_dtype(data))),
    }
}

fn attribute_u64(request: &TrainRequestV1<'_>, name: &str) -> Result<u64, TrainBackendError> {
    match request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .map(|attribute| attribute.value)
    {
        Some(TrainAttributeValueV1::U64(value)) => Ok(value),
        _ => Err(attribute_type(name, "u64")),
    }
}

fn attribute_f32(request: &TrainRequestV1<'_>, name: &str) -> Result<f32, TrainBackendError> {
    match request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .map(|attribute| attribute.value)
    {
        Some(TrainAttributeValueV1::F32(value)) => Ok(value),
        _ => Err(attribute_type(name, "f32")),
    }
}

fn reject_nonfinite(name: &str, data: &[f32]) -> Result<(), TrainBackendError> {
    if data.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::NonFinite {
                name: name.to_owned(),
            },
        ))
    }
}

fn role_error(namespace: &'static str) -> TrainBackendError {
    TrainBackendError::InvalidOperation(TrainOperationErrorV1::Roles { namespace })
}

fn dtype_error(name: &str, expected: TrainDTypeV1, got: TrainDTypeV1) -> TrainBackendError {
    TrainBackendError::InvalidOperation(TrainOperationErrorV1::DType {
        name: name.to_owned(),
        expected,
        got,
    })
}

fn shape_error() -> TrainBackendError {
    TrainBackendError::InvalidOperation(TrainOperationErrorV1::Shape)
}

fn attribute_type(name: &str, expected: &'static str) -> TrainBackendError {
    TrainBackendError::InvalidOperation(TrainOperationErrorV1::AttributeType {
        name: name.to_owned(),
        expected,
    })
}

fn attribute_value(name: &str, constraint: &'static str) -> TrainBackendError {
    TrainBackendError::InvalidOperation(TrainOperationErrorV1::AttributeValue {
        name: name.to_owned(),
        constraint,
    })
}

const fn ref_dtype(data: TrainBufferDataRefV1<'_>) -> TrainDTypeV1 {
    match data {
        TrainBufferDataRefV1::F32(_) => TrainDTypeV1::F32,
        TrainBufferDataRefV1::U32(_) => TrainDTypeV1::U32,
        TrainBufferDataRefV1::Bytes(_) => TrainDTypeV1::Bytes,
    }
}

const fn mut_dtype(data: &TrainBufferDataMutV1<'_>) -> TrainDTypeV1 {
    match data {
        TrainBufferDataMutV1::F32(_) => TrainDTypeV1::F32,
        TrainBufferDataMutV1::U32(_) => TrainDTypeV1::U32,
        TrainBufferDataMutV1::Bytes(_) => TrainDTypeV1::Bytes,
    }
}

fn resident_bytes(
    request: &TrainRequestV1<'_>,
    output: &TrainOutputV1<'_>,
) -> Result<u64, TrainBackendError> {
    let mut total = 0_u64;
    for buffer in request.inputs {
        total = checked_resident_add(total, ref_payload_bytes(buffer.data)?)?;
    }
    for buffer in output.buffers.iter() {
        total = checked_resident_add(total, mut_payload_bytes(&buffer.data)?)?;
    }
    Ok(total)
}

fn ref_payload_bytes(data: TrainBufferDataRefV1<'_>) -> Result<u64, TrainBackendError> {
    match data {
        TrainBufferDataRefV1::F32(values) => payload_bytes(values.len(), 4),
        TrainBufferDataRefV1::U32(values) => payload_bytes(values.len(), 4),
        TrainBufferDataRefV1::Bytes(values) => payload_bytes(values.len(), 1),
    }
}

fn mut_payload_bytes(data: &TrainBufferDataMutV1<'_>) -> Result<u64, TrainBackendError> {
    match data {
        TrainBufferDataMutV1::F32(values) => payload_bytes(values.len(), 4),
        TrainBufferDataMutV1::U32(values) => payload_bytes(values.len(), 4),
        TrainBufferDataMutV1::Bytes(values) => payload_bytes(values.len(), 1),
    }
}

fn payload_bytes(elements: usize, width: u64) -> Result<u64, TrainBackendError> {
    u64::try_from(elements)
        .ok()
        .and_then(|elements| elements.checked_mul(width))
        .ok_or_else(receipt_overflow)
}

fn checked_resident_add(total: u64, bytes: u64) -> Result<u64, TrainBackendError> {
    total.checked_add(bytes).ok_or_else(receipt_overflow)
}

fn receipt_overflow() -> TrainBackendError {
    TrainBackendError::Backend {
        code: "receipt_overflow".to_owned(),
        message: "resident tensor byte count exceeds u64".to_owned(),
    }
}

fn backend_build_identity() -> String {
    let mut hasher = Hasher::new();
    hasher.update(env!("CARGO_PKG_NAME").as_bytes());
    hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
    hasher.update(include_bytes!("portable.rs"));
    hasher.update(include_bytes!("optim.rs"));
    hasher.update(include_bytes!("../../tritium-spec/src/train_backend.rs"));
    hasher.update(include_bytes!("../../../spec/training/v1/manifest.json"));
    format!(
        "{}@{}+source-blake3:{}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        hasher.finalize().to_hex()
    )
}

fn digest_request(request: &TrainRequestV1<'_>) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hash_str(&mut hasher, request.operation);
    hasher.update(&[execution_tag(request.execution)]);
    hash_ref_buffers(&mut hasher, request.inputs);
    hash_attributes(&mut hasher, request.attributes);
    *hasher.finalize().as_bytes()
}

fn digest_output(output: &TrainOutputV1<'_>) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hash_u64(&mut hasher, output.buffers.len() as u64);
    for buffer in output.buffers.iter() {
        hash_str(&mut hasher, buffer.name);
        hash_shape(&mut hasher, buffer.shape);
        match &buffer.data {
            TrainBufferDataMutV1::F32(data) => hash_f32(&mut hasher, data),
            TrainBufferDataMutV1::U32(data) => hash_u32(&mut hasher, data),
            TrainBufferDataMutV1::Bytes(data) => hash_bytes(&mut hasher, data),
        }
    }
    *hasher.finalize().as_bytes()
}

fn hash_ref_buffers(hasher: &mut Hasher, buffers: &[TrainNamedBufferRefV1<'_>]) {
    hash_u64(hasher, buffers.len() as u64);
    for buffer in buffers {
        hash_str(hasher, buffer.name);
        hash_shape(hasher, buffer.shape);
        match buffer.data {
            TrainBufferDataRefV1::F32(data) => hash_f32(hasher, data),
            TrainBufferDataRefV1::U32(data) => hash_u32(hasher, data),
            TrainBufferDataRefV1::Bytes(data) => hash_bytes(hasher, data),
        }
    }
}

fn hash_attributes(hasher: &mut Hasher, attributes: &[TrainAttributeV1<'_>]) {
    hash_u64(hasher, attributes.len() as u64);
    for attribute in attributes {
        hash_str(hasher, attribute.name);
        match attribute.value {
            TrainAttributeValueV1::F32(value) => {
                hasher.update(&[0]);
                hasher.update(&value.to_bits().to_le_bytes());
            }
            TrainAttributeValueV1::U64(value) => {
                hasher.update(&[1]);
                hash_u64(hasher, value);
            }
            TrainAttributeValueV1::Bool(value) => {
                hasher.update(&[2, u8::from(value)]);
            }
            TrainAttributeValueV1::Text(value) => {
                hasher.update(&[3]);
                hash_str(hasher, value);
            }
            TrainAttributeValueV1::U64List(values) => {
                hasher.update(&[4]);
                hash_u64(hasher, values.len() as u64);
                for &value in values {
                    hash_u64(hasher, value);
                }
            }
            TrainAttributeValueV1::U32List(values) => {
                hasher.update(&[5]);
                hash_u32(hasher, values);
            }
        }
    }
}

fn hash_shape(hasher: &mut Hasher, shape: &[u64]) {
    hash_u64(hasher, shape.len() as u64);
    for &dimension in shape {
        hash_u64(hasher, dimension);
    }
}

fn hash_f32(hasher: &mut Hasher, values: &[f32]) {
    hasher.update(&[0]);
    hash_u64(hasher, values.len() as u64);
    for &value in values {
        hasher.update(&value.to_bits().to_le_bytes());
    }
}

fn hash_u32(hasher: &mut Hasher, values: &[u32]) {
    hasher.update(&[1]);
    hash_u64(hasher, values.len() as u64);
    for &value in values {
        hasher.update(&value.to_le_bytes());
    }
}

fn hash_bytes(hasher: &mut Hasher, values: &[u8]) {
    hasher.update(&[2]);
    hash_u64(hasher, values.len() as u64);
    hasher.update(values);
}

fn hash_str(hasher: &mut Hasher, value: &str) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value.as_bytes());
}

fn hash_u64(hasher: &mut Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

const fn execution_tag(execution: TrainExecutionV1) -> u8 {
    match execution {
        TrainExecutionV1::Forward => 0,
        TrainExecutionV1::Vjp => 1,
        TrainExecutionV1::Step => 2,
        TrainExecutionV1::Checkpoint => 3,
        TrainExecutionV1::Resume => 4,
        TrainExecutionV1::Export => 5,
        TrainExecutionV1::Reload => 6,
    }
}
