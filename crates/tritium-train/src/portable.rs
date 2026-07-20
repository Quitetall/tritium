//! CPU reference adapter for the plan-0049 portable-training seam.

use blake3::Hasher;
use tritium_spec::{
    TrainAttributeV1, TrainAttributeValueV1, TrainBackendError, TrainBackendV1,
    TrainBufferDataMutV1, TrainBufferDataRefV1, TrainCapabilitiesV1, TrainDTypeV1,
    TrainExecutionV1, TrainNamedBufferMutV1, TrainNamedBufferRefV1, TrainOutputV1, TrainReceiptV1,
    TrainRequestV1, TrainingOpManifestV1,
};

use crate::{Optimizer, Sgd};

const BACKEND_ID: &str = "cpu.reference.v1";
const SUPPORTED: &[&str] = &["graph.add", "optimizer.sgd"];

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
            supported_operations: SUPPORTED
                .iter()
                .map(|operation| (*operation).to_owned())
                .collect(),
            dtypes: vec![TrainDTypeV1::F32],
            max_rank: usize::MAX,
            max_elements: usize::MAX,
            device_resident: true,
        }
    }

    fn execute(
        &self,
        request: TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<TrainReceiptV1, TrainBackendError> {
        request.validate(output)?;
        let input_digest = digest_request(&request);
        match (request.operation, request.execution) {
            ("graph.add", TrainExecutionV1::Forward) => add_forward(&request, output)?,
            ("graph.add", TrainExecutionV1::Vjp) => add_vjp(&request, output)?,
            ("optimizer.sgd", TrainExecutionV1::Step) => sgd_step(&request, output)?,
            (operation, _) if SUPPORTED.contains(&operation) => {
                return Err(TrainBackendError::InvalidOperation(format!(
                    "{operation} does not implement requested phase"
                )));
            }
            (operation, _) => {
                return Err(TrainBackendError::UnsupportedOperation(
                    operation.to_owned(),
                ));
            }
        }
        Ok(TrainReceiptV1 {
            backend_id: BACKEND_ID.to_owned(),
            manifest_digest: TrainingOpManifestV1::digest(),
            operation: request.operation.to_owned(),
            execution: request.execution,
            input_digest,
            output_digest: digest_output(output),
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
    require_contract(request, output, &["left", "right"], &[], &["result"])?;
    let (left_shape, left) = input_f32(request, "left")?;
    let (right_shape, right) = input_f32(request, "right")?;
    let result = output
        .buffers
        .first()
        .ok_or_else(|| invalid("graph.add requires result output"))?;
    if left_shape != right_shape || left_shape != result.shape {
        return Err(invalid("graph.add shapes must match exactly"));
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
    require_contract(
        request,
        output,
        &["grad_output"],
        &[],
        &["grad_left", "grad_right"],
    )?;
    let (shape, grad_output) = input_f32(request, "grad_output")?;
    reject_nonfinite("grad_output", grad_output)?;
    if output.buffers.iter().any(|buffer| buffer.shape != shape) {
        return Err(invalid("graph.add VJP shapes must match exactly"));
    }
    for buffer in output.buffers.iter_mut() {
        match &mut buffer.data {
            TrainBufferDataMutV1::F32(data) => data.copy_from_slice(grad_output),
            _ => return Err(invalid("graph.add VJP outputs must be f32")),
        }
    }
    Ok(())
}

fn sgd_step(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(
        request,
        output,
        &["parameter", "gradient"],
        &["step", "lr", "weight_decay"],
        &["parameter"],
    )?;
    let (parameter_shape, parameter) = input_f32(request, "parameter")?;
    let (gradient_shape, gradient) = input_f32(request, "gradient")?;
    if parameter_shape != gradient_shape || output.buffers[0].shape != parameter_shape {
        return Err(invalid("optimizer.sgd shapes must match exactly"));
    }
    reject_nonfinite("parameter", parameter)?;
    reject_nonfinite("gradient", gradient)?;
    let step = attribute_u64(request, "step")?;
    let lr = attribute_f32(request, "lr")?;
    let weight_decay = attribute_f32(request, "weight_decay")?;
    if step == 0 {
        return Err(invalid("optimizer.sgd step must be one-based"));
    }
    if lr < 0.0 || weight_decay < 0.0 {
        return Err(invalid(
            "optimizer.sgd lr and weight_decay must be nonnegative",
        ));
    }
    let updated = output_f32(output, "parameter")?;
    updated.copy_from_slice(parameter);
    let optimizer = Sgd { lr, weight_decay };
    let mut state = optimizer.init_state(updated.len());
    optimizer.step(step, updated, gradient, &mut state);
    Ok(())
}

fn require_contract(
    request: &TrainRequestV1<'_>,
    output: &TrainOutputV1<'_>,
    inputs: &[&str],
    attributes: &[&str],
    outputs: &[&str],
) -> Result<(), TrainBackendError> {
    if !same_ref_names(request.inputs, inputs) {
        return Err(invalid("input roles differ from operation schema"));
    }
    if !same_attribute_names(request.attributes, attributes) {
        return Err(invalid("attributes differ from operation schema"));
    }
    if !same_mut_names(output.buffers, outputs) {
        return Err(invalid("output roles differ from operation schema"));
    }
    Ok(())
}

fn same_ref_names(observed: &[TrainNamedBufferRefV1<'_>], expected: &[&str]) -> bool {
    observed.len() == expected.len()
        && expected
            .iter()
            .all(|name| observed.iter().any(|buffer| buffer.name == *name))
}

fn same_mut_names(observed: &[TrainNamedBufferMutV1<'_>], expected: &[&str]) -> bool {
    observed.len() == expected.len()
        && expected
            .iter()
            .all(|name| observed.iter().any(|buffer| buffer.name == *name))
}

fn same_attribute_names(observed: &[TrainAttributeV1<'_>], expected: &[&str]) -> bool {
    observed.len() == expected.len()
        && expected
            .iter()
            .all(|name| observed.iter().any(|attribute| attribute.name == *name))
}

fn input_f32<'a>(
    request: &TrainRequestV1<'a>,
    name: &str,
) -> Result<(&'a [u64], &'a [f32]), TrainBackendError> {
    let buffer = request
        .inputs
        .iter()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| invalid(format!("missing input {name:?}")))?;
    match buffer.data {
        TrainBufferDataRefV1::F32(data) => Ok((buffer.shape, data)),
        _ => Err(invalid(format!("input {name:?} must be f32"))),
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
        .ok_or_else(|| invalid(format!("missing output {name:?}")))?;
    match &mut buffer.data {
        TrainBufferDataMutV1::F32(data) => Ok(data),
        _ => Err(invalid(format!("output {name:?} must be f32"))),
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
        _ => Err(invalid(format!("attribute {name:?} must be u64"))),
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
        _ => Err(invalid(format!("attribute {name:?} must be f32"))),
    }
}

fn reject_nonfinite(name: &str, data: &[f32]) -> Result<(), TrainBackendError> {
    if data.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid(format!("buffer {name:?} must be finite")))
    }
}

fn invalid(message: impl Into<String>) -> TrainBackendError {
    TrainBackendError::InvalidOperation(message.into())
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
