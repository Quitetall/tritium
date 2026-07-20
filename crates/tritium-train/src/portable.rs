//! CPU reference adapter for the plan-0049 portable-training seam.

use blake3::Hasher;
use tritium_spec::{
    TrainAttributeValueV1, TrainBackendError, TrainBackendV1, TrainBufferDataMutV1,
    TrainBufferDataRefV1, TrainCapabilitiesV1, TrainDTypeV1, TrainExecutionV1, TrainLimitsV1,
    TrainOperationErrorV1, TrainOutputV1, TrainReceiptV1, TrainRequestV1, TrainingOpManifestV1,
    train_output_digest_v1, train_request_digest_v1,
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
    Detach,
    ScaleConst,
    Bias,
    Add,
    Mul,
    Relu2,
    Silu,
    Mse,
    Sgd,
}

#[derive(Clone, Copy)]
struct CpuOperationEntry {
    id: &'static str,
    operation: CpuOperation,
}

const CPU_OPERATIONS: &[CpuOperationEntry] = &[
    CpuOperationEntry {
        id: "graph.detach",
        operation: CpuOperation::Detach,
    },
    CpuOperationEntry {
        id: "graph.scale_const",
        operation: CpuOperation::ScaleConst,
    },
    CpuOperationEntry {
        id: "graph.bias",
        operation: CpuOperation::Bias,
    },
    CpuOperationEntry {
        id: "graph.add",
        operation: CpuOperation::Add,
    },
    CpuOperationEntry {
        id: "graph.mul",
        operation: CpuOperation::Mul,
    },
    CpuOperationEntry {
        id: "graph.relu2",
        operation: CpuOperation::Relu2,
    },
    CpuOperationEntry {
        id: "graph.silu",
        operation: CpuOperation::Silu,
    },
    CpuOperationEntry {
        id: "loss.mse",
        operation: CpuOperation::Mse,
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
const MUL_FORWARD: OperationSchema = ADD_FORWARD;
const MUL_VJP: OperationSchema = OperationSchema {
    inputs: &["left", "right", "grad_output"],
    attributes: &[],
    outputs: &["grad_left", "grad_right"],
};
const DETACH_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x"],
    attributes: &[],
    outputs: &["result"],
};
const DETACH_VJP: OperationSchema = OperationSchema {
    inputs: &["grad_output"],
    attributes: &[],
    outputs: &["grad_x"],
};
const SCALE_CONST_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x"],
    attributes: &["scale"],
    outputs: &["result"],
};
const SCALE_CONST_VJP: OperationSchema = OperationSchema {
    inputs: &["grad_output"],
    attributes: &["scale"],
    outputs: &["grad_x"],
};
const BIAS_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x", "bias"],
    attributes: &["rows", "cols"],
    outputs: &["result"],
};
const BIAS_VJP: OperationSchema = OperationSchema {
    inputs: &["x", "bias", "grad_output"],
    attributes: &["rows", "cols"],
    outputs: &["grad_x", "grad_bias"],
};
const UNARY_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x"],
    attributes: &[],
    outputs: &["result"],
};
const UNARY_VJP: OperationSchema = OperationSchema {
    inputs: &["x", "grad_output"],
    attributes: &[],
    outputs: &["grad_x"],
};
const MSE_FORWARD: OperationSchema = OperationSchema {
    inputs: &["prediction", "target"],
    attributes: &[],
    outputs: &["result"],
};
const MSE_VJP: OperationSchema = OperationSchema {
    inputs: &["prediction", "target", "grad_output"],
    attributes: &[],
    outputs: &["grad_prediction"],
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
        let input_digest = train_request_digest_v1(&request);
        let operation = CPU_OPERATIONS
            .iter()
            .find(|entry| entry.id == request.operation)
            .ok_or_else(|| TrainBackendError::UnsupportedOperation(request.operation.to_owned()))?;
        match (operation.operation, request.execution) {
            (CpuOperation::Detach, TrainExecutionV1::Forward) => {
                detach_forward(&request, output)?;
            }
            (CpuOperation::Detach, TrainExecutionV1::Vjp) => detach_vjp(&request, output)?,
            (CpuOperation::ScaleConst, TrainExecutionV1::Forward) => {
                scale_const_forward(&request, output)?;
            }
            (CpuOperation::ScaleConst, TrainExecutionV1::Vjp) => {
                scale_const_vjp(&request, output)?;
            }
            (CpuOperation::Bias, TrainExecutionV1::Forward) => bias_forward(&request, output)?,
            (CpuOperation::Bias, TrainExecutionV1::Vjp) => bias_vjp(&request, output)?,
            (CpuOperation::Add, TrainExecutionV1::Forward) => add_forward(&request, output)?,
            (CpuOperation::Add, TrainExecutionV1::Vjp) => add_vjp(&request, output)?,
            (CpuOperation::Mul, TrainExecutionV1::Forward) => mul_forward(&request, output)?,
            (CpuOperation::Mul, TrainExecutionV1::Vjp) => mul_vjp(&request, output)?,
            (CpuOperation::Relu2, TrainExecutionV1::Forward) => relu2_forward(&request, output)?,
            (CpuOperation::Relu2, TrainExecutionV1::Vjp) => relu2_vjp(&request, output)?,
            (CpuOperation::Silu, TrainExecutionV1::Forward) => silu_forward(&request, output)?,
            (CpuOperation::Silu, TrainExecutionV1::Vjp) => silu_vjp(&request, output)?,
            (CpuOperation::Mse, TrainExecutionV1::Forward) => mse_forward(&request, output)?,
            (CpuOperation::Mse, TrainExecutionV1::Vjp) => mse_vjp(&request, output)?,
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
            output_digest: train_output_digest_v1(output),
            peak_resident_bytes: resident_bytes(&request, output)?,
            scratch_bytes: 0,
            host_transfers: 0,
            device_resident: true,
        })
    }
}

fn detach_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &DETACH_FORWARD)?;
    let (shape, x) = input_f32(request, "x")?;
    reject_nonfinite("x", x)?;
    require_f32_output(output, "result", shape)?;
    output_f32(output, "result")?.copy_from_slice(x);
    Ok(())
}

fn detach_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &DETACH_VJP)?;
    let (shape, grad_output) = input_f32(request, "grad_output")?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", shape)?;
    output_f32(output, "grad_x")?.fill(0.0);
    Ok(())
}

fn scale_const_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &SCALE_CONST_FORWARD)?;
    let (shape, x) = input_f32(request, "x")?;
    let scale = attribute_f32(request, "scale")?;
    reject_nonfinite("x", x)?;
    require_f32_output(output, "result", shape)?;
    for (result, &x) in output_f32(output, "result")?.iter_mut().zip(x) {
        *result = x * scale;
    }
    Ok(())
}

fn scale_const_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &SCALE_CONST_VJP)?;
    let (shape, grad_output) = input_f32(request, "grad_output")?;
    let scale = attribute_f32(request, "scale")?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", shape)?;
    for (grad_x, &grad_output) in output_f32(output, "grad_x")?.iter_mut().zip(grad_output) {
        *grad_x = grad_output * scale;
    }
    Ok(())
}

fn bias_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &BIAS_FORWARD)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (bias_shape, bias) = input_f32(request, "bias")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    if x_shape != [rows, cols] || bias_shape != [cols] {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("bias", bias)?;
    require_f32_output(output, "result", x_shape)?;
    let result = output_f32(output, "result")?;
    for row in 0..rows_usize {
        for col in 0..cols_usize {
            result[row * cols_usize + col] = x[row * cols_usize + col] + bias[col];
        }
    }
    Ok(())
}

fn bias_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &BIAS_VJP)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (bias_shape, bias) = input_f32(request, "bias")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    if x_shape != [rows, cols] || bias_shape != [cols] || grad_shape != x_shape {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("bias", bias)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", x_shape)?;
    require_f32_output(output, "grad_bias", bias_shape)?;
    output_f32(output, "grad_x")?.copy_from_slice(grad_output);
    let grad_bias = output_f32(output, "grad_bias")?;
    grad_bias.fill(0.0);
    for row in 0..rows_usize {
        for col in 0..cols_usize {
            grad_bias[col] += grad_output[row * cols_usize + col];
        }
    }
    Ok(())
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

fn mul_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &MUL_FORWARD)?;
    let (left_shape, left) = input_f32(request, "left")?;
    let (right_shape, right) = input_f32(request, "right")?;
    if left_shape != right_shape {
        return Err(shape_error());
    }
    reject_nonfinite("left", left)?;
    reject_nonfinite("right", right)?;
    require_f32_output(output, "result", left_shape)?;
    for ((result, &left), &right) in output_f32(output, "result")?
        .iter_mut()
        .zip(left)
        .zip(right)
    {
        *result = left * right;
    }
    Ok(())
}

fn mul_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &MUL_VJP)?;
    let (left_shape, left) = input_f32(request, "left")?;
    let (right_shape, right) = input_f32(request, "right")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    if left_shape != right_shape || grad_shape != left_shape {
        return Err(shape_error());
    }
    reject_nonfinite("left", left)?;
    reject_nonfinite("right", right)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_left", left_shape)?;
    require_f32_output(output, "grad_right", right_shape)?;
    for ((grad_left, &grad_output), &right) in output_f32(output, "grad_left")?
        .iter_mut()
        .zip(grad_output)
        .zip(right)
    {
        *grad_left = grad_output * right;
    }
    for ((grad_right, &grad_output), &left) in output_f32(output, "grad_right")?
        .iter_mut()
        .zip(grad_output)
        .zip(left)
    {
        *grad_right = grad_output * left;
    }
    Ok(())
}

fn relu2_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &UNARY_FORWARD)?;
    let (shape, x) = input_f32(request, "x")?;
    reject_nonfinite("x", x)?;
    require_f32_output(output, "result", shape)?;
    for (result, &x) in output_f32(output, "result")?.iter_mut().zip(x) {
        let relu = x.max(0.0);
        *result = relu * relu;
    }
    Ok(())
}

fn relu2_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &UNARY_VJP)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    if grad_shape != x_shape {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", x_shape)?;
    for ((grad_x, &x), &grad_output) in output_f32(output, "grad_x")?
        .iter_mut()
        .zip(x)
        .zip(grad_output)
    {
        *grad_x = grad_output * 2.0 * x.max(0.0);
    }
    Ok(())
}

fn silu_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &UNARY_FORWARD)?;
    let (shape, x) = input_f32(request, "x")?;
    reject_nonfinite("x", x)?;
    require_f32_output(output, "result", shape)?;
    for (result, &x) in output_f32(output, "result")?.iter_mut().zip(x) {
        *result = x * sigmoid(x);
    }
    Ok(())
}

fn silu_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &UNARY_VJP)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    if grad_shape != x_shape {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", x_shape)?;
    for ((grad_x, &x), &grad_output) in output_f32(output, "grad_x")?
        .iter_mut()
        .zip(x)
        .zip(grad_output)
    {
        let sigmoid = sigmoid(x);
        *grad_x = grad_output * (sigmoid + x * sigmoid * (1.0 - sigmoid));
    }
    Ok(())
}

fn mse_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &MSE_FORWARD)?;
    let (prediction_shape, prediction) = input_f32(request, "prediction")?;
    let (target_shape, target) = input_f32(request, "target")?;
    if prediction_shape != target_shape || prediction.is_empty() {
        return Err(shape_error());
    }
    reject_nonfinite("prediction", prediction)?;
    reject_nonfinite("target", target)?;
    require_f32_output(output, "result", &[])?;
    let sum: f32 = prediction
        .iter()
        .zip(target)
        .map(|(&prediction, &target)| {
            let difference = prediction - target;
            difference * difference
        })
        .sum();
    output_f32(output, "result")?[0] = sum / prediction.len() as f32;
    Ok(())
}

fn mse_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &MSE_VJP)?;
    let (prediction_shape, prediction) = input_f32(request, "prediction")?;
    let (target_shape, target) = input_f32(request, "target")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    if prediction_shape != target_shape || prediction.is_empty() || !grad_shape.is_empty() {
        return Err(shape_error());
    }
    reject_nonfinite("prediction", prediction)?;
    reject_nonfinite("target", target)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_prediction", prediction_shape)?;
    let factor = grad_output[0] * 2.0 / prediction.len() as f32;
    for ((grad_prediction, &prediction), &target) in output_f32(output, "grad_prediction")?
        .iter_mut()
        .zip(prediction)
        .zip(target)
    {
        *grad_prediction = factor * (prediction - target);
    }
    Ok(())
}

#[inline]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
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

fn require_f32_output(
    output: &TrainOutputV1<'_>,
    name: &str,
    shape: &[u64],
) -> Result<(), TrainBackendError> {
    let buffer = output
        .buffers
        .iter()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| role_error("output"))?;
    if buffer.shape != shape {
        return Err(shape_error());
    }
    if !matches!(&buffer.data, TrainBufferDataMutV1::F32(_)) {
        return Err(dtype_error(
            buffer.name,
            TrainDTypeV1::F32,
            mut_dtype(&buffer.data),
        ));
    }
    Ok(())
}

fn matrix_attributes(
    request: &TrainRequestV1<'_>,
) -> Result<(u64, u64, usize, usize), TrainBackendError> {
    let rows = attribute_u64(request, "rows")?;
    let cols = attribute_u64(request, "cols")?;
    let rows_usize = usize::try_from(rows).map_err(|_| attribute_value("rows", "usize"))?;
    let cols_usize = usize::try_from(cols).map_err(|_| attribute_value("cols", "usize"))?;
    rows_usize
        .checked_mul(cols_usize)
        .ok_or_else(|| attribute_value("rows", "rows_times_cols"))?;
    Ok((rows, cols, rows_usize, cols_usize))
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
