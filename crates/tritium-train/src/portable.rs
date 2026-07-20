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
    Transpose,
    EmbeddingGather,
    SliceCols,
    ConcatCols,
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
        id: "graph.transpose",
        operation: CpuOperation::Transpose,
    },
    CpuOperationEntry {
        id: "graph.embedding_gather",
        operation: CpuOperation::EmbeddingGather,
    },
    CpuOperationEntry {
        id: "graph.slice_cols",
        operation: CpuOperation::SliceCols,
    },
    CpuOperationEntry {
        id: "graph.concat_cols",
        operation: CpuOperation::ConcatCols,
    },
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
const TRANSPOSE_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x"],
    attributes: &["rows", "cols"],
    outputs: &["result"],
};
const TRANSPOSE_VJP: OperationSchema = OperationSchema {
    inputs: &["grad_output"],
    attributes: &["rows", "cols"],
    outputs: &["grad_x"],
};
const EMBEDDING_FORWARD: OperationSchema = OperationSchema {
    inputs: &["weight", "tokens"],
    attributes: &["vocab", "n_embd"],
    outputs: &["result"],
};
const EMBEDDING_VJP: OperationSchema = OperationSchema {
    inputs: &["weight", "tokens", "grad_output"],
    attributes: &["vocab", "n_embd"],
    outputs: &["grad_weight"],
};
const SLICE_COLS_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x"],
    attributes: &["rows", "cols", "start", "len"],
    outputs: &["result"],
};
const SLICE_COLS_VJP: OperationSchema = OperationSchema {
    inputs: &["grad_output"],
    attributes: &["rows", "cols", "start", "len"],
    outputs: &["grad_x"],
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
            (CpuOperation::Transpose, TrainExecutionV1::Forward) => {
                transpose_forward(&request, output)?;
            }
            (CpuOperation::Transpose, TrainExecutionV1::Vjp) => transpose_vjp(&request, output)?,
            (CpuOperation::EmbeddingGather, TrainExecutionV1::Forward) => {
                embedding_forward(&request, output)?;
            }
            (CpuOperation::EmbeddingGather, TrainExecutionV1::Vjp) => {
                embedding_vjp(&request, output)?;
            }
            (CpuOperation::SliceCols, TrainExecutionV1::Forward) => {
                slice_cols_forward(&request, output)?;
            }
            (CpuOperation::SliceCols, TrainExecutionV1::Vjp) => {
                slice_cols_vjp(&request, output)?;
            }
            (CpuOperation::ConcatCols, TrainExecutionV1::Forward) => {
                concat_cols_forward(&request, output)?;
            }
            (CpuOperation::ConcatCols, TrainExecutionV1::Vjp) => {
                concat_cols_vjp(&request, output)?;
            }
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

fn transpose_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &TRANSPOSE_FORWARD)?;
    let (shape, x) = input_f32(request, "x")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    if shape != [rows, cols] {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    require_f32_output(output, "result", &[cols, rows])?;
    let result = output_f32(output, "result")?;
    for row in 0..rows_usize {
        for col in 0..cols_usize {
            result[col * rows_usize + row] = x[row * cols_usize + col];
        }
    }
    Ok(())
}

fn transpose_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &TRANSPOSE_VJP)?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    if grad_shape != [cols, rows] {
        return Err(shape_error());
    }
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", &[rows, cols])?;
    let grad_x = output_f32(output, "grad_x")?;
    for row in 0..rows_usize {
        for col in 0..cols_usize {
            grad_x[row * cols_usize + col] = grad_output[col * rows_usize + row];
        }
    }
    Ok(())
}

fn embedding_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &EMBEDDING_FORWARD)?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (token_shape, tokens) = input_u32(request, "tokens")?;
    let (vocab, n_embd, vocab_usize, n_embd_usize) = embedding_attributes(request)?;
    let sequence = u64::try_from(tokens.len()).map_err(|_| shape_error())?;
    if weight_shape != [vocab, n_embd] || token_shape != [sequence] {
        return Err(shape_error());
    }
    reject_nonfinite("weight", weight)?;
    reject_token_bounds(tokens, vocab_usize)?;
    require_f32_output(output, "result", &[sequence, n_embd])?;
    let result = output_f32(output, "result")?;
    for (sequence_index, &token) in tokens.iter().enumerate() {
        let source = token as usize * n_embd_usize;
        let destination = sequence_index * n_embd_usize;
        result[destination..destination + n_embd_usize]
            .copy_from_slice(&weight[source..source + n_embd_usize]);
    }
    debug_assert_eq!(weight.len(), vocab_usize * n_embd_usize);
    Ok(())
}

fn embedding_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &EMBEDDING_VJP)?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (token_shape, tokens) = input_u32(request, "tokens")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    let (vocab, n_embd, vocab_usize, n_embd_usize) = embedding_attributes(request)?;
    let sequence = u64::try_from(tokens.len()).map_err(|_| shape_error())?;
    if weight_shape != [vocab, n_embd]
        || token_shape != [sequence]
        || grad_shape != [sequence, n_embd]
    {
        return Err(shape_error());
    }
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("grad_output", grad_output)?;
    reject_token_bounds(tokens, vocab_usize)?;
    require_f32_output(output, "grad_weight", weight_shape)?;
    let grad_weight = output_f32(output, "grad_weight")?;
    grad_weight.fill(0.0);
    for (sequence_index, &token) in tokens.iter().enumerate() {
        let destination = token as usize * n_embd_usize;
        let source = sequence_index * n_embd_usize;
        for column in 0..n_embd_usize {
            grad_weight[destination + column] += grad_output[source + column];
        }
    }
    debug_assert_eq!(grad_weight.len(), vocab_usize * n_embd_usize);
    Ok(())
}

fn slice_cols_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &SLICE_COLS_FORWARD)?;
    let (shape, x) = input_f32(request, "x")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    let (len, start_usize, len_usize) = slice_attributes(request, cols)?;
    if shape != [rows, cols] {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    require_f32_output(output, "result", &[rows, len])?;
    let result = output_f32(output, "result")?;
    for row in 0..rows_usize {
        let source = row * cols_usize + start_usize;
        let destination = row * len_usize;
        result[destination..destination + len_usize]
            .copy_from_slice(&x[source..source + len_usize]);
    }
    Ok(())
}

fn slice_cols_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &SLICE_COLS_VJP)?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    let (len, start_usize, len_usize) = slice_attributes(request, cols)?;
    if grad_shape != [rows, len] {
        return Err(shape_error());
    }
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", &[rows, cols])?;
    let grad_x = output_f32(output, "grad_x")?;
    grad_x.fill(0.0);
    for row in 0..rows_usize {
        let destination = row * cols_usize + start_usize;
        let source = row * len_usize;
        grad_x[destination..destination + len_usize]
            .copy_from_slice(&grad_output[source..source + len_usize]);
    }
    Ok(())
}

fn concat_cols_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    let (rows, rows_usize, lens, total, total_usize) = concat_attributes(request)?;
    require_concat_roles(request, output, lens.len(), false)?;
    for (buffer, &len) in request.inputs.iter().zip(lens) {
        if buffer.shape != [rows, len] {
            return Err(shape_error());
        }
        match buffer.data {
            TrainBufferDataRefV1::F32(data) => reject_nonfinite(buffer.name, data)?,
            data => return Err(dtype_error(buffer.name, TrainDTypeV1::F32, ref_dtype(data))),
        }
    }
    require_f32_output(output, "result", &[rows, total])?;
    let result = output_f32(output, "result")?;
    for row in 0..rows_usize {
        let mut column_offset = 0;
        for (buffer, &len) in request.inputs.iter().zip(lens) {
            let len = len as usize;
            let data = match buffer.data {
                TrainBufferDataRefV1::F32(data) => data,
                _ => unreachable!("all concat inputs validated before mutation"),
            };
            let source = row * len;
            let destination = row * total_usize + column_offset;
            result[destination..destination + len].copy_from_slice(&data[source..source + len]);
            column_offset += len;
        }
    }
    Ok(())
}

fn concat_cols_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    let (rows, rows_usize, lens, total, total_usize) = concat_attributes(request)?;
    require_concat_roles(request, output, lens.len(), true)?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    if grad_shape != [rows, total] {
        return Err(shape_error());
    }
    reject_nonfinite("grad_output", grad_output)?;
    for (buffer, &len) in output.buffers.iter().zip(lens) {
        require_f32_output(output, buffer.name, &[rows, len])?;
    }
    let mut column_offset = 0;
    for (&len, buffer) in lens.iter().zip(output.buffers.iter_mut()) {
        let len = len as usize;
        let data = match &mut buffer.data {
            TrainBufferDataMutV1::F32(data) => data,
            _ => unreachable!("all concat outputs validated before mutation"),
        };
        for row in 0..rows_usize {
            let source = row * total_usize + column_offset;
            let destination = row * len;
            data[destination..destination + len]
                .copy_from_slice(&grad_output[source..source + len]);
        }
        column_offset += len;
    }
    Ok(())
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
    let element_count = prediction.len() as f32;
    for ((grad_prediction, &prediction), &target) in output_f32(output, "grad_prediction")?
        .iter_mut()
        .zip(prediction)
        .zip(target)
    {
        *grad_prediction = grad_output[0] * 2.0 * (prediction - target) / element_count;
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

fn input_u32<'a>(
    request: &TrainRequestV1<'a>,
    name: &str,
) -> Result<(&'a [u64], &'a [u32]), TrainBackendError> {
    let buffer = request
        .inputs
        .iter()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| role_error("input"))?;
    match buffer.data {
        TrainBufferDataRefV1::U32(data) => Ok((buffer.shape, data)),
        data => Err(dtype_error(name, TrainDTypeV1::U32, ref_dtype(data))),
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

fn embedding_attributes(
    request: &TrainRequestV1<'_>,
) -> Result<(u64, u64, usize, usize), TrainBackendError> {
    let vocab = attribute_u64(request, "vocab")?;
    let n_embd = attribute_u64(request, "n_embd")?;
    let vocab_usize = usize::try_from(vocab).map_err(|_| attribute_value("vocab", "usize"))?;
    let n_embd_usize = usize::try_from(n_embd).map_err(|_| attribute_value("n_embd", "usize"))?;
    vocab_usize
        .checked_mul(n_embd_usize)
        .ok_or_else(|| attribute_value("vocab", "vocab_times_n_embd"))?;
    Ok((vocab, n_embd, vocab_usize, n_embd_usize))
}

fn reject_token_bounds(tokens: &[u32], vocab: usize) -> Result<(), TrainBackendError> {
    if tokens.iter().all(|&token| (token as usize) < vocab) {
        Ok(())
    } else {
        Err(shape_error())
    }
}

fn slice_attributes(
    request: &TrainRequestV1<'_>,
    cols: u64,
) -> Result<(u64, usize, usize), TrainBackendError> {
    let start = attribute_u64(request, "start")?;
    let len = attribute_u64(request, "len")?;
    if start.checked_add(len).is_none_or(|end| end > cols) {
        return Err(attribute_value("start", "slice_bounds"));
    }
    let start_usize = usize::try_from(start).map_err(|_| attribute_value("start", "usize"))?;
    let len_usize = usize::try_from(len).map_err(|_| attribute_value("len", "usize"))?;
    Ok((len, start_usize, len_usize))
}

fn concat_attributes<'a>(
    request: &'a TrainRequestV1<'_>,
) -> Result<(u64, usize, &'a [u64], u64, usize), TrainBackendError> {
    if !same_names(
        request.attributes.iter().map(|attribute| attribute.name),
        &["rows", "lens"],
    ) {
        return Err(role_error("attribute"));
    }
    let rows = attribute_u64(request, "rows")?;
    let rows_usize = usize::try_from(rows).map_err(|_| attribute_value("rows", "usize"))?;
    let lens = attribute_u64_list(request, "lens")?;
    if lens.is_empty() {
        return Err(attribute_value("lens", "nonempty"));
    }
    let mut total = 0_u64;
    let mut total_usize = 0_usize;
    for &len in lens {
        total = total
            .checked_add(len)
            .ok_or_else(|| attribute_value("lens", "sum"))?;
        let len = usize::try_from(len).map_err(|_| attribute_value("lens", "usize"))?;
        total_usize = total_usize
            .checked_add(len)
            .ok_or_else(|| attribute_value("lens", "sum"))?;
    }
    rows_usize
        .checked_mul(total_usize)
        .ok_or_else(|| attribute_value("rows", "rows_times_lens"))?;
    Ok((rows, rows_usize, lens, total, total_usize))
}

fn require_concat_roles(
    request: &TrainRequestV1<'_>,
    output: &TrainOutputV1<'_>,
    parts: usize,
    vjp: bool,
) -> Result<(), TrainBackendError> {
    let valid_inputs = if vjp {
        same_names(
            request.inputs.iter().map(|buffer| buffer.name),
            &["grad_output"],
        )
    } else {
        indexed_names(
            request.inputs.iter().map(|buffer| buffer.name),
            "part.",
            parts,
        )
    };
    if !valid_inputs {
        return Err(role_error("input"));
    }
    let valid_outputs = if vjp {
        indexed_names(
            output.buffers.iter().map(|buffer| buffer.name),
            "grad_part.",
            parts,
        )
    } else {
        same_names(output.buffers.iter().map(|buffer| buffer.name), &["result"])
    };
    if !valid_outputs {
        return Err(role_error("output"));
    }
    Ok(())
}

fn indexed_names<'a>(
    observed: impl Iterator<Item = &'a str>,
    prefix: &str,
    expected: usize,
) -> bool {
    let mut count = 0;
    for (index, name) in observed.enumerate() {
        let Some(suffix) = name.strip_prefix(prefix) else {
            return false;
        };
        if suffix.parse::<usize>().ok() != Some(index)
            || (suffix.len() > 1 && suffix.starts_with('0'))
        {
            return false;
        }
        count += 1;
    }
    count == expected
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

fn attribute_u64_list<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<&'a [u64], TrainBackendError> {
    match request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .map(|attribute| attribute.value)
    {
        Some(TrainAttributeValueV1::U64List(value)) => Ok(value),
        _ => Err(attribute_type(name, "u64_list")),
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
