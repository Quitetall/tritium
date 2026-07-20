//! Portable-training lifecycle adapter for the native wgpu device.

use tritium_spec::{
    BackendError, TrainAttributeValueV1, TrainBackendError, TrainBackendV1, TrainBufferDataMutV1,
    TrainBufferDataRefV1, TrainCapabilitiesV1, TrainDTypeV1, TrainExecutionV1, TrainLimitsV1,
    TrainOperationErrorV1, TrainOutputV1, TrainReceiptV1, TrainRequestV1, TrainingOpManifestV1,
    train_output_digest_v1, train_request_digest_v1,
};

use crate::{
    WgpuBackend,
    backend::{AdamWParams, AdamWScalars},
};

const OPERATIONS: &[&str] = &[
    "graph.ste_surrogate",
    "graph.lsq_ste",
    "graph.dense_matmul",
    "graph.ternary_matmul",
    "graph.embedding_gather",
    "graph.transpose",
    "graph.slice_cols",
    "graph.concat_cols",
    "graph.detach",
    "graph.scale_const",
    "graph.bias",
    "graph.add",
    "graph.mul",
    "graph.relu2",
    "graph.silu",
    "graph.causal_mask",
    "graph.rmsnorm",
    "graph.softmax",
    "loss.mse",
    "optimizer.sgd",
    "optimizer.adamw",
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
                vec![
                    self.backend
                        .pointwise(first, second, first, 0, 0.0, auxiliary),
                ]
            }
            ("graph.detach", TrainExecutionV1::Vjp) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 1, 0.0, auxiliary),
                ]
            }
            ("graph.scale_const", _) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 2, scalar, auxiliary),
                ]
            }
            ("graph.add", TrainExecutionV1::Forward) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 3, 0.0, auxiliary),
                ]
            }
            ("graph.add", TrainExecutionV1::Vjp) => vec![
                self.backend
                    .pointwise(first, second, first, 0, 0.0, auxiliary),
                self.backend
                    .pointwise(first, second, first, 0, 0.0, auxiliary),
            ],
            ("graph.mul", TrainExecutionV1::Forward) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 4, 0.0, auxiliary),
                ]
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
                    self.backend
                        .pointwise(gradient, right, gradient, 4, 0.0, auxiliary),
                    self.backend
                        .pointwise(gradient, left, gradient, 4, 0.0, auxiliary),
                ]
            }
            ("graph.relu2", TrainExecutionV1::Forward) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 5, 0.0, auxiliary),
                ]
            }
            ("graph.relu2", TrainExecutionV1::Vjp) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 6, 0.0, auxiliary),
                ]
            }
            ("graph.silu", TrainExecutionV1::Forward) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 7, 0.0, auxiliary),
                ]
            }
            ("graph.silu", TrainExecutionV1::Vjp) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 8, 0.0, auxiliary),
                ]
            }
            ("graph.causal_mask", TrainExecutionV1::Forward) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 9, 0.0, auxiliary),
                ]
            }
            ("graph.causal_mask", TrainExecutionV1::Vjp) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 10, 0.0, auxiliary),
                ]
            }
            ("graph.softmax", TrainExecutionV1::Forward) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 11, 0.0, auxiliary),
                ]
            }
            ("graph.softmax", TrainExecutionV1::Vjp) => {
                vec![
                    self.backend
                        .pointwise(first, second, first, 12, 0.0, auxiliary),
                ]
            }
            _ => unreachable!(),
        };
        for (name, result) in output_names.iter().zip(results) {
            let result = result.map_err(wgpu_error)?;
            output_f32(output, name, shape, first.len())?.copy_from_slice(&result);
        }
        Ok(())
    }

    fn execute_rmsnorm(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_names, output_names): (&[&str], &[&str]) = match request.execution {
            TrainExecutionV1::Forward => (&["x", "weight"], &["result"]),
            TrainExecutionV1::Vjp => (&["x", "weight", "grad_output"], &["grad_x", "grad_weight"]),
            _ => return Err(invariant("RMSNorm received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            input_names,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["rows", "cols", "eps"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            output_names,
            "outputs",
        )?;
        let rows = attribute_u64(request, "rows")?;
        let cols = attribute_u64(request, "cols")?;
        let eps = attribute_f32(request, "eps")?;
        if cols == 0 {
            return Err(attribute_value("cols", "positive"));
        }
        if eps < 0.0 {
            return Err(attribute_value("eps", "nonnegative"));
        }
        if rows > u32::MAX as u64 || cols > u32::MAX as u64 {
            return Err(shape_error());
        }
        let matrix_shape = [rows, cols];
        let weight_shape = [cols];
        let (x_shape, x) = input_f32(request, "x")?;
        let (actual_weight_shape, weight) = input_f32(request, "weight")?;
        if x_shape != matrix_shape || actual_weight_shape != weight_shape {
            return Err(shape_error());
        }
        require_finite("x", x)?;
        require_finite("weight", weight)?;
        let gradient = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != matrix_shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient
        } else {
            x
        };
        if request.execution == TrainExecutionV1::Forward {
            let result = self
                .backend
                .pointwise(x, weight, gradient, 13, eps, cols as u32)
                .map_err(wgpu_error)?;
            output_f32(output, "result", &matrix_shape, x.len())?.copy_from_slice(&result);
        } else {
            let grad_x = self
                .backend
                .pointwise(x, weight, gradient, 14, eps, cols as u32)
                .map_err(wgpu_error)?;
            let grad_weight_full = self
                .backend
                .pointwise(x, weight, gradient, 15, eps, cols as u32)
                .map_err(wgpu_error)?;
            output_f32(output, "grad_x", &matrix_shape, x.len())?.copy_from_slice(&grad_x);
            output_f32(output, "grad_weight", &weight_shape, weight.len())?
                .copy_from_slice(&grad_weight_full[..weight.len()]);
        }
        Ok(())
    }

    fn execute_mse(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_names, output_names): (&[&str], &[&str]) = match request.execution {
            TrainExecutionV1::Forward => (&["prediction", "target"], &["result"]),
            TrainExecutionV1::Vjp => (
                &["prediction", "target", "grad_output"],
                &["grad_prediction"],
            ),
            _ => return Err(invariant("MSE received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            input_names,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &[],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            output_names,
            "outputs",
        )?;
        let (shape, prediction) = input_f32(request, "prediction")?;
        let (target_shape, target) = input_f32(request, "target")?;
        if shape != target_shape || prediction.is_empty() {
            return Err(shape_error());
        }
        require_finite("prediction", prediction)?;
        require_finite("target", target)?;
        let grad_output = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if !gradient_shape.is_empty() || gradient.len() != 1 {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient[0]
        } else {
            0.0
        };
        let operation = if request.execution == TrainExecutionV1::Forward {
            16
        } else {
            17
        };
        let result = self
            .backend
            .pointwise(prediction, target, prediction, operation, grad_output, 0)
            .map_err(wgpu_error)?;
        if request.execution == TrainExecutionV1::Forward {
            output_f32(output, "result", &[], 1)?[0] = result[0];
        } else {
            output_f32(output, "grad_prediction", shape, prediction.len())?
                .copy_from_slice(&result);
        }
        Ok(())
    }

    fn execute_bias(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_names, output_names): (&[&str], &[&str]) = match request.execution {
            TrainExecutionV1::Forward => (&["x", "bias"], &["result"]),
            TrainExecutionV1::Vjp => (&["x", "bias", "grad_output"], &["grad_x", "grad_bias"]),
            _ => return Err(invariant("bias received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            input_names,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["rows", "cols"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            output_names,
            "outputs",
        )?;
        let rows = attribute_u64(request, "rows")?;
        let cols = attribute_u64(request, "cols")?;
        if rows == 0 || cols == 0 || rows > u32::MAX as u64 || cols > u32::MAX as u64 {
            return Err(shape_error());
        }
        let matrix_shape = [rows, cols];
        let bias_shape = [cols];
        let (x_shape, x) = input_f32(request, "x")?;
        let (actual_bias_shape, bias) = input_f32(request, "bias")?;
        if x_shape != matrix_shape || actual_bias_shape != bias_shape {
            return Err(shape_error());
        }
        require_finite("x", x)?;
        require_finite("bias", bias)?;
        let gradient = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != matrix_shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient
        } else {
            x
        };
        if request.execution == TrainExecutionV1::Forward {
            let result = self
                .backend
                .pointwise(x, bias, gradient, 18, 0.0, cols as u32)
                .map_err(wgpu_error)?;
            output_f32(output, "result", &matrix_shape, x.len())?.copy_from_slice(&result);
        } else {
            let grad_x = self
                .backend
                .pointwise(x, bias, gradient, 19, 0.0, cols as u32)
                .map_err(wgpu_error)?;
            let grad_bias_full = self
                .backend
                .pointwise(x, bias, gradient, 20, 0.0, cols as u32)
                .map_err(wgpu_error)?;
            output_f32(output, "grad_x", &matrix_shape, x.len())?.copy_from_slice(&grad_x);
            output_f32(output, "grad_bias", &bias_shape, bias.len())?
                .copy_from_slice(&grad_bias_full[..bias.len()]);
        }
        Ok(())
    }

    fn execute_sgd(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        if request.execution != TrainExecutionV1::Step {
            return Err(invariant("SGD received an illegal phase"));
        }
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            &["parameter", "gradient"],
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["step", "lr"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &["parameter"],
            "outputs",
        )?;
        let step = attribute_u64(request, "step")?;
        let learning_rate = attribute_f32(request, "lr")?;
        if step == 0 {
            return Err(attribute_value("step", "one_based"));
        }
        if learning_rate < 0.0 {
            return Err(attribute_value("lr", "nonnegative"));
        }
        let (shape, parameter) = input_f32(request, "parameter")?;
        let (gradient_shape, gradient) = input_f32(request, "gradient")?;
        if shape != gradient_shape {
            return Err(shape_error());
        }
        require_finite("parameter", parameter)?;
        require_finite("gradient", gradient)?;
        let updated = self
            .backend
            .pointwise(parameter, gradient, parameter, 21, learning_rate, 0)
            .map_err(wgpu_error)?;
        output_f32(output, "parameter", shape, parameter.len())?.copy_from_slice(&updated);
        Ok(())
    }

    fn execute_transpose(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_name, output_name) = match request.execution {
            TrainExecutionV1::Forward => ("x", "result"),
            TrainExecutionV1::Vjp => ("grad_output", "grad_x"),
            _ => return Err(invariant("transpose received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            &[input_name],
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["rows", "cols"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
        let rows = attribute_u64(request, "rows")?;
        let cols = attribute_u64(request, "cols")?;
        if rows == 0 || cols == 0 || rows > u32::MAX as u64 || cols > u32::MAX as u64 {
            return Err(shape_error());
        }
        let input_shape = if request.execution == TrainExecutionV1::Forward {
            [rows, cols]
        } else {
            [cols, rows]
        };
        let output_shape = if request.execution == TrainExecutionV1::Forward {
            [cols, rows]
        } else {
            [rows, cols]
        };
        let (actual_shape, input) = input_f32(request, input_name)?;
        if actual_shape != input_shape {
            return Err(shape_error());
        }
        require_finite(input_name, input)?;
        let operation = if request.execution == TrainExecutionV1::Forward {
            22
        } else {
            23
        };
        let result = self
            .backend
            .pointwise(input, input, input, operation, 0.0, cols as u32)
            .map_err(wgpu_error)?;
        output_f32(output, output_name, &output_shape, input.len())?.copy_from_slice(&result);
        Ok(())
    }

    fn execute_slice_cols(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_name, output_name) = match request.execution {
            TrainExecutionV1::Forward => ("x", "result"),
            TrainExecutionV1::Vjp => ("grad_output", "grad_x"),
            _ => return Err(invariant("column slice received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            &[input_name],
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["rows", "cols", "start", "len"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
        let rows = attribute_u64(request, "rows")?;
        let cols = attribute_u64(request, "cols")?;
        let start = attribute_u64(request, "start")?;
        let len = attribute_u64(request, "len")?;
        if start.checked_add(len).is_none_or(|end| end > cols) {
            return Err(attribute_value("start", "slice_bounds"));
        }
        if rows > u32::MAX as u64
            || cols > u32::MAX as u64
            || start > u32::MAX as u64
            || len > u32::MAX as u64
        {
            return Err(shape_error());
        }
        let input_shape = if request.execution == TrainExecutionV1::Forward {
            [rows, cols]
        } else {
            [rows, len]
        };
        let output_shape = if request.execution == TrainExecutionV1::Forward {
            [rows, len]
        } else {
            [rows, cols]
        };
        let (actual_shape, input) = input_f32(request, input_name)?;
        if actual_shape != input_shape {
            return Err(shape_error());
        }
        require_finite(input_name, input)?;
        let output_len =
            usize::try_from(rows.checked_mul(output_shape[1]).ok_or_else(shape_error)?)
                .map_err(|_| shape_error())?;
        let operation = if request.execution == TrainExecutionV1::Forward {
            24
        } else {
            25
        };
        let result = self
            .backend
            .pointwise_sized(
                input,
                input,
                input,
                operation,
                0.0,
                cols as u32,
                start as u32,
                len as u32,
                output_len,
            )
            .map_err(wgpu_error)?;
        output_f32(output, output_name, &output_shape, output_len)?.copy_from_slice(&result);
        Ok(())
    }

    fn execute_dense_matmul(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_names, output_names): (&[&str], &[&str]) = match request.execution {
            TrainExecutionV1::Forward => (&["x", "weight"], &["result"]),
            TrainExecutionV1::Vjp => (&["x", "weight", "grad_output"], &["grad_x", "grad_weight"]),
            _ => return Err(invariant("dense matmul received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            input_names,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["m", "n", "k"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            output_names,
            "outputs",
        )?;
        let m = attribute_u64(request, "m")?;
        let n = attribute_u64(request, "n")?;
        let k = attribute_u64(request, "k")?;
        if m > u32::MAX as u64 || n > u32::MAX as u64 || k > u32::MAX as u64 {
            return Err(shape_error());
        }
        let x_shape = [m, k];
        let weight_shape = [n, k];
        let result_shape = [m, n];
        let (actual_x_shape, x) = input_f32(request, "x")?;
        let (actual_weight_shape, weight) = input_f32(request, "weight")?;
        if actual_x_shape != x_shape || actual_weight_shape != weight_shape {
            return Err(shape_error());
        }
        require_finite("x", x)?;
        require_finite("weight", weight)?;
        let gradient = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != result_shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient
        } else {
            x
        };
        if request.execution == TrainExecutionV1::Forward {
            let result_len = usize::try_from(m.checked_mul(n).ok_or_else(shape_error)?)
                .map_err(|_| shape_error())?;
            let result = self
                .backend
                .pointwise_sized(
                    x, weight, gradient, 26, 0.0, m as u32, n as u32, k as u32, result_len,
                )
                .map_err(wgpu_error)?;
            output_f32(output, "result", &result_shape, result_len)?.copy_from_slice(&result);
        } else {
            let grad_x = self
                .backend
                .pointwise_sized(
                    x,
                    weight,
                    gradient,
                    27,
                    0.0,
                    m as u32,
                    n as u32,
                    k as u32,
                    x.len(),
                )
                .map_err(wgpu_error)?;
            let grad_weight = self
                .backend
                .pointwise_sized(
                    x,
                    weight,
                    gradient,
                    28,
                    0.0,
                    m as u32,
                    n as u32,
                    k as u32,
                    weight.len(),
                )
                .map_err(wgpu_error)?;
            output_f32(output, "grad_x", &x_shape, x.len())?.copy_from_slice(&grad_x);
            output_f32(output, "grad_weight", &weight_shape, weight.len())?
                .copy_from_slice(&grad_weight);
        }
        Ok(())
    }

    fn execute_ternary_matmul(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_names, output_names): (&[&str], &[&str]) = match request.execution {
            TrainExecutionV1::Forward => (&["activation", "weight", "scale"], &["result"]),
            TrainExecutionV1::Vjp => (
                &["activation", "weight", "scale", "grad_output"],
                &["grad_activation", "grad_weight", "grad_scale"],
            ),
            _ => return Err(invariant("ternary matmul received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            input_names,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["m", "n", "k"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            output_names,
            "outputs",
        )?;
        let m = attribute_u64(request, "m")?;
        let n = attribute_u64(request, "n")?;
        let k = attribute_u64(request, "k")?;
        if m > u32::MAX as u64 || n > u32::MAX as u64 || k > u32::MAX as u64 {
            return Err(shape_error());
        }
        let activation_shape = [m, k];
        let weight_shape = [n, k];
        let scale_shape = [n];
        let result_shape = [m, n];
        let (actual_activation_shape, activation) = input_f32(request, "activation")?;
        let (actual_weight_shape, weight) = input_f32(request, "weight")?;
        let (actual_scale_shape, scale) = input_f32(request, "scale")?;
        if actual_activation_shape != activation_shape
            || actual_weight_shape != weight_shape
            || actual_scale_shape != scale_shape
        {
            return Err(shape_error());
        }
        require_finite("activation", activation)?;
        require_finite("weight", weight)?;
        require_finite("scale", scale)?;
        let result_len = usize::try_from(m.checked_mul(n).ok_or_else(shape_error)?)
            .map_err(|_| shape_error())?;
        if request.execution == TrainExecutionV1::Forward {
            let result = self
                .backend
                .pointwise_sized(
                    activation, weight, scale, 29, 0.0, m as u32, n as u32, k as u32, result_len,
                )
                .map_err(wgpu_error)?;
            output_f32(output, "result", &result_shape, result_len)?.copy_from_slice(&result);
        } else {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != result_shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            let grad_activation = self
                .backend
                .pointwise_sized(
                    gradient,
                    weight,
                    scale,
                    30,
                    0.0,
                    m as u32,
                    n as u32,
                    k as u32,
                    activation.len(),
                )
                .map_err(wgpu_error)?;
            let grad_weight = self
                .backend
                .pointwise_sized(
                    gradient,
                    activation,
                    scale,
                    31,
                    0.0,
                    m as u32,
                    n as u32,
                    k as u32,
                    weight.len(),
                )
                .map_err(wgpu_error)?;
            let grad_scale = self
                .backend
                .pointwise_sized(
                    gradient,
                    activation,
                    weight,
                    32,
                    0.0,
                    m as u32,
                    n as u32,
                    k as u32,
                    scale.len(),
                )
                .map_err(wgpu_error)?;
            output_f32(
                output,
                "grad_activation",
                &activation_shape,
                activation.len(),
            )?
            .copy_from_slice(&grad_activation);
            output_f32(output, "grad_weight", &weight_shape, weight.len())?
                .copy_from_slice(&grad_weight);
            output_f32(output, "grad_scale", &scale_shape, scale.len())?
                .copy_from_slice(&grad_scale);
        }
        Ok(())
    }

    fn execute_concat_cols(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["rows", "lens"],
            "attributes",
        )?;
        let rows = attribute_u64(request, "rows")?;
        let lens = attribute_u64_list(request, "lens")?;
        if lens.is_empty() {
            return Err(attribute_value("lens", "nonempty"));
        }
        if rows > u32::MAX as u64 || lens.iter().any(|&length| length > u32::MAX as u64) {
            return Err(shape_error());
        }
        let total = lens.iter().try_fold(0_u64, |sum, &length| {
            sum.checked_add(length).ok_or_else(shape_error)
        })?;
        if total > u32::MAX as u64 {
            return Err(shape_error());
        }
        let lengths = lens
            .iter()
            .map(|&length| usize::try_from(length).map_err(|_| shape_error()))
            .collect::<Result<Vec<_>, _>>()?;
        let rows_usize = usize::try_from(rows).map_err(|_| shape_error())?;
        match request.execution {
            TrainExecutionV1::Forward => {
                if request.inputs.len() != lens.len()
                    || request
                        .inputs
                        .iter()
                        .enumerate()
                        .any(|(index, buffer)| buffer.name != format!("part.{index}"))
                {
                    return Err(TrainBackendError::InvalidOperation(
                        TrainOperationErrorV1::Roles {
                            namespace: "inputs",
                        },
                    ));
                }
                require_names(
                    output.buffers.iter().map(|buffer| buffer.name),
                    &["result"],
                    "outputs",
                )?;
                let mut parts = Vec::with_capacity(lens.len());
                for (index, (&length, buffer)) in lens.iter().zip(request.inputs.iter()).enumerate()
                {
                    if buffer.shape != [rows, length] {
                        return Err(shape_error());
                    }
                    let (_, values) = input_f32(request, &format!("part.{index}"))?;
                    require_finite(buffer.name, values)?;
                    parts.push(values);
                }
                let result = self
                    .backend
                    .concat_cols(&parts, rows_usize, &lengths)
                    .map_err(wgpu_error)?;
                let result_len = usize::try_from(rows.checked_mul(total).ok_or_else(shape_error)?)
                    .map_err(|_| shape_error())?;
                output_f32(output, "result", &[rows, total], result_len)?.copy_from_slice(&result);
            }
            TrainExecutionV1::Vjp => {
                require_names(
                    request.inputs.iter().map(|buffer| buffer.name),
                    &["grad_output"],
                    "inputs",
                )?;
                if output.buffers.len() != lens.len()
                    || output
                        .buffers
                        .iter()
                        .enumerate()
                        .any(|(index, buffer)| buffer.name != format!("grad_part.{index}"))
                {
                    return Err(TrainBackendError::InvalidOperation(
                        TrainOperationErrorV1::Roles {
                            namespace: "outputs",
                        },
                    ));
                }
                let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
                if gradient_shape != [rows, total] {
                    return Err(shape_error());
                }
                require_finite("grad_output", gradient)?;
                let mut start = 0_u64;
                for (index, &length) in lens.iter().enumerate() {
                    let output_len =
                        usize::try_from(rows.checked_mul(length).ok_or_else(shape_error)?)
                            .map_err(|_| shape_error())?;
                    let part = self
                        .backend
                        .pointwise_sized(
                            gradient,
                            gradient,
                            gradient,
                            24,
                            0.0,
                            total as u32,
                            start as u32,
                            length as u32,
                            output_len,
                        )
                        .map_err(wgpu_error)?;
                    output_f32(
                        output,
                        &format!("grad_part.{index}"),
                        &[rows, length],
                        output_len,
                    )?
                    .copy_from_slice(&part);
                    start += length;
                }
            }
            _ => return Err(invariant("column concat received an illegal phase")),
        }
        Ok(())
    }

    fn execute_embedding(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_names, output_names): (&[&str], &[&str]) = match request.execution {
            TrainExecutionV1::Forward => (&["weight", "tokens"], &["result"]),
            TrainExecutionV1::Vjp => (&["weight", "tokens", "grad_output"], &["grad_weight"]),
            _ => return Err(invariant("embedding gather received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            input_names,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["vocab", "n_embd"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            output_names,
            "outputs",
        )?;
        let vocab = attribute_u64(request, "vocab")?;
        let width = attribute_u64(request, "n_embd")?;
        if vocab > u32::MAX as u64 || width > u32::MAX as u64 {
            return Err(shape_error());
        }
        let (weight_shape, weight) = input_f32(request, "weight")?;
        let (token_shape, tokens) = input_u32(request, "tokens")?;
        let sequence = tokens.len() as u64;
        if weight_shape != [vocab, width] || token_shape != [sequence] {
            return Err(shape_error());
        }
        if tokens.iter().any(|&token| token as u64 >= vocab) {
            return Err(shape_error());
        }
        require_finite("weight", weight)?;
        let result_shape = [sequence, width];
        let gradient = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != result_shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient
        } else {
            &[]
        };
        let result = self
            .backend
            .embedding(
                weight,
                tokens,
                gradient,
                vocab as usize,
                width as usize,
                request.execution == TrainExecutionV1::Vjp,
            )
            .map_err(wgpu_error)?;
        if request.execution == TrainExecutionV1::Forward {
            output_f32(output, "result", &result_shape, result.len())?.copy_from_slice(&result);
        } else {
            output_f32(output, "grad_weight", &[vocab, width], weight.len())?
                .copy_from_slice(&result);
        }
        Ok(())
    }

    fn execute_ste_surrogate(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_names, output_names): (&[&str], &[&str]) = match request.execution {
            TrainExecutionV1::Forward => (&["weight", "scale"], &["result"]),
            TrainExecutionV1::Vjp => (
                &["weight", "scale", "grad_output"],
                &["grad_weight", "grad_scale"],
            ),
            _ => return Err(invariant("STE surrogate received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            input_names,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["rows", "cols"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            output_names,
            "outputs",
        )?;
        let rows = attribute_u64(request, "rows")?;
        let cols = attribute_u64(request, "cols")?;
        if rows > u32::MAX as u64 || cols == 0 || cols > u32::MAX as u64 {
            return Err(shape_error());
        }
        let weight_shape = [rows, cols];
        let scale_shape = [rows];
        let (actual_weight_shape, weight) = input_f32(request, "weight")?;
        let (actual_scale_shape, scale) = input_f32(request, "scale")?;
        if actual_weight_shape != weight_shape || actual_scale_shape != scale_shape {
            return Err(shape_error());
        }
        require_finite("weight", weight)?;
        require_finite("scale", scale)?;
        let gradient = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != weight_shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient
        } else {
            weight
        };
        if request.execution == TrainExecutionV1::Forward {
            let result = self
                .backend
                .pointwise(weight, scale, gradient, 33, 0.0, cols as u32)
                .map_err(wgpu_error)?;
            output_f32(output, "result", &weight_shape, weight.len())?.copy_from_slice(&result);
        } else {
            let grad_weight = self
                .backend
                .pointwise(weight, scale, gradient, 34, 0.0, cols as u32)
                .map_err(wgpu_error)?;
            let grad_scale = self
                .backend
                .pointwise(scale, scale, scale, 1, 0.0, 0)
                .map_err(wgpu_error)?;
            output_f32(output, "grad_weight", &weight_shape, weight.len())?
                .copy_from_slice(&grad_weight);
            output_f32(output, "grad_scale", &scale_shape, scale.len())?
                .copy_from_slice(&grad_scale);
        }
        Ok(())
    }

    fn execute_lsq(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_names, output_names): (&[&str], &[&str]) = match request.execution {
            TrainExecutionV1::Forward => (&["weight", "alpha"], &["result"]),
            TrainExecutionV1::Vjp => (
                &["weight", "alpha", "grad_output"],
                &["grad_weight", "grad_alpha"],
            ),
            _ => return Err(invariant("LSQ received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            input_names,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["rows", "cols"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            output_names,
            "outputs",
        )?;
        let rows = attribute_u64(request, "rows")?;
        let cols = attribute_u64(request, "cols")?;
        if cols == 0 {
            return Err(attribute_value("cols", "positive"));
        }
        if rows > u32::MAX as u64 || cols > u32::MAX as u64 {
            return Err(shape_error());
        }
        let weight_shape = [rows, cols];
        let alpha_shape = [rows];
        let (actual_weight_shape, weight) = input_f32(request, "weight")?;
        let (actual_alpha_shape, alpha) = input_f32(request, "alpha")?;
        if actual_weight_shape != weight_shape || actual_alpha_shape != alpha_shape {
            return Err(shape_error());
        }
        require_finite("weight", weight)?;
        require_finite("alpha", alpha)?;
        let gradient = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != weight_shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient
        } else {
            weight
        };
        if request.execution == TrainExecutionV1::Forward {
            let result = self
                .backend
                .pointwise(weight, alpha, gradient, 35, 0.0, cols as u32)
                .map_err(wgpu_error)?;
            output_f32(output, "result", &weight_shape, weight.len())?.copy_from_slice(&result);
        } else {
            let grad_weight = self
                .backend
                .pointwise(weight, alpha, gradient, 36, 0.0, cols as u32)
                .map_err(wgpu_error)?;
            let grad_alpha = self
                .backend
                .pointwise_sized(
                    weight,
                    alpha,
                    gradient,
                    37,
                    0.0,
                    cols as u32,
                    0,
                    0,
                    alpha.len(),
                )
                .map_err(wgpu_error)?;
            output_f32(output, "grad_weight", &weight_shape, weight.len())?
                .copy_from_slice(&grad_weight);
            output_f32(output, "grad_alpha", &alpha_shape, alpha.len())?
                .copy_from_slice(&grad_alpha);
        }
        Ok(())
    }

    fn execute_adamw(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        if request.execution != TrainExecutionV1::Step {
            return Err(invariant("AdamW received an illegal phase"));
        }
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            &["parameter", "gradient", "moment1", "moment2"],
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["step", "lr", "beta1", "beta2", "eps", "weight_decay"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &["parameter", "moment1", "moment2"],
            "outputs",
        )?;
        let step = attribute_u64(request, "step")?;
        let learning_rate = attribute_f32(request, "lr")?;
        let beta1 = attribute_f32(request, "beta1")?;
        let beta2 = attribute_f32(request, "beta2")?;
        let epsilon = attribute_f32(request, "eps")?;
        let weight_decay = attribute_f32(request, "weight_decay")?;
        if step == 0 {
            return Err(attribute_value("step", "one_based"));
        }
        if learning_rate < 0.0 {
            return Err(attribute_value("lr", "nonnegative"));
        }
        if !(0.0..1.0).contains(&beta1) {
            return Err(attribute_value("beta1", "unit_interval_open"));
        }
        if !(0.0..1.0).contains(&beta2) {
            return Err(attribute_value("beta2", "unit_interval_open"));
        }
        if epsilon <= 0.0 {
            return Err(attribute_value("eps", "positive"));
        }
        if weight_decay < 0.0 {
            return Err(attribute_value("weight_decay", "nonnegative"));
        }
        let (shape, parameter) = input_f32(request, "parameter")?;
        let (gradient_shape, gradient) = input_f32(request, "gradient")?;
        let (moment1_shape, moment1) = input_f32(request, "moment1")?;
        let (moment2_shape, moment2) = input_f32(request, "moment2")?;
        if parameter.is_empty()
            || gradient_shape != shape
            || moment1_shape != shape
            || moment2_shape != shape
            || parameter.len() > u32::MAX as usize
        {
            return Err(shape_error());
        }
        require_finite("parameter", parameter)?;
        require_finite("gradient", gradient)?;
        require_finite("moment1", moment1)?;
        require_finite("moment2", moment2)?;
        let exponent = i32::try_from(step).unwrap_or(i32::MAX);
        let correction1 = 1.0 - beta1.powi(exponent);
        let correction2 = 1.0 - beta2.powi(exponent);
        let shrink = 1.0 - learning_rate * weight_decay;
        let params = AdamWParams::new(
            parameter.len() as u32,
            AdamWScalars {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                correction1,
                correction2,
                shrink,
            },
        );
        let (updated_parameter, updated_moment1, updated_moment2) = self
            .backend
            .adamw(parameter, gradient, moment1, moment2, params)
            .map_err(wgpu_error)?;
        output_f32(output, "parameter", shape, parameter.len())?
            .copy_from_slice(&updated_parameter);
        output_f32(output, "moment1", shape, parameter.len())?.copy_from_slice(&updated_moment1);
        output_f32(output, "moment2", shape, parameter.len())?.copy_from_slice(&updated_moment2);
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
            dtypes: vec![TrainDTypeV1::F32, TrainDTypeV1::U32, TrainDTypeV1::Bytes],
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
        } else if request.operation == "graph.rmsnorm" {
            self.execute_rmsnorm(&request, output)?;
            0
        } else if request.operation == "loss.mse" {
            self.execute_mse(&request, output)?;
            0
        } else if request.operation == "graph.bias" {
            self.execute_bias(&request, output)?;
            0
        } else if request.operation == "optimizer.sgd" {
            self.execute_sgd(&request, output)?;
            0
        } else if request.operation == "graph.transpose" {
            self.execute_transpose(&request, output)?;
            0
        } else if request.operation == "graph.slice_cols" {
            self.execute_slice_cols(&request, output)?;
            0
        } else if request.operation == "graph.dense_matmul" {
            self.execute_dense_matmul(&request, output)?;
            0
        } else if request.operation == "graph.ternary_matmul" {
            self.execute_ternary_matmul(&request, output)?;
            0
        } else if request.operation == "graph.concat_cols" {
            self.execute_concat_cols(&request, output)?;
            0
        } else if request.operation == "graph.embedding_gather" {
            self.execute_embedding(&request, output)?;
            0
        } else if request.operation == "graph.ste_surrogate" {
            self.execute_ste_surrogate(&request, output)?;
            0
        } else if request.operation == "graph.lsq_ste" {
            self.execute_lsq(&request, output)?;
            0
        } else if request.operation == "optimizer.adamw" {
            self.execute_adamw(&request, output)?;
            0
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

fn input_u32<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<(&'a [u64], &'a [u32]), TrainBackendError> {
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
        TrainBufferDataRefV1::U32(data) => Ok((buffer.shape, data)),
        TrainBufferDataRefV1::F32(_) => Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::DType {
                name: name.to_owned(),
                expected: TrainDTypeV1::U32,
                got: TrainDTypeV1::F32,
            },
        )),
        TrainBufferDataRefV1::Bytes(_) => Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::DType {
                name: name.to_owned(),
                expected: TrainDTypeV1::U32,
                got: TrainDTypeV1::Bytes,
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

fn attribute_u64_list<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<&'a [u64], TrainBackendError> {
    let attribute = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or({
            TrainBackendError::InvalidOperation(TrainOperationErrorV1::Roles {
                namespace: "attributes",
            })
        })?;
    let TrainAttributeValueV1::U64List(value) = attribute.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "u64_list",
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

fn attribute_value(name: &str, constraint: &'static str) -> TrainBackendError {
    TrainBackendError::InvalidOperation(TrainOperationErrorV1::AttributeValue {
        name: name.to_owned(),
        constraint,
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
