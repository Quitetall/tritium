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
const OPERATIONS: &[&str] = &[
    "graph.dense_matmul",
    "graph.transpose",
    "graph.slice_cols",
    "graph.concat_cols",
    "graph.scale_const",
    "graph.add",
    "graph.mul",
    "graph.silu",
    "graph.rmsnorm",
    "graph.softmax",
    "graph.causal_mask",
];
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

    fn execute_transpose(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_name, input_shape, output_name, output_shape) = match request.execution {
            TrainExecutionV1::Forward => ("x", [0, 1], "result", [1, 0]),
            TrainExecutionV1::Vjp => ("grad_output", [1, 0], "grad_x", [0, 1]),
            _ => return Err(invariant("transpose received an illegal execution phase")),
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
        let rows = attribute_usize(request, "rows")?;
        let cols = attribute_usize(request, "cols")?;
        let elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        let dimensions = [rows as u64, cols as u64];
        let expected_input = [dimensions[input_shape[0]], dimensions[input_shape[1]]];
        let expected_output = [dimensions[output_shape[0]], dimensions[output_shape[1]]];
        let input = input_f32(request, input_name, &expected_input)?;
        require_finite(input_name, input)?;
        let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
        let input_id = tape.leaf(input).map_err(cuda_error)?;
        let result_id = tape
            .transpose(
                input_id,
                expected_input[0] as usize,
                expected_input[1] as usize,
            )
            .map_err(cuda_error)?;
        let result = tape.value(result_id).map_err(cuda_error)?;
        output_f32(output, output_name, &expected_output, elements)?.copy_from_slice(&result);
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
            _ => return Err(invariant("slice_cols received an illegal execution phase")),
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
        let rows = attribute_usize(request, "rows")?;
        let cols = attribute_usize(request, "cols")?;
        let start = attribute_usize(request, "start")?;
        let len = attribute_usize(request, "len")?;
        if start.checked_add(len).is_none_or(|end| end > cols) {
            return Err(attribute_value("start", "slice_bounds"));
        }
        let input_shape = if request.execution == TrainExecutionV1::Forward {
            [rows as u64, cols as u64]
        } else {
            [rows as u64, len as u64]
        };
        let output_shape = if request.execution == TrainExecutionV1::Forward {
            [rows as u64, len as u64]
        } else {
            [rows as u64, cols as u64]
        };
        let input = input_f32(request, input_name, &input_shape)?;
        require_finite(input_name, input)?;
        let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
        let (result, result_len) = if request.execution == TrainExecutionV1::Forward {
            let input_id = tape.leaf(input).map_err(cuda_error)?;
            let result_id = tape
                .slice_cols(input_id, rows, cols, start, len)
                .map_err(cuda_error)?;
            (tape.value(result_id).map_err(cuda_error)?, rows * len)
        } else {
            let dummy = vec![0.0; rows.checked_mul(cols).ok_or_else(shape_error)?];
            let input_id = tape.leaf(&dummy).map_err(cuda_error)?;
            let result_id = tape
                .slice_cols(input_id, rows, cols, start, len)
                .map_err(cuda_error)?;
            let seed = self.backend.dev_upload(input).map_err(cuda_error)?;
            let gradients = tape
                .backward_retain(result_id, &seed, &[input_id])
                .map_err(cuda_error)?;
            (
                download_gradient(&self.backend, &gradients, input_id, dummy.len())?,
                dummy.len(),
            )
        };
        output_f32(output, output_name, &output_shape, result_len)?.copy_from_slice(&result);
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
        let rows = attribute_usize(request, "rows")?;
        let lens_u64 = attribute_u64_list(request, "lens")?;
        if lens_u64.is_empty() {
            return Err(attribute_value("lens", "nonempty"));
        }
        let lens: Vec<usize> = lens_u64
            .iter()
            .map(|&len| usize::try_from(len).map_err(|_| shape_error()))
            .collect::<Result<_, _>>()?;
        let total = lens.iter().try_fold(0_usize, |total, &len| {
            total.checked_add(len).ok_or_else(shape_error)
        })?;
        let output_elements = rows.checked_mul(total).ok_or_else(shape_error)?;
        let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
        match request.execution {
            TrainExecutionV1::Forward => {
                if request.inputs.len() != lens.len() {
                    return Err(roles("inputs"));
                }
                require_names(
                    output.buffers.iter().map(|buffer| buffer.name),
                    &["result"],
                    "outputs",
                )?;
                let mut ids = Vec::with_capacity(lens.len());
                for (index, &len) in lens.iter().enumerate() {
                    let expected_name = format!("part.{index}");
                    let buffer = request.inputs.get(index).ok_or_else(|| roles("inputs"))?;
                    if buffer.name != expected_name {
                        return Err(roles("inputs"));
                    }
                    let part = input_f32(request, &expected_name, &[rows as u64, len as u64])?;
                    require_finite(&expected_name, part)?;
                    ids.push(tape.leaf(part).map_err(cuda_error)?);
                }
                let result_id = tape.concat(&ids, rows, &lens).map_err(cuda_error)?;
                let result = tape.value(result_id).map_err(cuda_error)?;
                output_f32(
                    output,
                    "result",
                    &[rows as u64, total as u64],
                    output_elements,
                )?
                .copy_from_slice(&result);
            }
            TrainExecutionV1::Vjp => {
                require_names(
                    request.inputs.iter().map(|buffer| buffer.name),
                    &["grad_output"],
                    "inputs",
                )?;
                if output.buffers.len() != lens.len() {
                    return Err(roles("outputs"));
                }
                let grad_output = input_f32(request, "grad_output", &[rows as u64, total as u64])?;
                require_finite("grad_output", grad_output)?;
                let mut storage = Vec::with_capacity(lens.len());
                for &len in &lens {
                    storage.push(vec![0.0; rows.checked_mul(len).ok_or_else(shape_error)?]);
                }
                let ids: Vec<_> = storage
                    .iter()
                    .map(|part| tape.leaf(part).map_err(cuda_error))
                    .collect::<Result<_, _>>()?;
                let result_id = tape.concat(&ids, rows, &lens).map_err(cuda_error)?;
                let seed = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
                let gradients = tape
                    .backward_retain(result_id, &seed, &ids)
                    .map_err(cuda_error)?;
                for (index, (&id, &len)) in ids.iter().zip(&lens).enumerate() {
                    let expected_name = format!("grad_part.{index}");
                    if output.buffers[index].name != expected_name {
                        return Err(roles("outputs"));
                    }
                    let part_elements = rows.checked_mul(len).ok_or_else(shape_error)?;
                    let gradient = download_gradient(&self.backend, &gradients, id, part_elements)?;
                    output_f32(
                        output,
                        &expected_name,
                        &[rows as u64, len as u64],
                        part_elements,
                    )?
                    .copy_from_slice(&gradient);
                }
            }
            _ => return Err(invariant("concat_cols received an illegal execution phase")),
        }
        Ok(())
    }

    fn execute_scale_const(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_name, output_name) = match request.execution {
            TrainExecutionV1::Forward => ("x", "result"),
            TrainExecutionV1::Vjp => ("grad_output", "grad_x"),
            _ => return Err(invariant("scale_const received an illegal execution phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            &[input_name],
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["scale"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
        let scale = attribute_f32(request, "scale")?;
        let (shape, input) = input_f32_any_shape(request, input_name)?;
        require_finite(input_name, input)?;
        let shape = shape.to_vec();
        let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
        let input_id = tape.leaf(input).map_err(cuda_error)?;
        let result_id = tape.scale_const(input_id, scale).map_err(cuda_error)?;
        let result = tape.value(result_id).map_err(cuda_error)?;
        output_f32(output, output_name, &shape, result.len())?.copy_from_slice(&result);
        Ok(())
    }

    fn execute_add(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        match request.execution {
            TrainExecutionV1::Forward => {
                require_names(
                    request.inputs.iter().map(|buffer| buffer.name),
                    &["left", "right"],
                    "inputs",
                )?;
                require_names(
                    output.buffers.iter().map(|buffer| buffer.name),
                    &["result"],
                    "outputs",
                )?;
                require_names(request.attributes.iter().map(|a| a.name), &[], "attributes")?;
                let (shape, left) = input_f32_any_shape(request, "left")?;
                let (right_shape, right) = input_f32_any_shape(request, "right")?;
                if shape != right_shape {
                    return Err(shape_error());
                }
                require_finite("left", left)?;
                require_finite("right", right)?;
                let shape = shape.to_vec();
                let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
                let left_id = tape.leaf(left).map_err(cuda_error)?;
                let right_id = tape.leaf(right).map_err(cuda_error)?;
                let result_id = tape.add(left_id, right_id).map_err(cuda_error)?;
                let result = tape.value(result_id).map_err(cuda_error)?;
                output_f32(output, "result", &shape, result.len())?.copy_from_slice(&result);
            }
            TrainExecutionV1::Vjp => {
                require_names(
                    request.inputs.iter().map(|buffer| buffer.name),
                    &["grad_output"],
                    "inputs",
                )?;
                require_names(
                    output.buffers.iter().map(|buffer| buffer.name),
                    &["grad_left", "grad_right"],
                    "outputs",
                )?;
                require_names(request.attributes.iter().map(|a| a.name), &[], "attributes")?;
                let (shape, grad_output) = input_f32_any_shape(request, "grad_output")?;
                require_finite("grad_output", grad_output)?;
                let shape = shape.to_vec();
                let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
                let grad_id = tape.leaf(grad_output).map_err(cuda_error)?;
                let result_id = tape.scale_const(grad_id, 1.0).map_err(cuda_error)?;
                let gradient = tape.value(result_id).map_err(cuda_error)?;
                output_f32(output, "grad_left", &shape, gradient.len())?.copy_from_slice(&gradient);
                output_f32(output, "grad_right", &shape, gradient.len())?
                    .copy_from_slice(&gradient);
            }
            _ => return Err(invariant("add received an illegal execution phase")),
        }
        Ok(())
    }

    fn execute_mul(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["left", "right"],
            TrainExecutionV1::Vjp => &["left", "right", "grad_output"],
            _ => return Err(invariant("mul received an illegal execution phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(request.attributes.iter().map(|a| a.name), &[], "attributes")?;
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_left", "grad_right"],
            _ => unreachable!(),
        };
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            expected_outputs,
            "outputs",
        )?;
        let (shape, left) = input_f32_any_shape(request, "left")?;
        let (right_shape, right) = input_f32_any_shape(request, "right")?;
        if shape != right_shape {
            return Err(shape_error());
        }
        require_finite("left", left)?;
        require_finite("right", right)?;
        let shape = shape.to_vec();
        let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
        let left_id = tape.leaf(left).map_err(cuda_error)?;
        let right_id = tape.leaf(right).map_err(cuda_error)?;
        let result_id = tape.mul(left_id, right_id).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let result = tape.value(result_id).map_err(cuda_error)?;
            output_f32(output, "result", &shape, result.len())?.copy_from_slice(&result);
        } else {
            let (_, grad_output) = input_f32_any_shape(request, "grad_output")?;
            if grad_output.len() != left.len() {
                return Err(shape_error());
            }
            require_finite("grad_output", grad_output)?;
            let seed = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
            let gradients = tape
                .backward_retain(result_id, &seed, &[left_id, right_id])
                .map_err(cuda_error)?;
            let grad_left = download_gradient(&self.backend, &gradients, left_id, left.len())?;
            let grad_right = download_gradient(&self.backend, &gradients, right_id, right.len())?;
            output_f32(output, "grad_left", &shape, grad_left.len())?.copy_from_slice(&grad_left);
            output_f32(output, "grad_right", &shape, grad_right.len())?
                .copy_from_slice(&grad_right);
        }
        Ok(())
    }

    fn execute_silu(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x"],
            TrainExecutionV1::Vjp => &["x", "grad_output"],
            _ => return Err(invariant("silu received an illegal execution phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(request.attributes.iter().map(|a| a.name), &[], "attributes")?;
        let output_name = if request.execution == TrainExecutionV1::Forward {
            "result"
        } else {
            "grad_x"
        };
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
        let (shape, x) = input_f32_any_shape(request, "x")?;
        require_finite("x", x)?;
        let shape = shape.to_vec();
        let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
        let x_id = tape.leaf(x).map_err(cuda_error)?;
        let result_id = tape.silu(x_id).map_err(cuda_error)?;
        let result = if request.execution == TrainExecutionV1::Forward {
            tape.value(result_id).map_err(cuda_error)?
        } else {
            let (_, grad_output) = input_f32_any_shape(request, "grad_output")?;
            if grad_output.len() != x.len() {
                return Err(shape_error());
            }
            require_finite("grad_output", grad_output)?;
            let seed = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
            let gradients = tape
                .backward_retain(result_id, &seed, &[x_id])
                .map_err(cuda_error)?;
            download_gradient(&self.backend, &gradients, x_id, x.len())?
        };
        output_f32(output, output_name, &shape, result.len())?.copy_from_slice(&result);
        Ok(())
    }

    fn execute_rmsnorm(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x", "weight"],
            TrainExecutionV1::Vjp => &["x", "weight", "grad_output"],
            _ => return Err(invariant("rmsnorm received an illegal execution phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["rows", "cols", "eps"],
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
        let rows = attribute_usize(request, "rows")?;
        let cols = attribute_usize(request, "cols")?;
        let eps = attribute_f32(request, "eps")?;
        if rows == 0 || cols == 0 || !eps.is_finite() || eps <= 0.0 {
            return Err(attribute_value("eps", "positive"));
        }
        let elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        let x = input_f32(request, "x", &[rows as u64, cols as u64])?;
        let weight = input_f32(request, "weight", &[cols as u64])?;
        require_finite("x", x)?;
        require_finite("weight", weight)?;
        let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
        let x_id = tape.leaf(x).map_err(cuda_error)?;
        let weight_id = tape.leaf(weight).map_err(cuda_error)?;
        let result_id = tape
            .rmsnorm(x_id, weight_id, rows, cols, eps)
            .map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let result = tape.value(result_id).map_err(cuda_error)?;
            output_f32(output, "result", &[rows as u64, cols as u64], elements)?
                .copy_from_slice(&result);
        } else {
            let grad_output = input_f32(request, "grad_output", &[rows as u64, cols as u64])?;
            require_finite("grad_output", grad_output)?;
            let seed = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
            let gradients = tape
                .backward_retain(result_id, &seed, &[x_id, weight_id])
                .map_err(cuda_error)?;
            let grad_x = download_gradient(&self.backend, &gradients, x_id, elements)?;
            let grad_weight = download_gradient(&self.backend, &gradients, weight_id, cols)?;
            output_f32(output, "grad_x", &[rows as u64, cols as u64], elements)?
                .copy_from_slice(&grad_x);
            output_f32(output, "grad_weight", &[cols as u64], cols)?.copy_from_slice(&grad_weight);
        }
        Ok(())
    }

    fn execute_softmax(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x"],
            TrainExecutionV1::Vjp => &["x", "grad_output"],
            _ => return Err(invariant("softmax received an illegal execution phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["rows", "cols"],
            "attributes",
        )?;
        let output_name = if request.execution == TrainExecutionV1::Forward {
            "result"
        } else {
            "grad_x"
        };
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
        let rows = attribute_usize(request, "rows")?;
        let cols = attribute_usize(request, "cols")?;
        let elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        let shape = [rows as u64, cols as u64];
        let x = input_f32(request, "x", &shape)?;
        require_finite("x", x)?;
        let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
        let x_id = tape.leaf(x).map_err(cuda_error)?;
        let result_id = tape.softmax(x_id, rows, cols).map_err(cuda_error)?;
        let result = if request.execution == TrainExecutionV1::Forward {
            tape.value(result_id).map_err(cuda_error)?
        } else {
            let grad_output = input_f32(request, "grad_output", &shape)?;
            require_finite("grad_output", grad_output)?;
            let seed = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
            let gradients = tape
                .backward_retain(result_id, &seed, &[x_id])
                .map_err(cuda_error)?;
            download_gradient(&self.backend, &gradients, x_id, elements)?
        };
        output_f32(output, output_name, &shape, elements)?.copy_from_slice(&result);
        Ok(())
    }

    fn execute_causal_mask(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_name, output_name) = match request.execution {
            TrainExecutionV1::Forward => ("x", "result"),
            TrainExecutionV1::Vjp => ("grad_output", "grad_x"),
            _ => return Err(invariant("causal_mask received an illegal execution phase")),
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
        let rows = attribute_usize(request, "rows")?;
        let cols = attribute_usize(request, "cols")?;
        let elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        let shape = [rows as u64, cols as u64];
        let input = input_f32(request, input_name, &shape)?;
        require_finite(input_name, input)?;
        let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
        let input_id = tape.leaf(input).map_err(cuda_error)?;
        let result_id = tape.causal_mask(input_id, rows, cols).map_err(cuda_error)?;
        let result = if request.execution == TrainExecutionV1::Forward {
            tape.value(result_id).map_err(cuda_error)?
        } else {
            let seed = self.backend.dev_upload(input).map_err(cuda_error)?;
            let gradients = tape
                .backward_retain(result_id, &seed, &[input_id])
                .map_err(cuda_error)?;
            download_gradient(&self.backend, &gradients, input_id, elements)?
        };
        output_f32(output, output_name, &shape, elements)?.copy_from_slice(&result);
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
            "graph.transpose" => self.execute_transpose(&request, output)?,
            "graph.slice_cols" => self.execute_slice_cols(&request, output)?,
            "graph.concat_cols" => self.execute_concat_cols(&request, output)?,
            "graph.scale_const" => self.execute_scale_const(&request, output)?,
            "graph.add" => self.execute_add(&request, output)?,
            "graph.mul" => self.execute_mul(&request, output)?,
            "graph.silu" => self.execute_silu(&request, output)?,
            "graph.rmsnorm" => self.execute_rmsnorm(&request, output)?,
            "graph.softmax" => self.execute_softmax(&request, output)?,
            "graph.causal_mask" => self.execute_causal_mask(&request, output)?,
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

fn attribute_f32(request: &TrainRequestV1<'_>, name: &str) -> Result<f32, TrainBackendError> {
    let value = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or_else(|| roles("attributes"))?;
    let TrainAttributeValueV1::F32(value) = value.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "f32",
            },
        ));
    };
    Ok(value)
}

fn attribute_u64_list<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<&'a [u64], TrainBackendError> {
    let value = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or_else(|| roles("attributes"))?;
    let TrainAttributeValueV1::U64List(value) = value.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "u64_list",
            },
        ));
    };
    Ok(value)
}

fn input_f32_any_shape<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<(&'a [u64], &'a [f32]), TrainBackendError> {
    let buffer = request
        .inputs
        .iter()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| roles("inputs"))?;
    match buffer.data {
        TrainBufferDataRefV1::F32(data) => Ok((buffer.shape, data)),
        TrainBufferDataRefV1::U32(_) => Err(dtype_error(name, TrainDTypeV1::U32)),
        TrainBufferDataRefV1::Bytes(_) => Err(dtype_error(name, TrainDTypeV1::Bytes)),
    }
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

fn download_gradient(
    backend: &CudaBackend,
    gradients: &super::DeviceBackwardResult,
    id: usize,
    len: usize,
) -> Result<Vec<f32>, TrainBackendError> {
    let mut host = vec![0.0; len];
    backend
        .dev_download(
            gradients.grads[id]
                .as_ref()
                .ok_or_else(|| invariant("missing retained gradient"))?,
            &mut host,
        )
        .map_err(cuda_error)?;
    Ok(host)
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

fn cuda_error(error: BackendError) -> TrainBackendError {
    TrainBackendError::Backend {
        code: "cuda".to_owned(),
        message: error.to_string(),
    }
}
