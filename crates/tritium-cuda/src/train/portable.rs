//! Plan-0049 portable-training adapter backed by resident CUDA kernels.

use std::mem::size_of;

use tritium_core::GemmShape;
use tritium_spec::{
    BackendError, TernaryBackend, TrainAttributeValueV1, TrainBackendError, TrainBackendV1,
    TrainBufferDataMutV1, TrainBufferDataRefV1, TrainCapabilitiesV1, TrainDTypeV1,
    TrainExecutionV1, TrainLimitsV1, TrainOperationErrorV1, TrainOutputV1, TrainReceiptV1,
    TrainRequestV1, TrainingOpManifestV1, train_output_digest_v1, train_request_digest_v1,
};

use super::DeviceTape;
use crate::CudaBackend;

const BACKEND_FAMILY: &str = "cuda.portable.v1";
const MAX_SALT_PLANES: usize = 64;
const MAX_SALT_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
const OPERATIONS: &[&str] = &[
    "graph.ste_surrogate",
    "graph.salt_ste",
    "graph.lsq_ste",
    "graph.fsq",
    "graph.dense_matmul",
    "graph.ternary_matmul",
    "graph.transpose",
    "graph.embedding_gather",
    "graph.slice_cols",
    "graph.concat_cols",
    "graph.detach",
    "graph.scale_const",
    "graph.bias",
    "graph.add",
    "graph.mul",
    "graph.relu2",
    "graph.silu",
    "graph.rmsnorm",
    "graph.softmax",
    "graph.causal_mask",
    "graph.rope",
    "loss.mse",
    "loss.softmax_cross_entropy",
    "optimizer.sgd",
];
const LIMITS: TrainLimitsV1 = TrainLimitsV1 {
    max_rank: 4,
    max_elements: i32::MAX as u64,
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

    fn execute_ste_surrogate(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["weight", "scale"],
            TrainExecutionV1::Vjp => &["weight", "scale", "grad_output"],
            _ => return Err(invariant("ste_surrogate received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|b| b.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &["rows", "cols"],
            "attributes",
        )?;
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_weight", "grad_scale"],
            _ => unreachable!(),
        };
        require_names(
            output.buffers.iter().map(|b| b.name),
            expected_outputs,
            "outputs",
        )?;
        let rows = attribute_usize(request, "rows")?;
        let cols = attribute_usize(request, "cols")?;
        let elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        let shape = [rows as u64, cols as u64];
        let weight = input_f32(request, "weight", &shape)?;
        let scale = input_f32(request, "scale", &[rows as u64])?;
        require_finite("weight", weight)?;
        require_finite("scale", scale)?;
        let device_weight = self.backend.dev_upload(weight).map_err(cuda_error)?;
        let device_scale = self.backend.dev_upload(scale).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let mut device_result = self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            self.backend
                .ste_surrogate_forward_dev(
                    &device_weight,
                    &device_scale,
                    &mut device_result,
                    rows,
                    cols,
                )
                .map_err(cuda_error)?;
            let result = download_device(&self.backend, &device_result, elements)?;
            output_f32(output, "result", &shape, elements)?.copy_from_slice(&result);
        } else {
            let upstream = input_f32(request, "grad_output", &shape)?;
            require_finite("grad_output", upstream)?;
            let device_upstream = self.backend.dev_upload(upstream).map_err(cuda_error)?;
            let mut device_grad_weight =
                self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            self.backend
                .ste_surrogate_backward_dev(
                    &device_weight,
                    &device_scale,
                    &device_upstream,
                    &mut device_grad_weight,
                    rows,
                    cols,
                )
                .map_err(cuda_error)?;
            let grad_weight = download_device(&self.backend, &device_grad_weight, elements)?;
            output_f32(output, "grad_weight", &shape, elements)?.copy_from_slice(&grad_weight);
            output_f32(output, "grad_scale", &[rows as u64], rows)?.fill(0.0);
        }
        Ok(())
    }

    fn execute_salt_ste(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["weight"],
            TrainExecutionV1::Vjp => &["weight", "grad_output"],
            _ => return Err(invariant("salt_ste received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|b| b.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &["rows", "cols", "planes"],
            "attribute",
        )?;
        let output_name = if request.execution == TrainExecutionV1::Forward {
            "result"
        } else {
            "grad_weight"
        };
        require_names(
            output.buffers.iter().map(|b| b.name),
            &[output_name],
            "outputs",
        )?;
        let rows = attribute_usize(request, "rows")?;
        let cols = attribute_usize(request, "cols")?;
        let planes = attribute_usize(request, "planes")?;
        if planes == 0 {
            return Err(attribute_value("planes", "positive"));
        }
        if planes > MAX_SALT_PLANES {
            return Err(attribute_value("planes", "max_64"));
        }
        if request.execution == TrainExecutionV1::Forward && rows == 0 {
            return Err(attribute_value("rows", "positive"));
        }
        if cols == 0 {
            return Err(attribute_value("cols", "positive"));
        }
        let scratch_bytes = (cols as u64)
            .checked_mul(size_of::<f32>() as u64)
            .ok_or_else(|| attribute_value("cols", "scratch_bytes"))?;
        if request.execution == TrainExecutionV1::Forward && scratch_bytes > MAX_SALT_SCRATCH_BYTES
        {
            return Err(attribute_value("cols", "scratch_limit"));
        }
        let elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        let shape = [rows as u64, cols as u64];
        let weight = input_f32(request, "weight", &shape)?;
        require_finite("weight", weight)?;
        if request.execution == TrainExecutionV1::Forward {
            let device_weight = self.backend.dev_upload(weight).map_err(cuda_error)?;
            let mut device_residual = self.backend.dev_alloc_zeros(cols).map_err(cuda_error)?;
            let mut device_result = self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            self.backend
                .salt_quantize_forward_bounded_dev(
                    &device_weight,
                    &mut device_residual,
                    &mut device_result,
                    rows,
                    cols,
                    planes,
                )
                .map_err(cuda_error)?;
            let result = download_device(&self.backend, &device_result, elements)?;
            output_f32(output, "result", &shape, elements)?.copy_from_slice(&result);
        } else {
            let upstream = input_f32(request, "grad_output", &shape)?;
            require_finite("grad_output", upstream)?;
            let device_upstream = self.backend.dev_upload(upstream).map_err(cuda_error)?;
            let mut device_gradient = self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            self.backend
                .scale_const_dev(&device_upstream, &mut device_gradient, 1.0, elements)
                .map_err(cuda_error)?;
            let gradient = download_device(&self.backend, &device_gradient, elements)?;
            output_f32(output, "grad_weight", &shape, elements)?.copy_from_slice(&gradient);
        }
        Ok(())
    }

    fn execute_lsq(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["weight", "alpha"],
            TrainExecutionV1::Vjp => &["weight", "alpha", "grad_output"],
            _ => return Err(invariant("lsq_ste received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|b| b.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &["rows", "cols"],
            "attributes",
        )?;
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_weight", "grad_alpha"],
            _ => unreachable!(),
        };
        require_names(
            output.buffers.iter().map(|b| b.name),
            expected_outputs,
            "outputs",
        )?;
        let rows = attribute_usize(request, "rows")?;
        let cols = attribute_usize(request, "cols")?;
        if cols == 0 {
            return Err(attribute_value("cols", "positive"));
        }
        let elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        let shape = [rows as u64, cols as u64];
        let weight = input_f32(request, "weight", &shape)?;
        let alpha = input_f32(request, "alpha", &[rows as u64])?;
        require_finite("weight", weight)?;
        require_finite("alpha", alpha)?;
        let device_weight = self.backend.dev_upload(weight).map_err(cuda_error)?;
        let device_alpha = self.backend.dev_upload(alpha).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let mut device_result = self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            self.backend
                .lsq_forward_dev(
                    &device_weight,
                    &device_alpha,
                    &mut device_result,
                    rows,
                    cols,
                )
                .map_err(cuda_error)?;
            let result = download_device(&self.backend, &device_result, elements)?;
            output_f32(output, "result", &shape, elements)?.copy_from_slice(&result);
        } else {
            let upstream = input_f32(request, "grad_output", &shape)?;
            require_finite("grad_output", upstream)?;
            let device_upstream = self.backend.dev_upload(upstream).map_err(cuda_error)?;
            let mut device_grad_weight =
                self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            let mut device_grad_alpha = self.backend.dev_alloc_zeros(rows).map_err(cuda_error)?;
            self.backend
                .lsq_backward_dev(
                    &device_weight,
                    &device_alpha,
                    &device_upstream,
                    &mut device_grad_weight,
                    &mut device_grad_alpha,
                    rows,
                    cols,
                )
                .map_err(cuda_error)?;
            let grad_weight = download_device(&self.backend, &device_grad_weight, elements)?;
            let grad_alpha = download_device(&self.backend, &device_grad_alpha, rows)?;
            output_f32(output, "grad_weight", &shape, elements)?.copy_from_slice(&grad_weight);
            output_f32(output, "grad_alpha", &[rows as u64], rows)?.copy_from_slice(&grad_alpha);
        }
        Ok(())
    }

    fn execute_fsq(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x"],
            TrainExecutionV1::Vjp => &["x", "grad_output"],
            _ => return Err(invariant("fsq received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|b| b.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &["channels", "len", "levels", "bound", "ste", "alpha", "seed"],
            "attributes",
        )?;
        let output_name = if request.execution == TrainExecutionV1::Forward {
            "result"
        } else {
            "grad_x"
        };
        require_names(
            output.buffers.iter().map(|b| b.name),
            &[output_name],
            "outputs",
        )?;

        let channels = attribute_usize(request, "channels")?;
        let len = attribute_usize(request, "len")?;
        if channels == 0 {
            return Err(attribute_value("channels", "positive"));
        }
        if len == 0 {
            return Err(attribute_value("len", "positive"));
        }
        let elements = channels.checked_mul(len).ok_or_else(shape_error)?;
        let levels = attribute_u32_list(request, "levels")?;
        if levels.len() != channels {
            return Err(attribute_value("levels", "channels"));
        }
        if levels.iter().any(|&level| level < 2) {
            return Err(attribute_value("levels", "min_two"));
        }
        let bound = match attribute_text(request, "bound")? {
            "clamp" => 0,
            "tanh" => 1,
            _ => return Err(attribute_value("bound", "known")),
        };
        let estimator = match attribute_text(request, "ste")? {
            "hard" => 0,
            "soft_round" => 1,
            "stochastic" => 2,
            _ => return Err(attribute_value("ste", "known")),
        };
        let alpha = attribute_f32(request, "alpha")?;
        if !(0.0..=1.0).contains(&alpha) {
            return Err(attribute_value("alpha", "unit_interval"));
        }
        let seed = attribute_u64(request, "seed")?;
        let shape = [channels as u64, len as u64];
        let x = input_f32(request, "x", &shape)?;
        require_finite("x", x)?;
        let device_x = self.backend.dev_upload(x).map_err(cuda_error)?;
        let device_levels = self.backend.dev_upload_u32(levels).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let mut device_result = self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            self.backend
                .fsq_forward_dev(
                    &device_x,
                    &device_levels,
                    &mut device_result,
                    channels,
                    len,
                    bound,
                    estimator,
                    seed,
                )
                .map_err(cuda_error)?;
            let result = download_device(&self.backend, &device_result, elements)?;
            output_f32(output, "result", &shape, elements)?.copy_from_slice(&result);
        } else {
            let upstream = input_f32(request, "grad_output", &shape)?;
            require_finite("grad_output", upstream)?;
            let device_upstream = self.backend.dev_upload(upstream).map_err(cuda_error)?;
            let mut device_grad_x = self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            self.backend
                .fsq_backward_dev(
                    &device_x,
                    &device_levels,
                    &device_upstream,
                    &mut device_grad_x,
                    channels,
                    len,
                    bound,
                    estimator,
                    alpha,
                )
                .map_err(cuda_error)?;
            let grad_x = download_device(&self.backend, &device_grad_x, elements)?;
            output_f32(output, "grad_x", &shape, elements)?.copy_from_slice(&grad_x);
        }
        Ok(())
    }

    fn execute_ternary_matmul(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["activation", "weight", "scale"],
            TrainExecutionV1::Vjp => &["activation", "weight", "scale", "grad_output"],
            _ => {
                return Err(invariant(
                    "ternary_matmul received an illegal execution phase",
                ));
            }
        };
        require_names(
            request.inputs.iter().map(|b| b.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &["m", "n", "k"],
            "attributes",
        )?;
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_activation", "grad_weight", "grad_scale"],
            _ => unreachable!(),
        };
        require_names(
            output.buffers.iter().map(|b| b.name),
            expected_outputs,
            "outputs",
        )?;
        let m = attribute_usize(request, "m")?;
        let n = attribute_usize(request, "n")?;
        let k = attribute_usize(request, "k")?;
        let mk = m.checked_mul(k).ok_or_else(shape_error)?;
        let nk = n.checked_mul(k).ok_or_else(shape_error)?;
        let mn = m.checked_mul(n).ok_or_else(shape_error)?;
        let activation = input_f32(request, "activation", &[m as u64, k as u64])?;
        let weight = input_f32(request, "weight", &[n as u64, k as u64])?;
        let scale = input_f32(request, "scale", &[n as u64])?;
        require_finite("activation", activation)?;
        require_finite("weight", weight)?;
        require_finite("scale", scale)?;
        let device_activation = self.backend.dev_upload(activation).map_err(cuda_error)?;
        let device_weight = self.backend.dev_upload(weight).map_err(cuda_error)?;
        let device_scale = self.backend.dev_upload(scale).map_err(cuda_error)?;
        let shape = GemmShape { m, n, k };
        if request.execution == TrainExecutionV1::Forward {
            let mut device_result = self.backend.dev_alloc_zeros(mn).map_err(cuda_error)?;
            self.backend
                .matmul_forward_dev(
                    &device_activation,
                    &device_weight,
                    &device_scale,
                    shape,
                    &mut device_result,
                )
                .map_err(cuda_error)?;
            let result = download_device(&self.backend, &device_result, mn)?;
            output_f32(output, "result", &[m as u64, n as u64], mn)?.copy_from_slice(&result);
        } else {
            let grad_output = input_f32(request, "grad_output", &[m as u64, n as u64])?;
            require_finite("grad_output", grad_output)?;
            let device_grad = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
            let mut grad_activation = self.backend.dev_alloc_zeros(mk).map_err(cuda_error)?;
            let mut grad_weight = self.backend.dev_alloc_zeros(nk).map_err(cuda_error)?;
            let mut grad_scale = self.backend.dev_alloc_zeros(n).map_err(cuda_error)?;
            self.backend
                .grad_a_dev(
                    &device_grad,
                    &device_weight,
                    &device_scale,
                    shape,
                    &mut grad_activation,
                )
                .map_err(cuda_error)?;
            self.backend
                .grad_w_dev(
                    &device_grad,
                    &device_activation,
                    &device_scale,
                    shape,
                    &mut grad_weight,
                )
                .map_err(cuda_error)?;
            self.backend
                .grad_s_dev(
                    &device_grad,
                    &device_activation,
                    &device_weight,
                    shape,
                    &mut grad_scale,
                )
                .map_err(cuda_error)?;
            let grad_activation = download_device(&self.backend, &grad_activation, mk)?;
            let grad_weight = download_device(&self.backend, &grad_weight, nk)?;
            let grad_scale = download_device(&self.backend, &grad_scale, n)?;
            output_f32(output, "grad_activation", &[m as u64, k as u64], mk)?
                .copy_from_slice(&grad_activation);
            output_f32(output, "grad_weight", &[n as u64, k as u64], nk)?
                .copy_from_slice(&grad_weight);
            output_f32(output, "grad_scale", &[n as u64], n)?.copy_from_slice(&grad_scale);
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

    fn execute_embedding_gather(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["weight", "tokens"],
            TrainExecutionV1::Vjp => &["weight", "tokens", "grad_output"],
            _ => {
                return Err(invariant(
                    "embedding_gather received an illegal execution phase",
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
            &["vocab", "n_embd"],
            "attributes",
        )?;
        let output_name = if request.execution == TrainExecutionV1::Forward {
            "result"
        } else {
            "grad_weight"
        };
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
        let vocab = attribute_usize(request, "vocab")?;
        let n_embd = attribute_usize(request, "n_embd")?;
        let weight = input_f32(request, "weight", &[vocab as u64, n_embd as u64])?;
        let (token_shape, tokens) = input_u32_any_shape(request, "tokens")?;
        if token_shape.len() != 1 || tokens.iter().any(|&token| token as usize >= vocab) {
            return Err(shape_error());
        }
        require_finite("weight", weight)?;
        let seq = tokens.len();
        let tokens: Vec<i32> = tokens
            .iter()
            .map(|&token| i32::try_from(token).map_err(|_| shape_error()))
            .collect::<Result<_, _>>()?;
        let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
        let weight_id = tape.leaf(weight).map_err(cuda_error)?;
        let result_id = tape
            .embed(weight_id, &tokens, seq, n_embd, vocab)
            .map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let result = tape.value(result_id).map_err(cuda_error)?;
            output_f32(
                output,
                "result",
                &[seq as u64, n_embd as u64],
                seq.checked_mul(n_embd).ok_or_else(shape_error)?,
            )?
            .copy_from_slice(&result);
        } else {
            let grad_output = input_f32(request, "grad_output", &[seq as u64, n_embd as u64])?;
            require_finite("grad_output", grad_output)?;
            let seed = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
            let gradients = tape
                .backward_retain(result_id, &seed, &[weight_id])
                .map_err(cuda_error)?;
            let weight_elements = vocab.checked_mul(n_embd).ok_or_else(shape_error)?;
            let gradient =
                download_gradient(&self.backend, &gradients, weight_id, weight_elements)?;
            output_f32(
                output,
                "grad_weight",
                &[vocab as u64, n_embd as u64],
                weight_elements,
            )?
            .copy_from_slice(&gradient);
        }
        Ok(())
    }

    fn execute_detach(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_name, output_name, scale) = match request.execution {
            TrainExecutionV1::Forward => ("x", "result", 1.0),
            TrainExecutionV1::Vjp => ("grad_output", "grad_x", 0.0),
            _ => return Err(invariant("detach received an illegal execution phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            &[input_name],
            "inputs",
        )?;
        require_names(request.attributes.iter().map(|a| a.name), &[], "attributes")?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
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

    fn execute_bias(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x", "bias"],
            TrainExecutionV1::Vjp => &["x", "bias", "grad_output"],
            _ => return Err(invariant("bias received an illegal execution phase")),
        };
        require_names(
            request.inputs.iter().map(|b| b.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &["rows", "cols"],
            "attributes",
        )?;
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_x", "grad_bias"],
            _ => unreachable!(),
        };
        require_names(
            output.buffers.iter().map(|b| b.name),
            expected_outputs,
            "outputs",
        )?;
        let rows = attribute_usize(request, "rows")?;
        let cols = attribute_usize(request, "cols")?;
        let elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        let shape = [rows as u64, cols as u64];
        let x = input_f32(request, "x", &shape)?;
        let bias = input_f32(request, "bias", &[cols as u64])?;
        require_finite("x", x)?;
        require_finite("bias", bias)?;
        let device_x = self.backend.dev_upload(x).map_err(cuda_error)?;
        let device_bias = self.backend.dev_upload(bias).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let mut device_result = self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            self.backend
                .bias_forward_dev(&device_x, &device_bias, &mut device_result, rows, cols)
                .map_err(cuda_error)?;
            let result = download_device(&self.backend, &device_result, elements)?;
            output_f32(output, "result", &shape, elements)?.copy_from_slice(&result);
        } else {
            let grad_output = input_f32(request, "grad_output", &shape)?;
            require_finite("grad_output", grad_output)?;
            let device_grad = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
            let mut device_grad_x = self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            let mut device_grad_bias = self.backend.dev_alloc_zeros(cols).map_err(cuda_error)?;
            self.backend
                .scale_const_dev(&device_grad, &mut device_grad_x, 1.0, elements)
                .map_err(cuda_error)?;
            self.backend
                .bias_backward_dev(&device_grad, &mut device_grad_bias, rows, cols)
                .map_err(cuda_error)?;
            let grad_x = download_device(&self.backend, &device_grad_x, elements)?;
            let grad_bias = download_device(&self.backend, &device_grad_bias, cols)?;
            output_f32(output, "grad_x", &shape, elements)?.copy_from_slice(&grad_x);
            output_f32(output, "grad_bias", &[cols as u64], cols)?.copy_from_slice(&grad_bias);
        }
        Ok(())
    }

    fn execute_relu2(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x"],
            TrainExecutionV1::Vjp => &["x", "grad_output"],
            _ => return Err(invariant("relu2 received an illegal execution phase")),
        };
        require_names(
            request.inputs.iter().map(|b| b.name),
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
            output.buffers.iter().map(|b| b.name),
            &[output_name],
            "outputs",
        )?;
        let (shape, x) = input_f32_any_shape(request, "x")?;
        require_finite("x", x)?;
        let shape = shape.to_vec();
        let device_x = self.backend.dev_upload(x).map_err(cuda_error)?;
        let mut device_result = self.backend.dev_alloc_zeros(x.len()).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            self.backend
                .relu2_forward_dev(&device_x, &mut device_result, x.len())
                .map_err(cuda_error)?;
        } else {
            let (grad_shape, grad_output) = input_f32_any_shape(request, "grad_output")?;
            if grad_shape != shape || grad_output.len() != x.len() {
                return Err(shape_error());
            }
            require_finite("grad_output", grad_output)?;
            let device_grad = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
            self.backend
                .relu2_backward_dev(&device_x, &device_grad, &mut device_result, x.len())
                .map_err(cuda_error)?;
        }
        let result = download_device(&self.backend, &device_result, x.len())?;
        output_f32(output, output_name, &shape, x.len())?.copy_from_slice(&result);
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

    fn execute_rope(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_name, output_name) = match request.execution {
            TrainExecutionV1::Forward => ("x", "result"),
            TrainExecutionV1::Vjp => ("grad_output", "grad_x"),
            _ => return Err(invariant("rope received an illegal execution phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            &[input_name],
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["positions", "n_head", "head_dim", "theta"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
        let positions: Vec<u32> = attribute_u64_list(request, "positions")?
            .iter()
            .map(|&position| {
                u32::try_from(position).map_err(|_| attribute_value("positions", "u32"))
            })
            .collect::<Result<_, _>>()?;
        let n_head = attribute_usize(request, "n_head")?;
        let head_dim = attribute_usize(request, "head_dim")?;
        let theta = attribute_f32(request, "theta")?;
        if !head_dim.is_multiple_of(2) {
            return Err(attribute_value("head_dim", "even"));
        }
        if n_head == 0 || head_dim == 0 || !theta.is_finite() || theta <= 0.0 {
            return Err(attribute_value("theta", "positive"));
        }
        let n_token = positions.len();
        let elements = n_token
            .checked_mul(n_head)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(shape_error)?;
        let shape = [n_token as u64, n_head as u64, head_dim as u64];
        let input = input_f32(request, input_name, &shape)?;
        require_finite(input_name, input)?;
        let result = if request.execution == TrainExecutionV1::Forward {
            let mut tape = DeviceTape::new(&self.backend, 1).map_err(cuda_error)?;
            let input_id = tape.leaf(input).map_err(cuda_error)?;
            let result_id = tape
                .rope(input_id, &positions, n_head, head_dim, theta, n_token)
                .map_err(cuda_error)?;
            tape.value(result_id).map_err(cuda_error)?
        } else {
            let device_input = self.backend.dev_upload(input).map_err(cuda_error)?;
            let device_positions = self
                .backend
                .dev_upload_u32(&positions)
                .map_err(cuda_error)?;
            let mut device_result = self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            self.backend
                .rope_apply_dev(
                    &device_input,
                    &mut device_result,
                    &device_positions,
                    n_head,
                    head_dim,
                    theta,
                    n_token,
                    -1.0,
                )
                .map_err(cuda_error)?;
            let mut host = vec![0.0; elements];
            self.backend
                .dev_download(&device_result, &mut host)
                .map_err(cuda_error)?;
            host
        };
        output_f32(output, output_name, &shape, elements)?.copy_from_slice(&result);
        Ok(())
    }

    fn execute_mse(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["prediction", "target"],
            TrainExecutionV1::Vjp => &["prediction", "target", "grad_output"],
            _ => return Err(invariant("mse received an illegal execution phase")),
        };
        require_names(
            request.inputs.iter().map(|b| b.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(request.attributes.iter().map(|a| a.name), &[], "attributes")?;
        let output_name = if request.execution == TrainExecutionV1::Forward {
            "result"
        } else {
            "grad_prediction"
        };
        require_names(
            output.buffers.iter().map(|b| b.name),
            &[output_name],
            "outputs",
        )?;
        let (shape, prediction) = input_f32_any_shape(request, "prediction")?;
        let (target_shape, target) = input_f32_any_shape(request, "target")?;
        if shape != target_shape || prediction.is_empty() {
            return Err(shape_error());
        }
        require_finite("prediction", prediction)?;
        require_finite("target", target)?;
        let shape = shape.to_vec();
        let device_prediction = self.backend.dev_upload(prediction).map_err(cuda_error)?;
        let device_target = self.backend.dev_upload(target).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let mut device_loss = self.backend.dev_alloc_zeros(1).map_err(cuda_error)?;
            self.backend
                .mse_forward_dev(
                    &device_prediction,
                    &device_target,
                    &mut device_loss,
                    prediction.len(),
                )
                .map_err(cuda_error)?;
            let loss = download_device(&self.backend, &device_loss, 1)?;
            output_f32(output, "result", &[], 1)?.copy_from_slice(&loss);
        } else {
            let upstream = input_f32(request, "grad_output", &[])?;
            require_finite("grad_output", upstream)?;
            let device_upstream = self.backend.dev_upload(upstream).map_err(cuda_error)?;
            let mut device_gradient = self
                .backend
                .dev_alloc_zeros(prediction.len())
                .map_err(cuda_error)?;
            self.backend
                .mse_backward_dev(
                    &device_prediction,
                    &device_target,
                    &device_upstream,
                    &mut device_gradient,
                    prediction.len(),
                )
                .map_err(cuda_error)?;
            let gradient = download_device(&self.backend, &device_gradient, prediction.len())?;
            output_f32(output, "grad_prediction", &shape, prediction.len())?
                .copy_from_slice(&gradient);
        }
        Ok(())
    }

    fn execute_softmax_xent(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["logits", "target"],
            TrainExecutionV1::Vjp => &["logits", "target", "grad_output"],
            _ => return Err(invariant("softmax_cross_entropy received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|b| b.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &["rows", "cols"],
            "attributes",
        )?;
        let output_name = if request.execution == TrainExecutionV1::Forward {
            "result"
        } else {
            "grad_logits"
        };
        require_names(
            output.buffers.iter().map(|b| b.name),
            &[output_name],
            "outputs",
        )?;
        let rows = attribute_usize(request, "rows")?;
        let cols = attribute_usize(request, "cols")?;
        if rows == 0 {
            return Err(attribute_value("rows", "positive"));
        }
        if cols == 0 {
            return Err(attribute_value("cols", "positive"));
        }
        let elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        let shape = [rows as u64, cols as u64];
        let logits = input_f32(request, "logits", &shape)?;
        let target = input_f32(request, "target", &shape)?;
        require_finite("logits", logits)?;
        require_finite("target", target)?;
        let device_logits = self.backend.dev_upload(logits).map_err(cuda_error)?;
        let device_target = self.backend.dev_upload(target).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let mut device_loss = self.backend.dev_alloc_zeros(1).map_err(cuda_error)?;
            self.backend
                .softmax_xent_forward_dev(
                    &device_logits,
                    &device_target,
                    &mut device_loss,
                    rows,
                    cols,
                )
                .map_err(cuda_error)?;
            let loss = download_device(&self.backend, &device_loss, 1)?;
            output_f32(output, "result", &[], 1)?.copy_from_slice(&loss);
        } else {
            let upstream = input_f32(request, "grad_output", &[])?;
            require_finite("grad_output", upstream)?;
            let mut device_gradient = self.backend.dev_alloc_zeros(elements).map_err(cuda_error)?;
            self.backend
                .softmax_xent_backward_dev(
                    &device_logits,
                    &device_target,
                    &mut device_gradient,
                    rows,
                    cols,
                    upstream[0] / rows as f32,
                )
                .map_err(cuda_error)?;
            let gradient = download_device(&self.backend, &device_gradient, elements)?;
            output_f32(output, "grad_logits", &shape, elements)?.copy_from_slice(&gradient);
        }
        Ok(())
    }

    fn execute_sgd(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        if request.execution != TrainExecutionV1::Step {
            return Err(invariant("sgd received an illegal phase"));
        }
        require_names(
            request.inputs.iter().map(|b| b.name),
            &["parameter", "gradient"],
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &["step", "lr"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|b| b.name),
            &["parameter"],
            "outputs",
        )?;
        let (shape, parameter) = input_f32_any_shape(request, "parameter")?;
        let (gradient_shape, gradient) = input_f32_any_shape(request, "gradient")?;
        if shape != gradient_shape {
            return Err(shape_error());
        }
        require_finite("parameter", parameter)?;
        require_finite("gradient", gradient)?;
        if attribute_u64(request, "step")? == 0 {
            return Err(attribute_value("step", "one_based"));
        }
        let lr = attribute_f32(request, "lr")?;
        if lr < 0.0 {
            return Err(attribute_value("lr", "nonnegative"));
        }
        let device_parameter = self.backend.dev_upload(parameter).map_err(cuda_error)?;
        let device_gradient = self.backend.dev_upload(gradient).map_err(cuda_error)?;
        let mut device_updated = self
            .backend
            .dev_alloc_zeros(parameter.len())
            .map_err(cuda_error)?;
        self.backend
            .sgd_step_portable_dev(
                &device_parameter,
                &device_gradient,
                &mut device_updated,
                lr,
                parameter.len(),
            )
            .map_err(cuda_error)?;
        let updated = download_device(&self.backend, &device_updated, parameter.len())?;
        output_f32(output, "parameter", shape, parameter.len())?.copy_from_slice(&updated);
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
            "graph.ste_surrogate" => self.execute_ste_surrogate(&request, output)?,
            "graph.salt_ste" => self.execute_salt_ste(&request, output)?,
            "graph.lsq_ste" => self.execute_lsq(&request, output)?,
            "graph.fsq" => self.execute_fsq(&request, output)?,
            "graph.dense_matmul" => self.execute_dense_matmul(&request, output)?,
            "graph.ternary_matmul" => self.execute_ternary_matmul(&request, output)?,
            "graph.transpose" => self.execute_transpose(&request, output)?,
            "graph.embedding_gather" => self.execute_embedding_gather(&request, output)?,
            "graph.slice_cols" => self.execute_slice_cols(&request, output)?,
            "graph.concat_cols" => self.execute_concat_cols(&request, output)?,
            "graph.detach" => self.execute_detach(&request, output)?,
            "graph.scale_const" => self.execute_scale_const(&request, output)?,
            "graph.bias" => self.execute_bias(&request, output)?,
            "graph.add" => self.execute_add(&request, output)?,
            "graph.mul" => self.execute_mul(&request, output)?,
            "graph.relu2" => self.execute_relu2(&request, output)?,
            "graph.silu" => self.execute_silu(&request, output)?,
            "graph.rmsnorm" => self.execute_rmsnorm(&request, output)?,
            "graph.softmax" => self.execute_softmax(&request, output)?,
            "graph.causal_mask" => self.execute_causal_mask(&request, output)?,
            "graph.rope" => self.execute_rope(&request, output)?,
            "loss.mse" => self.execute_mse(&request, output)?,
            "loss.softmax_cross_entropy" => self.execute_softmax_xent(&request, output)?,
            "optimizer.sgd" => self.execute_sgd(&request, output)?,
            operation => {
                return Err(TrainBackendError::UnsupportedOperation(
                    operation.to_owned(),
                ));
            }
        }
        let scratch_bytes = if request.operation == "graph.salt_ste"
            && request.execution == TrainExecutionV1::Forward
        {
            attribute_u64(&request, "cols")?
                .checked_mul(size_of::<f32>() as u64)
                .ok_or_else(shape_error)?
        } else {
            0
        };
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
            scratch_bytes,
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

fn attribute_u64(request: &TrainRequestV1<'_>, name: &str) -> Result<u64, TrainBackendError> {
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
    Ok(value)
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

fn attribute_u32_list<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<&'a [u32], TrainBackendError> {
    let value = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or_else(|| roles("attributes"))?;
    let TrainAttributeValueV1::U32List(value) = value.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "u32_list",
            },
        ));
    };
    Ok(value)
}

fn attribute_text<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<&'a str, TrainBackendError> {
    let value = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or_else(|| roles("attributes"))?;
    let TrainAttributeValueV1::Text(value) = value.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "text",
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

fn input_u32_any_shape<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<(&'a [u64], &'a [u32]), TrainBackendError> {
    let buffer = request
        .inputs
        .iter()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| roles("inputs"))?;
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

fn download_device(
    backend: &CudaBackend,
    device: &cudarc::driver::CudaSlice<f32>,
    len: usize,
) -> Result<Vec<f32>, TrainBackendError> {
    let mut host = vec![0.0; len];
    backend
        .dev_download(device, &mut host)
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
