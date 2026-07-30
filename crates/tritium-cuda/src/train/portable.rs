//! Plan-0049 portable-training adapter backed by resident CUDA kernels.

use std::mem::size_of;

use tritium_core::GemmShape;
use tritium_spec::{
    BackendError, TernaryBackend, TrainAttributeValueV1, TrainBackendError, TrainBackendV1,
    TrainBufferDataMutV1, TrainBufferDataRefV1, TrainCapabilitiesV1, TrainDTypeV1,
    TrainExecutionV1, TrainLimitsV1, TrainOperationErrorV1, TrainOutputV1, TrainReceiptV1,
    TrainRequestV1, TrainingOpManifestV2, train_output_digest_v1, train_request_digest_v1,
};

use super::DeviceTape;
use crate::CudaBackend;

const BACKEND_FAMILY: &str = "cuda.portable.v1";
const MAX_SALT_PLANES: usize = 64;
const MAX_SALT_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CONV_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ATTENTION_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
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
    "graph.conv1d",
    "graph.conv2d",
    "graph.attention",
    "graph.relu2",
    "graph.silu",
    "graph.rmsnorm",
    "graph.softmax",
    "graph.causal_mask",
    "graph.rope",
    "loss.mse",
    "loss.softmax_cross_entropy",
    "loss.topk_knowledge_distillation",
    "optimizer.sgd",
    "optimizer.adamw",
    "optimizer.cautious_adamw",
    "optimizer.int8_adamw",
    "optimizer.muon",
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

    fn execute_topk_kd(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["logits", "indices", "probabilities"],
            TrainExecutionV1::Vjp => &["logits", "indices", "probabilities", "grad_output"],
            _ => {
                return Err(invariant(
                    "topk_knowledge_distillation received an illegal phase",
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
            &["rows", "cols", "k"],
            "attributes",
        )?;
        let output_name = if request.execution == TrainExecutionV1::Forward {
            "result"
        } else {
            "grad_logits"
        };
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
        let rows = attribute_usize(request, "rows")?;
        let cols = attribute_usize(request, "cols")?;
        let k = attribute_usize(request, "k")?;
        if rows == 0 {
            return Err(attribute_value("rows", "positive"));
        }
        if cols == 0 {
            return Err(attribute_value("cols", "positive"));
        }
        if k == 0 {
            return Err(attribute_value("k", "positive"));
        }
        if k > cols {
            return Err(attribute_value("k", "at_most_cols"));
        }
        let dense_elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        let sparse_elements = rows.checked_mul(k).ok_or_else(shape_error)?;
        let dense_shape = [rows as u64, cols as u64];
        let sparse_shape = [rows as u64, k as u64];
        let logits = input_f32(request, "logits", &dense_shape)?;
        let (indices_shape, indices) = input_u32_any_shape(request, "indices")?;
        let probabilities = input_f32(request, "probabilities", &sparse_shape)?;
        if indices_shape != sparse_shape || indices.len() != sparse_elements {
            return Err(shape_error());
        }
        require_finite("logits", logits)?;
        require_finite("probabilities", probabilities)?;
        if probabilities.iter().any(|&probability| probability < 0.0) {
            return Err(attribute_value("probabilities", "nonnegative"));
        }
        if indices.iter().any(|&index| index as usize >= cols) {
            return Err(attribute_value("indices", "in_range"));
        }
        let device_logits = self.backend.dev_upload(logits).map_err(cuda_error)?;
        let device_indices = self.backend.dev_upload_u32(indices).map_err(cuda_error)?;
        let device_probabilities = self.backend.dev_upload(probabilities).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let mut device_loss = self.backend.dev_alloc_zeros(1).map_err(cuda_error)?;
            self.backend
                .topk_kd_forward_dev(
                    &device_logits,
                    &device_indices,
                    &device_probabilities,
                    &mut device_loss,
                    rows,
                    cols,
                    k,
                )
                .map_err(cuda_error)?;
            let loss = download_device(&self.backend, &device_loss, 1)?;
            output_f32(output, "result", &[], 1)?.copy_from_slice(&loss);
        } else {
            let upstream = input_f32(request, "grad_output", &[])?;
            require_finite("grad_output", upstream)?;
            let mut device_gradient = self
                .backend
                .dev_alloc_zeros(dense_elements)
                .map_err(cuda_error)?;
            self.backend
                .topk_kd_backward_dev(
                    &device_logits,
                    &device_indices,
                    &device_probabilities,
                    &mut device_gradient,
                    rows,
                    cols,
                    k,
                    upstream[0] / rows as f32,
                )
                .map_err(cuda_error)?;
            let gradient = download_device(&self.backend, &device_gradient, dense_elements)?;
            output_f32(output, "grad_logits", &dense_shape, dense_elements)?
                .copy_from_slice(&gradient);
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

    fn execute_adamw(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
        cautious: bool,
    ) -> Result<(), TrainBackendError> {
        if request.execution != TrainExecutionV1::Step {
            return Err(invariant("AdamW received an illegal phase"));
        }
        require_names(
            request.inputs.iter().map(|b| b.name),
            &["parameter", "gradient", "moment1", "moment2"],
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &["step", "lr", "beta1", "beta2", "eps", "weight_decay"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|b| b.name),
            &["parameter", "moment1", "moment2"],
            "outputs",
        )?;
        let step = attribute_u64(request, "step")?;
        let optimizer = tritium_train::AdamW {
            lr: attribute_f32(request, "lr")?,
            beta1: attribute_f32(request, "beta1")?,
            beta2: attribute_f32(request, "beta2")?,
            eps: attribute_f32(request, "eps")?,
            weight_decay: attribute_f32(request, "weight_decay")?,
        };
        validate_adamw_attributes(step, &optimizer)?;
        let (shape, parameter) = input_f32_any_shape(request, "parameter")?;
        let (gradient_shape, gradient) = input_f32_any_shape(request, "gradient")?;
        let (moment1_shape, moment1) = input_f32_any_shape(request, "moment1")?;
        let (moment2_shape, moment2) = input_f32_any_shape(request, "moment2")?;
        if parameter.is_empty()
            || gradient_shape != shape
            || moment1_shape != shape
            || moment2_shape != shape
        {
            return Err(shape_error());
        }
        require_finite("parameter", parameter)?;
        require_finite("gradient", gradient)?;
        require_finite("moment1", moment1)?;
        require_finite("moment2", moment2)?;
        let mut device_parameter = self.backend.dev_upload(parameter).map_err(cuda_error)?;
        let device_gradient = self.backend.dev_upload(gradient).map_err(cuda_error)?;
        let mut device_moment1 = self.backend.dev_upload(moment1).map_err(cuda_error)?;
        let mut device_moment2 = self.backend.dev_upload(moment2).map_err(cuda_error)?;
        if cautious {
            self.backend
                .cautious_adamw_step_dev(
                    &mut device_parameter,
                    &device_gradient,
                    &mut device_moment1,
                    &mut device_moment2,
                    step,
                    &optimizer,
                )
                .map_err(cuda_error)?;
        } else {
            self.backend
                .adamw_step_dev(
                    &mut device_parameter,
                    &device_gradient,
                    &mut device_moment1,
                    &mut device_moment2,
                    step,
                    &optimizer,
                )
                .map_err(cuda_error)?;
        }
        let parameter = download_device(&self.backend, &device_parameter, parameter.len())?;
        let moment1 = download_device(&self.backend, &device_moment1, moment1.len())?;
        let moment2 = download_device(&self.backend, &device_moment2, moment2.len())?;
        output_f32(output, "parameter", shape, parameter.len())?.copy_from_slice(&parameter);
        output_f32(output, "moment1", shape, moment1.len())?.copy_from_slice(&moment1);
        output_f32(output, "moment2", shape, moment2.len())?.copy_from_slice(&moment2);
        Ok(())
    }

    fn execute_int8_adamw(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        if request.execution != TrainExecutionV1::Step {
            return Err(invariant("int8 AdamW received an illegal phase"));
        }
        require_names(
            request.inputs.iter().map(|b| b.name),
            &[
                "parameter",
                "gradient",
                "moment1_q8",
                "moment2_q8",
                "moment1_scale",
                "moment2_scale",
            ],
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &["step", "lr", "beta1", "beta2", "eps", "weight_decay"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|b| b.name),
            &[
                "parameter",
                "moment1_q8",
                "moment2_q8",
                "moment1_scale",
                "moment2_scale",
            ],
            "outputs",
        )?;
        let step = attribute_u64(request, "step")?;
        let optimizer = tritium_train::AdamW {
            lr: attribute_f32(request, "lr")?,
            beta1: attribute_f32(request, "beta1")?,
            beta2: attribute_f32(request, "beta2")?,
            eps: attribute_f32(request, "eps")?,
            weight_decay: attribute_f32(request, "weight_decay")?,
        };
        validate_adamw_attributes(step, &optimizer)?;
        let (shape, parameter) = input_f32_any_shape(request, "parameter")?;
        let (gradient_shape, gradient) = input_f32_any_shape(request, "gradient")?;
        let (moment1_shape, moment1_q8) = input_bytes_any_shape(request, "moment1_q8")?;
        let (moment2_shape, moment2_q8) = input_bytes_any_shape(request, "moment2_q8")?;
        let (moment1_scale_shape, moment1_scale) = input_f32_any_shape(request, "moment1_scale")?;
        let (moment2_scale_shape, moment2_scale) = input_f32_any_shape(request, "moment2_scale")?;
        let len = parameter.len();
        let blocks = len.div_ceil(tritium_train::INT8_ADAM_BLOCK);
        let code_shape = [len as u64];
        let scale_shape = [blocks as u64];
        if len == 0
            || gradient_shape != shape
            || moment1_shape != code_shape
            || moment2_shape != code_shape
            || moment1_scale_shape != scale_shape
            || moment2_scale_shape != scale_shape
        {
            return Err(shape_error());
        }
        require_finite("parameter", parameter)?;
        require_finite("gradient", gradient)?;
        require_finite("moment1_scale", moment1_scale)?;
        require_finite("moment2_scale", moment2_scale)?;
        if moment1_scale.iter().any(|&value| value < 0.0) {
            return Err(attribute_value("moment1_scale", "nonnegative"));
        }
        if moment2_scale.iter().any(|&value| value < 0.0) {
            return Err(attribute_value("moment2_scale", "nonnegative"));
        }
        output_f32(output, "parameter", shape, len)?;
        output_bytes(output, "moment1_q8", &code_shape, len)?;
        output_bytes(output, "moment2_q8", &code_shape, len)?;
        output_f32(output, "moment1_scale", &scale_shape, blocks)?;
        output_f32(output, "moment2_scale", &scale_shape, blocks)?;

        let moment1_signed: Vec<i8> = moment1_q8.iter().map(|&value| value as i8).collect();
        let mut device_parameter = self.backend.dev_upload(parameter).map_err(cuda_error)?;
        let device_gradient = self.backend.dev_upload(gradient).map_err(cuda_error)?;
        let mut device_moment1 = self
            .backend
            .dev_upload_i8(&moment1_signed)
            .map_err(cuda_error)?;
        let mut device_moment2 = self.backend.dev_upload_u8(moment2_q8).map_err(cuda_error)?;
        let mut device_moment1_scale =
            self.backend.dev_upload(moment1_scale).map_err(cuda_error)?;
        let mut device_moment2_scale =
            self.backend.dev_upload(moment2_scale).map_err(cuda_error)?;
        self.backend
            .adamw_step_8bit_dev(
                &mut device_parameter,
                &device_gradient,
                &mut device_moment1,
                &mut device_moment2,
                &mut device_moment1_scale,
                &mut device_moment2_scale,
                step,
                &optimizer,
            )
            .map_err(cuda_error)?;
        let parameter = download_device(&self.backend, &device_parameter, len)?;
        let mut moment1 = vec![0_i8; len];
        let mut moment2 = vec![0_u8; len];
        self.backend
            .dev_download_i8(&device_moment1, &mut moment1)
            .map_err(cuda_error)?;
        self.backend
            .dev_download_u8(&device_moment2, &mut moment2)
            .map_err(cuda_error)?;
        let moment1_scale = download_device(&self.backend, &device_moment1_scale, blocks)?;
        let moment2_scale = download_device(&self.backend, &device_moment2_scale, blocks)?;
        output_f32(output, "parameter", shape, len)?.copy_from_slice(&parameter);
        for (target, value) in output_bytes(output, "moment1_q8", &code_shape, len)?
            .iter_mut()
            .zip(moment1)
        {
            *target = value as u8;
        }
        output_bytes(output, "moment2_q8", &code_shape, len)?.copy_from_slice(&moment2);
        output_f32(output, "moment1_scale", &scale_shape, blocks)?.copy_from_slice(&moment1_scale);
        output_f32(output, "moment2_scale", &scale_shape, blocks)?.copy_from_slice(&moment2_scale);
        Ok(())
    }

    fn execute_muon(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        if request.execution != TrainExecutionV1::Step {
            return Err(invariant("Muon received an illegal phase"));
        }
        require_names(
            request.inputs.iter().map(|b| b.name),
            &["parameter", "gradient", "momentum"],
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &[
                "step",
                "lr",
                "momentum",
                "weight_decay",
                "rows",
                "cols",
                "ns_steps",
            ],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|b| b.name),
            &["parameter", "momentum"],
            "outputs",
        )?;
        let step = attribute_u64(request, "step")?;
        let optimizer = tritium_train::Muon {
            lr: attribute_f32(request, "lr")?,
            momentum: attribute_f32(request, "momentum")?,
            weight_decay: attribute_f32(request, "weight_decay")?,
            rows: attribute_usize(request, "rows")?,
            cols: attribute_usize(request, "cols")?,
            ns_steps: attribute_usize(request, "ns_steps")?,
        };
        if step == 0 {
            return Err(attribute_value("step", "one_based"));
        }
        if optimizer.lr < 0.0 {
            return Err(attribute_value("lr", "nonnegative"));
        }
        if !(0.0..1.0).contains(&optimizer.momentum) {
            return Err(attribute_value("momentum", "unit_interval_open"));
        }
        if optimizer.weight_decay < 0.0 {
            return Err(attribute_value("weight_decay", "nonnegative"));
        }
        if optimizer.rows == 0 {
            return Err(attribute_value("rows", "positive"));
        }
        if optimizer.cols == 0 {
            return Err(attribute_value("cols", "positive"));
        }
        if optimizer.ns_steps == 0 {
            return Err(attribute_value("ns_steps", "positive"));
        }
        if optimizer.ns_steps > 32 {
            return Err(attribute_value("ns_steps", "max_32"));
        }
        let len = optimizer
            .rows
            .checked_mul(optimizer.cols)
            .ok_or_else(|| attribute_value("rows", "arithmetic"))?;
        if len > u32::MAX as usize {
            return Err(attribute_value("rows", "max_elements"));
        }
        let shape = [optimizer.rows as u64, optimizer.cols as u64];
        let parameter = input_f32(request, "parameter", &shape)?;
        let gradient = input_f32(request, "gradient", &shape)?;
        let momentum = input_f32(request, "momentum", &shape)?;
        require_finite("parameter", parameter)?;
        require_finite("gradient", gradient)?;
        require_finite("momentum", momentum)?;
        output_f32(output, "parameter", &shape, len)?;
        output_f32(output, "momentum", &shape, len)?;
        let mut device_parameter = self.backend.dev_upload(parameter).map_err(cuda_error)?;
        let device_gradient = self.backend.dev_upload(gradient).map_err(cuda_error)?;
        let mut device_momentum = self.backend.dev_upload(momentum).map_err(cuda_error)?;
        self.backend
            .muon_step_portable_dev(
                &mut device_parameter,
                &device_gradient,
                &mut device_momentum,
                &optimizer,
            )
            .map_err(cuda_error)?;
        let parameter = download_device(&self.backend, &device_parameter, len)?;
        let momentum = download_device(&self.backend, &device_momentum, len)?;
        output_f32(output, "parameter", &shape, len)?.copy_from_slice(&parameter);
        output_f32(output, "momentum", &shape, len)?.copy_from_slice(&momentum);
        Ok(())
    }

    fn execute_conv1d(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x", "weight", "scale"],
            TrainExecutionV1::Vjp => &["x", "weight", "scale", "grad_output"],
            _ => return Err(invariant("Conv1d received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|b| b.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &[
                "batch",
                "c_in",
                "c_out",
                "l_in",
                "k",
                "stride",
                "dilation",
                "pad_left",
                "pad_right",
                "groups",
            ],
            "attributes",
        )?;
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_x", "grad_weight", "grad_scale"],
            _ => unreachable!(),
        };
        require_names(
            output.buffers.iter().map(|b| b.name),
            expected_outputs,
            "outputs",
        )?;
        let (config, l_out) = conv1d_attributes(request)?;
        if conv1d_contract_scratch(&config, l_out, request.execution)? > MAX_CONV_SCRATCH_BYTES {
            return Err(attribute_value("scratch", "limit_64_mib"));
        }
        let input_shape = [config.batch as u64, config.c_in as u64, config.l_in as u64];
        let weight_shape = [
            config.c_out as u64,
            config.c_in_pg() as u64,
            config.k as u64,
        ];
        let scale_shape = [config.c_out as u64];
        let result_shape = [config.batch as u64, config.c_out as u64, l_out as u64];
        let x = input_f32(request, "x", &input_shape)?;
        let weight = input_f32(request, "weight", &weight_shape)?;
        let scale = input_f32(request, "scale", &scale_shape)?;
        require_finite("x", x)?;
        require_finite("weight", weight)?;
        require_finite("scale", scale)?;
        let device_x = self.backend.dev_upload(x).map_err(cuda_error)?;
        let device_weight = self.backend.dev_upload(weight).map_err(cuda_error)?;
        let device_scale = self.backend.dev_upload(scale).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let result_len = config.batch * config.c_out * l_out;
            output_f32(output, "result", &result_shape, result_len)?;
            let mut device_result = self
                .backend
                .dev_alloc_zeros(result_len)
                .map_err(cuda_error)?;
            self.backend
                .conv1d_forward_portable_dev(
                    &device_x,
                    &device_weight,
                    &device_scale,
                    &mut device_result,
                    &config,
                    l_out,
                )
                .map_err(cuda_error)?;
            let result = download_device(&self.backend, &device_result, result_len)?;
            output_f32(output, "result", &result_shape, result_len)?.copy_from_slice(&result);
        } else {
            let grad_output = input_f32(request, "grad_output", &result_shape)?;
            require_finite("grad_output", grad_output)?;
            output_f32(output, "grad_x", &input_shape, x.len())?;
            output_f32(output, "grad_weight", &weight_shape, weight.len())?;
            output_f32(output, "grad_scale", &scale_shape, scale.len())?;
            let device_grad_output = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
            let mut device_grad_x = self.backend.dev_alloc_zeros(x.len()).map_err(cuda_error)?;
            let mut device_grad_weight = self
                .backend
                .dev_alloc_zeros(weight.len())
                .map_err(cuda_error)?;
            let mut device_grad_scale = self
                .backend
                .dev_alloc_zeros(scale.len())
                .map_err(cuda_error)?;
            self.backend
                .conv1d_backward_portable_dev(
                    &device_x,
                    &device_weight,
                    &device_scale,
                    &device_grad_output,
                    &mut device_grad_x,
                    &mut device_grad_weight,
                    &mut device_grad_scale,
                    &config,
                    l_out,
                )
                .map_err(cuda_error)?;
            let grad_x = download_device(&self.backend, &device_grad_x, x.len())?;
            let grad_weight = download_device(&self.backend, &device_grad_weight, weight.len())?;
            let grad_scale = download_device(&self.backend, &device_grad_scale, scale.len())?;
            output_f32(output, "grad_x", &input_shape, x.len())?.copy_from_slice(&grad_x);
            output_f32(output, "grad_weight", &weight_shape, weight.len())?
                .copy_from_slice(&grad_weight);
            output_f32(output, "grad_scale", &scale_shape, scale.len())?
                .copy_from_slice(&grad_scale);
        }
        Ok(())
    }

    fn execute_conv2d(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x", "weight", "scale"],
            TrainExecutionV1::Vjp => &["x", "weight", "scale", "grad_output"],
            _ => return Err(invariant("Conv2d received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|b| b.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|a| a.name),
            &[
                "batch",
                "c_in",
                "c_out",
                "input_h",
                "input_w",
                "kernel_h",
                "kernel_w",
                "stride_h",
                "stride_w",
                "dilation_h",
                "dilation_w",
                "pad_top",
                "pad_bottom",
                "pad_left",
                "pad_right",
                "groups",
            ],
            "attributes",
        )?;
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_x", "grad_weight", "grad_scale"],
            _ => unreachable!(),
        };
        require_names(
            output.buffers.iter().map(|b| b.name),
            expected_outputs,
            "outputs",
        )?;
        let (config, height_out, width_out) = conv2d_attributes(request)?;
        if conv2d_contract_scratch(&config, height_out, width_out, request.execution)?
            > MAX_CONV_SCRATCH_BYTES
        {
            return Err(attribute_value("scratch", "limit_64_mib"));
        }
        let input_shape = [
            config.batch as u64,
            config.c_in as u64,
            config.input_h as u64,
            config.input_w as u64,
        ];
        let weight_shape = [
            config.c_out as u64,
            config.c_in_per_group() as u64,
            config.kernel_h as u64,
            config.kernel_w as u64,
        ];
        let scale_shape = [config.c_out as u64];
        let result_shape = [
            config.batch as u64,
            config.c_out as u64,
            height_out as u64,
            width_out as u64,
        ];
        let x = input_f32(request, "x", &input_shape)?;
        let weight = input_f32(request, "weight", &weight_shape)?;
        let scale = input_f32(request, "scale", &scale_shape)?;
        require_finite("x", x)?;
        require_finite("weight", weight)?;
        require_finite("scale", scale)?;
        let device_x = self.backend.dev_upload(x).map_err(cuda_error)?;
        let device_weight = self.backend.dev_upload(weight).map_err(cuda_error)?;
        let device_scale = self.backend.dev_upload(scale).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            let result_len = config.output_elements();
            output_f32(output, "result", &result_shape, result_len)?;
            let mut device_result = self
                .backend
                .dev_alloc_zeros(result_len)
                .map_err(cuda_error)?;
            self.backend
                .conv2d_forward_portable_dev(
                    &device_x,
                    &device_weight,
                    &device_scale,
                    &mut device_result,
                    &config,
                    height_out,
                    width_out,
                )
                .map_err(cuda_error)?;
            let result = download_device(&self.backend, &device_result, result_len)?;
            output_f32(output, "result", &result_shape, result_len)?.copy_from_slice(&result);
        } else {
            let grad_output = input_f32(request, "grad_output", &result_shape)?;
            require_finite("grad_output", grad_output)?;
            output_f32(output, "grad_x", &input_shape, x.len())?;
            output_f32(output, "grad_weight", &weight_shape, weight.len())?;
            output_f32(output, "grad_scale", &scale_shape, scale.len())?;
            let device_grad_output = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
            let mut device_grad_x = self.backend.dev_alloc_zeros(x.len()).map_err(cuda_error)?;
            let mut device_grad_weight = self
                .backend
                .dev_alloc_zeros(weight.len())
                .map_err(cuda_error)?;
            let mut device_grad_scale = self
                .backend
                .dev_alloc_zeros(scale.len())
                .map_err(cuda_error)?;
            self.backend
                .conv2d_backward_portable_dev(
                    &device_x,
                    &device_weight,
                    &device_scale,
                    &device_grad_output,
                    &mut device_grad_x,
                    &mut device_grad_weight,
                    &mut device_grad_scale,
                    &config,
                    height_out,
                    width_out,
                )
                .map_err(cuda_error)?;
            let grad_x = download_device(&self.backend, &device_grad_x, x.len())?;
            let grad_weight = download_device(&self.backend, &device_grad_weight, weight.len())?;
            let grad_scale = download_device(&self.backend, &device_grad_scale, scale.len())?;
            output_f32(output, "grad_x", &input_shape, x.len())?.copy_from_slice(&grad_x);
            output_f32(output, "grad_weight", &weight_shape, weight.len())?
                .copy_from_slice(&grad_weight);
            output_f32(output, "grad_scale", &scale_shape, scale.len())?
                .copy_from_slice(&grad_scale);
        }
        Ok(())
    }

    fn execute_attention(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["q", "k", "v"],
            TrainExecutionV1::Vjp => &["q", "k", "v", "grad_output"],
            _ => return Err(invariant("attention received an illegal phase")),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["seq", "n_head", "n_kv_head", "head_dim", "causal"],
            "attributes",
        )?;
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_q", "grad_k", "grad_v"],
            _ => unreachable!(),
        };
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            expected_outputs,
            "outputs",
        )?;
        let config = attention_attributes(request)?;
        let query_len = config.query_elements().ok_or_else(shape_error)?;
        let kv_len = config.kv_elements().ok_or_else(shape_error)?;
        let probability_len = bounded_u32_product(&[config.seq, config.seq], "seq")?;
        let contract_elements = match request.execution {
            TrainExecutionV1::Forward => query_len.checked_add(probability_len),
            TrainExecutionV1::Vjp => query_len
                .checked_add(kv_len)
                .and_then(|value| value.checked_add(kv_len))
                .and_then(|value| value.checked_add(probability_len))
                .and_then(|value| value.checked_add(probability_len)),
            _ => unreachable!(),
        }
        .ok_or_else(shape_error)?;
        let contract_scratch = (contract_elements as u64)
            .checked_mul(size_of::<f32>() as u64)
            .ok_or_else(shape_error)?;
        if contract_scratch > MAX_ATTENTION_SCRATCH_BYTES {
            return Err(attribute_value("scratch", "limit_64_mib"));
        }
        let query_shape = [
            config.seq as u64,
            config.n_head as u64,
            config.head_dim as u64,
        ];
        let kv_shape = [
            config.seq as u64,
            config.n_kv_head as u64,
            config.head_dim as u64,
        ];
        let q = input_f32(request, "q", &query_shape)?;
        let k = input_f32(request, "k", &kv_shape)?;
        let v = input_f32(request, "v", &kv_shape)?;
        require_finite("q", q)?;
        require_finite("k", k)?;
        require_finite("v", v)?;
        debug_assert_eq!(q.len(), query_len);
        debug_assert_eq!(k.len(), kv_len);
        let device_q = self.backend.dev_upload(q).map_err(cuda_error)?;
        let device_k = self.backend.dev_upload(k).map_err(cuda_error)?;
        let device_v = self.backend.dev_upload(v).map_err(cuda_error)?;
        if request.execution == TrainExecutionV1::Forward {
            output_f32(output, "result", &query_shape, query_len)?;
            let mut device_result = self
                .backend
                .dev_alloc_zeros(query_len)
                .map_err(cuda_error)?;
            self.backend
                .attention_forward_portable_dev(
                    &device_q,
                    &device_k,
                    &device_v,
                    &mut device_result,
                    &config,
                )
                .map_err(cuda_error)?;
            let result = download_device(&self.backend, &device_result, query_len)?;
            output_f32(output, "result", &query_shape, query_len)?.copy_from_slice(&result);
        } else {
            let grad_output = input_f32(request, "grad_output", &query_shape)?;
            require_finite("grad_output", grad_output)?;
            output_f32(output, "grad_q", &query_shape, query_len)?;
            output_f32(output, "grad_k", &kv_shape, kv_len)?;
            output_f32(output, "grad_v", &kv_shape, kv_len)?;
            let device_grad_output = self.backend.dev_upload(grad_output).map_err(cuda_error)?;
            let mut device_grad_q = self
                .backend
                .dev_alloc_zeros(query_len)
                .map_err(cuda_error)?;
            let mut device_grad_k = self.backend.dev_alloc_zeros(kv_len).map_err(cuda_error)?;
            let mut device_grad_v = self.backend.dev_alloc_zeros(kv_len).map_err(cuda_error)?;
            self.backend
                .attention_backward_portable_dev(
                    &device_q,
                    &device_k,
                    &device_v,
                    &device_grad_output,
                    &mut device_grad_q,
                    &mut device_grad_k,
                    &mut device_grad_v,
                    &config,
                )
                .map_err(cuda_error)?;
            let grad_q = download_device(&self.backend, &device_grad_q, query_len)?;
            let grad_k = download_device(&self.backend, &device_grad_k, kv_len)?;
            let grad_v = download_device(&self.backend, &device_grad_v, kv_len)?;
            output_f32(output, "grad_q", &query_shape, query_len)?.copy_from_slice(&grad_q);
            output_f32(output, "grad_k", &kv_shape, kv_len)?.copy_from_slice(&grad_k);
            output_f32(output, "grad_v", &kv_shape, kv_len)?.copy_from_slice(&grad_v);
        }
        Ok(())
    }
}

impl TrainBackendV1 for CudaTrainBackendV1 {
    fn capabilities(&self) -> TrainCapabilitiesV1 {
        TrainCapabilitiesV1 {
            backend_id: self.backend_id.clone(),
            manifest_digest: TrainingOpManifestV2::digest(),
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
            "graph.conv1d" => self.execute_conv1d(&request, output)?,
            "graph.conv2d" => self.execute_conv2d(&request, output)?,
            "graph.attention" => self.execute_attention(&request, output)?,
            "graph.relu2" => self.execute_relu2(&request, output)?,
            "graph.silu" => self.execute_silu(&request, output)?,
            "graph.rmsnorm" => self.execute_rmsnorm(&request, output)?,
            "graph.softmax" => self.execute_softmax(&request, output)?,
            "graph.causal_mask" => self.execute_causal_mask(&request, output)?,
            "graph.rope" => self.execute_rope(&request, output)?,
            "loss.mse" => self.execute_mse(&request, output)?,
            "loss.softmax_cross_entropy" => self.execute_softmax_xent(&request, output)?,
            "loss.topk_knowledge_distillation" => self.execute_topk_kd(&request, output)?,
            "optimizer.sgd" => self.execute_sgd(&request, output)?,
            "optimizer.adamw" => self.execute_adamw(&request, output, false)?,
            "optimizer.cautious_adamw" => self.execute_adamw(&request, output, true)?,
            "optimizer.int8_adamw" => self.execute_int8_adamw(&request, output)?,
            "optimizer.muon" => self.execute_muon(&request, output)?,
            "lifecycle.checkpoint"
            | "lifecycle.resume"
            | "lifecycle.export"
            | "lifecycle.reload" => {
                tritium_train::portable::execute_lifecycle_control_plane(&request, output)?;
            }
            operation => {
                return Err(TrainBackendError::UnsupportedOperation(
                    operation.to_owned(),
                ));
            }
        }
        let scratch_bytes = match (request.operation, request.execution) {
            ("graph.salt_ste", TrainExecutionV1::Forward) => attribute_u64(&request, "cols")?
                .checked_mul(size_of::<f32>() as u64)
                .ok_or_else(shape_error)?,
            ("optimizer.cautious_adamw", TrainExecutionV1::Step) => {
                let (_, parameter) = input_f32_any_shape(&request, "parameter")?;
                (parameter.len() as u64)
                    .checked_mul(size_of::<f32>() as u64)
                    .and_then(|bytes| bytes.checked_add(size_of::<u32>() as u64))
                    .ok_or_else(shape_error)?
            }
            ("optimizer.muon", TrainExecutionV1::Step) => {
                let rows = attribute_u64(&request, "rows")?;
                let cols = attribute_u64(&request, "cols")?;
                let matrix = rows.checked_mul(cols).ok_or_else(shape_error)?;
                let axis = rows.min(cols);
                let gram = axis.checked_mul(axis).ok_or_else(shape_error)?;
                matrix
                    .checked_mul(3)
                    .and_then(|value| gram.checked_mul(3).and_then(|g| value.checked_add(g)))
                    .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
                    .ok_or_else(shape_error)?
            }
            ("graph.conv1d", TrainExecutionV1::Vjp) => {
                let (config, l_out) = conv1d_attributes(&request)?;
                let columns = l_out.checked_mul(config.k_g()).ok_or_else(shape_error)?;
                let group_output = l_out.checked_mul(config.n_g()).ok_or_else(shape_error)?;
                let elements = columns
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(group_output))
                    .and_then(|value| value.checked_add(config.n_g() * config.k_g()))
                    .and_then(|value| value.checked_add(config.n_g()))
                    .ok_or_else(shape_error)?;
                (elements as u64)
                    .checked_mul(size_of::<f32>() as u64)
                    .ok_or_else(shape_error)?
            }
            ("graph.conv2d", TrainExecutionV1::Vjp) => {
                let (config, height_out, width_out) = conv2d_attributes(&request)?;
                conv2d_actual_scratch(&config, height_out, width_out)?
            }
            ("graph.attention", execution) => {
                let config = attention_attributes(&request)?;
                let probability_elements = bounded_u32_product(&[config.seq, config.seq], "seq")?;
                let multiplier = match execution {
                    TrainExecutionV1::Forward => 1_u64,
                    TrainExecutionV1::Vjp => 2_u64,
                    _ => 0_u64,
                };
                (probability_elements as u64)
                    .checked_mul(multiplier)
                    .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
                    .ok_or_else(shape_error)?
            }
            (operation, _execution) if operation.starts_with("lifecycle.") => {
                tritium_train::portable::lifecycle_control_plane_scratch_bytes(&request)?
            }
            _ => 0,
        };
        Ok(TrainReceiptV1 {
            backend_id: self.backend_id.clone(),
            backend_build: backend_build_identity(),
            physical_device: Some(self.physical_device.clone()),
            manifest_digest: TrainingOpManifestV2::digest(),
            vector_digest: request.vector_digest,
            operation: request.operation.to_owned(),
            execution: request.execution,
            dtype: if request.operation.starts_with("lifecycle.") {
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

fn backend_build_identity() -> String {
    format!(
        "{}@{}+{}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("TRITIUM_SOURCE_ID")
    )
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

fn validate_adamw_attributes(
    step: u64,
    optimizer: &tritium_train::AdamW,
) -> Result<(), TrainBackendError> {
    if step == 0 {
        return Err(attribute_value("step", "one_based"));
    }
    if optimizer.lr < 0.0 {
        return Err(attribute_value("lr", "nonnegative"));
    }
    if !(0.0..1.0).contains(&optimizer.beta1) {
        return Err(attribute_value("beta1", "unit_interval_open"));
    }
    if !(0.0..1.0).contains(&optimizer.beta2) {
        return Err(attribute_value("beta2", "unit_interval_open"));
    }
    if optimizer.eps <= 0.0 {
        return Err(attribute_value("eps", "positive"));
    }
    if optimizer.weight_decay < 0.0 {
        return Err(attribute_value("weight_decay", "nonnegative"));
    }
    Ok(())
}

fn conv1d_attributes(
    request: &TrainRequestV1<'_>,
) -> Result<(tritium_train::ops::conv1d::Conv1dCfg, usize), TrainBackendError> {
    let config = tritium_train::ops::conv1d::Conv1dCfg {
        batch: attribute_usize(request, "batch")?,
        c_in: attribute_usize(request, "c_in")?,
        c_out: attribute_usize(request, "c_out")?,
        l_in: attribute_usize(request, "l_in")?,
        k: attribute_usize(request, "k")?,
        stride: attribute_usize(request, "stride")?,
        dilation: attribute_usize(request, "dilation")?,
        pad_left: attribute_usize(request, "pad_left")?,
        pad_right: attribute_usize(request, "pad_right")?,
        groups: attribute_usize(request, "groups")?,
    };
    for (name, value) in [
        ("batch", config.batch),
        ("c_in", config.c_in),
        ("c_out", config.c_out),
        ("l_in", config.l_in),
        ("k", config.k),
        ("stride", config.stride),
        ("dilation", config.dilation),
        ("groups", config.groups),
    ] {
        if value == 0 {
            return Err(attribute_value(name, "positive"));
        }
    }
    if !config.c_in.is_multiple_of(config.groups) || !config.c_out.is_multiple_of(config.groups) {
        return Err(attribute_value("groups", "divides_channels"));
    }
    let effective = (config.dilation as u64)
        .checked_mul((config.k - 1) as u64)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| attribute_value("k", "arithmetic"))?;
    let padded = (config.l_in as u64)
        .checked_add(config.pad_left as u64)
        .and_then(|value| value.checked_add(config.pad_right as u64))
        .ok_or_else(|| attribute_value("k", "arithmetic"))?;
    if effective > u32::MAX as u64 || padded > u32::MAX as u64 {
        return Err(attribute_value("k", "axis_u32"));
    }
    if padded < effective {
        return Err(attribute_value("k", "output_nonzero"));
    }
    let l_out = usize::try_from((padded - effective) / config.stride as u64 + 1)
        .map_err(|_| attribute_value("k", "axis_u32"))?;
    let maximum_position = ((l_out - 1) as u64)
        .checked_mul(config.stride as u64)
        .and_then(|value| {
            (config.k - 1)
                .checked_mul(config.dilation)
                .and_then(|tail| value.checked_add(tail as u64))
        })
        .ok_or_else(|| attribute_value("k", "index_arithmetic"))?;
    if maximum_position > i32::MAX as u64 || config.pad_left > i32::MAX as usize {
        return Err(attribute_value("k", "index_i32"));
    }
    bounded_u32_product(&[config.batch, config.c_in, config.l_in], "batch")?;
    bounded_u32_product(&[config.c_out, config.c_in_pg(), config.k], "c_out")?;
    bounded_u32_product(&[config.batch, config.c_out, l_out], "batch")?;
    Ok((config, l_out))
}

fn bounded_u32_product(values: &[usize], name: &str) -> Result<usize, TrainBackendError> {
    let product = values.iter().try_fold(1_u64, |total, &value| {
        total
            .checked_mul(value as u64)
            .ok_or_else(|| attribute_value(name, "arithmetic"))
    })?;
    if product > u32::MAX as u64 {
        return Err(attribute_value(name, "max_elements"));
    }
    Ok(product as usize)
}

fn attention_attributes(
    request: &TrainRequestV1<'_>,
) -> Result<tritium_train::ops::attention::AttentionCfg, TrainBackendError> {
    let config = tritium_train::ops::attention::AttentionCfg {
        seq: attribute_usize(request, "seq")?,
        n_head: attribute_usize(request, "n_head")?,
        n_kv_head: attribute_usize(request, "n_kv_head")?,
        head_dim: attribute_usize(request, "head_dim")?,
        causal: attribute_bool(request, "causal")?,
    };
    for (name, value) in [
        ("seq", config.seq),
        ("n_head", config.n_head),
        ("n_kv_head", config.n_kv_head),
        ("head_dim", config.head_dim),
    ] {
        if value == 0 {
            return Err(attribute_value(name, "positive"));
        }
    }
    if !config.n_head.is_multiple_of(config.n_kv_head) {
        return Err(attribute_value("n_kv_head", "divides_n_head"));
    }
    bounded_u32_product(&[config.seq, config.n_head, config.head_dim], "seq")?;
    bounded_u32_product(&[config.seq, config.n_kv_head, config.head_dim], "seq")?;
    Ok(config)
}

fn conv1d_contract_scratch(
    config: &tritium_train::ops::conv1d::Conv1dCfg,
    l_out: usize,
    execution: TrainExecutionV1,
) -> Result<u64, TrainBackendError> {
    let input = config.batch * config.c_in * config.l_in;
    let weight = config.c_out * config.k_g();
    let columns = l_out * config.k_g();
    let group_output = l_out * config.n_g();
    let elements = match execution {
        TrainExecutionV1::Forward => config
            .batch
            .checked_mul(config.c_out)
            .and_then(|value| value.checked_mul(l_out))
            .and_then(|value| value.checked_add(columns))
            .and_then(|value| value.checked_add(group_output)),
        TrainExecutionV1::Vjp => input
            .checked_add(weight)
            .and_then(|value| value.checked_add(config.c_out))
            .and_then(|value| value.checked_add(columns))
            .and_then(|value| value.checked_add(group_output))
            .and_then(|value| value.checked_add(columns))
            .and_then(|value| value.checked_add(weight / config.groups))
            .and_then(|value| value.checked_add(config.c_out / config.groups)),
        _ => Some(0),
    }
    .ok_or_else(shape_error)?;
    (elements as u64)
        .checked_mul(size_of::<f32>() as u64)
        .ok_or_else(shape_error)
}

fn conv2d_attributes(
    request: &TrainRequestV1<'_>,
) -> Result<(tritium_train::ops::conv2d::Conv2dCfg, usize, usize), TrainBackendError> {
    let config = tritium_train::ops::conv2d::Conv2dCfg {
        batch: attribute_usize(request, "batch")?,
        c_in: attribute_usize(request, "c_in")?,
        c_out: attribute_usize(request, "c_out")?,
        input_h: attribute_usize(request, "input_h")?,
        input_w: attribute_usize(request, "input_w")?,
        kernel_h: attribute_usize(request, "kernel_h")?,
        kernel_w: attribute_usize(request, "kernel_w")?,
        stride_h: attribute_usize(request, "stride_h")?,
        stride_w: attribute_usize(request, "stride_w")?,
        dilation_h: attribute_usize(request, "dilation_h")?,
        dilation_w: attribute_usize(request, "dilation_w")?,
        pad_top: attribute_usize(request, "pad_top")?,
        pad_bottom: attribute_usize(request, "pad_bottom")?,
        pad_left: attribute_usize(request, "pad_left")?,
        pad_right: attribute_usize(request, "pad_right")?,
        groups: attribute_usize(request, "groups")?,
    };
    for (name, value) in [
        ("batch", config.batch),
        ("c_in", config.c_in),
        ("c_out", config.c_out),
        ("input_h", config.input_h),
        ("input_w", config.input_w),
        ("kernel_h", config.kernel_h),
        ("kernel_w", config.kernel_w),
        ("stride_h", config.stride_h),
        ("stride_w", config.stride_w),
        ("dilation_h", config.dilation_h),
        ("dilation_w", config.dilation_w),
        ("groups", config.groups),
    ] {
        if value == 0 {
            return Err(attribute_value(name, "positive"));
        }
    }
    if !config.c_in.is_multiple_of(config.groups) || !config.c_out.is_multiple_of(config.groups) {
        return Err(attribute_value("groups", "divides_channels"));
    }
    let height_out = checked_conv_output_axis(
        config.input_h,
        config.kernel_h,
        config.stride_h,
        config.dilation_h,
        config.pad_top,
        config.pad_bottom,
        "kernel_h",
    )?;
    let width_out = checked_conv_output_axis(
        config.input_w,
        config.kernel_w,
        config.stride_w,
        config.dilation_w,
        config.pad_left,
        config.pad_right,
        "kernel_w",
    )?;
    bounded_u32_product(
        &[config.batch, config.c_in, config.input_h, config.input_w],
        "batch",
    )?;
    bounded_u32_product(
        &[
            config.c_out,
            config.c_in_per_group(),
            config.kernel_h,
            config.kernel_w,
        ],
        "c_out",
    )?;
    bounded_u32_product(
        &[config.batch, config.c_out, height_out, width_out],
        "batch",
    )?;
    Ok((config, height_out, width_out))
}

#[allow(clippy::too_many_arguments)]
fn checked_conv_output_axis(
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    pad_before: usize,
    pad_after: usize,
    name: &str,
) -> Result<usize, TrainBackendError> {
    let effective = (dilation as u64)
        .checked_mul((kernel - 1) as u64)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| attribute_value(name, "arithmetic"))?;
    let padded = (input as u64)
        .checked_add(pad_before as u64)
        .and_then(|value| value.checked_add(pad_after as u64))
        .ok_or_else(|| attribute_value(name, "arithmetic"))?;
    if effective > u32::MAX as u64 || padded > u32::MAX as u64 {
        return Err(attribute_value(name, "axis_u32"));
    }
    if padded < effective {
        return Err(attribute_value(name, "output_nonzero"));
    }
    usize::try_from((padded - effective) / stride as u64 + 1)
        .map_err(|_| attribute_value(name, "axis_u32"))
}

fn conv2d_contract_scratch(
    config: &tritium_train::ops::conv2d::Conv2dCfg,
    height_out: usize,
    width_out: usize,
    execution: TrainExecutionV1,
) -> Result<u64, TrainBackendError> {
    let tile_rows =
        (height_out * width_out).min(tritium_train::ops::conv2d::CONV2D_PATCH_TILE_ROWS);
    let patch_columns = config.kernel_elements_per_output();
    let group_channels = config.c_out_per_group();
    let columns = tile_rows * patch_columns;
    let group_output = tile_rows * group_channels;
    let elements = match execution {
        TrainExecutionV1::Forward => config
            .output_elements()
            .checked_add(columns)
            .and_then(|value| value.checked_add(group_output)),
        TrainExecutionV1::Vjp => config
            .input_elements()
            .checked_add(config.weight_elements())
            .and_then(|value| value.checked_add(config.c_out))
            .and_then(|value| value.checked_add(columns))
            .and_then(|value| value.checked_add(group_output))
            .and_then(|value| value.checked_add(columns))
            .and_then(|value| value.checked_add(config.c_out_per_group() * patch_columns))
            .and_then(|value| value.checked_add(group_channels)),
        _ => Some(0),
    }
    .ok_or_else(shape_error)?;
    (elements as u64)
        .checked_mul(size_of::<f32>() as u64)
        .ok_or_else(shape_error)
}

fn conv2d_actual_scratch(
    config: &tritium_train::ops::conv2d::Conv2dCfg,
    height_out: usize,
    width_out: usize,
) -> Result<u64, TrainBackendError> {
    let tile_rows =
        (height_out * width_out).min(tritium_train::ops::conv2d::CONV2D_PATCH_TILE_ROWS);
    let patch_columns = config.kernel_elements_per_output();
    let group_channels = config.c_out_per_group();
    let elements = (tile_rows * patch_columns)
        .checked_mul(2)
        .and_then(|value| value.checked_add(tile_rows * group_channels))
        .and_then(|value| value.checked_add(group_channels * patch_columns))
        .and_then(|value| value.checked_add(group_channels))
        .ok_or_else(shape_error)?;
    (elements as u64)
        .checked_mul(size_of::<f32>() as u64)
        .ok_or_else(shape_error)
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

fn attribute_bool(request: &TrainRequestV1<'_>, name: &str) -> Result<bool, TrainBackendError> {
    let value = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or_else(|| roles("attributes"))?;
    let TrainAttributeValueV1::Bool(value) = value.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "bool",
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

fn input_bytes_any_shape<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<(&'a [u64], &'a [u8]), TrainBackendError> {
    let buffer = request
        .inputs
        .iter()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| roles("inputs"))?;
    match buffer.data {
        TrainBufferDataRefV1::Bytes(data) => Ok((buffer.shape, data)),
        TrainBufferDataRefV1::F32(_) => Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::DType {
                name: name.to_owned(),
                expected: TrainDTypeV1::Bytes,
                got: TrainDTypeV1::F32,
            },
        )),
        TrainBufferDataRefV1::U32(_) => Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::DType {
                name: name.to_owned(),
                expected: TrainDTypeV1::Bytes,
                got: TrainDTypeV1::U32,
            },
        )),
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

fn output_bytes<'a>(
    output: &'a mut TrainOutputV1<'_>,
    name: &str,
    shape: &[u64],
    len: usize,
) -> Result<&'a mut [u8], TrainBackendError> {
    let buffer = output
        .buffers
        .iter_mut()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| roles("outputs"))?;
    if buffer.shape != shape {
        return Err(shape_error());
    }
    match &mut buffer.data {
        TrainBufferDataMutV1::Bytes(data) if data.len() == len => Ok(data),
        TrainBufferDataMutV1::Bytes(_) => Err(shape_error()),
        TrainBufferDataMutV1::F32(_) => Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::DType {
                name: name.to_owned(),
                expected: TrainDTypeV1::Bytes,
                got: TrainDTypeV1::F32,
            },
        )),
        TrainBufferDataMutV1::U32(_) => Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::DType {
                name: name.to_owned(),
                expected: TrainDTypeV1::Bytes,
                got: TrainDTypeV1::U32,
            },
        )),
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
