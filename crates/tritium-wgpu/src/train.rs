//! Portable-training lifecycle adapter for the native wgpu device.

use tritium_spec::{
    BackendError, TrainAttributeValueV1, TrainBackendError, TrainBackendV1, TrainBufferDataMutV1,
    TrainBufferDataRefV1, TrainCapabilitiesV1, TrainDTypeV1, TrainExecutionV1, TrainLimitsV1,
    TrainOperationErrorV1, TrainOutputV1, TrainReceiptV1, TrainRequestV1, TrainingOpManifestV1,
    train_output_digest_v1, train_request_digest_v1,
};

use crate::{
    WgpuBackend,
    backend::{AdamWParams, AdamWScalars, AttentionParams, ConvParams, FsqParams, MuonParams},
    dispatch_catalog::portable_pointwise_selector_v1,
};

const OPERATIONS: &[&str] = &[
    "graph.ste_surrogate",
    "graph.salt_ste",
    "graph.lsq_ste",
    "graph.fsq",
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
    "graph.conv1d",
    "graph.conv2d",
    "graph.attention",
    "graph.relu2",
    "graph.silu",
    "graph.causal_mask",
    "graph.rope",
    "graph.rmsnorm",
    "graph.softmax",
    "loss.mse",
    "loss.softmax_cross_entropy",
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
const MAX_SALT_PLANES: u64 = 64;
const MAX_SALT_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CONV_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ATTENTION_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;

fn execution_id(execution: TrainExecutionV1) -> &'static str {
    match execution {
        TrainExecutionV1::Forward => "forward",
        TrainExecutionV1::Vjp => "vjp",
        TrainExecutionV1::Step => "step",
        TrainExecutionV1::Checkpoint => "checkpoint",
        TrainExecutionV1::Resume => "resume",
        TrainExecutionV1::Export => "export",
        TrainExecutionV1::Reload => "reload",
    }
}

fn catalog_selector(request: &TrainRequestV1<'_>, stage: usize) -> Result<u32, TrainBackendError> {
    portable_pointwise_selector_v1(request.operation, execution_id(request.execution), stage)
        .ok_or_else(|| invariant("pointwise dispatch selector missing from shared catalog"))
}

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

    /// Compile every shared portable dispatch module/entry point on selected
    /// physical adapter.
    ///
    /// # Errors
    /// Returns backend error when catalog WGSL or an entry point fails device
    /// validation.
    pub fn validate_dispatch_catalog(&self) -> Result<(), BackendError> {
        self.backend.validate_portable_dispatch_catalog()
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
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
            }
            ("graph.detach", TrainExecutionV1::Vjp) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
            }
            ("graph.scale_const", _) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    scalar,
                    auxiliary,
                )]
            }
            ("graph.add", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
            }
            ("graph.add", TrainExecutionV1::Vjp) => vec![
                self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                ),
                self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 1)?,
                    0.0,
                    auxiliary,
                ),
            ],
            ("graph.mul", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
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
                    self.backend.pointwise(
                        gradient,
                        right,
                        gradient,
                        catalog_selector(request, 0)?,
                        0.0,
                        auxiliary,
                    ),
                    self.backend.pointwise(
                        gradient,
                        left,
                        gradient,
                        catalog_selector(request, 1)?,
                        0.0,
                        auxiliary,
                    ),
                ]
            }
            ("graph.relu2", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
            }
            ("graph.relu2", TrainExecutionV1::Vjp) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
            }
            ("graph.silu", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
            }
            ("graph.silu", TrainExecutionV1::Vjp) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
            }
            ("graph.causal_mask", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
            }
            ("graph.causal_mask", TrainExecutionV1::Vjp) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
            }
            ("graph.softmax", TrainExecutionV1::Forward) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
            }
            ("graph.softmax", TrainExecutionV1::Vjp) => {
                vec![self.backend.pointwise(
                    first,
                    second,
                    first,
                    catalog_selector(request, 0)?,
                    0.0,
                    auxiliary,
                )]
            }
            _ => unreachable!(),
        };
        for (name, result) in output_names.iter().zip(results) {
            let result = result.map_err(wgpu_error)?;
            output_f32(output, name, shape, first.len())?.copy_from_slice(&result);
        }
        Ok(())
    }

    fn execute_salt(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<u64, TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["weight"],
            TrainExecutionV1::Vjp => &["weight", "grad_output"],
            _ => return Err(invariant("SALT STE received an illegal phase")),
        };
        let output_name = if request.execution == TrainExecutionV1::Forward {
            "result"
        } else {
            "grad_weight"
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["rows", "cols", "planes"],
            "attribute",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
        let rows = attribute_u64(request, "rows")?;
        let cols = attribute_u64(request, "cols")?;
        let planes = attribute_u64(request, "planes")?;
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
        let scratch_bytes = cols
            .checked_mul(core::mem::size_of::<f32>() as u64)
            .ok_or_else(|| attribute_value("cols", "scratch_bytes"))?;
        if request.execution == TrainExecutionV1::Forward && scratch_bytes > MAX_SALT_SCRATCH_BYTES
        {
            return Err(attribute_value("cols", "scratch_limit"));
        }
        let elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        if rows > u32::MAX as u64 || cols > u32::MAX as u64 || elements > usize::MAX as u64 {
            return Err(shape_error());
        }
        let shape = [rows, cols];
        let (weight_shape, weight) = input_f32(request, "weight")?;
        if weight_shape != shape || weight.len() != elements as usize {
            return Err(shape_error());
        }
        require_finite("weight", weight)?;
        if request.execution == TrainExecutionV1::Forward {
            let result = self
                .backend
                .salt(weight, rows as u32, cols as u32, planes as u32)
                .map_err(wgpu_error)?;
            output_f32(output, "result", &shape, elements as usize)?.copy_from_slice(&result);
            Ok(scratch_bytes)
        } else {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            let result = self
                .backend
                .pointwise(
                    gradient,
                    gradient,
                    gradient,
                    catalog_selector(request, 0)?,
                    0.0,
                    0,
                )
                .map_err(wgpu_error)?;
            output_f32(output, "grad_weight", &shape, elements as usize)?.copy_from_slice(&result);
            Ok(0)
        }
    }

    fn execute_fsq(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x"],
            TrainExecutionV1::Vjp => &["x", "grad_output"],
            _ => return Err(invariant("FSQ received an illegal phase")),
        };
        let output_name = if request.execution == TrainExecutionV1::Forward {
            "result"
        } else {
            "grad_x"
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
            &["channels", "len", "levels", "bound", "ste", "alpha", "seed"],
            "attribute",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
        let channels = attribute_u64(request, "channels")?;
        let len = attribute_u64(request, "len")?;
        if channels == 0 {
            return Err(attribute_value("channels", "positive"));
        }
        if len == 0 {
            return Err(attribute_value("len", "positive"));
        }
        let elements = channels.checked_mul(len).ok_or_else(shape_error)?;
        if channels > u32::MAX as u64 || len > u32::MAX as u64 || elements > u32::MAX as u64 {
            return Err(shape_error());
        }
        let levels = attribute_u32_list(request, "levels")?;
        if levels.len() != channels as usize {
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
        let shape = [channels, len];
        let (x_shape, x) = input_f32(request, "x")?;
        if x_shape != shape || x.len() != elements as usize {
            return Err(shape_error());
        }
        require_finite("x", x)?;
        let upstream = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient
        } else {
            x
        };
        let params = FsqParams::new(
            elements as u32,
            len as u32,
            bound,
            estimator,
            u32::from(request.execution == TrainExecutionV1::Vjp),
            alpha,
            seed,
        );
        let result = self
            .backend
            .fsq(x, upstream, levels, params)
            .map_err(wgpu_error)?;
        output_f32(output, output_name, &shape, elements as usize)?.copy_from_slice(&result);
        Ok(())
    }

    fn execute_rope(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let (input_name, output_name, inverse) = match request.execution {
            TrainExecutionV1::Forward => ("x", "result", false),
            TrainExecutionV1::Vjp => ("grad_output", "grad_x", true),
            _ => return Err(invariant("RoPE received an illegal phase")),
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
        let n_head = attribute_u64(request, "n_head")?;
        let head_dim = attribute_u64(request, "head_dim")?;
        if n_head == 0 {
            return Err(attribute_value("n_head", "positive"));
        }
        if head_dim == 0 {
            return Err(attribute_value("head_dim", "positive"));
        }
        if !head_dim.is_multiple_of(2) {
            return Err(attribute_value("head_dim", "even"));
        }
        if n_head > u32::MAX as u64 || head_dim > u32::MAX as u64 {
            return Err(shape_error());
        }
        let elements = positions
            .len()
            .checked_mul(n_head as usize)
            .and_then(|value| value.checked_mul(head_dim as usize))
            .ok_or_else(shape_error)?;
        let theta = attribute_f32(request, "theta")?;
        if theta <= 0.0 {
            return Err(attribute_value("theta", "positive"));
        }
        let shape = [positions.len() as u64, n_head, head_dim];
        let (input_shape, input) = input_f32(request, input_name)?;
        if input_shape != shape || input.len() != elements {
            return Err(shape_error());
        }
        require_finite(input_name, input)?;
        let target = output_f32(output, output_name, &shape, elements)?;
        if elements == 0 {
            return Ok(());
        }
        let result = self
            .backend
            .rope(
                input,
                &positions,
                n_head as u32,
                head_dim as u32,
                theta,
                inverse,
            )
            .map_err(wgpu_error)?;
        target.copy_from_slice(&result);
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
                .pointwise(
                    x,
                    weight,
                    gradient,
                    catalog_selector(request, 0)?,
                    eps,
                    cols as u32,
                )
                .map_err(wgpu_error)?;
            output_f32(output, "result", &matrix_shape, x.len())?.copy_from_slice(&result);
        } else {
            let grad_x = self
                .backend
                .pointwise(
                    x,
                    weight,
                    gradient,
                    catalog_selector(request, 0)?,
                    eps,
                    cols as u32,
                )
                .map_err(wgpu_error)?;
            let grad_weight_full = self
                .backend
                .pointwise(
                    x,
                    weight,
                    gradient,
                    catalog_selector(request, 1)?,
                    eps,
                    cols as u32,
                )
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
        let operation = catalog_selector(request, 0)?;
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

    fn execute_softmax_xent(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<(), TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["logits", "target"],
            TrainExecutionV1::Vjp => &["logits", "target", "grad_output"],
            _ => return Err(invariant("softmax cross-entropy received an illegal phase")),
        };
        let output_name = if request.execution == TrainExecutionV1::Forward {
            "result"
        } else {
            "grad_logits"
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
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            &[output_name],
            "outputs",
        )?;
        let rows = attribute_u64(request, "rows")?;
        let cols = attribute_u64(request, "cols")?;
        if rows == 0 {
            return Err(attribute_value("rows", "positive"));
        }
        if cols == 0 {
            return Err(attribute_value("cols", "positive"));
        }
        let elements = rows.checked_mul(cols).ok_or_else(shape_error)?;
        if rows > u32::MAX as u64 || cols > u32::MAX as u64 || elements > u32::MAX as u64 {
            return Err(shape_error());
        }
        let shape = [rows, cols];
        let (logits_shape, logits) = input_f32(request, "logits")?;
        let (target_shape, target) = input_f32(request, "target")?;
        if logits_shape != shape
            || target_shape != shape
            || logits.len() != elements as usize
            || target.len() != elements as usize
        {
            return Err(shape_error());
        }
        require_finite("logits", logits)?;
        require_finite("target", target)?;
        let gradient_scale = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if !gradient_shape.is_empty() || gradient.len() != 1 {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient[0] / rows as f32
        } else {
            0.0
        };
        let result = self
            .backend
            .softmax_xent(
                logits,
                target,
                rows as u32,
                cols as u32,
                gradient_scale,
                request.execution == TrainExecutionV1::Vjp,
            )
            .map_err(wgpu_error)?;
        let (output_shape, output_len): (&[u64], usize) =
            if request.execution == TrainExecutionV1::Forward {
                (&[], 1)
            } else {
                (&shape, elements as usize)
            };
        output_f32(output, output_name, output_shape, output_len)?.copy_from_slice(&result);
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
                .pointwise(
                    x,
                    bias,
                    gradient,
                    catalog_selector(request, 0)?,
                    0.0,
                    cols as u32,
                )
                .map_err(wgpu_error)?;
            output_f32(output, "result", &matrix_shape, x.len())?.copy_from_slice(&result);
        } else {
            let grad_x = self
                .backend
                .pointwise(
                    x,
                    bias,
                    gradient,
                    catalog_selector(request, 0)?,
                    0.0,
                    cols as u32,
                )
                .map_err(wgpu_error)?;
            let grad_bias_full = self
                .backend
                .pointwise(
                    x,
                    bias,
                    gradient,
                    catalog_selector(request, 1)?,
                    0.0,
                    cols as u32,
                )
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
            .pointwise(
                parameter,
                gradient,
                parameter,
                catalog_selector(request, 0)?,
                learning_rate,
                0,
            )
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
        let operation = catalog_selector(request, 0)?;
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
        let operation = catalog_selector(request, 0)?;
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
                    x,
                    weight,
                    gradient,
                    catalog_selector(request, 0)?,
                    0.0,
                    m as u32,
                    n as u32,
                    k as u32,
                    result_len,
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
                    catalog_selector(request, 0)?,
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
                    catalog_selector(request, 1)?,
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
                    activation,
                    weight,
                    scale,
                    catalog_selector(request, 0)?,
                    0.0,
                    m as u32,
                    n as u32,
                    k as u32,
                    result_len,
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
                    catalog_selector(request, 0)?,
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
                    catalog_selector(request, 1)?,
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
                    catalog_selector(request, 2)?,
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
                            catalog_selector(request, 0)?,
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
                .pointwise(
                    weight,
                    scale,
                    gradient,
                    catalog_selector(request, 0)?,
                    0.0,
                    cols as u32,
                )
                .map_err(wgpu_error)?;
            output_f32(output, "result", &weight_shape, weight.len())?.copy_from_slice(&result);
        } else {
            let grad_weight = self
                .backend
                .pointwise(
                    weight,
                    scale,
                    gradient,
                    catalog_selector(request, 0)?,
                    0.0,
                    cols as u32,
                )
                .map_err(wgpu_error)?;
            let grad_scale = self
                .backend
                .pointwise(scale, scale, scale, catalog_selector(request, 1)?, 0.0, 0)
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
                .pointwise(
                    weight,
                    alpha,
                    gradient,
                    catalog_selector(request, 0)?,
                    0.0,
                    cols as u32,
                )
                .map_err(wgpu_error)?;
            output_f32(output, "result", &weight_shape, weight.len())?.copy_from_slice(&result);
        } else {
            let grad_weight = self
                .backend
                .pointwise(
                    weight,
                    alpha,
                    gradient,
                    catalog_selector(request, 0)?,
                    0.0,
                    cols as u32,
                )
                .map_err(wgpu_error)?;
            let grad_alpha = self
                .backend
                .pointwise_sized(
                    weight,
                    alpha,
                    gradient,
                    catalog_selector(request, 1)?,
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
        cautious: bool,
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
            .adamw(parameter, gradient, moment1, moment2, params, cautious)
            .map_err(wgpu_error)?;
        output_f32(output, "parameter", shape, parameter.len())?
            .copy_from_slice(&updated_parameter);
        output_f32(output, "moment1", shape, parameter.len())?.copy_from_slice(&updated_moment1);
        output_f32(output, "moment2", shape, parameter.len())?.copy_from_slice(&updated_moment2);
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
            request.inputs.iter().map(|buffer| buffer.name),
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
            request.attributes.iter().map(|attribute| attribute.name),
            &["step", "lr", "beta1", "beta2", "eps", "weight_decay"],
            "attributes",
        )?;
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
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
        let (moment1_shape, moment1_q8) = input_bytes(request, "moment1_q8")?;
        let (moment2_shape, moment2_q8) = input_bytes(request, "moment2_q8")?;
        let (moment1_scale_shape, moment1_scale) = input_f32(request, "moment1_scale")?;
        let (moment2_scale_shape, moment2_scale) = input_f32(request, "moment2_scale")?;
        let len = parameter.len();
        let blocks = len.div_ceil(256);
        if len == 0
            || len > u32::MAX as usize
            || gradient_shape != shape
            || moment1_shape != [len as u64]
            || moment2_shape != [len as u64]
            || moment1_scale_shape != [blocks as u64]
            || moment2_scale_shape != [blocks as u64]
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
        let exponent = i32::try_from(step).unwrap_or(i32::MAX);
        let params = AdamWParams::new(
            len as u32,
            AdamWScalars {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                correction1: 1.0 - beta1.powi(exponent),
                correction2: 1.0 - beta2.powi(exponent),
                shrink: 1.0 - learning_rate * weight_decay,
            },
        );
        let updated = self
            .backend
            .int8_adamw(
                parameter,
                gradient,
                moment1_q8,
                moment2_q8,
                moment1_scale,
                moment2_scale,
                params,
            )
            .map_err(wgpu_error)?;
        output_f32(output, "parameter", shape, len)?.copy_from_slice(&updated.parameter);
        output_bytes(output, "moment1_q8", &[len as u64], len)?
            .copy_from_slice(&updated.moment1_q8);
        output_bytes(output, "moment2_q8", &[len as u64], len)?
            .copy_from_slice(&updated.moment2_q8);
        output_f32(output, "moment1_scale", &[blocks as u64], blocks)?
            .copy_from_slice(&updated.moment1_scale);
        output_f32(output, "moment2_scale", &[blocks as u64], blocks)?
            .copy_from_slice(&updated.moment2_scale);
        Ok(())
    }

    fn execute_attention(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<u64, TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["q", "k", "v"],
            TrainExecutionV1::Vjp => &["q", "k", "v", "grad_output"],
            _ => return Err(invariant("attention received an illegal phase")),
        };
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_q", "grad_k", "grad_v"],
            _ => unreachable!(),
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
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            expected_outputs,
            "outputs",
        )?;
        let config = attention_attributes(request)?;
        let query_len = bounded_u32_product(&[config.seq, config.n_head, config.head_dim], "seq")?;
        let kv_len = bounded_u32_product(&[config.seq, config.n_kv_head, config.head_dim], "seq")?;
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
            .checked_mul(4)
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
        let (actual_query_shape, q) = input_f32(request, "q")?;
        let (actual_key_shape, k) = input_f32(request, "k")?;
        let (actual_value_shape, v) = input_f32(request, "v")?;
        if actual_query_shape != query_shape
            || actual_key_shape != kv_shape
            || actual_value_shape != kv_shape
            || q.len() != query_len
            || k.len() != kv_len
            || v.len() != kv_len
        {
            return Err(shape_error());
        }
        require_finite("q", q)?;
        require_finite("k", k)?;
        require_finite("v", v)?;
        let grad_output = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != query_shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient
        } else {
            q
        };
        let result = self
            .backend
            .attention(
                q,
                k,
                v,
                grad_output,
                AttentionParams {
                    seq: config.seq as u32,
                    n_head: config.n_head as u32,
                    n_kv_head: config.n_kv_head as u32,
                    head_dim: config.head_dim as u32,
                    causal: u32::from(config.causal),
                    execution: u32::from(request.execution == TrainExecutionV1::Vjp),
                    padding_0: 0,
                    padding_1: 0,
                },
            )
            .map_err(wgpu_error)?;
        if request.execution == TrainExecutionV1::Forward {
            output_f32(output, "result", &query_shape, query_len)?
                .copy_from_slice(&result.result_or_grad_q);
        } else {
            output_f32(output, "grad_q", &query_shape, query_len)?
                .copy_from_slice(&result.result_or_grad_q);
            output_f32(output, "grad_k", &kv_shape, kv_len)?.copy_from_slice(&result.grad_k);
            output_f32(output, "grad_v", &kv_shape, kv_len)?.copy_from_slice(&result.grad_v);
        }
        let actual_scratch = probability_len as u64
            * if request.execution == TrainExecutionV1::Forward {
                4
            } else {
                8
            };
        Ok(actual_scratch)
    }

    fn execute_conv1d(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<u64, TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x", "weight", "scale"],
            TrainExecutionV1::Vjp => &["x", "weight", "scale", "grad_output"],
            _ => return Err(invariant("Conv1d received an illegal phase")),
        };
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_x", "grad_weight", "grad_scale"],
            _ => unreachable!(),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
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
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            expected_outputs,
            "outputs",
        )?;
        let (config, output_len) = conv1d_attributes(request)?;
        let scratch = conv1d_contract_scratch(&config, output_len, request.execution)?;
        if scratch > MAX_CONV_SCRATCH_BYTES {
            return Err(attribute_value("scratch", "limit_64_mib"));
        }
        let input_shape = [config.batch as u64, config.c_in as u64, config.l_in as u64];
        let weight_shape = [
            config.c_out as u64,
            (config.c_in / config.groups) as u64,
            config.k as u64,
        ];
        let scale_shape = [config.c_out as u64];
        let result_shape = [config.batch as u64, config.c_out as u64, output_len as u64];
        let (actual_input_shape, x) = input_f32(request, "x")?;
        let (actual_weight_shape, weight) = input_f32(request, "weight")?;
        let (actual_scale_shape, scale) = input_f32(request, "scale")?;
        if actual_input_shape != input_shape
            || actual_weight_shape != weight_shape
            || actual_scale_shape != scale_shape
        {
            return Err(shape_error());
        }
        require_finite("x", x)?;
        require_finite("weight", weight)?;
        require_finite("scale", scale)?;
        let grad_output = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != result_shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient
        } else {
            x
        };
        let result = self
            .backend
            .convolution(
                x,
                weight,
                scale,
                grad_output,
                ConvParams {
                    batch: config.batch as u32,
                    c_in: config.c_in as u32,
                    c_out: config.c_out as u32,
                    input_h: 1,
                    input_w: config.l_in as u32,
                    kernel_h: 1,
                    kernel_w: config.k as u32,
                    stride_h: 1,
                    stride_w: config.stride as u32,
                    dilation_h: 1,
                    dilation_w: config.dilation as u32,
                    pad_top: 0,
                    pad_left: config.pad_left as u32,
                    groups: config.groups as u32,
                    output_h: 1,
                    output_w: output_len as u32,
                    execution: u32::from(request.execution == TrainExecutionV1::Vjp),
                    pad_bottom: 0,
                    pad_right: config.pad_right as u32,
                    padding: 0,
                },
            )
            .map_err(wgpu_error)?;
        if request.execution == TrainExecutionV1::Forward {
            output_f32(output, "result", &result_shape, result.result.len())?
                .copy_from_slice(&result.result);
        } else {
            output_f32(output, "grad_x", &input_shape, x.len())?.copy_from_slice(&result.result);
            output_f32(output, "grad_weight", &weight_shape, weight.len())?
                .copy_from_slice(&result.grad_weight);
            output_f32(output, "grad_scale", &scale_shape, scale.len())?
                .copy_from_slice(&result.grad_scale);
        }
        Ok(scratch)
    }

    fn execute_conv2d(
        &self,
        request: &TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<u64, TrainBackendError> {
        let expected_inputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["x", "weight", "scale"],
            TrainExecutionV1::Vjp => &["x", "weight", "scale", "grad_output"],
            _ => return Err(invariant("Conv2d received an illegal phase")),
        };
        let expected_outputs: &[&str] = match request.execution {
            TrainExecutionV1::Forward => &["result"],
            TrainExecutionV1::Vjp => &["grad_x", "grad_weight", "grad_scale"],
            _ => unreachable!(),
        };
        require_names(
            request.inputs.iter().map(|buffer| buffer.name),
            expected_inputs,
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
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
        require_names(
            output.buffers.iter().map(|buffer| buffer.name),
            expected_outputs,
            "outputs",
        )?;
        let (config, output_h, output_w) = conv2d_attributes(request)?;
        let scratch = conv2d_contract_scratch(&config, output_h, output_w, request.execution)?;
        if scratch > MAX_CONV_SCRATCH_BYTES {
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
            (config.c_in / config.groups) as u64,
            config.kernel_h as u64,
            config.kernel_w as u64,
        ];
        let scale_shape = [config.c_out as u64];
        let result_shape = [
            config.batch as u64,
            config.c_out as u64,
            output_h as u64,
            output_w as u64,
        ];
        let (actual_input_shape, x) = input_f32(request, "x")?;
        let (actual_weight_shape, weight) = input_f32(request, "weight")?;
        let (actual_scale_shape, scale) = input_f32(request, "scale")?;
        if actual_input_shape != input_shape
            || actual_weight_shape != weight_shape
            || actual_scale_shape != scale_shape
        {
            return Err(shape_error());
        }
        require_finite("x", x)?;
        require_finite("weight", weight)?;
        require_finite("scale", scale)?;
        let grad_output = if request.execution == TrainExecutionV1::Vjp {
            let (gradient_shape, gradient) = input_f32(request, "grad_output")?;
            if gradient_shape != result_shape {
                return Err(shape_error());
            }
            require_finite("grad_output", gradient)?;
            gradient
        } else {
            x
        };
        let result = self
            .backend
            .convolution(
                x,
                weight,
                scale,
                grad_output,
                ConvParams {
                    batch: config.batch as u32,
                    c_in: config.c_in as u32,
                    c_out: config.c_out as u32,
                    input_h: config.input_h as u32,
                    input_w: config.input_w as u32,
                    kernel_h: config.kernel_h as u32,
                    kernel_w: config.kernel_w as u32,
                    stride_h: config.stride_h as u32,
                    stride_w: config.stride_w as u32,
                    dilation_h: config.dilation_h as u32,
                    dilation_w: config.dilation_w as u32,
                    pad_top: config.pad_top as u32,
                    pad_left: config.pad_left as u32,
                    groups: config.groups as u32,
                    output_h: output_h as u32,
                    output_w: output_w as u32,
                    execution: u32::from(request.execution == TrainExecutionV1::Vjp),
                    pad_bottom: config.pad_bottom as u32,
                    pad_right: config.pad_right as u32,
                    padding: 0,
                },
            )
            .map_err(wgpu_error)?;
        if request.execution == TrainExecutionV1::Forward {
            output_f32(output, "result", &result_shape, result.result.len())?
                .copy_from_slice(&result.result);
        } else {
            output_f32(output, "grad_x", &input_shape, x.len())?.copy_from_slice(&result.result);
            output_f32(output, "grad_weight", &weight_shape, weight.len())?
                .copy_from_slice(&result.grad_weight);
            output_f32(output, "grad_scale", &scale_shape, scale.len())?
                .copy_from_slice(&result.grad_scale);
        }
        Ok(scratch)
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
            request.inputs.iter().map(|buffer| buffer.name),
            &["parameter", "gradient", "momentum"],
            "inputs",
        )?;
        require_names(
            request.attributes.iter().map(|attribute| attribute.name),
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
            output.buffers.iter().map(|buffer| buffer.name),
            &["parameter", "momentum"],
            "outputs",
        )?;
        let step = attribute_u64(request, "step")?;
        let learning_rate = attribute_f32(request, "lr")?;
        let momentum_decay = attribute_f32(request, "momentum")?;
        let weight_decay = attribute_f32(request, "weight_decay")?;
        let rows = attribute_u64(request, "rows")?;
        let cols = attribute_u64(request, "cols")?;
        let steps = attribute_u64(request, "ns_steps")?;
        if step == 0 {
            return Err(attribute_value("step", "one_based"));
        }
        if learning_rate < 0.0 {
            return Err(attribute_value("lr", "nonnegative"));
        }
        if !(0.0..1.0).contains(&momentum_decay) {
            return Err(attribute_value("momentum", "unit_interval_open"));
        }
        if weight_decay < 0.0 {
            return Err(attribute_value("weight_decay", "nonnegative"));
        }
        if rows == 0 || rows > u32::MAX as u64 {
            return Err(attribute_value("rows", "positive_u32"));
        }
        if cols == 0 || cols > u32::MAX as u64 {
            return Err(attribute_value("cols", "positive_u32"));
        }
        if steps == 0 {
            return Err(attribute_value("ns_steps", "positive"));
        }
        if steps > 32 {
            return Err(attribute_value("ns_steps", "max_32"));
        }
        let len = rows.checked_mul(cols).ok_or_else(shape_error)?;
        if len > u32::MAX as u64 {
            return Err(shape_error());
        }
        let expected_shape = [rows, cols];
        let (parameter_shape, parameter) = input_f32(request, "parameter")?;
        let (gradient_shape, gradient) = input_f32(request, "gradient")?;
        let (momentum_shape, momentum) = input_f32(request, "momentum")?;
        if parameter_shape != expected_shape
            || gradient_shape != expected_shape
            || momentum_shape != expected_shape
            || parameter.len() != len as usize
        {
            return Err(shape_error());
        }
        require_finite("parameter", parameter)?;
        require_finite("gradient", gradient)?;
        require_finite("momentum", momentum)?;
        let scale = learning_rate * (rows.max(cols) as f32).sqrt();
        let shrink = 1.0 - learning_rate * weight_decay;
        let params = MuonParams::new(
            rows as u32,
            cols as u32,
            steps as u32,
            momentum_decay,
            scale,
            shrink,
        );
        let (updated_parameter, updated_momentum) = self
            .backend
            .muon(parameter, gradient, momentum, params)
            .map_err(wgpu_error)?;
        output_f32(output, "parameter", &expected_shape, len as usize)?
            .copy_from_slice(&updated_parameter);
        output_f32(output, "momentum", &expected_shape, len as usize)?
            .copy_from_slice(&updated_momentum);
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
        } else if request.operation == "loss.softmax_cross_entropy" {
            self.execute_softmax_xent(&request, output)?;
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
        } else if request.operation == "graph.salt_ste" {
            self.execute_salt(&request, output)?
        } else if request.operation == "graph.fsq" {
            self.execute_fsq(&request, output)?;
            0
        } else if request.operation == "graph.rope" {
            self.execute_rope(&request, output)?;
            0
        } else if request.operation == "graph.conv1d" {
            self.execute_conv1d(&request, output)?
        } else if request.operation == "graph.conv2d" {
            self.execute_conv2d(&request, output)?
        } else if request.operation == "graph.attention" {
            self.execute_attention(&request, output)?
        } else if request.operation == "graph.lsq_ste" {
            self.execute_lsq(&request, output)?;
            0
        } else if request.operation == "optimizer.adamw" {
            self.execute_adamw(&request, output, false)?;
            0
        } else if request.operation == "optimizer.cautious_adamw" {
            self.execute_adamw(&request, output, true)?;
            0
        } else if request.operation == "optimizer.int8_adamw" {
            self.execute_int8_adamw(&request, output)?;
            0
        } else if request.operation == "optimizer.muon" {
            self.execute_muon(&request, output)?;
            0
        } else {
            self.execute_pointwise(&request, output)?;
            0
        };
        Ok(TrainReceiptV1 {
            backend_id: "wgpu.portable.v1:wgpu".to_owned(),
            backend_build: backend_build_identity(),
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

fn backend_build_identity() -> String {
    format!(
        "{}@{}+{}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("TRITIUM_SOURCE_ID")
    )
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

fn input_bytes<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<(&'a [u64], &'a [u8]), TrainBackendError> {
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
        TrainBufferDataRefV1::Bytes(data) => Ok((buffer.shape, data)),
        TrainBufferDataRefV1::F32(_) => Err(dtype_error(name, TrainDTypeV1::F32)),
        TrainBufferDataRefV1::U32(_) => Err(dtype_error(name, TrainDTypeV1::U32)),
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
        .ok_or({
            TrainBackendError::InvalidOperation(TrainOperationErrorV1::Roles {
                namespace: "outputs",
            })
        })?;
    if buffer.shape != shape {
        return Err(shape_error());
    }
    match &mut buffer.data {
        TrainBufferDataMutV1::Bytes(data) if data.len() == len => Ok(data),
        TrainBufferDataMutV1::Bytes(_) => Err(shape_error()),
        TrainBufferDataMutV1::F32(_) => Err(dtype_error(name, TrainDTypeV1::F32)),
        TrainBufferDataMutV1::U32(_) => Err(dtype_error(name, TrainDTypeV1::U32)),
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

fn attribute_usize(request: &TrainRequestV1<'_>, name: &str) -> Result<usize, TrainBackendError> {
    usize::try_from(attribute_u64(request, name)?).map_err(|_| shape_error())
}

fn attribute_bool(request: &TrainRequestV1<'_>, name: &str) -> Result<bool, TrainBackendError> {
    let attribute = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or({
            TrainBackendError::InvalidOperation(TrainOperationErrorV1::Roles {
                namespace: "attributes",
            })
        })?;
    let TrainAttributeValueV1::Bool(value) = attribute.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "bool",
            },
        ));
    };
    Ok(value)
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
    let output_len = checked_conv_output_axis(
        config.l_in,
        config.k,
        config.stride,
        config.dilation,
        config.pad_left,
        config.pad_right,
        "k",
    )?;
    let maximum_position = ((output_len - 1) as u64)
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
    bounded_u32_product(
        &[config.c_out, config.c_in / config.groups, config.k],
        "c_out",
    )?;
    bounded_u32_product(&[config.batch, config.c_out, output_len], "batch")?;
    Ok((config, output_len))
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
    let output_h = checked_conv_output_axis(
        config.input_h,
        config.kernel_h,
        config.stride_h,
        config.dilation_h,
        config.pad_top,
        config.pad_bottom,
        "kernel_h",
    )?;
    let output_w = checked_conv_output_axis(
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
            config.c_in / config.groups,
            config.kernel_h,
            config.kernel_w,
        ],
        "c_out",
    )?;
    bounded_u32_product(&[config.batch, config.c_out, output_h, output_w], "batch")?;
    Ok((config, output_h, output_w))
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

fn conv1d_contract_scratch(
    config: &tritium_train::ops::conv1d::Conv1dCfg,
    output_len: usize,
    execution: TrainExecutionV1,
) -> Result<u64, TrainBackendError> {
    let input = config.batch * config.c_in * config.l_in;
    let patch_columns = (config.c_in / config.groups) * config.k;
    let weight = config.c_out * patch_columns;
    let columns = output_len * patch_columns;
    let group_output = output_len * (config.c_out / config.groups);
    let elements = match execution {
        TrainExecutionV1::Forward => config
            .batch
            .checked_mul(config.c_out)
            .and_then(|value| value.checked_mul(output_len))
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
    (elements as u64).checked_mul(4).ok_or_else(shape_error)
}

fn conv2d_contract_scratch(
    config: &tritium_train::ops::conv2d::Conv2dCfg,
    output_h: usize,
    output_w: usize,
    execution: TrainExecutionV1,
) -> Result<u64, TrainBackendError> {
    let tile_rows = (output_h * output_w).min(32);
    let patch_columns = (config.c_in / config.groups) * config.kernel_h * config.kernel_w;
    let group_channels = config.c_out / config.groups;
    let columns = tile_rows * patch_columns;
    let group_output = tile_rows * group_channels;
    let output_elements = config.batch * config.c_out * output_h * output_w;
    let input_elements = config.batch * config.c_in * config.input_h * config.input_w;
    let weight_elements = config.c_out * patch_columns;
    let elements = match execution {
        TrainExecutionV1::Forward => output_elements
            .checked_add(columns)
            .and_then(|value| value.checked_add(group_output)),
        TrainExecutionV1::Vjp => input_elements
            .checked_add(weight_elements)
            .and_then(|value| value.checked_add(config.c_out))
            .and_then(|value| value.checked_add(columns))
            .and_then(|value| value.checked_add(group_output))
            .and_then(|value| value.checked_add(columns))
            .and_then(|value| value.checked_add(group_channels * patch_columns))
            .and_then(|value| value.checked_add(group_channels)),
        _ => Some(0),
    }
    .ok_or_else(shape_error)?;
    (elements as u64).checked_mul(4).ok_or_else(shape_error)
}

fn attribute_text<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<&'a str, TrainBackendError> {
    let attribute = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or({
            TrainBackendError::InvalidOperation(TrainOperationErrorV1::Roles {
                namespace: "attributes",
            })
        })?;
    let TrainAttributeValueV1::Text(value) = attribute.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "text",
            },
        ));
    };
    Ok(value)
}

fn attribute_u32_list<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<&'a [u32], TrainBackendError> {
    let attribute = request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .ok_or({
            TrainBackendError::InvalidOperation(TrainOperationErrorV1::Roles {
                namespace: "attributes",
            })
        })?;
    let TrainAttributeValueV1::U32List(value) = attribute.value else {
        return Err(TrainBackendError::InvalidOperation(
            TrainOperationErrorV1::AttributeType {
                name: name.to_owned(),
                expected: "u32_list",
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
