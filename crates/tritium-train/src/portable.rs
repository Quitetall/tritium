//! CPU reference adapter for the plan-0049 portable-training seam.

use core::f32::consts::PI;
use std::io::Cursor as IoCursor;
use std::sync::OnceLock;

use blake3::Hasher;
use tritium_format::salt_v2_package::SaltV2PackageReader;
use tritium_spec::{
    TrainAttributeValueV1, TrainBackendError, TrainBackendV1, TrainBufferDataMutV1,
    TrainBufferDataRefV1, TrainCapabilitiesV1, TrainDTypeV1, TrainExecutionV1, TrainLimitsV1,
    TrainOperationErrorV1, TrainOutputV1, TrainReceiptV1, TrainRequestV1, TrainingOpManifestV1,
    train_output_digest_v1, train_request_digest_v1,
};

use crate::{
    AdamState, AdamW, CautiousAdamW, INT8_ADAM_BLOCK, Int8AdamState, Int8AdamW, Muon, MuonState,
    Optimizer, Sgd, SgdState,
    checkpoint::{Checkpoint, CheckpointError, LeafCheckpoint, read_checkpoint, write_checkpoint},
    ops::{attention, conv1d, conv2d},
};

const BACKEND_ID: &str = "cpu.reference.v1";
const MAX_SALT_PLANES: u64 = 64;
const MAX_SALT_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CONV_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ATTENTION_SCRATCH_BYTES: u64 = 64 * 1024 * 1024;
// `SaltV2PackageReader::new_strict` retains parsed tensor/index metadata and uses
// a bounded 1,024-plane validation batch. Eight input bytes of allowance per
// encoded byte cover persistent metadata plus batch payload/scales; 128 KiB
// covers descriptors, one decoded allocation tile, and fixed hash state.
const SALT_V2_VALIDATION_FIXED_SCRATCH_BYTES: u64 = 128 * 1024;
const CPU_LIMITS: TrainLimitsV1 = TrainLimitsV1 {
    max_rank: u32::MAX,
    max_elements: usize::MAX as u64,
    max_bytes: usize::MAX as u64,
};

#[derive(Clone, Copy)]
enum CpuOperation {
    SteSurrogate,
    SaltSte,
    LsqSte,
    Fsq,
    DenseMatmul,
    TernaryMatmul,
    Transpose,
    EmbeddingGather,
    SliceCols,
    ConcatCols,
    Detach,
    ScaleConst,
    Bias,
    Add,
    Mul,
    Conv1d,
    Conv2d,
    Relu2,
    Silu,
    Rmsnorm,
    Softmax,
    CausalMask,
    Rope,
    Attention,
    Mse,
    SoftmaxCrossEntropy,
    Sgd,
    AdamW,
    CautiousAdamW,
    Int8AdamW,
    Muon,
    Checkpoint,
    Resume,
    Export,
    Reload,
}

#[derive(Clone, Copy)]
struct CpuOperationEntry {
    id: &'static str,
    operation: CpuOperation,
}

const CPU_OPERATIONS: &[CpuOperationEntry] = &[
    CpuOperationEntry {
        id: "graph.ste_surrogate",
        operation: CpuOperation::SteSurrogate,
    },
    CpuOperationEntry {
        id: "graph.salt_ste",
        operation: CpuOperation::SaltSte,
    },
    CpuOperationEntry {
        id: "graph.lsq_ste",
        operation: CpuOperation::LsqSte,
    },
    CpuOperationEntry {
        id: "graph.fsq",
        operation: CpuOperation::Fsq,
    },
    CpuOperationEntry {
        id: "graph.dense_matmul",
        operation: CpuOperation::DenseMatmul,
    },
    CpuOperationEntry {
        id: "graph.ternary_matmul",
        operation: CpuOperation::TernaryMatmul,
    },
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
        id: "graph.conv1d",
        operation: CpuOperation::Conv1d,
    },
    CpuOperationEntry {
        id: "graph.conv2d",
        operation: CpuOperation::Conv2d,
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
        id: "graph.rmsnorm",
        operation: CpuOperation::Rmsnorm,
    },
    CpuOperationEntry {
        id: "graph.softmax",
        operation: CpuOperation::Softmax,
    },
    CpuOperationEntry {
        id: "graph.causal_mask",
        operation: CpuOperation::CausalMask,
    },
    CpuOperationEntry {
        id: "graph.rope",
        operation: CpuOperation::Rope,
    },
    CpuOperationEntry {
        id: "graph.attention",
        operation: CpuOperation::Attention,
    },
    CpuOperationEntry {
        id: "loss.mse",
        operation: CpuOperation::Mse,
    },
    CpuOperationEntry {
        id: "loss.softmax_cross_entropy",
        operation: CpuOperation::SoftmaxCrossEntropy,
    },
    CpuOperationEntry {
        id: "optimizer.sgd",
        operation: CpuOperation::Sgd,
    },
    CpuOperationEntry {
        id: "optimizer.adamw",
        operation: CpuOperation::AdamW,
    },
    CpuOperationEntry {
        id: "optimizer.cautious_adamw",
        operation: CpuOperation::CautiousAdamW,
    },
    CpuOperationEntry {
        id: "optimizer.int8_adamw",
        operation: CpuOperation::Int8AdamW,
    },
    CpuOperationEntry {
        id: "optimizer.muon",
        operation: CpuOperation::Muon,
    },
    CpuOperationEntry {
        id: "lifecycle.checkpoint",
        operation: CpuOperation::Checkpoint,
    },
    CpuOperationEntry {
        id: "lifecycle.resume",
        operation: CpuOperation::Resume,
    },
    CpuOperationEntry {
        id: "lifecycle.export",
        operation: CpuOperation::Export,
    },
    CpuOperationEntry {
        id: "lifecycle.reload",
        operation: CpuOperation::Reload,
    },
];

struct OperationSchema {
    inputs: &'static [&'static str],
    attributes: &'static [&'static str],
    outputs: &'static [&'static str],
}

const STE_FORWARD: OperationSchema = OperationSchema {
    inputs: &["weight", "scale"],
    attributes: &["rows", "cols"],
    outputs: &["result"],
};
const STE_VJP: OperationSchema = OperationSchema {
    inputs: &["weight", "scale", "grad_output"],
    attributes: &["rows", "cols"],
    outputs: &["grad_weight", "grad_scale"],
};
const SALT_FORWARD: OperationSchema = OperationSchema {
    inputs: &["weight"],
    attributes: &["rows", "cols", "planes"],
    outputs: &["result"],
};
const SALT_VJP: OperationSchema = OperationSchema {
    inputs: &["weight", "grad_output"],
    attributes: &["rows", "cols", "planes"],
    outputs: &["grad_weight"],
};
const LSQ_FORWARD: OperationSchema = OperationSchema {
    inputs: &["weight", "alpha"],
    attributes: &["rows", "cols"],
    outputs: &["result"],
};
const LSQ_VJP: OperationSchema = OperationSchema {
    inputs: &["weight", "alpha", "grad_output"],
    attributes: &["rows", "cols"],
    outputs: &["grad_weight", "grad_alpha"],
};
const FSQ_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x"],
    attributes: &["channels", "len", "levels", "bound", "ste", "alpha", "seed"],
    outputs: &["result"],
};
const FSQ_VJP: OperationSchema = OperationSchema {
    inputs: &["x", "grad_output"],
    attributes: &["channels", "len", "levels", "bound", "ste", "alpha", "seed"],
    outputs: &["grad_x"],
};

const ADD_FORWARD: OperationSchema = OperationSchema {
    inputs: &["left", "right"],
    attributes: &[],
    outputs: &["result"],
};
const DENSE_MATMUL_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x", "weight"],
    attributes: &["m", "n", "k"],
    outputs: &["result"],
};
const DENSE_MATMUL_VJP: OperationSchema = OperationSchema {
    inputs: &["x", "weight", "grad_output"],
    attributes: &["m", "n", "k"],
    outputs: &["grad_x", "grad_weight"],
};
const TERNARY_MATMUL_FORWARD: OperationSchema = OperationSchema {
    inputs: &["activation", "weight", "scale"],
    attributes: &["m", "n", "k"],
    outputs: &["result"],
};
const TERNARY_MATMUL_VJP: OperationSchema = OperationSchema {
    inputs: &["activation", "weight", "scale", "grad_output"],
    attributes: &["m", "n", "k"],
    outputs: &["grad_activation", "grad_weight", "grad_scale"],
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
const CONV1D_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x", "weight", "scale"],
    attributes: &[
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
    outputs: &["result"],
};
const CONV1D_VJP: OperationSchema = OperationSchema {
    inputs: &["x", "weight", "scale", "grad_output"],
    attributes: CONV1D_FORWARD.attributes,
    outputs: &["grad_x", "grad_weight", "grad_scale"],
};
const CONV2D_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x", "weight", "scale"],
    attributes: &[
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
    outputs: &["result"],
};
const CONV2D_VJP: OperationSchema = OperationSchema {
    inputs: &["x", "weight", "scale", "grad_output"],
    attributes: CONV2D_FORWARD.attributes,
    outputs: &["grad_x", "grad_weight", "grad_scale"],
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
const RMSNORM_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x", "weight"],
    attributes: &["rows", "cols", "eps"],
    outputs: &["result"],
};
const RMSNORM_VJP: OperationSchema = OperationSchema {
    inputs: &["x", "weight", "grad_output"],
    attributes: &["rows", "cols", "eps"],
    outputs: &["grad_x", "grad_weight"],
};
const MATRIX_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x"],
    attributes: &["rows", "cols"],
    outputs: &["result"],
};
const SOFTMAX_VJP: OperationSchema = OperationSchema {
    inputs: &["x", "grad_output"],
    attributes: &["rows", "cols"],
    outputs: &["grad_x"],
};
const CAUSAL_MASK_VJP: OperationSchema = OperationSchema {
    inputs: &["grad_output"],
    attributes: &["rows", "cols"],
    outputs: &["grad_x"],
};
const ROPE_FORWARD: OperationSchema = OperationSchema {
    inputs: &["x"],
    attributes: &["positions", "n_head", "head_dim", "theta"],
    outputs: &["result"],
};
const ROPE_VJP: OperationSchema = OperationSchema {
    inputs: &["grad_output"],
    attributes: &["positions", "n_head", "head_dim", "theta"],
    outputs: &["grad_x"],
};
const ATTENTION_FORWARD: OperationSchema = OperationSchema {
    inputs: &["q", "k", "v"],
    attributes: &["seq", "n_head", "n_kv_head", "head_dim", "causal"],
    outputs: &["result"],
};
const ATTENTION_VJP: OperationSchema = OperationSchema {
    inputs: &["q", "k", "v", "grad_output"],
    attributes: ATTENTION_FORWARD.attributes,
    outputs: &["grad_q", "grad_k", "grad_v"],
};
const SOFTMAX_XENT_FORWARD: OperationSchema = OperationSchema {
    inputs: &["logits", "target"],
    attributes: &["rows", "cols"],
    outputs: &["result"],
};
const SOFTMAX_XENT_VJP: OperationSchema = OperationSchema {
    inputs: &["logits", "target", "grad_output"],
    attributes: &["rows", "cols"],
    outputs: &["grad_logits"],
};
const SGD_STEP: OperationSchema = OperationSchema {
    inputs: &["parameter", "gradient"],
    attributes: &["step", "lr"],
    outputs: &["parameter"],
};
const ADAM_STEP: OperationSchema = OperationSchema {
    inputs: &["parameter", "gradient", "moment1", "moment2"],
    attributes: &["step", "lr", "beta1", "beta2", "eps", "weight_decay"],
    outputs: &["parameter", "moment1", "moment2"],
};
const INT8_ADAM_STEP: OperationSchema = OperationSchema {
    inputs: &[
        "parameter",
        "gradient",
        "moment1_q8",
        "moment2_q8",
        "moment1_scale",
        "moment2_scale",
    ],
    attributes: ADAM_STEP.attributes,
    outputs: &[
        "parameter",
        "moment1_q8",
        "moment2_q8",
        "moment1_scale",
        "moment2_scale",
    ],
};
const MUON_STEP: OperationSchema = OperationSchema {
    inputs: &["parameter", "gradient", "momentum"],
    attributes: &[
        "step",
        "lr",
        "momentum",
        "weight_decay",
        "rows",
        "cols",
        "ns_steps",
    ],
    outputs: &["parameter", "momentum"],
};
const EXPORT: OperationSchema = OperationSchema {
    inputs: &["package"],
    attributes: &["format"],
    outputs: &["artifact"],
};
const RELOAD: OperationSchema = OperationSchema {
    inputs: &["artifact"],
    attributes: &["format"],
    outputs: &["package"],
};

/// Complete CPU semantic-reference adapter for `TrainBackendV1` manifest v1.
///
/// Every advertised forward/VJP, optimizer and lifecycle operation passes the
/// canonical corpus through this exact seam. Accelerator residency and physical
/// device receipts are separate backend gates.
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
            dtypes: vec![TrainDTypeV1::F32, TrainDTypeV1::U32, TrainDTypeV1::Bytes],
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
            (CpuOperation::SaltSte, TrainExecutionV1::Forward) => {
                require_contract(&request, output, &SALT_FORWARD)?;
            }
            (CpuOperation::Conv1d, TrainExecutionV1::Forward) => {
                require_contract(&request, output, &CONV1D_FORWARD)?;
            }
            (CpuOperation::Conv1d, TrainExecutionV1::Vjp) => {
                require_contract(&request, output, &CONV1D_VJP)?;
            }
            (CpuOperation::Conv2d, TrainExecutionV1::Forward) => {
                require_contract(&request, output, &CONV2D_FORWARD)?;
            }
            (CpuOperation::Conv2d, TrainExecutionV1::Vjp) => {
                require_contract(&request, output, &CONV2D_VJP)?;
            }
            (CpuOperation::Attention, TrainExecutionV1::Forward) => {
                require_contract(&request, output, &ATTENTION_FORWARD)?;
            }
            (CpuOperation::Attention, TrainExecutionV1::Vjp) => {
                require_contract(&request, output, &ATTENTION_VJP)?;
            }
            (CpuOperation::AdamW | CpuOperation::CautiousAdamW, TrainExecutionV1::Step) => {
                require_contract(&request, output, &ADAM_STEP)?;
            }
            (CpuOperation::Int8AdamW, TrainExecutionV1::Step) => {
                require_contract(&request, output, &INT8_ADAM_STEP)?;
            }
            (CpuOperation::Muon, TrainExecutionV1::Step) => {
                require_contract(&request, output, &MUON_STEP)?;
            }
            (CpuOperation::Checkpoint, TrainExecutionV1::Checkpoint) => {
                require_checkpoint_contract(&request, output)?;
            }
            (CpuOperation::Resume, TrainExecutionV1::Resume) => {
                require_resume_contract(&request, output)?;
            }
            (CpuOperation::Export, TrainExecutionV1::Export) => {
                require_contract(&request, output, &EXPORT)?;
            }
            (CpuOperation::Reload, TrainExecutionV1::Reload) => {
                require_contract(&request, output, &RELOAD)?;
            }
            _ => {}
        }
        let scratch_bytes =
            operation_scratch_bytes(operation.operation, request.execution, &request)?;
        if matches!(
            operation.operation,
            CpuOperation::Conv1d | CpuOperation::Conv2d
        ) && scratch_bytes > MAX_CONV_SCRATCH_BYTES
        {
            return Err(attribute_value("scratch", "limit_64_mib"));
        }
        if matches!(operation.operation, CpuOperation::Attention)
            && scratch_bytes > MAX_ATTENTION_SCRATCH_BYTES
        {
            return Err(attribute_value("scratch", "limit_64_mib"));
        }
        match (operation.operation, request.execution) {
            (CpuOperation::SteSurrogate, TrainExecutionV1::Forward) => {
                ste_surrogate_forward(&request, output)?;
            }
            (CpuOperation::SteSurrogate, TrainExecutionV1::Vjp) => {
                ste_surrogate_vjp(&request, output)?;
            }
            (CpuOperation::SaltSte, TrainExecutionV1::Forward) => {
                salt_ste_forward(&request, output)?;
            }
            (CpuOperation::SaltSte, TrainExecutionV1::Vjp) => {
                salt_ste_vjp(&request, output)?;
            }
            (CpuOperation::LsqSte, TrainExecutionV1::Forward) => {
                lsq_ste_forward(&request, output)?;
            }
            (CpuOperation::LsqSte, TrainExecutionV1::Vjp) => {
                lsq_ste_vjp(&request, output)?;
            }
            (CpuOperation::Fsq, TrainExecutionV1::Forward) => fsq_forward(&request, output)?,
            (CpuOperation::Fsq, TrainExecutionV1::Vjp) => fsq_vjp(&request, output)?,
            (CpuOperation::DenseMatmul, TrainExecutionV1::Forward) => {
                dense_matmul_forward(&request, output)?;
            }
            (CpuOperation::DenseMatmul, TrainExecutionV1::Vjp) => {
                dense_matmul_vjp(&request, output)?;
            }
            (CpuOperation::TernaryMatmul, TrainExecutionV1::Forward) => {
                ternary_matmul_forward(&request, output)?;
            }
            (CpuOperation::TernaryMatmul, TrainExecutionV1::Vjp) => {
                ternary_matmul_vjp(&request, output)?;
            }
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
            (CpuOperation::Conv1d, TrainExecutionV1::Forward) => {
                conv1d_forward(&request, output)?;
            }
            (CpuOperation::Conv1d, TrainExecutionV1::Vjp) => conv1d_vjp(&request, output)?,
            (CpuOperation::Conv2d, TrainExecutionV1::Forward) => {
                conv2d_forward(&request, output)?;
            }
            (CpuOperation::Conv2d, TrainExecutionV1::Vjp) => conv2d_vjp(&request, output)?,
            (CpuOperation::Relu2, TrainExecutionV1::Forward) => relu2_forward(&request, output)?,
            (CpuOperation::Relu2, TrainExecutionV1::Vjp) => relu2_vjp(&request, output)?,
            (CpuOperation::Silu, TrainExecutionV1::Forward) => silu_forward(&request, output)?,
            (CpuOperation::Silu, TrainExecutionV1::Vjp) => silu_vjp(&request, output)?,
            (CpuOperation::Rmsnorm, TrainExecutionV1::Forward) => {
                rmsnorm_forward(&request, output)?;
            }
            (CpuOperation::Rmsnorm, TrainExecutionV1::Vjp) => rmsnorm_vjp(&request, output)?,
            (CpuOperation::Softmax, TrainExecutionV1::Forward) => {
                softmax_forward(&request, output)?;
            }
            (CpuOperation::Softmax, TrainExecutionV1::Vjp) => softmax_vjp(&request, output)?,
            (CpuOperation::CausalMask, TrainExecutionV1::Forward) => {
                causal_mask_forward(&request, output)?;
            }
            (CpuOperation::CausalMask, TrainExecutionV1::Vjp) => {
                causal_mask_vjp(&request, output)?;
            }
            (CpuOperation::Rope, TrainExecutionV1::Forward) => rope_forward(&request, output)?,
            (CpuOperation::Rope, TrainExecutionV1::Vjp) => rope_vjp(&request, output)?,
            (CpuOperation::Attention, TrainExecutionV1::Forward) => {
                attention_forward(&request, output)?;
            }
            (CpuOperation::Attention, TrainExecutionV1::Vjp) => {
                attention_vjp(&request, output)?;
            }
            (CpuOperation::Mse, TrainExecutionV1::Forward) => mse_forward(&request, output)?,
            (CpuOperation::Mse, TrainExecutionV1::Vjp) => mse_vjp(&request, output)?,
            (CpuOperation::SoftmaxCrossEntropy, TrainExecutionV1::Forward) => {
                softmax_xent_forward(&request, output)?;
            }
            (CpuOperation::SoftmaxCrossEntropy, TrainExecutionV1::Vjp) => {
                softmax_xent_vjp(&request, output)?;
            }
            (CpuOperation::Sgd, TrainExecutionV1::Step) => sgd_step(&request, output)?,
            (CpuOperation::AdamW, TrainExecutionV1::Step) => adamw_step(&request, output)?,
            (CpuOperation::CautiousAdamW, TrainExecutionV1::Step) => {
                cautious_adamw_step(&request, output)?;
            }
            (CpuOperation::Int8AdamW, TrainExecutionV1::Step) => {
                int8_adamw_step(&request, output)?;
            }
            (CpuOperation::Muon, TrainExecutionV1::Step) => muon_step(&request, output)?,
            (CpuOperation::Checkpoint, TrainExecutionV1::Checkpoint) => {
                lifecycle_checkpoint(&request, output)?;
            }
            (CpuOperation::Resume, TrainExecutionV1::Resume) => {
                lifecycle_resume(&request, output)?;
            }
            (CpuOperation::Export, TrainExecutionV1::Export) => {
                lifecycle_export(&request, output)?;
            }
            (CpuOperation::Reload, TrainExecutionV1::Reload) => {
                lifecycle_reload(&request, output)?;
            }
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
            physical_device: Some(cpu_physical_device().to_owned()),
            manifest_digest: TrainingOpManifestV1::digest(),
            vector_digest: request.vector_digest,
            operation: request.operation.to_owned(),
            execution: request.execution,
            dtype: if matches!(
                operation.operation,
                CpuOperation::Checkpoint
                    | CpuOperation::Resume
                    | CpuOperation::Export
                    | CpuOperation::Reload
            ) {
                TrainDTypeV1::Bytes
            } else {
                TrainDTypeV1::F32
            },
            limits: CPU_LIMITS,
            input_digest,
            output_digest: train_output_digest_v1(output),
            peak_resident_bytes: resident_bytes(&request, output)?,
            scratch_bytes,
            host_transfers: 0,
            device_resident: true,
        })
    }
}

fn ste_surrogate_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &STE_FORWARD)?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (scale_shape, scale) = input_f32(request, "scale")?;
    let (rows, cols, _, cols_usize) = matrix_attributes(request)?;
    if weight_shape != [rows, cols] || scale_shape != [rows] {
        return Err(shape_error());
    }
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("scale", scale)?;
    require_f32_output(output, "result", weight_shape)?;
    let result = output_f32(output, "result")?;
    for (row, &row_scale) in scale.iter().enumerate() {
        for column in 0..cols_usize {
            let index = row * cols_usize + column;
            result[index] = if row_scale == 0.0 {
                0.0
            } else {
                (weight[index] / row_scale).clamp(-1.0, 1.0)
            };
        }
    }
    Ok(())
}

fn ste_surrogate_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &STE_VJP)?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (scale_shape, scale) = input_f32(request, "scale")?;
    let (gradient_shape, grad_output) = input_f32(request, "grad_output")?;
    let (rows, cols, _, cols_usize) = matrix_attributes(request)?;
    if weight_shape != [rows, cols] || scale_shape != [rows] || gradient_shape != weight_shape {
        return Err(shape_error());
    }
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("scale", scale)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_weight", weight_shape)?;
    require_f32_output(output, "grad_scale", scale_shape)?;
    let grad_weight = output_f32(output, "grad_weight")?;
    grad_weight.fill(0.0);
    for (row, &row_scale) in scale.iter().enumerate() {
        if row_scale == 0.0 {
            continue;
        }
        for column in 0..cols_usize {
            let index = row * cols_usize + column;
            if (weight[index] / row_scale).abs() < 1.0 {
                grad_weight[index] = grad_output[index] / row_scale;
            }
        }
    }
    output_f32(output, "grad_scale")?.fill(0.0);
    Ok(())
}

fn salt_ste_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &SALT_FORWARD)?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    let planes = salt_planes(request)?;
    if rows_usize == 0 {
        return Err(attribute_value("rows", "positive"));
    }
    if cols_usize == 0 {
        return Err(attribute_value("cols", "positive"));
    }
    let scratch_bytes = cols
        .checked_mul(size_of::<f32>() as u64)
        .ok_or_else(|| attribute_value("cols", "scratch_bytes"))?;
    if scratch_bytes > MAX_SALT_SCRATCH_BYTES {
        return Err(attribute_value("cols", "scratch_limit"));
    }
    if weight_shape != [rows, cols] {
        return Err(shape_error());
    }
    reject_nonfinite("weight", weight)?;
    require_f32_output(output, "result", weight_shape)?;
    let result = output_f32(output, "result")?;
    let mut residual = vec![0.0_f32; cols_usize];
    for row in 0..rows_usize {
        let start = row * cols_usize;
        residual.copy_from_slice(&weight[start..start + cols_usize]);
        let reconstruction = &mut result[start..start + cols_usize];
        reconstruction.fill(0.0);
        for _ in 0..planes {
            let mut sum = 0.0_f32;
            for value in &residual {
                sum += value.abs();
            }
            let scale = sum / cols_usize as f32;
            if scale == 0.0 {
                continue;
            }
            for column in 0..cols_usize {
                let contribution = scale * (residual[column] / scale).round().clamp(-1.0, 1.0);
                reconstruction[column] += contribution;
                residual[column] -= contribution;
            }
        }
    }
    Ok(())
}

fn salt_ste_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &SALT_VJP)?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (gradient_shape, grad_output) = input_f32(request, "grad_output")?;
    let (rows, cols, _, cols_usize) = matrix_attributes(request)?;
    salt_planes(request)?;
    if cols_usize == 0 {
        return Err(attribute_value("cols", "positive"));
    }
    if weight_shape != [rows, cols] || gradient_shape != weight_shape {
        return Err(shape_error());
    }
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_weight", weight_shape)?;
    output_f32(output, "grad_weight")?.copy_from_slice(grad_output);
    Ok(())
}

fn lsq_ste_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &LSQ_FORWARD)?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (alpha_shape, alpha) = input_f32(request, "alpha")?;
    let (rows, cols, _, cols_usize) = matrix_attributes(request)?;
    if weight_shape != [rows, cols] || alpha_shape != [rows] {
        return Err(shape_error());
    }
    if cols_usize == 0 {
        return Err(attribute_value("cols", "positive"));
    }
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("alpha", alpha)?;
    require_f32_output(output, "result", weight_shape)?;
    let result = output_f32(output, "result")?;
    result.fill(0.0);
    for (row, &row_alpha) in alpha.iter().enumerate() {
        if row_alpha <= 0.0 {
            continue;
        }
        for column in 0..cols_usize {
            let index = row * cols_usize + column;
            result[index] = (weight[index] / row_alpha).round().clamp(-1.0, 1.0) * row_alpha;
        }
    }
    Ok(())
}

fn lsq_ste_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &LSQ_VJP)?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (alpha_shape, alpha) = input_f32(request, "alpha")?;
    let (gradient_shape, grad_output) = input_f32(request, "grad_output")?;
    let (rows, cols, _, cols_usize) = matrix_attributes(request)?;
    if weight_shape != [rows, cols] || alpha_shape != [rows] || gradient_shape != weight_shape {
        return Err(shape_error());
    }
    if cols_usize == 0 {
        return Err(attribute_value("cols", "positive"));
    }
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("alpha", alpha)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_weight", weight_shape)?;
    require_f32_output(output, "grad_alpha", alpha_shape)?;
    let grad_weight = output_f32(output, "grad_weight")?;
    grad_weight.fill(0.0);
    for (row, &row_alpha) in alpha.iter().enumerate() {
        if row_alpha <= 0.0 {
            continue;
        }
        for column in 0..cols_usize {
            let index = row * cols_usize + column;
            let normalized = weight[index] / row_alpha;
            if normalized.abs() < 1.0 {
                grad_weight[index] = grad_output[index];
            }
        }
    }
    let grad_alpha = output_f32(output, "grad_alpha")?;
    grad_alpha.fill(0.0);
    let gradient_scale = 1.0 / (cols_usize as f32).sqrt();
    for (row, &row_alpha) in alpha.iter().enumerate() {
        if row_alpha <= 0.0 {
            continue;
        }
        let mut gradient = 0.0_f32;
        for column in 0..cols_usize {
            let index = row * cols_usize + column;
            let normalized = weight[index] / row_alpha;
            let local = if normalized.abs() < 1.0 {
                normalized.round() - normalized
            } else {
                normalized.signum()
            };
            gradient += grad_output[index] * local;
        }
        grad_alpha[row] = gradient * gradient_scale;
    }
    Ok(())
}

fn fsq_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &FSQ_FORWARD)?;
    let (shape, x) = input_f32(request, "x")?;
    let config = fsq_attributes(request)?;
    if shape != [config.channels, config.len] {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    require_f32_output(output, "result", shape)?;
    for (index, (result, &x)) in output_f32(output, "result")?.iter_mut().zip(x).enumerate() {
        let level = config.levels[index / config.len_usize];
        let bounded = fsq_bound(x, config.bound);
        *result = match config.estimator {
            FsqEstimator::Stochastic => fsq_quantize_stochastic(bounded, level, config.seed, index),
            FsqEstimator::Hard | FsqEstimator::SoftRound => fsq_quantize(bounded, level),
        };
    }
    Ok(())
}

fn fsq_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &FSQ_VJP)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (gradient_shape, grad_output) = input_f32(request, "grad_output")?;
    let config = fsq_attributes(request)?;
    if x_shape != [config.channels, config.len] || gradient_shape != x_shape {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", x_shape)?;
    for (index, ((grad_x, &x), &gradient)) in output_f32(output, "grad_x")?
        .iter_mut()
        .zip(x)
        .zip(grad_output)
        .enumerate()
    {
        let derivative = fsq_bound_derivative(x, config.bound);
        *grad_x = match config.estimator {
            FsqEstimator::SoftRound => {
                let bounded = fsq_bound(x, config.bound);
                let level = config.levels[index / config.len_usize];
                let position = (bounded + 1.0) * 0.5 * (level - 1) as f32;
                gradient * (1.0 - config.alpha * (2.0 * PI * position).cos()) * derivative
            }
            FsqEstimator::Hard | FsqEstimator::Stochastic => gradient * derivative,
        };
    }
    Ok(())
}

fn conv1d_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &CONV1D_FORWARD)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (scale_shape, scale) = input_f32(request, "scale")?;
    let (cfg, l_out) = conv1d_attributes(request)?;
    let input_shape = [cfg.batch as u64, cfg.c_in as u64, cfg.l_in as u64];
    let weight_shape_expected = [
        cfg.c_out as u64,
        (cfg.c_in / cfg.groups) as u64,
        cfg.k as u64,
    ];
    let scale_shape_expected = [cfg.c_out as u64];
    let result_shape = [cfg.batch as u64, cfg.c_out as u64, l_out as u64];
    if x_shape != input_shape
        || weight_shape != weight_shape_expected
        || scale_shape != scale_shape_expected
    {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("scale", scale)?;
    require_f32_output(output, "result", &result_shape)?;
    let result = conv1d::forward(x, weight, scale, &cfg);
    output_f32(output, "result")?.copy_from_slice(&result);
    Ok(())
}

fn conv1d_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &CONV1D_VJP)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (scale_shape, scale) = input_f32(request, "scale")?;
    let (gradient_shape, grad_output) = input_f32(request, "grad_output")?;
    let (cfg, l_out) = conv1d_attributes(request)?;
    let input_shape = [cfg.batch as u64, cfg.c_in as u64, cfg.l_in as u64];
    let weight_shape_expected = [
        cfg.c_out as u64,
        (cfg.c_in / cfg.groups) as u64,
        cfg.k as u64,
    ];
    let scale_shape_expected = [cfg.c_out as u64];
    let result_shape = [cfg.batch as u64, cfg.c_out as u64, l_out as u64];
    if x_shape != input_shape
        || weight_shape != weight_shape_expected
        || scale_shape != scale_shape_expected
        || gradient_shape != result_shape
    {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("scale", scale)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", x_shape)?;
    require_f32_output(output, "grad_weight", weight_shape)?;
    require_f32_output(output, "grad_scale", scale_shape)?;
    let gradients = conv1d::vjp(x, weight, scale, &cfg, grad_output);
    output_f32(output, "grad_x")?.copy_from_slice(&gradients[0]);
    output_f32(output, "grad_weight")?.copy_from_slice(&gradients[1]);
    output_f32(output, "grad_scale")?.copy_from_slice(&gradients[2]);
    Ok(())
}

fn conv2d_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &CONV2D_FORWARD)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (scale_shape, scale) = input_f32(request, "scale")?;
    let (cfg, height_out, width_out) = conv2d_attributes(request)?;
    let input_shape = [
        cfg.batch as u64,
        cfg.c_in as u64,
        cfg.input_h as u64,
        cfg.input_w as u64,
    ];
    let weight_shape_expected = [
        cfg.c_out as u64,
        (cfg.c_in / cfg.groups) as u64,
        cfg.kernel_h as u64,
        cfg.kernel_w as u64,
    ];
    let scale_shape_expected = [cfg.c_out as u64];
    let result_shape = [
        cfg.batch as u64,
        cfg.c_out as u64,
        height_out as u64,
        width_out as u64,
    ];
    if x_shape != input_shape
        || weight_shape != weight_shape_expected
        || scale_shape != scale_shape_expected
    {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("scale", scale)?;
    require_f32_output(output, "result", &result_shape)?;
    let result = conv2d::try_forward(x, weight, scale, &cfg).map_err(conv2d_reference_error)?;
    output_f32(output, "result")?.copy_from_slice(&result);
    Ok(())
}

fn conv2d_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &CONV2D_VJP)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (scale_shape, scale) = input_f32(request, "scale")?;
    let (gradient_shape, grad_output) = input_f32(request, "grad_output")?;
    let (cfg, height_out, width_out) = conv2d_attributes(request)?;
    let input_shape = [
        cfg.batch as u64,
        cfg.c_in as u64,
        cfg.input_h as u64,
        cfg.input_w as u64,
    ];
    let weight_shape_expected = [
        cfg.c_out as u64,
        (cfg.c_in / cfg.groups) as u64,
        cfg.kernel_h as u64,
        cfg.kernel_w as u64,
    ];
    let scale_shape_expected = [cfg.c_out as u64];
    let result_shape = [
        cfg.batch as u64,
        cfg.c_out as u64,
        height_out as u64,
        width_out as u64,
    ];
    if x_shape != input_shape
        || weight_shape != weight_shape_expected
        || scale_shape != scale_shape_expected
        || gradient_shape != result_shape
    {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("scale", scale)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", x_shape)?;
    require_f32_output(output, "grad_weight", weight_shape)?;
    require_f32_output(output, "grad_scale", scale_shape)?;
    let gradients =
        conv2d::try_vjp(x, weight, scale, &cfg, grad_output).map_err(conv2d_reference_error)?;
    output_f32(output, "grad_x")?.copy_from_slice(&gradients[0]);
    output_f32(output, "grad_weight")?.copy_from_slice(&gradients[1]);
    output_f32(output, "grad_scale")?.copy_from_slice(&gradients[2]);
    Ok(())
}

fn dense_matmul_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &DENSE_MATMUL_FORWARD)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (m, n, k, m_usize, n_usize, k_usize) = matmul_attributes(request)?;
    if x_shape != [m, k] || weight_shape != [n, k] {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("weight", weight)?;
    require_f32_output(output, "result", &[m, n])?;
    let result = output_f32(output, "result")?;
    for row in 0..m_usize {
        for output_column in 0..n_usize {
            let mut accumulator = 0.0_f32;
            for inner in 0..k_usize {
                accumulator += x[row * k_usize + inner] * weight[output_column * k_usize + inner];
            }
            result[row * n_usize + output_column] = accumulator;
        }
    }
    Ok(())
}

fn dense_matmul_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &DENSE_MATMUL_VJP)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    let (m, n, k, m_usize, n_usize, k_usize) = matmul_attributes(request)?;
    if x_shape != [m, k] || weight_shape != [n, k] || grad_shape != [m, n] {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", x_shape)?;
    require_f32_output(output, "grad_weight", weight_shape)?;
    let grad_x = output_f32(output, "grad_x")?;
    grad_x.fill(0.0);
    for row in 0..m_usize {
        for output_column in 0..n_usize {
            let gradient = grad_output[row * n_usize + output_column];
            for inner in 0..k_usize {
                grad_x[row * k_usize + inner] += gradient * weight[output_column * k_usize + inner];
            }
        }
    }
    let grad_weight = output_f32(output, "grad_weight")?;
    grad_weight.fill(0.0);
    for output_column in 0..n_usize {
        for row in 0..m_usize {
            let gradient = grad_output[row * n_usize + output_column];
            for inner in 0..k_usize {
                grad_weight[output_column * k_usize + inner] += gradient * x[row * k_usize + inner];
            }
        }
    }
    Ok(())
}

fn ternary_matmul_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &TERNARY_MATMUL_FORWARD)?;
    let (activation_shape, activation) = input_f32(request, "activation")?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (scale_shape, scale) = input_f32(request, "scale")?;
    let (m, n, k, m_usize, n_usize, k_usize) = matmul_attributes(request)?;
    if activation_shape != [m, k] || weight_shape != [n, k] || scale_shape != [n] {
        return Err(shape_error());
    }
    reject_nonfinite("activation", activation)?;
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("scale", scale)?;
    require_f32_output(output, "result", &[m, n])?;
    let result = output_f32(output, "result")?;
    for row in 0..m_usize {
        for output_column in 0..n_usize {
            let mut accumulator = 0.0_f32;
            for inner in 0..k_usize {
                accumulator +=
                    activation[row * k_usize + inner] * weight[output_column * k_usize + inner];
            }
            result[row * n_usize + output_column] = scale[output_column] * accumulator;
        }
    }
    Ok(())
}

fn ternary_matmul_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &TERNARY_MATMUL_VJP)?;
    let (activation_shape, activation) = input_f32(request, "activation")?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (scale_shape, scale) = input_f32(request, "scale")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    let (m, n, k, m_usize, n_usize, k_usize) = matmul_attributes(request)?;
    if activation_shape != [m, k]
        || weight_shape != [n, k]
        || scale_shape != [n]
        || grad_shape != [m, n]
    {
        return Err(shape_error());
    }
    reject_nonfinite("activation", activation)?;
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("scale", scale)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_activation", activation_shape)?;
    require_f32_output(output, "grad_weight", weight_shape)?;
    require_f32_output(output, "grad_scale", scale_shape)?;

    let grad_activation = output_f32(output, "grad_activation")?;
    grad_activation.fill(0.0);
    for row in 0..m_usize {
        for output_column in 0..n_usize {
            let gradient = grad_output[row * n_usize + output_column];
            for inner in 0..k_usize {
                grad_activation[row * k_usize + inner] +=
                    gradient * scale[output_column] * weight[output_column * k_usize + inner];
            }
        }
    }

    let grad_weight = output_f32(output, "grad_weight")?;
    grad_weight.fill(0.0);
    for output_column in 0..n_usize {
        for row in 0..m_usize {
            let gradient = grad_output[row * n_usize + output_column];
            for inner in 0..k_usize {
                grad_weight[output_column * k_usize + inner] +=
                    gradient * scale[output_column] * activation[row * k_usize + inner];
            }
        }
    }

    let grad_scale = output_f32(output, "grad_scale")?;
    grad_scale.fill(0.0);
    for row in 0..m_usize {
        for output_column in 0..n_usize {
            let mut contraction = 0.0_f32;
            for inner in 0..k_usize {
                contraction +=
                    activation[row * k_usize + inner] * weight[output_column * k_usize + inner];
            }
            grad_scale[output_column] += grad_output[row * n_usize + output_column] * contraction;
        }
    }
    Ok(())
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

fn rmsnorm_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &RMSNORM_FORWARD)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    let eps = attribute_f32(request, "eps")?;
    if cols_usize == 0 {
        return Err(attribute_value("cols", "positive"));
    }
    if eps < 0.0 {
        return Err(attribute_value("eps", "nonnegative"));
    }
    if x_shape != [rows, cols] || weight_shape != [cols] {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("weight", weight)?;
    require_f32_output(output, "result", x_shape)?;
    let result = output_f32(output, "result")?;
    for row in 0..rows_usize {
        let row_start = row * cols_usize;
        let x_row = &x[row_start..row_start + cols_usize];
        let mean_square = x_row.iter().map(|value| value * value).sum::<f32>() / cols_usize as f32;
        let inverse = 1.0 / (mean_square + eps).sqrt();
        for column in 0..cols_usize {
            result[row_start + column] = x_row[column] * inverse * weight[column];
        }
    }
    Ok(())
}

fn rmsnorm_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &RMSNORM_VJP)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (weight_shape, weight) = input_f32(request, "weight")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    let eps = attribute_f32(request, "eps")?;
    if cols_usize == 0 {
        return Err(attribute_value("cols", "positive"));
    }
    if eps < 0.0 {
        return Err(attribute_value("eps", "nonnegative"));
    }
    if x_shape != [rows, cols] || weight_shape != [cols] || grad_shape != x_shape {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("weight", weight)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", x_shape)?;
    require_f32_output(output, "grad_weight", weight_shape)?;
    let grad_x = output_f32(output, "grad_x")?;
    for row in 0..rows_usize {
        let row_start = row * cols_usize;
        let x_row = &x[row_start..row_start + cols_usize];
        let grad_row = &grad_output[row_start..row_start + cols_usize];
        let mean_square = x_row.iter().map(|value| value * value).sum::<f32>() / cols_usize as f32;
        let inverse = 1.0 / (mean_square + eps).sqrt();
        let mut contraction = 0.0_f32;
        for column in 0..cols_usize {
            contraction += grad_row[column] * weight[column] * x_row[column];
        }
        let correction = inverse * inverse * inverse * contraction / cols_usize as f32;
        for column in 0..cols_usize {
            grad_x[row_start + column] =
                inverse * grad_row[column] * weight[column] - correction * x_row[column];
        }
    }
    let grad_weight = output_f32(output, "grad_weight")?;
    grad_weight.fill(0.0);
    for row in 0..rows_usize {
        let row_start = row * cols_usize;
        let x_row = &x[row_start..row_start + cols_usize];
        let mean_square = x_row.iter().map(|value| value * value).sum::<f32>() / cols_usize as f32;
        let inverse = 1.0 / (mean_square + eps).sqrt();
        for column in 0..cols_usize {
            grad_weight[column] += grad_output[row_start + column] * x_row[column] * inverse;
        }
    }
    Ok(())
}

fn softmax_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &MATRIX_FORWARD)?;
    let (shape, x) = input_f32(request, "x")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    if cols_usize == 0 {
        return Err(attribute_value("cols", "positive"));
    }
    if shape != [rows, cols] {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    require_f32_output(output, "result", shape)?;
    softmax_rows_into(x, rows_usize, cols_usize, output_f32(output, "result")?);
    Ok(())
}

fn softmax_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &SOFTMAX_VJP)?;
    let (x_shape, x) = input_f32(request, "x")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    if cols_usize == 0 {
        return Err(attribute_value("cols", "positive"));
    }
    if x_shape != [rows, cols] || grad_shape != x_shape {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", x_shape)?;
    let grad_x = output_f32(output, "grad_x")?;
    softmax_rows_into(x, rows_usize, cols_usize, grad_x);
    for row in 0..rows_usize {
        let row_start = row * cols_usize;
        let probabilities = &mut grad_x[row_start..row_start + cols_usize];
        let grad_row = &grad_output[row_start..row_start + cols_usize];
        let dot = (0..cols_usize)
            .map(|column| probabilities[column] * grad_row[column])
            .sum::<f32>();
        for column in 0..cols_usize {
            probabilities[column] *= grad_row[column] - dot;
        }
    }
    Ok(())
}

fn causal_mask_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &MATRIX_FORWARD)?;
    let (shape, x) = input_f32(request, "x")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    if shape != [rows, cols] {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    require_f32_output(output, "result", shape)?;
    let result = output_f32(output, "result")?;
    for row in 0..rows_usize {
        for column in 0..cols_usize {
            result[row * cols_usize + column] = if column <= row {
                x[row * cols_usize + column]
            } else {
                -1.0e30_f32
            };
        }
    }
    Ok(())
}

fn causal_mask_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &CAUSAL_MASK_VJP)?;
    let (shape, grad_output) = input_f32(request, "grad_output")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    if shape != [rows, cols] {
        return Err(shape_error());
    }
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", shape)?;
    let grad_x = output_f32(output, "grad_x")?;
    grad_x.fill(0.0);
    for row in 0..rows_usize {
        for column in 0..cols_usize {
            if column <= row {
                grad_x[row * cols_usize + column] = grad_output[row * cols_usize + column];
            }
        }
    }
    Ok(())
}

fn rope_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &ROPE_FORWARD)?;
    let (shape, x) = input_f32(request, "x")?;
    let (positions, n_token, n_head, head_dim, n_head_usize, head_dim_usize, theta) =
        rope_attributes(request)?;
    if shape != [n_token, n_head, head_dim] {
        return Err(shape_error());
    }
    reject_nonfinite("x", x)?;
    require_f32_output(output, "result", shape)?;
    let result = output_f32(output, "result")?;
    result.copy_from_slice(x);
    apply_rope_in_place(
        result,
        positions,
        n_head_usize,
        head_dim_usize,
        theta,
        false,
    );
    Ok(())
}

fn rope_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &ROPE_VJP)?;
    let (shape, grad_output) = input_f32(request, "grad_output")?;
    let (positions, n_token, n_head, head_dim, n_head_usize, head_dim_usize, theta) =
        rope_attributes(request)?;
    if shape != [n_token, n_head, head_dim] {
        return Err(shape_error());
    }
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_x", shape)?;
    let grad_x = output_f32(output, "grad_x")?;
    grad_x.copy_from_slice(grad_output);
    apply_rope_in_place(grad_x, positions, n_head_usize, head_dim_usize, theta, true);
    Ok(())
}

fn attention_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &ATTENTION_FORWARD)?;
    let (q_shape, q) = input_f32(request, "q")?;
    let (k_shape, k) = input_f32(request, "k")?;
    let (v_shape, v) = input_f32(request, "v")?;
    let cfg = attention_attributes(request)?;
    let query_shape = [cfg.seq as u64, cfg.n_head as u64, cfg.head_dim as u64];
    let kv_shape = [cfg.seq as u64, cfg.n_kv_head as u64, cfg.head_dim as u64];
    if q_shape != query_shape || k_shape != kv_shape || v_shape != kv_shape {
        return Err(shape_error());
    }
    reject_nonfinite("q", q)?;
    reject_nonfinite("k", k)?;
    reject_nonfinite("v", v)?;
    require_f32_output(output, "result", &query_shape)?;
    let reference = attention::forward(q, k, v, cfg);
    output_f32(output, "result")?.copy_from_slice(&reference);
    Ok(())
}

fn attention_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &ATTENTION_VJP)?;
    let (q_shape, q) = input_f32(request, "q")?;
    let (k_shape, k) = input_f32(request, "k")?;
    let (v_shape, v) = input_f32(request, "v")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    let cfg = attention_attributes(request)?;
    let query_shape = [cfg.seq as u64, cfg.n_head as u64, cfg.head_dim as u64];
    let kv_shape = [cfg.seq as u64, cfg.n_kv_head as u64, cfg.head_dim as u64];
    if q_shape != query_shape
        || k_shape != kv_shape
        || v_shape != kv_shape
        || grad_shape != query_shape
    {
        return Err(shape_error());
    }
    reject_nonfinite("q", q)?;
    reject_nonfinite("k", k)?;
    reject_nonfinite("v", v)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_q", &query_shape)?;
    require_f32_output(output, "grad_k", &kv_shape)?;
    require_f32_output(output, "grad_v", &kv_shape)?;
    let gradients = attention::vjp(q, k, v, cfg, grad_output);
    output_f32(output, "grad_q")?.copy_from_slice(&gradients[0]);
    output_f32(output, "grad_k")?.copy_from_slice(&gradients[1]);
    output_f32(output, "grad_v")?.copy_from_slice(&gradients[2]);
    Ok(())
}

fn apply_rope_in_place(
    buffer: &mut [f32],
    positions: &[u64],
    n_head: usize,
    head_dim: usize,
    theta: f32,
    inverse: bool,
) {
    let half = head_dim / 2;
    let theta = f64::from(theta);
    let inverse_head_dim = 1.0 / head_dim as f64;
    let sine_sign = if inverse { -1.0_f32 } else { 1.0_f32 };
    for (token, &position) in positions.iter().enumerate() {
        let token_base = token * n_head * head_dim;
        for head in 0..n_head {
            let head_base = token_base + head * head_dim;
            for lane in 0..half {
                let frequency = theta.powf(-2.0 * lane as f64 * inverse_head_dim);
                let (sine, cosine) = (position as f64 * frequency).sin_cos();
                let sine = sine_sign * sine as f32;
                let cosine = cosine as f32;
                let left = buffer[head_base + lane];
                let right = buffer[head_base + lane + half];
                buffer[head_base + lane] = left * cosine - right * sine;
                buffer[head_base + lane + half] = right * cosine + left * sine;
            }
        }
    }
}

fn softmax_xent_forward(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &SOFTMAX_XENT_FORWARD)?;
    let (logits_shape, logits) = input_f32(request, "logits")?;
    let (target_shape, target) = input_f32(request, "target")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    if rows_usize == 0 {
        return Err(attribute_value("rows", "positive"));
    }
    if cols_usize == 0 {
        return Err(attribute_value("cols", "positive"));
    }
    if logits_shape != [rows, cols] || target_shape != logits_shape {
        return Err(shape_error());
    }
    reject_nonfinite("logits", logits)?;
    reject_nonfinite("target", target)?;
    require_f32_output(output, "result", &[])?;
    let mut loss = 0.0_f32;
    for row in 0..rows_usize {
        let row_start = row * cols_usize;
        let logits_row = &logits[row_start..row_start + cols_usize];
        let maximum = logits_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = logits_row
            .iter()
            .map(|value| (*value - maximum).exp())
            .sum::<f32>();
        for column in 0..cols_usize {
            let probability = (logits_row[column] - maximum).exp() / sum;
            loss -= target[row_start + column] * probability.max(f32::MIN_POSITIVE).ln();
        }
    }
    output_f32(output, "result")?[0] = loss / rows_usize as f32;
    Ok(())
}

fn softmax_xent_vjp(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &SOFTMAX_XENT_VJP)?;
    let (logits_shape, logits) = input_f32(request, "logits")?;
    let (target_shape, target) = input_f32(request, "target")?;
    let (grad_shape, grad_output) = input_f32(request, "grad_output")?;
    let (rows, cols, rows_usize, cols_usize) = matrix_attributes(request)?;
    if rows_usize == 0 {
        return Err(attribute_value("rows", "positive"));
    }
    if cols_usize == 0 {
        return Err(attribute_value("cols", "positive"));
    }
    if logits_shape != [rows, cols] || target_shape != logits_shape || !grad_shape.is_empty() {
        return Err(shape_error());
    }
    reject_nonfinite("logits", logits)?;
    reject_nonfinite("target", target)?;
    reject_nonfinite("grad_output", grad_output)?;
    require_f32_output(output, "grad_logits", logits_shape)?;
    let grad_logits = output_f32(output, "grad_logits")?;
    softmax_rows_into(logits, rows_usize, cols_usize, grad_logits);
    let upstream = grad_output[0] / rows_usize as f32;
    for row in 0..rows_usize {
        let row_start = row * cols_usize;
        let target_sum = target[row_start..row_start + cols_usize]
            .iter()
            .sum::<f32>();
        for column in 0..cols_usize {
            let index = row_start + column;
            grad_logits[index] = upstream * (grad_logits[index] * target_sum - target[index]);
        }
    }
    Ok(())
}

fn softmax_rows_into(x: &[f32], rows: usize, cols: usize, output: &mut [f32]) {
    for row in 0..rows {
        let row_start = row * cols;
        let x_row = &x[row_start..row_start + cols];
        let maximum = x_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for column in 0..cols {
            let exponential = (x_row[column] - maximum).exp();
            output[row_start + column] = exponential;
            sum += exponential;
        }
        for column in 0..cols {
            output[row_start + column] /= sum;
        }
    }
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

fn adamw_step(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    let (step, optimizer) = adam_optimizer_attributes(request)?;
    adam_family_step(request, output, step, optimizer, false)
}

fn cautious_adamw_step(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    let (step, optimizer) = adam_optimizer_attributes(request)?;
    adam_family_step(request, output, step, optimizer, true)
}

fn adam_family_step(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
    step: u64,
    optimizer: AdamW,
    cautious: bool,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &ADAM_STEP)?;
    let (parameter_shape, parameter) = input_f32(request, "parameter")?;
    let (gradient_shape, gradient) = input_f32(request, "gradient")?;
    let (moment1_shape, moment1) = input_f32(request, "moment1")?;
    let (moment2_shape, moment2) = input_f32(request, "moment2")?;
    if parameter.is_empty()
        || gradient_shape != parameter_shape
        || moment1_shape != parameter_shape
        || moment2_shape != parameter_shape
    {
        return Err(shape_error());
    }
    reject_nonfinite("parameter", parameter)?;
    reject_nonfinite("gradient", gradient)?;
    reject_nonfinite("moment1", moment1)?;
    reject_nonfinite("moment2", moment2)?;
    require_f32_output(output, "parameter", parameter_shape)?;
    require_f32_output(output, "moment1", parameter_shape)?;
    require_f32_output(output, "moment2", parameter_shape)?;

    let mut state = AdamState {
        m: moment1.to_vec(),
        v: moment2.to_vec(),
    };
    let updated = output_f32(output, "parameter")?;
    updated.copy_from_slice(parameter);
    if cautious {
        CautiousAdamW(optimizer).step(step, updated, gradient, &mut state);
    } else {
        optimizer.step(step, updated, gradient, &mut state);
    }
    output_f32(output, "moment1")?.copy_from_slice(&state.m);
    output_f32(output, "moment2")?.copy_from_slice(&state.v);
    Ok(())
}

fn int8_adamw_step(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &INT8_ADAM_STEP)?;
    let (step, optimizer) = adam_optimizer_attributes(request)?;
    let (parameter_shape, parameter) = input_f32(request, "parameter")?;
    let (gradient_shape, gradient) = input_f32(request, "gradient")?;
    let (moment1_shape, moment1_q8) = input_bytes(request, "moment1_q8")?;
    let (moment2_shape, moment2_q8) = input_bytes(request, "moment2_q8")?;
    let (moment1_scale_shape, moment1_scale) = input_f32(request, "moment1_scale")?;
    let (moment2_scale_shape, moment2_scale) = input_f32(request, "moment2_scale")?;
    let len = parameter.len();
    let len_u64 = u64::try_from(len).map_err(|_| attribute_value("parameter", "u64"))?;
    let blocks = len.div_ceil(INT8_ADAM_BLOCK);
    let blocks_u64 = u64::try_from(blocks).map_err(|_| attribute_value("parameter", "blocks"))?;
    if len == 0
        || gradient_shape != parameter_shape
        || moment1_shape != [len_u64]
        || moment2_shape != [len_u64]
        || moment1_scale_shape != [blocks_u64]
        || moment2_scale_shape != [blocks_u64]
    {
        return Err(shape_error());
    }
    reject_nonfinite("parameter", parameter)?;
    reject_nonfinite("gradient", gradient)?;
    reject_nonfinite("moment1_scale", moment1_scale)?;
    reject_nonfinite("moment2_scale", moment2_scale)?;
    if moment1_scale.iter().any(|&value| value < 0.0) {
        return Err(attribute_value("moment1_scale", "nonnegative"));
    }
    if moment2_scale.iter().any(|&value| value < 0.0) {
        return Err(attribute_value("moment2_scale", "nonnegative"));
    }
    require_f32_output(output, "parameter", parameter_shape)?;
    require_bytes_output(output, "moment1_q8", &[len_u64])?;
    require_bytes_output(output, "moment2_q8", &[len_u64])?;
    require_f32_output(output, "moment1_scale", &[blocks_u64])?;
    require_f32_output(output, "moment2_scale", &[blocks_u64])?;

    let mut state = Int8AdamState {
        m_q: moment1_q8.iter().map(|&value| value as i8).collect(),
        v_q: moment2_q8.to_vec(),
        m_scale: moment1_scale.to_vec(),
        v_scale: moment2_scale.to_vec(),
        len,
    };
    let updated = output_f32(output, "parameter")?;
    updated.copy_from_slice(parameter);
    Int8AdamW(optimizer).step(step, updated, gradient, &mut state);
    for (output, &value) in output_bytes(output, "moment1_q8")?
        .iter_mut()
        .zip(&state.m_q)
    {
        *output = value as u8;
    }
    output_bytes(output, "moment2_q8")?.copy_from_slice(&state.v_q);
    output_f32(output, "moment1_scale")?.copy_from_slice(&state.m_scale);
    output_f32(output, "moment2_scale")?.copy_from_slice(&state.v_scale);
    Ok(())
}

fn muon_step(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &MUON_STEP)?;
    let (step, optimizer) = muon_optimizer_attributes(request)?;
    let (parameter_shape, parameter) = input_f32(request, "parameter")?;
    let (gradient_shape, gradient) = input_f32(request, "gradient")?;
    let (momentum_shape, momentum) = input_f32(request, "momentum")?;
    let expected_shape = [optimizer.rows as u64, optimizer.cols as u64];
    if parameter_shape != expected_shape
        || gradient_shape != expected_shape
        || momentum_shape != expected_shape
    {
        return Err(shape_error());
    }
    reject_nonfinite("parameter", parameter)?;
    reject_nonfinite("gradient", gradient)?;
    reject_nonfinite("momentum", momentum)?;
    require_f32_output(output, "parameter", &expected_shape)?;
    require_f32_output(output, "momentum", &expected_shape)?;
    let mut state = MuonState {
        momentum: momentum.to_vec(),
    };
    let updated = output_f32(output, "parameter")?;
    updated.copy_from_slice(parameter);
    optimizer.step(step, updated, gradient, &mut state);
    output_f32(output, "momentum")?.copy_from_slice(&state.momentum);
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CheckpointOptimizer {
    Sgd,
    AdamW,
    CautiousAdamW,
    Int8AdamW,
    Muon,
}

impl CheckpointOptimizer {
    fn parse(request: &TrainRequestV1<'_>) -> Result<Self, TrainBackendError> {
        match attribute_text(request, "optimizer")? {
            "sgd" => Ok(Self::Sgd),
            "adamw" => Ok(Self::AdamW),
            "cautious_adamw" => Ok(Self::CautiousAdamW),
            "int8_adamw" => Ok(Self::Int8AdamW),
            "muon" => Ok(Self::Muon),
            _ => Err(attribute_value("optimizer", "known")),
        }
    }

    const fn planes(self) -> &'static [&'static str] {
        match self {
            Self::Sgd => &["parameter"],
            Self::AdamW | Self::CautiousAdamW => &["parameter", "moment1", "moment2"],
            Self::Int8AdamW => &[
                "parameter",
                "moment1_q8",
                "moment2_q8",
                "moment1_scale",
                "moment2_scale",
            ],
            Self::Muon => &["parameter", "momentum"],
        }
    }
}

fn lifecycle_attributes<'a>(
    request: &'a TrainRequestV1<'_>,
    checkpoint: bool,
) -> Result<(CheckpointOptimizer, u64, &'a [u64]), TrainBackendError> {
    let expected = if checkpoint {
        &["optimizer", "step", "leaf_lens"][..]
    } else {
        &["optimizer", "leaf_lens"][..]
    };
    if !same_names(
        request.attributes.iter().map(|attribute| attribute.name),
        expected,
    ) {
        return Err(role_error("attribute"));
    }
    let optimizer = CheckpointOptimizer::parse(request)?;
    let step = if checkpoint {
        attribute_u64(request, "step")?
    } else {
        0
    };
    let leaf_lens = attribute_u64_list(request, "leaf_lens")?;
    if leaf_lens.is_empty() {
        return Err(attribute_value("leaf_lens", "nonempty"));
    }
    u32::try_from(leaf_lens.len()).map_err(|_| attribute_value("leaf_lens", "u32_count"))?;
    for &len in leaf_lens {
        if len == 0 {
            return Err(attribute_value("leaf_lens", "positive"));
        }
        u32::try_from(len).map_err(|_| attribute_value("leaf_lens", "u32"))?;
    }
    Ok((optimizer, step, leaf_lens))
}

fn indexed_plane_name(plane: &str, index: usize) -> String {
    format!("{plane}.{index}")
}

fn checkpoint_roles_match<'a>(
    actual: impl Iterator<Item = &'a str>,
    optimizer: CheckpointOptimizer,
    leaves: usize,
) -> Result<bool, TrainBackendError> {
    let expected_count = optimizer
        .planes()
        .len()
        .checked_mul(leaves)
        .ok_or_else(|| attribute_value("leaf_lens", "role_count"))?;
    let mut actual = actual;
    if actual.size_hint().0 > expected_count {
        return Ok(false);
    }
    for index in 0..leaves {
        for &plane in optimizer.planes() {
            let expected = indexed_plane_name(plane, index);
            if actual.next() != Some(expected.as_str()) {
                return Ok(false);
            }
        }
    }
    Ok(actual.next().is_none())
}

fn require_checkpoint_contract(
    request: &TrainRequestV1<'_>,
    output: &TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    let (optimizer, _, leaf_lens) = lifecycle_attributes(request, true)?;
    if !checkpoint_roles_match(
        request.inputs.iter().map(|buffer| buffer.name),
        optimizer,
        leaf_lens.len(),
    )? {
        return Err(role_error("input"));
    }
    if !same_names(
        output.buffers.iter().map(|buffer| buffer.name),
        &["checkpoint"],
    ) {
        return Err(role_error("output"));
    }
    validate_checkpoint_input_shapes(request, optimizer, leaf_lens)?;
    let encoded_bytes = checkpoint_encoded_bytes(optimizer, leaf_lens)?;
    require_bytes_output(output, "checkpoint", &[encoded_bytes])
}

fn require_resume_contract(
    request: &TrainRequestV1<'_>,
    output: &TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    let (optimizer, _, leaf_lens) = lifecycle_attributes(request, false)?;
    if !same_names(
        request.inputs.iter().map(|buffer| buffer.name),
        &["checkpoint"],
    ) {
        return Err(role_error("input"));
    }
    input_bytes(request, "checkpoint")?;
    if output.buffers.first().map(|buffer| buffer.name) != Some("step")
        || !checkpoint_roles_match(
            output.buffers.iter().skip(1).map(|buffer| buffer.name),
            optimizer,
            leaf_lens.len(),
        )?
    {
        return Err(role_error("output"));
    }
    require_bytes_output(output, "step", &[8])?;
    validate_checkpoint_output_shapes(output, optimizer, leaf_lens)
}

fn validate_checkpoint_input_shapes(
    request: &TrainRequestV1<'_>,
    optimizer: CheckpointOptimizer,
    leaf_lens: &[u64],
) -> Result<(), TrainBackendError> {
    for (index, &len) in leaf_lens.iter().enumerate() {
        let blocks = len.div_ceil(INT8_ADAM_BLOCK as u64);
        for &plane in optimizer.planes() {
            let name = indexed_plane_name(plane, index);
            let expected = if plane.ends_with("_scale") {
                blocks
            } else {
                len
            };
            if plane.ends_with("_q8") {
                let (shape, _) = input_bytes(request, &name)?;
                if shape != [expected] {
                    return Err(shape_error());
                }
            } else {
                let (shape, values) = input_f32(request, &name)?;
                if shape != [expected] {
                    return Err(shape_error());
                }
                reject_nonfinite(&name, values)?;
                if (plane == "moment2" || plane.ends_with("_scale"))
                    && values.iter().any(|&value| value < 0.0)
                {
                    return Err(attribute_value(plane, "nonnegative"));
                }
            }
        }
    }
    Ok(())
}

fn validate_checkpoint_output_shapes(
    output: &TrainOutputV1<'_>,
    optimizer: CheckpointOptimizer,
    leaf_lens: &[u64],
) -> Result<(), TrainBackendError> {
    for (index, &len) in leaf_lens.iter().enumerate() {
        let blocks = len.div_ceil(INT8_ADAM_BLOCK as u64);
        for &plane in optimizer.planes() {
            let name = indexed_plane_name(plane, index);
            let shape = [if plane.ends_with("_scale") {
                blocks
            } else {
                len
            }];
            if plane.ends_with("_q8") {
                require_bytes_output(output, &name, &shape)?;
            } else {
                require_f32_output(output, &name, &shape)?;
            }
        }
    }
    Ok(())
}

fn checkpoint_input_f32(
    request: &TrainRequestV1<'_>,
    plane: &str,
    index: usize,
) -> Result<Vec<f32>, TrainBackendError> {
    Ok(input_f32(request, &indexed_plane_name(plane, index))?
        .1
        .to_vec())
}

fn checkpoint_input_bytes(
    request: &TrainRequestV1<'_>,
    plane: &str,
    index: usize,
) -> Result<Vec<u8>, TrainBackendError> {
    Ok(input_bytes(request, &indexed_plane_name(plane, index))?
        .1
        .to_vec())
}

fn lifecycle_checkpoint(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_checkpoint_contract(request, output)?;
    let (optimizer, step, leaf_lens) = lifecycle_attributes(request, true)?;
    let bytes = match optimizer {
        CheckpointOptimizer::Sgd => {
            let leaves = (0..leaf_lens.len())
                .map(|index| {
                    Ok(LeafCheckpoint {
                        param: checkpoint_input_f32(request, "parameter", index)?,
                        state: SgdState,
                    })
                })
                .collect::<Result<Vec<_>, TrainBackendError>>()?;
            write_checkpoint(&Sgd::new(0.0), &Checkpoint { step, leaves })
        }
        CheckpointOptimizer::AdamW | CheckpointOptimizer::CautiousAdamW => {
            let leaves = (0..leaf_lens.len())
                .map(|index| {
                    Ok(LeafCheckpoint {
                        param: checkpoint_input_f32(request, "parameter", index)?,
                        state: AdamState {
                            m: checkpoint_input_f32(request, "moment1", index)?,
                            v: checkpoint_input_f32(request, "moment2", index)?,
                        },
                    })
                })
                .collect::<Result<Vec<_>, TrainBackendError>>()?;
            write_checkpoint(&AdamW::new(0.0), &Checkpoint { step, leaves })
        }
        CheckpointOptimizer::Int8AdamW => {
            let leaves = (0..leaf_lens.len())
                .map(|index| {
                    let len = leaf_lens[index] as usize;
                    Ok(LeafCheckpoint {
                        param: checkpoint_input_f32(request, "parameter", index)?,
                        state: Int8AdamState {
                            m_q: checkpoint_input_bytes(request, "moment1_q8", index)?
                                .into_iter()
                                .map(|value| value as i8)
                                .collect(),
                            v_q: checkpoint_input_bytes(request, "moment2_q8", index)?,
                            m_scale: checkpoint_input_f32(request, "moment1_scale", index)?,
                            v_scale: checkpoint_input_f32(request, "moment2_scale", index)?,
                            len,
                        },
                    })
                })
                .collect::<Result<Vec<_>, TrainBackendError>>()?;
            write_checkpoint(&Int8AdamW::new(0.0), &Checkpoint { step, leaves })
        }
        CheckpointOptimizer::Muon => {
            let leaves = (0..leaf_lens.len())
                .map(|index| {
                    Ok(LeafCheckpoint {
                        param: checkpoint_input_f32(request, "parameter", index)?,
                        state: MuonState {
                            momentum: checkpoint_input_f32(request, "momentum", index)?,
                        },
                    })
                })
                .collect::<Result<Vec<_>, TrainBackendError>>()?;
            write_checkpoint(&Muon::new(0.0, 1, 1), &Checkpoint { step, leaves })
        }
    };
    require_bytes_output(output, "checkpoint", &[bytes.len() as u64])?;
    output_bytes(output, "checkpoint")?.copy_from_slice(&bytes);
    Ok(())
}

fn checkpoint_error(error: CheckpointError) -> TrainBackendError {
    let constraint = match error {
        CheckpointError::BadMagic => "bad_magic",
        CheckpointError::UnsupportedVersion(_) => "unsupported_version",
        CheckpointError::Truncated { .. } => "truncated",
        CheckpointError::TrailingBytes(_) => "trailing_bytes",
    };
    attribute_value("checkpoint", constraint)
}

fn lifecycle_resume(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_resume_contract(request, output)?;
    let (optimizer, _, leaf_lens) = lifecycle_attributes(request, false)?;
    let checkpoint = input_bytes(request, "checkpoint")?.1;
    match optimizer {
        CheckpointOptimizer::Sgd => {
            let parsed = read_checkpoint(&Sgd::new(0.0), checkpoint).map_err(checkpoint_error)?;
            validate_resumed_parameters(&parsed)?;
            restore_checkpoint(output, leaf_lens, parsed, |_, _, _| Ok(()))?;
        }
        CheckpointOptimizer::AdamW | CheckpointOptimizer::CautiousAdamW => {
            let parsed = read_checkpoint(&AdamW::new(0.0), checkpoint).map_err(checkpoint_error)?;
            validate_resumed_parameters(&parsed)?;
            for leaf in &parsed.leaves {
                reject_nonfinite("moment1", &leaf.state.m)?;
                reject_nonfinite("moment2", &leaf.state.v)?;
                if leaf.state.v.iter().any(|&value| value < 0.0) {
                    return Err(attribute_value("moment2", "nonnegative"));
                }
            }
            restore_checkpoint(output, leaf_lens, parsed, |output, index, state| {
                checkpoint_output_f32(output, "moment1", index)?.copy_from_slice(&state.m);
                checkpoint_output_f32(output, "moment2", index)?.copy_from_slice(&state.v);
                Ok(())
            })?;
        }
        CheckpointOptimizer::Int8AdamW => {
            let parsed =
                read_checkpoint(&Int8AdamW::new(0.0), checkpoint).map_err(checkpoint_error)?;
            validate_resumed_parameters(&parsed)?;
            for leaf in &parsed.leaves {
                reject_nonfinite("moment1_scale", &leaf.state.m_scale)?;
                reject_nonfinite("moment2_scale", &leaf.state.v_scale)?;
                if leaf.state.m_scale.iter().any(|&value| value < 0.0) {
                    return Err(attribute_value("moment1_scale", "nonnegative"));
                }
                if leaf.state.v_scale.iter().any(|&value| value < 0.0) {
                    return Err(attribute_value("moment2_scale", "nonnegative"));
                }
            }
            restore_checkpoint(output, leaf_lens, parsed, |output, index, state| {
                for (out, &value) in checkpoint_output_bytes(output, "moment1_q8", index)?
                    .iter_mut()
                    .zip(&state.m_q)
                {
                    *out = value as u8;
                }
                checkpoint_output_bytes(output, "moment2_q8", index)?.copy_from_slice(&state.v_q);
                checkpoint_output_f32(output, "moment1_scale", index)?
                    .copy_from_slice(&state.m_scale);
                checkpoint_output_f32(output, "moment2_scale", index)?
                    .copy_from_slice(&state.v_scale);
                Ok(())
            })?;
        }
        CheckpointOptimizer::Muon => {
            let parsed =
                read_checkpoint(&Muon::new(0.0, 1, 1), checkpoint).map_err(checkpoint_error)?;
            validate_resumed_parameters(&parsed)?;
            for leaf in &parsed.leaves {
                reject_nonfinite("momentum", &leaf.state.momentum)?;
            }
            restore_checkpoint(output, leaf_lens, parsed, |output, index, state| {
                checkpoint_output_f32(output, "momentum", index)?.copy_from_slice(&state.momentum);
                Ok(())
            })?;
        }
    }
    Ok(())
}

fn validate_resumed_parameters<S>(checkpoint: &Checkpoint<S>) -> Result<(), TrainBackendError> {
    for leaf in &checkpoint.leaves {
        reject_nonfinite("parameter", &leaf.param)?;
    }
    Ok(())
}

fn restore_checkpoint<S>(
    output: &mut TrainOutputV1<'_>,
    leaf_lens: &[u64],
    checkpoint: Checkpoint<S>,
    mut restore_state: impl FnMut(&mut TrainOutputV1<'_>, usize, &S) -> Result<(), TrainBackendError>,
) -> Result<(), TrainBackendError> {
    if checkpoint.leaves.len() != leaf_lens.len()
        || checkpoint
            .leaves
            .iter()
            .zip(leaf_lens)
            .any(|(leaf, &len)| leaf.param.len() as u64 != len)
    {
        return Err(attribute_value("leaf_lens", "checkpoint_match"));
    }
    output_bytes(output, "step")?.copy_from_slice(&checkpoint.step.to_le_bytes());
    for (index, leaf) in checkpoint.leaves.iter().enumerate() {
        checkpoint_output_f32(output, "parameter", index)?.copy_from_slice(&leaf.param);
        restore_state(output, index, &leaf.state)?;
    }
    Ok(())
}

fn checkpoint_output_f32<'a>(
    output: &'a mut TrainOutputV1<'_>,
    plane: &str,
    index: usize,
) -> Result<&'a mut [f32], TrainBackendError> {
    output_f32(output, &indexed_plane_name(plane, index))
}

fn checkpoint_output_bytes<'a>(
    output: &'a mut TrainOutputV1<'_>,
    plane: &str,
    index: usize,
) -> Result<&'a mut [u8], TrainBackendError> {
    output_bytes(output, &indexed_plane_name(plane, index))
}

fn validate_salt_v2_package<'a>(
    request: &TrainRequestV1<'a>,
    input_name: &str,
) -> Result<&'a [u8], TrainBackendError> {
    if attribute_text(request, "format")? != "salt_v2_package_v1" {
        return Err(attribute_value("format", "salt_v2_package_v1"));
    }
    let (shape, bytes) = input_bytes(request, input_name)?;
    if shape != [bytes.len() as u64] || bytes.is_empty() {
        return Err(shape_error());
    }
    SaltV2PackageReader::new_strict(IoCursor::new(bytes))
        .map_err(|_| attribute_value(input_name, "salt_v2_package"))?;
    Ok(bytes)
}

fn lifecycle_export(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &EXPORT)?;
    let package = validate_salt_v2_package(request, "package")?;
    require_bytes_output(output, "artifact", &[package.len() as u64])?;
    output_bytes(output, "artifact")?.copy_from_slice(package);
    Ok(())
}

fn lifecycle_reload(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    require_contract(request, output, &RELOAD)?;
    let artifact = validate_salt_v2_package(request, "artifact")?;
    require_bytes_output(output, "package", &[artifact.len() as u64])?;
    output_bytes(output, "package")?.copy_from_slice(artifact);
    Ok(())
}

/// Execute one canonical lifecycle operation shared by accelerator adapters.
///
/// Lifecycle payloads are host-visible control-plane bytes by contract. This
/// helper performs serialization and strict artifact validation only; it never
/// invokes the CPU tensor backend or executes graph/optimizer work.
pub fn execute_lifecycle_control_plane(
    request: &TrainRequestV1<'_>,
    output: &mut TrainOutputV1<'_>,
) -> Result<(), TrainBackendError> {
    match (request.operation, request.execution) {
        ("lifecycle.checkpoint", TrainExecutionV1::Checkpoint) => {
            lifecycle_checkpoint(request, output)
        }
        ("lifecycle.resume", TrainExecutionV1::Resume) => lifecycle_resume(request, output),
        ("lifecycle.export", TrainExecutionV1::Export) => lifecycle_export(request, output),
        ("lifecycle.reload", TrainExecutionV1::Reload) => lifecycle_reload(request, output),
        _ => Err(TrainBackendError::UnsupportedOperation(
            request.operation.to_owned(),
        )),
    }
}

/// Return the canonical peak scratch ledger for a lifecycle request.
pub fn lifecycle_control_plane_scratch_bytes(
    request: &TrainRequestV1<'_>,
) -> Result<u64, TrainBackendError> {
    let operation = match request.operation {
        "lifecycle.checkpoint" => CpuOperation::Checkpoint,
        "lifecycle.resume" => CpuOperation::Resume,
        "lifecycle.export" => CpuOperation::Export,
        "lifecycle.reload" => CpuOperation::Reload,
        operation => {
            return Err(TrainBackendError::UnsupportedOperation(
                operation.to_owned(),
            ));
        }
    };
    operation_scratch_bytes(operation, request.execution, request)
}

#[derive(Clone, Copy)]
enum FsqBoundKind {
    Clamp,
    Tanh,
}

#[derive(Clone, Copy)]
enum FsqEstimator {
    Hard,
    SoftRound,
    Stochastic,
}

struct FsqAttributes<'a> {
    channels: u64,
    len: u64,
    len_usize: usize,
    levels: &'a [u32],
    bound: FsqBoundKind,
    estimator: FsqEstimator,
    alpha: f32,
    seed: u64,
}

fn fsq_attributes<'a>(
    request: &'a TrainRequestV1<'_>,
) -> Result<FsqAttributes<'a>, TrainBackendError> {
    let channels = attribute_u64(request, "channels")?;
    let len = attribute_u64(request, "len")?;
    let channels_usize =
        usize::try_from(channels).map_err(|_| attribute_value("channels", "usize"))?;
    let len_usize = usize::try_from(len).map_err(|_| attribute_value("len", "usize"))?;
    if channels_usize == 0 {
        return Err(attribute_value("channels", "positive"));
    }
    if len_usize == 0 {
        return Err(attribute_value("len", "positive"));
    }
    channels_usize
        .checked_mul(len_usize)
        .ok_or_else(|| attribute_value("channels", "channels_times_len"))?;
    let levels = attribute_u32_list(request, "levels")?;
    if levels.len() != channels_usize {
        return Err(attribute_value("levels", "channels"));
    }
    if levels.iter().any(|&level| level < 2) {
        return Err(attribute_value("levels", "min_two"));
    }
    let bound = match attribute_text(request, "bound")? {
        "clamp" => FsqBoundKind::Clamp,
        "tanh" => FsqBoundKind::Tanh,
        _ => return Err(attribute_value("bound", "known")),
    };
    let estimator = match attribute_text(request, "ste")? {
        "hard" => FsqEstimator::Hard,
        "soft_round" => FsqEstimator::SoftRound,
        "stochastic" => FsqEstimator::Stochastic,
        _ => return Err(attribute_value("ste", "known")),
    };
    let alpha = attribute_f32(request, "alpha")?;
    if !(0.0..=1.0).contains(&alpha) {
        return Err(attribute_value("alpha", "unit_interval"));
    }
    let seed = attribute_u64(request, "seed")?;
    Ok(FsqAttributes {
        channels,
        len,
        len_usize,
        levels,
        bound,
        estimator,
        alpha,
        seed,
    })
}

fn salt_planes(request: &TrainRequestV1<'_>) -> Result<usize, TrainBackendError> {
    let value = attribute_u64(request, "planes")?;
    if value == 0 {
        return Err(attribute_value("planes", "positive"));
    }
    if value > MAX_SALT_PLANES {
        return Err(attribute_value("planes", "max_64"));
    }
    Ok(value as usize)
}

fn fsq_bound(value: f32, bound: FsqBoundKind) -> f32 {
    match bound {
        FsqBoundKind::Clamp => value.clamp(-1.0, 1.0),
        FsqBoundKind::Tanh => value.tanh(),
    }
}

fn fsq_bound_derivative(value: f32, bound: FsqBoundKind) -> f32 {
    match bound {
        FsqBoundKind::Clamp => {
            if value.abs() < 1.0 {
                1.0
            } else {
                0.0
            }
        }
        FsqBoundKind::Tanh => {
            let bounded = value.tanh();
            1.0 - bounded * bounded
        }
    }
}

fn fsq_quantize(value: f32, level: u32) -> f32 {
    let maximum = (level - 1) as f32;
    let position = (value + 1.0) * 0.5 * maximum;
    let code = round_half_away(position).clamp(0.0, maximum);
    code / maximum * 2.0 - 1.0
}

fn fsq_quantize_stochastic(value: f32, level: u32, seed: u64, index: usize) -> f32 {
    let maximum = (level - 1) as f32;
    let position = ((value + 1.0) * 0.5 * maximum).clamp(0.0, maximum);
    let floor = position.floor();
    let increment = if fsq_uniform(seed, index) < position - floor {
        1.0
    } else {
        0.0
    };
    (floor + increment).clamp(0.0, maximum) / maximum * 2.0 - 1.0
}

fn round_half_away(value: f32) -> f32 {
    if value >= 0.0 {
        (value + 0.5).floor()
    } else {
        (value - 0.5).ceil()
    }
}

fn fsq_uniform(seed: u64, index: usize) -> f32 {
    let mut state = (seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    (state % 1_000_000) as f32 / 1_000_000.0
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

fn input_bytes<'a>(
    request: &TrainRequestV1<'a>,
    name: &str,
) -> Result<(&'a [u64], &'a [u8]), TrainBackendError> {
    let buffer = request
        .inputs
        .iter()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| role_error("input"))?;
    match buffer.data {
        TrainBufferDataRefV1::Bytes(data) => Ok((buffer.shape, data)),
        data => Err(dtype_error(name, TrainDTypeV1::Bytes, ref_dtype(data))),
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

fn output_bytes<'a>(
    output: &'a mut TrainOutputV1<'_>,
    name: &str,
) -> Result<&'a mut [u8], TrainBackendError> {
    let buffer = output
        .buffers
        .iter_mut()
        .find(|buffer| buffer.name == name)
        .ok_or_else(|| role_error("output"))?;
    match &mut buffer.data {
        TrainBufferDataMutV1::Bytes(data) => Ok(data),
        data => Err(dtype_error(name, TrainDTypeV1::Bytes, mut_dtype(data))),
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

fn require_bytes_output(
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
    if !matches!(&buffer.data, TrainBufferDataMutV1::Bytes(_)) {
        return Err(dtype_error(
            buffer.name,
            TrainDTypeV1::Bytes,
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

fn adam_optimizer_attributes(
    request: &TrainRequestV1<'_>,
) -> Result<(u64, AdamW), TrainBackendError> {
    let step = attribute_u64(request, "step")?;
    let optimizer = AdamW {
        lr: attribute_f32(request, "lr")?,
        beta1: attribute_f32(request, "beta1")?,
        beta2: attribute_f32(request, "beta2")?,
        eps: attribute_f32(request, "eps")?,
        weight_decay: attribute_f32(request, "weight_decay")?,
    };
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
    Ok((step, optimizer))
}

fn muon_optimizer_attributes(
    request: &TrainRequestV1<'_>,
) -> Result<(u64, Muon), TrainBackendError> {
    let step = attribute_u64(request, "step")?;
    let optimizer = Muon {
        lr: attribute_f32(request, "lr")?,
        momentum: attribute_f32(request, "momentum")?,
        weight_decay: attribute_f32(request, "weight_decay")?,
        rows: portable_usize_attribute(request, "rows")?,
        cols: portable_usize_attribute(request, "cols")?,
        ns_steps: portable_usize_attribute(request, "ns_steps")?,
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
    bounded_product(&[optimizer.rows, optimizer.cols], "rows")?;
    Ok((step, optimizer))
}

fn matmul_attributes(
    request: &TrainRequestV1<'_>,
) -> Result<(u64, u64, u64, usize, usize, usize), TrainBackendError> {
    let m = attribute_u64(request, "m")?;
    let n = attribute_u64(request, "n")?;
    let k = attribute_u64(request, "k")?;
    let m_usize = usize::try_from(m).map_err(|_| attribute_value("m", "usize"))?;
    let n_usize = usize::try_from(n).map_err(|_| attribute_value("n", "usize"))?;
    let k_usize = usize::try_from(k).map_err(|_| attribute_value("k", "usize"))?;
    m_usize
        .checked_mul(k_usize)
        .ok_or_else(|| attribute_value("m", "m_times_k"))?;
    n_usize
        .checked_mul(k_usize)
        .ok_or_else(|| attribute_value("n", "n_times_k"))?;
    m_usize
        .checked_mul(n_usize)
        .ok_or_else(|| attribute_value("m", "m_times_n"))?;
    Ok((m, n, k, m_usize, n_usize, k_usize))
}

fn portable_usize_attribute(
    request: &TrainRequestV1<'_>,
    name: &str,
) -> Result<usize, TrainBackendError> {
    let value = attribute_u64(request, name)?;
    u32::try_from(value)
        .map(|value| value as usize)
        .map_err(|_| attribute_value(name, "u32"))
}

fn checked_output_axis(
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
    let output = (padded - effective) / stride as u64 + 1;
    u32::try_from(output)
        .map(|value| value as usize)
        .map_err(|_| attribute_value(name, "axis_u32"))
}

fn bounded_product(values: &[usize], name: &str) -> Result<usize, TrainBackendError> {
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

fn conv1d_attributes(
    request: &TrainRequestV1<'_>,
) -> Result<(conv1d::Conv1dCfg, usize), TrainBackendError> {
    let cfg = conv1d::Conv1dCfg {
        batch: portable_usize_attribute(request, "batch")?,
        c_in: portable_usize_attribute(request, "c_in")?,
        c_out: portable_usize_attribute(request, "c_out")?,
        l_in: portable_usize_attribute(request, "l_in")?,
        k: portable_usize_attribute(request, "k")?,
        stride: portable_usize_attribute(request, "stride")?,
        dilation: portable_usize_attribute(request, "dilation")?,
        pad_left: portable_usize_attribute(request, "pad_left")?,
        pad_right: portable_usize_attribute(request, "pad_right")?,
        groups: portable_usize_attribute(request, "groups")?,
    };
    for (name, value) in [
        ("batch", cfg.batch),
        ("c_in", cfg.c_in),
        ("c_out", cfg.c_out),
        ("l_in", cfg.l_in),
        ("k", cfg.k),
        ("stride", cfg.stride),
        ("dilation", cfg.dilation),
        ("groups", cfg.groups),
    ] {
        if value == 0 {
            return Err(attribute_value(name, "positive"));
        }
    }
    if !cfg.c_in.is_multiple_of(cfg.groups) || !cfg.c_out.is_multiple_of(cfg.groups) {
        return Err(attribute_value("groups", "divides_channels"));
    }
    let l_out = checked_output_axis(
        cfg.l_in,
        cfg.k,
        cfg.stride,
        cfg.dilation,
        cfg.pad_left,
        cfg.pad_right,
        "k",
    )?;
    let maximum_position = ((l_out - 1) as u64)
        .checked_mul(cfg.stride as u64)
        .and_then(|value| value.checked_add((cfg.k - 1) as u64 * cfg.dilation as u64))
        .ok_or_else(|| attribute_value("k", "index_arithmetic"))?;
    if maximum_position > i32::MAX as u64 || cfg.pad_left > i32::MAX as usize {
        return Err(attribute_value("k", "index_i32"));
    }
    let channels_per_group = cfg.c_in / cfg.groups;
    bounded_product(&[cfg.batch, cfg.c_in, cfg.l_in], "batch")?;
    bounded_product(&[cfg.c_out, channels_per_group, cfg.k], "c_out")?;
    bounded_product(&[cfg.batch, cfg.c_out, l_out], "batch")?;
    Ok((cfg, l_out))
}

fn conv2d_attributes(
    request: &TrainRequestV1<'_>,
) -> Result<(conv2d::Conv2dCfg, usize, usize), TrainBackendError> {
    let cfg = conv2d::Conv2dCfg {
        batch: portable_usize_attribute(request, "batch")?,
        c_in: portable_usize_attribute(request, "c_in")?,
        c_out: portable_usize_attribute(request, "c_out")?,
        input_h: portable_usize_attribute(request, "input_h")?,
        input_w: portable_usize_attribute(request, "input_w")?,
        kernel_h: portable_usize_attribute(request, "kernel_h")?,
        kernel_w: portable_usize_attribute(request, "kernel_w")?,
        stride_h: portable_usize_attribute(request, "stride_h")?,
        stride_w: portable_usize_attribute(request, "stride_w")?,
        dilation_h: portable_usize_attribute(request, "dilation_h")?,
        dilation_w: portable_usize_attribute(request, "dilation_w")?,
        pad_top: portable_usize_attribute(request, "pad_top")?,
        pad_bottom: portable_usize_attribute(request, "pad_bottom")?,
        pad_left: portable_usize_attribute(request, "pad_left")?,
        pad_right: portable_usize_attribute(request, "pad_right")?,
        groups: portable_usize_attribute(request, "groups")?,
    };
    for (name, value) in [
        ("batch", cfg.batch),
        ("c_in", cfg.c_in),
        ("c_out", cfg.c_out),
        ("input_h", cfg.input_h),
        ("input_w", cfg.input_w),
        ("kernel_h", cfg.kernel_h),
        ("kernel_w", cfg.kernel_w),
        ("stride_h", cfg.stride_h),
        ("stride_w", cfg.stride_w),
        ("dilation_h", cfg.dilation_h),
        ("dilation_w", cfg.dilation_w),
        ("groups", cfg.groups),
    ] {
        if value == 0 {
            return Err(attribute_value(name, "positive"));
        }
    }
    if !cfg.c_in.is_multiple_of(cfg.groups) || !cfg.c_out.is_multiple_of(cfg.groups) {
        return Err(attribute_value("groups", "divides_channels"));
    }
    let height_out = checked_output_axis(
        cfg.input_h,
        cfg.kernel_h,
        cfg.stride_h,
        cfg.dilation_h,
        cfg.pad_top,
        cfg.pad_bottom,
        "kernel_h",
    )?;
    let width_out = checked_output_axis(
        cfg.input_w,
        cfg.kernel_w,
        cfg.stride_w,
        cfg.dilation_w,
        cfg.pad_left,
        cfg.pad_right,
        "kernel_w",
    )?;
    let channels_per_group = cfg.c_in / cfg.groups;
    bounded_product(&[cfg.batch, cfg.c_in, cfg.input_h, cfg.input_w], "batch")?;
    bounded_product(
        &[cfg.c_out, channels_per_group, cfg.kernel_h, cfg.kernel_w],
        "c_out",
    )?;
    bounded_product(&[cfg.batch, cfg.c_out, height_out, width_out], "batch")?;
    Ok((cfg, height_out, width_out))
}

fn attention_attributes(
    request: &TrainRequestV1<'_>,
) -> Result<attention::AttentionCfg, TrainBackendError> {
    let seq = portable_usize_attribute(request, "seq")?;
    let n_head = portable_usize_attribute(request, "n_head")?;
    let n_kv_head = portable_usize_attribute(request, "n_kv_head")?;
    let head_dim = portable_usize_attribute(request, "head_dim")?;
    for (name, value) in [
        ("seq", seq),
        ("n_head", n_head),
        ("n_kv_head", n_kv_head),
        ("head_dim", head_dim),
    ] {
        if value == 0 {
            return Err(attribute_value(name, "positive"));
        }
    }
    if !n_head.is_multiple_of(n_kv_head) {
        return Err(attribute_value("n_kv_head", "divides_n_head"));
    }
    bounded_product(&[seq, n_head, head_dim], "seq")?;
    bounded_product(&[seq, n_kv_head, head_dim], "seq")?;
    bounded_product(&[seq, seq], "seq")?;
    Ok(attention::AttentionCfg {
        seq,
        n_head,
        n_kv_head,
        head_dim,
        causal: attribute_bool(request, "causal")?,
    })
}

fn conv2d_reference_error(error: conv2d::Conv2dError) -> TrainBackendError {
    match error {
        conv2d::Conv2dError::BufferLength { .. } => shape_error(),
        conv2d::Conv2dError::InvalidGeometry(_) => attribute_value("geometry", "invalid"),
        conv2d::Conv2dError::ArithmeticOverflow => attribute_value("geometry", "arithmetic"),
    }
}

fn checked_scratch_product(values: &[u64]) -> Result<u64, TrainBackendError> {
    values.iter().try_fold(1_u64, |total, &value| {
        total
            .checked_mul(value)
            .ok_or_else(|| attribute_value("scratch", "arithmetic"))
    })
}

fn checked_scratch_sum(values: &[u64]) -> Result<u64, TrainBackendError> {
    values.iter().try_fold(0_u64, |total, &value| {
        total
            .checked_add(value)
            .ok_or_else(|| attribute_value("scratch", "arithmetic"))
    })
}

fn operation_scratch_bytes(
    operation: CpuOperation,
    execution: TrainExecutionV1,
    request: &TrainRequestV1<'_>,
) -> Result<u64, TrainBackendError> {
    let elements = match (operation, execution) {
        (CpuOperation::SaltSte, TrainExecutionV1::Forward) => attribute_u64(request, "cols")?,
        (CpuOperation::Conv1d, phase) => {
            let (cfg, l_out) = conv1d_attributes(request)?;
            let input =
                checked_scratch_product(&[cfg.batch as u64, cfg.c_in as u64, cfg.l_in as u64])?;
            let weight = checked_scratch_product(&[
                cfg.c_out as u64,
                (cfg.c_in / cfg.groups) as u64,
                cfg.k as u64,
            ])?;
            let scale = cfg.c_out as u64;
            let columns = checked_scratch_product(&[
                l_out as u64,
                (cfg.c_in / cfg.groups) as u64,
                cfg.k as u64,
            ])?;
            let group_output =
                checked_scratch_product(&[l_out as u64, (cfg.c_out / cfg.groups) as u64])?;
            match phase {
                TrainExecutionV1::Forward => {
                    let output = checked_scratch_product(&[
                        cfg.batch as u64,
                        cfg.c_out as u64,
                        l_out as u64,
                    ])?;
                    checked_scratch_sum(&[output, columns, group_output])?
                }
                TrainExecutionV1::Vjp => {
                    let matmul_gradients = checked_scratch_sum(&[
                        columns,
                        weight / cfg.groups as u64,
                        scale / cfg.groups as u64,
                    ])?;
                    checked_scratch_sum(&[
                        input,
                        weight,
                        scale,
                        columns,
                        group_output,
                        matmul_gradients,
                    ])?
                }
                _ => 0,
            }
        }
        (CpuOperation::Conv2d, phase) => {
            let (cfg, height_out, width_out) = conv2d_attributes(request)?;
            let tile_rows = (height_out * width_out).min(conv2d::CONV2D_PATCH_TILE_ROWS) as u64;
            let patch_columns = checked_scratch_product(&[
                (cfg.c_in / cfg.groups) as u64,
                cfg.kernel_h as u64,
                cfg.kernel_w as u64,
            ])?;
            let group_channels = (cfg.c_out / cfg.groups) as u64;
            let columns = checked_scratch_product(&[tile_rows, patch_columns])?;
            let group_output = checked_scratch_product(&[tile_rows, group_channels])?;
            match phase {
                TrainExecutionV1::Forward => {
                    let output = checked_scratch_product(&[
                        cfg.batch as u64,
                        cfg.c_out as u64,
                        height_out as u64,
                        width_out as u64,
                    ])?;
                    checked_scratch_sum(&[output, columns, group_output])?
                }
                TrainExecutionV1::Vjp => {
                    let input = checked_scratch_product(&[
                        cfg.batch as u64,
                        cfg.c_in as u64,
                        cfg.input_h as u64,
                        cfg.input_w as u64,
                    ])?;
                    let weight = checked_scratch_product(&[cfg.c_out as u64, patch_columns])?;
                    let scale = cfg.c_out as u64;
                    let matmul_gradients = checked_scratch_sum(&[
                        columns,
                        weight / cfg.groups as u64,
                        group_channels,
                    ])?;
                    checked_scratch_sum(&[
                        input,
                        weight,
                        scale,
                        columns,
                        group_output,
                        matmul_gradients,
                    ])?
                }
                _ => 0,
            }
        }
        (CpuOperation::Attention, phase) => {
            let cfg = attention_attributes(request)?;
            let query = u64::try_from(
                cfg.query_elements()
                    .ok_or_else(|| attribute_value("seq", "query_elements"))?,
            )
            .map_err(|_| attribute_value("seq", "query_elements"))?;
            let kv = u64::try_from(
                cfg.kv_elements()
                    .ok_or_else(|| attribute_value("seq", "kv_elements"))?,
            )
            .map_err(|_| attribute_value("seq", "kv_elements"))?;
            let scores = checked_scratch_product(&[cfg.seq as u64, cfg.seq as u64])?;
            match phase {
                TrainExecutionV1::Forward => checked_scratch_sum(&[query, scores])?,
                TrainExecutionV1::Vjp => checked_scratch_sum(&[query, kv, kv, scores, scores])?,
                _ => 0,
            }
        }
        (CpuOperation::AdamW, TrainExecutionV1::Step) => {
            adam_optimizer_attributes(request)?;
            let (_, parameter) = input_f32(request, "parameter")?;
            checked_scratch_product(&[parameter.len() as u64, 2])?
        }
        (CpuOperation::CautiousAdamW, TrainExecutionV1::Step) => {
            adam_optimizer_attributes(request)?;
            let (_, parameter) = input_f32(request, "parameter")?;
            checked_scratch_product(&[parameter.len() as u64, 3])?
        }
        (CpuOperation::Int8AdamW, TrainExecutionV1::Step) => {
            adam_optimizer_attributes(request)?;
            let (_, parameter) = input_f32(request, "parameter")?;
            let len = parameter.len() as u64;
            let blocks = parameter.len().div_ceil(INT8_ADAM_BLOCK) as u64;
            let state_bytes = checked_scratch_sum(&[
                checked_scratch_product(&[len, 2])?,
                checked_scratch_product(&[blocks, 8])?,
            ])?;
            let block_elements = parameter.len().min(INT8_ADAM_BLOCK) as u64;
            let block_bytes = checked_scratch_product(&[block_elements, 2, 4])?;
            return checked_scratch_sum(&[state_bytes, block_bytes]);
        }
        (CpuOperation::Muon, TrainExecutionV1::Step) => {
            let (_, optimizer) = muon_optimizer_attributes(request)?;
            let matrix = checked_scratch_product(&[optimizer.rows as u64, optimizer.cols as u64])?;
            let gram_axis = optimizer.rows.min(optimizer.cols) as u64;
            let gram = checked_scratch_product(&[gram_axis, gram_axis])?;
            checked_scratch_sum(&[
                checked_scratch_product(&[matrix, 4])?,
                checked_scratch_product(&[gram, 3])?,
            ])?
        }
        (CpuOperation::Checkpoint, TrainExecutionV1::Checkpoint) => {
            let (optimizer, _, leaf_lens) = lifecycle_attributes(request, true)?;
            return checked_scratch_sum(&[
                checkpoint_payload_bytes(optimizer, leaf_lens)?,
                checkpoint_encoded_bytes(optimizer, leaf_lens)?,
            ]);
        }
        (CpuOperation::Resume, TrainExecutionV1::Resume) => {
            let (optimizer, _, leaf_lens) = lifecycle_attributes(request, false)?;
            return checkpoint_payload_bytes(optimizer, leaf_lens);
        }
        (CpuOperation::Export, TrainExecutionV1::Export) => {
            let (_, package) = input_bytes(request, "package")?;
            return salt_v2_validation_scratch_bytes(package.len());
        }
        (CpuOperation::Reload, TrainExecutionV1::Reload) => {
            let (_, artifact) = input_bytes(request, "artifact")?;
            return salt_v2_validation_scratch_bytes(artifact.len());
        }
        _ => 0,
    };
    elements
        .checked_mul(size_of::<f32>() as u64)
        .ok_or_else(|| attribute_value("scratch", "bytes"))
}

fn checkpoint_encoded_bytes(
    optimizer: CheckpointOptimizer,
    leaf_lens: &[u64],
) -> Result<u64, TrainBackendError> {
    let mut bytes = 4_u64 + 1 + 8 + 4;
    for &len in leaf_lens {
        let state = checkpoint_state_bytes(optimizer, len)?;
        bytes = checked_scratch_sum(&[bytes, 8, checked_scratch_product(&[len, 4])?, state])?;
    }
    Ok(bytes)
}

fn checkpoint_payload_bytes(
    optimizer: CheckpointOptimizer,
    leaf_lens: &[u64],
) -> Result<u64, TrainBackendError> {
    let mut bytes = 0_u64;
    for &len in leaf_lens {
        let state = checkpoint_state_bytes(optimizer, len)?;
        bytes = checked_scratch_sum(&[bytes, checked_scratch_product(&[len, 4])?, state])?;
    }
    Ok(bytes)
}

fn checkpoint_state_bytes(
    optimizer: CheckpointOptimizer,
    len: u64,
) -> Result<u64, TrainBackendError> {
    match optimizer {
        CheckpointOptimizer::Sgd => Ok(0),
        CheckpointOptimizer::AdamW | CheckpointOptimizer::CautiousAdamW => {
            checked_scratch_product(&[len, 8])
        }
        CheckpointOptimizer::Int8AdamW => checked_scratch_sum(&[
            checked_scratch_product(&[len, 2])?,
            checked_scratch_product(&[len.div_ceil(INT8_ADAM_BLOCK as u64), 8])?,
        ]),
        CheckpointOptimizer::Muon => checked_scratch_product(&[len, 4]),
    }
}

fn salt_v2_validation_scratch_bytes(package_len: usize) -> Result<u64, TrainBackendError> {
    checked_scratch_sum(&[
        payload_bytes(package_len, 8)?,
        SALT_V2_VALIDATION_FIXED_SCRATCH_BYTES,
    ])
}

fn cpu_physical_device() -> &'static str {
    static IDENTITY: OnceLock<String> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        let model = linux_cpu_model().unwrap_or_else(|| "unknown-model".to_owned());
        let logical_cpus = std::thread::available_parallelism()
            .map(core::num::NonZeroUsize::get)
            .unwrap_or(1);
        format!(
            "cpu:{}:{}:{model}:{logical_cpus}-logical",
            std::env::consts::OS,
            std::env::consts::ARCH
        )
    })
}

fn linux_cpu_model() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        cpuinfo.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            matches!(key.trim(), "model name" | "hardware")
                .then(|| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
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

#[allow(clippy::type_complexity)]
fn rope_attributes<'a>(
    request: &'a TrainRequestV1<'_>,
) -> Result<(&'a [u64], u64, u64, u64, usize, usize, f32), TrainBackendError> {
    let positions = attribute_u64_list(request, "positions")?;
    for &position in positions {
        u32::try_from(position).map_err(|_| attribute_value("positions", "u32"))?;
    }
    let n_token =
        u64::try_from(positions.len()).map_err(|_| attribute_value("positions", "u64"))?;
    let n_head = attribute_u64(request, "n_head")?;
    let head_dim = attribute_u64(request, "head_dim")?;
    if n_head == 0 {
        return Err(attribute_value("n_head", "positive"));
    }
    if head_dim == 0 {
        return Err(attribute_value("head_dim", "positive"));
    }
    let n_head_usize = usize::try_from(n_head).map_err(|_| attribute_value("n_head", "usize"))?;
    let head_dim_usize =
        usize::try_from(head_dim).map_err(|_| attribute_value("head_dim", "usize"))?;
    if !head_dim_usize.is_multiple_of(2) {
        return Err(attribute_value("head_dim", "even"));
    }
    positions
        .len()
        .checked_mul(n_head_usize)
        .and_then(|elements| elements.checked_mul(head_dim_usize))
        .ok_or_else(|| attribute_value("positions", "tensor_elements"))?;
    let theta = attribute_f32(request, "theta")?;
    if theta <= 0.0 {
        return Err(attribute_value("theta", "positive"));
    }
    Ok((
        positions,
        n_token,
        n_head,
        head_dim,
        n_head_usize,
        head_dim_usize,
        theta,
    ))
}

#[allow(clippy::type_complexity)]
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

fn attribute_bool(request: &TrainRequestV1<'_>, name: &str) -> Result<bool, TrainBackendError> {
    match request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .map(|attribute| attribute.value)
    {
        Some(TrainAttributeValueV1::Bool(value)) => Ok(value),
        _ => Err(attribute_type(name, "bool")),
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

fn attribute_u32_list<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<&'a [u32], TrainBackendError> {
    match request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .map(|attribute| attribute.value)
    {
        Some(TrainAttributeValueV1::U32List(value)) => Ok(value),
        _ => Err(attribute_type(name, "u32_list")),
    }
}

fn attribute_text<'a>(
    request: &'a TrainRequestV1<'_>,
    name: &str,
) -> Result<&'a str, TrainBackendError> {
    match request
        .attributes
        .iter()
        .find(|attribute| attribute.name == name)
        .map(|attribute| attribute.value)
    {
        Some(TrainAttributeValueV1::Text(value)) => Ok(value),
        _ => Err(attribute_type(name, "text")),
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
