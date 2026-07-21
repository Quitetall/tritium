//! Layer 2 — `ort` 2.x custom operators wrapping the always-on Layer-1 kernels.
//!
//! [`TritiumTernaryMpGemmOp`] implements [`ort::operator::Operator`], describing
//! an ONNX node [`ONNX_OP_NAME`] with three tensor inputs and one output:
//!
//! | slot     | name       | dtype | shape       | meaning                              |
//! |----------|------------|-------|-------------|--------------------------------------|
//! | input 0  | `act`      | f32   | `[M, K]`    | row-major activations                |
//! | input 1  | `packed`   | u8    | `[bytes]`   | `[N, K]` ternary weights in `format` |
//! | input 2  | `scales`   | f32   | `[N]`       | per-output-channel scales            |
//! | output 0 | `out`      | f32   | `[M, N]`    | `scale[n] · Σ_k act[m,k]·w[n,k]`     |
//!
//! plus two `i64` node attributes:
//! - `K` — the contraction dimension (the packed byte count alone is ambiguous
//!   because the last quant block is zero-padded), and
//! - `format` — the packing scheme: `0` = TQ2_0, `1` = TQ1_0 (see
//!   [`format_from_attr`]).
//!
//! `M` is read from `act`'s shape, `N` from `scales`' length. The kernel's
//! compute logic (`run`) calls [`crate::ternary_mpgemm_kernel`] — the same
//! bit-exact reference path the always-on Layer-1 conformance gate covers — and
//! `run` is itself tested bit-exact against the frozen vectors
//! (`kernel_run_matches_reference_on_frozen_set`). CI also serializes a real
//! ONNX graph, registers the production custom domain, opens an ONNX Runtime
//! session and proves that runtime dispatch invokes the kernel bit-exactly.
//!
//! [`TritiumTernaryEmbeddingOp`] uses the same packed/scales representation but
//! accepts an arbitrary-rank `i64` token tensor and appends `K` to its output
//! shape. Only selected rows are unpacked; no dense vocabulary-sized table is
//! constructed.

use ort::error::Error as OrtError;
use ort::operator::{
    Kernel, KernelAttributes, KernelContext, Operator, OperatorDomain, OperatorInput,
    OperatorOutput,
};
use ort::value::TensorElementType;
use tritium_core::TernaryFormat;

use crate::{
    ATTR_CODEC, ATTR_CONV_KERNEL_DIM, ATTR_FORMAT, ATTR_HEAD_DIM, ATTR_K, ATTR_KEY_HEAD_DIM,
    ATTR_N_HEAD, ATTR_N_KV_HEAD, ATTR_NUM_KEY_HEADS, ATTR_NUM_VALUE_HEADS, ATTR_PAST_TOKENS,
    ATTR_ROWS, ATTR_TERMINAL_MAP_VALUE, ATTR_VALUE_HEAD_DIM, ONNX_DOMAIN, ONNX_EMBEDDING_OP_NAME,
    ONNX_KV_ATTENTION_OP_NAME, ONNX_OP_NAME, ONNX_QWEN_DELTANET_OP_NAME,
    ONNX_SALT_V2_EMBEDDING_OP_NAME, ONNX_SALT_V2_OP_NAME, QwenDeltaNetGeometry, QwenDeltaNetInput,
    QwenDeltaNetOutputSlot, SaltV2PackedMatrix, kv_attention_kernel, salt_v2_embedding_kernel,
    salt_v2_embedding_kernel_admitted, salt_v2_mpgemm_kernel, ternary_embedding_kernel,
    ternary_mpgemm_kernel,
};

/// Map the integer `format` node attribute to a [`TernaryFormat`].
///
/// `0` → [`TernaryFormat::Tq2_0`], `1` → [`TernaryFormat::Tq1_0`]. Any other
/// value is an error (the GPU-only `I2sInt8` packing is not consumed here).
///
/// # Errors
/// An [`ort::Error`] for an unrecognized format code.
fn format_from_attr(code: i64) -> Result<TernaryFormat, OrtError> {
    match code {
        0 => Ok(TernaryFormat::Tq2_0),
        1 => Ok(TernaryFormat::Tq1_0),
        other => Err(OrtError::new(format!(
            "tritium-onnx: unknown format code {other} (expected 0=TQ2_0, 1=TQ1_0)"
        ))),
    }
}

fn kernel_config(attributes: &KernelAttributes) -> ort::Result<(usize, TernaryFormat)> {
    let k: i64 = attributes
        .get(ATTR_K)
        .ok_or_else(|| OrtError::new(format!("tritium-onnx: missing i64 attribute `{ATTR_K}`")))?;
    if k <= 0 {
        return Err(OrtError::new(format!(
            "tritium-onnx: attribute `{ATTR_K}` must be positive, got {k}"
        )));
    }
    let format_code: i64 = attributes.get(ATTR_FORMAT).ok_or_else(|| {
        OrtError::new(format!(
            "tritium-onnx: missing i64 attribute `{ATTR_FORMAT}`"
        ))
    })?;
    let k = usize::try_from(k)
        .map_err(|_| OrtError::new("tritium-onnx: attribute `K` exceeds usize"))?;
    Ok((k, format_from_attr(format_code)?))
}

/// The `ort` custom operator descriptor for Tritium's ternary mpGEMM node.
///
/// Register it on a session via [`tritium_operator_domain`] /
/// [`ort::operator::OperatorDomain`]. It is stateless (the per-node `K` and
/// `format` arrive as attributes), so a single value can describe every
/// `TritiumTernaryMpGemm` node in a graph.
#[derive(Debug, Default, Clone, Copy)]
pub struct TritiumTernaryMpGemmOp;

impl Operator for TritiumTernaryMpGemmOp {
    fn name(&self) -> &str {
        ONNX_OP_NAME
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            // act [M, K] f32
            OperatorInput::required(TensorElementType::Float32),
            // packed [bytes] u8
            OperatorInput::required(TensorElementType::Uint8),
            // scales [N] f32
            OperatorInput::required(TensorElementType::Float32),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        // out [M, N] f32
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
        // K and format arrive as i64 node attributes; the byte count alone does
        // not determine K (the last quant block is zero-padded).
        let (k, format) = kernel_config(attributes)?;
        Ok(Box::new(TritiumTernaryMpGemmKernel { k, format }))
    }
}

/// Custom operator for descriptor-free additive SALT V2 matrix multiplication.
#[derive(Debug, Default, Clone, Copy)]
pub struct TritiumSaltV2MpGemmOp;

fn salt_v2_kernel_config(
    attributes: &KernelAttributes,
) -> ort::Result<(usize, usize, tritium_format::salt_v2::SaltV2Codec, u32)> {
    let columns = usize_attribute(attributes, ATTR_K, false)?;
    let rows = usize_attribute(attributes, ATTR_ROWS, false)?;
    let codec: i64 = attributes
        .get(ATTR_CODEC)
        .ok_or_else(|| OrtError::new("tritium-onnx: missing i64 attribute `codec`"))?;
    let codec = match codec {
        0 => tritium_format::salt_v2::SaltV2Codec::D2,
        1 => tritium_format::salt_v2::SaltV2Codec::B3,
        2 => tritium_format::salt_v2::SaltV2Codec::S34,
        value => {
            return Err(OrtError::new(format!(
                "tritium-onnx: unknown SALT V2 codec {value}"
            )));
        }
    };
    let terminal: i64 = attributes
        .get(ATTR_TERMINAL_MAP_VALUE)
        .ok_or_else(|| OrtError::new("tritium-onnx: missing i64 attribute `terminal_map_value`"))?;
    let terminal_map_value = u32::try_from(terminal)
        .map_err(|_| OrtError::new("tritium-onnx: terminal_map_value must fit unsigned 32 bits"))?;
    Ok((rows, columns, codec, terminal_map_value))
}

impl Operator for TritiumSaltV2MpGemmOp {
    fn name(&self) -> &str {
        ONNX_SALT_V2_OP_NAME
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float32),
            OperatorInput::required(TensorElementType::Uint8),
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Uint8),
            OperatorInput::required(TensorElementType::Uint32),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn min_version(&self) -> i32 {
        2
    }

    fn max_version(&self) -> i32 {
        2
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
        let (rows, columns, codec, terminal_map_value) = salt_v2_kernel_config(attributes)?;
        Ok(Box::new(TritiumSaltV2MpGemmKernel {
            rows,
            columns,
            codec,
            terminal_map_value,
        }))
    }
}

/// Per-node additive SALT V2 ORT kernel.
#[derive(Debug, Clone, Copy)]
pub struct TritiumSaltV2MpGemmKernel {
    rows: usize,
    columns: usize,
    codec: tritium_format::salt_v2::SaltV2Codec,
    terminal_map_value: u32,
}

impl TritiumSaltV2MpGemmKernel {
    /// Execute one extracted ORT operand set.
    ///
    /// # Errors
    /// Returns an [`ort::Error`] when shapes, arenas, metadata, or arithmetic
    /// violate the SALT V2 contract.
    pub fn run(
        &self,
        act_shape: &[i64],
        activation: &[f32],
        payload: &[u8],
        scales: &[half::f16],
        allocation_map: &[u8],
        rank_prefixes: &[u32],
    ) -> Result<Vec<f32>, OrtError> {
        if act_shape.len() != 2 || usize::try_from(act_shape[1]) != Ok(self.columns) {
            return Err(OrtError::new(format!(
                "tritium-onnx: SALT V2 activation must be [M, {}], got {act_shape:?}",
                self.columns
            )));
        }
        let m = usize::try_from(act_shape[0])
            .map_err(|_| OrtError::new("tritium-onnx: SALT V2 activation M must be nonnegative"))?;
        salt_v2_mpgemm_kernel(
            activation,
            m,
            SaltV2PackedMatrix {
                rows: self.rows,
                columns: self.columns,
                codec: self.codec,
                payload,
                scales,
                allocation_map,
                rank_prefixes,
                terminal_map_value: self.terminal_map_value,
            },
        )
        .map_err(|error| OrtError::new(error.to_string()))
    }
}

impl Kernel for TritiumSaltV2MpGemmKernel {
    fn compute(&mut self, ctx: &KernelContext) -> ort::Result<()> {
        let act = ctx
            .input(0)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 activation"))?;
        let payload = ctx
            .input(1)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 payload"))?;
        let scales = ctx
            .input(2)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 scales"))?;
        let allocation_map = ctx
            .input(3)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 allocation map"))?;
        let rank_prefixes = ctx
            .input(4)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 rank prefixes"))?;
        let (shape, activation) = act.try_extract_tensor::<f32>()?;
        let (_, payload) = payload.try_extract_tensor::<u8>()?;
        let (_, scales) = scales.try_extract_tensor::<half::f16>()?;
        let (_, allocation_map) = allocation_map.try_extract_tensor::<u8>()?;
        let (_, rank_prefixes) = rank_prefixes.try_extract_tensor::<u32>()?;
        let shape: Vec<i64> = shape.iter().copied().collect();
        let output = self.run(
            &shape,
            activation,
            payload,
            scales,
            allocation_map,
            rank_prefixes,
        )?;
        let mut value = ctx
            .output(0, vec![shape[0], self.rows as i64])?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 output"))?;
        let (_, destination) = value.try_extract_tensor_mut::<f32>()?;
        destination.copy_from_slice(&output);
        Ok(())
    }
}

/// Custom operator for descriptor-free additive SALT V2 embedding lookup.
#[derive(Debug, Default, Clone, Copy)]
pub struct TritiumSaltV2EmbeddingOp;

impl Operator for TritiumSaltV2EmbeddingOp {
    fn name(&self) -> &str {
        ONNX_SALT_V2_EMBEDDING_OP_NAME
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Int64),
            OperatorInput::required(TensorElementType::Uint8),
            OperatorInput::required(TensorElementType::Float16),
            OperatorInput::required(TensorElementType::Uint8),
            OperatorInput::required(TensorElementType::Uint32),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn min_version(&self) -> i32 {
        2
    }

    fn max_version(&self) -> i32 {
        2
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
        let (rows, columns, codec, terminal_map_value) = salt_v2_kernel_config(attributes)?;
        Ok(Box::new(TritiumSaltV2EmbeddingKernel {
            rows,
            columns,
            codec,
            terminal_map_value,
            validated: false,
        }))
    }
}

/// Per-node additive SALT V2 selected-row ORT kernel.
#[derive(Debug, Clone)]
pub struct TritiumSaltV2EmbeddingKernel {
    rows: usize,
    columns: usize,
    codec: tritium_format::salt_v2::SaltV2Codec,
    terminal_map_value: u32,
    validated: bool,
}

impl TritiumSaltV2EmbeddingKernel {
    /// Execute one extracted ORT operand set and return output shape plus values.
    ///
    /// # Errors
    /// Returns an [`ort::Error`] when token shape/IDs, arenas, metadata, or
    /// codec data violate the SALT V2 contract.
    pub fn run(
        &self,
        token_shape: &[i64],
        tokens: &[i64],
        payload: &[u8],
        scales: &[half::f16],
        allocation_map: &[u8],
        rank_prefixes: &[u32],
    ) -> Result<(Vec<i64>, Vec<f32>), OrtError> {
        self.run_impl(
            token_shape,
            tokens,
            payload,
            scales,
            allocation_map,
            rank_prefixes,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_impl(
        &self,
        token_shape: &[i64],
        tokens: &[i64],
        payload: &[u8],
        scales: &[half::f16],
        allocation_map: &[u8],
        rank_prefixes: &[u32],
        validate_payload: bool,
    ) -> Result<(Vec<i64>, Vec<f32>), OrtError> {
        let dimensions: Vec<usize> = token_shape
            .iter()
            .enumerate()
            .map(|(axis, &dimension)| {
                usize::try_from(dimension).map_err(|_| {
                    OrtError::new(format!(
                        "tritium-onnx: SALT V2 token dimension {axis} must be nonnegative"
                    ))
                })
            })
            .collect::<Result<_, _>>()?;
        let matrix = SaltV2PackedMatrix {
            rows: self.rows,
            columns: self.columns,
            codec: self.codec,
            payload,
            scales,
            allocation_map,
            rank_prefixes,
            terminal_map_value: self.terminal_map_value,
        };
        let output = if validate_payload {
            salt_v2_embedding_kernel(tokens, &dimensions, matrix)
        } else {
            salt_v2_embedding_kernel_admitted(tokens, &dimensions, matrix)
        }
        .map_err(|error| OrtError::new(error.to_string()))?;
        let mut output_shape = token_shape.to_vec();
        output_shape.push(self.columns as i64);
        Ok((output_shape, output))
    }
}

impl Kernel for TritiumSaltV2EmbeddingKernel {
    fn compute(&mut self, ctx: &KernelContext) -> ort::Result<()> {
        let tokens = ctx
            .input(0)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 tokens"))?;
        let payload = ctx
            .input(1)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 payload"))?;
        let scales = ctx
            .input(2)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 scales"))?;
        let allocation_map = ctx
            .input(3)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 allocation map"))?;
        let rank_prefixes = ctx
            .input(4)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 rank prefixes"))?;
        let (shape, tokens) = tokens.try_extract_tensor::<i64>()?;
        let (_, payload) = payload.try_extract_tensor::<u8>()?;
        let (_, scales) = scales.try_extract_tensor::<half::f16>()?;
        let (_, allocation_map) = allocation_map.try_extract_tensor::<u8>()?;
        let (_, rank_prefixes) = rank_prefixes.try_extract_tensor::<u32>()?;
        let shape: Vec<i64> = shape.iter().copied().collect();
        if !self.validated {
            SaltV2PackedMatrix {
                rows: self.rows,
                columns: self.columns,
                codec: self.codec,
                payload,
                scales,
                allocation_map,
                rank_prefixes,
                terminal_map_value: self.terminal_map_value,
            }
            .validate()
            .map_err(|error| OrtError::new(error.to_string()))?;
            self.validated = true;
        }
        let (output_shape, output) = self.run_impl(
            &shape,
            tokens,
            payload,
            scales,
            allocation_map,
            rank_prefixes,
            false,
        )?;
        let mut value = ctx
            .output(0, output_shape)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing SALT V2 embedding output"))?;
        let (_, destination) = value.try_extract_tensor_mut::<f32>()?;
        destination.copy_from_slice(&output);
        Ok(())
    }
}

/// The custom operator descriptor for packed ternary embedding lookup.
#[derive(Debug, Default, Clone, Copy)]
pub struct TritiumTernaryEmbeddingOp;

impl Operator for TritiumTernaryEmbeddingOp {
    fn name(&self) -> &str {
        ONNX_EMBEDDING_OP_NAME
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Int64),
            OperatorInput::required(TensorElementType::Uint8),
            OperatorInput::required(TensorElementType::Float32),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
        let (k, format) = kernel_config(attributes)?;
        Ok(Box::new(TritiumTernaryEmbeddingKernel { k, format }))
    }
}

/// Per-node packed ternary embedding kernel.
#[derive(Debug, Clone, Copy)]
pub struct TritiumTernaryEmbeddingKernel {
    k: usize,
    format: TernaryFormat,
}

impl TritiumTernaryEmbeddingKernel {
    /// Gather selected rows and return `(output_shape, flat_output)`.
    ///
    /// # Errors
    /// An [`ort::Error`] if a token dimension is negative or the Layer-1
    /// embedding kernel rejects the inputs.
    pub fn run(
        &self,
        token_shape: &[i64],
        tokens: &[i64],
        packed: &[u8],
        scales: &[f32],
    ) -> Result<(Vec<i64>, Vec<f32>), OrtError> {
        let dimensions: Vec<usize> = token_shape
            .iter()
            .enumerate()
            .map(|(axis, &dimension)| {
                usize::try_from(dimension).map_err(|_| {
                    OrtError::new(format!(
                        "tritium-onnx: token dimension {axis} must be nonnegative, got {dimension}"
                    ))
                })
            })
            .collect::<Result<_, _>>()?;
        let output =
            ternary_embedding_kernel(tokens, &dimensions, packed, scales, self.k, self.format)
                .map_err(|error| OrtError::new(error.to_string()))?;
        let mut output_shape = token_shape.to_vec();
        output_shape.push(self.k as i64);
        Ok((output_shape, output))
    }
}

impl Kernel for TritiumTernaryEmbeddingKernel {
    fn compute(&mut self, ctx: &KernelContext) -> ort::Result<()> {
        let tokens_v = ctx
            .input(0)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing input 0 (tokens)"))?;
        let packed_v = ctx
            .input(1)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing input 1 (packed)"))?;
        let scales_v = ctx
            .input(2)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing input 2 (scales)"))?;

        let (token_shape, tokens) = tokens_v.try_extract_tensor::<i64>()?;
        let (_, packed) = packed_v.try_extract_tensor::<u8>()?;
        let (_, scales) = scales_v.try_extract_tensor::<f32>()?;
        let token_dimensions: Vec<i64> = token_shape.iter().copied().collect();
        let (output_shape, output) = self.run(&token_dimensions, tokens, packed, scales)?;
        let mut output_v = ctx
            .output(0, output_shape)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing output 0"))?;
        let (_, output_ref) = output_v.try_extract_tensor_mut::<f32>()?;
        output_ref.copy_from_slice(&output);
        Ok(())
    }
}

fn usize_attribute(
    attributes: &KernelAttributes,
    name: &'static str,
    allow_zero: bool,
) -> ort::Result<usize> {
    let value: i64 = attributes
        .get(name)
        .ok_or_else(|| OrtError::new(format!("tritium-onnx: missing i64 attribute `{name}`")))?;
    checked_usize_attribute(value, name, allow_zero)
}

fn checked_usize_attribute(value: i64, name: &'static str, allow_zero: bool) -> ort::Result<usize> {
    let minimum = if allow_zero { 0 } else { 1 };
    if value < minimum {
        return Err(OrtError::new(format!(
            "tritium-onnx: attribute `{name}` must be {}, got {value}",
            if allow_zero {
                "nonnegative"
            } else {
                "positive"
            }
        )));
    }
    usize::try_from(value)
        .map_err(|_| OrtError::new(format!("tritium-onnx: attribute `{name}` exceeds usize")))
}

/// Custom operator descriptor for cache-aware causal grouped-query attention.
#[derive(Debug, Default, Clone, Copy)]
pub struct TritiumKvAttentionOp;

impl Operator for TritiumKvAttentionOp {
    fn name(&self) -> &str {
        ONNX_KV_ATTENTION_OP_NAME
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Float32),
            OperatorInput::required(TensorElementType::Float32),
            OperatorInput::required(TensorElementType::Float32),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }

    fn min_version(&self) -> i32 {
        2
    }

    fn max_version(&self) -> i32 {
        3
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
        Ok(Box::new(TritiumKvAttentionKernel {
            n_head: usize_attribute(attributes, ATTR_N_HEAD, false)?,
            n_kv_head: usize_attribute(attributes, ATTR_N_KV_HEAD, false)?,
            head_dim: usize_attribute(attributes, ATTR_HEAD_DIM, false)?,
            past_tokens: attributes
                .get::<i64>(ATTR_PAST_TOKENS)
                .map(|value| checked_usize_attribute(value, ATTR_PAST_TOKENS, true))
                .transpose()?,
        }))
    }
}

/// Per-node cache-aware attention kernel with frozen head geometry.
#[derive(Debug, Clone, Copy)]
pub struct TritiumKvAttentionKernel {
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    past_tokens: Option<usize>,
}

impl TritiumKvAttentionKernel {
    /// Execute over extracted rank-3 Q/K/V tensors.
    ///
    /// # Errors
    /// An [`ort::Error`] if shapes disagree with attributes/cache offset or the
    /// semantic oracle rejects flat input lengths.
    pub fn run(
        &self,
        q_shape: &[i64],
        q: &[f32],
        k_shape: &[i64],
        k_cache: &[f32],
        v_shape: &[i64],
        v_cache: &[f32],
    ) -> Result<Vec<f32>, OrtError> {
        if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
            return Err(OrtError::new(format!(
                "tritium-onnx: KV attention requires rank-3 Q/K/V, got {q_shape:?}, {k_shape:?}, {v_shape:?}"
            )));
        }
        let expected_q_tail = [self.n_head as i64, self.head_dim as i64];
        let expected_kv_tail = [self.n_kv_head as i64, self.head_dim as i64];
        if q_shape[1..] != expected_q_tail {
            return Err(OrtError::new(format!(
                "tritium-onnx: Q tail {:?} does not match {:?}",
                &q_shape[1..],
                expected_q_tail
            )));
        }
        if k_shape[1..] != expected_kv_tail || v_shape != k_shape {
            return Err(OrtError::new(format!(
                "tritium-onnx: K/V shapes {k_shape:?}/{v_shape:?} do not match cache tail {expected_kv_tail:?}"
            )));
        }
        let query_tokens = usize::try_from(q_shape[0])
            .map_err(|_| OrtError::new("tritium-onnx: negative query token count"))?;
        let total_tokens = usize::try_from(k_shape[0])
            .map_err(|_| OrtError::new("tritium-onnx: negative KV cache token count"))?;
        let inferred_past = total_tokens.checked_sub(query_tokens).ok_or_else(|| {
            OrtError::new(format!(
                "tritium-onnx: cache token count {total_tokens} is smaller than query count {query_tokens}"
            ))
        })?;
        if self
            .past_tokens
            .is_some_and(|declared| declared != inferred_past)
        {
            return Err(OrtError::new(format!(
                "tritium-onnx: inferred past {inferred_past} differs from declared past {}",
                self.past_tokens.expect("checked present")
            )));
        }
        kv_attention_kernel(
            q,
            k_cache,
            v_cache,
            query_tokens,
            self.n_head,
            self.n_kv_head,
            self.head_dim,
            inferred_past,
        )
        .map_err(|error| OrtError::new(error.to_string()))
    }
}

impl Kernel for TritiumKvAttentionKernel {
    fn compute(&mut self, ctx: &KernelContext) -> ort::Result<()> {
        let q_value = ctx
            .input(0)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing input 0 (q)"))?;
        let k_value = ctx
            .input(1)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing input 1 (k_cache)"))?;
        let v_value = ctx
            .input(2)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing input 2 (v_cache)"))?;
        let (q_shape, q) = q_value.try_extract_tensor::<f32>()?;
        let (k_shape, k_cache) = k_value.try_extract_tensor::<f32>()?;
        let (v_shape, v_cache) = v_value.try_extract_tensor::<f32>()?;
        let q_dimensions: Vec<i64> = q_shape.iter().copied().collect();
        let k_dimensions: Vec<i64> = k_shape.iter().copied().collect();
        let v_dimensions: Vec<i64> = v_shape.iter().copied().collect();
        let output = self.run(
            &q_dimensions,
            q,
            &k_dimensions,
            k_cache,
            &v_dimensions,
            v_cache,
        )?;
        let mut output_value = ctx
            .output(0, q_dimensions)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing output 0"))?;
        let (_, output_ref) = output_value.try_extract_tensor_mut::<f32>()?;
        output_ref.copy_from_slice(&output);
        Ok(())
    }
}

/// Custom operator descriptor for projected Qwen3.5 Gated DeltaNet recurrence.
#[derive(Debug, Default, Clone, Copy)]
pub struct TritiumQwenDeltaNetOp;

impl Operator for TritiumQwenDeltaNetOp {
    fn name(&self) -> &str {
        ONNX_QWEN_DELTANET_OP_NAME
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        QwenDeltaNetInput::ALL
            .iter()
            .map(|_| OperatorInput::required(TensorElementType::Float32))
            .collect()
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        QwenDeltaNetOutputSlot::ALL
            .iter()
            .map(|_| OperatorOutput::required(TensorElementType::Float32))
            .collect()
    }

    fn min_version(&self) -> i32 {
        2
    }

    fn max_version(&self) -> i32 {
        2
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
        let geometry = QwenDeltaNetGeometry::new(
            usize_attribute(attributes, ATTR_CONV_KERNEL_DIM, false)?,
            usize_attribute(attributes, ATTR_NUM_KEY_HEADS, false)?,
            usize_attribute(attributes, ATTR_NUM_VALUE_HEADS, false)?,
            usize_attribute(attributes, ATTR_KEY_HEAD_DIM, false)?,
            usize_attribute(attributes, ATTR_VALUE_HEAD_DIM, false)?,
        )
        .map_err(delta_geometry_error)?;
        Ok(Box::new(TritiumQwenDeltaNetKernel::new(geometry)))
    }
}

fn delta_geometry_error(error: crate::QwenDeltaNetGeometryError) -> OrtError {
    OrtError::new(format!("tritium-onnx: {error}"))
}

/// One borrowed projected DeltaNet tensor with its exact ONNX shape.
#[derive(Debug, Clone, Copy)]
pub struct QwenDeltaNetTensor<'a> {
    /// Exact ONNX tensor shape.
    pub shape: &'a [i64],
    /// Flat row-major tensor elements.
    pub values: &'a [f32],
}

/// Borrowed projected operands, parameters, and prior state for one transition.
#[derive(Debug, Clone, Copy)]
pub struct QwenDeltaNetInputs<'a> {
    /// Raw globally split Q/K/V projection.
    pub raw_qkv: QwenDeltaNetTensor<'a>,
    /// Swish output-gate projection.
    pub z: QwenDeltaNetTensor<'a>,
    /// Per-value-head beta logits.
    pub beta_logits: QwenDeltaNetTensor<'a>,
    /// Per-value-head decay logits.
    pub decay_logits: QwenDeltaNetTensor<'a>,
    /// Depthwise convolution weights and exact `[conv_width, kernel]` shape.
    pub conv_weight: QwenDeltaNetTensor<'a>,
    /// Gated RMSNorm weights and exact `[value_head_dim]` shape.
    pub norm_weight: QwenDeltaNetTensor<'a>,
    /// Per-value-head delta-time bias and exact `[num_value_heads]` shape.
    pub dt_bias: QwenDeltaNetTensor<'a>,
    /// Per-value-head log decay and exact `[num_value_heads]` shape.
    pub a_log: QwenDeltaNetTensor<'a>,
    /// Prior depthwise history and exact `[conv_width, kernel]` shape.
    pub conv_state: QwenDeltaNetTensor<'a>,
    /// Prior recurrence and exact `[num_value_heads, key_head_dim, value_head_dim]` shape.
    pub recurrent_state: QwenDeltaNetTensor<'a>,
    /// Positive finite RMSNorm epsilon from a scalar tensor.
    pub epsilon: f32,
}

#[derive(Debug, Clone, Copy)]
/// Projected Qwen3.5 Gated DeltaNet recurrent core.
///
/// Packed QKV/gate/output projections remain separate Tritium mpGEMM nodes.
/// This kernel owns depthwise causal convolution, normalized delta-rule state,
/// gated RMSNorm, and explicit next-state publication.
pub struct TritiumQwenDeltaNetKernel {
    geometry: QwenDeltaNetGeometry,
}

/// Owned outputs from one projected DeltaNet state transition.
#[derive(Debug, Clone, PartialEq)]
pub struct QwenDeltaNetOutput {
    /// Gated normalized recurrent output `[tokens, value_width]`.
    pub normalized_core: Vec<f32>,
    /// Updated depthwise history `[conv_width, conv_kernel_dim]`.
    pub conv_state: Vec<f32>,
    /// Updated delta-rule memory `[num_value_heads, key_head_dim, value_head_dim]`.
    pub recurrent_state: Vec<f32>,
}

impl TritiumQwenDeltaNetKernel {
    /// Construct a direct-execution kernel from validated geometry.
    #[must_use]
    pub const fn new(geometry: QwenDeltaNetGeometry) -> Self {
        Self { geometry }
    }

    /// Execute one or more projected rows and publish complete next state.
    ///
    /// # Errors
    /// Returns an [`ort::Error`] for invalid shapes, non-finite preserved
    /// parameters, zero tokens, or invalid RMSNorm epsilon.
    pub fn run(&self, inputs: QwenDeltaNetInputs<'_>) -> Result<QwenDeltaNetOutput, OrtError> {
        let dimensions = self.geometry.dimensions().map_err(delta_geometry_error)?;
        let conv_kernel_dim = self.geometry.conv_kernel_dim();
        let num_key_heads = self.geometry.num_key_heads();
        let num_value_heads = self.geometry.num_value_heads();
        let key_head_dim = self.geometry.key_head_dim();
        let value_head_dim = self.geometry.value_head_dim();
        let key_width = dimensions.key_width();
        let value_width = dimensions.value_width();
        let conv_width = dimensions.conv_width();
        if !inputs.epsilon.is_finite() || inputs.epsilon <= 0.0 {
            return Err(OrtError::new(
                "tritium-onnx: DeltaNet RMSNorm epsilon must be finite and positive",
            ));
        }
        let tokens = matrix_tokens(inputs.raw_qkv.shape, conv_width, "raw_qkv")?;
        if tokens == 0 {
            return Err(OrtError::new(
                "tritium-onnx: DeltaNet token count must be positive",
            ));
        }
        require_len(inputs.raw_qkv.values, tokens, conv_width, "raw_qkv")?;
        require_matrix(inputs.z, tokens, value_width, "z")?;
        require_matrix(inputs.beta_logits, tokens, num_value_heads, "beta_logits")?;
        require_matrix(inputs.decay_logits, tokens, num_value_heads, "decay_logits")?;
        require_shaped(
            inputs.conv_weight,
            &[conv_width, conv_kernel_dim],
            "conv_weight",
        )?;
        require_shaped(
            inputs.conv_state,
            &[conv_width, conv_kernel_dim],
            "conv_state",
        )?;
        require_shaped(
            inputs.recurrent_state,
            &[num_value_heads, key_head_dim, value_head_dim],
            "recurrent_state",
        )?;
        require_shaped(inputs.norm_weight, &[value_head_dim], "norm_weight")?;
        require_shaped(inputs.dt_bias, &[num_value_heads], "dt_bias")?;
        require_shaped(inputs.a_log, &[num_value_heads], "a_log")?;
        debug_assert_eq!(inputs.conv_weight.values.len(), dimensions.conv_state_len());
        debug_assert_eq!(
            inputs.recurrent_state.values.len(),
            dimensions.recurrent_state_len()
        );
        if inputs
            .conv_weight
            .values
            .iter()
            .chain(inputs.norm_weight.values)
            .chain(inputs.dt_bias.values)
            .chain(inputs.a_log.values)
            .any(|value| !value.is_finite())
        {
            return Err(OrtError::new(
                "tritium-onnx: DeltaNet preserved parameters must be finite",
            ));
        }

        let mut next_conv = inputs.conv_state.values.to_vec();
        let mut next_recurrent = inputs.recurrent_state.values.to_vec();
        let mut convolved = vec![0.0; inputs.raw_qkv.values.len()];
        for token in 0..tokens {
            for channel in 0..conv_width {
                let state_start = channel * conv_kernel_dim;
                let state = &mut next_conv[state_start..state_start + conv_kernel_dim];
                state.copy_within(1..conv_kernel_dim, 0);
                state[conv_kernel_dim - 1] = inputs.raw_qkv.values[token * conv_width + channel];
                let weight = &inputs.conv_weight.values[state_start..state_start + conv_kernel_dim];
                let sum = state
                    .iter()
                    .zip(weight)
                    .map(|(value, weight)| value * weight)
                    .sum::<f32>();
                convolved[token * conv_width + channel] = deltanet_silu(sum);
            }
        }

        let mut normalized = vec![0.0; tokens * value_width];
        let mut core = vec![0.0; value_width];
        let group_size = num_value_heads / num_key_heads;
        let query_scale = 1.0 / (key_head_dim as f32).sqrt();
        for token in 0..tokens {
            let qkv = &convolved[token * conv_width..(token + 1) * conv_width];
            let query = &qkv[..key_width];
            let key = &qkv[key_width..2 * key_width];
            let value = &qkv[2 * key_width..];
            let gate_base = token * num_value_heads;
            let value_base = token * value_width;
            for value_head in 0..num_value_heads {
                let key_head = value_head / group_size;
                let q = &query[key_head * key_head_dim..(key_head + 1) * key_head_dim];
                let k = &key[key_head * key_head_dim..(key_head + 1) * key_head_dim];
                let v = &value[value_head * value_head_dim..(value_head + 1) * value_head_dim];
                let q_inverse = l2_inverse(q) * query_scale;
                let k_inverse = l2_inverse(k);
                let beta = deltanet_sigmoid(inputs.beta_logits.values[gate_base + value_head]);
                let g = -inputs.a_log.values[value_head].exp()
                    * deltanet_softplus(
                        inputs.decay_logits.values[gate_base + value_head]
                            + inputs.dt_bias.values[value_head],
                    );
                let decay = g.exp();
                let state_base = value_head * key_head_dim * value_head_dim;
                for state in
                    &mut next_recurrent[state_base..state_base + key_head_dim * value_head_dim]
                {
                    *state *= decay;
                }
                for value_lane in 0..value_head_dim {
                    let mut memory = 0.0;
                    for key_lane in 0..key_head_dim {
                        memory += k[key_lane]
                            * k_inverse
                            * next_recurrent[state_base + key_lane * value_head_dim + value_lane];
                    }
                    let delta = beta * (v[value_lane] - memory);
                    for key_lane in 0..key_head_dim {
                        next_recurrent[state_base + key_lane * value_head_dim + value_lane] +=
                            k[key_lane] * k_inverse * delta;
                    }
                    let mut mixed = 0.0;
                    for key_lane in 0..key_head_dim {
                        mixed += q[key_lane]
                            * q_inverse
                            * next_recurrent[state_base + key_lane * value_head_dim + value_lane];
                    }
                    core[value_head * value_head_dim + value_lane] = mixed;
                }
            }
            for value_head in 0..num_value_heads {
                let row_start = value_head * value_head_dim;
                let variance = core[row_start..row_start + value_head_dim]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    / value_head_dim as f32;
                let inverse_rms = 1.0 / (variance + inputs.epsilon).sqrt();
                for (value_lane, &weight) in inputs.norm_weight.values.iter().enumerate() {
                    let source = row_start + value_lane;
                    let destination = value_base + source;
                    normalized[destination] = core[source]
                        * inverse_rms
                        * weight
                        * deltanet_silu(inputs.z.values[destination]);
                }
            }
        }
        Ok(QwenDeltaNetOutput {
            normalized_core: normalized,
            conv_state: next_conv,
            recurrent_state: next_recurrent,
        })
    }
}

impl Kernel for TritiumQwenDeltaNetKernel {
    fn compute(&mut self, ctx: &KernelContext) -> ort::Result<()> {
        let raw_qkv_value = required_input(ctx, QwenDeltaNetInput::RawQkv)?;
        let z_value = required_input(ctx, QwenDeltaNetInput::Z)?;
        let beta_value = required_input(ctx, QwenDeltaNetInput::BetaLogits)?;
        let decay_value = required_input(ctx, QwenDeltaNetInput::DecayLogits)?;
        let conv_weight_value = required_input(ctx, QwenDeltaNetInput::ConvWeight)?;
        let norm_weight_value = required_input(ctx, QwenDeltaNetInput::NormWeight)?;
        let dt_bias_value = required_input(ctx, QwenDeltaNetInput::DtBias)?;
        let a_log_value = required_input(ctx, QwenDeltaNetInput::ALog)?;
        let conv_state_value = required_input(ctx, QwenDeltaNetInput::ConvState)?;
        let recurrent_state_value = required_input(ctx, QwenDeltaNetInput::RecurrentState)?;
        let epsilon_value = required_input(ctx, QwenDeltaNetInput::Epsilon)?;

        let (raw_qkv_shape, raw_qkv) = raw_qkv_value.try_extract_tensor::<f32>()?;
        let (z_shape, z) = z_value.try_extract_tensor::<f32>()?;
        let (beta_shape, beta) = beta_value.try_extract_tensor::<f32>()?;
        let (decay_shape, decay) = decay_value.try_extract_tensor::<f32>()?;
        let (conv_weight_shape, conv_weight) = conv_weight_value.try_extract_tensor::<f32>()?;
        let (norm_weight_shape, norm_weight) = norm_weight_value.try_extract_tensor::<f32>()?;
        let (dt_bias_shape, dt_bias) = dt_bias_value.try_extract_tensor::<f32>()?;
        let (a_log_shape, a_log) = a_log_value.try_extract_tensor::<f32>()?;
        let (conv_state_shape, conv_state) = conv_state_value.try_extract_tensor::<f32>()?;
        let (recurrent_state_shape, recurrent_state) =
            recurrent_state_value.try_extract_tensor::<f32>()?;
        let (epsilon_shape, epsilon) = epsilon_value.try_extract_tensor::<f32>()?;
        if !epsilon_shape.is_empty() || epsilon.len() != 1 {
            return Err(OrtError::new(format!(
                "tritium-onnx: DeltaNet epsilon must be scalar, got shape {:?}",
                epsilon_shape.as_ref()
            )));
        }
        let raw_qkv_dimensions = raw_qkv_shape.to_vec();
        let z_dimensions = z_shape.to_vec();
        let beta_dimensions = beta_shape.to_vec();
        let decay_dimensions = decay_shape.to_vec();
        let conv_weight_dimensions = conv_weight_shape.to_vec();
        let norm_weight_dimensions = norm_weight_shape.to_vec();
        let dt_bias_dimensions = dt_bias_shape.to_vec();
        let a_log_dimensions = a_log_shape.to_vec();
        let conv_state_dimensions = conv_state_shape.to_vec();
        let recurrent_state_dimensions = recurrent_state_shape.to_vec();
        let output = self.run(QwenDeltaNetInputs {
            raw_qkv: borrowed_tensor(&raw_qkv_dimensions, raw_qkv),
            z: borrowed_tensor(&z_dimensions, z),
            beta_logits: borrowed_tensor(&beta_dimensions, beta),
            decay_logits: borrowed_tensor(&decay_dimensions, decay),
            conv_weight: borrowed_tensor(&conv_weight_dimensions, conv_weight),
            norm_weight: borrowed_tensor(&norm_weight_dimensions, norm_weight),
            dt_bias: borrowed_tensor(&dt_bias_dimensions, dt_bias),
            a_log: borrowed_tensor(&a_log_dimensions, a_log),
            conv_state: borrowed_tensor(&conv_state_dimensions, conv_state),
            recurrent_state: borrowed_tensor(&recurrent_state_dimensions, recurrent_state),
            epsilon: epsilon[0],
        })?;
        let dimensions = self.geometry.dimensions().map_err(delta_geometry_error)?;
        let mut core_value = ctx
            .output(
                QwenDeltaNetOutputSlot::NormalizedCore as usize,
                z_dimensions,
            )?
            .ok_or_else(|| missing_output(QwenDeltaNetOutputSlot::NormalizedCore))?;
        core_value
            .try_extract_tensor_mut::<f32>()?
            .1
            .copy_from_slice(&output.normalized_core);
        let mut conv_value = ctx
            .output(
                QwenDeltaNetOutputSlot::ConvState as usize,
                vec![
                    dimensions.conv_width() as i64,
                    self.geometry.conv_kernel_dim() as i64,
                ],
            )?
            .ok_or_else(|| missing_output(QwenDeltaNetOutputSlot::ConvState))?;
        conv_value
            .try_extract_tensor_mut::<f32>()?
            .1
            .copy_from_slice(&output.conv_state);
        let mut recurrent_value = ctx
            .output(
                QwenDeltaNetOutputSlot::RecurrentState as usize,
                vec![
                    self.geometry.num_value_heads() as i64,
                    self.geometry.key_head_dim() as i64,
                    self.geometry.value_head_dim() as i64,
                ],
            )?
            .ok_or_else(|| missing_output(QwenDeltaNetOutputSlot::RecurrentState))?;
        recurrent_value
            .try_extract_tensor_mut::<f32>()?
            .1
            .copy_from_slice(&output.recurrent_state);
        Ok(())
    }
}

fn required_input<'a>(
    ctx: &'a KernelContext,
    slot: QwenDeltaNetInput,
) -> ort::Result<ort::value::ValueRef<'a>> {
    ctx.input(slot as usize)?.ok_or_else(|| {
        OrtError::new(format!(
            "tritium-onnx: missing input {} ({})",
            slot as usize,
            slot.name()
        ))
    })
}

fn missing_output(slot: QwenDeltaNetOutputSlot) -> OrtError {
    OrtError::new(format!(
        "tritium-onnx: missing output {} ({})",
        slot as usize,
        slot.name()
    ))
}

fn borrowed_tensor<'a>(shape: &'a [i64], values: &'a [f32]) -> QwenDeltaNetTensor<'a> {
    QwenDeltaNetTensor { shape, values }
}

fn matrix_tokens(shape: &[i64], width: usize, name: &str) -> Result<usize, OrtError> {
    if shape.len() != 2 || usize::try_from(shape[1]) != Ok(width) {
        return Err(OrtError::new(format!(
            "tritium-onnx: DeltaNet {name} shape {shape:?} must be [tokens, {width}]"
        )));
    }
    usize::try_from(shape[0]).map_err(|_| {
        OrtError::new(format!(
            "tritium-onnx: DeltaNet {name} token count must be nonnegative"
        ))
    })
}

fn require_matrix(
    matrix: QwenDeltaNetTensor<'_>,
    tokens: usize,
    width: usize,
    name: &str,
) -> Result<(), OrtError> {
    if matrix.shape != [tokens as i64, width as i64] {
        return Err(OrtError::new(format!(
            "tritium-onnx: DeltaNet {name} shape {:?} must be [{tokens}, {width}]",
            matrix.shape
        )));
    }
    require_len(matrix.values, tokens, width, name)
}

fn require_shaped(
    matrix: QwenDeltaNetTensor<'_>,
    expected_shape: &[usize],
    name: &str,
) -> Result<(), OrtError> {
    let exact_shape = matrix.shape.len() == expected_shape.len()
        && matrix
            .shape
            .iter()
            .zip(expected_shape)
            .all(|(&actual, &expected)| usize::try_from(actual) == Ok(expected));
    if !exact_shape {
        return Err(OrtError::new(format!(
            "tritium-onnx: DeltaNet {name} shape {:?} must be {expected_shape:?}",
            matrix.shape
        )));
    }
    let expected_len = expected_shape
        .iter()
        .try_fold(1_usize, |size, &dimension| {
            size.checked_mul(dimension)
                .ok_or_else(|| OrtError::new("tritium-onnx: DeltaNet input length overflow"))
        })?;
    require_exact_len(matrix.values, expected_len, name)
}

fn require_len(values: &[f32], rows: usize, columns: usize, name: &str) -> Result<(), OrtError> {
    let expected = rows
        .checked_mul(columns)
        .ok_or_else(|| OrtError::new("tritium-onnx: DeltaNet input length overflow"))?;
    require_exact_len(values, expected, name)
}

fn require_exact_len(values: &[f32], expected: usize, name: &str) -> Result<(), OrtError> {
    if values.len() == expected {
        Ok(())
    } else {
        Err(OrtError::new(format!(
            "tritium-onnx: DeltaNet {name} has {} elements, expected {expected}",
            values.len()
        )))
    }
}

fn l2_inverse(values: &[f32]) -> f32 {
    1.0 / (values.iter().map(|value| value * value).sum::<f32>() + 1.0e-6).sqrt()
}

fn deltanet_sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn deltanet_silu(value: f32) -> f32 {
    value * deltanet_sigmoid(value)
}

fn deltanet_softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

/// The per-node kernel produced by [`TritiumTernaryMpGemmOp::create_kernel`],
/// carrying the node's resolved `K` and packing `format`. Its
/// [`compute`](Kernel::compute) extracts the three input tensors and runs the
/// always-on [`ternary_mpgemm_kernel`].
#[derive(Debug, Clone, Copy)]
pub struct TritiumTernaryMpGemmKernel {
    k: usize,
    format: TernaryFormat,
}

impl TritiumTernaryMpGemmKernel {
    /// Run the kernel over already-extracted operands, returning the flat
    /// `[M, N]` output. Split out from [`Kernel::compute`] so it can be unit
    /// tested without an ONNX session / the native runtime.
    ///
    /// `act_shape` is `act`'s tensor shape (its first dim is `M`).
    ///
    /// # Errors
    /// An [`ort::Error`] if the activation is not 2-D, or the Layer-1 kernel
    /// rejects the operands (bad packed length, shape mismatch, ...).
    pub fn run(
        &self,
        act_shape: &[i64],
        act: &[f32],
        packed: &[u8],
        scales: &[f32],
    ) -> Result<Vec<f32>, OrtError> {
        if act_shape.len() != 2 {
            return Err(OrtError::new(format!(
                "tritium-onnx: activation must be 2-D [M, K], got shape {act_shape:?}"
            )));
        }
        let m = usize::try_from(act_shape[0]).map_err(|_| {
            OrtError::new(format!(
                "tritium-onnx: activation M must be nonnegative, got {}",
                act_shape[0]
            ))
        })?;
        if usize::try_from(act_shape[1]) != Ok(self.k) {
            return Err(OrtError::new(format!(
                "tritium-onnx: activation K {} does not match attribute K {}",
                act_shape[1], self.k
            )));
        }
        ternary_mpgemm_kernel(act, packed, scales, m, self.k, self.format)
            .map_err(|e| OrtError::new(e.to_string()))
    }
}

impl Kernel for TritiumTernaryMpGemmKernel {
    fn compute(&mut self, ctx: &KernelContext) -> ort::Result<()> {
        let act_v = ctx
            .input(0)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing input 0 (act)"))?;
        let packed_v = ctx
            .input(1)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing input 1 (packed)"))?;
        let scales_v = ctx
            .input(2)?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing input 2 (scales)"))?;

        let (act_shape, act) = act_v.try_extract_tensor::<f32>()?;
        let (_, packed) = packed_v.try_extract_tensor::<u8>()?;
        let (_, scales) = scales_v.try_extract_tensor::<f32>()?;

        let act_dims: Vec<i64> = act_shape.iter().copied().collect();
        let out = self.run(&act_dims, act, packed, scales)?;

        let m = if act_dims.is_empty() { 0 } else { act_dims[0] };
        let n = scales.len() as i64;
        let mut out_v = ctx
            .output(0, vec![m, n])?
            .ok_or_else(|| OrtError::new("tritium-onnx: missing output 0"))?;
        let (_, out_ref) = out_v.try_extract_tensor_mut::<f32>()?;
        out_ref.copy_from_slice(&out);
        Ok(())
    }
}

/// Build an [`OperatorDomain`] named [`ONNX_DOMAIN`] with
/// opset-1 [`TritiumTernaryMpGemmOp`]/[`TritiumTernaryEmbeddingOp`] plus
/// opset-2 [`TritiumKvAttentionOp`] registered, ready to pass to
/// [`ort::session::builder::SessionBuilder::with_operators`].
///
/// # Errors
/// An [`ort::Error`] if the domain cannot be created or the operator cannot be
/// added (e.g. the native runtime failed to initialize).
pub fn tritium_operator_domain() -> ort::Result<OperatorDomain> {
    OperatorDomain::new(ONNX_DOMAIN)?
        .add(TritiumTernaryMpGemmOp)?
        .add(TritiumTernaryEmbeddingOp)?
        .add(TritiumSaltV2MpGemmOp)?
        .add(TritiumSaltV2EmbeddingOp)?
        .add(TritiumKvAttentionOp)?
        .add(TritiumQwenDeltaNetOp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tritium_core::Trit;
    use tritium_format::{
        num_blocks, pack_tq1_0_row, pack_tq2_0_row, salt_v2::SaltV2Codec,
        salt_v2_package::pack_salt_v2_plane,
    };
    use tritium_testkit::{ConformanceVector, FROZEN_COUNT, FROZEN_SEED, generate_vectors};

    fn block_bytes(format: TernaryFormat) -> usize {
        match format {
            TernaryFormat::Tq2_0 => tritium_format::TQ2_0_BLOCK_BYTES,
            TernaryFormat::Tq1_0 => tritium_format::TQ1_0_BLOCK_BYTES,
            other => panic!("cannot pack {other:?}"),
        }
    }

    fn format_tag(format: TernaryFormat) -> i64 {
        match format {
            TernaryFormat::Tq2_0 => 0,
            TernaryFormat::Tq1_0 => 1,
            other => panic!("unexpected format {other:?}"),
        }
    }

    fn pack(v: &ConformanceVector, format: TernaryFormat) -> Vec<u8> {
        let nb = num_blocks(v.k);
        let unit = vec![f16::ONE; nb];
        let row_bytes = nb * block_bytes(format);
        let mut packed = vec![0u8; v.n * row_bytes];
        for ni in 0..v.n {
            let trits: Vec<Trit> = v.weights[ni * v.k..ni * v.k + v.k]
                .iter()
                .map(|&w| Trit::from_i8(w).unwrap())
                .collect();
            let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
            match format {
                TernaryFormat::Tq2_0 => pack_tq2_0_row(&trits, &unit, out).unwrap(),
                TernaryFormat::Tq1_0 => pack_tq1_0_row(&trits, &unit, out).unwrap(),
                other => panic!("cannot pack {other:?}"),
            };
        }
        packed
    }

    fn pack_rows(rows: &[Vec<Trit>], format: TernaryFormat) -> Vec<u8> {
        let k = rows.first().map_or(0, Vec::len);
        let nb = num_blocks(k);
        let unit = vec![f16::ONE; nb];
        let row_bytes = nb * block_bytes(format);
        let mut packed = vec![0u8; rows.len() * row_bytes];
        for (row, output) in rows.iter().zip(packed.chunks_exact_mut(row_bytes)) {
            match format {
                TernaryFormat::Tq2_0 => pack_tq2_0_row(row, &unit, output).unwrap(),
                TernaryFormat::Tq1_0 => pack_tq1_0_row(row, &unit, output).unwrap(),
                other => panic!("cannot pack {other:?}"),
            }
        }
        packed
    }

    fn packed_matrix<'a>(
        rows: usize,
        columns: usize,
        packed: &'a [u8],
        scales: &'a [f32],
        format: TernaryFormat,
    ) -> crate::PackedTernaryMatrix<'a> {
        crate::PackedTernaryMatrix {
            rows,
            columns,
            packed,
            scales,
            format,
        }
    }

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            loop {
                let path = std::env::temp_dir().join(format!(
                    "tritium-onnx-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match std::fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => panic!("create ONNX test directory: {error}"),
                }
            }
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn format_attr_maps_codes() {
        assert_eq!(format_from_attr(0).unwrap(), TernaryFormat::Tq2_0);
        assert_eq!(format_from_attr(1).unwrap(), TernaryFormat::Tq1_0);
        assert!(format_from_attr(2).is_err());
        assert!(format_from_attr(-1).is_err());
    }

    #[test]
    fn operator_describes_three_inputs_one_output() {
        let op = TritiumTernaryMpGemmOp;
        assert_eq!(op.name(), ONNX_OP_NAME);
        assert_eq!(op.inputs().len(), 3, "act, packed, scales");
        assert_eq!(op.outputs().len(), 1, "out");
        let embedding = TritiumTernaryEmbeddingOp;
        assert_eq!(embedding.name(), ONNX_EMBEDDING_OP_NAME);
        assert_eq!(embedding.inputs().len(), 3, "tokens, packed, scales");
        assert_eq!(embedding.outputs().len(), 1, "embedding");
        let salt_v2 = TritiumSaltV2MpGemmOp;
        assert_eq!(salt_v2.name(), ONNX_SALT_V2_OP_NAME);
        assert_eq!(salt_v2.inputs().len(), 5);
        assert_eq!(salt_v2.outputs().len(), 1);
        let salt_embedding = TritiumSaltV2EmbeddingOp;
        assert_eq!(salt_embedding.name(), ONNX_SALT_V2_EMBEDDING_OP_NAME);
        assert_eq!(salt_embedding.inputs().len(), 5);
        assert_eq!(salt_embedding.outputs().len(), 1);
        let attention = TritiumKvAttentionOp;
        assert_eq!(attention.name(), ONNX_KV_ATTENTION_OP_NAME);
        assert_eq!(attention.inputs().len(), 3, "q, k_cache, v_cache");
        assert_eq!(attention.outputs().len(), 1, "context");
    }

    #[test]
    fn salt_v2_custom_op_executes_additive_packed_operand_in_real_ort() {
        use ort::value::Tensor;

        let columns = 8;
        let rows = 1;
        let first = [
            Trit::NEG,
            Trit::ZERO,
            Trit::POS,
            Trit::NEG,
            Trit::ZERO,
            Trit::POS,
            Trit::NEG,
            Trit::POS,
        ];
        let second = [
            Trit::POS,
            Trit::POS,
            Trit::ZERO,
            Trit::NEG,
            Trit::NEG,
            Trit::ZERO,
            Trit::POS,
            Trit::ZERO,
        ];
        let mut payload = pack_salt_v2_plane(SaltV2Codec::B3, &first).unwrap();
        payload.extend(pack_salt_v2_plane(SaltV2Codec::B3, &second).unwrap());
        let scales = [f16::from_f32(0.5), f16::from_f32(0.25)];
        let activation = vec![1.0_f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let matrix = crate::SaltV2PackedMatrix {
            rows,
            columns,
            codec: SaltV2Codec::B3,
            payload: &payload,
            scales: &scales,
            allocation_map: &[],
            rank_prefixes: &[],
            terminal_map_value: 1,
        };
        let expected = crate::salt_v2_mpgemm_kernel(&activation, 1, matrix).unwrap();
        let encoded = crate::model::encode_salt_v2_mpgemm_test_graph(matrix);
        let diagnostics = crate::diagnose_unsupported_graph(&encoded).unwrap();
        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        let mut session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&encoded)
            .unwrap();
        let act = Tensor::from_array(([1, columns], activation.clone())).unwrap();
        {
            let outputs = session.run(ort::inputs![&act]).unwrap();
            let (_, actual) = outputs[0].try_extract_tensor::<f32>().unwrap();
            assert_eq!(actual, expected);
        }

        let two_rows = [activation.as_slice(), activation.as_slice()].concat();
        let expected_two = crate::salt_v2_mpgemm_kernel(&two_rows, 2, matrix).unwrap();
        let act = Tensor::from_array(([2, columns], two_rows)).unwrap();
        let outputs = session.run(ort::inputs![&act]).unwrap();
        let (_, actual) = outputs[0].try_extract_tensor::<f32>().unwrap();
        assert_eq!(actual, expected_two);
    }

    #[test]
    fn salt_v2_embedding_custom_op_gathers_selected_rows_in_real_ort() {
        use ort::value::Tensor;

        let rows = 2;
        let columns = 8;
        let trits: Vec<Trit> = (0..rows * columns)
            .map(|index| match index % 3 {
                0 => Trit::NEG,
                1 => Trit::ZERO,
                _ => Trit::POS,
            })
            .collect();
        let payload = pack_salt_v2_plane(SaltV2Codec::B3, &trits).unwrap();
        let scales = [f16::from_f32(0.5)];
        let matrix = crate::SaltV2PackedMatrix {
            rows,
            columns,
            codec: SaltV2Codec::B3,
            payload: &payload,
            scales: &scales,
            allocation_map: &[],
            rank_prefixes: &[],
            terminal_map_value: 0,
        };
        let tokens = [1_i64, 0, 1];
        let expected = crate::salt_v2_embedding_kernel(&tokens, &[1, 3], matrix).unwrap();
        let encoded = crate::model::encode_salt_v2_embedding_test_graph(matrix, &[1, 3]);
        let diagnostics = crate::diagnose_unsupported_graph(&encoded).unwrap();
        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        let mut session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&encoded)
            .unwrap();
        let tokens = Tensor::from_array(([1, 3], tokens.to_vec())).unwrap();
        let outputs = session.run(ort::inputs![&tokens]).unwrap();
        let (shape, actual) = outputs[0].try_extract_tensor::<f32>().unwrap();
        assert_eq!(shape.as_ref(), &[1, 3, columns as i64]);
        assert_eq!(actual, expected);
    }

    #[test]
    fn kv_attention_kernel_runs_prompt_and_cached_decode() {
        let prompt = TritiumKvAttentionKernel {
            n_head: 1,
            n_kv_head: 1,
            head_dim: 1,
            past_tokens: Some(0),
        };
        assert_eq!(
            prompt
                .run(
                    &[2, 1, 1],
                    &[1.0, 2.0],
                    &[2, 1, 1],
                    &[1.0, 1.0],
                    &[2, 1, 1],
                    &[10.0, 20.0],
                )
                .unwrap(),
            vec![10.0, 15.0]
        );
        let decode = TritiumKvAttentionKernel {
            past_tokens: Some(2),
            ..prompt
        };
        let output = decode
            .run(
                &[1, 1, 1],
                &[3.0],
                &[3, 1, 1],
                &[1.0, 1.0, 1.0],
                &[3, 1, 1],
                &[10.0, 20.0, 40.0],
            )
            .unwrap();
        assert!((output[0] - 70.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn qwen_deltanet_kernel_matches_frozen_recurrent_oracle() {
        let kernel = super::TritiumQwenDeltaNetKernel::new(
            super::QwenDeltaNetGeometry::new(2, 1, 1, 1, 1).unwrap(),
        );
        let raw_qkv = [0.2, -0.3, 0.4];
        let z = [0.5];
        let beta = [0.1];
        let decay = [-0.2];
        let conv_weight = [0.2, 0.8, -0.1, 0.7, 0.3, 0.9];
        let norm_weight = [1.1];
        let dt_bias = [0.2];
        let a_log = [-0.3];
        let conv_state = [0.0; 6];
        let recurrent_state = [0.0];
        let inputs = super::QwenDeltaNetInputs {
            raw_qkv: super::borrowed_tensor(&[1, 3], &raw_qkv),
            z: super::borrowed_tensor(&[1, 1], &z),
            beta_logits: super::borrowed_tensor(&[1, 1], &beta),
            decay_logits: super::borrowed_tensor(&[1, 1], &decay),
            conv_weight: super::borrowed_tensor(&[3, 2], &conv_weight),
            norm_weight: super::borrowed_tensor(&[1], &norm_weight),
            dt_bias: super::borrowed_tensor(&[1], &dt_bias),
            a_log: super::borrowed_tensor(&[1], &a_log),
            conv_state: super::borrowed_tensor(&[3, 2], &conv_state),
            recurrent_state: super::borrowed_tensor(&[1, 1, 1], &recurrent_state),
            epsilon: 1.0e-6,
        };
        let output = kernel.run(inputs).unwrap();
        assert_f32_close(&output.normalized_core, &[-0.342_338_83], 1.0e-6);
        assert_f32_close(&output.conv_state, &[0.0, 0.2, 0.0, -0.3, 0.0, 0.4], 0.0);
        assert_f32_close(&output.recurrent_state, &[-0.111_317_93], 1.0e-6);

        let mut wrong_shape = inputs;
        wrong_shape.conv_weight.shape = &[2, 3];
        assert!(kernel.run(wrong_shape).is_err());
        let mut zero_tokens = inputs;
        zero_tokens.raw_qkv = super::borrowed_tensor(&[0, 3], &[]);
        assert!(kernel.run(zero_tokens).is_err());
    }

    #[test]
    fn qwen_deltanet_custom_op_executes_frozen_recurrent_oracle() {
        use ort::value::Tensor;

        let model = crate::model::encode_qwen_deltanet_test_graph();
        let diagnostics = crate::diagnose_unsupported_graph(&model).unwrap();
        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        let mut session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&model)
            .unwrap();
        let raw_qkv = Tensor::from_array(([1, 3], vec![0.2_f32, -0.3, 0.4])).unwrap();
        let z = Tensor::from_array(([1, 1], vec![0.5_f32])).unwrap();
        let beta = Tensor::from_array(([1, 1], vec![0.1_f32])).unwrap();
        let decay = Tensor::from_array(([1, 1], vec![-0.2_f32])).unwrap();
        let conv_weight =
            Tensor::from_array(([3, 2], vec![0.2_f32, 0.8, -0.1, 0.7, 0.3, 0.9])).unwrap();
        let norm_weight = Tensor::from_array(([1], vec![1.1_f32])).unwrap();
        let dt_bias = Tensor::from_array(([1], vec![0.2_f32])).unwrap();
        let a_log = Tensor::from_array(([1], vec![-0.3_f32])).unwrap();
        let conv_state = Tensor::from_array(([3, 2], vec![0.0_f32; 6])).unwrap();
        let recurrent_state = Tensor::from_array(([1, 1, 1], vec![0.0_f32])).unwrap();
        let epsilon = Tensor::from_array((Vec::<usize>::new(), vec![1.0e-6_f32])).unwrap();
        let outputs = session
            .run(ort::inputs![
                &raw_qkv,
                &z,
                &beta,
                &decay,
                &conv_weight,
                &norm_weight,
                &dt_bias,
                &a_log,
                &conv_state,
                &recurrent_state,
                &epsilon,
            ])
            .unwrap();
        let (_, core) = outputs[0].try_extract_tensor::<f32>().unwrap();
        let (_, next_conv) = outputs[1].try_extract_tensor::<f32>().unwrap();
        let (_, next_recurrent) = outputs[2].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(core, &[-0.342_338_83], 1.0e-6);
        assert_f32_close(next_conv, &[0.0, 0.2, 0.0, -0.3, 0.0, 0.4], 0.0);
        assert_f32_close(next_recurrent, &[-0.111_317_93], 1.0e-6);
        let next_conv = next_conv.to_vec();
        let next_recurrent = next_recurrent.to_vec();
        drop(outputs);

        let decode_raw_qkv = Tensor::from_array(([1, 3], vec![-0.1_f32, 0.25, -0.2])).unwrap();
        let decode_z = Tensor::from_array(([1, 1], vec![-0.3_f32])).unwrap();
        let decode_beta = Tensor::from_array(([1, 1], vec![-0.4_f32])).unwrap();
        let decode_decay = Tensor::from_array(([1, 1], vec![0.6_f32])).unwrap();
        let decode_conv_state = Tensor::from_array(([3, 2], next_conv)).unwrap();
        let decode_recurrent_state = Tensor::from_array(([1, 1, 1], next_recurrent)).unwrap();
        let decode_outputs = session
            .run(ort::inputs![
                &decode_raw_qkv,
                &decode_z,
                &decode_beta,
                &decode_decay,
                &conv_weight,
                &norm_weight,
                &dt_bias,
                &a_log,
                &decode_conv_state,
                &decode_recurrent_state,
                &epsilon,
            ])
            .unwrap();
        let (_, decode_core) = decode_outputs[0].try_extract_tensor::<f32>().unwrap();
        let (_, decode_conv) = decode_outputs[1].try_extract_tensor::<f32>().unwrap();
        let (_, decode_recurrent) = decode_outputs[2].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(decode_core, &[-0.140_389_25], 1.0e-6);
        assert_f32_close(decode_conv, &[0.2, -0.1, -0.3, 0.25, 0.4, -0.2], 0.0);
        assert_f32_close(decode_recurrent, &[-0.039_668_053], 1.0e-6);
    }

    #[test]
    fn qwen_deltanet_whole_layer_executes_packed_projections_and_state() {
        use ort::value::Tensor;

        let format = TernaryFormat::Tq2_0;
        let hidden = 256;
        let basis = |sign: i8, columns: usize| {
            let mut row = vec![Trit::ZERO; columns];
            row[0] = Trit::from_i8(sign).unwrap();
            row
        };
        let zero = |columns: usize| vec![Trit::ZERO; columns];
        let qkv_packed = pack_rows(
            &[basis(1, hidden), basis(-1, hidden), basis(1, hidden)],
            format,
        );
        let z_packed = pack_rows(&[basis(1, hidden)], format);
        let beta_packed = pack_rows(&[basis(1, hidden)], format);
        let decay_packed = pack_rows(&[basis(-1, hidden)], format);
        let mut output_rows = vec![zero(1); hidden];
        output_rows[0][0] = Trit::from_i8(1).unwrap();
        let output_packed = pack_rows(&output_rows, format);
        let gate_packed = pack_rows(&[basis(1, hidden)], format);
        let up_packed = pack_rows(&[basis(1, hidden)], format);
        let mut down_rows = vec![zero(1); hidden];
        down_rows[0][0] = Trit::from_i8(1).unwrap();
        let down_packed = pack_rows(&down_rows, format);
        let qkv_scales = [0.2, 0.3, 0.4];
        let gate_scales = [0.5];
        let up_scales = [0.25];
        let z_scales = [0.5];
        let beta_scales = [0.1];
        let decay_scales = [0.2];
        let output_scales = {
            let mut scales = vec![0.0; hidden];
            scales[0] = 2.0;
            scales
        };
        let down_scales = {
            let mut scales = vec![0.0; hidden];
            scales[0] = 3.0;
            scales
        };
        let norm_offset = (1.0_f32 + 1.0e-6).sqrt() - 1.0;
        let attention_norm = vec![norm_offset; hidden];
        let ffn_norm = vec![0.25; hidden];
        let identity = crate::OnnxArtifactIdentityV2 {
            source_model_id: "qwen-source",
            tokenizer_id: "qwen-tokenizer",
            recipe_id: "qwen-recipe",
            tritium_build_id: "qwen-build",
            package_id: "qwen-package",
            converted_coverage_id: "qwen-converted",
            deferred_coverage_id: "qwen-deferred",
        };
        let matrix = |rows, columns, packed, scales| crate::PackedTernaryMatrix {
            rows,
            columns,
            packed,
            scales,
            format,
        };
        let model = crate::QwenDeltaNetLayerModel {
            tokens: 1,
            hidden,
            rms_epsilon: 1.0e-6,
            geometry: crate::QwenDeltaNetGeometry::new(2, 1, 1, 1, 1).unwrap(),
            layer: crate::QwenDeltaNetDecoderLayer {
                attention_norm: &attention_norm,
                qkv: matrix(3, hidden, &qkv_packed, &qkv_scales),
                z: matrix(1, hidden, &z_packed, &z_scales),
                beta: matrix(1, hidden, &beta_packed, &beta_scales),
                decay: matrix(1, hidden, &decay_packed, &decay_scales),
                conv_weight: &[0.2, 0.8, -0.1, 0.7, 0.3, 0.9],
                norm_weight: &[1.1],
                dt_bias: &[0.2],
                a_log: &[-0.3],
                output: matrix(256, 1, &output_packed, &output_scales),
                ffn_norm: &ffn_norm,
                gate: matrix(1, hidden, &gate_packed, &gate_scales),
                up: matrix(1, hidden, &up_packed, &up_scales),
                down: matrix(256, 1, &down_packed, &down_scales),
            },
            identity,
        };
        let encoded = crate::encode_qwen_deltanet_layer(model).unwrap();
        assert!(
            crate::diagnose_unsupported_graph(&encoded)
                .unwrap()
                .is_empty()
        );
        let mut session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&encoded)
            .unwrap();
        let hidden_input = Tensor::from_array(([1, hidden], vec![1.0_f32; hidden])).unwrap();
        let conv_state = Tensor::from_array(([3, 2], vec![0.0_f32; 6])).unwrap();
        let recurrent_state = Tensor::from_array(([1, 1, 1], vec![0.0_f32])).unwrap();
        let outputs = session
            .run(ort::inputs![&hidden_input, &conv_state, &recurrent_state])
            .unwrap();
        let (_, next_hidden) = outputs[0].try_extract_tensor::<f32>().unwrap();
        let (_, next_conv) = outputs[1].try_extract_tensor::<f32>().unwrap();
        let (_, next_recurrent) = outputs[2].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(&next_hidden[..1], &[0.347_430_7], 3.0e-6);
        assert_f32_close(&next_hidden[1..], &vec![1.0; hidden - 1], 2.0e-6);
        assert_f32_close(next_conv, &[0.0, 0.2, 0.0, -0.3, 0.0, 0.4], 1.0e-6);
        assert_f32_close(next_recurrent, &[-0.111_317_93], 2.0e-6);
    }

    #[test]
    fn qwen_heterogeneous_schedule_runs_prompt_and_cached_decode() {
        use ort::value::Tensor;

        let format = TernaryFormat::Tq2_0;
        let hidden = 256;
        let basis = |sign: i8, columns: usize| {
            let mut row = vec![Trit::ZERO; columns];
            row[0] = Trit::from_i8(sign).unwrap();
            row
        };
        let zero = |columns: usize| vec![Trit::ZERO; columns];
        let matrix = |rows, columns, packed, scales| crate::PackedTernaryMatrix {
            rows,
            columns,
            packed,
            scales,
            format,
        };

        let embedding_packed = pack_rows(&[basis(1, hidden), basis(-1, hidden)], format);
        let embedding_scales = [1.0, 1.0];
        let qkv_packed = pack_rows(
            &[basis(1, hidden), basis(-1, hidden), basis(1, hidden)],
            format,
        );
        let qkv_scales = [0.2, 0.3, 0.4];
        let z_packed = pack_rows(&[basis(1, hidden)], format);
        let z_scales = [0.5];
        let beta_packed = pack_rows(&[basis(1, hidden)], format);
        let beta_scales = [0.1];
        let decay_packed = pack_rows(&[basis(-1, hidden)], format);
        let decay_scales = [0.2];
        let delta_output_packed = pack_rows(&vec![zero(1); hidden], format);
        let hidden_zero_scales = vec![0.0; hidden];
        let ffn_gate_packed = pack_rows(&[zero(hidden)], format);
        let ffn_up_packed = pack_rows(&[zero(hidden)], format);
        let ffn_down_packed = pack_rows(&vec![zero(1); hidden], format);
        let one_scale = [1.0];
        let input_rms = (1.0_f32 / hidden as f32 + 1.0e-6).sqrt();
        let input_norm = vec![input_rms - 1.0; hidden];
        let zero_norm = vec![0.0; hidden];
        let delta = crate::QwenDeltaNetDecoderLayer {
            attention_norm: &input_norm,
            qkv: matrix(3, hidden, &qkv_packed, &qkv_scales),
            z: matrix(1, hidden, &z_packed, &z_scales),
            beta: matrix(1, hidden, &beta_packed, &beta_scales),
            decay: matrix(1, hidden, &decay_packed, &decay_scales),
            conv_weight: &[0.2, 0.8, -0.1, 0.7, 0.3, 0.9],
            norm_weight: &[1.1],
            dt_bias: &[0.2],
            a_log: &[-0.3],
            output: matrix(hidden, 1, &delta_output_packed, &hidden_zero_scales),
            ffn_norm: &zero_norm,
            gate: matrix(1, hidden, &ffn_gate_packed, &one_scale),
            up: matrix(1, hidden, &ffn_up_packed, &one_scale),
            down: matrix(hidden, 1, &ffn_down_packed, &hidden_zero_scales),
        };

        let mut fused_rows = vec![zero(hidden); hidden * 2];
        fused_rows[0][0] = Trit::from_i8(1).unwrap();
        fused_rows[hidden][0] = Trit::from_i8(1).unwrap();
        let fused_packed = pack_rows(&fused_rows, format);
        let mut fused_scales = vec![0.0; hidden * 2];
        fused_scales[0] = 1.0;
        fused_scales[hidden] = 20.0;
        let mut attention_rows = vec![zero(hidden); hidden];
        attention_rows[0][0] = Trit::from_i8(1).unwrap();
        let attention_packed = pack_rows(&attention_rows, format);
        let mut attention_scales = vec![0.0; hidden];
        attention_scales[0] = 1.0;
        let head_norm = vec![input_rms - 1.0; hidden];
        let full = crate::QwenFullAttentionDecoderLayer {
            attention_norm: &input_norm,
            query_norm: &head_norm,
            key_norm: &head_norm,
            fused_query_gate: matrix(hidden * 2, hidden, &fused_packed, &fused_scales),
            key: matrix(hidden, hidden, &attention_packed, &attention_scales),
            value: matrix(hidden, hidden, &attention_packed, &attention_scales),
            attention_output: matrix(hidden, hidden, &attention_packed, &attention_scales),
            ffn_norm: &zero_norm,
            gate: matrix(1, hidden, &ffn_gate_packed, &one_scale),
            up: matrix(1, hidden, &ffn_up_packed, &one_scale),
            down: matrix(hidden, 1, &ffn_down_packed, &hidden_zero_scales),
        };
        let layers = [
            crate::QwenCausalLmDecoderLayer::DeltaNet(delta),
            crate::QwenCausalLmDecoderLayer::FullAttention(full),
        ];
        let final_rms = (4.0_f32 / hidden as f32 + 1.0e-6).sqrt();
        let final_norm = vec![final_rms - 1.0; hidden];
        let identity = crate::OnnxArtifactIdentityV2 {
            source_model_id: "qwen-source",
            tokenizer_id: "qwen-tokenizer",
            recipe_id: "qwen-recipe",
            tritium_build_id: "qwen-build",
            package_id: "qwen-package",
            converted_coverage_id: "qwen-converted",
            deferred_coverage_id: "qwen-deferred",
        };
        let base = crate::QwenCausalLmModel {
            tokens: 1,
            past_tokens: 0,
            n_head: 1,
            n_kv_head: 1,
            head_dim: hidden,
            rotary: crate::RotaryEmbedding {
                theta: 10_000.0,
                dimensions: hidden,
            },
            rms_epsilon: 1.0e-6,
            delta_geometry: crate::QwenDeltaNetGeometry::new(2, 1, 1, 1, 1).unwrap(),
            embedding: matrix(2, hidden, &embedding_packed, &embedding_scales),
            lm_head: None,
            layers: &layers,
            final_norm: &final_norm,
            identity,
        };
        let prompt_model = crate::encode_qwen_causal_lm(base).unwrap();
        assert!(
            crate::diagnose_unsupported_graph(&prompt_model)
                .unwrap()
                .is_empty()
        );
        let mut prompt_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&prompt_model)
            .unwrap();
        let token = Tensor::from_array(([1], vec![0_i64])).unwrap();
        let conv = Tensor::from_array(([3, 2], vec![0.0_f32; 6])).unwrap();
        let recurrent = Tensor::from_array(([1, 1, 1], vec![0.0_f32])).unwrap();
        let prompt = prompt_session
            .run(ort::inputs![&token, &conv, &recurrent])
            .unwrap();
        let (_, logits) = prompt[0].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(logits, &[2.0, -2.0], 2.0e-5);
        let (_, next_conv) = prompt[1].try_extract_tensor::<f32>().unwrap();
        let (_, next_recurrent) = prompt[2].try_extract_tensor::<f32>().unwrap();
        let (_, present_k) = prompt[3].try_extract_tensor::<f32>().unwrap();
        let (_, present_v) = prompt[4].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(next_conv, &[0.0, 0.2, 0.0, -0.3, 0.0, 0.4], 1.0e-6);
        assert_f32_close(next_recurrent, &[-0.111_317_93], 2.0e-6);
        let next_conv = next_conv.to_vec();
        let next_recurrent = next_recurrent.to_vec();
        let present_k = present_k.to_vec();
        let present_v = present_v.to_vec();
        drop(prompt);

        let decode_model = crate::encode_qwen_causal_lm(crate::QwenCausalLmModel {
            past_tokens: 1,
            ..base
        })
        .unwrap();
        let mut decode_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&decode_model)
            .unwrap();
        let token = Tensor::from_array(([1], vec![1_i64])).unwrap();
        let conv = Tensor::from_array(([3, 2], next_conv)).unwrap();
        let recurrent = Tensor::from_array(([1, 1, 1], next_recurrent)).unwrap();
        let past_k = Tensor::from_array(([1, 1, hidden], present_k)).unwrap();
        let past_v = Tensor::from_array(([1, 1, hidden], present_v)).unwrap();
        let decode = decode_session
            .run(ort::inputs![&token, &conv, &recurrent, &past_k, &past_v])
            .unwrap();
        let (_, decode_logits) = decode[0].try_extract_tensor::<f32>().unwrap();
        assert!(decode_logits.iter().all(|value| value.is_finite()));
        let (conv_shape, _) = decode[1].try_extract_tensor::<f32>().unwrap();
        let (key_shape, _) = decode[3].try_extract_tensor::<f32>().unwrap();
        assert_eq!(conv_shape.as_ref(), &[3, 2]);
        assert_eq!(key_shape.as_ref(), &[2, 1, hidden as i64]);

        let external = crate::encode_external_qwen_causal_lm(base).unwrap();
        let directory = TestDirectory::new();
        let model_path = directory.0.join("model.onnx");
        std::fs::write(&model_path, external.model_bytes).unwrap();
        std::fs::write(directory.0.join("weights.bin"), external.weights_bytes).unwrap();
        let mut external_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_file(&model_path)
            .unwrap();
        let token = Tensor::from_array(([1], vec![0_i64])).unwrap();
        let conv = Tensor::from_array(([3, 2], vec![0.0_f32; 6])).unwrap();
        let recurrent = Tensor::from_array(([1, 1, 1], vec![0.0_f32])).unwrap();
        let external_prompt = external_session
            .run(ort::inputs![&token, &conv, &recurrent])
            .unwrap();
        let (_, external_logits) = external_prompt[0].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(external_logits, &[2.0, -2.0], 2.0e-5);
        let (_, external_conv) = external_prompt[1].try_extract_tensor::<f32>().unwrap();
        let (_, external_recurrent) = external_prompt[2].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(external_conv, &[0.0, 0.2, 0.0, -0.3, 0.0, 0.4], 1.0e-6);
        assert_f32_close(external_recurrent, &[-0.111_317_93], 2.0e-6);

        let mut mtp_fusion_rows = vec![zero(hidden * 2); hidden];
        for (row, values) in mtp_fusion_rows.iter_mut().enumerate() {
            values[row] = Trit::from_i8(1).unwrap();
        }
        let mtp_fusion_packed = pack_rows(&mtp_fusion_rows, format);
        let mtp_fusion_scales = vec![1.0; hidden];
        let mtp_model = crate::Qwen35MtpModel {
            tokens: 1,
            past_tokens: 0,
            n_head: base.n_head,
            n_kv_head: base.n_kv_head,
            head_dim: base.head_dim,
            rotary: base.rotary,
            rms_epsilon: base.rms_epsilon,
            embedding: base.embedding,
            lm_head: base.embedding,
            mtp: crate::Qwen35MtpDecoder {
                pre_fc_norm_embedding: &zero_norm,
                pre_fc_norm_hidden: &zero_norm,
                fusion: matrix(hidden, hidden * 2, &mtp_fusion_packed, &mtp_fusion_scales),
                layer: full,
                final_norm: &final_norm,
            },
            identity,
        };
        let bundle = crate::encode_external_qwen35_bundle(
            crate::QwenCausalLmModel {
                lm_head: Some(base.embedding),
                ..base
            },
            mtp_model,
        )
        .unwrap();
        let bundle_directory = TestDirectory::new();
        let language_path = bundle_directory.0.join("language.onnx");
        let mtp_path = bundle_directory.0.join("mtp.onnx");
        std::fs::write(&language_path, &bundle.language_model_bytes).unwrap();
        std::fs::write(&mtp_path, &bundle.mtp_model_bytes).unwrap();
        std::fs::write(
            bundle_directory.0.join("weights.bin"),
            &bundle.weights_bytes,
        )
        .unwrap();
        let mut bundled_language = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_file(&language_path)
            .unwrap();
        let token = Tensor::from_array(([1], vec![0_i64])).unwrap();
        let conv = Tensor::from_array(([3, 2], vec![0.0_f32; 6])).unwrap();
        let recurrent = Tensor::from_array(([1, 1, 1], vec![0.0_f32])).unwrap();
        let bundled_prompt = bundled_language
            .run(ort::inputs![&token, &conv, &recurrent])
            .unwrap();
        let (_, bundled_logits) = bundled_prompt[0].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(bundled_logits, &[2.0, -2.0], 2.0e-5);

        let inline_mtp = crate::encode_qwen35_mtp(mtp_model).unwrap();
        let mut inline_mtp_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&inline_mtp)
            .unwrap();
        let mut bundled_mtp = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_file(&mtp_path)
            .unwrap();
        let shifted = Tensor::from_array(([1], vec![0_i64])).unwrap();
        let target = Tensor::from_array(([1, hidden], vec![0.0_f32; hidden])).unwrap();
        let inline_outputs = inline_mtp_session
            .run(ort::inputs![&shifted, &target])
            .unwrap();
        let bundled_outputs = bundled_mtp.run(ort::inputs![&shifted, &target]).unwrap();
        for index in 0..4 {
            let (_, inline) = inline_outputs[index].try_extract_tensor::<f32>().unwrap();
            let (_, bundled) = bundled_outputs[index].try_extract_tensor::<f32>().unwrap();
            assert_f32_close(bundled, inline, 0.0);
        }
    }

    #[test]
    fn additive_qwen_language_and_mtp_bundle_run_in_real_ort() {
        use ort::value::Tensor;

        struct OwnedSaltMatrix {
            rows: usize,
            columns: usize,
            payload: Vec<u8>,
            scales: Vec<f16>,
            allocation_map: Vec<u8>,
            rank_prefixes: Vec<u32>,
            terminal_map_value: u32,
        }

        impl OwnedSaltMatrix {
            fn new(rows: usize, columns: usize, planes: &[Vec<Trit>]) -> Self {
                let logical = rows * columns;
                assert!((1..=3).contains(&planes.len()));
                assert!(planes.iter().all(|plane| plane.len() == logical));
                let tiles = logical.div_ceil(256);
                let map_bytes = tiles * 2 / 8;
                let mut allocation_map = vec![0_u8; map_bytes];
                let mut terminal_map_value = 0_u32;
                let code = u8::try_from(planes.len() - 1).unwrap();
                for tile in 0..tiles {
                    let bit = tile * 2;
                    if bit < map_bytes * 8 {
                        allocation_map[bit / 8] |= code << (bit % 8);
                    } else {
                        terminal_map_value |= u32::from(code) << (bit - map_bytes * 8);
                    }
                }
                let mut payload = Vec::new();
                let mut scales = Vec::new();
                let mut rank_prefixes = Vec::new();
                for tile in 0..tiles {
                    if tile > 0 && tile.is_multiple_of(256) {
                        rank_prefixes.push(u32::try_from(tile * planes.len()).unwrap());
                    }
                    let start = tile * 256;
                    let end = (start + 256).min(logical);
                    for (plane, trits) in planes.iter().enumerate() {
                        payload.extend(
                            pack_salt_v2_plane(SaltV2Codec::B3, &trits[start..end]).unwrap(),
                        );
                        scales.extend(vec![
                            f16::from_f32(1.0 / (plane + 1) as f32);
                            (end - start).div_ceil(128)
                        ]);
                    }
                }
                Self {
                    rows,
                    columns,
                    payload,
                    scales,
                    allocation_map,
                    rank_prefixes,
                    terminal_map_value,
                }
            }

            fn view(&self) -> crate::SaltV2PackedMatrix<'_> {
                crate::SaltV2PackedMatrix {
                    rows: self.rows,
                    columns: self.columns,
                    codec: SaltV2Codec::B3,
                    payload: &self.payload,
                    scales: &self.scales,
                    allocation_map: &self.allocation_map,
                    rank_prefixes: &self.rank_prefixes,
                    terminal_map_value: self.terminal_map_value,
                }
            }
        }

        let hidden = 4;
        let vocab = 16_385;
        let zeros = |len| vec![Trit::ZERO; len];
        let additive = |rows, columns| {
            let mut first = zeros(rows * columns);
            let mut second = zeros(rows * columns);
            first[0] = Trit::POS;
            second[0] = Trit::POS;
            OwnedSaltMatrix::new(rows, columns, &[first, second])
        };
        let mut embedding_first = zeros(vocab * hidden);
        let mut embedding_second = zeros(vocab * hidden);
        embedding_first[0] = Trit::POS;
        embedding_first[hidden] = Trit::NEG;
        embedding_second[0] = Trit::POS;
        let embedding = OwnedSaltMatrix::new(vocab, hidden, &[embedding_first, embedding_second]);
        assert!(!embedding.allocation_map.is_empty());
        assert!(!embedding.rank_prefixes.is_empty());
        let qkv = additive(3, hidden);
        let one_by_hidden = additive(1, hidden);
        let hidden_by_one = additive(hidden, 1);
        let fused_query_gate = additive(2 * hidden, hidden);
        let hidden_by_hidden = additive(hidden, hidden);
        let fusion = additive(hidden, 2 * hidden);
        let norm = vec![0.0_f32; hidden];
        let delta = crate::QwenDeltaNetDecoderLayer {
            attention_norm: &norm,
            qkv: qkv.view(),
            z: one_by_hidden.view(),
            beta: one_by_hidden.view(),
            decay: one_by_hidden.view(),
            conv_weight: &[0.0; 6],
            norm_weight: &[1.0],
            dt_bias: &[0.0],
            a_log: &[0.0],
            output: hidden_by_one.view(),
            ffn_norm: &norm,
            gate: one_by_hidden.view(),
            up: one_by_hidden.view(),
            down: hidden_by_one.view(),
        };
        let full = crate::QwenFullAttentionDecoderLayer {
            attention_norm: &norm,
            query_norm: &norm,
            key_norm: &norm,
            fused_query_gate: fused_query_gate.view(),
            key: hidden_by_hidden.view(),
            value: hidden_by_hidden.view(),
            attention_output: hidden_by_hidden.view(),
            ffn_norm: &norm,
            gate: one_by_hidden.view(),
            up: one_by_hidden.view(),
            down: hidden_by_one.view(),
        };
        let layers = [
            crate::QwenCausalLmDecoderLayer::DeltaNet(delta),
            crate::QwenCausalLmDecoderLayer::FullAttention(full),
        ];
        let identity = crate::OnnxArtifactIdentityV2 {
            source_model_id: "salt-qwen-source",
            tokenizer_id: "salt-qwen-tokenizer",
            recipe_id: "salt-qwen-recipe",
            tritium_build_id: "salt-qwen-build",
            package_id: "salt-qwen-package",
            converted_coverage_id: "language-mtp",
            deferred_coverage_id: "vision",
        };
        let language = crate::QwenCausalLmModel {
            tokens: 1,
            past_tokens: 0,
            n_head: 1,
            n_kv_head: 1,
            head_dim: hidden,
            rotary: crate::RotaryEmbedding {
                theta: 10_000.0,
                dimensions: hidden,
            },
            rms_epsilon: 1.0e-6,
            delta_geometry: crate::QwenDeltaNetGeometry::new(2, 1, 1, 1, 1).unwrap(),
            embedding: embedding.view(),
            lm_head: Some(embedding.view()),
            layers: &layers,
            final_norm: &norm,
            identity,
        };
        let mtp = crate::Qwen35MtpModel {
            tokens: 1,
            past_tokens: 0,
            n_head: 1,
            n_kv_head: 1,
            head_dim: hidden,
            rotary: language.rotary,
            rms_epsilon: language.rms_epsilon,
            embedding: embedding.view(),
            lm_head: embedding.view(),
            mtp: crate::Qwen35MtpDecoder {
                pre_fc_norm_embedding: &norm,
                pre_fc_norm_hidden: &norm,
                fusion: fusion.view(),
                layer: full,
                final_norm: &norm,
            },
            identity,
        };

        let inline_language = crate::encode_qwen_causal_lm(language).unwrap();
        let inline_mtp = crate::encode_qwen35_mtp(mtp).unwrap();
        for encoded in [&inline_language, &inline_mtp] {
            let diagnostics = crate::diagnose_unsupported_graph(encoded).unwrap();
            assert!(diagnostics.is_empty(), "{diagnostics:#?}");
            assert!(
                encoded
                    .windows(ONNX_SALT_V2_OP_NAME.len())
                    .any(|bytes| bytes == ONNX_SALT_V2_OP_NAME.as_bytes())
            );
            assert!(
                encoded
                    .windows(ONNX_SALT_V2_EMBEDDING_OP_NAME.len())
                    .any(|bytes| bytes == ONNX_SALT_V2_EMBEDDING_OP_NAME.as_bytes())
            );
            assert!(
                !encoded
                    .windows(ONNX_OP_NAME.len())
                    .any(|bytes| bytes == ONNX_OP_NAME.as_bytes())
            );
        }

        let bundle = crate::encode_external_qwen35_bundle(language, mtp).unwrap();
        let admitted = crate::AdmittedExternalQwen35BundleDigests {
            language_model_blake3: *blake3::hash(&bundle.language_model_bytes).as_bytes(),
            mtp_model_blake3: *blake3::hash(&bundle.mtp_model_bytes).as_bytes(),
            weights_blake3: bundle.weights_blake3,
        };
        crate::verify_external_qwen35_bundle(
            crate::ExternalQwen35BundleFiles {
                language_model_bytes: &bundle.language_model_bytes,
                mtp_model_bytes: &bundle.mtp_model_bytes,
                weights_bytes: &bundle.weights_bytes,
            },
            admitted,
        )
        .unwrap();

        let directory = TestDirectory::new();
        let language_path = directory.0.join("language.onnx");
        let mtp_path = directory.0.join("mtp.onnx");
        std::fs::write(&language_path, &bundle.language_model_bytes).unwrap();
        std::fs::write(&mtp_path, &bundle.mtp_model_bytes).unwrap();
        std::fs::write(directory.0.join("weights.bin"), &bundle.weights_bytes).unwrap();

        let mut inline_language_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&inline_language)
            .unwrap();
        let mut external_language_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_file(&language_path)
            .unwrap();
        let token = Tensor::from_array(([1], vec![0_i64])).unwrap();
        let conv = Tensor::from_array(([3, 2], vec![0.0_f32; 6])).unwrap();
        let recurrent = Tensor::from_array(([1, 1, 1], vec![0.0_f32])).unwrap();
        let inline_outputs = inline_language_session
            .run(ort::inputs![&token, &conv, &recurrent])
            .unwrap();
        let external_outputs = external_language_session
            .run(ort::inputs![&token, &conv, &recurrent])
            .unwrap();
        for index in 0..5 {
            let (_, inline) = inline_outputs[index].try_extract_tensor::<f32>().unwrap();
            let (_, external) = external_outputs[index].try_extract_tensor::<f32>().unwrap();
            assert!(external.iter().all(|value| value.is_finite()));
            assert_f32_close(external, inline, 0.0);
        }
        drop(inline_outputs);
        drop(external_outputs);

        let mut inline_mtp_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&inline_mtp)
            .unwrap();
        let mut external_mtp_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_file(&mtp_path)
            .unwrap();
        let shifted = Tensor::from_array(([1], vec![0_i64])).unwrap();
        let target = Tensor::from_array(([1, hidden], vec![0.0_f32; hidden])).unwrap();
        let inline_outputs = inline_mtp_session
            .run(ort::inputs![&shifted, &target])
            .unwrap();
        let external_outputs = external_mtp_session
            .run(ort::inputs![&shifted, &target])
            .unwrap();
        for index in 0..4 {
            let (_, inline) = inline_outputs[index].try_extract_tensor::<f32>().unwrap();
            let (_, external) = external_outputs[index].try_extract_tensor::<f32>().unwrap();
            assert!(external.iter().all(|value| value.is_finite()));
            assert_f32_close(external, inline, 0.0);
        }

        let causal_layer = crate::CausalLmDecoderLayer {
            attention_norm: &norm,
            query_norm: Some(&norm),
            key_norm: Some(&norm),
            query: crate::CausalQueryProjection::HeadInterleavedQueryGate {
                fused: fused_query_gate.view(),
            },
            key: hidden_by_hidden.view(),
            value: hidden_by_hidden.view(),
            attention_output: hidden_by_hidden.view(),
            attention_sub_norm: None,
            ffn_norm: &norm,
            gate: one_by_hidden.view(),
            up: one_by_hidden.view(),
            ffn_sub_norm: None,
            activation: crate::CausalActivation::SwiGlu,
            down: hidden_by_one.view(),
        };
        let causal_layers = [causal_layer];
        let causal = crate::CausalLmModel {
            tokens: 1,
            past_tokens: 0,
            n_head: 1,
            n_kv_head: 1,
            head_dim: hidden,
            rotary: Some(language.rotary),
            rms_epsilon: 1.0e-6,
            zero_centered_norm: true,
            embedding: embedding.view(),
            lm_head: Some(embedding.view()),
            layers: &causal_layers,
            final_norm: &norm,
            identity,
        };
        let inline_causal = crate::encode_causal_lm(causal).unwrap();
        assert!(
            crate::diagnose_unsupported_graph(&inline_causal)
                .unwrap()
                .is_empty()
        );
        let external_causal = crate::encode_external_causal_lm(causal).unwrap();
        let causal_directory = TestDirectory::new();
        let causal_path = causal_directory.0.join("model.onnx");
        std::fs::write(&causal_path, &external_causal.model_bytes).unwrap();
        std::fs::write(
            causal_directory.0.join("weights.bin"),
            &external_causal.weights_bytes,
        )
        .unwrap();
        let mut inline_causal_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&inline_causal)
            .unwrap();
        let mut external_causal_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_file(&causal_path)
            .unwrap();
        let inline_outputs = inline_causal_session.run(ort::inputs![&token]).unwrap();
        let external_outputs = external_causal_session.run(ort::inputs![&token]).unwrap();
        for index in 0..3 {
            let (_, inline) = inline_outputs[index].try_extract_tensor::<f32>().unwrap();
            let (_, external) = external_outputs[index].try_extract_tensor::<f32>().unwrap();
            assert_f32_close(external, inline, 0.0);
        }
    }

    #[test]
    fn qwen_mtp_graph_runs_prompt_and_cached_decode() {
        use ort::value::Tensor;

        let format = TernaryFormat::Tq2_0;
        let hidden = 16;
        let zero = |columns: usize| vec![Trit::ZERO; columns];
        let basis = |lane: usize, sign: i8, columns: usize| {
            let mut row = zero(columns);
            row[lane] = Trit::from_i8(sign).unwrap();
            row
        };
        let matrix = |rows, columns, packed, scales| crate::PackedTernaryMatrix {
            rows,
            columns,
            packed,
            scales,
            format,
        };
        let dense = |rows: &[Vec<Trit>], scales: &[f32]| {
            rows.iter()
                .zip(scales)
                .map(|(row, scale)| {
                    row.iter()
                        .map(|value| f32::from(*value) * scale)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let mut embedding_rows = vec![zero(hidden); hidden];
        embedding_rows[0] = basis(0, 1, hidden);
        embedding_rows[0][3] = Trit::from_i8(-1).unwrap();
        embedding_rows[1] = basis(1, 1, hidden);
        embedding_rows[1][4] = Trit::from_i8(1).unwrap();
        embedding_rows[2] = basis(2, -1, hidden);
        embedding_rows[2][5] = Trit::from_i8(1).unwrap();
        let embedding_scales = vec![1.0; hidden];
        let embedding_packed = pack_rows(&embedding_rows, format);

        let pre_embedding_norm = (0..hidden)
            .map(|lane| -0.12 + lane as f32 * 0.011)
            .collect::<Vec<_>>();
        let pre_hidden_norm = (0..hidden)
            .map(|lane| 0.09 - lane as f32 * 0.007)
            .collect::<Vec<_>>();
        let mut fusion_rows = vec![zero(hidden * 2); hidden];
        for (row, values) in fusion_rows.iter_mut().enumerate() {
            values[row] = Trit::from_i8(1).unwrap();
            values[hidden + (row + 3) % hidden] =
                Trit::from_i8(if row % 2 == 0 { 1 } else { -1 }).unwrap();
        }
        let fusion_scales = (0..hidden)
            .map(|row| 0.13 + 0.006 * row as f32)
            .collect::<Vec<_>>();
        let fusion_packed = pack_rows(&fusion_rows, format);

        let mut fused_query_gate_rows = vec![zero(hidden); hidden * 2];
        for row in 0..hidden {
            fused_query_gate_rows[row][row] = Trit::from_i8(1).unwrap();
            fused_query_gate_rows[hidden + row][(row + 5) % hidden] =
                Trit::from_i8(if row % 3 == 0 { -1 } else { 1 }).unwrap();
        }
        let fused_query_gate_scales = (0..hidden * 2)
            .map(|row| 0.18 + 0.003 * row as f32)
            .collect::<Vec<_>>();
        let fused_query_gate_packed = pack_rows(&fused_query_gate_rows, format);

        let key_rows = (0..hidden)
            .map(|row| {
                basis(
                    (row + 1) % hidden,
                    if row % 2 == 0 { 1 } else { -1 },
                    hidden,
                )
            })
            .collect::<Vec<_>>();
        let value_rows = (0..hidden)
            .map(|row| {
                basis(
                    (row + 4) % hidden,
                    if row % 3 == 0 { -1 } else { 1 },
                    hidden,
                )
            })
            .collect::<Vec<_>>();
        let output_rows = (0..hidden)
            .map(|row| {
                basis(
                    (row + 2) % hidden,
                    if row % 4 == 0 { -1 } else { 1 },
                    hidden,
                )
            })
            .collect::<Vec<_>>();
        let projection_scales = (0..hidden)
            .map(|row| 0.16 + 0.004 * row as f32)
            .collect::<Vec<_>>();
        let key_packed = pack_rows(&key_rows, format);
        let value_packed = pack_rows(&value_rows, format);
        let output_packed = pack_rows(&output_rows, format);

        let gate_rows = (0..hidden)
            .map(|row| basis(row, if row % 2 == 0 { 1 } else { -1 }, hidden))
            .collect::<Vec<_>>();
        let up_rows = (0..hidden)
            .map(|row| basis((row + 1) % hidden, 1, hidden))
            .collect::<Vec<_>>();
        let down_rows = (0..hidden)
            .map(|row| {
                basis(
                    (row + 3) % hidden,
                    if row % 3 == 0 { -1 } else { 1 },
                    hidden,
                )
            })
            .collect::<Vec<_>>();
        let ffn_scales = (0..hidden)
            .map(|row| 0.11 + 0.005 * row as f32)
            .collect::<Vec<_>>();
        let gate_packed = pack_rows(&gate_rows, format);
        let up_packed = pack_rows(&up_rows, format);
        let down_packed = pack_rows(&down_rows, format);

        let attention_norm = (0..hidden)
            .map(|lane| 0.08 - 0.006 * lane as f32)
            .collect::<Vec<_>>();
        let query_norm = (0..hidden)
            .map(|lane| -0.05 + 0.004 * lane as f32)
            .collect::<Vec<_>>();
        let key_norm = (0..hidden)
            .map(|lane| 0.06 - 0.003 * lane as f32)
            .collect::<Vec<_>>();
        let ffn_norm = (0..hidden)
            .map(|lane| -0.07 + 0.005 * lane as f32)
            .collect::<Vec<_>>();
        let final_norm = (0..hidden)
            .map(|lane| 0.04 - 0.002 * lane as f32)
            .collect::<Vec<_>>();
        let layer = crate::QwenFullAttentionDecoderLayer {
            attention_norm: &attention_norm,
            query_norm: &query_norm,
            key_norm: &key_norm,
            fused_query_gate: matrix(
                hidden * 2,
                hidden,
                &fused_query_gate_packed,
                &fused_query_gate_scales,
            ),
            key: matrix(hidden, hidden, &key_packed, &projection_scales),
            value: matrix(hidden, hidden, &value_packed, &projection_scales),
            attention_output: matrix(hidden, hidden, &output_packed, &projection_scales),
            ffn_norm: &ffn_norm,
            gate: matrix(hidden, hidden, &gate_packed, &ffn_scales),
            up: matrix(hidden, hidden, &up_packed, &ffn_scales),
            down: matrix(hidden, hidden, &down_packed, &ffn_scales),
        };
        let identity = crate::OnnxArtifactIdentityV2 {
            source_model_id: "qwen-source",
            tokenizer_id: "qwen-tokenizer",
            recipe_id: "qwen-recipe",
            tritium_build_id: "qwen-build",
            package_id: "qwen-package",
            converted_coverage_id: "qwen-language-mtp",
            deferred_coverage_id: "qwen-vision",
        };
        let head_rows = (0..hidden)
            .map(|row| basis(row, 1, hidden))
            .collect::<Vec<_>>();
        let head_scales = vec![1.0; hidden];
        let head_packed = pack_rows(&head_rows, format);
        let base = crate::Qwen35MtpModel {
            tokens: 2,
            past_tokens: 0,
            n_head: 1,
            n_kv_head: 1,
            head_dim: hidden,
            rotary: crate::RotaryEmbedding {
                theta: 10_000.0,
                dimensions: 8,
            },
            rms_epsilon: 1.0e-6,
            embedding: matrix(hidden, hidden, &embedding_packed, &embedding_scales),
            lm_head: matrix(hidden, hidden, &head_packed, &head_scales),
            mtp: crate::Qwen35MtpDecoder {
                pre_fc_norm_embedding: &pre_embedding_norm,
                pre_fc_norm_hidden: &pre_hidden_norm,
                fusion: matrix(hidden, hidden * 2, &fusion_packed, &fusion_scales),
                layer,
                final_norm: &final_norm,
            },
            identity,
        };

        let dense_embedding = dense(&embedding_rows, &embedding_scales);
        let dense_fusion = dense(&fusion_rows, &fusion_scales);
        let dense_fused_query_gate = dense(&fused_query_gate_rows, &fused_query_gate_scales);
        let dense_query = &dense_fused_query_gate[..hidden];
        let dense_attention_gate = &dense_fused_query_gate[hidden..];
        let dense_key = dense(&key_rows, &projection_scales);
        let dense_value = dense(&value_rows, &projection_scales);
        let dense_output = dense(&output_rows, &projection_scales);
        let dense_gate = dense(&gate_rows, &ffn_scales);
        let dense_up = dense(&up_rows, &ffn_scales);
        let dense_down = dense(&down_rows, &ffn_scales);
        let dense_head = identity_matrix(hidden);
        let fuse = |tokens: &[i64], target: &[f32]| {
            let embedded = tokens
                .iter()
                .flat_map(|token| dense_embedding[*token as usize].iter().copied())
                .collect::<Vec<_>>();
            let normalized_embedding = rms_norm(&embedded, &pre_embedding_norm, 1.0e-6, true);
            let normalized_target = rms_norm(target, &pre_hidden_norm, 1.0e-6, true);
            let concatenated = normalized_embedding
                .chunks_exact(hidden)
                .zip(normalized_target.chunks_exact(hidden))
                .flat_map(|(embedding, target)| embedding.iter().chain(target).copied())
                .collect::<Vec<_>>();
            dense_project(&concatenated, &dense_fusion)
        };
        let reference = |fused: &[f32], past_k: &[f32], past_v: &[f32]| {
            let rows = fused
                .chunks_exact(hidden)
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>();
            let row_ids = (0..rows.len()).map(|row| row as i64).collect::<Vec<_>>();
            reference_causal_lm(
                &row_ids,
                past_k,
                past_v,
                &rows,
                dense_query,
                &dense_key,
                &dense_value,
                &dense_output,
                Some(dense_attention_gate),
                &dense_gate,
                &dense_up,
                &dense_down,
                &attention_norm,
                &query_norm,
                &key_norm,
                &ffn_norm,
                None,
                None,
                crate::CausalActivation::SwiGlu,
                &final_norm,
                Some(&dense_head),
                1,
                1,
                hidden,
                8,
                10_000.0,
                1.0e-6,
                true,
            )
        };

        let prompt_model = crate::encode_qwen35_mtp(base).unwrap();
        let diagnostics = crate::diagnose_unsupported_graph(&prompt_model).unwrap();
        assert!(diagnostics.is_empty(), "{diagnostics:?}");
        let mut prompt_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&prompt_model)
            .unwrap();
        let prompt_tokens = vec![0_i64, 1];
        let prompt_target = (0..2 * hidden)
            .map(|lane| ((lane % 7) as f32 - 3.0) * 0.17)
            .collect::<Vec<_>>();
        let prompt_fused = fuse(&prompt_tokens, &prompt_target);
        let (expected_logits, expected_k, expected_v) = reference(&prompt_fused, &[], &[]);
        let shifted_tokens = Tensor::from_array(([2], prompt_tokens.into_boxed_slice())).unwrap();
        let target_hidden =
            Tensor::from_array(([2, hidden], prompt_target.into_boxed_slice())).unwrap();
        let prompt = prompt_session
            .run(ort::inputs![&shifted_tokens, &target_hidden])
            .unwrap();
        let (_, logits) = prompt[0].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(logits, &expected_logits, 2.0e-4);
        let (_, final_hidden) = prompt[1].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(final_hidden, &expected_logits, 2.0e-4);
        let (_, present_k) = prompt[2].try_extract_tensor::<f32>().unwrap();
        let (_, present_v) = prompt[3].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(present_k, &expected_k, 2.0e-4);
        assert_f32_close(present_v, &expected_v, 2.0e-4);
        let external = crate::encode_external_qwen35_mtp(base).unwrap();
        let directory = TestDirectory::new();
        let model_path = directory.0.join("model.onnx");
        std::fs::write(&model_path, external.model_bytes).unwrap();
        std::fs::write(directory.0.join("weights.bin"), external.weights_bytes).unwrap();
        let mut external_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_file(&model_path)
            .unwrap();
        let shifted_tokens = Tensor::from_array(([2], vec![0_i64, 1])).unwrap();
        let target_hidden = Tensor::from_array((
            [2, hidden],
            (0..2 * hidden)
                .map(|lane| ((lane % 7) as f32 - 3.0) * 0.17)
                .collect::<Vec<_>>(),
        ))
        .unwrap();
        let external_prompt = external_session
            .run(ort::inputs![&shifted_tokens, &target_hidden])
            .unwrap();
        let (_, external_logits) = external_prompt[0].try_extract_tensor::<f32>().unwrap();
        let (_, external_hidden) = external_prompt[1].try_extract_tensor::<f32>().unwrap();
        let (_, external_k) = external_prompt[2].try_extract_tensor::<f32>().unwrap();
        let (_, external_v) = external_prompt[3].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(external_logits, &expected_logits, 2.0e-4);
        assert_f32_close(external_hidden, &expected_logits, 2.0e-4);
        assert_f32_close(external_k, &expected_k, 2.0e-4);
        assert_f32_close(external_v, &expected_v, 2.0e-4);
        let present_k = present_k.to_vec();
        let present_v = present_v.to_vec();
        drop(prompt);

        let decode_model = crate::encode_qwen35_mtp(crate::Qwen35MtpModel {
            tokens: 1,
            past_tokens: 2,
            ..base
        })
        .unwrap();
        let mut decode_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&decode_model)
            .unwrap();
        let decode_tokens = vec![2_i64];
        let decode_target = (0..hidden)
            .map(|lane| ((lane % 5) as f32 - 2.0) * -0.21)
            .collect::<Vec<_>>();
        let decode_fused = fuse(&decode_tokens, &decode_target);
        let (expected_logits, expected_k, expected_v) =
            reference(&decode_fused, &present_k, &present_v);
        let (zero_cache_logits, _, _) = reference(
            &decode_fused,
            &vec![0.0; present_k.len()],
            &vec![0.0; present_v.len()],
        );
        assert_ne!(
            expected_logits, zero_cache_logits,
            "decode must consume KV cache"
        );
        let shifted_tokens = Tensor::from_array(([1], decode_tokens.into_boxed_slice())).unwrap();
        let target_hidden =
            Tensor::from_array(([1, hidden], decode_target.into_boxed_slice())).unwrap();
        let past_k = Tensor::from_array(([2, 1, hidden], present_k)).unwrap();
        let past_v = Tensor::from_array(([2, 1, hidden], present_v)).unwrap();
        let decode = decode_session
            .run(ort::inputs![
                &shifted_tokens,
                &target_hidden,
                &past_k,
                &past_v
            ])
            .unwrap();
        let (_, logits) = decode[0].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(logits, &expected_logits, 2.0e-4);
        let (_, final_hidden) = decode[1].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(final_hidden, &expected_logits, 2.0e-4);
        let (key_shape, present_k) = decode[2].try_extract_tensor::<f32>().unwrap();
        let (_, present_v) = decode[3].try_extract_tensor::<f32>().unwrap();
        assert_eq!(key_shape.as_ref(), &[3, 1, hidden as i64]);
        assert_f32_close(present_k, &expected_k, 2.0e-4);
        assert_f32_close(present_v, &expected_v, 2.0e-4);
    }

    #[test]
    fn end_to_end_kv_attention_runs_prompt_and_cached_decode() {
        use ort::value::Tensor;

        for (query_tokens, past_tokens, q, k, v, expected) in [
            (
                2usize,
                0usize,
                vec![1.0_f32, 2.0],
                vec![1.0_f32, 1.0],
                vec![10.0_f32, 20.0],
                vec![10.0_f32, 15.0],
            ),
            (
                1usize,
                2usize,
                vec![3.0_f32],
                vec![1.0_f32, 1.0, 1.0],
                vec![10.0_f32, 20.0, 40.0],
                vec![70.0_f32 / 3.0],
            ),
        ] {
            let model =
                crate::model::encode_kv_attention_test_graph(query_tokens, past_tokens, 1, 1, 1);
            let mut session = ort::session::Session::builder()
                .unwrap()
                .with_operators(tritium_operator_domain().unwrap())
                .unwrap()
                .commit_from_memory(&model)
                .unwrap();
            let total_tokens = query_tokens + past_tokens;
            let q_tensor =
                Tensor::from_array(([query_tokens, 1, 1], q.into_boxed_slice())).unwrap();
            let k_tensor =
                Tensor::from_array(([total_tokens, 1, 1], k.into_boxed_slice())).unwrap();
            let v_tensor =
                Tensor::from_array(([total_tokens, 1, 1], v.into_boxed_slice())).unwrap();
            let outputs = session
                .run(ort::inputs![&q_tensor, &k_tensor, &v_tensor])
                .unwrap();
            let (_, actual) = outputs[0].try_extract_tensor::<f32>().unwrap();
            for (&actual, &expected) in actual.iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn dynamic_kv_attention_runs_prompt_and_decode_on_one_session() {
        use ort::value::Tensor;

        let model = crate::model::encode_dynamic_kv_attention_test_graph(1, 1, 1);
        let diagnostics = crate::diagnose_unsupported_graph(&model).unwrap();
        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        let mut session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&model)
            .unwrap();

        let prompt_q = Tensor::from_array(([2, 1, 1], vec![1.0_f32, 2.0])).unwrap();
        let prompt_k = Tensor::from_array(([2, 1, 1], vec![1.0_f32, 1.0])).unwrap();
        let prompt_v = Tensor::from_array(([2, 1, 1], vec![10.0_f32, 20.0])).unwrap();
        let prompt = session
            .run(ort::inputs![&prompt_q, &prompt_k, &prompt_v])
            .unwrap();
        let (prompt_shape, prompt_values) = prompt[0].try_extract_tensor::<f32>().unwrap();
        assert_eq!(prompt_shape.as_ref(), &[2, 1, 1]);
        assert_f32_close(prompt_values, &[10.0, 15.0], 1.0e-5);
        drop(prompt);

        let decode_q = Tensor::from_array(([1, 1, 1], vec![3.0_f32])).unwrap();
        let decode_k = Tensor::from_array(([3, 1, 1], vec![1.0_f32, 1.0, 1.0])).unwrap();
        let decode_v = Tensor::from_array(([3, 1, 1], vec![10.0_f32, 20.0, 40.0])).unwrap();
        let decode = session
            .run(ort::inputs![&decode_q, &decode_k, &decode_v])
            .unwrap();
        let (decode_shape, decode_values) = decode[0].try_extract_tensor::<f32>().unwrap();
        assert_eq!(decode_shape.as_ref(), &[1, 1, 1]);
        assert_f32_close(decode_values, &[70.0 / 3.0], 1.0e-5);
    }

    #[test]
    fn end_to_end_standard_onnx_attention_runs_prompt_and_cached_decode() {
        use ort::value::Tensor;

        for (query_tokens, past_tokens, head_dim, q, k, v, mask) in [
            (
                2usize,
                0usize,
                2usize,
                vec![1.0_f32, 0.0, 0.0, 2.0],
                vec![1.0_f32, 0.0, 0.0, 1.0],
                vec![10.0_f32, 1.0, 20.0, 4.0],
                vec![0.0_f32, -1.0e9, 0.0, 0.0],
            ),
            (
                1usize,
                2usize,
                2usize,
                vec![1.0_f32, 1.0],
                vec![1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![10.0_f32, 1.0, 20.0, 4.0, 40.0, 8.0],
                vec![0.0_f32; 3],
            ),
        ] {
            let expected =
                kv_attention_kernel(&q, &k, &v, query_tokens, 1, 1, head_dim, past_tokens).unwrap();
            let model = crate::model::encode_standard_attention_test_graph(
                query_tokens,
                past_tokens + query_tokens,
                head_dim,
            );
            let mut session = ort::session::Session::builder()
                .unwrap()
                .commit_from_memory(&model)
                .unwrap();
            let total_tokens = past_tokens + query_tokens;
            let q_tensor =
                Tensor::from_array(([query_tokens, 1, head_dim], q.into_boxed_slice())).unwrap();
            let k_tensor =
                Tensor::from_array(([total_tokens, 1, head_dim], k.into_boxed_slice())).unwrap();
            let v_tensor =
                Tensor::from_array(([total_tokens, 1, head_dim], v.into_boxed_slice())).unwrap();
            let mask_tensor =
                Tensor::from_array(([query_tokens, 1, total_tokens], mask.into_boxed_slice()))
                    .unwrap();
            let outputs = session
                .run(ort::inputs![&q_tensor, &k_tensor, &v_tensor, &mask_tensor])
                .unwrap();
            let (_, actual) = outputs[0].try_extract_tensor::<f32>().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (&actual, &expected) in actual.iter().zip(&expected) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
    }

    /// The kernel's operand path (`TritiumTernaryMpGemmKernel::run`) reproduces
    /// the frozen conformance set bit-exactly — it routes through the same
    /// Layer-1 `ternary_mpgemm_kernel`. This exercises the Layer-2 plumbing
    /// (shape -> M, format dispatch, output sizing) without needing the native
    /// runtime, so it runs on every `--features onnx` build.
    #[test]
    fn kernel_run_matches_reference_on_frozen_set() {
        let vs = generate_vectors(FROZEN_SEED, FROZEN_COUNT);
        assert!(vs.len() > FROZEN_COUNT, "boundary vectors must be included");
        for v in vs {
            let format = v.format;
            let code = format_tag(format);
            let packed = pack(&v, format);
            let kernel = TritiumTernaryMpGemmKernel { k: v.k, format };
            // Sanity: the code round-trips to the same format the kernel uses.
            assert_eq!(format_from_attr(code).unwrap(), format);
            let act_shape = [v.m as i64, v.k as i64];
            let got = kernel
                .run(&act_shape, &v.activation, &packed, &v.scales)
                .unwrap();
            assert_eq!(
                got, v.expected,
                "vector {}: onnx kernel must be bit-exact with the reference",
                v.id
            );
        }
    }

    #[test]
    fn kernel_run_rejects_non_2d_activation() {
        let kernel = TritiumTernaryMpGemmKernel {
            k: 256,
            format: TernaryFormat::Tq2_0,
        };
        let r = kernel.run(&[1, 2, 3], &[0.0; 6], &[], &[1.0]);
        assert!(r.is_err(), "3-D activation must error");
    }

    #[test]
    fn kernel_run_rejects_packed_len_mismatch() {
        let kernel = TritiumTernaryMpGemmKernel {
            k: 256,
            format: TernaryFormat::Tq2_0,
        };
        // act [M=1, K=256] is valid, but packed is empty for a real [N=1, K=256].
        let r = kernel.run(&[1, 256], &[0.0; 256], &[], &[1.0]);
        assert!(r.is_err(), "packed length mismatch must error");
    }

    #[test]
    fn embedding_kernel_preserves_token_rank_and_values() {
        let k = 256;
        let format = TernaryFormat::Tq2_0;
        let rows = vec![vec![Trit::NEG; k], vec![Trit::ZERO; k], vec![Trit::POS; k]];
        let packed = pack_rows(&rows, format);
        let scales = [0.5, 2.0, 1.25];
        let tokens = [2, 0, 1, 2];
        let kernel = TritiumTernaryEmbeddingKernel { k, format };
        let (shape, output) = kernel.run(&[2, 2], &tokens, &packed, &scales).unwrap();
        assert_eq!(shape, [2, 2, k as i64]);
        let expected =
            ternary_embedding_kernel(&tokens, &[2, 2], &packed, &scales, k, format).unwrap();
        assert_eq!(output, expected);
    }

    /// Compile-level registration: the operator binds into a domain. This proves
    /// the `Operator`/`Kernel` trait wiring satisfies `ort`'s registration path.
    /// Constructing the domain initializes the native runtime, so it is gated to
    /// the same builds that have the fetched onnxruntime available.
    #[test]
    fn operator_domain_registers() {
        let domain = tritium_operator_domain();
        assert!(
            domain.is_ok(),
            "registering the tritium domain must succeed: {:?}",
            domain.err()
        );
    }

    /// Production serializer plus both custom operators: the only runtime input
    /// is token IDs; packed weights and scales are tied graph initializers.
    #[test]
    fn end_to_end_tied_embedding_head_matches_reference() {
        use ort::value::Tensor;

        let k = 256;
        let format = TernaryFormat::Tq2_0;
        let rows = vec![vec![Trit::NEG; k], vec![Trit::ZERO; k], vec![Trit::POS; k]];
        let packed = pack_rows(&rows, format);
        let scales = vec![0.5, 2.0, 1.25];
        let tokens = vec![2i64, 0, 1, 2];
        let hidden =
            ternary_embedding_kernel(&tokens, &[tokens.len()], &packed, &scales, k, format)
                .unwrap();
        let expected =
            ternary_mpgemm_kernel(&hidden, &packed, &scales, tokens.len(), k, format).unwrap();
        let model = crate::encode_tied_embedding_head(crate::TiedEmbeddingHeadModel {
            tokens: tokens.len(),
            vocab: rows.len(),
            hidden: k,
            packed: &packed,
            scales: &scales,
            format,
            source_model_id: "test-source",
            recipe_id: "test-recipe",
            package_id: "test-package",
        })
        .unwrap();
        let mut session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&model)
            .unwrap();
        assert_eq!(session.opset_for_domain(ONNX_DOMAIN).unwrap(), 1);
        let tokens_t = Tensor::from_array(([tokens.len()], tokens.into_boxed_slice())).unwrap();
        let outputs = session.run(ort::inputs![&tokens_t]).unwrap();
        let (shape, got) = outputs[0].try_extract_tensor::<f32>().unwrap();
        assert_eq!(shape.as_ref(), &[4, rows.len() as i64]);
        assert_eq!(got, expected.as_slice(), "e2e tied graph bit-exact");
    }

    #[test]
    fn end_to_end_external_data_matches_reference() {
        use ort::value::Tensor;

        let k = 256;
        let format = TernaryFormat::Tq1_0;
        let rows = vec![vec![Trit::NEG; k], vec![Trit::ZERO; k], vec![Trit::POS; k]];
        let packed = pack_rows(&rows, format);
        let scales = vec![0.5, 2.0, 1.25];
        let tokens = vec![2i64, 0, 1, 2];
        let hidden =
            ternary_embedding_kernel(&tokens, &[tokens.len()], &packed, &scales, k, format)
                .unwrap();
        let expected =
            ternary_mpgemm_kernel(&hidden, &packed, &scales, tokens.len(), k, format).unwrap();
        let bundle =
            crate::encode_external_tied_embedding_head_v2(crate::TiedEmbeddingHeadModelV2 {
                tokens: tokens.len(),
                vocab: rows.len(),
                hidden: k,
                packed: &packed,
                scales: &scales,
                format,
                identity: crate::OnnxArtifactIdentityV2 {
                    source_model_id: "test-source",
                    tokenizer_id: "test-tokenizer",
                    recipe_id: "test-recipe",
                    tritium_build_id: "test-build",
                    package_id: "test-package",
                    converted_coverage_id: "test-converted",
                    deferred_coverage_id: "test-deferred",
                },
            })
            .unwrap();
        let directory = TestDirectory::new();
        let model_path = directory.0.join("model.onnx");
        std::fs::write(&model_path, bundle.model_bytes).unwrap();
        std::fs::write(directory.0.join("weights.bin"), bundle.weights_bytes).unwrap();
        let mut session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_file(&model_path)
            .unwrap();
        let tokens_t = Tensor::from_array(([tokens.len()], tokens.into_boxed_slice())).unwrap();
        let outputs = session.run(ort::inputs![&tokens_t]).unwrap();
        let (_, got) = outputs[0].try_extract_tensor::<f32>().unwrap();
        assert_eq!(got, expected.as_slice(), "external-data graph bit-exact");
    }

    fn dense_project(input: &[f32], rows: &[Vec<f32>]) -> Vec<f32> {
        let columns = rows[0].len();
        input
            .chunks_exact(columns)
            .flat_map(|x| {
                rows.iter()
                    .map(move |w| x.iter().zip(w).map(|(x, w)| x * w).sum::<f32>())
            })
            .collect()
    }

    fn identity_matrix(size: usize) -> Vec<Vec<f32>> {
        (0..size)
            .map(|row| (0..size).map(|column| f32::from(row == column)).collect())
            .collect()
    }

    fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32, zero_centered: bool) -> Vec<f32> {
        let width = weight.len();
        input
            .chunks_exact(width)
            .flat_map(|row| {
                let denominator =
                    (row.iter().map(|value| value * value).sum::<f32>() / width as f32 + epsilon)
                        .sqrt();
                row.iter().zip(weight).map(move |(value, weight)| {
                    let scale = if zero_centered { 1.0 + weight } else { *weight };
                    value * scale / denominator
                })
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_causal_lm(
        tokens: &[i64],
        past_k: &[f32],
        past_v: &[f32],
        embedding: &[Vec<f32>],
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        o: &[Vec<f32>],
        attention_gate: Option<&[Vec<f32>]>,
        gate: &[Vec<f32>],
        up: &[Vec<f32>],
        down: &[Vec<f32>],
        attention_norm: &[f32],
        query_norm: &[f32],
        key_norm: &[f32],
        ffn_norm: &[f32],
        attention_sub_norm: Option<&[f32]>,
        ffn_sub_norm: Option<&[f32]>,
        activation: crate::CausalActivation,
        final_norm: &[f32],
        lm_head: Option<&[Vec<f32>]>,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        rotary_dim: usize,
        rope_theta: f32,
        epsilon: f32,
        zero_centered_norm: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let hidden: Vec<f32> = tokens
            .iter()
            .flat_map(|&token| embedding[usize::try_from(token).unwrap()].iter().copied())
            .collect();
        let attention_input = rms_norm(&hidden, attention_norm, epsilon, zero_centered_norm);
        let query = apply_rope(
            &rms_norm(
                &dense_project(&attention_input, q),
                query_norm,
                epsilon,
                zero_centered_norm,
            ),
            tokens.len(),
            n_head,
            head_dim,
            rotary_dim,
            past_k.len() / (n_kv_head * head_dim),
            rope_theta,
        );
        let current_k = apply_rope(
            &rms_norm(
                &dense_project(&attention_input, k),
                key_norm,
                epsilon,
                zero_centered_norm,
            ),
            tokens.len(),
            n_kv_head,
            head_dim,
            rotary_dim,
            past_k.len() / (n_kv_head * head_dim),
            rope_theta,
        );
        let current_v = dense_project(&attention_input, v);
        let present_k = [past_k, &current_k].concat();
        let present_v = [past_v, &current_v].concat();
        let context = kv_attention_kernel(
            &query,
            &present_k,
            &present_v,
            tokens.len(),
            n_head,
            n_kv_head,
            head_dim,
            past_k.len() / (n_kv_head * head_dim),
        )
        .unwrap();
        let gated_context = attention_gate.map_or(context.clone(), |weight| {
            let gates = dense_project(&attention_input, weight);
            context
                .iter()
                .zip(gates)
                .map(|(value, gate)| value * (1.0 / (1.0 + (-gate).exp())))
                .collect()
        });
        let output_input = attention_sub_norm.map_or(gated_context.clone(), |weight| {
            rms_norm(&gated_context, weight, epsilon, zero_centered_norm)
        });
        let attention_output = dense_project(&output_input, o);
        let post_attention: Vec<f32> = hidden
            .iter()
            .zip(attention_output)
            .map(|(residual, update)| residual + update)
            .collect();
        let ffn_input = rms_norm(&post_attention, ffn_norm, epsilon, zero_centered_norm);
        let gate_values = dense_project(&ffn_input, gate);
        let up_values = dense_project(&ffn_input, up);
        let activated: Vec<f32> = gate_values
            .into_iter()
            .zip(up_values)
            .map(|(gate, up)| {
                let activated_gate = match activation {
                    crate::CausalActivation::SwiGlu => gate * (1.0 / (1.0 + (-gate).exp())),
                    crate::CausalActivation::Relu2 => gate.max(0.0).powi(2),
                };
                activated_gate * up
            })
            .collect();
        let down_input = ffn_sub_norm.map_or(activated.clone(), |weight| {
            rms_norm(&activated, weight, epsilon, zero_centered_norm)
        });
        let ffn_output = dense_project(&down_input, down);
        let post_ffn: Vec<f32> = post_attention
            .iter()
            .zip(ffn_output)
            .map(|(residual, update)| residual + update)
            .collect();
        let final_hidden = rms_norm(&post_ffn, final_norm, epsilon, zero_centered_norm);
        let logits = dense_project(&final_hidden, lm_head.unwrap_or(embedding));
        (logits, present_k, present_v)
    }

    fn apply_rope(
        input: &[f32],
        tokens: usize,
        heads: usize,
        head_dim: usize,
        rotary_dim: usize,
        past_tokens: usize,
        theta: f32,
    ) -> Vec<f32> {
        let half = rotary_dim / 2;
        let mut output = input.to_vec();
        for token in 0..tokens {
            let position = (past_tokens + token) as f64;
            for head in 0..heads {
                let base = (token * heads + head) * head_dim;
                for lane in 0..half {
                    let angle =
                        position * f64::from(theta).powf(-2.0 * lane as f64 / rotary_dim as f64);
                    let (sin, cos) = angle.sin_cos();
                    let first = input[base + lane];
                    let second = input[base + half + lane];
                    output[base + lane] = first * cos as f32 - second * sin as f32;
                    output[base + half + lane] = second * cos as f32 + first * sin as f32;
                }
            }
        }
        output
    }

    #[test]
    fn end_to_end_packed_causal_lm_runs_prompt_and_cached_decode() {
        use ort::value::Tensor;

        let format = TernaryFormat::Tq2_0;
        let dense_embedding: Vec<Vec<f32>> = vec![
            (0..16).map(|lane| f32::from(lane % 2 == 0)).collect(),
            (0..16).map(|lane| f32::from(lane % 2 == 1)).collect(),
            (0..16)
                .map(|lane| match lane % 4 {
                    0 | 1 => 1.0,
                    2 => 0.0,
                    _ => -1.0,
                })
                .collect(),
        ];
        let dense_q = identity_matrix(16);
        let dense_lm_head = vec![
            dense_embedding[1].clone(),
            dense_embedding[2].clone(),
            dense_embedding[0].clone(),
        ];
        let dense_k = (0..8)
            .map(|row| {
                let mut values = vec![0.0; 16];
                values[row] = 1.0;
                values[(row + 5) % 16] = 1.0;
                values
            })
            .collect::<Vec<_>>();
        let dense_v = (0..8)
            .map(|row| {
                let mut values = vec![0.0; 16];
                values[(row * 2) % 16] = 1.0;
                values[(row * 2 + 1) % 16] = -1.0;
                values
            })
            .collect::<Vec<_>>();
        let dense_o = identity_matrix(16);
        let dense_attention_gate = (0..8)
            .map(|row| {
                let mut values = vec![0.0; 16];
                values[(row + 3) % 16] = if row % 2 == 0 { 1.0 } else { -1.0 };
                values
            })
            .collect::<Vec<_>>();
        let dense_fused_query_gate = (0..2)
            .flat_map(|head| {
                dense_q[head * 4..head * 4 + 4]
                    .iter()
                    .chain(&dense_attention_gate[head * 4..head * 4 + 4])
                    .cloned()
            })
            .collect::<Vec<_>>();
        let dense_fused_output = (0..16)
            .map(|row| {
                let mut values = vec![0.0; 8];
                values[row % 8] = if row < 8 { 1.0 } else { -1.0 };
                values
            })
            .collect::<Vec<_>>();
        let dense_gate = identity_matrix(16);
        let dense_up = identity_matrix(16);
        let dense_down = identity_matrix(16);
        let trits = |rows: &[Vec<f32>]| {
            rows.iter()
                .map(|row| {
                    row.iter()
                        .map(|&value| Trit::from_i8(value as i8).unwrap())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let embedding_packed = pack_rows(&trits(&dense_embedding), format);
        let lm_head_packed = pack_rows(&trits(&dense_lm_head), TernaryFormat::Tq1_0);
        let q_packed = pack_rows(&trits(&dense_q), format);
        let k_packed = pack_rows(&trits(&dense_k), format);
        let v_packed = pack_rows(&trits(&dense_v), format);
        let o_packed = pack_rows(&trits(&dense_o), format);
        let fused_query_gate_packed = pack_rows(&trits(&dense_fused_query_gate), format);
        let fused_k_packed = pack_rows(&trits(&dense_k[..4]), format);
        let fused_v_packed = pack_rows(&trits(&dense_v[..4]), format);
        let fused_output_packed = pack_rows(&trits(&dense_fused_output), format);
        let gate_packed = pack_rows(&trits(&dense_gate), format);
        let up_packed = pack_rows(&trits(&dense_up), format);
        let down_packed = pack_rows(&trits(&dense_down), format);
        let embedding_scales = [1.0; 3];
        let sixteen_scales = vec![1.0; 16];
        let eight_scales = vec![1.0; 8];
        let four_scales = vec![1.0; 4];
        let attention_norm = (0..16)
            .map(|lane| 0.5 + 0.1 * (lane % 10) as f32)
            .collect::<Vec<_>>();
        let query_norm = [1.0, 0.5, 1.25, 0.75];
        let key_norm = [0.75, 1.25, 0.6, 1.4];
        let ffn_norm = (0..16)
            .map(|lane| 0.6 + 0.08 * (lane % 9) as f32)
            .collect::<Vec<_>>();
        let final_norm = (0..16)
            .map(|lane| 0.7 + 0.07 * (lane % 8) as f32)
            .collect::<Vec<_>>();
        let rope_theta = 10_000.0;
        let epsilon = 1.0e-5;
        let layer = crate::CausalLmDecoderLayer {
            attention_norm: &attention_norm,
            query_norm: Some(&query_norm),
            key_norm: Some(&key_norm),
            query: crate::CausalQueryProjection::Separate {
                query: packed_matrix(16, 16, &q_packed, &sixteen_scales, format),
                gate: None,
            },
            key: packed_matrix(8, 16, &k_packed, &eight_scales, format),
            value: packed_matrix(8, 16, &v_packed, &eight_scales, format),
            attention_output: packed_matrix(16, 16, &o_packed, &sixteen_scales, format),
            attention_sub_norm: None,
            ffn_norm: &ffn_norm,
            gate: packed_matrix(16, 16, &gate_packed, &sixteen_scales, format),
            up: packed_matrix(16, 16, &up_packed, &sixteen_scales, format),
            ffn_sub_norm: None,
            activation: crate::CausalActivation::SwiGlu,
            down: packed_matrix(16, 16, &down_packed, &sixteen_scales, format),
        };
        let identity = crate::OnnxArtifactIdentityV2 {
            source_model_id: "tiny-source@revision",
            tokenizer_id: "tiny-tokenizer@revision",
            recipe_id: "tiny-recipe",
            tritium_build_id: "tiny-build",
            package_id: "tiny-package",
            converted_coverage_id: "tiny-converted",
            deferred_coverage_id: "tiny-deferred",
        };
        let embedding = packed_matrix(3, 16, &embedding_packed, &embedding_scales, format);
        let lm_head = packed_matrix(
            3,
            16,
            &lm_head_packed,
            &embedding_scales,
            TernaryFormat::Tq1_0,
        );
        let encode_result = |tokens, past_tokens, layer, rotary, lm_head| {
            crate::encode_causal_lm(crate::CausalLmModel {
                tokens,
                past_tokens,
                n_head: 4,
                n_kv_head: 2,
                head_dim: 4,
                rotary,
                rms_epsilon: epsilon,
                zero_centered_norm: false,
                embedding,
                lm_head,
                layers: std::slice::from_ref(&layer),
                final_norm: &final_norm,
                identity,
            })
        };
        let encode = |tokens, past_tokens| {
            encode_result(
                tokens,
                past_tokens,
                layer,
                Some(crate::RotaryEmbedding {
                    theta: rope_theta,
                    dimensions: 4,
                }),
                None,
            )
            .unwrap()
        };
        assert!(
            encode_result(
                1,
                0,
                layer,
                Some(crate::RotaryEmbedding {
                    theta: f32::NAN,
                    dimensions: 4,
                }),
                None,
            )
            .is_err()
        );
        let short_norm = [1.0];
        assert!(
            encode_result(
                1,
                0,
                crate::CausalLmDecoderLayer {
                    query_norm: Some(&short_norm),
                    ..layer
                },
                Some(crate::RotaryEmbedding {
                    theta: rope_theta,
                    dimensions: 4,
                }),
                None,
            )
            .is_err()
        );
        let plain_layer = crate::CausalLmDecoderLayer {
            query_norm: None,
            key_norm: None,
            ..layer
        };
        let plain = encode_result(1, 0, plain_layer, None, None).unwrap();
        assert!(
            crate::diagnose_unsupported_graph(&plain)
                .unwrap()
                .is_empty()
        );

        let prompt_tokens = vec![0_i64, 1];
        let (expected_logits, expected_k, expected_v) = reference_causal_lm(
            &prompt_tokens,
            &[],
            &[],
            &dense_embedding,
            &dense_q,
            &dense_k,
            &dense_v,
            &dense_o,
            None,
            &dense_gate,
            &dense_up,
            &dense_down,
            &attention_norm,
            &query_norm,
            &key_norm,
            &ffn_norm,
            None,
            None,
            crate::CausalActivation::SwiGlu,
            &final_norm,
            None,
            4,
            2,
            4,
            4,
            rope_theta,
            epsilon,
            false,
        );
        let prompt_model = encode(2, 0);
        assert_eq!(prompt_model, encode(2, 0));
        let (untied_expected, _, _) = reference_causal_lm(
            &prompt_tokens,
            &[],
            &[],
            &dense_embedding,
            &dense_q,
            &dense_k,
            &dense_v,
            &dense_o,
            None,
            &dense_gate,
            &dense_up,
            &dense_down,
            &attention_norm,
            &query_norm,
            &key_norm,
            &ffn_norm,
            None,
            None,
            crate::CausalActivation::SwiGlu,
            &final_norm,
            Some(&dense_lm_head),
            4,
            2,
            4,
            4,
            rope_theta,
            epsilon,
            false,
        );
        let untied_model = encode_result(
            2,
            0,
            layer,
            Some(crate::RotaryEmbedding {
                theta: rope_theta,
                dimensions: 4,
            }),
            Some(lm_head),
        )
        .unwrap();
        let mut untied_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&untied_model)
            .unwrap();
        let untied_tokens =
            Tensor::from_array(([2], prompt_tokens.clone().into_boxed_slice())).unwrap();
        let untied_outputs = untied_session.run(ort::inputs![&untied_tokens]).unwrap();
        let (_, untied_logits) = untied_outputs[0].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(untied_logits, &untied_expected, 2.0e-5);
        assert_ne!(untied_logits, expected_logits.as_slice());
        let partial_query_norm = [1.0, 0.5, 1.25, 0.75, 0.8, 1.1, 0.6, 1.4];
        let partial_key_norm = [0.75, 1.25, 0.6, 1.4, 1.0, 0.9, 1.2, 0.7];
        let (partial_rope_expected, _, _) = reference_causal_lm(
            &prompt_tokens,
            &[],
            &[],
            &dense_embedding,
            &dense_q,
            &dense_k,
            &dense_v,
            &dense_o,
            None,
            &dense_gate,
            &dense_up,
            &dense_down,
            &attention_norm,
            &partial_query_norm,
            &partial_key_norm,
            &ffn_norm,
            None,
            None,
            crate::CausalActivation::SwiGlu,
            &final_norm,
            None,
            2,
            1,
            8,
            4,
            rope_theta,
            epsilon,
            true,
        );
        let partial_rope_layer = crate::CausalLmDecoderLayer {
            query_norm: Some(&partial_query_norm),
            key_norm: Some(&partial_key_norm),
            ..layer
        };
        let partial_rope_model = crate::encode_causal_lm(crate::CausalLmModel {
            tokens: 2,
            past_tokens: 0,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 8,
            zero_centered_norm: true,
            rotary: Some(crate::RotaryEmbedding {
                theta: rope_theta,
                dimensions: 4,
            }),
            rms_epsilon: epsilon,
            embedding,
            lm_head: None,
            layers: std::slice::from_ref(&partial_rope_layer),
            final_norm: &final_norm,
            identity,
        })
        .unwrap();
        let partial_diagnostics = crate::diagnose_unsupported_graph(&partial_rope_model).unwrap();
        assert!(partial_diagnostics.is_empty(), "{partial_diagnostics:#?}");
        let mut partial_rope_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&partial_rope_model)
            .unwrap();
        let partial_rope_tokens =
            Tensor::from_array(([2], prompt_tokens.clone().into_boxed_slice())).unwrap();
        let partial_rope_outputs = partial_rope_session
            .run(ort::inputs![&partial_rope_tokens])
            .unwrap();
        let (_, partial_rope_logits) = partial_rope_outputs[0].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(partial_rope_logits, &partial_rope_expected, 2.0e-5);
        assert_ne!(partial_rope_logits, expected_logits.as_slice());
        let gate_sub_norm = (0..16)
            .map(|lane| 0.9 + 0.025 * lane as f32)
            .collect::<Vec<_>>();
        let (attention_gate_expected, _, _) = reference_causal_lm(
            &prompt_tokens,
            &[],
            &[],
            &dense_embedding,
            &dense_q,
            &dense_k,
            &dense_v,
            &dense_o,
            Some(&dense_q),
            &dense_gate,
            &dense_up,
            &dense_down,
            &attention_norm,
            &query_norm,
            &key_norm,
            &ffn_norm,
            Some(&gate_sub_norm),
            None,
            crate::CausalActivation::SwiGlu,
            &final_norm,
            None,
            4,
            2,
            4,
            4,
            rope_theta,
            epsilon,
            false,
        );
        let attention_gate_model = encode_result(
            2,
            0,
            crate::CausalLmDecoderLayer {
                query: crate::CausalQueryProjection::Separate {
                    query: packed_matrix(16, 16, &q_packed, &sixteen_scales, format),
                    gate: Some(packed_matrix(16, 16, &q_packed, &sixteen_scales, format)),
                },
                attention_sub_norm: Some(&gate_sub_norm),
                ..layer
            },
            Some(crate::RotaryEmbedding {
                theta: rope_theta,
                dimensions: 4,
            }),
            None,
        )
        .unwrap();
        let mut attention_gate_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&attention_gate_model)
            .unwrap();
        let attention_gate_tokens =
            Tensor::from_array(([2], prompt_tokens.clone().into_boxed_slice())).unwrap();
        let attention_gate_outputs = attention_gate_session
            .run(ort::inputs![&attention_gate_tokens])
            .unwrap();
        let (_, attention_gate_logits) = attention_gate_outputs[0]
            .try_extract_tensor::<f32>()
            .unwrap();
        assert_f32_close(attention_gate_logits, &attention_gate_expected, 2.0e-5);
        assert_ne!(attention_gate_logits, expected_logits.as_slice());
        let fused_attention_sub_norm = [0.7, 1.1, 0.8, 1.2, 0.9, 1.3, 0.6, 1.4];
        let (fused_query_gate_expected, _, _) = reference_causal_lm(
            &prompt_tokens,
            &[],
            &[],
            &dense_embedding,
            &dense_q[..8],
            &dense_k[..4],
            &dense_v[..4],
            &dense_fused_output,
            Some(&dense_attention_gate),
            &dense_gate,
            &dense_up,
            &dense_down,
            &attention_norm,
            &query_norm,
            &key_norm,
            &ffn_norm,
            Some(&fused_attention_sub_norm),
            None,
            crate::CausalActivation::SwiGlu,
            &final_norm,
            None,
            2,
            1,
            4,
            4,
            rope_theta,
            epsilon,
            false,
        );
        let fused_layer = crate::CausalLmDecoderLayer {
            query: crate::CausalQueryProjection::HeadInterleavedQueryGate {
                fused: packed_matrix(16, 16, &fused_query_gate_packed, &sixteen_scales, format),
            },
            key: packed_matrix(4, 16, &fused_k_packed, &four_scales, format),
            value: packed_matrix(4, 16, &fused_v_packed, &four_scales, format),
            attention_output: packed_matrix(16, 8, &fused_output_packed, &sixteen_scales, format),
            attention_sub_norm: Some(&fused_attention_sub_norm),
            ..layer
        };
        let fused_query_gate_model = crate::encode_causal_lm(crate::CausalLmModel {
            tokens: 2,
            past_tokens: 0,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 4,
            rotary: Some(crate::RotaryEmbedding {
                theta: rope_theta,
                dimensions: 4,
            }),
            rms_epsilon: epsilon,
            zero_centered_norm: false,
            embedding,
            lm_head: None,
            layers: std::slice::from_ref(&fused_layer),
            final_norm: &final_norm,
            identity,
        })
        .unwrap();
        let fused_diagnostics = crate::diagnose_unsupported_graph(&fused_query_gate_model).unwrap();
        assert!(fused_diagnostics.is_empty(), "{fused_diagnostics:#?}");
        let mut fused_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&fused_query_gate_model)
            .unwrap();
        let fused_tokens =
            Tensor::from_array(([2], prompt_tokens.clone().into_boxed_slice())).unwrap();
        let fused_outputs = fused_session.run(ort::inputs![&fused_tokens]).unwrap();
        let (_, fused_logits) = fused_outputs[0].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(fused_logits, &fused_query_gate_expected, 2.0e-5);
        assert_ne!(fused_logits, attention_gate_logits);
        let attention_sub_norm = (0..16)
            .map(|lane| 0.8 + 0.03 * lane as f32)
            .collect::<Vec<_>>();
        let ffn_sub_norm = (0..16)
            .map(|lane| 1.2 - 0.02 * lane as f32)
            .collect::<Vec<_>>();
        let bitnet_layer = crate::CausalLmDecoderLayer {
            attention_sub_norm: Some(&attention_sub_norm),
            ffn_sub_norm: Some(&ffn_sub_norm),
            activation: crate::CausalActivation::Relu2,
            ..layer
        };
        let bitnet_model = encode_result(
            2,
            0,
            bitnet_layer,
            Some(crate::RotaryEmbedding {
                theta: rope_theta,
                dimensions: 4,
            }),
            None,
        )
        .unwrap();
        assert!(
            crate::diagnose_unsupported_graph(&bitnet_model)
                .unwrap()
                .is_empty()
        );
        let (bitnet_expected, _, _) = reference_causal_lm(
            &prompt_tokens,
            &[],
            &[],
            &dense_embedding,
            &dense_q,
            &dense_k,
            &dense_v,
            &dense_o,
            None,
            &dense_gate,
            &dense_up,
            &dense_down,
            &attention_norm,
            &query_norm,
            &key_norm,
            &ffn_norm,
            Some(&attention_sub_norm),
            Some(&ffn_sub_norm),
            crate::CausalActivation::Relu2,
            &final_norm,
            None,
            4,
            2,
            4,
            4,
            rope_theta,
            epsilon,
            false,
        );
        let mut bitnet_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&bitnet_model)
            .unwrap();
        let bitnet_tokens =
            Tensor::from_array(([2], prompt_tokens.clone().into_boxed_slice())).unwrap();
        let bitnet_outputs = bitnet_session.run(ort::inputs![&bitnet_tokens]).unwrap();
        let (_, bitnet_logits) = bitnet_outputs[0].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(bitnet_logits, &bitnet_expected, 2.0e-5);
        assert!(
            encode_result(
                1,
                0,
                crate::CausalLmDecoderLayer {
                    ffn_sub_norm: Some(&short_norm),
                    ..bitnet_layer
                },
                Some(crate::RotaryEmbedding {
                    theta: rope_theta,
                    dimensions: 4,
                }),
                None,
            )
            .is_err()
        );
        let external = crate::encode_external_causal_lm(crate::CausalLmModel {
            tokens: 2,
            past_tokens: 0,
            n_head: 4,
            n_kv_head: 2,
            head_dim: 4,
            rotary: Some(crate::RotaryEmbedding {
                theta: rope_theta,
                dimensions: 4,
            }),
            rms_epsilon: epsilon,
            zero_centered_norm: false,
            embedding,
            lm_head: None,
            layers: std::slice::from_ref(&layer),
            final_norm: &final_norm,
            identity,
        })
        .unwrap();
        // Simulate digests copied from package admission, before load-time verification.
        let admitted = crate::AdmittedExternalCausalLmDigests {
            model_blake3: *blake3::hash(&external.model_bytes).as_bytes(),
            weights_blake3: external.weights_blake3,
        };
        let verified = crate::verify_external_causal_lm(
            &external.model_bytes,
            &external.weights_bytes,
            admitted,
        )
        .unwrap();
        assert_eq!(verified.weights_blake3, external.weights_blake3);
        let mut corrupt = external.weights_bytes.clone();
        corrupt[0] ^= 1;
        assert!(
            crate::verify_external_causal_lm(&external.model_bytes, &corrupt, admitted,).is_err()
        );
        let mut rewired_model = external.model_bytes.clone();
        let last = rewired_model.len() - 1;
        rewired_model[last] ^= 1;
        assert!(
            crate::verify_external_causal_lm(&rewired_model, &external.weights_bytes, admitted,)
                .is_err()
        );
        let oversized_past = 16_777_216;
        assert!(
            encode_result(
                1,
                oversized_past,
                layer,
                Some(crate::RotaryEmbedding {
                    theta: rope_theta,
                    dimensions: 4,
                }),
                None,
            )
            .is_err()
        );
        let large_external = crate::encode_external_causal_lm(crate::CausalLmModel {
            tokens: 1,
            past_tokens: oversized_past,
            n_head: 4,
            n_kv_head: 2,
            head_dim: 4,
            rotary: Some(crate::RotaryEmbedding {
                theta: rope_theta,
                dimensions: 4,
            }),
            rms_epsilon: epsilon,
            zero_centered_norm: false,
            embedding,
            lm_head: None,
            layers: std::slice::from_ref(&layer),
            final_norm: &final_norm,
            identity,
        })
        .unwrap();
        assert!(large_external.weights_bytes.len() > 64 * 1024 * 1024);
        assert!(
            crate::encode_external_causal_lm(crate::CausalLmModel {
                tokens: 1,
                past_tokens: i64::MAX as usize - 1,
                n_head: 4,
                n_kv_head: 2,
                head_dim: 4,
                rotary: Some(crate::RotaryEmbedding {
                    theta: rope_theta,
                    dimensions: 4,
                }),
                rms_epsilon: epsilon,
                zero_centered_norm: false,
                embedding,
                lm_head: None,
                layers: std::slice::from_ref(&layer),
                final_norm: &final_norm,
                identity,
            })
            .is_err()
        );
        let external_dir = std::env::temp_dir().join(format!(
            "tritium-onnx-causal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&external_dir).unwrap();
        let external_model_path = external_dir.join("model.onnx");
        std::fs::write(&external_model_path, &external.model_bytes).unwrap();
        std::fs::write(external_dir.join("weights.bin"), &external.weights_bytes).unwrap();
        let mut external_session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_file(&external_model_path)
            .unwrap();
        let external_tokens = Tensor::from_array(([2], vec![0_i64, 1].into_boxed_slice())).unwrap();
        let external_outputs = external_session
            .run(ort::inputs![&external_tokens])
            .unwrap();
        let (_, external_logits) = external_outputs[0].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(external_logits, &expected_logits, 2.0e-5);
        drop(external_outputs);
        drop(external_session);
        std::fs::remove_dir_all(external_dir).unwrap();
        let diagnostics = crate::diagnose_unsupported_graph(&prompt_model).unwrap();
        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        let mut prompt = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&prompt_model)
            .unwrap();
        let prompt_tensor = Tensor::from_array(([2], prompt_tokens.into_boxed_slice())).unwrap();
        let outputs = prompt.run(ort::inputs![&prompt_tensor]).unwrap();
        let (_, logits) = outputs[0].try_extract_tensor::<f32>().unwrap();
        let (_, present_k) = outputs[1].try_extract_tensor::<f32>().unwrap();
        let (_, present_v) = outputs[2].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(logits, &expected_logits, 2.0e-5);
        assert_eq!(
            greedy_last_token(logits, 3),
            greedy_last_token(&expected_logits, 3)
        );
        assert_f32_close(present_k, &expected_k, 2.0e-5);
        assert_f32_close(present_v, &expected_v, 2.0e-5);

        let decode_tokens = vec![2_i64];
        let (expected_logits, expected_k, expected_v) = reference_causal_lm(
            &decode_tokens,
            &expected_k,
            &expected_v,
            &dense_embedding,
            &dense_q,
            &dense_k,
            &dense_v,
            &dense_o,
            None,
            &dense_gate,
            &dense_up,
            &dense_down,
            &attention_norm,
            &query_norm,
            &key_norm,
            &ffn_norm,
            None,
            None,
            crate::CausalActivation::SwiGlu,
            &final_norm,
            None,
            4,
            2,
            4,
            4,
            rope_theta,
            epsilon,
            false,
        );
        let decode_model = encode(1, 2);
        let diagnostics = crate::diagnose_unsupported_graph(&decode_model).unwrap();
        assert!(diagnostics.is_empty(), "{diagnostics:#?}");
        let mut decode = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&decode_model)
            .unwrap();
        let tokens_tensor = Tensor::from_array(([1], decode_tokens.into_boxed_slice())).unwrap();
        let k_tensor =
            Tensor::from_array(([2, 2, 4], present_k.to_vec().into_boxed_slice())).unwrap();
        let v_tensor =
            Tensor::from_array(([2, 2, 4], present_v.to_vec().into_boxed_slice())).unwrap();
        let outputs = decode
            .run(ort::inputs![&tokens_tensor, &k_tensor, &v_tensor])
            .unwrap();
        let (_, logits) = outputs[0].try_extract_tensor::<f32>().unwrap();
        let (_, present_k) = outputs[1].try_extract_tensor::<f32>().unwrap();
        let (_, present_v) = outputs[2].try_extract_tensor::<f32>().unwrap();
        assert_f32_close(logits, &expected_logits, 2.0e-5);
        assert_eq!(
            greedy_last_token(logits, 3),
            greedy_last_token(&expected_logits, 3)
        );
        assert_f32_close(present_k, &expected_k, 2.0e-5);
        assert_f32_close(present_v, &expected_v, 2.0e-5);
    }

    fn assert_f32_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index}: {actual} != {expected} within {tolerance}"
            );
        }
    }

    fn greedy_last_token(logits: &[f32], vocabulary: usize) -> usize {
        logits[logits.len() - vocabulary..]
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .unwrap()
            .0
    }
}
