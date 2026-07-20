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
    ATTR_FORMAT, ATTR_HEAD_DIM, ATTR_K, ATTR_N_HEAD, ATTR_N_KV_HEAD, ATTR_PAST_TOKENS, ONNX_DOMAIN,
    ONNX_EMBEDDING_OP_NAME, ONNX_KV_ATTENTION_OP_NAME, ONNX_OP_NAME, kv_attention_kernel,
    ternary_embedding_kernel, ternary_mpgemm_kernel,
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
        2
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> ort::Result<Box<dyn Kernel>> {
        Ok(Box::new(TritiumKvAttentionKernel {
            n_head: usize_attribute(attributes, ATTR_N_HEAD, false)?,
            n_kv_head: usize_attribute(attributes, ATTR_N_KV_HEAD, false)?,
            head_dim: usize_attribute(attributes, ATTR_HEAD_DIM, false)?,
            past_tokens: usize_attribute(attributes, ATTR_PAST_TOKENS, true)?,
        }))
    }
}

/// Per-node cache-aware attention kernel with frozen head geometry.
#[derive(Debug, Clone, Copy)]
pub struct TritiumKvAttentionKernel {
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    past_tokens: usize,
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
        let total_tokens = self
            .past_tokens
            .checked_add(query_tokens)
            .ok_or_else(|| OrtError::new("tritium-onnx: KV cache token count overflow"))?;
        if usize::try_from(k_shape[0]) != Ok(total_tokens) {
            return Err(OrtError::new(format!(
                "tritium-onnx: cache token count {} does not equal past {} + query {query_tokens}",
                k_shape[0], self.past_tokens
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
            self.past_tokens,
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
        .add(TritiumKvAttentionOp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tritium_core::Trit;
    use tritium_format::{num_blocks, pack_tq1_0_row, pack_tq2_0_row};
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
        let attention = TritiumKvAttentionOp;
        assert_eq!(attention.name(), ONNX_KV_ATTENTION_OP_NAME);
        assert_eq!(attention.inputs().len(), 3, "q, k_cache, v_cache");
        assert_eq!(attention.outputs().len(), 1, "context");
    }

    #[test]
    fn kv_attention_kernel_runs_prompt_and_cached_decode() {
        let prompt = TritiumKvAttentionKernel {
            n_head: 1,
            n_kv_head: 1,
            head_dim: 1,
            past_tokens: 0,
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
            past_tokens: 2,
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

    fn rms_norm(input: &[f32], weight: &[f32], epsilon: f32) -> Vec<f32> {
        let width = weight.len();
        input
            .chunks_exact(width)
            .flat_map(|row| {
                let denominator =
                    (row.iter().map(|value| value * value).sum::<f32>() / width as f32 + epsilon)
                        .sqrt();
                row.iter()
                    .zip(weight)
                    .map(move |(value, weight)| value * weight / denominator)
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
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let hidden: Vec<f32> = tokens
            .iter()
            .flat_map(|&token| embedding[usize::try_from(token).unwrap()].iter().copied())
            .collect();
        let attention_input = rms_norm(&hidden, attention_norm, epsilon);
        let query = apply_rope(
            &rms_norm(&dense_project(&attention_input, q), query_norm, epsilon),
            tokens.len(),
            n_head,
            head_dim,
            rotary_dim,
            past_k.len() / (n_kv_head * head_dim),
            rope_theta,
        );
        let current_k = apply_rope(
            &rms_norm(&dense_project(&attention_input, k), key_norm, epsilon),
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
            rms_norm(&gated_context, weight, epsilon)
        });
        let attention_output = dense_project(&output_input, o);
        let post_attention: Vec<f32> = hidden
            .iter()
            .zip(attention_output)
            .map(|(residual, update)| residual + update)
            .collect();
        let ffn_input = rms_norm(&post_attention, ffn_norm, epsilon);
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
            rms_norm(&activated, weight, epsilon)
        });
        let ffn_output = dense_project(&down_input, down);
        let post_ffn: Vec<f32> = post_attention
            .iter()
            .zip(ffn_output)
            .map(|(residual, update)| residual + update)
            .collect();
        let final_hidden = rms_norm(&post_ffn, final_norm, epsilon);
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
        let gate_packed = pack_rows(&trits(&dense_gate), format);
        let up_packed = pack_rows(&trits(&dense_up), format);
        let down_packed = pack_rows(&trits(&dense_down), format);
        let embedding_scales = [1.0; 3];
        let sixteen_scales = vec![1.0; 16];
        let eight_scales = vec![1.0; 8];
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
            query: packed_matrix(16, 16, &q_packed, &sixteen_scales, format),
            key: packed_matrix(8, 16, &k_packed, &eight_scales, format),
            value: packed_matrix(8, 16, &v_packed, &eight_scales, format),
            attention_output: packed_matrix(16, 16, &o_packed, &sixteen_scales, format),
            attention_gate: None,
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
        );
        let attention_gate_model = encode_result(
            2,
            0,
            crate::CausalLmDecoderLayer {
                attention_gate: Some(packed_matrix(16, 16, &q_packed, &sixteen_scales, format)),
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
