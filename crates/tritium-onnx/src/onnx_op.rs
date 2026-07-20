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
    ATTR_FORMAT, ATTR_K, ONNX_DOMAIN, ONNX_EMBEDDING_OP_NAME, ONNX_OP_NAME,
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
/// [`TritiumTernaryMpGemmOp`] and [`TritiumTernaryEmbeddingOp`] registered,
/// ready to pass to
/// [`ort::session::builder::SessionBuilder::with_operators`].
///
/// # Errors
/// An [`ort::Error`] if the domain cannot be created or the operator cannot be
/// added (e.g. the native runtime failed to initialize).
pub fn tritium_operator_domain() -> ort::Result<OperatorDomain> {
    OperatorDomain::new(ONNX_DOMAIN)?
        .add(TritiumTernaryMpGemmOp)?
        .add(TritiumTernaryEmbeddingOp)
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
}
