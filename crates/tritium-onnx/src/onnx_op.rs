//! Layer 2 — the `ort` 2.x custom operator wrapping the always-on Layer-1 kernel.
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

use ort::error::Error as OrtError;
use ort::operator::{
    Kernel, KernelAttributes, KernelContext, Operator, OperatorDomain, OperatorInput,
    OperatorOutput,
};
use ort::value::TensorElementType;
use tritium_core::TernaryFormat;

use crate::ternary_mpgemm_kernel;

/// The ONNX node op type this operator registers.
pub const ONNX_OP_NAME: &str = "TritiumTernaryMpGemm";

/// The custom-operator domain Tritium registers [`ONNX_OP_NAME`] under.
pub const ONNX_DOMAIN: &str = "com.tritium";

/// Node-attribute name for the contraction dimension `K`.
pub const ATTR_K: &str = "K";

/// Node-attribute name for the packing format (`0` = TQ2_0, `1` = TQ1_0).
pub const ATTR_FORMAT: &str = "format";

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
        let k: i64 = attributes.get(ATTR_K).ok_or_else(|| {
            OrtError::new(format!("tritium-onnx: missing i64 attribute `{ATTR_K}`"))
        })?;
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
        let format = format_from_attr(format_code)?;
        Ok(Box::new(TritiumTernaryMpGemmKernel {
            k: k as usize,
            format,
        }))
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
        let m = act_shape[0] as usize;
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
/// [`TritiumTernaryMpGemmOp`] registered, ready to pass to
/// [`ort::session::builder::SessionBuilder::with_operators`].
///
/// # Errors
/// An [`ort::Error`] if the domain cannot be created or the operator cannot be
/// added (e.g. the native runtime failed to initialize).
pub fn tritium_operator_domain() -> ort::Result<OperatorDomain> {
    OperatorDomain::new(ONNX_DOMAIN)?.add(TritiumTernaryMpGemmOp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use prost::Message;
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

    #[derive(Clone, PartialEq, Message)]
    struct ModelProto {
        #[prost(int64, tag = "1")]
        ir_version: i64,
        #[prost(string, tag = "2")]
        producer_name: String,
        #[prost(message, optional, tag = "7")]
        graph: Option<GraphProto>,
        #[prost(message, repeated, tag = "8")]
        opset_import: Vec<OperatorSetIdProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct OperatorSetIdProto {
        #[prost(string, tag = "1")]
        domain: String,
        #[prost(int64, tag = "2")]
        version: i64,
    }

    #[derive(Clone, PartialEq, Message)]
    struct GraphProto {
        #[prost(message, repeated, tag = "1")]
        node: Vec<NodeProto>,
        #[prost(string, tag = "2")]
        name: String,
        #[prost(message, repeated, tag = "11")]
        input: Vec<ValueInfoProto>,
        #[prost(message, repeated, tag = "12")]
        output: Vec<ValueInfoProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct NodeProto {
        #[prost(string, repeated, tag = "1")]
        input: Vec<String>,
        #[prost(string, repeated, tag = "2")]
        output: Vec<String>,
        #[prost(string, tag = "3")]
        name: String,
        #[prost(string, tag = "4")]
        op_type: String,
        #[prost(message, repeated, tag = "5")]
        attribute: Vec<AttributeProto>,
        #[prost(string, tag = "7")]
        domain: String,
    }

    #[derive(Clone, PartialEq, Message)]
    struct AttributeProto {
        #[prost(string, tag = "1")]
        name: String,
        #[prost(int64, tag = "3")]
        value: i64,
        #[prost(int32, tag = "20")]
        kind: i32,
    }

    #[derive(Clone, PartialEq, Message)]
    struct ValueInfoProto {
        #[prost(string, tag = "1")]
        name: String,
        #[prost(message, optional, tag = "2")]
        r#type: Option<TypeProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct TypeProto {
        #[prost(message, optional, tag = "1")]
        tensor_type: Option<TensorTypeProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct TensorTypeProto {
        #[prost(int32, tag = "1")]
        elem_type: i32,
        #[prost(message, optional, tag = "2")]
        shape: Option<TensorShapeProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct TensorShapeProto {
        #[prost(message, repeated, tag = "1")]
        dim: Vec<TensorDimensionProto>,
    }

    #[derive(Clone, PartialEq, Message)]
    struct TensorDimensionProto {
        #[prost(int64, tag = "1")]
        dim_value: i64,
    }

    fn tensor_value(name: &str, elem_type: i32, dimensions: &[usize]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_owned(),
            r#type: Some(TypeProto {
                tensor_type: Some(TensorTypeProto {
                    elem_type,
                    shape: Some(TensorShapeProto {
                        dim: dimensions
                            .iter()
                            .map(|&dimension| TensorDimensionProto {
                                dim_value: dimension as i64,
                            })
                            .collect(),
                    }),
                }),
            }),
        }
    }

    fn session_model_bytes(m: usize, n: usize, k: usize, packed: usize) -> Vec<u8> {
        ModelProto {
            ir_version: 10,
            producer_name: "tritium-onnx-test".to_owned(),
            graph: Some(GraphProto {
                node: vec![NodeProto {
                    input: vec!["act".to_owned(), "packed".to_owned(), "scales".to_owned()],
                    output: vec!["out".to_owned()],
                    name: "ternary".to_owned(),
                    op_type: ONNX_OP_NAME.to_owned(),
                    attribute: vec![
                        AttributeProto {
                            name: ATTR_K.to_owned(),
                            value: k as i64,
                            kind: 2,
                        },
                        AttributeProto {
                            name: ATTR_FORMAT.to_owned(),
                            value: 0,
                            kind: 2,
                        },
                    ],
                    domain: ONNX_DOMAIN.to_owned(),
                }],
                name: "tritium-session-test".to_owned(),
                input: vec![
                    tensor_value("act", 1, &[m, k]),
                    tensor_value("packed", 2, &[packed]),
                    tensor_value("scales", 1, &[n]),
                ],
                output: vec![tensor_value("out", 1, &[m, n])],
            }),
            opset_import: vec![
                OperatorSetIdProto {
                    domain: String::new(),
                    version: 21,
                },
                OperatorSetIdProto {
                    domain: ONNX_DOMAIN.to_owned(),
                    version: 1,
                },
            ],
        }
        .encode_to_vec()
    }

    /// Full end-to-end ONNX session: serialize a real opset-1 custom-domain
    /// graph, load it through ONNX Runtime, execute it and compare the result
    /// bit-exactly with Tritium's reference kernel.
    #[test]
    fn end_to_end_session_matches_reference() {
        use ort::value::Tensor;

        // A small TQ2_0 case: M=1, N=1, K=256, weights all +1, scale 1.0 -> the
        // output is the sum of the activation row.
        let m = 1usize;
        let n = 1usize;
        let k = 256usize;
        let format = TernaryFormat::Tq2_0;
        let act: Vec<f32> = (0..k).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let scales = vec![1.0f32];
        let nb = num_blocks(k);
        let unit = vec![f16::ONE; nb];
        let mut packed = vec![0u8; nb * block_bytes(format)];
        let trits = vec![Trit::POS; k];
        pack_tq2_0_row(&trits, &unit, &mut packed).unwrap();

        let expected =
            ternary_mpgemm_kernel(&act, &packed, &scales, m, k, format).expect("layer-1 ref");

        let model = session_model_bytes(m, n, k, packed.len());
        let mut session = ort::session::Session::builder()
            .unwrap()
            .with_operators(tritium_operator_domain().unwrap())
            .unwrap()
            .commit_from_memory(&model)
            .unwrap();
        assert_eq!(session.opset_for_domain(ONNX_DOMAIN).unwrap(), 1);

        let act_t = Tensor::from_array(([m, k], act.into_boxed_slice())).unwrap();
        let packed_t = Tensor::from_array(([packed.len()], packed.into_boxed_slice())).unwrap();
        let scales_t = Tensor::from_array(([n], scales.into_boxed_slice())).unwrap();
        let outputs = session
            .run(ort::inputs![&act_t, &packed_t, &scales_t])
            .unwrap();
        let (_, got) = outputs[0].try_extract_tensor::<f32>().unwrap();
        assert_eq!(got, expected.as_slice(), "e2e onnx output bit-exact");
    }
}
