//! Deterministic ONNX protobuf serialization for Tritium inference graphs.

use std::collections::BTreeMap;

use half::f16;
use prost::Message;
use tritium_core::{TernaryFormat, Trit};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};

use crate::{
    ATTR_FORMAT, ATTR_HEAD_DIM, ATTR_K, ATTR_N_HEAD, ATTR_N_KV_HEAD, ATTR_PAST_TOKENS, ONNX_DOMAIN,
    ONNX_EMBEDDING_OP_NAME, ONNX_KV_ATTENTION_OP_NAME, ONNX_OP_NAME,
};

const ONNX_IR_VERSION: i64 = 10;
const ONNX_OPSET: i64 = 21;
const TRITIUM_OPSET: i64 = 1;
const TENSOR_FLOAT: i32 = 1;
const TENSOR_UINT8: i32 = 2;
const TENSOR_INT64: i32 = 7;
const ATTRIBUTE_INT: i32 = 2;
const ATTRIBUTE_INTS: i32 = 7;
const EXTERNAL_DATA: i32 = 1;
const EXTERNAL_WEIGHTS_FILE: &str = "weights.bin";
const EXTERNAL_ALIGNMENT: usize = 64;
const MAX_MODEL_BYTES: usize = 64 * 1024 * 1024;

/// A deterministic tied packed embedding/head graph.
///
/// The emitted model accepts one fixed-length `tokens` tensor, gathers its
/// packed embedding rows, and multiplies the hidden states by the same packed
/// table to produce `logits`. `packed` and `scales` become ONNX initializers and
/// are referenced by both nodes, preserving physical tying without a dense
/// shadow or duplicate initializer.
#[derive(Debug, Clone, Copy)]
pub struct TiedEmbeddingHeadModel<'a> {
    /// Number of input tokens in the fixed test/export graph.
    pub tokens: usize,
    /// Vocabulary/table row count.
    pub vocab: usize,
    /// Hidden/table column count.
    pub hidden: usize,
    /// Packed TQ2_0 or TQ1_0 table bytes, output-major.
    pub packed: &'a [u8],
    /// One finite nonnegative scale per vocabulary row.
    pub scales: &'a [f32],
    /// Canonical packed ternary format.
    pub format: TernaryFormat,
    /// Non-empty immutable source-model identity.
    pub source_model_id: &'a str,
    /// Non-empty conversion recipe identity.
    pub recipe_id: &'a str,
    /// Non-empty packed artifact/package identity.
    pub package_id: &'a str,
}

/// Complete schema-v2 identity binding for a Tritium ONNX artifact.
#[derive(Debug, Clone, Copy)]
pub struct OnnxArtifactIdentityV2<'a> {
    /// Immutable source-model identity, including resolved revision.
    pub source_model_id: &'a str,
    /// Tokenizer identity, including resolved revision.
    pub tokenizer_id: &'a str,
    /// Conversion recipe identity.
    pub recipe_id: &'a str,
    /// Tritium build/source identity that produced the graph.
    pub tritium_build_id: &'a str,
    /// Packed artifact/package identity.
    pub package_id: &'a str,
    /// Identity of the exact in-scope converted coverage ledger.
    pub converted_coverage_id: &'a str,
    /// Identity of the explicit deferred/preserved coverage ledger.
    pub deferred_coverage_id: &'a str,
}

/// A schema-v2 tied embedding/head graph with complete artifact identity.
///
/// This additive type leaves [`TiedEmbeddingHeadModel`] and its schema-v1 wire
/// format source-compatible and readable while allowing release artifacts to
/// bind the full conversion provenance contract.
#[derive(Debug, Clone, Copy)]
pub struct TiedEmbeddingHeadModelV2<'a> {
    /// Number of input tokens in the fixed graph.
    pub tokens: usize,
    /// Vocabulary/table row count.
    pub vocab: usize,
    /// Hidden/table column count.
    pub hidden: usize,
    /// Packed TQ2_0 or TQ1_0 table bytes, output-major.
    pub packed: &'a [u8],
    /// One finite nonnegative scale per vocabulary row.
    pub scales: &'a [f32],
    /// Canonical packed ternary format.
    pub format: TernaryFormat,
    /// Complete schema-v2 artifact identity.
    pub identity: OnnxArtifactIdentityV2<'a>,
}

/// One packed output-major ternary matrix consumed directly by Tritium ONNX
/// custom operators.
#[derive(Debug, Clone, Copy)]
pub struct PackedTernaryMatrix<'a> {
    /// Output row count.
    pub rows: usize,
    /// Input/contraction column count.
    pub columns: usize,
    /// Canonical TQ2_0 or TQ1_0 packed rows.
    pub packed: &'a [u8],
    /// One finite nonnegative scale per output row.
    pub scales: &'a [f32],
    /// Canonical packed ternary format.
    pub format: TernaryFormat,
}

/// Qwen-style full-head rotary position embedding parameters.
#[derive(Debug, Clone, Copy)]
pub struct RotaryEmbedding {
    /// Positive finite frequency base (`rope_theta`).
    pub theta: f32,
}

/// Packed weights and preserved RMSNorm vectors for one causal decoder layer.
#[derive(Debug, Clone, Copy)]
pub struct CausalLmDecoderLayer<'a> {
    /// Pre-attention RMSNorm weight.
    pub attention_norm: &'a [f32],
    /// Optional per-query-head RMSNorm weight applied before RoPE.
    pub query_norm: Option<&'a [f32]>,
    /// Optional per-key-head RMSNorm weight applied before RoPE/cache write.
    pub key_norm: Option<&'a [f32]>,
    /// Query projection.
    pub query: PackedTernaryMatrix<'a>,
    /// Key projection.
    pub key: PackedTernaryMatrix<'a>,
    /// Value projection.
    pub value: PackedTernaryMatrix<'a>,
    /// Attention output projection.
    pub attention_output: PackedTernaryMatrix<'a>,
    /// Pre-FFN RMSNorm weight.
    pub ffn_norm: &'a [f32],
    /// SwiGLU gate projection.
    pub gate: PackedTernaryMatrix<'a>,
    /// SwiGLU up projection.
    pub up: PackedTernaryMatrix<'a>,
    /// SwiGLU down projection.
    pub down: PackedTernaryMatrix<'a>,
}

/// Fixed-shape packed decoder-only causal language-model graph.
///
/// Prompt graphs set `past_tokens` to zero and consume token IDs only. Cached
/// decode graphs consume one `past_k.{layer}` and `past_v.{layer}` tensor per
/// layer and return complete `present_k.{layer}`/`present_v.{layer}` tensors.
#[derive(Debug, Clone, Copy)]
pub struct CausalLmModel<'a> {
    /// Query token count fixed into this graph.
    pub tokens: usize,
    /// Prefix-cache token count fixed into this graph.
    pub past_tokens: usize,
    /// Query attention head count.
    pub n_head: usize,
    /// Key/value attention head count.
    pub n_kv_head: usize,
    /// Elements per attention head.
    pub head_dim: usize,
    /// Optional rotary position embedding applied to Q and newly produced K.
    pub rotary: Option<RotaryEmbedding>,
    /// Positive finite RMSNorm epsilon.
    pub rms_epsilon: f32,
    /// Tied token embedding and language-model head table.
    pub embedding: PackedTernaryMatrix<'a>,
    /// Ordered decoder layers; at least one is required.
    pub layers: &'a [CausalLmDecoderLayer<'a>],
    /// Final RMSNorm weight.
    pub final_norm: &'a [f32],
    /// Complete artifact identity.
    pub identity: OnnxArtifactIdentityV2<'a>,
}

impl<'a> TiedEmbeddingHeadModelV2<'a> {
    fn legacy(self) -> TiedEmbeddingHeadModel<'a> {
        TiedEmbeddingHeadModel {
            tokens: self.tokens,
            vocab: self.vocab,
            hidden: self.hidden,
            packed: self.packed,
            scales: self.scales,
            format: self.format,
            source_model_id: self.identity.source_model_id,
            recipe_id: self.identity.recipe_id,
            package_id: self.identity.package_id,
        }
    }
}

/// A deterministic ONNX model plus its content-bound external initializer file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalOnnxModel {
    /// Serialized `model.onnx` bytes.
    pub model_bytes: Vec<u8>,
    /// Serialized `weights.bin` bytes referenced by the model.
    pub weights_bytes: Vec<u8>,
    /// BLAKE3 digest of `weights_bytes`, also embedded in model metadata.
    pub weights_blake3: [u8; 32],
}

/// Receipt from strict external ONNX model/data verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExternalOnnxModel {
    /// BLAKE3 digest of the canonical serialized ONNX protobuf.
    pub model_blake3: [u8; 32],
    /// BLAKE3 digest of the canonical external initializer bytes.
    pub weights_blake3: [u8; 32],
    /// Exact external initializer byte count.
    pub weights_bytes: usize,
    /// Fixed input token count.
    pub tokens: usize,
    /// Vocabulary row count.
    pub vocab: usize,
    /// Hidden column count.
    pub hidden: usize,
    /// Bound source model identity.
    pub source_model_id: String,
    /// Bound conversion recipe identity.
    pub recipe_id: String,
    /// Bound artifact package identity.
    pub package_id: String,
}

/// Owned identity receipt from schema-v2 external ONNX verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedOnnxArtifactIdentityV2 {
    /// Bound source model identity.
    pub source_model_id: String,
    /// Bound tokenizer identity.
    pub tokenizer_id: String,
    /// Bound conversion recipe identity.
    pub recipe_id: String,
    /// Bound Tritium build/source identity.
    pub tritium_build_id: String,
    /// Bound artifact package identity.
    pub package_id: String,
    /// Bound converted coverage-ledger identity.
    pub converted_coverage_id: String,
    /// Bound deferred/preserved coverage-ledger identity.
    pub deferred_coverage_id: String,
}

/// Receipt from strict schema-v2 external ONNX model/data verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExternalOnnxModelV2 {
    /// Verified graph, external data and geometry receipt.
    pub model: VerifiedExternalOnnxModel,
    /// Complete bound artifact identity.
    pub identity: VerifiedOnnxArtifactIdentityV2,
}

/// Receipt from strict external causal-LM graph/data verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedExternalCausalLmModel {
    /// BLAKE3 digest of canonical `model.onnx` bytes.
    pub model_blake3: [u8; 32],
    /// BLAKE3 digest of authenticated `weights.bin` bytes.
    pub weights_blake3: [u8; 32],
    /// Exact external-data byte count.
    pub weights_bytes: usize,
    /// Fixed query-token count.
    pub tokens: usize,
    /// Fixed prefix-cache token count.
    pub past_tokens: usize,
    /// Decoder-layer count.
    pub layers: usize,
    /// Complete bound artifact identity.
    pub identity: VerifiedOnnxArtifactIdentityV2,
}

/// Exact causal-LM file identities copied from an independently trusted package manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmittedExternalCausalLmDigests {
    /// Admitted `model.onnx` BLAKE3.
    pub model_blake3: [u8; 32],
    /// Admitted `weights.bin` BLAKE3.
    pub weights_blake3: [u8; 32],
}

/// Category of one unsupported ONNX graph item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnsupportedGraphItemKind {
    /// Unsupported node domain or operator.
    Node,
    /// Unsupported node attribute or attribute representation.
    Attribute,
    /// Unsupported tensor element type.
    Dtype,
    /// Unresolved conversion-coverage item.
    Coverage,
}

/// One typed, actionable unsupported-graph diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedGraphDiagnostic {
    /// Stable diagnostic category.
    pub kind: UnsupportedGraphItemKind,
    /// Node, attribute, tensor, or coverage path rejected.
    pub subject: String,
    /// Stable human-readable rejection reason.
    pub reason: String,
}

/// Validation or serialization errors from Tritium ONNX model encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OnnxModelError {
    /// A required scalar dimension is zero.
    EmptyDimension(&'static str),
    /// A required identity string is empty.
    EmptyIdentity(&'static str),
    /// The format is not a portable packed ternary format.
    UnsupportedFormat(TernaryFormat),
    /// The scale count does not match the vocabulary.
    ScaleCount {
        /// Expected vocabulary row count.
        expected: usize,
        /// Actual scale count.
        got: usize,
    },
    /// A scale is negative or non-finite.
    InvalidScale {
        /// Index of the rejected scale.
        index: usize,
    },
    /// The packed byte count does not match the declared table.
    PackedBytes {
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        got: usize,
    },
    /// A derived count cannot be represented by the host or ONNX `int64`.
    ShapeOverflow(&'static str),
    /// A packed row is malformed or carries a non-unit internal block scale.
    InvalidPackedRow {
        /// Rejected table row.
        row: usize,
        /// Stable diagnostic from the packed decoder.
        reason: String,
    },
    /// Serialized protobuf or Tritium metadata is malformed.
    InvalidModel(String),
    /// The supplied external data does not match its model binding.
    ExternalDataMismatch(String),
}

impl core::fmt::Display for OnnxModelError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyDimension(name) => write!(formatter, "ONNX {name} must be positive"),
            Self::EmptyIdentity(name) => write!(formatter, "ONNX {name} must be non-empty"),
            Self::UnsupportedFormat(format) => {
                write!(formatter, "ONNX export does not support {format}")
            }
            Self::ScaleCount { expected, got } => write!(
                formatter,
                "ONNX scale count {got} does not match vocabulary {expected}"
            ),
            Self::InvalidScale { index } => write!(
                formatter,
                "ONNX scale {index} must be finite and nonnegative"
            ),
            Self::PackedBytes { expected, got } => write!(
                formatter,
                "ONNX packed table has {got} bytes, expected {expected}"
            ),
            Self::ShapeOverflow(name) => write!(formatter, "ONNX {name} exceeds int64/usize"),
            Self::InvalidPackedRow { row, reason } => {
                write!(formatter, "ONNX packed row {row} is invalid: {reason}")
            }
            Self::InvalidModel(reason) => write!(formatter, "invalid Tritium ONNX model: {reason}"),
            Self::ExternalDataMismatch(reason) => {
                write!(formatter, "Tritium ONNX external data mismatch: {reason}")
            }
        }
    }
}

impl std::error::Error for OnnxModelError {}

/// Diagnose every unsupported node, attribute, dtype, and unresolved coverage
/// item visible in a serialized Tritium ONNX graph.
///
/// Diagnostics use deterministic category order: nodes, attributes, dtypes,
/// then coverage. An empty result means these support dimensions are admitted;
/// strict graph identity and external-data admission remain verifier duties.
///
/// # Errors
/// [`OnnxModelError::InvalidModel`] if protobuf is malformed, oversized, or has
/// no graph.
pub fn diagnose_unsupported_graph(
    model_bytes: &[u8],
) -> Result<Vec<UnsupportedGraphDiagnostic>, OnnxModelError> {
    if model_bytes.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    let protobuf = ModelProto::decode(model_bytes)
        .map_err(|error| OnnxModelError::InvalidModel(error.to_string()))?;
    let graph = protobuf
        .graph
        .as_ref()
        .ok_or_else(|| OnnxModelError::InvalidModel("graph is missing".to_owned()))?;
    let mut tritium_opsets = protobuf
        .opset_import
        .iter()
        .filter(|opset| opset.domain == ONNX_DOMAIN);
    let tritium_opset = tritium_opsets.next().map(|opset| opset.version);
    if tritium_opsets.next().is_some() {
        return Err(OnnxModelError::InvalidModel(format!(
            "duplicate opset import for domain {ONNX_DOMAIN}"
        )));
    }
    let mut standard_opsets = protobuf
        .opset_import
        .iter()
        .filter(|opset| opset.domain.is_empty());
    let standard_opset = standard_opsets.next().map(|opset| opset.version);
    if standard_opsets.next().is_some() {
        return Err(OnnxModelError::InvalidModel(
            "duplicate opset import for standard ONNX domain".to_owned(),
        ));
    }
    let mut diagnostics = Vec::new();

    for node in &graph.node {
        let supported = node_supported(node, tritium_opset, standard_opset);
        if !supported {
            let reason = if node.domain == ONNX_DOMAIN
                && node.op_type == ONNX_KV_ATTENTION_OP_NAME
                && tritium_opset != Some(2)
            {
                format!(
                    "operator requires {ONNX_DOMAIN} opset 2, imported {}",
                    tritium_opset
                        .map_or_else(|| "missing".to_owned(), |version| version.to_string())
                )
            } else if node.domain == ONNX_DOMAIN && !matches!(tritium_opset, Some(1 | 2)) {
                format!(
                    "unsupported {ONNX_DOMAIN} opset {}",
                    tritium_opset
                        .map_or_else(|| "missing".to_owned(), |version| version.to_string())
                )
            } else if node.domain.is_empty() && standard_opset != Some(ONNX_OPSET) {
                format!(
                    "unsupported standard ONNX opset {}",
                    standard_opset
                        .map_or_else(|| "missing".to_owned(), |version| version.to_string())
                )
            } else {
                format!("unsupported operator {}::{}", node.domain, node.op_type)
            };
            diagnostics.push(UnsupportedGraphDiagnostic {
                kind: UnsupportedGraphItemKind::Node,
                subject: diagnostic_subject(&node.name, "<unnamed node>"),
                reason,
            });
        }
    }
    for node in &graph.node {
        for attribute in &node.attribute {
            let subject = format!(
                "{}.{}",
                diagnostic_subject(&node.name, "<unnamed node>"),
                diagnostic_subject(&attribute.name, "<unnamed attribute>")
            );
            if !supported_attributes(node).contains(&attribute.name.as_str()) {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: "unsupported attribute".to_owned(),
                });
            } else if attribute.kind != expected_attribute_kind(&attribute.name) {
                let expected = expected_attribute_kind(&attribute.name);
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!(
                        "expected ONNX attribute type {expected}, got {}",
                        attribute.kind
                    ),
                });
            } else if attribute.name == ATTR_K && attribute.value <= 0 {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!("K must be positive, got {}", attribute.value),
                });
            } else if attribute.name == ATTR_FORMAT && !matches!(attribute.value, 0 | 1) {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!("unsupported format code {}", attribute.value),
                });
            } else if matches!(
                attribute.name.as_str(),
                ATTR_N_HEAD | ATTR_N_KV_HEAD | ATTR_HEAD_DIM
            ) && attribute.value <= 0
            {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!(
                        "{} must be positive, got {}",
                        attribute.name, attribute.value
                    ),
                });
            } else if attribute.name == ATTR_PAST_TOKENS && attribute.value < 0 {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!("past_tokens must be nonnegative, got {}", attribute.value),
                });
            } else if attribute.name == "perm" && !valid_rank_three_permutation(&attribute.ints) {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!(
                        "expected a rank-3 permutation of [0, 1, 2], got {:?}",
                        attribute.ints
                    ),
                });
            } else if node.op_type == "Softmax" && attribute.name == "axis" && attribute.value != -1
            {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!("attention softmax axis must be -1, got {}", attribute.value),
                });
            } else if node.op_type == "Concat" && attribute.name == "axis" {
                match concat_axis(node, graph) {
                    Some(expected) if attribute.value != expected => {
                        diagnostics.push(UnsupportedGraphDiagnostic {
                            kind: UnsupportedGraphItemKind::Attribute,
                            subject,
                            reason: format!(
                                "{} concat axis must be {expected}, got {}",
                                if expected == 0 { "KV cache" } else { "RoPE" },
                                attribute.value
                            ),
                        });
                    }
                    None => diagnostics.push(UnsupportedGraphDiagnostic {
                        kind: UnsupportedGraphItemKind::Attribute,
                        subject,
                        reason: "concat node has no supported semantic role".to_owned(),
                    }),
                    Some(_) => {}
                }
            } else if node.op_type == "ReduceMean"
                && attribute.name == "keepdims"
                && attribute.value != 1
            {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!(
                        "RMSNorm reduction must keep dimensions, got {}",
                        attribute.value
                    ),
                });
            }
        }
        let supported = node_supported(node, tritium_opset, standard_opset);
        if supported {
            for &required in supported_attributes(node) {
                let count = node
                    .attribute
                    .iter()
                    .filter(|attribute| attribute.name == required)
                    .count();
                let subject = format!(
                    "{}.{}",
                    diagnostic_subject(&node.name, "<unnamed node>"),
                    required
                );
                match count {
                    0 => diagnostics.push(UnsupportedGraphDiagnostic {
                        kind: UnsupportedGraphItemKind::Attribute,
                        subject,
                        reason: "missing required attribute".to_owned(),
                    }),
                    1 => {}
                    count => diagnostics.push(UnsupportedGraphDiagnostic {
                        kind: UnsupportedGraphItemKind::Attribute,
                        subject,
                        reason: format!("duplicate attribute appears {count} times"),
                    }),
                }
            }
            if node.op_type == ONNX_KV_ATTENTION_OP_NAME {
                let positive_value = |name: &str| {
                    let mut attributes = node.attribute.iter().filter(|attribute| {
                        attribute.name == name
                            && attribute.kind == ATTRIBUTE_INT
                            && attribute.value > 0
                    });
                    let value = attributes.next().map(|attribute| attribute.value);
                    if attributes.next().is_none() {
                        value
                    } else {
                        None
                    }
                };
                if let (Some(n_head), Some(n_kv_head)) =
                    (positive_value(ATTR_N_HEAD), positive_value(ATTR_N_KV_HEAD))
                    && n_head % n_kv_head != 0
                {
                    diagnostics.push(UnsupportedGraphDiagnostic {
                        kind: UnsupportedGraphItemKind::Attribute,
                        subject: format!(
                            "{}.{}",
                            diagnostic_subject(&node.name, "<unnamed node>"),
                            ATTR_N_KV_HEAD
                        ),
                        reason: format!(
                            "n_head {n_head} is not divisible by n_kv_head {n_kv_head}"
                        ),
                    });
                }
            }
        }
    }
    for initializer in &graph.initializer {
        let expected = match initializer.name.as_str() {
            name if name == "tritium.packed" || name.ends_with(".packed") => Some(TENSOR_UINT8),
            name if name == "tritium.scales"
                || name == "attention.scale"
                || name.ends_with(".scales")
                || name.ends_with(".weight")
                || name.ends_with(".epsilon")
                || name.ends_with(".attention_scale")
                || name.ends_with(".attention_mask")
                || name.ends_with(".cos")
                || name.ends_with(".sin") =>
            {
                Some(TENSOR_FLOAT)
            }
            name if name.ends_with(".axes")
                || name.ends_with(".shape")
                || name.ends_with("_shape")
                || name.ends_with(".gqa_repeats")
                || name.ends_with(".first_start")
                || name.ends_with(".first_end")
                || name.ends_with(".second_start")
                || name.ends_with(".second_end")
                || name.ends_with(".steps") =>
            {
                Some(TENSOR_INT64)
            }
            _ => None,
        };
        let subject = format!("initializer {}", initializer.name);
        match expected {
            Some(expected) => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                Some(initializer.data_type),
                expected,
            ),
            None => push_uncontracted_dtype_diagnostic(
                &mut diagnostics,
                subject,
                Some(initializer.data_type),
            ),
        }
    }
    for input in &graph.input {
        let subject = format!("input {}", input.name);
        match input.name.as_str() {
            "tokens" => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(input),
                TENSOR_INT64,
            ),
            "q" | "k_cache" | "v_cache" | "attention_mask" => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(input),
                TENSOR_FLOAT,
            ),
            name if name.starts_with("past_k.") || name.starts_with("past_v.") => {
                push_dtype_diagnostic(
                    &mut diagnostics,
                    subject,
                    tensor_elem_type(input),
                    TENSOR_FLOAT,
                )
            }
            _ => push_uncontracted_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(input),
            ),
        }
    }
    for output in &graph.output {
        let subject = format!("output {}", output.name);
        match output.name.as_str() {
            "logits" | "context" => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(output),
                TENSOR_FLOAT,
            ),
            name if name.starts_with("present_k.") || name.starts_with("present_v.") => {
                push_dtype_diagnostic(
                    &mut diagnostics,
                    subject,
                    tensor_elem_type(output),
                    TENSOR_FLOAT,
                )
            }
            _ => push_uncontracted_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(output),
            ),
        }
    }
    for value in &graph.value_info {
        let subject = format!("value_info {}", value.name);
        match value.name.as_str() {
            "hidden" => push_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(value),
                TENSOR_FLOAT,
            ),
            _ => push_uncontracted_dtype_diagnostic(
                &mut diagnostics,
                subject,
                tensor_elem_type(value),
            ),
        }
    }
    for entry in &protobuf.metadata_props {
        if let Some(subject) = entry.key.strip_prefix("tritium.coverage.unresolved.") {
            diagnostics.push(UnsupportedGraphDiagnostic {
                kind: UnsupportedGraphItemKind::Coverage,
                subject: diagnostic_subject(subject, "<unnamed coverage item>"),
                reason: format!("coverage item is unresolved: {}", entry.value),
            });
        }
    }
    Ok(diagnostics)
}

fn supported_attributes(node: &NodeProto) -> &'static [&'static str] {
    match (node.domain.as_str(), node.op_type.as_str()) {
        (ONNX_DOMAIN, ONNX_OP_NAME | ONNX_EMBEDDING_OP_NAME) => &[ATTR_K, ATTR_FORMAT],
        (ONNX_DOMAIN, ONNX_KV_ATTENTION_OP_NAME) => {
            &[ATTR_N_HEAD, ATTR_N_KV_HEAD, ATTR_HEAD_DIM, ATTR_PAST_TOKENS]
        }
        ("", "Transpose") => &["perm"],
        ("", "Softmax") => &["axis"],
        ("", "Concat") => &["axis"],
        ("", "ReduceMean") => &["keepdims"],
        (
            "",
            "MatMul" | "Mul" | "Add" | "Div" | "Sqrt" | "Sigmoid" | "Reshape" | "Tile" | "Identity"
            | "Slice" | "Neg",
        ) => &[],
        _ => &[],
    }
}

fn concat_axis(node: &NodeProto, graph: &GraphProto) -> Option<i64> {
    let producer = |value: &str| {
        graph
            .node
            .iter()
            .find(|candidate| candidate.output.iter().any(|output| output == value))
    };
    let rotary = node.input.len() == 2
        && producer(&node.input[0]).is_some_and(|candidate| candidate.op_type == "Neg")
        && producer(&node.input[1]).is_some_and(|candidate| candidate.op_type == "Slice");
    if rotary {
        return Some(-1);
    }
    let cache = node.input.len() == 2
        && node.output.len() == 1
        && graph.input.iter().any(|input| input.name == node.input[0])
        && graph
            .output
            .iter()
            .any(|output| output.name == node.output[0]);
    cache.then_some(0)
}

fn expected_attribute_kind(name: &str) -> i32 {
    if name == "perm" {
        ATTRIBUTE_INTS
    } else {
        ATTRIBUTE_INT
    }
}

fn valid_rank_three_permutation(values: &[i64]) -> bool {
    values.len() == 3 && values.iter().all(|axis| (0..3).contains(axis)) && {
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        sorted == [0, 1, 2]
    }
}

fn node_supported(
    node: &NodeProto,
    tritium_opset: Option<i64>,
    standard_opset: Option<i64>,
) -> bool {
    match (node.domain.as_str(), node.op_type.as_str()) {
        (ONNX_DOMAIN, ONNX_OP_NAME | ONNX_EMBEDDING_OP_NAME) => {
            matches!(tritium_opset, Some(1 | 2))
        }
        (ONNX_DOMAIN, ONNX_KV_ATTENTION_OP_NAME) => tritium_opset == Some(2),
        (
            "",
            "Transpose" | "MatMul" | "Mul" | "Add" | "Div" | "Sqrt" | "Sigmoid" | "Reshape"
            | "Tile" | "Identity" | "Slice" | "Neg" | "Concat" | "ReduceMean" | "Softmax",
        ) => standard_opset == Some(ONNX_OPSET),
        _ => false,
    }
}

fn diagnostic_subject(value: &str, fallback: &str) -> String {
    if value.is_empty() {
        fallback.to_owned()
    } else {
        value.to_owned()
    }
}

fn tensor_elem_type(value: &ValueInfoProto) -> Option<i32> {
    value
        .r#type
        .as_ref()
        .and_then(|kind| kind.tensor_type.as_ref())
        .map(|tensor| tensor.elem_type)
}

fn push_dtype_diagnostic(
    diagnostics: &mut Vec<UnsupportedGraphDiagnostic>,
    subject: String,
    actual: Option<i32>,
    expected: i32,
) {
    if actual != Some(expected) {
        diagnostics.push(UnsupportedGraphDiagnostic {
            kind: UnsupportedGraphItemKind::Dtype,
            subject,
            reason: actual.map_or_else(
                || format!("expected ONNX dtype {expected}, got missing type"),
                |actual| format!("expected ONNX dtype {expected}, got {actual}"),
            ),
        });
    }
}

fn push_uncontracted_dtype_diagnostic(
    diagnostics: &mut Vec<UnsupportedGraphDiagnostic>,
    subject: String,
    actual: Option<i32>,
) {
    diagnostics.push(UnsupportedGraphDiagnostic {
        kind: UnsupportedGraphItemKind::Dtype,
        subject,
        reason: actual.map_or_else(
            || "missing type has no supported contract for this tensor".to_owned(),
            |actual| format!("dtype {actual} has no supported contract for this tensor"),
        ),
    });
}

/// Encode a tied packed embedding/head graph as a deterministic ONNX model.
///
/// # Errors
/// [`OnnxModelError`] if dimensions, identities, scales, format, or packed byte
/// count violate the frozen `com.tritium` opset-1 contract.
pub fn encode_tied_embedding_head(
    model: TiedEmbeddingHeadModel<'_>,
) -> Result<Vec<u8>, OnnxModelError> {
    validate(&model)?;
    let scale_bytes = scale_bytes(model.scales);
    let packed_bytes = as_i64(model.packed.len(), "packed byte count")?;
    let vocab = as_i64(model.vocab, "vocabulary")?;
    encode_model(
        model,
        vec![
            inline_tensor(
                "tritium.packed",
                TENSOR_UINT8,
                vec![packed_bytes],
                model.packed.to_vec(),
            ),
            inline_tensor("tritium.scales", TENSOR_FLOAT, vec![vocab], scale_bytes),
        ],
        Vec::new(),
        "1",
    )
}

/// Encode a schema-v2 tied embedding/head graph with complete artifact identity.
///
/// # Errors
/// [`OnnxModelError`] if graph geometry, payload, or any identity is invalid.
pub fn encode_tied_embedding_head_v2(
    model: TiedEmbeddingHeadModelV2<'_>,
) -> Result<Vec<u8>, OnnxModelError> {
    validate_v2(&model)?;
    let legacy = model.legacy();
    let scale_bytes = scale_bytes(legacy.scales);
    let packed_bytes = as_i64(legacy.packed.len(), "packed byte count")?;
    let vocab = as_i64(legacy.vocab, "vocabulary")?;
    encode_model(
        legacy,
        vec![
            inline_tensor(
                "tritium.packed",
                TENSOR_UINT8,
                vec![packed_bytes],
                legacy.packed.to_vec(),
            ),
            inline_tensor("tritium.scales", TENSOR_FLOAT, vec![vocab], scale_bytes),
        ],
        identity_metadata_v2(model.identity),
        "2",
    )
}

/// Encode a packed decoder-only causal LM as a deterministic ONNX model.
///
/// Packed embedding/projection initializers remain compressed and execute via
/// `com.tritium` opset 1. RMSNorm, GQA expansion, causal attention, residuals,
/// SwiGLU, cache concatenation, and cache outputs use standard ONNX opset 21.
///
/// # Errors
/// [`OnnxModelError`] if model geometry, packed payloads, preserved vectors,
/// cache sizes, epsilon, or artifact identity violate the causal-LM contract.
pub fn encode_causal_lm(model: CausalLmModel<'_>) -> Result<Vec<u8>, OnnxModelError> {
    validate_causal_lm(&model)?;
    validate_causal_initializer_budget(&model)?;
    let (protobuf, weights) = build_causal_lm_graph(model, false)?;
    debug_assert!(weights.is_none());
    let encoded = protobuf.encode_to_vec();
    if encoded.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    Ok(encoded)
}

/// Encode a causal LM with weight/value initializers in authenticated `weights.bin`.
///
/// Initializers retain packed ternary representation; no dense weight shadow is
/// emitted. Shape-driving `int64` constants remain inline for ONNX shape
/// inference. External ranges are 64-byte aligned and graph metadata binds
/// exact byte length plus BLAKE3 digest.
///
/// # Errors
/// [`OnnxModelError`] if model validation, range arithmetic, allocation, or
/// protobuf bounds fail.
pub fn encode_external_causal_lm(
    model: CausalLmModel<'_>,
) -> Result<ExternalOnnxModel, OnnxModelError> {
    validate_causal_lm(&model)?;
    let (mut protobuf, weights) = build_causal_lm_graph(model, true)?;
    let weights_bytes = weights.ok_or_else(|| {
        OnnxModelError::InvalidModel("external graph builder returned inline storage".to_owned())
    })?;
    let weights_blake3 = *blake3::hash(&weights_bytes).as_bytes();
    let digest = blake3::Hash::from_bytes(weights_blake3)
        .to_hex()
        .to_string();
    protobuf.metadata_props.extend([
        metadata("tritium.external_data.file", EXTERNAL_WEIGHTS_FILE),
        metadata(
            "tritium.external_data.bytes",
            &weights_bytes.len().to_string(),
        ),
        metadata("tritium.external_data.blake3", &digest),
    ]);
    let model_bytes = protobuf.encode_to_vec();
    if model_bytes.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    Ok(ExternalOnnxModel {
        model_bytes,
        weights_bytes,
        weights_blake3,
    })
}

/// Verify an external causal-LM model before ONNX Runtime session creation.
///
/// # Security
/// `admitted` must come from an independently authenticated package manifest.
/// Never compute it from `model_bytes` or `weights_bytes`: doing so removes the
/// trust root and turns integrity verification into a self-consistency check.
///
/// # Errors
/// [`OnnxModelError`] for malformed metadata, unsupported graph semantics,
/// noncanonical/overlapping ranges, nonzero padding, or manifest digest/length
/// mismatch.
pub fn verify_external_causal_lm(
    model_bytes: &[u8],
    weights_bytes: &[u8],
    admitted: AdmittedExternalCausalLmDigests,
) -> Result<VerifiedExternalCausalLmModel, OnnxModelError> {
    if model_bytes.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    let actual_model_hash = blake3::hash(model_bytes);
    if actual_model_hash.as_bytes() != &admitted.model_blake3 {
        return Err(OnnxModelError::ExternalDataMismatch(
            "model BLAKE3 differs from admitted package manifest".to_owned(),
        ));
    }
    let actual_hash = blake3::hash(weights_bytes);
    if actual_hash.as_bytes() != &admitted.weights_blake3 {
        return Err(OnnxModelError::ExternalDataMismatch(
            "weights BLAKE3 differs from admitted package manifest".to_owned(),
        ));
    }
    let protobuf = ModelProto::decode(model_bytes)
        .map_err(|error| OnnxModelError::InvalidModel(error.to_string()))?;
    let mut metadata = BTreeMap::new();
    for entry in &protobuf.metadata_props {
        if metadata
            .insert(entry.key.clone(), entry.value.clone())
            .is_some()
        {
            return Err(OnnxModelError::InvalidModel(format!(
                "duplicate metadata key {}",
                entry.key
            )));
        }
    }
    require_metadata(&metadata, "tritium.schema_version", "2")?;
    require_metadata(&metadata, "tritium.graph_kind", "causal-lm")?;
    require_metadata(
        &metadata,
        "tritium.external_data.file",
        EXTERNAL_WEIGHTS_FILE,
    )?;
    let declared_weights = parse_usize(&metadata, "tritium.external_data.bytes")?;
    if declared_weights != weights_bytes.len() {
        return Err(OnnxModelError::ExternalDataMismatch(format!(
            "byte length {} differs from declared {declared_weights}",
            weights_bytes.len()
        )));
    }
    if metadata_value(&metadata, "tritium.external_data.blake3")?
        != actual_hash.to_hex().to_string()
    {
        return Err(OnnxModelError::ExternalDataMismatch(
            "BLAKE3 digest differs from model metadata".to_owned(),
        ));
    }
    let diagnostics = diagnose_unsupported_graph(model_bytes)?;
    if !diagnostics.is_empty() {
        return Err(OnnxModelError::InvalidModel(format!(
            "graph has {} unsupported items",
            diagnostics.len()
        )));
    }
    let graph = protobuf
        .graph
        .as_ref()
        .ok_or_else(|| OnnxModelError::InvalidModel("model has no graph".to_owned()))?;
    let mut cursor = 0_usize;
    let mut names = BTreeMap::new();
    for initializer in &graph.initializer {
        if names.insert(initializer.name.as_str(), ()).is_some() {
            return Err(OnnxModelError::InvalidModel(format!(
                "duplicate initializer {}",
                initializer.name
            )));
        }
        let element_bytes = match initializer.data_type {
            TENSOR_UINT8 => 1,
            TENSOR_FLOAT => core::mem::size_of::<f32>(),
            TENSOR_INT64 => core::mem::size_of::<i64>(),
            other => {
                return Err(OnnxModelError::ExternalDataMismatch(format!(
                    "initializer {} has unsupported dtype {other}",
                    initializer.name
                )));
            }
        };
        let expected_length =
            initializer
                .dims
                .iter()
                .try_fold(element_bytes, |bytes, &dimension| {
                    let dimension = usize::try_from(dimension).map_err(|_| {
                        OnnxModelError::ExternalDataMismatch(format!(
                            "initializer {} has negative dimension",
                            initializer.name
                        ))
                    })?;
                    bytes
                        .checked_mul(dimension)
                        .ok_or(OnnxModelError::ShapeOverflow(
                            "external initializer byte count",
                        ))
                })?;
        if initializer.data_type == TENSOR_INT64 {
            if initializer.data_location != 0
                || !initializer.external_data.is_empty()
                || initializer.raw_data.len() != expected_length
            {
                return Err(OnnxModelError::ExternalDataMismatch(format!(
                    "shape initializer {} is not canonical inline data",
                    initializer.name
                )));
            }
            continue;
        }
        if initializer.data_location != EXTERNAL_DATA || !initializer.raw_data.is_empty() {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} is not exclusively external",
                initializer.name
            )));
        }
        let mut entries = BTreeMap::new();
        for entry in &initializer.external_data {
            if entries
                .insert(entry.key.as_str(), entry.value.as_str())
                .is_some()
            {
                return Err(OnnxModelError::ExternalDataMismatch(format!(
                    "initializer {} has duplicate external-data key {}",
                    initializer.name, entry.key
                )));
            }
        }
        if entries.len() != 3 || entries.get("location") != Some(&EXTERNAL_WEIGHTS_FILE) {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} external-data descriptor is not canonical",
                initializer.name
            )));
        }
        let parse_range = |key: &str| {
            let value = entries.get(key).ok_or_else(|| {
                OnnxModelError::ExternalDataMismatch(format!(
                    "initializer {} missing {key}",
                    initializer.name
                ))
            })?;
            let parsed = value.parse::<usize>().map_err(|_| {
                OnnxModelError::ExternalDataMismatch(format!(
                    "initializer {} {key} is not usize",
                    initializer.name
                ))
            })?;
            if parsed.to_string() != *value {
                return Err(OnnxModelError::ExternalDataMismatch(format!(
                    "initializer {} {key} is not canonical decimal",
                    initializer.name
                )));
            }
            Ok(parsed)
        };
        let offset = parse_range("offset")?;
        let length = parse_range("length")?;
        if length != expected_length {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} length {length} differs from shape-derived {expected_length}",
                initializer.name
            )));
        }
        let expected_offset = align_up(cursor, EXTERNAL_ALIGNMENT)?;
        if offset != expected_offset {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} offset {offset} is not canonical {expected_offset}",
                initializer.name
            )));
        }
        if offset > weights_bytes.len() {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} offset exceeds weights.bin",
                initializer.name
            )));
        }
        if weights_bytes[cursor..offset].iter().any(|&byte| byte != 0) {
            return Err(OnnxModelError::ExternalDataMismatch(
                "alignment padding is not zero".to_owned(),
            ));
        }
        let end = offset
            .checked_add(length)
            .ok_or(OnnxModelError::ShapeOverflow("external initializer range"))?;
        if end > weights_bytes.len() {
            return Err(OnnxModelError::ExternalDataMismatch(format!(
                "initializer {} range exceeds weights.bin",
                initializer.name
            )));
        }
        cursor = end;
    }
    if cursor != weights_bytes.len() {
        return Err(OnnxModelError::ExternalDataMismatch(
            "weights.bin has unreferenced trailing bytes".to_owned(),
        ));
    }
    let identity = VerifiedOnnxArtifactIdentityV2 {
        source_model_id: metadata_value(&metadata, "tritium.source_model_id")?.to_owned(),
        tokenizer_id: metadata_value(&metadata, "tritium.tokenizer_id")?.to_owned(),
        recipe_id: metadata_value(&metadata, "tritium.recipe_id")?.to_owned(),
        tritium_build_id: metadata_value(&metadata, "tritium.build_id")?.to_owned(),
        package_id: metadata_value(&metadata, "tritium.package_id")?.to_owned(),
        converted_coverage_id: metadata_value(&metadata, "tritium.coverage.converted_id")?
            .to_owned(),
        deferred_coverage_id: metadata_value(&metadata, "tritium.coverage.deferred_id")?.to_owned(),
    };
    Ok(VerifiedExternalCausalLmModel {
        model_blake3: *actual_model_hash.as_bytes(),
        weights_blake3: *actual_hash.as_bytes(),
        weights_bytes: weights_bytes.len(),
        tokens: parse_usize(&metadata, "tritium.tokens")?,
        past_tokens: parse_usize(&metadata, "tritium.past_tokens")?,
        layers: parse_usize(&metadata, "tritium.layers")?,
        identity,
    })
}

/// Encode a tied packed embedding/head graph with 64-byte-aligned initializers
/// in the fixed relative file `weights.bin`.
///
/// The model metadata binds the external filename, exact byte length and BLAKE3
/// digest. A loader must verify those fields before creating an ORT session.
///
/// # Errors
/// [`OnnxModelError`] if the graph contract is invalid or external offsets and
/// lengths overflow ONNX `int64`/host `usize`.
pub fn encode_external_tied_embedding_head(
    model: TiedEmbeddingHeadModel<'_>,
) -> Result<ExternalOnnxModel, OnnxModelError> {
    validate(&model)?;
    encode_external_model(model, Vec::new(), "1")
}

/// Encode a schema-v2 tied embedding/head graph with content-bound external data.
///
/// # Errors
/// [`OnnxModelError`] if graph geometry, payload, external layout, or any
/// identity is invalid.
pub fn encode_external_tied_embedding_head_v2(
    model: TiedEmbeddingHeadModelV2<'_>,
) -> Result<ExternalOnnxModel, OnnxModelError> {
    validate_v2(&model)?;
    encode_external_model(model.legacy(), identity_metadata_v2(model.identity), "2")
}

fn encode_external_model(
    model: TiedEmbeddingHeadModel<'_>,
    mut identity_metadata: Vec<StringStringEntryProto>,
    schema_version: &'static str,
) -> Result<ExternalOnnxModel, OnnxModelError> {
    let scale_bytes = scale_bytes(model.scales);
    let scale_offset = align_up(model.packed.len(), EXTERNAL_ALIGNMENT)?;
    let weights_len = scale_offset
        .checked_add(scale_bytes.len())
        .ok_or(OnnxModelError::ShapeOverflow("external weight byte count"))?;
    let mut weights_bytes = Vec::new();
    weights_bytes
        .try_reserve_exact(weights_len)
        .map_err(|_| OnnxModelError::ShapeOverflow("external weight allocation"))?;
    weights_bytes.extend_from_slice(model.packed);
    weights_bytes.resize(scale_offset, 0);
    weights_bytes.extend_from_slice(&scale_bytes);
    let weights_blake3 = *blake3::hash(&weights_bytes).as_bytes();
    let digest_hex = blake3::Hash::from_bytes(weights_blake3)
        .to_hex()
        .to_string();
    let packed_len = as_i64(model.packed.len(), "packed byte count")?;
    let scale_offset_i64 = as_i64(scale_offset, "external scale offset")?;
    let scale_len = as_i64(scale_bytes.len(), "external scale byte count")?;
    let vocab = as_i64(model.vocab, "vocabulary")?;
    let model_bytes = encode_model(
        model,
        vec![
            external_tensor(
                "tritium.packed",
                TENSOR_UINT8,
                vec![packed_len],
                0,
                packed_len,
            ),
            external_tensor(
                "tritium.scales",
                TENSOR_FLOAT,
                vec![vocab],
                scale_offset_i64,
                scale_len,
            ),
        ],
        {
            identity_metadata.extend([
                metadata("tritium.external_data.file", EXTERNAL_WEIGHTS_FILE),
                metadata("tritium.external_data.bytes", &weights_len.to_string()),
                metadata("tritium.external_data.blake3", &digest_hex),
            ]);
            identity_metadata
        },
        schema_version,
    )?;
    Ok(ExternalOnnxModel {
        model_bytes,
        weights_bytes,
        weights_blake3,
    })
}

/// Verify a serialized external-data graph before it is handed to ONNX Runtime.
///
/// Verification is fail-closed: metadata keys must be unique, the fixed
/// filename and geometry bindings must be canonical, length and BLAKE3 must
/// match, padding must be zero, scales and packed rows must validate, and a
/// deterministic re-encode must reproduce both files byte-for-byte.
///
/// # Errors
/// [`OnnxModelError`] for malformed protobuf/metadata, a digest or length
/// mismatch, invalid packed/scales data, or any non-canonical graph mutation.
pub fn verify_external_tied_embedding_head(
    model_bytes: &[u8],
    weights_bytes: &[u8],
) -> Result<VerifiedExternalOnnxModel, OnnxModelError> {
    let verified = verify_external_parts(model_bytes, weights_bytes, "1")?;
    let source_model_id = metadata_value(&verified.metadata, "tritium.source_model_id")?.to_owned();
    let recipe_id = metadata_value(&verified.metadata, "tritium.recipe_id")?.to_owned();
    let package_id = metadata_value(&verified.metadata, "tritium.package_id")?.to_owned();
    let specification = TiedEmbeddingHeadModel {
        tokens: verified.tokens,
        vocab: verified.vocab,
        hidden: verified.hidden,
        packed: &weights_bytes[..verified.packed_len],
        scales: &verified.scales,
        format: verified.format,
        source_model_id: &source_model_id,
        recipe_id: &recipe_id,
        package_id: &package_id,
    };
    let expected = encode_external_tied_embedding_head(specification)?;
    verify_canonical_external(&expected, model_bytes, weights_bytes)?;
    Ok(VerifiedExternalOnnxModel {
        model_blake3: *blake3::hash(model_bytes).as_bytes(),
        weights_blake3: verified.weights_blake3,
        weights_bytes: weights_bytes.len(),
        tokens: verified.tokens,
        vocab: verified.vocab,
        hidden: verified.hidden,
        source_model_id,
        recipe_id,
        package_id,
    })
}

/// Verify a schema-v2 external-data graph and its complete artifact identity.
///
/// Schema-v1 remains available through [`verify_external_tied_embedding_head`];
/// callers choose the expected contract explicitly, so neither version can be
/// silently interpreted as the other.
///
/// # Errors
/// [`OnnxModelError`] for malformed protobuf/metadata, an identity, digest or
/// length mismatch, invalid packed/scales data, or non-canonical mutation.
pub fn verify_external_tied_embedding_head_v2(
    model_bytes: &[u8],
    weights_bytes: &[u8],
    expected_identity: OnnxArtifactIdentityV2<'_>,
) -> Result<VerifiedExternalOnnxModelV2, OnnxModelError> {
    validate_identity_v2(expected_identity)?;
    let verified = verify_external_parts(model_bytes, weights_bytes, "2")?;
    let identity = VerifiedOnnxArtifactIdentityV2 {
        source_model_id: metadata_value(&verified.metadata, "tritium.source_model_id")?.to_owned(),
        tokenizer_id: metadata_value(&verified.metadata, "tritium.tokenizer_id")?.to_owned(),
        recipe_id: metadata_value(&verified.metadata, "tritium.recipe_id")?.to_owned(),
        tritium_build_id: metadata_value(&verified.metadata, "tritium.build_id")?.to_owned(),
        package_id: metadata_value(&verified.metadata, "tritium.package_id")?.to_owned(),
        converted_coverage_id: metadata_value(&verified.metadata, "tritium.coverage.converted_id")?
            .to_owned(),
        deferred_coverage_id: metadata_value(&verified.metadata, "tritium.coverage.deferred_id")?
            .to_owned(),
    };
    for (name, actual, expected) in [
        (
            "source_model_id",
            identity.source_model_id.as_str(),
            expected_identity.source_model_id,
        ),
        (
            "tokenizer_id",
            identity.tokenizer_id.as_str(),
            expected_identity.tokenizer_id,
        ),
        (
            "recipe_id",
            identity.recipe_id.as_str(),
            expected_identity.recipe_id,
        ),
        (
            "tritium_build_id",
            identity.tritium_build_id.as_str(),
            expected_identity.tritium_build_id,
        ),
        (
            "package_id",
            identity.package_id.as_str(),
            expected_identity.package_id,
        ),
        (
            "converted_coverage_id",
            identity.converted_coverage_id.as_str(),
            expected_identity.converted_coverage_id,
        ),
        (
            "deferred_coverage_id",
            identity.deferred_coverage_id.as_str(),
            expected_identity.deferred_coverage_id,
        ),
    ] {
        if actual != expected {
            return Err(OnnxModelError::InvalidModel(format!(
                "artifact identity {name} does not match admission contract"
            )));
        }
    }
    let specification = TiedEmbeddingHeadModelV2 {
        tokens: verified.tokens,
        vocab: verified.vocab,
        hidden: verified.hidden,
        packed: &weights_bytes[..verified.packed_len],
        scales: &verified.scales,
        format: verified.format,
        identity: OnnxArtifactIdentityV2 {
            source_model_id: &identity.source_model_id,
            tokenizer_id: &identity.tokenizer_id,
            recipe_id: &identity.recipe_id,
            tritium_build_id: &identity.tritium_build_id,
            package_id: &identity.package_id,
            converted_coverage_id: &identity.converted_coverage_id,
            deferred_coverage_id: &identity.deferred_coverage_id,
        },
    };
    let expected = encode_external_tied_embedding_head_v2(specification)?;
    verify_canonical_external(&expected, model_bytes, weights_bytes)?;
    Ok(VerifiedExternalOnnxModelV2 {
        model: VerifiedExternalOnnxModel {
            model_blake3: *blake3::hash(model_bytes).as_bytes(),
            weights_blake3: verified.weights_blake3,
            weights_bytes: weights_bytes.len(),
            tokens: verified.tokens,
            vocab: verified.vocab,
            hidden: verified.hidden,
            source_model_id: identity.source_model_id.clone(),
            recipe_id: identity.recipe_id.clone(),
            package_id: identity.package_id.clone(),
        },
        identity,
    })
}

struct VerifiedExternalParts {
    metadata: BTreeMap<String, String>,
    weights_blake3: [u8; 32],
    tokens: usize,
    vocab: usize,
    hidden: usize,
    format: TernaryFormat,
    packed_len: usize,
    scales: Vec<f32>,
}

fn verify_external_parts(
    model_bytes: &[u8],
    weights_bytes: &[u8],
    schema_version: &'static str,
) -> Result<VerifiedExternalParts, OnnxModelError> {
    if model_bytes.len() > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "protobuf exceeds {MAX_MODEL_BYTES} bytes"
        )));
    }
    let protobuf = ModelProto::decode(model_bytes)
        .map_err(|error| OnnxModelError::InvalidModel(error.to_string()))?;
    let mut metadata = BTreeMap::new();
    for entry in protobuf.metadata_props {
        if metadata.insert(entry.key.clone(), entry.value).is_some() {
            return Err(OnnxModelError::InvalidModel(format!(
                "duplicate metadata key {}",
                entry.key
            )));
        }
    }
    require_metadata(&metadata, "tritium.schema_version", schema_version)?;
    require_metadata(&metadata, "tritium.tied_embedding_head", "true")?;
    require_metadata(
        &metadata,
        "tritium.external_data.file",
        EXTERNAL_WEIGHTS_FILE,
    )?;
    let tokens = parse_usize(&metadata, "tritium.tokens")?;
    let vocab = parse_usize(&metadata, "tritium.vocab")?;
    let hidden = parse_usize(&metadata, "tritium.hidden")?;
    let declared_weights = parse_usize(&metadata, "tritium.external_data.bytes")?;
    let format = match metadata_value(&metadata, "tritium.weight_format")? {
        "tq2_0" => TernaryFormat::Tq2_0,
        "tq1_0" => TernaryFormat::Tq1_0,
        other => {
            return Err(OnnxModelError::InvalidModel(format!(
                "unsupported weight format {other}"
            )));
        }
    };
    let actual_weights_hash = blake3::hash(weights_bytes);
    let actual_weights_hex = actual_weights_hash.to_hex().to_string();
    if metadata_value(&metadata, "tritium.external_data.blake3")? != actual_weights_hex {
        return Err(OnnxModelError::ExternalDataMismatch(
            "BLAKE3 digest differs from model metadata".to_owned(),
        ));
    }
    if weights_bytes.len() != declared_weights {
        return Err(OnnxModelError::ExternalDataMismatch(format!(
            "byte length {} differs from declared {declared_weights}",
            weights_bytes.len()
        )));
    }
    let block_bytes = match format {
        TernaryFormat::Tq2_0 => TQ2_0_BLOCK_BYTES,
        TernaryFormat::Tq1_0 => TQ1_0_BLOCK_BYTES,
        other => return Err(OnnxModelError::UnsupportedFormat(other)),
    };
    let packed_len = num_blocks(hidden)
        .checked_mul(block_bytes)
        .and_then(|row| row.checked_mul(vocab))
        .ok_or(OnnxModelError::ShapeOverflow("packed byte count"))?;
    let scale_offset = align_up(packed_len, EXTERNAL_ALIGNMENT)?;
    let scale_len = vocab
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(OnnxModelError::ShapeOverflow("scale byte count"))?;
    let expected_weights = scale_offset
        .checked_add(scale_len)
        .ok_or(OnnxModelError::ShapeOverflow("external weight byte count"))?;
    if weights_bytes.len() != expected_weights {
        return Err(OnnxModelError::ExternalDataMismatch(format!(
            "canonical geometry requires {expected_weights} bytes, got {}",
            weights_bytes.len()
        )));
    }
    if weights_bytes[packed_len..scale_offset]
        .iter()
        .any(|&byte| byte != 0)
    {
        return Err(OnnxModelError::ExternalDataMismatch(
            "alignment padding is not zero".to_owned(),
        ));
    }
    let scales: Vec<f32> = weights_bytes[scale_offset..]
        .chunks_exact(core::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect();
    Ok(VerifiedExternalParts {
        metadata,
        weights_blake3: *actual_weights_hash.as_bytes(),
        tokens,
        vocab,
        hidden,
        format,
        packed_len,
        scales,
    })
}

fn verify_canonical_external(
    expected: &ExternalOnnxModel,
    model_bytes: &[u8],
    weights_bytes: &[u8],
) -> Result<(), OnnxModelError> {
    if expected.model_bytes != model_bytes {
        return Err(OnnxModelError::InvalidModel(
            "graph does not match the canonical bound Tritium graph".to_owned(),
        ));
    }
    if expected.weights_bytes != weights_bytes {
        return Err(OnnxModelError::ExternalDataMismatch(
            "initializer layout is not canonical".to_owned(),
        ));
    }
    Ok(())
}

fn metadata_value<'a>(
    metadata: &'a BTreeMap<String, String>,
    key: &str,
) -> Result<&'a str, OnnxModelError> {
    metadata
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| OnnxModelError::InvalidModel(format!("missing metadata key {key}")))
}

fn require_metadata(
    metadata: &BTreeMap<String, String>,
    key: &str,
    expected: &str,
) -> Result<(), OnnxModelError> {
    let value = metadata_value(metadata, key)?;
    if value != expected {
        return Err(OnnxModelError::InvalidModel(format!(
            "metadata {key} must be {expected:?}, got {value:?}"
        )));
    }
    Ok(())
}

fn parse_usize(metadata: &BTreeMap<String, String>, key: &str) -> Result<usize, OnnxModelError> {
    let value = metadata_value(metadata, key)?;
    let parsed = value
        .parse::<usize>()
        .map_err(|_| OnnxModelError::InvalidModel(format!("metadata {key} is not usize")))?;
    if parsed.to_string() != value {
        return Err(OnnxModelError::InvalidModel(format!(
            "metadata {key} is not canonical decimal"
        )));
    }
    Ok(parsed)
}

fn encode_model(
    model: TiedEmbeddingHeadModel<'_>,
    initializer: Vec<TensorProto>,
    mut extra_metadata: Vec<StringStringEntryProto>,
    schema_version: &'static str,
) -> Result<Vec<u8>, OnnxModelError> {
    let tokens = as_i64(model.tokens, "token count")?;
    let vocab = as_i64(model.vocab, "vocabulary")?;
    let hidden = as_i64(model.hidden, "hidden size")?;
    let format = format_code(model.format)?;

    let graph = GraphProto {
        node: vec![
            NodeProto {
                input: strings(["tokens", "tritium.packed", "tritium.scales"]),
                output: strings(["hidden"]),
                name: "tritium.embedding".to_owned(),
                op_type: ONNX_EMBEDDING_OP_NAME.to_owned(),
                attribute: attributes(hidden, format),
                domain: ONNX_DOMAIN.to_owned(),
            },
            NodeProto {
                input: strings(["hidden", "tritium.packed", "tritium.scales"]),
                output: strings(["logits"]),
                name: "tritium.lm_head".to_owned(),
                op_type: ONNX_OP_NAME.to_owned(),
                attribute: attributes(hidden, format),
                domain: ONNX_DOMAIN.to_owned(),
            },
        ],
        name: "tritium.tied_embedding_head".to_owned(),
        initializer,
        input: vec![tensor_value("tokens", TENSOR_INT64, &[tokens])],
        output: vec![tensor_value("logits", TENSOR_FLOAT, &[tokens, vocab])],
        value_info: Vec::new(),
    };
    let mut metadata_props = vec![
        metadata("tritium.schema_version", schema_version),
        metadata("tritium.source_model_id", model.source_model_id),
        metadata("tritium.recipe_id", model.recipe_id),
        metadata("tritium.package_id", model.package_id),
        metadata("tritium.weight_format", &model.format.to_string()),
        metadata("tritium.tied_embedding_head", "true"),
        metadata("tritium.tokens", &model.tokens.to_string()),
        metadata("tritium.vocab", &model.vocab.to_string()),
        metadata("tritium.hidden", &model.hidden.to_string()),
    ];
    metadata_props.append(&mut extra_metadata);
    Ok(ModelProto {
        ir_version: ONNX_IR_VERSION,
        producer_name: "tritium-onnx".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        domain: ONNX_DOMAIN.to_owned(),
        model_version: schema_version
            .parse()
            .expect("static schema version is numeric"),
        graph: Some(graph),
        opset_import: vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: ONNX_OPSET,
            },
            OperatorSetIdProto {
                domain: ONNX_DOMAIN.to_owned(),
                version: TRITIUM_OPSET,
            },
        ],
        metadata_props,
    }
    .encode_to_vec())
}

struct CausalGraphBuilder {
    nodes: Vec<NodeProto>,
    initializers: Vec<TensorProto>,
    storage: CausalInitializerStorage,
    failure: Option<OnnxModelError>,
}

enum CausalInitializerStorage {
    Inline,
    External(Vec<u8>),
}

fn reserve_external_range(weights: &mut Vec<u8>, length: usize) -> Result<usize, OnnxModelError> {
    let offset = align_up(weights.len(), EXTERNAL_ALIGNMENT)?;
    let required = offset
        .checked_sub(weights.len())
        .and_then(|padding| padding.checked_add(length))
        .ok_or(OnnxModelError::ShapeOverflow(
            "external initializer allocation",
        ))?;
    weights
        .try_reserve_exact(required)
        .map_err(|_| OnnxModelError::ShapeOverflow("external initializer allocation"))?;
    weights.resize(offset, 0);
    Ok(offset)
}

impl Default for CausalGraphBuilder {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            initializers: Vec::new(),
            storage: CausalInitializerStorage::Inline,
            failure: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CausalGraphGeometry {
    past_tokens: i64,
    total_tokens: i64,
    n_head: i64,
    n_kv_head: i64,
    head_dim: i64,
    gqa_repeat: i64,
}

#[derive(Debug, Clone, Copy)]
struct RotaryGraphConfig {
    tokens: usize,
    head_dim: usize,
    past_tokens: usize,
    theta: f32,
}

struct RotaryGraphState {
    first_start: String,
    first_end: String,
    second_start: String,
    second_end: String,
    axes: String,
    steps: String,
    cos: String,
    sin: String,
}

fn rotary_tables(config: RotaryGraphConfig) -> Result<(Vec<f32>, Vec<f32>), OnnxModelError> {
    let RotaryGraphConfig {
        tokens, head_dim, ..
    } = config;
    let elements = tokens
        .checked_mul(head_dim)
        .ok_or(OnnxModelError::ShapeOverflow("RoPE table element count"))?;
    let table_bytes = elements
        .checked_mul(2)
        .and_then(|values| values.checked_mul(core::mem::size_of::<f32>()))
        .ok_or(OnnxModelError::ShapeOverflow("RoPE table byte count"))?;
    if table_bytes > MAX_MODEL_BYTES {
        return Err(OnnxModelError::InvalidModel(format!(
            "RoPE tables require {table_bytes} bytes, exceed bounded inline graph"
        )));
    }
    let mut cos = Vec::with_capacity(elements);
    let mut sin = Vec::with_capacity(elements);
    for token in 0..tokens {
        for lane in 0..head_dim {
            let (lane_cos, lane_sin) = rotary_pair(config, token, lane)?;
            cos.push(lane_cos as f32);
            sin.push(lane_sin as f32);
        }
    }
    Ok((cos, sin))
}

fn rotary_pair(
    config: RotaryGraphConfig,
    token: usize,
    lane: usize,
) -> Result<(f64, f64), OnnxModelError> {
    let half = config.head_dim / 2;
    let frequency_lane = lane % half;
    let position = (config.past_tokens + token) as f64;
    let angle = position
        * f64::from(config.theta).powf(-2.0 * frequency_lane as f64 / config.head_dim as f64);
    if !angle.is_finite() {
        return Err(OnnxModelError::InvalidModel(format!(
            "RoPE angle is non-finite at position {} frequency lane {frequency_lane}",
            config.past_tokens + token
        )));
    }
    let (sin, cos) = angle.sin_cos();
    Ok((cos, sin))
}

struct CacheGraphOutputs {
    expanded_k: String,
    expanded_v: String,
    declarations: [ValueInfoProto; 2],
}

impl CausalGraphBuilder {
    fn external() -> Self {
        Self {
            storage: CausalInitializerStorage::External(Vec::new()),
            ..Self::default()
        }
    }

    fn is_external(&self) -> bool {
        matches!(self.storage, CausalInitializerStorage::External(_))
    }

    fn storage_result(&self) -> Result<(), OnnxModelError> {
        self.failure.clone().map_or(Ok(()), Err)
    }

    fn store_bytes(&mut self, name: &str, data_type: i32, dimensions: Vec<i64>, bytes: &[u8]) {
        if self.failure.is_some() {
            return;
        }
        let tensor = match &mut self.storage {
            CausalInitializerStorage::Inline => {
                inline_tensor(name, data_type, dimensions, bytes.to_vec())
            }
            CausalInitializerStorage::External(weights) => {
                let offset = match reserve_external_range(weights, bytes.len()) {
                    Ok(offset) => offset,
                    Err(error) => {
                        self.failure.get_or_insert(error);
                        return;
                    }
                };
                weights.extend_from_slice(bytes);
                let offset = match i64::try_from(offset) {
                    Ok(offset) => offset,
                    Err(_) => {
                        self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                            "external initializer offset",
                        ));
                        return;
                    }
                };
                let length = match i64::try_from(bytes.len()) {
                    Ok(length) => length,
                    Err(_) => {
                        self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                            "external initializer length",
                        ));
                        return;
                    }
                };
                external_tensor(name, data_type, dimensions, offset, length)
            }
        };
        self.initializers.push(tensor);
    }

    fn standard(
        &mut self,
        name: impl Into<String>,
        op_type: &str,
        inputs: &[&str],
        outputs: &[&str],
        attribute: Vec<AttributeProto>,
    ) {
        if self.failure.is_some() {
            return;
        }
        self.nodes.push(NodeProto {
            input: inputs.iter().map(|value| (*value).to_owned()).collect(),
            output: outputs.iter().map(|value| (*value).to_owned()).collect(),
            name: name.into(),
            op_type: op_type.to_owned(),
            attribute,
            domain: String::new(),
        });
    }

    fn add_f32(&mut self, name: &str, dimensions: Vec<i64>, values: &[f32]) {
        if self.failure.is_some() {
            return;
        }
        if matches!(self.storage, CausalInitializerStorage::Inline) {
            self.initializers.push(inline_tensor(
                name,
                TENSOR_FLOAT,
                dimensions,
                scale_bytes(values),
            ));
            return;
        }
        let length = match values.len().checked_mul(core::mem::size_of::<f32>()) {
            Some(length) => length,
            None => {
                self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                    "external f32 initializer byte count",
                ));
                return;
            }
        };
        let CausalInitializerStorage::External(weights) = &mut self.storage else {
            unreachable!();
        };
        let offset = match reserve_external_range(weights, length) {
            Ok(offset) => offset,
            Err(error) => {
                self.failure.get_or_insert(error);
                return;
            }
        };
        for value in values {
            weights.extend_from_slice(&value.to_le_bytes());
        }
        let Ok(offset) = i64::try_from(offset) else {
            self.failure
                .get_or_insert(OnnxModelError::ShapeOverflow("external initializer offset"));
            return;
        };
        let Ok(length) = i64::try_from(length) else {
            self.failure
                .get_or_insert(OnnxModelError::ShapeOverflow("external initializer length"));
            return;
        };
        self.initializers.push(external_tensor(
            name,
            TENSOR_FLOAT,
            dimensions,
            offset,
            length,
        ));
    }

    fn add_causal_mask(
        &mut self,
        name: &str,
        tokens: usize,
        total_tokens: usize,
        past_tokens: usize,
        dimensions: Vec<i64>,
    ) {
        if self.failure.is_some() {
            return;
        }
        let elements = match tokens.checked_mul(total_tokens) {
            Some(elements) => elements,
            None => {
                self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                    "attention mask element count",
                ));
                return;
            }
        };
        if matches!(self.storage, CausalInitializerStorage::Inline) {
            let mut values = Vec::new();
            if values.try_reserve_exact(elements).is_err() {
                self.failure
                    .get_or_insert(OnnxModelError::ShapeOverflow("attention mask allocation"));
                return;
            }
            for query_index in 0..tokens {
                let last_visible = past_tokens + query_index;
                for key_index in 0..total_tokens {
                    values.push(if key_index <= last_visible {
                        0.0
                    } else {
                        -1.0e9
                    });
                }
            }
            self.add_f32(name, dimensions, &values);
            return;
        }
        let length = match elements.checked_mul(core::mem::size_of::<f32>()) {
            Some(length) => length,
            None => {
                self.failure
                    .get_or_insert(OnnxModelError::ShapeOverflow("attention mask byte count"));
                return;
            }
        };
        let CausalInitializerStorage::External(weights) = &mut self.storage else {
            unreachable!();
        };
        let offset = match reserve_external_range(weights, length) {
            Ok(offset) => offset,
            Err(error) => {
                self.failure.get_or_insert(error);
                return;
            }
        };
        let (Ok(offset_i64), Ok(length_i64)) = (i64::try_from(offset), i64::try_from(length))
        else {
            self.failure.get_or_insert(OnnxModelError::ShapeOverflow(
                "external attention mask range",
            ));
            return;
        };
        for query_index in 0..tokens {
            let last_visible = past_tokens + query_index;
            for key_index in 0..total_tokens {
                let value = if key_index <= last_visible {
                    0.0_f32
                } else {
                    -1.0e9
                };
                weights.extend_from_slice(&value.to_le_bytes());
            }
        }
        self.initializers.push(external_tensor(
            name,
            TENSOR_FLOAT,
            dimensions,
            offset_i64,
            length_i64,
        ));
    }

    fn add_external_rotary_table(
        &mut self,
        name: &str,
        config: RotaryGraphConfig,
        cosine: bool,
        dimensions: Vec<i64>,
    ) {
        if self.failure.is_some() {
            return;
        }
        let elements = match config.tokens.checked_mul(config.head_dim) {
            Some(elements) => elements,
            None => {
                self.failure
                    .get_or_insert(OnnxModelError::ShapeOverflow("RoPE table element count"));
                return;
            }
        };
        let length = match elements.checked_mul(core::mem::size_of::<f32>()) {
            Some(length) => length,
            None => {
                self.failure
                    .get_or_insert(OnnxModelError::ShapeOverflow("RoPE table byte count"));
                return;
            }
        };
        let CausalInitializerStorage::External(weights) = &mut self.storage else {
            self.failure.get_or_insert(OnnxModelError::InvalidModel(
                "direct RoPE table emission requires external storage".to_owned(),
            ));
            return;
        };
        let offset = match reserve_external_range(weights, length) {
            Ok(offset) => offset,
            Err(error) => {
                self.failure.get_or_insert(error);
                return;
            }
        };
        let (Ok(offset_i64), Ok(length_i64)) = (i64::try_from(offset), i64::try_from(length))
        else {
            self.failure
                .get_or_insert(OnnxModelError::ShapeOverflow("external RoPE table range"));
            return;
        };
        for token in 0..config.tokens {
            for lane in 0..config.head_dim {
                let pair = match rotary_pair(config, token, lane) {
                    Ok(pair) => pair,
                    Err(error) => {
                        self.failure.get_or_insert(error);
                        return;
                    }
                };
                let value = if cosine { pair.0 } else { pair.1 } as f32;
                weights.extend_from_slice(&value.to_le_bytes());
            }
        }
        self.initializers.push(external_tensor(
            name,
            TENSOR_FLOAT,
            dimensions,
            offset_i64,
            length_i64,
        ));
    }

    fn add_i64(&mut self, name: &str, dimensions: Vec<i64>, values: &[i64]) {
        if self.failure.is_some() {
            return;
        }
        self.initializers.push(inline_tensor(
            name,
            TENSOR_INT64,
            dimensions,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        ));
    }

    fn add_matrix(&mut self, prefix: &str, matrix: PackedTernaryMatrix<'_>) {
        self.store_bytes(
            &format!("{prefix}.packed"),
            TENSOR_UINT8,
            vec![i64::try_from(matrix.packed.len()).expect("validated packed byte count")],
            matrix.packed,
        );
        self.add_f32(
            &format!("{prefix}.scales"),
            vec![i64::try_from(matrix.rows).expect("validated matrix rows")],
            matrix.scales,
        );
    }

    fn projection(
        &mut self,
        name: &str,
        input: &str,
        output: &str,
        matrix_prefix: &str,
        matrix: PackedTernaryMatrix<'_>,
    ) -> Result<(), OnnxModelError> {
        self.add_matrix(matrix_prefix, matrix);
        self.storage_result()?;
        self.nodes.push(NodeProto {
            input: vec![
                input.to_owned(),
                format!("{matrix_prefix}.packed"),
                format!("{matrix_prefix}.scales"),
            ],
            output: vec![output.to_owned()],
            name: name.to_owned(),
            op_type: ONNX_OP_NAME.to_owned(),
            attribute: attributes(
                as_i64(matrix.columns, "matrix columns")?,
                format_code(matrix.format)?,
            ),
            domain: ONNX_DOMAIN.to_owned(),
        });
        Ok(())
    }

    fn rms_norm(&mut self, prefix: &str, input: &str, output: &str, weight: &[f32], epsilon: f32) {
        let squared = format!("{prefix}.squared");
        let mean = format!("{prefix}.mean");
        let stabilized = format!("{prefix}.stabilized");
        let root = format!("{prefix}.root");
        let normalized = format!("{prefix}.normalized");
        let axes = format!("{prefix}.axes");
        let epsilon_name = format!("{prefix}.epsilon");
        let weight_name = format!("{prefix}.weight");
        self.add_i64(&axes, vec![1], &[-1]);
        self.add_f32(&epsilon_name, Vec::new(), &[epsilon]);
        self.add_f32(
            &weight_name,
            vec![i64::try_from(weight.len()).expect("validated norm width")],
            weight,
        );
        self.standard(
            format!("{prefix}.square"),
            "Mul",
            &[input, input],
            &[&squared],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.reduce"),
            "ReduceMean",
            &[&squared, &axes],
            &[&mean],
            vec![int_attribute("keepdims", 1)],
        );
        self.standard(
            format!("{prefix}.stabilize"),
            "Add",
            &[&mean, &epsilon_name],
            &[&stabilized],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.sqrt"),
            "Sqrt",
            &[&stabilized],
            &[&root],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.divide"),
            "Div",
            &[input, &root],
            &[&normalized],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.scale"),
            "Mul",
            &[&normalized, &weight_name],
            &[output],
            Vec::new(),
        );
    }

    fn prepare_rotary(
        &mut self,
        config: RotaryGraphConfig,
    ) -> Result<RotaryGraphState, OnnxModelError> {
        let RotaryGraphConfig {
            tokens, head_dim, ..
        } = config;
        let half = head_dim / 2;
        let first_start = "rope.first_start".to_owned();
        let first_end = "rope.first_end".to_owned();
        let second_start = "rope.second_start".to_owned();
        let second_end = "rope.second_end".to_owned();
        let axes = "rope.axes".to_owned();
        let steps = "rope.steps".to_owned();
        self.add_i64(&first_start, vec![1], &[0]);
        self.add_i64(&first_end, vec![1], &[as_i64(half, "RoPE half dimension")?]);
        self.add_i64(
            &second_start,
            vec![1],
            &[as_i64(half, "RoPE half dimension")?],
        );
        self.add_i64(
            &second_end,
            vec![1],
            &[as_i64(head_dim, "RoPE head dimension")?],
        );
        self.add_i64(&axes, vec![1], &[-1]);
        self.add_i64(&steps, vec![1], &[1]);
        let cos_name = "rope.cos".to_owned();
        let sin_name = "rope.sin".to_owned();
        let dimensions = vec![
            as_i64(tokens, "RoPE token count")?,
            1,
            as_i64(head_dim, "RoPE head dimension")?,
        ];
        if self.is_external() {
            self.add_external_rotary_table(&cos_name, config, true, dimensions.clone());
            self.add_external_rotary_table(&sin_name, config, false, dimensions);
        } else {
            let (cos, sin) = rotary_tables(config)?;
            self.add_f32(&cos_name, dimensions.clone(), &cos);
            self.add_f32(&sin_name, dimensions, &sin);
        }
        self.storage_result()?;
        Ok(RotaryGraphState {
            first_start,
            first_end,
            second_start,
            second_end,
            axes,
            steps,
            cos: cos_name,
            sin: sin_name,
        })
    }

    fn rotary(&mut self, prefix: &str, input: &str, output: &str, state: &RotaryGraphState) {
        let first = format!("{prefix}.first");
        let second = format!("{prefix}.second");
        self.standard(
            format!("{prefix}.slice_first"),
            "Slice",
            &[
                input,
                &state.first_start,
                &state.first_end,
                &state.axes,
                &state.steps,
            ],
            &[&first],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.slice_second"),
            "Slice",
            &[
                input,
                &state.second_start,
                &state.second_end,
                &state.axes,
                &state.steps,
            ],
            &[&second],
            Vec::new(),
        );
        let negated_second = format!("{prefix}.negated_second");
        self.standard(
            format!("{prefix}.negate_second"),
            "Neg",
            &[&second],
            &[&negated_second],
            Vec::new(),
        );
        let rotated = format!("{prefix}.rotated");
        self.standard(
            format!("{prefix}.rotate_half"),
            "Concat",
            &[&negated_second, &first],
            &[&rotated],
            vec![int_attribute("axis", -1)],
        );
        let direct = format!("{prefix}.direct");
        let crossed = format!("{prefix}.crossed");
        self.standard(
            format!("{prefix}.direct_mul"),
            "Mul",
            &[input, &state.cos],
            &[&direct],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.cross_mul"),
            "Mul",
            &[&rotated, &state.sin],
            &[&crossed],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.sum"),
            "Add",
            &[&direct, &crossed],
            &[output],
            Vec::new(),
        );
    }

    fn cache_and_expand_gqa(
        &mut self,
        index: usize,
        current_k: &str,
        current_v: &str,
        geometry: CausalGraphGeometry,
        inputs: &mut Vec<ValueInfoProto>,
    ) -> CacheGraphOutputs {
        let prefix = format!("layer.{index}");
        let present_k = format!("present_k.{index}");
        let present_v = format!("present_v.{index}");
        if geometry.past_tokens == 0 {
            self.standard(
                format!("{prefix}.key_identity"),
                "Identity",
                &[current_k],
                &[&present_k],
                Vec::new(),
            );
            self.standard(
                format!("{prefix}.value_identity"),
                "Identity",
                &[current_v],
                &[&present_v],
                Vec::new(),
            );
        } else {
            let past_k = format!("past_k.{index}");
            let past_v = format!("past_v.{index}");
            inputs.push(tensor_value(
                &past_k,
                TENSOR_FLOAT,
                &[geometry.past_tokens, geometry.n_kv_head, geometry.head_dim],
            ));
            inputs.push(tensor_value(
                &past_v,
                TENSOR_FLOAT,
                &[geometry.past_tokens, geometry.n_kv_head, geometry.head_dim],
            ));
            self.standard(
                format!("{prefix}.key_concat"),
                "Concat",
                &[&past_k, current_k],
                &[&present_k],
                vec![int_attribute("axis", 0)],
            );
            self.standard(
                format!("{prefix}.value_concat"),
                "Concat",
                &[&past_v, current_v],
                &[&present_v],
                vec![int_attribute("axis", 0)],
            );
        }
        let declarations = [
            tensor_value(
                &present_k,
                TENSOR_FLOAT,
                &[geometry.total_tokens, geometry.n_kv_head, geometry.head_dim],
            ),
            tensor_value(
                &present_v,
                TENSOR_FLOAT,
                &[geometry.total_tokens, geometry.n_kv_head, geometry.head_dim],
            ),
        ];
        let grouped_k_shape = format!("{prefix}.grouped_k_shape");
        let grouped_v_shape = format!("{prefix}.grouped_v_shape");
        let grouped_shape = [
            geometry.total_tokens,
            geometry.n_kv_head,
            1,
            geometry.head_dim,
        ];
        self.add_i64(&grouped_k_shape, vec![4], &grouped_shape);
        self.add_i64(&grouped_v_shape, vec![4], &grouped_shape);
        let grouped_k = format!("{prefix}.grouped_k");
        let grouped_v = format!("{prefix}.grouped_v");
        self.standard(
            format!("{prefix}.key_group_reshape"),
            "Reshape",
            &[&present_k, &grouped_k_shape],
            &[&grouped_k],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.value_group_reshape"),
            "Reshape",
            &[&present_v, &grouped_v_shape],
            &[&grouped_v],
            Vec::new(),
        );
        let repeats = format!("{prefix}.gqa_repeats");
        self.add_i64(&repeats, vec![4], &[1, 1, geometry.gqa_repeat, 1]);
        let tiled_k = format!("{prefix}.tiled_k");
        let tiled_v = format!("{prefix}.tiled_v");
        self.standard(
            format!("{prefix}.key_gqa_tile"),
            "Tile",
            &[&grouped_k, &repeats],
            &[&tiled_k],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.value_gqa_tile"),
            "Tile",
            &[&grouped_v, &repeats],
            &[&tiled_v],
            Vec::new(),
        );
        let expanded_shape = format!("{prefix}.expanded_kv_shape");
        self.add_i64(
            &expanded_shape,
            vec![3],
            &[geometry.total_tokens, geometry.n_head, geometry.head_dim],
        );
        let expanded_k = format!("{prefix}.expanded_k");
        let expanded_v = format!("{prefix}.expanded_v");
        self.standard(
            format!("{prefix}.key_gqa"),
            "Reshape",
            &[&tiled_k, &expanded_shape],
            &[&expanded_k],
            Vec::new(),
        );
        self.standard(
            format!("{prefix}.value_gqa"),
            "Reshape",
            &[&tiled_v, &expanded_shape],
            &[&expanded_v],
            Vec::new(),
        );
        CacheGraphOutputs {
            expanded_k,
            expanded_v,
            declarations,
        }
    }

    fn ffn_block(
        &mut self,
        prefix: &str,
        input: &str,
        output: &str,
        layer: CausalLmDecoderLayer<'_>,
        epsilon: f32,
    ) -> Result<(), OnnxModelError> {
        let ffn_input = format!("{prefix}.ffn_input");
        self.rms_norm(
            &format!("{prefix}.ffn_norm"),
            input,
            &ffn_input,
            layer.ffn_norm,
            epsilon,
        );
        let gate = format!("{prefix}.gate");
        let up = format!("{prefix}.up");
        self.projection(
            &format!("{prefix}.gate_projection"),
            &ffn_input,
            &gate,
            &format!("{prefix}.gate"),
            layer.gate,
        )?;
        self.projection(
            &format!("{prefix}.up_projection"),
            &ffn_input,
            &up,
            &format!("{prefix}.up"),
            layer.up,
        )?;
        let sigmoid = format!("{prefix}.gate_sigmoid");
        self.standard(
            format!("{prefix}.sigmoid"),
            "Sigmoid",
            &[&gate],
            &[&sigmoid],
            Vec::new(),
        );
        let silu = format!("{prefix}.silu");
        self.standard(
            format!("{prefix}.silu_mul"),
            "Mul",
            &[&gate, &sigmoid],
            &[&silu],
            Vec::new(),
        );
        let gated = format!("{prefix}.gated");
        self.standard(
            format!("{prefix}.swiglu"),
            "Mul",
            &[&silu, &up],
            &[&gated],
            Vec::new(),
        );
        let ffn_output = format!("{prefix}.ffn_output");
        self.projection(
            &format!("{prefix}.down_projection"),
            &gated,
            &ffn_output,
            &format!("{prefix}.down"),
            layer.down,
        )?;
        self.standard(
            format!("{prefix}.ffn_residual"),
            "Add",
            &[input, &ffn_output],
            &[output],
            Vec::new(),
        );
        Ok(())
    }
}

fn build_causal_lm_graph(
    model: CausalLmModel<'_>,
    external: bool,
) -> Result<(ModelProto, Option<Vec<u8>>), OnnxModelError> {
    let tokens = as_i64(model.tokens, "token count")?;
    let past_tokens = as_i64(model.past_tokens, "past token count")?;
    let total_tokens = tokens
        .checked_add(past_tokens)
        .ok_or(OnnxModelError::ShapeOverflow("total token count"))?;
    let total_tokens_usize = model
        .tokens
        .checked_add(model.past_tokens)
        .ok_or(OnnxModelError::ShapeOverflow("total token count"))?;
    let hidden = as_i64(model.embedding.columns, "hidden size")?;
    let vocab = as_i64(model.embedding.rows, "vocabulary")?;
    let n_head = as_i64(model.n_head, "attention head count")?;
    let n_kv_head = as_i64(model.n_kv_head, "KV head count")?;
    let head_dim = as_i64(model.head_dim, "head dimension")?;
    let gqa_repeat = as_i64(model.n_head / model.n_kv_head, "GQA repeat")?;
    let geometry = CausalGraphGeometry {
        past_tokens,
        total_tokens,
        n_head,
        n_kv_head,
        head_dim,
        gqa_repeat,
    };
    let mut graph = if external {
        CausalGraphBuilder::external()
    } else {
        CausalGraphBuilder::default()
    };
    let rotary_state = model
        .rotary
        .map(|rotary| {
            graph.prepare_rotary(RotaryGraphConfig {
                tokens: model.tokens,
                head_dim: model.head_dim,
                past_tokens: model.past_tokens,
                theta: rotary.theta,
            })
        })
        .transpose()?;
    let mask_elements =
        model
            .tokens
            .checked_mul(total_tokens_usize)
            .ok_or(OnnxModelError::ShapeOverflow(
                "attention mask element count",
            ))?;
    if !graph.is_external() && mask_elements > MAX_MODEL_BYTES / core::mem::size_of::<f32>() {
        return Err(OnnxModelError::InvalidModel(format!(
            "attention mask requires {mask_elements} elements, exceeds bounded inline graph"
        )));
    }
    let attention_mask = "attention.attention_mask";
    graph.add_causal_mask(
        attention_mask,
        model.tokens,
        total_tokens_usize,
        model.past_tokens,
        vec![1, tokens, total_tokens],
    );
    graph.add_matrix("tok_embeddings", model.embedding);
    graph.nodes.push(NodeProto {
        input: strings(["tokens", "tok_embeddings.packed", "tok_embeddings.scales"]),
        output: strings(["layer.0.input"]),
        name: "tok_embeddings".to_owned(),
        op_type: ONNX_EMBEDDING_OP_NAME.to_owned(),
        attribute: attributes(hidden, format_code(model.embedding.format)?),
        domain: ONNX_DOMAIN.to_owned(),
    });
    let mut hidden_name = "layer.0.input".to_owned();
    let mut inputs = vec![tensor_value("tokens", TENSOR_INT64, &[tokens])];
    let mut cache_outputs = Vec::with_capacity(model.layers.len() * 2);

    for (index, layer) in model.layers.iter().copied().enumerate() {
        let prefix = format!("layer.{index}");
        let attention_input = format!("{prefix}.attention_input");
        graph.rms_norm(
            &format!("{prefix}.attention_norm"),
            &hidden_name,
            &attention_input,
            layer.attention_norm,
            model.rms_epsilon,
        );
        let query_flat = format!("{prefix}.query_flat");
        let current_k_flat = format!("{prefix}.current_k_flat");
        let current_v_flat = format!("{prefix}.current_v_flat");
        graph.projection(
            &format!("{prefix}.query"),
            &attention_input,
            &query_flat,
            &format!("{prefix}.query"),
            layer.query,
        )?;
        graph.projection(
            &format!("{prefix}.key"),
            &attention_input,
            &current_k_flat,
            &format!("{prefix}.key"),
            layer.key,
        )?;
        graph.projection(
            &format!("{prefix}.value"),
            &attention_input,
            &current_v_flat,
            &format!("{prefix}.value"),
            layer.value,
        )?;
        let query_shape = format!("{prefix}.query_shape");
        let kv_shape = format!("{prefix}.kv_shape");
        graph.add_i64(&query_shape, vec![3], &[tokens, n_head, head_dim]);
        graph.add_i64(&kv_shape, vec![3], &[tokens, n_kv_head, head_dim]);
        let query_tokens = format!("{prefix}.query_tokens");
        let query = format!("{prefix}.query_heads");
        let current_k = format!("{prefix}.current_k");
        let current_v = format!("{prefix}.current_v");
        graph.standard(
            format!("{prefix}.query_reshape"),
            "Reshape",
            &[&query_flat, &query_shape],
            &[&query_tokens],
            Vec::new(),
        );
        graph.standard(
            format!("{prefix}.key_reshape"),
            "Reshape",
            &[&current_k_flat, &kv_shape],
            &[&current_k],
            Vec::new(),
        );
        graph.standard(
            format!("{prefix}.value_reshape"),
            "Reshape",
            &[&current_v_flat, &kv_shape],
            &[&current_v],
            Vec::new(),
        );
        let mut query_ready = query_tokens;
        if let Some(weight) = layer.query_norm {
            let normalized = format!("{prefix}.query_norm.output");
            graph.rms_norm(
                &format!("{prefix}.query_norm"),
                &query_ready,
                &normalized,
                weight,
                model.rms_epsilon,
            );
            query_ready = normalized;
        }
        let mut key_ready = current_k;
        if let Some(weight) = layer.key_norm {
            let normalized = format!("{prefix}.key_norm.output");
            graph.rms_norm(
                &format!("{prefix}.key_norm"),
                &key_ready,
                &normalized,
                weight,
                model.rms_epsilon,
            );
            key_ready = normalized;
        }
        if let Some(rotary) = &rotary_state {
            let rotated_query = format!("{prefix}.query_rope.output");
            graph.rotary(
                &format!("{prefix}.query_rope"),
                &query_ready,
                &rotated_query,
                rotary,
            );
            query_ready = rotated_query;
            let rotated_key = format!("{prefix}.key_rope.output");
            graph.rotary(
                &format!("{prefix}.key_rope"),
                &key_ready,
                &rotated_key,
                rotary,
            );
            key_ready = rotated_key;
        }
        graph.standard(
            format!("{prefix}.query_transpose"),
            "Transpose",
            &[&query_ready],
            &[&query],
            vec![ints_attribute("perm", &[1, 0, 2])],
        );
        let cache =
            graph.cache_and_expand_gqa(index, &key_ready, &current_v, geometry, &mut inputs);
        cache_outputs.extend(cache.declarations);
        let expanded_k = cache.expanded_k;
        let expanded_v = cache.expanded_v;
        let transposed_k = format!("{prefix}.transposed_k");
        let transposed_v = format!("{prefix}.transposed_v");
        graph.standard(
            format!("{prefix}.key_transpose"),
            "Transpose",
            &[&expanded_k],
            &[&transposed_k],
            vec![ints_attribute("perm", &[1, 2, 0])],
        );
        graph.standard(
            format!("{prefix}.value_transpose"),
            "Transpose",
            &[&expanded_v],
            &[&transposed_v],
            vec![ints_attribute("perm", &[1, 0, 2])],
        );
        let scores = format!("{prefix}.scores");
        graph.standard(
            format!("{prefix}.score_matmul"),
            "MatMul",
            &[&query, &transposed_k],
            &[&scores],
            Vec::new(),
        );
        let attention_scale = format!("{prefix}.attention_scale");
        graph.add_f32(
            &attention_scale,
            Vec::new(),
            &[1.0 / (model.head_dim as f32).sqrt()],
        );
        let scaled_scores = format!("{prefix}.scaled_scores");
        graph.standard(
            format!("{prefix}.score_scale"),
            "Mul",
            &[&scores, &attention_scale],
            &[&scaled_scores],
            Vec::new(),
        );
        let masked_scores = format!("{prefix}.masked_scores");
        graph.standard(
            format!("{prefix}.mask"),
            "Add",
            &[&scaled_scores, attention_mask],
            &[&masked_scores],
            Vec::new(),
        );
        let probabilities = format!("{prefix}.probabilities");
        graph.standard(
            format!("{prefix}.softmax"),
            "Softmax",
            &[&masked_scores],
            &[&probabilities],
            vec![int_attribute("axis", -1)],
        );
        let context_heads = format!("{prefix}.context_heads");
        graph.standard(
            format!("{prefix}.context_matmul"),
            "MatMul",
            &[&probabilities, &transposed_v],
            &[&context_heads],
            Vec::new(),
        );
        let context_tokens = format!("{prefix}.context_tokens");
        graph.standard(
            format!("{prefix}.context_transpose"),
            "Transpose",
            &[&context_heads],
            &[&context_tokens],
            vec![ints_attribute("perm", &[1, 0, 2])],
        );
        let hidden_shape = format!("{prefix}.hidden_shape");
        graph.add_i64(&hidden_shape, vec![2], &[tokens, hidden]);
        let context = format!("{prefix}.context");
        graph.standard(
            format!("{prefix}.context_reshape"),
            "Reshape",
            &[&context_tokens, &hidden_shape],
            &[&context],
            Vec::new(),
        );
        let attention_output = format!("{prefix}.attention_output");
        graph.projection(
            &format!("{prefix}.output"),
            &context,
            &attention_output,
            &format!("{prefix}.attention_output"),
            layer.attention_output,
        )?;
        let post_attention = format!("{prefix}.post_attention");
        graph.standard(
            format!("{prefix}.attention_residual"),
            "Add",
            &[&hidden_name, &attention_output],
            &[&post_attention],
            Vec::new(),
        );
        let layer_output = format!("layer.{}.input", index + 1);
        graph.ffn_block(
            &prefix,
            &post_attention,
            &layer_output,
            layer,
            model.rms_epsilon,
        )?;
        hidden_name = layer_output;
    }
    let final_hidden = "final_norm.output";
    graph.rms_norm(
        "final_norm",
        &hidden_name,
        final_hidden,
        model.final_norm,
        model.rms_epsilon,
    );
    graph.nodes.push(NodeProto {
        input: strings([
            final_hidden,
            "tok_embeddings.packed",
            "tok_embeddings.scales",
        ]),
        output: strings(["logits"]),
        name: "lm_head".to_owned(),
        op_type: ONNX_OP_NAME.to_owned(),
        attribute: attributes(hidden, format_code(model.embedding.format)?),
        domain: ONNX_DOMAIN.to_owned(),
    });
    let mut outputs = vec![tensor_value("logits", TENSOR_FLOAT, &[tokens, vocab])];
    outputs.extend(cache_outputs);
    let mut metadata_props = vec![
        metadata("tritium.schema_version", "2"),
        metadata("tritium.graph_kind", "causal-lm"),
        metadata("tritium.source_model_id", model.identity.source_model_id),
        metadata("tritium.tokenizer_id", model.identity.tokenizer_id),
        metadata("tritium.recipe_id", model.identity.recipe_id),
        metadata("tritium.build_id", model.identity.tritium_build_id),
        metadata("tritium.package_id", model.identity.package_id),
        metadata(
            "tritium.coverage.converted_id",
            model.identity.converted_coverage_id,
        ),
        metadata(
            "tritium.coverage.deferred_id",
            model.identity.deferred_coverage_id,
        ),
        metadata("tritium.tokens", &model.tokens.to_string()),
        metadata("tritium.past_tokens", &model.past_tokens.to_string()),
        metadata("tritium.layers", &model.layers.len().to_string()),
        metadata("tritium.hidden", &model.embedding.columns.to_string()),
        metadata("tritium.vocab", &model.embedding.rows.to_string()),
        metadata("tritium.n_head", &model.n_head.to_string()),
        metadata("tritium.n_kv_head", &model.n_kv_head.to_string()),
        metadata("tritium.head_dim", &model.head_dim.to_string()),
        metadata("tritium.tied_embedding_head", "true"),
    ];
    if let Some(rotary) = model.rotary {
        metadata_props.push(metadata("tritium.rope_theta", &rotary.theta.to_string()));
    }
    if let Some(error) = graph.failure {
        return Err(error);
    }
    let external_weights = match graph.storage {
        CausalInitializerStorage::Inline => None,
        CausalInitializerStorage::External(weights) => Some(weights),
    };
    let protobuf = ModelProto {
        ir_version: ONNX_IR_VERSION,
        producer_name: "tritium-onnx".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        domain: ONNX_DOMAIN.to_owned(),
        model_version: 2,
        graph: Some(GraphProto {
            node: graph.nodes,
            name: "tritium.causal_lm".to_owned(),
            initializer: graph.initializers,
            input: inputs,
            output: outputs,
            value_info: Vec::new(),
        }),
        opset_import: vec![
            OperatorSetIdProto {
                domain: String::new(),
                version: ONNX_OPSET,
            },
            OperatorSetIdProto {
                domain: ONNX_DOMAIN.to_owned(),
                version: TRITIUM_OPSET,
            },
        ],
        metadata_props,
    };
    Ok((protobuf, external_weights))
}

#[cfg(test)]
pub(crate) fn encode_kv_attention_test_graph(
    query_tokens: usize,
    past_tokens: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> Vec<u8> {
    let query_tokens = i64::try_from(query_tokens).unwrap();
    let past_tokens = i64::try_from(past_tokens).unwrap();
    let total_tokens = query_tokens.checked_add(past_tokens).unwrap();
    let n_head = i64::try_from(n_head).unwrap();
    let n_kv_head = i64::try_from(n_kv_head).unwrap();
    let head_dim = i64::try_from(head_dim).unwrap();
    let graph = GraphProto {
        node: vec![NodeProto {
            input: strings(["q", "k_cache", "v_cache"]),
            output: strings(["context"]),
            name: "tritium.kv_attention".to_owned(),
            op_type: crate::ONNX_KV_ATTENTION_OP_NAME.to_owned(),
            attribute: vec![
                AttributeProto {
                    name: crate::ATTR_N_HEAD.to_owned(),
                    value: n_head,
                    kind: ATTRIBUTE_INT,
                    ints: Vec::new(),
                },
                AttributeProto {
                    name: crate::ATTR_N_KV_HEAD.to_owned(),
                    value: n_kv_head,
                    kind: ATTRIBUTE_INT,
                    ints: Vec::new(),
                },
                AttributeProto {
                    name: crate::ATTR_HEAD_DIM.to_owned(),
                    value: head_dim,
                    kind: ATTRIBUTE_INT,
                    ints: Vec::new(),
                },
                AttributeProto {
                    name: crate::ATTR_PAST_TOKENS.to_owned(),
                    value: past_tokens,
                    kind: ATTRIBUTE_INT,
                    ints: Vec::new(),
                },
            ],
            domain: ONNX_DOMAIN.to_owned(),
        }],
        name: "tritium.kv_attention.test".to_owned(),
        initializer: Vec::new(),
        input: vec![
            tensor_value("q", TENSOR_FLOAT, &[query_tokens, n_head, head_dim]),
            tensor_value(
                "k_cache",
                TENSOR_FLOAT,
                &[total_tokens, n_kv_head, head_dim],
            ),
            tensor_value(
                "v_cache",
                TENSOR_FLOAT,
                &[total_tokens, n_kv_head, head_dim],
            ),
        ],
        output: vec![tensor_value(
            "context",
            TENSOR_FLOAT,
            &[query_tokens, n_head, head_dim],
        )],
        value_info: Vec::new(),
    };
    ModelProto {
        ir_version: ONNX_IR_VERSION,
        producer_name: "tritium-onnx-test".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        domain: ONNX_DOMAIN.to_owned(),
        model_version: 1,
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: ONNX_DOMAIN.to_owned(),
            version: 2,
        }],
        metadata_props: Vec::new(),
    }
    .encode_to_vec()
}

#[cfg(test)]
pub(crate) fn encode_standard_attention_test_graph(
    query_tokens: usize,
    total_tokens: usize,
    head_dim: usize,
) -> Vec<u8> {
    let query_tokens = i64::try_from(query_tokens).unwrap();
    let total_tokens = i64::try_from(total_tokens).unwrap();
    let head_dim_i64 = i64::try_from(head_dim).unwrap();
    let transpose = |name: &str, input: &str, output: &str, permutation: &[i64]| NodeProto {
        input: strings([input]),
        output: strings([output]),
        name: name.to_owned(),
        op_type: "Transpose".to_owned(),
        attribute: vec![AttributeProto {
            name: "perm".to_owned(),
            value: 0,
            kind: ATTRIBUTE_INTS,
            ints: permutation.to_vec(),
        }],
        domain: String::new(),
    };
    let binary = |name: &str, op_type: &str, left: &str, right: &str, output: &str| NodeProto {
        input: strings([left, right]),
        output: strings([output]),
        name: name.to_owned(),
        op_type: op_type.to_owned(),
        attribute: Vec::new(),
        domain: String::new(),
    };
    let graph = GraphProto {
        node: vec![
            transpose(
                "attention.k_transpose",
                "k_cache",
                "k_transposed",
                &[1, 2, 0],
            ),
            transpose(
                "attention.v_transpose",
                "v_cache",
                "v_transposed",
                &[1, 0, 2],
            ),
            binary("attention.scores", "MatMul", "q", "k_transposed", "scores"),
            binary(
                "attention.scale",
                "Mul",
                "scores",
                "attention.scale",
                "scaled",
            ),
            binary(
                "attention.mask",
                "Add",
                "scaled",
                "attention_mask",
                "masked",
            ),
            NodeProto {
                input: strings(["masked"]),
                output: strings(["probabilities"]),
                name: "attention.softmax".to_owned(),
                op_type: "Softmax".to_owned(),
                attribute: vec![int_attribute("axis", -1)],
                domain: String::new(),
            },
            binary(
                "attention.context",
                "MatMul",
                "probabilities",
                "v_transposed",
                "context",
            ),
        ],
        name: "tritium.standard_attention.test".to_owned(),
        initializer: vec![inline_tensor(
            "attention.scale",
            TENSOR_FLOAT,
            Vec::new(),
            (1.0_f32 / (head_dim as f32).sqrt()).to_le_bytes().to_vec(),
        )],
        input: vec![
            tensor_value("q", TENSOR_FLOAT, &[query_tokens, 1, head_dim_i64]),
            tensor_value("k_cache", TENSOR_FLOAT, &[total_tokens, 1, head_dim_i64]),
            tensor_value("v_cache", TENSOR_FLOAT, &[total_tokens, 1, head_dim_i64]),
            tensor_value(
                "attention_mask",
                TENSOR_FLOAT,
                &[query_tokens, 1, total_tokens],
            ),
        ],
        output: vec![tensor_value(
            "context",
            TENSOR_FLOAT,
            &[query_tokens, 1, head_dim_i64],
        )],
        value_info: Vec::new(),
    };
    ModelProto {
        ir_version: ONNX_IR_VERSION,
        producer_name: "tritium-onnx-test".to_owned(),
        producer_version: env!("CARGO_PKG_VERSION").to_owned(),
        domain: String::new(),
        model_version: 1,
        graph: Some(graph),
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: ONNX_OPSET,
        }],
        metadata_props: Vec::new(),
    }
    .encode_to_vec()
}

fn validate(model: &TiedEmbeddingHeadModel<'_>) -> Result<(), OnnxModelError> {
    for (name, dimension) in [
        ("token count", model.tokens),
        ("vocabulary", model.vocab),
        ("hidden size", model.hidden),
    ] {
        if dimension == 0 {
            return Err(OnnxModelError::EmptyDimension(name));
        }
    }
    for (name, identity) in [
        ("source_model_id", model.source_model_id),
        ("recipe_id", model.recipe_id),
        ("package_id", model.package_id),
    ] {
        if identity.is_empty() {
            return Err(OnnxModelError::EmptyIdentity(name));
        }
    }
    if model.scales.len() != model.vocab {
        return Err(OnnxModelError::ScaleCount {
            expected: model.vocab,
            got: model.scales.len(),
        });
    }
    if let Some((index, _)) = model
        .scales
        .iter()
        .enumerate()
        .find(|(_, scale)| !scale.is_finite() || **scale < 0.0)
    {
        return Err(OnnxModelError::InvalidScale { index });
    }
    let block_bytes = match model.format {
        TernaryFormat::Tq2_0 => TQ2_0_BLOCK_BYTES,
        TernaryFormat::Tq1_0 => TQ1_0_BLOCK_BYTES,
        other => return Err(OnnxModelError::UnsupportedFormat(other)),
    };
    let expected = num_blocks(model.hidden)
        .checked_mul(block_bytes)
        .and_then(|row| row.checked_mul(model.vocab))
        .ok_or(OnnxModelError::ShapeOverflow("packed byte count"))?;
    if model.packed.len() != expected {
        return Err(OnnxModelError::PackedBytes {
            expected,
            got: model.packed.len(),
        });
    }
    validate_packed_payload(model)?;
    for (name, dimension) in [
        ("token count", model.tokens),
        ("vocabulary", model.vocab),
        ("hidden size", model.hidden),
        ("packed byte count", model.packed.len()),
    ] {
        as_i64(dimension, name)?;
    }
    Ok(())
}

fn validate_causal_lm(model: &CausalLmModel<'_>) -> Result<(), OnnxModelError> {
    if model.layers.is_empty() {
        return Err(OnnxModelError::InvalidModel(
            "causal LM requires at least one decoder layer".to_owned(),
        ));
    }
    for (name, value) in [
        ("token count", model.tokens),
        ("attention head count", model.n_head),
        ("KV head count", model.n_kv_head),
        ("head dimension", model.head_dim),
        ("vocabulary", model.embedding.rows),
        ("hidden size", model.embedding.columns),
    ] {
        if value == 0 {
            return Err(OnnxModelError::EmptyDimension(name));
        }
        as_i64(value, name)?;
    }
    as_i64(model.past_tokens, "past token count")?;
    if !model.rms_epsilon.is_finite() || model.rms_epsilon <= 0.0 {
        return Err(OnnxModelError::InvalidModel(
            "RMSNorm epsilon must be finite and positive".to_owned(),
        ));
    }
    if !model.n_head.is_multiple_of(model.n_kv_head) {
        return Err(OnnxModelError::InvalidModel(format!(
            "n_head {} is not divisible by n_kv_head {}",
            model.n_head, model.n_kv_head
        )));
    }
    let hidden = model
        .n_head
        .checked_mul(model.head_dim)
        .ok_or(OnnxModelError::ShapeOverflow("attention hidden size"))?;
    if hidden != model.embedding.columns {
        return Err(OnnxModelError::InvalidModel(format!(
            "n_head * head_dim is {hidden}, hidden size is {}",
            model.embedding.columns
        )));
    }
    if let Some(rotary) = model.rotary {
        if !rotary.theta.is_finite() || rotary.theta <= 0.0 {
            return Err(OnnxModelError::InvalidModel(
                "RoPE theta must be finite and positive".to_owned(),
            ));
        }
        if model.head_dim < 2 || !model.head_dim.is_multiple_of(2) {
            return Err(OnnxModelError::InvalidModel(format!(
                "RoPE requires positive even head_dim, got {}",
                model.head_dim
            )));
        }
    }
    validate_matrix("embedding", model.embedding)?;
    validate_vector("final_norm", model.final_norm, hidden)?;
    for (index, layer) in model.layers.iter().copied().enumerate() {
        validate_vector(
            &format!("layer.{index}.attention_norm"),
            layer.attention_norm,
            hidden,
        )?;
        if let Some(weight) = layer.query_norm {
            validate_vector(&format!("layer.{index}.query_norm"), weight, model.head_dim)?;
        }
        if let Some(weight) = layer.key_norm {
            validate_vector(&format!("layer.{index}.key_norm"), weight, model.head_dim)?;
        }
        validate_vector(&format!("layer.{index}.ffn_norm"), layer.ffn_norm, hidden)?;
        let kv_width = model
            .n_kv_head
            .checked_mul(model.head_dim)
            .ok_or(OnnxModelError::ShapeOverflow("KV projection width"))?;
        validate_matrix_shape(&format!("layer.{index}.query"), layer.query, hidden, hidden)?;
        validate_matrix_shape(&format!("layer.{index}.key"), layer.key, kv_width, hidden)?;
        validate_matrix_shape(
            &format!("layer.{index}.value"),
            layer.value,
            kv_width,
            hidden,
        )?;
        validate_matrix_shape(
            &format!("layer.{index}.attention_output"),
            layer.attention_output,
            hidden,
            hidden,
        )?;
        if layer.gate.rows == 0 || layer.gate.rows != layer.up.rows {
            return Err(OnnxModelError::InvalidModel(format!(
                "layer.{index} gate/up intermediate widths disagree"
            )));
        }
        let intermediate = layer.gate.rows;
        validate_matrix_shape(
            &format!("layer.{index}.gate"),
            layer.gate,
            intermediate,
            hidden,
        )?;
        validate_matrix_shape(&format!("layer.{index}.up"), layer.up, intermediate, hidden)?;
        validate_matrix_shape(
            &format!("layer.{index}.down"),
            layer.down,
            hidden,
            intermediate,
        )?;
    }
    validate_identity_v2(model.identity)
}

fn validate_causal_initializer_budget(model: &CausalLmModel<'_>) -> Result<(), OnnxModelError> {
    fn add(total: &mut usize, bytes: usize) -> Result<(), OnnxModelError> {
        *total = total
            .checked_add(bytes)
            .ok_or(OnnxModelError::ShapeOverflow(
                "causal initializer byte count",
            ))?;
        if *total > MAX_MODEL_BYTES {
            return Err(OnnxModelError::InvalidModel(format!(
                "causal initializers require at least {total} bytes, exceed bounded inline graph"
            )));
        }
        Ok(())
    }
    fn add_vector(total: &mut usize, values: &[f32]) -> Result<(), OnnxModelError> {
        let bytes = values
            .len()
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(OnnxModelError::ShapeOverflow("preserved vector byte count"))?;
        add(total, bytes)
    }
    fn add_matrix(
        total: &mut usize,
        matrix: PackedTernaryMatrix<'_>,
    ) -> Result<(), OnnxModelError> {
        add(total, matrix.packed.len())?;
        add_vector(total, matrix.scales)
    }

    let mut total = 0;
    add_matrix(&mut total, model.embedding)?;
    add_vector(&mut total, model.final_norm)?;
    for layer in model.layers.iter().copied() {
        add_vector(&mut total, layer.attention_norm)?;
        if let Some(weight) = layer.query_norm {
            add_vector(&mut total, weight)?;
        }
        if let Some(weight) = layer.key_norm {
            add_vector(&mut total, weight)?;
        }
        add_vector(&mut total, layer.ffn_norm)?;
        for matrix in [
            layer.query,
            layer.key,
            layer.value,
            layer.attention_output,
            layer.gate,
            layer.up,
            layer.down,
        ] {
            add_matrix(&mut total, matrix)?;
        }
    }
    let total_tokens = model
        .tokens
        .checked_add(model.past_tokens)
        .ok_or(OnnxModelError::ShapeOverflow("total token count"))?;
    let mask_bytes = model
        .tokens
        .checked_mul(total_tokens)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .ok_or(OnnxModelError::ShapeOverflow("attention mask byte count"))?;
    add(&mut total, mask_bytes)?;
    if model.rotary.is_some() {
        let rotary_bytes = model
            .tokens
            .checked_mul(model.head_dim)
            .and_then(|elements| elements.checked_mul(2))
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
            .ok_or(OnnxModelError::ShapeOverflow("RoPE table byte count"))?;
        add(&mut total, rotary_bytes)?;
    }
    Ok(())
}

fn validate_matrix_shape(
    name: &str,
    matrix: PackedTernaryMatrix<'_>,
    rows: usize,
    columns: usize,
) -> Result<(), OnnxModelError> {
    if matrix.rows != rows || matrix.columns != columns {
        return Err(OnnxModelError::InvalidModel(format!(
            "{name} shape [{}, {}] must be [{rows}, {columns}]",
            matrix.rows, matrix.columns
        )));
    }
    validate_matrix(name, matrix)
}

fn validate_matrix(name: &str, matrix: PackedTernaryMatrix<'_>) -> Result<(), OnnxModelError> {
    if matrix.rows == 0 || matrix.columns == 0 {
        return Err(OnnxModelError::InvalidModel(format!(
            "{name} matrix dimensions must be nonzero"
        )));
    }
    validate(&TiedEmbeddingHeadModel {
        tokens: 1,
        vocab: matrix.rows,
        hidden: matrix.columns,
        packed: matrix.packed,
        scales: matrix.scales,
        format: matrix.format,
        source_model_id: "matrix-validation",
        recipe_id: "matrix-validation",
        package_id: "matrix-validation",
    })
    .map_err(|error| OnnxModelError::InvalidModel(format!("{name}: {error}")))
}

fn validate_vector(name: &str, values: &[f32], expected: usize) -> Result<(), OnnxModelError> {
    if values.len() != expected {
        return Err(OnnxModelError::InvalidModel(format!(
            "{name} has {} elements, expected {expected}",
            values.len()
        )));
    }
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(OnnxModelError::InvalidModel(format!(
            "{name}[{index}] must be finite"
        )));
    }
    Ok(())
}

fn validate_v2(model: &TiedEmbeddingHeadModelV2<'_>) -> Result<(), OnnxModelError> {
    validate(&model.legacy())?;
    validate_identity_v2(model.identity)
}

fn validate_identity_v2(identity: OnnxArtifactIdentityV2<'_>) -> Result<(), OnnxModelError> {
    for (name, identity) in [
        ("source_model_id", identity.source_model_id),
        ("tokenizer_id", identity.tokenizer_id),
        ("recipe_id", identity.recipe_id),
        ("tritium_build_id", identity.tritium_build_id),
        ("package_id", identity.package_id),
        ("converted_coverage_id", identity.converted_coverage_id),
        ("deferred_coverage_id", identity.deferred_coverage_id),
    ] {
        if identity.is_empty() {
            return Err(OnnxModelError::EmptyIdentity(name));
        }
    }
    Ok(())
}

fn identity_metadata_v2(identity: OnnxArtifactIdentityV2<'_>) -> Vec<StringStringEntryProto> {
    vec![
        metadata("tritium.tokenizer_id", identity.tokenizer_id),
        metadata("tritium.build_id", identity.tritium_build_id),
        metadata(
            "tritium.coverage.converted_id",
            identity.converted_coverage_id,
        ),
        metadata(
            "tritium.coverage.deferred_id",
            identity.deferred_coverage_id,
        ),
    ]
}

fn validate_packed_payload(model: &TiedEmbeddingHeadModel<'_>) -> Result<(), OnnxModelError> {
    let block_bytes = match model.format {
        TernaryFormat::Tq2_0 => TQ2_0_BLOCK_BYTES,
        TernaryFormat::Tq1_0 => TQ1_0_BLOCK_BYTES,
        other => return Err(OnnxModelError::UnsupportedFormat(other)),
    };
    let blocks = num_blocks(model.hidden);
    let row_bytes = blocks
        .checked_mul(block_bytes)
        .ok_or(OnnxModelError::ShapeOverflow("packed row byte count"))?;
    let mut trits = vec![Trit::ZERO; model.hidden];
    let mut scales = vec![f16::ONE; blocks];
    for (row, packed) in model.packed.chunks_exact(row_bytes).enumerate() {
        let result = match model.format {
            TernaryFormat::Tq2_0 => unpack_tq2_0_row(packed, &mut trits, &mut scales),
            TernaryFormat::Tq1_0 => unpack_tq1_0_row(packed, &mut trits, &mut scales),
            other => return Err(OnnxModelError::UnsupportedFormat(other)),
        };
        result.map_err(|error| OnnxModelError::InvalidPackedRow {
            row,
            reason: error.to_string(),
        })?;
        if let Some((block, scale)) = scales
            .iter()
            .copied()
            .enumerate()
            .find(|(_, scale)| *scale != f16::ONE)
        {
            return Err(OnnxModelError::InvalidPackedRow {
                row,
                reason: format!("block {block} has non-unit scale {scale:?}"),
            });
        }
    }
    Ok(())
}

fn as_i64(value: usize, name: &'static str) -> Result<i64, OnnxModelError> {
    i64::try_from(value).map_err(|_| OnnxModelError::ShapeOverflow(name))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, OnnxModelError> {
    value
        .checked_add(alignment - 1)
        .map(|aligned| aligned / alignment * alignment)
        .ok_or(OnnxModelError::ShapeOverflow("external data alignment"))
}

fn scale_bytes(scales: &[f32]) -> Vec<u8> {
    scales
        .iter()
        .flat_map(|scale| scale.to_le_bytes())
        .collect()
}

fn format_code(format: TernaryFormat) -> Result<i64, OnnxModelError> {
    match format {
        TernaryFormat::Tq2_0 => Ok(0),
        TernaryFormat::Tq1_0 => Ok(1),
        other => Err(OnnxModelError::UnsupportedFormat(other)),
    }
}

fn strings<const N: usize>(values: [&str; N]) -> Vec<String> {
    values.into_iter().map(str::to_owned).collect()
}

fn attributes(k: i64, format: i64) -> Vec<AttributeProto> {
    vec![int_attribute(ATTR_K, k), int_attribute(ATTR_FORMAT, format)]
}

fn int_attribute(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_owned(),
        value,
        kind: ATTRIBUTE_INT,
        ints: Vec::new(),
    }
}

fn ints_attribute(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_owned(),
        value: 0,
        kind: ATTRIBUTE_INTS,
        ints: values.to_vec(),
    }
}

fn metadata(key: &str, value: &str) -> StringStringEntryProto {
    StringStringEntryProto {
        key: key.to_owned(),
        value: value.to_owned(),
    }
}

fn inline_tensor(name: &str, data_type: i32, dims: Vec<i64>, raw_data: Vec<u8>) -> TensorProto {
    TensorProto {
        dims,
        data_type,
        name: name.to_owned(),
        raw_data,
        external_data: Vec::new(),
        data_location: 0,
    }
}

fn external_tensor(
    name: &str,
    data_type: i32,
    dims: Vec<i64>,
    offset: i64,
    length: i64,
) -> TensorProto {
    TensorProto {
        dims,
        data_type,
        name: name.to_owned(),
        raw_data: Vec::new(),
        external_data: vec![
            metadata("location", EXTERNAL_WEIGHTS_FILE),
            metadata("offset", &offset.to_string()),
            metadata("length", &length.to_string()),
        ],
        data_location: EXTERNAL_DATA,
    }
}

fn tensor_value(name: &str, elem_type: i32, dimensions: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_owned(),
        r#type: Some(TypeProto {
            tensor_type: Some(TensorTypeProto {
                elem_type,
                shape: Some(TensorShapeProto {
                    dim: dimensions
                        .iter()
                        .map(|&dim_value| TensorDimensionProto { dim_value })
                        .collect(),
                }),
            }),
        }),
    }
}

#[derive(Clone, PartialEq, Message)]
struct ModelProto {
    #[prost(int64, tag = "1")]
    ir_version: i64,
    #[prost(string, tag = "2")]
    producer_name: String,
    #[prost(string, tag = "3")]
    producer_version: String,
    #[prost(string, tag = "4")]
    domain: String,
    #[prost(int64, tag = "5")]
    model_version: i64,
    #[prost(message, optional, tag = "7")]
    graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    opset_import: Vec<OperatorSetIdProto>,
    #[prost(message, repeated, tag = "14")]
    metadata_props: Vec<StringStringEntryProto>,
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
    #[prost(message, repeated, tag = "5")]
    initializer: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    output: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "13")]
    value_info: Vec<ValueInfoProto>,
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
    #[prost(int64, repeated, packed = "true", tag = "8")]
    ints: Vec<i64>,
}

#[derive(Clone, PartialEq, Message)]
struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    data_type: i32,
    #[prost(string, tag = "8")]
    name: String,
    #[prost(bytes = "vec", tag = "9")]
    raw_data: Vec<u8>,
    #[prost(message, repeated, tag = "13")]
    external_data: Vec<StringStringEntryProto>,
    #[prost(int32, tag = "14")]
    data_location: i32,
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

#[derive(Clone, PartialEq, Message)]
struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_format::{pack_tq1_0_row, pack_tq2_0_row};

    fn unit_packed(format: TernaryFormat, rows: usize) -> Vec<u8> {
        let trits = vec![Trit::ZERO; 256];
        let scales = [f16::ONE];
        let block_bytes = match format {
            TernaryFormat::Tq2_0 => TQ2_0_BLOCK_BYTES,
            TernaryFormat::Tq1_0 => TQ1_0_BLOCK_BYTES,
            other => panic!("unsupported test format {other}"),
        };
        let mut packed = vec![0; block_bytes * rows];
        for row in packed.chunks_exact_mut(block_bytes) {
            match format {
                TernaryFormat::Tq2_0 => pack_tq2_0_row(&trits, &scales, row).unwrap(),
                TernaryFormat::Tq1_0 => pack_tq1_0_row(&trits, &scales, row).unwrap(),
                other => panic!("unsupported test format {other}"),
            }
        }
        packed
    }

    #[test]
    fn validation_is_fail_closed() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let base = TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        assert!(encode_tied_embedding_head(base).is_ok());
        assert!(matches!(
            encode_tied_embedding_head(TiedEmbeddingHeadModel {
                scales: &[f32::NAN],
                ..base
            }),
            Err(OnnxModelError::InvalidScale { .. })
        ));
        assert!(matches!(
            encode_tied_embedding_head(TiedEmbeddingHeadModel {
                packed: &[],
                ..base
            }),
            Err(OnnxModelError::PackedBytes { .. })
        ));
        assert!(matches!(
            encode_tied_embedding_head(TiedEmbeddingHeadModel {
                source_model_id: "",
                ..base
            }),
            Err(OnnxModelError::EmptyIdentity("source_model_id"))
        ));
    }

    #[test]
    fn encoding_is_deterministic() {
        let packed = unit_packed(TernaryFormat::Tq1_0, 2);
        let model = TiedEmbeddingHeadModel {
            tokens: 3,
            vocab: 2,
            hidden: 256,
            packed: &packed,
            scales: &[1.0, 0.5],
            format: TernaryFormat::Tq1_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        assert_eq!(
            encode_tied_embedding_head(model).unwrap(),
            encode_tied_embedding_head(model).unwrap()
        );
    }

    #[test]
    fn external_encoding_binds_layout_and_digest() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 2);
        let model = TiedEmbeddingHeadModel {
            tokens: 2,
            vocab: 2,
            hidden: 256,
            packed: &packed,
            scales: &[1.0, 0.5],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        let encoded = encode_external_tied_embedding_head(model).unwrap();
        let receipt = verify_external_tied_embedding_head(
            encoded.model_bytes.as_slice(),
            encoded.weights_bytes.as_slice(),
        )
        .unwrap();
        assert_eq!(receipt.tokens, 2);
        assert_eq!(receipt.vocab, 2);
        assert_eq!(receipt.hidden, 256);
        assert_eq!(
            encoded.weights_blake3,
            *blake3::hash(&encoded.weights_bytes).as_bytes()
        );
        assert_eq!(encoded.weights_bytes.len() % 4, 0);
        let protobuf = ModelProto::decode(encoded.model_bytes.as_slice()).unwrap();
        let graph = protobuf.graph.unwrap();
        assert_eq!(graph.initializer.len(), 2);
        assert!(graph.initializer.iter().all(|tensor| {
            tensor.raw_data.is_empty()
                && tensor.data_location == EXTERNAL_DATA
                && tensor
                    .external_data
                    .iter()
                    .any(|entry| entry.key == "location" && entry.value == EXTERNAL_WEIGHTS_FILE)
        }));
        let expected_digest = blake3::Hash::from_bytes(encoded.weights_blake3)
            .to_hex()
            .to_string();
        assert!(protobuf.metadata_props.iter().any(|entry| {
            entry.key == "tritium.external_data.blake3" && entry.value == expected_digest
        }));

        let mut corrupted = encoded.weights_bytes.clone();
        corrupted[0] ^= 1;
        assert!(matches!(
            verify_external_tied_embedding_head(&encoded.model_bytes, &corrupted),
            Err(OnnxModelError::ExternalDataMismatch(_))
        ));
    }

    #[test]
    fn schema_v2_binds_complete_identity_without_breaking_v1() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 2);
        let identity = OnnxArtifactIdentityV2 {
            source_model_id: "source@revision",
            tokenizer_id: "tokenizer@revision",
            recipe_id: "recipe@digest",
            tritium_build_id: "tritium@git-sha",
            package_id: "package@digest",
            converted_coverage_id: "converted@digest",
            deferred_coverage_id: "deferred@digest",
        };
        let model = TiedEmbeddingHeadModelV2 {
            tokens: 2,
            vocab: 2,
            hidden: 256,
            packed: &packed,
            scales: &[1.0, 0.5],
            format: TernaryFormat::Tq2_0,
            identity,
        };
        let encoded = encode_external_tied_embedding_head_v2(model).unwrap();
        let receipt = verify_external_tied_embedding_head_v2(
            &encoded.model_bytes,
            &encoded.weights_bytes,
            identity,
        )
        .unwrap();
        assert_eq!(receipt.model.source_model_id, identity.source_model_id);
        assert_eq!(receipt.model.recipe_id, identity.recipe_id);
        assert_eq!(receipt.model.package_id, identity.package_id);
        assert_eq!(receipt.identity.tokenizer_id, identity.tokenizer_id);
        assert_eq!(receipt.identity.tritium_build_id, identity.tritium_build_id);
        assert_eq!(
            receipt.identity.converted_coverage_id,
            identity.converted_coverage_id
        );
        assert_eq!(
            receipt.identity.deferred_coverage_id,
            identity.deferred_coverage_id
        );
        assert!(
            verify_external_tied_embedding_head(&encoded.model_bytes, &encoded.weights_bytes)
                .is_err()
        );

        let mut protobuf = ModelProto::decode(encoded.model_bytes.as_slice()).unwrap();
        protobuf
            .metadata_props
            .retain(|entry| entry.key != "tritium.tokenizer_id");
        assert!(matches!(
            verify_external_tied_embedding_head_v2(
                &protobuf.encode_to_vec(),
                &encoded.weights_bytes,
                identity,
            ),
            Err(OnnxModelError::InvalidModel(_))
        ));

        let legacy = encode_external_tied_embedding_head(model.legacy()).unwrap();
        assert!(
            verify_external_tied_embedding_head(&legacy.model_bytes, &legacy.weights_bytes).is_ok()
        );
        assert!(
            verify_external_tied_embedding_head_v2(
                &legacy.model_bytes,
                &legacy.weights_bytes,
                identity,
            )
            .is_err()
        );
        assert!(matches!(
            verify_external_tied_embedding_head_v2(
                &encoded.model_bytes,
                &encoded.weights_bytes,
                OnnxArtifactIdentityV2 {
                    package_id: "different-package",
                    ..identity
                },
            ),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("package_id")
        ));
        assert!(matches!(
            encode_tied_embedding_head_v2(TiedEmbeddingHeadModelV2 {
                identity: OnnxArtifactIdentityV2 {
                    tokenizer_id: "",
                    ..identity
                },
                ..model
            }),
            Err(OnnxModelError::EmptyIdentity("tokenizer_id"))
        ));
    }

    #[test]
    fn unsupported_graph_diagnostics_are_typed_and_exhaustive() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let model = TiedEmbeddingHeadModelV2 {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            identity: OnnxArtifactIdentityV2 {
                source_model_id: "source",
                tokenizer_id: "tokenizer",
                recipe_id: "recipe",
                tritium_build_id: "build",
                package_id: "package",
                converted_coverage_id: "converted",
                deferred_coverage_id: "deferred",
            },
        };
        let mut protobuf =
            ModelProto::decode(encode_tied_embedding_head_v2(model).unwrap().as_slice()).unwrap();
        let graph = protobuf.graph.as_mut().unwrap();
        graph.node[0].domain = "ai.onnx".to_owned();
        graph.node[0].op_type = "Gather".to_owned();
        graph.node[1].attribute.push(AttributeProto {
            name: "axis".to_owned(),
            value: 0,
            kind: ATTRIBUTE_INT,
            ints: Vec::new(),
        });
        graph.initializer[0].data_type = TENSOR_INT64;
        protobuf.metadata_props.push(metadata(
            "tritium.coverage.unresolved.language.layers.0.mlp",
            "preserved",
        ));

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Node,
                    subject: "tritium.embedding".to_owned(),
                    reason: "unsupported operator ai.onnx::Gather".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.embedding.K".to_owned(),
                    reason: "unsupported attribute".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.embedding.format".to_owned(),
                    reason: "unsupported attribute".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.lm_head.axis".to_owned(),
                    reason: "unsupported attribute".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "initializer tritium.packed".to_owned(),
                    reason: "expected ONNX dtype 2, got 7".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Coverage,
                    subject: "language.layers.0.mlp".to_owned(),
                    reason: "coverage item is unresolved: preserved".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn unsupported_graph_diagnostics_reject_invalid_attribute_values() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let model = TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        let mut protobuf =
            ModelProto::decode(encode_tied_embedding_head(model).unwrap().as_slice()).unwrap();
        let graph = protobuf.graph.as_mut().unwrap();
        graph.node[0]
            .attribute
            .iter_mut()
            .find(|attribute| attribute.name == ATTR_K)
            .unwrap()
            .value = 0;
        graph.node[1]
            .attribute
            .iter_mut()
            .find(|attribute| attribute.name == ATTR_FORMAT)
            .unwrap()
            .value = 9;

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.embedding.K".to_owned(),
                    reason: "K must be positive, got 0".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.lm_head.format".to_owned(),
                    reason: "unsupported format code 9".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn unsupported_graph_diagnostics_name_missing_and_duplicate_attributes() {
        let packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let model = TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        let mut protobuf =
            ModelProto::decode(encode_tied_embedding_head(model).unwrap().as_slice()).unwrap();
        let graph = protobuf.graph.as_mut().unwrap();
        graph.node[0]
            .attribute
            .retain(|attribute| attribute.name != ATTR_K);
        let duplicate_format = graph.node[1]
            .attribute
            .iter()
            .find(|attribute| attribute.name == ATTR_FORMAT)
            .unwrap()
            .clone();
        graph.node[1].attribute.push(duplicate_format);

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.embedding.K".to_owned(),
                    reason: "missing required attribute".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "tritium.lm_head.format".to_owned(),
                    reason: "duplicate attribute appears 2 times".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn unsupported_graph_diagnostics_cover_every_typed_tensor_declaration() {
        const ONNX_TENSOR_STRING: i32 = 8;
        let packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let model = TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        let mut protobuf =
            ModelProto::decode(encode_tied_embedding_head(model).unwrap().as_slice()).unwrap();
        let graph = protobuf.graph.as_mut().unwrap();
        let mut extra_initializer = graph.initializer[0].clone();
        extra_initializer.name = "extra".to_owned();
        extra_initializer.data_type = ONNX_TENSOR_STRING;
        graph.initializer.push(extra_initializer);
        graph
            .input
            .push(tensor_value("extra_input", ONNX_TENSOR_STRING, &[1]));
        graph
            .output
            .push(tensor_value("extra_output", ONNX_TENSOR_STRING, &[1]));
        graph
            .value_info
            .push(tensor_value("hidden", ONNX_TENSOR_STRING, &[1, 256]));
        graph
            .value_info
            .push(tensor_value("scratch", ONNX_TENSOR_STRING, &[1]));

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "initializer extra".to_owned(),
                    reason: "dtype 8 has no supported contract for this tensor".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "input extra_input".to_owned(),
                    reason: "dtype 8 has no supported contract for this tensor".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "output extra_output".to_owned(),
                    reason: "dtype 8 has no supported contract for this tensor".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "value_info hidden".to_owned(),
                    reason: "expected ONNX dtype 1, got 8".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Dtype,
                    subject: "value_info scratch".to_owned(),
                    reason: "dtype 8 has no supported contract for this tensor".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn supported_graph_has_no_unsupported_diagnostics() {
        let packed = unit_packed(TernaryFormat::Tq1_0, 1);
        let model = TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq1_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        };
        let encoded = encode_tied_embedding_head(model).unwrap();
        assert!(diagnose_unsupported_graph(&encoded).unwrap().is_empty());
    }

    #[test]
    fn supported_kv_attention_graph_has_no_unsupported_diagnostics() {
        let encoded = encode_kv_attention_test_graph(1, 2, 2, 1, 4);
        assert!(diagnose_unsupported_graph(&encoded).unwrap().is_empty());
    }

    #[test]
    fn supported_standard_attention_graph_has_no_unsupported_diagnostics() {
        let encoded = encode_standard_attention_test_graph(2, 2, 1);
        assert!(diagnose_unsupported_graph(&encoded).unwrap().is_empty());
    }

    #[test]
    fn standard_attention_diagnostics_reject_invalid_permutation_and_softmax_axis() {
        let encoded = encode_standard_attention_test_graph(2, 2, 1);
        let mut protobuf = ModelProto::decode(encoded.as_slice()).unwrap();
        let graph = protobuf.graph.as_mut().unwrap();
        graph.node[0].attribute[0].ints = vec![0, 0, 2];
        graph
            .node
            .iter_mut()
            .find(|node| node.op_type == "Softmax")
            .unwrap()
            .attribute[0]
            .value = 0;

        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "attention.k_transpose.perm".to_owned(),
                    reason: "expected a rank-3 permutation of [0, 1, 2], got [0, 0, 2]".to_owned(),
                },
                UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject: "attention.softmax.axis".to_owned(),
                    reason: "attention softmax axis must be -1, got 0".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn concat_diagnostics_enforce_cache_and_rotary_axes_by_role() {
        let concat = |inputs: &[&str], output: &str, axis: i64| NodeProto {
            input: inputs.iter().map(|value| (*value).to_owned()).collect(),
            output: strings([output]),
            name: "renamed-cosmetic-node".to_owned(),
            op_type: "Concat".to_owned(),
            attribute: vec![int_attribute("axis", axis)],
            domain: String::new(),
        };
        let unary = |op_type: &str, input: &str, output: &str| NodeProto {
            input: strings([input]),
            output: strings([output]),
            name: String::new(),
            op_type: op_type.to_owned(),
            attribute: Vec::new(),
            domain: String::new(),
        };
        let protobuf = ModelProto {
            ir_version: ONNX_IR_VERSION,
            producer_name: "tritium-onnx-test".to_owned(),
            producer_version: env!("CARGO_PKG_VERSION").to_owned(),
            domain: String::new(),
            model_version: 1,
            graph: Some(GraphProto {
                node: vec![
                    concat(&["past_k.0", "current_k"], "present_k.0", -1),
                    unary("Slice", "query", "first_half"),
                    unary("Neg", "second_half", "negated_second"),
                    concat(&["negated_second", "first_half"], "rotated", 0),
                ],
                name: "concat-axis-mutations".to_owned(),
                initializer: Vec::new(),
                input: vec![tensor_value("past_k.0", TENSOR_FLOAT, &[1, 1, 2])],
                output: vec![tensor_value("present_k.0", TENSOR_FLOAT, &[2, 1, 2])],
                value_info: Vec::new(),
            }),
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: ONNX_OPSET,
            }],
            metadata_props: Vec::new(),
        };
        let diagnostics = diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap();
        assert_eq!(diagnostics.len(), 2);
        assert!(
            diagnostics[0]
                .reason
                .contains("KV cache concat axis must be 0")
        );
        assert!(
            diagnostics[1]
                .reason
                .contains("RoPE concat axis must be -1")
        );
    }

    #[test]
    fn kv_attention_diagnostics_reject_nondivisible_gqa_heads() {
        let encoded = encode_kv_attention_test_graph(1, 0, 3, 2, 4);
        assert_eq!(
            diagnose_unsupported_graph(&encoded).unwrap(),
            vec![UnsupportedGraphDiagnostic {
                kind: UnsupportedGraphItemKind::Attribute,
                subject: "tritium.kv_attention.n_kv_head".to_owned(),
                reason: "n_head 3 is not divisible by n_kv_head 2".to_owned(),
            }]
        );
    }

    #[test]
    fn kv_attention_diagnostics_reject_opset_one_import() {
        let encoded = encode_kv_attention_test_graph(1, 0, 2, 1, 4);
        let mut protobuf = ModelProto::decode(encoded.as_slice()).unwrap();
        protobuf.opset_import[0].version = 1;
        assert_eq!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()).unwrap(),
            vec![UnsupportedGraphDiagnostic {
                kind: UnsupportedGraphItemKind::Node,
                subject: "tritium.kv_attention".to_owned(),
                reason: "operator requires com.tritium opset 2, imported 1".to_owned(),
            }]
        );
    }

    #[test]
    fn diagnostics_reject_duplicate_tritium_opset_imports() {
        let encoded = encode_kv_attention_test_graph(1, 0, 2, 1, 4);
        let mut protobuf = ModelProto::decode(encoded.as_slice()).unwrap();
        protobuf.opset_import.push(OperatorSetIdProto {
            domain: ONNX_DOMAIN.to_owned(),
            version: 3,
        });
        assert!(matches!(
            diagnose_unsupported_graph(&protobuf.encode_to_vec()),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("duplicate opset import")
        ));
    }

    #[test]
    fn encoder_rejects_non_unit_internal_scales() {
        let mut packed = unit_packed(TernaryFormat::Tq2_0, 1);
        let scale_offset = TQ2_0_BLOCK_BYTES - core::mem::size_of::<f16>();
        packed[scale_offset..].copy_from_slice(&f16::ZERO.to_le_bytes());
        let result = encode_tied_embedding_head(TiedEmbeddingHeadModel {
            tokens: 1,
            vocab: 1,
            hidden: 256,
            packed: &packed,
            scales: &[1.0],
            format: TernaryFormat::Tq2_0,
            source_model_id: "source",
            recipe_id: "recipe",
            package_id: "package",
        });
        assert!(matches!(
            result,
            Err(OnnxModelError::InvalidPackedRow { .. })
        ));
    }

    #[test]
    fn rotary_tables_remain_finite_at_extreme_valid_theta_and_position() {
        let (cos, sin) = rotary_tables(RotaryGraphConfig {
            tokens: 2,
            head_dim: 128,
            past_tokens: 10_000_000,
            theta: f32::MIN_POSITIVE,
        })
        .unwrap();
        assert_eq!(cos.len(), 256);
        assert_eq!(sin.len(), 256);
        assert!(cos.iter().chain(&sin).all(|value| value.is_finite()));
    }

    #[test]
    fn rotary_tables_reject_oversized_allocation_before_building_values() {
        let result = rotary_tables(RotaryGraphConfig {
            tokens: MAX_MODEL_BYTES,
            head_dim: 2,
            past_tokens: 0,
            theta: 10_000.0,
        });
        assert!(matches!(
            result,
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("RoPE tables require")
        ));
    }

    #[test]
    fn causal_initializer_budget_rejects_aggregate_payload_before_cloning() {
        let empty_matrix = PackedTernaryMatrix {
            rows: 1,
            columns: 1,
            packed: &[],
            scales: &[],
            format: TernaryFormat::Tq2_0,
        };
        let model = CausalLmModel {
            tokens: 4097,
            past_tokens: 0,
            n_head: 1,
            n_kv_head: 1,
            head_dim: 1,
            rotary: None,
            rms_epsilon: 1.0e-5,
            embedding: empty_matrix,
            final_norm: &[],
            layers: &[],
            identity: OnnxArtifactIdentityV2 {
                source_model_id: "source",
                tokenizer_id: "tokenizer",
                recipe_id: "recipe",
                tritium_build_id: "build",
                package_id: "package",
                converted_coverage_id: "converted",
                deferred_coverage_id: "deferred",
            },
        };
        assert!(matches!(
            validate_causal_initializer_budget(&model),
            Err(OnnxModelError::InvalidModel(reason))
                if reason.contains("causal initializers require at least")
        ));
    }
}
