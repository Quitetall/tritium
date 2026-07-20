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
            } else if attribute.name == "axis" && attribute.value != -1 {
                diagnostics.push(UnsupportedGraphDiagnostic {
                    kind: UnsupportedGraphItemKind::Attribute,
                    subject,
                    reason: format!("attention softmax axis must be -1, got {}", attribute.value),
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
            "tritium.packed" => Some(TENSOR_UINT8),
            "tritium.scales" => Some(TENSOR_FLOAT),
            "attention.scale" => Some(TENSOR_FLOAT),
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
        ("", "MatMul" | "Mul" | "Add") => &[],
        _ => &[],
    }
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
        ("", "Transpose" | "MatMul" | "Mul" | "Add" | "Softmax") => {
            standard_opset == Some(ONNX_OPSET)
        }
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
}
