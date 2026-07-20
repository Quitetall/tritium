//! Deterministic ONNX protobuf serialization for Tritium inference graphs.

use std::collections::BTreeMap;

use half::f16;
use prost::Message;
use tritium_core::{TernaryFormat, Trit};
use tritium_format::{
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, unpack_tq1_0_row, unpack_tq2_0_row,
};

use crate::{ATTR_FORMAT, ATTR_K, ONNX_DOMAIN, ONNX_EMBEDDING_OP_NAME, ONNX_OP_NAME};

const ONNX_IR_VERSION: i64 = 10;
const ONNX_OPSET: i64 = 21;
const TRITIUM_OPSET: i64 = 1;
const TENSOR_FLOAT: i32 = 1;
const TENSOR_UINT8: i32 = 2;
const TENSOR_INT64: i32 = 7;
const ATTRIBUTE_INT: i32 = 2;
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
        vec![
            metadata("tritium.external_data.file", EXTERNAL_WEIGHTS_FILE),
            metadata("tritium.external_data.bytes", &weights_len.to_string()),
            metadata("tritium.external_data.blake3", &digest_hex),
        ],
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
    require_metadata(&metadata, "tritium.schema_version", "1")?;
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
    let source_model_id = metadata_value(&metadata, "tritium.source_model_id")?.to_owned();
    let recipe_id = metadata_value(&metadata, "tritium.recipe_id")?.to_owned();
    let package_id = metadata_value(&metadata, "tritium.package_id")?.to_owned();
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
    let specification = TiedEmbeddingHeadModel {
        tokens,
        vocab,
        hidden,
        packed: &weights_bytes[..packed_len],
        scales: &scales,
        format,
        source_model_id: &source_model_id,
        recipe_id: &recipe_id,
        package_id: &package_id,
    };
    let expected = encode_external_tied_embedding_head(specification)?;
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
    Ok(VerifiedExternalOnnxModel {
        model_blake3: *blake3::hash(model_bytes).as_bytes(),
        weights_blake3: *actual_weights_hash.as_bytes(),
        weights_bytes: weights_bytes.len(),
        tokens,
        vocab,
        hidden,
        source_model_id,
        recipe_id,
        package_id,
    })
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
    };
    let mut metadata_props = vec![
        metadata("tritium.schema_version", "1"),
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
        model_version: 1,
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
