//! Deterministic ONNX protobuf serialization for Tritium inference graphs.

use prost::Message;
use tritium_core::TernaryFormat;
use tritium_format::{TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, num_blocks};

use crate::{ATTR_FORMAT, ATTR_K, ONNX_DOMAIN, ONNX_EMBEDDING_OP_NAME, ONNX_OP_NAME};

const ONNX_IR_VERSION: i64 = 10;
const ONNX_OPSET: i64 = 21;
const TRITIUM_OPSET: i64 = 1;
const TENSOR_FLOAT: i32 = 1;
const TENSOR_UINT8: i32 = 2;
const TENSOR_INT64: i32 = 7;
const ATTRIBUTE_INT: i32 = 2;

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
    let tokens = as_i64(model.tokens, "token count")?;
    let vocab = as_i64(model.vocab, "vocabulary")?;
    let hidden = as_i64(model.hidden, "hidden size")?;
    let packed_bytes = as_i64(model.packed.len(), "packed byte count")?;
    let format = format_code(model.format)?;
    let scale_bytes = model
        .scales
        .iter()
        .flat_map(|scale| scale.to_le_bytes())
        .collect();

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
        initializer: vec![
            TensorProto {
                dims: vec![packed_bytes],
                data_type: TENSOR_UINT8,
                name: "tritium.packed".to_owned(),
                raw_data: model.packed.to_vec(),
            },
            TensorProto {
                dims: vec![vocab],
                data_type: TENSOR_FLOAT,
                name: "tritium.scales".to_owned(),
                raw_data: scale_bytes,
            },
        ],
        input: vec![tensor_value("tokens", TENSOR_INT64, &[tokens])],
        output: vec![tensor_value("logits", TENSOR_FLOAT, &[tokens, vocab])],
    };
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
        metadata_props: vec![
            metadata("tritium.schema_version", "1"),
            metadata("tritium.source_model_id", model.source_model_id),
            metadata("tritium.recipe_id", model.recipe_id),
            metadata("tritium.package_id", model.package_id),
            metadata("tritium.weight_format", &model.format.to_string()),
            metadata("tritium.tied_embedding_head", "true"),
        ],
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

fn as_i64(value: usize, name: &'static str) -> Result<i64, OnnxModelError> {
    i64::try_from(value).map_err(|_| OnnxModelError::ShapeOverflow(name))
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

    #[test]
    fn validation_is_fail_closed() {
        let packed = vec![0; TQ2_0_BLOCK_BYTES];
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
        let packed = vec![0; TQ1_0_BLOCK_BYTES * 2];
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
}
