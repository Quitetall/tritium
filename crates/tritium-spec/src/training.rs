//! Language-neutral portable-training operation manifest.

use core::fmt;

use serde::Deserialize;

const SCHEMA_ID: &str = "tritium.training_op_manifest";
const SCHEMA_VERSION: u32 = 1;
const DTYPE: &str = "f32";
const CANONICAL_JSON: &[u8] = include_bytes!("../../../spec/training/v1/manifest.json");

/// Semantic category for one portable training operation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingOpCategoryV1 {
    /// Differentiable graph operation.
    Graph,
    /// Scalar loss operation.
    Loss,
    /// Stateful optimizer update.
    Optimizer,
    /// Checkpoint or artifact lifecycle operation.
    Lifecycle,
}

/// First-order reverse-mode behavior declared by an operation.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingVjpV1 {
    /// Operation has no reverse-mode derivative.
    None,
    /// Operation implements a first-order vector-Jacobian product.
    FirstOrder,
}

/// Frozen descriptor for one `TrainingOpManifestV1` operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainingOpDescriptorV1 {
    /// Permanent lowercase ASCII operation identifier.
    pub id: &'static str,
    /// Semantic category.
    pub category: TrainingOpCategoryV1,
    /// Whether the operation has forward tensor semantics.
    pub forward: bool,
    /// Reverse-mode derivative behavior.
    pub vjp: TrainingVjpV1,
    /// Whether execution mutates caller-visible tensor or optimizer state.
    pub mutates: bool,
    /// Persistent planes required for checkpoint/resume.
    pub checkpoint_planes: &'static [&'static str],
}

const NO_PLANES: &[&str] = &[];
const PARAMETER: &[&str] = &["parameter"];
const ADAM_PLANES: &[&str] = &["parameter", "moment1", "moment2"];
const INT8_ADAM_PLANES: &[&str] = &[
    "parameter",
    "moment1_q8",
    "moment2_q8",
    "moment1_scale",
    "moment2_scale",
];
const MUON_PLANES: &[&str] = &["parameter", "momentum"];

const fn graph(id: &'static str) -> TrainingOpDescriptorV1 {
    TrainingOpDescriptorV1 {
        id,
        category: TrainingOpCategoryV1::Graph,
        forward: true,
        vjp: TrainingVjpV1::FirstOrder,
        mutates: false,
        checkpoint_planes: NO_PLANES,
    }
}

const fn loss(id: &'static str) -> TrainingOpDescriptorV1 {
    TrainingOpDescriptorV1 {
        id,
        category: TrainingOpCategoryV1::Loss,
        forward: true,
        vjp: TrainingVjpV1::FirstOrder,
        mutates: false,
        checkpoint_planes: NO_PLANES,
    }
}

const fn optimizer(
    id: &'static str,
    checkpoint_planes: &'static [&'static str],
) -> TrainingOpDescriptorV1 {
    TrainingOpDescriptorV1 {
        id,
        category: TrainingOpCategoryV1::Optimizer,
        forward: false,
        vjp: TrainingVjpV1::None,
        mutates: true,
        checkpoint_planes,
    }
}

const fn lifecycle(id: &'static str, mutates: bool) -> TrainingOpDescriptorV1 {
    TrainingOpDescriptorV1 {
        id,
        category: TrainingOpCategoryV1::Lifecycle,
        forward: false,
        vjp: TrainingVjpV1::None,
        mutates,
        checkpoint_planes: NO_PLANES,
    }
}

static OPERATIONS: &[TrainingOpDescriptorV1] = &[
    graph("graph.ste_surrogate"),
    graph("graph.salt_ste"),
    graph("graph.lsq_ste"),
    graph("graph.fsq"),
    graph("graph.dense_matmul"),
    graph("graph.ternary_matmul"),
    graph("graph.transpose"),
    graph("graph.embedding_gather"),
    graph("graph.slice_cols"),
    graph("graph.concat_cols"),
    graph("graph.detach"),
    graph("graph.scale_const"),
    graph("graph.bias"),
    graph("graph.add"),
    graph("graph.mul"),
    graph("graph.conv1d"),
    graph("graph.conv2d"),
    graph("graph.relu2"),
    graph("graph.silu"),
    graph("graph.rmsnorm"),
    graph("graph.softmax"),
    graph("graph.causal_mask"),
    graph("graph.rope"),
    graph("graph.attention"),
    loss("loss.mse"),
    loss("loss.softmax_cross_entropy"),
    optimizer("optimizer.sgd", PARAMETER),
    optimizer("optimizer.adamw", ADAM_PLANES),
    optimizer("optimizer.cautious_adamw", ADAM_PLANES),
    optimizer("optimizer.int8_adamw", INT8_ADAM_PLANES),
    optimizer("optimizer.muon", MUON_PLANES),
    lifecycle("lifecycle.checkpoint", false),
    lifecycle("lifecycle.resume", true),
    lifecycle("lifecycle.export", false),
    lifecycle("lifecycle.reload", true),
];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWire {
    schema_id: String,
    schema_version: u32,
    dtype: String,
    operations: Vec<DescriptorWire>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DescriptorWire {
    id: String,
    category: TrainingOpCategoryV1,
    forward: bool,
    vjp: TrainingVjpV1,
    mutates: bool,
    checkpoint_planes: Vec<String>,
}

/// Strict `TrainingOpManifestV1` parse failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrainingManifestError {
    /// Input is not valid strict manifest JSON.
    InvalidJson(String),
    /// Schema identifier is not the frozen v1 identifier.
    UnsupportedSchemaId(String),
    /// Schema version is not one.
    UnsupportedSchemaVersion(u32),
    /// Dtype is not mandatory v1 f32.
    UnsupportedDtype(String),
    /// Operation count differs from the frozen registry.
    OperationCount {
        /// Required descriptor count.
        expected: usize,
        /// Parsed descriptor count.
        got: usize,
    },
    /// Descriptor at a canonical index differs from the frozen registry.
    OperationMismatch {
        /// Canonical descriptor index.
        index: usize,
        /// Parsed descriptor ID at that index.
        id: String,
    },
}

impl fmt::Display for TrainingManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(message) => write!(f, "invalid training manifest JSON: {message}"),
            Self::UnsupportedSchemaId(id) => {
                write!(f, "unsupported training manifest schema_id {id:?}")
            }
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported training manifest schema_version {version}")
            }
            Self::UnsupportedDtype(dtype) => {
                write!(f, "unsupported training manifest dtype {dtype:?}")
            }
            Self::OperationCount { expected, got } => {
                write!(
                    f,
                    "training manifest operation count {got}, expected {expected}"
                )
            }
            Self::OperationMismatch { index, id } => write!(
                f,
                "training manifest operation {index} ({id:?}) differs from frozen v1 descriptor"
            ),
        }
    }
}

impl std::error::Error for TrainingManifestError {}

/// Frozen language-neutral v1 portable-training registry.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TrainingOpManifestV1;

impl TrainingOpManifestV1 {
    /// Permanent schema identifier.
    pub const SCHEMA_ID: &'static str = SCHEMA_ID;
    /// Frozen schema version.
    pub const SCHEMA_VERSION: u32 = SCHEMA_VERSION;

    /// Canonically ordered operation registry.
    #[must_use]
    pub const fn operations() -> &'static [TrainingOpDescriptorV1] {
        OPERATIONS
    }

    /// Exact canonical JSON bytes, including one terminal newline.
    #[must_use]
    pub const fn canonical_json() -> &'static [u8] {
        CANONICAL_JSON
    }

    /// BLAKE3 digest of exact canonical JSON bytes.
    #[must_use]
    pub fn digest() -> [u8; 32] {
        *blake3::hash(CANONICAL_JSON).as_bytes()
    }

    /// Parse and validate a manifest against the complete frozen v1 registry.
    ///
    /// Formatting may differ, but fields, descriptor order and values must be
    /// exact. Call [`Self::canonical_json`] to re-emit canonical bytes.
    ///
    /// # Errors
    /// Returns [`TrainingManifestError`] for malformed JSON or any schema,
    /// dtype, count, order, identifier, capability or state-plane mismatch.
    pub fn parse_json(bytes: &[u8]) -> Result<Self, TrainingManifestError> {
        let wire: ManifestWire = serde_json::from_slice(bytes)
            .map_err(|error| TrainingManifestError::InvalidJson(error.to_string()))?;
        if wire.schema_id != SCHEMA_ID {
            return Err(TrainingManifestError::UnsupportedSchemaId(wire.schema_id));
        }
        if wire.schema_version != SCHEMA_VERSION {
            return Err(TrainingManifestError::UnsupportedSchemaVersion(
                wire.schema_version,
            ));
        }
        if wire.dtype != DTYPE {
            return Err(TrainingManifestError::UnsupportedDtype(wire.dtype));
        }
        if wire.operations.len() != OPERATIONS.len() {
            return Err(TrainingManifestError::OperationCount {
                expected: OPERATIONS.len(),
                got: wire.operations.len(),
            });
        }
        for (index, (got, expected)) in wire.operations.iter().zip(OPERATIONS).enumerate() {
            let matches = got.id == expected.id
                && got.category == expected.category
                && got.forward == expected.forward
                && got.vjp == expected.vjp
                && got.mutates == expected.mutates
                && got
                    .checkpoint_planes
                    .iter()
                    .map(String::as_str)
                    .eq(expected.checkpoint_planes.iter().copied());
            if !matches {
                return Err(TrainingManifestError::OperationMismatch {
                    index,
                    id: got.id.clone(),
                });
            }
        }
        Ok(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_manifest_roundtrips_and_registry_is_unique() {
        assert_eq!(
            TrainingOpManifestV1::parse_json(CANONICAL_JSON),
            Ok(TrainingOpManifestV1)
        );
        assert_eq!(CANONICAL_JSON.last(), Some(&b'\n'));
        for (index, operation) in OPERATIONS.iter().enumerate() {
            assert!(!operation.id.is_empty());
            assert!(operation.id.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_')
            }));
            assert!(!OPERATIONS[..index]
                .iter()
                .any(|prior| prior.id == operation.id));
        }
    }

    #[test]
    fn strict_parser_rejects_schema_fields_and_registry_drift() {
        let source = core::str::from_utf8(CANONICAL_JSON).unwrap();
        let cases = [
            source.replacen("\"schema_version\": 1", "\"schema_version\": 2", 1),
            source.replacen("\"dtype\": \"f32\"", "\"dtype\": \"f16\"", 1),
            source.replacen("\"graph.ste_surrogate\"", "\"graph.unknown\"", 1),
            source.replacen("\"forward\":true", "\"forward\":false", 1),
            source.replacen(
                "\"schema_id\": \"tritium.training_op_manifest\"",
                "\"schema_id\": \"tritium.training_op_manifest\",\n  \"extra\": true",
                1,
            ),
        ];
        for case in cases {
            assert!(TrainingOpManifestV1::parse_json(case.as_bytes()).is_err());
        }
    }

    #[test]
    fn strict_parser_rejects_missing_duplicate_and_reordered_operations() {
        let mut value: serde_json::Value = serde_json::from_slice(CANONICAL_JSON).unwrap();
        let operations = value["operations"].as_array_mut().unwrap();
        operations.pop();
        assert!(matches!(
            TrainingOpManifestV1::parse_json(serde_json::to_string(&value).unwrap().as_bytes()),
            Err(TrainingManifestError::OperationCount { .. })
        ));

        let mut value: serde_json::Value = serde_json::from_slice(CANONICAL_JSON).unwrap();
        let operations = value["operations"].as_array_mut().unwrap();
        operations[1] = operations[0].clone();
        assert!(matches!(
            TrainingOpManifestV1::parse_json(serde_json::to_string(&value).unwrap().as_bytes()),
            Err(TrainingManifestError::OperationMismatch { index: 1, .. })
        ));

        let mut value: serde_json::Value = serde_json::from_slice(CANONICAL_JSON).unwrap();
        value["operations"].as_array_mut().unwrap().swap(0, 1);
        assert!(matches!(
            TrainingOpManifestV1::parse_json(serde_json::to_string(&value).unwrap().as_bytes()),
            Err(TrainingManifestError::OperationMismatch { index: 0, .. })
        ));
    }
}
