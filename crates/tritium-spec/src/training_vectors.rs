//! Strict language-neutral semantic vectors for portable training backends.

use core::fmt;
use std::collections::HashSet;

use serde::Deserialize;

use crate::{
    TrainExecutionV1, TrainingOpCategoryV1, TrainingOpDescriptorV1, TrainingOpManifestV1,
    TrainingOpManifestV2, TrainingOpManifestV3, TrainingVjpV1,
};

const SCHEMA_ID: &str = "tritium.training_vectors";
const CANONICAL_JSON_V1: &[u8] = include_bytes!("../data/training/v1/vectors/v1.json");
const CANONICAL_JSON_V2: &[u8] = include_bytes!("../data/training/v2/vectors/v2.json");
const CANONICAL_JSON_V3: &[u8] = include_bytes!("../data/training/v3/vectors/v3.json");

/// Exact or bounded comparison policy for one semantic vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingToleranceV1 {
    /// Outputs and state must match f32 bit patterns exactly.
    BitExact,
    /// Outputs may differ by fixed finite nonnegative absolute/relative bounds.
    AbsoluteRelative {
        /// IEEE-754 bits of the absolute bound.
        absolute_bits: u32,
        /// IEEE-754 bits of the relative bound.
        relative_bits: u32,
    },
}

/// Exact owned payload encoded by one vector buffer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrainingVectorBufferDataV1 {
    /// IEEE-754 bit patterns for f32 elements.
    F32Bits(Vec<u32>),
    /// Unsigned 32-bit elements.
    U32(Vec<u32>),
    /// Opaque canonical bytes.
    Bytes(Vec<u8>),
}

impl TrainingVectorBufferDataV1 {
    fn len(&self) -> usize {
        match self {
            Self::F32Bits(values) | Self::U32(values) => values.len(),
            Self::Bytes(values) => values.len(),
        }
    }
}

/// One named owned input, output, state plane or artifact payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingVectorBufferV1 {
    /// Stable operation-local role name.
    pub name: String,
    /// Row-major dimensions; empty means scalar.
    pub shape: Vec<u64>,
    /// Exact typed payload.
    pub data: TrainingVectorBufferDataV1,
}

/// Exact owned attribute value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrainingVectorAttributeValueV1 {
    /// IEEE-754 bits of a finite f32 scalar.
    F32Bits(u32),
    /// Unsigned integer scalar.
    U64(u64),
    /// Boolean flag.
    Bool(bool),
    /// UTF-8 identifier/text.
    Text(String),
    /// Unsigned integer list.
    U64List(Vec<u64>),
    /// Unsigned index list.
    U32List(Vec<u32>),
}

/// One named exact operation attribute.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingVectorAttributeV1 {
    /// Stable operation-local attribute name.
    pub name: String,
    /// Exact typed value.
    pub value: TrainingVectorAttributeValueV1,
}

/// Stable category for an expected backend failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingVectorErrorCategoryV1 {
    /// Shared request validation rejects the case.
    InvalidRequest,
    /// Operation-specific validation rejects the case.
    InvalidOperation,
    /// Backend/device execution rejects the case.
    Backend,
}

/// Expected result of one semantic vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrainingVectorExpectedV1 {
    /// Successful execution with exact reference outputs and a scratch ceiling.
    Success {
        /// Exact output/state/artifact payloads.
        outputs: Vec<TrainingVectorBufferV1>,
        /// Maximum temporary tensor bytes admitted for this case.
        scratch_bytes_max: u64,
    },
    /// Structured failure with a stable category and code.
    Error {
        /// Error layer.
        category: TrainingVectorErrorCategoryV1,
        /// Portable lowercase error code.
        code: String,
        /// Initial output payloads that must remain bit-exact on failure.
        outputs: Vec<TrainingVectorBufferV1>,
    },
}

/// One self-contained forward, VJP, update or lifecycle case.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingVectorCaseV1 {
    /// Permanent case identifier.
    pub case_id: String,
    /// Permanent manifest operation identifier.
    pub operation: String,
    /// Requested semantic phase.
    pub execution: TrainExecutionV1,
    /// Comparison policy for a successful result.
    pub tolerance: TrainingToleranceV1,
    /// Exact named inputs and pre-mutation state.
    pub inputs: Vec<TrainingVectorBufferV1>,
    /// Exact named typed attributes.
    pub attributes: Vec<TrainingVectorAttributeV1>,
    /// Expected success payload or structured error.
    pub expected: TrainingVectorExpectedV1,
}

/// Parsed v1 semantic-vector corpus bound to one exact training manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingVectorSetV1 {
    manifest_digest: [u8; 32],
    source_digest: [u8; 32],
    cases: Vec<TrainingVectorCaseV1>,
}

impl TrainingVectorSetV1 {
    /// Permanent schema identifier.
    pub const SCHEMA_ID: &'static str = SCHEMA_ID;
    /// Frozen schema version.
    pub const SCHEMA_VERSION: u32 = 1;

    /// Exact canonical tracer-corpus bytes, including one terminal newline.
    ///
    /// Its digest identifies exact corpus used by a backend receipt.
    #[must_use]
    pub const fn canonical_json() -> &'static [u8] {
        CANONICAL_JSON_V1
    }

    /// BLAKE3 digest of exact canonical corpus bytes.
    #[must_use]
    pub fn digest() -> [u8; 32] {
        *blake3::hash(CANONICAL_JSON_V1).as_bytes()
    }

    /// Parse and validate a vector corpus.
    ///
    /// # Errors
    /// Returns [`TrainingVectorError`] for malformed JSON, schema/manifest
    /// drift, duplicate identifiers, illegal phases or invalid payloads.
    pub fn parse_json(bytes: &[u8]) -> Result<Self, TrainingVectorError> {
        let parsed = parse_vector_set(
            bytes,
            Self::SCHEMA_VERSION,
            TrainingOpManifestV1::digest(),
            TrainingOpManifestV1::operations(),
        )?;
        Ok(Self {
            manifest_digest: parsed.manifest_digest,
            source_digest: parsed.source_digest,
            cases: parsed.cases,
        })
    }

    /// Exact manifest digest declared by this corpus.
    #[must_use]
    pub const fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }

    /// BLAKE3 digest of exact source bytes accepted by parser.
    #[must_use]
    pub const fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }

    /// Canonically ordered cases as parsed from corpus.
    #[must_use]
    pub fn cases(&self) -> &[TrainingVectorCaseV1] {
        &self.cases
    }
}

/// Parsed v2 semantic-vector corpus bound to exact v2 manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingVectorSetV2 {
    manifest_digest: [u8; 32],
    source_digest: [u8; 32],
    cases: Vec<TrainingVectorCaseV1>,
}

impl TrainingVectorSetV2 {
    /// Permanent schema identifier.
    pub const SCHEMA_ID: &'static str = SCHEMA_ID;
    /// Frozen schema version.
    pub const SCHEMA_VERSION: u32 = 2;

    /// Exact canonical corpus bytes, including one terminal newline.
    #[must_use]
    pub const fn canonical_json() -> &'static [u8] {
        CANONICAL_JSON_V2
    }

    /// BLAKE3 digest of exact canonical corpus bytes.
    #[must_use]
    pub fn digest() -> [u8; 32] {
        *blake3::hash(CANONICAL_JSON_V2).as_bytes()
    }

    /// Parse and validate a v2 vector corpus.
    ///
    /// # Errors
    /// Returns [`TrainingVectorError`] for malformed JSON, schema/manifest
    /// drift, duplicate identifiers, illegal phases or invalid payloads.
    pub fn parse_json(bytes: &[u8]) -> Result<Self, TrainingVectorError> {
        let parsed = parse_vector_set(
            bytes,
            Self::SCHEMA_VERSION,
            TrainingOpManifestV2::digest(),
            TrainingOpManifestV2::operations(),
        )?;
        Ok(Self {
            manifest_digest: parsed.manifest_digest,
            source_digest: parsed.source_digest,
            cases: parsed.cases,
        })
    }

    /// Exact manifest digest declared by this corpus.
    #[must_use]
    pub const fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }

    /// BLAKE3 digest of exact source bytes accepted by parser.
    #[must_use]
    pub const fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }

    /// Canonically ordered cases as parsed from corpus.
    #[must_use]
    pub fn cases(&self) -> &[TrainingVectorCaseV1] {
        &self.cases
    }
}

/// Parsed v3 semantic-vector corpus bound to exact v3 manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingVectorSetV3 {
    manifest_digest: [u8; 32],
    source_digest: [u8; 32],
    cases: Vec<TrainingVectorCaseV1>,
}

impl TrainingVectorSetV3 {
    /// Permanent schema identifier.
    pub const SCHEMA_ID: &'static str = SCHEMA_ID;
    /// Frozen schema version.
    pub const SCHEMA_VERSION: u32 = 3;

    /// Exact canonical corpus bytes, including one terminal newline.
    #[must_use]
    pub const fn canonical_json() -> &'static [u8] {
        CANONICAL_JSON_V3
    }

    /// BLAKE3 digest of exact canonical corpus bytes.
    #[must_use]
    pub fn digest() -> [u8; 32] {
        *blake3::hash(CANONICAL_JSON_V3).as_bytes()
    }

    /// Parse and validate a v3 vector corpus.
    ///
    /// # Errors
    /// Returns [`TrainingVectorError`] for malformed JSON, schema/manifest
    /// drift, duplicate identifiers, illegal phases or invalid payloads.
    pub fn parse_json(bytes: &[u8]) -> Result<Self, TrainingVectorError> {
        let parsed = parse_vector_set(
            bytes,
            Self::SCHEMA_VERSION,
            TrainingOpManifestV3::digest(),
            TrainingOpManifestV3::operations(),
        )?;
        Ok(Self {
            manifest_digest: parsed.manifest_digest,
            source_digest: parsed.source_digest,
            cases: parsed.cases,
        })
    }

    /// Exact manifest digest declared by this corpus.
    #[must_use]
    pub const fn manifest_digest(&self) -> [u8; 32] {
        self.manifest_digest
    }

    /// BLAKE3 digest of exact source bytes accepted by parser.
    #[must_use]
    pub const fn source_digest(&self) -> [u8; 32] {
        self.source_digest
    }

    /// Canonically ordered cases as parsed from corpus.
    #[must_use]
    pub fn cases(&self) -> &[TrainingVectorCaseV1] {
        &self.cases
    }
}

struct ParsedVectorSet {
    manifest_digest: [u8; 32],
    source_digest: [u8; 32],
    cases: Vec<TrainingVectorCaseV1>,
}

fn parse_vector_set(
    bytes: &[u8],
    schema_version: u32,
    expected_manifest_digest: [u8; 32],
    operations: &[TrainingOpDescriptorV1],
) -> Result<ParsedVectorSet, TrainingVectorError> {
    let wire: VectorSetWire = serde_json::from_slice(bytes)
        .map_err(|error| TrainingVectorError::InvalidJson(error.to_string()))?;
    if wire.schema_id != SCHEMA_ID {
        return Err(TrainingVectorError::UnsupportedSchemaId(wire.schema_id));
    }
    if wire.schema_version != schema_version {
        return Err(TrainingVectorError::UnsupportedSchemaVersion(
            wire.schema_version,
        ));
    }
    let manifest_digest = parse_digest(&wire.manifest_digest)?;
    if manifest_digest != expected_manifest_digest {
        return Err(TrainingVectorError::ManifestDigestMismatch);
    }
    if wire.cases.is_empty() {
        return Err(TrainingVectorError::EmptyCases);
    }

    let mut case_ids = HashSet::with_capacity(wire.cases.len());
    let mut cases = Vec::with_capacity(wire.cases.len());
    for case in wire.cases {
        validate_identifier("case", &case.case_id, true)?;
        if !case_ids.insert(case.case_id.clone()) {
            return Err(TrainingVectorError::DuplicateCaseId(case.case_id));
        }
        let descriptor = operations
            .iter()
            .find(|descriptor| descriptor.id == case.operation)
            .ok_or_else(|| TrainingVectorError::UnknownOperation(case.operation.clone()))?;
        let execution = case.execution.into();
        if !execution_allowed(
            descriptor.id,
            descriptor.category,
            descriptor.forward,
            descriptor.vjp,
            execution,
        ) {
            return Err(TrainingVectorError::IllegalExecution {
                operation: case.operation,
                execution,
            });
        }
        let tolerance = validate_tolerance(case.tolerance)?;
        let carries_invalid_request = matches!(
            &case.expected,
            ExpectedWire::Error {
                category: ErrorCategoryWire::InvalidRequest,
                ..
            }
        );
        let inputs = validate_buffers("input", case.inputs, carries_invalid_request)?;
        let attributes = validate_attributes(case.attributes, carries_invalid_request)?;
        let expected = validate_expected(case.expected)?;
        cases.push(TrainingVectorCaseV1 {
            case_id: case.case_id,
            operation: descriptor.id.to_owned(),
            execution,
            tolerance,
            inputs,
            attributes,
            expected,
        });
    }
    Ok(ParsedVectorSet {
        manifest_digest,
        source_digest: *blake3::hash(bytes).as_bytes(),
        cases,
    })
}

/// Strict semantic-vector parse or validation failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrainingVectorError {
    /// Input is not valid strict vector JSON.
    InvalidJson(String),
    /// Schema identifier is unsupported.
    UnsupportedSchemaId(String),
    /// Schema version is unsupported by requested reader.
    UnsupportedSchemaVersion(u32),
    /// Manifest digest is not 64 lowercase hexadecimal characters.
    InvalidManifestDigest(String),
    /// Declared manifest digest differs from this build's exact manifest.
    ManifestDigestMismatch,
    /// Corpus contains no cases.
    EmptyCases,
    /// Case, role, attribute or error code is not portable lowercase ASCII.
    InvalidIdentifier {
        /// Identifier namespace.
        namespace: &'static str,
        /// Rejected identifier.
        value: String,
    },
    /// A case identifier appears twice.
    DuplicateCaseId(String),
    /// An operation ID is absent from the frozen manifest.
    UnknownOperation(String),
    /// Operation does not implement the requested phase.
    IllegalExecution {
        /// Permanent operation ID.
        operation: String,
        /// Rejected semantic phase.
        execution: TrainExecutionV1,
    },
    /// A buffer or attribute role appears twice in one namespace.
    DuplicateName {
        /// Input, output or attribute namespace.
        namespace: &'static str,
        /// Duplicated role.
        name: String,
    },
    /// Shape product does not fit host `usize`.
    ShapeOverflow(String),
    /// Shape element count differs from encoded payload length.
    BufferLength {
        /// Buffer role.
        name: String,
        /// Shape-derived count.
        expected: usize,
        /// Encoded payload count.
        got: usize,
    },
    /// Absolute/relative tolerance is negative or non-finite.
    InvalidTolerance,
    /// F32 attribute intended for execution is non-finite.
    NonFiniteAttribute(String),
}

impl fmt::Display for TrainingVectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(message) => write!(f, "invalid training vector JSON: {message}"),
            Self::UnsupportedSchemaId(id) => write!(f, "unsupported vector schema_id {id:?}"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(f, "unsupported vector schema_version {version}")
            }
            Self::InvalidManifestDigest(digest) => {
                write!(f, "invalid manifest digest {digest:?}")
            }
            Self::ManifestDigestMismatch => write!(f, "vector manifest digest mismatch"),
            Self::EmptyCases => write!(f, "training vector corpus has no cases"),
            Self::InvalidIdentifier { namespace, value } => {
                write!(f, "invalid {namespace} identifier {value:?}")
            }
            Self::DuplicateCaseId(case_id) => write!(f, "duplicate case ID {case_id:?}"),
            Self::UnknownOperation(operation) => write!(f, "unknown operation {operation:?}"),
            Self::IllegalExecution {
                operation,
                execution,
            } => write!(f, "operation {operation:?} does not support {execution:?}"),
            Self::DuplicateName { namespace, name } => {
                write!(f, "duplicate {namespace} name {name:?}")
            }
            Self::ShapeOverflow(name) => write!(f, "shape product overflows for {name:?}"),
            Self::BufferLength {
                name,
                expected,
                got,
            } => write!(
                f,
                "buffer {name:?} has {got} encoded elements, shape requires {expected}"
            ),
            Self::InvalidTolerance => write!(f, "training vector tolerance is invalid"),
            Self::NonFiniteAttribute(name) => {
                write!(f, "attribute {name:?} must encode a finite f32")
            }
        }
    }
}

impl std::error::Error for TrainingVectorError {}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VectorSetWire {
    schema_id: String,
    schema_version: u32,
    manifest_digest: String,
    cases: Vec<VectorCaseWire>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct VectorCaseWire {
    case_id: String,
    operation: String,
    execution: ExecutionWire,
    tolerance: ToleranceWire,
    inputs: Vec<BufferWire>,
    attributes: Vec<AttributeWire>,
    expected: ExpectedWire,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ExecutionWire {
    Forward,
    Vjp,
    Step,
    Checkpoint,
    Resume,
    Export,
    Reload,
}

impl From<ExecutionWire> for TrainExecutionV1 {
    fn from(execution: ExecutionWire) -> Self {
        match execution {
            ExecutionWire::Forward => Self::Forward,
            ExecutionWire::Vjp => Self::Vjp,
            ExecutionWire::Step => Self::Step,
            ExecutionWire::Checkpoint => Self::Checkpoint,
            ExecutionWire::Resume => Self::Resume,
            ExecutionWire::Export => Self::Export,
            ExecutionWire::Reload => Self::Reload,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ToleranceWire {
    BitExact,
    AbsoluteRelative {
        absolute_bits: u32,
        relative_bits: u32,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BufferWire {
    name: String,
    shape: Vec<u64>,
    data: BufferDataWire,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "dtype", rename_all = "snake_case", deny_unknown_fields)]
enum BufferDataWire {
    F32 { bits: Vec<u32> },
    U32 { values: Vec<u32> },
    Bytes { values: Vec<u8> },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum AttributeWire {
    F32 { name: String, bits: u32 },
    U64 { name: String, value: u64 },
    Bool { name: String, value: bool },
    Text { name: String, value: String },
    U64List { name: String, values: Vec<u64> },
    U32List { name: String, values: Vec<u32> },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ExpectedWire {
    Success {
        outputs: Vec<BufferWire>,
        scratch_bytes_max: u64,
    },
    Error {
        category: ErrorCategoryWire,
        code: String,
        outputs: Vec<BufferWire>,
    },
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ErrorCategoryWire {
    InvalidRequest,
    InvalidOperation,
    Backend,
}

fn parse_digest(hex: &str) -> Result<[u8; 32], TrainingVectorError> {
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(TrainingVectorError::InvalidManifestDigest(hex.to_owned()));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0])
            .ok_or_else(|| TrainingVectorError::InvalidManifestDigest(hex.to_owned()))?;
        let low = hex_nibble(pair[1])
            .ok_or_else(|| TrainingVectorError::InvalidManifestDigest(hex.to_owned()))?;
        digest[index] = (high << 4) | low;
    }
    Ok(digest)
}

const fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn validate_tolerance(wire: ToleranceWire) -> Result<TrainingToleranceV1, TrainingVectorError> {
    match wire {
        ToleranceWire::BitExact => Ok(TrainingToleranceV1::BitExact),
        ToleranceWire::AbsoluteRelative {
            absolute_bits,
            relative_bits,
        } => {
            let absolute = f32::from_bits(absolute_bits);
            let relative = f32::from_bits(relative_bits);
            if !absolute.is_finite() || absolute < 0.0 || !relative.is_finite() || relative < 0.0 {
                return Err(TrainingVectorError::InvalidTolerance);
            }
            Ok(TrainingToleranceV1::AbsoluteRelative {
                absolute_bits,
                relative_bits,
            })
        }
    }
}

fn validate_buffers(
    namespace: &'static str,
    buffers: Vec<BufferWire>,
    allow_duplicate_names: bool,
) -> Result<Vec<TrainingVectorBufferV1>, TrainingVectorError> {
    let mut names = HashSet::with_capacity(buffers.len());
    let mut parsed = Vec::with_capacity(buffers.len());
    for buffer in buffers {
        validate_identifier(namespace, &buffer.name, false)?;
        if !names.insert(buffer.name.clone()) && !allow_duplicate_names {
            return Err(TrainingVectorError::DuplicateName {
                namespace,
                name: buffer.name,
            });
        }
        let expected = buffer.shape.iter().try_fold(1_u64, |count, &dimension| {
            count
                .checked_mul(dimension)
                .ok_or_else(|| TrainingVectorError::ShapeOverflow(buffer.name.clone()))
        })?;
        let expected = usize::try_from(expected)
            .map_err(|_| TrainingVectorError::ShapeOverflow(buffer.name.clone()))?;
        let data = match buffer.data {
            BufferDataWire::F32 { bits } => TrainingVectorBufferDataV1::F32Bits(bits),
            BufferDataWire::U32 { values } => TrainingVectorBufferDataV1::U32(values),
            BufferDataWire::Bytes { values } => TrainingVectorBufferDataV1::Bytes(values),
        };
        if data.len() != expected {
            return Err(TrainingVectorError::BufferLength {
                name: buffer.name,
                expected,
                got: data.len(),
            });
        }
        parsed.push(TrainingVectorBufferV1 {
            name: buffer.name,
            shape: buffer.shape,
            data,
        });
    }
    Ok(parsed)
}

fn validate_attributes(
    attributes: Vec<AttributeWire>,
    allow_duplicate_names: bool,
) -> Result<Vec<TrainingVectorAttributeV1>, TrainingVectorError> {
    let mut names = HashSet::with_capacity(attributes.len());
    let mut parsed = Vec::with_capacity(attributes.len());
    for attribute in attributes {
        let (name, value) = match attribute {
            AttributeWire::F32 { name, bits } => {
                if !f32::from_bits(bits).is_finite() {
                    return Err(TrainingVectorError::NonFiniteAttribute(name));
                }
                (name, TrainingVectorAttributeValueV1::F32Bits(bits))
            }
            AttributeWire::U64 { name, value } => {
                (name, TrainingVectorAttributeValueV1::U64(value))
            }
            AttributeWire::Bool { name, value } => {
                (name, TrainingVectorAttributeValueV1::Bool(value))
            }
            AttributeWire::Text { name, value } => {
                (name, TrainingVectorAttributeValueV1::Text(value))
            }
            AttributeWire::U64List { name, values } => {
                (name, TrainingVectorAttributeValueV1::U64List(values))
            }
            AttributeWire::U32List { name, values } => {
                (name, TrainingVectorAttributeValueV1::U32List(values))
            }
        };
        validate_identifier("attribute", &name, false)?;
        if !names.insert(name.clone()) && !allow_duplicate_names {
            return Err(TrainingVectorError::DuplicateName {
                namespace: "attribute",
                name,
            });
        }
        parsed.push(TrainingVectorAttributeV1 { name, value });
    }
    Ok(parsed)
}

fn validate_expected(wire: ExpectedWire) -> Result<TrainingVectorExpectedV1, TrainingVectorError> {
    match wire {
        ExpectedWire::Success {
            outputs,
            scratch_bytes_max,
        } => Ok(TrainingVectorExpectedV1::Success {
            outputs: validate_buffers("output", outputs, false)?,
            scratch_bytes_max,
        }),
        ExpectedWire::Error {
            category,
            code,
            outputs,
        } => {
            validate_identifier("error", &code, false)?;
            let category = match category {
                ErrorCategoryWire::InvalidRequest => TrainingVectorErrorCategoryV1::InvalidRequest,
                ErrorCategoryWire::InvalidOperation => {
                    TrainingVectorErrorCategoryV1::InvalidOperation
                }
                ErrorCategoryWire::Backend => TrainingVectorErrorCategoryV1::Backend,
            };
            Ok(TrainingVectorExpectedV1::Error {
                category,
                code,
                outputs: validate_buffers("output", outputs, false)?,
            })
        }
    }
}

fn validate_identifier(
    namespace: &'static str,
    value: &str,
    allow_hyphen: bool,
) -> Result<(), TrainingVectorError> {
    let valid = !value.is_empty()
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'_')
                || (allow_hyphen && byte == b'-')
        });
    if valid {
        Ok(())
    } else {
        Err(TrainingVectorError::InvalidIdentifier {
            namespace,
            value: value.to_owned(),
        })
    }
}

fn execution_allowed(
    id: &str,
    category: TrainingOpCategoryV1,
    forward: bool,
    vjp: TrainingVjpV1,
    execution: TrainExecutionV1,
) -> bool {
    match execution {
        TrainExecutionV1::Forward => forward,
        TrainExecutionV1::Vjp => vjp == TrainingVjpV1::FirstOrder,
        TrainExecutionV1::Step => category == TrainingOpCategoryV1::Optimizer,
        TrainExecutionV1::Checkpoint => id == "lifecycle.checkpoint",
        TrainExecutionV1::Resume => id == "lifecycle.resume",
        TrainExecutionV1::Export => id == "lifecycle.export",
        TrainExecutionV1::Reload => id == "lifecycle.reload",
    }
}
