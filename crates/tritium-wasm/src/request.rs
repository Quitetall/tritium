//! Strict JSON bridge for browser-owned portable-training buffers.

use serde::{Deserialize, Serialize};
use tritium_spec::{
    TrainAttributeV1, TrainAttributeValueV1, TrainBackendError, TrainBackendV1,
    TrainBufferDataMutV1, TrainBufferDataRefV1, TrainDTypeV1, TrainExecutionV1,
    TrainNamedBufferMutV1, TrainNamedBufferRefV1, TrainOperationErrorV1, TrainOutputV1,
    TrainReceiptV1, TrainRequestError, TrainRequestV1,
};

use crate::WasmTrainBackendV1;

const REQUEST_SCHEMA_ID: &str = "tritium.portable_training_request";
const RESPONSE_SCHEMA_ID: &str = "tritium.portable_training_response";
const SCHEMA_VERSION: u32 = 1;
const MAX_REQUEST_JSON_BYTES: usize = 8 * 1024 * 1024;
const MAX_CALLER_BYTES: usize = 64 * 1024 * 1024;
const MAX_COLLECTION_ITEMS: usize = 64;
const MAX_NAME_BYTES: usize = 128;
const MAX_TEXT_BYTES: usize = 4096;
const MAX_LIST_ITEMS: usize = 1024;
const MAX_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RequestWire {
    schema_id: String,
    schema_version: u32,
    physical_device: String,
    operation: String,
    execution: ExecutionWire,
    #[serde(default)]
    vector_digest: RequiredVectorDigest,
    inputs: Vec<BufferWire>,
    attributes: Vec<AttributeWire>,
    outputs: Vec<BufferWire>,
}

#[derive(Default)]
enum RequiredVectorDigest {
    #[default]
    Missing,
    Present(Option<String>),
}

impl<'de> Deserialize<'de> for RequiredVectorDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Option::<String>::deserialize(deserializer).map(Self::Present)
    }
}

#[derive(Clone, Copy, Deserialize)]
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
    fn from(value: ExecutionWire) -> Self {
        match value {
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

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BufferWire {
    name: String,
    shape: Vec<u64>,
    data: BufferDataWire,
}

#[derive(Deserialize)]
#[serde(tag = "dtype", rename_all = "snake_case", deny_unknown_fields)]
enum BufferDataWire {
    F32 { bits: Vec<u32> },
    U32 { values: Vec<u32> },
    Bytes { values: Vec<u8> },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
enum AttributeWire {
    F32 { name: String, bits: u32 },
    U64 { name: String, value: u64 },
    Bool { name: String, value: bool },
    Text { name: String, value: String },
    U64List { name: String, values: Vec<u64> },
    U32List { name: String, values: Vec<u32> },
}

enum OwnedAttribute {
    F32(String, f32),
    U64(String, u64),
    Bool(String, bool),
    Text(String, String),
    U64List(String, Vec<u64>),
    U32List(String, Vec<u32>),
}

impl OwnedAttribute {
    fn as_ref(&self) -> TrainAttributeV1<'_> {
        match self {
            Self::F32(name, value) => {
                TrainAttributeV1::new(name, TrainAttributeValueV1::F32(*value))
            }
            Self::U64(name, value) => {
                TrainAttributeV1::new(name, TrainAttributeValueV1::U64(*value))
            }
            Self::Bool(name, value) => {
                TrainAttributeV1::new(name, TrainAttributeValueV1::Bool(*value))
            }
            Self::Text(name, value) => {
                TrainAttributeV1::new(name, TrainAttributeValueV1::Text(value))
            }
            Self::U64List(name, values) => {
                TrainAttributeV1::new(name, TrainAttributeValueV1::U64List(values))
            }
            Self::U32List(name, values) => {
                TrainAttributeV1::new(name, TrainAttributeValueV1::U32List(values))
            }
        }
    }
}

#[derive(Clone)]
struct OwnedBuffer {
    name: String,
    shape: Vec<u64>,
    data: OwnedBufferData,
}

#[derive(Clone)]
enum OwnedBufferData {
    F32(Vec<f32>),
    U32(Vec<u32>),
    Bytes(Vec<u8>),
}

impl OwnedBuffer {
    fn as_ref(&self) -> TrainNamedBufferRefV1<'_> {
        let data = match &self.data {
            OwnedBufferData::F32(values) => TrainBufferDataRefV1::F32(values),
            OwnedBufferData::U32(values) => TrainBufferDataRefV1::U32(values),
            OwnedBufferData::Bytes(values) => TrainBufferDataRefV1::Bytes(values),
        };
        TrainNamedBufferRefV1::new(&self.name, &self.shape, data)
    }

    fn as_mut(&mut self) -> TrainNamedBufferMutV1<'_> {
        let data = match &mut self.data {
            OwnedBufferData::F32(values) => TrainBufferDataMutV1::F32(values),
            OwnedBufferData::U32(values) => TrainBufferDataMutV1::U32(values),
            OwnedBufferData::Bytes(values) => TrainBufferDataMutV1::Bytes(values),
        };
        TrainNamedBufferMutV1::new(&self.name, &self.shape, data)
    }
}

impl From<BufferWire> for OwnedBuffer {
    fn from(value: BufferWire) -> Self {
        let data = match value.data {
            BufferDataWire::F32 { bits } => {
                OwnedBufferData::F32(bits.into_iter().map(f32::from_bits).collect())
            }
            BufferDataWire::U32 { values } => OwnedBufferData::U32(values),
            BufferDataWire::Bytes { values } => OwnedBufferData::Bytes(values),
        };
        Self {
            name: value.name,
            shape: value.shape,
            data,
        }
    }
}

impl From<AttributeWire> for OwnedAttribute {
    fn from(value: AttributeWire) -> Self {
        match value {
            AttributeWire::F32 { name, bits } => Self::F32(name, f32::from_bits(bits)),
            AttributeWire::U64 { name, value } => Self::U64(name, value),
            AttributeWire::Bool { name, value } => Self::Bool(name, value),
            AttributeWire::Text { name, value } => Self::Text(name, value),
            AttributeWire::U64List { name, values } => Self::U64List(name, values),
            AttributeWire::U32List { name, values } => Self::U32List(name, values),
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct BufferResponse {
    name: String,
    shape: Vec<u64>,
    data: BufferDataResponse,
}

#[derive(Serialize)]
#[serde(tag = "dtype", rename_all = "snake_case")]
enum BufferDataResponse {
    F32 { bits: Vec<u32> },
    U32 { values: Vec<u32> },
    Bytes { values: Vec<u8> },
}

impl From<OwnedBuffer> for BufferResponse {
    fn from(value: OwnedBuffer) -> Self {
        let data = match value.data {
            OwnedBufferData::F32(values) => BufferDataResponse::F32 {
                bits: values.into_iter().map(f32::to_bits).collect(),
            },
            OwnedBufferData::U32(values) => BufferDataResponse::U32 { values },
            OwnedBufferData::Bytes(values) => BufferDataResponse::Bytes { values },
        };
        Self {
            name: value.name,
            shape: value.shape,
            data,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ReceiptResponse {
    backend_id: String,
    backend_build: String,
    physical_device: Option<String>,
    manifest_digest: String,
    vector_digest: Option<String>,
    operation: String,
    execution: &'static str,
    dtype: &'static str,
    max_rank: u32,
    max_elements: u64,
    max_bytes: u64,
    input_digest: String,
    output_digest: String,
    peak_resident_bytes: u64,
    scratch_bytes: u64,
    host_transfers: u64,
    device_resident: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ErrorResponse {
    category: &'static str,
    code: String,
    message: String,
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum ResponseWire {
    Ok {
        #[serde(rename = "schemaId")]
        schema_id: &'static str,
        #[serde(rename = "schemaVersion")]
        schema_version: u32,
        outputs: Vec<BufferResponse>,
        receipt: Box<ReceiptResponse>,
    },
    Error {
        #[serde(rename = "schemaId")]
        schema_id: &'static str,
        #[serde(rename = "schemaVersion")]
        schema_version: u32,
        outputs: Vec<BufferResponse>,
        error: ErrorResponse,
    },
}

fn digest_hex(bytes: [u8; 32]) -> String {
    let mut result = String::with_capacity(64);
    for byte in bytes {
        use core::fmt::Write;
        write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
    }
    result
}

fn parse_digest(value: Option<String>) -> Result<Option<[u8; 32]>, ErrorResponse> {
    let Some(value) = value else { return Ok(None) };
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(error(
            "invalid_request",
            "invalid_digest",
            "vectorDigest must be 64 hex characters",
        ));
    }
    let mut digest = [0_u8; 32];
    for (index, output) in digest.iter_mut().enumerate() {
        *output = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).map_err(|_| {
            error(
                "invalid_request",
                "invalid_digest",
                "vectorDigest is invalid",
            )
        })?;
    }
    Ok(Some(digest))
}

fn error(category: &'static str, code: &'static str, message: impl Into<String>) -> ErrorResponse {
    ErrorResponse {
        category,
        code: code.to_owned(),
        message: message.into(),
    }
}

fn capacity_error(code: &'static str, message: impl Into<String>) -> ErrorResponse {
    error("capacity", code, message)
}

fn validate_name(value: &str, field: &'static str) -> Result<(), ErrorResponse> {
    if value.is_empty() || value.len() > MAX_NAME_BYTES {
        return Err(error(
            "invalid_request",
            "invalid_name",
            format!("{field} must contain 1..={MAX_NAME_BYTES} UTF-8 bytes"),
        ));
    }
    Ok(())
}

fn buffer_bytes(value: &BufferDataWire) -> Result<usize, ErrorResponse> {
    match value {
        BufferDataWire::F32 { bits } => bits.len().checked_mul(4),
        BufferDataWire::U32 { values } => values.len().checked_mul(4),
        BufferDataWire::Bytes { values } => Some(values.len()),
    }
    .ok_or_else(|| capacity_error("caller_bytes", "caller buffer byte count overflowed"))
}

fn validate_buffer(value: &BufferWire, total_bytes: &mut usize) -> Result<(), ErrorResponse> {
    validate_name(&value.name, "buffer name")?;
    if value.shape.len() > 4 {
        return Err(capacity_error("rank", "buffer rank exceeds 4"));
    }
    if value
        .shape
        .iter()
        .any(|&dimension| dimension > MAX_SAFE_INTEGER)
    {
        return Err(error(
            "invalid_request",
            "unsafe_integer",
            "shape dimensions must be JavaScript safe integers",
        ));
    }
    *total_bytes = total_bytes
        .checked_add(buffer_bytes(&value.data)?)
        .ok_or_else(|| capacity_error("caller_bytes", "caller buffer byte count overflowed"))?;
    if *total_bytes > MAX_CALLER_BYTES {
        return Err(capacity_error(
            "caller_bytes",
            "caller buffers exceed 64 MiB",
        ));
    }
    Ok(())
}

fn validate_attribute(value: &AttributeWire) -> Result<(), ErrorResponse> {
    let name = match value {
        AttributeWire::F32 { name, .. }
        | AttributeWire::U64 { name, .. }
        | AttributeWire::Bool { name, .. }
        | AttributeWire::Text { name, .. }
        | AttributeWire::U64List { name, .. }
        | AttributeWire::U32List { name, .. } => name,
    };
    validate_name(name, "attribute name")?;
    match value {
        AttributeWire::U64 { value, .. } if *value > MAX_SAFE_INTEGER => Err(error(
            "invalid_request",
            "unsafe_integer",
            "u64 attributes must be JavaScript safe integers",
        )),
        AttributeWire::Text { value, .. } if value.len() > MAX_TEXT_BYTES => Err(capacity_error(
            "text_bytes",
            "text attribute exceeds 4096 UTF-8 bytes",
        )),
        AttributeWire::U64List { values, .. } => {
            if values.len() > MAX_LIST_ITEMS {
                Err(capacity_error(
                    "list_items",
                    "attribute list exceeds 1024 items",
                ))
            } else if values.iter().any(|&item| item > MAX_SAFE_INTEGER) {
                Err(error(
                    "invalid_request",
                    "unsafe_integer",
                    "u64 list values must be JavaScript safe integers",
                ))
            } else {
                Ok(())
            }
        }
        AttributeWire::U32List { values, .. } if values.len() > MAX_LIST_ITEMS => Err(
            capacity_error("list_items", "attribute list exceeds 1024 items"),
        ),
        _ => Ok(()),
    }
}

fn preflight_request(value: &RequestWire) -> Result<(), ErrorResponse> {
    if value.physical_device.is_empty() || value.physical_device.len() > MAX_TEXT_BYTES {
        return Err(error(
            "invalid_request",
            "physical_device",
            "physicalDevice must contain 1..=4096 UTF-8 bytes",
        ));
    }
    validate_name(&value.operation, "operation")?;
    for (name, count) in [
        ("inputs", value.inputs.len()),
        ("attributes", value.attributes.len()),
        ("outputs", value.outputs.len()),
    ] {
        if count > MAX_COLLECTION_ITEMS {
            return Err(capacity_error(
                "collection_items",
                format!("{name} exceeds {MAX_COLLECTION_ITEMS} items"),
            ));
        }
    }
    let mut total_bytes = 0_usize;
    for buffer in value.inputs.iter().chain(&value.outputs) {
        validate_buffer(buffer, &mut total_bytes)?;
    }
    for attribute in &value.attributes {
        validate_attribute(attribute)?;
    }
    Ok(())
}

fn request_error_code(error: &TrainRequestError) -> String {
    match error {
        TrainRequestError::UnknownOperation(operation) => format!("unknown_operation.{operation}"),
        TrainRequestError::IllegalExecution {
            operation,
            execution,
        } => format!("illegal_execution.{operation}.{execution:?}").to_ascii_lowercase(),
        TrainRequestError::InvalidName { namespace, name } => {
            format!("invalid_name.{namespace}.{name}")
        }
        TrainRequestError::DuplicateName { namespace, name } => {
            format!("duplicate_name.{namespace}.{name}")
        }
        TrainRequestError::ShapeOverflow { name } => format!("shape_overflow.{name}"),
        TrainRequestError::RankLimit { name, got, max } => {
            format!("rank_limit.{name}.{got}.{max}")
        }
        TrainRequestError::ElementLimit { name, got, max } => {
            format!("element_limit.{name}.{got}.{max}")
        }
        TrainRequestError::ByteCountOverflow { name } => {
            format!("byte_count_overflow.{name}")
        }
        TrainRequestError::ByteLimit { name, got, max } => {
            format!("byte_limit.{name}.{got}.{max}")
        }
        TrainRequestError::BufferLength {
            name,
            expected,
            got,
        } => format!("buffer_length.{name}.{expected}.{got}"),
        TrainRequestError::NonFiniteAttribute(name) => {
            format!("non_finite_attribute.{name}")
        }
    }
}

fn operation_error_code(error: &TrainOperationErrorV1) -> String {
    match error {
        TrainOperationErrorV1::Roles { namespace } => format!("roles.{namespace}"),
        TrainOperationErrorV1::DType {
            name,
            expected,
            got,
        } => format!("dtype.{name}.{expected:?}.{got:?}").to_ascii_lowercase(),
        TrainOperationErrorV1::Shape => "shape".to_owned(),
        TrainOperationErrorV1::NonFinite { name } => format!("non_finite.{name}"),
        TrainOperationErrorV1::AttributeType { name, expected } => {
            format!("attribute_type.{name}.{expected}")
        }
        TrainOperationErrorV1::AttributeValue { name, constraint } => {
            format!("attribute_value.{name}.{constraint}")
        }
    }
}

fn backend_error(value: TrainBackendError) -> ErrorResponse {
    let message = value.to_string();
    match value {
        TrainBackendError::InvalidRequest(error_value) => ErrorResponse {
            category: "invalid_request",
            code: request_error_code(&error_value),
            message,
        },
        TrainBackendError::UnsupportedOperation(_) => {
            error("unsupported_operation", "unsupported_operation", message)
        }
        TrainBackendError::InvalidOperation(error_value) => ErrorResponse {
            category: "invalid_operation",
            code: operation_error_code(&error_value),
            message,
        },
        TrainBackendError::Backend { code, .. } => ErrorResponse {
            category: "backend",
            code,
            message,
        },
    }
}

fn execution_name(value: TrainExecutionV1) -> &'static str {
    match value {
        TrainExecutionV1::Forward => "forward",
        TrainExecutionV1::Vjp => "vjp",
        TrainExecutionV1::Step => "step",
        TrainExecutionV1::Checkpoint => "checkpoint",
        TrainExecutionV1::Resume => "resume",
        TrainExecutionV1::Export => "export",
        TrainExecutionV1::Reload => "reload",
    }
}

fn dtype_name(value: TrainDTypeV1) -> &'static str {
    match value {
        TrainDTypeV1::F32 => "f32",
        TrainDTypeV1::U32 => "u32",
        TrainDTypeV1::Bytes => "bytes",
    }
}

impl From<TrainReceiptV1> for ReceiptResponse {
    fn from(value: TrainReceiptV1) -> Self {
        Self {
            backend_id: value.backend_id,
            backend_build: value.backend_build,
            physical_device: value.physical_device,
            manifest_digest: digest_hex(value.manifest_digest),
            vector_digest: value.vector_digest.map(digest_hex),
            operation: value.operation,
            execution: execution_name(value.execution),
            dtype: dtype_name(value.dtype),
            max_rank: value.limits.max_rank,
            max_elements: value.limits.max_elements,
            max_bytes: value.limits.max_bytes,
            input_digest: digest_hex(value.input_digest),
            output_digest: digest_hex(value.output_digest),
            peak_resident_bytes: value.peak_resident_bytes,
            scratch_bytes: value.scratch_bytes,
            host_transfers: value.host_transfers,
            device_resident: value.device_resident,
        }
    }
}

fn response_error(outputs: Vec<OwnedBuffer>, value: ErrorResponse) -> ResponseWire {
    ResponseWire::Error {
        schema_id: RESPONSE_SCHEMA_ID,
        schema_version: SCHEMA_VERSION,
        outputs: outputs.into_iter().map(BufferResponse::from).collect(),
        error: value,
    }
}

fn execute(request_json: &str) -> ResponseWire {
    if request_json.len() > MAX_REQUEST_JSON_BYTES {
        return response_error(
            Vec::new(),
            error("capacity", "request_bytes", "request JSON exceeds 8 MiB"),
        );
    }
    let wire: RequestWire = match serde_json::from_str(request_json) {
        Ok(value) => value,
        Err(parse_error) => {
            return response_error(
                Vec::new(),
                error("invalid_request", "invalid_json", parse_error.to_string()),
            );
        }
    };
    if wire.schema_id != REQUEST_SCHEMA_ID || wire.schema_version != SCHEMA_VERSION {
        return response_error(
            Vec::new(),
            error(
                "invalid_request",
                "unsupported_schema",
                "unsupported request schema",
            ),
        );
    }
    let vector_digest = match &wire.vector_digest {
        RequiredVectorDigest::Missing => {
            return response_error(
                Vec::new(),
                error(
                    "invalid_request",
                    "missing_field",
                    "vectorDigest is required",
                ),
            );
        }
        RequiredVectorDigest::Present(value) => match parse_digest(value.clone()) {
            Ok(value) => value,
            Err(value) => return response_error(Vec::new(), value),
        },
    };
    if vector_digest.is_some_and(|digest| digest != tritium_spec::TrainingVectorSetV1::digest()) {
        return response_error(
            Vec::new(),
            error(
                "invalid_request",
                "vector_digest_mismatch",
                "vectorDigest does not identify the canonical corpus",
            ),
        );
    }
    if let Err(value) = preflight_request(&wire) {
        return response_error(Vec::new(), value);
    }
    let backend = match WasmTrainBackendV1::new(wire.physical_device) {
        Ok(value) => value,
        Err(value) => {
            return response_error(
                Vec::new(),
                error("invalid_request", "physical_device", value.to_string()),
            );
        }
    };
    let inputs: Vec<OwnedBuffer> = wire.inputs.into_iter().map(OwnedBuffer::from).collect();
    let attributes: Vec<OwnedAttribute> = wire
        .attributes
        .into_iter()
        .map(OwnedAttribute::from)
        .collect();
    let mut outputs: Vec<OwnedBuffer> = wire.outputs.into_iter().map(OwnedBuffer::from).collect();
    let original_outputs = outputs.clone();
    let input_views: Vec<_> = inputs.iter().map(OwnedBuffer::as_ref).collect();
    let attribute_views: Vec<_> = attributes.iter().map(OwnedAttribute::as_ref).collect();
    let execution = TrainExecutionV1::from(wire.execution);
    let result = {
        let mut output_views: Vec<_> = outputs.iter_mut().map(OwnedBuffer::as_mut).collect();
        let mut output = TrainOutputV1::new(&mut output_views);
        let request =
            TrainRequestV1::new(&wire.operation, execution, &input_views, &attribute_views);
        let request = match vector_digest {
            Some(digest) => request.with_vector_digest(digest),
            None => request,
        };
        backend.execute(request, &mut output)
    };
    match result {
        Ok(receipt) => ResponseWire::Ok {
            schema_id: RESPONSE_SCHEMA_ID,
            schema_version: SCHEMA_VERSION,
            outputs: outputs.into_iter().map(BufferResponse::from).collect(),
            receipt: Box::new(ReceiptResponse::from(receipt)),
        },
        Err(value) => response_error(original_outputs, backend_error(value)),
    }
}

fn serialize_response(response: &ResponseWire) -> String {
    serde_json::to_string(response).unwrap_or_else(|_| {
        "{\"status\":\"error\",\"schemaId\":\"tritium.portable_training_response\",\"schemaVersion\":1,\"outputs\":[],\"error\":{\"category\":\"internal\",\"code\":\"serialization\",\"message\":\"response serialization failed\"}}".to_owned()
    })
}

/// Execute one strict, versioned portable-training request.
///
/// Expected validation and backend failures return versioned JSON with stable
/// category/code fields. A guest trap is mapped by the JavaScript wrapper.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen)]
pub fn tritium_execute_portable_request_json(request_json: &str) -> String {
    serialize_response(&execute(request_json))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use tritium_spec::{
        TrainingOpManifestV1, TrainingVectorAttributeV1, TrainingVectorAttributeValueV1,
        TrainingVectorBufferDataV1, TrainingVectorBufferV1, TrainingVectorErrorCategoryV1,
        TrainingVectorExpectedV1, TrainingVectorSetV1,
    };

    use super::*;

    fn execute_value(request: Value) -> Value {
        serde_json::from_str(&tritium_execute_portable_request_json(
            &serde_json::to_string(&request).expect("serialize request"),
        ))
        .expect("parse response")
    }

    fn sgd_request() -> Value {
        json!({
            "schemaId": REQUEST_SCHEMA_ID,
            "schemaVersion": 1,
            "physicalDevice": "wasm32:test",
            "operation": "optimizer.sgd",
            "execution": "step",
            "vectorDigest": null,
            "inputs": [
                {
                    "name": "parameter",
                    "shape": [2],
                    "data": { "dtype": "f32", "bits": [1065353216_u32, 3221225472_u32] }
                },
                {
                    "name": "gradient",
                    "shape": [2],
                    "data": { "dtype": "f32", "bits": [1056964608_u32, 3196059648_u32] }
                }
            ],
            "attributes": [
                { "kind": "u64", "name": "step", "value": 1 },
                { "kind": "f32", "name": "lr", "bits": 1036831949_u32 }
            ],
            "outputs": [{
                "name": "parameter",
                "shape": [2],
                "data": { "dtype": "f32", "bits": [0, 0] }
            }]
        })
    }

    fn vector_buffer(value: &TrainingVectorBufferV1) -> Value {
        let data = match &value.data {
            TrainingVectorBufferDataV1::F32Bits(bits) => {
                json!({ "dtype": "f32", "bits": bits })
            }
            TrainingVectorBufferDataV1::U32(values) => {
                json!({ "dtype": "u32", "values": values })
            }
            TrainingVectorBufferDataV1::Bytes(values) => {
                json!({ "dtype": "bytes", "values": values })
            }
        };
        json!({ "name": value.name, "shape": value.shape, "data": data })
    }

    fn vector_attribute(value: &TrainingVectorAttributeV1) -> Value {
        match &value.value {
            TrainingVectorAttributeValueV1::F32Bits(bits) => {
                json!({ "kind": "f32", "name": value.name, "bits": bits })
            }
            TrainingVectorAttributeValueV1::U64(attribute) => {
                json!({ "kind": "u64", "name": value.name, "value": attribute })
            }
            TrainingVectorAttributeValueV1::Bool(attribute) => {
                json!({ "kind": "bool", "name": value.name, "value": attribute })
            }
            TrainingVectorAttributeValueV1::Text(attribute) => {
                json!({ "kind": "text", "name": value.name, "value": attribute })
            }
            TrainingVectorAttributeValueV1::U64List(values) => {
                json!({ "kind": "u64-list", "name": value.name, "values": values })
            }
            TrainingVectorAttributeValueV1::U32List(values) => {
                json!({ "kind": "u32-list", "name": value.name, "values": values })
            }
        }
    }

    fn vector_error_category(value: TrainingVectorErrorCategoryV1) -> &'static str {
        match value {
            TrainingVectorErrorCategoryV1::InvalidRequest => "invalid_request",
            TrainingVectorErrorCategoryV1::InvalidOperation => "invalid_operation",
            TrainingVectorErrorCategoryV1::Backend => "backend",
        }
    }

    fn vector_dtype(value: &TrainingVectorBufferV1) -> &'static str {
        match &value.data {
            TrainingVectorBufferDataV1::F32Bits(_) => "f32",
            TrainingVectorBufferDataV1::U32(_) => "u32",
            TrainingVectorBufferDataV1::Bytes(_) => "bytes",
        }
    }

    #[test]
    fn executes_strict_sgd_request() {
        let response = execute_value(sgd_request());
        assert_eq!(response["status"], "ok");
        assert_eq!(response["schemaId"], RESPONSE_SCHEMA_ID);
        assert_eq!(
            response["outputs"][0]["data"]["bits"],
            json!([1064514355_u32, 3221015757_u32])
        );
        assert_eq!(response["receipt"]["backendId"], "wasm.portable.v1");
        assert_eq!(response["receipt"]["physicalDevice"], "wasm32:test");
        assert_eq!(response["receipt"]["hostTransfers"], 0);
        assert_eq!(response["receipt"]["deviceResident"], true);
    }

    #[test]
    fn rejects_unknown_fields_before_execution() {
        let mut request = sgd_request();
        request["unknown"] = json!(true);
        let response = execute_value(request);
        assert_eq!(response["status"], "error");
        assert_eq!(response["error"]["code"], "invalid_json");
        assert_eq!(response["outputs"], json!([]));
    }

    #[test]
    fn rejects_duplicate_fields_before_execution() {
        let request = serde_json::to_string(&sgd_request()).expect("serialize request");
        let request = request.replacen(
            "\"schemaVersion\":1",
            "\"schemaVersion\":1,\"schemaVersion\":1",
            1,
        );
        let response: Value =
            serde_json::from_str(&tritium_execute_portable_request_json(&request))
                .expect("parse response");
        assert_eq!(response["status"], "error");
        assert_eq!(response["error"]["code"], "invalid_json");
    }

    #[test]
    fn distinguishes_missing_vector_digest_from_null() {
        let mut request = sgd_request();
        request
            .as_object_mut()
            .expect("request object")
            .remove("vectorDigest");
        let response = execute_value(request);
        assert_eq!(response["status"], "error");
        assert_eq!(response["error"]["code"], "missing_field");
    }

    #[test]
    fn rejects_javascript_unsafe_u64_values() {
        let mut request = sgd_request();
        request["attributes"][0]["value"] = json!(MAX_SAFE_INTEGER + 1);
        let response = execute_value(request);
        assert_eq!(response["status"], "error");
        assert_eq!(response["error"]["code"], "unsafe_integer");
    }

    #[test]
    fn backend_error_preserves_output_sentinel() {
        let mut request = sgd_request();
        request["inputs"][0]["shape"] = json!([3]);
        request["outputs"][0]["data"]["bits"] = json!([0x7fc0_0001_u32, 0x7fc0_0002_u32]);
        let response = execute_value(request);
        assert_eq!(response["status"], "error");
        assert_eq!(response["error"]["code"], "buffer_length.parameter.3.2");
        assert_eq!(
            response["outputs"][0]["data"]["bits"],
            json!([0x7fc0_0001_u32, 0x7fc0_0002_u32])
        );
    }

    #[test]
    fn strict_bridge_reaches_every_canonical_case() {
        let vectors = TrainingVectorSetV1::parse_json(TrainingVectorSetV1::canonical_json())
            .expect("canonical vectors");
        for case in vectors.cases() {
            let expected_outputs = match &case.expected {
                TrainingVectorExpectedV1::Success { outputs, .. }
                | TrainingVectorExpectedV1::Error { outputs, .. } => outputs,
            };
            let request = json!({
                "schemaId": REQUEST_SCHEMA_ID,
                "schemaVersion": 1,
                "physicalDevice": "wasm32:bridge-test",
                "operation": case.operation,
                "execution": execution_name(case.execution),
                "vectorDigest": digest_hex(vectors.source_digest()),
                "inputs": case.inputs.iter().map(vector_buffer).collect::<Vec<_>>(),
                "attributes": case.attributes.iter().map(vector_attribute).collect::<Vec<_>>(),
                "outputs": expected_outputs.iter().map(vector_buffer).collect::<Vec<_>>(),
            });
            let response = execute_value(request);
            match &case.expected {
                TrainingVectorExpectedV1::Success {
                    outputs,
                    scratch_bytes_max,
                } => {
                    assert_eq!(response["status"], "ok", "case {}", case.case_id);
                    assert_eq!(
                        response["outputs"],
                        Value::Array(outputs.iter().map(vector_buffer).collect()),
                        "case {} output mismatch",
                        case.case_id
                    );
                    let receipt = &response["receipt"];
                    assert_eq!(receipt["backendId"], "wasm.portable.v1");
                    assert_eq!(
                        receipt["backendBuild"],
                        format!(
                            "{}@{}+{}",
                            env!("CARGO_PKG_NAME"),
                            env!("CARGO_PKG_VERSION"),
                            env!("TRITIUM_SOURCE_ID")
                        )
                    );
                    assert_eq!(receipt["physicalDevice"], "wasm32:bridge-test");
                    assert_eq!(
                        receipt["manifestDigest"],
                        digest_hex(TrainingOpManifestV1::digest())
                    );
                    assert_eq!(receipt["vectorDigest"], digest_hex(vectors.source_digest()));
                    assert_eq!(receipt["operation"], case.operation);
                    assert_eq!(receipt["execution"], execution_name(case.execution));
                    assert_eq!(receipt["dtype"], vector_dtype(&outputs[0]));
                    assert_eq!(receipt["maxRank"], 4);
                    assert_eq!(receipt["maxElements"], 8 * 1024 * 1024);
                    assert_eq!(receipt["maxBytes"], 8 * 1024 * 1024);
                    assert_eq!(receipt["hostTransfers"], 0);
                    assert_eq!(receipt["deviceResident"], true);
                    assert!(
                        receipt["peakResidentBytes"]
                            .as_u64()
                            .is_some_and(|value| value <= 64 * 1024 * 1024)
                    );
                    assert!(
                        receipt["scratchBytes"]
                            .as_u64()
                            .is_some_and(|value| value <= *scratch_bytes_max)
                    );
                    assert!(
                        receipt["inputDigest"]
                            .as_str()
                            .is_some_and(|value| value.len() == 64)
                    );
                    assert!(
                        receipt["outputDigest"]
                            .as_str()
                            .is_some_and(|value| value.len() == 64)
                    );
                }
                TrainingVectorExpectedV1::Error {
                    category,
                    code,
                    outputs,
                } => {
                    assert_eq!(response["status"], "error", "case {}", case.case_id);
                    assert_eq!(
                        response["error"]["category"],
                        vector_error_category(*category)
                    );
                    assert_eq!(response["error"]["code"], code.as_str());
                    assert_eq!(
                        response["outputs"],
                        Value::Array(outputs.iter().map(vector_buffer).collect()),
                        "case {} sentinel mismatch",
                        case.case_id
                    );
                }
            }
        }
    }
}
