//! Generic replay harness for portable-training semantic vectors.

use tritium_spec::{
    TrainAttributeV1, TrainAttributeValueV1, TrainBackendError, TrainBackendV1, TrainExecutionV1,
    TrainOperationErrorV1, TrainOutputV1, TrainOwnedBufferDataV1, TrainOwnedBufferV1,
    TrainReceiptV1, TrainRequestError, TrainRequestV1, TrainingToleranceV1,
    TrainingVectorAttributeV1, TrainingVectorAttributeValueV1, TrainingVectorBufferDataV1,
    TrainingVectorBufferV1, TrainingVectorCaseV1, TrainingVectorErrorCategoryV1,
    TrainingVectorExpectedV1, TrainingVectorSetV2, train_output_digest_v1, train_request_digest_v1,
};

/// One portable-training vector that matched its frozen result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingVectorPass {
    /// Permanent case identifier.
    pub case_id: String,
    /// Corpus-bound success receipt; expected failures emit no success receipt.
    pub receipt: Option<TrainReceiptV1>,
}

/// Why one portable-training vector failed conformance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrainingVectorFailureReason {
    /// Backend returned an error for a success vector.
    Backend(TrainBackendError),
    /// Backend succeeded when the vector required a structured failure.
    ExpectedErrorNotRaised,
    /// Backend error category or stable code differed from the vector.
    ErrorMismatch {
        /// Frozen category.
        expected_category: TrainingVectorErrorCategoryV1,
        /// Frozen stable code.
        expected_code: String,
        /// Observed category.
        got_category: TrainingVectorErrorCategoryV1,
        /// Observed stable code.
        got_code: String,
    },
    /// Backend returned a different output count.
    OutputCount {
        /// Frozen count.
        expected: usize,
        /// Backend count.
        got: usize,
    },
    /// Output role, shape, dtype, length or numeric payload disagreed.
    OutputMismatch {
        /// Output role when available.
        name: String,
        /// First mismatching element, or `None` for metadata mismatch.
        index: Option<usize>,
    },
    /// Receipt field disagreed with the executed vector/corpus.
    Receipt(String),
    /// Backend exceeded the vector's peak temporary-byte ceiling.
    ScratchBytes {
        /// Frozen ceiling.
        maximum: u64,
        /// Receipted peak.
        got: u64,
    },
}

/// One failed portable-training vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingVectorFailure {
    /// Permanent case identifier.
    pub case_id: String,
    /// Exact failure reason.
    pub reason: TrainingVectorFailureReason,
}

/// Aggregate portable-training conformance result.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TrainingConformanceReport {
    /// Successful cases and any content-bound success receipts.
    pub passed: Vec<TrainingVectorPass>,
    /// Failed cases; an admissible backend report has none.
    pub failed: Vec<TrainingVectorFailure>,
}

impl TrainingConformanceReport {
    /// True only when every supplied vector passed.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.failed.is_empty()
    }
}

/// Execute every parsed semantic vector through one backend.
///
/// Success vectors are graded with their frozen tolerance plus receipt and
/// scratch invariants. Error vectors grade stable category/code and prove every
/// caller-owned output sentinel remains bit-exact.
#[must_use]
pub fn run_training_conformance(
    backend: &dyn TrainBackendV1,
    vectors: &TrainingVectorSetV2,
) -> TrainingConformanceReport {
    run_training_conformance_matching(backend, vectors, |_| true)
}

/// Execute the canonical cases for every operation a partial adapter advertises.
///
/// This is an incremental-development gate, not release admission: a v1 release
/// backend must still advertise the complete manifest and pass
/// [`run_training_conformance`].
#[must_use]
pub fn run_supported_training_conformance(
    backend: &dyn TrainBackendV1,
    vectors: &TrainingVectorSetV2,
) -> TrainingConformanceReport {
    let supported = backend.capabilities().supported_operations;
    run_training_conformance_matching(backend, vectors, |operation| {
        supported.iter().any(|supported| supported == operation)
    })
}

fn run_training_conformance_matching(
    backend: &dyn TrainBackendV1,
    vectors: &TrainingVectorSetV2,
    include: impl Fn(&str) -> bool,
) -> TrainingConformanceReport {
    let mut report = TrainingConformanceReport::default();
    for case in vectors
        .cases()
        .iter()
        .filter(|case| include(&case.operation))
    {
        match run_case(backend, vectors, case) {
            Ok(receipt) => report.passed.push(TrainingVectorPass {
                case_id: case.case_id.clone(),
                receipt,
            }),
            Err(reason) => report.failed.push(TrainingVectorFailure {
                case_id: case.case_id.clone(),
                reason,
            }),
        }
    }
    report
}

fn run_case(
    backend: &dyn TrainBackendV1,
    vectors: &TrainingVectorSetV2,
    case: &TrainingVectorCaseV1,
) -> Result<Option<TrainReceiptV1>, TrainingVectorFailureReason> {
    match &case.expected {
        TrainingVectorExpectedV1::Success {
            outputs,
            scratch_bytes_max,
        } => run_success_case(backend, vectors, case, outputs, *scratch_bytes_max).map(Some),
        TrainingVectorExpectedV1::Error {
            category,
            code,
            outputs,
        } => run_error_case(backend, vectors, case, *category, code, outputs),
    }
}

fn run_success_case(
    backend: &dyn TrainBackendV1,
    vectors: &TrainingVectorSetV2,
    case: &TrainingVectorCaseV1,
    expected_outputs: &[TrainingVectorBufferV1],
    scratch_bytes_max: u64,
) -> Result<TrainReceiptV1, TrainingVectorFailureReason> {
    let inputs: Vec<_> = case.inputs.iter().map(materialize_buffer).collect();
    let input_views: Vec<_> = inputs.iter().map(TrainOwnedBufferV1::as_ref).collect();
    let attributes: Vec<_> = case.attributes.iter().map(attribute_view).collect();
    let mut outputs: Vec<_> = expected_outputs
        .iter()
        .map(|buffer| poison_buffer(buffer, case.tolerance))
        .collect();
    let (receipt, expected_input_digest, expected_output_digest) = {
        let mut output_views: Vec<_> = outputs.iter_mut().map(TrainOwnedBufferV1::as_mut).collect();
        let mut output = TrainOutputV1::new(&mut output_views);
        let request = vector_request(case, vectors, &input_views, &attributes);
        let expected_input_digest = train_request_digest_v1(&request);
        let receipt = backend
            .execute(request, &mut output)
            .map_err(TrainingVectorFailureReason::Backend)?;
        let expected_output_digest = train_output_digest_v1(&output);
        (receipt, expected_input_digest, expected_output_digest)
    };

    grade_outputs(&outputs, expected_outputs, case.tolerance)?;
    grade_receipt(
        backend,
        &receipt,
        vectors,
        &case.operation,
        case.execution,
        expected_input_digest,
        expected_output_digest,
    )?;
    if receipt.scratch_bytes > scratch_bytes_max {
        return Err(TrainingVectorFailureReason::ScratchBytes {
            maximum: scratch_bytes_max,
            got: receipt.scratch_bytes,
        });
    }
    Ok(receipt)
}

fn run_error_case(
    backend: &dyn TrainBackendV1,
    vectors: &TrainingVectorSetV2,
    case: &TrainingVectorCaseV1,
    expected_category: TrainingVectorErrorCategoryV1,
    expected_code: &str,
    expected_outputs: &[TrainingVectorBufferV1],
) -> Result<Option<TrainReceiptV1>, TrainingVectorFailureReason> {
    let inputs: Vec<_> = case.inputs.iter().map(materialize_buffer).collect();
    let input_views: Vec<_> = inputs.iter().map(TrainOwnedBufferV1::as_ref).collect();
    let attributes: Vec<_> = case.attributes.iter().map(attribute_view).collect();
    let mut outputs: Vec<_> = expected_outputs.iter().map(materialize_buffer).collect();
    let result = {
        let mut output_views: Vec<_> = outputs.iter_mut().map(TrainOwnedBufferV1::as_mut).collect();
        let mut output = TrainOutputV1::new(&mut output_views);
        backend.execute(
            vector_request(case, vectors, &input_views, &attributes),
            &mut output,
        )
    };

    let error = match result {
        Ok(_) => return Err(TrainingVectorFailureReason::ExpectedErrorNotRaised),
        Err(error) => error,
    };
    let (got_category, got_code) = error_identity(&error);
    if got_category != expected_category || got_code != expected_code {
        return Err(TrainingVectorFailureReason::ErrorMismatch {
            expected_category,
            expected_code: expected_code.to_owned(),
            got_category,
            got_code,
        });
    }
    grade_outputs(&outputs, expected_outputs, TrainingToleranceV1::BitExact)?;
    Ok(None)
}

fn vector_request<'a>(
    case: &'a TrainingVectorCaseV1,
    vectors: &TrainingVectorSetV2,
    inputs: &'a [tritium_spec::TrainNamedBufferRefV1<'a>],
    attributes: &'a [TrainAttributeV1<'a>],
) -> TrainRequestV1<'a> {
    TrainRequestV1::new(&case.operation, case.execution, inputs, attributes)
        .with_vector_digest(vectors.source_digest())
}

pub(crate) fn canonical_case_input_digest(
    case: &TrainingVectorCaseV1,
    vectors: &TrainingVectorSetV2,
) -> [u8; 32] {
    let inputs: Vec<_> = case.inputs.iter().map(materialize_buffer).collect();
    let input_views: Vec<_> = inputs.iter().map(TrainOwnedBufferV1::as_ref).collect();
    let attributes: Vec<_> = case.attributes.iter().map(attribute_view).collect();
    train_request_digest_v1(&vector_request(case, vectors, &input_views, &attributes))
}

fn materialize_buffer(buffer: &TrainingVectorBufferV1) -> TrainOwnedBufferV1 {
    let data = match &buffer.data {
        TrainingVectorBufferDataV1::F32Bits(bits) => {
            TrainOwnedBufferDataV1::F32(bits.iter().map(|bits| f32::from_bits(*bits)).collect())
        }
        TrainingVectorBufferDataV1::U32(values) => TrainOwnedBufferDataV1::U32(values.clone()),
        TrainingVectorBufferDataV1::Bytes(values) => TrainOwnedBufferDataV1::Bytes(values.clone()),
    };
    TrainOwnedBufferV1 {
        name: buffer.name.clone(),
        shape: buffer.shape.clone(),
        data,
    }
}

fn poison_buffer(
    buffer: &TrainingVectorBufferV1,
    tolerance: TrainingToleranceV1,
) -> TrainOwnedBufferV1 {
    let data = match &buffer.data {
        TrainingVectorBufferDataV1::F32Bits(bits) => {
            let values = match tolerance {
                TrainingToleranceV1::BitExact => bits
                    .iter()
                    .map(|bits| f32::from_bits(bits ^ 0x0040_0000))
                    .collect(),
                TrainingToleranceV1::AbsoluteRelative { .. } => vec![f32::NAN; bits.len()],
            };
            TrainOwnedBufferDataV1::F32(values)
        }
        TrainingVectorBufferDataV1::U32(values) => {
            TrainOwnedBufferDataV1::U32(values.iter().map(|value| value ^ u32::MAX).collect())
        }
        TrainingVectorBufferDataV1::Bytes(values) => {
            TrainOwnedBufferDataV1::Bytes(values.iter().map(|value| value ^ u8::MAX).collect())
        }
    };
    TrainOwnedBufferV1 {
        name: buffer.name.clone(),
        shape: buffer.shape.clone(),
        data,
    }
}

fn attribute_view(attribute: &TrainingVectorAttributeV1) -> TrainAttributeV1<'_> {
    let value = match &attribute.value {
        TrainingVectorAttributeValueV1::F32Bits(bits) => {
            TrainAttributeValueV1::F32(f32::from_bits(*bits))
        }
        TrainingVectorAttributeValueV1::U64(value) => TrainAttributeValueV1::U64(*value),
        TrainingVectorAttributeValueV1::Bool(value) => TrainAttributeValueV1::Bool(*value),
        TrainingVectorAttributeValueV1::Text(value) => TrainAttributeValueV1::Text(value),
        TrainingVectorAttributeValueV1::U64List(values) => TrainAttributeValueV1::U64List(values),
        TrainingVectorAttributeValueV1::U32List(values) => TrainAttributeValueV1::U32List(values),
    };
    TrainAttributeV1::new(&attribute.name, value)
}

fn grade_outputs(
    actual: &[TrainOwnedBufferV1],
    expected: &[TrainingVectorBufferV1],
    tolerance: TrainingToleranceV1,
) -> Result<(), TrainingVectorFailureReason> {
    if actual.len() != expected.len() {
        return Err(TrainingVectorFailureReason::OutputCount {
            expected: expected.len(),
            got: actual.len(),
        });
    }
    for (actual, expected) in actual.iter().zip(expected) {
        let name = expected.name.clone();
        if actual.name != expected.name || actual.shape != expected.shape {
            return Err(TrainingVectorFailureReason::OutputMismatch { name, index: None });
        }
        match (&actual.data, &expected.data) {
            (TrainOwnedBufferDataV1::F32(values), TrainingVectorBufferDataV1::F32Bits(bits)) => {
                if values.len() != bits.len() {
                    return Err(TrainingVectorFailureReason::OutputMismatch { name, index: None });
                }
                for (index, (&actual, &expected)) in values.iter().zip(bits).enumerate() {
                    if !accepts(tolerance, actual, f32::from_bits(expected)) {
                        return Err(TrainingVectorFailureReason::OutputMismatch {
                            name,
                            index: Some(index),
                        });
                    }
                }
            }
            (TrainOwnedBufferDataV1::U32(actual), TrainingVectorBufferDataV1::U32(expected))
                if actual == expected => {}
            (
                TrainOwnedBufferDataV1::Bytes(actual),
                TrainingVectorBufferDataV1::Bytes(expected),
            ) if actual == expected => {}
            _ => return Err(TrainingVectorFailureReason::OutputMismatch { name, index: None }),
        }
    }
    Ok(())
}

fn accepts(tolerance: TrainingToleranceV1, actual: f32, expected: f32) -> bool {
    match tolerance {
        TrainingToleranceV1::BitExact => actual.to_bits() == expected.to_bits(),
        TrainingToleranceV1::AbsoluteRelative {
            absolute_bits,
            relative_bits,
        } => {
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs()
                    <= f32::from_bits(absolute_bits)
                        + f32::from_bits(relative_bits) * expected.abs()
        }
    }
}

fn grade_receipt(
    backend: &dyn TrainBackendV1,
    receipt: &TrainReceiptV1,
    vectors: &TrainingVectorSetV2,
    operation: &str,
    execution: TrainExecutionV1,
    expected_input_digest: [u8; 32],
    expected_output_digest: [u8; 32],
) -> Result<(), TrainingVectorFailureReason> {
    let capabilities = backend.capabilities();
    if receipt.backend_id != capabilities.backend_id {
        return Err(TrainingVectorFailureReason::Receipt(
            "backend_id".to_owned(),
        ));
    }
    if receipt.backend_build.is_empty() {
        return Err(TrainingVectorFailureReason::Receipt(
            "backend_build".to_owned(),
        ));
    }
    if !matches!(receipt.physical_device.as_deref(), Some(device) if !device.is_empty()) {
        return Err(TrainingVectorFailureReason::Receipt(
            "physical_device".to_owned(),
        ));
    }
    if receipt.manifest_digest != vectors.manifest_digest() {
        return Err(TrainingVectorFailureReason::Receipt(
            "manifest_digest".to_owned(),
        ));
    }
    if receipt.vector_digest != Some(vectors.source_digest()) {
        return Err(TrainingVectorFailureReason::Receipt(
            "vector_digest".to_owned(),
        ));
    }
    if receipt.operation != operation {
        return Err(TrainingVectorFailureReason::Receipt("operation".to_owned()));
    }
    if receipt.execution != execution {
        return Err(TrainingVectorFailureReason::Receipt("execution".to_owned()));
    }
    if receipt.input_digest != expected_input_digest {
        return Err(TrainingVectorFailureReason::Receipt(
            "input_digest".to_owned(),
        ));
    }
    if receipt.output_digest != expected_output_digest {
        return Err(TrainingVectorFailureReason::Receipt(
            "output_digest".to_owned(),
        ));
    }
    if !capabilities.dtypes.contains(&receipt.dtype) {
        return Err(TrainingVectorFailureReason::Receipt("dtype".to_owned()));
    }
    if receipt.limits != capabilities.limits {
        return Err(TrainingVectorFailureReason::Receipt("limits".to_owned()));
    }
    if !receipt.device_resident || receipt.host_transfers != 0 {
        return Err(TrainingVectorFailureReason::Receipt("residency".to_owned()));
    }
    Ok(())
}

fn error_identity(error: &TrainBackendError) -> (TrainingVectorErrorCategoryV1, String) {
    match error {
        TrainBackendError::InvalidRequest(error) => (
            TrainingVectorErrorCategoryV1::InvalidRequest,
            request_error_code(error),
        ),
        TrainBackendError::InvalidOperation(error) => (
            TrainingVectorErrorCategoryV1::InvalidOperation,
            operation_error_code(error),
        ),
        TrainBackendError::UnsupportedOperation(_) => (
            TrainingVectorErrorCategoryV1::Backend,
            "unsupported_operation".to_owned(),
        ),
        TrainBackendError::Backend { code, .. } => {
            (TrainingVectorErrorCategoryV1::Backend, code.clone())
        }
    }
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
