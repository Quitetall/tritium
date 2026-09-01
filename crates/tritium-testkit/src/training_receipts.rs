//! Canonical release evidence for portable-training backend conformance.

use core::fmt;
use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use tritium_spec::{
    TrainBackendV1, TrainCapabilitiesV1, TrainDTypeV1, TrainExecutionV1, TrainLimitsV1,
    TrainReceiptV1, TrainingVectorBufferDataV1, TrainingVectorBufferV1, TrainingVectorExpectedV1,
};

use crate::{
    TrainingVectorCorpus, portable_training::canonical_case_input_digest, run_training_conformance,
};

const SCHEMA_ID: &str = "tritium.training_receipts";
const SCHEMA_VERSION: u32 = 1;

/// Failure while sealing or admitting portable-training release evidence.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TrainingReceiptBundleError {
    /// JSON syntax or shape was invalid.
    Json(String),
    /// Schema identity/version was not v1.
    Schema,
    /// Bytes were valid JSON but not canonical Tritium encoding.
    NonCanonical,
    /// Canonical bytes did not match the trusted release-candidate identity.
    ContentDigest,
    /// Development evidence was offered to a release-only renderer.
    DevelopmentEvidence,
    /// Backend capability declaration was not release-admissible.
    Capabilities(String),
    /// Conformance report did not prove every canonical vector.
    Coverage(String),
    /// One success receipt disagreed with its case or bundle identity.
    Receipt {
        /// Canonical vector case identity.
        case_id: String,
        /// Receipt field that failed admission.
        field: String,
    },
    /// Capability-table inputs repeated one backend/device identity.
    DuplicateBackend {
        /// Stable backend adapter identity.
        backend_id: String,
        /// Physical target identity.
        device: String,
    },
}

impl fmt::Display for TrainingReceiptBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(message) => write!(f, "invalid training receipt JSON: {message}"),
            Self::Schema => write!(f, "training receipt schema identity is not v1"),
            Self::NonCanonical => write!(f, "training receipt bytes are not canonical"),
            Self::ContentDigest => write!(f, "training receipt content digest is not trusted"),
            Self::DevelopmentEvidence => {
                write!(
                    f,
                    "development training receipts cannot enter a release table"
                )
            }
            Self::Capabilities(field) => {
                write!(f, "training receipt capabilities reject field {field}")
            }
            Self::Coverage(message) => write!(f, "training receipt coverage: {message}"),
            Self::Receipt { case_id, field } => {
                write!(f, "training receipt {case_id:?} rejects field {field}")
            }
            Self::DuplicateBackend { backend_id, device } => write!(
                f,
                "duplicate training receipt backend/device {backend_id:?}/{device:?}"
            ),
        }
    }
}

impl std::error::Error for TrainingReceiptBundleError {}

/// Canonical JSON plus its BLAKE3 content identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedTrainingReceiptBundleV1 {
    bytes: Vec<u8>,
    digest: [u8; 32],
}

impl SealedTrainingReceiptBundleV1 {
    /// Exact canonical JSON bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// BLAKE3 identity of [`Self::bytes`].
    #[must_use]
    pub const fn digest(&self) -> [u8; 32] {
        self.digest
    }

    /// Lowercase hexadecimal BLAKE3 identity of [`Self::bytes`].
    #[must_use]
    pub fn digest_hex(&self) -> String {
        hex_digest(self.digest)
    }
}

/// Strictly admitted receipt bundle summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedTrainingReceiptBundleV1 {
    /// Source-state policy used during admission.
    source_policy: TrainingReceiptSourcePolicyV1,
    /// Stable backend adapter identity.
    backend_id: String,
    /// Adapter build identity.
    backend_build: String,
    /// Physical target identity.
    physical_device: String,
    /// Number of permanent operations covered.
    operation_count: usize,
    /// Number of canonical success/error cases covered.
    case_count: usize,
    /// Largest caller-visible resident-byte receipt.
    peak_resident_bytes: u64,
    /// Largest temporary scratch-byte receipt.
    peak_scratch_bytes: u64,
    /// BLAKE3 identity of admitted canonical bytes.
    bundle_digest: [u8; 32],
}

impl AdmittedTrainingReceiptBundleV1 {
    /// Source-state policy used during admission.
    #[must_use]
    pub const fn source_policy(&self) -> TrainingReceiptSourcePolicyV1 {
        self.source_policy
    }

    /// Stable backend adapter identity.
    #[must_use]
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }

    /// Adapter build identity.
    #[must_use]
    pub fn backend_build(&self) -> &str {
        &self.backend_build
    }

    /// Physical target identity.
    #[must_use]
    pub fn physical_device(&self) -> &str {
        &self.physical_device
    }

    /// Number of permanent operations covered.
    #[must_use]
    pub const fn operation_count(&self) -> usize {
        self.operation_count
    }

    /// Number of canonical success/error cases covered.
    #[must_use]
    pub const fn case_count(&self) -> usize {
        self.case_count
    }

    /// Largest caller-visible resident-byte receipt.
    #[must_use]
    pub const fn peak_resident_bytes(&self) -> u64 {
        self.peak_resident_bytes
    }

    /// Largest temporary scratch-byte receipt.
    #[must_use]
    pub const fn peak_scratch_bytes(&self) -> u64 {
        self.peak_scratch_bytes
    }

    /// BLAKE3 identity of admitted canonical bytes.
    #[must_use]
    pub const fn bundle_digest(&self) -> [u8; 32] {
        self.bundle_digest
    }
}

/// Source-state policy for receipt admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingReceiptSourcePolicyV1 {
    /// Release evidence must name one clean, immutable Git commit.
    ReleaseCandidate,
    /// Local diagnostics may admit a dirty-tree identity; never use for release claims.
    Development,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleWire {
    schema_id: String,
    schema_version: u32,
    backend_id: String,
    backend_build: String,
    physical_device: String,
    manifest_digest: String,
    vector_digest: String,
    supported_operations: Vec<String>,
    dtypes: Vec<String>,
    limits: LimitsWire,
    device_resident: bool,
    cases: Vec<CaseWire>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LimitsWire {
    max_rank: u32,
    max_elements: u64,
    max_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseWire {
    case_id: String,
    receipt: Option<ReceiptWire>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptWire {
    operation: String,
    execution: String,
    dtype: String,
    input_digest: String,
    output_digest: String,
    peak_resident_bytes: u64,
    scratch_bytes: u64,
    host_transfers: u64,
    device_resident: bool,
}

/// Seal a complete, passing report into deterministic release evidence.
///
/// # Errors
/// Rejects partial capabilities, failed/missing/reordered cases, anonymous
/// devices, fallback residency, or inconsistent receipts.
pub fn seal_training_receipts(
    backend: &dyn TrainBackendV1,
    vectors: &impl TrainingVectorCorpus,
) -> Result<SealedTrainingReceiptBundleV1, TrainingReceiptBundleError> {
    let report = run_training_conformance(backend, vectors);
    if !report.failed.is_empty() {
        return Err(TrainingReceiptBundleError::Coverage(format!(
            "{} cases failed",
            report.failed.len()
        )));
    }
    let capabilities = backend.capabilities();
    validate_capabilities(&capabilities, vectors)?;
    if report.passed.len() != vectors.cases().len() {
        return Err(TrainingReceiptBundleError::Coverage(format!(
            "passed {} of {} cases",
            report.passed.len(),
            vectors.cases().len()
        )));
    }
    let physical_device = report
        .passed
        .iter()
        .find_map(|pass| pass.receipt.as_ref()?.physical_device.clone())
        .ok_or_else(|| TrainingReceiptBundleError::Capabilities("physical_device".to_owned()))?;
    let backend_build = report
        .passed
        .iter()
        .find_map(|pass| Some(pass.receipt.as_ref()?.backend_build.clone()))
        .ok_or_else(|| TrainingReceiptBundleError::Capabilities("backend_build".to_owned()))?;
    if physical_device.trim().is_empty()
        || !admissible_build_identity(&backend_build, TrainingReceiptSourcePolicyV1::Development)
    {
        return Err(TrainingReceiptBundleError::Capabilities(
            "physical_device/backend_build".to_owned(),
        ));
    }

    let mut cases = Vec::with_capacity(report.passed.len());
    for (pass, case) in report.passed.iter().zip(vectors.cases()) {
        if pass.case_id != case.case_id {
            return Err(TrainingReceiptBundleError::Coverage(format!(
                "case order differs at {:?}",
                case.case_id
            )));
        }
        let success = matches!(case.expected, TrainingVectorExpectedV1::Success { .. });
        if success != pass.receipt.is_some() {
            return Err(receipt_error(&case.case_id, "outcome"));
        }
        let receipt = pass
            .receipt
            .as_ref()
            .map(|receipt| {
                validate_receipt(
                    receipt,
                    &capabilities,
                    vectors,
                    &physical_device,
                    &backend_build,
                    &case.case_id,
                    &case.operation,
                    case.execution,
                )?;
                Ok(receipt_wire(receipt))
            })
            .transpose()?;
        cases.push(CaseWire {
            case_id: case.case_id.clone(),
            receipt,
        });
    }
    let wire = BundleWire {
        schema_id: SCHEMA_ID.to_owned(),
        schema_version: SCHEMA_VERSION,
        backend_id: capabilities.backend_id,
        backend_build,
        physical_device,
        manifest_digest: hex_digest(vectors.manifest_digest()),
        vector_digest: hex_digest(vectors.source_digest()),
        supported_operations: capabilities.supported_operations,
        dtypes: capabilities.dtypes.into_iter().map(dtype_name).collect(),
        limits: limits_wire(capabilities.limits),
        device_resident: capabilities.device_resident,
        cases,
    };
    let bytes = canonical_bytes(&wire)?;
    let digest = *blake3::hash(&bytes).as_bytes();
    Ok(SealedTrainingReceiptBundleV1 { bytes, digest })
}

/// Parse and independently admit canonical receipt evidence.
///
/// # Errors
/// Rejects noncanonical bytes, partial/reordered coverage, identity drift,
/// malformed digests, anonymous devices, or nonresident/fallback receipts.
pub fn admit_training_receipts(
    bytes: &[u8],
    vectors: &impl TrainingVectorCorpus,
    expected_bundle_digest: [u8; 32],
    source_policy: TrainingReceiptSourcePolicyV1,
) -> Result<AdmittedTrainingReceiptBundleV1, TrainingReceiptBundleError> {
    let actual_bundle_digest = *blake3::hash(bytes).as_bytes();
    if actual_bundle_digest != expected_bundle_digest {
        return Err(TrainingReceiptBundleError::ContentDigest);
    }
    let wire: BundleWire = serde_json::from_slice(bytes)
        .map_err(|error| TrainingReceiptBundleError::Json(error.to_string()))?;
    if wire.schema_id != SCHEMA_ID || wire.schema_version != SCHEMA_VERSION {
        return Err(TrainingReceiptBundleError::Schema);
    }
    if canonical_bytes(&wire)? != bytes {
        return Err(TrainingReceiptBundleError::NonCanonical);
    }
    validate_wire(&wire, vectors, source_policy)?;
    let mut peak_resident_bytes = 0;
    let mut peak_scratch_bytes = 0;
    for case in &wire.cases {
        if let Some(receipt) = &case.receipt {
            peak_resident_bytes = peak_resident_bytes.max(receipt.peak_resident_bytes);
            peak_scratch_bytes = peak_scratch_bytes.max(receipt.scratch_bytes);
        }
    }
    Ok(AdmittedTrainingReceiptBundleV1 {
        source_policy,
        backend_id: wire.backend_id,
        backend_build: wire.backend_build,
        physical_device: wire.physical_device,
        operation_count: wire.supported_operations.len(),
        case_count: wire.cases.len(),
        peak_resident_bytes,
        peak_scratch_bytes,
        bundle_digest: actual_bundle_digest,
    })
}

/// Render deterministic Markdown from admitted receipt bundles only.
///
/// # Errors
/// Rejects duplicate backend identities rather than hiding one row.
pub fn render_training_capability_table(
    bundles: &[AdmittedTrainingReceiptBundleV1],
) -> Result<String, TrainingReceiptBundleError> {
    if bundles
        .iter()
        .any(|bundle| bundle.source_policy != TrainingReceiptSourcePolicyV1::ReleaseCandidate)
    {
        return Err(TrainingReceiptBundleError::DevelopmentEvidence);
    }
    render_training_capability_rows(bundles)
}

/// Render visibly non-release Markdown from development-admitted bundles.
///
/// # Errors
/// Rejects duplicate backend identities rather than hiding one row.
pub fn render_development_training_capability_table(
    bundles: &[AdmittedTrainingReceiptBundleV1],
) -> Result<String, TrainingReceiptBundleError> {
    let mut output = String::from(
        "> **Development evidence only.** Dirty source identities are not release-admissible.\n\n",
    );
    output.push_str(&render_training_capability_rows(bundles)?);
    Ok(output)
}

fn render_training_capability_rows(
    bundles: &[AdmittedTrainingReceiptBundleV1],
) -> Result<String, TrainingReceiptBundleError> {
    let mut rows = bundles.to_vec();
    rows.sort_by(|left, right| {
        (&left.backend_id, &left.physical_device).cmp(&(&right.backend_id, &right.physical_device))
    });
    let mut identities = HashSet::new();
    for row in &rows {
        if !identities.insert(row.backend_id.clone()) {
            return Err(TrainingReceiptBundleError::DuplicateBackend {
                backend_id: row.backend_id.clone(),
                device: row.physical_device.clone(),
            });
        }
    }
    let mut table = String::from(
        "| Backend | Build | Physical device | Operations | Cases | Peak resident bytes | Peak scratch bytes | Receipt bundle |\n|---|---|---|---:|---:|---:|---:|---|\n",
    );
    for row in rows {
        table.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | `{}` |\n",
            escape_cell(&row.backend_id),
            escape_cell(&row.backend_build),
            escape_cell(&row.physical_device),
            row.operation_count,
            row.case_count,
            row.peak_resident_bytes,
            row.peak_scratch_bytes,
            hex_digest(row.bundle_digest),
        ));
    }
    Ok(table)
}

fn validate_capabilities(
    capabilities: &TrainCapabilitiesV1,
    vectors: &impl TrainingVectorCorpus,
) -> Result<(), TrainingReceiptBundleError> {
    if capabilities.backend_id.trim().is_empty() {
        return Err(TrainingReceiptBundleError::Capabilities(
            "backend_id".to_owned(),
        ));
    }
    if capabilities.manifest_digest != vectors.manifest_digest() {
        return Err(TrainingReceiptBundleError::Capabilities(
            "manifest_digest".to_owned(),
        ));
    }
    let expected: Vec<_> = vectors
        .operations()
        .iter()
        .map(|operation| operation.id.to_owned())
        .collect();
    if capabilities.supported_operations != expected {
        return Err(TrainingReceiptBundleError::Capabilities(
            "supported_operations".to_owned(),
        ));
    }
    if !capabilities.dtypes.contains(&TrainDTypeV1::F32) {
        return Err(TrainingReceiptBundleError::Capabilities(
            "dtypes".to_owned(),
        ));
    }
    if !capabilities.device_resident {
        return Err(TrainingReceiptBundleError::Capabilities(
            "device_resident".to_owned(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_receipt(
    receipt: &TrainReceiptV1,
    capabilities: &TrainCapabilitiesV1,
    vectors: &impl TrainingVectorCorpus,
    physical_device: &str,
    backend_build: &str,
    case_id: &str,
    operation: &str,
    execution: TrainExecutionV1,
) -> Result<(), TrainingReceiptBundleError> {
    let checks = [
        (receipt.backend_id == capabilities.backend_id, "backend_id"),
        (receipt.backend_build == backend_build, "backend_build"),
        (
            receipt.physical_device.as_deref() == Some(physical_device),
            "physical_device",
        ),
        (
            receipt.manifest_digest == vectors.manifest_digest(),
            "manifest_digest",
        ),
        (
            receipt.vector_digest == Some(vectors.source_digest()),
            "vector_digest",
        ),
        (receipt.operation == operation, "operation"),
        (receipt.execution == execution, "execution"),
        (receipt.limits == capabilities.limits, "limits"),
        (receipt.host_transfers == 0, "host_transfers"),
        (receipt.device_resident, "device_resident"),
    ];
    for (valid, field) in checks {
        if !valid {
            return Err(receipt_error(case_id, field));
        }
    }
    if !capabilities.dtypes.contains(&receipt.dtype) {
        return Err(receipt_error(case_id, "dtype"));
    }
    Ok(())
}

fn validate_wire(
    wire: &BundleWire,
    vectors: &impl TrainingVectorCorpus,
    source_policy: TrainingReceiptSourcePolicyV1,
) -> Result<(), TrainingReceiptBundleError> {
    if wire.backend_id.trim().is_empty()
        || !admissible_build_identity(&wire.backend_build, source_policy)
        || wire.physical_device.trim().is_empty()
        || !wire.device_resident
    {
        return Err(TrainingReceiptBundleError::Capabilities(
            "identity/residency".to_owned(),
        ));
    }
    if parse_digest(&wire.manifest_digest)? != vectors.manifest_digest()
        || parse_digest(&wire.vector_digest)? != vectors.source_digest()
    {
        return Err(TrainingReceiptBundleError::Capabilities(
            "manifest/vector_digest".to_owned(),
        ));
    }
    let expected_operations: Vec<_> = vectors
        .operations()
        .iter()
        .map(|operation| operation.id.to_owned())
        .collect();
    if wire.supported_operations != expected_operations
        || !wire.dtypes.iter().any(|dtype| dtype == "f32")
        || wire.dtypes.iter().any(|dtype| parse_dtype(dtype).is_none())
        || wire.dtypes.iter().collect::<HashSet<_>>().len() != wire.dtypes.len()
    {
        return Err(TrainingReceiptBundleError::Capabilities(
            "operations/dtypes".to_owned(),
        ));
    }
    if wire.cases.len() != vectors.cases().len() {
        return Err(TrainingReceiptBundleError::Coverage(format!(
            "contains {} of {} cases",
            wire.cases.len(),
            vectors.cases().len()
        )));
    }
    for (wire_case, case) in wire.cases.iter().zip(vectors.cases()) {
        if wire_case.case_id != case.case_id {
            return Err(TrainingReceiptBundleError::Coverage(format!(
                "case order differs at {:?}",
                case.case_id
            )));
        }
        let success = matches!(case.expected, TrainingVectorExpectedV1::Success { .. });
        if success != wire_case.receipt.is_some() {
            return Err(receipt_error(&case.case_id, "outcome"));
        }
        if let Some(receipt) = &wire_case.receipt
            && (receipt.operation != case.operation
                || parse_execution(&receipt.execution) != Some(case.execution)
                || !wire.dtypes.iter().any(|dtype| dtype == &receipt.dtype)
                || parse_digest(&receipt.input_digest).ok()
                    != Some(canonical_case_input_digest(case, vectors))
                || parse_digest(&receipt.output_digest).is_err()
                || receipt.peak_resident_bytes != expected_resident_bytes(case)
                || receipt.host_transfers != 0
                || !receipt.device_resident)
        {
            return Err(receipt_error(&case.case_id, "payload"));
        }
        if let (
            Some(receipt),
            TrainingVectorExpectedV1::Success {
                scratch_bytes_max, ..
            },
        ) = (&wire_case.receipt, &case.expected)
            && receipt.scratch_bytes > *scratch_bytes_max
        {
            return Err(receipt_error(&case.case_id, "scratch_bytes"));
        }
    }
    Ok(())
}

fn admissible_build_identity(identity: &str, source_policy: TrainingReceiptSourcePolicyV1) -> bool {
    let Some((package, source)) = identity.split_once("+source-git:") else {
        return false;
    };
    if package.trim().is_empty() {
        return false;
    }
    let (commit, dirty) = source
        .split_once("+dirty-blake3:")
        .map_or((source, None), |(commit, digest)| (commit, Some(digest)));
    ((commit.len() == 40) || (commit.len() == 64))
        && commit.bytes().all(is_lower_hex)
        && match (source_policy, dirty) {
            (TrainingReceiptSourcePolicyV1::ReleaseCandidate, None) => true,
            (TrainingReceiptSourcePolicyV1::Development, None) => true,
            (TrainingReceiptSourcePolicyV1::Development, Some(digest)) => {
                digest.len() == 64 && digest.bytes().all(is_lower_hex)
            }
            (TrainingReceiptSourcePolicyV1::ReleaseCandidate, Some(_)) => false,
        }
}

fn is_lower_hex(byte: u8) -> bool {
    byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()
}

fn expected_resident_bytes(case: &tritium_spec::TrainingVectorCaseV1) -> u64 {
    let outputs = match &case.expected {
        TrainingVectorExpectedV1::Success { outputs, .. }
        | TrainingVectorExpectedV1::Error { outputs, .. } => outputs,
    };
    case.inputs.iter().chain(outputs).map(buffer_bytes).sum()
}

fn buffer_bytes(buffer: &TrainingVectorBufferV1) -> u64 {
    match &buffer.data {
        TrainingVectorBufferDataV1::F32Bits(values) | TrainingVectorBufferDataV1::U32(values) => {
            values.len() as u64 * 4
        }
        TrainingVectorBufferDataV1::Bytes(values) => values.len() as u64,
    }
}

fn receipt_wire(receipt: &TrainReceiptV1) -> ReceiptWire {
    ReceiptWire {
        operation: receipt.operation.clone(),
        execution: execution_name(receipt.execution).to_owned(),
        dtype: dtype_name(receipt.dtype),
        input_digest: hex_digest(receipt.input_digest),
        output_digest: hex_digest(receipt.output_digest),
        peak_resident_bytes: receipt.peak_resident_bytes,
        scratch_bytes: receipt.scratch_bytes,
        host_transfers: receipt.host_transfers,
        device_resident: receipt.device_resident,
    }
}

fn canonical_bytes(wire: &BundleWire) -> Result<Vec<u8>, TrainingReceiptBundleError> {
    let mut bytes = serde_json::to_vec_pretty(wire)
        .map_err(|error| TrainingReceiptBundleError::Json(error.to_string()))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn receipt_error(case_id: &str, field: &str) -> TrainingReceiptBundleError {
    TrainingReceiptBundleError::Receipt {
        case_id: case_id.to_owned(),
        field: field.to_owned(),
    }
}

fn limits_wire(limits: TrainLimitsV1) -> LimitsWire {
    LimitsWire {
        max_rank: limits.max_rank,
        max_elements: limits.max_elements,
        max_bytes: limits.max_bytes,
    }
}

fn execution_name(execution: TrainExecutionV1) -> &'static str {
    match execution {
        TrainExecutionV1::Forward => "forward",
        TrainExecutionV1::Vjp => "vjp",
        TrainExecutionV1::Step => "step",
        TrainExecutionV1::Checkpoint => "checkpoint",
        TrainExecutionV1::Resume => "resume",
        TrainExecutionV1::Export => "export",
        TrainExecutionV1::Reload => "reload",
    }
}

fn parse_execution(value: &str) -> Option<TrainExecutionV1> {
    match value {
        "forward" => Some(TrainExecutionV1::Forward),
        "vjp" => Some(TrainExecutionV1::Vjp),
        "step" => Some(TrainExecutionV1::Step),
        "checkpoint" => Some(TrainExecutionV1::Checkpoint),
        "resume" => Some(TrainExecutionV1::Resume),
        "export" => Some(TrainExecutionV1::Export),
        "reload" => Some(TrainExecutionV1::Reload),
        _ => None,
    }
}

fn dtype_name(dtype: TrainDTypeV1) -> String {
    match dtype {
        TrainDTypeV1::F32 => "f32",
        TrainDTypeV1::U32 => "u32",
        TrainDTypeV1::Bytes => "bytes",
    }
    .to_owned()
}

fn parse_dtype(value: &str) -> Option<TrainDTypeV1> {
    match value {
        "f32" => Some(TrainDTypeV1::F32),
        "u32" => Some(TrainDTypeV1::U32),
        "bytes" => Some(TrainDTypeV1::Bytes),
        _ => None,
    }
}

fn hex_digest(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn parse_digest(value: &str) -> Result<[u8; 32], TrainingReceiptBundleError> {
    if value.len() != 64 {
        return Err(TrainingReceiptBundleError::Capabilities(
            "digest".to_owned(),
        ));
    }
    let mut output = [0_u8; 32];
    for (index, pair) in value.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        output[index] = (hex_nibble(pair[0])? << 4) | hex_nibble(pair[1])?;
    }
    Ok(output)
}

fn hex_nibble(value: u8) -> Result<u8, TrainingReceiptBundleError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(TrainingReceiptBundleError::Capabilities(
            "digest".to_owned(),
        )),
    }
}

fn escape_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}
