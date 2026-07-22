//! Bounded binary-batch Python ingestion for native grouped-curvature evidence.

use pyo3::{
    create_exception,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use tritium_quantize::{
    CurvatureSourceId, SaltV2Curvature, SaltV2KroneckerEvidenceBuildError,
    SaltV2KroneckerEvidenceBuilder, SaltV2KroneckerEvidenceError, SaltV2KroneckerEvidenceSpec,
};
use tritium_salt::{
    Qwen36PtqDriverError, Qwen36PtqEvidenceDirectory, Qwen36TensorWorkError, TensorWorkError,
};

const DEFAULT_MAX_BATCH_BYTES: usize = 256 * 1024 * 1024;

create_exception!(
    tritium._tritium,
    KroneckerContractError,
    PyValueError,
    "A terminal grouped-evidence API or data contract violation."
);
create_exception!(
    tritium._tritium,
    KroneckerResourceError,
    PyRuntimeError,
    "A bounded grouped-evidence allocation or resource failure."
);
create_exception!(
    tritium._tritium,
    KroneckerPublicationError,
    PyRuntimeError,
    "A grouped-evidence storage or publication failure that may be retried."
);
create_exception!(
    tritium._tritium,
    KroneckerConflictError,
    PyRuntimeError,
    "A terminal immutable grouped-evidence publication conflict."
);
create_exception!(
    tritium._tritium,
    KroneckerStateError,
    PyRuntimeError,
    "An operation was attempted on a terminal grouped-evidence builder."
);

/// Exact receipt for one atomically published grouped-curvature record.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct KroneckerEvidenceReceipt {
    tensor_index: u64,
    record_digest: String,
    bytes: u64,
}

#[pymethods]
impl KroneckerEvidenceReceipt {
    /// Canonical additive-tensor ordinal.
    #[getter]
    const fn tensor_index(&self) -> u64 {
        self.tensor_index
    }

    /// Hex digest of the exact canonical S2KF record.
    #[getter]
    fn record_digest(&self) -> &str {
        &self.record_digest
    }

    /// Exact published bytes including checksum.
    #[getter]
    const fn bytes(&self) -> u64 {
        self.bytes
    }

    fn __repr__(&self) -> String {
        format!(
            "KroneckerEvidenceReceipt(tensor_index={}, bytes={}, record_digest='{}')",
            self.tensor_index, self.bytes, self.record_digest
        )
    }
}

/// One bounded native producer for a canonical grouped-curvature record.
///
/// Binary inputs use canonical little-endian IEEE-754 payloads so PyTorch can
/// pass `tensor.contiguous().cpu().numpy().tobytes()` without constructing
/// millions of Python scalar objects.
#[pyclass(module = "tritium._tritium")]
pub(crate) struct KroneckerEvidenceBuilder {
    directory: Qwen36PtqEvidenceDirectory,
    builder: Option<SaltV2KroneckerEvidenceBuilder>,
    tensor_index: u64,
    max_batch_bytes: usize,
}

#[pymethods]
impl KroneckerEvidenceBuilder {
    /// Create a source-bound builder under exact record and batch byte ceilings.
    #[new]
    #[pyo3(signature = (
        evidence_dir,
        tensor_index,
        tensor_name,
        rows,
        columns,
        curvature,
        source_model_digest,
        activation_cache_digest,
        token_stream_digest,
        damping,
        *,
        max_evidence_bytes = 67_108_864,
        max_batch_bytes = DEFAULT_MAX_BATCH_BYTES
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        evidence_dir: &str,
        tensor_index: u64,
        tensor_name: &str,
        rows: usize,
        columns: usize,
        curvature: &str,
        source_model_digest: &str,
        activation_cache_digest: &str,
        token_stream_digest: &str,
        damping: f64,
        max_evidence_bytes: u64,
        max_batch_bytes: usize,
    ) -> PyResult<Self> {
        if max_batch_bytes == 0 {
            return Err(contract_error("max_batch_bytes must be positive"));
        }
        if tensor_index >= 1_000_000 {
            return Err(contract_error(
                "tensor_index must fit the six-digit Qwen evidence namespace",
            ));
        }
        let curvature = parse_curvature(curvature)?;
        let source_id = CurvatureSourceId::new(
            parse_digest("source_model_digest", source_model_digest)?,
            parse_digest("activation_cache_digest", activation_cache_digest)?,
            parse_digest("token_stream_digest", token_stream_digest)?,
        )
        .map_err(contract_error)?;
        let spec = SaltV2KroneckerEvidenceSpec::new_bounded(
            curvature,
            source_id,
            tensor_index,
            tensor_name,
            rows,
            columns,
            damping,
            max_evidence_bytes,
        )
        .map_err(build_error)?;
        let directory =
            Qwen36PtqEvidenceDirectory::create_bounded(evidence_dir, max_evidence_bytes)
                .map_err(directory_error)?;
        let builder = directory.create_builder(spec).map_err(driver_build_error)?;
        Ok(Self {
            directory,
            builder: Some(builder),
            tensor_index,
            max_batch_bytes,
        })
    }

    /// Atomically append one row-major binary batch.
    #[pyo3(signature = (
        activations_f32le,
        samples,
        *,
        output_factors_f32le = None,
        token_weights_f64le = None,
        token_mask_u8 = None
    ))]
    fn append_batch(
        &mut self,
        py: Python<'_>,
        activations_f32le: &[u8],
        samples: usize,
        output_factors_f32le: Option<&[u8]>,
        token_weights_f64le: Option<&[u8]>,
        token_mask_u8: Option<&[u8]>,
    ) -> PyResult<(usize, usize)> {
        let builder = self
            .builder
            .as_ref()
            .ok_or_else(|| KroneckerStateError::new_err("evidence builder is already finished"))?;
        let columns = builder.spec().columns();
        let rows = builder.spec().rows();
        let expected_activations = checked_elements(samples, columns, "activations")?;
        let expected_factors = output_factors_f32le
            .map(|_| checked_elements(samples, rows, "output factors"))
            .transpose()?;
        let total_bytes = activations_f32le
            .len()
            .checked_add(output_factors_f32le.map_or(0, <[u8]>::len))
            .and_then(|bytes| bytes.checked_add(token_weights_f64le.map_or(0, <[u8]>::len)))
            .and_then(|bytes| bytes.checked_add(token_mask_u8.map_or(0, <[u8]>::len)))
            .filter(|bytes| *bytes <= self.max_batch_bytes)
            .ok_or_else(|| contract_error("batch exceeds max_batch_bytes"))?;
        debug_assert!(total_bytes <= self.max_batch_bytes);
        let activations = decode_f32le("activations", activations_f32le, expected_activations)?;
        let output_factors = output_factors_f32le
            .map(|bytes| decode_f32le("output factors", bytes, expected_factors.unwrap_or(0)))
            .transpose()?;
        let token_weights = token_weights_f64le
            .map(|bytes| decode_f64le("token weights", bytes, samples))
            .transpose()?;
        let token_mask = token_mask_u8
            .map(|bytes| decode_mask(bytes, samples))
            .transpose()?;
        let builder = self.builder.as_mut().expect("checked active builder");
        py.detach(move || {
            builder.accumulate_batch(
                &activations,
                output_factors.as_deref(),
                samples,
                token_weights.as_deref(),
                token_mask.as_deref(),
            )
        })
        .map_err(build_error)?;
        let residency = self
            .builder
            .as_ref()
            .expect("builder remains active after successful append")
            .residency();
        Ok((residency.input_segments(), residency.output_segments()))
    }

    /// Finalize and atomically publish the canonical record.
    fn finish(&mut self, py: Python<'_>) -> PyResult<KroneckerEvidenceReceipt> {
        let builder = self
            .builder
            .as_ref()
            .ok_or_else(|| KroneckerStateError::new_err("evidence builder is already finished"))?;
        let record = py.detach(move || builder.finish()).map_err(build_error)?;
        let directory = self.directory.clone();
        let receipt = match py.detach(move || directory.install(&record)) {
            Ok(receipt) => receipt,
            Err(error) => {
                if terminal_publication_error(&error) {
                    self.builder = None;
                }
                return Err(publication_error(error));
            }
        };
        self.builder = None;
        Ok(KroneckerEvidenceReceipt {
            tensor_index: self.tensor_index,
            record_digest: encode_digest(&receipt.record_digest()),
            bytes: receipt.bytes(),
        })
    }

    /// Drop unpublished accumulator state and make this builder terminal.
    fn abort(&mut self) {
        self.builder = None;
    }

    /// Whether append/finalize operations are still admitted.
    #[getter]
    fn active(&self) -> bool {
        self.builder.is_some()
    }

    fn __repr__(&self) -> String {
        format!(
            "KroneckerEvidenceBuilder(tensor_index={}, active={}, max_batch_bytes={})",
            self.tensor_index,
            self.builder.is_some(),
            self.max_batch_bytes
        )
    }
}

pub(crate) fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "KroneckerContractError",
        py.get_type::<KroneckerContractError>(),
    )?;
    m.add(
        "KroneckerResourceError",
        py.get_type::<KroneckerResourceError>(),
    )?;
    m.add(
        "KroneckerPublicationError",
        py.get_type::<KroneckerPublicationError>(),
    )?;
    m.add(
        "KroneckerConflictError",
        py.get_type::<KroneckerConflictError>(),
    )?;
    m.add("KroneckerStateError", py.get_type::<KroneckerStateError>())?;
    Ok(())
}

fn contract_error(error: impl core::fmt::Display) -> PyErr {
    KroneckerContractError::new_err(error.to_string())
}

fn resource_error(error: impl core::fmt::Display) -> PyErr {
    KroneckerResourceError::new_err(error.to_string())
}

fn build_error(error: SaltV2KroneckerEvidenceBuildError) -> PyErr {
    match &error {
        SaltV2KroneckerEvidenceBuildError::AllocationFailed
        | SaltV2KroneckerEvidenceBuildError::Evidence(
            SaltV2KroneckerEvidenceError::AllocationFailed,
        ) => resource_error(error),
        SaltV2KroneckerEvidenceBuildError::Evidence(SaltV2KroneckerEvidenceError::Io {
            ..
        }) => KroneckerPublicationError::new_err(error.to_string()),
        _ => contract_error(error),
    }
}

fn driver_build_error(error: Qwen36PtqDriverError) -> PyErr {
    match error {
        Qwen36PtqDriverError::EvidenceBuild { source, .. } => build_error(source),
        Qwen36PtqDriverError::AllocationFailed => resource_error(error),
        _ => contract_error(error),
    }
}

fn directory_error(error: Qwen36PtqDriverError) -> PyErr {
    match &error {
        Qwen36PtqDriverError::InvalidEvidencePath(_) => contract_error(error),
        Qwen36PtqDriverError::AllocationFailed => resource_error(error),
        Qwen36PtqDriverError::Workspace(source) if workspace_resource_failure(source) => {
            resource_error(error)
        }
        Qwen36PtqDriverError::Workspace(source) if !workspace_retryable_io(source) => {
            contract_error(error)
        }
        _ => KroneckerPublicationError::new_err(error.to_string()),
    }
}

fn publication_error(error: Qwen36PtqDriverError) -> PyErr {
    match &error {
        Qwen36PtqDriverError::EvidenceMismatch {
            field: "global tensor ordinal",
            ..
        } => contract_error(error),
        Qwen36PtqDriverError::EvidenceMismatch {
            field: "immutable record identity",
            ..
        } => KroneckerConflictError::new_err(error.to_string()),
        Qwen36PtqDriverError::AllocationFailed
        | Qwen36PtqDriverError::Evidence {
            source: SaltV2KroneckerEvidenceError::AllocationFailed,
            ..
        } => resource_error(error),
        Qwen36PtqDriverError::EvidenceBuild { .. } => driver_build_error(error),
        Qwen36PtqDriverError::Workspace(source) if workspace_resource_failure(source) => {
            resource_error(error)
        }
        Qwen36PtqDriverError::Workspace(source) if !workspace_retryable_io(source) => {
            contract_error(error)
        }
        Qwen36PtqDriverError::Evidence {
            source:
                SaltV2KroneckerEvidenceError::Malformed(_)
                | SaltV2KroneckerEvidenceError::WeightLengthMismatch { .. }
                | SaltV2KroneckerEvidenceError::SizeLimitExceeded { .. },
            ..
        } => contract_error(error),
        _ => KroneckerPublicationError::new_err(error.to_string()),
    }
}

fn terminal_publication_error(error: &Qwen36PtqDriverError) -> bool {
    match error {
        Qwen36PtqDriverError::InvalidEvidencePath(_)
        | Qwen36PtqDriverError::EvidenceMismatch { .. } => true,
        Qwen36PtqDriverError::Evidence { source, .. } => !matches!(
            source,
            SaltV2KroneckerEvidenceError::AllocationFailed
                | SaltV2KroneckerEvidenceError::Io { .. }
        ),
        Qwen36PtqDriverError::EvidenceBuild { source, .. } => !matches!(
            source,
            SaltV2KroneckerEvidenceBuildError::AllocationFailed
                | SaltV2KroneckerEvidenceBuildError::Evidence(
                    SaltV2KroneckerEvidenceError::AllocationFailed
                        | SaltV2KroneckerEvidenceError::Io { .. }
                )
        ),
        Qwen36PtqDriverError::Workspace(source) => {
            !workspace_resource_failure(source) && !workspace_retryable_io(source)
        }
        _ => false,
    }
}

fn workspace_resource_failure(error: &Qwen36TensorWorkError) -> bool {
    matches!(
        error,
        Qwen36TensorWorkError::AllocationFailed
            | Qwen36TensorWorkError::TensorStore(TensorWorkError::AllocationFailed)
    )
}

fn workspace_retryable_io(error: &Qwen36TensorWorkError) -> bool {
    matches!(
        error,
        Qwen36TensorWorkError::Io { .. }
            | Qwen36TensorWorkError::TensorStore(TensorWorkError::Io { .. })
    )
}

fn checked_elements(samples: usize, width: usize, field: &'static str) -> PyResult<usize> {
    samples
        .checked_mul(width)
        .ok_or_else(|| contract_error(format!("{field} element count overflows")))
}

fn decode_f32le(field: &'static str, bytes: &[u8], expected: usize) -> PyResult<Vec<f32>> {
    decode_le(field, bytes, expected, f32::from_le_bytes)
}

fn decode_f64le(field: &'static str, bytes: &[u8], expected: usize) -> PyResult<Vec<f64>> {
    decode_le(field, bytes, expected, f64::from_le_bytes)
}

fn decode_le<const N: usize, T>(
    field: &'static str,
    bytes: &[u8],
    expected: usize,
    decode: impl Fn([u8; N]) -> T,
) -> PyResult<Vec<T>> {
    let expected_bytes = expected
        .checked_mul(N)
        .ok_or_else(|| contract_error(format!("{field} byte count overflows")))?;
    if bytes.len() != expected_bytes {
        return Err(contract_error(format!(
            "{field} requires {expected_bytes} bytes, got {}",
            bytes.len()
        )));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(expected)
        .map_err(|_| resource_error(format!("allocate {field} failed")))?;
    for chunk in bytes.chunks_exact(N) {
        let mut value = [0_u8; N];
        value.copy_from_slice(chunk);
        values.push(decode(value));
    }
    Ok(values)
}

fn decode_mask(bytes: &[u8], expected: usize) -> PyResult<Vec<bool>> {
    if bytes.len() != expected || bytes.iter().any(|value| *value > 1) {
        return Err(contract_error(
            "token_mask_u8 must contain exactly samples bytes in {0,1}",
        ));
    }
    let mut mask = Vec::new();
    mask.try_reserve_exact(expected)
        .map_err(|_| resource_error("allocate token mask failed"))?;
    mask.extend(bytes.iter().map(|value| *value == 1));
    Ok(mask)
}

fn parse_curvature(value: &str) -> PyResult<SaltV2Curvature> {
    match value {
        "input-hessian" => Ok(SaltV2Curvature::InputHessian),
        "guided-fisher" => Ok(SaltV2Curvature::GuidedFisher),
        "forward-kl-kronecker" => Ok(SaltV2Curvature::ForwardKlKronecker),
        _ => Err(contract_error(
            "curvature must be input-hessian, guided-fisher, or forward-kl-kronecker",
        )),
    }
}

fn parse_digest(field: &'static str, value: &str) -> PyResult<[u8; 32]> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(contract_error(format!(
            "{field} must be exactly 64 hexadecimal characters"
        )));
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| contract_error(format!("invalid {field}")))?;
    }
    Ok(digest)
}

fn encode_digest(digest: &[u8; 32]) -> String {
    use core::fmt::Write as _;

    let mut output = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_decoders_are_exact_and_bounded() {
        let f32_bytes = [1.5_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat();
        assert_eq!(decode_f32le("values", &f32_bytes, 2).unwrap(), [1.5, -2.0]);
        assert!(decode_f32le("values", &f32_bytes, 3).is_err());
        assert_eq!(decode_mask(&[1, 0, 1], 3).unwrap(), [true, false, true]);
        assert!(decode_mask(&[2], 1).is_err());
    }

    #[test]
    fn digest_and_curvature_parsers_fail_closed() {
        assert_eq!(
            parse_digest("digest", &"ab".repeat(32)).unwrap(),
            [0xab; 32]
        );
        assert!(parse_digest("digest", &"ab".repeat(31)).is_err());
        assert_eq!(
            parse_curvature("forward-kl-kronecker").unwrap(),
            SaltV2Curvature::ForwardKlKronecker
        );
        assert!(parse_curvature("diagonal-fisher").is_err());
    }
}
