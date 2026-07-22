//! Bounded binary-batch Python ingestion for native grouped-curvature evidence.

use pyo3::{
    create_exception,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
};
use tritium_quantize::{
    CurvatureSourceId, Qwen35TensorRole, Qwen35TensorScope, SaltV2Curvature,
    SaltV2KroneckerEvidenceBuildError, SaltV2KroneckerEvidenceBuilder,
    SaltV2KroneckerEvidenceError, SaltV2KroneckerEvidenceSpec,
};
use tritium_salt::{
    Qwen36AdmissionError, Qwen36AdmittedSource, Qwen36PtqDriverError,
    Qwen36PtqEvidenceCaptureReceipt, Qwen36PtqEvidenceCaptureSession, Qwen36PtqEvidenceCaptureTask,
    Qwen36PtqEvidenceDirectory, Qwen36TensorWorkError, TensorWorkError,
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

/// One missing source-bound tensor in the canonical pinned-Qwen catalog.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct Qwen36KroneckerCaptureTask {
    tensor_index: u64,
    tensor_name: String,
    rows: usize,
    columns: usize,
    scope: String,
    role: String,
    source_model_digest: String,
    activation_cache_digest: String,
    token_stream_digest: String,
    curvature: String,
    damping: f64,
}

impl From<Qwen36PtqEvidenceCaptureTask> for Qwen36KroneckerCaptureTask {
    fn from(task: Qwen36PtqEvidenceCaptureTask) -> Self {
        let source = task.source_id();
        Self {
            tensor_index: task.tensor_index(),
            tensor_name: task.tensor_name().to_owned(),
            rows: task.rows(),
            columns: task.columns(),
            scope: scope_label(task.scope()).to_owned(),
            role: role_label(task.role()).to_owned(),
            source_model_digest: encode_digest(&source.source_model_digest()),
            activation_cache_digest: encode_digest(&source.activation_cache_digest()),
            token_stream_digest: encode_digest(&source.token_stream_digest()),
            curvature: curvature_label(task.curvature()).to_owned(),
            damping: task.damping(),
        }
    }
}

#[pymethods]
impl Qwen36KroneckerCaptureTask {
    #[getter]
    const fn tensor_index(&self) -> u64 {
        self.tensor_index
    }

    #[getter]
    fn tensor_name(&self) -> &str {
        &self.tensor_name
    }

    #[getter]
    const fn rows(&self) -> usize {
        self.rows
    }

    #[getter]
    const fn columns(&self) -> usize {
        self.columns
    }

    #[getter]
    fn scope(&self) -> &str {
        &self.scope
    }

    #[getter]
    fn role(&self) -> &str {
        &self.role
    }

    #[getter]
    fn source_model_digest(&self) -> &str {
        &self.source_model_digest
    }

    #[getter]
    fn activation_cache_digest(&self) -> &str {
        &self.activation_cache_digest
    }

    #[getter]
    fn token_stream_digest(&self) -> &str {
        &self.token_stream_digest
    }

    #[getter]
    fn curvature(&self) -> &str {
        &self.curvature
    }

    #[getter]
    const fn damping(&self) -> f64 {
        self.damping
    }

    fn __repr__(&self) -> String {
        format!(
            "Qwen36KroneckerCaptureTask(tensor_index={}, tensor_name='{}', rows={}, columns={})",
            self.tensor_index, self.tensor_name, self.rows, self.columns
        )
    }
}

/// Final ordered identity and resume accounting for the pinned-Qwen catalog.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct Qwen36KroneckerCaptureReceipt {
    evidence_set_digest: String,
    source_model_digest: String,
    activation_cache_digest: String,
    token_stream_digest: String,
    curvature: String,
    damping: f64,
    records: u64,
    produced: u64,
    reused: u64,
}

impl From<Qwen36PtqEvidenceCaptureReceipt> for Qwen36KroneckerCaptureReceipt {
    fn from(receipt: Qwen36PtqEvidenceCaptureReceipt) -> Self {
        let source = receipt.source_id();
        Self {
            evidence_set_digest: encode_digest(receipt.evidence_set_digest()),
            source_model_digest: encode_digest(&source.source_model_digest()),
            activation_cache_digest: encode_digest(&source.activation_cache_digest()),
            token_stream_digest: encode_digest(&source.token_stream_digest()),
            curvature: curvature_label(receipt.curvature()).to_owned(),
            damping: receipt.damping(),
            records: receipt.records(),
            produced: receipt.produced(),
            reused: receipt.reused(),
        }
    }
}

#[pymethods]
impl Qwen36KroneckerCaptureReceipt {
    #[getter]
    fn evidence_set_digest(&self) -> &str {
        &self.evidence_set_digest
    }

    #[getter]
    fn source_model_digest(&self) -> &str {
        &self.source_model_digest
    }

    #[getter]
    fn activation_cache_digest(&self) -> &str {
        &self.activation_cache_digest
    }

    #[getter]
    fn token_stream_digest(&self) -> &str {
        &self.token_stream_digest
    }

    #[getter]
    fn curvature(&self) -> &str {
        &self.curvature
    }

    #[getter]
    const fn damping(&self) -> f64 {
        self.damping
    }

    #[getter]
    const fn records(&self) -> u64 {
        self.records
    }

    #[getter]
    const fn produced(&self) -> u64 {
        self.produced
    }

    #[getter]
    const fn reused(&self) -> u64 {
        self.reused
    }

    fn __repr__(&self) -> String {
        format!(
            "Qwen36KroneckerCaptureReceipt(records={}, produced={}, reused={}, evidence_set_digest='{}')",
            self.records, self.produced, self.reused, self.evidence_set_digest
        )
    }
}

/// Stateful Python boundary for the native canonical pinned-Qwen scheduler.
#[pyclass(module = "tritium._tritium")]
pub(crate) struct Qwen36KroneckerCaptureSession {
    session: Qwen36PtqEvidenceCaptureSession,
}

#[pymethods]
impl Qwen36KroneckerCaptureSession {
    #[new]
    #[pyo3(signature = (
        model_dir,
        declared_revision,
        work_dir,
        evidence_dir,
        curvature,
        activation_cache_digest,
        token_stream_digest,
        damping,
        *,
        max_evidence_bytes = 67_108_864
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        model_dir: &str,
        declared_revision: &str,
        work_dir: &str,
        evidence_dir: &str,
        curvature: &str,
        activation_cache_digest: &str,
        token_stream_digest: &str,
        damping: f64,
        max_evidence_bytes: u64,
    ) -> PyResult<Self> {
        for (field, value) in [
            ("model_dir", model_dir),
            ("declared_revision", declared_revision),
            ("work_dir", work_dir),
            ("evidence_dir", evidence_dir),
        ] {
            if value.is_empty() {
                return Err(contract_error(format!("{field} must not be empty")));
            }
        }
        if declared_revision != tritium_nn::QWEN36_27B_REVISION {
            return Err(contract_error(format!(
                "declared_revision must equal the pinned Qwen3.6 revision {}",
                tritium_nn::QWEN36_27B_REVISION
            )));
        }
        if max_evidence_bytes == 0 {
            return Err(contract_error("max_evidence_bytes must be positive"));
        }
        if !damping.is_finite() || damping < 0.0 {
            return Err(contract_error(
                "damping must be finite and greater than or equal to zero",
            ));
        }
        let curvature = parse_curvature(curvature)?;
        let activation_cache_digest =
            parse_digest("activation_cache_digest", activation_cache_digest)?;
        let token_stream_digest = parse_digest("token_stream_digest", token_stream_digest)?;
        let model_dir = model_dir.to_owned();
        let declared_revision = declared_revision.to_owned();
        let work_dir = work_dir.to_owned();
        let evidence_dir = evidence_dir.to_owned();
        let session = py.detach(move || {
            let admitted = Qwen36AdmittedSource::open(
                model_dir.as_ref(),
                &declared_revision,
                work_dir.as_ref(),
            )
            .map_err(admission_error)?;
            let evidence =
                Qwen36PtqEvidenceDirectory::create_bounded(evidence_dir, max_evidence_bytes)
                    .map_err(directory_error)?;
            Qwen36PtqEvidenceCaptureSession::open(
                &admitted,
                evidence,
                curvature,
                activation_cache_digest,
                token_stream_digest,
                damping,
            )
            .map_err(directory_error)
        })?;
        Ok(Self { session })
    }

    /// Return the next missing canonical tensor, idempotently while pending.
    fn next_request(&mut self, py: Python<'_>) -> PyResult<Option<Qwen36KroneckerCaptureTask>> {
        py.detach(|| self.session.next_request())
            .map(|task| task.map(Into::into))
            .map_err(directory_error)
    }

    /// Advance only after the current exact record strictly reopens.
    fn accept_current(&mut self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.session.accept_current())
            .map_err(directory_error)
    }

    /// Return total, newly accepted, and reused records for this invocation.
    #[getter]
    fn counts(&self) -> (u64, u64, u64) {
        self.session.counts()
    }

    /// Seal only after every canonical record freshly validates.
    fn finish(&mut self, py: Python<'_>) -> PyResult<Option<Qwen36KroneckerCaptureReceipt>> {
        py.detach(|| self.session.finish())
            .map(|receipt| receipt.map(Into::into))
            .map_err(directory_error)
    }

    fn __repr__(&self) -> String {
        let (records, produced, reused) = self.session.counts();
        format!(
            "Qwen36KroneckerCaptureSession(records={records}, produced={produced}, reused={reused})"
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
    indexed_output: bool,
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
        max_batch_bytes = DEFAULT_MAX_BATCH_BYTES,
        indexed_output = false
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
        indexed_output: bool,
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
        if indexed_output && matches!(curvature, SaltV2Curvature::InputHessian) {
            return Err(contract_error(
                "indexed output factors are unsupported for input-hessian curvature",
            ));
        }
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
        let builder = if indexed_output {
            directory.create_indexed_output_builder(spec)
        } else {
            directory.create_builder(spec)
        }
        .map_err(driver_build_error)?;
        Ok(Self {
            directory,
            builder: Some(builder),
            tensor_index,
            max_batch_bytes,
            indexed_output,
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
        if self.indexed_output {
            return Err(contract_error(
                "Kronecker builder output-factor encoding does not match append operation",
            ));
        }
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

    /// Atomically append one sparse indexed output-factor batch.
    #[pyo3(signature = (
        activations_f32le,
        output_indices_u64le,
        samples,
        *,
        output_factors_f32le = None,
        token_weights_f64le = None,
        token_mask_u8 = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn append_indexed_batch(
        &mut self,
        py: Python<'_>,
        activations_f32le: &[u8],
        output_indices_u64le: &[u8],
        samples: usize,
        output_factors_f32le: Option<&[u8]>,
        token_weights_f64le: Option<&[u8]>,
        token_mask_u8: Option<&[u8]>,
    ) -> PyResult<(usize, usize)> {
        if !self.indexed_output {
            return Err(contract_error(
                "Kronecker builder output-factor encoding does not match append operation",
            ));
        }
        let builder = self
            .builder
            .as_ref()
            .ok_or_else(|| KroneckerStateError::new_err("evidence builder is already finished"))?;
        let columns = builder.spec().columns();
        let expected_activations = checked_elements(samples, columns, "activations")?;
        let logical_factor_bytes = samples
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| contract_error("indexed output factor byte count overflows"))?;
        let total_bytes = activations_f32le
            .len()
            .checked_add(output_indices_u64le.len())
            .and_then(|bytes| {
                bytes.checked_add(output_factors_f32le.map_or(logical_factor_bytes, <[u8]>::len))
            })
            .and_then(|bytes| bytes.checked_add(token_weights_f64le.map_or(0, <[u8]>::len)))
            .and_then(|bytes| bytes.checked_add(token_mask_u8.map_or(0, <[u8]>::len)))
            .filter(|bytes| *bytes <= self.max_batch_bytes)
            .ok_or_else(|| contract_error("batch exceeds max_batch_bytes"))?;
        debug_assert!(total_bytes <= self.max_batch_bytes);
        let activations = decode_f32le("activations", activations_f32le, expected_activations)?;
        let encoded_indices = decode_u64le("output indices", output_indices_u64le, samples)?;
        let mut output_indices = Vec::new();
        output_indices
            .try_reserve_exact(samples)
            .map_err(|_| resource_error("allocate output indices failed"))?;
        for index in encoded_indices {
            output_indices.push(
                usize::try_from(index)
                    .map_err(|_| contract_error("output index does not fit this platform"))?,
            );
        }
        let output_factors = match output_factors_f32le {
            Some(bytes) => decode_f32le("indexed output factors", bytes, samples)?,
            None => {
                let mut factors = Vec::new();
                factors
                    .try_reserve_exact(samples)
                    .map_err(|_| resource_error("allocate indexed output factors failed"))?;
                factors.resize(samples, 1.0);
                factors
            }
        };
        let token_weights = token_weights_f64le
            .map(|bytes| decode_f64le("token weights", bytes, samples))
            .transpose()?;
        let token_mask = token_mask_u8
            .map(|bytes| decode_mask(bytes, samples))
            .transpose()?;
        let builder = self.builder.as_mut().expect("checked active builder");
        py.detach(move || {
            builder.accumulate_indexed_output_batch(
                &activations,
                &output_indices,
                &output_factors,
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
            "KroneckerEvidenceBuilder(tensor_index={}, active={}, indexed_output={}, max_batch_bytes={})",
            self.tensor_index,
            self.builder.is_some(),
            self.indexed_output,
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
        Qwen36PtqDriverError::EvidenceMismatch { .. } => {
            KroneckerConflictError::new_err(error.to_string())
        }
        Qwen36PtqDriverError::EvidenceBuild { .. } => driver_build_error(error),
        Qwen36PtqDriverError::Evidence {
            source: SaltV2KroneckerEvidenceError::AllocationFailed,
            ..
        } => resource_error(error),
        Qwen36PtqDriverError::Evidence {
            source: SaltV2KroneckerEvidenceError::Io { .. },
            ..
        } => KroneckerPublicationError::new_err(error.to_string()),
        Qwen36PtqDriverError::Evidence { .. } => contract_error(error),
        Qwen36PtqDriverError::Workspace(source) if workspace_resource_failure(source) => {
            resource_error(error)
        }
        Qwen36PtqDriverError::Workspace(source) if !workspace_retryable_io(source) => {
            contract_error(error)
        }
        _ => KroneckerPublicationError::new_err(error.to_string()),
    }
}

fn admission_error(error: Qwen36AdmissionError) -> PyErr {
    match error {
        Qwen36AdmissionError::Io { .. } | Qwen36AdmissionError::AlreadyLocked => {
            KroneckerPublicationError::new_err(error.to_string())
        }
        Qwen36AdmissionError::ExistingProofMismatch => {
            KroneckerConflictError::new_err(error.to_string())
        }
        _ => contract_error(error),
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

fn decode_u64le(field: &'static str, bytes: &[u8], expected: usize) -> PyResult<Vec<u64>> {
    decode_le(field, bytes, expected, u64::from_le_bytes)
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

const fn curvature_label(value: SaltV2Curvature) -> &'static str {
    match value {
        SaltV2Curvature::InputHessian => "input-hessian",
        SaltV2Curvature::GuidedFisher => "guided-fisher",
        SaltV2Curvature::ForwardKlKronecker => "forward-kl-kronecker",
        SaltV2Curvature::DiagonalFisher => "diagonal-fisher",
    }
}

const fn scope_label(value: Qwen35TensorScope) -> &'static str {
    match value {
        Qwen35TensorScope::Language => "language",
        Qwen35TensorScope::MtpDrafter => "mtp-drafter",
        Qwen35TensorScope::DeferredVision => "deferred-vision",
    }
}

const fn role_label(value: Qwen35TensorRole) -> &'static str {
    match value {
        Qwen35TensorRole::TokenEmbedding => "token-embedding",
        Qwen35TensorRole::OutputHead => "output-head",
        Qwen35TensorRole::Normalization => "normalization",
        Qwen35TensorRole::MlpProjection => "mlp-projection",
        Qwen35TensorRole::FullAttentionProjection => "full-attention-projection",
        Qwen35TensorRole::DeltaNetProjection => "delta-net-projection",
        Qwen35TensorRole::DeltaNetState => "delta-net-state",
        Qwen35TensorRole::DeltaNetConvolution => "delta-net-convolution",
        Qwen35TensorRole::MtpFusionProjection => "mtp-fusion-projection",
        Qwen35TensorRole::VisionAttentionProjection => "vision-attention-projection",
        Qwen35TensorRole::VisionMlpProjection => "vision-mlp-projection",
        Qwen35TensorRole::VisionPatchEmbedding => "vision-patch-embedding",
        Qwen35TensorRole::VisionPositionalEmbedding => "vision-positional-embedding",
        Qwen35TensorRole::VisionMergerProjection => "vision-merger-projection",
        Qwen35TensorRole::Bias => "bias",
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
