//! Bounded binary-batch Python ingestion for native grouped-curvature evidence.

use pyo3::{
    create_exception,
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    pybacked::PyBackedBytes,
};
use tritium_quantize::{
    CurvatureError, CurvatureSourceId, DensePsdMetric, JointFitConfig, JointFitMetric,
    Qwen35TensorRole, Qwen35TensorScope, RelayBasins, SaltV2Curvature, SaltV2KroneckerEvidence,
    SaltV2KroneckerEvidenceBuildError, SaltV2KroneckerEvidenceBuilder,
    SaltV2KroneckerEvidenceError, SaltV2KroneckerEvidenceSpec, ScalePrecision, fit_joint_ternary,
};
use tritium_salt::{
    Qwen36AdmissionError, Qwen36AdmittedSource, Qwen36PtqDriverError,
    Qwen36PtqEvidenceCaptureReceipt, Qwen36PtqEvidenceCaptureSession, Qwen36PtqEvidenceCaptureTask,
    Qwen36PtqEvidenceDirectory, Qwen36TensorWorkError, TensorWorkError,
};

type KroneckerJointFitResult = (
    usize,
    usize,
    Vec<Vec<f32>>,
    Vec<Vec<i8>>,
    Vec<f32>,
    f64,
    String,
);

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

    /// Return a bounded missing-task window without advancing acceptance.
    fn next_requests(
        &mut self,
        py: Python<'_>,
        max_tasks: usize,
    ) -> PyResult<Vec<Qwen36KroneckerCaptureTask>> {
        if max_tasks == 0 {
            return Err(contract_error("max_tasks must be positive"));
        }
        py.detach(|| self.session.next_requests(max_tasks))
            .map(|tasks| tasks.into_iter().map(Into::into).collect())
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

/// Fit every row/group of one canonical S2KF record with native joint ternary search.
///
/// This primitive consumes one already verified record and one row-major source
/// matrix (optionally one bounded output-row window). It returns
/// `(rows, columns, scales, trits, reconstruction, objective, record_digest)`,
/// with scales flattened as `plane -> rows * (columns / 128)` and trits as
/// `plane -> rows * columns`.
/// It performs no source-model admission or package publication; callers must
/// bind the returned record digest and solver settings into their own receipt.
#[pyfunction]
#[pyo3(signature = (
    weights_f32le,
    evidence_s2kf,
    *,
    planes = 3,
    max_iterations = 16,
    ridge = 1e-8,
    em_restarts = 4,
    ridge_condition_limit = 1e6,
    scale_precision = "f16",
    softened_relay = false,
    modulated_relay = false,
    row_start = 0,
    row_count = None
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_kronecker_ternary(
    py: Python<'_>,
    weights_f32le: &[u8],
    evidence_s2kf: &[u8],
    planes: usize,
    max_iterations: usize,
    ridge: f64,
    em_restarts: usize,
    ridge_condition_limit: f64,
    scale_precision: &str,
    softened_relay: bool,
    modulated_relay: bool,
    row_start: usize,
    row_count: Option<usize>,
) -> PyResult<KroneckerJointFitResult> {
    let scale_precision = match scale_precision {
        "f32" => ScalePrecision::F32,
        "f16" => ScalePrecision::F16,
        _ => return Err(contract_error("scale_precision must be 'f32' or 'f16'")),
    };
    let weights = decode_f32le("weights", weights_f32le, weights_f32le.len() / 4)?;
    let evidence =
        SaltV2KroneckerEvidence::from_canonical_bytes(evidence_s2kf).map_err(contract_error)?;
    let rows = evidence.rows();
    let columns = evidence.columns();
    if columns == 0 || !columns.is_multiple_of(128) {
        return Err(contract_error(
            "S2KF fitting requires columns divisible by native group size 128",
        ));
    }
    if row_start > rows {
        return Err(contract_error("row_start exceeds S2KF output rows"));
    }
    let fit_rows = row_count.unwrap_or(rows - row_start);
    if fit_rows == 0 || fit_rows > rows - row_start {
        return Err(contract_error(
            "row_count must select at least one row within S2KF output rows",
        ));
    }
    let expected_weights = fit_rows
        .checked_mul(columns)
        .ok_or_else(|| contract_error("weight geometry overflows platform usize"))?;
    if weights.len() != expected_weights {
        return Err(contract_error(format!(
            "weights requires {expected_weights} f32 values, got {}",
            weights.len()
        )));
    }
    let metric_groups = evidence.input_groups().to_vec();
    let output_weights = evidence.output_weights()[row_start..row_start + fit_rows].to_vec();
    let damping = evidence.damping();
    let record_digest = encode_digest(&evidence.record_digest());
    let group_count = columns / 128;
    let result = py
        .detach(move || {
            let mut all_scales = (0..planes).map(|_| Vec::new()).collect::<Vec<_>>();
            let mut all_trits = (0..planes).map(|_| Vec::new()).collect::<Vec<_>>();
            for values in &mut all_scales {
                values
                    .try_reserve_exact(fit_rows.checked_mul(group_count).ok_or_else(|| {
                        "scale output geometry overflows platform usize".to_owned()
                    })?)
                    .map_err(|_| "scale output allocation failed".to_owned())?;
            }
            for values in &mut all_trits {
                values
                    .try_reserve_exact(expected_weights)
                    .map_err(|_| "trit output allocation failed".to_owned())?;
            }
            let mut reconstruction = Vec::new();
            reconstruction
                .try_reserve_exact(expected_weights)
                .map_err(|_| "reconstruction allocation failed".to_owned())?;
            let mut objective = 0.0_f64;
            for (row, output_weight) in output_weights.iter().enumerate() {
                for (group, metric_group) in metric_groups.iter().enumerate() {
                    let mut metric_values = Vec::with_capacity(128 * 128);
                    for (index, value) in metric_group.as_slice().iter().enumerate() {
                        let diagonal = if index / 128 == index % 128 {
                            damping
                        } else {
                            0.0
                        };
                        metric_values.push(*value * output_weight + diagonal);
                    }
                    let metric = DensePsdMetric::new(128, &metric_values)
                        .map_err(|error| error.to_string())?;
                    let start = row * columns + group * 128;
                    let fit = fit_joint_ternary(
                        &weights[start..start + 128],
                        JointFitMetric::Dense(&metric),
                        JointFitConfig {
                            planes,
                            max_iterations,
                            ridge,
                            em_restarts,
                            ridge_condition_limit,
                            scale_precision,
                            relay_basins: RelayBasins {
                                softened: softened_relay,
                                modulated: modulated_relay,
                            },
                        },
                    )
                    .map_err(|error| error.to_string())?;
                    objective += fit.objective;
                    reconstruction.extend_from_slice(&fit.reconstruction);
                    for plane in 0..planes {
                        all_scales[plane].push(fit.scales[plane]);
                        all_trits[plane].extend_from_slice(&fit.trits[plane]);
                    }
                }
            }
            Ok::<_, String>((all_scales, all_trits, reconstruction, objective))
        })
        .map_err(contract_error)?;
    Ok((
        fit_rows,
        columns,
        result.0,
        result.1,
        result.2,
        result.3,
        record_digest,
    ))
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

/// Plain-Rust shared-forward fan-out over per-tensor evidence builders.
///
/// One validated activation batch per input stream feeds every member builder
/// with exactly the bytes and global sample ordinals a standalone builder
/// would receive: evidence identity is keyed on sample ordinals, never batch
/// or orchestration boundaries, so shared-forward records stay byte-identical
/// to per-tensor records (ADR 0035 WS-A2).
#[derive(Debug)]
struct SharedForwardGroupCore {
    members: Vec<SharedForwardMember>,
    stream_columns: Vec<usize>,
}

#[derive(Debug)]
struct SharedForwardMember {
    builder: SaltV2KroneckerEvidenceBuilder,
    input_stream: usize,
}

/// Group append failure split by whether member ordinals may have diverged.
#[derive(Debug)]
enum SharedForwardAppendError {
    /// Preflight rejected the batch; no member builder was touched.
    Rejected(SaltV2KroneckerEvidenceBuildError),
    /// A member mutation failed after an earlier member advanced; ordinals
    /// across members are inconsistent and the group must become terminal.
    Poisoned(SaltV2KroneckerEvidenceBuildError),
}

impl SharedForwardAppendError {
    fn into_build_error(self) -> SaltV2KroneckerEvidenceBuildError {
        match self {
            Self::Rejected(error) | Self::Poisoned(error) => error,
        }
    }
}

impl SharedForwardGroupCore {
    /// Validate one dense shared-forward group over contiguous input streams.
    fn new(
        members: Vec<(SaltV2KroneckerEvidenceBuilder, usize)>,
    ) -> Result<Self, SaltV2KroneckerEvidenceBuildError> {
        if members.is_empty() {
            return Err(SaltV2KroneckerEvidenceBuildError::Malformed(
                "shared-forward member count",
            ));
        }
        // Streams must be densely indexed: the zero-coverage check below already
        // rejects any gap, so an index >= member count can never pass — reject it
        // BEFORE it sizes the allocation (a hostile index would otherwise drive an
        // infallible multi-TB vec!).
        if members.iter().any(|(_, stream)| *stream >= members.len()) {
            return Err(SaltV2KroneckerEvidenceBuildError::Malformed(
                "shared-forward stream index",
            ));
        }
        let stream_count = members
            .iter()
            .map(|(_, stream)| *stream)
            .max()
            .map_or(0, |last| last + 1);
        let mut stream_columns = vec![0_usize; stream_count];
        for (builder, stream) in &members {
            let columns = builder.spec().columns();
            match stream_columns[*stream] {
                0 => stream_columns[*stream] = columns,
                existing if existing != columns => {
                    return Err(SaltV2KroneckerEvidenceBuildError::Malformed(
                        "shared-forward stream columns",
                    ));
                }
                _ => {}
            }
        }
        if stream_columns.contains(&0) {
            return Err(SaltV2KroneckerEvidenceBuildError::Malformed(
                "shared-forward stream coverage",
            ));
        }
        let input_hessian_members = members
            .iter()
            .filter(|(builder, _)| matches!(builder.spec().kind(), SaltV2Curvature::InputHessian))
            .count();
        if input_hessian_members != 0 && input_hessian_members != members.len() {
            return Err(SaltV2KroneckerEvidenceBuildError::Malformed(
                "shared-forward curvature mix",
            ));
        }
        Ok(Self {
            members: members
                .into_iter()
                .map(|(builder, input_stream)| SharedForwardMember {
                    builder,
                    input_stream,
                })
                .collect(),
            stream_columns,
        })
    }

    fn needs_output_factors(&self) -> bool {
        !matches!(
            self.members[0].builder.spec().kind(),
            SaltV2Curvature::InputHessian
        )
    }

    /// Preflight one batch so member mutation can only fail on resources.
    ///
    /// Every builder-level contract check (shape, finiteness, weights, mask)
    /// runs before any member advances; contract violations therefore reject
    /// atomically for the whole group.
    fn preflight(
        &self,
        streams: &[Vec<f32>],
        samples: usize,
        output_factors: Option<&[Vec<f32>]>,
        token_weights: Option<&[f64]>,
        token_mask: Option<&[bool]>,
    ) -> Result<(), SaltV2KroneckerEvidenceBuildError> {
        if streams.len() != self.stream_columns.len() {
            return Err(SaltV2KroneckerEvidenceBuildError::Malformed(
                "shared-forward stream count",
            ));
        }
        if samples == 0 {
            return Err(CurvatureError::EmptyBatch.into());
        }
        for (stream, columns) in streams.iter().zip(&self.stream_columns) {
            let expected = samples
                .checked_mul(*columns)
                .ok_or(SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
            if stream.len() != expected {
                return Err(SaltV2KroneckerEvidenceBuildError::BatchLengthMismatch {
                    field: "activations",
                    expected,
                    got: stream.len(),
                });
            }
            if let Some(index) = stream.iter().position(|value| !value.is_finite()) {
                return Err(CurvatureError::NonFiniteActivation {
                    sample: index / columns,
                    feature: index % columns,
                }
                .into());
            }
        }
        match (self.needs_output_factors(), output_factors) {
            (true, None) => {
                return Err(SaltV2KroneckerEvidenceBuildError::MissingOutputFactors);
            }
            (false, Some(_)) => {
                return Err(SaltV2KroneckerEvidenceBuildError::UnexpectedOutputFactors);
            }
            (true, Some(factors)) => {
                if factors.len() != self.members.len() {
                    return Err(SaltV2KroneckerEvidenceBuildError::Malformed(
                        "shared-forward output factor count",
                    ));
                }
                for (member, member_factors) in self.members.iter().zip(factors) {
                    let rows = member.builder.spec().rows();
                    let expected = samples
                        .checked_mul(rows)
                        .ok_or(SaltV2KroneckerEvidenceBuildError::AllocationFailed)?;
                    if member_factors.len() != expected {
                        return Err(SaltV2KroneckerEvidenceBuildError::BatchLengthMismatch {
                            field: "output factors",
                            expected,
                            got: member_factors.len(),
                        });
                    }
                    if let Some(index) = member_factors.iter().position(|value| !value.is_finite())
                    {
                        return Err(CurvatureError::NonFiniteGradient {
                            sample: index / rows,
                            output_row: index % rows,
                        }
                        .into());
                    }
                }
            }
            (false, None) => {}
        }
        if let Some(weights) = token_weights {
            if weights.len() != samples {
                return Err(CurvatureError::TokenWeightLengthMismatch {
                    expected: samples,
                    got: weights.len(),
                }
                .into());
            }
            if let Some(sample) = weights
                .iter()
                .position(|weight| !weight.is_finite() || *weight < 0.0)
            {
                return Err(CurvatureError::InvalidTokenWeight { sample }.into());
            }
        }
        if let Some(mask) = token_mask
            && mask.len() != samples
        {
            return Err(CurvatureError::TokenMaskLengthMismatch {
                expected: samples,
                got: mask.len(),
            }
            .into());
        }
        Ok(())
    }

    /// Feed one shared batch to every member and report retained segments.
    fn append(
        &mut self,
        streams: &[Vec<f32>],
        samples: usize,
        output_factors: Option<&[Vec<f32>]>,
        token_weights: Option<&[f64]>,
        token_mask: Option<&[bool]>,
    ) -> Result<Vec<(usize, usize)>, SharedForwardAppendError> {
        self.preflight(streams, samples, output_factors, token_weights, token_mask)
            .map_err(SharedForwardAppendError::Rejected)?;
        let mut residency = Vec::with_capacity(self.members.len());
        for (index, member) in self.members.iter_mut().enumerate() {
            let member_factors = output_factors.map(|factors| factors[index].as_slice());
            if let Err(error) = member.builder.accumulate_batch(
                &streams[member.input_stream],
                member_factors,
                samples,
                token_weights,
                token_mask,
            ) {
                // Every remaining failure mode is resource-shaped; only a
                // first-member failure leaves the whole group untouched.
                return Err(if index == 0 {
                    SharedForwardAppendError::Rejected(error)
                } else {
                    SharedForwardAppendError::Poisoned(error)
                });
            }
            let segments = member.builder.residency();
            residency.push((segments.input_segments(), segments.output_segments()));
        }
        Ok(residency)
    }
}

enum SharedForwardFinishError {
    Build(SaltV2KroneckerEvidenceBuildError),
    Publish(Qwen36PtqDriverError),
}

/// Shared-forward producer group: one calibration batch feeds many tensors.
///
/// Dense output factors only; embedding tables keep the standalone
/// indexed-output [`KroneckerEvidenceBuilder`] path.
#[pyclass(module = "tritium._tritium")]
pub(crate) struct KroneckerSharedForwardGroup {
    directory: Qwen36PtqEvidenceDirectory,
    core: Option<SharedForwardGroupCore>,
    max_batch_bytes: usize,
}

#[pymethods]
impl KroneckerSharedForwardGroup {
    /// Create one source-bound builder per `(tensor_index, tensor_name, rows,
    /// columns, input_stream)` member under shared record and batch ceilings.
    #[new]
    #[pyo3(signature = (
        evidence_dir,
        members,
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
        members: Vec<(u64, String, usize, usize, usize)>,
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
        if members.is_empty() {
            return Err(contract_error("members must not be empty"));
        }
        let curvature = parse_curvature(curvature)?;
        let source_id = CurvatureSourceId::new(
            parse_digest("source_model_digest", source_model_digest)?,
            parse_digest("activation_cache_digest", activation_cache_digest)?,
            parse_digest("token_stream_digest", token_stream_digest)?,
        )
        .map_err(contract_error)?;
        let directory =
            Qwen36PtqEvidenceDirectory::create_bounded(evidence_dir, max_evidence_bytes)
                .map_err(directory_error)?;
        let mut seen_indices = std::collections::BTreeSet::new();
        let mut builders = Vec::with_capacity(members.len());
        for (tensor_index, tensor_name, rows, columns, input_stream) in members {
            if tensor_index >= 1_000_000 {
                return Err(contract_error(
                    "tensor_index must fit the six-digit Qwen evidence namespace",
                ));
            }
            if !seen_indices.insert(tensor_index) {
                return Err(contract_error(
                    "members must not repeat one global tensor ordinal",
                ));
            }
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
            let builder = directory.create_builder(spec).map_err(driver_build_error)?;
            builders.push((builder, input_stream));
        }
        let core = SharedForwardGroupCore::new(builders).map_err(build_error)?;
        Ok(Self {
            directory,
            core: Some(core),
            max_batch_bytes,
        })
    }

    /// Atomically append one shared batch: one buffer per input stream, one
    /// dense output-factor buffer per member (omitted for input-hessian).
    #[pyo3(signature = (
        activations_f32le,
        samples,
        *,
        output_factors_f32le = None,
        token_weights_f64le = None,
        token_mask_u8 = None
    ))]
    fn append_group(
        &mut self,
        py: Python<'_>,
        activations_f32le: Vec<PyBackedBytes>,
        samples: usize,
        output_factors_f32le: Option<Vec<PyBackedBytes>>,
        token_weights_f64le: Option<&[u8]>,
        token_mask_u8: Option<&[u8]>,
    ) -> PyResult<Vec<(usize, usize)>> {
        let core = self
            .core
            .as_ref()
            .ok_or_else(|| KroneckerStateError::new_err("evidence group is already finished"))?;
        if activations_f32le.len() != core.stream_columns.len() {
            return Err(contract_error(format!(
                "activations_f32le requires {} stream buffers, got {}",
                core.stream_columns.len(),
                activations_f32le.len()
            )));
        }
        if core.needs_output_factors() {
            let factor_buffers = output_factors_f32le
                .as_ref()
                .map_or(0, |factors| factors.len());
            if factor_buffers != core.members.len() {
                return Err(contract_error(format!(
                    "output_factors_f32le requires {} member buffers, got {factor_buffers}",
                    core.members.len()
                )));
            }
        } else if output_factors_f32le.is_some() {
            return Err(contract_error(
                "input-Hessian builder rejects output factors",
            ));
        }
        let mut total_bytes = 0_usize;
        for buffer in activations_f32le
            .iter()
            .map(|buffer| &buffer[..])
            .chain(
                output_factors_f32le
                    .iter()
                    .flatten()
                    .map(|buffer| &buffer[..]),
            )
            .chain(token_weights_f64le)
            .chain(token_mask_u8)
        {
            total_bytes = total_bytes
                .checked_add(buffer.len())
                .filter(|bytes| *bytes <= self.max_batch_bytes)
                .ok_or_else(|| contract_error("batch exceeds max_batch_bytes"))?;
        }
        let mut streams = Vec::with_capacity(activations_f32le.len());
        for (buffer, columns) in activations_f32le.iter().zip(&core.stream_columns) {
            let expected = checked_elements(samples, *columns, "activations")?;
            streams.push(decode_f32le("activations", buffer, expected)?);
        }
        let output_factors = match &output_factors_f32le {
            Some(factor_buffers) => {
                let mut decoded = Vec::with_capacity(factor_buffers.len());
                for (buffer, member) in factor_buffers.iter().zip(&core.members) {
                    let expected =
                        checked_elements(samples, member.builder.spec().rows(), "output factors")?;
                    decoded.push(decode_f32le("output factors", buffer, expected)?);
                }
                Some(decoded)
            }
            None => None,
        };
        let token_weights = token_weights_f64le
            .map(|bytes| decode_f64le("token weights", bytes, samples))
            .transpose()?;
        let token_mask = token_mask_u8
            .map(|bytes| decode_mask(bytes, samples))
            .transpose()?;
        let core = self.core.as_mut().expect("checked active group");
        let outcome = py.detach(move || {
            core.append(
                &streams,
                samples,
                output_factors.as_deref(),
                token_weights.as_deref(),
                token_mask.as_deref(),
            )
        });
        match outcome {
            Ok(residency) => Ok(residency),
            Err(error @ SharedForwardAppendError::Rejected(_)) => {
                Err(build_error(error.into_build_error()))
            }
            Err(error @ SharedForwardAppendError::Poisoned(_)) => {
                self.core = None;
                Err(build_error(error.into_build_error()))
            }
        }
    }

    /// Finalize and atomically publish every member record in member order.
    ///
    /// Reinstalling byte-identical records is idempotent, so a retryable
    /// publication failure keeps the group active and a retried finish
    /// republishes already-installed members without conflict.
    fn finish(&mut self, py: Python<'_>) -> PyResult<Vec<KroneckerEvidenceReceipt>> {
        let core = self
            .core
            .as_ref()
            .ok_or_else(|| KroneckerStateError::new_err("evidence group is already finished"))?;
        let directory = self.directory.clone();
        let outcome = py.detach(move || {
            let mut receipts = Vec::with_capacity(core.members.len());
            for member in &core.members {
                let record = member
                    .builder
                    .finish()
                    .map_err(SharedForwardFinishError::Build)?;
                let receipt = directory
                    .install(&record)
                    .map_err(SharedForwardFinishError::Publish)?;
                receipts.push((member.builder.spec().tensor_index(), receipt));
            }
            Ok(receipts)
        });
        match outcome {
            Ok(receipts) => {
                self.core = None;
                Ok(receipts
                    .into_iter()
                    .map(|(tensor_index, receipt)| KroneckerEvidenceReceipt {
                        tensor_index,
                        record_digest: encode_digest(&receipt.record_digest()),
                        bytes: receipt.bytes(),
                    })
                    .collect())
            }
            Err(SharedForwardFinishError::Build(error)) => Err(build_error(error)),
            Err(SharedForwardFinishError::Publish(error)) => {
                if terminal_publication_error(&error) {
                    self.core = None;
                }
                Err(publication_error(error))
            }
        }
    }

    /// Drop unpublished accumulator state and make this group terminal.
    fn abort(&mut self) {
        self.core = None;
    }

    /// Whether append/finalize operations are still admitted.
    #[getter]
    fn active(&self) -> bool {
        self.core.is_some()
    }

    fn __repr__(&self) -> String {
        format!(
            "KroneckerSharedForwardGroup(members={}, streams={}, active={}, max_batch_bytes={})",
            self.core.as_ref().map_or(0, |core| core.members.len()),
            self.core
                .as_ref()
                .map_or(0, |core| core.stream_columns.len()),
            self.core.is_some(),
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

    fn group_spec(
        kind: SaltV2Curvature,
        tensor_index: u64,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> SaltV2KroneckerEvidenceSpec {
        SaltV2KroneckerEvidenceSpec::new(
            kind,
            CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap(),
            tensor_index,
            name,
            rows,
            columns,
            0.125,
        )
        .unwrap()
    }

    fn ramp(count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|index| index as f32 * scale + 0.5).collect()
    }

    // Constant within each G128 group per sample: group Grams stay numerically
    // inside the strict PSD admission tolerance while differing across
    // samples, groups, and streams.
    fn grouped_rows(samples: usize, columns: usize, seed: f32) -> Vec<f32> {
        let mut values = Vec::with_capacity(samples * columns);
        for sample in 0..samples {
            for column in 0..columns {
                let group = column / 128;
                values.push(seed + sample as f32 + group as f32 * 0.5);
            }
        }
        values
    }

    #[test]
    fn shared_forward_group_records_are_byte_identical_to_standalone() {
        // Two tensors on one shared stream plus one tensor on a second stream,
        // fed in two batches, must reproduce standalone records exactly.
        let specs = [
            (
                group_spec(SaltV2Curvature::GuidedFisher, 0, "a.weight", 3, 128),
                0_usize,
            ),
            (
                group_spec(SaltV2Curvature::GuidedFisher, 1, "b.weight", 2, 128),
                0_usize,
            ),
            (
                group_spec(SaltV2Curvature::GuidedFisher, 2, "c.weight", 4, 256),
                1_usize,
            ),
        ];
        let batches = [(2_usize, 0.25_f32), (3_usize, -0.5_f32)];
        let weights = [1.0_f64, 2.0, 0.5, 1.0, 3.0];
        let mask = [true, false, true, true, true];

        let mut group = SharedForwardGroupCore::new(
            specs
                .iter()
                .map(|(spec, stream)| {
                    (
                        SaltV2KroneckerEvidenceBuilder::new(spec.clone()).unwrap(),
                        *stream,
                    )
                })
                .collect(),
        )
        .unwrap();
        let mut standalone = specs
            .iter()
            .map(|(spec, _)| SaltV2KroneckerEvidenceBuilder::new(spec.clone()).unwrap())
            .collect::<Vec<_>>();

        let mut offset = 0;
        for (samples, scale) in batches {
            let streams = [
                grouped_rows(samples, 128, scale),
                grouped_rows(samples, 256, -scale),
            ];
            let factors = specs
                .iter()
                .map(|(spec, _)| ramp(samples * spec.rows(), scale * 2.0))
                .collect::<Vec<_>>();
            let batch_weights = &weights[offset..offset + samples];
            let batch_mask = &mask[offset..offset + samples];
            group
                .append(
                    &streams,
                    samples,
                    Some(&factors),
                    Some(batch_weights),
                    Some(batch_mask),
                )
                .unwrap();
            for ((builder, (_, stream)), member_factors) in
                standalone.iter_mut().zip(&specs).zip(&factors)
            {
                builder
                    .accumulate_batch(
                        &streams[*stream],
                        Some(member_factors),
                        samples,
                        Some(batch_weights),
                        Some(batch_mask),
                    )
                    .unwrap();
            }
            offset += samples;
        }

        for (member, builder) in group.members.iter().zip(&standalone) {
            let grouped = member.builder.finish().unwrap();
            let solo = builder.finish().unwrap();
            assert_eq!(grouped.record_digest(), solo.record_digest());
            assert_eq!(
                grouped.canonical_bytes().unwrap(),
                solo.canonical_bytes().unwrap()
            );
        }
    }

    #[test]
    fn shared_forward_group_preflight_rejects_without_member_drift() {
        let mut group = SharedForwardGroupCore::new(vec![
            (
                SaltV2KroneckerEvidenceBuilder::new(group_spec(
                    SaltV2Curvature::GuidedFisher,
                    0,
                    "a.weight",
                    2,
                    128,
                ))
                .unwrap(),
                0,
            ),
            (
                SaltV2KroneckerEvidenceBuilder::new(group_spec(
                    SaltV2Curvature::GuidedFisher,
                    1,
                    "b.weight",
                    2,
                    128,
                ))
                .unwrap(),
                0,
            ),
        ])
        .unwrap();
        let stream = [ramp(128, 1.0)];
        let good = [ramp(2, 1.0), ramp(2, 2.0)];
        group.append(&stream, 1, Some(&good), None, None).unwrap();
        let baseline = group
            .members
            .iter()
            .map(|member| member.builder.finish().unwrap().record_digest())
            .collect::<Vec<_>>();

        // Second member's factors are malformed: length, then finiteness.
        let short = [ramp(2, 1.0), ramp(1, 2.0)];
        assert!(matches!(
            group.append(&stream, 1, Some(&short), None, None),
            Err(SharedForwardAppendError::Rejected(
                SaltV2KroneckerEvidenceBuildError::BatchLengthMismatch { .. }
            ))
        ));
        let non_finite = [ramp(2, 1.0), vec![1.0, f32::NAN]];
        assert!(matches!(
            group.append(&stream, 1, Some(&non_finite), None, None),
            Err(SharedForwardAppendError::Rejected(
                SaltV2KroneckerEvidenceBuildError::Curvature(_)
            ))
        ));
        assert!(matches!(
            group.append(&stream, 1, None, None, None),
            Err(SharedForwardAppendError::Rejected(
                SaltV2KroneckerEvidenceBuildError::MissingOutputFactors
            ))
        ));

        // No member ordinal advanced through any rejected batch.
        let after = group
            .members
            .iter()
            .map(|member| member.builder.finish().unwrap().record_digest())
            .collect::<Vec<_>>();
        assert_eq!(baseline, after);
    }

    #[test]
    fn shared_forward_group_rejects_malformed_membership() {
        let builder = || {
            SaltV2KroneckerEvidenceBuilder::new(group_spec(
                SaltV2Curvature::GuidedFisher,
                0,
                "a.weight",
                2,
                128,
            ))
            .unwrap()
        };
        assert!(SharedForwardGroupCore::new(Vec::new()).is_err());
        // Stream 0 has no member, so its column width is undefined.
        assert!(SharedForwardGroupCore::new(vec![(builder(), 1)]).is_err());
        // One stream cannot serve two different column widths.
        assert!(
            SharedForwardGroupCore::new(vec![
                (builder(), 0),
                (
                    SaltV2KroneckerEvidenceBuilder::new(group_spec(
                        SaltV2Curvature::GuidedFisher,
                        1,
                        "b.weight",
                        2,
                        256,
                    ))
                    .unwrap(),
                    0,
                ),
            ])
            .is_err()
        );
        // Input-Hessian members cannot mix with factor-bearing members.
        assert!(
            SharedForwardGroupCore::new(vec![
                (builder(), 0),
                (
                    SaltV2KroneckerEvidenceBuilder::new(group_spec(
                        SaltV2Curvature::InputHessian,
                        1,
                        "b.weight",
                        2,
                        128,
                    ))
                    .unwrap(),
                    0,
                ),
            ])
            .is_err()
        );
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
