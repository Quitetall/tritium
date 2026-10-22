//! Thin abi3 receiver over the shared file source, engine and publication modules.
use pyo3::{
    exceptions::{PyFileExistsError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use std::{
    collections::BTreeMap,
    fs::File,
    path::{Path, PathBuf},
};
use tritium_evidence::{
    CALIBRATED_LEVELS_FITTED_REGISTRATION, CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
    CALIBRATED_QUANTIZATION_STARTED_REGISTRATION, DIRECT_LEVELS_FITTED_REGISTRATION,
    EvidenceRecorder, OWNER_QUANTIZATION_COMPLETED_REGISTRATION,
    OWNER_QUANTIZATION_STARTED_REGISTRATION, OWNER_SCORE_LEVELS_FITTED_REGISTRATION,
    QUANTIZATION_COMPLETED_REGISTRATION, QUANTIZATION_STARTED_REGISTRATION, VerifiedLog,
    verify_jsonl,
};
use tritium_format::{TritPackageReadLimits, trit_model::TritModelReader};
use tritium_quantize::engine::{
    DirectoryTensorSink, Engine, FileWeightSource, QuantEngineError, TritOwnerCalibrationSource,
    file_calibration_source, open_model_reader_with_fit_log, open_original_file,
    read_original_file_bounded, verify_package_with_fit_log,
};
use tritium_schema::{
    CalibratedQuantPlan, ModelQuantPlan, OwnerQuantPlan, QuantPlan, TensorStorage,
};

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn paths(input: Option<BTreeMap<String, String>>) -> BTreeMap<String, PathBuf> {
    input
        .unwrap_or_default()
        .into_iter()
        .map(|(name, path)| (name, path.into()))
        .collect()
}

fn open_bound_model(
    root: &Path,
    maximum: u64,
    evidence_maximum: u64,
) -> PyResult<(TritModelReader<File>, VerifiedLog, u64)> {
    if maximum == 0 || evidence_maximum == 0 {
        return Err(error("package and evidence byte limits must be positive"));
    }
    let text = read_original_file_bounded(&root.join("evidence.jsonl"), evidence_maximum)
        .map_err(error)?;
    let log = verify_jsonl(std::str::from_utf8(&text).map_err(error)?).map_err(error)?;
    // Preserve existing package-byte admission for metadata as well. This does
    // not introduce a hidden lower metadata ceiling or claim a total RSS bound.
    let metadata_maximum = usize::try_from(maximum).unwrap_or(usize::MAX);
    let limits = TritPackageReadLimits {
        max_source_bytes: maximum,
        max_manifest_bytes: metadata_maximum,
        max_directory_bytes: metadata_maximum,
    };
    let reader = open_model_reader_with_fit_log(
        open_original_file(&root.join("model.trit")).map_err(error)?,
        limits,
        &log,
    )
    .map_err(error)?;
    Ok((reader, log, text.len() as u64))
}

enum Plan {
    Plain(QuantPlan),
    Calibrated(CalibratedQuantPlan),
    Owner(OwnerQuantPlan),
}
impl Plan {
    fn view(&self) -> ModelQuantPlan<'_> {
        match self {
            Self::Plain(plan) => ModelQuantPlan::Uncalibrated(plan),
            Self::Calibrated(plan) => ModelQuantPlan::Calibrated(plan),
            Self::Owner(plan) => ModelQuantPlan::Owner(plan),
        }
    }
}

fn parse_plan(json: &str, calibrated: bool, maximum: usize) -> PyResult<Plan> {
    if maximum == 0 || json.len() > maximum {
        return Err(error("plan JSON exceeds positive byte limit"));
    }
    let plan = if calibrated {
        Plan::Calibrated(serde_json::from_str(json).map_err(error)?)
    } else {
        Plan::Plain(serde_json::from_str(json).map_err(error)?)
    };
    plan.view().validate().map_err(error)?;
    Ok(plan)
}

/// Internal frontend preflight; the same parser runs before native conversion.
#[pyfunction]
#[pyo3(signature=(plan_json, *, calibrated=false, max_plan_bytes=16_777_216))]
pub(crate) fn _validate_model_plan(
    plan_json: &str,
    calibrated: bool,
    max_plan_bytes: usize,
) -> PyResult<()> {
    parse_plan(plan_json, calibrated, max_plan_bytes).map(|_| ())
}

/// Actual conversion identities, not a qualification or complete architecture claim.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct NativeModelConversion {
    #[pyo3(get)]
    output_dir: String,
    #[pyo3(get)]
    source_id: String,
    #[pyo3(get)]
    package_id: String,
    #[pyo3(get)]
    model_ids: BTreeMap<String, String>,
    #[pyo3(get)]
    evidence_root: String,
    #[pyo3(get)]
    run_id: String,
    #[pyo3(get)]
    publication_warnings: Vec<String>,
    max_package_bytes: u64,
    evidence_bytes: u64,
}
impl NativeModelConversion {
    /// Reopen an engine package/log without trusting caller identity fields.
    fn open_inner(
        py: Python<'_>,
        output_dir: String,
        max_package_bytes: u64,
        max_evidence_bytes: u64,
    ) -> PyResult<Self> {
        if output_dir.is_empty() || max_package_bytes == 0 || max_evidence_bytes == 0 {
            return Err(error("directory and byte limits must be positive"));
        }
        let root = std::env::current_dir().map_err(error)?.join(output_dir);
        let output_dir = root
            .clone()
            .into_os_string()
            .into_string()
            .map_err(|_| error("output path is not UTF-8"))?;
        py.detach(|| {
            let (model, log, evidence_bytes) =
                open_bound_model(&root, max_package_bytes, max_evidence_bytes)?;
            let mut ids = tritium_evidence::QuantizationReceiptView::fold(&log)
                .map_err(error)?
                .into_iter()
                .map(|view| view.source_id())
                .collect::<Vec<_>>();
            ids.extend(
                tritium_evidence::CalibratedQuantizationReceiptView::fold(&log)
                    .map_err(error)?
                    .into_iter()
                    .map(|view| view.source_id()),
            );
            ids.extend(
                tritium_evidence::OwnerQuantizationReceiptView::fold(&log)
                    .map_err(error)?
                    .into_iter()
                    .map(|view| view.source_id()),
            );
            if ids.len() != 1 {
                return Err(error("exactly one conversion identity required"));
            }
            let model_ids = model
                .manifest()
                .levels
                .iter()
                .map(|level| {
                    Ok((
                        level.name.clone(),
                        hex(model.model_id(&level.name).map_err(error)?.as_bytes()),
                    ))
                })
                .collect::<PyResult<_>>()?;
            let run_id = log
                .events
                .first()
                .ok_or_else(|| error("conversion log empty"))?
                .run
                .clone();
            Ok(Self {
                output_dir,
                source_id: hex(ids[0].as_bytes()),
                package_id: hex(model.package_id().as_bytes()),
                model_ids,
                evidence_root: log.root_digest,
                run_id,
                publication_warnings: Vec::new(),
                max_package_bytes,
                evidence_bytes,
            })
        })
    }
    fn verified_model_bounded(
        &self,
        maximum: u64,
        evidence_maximum: u64,
    ) -> PyResult<TritModelReader<File>> {
        let (model, log, _) = open_bound_model(
            Path::new(&self.output_dir),
            self.max_package_bytes.min(maximum),
            self.evidence_bytes.min(evidence_maximum),
        )?;
        if hex(model.package_id().as_bytes()) != self.package_id
            || log.root_digest != self.evidence_root
        {
            return Err(error("conversion package or evidence identity changed"));
        }
        Ok(model)
    }
    fn verified_contents_bounded(
        &self,
        maximum: u64,
        evidence_maximum: u64,
    ) -> PyResult<(Vec<u8>, Vec<u8>)> {
        if maximum == 0 || evidence_maximum == 0 {
            return Err(error("package and evidence byte limits must be positive"));
        }
        let root = Path::new(&self.output_dir);
        let bytes = read_original_file_bounded(
            &root.join("model.trit"),
            self.max_package_bytes.min(maximum),
        )
        .map_err(error)?;
        let text = read_original_file_bounded(
            &root.join("evidence.jsonl"),
            self.evidence_bytes.min(evidence_maximum),
        )
        .map_err(error)?;
        let log = verify_jsonl(std::str::from_utf8(&text).map_err(error)?).map_err(error)?;
        verify_package_with_fit_log(&bytes, &log).map_err(error)?;
        let model = tritium_format::trit_model::read_trit_model(&bytes).map_err(error)?;
        if hex(model.package_id().as_bytes()) != self.package_id
            || log.root_digest != self.evidence_root
        {
            return Err(error("conversion package or evidence identity changed"));
        }
        Ok((bytes, text))
    }
}
#[pymethods]
impl NativeModelConversion {
    #[staticmethod]
    #[pyo3(signature=(output_dir, *, max_package_bytes=8_589_934_592, max_evidence_bytes=16_777_216))]
    fn open(
        py: Python<'_>,
        output_dir: String,
        max_package_bytes: u64,
        max_evidence_bytes: u64,
    ) -> PyResult<Self> {
        Self::open_inner(py, output_dir, max_package_bytes, max_evidence_bytes)
    }
    /// Reopen exact files and verify package/log binding against this result.
    #[pyo3(signature=(*, max_package_bytes=None, max_evidence_bytes=None))]
    fn verify(
        &self,
        py: Python<'_>,
        max_package_bytes: Option<u64>,
        max_evidence_bytes: Option<u64>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.verified_model_bounded(
                max_package_bytes.unwrap_or(self.max_package_bytes),
                max_evidence_bytes.unwrap_or(self.evidence_bytes),
            )
            .map(|_| ())
        })
    }
    /// Export the verified original files without refitting or changing identities.
    /// Uses checked bounded package chunks, not a full-process RSS guarantee.
    #[pyo3(signature=(output_dir, *, max_package_bytes=None, max_evidence_bytes=None))]
    fn export(
        &self,
        py: Python<'_>,
        output_dir: String,
        max_package_bytes: Option<u64>,
        max_evidence_bytes: Option<u64>,
    ) -> PyResult<Self> {
        if output_dir.is_empty() {
            return Err(error("output_dir must not be empty"));
        }
        let output_dir = std::env::current_dir()
            .map_err(error)?
            .join(output_dir)
            .into_os_string()
            .into_string()
            .map_err(|_| error("output path is not UTF-8"))?;
        let maximum = max_package_bytes
            .unwrap_or(self.max_package_bytes)
            .min(self.max_package_bytes);
        py.detach(|| {
            let evidence_maximum = max_evidence_bytes
                .unwrap_or(self.evidence_bytes)
                .min(self.evidence_bytes);
            if maximum == 0 || evidence_maximum == 0 {
                return Err(error("package and evidence byte limits must be positive"));
            }
            let root = Path::new(&self.output_dir);
            let text = read_original_file_bounded(&root.join("evidence.jsonl"), evidence_maximum)
                .map_err(error)?;
            let text = std::str::from_utf8(&text).map_err(error)?;
            if verify_jsonl(text).map_err(error)?.root_digest != self.evidence_root {
                return Err(error("conversion package or evidence identity changed"));
            }
            let expected = tritium_format::PackageId::from_digest(
                *blake3::Hash::from_hex(&self.package_id)
                    .map_err(error)?
                    .as_bytes(),
            );
            let metadata_maximum = usize::try_from(maximum).unwrap_or(usize::MAX);
            let warnings = DirectoryTensorSink::new(&output_dir)
                .publish_verified_model_reader(
                    open_original_file(&root.join("model.trit")).map_err(error)?,
                    TritPackageReadLimits {
                        max_source_bytes: maximum,
                        max_manifest_bytes: metadata_maximum,
                        max_directory_bytes: metadata_maximum,
                    },
                    expected,
                    text,
                )
                .map_err(|failure| match failure {
                    QuantEngineError::DestinationExists => {
                        PyFileExistsError::new_err("output directory already exists")
                    }
                    other => error(other),
                })?;
            let mut result = self.clone();
            result.output_dir = output_dir;
            result.max_package_bytes = maximum;
            result.publication_warnings.extend(
                warnings
                    .into_iter()
                    .map(|warning| format!("Export: {warning:?}")),
            );
            Ok(result)
        })
    }
    #[pyo3(signature=(level, load_run_id, *, max_payload_bytes=2_147_483_648,
        max_workspace_bytes=67_108_864, max_output_bytes=67_108_864, max_package_bytes=None,
        max_evidence_bytes=None))]
    #[expect(
        clippy::too_many_arguments,
        reason = "PyO3 exposes independent caller byte ceilings as explicit keyword arguments"
    )]
    fn open_level(
        &self,
        py: Python<'_>,
        level: String,
        load_run_id: String,
        max_payload_bytes: u64,
        max_workspace_bytes: u64,
        max_output_bytes: usize,
        max_package_bytes: Option<u64>,
        max_evidence_bytes: Option<u64>,
    ) -> PyResult<crate::model_level::NativeModelLevel> {
        py.detach(|| {
            crate::model_level::open(
                self.verified_contents_bounded(
                    max_package_bytes.unwrap_or(self.max_package_bytes),
                    max_evidence_bytes.unwrap_or(self.evidence_bytes),
                )?
                .0,
                level,
                load_run_id,
                max_payload_bytes,
                max_workspace_bytes,
                max_output_bytes,
            )
        })
    }
    // The source adapter can report cleanup after a successful commit without
    // turning a published artifact into a misleading conversion failure.
    fn _with_source_cleanup_warning(&self, message: &str) -> Self {
        let mut result = self.clone();
        result.publication_warnings.push(format!(
            "SourceCleanup: {}",
            message.chars().take(1024).collect::<String>()
        ));
        result
    }
    /// Read exact original asset bytes after package/log identity verification.
    #[pyo3(signature=(name, *, max_bytes=67_108_864))]
    fn asset<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        max_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if max_bytes == 0 {
            return Err(error("max_bytes must be positive"));
        }
        let payload = py.detach(|| {
            let mut model =
                self.verified_model_bounded(self.max_package_bytes, self.evidence_bytes)?;
            let length = model.asset_byte_len(name).map_err(error)?;
            if length > max_bytes as u64 {
                return Err(error("asset exceeds byte limit"));
            }
            let mut payload = Vec::new();
            payload.try_reserve_exact(length as usize).map_err(error)?;
            model
                .visit_asset(name, |chunk| {
                    payload.extend_from_slice(chunk);
                    Ok::<_, std::convert::Infallible>(())
                })
                .map_err(error)?;
            model.verify().map_err(error)?;
            Ok::<_, PyErr>(payload)
        })?;
        Ok(PyBytes::new(py, &payload))
    }
    /// Return a declared dense-preserved tensor's original bytes, never additive decode.
    #[pyo3(signature=(level, name, *, max_bytes=67_108_864))]
    fn preserved_tensor<'py>(
        &self,
        py: Python<'py>,
        level: &str,
        name: &str,
        max_bytes: usize,
    ) -> PyResult<(String, Vec<u64>, Bound<'py, PyBytes>)> {
        if max_bytes == 0 {
            return Err(error("max_bytes must be positive"));
        }
        let (precision, shape, payload) = py.detach(|| {
            let mut model =
                self.verified_model_bounded(self.max_package_bytes, self.evidence_bytes)?;
            let length = model.tensor_byte_len(level, name).map_err(error)?;
            if length > max_bytes as u64 {
                return Err(error("preserved tensor exceeds byte limit"));
            }
            let (precision, shape) = match &model.tensor_meta(level, name).map_err(error)?.storage {
                TensorStorage::Dense { precision, shape } => {
                    (format!("{precision:?}"), shape.clone())
                }
                _ => return Err(error("requested tensor is not dense-preserved")),
            };
            let mut payload = Vec::new();
            payload.try_reserve_exact(length as usize).map_err(error)?;
            model
                .visit_tensor(level, name, |chunk| {
                    payload.extend_from_slice(chunk);
                    Ok::<_, std::convert::Infallible>(())
                })
                .map_err(error)?;
            model.verify().map_err(error)?;
            Ok((precision, shape, payload))
        })?;
        Ok((precision, shape, PyBytes::new(py, &payload)))
    }
}

/// Convert complete declared float inventories through the common native engine.
#[pyfunction]
#[pyo3(signature=(plan_json, source_shards, output_dir, run_id, *, asset_paths=None,
    input_grams=None, calibration_caches=None, calibrated=false, max_plan_bytes=16_777_216))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn convert_model_plan(
    py: Python<'_>,
    plan_json: &str,
    source_shards: Vec<String>,
    output_dir: String,
    run_id: String,
    asset_paths: Option<BTreeMap<String, String>>,
    input_grams: Option<BTreeMap<String, String>>,
    calibration_caches: Option<BTreeMap<String, String>>,
    calibrated: bool,
    max_plan_bytes: usize,
) -> PyResult<NativeModelConversion> {
    let request = parse_plan(plan_json, calibrated, max_plan_bytes)?;
    run_conversion(
        py,
        request,
        ConversionInputs {
            source_shards,
            output_dir,
            run_id,
            asset_paths,
            input_grams,
            calibration_caches,
            owner_cache_paths: BTreeMap::new(),
        },
    )
}

/// Distinct owner-model request; no interpretation change to the old bool route.
#[pyfunction]
#[pyo3(signature=(plan_json, source_shards, output_dir, run_id, *, owner_cache_paths,
    asset_paths=None, input_grams=None, calibration_caches=None, max_plan_bytes=16_777_216))]
#[expect(
    clippy::too_many_arguments,
    reason = "Explicit PyO3 source and immutable-cache inputs"
)]
pub(crate) fn convert_owner_model_plan(
    py: Python<'_>,
    plan_json: &str,
    source_shards: Vec<String>,
    output_dir: String,
    run_id: String,
    owner_cache_paths: BTreeMap<String, BTreeMap<String, String>>,
    asset_paths: Option<BTreeMap<String, String>>,
    input_grams: Option<BTreeMap<String, String>>,
    calibration_caches: Option<BTreeMap<String, String>>,
    max_plan_bytes: usize,
) -> PyResult<NativeModelConversion> {
    if max_plan_bytes == 0 || plan_json.len() > max_plan_bytes {
        return Err(error("plan JSON exceeds positive byte limit"));
    }
    let request = Plan::Owner(serde_json::from_str(plan_json).map_err(error)?);
    request.view().validate().map_err(error)?;
    run_conversion(
        py,
        request,
        ConversionInputs {
            source_shards,
            output_dir,
            run_id,
            asset_paths,
            input_grams,
            calibration_caches,
            owner_cache_paths,
        },
    )
}

struct ConversionInputs {
    source_shards: Vec<String>,
    output_dir: String,
    run_id: String,
    asset_paths: Option<BTreeMap<String, String>>,
    input_grams: Option<BTreeMap<String, String>>,
    calibration_caches: Option<BTreeMap<String, String>>,
    owner_cache_paths: BTreeMap<String, BTreeMap<String, String>>,
}

fn run_conversion(
    py: Python<'_>,
    request: Plan,
    inputs: ConversionInputs,
) -> PyResult<NativeModelConversion> {
    let ConversionInputs {
        source_shards,
        output_dir,
        run_id,
        asset_paths,
        input_grams,
        calibration_caches,
        owner_cache_paths,
    } = inputs;
    if run_id.is_empty() {
        return Err(error("run_id must not be empty"));
    }
    if output_dir.is_empty() {
        return Err(error("output_dir must not be empty"));
    }
    // Bind relative destinations at conversion time, not later at verification.
    let output_dir = std::env::current_dir()
        .map_err(error)?
        .join(output_dir)
        .into_os_string()
        .into_string()
        .map_err(|_| error("output directory is not representable as a Python string"))?;
    let shards = source_shards
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let assets = paths(asset_paths);
    let grams = paths(input_grams);
    let caches = paths(calibration_caches);
    let owners = owner_cache_paths
        .into_iter()
        .map(|(name, files)| {
            let files = files
                .into_iter()
                .map(|(id, path)| {
                    let hash = blake3::Hash::from_hex(&id).map_err(error)?;
                    if id != hex(hash.as_bytes()) {
                        return Err(error("owner cache identity must be canonical lower hex"));
                    }
                    Ok((*hash.as_bytes(), PathBuf::from(path)))
                })
                .collect::<PyResult<BTreeMap<_, _>>>()?;
            Ok((name, files))
        })
        .collect::<PyResult<BTreeMap<_, _>>>()?;
    py.detach(move || {
        let mut calibration = file_calibration_source(request.view(), caches).map_err(error)?;
        let mut owners = TritOwnerCalibrationSource::new(owners).map_err(error)?;
        let mut source =
            FileWeightSource::open(request.view(), &shards, assets, grams).map_err(error)?;
        let recorder = EvidenceRecorder::new(
            &run_id,
            [
                DIRECT_LEVELS_FITTED_REGISTRATION,
                QUANTIZATION_STARTED_REGISTRATION,
                QUANTIZATION_COMPLETED_REGISTRATION,
                CALIBRATED_LEVELS_FITTED_REGISTRATION,
                CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
                CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
                OWNER_SCORE_LEVELS_FITTED_REGISTRATION,
                OWNER_QUANTIZATION_STARTED_REGISTRATION,
                OWNER_QUANTIZATION_COMPLETED_REGISTRATION,
            ],
        )
        .map_err(error)?;
        let mut sink = DirectoryTensorSink::new(&output_dir);
        let result = match &request {
            Plan::Plain(plan) => Engine::run(plan, &mut source, &mut sink, recorder),
            Plan::Calibrated(plan) => {
                Engine::run_calibrated(plan, &mut source, &mut calibration, &mut sink, recorder)
            }
            Plan::Owner(plan) => Engine::run_owner(
                plan,
                &mut source,
                &mut owners,
                Some(&mut calibration),
                &mut sink,
                recorder,
            ),
        }
        .map_err(error)?;
        Ok(NativeModelConversion {
            output_dir,
            source_id: hex(result.source_id.as_bytes()),
            package_id: hex(result.package_id.as_bytes()),
            model_ids: result
                .model_ids
                .iter()
                .map(|(name, id)| (name.clone(), hex(id.as_bytes())))
                .collect(),
            evidence_root: result.evidence.log().root_digest.clone(),
            run_id,
            publication_warnings: result
                .warnings
                .iter()
                .map(|warning| format!("{warning:?}"))
                .collect(),
            max_package_bytes: request.view().limits().max_package_bytes,
            evidence_bytes: result.evidence.jsonl().len() as u64,
        })
    })
}
