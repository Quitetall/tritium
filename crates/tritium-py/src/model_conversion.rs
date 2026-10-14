//! Thin abi3 receiver over the shared file source, engine and publication modules.
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use tritium_evidence::{
    CALIBRATED_LEVELS_FITTED_REGISTRATION, CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
    CALIBRATED_QUANTIZATION_STARTED_REGISTRATION, DIRECT_LEVELS_FITTED_REGISTRATION,
    EvidenceRecorder, QUANTIZATION_COMPLETED_REGISTRATION, QUANTIZATION_STARTED_REGISTRATION,
    verify_jsonl,
};
use tritium_quantize::engine::{
    DirectoryTensorSink, Engine, FileWeightSource, file_calibration_source,
    read_original_file_bounded, verify_package_with_fit_log,
};
use tritium_schema::{CalibratedQuantPlan, ModelQuantPlan, QuantPlan, TensorStorage};

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn paths(input: Option<BTreeMap<String, String>>) -> BTreeMap<String, PathBuf> {
    input
        .unwrap_or_default()
        .into_iter()
        .map(|(name, path)| (name, path.into()))
        .collect()
}

enum Plan {
    Plain(QuantPlan),
    Calibrated(CalibratedQuantPlan),
}
impl Plan {
    fn view(&self) -> ModelQuantPlan<'_> {
        match self {
            Self::Plain(plan) => ModelQuantPlan::Uncalibrated(plan),
            Self::Calibrated(plan) => ModelQuantPlan::Calibrated(plan),
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
    fn verified_bytes(&self) -> PyResult<Vec<u8>> {
        let root = Path::new(&self.output_dir);
        let bytes = read_original_file_bounded(&root.join("model.trit"), self.max_package_bytes)
            .map_err(error)?;
        let text = read_original_file_bounded(&root.join("evidence.jsonl"), self.evidence_bytes)
            .map_err(error)?;
        let log = verify_jsonl(std::str::from_utf8(&text).map_err(error)?).map_err(error)?;
        verify_package_with_fit_log(&bytes, &log).map_err(error)?;
        let model = tritium_format::trit_model::read_trit_model(&bytes).map_err(error)?;
        if hex(model.package_id().as_bytes()) != self.package_id
            || log.root_digest != self.evidence_root
        {
            return Err(error("conversion package or evidence identity changed"));
        }
        Ok(bytes)
    }
}
#[pymethods]
impl NativeModelConversion {
    /// Reopen exact files and verify package/log binding against this result.
    fn verify(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.verified_bytes().map(|_| ()))
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
            let bytes = self.verified_bytes()?;
            let model = tritium_format::trit_model::read_trit_model(&bytes).map_err(error)?;
            let payload = model.asset(name).map_err(error)?;
            if payload.len() > max_bytes {
                return Err(error("asset exceeds byte limit"));
            }
            Ok::<_, PyErr>(payload.to_vec())
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
            let bytes = self.verified_bytes()?;
            let model = tritium_format::trit_model::read_trit_model(&bytes).map_err(error)?;
            let (meta, payload) = model.tensor(level, name).map_err(error)?;
            if payload.len() > max_bytes {
                return Err(error("preserved tensor exceeds byte limit"));
            }
            match &meta.storage {
                TensorStorage::Dense { precision, shape } => {
                    Ok((format!("{precision:?}"), shape.clone(), payload.to_vec()))
                }
                _ => Err(error("requested tensor is not dense-preserved")),
            }
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
    py.detach(move || {
        let mut calibration = file_calibration_source(request.view(), caches).map_err(error)?;
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
            ],
        )
        .map_err(error)?;
        let mut sink = DirectoryTensorSink::new(&output_dir);
        let result = match &request {
            Plan::Plain(plan) => Engine::run(plan, &mut source, &mut sink, recorder),
            Plan::Calibrated(plan) => {
                Engine::run_calibrated(plan, &mut source, &mut calibration, &mut sink, recorder)
            }
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
