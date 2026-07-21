//! Strict native admission and atomic publication for Qwen language-plus-MTP ONNX bundles.

use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use ort::{
    session::{Session, SessionInputValue},
    value::Tensor,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::json;
use tritium_nn::{
    Qwen35LayerType as NnQwen35LayerType, Qwen35SaltV2BundleAdmission, Qwen35TensorSchemaRole,
    qwen35_language_mtp_tensor_schema,
};
use tritium_onnx::{
    AdmittedExternalQwen35BundleDigests, OnnxArtifactIdentityV2, Qwen35Config, Qwen35LayerType,
    Qwen35OnnxAncestryV1, Qwen35PackageMatrixSpec, Qwen35PackagePreservedSpec,
    Qwen35PackageSourceSpec, Qwen35SaltV2PackageSource, QwenDeltaNetGeometry,
    VerifiedExternalQwen35Bundle, encode_external_qwen35_bundle_to_file_with_ancestry,
    map_qwen36_27b_packed_causal_lm, tritium_operator_domain,
    verify_external_qwen35_bundle_from_file,
};

const LANGUAGE_FILE: &str = "language.onnx";
const MTP_FILE: &str = "mtp.onnx";
const WEIGHTS_FILE: &str = "weights.bin";
const MANIFEST_FILE: &str = "tritium-onnx-manifest.json";
static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

const MAX_MANIFEST_BYTES: u64 = 1_048_576;
const MAX_GRAPH_BYTES: u64 = 268_435_456;
const MAX_WEIGHTS_BYTES: u64 = 68_719_476_736;

/// Native receipt proving both Qwen graphs and their shared external data were admitted.
#[pyclass(module = "tritium", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct QwenOnnxBundleReceipt {
    language_blake3: String,
    mtp_blake3: String,
    weights_blake3: String,
    weights_bytes: u64,
    language_tokens: u64,
    language_past_tokens: u64,
    language_layers: u64,
    mtp_tokens: u64,
    mtp_past_tokens: u64,
    mtp_layers: u64,
    source_model_id: String,
    tokenizer_id: String,
    recipe_id: String,
    package_id: String,
    converted_coverage_id: String,
    deferred_coverage_id: String,
    conversion_mode: Option<String>,
    completion_id: Option<String>,
    campaign_id: Option<String>,
    admission_id: Option<String>,
    selection_id: Option<String>,
}

#[pymethods]
impl QwenOnnxBundleReceipt {
    #[getter]
    fn language_blake3(&self) -> &str {
        &self.language_blake3
    }
    #[getter]
    fn mtp_blake3(&self) -> &str {
        &self.mtp_blake3
    }
    #[getter]
    fn weights_blake3(&self) -> &str {
        &self.weights_blake3
    }
    #[getter]
    const fn weights_bytes(&self) -> u64 {
        self.weights_bytes
    }
    #[getter]
    const fn language_tokens(&self) -> u64 {
        self.language_tokens
    }
    #[getter]
    const fn language_past_tokens(&self) -> u64 {
        self.language_past_tokens
    }
    #[getter]
    const fn language_layers(&self) -> u64 {
        self.language_layers
    }
    #[getter]
    const fn mtp_tokens(&self) -> u64 {
        self.mtp_tokens
    }
    #[getter]
    const fn mtp_past_tokens(&self) -> u64 {
        self.mtp_past_tokens
    }
    #[getter]
    const fn mtp_layers(&self) -> u64 {
        self.mtp_layers
    }
    #[getter]
    fn source_model_id(&self) -> &str {
        &self.source_model_id
    }
    #[getter]
    fn tokenizer_id(&self) -> &str {
        &self.tokenizer_id
    }
    #[getter]
    fn recipe_id(&self) -> &str {
        &self.recipe_id
    }
    #[getter]
    fn package_id(&self) -> &str {
        &self.package_id
    }
    #[getter]
    fn converted_coverage_id(&self) -> &str {
        &self.converted_coverage_id
    }
    #[getter]
    fn deferred_coverage_id(&self) -> &str {
        &self.deferred_coverage_id
    }
    #[getter]
    fn conversion_mode(&self) -> Option<&str> {
        self.conversion_mode.as_deref()
    }
    #[getter]
    fn completion_id(&self) -> Option<&str> {
        self.completion_id.as_deref()
    }
    #[getter]
    fn campaign_id(&self) -> Option<&str> {
        self.campaign_id.as_deref()
    }
    #[getter]
    fn admission_id(&self) -> Option<&str> {
        self.admission_id.as_deref()
    }
    #[getter]
    fn selection_id(&self) -> Option<&str> {
        self.selection_id.as_deref()
    }
}

/// Authenticated, file-backed ONNX Runtime sessions for one Qwen language-plus-MTP bundle.
///
/// Loading verifies the manifest and all external-data ranges, creates both real
/// ORT sessions with Tritium's custom operator domain, then verifies the bundle a
/// second time. Only CPU execution is admitted until wheel-specific provider
/// qualification lands.
#[pyclass(module = "tritium", frozen, skip_from_py_object)]
pub(crate) struct QwenOnnxModel {
    artifact_dir: String,
    device: String,
    receipt: QwenOnnxBundleReceipt,
    language_inputs: Vec<String>,
    language_outputs: Vec<String>,
    mtp_inputs: Vec<String>,
    mtp_outputs: Vec<String>,
    language_session: Mutex<Session>,
    mtp_session: Mutex<Session>,
}

/// Owned output of one fixed-shape Qwen language graph execution.
#[pyclass(module = "tritium", frozen, skip_from_py_object)]
pub(crate) struct QwenOnnxLanguageOutput {
    logits_shape: Vec<usize>,
    logits: Vec<f32>,
    state_names: Vec<String>,
    state_shapes: Vec<Vec<usize>>,
    states: Vec<Vec<f32>>,
}

/// Owned output of one fixed-shape Qwen MTP graph execution.
#[pyclass(module = "tritium", frozen, skip_from_py_object)]
pub(crate) struct QwenOnnxMtpOutput {
    logits_shape: Vec<usize>,
    logits: Vec<f32>,
    final_hidden_shape: Vec<usize>,
    final_hidden: Vec<f32>,
    state_names: Vec<String>,
    state_shapes: Vec<Vec<usize>>,
    states: Vec<Vec<f32>>,
}

#[pymethods]
impl QwenOnnxLanguageOutput {
    #[getter]
    fn logits_shape(&self) -> Vec<usize> {
        self.logits_shape.clone()
    }

    #[getter]
    fn logits(&self) -> Vec<f32> {
        self.logits.clone()
    }

    #[getter]
    fn state_names(&self) -> Vec<String> {
        self.state_names.clone()
    }

    #[getter]
    fn state_shapes(&self) -> Vec<Vec<usize>> {
        self.state_shapes.clone()
    }

    #[getter]
    fn states(&self) -> Vec<Vec<f32>> {
        self.states.clone()
    }
}

#[pymethods]
impl QwenOnnxMtpOutput {
    #[getter]
    fn logits_shape(&self) -> Vec<usize> {
        self.logits_shape.clone()
    }

    #[getter]
    fn logits(&self) -> Vec<f32> {
        self.logits.clone()
    }

    #[getter]
    fn final_hidden_shape(&self) -> Vec<usize> {
        self.final_hidden_shape.clone()
    }

    #[getter]
    fn final_hidden(&self) -> Vec<f32> {
        self.final_hidden.clone()
    }

    #[getter]
    fn state_names(&self) -> Vec<String> {
        self.state_names.clone()
    }

    #[getter]
    fn state_shapes(&self) -> Vec<Vec<usize>> {
        self.state_shapes.clone()
    }

    #[getter]
    fn states(&self) -> Vec<Vec<f32>> {
        self.states.clone()
    }
}

#[pymethods]
impl QwenOnnxModel {
    /// Strictly admit a published bundle and create both real ORT sessions.
    #[staticmethod]
    #[pyo3(signature = (artifact_dir, *, device = "cpu"))]
    fn load(py: Python<'_>, artifact_dir: &str, device: &str) -> PyResult<Self> {
        if artifact_dir.is_empty() {
            return Err(PyValueError::new_err("artifact_dir must not be empty"));
        }
        if device != "cpu" {
            return Err(PyValueError::new_err(format!(
                "unsupported ONNX device {device:?}; this wheel admits only 'cpu'"
            )));
        }
        let path = PathBuf::from(artifact_dir);
        py.detach(move || load_onnx_runtime(&path))
            .map_err(PyRuntimeError::new_err)
    }

    #[getter]
    fn artifact_dir(&self) -> &str {
        &self.artifact_dir
    }

    #[getter]
    fn device(&self) -> &str {
        &self.device
    }

    #[getter]
    fn receipt(&self) -> QwenOnnxBundleReceipt {
        self.receipt.clone()
    }

    #[getter]
    fn language_inputs(&self) -> Vec<String> {
        self.language_inputs.clone()
    }

    #[getter]
    fn language_outputs(&self) -> Vec<String> {
        self.language_outputs.clone()
    }

    #[getter]
    fn mtp_inputs(&self) -> Vec<String> {
        self.mtp_inputs.clone()
    }

    #[getter]
    fn mtp_outputs(&self) -> Vec<String> {
        self.mtp_outputs.clone()
    }

    /// Execute the authenticated fixed-shape language graph.
    ///
    /// `states` is ordered exactly like `language_inputs[1:]`. It may be
    /// omitted only for a graph whose authenticated `past_tokens` is zero, in
    /// which case recurrent and empty KV inputs are initialized to zero.
    #[pyo3(signature = (token_ids, states = None))]
    fn forward_language(
        &self,
        py: Python<'_>,
        token_ids: Vec<i64>,
        states: Option<Vec<Vec<f32>>>,
    ) -> PyResult<QwenOnnxLanguageOutput> {
        if token_ids.len()
            != usize::try_from(self.receipt.language_tokens).map_err(|_| {
                PyValueError::new_err("authenticated language token count exceeds platform bounds")
            })?
        {
            return Err(PyValueError::new_err(format!(
                "token_ids has length {}, expected {} for this fixed-shape graph",
                token_ids.len(),
                self.receipt.language_tokens
            )));
        }
        if states.is_none() && self.receipt.language_past_tokens != 0 {
            return Err(PyValueError::new_err(
                "states are required when the authenticated graph has past_tokens > 0",
            ));
        }
        py.detach(|| run_language(self, token_ids, states))
            .map_err(PyRuntimeError::new_err)
    }

    /// Execute the authenticated fixed-shape bundled MTP drafter graph.
    ///
    /// `target_hidden` is flattened row-major from `[tokens, hidden]`. `states`
    /// follows `mtp_inputs[2:]` and may be omitted only for a prompt graph.
    #[pyo3(signature = (token_ids, target_hidden, states = None))]
    fn forward_mtp(
        &self,
        py: Python<'_>,
        token_ids: Vec<i64>,
        target_hidden: Vec<f32>,
        states: Option<Vec<Vec<f32>>>,
    ) -> PyResult<QwenOnnxMtpOutput> {
        if token_ids.len()
            != usize::try_from(self.receipt.mtp_tokens).map_err(|_| {
                PyValueError::new_err("authenticated MTP token count exceeds platform bounds")
            })?
        {
            return Err(PyValueError::new_err(format!(
                "token_ids has length {}, expected {} for this fixed-shape MTP graph",
                token_ids.len(),
                self.receipt.mtp_tokens
            )));
        }
        if states.is_none() && self.receipt.mtp_past_tokens != 0 {
            return Err(PyValueError::new_err(
                "states are required when the authenticated MTP graph has past_tokens > 0",
            ));
        }
        py.detach(|| run_mtp(self, token_ids, target_hidden, states))
            .map_err(PyRuntimeError::new_err)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PublishedManifest {
    schema: String,
    language: ManifestGraph,
    mtp: ManifestGraph,
    weights: ManifestWeights,
    identity: ManifestIdentity,
    conversion: ManifestConversion,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestGraph {
    file: String,
    blake3: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestWeights {
    file: String,
    blake3: String,
    bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestIdentity {
    source_model_id: String,
    tokenizer_id: String,
    recipe_id: String,
    package_id: String,
    converted_coverage_id: String,
    deferred_coverage_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestConversion {
    mode: String,
    completion_id: String,
    campaign_id: String,
    admission_id: String,
    selection_id: String,
}

fn load_onnx_runtime(requested: &Path) -> Result<QwenOnnxModel, String> {
    let before = fs::symlink_metadata(requested)
        .map_err(|error| format!("inspect ONNX artifact directory failed: {:?}", error.kind()))?;
    if before.file_type().is_symlink() || !before.is_dir() {
        return Err("ONNX artifact must be an ordinary non-symlink directory".to_owned());
    }
    let directory = fs::canonicalize(requested).map_err(|error| {
        format!(
            "canonicalize ONNX artifact directory failed: {:?}",
            error.kind()
        )
    })?;
    let opened = fs::metadata(&directory).map_err(|error| {
        format!(
            "inspect canonical ONNX artifact directory failed: {:?}",
            error.kind()
        )
    })?;
    if !opened.is_dir() || !same_file(&before, &opened) {
        return Err("ONNX artifact directory changed during canonicalization".to_owned());
    }

    let manifest = read_published_manifest(&directory)?;
    let inputs = manifest_inputs(&directory, &manifest)?;
    let receipt = verify_paths(&inputs)?;
    validate_manifest_receipt(&manifest, &receipt)?;

    let language_session =
        Session::builder()
            .map_err(|error| format!("create language ORT session builder failed: {error}"))?
            .with_operators(tritium_operator_domain().map_err(|error| {
                format!("create language Tritium operator domain failed: {error}")
            })?)
            .map_err(|error| format!("register language Tritium operators failed: {error}"))?
            .commit_from_file(&inputs.language)
            .map_err(|error| format!("open authenticated language ORT graph failed: {error}"))?;
    let mtp_session = Session::builder()
        .map_err(|error| format!("create MTP ORT session builder failed: {error}"))?
        .with_operators(
            tritium_operator_domain()
                .map_err(|error| format!("create MTP Tritium operator domain failed: {error}"))?,
        )
        .map_err(|error| format!("register MTP Tritium operators failed: {error}"))?
        .commit_from_file(&inputs.mtp)
        .map_err(|error| format!("open authenticated MTP ORT graph failed: {error}"))?;

    let language_inputs = outlet_names(language_session.inputs());
    let language_outputs = outlet_names(language_session.outputs());
    let mtp_inputs = outlet_names(mtp_session.inputs());
    let mtp_outputs = outlet_names(mtp_session.outputs());
    validate_session_interface(
        &receipt,
        &language_inputs,
        &language_outputs,
        &mtp_inputs,
        &mtp_outputs,
    )?;

    // ORT resolves external data while committing each session. Re-admission
    // after both commits catches replacement or mutation during that window.
    let after = verify_paths(&inputs)?;
    if !same_receipt(&receipt, &after) {
        return Err("ONNX bundle changed while ORT sessions were being created".to_owned());
    }
    validate_manifest_receipt(&manifest, &after)?;
    let final_directory = fs::metadata(&directory).map_err(|error| {
        format!(
            "reinspect ONNX artifact directory failed: {:?}",
            error.kind()
        )
    })?;
    if !same_file(&opened, &final_directory) {
        return Err("ONNX artifact directory changed while loading".to_owned());
    }
    let artifact_dir = directory
        .to_str()
        .ok_or_else(|| "ONNX artifact path must be valid UTF-8".to_owned())?
        .to_owned();

    Ok(QwenOnnxModel {
        artifact_dir,
        device: "cpu".to_owned(),
        receipt,
        language_inputs,
        language_outputs,
        mtp_inputs,
        mtp_outputs,
        language_session: Mutex::new(language_session),
        mtp_session: Mutex::new(mtp_session),
    })
}

fn run_language(
    model: &QwenOnnxModel,
    token_ids: Vec<i64>,
    states: Option<Vec<Vec<f32>>>,
) -> Result<QwenOnnxLanguageOutput, String> {
    let mut session = model
        .language_session
        .lock()
        .map_err(|_| "language ORT session lock was poisoned".to_owned())?;
    let state_shapes = session
        .inputs()
        .iter()
        .skip(1)
        .map(|outlet| fixed_f32_shape(outlet.dtype(), outlet.name()))
        .collect::<Result<Vec<_>, _>>()?;
    let states = match states {
        Some(states) => {
            if states.len() != state_shapes.len() {
                return Err(format!(
                    "language state count is {}, expected {}",
                    states.len(),
                    state_shapes.len()
                ));
            }
            for (index, (values, shape)) in states.iter().zip(&state_shapes).enumerate() {
                let expected = shape_elements(shape, "language state")?;
                if values.len() != expected {
                    return Err(format!(
                        "language state {index} has {} values, expected {expected}",
                        values.len()
                    ));
                }
            }
            states
        }
        None => state_shapes
            .iter()
            .map(|shape| {
                shape_elements(shape, "language state").map(|elements| vec![0.0_f32; elements])
            })
            .collect::<Result<Vec<_>, _>>()?,
    };

    let tokens = Tensor::from_array((vec![token_ids.len()], token_ids.into_boxed_slice()))
        .map_err(|error| format!("create language token tensor failed: {error}"))?;
    let mut inputs: Vec<SessionInputValue<'_>> = Vec::with_capacity(1 + states.len());
    inputs.push(tokens.into_dyn().into());
    for (shape, values) in state_shapes.iter().cloned().zip(states) {
        let tensor = Tensor::from_array((shape, values.into_boxed_slice()))
            .map_err(|error| format!("create language state tensor failed: {error}"))?;
        inputs.push(tensor.into_dyn().into());
    }
    let outputs = session
        .run(inputs.as_slice())
        .map_err(|error| format!("execute authenticated language ORT graph failed: {error}"))?;
    let (logits_shape, logits) = extract_f32_output(&outputs, "logits", "language logits")?;
    let state_names = model
        .language_outputs
        .iter()
        .skip(1)
        .cloned()
        .collect::<Vec<_>>();
    let mut output_shapes = Vec::with_capacity(state_names.len());
    let mut output_states = Vec::with_capacity(state_names.len());
    for name in &state_names {
        let (shape, values) = extract_f32_output(&outputs, name, "language state output")?;
        output_shapes.push(shape);
        output_states.push(values);
    }
    Ok(QwenOnnxLanguageOutput {
        logits_shape,
        logits,
        state_names,
        state_shapes: output_shapes,
        states: output_states,
    })
}

fn run_mtp(
    model: &QwenOnnxModel,
    token_ids: Vec<i64>,
    target_hidden: Vec<f32>,
    states: Option<Vec<Vec<f32>>>,
) -> Result<QwenOnnxMtpOutput, String> {
    let mut session = model
        .mtp_session
        .lock()
        .map_err(|_| "MTP ORT session lock was poisoned".to_owned())?;
    let target_input = session
        .inputs()
        .get(1)
        .ok_or_else(|| "authenticated MTP session has no target_hidden input".to_owned())?;
    let target_shape = fixed_f32_shape(target_input.dtype(), "target_hidden")?;
    let expected_hidden = shape_elements(&target_shape, "MTP target hidden")?;
    if target_hidden.len() != expected_hidden {
        return Err(format!(
            "MTP target hidden has {} values, expected {expected_hidden}",
            target_hidden.len()
        ));
    }
    let state_shapes = session
        .inputs()
        .iter()
        .skip(2)
        .map(|outlet| fixed_f32_shape(outlet.dtype(), outlet.name()))
        .collect::<Result<Vec<_>, _>>()?;
    let states = prepare_states(states, &state_shapes, "MTP")?;

    let tokens = Tensor::from_array((vec![token_ids.len()], token_ids.into_boxed_slice()))
        .map_err(|error| format!("create MTP token tensor failed: {error}"))?;
    let hidden = Tensor::from_array((target_shape, target_hidden.into_boxed_slice()))
        .map_err(|error| format!("create MTP target hidden tensor failed: {error}"))?;
    let mut inputs: Vec<SessionInputValue<'_>> = Vec::with_capacity(2 + states.len());
    inputs.push(tokens.into_dyn().into());
    inputs.push(hidden.into_dyn().into());
    for (shape, values) in state_shapes.iter().cloned().zip(states) {
        let tensor = Tensor::from_array((shape, values.into_boxed_slice()))
            .map_err(|error| format!("create MTP state tensor failed: {error}"))?;
        inputs.push(tensor.into_dyn().into());
    }
    let outputs = session
        .run(inputs.as_slice())
        .map_err(|error| format!("execute authenticated MTP ORT graph failed: {error}"))?;
    let (logits_shape, logits) = extract_f32_output(&outputs, "mtp.logits", "MTP logits")?;
    let (final_hidden_shape, final_hidden) =
        extract_f32_output(&outputs, "mtp.final_hidden", "MTP final hidden")?;
    let state_names = model
        .mtp_outputs
        .iter()
        .skip(2)
        .cloned()
        .collect::<Vec<_>>();
    let mut output_shapes = Vec::with_capacity(state_names.len());
    let mut output_states = Vec::with_capacity(state_names.len());
    for name in &state_names {
        let (shape, values) = extract_f32_output(&outputs, name, "MTP state output")?;
        output_shapes.push(shape);
        output_states.push(values);
    }
    Ok(QwenOnnxMtpOutput {
        logits_shape,
        logits,
        final_hidden_shape,
        final_hidden,
        state_names,
        state_shapes: output_shapes,
        states: output_states,
    })
}

fn extract_f32_output(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
    label: &str,
) -> Result<(Vec<usize>, Vec<f32>), String> {
    let output = outputs
        .get(name)
        .ok_or_else(|| format!("authenticated ORT session did not return {name:?}"))?;
    let (shape, values) = output
        .try_extract_tensor::<f32>()
        .map_err(|error| format!("extract {label} failed: {error}"))?;
    Ok((runtime_shape(shape, label)?, values.to_vec()))
}

fn prepare_states(
    states: Option<Vec<Vec<f32>>>,
    state_shapes: &[Vec<usize>],
    label: &str,
) -> Result<Vec<Vec<f32>>, String> {
    match states {
        Some(states) => {
            if states.len() != state_shapes.len() {
                return Err(format!(
                    "{label} state count is {}, expected {}",
                    states.len(),
                    state_shapes.len()
                ));
            }
            for (index, (values, shape)) in states.iter().zip(state_shapes).enumerate() {
                let expected = shape_elements(shape, "state")?;
                if values.len() != expected {
                    return Err(format!(
                        "{label} state {index} has {} values, expected {expected}",
                        values.len()
                    ));
                }
            }
            Ok(states)
        }
        None => state_shapes
            .iter()
            .map(|shape| shape_elements(shape, "state").map(|elements| vec![0.0; elements]))
            .collect(),
    }
}

fn fixed_f32_shape(dtype: &ort::value::ValueType, name: &str) -> Result<Vec<usize>, String> {
    if dtype.tensor_type() != Some(ort::value::TensorElementType::Float32) {
        return Err(format!("ONNX input {name:?} is not float32"));
    }
    let shape = dtype
        .tensor_shape()
        .ok_or_else(|| format!("language state input {name:?} is not a tensor"))?;
    runtime_shape(shape, name)
}

fn runtime_shape(shape: &[i64], label: &str) -> Result<Vec<usize>, String> {
    shape
        .iter()
        .map(|&axis| {
            usize::try_from(axis)
                .map_err(|_| format!("{label} contains a dynamic or negative dimension"))
        })
        .collect()
}

fn shape_elements(shape: &[usize], label: &str) -> Result<usize, String> {
    shape.iter().try_fold(1_usize, |elements, &axis| {
        elements
            .checked_mul(axis)
            .ok_or_else(|| format!("{label} element count overflow"))
    })
}

fn read_published_manifest(directory: &Path) -> Result<PublishedManifest, String> {
    let bytes = read_regular_bounded(
        &directory.join(MANIFEST_FILE),
        MAX_MANIFEST_BYTES,
        "ONNX manifest",
    )?;
    let manifest: PublishedManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse strict ONNX manifest failed: {error}"))?;
    if manifest.schema != "tritium-qwen35-onnx-bundle-v1" {
        return Err("unsupported ONNX bundle schema".to_owned());
    }
    if manifest.language.file != LANGUAGE_FILE
        || manifest.mtp.file != MTP_FILE
        || manifest.weights.file != WEIGHTS_FILE
    {
        return Err("ONNX manifest filenames are not canonical".to_owned());
    }
    if manifest.weights.bytes == 0 || manifest.weights.bytes > MAX_WEIGHTS_BYTES {
        return Err("ONNX manifest weights byte count is outside runtime bounds".to_owned());
    }
    if !matches!(
        manifest.conversion.mode.as_str(),
        "qat-hard" | "ptq" | "refined"
    ) {
        return Err("unsupported ONNX conversion mode".to_owned());
    }
    Ok(manifest)
}

fn manifest_inputs(directory: &Path, manifest: &PublishedManifest) -> Result<Inputs, String> {
    Ok(Inputs {
        language: directory.join(LANGUAGE_FILE),
        mtp: directory.join(MTP_FILE),
        weights: directory.join(WEIGHTS_FILE),
        admitted: AdmittedExternalQwen35BundleDigests {
            language_model_blake3: parse_manifest_digest(
                &manifest.language.blake3,
                "language.blake3",
            )?,
            mtp_model_blake3: parse_manifest_digest(&manifest.mtp.blake3, "mtp.blake3")?,
            weights_blake3: parse_manifest_digest(&manifest.weights.blake3, "weights.blake3")?,
        },
        max_graph_bytes: MAX_GRAPH_BYTES,
        max_weights_bytes: manifest.weights.bytes,
    })
}

fn parse_manifest_digest(value: &str, label: &str) -> Result<[u8; 32], String> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
    {
        return Err(format!(
            "{label} must be 64 lowercase hexadecimal characters"
        ));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        digest[index] = u8::from_str_radix(
            std::str::from_utf8(pair).expect("validated hexadecimal is ASCII"),
            16,
        )
        .map_err(|_| format!("{label} is not hexadecimal"))?;
    }
    Ok(digest)
}

fn validate_manifest_receipt(
    manifest: &PublishedManifest,
    receipt: &QwenOnnxBundleReceipt,
) -> Result<(), String> {
    let identity_matches = manifest.identity.source_model_id == receipt.source_model_id
        && manifest.identity.tokenizer_id == receipt.tokenizer_id
        && manifest.identity.recipe_id == receipt.recipe_id
        && manifest.identity.package_id == receipt.package_id
        && manifest.identity.converted_coverage_id == receipt.converted_coverage_id
        && manifest.identity.deferred_coverage_id == receipt.deferred_coverage_id;
    let conversion_matches = receipt.conversion_mode.as_deref() == Some(&manifest.conversion.mode)
        && receipt.completion_id.as_deref() == Some(&manifest.conversion.completion_id)
        && receipt.campaign_id.as_deref() == Some(&manifest.conversion.campaign_id)
        && receipt.admission_id.as_deref() == Some(&manifest.conversion.admission_id)
        && receipt.selection_id.as_deref() == Some(&manifest.conversion.selection_id);
    if manifest.language.blake3 != receipt.language_blake3
        || manifest.mtp.blake3 != receipt.mtp_blake3
        || manifest.weights.blake3 != receipt.weights_blake3
        || manifest.weights.bytes != receipt.weights_bytes
        || !identity_matches
        || !conversion_matches
    {
        return Err("ONNX manifest does not match authenticated graph metadata".to_owned());
    }
    Ok(())
}

fn outlet_names(outlets: &[ort::value::Outlet]) -> Vec<String> {
    outlets
        .iter()
        .map(|outlet| outlet.name().to_owned())
        .collect()
}

fn validate_session_interface(
    receipt: &QwenOnnxBundleReceipt,
    language_inputs: &[String],
    language_outputs: &[String],
    mtp_inputs: &[String],
    mtp_outputs: &[String],
) -> Result<(), String> {
    let language_layers = usize::try_from(receipt.language_layers)
        .map_err(|_| "language layer count exceeds platform bounds".to_owned())?;
    let delta_layers = paired_state_indices(language_outputs, "next_conv.", "next_recurrent.")?;
    let attention_layers = paired_state_indices(language_outputs, "present_k.", "present_v.")?;
    if delta_layers.len().checked_add(attention_layers.len()) != Some(language_layers)
        || !delta_layers.is_disjoint(&attention_layers)
    {
        return Err("ORT language state outputs do not cover each layer exactly once".to_owned());
    }
    let mut expected_language_inputs = 1_usize
        .checked_add(
            delta_layers
                .len()
                .checked_mul(2)
                .ok_or_else(|| "language DeltaNet input count overflow".to_owned())?,
        )
        .ok_or_else(|| "language input count overflow".to_owned())?;
    if receipt.language_past_tokens != 0 {
        expected_language_inputs = expected_language_inputs
            .checked_add(
                attention_layers
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| "language attention input count overflow".to_owned())?,
            )
            .ok_or_else(|| "language input count overflow".to_owned())?;
    }
    let expected_language_outputs = 1_usize
        .checked_add(
            language_layers
                .checked_mul(2)
                .ok_or_else(|| "language output count overflow".to_owned())?,
        )
        .ok_or_else(|| "language output count overflow".to_owned())?;
    let expected_mtp_inputs = if receipt.mtp_past_tokens == 0 { 2 } else { 4 };
    let expected_mtp_outputs = 4;
    if language_inputs.first().map(String::as_str) != Some("tokens")
        || language_outputs.first().map(String::as_str) != Some("logits")
        || mtp_inputs.first().map(String::as_str) != Some("shifted_tokens")
        || mtp_inputs.get(1).map(String::as_str) != Some("target_hidden")
        || mtp_outputs.first().map(String::as_str) != Some("mtp.logits")
        || mtp_outputs.get(1).map(String::as_str) != Some("mtp.final_hidden")
        || language_inputs.len() != expected_language_inputs
        || language_outputs.len() != expected_language_outputs
        || mtp_inputs.len() != expected_mtp_inputs
        || mtp_outputs.len() != expected_mtp_outputs
        || !required_state_inputs(
            language_inputs,
            &delta_layers,
            &attention_layers,
            receipt.language_past_tokens != 0,
        )
        || (receipt.mtp_past_tokens != 0
            && (mtp_inputs.get(2).map(String::as_str) != Some("past_k.0")
                || mtp_inputs.get(3).map(String::as_str) != Some("past_v.0")))
        || mtp_outputs.get(2).map(String::as_str) != Some("present_k.0")
        || mtp_outputs.get(3).map(String::as_str) != Some("present_v.0")
    {
        return Err("ORT session interface differs from authenticated Qwen schema".to_owned());
    }
    Ok(())
}

fn paired_state_indices(
    names: &[String],
    left_prefix: &str,
    right_prefix: &str,
) -> Result<std::collections::BTreeSet<usize>, String> {
    let parse = |prefix: &str| -> Result<std::collections::BTreeSet<usize>, String> {
        names
            .iter()
            .filter_map(|name| name.strip_prefix(prefix))
            .map(|suffix| {
                suffix.parse::<usize>().map_err(|_| {
                    format!("ORT state output {prefix}{suffix} has invalid layer index")
                })
            })
            .collect()
    };
    let left = parse(left_prefix)?;
    let right = parse(right_prefix)?;
    if left != right {
        return Err(format!(
            "ORT state outputs {left_prefix}* and {right_prefix}* are not paired"
        ));
    }
    Ok(left)
}

fn required_state_inputs(
    names: &[String],
    delta_layers: &std::collections::BTreeSet<usize>,
    attention_layers: &std::collections::BTreeSet<usize>,
    cached: bool,
) -> bool {
    let names = names
        .iter()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    delta_layers.iter().all(|index| {
        names.contains(format!("conv_state.{index}").as_str())
            && names.contains(format!("recurrent_state.{index}").as_str())
    }) && (!cached
        || attention_layers.iter().all(|index| {
            names.contains(format!("past_k.{index}").as_str())
                && names.contains(format!("past_v.{index}").as_str())
        }))
}

fn same_receipt(left: &QwenOnnxBundleReceipt, right: &QwenOnnxBundleReceipt) -> bool {
    left.language_blake3 == right.language_blake3
        && left.mtp_blake3 == right.mtp_blake3
        && left.weights_blake3 == right.weights_blake3
        && left.weights_bytes == right.weights_bytes
        && left.language_tokens == right.language_tokens
        && left.language_past_tokens == right.language_past_tokens
        && left.language_layers == right.language_layers
        && left.mtp_tokens == right.mtp_tokens
        && left.mtp_past_tokens == right.mtp_past_tokens
        && left.mtp_layers == right.mtp_layers
        && left.source_model_id == right.source_model_id
        && left.tokenizer_id == right.tokenizer_id
        && left.recipe_id == right.recipe_id
        && left.package_id == right.package_id
        && left.converted_coverage_id == right.converted_coverage_id
        && left.deferred_coverage_id == right.deferred_coverage_id
        && left.conversion_mode == right.conversion_mode
        && left.completion_id == right.completion_id
        && left.campaign_id == right.campaign_id
        && left.admission_id == right.admission_id
        && left.selection_id == right.selection_id
}

/// Verify an existing three-file Qwen ONNX bundle against independently admitted digests.
#[pyfunction]
#[pyo3(signature = (language_path, mtp_path, weights_path, *, language_blake3, mtp_blake3, weights_blake3, max_graph_bytes = 268_435_456, max_weights_bytes = 68_719_476_736))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_qwen35_onnx_bundle(
    py: Python<'_>,
    language_path: &str,
    mtp_path: &str,
    weights_path: &str,
    language_blake3: &str,
    mtp_blake3: &str,
    weights_blake3: &str,
    max_graph_bytes: u64,
    max_weights_bytes: u64,
) -> PyResult<QwenOnnxBundleReceipt> {
    let inputs = Inputs::parse(
        language_path,
        mtp_path,
        weights_path,
        language_blake3,
        mtp_blake3,
        weights_blake3,
        max_graph_bytes,
        max_weights_bytes,
    )?;
    py.detach(move || verify_paths(&inputs))
        .map_err(PyValueError::new_err)
}

/// Verify, durably stage, and atomically publish a Qwen ONNX bundle without replacement.
#[pyfunction]
#[pyo3(signature = (language_path, mtp_path, weights_path, output_dir, *, language_blake3, mtp_blake3, weights_blake3, max_graph_bytes = 268_435_456, max_weights_bytes = 68_719_476_736))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn stage_qwen35_onnx_bundle(
    py: Python<'_>,
    language_path: &str,
    mtp_path: &str,
    weights_path: &str,
    output_dir: &str,
    language_blake3: &str,
    mtp_blake3: &str,
    weights_blake3: &str,
    max_graph_bytes: u64,
    max_weights_bytes: u64,
) -> PyResult<QwenOnnxBundleReceipt> {
    let inputs = Inputs::parse(
        language_path,
        mtp_path,
        weights_path,
        language_blake3,
        mtp_blake3,
        weights_blake3,
        max_graph_bytes,
        max_weights_bytes,
    )?;
    if output_dir.is_empty() {
        return Err(PyValueError::new_err("output_dir must not be empty"));
    }
    let output = PathBuf::from(output_dir);
    py.detach(move || stage_paths(&inputs, &output))
        .map_err(PyRuntimeError::new_err)
}

/// Export one authenticated schema-v3 Qwen PTQ bundle into canonical ONNX files.
#[pyfunction]
#[pyo3(signature = (bundle_dir, output_dir, *, profile = "compact-v1", tokens = 1, past_tokens = 0, max_package_bytes = 34_359_738_368, max_preserved_bytes = 8_589_934_592, max_salt_resident_bytes = 34_359_738_368, max_preserved_fp32_bytes = 8_589_934_592))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn export_qwen35_onnx_bundle(
    py: Python<'_>,
    bundle_dir: &str,
    output_dir: &str,
    profile: &str,
    tokens: usize,
    past_tokens: usize,
    max_package_bytes: u64,
    max_preserved_bytes: u64,
    max_salt_resident_bytes: u64,
    max_preserved_fp32_bytes: u64,
) -> PyResult<QwenOnnxBundleReceipt> {
    if bundle_dir.is_empty() || output_dir.is_empty() {
        return Err(PyValueError::new_err(
            "bundle_dir and output_dir must not be empty",
        ));
    }
    if tokens == 0 {
        return Err(PyValueError::new_err("tokens must be positive"));
    }
    if [
        max_package_bytes,
        max_preserved_bytes,
        max_salt_resident_bytes,
        max_preserved_fp32_bytes,
    ]
    .contains(&0)
    {
        return Err(PyValueError::new_err(
            "ONNX export byte ceilings must be positive",
        ));
    }
    let request = ExportRequest {
        bundle: PathBuf::from(bundle_dir),
        output: PathBuf::from(output_dir),
        profile: profile.to_owned(),
        tokens,
        past_tokens,
        max_package_bytes,
        max_preserved_bytes,
        max_salt_resident_bytes,
        max_preserved_fp32_bytes,
    };
    py.detach(move || export_paths(&request))
        .map_err(PyRuntimeError::new_err)
}

struct ExportRequest {
    bundle: PathBuf,
    output: PathBuf,
    profile: String,
    tokens: usize,
    past_tokens: usize,
    max_package_bytes: u64,
    max_preserved_bytes: u64,
    max_salt_resident_bytes: u64,
    max_preserved_fp32_bytes: u64,
}

fn export_paths(request: &ExportRequest) -> Result<QwenOnnxBundleReceipt, String> {
    let (parent, output) = canonical_output(&request.output)?;
    let admission = Qwen35SaltV2BundleAdmission::admit(&request.bundle, &request.profile)
        .map_err(|error| error.to_string())?;
    for (label, actual, limit) in [
        (
            "SALT package",
            admission.serialized_bytes(),
            request.max_package_bytes,
        ),
        (
            "preserved snapshot",
            admission.preserved_serialized_bytes(),
            request.max_preserved_bytes,
        ),
        (
            "SALT resident arena",
            admission.salt_resident_bytes(),
            request.max_salt_resident_bytes,
        ),
        (
            "preserved fp32 arena",
            admission.preserved_fp32_bytes(),
            request.max_preserved_fp32_bytes,
        ),
    ] {
        if actual > limit {
            return Err(format!(
                "{label} requires {actual} bytes, exceeds caller ceiling {limit}"
            ));
        }
    }
    let schema = qwen35_language_mtp_tensor_schema(&admission.config().text)
        .map_err(|error| error.to_string())?;
    let matrices = schema
        .iter()
        .filter_map(|tensor| match tensor.role {
            Qwen35TensorSchemaRole::Matrix => Some(Qwen35PackageMatrixSpec {
                name: &tensor.name,
                rows: tensor.shape[0],
                columns: tensor.shape[1],
            }),
            Qwen35TensorSchemaRole::Preserved => None,
        })
        .collect::<Vec<_>>();
    let preserved = schema
        .iter()
        .filter_map(|tensor| match tensor.role {
            Qwen35TensorSchemaRole::Matrix => None,
            Qwen35TensorSchemaRole::Preserved => Some(Qwen35PackagePreservedSpec {
                name: &tensor.name,
                shape: &tensor.shape,
            }),
        })
        .collect::<Vec<_>>();
    if matrices.len() != admission.matrix_tensors()
        || preserved.len() != admission.preserved_tensors()
    {
        return Err("admitted Qwen tensor counts differ from public schema".to_owned());
    }
    let package = open_regular_handle(
        &request.bundle.join(admission.profile_file()),
        admission.serialized_bytes(),
        "SALT V2 profile",
    )?;
    let preserved_file = open_regular_handle(
        &request.bundle.join(admission.preserved_file()),
        admission.preserved_serialized_bytes(),
        "preserved safetensors",
    )?;
    let source = Qwen35SaltV2PackageSource::from_files(
        package,
        preserved_file,
        admission.package_id(),
        admission.preserved_package_id(),
        Qwen35PackageSourceSpec {
            matrices: &matrices,
            preserved: &preserved,
            package_bytes: admission.serialized_bytes(),
            max_package_bytes: request.max_package_bytes,
            preserved_bytes: admission.preserved_serialized_bytes(),
            max_preserved_snapshot_bytes: request.max_preserved_bytes,
            salt_resident_bytes: admission.salt_resident_bytes(),
            max_salt_resident_bytes: request.max_salt_resident_bytes,
            preserved_fp32_bytes: admission.preserved_fp32_bytes(),
            max_preserved_fp32_bytes: request.max_preserved_fp32_bytes,
        },
    )
    .map_err(|error| error.to_string())?;

    let text = &admission.config().text;
    let layer_types = text
        .layer_types
        .iter()
        .map(|layer| match layer {
            NnQwen35LayerType::DeltaNet => Qwen35LayerType::DeltaNet,
            NnQwen35LayerType::FullAttention => Qwen35LayerType::FullAttention,
        })
        .collect::<Vec<_>>();
    let config = Qwen35Config {
        hidden: axis(text.hidden_size, "hidden_size")?,
        intermediate: axis(text.intermediate_size, "intermediate_size")?,
        vocab: axis(text.vocab_size, "vocab_size")?,
        n_head: axis(text.full_attention.num_heads, "num_heads")?,
        n_kv_head: axis(
            text.full_attention.num_key_value_heads,
            "num_key_value_heads",
        )?,
        head_dim: axis(text.full_attention.head_dim, "head_dim")?,
        rotary_dim: axis(text.rope.rotary_dim, "rotary_dim")?,
        rope_theta: text.rope.theta as f32,
        rms_epsilon: text.rms_norm_eps as f32,
        delta_geometry: QwenDeltaNetGeometry::new(
            axis(text.delta_net.conv_kernel_dim, "conv_kernel_dim")?,
            axis(text.delta_net.num_key_heads, "num_key_heads")?,
            axis(text.delta_net.num_value_heads, "num_value_heads")?,
            axis(text.delta_net.key_head_dim, "key_head_dim")?,
            axis(text.delta_net.value_head_dim, "value_head_dim")?,
        )
        .map_err(|error| error.to_string())?,
        layer_types: &layer_types,
        full_attention_interval: axis(text.full_attention_interval, "full_attention_interval")?,
        tied_embeddings: text.tied_embeddings,
        mtp_layers: axis(text.mtp.num_hidden_layers, "mtp.num_hidden_layers")?,
        mtp_dedicated_embeddings: text.mtp.dedicated_embeddings,
    };
    let source_model_id = format!(
        "{}@{}",
        admission.source_model_id(),
        admission.source_revision()
    );
    let tokenizer_id = format!("{}#hf-tokenizer-assets", admission.manifest_package_id());
    let recipe_id = format!("tritium.s2kf-additive-ptq@1/{}", admission.profile());
    let build_id = env!("TRITIUM_SOURCE_ID");
    let converted_coverage_id = format!("{}#language-mtp", admission.manifest_package_id());
    let deferred_coverage_id = format!("{}#vision", admission.manifest_package_id());
    let mapped = map_qwen36_27b_packed_causal_lm(
        &source,
        config,
        OnnxArtifactIdentityV2 {
            source_model_id: &source_model_id,
            tokenizer_id: &tokenizer_id,
            recipe_id: &recipe_id,
            tritium_build_id: build_id,
            package_id: admission.package_id(),
            converted_coverage_id: &converted_coverage_id,
            deferred_coverage_id: &deferred_coverage_id,
        },
    )
    .map_err(|error| error.to_string())?;

    let staging = create_staging_directory(&parent)?;
    let result = (|| {
        let emitted = encode_external_qwen35_bundle_to_file_with_ancestry(
            mapped.model(request.tokens, request.past_tokens),
            mapped.mtp_model(request.tokens, request.past_tokens),
            Qwen35OnnxAncestryV1 {
                conversion_mode: "ptq",
                completion_id: admission.completion_id(),
                campaign_id: admission.campaign_id(),
                admission_id: admission.admission_id(),
                selection_id: admission.selection_id(),
            },
            File::create_new(staging.join(WEIGHTS_FILE)).map_err(|error| {
                format!("create staged external weights failed: {:?}", error.kind())
            })?,
        )
        .map_err(|error| error.to_string())?;
        write_sync(
            &staging.join(LANGUAGE_FILE),
            &emitted.language_model_bytes,
            "language graph",
        )?;
        write_sync(
            &staging.join(MTP_FILE),
            &emitted.mtp_model_bytes,
            "MTP graph",
        )?;
        let admitted = AdmittedExternalQwen35BundleDigests {
            language_model_blake3: *blake3::hash(&emitted.language_model_bytes).as_bytes(),
            mtp_model_blake3: *blake3::hash(&emitted.mtp_model_bytes).as_bytes(),
            weights_blake3: emitted.weights_blake3,
        };
        let verified = verify_external_qwen35_bundle_from_file(
            &emitted.language_model_bytes,
            &emitted.mtp_model_bytes,
            open_regular_handle(
                &staging.join(WEIGHTS_FILE),
                emitted.weights_bytes,
                "staged external weights",
            )?,
            admitted,
        )
        .map_err(|error| error.to_string())?;
        let receipt = receipt(&verified, admitted)?;
        write_manifest(&staging, &receipt)?;
        publish_staging(&staging, &parent, &output)?;
        Ok(receipt)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

fn axis(value: u32, name: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("Qwen {name} exceeds host usize"))
}

struct Inputs {
    language: PathBuf,
    mtp: PathBuf,
    weights: PathBuf,
    admitted: AdmittedExternalQwen35BundleDigests,
    max_graph_bytes: u64,
    max_weights_bytes: u64,
}

impl Inputs {
    #[allow(clippy::too_many_arguments)]
    fn parse(
        language: &str,
        mtp: &str,
        weights: &str,
        language_digest: &str,
        mtp_digest: &str,
        weights_digest: &str,
        max_graph_bytes: u64,
        max_weights_bytes: u64,
    ) -> PyResult<Self> {
        if language.is_empty() || mtp.is_empty() || weights.is_empty() {
            return Err(PyValueError::new_err("ONNX input paths must not be empty"));
        }
        if max_graph_bytes == 0 || max_weights_bytes == 0 {
            return Err(PyValueError::new_err("ONNX byte ceilings must be positive"));
        }
        Ok(Self {
            language: language.into(),
            mtp: mtp.into(),
            weights: weights.into(),
            admitted: AdmittedExternalQwen35BundleDigests {
                language_model_blake3: parse_digest(language_digest, "language_blake3")?,
                mtp_model_blake3: parse_digest(mtp_digest, "mtp_blake3")?,
                weights_blake3: parse_digest(weights_digest, "weights_blake3")?,
            },
            max_graph_bytes,
            max_weights_bytes,
        })
    }
}

fn verify_paths(inputs: &Inputs) -> Result<QwenOnnxBundleReceipt, String> {
    let language =
        read_regular_bounded(&inputs.language, inputs.max_graph_bytes, "language graph")?;
    let mtp = read_regular_bounded(&inputs.mtp, inputs.max_graph_bytes, "MTP graph")?;
    let weights = open_regular_bounded(
        &inputs.weights,
        inputs.max_weights_bytes,
        "external weights",
    )?;
    let verified =
        verify_external_qwen35_bundle_from_file(&language, &mtp, weights, inputs.admitted)
            .map_err(|error| error.to_string())?;
    receipt(&verified, inputs.admitted)
}

fn stage_paths(inputs: &Inputs, output: &Path) -> Result<QwenOnnxBundleReceipt, String> {
    let (parent, output) = canonical_output(output)?;
    let staging = create_staging_directory(&parent)?;
    let result = (|| {
        copy_sync(
            &inputs.language,
            &staging.join(LANGUAGE_FILE),
            inputs.max_graph_bytes,
            "language graph",
        )?;
        copy_sync(
            &inputs.mtp,
            &staging.join(MTP_FILE),
            inputs.max_graph_bytes,
            "MTP graph",
        )?;
        copy_sync_streaming(
            &inputs.weights,
            &staging.join(WEIGHTS_FILE),
            inputs.max_weights_bytes,
            "external weights",
        )?;
        let staged = Inputs {
            language: staging.join(LANGUAGE_FILE),
            mtp: staging.join(MTP_FILE),
            weights: staging.join(WEIGHTS_FILE),
            admitted: inputs.admitted,
            max_graph_bytes: inputs.max_graph_bytes,
            max_weights_bytes: inputs.max_weights_bytes,
        };
        let receipt = verify_paths(&staged)?;
        if receipt.conversion_mode.is_none() {
            return Err(
                "public Qwen ONNX publication requires complete conversion ancestry".to_owned(),
            );
        }
        write_manifest(&staging, &receipt)?;
        publish_staging(&staging, &parent, &output)?;
        Ok(receipt)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

fn canonical_output(output: &Path) -> Result<(PathBuf, PathBuf), String> {
    let name = output
        .file_name()
        .ok_or_else(|| "output_dir must name a new directory".to_owned())?;
    let parent = fs::canonicalize(output.parent().unwrap_or_else(|| Path::new(".")))
        .map_err(|error| format!("canonicalize output parent failed: {:?}", error.kind()))?;
    let output = parent.join(name);
    if fs::symlink_metadata(&output).is_ok() {
        return Err("output directory already exists".to_owned());
    }
    Ok((parent, output))
}

fn write_manifest(staging: &Path, receipt: &QwenOnnxBundleReceipt) -> Result<(), String> {
    let manifest = serde_json::to_vec_pretty(&json!({
        "schema": "tritium-qwen35-onnx-bundle-v1",
        "language": {"file": LANGUAGE_FILE, "blake3": receipt.language_blake3},
        "mtp": {"file": MTP_FILE, "blake3": receipt.mtp_blake3},
        "weights": {"file": WEIGHTS_FILE, "blake3": receipt.weights_blake3, "bytes": receipt.weights_bytes},
        "identity": {"source_model_id": receipt.source_model_id, "tokenizer_id": receipt.tokenizer_id, "recipe_id": receipt.recipe_id, "package_id": receipt.package_id, "converted_coverage_id": receipt.converted_coverage_id, "deferred_coverage_id": receipt.deferred_coverage_id},
        "conversion": {"mode": receipt.conversion_mode, "completion_id": receipt.completion_id, "campaign_id": receipt.campaign_id, "admission_id": receipt.admission_id, "selection_id": receipt.selection_id}
    })).map_err(|error| format!("encode ONNX manifest failed: {error}"))?;
    write_sync(&staging.join(MANIFEST_FILE), &manifest, "ONNX manifest")
}

fn write_sync(path: &Path, bytes: &[u8], label: &str) -> Result<(), String> {
    let mut file = File::create_new(path)
        .map_err(|error| format!("create staged {label} failed: {:?}", error.kind()))?;
    file.write_all(bytes)
        .map_err(|error| format!("write staged {label} failed: {:?}", error.kind()))?;
    file.sync_all()
        .map_err(|error| format!("sync staged {label} failed: {:?}", error.kind()))
}

fn publish_staging(staging: &Path, parent: &Path, output: &Path) -> Result<(), String> {
    File::open(staging)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("sync staging directory failed: {:?}", error.kind()))?;
    rename_directory_noreplace(staging, output)
        .map_err(|error| format!("publish output directory failed: {:?}", error.kind()))?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("sync output parent failed: {:?}", error.kind()))
}

fn receipt(
    verified: &VerifiedExternalQwen35Bundle,
    admitted: AdmittedExternalQwen35BundleDigests,
) -> Result<QwenOnnxBundleReceipt, String> {
    let identity = &verified.language.identity;
    let ancestry = verified.conversion_ancestry.as_ref();
    Ok(QwenOnnxBundleReceipt {
        language_blake3: hex_digest(admitted.language_model_blake3),
        mtp_blake3: hex_digest(admitted.mtp_model_blake3),
        weights_blake3: hex_digest(admitted.weights_blake3),
        weights_bytes: u64::try_from(verified.language.weights_bytes)
            .map_err(|_| "weights byte count exceeds u64")?,
        language_tokens: u64::try_from(verified.language.tokens)
            .map_err(|_| "language token count exceeds u64")?,
        language_past_tokens: u64::try_from(verified.language.past_tokens)
            .map_err(|_| "language past-token count exceeds u64")?,
        language_layers: u64::try_from(verified.language.layers)
            .map_err(|_| "language layer count exceeds u64")?,
        mtp_tokens: u64::try_from(verified.mtp.tokens)
            .map_err(|_| "MTP token count exceeds u64")?,
        mtp_past_tokens: u64::try_from(verified.mtp.past_tokens)
            .map_err(|_| "MTP past-token count exceeds u64")?,
        mtp_layers: u64::try_from(verified.mtp.layers)
            .map_err(|_| "MTP layer count exceeds u64")?,
        source_model_id: identity.source_model_id.clone(),
        tokenizer_id: identity.tokenizer_id.clone(),
        recipe_id: identity.recipe_id.clone(),
        package_id: identity.package_id.clone(),
        converted_coverage_id: identity.converted_coverage_id.clone(),
        deferred_coverage_id: identity.deferred_coverage_id.clone(),
        conversion_mode: ancestry.map(|value| value.conversion_mode.clone()),
        completion_id: ancestry.map(|value| value.completion_id.clone()),
        campaign_id: ancestry.map(|value| value.campaign_id.clone()),
        admission_id: ancestry.map(|value| value.admission_id.clone()),
        selection_id: ancestry.map(|value| value.selection_id.clone()),
    })
}

fn parse_digest(value: &str, label: &str) -> PyResult<[u8; 32]> {
    if value.len() != 64 {
        return Err(PyValueError::new_err(format!(
            "{label} must be 64 lowercase hexadecimal characters"
        )));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(pair).expect("hexadecimal bytes are ASCII");
        digest[index] = u8::from_str_radix(text, 16).map_err(|_| {
            PyValueError::new_err(format!(
                "{label} must be 64 lowercase hexadecimal characters"
            ))
        })?;
        if pair.iter().any(u8::is_ascii_uppercase) {
            return Err(PyValueError::new_err(format!(
                "{label} must be 64 lowercase hexadecimal characters"
            )));
        }
    }
    Ok(digest)
}

fn open_regular_handle(path: &Path, expected_bytes: u64, label: &str) -> Result<File, String> {
    let before = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} failed: {:?}", error.kind()))?;
    if before.file_type().is_symlink() || !before.is_file() || before.len() != expected_bytes {
        return Err(format!(
            "{label} must be an ordinary non-symlink file of exactly {expected_bytes} bytes"
        ));
    }
    let file =
        open_nofollow(path).map_err(|error| format!("open {label} failed: {:?}", error.kind()))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("inspect opened {label} failed: {:?}", error.kind()))?;
    if !opened.is_file() || opened.len() != expected_bytes || !same_file(&before, &opened) {
        return Err(format!("{label} changed before open"));
    }
    Ok(file)
}

fn open_regular_bounded(path: &Path, max_bytes: u64, label: &str) -> Result<File, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} failed: {:?}", error.kind()))?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() == 0
        || metadata.len() > max_bytes
    {
        return Err(format!(
            "{label} must be non-empty and no larger than {max_bytes} bytes"
        ));
    }
    open_regular_handle(path, metadata.len(), label)
}

#[cfg(unix)]
fn open_nofollow(path: &Path) -> io::Result<File> {
    rustix::fs::open(
        path,
        rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::CLOEXEC | rustix::fs::OFlags::NOFOLLOW,
        rustix::fs::Mode::empty(),
    )
    .map(Into::into)
    .map_err(Into::into)
}

#[cfg(windows)]
fn open_nofollow(path: &Path) -> io::Result<File> {
    File::open(path)
}

fn read_regular_bounded(path: &Path, max_bytes: u64, label: &str) -> Result<Vec<u8>, String> {
    let before = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} failed: {:?}", error.kind()))?;
    if before.file_type().is_symlink()
        || !before.is_file()
        || before.len() == 0
        || before.len() > max_bytes
    {
        return Err(format!(
            "{label} must be a non-empty ordinary file no larger than {max_bytes} bytes"
        ));
    }
    let mut file =
        open_nofollow(path).map_err(|error| format!("open {label} failed: {:?}", error.kind()))?;
    let opened = file
        .metadata()
        .map_err(|error| format!("inspect opened {label} failed: {:?}", error.kind()))?;
    if !opened.is_file() || opened.len() > max_bytes || !same_file(&before, &opened) {
        return Err(format!("{label} changed before open"));
    }
    let capacity =
        usize::try_from(opened.len()).map_err(|_| format!("{label} exceeds platform bounds"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| format!("allocate {label} failed"))?;
    let read_ceiling = max_bytes
        .checked_add(1)
        .ok_or_else(|| format!("{label} byte ceiling is too large"))?;
    Read::by_ref(&mut file)
        .take(read_ceiling)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read {label} failed: {:?}", error.kind()))?;
    let after = fs::symlink_metadata(path)
        .map_err(|error| format!("reinspect {label} failed: {:?}", error.kind()))?;
    let final_opened = file
        .metadata()
        .map_err(|error| format!("reinspect opened {label} failed: {:?}", error.kind()))?;
    if u64::try_from(bytes.len()).ok() != Some(opened.len())
        || before.len() != opened.len()
        || after.len() != opened.len()
        || final_opened.len() != opened.len()
        || !same_file(&before, &after)
        || !same_file(&opened, &final_opened)
    {
        return Err(format!("{label} changed while reading"));
    }
    Ok(bytes)
}

fn copy_sync(source: &Path, destination: &Path, max_bytes: u64, label: &str) -> Result<(), String> {
    let bytes = read_regular_bounded(source, max_bytes, label)?;
    let mut output = File::create_new(destination)
        .map_err(|error| format!("create staged {label} failed: {:?}", error.kind()))?;
    output
        .write_all(&bytes)
        .map_err(|error| format!("write staged {label} failed: {:?}", error.kind()))?;
    output
        .sync_all()
        .map_err(|error| format!("sync staged {label} failed: {:?}", error.kind()))
}

fn copy_sync_streaming(
    source: &Path,
    destination: &Path,
    max_bytes: u64,
    label: &str,
) -> Result<(), String> {
    let before = fs::symlink_metadata(source)
        .map_err(|error| format!("inspect {label} failed: {:?}", error.kind()))?;
    let mut input = open_regular_bounded(source, max_bytes, label)?;
    let opened = input
        .metadata()
        .map_err(|error| format!("inspect opened {label} failed: {:?}", error.kind()))?;
    if !same_file(&before, &opened) || before.len() != opened.len() {
        return Err(format!("{label} changed before streaming"));
    }
    let expected = opened.len();
    let mut output = File::create_new(destination)
        .map_err(|error| format!("create staged {label} failed: {:?}", error.kind()))?;
    let copied = io::copy(&mut input, &mut output)
        .map_err(|error| format!("copy staged {label} failed: {:?}", error.kind()))?;
    let final_opened = input
        .metadata()
        .map_err(|error| format!("reinspect opened {label} failed: {:?}", error.kind()))?;
    let after = fs::symlink_metadata(source)
        .map_err(|error| format!("reinspect {label} failed: {:?}", error.kind()))?;
    if copied != expected
        || final_opened.len() != expected
        || after.len() != expected
        || !same_file(&before, &after)
        || !same_file(&opened, &final_opened)
    {
        return Err(format!("{label} changed while streaming"));
    }
    output
        .sync_all()
        .map_err(|error| format!("sync staged {label} failed: {:?}", error.kind()))
}

fn create_staging_directory(parent: &Path) -> Result<PathBuf, String> {
    for _ in 0..64 {
        let nonce = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".tritium-onnx-stage-{}-{nonce}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(format!(
                    "create staging directory failed: {:?}",
                    error.kind()
                ));
            }
        }
    }
    Err("could not allocate a unique staging directory".to_owned())
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple",
    target_os = "redox"
))]
fn rename_directory_noreplace(source: &Path, target: &Path) -> io::Result<()> {
    rustix::fs::renameat_with(
        rustix::fs::CWD,
        source,
        rustix::fs::CWD,
        target,
        rustix::fs::RenameFlags::NOREPLACE,
    )
    .map_err(Into::into)
}
#[cfg(all(
    unix,
    not(any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        target_os = "redox"
    ))
))]
fn rename_directory_noreplace(_: &Path, _: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace directory publication is unsupported on this Unix target",
    ))
}
#[cfg(windows)]
fn rename_directory_noreplace(_: &Path, _: &Path) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "atomic no-replace directory publication is unsupported on Windows",
    ))
}

#[cfg(unix)]
fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}
#[cfg(windows)]
fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    left.volume_serial_number() == right.volume_serial_number()
        && left.file_index() == right.file_index()
}

fn hex_digest(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_parser_is_strict() {
        assert_eq!(
            parse_digest(&"ab".repeat(32), "digest").unwrap(),
            [0xab; 32]
        );
        assert!(parse_digest(&"AB".repeat(32), "digest").is_err());
        assert!(parse_digest("00", "digest").is_err());
        assert!(parse_digest(&"gg".repeat(32), "digest").is_err());
    }

    #[test]
    fn failed_stage_never_publishes_partial_directory() {
        let root = std::env::temp_dir().join(format!(
            "tritium-py-onnx-{}-{}",
            std::process::id(),
            STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        for name in [LANGUAGE_FILE, MTP_FILE, WEIGHTS_FILE] {
            fs::write(root.join(name), b"invalid").unwrap();
        }
        let output = root.join("published");
        let inputs = Inputs {
            language: root.join(LANGUAGE_FILE),
            mtp: root.join(MTP_FILE),
            weights: root.join(WEIGHTS_FILE),
            admitted: AdmittedExternalQwen35BundleDigests {
                language_model_blake3: [0; 32],
                mtp_model_blake3: [0; 32],
                weights_blake3: [0; 32],
            },
            max_graph_bytes: 1024,
            max_weights_bytes: 1024,
        };
        assert!(stage_paths(&inputs, &output).is_err());
        assert!(!output.exists());
        assert!(fs::read_dir(&root).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".tritium-onnx-stage-")
        }));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn export_fails_on_existing_output_before_bundle_admission() {
        let root = std::env::temp_dir().join(format!(
            "tritium-py-onnx-export-{}-{}",
            std::process::id(),
            STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let output = root.join("published");
        fs::create_dir(&output).unwrap();
        let request = ExportRequest {
            bundle: root.join("missing-bundle"),
            output: output.clone(),
            profile: "compact-v1".to_owned(),
            tokens: 1,
            past_tokens: 0,
            max_package_bytes: 1,
            max_preserved_bytes: 1,
            max_salt_resident_bytes: 1,
            max_preserved_fp32_bytes: 1,
        };
        let error = export_paths(&request).err().expect("export must fail");
        assert!(error.contains("output directory already exists"));
        assert!(output.is_dir());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn published_manifest_parser_rejects_duplicates_and_unknown_fields() {
        let root = std::env::temp_dir().join(format!(
            "tritium-py-onnx-manifest-{}-{}",
            std::process::id(),
            STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let valid = format!(
            r#"{{
                "schema":"tritium-qwen35-onnx-bundle-v1",
                "language":{{"file":"language.onnx","blake3":"{}"}},
                "mtp":{{"file":"mtp.onnx","blake3":"{}"}},
                "weights":{{"file":"weights.bin","blake3":"{}","bytes":7}},
                "identity":{{
                    "source_model_id":"source","tokenizer_id":"tokenizer",
                    "recipe_id":"recipe","package_id":"package",
                    "converted_coverage_id":"converted","deferred_coverage_id":"deferred"
                }},
                "conversion":{{
                    "mode":"ptq","completion_id":"completion","campaign_id":"campaign",
                    "admission_id":"admission","selection_id":"selection"
                }}
            }}"#,
            "11".repeat(32),
            "22".repeat(32),
            "33".repeat(32),
        );
        fs::write(root.join(MANIFEST_FILE), &valid).unwrap();
        assert!(read_published_manifest(&root).is_ok());

        let duplicate = valid.replacen(
            "\"schema\":",
            "\"schema\":\"tritium-qwen35-onnx-bundle-v1\",\"schema\":",
            1,
        );
        fs::write(root.join(MANIFEST_FILE), duplicate).unwrap();
        assert!(read_published_manifest(&root).is_err());

        let unknown = valid.replacen("\"schema\":", "\"unknown\":true,\"schema\":", 1);
        fs::write(root.join(MANIFEST_FILE), unknown).unwrap();
        assert!(read_published_manifest(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fixed_runtime_shapes_are_checked_without_wrapping() {
        assert_eq!(runtime_shape(&[0, 2, 4], "state").unwrap(), [0, 2, 4]);
        assert!(runtime_shape(&[-1, 2], "state").is_err());
        assert_eq!(shape_elements(&[0, usize::MAX], "state").unwrap(), 0);
        assert!(shape_elements(&[usize::MAX, 2], "state").is_err());
    }
}
