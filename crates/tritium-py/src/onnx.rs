//! Strict native admission and atomic publication for Qwen language-plus-MTP ONNX bundles.

use std::{
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
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
    map_qwen36_27b_packed_causal_lm, verify_external_qwen35_bundle_from_file,
};

const LANGUAGE_FILE: &str = "language.onnx";
const MTP_FILE: &str = "mtp.onnx";
const WEIGHTS_FILE: &str = "weights.bin";
const MANIFEST_FILE: &str = "tritium-onnx-manifest.json";
static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

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
}
