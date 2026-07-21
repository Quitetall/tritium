//! Python runtime adapter for complete packed Qwen3.6 language-plus-MTP bundles.

use std::path::Path;
use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tritium_nn::{Qwen35SaltV2LanguageMtpModel, Qwen35SaltV2LoadReceipt};
use tritium_spec::TernaryBackend;

/// Immutable identities, coverage, and physical ledgers for a packed Qwen load.
#[pyclass(module = "tritium", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct QwenLoadReceipt {
    manifest_package_id: String,
    profile: String,
    source_revision: String,
    declared_completion_id: String,
    declared_campaign_id: String,
    declared_admission_id: String,
    declared_selection_id: String,
    declared_source_model_id: String,
    declared_source_identity_status: String,
    declared_official_payload_authenticated: bool,
    config_package_id: String,
    package_id: String,
    preserved_package_id: String,
    codec: String,
    matrix_tensors: usize,
    preserved_tensors: usize,
    serialized_bytes: u64,
    preserved_serialized_bytes: u64,
    manifest_bytes: u64,
    hf_asset_bytes: u64,
    loaded_bundle_bytes: u64,
    device_resident_salt: bool,
    salt_resident_bytes: u64,
    preserved_fp32_bytes: u64,
    resident_bytes: u64,
}

impl From<&Qwen35SaltV2LoadReceipt> for QwenLoadReceipt {
    fn from(receipt: &Qwen35SaltV2LoadReceipt) -> Self {
        Self {
            manifest_package_id: receipt.manifest_package_id().to_owned(),
            profile: receipt.profile().to_owned(),
            source_revision: receipt.source_revision().to_owned(),
            declared_completion_id: receipt.declared_completion_id().to_owned(),
            declared_campaign_id: receipt.declared_campaign_id().to_owned(),
            declared_admission_id: receipt.declared_admission_id().to_owned(),
            declared_selection_id: receipt.declared_selection_id().to_owned(),
            declared_source_model_id: receipt.declared_source_model_id().to_owned(),
            declared_source_identity_status: receipt.declared_source_identity_status().to_owned(),
            declared_official_payload_authenticated: receipt
                .declared_official_payload_authenticated(),
            config_package_id: receipt.config_package_id().to_owned(),
            package_id: receipt.package_id().to_owned(),
            preserved_package_id: receipt.preserved_package_id().to_owned(),
            codec: format!("{:?}", receipt.codec()).to_ascii_lowercase(),
            matrix_tensors: receipt.matrix_tensors(),
            preserved_tensors: receipt.preserved_tensors(),
            serialized_bytes: receipt.serialized_bytes(),
            preserved_serialized_bytes: receipt.preserved_serialized_bytes(),
            manifest_bytes: receipt.manifest_bytes(),
            hf_asset_bytes: receipt.hf_asset_bytes(),
            loaded_bundle_bytes: receipt.loaded_bundle_bytes(),
            device_resident_salt: receipt.device_resident_salt(),
            salt_resident_bytes: receipt.salt_resident_bytes(),
            preserved_fp32_bytes: receipt.preserved_fp32_bytes(),
            resident_bytes: receipt.resident_bytes(),
        }
    }
}

#[pymethods]
impl QwenLoadReceipt {
    #[getter]
    fn manifest_package_id(&self) -> &str {
        &self.manifest_package_id
    }

    #[getter]
    fn profile(&self) -> &str {
        &self.profile
    }

    #[getter]
    fn source_revision(&self) -> &str {
        &self.source_revision
    }

    /// Untrusted declaration until external admission authorizes the manifest ID.
    #[getter]
    fn declared_completion_id(&self) -> &str {
        &self.declared_completion_id
    }

    /// Untrusted declaration until external admission authorizes the manifest ID.
    #[getter]
    fn declared_campaign_id(&self) -> &str {
        &self.declared_campaign_id
    }

    /// Untrusted declaration until external admission authorizes the manifest ID.
    #[getter]
    fn declared_admission_id(&self) -> &str {
        &self.declared_admission_id
    }

    /// Untrusted declaration until external admission authorizes the manifest ID.
    #[getter]
    fn declared_selection_id(&self) -> &str {
        &self.declared_selection_id
    }

    /// Untrusted source-model declaration copied from the content-bound manifest.
    #[getter]
    fn declared_source_model_id(&self) -> &str {
        &self.declared_source_model_id
    }

    /// Untrusted source status copied from the content-bound manifest.
    #[getter]
    fn declared_source_identity_status(&self) -> &str {
        &self.declared_source_identity_status
    }

    /// Untrusted manifest boolean; it does not authenticate this load.
    #[getter]
    fn declared_official_payload_authenticated(&self) -> bool {
        self.declared_official_payload_authenticated
    }

    #[getter]
    fn config_package_id(&self) -> &str {
        &self.config_package_id
    }

    #[getter]
    fn package_id(&self) -> &str {
        &self.package_id
    }

    #[getter]
    fn preserved_package_id(&self) -> &str {
        &self.preserved_package_id
    }

    #[getter]
    fn codec(&self) -> &str {
        &self.codec
    }

    #[getter]
    fn matrix_tensors(&self) -> usize {
        self.matrix_tensors
    }

    #[getter]
    fn preserved_tensors(&self) -> usize {
        self.preserved_tensors
    }

    #[getter]
    fn serialized_bytes(&self) -> u64 {
        self.serialized_bytes
    }

    #[getter]
    fn preserved_serialized_bytes(&self) -> u64 {
        self.preserved_serialized_bytes
    }

    #[getter]
    fn manifest_bytes(&self) -> u64 {
        self.manifest_bytes
    }

    #[getter]
    fn hf_asset_bytes(&self) -> u64 {
        self.hf_asset_bytes
    }

    #[getter]
    fn loaded_bundle_bytes(&self) -> u64 {
        self.loaded_bundle_bytes
    }

    #[getter]
    fn device_resident_salt(&self) -> bool {
        self.device_resident_salt
    }

    #[getter]
    fn salt_resident_bytes(&self) -> u64 {
        self.salt_resident_bytes
    }

    #[getter]
    fn preserved_fp32_bytes(&self) -> u64 {
        self.preserved_fp32_bytes
    }

    #[getter]
    fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    fn __repr__(&self) -> String {
        format!(
            "QwenLoadReceipt(profile='{}', codec='{}', matrix_tensors={}, preserved_tensors={}, resident_bytes={})",
            self.profile,
            self.codec,
            self.matrix_tensors,
            self.preserved_tensors,
            self.resident_bytes,
        )
    }
}

/// A content-bound packed Qwen3.6 language runtime.
///
/// The MTP graph is loaded and ternarized with the language model, but draft
/// execution remains gated on the separately pinned serving-oracle promotion.
#[pyclass(module = "tritium")]
pub(crate) struct QwenModel {
    model: Mutex<Qwen35SaltV2LanguageMtpModel>,
    profile: String,
    device: String,
}

#[pymethods]
impl QwenModel {
    /// Load one exact profile from a schema-v3 Tritium bundle.
    #[staticmethod]
    #[pyo3(signature = (bundle_dir, profile = "compact-v1", device = "cpu"))]
    fn load(py: Python<'_>, bundle_dir: &str, profile: &str, device: &str) -> PyResult<Self> {
        let (device_kind, device) = parse_device(device)?;
        let bundle_dir = bundle_dir.to_owned();
        let profile = profile.to_owned();
        let selected_profile = profile.clone();
        let selected_device = device.clone();
        let model = py.detach(move || {
            let backend = qwen_backend(device_kind)?;
            Qwen35SaltV2LanguageMtpModel::load_bundle_profile(
                Path::new(&bundle_dir),
                &profile,
                backend,
            )
            .map_err(|error| error.to_string())
        });
        model
            .map(|model| Self {
                model: Mutex::new(model),
                profile: selected_profile,
                device: selected_device,
            })
            .map_err(PyRuntimeError::new_err)
    }

    /// Greedily generate token IDs from a fresh prompt and cache.
    ///
    /// Every call is independent and allocates a cache sized to the requested
    /// prompt plus continuation, rather than reserving the checkpoint maximum.
    #[pyo3(signature = (token_ids, max_new_tokens, eos = 151_645))]
    fn generate(
        &self,
        py: Python<'_>,
        token_ids: Vec<i64>,
        max_new_tokens: usize,
        eos: u32,
    ) -> PyResult<Vec<u32>> {
        if token_ids.is_empty() {
            return Err(PyValueError::new_err("token_ids must not be empty"));
        }
        let prompt = token_ids
            .into_iter()
            .map(|token| {
                u32::try_from(token).map_err(|_| {
                    PyValueError::new_err(format!("token id {token} out of range for u32"))
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        py.detach(move || {
            let model = self
                .model
                .lock()
                .map_err(|_| PyRuntimeError::new_err("Qwen model mutex is poisoned"))?;
            let runner = model.runner();
            if let Some(token) = prompt
                .iter()
                .copied()
                .find(|token| *token as usize >= runner.vocab_size())
            {
                return Err(PyValueError::new_err(format!(
                    "token id {token} is outside vocabulary size {}",
                    runner.vocab_size()
                )));
            }
            if max_new_tokens == 0 {
                return Ok(Vec::new());
            }
            let capacity = prompt.len().checked_add(max_new_tokens).ok_or_else(|| {
                PyValueError::new_err("prompt plus continuation length overflows usize")
            })?;
            let mut cache = runner.new_cache(capacity).map_err(|error| {
                PyValueError::new_err(format!("invalid Qwen generation length: {error}"))
            })?;
            let mut output = runner
                .forward(&prompt, &mut cache)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            let mut generated = Vec::new();
            generated.try_reserve_exact(max_new_tokens).map_err(|_| {
                PyRuntimeError::new_err("could not allocate generated token buffer")
            })?;
            for step in 0..max_new_tokens {
                let next = greedy_token(output.last_logits()).map_err(PyRuntimeError::new_err)?;
                generated.push(next);
                if next == eos || step + 1 == max_new_tokens {
                    break;
                }
                output = runner
                    .forward(&[next], &mut cache)
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
            }
            Ok(generated)
        })
    }

    /// Selected governed profile filename.
    #[getter]
    fn profile(&self) -> &str {
        &self.profile
    }

    /// Active execution device.
    #[getter]
    fn device(&self) -> &str {
        &self.device
    }

    /// Whether the loaded draft graph has passed the production oracle gate.
    #[getter]
    fn mtp_verified(&self) -> bool {
        false
    }

    /// Tracked steady bytes for packed matrices and widened preserved tensors.
    #[getter]
    fn resident_bytes(&self) -> PyResult<u64> {
        self.model
            .lock()
            .map(|model| model.receipt().resident_bytes())
            .map_err(|_| PyRuntimeError::new_err("Qwen model mutex is poisoned"))
    }

    /// Immutable identities and exact physical ledgers for this load.
    #[getter]
    fn receipt(&self) -> PyResult<QwenLoadReceipt> {
        self.model
            .lock()
            .map(|model| QwenLoadReceipt::from(model.receipt()))
            .map_err(|_| PyRuntimeError::new_err("Qwen model mutex is poisoned"))
    }

    fn __repr__(&self) -> String {
        match self.model.lock() {
            Ok(model) => format!(
                "QwenModel(profile='{}', device='{}', vocab_size={}, mtp_verified=False)",
                self.profile,
                self.device,
                model.runner().vocab_size(),
            ),
            Err(_) => "QwenModel(<poisoned>)".to_owned(),
        }
    }
}

#[derive(Clone, Copy)]
enum QwenDevice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

fn parse_device(device: &str) -> PyResult<(QwenDevice, String)> {
    if device == "cpu" {
        return Ok((QwenDevice::Cpu, "cpu".to_owned()));
    }
    if device == "cuda" {
        #[cfg(feature = "cuda")]
        return Ok((QwenDevice::Cuda(0), "cuda:0".to_owned()));
    }
    #[cfg(feature = "cuda")]
    if let Some(ordinal) = device.strip_prefix("cuda:") {
        let ordinal = ordinal.parse::<usize>().map_err(|_| {
            PyValueError::new_err("CUDA device must be `cuda` or `cuda:<ordinal>`")
        })?;
        return Ok((QwenDevice::Cuda(ordinal), format!("cuda:{ordinal}")));
    }
    #[cfg(feature = "cuda")]
    let expected = "'cpu', 'cuda', or 'cuda:<ordinal>'";
    #[cfg(not(feature = "cuda"))]
    let expected = "'cpu' (this wheel was built without CUDA)";
    Err(PyValueError::new_err(format!(
        "Qwen SALT V2 device must be {expected}, got {device:?}"
    )))
}

fn qwen_backend(device: QwenDevice) -> Result<Box<dyn TernaryBackend>, String> {
    match device {
        QwenDevice::Cpu => crate::cpu_backend(),
        #[cfg(feature = "cuda")]
        QwenDevice::Cuda(ordinal) => tritium_cuda::CudaBackend::new(ordinal)
            .map(|backend| Box::new(backend) as Box<dyn TernaryBackend>)
            .map_err(|error| error.to_string()),
    }
}

fn greedy_token(logits: &[f32]) -> Result<u32, String> {
    let (&first, remaining) = logits
        .split_first()
        .ok_or_else(|| "Qwen language head returned empty logits".to_owned())?;
    if !first.is_finite() {
        return Err("Qwen language head returned non-finite logits".to_owned());
    }
    let mut best_index = 0usize;
    let mut best_value = first;
    for (index, &value) in remaining.iter().enumerate() {
        if !value.is_finite() {
            return Err("Qwen language head returned non-finite logits".to_owned());
        }
        if value > best_value {
            best_index = index + 1;
            best_value = value;
        }
    }
    u32::try_from(best_index).map_err(|_| "Qwen token index exceeds u32".to_owned())
}

#[cfg(test)]
mod tests {
    use super::{greedy_token, parse_device};

    #[test]
    fn greedy_token_uses_first_maximum_and_rejects_nonfinite_logits() {
        assert_eq!(greedy_token(&[-2.0, 3.0, 3.0, 1.0]).unwrap(), 1);
        assert!(greedy_token(&[]).is_err());
        assert!(greedy_token(&[0.0, f32::NAN]).is_err());
    }

    #[test]
    fn device_parser_is_explicit_about_compiled_cuda_support() {
        assert_eq!(parse_device("cpu").unwrap().1, "cpu");
        #[cfg(feature = "cuda")]
        {
            assert_eq!(parse_device("cuda").unwrap().1, "cuda:0");
            assert_eq!(parse_device("cuda:3").unwrap().1, "cuda:3");
        }
        #[cfg(not(feature = "cuda"))]
        assert!(parse_device("cuda").is_err());
        assert!(parse_device("cuda:not-an-ordinal").is_err());
        assert!(parse_device("metal").is_err());
    }
}
