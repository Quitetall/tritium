//! Python runtime adapter for complete packed Qwen3.6 language-plus-MTP bundles.

use std::path::Path;
use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tritium_nn::Qwen35SaltV2LanguageMtpModel;

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
        if device != "cpu" {
            return Err(PyValueError::new_err(format!(
                "Qwen SALT V2 runtime currently supports device='cpu', got {device:?}"
            )));
        }
        let bundle_dir = bundle_dir.to_owned();
        let profile = profile.to_owned();
        let selected_profile = profile.clone();
        let model = py.detach(move || {
            let backend = crate::cpu_backend()?;
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
                device: device.to_owned(),
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
    use super::greedy_token;

    #[test]
    fn greedy_token_uses_first_maximum_and_rejects_nonfinite_logits() {
        assert_eq!(greedy_token(&[-2.0, 3.0, 3.0, 1.0]).unwrap(), 1);
        assert!(greedy_token(&[]).is_err());
        assert!(greedy_token(&[0.0, f32::NAN]).is_err());
    }
}
