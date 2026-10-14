//! Thin actual-source/cache receivers; shared quantization owns every identity and reducer.
use crate::{
    model_conversion::hex,
    torch_native::{cpu_f32, tensor_shape},
};
use dlpark::legacy::Dlpack;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyAny, PyBytes},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Mutex,
};
use tritium_quantize::engine::{
    CalibrationBatch, CalibrationCache, CalibrationCacheBuilder, CalibrationCollector,
    CalibrationInput, CalibrationOutputBatch, FileWeightSource, ObservedWeightSource,
    read_original_file_bounded,
};
use tritium_schema::{
    BlobId, CalibratedFitPlan, CalibratedQuantAction, CalibratedQuantPlan,
    CalibratedQuantTensorPlan, CalibrationCacheBinding, CalibrationCacheId,
    CalibrationCacheRequest, CalibrationConsumer, CalibrationOutput, CalibrationPartition,
    DirectFitMetric, ModelQuantPlan, NamedAssetDigest, OriginalSourceId, QuantAction, QuantPlan,
    QuantSourceBinding,
};

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}
fn json<T: serde::de::DeserializeOwned>(value: &str) -> PyResult<T> {
    if value.is_empty() || value.len() > 1_048_576 {
        return Err(error("calibration JSON exceeds byte limit"));
    }
    serde_json::from_str(value).map_err(error)
}

/// Actual complete stored-byte observation. It is neither a source lock nor a qualification.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct NativeSourceObservation {
    architecture: String,
    source: OriginalSourceId,
    tensors: Vec<QuantSourceBinding>,
    assets: Vec<NamedAssetDigest>,
    #[pyo3(get)]
    publication_warnings: Vec<String>,
}

#[pyfunction]
pub(crate) fn _observe_model_plan_source(
    py: Python<'_>,
    plan_json: &str,
    source_shards: Vec<String>,
    asset_paths: BTreeMap<String, String>,
) -> PyResult<NativeSourceObservation> {
    let plan: QuantPlan = json(plan_json)?;
    let shards = source_shards
        .into_iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();
    let assets = asset_paths
        .into_iter()
        .map(|(n, p)| (n, PathBuf::from(p)))
        .collect();
    py.detach(|| {
        let mut source = FileWeightSource::open(
            ModelQuantPlan::Uncalibrated(&plan),
            &shards,
            assets,
            BTreeMap::new(),
        )
        .map_err(error)?;
        let observed = ObservedWeightSource::new(&plan, &mut source).map_err(error)?;
        Ok(NativeSourceObservation {
            architecture: observed.architecture().to_owned(),
            source: observed.source_id(),
            tensors: observed.tensor_bindings().to_vec(),
            assets: observed.asset_bindings().to_vec(),
            publication_warnings: Vec::new(),
        })
    })
}

#[pymethods]
impl NativeSourceObservation {
    fn validate_cache(&self, cache: &NativeCalibrationCache) -> PyResult<()> {
        let binding = cache.cache.binding();
        if binding.request.model != self.source
            || binding.request.consumer != CalibrationConsumer::Matmul
            || !matches!(
                binding.request.output,
                CalibrationOutput::None | CalibrationOutput::DenseFisher
            )
            || !self
                .tensors
                .iter()
                .any(|t| t.source == binding.request.tensor)
            || !self
                .assets
                .iter()
                .any(|a| a.name == "tokenizer.json" && a.content == binding.tokenizer)
        {
            return Err(error(
                "calibration cache differs from observed source/consumer/tokenizer",
            ));
        }
        Ok(())
    }
    fn _with_source_cleanup_warning(&self, message: &str) -> Self {
        let mut result = self.clone();
        result.publication_warnings.push(format!(
            "SourceCleanup: {}",
            message.chars().take(1024).collect::<String>()
        ));
        result
    }
    #[getter]
    fn source_id(&self) -> String {
        hex(self.source.as_bytes())
    }
    /// Context comes from actual observed source, not caller-supplied digests.
    fn cache_builder(
        &self,
        name: &str,
        tokenizer: &[u8],
        objective: &[u8],
        output_json: &str,
        limits_json: &str,
    ) -> PyResult<NativeCalibrationBuilder> {
        let tensor = self
            .tensors
            .iter()
            .find(|t| t.source.name == name)
            .ok_or_else(|| error("tensor absent from observed source"))?
            .source
            .clone();
        let tokenizer_id = BlobId::from_bytes(*blake3::hash(tokenizer).as_bytes());
        if !self
            .assets
            .iter()
            .any(|a| a.name == "tokenizer.json" && a.content == tokenizer_id)
        {
            return Err(error("tokenizer bytes differ from observed source"));
        }
        let output: CalibrationOutput = json(output_json)?;
        if !matches!(
            output,
            CalibrationOutput::None | CalibrationOutput::DenseFisher
        ) {
            return Err(error("this Matmul adapter requires None or DenseFisher"));
        }
        let request = CalibrationCacheRequest {
            tensor,
            model: self.source,
            consumer: CalibrationConsumer::Matmul,
            output,
            sample_start: 0,
            limits: json(limits_json)?,
        };
        let builder =
            CalibrationCacheBuilder::new(request.clone(), tokenizer, objective).map_err(error)?;
        Ok(NativeCalibrationBuilder {
            state: Mutex::new(Some(builder)),
            request,
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit PyO3 numerical/cache byte ceilings"
    )]
    fn calibrated_plan_json(
        &self,
        plan_json: &str,
        caches: Vec<PyRef<'_, NativeCalibrationCache>>,
        partition_json: &str,
        damping: f64,
        max_numeric_bytes: u64,
        max_cache_payload_bytes: u64,
        max_cache_package_bytes: u64,
    ) -> PyResult<String> {
        let plain: QuantPlan = json(plan_json)?;
        plain.validate().map_err(error)?;
        if plain.architecture != self.architecture {
            return Err(error(
                "calibrated architecture differs from observed source",
            ));
        }
        let partition: CalibrationPartition = json(partition_json)?;
        let mut bindings = BTreeMap::new();
        for cache in caches {
            self.validate_cache(&cache)?;
            let binding = cache.cache.binding();
            if binding.request.model != self.source
                || bindings
                    .insert(binding.request.tensor.name.clone(), binding.clone())
                    .is_some()
            {
                return Err(error("foreign or duplicate calibrated cache"));
            }
        }
        let mut tensors = Vec::new();
        for tensor in plain.tensors {
            if !self.tensors.iter().any(|t| t.source == tensor.source) {
                return Err(error("source interpretation changed"));
            }
            let action = match tensor.action {
                QuantAction::KeepDense => CalibratedQuantAction::KeepDense,
                QuantAction::Fit {
                    plan,
                    metric: DirectFitMetric::Identity,
                } => {
                    let cache = bindings
                        .remove(&tensor.source.name)
                        .ok_or_else(|| error("selected owner missing calibration"))?;
                    if cache.request.tensor != tensor.source {
                        return Err(error("cache source interpretation differs"));
                    }
                    CalibratedQuantAction::CalibratedFit {
                        plan: Box::new(CalibratedFitPlan {
                            direct: *plan,
                            cache,
                            partition,
                            damping,
                            max_numeric_bytes,
                        }),
                    }
                }
                _ => {
                    return Err(error(
                        "native calibrated adapter cannot reinterpret raw-Gram or unsupported actions",
                    ));
                }
            };
            tensors.push(CalibratedQuantTensorPlan {
                source: tensor.source,
                action,
            });
        }
        if !bindings.is_empty()
            || tensors.len() != self.tensors.len()
            || plain.assets.iter().ne(self.assets.iter().map(|a| &a.name))
        {
            return Err(error("calibrated source/cache inventory differs"));
        }
        let plan = CalibratedQuantPlan {
            architecture: plain.architecture,
            original_source: self.source,
            tensors,
            assets: plain.assets,
            levels: plain.levels,
            limits: plain.limits,
            max_cache_payload_bytes,
            max_cache_package_bytes,
        };
        plan.validate().map_err(error)?;
        serde_json::to_string(&plan).map_err(error)
    }
}

/// Transactional core cache builder. Finalization is terminal, including a failed seal.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativeCalibrationBuilder {
    state: Mutex<Option<CalibrationCacheBuilder>>,
    request: CalibrationCacheRequest,
}
#[pymethods]
impl NativeCalibrationBuilder {
    #[pyo3(signature=(tokens, inputs, *, gradients=None, weights=None, mask=None))]
    fn capture(
        &self,
        py: Python<'_>,
        tokens: &Bound<'_, PyAny>,
        inputs: &Bound<'_, PyAny>,
        gradients: Option<&Bound<'_, PyAny>>,
        weights: Option<&Bound<'_, PyAny>>,
        mask: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let tokens: Dlpack = tokens.extract()?;
        let input: Dlpack = inputs.extract()?;
        let gradients: Option<Dlpack> = gradients.map(|v| v.extract()).transpose()?;
        let weights: Option<Dlpack> = weights.map(|v| v.extract()).transpose()?;
        let mask: Option<Dlpack> = mask.map(|v| v.extract()).transpose()?;
        let token_shape = tensor_shape(&tokens, "tokens")?;
        let shape = tensor_shape(&input, "inputs")?;
        if token_shape.len() != 1
            || shape.len() != 2
            || shape[0] != token_shape[0]
            || shape[1] as u64 != self.request.tensor.shape[1]
        {
            return Err(error("calibration tokens/activation geometry differs"));
        }
        let n = shape[0];
        if n == 0
            || n as u64 > self.request.limits.max_samples
            || (n as u64)
                .checked_mul(4)
                .is_none_or(|v| v > self.request.limits.max_payload_bytes)
        {
            return Err(error("calibration token count exceeds logical limit"));
        }
        let token_values = tokens.cpu_data_slice::<i64>().map_err(error)?;
        let values = cpu_f32(&input, "inputs")?;
        let gradients = match (self.request.output, gradients.as_ref()) {
            (CalibrationOutput::None, None) => None,
            (CalibrationOutput::DenseFisher, Some(g)) => {
                if tensor_shape(g, "gradients")? != [n, self.request.tensor.shape[0] as usize] {
                    return Err(error("calibration gradient geometry differs"));
                }
                Some(cpu_f32(g, "gradients")?)
            }
            _ => return Err(error("calibration output encoding differs")),
        };
        let weights = weights
            .as_ref()
            .map(|v| {
                if tensor_shape(v, "weights")? != [n] {
                    return Err(error("calibration weight geometry differs"));
                }
                v.cpu_data_slice::<f64>().map_err(error)
            })
            .transpose()?;
        let mask = mask
            .as_ref()
            .map(|v| {
                if tensor_shape(v, "mask")? != [n] {
                    return Err(error("calibration mask geometry differs"));
                }
                let values = v.cpu_data_slice::<u8>().map_err(error)?;
                if values.iter().any(|&value| value > 1) {
                    return Err(error("calibration mask must contain only 0/1"));
                }
                Ok(values)
            })
            .transpose()?;
        py.detach(|| {
            let mut tokens = Vec::new();
            tokens
                .try_reserve_exact(token_values.len())
                .map_err(error)?;
            for &value in token_values {
                tokens.push(u32::try_from(value).map_err(|_| error("token is outside u32"))?);
            }
            let mask = mask
                .map(|values| {
                    let mut output = Vec::new();
                    output.try_reserve_exact(values.len()).map_err(error)?;
                    output.extend(values.iter().map(|&value| value != 0));
                    Ok::<_, PyErr>(output)
                })
                .transpose()?;
            let mut state = self.state.lock().map_err(error)?;
            let builder = state
                .as_mut()
                .ok_or_else(|| error("calibration builder is sealed"))?;
            builder
                .capture(
                    &tokens,
                    CalibrationBatch {
                        samples: n,
                        input: CalibrationInput::Activations(values),
                        output: gradients
                            .map_or(CalibrationOutputBatch::None, CalibrationOutputBatch::Dense),
                        weights,
                        mask: mask.as_deref(),
                    },
                )
                .map_err(error)
        })
    }
    fn finish(&self, py: Python<'_>) -> PyResult<NativeCalibrationCache> {
        py.detach(|| {
            let builder = self
                .state
                .lock()
                .map_err(error)?
                .take()
                .ok_or_else(|| error("calibration builder is sealed"))?;
            Ok(NativeCalibrationCache {
                cache: builder.finish().map_err(error)?,
            })
        })
    }
}

/// Immutable actual cache, not independently verified forward/backward execution.
#[pyclass(frozen, module = "tritium._tritium", skip_from_py_object)]
pub(crate) struct NativeCalibrationCache {
    cache: CalibrationCache,
}
#[pymethods]
impl NativeCalibrationCache {
    #[getter]
    fn binding_json(&self) -> PyResult<String> {
        serde_json::to_string(self.cache.binding()).map_err(error)
    }
    #[getter]
    fn cache_id(&self) -> String {
        hex(self.cache.id().as_bytes())
    }
    #[getter]
    fn payload_bytes(&self) -> u64 {
        self.cache.binding().payload_bytes
    }
    fn required_numeric_bytes(&self, partition_json: &str, damping: f64) -> PyResult<u64> {
        let plan = self
            .cache
            .binding()
            .reduction_plan(json(partition_json)?, damping, u64::MAX);
        CalibrationCollector::required_numeric_bytes(&plan)
            .map_err(error)?
            .checked_add(self.payload_bytes())
            .ok_or_else(|| error("numeric budget overflow"))
    }
    fn package<'py>(
        &self,
        py: Python<'py>,
        max_package_bytes: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py.detach(|| self.cache.encode_package(max_package_bytes).map_err(error))?;
        Ok(PyBytes::new(py, &bytes))
    }
    #[staticmethod]
    #[pyo3(signature=(path, binding_json, max_package_bytes, *, max_payload_bytes=None))]
    fn open(
        py: Python<'_>,
        path: &str,
        binding_json: &str,
        max_package_bytes: u64,
        max_payload_bytes: Option<u64>,
    ) -> PyResult<Self> {
        let binding: CalibrationCacheBinding = json(binding_json)?;
        let mut limits = binding.request.limits;
        limits.max_payload_bytes = limits
            .max_payload_bytes
            .min(max_payload_bytes.unwrap_or(u64::MAX));
        if limits.max_payload_bytes == 0 {
            return Err(error("cache payload limit must be positive"));
        }
        py.detach(|| {
            let bytes =
                read_original_file_bounded(Path::new(path), max_package_bytes).map_err(error)?;
            let id = CalibrationCacheId::from_bytes(binding.source.cache.0);
            let cache = CalibrationCache::decode_package(
                &bytes,
                id,
                limits,
                usize::try_from(max_package_bytes).map_err(error)?,
            )
            .map_err(error)?;
            if cache.binding() != &binding {
                return Err(error("actual calibration cache binding differs"));
            }
            Ok(Self { cache })
        })
    }
}
