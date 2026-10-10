//! Shared admitted source loading; typed profile remains experimental (ADR 0054).

use std::{cell::RefCell, collections::BTreeSet, path::Path};

use half::{bf16, f16};
use tritium_format::{
    AdditiveTensor,
    trit_model::{TritModel, read_trit_model},
};
#[cfg(feature = "tokenizer")]
use tritium_schema::{LoadedSourceIdentity, ModelLoaded, UnknownReason};
use tritium_schema::{PlaneCounts, ScalePrecision, TensorMeta, TensorStorage};
use tritium_spec::TensorView;

use super::{
    ModelWeights,
    hf::{
        DenseTensorRequest, MAX_CONFIG_JSON_BYTES, NameSchema, build_standard_model_with_embedding,
        declared_vocab_size, resolve_optional_attention_names,
    },
};
use crate::{ArchSpec, ModelConfig, NnError, Projection, TensorLoader, TokenEmbedding};

#[cfg(feature = "tokenizer")]
const MAX_TOKENIZER_JSON_BYTES: u64 = 256 * 1024 * 1024;
#[cfg(feature = "tokenizer")]
const MAX_TOKENIZER_CONFIG_BYTES: u64 = 16 * 1024 * 1024;

/// Supported sources for the shared admitted standard-model loading path.
///
/// Typed `.trit` is a proposed profile. Shards/per-tile/entropy and hybrid/MTP
/// execution require later reader/architecture support, not a silent fallback.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum ModelSource<'a> {
    /// Existing indexed HF safetensors directory; paths are not content identities.
    HuggingFace(&'a Path),
    /// Authenticated container bytes and the selected complete level.
    Trit {
        /// Complete `.trit` source bytes, borrowed only during loading.
        bytes: &'a [u8],
        /// Explicit level name; never infer a compact prefix or choose a default.
        level: &'a str,
    },
}

enum PreparedSource<'a> {
    Hf(&'a Path),
    Trit {
        model: Box<TritModel<'a>>,
        level: &'a str,
    },
}

impl<'a> PreparedSource<'a> {
    #[cfg(feature = "tokenizer")]
    fn identity(&self) -> Result<LoadedSourceIdentity, NnError> {
        match self {
            Self::Hf(_) => Ok(LoadedSourceIdentity::HuggingFace {
                reason: UnknownReason::MissingCapability,
            }),
            Self::Trit { model, level } => Ok(LoadedSourceIdentity::Trit {
                package_id: tritium_schema::PackageId::from_bytes(*model.package_id().as_bytes()),
                model_id: model.model_id(level).map_err(format_error)?,
                level: (*level).to_owned(),
            }),
        }
    }

    fn new(source: ModelSource<'a>) -> Result<Self, NnError> {
        match source {
            ModelSource::HuggingFace(path) => Ok(Self::Hf(path)),
            ModelSource::Trit { bytes, level } => {
                let model = read_trit_model(bytes).map_err(format_error)?;
                if !model.manifest().levels.iter().any(|l| l.name == level) {
                    return Err(NnError::MissingTensor(format!(
                        "absent model level {level}"
                    )));
                }
                Ok(Self::Trit {
                    model: Box::new(model),
                    level,
                })
            }
        }
    }

    fn weights(
        &self,
        loader: &mut TensorLoader<'_>,
    ) -> Result<(ModelConfig, ArchSpec, ModelWeights), NnError> {
        match self {
            Self::Hf(path) => ModelWeights::load_hf_admitted_impl(path, loader),
            Self::Trit { model, level } => load_trit(model, level, loader),
        }
    }

    #[cfg(feature = "tokenizer")]
    fn tokenizer(&self) -> Result<super::HfJsonTokenizer, NnError> {
        match self {
            Self::Hf(path) => {
                let tokenizer =
                    read_tokenizer_asset(path, "tokenizer.json", MAX_TOKENIZER_JSON_BYTES)?;
                let config = read_tokenizer_asset(
                    path,
                    "tokenizer_config.json",
                    MAX_TOKENIZER_CONFIG_BYTES,
                )?;
                super::HfJsonTokenizer::from_bytes(&tokenizer, &config)
            }
            Self::Trit { model, .. } => {
                let tokenizer = model.asset("tokenizer.json").map_err(format_error)?;
                let config = model.asset("tokenizer_config.json").map_err(format_error)?;
                check_asset_size(
                    tokenizer.len() as u64,
                    "tokenizer.json",
                    MAX_TOKENIZER_JSON_BYTES,
                )?;
                check_asset_size(
                    config.len() as u64,
                    "tokenizer_config.json",
                    MAX_TOKENIZER_CONFIG_BYTES,
                )?;
                super::HfJsonTokenizer::from_bytes(tokenizer, config)
            }
        }
    }
}

#[cfg(feature = "tokenizer")]
fn check_asset_size(size: u64, name: &str, limit: u64) -> Result<(), NnError> {
    if size > limit {
        return Err(NnError::Tokenizer(format!(
            "{name} exceeds source size limit ({limit} bytes)"
        )));
    }
    Ok(())
}

#[cfg(feature = "tokenizer")]
fn read_tokenizer_asset(path: &Path, name: &str, limit: u64) -> Result<Vec<u8>, NnError> {
    use std::io::Read;
    let path = path.join(name);
    let metadata = std::fs::metadata(&path)
        .map_err(|error| NnError::Tokenizer(format!("metadata {name}: {error}")))?;
    if !metadata.is_file() {
        return Err(NnError::Tokenizer(format!("{name} must be a regular file")));
    }
    check_asset_size(metadata.len(), name, limit)?;
    let file = std::fs::File::open(path)
        .map_err(|error| NnError::Tokenizer(format!("read {name}: {error}")))?;
    let metadata = file
        .metadata()
        .map_err(|error| NnError::Tokenizer(format!("metadata {name}: {error}")))?;
    if !metadata.is_file() {
        return Err(NnError::Tokenizer(format!("{name} must be a regular file")));
    }
    check_asset_size(metadata.len(), name, limit)?;
    let mut bytes = Vec::new();
    // Bound the actual read as well as metadata: a concurrent writer can grow it.
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| NnError::Tokenizer(format!("read {name}: {error}")))?;
    check_asset_size(bytes.len() as u64, name, limit)?;
    Ok(bytes)
}

impl ModelWeights {
    /// Load backend-owned standard-model weights through the shared source path.
    ///
    /// Norm/bias vectors remain finite host data; matrix uploads use the same
    /// capability/payload-budget/evidence interface. A tied head shares its
    /// embedding handle. Failed assembly releases all unpublished handles.
    /// Tokenizers are not inferred by this weights-only operation.
    ///
    /// # Errors
    /// Source/config/tensor coverage, geometry, codec, upload or evidence errors.
    pub fn load_admitted(
        source: ModelSource<'_>,
        loader: &mut TensorLoader<'_>,
    ) -> Result<(ModelConfig, ArchSpec, Self), NnError> {
        PreparedSource::new(source)?.weights(loader)
    }
}

/// Actual text tokenizer, architecture, runner and recorded initial-load view.
///
/// The receipt is not independent qualification or current mutable runner state.
#[cfg(feature = "tokenizer")]
#[allow(missing_debug_implementations)]
pub struct LoadedModel {
    /// Shared standard-model runner; config and one weights owner live here.
    pub runner: super::ModelRunner,
    /// Parsed real tokenizer and special-token contract.
    pub tokenizer: super::HfJsonTokenizer,
    /// Architecture variation data used by the same model builder.
    pub architecture: ArchSpec,
    /// Immutable view of the registered initial-load observation.
    pub receipt: tritium_evidence::ModelLoadReceiptView,
}

/// Load a text model through one source selector and shared admitted execution.
///
/// Parses tokenizer assets before upload; validates all tokenizer IDs against
/// embedding rows before returning a runner. A `.trit` container is parsed only
/// once; no filesystem extraction or retained dense model shadow is introduced.
/// Tokenizer JSON is bounded to 256 MiB and its config to 16 MiB; these are
/// source limits, not an accounting of parsed tokenizer or total model memory.
/// This is standard Attention/SwiGLU/Relu2 execution, not hybrid/MTP, resident
/// executor, full-memory or release qualification. Typed sources bind candidate
/// model/package IDs; HF identity remains explicitly unavailable. Final load
/// event recording failure drops the unpublished runner and payload leases.
/// The recorder must register both `TENSOR_UPLOADED_REGISTRATION` and
/// `MODEL_LOADED_REGISTRATION` from `tritium_evidence`.
///
/// # Errors
/// Tokenizer, source/config/coverage, payload admission or runner errors.
#[cfg(feature = "tokenizer")]
pub fn load(
    source: ModelSource<'_>,
    backend: Box<dyn tritium_spec::TernaryBackend>,
    policy: tritium_schema::TensorUploadPolicy,
    budget: &tritium_runtime::TensorPayloadBudget,
    context: tritium_runtime::TensorUploadContext<'_>,
) -> Result<LoadedModel, NnError> {
    let source = PreparedSource::new(source)?;
    let identity = source.identity()?;
    let tokenizer = source.tokenizer()?;
    let (config, architecture, weights) = {
        let mut loader = TensorLoader::new(
            backend.as_ref(),
            policy,
            budget,
            tritium_runtime::TensorUploadContext {
                recorder: &mut *context.recorder,
                span: context.span,
                parent: context.parent,
                logical_time: context.logical_time,
            },
        );
        source.weights(&mut loader)?
    };
    tokenizer.validate_model_vocab(weights.vocab)?;
    let observation = ModelLoaded {
        source: identity,
        architecture: config.arch.clone(),
        backend: backend.device_id().to_owned(),
        physical_device: backend.physical_device_id().to_owned(),
        vocab_size: u64::try_from(weights.vocab)
            .map_err(|_| NnError::Backend("model vocabulary exceeds u64".into()))?,
        layer_count: u64::from(config.n_layers),
    };
    let runner = super::ModelRunner::try_from_weights(config, weights, backend)?;
    let receipt = context
        .recorder
        .emit_model_loaded(
            context.span,
            context.parent,
            context.logical_time,
            &observation,
        )
        .map_err(|error| NnError::Backend(format!("model load evidence: {error}")))?;
    Ok(LoadedModel {
        runner,
        tokenizer,
        architecture,
        receipt,
    })
}

fn format_error(error: tritium_format::trit_model::TritModelError) -> NnError {
    NnError::InvalidArtifact(format!("typed .trit model: {error}"))
}

fn dense_values(meta: &TensorMeta, bytes: &[u8]) -> Result<Vec<f32>, NnError> {
    let TensorStorage::Dense { precision, .. } = &meta.storage else {
        return Err(NnError::MissingTensor(format!(
            "{} must be a dense vector",
            meta.name
        )));
    };
    let width = match precision {
        ScalePrecision::F32 => 4,
        ScalePrecision::F16 | ScalePrecision::Bf16 => 2,
        _ => return Err(NnError::Backend("unsupported dense storage".into())),
    };
    let mut values = Vec::new();
    values
        .try_reserve_exact(bytes.len() / width)
        .map_err(|e| NnError::Backend(format!("decode {}: {e}", meta.name)))?;
    match precision {
        ScalePrecision::F32 => values.extend(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b)),
        ),
        ScalePrecision::F16 => values.extend(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f16::from_bits(u16::from_le_bytes(*b)).to_f32()),
        ),
        ScalePrecision::Bf16 => values.extend(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| bf16::from_bits(u16::from_le_bytes(*b)).to_f32()),
        ),
        _ => return Err(NnError::Backend("unsupported dense storage".into())),
    }
    Ok(values)
}

fn matrix_shape(meta: &TensorMeta) -> Result<(usize, usize), NnError> {
    let dims = match &meta.storage {
        TensorStorage::Dense { shape, .. } if shape.len() == 2 => (shape[0], shape[1]),
        TensorStorage::Additive { layout, .. } => (layout.rows, layout.cols),
        _ => {
            return Err(NnError::MissingTensor(format!(
                "{} must be a matrix",
                meta.name
            )));
        }
    };
    let rows = usize::try_from(dims.0)
        .map_err(|_| NnError::Backend("matrix rows overflow usize".into()))?;
    let cols = usize::try_from(dims.1)
        .map_err(|_| NnError::Backend("matrix cols overflow usize".into()))?;
    Ok((rows, cols))
}

fn upload_matrix(
    loader: &mut TensorLoader<'_>,
    meta: &TensorMeta,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Result<crate::Linear, NnError> {
    if matrix_shape(meta)? != (rows, cols) {
        return Err(NnError::MissingTensor(format!(
            "{} has incompatible matrix geometry",
            meta.name
        )));
    }
    match &meta.storage {
        TensorStorage::Dense { .. } => {
            let values = dense_values(meta, bytes)?;
            loader.upload(
                &meta.name,
                TensorView::Dense {
                    rows,
                    cols,
                    values: &values,
                },
            )
        }
        TensorStorage::Additive {
            layout,
            planes: PlaneCounts::Uniform(count),
        } => {
            let tensor = AdditiveTensor::decode_payload(*layout, *count, bytes)
                .map_err(|e| NnError::InvalidArtifact(format!("{}: {e:?}", meta.name)))?;
            loader.upload(&meta.name, TensorView::Additive(tensor.view()))
        }
        _ => Err(NnError::Backend("unsupported model tensor payload".into())),
    }
}

fn load_trit(
    model: &TritModel<'_>,
    level: &str,
    loader: &mut TensorLoader<'_>,
) -> Result<(ModelConfig, ArchSpec, ModelWeights), NnError> {
    let config_bytes = model.asset("config.json").map_err(format_error)?;
    if u64::try_from(config_bytes.len()).unwrap_or(u64::MAX) > MAX_CONFIG_JSON_BYTES {
        return Err(NnError::MissingConfig(
            "config.json exceeds source size limit".into(),
        ));
    }
    let text = std::str::from_utf8(config_bytes)
        .map_err(|e| NnError::MissingConfig(format!("config.json: {e}")))?;
    let (config, mut spec) = ModelConfig::from_hf_config(text)?;
    if model.manifest().architecture != config.arch {
        return Err(NnError::MissingConfig(
            "manifest/config architecture mismatch".into(),
        ));
    }
    let json = serde_json::from_str(text)
        .map_err(|e| NnError::MissingConfig(format!("config.json: {e}")))?;
    let selected = model
        .manifest()
        .levels
        .iter()
        .find(|l| l.name == level)
        .ok_or_else(|| NnError::MissingTensor(level.into()))?;
    (spec.qkv_bias, spec.qk_norm) = resolve_optional_attention_names(
        &config,
        &json,
        selected.tensors.iter().map(|t| t.name.as_str()),
    )?;
    let consumed = RefCell::new(BTreeSet::new());
    let tensor = |name: &str| {
        let binding = model.tensor(level, name).map_err(format_error)?;
        consumed.borrow_mut().insert(name.to_owned());
        Ok::<_, NnError>(binding)
    };
    let (meta, bytes) = tensor(NameSchema::Hf.top("token_embd"))?;
    let (vocab, columns) = matrix_shape(meta)?;
    if columns != config.n_embd as usize || declared_vocab_size(&json)?.is_some_and(|n| n != vocab)
    {
        return Err(NnError::MissingTensor(
            "embedding geometry disagrees with config".into(),
        ));
    }
    let embedding =
        TokenEmbedding::from_linear(upload_matrix(loader, meta, bytes, vocab, columns)?);
    let loader = RefCell::new(loader);
    let vectors = |name: &str, request: DenseTensorRequest| {
        let DenseTensorRequest::Vector { len } = request else {
            return Err(NnError::Backend(
                "embedding must use the shared handle".into(),
            ));
        };
        let (meta, bytes) = tensor(name)?;
        if !matches!(&meta.storage, TensorStorage::Dense { shape, .. } if shape.as_slice() == [len as u64])
        {
            return Err(NnError::MissingTensor(format!(
                "{name} has incompatible vector geometry"
            )));
        }
        dense_values(meta, bytes)
    };
    let projections = |name: &str, rows, cols| {
        let (meta, bytes) = tensor(name)?;
        upload_matrix(&mut loader.borrow_mut(), meta, bytes, rows, cols).map(Projection::Tensor)
    };
    let weights = build_standard_model_with_embedding(
        &config,
        &spec,
        NameSchema::Hf,
        embedding,
        vectors,
        projections,
    )?;
    if selected
        .tensors
        .iter()
        .any(|t| !consumed.borrow().contains(&t.name))
    {
        return Err(NnError::MissingTensor(
            "unexpected tensor in the selected complete level".into(),
        ));
    }
    Ok((config, spec, weights))
}
