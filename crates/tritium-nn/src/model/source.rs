//! Shared admitted source loading; typed profile remains experimental (ADR 0054).

use std::{cell::RefCell, collections::BTreeMap, path::Path};

use half::{bf16, f16};
use tritium_format::trit_model::{TritModel, decode_additive_payload, read_trit_model};
use tritium_quantize::engine::JoinedSalienceModel;
#[cfg(feature = "tokenizer")]
use tritium_schema::{LoadedSourceIdentity, ModelLoaded, UnknownReason};
use tritium_schema::{ScalePrecision, TensorMeta, TensorStorage};
use tritium_spec::TensorView;

use super::{
    ModelWeights,
    hf::{
        DenseTensorRequest, MAX_CONFIG_JSON_BYTES, NameSchema, declared_vocab_size,
        open_hf_standard, resolve_optional_attention_names,
    },
    hf_shards::HfShardSet,
    standard::{StandardModelPlan, TensorGeometry},
};
use crate::{ArchSpec, ModelConfig, NnError, Projection, TensorLoader, TokenEmbedding};

#[cfg(feature = "tokenizer")]
const MAX_TOKENIZER_JSON_BYTES: u64 = 256 * 1024 * 1024;
#[cfg(feature = "tokenizer")]
const MAX_TOKENIZER_CONFIG_BYTES: u64 = 16 * 1024 * 1024;

/// Sources for shared admitted loading; retained files require the text loader.
///
/// Typed `.trit` is a proposed profile. Validated raw Uniform/PerTile additive
/// matrices use the shared tensor decoder and selected backend's explicit tier.
/// The common text loader also admits complete typed hybrid language/MTP graphs.
/// Shards/entropy and HF hybrid graphs still require later support, not a silent
/// fallback or independent compact-profile acceptance.
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
    /// Complete retained-file source for the common text loader (ADR0068).
    TritFile {
        /// Actual container path, opened once; external immutable custody required.
        path: &'a Path,
        /// Explicit complete selected level.
        level: &'a str,
        /// Total source/manifest/directory admission before retained-reader allocation.
        receiving: tritium_format::TritPackageReadLimits,
        /// Positive config/encoded/logical-decode limits; not total RSS.
        limits: super::Qwen35TritReadLimits,
    },
    /// Mandatory lineage carrier already admitted against the actual B stage
    /// and both logs by the shared joined verifier (proposed ADR 0064).
    /// This is structural admission, not transform arithmetic or quality proof.
    JoinedSalience {
        /// Checked view borrowing only C bytes; the B payload need not be retained.
        model: &'a JoinedSalienceModel<'a>,
        /// Explicit complete level, subject to the same architecture admission.
        level: &'a str,
    },
}

enum PreparedTrit<'a> {
    Owned(Box<TritModel<'a>>),
    Joined(&'a TritModel<'a>),
}

impl<'a> std::ops::Deref for PreparedTrit<'a> {
    type Target = TritModel<'a>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(model) => model,
            Self::Joined(model) => model,
        }
    }
}

enum PreparedSource<'a> {
    Hf {
        #[cfg(feature = "tokenizer")]
        path: &'a Path,
        shards: HfShardSet,
        config: ModelConfig,
        spec: ArchSpec,
        plan: StandardModelPlan,
    },
    Trit {
        model: PreparedTrit<'a>,
        level: &'a str,
        config: ModelConfig,
        spec: ArchSpec,
        plan: StandardModelPlan,
    },
}

impl<'a> PreparedSource<'a> {
    #[cfg(feature = "tokenizer")]
    fn identity(&self) -> Result<LoadedSourceIdentity, NnError> {
        match self {
            Self::Hf { .. } => Ok(LoadedSourceIdentity::HuggingFace {
                reason: UnknownReason::MissingCapability,
            }),
            Self::Trit { model, level, .. } => Ok(LoadedSourceIdentity::Trit {
                package_id: tritium_schema::PackageId::from_bytes(*model.package_id().as_bytes()),
                model_id: model.model_id(level).map_err(format_error)?,
                level: (*level).to_owned(),
            }),
        }
    }

    fn new(source: ModelSource<'a>) -> Result<Self, NnError> {
        match source {
            ModelSource::HuggingFace(path) => {
                let (config, spec, shards, declared_vocab) = open_hf_standard(path)?;
                let inventory: BTreeMap<_, _> = shards.metadata().map(|t| (t.name, t)).collect();
                let embedding = inventory
                    .get(NameSchema::Hf.top("token_embd"))
                    .ok_or_else(|| NnError::MissingTensor("embedding absent".into()))?;
                let vocab =
                    embedding.shape.first().copied().ok_or_else(|| {
                        NnError::MissingTensor("embedding must be a matrix".into())
                    })?;
                if declared_vocab.is_some_and(|n| n != vocab) {
                    return Err(NnError::MissingTensor(
                        "embedding vocabulary disagrees with config".into(),
                    ));
                }
                let plan = StandardModelPlan::new(&config, &spec, NameSchema::Hf, vocab)?;
                plan.admit_inventory(inventory.len(), |name| {
                    inventory.get(name).map(|actual| {
                        if matches!(actual.dtype, "F32" | "F16" | "BF16") {
                            TensorGeometry::Native(actual.shape)
                        } else {
                            TensorGeometry::Unsupported
                        }
                    })
                })?;
                Ok(Self::Hf {
                    #[cfg(feature = "tokenizer")]
                    path,
                    shards,
                    config,
                    spec,
                    plan,
                })
            }
            ModelSource::Trit { bytes, level } => {
                let model = read_trit_model(bytes).map_err(format_error)?;
                Self::typed(PreparedTrit::Owned(Box::new(model)), level)
            }
            ModelSource::JoinedSalience { model, level } => {
                Self::typed(PreparedTrit::Joined(model.model()), level)
            }
            ModelSource::TritFile { .. } => Err(NnError::MissingConfig(
                "retained file graphs require the common text load entrypoint".into(),
            )),
        }
    }

    fn typed(model: PreparedTrit<'a>, level: &'a str) -> Result<Self, NnError> {
        if !model.manifest().levels.iter().any(|l| l.name == level) {
            return Err(NnError::MissingTensor(format!(
                "absent model level {level}"
            )));
        }
        let (config, spec, plan) = prepare_trit(&model, level)?;
        Ok(Self::Trit {
            model,
            level,
            config,
            spec,
            plan,
        })
    }

    fn weights(
        &self,
        loader: &mut TensorLoader<'_>,
    ) -> Result<(ModelConfig, ArchSpec, ModelWeights), NnError> {
        match self {
            Self::Hf {
                shards,
                config,
                spec,
                plan,
                ..
            } => Ok((
                config.clone(),
                spec.clone(),
                ModelWeights::load_hf_admitted_impl(config, spec, shards, plan, loader)?,
            )),
            Self::Trit {
                model,
                level,
                config,
                spec,
                plan,
            } => Ok((
                config.clone(),
                spec.clone(),
                load_trit(model, level, config, spec, plan, loader)?,
            )),
        }
    }

    #[cfg(feature = "tokenizer")]
    fn tokenizer(&self) -> Result<super::HfJsonTokenizer, NnError> {
        match self {
            Self::Hf { path, .. } => {
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

    #[cfg(feature = "tokenizer")]
    fn vocab(&self) -> usize {
        match self {
            Self::Hf { plan, .. } | Self::Trit { plan, .. } => plan.vocab(),
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
    /// Config, complete name inventory, shape and storage admission precede
    /// matrix upload. Norm/bias vectors remain finite host data; matrices use the same
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
        loader.check_evidence()?;
        PreparedSource::new(source)?.weights(loader)
    }
}

/// Actual text tokenizer, architecture, runner and recorded initial-load view.
///
/// The receipt is not independent qualification or current mutable runner state.
#[cfg(feature = "tokenizer")]
#[allow(missing_debug_implementations)]
pub struct LoadedModel {
    /// Common language-stream runner, owning its actual graph and private cache.
    pub runner: super::Runner,
    /// Parsed real tokenizer and special-token contract.
    pub tokenizer: super::HfJsonTokenizer,
    /// Initial architecture snapshot; the runner's immutable accessor is authoritative.
    pub architecture: super::ArchitectureSpec,
    /// Immutable view of the registered initial-load observation.
    pub receipt: tritium_evidence::ModelLoadReceiptView,
}

/// Load a text model through one source selector and shared admitted execution.
///
/// Checks config, complete name inventory, shapes/storage and recorder context
/// before matrix upload. Parses tokenizer assets and checks all tokenizer IDs
/// against embedding rows before upload too. A `.trit` container is parsed only
/// once; no filesystem extraction or retained dense model shadow is introduced.
/// Tokenizer JSON is bounded to 256 MiB and its config to 16 MiB; these are
/// source limits, not an accounting of parsed tokenizer or total model memory.
/// Standard and complete typed hybrid language graphs share the portable runner.
/// Hybrid MTP is privately bound but Unverified, not executable draft authority.
/// This is not resident-executor, full-memory or release qualification. File
/// sources require explicit receiving and per-tensor read/decode caps, and keep
/// one source handle through final source verification. Typed sources bind candidate
/// model/package IDs; HF identity remains explicitly unavailable. Final load
/// event recording still validates the actual payload and drops the unpublished
/// runner and payload leases on failure. HF numerical payload checks remain
/// seek-read during assembly; metadata preflight is not a full snapshot, payload
/// scan, capability/aggregate-budget admission or total-memory qualification.
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
    context
        .recorder
        .check_emit::<ModelLoaded>(context.span, context.parent, context.logical_time)
        .map_err(|error| NnError::Backend(format!("model load evidence: {error}")))?;
    context
        .recorder
        .check_emit::<tritium_evidence::TensorUploaded>(
            context.span,
            context.parent,
            context.logical_time,
        )
        .map_err(|error| NnError::Backend(format!("tensor upload evidence: {error}")))?;
    match source {
        ModelSource::TritFile {
            path,
            level,
            receiving,
            limits,
        } => {
            return super::retained::load_file(
                path, level, receiving, limits, backend, policy, budget, context,
            );
        }
        ModelSource::Trit { bytes, level } => {
            return super::retained::load_bytes(bytes, level, backend, policy, budget, context);
        }
        _ => {}
    }
    let source = PreparedSource::new(source)?;
    let identity = source.identity()?;
    let tokenizer = source.tokenizer()?;
    tokenizer.validate_model_vocab(source.vocab())?;
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
    let observation = ModelLoaded {
        source: identity,
        architecture: config.arch.clone(),
        backend: backend.device_id().to_owned(),
        physical_device: backend.physical_device_id().to_owned(),
        vocab_size: u64::try_from(weights.vocab)
            .map_err(|_| NnError::Backend("model vocabulary exceeds u64".into()))?,
        layer_count: u64::from(config.n_layers),
    };
    let runner = super::Runner::standard(
        super::ModelRunner::try_from_weights(config, weights, backend)?,
        architecture,
    );
    let architecture = runner.architecture().clone();
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

pub(super) fn dense_values(meta: &TensorMeta, bytes: &[u8]) -> Result<Vec<f32>, NnError> {
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

pub(super) fn matrix_shape(meta: &TensorMeta) -> Result<(usize, usize), NnError> {
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

pub(super) fn upload_matrix(
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
        TensorStorage::Additive { .. } => {
            let tensor = decode_additive_payload(meta, bytes)
                .map_err(|e| NnError::InvalidArtifact(format!("{}: {e}", meta.name)))?;
            loader.upload(&meta.name, TensorView::Additive(tensor.view()))
        }
        _ => Err(NnError::Backend("unsupported model tensor payload".into())),
    }
}

fn prepare_trit(
    model: &TritModel<'_>,
    level: &str,
) -> Result<(ModelConfig, ArchSpec, StandardModelPlan), NnError> {
    let config_bytes = model.asset("config.json").map_err(format_error)?;
    prepare_trit_metadata(model.manifest(), level, config_bytes)
}

pub(super) fn prepare_trit_metadata(
    manifest: &tritium_schema::ModelManifest,
    level: &str,
    config_bytes: &[u8],
) -> Result<(ModelConfig, ArchSpec, StandardModelPlan), NnError> {
    if u64::try_from(config_bytes.len()).unwrap_or(u64::MAX) > MAX_CONFIG_JSON_BYTES {
        return Err(NnError::MissingConfig(
            "config.json exceeds source size limit".into(),
        ));
    }
    let text = std::str::from_utf8(config_bytes)
        .map_err(|e| NnError::MissingConfig(format!("config.json: {e}")))?;
    let (config, mut spec) = ModelConfig::from_hf_config(text)?;
    if manifest.architecture != config.arch {
        return Err(NnError::MissingConfig(
            "manifest/config architecture mismatch".into(),
        ));
    }
    let json = serde_json::from_str(text)
        .map_err(|e| NnError::MissingConfig(format!("config.json: {e}")))?;
    let selected = manifest
        .levels
        .iter()
        .find(|l| l.name == level)
        .ok_or_else(|| NnError::MissingTensor(level.into()))?;
    (spec.qkv_bias, spec.qk_norm) = resolve_optional_attention_names(
        &config,
        &json,
        selected.tensors.iter().map(|t| t.name.as_str()),
    )?;
    let meta = selected
        .tensors
        .iter()
        .find(|t| t.name == NameSchema::Hf.top("token_embd"))
        .ok_or_else(|| NnError::MissingTensor("embedding absent".into()))?;
    let (vocab, columns) = matrix_shape(meta)?;
    if columns != config.n_embd as usize || declared_vocab_size(&json)?.is_some_and(|n| n != vocab)
    {
        return Err(NnError::MissingTensor(
            "embedding geometry disagrees with config".into(),
        ));
    }
    let plan = StandardModelPlan::new(&config, &spec, NameSchema::Hf, vocab)?;
    let inventory: BTreeMap<_, _> = selected
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    plan.admit_inventory(inventory.len(), |name| {
        inventory.get(name).map(|meta| match &meta.storage {
            TensorStorage::Dense { shape, .. } => TensorGeometry::Stored(shape),
            TensorStorage::Additive { layout, .. } => TensorGeometry::Matrix {
                rows: layout.rows,
                cols: layout.cols,
            },
            _ => TensorGeometry::Unsupported,
        })
    })?;
    Ok((config, spec, plan))
}

fn load_trit(
    model: &TritModel<'_>,
    level: &str,
    config: &ModelConfig,
    spec: &ArchSpec,
    plan: &StandardModelPlan,
    loader: &mut TensorLoader<'_>,
) -> Result<ModelWeights, NnError> {
    let tensor = |name: &str| model.tensor(level, name).map_err(format_error);
    let (meta, bytes) = tensor(&plan.embedding.name)?;
    let (vocab, columns) = matrix_shape(meta)?;
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
    let weights = plan.assemble(config, spec, embedding, vectors, projections)?;
    Ok(weights)
}
