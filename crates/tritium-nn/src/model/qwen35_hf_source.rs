//! Once-opened, content-bound Hugging Face sources for Qwen3.5-family models.

use std::convert::Infallible;
use std::error::Error;
use std::fmt;
use std::path::Path;

use tritium_format::{
    ModelId, SafeTensorsError, SemanticModelManifest, SemanticTensor, SemanticTensorHasher,
};
use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::model::hf::read_config_json;
use crate::model::hf_shards::{HfShardSet, HfTensorBytesError};
use crate::model::qwen35_hf::{
    Qwen35HfLanguageModel, Qwen35HfLanguageMtpModel, Qwen35HfLanguageReceipt, Qwen35HfTensorSource,
    language_schema, load_language_weights, load_mtp_weights, mtp_schema, preflight_mtp_source,
    preflight_source,
};
use crate::model::{Qwen35TextRunner, UnverifiedQwen35Mtp};
use crate::qwen35_config::{
    Qwen35CheckpointConfig, Qwen35DeltaNetConfig, Qwen35Dtype, Qwen35FullAttentionConfig,
    Qwen35LayerType, Qwen35MtpConfig, Qwen35NormWeightSemantics, Qwen35OutputGate,
    Qwen35RopeConfig, Qwen35RopeType, Qwen35TextConfig, Qwen35VisionScope,
};

/// Semantic-manifest architecture for the content-bound Qwen3.5-family source adapter.
pub const QWEN35_HF_SOURCE_ARCHITECTURE: &str =
    "qwen3_5::content_bound_language_with_deferred_mtp_vision_v1";
const SOURCE_CONFIG_MAGIC: &[u8; 8] = b"TRQ35SC\0";
const SOURCE_CONFIG_VERSION: u8 = 1;
const SOURCE_TENSOR_MAGIC: &[u8; 8] = b"TRQ35HF\0";
const SOURCE_TENSOR_VERSION: u8 = 1;
const SOURCE_STREAM_CHUNK_BYTES: usize = 64 * 1024;

/// Owned canonical metadata for one tensor in a once-opened HF source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35HfTensorMetadata {
    name: String,
    dtype: String,
    shape: Vec<u64>,
}

/// Failure while streaming exact bytes from a content-verified Qwen source tensor.
#[derive(Debug)]
pub enum Qwen35TensorStreamError<E> {
    /// Retained source metadata, payload, or semantic identity failed validation.
    Source(NnError),
    /// Caller-provided byte sink stopped the stream.
    Sink(E),
}

impl<E: fmt::Display> fmt::Display for Qwen35TensorStreamError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(error) => {
                write!(formatter, "Qwen3.5 tensor source stream failed: {error}")
            }
            Self::Sink(error) => write!(formatter, "Qwen3.5 tensor byte sink failed: {error}"),
        }
    }
}

impl<E: Error + 'static> Error for Qwen35TensorStreamError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::Sink(error) => Some(error),
        }
    }
}

impl Qwen35HfTensorMetadata {
    /// Canonical tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// SafeTensors logical storage dtype.
    #[must_use]
    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    /// Logical tensor shape in SafeTensors order.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

/// Content-derived semantic identity of a Qwen HF source boundary.
///
/// Every present tensor payload, including deferred MTP and vision tensors, is
/// bit-bound. Completeness is not implied; a later campaign typestate joins the
/// pinned revision and exact coverage. The v1 canonical configuration represents
/// vision only as deferred, so this is not an executable multimodal identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen35HfSourceIdentity {
    manifest: SemanticModelManifest,
    payload_bytes: u64,
}

impl Qwen35HfSourceIdentity {
    /// Semantic source-model identity consumed by campaign and conversion records.
    #[must_use]
    pub fn model_id(&self) -> ModelId {
        self.manifest.model_id()
    }

    /// Canonical semantic manifest proving possession of the model identity.
    #[must_use]
    pub const fn manifest(&self) -> &SemanticModelManifest {
        &self.manifest
    }

    /// Exact stored tensor payload bytes streamed into this identity.
    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }
}

/// Once-opened, seek-backed Qwen3.5-family HF source capability.
///
/// Headers and canonical metadata remain resident. Tensor payloads stay on disk
/// and are read only through the original open shard handles.
#[derive(Debug)]
pub struct Qwen35HfSource {
    config: Qwen35CheckpointConfig,
    canonical_config: Vec<u8>,
    metadata: Vec<Qwen35HfTensorMetadata>,
    shards: HfShardSet,
}

impl Qwen35HfSource {
    /// Open and validate `config.json`, the shard index, and every shard header.
    ///
    /// This operation does not read tensor payloads or widen the model.
    ///
    /// # Errors
    /// Returns [`NnError`] for an invalid Qwen configuration, unsafe/incomplete
    /// shard index, duplicate tensor, malformed header, or bounded allocation
    /// failure.
    pub fn open(dir: &Path) -> Result<Self, NnError> {
        let config_json = read_config_json(&dir.join("config.json"))?;
        let config = Qwen35CheckpointConfig::from_hf_config(&config_json)?;
        let canonical_config = canonical_source_config(&config)?;
        let shards = HfShardSet::open(dir)?;
        let metadata = owned_metadata(&shards)?;
        Ok(Self {
            config,
            canonical_config,
            metadata,
            shards,
        })
    }

    /// Validated family configuration represented by this source.
    #[must_use]
    pub const fn config(&self) -> &Qwen35CheckpointConfig {
        &self.config
    }

    /// Versioned canonical language-plus-MTP configuration bytes.
    #[must_use]
    pub fn canonical_config_bytes(&self) -> &[u8] {
        &self.canonical_config
    }

    /// Complete canonical tensor metadata in global tensor-name order.
    #[must_use]
    pub fn metadata(&self) -> &[Qwen35HfTensorMetadata] {
        &self.metadata
    }

    /// Consume the opened source by streaming every tensor into a semantic identity.
    ///
    /// The returned typestate retains these same open handles. Its tensor reads
    /// re-hash the exact raw chunks widened and reject any in-place mutation
    /// since this verification pass.
    ///
    /// # Errors
    /// Returns [`NnError`] for an unsupported dtype, source read/mutation error,
    /// byte-count overflow, or invalid semantic manifest.
    pub fn verify_semantic_identity(self) -> Result<Qwen35ContentVerifiedHfSource, NnError> {
        let schema = language_schema(&self.config.text)?;
        let language_receipt = preflight_source(&self.shards, &schema)?;
        let identity = derive_source_identity(&self)?;
        Ok(Qwen35ContentVerifiedHfSource {
            source: self,
            identity,
            language_receipt,
        })
    }
}

/// A same-handle HF source whose present payloads and language schema are verified.
///
/// This is not pinned campaign proof: repository revision, exact 1,199-tensor
/// coverage, and conversion policy are joined by a later campaign typestate.
#[derive(Debug)]
pub struct Qwen35ContentVerifiedHfSource {
    source: Qwen35HfSource,
    identity: Qwen35HfSourceIdentity,
    language_receipt: Qwen35HfLanguageReceipt,
}

impl Qwen35ContentVerifiedHfSource {
    /// Validated family configuration bound into this source identity.
    #[must_use]
    pub const fn config(&self) -> &Qwen35CheckpointConfig {
        &self.source.config
    }

    /// Versioned canonical language-plus-MTP configuration bytes bound into the manifest.
    #[must_use]
    pub fn canonical_config_bytes(&self) -> &[u8] {
        &self.source.canonical_config
    }

    /// Complete canonical source metadata.
    #[must_use]
    pub fn metadata(&self) -> &[Qwen35HfTensorMetadata] {
        &self.source.metadata
    }

    /// Semantic identity derived from every source payload.
    #[must_use]
    pub const fn identity(&self) -> &Qwen35HfSourceIdentity {
        &self.identity
    }

    /// Semantic model ID derived from the retained manifest.
    #[must_use]
    pub fn model_id(&self) -> ModelId {
        self.identity.model_id()
    }

    /// Language schema coverage proven before payload identity streaming.
    #[must_use]
    pub const fn language_receipt(&self) -> &Qwen35HfLanguageReceipt {
        &self.language_receipt
    }

    /// Widen one tensor while proving the exact consumed chunks still match the manifest.
    ///
    /// This is the bounded tensor-at-a-time seam for later campaign conversion.
    ///
    /// # Errors
    /// Returns [`NnError`] if the tensor is absent, its metadata is inconsistent,
    /// its payload changed since verification, or widening fails.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, NnError> {
        let metadata = find_metadata(&self.source.metadata, name)?;
        let mut expected = Vec::new();
        expected
            .try_reserve_exact(metadata.shape.len())
            .map_err(|error| {
                NnError::Backend(format!(
                    "allocate verified shape for Qwen3.5 tensor `{name}`: {error}"
                ))
            })?;
        for &axis in &metadata.shape {
            expected.push(usize::try_from(axis).map_err(|_| {
                NnError::Backend(format!(
                    "Qwen3.5 tensor `{name}` axis {axis} exceeds host usize"
                ))
            })?);
        }
        self.tensor_f32_exact(name, &expected)
    }

    /// Construct the architecture-framed semantic hasher for one source tensor.
    ///
    /// This lets durable downstream stores revalidate copied raw payload bytes
    /// against the same tensor identity used by source admission without
    /// duplicating Qwen framing constants.
    ///
    /// # Errors
    /// Returns [`NnError`] if the tensor is absent, its dtype is unsupported, or
    /// bounded owned metadata cannot be constructed.
    pub fn source_tensor_semantic_hasher(
        &self,
        name: &str,
    ) -> Result<SemanticTensorHasher, NnError> {
        let metadata = find_metadata(&self.source.metadata, name)?;
        let dtype_tag = source_dtype_tag(name, &metadata.dtype)?;
        let hasher_name = try_owned_string(name, "verified semantic tensor name")?;
        let hasher_shape = try_owned_u64_shape(&metadata.shape, name)?;
        let mut hasher = SemanticTensorHasher::new(hasher_name, hasher_shape);
        update_tensor_frame(&mut hasher, dtype_tag);
        Ok(hasher)
    }

    /// Stream exact stored tensor bytes from retained shard handles.
    ///
    /// Chunks never exceed `max_chunk_bytes`. The same semantic frame used by
    /// source admission is recomputed around the streamed payload and compared
    /// with the retained manifest before success is returned.
    ///
    /// The retained source is live, not a filesystem snapshot, and callbacks
    /// are nontransactional. The caller must prevent concurrent source mutation
    /// for the full visit and discard partial sink effects on any error.
    ///
    /// # Errors
    /// Returns [`Qwen35TensorStreamError::Source`] for absent, changed, or
    /// malformed source content and [`Qwen35TensorStreamError::Sink`] without
    /// erasing the caller's typed sink failure.
    pub fn try_visit_tensor_bytes<E>(
        &self,
        name: &str,
        max_chunk_bytes: usize,
        mut visit: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<u64, Qwen35TensorStreamError<E>> {
        let expected_semantic = find_semantic_tensor(self.identity.manifest(), name)
            .map_err(Qwen35TensorStreamError::Source)?;
        let mut hasher = self
            .source_tensor_semantic_hasher(name)
            .map_err(Qwen35TensorStreamError::Source)?;
        let mut payload_bytes = 0_u64;
        self.source
            .shards
            .try_visit_tensor_bytes(name, max_chunk_bytes, |chunk| {
                let chunk_bytes = u64::try_from(chunk.len()).map_err(|_| {
                    VerifiedStreamSinkError::Source(NnError::Backend(
                        "Qwen3.5 tensor stream chunk length exceeds u64".into(),
                    ))
                })?;
                payload_bytes = payload_bytes.checked_add(chunk_bytes).ok_or_else(|| {
                    VerifiedStreamSinkError::Source(NnError::Backend(
                        "Qwen3.5 tensor stream byte count overflow".into(),
                    ))
                })?;
                hasher.update(chunk);
                visit(chunk).map_err(VerifiedStreamSinkError::Sink)
            })
            .map_err(map_verified_stream_error)?;
        let actual = hasher
            .finalize()
            .map_err(|error| {
                NnError::MissingTensor(format!(
                    "finalize streamed Qwen3.5 tensor `{name}`: {error}"
                ))
            })
            .map_err(Qwen35TensorStreamError::Source)?;
        if &actual != expected_semantic {
            return Err(Qwen35TensorStreamError::Source(NnError::MissingTensor(
                format!(
                    "verified Qwen3.5 source tensor `{name}` changed after semantic identity verification"
                ),
            )));
        }
        Ok(payload_bytes)
    }

    /// Consume the verified source into the exact-fp32 language reference.
    ///
    /// Every language tensor is checked against the manifest from the exact raw
    /// chunks widened. MTP and vision remain deferred by this reference graph.
    ///
    /// # Errors
    /// Returns an error for incomplete/unknown language coverage, changed source
    /// bytes, wrong shapes/dtypes, decode failures, or runner construction.
    pub fn load_language(
        self,
        backend: Box<dyn TernaryBackend>,
    ) -> Result<Qwen35HfLanguageModel, NnError> {
        let weights = load_language_weights(&self, &self.source.config.text)?;
        let runner = Qwen35TextRunner::new(&self.source.config.text, weights, backend)?;
        let Self {
            source,
            identity,
            language_receipt,
        } = self;
        Ok(Qwen35HfLanguageModel::from_verified_source(
            source.config,
            runner,
            language_receipt,
            identity,
        ))
    }

    /// Consume the verified source into exact-fp32 language and MTP graphs.
    ///
    /// The combined loader requires exactly the official 15-tensor `mtp.*`
    /// schema. It returns the MTP graph structurally unverified; execution remains
    /// unavailable until [`Qwen35HfLanguageMtpModel::verify_mtp`] matches a pinned
    /// pinned vLLM fixture prefill/decode trace.
    ///
    /// # Errors
    /// Returns an error for a missing, unknown, duplicate, wrong-shape, changed,
    /// or unsupported MTP tensor, or for language/MTP assembly failure.
    pub fn load_language_mtp(
        self,
        backend: Box<dyn TernaryBackend>,
    ) -> Result<Qwen35HfLanguageMtpModel, NnError> {
        let mtp_schema = mtp_schema(&self.source.config.text)?;
        let receipt =
            preflight_mtp_source(&self.source.shards, &mtp_schema, self.language_receipt)?;
        let language_weights = load_language_weights(&self, &self.source.config.text)?;
        let mtp_weights = load_mtp_weights(&self, &self.source.config.text)?;
        let runner = Qwen35TextRunner::new(&self.source.config.text, language_weights, backend)?;
        let mtp = UnverifiedQwen35Mtp::new(&runner, mtp_weights)?;
        let Self {
            source,
            identity,
            language_receipt: _,
        } = self;
        Ok(Qwen35HfLanguageMtpModel::from_verified_source(
            source.config,
            runner,
            mtp,
            receipt,
            identity,
        ))
    }
}

impl Qwen35HfTensorSource for Qwen35ContentVerifiedHfSource {
    fn tensor_f32_exact(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, NnError> {
        let metadata = find_metadata(&self.source.metadata, name)?;
        let expected_semantic = find_semantic_tensor(self.identity.manifest(), name)?;
        let dtype_tag = source_dtype_tag(name, &metadata.dtype)?;
        let hasher_name = try_owned_string(name, "verified semantic tensor name")?;
        let hasher_shape = try_owned_u64_shape(&metadata.shape, name)?;
        let mut hasher = SemanticTensorHasher::new(hasher_name, hasher_shape);
        update_tensor_frame(&mut hasher, dtype_tag);
        let values = self
            .source
            .shards
            .try_tensor_f32_exact_with_raw_chunks(
                name,
                expected,
                SOURCE_STREAM_CHUNK_BYTES,
                |chunk| {
                    hasher.update(chunk);
                    Ok::<(), Infallible>(())
                },
            )
            .map_err(|error| map_hf_error(error, |never| match never {}))?;
        let actual = hasher.finalize().map_err(|error| {
            NnError::MissingTensor(format!(
                "finalize verified Qwen3.5 tensor `{name}`: {error}"
            ))
        })?;
        if &actual != expected_semantic {
            return Err(NnError::MissingTensor(format!(
                "verified Qwen3.5 source tensor `{name}` changed after semantic identity verification"
            )));
        }
        Ok(values)
    }
}

fn derive_source_identity(source: &Qwen35HfSource) -> Result<Qwen35HfSourceIdentity, NnError> {
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(source.metadata.len())
        .map_err(|error| {
            NnError::Backend(format!(
                "allocate {} Qwen3.5 semantic tensor digests: {error}",
                source.metadata.len()
            ))
        })?;
    let mut payload_bytes = 0u64;
    for metadata in &source.metadata {
        let dtype_tag = source_dtype_tag(&metadata.name, &metadata.dtype)?;
        let hasher_name = try_owned_string(&metadata.name, "semantic tensor name")?;
        let hasher_shape = try_owned_u64_shape(&metadata.shape, &metadata.name)?;
        let mut hasher = SemanticTensorHasher::new(hasher_name, hasher_shape);
        update_tensor_frame(&mut hasher, dtype_tag);
        source
            .shards
            .try_visit_tensor_bytes(
                &metadata.name,
                SOURCE_STREAM_CHUNK_BYTES,
                |chunk| -> Result<(), NnError> {
                    let chunk_bytes = u64::try_from(chunk.len()).map_err(|_| {
                        NnError::Backend(
                            "Qwen3.5 source chunk length exceeds u64 byte count".into(),
                        )
                    })?;
                    payload_bytes = payload_bytes.checked_add(chunk_bytes).ok_or_else(|| {
                        NnError::Backend("Qwen3.5 source payload byte count overflow".into())
                    })?;
                    hasher.update(chunk);
                    Ok(())
                },
            )
            .map_err(|error| map_hf_error(error, |sink| sink))?;
        tensors.push(hasher.finalize().map_err(|error| {
            NnError::MissingTensor(format!(
                "build semantic identity for tensor `{}`: {error}",
                metadata.name
            ))
        })?);
    }
    let architecture = try_owned_string(QWEN35_HF_SOURCE_ARCHITECTURE, "semantic architecture")?;
    let manifest = SemanticModelManifest::new(architecture, &source.canonical_config, tensors)
        .map_err(|error| {
            NnError::MissingConfig(format!("build Qwen3.5 semantic source manifest: {error}"))
        })?;
    Ok(Qwen35HfSourceIdentity {
        manifest,
        payload_bytes,
    })
}

fn update_tensor_frame(hasher: &mut SemanticTensorHasher, dtype_tag: u8) {
    hasher.update(SOURCE_TENSOR_MAGIC);
    hasher.update(&[SOURCE_TENSOR_VERSION, dtype_tag]);
}

fn source_dtype_tag(name: &str, dtype: &str) -> Result<u8, NnError> {
    match dtype {
        "BF16" => Ok(0),
        "F16" => Ok(1),
        "F32" => Ok(2),
        _ => Err(NnError::MissingTensor(format!(
            "tensor `{name}` has unsupported semantic source dtype `{dtype}`"
        ))),
    }
}

fn find_metadata<'a>(
    metadata: &'a [Qwen35HfTensorMetadata],
    name: &str,
) -> Result<&'a Qwen35HfTensorMetadata, NnError> {
    metadata
        .binary_search_by(|tensor| tensor.name.as_str().cmp(name))
        .ok()
        .map(|index| &metadata[index])
        .ok_or_else(|| NnError::MissingTensor(name.to_owned()))
}

fn find_semantic_tensor<'a>(
    manifest: &'a SemanticModelManifest,
    name: &str,
) -> Result<&'a SemanticTensor, NnError> {
    manifest
        .tensors()
        .binary_search_by(|tensor| tensor.name().cmp(name))
        .ok()
        .map(|index| &manifest.tensors()[index])
        .ok_or_else(|| {
            NnError::Backend(format!(
                "verified Qwen3.5 manifest omits source tensor `{name}`"
            ))
        })
}

fn owned_metadata(shards: &HfShardSet) -> Result<Vec<Qwen35HfTensorMetadata>, NnError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(shards.metadata().len())
        .map_err(|error| {
            NnError::Backend(format!(
                "allocate {} Qwen3.5 source metadata entries: {error}",
                shards.metadata().len()
            ))
        })?;
    for tensor in shards.metadata() {
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(tensor.shape.len())
            .map_err(|error| {
                NnError::Backend(format!(
                    "allocate shape for Qwen3.5 tensor `{}`: {error}",
                    tensor.name
                ))
            })?;
        for &axis in tensor.shape {
            shape.push(u64::try_from(axis).map_err(|_| {
                NnError::Backend(format!(
                    "Qwen3.5 tensor `{}` axis {axis} exceeds u64",
                    tensor.name
                ))
            })?);
        }
        let name = try_owned_string(tensor.name, "source tensor name")?;
        let dtype = try_owned_string(tensor.dtype, "source tensor dtype")?;
        output.push(Qwen35HfTensorMetadata { name, dtype, shape });
    }
    Ok(output)
}

fn try_owned_string(value: &str, label: &str) -> Result<String, NnError> {
    let mut output = String::new();
    output.try_reserve_exact(value.len()).map_err(|error| {
        NnError::Backend(format!(
            "allocate {} bytes for Qwen3.5 {label}: {error}",
            value.len()
        ))
    })?;
    output.push_str(value);
    Ok(output)
}

fn try_owned_u64_shape(shape: &[u64], name: &str) -> Result<Vec<u64>, NnError> {
    let mut output = Vec::new();
    output.try_reserve_exact(shape.len()).map_err(|error| {
        NnError::Backend(format!(
            "allocate {} semantic axes for Qwen3.5 tensor `{name}`: {error}",
            shape.len()
        ))
    })?;
    output.extend_from_slice(shape);
    Ok(output)
}

enum VerifiedStreamSinkError<E> {
    Source(NnError),
    Sink(E),
}

fn map_verified_stream_error<E>(
    error: HfTensorBytesError<VerifiedStreamSinkError<E>>,
) -> Qwen35TensorStreamError<E> {
    match partition_hf_error(error) {
        HfErrorPartition::Source(error)
        | HfErrorPartition::Sink(VerifiedStreamSinkError::Source(error)) => {
            Qwen35TensorStreamError::Source(error)
        }
        HfErrorPartition::Sink(VerifiedStreamSinkError::Sink(error)) => {
            Qwen35TensorStreamError::Sink(error)
        }
    }
}

fn map_hf_error<E>(error: HfTensorBytesError<E>, map_sink: impl FnOnce(E) -> NnError) -> NnError {
    match partition_hf_error(error) {
        HfErrorPartition::Source(error) => error,
        HfErrorPartition::Sink(error) => map_sink(error),
    }
}

enum HfErrorPartition<E> {
    Source(NnError),
    Sink(E),
}

fn partition_hf_error<E>(error: HfTensorBytesError<E>) -> HfErrorPartition<E> {
    match error {
        HfTensorBytesError::MissingTensor(name) => {
            HfErrorPartition::Source(NnError::MissingTensor(name))
        }
        HfTensorBytesError::ShapeMismatch {
            name,
            shard_path,
            actual,
            expected,
        } => HfErrorPartition::Source(NnError::MissingTensor(format!(
            "tensor `{name}` in {} has shape {actual:?}, expected {expected:?}",
            shard_path.display()
        ))),
        HfTensorBytesError::Source {
            name,
            shard_path,
            error:
                error @ (SafeTensorsError::AllocationFailed { .. }
                | SafeTensorsError::InvalidChunkSize { .. }),
        } => HfErrorPartition::Source(NnError::Backend(format!(
            "read Qwen3.5 tensor `{name}` from {}: {error}",
            shard_path.display()
        ))),
        HfTensorBytesError::Source {
            name,
            shard_path,
            error,
        } => HfErrorPartition::Source(NnError::MissingTensor(format!(
            "read Qwen3.5 tensor `{name}` from {}: {error}",
            shard_path.display()
        ))),
        HfTensorBytesError::ReentrantShard(path) => {
            HfErrorPartition::Source(NnError::Backend(format!(
                "reentrant read of Qwen3.5 safetensors shard {}",
                path.display()
            )))
        }
        HfTensorBytesError::InvalidShardId { name, shard_id } => {
            HfErrorPartition::Source(NnError::Backend(format!(
                "Qwen3.5 tensor `{name}` references absent retained shard {shard_id}"
            )))
        }
        HfTensorBytesError::MetadataMismatch { name, shard_path } => {
            HfErrorPartition::Source(NnError::Backend(format!(
                "Qwen3.5 tensor `{name}` metadata disagrees with retained shard {}",
                shard_path.display()
            )))
        }
        HfTensorBytesError::Sink(error) => HfErrorPartition::Sink(error),
    }
}

fn canonical_source_config(config: &Qwen35CheckpointConfig) -> Result<Vec<u8>, NnError> {
    let Qwen35CheckpointConfig {
        model_type,
        architecture,
        language_model_only,
        tied_embeddings: checkpoint_tied_embeddings,
        text,
        vision_scope,
    } = config;
    let Qwen35TextConfig {
        model_type: text_model_type,
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
        max_position_embeddings,
        full_attention_interval,
        layer_types,
        full_attention,
        delta_net,
        rope,
        rms_norm_eps,
        source_dtype,
        use_cache,
        tied_embeddings: text_tied_embeddings,
        mtp,
    } = text;
    let Qwen35FullAttentionConfig {
        num_heads,
        num_key_value_heads,
        head_dim,
        bias,
        dropout,
        output_gate: attention_output_gate,
        norm_weight_semantics: attention_norm_semantics,
    } = full_attention;
    let Qwen35DeltaNetConfig {
        conv_kernel_dim,
        num_key_heads,
        num_value_heads,
        key_head_dim,
        value_head_dim,
        state_arithmetic_dtype,
        output_gate: delta_output_gate,
        gated_norm_weight_semantics,
    } = delta_net;
    let Qwen35RopeConfig {
        theta,
        partial_rotary_factor,
        rotary_dim,
        rope_type,
        mrope_interleaved,
        mrope_section,
    } = rope;
    let Qwen35MtpConfig {
        num_hidden_layers: mtp_num_hidden_layers,
        dedicated_embeddings,
    } = mtp;

    let capacity = 512usize
        .checked_add(layer_types.len())
        .ok_or_else(|| NnError::Backend("Qwen3.5 canonical config size overflow".into()))?;
    let mut output = Vec::new();
    output.try_reserve_exact(capacity).map_err(|error| {
        NnError::Backend(format!(
            "allocate Qwen3.5 canonical source configuration: {error}"
        ))
    })?;
    output.extend_from_slice(SOURCE_CONFIG_MAGIC);
    output.push(SOURCE_CONFIG_VERSION);
    write_string(&mut output, model_type)?;
    write_string(&mut output, architecture)?;
    write_bool(&mut output, *language_model_only);
    write_bool(&mut output, *checkpoint_tied_embeddings);
    write_string(&mut output, text_model_type)?;
    output.push(0); // SwiGLU with SiLU activation.
    for value in [
        *num_hidden_layers,
        *hidden_size,
        *intermediate_size,
        *vocab_size,
        *max_position_embeddings,
        *full_attention_interval,
    ] {
        write_u32(&mut output, value);
    }
    write_u32(
        &mut output,
        u32::try_from(layer_types.len())
            .map_err(|_| NnError::MissingConfig("Qwen3.5 layer schedule exceeds u32".into()))?,
    );
    for &layer in layer_types {
        output.push(layer_type_tag(layer));
    }

    for value in [*num_heads, *num_key_value_heads, *head_dim] {
        write_u32(&mut output, value);
    }
    write_bool(&mut output, *bias);
    write_f64(&mut output, *dropout, "attention dropout")?;
    output.push(output_gate_tag(*attention_output_gate));
    output.push(norm_semantics_tag(*attention_norm_semantics));

    for value in [
        *conv_kernel_dim,
        *num_key_heads,
        *num_value_heads,
        *key_head_dim,
        *value_head_dim,
    ] {
        write_u32(&mut output, value);
    }
    output.push(dtype_tag(*state_arithmetic_dtype));
    output.push(output_gate_tag(*delta_output_gate));
    output.push(norm_semantics_tag(*gated_norm_weight_semantics));

    write_f64(&mut output, *theta, "RoPE theta")?;
    write_f64(&mut output, *partial_rotary_factor, "partial rotary factor")?;
    write_u32(&mut output, *rotary_dim);
    output.push(rope_type_tag(*rope_type));
    output.push(0); // No RoPE scaling.
    write_bool(&mut output, *mrope_interleaved);
    for &value in mrope_section {
        write_u32(&mut output, value);
    }

    write_f64(&mut output, *rms_norm_eps, "RMSNorm epsilon")?;
    output.push(dtype_tag(*source_dtype));
    write_bool(&mut output, *use_cache);
    write_bool(&mut output, *text_tied_embeddings);
    write_u32(&mut output, *mtp_num_hidden_layers);
    write_bool(&mut output, *dedicated_embeddings);
    output.push(vision_scope_tag(*vision_scope));
    Ok(output)
}

/// Exact canonical configuration selected by the pinned Qwen3.6-27B campaign.
///
/// # Errors
/// Returns [`NnError`] only if encoding the frozen typed configuration cannot
/// allocate its bounded record.
pub fn qwen36_27b_canonical_source_config() -> Result<Vec<u8>, NnError> {
    canonical_source_config(&Qwen35CheckpointConfig::pinned_qwen36_27b())
}

fn write_string(output: &mut Vec<u8>, value: &str) -> Result<(), NnError> {
    write_u32(
        output,
        u32::try_from(value.len())
            .map_err(|_| NnError::MissingConfig("Qwen3.5 config string exceeds u32".into()))?,
    );
    output.extend_from_slice(value.as_bytes());
    Ok(())
}

fn write_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn write_bool(output: &mut Vec<u8>, value: bool) {
    output.push(if value { 1 } else { 0 });
}

fn write_f64(output: &mut Vec<u8>, value: f64, label: &str) -> Result<(), NnError> {
    if !value.is_finite() {
        return Err(NnError::MissingConfig(format!(
            "Qwen3.5 canonical {label} must be finite"
        )));
    }
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    output.extend_from_slice(&bits.to_le_bytes());
    Ok(())
}

const fn layer_type_tag(value: Qwen35LayerType) -> u8 {
    match value {
        Qwen35LayerType::DeltaNet => 0,
        Qwen35LayerType::FullAttention => 1,
    }
}

const fn dtype_tag(value: Qwen35Dtype) -> u8 {
    match value {
        Qwen35Dtype::Bfloat16 => 0,
        Qwen35Dtype::Float32 => 1,
    }
}

const fn norm_semantics_tag(value: Qwen35NormWeightSemantics) -> u8 {
    match value {
        Qwen35NormWeightSemantics::ZeroCenteredOnePlusWeight => 0,
        Qwen35NormWeightSemantics::UnitCenteredDirectWeight => 1,
    }
}

const fn output_gate_tag(value: Qwen35OutputGate) -> u8 {
    match value {
        Qwen35OutputGate::Sigmoid => 0,
        Qwen35OutputGate::Swish => 1,
    }
}

const fn rope_type_tag(value: Qwen35RopeType) -> u8 {
    match value {
        Qwen35RopeType::Default => 0,
    }
}

const fn vision_scope_tag(value: Qwen35VisionScope) -> u8 {
    match value {
        Qwen35VisionScope::PresentDeferred => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINNED_CONFIG: &str = include_str!("../../tests/fixtures/qwen36-27b-config.json");

    #[test]
    fn frozen_canonical_config_matches_the_parsed_pinned_fixture() {
        let parsed = Qwen35CheckpointConfig::from_hf_config(PINNED_CONFIG).unwrap();
        parsed
            .validate_pinned_qwen36_27b(crate::QWEN36_27B_REVISION)
            .unwrap();
        assert_eq!(
            canonical_source_config(&parsed).unwrap(),
            qwen36_27b_canonical_source_config().unwrap()
        );
    }
}
