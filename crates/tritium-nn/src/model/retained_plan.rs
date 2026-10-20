//! Shared retained-source preparation. Logical observations, not backend admission.
use super::{
    ArchitectureSpec, CacheAdmission, HfJsonTokenizer, Qwen35TritReadLimits, StreamLimits,
    qwen35_trit::{check, collect, decoded_bytes, prepare_metadata},
    source::prepare_trit_metadata,
    standard::StandardModelPlan,
};
use crate::{ArchSpec, ModelConfig, NnError, Qwen35CheckpointConfig};
use std::{
    fs::File,
    io::{Read, Seek},
    path::Path,
};
use tritium_format::{TritPackageReadLimits, trit_model::TritModelReader};
use tritium_schema::LoadedSourceIdentity;

/// One selected tensor's logical encoded/decode requirements. Not owned backend bytes.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorLoadPlan {
    name: String,
    encoded_bytes: u64,
    logical_decoded_bytes: u64,
}
impl TensorLoadPlan {
    /// Actual selected manifest tensor name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Actual referenced encoded blob length (not deduplicated across tensor names).
    pub const fn encoded_bytes(&self) -> u64 {
        self.encoded_bytes
    }
    /// Logical full decoded buffers used by the current compatibility upload path.
    pub const fn logical_decoded_bytes(&self) -> u64 {
        self.logical_decoded_bytes
    }
}

/// Read-only source preparation through the same module as ordinary loading.
///
/// This is not a loaded model, backend support, fit, physical-memory, source-
/// authentication or release receipt. No full tensor is materialized/uploaded and no
/// ModelLoaded event is emitted. Backend/allocator/driver/tokenizer heap costs
/// remain unknown; source bytes and full-decoding sums are not process RSS.
#[derive(Debug, Clone)]
pub struct ModelLoadPlan {
    source: LoadedSourceIdentity,
    architecture: ArchitectureSpec,
    source_bytes: u64,
    tensors: Vec<TensorLoadPlan>,
    encoded_tensor_bytes_sum: u64,
    logical_decoded_tensor_bytes_sum: u64,
    maximum_tensor_working_payload_bytes: u64,
    tokenizer_encoded_bytes: u64,
    cache: Option<CacheAdmission>,
}
impl ModelLoadPlan {
    /// Actual package/model/selected-level binding; not official source authentication.
    pub fn source(&self) -> &LoadedSourceIdentity {
        &self.source
    }
    /// Config and exact inventory admitted by shared standard/hybrid preparation.
    pub fn architecture(&self) -> &ArchitectureSpec {
        &self.architecture
    }
    /// Actual complete retained file length, not resident memory.
    pub const fn source_bytes(&self) -> u64 {
        self.source_bytes
    }
    /// Selected manifest tensors; no per-tile counts or payloads are copied here.
    pub fn tensors(&self) -> &[TensorLoadPlan] {
        &self.tensors
    }
    /// Sum by selected tensor name; aliases are not a physical storage ledger.
    pub const fn encoded_tensor_bytes_sum(&self) -> u64 {
        self.encoded_tensor_bytes_sum
    }
    /// Sum of current logical full-decode buffers, not retained backend payload.
    pub const fn logical_decoded_tensor_bytes_sum(&self) -> u64 {
        self.logical_decoded_tensor_bytes_sum
    }
    /// Maximum encoded plus decoded payload pair for one tensor; excludes source
    /// metadata, allocator slack, backend copies, vectors, tokenizer and caches.
    pub const fn maximum_tensor_working_payload_bytes(&self) -> u64 {
        self.maximum_tensor_working_payload_bytes
    }
    /// Combined actual tokenizer JSON/config encoded lengths, not tokenizer heap.
    pub const fn tokenizer_encoded_bytes(&self) -> u64 {
        self.tokenizer_encoded_bytes
    }
    /// Actual requested logical host-cache plan; absent when no stream policy supplied.
    pub const fn cache_admission(&self) -> Option<CacheAdmission> {
        self.cache
    }
    /// Not planned by a backend in this source-only inspection. Unknown, never zero.
    pub const fn backend_payload_bytes(&self) -> Option<u64> {
        None
    }
}

/// Descriptive aggregate encoded-streaming costs, not runtime admission.
/// No backend is initialized, payload uploaded or ledger reserved by inspection.
/// The supplied planner declares logical costs; physical/support/quality/source
/// authentication and independently qualified release claims remain unknown.
#[derive(Debug, Clone)]
pub struct StreamedModelLoadPlan {
    source: ModelLoadPlan,
    covered: super::ModelResourceAdmission,
}
impl StreamedModelLoadPlan {
    /// Source identity, inventory and cache preparation shared with ordinary loading.
    pub fn source_plan(&self) -> &ModelLoadPlan {
        &self.source
    }
    /// Aggregate backend-declared matrix bytes, with tied graph handles counted once.
    pub const fn payload_bytes(&self) -> u64 {
        self.covered.payload_bytes
    }
    /// Configured resident envelope, when structural resident planning was requested.
    pub const fn resident_device_payload_limit(&self) -> Option<u64> {
        self.covered.resident_device_payload_limit
    }
    /// Checked logical device categories, not a reservation or physical measurement.
    pub const fn resident_device_payload_plan(&self) -> Option<tritium_spec::DecoderPayloadPlan> {
        self.covered.resident_device_payload_plan
    }
    /// Maximum bounded matrix source window or preserved encoded/decoded pair.
    pub const fn source_workspace_bytes(&self) -> u64 {
        self.covered.source_workspace_bytes
    }
    /// Maximum additional declared backend host-upload working bytes.
    pub const fn host_workspace_bytes(&self) -> u64 {
        self.covered.host_workspace_bytes
    }
    /// Maximum additional declared backend device-upload working bytes.
    pub const fn device_workspace_bytes(&self) -> u64 {
        self.covered.device_workspace_bytes
    }
    /// Widened preserved host tensors and maximum logical language cache.
    pub const fn retained_host_bytes(&self) -> u64 {
        self.covered.retained_host_bytes
    }
}

pub(super) enum Graph {
    Standard(Box<(ModelConfig, ArchSpec, StandardModelPlan)>),
    Hybrid(Box<Qwen35CheckpointConfig>),
}
pub(super) struct Prepared {
    pub(super) graph: Graph,
    pub(super) tokenizer: HfJsonTokenizer,
    pub(super) plan: ModelLoadPlan,
}
fn invalid(e: impl std::fmt::Display) -> NnError {
    NnError::InvalidArtifact(format!("retained source preparation: {e}"))
}
/// Open once, without waiting for a FIFO writer, then inspect that same handle.
/// A pathname precheck alone would race substitution with a non-regular source.
/// Regular-file symlinks keep their existing behavior; custody remains external.
pub(super) fn open_regular_source(path: &Path) -> std::io::Result<(File, std::fs::Metadata)> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "source must be a regular file",
        ));
    }
    Ok((file, metadata))
}
pub(super) fn positive(l: Qwen35TritReadLimits) -> Result<(), NnError> {
    if [
        l.max_config_bytes,
        l.max_tensor_bytes,
        l.max_decoded_tensor_bytes,
    ]
    .contains(&0)
    {
        return Err(invalid("read/decode caps must be positive"));
    }
    Ok(())
}
fn asset<R: Read + Seek>(
    model: &mut TritModelReader<R>,
    name: &str,
    max: u64,
) -> Result<Vec<u8>, NnError> {
    let n = model.asset_byte_len(name).map_err(invalid)?;
    check(n, max, "asset bytes")?;
    collect(n, |visit| model.visit_asset(name, visit).map_err(invalid))
}

pub(super) fn prepare<R: Read + Seek>(
    model: &mut TritModelReader<R>,
    level: &str,
    source_bytes: u64,
    limits: Qwen35TritReadLimits,
    execution: Option<StreamLimits>,
) -> Result<Prepared, NnError> {
    positive(limits)?;
    if let Some(execution) = execution {
        execution.validate()?;
    }
    model
        .validate_complete_payload_inventory()
        .map_err(invalid)?;
    model.verify().map_err(invalid)?;
    let bytes = asset(model, "config.json", limits.max_config_bytes)?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(invalid)?;
    let graph = if json["model_type"] == "qwen3_5" {
        Graph::Hybrid(Box::new(prepare_metadata(model, level, limits, &bytes)?))
    } else {
        Graph::Standard(Box::new(prepare_trit_metadata(
            model.manifest(),
            level,
            &bytes,
        )?))
    };
    let architecture = match &graph {
        Graph::Standard(plan) => ArchitectureSpec::Standard {
            config: plan.0.clone(),
            variation: plan.1.clone(),
            vocab_size: plan.2.embedding.shape[0],
        },
        Graph::Hybrid(config) => ArchitectureSpec::Hybrid((**config).clone()),
    };
    let cache = execution
        .map(|l| match &graph {
            Graph::Standard(plan) => CacheAdmission::standard(&plan.0, l),
            Graph::Hybrid(config) => CacheAdmission::hybrid(config, l),
        })
        .transpose()?;
    let selected = model
        .manifest()
        .levels
        .iter()
        .find(|l| l.name == level)
        .ok_or_else(|| invalid("absent selected level"))?;
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(selected.tensors.len())
        .map_err(invalid)?;
    let mut encoded_sum = 0u64;
    let mut decoded_sum = 0u64;
    let mut maximum = 0u64;
    for meta in &selected.tensors {
        let encoded = model.tensor_byte_len(level, &meta.name).map_err(invalid)?;
        let decoded = decoded_bytes(meta)?;
        check(encoded, limits.max_tensor_bytes, "encoded tensor bytes")?;
        check(
            decoded,
            limits.max_decoded_tensor_bytes,
            "logical decoded tensor bytes",
        )?;
        encoded_sum = encoded_sum
            .checked_add(encoded)
            .ok_or_else(|| invalid("encoded sum overflow"))?;
        decoded_sum = decoded_sum
            .checked_add(decoded)
            .ok_or_else(|| invalid("decode sum overflow"))?;
        maximum = maximum.max(
            encoded
                .checked_add(decoded)
                .ok_or_else(|| invalid("working pair overflow"))?,
        );
        tensors.push(TensorLoadPlan {
            name: meta.name.clone(),
            encoded_bytes: encoded,
            logical_decoded_bytes: decoded,
        });
    }
    let tokenizer_bytes = asset(model, "tokenizer.json", 256 * 1024 * 1024)?;
    let tokenizer_config = asset(model, "tokenizer_config.json", 16 * 1024 * 1024)?;
    let tokenizer_encoded_bytes = (tokenizer_bytes.len() as u64)
        .checked_add(tokenizer_config.len() as u64)
        .ok_or_else(|| invalid("tokenizer encoded sum overflow"))?;
    let tokenizer = HfJsonTokenizer::from_bytes(&tokenizer_bytes, &tokenizer_config)?;
    tokenizer.validate_model_vocab(architecture.vocab_size())?;
    let plan = ModelLoadPlan {
        source: LoadedSourceIdentity::Trit {
            package_id: tritium_schema::PackageId::from_bytes(*model.package_id().as_bytes()),
            model_id: model.model_id(level).map_err(invalid)?,
            level: level.to_owned(),
        },
        architecture,
        source_bytes,
        tensors,
        encoded_tensor_bytes_sum: encoded_sum,
        logical_decoded_tensor_bytes_sum: decoded_sum,
        maximum_tensor_working_payload_bytes: maximum,
        tokenizer_encoded_bytes,
        cache,
    };
    Ok(Prepared {
        graph,
        tokenizer,
        plan,
    })
}

/// Inspect an actual retained typed file using ordinary loader preparation.
/// No backend initialization, full tensor materialization/upload or model-load
/// evidence. The format reader still validates canonical codec windows.
/// External immutable custody and independent runtime/physical gates remain required.
/// # Errors
/// Same source/config/inventory/tokenizer/cap/stream faults as ordinary preparation,
/// overflow/allocation failure, or terminal source mutation.
pub fn inspect_trit_file(
    path: &Path,
    level: &str,
    receiving: TritPackageReadLimits,
    limits: Qwen35TritReadLimits,
    execution: Option<StreamLimits>,
) -> Result<ModelLoadPlan, NnError> {
    positive(limits)?;
    if let Some(execution) = execution {
        execution.validate()?;
    }
    let (file, metadata) = open_regular_source(path).map_err(invalid)?;
    let mut model = TritModelReader::open(file, receiving).map_err(invalid)?;
    let prepared = prepare(&mut model, level, metadata.len(), limits, execution)?;
    model.verify().map_err(invalid)?;
    Ok(prepared.plan)
}

/// Describe aggregate streamed-loader costs without initializing a backend.
/// Uses the ordinary loader's exact inventory and resource calculation, but does
/// not reserve resources, materialize/upload weights, or emit ModelLoaded.
/// Callers supply a pure structural planner, not a registry/device initializer.
/// No descriptive result may be substituted for actual load admission/custody.
/// # Errors
/// The errors of inspect_trit_file, unknown/invalid backend streaming plans,
/// encoded geometry/policy faults, aggregate overflow, or terminal source mutation.
#[allow(clippy::too_many_arguments)]
pub fn inspect_trit_file_streamed(
    path: &Path,
    level: &str,
    receiving: TritPackageReadLimits,
    limits: Qwen35TritReadLimits,
    execution: StreamLimits,
    policy: tritium_schema::TensorUploadPolicy,
    planner: impl FnMut(
        tritium_spec::TensorMetadata<'_>,
        tritium_spec::TensorStreamEncoding,
    )
        -> Result<Option<tritium_spec::TensorStreamPlan>, tritium_spec::BackendError>,
) -> Result<StreamedModelLoadPlan, NnError> {
    inspect_streamed_with(
        path,
        level,
        receiving,
        limits,
        execution,
        |model, prepared| {
            super::resources::describe_streamed(
                model,
                level,
                &prepared.graph,
                &prepared.plan,
                policy,
                planner,
            )
        },
    )
}

/// Describe resident encoded-streaming costs using the common source/resource plan.
/// Both injected planners must be pure: this does not initialize a backend,
/// reserve a budget, upload weights, emit ModelLoaded or certify physical fit.
/// # Errors
/// Source/preparation errors, unknown/inconsistent structural declarations,
/// an exceeded positive resident ceiling, overflow or terminal source mutation.
#[allow(clippy::too_many_arguments)]
pub fn inspect_trit_file_streamed_resident(
    path: &Path,
    level: &str,
    receiving: TritPackageReadLimits,
    limits: Qwen35TritReadLimits,
    execution: StreamLimits,
    policy: tritium_schema::TensorUploadPolicy,
    max_device_payload_bytes: u64,
    planner: impl FnMut(
        tritium_spec::TensorMetadata<'_>,
        tritium_spec::TensorStreamEncoding,
    )
        -> Result<Option<tritium_spec::TensorStreamPlan>, tritium_spec::BackendError>,
    decoder: impl FnMut(
        tritium_spec::DecoderAllocationMetadata<'_>,
        tritium_spec::DecoderLimits,
    )
        -> Result<Option<tritium_spec::DecoderPayloadPlan>, tritium_spec::BackendError>,
) -> Result<StreamedModelLoadPlan, NnError> {
    if max_device_payload_bytes == 0 {
        return Err(invalid("resident payload ceiling must be positive"));
    }
    inspect_streamed_with(
        path,
        level,
        receiving,
        limits,
        execution,
        |model, prepared| {
            super::resources::describe_streamed_resident(
                model,
                level,
                &prepared.graph,
                &prepared.plan,
                policy,
                planner,
                max_device_payload_bytes,
                decoder,
            )
        },
    )
}

fn inspect_streamed_with(
    path: &Path,
    level: &str,
    receiving: TritPackageReadLimits,
    limits: Qwen35TritReadLimits,
    execution: StreamLimits,
    describe: impl FnOnce(
        &TritModelReader<File>,
        &Prepared,
    ) -> Result<super::ModelResourceAdmission, NnError>,
) -> Result<StreamedModelLoadPlan, NnError> {
    positive(limits)?;
    execution.validate()?;
    let (file, metadata) = open_regular_source(path).map_err(invalid)?;
    let mut model = TritModelReader::open(file, receiving).map_err(invalid)?;
    let prepared = prepare(&mut model, level, metadata.len(), limits, Some(execution))?;
    let covered = describe(&model, &prepared)?;
    model.verify().map_err(invalid)?;
    Ok(StreamedModelLoadPlan {
        source: prepared.plan,
        covered,
    })
}
