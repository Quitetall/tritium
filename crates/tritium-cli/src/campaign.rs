//! Reproducible offline-teacher and packed-SALT campaign orchestration.

use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(any(feature = "cuda", test))]
use std::collections::BTreeSet;

#[cfg(any(feature = "cuda", test))]
use std::fs::OpenOptions;
#[cfg(any(feature = "cuda", test))]
use std::io::Write;
#[cfg(all(feature = "cuda", unix))]
use std::os::unix::fs::MetadataExt as _;
#[cfg(all(feature = "cuda", windows))]
use std::os::windows::fs::MetadataExt as _;

#[cfg(feature = "cuda")]
use anyhow::ensure;
use anyhow::{Context, bail};
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use tritium_format::TeacherCacheHeader;
#[cfg(feature = "cuda")]
use tritium_format::write_teacher_cache_header;
#[cfg(test)]
use tritium_nn::MlpKind;
use tritium_nn::{
    ArchSpec, ModelConfig, ModelRunner, ModelWeights, TeacherCacheWriter, TiedSwiGluTrainingModel,
    hash_teacher_corpus, semantic_training_model_digest,
};

#[cfg(feature = "nccl")]
use std::time::Duration;
#[cfg(feature = "cuda")]
use std::time::Instant;
#[cfg(feature = "nccl")]
use tritium_cuda::NcclProcessGroup;
#[cfg(feature = "cuda")]
use tritium_cuda::train::{
    CheckpointPolicy, DevicePackedSaltWeight, DeviceTape, DeviceTensor, DeviceTrainParam,
    DeviceTrainer, DeviceTrainerWeightStorage, GradientLeafBinding, GradientStreamReport,
    HostOffloadMemoryGeometry, HostOffloadParamMetadata, HostOffloadStats, HostOffloadTrainParam,
    HostOffloadTrainer, PackedSaltComputePolicy, ResidentTrainerStats,
    host_offload_memory_geometry,
};
#[cfg(feature = "cuda")]
use tritium_cuda::{CudaBackend, CudaDeviceIdentity, CudaMemorySnapshot};
#[cfg(feature = "cuda")]
use tritium_nn::{
    TeacherCacheReader, TeacherForcedPerplexity, packed_device_forward,
    parse_training_salt_artifact_metadata, teacher_forced_perplexity_windows,
};
#[cfg(feature = "cuda")]
use tritium_quantize::MeasuredPackage;
#[cfg(feature = "cuda")]
use tritium_train::AdamW;

#[cfg(feature = "cuda")]
use crate::campaign_artifact::{
    ArtifactProvenance, CampaignArtifactSummary, GrowthReceipt, MasterSource,
    ResidentMasterSnapshot, export_campaign_artifact, verify_campaign_artifact,
};
#[cfg(feature = "nccl")]
use crate::campaign_world::{
    DeviceFleet, DistributedConfig, WindowPartition, WorkerRendezvous, supervise_current_exe,
};
#[cfg(feature = "cuda")]
use crate::nvml_probe::{NvmlGpuSnapshot, probe_cuda_device};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "cuda")]
const NAIVE_SALT_PTQ_EVALUATION_KIND: &str = "naive-salt-ptq";
#[cfg(feature = "cuda")]
const NAIVE_SALT_PTQ_METHOD: &str = "whole-row-absmean-f32-scales-exact-v1";

/// Production training-campaign operations.
#[derive(Debug, Subcommand)]
pub(crate) enum CampaignCommand {
    /// Materialize dense fp32 teacher probabilities for fixed corpus windows.
    TeacherCache {
        /// HuggingFace model directory (`config.json` plus safetensors shards).
        #[arg(long)]
        model_dir: PathBuf,
        /// JSON token corpus: either `[token, ...]` or `{ "train_ids": [...] }`.
        #[arg(long)]
        corpus: PathBuf,
        /// Tokens per independent teacher window.
        #[arg(long)]
        seq_len: usize,
        /// Atomic output path for the teacher cache.
        #[arg(long)]
        output: PathBuf,
    },
    /// Run or resume an offline-teacher packed-SALT campaign on CUDA.
    Run {
        /// JSON campaign configuration. Required keys: model_dir, corpus,
        /// evaluation_corpus, teacher_cache, checkpoint_dir, report, artifact,
        /// seq_len, steps, and checkpoint_every. Optional keys: growth,
        /// salt_planes, cuda_device, distributed, state_policy, packed_compute,
        /// timing_warmup_steps, checkpoint_shards, and adam. Relative paths
        /// resolve beside this file.
        #[arg(long)]
        config: PathBuf,
    },
}

pub(crate) fn run(command: CampaignCommand) -> anyhow::Result<()> {
    match command {
        CampaignCommand::TeacherCache {
            model_dir,
            corpus,
            seq_len,
            output,
        } => teacher_cache(&model_dir, &corpus, seq_len, &output),
        CampaignCommand::Run { config } => run_campaign(&config),
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum CorpusDocument {
    Tokens(Vec<u32>),
    Training { train_ids: Vec<u32> },
}

fn read_corpus(path: &Path) -> anyhow::Result<(Vec<u32>, [u8; 32])> {
    let bytes = std::fs::read(path).with_context(|| format!("read corpus {}", path.display()))?;
    let document: CorpusDocument = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse token corpus {}", path.display()))?;
    let tokens = match document {
        CorpusDocument::Tokens(tokens) => tokens,
        CorpusDocument::Training { train_ids } => train_ids,
    };
    Ok((tokens, blake3_digest(&bytes)))
}

fn validate_windows(
    tokens: &[u32],
    seq_len: usize,
    vocab: usize,
    max_context: usize,
) -> anyhow::Result<u64> {
    if seq_len == 0 {
        bail!("sequence length must be non-zero");
    }
    if seq_len > max_context {
        bail!("sequence length {seq_len} exceeds model context length {max_context}");
    }
    if tokens.is_empty() {
        bail!("training corpus is empty");
    }
    if !tokens.len().is_multiple_of(seq_len) {
        bail!(
            "corpus has {} tokens, not an exact multiple of sequence length {seq_len}",
            tokens.len()
        );
    }
    if let Some((position, token)) = tokens
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| usize::try_from(*token).map_or(true, |token| token >= vocab))
    {
        bail!("corpus token {token} at position {position} is outside vocabulary 0..{vocab}");
    }
    u64::try_from(tokens.len() / seq_len).context("window count exceeds u64")
}

#[cfg(any(feature = "cuda", test))]
fn validate_held_out_corpus(
    training_tokens: &[u32],
    training_digest: [u8; 32],
    evaluation_tokens: &[u32],
    evaluation_digest: [u8; 32],
    seq_len: usize,
) -> anyhow::Result<()> {
    if seq_len == 0
        || training_tokens.is_empty()
        || evaluation_tokens.is_empty()
        || !training_tokens.len().is_multiple_of(seq_len)
        || !evaluation_tokens.len().is_multiple_of(seq_len)
    {
        bail!("held-out overlap validation requires non-empty exact windows");
    }
    if training_digest == evaluation_digest {
        bail!(
            "training and evaluation corpora have identical semantic token content; held-out evidence would be contaminated"
        );
    }
    let training_windows: BTreeSet<_> = training_tokens
        .chunks_exact(seq_len)
        .map(hash_token_window)
        .collect();
    if let Some(index) = evaluation_tokens
        .chunks_exact(seq_len)
        .map(hash_token_window)
        .position(|digest| training_windows.contains(&digest))
    {
        bail!(
            "evaluation window {index} is also present in the training corpus; held-out evidence would be contaminated"
        );
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn hash_token_window(tokens: &[u32]) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium-held-out-token-window-v1");
    for token in tokens {
        hash.update(&token.to_le_bytes());
    }
    *hash.finalize().as_bytes()
}

fn teacher_cache(
    model_dir: &Path,
    corpus_path: &Path,
    seq_len: usize,
    output: &Path,
) -> anyhow::Result<()> {
    validate_teacher_cache_paths(model_dir, corpus_path, output)?;
    let (config, spec, weights) = ModelWeights::load_hf(model_dir)
        .with_context(|| format!("load HuggingFace model {}", model_dir.display()))?;
    let training_model = TiedSwiGluTrainingModel::extract(&config, &spec, &weights)
        .context("validate tied-SwiGLU training adapter")?;
    let model_hash = semantic_model_digest(&config, &spec, &training_model);
    let model_config_file_hash = hash_file(&model_dir.join("config.json"))?;
    let vocab = training_model.architecture().vocab;
    let max_context = training_model.architecture().n_ctx;
    drop(training_model);

    let (tokens, corpus_file_hash) = read_corpus(corpus_path)?;
    let windows = validate_windows(&tokens, seq_len, vocab, max_context)?;
    let seq_len_u32 = u32::try_from(seq_len).context("sequence length exceeds u32")?;
    let vocab_u32 = u32::try_from(vocab).context("vocabulary exceeds u32")?;
    let corpus_hash = hash_teacher_corpus(&tokens, seq_len_u32);
    let header = TeacherCacheHeader {
        seq_len: seq_len_u32,
        vocab: vocab_u32,
        windows,
        model_hash,
        corpus_hash,
    };
    let window_elements = header
        .window_elements()
        .context("teacher-cache window shape overflows usize")?;

    let mut runner =
        ModelRunner::from_weights(config, weights, Box::new(tritium_cpu::CpuBackend::new()));
    publish_teacher_cache(output, header, |writer| {
        let mut probabilities = Vec::with_capacity(window_elements);
        for window in tokens.chunks_exact(seq_len) {
            runner.reset();
            probabilities.clear();
            for (position, &token) in window.iter().enumerate() {
                let mut logits = runner
                    .forward(&[token], &[position])
                    .context("CPU teacher forward")?;
                row_softmax_in_place(&mut logits)?;
                probabilities.extend_from_slice(&logits);
            }
            writer
                .write_window(&probabilities)
                .context("write teacher probability window")?;
        }
        Ok(())
    })?;
    let teacher_cache_hash = hash_file(output)?;

    let report = TeacherCacheReport {
        output: output.display().to_string(),
        windows,
        seq_len,
        vocab,
        model_digest: hex_digest(model_hash),
        model_config_file_digest: hex_digest(model_config_file_hash),
        corpus_digest: hex_digest(corpus_hash),
        corpus_file_digest: hex_digest(corpus_file_hash),
        teacher_cache_digest: hex_digest(teacher_cache_hash),
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn row_softmax_in_place(row: &mut [f32]) -> anyhow::Result<()> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        bail!("teacher logits contain no finite maximum");
    }
    let mut sum = 0.0f32;
    for value in row.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        bail!("teacher softmax normalization is not finite and positive");
    }
    for value in row {
        *value /= sum;
    }
    Ok(())
}

fn publish_teacher_cache(
    output: &Path,
    header: TeacherCacheHeader,
    write_windows: impl FnOnce(&mut TeacherCacheWriter) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let publish = AtomicPath::new(output)?;
    let mut writer = TeacherCacheWriter::create(publish.temporary(), header)
        .context("create temporary teacher cache")?;
    write_windows(&mut writer)?;
    writer.finish().context("finish and fsync teacher cache")?;
    publish.commit()
}

#[derive(Debug, Serialize)]
struct TeacherCacheReport {
    output: String,
    windows: u64,
    seq_len: usize,
    vocab: usize,
    model_digest: String,
    model_config_file_digest: String,
    corpus_digest: String,
    corpus_file_digest: String,
    teacher_cache_digest: String,
}

/// Hash every semantic input consumed by the tied-SwiGLU graph. The cache header
/// has one 32-byte model field, so config/spec, names, shapes, masters, and all
/// preserved norm/bias vectors are domain-separated into this digest rather
/// than hashing only the 2D master values.
fn semantic_model_digest(
    config: &ModelConfig,
    spec: &ArchSpec,
    model: &TiedSwiGluTrainingModel,
) -> [u8; 32] {
    semantic_training_model_digest(config, spec, model)
}

#[cfg(feature = "cuda")]
fn training_parameter_count(model: &TiedSwiGluTrainingModel) -> anyhow::Result<u64> {
    let matrix_count = model
        .parameters()
        .iter()
        .try_fold(0_u64, |total, parameter| {
            let elements = u64::try_from(parameter.elements())
                .with_context(|| format!("{} element count exceeds u64", parameter.name))?;
            total
                .checked_add(elements)
                .with_context(|| format!("parameter count overflows at {}", parameter.name))
        })?;
    let with_norms = model
        .architecture()
        .attn_norms
        .iter()
        .chain(&model.architecture().ffn_norms)
        .chain(core::iter::once(&model.architecture().output_norm))
        .try_fold(matrix_count, |total, norm| {
            let elements = u64::try_from(norm.len()).context("norm element count exceeds u64")?;
            total
                .checked_add(elements)
                .context("parameter count overflows")
        })?;
    model
        .architecture()
        .attention_constants
        .iter()
        .flat_map(|constants| {
            [
                &constants.q_bias,
                &constants.k_bias,
                &constants.v_bias,
                &constants.q_norm,
                &constants.k_norm,
            ]
        })
        .try_fold(with_norms, |total, values| {
            let elements =
                u64::try_from(values.len()).context("attention constant count exceeds u64")?;
            total
                .checked_add(elements)
                .context("parameter count overflows")
        })
}

#[cfg(test)]
fn semantic_model_digest_parts(
    config: &ModelConfig,
    spec: &ArchSpec,
    architecture: &tritium_nn::TiedSwiGluTrainingArchitecture,
    parameters: &[tritium_nn::TrainingParameter],
) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium-tied-swiglu-training-model-v1");
    hash_bytes(&mut hash, config.arch.as_bytes());
    for value in [
        config.n_layers,
        config.n_embd,
        config.n_head,
        config.n_head_kv,
        config.head_dim,
        config.n_ff,
        config.n_ctx,
    ] {
        hash.update(&(value as u64).to_le_bytes());
    }
    hash.update(&config.rope_theta.to_bits().to_le_bytes());
    hash.update(&config.rms_eps.to_bits().to_le_bytes());
    hash.update(&[match spec.mlp {
        MlpKind::Relu2 => 0,
        MlpKind::SwiGlu => 1,
    }]);
    hash.update(&[
        u8::from(spec.attn_sub_norm),
        u8::from(spec.ffn_sub_norm),
        u8::from(spec.qk_norm),
        u8::from(spec.qkv_bias),
        u8::from(spec.tied_embeddings),
    ]);
    hash.update(&(parameters.len() as u64).to_le_bytes());
    for parameter in parameters {
        hash_bytes(&mut hash, parameter.name.as_bytes());
        hash.update(&(parameter.rows as u64).to_le_bytes());
        hash.update(&(parameter.cols as u64).to_le_bytes());
        hash_f32s(&mut hash, &parameter.master);
    }
    hash.update(&(architecture.attn_norms.len() as u64).to_le_bytes());
    for norm in &architecture.attn_norms {
        hash_f32s(&mut hash, norm);
    }
    hash.update(&(architecture.ffn_norms.len() as u64).to_le_bytes());
    for norm in &architecture.ffn_norms {
        hash_f32s(&mut hash, norm);
    }
    if architecture.attention_constants.iter().any(|constants| {
        !constants.q_bias.is_empty()
            || !constants.k_bias.is_empty()
            || !constants.v_bias.is_empty()
            || !constants.q_norm.is_empty()
            || !constants.k_norm.is_empty()
    }) {
        hash.update(b"tritium-standard-qwen-attention-constants-v1");
        hash.update(&(architecture.attention_constants.len() as u64).to_le_bytes());
        for constants in &architecture.attention_constants {
            hash_f32s(&mut hash, &constants.q_bias);
            hash_f32s(&mut hash, &constants.k_bias);
            hash_f32s(&mut hash, &constants.v_bias);
            hash_f32s(&mut hash, &constants.q_norm);
            hash_f32s(&mut hash, &constants.k_norm);
        }
    }
    hash_f32s(&mut hash, &architecture.output_norm);
    *hash.finalize().as_bytes()
}

#[cfg(any(feature = "cuda", test))]
fn hash_bytes(hash: &mut blake3::Hasher, bytes: &[u8]) {
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

#[cfg(any(feature = "cuda", test))]
fn hash_f32s(hash: &mut blake3::Hasher, values: &[f32]) {
    hash.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hash.update(&value.to_bits().to_le_bytes());
    }
}

fn blake3_digest(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

fn hash_file(path: &Path) -> anyhow::Result<[u8; 32]> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = blake3::Hasher::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(*hash.finalize().as_bytes())
}

fn hex_digest(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for byte in digest {
        text.push(char::from(HEX[usize::from(byte >> 4)]));
        text.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    text
}

struct AtomicPath {
    destination: PathBuf,
    temporary: PathBuf,
    committed: bool,
}

impl AtomicPath {
    fn new(destination: &Path) -> anyhow::Result<Self> {
        let parent = output_parent(destination);
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
        let file_name = destination
            .file_name()
            .context("atomic output path has no file name")?;
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut temp_name = OsString::from(file_name);
        temp_name.push(format!(".tmp-{}-{sequence}", std::process::id()));
        Ok(Self {
            destination: destination.to_owned(),
            temporary: parent.join(temp_name),
            committed: false,
        })
    }

    fn temporary(&self) -> &Path {
        &self.temporary
    }

    fn commit(mut self) -> anyhow::Result<()> {
        std::fs::rename(&self.temporary, &self.destination).with_context(|| {
            format!(
                "atomically publish {} as {}",
                self.temporary.display(),
                self.destination.display()
            )
        })?;
        // The rename already published the destination, so Drop must not try to
        // clean up the now-nonexistent temporary path if directory sync fails.
        self.committed = true;
        #[cfg(unix)]
        {
            let parent = output_parent(&self.destination);
            File::open(parent)
                .and_then(|directory| directory.sync_all())
                .with_context(|| format!("fsync output directory {}", parent.display()))?;
        }
        Ok(())
    }
}

fn output_parent(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

fn path_identity(path: &Path) -> anyhow::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .context("resolve current directory for path validation")?
            .join(path)
    };
    let mut prefix = PathBuf::new();
    let mut existing_prefix = None;
    let mut suffix = Vec::new();
    let mut found_missing_component = false;
    for component in absolute.components() {
        if found_missing_component {
            suffix.push(component.as_os_str().to_owned());
            continue;
        }
        prefix.push(component.as_os_str());
        if prefix
            .try_exists()
            .with_context(|| format!("inspect path {}", prefix.display()))?
        {
            existing_prefix = Some(prefix.clone());
        } else {
            found_missing_component = true;
            suffix.push(component.as_os_str().to_owned());
        }
    }

    // Canonicalize before reducing any unresolved `..` suffix. This preserves
    // filesystem semantics for paths such as `symlink/../output`, whose parent
    // is the symlink target's parent rather than the symlink's lexical parent.
    let existing_prefix = existing_prefix
        .with_context(|| format!("path {} has no existing ancestor", path.display()))?;
    let mut identity = std::fs::canonicalize(&existing_prefix)
        .with_context(|| format!("canonicalize path {}", existing_prefix.display()))?;
    for component in suffix {
        if component == "." {
            continue;
        }
        if component == ".." {
            identity.pop();
        } else {
            identity.push(component);
        }
    }
    Ok(identity)
}

fn validate_existing_output_type(path: &Path, label: &str) -> anyhow::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(()),
        Ok(_) => bail!(
            "{label} {} exists but is not a regular file",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("inspect {label} {}", path.display())),
    }
}

fn validate_teacher_cache_paths(
    model_dir: &Path,
    corpus: &Path,
    output: &Path,
) -> anyhow::Result<()> {
    validate_existing_output_type(output, "teacher-cache output")?;
    let output_identity = path_identity(output)?;
    let corpus_identity = path_identity(corpus)?;
    let model_identity = path_identity(model_dir)?;
    if output_identity == corpus_identity {
        bail!(
            "teacher-cache output {} aliases its input corpus",
            output.display()
        );
    }
    if output_identity.starts_with(model_identity) {
        bail!(
            "teacher-cache output {} must be outside model directory {}",
            output.display(),
            model_dir.display()
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_campaign_paths(config_path: &Path, config: &CampaignConfig) -> anyhow::Result<()> {
    validate_existing_output_type(&config.report, "campaign report")?;
    validate_existing_output_type(&config.artifact, "campaign artifact")?;
    let artifact_parent = output_parent(&config.artifact);
    let artifact_parent_metadata =
        std::fs::symlink_metadata(artifact_parent).with_context(|| {
            format!(
                "inspect campaign artifact parent {}",
                artifact_parent.display()
            )
        })?;
    if !artifact_parent_metadata.file_type().is_dir() {
        bail!(
            "campaign artifact parent {} must be an existing real directory",
            artifact_parent.display()
        );
    }
    let report = path_identity(&config.report)?;
    let artifact = path_identity(&config.artifact)?;
    let checkpoint = path_identity(&config.checkpoint_dir)?;
    let model_dir = path_identity(&config.model_dir)?;
    let protected_files = [
        ("campaign config", path_identity(config_path)?),
        ("training corpus", path_identity(&config.corpus)?),
        (
            "evaluation corpus",
            path_identity(&config.evaluation_corpus)?,
        ),
        ("teacher cache", path_identity(&config.teacher_cache)?),
        (
            "model config",
            path_identity(&config.model_dir.join("config.json"))?,
        ),
    ];

    for (left_label, left, right_label, right) in [
        (
            "campaign report",
            &report,
            "checkpoint directory",
            &checkpoint,
        ),
        (
            "campaign artifact",
            &artifact,
            "checkpoint directory",
            &checkpoint,
        ),
        ("campaign report", &report, "campaign artifact", &artifact),
    ] {
        if left.starts_with(right) || right.starts_with(left) {
            bail!("{left_label} and {right_label} must not alias or contain one another");
        }
    }
    for (label, output) in [
        ("campaign report", &report),
        ("campaign artifact", &artifact),
    ] {
        if output.starts_with(&model_dir) {
            bail!(
                "{label} must be outside model directory {}",
                config.model_dir.display()
            );
        }
    }
    if checkpoint.starts_with(&model_dir) || model_dir.starts_with(&checkpoint) {
        bail!(
            "checkpoint directory {} must not overlap model directory {}",
            config.checkpoint_dir.display(),
            config.model_dir.display()
        );
    }
    for (label, protected) in protected_files {
        for (output_label, output) in [
            ("campaign report", &report),
            ("campaign artifact", &artifact),
        ] {
            if *output == protected {
                bail!("{output_label} aliases the {label}");
            }
        }
        if protected.starts_with(&checkpoint) {
            bail!(
                "checkpoint directory {} contains the {label}; choose a dedicated checkpoint directory",
                config.checkpoint_dir.display()
            );
        }
    }
    if path_identity(&config.corpus)? == path_identity(&config.evaluation_corpus)? {
        bail!("training and evaluation corpora must be distinct files");
    }
    Ok(())
}

impl Drop for AtomicPath {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.temporary);
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CampaignLock {
    _file: File,
}

#[cfg(feature = "cuda")]
impl CampaignLock {
    fn acquire(path: &Path) -> anyhow::Result<Self> {
        let file = open_campaign_lock(path)?;
        file.try_lock().with_context(|| {
            format!(
                "acquire exclusive campaign lock {}; another campaign process holds it",
                path.display()
            )
        })?;
        Ok(Self { _file: file })
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
struct CampaignLocks {
    _locks: Vec<CampaignLock>,
}

#[cfg(feature = "cuda")]
impl CampaignLocks {
    fn acquire(checkpoint_dir: &Path, report: &Path, artifact: &Path) -> anyhow::Result<Self> {
        let paths = campaign_lock_paths(checkpoint_dir, report, artifact)?;
        let report_identity = path_identity(report)?;
        let artifact_identity = path_identity(artifact)?;
        for path in &paths {
            if path == &report_identity || path == &artifact_identity {
                bail!(
                    "campaign lock sidecar {} aliases a campaign output",
                    path.display()
                );
            }
        }
        let locks = paths
            .iter()
            .map(|path| CampaignLock::acquire(path))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self { _locks: locks })
    }
}

#[cfg(feature = "cuda")]
fn campaign_lock_paths(
    checkpoint_dir: &Path,
    report: &Path,
    artifact: &Path,
) -> anyhow::Result<Vec<PathBuf>> {
    std::fs::create_dir_all(checkpoint_dir)
        .with_context(|| format!("create checkpoint directory {}", checkpoint_dir.display()))?;
    let checkpoint_dir = std::fs::canonicalize(checkpoint_dir).with_context(|| {
        format!(
            "canonicalize checkpoint directory {}",
            checkpoint_dir.display()
        )
    })?;
    let mut paths = vec![
        checkpoint_dir.join("campaign.lock"),
        campaign_output_lock_path(report)?,
        campaign_output_lock_path(artifact)?,
    ];
    paths.sort();
    paths.dedup();
    Ok(paths)
}

#[cfg(feature = "cuda")]
fn campaign_output_lock_path(output: &Path) -> anyhow::Result<PathBuf> {
    let parent = output_parent(output);
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create campaign output directory {}", parent.display()))?;
    let parent = std::fs::canonicalize(parent).with_context(|| {
        format!(
            "canonicalize campaign output directory {}",
            parent.display()
        )
    })?;
    let mut name = output
        .file_name()
        .context("campaign output path has no file name")?
        .to_os_string();
    name.push(".campaign.lock");
    Ok(parent.join(name))
}

#[cfg(feature = "cuda")]
fn open_campaign_lock(path: &Path) -> anyhow::Result<File> {
    const MAX_OPEN_ATTEMPTS: usize = 4;

    for _ in 0..MAX_OPEN_ATTEMPTS {
        match std::fs::symlink_metadata(path) {
            Ok(before) => {
                validate_campaign_lock_metadata(path, &before)?;
                let file = match OpenOptions::new().read(true).write(true).open(path) {
                    Ok(file) => file,
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(error) => {
                        return Err(error)
                            .with_context(|| format!("open campaign lock {}", path.display()));
                    }
                };
                let opened = file
                    .metadata()
                    .with_context(|| format!("inspect opened campaign lock {}", path.display()))?;
                validate_campaign_lock_metadata(path, &opened)?;
                if !same_campaign_lock_file(&before, &opened) {
                    bail!("campaign lock {} changed while opening", path.display());
                }
                validate_open_campaign_lock_path(path, &opened)?;
                return Ok(file);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match OpenOptions::new()
                    .create_new(true)
                    .read(true)
                    .write(true)
                    .open(path)
                {
                    Ok(file) => {
                        let opened = file.metadata().with_context(|| {
                            format!("inspect created campaign lock {}", path.display())
                        })?;
                        validate_campaign_lock_metadata(path, &opened)?;
                        validate_open_campaign_lock_path(path, &opened)?;
                        return Ok(file);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(error) => {
                        return Err(error)
                            .with_context(|| format!("create campaign lock {}", path.display()));
                    }
                }
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspect campaign lock {}", path.display()));
            }
        }
    }
    bail!(
        "campaign lock {} changed repeatedly while opening",
        path.display()
    )
}

#[cfg(feature = "cuda")]
fn validate_open_campaign_lock_path(path: &Path, opened: &std::fs::Metadata) -> anyhow::Result<()> {
    let current = std::fs::symlink_metadata(path)
        .with_context(|| format!("reinspect campaign lock {}", path.display()))?;
    validate_campaign_lock_metadata(path, &current)?;
    if !same_campaign_lock_file(opened, &current) {
        bail!("campaign lock {} changed while opening", path.display());
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_campaign_lock_metadata(
    path: &Path,
    metadata: &std::fs::Metadata,
) -> anyhow::Result<()> {
    if !metadata.file_type().is_file() {
        bail!(
            "campaign lock {} must be a regular file, not a symlink or special file",
            path.display()
        );
    }
    validate_campaign_lock_link_count(path, metadata)
}

#[cfg(all(feature = "cuda", unix))]
fn validate_campaign_lock_link_count(
    path: &Path,
    metadata: &std::fs::Metadata,
) -> anyhow::Result<()> {
    if metadata.nlink() != 1 {
        bail!(
            "campaign lock {} must not be a hard-linked inode",
            path.display()
        );
    }
    Ok(())
}

#[cfg(all(feature = "cuda", windows))]
fn validate_campaign_lock_link_count(
    path: &Path,
    metadata: &std::fs::Metadata,
) -> anyhow::Result<()> {
    if metadata.number_of_links() != Some(1) {
        bail!(
            "campaign lock {} must not be a hard-linked inode",
            path.display()
        );
    }
    Ok(())
}

#[cfg(all(feature = "cuda", not(any(unix, windows))))]
fn validate_campaign_lock_link_count(
    path: &Path,
    _metadata: &std::fs::Metadata,
) -> anyhow::Result<()> {
    bail!(
        "campaign locking on this platform cannot verify hard-link safety for {}",
        path.display()
    )
}

#[cfg(all(feature = "cuda", unix))]
fn same_campaign_lock_file(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(all(feature = "cuda", windows))]
fn same_campaign_lock_file(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    left.volume_serial_number() == right.volume_serial_number()
        && left.file_index() == right.file_index()
        && left.volume_serial_number().is_some()
        && left.file_index().is_some()
}

#[cfg(all(feature = "cuda", not(any(unix, windows))))]
fn same_campaign_lock_file(_left: &std::fs::Metadata, _right: &std::fs::Metadata) -> bool {
    false
}

#[cfg(any(feature = "cuda", test))]
fn atomic_write(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let publish = AtomicPath::new(path)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(publish.temporary())
        .with_context(|| format!("create temporary output {}", publish.temporary().display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write temporary output {}", publish.temporary().display()))?;
    file.sync_all()
        .with_context(|| format!("fsync temporary output {}", publish.temporary().display()))?;
    drop(file);
    publish.commit()
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CampaignAdam {
    #[serde(default = "default_lr")]
    lr: f32,
    #[serde(default = "default_beta1")]
    beta1: f32,
    #[serde(default = "default_beta2")]
    beta2: f32,
    #[serde(default = "default_eps")]
    eps: f32,
    #[serde(default = "default_weight_decay")]
    weight_decay: f32,
}

#[cfg(any(feature = "cuda", test))]
impl Default for CampaignAdam {
    fn default() -> Self {
        Self {
            lr: default_lr(),
            beta1: default_beta1(),
            beta2: default_beta2(),
            eps: default_eps(),
            weight_decay: default_weight_decay(),
        }
    }
}

#[cfg(any(feature = "cuda", test))]
const fn default_lr() -> f32 {
    2e-3
}
#[cfg(any(feature = "cuda", test))]
const fn default_beta1() -> f32 {
    0.9
}
#[cfg(any(feature = "cuda", test))]
const fn default_beta2() -> f32 {
    0.999
}
#[cfg(any(feature = "cuda", test))]
const fn default_eps() -> f32 {
    1e-8
}
#[cfg(any(feature = "cuda", test))]
const fn default_weight_decay() -> f32 {
    0.01
}
#[cfg(feature = "cuda")]
const fn default_planes() -> usize {
    2
}
#[cfg(feature = "cuda")]
const fn default_checkpoint_shards() -> usize {
    1
}
#[cfg(feature = "cuda")]
const fn default_timing_warmup_steps() -> usize {
    1
}
#[cfg(feature = "cuda")]
const fn default_worker_timeout_seconds() -> u64 {
    86_400
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CampaignStatePolicy {
    #[default]
    Resident,
    HostOffload,
}

#[cfg(any(feature = "cuda", test))]
impl CampaignStatePolicy {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::HostOffload => "host-offload",
        }
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CampaignPackedComputePolicy {
    #[default]
    Exact,
    Fast,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CampaignDistributed {
    devices: Vec<usize>,
    #[serde(default = "default_worker_timeout_seconds")]
    worker_timeout_seconds: u64,
}

#[cfg(any(feature = "cuda", test))]
impl CampaignPackedComputePolicy {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Fast => "fast",
        }
    }
}

#[cfg(feature = "cuda")]
impl From<CampaignPackedComputePolicy> for PackedSaltComputePolicy {
    fn from(policy: CampaignPackedComputePolicy) -> Self {
        match policy {
            CampaignPackedComputePolicy::Exact => Self::Exact,
            CampaignPackedComputePolicy::Fast => Self::Fast,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CampaignConfig {
    model_dir: PathBuf,
    corpus: PathBuf,
    evaluation_corpus: PathBuf,
    teacher_cache: PathBuf,
    checkpoint_dir: PathBuf,
    report: PathBuf,
    artifact: PathBuf,
    seq_len: usize,
    steps: u64,
    #[serde(default = "default_planes")]
    salt_planes: usize,
    #[serde(default)]
    cuda_device: Option<usize>,
    #[serde(default)]
    distributed: Option<CampaignDistributed>,
    #[serde(default)]
    state_policy: CampaignStatePolicy,
    #[serde(default)]
    packed_compute: CampaignPackedComputePolicy,
    checkpoint_every: u64,
    #[serde(default = "default_timing_warmup_steps")]
    timing_warmup_steps: usize,
    #[serde(default = "default_checkpoint_shards")]
    checkpoint_shards: usize,
    #[serde(default)]
    adam: CampaignAdam,
    #[serde(default)]
    growth: Option<CampaignGrowth>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CampaignGrowth {
    intermediate_size: usize,
    seed: u64,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CampaignGrowthReceipt {
    algorithm: String,
    old_width: u64,
    new_width: u64,
    seed: u64,
    source_indices: Vec<u64>,
    replication_counts: Vec<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    split_denominator_log2: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    split_numerators: Option<Vec<u32>>,
}

#[cfg(feature = "cuda")]
fn campaign_growth_receipt(
    plan: &tritium_train::Net2WiderPlan,
    seed: u64,
) -> anyhow::Result<CampaignGrowthReceipt> {
    Ok(CampaignGrowthReceipt {
        algorithm: plan.algorithm().to_owned(),
        old_width: u64::try_from(plan.replication_counts().len())
            .context("growth old width exceeds u64")?,
        new_width: u64::try_from(plan.source_indices().len())
            .context("growth new width exceeds u64")?,
        seed,
        source_indices: plan
            .source_indices()
            .iter()
            .map(|&value| u64::try_from(value).context("growth source index exceeds u64"))
            .collect::<anyhow::Result<Vec<_>>>()?,
        replication_counts: plan
            .replication_counts()
            .iter()
            .map(|&value| u64::try_from(value).context("growth replication count exceeds u64"))
            .collect::<anyhow::Result<Vec<_>>>()?,
        split_denominator_log2: plan.split_denominator_log2(),
        split_numerators: plan.split_numerators().map(<[u32]>::to_vec),
    })
}

#[cfg(feature = "cuda")]
impl CampaignConfig {
    fn resolve_paths(&mut self, config_path: &Path) {
        let base = config_path.parent().unwrap_or_else(|| Path::new("."));
        for path in [
            &mut self.model_dir,
            &mut self.corpus,
            &mut self.evaluation_corpus,
            &mut self.teacher_cache,
            &mut self.checkpoint_dir,
            &mut self.report,
            &mut self.artifact,
        ] {
            if path.is_relative() {
                *path = base.join(&*path);
            }
        }
    }
}

#[cfg(feature = "cuda")]
enum CampaignWorld {
    Single,
    #[cfg(feature = "nccl")]
    Distributed {
        rendezvous: Box<WorkerRendezvous>,
        process_group: Box<NcclProcessGroup>,
    },
}

#[cfg(feature = "cuda")]
impl CampaignWorld {
    fn single() -> Self {
        Self::Single
    }

    #[cfg(feature = "nccl")]
    fn distributed(rendezvous: WorkerRendezvous, backend: &CudaBackend) -> anyhow::Result<Self> {
        let process_group = rendezvous
            .join(backend)
            .context("join supervised NCCL campaign world")?;
        Ok(Self::Distributed {
            rendezvous: Box::new(rendezvous),
            process_group: Box::new(process_group),
        })
    }

    fn rank(&self) -> usize {
        match self {
            Self::Single => 0,
            #[cfg(feature = "nccl")]
            Self::Distributed { rendezvous, .. } => rendezvous.slot().rank().get(),
        }
    }

    fn world_size(&self) -> usize {
        match self {
            Self::Single => 1,
            #[cfg(feature = "nccl")]
            Self::Distributed { rendezvous, .. } => rendezvous.world_size().get(),
        }
    }

    fn is_owner(&self) -> bool {
        match self {
            Self::Single => true,
            #[cfg(feature = "nccl")]
            Self::Distributed { rendezvous, .. } => rendezvous.owns_campaign_lock(),
        }
    }

    fn window_index(&self, step: u64, windows: u64) -> anyhow::Result<u64> {
        match self {
            Self::Single => Ok((step - 1) % windows),
            #[cfg(feature = "nccl")]
            Self::Distributed { rendezvous, .. } => {
                let partition = WindowPartition::new(
                    usize::try_from(windows).context("window count exceeds usize")?,
                    rendezvous.world_size().get(),
                )?;
                u64::try_from(partition.window_for_step(rendezvous.slot().rank(), step)?)
                    .context("partitioned window index exceeds u64")
            }
        }
    }

    fn barrier(&self) -> anyhow::Result<()> {
        match self {
            Self::Single => Ok(()),
            #[cfg(feature = "nccl")]
            Self::Distributed { process_group, .. } => {
                process_group.barrier().context("NCCL campaign barrier")
            }
        }
    }

    fn verify_contract_digest(&self, digest: [u8; 32]) -> anyhow::Result<()> {
        match self {
            Self::Single => {
                let _ = digest;
                Ok(())
            }
            #[cfg(feature = "nccl")]
            Self::Distributed { process_group, .. } => {
                let words = digest
                    .chunks_exact(8)
                    .map(|chunk| u64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
                    .collect::<Vec<_>>();
                process_group
                    .verify_u64_consensus(&words)
                    .context("verify immutable campaign plan across ranks")
            }
        }
    }

    fn verify_completed_step(&self, step: u64) -> anyhow::Result<()> {
        match self {
            Self::Single => {
                let _ = step;
                Ok(())
            }
            #[cfg(feature = "nccl")]
            Self::Distributed { process_group, .. } => process_group
                .verify_u64_consensus(&[step])
                .context("verify checkpoint completed step across ranks"),
        }
    }

    fn gather_hardware(
        &self,
        local: &CampaignHardwareIdentity,
    ) -> anyhow::Result<Vec<CampaignHardwareIdentity>> {
        match self {
            Self::Single => Ok(vec![local.clone()]),
            #[cfg(feature = "nccl")]
            Self::Distributed { process_group, .. } => {
                const MAX_BYTES: usize = 1024;
                const WORDS: usize = 1 + MAX_BYTES / 8;
                let encoded = serde_json::to_vec(local)?;
                if encoded.len() > MAX_BYTES {
                    bail!("serialized campaign hardware identity exceeds {MAX_BYTES} bytes");
                }
                let mut record = vec![0_u64; WORDS];
                record[0] =
                    u64::try_from(encoded.len()).context("hardware JSON length exceeds u64")?;
                for (index, chunk) in encoded.chunks(8).enumerate() {
                    let mut word = [0_u8; 8];
                    word[..chunk.len()].copy_from_slice(chunk);
                    record[index + 1] = u64::from_le_bytes(word);
                }
                let gathered = process_group
                    .all_gather_u64(&record)
                    .context("gather campaign hardware identities")?;
                if gathered.len() != self.world_size() * WORDS {
                    bail!("NCCL hardware gather returned an invalid record count");
                }
                let fleet: Vec<CampaignHardwareIdentity> = gathered
                    .chunks_exact(WORDS)
                    .enumerate()
                    .map(|(rank, record)| {
                        let len = usize::try_from(record[0])
                            .context("gathered hardware JSON length exceeds usize")?;
                        if len > MAX_BYTES {
                            bail!("rank {rank} reported oversized hardware identity");
                        }
                        let mut bytes = Vec::with_capacity(MAX_BYTES);
                        for word in &record[1..] {
                            bytes.extend_from_slice(&word.to_le_bytes());
                        }
                        bytes.truncate(len);
                        serde_json::from_slice(&bytes)
                            .with_context(|| format!("decode rank {rank} hardware identity"))
                    })
                    .collect::<anyhow::Result<_>>()?;
                let unique_pci: BTreeSet<_> = fleet
                    .iter()
                    .map(|hardware| hardware.pci_bus_id.as_str())
                    .collect();
                if unique_pci.len() != fleet.len() {
                    bail!("distributed ranks resolved to duplicate physical PCI devices");
                }
                Ok(fleet)
            }
        }
    }

    fn gather_step_evidence(
        &self,
        local: RankStepEvidence,
    ) -> anyhow::Result<Vec<RankStepEvidence>> {
        match self {
            Self::Single => Ok(vec![local]),
            #[cfg(feature = "nccl")]
            Self::Distributed { process_group, .. } => {
                let snapshot = local.cuda_memory;
                let record = [
                    u64::try_from(local.rank).context("rank exceeds u64")?,
                    local.window_index,
                    local.training_elapsed_ms.to_bits(),
                    local.nvml_used_memory_bytes,
                    snapshot.device_free_bytes,
                    snapshot.device_total_bytes,
                    snapshot.device_used_sample_bytes,
                    snapshot.pool_used_current_bytes,
                    snapshot.pool_used_high_water_bytes,
                    snapshot.pool_reserved_current_bytes,
                    snapshot.pool_reserved_high_water_bytes,
                ];
                let gathered = process_group
                    .all_gather_u64(&record)
                    .context("gather per-rank campaign step evidence")?;
                if gathered.len() != self.world_size() * record.len() {
                    bail!("NCCL step-evidence gather returned an invalid record count");
                }
                let evidence: Vec<RankStepEvidence> = gathered
                    .chunks_exact(record.len())
                    .map(|rank| {
                        Ok(RankStepEvidence {
                            rank: usize::try_from(rank[0])
                                .context("gathered rank exceeds usize")?,
                            window_index: rank[1],
                            training_elapsed_ms: f64::from_bits(rank[2]),
                            nvml_used_memory_bytes: rank[3],
                            cuda_memory: CampaignCudaMemorySnapshot {
                                device_free_bytes: rank[4],
                                device_total_bytes: rank[5],
                                device_used_sample_bytes: rank[6],
                                pool_used_current_bytes: rank[7],
                                pool_used_high_water_bytes: rank[8],
                                pool_reserved_current_bytes: rank[9],
                                pool_reserved_high_water_bytes: rank[10],
                            },
                        })
                    })
                    .collect::<anyhow::Result<_>>()?;
                if evidence
                    .iter()
                    .enumerate()
                    .any(|(rank, evidence)| evidence.rank != rank)
                {
                    bail!("NCCL step evidence rank ordering is inconsistent");
                }
                Ok(evidence)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn backward_into<'backend, 'leaf>(
        &self,
        tape: DeviceTape<'backend, 'leaf>,
        logits: usize,
        target: &DeviceTensor,
        rows: usize,
        cols: usize,
        bindings: &[GradientLeafBinding],
        trainer: &mut HostOffloadTrainer<'backend>,
        step: u64,
    ) -> anyhow::Result<GradientStreamReport> {
        match self {
            Self::Single => tape
                .xent_backward_into(logits, target, rows, cols, bindings, trainer, step)
                .map_err(anyhow::Error::from),
            #[cfg(feature = "nccl")]
            Self::Distributed { process_group, .. } => process_group
                .xent_backward_into(tape, logits, target, rows, cols, bindings, trainer, step)
                .map_err(anyhow::Error::from),
        }
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterIdentity {
    name: String,
    rows: usize,
    cols: usize,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CampaignHardwareIdentity {
    cuda_ordinal: usize,
    device_name: String,
    pci_bus_id: String,
    cuda_driver_version: u32,
    nvidia_driver_version: String,
    nvml_cuda_driver_version: u32,
    total_memory_bytes: u64,
}

#[cfg(feature = "cuda")]
fn campaign_hardware_identity(
    cuda: &CudaDeviceIdentity,
    nvml: &NvmlGpuSnapshot,
) -> CampaignHardwareIdentity {
    CampaignHardwareIdentity {
        cuda_ordinal: cuda.ordinal,
        device_name: cuda.device_name.clone(),
        pci_bus_id: cuda.pci_bus_id.clone(),
        cuda_driver_version: cuda.cuda_driver_version,
        nvidia_driver_version: nvml.driver_version.clone(),
        nvml_cuda_driver_version: nvml.nvml_cuda_driver_version,
        total_memory_bytes: nvml.total_memory_bytes,
    }
}

#[cfg(feature = "cuda")]
fn sample_nvml_used_memory(
    cuda: &CudaDeviceIdentity,
    expected: &CampaignHardwareIdentity,
) -> anyhow::Result<u64> {
    let sample = probe_cuda_device(cuda).context("sample physical GPU memory through NVML")?;
    if campaign_hardware_identity(cuda, &sample) != *expected {
        bail!("CUDA/NVML hardware identity changed during the campaign");
    }
    Ok(sample.used_memory_bytes)
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanSidecar {
    version: u32,
    fingerprint: String,
    hardware: Vec<CampaignHardwareIdentity>,
    campaign_config_file_digest: String,
    model_config_file_digest: String,
    source_model_digest: String,
    initial_student_digest: String,
    corpus_digest: String,
    corpus_file_digest: String,
    teacher_cache_digest: String,
    evaluation_corpus_path: String,
    evaluation_corpus_digest: String,
    evaluation_corpus_file_digest: String,
    evaluation_windows: u64,
    artifact_path: String,
    growth: CampaignGrowthReceipt,
    seq_len: usize,
    windows: u64,
    total_steps: u64,
    salt_planes: usize,
    state_policy: CampaignStatePolicy,
    packed_compute: CampaignPackedComputePolicy,
    world_size: usize,
    window_partition: String,
    adam_f32_bits: [u32; 5],
    activation_checkpoint_policy: String,
    checkpoint_every: u64,
    timing_warmup_steps: usize,
    checkpoint_shards: usize,
    parameters: Vec<ParameterIdentity>,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy)]
struct PlanInputs<'a> {
    hardware: &'a [CampaignHardwareIdentity],
    campaign_config_file_digest: [u8; 32],
    model_config_file_digest: [u8; 32],
    source_model_digest: [u8; 32],
    initial_student_digest: [u8; 32],
    corpus_digest: [u8; 32],
    corpus_file_digest: [u8; 32],
    teacher_cache_digest: [u8; 32],
    evaluation_corpus_path: &'a Path,
    evaluation_corpus_digest: [u8; 32],
    evaluation_corpus_file_digest: [u8; 32],
    evaluation_windows: u64,
    artifact_path: &'a Path,
    growth: &'a CampaignGrowthReceipt,
    seq_len: usize,
    windows: u64,
    total_steps: u64,
    salt_planes: usize,
    state_policy: CampaignStatePolicy,
    packed_compute: CampaignPackedComputePolicy,
    world_size: usize,
    adam: CampaignAdam,
    depth: usize,
    checkpoint_every: u64,
    timing_warmup_steps: usize,
    checkpoint_shards: usize,
    parameters: &'a [tritium_nn::TrainingParameter],
}

#[cfg(any(feature = "cuda", test))]
fn build_plan_sidecar(inputs: &PlanInputs<'_>) -> PlanSidecar {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium-packed-campaign-plan-v5");
    let hardware_json = serde_json::to_vec(inputs.hardware)
        .expect("serializing campaign hardware identity cannot fail");
    hash_bytes(&mut hash, &hardware_json);
    hash.update(&inputs.campaign_config_file_digest);
    hash.update(&inputs.model_config_file_digest);
    hash.update(&inputs.source_model_digest);
    hash.update(&inputs.initial_student_digest);
    hash.update(&inputs.corpus_digest);
    hash.update(&inputs.corpus_file_digest);
    hash.update(&inputs.teacher_cache_digest);
    hash_bytes(
        &mut hash,
        inputs.evaluation_corpus_path.as_os_str().as_encoded_bytes(),
    );
    hash.update(&inputs.evaluation_corpus_digest);
    hash.update(&inputs.evaluation_corpus_file_digest);
    hash.update(&inputs.evaluation_windows.to_le_bytes());
    hash_bytes(
        &mut hash,
        inputs.artifact_path.as_os_str().as_encoded_bytes(),
    );
    let growth_json =
        serde_json::to_vec(inputs.growth).expect("serializing growth receipt cannot fail");
    hash_bytes(&mut hash, &growth_json);
    hash.update(&(inputs.seq_len as u64).to_le_bytes());
    hash.update(&inputs.windows.to_le_bytes());
    hash.update(&inputs.total_steps.to_le_bytes());
    hash.update(&(inputs.salt_planes as u64).to_le_bytes());
    hash_bytes(&mut hash, inputs.state_policy.as_str().as_bytes());
    hash_bytes(&mut hash, inputs.packed_compute.as_str().as_bytes());
    hash.update(&(inputs.world_size as u64).to_le_bytes());
    hash.update(b"contiguous-modulo-v1");
    let adam_bits = [
        inputs.adam.lr.to_bits(),
        inputs.adam.beta1.to_bits(),
        inputs.adam.beta2.to_bits(),
        inputs.adam.eps.to_bits(),
        inputs.adam.weight_decay.to_bits(),
    ];
    for bits in adam_bits {
        hash.update(&bits.to_le_bytes());
    }
    hash.update(b"sqrt_depth");
    hash.update(&(inputs.depth as u64).to_le_bytes());
    hash.update(&inputs.checkpoint_every.to_le_bytes());
    hash.update(&(inputs.timing_warmup_steps as u64).to_le_bytes());
    hash.update(&(inputs.checkpoint_shards as u64).to_le_bytes());
    let parameters = inputs
        .parameters
        .iter()
        .map(|parameter| {
            hash_bytes(&mut hash, parameter.name.as_bytes());
            hash.update(&(parameter.rows as u64).to_le_bytes());
            hash.update(&(parameter.cols as u64).to_le_bytes());
            ParameterIdentity {
                name: parameter.name.clone(),
                rows: parameter.rows,
                cols: parameter.cols,
            }
        })
        .collect();
    let fingerprint = hex_digest(*hash.finalize().as_bytes());
    PlanSidecar {
        version: 5,
        fingerprint,
        hardware: inputs.hardware.to_vec(),
        campaign_config_file_digest: hex_digest(inputs.campaign_config_file_digest),
        model_config_file_digest: hex_digest(inputs.model_config_file_digest),
        source_model_digest: hex_digest(inputs.source_model_digest),
        initial_student_digest: hex_digest(inputs.initial_student_digest),
        corpus_digest: hex_digest(inputs.corpus_digest),
        corpus_file_digest: hex_digest(inputs.corpus_file_digest),
        teacher_cache_digest: hex_digest(inputs.teacher_cache_digest),
        evaluation_corpus_path: inputs.evaluation_corpus_path.display().to_string(),
        evaluation_corpus_digest: hex_digest(inputs.evaluation_corpus_digest),
        evaluation_corpus_file_digest: hex_digest(inputs.evaluation_corpus_file_digest),
        evaluation_windows: inputs.evaluation_windows,
        artifact_path: inputs.artifact_path.display().to_string(),
        growth: inputs.growth.clone(),
        seq_len: inputs.seq_len,
        windows: inputs.windows,
        total_steps: inputs.total_steps,
        salt_planes: inputs.salt_planes,
        state_policy: inputs.state_policy,
        packed_compute: inputs.packed_compute,
        world_size: inputs.world_size,
        window_partition: "contiguous-modulo-v1".to_owned(),
        adam_f32_bits: adam_bits,
        activation_checkpoint_policy: format!("sqrt_depth:{}", inputs.depth),
        checkpoint_every: inputs.checkpoint_every,
        timing_warmup_steps: inputs.timing_warmup_steps,
        checkpoint_shards: inputs.checkpoint_shards,
        parameters,
    }
}

#[cfg(any(feature = "cuda", test))]
fn ensure_plan_sidecar(checkpoint_dir: &Path, expected: &PlanSidecar) -> anyhow::Result<()> {
    let path = checkpoint_dir.join("campaign-plan.json");
    let manifest_path = checkpoint_dir.join("manifest.tdcp");
    let manifest_exists = manifest_path
        .try_exists()
        .with_context(|| format!("inspect DCP manifest {}", manifest_path.display()))?;
    match std::fs::read(&path) {
        Ok(bytes) => {
            let actual: PlanSidecar = serde_json::from_slice(&bytes)
                .with_context(|| format!("parse immutable plan sidecar {}", path.display()))?;
            if &actual != expected {
                bail!(
                    "campaign plan mismatch at {}; refusing checkpoint load before mutating trainer state",
                    path.display()
                );
            }
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && manifest_exists => bail!(
            "checkpoint {} has a committed manifest but no immutable campaign-plan.json; refusing load",
            checkpoint_dir.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let bytes = serde_json::to_vec_pretty(expected)?;
            atomic_write(&path, &bytes)
        }
        Err(error) => Err(error).with_context(|| format!("read plan sidecar {}", path.display())),
    }
}

#[cfg(feature = "cuda")]
fn snapshot_teacher_cache(
    path: &Path,
    expected: &TeacherCacheHeader,
    window_elements: usize,
) -> anyhow::Result<([u8; 32], Vec<[u8; 32]>)> {
    let mut reader = TeacherCacheReader::open(path, expected)
        .with_context(|| format!("open validated offline teacher cache {}", path.display()))?;
    let mut file_hash = blake3::Hasher::new();
    file_hash.update(
        &write_teacher_cache_header(expected).context("encode teacher cache header for digest")?,
    );
    let window_count = usize::try_from(expected.windows).context("window count exceeds usize")?;
    let mut window = vec![0.0f32; window_elements];
    let mut window_digests = Vec::with_capacity(window_count);
    for index in 0..expected.windows {
        reader
            .read_window(index, &mut window)
            .with_context(|| format!("snapshot teacher window {index}"))?;
        for value in &window {
            file_hash.update(&value.to_bits().to_le_bytes());
        }
        window_digests.push(hash_probability_window(&window));
    }
    Ok((*file_hash.finalize().as_bytes(), window_digests))
}

#[cfg(feature = "cuda")]
fn hash_probability_window(values: &[f32]) -> [u8; 32] {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium-teacher-probability-window-v1");
    hash_f32s(&mut hash, values);
    *hash.finalize().as_bytes()
}

#[cfg(not(feature = "cuda"))]
fn run_campaign(_config_path: &Path) -> anyhow::Result<()> {
    bail!("`tritium campaign run` requires rebuilding tritium-cli with `--features cuda`")
}

#[cfg(feature = "cuda")]
fn run_campaign(config_path: &Path) -> anyhow::Result<()> {
    let config_bytes = std::fs::read(config_path)
        .with_context(|| format!("read campaign config {}", config_path.display()))?;
    let config_file_digest = blake3_digest(&config_bytes);
    let mut campaign: CampaignConfig = serde_json::from_slice(&config_bytes)
        .with_context(|| format!("parse campaign config {}", config_path.display()))?;
    campaign.resolve_paths(config_path);
    validate_campaign_config(&campaign)?;
    validate_campaign_paths(config_path, &campaign)?;

    #[cfg(feature = "nccl")]
    let worker_rendezvous = {
        let inherited =
            WorkerRendezvous::from_env().context("decode campaign worker rendezvous")?;
        match (&campaign.distributed, inherited) {
            (None, None) => None,
            (None, Some(_)) => {
                bail!("distributed worker rendezvous is present but campaign.distributed is absent")
            }
            (Some(distributed), None) => {
                let fleet = DeviceFleet::new(distributed.devices.clone(), 0)?;
                let worker_count = fleet.world_size().get();
                let supervisor = DistributedConfig::new(
                    fleet,
                    Duration::from_secs(distributed.worker_timeout_seconds),
                )?;
                let arguments: Vec<OsString> = std::env::args_os().skip(1).collect();
                let report = supervise_current_exe(&supervisor, &arguments)
                    .context("supervise distributed campaign workers")?;
                if report.workers().len() != worker_count
                    || report.workers().iter().enumerate().any(|(rank, worker)| {
                        worker.rank().get() != rank || !worker.status().success()
                    })
                {
                    bail!("distributed supervisor returned incomplete worker evidence");
                }
                eprintln!(
                    "distributed campaign {} completed {} ranks in {:.3}s",
                    report.job_nonce().as_str(),
                    report.workers().len(),
                    report.elapsed().as_secs_f64()
                );
                return Ok(());
            }
            (Some(distributed), Some(worker)) => {
                let fleet = DeviceFleet::new(distributed.devices.clone(), 0)?;
                if worker.world_size() != fleet.world_size() {
                    bail!("worker world size differs from campaign distributed device count");
                }
                let rank = worker.slot().rank().get();
                let expected_slot = fleet.slot(rank)?;
                if worker.slot().device_ordinal() != expected_slot.device_ordinal()
                    || worker.lock_owner().get() != 0
                    || worker.owns_campaign_lock() != (rank == 0)
                {
                    bail!("worker rendezvous disagrees with immutable distributed topology");
                }
                Some(worker)
            }
        }
    };

    #[cfg(feature = "nccl")]
    let owns_campaign_lock = worker_rendezvous
        .as_ref()
        .is_none_or(WorkerRendezvous::owns_campaign_lock);
    #[cfg(not(feature = "nccl"))]
    let owns_campaign_lock = true;
    let _campaign_locks = owns_campaign_lock
        .then(|| {
            CampaignLocks::acquire(
                &campaign.checkpoint_dir,
                &campaign.report,
                &campaign.artifact,
            )
        })
        .transpose()?;

    #[cfg(feature = "nccl")]
    let cuda_device = worker_rendezvous
        .as_ref()
        .map(|worker| worker.slot().device_ordinal())
        .or(campaign.cuda_device)
        .unwrap_or(0);
    #[cfg(not(feature = "nccl"))]
    let cuda_device = campaign.cuda_device.unwrap_or(0);
    let backend =
        CudaBackend::new(cuda_device).with_context(|| format!("open CUDA device {cuda_device}"))?;
    #[cfg(feature = "nccl")]
    let world = match worker_rendezvous {
        Some(worker) => CampaignWorld::distributed(worker, &backend)?,
        None => CampaignWorld::single(),
    };
    #[cfg(not(feature = "nccl"))]
    let world = CampaignWorld::single();
    let (cuda_identity, telemetry) = backend
        .start_memory_telemetry()
        .context("start exact CUDA allocator telemetry")?;
    let nvml = probe_cuda_device(&cuda_identity).context("validate CUDA device through NVML")?;
    let local_hardware = campaign_hardware_identity(&cuda_identity, &nvml);
    let hardware = world.gather_hardware(&local_hardware)?;
    telemetry
        .reset_synchronized()
        .context("reset CUDA allocator telemetry after hardware validation")?;
    let raw_model_config = std::fs::read(campaign.model_dir.join("config.json"))
        .with_context(|| format!("read model config from {}", campaign.model_dir.display()))?;
    let raw_model_config_json =
        std::str::from_utf8(&raw_model_config).context("model config.json is not UTF-8")?;
    let (model_config, spec, weights) = ModelWeights::load_hf(&campaign.model_dir)
        .with_context(|| format!("load HuggingFace model {}", campaign.model_dir.display()))?;
    let mut model = TiedSwiGluTrainingModel::extract(&model_config, &spec, &weights)
        .context("validate tied-SwiGLU training adapter")?;
    let source_parameter_count =
        training_parameter_count(&model).context("count source model parameters")?;
    let source_model_digest = semantic_model_digest(&model_config, &spec, &model);
    let model_config_file_digest = blake3_digest(&raw_model_config);
    let (tokens, corpus_file_digest) = read_corpus(&campaign.corpus)?;
    let windows = validate_windows(
        &tokens,
        campaign.seq_len,
        model.architecture().vocab,
        model.architecture().n_ctx,
    )?;
    if windows < u64::try_from(world.world_size()).context("world size exceeds u64")? {
        bail!(
            "training corpus has {windows} windows, fewer than distributed world size {}",
            world.world_size()
        );
    }
    let seq_len_u32 = u32::try_from(campaign.seq_len).context("sequence length exceeds u32")?;
    let corpus_digest = hash_teacher_corpus(&tokens, seq_len_u32);
    let expected_cache = TeacherCacheHeader {
        seq_len: seq_len_u32,
        vocab: u32::try_from(model.architecture().vocab).context("vocabulary exceeds u32")?,
        windows,
        model_hash: source_model_digest,
        corpus_hash: corpus_digest,
    };
    let window_elements = expected_cache
        .window_elements()
        .context("teacher-cache window shape overflows usize")?;
    let (teacher_cache_digest, teacher_window_digests) =
        snapshot_teacher_cache(&campaign.teacher_cache, &expected_cache, window_elements)?;
    let mut teacher = TeacherCacheReader::open(&campaign.teacher_cache, &expected_cache)
        .with_context(|| {
            format!(
                "reopen snapshotted offline teacher cache {}",
                campaign.teacher_cache.display()
            )
        })?;

    let (evaluation_tokens, evaluation_corpus_file_digest) =
        read_corpus(&campaign.evaluation_corpus)?;
    let evaluation_windows = validate_windows(
        &evaluation_tokens,
        campaign.seq_len,
        model.architecture().vocab,
        model.architecture().n_ctx,
    )?;
    let evaluation_corpus_digest = hash_teacher_corpus(&evaluation_tokens, seq_len_u32);
    validate_held_out_corpus(
        &tokens,
        corpus_digest,
        &evaluation_tokens,
        evaluation_corpus_digest,
        campaign.seq_len,
    )?;
    let source_evaluation_path = campaign
        .checkpoint_dir
        .join("source-evaluation-progress.json");
    let source_score = if world.is_owner() {
        let source_runner_config = model_config.clone();
        Some(
            resumable_teacher_forced_score(
                &source_evaluation_path,
                "source-fp",
                &hex_digest(source_model_digest),
                &evaluation_tokens,
                campaign.seq_len,
                evaluation_corpus_file_digest,
                evaluation_corpus_digest,
                move || {
                    Ok(ModelRunner::from_weights(
                        source_runner_config,
                        weights,
                        Box::new(tritium_cpu::CpuBackend::new()),
                    ))
                },
            )
            .context("score source fp model on the held-out evaluation corpus")?,
        )
    } else {
        drop(weights);
        None
    };

    let old_width = model.architecture().n_ff;
    let (new_width, growth_seed) = campaign.growth.map_or((old_width, 0), |growth| {
        (growth.intermediate_size, growth.seed)
    });
    let widening = model
        .widen_intermediate(new_width, growth_seed)
        .context("apply deterministic intermediate-width growth")?;
    let growth = campaign_growth_receipt(&widening, growth_seed)?;
    let artifact_growth = GrowthReceipt::from_plan(&widening, growth_seed)
        .context("build artifact growth receipt")?;
    let mut student_config = model_config.clone();
    student_config.n_ff =
        u32::try_from(new_width).context("grown intermediate width exceeds u32")?;
    let grown_parameter_count =
        training_parameter_count(&model).context("count initial student parameters")?;
    let initial_student_digest = semantic_model_digest(&student_config, &spec, &model);
    let evaluation_corpus_path = path_identity(&campaign.evaluation_corpus)?;
    let artifact_path = path_identity(&campaign.artifact)?;

    let plan = build_plan_sidecar(&PlanInputs {
        hardware: &hardware,
        campaign_config_file_digest: config_file_digest,
        model_config_file_digest,
        source_model_digest,
        initial_student_digest,
        corpus_digest,
        corpus_file_digest,
        teacher_cache_digest,
        evaluation_corpus_path: &evaluation_corpus_path,
        evaluation_corpus_digest,
        evaluation_corpus_file_digest,
        evaluation_windows,
        artifact_path: &artifact_path,
        growth: &growth,
        seq_len: campaign.seq_len,
        windows,
        total_steps: campaign.steps,
        salt_planes: campaign.salt_planes,
        state_policy: campaign.state_policy,
        packed_compute: campaign.packed_compute,
        world_size: world.world_size(),
        adam: campaign.adam,
        depth: model.architecture().n_layers,
        checkpoint_every: campaign.checkpoint_every,
        timing_warmup_steps: campaign.timing_warmup_steps,
        checkpoint_shards: campaign.checkpoint_shards,
        parameters: model.parameters(),
    });
    let input_hashes = CampaignInputHashes {
        campaign_config_file: hex_digest(config_file_digest),
        model_config_file: hex_digest(model_config_file_digest),
        source_model: hex_digest(source_model_digest),
        initial_student_model: hex_digest(initial_student_digest),
        corpus_file: hex_digest(corpus_file_digest),
        corpus: hex_digest(corpus_digest),
        teacher_cache: hex_digest(teacher_cache_digest),
        evaluation_corpus_file: hex_digest(evaluation_corpus_file_digest),
        evaluation_corpus: hex_digest(evaluation_corpus_digest),
    };
    // This comparison/creation happens before DCP load can mutate trainer state.
    if world.is_owner() {
        ensure_plan_sidecar(&campaign.checkpoint_dir, &plan)?;
    }
    world.barrier()?;
    if !world.is_owner() {
        ensure_plan_sidecar(&campaign.checkpoint_dir, &plan)?;
    }
    world.verify_contract_digest(parse_lower_hex_digest(&plan.fingerprint)?)?;

    let initial_student_model_identity = hex_digest(initial_student_digest);
    let grown_fp_score = if world.is_owner() && new_width > old_width {
        let grown_runner_config = student_config.clone();
        Some(
            resumable_teacher_forced_score(
                &campaign
                    .checkpoint_dir
                    .join("grown-fp-evaluation-progress.json"),
                "grown-fp",
                &initial_student_model_identity,
                &evaluation_tokens,
                campaign.seq_len,
                evaluation_corpus_file_digest,
                evaluation_corpus_digest,
                || {
                    let dense_weights = model
                        .to_dense_weights()
                        .context("reconstruct widened dense-fp evaluation weights")?;
                    Ok(ModelRunner::from_weights(
                        grown_runner_config,
                        dense_weights,
                        Box::new(tritium_cpu::CpuBackend::new()),
                    ))
                },
            )
            .context("score grown fp model on the held-out evaluation corpus")?,
        )
    } else {
        None
    };

    // Score the immutable, widened student before optimizer ownership drains or
    // mutates any master. Only rank 0 owns durable evaluation evidence. Packing
    // is lazy, so a completed progress sidecar resumes without rebuilding the
    // compact PTQ model.
    let naive_salt_ptq_model_identity =
        naive_salt_ptq_model_identity(&initial_student_model_identity, campaign.salt_planes)?;
    let naive_salt_ptq_score = if world.is_owner() {
        Some(
            resumable_naive_salt_ptq_score(
                &campaign
                    .checkpoint_dir
                    .join("naive-salt-ptq-evaluation-progress.json"),
                &backend,
                &model,
                &naive_salt_ptq_model_identity,
                &evaluation_tokens,
                campaign.seq_len,
                evaluation_corpus_file_digest,
                evaluation_corpus_digest,
                campaign.salt_planes,
            )
            .context("score naive SALT PTQ on the held-out evaluation corpus")?,
        )
    } else {
        None
    };
    world.barrier()?;
    let telemetry_baseline = telemetry
        .reset_synchronized()
        .context("reset CUDA allocator telemetry after PTQ baseline evaluation")?;

    let parameter_geometry: Vec<_> = model
        .parameters()
        .iter()
        .map(|parameter| HostOffloadParamMetadata {
            rows: parameter.rows,
            cols: parameter.cols,
            salt_planes: campaign.salt_planes,
        })
        .collect();
    let host_geometry = host_offload_memory_geometry(&parameter_geometry)
        .context("compute static SALT training memory geometry")?;
    let static_memory = campaign_static_memory(campaign.state_policy, host_geometry)?;
    let optimizer = AdamW {
        lr: campaign.adam.lr,
        beta1: campaign.adam.beta1,
        beta2: campaign.adam.beta2,
        eps: campaign.adam.eps,
        weight_decay: campaign.adam.weight_decay,
    };
    let masters = model.take_parameter_masters();
    let specs: Vec<_> = model
        .parameters()
        .iter()
        .zip(masters)
        .map(|(parameter, master)| HostOffloadTrainParam {
            master,
            rows: parameter.rows,
            cols: parameter.cols,
            salt_planes: campaign.salt_planes,
            optimizer,
        })
        .collect();
    if campaign.state_policy == CampaignStatePolicy::Resident {
        let resident_floor = static_memory
            .resident_training_state_bytes
            .checked_add(static_memory.packed_parameter_bytes)
            .context("resident state plus packed-weight bytes overflow")?;
        if u64::try_from(resident_floor).context("resident byte floor exceeds u64")?
            > local_hardware.total_memory_bytes
        {
            bail!(
                "resident training state plus packed weights require at least {resident_floor} bytes, exceeding device capacity {}; use state_policy=\"host-offload\"",
                local_hardware.total_memory_bytes
            );
        }
    }
    let mut trainer = CampaignTrainer::new(&backend, specs, campaign.state_policy)?;
    let manifest = campaign.checkpoint_dir.join("manifest.tdcp");
    if manifest
        .try_exists()
        .with_context(|| format!("inspect DCP manifest {}", manifest.display()))?
    {
        trainer
            .load_checkpoint(&campaign.checkpoint_dir)
            .with_context(|| {
                format!("load DCP checkpoint {}", campaign.checkpoint_dir.display())
            })?;
    }
    if trainer.completed_step() > campaign.steps {
        bail!(
            "checkpoint completed step {} exceeds configured terminal step {}",
            trainer.completed_step(),
            campaign.steps
        );
    }
    world.verify_completed_step(trainer.completed_step())?;
    let report_expectations = ReportExpectations {
        plan_fingerprint: &plan.fingerprint,
        state_policy: campaign.state_policy,
        packed_compute: campaign.packed_compute,
        checkpoint_step: trainer.completed_step(),
        checkpoint_dir: &campaign.checkpoint_dir,
        windows,
        configured_steps: campaign.steps,
        timing_warmup_steps: campaign.timing_warmup_steps,
        evaluation_seq_len: campaign.seq_len,
        evaluation_windows,
        evaluation_corpus_path: &evaluation_corpus_path,
        hardware: &hardware,
        input_hashes: &input_hashes,
        static_memory,
        salt_planes: campaign.salt_planes,
        source_parameter_count,
        grown_parameter_count,
    };
    let previous_report = if world.is_owner() {
        load_existing_report(&campaign.report, &report_expectations)?
    } else {
        None
    };
    let previous_memory = previous_report.as_ref().map(|report| report.memory.clone());
    let mut timings = previous_report
        .as_ref()
        .map(|report| report.step_timings.clone())
        .unwrap_or_default();
    let mut memory_segments = previous_report
        .as_ref()
        .map(|report| report.cuda_memory_segments.clone())
        .unwrap_or_default();
    let mut max_gradient_elements = 0usize;
    let mut materialized_gradient_elements = 0usize;
    let mut max_activation_elements = 0usize;
    let mut naive_activation_elements = 0usize;
    let resumed_from_step = trainer.completed_step();
    let baseline = CampaignCudaMemorySnapshot::from(telemetry_baseline);
    let mut current_segment = CampaignCudaMemorySegment {
        resumed_from_step,
        completed_step: resumed_from_step,
        nvml_used_memory_bytes_at_launch: nvml.used_memory_bytes,
        nvml_used_memory_latest_sample_bytes: nvml.used_memory_bytes,
        nvml_used_memory_sample_high_water_bytes: nvml.used_memory_bytes,
        baseline,
        post_setup: None,
        latest_or_terminal: baseline,
    };

    if trainer.completed_step() == campaign.steps {
        current_segment.latest_or_terminal = telemetry
            .sample_synchronized()
            .context("sample CUDA memory before terminal artifact verification")?
            .into();
        current_segment.observe_nvml(sample_nvml_used_memory(&cuda_identity, &local_hardware)?);
        world.barrier()?;
        if !world.is_owner() {
            drop(trainer);
            world.barrier()?;
            return Ok(());
        }
        memory_segments.push(current_segment);
        let completed_step = trainer.completed_step();
        let trainer_stats = trainer.stats();
        let master_source = trainer.into_master_source()?;
        let artifact = prepare_terminal_artifact(
            &campaign,
            raw_model_config_json,
            &model,
            master_source.as_ref(),
            &plan,
            source_model_digest,
            initial_student_digest,
            &artifact_growth,
        )?;
        drop(master_source);
        let evaluation = evaluate_terminal_artifact(
            &campaign,
            &artifact,
            &plan,
            source_model_digest,
            initial_student_digest,
            &growth,
            &evaluation_tokens,
            source_score.context("rank 0 source evaluation evidence is missing")?,
            grown_fp_score,
            source_parameter_count,
            grown_parameter_count,
            naive_salt_ptq_score.context("rank 0 naive SALT PTQ evidence is missing")?,
            &naive_salt_ptq_model_identity,
            evaluation_corpus_file_digest,
            evaluation_corpus_digest,
        )?;
        let report = build_campaign_report(
            &campaign,
            &plan,
            &hardware,
            &input_hashes,
            completed_step,
            trainer_stats,
            resumed_from_step,
            &timings,
            static_memory,
            previous_memory.as_ref(),
            0,
            0,
            0,
            0,
            memory_segments,
            Some(artifact),
            Some(evaluation),
        )?;
        atomic_write(&campaign.report, &serde_json::to_vec_pretty(&report)?)?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        world.barrier()?;
        return Ok(());
    }

    trainer.prepare_execution(&backend)?;
    let mut target_host = vec![0.0f32; window_elements];
    let timing_fence =
        DeviceTensor::upload(&backend, &[0.0]).context("allocate training-stream timing fence")?;
    current_segment.post_setup = Some(
        telemetry
            .sample_synchronized()
            .context("sample CUDA memory after campaign setup")?
            .into(),
    );
    current_segment.observe_nvml(sample_nvml_used_memory(&cuda_identity, &local_hardware)?);
    let first_step = trainer
        .completed_step()
        .checked_add(1)
        .context("completed step overflow")?;

    for step in first_step..=campaign.steps {
        let started = Instant::now();
        let window_index = world.window_index(step, windows)?;
        teacher
            .read_window(window_index, &mut target_host)
            .with_context(|| format!("read teacher window {window_index}"))?;
        let expected_window_digest = teacher_window_digests
            .get(usize::try_from(window_index).context("window index exceeds usize")?)
            .context("teacher window digest index is out of range")?;
        if &hash_probability_window(&target_host) != expected_window_digest {
            bail!(
                "teacher cache window {window_index} changed after plan snapshot; refusing optimizer mutation"
            );
        }
        let target = DeviceTensor::upload(&backend, &target_host)
            .context("upload offline teacher probability window")?;
        let token_offset = usize::try_from(window_index)
            .context("window index exceeds usize")?
            .checked_mul(campaign.seq_len)
            .context("window token offset overflow")?;
        let window_tokens = &tokens[token_offset..token_offset + campaign.seq_len];
        let tokens_i32 = window_tokens
            .iter()
            .map(|&token| i32::try_from(token).context("token id exceeds i32"))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let evidence = trainer.run_step(
            &world,
            &backend,
            &model,
            &target,
            &tokens_i32,
            step,
            &timing_fence,
            campaign.packed_compute,
        )?;

        max_gradient_elements =
            max_gradient_elements.max(evidence.peak_live_requested_gradient_elements);
        materialized_gradient_elements =
            materialized_gradient_elements.max(evidence.materialized_gradient_elements);
        max_activation_elements =
            max_activation_elements.max(evidence.peak_live_activation_elements);
        naive_activation_elements =
            naive_activation_elements.max(evidence.naive_activation_elements);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        current_segment.completed_step = step;
        current_segment.latest_or_terminal = telemetry
            .sample_synchronized()
            .with_context(|| format!("sample CUDA allocator memory after step {step}"))?
            .into();
        current_segment.observe_nvml(sample_nvml_used_memory(&cuda_identity, &local_hardware)?);
        let rank_evidence = world.gather_step_evidence(RankStepEvidence {
            rank: world.rank(),
            window_index,
            training_elapsed_ms: elapsed_ms,
            nvml_used_memory_bytes: current_segment.nvml_used_memory_latest_sample_bytes,
            cuda_memory: current_segment.latest_or_terminal,
        })?;
        if world.is_owner() {
            let max_elapsed = rank_evidence
                .iter()
                .map(|evidence| evidence.training_elapsed_ms)
                .fold(0.0_f64, f64::max);
            let owner_window = rank_evidence
                .first()
                .context("distributed step evidence has no rank 0 record")?
                .window_index;
            timings.push(StepTiming {
                step,
                window_index: owner_window,
                warmup: step - first_step
                    < u64::try_from(campaign.timing_warmup_steps)
                        .context("timing warmup step count exceeds u64")?,
                training_elapsed_ms_excluding_checkpoint: max_elapsed,
                rank_evidence,
            });
        }

        if step.is_multiple_of(campaign.checkpoint_every) || step == campaign.steps {
            world.verify_completed_step(trainer.completed_step())?;
            if world.is_owner() {
                let mut progress_segments = memory_segments.clone();
                progress_segments.push(current_segment.clone());
                let progress = build_campaign_report(
                    &campaign,
                    &plan,
                    &hardware,
                    &input_hashes,
                    trainer.completed_step(),
                    trainer.stats(),
                    resumed_from_step,
                    &timings,
                    static_memory,
                    previous_memory.as_ref(),
                    max_gradient_elements,
                    materialized_gradient_elements,
                    max_activation_elements,
                    naive_activation_elements,
                    progress_segments,
                    None,
                    None,
                )?;
                atomic_write(&campaign.report, &serde_json::to_vec_pretty(&progress)?)?;
                // Publish evidence before DCP's manifest commit. If saving
                // stops early, resume truncates timings beyond the manifest.
                trainer
                    .save_checkpoint(&campaign.checkpoint_dir, campaign.checkpoint_shards)
                    .with_context(|| format!("save DCP at completed step {step}"))?;
            }
            world.barrier()?;
        }
    }

    if !world.is_owner() {
        drop(trainer);
        world.barrier()?;
        return Ok(());
    }
    memory_segments.push(current_segment);
    let completed_step = trainer.completed_step();
    let trainer_stats = trainer.stats();
    let master_source = trainer.into_master_source()?;
    let artifact = prepare_terminal_artifact(
        &campaign,
        raw_model_config_json,
        &model,
        master_source.as_ref(),
        &plan,
        source_model_digest,
        initial_student_digest,
        &artifact_growth,
    )?;
    drop(master_source);
    let evaluation = evaluate_terminal_artifact(
        &campaign,
        &artifact,
        &plan,
        source_model_digest,
        initial_student_digest,
        &growth,
        &evaluation_tokens,
        source_score.context("rank 0 source evaluation evidence is missing")?,
        grown_fp_score,
        source_parameter_count,
        grown_parameter_count,
        naive_salt_ptq_score.context("rank 0 naive SALT PTQ evidence is missing")?,
        &naive_salt_ptq_model_identity,
        evaluation_corpus_file_digest,
        evaluation_corpus_digest,
    )?;
    let report = build_campaign_report(
        &campaign,
        &plan,
        &hardware,
        &input_hashes,
        completed_step,
        trainer_stats,
        resumed_from_step,
        &timings,
        static_memory,
        previous_memory.as_ref(),
        max_gradient_elements,
        materialized_gradient_elements,
        max_activation_elements,
        naive_activation_elements,
        memory_segments,
        Some(artifact),
        Some(evaluation),
    )?;
    atomic_write(&campaign.report, &serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    world.barrier()?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_campaign_config(config: &CampaignConfig) -> anyhow::Result<()> {
    if config.seq_len < 2 {
        bail!("campaign seq_len must be at least 2 for held-out next-token scoring");
    }
    if config.steps == 0 {
        bail!("campaign steps must be non-zero");
    }
    if !(1..=3).contains(&config.salt_planes) {
        bail!("salt_planes must be in 1..=3");
    }
    if let Some(distributed) = &config.distributed {
        if config.cuda_device.is_some() {
            bail!("cuda_device and distributed.devices are mutually exclusive");
        }
        if distributed.devices.len() < 2 {
            bail!("distributed.devices must contain at least two CUDA ordinals");
        }
        if distributed.worker_timeout_seconds == 0 {
            bail!("distributed.worker_timeout_seconds must be non-zero");
        }
        let unique: BTreeSet<_> = distributed.devices.iter().copied().collect();
        if unique.len() != distributed.devices.len() {
            bail!("distributed.devices must not contain duplicate CUDA ordinals");
        }
        if config.state_policy != CampaignStatePolicy::HostOffload {
            bail!("distributed campaigns currently require state_policy=\"host-offload\"");
        }
        #[cfg(not(feature = "nccl"))]
        bail!("distributed campaigns require rebuilding tritium-cli with --features nccl");
    }
    if config.checkpoint_every == 0 {
        bail!("checkpoint_every must be non-zero");
    }
    if u64::try_from(config.timing_warmup_steps).map_or(true, |warmup| warmup > config.steps) {
        bail!("timing_warmup_steps must not exceed configured steps");
    }
    if config.checkpoint_shards == 0 {
        bail!("checkpoint_shards must be non-zero");
    }
    if config
        .growth
        .is_some_and(|growth| growth.intermediate_size == 0)
    {
        bail!("growth intermediate_size must be non-zero");
    }
    let adam = config.adam;
    if !adam.lr.is_finite() || adam.lr <= 0.0 {
        bail!("AdamW learning rate must be finite and positive");
    }
    if !adam.beta1.is_finite() || !(0.0..1.0).contains(&adam.beta1) {
        bail!("AdamW beta1 must be finite and in [0, 1)");
    }
    if !adam.beta2.is_finite() || !(0.0..1.0).contains(&adam.beta2) {
        bail!("AdamW beta2 must be finite and in [0, 1)");
    }
    if !adam.eps.is_finite() || adam.eps <= 0.0 {
        bail!("AdamW epsilon must be finite and positive");
    }
    if !adam.weight_decay.is_finite() || adam.weight_decay < 0.0 {
        bail!("AdamW weight decay must be finite and non-negative");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn build_packed_weights(
    backend: &CudaBackend,
    trainer: &HostOffloadTrainer<'_>,
) -> anyhow::Result<Vec<DevicePackedSaltWeight>> {
    (0..trainer.len())
        .map(|index| {
            let metadata = trainer.parameter_metadata(index)?;
            DevicePackedSaltWeight::from_host(
                backend,
                trainer.master(index)?,
                metadata.rows,
                metadata.cols,
                metadata.salt_planes,
            )
            .map_err(anyhow::Error::from)
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn repack_all_or_stale(
    backend: &CudaBackend,
    packed: &mut [DevicePackedSaltWeight],
    trainer: &HostOffloadTrainer<'_>,
) -> anyhow::Result<()> {
    if packed.len() != trainer.len() {
        bail!(
            "packed parameter count {} differs from trainer count {}",
            packed.len(),
            trainer.len()
        );
    }
    for index in 0..packed.len() {
        let result = trainer
            .master(index)
            .and_then(|master| packed[index].repack_from_host(backend, master));
        if let Err(error) = result {
            for weight in packed.iter_mut() {
                weight.mark_stale();
            }
            return Err(error.into());
        }
    }
    if packed.iter().any(|weight| !weight.is_prepared()) {
        for weight in packed.iter_mut() {
            weight.mark_stale();
        }
        bail!("successful all-parameter repack left a stale packed handle");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
struct CampaignStepEvidence {
    peak_live_requested_gradient_elements: usize,
    materialized_gradient_elements: usize,
    peak_live_activation_elements: usize,
    naive_activation_elements: usize,
}

#[cfg(feature = "cuda")]
enum CampaignTrainer<'a> {
    Resident {
        trainer: Box<DeviceTrainer<'a>>,
        packed: Vec<DevicePackedSaltWeight>,
        stable_bindings: Option<Vec<GradientLeafBinding>>,
    },
    HostOffload {
        trainer: Box<HostOffloadTrainer<'a>>,
        packed: Vec<DevicePackedSaltWeight>,
        stable_bindings: Option<Vec<GradientLeafBinding>>,
    },
}

#[cfg(feature = "cuda")]
impl<'a> CampaignTrainer<'a> {
    fn new(
        backend: &'a CudaBackend,
        specs: Vec<HostOffloadTrainParam>,
        policy: CampaignStatePolicy,
    ) -> anyhow::Result<Self> {
        match policy {
            CampaignStatePolicy::Resident => {
                let borrowed: Vec<_> = specs
                    .iter()
                    .map(|parameter| DeviceTrainParam {
                        master: &parameter.master,
                        rows: parameter.rows,
                        cols: parameter.cols,
                        salt_planes: parameter.salt_planes,
                        optimizer: parameter.optimizer,
                    })
                    .collect();
                DeviceTrainer::new_with_weight_storage(
                    backend,
                    &borrowed,
                    DeviceTrainerWeightStorage::Packed,
                )
                    .map(|trainer| Self::Resident {
                        trainer: Box::new(trainer),
                        packed: Vec::new(),
                        stable_bindings: None,
                    })
                    .context(
                        "construct resident AdamW trainer; use state_policy=\"host-offload\" if resident CUDA memory is insufficient",
                    )
            }
            CampaignStatePolicy::HostOffload => HostOffloadTrainer::new_owned(backend, specs)
                .map(|trainer| Self::HostOffload {
                    trainer: Box::new(trainer),
                    packed: Vec::new(),
                    stable_bindings: None,
                })
                .context("construct host-offloaded AdamW trainer"),
        }
    }

    fn completed_step(&self) -> u64 {
        match self {
            Self::Resident { trainer, .. } => trainer.completed_step(),
            Self::HostOffload { trainer, .. } => trainer.completed_step(),
        }
    }

    fn stats(&self) -> CampaignTrainerStats {
        match self {
            Self::Resident { trainer, .. } => trainer.resident_stats().into(),
            Self::HostOffload { trainer, .. } => trainer.stats().into(),
        }
    }

    fn load_checkpoint(&mut self, checkpoint_dir: &Path) -> anyhow::Result<()> {
        match self {
            Self::Resident { trainer, .. } => {
                tritium_train::dcp::load_into(checkpoint_dir, trainer.as_mut())
            }
            Self::HostOffload { trainer, .. } => {
                tritium_train::dcp::load_into(checkpoint_dir, trainer.as_mut())
            }
        }
        .map_err(anyhow::Error::from)
    }

    fn save_checkpoint(&mut self, checkpoint_dir: &Path, shard_count: usize) -> anyhow::Result<()> {
        match self {
            Self::Resident { trainer, .. } => {
                tritium_train::dcp::save_from(checkpoint_dir, trainer.as_mut(), shard_count)
            }
            Self::HostOffload { trainer, .. } => {
                tritium_train::dcp::save_from(checkpoint_dir, trainer.as_mut(), shard_count)
            }
        }
        .map_err(anyhow::Error::from)
    }

    fn prepare_execution(&mut self, backend: &CudaBackend) -> anyhow::Result<()> {
        match self {
            Self::Resident {
                trainer, packed, ..
            } => {
                if !packed.is_empty() {
                    bail!("campaign packed execution state was prepared more than once");
                }
                *packed = (0..trainer.len())
                    .map(|index| trainer.packed_weight(index).map_err(anyhow::Error::from))
                    .collect::<anyhow::Result<Vec<_>>>()?;
            }
            Self::HostOffload {
                trainer, packed, ..
            } => {
                if !packed.is_empty() {
                    bail!("campaign packed execution state was prepared more than once");
                }
                *packed = build_packed_weights(backend, trainer)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_step(
        &mut self,
        world: &CampaignWorld,
        backend: &'a CudaBackend,
        model: &TiedSwiGluTrainingModel,
        target: &DeviceTensor,
        tokens: &[i32],
        step: u64,
        timing_fence: &DeviceTensor,
        packed_compute: CampaignPackedComputePolicy,
    ) -> anyhow::Result<CampaignStepEvidence> {
        match self {
            Self::Resident {
                trainer,
                packed,
                stable_bindings,
            } => {
                if packed.len() != trainer.len() {
                    bail!("resident packed execution state is not prepared");
                }
                let stream_result = {
                    let mut tape = DeviceTape::new_with_policies(
                        backend,
                        model.architecture().vocab,
                        CheckpointPolicy::SqrtDepth(model.architecture().n_layers),
                        packed_compute.into(),
                    )
                    .context("create sqrt-depth resident packed device tape")?;
                    let forward = packed_device_forward(&mut tape, model, packed, tokens)
                        .context("build resident packed tied-SwiGLU forward")?;
                    let bindings: Vec<_> = forward
                        .master_leaves
                        .iter()
                        .enumerate()
                        .map(|(parameter_index, &leaf_id)| GradientLeafBinding {
                            leaf_id,
                            parameter_index,
                        })
                        .collect();
                    if let Some(expected) = stable_bindings.as_ref() {
                        if expected != &bindings {
                            bail!("resident packed gradient leaf bindings changed between steps");
                        }
                    } else {
                        *stable_bindings = Some(bindings.clone());
                    }
                    tape.xent_backward_into_resident(
                        forward.logits,
                        target,
                        tokens.len(),
                        model.architecture().vocab,
                        &bindings,
                        trainer,
                        step,
                    )
                    .with_context(|| format!("resident packed training step {step}"))
                };
                for weight in packed.iter_mut() {
                    weight.mark_stale();
                }
                let stream = stream_result?;
                for index in 0..packed.len() {
                    if let Err(error) = trainer.repack_packed_weight(index, &mut packed[index]) {
                        for weight in packed.iter_mut() {
                            weight.mark_stale();
                        }
                        return Err(error).with_context(|| {
                            format!("resident repack parameter {index} at step {step}")
                        });
                    }
                }
                timing_fence
                    .download(backend)
                    .with_context(|| format!("synchronize resident packed step {step}"))?;
                Ok(CampaignStepEvidence {
                    peak_live_requested_gradient_elements: stream
                        .peak_live_requested_gradient_elements,
                    materialized_gradient_elements: stream.materialized_collection_elements,
                    peak_live_activation_elements: stream
                        .backward_stats
                        .peak_live_activation_elements,
                    naive_activation_elements: stream.backward_stats.naive_activation_elements,
                })
            }
            Self::HostOffload {
                trainer,
                packed,
                stable_bindings,
            } => {
                if packed.len() != trainer.len() {
                    bail!("host-offload packed execution state is not prepared");
                }
                let stream_result = {
                    let mut tape = DeviceTape::new_with_policies(
                        backend,
                        model.architecture().vocab,
                        CheckpointPolicy::SqrtDepth(model.architecture().n_layers),
                        packed_compute.into(),
                    )
                    .context("create sqrt-depth packed device tape")?;
                    let forward = packed_device_forward(&mut tape, model, packed, tokens)
                        .context("build packed tied-SwiGLU forward")?;
                    let bindings: Vec<_> = forward
                        .master_leaves
                        .iter()
                        .enumerate()
                        .map(|(parameter_index, &leaf_id)| GradientLeafBinding {
                            leaf_id,
                            parameter_index,
                        })
                        .collect();
                    if let Some(expected) = stable_bindings.as_ref() {
                        if expected != &bindings {
                            bail!("packed gradient leaf bindings changed between steps");
                        }
                    } else {
                        *stable_bindings = Some(bindings.clone());
                    }
                    world.backward_into(
                        tape,
                        forward.logits,
                        target,
                        tokens.len(),
                        model.architecture().vocab,
                        &bindings,
                        trainer,
                        step,
                    )
                };
                for weight in packed.iter_mut() {
                    weight.mark_stale();
                }
                let stream = stream_result.with_context(|| format!("training step {step}"))?;
                repack_all_or_stale(backend, packed, trainer)
                    .with_context(|| format!("repack every parameter after step {step}"))?;
                timing_fence
                    .download(backend)
                    .with_context(|| format!("synchronize packed repack at step {step}"))?;
                Ok(CampaignStepEvidence {
                    peak_live_requested_gradient_elements: stream
                        .peak_live_requested_gradient_elements,
                    materialized_gradient_elements: stream.materialized_collection_elements,
                    peak_live_activation_elements: stream
                        .backward_stats
                        .peak_live_activation_elements,
                    naive_activation_elements: stream.backward_stats.naive_activation_elements,
                })
            }
        }
    }

    fn into_master_source(self) -> anyhow::Result<Box<dyn MasterSource + 'a>> {
        match self {
            Self::HostOffload { trainer, .. } => Ok(trainer),
            Self::Resident { trainer, .. } => {
                let completed_step = trainer.completed_step();
                let geometry = (0..trainer.len())
                    .map(|index| -> anyhow::Result<_> {
                        let metadata = trainer
                            .parameter_metadata(index)
                            .with_context(|| format!("read resident parameter {index} metadata"))?;
                        Ok((metadata.rows, metadata.cols, metadata.salt_planes))
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let masters = (0..trainer.len())
                    .map(|index| trainer.download_master(index).map_err(anyhow::Error::from))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let snapshot = ResidentMasterSnapshot::new(completed_step, masters, geometry)?;
                Ok(Box::new(snapshot))
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn elements_to_bytes(elements: usize) -> anyhow::Result<usize> {
    elements
        .checked_mul(size_of::<f32>())
        .context("f32 byte count overflow")
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CampaignTrainerStats {
    host_optimizer_elements: usize,
    resident_training_state_elements: usize,
    peak_optimizer_device_elements: usize,
}

#[cfg(feature = "cuda")]
impl From<HostOffloadStats> for CampaignTrainerStats {
    fn from(stats: HostOffloadStats) -> Self {
        Self {
            host_optimizer_elements: stats.host_optimizer_elements,
            resident_training_state_elements: 0,
            peak_optimizer_device_elements: stats.peak_optimizer_device_elements,
        }
    }
}

#[cfg(feature = "cuda")]
impl From<ResidentTrainerStats> for CampaignTrainerStats {
    fn from(stats: ResidentTrainerStats) -> Self {
        Self {
            host_optimizer_elements: 0,
            resident_training_state_elements: stats.resident_elements,
            peak_optimizer_device_elements: 0,
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn build_campaign_report(
    campaign: &CampaignConfig,
    plan: &PlanSidecar,
    hardware: &[CampaignHardwareIdentity],
    input_hashes: &CampaignInputHashes,
    completed_step: u64,
    trainer_stats: CampaignTrainerStats,
    resumed_from_step: u64,
    step_timings: &[StepTiming],
    static_memory: CampaignStaticMemory,
    previous_memory: Option<&CampaignMemoryReport>,
    max_gradient_elements: usize,
    materialized_gradient_elements: usize,
    max_activation_elements: usize,
    naive_activation_elements: usize,
    cuda_memory_segments: Vec<CampaignCudaMemorySegment>,
    artifact: Option<CampaignArtifactReport>,
    evaluation: Option<CampaignEvaluationReport>,
) -> anyhow::Result<CampaignReport> {
    let host_optimizer_bytes = elements_to_bytes(trainer_stats.host_optimizer_elements)?;
    let resident_training_state_bytes =
        elements_to_bytes(trainer_stats.resident_training_state_elements)?;
    let optimizer_staging_bytes = elements_to_bytes(trainer_stats.peak_optimizer_device_elements)?;
    for (label, actual, expected) in [
        (
            "host optimizer",
            host_optimizer_bytes,
            static_memory.host_optimizer_bytes,
        ),
        (
            "resident training state",
            resident_training_state_bytes,
            static_memory.resident_training_state_bytes,
        ),
        (
            "optimizer staging",
            optimizer_staging_bytes,
            static_memory.peak_optimizer_staging_bytes,
        ),
    ] {
        if actual != expected {
            bail!(
                "runtime {label} bytes {actual} do not match static campaign geometry {expected}"
            );
        }
    }
    let previous_peak =
        |select: fn(&CampaignMemoryReport) -> usize| previous_memory.map_or(0, select);
    let peak_optimizer_staging_bytes =
        optimizer_staging_bytes.max(previous_peak(|memory| memory.peak_optimizer_staging_bytes));
    let fleet_host_optimizer_bytes = host_optimizer_bytes
        .checked_mul(plan.world_size)
        .context("fleet host optimizer bytes overflow")?;
    let fleet_peak_optimizer_staging_bytes = peak_optimizer_staging_bytes
        .checked_mul(plan.world_size)
        .context("fleet optimizer staging bytes overflow")?;
    let optimizer_state_device_fraction = match campaign.state_policy {
        CampaignStatePolicy::Resident => 1.0,
        CampaignStatePolicy::HostOffload => 0.0,
    };
    Ok(CampaignReport {
        schema_version: 6,
        plan_fingerprint: plan.fingerprint.clone(),
        state_policy: campaign.state_policy,
        packed_compute: campaign.packed_compute,
        world_size: plan.world_size,
        window_partition: plan.window_partition.clone(),
        hardware: hardware.to_vec(),
        input_hashes: input_hashes.clone(),
        completed_step,
        resumed_from_step,
        configured_steps: campaign.steps,
        timing_warmup_steps: campaign.timing_warmup_steps,
        checkpoint_path: campaign.checkpoint_dir.display().to_string(),
        step_timings: step_timings.to_vec(),
        memory: CampaignMemoryReport {
            packed_parameter_bytes: static_memory.packed_parameter_bytes,
            packed_code_bytes: static_memory.packed_code_bytes,
            packed_scale_bytes: static_memory.packed_scale_bytes,
            dense_parameter_bytes: static_memory.dense_parameter_bytes,
            host_optimizer_bytes: static_memory.host_optimizer_bytes,
            fleet_host_optimizer_bytes,
            host_adapter_master_bytes: 0,
            logical_host_training_state_bytes: static_memory.host_optimizer_bytes,
            resident_training_state_bytes: static_memory.resident_training_state_bytes,
            optimizer_state_device_fraction,
            peak_optimizer_staging_bytes,
            fleet_peak_optimizer_staging_bytes,
            peak_live_requested_gradient_bytes: elements_to_bytes(max_gradient_elements)?.max(
                previous_peak(|memory| memory.peak_live_requested_gradient_bytes),
            ),
            materialized_gradient_bytes: elements_to_bytes(materialized_gradient_elements)?
                .max(previous_peak(|memory| memory.materialized_gradient_bytes)),
            logical_peak_activation_bytes: elements_to_bytes(max_activation_elements)?
                .max(previous_peak(|memory| memory.logical_peak_activation_bytes)),
            logical_naive_activation_bytes: elements_to_bytes(naive_activation_elements)?.max(
                previous_peak(|memory| memory.logical_naive_activation_bytes),
            ),
        },
        cuda_memory_segments,
        artifact,
        evaluation,
    })
}

#[cfg(feature = "cuda")]
fn artifact_report(path: &Path, summary: CampaignArtifactSummary) -> CampaignArtifactReport {
    let package = summary.package();
    CampaignArtifactReport {
        path: path.display().to_string(),
        package_id: package.id().to_string(),
        physical_bytes: package.physical_bytes(),
        scale_max_abs_error: summary.scale_max_abs_error(),
        reconstruction_max_abs_delta: summary.reconstruction_max_abs_delta(),
        reconstruction_squared_error_sum: summary.reconstruction_squared_error_sum(),
        reconstruction_element_count: summary.reconstruction_element_count(),
        reconstruction_mse: summary.reconstruction_mse(),
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn prepare_terminal_artifact(
    campaign: &CampaignConfig,
    raw_hf_config_json: &str,
    model: &TiedSwiGluTrainingModel,
    trainer: &dyn MasterSource,
    plan: &PlanSidecar,
    source_model_digest: [u8; 32],
    initial_student_digest: [u8; 32],
    artifact_growth: &GrowthReceipt,
) -> anyhow::Result<CampaignArtifactReport> {
    if trainer.completed_step() != campaign.steps {
        bail!(
            "cannot finalize artifact at step {}; configured terminal step is {}",
            trainer.completed_step(),
            campaign.steps
        );
    }
    let provenance = ArtifactProvenance::new(
        &plan.fingerprint,
        source_model_digest,
        initial_student_digest,
        trainer.completed_step(),
    )
    .context("construct artifact provenance")?;
    let exists = campaign
        .artifact
        .try_exists()
        .with_context(|| format!("inspect campaign artifact {}", campaign.artifact.display()))?;
    let summary = if exists {
        verify_campaign_artifact(
            &campaign.artifact,
            raw_hf_config_json,
            model,
            trainer,
            campaign.salt_planes,
            provenance,
            artifact_growth,
        )
        .context("verify existing terminal campaign artifact")?
    } else {
        export_campaign_artifact(
            &campaign.artifact,
            raw_hf_config_json,
            model,
            trainer,
            campaign.salt_planes,
            provenance,
            artifact_growth,
        )
        .context("export terminal campaign artifact")?
    };
    Ok(artifact_report(&campaign.artifact, summary))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn evaluate_terminal_artifact(
    campaign: &CampaignConfig,
    artifact: &CampaignArtifactReport,
    plan: &PlanSidecar,
    source_model_digest: [u8; 32],
    initial_student_digest: [u8; 32],
    growth: &CampaignGrowthReceipt,
    evaluation_tokens: &[u32],
    source_score: TeacherForcedPerplexity,
    grown_fp_score: Option<TeacherForcedPerplexity>,
    source_parameter_count: u64,
    grown_parameter_count: u64,
    naive_salt_ptq_score: TeacherForcedPerplexity,
    naive_salt_ptq_model_identity: &str,
    evaluation_corpus_file_digest: [u8; 32],
    evaluation_corpus_digest: [u8; 32],
) -> anyhow::Result<CampaignEvaluationReport> {
    let file = File::open(&campaign.artifact)
        .with_context(|| format!("open campaign artifact {}", campaign.artifact.display()))?;
    if !file
        .metadata()
        .with_context(|| format!("inspect opened artifact {}", campaign.artifact.display()))?
        .is_file()
    {
        bail!("opened campaign artifact is not a regular file");
    }
    // SAFETY: the mapping retains this single opened file description. Any later
    // pathname replacement cannot change the inode used for hashing and evaluation.
    let mapped = unsafe { memmap2::Mmap::map(&file) }
        .with_context(|| format!("map campaign artifact {}", campaign.artifact.display()))?;
    let opened_package =
        MeasuredPackage::from_bytes(&mapped).context("measure the exact opened artifact handle")?;
    if opened_package.id().to_string() != artifact.package_id
        || opened_package.physical_bytes() != artifact.physical_bytes
    {
        bail!(
            "campaign artifact changed between deterministic verification and opening; refusing mixed package/evaluation evidence"
        );
    }
    let metadata = parse_training_salt_artifact_metadata(&mapped)
        .context("parse terminal artifact provenance")?;
    let expected_plan = parse_lower_hex_digest(&plan.fingerprint)?;
    if metadata.plan_fingerprint != expected_plan
        || metadata.source_model_digest != source_model_digest
        || metadata.initial_student_digest != initial_student_digest
        || metadata.completed_step != campaign.steps
        || metadata.salt_planes != campaign.salt_planes as u32
    {
        bail!("terminal artifact provenance does not match the immutable campaign plan");
    }
    if metadata.growth.algorithm != growth.algorithm
        || metadata.growth.old_width != growth.old_width
        || metadata.growth.new_width != growth.new_width
        || metadata.growth.seed != growth.seed
        || metadata.growth.source_indices != growth.source_indices
        || metadata.growth.replication_counts != growth.replication_counts
        || metadata.growth.split_denominator_log2 != growth.split_denominator_log2
        || metadata.growth.split_numerators != growth.split_numerators
    {
        bail!("terminal artifact growth receipt does not match the immutable campaign plan");
    }
    let artifact_evaluation_path = campaign
        .checkpoint_dir
        .join("artifact-evaluation-progress.json");
    let score = resumable_teacher_forced_score(
        &artifact_evaluation_path,
        "training-salt-artifact",
        &artifact.package_id,
        evaluation_tokens,
        campaign.seq_len,
        evaluation_corpus_file_digest,
        evaluation_corpus_digest,
        || {
            ModelRunner::from_training_salt_gguf(&mapped, Box::new(tritium_cpu::CpuBackend::new()))
                .context("freshly reload terminal artifact on the CPU reference backend")
        },
    )
    .context("score held-out evaluation corpus from the reloaded artifact")?;
    let evaluation = evaluation_report(
        campaign,
        artifact,
        &plan.initial_student_digest,
        naive_salt_ptq_model_identity,
        source_score,
        grown_fp_score,
        source_parameter_count,
        grown_parameter_count,
        naive_salt_ptq_score,
        score,
        evaluation_corpus_file_digest,
        evaluation_corpus_digest,
    )?;
    Ok(evaluation)
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TeacherForcedEvaluationProgress {
    schema_version: u32,
    evaluation_kind: String,
    model_identity: String,
    corpus_file_digest: String,
    corpus_digest: String,
    seq_len: usize,
    total_windows: u64,
    completed_windows: u64,
    token_count: u64,
    negative_log_likelihood_f64_bits: u64,
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn resumable_teacher_forced_score(
    progress_path: &Path,
    evaluation_kind: &str,
    model_identity: &str,
    tokens: &[u32],
    seq_len: usize,
    corpus_file_digest: [u8; 32],
    corpus_digest: [u8; 32],
    build_runner: impl FnOnce() -> anyhow::Result<ModelRunner>,
) -> anyhow::Result<TeacherForcedPerplexity> {
    let mut runner = None;
    let mut build_runner = Some(build_runner);
    resumable_teacher_forced_score_with(
        progress_path,
        evaluation_kind,
        model_identity,
        tokens,
        seq_len,
        corpus_file_digest,
        corpus_digest,
        |window| {
            if runner.is_none() {
                let build_runner = build_runner
                    .take()
                    .context("evaluation runner builder was already consumed")?;
                runner = Some(build_runner()?);
            }
            teacher_forced_perplexity_windows(
                runner.as_mut().expect("runner initialized above"),
                window,
                seq_len,
            )
            .context("score teacher-forced evaluation window")
        },
    )
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn resumable_teacher_forced_score_with(
    progress_path: &Path,
    evaluation_kind: &str,
    model_identity: &str,
    tokens: &[u32],
    seq_len: usize,
    corpus_file_digest: [u8; 32],
    corpus_digest: [u8; 32],
    mut score_window: impl FnMut(&[u32]) -> anyhow::Result<TeacherForcedPerplexity>,
) -> anyhow::Result<TeacherForcedPerplexity> {
    if evaluation_kind.is_empty() || model_identity.is_empty() || seq_len < 2 {
        bail!("resumable evaluation identity and window geometry must be non-empty");
    }
    if tokens.is_empty() || !tokens.len().is_multiple_of(seq_len) {
        bail!("resumable evaluation requires non-empty exact token windows");
    }
    let total_windows =
        u64::try_from(tokens.len() / seq_len).context("evaluation window count exceeds u64")?;
    let corpus_file_digest = hex_digest(corpus_file_digest);
    let corpus_digest = hex_digest(corpus_digest);
    let mut progress = match std::fs::read(progress_path) {
        Ok(bytes) => serde_json::from_slice::<TeacherForcedEvaluationProgress>(&bytes)
            .with_context(|| format!("parse evaluation progress {}", progress_path.display()))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            TeacherForcedEvaluationProgress {
                schema_version: 1,
                evaluation_kind: evaluation_kind.to_owned(),
                model_identity: model_identity.to_owned(),
                corpus_file_digest: corpus_file_digest.clone(),
                corpus_digest: corpus_digest.clone(),
                seq_len,
                total_windows,
                completed_windows: 0,
                token_count: 0,
                negative_log_likelihood_f64_bits: 0.0_f64.to_bits(),
            }
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read evaluation progress {}", progress_path.display()));
        }
    };
    if progress.schema_version != 1
        || progress.evaluation_kind != evaluation_kind
        || progress.model_identity != model_identity
        || progress.corpus_file_digest != corpus_file_digest
        || progress.corpus_digest != corpus_digest
        || progress.seq_len != seq_len
        || progress.total_windows != total_windows
    {
        bail!(
            "evaluation progress {} does not match the immutable model/corpus contract",
            progress_path.display()
        );
    }
    let scored_per_window = u64::try_from(seq_len - 1).context("sequence length exceeds u64")?;
    let expected_token_count = progress
        .completed_windows
        .checked_mul(scored_per_window)
        .context("evaluation progress token count overflow")?;
    let mut negative_log_likelihood = f64::from_bits(progress.negative_log_likelihood_f64_bits);
    if progress.completed_windows > total_windows
        || progress.token_count != expected_token_count
        || !negative_log_likelihood.is_finite()
        || negative_log_likelihood < 0.0
    {
        bail!(
            "evaluation progress {} has inconsistent accumulated evidence",
            progress_path.display()
        );
    }
    if progress.completed_windows < total_windows {
        for window_index in progress.completed_windows..total_windows {
            let start = usize::try_from(window_index)
                .context("evaluation window index exceeds usize")?
                .checked_mul(seq_len)
                .context("evaluation window offset overflow")?;
            let end = start
                .checked_add(seq_len)
                .context("evaluation window end overflow")?;
            let score = score_window(&tokens[start..end])
                .with_context(|| format!("score evaluation window {window_index}"))?;
            if score.window_count != 1
                || score.token_count != scored_per_window
                || !score.negative_log_likelihood.is_finite()
                || score.negative_log_likelihood < 0.0
                || !score.perplexity.is_finite()
                || score.perplexity <= 0.0
            {
                bail!("single-window evaluation returned inconsistent evidence");
            }
            negative_log_likelihood += score.negative_log_likelihood;
            if !negative_log_likelihood.is_finite() {
                bail!("evaluation negative log likelihood became non-finite");
            }
            progress.completed_windows = window_index
                .checked_add(1)
                .context("evaluation completed-window count overflow")?;
            progress.token_count = progress
                .completed_windows
                .checked_mul(scored_per_window)
                .context("evaluation token count overflow")?;
            progress.negative_log_likelihood_f64_bits = negative_log_likelihood.to_bits();
            atomic_write(progress_path, &serde_json::to_vec_pretty(&progress)?)?;
        }
    }
    if progress.token_count == 0 {
        bail!("completed evaluation progress has no scored tokens");
    }
    let mean = negative_log_likelihood / progress.token_count as f64;
    let perplexity = mean.exp();
    if !mean.is_finite() || !perplexity.is_finite() {
        bail!("completed evaluation progress has invalid perplexity evidence");
    }
    Ok(TeacherForcedPerplexity {
        perplexity,
        negative_log_likelihood,
        token_count: progress.token_count,
        window_count: progress.completed_windows,
    })
}

#[cfg(feature = "cuda")]
fn naive_salt_ptq_model_identity(
    initial_student_model_digest: &str,
    salt_planes: usize,
) -> anyhow::Result<String> {
    let initial_student_model_digest = parse_lower_hex_digest(initial_student_model_digest)?;
    let salt_planes = u64::try_from(salt_planes).context("SALT plane count exceeds u64")?;
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium-naive-salt-ptq-model-v1");
    hash.update(NAIVE_SALT_PTQ_METHOD.as_bytes());
    hash.update(&initial_student_model_digest);
    hash.update(&salt_planes.to_le_bytes());
    Ok(hex_digest(*hash.finalize().as_bytes()))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn resumable_naive_salt_ptq_score(
    progress_path: &Path,
    backend: &CudaBackend,
    model: &TiedSwiGluTrainingModel,
    model_identity: &str,
    tokens: &[u32],
    seq_len: usize,
    corpus_file_digest: [u8; 32],
    corpus_digest: [u8; 32],
    salt_planes: usize,
) -> anyhow::Result<TeacherForcedPerplexity> {
    let mut packed_weights = None;
    resumable_teacher_forced_score_with(
        progress_path,
        NAIVE_SALT_PTQ_EVALUATION_KIND,
        model_identity,
        tokens,
        seq_len,
        corpus_file_digest,
        corpus_digest,
        |window| {
            if packed_weights.is_none() {
                let mut weights = Vec::with_capacity(model.parameters().len());
                for (index, parameter) in model.parameters().iter().enumerate() {
                    weights.push(
                        DevicePackedSaltWeight::from_host(
                            backend,
                            &parameter.master,
                            parameter.rows,
                            parameter.cols,
                            salt_planes,
                        )
                        .with_context(|| format!("pack naive SALT PTQ parameter {index}"))?,
                    );
                }
                packed_weights = Some(weights);
            }
            score_packed_salt_window(
                backend,
                model,
                packed_weights
                    .as_ref()
                    .expect("packed PTQ weights initialized above"),
                window,
            )
        },
    )
}

#[cfg(feature = "cuda")]
fn score_packed_salt_window(
    backend: &CudaBackend,
    model: &TiedSwiGluTrainingModel,
    packed_weights: &[DevicePackedSaltWeight],
    tokens: &[u32],
) -> anyhow::Result<TeacherForcedPerplexity> {
    let tokens_i32: Vec<i32> = tokens
        .iter()
        .copied()
        .map(|token| i32::try_from(token).context("PTQ evaluation token exceeds i32"))
        .collect::<Result<_, _>>()?;
    let mut tape = DeviceTape::new_with_policies(
        backend,
        model.architecture().vocab,
        CheckpointPolicy::SqrtDepth(model.architecture().n_layers),
        PackedSaltComputePolicy::Exact,
    )
    .context("create exact packed PTQ evaluation tape")?;
    let forward = packed_device_forward(&mut tape, model, packed_weights, &tokens_i32)
        .context("run exact packed PTQ evaluation forward")?;
    let logits = tape
        .value(forward.logits)
        .context("download exact packed PTQ evaluation logits")?;
    teacher_forced_score_from_full_logits(tokens, model.architecture().vocab, &logits)
}

#[cfg(feature = "cuda")]
fn teacher_forced_score_from_full_logits(
    tokens: &[u32],
    vocab: usize,
    logits: &[f32],
) -> anyhow::Result<TeacherForcedPerplexity> {
    if tokens.len() < 2 || vocab == 0 {
        bail!("PTQ evaluation window requires at least two tokens and a non-zero vocabulary");
    }
    let expected_logits = tokens
        .len()
        .checked_mul(vocab)
        .context("PTQ evaluation logit geometry overflow")?;
    if logits.len() != expected_logits {
        bail!(
            "PTQ evaluation returned {} logits, expected {expected_logits}",
            logits.len()
        );
    }
    let mut negative_log_likelihood = 0.0_f64;
    for position in 0..tokens.len() - 1 {
        let target =
            usize::try_from(tokens[position + 1]).context("PTQ evaluation target exceeds usize")?;
        if target >= vocab {
            bail!("PTQ evaluation target {target} is outside vocabulary 0..{vocab}");
        }
        let row_start = position
            .checked_mul(vocab)
            .context("PTQ evaluation row offset overflow")?;
        let row = &logits[row_start..row_start + vocab];
        if row.iter().any(|value| !value.is_finite()) {
            bail!("PTQ evaluation logits contain a non-finite value");
        }
        let max = row
            .iter()
            .copied()
            .map(f64::from)
            .fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row
            .iter()
            .map(|&value| (f64::from(value) - max).exp())
            .sum();
        let loss = max + sum_exp.ln() - f64::from(row[target]);
        if !loss.is_finite() || loss < 0.0 {
            bail!("PTQ evaluation log probability is invalid");
        }
        negative_log_likelihood += loss;
        if !negative_log_likelihood.is_finite() {
            bail!("PTQ evaluation negative log likelihood became non-finite");
        }
    }
    let token_count = u64::try_from(tokens.len() - 1).context("PTQ token count exceeds u64")?;
    let perplexity = (negative_log_likelihood / token_count as f64).exp();
    if !perplexity.is_finite() {
        bail!("PTQ evaluation perplexity is non-finite");
    }
    Ok(TeacherForcedPerplexity {
        perplexity,
        negative_log_likelihood,
        token_count,
        window_count: 1,
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn evaluation_report(
    campaign: &CampaignConfig,
    artifact: &CampaignArtifactReport,
    grown_fp_model_identity: &str,
    naive_salt_ptq_model_identity: &str,
    source_score: TeacherForcedPerplexity,
    grown_fp_score: Option<TeacherForcedPerplexity>,
    source_parameter_count: u64,
    grown_parameter_count: u64,
    naive_salt_ptq_score: TeacherForcedPerplexity,
    score: TeacherForcedPerplexity,
    corpus_file_digest: [u8; 32],
    corpus_digest: [u8; 32],
) -> anyhow::Result<CampaignEvaluationReport> {
    if source_score.window_count != score.window_count
        || source_score.token_count != score.token_count
        || naive_salt_ptq_score.window_count != score.window_count
        || naive_salt_ptq_score.token_count != score.token_count
        || grown_fp_score.is_some_and(|grown| {
            grown.window_count != score.window_count || grown.token_count != score.token_count
        })
    {
        bail!("source, grown-fp, naive PTQ, and artifact evaluation coverage differ");
    }
    if naive_salt_ptq_model_identity.is_empty() {
        bail!("naive SALT PTQ model identity is empty");
    }
    let grown_fp = match grown_fp_score {
        Some(grown) => {
            if grown_fp_model_identity.is_empty()
                || source_parameter_count == 0
                || grown_parameter_count <= source_parameter_count
            {
                bail!("grown-fp evidence has invalid identity or parameter geometry");
            }
            Some(CampaignGrownFpReport {
                model_identity: grown_fp_model_identity.to_owned(),
                source_parameter_count,
                grown_parameter_count,
                negative_log_likelihood: grown.negative_log_likelihood,
                perplexity: grown.perplexity,
                relative_perplexity_delta_percent: (grown.perplexity / source_score.perplexity
                    - 1.0)
                    * 100.0,
            })
        }
        None => {
            if grown_parameter_count != source_parameter_count {
                bail!("missing grown-fp score for a widened parameter geometry");
            }
            None
        }
    };
    Ok(CampaignEvaluationReport {
        artifact_package_id: artifact.package_id.clone(),
        corpus_path: path_identity(&campaign.evaluation_corpus)?
            .display()
            .to_string(),
        corpus_file_digest: hex_digest(corpus_file_digest),
        corpus_digest: hex_digest(corpus_digest),
        seq_len: campaign.seq_len,
        window_count: score.window_count,
        token_count: score.token_count,
        source_fp_negative_log_likelihood: source_score.negative_log_likelihood,
        source_fp_perplexity: source_score.perplexity,
        naive_salt_ptq: CampaignPtqBaselineReport {
            method: NAIVE_SALT_PTQ_METHOD.into(),
            model_identity: naive_salt_ptq_model_identity.into(),
            salt_planes: campaign.salt_planes,
            negative_log_likelihood: naive_salt_ptq_score.negative_log_likelihood,
            perplexity: naive_salt_ptq_score.perplexity,
        },
        negative_log_likelihood: score.negative_log_likelihood,
        perplexity: score.perplexity,
        relative_perplexity_delta_percent: (score.perplexity / source_score.perplexity - 1.0)
            * 100.0,
        recovery_vs_naive_salt_ptq: naive_salt_ptq_score.perplexity / score.perplexity,
        grown_fp,
    })
}

#[cfg(feature = "cuda")]
fn parse_lower_hex_digest(value: &str) -> anyhow::Result<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("campaign plan fingerprint is not lowercase 64-hex");
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let pair = core::str::from_utf8(pair).expect("validated ASCII hex");
        digest[index] = u8::from_str_radix(pair, 16).expect("validated lowercase hex");
    }
    Ok(digest)
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Deserialize, Serialize)]
struct RankStepEvidence {
    rank: usize,
    window_index: u64,
    training_elapsed_ms: f64,
    nvml_used_memory_bytes: u64,
    cuda_memory: CampaignCudaMemorySnapshot,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Deserialize, Serialize)]
struct StepTiming {
    step: u64,
    window_index: u64,
    warmup: bool,
    training_elapsed_ms_excluding_checkpoint: f64,
    rank_evidence: Vec<RankStepEvidence>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
struct CampaignInputHashes {
    campaign_config_file: String,
    model_config_file: String,
    source_model: String,
    initial_student_model: String,
    corpus_file: String,
    corpus: String,
    teacher_cache: String,
    evaluation_corpus_file: String,
    evaluation_corpus: String,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
struct CampaignCudaMemorySnapshot {
    device_free_bytes: u64,
    device_total_bytes: u64,
    device_used_sample_bytes: u64,
    pool_used_current_bytes: u64,
    pool_used_high_water_bytes: u64,
    pool_reserved_current_bytes: u64,
    pool_reserved_high_water_bytes: u64,
}

#[cfg(feature = "cuda")]
impl From<CudaMemorySnapshot> for CampaignCudaMemorySnapshot {
    fn from(snapshot: CudaMemorySnapshot) -> Self {
        Self {
            device_free_bytes: snapshot.device_free_bytes,
            device_total_bytes: snapshot.device_total_bytes,
            device_used_sample_bytes: snapshot.device_used_sample_bytes,
            pool_used_current_bytes: snapshot.pool_used_current_bytes,
            pool_used_high_water_bytes: snapshot.pool_used_high_water_bytes,
            pool_reserved_current_bytes: snapshot.pool_reserved_current_bytes,
            pool_reserved_high_water_bytes: snapshot.pool_reserved_high_water_bytes,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
struct CampaignCudaMemorySegment {
    resumed_from_step: u64,
    completed_step: u64,
    nvml_used_memory_bytes_at_launch: u64,
    nvml_used_memory_latest_sample_bytes: u64,
    nvml_used_memory_sample_high_water_bytes: u64,
    baseline: CampaignCudaMemorySnapshot,
    post_setup: Option<CampaignCudaMemorySnapshot>,
    latest_or_terminal: CampaignCudaMemorySnapshot,
}

#[cfg(feature = "cuda")]
impl CampaignCudaMemorySegment {
    fn observe_nvml(&mut self, used_memory_bytes: u64) {
        self.nvml_used_memory_latest_sample_bytes = used_memory_bytes;
        self.nvml_used_memory_sample_high_water_bytes = self
            .nvml_used_memory_sample_high_water_bytes
            .max(used_memory_bytes);
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
struct CampaignArtifactReport {
    path: String,
    package_id: String,
    physical_bytes: u64,
    scale_max_abs_error: f64,
    reconstruction_max_abs_delta: f64,
    reconstruction_squared_error_sum: f64,
    reconstruction_element_count: u64,
    reconstruction_mse: f64,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CampaignPtqBaselineReport {
    method: String,
    model_identity: String,
    salt_planes: usize,
    negative_log_likelihood: f64,
    perplexity: f64,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CampaignGrownFpReport {
    model_identity: String,
    source_parameter_count: u64,
    grown_parameter_count: u64,
    negative_log_likelihood: f64,
    perplexity: f64,
    relative_perplexity_delta_percent: f64,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CampaignEvaluationReport {
    artifact_package_id: String,
    corpus_path: String,
    corpus_file_digest: String,
    corpus_digest: String,
    seq_len: usize,
    window_count: u64,
    token_count: u64,
    source_fp_negative_log_likelihood: f64,
    source_fp_perplexity: f64,
    naive_salt_ptq: CampaignPtqBaselineReport,
    negative_log_likelihood: f64,
    perplexity: f64,
    relative_perplexity_delta_percent: f64,
    recovery_vs_naive_salt_ptq: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    grown_fp: Option<CampaignGrownFpReport>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Deserialize, Serialize)]
struct CampaignMemoryReport {
    packed_parameter_bytes: usize,
    packed_code_bytes: usize,
    packed_scale_bytes: usize,
    dense_parameter_bytes: usize,
    /// Complete master plus Adam moments held by one rank.
    host_optimizer_bytes: usize,
    /// Physical host optimizer allocation across all replicated ranks.
    fleet_host_optimizer_bytes: usize,
    host_adapter_master_bytes: usize,
    logical_host_training_state_bytes: usize,
    resident_training_state_bytes: usize,
    /// Fraction of persistent master plus Adam moment bytes resident on CUDA.
    optimizer_state_device_fraction: f64,
    peak_optimizer_staging_bytes: usize,
    /// Sum of the per-rank peak optimizer staging allocation.
    fleet_peak_optimizer_staging_bytes: usize,
    peak_live_requested_gradient_bytes: usize,
    materialized_gradient_bytes: usize,
    logical_peak_activation_bytes: usize,
    logical_naive_activation_bytes: usize,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Deserialize, Serialize)]
struct CampaignReport {
    schema_version: u32,
    plan_fingerprint: String,
    state_policy: CampaignStatePolicy,
    packed_compute: CampaignPackedComputePolicy,
    world_size: usize,
    window_partition: String,
    hardware: Vec<CampaignHardwareIdentity>,
    input_hashes: CampaignInputHashes,
    completed_step: u64,
    resumed_from_step: u64,
    configured_steps: u64,
    timing_warmup_steps: usize,
    checkpoint_path: String,
    step_timings: Vec<StepTiming>,
    memory: CampaignMemoryReport,
    cuda_memory_segments: Vec<CampaignCudaMemorySegment>,
    artifact: Option<CampaignArtifactReport>,
    evaluation: Option<CampaignEvaluationReport>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CampaignStaticMemory {
    packed_parameter_bytes: usize,
    packed_code_bytes: usize,
    packed_scale_bytes: usize,
    dense_parameter_bytes: usize,
    host_optimizer_bytes: usize,
    resident_training_state_bytes: usize,
    peak_optimizer_staging_bytes: usize,
    materialized_gradient_bytes: usize,
}

#[cfg(feature = "cuda")]
fn campaign_static_memory(
    policy: CampaignStatePolicy,
    host: HostOffloadMemoryGeometry,
) -> anyhow::Result<CampaignStaticMemory> {
    match policy {
        CampaignStatePolicy::HostOffload => Ok(CampaignStaticMemory {
            packed_parameter_bytes: host.packed_parameter_bytes,
            packed_code_bytes: host.packed_code_bytes,
            packed_scale_bytes: host.packed_scale_bytes,
            dense_parameter_bytes: host.dense_parameter_bytes,
            host_optimizer_bytes: host.host_optimizer_bytes,
            resident_training_state_bytes: 0,
            peak_optimizer_staging_bytes: host.peak_optimizer_staging_bytes,
            materialized_gradient_bytes: host.materialized_gradient_bytes,
        }),
        CampaignStatePolicy::Resident => Ok(CampaignStaticMemory {
            packed_parameter_bytes: host.packed_parameter_bytes,
            packed_code_bytes: host.packed_code_bytes,
            packed_scale_bytes: host.packed_scale_bytes,
            dense_parameter_bytes: host.dense_parameter_bytes,
            host_optimizer_bytes: 0,
            resident_training_state_bytes: host
                .host_optimizer_bytes
                .checked_add(host.largest_parameter_bytes)
                .context("resident training-state bytes overflow")?,
            peak_optimizer_staging_bytes: 0,
            materialized_gradient_bytes: host.materialized_gradient_bytes,
        }),
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct ReportExpectations<'a> {
    plan_fingerprint: &'a str,
    state_policy: CampaignStatePolicy,
    packed_compute: CampaignPackedComputePolicy,
    checkpoint_step: u64,
    checkpoint_dir: &'a Path,
    windows: u64,
    configured_steps: u64,
    timing_warmup_steps: usize,
    evaluation_seq_len: usize,
    evaluation_windows: u64,
    evaluation_corpus_path: &'a Path,
    hardware: &'a [CampaignHardwareIdentity],
    input_hashes: &'a CampaignInputHashes,
    static_memory: CampaignStaticMemory,
    salt_planes: usize,
    source_parameter_count: u64,
    grown_parameter_count: u64,
}

#[cfg(feature = "cuda")]
fn validate_report_memory(
    path: &Path,
    memory: &CampaignMemoryReport,
    expected: CampaignStaticMemory,
    completed_step: u64,
    world_size: usize,
    state_policy: CampaignStatePolicy,
) -> anyhow::Result<()> {
    let expected_fleet_host_optimizer_bytes = expected
        .host_optimizer_bytes
        .checked_mul(world_size)
        .context("expected fleet host optimizer bytes overflow")?;
    let expected_fleet_optimizer_staging_bytes = expected
        .peak_optimizer_staging_bytes
        .checked_mul(world_size)
        .context("expected fleet optimizer staging bytes overflow")?;
    for (label, actual, expected) in [
        (
            "packed code",
            memory.packed_code_bytes,
            expected.packed_code_bytes,
        ),
        (
            "packed scales",
            memory.packed_scale_bytes,
            expected.packed_scale_bytes,
        ),
        (
            "packed parameter",
            memory.packed_parameter_bytes,
            expected.packed_parameter_bytes,
        ),
        (
            "dense parameter",
            memory.dense_parameter_bytes,
            expected.dense_parameter_bytes,
        ),
        (
            "host optimizer",
            memory.host_optimizer_bytes,
            expected.host_optimizer_bytes,
        ),
        (
            "fleet host optimizer",
            memory.fleet_host_optimizer_bytes,
            expected_fleet_host_optimizer_bytes,
        ),
        ("host adapter master", memory.host_adapter_master_bytes, 0),
        (
            "logical host training-state",
            memory.logical_host_training_state_bytes,
            expected.host_optimizer_bytes,
        ),
        (
            "resident training-state",
            memory.resident_training_state_bytes,
            expected.resident_training_state_bytes,
        ),
        (
            "optimizer staging",
            memory.peak_optimizer_staging_bytes,
            expected.peak_optimizer_staging_bytes,
        ),
        (
            "fleet optimizer staging",
            memory.fleet_peak_optimizer_staging_bytes,
            expected_fleet_optimizer_staging_bytes,
        ),
    ] {
        if actual != expected {
            bail!(
                "prior report {} has {actual} {label} bytes, expected {expected}",
                path.display()
            );
        }
    }
    if completed_step > 0
        && memory.materialized_gradient_bytes != expected.materialized_gradient_bytes
    {
        bail!(
            "prior report {} has {} materialized gradient bytes, expected {} after completed training",
            path.display(),
            memory.materialized_gradient_bytes,
            expected.materialized_gradient_bytes
        );
    }
    let packed_parameter_bytes = memory
        .packed_code_bytes
        .checked_add(memory.packed_scale_bytes)
        .context("campaign report packed byte count overflow")?;
    if memory.packed_parameter_bytes != packed_parameter_bytes {
        bail!(
            "prior report {} has inconsistent packed parameter bytes",
            path.display()
        );
    }
    let logical_host_training_state_bytes = memory
        .host_optimizer_bytes
        .checked_add(memory.host_adapter_master_bytes)
        .context("campaign report host training-state byte count overflow")?;
    if memory.logical_host_training_state_bytes != logical_host_training_state_bytes {
        bail!(
            "prior report {} has inconsistent host training-state bytes",
            path.display()
        );
    }
    let expected_device_fraction = match state_policy {
        CampaignStatePolicy::Resident => 1.0_f64,
        CampaignStatePolicy::HostOffload => 0.0_f64,
    };
    if memory.optimizer_state_device_fraction.to_bits() != expected_device_fraction.to_bits() {
        bail!(
            "prior report {} has optimizer-state device fraction {}, expected {expected_device_fraction}",
            path.display(),
            memory.optimizer_state_device_fraction
        );
    }
    if memory.peak_live_requested_gradient_bytes > memory.materialized_gradient_bytes {
        bail!(
            "prior report {} has live requested-gradient peak larger than its materialized-gradient baseline",
            path.display()
        );
    }
    if memory.logical_peak_activation_bytes > memory.logical_naive_activation_bytes {
        bail!(
            "prior report {} has checkpointed activation peak larger than its naive activation baseline",
            path.display()
        );
    }
    for (label, bytes) in [
        ("packed scales", memory.packed_scale_bytes),
        ("dense parameters", memory.dense_parameter_bytes),
        ("host optimizer", memory.host_optimizer_bytes),
        ("fleet host optimizer", memory.fleet_host_optimizer_bytes),
        ("host adapter masters", memory.host_adapter_master_bytes),
        (
            "logical host training state",
            memory.logical_host_training_state_bytes,
        ),
        (
            "resident training state",
            memory.resident_training_state_bytes,
        ),
        ("optimizer staging", memory.peak_optimizer_staging_bytes),
        (
            "fleet optimizer staging",
            memory.fleet_peak_optimizer_staging_bytes,
        ),
        (
            "live requested gradients",
            memory.peak_live_requested_gradient_bytes,
        ),
        ("materialized gradients", memory.materialized_gradient_bytes),
        (
            "checkpointed activations",
            memory.logical_peak_activation_bytes,
        ),
        ("naive activations", memory.logical_naive_activation_bytes),
    ] {
        if !bytes.is_multiple_of(size_of::<f32>()) {
            bail!(
                "prior report {} has non-f32-aligned {label} byte count {bytes}",
                path.display()
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_report_timing_coverage(
    path: &Path,
    report: &CampaignReport,
    windows: u64,
    hardware: &[CampaignHardwareIdentity],
) -> anyhow::Result<()> {
    if windows == 0 {
        bail!("cannot validate campaign timings against zero windows");
    }
    if hardware.is_empty() {
        bail!("cannot validate campaign timings against an empty hardware fleet");
    }
    let world_size = u64::try_from(hardware.len()).context("hardware fleet exceeds u64")?;
    let timing_count = u64::try_from(report.step_timings.len())
        .context("campaign report timing count exceeds u64")?;
    if timing_count != report.completed_step {
        bail!(
            "prior report {} has {timing_count} timings but declares completed step {}; expected exact coverage of steps 1..={}",
            path.display(),
            report.completed_step,
            report.completed_step
        );
    }
    for (offset, timing) in report.step_timings.iter().enumerate() {
        let expected_step = u64::try_from(offset)
            .context("campaign report timing index exceeds u64")?
            .checked_add(1)
            .context("campaign report timing step overflow")?;
        if timing.step != expected_step {
            bail!(
                "prior report {} timing {} records step {}, expected ordered unique step {expected_step}",
                path.display(),
                offset,
                timing.step
            );
        }
        let expected_window_index = (expected_step - 1)
            .checked_mul(world_size)
            .context("campaign timing window cursor overflow")?
            % windows;
        if timing.window_index != expected_window_index {
            bail!(
                "prior report {} step {expected_step} records window {}, expected {expected_window_index}",
                path.display(),
                timing.window_index
            );
        }
        if !timing.training_elapsed_ms_excluding_checkpoint.is_finite()
            || timing.training_elapsed_ms_excluding_checkpoint < 0.0
        {
            bail!(
                "prior report {} step {expected_step} has invalid elapsed time {}",
                path.display(),
                timing.training_elapsed_ms_excluding_checkpoint
            );
        }
        if timing.rank_evidence.len() != hardware.len() {
            bail!(
                "prior report {} step {expected_step} has {} rank evidence records, expected {}",
                path.display(),
                timing.rank_evidence.len(),
                hardware.len()
            );
        }
        let mut max_elapsed = 0.0_f64;
        for (rank, (evidence, rank_hardware)) in
            timing.rank_evidence.iter().zip(hardware).enumerate()
        {
            if evidence.rank != rank {
                bail!(
                    "prior report {} step {expected_step} rank evidence {} identifies rank {}",
                    path.display(),
                    rank,
                    evidence.rank
                );
            }
            let rank_u64 = u64::try_from(rank).context("campaign rank exceeds u64")?;
            let expected_rank_window = expected_window_index
                .checked_add(rank_u64)
                .context("rank window index overflow")?
                % windows;
            if evidence.window_index != expected_rank_window {
                bail!(
                    "prior report {} step {expected_step} rank {rank} records window {}, expected {expected_rank_window}",
                    path.display(),
                    evidence.window_index
                );
            }
            if !evidence.training_elapsed_ms.is_finite() || evidence.training_elapsed_ms < 0.0 {
                bail!(
                    "prior report {} step {expected_step} rank {rank} has invalid elapsed time {}",
                    path.display(),
                    evidence.training_elapsed_ms
                );
            }
            max_elapsed = max_elapsed.max(evidence.training_elapsed_ms);
            if evidence.nvml_used_memory_bytes > rank_hardware.total_memory_bytes
                || evidence.cuda_memory.device_total_bytes == 0
                || evidence.cuda_memory.device_total_bytes > rank_hardware.total_memory_bytes
                || evidence
                    .cuda_memory
                    .device_free_bytes
                    .checked_add(evidence.cuda_memory.device_used_sample_bytes)
                    != Some(evidence.cuda_memory.device_total_bytes)
                || evidence.cuda_memory.pool_used_high_water_bytes
                    < evidence.cuda_memory.pool_used_current_bytes
                || evidence.cuda_memory.pool_reserved_high_water_bytes
                    < evidence.cuda_memory.pool_reserved_current_bytes
                || evidence.cuda_memory.pool_reserved_current_bytes
                    < evidence.cuda_memory.pool_used_current_bytes
                || evidence.cuda_memory.pool_reserved_high_water_bytes
                    < evidence.cuda_memory.pool_used_high_water_bytes
            {
                bail!(
                    "prior report {} step {expected_step} rank {rank} has invalid memory evidence",
                    path.display()
                );
            }
        }
        if timing.training_elapsed_ms_excluding_checkpoint.to_bits() != max_elapsed.to_bits() {
            bail!(
                "prior report {} step {expected_step} max-rank elapsed time is inconsistent",
                path.display()
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn validate_cuda_memory_segments(
    path: &Path,
    report: &CampaignReport,
    hardware: &CampaignHardwareIdentity,
    salt_planes: usize,
    evaluation_seq_len: usize,
    evaluation_windows: u64,
    evaluation_corpus_path: &Path,
    source_parameter_count: u64,
    grown_parameter_count: u64,
) -> anyhow::Result<()> {
    if report.cuda_memory_segments.is_empty() {
        bail!("prior report {} has no CUDA memory segment", path.display());
    }
    let mut prior_completed = 0_u64;
    for (index, segment) in report.cuda_memory_segments.iter().enumerate() {
        if segment.resumed_from_step != prior_completed {
            bail!(
                "prior report {} CUDA memory segment {index} resumes at {}, expected {prior_completed}",
                path.display(),
                segment.resumed_from_step
            );
        }
        if segment.completed_step < segment.resumed_from_step
            || segment.completed_step > report.completed_step
        {
            bail!(
                "prior report {} CUDA memory segment {index} has invalid step range {}..={}",
                path.display(),
                segment.resumed_from_step,
                segment.completed_step
            );
        }
        if segment.nvml_used_memory_bytes_at_launch > hardware.total_memory_bytes
            || segment.nvml_used_memory_latest_sample_bytes > hardware.total_memory_bytes
            || segment.nvml_used_memory_sample_high_water_bytes > hardware.total_memory_bytes
            || segment.nvml_used_memory_sample_high_water_bytes
                < segment.nvml_used_memory_bytes_at_launch
            || segment.nvml_used_memory_sample_high_water_bytes
                < segment.nvml_used_memory_latest_sample_bytes
        {
            bail!(
                "prior report {} CUDA memory segment {index} has inconsistent NVML samples",
                path.display()
            );
        }
        for (label, snapshot) in [
            ("baseline", Some(segment.baseline)),
            ("post-setup", segment.post_setup),
            ("latest", Some(segment.latest_or_terminal)),
        ] {
            let Some(snapshot) = snapshot else {
                continue;
            };
            if snapshot.device_total_bytes == 0
                || snapshot.device_total_bytes > hardware.total_memory_bytes
                || snapshot
                    .device_free_bytes
                    .checked_add(snapshot.device_used_sample_bytes)
                    != Some(snapshot.device_total_bytes)
                || snapshot.pool_used_high_water_bytes < snapshot.pool_used_current_bytes
                || snapshot.pool_reserved_high_water_bytes < snapshot.pool_reserved_current_bytes
                || snapshot.pool_reserved_current_bytes < snapshot.pool_used_current_bytes
                || snapshot.pool_reserved_high_water_bytes < snapshot.pool_used_high_water_bytes
            {
                bail!(
                    "prior report {} CUDA memory segment {index} has inconsistent {label} accounting",
                    path.display()
                );
            }
        }
        if segment.completed_step > segment.resumed_from_step && segment.post_setup.is_none() {
            bail!(
                "prior report {} CUDA memory segment {index} trained without a post-setup sample",
                path.display()
            );
        }
        prior_completed = segment.completed_step;
    }
    if prior_completed != report.completed_step {
        bail!(
            "prior report {} CUDA memory segments end at step {prior_completed}, expected {}",
            path.display(),
            report.completed_step
        );
    }
    let warmup_steps = u64::try_from(report.timing_warmup_steps)
        .context("campaign report timing warmup count exceeds u64")?;
    for timing in &report.step_timings {
        let segment = report
            .cuda_memory_segments
            .iter()
            .find(|segment| {
                timing.step > segment.resumed_from_step && timing.step <= segment.completed_step
            })
            .with_context(|| {
                format!(
                    "prior report {} timing step {} belongs to no CUDA memory segment",
                    path.display(),
                    timing.step
                )
            })?;
        let expected_warmup = timing.step - segment.resumed_from_step <= warmup_steps;
        if timing.warmup != expected_warmup {
            bail!(
                "prior report {} timing step {} has warmup={}, expected {expected_warmup}",
                path.display(),
                timing.step,
                timing.warmup
            );
        }
    }
    if report.artifact.is_some() != report.evaluation.is_some() {
        bail!(
            "prior report {} has only one of artifact and evaluation evidence",
            path.display()
        );
    }
    if report.completed_step < report.configured_steps
        && (report.artifact.is_some() || report.evaluation.is_some())
    {
        bail!(
            "prior report {} records terminal artifact evidence before its configured final step",
            path.display()
        );
    }
    if let (Some(artifact), Some(evaluation)) = (&report.artifact, &report.evaluation) {
        let scored_per_window = evaluation.seq_len.checked_sub(1).and_then(|count| {
            u64::try_from(count)
                .ok()?
                .checked_mul(evaluation.window_count)
        });
        let expected_ptq_identity =
            naive_salt_ptq_model_identity(&report.input_hashes.initial_student_model, salt_planes)
                .with_context(|| {
                    format!(
                        "prior report {} has invalid naive PTQ provenance",
                        path.display()
                    )
                })?;
        let expected_source_perplexity =
            (evaluation.source_fp_negative_log_likelihood / evaluation.token_count as f64).exp();
        let expected_ptq_perplexity = (evaluation.naive_salt_ptq.negative_log_likelihood
            / evaluation.token_count as f64)
            .exp();
        let expected_artifact_perplexity =
            (evaluation.negative_log_likelihood / evaluation.token_count as f64).exp();
        let expected_relative_delta =
            (evaluation.perplexity / evaluation.source_fp_perplexity - 1.0) * 100.0;
        let expected_recovery = evaluation.naive_salt_ptq.perplexity / evaluation.perplexity;
        let expects_growth = grown_parameter_count > source_parameter_count;
        let valid_grown_fp = match &evaluation.grown_fp {
            Some(grown) => {
                let expected_perplexity =
                    (grown.negative_log_likelihood / evaluation.token_count as f64).exp();
                let expected_relative_delta =
                    (grown.perplexity / evaluation.source_fp_perplexity - 1.0) * 100.0;
                expects_growth
                    && grown.model_identity == report.input_hashes.initial_student_model
                    && grown.source_parameter_count == source_parameter_count
                    && grown.grown_parameter_count == grown_parameter_count
                    && grown.negative_log_likelihood.is_finite()
                    && grown.negative_log_likelihood >= 0.0
                    && grown.perplexity.is_finite()
                    && grown.perplexity > 0.0
                    && grown.relative_perplexity_delta_percent.is_finite()
                    && expected_perplexity.to_bits() == grown.perplexity.to_bits()
                    && expected_relative_delta.to_bits()
                        == grown.relative_perplexity_delta_percent.to_bits()
            }
            None => !expects_growth && grown_parameter_count == source_parameter_count,
        };
        if artifact.package_id != evaluation.artifact_package_id
            || artifact.physical_bytes == 0
            || !artifact.scale_max_abs_error.is_finite()
            || !artifact.reconstruction_max_abs_delta.is_finite()
            || !artifact.reconstruction_squared_error_sum.is_finite()
            || artifact.reconstruction_element_count == 0
            || !artifact.reconstruction_mse.is_finite()
            || evaluation.corpus_file_digest != report.input_hashes.evaluation_corpus_file
            || evaluation.corpus_digest != report.input_hashes.evaluation_corpus
            || Path::new(&evaluation.corpus_path) != evaluation_corpus_path
            || evaluation.seq_len != evaluation_seq_len
            || evaluation.window_count != evaluation_windows
            || evaluation.naive_salt_ptq.method != NAIVE_SALT_PTQ_METHOD
            || evaluation.naive_salt_ptq.model_identity != expected_ptq_identity
            || evaluation.naive_salt_ptq.salt_planes != salt_planes
            || !evaluation.source_fp_negative_log_likelihood.is_finite()
            || evaluation.source_fp_negative_log_likelihood < 0.0
            || !evaluation.source_fp_perplexity.is_finite()
            || evaluation.source_fp_perplexity <= 0.0
            || !evaluation
                .naive_salt_ptq
                .negative_log_likelihood
                .is_finite()
            || evaluation.naive_salt_ptq.negative_log_likelihood < 0.0
            || !evaluation.naive_salt_ptq.perplexity.is_finite()
            || evaluation.naive_salt_ptq.perplexity <= 0.0
            || !evaluation.negative_log_likelihood.is_finite()
            || evaluation.negative_log_likelihood < 0.0
            || !evaluation.perplexity.is_finite()
            || evaluation.perplexity <= 0.0
            || !evaluation.relative_perplexity_delta_percent.is_finite()
            || !evaluation.recovery_vs_naive_salt_ptq.is_finite()
            || evaluation.recovery_vs_naive_salt_ptq <= 0.0
            || evaluation.token_count == 0
            || evaluation.window_count == 0
            || scored_per_window != Some(evaluation.token_count)
            || expected_source_perplexity.to_bits() != evaluation.source_fp_perplexity.to_bits()
            || expected_ptq_perplexity.to_bits() != evaluation.naive_salt_ptq.perplexity.to_bits()
            || expected_artifact_perplexity.to_bits() != evaluation.perplexity.to_bits()
            || expected_relative_delta.to_bits()
                != evaluation.relative_perplexity_delta_percent.to_bits()
            || expected_recovery.to_bits() != evaluation.recovery_vs_naive_salt_ptq.to_bits()
            || !valid_grown_fp
        {
            bail!(
                "prior report {} has inconsistent artifact/evaluation evidence",
                path.display()
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn load_existing_report(
    path: &Path,
    expected: &ReportExpectations<'_>,
) -> anyhow::Result<Option<CampaignReport>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error)
            if error.kind() == std::io::ErrorKind::NotFound && expected.checkpoint_step == 0 =>
        {
            return Ok(None);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            bail!(
                "committed checkpoint step {} has no matching campaign report {}; refusing resume with incomplete evidence",
                expected.checkpoint_step,
                path.display()
            );
        }
        Err(error) => {
            return Err(error).with_context(|| format!("read prior report {}", path.display()));
        }
    };
    let mut value: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse prior report {}", path.display()))?;
    let schema_version = value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .with_context(|| {
            format!(
                "prior report {} has a missing or invalid schema version",
                path.display()
            )
        })?;
    match schema_version {
        6 => {}
        5 => {
            let object = value
                .as_object_mut()
                .with_context(|| format!("prior report {} is not a JSON object", path.display()))?;
            let has_artifact = object
                .get("artifact")
                .is_some_and(|artifact| !artifact.is_null());
            let has_evaluation = object
                .get("evaluation")
                .is_some_and(|evaluation| !evaluation.is_null());
            ensure!(
                has_artifact == has_evaluation,
                "prior schema-5 report {} has unpaired artifact/evaluation evidence",
                path.display()
            );
            // Schema 6 adds independently persisted naive-PTQ evidence. Preserve and
            // validate all common checkpoint/timing/memory fields, but discard legacy
            // terminal evidence: the terminal path deterministically rebuilds and
            // re-evaluates the artifact against the persisted PTQ accumulator.
            object.insert("schema_version".into(), serde_json::Value::from(6));
            object.insert("artifact".into(), serde_json::Value::Null);
            object.insert("evaluation".into(), serde_json::Value::Null);
        }
        unsupported => {
            bail!(
                "prior report {} uses unsupported schema version {unsupported}",
                path.display()
            );
        }
    }
    let mut report: CampaignReport = serde_json::from_value(value)
        .with_context(|| format!("decode prior report {}", path.display()))?;
    if report.plan_fingerprint != expected.plan_fingerprint {
        bail!(
            "prior report {} belongs to a different immutable campaign plan",
            path.display()
        );
    }
    if report.state_policy != expected.state_policy
        || report.packed_compute != expected.packed_compute
    {
        bail!(
            "prior report {} records a different training or packed-compute policy",
            path.display()
        );
    }
    if report.world_size != expected.hardware.len()
        || report.window_partition != "contiguous-modulo-v1"
    {
        bail!(
            "prior report {} records a different distributed topology",
            path.display()
        );
    }
    if report.hardware != expected.hardware {
        bail!(
            "prior report {} records different stable CUDA/NVML hardware identity",
            path.display(),
        );
    }
    if report.configured_steps != expected.configured_steps {
        bail!(
            "prior report {} records {} configured steps, expected {}",
            path.display(),
            report.configured_steps,
            expected.configured_steps
        );
    }
    if report.timing_warmup_steps != expected.timing_warmup_steps {
        bail!(
            "prior report {} records {} timing warmup steps, expected {}",
            path.display(),
            report.timing_warmup_steps,
            expected.timing_warmup_steps
        );
    }
    if report.input_hashes != *expected.input_hashes {
        bail!(
            "prior report {} input hashes do not match the current campaign inputs",
            path.display()
        );
    }
    let reported_checkpoint = path_identity(Path::new(&report.checkpoint_path))?;
    let expected_checkpoint = path_identity(expected.checkpoint_dir)?;
    if reported_checkpoint != expected_checkpoint {
        bail!(
            "prior report {} points at checkpoint {}, expected {}",
            path.display(),
            report.checkpoint_path,
            expected.checkpoint_dir.display()
        );
    }
    if report.completed_step > report.configured_steps {
        bail!(
            "prior report {} completed step {} exceeds configured step {}",
            path.display(),
            report.completed_step,
            report.configured_steps
        );
    }
    if report.resumed_from_step > report.completed_step {
        bail!(
            "prior report {} resumed from step {} after its completed step {}",
            path.display(),
            report.resumed_from_step,
            report.completed_step
        );
    }
    // Validate the full persisted report before either a terminal shortcut or
    // truncating evidence that ran ahead of DCP's manifest commit.
    validate_report_memory(
        path,
        &report.memory,
        expected.static_memory,
        report.completed_step,
        expected.hardware.len(),
        expected.state_policy,
    )?;
    validate_report_timing_coverage(path, &report, expected.windows, expected.hardware)?;
    let owner_hardware = expected
        .hardware
        .first()
        .context("campaign hardware fleet is empty")?;
    validate_cuda_memory_segments(
        path,
        &report,
        owner_hardware,
        expected.salt_planes,
        expected.evaluation_seq_len,
        expected.evaluation_windows,
        expected.evaluation_corpus_path,
        expected.source_parameter_count,
        expected.grown_parameter_count,
    )?;
    if report.completed_step < expected.checkpoint_step {
        bail!(
            "prior report {} covers completed step {}, but committed checkpoint is step {}; refusing resume with an evidence gap",
            path.display(),
            report.completed_step,
            expected.checkpoint_step
        );
    }
    if report.completed_step > expected.checkpoint_step {
        // Reports precede DCP manifest commits. A crash can leave evidence for
        // optimizer work whose state did not commit; replay only the timings
        // covered by the live manifest. Peak-memory values remain conservative.
        report
            .step_timings
            .retain(|timing| timing.step <= expected.checkpoint_step);
        report
            .cuda_memory_segments
            .retain(|segment| segment.resumed_from_step <= expected.checkpoint_step);
        if let Some(segment) = report.cuda_memory_segments.last_mut() {
            segment.completed_step = expected.checkpoint_step;
        }
        report.completed_step = expected.checkpoint_step;
        report.artifact = None;
        report.evaluation = None;
    }
    Ok(Some(report))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use tritium_nn::{DenseLinear, Mlp, Projection, SwiGluMlp, TransformerBlock};
    use tritium_nn::{TiedSwiGluTrainingArchitecture, TrainingParameter};

    fn temp_path(name: &str) -> PathBuf {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "tritium-campaign-{name}-{}-{sequence}",
            std::process::id()
        ))
    }

    fn test_hardware() -> CampaignHardwareIdentity {
        CampaignHardwareIdentity {
            cuda_ordinal: 0,
            device_name: "NVIDIA test GPU".into(),
            pci_bus_id: "0000:01:00.0".into(),
            cuda_driver_version: 13_010,
            nvidia_driver_version: "610.1".into(),
            nvml_cuda_driver_version: 13_010,
            total_memory_bytes: 100,
        }
    }

    fn test_growth() -> CampaignGrowthReceipt {
        CampaignGrowthReceipt {
            algorithm: "net2wider.intermediate-swiglu.splitmix64.v1".into(),
            old_width: 3,
            new_width: 3,
            seed: 0,
            source_indices: vec![0, 1, 2],
            replication_counts: vec![1, 1, 1],
            split_denominator_log2: None,
            split_numerators: None,
        }
    }

    #[test]
    fn identity_growth_receipt_keeps_canonical_v1_json() {
        assert_eq!(
            serde_json::to_string(&test_growth()).unwrap(),
            r#"{"algorithm":"net2wider.intermediate-swiglu.splitmix64.v1","old_width":3,"new_width":3,"seed":0,"source_indices":[0,1,2],"replication_counts":[1,1,1]}"#
        );
    }

    #[cfg(feature = "cuda")]
    fn test_memory_snapshot() -> CampaignCudaMemorySnapshot {
        CampaignCudaMemorySnapshot {
            device_free_bytes: 80,
            device_total_bytes: 100,
            device_used_sample_bytes: 20,
            pool_used_current_bytes: 4,
            pool_used_high_water_bytes: 8,
            pool_reserved_current_bytes: 8,
            pool_reserved_high_water_bytes: 12,
        }
    }

    #[cfg(feature = "cuda")]
    fn evaluation_test_runner() -> ModelRunner {
        let config = ModelConfig {
            arch: "llama".into(),
            n_layers: 0,
            n_embd: 2,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 2,
            n_ff: 2,
            n_ctx: 8,
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
        };
        let weights = ModelWeights {
            token_embd: tritium_nn::TokenEmbedding::from_dense(
                vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.5],
                3,
                2,
            )
            .unwrap(),
            vocab: 3,
            n_embd: 2,
            layers: Vec::new(),
            output_norm: vec![1.0, 1.0],
            lm_head: None,
        };
        ModelRunner::from_weights(config, weights, Box::new(tritium_cpu::CpuBackend::new()))
    }

    #[cfg(feature = "cuda")]
    fn resident_campaign_test_model() -> TiedSwiGluTrainingModel {
        let config = ModelConfig {
            arch: "llama".into(),
            n_layers: 1,
            n_embd: 4,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 2,
            n_ff: 6,
            n_ctx: 8,
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
        };
        let spec = ArchSpec {
            mlp: MlpKind::SwiGlu,
            attn_sub_norm: false,
            ffn_sub_norm: false,
            qk_norm: false,
            qkv_bias: false,
            tied_embeddings: true,
        };
        let dense = |rows: usize, cols: usize, seed: usize| {
            let values = (0..rows * cols)
                .map(|index| ((index * 7 + seed * 5) % 23) as f32 / 32.0 - 0.35)
                .collect();
            Projection::Dense(DenseLinear::new_exact(values, rows, cols).unwrap())
        };
        let weights = ModelWeights {
            token_embd: tritium_nn::TokenEmbedding::from_dense(
                (0..32).map(|index| index as f32 / 64.0 - 0.2).collect(),
                8,
                4,
            )
            .unwrap(),
            vocab: 8,
            n_embd: 4,
            layers: vec![TransformerBlock {
                attn_norm: vec![1.0; 4],
                q_proj: dense(4, 4, 1),
                k_proj: dense(2, 4, 2),
                v_proj: dense(2, 4, 3),
                o_proj: dense(4, 4, 4),
                attn_sub_norm: Vec::new(),
                q_bias: Vec::new(),
                k_bias: Vec::new(),
                v_bias: Vec::new(),
                q_norm: Vec::new(),
                k_norm: Vec::new(),
                ffn_norm: vec![1.0; 4],
                mlp: Mlp::SwiGlu(SwiGluMlp {
                    gate: dense(6, 4, 5),
                    up: dense(6, 4, 6),
                    down: dense(4, 6, 7),
                }),
            }],
            output_norm: vec![1.0; 4],
            lm_head: None,
        };
        TiedSwiGluTrainingModel::extract(&config, &spec, &weights).unwrap()
    }

    fn fixture() -> (
        ModelConfig,
        ArchSpec,
        TiedSwiGluTrainingArchitecture,
        Vec<TrainingParameter>,
    ) {
        let config = ModelConfig {
            arch: "llama".into(),
            n_layers: 1,
            n_embd: 2,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 2,
            n_ff: 3,
            n_ctx: 8,
            rope_theta: 10_000.0,
            rms_eps: 1e-5,
        };
        let spec = ArchSpec {
            mlp: MlpKind::SwiGlu,
            attn_sub_norm: false,
            ffn_sub_norm: false,
            qk_norm: false,
            qkv_bias: false,
            tied_embeddings: true,
        };
        let architecture = TiedSwiGluTrainingArchitecture {
            attn_norms: vec![vec![1.0, 1.1]],
            ffn_norms: vec![vec![0.9, 1.2]],
            attention_constants: vec![tritium_nn::TrainingAttentionConstants::default()],
            output_norm: vec![1.0, 0.8],
            n_embd: 2,
            n_head: 1,
            n_head_kv: 1,
            head_dim: 2,
            n_ff: 3,
            vocab: 2,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            n_layers: 1,
            n_ctx: 8,
        };
        let parameters = vec![TrainingParameter {
            name: "model.embed_tokens.weight".into(),
            master: vec![0.1, 0.2, 0.3, 0.4],
            rows: 2,
            cols: 2,
        }];
        (config, spec, architecture, parameters)
    }

    #[test]
    fn semantic_model_digest_is_stable_and_covers_norms_names_shapes_and_values() {
        let (config, spec, architecture, parameters) = fixture();
        let first = semantic_model_digest_parts(&config, &spec, &architecture, &parameters);
        assert_eq!(
            first,
            semantic_model_digest_parts(&config, &spec, &architecture, &parameters)
        );

        let mut changed = architecture.clone();
        changed.output_norm[0] = f32::from_bits(changed.output_norm[0].to_bits() + 1);
        assert_ne!(
            first,
            semantic_model_digest_parts(&config, &spec, &changed, &parameters)
        );
        let mut changed = architecture.clone();
        changed.attention_constants[0].q_bias = vec![0.125, -0.25];
        assert_ne!(
            first,
            semantic_model_digest_parts(&config, &spec, &changed, &parameters)
        );
        let mut changed = parameters.clone();
        changed[0].name.push_str(".changed");
        assert_ne!(
            first,
            semantic_model_digest_parts(&config, &spec, &architecture, &changed)
        );
        let mut changed = parameters.clone();
        changed[0].rows = 1;
        changed[0].cols = 4;
        assert_ne!(
            first,
            semantic_model_digest_parts(&config, &spec, &architecture, &changed)
        );
        let mut changed = parameters.clone();
        changed[0].master[0] = 9.0;
        assert_ne!(
            first,
            semantic_model_digest_parts(&config, &spec, &architecture, &changed)
        );
    }

    #[test]
    fn held_out_corpus_rejects_semantic_training_content() {
        assert!(
            validate_held_out_corpus(&[1, 2, 3, 4], [1; 32], &[1, 2, 3, 4], [1; 32], 2).is_err()
        );
        assert!(
            validate_held_out_corpus(&[1, 2, 3, 4], [1; 32], &[8, 9, 3, 4], [2; 32], 2).is_err()
        );
        validate_held_out_corpus(&[1, 2, 3, 4], [1; 32], &[8, 9, 10, 11], [2; 32], 2).unwrap();
    }

    #[test]
    fn teacher_row_softmax_normalizes_and_rejects_non_finite_logits() {
        let mut row = [1.0, 2.0, 3.0];
        row_softmax_in_place(&mut row).unwrap();
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(row[0] < row[1] && row[1] < row[2]);

        assert!(row_softmax_in_place(&mut [f32::NAN]).is_err());
    }

    #[test]
    fn teacher_cache_paths_reject_input_aliases() {
        let directory = temp_path("teacher-path-collisions");
        let model_dir = directory.join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        let corpus = directory.join("corpus.json");
        std::fs::write(&corpus, b"[]").unwrap();

        validate_teacher_cache_paths(&model_dir, &corpus, &directory.join("teacher.ttpr")).unwrap();
        assert!(validate_teacher_cache_paths(&model_dir, &corpus, &corpus).is_err());
        assert!(
            validate_teacher_cache_paths(&model_dir, &corpus, &model_dir.join("teacher.ttpr"))
                .is_err()
        );
        let output_directory = directory.join("output-directory");
        std::fs::create_dir_all(&output_directory).unwrap();
        assert!(validate_teacher_cache_paths(&model_dir, &corpus, &output_directory).is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let regular = directory.join("regular-output-target");
            let special = directory.join("special-output");
            std::fs::write(&regular, b"old").unwrap();
            symlink(&regular, &special).unwrap();
            assert!(validate_teacher_cache_paths(&model_dir, &corpus, &special).is_err());
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn path_identity_resolves_parent_components_after_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = temp_path("symlink-parent-alias");
        let target_parent = directory.join("target-parent");
        let target = target_parent.join("target");
        let model_dir = directory.join("model");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::create_dir_all(&model_dir).unwrap();
        let link = directory.join("link");
        symlink(&target, &link).unwrap();
        let corpus = target_parent.join("corpus.json");
        std::fs::write(&corpus, b"[]").unwrap();
        let alias = link.join("..").join("corpus.json");

        assert_eq!(
            path_identity(&alias).unwrap(),
            path_identity(&corpus).unwrap()
        );
        assert!(validate_teacher_cache_paths(&model_dir, &corpus, &alias).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn campaign_run_fails_closed_without_cuda_feature() {
        let error = run_campaign(Path::new("campaign.json")).unwrap_err();
        assert!(error.to_string().contains("--features cuda"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn campaign_config_requires_checkpoint_cadence_and_resolves_paths() {
        let mut config: CampaignConfig = serde_json::from_str(
            r#"{
                "model_dir":"model", "corpus":"corpus.json",
                "evaluation_corpus":"evaluation.json",
                "teacher_cache":"teacher.ttpr", "checkpoint_dir":"checkpoints",
                "report":"report.json", "artifact":"model.gguf",
                "seq_len":32, "steps":40,
                "checkpoint_every":5
            }"#,
        )
        .unwrap();
        config.resolve_paths(Path::new("/campaign/config.json"));
        assert_eq!(config.model_dir, Path::new("/campaign/model"));
        assert_eq!(config.report, Path::new("/campaign/report.json"));
        assert_eq!(config.artifact, Path::new("/campaign/model.gguf"));
        assert_eq!(
            config.evaluation_corpus,
            Path::new("/campaign/evaluation.json")
        );
        assert_eq!(config.growth, None);
        assert_eq!(config.cuda_device, None);
        assert_eq!(config.distributed, None);
        assert_eq!(config.salt_planes, 2);
        assert_eq!(config.state_policy, CampaignStatePolicy::Resident);
        assert_eq!(config.packed_compute, CampaignPackedComputePolicy::Exact);
        assert_eq!(config.checkpoint_every, 5);
        assert_eq!(config.timing_warmup_steps, 1);
        assert_eq!(config.checkpoint_shards, 1);
        assert_eq!(config.adam.lr.to_bits(), default_lr().to_bits());

        let missing_cadence = r#"{
            "model_dir":"model", "corpus":"corpus.json",
            "evaluation_corpus":"evaluation.json",
            "teacher_cache":"teacher.ttpr", "checkpoint_dir":"checkpoints",
            "report":"report.json", "artifact":"model.gguf",
            "seq_len":32, "steps":40
        }"#;
        assert!(serde_json::from_str::<CampaignConfig>(missing_cadence).is_err());
        config.seq_len = 1;
        assert!(validate_campaign_config(&config).is_err());

        let host_offload: CampaignConfig = serde_json::from_str(
            r#"{
                "model_dir":"model", "corpus":"corpus.json",
                "evaluation_corpus":"evaluation.json",
                "teacher_cache":"teacher.ttpr", "checkpoint_dir":"checkpoints",
                "report":"report.json", "artifact":"model.gguf",
                "seq_len":32, "steps":40, "checkpoint_every":5,
                "state_policy":"host-offload", "packed_compute":"fast"
            }"#,
        )
        .unwrap();
        validate_campaign_config(&host_offload).unwrap();
        assert_eq!(host_offload.state_policy, CampaignStatePolicy::HostOffload);
        assert_eq!(
            host_offload.packed_compute,
            CampaignPackedComputePolicy::Fast
        );
        let resident_fast: CampaignConfig = serde_json::from_str(
            r#"{
                "model_dir":"model", "corpus":"corpus.json",
                "evaluation_corpus":"evaluation.json",
                "teacher_cache":"teacher.ttpr", "checkpoint_dir":"checkpoints",
                "report":"report.json", "artifact":"model.gguf",
                "seq_len":32, "steps":40, "checkpoint_every":5,
                "state_policy":"resident", "packed_compute":"fast"
            }"#,
        )
        .unwrap();
        validate_campaign_config(&resident_fast).unwrap();

        #[cfg(feature = "nccl")]
        {
            let mut distributed: CampaignConfig = serde_json::from_str(
                r#"{
                    "model_dir":"model", "corpus":"corpus.json",
                    "evaluation_corpus":"evaluation.json",
                    "teacher_cache":"teacher.ttpr", "checkpoint_dir":"checkpoints",
                    "report":"report.json", "artifact":"model.gguf",
                    "seq_len":32, "steps":40, "checkpoint_every":5,
                    "state_policy":"host-offload",
                    "distributed":{"devices":[0,1],"worker_timeout_seconds":30}
                }"#,
            )
            .unwrap();
            validate_campaign_config(&distributed).unwrap();
            distributed.cuda_device = Some(0);
            assert!(validate_campaign_config(&distributed).is_err());
            distributed.cuda_device = None;
            distributed.distributed.as_mut().unwrap().devices = vec![0, 0];
            assert!(validate_campaign_config(&distributed).is_err());
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn resident_campaign_runs_exact_and_fast_packed_steps_to_terminal_snapshot() {
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping resident campaign packed-step gate: {error}");
                return;
            }
        };
        for packed_compute in [
            CampaignPackedComputePolicy::Exact,
            CampaignPackedComputePolicy::Fast,
        ] {
            let mut model = resident_campaign_test_model();
            let masters = model.take_parameter_masters();
            let specs = model
                .parameters()
                .iter()
                .zip(masters)
                .map(|(parameter, master)| HostOffloadTrainParam {
                    master,
                    rows: parameter.rows,
                    cols: parameter.cols,
                    salt_planes: 2,
                    optimizer: AdamW::new(1e-3),
                })
                .collect();
            let mut trainer =
                CampaignTrainer::new(&backend, specs, CampaignStatePolicy::Resident).unwrap();
            trainer.prepare_execution(&backend).unwrap();
            let target = DeviceTensor::upload(&backend, &[0.125; 16]).unwrap();
            let timing_fence = DeviceTensor::upload(&backend, &[0.0]).unwrap();
            let evidence = trainer
                .run_step(
                    &CampaignWorld::single(),
                    &backend,
                    &model,
                    &target,
                    &[0, 1],
                    1,
                    &timing_fence,
                    packed_compute,
                )
                .unwrap();
            assert!(evidence.peak_live_requested_gradient_elements > 0);
            assert!(evidence.peak_live_activation_elements < evidence.naive_activation_elements);
            assert_eq!(trainer.completed_step(), 1);
            let terminal = trainer.into_master_source().unwrap();
            assert_eq!(terminal.completed_step(), 1);
            assert_eq!(terminal.len(), model.parameters().len());
            assert!(
                terminal
                    .master(0)
                    .unwrap()
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn naive_ptq_evaluation_runs_packed_exact_path_and_persists_complete_progress() {
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping packed PTQ campaign gate: {error}");
                return;
            }
        };
        let model = resident_campaign_test_model();
        let directory = temp_path("packed-ptq-evaluation");
        std::fs::create_dir_all(&directory).unwrap();
        let progress_path = directory.join("naive-salt-ptq-evaluation-progress.json");
        let model_identity = naive_salt_ptq_model_identity(&hex_digest([9; 32]), 2).unwrap();
        let score = resumable_naive_salt_ptq_score(
            &progress_path,
            &backend,
            &model,
            &model_identity,
            &[0, 1, 2, 3, 4, 5],
            3,
            [7; 32],
            [8; 32],
            2,
        )
        .unwrap();
        assert_eq!(score.window_count, 2);
        assert_eq!(score.token_count, 4);
        assert!(score.negative_log_likelihood.is_finite());
        assert!(score.perplexity.is_finite());
        let progress: TeacherForcedEvaluationProgress =
            serde_json::from_slice(&std::fs::read(&progress_path).unwrap()).unwrap();
        assert_eq!(progress.evaluation_kind, NAIVE_SALT_PTQ_EVALUATION_KIND);
        assert_eq!(progress.model_identity, model_identity);
        assert_eq!(progress.completed_windows, 2);
        assert_eq!(progress.token_count, 4);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn completed_evaluation_progress_resumes_without_reloading_model() {
        let directory = temp_path("evaluation-progress");
        std::fs::create_dir_all(&directory).unwrap();
        let progress_path = directory.join("source-evaluation-progress.json");
        let tokens = [1_u32, 2, 3, 4, 5, 6];
        let negative_log_likelihood = 4.0 * 2.0_f64.ln();
        let progress = TeacherForcedEvaluationProgress {
            schema_version: 1,
            evaluation_kind: "source-fp".into(),
            model_identity: "model-id".into(),
            corpus_file_digest: hex_digest([1; 32]),
            corpus_digest: hex_digest([2; 32]),
            seq_len: 3,
            total_windows: 2,
            completed_windows: 2,
            token_count: 4,
            negative_log_likelihood_f64_bits: negative_log_likelihood.to_bits(),
        };
        std::fs::write(&progress_path, serde_json::to_vec(&progress).unwrap()).unwrap();

        let score = resumable_teacher_forced_score(
            &progress_path,
            "source-fp",
            "model-id",
            &tokens,
            3,
            [1; 32],
            [2; 32],
            || -> anyhow::Result<ModelRunner> { bail!("completed resume rebuilt the model") },
        )
        .unwrap();
        assert_eq!(score.window_count, 2);
        assert_eq!(score.token_count, 4);
        assert_eq!(
            score.negative_log_likelihood.to_bits(),
            negative_log_likelihood.to_bits()
        );
        assert!((score.perplexity - 2.0).abs() < f64::EPSILON);

        assert!(
            resumable_teacher_forced_score(
                &progress_path,
                "source-fp",
                "different-model-id",
                &tokens,
                3,
                [1; 32],
                [2; 32],
                || -> anyhow::Result<ModelRunner> { bail!("mismatch rebuilt the model") },
            )
            .is_err()
        );

        let mut corrupt = progress;
        corrupt.token_count = 3;
        std::fs::write(&progress_path, serde_json::to_vec(&corrupt).unwrap()).unwrap();
        assert!(
            resumable_teacher_forced_score(
                &progress_path,
                "source-fp",
                "model-id",
                &tokens,
                3,
                [1; 32],
                [2; 32],
                || -> anyhow::Result<ModelRunner> { bail!("corruption rebuilt the model") },
            )
            .is_err()
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn partial_evaluation_progress_scores_only_remaining_windows() {
        use std::cell::Cell;

        let directory = temp_path("partial-evaluation-progress");
        std::fs::create_dir_all(&directory).unwrap();
        let progress_path = directory.join("artifact-evaluation-progress.json");
        let tokens = [0_u32, 1, 2, 2, 1, 0];
        let first =
            teacher_forced_perplexity_windows(&mut evaluation_test_runner(), &tokens[..3], 3)
                .unwrap();
        let second =
            teacher_forced_perplexity_windows(&mut evaluation_test_runner(), &tokens[3..], 3)
                .unwrap();
        let progress = TeacherForcedEvaluationProgress {
            schema_version: 1,
            evaluation_kind: "training-salt-artifact".into(),
            model_identity: "package-id".into(),
            corpus_file_digest: hex_digest([3; 32]),
            corpus_digest: hex_digest([4; 32]),
            seq_len: 3,
            total_windows: 2,
            completed_windows: 1,
            token_count: first.token_count,
            negative_log_likelihood_f64_bits: first.negative_log_likelihood.to_bits(),
        };
        std::fs::write(&progress_path, serde_json::to_vec(&progress).unwrap()).unwrap();
        let builds = Cell::new(0_u32);

        let score = resumable_teacher_forced_score(
            &progress_path,
            "training-salt-artifact",
            "package-id",
            &tokens,
            3,
            [3; 32],
            [4; 32],
            || {
                builds.set(builds.get() + 1);
                Ok(evaluation_test_runner())
            },
        )
        .unwrap();
        let expected_nll = first.negative_log_likelihood + second.negative_log_likelihood;
        assert_eq!(builds.get(), 1);
        assert_eq!(score.window_count, 2);
        assert_eq!(score.token_count, 4);
        assert_eq!(
            score.negative_log_likelihood.to_bits(),
            expected_nll.to_bits()
        );
        let persisted: TeacherForcedEvaluationProgress =
            serde_json::from_slice(&std::fs::read(&progress_path).unwrap()).unwrap();
        assert_eq!(persisted.completed_windows, 2);
        assert_eq!(persisted.token_count, 4);
        assert_eq!(
            persisted.negative_log_likelihood_f64_bits,
            expected_nll.to_bits()
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn completed_naive_ptq_progress_resumes_without_scoring_or_accepting_new_provenance() {
        let directory = temp_path("naive-ptq-evaluation-progress");
        std::fs::create_dir_all(&directory).unwrap();
        let progress_path = directory.join("naive-salt-ptq-evaluation-progress.json");
        let tokens = [1_u32, 2, 3, 4, 5, 6];
        let negative_log_likelihood = 4.0 * 3.0_f64.ln();
        let progress = TeacherForcedEvaluationProgress {
            schema_version: 1,
            evaluation_kind: NAIVE_SALT_PTQ_EVALUATION_KIND.into(),
            model_identity: "immutable-plan".into(),
            corpus_file_digest: hex_digest([5; 32]),
            corpus_digest: hex_digest([6; 32]),
            seq_len: 3,
            total_windows: 2,
            completed_windows: 2,
            token_count: 4,
            negative_log_likelihood_f64_bits: negative_log_likelihood.to_bits(),
        };
        std::fs::write(&progress_path, serde_json::to_vec(&progress).unwrap()).unwrap();

        let score = resumable_teacher_forced_score_with(
            &progress_path,
            NAIVE_SALT_PTQ_EVALUATION_KIND,
            "immutable-plan",
            &tokens,
            3,
            [5; 32],
            [6; 32],
            |_| -> anyhow::Result<TeacherForcedPerplexity> {
                bail!("completed PTQ resume rescored a window")
            },
        )
        .unwrap();
        assert!((score.perplexity - 3.0).abs() < 1e-12);

        assert!(
            resumable_teacher_forced_score_with(
                &progress_path,
                NAIVE_SALT_PTQ_EVALUATION_KIND,
                "different-plan",
                &tokens,
                3,
                [5; 32],
                [6; 32],
                |_| -> anyhow::Result<TeacherForcedPerplexity> {
                    bail!("provenance mismatch scored a window")
                },
            )
            .is_err()
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn packed_ptq_logits_score_exact_next_tokens_and_fail_closed() {
        let score = teacher_forced_score_from_full_logits(
            &[0, 1, 0],
            2,
            &[0.0, 0.0, 0.0, 0.0, f32::NAN, f32::NAN],
        )
        .unwrap();
        assert_eq!(score.window_count, 1);
        assert_eq!(score.token_count, 2);
        assert!((score.negative_log_likelihood - 2.0_f64.ln() * 2.0).abs() < 1e-12);
        assert!((score.perplexity - 2.0).abs() < 1e-12);

        assert!(teacher_forced_score_from_full_logits(&[0, 2], 2, &[0.0; 4]).is_err());
        assert!(teacher_forced_score_from_full_logits(&[0, 1], 2, &[0.0; 3]).is_err());
        assert!(
            teacher_forced_score_from_full_logits(&[0, 1], 2, &[f32::NAN, 0.0, 0.0, 0.0]).is_err()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn resumable_evaluation_rejects_invalid_window_evidence_before_persisting() {
        let progress_path = temp_path("invalid-window-evidence");
        let error = resumable_teacher_forced_score_with(
            &progress_path,
            NAIVE_SALT_PTQ_EVALUATION_KIND,
            "model-id",
            &[0, 1, 2],
            3,
            [1; 32],
            [2; 32],
            |_| {
                Ok(TeacherForcedPerplexity {
                    perplexity: 1.0,
                    negative_log_likelihood: -0.5,
                    token_count: 2,
                    window_count: 1,
                })
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("inconsistent evidence"));
        assert!(!progress_path.exists());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn evaluation_report_records_naive_ptq_recovery_and_provenance() {
        let campaign = CampaignConfig {
            model_dir: PathBuf::from("model"),
            corpus: PathBuf::from("train.json"),
            evaluation_corpus: PathBuf::from("held-out.json"),
            teacher_cache: PathBuf::from("teacher.ttpr"),
            checkpoint_dir: PathBuf::from("checkpoints"),
            report: PathBuf::from("report.json"),
            artifact: PathBuf::from("artifact.gguf"),
            seq_len: 3,
            steps: 5,
            salt_planes: 2,
            cuda_device: Some(0),
            distributed: None,
            state_policy: CampaignStatePolicy::HostOffload,
            packed_compute: CampaignPackedComputePolicy::Exact,
            checkpoint_every: 5,
            timing_warmup_steps: 1,
            checkpoint_shards: 1,
            adam: CampaignAdam::default(),
            growth: None,
        };
        let artifact = CampaignArtifactReport {
            path: "artifact.gguf".into(),
            package_id: "package-id".into(),
            physical_bytes: 1,
            scale_max_abs_error: 0.0,
            reconstruction_max_abs_delta: 0.0,
            reconstruction_squared_error_sum: 0.0,
            reconstruction_element_count: 1,
            reconstruction_mse: 0.0,
        };
        let score = |perplexity| TeacherForcedPerplexity {
            perplexity,
            negative_log_likelihood: 4.0 * perplexity.ln(),
            token_count: 4,
            window_count: 2,
        };

        let report = evaluation_report(
            &campaign,
            &artifact,
            "immutable-plan",
            "immutable-plan",
            score(2.0),
            Some(score(2.0)),
            100,
            200,
            score(900.0),
            score(3.0),
            [7; 32],
            [8; 32],
        )
        .unwrap();
        assert_eq!(report.naive_salt_ptq.method, NAIVE_SALT_PTQ_METHOD);
        assert_eq!(report.naive_salt_ptq.model_identity, "immutable-plan");
        assert_eq!(
            report.naive_salt_ptq.perplexity.to_bits(),
            900.0_f64.to_bits()
        );
        assert_eq!(
            report.recovery_vs_naive_salt_ptq.to_bits(),
            300.0_f64.to_bits()
        );
        let grown_fp = report.grown_fp.expect("grown fp evidence");
        assert_eq!(grown_fp.model_identity, "immutable-plan");
        assert_eq!(grown_fp.source_parameter_count, 100);
        assert_eq!(grown_fp.grown_parameter_count, 200);
        assert_eq!(grown_fp.perplexity.to_bits(), 2.0_f64.to_bits());
        assert_eq!(grown_fp.relative_perplexity_delta_percent, 0.0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn campaign_lock_is_acquired_before_model_or_cache_io() {
        let directory = temp_path("early-lock");
        let checkpoint_dir = directory.join("checkpoints");
        std::fs::create_dir_all(&checkpoint_dir).unwrap();
        let lock_holder = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(checkpoint_dir.join("campaign.lock"))
            .unwrap();
        lock_holder.lock().unwrap();
        let config_path = directory.join("campaign.json");
        let config = serde_json::json!({
            "model_dir": directory.join("missing-model"),
            "corpus": directory.join("missing-corpus.json"),
            "evaluation_corpus": directory.join("missing-evaluation.json"),
            "teacher_cache": directory.join("missing-teacher.ttpr"),
            "checkpoint_dir": checkpoint_dir,
            "report": directory.join("report.json"),
            "artifact": directory.join("artifact.gguf"),
            "seq_len": 2,
            "steps": 1,
            "checkpoint_every": 1
        });
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();

        let error = run_campaign(&config_path).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("acquire exclusive campaign lock"));
        assert!(!message.contains("load HuggingFace model"));
        assert!(!message.contains("teacher cache"));
        drop(lock_holder);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn campaign_lock_reuses_an_unlocked_sentinel_after_process_exit() {
        let directory = temp_path("stale-advisory-lock");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("campaign.lock");
        let stale_contents = b"pid=dead\n";
        std::fs::write(&path, stale_contents).unwrap();

        let first = CampaignLock::acquire(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), stale_contents);
        assert!(CampaignLock::acquire(&path).is_err());
        drop(first);
        let second = CampaignLock::acquire(&path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), stale_contents);
        drop(second);

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn campaign_lock_rejects_a_non_regular_path() {
        let directory = temp_path("non-regular-advisory-lock");
        let path = directory.join("campaign.lock");
        std::fs::create_dir_all(&path).unwrap();

        let error = CampaignLock::acquire(&path).unwrap_err();
        assert!(error.to_string().contains("must be a regular file"));

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(all(feature = "cuda", unix))]
    #[test]
    fn campaign_lock_rejects_a_symlink_without_mutating_its_target() {
        use std::os::unix::fs::symlink;

        let directory = temp_path("symlink-advisory-lock");
        std::fs::create_dir_all(&directory).unwrap();
        let target = directory.join("protected.json");
        let contents = b"do-not-truncate";
        std::fs::write(&target, contents).unwrap();
        let path = directory.join("campaign.lock");
        symlink(&target, &path).unwrap();

        let error = CampaignLock::acquire(&path).unwrap_err();
        assert!(error.to_string().contains("must be a regular file"));
        assert_eq!(std::fs::read(&target).unwrap(), contents);

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(all(feature = "cuda", unix))]
    #[test]
    fn campaign_lock_rejects_a_hard_link_without_mutating_its_target() {
        let directory = temp_path("hardlink-advisory-lock");
        std::fs::create_dir_all(&directory).unwrap();
        let target = directory.join("protected.json");
        let contents = b"do-not-truncate";
        std::fs::write(&target, contents).unwrap();
        let path = directory.join("campaign.lock");
        std::fs::hard_link(&target, &path).unwrap();

        let error = CampaignLock::acquire(&path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must not be a hard-linked inode")
        );
        assert_eq!(std::fs::read(&target).unwrap(), contents);

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn campaign_lock_set_serializes_shared_report_and_artifact_outputs() {
        let directory = temp_path("output-advisory-locks");
        std::fs::create_dir_all(&directory).unwrap();
        let checkpoint_a = directory.join("checkpoint-a");
        let checkpoint_b = directory.join("checkpoint-b");
        let report_a = directory.join("report-a.json");
        let report_b = directory.join("report-b.json");
        let artifact_a = directory.join("artifact-a.gguf");
        let artifact_b = directory.join("artifact-b.gguf");

        let report_holder = CampaignLocks::acquire(&checkpoint_a, &report_a, &artifact_a).unwrap();
        let report_error =
            CampaignLocks::acquire(&checkpoint_b, &report_a, &artifact_b).unwrap_err();
        assert!(
            report_error
                .to_string()
                .contains("report-a.json.campaign.lock")
        );
        drop(report_holder);
        drop(CampaignLocks::acquire(&checkpoint_b, &report_a, &artifact_b).unwrap());

        let artifact_holder =
            CampaignLocks::acquire(&checkpoint_a, &report_a, &artifact_a).unwrap();
        let artifact_error =
            CampaignLocks::acquire(&checkpoint_b, &report_b, &artifact_a).unwrap_err();
        assert!(
            artifact_error
                .to_string()
                .contains("artifact-a.gguf.campaign.lock")
        );
        drop(artifact_holder);
        drop(CampaignLocks::acquire(&checkpoint_b, &report_b, &artifact_a).unwrap());

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn campaign_lock_paths_are_sorted_and_deduplicated() {
        let directory = temp_path("ordered-advisory-locks");
        std::fs::create_dir_all(&directory).unwrap();
        let checkpoint = directory.join("z-checkpoint");
        let output = directory.join("a-output.json");

        let paths = campaign_lock_paths(&checkpoint, &output, &output).unwrap();
        assert_eq!(paths.len(), 2);
        assert!(paths.windows(2).all(|pair| pair[0] < pair[1]));

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn campaign_lock_sidecar_must_not_alias_an_output() {
        let directory = temp_path("advisory-lock-output-alias");
        std::fs::create_dir_all(&directory).unwrap();
        let report = directory.join("result");
        let artifact = directory.join("result.campaign.lock");

        let error =
            CampaignLocks::acquire(&directory.join("checkpoint"), &report, &artifact).unwrap_err();
        assert!(error.to_string().contains("aliases a campaign output"));
        assert!(!artifact.exists());

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn campaign_paths_reject_report_and_checkpoint_aliases() {
        let directory = temp_path("path-collisions");
        let model_dir = directory.join("model");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("config.json"), b"{}").unwrap();
        let config_path = directory.join("campaign.json");
        let corpus = directory.join("corpus.json");
        let evaluation_corpus = directory.join("evaluation.json");
        let teacher_cache = directory.join("teacher.ttpr");
        std::fs::write(&config_path, b"{}").unwrap();
        std::fs::write(&corpus, b"[]").unwrap();
        std::fs::write(&evaluation_corpus, b"[1,2]").unwrap();
        std::fs::write(&teacher_cache, b"cache").unwrap();
        let checkpoint_dir = directory.join("checkpoints");
        let mut config = CampaignConfig {
            model_dir: model_dir.clone(),
            corpus: corpus.clone(),
            evaluation_corpus: evaluation_corpus.clone(),
            teacher_cache: teacher_cache.clone(),
            checkpoint_dir: checkpoint_dir.clone(),
            report: directory.join("report.json"),
            artifact: directory.join("artifact.gguf"),
            seq_len: 2,
            steps: 1,
            salt_planes: 2,
            cuda_device: Some(0),
            distributed: None,
            state_policy: CampaignStatePolicy::Resident,
            packed_compute: CampaignPackedComputePolicy::Exact,
            checkpoint_every: 1,
            timing_warmup_steps: 1,
            checkpoint_shards: 1,
            adam: CampaignAdam::default(),
            growth: None,
        };
        validate_campaign_paths(&config_path, &config).unwrap();

        config.report = checkpoint_dir.join("manifest.tdcp");
        assert!(validate_campaign_paths(&config_path, &config).is_err());
        config.report = corpus.clone();
        assert!(validate_campaign_paths(&config_path, &config).is_err());
        config.report = model_dir.join("campaign-report.json");
        assert!(validate_campaign_paths(&config_path, &config).is_err());

        config.report = directory.join("report.json");
        config.artifact = evaluation_corpus.clone();
        assert!(validate_campaign_paths(&config_path, &config).is_err());
        config.artifact = checkpoint_dir.join("artifact.gguf");
        assert!(validate_campaign_paths(&config_path, &config).is_err());
        config.artifact = directory.join("artifact.gguf");

        let report_container = directory.join("report-container");
        config.report = report_container.clone();
        config.checkpoint_dir = report_container.join("checkpoints");
        assert!(validate_campaign_paths(&config_path, &config).is_err());

        let report_directory = directory.join("report-directory");
        std::fs::create_dir_all(&report_directory).unwrap();
        config.report = report_directory;
        config.checkpoint_dir = checkpoint_dir;
        assert!(validate_campaign_paths(&config_path, &config).is_err());

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;

            let regular = directory.join("regular-report-target");
            let special = directory.join("special-report");
            std::fs::write(&regular, b"{}").unwrap();
            symlink(&regular, &special).unwrap();
            config.report = special;
            assert!(validate_campaign_paths(&config_path, &config).is_err());
        }

        config.report = directory.join("report.json");
        config.checkpoint_dir = directory;
        assert!(validate_campaign_paths(&config_path, &config).is_err());
        std::fs::remove_dir_all(config.checkpoint_dir).unwrap();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn resume_validates_timing_coverage_before_truncating_orphan_report() {
        let directory = temp_path("orphan-report");
        let checkpoint_dir = directory.join("checkpoint");
        std::fs::create_dir_all(&checkpoint_dir).unwrap();
        let path = directory.join("report.json");
        let input_hashes = CampaignInputHashes {
            campaign_config_file: "config".into(),
            model_config_file: "model-config".into(),
            source_model: "source-model".into(),
            initial_student_model: hex_digest([9; 32]),
            corpus_file: "corpus-file".into(),
            corpus: "corpus".into(),
            teacher_cache: "teacher".into(),
            evaluation_corpus_file: hex_digest([7; 32]),
            evaluation_corpus: hex_digest([8; 32]),
        };
        let hardware = vec![test_hardware()];
        let memory = CampaignMemoryReport {
            packed_parameter_bytes: 6,
            packed_code_bytes: 2,
            packed_scale_bytes: 4,
            dense_parameter_bytes: 16,
            host_optimizer_bytes: 48,
            fleet_host_optimizer_bytes: 48,
            host_adapter_master_bytes: 0,
            logical_host_training_state_bytes: 48,
            resident_training_state_bytes: 0,
            optimizer_state_device_fraction: 0.0,
            peak_optimizer_staging_bytes: 24,
            fleet_peak_optimizer_staging_bytes: 24,
            peak_live_requested_gradient_bytes: 8,
            materialized_gradient_bytes: 16,
            logical_peak_activation_bytes: 12,
            logical_naive_activation_bytes: 20,
        };
        let static_memory = CampaignStaticMemory {
            packed_parameter_bytes: 6,
            packed_code_bytes: 2,
            packed_scale_bytes: 4,
            dense_parameter_bytes: 16,
            host_optimizer_bytes: 48,
            resident_training_state_bytes: 0,
            peak_optimizer_staging_bytes: 24,
            materialized_gradient_bytes: 16,
        };
        let report = CampaignReport {
            schema_version: 6,
            plan_fingerprint: "plan".into(),
            state_policy: CampaignStatePolicy::HostOffload,
            packed_compute: CampaignPackedComputePolicy::Exact,
            world_size: 1,
            window_partition: "contiguous-modulo-v1".into(),
            hardware: hardware.clone(),
            input_hashes: input_hashes.clone(),
            completed_step: 5,
            resumed_from_step: 0,
            configured_steps: 5,
            timing_warmup_steps: 1,
            checkpoint_path: checkpoint_dir.display().to_string(),
            step_timings: (1..=5)
                .map(|step| StepTiming {
                    step,
                    window_index: step - 1,
                    warmup: step == 1,
                    training_elapsed_ms_excluding_checkpoint: step as f64,
                    rank_evidence: vec![RankStepEvidence {
                        rank: 0,
                        window_index: step - 1,
                        training_elapsed_ms: step as f64,
                        nvml_used_memory_bytes: 10,
                        cuda_memory: test_memory_snapshot(),
                    }],
                })
                .collect(),
            memory,
            cuda_memory_segments: vec![CampaignCudaMemorySegment {
                resumed_from_step: 0,
                completed_step: 5,
                nvml_used_memory_bytes_at_launch: 10,
                nvml_used_memory_latest_sample_bytes: 10,
                nvml_used_memory_sample_high_water_bytes: 10,
                baseline: test_memory_snapshot(),
                post_setup: Some(test_memory_snapshot()),
                latest_or_terminal: test_memory_snapshot(),
            }],
            artifact: None,
            evaluation: None,
        };
        std::fs::write(&path, serde_json::to_vec(&report).unwrap()).unwrap();
        let expected = ReportExpectations {
            plan_fingerprint: "plan",
            state_policy: CampaignStatePolicy::HostOffload,
            packed_compute: CampaignPackedComputePolicy::Exact,
            checkpoint_step: 3,
            checkpoint_dir: &checkpoint_dir,
            windows: 5,
            configured_steps: 5,
            timing_warmup_steps: 1,
            evaluation_seq_len: 3,
            evaluation_windows: 2,
            evaluation_corpus_path: Path::new("evaluation.json"),
            hardware: &hardware,
            input_hashes: &input_hashes,
            static_memory,
            salt_planes: 2,
            source_parameter_count: 100,
            grown_parameter_count: 100,
        };

        let resumed = load_existing_report(&path, &expected).unwrap().unwrap();
        assert_eq!(resumed.completed_step, 3);
        assert_eq!(resumed.step_timings.len(), 3);
        assert_eq!(resumed.memory.peak_optimizer_staging_bytes, 24);
        assert!(
            load_existing_report(
                &path,
                &ReportExpectations {
                    checkpoint_step: 6,
                    ..expected
                }
            )
            .is_err()
        );
        let mut other_hardware = hardware.clone();
        other_hardware[0].cuda_ordinal = 1;
        assert!(
            load_existing_report(
                &path,
                &ReportExpectations {
                    hardware: &other_hardware,
                    ..expected
                }
            )
            .is_err()
        );

        let rejects = |candidate: &CampaignReport| {
            std::fs::write(&path, serde_json::to_vec(candidate).unwrap()).unwrap();
            load_existing_report(&path, &expected).unwrap_err()
        };
        let score = |perplexity: f64| {
            let negative_log_likelihood = 4.0 * perplexity.ln();
            (
                negative_log_likelihood,
                (negative_log_likelihood / 4.0).exp(),
            )
        };
        let (source_nll, source_ppl) = score(2.0);
        let (ptq_nll, ptq_ppl) = score(900.0);
        let (artifact_nll, artifact_ppl) = score(3.0);
        let mut terminal = report.clone();
        terminal.artifact = Some(CampaignArtifactReport {
            path: "artifact.gguf".into(),
            package_id: "package-id".into(),
            physical_bytes: 1,
            scale_max_abs_error: 0.0,
            reconstruction_max_abs_delta: 0.0,
            reconstruction_squared_error_sum: 0.0,
            reconstruction_element_count: 1,
            reconstruction_mse: 0.0,
        });
        terminal.evaluation = Some(CampaignEvaluationReport {
            artifact_package_id: "package-id".into(),
            corpus_path: "evaluation.json".into(),
            corpus_file_digest: input_hashes.evaluation_corpus_file.clone(),
            corpus_digest: input_hashes.evaluation_corpus.clone(),
            seq_len: 3,
            window_count: 2,
            token_count: 4,
            source_fp_negative_log_likelihood: source_nll,
            source_fp_perplexity: source_ppl,
            naive_salt_ptq: CampaignPtqBaselineReport {
                method: NAIVE_SALT_PTQ_METHOD.into(),
                model_identity: naive_salt_ptq_model_identity(
                    &input_hashes.initial_student_model,
                    2,
                )
                .unwrap(),
                salt_planes: 2,
                negative_log_likelihood: ptq_nll,
                perplexity: ptq_ppl,
            },
            negative_log_likelihood: artifact_nll,
            perplexity: artifact_ppl,
            relative_perplexity_delta_percent: (artifact_ppl / source_ppl - 1.0) * 100.0,
            recovery_vs_naive_salt_ptq: ptq_ppl / artifact_ppl,
            grown_fp: None,
        });
        std::fs::write(&path, serde_json::to_vec(&terminal).unwrap()).unwrap();
        let resumed_terminal = load_existing_report(&path, &expected).unwrap().unwrap();
        assert!(resumed_terminal.artifact.is_none());
        assert!(resumed_terminal.evaluation.is_none());

        let grown_expected = ReportExpectations {
            grown_parameter_count: 200,
            ..expected
        };
        let mut grown_terminal = terminal.clone();
        grown_terminal.evaluation.as_mut().unwrap().grown_fp = Some(CampaignGrownFpReport {
            model_identity: input_hashes.initial_student_model.clone(),
            source_parameter_count: 100,
            grown_parameter_count: 200,
            negative_log_likelihood: source_nll,
            perplexity: source_ppl,
            relative_perplexity_delta_percent: 0.0,
        });
        std::fs::write(&path, serde_json::to_vec(&grown_terminal).unwrap()).unwrap();
        load_existing_report(&path, &grown_expected)
            .expect("grown report validation")
            .expect("grown report exists");
        let mut corrupt_grown = grown_terminal;
        corrupt_grown
            .evaluation
            .as_mut()
            .unwrap()
            .grown_fp
            .as_mut()
            .unwrap()
            .relative_perplexity_delta_percent = 1.0;
        std::fs::write(&path, serde_json::to_vec(&corrupt_grown).unwrap()).unwrap();
        assert!(
            load_existing_report(&path, &grown_expected)
                .unwrap_err()
                .to_string()
                .contains("inconsistent artifact/evaluation evidence")
        );

        let mut legacy_terminal = serde_json::to_value(&terminal).unwrap();
        legacy_terminal["schema_version"] = serde_json::Value::from(5);
        let legacy_evaluation = legacy_terminal["evaluation"].as_object_mut().unwrap();
        legacy_evaluation.remove("naive_salt_ptq");
        legacy_evaluation.remove("recovery_vs_naive_salt_ptq");
        std::fs::write(&path, serde_json::to_vec(&legacy_terminal).unwrap()).unwrap();
        let migrated_terminal = load_existing_report(&path, &expected).unwrap().unwrap();
        assert_eq!(migrated_terminal.schema_version, 6);
        assert_eq!(migrated_terminal.completed_step, 3);
        assert!(migrated_terminal.artifact.is_none());
        assert!(migrated_terminal.evaluation.is_none());
        let mut unpaired_legacy = legacy_terminal;
        unpaired_legacy["evaluation"] = serde_json::Value::Null;
        std::fs::write(&path, serde_json::to_vec(&unpaired_legacy).unwrap()).unwrap();
        assert!(
            load_existing_report(&path, &expected)
                .unwrap_err()
                .to_string()
                .contains("unpaired artifact/evaluation")
        );

        let mut corrupt = terminal.clone();
        corrupt
            .evaluation
            .as_mut()
            .unwrap()
            .recovery_vs_naive_salt_ptq = 299.0;
        assert!(
            rejects(&corrupt)
                .to_string()
                .contains("inconsistent artifact/evaluation evidence")
        );
        let mut corrupt = terminal.clone();
        corrupt.evaluation.as_mut().unwrap().corpus_path = "other-evaluation.json".into();
        assert!(
            rejects(&corrupt)
                .to_string()
                .contains("inconsistent artifact/evaluation evidence")
        );
        let mut corrupt = terminal;
        corrupt
            .evaluation
            .as_mut()
            .unwrap()
            .naive_salt_ptq
            .model_identity
            .push_str("-different");
        assert!(
            rejects(&corrupt)
                .to_string()
                .contains("inconsistent artifact/evaluation evidence")
        );

        let mut legacy = report.clone();
        legacy.schema_version = 5;
        std::fs::write(&path, serde_json::to_vec(&legacy).unwrap()).unwrap();
        let migrated = load_existing_report(&path, &expected).unwrap().unwrap();
        assert_eq!(migrated.schema_version, 6);
        assert_eq!(migrated.completed_step, 3);
        let mut corrupt = report.clone();
        corrupt.schema_version = 4;
        assert!(rejects(&corrupt).to_string().contains("unsupported schema"));
        let mut corrupt = report.clone();
        corrupt.configured_steps = 6;
        assert!(rejects(&corrupt).to_string().contains("configured steps"));
        let mut corrupt = report.clone();
        corrupt.checkpoint_path = directory.join("other-checkpoint").display().to_string();
        assert!(
            rejects(&corrupt)
                .to_string()
                .contains("points at checkpoint")
        );
        let mut corrupt = report.clone();
        corrupt.input_hashes.source_model.push_str("-different");
        assert!(rejects(&corrupt).to_string().contains("input hashes"));
        let mut corrupt = report.clone();
        corrupt.resumed_from_step = 6;
        assert!(rejects(&corrupt).to_string().contains("resumed from step"));
        let mut corrupt = report.clone();
        corrupt.memory.logical_host_training_state_bytes += size_of::<f32>();
        assert!(
            rejects(&corrupt)
                .to_string()
                .contains("training-state bytes")
        );
        let mut corrupt = report.clone();
        corrupt.memory.packed_code_bytes += 64;
        corrupt.memory.packed_parameter_bytes += 64;
        assert!(rejects(&corrupt).to_string().contains("packed code bytes"));
        let mut corrupt = report.clone();
        corrupt.memory.dense_parameter_bytes += size_of::<f32>();
        corrupt.memory.host_optimizer_bytes = corrupt.memory.dense_parameter_bytes * 3;
        corrupt.memory.logical_host_training_state_bytes = corrupt.memory.host_optimizer_bytes;
        corrupt.memory.materialized_gradient_bytes = corrupt.memory.dense_parameter_bytes;
        assert!(
            rejects(&corrupt)
                .to_string()
                .contains("dense parameter bytes")
        );
        let mut corrupt = report.clone();
        corrupt.memory.peak_optimizer_staging_bytes += size_of::<f32>();
        assert!(
            rejects(&corrupt)
                .to_string()
                .contains("optimizer staging bytes")
        );
        let mut corrupt = report.clone();
        corrupt.memory.fleet_host_optimizer_bytes += size_of::<f32>();
        assert!(
            rejects(&corrupt)
                .to_string()
                .contains("fleet host optimizer bytes")
        );
        let mut corrupt = report.clone();
        corrupt.memory.optimizer_state_device_fraction = 1.0;
        assert!(
            rejects(&corrupt)
                .to_string()
                .contains("optimizer-state device fraction")
        );
        let mut corrupt = report.clone();
        corrupt.memory.materialized_gradient_bytes += size_of::<f32>();
        assert!(
            rejects(&corrupt)
                .to_string()
                .contains("materialized gradient bytes")
        );

        let mut corrupt = report.clone();
        corrupt.step_timings[4].step = 4;
        let error = rejects(&corrupt);
        assert!(
            error.to_string().contains("expected ordered unique step 5"),
            "orphan rows must be validated before truncation: {error:#}"
        );

        let mut corrupt = report.clone();
        corrupt.step_timings.pop();
        assert!(validate_report_timing_coverage(&path, &corrupt, 5, &hardware).is_err());
        let mut corrupt = report.clone();
        corrupt.step_timings[2].window_index = 4;
        assert!(validate_report_timing_coverage(&path, &corrupt, 5, &hardware).is_err());
        let mut corrupt = report.clone();
        corrupt.step_timings[2].training_elapsed_ms_excluding_checkpoint = f64::NAN;
        assert!(validate_report_timing_coverage(&path, &corrupt, 5, &hardware).is_err());

        let mut distributed_hardware = hardware.clone();
        let mut peer_hardware = test_hardware();
        peer_hardware.cuda_ordinal = 1;
        peer_hardware.pci_bus_id = "0000:02:00.0".into();
        distributed_hardware.push(peer_hardware);
        let mut distributed_report = report.clone();
        distributed_report.world_size = 2;
        distributed_report.memory.fleet_host_optimizer_bytes = 96;
        distributed_report.memory.fleet_peak_optimizer_staging_bytes = 48;
        for timing in &mut distributed_report.step_timings {
            let owner_window = ((timing.step - 1) * 2) % 5;
            timing.window_index = owner_window;
            timing.rank_evidence[0].window_index = owner_window;
            let peer_elapsed = timing.training_elapsed_ms_excluding_checkpoint + 0.5;
            timing.rank_evidence.push(RankStepEvidence {
                rank: 1,
                window_index: (owner_window + 1) % 5,
                training_elapsed_ms: peer_elapsed,
                nvml_used_memory_bytes: 10,
                cuda_memory: test_memory_snapshot(),
            });
            timing.training_elapsed_ms_excluding_checkpoint = peer_elapsed;
        }
        validate_report_timing_coverage(&path, &distributed_report, 5, &distributed_hardware)
            .unwrap();
        validate_report_memory(
            &path,
            &distributed_report.memory,
            static_memory,
            distributed_report.completed_step,
            2,
            CampaignStatePolicy::HostOffload,
        )
        .unwrap();
        let mut underreported_fleet_memory = distributed_report.memory.clone();
        underreported_fleet_memory.fleet_host_optimizer_bytes = 48;
        assert!(
            validate_report_memory(
                &path,
                &underreported_fleet_memory,
                static_memory,
                distributed_report.completed_step,
                2,
                CampaignStatePolicy::HostOffload,
            )
            .is_err()
        );
        let mut invalid_peer_memory = distributed_report.clone();
        invalid_peer_memory.step_timings[0].rank_evidence[1]
            .cuda_memory
            .pool_reserved_current_bytes = 0;
        assert!(
            validate_report_timing_coverage(&path, &invalid_peer_memory, 5, &distributed_hardware,)
                .is_err()
        );
        distributed_report.step_timings[0].rank_evidence[1].window_index = 4;
        assert!(
            validate_report_timing_coverage(&path, &distributed_report, 5, &distributed_hardware,)
                .is_err()
        );

        let mut corrupt = report;
        corrupt.step_timings[2].training_elapsed_ms_excluding_checkpoint = -1.0;
        assert!(validate_report_timing_coverage(&path, &corrupt, 5, &hardware).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn plan_fingerprint_is_stable_and_changes_with_optimizer_or_geometry() {
        let (config, spec, architecture, parameters) = fixture();
        let model_digest = semantic_model_digest_parts(&config, &spec, &architecture, &parameters);
        let hardware = vec![test_hardware()];
        let growth = test_growth();
        let base = PlanInputs {
            hardware: &hardware,
            campaign_config_file_digest: [5; 32],
            model_config_file_digest: [6; 32],
            source_model_digest: model_digest,
            initial_student_digest: [9; 32],
            corpus_digest: [2; 32],
            corpus_file_digest: [3; 32],
            teacher_cache_digest: [4; 32],
            evaluation_corpus_path: Path::new("/campaign/evaluation.json"),
            evaluation_corpus_digest: [10; 32],
            evaluation_corpus_file_digest: [11; 32],
            evaluation_windows: 2,
            artifact_path: Path::new("/campaign/model.gguf"),
            growth: &growth,
            seq_len: 4,
            windows: 2,
            total_steps: 20,
            salt_planes: 2,
            state_policy: CampaignStatePolicy::Resident,
            packed_compute: CampaignPackedComputePolicy::Exact,
            world_size: 1,
            adam: CampaignAdam::default(),
            depth: 1,
            checkpoint_every: 5,
            timing_warmup_steps: 1,
            checkpoint_shards: 1,
            parameters: &parameters,
        };
        let first = build_plan_sidecar(&base);
        assert_eq!(first, build_plan_sidecar(&base));
        let changed = PlanInputs {
            adam: CampaignAdam {
                lr: 1e-3,
                ..base.adam
            },
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let changed = PlanInputs {
            salt_planes: 3,
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let changed = PlanInputs {
            state_policy: CampaignStatePolicy::HostOffload,
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let changed = PlanInputs {
            packed_compute: CampaignPackedComputePolicy::Fast,
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let changed = PlanInputs {
            world_size: 2,
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let mut changed_hardware = hardware.clone();
        changed_hardware[0].cuda_ordinal = 1;
        let changed = PlanInputs {
            hardware: &changed_hardware,
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let changed = PlanInputs {
            campaign_config_file_digest: [7; 32],
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let changed = PlanInputs {
            model_config_file_digest: [8; 32],
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let changed = PlanInputs {
            evaluation_corpus_digest: [12; 32],
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let changed = PlanInputs {
            artifact_path: Path::new("/campaign/other.gguf"),
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let mut changed_growth = growth.clone();
        changed_growth.seed = 1;
        let changed = PlanInputs {
            growth: &changed_growth,
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let mut changed_growth = growth.clone();
        changed_growth.algorithm = tritium_train::grow::NET2WIDER_ALGORITHM_V2.into();
        changed_growth.old_width = 2;
        changed_growth.new_width = 3;
        changed_growth.seed = 99;
        changed_growth.source_indices = vec![0, 1, 1];
        changed_growth.replication_counts = vec![1, 2];
        changed_growth.split_denominator_log2 = Some(24);
        changed_growth.split_numerators = Some(vec![16_777_216, 4_194_304, 12_582_912]);
        let changed = PlanInputs {
            growth: &changed_growth,
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        assert_eq!(first.hardware, hardware);
        assert_eq!(first.version, 5);
        assert_eq!(first.campaign_config_file_digest, hex_digest([5; 32]));
        assert_eq!(first.model_config_file_digest, hex_digest([6; 32]));
    }

    #[test]
    fn immutable_plan_sidecar_rejects_mismatch_and_missing_sidecar_for_checkpoint() {
        let directory = temp_path("plan-sidecar");
        let (config, spec, architecture, parameters) = fixture();
        let hardware = vec![test_hardware()];
        let growth = test_growth();
        let expected = build_plan_sidecar(&PlanInputs {
            hardware: &hardware,
            campaign_config_file_digest: [5; 32],
            model_config_file_digest: [6; 32],
            source_model_digest: semantic_model_digest_parts(
                &config,
                &spec,
                &architecture,
                &parameters,
            ),
            initial_student_digest: [9; 32],
            corpus_digest: [2; 32],
            corpus_file_digest: [3; 32],
            teacher_cache_digest: [4; 32],
            evaluation_corpus_path: Path::new("/campaign/evaluation.json"),
            evaluation_corpus_digest: [10; 32],
            evaluation_corpus_file_digest: [11; 32],
            evaluation_windows: 2,
            artifact_path: Path::new("/campaign/model.gguf"),
            growth: &growth,
            seq_len: 4,
            windows: 2,
            total_steps: 20,
            salt_planes: 2,
            state_policy: CampaignStatePolicy::Resident,
            packed_compute: CampaignPackedComputePolicy::Exact,
            world_size: 1,
            adam: CampaignAdam::default(),
            depth: 1,
            checkpoint_every: 5,
            timing_warmup_steps: 1,
            checkpoint_shards: 1,
            parameters: &parameters,
        });
        ensure_plan_sidecar(&directory, &expected).unwrap();
        ensure_plan_sidecar(&directory, &expected).unwrap();
        let mut mismatch = expected.clone();
        mismatch.fingerprint = "different".into();
        assert!(ensure_plan_sidecar(&directory, &mismatch).is_err());
        std::fs::remove_dir_all(&directory).unwrap();

        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("manifest.tdcp"), b"committed").unwrap();
        assert!(ensure_plan_sidecar(&directory, &expected).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn teacher_cache_publish_is_atomic_and_interruption_preserves_old_output() {
        assert_eq!(output_parent(Path::new("teacher.ttpr")), Path::new("."));
        let directory = temp_path("atomic-cache");
        std::fs::create_dir_all(&directory).unwrap();
        let output = directory.join("teacher.ttpr");
        std::fs::write(&output, b"old").unwrap();
        let header = TeacherCacheHeader {
            seq_len: 1,
            vocab: 2,
            windows: 1,
            model_hash: [1; 32],
            corpus_hash: [2; 32],
        };
        let failure =
            publish_teacher_cache(&output, header, |_writer| bail!("injected interruption"));
        assert!(failure.is_err());
        assert_eq!(std::fs::read(&output).unwrap(), b"old");
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);

        publish_teacher_cache(&output, header, |writer| {
            writer.write_window(&[0.25, 0.75])?;
            Ok(())
        })
        .unwrap();
        assert_ne!(std::fs::read(&output).unwrap(), b"old");
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
