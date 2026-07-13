//! Reproducible offline-teacher and packed-SALT campaign orchestration.

use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(any(feature = "cuda", test))]
use std::fs::OpenOptions;
#[cfg(any(feature = "cuda", test))]
use std::io::Write;

use anyhow::{Context, bail};
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use tritium_format::TeacherCacheHeader;
#[cfg(feature = "cuda")]
use tritium_format::write_teacher_cache_header;
use tritium_nn::{
    ArchSpec, MlpKind, ModelConfig, ModelRunner, ModelWeights, TeacherCacheWriter,
    TiedSwiGluTrainingModel, hash_teacher_corpus,
};

#[cfg(feature = "cuda")]
use std::time::Instant;
#[cfg(feature = "cuda")]
use tritium_cuda::CudaBackend;
#[cfg(feature = "cuda")]
use tritium_cuda::train::{
    CheckpointPolicy, DevicePackedSaltWeight, DeviceTape, DeviceTensor, GradientLeafBinding,
    HostOffloadTrainParam, HostOffloadTrainer,
};
#[cfg(feature = "cuda")]
use tritium_nn::{TeacherCacheReader, TrainingParameter, packed_device_forward};
#[cfg(feature = "cuda")]
use tritium_spec::TernaryBackend;
#[cfg(feature = "cuda")]
use tritium_train::AdamW;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

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
        /// teacher_cache, checkpoint_dir, report, seq_len, steps, and
        /// checkpoint_every. Optional keys: salt_planes, cuda_device,
        /// checkpoint_shards, and adam. Relative paths resolve beside this file.
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
/// norm vectors are domain-separated into this digest rather than hashing only
/// the 2D master values.
fn semantic_model_digest(
    config: &ModelConfig,
    spec: &ArchSpec,
    model: &TiedSwiGluTrainingModel,
) -> [u8; 32] {
    semantic_model_digest_parts(config, spec, model.architecture(), model.parameters())
}

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
    hash_f32s(&mut hash, &architecture.output_norm);
    *hash.finalize().as_bytes()
}

fn hash_bytes(hash: &mut blake3::Hasher, bytes: &[u8]) {
    hash.update(&(bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

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
    let report = path_identity(&config.report)?;
    let checkpoint = path_identity(&config.checkpoint_dir)?;
    let model_dir = path_identity(&config.model_dir)?;
    let protected_files = [
        ("campaign config", path_identity(config_path)?),
        ("corpus", path_identity(&config.corpus)?),
        ("teacher cache", path_identity(&config.teacher_cache)?),
        (
            "model config",
            path_identity(&config.model_dir.join("config.json"))?,
        ),
    ];

    if report.starts_with(&checkpoint) || checkpoint.starts_with(&report) {
        bail!(
            "campaign report {} and checkpoint directory {} must not contain one another",
            config.report.display(),
            config.checkpoint_dir.display()
        );
    }
    if report.starts_with(&model_dir) {
        bail!(
            "campaign report {} must be outside model directory {}",
            config.report.display(),
            config.model_dir.display()
        );
    }
    if checkpoint.starts_with(&model_dir) || model_dir.starts_with(&checkpoint) {
        bail!(
            "checkpoint directory {} must not overlap model directory {}",
            config.checkpoint_dir.display(),
            config.model_dir.display()
        );
    }
    for (label, protected) in protected_files {
        if report == protected {
            bail!(
                "campaign report {} aliases the {label}",
                config.report.display()
            );
        }
        if protected.starts_with(&checkpoint) {
            bail!(
                "checkpoint directory {} contains the {label}; choose a dedicated checkpoint directory",
                config.checkpoint_dir.display()
            );
        }
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
struct CampaignLock {
    path: PathBuf,
    _file: File,
}

#[cfg(feature = "cuda")]
impl CampaignLock {
    fn acquire(checkpoint_dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(checkpoint_dir)
            .with_context(|| format!("create checkpoint directory {}", checkpoint_dir.display()))?;
        let path = checkpoint_dir.join("campaign.lock");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .with_context(|| {
                format!(
                    "acquire exclusive campaign lock {}; if no campaign is running, remove this stale lock after verifying the prior process exited",
                    path.display()
                )
            })?;
        writeln!(file, "pid={}", std::process::id())?;
        file.sync_all()?;
        Ok(Self { path, _file: file })
    }
}

#[cfg(feature = "cuda")]
impl Drop for CampaignLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
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
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CampaignConfig {
    model_dir: PathBuf,
    corpus: PathBuf,
    teacher_cache: PathBuf,
    checkpoint_dir: PathBuf,
    report: PathBuf,
    seq_len: usize,
    steps: u64,
    #[serde(default = "default_planes")]
    salt_planes: usize,
    #[serde(default)]
    cuda_device: usize,
    checkpoint_every: u64,
    #[serde(default = "default_checkpoint_shards")]
    checkpoint_shards: usize,
    #[serde(default)]
    adam: CampaignAdam,
}

#[cfg(feature = "cuda")]
impl CampaignConfig {
    fn resolve_paths(&mut self, config_path: &Path) {
        let base = config_path.parent().unwrap_or_else(|| Path::new("."));
        for path in [
            &mut self.model_dir,
            &mut self.corpus,
            &mut self.teacher_cache,
            &mut self.checkpoint_dir,
            &mut self.report,
        ] {
            if path.is_relative() {
                *path = base.join(&*path);
            }
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
struct PlanSidecar {
    version: u32,
    fingerprint: String,
    cuda_device: usize,
    cuda_device_name: String,
    campaign_config_file_digest: String,
    model_config_file_digest: String,
    model_digest: String,
    corpus_digest: String,
    corpus_file_digest: String,
    teacher_cache_digest: String,
    seq_len: usize,
    windows: u64,
    total_steps: u64,
    salt_planes: usize,
    adam_f32_bits: [u32; 5],
    activation_checkpoint_policy: String,
    checkpoint_every: u64,
    checkpoint_shards: usize,
    parameters: Vec<ParameterIdentity>,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy)]
struct PlanInputs<'a> {
    cuda_device: usize,
    cuda_device_name: &'a str,
    campaign_config_file_digest: [u8; 32],
    model_config_file_digest: [u8; 32],
    model_digest: [u8; 32],
    corpus_digest: [u8; 32],
    corpus_file_digest: [u8; 32],
    teacher_cache_digest: [u8; 32],
    seq_len: usize,
    windows: u64,
    total_steps: u64,
    salt_planes: usize,
    adam: CampaignAdam,
    depth: usize,
    checkpoint_every: u64,
    checkpoint_shards: usize,
    parameters: &'a [tritium_nn::TrainingParameter],
}

#[cfg(any(feature = "cuda", test))]
fn build_plan_sidecar(inputs: &PlanInputs<'_>) -> PlanSidecar {
    let mut hash = blake3::Hasher::new();
    hash.update(b"tritium-packed-campaign-plan-v3");
    hash.update(&(inputs.cuda_device as u64).to_le_bytes());
    hash_bytes(&mut hash, inputs.cuda_device_name.as_bytes());
    hash.update(&inputs.campaign_config_file_digest);
    hash.update(&inputs.model_config_file_digest);
    hash.update(&inputs.model_digest);
    hash.update(&inputs.corpus_digest);
    hash.update(&inputs.corpus_file_digest);
    hash.update(&inputs.teacher_cache_digest);
    hash.update(&(inputs.seq_len as u64).to_le_bytes());
    hash.update(&inputs.windows.to_le_bytes());
    hash.update(&inputs.total_steps.to_le_bytes());
    hash.update(&(inputs.salt_planes as u64).to_le_bytes());
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
        version: 3,
        fingerprint,
        cuda_device: inputs.cuda_device,
        cuda_device_name: inputs.cuda_device_name.to_owned(),
        campaign_config_file_digest: hex_digest(inputs.campaign_config_file_digest),
        model_config_file_digest: hex_digest(inputs.model_config_file_digest),
        model_digest: hex_digest(inputs.model_digest),
        corpus_digest: hex_digest(inputs.corpus_digest),
        corpus_file_digest: hex_digest(inputs.corpus_file_digest),
        teacher_cache_digest: hex_digest(inputs.teacher_cache_digest),
        seq_len: inputs.seq_len,
        windows: inputs.windows,
        total_steps: inputs.total_steps,
        salt_planes: inputs.salt_planes,
        adam_f32_bits: adam_bits,
        activation_checkpoint_policy: format!("sqrt_depth:{}", inputs.depth),
        checkpoint_every: inputs.checkpoint_every,
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
    let _campaign_lock = CampaignLock::acquire(&campaign.checkpoint_dir)?;
    let backend = CudaBackend::new(campaign.cuda_device)
        .with_context(|| format!("open CUDA device {}", campaign.cuda_device))?;
    let cuda_device_name = backend.capabilities().device_name;

    let (model_config, spec, weights) = ModelWeights::load_hf(&campaign.model_dir)
        .with_context(|| format!("load HuggingFace model {}", campaign.model_dir.display()))?;
    let mut model = TiedSwiGluTrainingModel::extract(&model_config, &spec, &weights)
        .context("validate tied-SwiGLU training adapter")?;
    drop(weights);
    let model_digest = semantic_model_digest(&model_config, &spec, &model);
    let model_config_file_digest = hash_file(&campaign.model_dir.join("config.json"))?;
    let (tokens, corpus_file_digest) = read_corpus(&campaign.corpus)?;
    let windows = validate_windows(
        &tokens,
        campaign.seq_len,
        model.architecture().vocab,
        model.architecture().n_ctx,
    )?;
    let seq_len_u32 = u32::try_from(campaign.seq_len).context("sequence length exceeds u32")?;
    let corpus_digest = hash_teacher_corpus(&tokens, seq_len_u32);
    let expected_cache = TeacherCacheHeader {
        seq_len: seq_len_u32,
        vocab: u32::try_from(model.architecture().vocab).context("vocabulary exceeds u32")?,
        windows,
        model_hash: model_digest,
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

    let plan = build_plan_sidecar(&PlanInputs {
        cuda_device: campaign.cuda_device,
        cuda_device_name: &cuda_device_name,
        campaign_config_file_digest: config_file_digest,
        model_config_file_digest,
        model_digest,
        corpus_digest,
        corpus_file_digest,
        teacher_cache_digest,
        seq_len: campaign.seq_len,
        windows,
        total_steps: campaign.steps,
        salt_planes: campaign.salt_planes,
        adam: campaign.adam,
        depth: model.architecture().n_layers,
        checkpoint_every: campaign.checkpoint_every,
        checkpoint_shards: campaign.checkpoint_shards,
        parameters: model.parameters(),
    });
    let input_hashes = CampaignInputHashes {
        campaign_config_file: hex_digest(config_file_digest),
        model_config_file: hex_digest(model_config_file_digest),
        model: hex_digest(model_digest),
        corpus_file: hex_digest(corpus_file_digest),
        corpus: hex_digest(corpus_digest),
        teacher_cache: hex_digest(teacher_cache_digest),
    };
    // This comparison/creation happens before DCP load can mutate trainer state.
    ensure_plan_sidecar(&campaign.checkpoint_dir, &plan)?;

    let static_memory = expected_static_memory(model.parameters(), campaign.salt_planes)?;
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
    let mut trainer = HostOffloadTrainer::new_owned(&backend, specs)
        .context("construct host-offloaded AdamW trainer")?;
    let manifest = campaign.checkpoint_dir.join("manifest.tdcp");
    if manifest
        .try_exists()
        .with_context(|| format!("inspect DCP manifest {}", manifest.display()))?
    {
        tritium_train::dcp::load_into(&campaign.checkpoint_dir, &mut trainer).with_context(
            || format!("load DCP checkpoint {}", campaign.checkpoint_dir.display()),
        )?;
    }
    if trainer.completed_step() > campaign.steps {
        bail!(
            "checkpoint completed step {} exceeds configured terminal step {}",
            trainer.completed_step(),
            campaign.steps
        );
    }
    let report_expectations = ReportExpectations {
        plan_fingerprint: &plan.fingerprint,
        checkpoint_step: trainer.completed_step(),
        checkpoint_dir: &campaign.checkpoint_dir,
        windows,
        configured_steps: campaign.steps,
        cuda_device: campaign.cuda_device,
        cuda_device_name: &cuda_device_name,
        input_hashes: &input_hashes,
        static_memory,
    };
    let previous_report = load_existing_report(&campaign.report, &report_expectations)?;
    if trainer.completed_step() == campaign.steps
        && let Some(report) = previous_report.as_ref()
        && report.completed_step == campaign.steps
    {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }

    let mut packed = build_packed_weights(&backend, &trainer)?;
    let packed_parameter_bytes = packed.iter().try_fold(0usize, |sum, weight| {
        sum.checked_add(weight.resident_bytes())
            .context("packed parameter byte count overflow")
    })?;
    let packed_code_bytes = packed.iter().try_fold(0usize, |sum, weight| {
        sum.checked_add(weight.packed_bytes())
            .context("packed code byte count overflow")
    })?;
    let packed_scale_bytes = packed.iter().try_fold(0usize, |sum, weight| {
        sum.checked_add(weight.scale_bytes())
            .context("packed scale byte count overflow")
    })?;

    let mut target_host = vec![0.0f32; window_elements];
    let timing_fence =
        DeviceTensor::upload(&backend, &[0.0]).context("allocate training-stream timing fence")?;
    let previous_memory = previous_report.as_ref().map(|report| report.memory.clone());
    let mut timings = previous_report
        .map(|report| report.step_timings)
        .unwrap_or_default();
    let mut max_gradient_elements = 0usize;
    let mut materialized_gradient_elements = 0usize;
    let mut max_activation_elements = 0usize;
    let mut naive_activation_elements = 0usize;
    let mut stable_bindings: Option<Vec<GradientLeafBinding>> = None;
    let resumed_from_step = trainer.completed_step();
    let host_adapter_master_bytes = model
        .parameters()
        .iter()
        .try_fold(0usize, |sum, parameter| {
            sum.checked_add(parameter.master.len())
                .context("adapter master element count overflow")
        })
        .and_then(elements_to_bytes)?;
    let previous_peak =
        |select: fn(&CampaignMemoryReport) -> usize| previous_memory.as_ref().map_or(0, select);
    let build_report = |trainer: &HostOffloadTrainer<'_>,
                        step_timings: &[StepTiming],
                        max_gradient_elements: usize,
                        materialized_gradient_elements: usize,
                        max_activation_elements: usize,
                        naive_activation_elements: usize|
     -> anyhow::Result<CampaignReport> {
        let stats = trainer.stats();
        let host_optimizer_bytes = elements_to_bytes(stats.host_optimizer_elements)?;
        let logical_host_training_state_bytes = host_optimizer_bytes
            .checked_add(host_adapter_master_bytes)
            .context("logical host training-state byte count overflow")?;
        Ok(CampaignReport {
            schema_version: 3,
            plan_fingerprint: plan.fingerprint.clone(),
            cuda_device: campaign.cuda_device,
            cuda_device_name: cuda_device_name.clone(),
            input_hashes: input_hashes.clone(),
            completed_step: trainer.completed_step(),
            resumed_from_step,
            configured_steps: campaign.steps,
            checkpoint_path: campaign.checkpoint_dir.display().to_string(),
            step_timings: step_timings.to_vec(),
            memory: CampaignMemoryReport {
                packed_parameter_bytes,
                packed_code_bytes,
                packed_scale_bytes,
                dense_parameter_bytes: static_memory.dense_parameter_bytes,
                host_optimizer_bytes,
                host_adapter_master_bytes,
                logical_host_training_state_bytes,
                peak_optimizer_staging_bytes: elements_to_bytes(
                    stats.peak_optimizer_device_elements,
                )?
                .max(previous_peak(|memory| memory.peak_optimizer_staging_bytes)),
                peak_streamed_gradient_bytes: elements_to_bytes(max_gradient_elements)?
                    .max(previous_peak(|memory| memory.peak_streamed_gradient_bytes)),
                materialized_gradient_bytes: elements_to_bytes(materialized_gradient_elements)?
                    .max(previous_peak(|memory| memory.materialized_gradient_bytes)),
                logical_peak_activation_bytes: elements_to_bytes(max_activation_elements)?
                    .max(previous_peak(|memory| memory.logical_peak_activation_bytes)),
                logical_naive_activation_bytes: elements_to_bytes(naive_activation_elements)?.max(
                    previous_peak(|memory| memory.logical_naive_activation_bytes),
                ),
            },
        })
    };
    let first_step = trainer
        .completed_step()
        .checked_add(1)
        .context("completed step overflow")?;

    for step in first_step..=campaign.steps {
        let started = Instant::now();
        let window_index = (step - 1) % windows;
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

        let stream_result = {
            let mut tape = DeviceTape::new_with_checkpoint_policy(
                &backend,
                model.architecture().vocab,
                CheckpointPolicy::SqrtDepth(model.architecture().n_layers),
            )
            .context("create sqrt-depth device tape")?;
            let forward = packed_device_forward(&mut tape, &model, &packed, &tokens_i32)
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
            if let Some(expected) = &stable_bindings {
                if expected != &bindings {
                    bail!("packed gradient leaf bindings changed between steps");
                }
            } else {
                stable_bindings = Some(bindings.clone());
            }
            tape.xent_backward_into(
                forward.logits,
                &target,
                campaign.seq_len,
                model.architecture().vocab,
                &bindings,
                &mut trainer,
                step,
            )
        };
        // Any attempted update invalidates every packed generation. A success
        // repacks all weights; a failure returns with all handles visibly stale.
        for weight in &mut packed {
            weight.mark_stale();
        }
        let stream = stream_result.with_context(|| format!("training step {step}"))?;
        repack_all_or_stale(&backend, &mut packed, &trainer)
            .with_context(|| format!("repack every parameter after step {step}"))?;
        // A tiny same-stream round trip fences every queued repack kernel so its
        // cost is charged to this step rather than leaking into the next one.
        timing_fence
            .download(&backend)
            .with_context(|| format!("synchronize packed repack at step {step}"))?;

        max_gradient_elements =
            max_gradient_elements.max(stream.peak_live_requested_gradient_elements);
        materialized_gradient_elements =
            materialized_gradient_elements.max(stream.materialized_collection_elements);
        max_activation_elements =
            max_activation_elements.max(stream.backward_stats.peak_live_activation_elements);
        naive_activation_elements =
            naive_activation_elements.max(stream.backward_stats.naive_activation_elements);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        timings.push(StepTiming {
            step,
            window_index,
            training_elapsed_ms_excluding_checkpoint: elapsed_ms,
        });

        if step.is_multiple_of(campaign.checkpoint_every) || step == campaign.steps {
            let progress = build_report(
                &trainer,
                &timings,
                max_gradient_elements,
                materialized_gradient_elements,
                max_activation_elements,
                naive_activation_elements,
            )?;
            atomic_write(&campaign.report, &serde_json::to_vec_pretty(&progress)?)?;
            // Publish evidence before DCP's manifest commit. If saving stops
            // early, resume treats timings beyond the prior manifest as orphaned.
            tritium_train::dcp::save_from(
                &campaign.checkpoint_dir,
                &mut trainer,
                campaign.checkpoint_shards,
            )
            .with_context(|| format!("save DCP at completed step {step}"))?;
        }
    }

    let report = build_report(
        &trainer,
        &timings,
        max_gradient_elements,
        materialized_gradient_elements,
        max_activation_elements,
        naive_activation_elements,
    )?;
    atomic_write(&campaign.report, &serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_campaign_config(config: &CampaignConfig) -> anyhow::Result<()> {
    if config.steps == 0 {
        bail!("campaign steps must be non-zero");
    }
    if !(1..=3).contains(&config.salt_planes) {
        bail!("salt_planes must be in 1..=3");
    }
    if config.checkpoint_every == 0 {
        bail!("checkpoint_every must be non-zero");
    }
    if config.checkpoint_shards == 0 {
        bail!("checkpoint_shards must be non-zero");
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
fn expected_static_memory(
    parameters: &[TrainingParameter],
    salt_planes: usize,
) -> anyhow::Result<StaticMemoryExpectations> {
    if !(1..=3).contains(&salt_planes) {
        bail!("SALT plane count must be in 1..=3");
    }
    let mut dense_elements = 0usize;
    let mut largest_parameter_elements = 0usize;
    let mut packed_code_bytes = 0usize;
    let mut packed_scale_bytes = 0usize;
    for parameter in parameters {
        let elements = parameter
            .rows
            .checked_mul(parameter.cols)
            .context("dense parameter shape overflow")?;
        dense_elements = dense_elements
            .checked_add(elements)
            .context("dense parameter element count overflow")?;
        largest_parameter_elements = largest_parameter_elements.max(elements);

        let blocks_per_row = parameter.cols.div_ceil(tritium_format::QK_K);
        let parameter_code_bytes = salt_planes
            .checked_mul(parameter.rows)
            .and_then(|count| count.checked_mul(blocks_per_row))
            .and_then(|count| count.checked_mul(tritium_format::QK_K / 4))
            .context("packed SALT code byte count overflow")?;
        packed_code_bytes = packed_code_bytes
            .checked_add(parameter_code_bytes)
            .context("packed SALT code byte total overflow")?;
        let parameter_scale_bytes = salt_planes
            .checked_mul(parameter.rows)
            .and_then(|count| count.checked_mul(size_of::<f32>()))
            .context("packed SALT scale byte count overflow")?;
        packed_scale_bytes = packed_scale_bytes
            .checked_add(parameter_scale_bytes)
            .context("packed SALT scale byte total overflow")?;
    }

    let dense_parameter_bytes = elements_to_bytes(dense_elements)?;
    let host_optimizer_bytes = dense_parameter_bytes
        .checked_mul(3)
        .context("host optimizer byte count overflow")?;
    let peak_optimizer_staging_bytes = elements_to_bytes(largest_parameter_elements)?
        .checked_mul(6)
        .context("optimizer staging byte count overflow")?;
    let packed_parameter_bytes = packed_code_bytes
        .checked_add(packed_scale_bytes)
        .context("packed parameter byte count overflow")?;
    Ok(StaticMemoryExpectations {
        packed_parameter_bytes,
        packed_code_bytes,
        packed_scale_bytes,
        dense_parameter_bytes,
        host_optimizer_bytes,
        host_adapter_master_bytes: 0,
        logical_host_training_state_bytes: host_optimizer_bytes,
        peak_optimizer_staging_bytes,
        materialized_gradient_bytes: dense_parameter_bytes,
    })
}

#[cfg(feature = "cuda")]
fn elements_to_bytes(elements: usize) -> anyhow::Result<usize> {
    elements
        .checked_mul(size_of::<f32>())
        .context("f32 byte count overflow")
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Deserialize, Serialize)]
struct StepTiming {
    step: u64,
    window_index: u64,
    training_elapsed_ms_excluding_checkpoint: f64,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
struct CampaignInputHashes {
    campaign_config_file: String,
    model_config_file: String,
    model: String,
    corpus_file: String,
    corpus: String,
    teacher_cache: String,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Deserialize, Serialize)]
struct CampaignMemoryReport {
    packed_parameter_bytes: usize,
    packed_code_bytes: usize,
    packed_scale_bytes: usize,
    dense_parameter_bytes: usize,
    host_optimizer_bytes: usize,
    host_adapter_master_bytes: usize,
    logical_host_training_state_bytes: usize,
    peak_optimizer_staging_bytes: usize,
    peak_streamed_gradient_bytes: usize,
    materialized_gradient_bytes: usize,
    logical_peak_activation_bytes: usize,
    logical_naive_activation_bytes: usize,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Deserialize, Serialize)]
struct CampaignReport {
    schema_version: u32,
    plan_fingerprint: String,
    cuda_device: usize,
    cuda_device_name: String,
    input_hashes: CampaignInputHashes,
    completed_step: u64,
    resumed_from_step: u64,
    configured_steps: u64,
    checkpoint_path: String,
    step_timings: Vec<StepTiming>,
    memory: CampaignMemoryReport,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StaticMemoryExpectations {
    packed_parameter_bytes: usize,
    packed_code_bytes: usize,
    packed_scale_bytes: usize,
    dense_parameter_bytes: usize,
    host_optimizer_bytes: usize,
    host_adapter_master_bytes: usize,
    logical_host_training_state_bytes: usize,
    peak_optimizer_staging_bytes: usize,
    materialized_gradient_bytes: usize,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct ReportExpectations<'a> {
    plan_fingerprint: &'a str,
    checkpoint_step: u64,
    checkpoint_dir: &'a Path,
    windows: u64,
    configured_steps: u64,
    cuda_device: usize,
    cuda_device_name: &'a str,
    input_hashes: &'a CampaignInputHashes,
    static_memory: StaticMemoryExpectations,
}

#[cfg(feature = "cuda")]
fn validate_report_memory(
    path: &Path,
    memory: &CampaignMemoryReport,
    expected: StaticMemoryExpectations,
    completed_step: u64,
) -> anyhow::Result<()> {
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
            "host adapter master",
            memory.host_adapter_master_bytes,
            expected.host_adapter_master_bytes,
        ),
        (
            "logical host training-state",
            memory.logical_host_training_state_bytes,
            expected.logical_host_training_state_bytes,
        ),
        (
            "optimizer staging",
            memory.peak_optimizer_staging_bytes,
            expected.peak_optimizer_staging_bytes,
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
    if memory.peak_streamed_gradient_bytes > memory.materialized_gradient_bytes {
        bail!(
            "prior report {} has streamed-gradient peak larger than its materialized-gradient baseline",
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
        ("host adapter masters", memory.host_adapter_master_bytes),
        (
            "logical host training state",
            memory.logical_host_training_state_bytes,
        ),
        ("optimizer staging", memory.peak_optimizer_staging_bytes),
        ("streamed gradients", memory.peak_streamed_gradient_bytes),
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
) -> anyhow::Result<()> {
    if windows == 0 {
        bail!("cannot validate campaign timings against zero windows");
    }
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
        let expected_window_index = (expected_step - 1) % windows;
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
    let mut report: CampaignReport = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse prior report {}", path.display()))?;
    if report.schema_version != 3 {
        bail!(
            "prior report {} uses unsupported schema version {}",
            path.display(),
            report.schema_version
        );
    }
    if report.plan_fingerprint != expected.plan_fingerprint {
        bail!(
            "prior report {} belongs to a different immutable campaign plan",
            path.display()
        );
    }
    if report.cuda_device != expected.cuda_device
        || report.cuda_device_name != expected.cuda_device_name
    {
        bail!(
            "prior report {} records CUDA device {} ({:?}), expected {} ({:?})",
            path.display(),
            report.cuda_device,
            report.cuda_device_name,
            expected.cuda_device,
            expected.cuda_device_name
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
    )?;
    validate_report_timing_coverage(path, &report, expected.windows)?;
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
        report.completed_step = expected.checkpoint_step;
    }
    Ok(Some(report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_nn::{TiedSwiGluTrainingArchitecture, TrainingParameter};

    fn temp_path(name: &str) -> PathBuf {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "tritium-campaign-{name}-{}-{sequence}",
            std::process::id()
        ))
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
                "teacher_cache":"teacher.ttpr", "checkpoint_dir":"checkpoints",
                "report":"report.json", "seq_len":32, "steps":40,
                "checkpoint_every":5
            }"#,
        )
        .unwrap();
        config.resolve_paths(Path::new("/campaign/config.json"));
        assert_eq!(config.model_dir, Path::new("/campaign/model"));
        assert_eq!(config.report, Path::new("/campaign/report.json"));
        assert_eq!(config.salt_planes, 2);
        assert_eq!(config.checkpoint_every, 5);
        assert_eq!(config.checkpoint_shards, 1);
        assert_eq!(config.adam.lr.to_bits(), default_lr().to_bits());

        let missing_cadence = r#"{
            "model_dir":"model", "corpus":"corpus.json",
            "teacher_cache":"teacher.ttpr", "checkpoint_dir":"checkpoints",
            "report":"report.json", "seq_len":32, "steps":40
        }"#;
        assert!(serde_json::from_str::<CampaignConfig>(missing_cadence).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn campaign_lock_is_acquired_before_model_or_cache_io() {
        let directory = temp_path("early-lock");
        let checkpoint_dir = directory.join("checkpoints");
        std::fs::create_dir_all(&checkpoint_dir).unwrap();
        std::fs::write(checkpoint_dir.join("campaign.lock"), b"pid=test\n").unwrap();
        let config_path = directory.join("campaign.json");
        let config = serde_json::json!({
            "model_dir": directory.join("missing-model"),
            "corpus": directory.join("missing-corpus.json"),
            "teacher_cache": directory.join("missing-teacher.ttpr"),
            "checkpoint_dir": checkpoint_dir,
            "report": directory.join("report.json"),
            "seq_len": 1,
            "steps": 1,
            "checkpoint_every": 1
        });
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();

        let error = run_campaign(&config_path).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("acquire exclusive campaign lock"));
        assert!(!message.contains("load HuggingFace model"));
        assert!(!message.contains("teacher cache"));
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
        let teacher_cache = directory.join("teacher.ttpr");
        std::fs::write(&config_path, b"{}").unwrap();
        std::fs::write(&corpus, b"[]").unwrap();
        std::fs::write(&teacher_cache, b"cache").unwrap();
        let checkpoint_dir = directory.join("checkpoints");
        let mut config = CampaignConfig {
            model_dir: model_dir.clone(),
            corpus: corpus.clone(),
            teacher_cache: teacher_cache.clone(),
            checkpoint_dir: checkpoint_dir.clone(),
            report: directory.join("report.json"),
            seq_len: 1,
            steps: 1,
            salt_planes: 2,
            cuda_device: 0,
            checkpoint_every: 1,
            checkpoint_shards: 1,
            adam: CampaignAdam::default(),
        };
        validate_campaign_paths(&config_path, &config).unwrap();

        config.report = checkpoint_dir.join("manifest.tdcp");
        assert!(validate_campaign_paths(&config_path, &config).is_err());
        config.report = corpus;
        assert!(validate_campaign_paths(&config_path, &config).is_err());
        config.report = model_dir.join("campaign-report.json");
        assert!(validate_campaign_paths(&config_path, &config).is_err());

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
    fn static_memory_expectations_cover_packed_geometry_and_optimizer_state() {
        let parameters = vec![
            TrainingParameter {
                name: "wide".into(),
                master: vec![0.0; 2 * 257],
                rows: 2,
                cols: 257,
            },
            TrainingParameter {
                name: "narrow".into(),
                master: vec![0.0; 3],
                rows: 3,
                cols: 1,
            },
        ];

        assert_eq!(
            expected_static_memory(&parameters, 2).unwrap(),
            StaticMemoryExpectations {
                packed_parameter_bytes: 936,
                packed_code_bytes: 896,
                packed_scale_bytes: 40,
                dense_parameter_bytes: 2_068,
                host_optimizer_bytes: 6_204,
                host_adapter_master_bytes: 0,
                logical_host_training_state_bytes: 6_204,
                peak_optimizer_staging_bytes: 12_336,
                materialized_gradient_bytes: 2_068,
            }
        );
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
            model: "model".into(),
            corpus_file: "corpus-file".into(),
            corpus: "corpus".into(),
            teacher_cache: "teacher".into(),
        };
        let memory = CampaignMemoryReport {
            packed_parameter_bytes: 6,
            packed_code_bytes: 2,
            packed_scale_bytes: 4,
            dense_parameter_bytes: 16,
            host_optimizer_bytes: 48,
            host_adapter_master_bytes: 0,
            logical_host_training_state_bytes: 48,
            peak_optimizer_staging_bytes: 24,
            peak_streamed_gradient_bytes: 8,
            materialized_gradient_bytes: 16,
            logical_peak_activation_bytes: 12,
            logical_naive_activation_bytes: 20,
        };
        let static_memory = StaticMemoryExpectations {
            packed_parameter_bytes: 6,
            packed_code_bytes: 2,
            packed_scale_bytes: 4,
            dense_parameter_bytes: 16,
            host_optimizer_bytes: 48,
            host_adapter_master_bytes: 0,
            logical_host_training_state_bytes: 48,
            peak_optimizer_staging_bytes: 24,
            materialized_gradient_bytes: 16,
        };
        let report = CampaignReport {
            schema_version: 3,
            plan_fingerprint: "plan".into(),
            cuda_device: 0,
            cuda_device_name: "test cuda".into(),
            input_hashes: input_hashes.clone(),
            completed_step: 5,
            resumed_from_step: 0,
            configured_steps: 5,
            checkpoint_path: checkpoint_dir.display().to_string(),
            step_timings: (1..=5)
                .map(|step| StepTiming {
                    step,
                    window_index: step - 1,
                    training_elapsed_ms_excluding_checkpoint: step as f64,
                })
                .collect(),
            memory,
        };
        std::fs::write(&path, serde_json::to_vec(&report).unwrap()).unwrap();
        let expected = ReportExpectations {
            plan_fingerprint: "plan",
            checkpoint_step: 3,
            checkpoint_dir: &checkpoint_dir,
            windows: 5,
            configured_steps: 5,
            cuda_device: 0,
            cuda_device_name: "test cuda",
            input_hashes: &input_hashes,
            static_memory,
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
        assert!(
            load_existing_report(
                &path,
                &ReportExpectations {
                    cuda_device: 1,
                    ..expected
                }
            )
            .is_err()
        );
        assert!(
            load_existing_report(
                &path,
                &ReportExpectations {
                    cuda_device_name: "other cuda",
                    ..expected
                }
            )
            .is_err()
        );

        let rejects = |candidate: &CampaignReport| {
            std::fs::write(&path, serde_json::to_vec(candidate).unwrap()).unwrap();
            load_existing_report(&path, &expected).unwrap_err()
        };
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
        corrupt.input_hashes.model.push_str("-different");
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
        assert!(validate_report_timing_coverage(&path, &corrupt, 5).is_err());
        let mut corrupt = report.clone();
        corrupt.step_timings[2].window_index = 4;
        assert!(validate_report_timing_coverage(&path, &corrupt, 5).is_err());
        let mut corrupt = report.clone();
        corrupt.step_timings[2].training_elapsed_ms_excluding_checkpoint = f64::NAN;
        assert!(validate_report_timing_coverage(&path, &corrupt, 5).is_err());
        let mut corrupt = report;
        corrupt.step_timings[2].training_elapsed_ms_excluding_checkpoint = -1.0;
        assert!(validate_report_timing_coverage(&path, &corrupt, 5).is_err());
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn plan_fingerprint_is_stable_and_changes_with_optimizer_or_geometry() {
        let (config, spec, architecture, parameters) = fixture();
        let model_digest = semantic_model_digest_parts(&config, &spec, &architecture, &parameters);
        let base = PlanInputs {
            cuda_device: 0,
            cuda_device_name: "NVIDIA test GPU",
            campaign_config_file_digest: [5; 32],
            model_config_file_digest: [6; 32],
            model_digest,
            corpus_digest: [2; 32],
            corpus_file_digest: [3; 32],
            teacher_cache_digest: [4; 32],
            seq_len: 4,
            windows: 2,
            total_steps: 20,
            salt_planes: 2,
            adam: CampaignAdam::default(),
            depth: 1,
            checkpoint_every: 5,
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
            cuda_device: 1,
            ..base
        };
        assert_ne!(first.fingerprint, build_plan_sidecar(&changed).fingerprint);
        let changed = PlanInputs {
            cuda_device_name: "different GPU",
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
        assert_eq!(first.cuda_device, 0);
        assert_eq!(first.cuda_device_name, "NVIDIA test GPU");
        assert_eq!(first.campaign_config_file_digest, hex_digest([5; 32]));
        assert_eq!(first.model_config_file_digest, hex_digest([6; 32]));
    }

    #[test]
    fn immutable_plan_sidecar_rejects_mismatch_and_missing_sidecar_for_checkpoint() {
        let directory = temp_path("plan-sidecar");
        let (config, spec, architecture, parameters) = fixture();
        let expected = build_plan_sidecar(&PlanInputs {
            cuda_device: 0,
            cuda_device_name: "NVIDIA test GPU",
            campaign_config_file_digest: [5; 32],
            model_config_file_digest: [6; 32],
            model_digest: semantic_model_digest_parts(&config, &spec, &architecture, &parameters),
            corpus_digest: [2; 32],
            corpus_file_digest: [3; 32],
            teacher_cache_digest: [4; 32],
            seq_len: 4,
            windows: 2,
            total_steps: 20,
            salt_planes: 2,
            adam: CampaignAdam::default(),
            depth: 1,
            checkpoint_every: 5,
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
