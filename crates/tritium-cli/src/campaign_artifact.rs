//! Durable, self-contained GGUF export for ADR 0027 training campaigns.
//!
//! The deployment artifact keeps every trained two-dimensional master as a
//! fixed-plane SALT tensor and every RMSNorm vector as exact little-endian f32.
//! Tensor payloads are streamed in canonical training order, so peak conversion
//! storage is bounded by one weight row rather than one model.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tritium_format::{
    GGML_TYPE_TRITIUM_SALT, GgufStreamWriter, GgufTensorSpec, GgufValue, GgufWriteError,
    PackageHasher, PackageId, SALT_GGUF_FORMAT_KEY, SALT_GGUF_FORMAT_VALUE, SALT_HEADER_BYTES,
    TQ2_0_BLOCK_BYTES, num_blocks, pack_salt_row,
};
use tritium_nn::{
    ModelConfig, TRAINING_SALT_COMPLETED_STEP_KEY, TRAINING_SALT_FORMAT_KEY,
    TRAINING_SALT_FORMAT_VALUE, TRAINING_SALT_GROWTH_RECEIPT_KEY, TRAINING_SALT_HF_CONFIG_KEY,
    TRAINING_SALT_INITIAL_STUDENT_DIGEST_KEY, TRAINING_SALT_PLAN_FINGERPRINT_KEY,
    TRAINING_SALT_PLANES_KEY, TRAINING_SALT_SOURCE_MODEL_DIGEST_KEY,
    TiedSwiGluTrainingArchitecture, TiedSwiGluTrainingModel,
};
use tritium_quantize::{
    CampaignError, MeasuredPackage, TrainingSaltExportError, TrainingSaltExportStats,
    export_training_salt_row,
};

#[cfg(feature = "cuda")]
use tritium_cuda::train::HostOffloadTrainer;

const GGUF_VERSION: u32 = 3;
const GGML_TYPE_F32: u32 = 0;
const GGUF_ALIGNMENT: u32 = 32;
const GROWTH_ALGORITHM: &str = "net2wider.intermediate-swiglu.splitmix64.v1";
const TEMP_ATTEMPTS: usize = 32;

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Exact identities bound into a campaign artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ArtifactProvenance {
    plan_fingerprint: [u8; 32],
    source_model_digest: [u8; 32],
    initial_student_digest: [u8; 32],
    completed_step: u64,
}

impl ArtifactProvenance {
    pub(crate) fn new(
        plan_fingerprint: &str,
        source_model_digest: [u8; 32],
        initial_student_digest: [u8; 32],
        completed_step: u64,
    ) -> Result<Self, CampaignArtifactError> {
        let plan_fingerprint = parse_hex_digest("plan fingerprint", plan_fingerprint)?;
        for (name, digest) in [
            ("plan fingerprint", plan_fingerprint),
            ("source model digest", source_model_digest),
            ("initial student digest", initial_student_digest),
        ] {
            if digest == [0; 32] {
                return Err(invalid(format!("{name} must not be the all-zero digest")));
            }
        }
        Ok(Self {
            plan_fingerprint,
            source_model_digest,
            initial_student_digest,
            completed_step,
        })
    }
}

/// Replayable receipt for the one Net2Wider transform applied before training.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub(crate) struct GrowthReceipt {
    algorithm: String,
    old_width: u64,
    new_width: u64,
    seed: u64,
    source_indices: Vec<u64>,
    replication_counts: Vec<u64>,
}

impl GrowthReceipt {
    pub(crate) fn from_plan(
        plan: &tritium_train::Net2WiderPlan,
        seed: u64,
    ) -> Result<Self, CampaignArtifactError> {
        let old_width = u64::try_from(plan.replication_counts().len())
            .map_err(|_| invalid("growth old width exceeds u64"))?;
        let new_width = u64::try_from(plan.source_indices().len())
            .map_err(|_| invalid("growth new width exceeds u64"))?;
        let source_indices = plan
            .source_indices()
            .iter()
            .map(|&value| {
                u64::try_from(value).map_err(|_| invalid("growth source index exceeds u64"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let replication_counts = plan
            .replication_counts()
            .iter()
            .map(|&value| {
                u64::try_from(value).map_err(|_| invalid("growth replication count exceeds u64"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(
            old_width,
            new_width,
            seed,
            source_indices,
            replication_counts,
        )
    }

    fn new(
        old_width: u64,
        new_width: u64,
        seed: u64,
        source_indices: Vec<u64>,
        replication_counts: Vec<u64>,
    ) -> Result<Self, CampaignArtifactError> {
        if old_width == 0 || new_width < old_width {
            return Err(invalid(format!(
                "growth widths must satisfy 0 < old <= new, got {old_width} -> {new_width}"
            )));
        }
        if source_indices.len() as u128 != u128::from(new_width) {
            return Err(invalid(
                "growth source-index count does not equal new width",
            ));
        }
        if replication_counts.len() as u128 != u128::from(old_width) {
            return Err(invalid(
                "growth replication-count count does not equal old width",
            ));
        }
        let old_width_usize = usize::try_from(old_width)
            .map_err(|_| invalid("growth old width exceeds this target"))?;
        let mut actual_counts = vec![0_u64; old_width_usize];
        for (index, &source) in source_indices.iter().enumerate() {
            if source >= old_width {
                return Err(invalid(format!(
                    "growth source index {source} at position {index} exceeds old width {old_width}"
                )));
            }
            if index < old_width_usize && source != index as u64 {
                return Err(invalid("growth mapping does not have an identity prefix"));
            }
            let count = &mut actual_counts[source as usize];
            *count = count
                .checked_add(1)
                .ok_or_else(|| invalid("growth replication count overflows u64"))?;
        }
        if actual_counts != replication_counts {
            return Err(invalid(
                "growth replication counts disagree with source mapping",
            ));
        }
        Ok(Self {
            algorithm: GROWTH_ALGORITHM.to_owned(),
            old_width,
            new_width,
            seed,
            source_indices,
            replication_counts,
        })
    }

    fn canonical_json(&self) -> Result<String, CampaignArtifactError> {
        serde_json::to_string(self).map_err(CampaignArtifactError::Json)
    }
}

/// Exact package measurement plus deployment-projection error accumulated while exporting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CampaignArtifactSummary {
    package: MeasuredPackage,
    scale_max_abs_error: f64,
    reconstruction_max_abs_delta: f64,
    reconstruction_squared_error_sum: f64,
    reconstruction_element_count: u64,
}

impl CampaignArtifactSummary {
    pub(crate) fn package(self) -> MeasuredPackage {
        self.package
    }

    pub(crate) fn scale_max_abs_error(self) -> f64 {
        self.scale_max_abs_error
    }

    pub(crate) fn reconstruction_max_abs_delta(self) -> f64 {
        self.reconstruction_max_abs_delta
    }

    pub(crate) fn reconstruction_squared_error_sum(self) -> f64 {
        self.reconstruction_squared_error_sum
    }

    pub(crate) fn reconstruction_element_count(self) -> u64 {
        self.reconstruction_element_count
    }

    pub(crate) fn reconstruction_mse(self) -> f64 {
        assert!(
            self.reconstruction_element_count > 0,
            "validated campaign artifacts contain at least one trained weight"
        );
        self.reconstruction_squared_error_sum / self.reconstruction_element_count as f64
    }
}

/// Failures that prevent an artifact from being safely or exactly published.
#[derive(Debug)]
pub(crate) enum CampaignArtifactError {
    InvalidInput(String),
    Json(serde_json::Error),
    Io {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    Gguf(GgufWriteError),
    Salt(TrainingSaltExportError),
    Campaign(CampaignError),
}

impl fmt::Display for CampaignArtifactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(reason) => write!(formatter, "invalid campaign artifact: {reason}"),
            Self::Json(error) => write!(formatter, "campaign artifact JSON: {error}"),
            Self::Io {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "{operation} campaign artifact {}: {source}",
                path.display()
            ),
            Self::Gguf(error) => write!(formatter, "write campaign artifact GGUF: {error}"),
            Self::Salt(error) => write!(formatter, "export campaign SALT row: {error}"),
            Self::Campaign(error) => write!(formatter, "measure campaign artifact: {error}"),
        }
    }
}

impl std::error::Error for CampaignArtifactError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            Self::Gguf(error) => Some(error),
            Self::Salt(error) => Some(error),
            Self::Campaign(error) => Some(error),
            Self::InvalidInput(_) => None,
        }
    }
}

impl From<GgufWriteError> for CampaignArtifactError {
    fn from(error: GgufWriteError) -> Self {
        Self::Gguf(error)
    }
}

impl From<TrainingSaltExportError> for CampaignArtifactError {
    fn from(error: TrainingSaltExportError) -> Self {
        Self::Salt(error)
    }
}

impl From<CampaignError> for CampaignArtifactError {
    fn from(error: CampaignError) -> Self {
        Self::Campaign(error)
    }
}

#[derive(Clone, Debug)]
struct MatrixLayout {
    name: String,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Debug)]
struct NormLayout<'a> {
    name: String,
    values: Cow<'a, [f32]>,
}

#[derive(Clone, Debug)]
struct ArchitectureLayout {
    n_layers: usize,
    n_embd: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_ff: usize,
    n_ctx: usize,
    vocab: usize,
    rope_theta: f32,
    rms_eps: f32,
}

#[derive(Clone, Debug)]
struct ModelLayout<'a> {
    architecture: ArchitectureLayout,
    matrices: Vec<MatrixLayout>,
    norms: Vec<NormLayout<'a>>,
}

#[derive(Clone, Debug)]
struct ArtifactExportPlan<'a> {
    metadata: BTreeMap<String, GgufValue>,
    tensor_specs: Vec<GgufTensorSpec>,
    matrices: Vec<MatrixLayout>,
    norms: Vec<NormLayout<'a>>,
    salt_planes: usize,
    completed_step: u64,
}

impl<'a> ModelLayout<'a> {
    fn from_model(model: &'a TiedSwiGluTrainingModel) -> Result<Self, CampaignArtifactError> {
        let architecture = model.architecture();
        validate_architecture(architecture)?;
        let expected_parameters = architecture
            .n_layers
            .checked_mul(7)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid("canonical parameter count overflows usize"))?;
        if model.parameters().len() != expected_parameters {
            return Err(invalid(format!(
                "canonical parameter count is {}, expected {expected_parameters}",
                model.parameters().len()
            )));
        }
        let mut names = BTreeSet::new();
        let mut matrices = Vec::with_capacity(expected_parameters);
        for (index, parameter) in model.parameters().iter().enumerate() {
            if parameter.rows == 0 || parameter.cols == 0 {
                return Err(invalid(format!(
                    "parameter {} has an empty matrix shape",
                    parameter.name
                )));
            }
            if parameter.name.is_empty() || !names.insert(parameter.name.clone()) {
                return Err(invalid(format!(
                    "parameter name at canonical index {index} is empty or duplicated"
                )));
            }
            if parameter.name.contains("lm_head") {
                return Err(invalid(
                    "tied output head must not be stored separately from token embeddings",
                ));
            }
            matrices.push(MatrixLayout {
                name: parameter.name.clone(),
                rows: parameter.rows,
                cols: parameter.cols,
            });
        }
        if matrices.first().map(|matrix| matrix.name.as_str()) != Some("model.embed_tokens.weight")
        {
            return Err(invalid(
                "first canonical parameter is not model.embed_tokens.weight",
            ));
        }

        let mut norms = Vec::with_capacity(architecture.n_layers * 2 + 1);
        for layer in 0..architecture.n_layers {
            norms.push(NormLayout {
                name: format!("model.layers.{layer}.input_layernorm.weight"),
                values: Cow::Borrowed(&architecture.attn_norms[layer]),
            });
            norms.push(NormLayout {
                name: format!("model.layers.{layer}.post_attention_layernorm.weight"),
                values: Cow::Borrowed(&architecture.ffn_norms[layer]),
            });
        }
        norms.push(NormLayout {
            name: "model.norm.weight".to_owned(),
            values: Cow::Borrowed(&architecture.output_norm),
        });

        Ok(Self {
            architecture: ArchitectureLayout {
                n_layers: architecture.n_layers,
                n_embd: architecture.n_embd,
                n_head: architecture.n_head,
                n_head_kv: architecture.n_head_kv,
                head_dim: architecture.head_dim,
                n_ff: architecture.n_ff,
                n_ctx: architecture.n_ctx,
                vocab: architecture.vocab,
                rope_theta: architecture.rope_theta,
                rms_eps: architecture.rms_eps,
            },
            matrices,
            norms,
        })
    }
}

fn validate_architecture(
    architecture: &TiedSwiGluTrainingArchitecture,
) -> Result<(), CampaignArtifactError> {
    if architecture.n_layers == 0
        || architecture.n_embd == 0
        || architecture.n_head == 0
        || architecture.n_head_kv == 0
        || architecture.head_dim == 0
        || architecture.n_ff == 0
        || architecture.vocab == 0
        || architecture.n_ctx == 0
    {
        return Err(invalid("training architecture contains a zero dimension"));
    }
    if architecture.attn_norms.len() != architecture.n_layers
        || architecture.ffn_norms.len() != architecture.n_layers
        || architecture.output_norm.len() != architecture.n_embd
    {
        return Err(invalid(
            "training architecture has incomplete RMSNorm vectors",
        ));
    }
    for (kind, norms) in [
        ("attention", &architecture.attn_norms),
        ("feed-forward", &architecture.ffn_norms),
    ] {
        for (layer, norm) in norms.iter().enumerate() {
            if norm.len() != architecture.n_embd || norm.iter().any(|value| !value.is_finite()) {
                return Err(invalid(format!(
                    "{kind} RMSNorm at layer {layer} has invalid length or value"
                )));
            }
        }
    }
    if architecture
        .output_norm
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(invalid("final RMSNorm contains a nonfinite value"));
    }
    if !architecture.rope_theta.is_finite()
        || architecture.rope_theta <= 0.0
        || !architecture.rms_eps.is_finite()
        || architecture.rms_eps <= 0.0
    {
        return Err(invalid(
            "training architecture has invalid RoPE or RMSNorm constants",
        ));
    }
    Ok(())
}

impl<'a> ArtifactExportPlan<'a> {
    fn build(
        raw_hf_config_json: &str,
        layout: ModelLayout<'a>,
        salt_planes: usize,
        provenance: ArtifactProvenance,
        growth: &GrowthReceipt,
    ) -> Result<Self, CampaignArtifactError> {
        if !(1..=3).contains(&salt_planes) {
            return Err(invalid(format!(
                "SALT plane count must be in 1..=3, got {salt_planes}"
            )));
        }
        if growth.new_width
            != u64::try_from(layout.architecture.n_ff)
                .map_err(|_| invalid("model intermediate width exceeds u64"))?
        {
            return Err(invalid(
                "growth receipt new width does not match trained model intermediate width",
            ));
        }
        let (canonical_hf_config_json, architecture_name) =
            canonical_patched_config(raw_hf_config_json, &layout.architecture, growth)?;
        let growth_json = growth.canonical_json()?;
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.alignment".to_owned(),
            GgufValue::U32(GGUF_ALIGNMENT),
        );
        metadata.insert(
            "general.architecture".to_owned(),
            GgufValue::String(architecture_name),
        );
        metadata.insert(
            SALT_GGUF_FORMAT_KEY.to_owned(),
            GgufValue::String(SALT_GGUF_FORMAT_VALUE.to_owned()),
        );
        metadata.insert(
            TRAINING_SALT_FORMAT_KEY.to_owned(),
            GgufValue::String(TRAINING_SALT_FORMAT_VALUE.to_owned()),
        );
        metadata.insert(
            TRAINING_SALT_HF_CONFIG_KEY.to_owned(),
            GgufValue::String(canonical_hf_config_json),
        );
        metadata.insert(
            TRAINING_SALT_PLANES_KEY.to_owned(),
            GgufValue::U32(salt_planes as u32),
        );
        metadata.insert(
            TRAINING_SALT_PLAN_FINGERPRINT_KEY.to_owned(),
            GgufValue::String(hex_digest(provenance.plan_fingerprint)),
        );
        metadata.insert(
            TRAINING_SALT_SOURCE_MODEL_DIGEST_KEY.to_owned(),
            GgufValue::String(hex_digest(provenance.source_model_digest)),
        );
        metadata.insert(
            TRAINING_SALT_INITIAL_STUDENT_DIGEST_KEY.to_owned(),
            GgufValue::String(hex_digest(provenance.initial_student_digest)),
        );
        metadata.insert(
            TRAINING_SALT_COMPLETED_STEP_KEY.to_owned(),
            GgufValue::U64(provenance.completed_step),
        );
        metadata.insert(
            TRAINING_SALT_GROWTH_RECEIPT_KEY.to_owned(),
            GgufValue::String(growth_json),
        );

        let mut tensor_specs = Vec::with_capacity(layout.matrices.len() + layout.norms.len());
        for matrix in &layout.matrices {
            tensor_specs.push(GgufTensorSpec {
                name: matrix.name.clone(),
                dims: vec![matrix.cols as u64, matrix.rows as u64],
                ggml_type: GGML_TYPE_TRITIUM_SALT,
                data_len: salt_tensor_data_len(matrix.rows, matrix.cols, salt_planes)?,
            });
        }
        for norm in &layout.norms {
            let elements = u64::try_from(norm.values.len())
                .map_err(|_| invalid("RMSNorm length exceeds u64"))?;
            let data_len = elements
                .checked_mul(4)
                .ok_or_else(|| invalid("RMSNorm byte length overflows u64"))?;
            tensor_specs.push(GgufTensorSpec {
                name: norm.name.clone(),
                dims: vec![elements],
                ggml_type: GGML_TYPE_F32,
                data_len,
            });
        }

        Ok(Self {
            metadata,
            tensor_specs,
            matrices: layout.matrices,
            norms: layout.norms,
            salt_planes,
            completed_step: provenance.completed_step,
        })
    }
}

trait MasterSource {
    fn len(&self) -> usize;
    fn completed_step(&self) -> u64;
    fn geometry(&self, index: usize) -> Result<(usize, usize, usize), CampaignArtifactError>;
    fn master(&self, index: usize) -> Result<&[f32], CampaignArtifactError>;
}

#[cfg(feature = "cuda")]
impl MasterSource for HostOffloadTrainer<'_> {
    fn len(&self) -> usize {
        HostOffloadTrainer::len(self)
    }

    fn completed_step(&self) -> u64 {
        HostOffloadTrainer::completed_step(self)
    }

    fn geometry(&self, index: usize) -> Result<(usize, usize, usize), CampaignArtifactError> {
        let metadata = HostOffloadTrainer::parameter_metadata(self, index).map_err(|error| {
            invalid(format!("read trainer parameter {index} metadata: {error}"))
        })?;
        Ok((metadata.rows, metadata.cols, metadata.salt_planes))
    }

    fn master(&self, index: usize) -> Result<&[f32], CampaignArtifactError> {
        HostOffloadTrainer::master(self, index)
            .map_err(|error| invalid(format!("read trainer parameter {index} master: {error}")))
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn export_campaign_artifact(
    output: &Path,
    raw_hf_config_json: &str,
    model: &TiedSwiGluTrainingModel,
    trainer: &HostOffloadTrainer<'_>,
    salt_planes: usize,
    provenance: ArtifactProvenance,
    growth: &GrowthReceipt,
) -> Result<CampaignArtifactSummary, CampaignArtifactError> {
    let layout = ModelLayout::from_model(model)?;
    let plan =
        ArtifactExportPlan::build(raw_hf_config_json, layout, salt_planes, provenance, growth)?;
    export_from_source(output, &plan, trainer)
}

/// Rebuild and verify a previously published terminal campaign artifact.
///
/// The expected package is streamed into a hash-and-count sink, so verification
/// retains no serialized model bytes. Export statistics are recomputed from the
/// terminal masters and are therefore identical to a fresh deterministic export.
#[cfg(feature = "cuda")]
pub(crate) fn verify_campaign_artifact(
    existing: &Path,
    raw_hf_config_json: &str,
    model: &TiedSwiGluTrainingModel,
    trainer: &HostOffloadTrainer<'_>,
    salt_planes: usize,
    provenance: ArtifactProvenance,
    growth: &GrowthReceipt,
) -> Result<CampaignArtifactSummary, CampaignArtifactError> {
    let layout = ModelLayout::from_model(model)?;
    let plan =
        ArtifactExportPlan::build(raw_hf_config_json, layout, salt_planes, provenance, growth)?;
    verify_from_source(existing, &plan, trainer)
}

fn export_from_source(
    output: &Path,
    plan: &ArtifactExportPlan<'_>,
    source: &impl MasterSource,
) -> Result<CampaignArtifactSummary, CampaignArtifactError> {
    validate_source(plan, source)?;
    let mut publish = AtomicArtifact::create(output)?;
    let stats = write_artifact(publish.take_file()?, plan, source)?;
    let temporary = publish.temporary().to_owned();
    {
        let file = publish.set_finished_file(stats.writer)?;
        file.sync_all()
            .map_err(|source| io_error("fsync temporary", &temporary, source))?;
    }
    publish.commit()?;
    let package = MeasuredPackage::from_file(output)?;
    Ok(stats.stats.with_package(package))
}

fn verify_from_source(
    existing: &Path,
    plan: &ArtifactExportPlan<'_>,
    source: &impl MasterSource,
) -> Result<CampaignArtifactSummary, CampaignArtifactError> {
    let measured = MeasuredPackage::from_file(existing)?;
    let rebuilt = write_artifact(CountingPackageWriter::default(), plan, source)?;
    let (expected_id, expected_bytes) = rebuilt.writer.finalize();
    if measured.id() != expected_id || measured.physical_bytes() != expected_bytes {
        return Err(invalid(format!(
            "existing artifact package mismatch: rebuilt {expected_id} ({expected_bytes} bytes), measured {} ({} bytes)",
            measured.id(),
            measured.physical_bytes()
        )));
    }
    Ok(rebuilt.stats.with_package(measured))
}

#[derive(Debug, Default)]
struct CountingPackageWriter {
    hasher: PackageHasher,
    physical_bytes: u64,
}

impl CountingPackageWriter {
    fn finalize(self) -> (PackageId, u64) {
        (self.hasher.finalize(), self.physical_bytes)
    }
}

impl Write for CountingPackageWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let chunk_bytes = u64::try_from(bytes.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "campaign artifact write length exceeds u64",
            )
        })?;
        self.physical_bytes = self
            .physical_bytes
            .checked_add(chunk_bytes)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "campaign artifact byte count overflows u64",
                )
            })?;
        self.hasher.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn validate_source(
    plan: &ArtifactExportPlan<'_>,
    source: &impl MasterSource,
) -> Result<(), CampaignArtifactError> {
    if source.completed_step() != plan.completed_step {
        return Err(invalid(format!(
            "trainer completed step {} does not match artifact step {}",
            source.completed_step(),
            plan.completed_step
        )));
    }
    if source.len() != plan.matrices.len() {
        return Err(invalid(format!(
            "trainer has {} parameters, artifact layout has {}",
            source.len(),
            plan.matrices.len()
        )));
    }
    for (index, matrix) in plan.matrices.iter().enumerate() {
        let geometry = source.geometry(index)?;
        if geometry != (matrix.rows, matrix.cols, plan.salt_planes) {
            return Err(invalid(format!(
                "trainer parameter {index} geometry {:?} does not match artifact ({}, {}, {})",
                geometry, matrix.rows, matrix.cols, plan.salt_planes
            )));
        }
        let expected = matrix
            .rows
            .checked_mul(matrix.cols)
            .ok_or_else(|| invalid(format!("parameter {} shape overflows usize", matrix.name)))?;
        let actual = source.master(index)?.len();
        if actual != expected {
            return Err(invalid(format!(
                "trainer parameter {} master has {actual} elements, expected {expected}",
                matrix.name
            )));
        }
    }
    Ok(())
}

struct WriteResult<W> {
    writer: W,
    stats: ExportStats,
}

fn write_artifact<W: Write>(
    writer: W,
    plan: &ArtifactExportPlan<'_>,
    source: &impl MasterSource,
) -> Result<WriteResult<W>, CampaignArtifactError> {
    validate_source(plan, source)?;
    let mut writer =
        GgufStreamWriter::new(writer, GGUF_VERSION, &plan.metadata, &plan.tensor_specs)?;
    let mut aggregate = ExportStats::default();
    for (tensor_index, matrix) in plan.matrices.iter().enumerate() {
        let master = source.master(tensor_index)?;
        for row in master.chunks_exact(matrix.cols) {
            let (salt_row, stats) = export_training_salt_row(row, plan.salt_planes)?;
            aggregate.add(stats)?;
            let packed = pack_salt_row(&salt_row)
                .map_err(TrainingSaltExportError::from)
                .map_err(CampaignArtifactError::from)?;
            writer.write_tensor_chunk(tensor_index, &packed)?;
        }
    }
    let norm_base = plan.matrices.len();
    for (norm_index, norm) in plan.norms.iter().enumerate() {
        let tensor_index = norm_base + norm_index;
        for value in norm.values.iter() {
            writer.write_tensor_chunk(tensor_index, &value.to_bits().to_le_bytes())?;
        }
    }
    Ok(WriteResult {
        writer: writer.finish()?,
        stats: aggregate,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct ExportStats {
    scale_max_abs_error: f64,
    reconstruction_max_abs_delta: f64,
    reconstruction_squared_error_sum: f64,
    reconstruction_element_count: u64,
}

impl ExportStats {
    fn add(&mut self, row: TrainingSaltExportStats) -> Result<(), CampaignArtifactError> {
        self.scale_max_abs_error = self.scale_max_abs_error.max(row.scale_max_abs_error);
        self.reconstruction_max_abs_delta = self
            .reconstruction_max_abs_delta
            .max(row.reconstruction_max_abs_delta);
        self.reconstruction_squared_error_sum += row.reconstruction_squared_error_sum;
        if !self.reconstruction_squared_error_sum.is_finite() {
            return Err(invalid("deployment projection squared error is nonfinite"));
        }
        self.reconstruction_element_count = self
            .reconstruction_element_count
            .checked_add(row.reconstruction_element_count)
            .ok_or_else(|| invalid("deployment projection element count overflows u64"))?;
        Ok(())
    }

    fn with_package(self, package: MeasuredPackage) -> CampaignArtifactSummary {
        CampaignArtifactSummary {
            package,
            scale_max_abs_error: self.scale_max_abs_error,
            reconstruction_max_abs_delta: self.reconstruction_max_abs_delta,
            reconstruction_squared_error_sum: self.reconstruction_squared_error_sum,
            reconstruction_element_count: self.reconstruction_element_count,
        }
    }
}

fn canonical_patched_config(
    raw: &str,
    architecture: &ArchitectureLayout,
    growth: &GrowthReceipt,
) -> Result<(String, String), CampaignArtifactError> {
    let mut value: Value = serde_json::from_str(raw).map_err(CampaignArtifactError::Json)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("HuggingFace config root must be an object"))?;
    let source_width = object
        .get("intermediate_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("HuggingFace config has no integer intermediate_size"))?;
    if source_width != growth.old_width {
        return Err(invalid(format!(
            "source config intermediate_size {source_width} does not match growth old width {}",
            growth.old_width
        )));
    }
    object.insert(
        "intermediate_size".to_owned(),
        Value::from(growth.new_width),
    );
    let vocab = object
        .get("vocab_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("HuggingFace config has no integer vocab_size"))?;
    if vocab
        != u64::try_from(architecture.vocab)
            .map_err(|_| invalid("trained model vocabulary exceeds u64"))?
    {
        return Err(invalid(format!(
            "source config vocab_size {vocab} does not match trained model vocabulary {}",
            architecture.vocab
        )));
    }
    let canonical = canonical_json_value(value);
    let encoded = serde_json::to_string(&canonical).map_err(CampaignArtifactError::Json)?;
    let (config, spec) = ModelConfig::from_hf_config(&encoded).map_err(|error| {
        invalid(format!(
            "patched HuggingFace config is unsupported: {error}"
        ))
    })?;
    if !spec.tied_embeddings || spec.mlp != tritium_nn::MlpKind::SwiGlu {
        return Err(invalid(
            "patched HuggingFace config must use tied embeddings and SwiGLU",
        ));
    }
    let parsed = [
        (
            "num_hidden_layers",
            u64::from(config.n_layers),
            architecture.n_layers as u64,
        ),
        (
            "hidden_size",
            u64::from(config.n_embd),
            architecture.n_embd as u64,
        ),
        (
            "num_attention_heads",
            u64::from(config.n_head),
            architecture.n_head as u64,
        ),
        (
            "num_key_value_heads",
            u64::from(config.n_head_kv),
            architecture.n_head_kv as u64,
        ),
        (
            "head_dim",
            u64::from(config.head_dim),
            architecture.head_dim as u64,
        ),
        (
            "intermediate_size",
            u64::from(config.n_ff),
            architecture.n_ff as u64,
        ),
        (
            "max_position_embeddings",
            u64::from(config.n_ctx),
            architecture.n_ctx as u64,
        ),
    ];
    if let Some((name, actual, expected)) = parsed
        .into_iter()
        .find(|(_, actual, expected)| actual != expected)
    {
        return Err(invalid(format!(
            "patched config {name} is {actual}, trained model requires {expected}"
        )));
    }
    if config.rope_theta.to_bits() != architecture.rope_theta.to_bits()
        || config.rms_eps.to_bits() != architecture.rms_eps.to_bits()
    {
        return Err(invalid(
            "patched config RoPE or RMSNorm constant differs from trained model",
        ));
    }
    Ok((encoded, config.arch))
}

fn canonical_json_value(value: Value) -> Value {
    match value {
        Value::Array(values) => {
            Value::Array(values.into_iter().map(canonical_json_value).collect())
        }
        Value::Object(values) => {
            let mut ordered: Vec<_> = values.into_iter().collect();
            ordered.sort_by(|left, right| left.0.cmp(&right.0));
            let mut object = Map::new();
            for (key, value) in ordered {
                object.insert(key, canonical_json_value(value));
            }
            Value::Object(object)
        }
        scalar => scalar,
    }
}

fn salt_tensor_data_len(
    rows: usize,
    cols: usize,
    planes: usize,
) -> Result<u64, CampaignArtifactError> {
    let plane_bytes = num_blocks(cols)
        .checked_mul(TQ2_0_BLOCK_BYTES)
        .ok_or_else(|| invalid("SALT plane byte length overflows usize"))?;
    let row_bytes = planes
        .checked_mul(plane_bytes)
        .and_then(|bytes| bytes.checked_add(SALT_HEADER_BYTES))
        .ok_or_else(|| invalid("SALT row byte length overflows usize"))?;
    let total = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| invalid("SALT tensor byte length overflows usize"))?;
    u64::try_from(total).map_err(|_| invalid("SALT tensor byte length exceeds u64"))
}

struct AtomicArtifact {
    destination: PathBuf,
    temporary: PathBuf,
    file: Option<File>,
    committed: bool,
}

impl AtomicArtifact {
    fn create(destination: &Path) -> Result<Self, CampaignArtifactError> {
        if destination.file_name().is_none() {
            return Err(invalid("artifact output path has no file name"));
        }
        match std::fs::symlink_metadata(destination) {
            Ok(_) => return Err(invalid("artifact output already exists")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(source) => return Err(io_error("inspect output", destination, source)),
        }
        let parent = output_parent(destination);
        let parent_metadata = std::fs::symlink_metadata(parent)
            .map_err(|source| io_error("inspect output directory", parent, source))?;
        if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
            return Err(invalid(
                "artifact output parent must be an existing non-symlink directory",
            ));
        }
        let file_name = destination
            .file_name()
            .expect("validated artifact output has a file name");
        for _ in 0..TEMP_ATTEMPTS {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let mut temporary_name = OsString::from(".");
            temporary_name.push(file_name);
            temporary_name.push(format!(".tmp-{}-{sequence}", std::process::id()));
            let temporary = parent.join(temporary_name);
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
            {
                Ok(file) => {
                    return Ok(Self {
                        destination: destination.to_owned(),
                        temporary,
                        file: Some(file),
                        committed: false,
                    });
                }
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(source) => return Err(io_error("create temporary", &temporary, source)),
            }
        }
        Err(invalid(
            "could not reserve a unique temporary artifact path",
        ))
    }

    fn temporary(&self) -> &Path {
        &self.temporary
    }

    fn take_file(&mut self) -> Result<File, CampaignArtifactError> {
        self.file
            .take()
            .ok_or_else(|| invalid("temporary artifact file is not available"))
    }

    fn set_finished_file(&mut self, file: File) -> Result<&File, CampaignArtifactError> {
        if self.file.replace(file).is_some() {
            return Err(invalid("temporary artifact file was finished twice"));
        }
        self.file
            .as_ref()
            .ok_or_else(|| invalid("finished artifact file is not available"))
    }

    fn commit(mut self) -> Result<(), CampaignArtifactError> {
        drop(self.file.take());
        match std::fs::symlink_metadata(&self.destination) {
            Ok(_) => return Err(invalid("artifact output appeared before atomic publish")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(source) => return Err(io_error("inspect output", &self.destination, source)),
        }
        // A same-directory hard link is an atomic no-replace publication: unlike
        // `rename`, it fails if another process creates the destination after the
        // preflight check. Both names reference the already-fsynced inode until
        // the private temporary name is removed.
        std::fs::hard_link(&self.temporary, &self.destination).map_err(|source| {
            io_error(
                "atomically publish without replacement",
                &self.destination,
                source,
            )
        })?;
        std::fs::remove_file(&self.temporary).map_err(|source| {
            io_error("remove published temporary link", &self.temporary, source)
        })?;
        self.committed = true;
        #[cfg(unix)]
        File::open(output_parent(&self.destination))
            .and_then(|directory| directory.sync_all())
            .map_err(|source| {
                io_error(
                    "fsync output directory",
                    output_parent(&self.destination),
                    source,
                )
            })?;
        Ok(())
    }
}

impl Drop for AtomicArtifact {
    fn drop(&mut self) {
        if !self.committed {
            drop(self.file.take());
            let _ = std::fs::remove_file(&self.temporary);
        }
    }
}

fn output_parent(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

fn invalid(reason: impl Into<String>) -> CampaignArtifactError {
    CampaignArtifactError::InvalidInput(reason.into())
}

fn io_error(
    operation: &'static str,
    path: impl AsRef<Path>,
    source: io::Error,
) -> CampaignArtifactError {
    CampaignArtifactError::Io {
        operation,
        path: path.as_ref().to_owned(),
        source,
    }
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("write to String cannot fail");
    }
    encoded
}

fn parse_hex_digest(name: &str, encoded: &str) -> Result<[u8; 32], CampaignArtifactError> {
    if encoded.len() != 64 || !encoded.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(format!(
            "{name} must be exactly 64 hexadecimal characters"
        )));
    }
    if encoded.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err(invalid(format!("{name} must use lowercase hexadecimal")));
    }
    let mut digest = [0_u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        let start = index * 2;
        *byte = u8::from_str_radix(&encoded[start..start + 2], 16)
            .map_err(|_| invalid(format!("{name} contains invalid hexadecimal")))?;
    }
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_format::{GgufValue, read_gguf};

    #[derive(Clone)]
    struct FakeSource {
        step: u64,
        masters: Vec<Vec<f32>>,
        geometry: Vec<(usize, usize, usize)>,
    }

    impl MasterSource for FakeSource {
        fn len(&self) -> usize {
            self.masters.len()
        }

        fn completed_step(&self) -> u64 {
            self.step
        }

        fn geometry(&self, index: usize) -> Result<(usize, usize, usize), CampaignArtifactError> {
            self.geometry
                .get(index)
                .copied()
                .ok_or_else(|| invalid("fake geometry missing"))
        }

        fn master(&self, index: usize) -> Result<&[f32], CampaignArtifactError> {
            self.masters
                .get(index)
                .map(Vec::as_slice)
                .ok_or_else(|| invalid("fake master missing"))
        }
    }

    fn provenance() -> ArtifactProvenance {
        ArtifactProvenance::new(&"01".repeat(32), [2; 32], [3; 32], 7).expect("provenance")
    }

    fn growth() -> GrowthReceipt {
        GrowthReceipt::new(2, 3, 99, vec![0, 1, 0], vec![2, 1]).expect("growth")
    }

    fn config() -> &'static str {
        r#"{
            "tie_word_embeddings": true,
            "vocab_size": 2,
            "rms_norm_eps": 0.00001,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "num_attention_heads": 1,
            "model_type": "test",
            "max_position_embeddings": 8,
            "intermediate_size": 2,
            "hidden_size": 2,
            "hidden_act": "silu",
            "head_dim": 2,
            "rope_theta": 10000,
            "nested": {"z": 1, "a": 2}
        }"#
    }

    fn layout() -> ModelLayout<'static> {
        let matrices = [
            ("model.embed_tokens.weight", 2, 2),
            ("model.layers.0.self_attn.q_proj.weight", 2, 2),
            ("model.layers.0.self_attn.k_proj.weight", 2, 2),
            ("model.layers.0.self_attn.v_proj.weight", 2, 2),
            ("model.layers.0.self_attn.o_proj.weight", 2, 2),
            ("model.layers.0.mlp.gate_proj.weight", 3, 2),
            ("model.layers.0.mlp.up_proj.weight", 3, 2),
            ("model.layers.0.mlp.down_proj.weight", 2, 3),
        ]
        .into_iter()
        .map(|(name, rows, cols)| MatrixLayout {
            name: name.to_owned(),
            rows,
            cols,
        })
        .collect();
        ModelLayout {
            architecture: ArchitectureLayout {
                n_layers: 1,
                n_embd: 2,
                n_head: 1,
                n_head_kv: 1,
                head_dim: 2,
                n_ff: 3,
                n_ctx: 8,
                vocab: 2,
                rope_theta: 10_000.0,
                rms_eps: 0.00001,
            },
            matrices,
            norms: vec![
                NormLayout {
                    name: "model.layers.0.input_layernorm.weight".to_owned(),
                    values: Cow::Owned(vec![1.0, 2.0]),
                },
                NormLayout {
                    name: "model.layers.0.post_attention_layernorm.weight".to_owned(),
                    values: Cow::Owned(vec![2.0, 3.0]),
                },
                NormLayout {
                    name: "model.norm.weight".to_owned(),
                    values: Cow::Owned(vec![3.0, 4.0]),
                },
            ],
        }
    }

    fn plan() -> ArtifactExportPlan<'static> {
        ArtifactExportPlan::build(config(), layout(), 2, provenance(), &growth()).expect("plan")
    }

    fn source() -> FakeSource {
        let geometry = vec![
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (3, 2, 2),
            (3, 2, 2),
            (2, 3, 2),
        ];
        let masters = geometry
            .iter()
            .enumerate()
            .map(|(tensor, &(rows, cols, _))| {
                (0..rows * cols)
                    .map(|index| ((tensor * 11 + index + 1) as f32) * 0.03125 - 0.75)
                    .collect()
            })
            .collect();
        FakeSource {
            step: 7,
            masters,
            geometry,
        }
    }

    #[test]
    fn plan_has_exact_metadata_and_tensor_contract() {
        let plan = plan();
        assert_eq!(
            plan.metadata.get(SALT_GGUF_FORMAT_KEY),
            Some(&GgufValue::String("salt-rows.v1".to_owned()))
        );
        assert_eq!(
            plan.metadata.get(TRAINING_SALT_FORMAT_KEY),
            Some(&GgufValue::String("tied-swiglu.v1".to_owned()))
        );
        assert_eq!(
            plan.metadata.get(TRAINING_SALT_PLANES_KEY),
            Some(&GgufValue::U32(2))
        );
        assert_eq!(
            plan.metadata.get(TRAINING_SALT_COMPLETED_STEP_KEY),
            Some(&GgufValue::U64(7))
        );
        assert_eq!(
            plan.metadata.get(TRAINING_SALT_PLAN_FINGERPRINT_KEY),
            Some(&GgufValue::String("01".repeat(32)))
        );
        assert_eq!(
            plan.metadata.get(TRAINING_SALT_GROWTH_RECEIPT_KEY),
            Some(&GgufValue::String(
                r#"{"algorithm":"net2wider.intermediate-swiglu.splitmix64.v1","old_width":2,"new_width":3,"seed":99,"source_indices":[0,1,0],"replication_counts":[2,1]}"#
                    .to_owned()
            ))
        );
        assert_eq!(
            plan.metadata.get("general.architecture"),
            Some(&GgufValue::String("test".to_owned()))
        );
        let config = plan
            .metadata
            .get(TRAINING_SALT_HF_CONFIG_KEY)
            .and_then(GgufValue::as_str)
            .expect("config metadata");
        assert!(!config.contains(char::is_whitespace));
        assert!(config.contains(r#""intermediate_size":3"#));
        assert!(config.contains(r#""nested":{"a":2,"z":1}"#));

        let row_len = SALT_HEADER_BYTES + 2 * TQ2_0_BLOCK_BYTES;
        assert_eq!(plan.tensor_specs.len(), 11);
        assert_eq!(
            plan.tensor_specs[0],
            GgufTensorSpec {
                name: "model.embed_tokens.weight".to_owned(),
                dims: vec![2, 2],
                ggml_type: GGML_TYPE_TRITIUM_SALT,
                data_len: (2 * row_len) as u64,
            }
        );
        assert_eq!(
            plan.tensor_specs[8..]
                .iter()
                .map(|tensor| tensor.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.norm.weight",
            ]
        );
    }

    #[test]
    fn streaming_bytes_are_deterministic_and_stats_cover_every_weight() {
        let plan = plan();
        let source = source();
        let first = write_artifact(Vec::new(), &plan, &source).expect("first export");
        let second = write_artifact(Vec::new(), &plan, &source).expect("second export");
        assert_eq!(first.writer, second.writer);
        assert_eq!(first.stats, second.stats);
        assert_eq!(first.stats.reconstruction_element_count, 38);
        assert!(first.stats.reconstruction_squared_error_sum.is_finite());

        let parsed = read_gguf(&first.writer).expect("read exported GGUF");
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.tensors.len(), 11);
        assert_eq!(parsed.tensors[0].dims, vec![2, 2]);
        assert_eq!(parsed.tensors[0].ggml_type, GGML_TYPE_TRITIUM_SALT);
        assert!(parsed.tensor("lm_head.weight").is_none());
        let norm = parsed.tensor("model.norm.weight").expect("final norm");
        let start = usize::try_from(parsed.tensor_data_offset + norm.offset).expect("offset");
        assert_eq!(
            &first.writer[start..start + 8],
            &[3.0f32.to_le_bytes(), 4.0f32.to_le_bytes()].concat()
        );
    }

    #[test]
    fn exported_bytes_load_as_a_self_contained_training_salt_model() {
        let written = write_artifact(Vec::new(), &plan(), &source()).expect("export artifact");
        let (config, weights) = tritium_nn::ModelWeights::load_training_salt_gguf(&written.writer)
            .expect("load self-contained training SALT artifact");
        assert_eq!(config.n_ff, 3);
        assert_eq!(weights.vocab, 2);
        assert_eq!(weights.layers.len(), 1);
        assert!(weights.lm_head.is_none(), "embedding must remain tied");
    }

    #[test]
    fn rejects_provenance_growth_config_and_source_mismatches() {
        assert!(ArtifactProvenance::new(&"00".repeat(32), [2; 32], [3; 32], 0).is_err());
        assert!(ArtifactProvenance::new(&"AA".repeat(32), [2; 32], [3; 32], 0).is_err());
        assert!(GrowthReceipt::new(2, 3, 0, vec![0, 1, 0], vec![1, 2]).is_err());

        let bad_config = config().replace("\"intermediate_size\": 2", "\"intermediate_size\": 4");
        assert!(
            ArtifactExportPlan::build(&bad_config, layout(), 2, provenance(), &growth()).is_err()
        );
        assert!(ArtifactExportPlan::build(config(), layout(), 0, provenance(), &growth()).is_err());

        let plan = plan();
        let mut source = source();
        source.step = 8;
        assert!(validate_source(&plan, &source).is_err());
        source.step = 7;
        source.geometry[0].2 = 1;
        assert!(validate_source(&plan, &source).is_err());
    }

    #[test]
    fn existing_output_is_rejected_without_mutation() {
        let root = std::env::temp_dir().join(format!(
            "tritium-artifact-test-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).expect("create test directory");
        let output = root.join("model.gguf");
        std::fs::write(&output, b"keep").expect("write sentinel");
        assert!(AtomicArtifact::create(&output).is_err());
        assert_eq!(std::fs::read(&output).expect("read sentinel"), b"keep");
        std::fs::remove_dir_all(root).expect("remove test directory");
    }

    #[test]
    fn durable_publish_reports_exact_file_identity_and_size() {
        let root = std::env::temp_dir().join(format!(
            "tritium-artifact-publish-test-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).expect("create test directory");
        let output = root.join("model.gguf");
        let summary = export_from_source(&output, &plan(), &source()).expect("publish artifact");
        let bytes = std::fs::read(&output).expect("read published artifact");
        let measured = MeasuredPackage::from_bytes(&bytes).expect("measure published bytes");
        assert_eq!(summary.package(), measured);
        assert_eq!(summary.reconstruction_element_count(), 38);
        assert!(summary.reconstruction_mse().is_finite());
        assert_eq!(
            std::fs::read_dir(&root)
                .expect("read output directory")
                .count(),
            1,
            "temporary artifact must be removed after rename"
        );
        std::fs::remove_dir_all(root).expect("remove test directory");
    }

    #[test]
    fn existing_exact_artifact_verifies_to_the_export_summary() {
        let root = std::env::temp_dir().join(format!(
            "tritium-artifact-verify-test-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).expect("create test directory");
        let output = root.join("model.gguf");
        let plan = plan();
        let source = source();

        let exported = export_from_source(&output, &plan, &source).expect("publish artifact");
        let verified =
            verify_from_source(&output, &plan, &source).expect("verify existing artifact");

        assert_eq!(verified, exported);
        std::fs::remove_dir_all(root).expect("remove test directory");
    }

    #[test]
    fn existing_wrong_or_truncated_artifact_fails_verification() {
        let root = std::env::temp_dir().join(format!(
            "tritium-artifact-corruption-test-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).expect("create test directory");
        let output = root.join("model.gguf");
        let plan = plan();
        let source = source();
        export_from_source(&output, &plan, &source).expect("publish artifact");
        let original = std::fs::read(&output).expect("read artifact");

        let mut wrong = original.clone();
        let last = wrong.last_mut().expect("artifact is nonempty");
        *last ^= 0x80;
        std::fs::write(&output, wrong).expect("write wrong same-length artifact");
        assert!(verify_from_source(&output, &plan, &source).is_err());

        std::fs::write(&output, &original[..original.len() - 1]).expect("write truncated artifact");
        assert!(verify_from_source(&output, &plan, &source).is_err());

        std::fs::remove_dir_all(root).expect("remove test directory");
    }
}
