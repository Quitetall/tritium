//! Reproducible calibration, evaluation, and multi-objective campaign records.

use core::{cmp::Ordering, fmt};
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};
use tritium_format::{ModelId, PackageHasher, PackageId};

const CALIBRATION_MAGIC: [u8; 4] = *b"TCAL";
const CAMPAIGN_MAGIC: [u8; 4] = *b"TCMP";
const EVALUATION_MAGIC: [u8; 4] = *b"TEVL";
const RECIPE_MAGIC: [u8; 4] = *b"TRCP";
const CAMPAIGN_VERSION: u8 = 2;
const PROVENANCE_VERSION: u8 = 1;
const RECIPE_VERSION: u8 = 1;
const CALIBRATION_ID_CONTEXT: &str = "tritium calibration provenance id v1";
const CAMPAIGN_ID_CONTEXT: &str = "tritium campaign ledger id v1";
const EVALUATION_ID_CONTEXT: &str = "tritium evaluation provenance id v1";
const RECIPE_ID_CONTEXT: &str = "tritium conversion recipe id v1";

fn domain_hash(context: &'static str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_derive_key(context);
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

fn fmt_id(f: &mut fmt::Formatter<'_>, prefix: &str, digest: &[u8; 32]) -> fmt::Result {
    f.write_str(prefix)?;
    for byte in digest {
        write!(f, "{byte:02x}")?;
    }
    Ok(())
}

fn write_string(out: &mut Vec<u8>, value: &str) {
    let len = u32::try_from(value.len()).expect("provenance constructor validates string length");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn write_optional_f64(out: &mut Vec<u8>, value: Option<f64>) {
    match value {
        None => out.push(0),
        Some(value) => {
            out.push(1);
            out.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
}

fn write_optional_u64(out: &mut Vec<u8>, value: Option<u64>) {
    match value {
        None => out.push(0),
        Some(value) => {
            out.push(1);
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
}

/// Content identity of an exact calibration provenance record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CalibrationId([u8; 32]);

impl CalibrationId {
    /// Return the raw domain-separated digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for CalibrationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_id(f, "trc1_", &self.0)
    }
}

/// Content identity of an exact evaluation provenance record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvaluationId([u8; 32]);

impl EvaluationId {
    /// Return the raw domain-separated digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for EvaluationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_id(f, "tre1_", &self.0)
    }
}

/// Content identity of an exact conversion recipe record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RecipeId([u8; 32]);

impl RecipeId {
    /// Return the raw domain-separated digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for RecipeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_id(f, "trr1_", &self.0)
    }
}

/// Content identity of an exact canonical campaign ledger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CampaignId([u8; 32]);

impl CampaignId {
    /// Return the raw domain-separated digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for CampaignId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_id(f, "trl1_", &self.0)
    }
}

/// Exact calibration-corpus and tokenizer selection used by a conversion run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CalibrationProvenance {
    dataset: String,
    revision: String,
    sample_digest: [u8; 32],
    tokenizer_digest: [u8; 32],
    sample_count: u64,
    token_count: u64,
    sequence_length: u32,
    seed: u64,
}

impl CalibrationProvenance {
    /// Build a validated calibration provenance record.
    ///
    /// `sample_digest` must cover ordered tokenized samples, including boundaries
    /// and padding. This makes order, filtering, and preprocessing part of identity.
    ///
    /// # Errors
    /// Returns [`CampaignError`] for empty/oversized identifiers or zero counts.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: impl Into<String>,
        revision: impl Into<String>,
        sample_digest: [u8; 32],
        tokenizer_digest: [u8; 32],
        sample_count: u64,
        token_count: u64,
        sequence_length: u32,
        seed: u64,
    ) -> Result<Self, CampaignError> {
        let dataset = dataset.into();
        let revision = revision.into();
        validate_string("dataset", &dataset)?;
        validate_string("revision", &revision)?;
        validate_nonzero("sample_count", sample_count)?;
        validate_nonzero("token_count", token_count)?;
        validate_nonzero("sequence_length", u64::from(sequence_length))?;
        Ok(Self {
            dataset,
            revision,
            sample_digest,
            tokenizer_digest,
            sample_count,
            token_count,
            sequence_length,
            seed,
        })
    }

    /// Dataset or corpus identifier without a local filesystem path.
    pub fn dataset(&self) -> &str {
        &self.dataset
    }

    /// Immutable dataset revision, commit, or snapshot identifier.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Digest of ordered, fully preprocessed calibration samples.
    pub fn sample_digest(&self) -> &[u8; 32] {
        &self.sample_digest
    }

    /// Digest of tokenizer configuration and vocabulary bytes.
    pub fn tokenizer_digest(&self) -> &[u8; 32] {
        &self.tokenizer_digest
    }

    /// Number of calibration sequences.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Number of non-padding calibration tokens.
    pub fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Padded sequence length used by calibration kernels.
    pub fn sequence_length(&self) -> u32 {
        self.sequence_length
    }

    /// Deterministic selection/shuffle seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Versioned canonical bytes, independent of local source paths.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&CALIBRATION_MAGIC);
        out.push(PROVENANCE_VERSION);
        write_string(&mut out, &self.dataset);
        write_string(&mut out, &self.revision);
        out.extend_from_slice(&self.sample_digest);
        out.extend_from_slice(&self.tokenizer_digest);
        out.extend_from_slice(&self.sample_count.to_le_bytes());
        out.extend_from_slice(&self.token_count.to_le_bytes());
        out.extend_from_slice(&self.sequence_length.to_le_bytes());
        out.extend_from_slice(&self.seed.to_le_bytes());
        out
    }

    /// Exact content identity of this calibration record.
    pub fn id(&self) -> CalibrationId {
        CalibrationId(domain_hash(CALIBRATION_ID_CONTEXT, &self.canonical_bytes()))
    }
}

/// Exact evaluation corpus and harness used to score campaign points.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationProvenance {
    suite: String,
    revision: String,
    sample_digest: [u8; 32],
    tokenizer_digest: [u8; 32],
    harness_digest: [u8; 32],
    sample_count: u64,
    token_count: u64,
}

impl EvaluationProvenance {
    /// Build a validated evaluation provenance record.
    ///
    /// `sample_digest` covers ordered, rendered model inputs and targets;
    /// `harness_digest` covers scoring code, metric configuration, and captured
    /// runtime environment (GPU, driver, clocks, and contention) when system
    /// metrics are recorded.
    ///
    /// # Errors
    /// Returns [`CampaignError`] for empty/oversized identifiers or zero counts.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        suite: impl Into<String>,
        revision: impl Into<String>,
        sample_digest: [u8; 32],
        tokenizer_digest: [u8; 32],
        harness_digest: [u8; 32],
        sample_count: u64,
        token_count: u64,
    ) -> Result<Self, CampaignError> {
        let suite = suite.into();
        let revision = revision.into();
        validate_string("suite", &suite)?;
        validate_string("revision", &revision)?;
        validate_nonzero("sample_count", sample_count)?;
        validate_nonzero("token_count", token_count)?;
        Ok(Self {
            suite,
            revision,
            sample_digest,
            tokenizer_digest,
            harness_digest,
            sample_count,
            token_count,
        })
    }

    /// Evaluation suite or dataset identifier.
    pub fn suite(&self) -> &str {
        &self.suite
    }

    /// Immutable suite revision, commit, or snapshot identifier.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Digest of ordered, fully rendered evaluation samples.
    pub fn sample_digest(&self) -> &[u8; 32] {
        &self.sample_digest
    }

    /// Digest of tokenizer configuration and vocabulary bytes.
    pub fn tokenizer_digest(&self) -> &[u8; 32] {
        &self.tokenizer_digest
    }

    /// Digest of evaluation code, metric configuration, and runtime environment.
    pub fn harness_digest(&self) -> &[u8; 32] {
        &self.harness_digest
    }

    /// Number of scored samples.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Number of scored non-padding tokens.
    pub fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Versioned canonical bytes, independent of local source paths.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&EVALUATION_MAGIC);
        out.push(PROVENANCE_VERSION);
        write_string(&mut out, &self.suite);
        write_string(&mut out, &self.revision);
        out.extend_from_slice(&self.sample_digest);
        out.extend_from_slice(&self.tokenizer_digest);
        out.extend_from_slice(&self.harness_digest);
        out.extend_from_slice(&self.sample_count.to_le_bytes());
        out.extend_from_slice(&self.token_count.to_le_bytes());
        out
    }

    /// Exact content identity of this evaluation record.
    pub fn id(&self) -> EvaluationId {
        EvaluationId(domain_hash(EVALUATION_ID_CONTEXT, &self.canonical_bytes()))
    }
}

/// Conversion implementation, immutable revision, configuration, and replay command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecipeProvenance {
    implementation: String,
    revision: String,
    canonical_config: Vec<u8>,
    command: String,
}

impl RecipeProvenance {
    /// Build a self-contained conversion recipe record.
    ///
    /// `canonical_config` must be the exact portable configuration consumed by
    /// `command`. `revision` should identify a clean source checkout.
    ///
    /// # Errors
    /// Returns [`CampaignError`] for empty/oversized fields.
    pub fn new(
        implementation: impl Into<String>,
        revision: impl Into<String>,
        canonical_config: Vec<u8>,
        command: impl Into<String>,
    ) -> Result<Self, CampaignError> {
        let implementation = implementation.into();
        let revision = revision.into();
        let command = command.into();
        validate_string("implementation", &implementation)?;
        validate_string("revision", &revision)?;
        validate_string("command", &command)?;
        if canonical_config.len() > u32::MAX as usize {
            return Err(CampaignError::FieldTooLong("canonical_config"));
        }
        if canonical_config.is_empty() {
            return Err(CampaignError::EmptyField("canonical_config"));
        }
        Ok(Self {
            implementation,
            revision,
            canonical_config,
            command,
        })
    }

    /// Conversion implementation or binary name.
    pub fn implementation(&self) -> &str {
        &self.implementation
    }

    /// Immutable implementation revision.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Exact portable conversion configuration.
    pub fn canonical_config(&self) -> &[u8] {
        &self.canonical_config
    }

    /// Command that consumes the embedded configuration.
    pub fn command(&self) -> &str {
        &self.command
    }

    /// Versioned canonical recipe bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&RECIPE_MAGIC);
        out.push(RECIPE_VERSION);
        write_string(&mut out, &self.implementation);
        write_string(&mut out, &self.revision);
        let config_len = u32::try_from(self.canonical_config.len())
            .expect("recipe constructor validates config length");
        out.extend_from_slice(&config_len.to_le_bytes());
        out.extend_from_slice(&self.canonical_config);
        write_string(&mut out, &self.command);
        out
    }

    /// Exact content identity of this recipe record.
    pub fn id(&self) -> RecipeId {
        RecipeId(domain_hash(RECIPE_ID_CONTEXT, &self.canonical_bytes()))
    }
}

/// Exact identity and physical size derived from serialized inference bytes.
///
/// Construction derives the digest and byte count together, so callers cannot
/// pair an artifact digest with a separately claimed byte count. Large packages
/// can be measured incrementally with [`Self::from_reader`] or [`Self::from_file`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeasuredPackage {
    id: PackageId,
    physical_bytes: u64,
}

impl MeasuredPackage {
    /// Hash and count an exact serialized inference artifact.
    ///
    /// # Errors
    /// Returns [`CampaignError`] when the artifact is empty or its length cannot
    /// be represented by the version-2 campaign record.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CampaignError> {
        let physical_bytes =
            u64::try_from(bytes.len()).map_err(|_| CampaignError::RecordTooLarge("package"))?;
        validate_nonzero("physical_bytes", physical_bytes)?;
        Ok(Self {
            id: PackageId::from_package_bytes(bytes),
            physical_bytes,
        })
    }

    /// Hash and count an exact serialized inference artifact from a bounded stream.
    ///
    /// The reader is consumed once in order, using a fixed-size buffer. The
    /// resulting identity is byte-for-byte equivalent to passing the same stream
    /// contents to [`Self::from_bytes`].
    ///
    /// # Errors
    /// Returns [`CampaignError::PackageIo`] when the reader fails,
    /// [`CampaignError::ZeroValue`] when it is empty, or
    /// [`CampaignError::RecordTooLarge`] if the exact byte count exceeds `u64`.
    pub fn from_reader(mut reader: impl Read) -> Result<Self, CampaignError> {
        const BUFFER_BYTES: usize = 64 * 1024;

        let mut hasher = PackageHasher::new();
        let mut physical_bytes = 0_u64;
        let mut buffer = [0_u8; BUFFER_BYTES];
        loop {
            let read = match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => {
                    return Err(CampaignError::PackageIo {
                        operation: "read",
                        kind: error.kind(),
                    });
                }
            };
            physical_bytes = checked_package_length(physical_bytes, read)?;
            hasher.update(&buffer[..read]);
        }
        validate_nonzero("physical_bytes", physical_bytes)?;
        Ok(Self {
            id: hasher.finalize(),
            physical_bytes,
        })
    }

    /// Open, hash, and count an exact serialized inference artifact file.
    ///
    /// # Errors
    /// Returns [`CampaignError::PackageIo`] when the file cannot be opened or
    /// read, and the same validation errors as [`Self::from_reader`].
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, CampaignError> {
        let file = File::open(path).map_err(|error| CampaignError::PackageIo {
            operation: "open",
            kind: error.kind(),
        })?;
        Self::from_reader(file)
    }

    /// Content identity of the exact serialized artifact.
    pub fn id(self) -> PackageId {
        self.id
    }

    /// Exact serialized artifact length.
    pub fn physical_bytes(self) -> u64 {
        self.physical_bytes
    }
}

/// Metric used as one axis of campaign Pareto dominance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CampaignObjective {
    /// Minimize exact serialized inference-artifact bytes.
    PhysicalBytes,
    /// Minimize logical allocator bits per source weight.
    LogicalBpw,
    /// Minimize held-out perplexity.
    Perplexity,
    /// Minimize source-vs-converted reconstruction mean squared error.
    ReconstructionMse,
    /// Maximize a pinned end-task score.
    TaskScore,
    /// Maximize measured decode throughput.
    TokensPerSecond,
    /// Minimize peak resident device memory during inference.
    PeakVramBytes,
}

/// Required storage metrics plus optional quality measurements for one artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct CampaignMetrics {
    logical_bpw: f64,
    perplexity: Option<f64>,
    reconstruction_mse: Option<f64>,
    task_score: Option<f64>,
    tokens_per_second: Option<f64>,
    peak_vram_bytes: Option<u64>,
}

impl CampaignMetrics {
    /// Build metrics with the logical allocator rate.
    ///
    /// # Errors
    /// Returns [`CampaignError::InvalidMetric`] for a non-finite/non-positive rate.
    pub fn new(logical_bpw: f64) -> Result<Self, CampaignError> {
        validate_positive_finite("logical_bpw", logical_bpw)?;
        Ok(Self {
            logical_bpw,
            perplexity: None,
            reconstruction_mse: None,
            task_score: None,
            tokens_per_second: None,
            peak_vram_bytes: None,
        })
    }

    /// Add held-out perplexity measured under pinned evaluation provenance.
    ///
    /// # Errors
    /// Returns [`CampaignError::InvalidMetric`] unless perplexity is finite and positive.
    pub fn with_perplexity(mut self, perplexity: f64) -> Result<Self, CampaignError> {
        validate_positive_finite("perplexity", perplexity)?;
        self.perplexity = Some(perplexity);
        Ok(self)
    }

    /// Add source-vs-converted reconstruction mean squared error.
    ///
    /// # Errors
    /// Returns [`CampaignError::InvalidMetric`] unless MSE is finite and non-negative.
    pub fn with_reconstruction_mse(mut self, mse: f64) -> Result<Self, CampaignError> {
        validate_nonnegative_finite("reconstruction_mse", mse)?;
        self.reconstruction_mse = Some(canonicalize_zero(mse));
        Ok(self)
    }

    /// Add a pinned end-task score. Higher values are better.
    ///
    /// # Errors
    /// Returns [`CampaignError::InvalidMetric`] unless the score is finite.
    pub fn with_task_score(mut self, score: f64) -> Result<Self, CampaignError> {
        validate_finite("task_score", score)?;
        self.task_score = Some(canonicalize_zero(score));
        Ok(self)
    }

    /// Add measured decode throughput. Higher values are better.
    ///
    /// # Errors
    /// Returns [`CampaignError::InvalidMetric`] unless throughput is finite and positive.
    pub fn with_tokens_per_second(mut self, throughput: f64) -> Result<Self, CampaignError> {
        validate_positive_finite("tokens_per_second", throughput)?;
        self.tokens_per_second = Some(throughput);
        Ok(self)
    }

    /// Add measured peak inference device memory.
    ///
    /// # Errors
    /// Returns [`CampaignError::InvalidMetric`] when the measurement is zero.
    pub fn with_peak_vram_bytes(mut self, bytes: u64) -> Result<Self, CampaignError> {
        if bytes == 0 {
            return Err(CampaignError::InvalidMetric("peak_vram_bytes"));
        }
        self.peak_vram_bytes = Some(bytes);
        Ok(self)
    }

    /// Logical bits per source weight assigned by the allocator.
    pub fn logical_bpw(&self) -> f64 {
        self.logical_bpw
    }

    /// Held-out perplexity, when measured.
    pub fn perplexity(&self) -> Option<f64> {
        self.perplexity
    }

    /// Reconstruction MSE, when measured.
    pub fn reconstruction_mse(&self) -> Option<f64> {
        self.reconstruction_mse
    }

    /// Pinned end-task score, when measured.
    pub fn task_score(&self) -> Option<f64> {
        self.task_score
    }

    /// Decode throughput, when measured.
    pub fn tokens_per_second(&self) -> Option<f64> {
        self.tokens_per_second
    }

    /// Peak inference device memory, when measured.
    pub fn peak_vram_bytes(&self) -> Option<u64> {
        self.peak_vram_bytes
    }
}

/// One immutable converted artifact and its measured campaign metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct CampaignPoint {
    source_model_id: ModelId,
    model_id: ModelId,
    package: MeasuredPackage,
    recipe: RecipeProvenance,
    calibration_id: CalibrationId,
    evaluation_id: EvaluationId,
    metrics: CampaignMetrics,
}

impl CampaignPoint {
    /// Record identities and measurements for one conversion recipe output.
    ///
    /// Construction alone does not prove provenance. [`CampaignLedger::add`]
    /// performs the trust-boundary validation against its pinned identities.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_model_id: ModelId,
        model_id: ModelId,
        package: MeasuredPackage,
        recipe: RecipeProvenance,
        calibration_id: CalibrationId,
        evaluation_id: EvaluationId,
        metrics: CampaignMetrics,
    ) -> Self {
        Self {
            source_model_id,
            model_id,
            package,
            recipe,
            calibration_id,
            evaluation_id,
            metrics,
        }
    }

    /// Full-precision source model identity.
    pub fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Converted model's semantic identity.
    pub fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// Exact converted package identity.
    pub fn package_id(&self) -> PackageId {
        self.package.id()
    }

    /// Exact serialized inference-artifact size.
    pub fn physical_bytes(&self) -> u64 {
        self.package.physical_bytes()
    }

    /// Embedded conversion recipe.
    pub fn recipe(&self) -> &RecipeProvenance {
        &self.recipe
    }

    /// Conversion recipe identity.
    pub fn recipe_id(&self) -> RecipeId {
        self.recipe.id()
    }

    /// Calibration provenance identity.
    pub fn calibration_id(&self) -> CalibrationId {
        self.calibration_id
    }

    /// Evaluation provenance identity.
    pub fn evaluation_id(&self) -> EvaluationId {
        self.evaluation_id
    }

    /// Measured campaign metrics.
    pub fn metrics(&self) -> &CampaignMetrics {
        &self.metrics
    }
}

/// Apples-to-apples campaign ledger pinned to one source, calibration, and evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct CampaignLedger {
    source_model_id: ModelId,
    calibration: CalibrationProvenance,
    calibration_id: CalibrationId,
    evaluation: EvaluationProvenance,
    evaluation_id: EvaluationId,
    points: Vec<CampaignPoint>,
}

impl CampaignLedger {
    /// Start an empty ledger with fixed comparison provenance.
    pub fn new(
        source_model_id: ModelId,
        calibration: CalibrationProvenance,
        evaluation: EvaluationProvenance,
    ) -> Self {
        let calibration_id = calibration.id();
        let evaluation_id = evaluation.id();
        Self {
            source_model_id,
            calibration,
            calibration_id,
            evaluation,
            evaluation_id,
            points: Vec::new(),
        }
    }

    /// Full-precision source model shared by every point.
    pub fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Pinned calibration provenance.
    pub fn calibration(&self) -> &CalibrationProvenance {
        &self.calibration
    }

    /// Cached identity of pinned calibration provenance.
    pub fn calibration_id(&self) -> CalibrationId {
        self.calibration_id
    }

    /// Pinned evaluation provenance.
    pub fn evaluation(&self) -> &EvaluationProvenance {
        &self.evaluation
    }

    /// Cached identity of pinned evaluation provenance.
    pub fn evaluation_id(&self) -> EvaluationId {
        self.evaluation_id
    }

    /// Accepted campaign points in insertion order.
    pub fn points(&self) -> &[CampaignPoint] {
        &self.points
    }

    /// Add a point only when every comparison identity matches this ledger.
    ///
    /// # Errors
    /// Returns [`CampaignError::ProvenanceMismatch`] for mixed source/calibration/
    /// evaluation records, or [`CampaignError::DuplicatePackage`] when exact
    /// package bytes were already recorded.
    pub fn add(&mut self, point: CampaignPoint) -> Result<(), CampaignError> {
        if point.source_model_id != self.source_model_id {
            return Err(CampaignError::ProvenanceMismatch("source_model"));
        }
        if point.calibration_id != self.calibration_id {
            return Err(CampaignError::ProvenanceMismatch("calibration"));
        }
        if point.evaluation_id != self.evaluation_id {
            return Err(CampaignError::ProvenanceMismatch("evaluation"));
        }
        if self
            .points
            .iter()
            .any(|existing| existing.package_id() == point.package_id())
        {
            return Err(CampaignError::DuplicatePackage(point.package_id()));
        }
        self.points.push(point);
        Ok(())
    }

    /// Return non-dominated measured points in deterministic objective order.
    ///
    /// Storage, error, perplexity, and VRAM are minimized; task score and
    /// throughput are maximized. A point dominates another when it is no worse
    /// on every requested objective and strictly better on at least one. No
    /// interpolation or extrapolation is performed. General
    /// multi-objective selection is `O(points² × objectives)`; campaign sweeps
    /// should persist raw points and compute larger frontiers offline if needed.
    ///
    /// # Errors
    /// Returns [`CampaignError`] for an empty/duplicate objective list or when a
    /// point lacks an optional requested measurement.
    pub fn pareto_frontier(
        &self,
        objectives: &[CampaignObjective],
    ) -> Result<Vec<&CampaignPoint>, CampaignError> {
        if objectives.is_empty() {
            return Err(CampaignError::EmptyObjectives);
        }
        for (index, &objective) in objectives.iter().enumerate() {
            if objectives[..index].contains(&objective) {
                return Err(CampaignError::DuplicateObjective(objective));
            }
        }
        for point in &self.points {
            for &objective in objectives {
                if !has_objective(point, objective) {
                    return Err(CampaignError::MissingObjective {
                        package_id: point.package_id(),
                        objective,
                    });
                }
            }
        }

        let mut frontier: Vec<_> = self
            .points
            .iter()
            .filter(|candidate| {
                !self.points.iter().any(|other| {
                    other.package_id() != candidate.package_id()
                        && dominates(other, candidate, objectives)
                })
            })
            .collect();
        frontier.sort_by(|left, right| {
            for &objective in objectives {
                let ordering = objective_cmp(left, right, objective);
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            left.package_id()
                .as_bytes()
                .cmp(right.package_id().as_bytes())
        });
        Ok(frontier)
    }

    /// Serialize a canonical campaign record independent of point insertion order.
    ///
    /// Points are ordered by exact [`PackageId`]. Calibration and evaluation
    /// records are embedded, making the artifact self-contained for audit. This
    /// version-2 encoding is a stable hash target, not yet an interchange format;
    /// no public decoder is provided.
    ///
    /// # Errors
    /// Returns [`CampaignError::RecordTooLarge`] if a version-2 u32 count or
    /// embedded-record length would overflow.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, CampaignError> {
        let calibration = self.calibration.canonical_bytes();
        let evaluation = self.evaluation.canonical_bytes();
        let calibration_len = u32::try_from(calibration.len())
            .map_err(|_| CampaignError::RecordTooLarge("calibration"))?;
        let evaluation_len = u32::try_from(evaluation.len())
            .map_err(|_| CampaignError::RecordTooLarge("evaluation"))?;
        let point_count = u32::try_from(self.points.len())
            .map_err(|_| CampaignError::RecordTooLarge("points"))?;

        let mut points: Vec<_> = self.points.iter().collect();
        // Raw fixed-size digest bytes avoid display-format coupling.
        points.sort_by(|left, right| {
            left.package_id()
                .as_bytes()
                .cmp(right.package_id().as_bytes())
        });

        let mut out = Vec::new();
        out.extend_from_slice(&CAMPAIGN_MAGIC);
        out.push(CAMPAIGN_VERSION);
        out.extend_from_slice(self.source_model_id.as_bytes());
        out.extend_from_slice(&calibration_len.to_le_bytes());
        out.extend_from_slice(&calibration);
        out.extend_from_slice(&evaluation_len.to_le_bytes());
        out.extend_from_slice(&evaluation);
        out.extend_from_slice(&point_count.to_le_bytes());
        for point in points {
            out.extend_from_slice(point.model_id.as_bytes());
            out.extend_from_slice(point.package_id().as_bytes());
            out.extend_from_slice(&point.physical_bytes().to_le_bytes());
            let recipe = point.recipe.canonical_bytes();
            let recipe_len =
                u32::try_from(recipe.len()).map_err(|_| CampaignError::RecordTooLarge("recipe"))?;
            out.extend_from_slice(&recipe_len.to_le_bytes());
            out.extend_from_slice(&recipe);
            out.extend_from_slice(&point.metrics.logical_bpw.to_bits().to_le_bytes());
            write_optional_f64(&mut out, point.metrics.perplexity);
            write_optional_f64(&mut out, point.metrics.reconstruction_mse);
            write_optional_f64(&mut out, point.metrics.task_score);
            write_optional_f64(&mut out, point.metrics.tokens_per_second);
            write_optional_u64(&mut out, point.metrics.peak_vram_bytes);
        }
        Ok(out)
    }

    /// Exact content identity of the canonical campaign record.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::canonical_bytes`].
    pub fn id(&self) -> Result<CampaignId, CampaignError> {
        Ok(CampaignId(domain_hash(
            CAMPAIGN_ID_CONTEXT,
            &self.canonical_bytes()?,
        )))
    }
}

fn has_objective(point: &CampaignPoint, objective: CampaignObjective) -> bool {
    match objective {
        CampaignObjective::PhysicalBytes | CampaignObjective::LogicalBpw => true,
        CampaignObjective::Perplexity => point.metrics.perplexity.is_some(),
        CampaignObjective::ReconstructionMse => point.metrics.reconstruction_mse.is_some(),
        CampaignObjective::TaskScore => point.metrics.task_score.is_some(),
        CampaignObjective::TokensPerSecond => point.metrics.tokens_per_second.is_some(),
        CampaignObjective::PeakVramBytes => point.metrics.peak_vram_bytes.is_some(),
    }
}

fn objective_cmp(
    left: &CampaignPoint,
    right: &CampaignPoint,
    objective: CampaignObjective,
) -> Ordering {
    match objective {
        CampaignObjective::PhysicalBytes => left.physical_bytes().cmp(&right.physical_bytes()),
        CampaignObjective::LogicalBpw => left
            .metrics
            .logical_bpw
            .total_cmp(&right.metrics.logical_bpw),
        CampaignObjective::Perplexity => left
            .metrics
            .perplexity
            .expect("objective presence validated")
            .total_cmp(
                &right
                    .metrics
                    .perplexity
                    .expect("objective presence validated"),
            ),
        CampaignObjective::ReconstructionMse => left
            .metrics
            .reconstruction_mse
            .expect("objective presence validated")
            .total_cmp(
                &right
                    .metrics
                    .reconstruction_mse
                    .expect("objective presence validated"),
            ),
        // Inverted ordering: higher task scores are better.
        CampaignObjective::TaskScore => right
            .metrics
            .task_score
            .expect("objective presence validated")
            .total_cmp(
                &left
                    .metrics
                    .task_score
                    .expect("objective presence validated"),
            ),
        // Inverted ordering: higher throughput is better.
        CampaignObjective::TokensPerSecond => right
            .metrics
            .tokens_per_second
            .expect("objective presence validated")
            .total_cmp(
                &left
                    .metrics
                    .tokens_per_second
                    .expect("objective presence validated"),
            ),
        CampaignObjective::PeakVramBytes => left
            .metrics
            .peak_vram_bytes
            .expect("objective presence validated")
            .cmp(
                &right
                    .metrics
                    .peak_vram_bytes
                    .expect("objective presence validated"),
            ),
    }
}

fn dominates(
    left: &CampaignPoint,
    right: &CampaignPoint,
    objectives: &[CampaignObjective],
) -> bool {
    let mut strictly_better = false;
    for &objective in objectives {
        match objective_cmp(left, right, objective) {
            Ordering::Less => strictly_better = true,
            Ordering::Equal => {}
            Ordering::Greater => return false,
        }
    }
    strictly_better
}

fn validate_string(field: &'static str, value: &str) -> Result<(), CampaignError> {
    if value.is_empty() {
        return Err(CampaignError::EmptyField(field));
    }
    if value.len() > u32::MAX as usize {
        return Err(CampaignError::FieldTooLong(field));
    }
    Ok(())
}

fn checked_package_length(current: u64, read: usize) -> Result<u64, CampaignError> {
    let read = u64::try_from(read).map_err(|_| CampaignError::RecordTooLarge("package"))?;
    current
        .checked_add(read)
        .ok_or(CampaignError::RecordTooLarge("package"))
}

fn validate_nonzero(field: &'static str, value: u64) -> Result<(), CampaignError> {
    if value == 0 {
        Err(CampaignError::ZeroValue(field))
    } else {
        Ok(())
    }
}

fn validate_positive_finite(field: &'static str, value: f64) -> Result<(), CampaignError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(CampaignError::InvalidMetric(field))
    }
}

fn validate_nonnegative_finite(field: &'static str, value: f64) -> Result<(), CampaignError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(CampaignError::InvalidMetric(field))
    }
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), CampaignError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(CampaignError::InvalidMetric(field))
    }
}

fn canonicalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

/// Why campaign provenance or measurements were rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CampaignError {
    /// Required string field was empty.
    EmptyField(&'static str),
    /// String field exceeded canonical u32 length.
    FieldTooLong(&'static str),
    /// Required numeric field was zero.
    ZeroValue(&'static str),
    /// Metric violated its finite, sign, or non-zero constraint.
    InvalidMetric(&'static str),
    /// Point did not use the ledger's pinned comparison identity.
    ProvenanceMismatch(&'static str),
    /// Exact package bytes were already present in the ledger.
    DuplicatePackage(PackageId),
    /// Pareto selection requested no objectives.
    EmptyObjectives,
    /// Pareto selection repeated an objective.
    DuplicateObjective(CampaignObjective),
    /// Point lacked a requested optional measurement.
    MissingObjective {
        /// Exact package missing the metric.
        package_id: PackageId,
        /// Requested metric.
        objective: CampaignObjective,
    },
    /// Version-2 campaign record exceeded a u32 count or embedded length.
    RecordTooLarge(&'static str),
    /// Exact package measurement could not open or read its byte stream.
    PackageIo {
        /// I/O operation that failed.
        operation: &'static str,
        /// Portable category of the underlying I/O failure.
        kind: io::ErrorKind,
    },
}

impl fmt::Display for CampaignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "campaign field `{field}` is empty"),
            Self::FieldTooLong(field) => {
                write!(f, "campaign field `{field}` exceeds u32 capacity")
            }
            Self::ZeroValue(field) => write!(f, "campaign field `{field}` must be non-zero"),
            Self::InvalidMetric(metric) => write!(f, "campaign metric `{metric}` is invalid"),
            Self::ProvenanceMismatch(kind) => {
                write!(f, "campaign point has mismatched {kind} provenance")
            }
            Self::DuplicatePackage(id) => write!(f, "duplicate campaign package `{id}`"),
            Self::EmptyObjectives => f.write_str("campaign Pareto objectives are empty"),
            Self::DuplicateObjective(objective) => {
                write!(f, "duplicate campaign objective {objective:?}")
            }
            Self::MissingObjective {
                package_id,
                objective,
            } => write!(
                f,
                "campaign package `{package_id}` lacks objective {objective:?}"
            ),
            Self::RecordTooLarge(field) => {
                write!(f, "campaign record `{field}` exceeds version-2 capacity")
            }
            Self::PackageIo { operation, kind } => {
                write!(f, "package {operation} failed: {kind}")
            }
        }
    }
}

impl std::error::Error for CampaignError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_format::{ModelId, SemanticModelManifest, SemanticTensor};

    fn model_id(seed: u8) -> ModelId {
        let tensor = SemanticTensor::new("w", vec![1, 1], &[seed]).expect("tensor");
        SemanticModelManifest::new("test", &[seed], vec![tensor])
            .expect("manifest")
            .model_id()
    }

    fn calibration(sample_digest: [u8; 32]) -> CalibrationProvenance {
        CalibrationProvenance::new(
            "fineweb-edu",
            "2026-06-01",
            sample_digest,
            [2; 32],
            128,
            65_536,
            512,
            0x5eed,
        )
        .expect("calibration")
    }

    fn evaluation() -> EvaluationProvenance {
        EvaluationProvenance::new(
            "wikitext-2-ppl",
            "raw-v1",
            [4; 32],
            [2; 32],
            [5; 32],
            245,
            300_000,
        )
        .expect("evaluation")
    }

    fn recipe(seed: u8) -> RecipeProvenance {
        RecipeProvenance::new(
            "tritium-cli",
            "67f7256",
            vec![seed],
            "tritium convert --recipe recipe.json",
        )
        .expect("recipe")
    }

    #[test]
    fn calibration_identity_binds_exact_sample_set() {
        let first = CalibrationProvenance::new(
            "fineweb-edu",
            "2026-06-01",
            [1; 32],
            [2; 32],
            128,
            65_536,
            512,
            0x5eed,
        )
        .expect("valid provenance");
        let reordered = CalibrationProvenance::new(
            "fineweb-edu",
            "2026-06-01",
            [3; 32],
            [2; 32],
            128,
            65_536,
            512,
            0x5eed,
        )
        .expect("valid provenance");

        assert_eq!(first.canonical_bytes(), first.canonical_bytes());
        assert_ne!(first.id(), reordered.id());
    }

    #[test]
    fn measured_package_binds_exact_bytes_and_length() {
        let bytes = b"exact serialized ternary package";
        let measured = MeasuredPackage::from_bytes(bytes).expect("measured package");
        assert_eq!(measured.id(), PackageId::from_package_bytes(bytes));
        assert_eq!(measured.physical_bytes(), bytes.len() as u64);
    }

    #[test]
    fn measured_package_stream_matches_one_shot_across_read_boundaries() {
        struct ShortReader<'a> {
            bytes: &'a [u8],
            chunk_bytes: usize,
        }

        impl Read for ShortReader<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                let read = self.bytes.len().min(self.chunk_bytes).min(buffer.len());
                buffer[..read].copy_from_slice(&self.bytes[..read]);
                self.bytes = &self.bytes[read..];
                Ok(read)
            }
        }

        let bytes = b"streamed package identity must bind every exact byte";
        let streamed = MeasuredPackage::from_reader(ShortReader {
            bytes,
            chunk_bytes: 3,
        })
        .expect("streamed package");
        assert_eq!(streamed, MeasuredPackage::from_bytes(bytes).expect("slice"));
    }

    #[test]
    fn streamed_package_rejects_empty_and_reports_io_kind() {
        struct FailingReader;

        impl Read for FailingReader {
            fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::InvalidData, "corrupt source"))
            }
        }

        assert_eq!(
            MeasuredPackage::from_reader(io::empty()),
            Err(CampaignError::ZeroValue("physical_bytes"))
        );
        assert_eq!(
            MeasuredPackage::from_reader(FailingReader),
            Err(CampaignError::PackageIo {
                operation: "read",
                kind: io::ErrorKind::InvalidData,
            })
        );
    }

    #[test]
    fn streamed_package_length_overflow_is_typed() {
        assert_eq!(
            checked_package_length(u64::MAX, 1),
            Err(CampaignError::RecordTooLarge("package"))
        );
    }

    #[test]
    fn evaluation_identity_binds_harness_and_samples() {
        let first = EvaluationProvenance::new(
            "wikitext-2-ppl",
            "raw-v1",
            [4; 32],
            [2; 32],
            [5; 32],
            245,
            300_000,
        )
        .expect("valid evaluation");
        let changed_harness = EvaluationProvenance::new(
            "wikitext-2-ppl",
            "raw-v1",
            [4; 32],
            [2; 32],
            [6; 32],
            245,
            300_000,
        )
        .expect("valid evaluation");

        assert_ne!(first.id(), changed_harness.id());
    }

    #[test]
    fn recipe_identity_binds_revision_config_and_command() {
        let first = RecipeProvenance::new(
            "tritium-cli",
            "67f7256",
            br#"{"passes":2}"#.to_vec(),
            "tritium convert --recipe recipe.json",
        )
        .expect("recipe");
        let equivalent = RecipeProvenance::new(
            "tritium-cli",
            "67f7256",
            br#"{"passes":2}"#.to_vec(),
            "tritium convert --recipe recipe.json",
        )
        .expect("recipe");
        let changed_config = RecipeProvenance::new(
            "tritium-cli",
            "67f7256",
            br#"{"passes":3}"#.to_vec(),
            "tritium convert --recipe recipe.json",
        )
        .expect("recipe");
        let changed_command = RecipeProvenance::new(
            "tritium-cli",
            "67f7256",
            br#"{"passes":2}"#.to_vec(),
            "tritium convert --recipe other.json",
        )
        .expect("recipe");
        let changed_revision = RecipeProvenance::new(
            "tritium-cli",
            "bf7ab42",
            br#"{"passes":2}"#.to_vec(),
            "tritium convert --recipe recipe.json",
        )
        .expect("recipe");

        assert_ne!(first.id(), changed_config.id());
        assert_ne!(first.id(), changed_command.id());
        assert_ne!(first.id(), changed_revision.id());
        assert_eq!(first.id(), equivalent.id());
        assert_eq!(first.canonical_bytes(), equivalent.canonical_bytes());
        assert!(first.id().to_string().starts_with("trr1_"));
    }

    #[test]
    fn metrics_normalize_signed_zero() {
        let positive = CampaignMetrics::new(2.0)
            .expect("metrics")
            .with_reconstruction_mse(0.0)
            .expect("mse")
            .with_task_score(0.0)
            .expect("task score");
        let negative = CampaignMetrics::new(2.0)
            .expect("metrics")
            .with_reconstruction_mse(-0.0)
            .expect("mse")
            .with_task_score(-0.0)
            .expect("task score");

        assert_eq!(positive, negative);
        assert_eq!(negative.reconstruction_mse().expect("mse").to_bits(), 0);
        assert_eq!(negative.task_score().expect("task score").to_bits(), 0);
    }

    #[test]
    fn pareto_frontier_honors_quality_and_system_directions() {
        let source = model_id(1);
        let calibration = calibration([1; 32]);
        let evaluation = evaluation();
        let mut ledger = CampaignLedger::new(source, calibration.clone(), evaluation.clone());
        let make_point = |seed: u8, package_len: usize, task_score: f64, throughput: f64| {
            let package_bytes = vec![seed; package_len];
            CampaignPoint::new(
                source,
                model_id(seed),
                MeasuredPackage::from_bytes(&package_bytes).expect("package"),
                RecipeProvenance::new(
                    "tritium-cli",
                    "67f7256",
                    vec![seed],
                    "tritium convert --recipe recipe.json",
                )
                .expect("recipe"),
                calibration.id(),
                evaluation.id(),
                CampaignMetrics::new(2.0)
                    .expect("metrics")
                    .with_reconstruction_mse(0.01 + f64::from(seed) / 100.0)
                    .expect("mse")
                    .with_task_score(task_score)
                    .expect("task score")
                    .with_tokens_per_second(throughput)
                    .expect("throughput")
                    .with_peak_vram_bytes(u64::from(seed) * 1_000)
                    .expect("vram"),
            )
        };
        let compact = make_point(2, 7, 0.70, 100.0);
        let capable = make_point(3, 8, 0.75, 120.0);
        let dominated = make_point(4, 9, 0.72, 90.0);
        for point in [dominated, capable.clone(), compact.clone()] {
            ledger.add(point).expect("add point");
        }

        let frontier = ledger
            .pareto_frontier(&[
                CampaignObjective::PhysicalBytes,
                CampaignObjective::TaskScore,
                CampaignObjective::TokensPerSecond,
                CampaignObjective::PeakVramBytes,
            ])
            .expect("frontier");
        assert_eq!(
            frontier
                .iter()
                .map(|point| point.package_id())
                .collect::<Vec<_>>(),
            vec![compact.package_id(), capable.package_id()]
        );
    }

    #[test]
    fn ledger_rejects_mixed_provenance_and_duplicate_packages() {
        let source = model_id(1);
        let calibration_provenance = calibration([1; 32]);
        let evaluation = evaluation();
        let mut ledger =
            CampaignLedger::new(source, calibration_provenance.clone(), evaluation.clone());
        let metrics = CampaignMetrics::new(2.1)
            .expect("valid metrics")
            .with_perplexity(8.2)
            .expect("valid perplexity");
        let wrong_calibration = CampaignPoint::new(
            source,
            model_id(2),
            MeasuredPackage::from_bytes(b"candidate-a").expect("package"),
            recipe(9),
            calibration([9; 32]).id(),
            evaluation.id(),
            metrics.clone(),
        );
        assert!(matches!(
            ledger.add(wrong_calibration),
            Err(CampaignError::ProvenanceMismatch("calibration"))
        ));

        let package = MeasuredPackage::from_bytes(b"candidate-b").expect("package");
        let point = CampaignPoint::new(
            source,
            model_id(3),
            package,
            recipe(8),
            calibration_provenance.id(),
            evaluation.id(),
            metrics,
        );
        ledger.add(point.clone()).expect("first package");
        assert!(matches!(
            ledger.add(point),
            Err(CampaignError::DuplicatePackage(id)) if id == package.id()
        ));
    }

    #[test]
    fn pareto_frontier_keeps_measured_storage_quality_tradeoffs() {
        let source = model_id(1);
        let calibration = calibration([1; 32]);
        let evaluation = evaluation();
        let mut ledger = CampaignLedger::new(source, calibration.clone(), evaluation.clone());
        let make_point = |seed: u8, package_len: usize, perplexity: f64| {
            let package_bytes = vec![seed; package_len];
            CampaignPoint::new(
                source,
                model_id(seed),
                MeasuredPackage::from_bytes(&package_bytes).expect("package"),
                recipe(seed),
                calibration.id(),
                evaluation.id(),
                CampaignMetrics::new(2.0)
                    .expect("metrics")
                    .with_perplexity(perplexity)
                    .expect("perplexity"),
            )
        };
        let compact = make_point(2, 7, 10.5);
        let accurate = make_point(3, 8, 9.8);
        let dominated = make_point(4, 10, 11.0);
        let fp_sized = make_point(5, 54, 10.0);
        for point in [fp_sized, dominated, accurate.clone(), compact.clone()] {
            ledger.add(point).expect("add point");
        }

        let frontier = ledger
            .pareto_frontier(&[
                CampaignObjective::PhysicalBytes,
                CampaignObjective::Perplexity,
            ])
            .expect("frontier");
        assert_eq!(
            frontier
                .iter()
                .map(|point| point.package_id())
                .collect::<Vec<_>>(),
            vec![compact.package_id(), accurate.package_id()]
        );
    }

    #[test]
    fn campaign_record_identity_is_insertion_order_independent() {
        let source = model_id(1);
        let calibration = calibration([1; 32]);
        let evaluation = evaluation();
        let make_point = |seed: u8| {
            let package_bytes = vec![seed; usize::from(seed)];
            CampaignPoint::new(
                source,
                model_id(seed),
                MeasuredPackage::from_bytes(&package_bytes).expect("package"),
                recipe(seed),
                calibration.id(),
                evaluation.id(),
                CampaignMetrics::new(2.0)
                    .expect("metrics")
                    .with_perplexity(10.0 - f64::from(seed))
                    .expect("perplexity"),
            )
        };
        let a = make_point(2);
        let b = make_point(3);
        let mut forward = CampaignLedger::new(source, calibration.clone(), evaluation.clone());
        forward.add(a.clone()).expect("add a");
        forward.add(b.clone()).expect("add b");
        let mut reverse = CampaignLedger::new(source, calibration, evaluation);
        reverse.add(b).expect("add b");
        reverse.add(a).expect("add a");

        assert_eq!(
            forward.canonical_bytes().expect("encode"),
            reverse.canonical_bytes().expect("encode")
        );
        assert_eq!(forward.id().expect("id"), reverse.id().expect("id"));
        assert_eq!(
            forward.id().expect("id").to_string(),
            "trl1_c536022c10e224592e9817fe7664a644a66828a7ddc002fc4c69adecb2aefad2"
        );
    }
}
