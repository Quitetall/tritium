//! Reproducible calibration, evaluation, and multi-objective campaign records.

use core::{cmp::Ordering, fmt};
use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};
use tritium_format::{
    ModelId, PackageHasher, PackageId,
    salt_v2_package::{SaltV2IndexedRuntimeLedger, SaltV2PackageError, read_salt_v2_package},
};

const CALIBRATION_MAGIC: [u8; 4] = *b"TCAL";
const CAMPAIGN_MAGIC: [u8; 4] = *b"TCMP";
const EVALUATION_MAGIC: [u8; 4] = *b"TEVL";
const RECIPE_MAGIC: [u8; 4] = *b"TRCP";
const LEGACY_CAMPAIGN_VERSION: u8 = 2;
const PHYSICAL_REPORT_CAMPAIGN_VERSION: u8 = 4;
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

fn write_optional_physical_report(out: &mut Vec<u8>, value: Option<PhysicalSizeReport>) {
    let Some(report) = value else {
        out.push(0);
        return;
    };
    out.push(1);
    for value in [
        report.core_parameter_count,
        report.model_parameter_count,
        report.logical_core_trits.get(),
        report.serialized.core_payload_bytes,
        report.serialized.core_scale_bytes,
        report.serialized.allocation_map_bytes,
        report.serialized.allocation_map_bits,
        report.serialized.allocation_map_embedded_bits,
        report.serialized.header_bytes,
        report.serialized.transform_bytes,
        report.serialized.preserved_bytes,
        report.serialized.alignment_bytes,
        report.resident.core_bytes,
        report.resident.map_bytes,
        report.resident.map_bits,
        report.resident.map_embedded_bits,
        report.resident.descriptor_bytes,
        report.resident.preserved_bytes,
        report.resident.shadow_bytes,
    ] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    write_optional_u64(out, report.resident.peak_workspace_bytes);
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

    /// Remeasure a file and require its identity and length to remain unchanged.
    ///
    /// This is intended for the final report/publish boundary: the exact opened
    /// artifact is hashed again instead of trusting a path, a caller-supplied
    /// length, or an earlier filesystem metadata observation.
    ///
    /// # Errors
    /// Returns the same I/O and empty-file errors as [`Self::from_file`], or
    /// [`CampaignError::PackageMeasurementMismatch`] if any byte changed.
    pub fn verify_file(self, path: impl AsRef<Path>) -> Result<(), CampaignError> {
        let actual = Self::from_file(path)?;
        if actual != self {
            return Err(CampaignError::PackageMeasurementMismatch {
                expected_id: self.id,
                actual_id: actual.id,
                expected_bytes: self.physical_bytes,
                actual_bytes: actual.physical_bytes,
            });
        }
        Ok(())
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

/// An exact rational storage rate measured in thousandths of a bit per weight.
///
/// The represented value is `numerator_millibits / denominator_weights`.
/// Keeping both integers avoids using a rounded floating-point display value to
/// authorize a package, resident allocation, or profile claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExactMillibpw {
    numerator_millibits: u128,
    denominator_weights: u64,
}

impl ExactMillibpw {
    fn from_bytes(bytes: u64, denominator_weights: u64) -> Self {
        Self {
            numerator_millibits: u128::from(bytes) * 8_000,
            denominator_weights,
        }
    }

    /// Numerator in millibits before division by the weight count.
    pub const fn numerator_millibits(self) -> u128 {
        self.numerator_millibits
    }

    /// Exact weight-count denominator.
    pub const fn denominator_weights(self) -> u64 {
        self.denominator_weights
    }

    /// Smallest integer millibpw value no lower than the exact rate.
    ///
    /// This conservative ceiling is suitable for display and integer-only
    /// budgets. Exact authorization should prefer [`Self::is_at_most`].
    pub fn ceiling(self) -> u128 {
        let denominator = u128::from(self.denominator_weights);
        self.numerator_millibits.div_ceil(denominator)
    }

    /// Test an integer millibpw ceiling without division or floating point.
    pub fn is_at_most(self, limit_millibpw: u64) -> bool {
        self.numerator_millibits
            <= u128::from(self.denominator_weights) * u128::from(limit_millibpw)
    }

    /// Test whether the exact rate equals an integer millibpw value.
    pub fn equals(self, millibpw: u64) -> bool {
        self.numerator_millibits == u128::from(self.denominator_weights) * u128::from(millibpw)
    }
}

/// Nonzero count of logical ternary symbols assigned by an additive allocator.
///
/// This wrapper prevents a rounded integer bit count from being passed where the
/// ADR requires a trit count. Information bits are derived only for display.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LogicalTritCount(u64);

impl LogicalTritCount {
    /// Construct a nonzero logical trit count.
    pub fn new(trits: u64) -> Result<Self, CampaignError> {
        validate_nonzero("logical_core_trits", trits)?;
        Ok(Self(trits))
    }

    /// Raw number of logical ternary symbols.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Exact rational logical trit rate plus its information-bit display projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LogicalTritRate {
    trits: LogicalTritCount,
    weights: u64,
}

impl LogicalTritRate {
    /// Exact trit numerator.
    pub const fn trits(self) -> u64 {
        self.trits.get()
    }

    /// Exact weight denominator.
    pub const fn weights(self) -> u64 {
        self.weights
    }

    /// Information-theoretic bits per weight, `trits * log2(3) / weights`.
    pub fn bpw(self) -> f64 {
        self.trits.get() as f64 * crate::TRIT_BITS / self.weights as f64
    }
}

/// Exact serialized component counters for one inference package.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SerializedSizeComponents {
    core_payload_bytes: u64,
    core_scale_bytes: u64,
    allocation_map_bytes: u64,
    allocation_map_bits: u64,
    allocation_map_embedded_bits: u64,
    header_bytes: u64,
    transform_bytes: u64,
    preserved_bytes: u64,
    alignment_bytes: u64,
}

impl SerializedSizeComponents {
    /// Record exact serialized byte counts in package order-independent classes.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        core_payload_bytes: u64,
        core_scale_bytes: u64,
        allocation_map_bytes: u64,
        allocation_map_bits: u64,
        allocation_map_embedded_bits: u64,
        header_bytes: u64,
        transform_bytes: u64,
        preserved_bytes: u64,
        alignment_bytes: u64,
    ) -> Self {
        Self {
            core_payload_bytes,
            core_scale_bytes,
            allocation_map_bytes,
            allocation_map_bits,
            allocation_map_embedded_bits,
            header_bytes,
            transform_bytes,
            preserved_bytes,
            alignment_bytes,
        }
    }

    /// Encoded ternary coefficient payload bytes.
    pub const fn core_payload_bytes(self) -> u64 {
        self.core_payload_bytes
    }

    /// Deployment-scale bytes associated with the core payload.
    pub const fn core_scale_bytes(self) -> u64 {
        self.core_scale_bytes
    }

    /// Plane-presence, plane-count, and other allocation-map bytes.
    pub const fn allocation_map_bytes(self) -> u64 {
        self.allocation_map_bytes
    }

    /// Exact logical allocation-map bits, including embedded terminal bits.
    pub const fn allocation_map_bits(self) -> u64 {
        self.allocation_map_bits
    }

    /// Logical map bits carried in mandatory package/tensor scalar fields.
    pub const fn allocation_map_embedded_bits(self) -> u64 {
        self.allocation_map_embedded_bits
    }

    /// Tensor, row, and container header bytes.
    pub const fn header_bytes(self) -> u64 {
        self.header_bytes
    }

    /// Serialized transform metadata bytes.
    pub const fn transform_bytes(self) -> u64 {
        self.transform_bytes
    }

    /// Serialized bytes of preserved non-core tensors and assets.
    pub const fn preserved_bytes(self) -> u64 {
        self.preserved_bytes
    }

    /// Alignment and padding bytes present in the exact package.
    pub const fn alignment_bytes(self) -> u64 {
        self.alignment_bytes
    }

    fn core_total_bytes(self) -> Result<u64, CampaignError> {
        checked_physical_sum(
            "serialized core",
            [
                self.core_payload_bytes,
                self.core_scale_bytes,
                self.allocation_map_bytes,
                self.header_bytes,
                self.transform_bytes,
                self.alignment_bytes,
            ],
        )
    }

    fn total_bytes(self) -> Result<u64, CampaignError> {
        checked_physical_sum(
            "serialized package",
            [
                self.core_payload_bytes,
                self.core_scale_bytes,
                self.allocation_map_bytes,
                self.header_bytes,
                self.transform_bytes,
                self.preserved_bytes,
                self.alignment_bytes,
            ],
        )
    }
}

/// Exact steady-state and peak resident component counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ResidentSizeComponents {
    core_bytes: u64,
    map_bytes: u64,
    map_bits: u64,
    map_embedded_bits: u64,
    descriptor_bytes: u64,
    preserved_bytes: u64,
    shadow_bytes: u64,
    peak_workspace_bytes: Option<u64>,
}

impl ResidentSizeComponents {
    /// Record exact resident allocation byte counts.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        core_bytes: u64,
        map_bytes: u64,
        map_bits: u64,
        map_embedded_bits: u64,
        descriptor_bytes: u64,
        preserved_bytes: u64,
        shadow_bytes: u64,
        peak_workspace_bytes: Option<u64>,
    ) -> Self {
        Self {
            core_bytes,
            map_bytes,
            map_bits,
            map_embedded_bits,
            descriptor_bytes,
            preserved_bytes,
            shadow_bytes,
            peak_workspace_bytes,
        }
    }

    /// Resident encoded core coefficient and scale bytes.
    pub const fn core_bytes(self) -> u64 {
        self.core_bytes
    }

    /// Resident plane/allocation map bytes.
    pub const fn map_bytes(self) -> u64 {
        self.map_bytes
    }

    /// Exact logical runtime map bits, including scalar-carried terminal bits.
    pub const fn map_bits(self) -> u64 {
        self.map_bits
    }

    /// Runtime map bits carried in a mandatory launch scalar rather than an allocation.
    pub const fn map_embedded_bits(self) -> u64 {
        self.map_embedded_bits
    }

    /// Resident row, tensor, and dispatch descriptor bytes.
    pub const fn descriptor_bytes(self) -> u64 {
        self.descriptor_bytes
    }

    /// Resident preserved non-core tensor bytes.
    pub const fn preserved_bytes(self) -> u64 {
        self.preserved_bytes
    }

    /// Required alternate layouts or other persistent weight shadows.
    pub const fn shadow_bytes(self) -> u64 {
        self.shadow_bytes
    }

    /// Additional transient allocation present at the measured peak.
    pub const fn peak_workspace_bytes(self) -> Option<u64> {
        self.peak_workspace_bytes
    }

    fn core_total_bytes(self) -> Result<u64, CampaignError> {
        checked_physical_sum(
            "resident core",
            [
                self.core_bytes,
                self.map_bytes,
                self.descriptor_bytes,
                self.shadow_bytes,
            ],
        )
    }

    fn steady_total_bytes(self) -> Result<u64, CampaignError> {
        checked_physical_sum(
            "steady resident",
            [
                self.core_bytes,
                self.map_bytes,
                self.descriptor_bytes,
                self.preserved_bytes,
                self.shadow_bytes,
            ],
        )
    }

    fn peak_total_bytes(self) -> Result<Option<u64>, CampaignError> {
        self.peak_workspace_bytes
            .map(|workspace| {
                self.steady_total_bytes()?
                    .checked_add(workspace)
                    .ok_or(CampaignError::PhysicalSizeOverflow("peak resident"))
            })
            .transpose()
    }
}

/// Checked exact logical, serialized, and resident accounting for one package.
///
/// A report is inseparably bound to the [`MeasuredPackage`] used at its
/// construction boundary. Every serialized component is checked and must sum
/// to that measured artifact's actual byte count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PhysicalSizeReport {
    package: MeasuredPackage,
    core_parameter_count: u64,
    model_parameter_count: u64,
    logical_core_trits: LogicalTritCount,
    serialized: SerializedSizeComponents,
    resident: ResidentSizeComponents,
    serialized_core_bytes: u64,
    steady_resident_bytes: u64,
    peak_resident_bytes: Option<u64>,
    resident_core_bytes: u64,
}

impl PhysicalSizeReport {
    /// Build and validate exact physical accounting against measured bytes.
    ///
    /// `model_parameter_count` is the denominator for whole-model rates and
    /// must include the `core_parameter_count` denominator used by core rates.
    ///
    /// # Errors
    /// Returns [`CampaignError`] for zero core/model counts, zero logical or
    /// physical core storage, counter overflow, an inverted parameter-count
    /// relationship, or a serialized total unlike `package.physical_bytes()`.
    pub(crate) fn new(
        package: MeasuredPackage,
        core_parameter_count: u64,
        model_parameter_count: u64,
        logical_core_trits: LogicalTritCount,
        serialized: SerializedSizeComponents,
        resident: ResidentSizeComponents,
    ) -> Result<Self, CampaignError> {
        validate_nonzero("core_parameter_count", core_parameter_count)?;
        validate_nonzero("model_parameter_count", model_parameter_count)?;
        if core_parameter_count > model_parameter_count {
            return Err(CampaignError::InvalidPhysicalSize(
                "core_parameter_count exceeds model_parameter_count",
            ));
        }

        validate_map_accounting(
            "serialized allocation map",
            serialized.allocation_map_bytes,
            serialized.allocation_map_bits,
            serialized.allocation_map_embedded_bits,
        )?;
        validate_map_accounting(
            "resident allocation map",
            resident.map_bytes,
            resident.map_bits,
            resident.map_embedded_bits,
        )?;

        let serialized_core_bytes = serialized.core_total_bytes()?;
        validate_nonzero("serialized_core_bytes", serialized_core_bytes)?;
        let serialized_total = serialized.total_bytes()?;
        if serialized_total != package.physical_bytes() {
            return Err(CampaignError::PhysicalPackageSizeMismatch {
                measured_bytes: package.physical_bytes(),
                component_bytes: serialized_total,
            });
        }

        let resident_core_bytes = resident.core_total_bytes()?;
        validate_nonzero("resident_core_bytes", resident_core_bytes)?;
        let steady_resident_bytes = resident.steady_total_bytes()?;
        let peak_resident_bytes = resident.peak_total_bytes()?;

        Ok(Self {
            package,
            core_parameter_count,
            model_parameter_count,
            logical_core_trits,
            serialized,
            resident,
            serialized_core_bytes,
            steady_resident_bytes,
            peak_resident_bytes,
            resident_core_bytes,
        })
    }

    /// Parse an exact SALT V2 artifact and derive all core accounting from it.
    ///
    /// This is the public construction boundary for SALT V2 reports. Callers
    /// supply only the whole-model denominator and an optional independently
    /// measured transient workspace; payload, scales, maps, headers,
    /// transforms, padding, logical trits, and indexed-runtime allocations are
    /// rederived from the canonical bytes. The package is treated as a
    /// core-only artifact, so preserved-model components and shadows are zero.
    /// This compatibility path predicts the indexed runtime layout but does not
    /// prove that a runtime allocated it. Physical resident claims should use
    /// [`Self::from_salt_v2_package_bytes_with_runtime_receipts`].
    ///
    /// # Errors
    /// Rejects a malformed/noncanonical package, accounting overflow, an
    /// undersized whole-model denominator, or inconsistent derived components.
    pub fn from_salt_v2_package_bytes(
        package_bytes: &[u8],
        model_parameter_count: u64,
        peak_workspace_bytes: Option<u64>,
    ) -> Result<Self, CampaignError> {
        Self::from_salt_v2_package_bytes_checked_runtime(
            package_bytes,
            model_parameter_count,
            None,
            peak_workspace_bytes,
        )
    }

    /// Parse an exact SALT V2 artifact and verify a runtime allocation receipt against it.
    ///
    /// Supply one ledger per package tensor in canonical order, obtained from the resident runtime
    /// handles after their allocations succeed, for example via
    /// `SaltV2ResidentAllocationReceipt::runtime_ledger()` on CUDA. Every component must equal the
    /// tensor layout rederived from the opened canonical package; a matching aggregate total with
    /// different per-tensor payload, scale, map, or rank-prefix components is rejected.
    ///
    /// # Errors
    /// Returns the same failures as [`Self::from_salt_v2_package_bytes`] and additionally rejects
    /// a missing, extra, or component-disagreeing runtime receipt.
    pub fn from_salt_v2_package_bytes_with_runtime_receipts(
        package_bytes: &[u8],
        model_parameter_count: u64,
        runtime_receipts: &[SaltV2IndexedRuntimeLedger],
        peak_workspace_bytes: Option<u64>,
    ) -> Result<Self, CampaignError> {
        Self::from_salt_v2_package_bytes_checked_runtime(
            package_bytes,
            model_parameter_count,
            Some(runtime_receipts),
            peak_workspace_bytes,
        )
    }

    fn from_salt_v2_package_bytes_checked_runtime(
        package_bytes: &[u8],
        model_parameter_count: u64,
        runtime_receipts: Option<&[SaltV2IndexedRuntimeLedger]>,
        peak_workspace_bytes: Option<u64>,
    ) -> Result<Self, CampaignError> {
        let decoded = read_salt_v2_package(package_bytes)?;
        if let Some(supplied) = runtime_receipts {
            if supplied.len() != decoded.package.tensors().len() {
                return Err(CampaignError::RuntimeAllocationReceiptCountMismatch {
                    expected: decoded.package.tensors().len(),
                    supplied: supplied.len(),
                });
            }
            for (tensor_index, (tensor, supplied)) in
                decoded.package.tensors().iter().zip(supplied).enumerate()
            {
                let expected =
                    SaltV2IndexedRuntimeLedger::for_tensor(tensor, decoded.package.codec())?;
                if supplied != &expected {
                    let (component, expected_value, supplied_value) = runtime_ledger_mismatch(
                        expected, *supplied,
                    )
                    .ok_or(CampaignError::InvalidPhysicalSize(
                        "runtime ledger comparison omitted a field",
                    ))?;
                    return Err(CampaignError::RuntimeAllocationMismatch {
                        tensor_index,
                        component,
                        expected: expected_value,
                        supplied: supplied_value,
                    });
                }
            }
        }
        let runtime = SaltV2IndexedRuntimeLedger::for_package(&decoded.package)?;
        let core_parameter_count =
            decoded
                .package
                .tensors()
                .iter()
                .try_fold(0_u64, |total, tensor| {
                    total
                        .checked_add(u64::try_from(tensor.logical_coefficients()).map_err(
                            |_| CampaignError::PhysicalSizeOverflow("core parameter count"),
                        )?)
                        .ok_or(CampaignError::PhysicalSizeOverflow("core parameter count"))
                })?;
        let logical_core_trits =
            decoded
                .package
                .tensors()
                .iter()
                .try_fold(0_u64, |tensor_total, tensor| {
                    tensor.tiles().iter().try_fold(tensor_total, |total, tile| {
                        let logical_len = u64::try_from(tile.logical_len()).map_err(|_| {
                            CampaignError::PhysicalSizeOverflow("logical core trits")
                        })?;
                        let plane_count = u64::try_from(tile.planes().len()).map_err(|_| {
                            CampaignError::PhysicalSizeOverflow("logical core trits")
                        })?;
                        total
                            .checked_add(
                                logical_len.checked_mul(plane_count).ok_or(
                                    CampaignError::PhysicalSizeOverflow("logical core trits"),
                                )?,
                            )
                            .ok_or(CampaignError::PhysicalSizeOverflow("logical core trits"))
                    })
                })?;
        let serialized = SerializedSizeComponents::new(
            decoded.ledger.payload_bytes,
            decoded.ledger.scales_bytes,
            decoded.ledger.maps_bytes,
            decoded.ledger.allocation_map_bits,
            decoded.ledger.allocation_map_embedded_bits,
            decoded.ledger.headers_bytes,
            decoded.ledger.transform_bytes,
            0,
            decoded.ledger.padding_bytes,
        );
        let resident = ResidentSizeComponents::new(
            runtime
                .payload_bytes()
                .checked_add(runtime.scale_bytes())
                .ok_or(CampaignError::PhysicalSizeOverflow("resident core"))?,
            runtime.allocation_map_bytes(),
            runtime.allocation_map_bits(),
            runtime.allocation_map_embedded_bits(),
            runtime.rank_prefix_bytes(),
            0,
            runtime.dense_shadow_bytes(),
            peak_workspace_bytes,
        );
        Self::new(
            MeasuredPackage::from_bytes(package_bytes)?,
            core_parameter_count,
            model_parameter_count,
            LogicalTritCount::new(logical_core_trits)?,
            serialized,
            resident,
        )
    }

    /// Exact package identity and measured byte count bound to this report.
    pub const fn package(self) -> MeasuredPackage {
        self.package
    }

    /// Number of core parameters used for logical/core physical rates.
    pub const fn core_parameter_count(self) -> u64 {
        self.core_parameter_count
    }

    /// Total model parameter count used for whole-model rates.
    pub const fn model_parameter_count(self) -> u64 {
        self.model_parameter_count
    }

    /// Exact aggregate logical ternary symbols assigned by the allocator.
    pub const fn logical_core_trits(self) -> LogicalTritCount {
        self.logical_core_trits
    }

    /// Exact serialized component counters.
    pub const fn serialized(self) -> SerializedSizeComponents {
        self.serialized
    }

    /// Exact resident component counters.
    pub const fn resident(self) -> ResidentSizeComponents {
        self.resident
    }

    /// Exact serialized core bytes, including maps, headers, transforms, and alignment.
    pub const fn serialized_core_bytes(self) -> u64 {
        self.serialized_core_bytes
    }

    /// Exact steady-state resident core bytes, including maps, descriptors, and shadows.
    pub const fn resident_core_bytes(self) -> u64 {
        self.resident_core_bytes
    }

    /// Exact whole-model steady-state resident bytes.
    pub const fn steady_resident_bytes(self) -> u64 {
        self.steady_resident_bytes
    }

    /// Exact whole-model peak resident bytes.
    pub const fn peak_resident_bytes(self) -> Option<u64> {
        self.peak_resident_bytes
    }

    /// Exact rational logical-trit rate over core parameters.
    pub const fn logical_core_rate(self) -> LogicalTritRate {
        LogicalTritRate {
            trits: self.logical_core_trits,
            weights: self.core_parameter_count,
        }
    }

    /// Exact serialized core rate over core parameters.
    pub fn serialized_core_millibpw(self) -> ExactMillibpw {
        ExactMillibpw::from_bytes(self.serialized_core_bytes, self.core_parameter_count)
    }

    /// Exact steady resident core rate over core parameters.
    pub fn resident_core_millibpw(self) -> ExactMillibpw {
        ExactMillibpw::from_bytes(self.resident_core_bytes, self.core_parameter_count)
    }

    /// Exact complete artifact-file rate over all model parameters.
    pub fn whole_model_serialized_millibpw(self) -> ExactMillibpw {
        ExactMillibpw::from_bytes(self.package.physical_bytes(), self.model_parameter_count)
    }

    /// Exact whole-model steady resident rate.
    pub fn whole_model_steady_resident_millibpw(self) -> ExactMillibpw {
        ExactMillibpw::from_bytes(self.steady_resident_bytes, self.model_parameter_count)
    }

    /// Exact whole-model peak resident rate.
    pub fn whole_model_peak_resident_millibpw(self) -> Option<ExactMillibpw> {
        self.peak_resident_bytes
            .map(|bytes| ExactMillibpw::from_bytes(bytes, self.model_parameter_count))
    }

    /// Remeasure the package file and require exact identity and length equality.
    ///
    /// # Errors
    /// Returns the same errors as [`MeasuredPackage::verify_file`].
    pub fn verify_file(self, path: impl AsRef<Path>) -> Result<(), CampaignError> {
        self.package.verify_file(path)
    }
}

/// Metric used as one axis of campaign Pareto dominance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CampaignObjective {
    /// Minimize exact serialized inference-artifact bytes.
    PhysicalBytes,
    /// Minimize the exact logical-trit ratio per source weight.
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

/// Legacy logical-rate metrics plus optional quality and checked physical evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct CampaignMetrics {
    logical_bpw: f64,
    perplexity: Option<f64>,
    reconstruction_mse: Option<f64>,
    task_score: Option<f64>,
    tokens_per_second: Option<f64>,
    peak_vram_bytes: Option<u64>,
    physical_size_report: Option<PhysicalSizeReport>,
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
            physical_size_report: None,
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
        if self
            .physical_size_report
            .and_then(PhysicalSizeReport::peak_resident_bytes)
            .is_some_and(|peak| peak != bytes)
        {
            return Err(CampaignError::PhysicalMetricMismatch("peak_vram_bytes"));
        }
        self.peak_vram_bytes = Some(bytes);
        Ok(self)
    }

    /// Attach checked physical accounting to these metrics.
    ///
    /// The logical-rate display is always rederived from the report's exact
    /// trit/weight ratio. When the report includes an independently measured
    /// peak, that total becomes the peak-VRAM objective; an existing value must
    /// agree. A report with no peak measurement does not invent one.
    ///
    /// # Errors
    /// Returns [`CampaignError::PhysicalMetricMismatch`] if an existing peak
    /// resident measurement disagrees with the checked report.
    pub fn with_physical_size_report(
        mut self,
        report: PhysicalSizeReport,
    ) -> Result<Self, CampaignError> {
        if let Some(report_peak) = report.peak_resident_bytes() {
            if self
                .peak_vram_bytes
                .is_some_and(|bytes| bytes != report_peak)
            {
                return Err(CampaignError::PhysicalMetricMismatch("peak_vram_bytes"));
            }
            self.peak_vram_bytes = Some(report_peak);
        }
        self.logical_bpw = report.logical_core_rate().bpw();
        self.physical_size_report = Some(report);
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

    /// Checked exact physical accounting, when supplied.
    pub const fn physical_size_report(&self) -> Option<&PhysicalSizeReport> {
        self.physical_size_report.as_ref()
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
    /// Construction alone does not prove provenance or bind an optional physical
    /// report to `package`. [`CampaignLedger::add`] performs both trust-boundary
    /// validations.
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
    /// evaluation records, [`CampaignError::PackageMeasurementMismatch`] when an
    /// attached report describes another package, or
    /// [`CampaignError::DuplicatePackage`] when exact package bytes were already
    /// recorded.
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
        if let Some(report) = point.metrics.physical_size_report {
            let expected = point.package;
            let actual = report.package();
            if actual != expected {
                return Err(CampaignError::PackageMeasurementMismatch {
                    expected_id: expected.id(),
                    actual_id: actual.id(),
                    expected_bytes: expected.physical_bytes(),
                    actual_bytes: actual.physical_bytes(),
                });
            }
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
    /// Legacy ledgers without physical reports retain their exact version-2 byte
    /// encoding and identity. A ledger containing any physical report uses the
    /// version-4 encoding, in which every point carries an explicit absent/present
    /// marker and present reports bind exact trits, map bits, components, and
    /// optional peak measurements. Neither
    /// version is yet an interchange format; no public decoder is provided.
    ///
    /// # Errors
    /// Returns [`CampaignError::RecordTooLarge`] if a canonical u32 count or
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
        let has_physical_reports = self
            .points
            .iter()
            .any(|point| point.metrics.physical_size_report.is_some());

        let mut points: Vec<_> = self.points.iter().collect();
        // Raw fixed-size digest bytes avoid display-format coupling.
        points.sort_by(|left, right| {
            left.package_id()
                .as_bytes()
                .cmp(right.package_id().as_bytes())
        });

        let mut out = Vec::new();
        out.extend_from_slice(&CAMPAIGN_MAGIC);
        out.push(if has_physical_reports {
            PHYSICAL_REPORT_CAMPAIGN_VERSION
        } else {
            LEGACY_CAMPAIGN_VERSION
        });
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
            if has_physical_reports {
                write_optional_physical_report(&mut out, point.metrics.physical_size_report);
            }
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
        CampaignObjective::PhysicalBytes => true,
        CampaignObjective::LogicalBpw => point.metrics.physical_size_report.is_some(),
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
        CampaignObjective::LogicalBpw => {
            let left = left
                .metrics
                .physical_size_report
                .expect("objective presence validated")
                .logical_core_rate();
            let right = right
                .metrics
                .physical_size_report
                .expect("objective presence validated")
                .logical_core_rate();
            (u128::from(left.trits()) * u128::from(right.weights()))
                .cmp(&(u128::from(right.trits()) * u128::from(left.weights())))
        }
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

fn checked_physical_sum<const N: usize>(
    field: &'static str,
    components: [u64; N],
) -> Result<u64, CampaignError> {
    components.into_iter().try_fold(0_u64, |total, bytes| {
        total
            .checked_add(bytes)
            .ok_or(CampaignError::PhysicalSizeOverflow(field))
    })
}

fn validate_map_accounting(
    field: &'static str,
    allocated_bytes: u64,
    logical_bits: u64,
    embedded_bits: u64,
) -> Result<(), CampaignError> {
    let accounted_bits = allocated_bytes
        .checked_mul(8)
        .and_then(|bits| bits.checked_add(embedded_bits))
        .ok_or(CampaignError::PhysicalSizeOverflow(field))?;
    if accounted_bits != logical_bits {
        return Err(CampaignError::InvalidPhysicalSize(field));
    }
    Ok(())
}

fn runtime_ledger_mismatch(
    expected: SaltV2IndexedRuntimeLedger,
    supplied: SaltV2IndexedRuntimeLedger,
) -> Option<(&'static str, u64, u64)> {
    [
        (
            "payload bytes",
            expected.payload_bytes(),
            supplied.payload_bytes(),
        ),
        (
            "scale bytes",
            expected.scale_bytes(),
            supplied.scale_bytes(),
        ),
        (
            "allocation-map bytes",
            expected.allocation_map_bytes(),
            supplied.allocation_map_bytes(),
        ),
        (
            "rank-prefix bytes",
            expected.rank_prefix_bytes(),
            supplied.rank_prefix_bytes(),
        ),
        (
            "allocation-map bits",
            expected.allocation_map_bits(),
            supplied.allocation_map_bits(),
        ),
        (
            "allocation-map embedded bits",
            expected.allocation_map_embedded_bits(),
            supplied.allocation_map_embedded_bits(),
        ),
        (
            "dense shadow bytes",
            expected.dense_shadow_bytes(),
            supplied.dense_shadow_bytes(),
        ),
        (
            "allocation tiles",
            expected.allocation_tiles(),
            supplied.allocation_tiles(),
        ),
        (
            "present planes",
            expected.present_planes(),
            supplied.present_planes(),
        ),
        (
            "steady resident bytes",
            expected.steady_resident_bytes(),
            supplied.steady_resident_bytes(),
        ),
    ]
    .into_iter()
    .find(|(_, expected, supplied)| expected != supplied)
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
    /// Canonical SALT V2 package parsing or runtime planning failed.
    SaltV2Package(SaltV2PackageError),
    /// Required string field was empty.
    EmptyField(&'static str),
    /// String field exceeded canonical u32 length.
    FieldTooLong(&'static str),
    /// Required numeric field was zero.
    ZeroValue(&'static str),
    /// Metric violated its finite, sign, or non-zero constraint.
    InvalidMetric(&'static str),
    /// A physical-size relationship was structurally invalid.
    InvalidPhysicalSize(&'static str),
    /// Exact physical component addition exceeded `u64`.
    PhysicalSizeOverflow(&'static str),
    /// Serialized components did not sum to the measured package length.
    PhysicalPackageSizeMismatch {
        /// Exact byte count obtained by measuring the package.
        measured_bytes: u64,
        /// Exact byte count obtained by checked component addition.
        component_bytes: u64,
    },
    /// The number of per-tensor runtime receipts differed from the opened package.
    RuntimeAllocationReceiptCountMismatch {
        /// Tensor count in the canonical package.
        expected: usize,
        /// Number of supplied resident receipts.
        supplied: usize,
    },
    /// A resident runtime's checked allocation ledger differed from one opened tensor.
    RuntimeAllocationMismatch {
        /// Tensor ordinal in canonical package order.
        tensor_index: usize,
        /// First canonical component that disagreed.
        component: &'static str,
        /// Component value rederived from the package.
        expected: u64,
        /// Component value reported by the resident runtime handle.
        supplied: u64,
    },
    /// A remeasured or point-associated package differed from the expected package.
    PackageMeasurementMismatch {
        /// Expected exact package identity.
        expected_id: PackageId,
        /// Actual exact package identity.
        actual_id: PackageId,
        /// Expected exact package length.
        expected_bytes: u64,
        /// Actual exact package length.
        actual_bytes: u64,
    },
    /// A legacy exact metric disagreed with an attached physical report.
    PhysicalMetricMismatch(&'static str),
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
    /// Canonical campaign record exceeded a u32 count or embedded length.
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
            Self::SaltV2Package(source) => write!(f, "SALT V2 package is invalid: {source}"),
            Self::EmptyField(field) => write!(f, "campaign field `{field}` is empty"),
            Self::FieldTooLong(field) => {
                write!(f, "campaign field `{field}` exceeds u32 capacity")
            }
            Self::ZeroValue(field) => write!(f, "campaign field `{field}` must be non-zero"),
            Self::InvalidMetric(metric) => write!(f, "campaign metric `{metric}` is invalid"),
            Self::InvalidPhysicalSize(reason) => {
                write!(f, "campaign physical size is invalid: {reason}")
            }
            Self::PhysicalSizeOverflow(field) => {
                write!(f, "campaign physical size `{field}` exceeds u64 capacity")
            }
            Self::PhysicalPackageSizeMismatch {
                measured_bytes,
                component_bytes,
            } => write!(
                f,
                "physical components total {component_bytes} bytes, measured package is {measured_bytes} bytes"
            ),
            Self::RuntimeAllocationReceiptCountMismatch { expected, supplied } => write!(
                f,
                "runtime allocation receipt count is {supplied}, package contains {expected} tensors"
            ),
            Self::RuntimeAllocationMismatch {
                tensor_index,
                component,
                expected,
                supplied,
            } => write!(
                f,
                "tensor {tensor_index} runtime allocation receipt has {component}={supplied}, package-derived value is {expected}"
            ),
            Self::PackageMeasurementMismatch {
                expected_id,
                actual_id,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "package measurement changed from `{expected_id}` ({expected_bytes} bytes) to `{actual_id}` ({actual_bytes} bytes)"
            ),
            Self::PhysicalMetricMismatch(metric) => {
                write!(
                    f,
                    "campaign physical report disagrees with metric `{metric}`"
                )
            }
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
                write!(f, "campaign record `{field}` exceeds canonical capacity")
            }
            Self::PackageIo { operation, kind } => {
                write!(f, "package {operation} failed: {kind}")
            }
        }
    }
}

impl std::error::Error for CampaignError {}

impl From<SaltV2PackageError> for CampaignError {
    fn from(value: SaltV2PackageError) -> Self {
        Self::SaltV2Package(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::fs;
    use tritium_format::{
        ModelId, SemanticModelManifest, SemanticTensor,
        salt_v2::SaltV2Codec,
        salt_v2_package::{
            SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, write_salt_v2_package,
        },
    };

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

    fn logical_trits(value: u64) -> LogicalTritCount {
        LogicalTritCount::new(value).expect("nonzero logical trit count")
    }

    fn simple_physical_report(
        package_bytes: &[u8],
        core_parameter_count: u64,
        model_parameter_count: u64,
        payload_bytes: u64,
        scale_bytes: u64,
        other_serialized_bytes: u64,
    ) -> PhysicalSizeReport {
        let package = MeasuredPackage::from_bytes(package_bytes).expect("package");
        PhysicalSizeReport::new(
            package,
            core_parameter_count,
            model_parameter_count,
            logical_trits(core_parameter_count),
            SerializedSizeComponents::new(
                payload_bytes,
                scale_bytes,
                0,
                0,
                0,
                other_serialized_bytes,
                0,
                0,
                0,
            ),
            ResidentSizeComponents::new(
                payload_bytes + scale_bytes,
                0,
                0,
                0,
                0,
                other_serialized_bytes,
                0,
                Some(7),
            ),
        )
        .expect("physical report")
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
    fn ptqtp_two_direct_planes_at_g128_are_exactly_4_25_bpw() {
        // Two 2-bit planes occupy 64 bytes; two fp16 scales add 4 bytes.
        let bytes = vec![0_u8; 68];
        let report = PhysicalSizeReport::new(
            MeasuredPackage::from_bytes(&bytes).expect("package"),
            128,
            128,
            LogicalTritCount::new(256).expect("logical trits"),
            SerializedSizeComponents::new(64, 4, 0, 0, 0, 0, 0, 0, 0),
            ResidentSizeComponents::new(68, 0, 0, 0, 0, 0, 0, None),
        )
        .expect("physical report");

        assert!(report.serialized_core_millibpw().equals(4_250));
        assert_eq!(report.serialized_core_millibpw().ceiling(), 4_250);
        assert!(!report.serialized_core_millibpw().is_at_most(1_580));
        assert!(report.serialized_core_millibpw().is_at_most(4_250));
        let logical = report.logical_core_rate();
        assert_eq!(logical.trits(), 256);
        assert_eq!(logical.weights(), 128);
        assert!((logical.bpw() - 2.0 * crate::TRIT_BITS).abs() <= f64::EPSILON);
    }

    #[test]
    fn salt_v2_report_is_derived_from_canonical_package_and_runtime_ledgers() {
        let plane = SaltV2Plane::new(vec![0; 256], vec![f16::ZERO; 2]).expect("plane");
        let tensor = SaltV2Tensor::new(
            "w",
            vec![256],
            vec![SaltV2Tile::new(vec![plane.clone()]).expect("tile")],
        )
        .expect("tensor");
        let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).expect("package");
        let runtime = SaltV2IndexedRuntimeLedger::for_package(&package).expect("runtime ledger");
        let encoded = write_salt_v2_package(&package).expect("encode");

        let report = PhysicalSizeReport::from_salt_v2_package_bytes(&encoded.bytes, 300, None)
            .expect("derived report");
        let receipt_report = PhysicalSizeReport::from_salt_v2_package_bytes_with_runtime_receipts(
            &encoded.bytes,
            300,
            &[runtime],
            Some(7),
        )
        .expect("receipt-checked report");

        assert_eq!(report.core_parameter_count(), 256);
        assert_eq!(report.model_parameter_count(), 300);
        assert_eq!(report.logical_core_trits().get(), 256);
        assert_eq!(
            report.package().physical_bytes(),
            encoded.ledger.total_bytes
        );
        assert_eq!(report.serialized().allocation_map_bits(), 2);
        assert_eq!(report.serialized().allocation_map_embedded_bits(), 2);
        assert_eq!(report.resident().map_bits(), 2);
        assert_eq!(report.resident().map_embedded_bits(), 2);
        assert_eq!(report.resident().descriptor_bytes(), 0);
        assert_eq!(report.peak_resident_bytes(), None);
        assert_eq!(
            receipt_report.steady_resident_bytes(),
            report.steady_resident_bytes()
        );
        assert_eq!(receipt_report.peak_resident_bytes(), Some(75));

        let two_plane_tensor = SaltV2Tensor::new(
            "w",
            vec![256],
            vec![SaltV2Tile::new(vec![plane.clone(), plane]).expect("two-plane tile")],
        )
        .expect("two-plane tensor");
        let two_plane_package =
            SaltV2Package::new(SaltV2Codec::D2, vec![two_plane_tensor]).expect("two-plane package");
        let wrong_runtime = SaltV2IndexedRuntimeLedger::for_package(&two_plane_package)
            .expect("different runtime ledger");
        assert_eq!(
            PhysicalSizeReport::from_salt_v2_package_bytes_with_runtime_receipts(
                &encoded.bytes,
                300,
                &[wrong_runtime],
                None,
            ),
            Err(CampaignError::RuntimeAllocationMismatch {
                tensor_index: 0,
                component: "payload bytes",
                expected: runtime.payload_bytes(),
                supplied: wrong_runtime.payload_bytes(),
            })
        );
        assert_eq!(
            PhysicalSizeReport::from_salt_v2_package_bytes_with_runtime_receipts(
                &encoded.bytes,
                300,
                &[],
                None,
            ),
            Err(CampaignError::RuntimeAllocationReceiptCountMismatch {
                expected: 1,
                supplied: 0,
            })
        );
    }

    #[test]
    fn physical_report_rejects_zero_overflow_and_package_mismatch() {
        assert_eq!(
            MeasuredPackage::from_bytes(&[]),
            Err(CampaignError::ZeroValue("physical_bytes"))
        );
        let package = MeasuredPackage::from_bytes(&[0; 8]).expect("package");
        let serialized = SerializedSizeComponents::new(8, 0, 0, 0, 0, 0, 0, 0, 0);
        let resident = ResidentSizeComponents::new(8, 0, 0, 0, 0, 0, 0, None);
        assert_eq!(
            PhysicalSizeReport::new(package, 0, 8, logical_trits(8), serialized, resident),
            Err(CampaignError::ZeroValue("core_parameter_count"))
        );
        assert_eq!(
            PhysicalSizeReport::new(
                package,
                8,
                8,
                logical_trits(8),
                SerializedSizeComponents::new(7, 0, 1, 7, 0, 0, 0, 0, 0),
                resident,
            ),
            Err(CampaignError::InvalidPhysicalSize(
                "serialized allocation map"
            ))
        );
        assert_eq!(
            PhysicalSizeReport::new(
                package,
                8,
                8,
                logical_trits(8),
                serialized,
                ResidentSizeComponents::new(7, 1, 7, 0, 0, 0, 0, None),
            ),
            Err(CampaignError::InvalidPhysicalSize(
                "resident allocation map"
            ))
        );
        assert_eq!(
            PhysicalSizeReport::new(
                package,
                8,
                8,
                logical_trits(8),
                SerializedSizeComponents::new(0, 0, 0, 0, 0, 0, 0, 8, 0),
                resident,
            ),
            Err(CampaignError::ZeroValue("serialized_core_bytes"))
        );
        assert_eq!(
            PhysicalSizeReport::new(
                package,
                8,
                8,
                logical_trits(8),
                serialized,
                ResidentSizeComponents::new(0, 0, 0, 0, 0, 8, 0, None),
            ),
            Err(CampaignError::ZeroValue("resident_core_bytes"))
        );
        assert_eq!(
            PhysicalSizeReport::new(
                package,
                8,
                8,
                logical_trits(8),
                SerializedSizeComponents::new(7, 0, 0, 0, 0, 0, 0, 0, 0),
                resident,
            ),
            Err(CampaignError::PhysicalPackageSizeMismatch {
                measured_bytes: 8,
                component_bytes: 7,
            })
        );
        assert_eq!(
            PhysicalSizeReport::new(
                package,
                8,
                8,
                logical_trits(8),
                SerializedSizeComponents::new(u64::MAX, 1, 0, 0, 0, 0, 0, 0, 0),
                resident,
            ),
            Err(CampaignError::PhysicalSizeOverflow("serialized core"))
        );
        assert_eq!(
            PhysicalSizeReport::new(
                package,
                8,
                8,
                logical_trits(8),
                SerializedSizeComponents::new(8, 0, 0, 0, 0, 0, 0, u64::MAX, 0),
                resident,
            ),
            Err(CampaignError::PhysicalSizeOverflow("serialized package"))
        );
        assert_eq!(
            PhysicalSizeReport::new(
                package,
                8,
                8,
                logical_trits(8),
                serialized,
                ResidentSizeComponents::new(u64::MAX, 0, 0, 0, 0, 0, 1, None),
            ),
            Err(CampaignError::PhysicalSizeOverflow("resident core"))
        );
        assert_eq!(
            PhysicalSizeReport::new(
                package,
                8,
                8,
                logical_trits(8),
                serialized,
                ResidentSizeComponents::new(8, 0, 0, 0, 0, u64::MAX, 0, None),
            ),
            Err(CampaignError::PhysicalSizeOverflow("steady resident"))
        );
        assert_eq!(
            PhysicalSizeReport::new(
                package,
                8,
                8,
                logical_trits(8),
                serialized,
                ResidentSizeComponents::new(8, 0, 0, 0, 0, 0, 0, Some(u64::MAX)),
            ),
            Err(CampaignError::PhysicalSizeOverflow("peak resident"))
        );
    }

    #[test]
    fn physical_report_remeasures_file_and_detects_change() {
        let unique = format!(
            "tritium-physical-report-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        let original = b"exact-package";
        fs::write(&path, original).expect("write package");
        let report = simple_physical_report(original, 64, 80, 8, 2, 3);
        report.verify_file(&path).expect("unchanged package");

        fs::write(&path, b"exact-packagf").expect("mutate same length");
        assert!(matches!(
            report.verify_file(&path),
            Err(CampaignError::PackageMeasurementMismatch {
                expected_bytes: 13,
                actual_bytes: 13,
                ..
            })
        ));
        fs::remove_file(path).expect("remove package");
    }

    #[test]
    fn physical_report_handles_mixed_planes_short_groups_and_alignment() {
        for coefficient_count in 1_u64..=257 {
            let mut remaining = coefficient_count;
            let mut group_index = 0_u64;
            let mut payload_bytes = 0_u64;
            let mut scale_bytes = 0_u64;
            let mut logical_bits = 0_u64;
            while remaining != 0 {
                let group_len = remaining.min(128);
                let planes = 1 + group_index % 3;
                // D2: each independently addressable plane is byte-ceiled.
                payload_bytes += planes * (group_len * 2).div_ceil(8);
                scale_bytes += planes * 2;
                logical_bits += planes * group_len;
                remaining -= group_len;
                group_index += 1;
            }
            let group_count = coefficient_count.div_ceil(128);
            let allocation_bits = group_count * 2;
            let allocation_bytes = allocation_bits / 8;
            let allocation_embedded_bits = allocation_bits % 8;
            let header_bytes = 13;
            let transform_bytes = 5;
            let unaligned =
                payload_bytes + scale_bytes + allocation_bytes + header_bytes + transform_bytes;
            let alignment_bytes = (64 - unaligned % 64) % 64;
            let package_bytes = unaligned + alignment_bytes;
            let bytes = vec![0_u8; usize::try_from(package_bytes).expect("small fixture")];
            let report = PhysicalSizeReport::new(
                MeasuredPackage::from_bytes(&bytes).expect("package"),
                coefficient_count,
                coefficient_count + 11,
                logical_trits(logical_bits),
                SerializedSizeComponents::new(
                    payload_bytes,
                    scale_bytes,
                    allocation_bytes,
                    allocation_bits,
                    allocation_embedded_bits,
                    header_bytes,
                    transform_bytes,
                    0,
                    alignment_bytes,
                ),
                ResidentSizeComponents::new(
                    payload_bytes + scale_bytes,
                    allocation_bytes,
                    allocation_bits,
                    allocation_embedded_bits,
                    header_bytes,
                    11,
                    transform_bytes,
                    Some(coefficient_count % 17),
                ),
            )
            .expect("mixed report");

            assert_eq!(report.package().physical_bytes(), package_bytes);
            assert_eq!(report.serialized_core_bytes(), package_bytes);
            assert_eq!(
                report.peak_resident_bytes(),
                Some(report.steady_resident_bytes() + coefficient_count % 17)
            );
            assert_eq!(
                report.serialized_core_millibpw().numerator_millibits(),
                u128::from(package_bytes) * 8_000
            );
        }
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
    fn metrics_validate_exact_peak_and_ledger_validates_package_binding() {
        let bytes = b"physical-size";
        let report = simple_physical_report(bytes, 64, 80, 8, 2, 3);
        let peak = report.peak_resident_bytes().expect("measured peak");
        assert_eq!(
            CampaignMetrics::new(2.0)
                .expect("metrics")
                .with_peak_vram_bytes(peak + 1)
                .expect("legacy peak")
                .with_physical_size_report(report),
            Err(CampaignError::PhysicalMetricMismatch("peak_vram_bytes"))
        );
        assert_eq!(
            CampaignMetrics::new(2.0)
                .expect("metrics")
                .with_physical_size_report(report)
                .expect("report")
                .with_peak_vram_bytes(peak + 1),
            Err(CampaignError::PhysicalMetricMismatch("peak_vram_bytes"))
        );

        let source = model_id(1);
        let calibration = calibration([1; 32]);
        let evaluation = evaluation();
        let metrics = CampaignMetrics::new(2.0)
            .expect("metrics")
            .with_physical_size_report(report)
            .expect("report");
        assert_eq!(metrics.peak_vram_bytes(), Some(peak));
        let mismatched_point = CampaignPoint::new(
            source,
            model_id(2),
            MeasuredPackage::from_bytes(b"different-package").expect("different"),
            recipe(2),
            calibration.id(),
            evaluation.id(),
            metrics,
        );
        let mut ledger = CampaignLedger::new(source, calibration, evaluation);
        assert!(matches!(
            ledger.add(mismatched_point),
            Err(CampaignError::PackageMeasurementMismatch { .. })
        ));
    }

    #[test]
    fn physical_reports_use_v4_and_bind_component_allocation() {
        let source = model_id(1);
        let calibration = calibration([1; 32]);
        let evaluation = evaluation();
        let bytes = b"component-ledger";
        let package = MeasuredPackage::from_bytes(bytes).expect("package");
        let make_ledger = |header_bytes: u64, transform_bytes: u64| {
            let report = PhysicalSizeReport::new(
                package,
                64,
                80,
                logical_trits(64),
                SerializedSizeComponents::new(8, 2, 0, 0, 0, header_bytes, transform_bytes, 0, 0),
                ResidentSizeComponents::new(10, 0, 0, 0, 3, 3, 0, Some(7)),
            )
            .expect("report");
            let metrics = CampaignMetrics::new(2.0)
                .expect("metrics")
                .with_physical_size_report(report)
                .expect("report metric");
            let point = CampaignPoint::new(
                source,
                model_id(2),
                package,
                recipe(2),
                calibration.id(),
                evaluation.id(),
                metrics,
            );
            let mut ledger = CampaignLedger::new(source, calibration.clone(), evaluation.clone());
            ledger.add(point).expect("point");
            ledger
        };
        // `component-ledger` is 16 bytes: 8 payload + 2 scale + 6 metadata.
        let headers = make_ledger(6, 0);
        let transforms = make_ledger(0, 6);

        assert_eq!(
            headers.canonical_bytes().expect("bytes")[4],
            PHYSICAL_REPORT_CAMPAIGN_VERSION
        );
        assert_ne!(
            headers.id().expect("header id"),
            transforms.id().expect("transform id")
        );
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
    fn logical_pareto_uses_exact_trit_ratios_beyond_f64_integer_precision() {
        let source = model_id(1);
        let calibration = calibration([1; 32]);
        let evaluation = evaluation();
        let denominator = (1_u64 << 53) + 2;
        let make_point = |seed: u8, trits: u64| {
            let bytes = [seed];
            let package = MeasuredPackage::from_bytes(&bytes).expect("package");
            let report = PhysicalSizeReport::new(
                package,
                denominator,
                denominator,
                logical_trits(trits),
                SerializedSizeComponents::new(1, 0, 0, 0, 0, 0, 0, 0, 0),
                ResidentSizeComponents::new(1, 0, 0, 0, 0, 0, 0, None),
            )
            .expect("report");
            let metrics = CampaignMetrics::new(42.0)
                .expect("placeholder display")
                .with_physical_size_report(report)
                .expect("physical report");
            assert_eq!(metrics.logical_bpw(), report.logical_core_rate().bpw());
            CampaignPoint::new(
                source,
                model_id(seed),
                package,
                recipe(seed),
                calibration.id(),
                evaluation.id(),
                metrics,
            )
        };
        let lower = make_point(2, 1_u64 << 53);
        let higher = make_point(3, (1_u64 << 53) + 1);
        assert_eq!(
            lower.metrics().logical_bpw(),
            higher.metrics().logical_bpw(),
            "the fixture must collide after f64 conversion"
        );
        let mut ledger = CampaignLedger::new(source, calibration, evaluation);
        ledger.add(higher).expect("higher point");
        ledger.add(lower.clone()).expect("lower point");

        let frontier = ledger
            .pareto_frontier(&[CampaignObjective::LogicalBpw])
            .expect("exact logical frontier");
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].package_id(), lower.package_id());
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

        let forward_bytes = forward.canonical_bytes().expect("encode");
        assert_eq!(forward_bytes[4], LEGACY_CAMPAIGN_VERSION);
        assert_eq!(forward_bytes, reverse.canonical_bytes().expect("encode"));
        assert_eq!(forward.id().expect("id"), reverse.id().expect("id"));
        assert_eq!(
            forward.id().expect("id").to_string(),
            "trl1_c536022c10e224592e9817fe7664a644a66828a7ddc002fc4c69adecb2aefad2"
        );
    }
}
