//! Reproducible calibration, evaluation, and multi-objective campaign records.

use core::{cmp::Ordering, fmt};
use tritium_format::{ModelId, PackageId};

const CALIBRATION_MAGIC: [u8; 4] = *b"TCAL";
const CAMPAIGN_MAGIC: [u8; 4] = *b"TCMP";
const EVALUATION_MAGIC: [u8; 4] = *b"TEVL";
const PROVENANCE_VERSION: u8 = 1;

fn write_string(out: &mut Vec<u8>, value: &str) {
    let len = u32::try_from(value.len()).expect("provenance constructor validates string length");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
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
    pub fn id(&self) -> PackageId {
        PackageId::from_package_bytes(&self.canonical_bytes())
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
    /// `harness_digest` covers scoring code and metric configuration.
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

    /// Digest of evaluation code and metric configuration.
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
    pub fn id(&self) -> PackageId {
        PackageId::from_package_bytes(&self.canonical_bytes())
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
}

/// Required storage metrics plus optional quality measurements for one artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct CampaignMetrics {
    physical_bytes: u64,
    logical_bpw: f64,
    perplexity: Option<f64>,
}

impl CampaignMetrics {
    /// Build metrics with exact artifact bytes and logical allocator rate.
    ///
    /// # Errors
    /// Returns [`CampaignError::InvalidMetric`] for zero bytes, or for a
    /// non-finite/non-positive logical rate.
    pub fn new(physical_bytes: u64, logical_bpw: f64) -> Result<Self, CampaignError> {
        if physical_bytes == 0 {
            return Err(CampaignError::InvalidMetric("physical_bytes"));
        }
        validate_positive_finite("logical_bpw", logical_bpw)?;
        Ok(Self {
            physical_bytes,
            logical_bpw,
            perplexity: None,
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

    /// Exact serialized inference-artifact size.
    pub fn physical_bytes(&self) -> u64 {
        self.physical_bytes
    }

    /// Logical bits per source weight assigned by the allocator.
    pub fn logical_bpw(&self) -> f64 {
        self.logical_bpw
    }

    /// Held-out perplexity, when measured.
    pub fn perplexity(&self) -> Option<f64> {
        self.perplexity
    }
}

/// One immutable converted artifact and its measured campaign metrics.
#[derive(Clone, Debug, PartialEq)]
pub struct CampaignPoint {
    source_model_id: ModelId,
    model_id: ModelId,
    package_id: PackageId,
    recipe_id: [u8; 32],
    calibration_id: PackageId,
    evaluation_id: PackageId,
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
        package_id: PackageId,
        recipe_id: [u8; 32],
        calibration_id: PackageId,
        evaluation_id: PackageId,
        metrics: CampaignMetrics,
    ) -> Self {
        Self {
            source_model_id,
            model_id,
            package_id,
            recipe_id,
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
        self.package_id
    }

    /// Conversion recipe identity.
    pub fn recipe_id(&self) -> &[u8; 32] {
        &self.recipe_id
    }

    /// Calibration provenance identity.
    pub fn calibration_id(&self) -> PackageId {
        self.calibration_id
    }

    /// Evaluation provenance identity.
    pub fn evaluation_id(&self) -> PackageId {
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
    calibration_id: PackageId,
    evaluation: EvaluationProvenance,
    evaluation_id: PackageId,
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
    pub fn calibration_id(&self) -> PackageId {
        self.calibration_id
    }

    /// Pinned evaluation provenance.
    pub fn evaluation(&self) -> &EvaluationProvenance {
        &self.evaluation
    }

    /// Cached identity of pinned evaluation provenance.
    pub fn evaluation_id(&self) -> PackageId {
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
            .any(|existing| existing.package_id == point.package_id)
        {
            return Err(CampaignError::DuplicatePackage(point.package_id));
        }
        self.points.push(point);
        Ok(())
    }

    /// Return non-dominated measured points in deterministic objective order.
    ///
    /// All current objectives are minimized. A point dominates another when it
    /// is no worse on every requested objective and strictly better on at least
    /// one. No interpolation or extrapolation is performed. General
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
                if !has_objective(&point.metrics, objective) {
                    return Err(CampaignError::MissingObjective {
                        package_id: point.package_id,
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
                    other.package_id != candidate.package_id
                        && dominates(&other.metrics, &candidate.metrics, objectives)
                })
            })
            .collect();
        frontier.sort_by(|left, right| {
            for &objective in objectives {
                let ordering = objective_cmp(&left.metrics, &right.metrics, objective);
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            left.package_id.as_bytes().cmp(right.package_id.as_bytes())
        });
        Ok(frontier)
    }

    /// Serialize a canonical campaign record independent of point insertion order.
    ///
    /// Points are ordered by exact [`PackageId`]. Calibration and evaluation
    /// records are embedded, making the artifact self-contained for audit. This
    /// version-1 encoding is a stable hash target, not yet an interchange format;
    /// no public decoder is provided.
    ///
    /// # Errors
    /// Returns [`CampaignError::RecordTooLarge`] if a version-1 u32 count or
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
        points.sort_by(|left, right| left.package_id.as_bytes().cmp(right.package_id.as_bytes()));

        let mut out = Vec::new();
        out.extend_from_slice(&CAMPAIGN_MAGIC);
        out.push(PROVENANCE_VERSION);
        out.extend_from_slice(self.source_model_id.as_bytes());
        out.extend_from_slice(&calibration_len.to_le_bytes());
        out.extend_from_slice(&calibration);
        out.extend_from_slice(&evaluation_len.to_le_bytes());
        out.extend_from_slice(&evaluation);
        out.extend_from_slice(&point_count.to_le_bytes());
        for point in points {
            out.extend_from_slice(point.model_id.as_bytes());
            out.extend_from_slice(point.package_id.as_bytes());
            out.extend_from_slice(&point.recipe_id);
            out.extend_from_slice(&point.metrics.physical_bytes.to_le_bytes());
            out.extend_from_slice(&point.metrics.logical_bpw.to_bits().to_le_bytes());
            match point.metrics.perplexity {
                None => out.push(0),
                Some(perplexity) => {
                    out.push(1);
                    out.extend_from_slice(&perplexity.to_bits().to_le_bytes());
                }
            }
        }
        Ok(out)
    }

    /// Exact content identity of the canonical campaign record.
    ///
    /// # Errors
    /// Returns the same errors as [`Self::canonical_bytes`].
    pub fn id(&self) -> Result<PackageId, CampaignError> {
        Ok(PackageId::from_package_bytes(&self.canonical_bytes()?))
    }
}

fn has_objective(metrics: &CampaignMetrics, objective: CampaignObjective) -> bool {
    match objective {
        CampaignObjective::PhysicalBytes | CampaignObjective::LogicalBpw => true,
        CampaignObjective::Perplexity => metrics.perplexity.is_some(),
    }
}

fn objective_cmp(
    left: &CampaignMetrics,
    right: &CampaignMetrics,
    objective: CampaignObjective,
) -> Ordering {
    match objective {
        CampaignObjective::PhysicalBytes => left.physical_bytes.cmp(&right.physical_bytes),
        CampaignObjective::LogicalBpw => left.logical_bpw.total_cmp(&right.logical_bpw),
        CampaignObjective::Perplexity => left
            .perplexity
            .expect("objective presence validated")
            .total_cmp(&right.perplexity.expect("objective presence validated")),
    }
}

fn dominates(
    left: &CampaignMetrics,
    right: &CampaignMetrics,
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
    /// Metric was missing physical support or was non-finite/non-positive.
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
    /// Version-1 campaign record exceeded a u32 count or embedded length.
    RecordTooLarge(&'static str),
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
                write!(f, "campaign record `{field}` exceeds version-1 capacity")
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
    fn ledger_rejects_mixed_provenance_and_duplicate_packages() {
        let source = model_id(1);
        let calibration_provenance = calibration([1; 32]);
        let evaluation = evaluation();
        let mut ledger =
            CampaignLedger::new(source, calibration_provenance.clone(), evaluation.clone());
        let metrics = CampaignMetrics::new(4_000_000_000, 2.1)
            .expect("valid metrics")
            .with_perplexity(8.2)
            .expect("valid perplexity");
        let wrong_calibration = CampaignPoint::new(
            source,
            model_id(2),
            PackageId::from_package_bytes(b"candidate-a"),
            [9; 32],
            calibration([9; 32]).id(),
            evaluation.id(),
            metrics.clone(),
        );
        assert!(matches!(
            ledger.add(wrong_calibration),
            Err(CampaignError::ProvenanceMismatch("calibration"))
        ));

        let package = PackageId::from_package_bytes(b"candidate-b");
        let point = CampaignPoint::new(
            source,
            model_id(3),
            package,
            [8; 32],
            calibration_provenance.id(),
            evaluation.id(),
            metrics,
        );
        ledger.add(point.clone()).expect("first package");
        assert!(matches!(
            ledger.add(point),
            Err(CampaignError::DuplicatePackage(id)) if id == package
        ));
    }

    #[test]
    fn pareto_frontier_keeps_measured_storage_quality_tradeoffs() {
        let source = model_id(1);
        let calibration = calibration([1; 32]);
        let evaluation = evaluation();
        let mut ledger = CampaignLedger::new(source, calibration.clone(), evaluation.clone());
        let make_point = |seed: u8, bytes: u64, perplexity: f64| {
            CampaignPoint::new(
                source,
                model_id(seed),
                PackageId::from_package_bytes(&[seed]),
                [seed; 32],
                calibration.id(),
                evaluation.id(),
                CampaignMetrics::new(bytes, 2.0)
                    .expect("metrics")
                    .with_perplexity(perplexity)
                    .expect("perplexity"),
            )
        };
        let compact = make_point(2, 70_000_000, 10.5);
        let accurate = make_point(3, 80_000_000, 9.8);
        let dominated = make_point(4, 100_000_000, 11.0);
        let fp_sized = make_point(5, 540_000_000, 10.0);
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
            CampaignPoint::new(
                source,
                model_id(seed),
                PackageId::from_package_bytes(&[seed]),
                [seed; 32],
                calibration.id(),
                evaluation.id(),
                CampaignMetrics::new(u64::from(seed) * 1_000_000, 2.0)
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
            "trp1_f6ff8a4a2e98622f1b21bc77441fece6e8946ab02a18345785414b4d8507bf14"
        );
    }
}
