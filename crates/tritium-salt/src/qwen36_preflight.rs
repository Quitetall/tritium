//! All-or-nothing typestate admission for the declared Qwen3.6-27B SALT campaign.
//!
//! Callers must keep the opened checkpoint files stable for the duration of
//! preflight. Same-handle streaming resists path replacement and later tensor
//! reads detect in-place changes, but this API is not a filesystem snapshot
//! against a concurrently mutating writer.

use core::fmt;
use std::path::Path;

use tritium_format::{ModelId, SemanticModelManifest};
use tritium_nn::{
    NnError, QWEN36_27B_REPOSITORY, QWEN36_27B_REVISION, Qwen35ContentVerifiedHfSource,
    Qwen35HfLanguageReceipt, Qwen35HfSource, Qwen35HfSourceIdentity, Qwen35HfTensorMetadata,
};
use tritium_quantize::{
    QWEN36_27B_COVERAGE_REVISION, Qwen35CoverageDisposition, Qwen35CoverageError,
    Qwen35CoverageManifest, Qwen35CoverageSummary, Qwen35TensorMetadata, Qwen35TensorScope,
};

/// Authentication status of the measured Qwen3.6 source payload.
///
/// The official expected [`ModelId`] has not yet been frozen. Exact config,
/// metadata, and a caller-declared revision do not authenticate payload bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Qwen36SourceIdentityStatus {
    /// Content is measured and bound, but still awaits an independently audited official ID.
    MeasuredAwaitingOfficialRegistration,
}

impl Qwen36SourceIdentityStatus {
    /// Stable machine-readable status label for evidence records.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MeasuredAwaitingOfficialRegistration => "measured-awaiting-official-registration",
        }
    }

    /// Whether independently audited official payload identity has been matched.
    #[must_use]
    pub const fn official_payload_authenticated(self) -> bool {
        match self {
            Self::MeasuredAwaitingOfficialRegistration => false,
        }
    }
}

/// Scalar evidence emitted by a completed Qwen3.6 campaign-candidate preflight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36CampaignPreflightReceipt {
    source_model_id: ModelId,
    payload_bytes: u64,
    metadata_digest: [u8; 32],
    metadata_record_bytes: u64,
    coverage: Qwen35CoverageSummary,
    language: Qwen35HfLanguageReceipt,
    identity_status: Qwen36SourceIdentityStatus,
}

impl Qwen36CampaignPreflightReceipt {
    /// Declared Hugging Face repository fixed by this campaign policy.
    #[must_use]
    pub const fn repository(&self) -> &'static str {
        QWEN36_27B_REPOSITORY
    }

    /// Declared immutable revision fixed by this campaign policy.
    #[must_use]
    pub const fn revision(&self) -> &'static str {
        QWEN36_27B_REVISION
    }

    /// Measured semantic model identity; never supplied by the caller.
    #[must_use]
    pub const fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Exact stored payload bytes streamed into the semantic identity.
    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Frozen digest of the exact 1,199-entry metadata policy.
    #[must_use]
    pub const fn metadata_digest(&self) -> &[u8; 32] {
        &self.metadata_digest
    }

    /// Canonical metadata record bytes covered by [`Self::metadata_digest`].
    #[must_use]
    pub const fn metadata_record_bytes(&self) -> u64 {
        self.metadata_record_bytes
    }

    /// Frozen component and conversion-disposition totals.
    #[must_use]
    pub const fn coverage(&self) -> Qwen35CoverageSummary {
        self.coverage
    }

    /// Exact generic language schema counts checked against pinned coverage.
    #[must_use]
    pub const fn language(&self) -> &Qwen35HfLanguageReceipt {
        &self.language
    }

    /// Whether the measured ID has been matched to independently audited official bytes.
    #[must_use]
    pub const fn identity_status(&self) -> Qwen36SourceIdentityStatus {
        self.identity_status
    }

    /// Total source tensors admitted by pinned coverage.
    #[must_use]
    pub const fn total_tensors(&self) -> u64 {
        self.coverage.total().tensors()
    }

    /// Language and MTP tensors included in the current product scope.
    #[must_use]
    pub const fn included_tensors(&self) -> u64 {
        self.coverage.included().tensors()
    }
}

/// Revision-declared Qwen3.6 campaign candidate with inseparable content and policy proof.
///
/// Construction validates the pinned config and complete 1,199-tensor metadata
/// before the expensive payload scan, then joins the measured source manifest,
/// byte count, and language receipt. It does not yet prove the bytes came from
/// the official repository revision; see [`Qwen36SourceIdentityStatus`].
#[derive(Debug)]
pub struct Qwen36CampaignPreflight {
    source: Qwen35ContentVerifiedHfSource,
    coverage: Qwen35CoverageManifest,
    receipt: Qwen36CampaignPreflightReceipt,
}

impl Qwen36CampaignPreflight {
    /// Open and admit a local candidate for the declared Qwen3.6-27B campaign.
    ///
    /// A wrong revision fails before any filesystem access. Config and metadata
    /// failures occur before the complete payload identity scan.
    ///
    /// The caller must prevent concurrent in-place mutation of the checkpoint
    /// during this operation; the retained handle is not a filesystem snapshot.
    ///
    /// # Errors
    /// Returns [`Qwen36CampaignPreflightError`] for a wrong/internal pin,
    /// invalid source/config/coverage, or a contradiction between content
    /// identity and pinned metadata policy.
    pub fn open(
        directory: &Path,
        declared_revision: &str,
    ) -> Result<Self, Qwen36CampaignPreflightError> {
        validate_revision(declared_revision)?;
        let source =
            Qwen35HfSource::open(directory).map_err(Qwen36CampaignPreflightError::SourceOpen)?;
        source
            .config()
            .validate_pinned_qwen36_27b(declared_revision)
            .map_err(Qwen36CampaignPreflightError::PinnedConfig)?;
        let coverage = build_coverage(source.metadata())?;
        let source = source
            .verify_semantic_identity()
            .map_err(Qwen36CampaignPreflightError::SourceIdentity)?;
        Self::finish(source, coverage)
    }

    /// Join an already content-verified source to the declared pinned campaign policy.
    ///
    /// This path preserves the same content/source typestate, but cannot recover
    /// the cheap-failure ordering once the caller has already performed the scan.
    ///
    /// # Errors
    /// Returns [`Qwen36CampaignPreflightError`] for a wrong/internal pin,
    /// invalid pinned config/coverage, or a content-policy contradiction.
    pub fn from_content_verified(
        source: Qwen35ContentVerifiedHfSource,
        declared_revision: &str,
    ) -> Result<Self, Qwen36CampaignPreflightError> {
        validate_revision(declared_revision)?;
        source
            .config()
            .validate_pinned_qwen36_27b(declared_revision)
            .map_err(Qwen36CampaignPreflightError::PinnedConfig)?;
        let coverage = build_coverage(source.metadata())?;
        Self::finish(source, coverage)
    }

    /// Scalar preflight evidence.
    #[must_use]
    pub const fn receipt(&self) -> &Qwen36CampaignPreflightReceipt {
        &self.receipt
    }

    /// Complete pinned conversion policy, in canonical tensor-name order.
    #[must_use]
    pub const fn coverage(&self) -> &Qwen35CoverageManifest {
        &self.coverage
    }

    /// Measured semantic source identity and retained manifest.
    #[must_use]
    pub const fn source_identity(&self) -> &Qwen35HfSourceIdentity {
        self.source.identity()
    }

    /// Versioned canonical typed configuration bytes bound into the source manifest.
    #[must_use]
    pub fn canonical_config_bytes(&self) -> &[u8] {
        self.source.canonical_config_bytes()
    }

    /// Measured semantic model ID.
    #[must_use]
    pub fn source_model_id(&self) -> ModelId {
        self.source.model_id()
    }

    /// Canonical source manifest joined against exact pinned coverage.
    #[must_use]
    pub const fn source_manifest(&self) -> &SemanticModelManifest {
        self.source.identity().manifest()
    }

    /// Widen one source tensor while re-verifying the exact consumed payload chunks.
    ///
    /// Keeping this operation on the campaign wrapper prevents conversion code
    /// from discarding the coverage and receipt typestate.
    ///
    /// # Errors
    /// Returns [`Qwen36CampaignPreflightError::SourceIdentity`] if the source
    /// changed or widening otherwise fails.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, Qwen36CampaignPreflightError> {
        self.source
            .tensor_f32(name)
            .map_err(Qwen36CampaignPreflightError::SourceIdentity)
    }

    fn finish(
        source: Qwen35ContentVerifiedHfSource,
        coverage: Qwen35CoverageManifest,
    ) -> Result<Self, Qwen36CampaignPreflightError> {
        validate_manifest_entries(
            coverage
                .entries()
                .iter()
                .map(|entry| (entry.name(), entry.shape())),
            coverage.entries().len(),
            source.identity().manifest(),
        )?;
        validate_payload_bytes(
            coverage.expected_source_payload_bytes(),
            source.identity().payload_bytes(),
        )?;
        let expected_language = expected_language_counts(&coverage)?;
        let actual_language = language_counts(source.language_receipt())?;
        validate_language_counts(expected_language, actual_language)?;

        let receipt = Qwen36CampaignPreflightReceipt {
            source_model_id: source.model_id(),
            payload_bytes: source.identity().payload_bytes(),
            metadata_digest: *coverage.metadata_digest(),
            metadata_record_bytes: coverage.metadata_record_bytes(),
            coverage: coverage.summary(),
            language: *source.language_receipt(),
            identity_status: Qwen36SourceIdentityStatus::MeasuredAwaitingOfficialRegistration,
        };
        Ok(Self {
            source,
            coverage,
            receipt,
        })
    }
}

/// Why a Qwen3.6 campaign-candidate source failed preflight.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Qwen36CampaignPreflightError {
    /// The caller did not declare the immutable campaign revision.
    WrongRevision,
    /// NN and quantization crates were compiled with contradictory pin constants.
    InternalPinMismatch,
    /// Opening config or shard headers failed.
    SourceOpen(NnError),
    /// The typed config does not equal the pinned Qwen3.6-27B geometry.
    PinnedConfig(NnError),
    /// Exact 1,199-tensor metadata coverage failed.
    Coverage(Qwen35CoverageError),
    /// Streaming or re-reading measured source content failed.
    SourceIdentity(NnError),
    /// A checked count or byte-size conversion overflowed.
    ArithmeticOverflow(&'static str),
    /// Numeric evidence contradicted pinned policy.
    ContractMismatch {
        /// Rejected contract field.
        field: &'static str,
        /// Value required by policy.
        expected: u64,
        /// Measured value.
        actual: u64,
    },
    /// One canonical manifest entry contradicted its coverage entry.
    SemanticTensorMismatch {
        /// Zero-based canonical entry index.
        index: usize,
        /// Contradictory field (`name` or `shape`).
        field: &'static str,
    },
}

impl fmt::Display for Qwen36CampaignPreflightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongRevision => formatter.write_str("Qwen3.6 campaign revision is not pinned"),
            Self::InternalPinMismatch => {
                formatter.write_str("Qwen3.6 revision constants disagree across crates")
            }
            Self::SourceOpen(error) => write!(formatter, "open Qwen3.6 source: {error}"),
            Self::PinnedConfig(error) => write!(formatter, "pin Qwen3.6 config: {error}"),
            Self::Coverage(error) => write!(formatter, "cover Qwen3.6 source metadata: {error}"),
            Self::SourceIdentity(error) => {
                write!(formatter, "verify Qwen3.6 source content: {error}")
            }
            Self::ArithmeticOverflow(field) => {
                write!(formatter, "Qwen3.6 preflight {field} overflow")
            }
            Self::ContractMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "Qwen3.6 preflight {field} mismatch: expected {expected}, got {actual}"
            ),
            Self::SemanticTensorMismatch { index, field } => write!(
                formatter,
                "Qwen3.6 semantic tensor {index} has mismatched {field}"
            ),
        }
    }
}

impl std::error::Error for Qwen36CampaignPreflightError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SourceOpen(error) | Self::PinnedConfig(error) | Self::SourceIdentity(error) => {
                Some(error)
            }
            Self::Coverage(error) => Some(error),
            Self::WrongRevision
            | Self::InternalPinMismatch
            | Self::ArithmeticOverflow(_)
            | Self::ContractMismatch { .. }
            | Self::SemanticTensorMismatch { .. } => None,
        }
    }
}

fn validate_revision(declared_revision: &str) -> Result<(), Qwen36CampaignPreflightError> {
    if QWEN36_27B_REVISION != QWEN36_27B_COVERAGE_REVISION {
        return Err(Qwen36CampaignPreflightError::InternalPinMismatch);
    }
    if declared_revision != QWEN36_27B_REVISION {
        return Err(Qwen36CampaignPreflightError::WrongRevision);
    }
    Ok(())
}

fn build_coverage(
    metadata: &[Qwen35HfTensorMetadata],
) -> Result<Qwen35CoverageManifest, Qwen36CampaignPreflightError> {
    Qwen35CoverageManifest::from_metadata(
        QWEN36_27B_COVERAGE_REVISION,
        metadata
            .iter()
            .map(|tensor| Qwen35TensorMetadata::new(tensor.name(), tensor.dtype(), tensor.shape())),
    )
    .map_err(Qwen36CampaignPreflightError::Coverage)
}

fn validate_manifest_entries<'a>(
    coverage: impl Iterator<Item = (&'a str, &'a [u64])>,
    expected_len: usize,
    manifest: &SemanticModelManifest,
) -> Result<(), Qwen36CampaignPreflightError> {
    if manifest.tensors().len() != expected_len {
        return Err(Qwen36CampaignPreflightError::ContractMismatch {
            field: "semantic tensor count",
            expected: u64::try_from(expected_len).map_err(|_| {
                Qwen36CampaignPreflightError::ArithmeticOverflow("coverage tensor count")
            })?,
            actual: u64::try_from(manifest.tensors().len()).map_err(|_| {
                Qwen36CampaignPreflightError::ArithmeticOverflow("manifest tensor count")
            })?,
        });
    }
    for (index, ((expected_name, expected_shape), actual)) in
        coverage.zip(manifest.tensors()).enumerate()
    {
        if actual.name() != expected_name {
            return Err(Qwen36CampaignPreflightError::SemanticTensorMismatch {
                index,
                field: "name",
            });
        }
        if actual.shape() != expected_shape {
            return Err(Qwen36CampaignPreflightError::SemanticTensorMismatch {
                index,
                field: "shape",
            });
        }
    }
    Ok(())
}

fn validate_payload_bytes(expected: u64, actual: u64) -> Result<(), Qwen36CampaignPreflightError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Qwen36CampaignPreflightError::ContractMismatch {
            field: "source payload bytes",
            expected,
            actual,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LanguageCounts {
    tensors: u64,
    matrices: u64,
    preserved: u64,
    deferred_mtp: u64,
    deferred_vision: u64,
}

fn expected_language_counts(
    coverage: &Qwen35CoverageManifest,
) -> Result<LanguageCounts, Qwen36CampaignPreflightError> {
    let matrices = coverage
        .entries()
        .iter()
        .filter(|entry| {
            entry.scope() == Qwen35TensorScope::Language
                && entry.disposition() == Qwen35CoverageDisposition::AdditiveTernary
        })
        .count();
    let preserved = coverage
        .entries()
        .iter()
        .filter(|entry| {
            entry.scope() == Qwen35TensorScope::Language
                && entry.disposition() == Qwen35CoverageDisposition::PreserveSource
        })
        .count();
    Ok(LanguageCounts {
        tensors: coverage.summary().language().tensors(),
        matrices: u64::try_from(matrices).map_err(|_| {
            Qwen36CampaignPreflightError::ArithmeticOverflow("language matrix count")
        })?,
        preserved: u64::try_from(preserved).map_err(|_| {
            Qwen36CampaignPreflightError::ArithmeticOverflow("language preserved count")
        })?,
        deferred_mtp: coverage.summary().mtp().tensors(),
        deferred_vision: coverage.summary().vision().tensors(),
    })
}

fn language_counts(
    receipt: &Qwen35HfLanguageReceipt,
) -> Result<LanguageCounts, Qwen36CampaignPreflightError> {
    Ok(LanguageCounts {
        tensors: count_u64(receipt.language_tensors(), "language tensor count")?,
        matrices: count_u64(receipt.language_matrices(), "language matrix count")?,
        preserved: count_u64(
            receipt.language_preserved_tensors(),
            "language preserved count",
        )?,
        deferred_mtp: count_u64(receipt.deferred_mtp_tensors(), "deferred MTP count")?,
        deferred_vision: count_u64(receipt.deferred_vision_tensors(), "deferred vision count")?,
    })
}

fn count_u64(value: usize, field: &'static str) -> Result<u64, Qwen36CampaignPreflightError> {
    u64::try_from(value).map_err(|_| Qwen36CampaignPreflightError::ArithmeticOverflow(field))
}

fn validate_language_counts(
    expected: LanguageCounts,
    actual: LanguageCounts,
) -> Result<(), Qwen36CampaignPreflightError> {
    for (field, expected, actual) in [
        ("language tensors", expected.tensors, actual.tensors),
        ("language matrices", expected.matrices, actual.matrices),
        (
            "language preserved tensors",
            expected.preserved,
            actual.preserved,
        ),
        (
            "deferred MTP tensors",
            expected.deferred_mtp,
            actual.deferred_mtp,
        ),
        (
            "deferred vision tensors",
            expected.deferred_vision,
            actual.deferred_vision,
        ),
    ] {
        if expected != actual {
            return Err(Qwen36CampaignPreflightError::ContractMismatch {
                field,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use tritium_format::{SemanticModelManifest, SemanticTensor};

    use super::*;

    fn manifest(entries: &[(&str, &[u64])]) -> SemanticModelManifest {
        let tensors = entries
            .iter()
            .map(|&(name, shape)| SemanticTensor::new(name, shape.to_vec(), b"fixture").unwrap())
            .collect();
        SemanticModelManifest::new("fixture", b"config", tensors).unwrap()
    }

    #[test]
    fn internal_revision_pins_agree() {
        validate_revision(QWEN36_27B_REVISION).unwrap();
    }

    #[test]
    fn manifest_join_accepts_exact_names_shapes_and_order() {
        let entries = [("a", &[2, 3][..]), ("b", &[4][..])];
        let manifest = manifest(&entries);
        validate_manifest_entries(entries.iter().copied(), entries.len(), &manifest).unwrap();
    }

    #[test]
    fn manifest_join_rejects_count_name_and_shape_mismatches() {
        let expected = [("a", &[2, 3][..]), ("b", &[4][..])];

        let short = manifest(&expected[..1]);
        assert!(matches!(
            validate_manifest_entries(expected.iter().copied(), expected.len(), &short),
            Err(Qwen36CampaignPreflightError::ContractMismatch {
                field: "semantic tensor count",
                expected: 2,
                actual: 1,
            })
        ));

        let wrong_name = manifest(&[("a", &[2, 3]), ("c", &[4])]);
        assert!(matches!(
            validate_manifest_entries(expected.iter().copied(), expected.len(), &wrong_name),
            Err(Qwen36CampaignPreflightError::SemanticTensorMismatch {
                index: 1,
                field: "name",
            })
        ));

        let wrong_shape = manifest(&[("a", &[2, 3]), ("b", &[5])]);
        assert!(matches!(
            validate_manifest_entries(expected.iter().copied(), expected.len(), &wrong_shape),
            Err(Qwen36CampaignPreflightError::SemanticTensorMismatch {
                index: 1,
                field: "shape",
            })
        ));
    }

    #[test]
    fn payload_and_language_contracts_fail_closed() {
        assert!(validate_payload_bytes(55_562_855_904, 55_562_855_904).is_ok());
        assert!(matches!(
            validate_payload_bytes(55_562_855_904, 55_562_855_903),
            Err(Qwen36CampaignPreflightError::ContractMismatch {
                field: "source payload bytes",
                ..
            })
        ));

        let expected = LanguageCounts {
            tensors: 851,
            matrices: 498,
            preserved: 353,
            deferred_mtp: 15,
            deferred_vision: 333,
        };
        let mut actual = expected;
        actual.deferred_mtp = 14;
        assert!(matches!(
            validate_language_counts(expected, actual),
            Err(Qwen36CampaignPreflightError::ContractMismatch {
                field: "deferred MTP tensors",
                expected: 15,
                actual: 14,
            })
        ));
    }
}
