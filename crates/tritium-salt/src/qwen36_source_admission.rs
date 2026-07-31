//! Durable, content-addressed admission proof for a Qwen3.6 campaign candidate.

use core::fmt;
use std::{
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};

use tritium_format::{ArtifactError, ModelId, SemanticModelManifest};
use tritium_quantize::{
    Qwen35CoverageDisposition, Qwen35CoverageError, Qwen35CoverageManifest, Qwen35TensorScope,
};

use crate::{
    ContentId, Qwen36CampaignPreflight, Qwen36CampaignPreflightError, Qwen36SourceIdentityStatus,
    TensorWorkError,
    tensor_work_store::{absolute_path, create_temporary_file, ensure_durable_directory},
};

const PROOF_MAGIC: [u8; 8] = *b"TSQ36AD\0";
const PROOF_VERSION: u8 = 1;
const PROOF_CHECKSUM_CONTEXT: &str = "tritium qwen3.6 source admission checksum v1";
const SOURCE_DIRECTORY: &str = "qwen36-source";
const LOCK_DIRECTORY: &str = ".qwen36-source-locks";
const PROOF_FILE: &str = "ingest.tq36";
const MAX_PROOF_BYTES: usize = 64 * 1024 * 1024;
const MAX_CONFIG_BYTES: usize = 4 * 1024 * 1024;
const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
const MAX_POLICY_BYTES: usize = 16 * 1024 * 1024;
const CHECKSUM_BYTES: usize = 32;
#[cfg(test)]
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Exact language-schema counts retained by the durable admission proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen36LanguageCoverage {
    tensors: u64,
    matrices: u64,
    preserved: u64,
    deferred_mtp: u64,
    deferred_vision: u64,
}

impl Qwen36LanguageCoverage {
    /// Main language-model tensor count.
    #[must_use]
    pub const fn tensors(self) -> u64 {
        self.tensors
    }

    /// Language rank-2 matrices assigned additive ternary conversion.
    #[must_use]
    pub const fn matrices(self) -> u64 {
        self.matrices
    }

    /// Language tensors retained at source precision.
    #[must_use]
    pub const fn preserved(self) -> u64 {
        self.preserved
    }

    /// Bundled MTP tensors structurally covered but not proven executable here.
    #[must_use]
    pub const fn deferred_mtp(self) -> u64 {
        self.deferred_mtp
    }

    /// Vision tensors identity-bound and excluded from this product slice.
    #[must_use]
    pub const fn deferred_vision(self) -> u64 {
        self.deferred_vision
    }
}

/// Canonical, transport-independent source proof for one admitted Qwen3.6 candidate.
///
/// The proof embeds exact typed configuration bytes, the complete semantic
/// manifest, and the full per-tensor conversion policy. It remains explicitly
/// candidate-only until its [`ModelId`] is matched to independently audited
/// official payload bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36SourceProof {
    canonical_config: Vec<u8>,
    manifest: SemanticModelManifest,
    coverage: Qwen35CoverageManifest,
    payload_bytes: u64,
    language: Qwen36LanguageCoverage,
    identity_status: Qwen36SourceIdentityStatus,
}

impl Qwen36SourceProof {
    /// Materialize a canonical durable proof from an inseparable preflight capability.
    ///
    /// # Errors
    /// Returns [`Qwen36SourceProofError`] if a bounded record length overflows or
    /// the preflight's retained evidence is internally contradictory.
    pub fn from_preflight(
        preflight: &Qwen36CampaignPreflight,
    ) -> Result<Self, Qwen36SourceProofError> {
        let language = preflight.receipt().language();
        let proof = Self {
            canonical_config: preflight.canonical_config_bytes().to_vec(),
            manifest: preflight.source_manifest().clone(),
            coverage: preflight.coverage().clone(),
            payload_bytes: preflight.receipt().payload_bytes(),
            language: Qwen36LanguageCoverage {
                tensors: count_u64(language.language_tensors(), "language tensors")?,
                matrices: count_u64(language.language_matrices(), "language matrices")?,
                preserved: count_u64(
                    language.language_preserved_tensors(),
                    "language preserved tensors",
                )?,
                deferred_mtp: count_u64(language.deferred_mtp_tensors(), "deferred MTP")?,
                deferred_vision: count_u64(language.deferred_vision_tensors(), "deferred vision")?,
            },
            identity_status: preflight.receipt().identity_status(),
        };
        proof.validate()?;
        Ok(proof)
    }

    /// Restore only a canonical, checksum-valid, internally consistent proof.
    ///
    /// Decoding revalidates the typed-config digest, semantic manifest,
    /// complete current coverage policy, payload byte count, and all language,
    /// MTP, and vision counts. It does not authenticate the official repository
    /// payload; [`Self::identity_status`] remains authoritative.
    ///
    /// # Errors
    /// Returns [`Qwen36SourceProofError`] for oversized, corrupt, unsupported,
    /// noncanonical, or contract-contradictory bytes.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, Qwen36SourceProofError> {
        if bytes.len() > MAX_PROOF_BYTES {
            return Err(Qwen36SourceProofError::ProofTooLarge);
        }
        if bytes.len() < PROOF_MAGIC.len() + 1 + CHECKSUM_BYTES {
            return Err(Qwen36SourceProofError::Malformed("truncated header"));
        }
        let checksum_offset = bytes.len() - CHECKSUM_BYTES;
        let (payload, checksum) = bytes.split_at(checksum_offset);
        let mut hasher = blake3::Hasher::new_derive_key(PROOF_CHECKSUM_CONTEXT);
        hasher.update(payload);
        if hasher.finalize().as_bytes() != checksum {
            return Err(Qwen36SourceProofError::ChecksumMismatch);
        }

        let mut cursor = ProofCursor::new(payload);
        if cursor.take(PROOF_MAGIC.len())? != PROOF_MAGIC {
            return Err(Qwen36SourceProofError::Malformed("magic"));
        }
        let version = cursor.u8()?;
        if version != PROOF_VERSION {
            return Err(Qwen36SourceProofError::UnsupportedVersion(version));
        }
        if cursor.string(128, "repository")? != tritium_nn::QWEN36_27B_REPOSITORY {
            return Err(Qwen36SourceProofError::WrongRepository);
        }
        if cursor.string(128, "revision")? != tritium_nn::QWEN36_27B_REVISION {
            return Err(Qwen36SourceProofError::WrongRevision);
        }
        let identity_status = status_from_tag(cursor.u8()?)?;
        let payload_bytes = cursor.u64()?;
        let language = Qwen36LanguageCoverage {
            tensors: cursor.u64()?,
            matrices: cursor.u64()?,
            preserved: cursor.u64()?,
            deferred_mtp: cursor.u64()?,
            deferred_vision: cursor.u64()?,
        };
        let canonical_config = cursor
            .length_prefixed(MAX_CONFIG_BYTES, "canonical config")?
            .to_vec();
        let manifest = SemanticModelManifest::from_canonical_bytes(
            cursor.length_prefixed(MAX_MANIFEST_BYTES, "semantic manifest")?,
        )
        .map_err(Qwen36SourceProofError::Manifest)?;
        let coverage = Qwen35CoverageManifest::from_canonical_policy_bytes(
            cursor.length_prefixed(MAX_POLICY_BYTES, "coverage policy")?,
        )
        .map_err(Qwen36SourceProofError::Coverage)?;
        if cursor.remaining() != 0 {
            return Err(Qwen36SourceProofError::NonCanonical);
        }

        let proof = Self {
            canonical_config,
            manifest,
            coverage,
            payload_bytes,
            language,
            identity_status,
        };
        proof.validate()?;
        if proof.canonical_bytes()? != bytes {
            return Err(Qwen36SourceProofError::NonCanonical);
        }
        Ok(proof)
    }

    /// Unique canonical proof bytes, including an internal checksum.
    ///
    /// # Errors
    /// Returns [`Qwen36SourceProofError`] if a component exceeds its frozen
    /// bounded-record limit or the complete record exceeds 64 MiB.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, Qwen36SourceProofError> {
        self.validate()?;
        self.encode_unchecked()
    }

    fn encode_unchecked(&self) -> Result<Vec<u8>, Qwen36SourceProofError> {
        let manifest = self.manifest.canonical_bytes();
        let policy = self.coverage.canonical_policy_bytes();
        validate_component_lengths(&self.canonical_config, &manifest, &policy)?;

        let mut output = Vec::new();
        output.extend_from_slice(&PROOF_MAGIC);
        output.push(PROOF_VERSION);
        write_bytes(&mut output, tritium_nn::QWEN36_27B_REPOSITORY.as_bytes())?;
        write_bytes(&mut output, tritium_nn::QWEN36_27B_REVISION.as_bytes())?;
        output.push(status_tag(self.identity_status));
        output.extend_from_slice(&self.payload_bytes.to_le_bytes());
        for value in [
            self.language.tensors,
            self.language.matrices,
            self.language.preserved,
            self.language.deferred_mtp,
            self.language.deferred_vision,
        ] {
            output.extend_from_slice(&value.to_le_bytes());
        }
        write_bytes(&mut output, &self.canonical_config)?;
        write_bytes(&mut output, &manifest)?;
        write_bytes(&mut output, &policy)?;
        let mut hasher = blake3::Hasher::new_derive_key(PROOF_CHECKSUM_CONTEXT);
        hasher.update(&output);
        output.extend_from_slice(hasher.finalize().as_bytes());
        if output.len() > MAX_PROOF_BYTES {
            return Err(Qwen36SourceProofError::ProofTooLarge);
        }
        Ok(output)
    }

    /// Semantic identity of every measured source tensor and typed config.
    #[must_use]
    pub fn source_model_id(&self) -> ModelId {
        self.manifest.model_id()
    }

    /// SALT-domain content identity of the exact semantic-manifest preimage.
    ///
    /// This is intentionally derived from manifest bytes, not by relabeling or
    /// rehashing the already-domain-separated [`ModelId`] digest.
    #[must_use]
    pub fn manifest_content_id(&self) -> ContentId {
        ContentId::of_bytes(&self.manifest.canonical_bytes())
    }

    /// SALT-domain content identity of the complete canonical proof record.
    ///
    /// # Errors
    /// Returns [`Qwen36SourceProofError`] under the same bounded encoding rules
    /// as [`Self::canonical_bytes`].
    pub fn proof_id(&self) -> Result<ContentId, Qwen36SourceProofError> {
        Ok(ContentId::of_bytes(&self.canonical_bytes()?))
    }

    /// Exact canonical typed configuration bytes.
    #[must_use]
    pub fn canonical_config_bytes(&self) -> &[u8] {
        &self.canonical_config
    }

    /// Complete semantic source manifest.
    #[must_use]
    pub const fn manifest(&self) -> &SemanticModelManifest {
        &self.manifest
    }

    /// Complete frozen tensor conversion policy.
    #[must_use]
    pub const fn coverage(&self) -> &Qwen35CoverageManifest {
        &self.coverage
    }

    /// Exact source payload bytes measured by preflight.
    #[must_use]
    pub const fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    /// Exact generic language and deferred-component counts.
    #[must_use]
    pub const fn language(&self) -> Qwen36LanguageCoverage {
        self.language
    }

    /// Official-authentication status retained by the proof.
    #[must_use]
    pub const fn identity_status(&self) -> Qwen36SourceIdentityStatus {
        self.identity_status
    }

    fn validate(&self) -> Result<(), Qwen36SourceProofError> {
        let manifest = self.manifest.canonical_bytes();
        let policy = self.coverage.canonical_policy_bytes();
        validate_component_lengths(&self.canonical_config, &manifest, &policy)?;
        let pinned_config = tritium_nn::qwen36_27b_canonical_source_config()
            .map_err(Qwen36SourceProofError::PinnedConfigEncoding)?;
        if self.canonical_config != pinned_config {
            return Err(Qwen36SourceProofError::PinnedConfigMismatch);
        }
        if self.manifest.architecture() != tritium_nn::QWEN35_HF_SOURCE_ARCHITECTURE {
            return Err(Qwen36SourceProofError::ArchitectureMismatch);
        }
        if !self
            .manifest
            .matches_canonical_config(&self.canonical_config)
        {
            return Err(Qwen36SourceProofError::ConfigDigestMismatch);
        }
        validate_manifest_coverage(&self.manifest, &self.coverage)?;
        validate_numeric(
            "source payload bytes",
            self.coverage.expected_source_payload_bytes(),
            self.payload_bytes,
        )?;
        validate_language(self.language, &self.coverage)
    }
}

/// Scalar receipt for one durable source-admission artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen36AdmissionReceipt {
    proof_id: ContentId,
    manifest_content_id: ContentId,
    source_model_id: ModelId,
    proof_bytes: u64,
    identity_status: Qwen36SourceIdentityStatus,
}

impl Qwen36AdmissionReceipt {
    /// Content identity of exact `ingest.tq36` bytes.
    #[must_use]
    pub const fn proof_id(&self) -> ContentId {
        self.proof_id
    }

    /// SALT content identity of canonical semantic-manifest bytes.
    #[must_use]
    pub const fn manifest_content_id(&self) -> ContentId {
        self.manifest_content_id
    }

    /// Original semantic model identity retained for quantization campaign ledgers.
    #[must_use]
    pub const fn source_model_id(&self) -> ModelId {
        self.source_model_id
    }

    /// Exact canonical proof file length.
    #[must_use]
    pub const fn proof_bytes(&self) -> u64 {
        self.proof_bytes
    }

    /// Candidate-only or future official-authentication status.
    #[must_use]
    pub const fn identity_status(&self) -> Qwen36SourceIdentityStatus {
        self.identity_status
    }
}

/// Same-handle Qwen3.6 source plus a verified durable admission proof.
#[derive(Debug)]
pub struct Qwen36AdmittedSource {
    preflight: Qwen36CampaignPreflight,
    proof: Qwen36SourceProof,
    receipt: Qwen36AdmissionReceipt,
    work_dir: PathBuf,
}

impl Qwen36AdmittedSource {
    /// Run preflight and durably admit a local source candidate in one call.
    ///
    /// Wrong revisions still fail before model or work-root I/O because
    /// [`Qwen36CampaignPreflight::open`] is invoked first.
    ///
    /// # Errors
    /// Returns [`Qwen36AdmissionError`] for preflight, proof, locking, or durable
    /// filesystem failures.
    pub fn open(
        model_dir: &Path,
        declared_revision: &str,
        work_root: &Path,
    ) -> Result<Self, Qwen36AdmissionError> {
        let preflight = Qwen36CampaignPreflight::open(model_dir, declared_revision)
            .map_err(Qwen36AdmissionError::Preflight)?;
        Self::admit(preflight, work_root)
    }

    /// Persist or verify a proof beneath a content-addressed work directory.
    ///
    /// The method consumes the preflight capability so downstream conversion
    /// cannot retain only a scalar receipt and discard the checked source,
    /// coverage policy, or same-handle mutation checks.
    ///
    /// # Errors
    /// Returns [`Qwen36AdmissionError`] for invalid proof construction, a
    /// concurrent admission, filesystem failure, or changed existing artifact.
    pub fn admit(
        preflight: Qwen36CampaignPreflight,
        work_root: &Path,
    ) -> Result<Self, Qwen36AdmissionError> {
        let proof =
            Qwen36SourceProof::from_preflight(&preflight).map_err(Qwen36AdmissionError::Proof)?;
        let bytes = proof
            .canonical_bytes()
            .map_err(Qwen36AdmissionError::Proof)?;
        let proof_id = ContentId::of_bytes(&bytes);
        let manifest_content_id = proof.manifest_content_id();
        let work_root = absolute_path(work_root).map_err(map_directory_error)?;
        let _lock = AdmissionLock::acquire(&work_root, proof_id)?;
        let source_dir = work_root.join(SOURCE_DIRECTORY);
        ensure_directory(&source_dir)?;
        let manifest_dir = source_dir.join(manifest_content_id.to_string());
        ensure_directory(&manifest_dir)?;
        let work_dir = manifest_dir.join(proof_id.to_string());
        ensure_directory(&work_dir)?;
        let proof_path = work_dir.join(PROOF_FILE);
        persist_exact(&proof_path, &bytes)?;
        let proof_bytes = u64::try_from(bytes.len()).map_err(|_| {
            Qwen36AdmissionError::Proof(Qwen36SourceProofError::LengthOverflow("proof bytes"))
        })?;
        let receipt = Qwen36AdmissionReceipt {
            proof_id,
            manifest_content_id,
            source_model_id: proof.source_model_id(),
            proof_bytes,
            identity_status: proof.identity_status(),
        };
        Ok(Self {
            preflight,
            proof,
            receipt,
            work_dir,
        })
    }

    /// Retained same-handle source preflight.
    #[must_use]
    pub const fn preflight(&self) -> &Qwen36CampaignPreflight {
        &self.preflight
    }

    /// Canonical source proof retained in memory.
    #[must_use]
    pub const fn proof(&self) -> &Qwen36SourceProof {
        &self.proof
    }

    /// Scalar durable-admission receipt.
    #[must_use]
    pub const fn receipt(&self) -> &Qwen36AdmissionReceipt {
        &self.receipt
    }

    /// Content-addressed directory containing `ingest.tq36`.
    #[must_use]
    pub fn work_dir(&self) -> &Path {
        &self.work_dir
    }

    /// Durable canonical proof path.
    #[must_use]
    pub fn proof_path(&self) -> PathBuf {
        self.work_dir.join(PROOF_FILE)
    }

    /// Widen one source tensor while retaining admission typestate.
    ///
    /// # Errors
    /// Returns [`Qwen36CampaignPreflightError`] if same-handle content no longer
    /// matches the admitted semantic manifest or widening fails.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, Qwen36CampaignPreflightError> {
        self.preflight.tensor_f32(name)
    }

    /// Stream exact admitted tensor bytes without widening or reopening source paths.
    ///
    /// This retains admission typestate while delegating to the same opened shard
    /// handles whose semantic identity is bound by [`Self::proof`].
    ///
    /// # Errors
    /// Returns [`tritium_nn::Qwen35TensorStreamError::Source`] for changed or
    /// malformed source content and
    /// [`tritium_nn::Qwen35TensorStreamError::Sink`] for a typed callback failure.
    pub fn try_visit_tensor_bytes<E>(
        &self,
        name: &str,
        max_chunk_bytes: usize,
        visit: impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<u64, tritium_nn::Qwen35TensorStreamError<E>> {
        self.preflight
            .try_visit_tensor_bytes(name, max_chunk_bytes, visit)
    }

    /// Construct the architecture-framed hasher used by source admission.
    ///
    /// Downstream stores use this to prove reopened copied payloads still match
    /// the source semantic digest bound by [`Self::proof`].
    ///
    /// # Errors
    /// Returns [`tritium_nn::NnError`] if the tensor is absent or its admitted
    /// metadata cannot initialize the hasher.
    pub fn source_tensor_semantic_hasher(
        &self,
        name: &str,
    ) -> Result<tritium_format::SemanticTensorHasher, tritium_nn::NnError> {
        self.preflight.source_tensor_semantic_hasher(name)
    }
}

/// Why a canonical Qwen3.6 source proof was rejected.
#[derive(Debug)]
#[non_exhaustive]
pub enum Qwen36SourceProofError {
    /// Complete proof exceeded its 64 MiB safety bound.
    ProofTooLarge,
    /// One proof component exceeded its fixed bound.
    ComponentTooLarge(&'static str),
    /// A checked length conversion overflowed.
    LengthOverflow(&'static str),
    /// Record was truncated or structurally malformed.
    Malformed(&'static str),
    /// Record used an unsupported schema version.
    UnsupportedVersion(u8),
    /// Internal checksum did not match the exact record bytes.
    ChecksumMismatch,
    /// Embedded repository was not the campaign pin.
    WrongRepository,
    /// Embedded revision was not the campaign pin.
    WrongRevision,
    /// Embedded identity-status tag is unknown.
    UnsupportedIdentityStatus(u8),
    /// Semantic manifest was invalid or noncanonical.
    Manifest(ArtifactError),
    /// Coverage policy was invalid or no longer equals the frozen policy.
    Coverage(Qwen35CoverageError),
    /// Frozen pinned configuration could not be encoded for comparison.
    PinnedConfigEncoding(tritium_nn::NnError),
    /// Embedded typed configuration was not the exact pinned campaign config.
    PinnedConfigMismatch,
    /// Semantic manifest used a different architecture adapter contract.
    ArchitectureMismatch,
    /// Typed configuration bytes did not match the semantic manifest digest.
    ConfigDigestMismatch,
    /// Manifest tensor count contradicted coverage.
    TensorCountMismatch { expected: u64, actual: u64 },
    /// One manifest entry contradicted coverage.
    TensorMismatch { index: usize, field: &'static str },
    /// Numeric evidence contradicted coverage or schema policy.
    ContractMismatch {
        field: &'static str,
        expected: u64,
        actual: u64,
    },
    /// Decoded proof had an alternate representation.
    NonCanonical,
}

impl fmt::Display for Qwen36SourceProofError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ProofTooLarge => formatter.write_str("Qwen3.6 source proof exceeds 64 MiB"),
            Self::ComponentTooLarge(field) => {
                write!(formatter, "Qwen3.6 source proof {field} exceeds its bound")
            }
            Self::LengthOverflow(field) => {
                write!(formatter, "Qwen3.6 source proof {field} length overflows")
            }
            Self::Malformed(field) => write!(formatter, "malformed Qwen3.6 source proof {field}"),
            Self::UnsupportedVersion(version) => {
                write!(
                    formatter,
                    "unsupported Qwen3.6 source proof version {version}"
                )
            }
            Self::ChecksumMismatch => formatter.write_str("Qwen3.6 source proof checksum mismatch"),
            Self::WrongRepository => {
                formatter.write_str("Qwen3.6 source proof repository is not pinned")
            }
            Self::WrongRevision => {
                formatter.write_str("Qwen3.6 source proof revision is not pinned")
            }
            Self::UnsupportedIdentityStatus(status) => {
                write!(
                    formatter,
                    "unsupported Qwen3.6 source identity status {status}"
                )
            }
            Self::Manifest(error) => write!(formatter, "invalid semantic manifest: {error}"),
            Self::Coverage(error) => write!(formatter, "invalid coverage policy: {error}"),
            Self::PinnedConfigEncoding(error) => {
                write!(formatter, "encode pinned Qwen3.6 configuration: {error}")
            }
            Self::PinnedConfigMismatch => {
                formatter.write_str("Qwen3.6 typed config is not the pinned campaign config")
            }
            Self::ArchitectureMismatch => {
                formatter.write_str("Qwen3.6 semantic manifest architecture is not pinned")
            }
            Self::ConfigDigestMismatch => {
                formatter.write_str("Qwen3.6 typed config does not match semantic manifest")
            }
            Self::TensorCountMismatch { expected, actual } => write!(
                formatter,
                "Qwen3.6 manifest tensor count mismatch: expected {expected}, got {actual}"
            ),
            Self::TensorMismatch { index, field } => {
                write!(
                    formatter,
                    "Qwen3.6 manifest tensor {index} mismatches {field}"
                )
            }
            Self::ContractMismatch {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "Qwen3.6 source proof {field} mismatch: expected {expected}, got {actual}"
            ),
            Self::NonCanonical => {
                formatter.write_str("Qwen3.6 source proof is not canonically encoded")
            }
        }
    }
}

impl std::error::Error for Qwen36SourceProofError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Manifest(error) => Some(error),
            Self::Coverage(error) => Some(error),
            Self::PinnedConfigEncoding(error) => Some(error),
            _ => None,
        }
    }
}

/// Why durable Qwen3.6 source admission failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum Qwen36AdmissionError {
    /// Source preflight failed.
    Preflight(Qwen36CampaignPreflightError),
    /// Canonical proof construction failed.
    Proof(Qwen36SourceProofError),
    /// Work directory or proof filesystem operation failed.
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
    },
    /// Work path exists with a forbidden type or symlink.
    InvalidWorkPath(&'static str),
    /// Another process currently admits the same exact proof.
    AlreadyLocked,
    /// Existing content-addressed proof bytes changed or are corrupt.
    ExistingProofMismatch,
}

impl fmt::Display for Qwen36AdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preflight(error) => write!(formatter, "Qwen3.6 preflight failed: {error}"),
            Self::Proof(error) => write!(formatter, "Qwen3.6 source proof failed: {error}"),
            Self::Io { operation, kind } => {
                write!(formatter, "Qwen3.6 admission {operation} failed: {kind}")
            }
            Self::InvalidWorkPath(field) => {
                write!(formatter, "Qwen3.6 admission has invalid {field}")
            }
            Self::AlreadyLocked => {
                formatter.write_str("Qwen3.6 source proof is already being admitted")
            }
            Self::ExistingProofMismatch => {
                formatter.write_str("existing Qwen3.6 source proof changed or is corrupt")
            }
        }
    }
}

impl std::error::Error for Qwen36AdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Preflight(error) => Some(error),
            Self::Proof(error) => Some(error),
            _ => None,
        }
    }
}

struct ProofCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ProofCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.offset
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], Qwen36SourceProofError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(Qwen36SourceProofError::LengthOverflow("record cursor"))?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(Qwen36SourceProofError::Malformed("truncated record"))?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, Qwen36SourceProofError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, Qwen36SourceProofError> {
        let mut bytes = [0_u8; 4];
        bytes.copy_from_slice(self.take(4)?);
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, Qwen36SourceProofError> {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(self.take(8)?);
        Ok(u64::from_le_bytes(bytes))
    }

    fn length_prefixed(
        &mut self,
        maximum: usize,
        field: &'static str,
    ) -> Result<&'a [u8], Qwen36SourceProofError> {
        let length = self.u32()? as usize;
        if length == 0 || length > maximum {
            return Err(Qwen36SourceProofError::ComponentTooLarge(field));
        }
        self.take(length)
    }

    fn string(
        &mut self,
        maximum: usize,
        field: &'static str,
    ) -> Result<&'a str, Qwen36SourceProofError> {
        core::str::from_utf8(self.length_prefixed(maximum, field)?)
            .map_err(|_| Qwen36SourceProofError::Malformed(field))
    }
}

fn validate_component_lengths(
    config: &[u8],
    manifest: &[u8],
    policy: &[u8],
) -> Result<(), Qwen36SourceProofError> {
    for (field, length, maximum) in [
        ("canonical config", config.len(), MAX_CONFIG_BYTES),
        ("semantic manifest", manifest.len(), MAX_MANIFEST_BYTES),
        ("coverage policy", policy.len(), MAX_POLICY_BYTES),
    ] {
        if length == 0 || length > maximum {
            return Err(Qwen36SourceProofError::ComponentTooLarge(field));
        }
        u32::try_from(length).map_err(|_| Qwen36SourceProofError::LengthOverflow(field))?;
    }
    Ok(())
}

fn write_bytes(output: &mut Vec<u8>, bytes: &[u8]) -> Result<(), Qwen36SourceProofError> {
    let length = u32::try_from(bytes.len())
        .map_err(|_| Qwen36SourceProofError::LengthOverflow("component"))?;
    output.extend_from_slice(&length.to_le_bytes());
    output.extend_from_slice(bytes);
    Ok(())
}

const fn status_tag(status: Qwen36SourceIdentityStatus) -> u8 {
    match status {
        Qwen36SourceIdentityStatus::MeasuredAwaitingOfficialRegistration => 1,
    }
}

fn status_from_tag(tag: u8) -> Result<Qwen36SourceIdentityStatus, Qwen36SourceProofError> {
    match tag {
        1 => Ok(Qwen36SourceIdentityStatus::MeasuredAwaitingOfficialRegistration),
        _ => Err(Qwen36SourceProofError::UnsupportedIdentityStatus(tag)),
    }
}

fn count_u64(value: usize, field: &'static str) -> Result<u64, Qwen36SourceProofError> {
    u64::try_from(value).map_err(|_| Qwen36SourceProofError::LengthOverflow(field))
}

fn validate_manifest_coverage(
    manifest: &SemanticModelManifest,
    coverage: &Qwen35CoverageManifest,
) -> Result<(), Qwen36SourceProofError> {
    if manifest.tensors().len() != coverage.entries().len() {
        return Err(Qwen36SourceProofError::TensorCountMismatch {
            expected: count_u64(coverage.entries().len(), "coverage tensor count")?,
            actual: count_u64(manifest.tensors().len(), "manifest tensor count")?,
        });
    }
    for (index, (actual, expected)) in manifest
        .tensors()
        .iter()
        .zip(coverage.entries())
        .enumerate()
    {
        if actual.name() != expected.name() {
            return Err(Qwen36SourceProofError::TensorMismatch {
                index,
                field: "name",
            });
        }
        if actual.shape() != expected.shape() {
            return Err(Qwen36SourceProofError::TensorMismatch {
                index,
                field: "shape",
            });
        }
    }
    Ok(())
}

fn validate_language(
    actual: Qwen36LanguageCoverage,
    coverage: &Qwen35CoverageManifest,
) -> Result<(), Qwen36SourceProofError> {
    let matrices = count_u64(
        coverage
            .entries()
            .iter()
            .filter(|entry| {
                entry.scope() == Qwen35TensorScope::Language
                    && entry.disposition() == Qwen35CoverageDisposition::AdditiveTernary
            })
            .count(),
        "language matrix count",
    )?;
    let preserved = count_u64(
        coverage
            .entries()
            .iter()
            .filter(|entry| {
                entry.scope() == Qwen35TensorScope::Language
                    && entry.disposition() == Qwen35CoverageDisposition::PreserveSource
            })
            .count(),
        "language preserved count",
    )?;
    let expected = Qwen36LanguageCoverage {
        tensors: coverage.summary().language().tensors(),
        matrices,
        preserved,
        deferred_mtp: coverage.summary().mtp().tensors(),
        deferred_vision: coverage.summary().vision().tensors(),
    };
    for (field, expected, actual) in [
        ("language tensors", expected.tensors, actual.tensors),
        ("language matrices", expected.matrices, actual.matrices),
        ("language preserved", expected.preserved, actual.preserved),
        ("deferred MTP", expected.deferred_mtp, actual.deferred_mtp),
        (
            "deferred vision",
            expected.deferred_vision,
            actual.deferred_vision,
        ),
    ] {
        validate_numeric(field, expected, actual)?;
    }
    Ok(())
}

fn validate_numeric(
    field: &'static str,
    expected: u64,
    actual: u64,
) -> Result<(), Qwen36SourceProofError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Qwen36SourceProofError::ContractMismatch {
            field,
            expected,
            actual,
        })
    }
}

fn ensure_directory(path: &Path) -> Result<(), Qwen36AdmissionError> {
    ensure_durable_directory(path, "work directory").map_err(map_directory_error)
}

fn persist_exact(path: &Path, bytes: &[u8]) -> Result<(), Qwen36AdmissionError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            verify_existing_proof(path, &metadata, bytes)?;
            return sync_directory(
                path.parent()
                    .ok_or(Qwen36AdmissionError::InvalidWorkPath("proof parent"))?,
                "sync existing source proof directory",
            );
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(admission_io("inspect source proof", error)),
    }

    let parent = path
        .parent()
        .ok_or(Qwen36AdmissionError::InvalidWorkPath("proof parent"))?;
    ensure_directory(parent)?;
    let (temporary, mut file) =
        create_temporary_file(parent, &format!("{PROOF_FILE}.tmp")).map_err(map_directory_error)?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(admission_io("write temporary source proof", error));
    }
    drop(file);
    match fs::hard_link(&temporary, path) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let metadata = fs::symlink_metadata(path)
                .map_err(|error| admission_io("inspect concurrent source proof", error))?;
            if let Err(error) = verify_existing_proof(path, &metadata, bytes) {
                let _ = fs::remove_file(&temporary);
                return Err(error);
            }
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(admission_io("publish source proof", error));
        }
    }
    if let Err(error) = sync_directory(parent, "sync source proof directory") {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    fs::remove_file(&temporary)
        .map_err(|error| admission_io("remove temporary source proof", error))?;
    sync_directory(parent, "resync source proof directory")
}

fn verify_existing_proof(
    path: &Path,
    metadata: &fs::Metadata,
    expected: &[u8],
) -> Result<(), Qwen36AdmissionError> {
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(Qwen36AdmissionError::InvalidWorkPath("proof file"));
    }
    if metadata.len() > MAX_PROOF_BYTES as u64 {
        return Err(Qwen36AdmissionError::ExistingProofMismatch);
    }
    let mut file =
        fs::File::open(path).map_err(|error| admission_io("open existing source proof", error))?;
    let opened = file
        .metadata()
        .map_err(|error| admission_io("inspect opened source proof", error))?;
    let after = fs::symlink_metadata(path)
        .map_err(|error| admission_io("reinspect source proof", error))?;
    if after.file_type().is_symlink()
        || !after.is_file()
        || !same_file_identity(metadata, &opened)
        || !same_file_identity(&opened, &after)
        || opened.len() != metadata.len()
        || opened.len() != after.len()
    {
        return Err(Qwen36AdmissionError::InvalidWorkPath("proof file"));
    }
    let opened_len =
        usize::try_from(opened.len()).map_err(|_| Qwen36AdmissionError::ExistingProofMismatch)?;
    let mut existing = Vec::new();
    existing
        .try_reserve_exact(opened_len)
        .map_err(|_| Qwen36AdmissionError::ExistingProofMismatch)?;
    (&mut file)
        .take(MAX_PROOF_BYTES as u64 + 1)
        .read_to_end(&mut existing)
        .map_err(|error| admission_io("read existing source proof", error))?;
    if existing.len() as u64 != opened.len() {
        return Err(Qwen36AdmissionError::ExistingProofMismatch);
    }
    let decoded = Qwen36SourceProof::from_canonical_bytes(&existing)
        .map_err(|_| Qwen36AdmissionError::ExistingProofMismatch)?;
    if existing != expected
        || decoded.proof_id().map_err(Qwen36AdmissionError::Proof)? != ContentId::of_bytes(expected)
    {
        return Err(Qwen36AdmissionError::ExistingProofMismatch);
    }
    Ok(())
}

fn map_directory_error(error: TensorWorkError) -> Qwen36AdmissionError {
    match error {
        TensorWorkError::Io { operation, kind } => Qwen36AdmissionError::Io { operation, kind },
        TensorWorkError::InvalidPath(_) => Qwen36AdmissionError::InvalidWorkPath("work directory"),
        _ => Qwen36AdmissionError::InvalidWorkPath("work directory"),
    }
}

#[cfg(unix)]
fn same_file_identity(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;

    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(not(unix))]
fn same_file_identity(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.is_file() == right.is_file() && left.len() == right.len()
}

#[cfg(unix)]
fn sync_directory(path: &Path, operation: &'static str) -> Result<(), Qwen36AdmissionError> {
    fs::File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| admission_io(operation, error))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path, _operation: &'static str) -> Result<(), Qwen36AdmissionError> {
    Ok(())
}

struct AdmissionLock {
    file: fs::File,
    creator_pid: u32,
}

impl AdmissionLock {
    fn acquire(work_root: &Path, proof_id: ContentId) -> Result<Self, Qwen36AdmissionError> {
        ensure_directory(work_root)?;
        let lock_directory = work_root.join(LOCK_DIRECTORY);
        ensure_directory(&lock_directory)?;
        let lock_path = lock_directory.join(format!("{proof_id}.lock"));
        match fs::symlink_metadata(&lock_path) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                return Err(Qwen36AdmissionError::InvalidWorkPath("proof lock"));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(admission_io("inspect source proof lock", error)),
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock_path)
            .map_err(|error| admission_io("open source proof lock", error))?;
        match file.try_lock() {
            Ok(()) => Ok(Self {
                file,
                creator_pid: std::process::id(),
            }),
            Err(fs::TryLockError::WouldBlock) => Err(Qwen36AdmissionError::AlreadyLocked),
            Err(fs::TryLockError::Error(error)) => Err(admission_io("lock source proof", error)),
        }
    }
}

impl Drop for AdmissionLock {
    fn drop(&mut self) {
        if std::process::id() == self.creator_pid {
            let _ = self.file.unlock();
        }
    }
}

fn admission_io(operation: &'static str, error: io::Error) -> Qwen36AdmissionError {
    Qwen36AdmissionError::Io {
        operation,
        kind: error.kind(),
    }
}

#[cfg(test)]
pub(crate) fn test_fixture_source_proof() -> Qwen36SourceProof {
    use tritium_format::SemanticTensor;
    use tritium_quantize::{QWEN36_27B_COVERAGE_REVISION, Qwen35TensorMetadata};

    const PINNED_METADATA: &str =
        include_str!("../../tritium-quantize/tests/fixtures/qwen36-27b-metadata.tsv");
    let owned: Vec<_> = PINNED_METADATA
        .lines()
        .map(|line| {
            let mut fields = line.split('\t');
            let name = fields.next().unwrap().to_owned();
            let dtype = fields.next().unwrap().to_owned();
            let shape = fields
                .next()
                .unwrap()
                .split(',')
                .map(|value| value.parse::<u64>().unwrap())
                .collect::<Vec<_>>();
            (name, dtype, shape)
        })
        .collect();
    let coverage = Qwen35CoverageManifest::from_metadata(
        QWEN36_27B_COVERAGE_REVISION,
        owned
            .iter()
            .map(|(name, dtype, shape)| Qwen35TensorMetadata::new(name, dtype, shape)),
    )
    .unwrap();
    let canonical_config = tritium_nn::qwen36_27b_canonical_source_config().unwrap();
    let tensors = coverage
        .entries()
        .iter()
        .map(|entry| SemanticTensor::new(entry.name(), entry.shape().to_vec(), b"fixture").unwrap())
        .collect();
    let manifest = SemanticModelManifest::new(
        tritium_nn::QWEN35_HF_SOURCE_ARCHITECTURE,
        &canonical_config,
        tensors,
    )
    .unwrap();
    Qwen36SourceProof {
        canonical_config,
        manifest,
        coverage,
        payload_bytes: 55_562_855_904,
        language: Qwen36LanguageCoverage {
            tensors: 851,
            matrices: 498,
            preserved: 353,
            deferred_mtp: 15,
            deferred_vision: 333,
        },
        identity_status: Qwen36SourceIdentityStatus::MeasuredAwaitingOfficialRegistration,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_proof() -> Qwen36SourceProof {
        test_fixture_source_proof()
    }

    #[test]
    fn source_proof_round_trips_and_keeps_id_domains_distinct() {
        let proof = fixture_proof();
        let bytes = proof.canonical_bytes().unwrap();
        let decoded = Qwen36SourceProof::from_canonical_bytes(&bytes).unwrap();

        assert_eq!(decoded, proof);
        assert_eq!(decoded.canonical_bytes().unwrap(), bytes);
        assert_ne!(
            decoded.manifest_content_id().as_bytes(),
            decoded.source_model_id().as_bytes()
        );
        assert_ne!(decoded.proof_id().unwrap(), decoded.manifest_content_id());
        assert!(!decoded.identity_status().official_payload_authenticated());
    }

    #[test]
    fn source_proof_rejects_corruption_and_contract_drift() {
        let proof = fixture_proof();
        let bytes = proof.canonical_bytes().unwrap();
        assert!(matches!(
            Qwen36SourceProof::from_canonical_bytes(&bytes[..bytes.len() - 1]),
            Err(Qwen36SourceProofError::ChecksumMismatch)
                | Err(Qwen36SourceProofError::Malformed(_))
        ));

        let mut corrupt = bytes.clone();
        corrupt[20] ^= 1;
        assert!(matches!(
            Qwen36SourceProof::from_canonical_bytes(&corrupt),
            Err(Qwen36SourceProofError::ChecksumMismatch)
        ));

        let mut wrong_config = proof.clone();
        wrong_config.canonical_config[0] ^= 1;
        assert!(matches!(
            wrong_config.canonical_bytes(),
            Err(Qwen36SourceProofError::PinnedConfigMismatch)
        ));

        let mut relabeled_config = proof.clone();
        relabeled_config.canonical_config = b"self-consistent but unpinned config".to_vec();
        relabeled_config.manifest = SemanticModelManifest::new(
            tritium_nn::QWEN35_HF_SOURCE_ARCHITECTURE,
            &relabeled_config.canonical_config,
            relabeled_config.manifest.tensors().to_vec(),
        )
        .unwrap();
        let relabeled_bytes = relabeled_config.encode_unchecked().unwrap();
        assert!(matches!(
            Qwen36SourceProof::from_canonical_bytes(&relabeled_bytes),
            Err(Qwen36SourceProofError::PinnedConfigMismatch)
        ));

        let mut wrong_architecture = proof.clone();
        wrong_architecture.manifest = SemanticModelManifest::new(
            "qwen3.6-fixture",
            &wrong_architecture.canonical_config,
            wrong_architecture.manifest.tensors().to_vec(),
        )
        .unwrap();
        assert!(matches!(
            wrong_architecture.canonical_bytes(),
            Err(Qwen36SourceProofError::ArchitectureMismatch)
        ));
        let relabeled_bytes = wrong_architecture.encode_unchecked().unwrap();
        assert!(matches!(
            Qwen36SourceProof::from_canonical_bytes(&relabeled_bytes),
            Err(Qwen36SourceProofError::ArchitectureMismatch)
        ));

        let mut wrong_payload = proof;
        wrong_payload.payload_bytes -= 1;
        assert!(matches!(
            wrong_payload.canonical_bytes(),
            Err(Qwen36SourceProofError::ContractMismatch {
                field: "source payload bytes",
                ..
            })
        ));
    }

    #[test]
    fn durable_proof_is_idempotent_locked_and_tamper_evident() {
        let proof = fixture_proof();
        let bytes = proof.canonical_bytes().unwrap();
        let root = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "tritium-qwen36-admission-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        ensure_directory(&root).unwrap();
        let proof_id = proof.proof_id().unwrap();
        let lock = AdmissionLock::acquire(&root, proof_id).unwrap();
        assert!(matches!(
            AdmissionLock::acquire(&root, proof_id),
            Err(Qwen36AdmissionError::AlreadyLocked)
        ));
        drop(lock);

        let directory = root.join("proof");
        ensure_directory(&directory).unwrap();
        let path = directory.join(PROOF_FILE);
        persist_exact(&path, &bytes).unwrap();
        persist_exact(&path, &bytes).unwrap();
        fs::write(&path, b"tampered").unwrap();
        assert!(matches!(
            persist_exact(&path, &bytes),
            Err(Qwen36AdmissionError::ExistingProofMismatch)
        ));
        assert_eq!(fs::read(&path).unwrap(), b"tampered");
        let _ = fs::remove_dir_all(root);
    }
}
