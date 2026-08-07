//! Strict, seek-backed admission for Plan-0043 Stage-7 token evidence.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    str::FromStr,
};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Frozen sampled-row input schema consumed by the Stage-7 builder.
pub const STAGE7_SAMPLED_ROWS_SCHEMA: &str = "tritium.stage7-sampled-rows.v1";
/// Frozen Stage-7 token-evidence manifest schema.
pub const STAGE7_TOKEN_EVIDENCE_SCHEMA: &str = "tritium.stage7-token-evidence-pack.v1";
/// Frozen token payload encoding.
pub const STAGE7_TOKEN_ENCODING: &str = "u32le";
/// Frozen token payload leaf name.
pub const STAGE7_TOKEN_PAYLOAD_FILE: &str = "stage7.u32le";
const MAX_MANIFEST_BYTES: u64 = 32 * 1024 * 1024;
const READ_CHUNK_BYTES: usize = 64 * 1024;

/// Tokens in every frozen Stage-7 sequence.
pub const STAGE7_TOKENS_PER_SEQUENCE: usize = 2_048;
/// Sequences in every frozen Stage-7 partition.
pub const STAGE7_PARTITION_SEQUENCE_COUNT: usize = 512;
/// Exact bytes in the four-partition u32-le token payload.
pub const STAGE7_TOKEN_PAYLOAD_BYTES: u64 = 4
    * STAGE7_PARTITION_SEQUENCE_COUNT as u64
    * STAGE7_TOKENS_PER_SEQUENCE as u64
    * u32::BITS as u64
    / 8;

const C4_REPOSITORY: &str = "allenai/c4";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Stage7DatasetContract {
    repo_id: &'static str,
    revision: &'static str,
    config: &'static str,
    data_dir: Option<&'static str>,
    split: &'static str,
    text_field: &'static str,
    sequence_count: usize,
}

impl Stage7DatasetContract {
    /// Hugging Face dataset repository.
    #[must_use]
    pub const fn repo_id(self) -> &'static str {
        self.repo_id
    }

    /// Immutable Hub revision.
    #[must_use]
    pub const fn revision(self) -> &'static str {
        self.revision
    }

    /// Hub dataset config.
    #[must_use]
    pub const fn config(self) -> &'static str {
        self.config
    }

    /// Optional Hub data directory, distinct from config.
    #[must_use]
    pub const fn data_dir(self) -> Option<&'static str> {
        self.data_dir
    }

    /// Frozen split.
    #[must_use]
    pub const fn split(self) -> &'static str {
        self.split
    }

    /// Source dataset field from which normalized text was read.
    #[must_use]
    pub const fn text_field(self) -> &'static str {
        self.text_field
    }

    /// Required sequences in each partition lane.
    #[must_use]
    pub const fn sequence_count(self) -> usize {
        self.sequence_count
    }
}

/// Frozen C4/OpenWebMath/StarCoderData composition, in canonical lane order.
pub const STAGE7_DATASETS: [Stage7DatasetContract; 3] = [
    Stage7DatasetContract {
        repo_id: C4_REPOSITORY,
        revision: "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
        config: "en",
        data_dir: None,
        split: "train",
        text_field: "text",
        sequence_count: 256,
    },
    Stage7DatasetContract {
        repo_id: "open-web-math/open-web-math",
        revision: "fde8ef8de2300f5e778f56261843dab89f230815",
        config: "default",
        data_dir: None,
        split: "train",
        text_field: "text",
        sequence_count: 128,
    },
    Stage7DatasetContract {
        repo_id: "bigcode/starcoderdata",
        revision: "9fc30b578cedaec69e47302df72cf00feed7c8c4",
        config: "default",
        data_dir: Some("python"),
        split: "train",
        text_field: "content",
        sequence_count: 128,
    },
];

/// Frozen Plan-0043 evidence partition.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage7Partition {
    /// PTQ fitting evidence.
    Calibration,
    /// Refined-track update evidence.
    Refinement,
    /// Refinement selection evidence.
    Validation,
    /// Final held-out evidence.
    Evaluation,
}

impl Stage7Partition {
    /// Canonical manifest order.
    pub const ALL: [Self; 4] = [
        Self::Calibration,
        Self::Refinement,
        Self::Validation,
        Self::Evaluation,
    ];

    /// Canonical manifest name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Calibration => "calibration",
            Self::Refinement => "refinement",
            Self::Validation => "validation",
            Self::Evaluation => "evaluation",
        }
    }
}

impl fmt::Display for Stage7Partition {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Stage7Partition {
    type Err = Stage7EvidenceError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "calibration" => Ok(Self::Calibration),
            "refinement" => Ok(Self::Refinement),
            "validation" => Ok(Self::Validation),
            "evaluation" => Ok(Self::Evaluation),
            _ => Err(Stage7EvidenceError::Invalid(
                "partition is outside frozen Stage-7 inventory".to_owned(),
            )),
        }
    }
}

/// Strict Stage-7 manifest, payload, or read failure.
#[derive(Debug)]
pub enum Stage7EvidenceError {
    /// Filesystem operation failed.
    Io {
        /// Stable operation label.
        operation: &'static str,
        /// Underlying I/O failure.
        source: io::Error,
    },
    /// Manifest JSON is malformed or outside its schema.
    Json(serde_json::Error),
    /// Manifest or payload violates a frozen Stage-7 invariant.
    Invalid(String),
}

impl fmt::Display for Stage7EvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, source } => write!(formatter, "{operation}: {source}"),
            Self::Json(source) => write!(formatter, "parse Stage-7 evidence manifest: {source}"),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for Stage7EvidenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json(source) => Some(source),
            Self::Invalid(_) => None,
        }
    }
}

impl From<serde_json::Error> for Stage7EvidenceError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FileRecord {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SourceRow {
    row_index: u64,
    text_field: String,
    content_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SequenceRecord {
    id: String,
    dataset_repo_id: String,
    dataset_revision: String,
    dataset_config: String,
    dataset_data_dir: Option<String>,
    dataset_split: String,
    source_rows: Vec<SourceRow>,
    token_offset: u64,
    token_count: u64,
    token_sha256: String,
}

#[derive(Serialize)]
struct SequenceScope<'a> {
    dataset_repo_id: &'a str,
    dataset_revision: &'a str,
    dataset_config: &'a str,
    dataset_data_dir: Option<&'a str>,
    dataset_split: &'a str,
    source_rows: &'a [SourceRow],
    token_offset: u64,
    token_count: u64,
    token_sha256: &'a str,
}

impl SequenceRecord {
    fn scope(&self) -> SequenceScope<'_> {
        SequenceScope {
            dataset_repo_id: &self.dataset_repo_id,
            dataset_revision: &self.dataset_revision,
            dataset_config: &self.dataset_config,
            dataset_data_dir: self.dataset_data_dir.as_deref(),
            dataset_split: &self.dataset_split,
            source_rows: &self.source_rows,
            token_offset: self.token_offset,
            token_count: self.token_count,
            token_sha256: &self.token_sha256,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PartitionRecord {
    sampling_seed: u64,
    sequences: Vec<SequenceRecord>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackManifest {
    schema: String,
    pack_id: String,
    tokenizer_digest: String,
    tokenizer_vocab_size: u64,
    token_encoding: String,
    tokens: FileRecord,
    partitions: BTreeMap<String, PartitionRecord>,
}

#[derive(Serialize)]
struct PackScope<'a> {
    schema: &'a str,
    tokenizer_digest: &'a str,
    tokenizer_vocab_size: u64,
    token_encoding: &'a str,
    tokens: &'a FileRecord,
    partitions: &'a BTreeMap<String, PartitionRecord>,
}

impl PackManifest {
    fn scope(&self) -> PackScope<'_> {
        PackScope {
            schema: &self.schema,
            tokenizer_digest: &self.tokenizer_digest,
            tokenizer_vocab_size: self.tokenizer_vocab_size,
            token_encoding: &self.token_encoding,
            tokens: &self.tokens,
            partitions: &self.partitions,
        }
    }
}

/// Immutable identity established by strict pack admission.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage7TokenEvidenceReceipt {
    pack_id: String,
    tokenizer_digest: String,
    tokenizer_vocab_size: u32,
    token_payload_sha256: String,
}

impl Stage7TokenEvidenceReceipt {
    /// Canonical SHA-256 pack identity.
    #[must_use]
    pub fn pack_id(&self) -> &str {
        &self.pack_id
    }

    /// Canonical tokenizer asset-tree identity.
    #[must_use]
    pub fn tokenizer_digest(&self) -> &str {
        &self.tokenizer_digest
    }

    /// Vocabulary ceiling enforced for every admitted token.
    #[must_use]
    pub const fn tokenizer_vocab_size(&self) -> u32 {
        self.tokenizer_vocab_size
    }

    /// SHA-256 digest of exact `stage7.u32le` bytes.
    #[must_use]
    pub fn token_payload_sha256(&self) -> &str {
        &self.token_payload_sha256
    }
}

/// Bounded sequence window read from a strictly admitted pack.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Stage7TokenBatch {
    partition: Stage7Partition,
    sampling_seed: u64,
    start_sequence: usize,
    sequence_ids: Vec<String>,
    ordered_token_sha256: String,
    tokens: Vec<u32>,
}

impl Stage7TokenBatch {
    /// Source partition.
    #[must_use]
    pub const fn partition(&self) -> Stage7Partition {
        self.partition
    }

    /// Frozen sampling seed carried by the partition manifest.
    #[must_use]
    pub const fn sampling_seed(&self) -> u64 {
        self.sampling_seed
    }

    /// First sequence ordinal in this window.
    #[must_use]
    pub const fn start_sequence(&self) -> usize {
        self.start_sequence
    }

    /// Ordered sequence identities.
    #[must_use]
    pub fn sequence_ids(&self) -> &[String] {
        &self.sequence_ids
    }

    /// SHA-256 identity of exact concatenated u32-le token bytes.
    #[must_use]
    pub fn ordered_token_sha256(&self) -> &str {
        &self.ordered_token_sha256
    }

    /// Row-major `[sequence][token]` values for this bounded window.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Number of complete sequences.
    #[must_use]
    pub fn sequence_count(&self) -> usize {
        self.sequence_ids.len()
    }
}

/// Same-handle reader for a strictly admitted Stage-7 token evidence pack.
#[derive(Debug)]
pub struct Stage7TokenEvidencePack {
    manifest: PackManifest,
    payload: File,
    receipt: Stage7TokenEvidenceReceipt,
}

impl Stage7TokenEvidencePack {
    /// Open and validate all manifest and payload bytes against one model tokenizer.
    ///
    /// Payload validation is bounded to an 8 KiB sequence buffer. The retained
    /// handle prevents later path replacement from redirecting reads.
    ///
    /// # Errors
    /// Returns [`Stage7EvidenceError`] for malformed, unbound, symlinked,
    /// truncated, duplicated, out-of-vocabulary, or content-mutated evidence.
    pub fn open(
        manifest_path: impl AsRef<Path>,
        expected_pack_id: &str,
        expected_tokenizer_digest: &str,
        expected_tokenizer_vocab_size: u32,
    ) -> Result<Self, Stage7EvidenceError> {
        validate_sha256(expected_pack_id, true, "expected token evidence pack id")?;
        validate_sha256(expected_tokenizer_digest, true, "expected tokenizer digest")?;
        if expected_tokenizer_vocab_size == 0 {
            return invalid("expected tokenizer vocabulary is empty");
        }
        let manifest_path = manifest_path.as_ref();
        let manifest_bytes = read_bounded_manifest(manifest_path)?;
        let manifest: PackManifest = serde_json::from_slice(&manifest_bytes)?;
        validate_manifest_identity(
            &manifest,
            expected_pack_id,
            expected_tokenizer_digest,
            expected_tokenizer_vocab_size,
        )?;
        let root = manifest_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let root =
            fs::canonicalize(root).map_err(|source| io_error("canonicalize pack root", source))?;
        let payload_path = root.join(STAGE7_TOKEN_PAYLOAD_FILE);
        let mut payload = open_ordinary(&payload_path, "open token payload")?;
        verify_payload_identity(&mut payload, &manifest.tokens)?;
        validate_payload_semantics(&mut payload, &manifest)?;
        let receipt = Stage7TokenEvidenceReceipt {
            pack_id: manifest.pack_id.clone(),
            tokenizer_digest: manifest.tokenizer_digest.clone(),
            tokenizer_vocab_size: expected_tokenizer_vocab_size,
            token_payload_sha256: manifest.tokens.sha256.clone(),
        };
        Ok(Self {
            manifest,
            payload,
            receipt,
        })
    }

    /// Construction-time content identity.
    #[must_use]
    pub const fn receipt(&self) -> &Stage7TokenEvidenceReceipt {
        &self.receipt
    }

    /// Read one nonempty bounded sequence window and revalidate every selected span.
    ///
    /// # Errors
    /// Returns [`Stage7EvidenceError`] for an empty/out-of-range window, I/O
    /// failure, or same-handle mutation of any selected sequence.
    pub fn read_sequences(
        &mut self,
        partition: Stage7Partition,
        start_sequence: usize,
        sequence_count: usize,
    ) -> Result<Stage7TokenBatch, Stage7EvidenceError> {
        if sequence_count == 0 {
            return invalid("Stage-7 sequence window must be nonempty");
        }
        let end = start_sequence
            .checked_add(sequence_count)
            .filter(|end| *end <= STAGE7_PARTITION_SEQUENCE_COUNT)
            .ok_or_else(|| {
                Stage7EvidenceError::Invalid("Stage-7 sequence window exceeds partition".to_owned())
            })?;
        let record = self
            .manifest
            .partitions
            .get(partition.as_str())
            .expect("strict construction validated partition inventory");
        let selected = &record.sequences[start_sequence..end];
        let token_capacity = sequence_count
            .checked_mul(STAGE7_TOKENS_PER_SEQUENCE)
            .ok_or_else(|| {
                Stage7EvidenceError::Invalid("token window size overflows".to_owned())
            })?;
        let mut tokens = Vec::new();
        tokens.try_reserve_exact(token_capacity).map_err(|_| {
            Stage7EvidenceError::Invalid("allocate Stage-7 token window".to_owned())
        })?;
        let mut sequence_ids = Vec::new();
        sequence_ids
            .try_reserve_exact(sequence_count)
            .map_err(|_| {
                Stage7EvidenceError::Invalid("allocate Stage-7 identity window".to_owned())
            })?;
        let mut ordered = Sha256::new();
        let mut bytes = [0_u8; STAGE7_TOKENS_PER_SEQUENCE * 4];
        for sequence in selected {
            self.payload
                .seek(SeekFrom::Start(sequence.token_offset * 4))
                .map_err(|source| io_error("seek token sequence", source))?;
            self.payload
                .read_exact(&mut bytes)
                .map_err(|source| io_error("read token sequence", source))?;
            if prefixed_sha256(&bytes) != sequence.token_sha256 {
                return invalid("selected token sequence changed after admission");
            }
            ordered.update(bytes);
            for chunk in bytes.chunks_exact(4) {
                let token = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if token >= self.receipt.tokenizer_vocab_size {
                    return invalid("selected token id exceeds tokenizer vocabulary");
                }
                tokens.push(token);
            }
            sequence_ids.push(sequence.id.clone());
        }
        Ok(Stage7TokenBatch {
            partition,
            sampling_seed: record.sampling_seed,
            start_sequence,
            sequence_ids,
            ordered_token_sha256: format!("sha256:{}", hex_digest(&ordered.finalize())),
            tokens,
        })
    }

    /// Recompute exact payload identity through retained handle before publication.
    ///
    /// Later mutation is outside this terminal check's guarantee.
    ///
    /// # Errors
    /// Returns [`Stage7EvidenceError`] when length or content differs from strict
    /// construction, or when terminal I/O fails.
    pub fn verify_unchanged(&mut self) -> Result<(), Stage7EvidenceError> {
        verify_payload_identity(&mut self.payload, &self.manifest.tokens)
    }

    /// Terminally verify evidence and return immutable receipt.
    ///
    /// # Errors
    /// Returns [`Stage7EvidenceError`] under same conditions as
    /// [`Self::verify_unchanged`].
    pub fn finish(mut self) -> Result<Stage7TokenEvidenceReceipt, Stage7EvidenceError> {
        self.verify_unchanged()?;
        Ok(self.receipt)
    }
}

fn validate_manifest_identity(
    manifest: &PackManifest,
    expected_pack_id: &str,
    expected_tokenizer_digest: &str,
    expected_tokenizer_vocab_size: u32,
) -> Result<(), Stage7EvidenceError> {
    if manifest.schema != STAGE7_TOKEN_EVIDENCE_SCHEMA {
        return invalid("token evidence pack schema differs");
    }
    validate_sha256(&manifest.pack_id, true, "token evidence pack id")?;
    let scope = serde_json::to_value(manifest.scope())?;
    if manifest.pack_id != stage7_prefixed_json_sha256(&scope)? {
        return invalid("token evidence pack id differs");
    }
    if manifest.pack_id != expected_pack_id {
        return invalid("token evidence pack differs from expected campaign");
    }
    validate_sha256(&manifest.tokenizer_digest, true, "tokenizer digest")?;
    if manifest.tokenizer_digest != expected_tokenizer_digest
        || manifest.tokenizer_vocab_size != u64::from(expected_tokenizer_vocab_size)
        || manifest.token_encoding != STAGE7_TOKEN_ENCODING
    {
        return invalid("token evidence pack tokenizer identity differs");
    }
    if manifest.tokens.path != STAGE7_TOKEN_PAYLOAD_FILE
        || manifest.tokens.bytes != STAGE7_TOKEN_PAYLOAD_BYTES
    {
        return invalid("token evidence payload byte geometry differs");
    }
    validate_sha256(&manifest.tokens.sha256, false, "token payload digest")?;
    let expected_names = Stage7Partition::ALL
        .into_iter()
        .map(|partition| partition.as_str().to_owned())
        .collect::<BTreeSet<_>>();
    if manifest.partitions.keys().cloned().collect::<BTreeSet<_>>() != expected_names {
        return invalid("token evidence partition inventory differs");
    }
    Ok(())
}

fn validate_payload_semantics(
    payload: &mut File,
    manifest: &PackManifest,
) -> Result<(), Stage7EvidenceError> {
    let mut expected_offset = 0_u64;
    let mut source_rows = BTreeSet::new();
    let mut source_content = BTreeSet::new();
    let mut sequence_ids = BTreeSet::new();
    let mut token_digests = BTreeSet::new();
    let mut bytes = [0_u8; STAGE7_TOKENS_PER_SEQUENCE * 4];
    payload
        .seek(SeekFrom::Start(0))
        .map_err(|source| io_error("seek token payload", source))?;
    for partition in Stage7Partition::ALL {
        let record = manifest
            .partitions
            .get(partition.as_str())
            .expect("manifest inventory validated");
        if record.sequences.len() != STAGE7_PARTITION_SEQUENCE_COUNT {
            return invalid(format!(
                "token evidence {} sequence inventory differs",
                partition.as_str()
            ));
        }
        let mut dataset_counts = BTreeMap::new();
        for (ordinal, sequence) in record.sequences.iter().enumerate() {
            let dataset = STAGE7_DATASETS
                .iter()
                .find(|dataset| dataset.repo_id == sequence.dataset_repo_id)
                .ok_or_else(|| {
                    Stage7EvidenceError::Invalid(
                        "token evidence dataset is outside frozen composition".to_owned(),
                    )
                })?;
            if sequence.dataset_revision != dataset.revision
                || sequence.dataset_config != dataset.config
                || sequence.dataset_data_dir.as_deref() != dataset.data_dir
                || sequence.dataset_split != dataset.split
            {
                return invalid("token evidence dataset provenance differs");
            }
            *dataset_counts.entry(dataset.repo_id).or_insert(0_usize) += 1;
            if sequence.source_rows.is_empty() {
                return invalid("token evidence source rows must be nonempty");
            }
            for row in &sequence.source_rows {
                if row.text_field != dataset.text_field {
                    return invalid("token evidence source-row text field differs");
                }
                validate_sha256(&row.content_sha256, false, "source content digest")?;
                let locator = (
                    sequence.dataset_repo_id.clone(),
                    sequence.dataset_revision.clone(),
                    sequence.dataset_config.clone(),
                    sequence.dataset_data_dir.clone(),
                    sequence.dataset_split.clone(),
                    row.row_index,
                    row.text_field.clone(),
                );
                if !source_rows.insert(locator) {
                    return invalid("token evidence reuses a source row");
                }
                if !source_content.insert(row.content_sha256.clone()) {
                    return invalid("token evidence reuses source content");
                }
            }
            if sequence.token_offset != expected_offset
                || sequence.token_count != STAGE7_TOKENS_PER_SEQUENCE as u64
            {
                return invalid("token evidence sequence span differs");
            }
            validate_sha256(&sequence.token_sha256, true, "token sequence digest")?;
            payload
                .read_exact(&mut bytes)
                .map_err(|source| io_error("read token payload sequence", source))?;
            let observed = prefixed_sha256(&bytes);
            if observed != sequence.token_sha256 {
                return invalid("token sequence payload digest differs");
            }
            if !token_digests.insert(observed) {
                return invalid("token evidence contains duplicate token sequences");
            }
            for chunk in bytes.chunks_exact(4) {
                let token = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                if u64::from(token) >= manifest.tokenizer_vocab_size {
                    return invalid("token id exceeds tokenizer vocabulary");
                }
            }
            validate_sha256(&sequence.id, true, "token sequence id")?;
            let expected_id =
                stage7_prefixed_json_sha256(&serde_json::to_value(sequence.scope())?)?;
            if sequence.id != expected_id {
                return invalid("token evidence sequence id differs");
            }
            if !sequence_ids.insert(sequence.id.clone()) {
                return invalid("token evidence sequence ids are not disjoint");
            }
            if partition == Stage7Partition::Calibration
                && ordinal < 128
                && sequence.dataset_repo_id != C4_REPOSITORY
            {
                return invalid("smoke token prefix is not frozen C4 prefix");
            }
            expected_offset += STAGE7_TOKENS_PER_SEQUENCE as u64;
        }
        for dataset in STAGE7_DATASETS {
            if dataset_counts.get(dataset.repo_id).copied() != Some(dataset.sequence_count) {
                return invalid(format!(
                    "token evidence {} dataset composition differs",
                    partition.as_str()
                ));
            }
        }
    }
    if expected_offset * 4 != STAGE7_TOKEN_PAYLOAD_BYTES {
        return invalid("token evidence payload has trailing or missing bytes");
    }
    Ok(())
}

fn verify_payload_identity(
    payload: &mut File,
    record: &FileRecord,
) -> Result<(), Stage7EvidenceError> {
    let metadata = payload
        .metadata()
        .map_err(|source| io_error("inspect token payload", source))?;
    if !metadata.is_file() || metadata.len() != record.bytes {
        return invalid("token payload byte geometry differs from opened file");
    }
    payload
        .seek(SeekFrom::Start(0))
        .map_err(|source| io_error("seek token payload", source))?;
    let mut hasher = Sha256::new();
    let mut remaining = record.bytes;
    let mut buffer = [0_u8; READ_CHUNK_BYTES];
    while remaining != 0 {
        let wanted = usize::try_from(remaining.min(READ_CHUNK_BYTES as u64))
            .expect("bounded read size fits usize");
        payload
            .read_exact(&mut buffer[..wanted])
            .map_err(|source| io_error("read token payload", source))?;
        hasher.update(&buffer[..wanted]);
        remaining -= wanted as u64;
    }
    let mut trailing = [0_u8; 1];
    if payload
        .read(&mut trailing)
        .map_err(|source| io_error("check token payload length", source))?
        != 0
    {
        return invalid("token payload byte geometry differs from opened file");
    }
    let terminal = payload
        .metadata()
        .map_err(|source| io_error("reinspect token payload", source))?;
    if terminal.len() != record.bytes || hex_digest(&hasher.finalize()) != record.sha256 {
        return invalid("token payload identity differs");
    }
    Ok(())
}

fn read_bounded_manifest(path: &Path) -> Result<Vec<u8>, Stage7EvidenceError> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent)
        .map_err(|source| io_error("canonicalize manifest parent", source))?;
    let name = path.file_name().ok_or_else(|| {
        Stage7EvidenceError::Invalid("Stage-7 manifest path has no file name".to_owned())
    })?;
    let canonical_path: PathBuf = parent.join(name);
    let file = open_ordinary(&canonical_path, "open Stage-7 manifest")?;
    let metadata = file
        .metadata()
        .map_err(|source| io_error("inspect Stage-7 manifest", source))?;
    if metadata.len() == 0 || metadata.len() > MAX_MANIFEST_BYTES {
        return invalid("Stage-7 manifest must be a bounded nonempty file");
    }
    let capacity = usize::try_from(metadata.len()).map_err(|_| {
        Stage7EvidenceError::Invalid("manifest size does not fit memory".to_owned())
    })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| Stage7EvidenceError::Invalid("allocate Stage-7 manifest".to_owned()))?;
    file.take(MAX_MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|source| io_error("read Stage-7 manifest", source))?;
    if bytes.len() as u64 != metadata.len() {
        return invalid("Stage-7 manifest changed while reading");
    }
    Ok(bytes)
}

fn open_ordinary(path: &Path, operation: &'static str) -> Result<File, Stage7EvidenceError> {
    let metadata = fs::symlink_metadata(path).map_err(|source| io_error(operation, source))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return invalid(format!("{} must be an ordinary file", path.display()));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let file = options
        .open(path)
        .map_err(|source| io_error(operation, source))?;
    if !file
        .metadata()
        .map_err(|source| io_error(operation, source))?
        .is_file()
    {
        return invalid(format!("{} must be an ordinary file", path.display()));
    }
    Ok(file)
}

/// SHA-256 identity of Stage-7 canonical JSON, including the `sha256:` prefix.
///
/// Object keys are sorted recursively and no insignificant whitespace is
/// emitted. Rust builders and readers share this implementation; the Python
/// qualifier remains an independent cross-language oracle.
///
/// # Errors
/// Returns [`Stage7EvidenceError`] when a JSON string cannot be serialized.
pub fn stage7_prefixed_json_sha256(value: &Value) -> Result<String, Stage7EvidenceError> {
    Ok(prefixed_sha256(&canonical(value)?))
}

fn prefixed_sha256(bytes: &[u8]) -> String {
    format!("sha256:{}", hex_digest(&Sha256::digest(bytes)))
}

fn canonical(value: &Value) -> Result<Vec<u8>, Stage7EvidenceError> {
    fn write(value: &Value, output: &mut Vec<u8>) -> Result<(), Stage7EvidenceError> {
        match value {
            Value::Null => output.extend_from_slice(b"null"),
            Value::Bool(value) => {
                output.extend_from_slice(if *value { b"true" } else { b"false" });
            }
            Value::Number(value) => output.extend_from_slice(value.to_string().as_bytes()),
            Value::String(value) => serde_json::to_writer(output, value)?,
            Value::Array(values) => {
                output.push(b'[');
                for (ordinal, value) in values.iter().enumerate() {
                    if ordinal != 0 {
                        output.push(b',');
                    }
                    write(value, output)?;
                }
                output.push(b']');
            }
            Value::Object(values) => {
                output.push(b'{');
                let mut fields = values.iter().collect::<Vec<_>>();
                fields.sort_by_key(|(name, _)| *name);
                for (ordinal, (name, value)) in fields.into_iter().enumerate() {
                    if ordinal != 0 {
                        output.push(b',');
                    }
                    serde_json::to_writer(&mut *output, name)?;
                    output.push(b':');
                    write(value, output)?;
                }
                output.push(b'}');
            }
        }
        Ok(())
    }
    let mut output = Vec::new();
    write(value, &mut output)?;
    Ok(output)
}

fn validate_sha256(value: &str, prefixed: bool, label: &str) -> Result<(), Stage7EvidenceError> {
    let digest = if prefixed {
        value.strip_prefix("sha256:").ok_or_else(|| {
            Stage7EvidenceError::Invalid(format!("{label} is not canonical SHA-256"))
        })?
    } else {
        value
    };
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return invalid(format!("{label} is not canonical SHA-256"));
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn invalid<T>(message: impl Into<String>) -> Result<T, Stage7EvidenceError> {
    Err(Stage7EvidenceError::Invalid(message.into()))
}

fn io_error(operation: &'static str, source: io::Error) -> Stage7EvidenceError {
    Stage7EvidenceError::Io { operation, source }
}
