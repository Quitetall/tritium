//! Deterministic Stage-7 token-evidence pack builder.

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, Write},
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tritium_nn::{HfJsonTokenizer, Tokenizer};
use tritium_salt::{
    STAGE7_DATASETS, STAGE7_PARTITION_SEQUENCE_COUNT, STAGE7_SAMPLED_ROWS_SCHEMA,
    STAGE7_TOKEN_ENCODING, STAGE7_TOKEN_EVIDENCE_SCHEMA, STAGE7_TOKEN_PAYLOAD_FILE,
    STAGE7_TOKENS_PER_SEQUENCE, Stage7DatasetContract, Stage7Partition,
    stage7_prefixed_json_sha256,
};

const MAX_JSON_BYTES: u64 = 32 * 1024 * 1024;
const MAX_ROWS_BYTES: u64 = 512 * 1024 * 1024;
const MAX_ROW_BYTES: usize = 16 * 1024 * 1024;
const TOKENIZER_FILES: [&str; 5] = [
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
];

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FileRecord {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceManifest {
    schema: String,
    partitions: BTreeMap<String, SourcePartition>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourcePartition {
    sampling_seed: u64,
    datasets: Vec<SourceDataset>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceDataset {
    repo_id: String,
    revision: String,
    config: String,
    data_dir: Option<String>,
    split: String,
    text_field: String,
    rows: FileRecord,
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SourceLocator {
    repo_id: String,
    revision: String,
    config: String,
    data_dir: Option<String>,
    split: String,
    row_index: u64,
    text_field: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SampledRow {
    row_index: u64,
    content_sha256: String,
    text: String,
}

#[derive(Clone, Debug, Serialize)]
struct SourceRowReceipt {
    row_index: u64,
    text_field: String,
    content_sha256: String,
}

#[derive(Debug)]
struct SequenceDraft {
    dataset_repo_id: String,
    dataset_revision: String,
    dataset_config: String,
    dataset_data_dir: Option<String>,
    dataset_split: String,
    source_rows: Vec<SourceRowReceipt>,
    tokens: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
struct SequenceScope {
    dataset_repo_id: String,
    dataset_revision: String,
    dataset_config: String,
    dataset_data_dir: Option<String>,
    dataset_split: String,
    source_rows: Vec<SourceRowReceipt>,
    token_offset: u64,
    token_count: u64,
    token_sha256: String,
}

#[derive(Debug, Serialize)]
struct SequenceReceipt {
    id: String,
    #[serde(flatten)]
    scope: SequenceScope,
}

#[derive(Debug, Serialize)]
struct PartitionReceipt {
    sampling_seed: u64,
    sequences: Vec<SequenceReceipt>,
}

#[derive(Debug, Serialize)]
struct PackScope {
    schema: &'static str,
    tokenizer_digest: String,
    tokenizer_vocab_size: u64,
    token_encoding: &'static str,
    tokens: FileRecord,
    partitions: BTreeMap<String, PartitionReceipt>,
}

#[derive(Debug, Serialize)]
struct PackManifest {
    pack_id: String,
    #[serde(flatten)]
    scope: PackScope,
}

pub(crate) fn build(model: &Path, sampled: &Path, output: &Path) -> anyhow::Result<()> {
    validate_output(model, sampled, output)?;
    let (tokenizer_digest, vocab_size) = model_tokenizer_identity(model)?;
    let tokenizer_path = open_model_file(model, "tokenizer.json")?;
    let tokenizer_config_path = open_model_file(model, "tokenizer_config.json")?;
    let tokenizer = HfJsonTokenizer::from_files(&tokenizer_path, &tokenizer_config_path)?;
    ensure!(
        tokenizer.eos() < vocab_size,
        "EOS token exceeds model vocabulary"
    );
    let source: SourceManifest = read_json(sampled, MAX_JSON_BYTES, "sampled-row manifest")?;
    ensure!(
        source.schema == STAGE7_SAMPLED_ROWS_SCHEMA,
        "sampled-row manifest schema differs"
    );
    ensure!(
        source.partitions.len() == Stage7Partition::ALL.len(),
        "sampled-row partition inventory differs"
    );
    let source_root = sampled.parent().unwrap_or_else(|| Path::new("."));
    let mut locators = BTreeSet::new();
    let mut content_digests = BTreeSet::new();
    let mut drafts = BTreeMap::new();
    let mut seeds = BTreeMap::new();
    for partition_kind in Stage7Partition::ALL {
        let partition_name = partition_kind.as_str();
        let partition = source
            .partitions
            .get(partition_name)
            .with_context(|| format!("sampled-row partition {partition_name} is missing"))?;
        ensure!(
            partition.datasets.len() == STAGE7_DATASETS.len(),
            "{partition_name} dataset inventory differs"
        );
        let mut partition_drafts = Vec::with_capacity(STAGE7_PARTITION_SEQUENCE_COUNT);
        for (ordinal, expected) in STAGE7_DATASETS.into_iter().enumerate() {
            let dataset = &partition.datasets[ordinal];
            validate_dataset(dataset, expected, partition_name)?;
            let rows = open_record(source_root, &dataset.rows, MAX_ROWS_BYTES, "sampled rows")?;
            partition_drafts.extend(build_lane(
                rows,
                dataset,
                expected.sequence_count(),
                &tokenizer,
                u64::from(vocab_size),
                &mut locators,
                &mut content_digests,
            )?);
        }
        ensure!(
            partition_drafts.len() == STAGE7_PARTITION_SEQUENCE_COUNT,
            "{partition_name} sequence inventory differs"
        );
        seeds.insert(partition_name.to_owned(), partition.sampling_seed);
        drafts.insert(partition_name.to_owned(), partition_drafts);
    }
    let staging = create_staging(output)?;
    let result = write_pack(
        &staging,
        drafts,
        seeds,
        tokenizer_digest,
        u64::from(vocab_size),
    );
    let manifest = match result {
        Ok(manifest) => manifest,
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
    };
    if let Err(error) = publish_directory(&staging, output) {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    let mut rendered = serde_json::to_vec_pretty(&manifest)?;
    rendered.push(b'\n');
    print!(
        "{}",
        String::from_utf8(rendered).expect("manifest JSON is UTF-8")
    );
    Ok(())
}

pub(crate) fn model_tokenizer_identity(model: &Path) -> anyhow::Result<(String, u32)> {
    let metadata = fs::symlink_metadata(model)
        .with_context(|| format!("inspect model directory {}", model.display()))?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "model path must be an ordinary directory"
    );
    let config_path = open_model_file(model, "config.json")?;
    let config: Value = read_json(&config_path, MAX_JSON_BYTES, "model config")?;
    let vocab_size = config
        .get("vocab_size")
        .and_then(Value::as_u64)
        .filter(|value| *value > 0 && *value <= u64::from(u32::MAX))
        .context("model config requires u32-representable positive vocab_size")?;
    Ok((tokenizer_digest(model)?, vocab_size as u32))
}

fn validate_output(model: &Path, sampled: &Path, output: &Path) -> anyhow::Result<()> {
    for (path, label) in [
        (model, "model directory"),
        (sampled, "sampled-row manifest"),
    ] {
        let metadata = fs::symlink_metadata(path)
            .with_context(|| format!("inspect {label} {}", path.display()))?;
        ensure!(
            !metadata.file_type().is_symlink(),
            "{label} must not be a symlink"
        );
    }
    ensure!(
        fs::metadata(model)?.is_dir(),
        "model path is not a directory"
    );
    ensure!(
        fs::metadata(sampled)?.is_file(),
        "sampled-row manifest is not a file"
    );
    match fs::symlink_metadata(output) {
        Ok(_) => anyhow::bail!("Stage-7 evidence output already exists"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("inspect Stage-7 evidence output"),
    }
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent).context("create Stage-7 evidence parent")?;
    let metadata = fs::symlink_metadata(parent).context("inspect Stage-7 evidence parent")?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "Stage-7 evidence parent must be an ordinary directory"
    );
    Ok(())
}

fn validate_dataset(
    dataset: &SourceDataset,
    expected: Stage7DatasetContract,
    partition: &str,
) -> anyhow::Result<()> {
    ensure!(
        dataset.repo_id == expected.repo_id()
            && dataset.revision == expected.revision()
            && dataset.config == expected.config()
            && dataset.data_dir.as_deref() == expected.data_dir()
            && dataset.split == expected.split()
            && dataset.text_field == expected.text_field(),
        "{partition} dataset {} provenance differs",
        expected.repo_id(),
    );
    Ok(())
}

fn build_lane(
    rows_file: File,
    dataset: &SourceDataset,
    wanted: usize,
    tokenizer: &HfJsonTokenizer,
    vocab_size: u64,
    locators: &mut BTreeSet<SourceLocator>,
    content_digests: &mut BTreeSet<String>,
) -> anyhow::Result<Vec<SequenceDraft>> {
    let mut reader = BufReader::new(rows_file);
    let mut result = Vec::with_capacity(wanted);
    let mut tokens = Vec::with_capacity(STAGE7_TOKENS_PER_SEQUENCE);
    let mut rows = Vec::new();
    let mut line_ordinal = 0_usize;
    loop {
        let mut bytes = Vec::new();
        let read = Read::by_ref(&mut reader)
            .take(MAX_ROW_BYTES as u64 + 2)
            .read_until(b'\n', &mut bytes)
            .context("read sampled row")?;
        if read == 0 {
            break;
        }
        line_ordinal += 1;
        ensure!(
            bytes.len() <= MAX_ROW_BYTES + 1,
            "sampled row {line_ordinal} has invalid size"
        );
        if bytes.last() == Some(&b'\n') {
            bytes.pop();
        }
        if bytes.last() == Some(&b'\r') {
            bytes.pop();
        }
        let line = std::str::from_utf8(&bytes).context("sampled row is not UTF-8")?;
        ensure!(
            !line.is_empty() && line.len() <= MAX_ROW_BYTES,
            "sampled row {} has invalid size",
            line_ordinal
        );
        let row: SampledRow = serde_json::from_str(line)
            .with_context(|| format!("parse sampled row {line_ordinal}"))?;
        ensure!(
            valid_hex(&row.content_sha256),
            "sampled row content digest is malformed"
        );
        ensure!(
            hex_digest(&Sha256::digest(row.text.as_bytes())) == row.content_sha256,
            "sampled row content digest differs"
        );
        ensure!(
            content_digests.insert(row.content_sha256.clone()),
            "sampled rows reuse source content"
        );
        let locator = SourceLocator {
            repo_id: dataset.repo_id.clone(),
            revision: dataset.revision.clone(),
            config: dataset.config.clone(),
            data_dir: dataset.data_dir.clone(),
            split: dataset.split.clone(),
            row_index: row.row_index,
            text_field: dataset.text_field.clone(),
        };
        ensure!(locators.insert(locator), "sampled-row locator is reused");
        let encoded = tokenizer.encode(&row.text)?;
        ensure!(
            !encoded.is_empty(),
            "sampled row tokenizes to an empty sequence"
        );
        ensure!(
            encoded.iter().all(|token| u64::from(*token) < vocab_size),
            "sampled row token exceeds model vocabulary"
        );
        tokens.extend(encoded);
        rows.push(SourceRowReceipt {
            row_index: row.row_index,
            text_field: dataset.text_field.clone(),
            content_sha256: row.content_sha256,
        });
        if tokens.len() < STAGE7_TOKENS_PER_SEQUENCE {
            tokens.push(tokenizer.eos());
        }
        if tokens.len() >= STAGE7_TOKENS_PER_SEQUENCE {
            tokens.truncate(STAGE7_TOKENS_PER_SEQUENCE);
            result.push(SequenceDraft {
                dataset_repo_id: dataset.repo_id.clone(),
                dataset_revision: dataset.revision.clone(),
                dataset_config: dataset.config.clone(),
                dataset_data_dir: dataset.data_dir.clone(),
                dataset_split: dataset.split.clone(),
                source_rows: std::mem::take(&mut rows),
                tokens: std::mem::take(&mut tokens),
            });
            tokens = Vec::with_capacity(STAGE7_TOKENS_PER_SEQUENCE);
            if result.len() == wanted {
                break;
            }
        }
    }
    ensure!(
        result.len() == wanted,
        "sampled rows cannot produce {wanted} complete sequences"
    );
    Ok(result)
}

fn write_pack(
    root: &Path,
    mut drafts: BTreeMap<String, Vec<SequenceDraft>>,
    seeds: BTreeMap<String, u64>,
    tokenizer_digest: String,
    vocab_size: u64,
) -> anyhow::Result<PackManifest> {
    let token_path = root.join(STAGE7_TOKEN_PAYLOAD_FILE);
    let mut token_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&token_path)?;
    let mut token_hasher = Sha256::new();
    let mut offset = 0_u64;
    let mut partitions = BTreeMap::new();
    let mut token_digests = BTreeSet::new();
    for partition_kind in Stage7Partition::ALL {
        let partition_name = partition_kind.as_str();
        let mut receipts = Vec::with_capacity(STAGE7_PARTITION_SEQUENCE_COUNT);
        for draft in drafts
            .remove(partition_name)
            .context("missing built partition")?
        {
            let mut bytes = Vec::with_capacity(STAGE7_TOKENS_PER_SEQUENCE * 4);
            for token in &draft.tokens {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
            let token_sha256 = format!("sha256:{}", hex_digest(&Sha256::digest(&bytes)));
            ensure!(
                token_digests.insert(token_sha256.clone()),
                "token evidence contains duplicate sequences"
            );
            token_file.write_all(&bytes)?;
            token_hasher.update(&bytes);
            let scope = SequenceScope {
                dataset_repo_id: draft.dataset_repo_id,
                dataset_revision: draft.dataset_revision,
                dataset_config: draft.dataset_config,
                dataset_data_dir: draft.dataset_data_dir,
                dataset_split: draft.dataset_split,
                source_rows: draft.source_rows,
                token_offset: offset,
                token_count: STAGE7_TOKENS_PER_SEQUENCE as u64,
                token_sha256,
            };
            let id = stage7_prefixed_json_sha256(&serde_json::to_value(&scope)?)?;
            receipts.push(SequenceReceipt { id, scope });
            offset += STAGE7_TOKENS_PER_SEQUENCE as u64;
        }
        partitions.insert(
            partition_name.to_owned(),
            PartitionReceipt {
                sampling_seed: seeds[partition_name],
                sequences: receipts,
            },
        );
    }
    token_file.sync_all()?;
    let tokens = FileRecord {
        path: STAGE7_TOKEN_PAYLOAD_FILE.to_owned(),
        bytes: offset * 4,
        sha256: hex_digest(&token_hasher.finalize()),
    };
    let scope = PackScope {
        schema: STAGE7_TOKEN_EVIDENCE_SCHEMA,
        tokenizer_digest,
        tokenizer_vocab_size: vocab_size,
        token_encoding: STAGE7_TOKEN_ENCODING,
        tokens,
        partitions,
    };
    let pack_id = stage7_prefixed_json_sha256(&serde_json::to_value(&scope)?)?;
    let manifest = PackManifest { pack_id, scope };
    let mut bytes = serde_json::to_vec_pretty(&manifest)?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join("manifest.json"))?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(manifest)
}

fn tokenizer_digest(model: &Path) -> anyhow::Result<String> {
    let mut records = Vec::new();
    for name in TOKENIZER_FILES {
        records.push(file_record(&open_model_file(model, name)?, name)?);
    }
    Ok(stage7_prefixed_json_sha256(&serde_json::to_value(
        records,
    )?)?)
}

fn open_model_file(model: &Path, name: &str) -> anyhow::Result<PathBuf> {
    let model_root = fs::canonicalize(model).context("canonicalize model root")?;
    let candidate = model.join(name);
    let metadata =
        fs::symlink_metadata(&candidate).with_context(|| format!("inspect model asset {name}"))?;
    if metadata.is_file() && !metadata.file_type().is_symlink() {
        let resolved = fs::canonicalize(&candidate)?;
        ensure!(
            resolved.starts_with(&model_root),
            "model asset {name} escapes model root"
        );
        return Ok(resolved);
    }
    ensure!(
        metadata.file_type().is_symlink(),
        "model asset {name} must be an ordinary file or Hub blob link"
    );
    ensure!(
        model.parent().and_then(Path::file_name) == Some(std::ffi::OsStr::new("snapshots")),
        "model asset {name} is an untrusted symlink"
    );
    let blob_root = model
        .parent()
        .and_then(Path::parent)
        .context("Hub snapshot has no repository root")?
        .join("blobs");
    let blob_root = fs::canonicalize(&blob_root).context("canonicalize Hub blob root")?;
    let resolved = fs::canonicalize(&candidate).context("resolve Hub model asset")?;
    let resolved_metadata = fs::metadata(&resolved)?;
    ensure!(
        resolved_metadata.is_file() && resolved.starts_with(blob_root),
        "model asset {name} escapes Hub blob root"
    );
    Ok(resolved)
}

fn file_record(path: &Path, logical: &str) -> anyhow::Result<FileRecord> {
    let metadata =
        fs::symlink_metadata(path).with_context(|| format!("inspect {}", path.display()))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "{} must be an ordinary file",
        path.display()
    );
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let bytes = std::io::copy(&mut file, &mut hasher)?;
    Ok(FileRecord {
        path: logical.to_owned(),
        bytes,
        sha256: hex_digest(&hasher.finalize()),
    })
}

fn open_record(
    root: &Path,
    record: &FileRecord,
    max_bytes: u64,
    label: &str,
) -> anyhow::Result<File> {
    ensure!(valid_hex(&record.sha256), "{label} digest is malformed");
    ensure!(
        record.bytes > 0 && record.bytes <= max_bytes,
        "{label} size is outside bounds"
    );
    let logical = Path::new(&record.path);
    let components: Vec<_> = logical.components().collect();
    ensure!(
        !components.is_empty()
            && !logical.is_absolute()
            && components
                .iter()
                .all(|part| matches!(part, Component::Normal(_))),
        "{label} path is not contained"
    );
    let root = fs::canonicalize(root).context("canonicalize sampled-row root")?;
    let mut cursor = root.clone();
    for component in &components[..components.len() - 1] {
        cursor.push(component.as_os_str());
        let metadata =
            fs::symlink_metadata(&cursor).with_context(|| format!("inspect {label} parent"))?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "{label} path traverses a symlink"
        );
    }
    let candidate = root.join(logical);
    let metadata = fs::symlink_metadata(&candidate).with_context(|| format!("inspect {label}"))?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "{label} must be an ordinary file"
    );
    let resolved = fs::canonicalize(&candidate)?;
    ensure!(
        resolved.starts_with(&root),
        "{label} escapes sampled-row root"
    );
    let mut file = File::open(&resolved).with_context(|| format!("open {label}"))?;
    let opened = file
        .metadata()
        .with_context(|| format!("inspect opened {label}"))?;
    ensure!(
        opened.is_file() && opened.len() == record.bytes,
        "{label} size differs"
    );
    let mut hasher = Sha256::new();
    let bytes = std::io::copy(&mut file, &mut hasher)?;
    ensure!(
        bytes == record.bytes && hex_digest(&hasher.finalize()) == record.sha256,
        "{label} identity differs"
    );
    file.rewind()?;
    Ok(file)
}

fn read_json<T: serde::de::DeserializeOwned>(
    path: &Path,
    max: u64,
    label: &str,
) -> anyhow::Result<T> {
    let metadata = fs::symlink_metadata(path).with_context(|| format!("inspect {label}"))?;
    ensure!(
        metadata.is_file()
            && !metadata.file_type().is_symlink()
            && metadata.len() > 0
            && metadata.len() <= max,
        "{label} must be a bounded ordinary file"
    );
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)?.take(max + 1).read_to_end(&mut bytes)?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {label}"))
}

fn create_staging(output: &Path) -> anyhow::Result<PathBuf> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = output
        .file_name()
        .context("Stage-7 evidence output has no name")?;
    let mut temporary = OsString::from(name);
    temporary.push(format!(
        ".tmp-{}-{}",
        std::process::id(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let staging = parent.join(temporary);
    fs::create_dir(&staging).context("create Stage-7 evidence staging directory")?;
    Ok(staging)
}

fn sync_parent(path: &Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    File::open(path.parent().unwrap_or(Path::new(".")))?.sync_all()?;
    Ok(())
}

fn publish_directory(staging: &Path, output: &Path) -> anyhow::Result<()> {
    fs::create_dir(output).context("reserve Stage-7 evidence output directory")?;
    let result = (|| {
        fs::hard_link(
            staging.join(STAGE7_TOKEN_PAYLOAD_FILE),
            output.join(STAGE7_TOKEN_PAYLOAD_FILE),
        )
        .context("publish Stage-7 token payload")?;
        fs::hard_link(staging.join("manifest.json"), output.join("manifest.json"))
            .context("publish Stage-7 manifest")?;
        #[cfg(unix)]
        File::open(output)?.sync_all()?;
        sync_parent(output)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(output);
    }
    result?;
    fs::remove_dir_all(staging).context("remove Stage-7 evidence staging directory")?;
    Ok(())
}

fn valid_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
