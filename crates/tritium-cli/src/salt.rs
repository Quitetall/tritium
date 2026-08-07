//! Honest local entrypoints for the SALT V2 production campaign.

use std::{
    ffi::OsString,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::{Context, bail};
use clap::Subcommand;
use serde::Serialize;
use tritium_quantize::{Qwen35CoverageSummary, Qwen35CoverageTotals};
use tritium_salt::{
    Qwen36AdmittedSource, Qwen36LanguageCoverage, STAGE7_TOKENS_PER_SEQUENCE, Stage7Partition,
    Stage7TokenEvidencePack,
};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// SALT V2 source-admission and synthesis operations.
#[derive(Debug, Subcommand)]
pub(crate) enum SaltCommand {
    /// Stream and durably admit the pinned Qwen3.6-27B campaign candidate.
    Qwen36Preflight {
        /// Stable local snapshot (`config.json`, index, and all weight shards).
        /// The caller must prevent concurrent in-place mutation during the scan.
        #[arg(long)]
        model_dir: PathBuf,
        /// Content-addressed root for the canonical `ingest.tq36` proof.
        #[arg(long)]
        work_root: PathBuf,
        /// Immutable JSON projection of the admitted candidate proof.
        #[arg(long)]
        output: PathBuf,
    },
    /// Build exact Stage-7 token evidence from pinned, preselected source rows.
    BuildStage7EvidencePack {
        /// SmolLM Hugging Face snapshot containing exact tokenizer assets.
        #[arg(long)]
        model_dir: PathBuf,
        /// Content-bound `tritium.stage7-sampled-rows.v1` input manifest.
        #[arg(long)]
        sampled_rows: PathBuf,
        /// New directory receiving `manifest.json` and `stage7.u32le`.
        #[arg(long)]
        output_dir: PathBuf,
    },
    /// Strictly reopen Stage-7 evidence and read one bounded token window.
    InspectStage7EvidencePack {
        /// Model snapshot whose tokenizer identity must match the pack.
        #[arg(long)]
        model_dir: PathBuf,
        /// Exact `tritium.stage7-token-evidence-pack.v1` manifest.
        #[arg(long)]
        manifest: PathBuf,
        /// Pack identity frozen by campaign/recipe provenance.
        #[arg(long)]
        expected_pack_id: String,
        /// Frozen evidence partition to read.
        #[arg(long)]
        partition: Stage7Partition,
        /// First sequence ordinal in the selected partition.
        #[arg(long)]
        start_sequence: usize,
        /// Number of complete 2,048-token sequences to read.
        #[arg(long)]
        sequence_count: usize,
    },
    /// Seal source-bound HESTIA analytic and portable CPU/CUDA Gate-C evidence.
    #[cfg(feature = "cuda")]
    SealHestiaGateC {
        /// Release identifier bound by the Stage-7 campaign.
        #[arg(long)]
        release: String,
        /// Exact clean 40-character Git revision compiled into both backends.
        #[arg(long)]
        source_revision: String,
        /// Immutable JSON receipt destination.
        #[arg(long)]
        output: PathBuf,
        /// CUDA device ordinal used for physical conformance.
        #[arg(long, default_value_t = 0)]
        cuda_device: usize,
    },
}

pub(crate) fn run(command: SaltCommand) -> anyhow::Result<()> {
    match command {
        SaltCommand::Qwen36Preflight {
            model_dir,
            work_root,
            output,
        } => qwen36_preflight(&model_dir, &work_root, &output),
        SaltCommand::BuildStage7EvidencePack {
            model_dir,
            sampled_rows,
            output_dir,
        } => crate::stage7_evidence::build(&model_dir, &sampled_rows, &output_dir),
        SaltCommand::InspectStage7EvidencePack {
            model_dir,
            manifest,
            expected_pack_id,
            partition,
            start_sequence,
            sequence_count,
        } => inspect_stage7_evidence_pack(
            &model_dir,
            &manifest,
            &expected_pack_id,
            partition,
            start_sequence,
            sequence_count,
        ),
        #[cfg(feature = "cuda")]
        SaltCommand::SealHestiaGateC {
            release,
            source_revision,
            output,
            cuda_device,
        } => crate::hestia_gate::seal(&release, &source_revision, &output, cuda_device),
    }
}

fn inspect_stage7_evidence_pack(
    model_dir: &Path,
    manifest: &Path,
    expected_pack_id: &str,
    partition: Stage7Partition,
    start_sequence: usize,
    sequence_count: usize,
) -> anyhow::Result<()> {
    let (tokenizer_digest, tokenizer_vocab_size) =
        crate::stage7_evidence::model_tokenizer_identity(model_dir)?;
    let mut evidence = Stage7TokenEvidencePack::open(
        manifest,
        expected_pack_id,
        &tokenizer_digest,
        tokenizer_vocab_size,
    )?;
    let batch = evidence.read_sequences(partition, start_sequence, sequence_count)?;
    let receipt = evidence.finish()?;
    let report = Stage7EvidenceReadReport {
        schema: "tritium.stage7-token-evidence-read.v1",
        pack_id: receipt.pack_id(),
        tokenizer_digest: receipt.tokenizer_digest(),
        tokenizer_vocab_size: receipt.tokenizer_vocab_size(),
        token_payload_sha256: receipt.token_payload_sha256(),
        partition: batch.partition(),
        sampling_seed: batch.sampling_seed(),
        start_sequence: batch.start_sequence(),
        sequence_count: batch.sequence_count(),
        tokens_per_sequence: STAGE7_TOKENS_PER_SEQUENCE,
        token_count: batch.tokens().len(),
        sequence_ids: batch.sequence_ids(),
        ordered_token_sha256: batch.ordered_token_sha256(),
        terminal_validated: true,
    };
    let mut rendered = serde_json::to_vec_pretty(&report)?;
    rendered.push(b'\n');
    print!(
        "{}",
        String::from_utf8(rendered).expect("report JSON is UTF-8")
    );
    Ok(())
}

#[derive(Debug, Serialize)]
struct Stage7EvidenceReadReport<'a> {
    schema: &'static str,
    pack_id: &'a str,
    tokenizer_digest: &'a str,
    tokenizer_vocab_size: u32,
    token_payload_sha256: &'a str,
    partition: Stage7Partition,
    sampling_seed: u64,
    start_sequence: usize,
    sequence_count: usize,
    tokens_per_sequence: usize,
    token_count: usize,
    sequence_ids: &'a [String],
    ordered_token_sha256: &'a str,
    terminal_validated: bool,
}

fn qwen36_preflight(model_dir: &Path, work_root: &Path, output: &Path) -> anyhow::Result<()> {
    validate_paths(model_dir, work_root, output)?;
    let admitted =
        Qwen36AdmittedSource::open(model_dir, tritium_nn::QWEN36_27B_REVISION, work_root)
            .with_context(|| {
                format!(
                    "admit pinned Qwen3.6-27B source candidate {}",
                    model_dir.display()
                )
            })?;
    let report = Qwen36PreflightReport::from_admitted(&admitted);
    let mut bytes = serde_json::to_vec_pretty(&report)?;
    bytes.push(b'\n');
    publish_immutable(output, &bytes)?;
    print!(
        "{}",
        String::from_utf8(bytes).expect("JSON output is UTF-8")
    );
    Ok(())
}

#[derive(Debug, Serialize)]
struct Qwen36PreflightReport {
    schema: &'static str,
    claim_scope: &'static str,
    repository: &'static str,
    revision: &'static str,
    source_model_id: String,
    manifest_content_id: String,
    proof_id: String,
    proof_relative_path: String,
    proof_bytes: u64,
    identity_status: &'static str,
    official_payload_authenticated: bool,
    source_architecture: String,
    source_config_digest: String,
    coverage_policy_digest: String,
    metadata_digest: String,
    metadata_record_bytes: u64,
    source_payload_bytes: u64,
    coverage: CoverageReport,
    language_schema: LanguageReport,
    mtp_execution_verified: bool,
    quality_evaluated: bool,
    physical_artifact_measured: bool,
    sota_claim_eligible: bool,
}

impl Qwen36PreflightReport {
    fn from_admitted(admitted: &Qwen36AdmittedSource) -> Self {
        let proof = admitted.proof();
        let receipt = admitted.receipt();
        let manifest_content_id = receipt.manifest_content_id().to_string();
        let proof_id = receipt.proof_id().to_string();
        Self {
            schema: "tritium.qwen36-source-admission.v1",
            claim_scope: "revision-declared-candidate-language-plus-mtp-vision-deferred",
            repository: tritium_nn::QWEN36_27B_REPOSITORY,
            revision: tritium_nn::QWEN36_27B_REVISION,
            source_model_id: receipt.source_model_id().to_string(),
            manifest_content_id: manifest_content_id.clone(),
            proof_id: proof_id.clone(),
            proof_relative_path: format!(
                "qwen36-source/{manifest_content_id}/{proof_id}/ingest.tq36"
            ),
            proof_bytes: receipt.proof_bytes(),
            identity_status: receipt.identity_status().as_str(),
            official_payload_authenticated: receipt
                .identity_status()
                .official_payload_authenticated(),
            source_architecture: proof.manifest().architecture().to_owned(),
            source_config_digest: hex_digest(proof.manifest().config_digest()),
            coverage_policy_digest: hex_digest(&proof.coverage().policy_digest()),
            metadata_digest: hex_digest(proof.coverage().metadata_digest()),
            metadata_record_bytes: proof.coverage().metadata_record_bytes(),
            source_payload_bytes: proof.payload_bytes(),
            coverage: CoverageReport::new(proof.coverage().summary()),
            language_schema: LanguageReport::new(proof.language()),
            mtp_execution_verified: false,
            quality_evaluated: false,
            physical_artifact_measured: false,
            sota_claim_eligible: false,
        }
    }
}

#[derive(Debug, Serialize)]
struct CoverageReport {
    total: TotalsReport,
    language: TotalsReport,
    mtp: TotalsReport,
    deferred_vision: TotalsReport,
    included: TotalsReport,
    additive_ternary: TotalsReport,
    preserve_source: TotalsReport,
    excluded_future_vision: TotalsReport,
}

impl CoverageReport {
    fn new(summary: Qwen35CoverageSummary) -> Self {
        Self {
            total: TotalsReport::new(summary.total()),
            language: TotalsReport::new(summary.language()),
            mtp: TotalsReport::new(summary.mtp()),
            deferred_vision: TotalsReport::new(summary.vision()),
            included: TotalsReport::new(summary.included()),
            additive_ternary: TotalsReport::new(summary.additive_ternary()),
            preserve_source: TotalsReport::new(summary.preserve_source()),
            excluded_future_vision: TotalsReport::new(summary.excluded_future_vision()),
        }
    }
}

#[derive(Debug, Serialize)]
struct TotalsReport {
    tensors: u64,
    coefficients: u64,
}

impl TotalsReport {
    const fn new(totals: Qwen35CoverageTotals) -> Self {
        Self {
            tensors: totals.tensors(),
            coefficients: totals.coefficients(),
        }
    }
}

#[derive(Debug, Serialize)]
struct LanguageReport {
    tensors: u64,
    additive_matrices: u64,
    preserved_tensors: u64,
    deferred_mtp_tensors: u64,
    deferred_vision_tensors: u64,
}

impl LanguageReport {
    const fn new(language: Qwen36LanguageCoverage) -> Self {
        Self {
            tensors: language.tensors(),
            additive_matrices: language.matrices(),
            preserved_tensors: language.preserved(),
            deferred_mtp_tensors: language.deferred_mtp(),
            deferred_vision_tensors: language.deferred_vision(),
        }
    }
}

fn validate_paths(model_dir: &Path, work_root: &Path, output: &Path) -> anyhow::Result<()> {
    let model = std::fs::canonicalize(model_dir)
        .with_context(|| format!("canonicalize model directory {}", model_dir.display()))?;
    if !std::fs::metadata(&model)
        .with_context(|| format!("inspect model directory {}", model.display()))?
        .is_dir()
    {
        bail!("model path {} is not a directory", model_dir.display());
    }
    let work = path_identity(work_root)?;
    let output_identity = path_identity(output)?;
    if work.starts_with(&model) || model.starts_with(&work) {
        bail!("SALT work root must not overlap model directory");
    }
    if output_identity.starts_with(&model) {
        bail!("preflight JSON output must be outside model directory");
    }
    if output_identity.starts_with(&work) || work.starts_with(&output_identity) {
        bail!("preflight JSON output and content-addressed work root must not overlap");
    }
    match fs::symlink_metadata(work_root) {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
        Ok(_) => bail!("SALT work root exists but is not a real directory"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("inspect SALT work root"),
    }
    match fs::symlink_metadata(output) {
        Ok(metadata) if metadata.file_type().is_file() => {}
        Ok(_) => bail!("preflight JSON output exists but is not a regular file"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("inspect preflight JSON output"),
    }
    Ok(())
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

pub(crate) fn publish_immutable(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    match fs::read(path) {
        Ok(existing) => {
            if existing == bytes {
                return Ok(());
            }
            bail!(
                "refusing to replace different immutable output {}",
                path.display()
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).context("read existing immutable output"),
    }
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)
        .with_context(|| format!("create immutable output directory {}", parent.display()))?;
    let file_name = path
        .file_name()
        .context("immutable output has no file name")?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let mut temporary_name = OsString::from(file_name);
    temporary_name.push(format!(".tmp-{}-{sequence}", std::process::id()));
    let temporary = parent.join(temporary_name);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .with_context(|| format!("create temporary immutable output {}", temporary.display()))?;
    if let Err(error) = file.write_all(bytes).and_then(|()| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(error).context("write temporary immutable output");
    }
    drop(file);
    match fs::hard_link(&temporary, path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing =
                fs::read(path).context("read concurrently published immutable output")?;
            if existing != bytes {
                let _ = fs::remove_file(&temporary);
                bail!("concurrent immutable output differs from this admission");
            }
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(error).context("publish immutable output");
        }
    }
    fs::remove_file(&temporary).context("remove temporary immutable output")?;
    #[cfg(unix)]
    fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("sync immutable output directory {}", parent.display()))?;
    Ok(())
}

fn hex_digest(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tritium-cli-salt-{label}-{}-{}",
            std::process::id(),
            TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn immutable_output_is_idempotent_and_never_replaced() {
        let root = test_root("immutable");
        fs::create_dir_all(&root).unwrap();
        let output = root.join("proof.json");
        publish_immutable(&output, b"candidate\n").unwrap();
        publish_immutable(&output, b"candidate\n").unwrap();
        assert!(publish_immutable(&output, b"different\n").is_err());
        assert_eq!(fs::read(&output).unwrap(), b"candidate\n");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn path_gate_rejects_outputs_and_work_below_the_model() {
        let root = test_root("paths");
        let model = root.join("model");
        let outside = root.join("outside");
        fs::create_dir_all(&model).unwrap();
        fs::create_dir_all(&outside).unwrap();
        assert!(validate_paths(&model, &model.join("work"), &outside.join("proof.json")).is_err());
        assert!(validate_paths(&model, &outside.join("work"), &model.join("proof.json")).is_err());
        assert!(
            validate_paths(
                &model,
                &outside.join("work"),
                &outside.join("work/proof.json")
            )
            .is_err()
        );
        let invalid_work = outside.join("work-file");
        fs::write(&invalid_work, b"not a directory").unwrap();
        assert!(validate_paths(&model, &invalid_work, &outside.join("proof.json")).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn digest_rendering_is_fixed_width_lower_hex() {
        assert_eq!(
            hex_digest(&[0xab; 32]),
            "abababababababababababababababababababababababababababababababab"
        );
    }
}
