//! One filesystem adapter for original-captured conversion and cold generation.
//! Actual graph/identity/profile checks remain in the shared Rust modules.
use anyhow::{Context, Result, ensure};
use clap::Args;
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
use tritium_evidence::{VerifiedLog, verify_jsonl};
use tritium_quantize::engine::StagedSalienceSource;
use tritium_schema::{SalienceConversionLimits, SalienceStageLimits};

pub(crate) const DEFAULT_MAX_JSON_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Args, Debug)]
pub(crate) struct StageArgs {
    /// Immutable original-captured B .trit stage; no implicit ancestry upgrade.
    #[arg(long, requires_all = ["transform_evidence", "original_salience_evidence", "capture_evidence", "stage_limits", "conversion_limits"])]
    original_captured_stage: Option<PathBuf>,
    /// Complete canonical A-to-B transform JSONL log.
    #[arg(long, requires = "original_captured_stage")]
    transform_evidence: Option<PathBuf>,
    /// Complete original-A salience capture JSONL log.
    #[arg(long, requires = "original_captured_stage")]
    original_salience_evidence: Option<PathBuf>,
    /// Complete actual B next-token capture JSONL log.
    #[arg(long, requires = "original_captured_stage")]
    capture_evidence: Option<PathBuf>,
    /// Strict Rust SalienceStageLimits JSON; independent receiving byte ceilings.
    #[arg(long, requires = "original_captured_stage")]
    stage_limits: Option<PathBuf>,
    /// Strict Rust SalienceConversionLimits JSON; physical C and manifest ceilings.
    #[arg(long, requires = "original_captured_stage")]
    conversion_limits: Option<PathBuf>,
    /// Positive aggregate input-log bound (default 16MiB), not RSS.
    #[arg(long, requires = "original_captured_stage")]
    max_evidence_bytes: Option<u64>,
}
impl StageArgs {
    pub(crate) fn enabled(&self) -> bool {
        self.original_captured_stage.is_some()
    }

    pub(crate) fn read(&self, max_json: u64) -> Result<StageInputs> {
        ensure!(max_json > 0, "limits JSON byte bound must be positive");
        let stage_limits: SalienceStageLimits =
            serde_json::from_slice(&read_bounded(required(&self.stage_limits)?, max_json)?)
                .context("parse Rust SalienceStageLimits")?;
        ensure!(
            [
                stage_limits.max_package_bytes,
                stage_limits.max_manifest_bytes,
                stage_limits.max_tensor_bytes,
                stage_limits.max_asset_bytes,
                stage_limits.max_statistics_bytes,
            ]
            .into_iter()
            .all(|limit| limit > 0),
            "stage limits must be positive"
        );
        let conversion_limits: SalienceConversionLimits =
            serde_json::from_slice(&read_bounded(required(&self.conversion_limits)?, max_json)?)
                .context("parse Rust SalienceConversionLimits")?;
        conversion_limits.validate()?;
        let mut remaining = self.max_evidence_bytes.unwrap_or(DEFAULT_MAX_JSON_BYTES);
        ensure!(remaining > 0, "--max-evidence-bytes must be positive");
        let transform = read_evidence(required(&self.transform_evidence)?, &mut remaining)?;
        let original = read_evidence(required(&self.original_salience_evidence)?, &mut remaining)?;
        let capture = read_evidence(required(&self.capture_evidence)?, &mut remaining)?;
        let bytes = read_bounded(
            required(&self.original_captured_stage)?,
            stage_limits.max_package_bytes,
        )?;
        Ok(StageInputs {
            bytes,
            transform,
            original,
            capture,
            stage_limits,
            conversion_limits,
            remaining,
        })
    }
}
pub(crate) fn required(path: &Option<PathBuf>) -> Result<&Path> {
    path.as_deref().context("missing required stage input")
}

pub(crate) struct StageInputs {
    bytes: Vec<u8>,
    transform: VerifiedLog,
    original: VerifiedLog,
    pub(crate) capture: VerifiedLog,
    stage_limits: SalienceStageLimits,
    pub(crate) conversion_limits: SalienceConversionLimits,
    remaining: u64,
}
impl StageInputs {
    pub(crate) fn open(&self) -> Result<StagedSalienceSource<'_>> {
        Ok(StagedSalienceSource::open_original_captured(
            &self.bytes,
            &self.transform,
            &self.original,
            self.stage_limits,
        )?)
    }
    pub(crate) fn read_conversion_log(&mut self, path: &Path) -> Result<VerifiedLog> {
        read_evidence(path, &mut self.remaining)
    }
}

pub(crate) fn regular_file(path: &Path) -> Result<File> {
    ensure!(
        std::fs::metadata(path)
            .with_context(|| format!("inspect {}", path.display()))?
            .is_file(),
        "input must be a regular file: {}",
        path.display()
    );
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    ensure!(
        file.metadata()?.is_file(),
        "opened input is not a regular file: {}",
        path.display()
    );
    Ok(file)
}
pub(crate) fn read_bounded(path: &Path, max: u64) -> Result<Vec<u8>> {
    let mut file = regular_file(path)?;
    let len = file.metadata()?.len();
    ensure!(
        len <= max,
        "input {} exceeds byte limit {max}",
        path.display()
    );
    let len = usize::try_from(len).context("file size exceeds platform limit")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .context("reserve bounded input")?;
    bytes.resize(len, 0);
    file.read_exact(&mut bytes)
        .with_context(|| format!("read {}", path.display()))?;
    ensure!(
        file.read(&mut [0])? == 0,
        "input grew during read: {}",
        path.display()
    );
    Ok(bytes)
}
fn read_evidence(path: &Path, remaining: &mut u64) -> Result<VerifiedLog> {
    let bytes = read_bounded(path, *remaining)?;
    *remaining -= u64::try_from(bytes.len()).context("evidence platform size")?;
    let text = std::str::from_utf8(&bytes).context("evidence must be UTF-8 JSONL")?;
    verify_jsonl(text).with_context(|| format!("verify evidence {}", path.display()))
}
