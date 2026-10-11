//! Filesystem inputs for the actual shared model conversion engine (ADR0063).
//! No fitting, identities, package assembly or publication logic lives here.
use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};
use tritium_evidence::{
    CALIBRATED_LEVELS_FITTED_REGISTRATION, CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
    CALIBRATED_QUANTIZATION_STARTED_REGISTRATION, DIRECT_LEVELS_FITTED_REGISTRATION,
    EvidenceRecorder, QUANTIZATION_COMPLETED_REGISTRATION, QUANTIZATION_STARTED_REGISTRATION,
    VerifiedLog, verify_jsonl,
};
use tritium_quantize::engine::{
    DirectoryTensorSink, Engine, QuantEngineError, QuantizedModel, SafeTensorsWeightSource,
    StagedSalienceSource, TritCalibrationSource, WeightSource,
};
use tritium_schema::{
    CalibratedQuantPlan, DirectFitMetric, ModelQuantPlan, QuantActionRef, QuantPlan,
    QuantSourceTensor, SalienceConversionLimits, SalienceStageLimits,
};

const DEFAULT_MAX_PLAN_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug)]
pub(crate) struct NamedPath {
    name: String,
    path: PathBuf,
}
fn parse_named_path(value: &str) -> std::result::Result<NamedPath, String> {
    let (name, path) = value.split_once('=').ok_or("expected NAME=PATH")?;
    if name.is_empty() || path.is_empty() {
        return Err("NAME and PATH must be nonempty".into());
    }
    Ok(NamedPath {
        name: name.into(),
        path: path.into(),
    })
}

#[derive(Args, Debug)]
pub(crate) struct PlanArgs {
    /// Existing Rust QuantPlan JSON; mutually exclusive with legacy recipe flags.
    #[arg(long, requires_all = ["conversion_source", "conversion_run_id"], conflicts_with = "original_captured_stage")]
    plan: Option<PathBuf>,
    /// Existing Rust CalibratedQuantPlan JSON, with actual bound .trit caches.
    #[arg(long, requires_all = ["conversion_source", "conversion_run_id"])]
    calibrated_plan: Option<PathBuf>,
    /// Repeat for every original safetensors shard; no source tensor may be omitted.
    #[arg(long, requires = "conversion_plan")]
    source_shard: Vec<PathBuf>,
    /// Original asset basename and file, NAME=PATH; repeat for the exact plan inventory.
    #[arg(long, requires = "conversion_plan", conflicts_with = "original_captured_stage", value_parser = parse_named_path)]
    asset: Vec<NamedPath>,
    /// Original row-major little-endian f64 Gram, TENSOR=PATH; no implicit fallback.
    #[arg(long, requires = "conversion_plan", conflicts_with = "original_captured_stage", value_parser = parse_named_path)]
    input_gram: Vec<NamedPath>,
    /// Actual .trit calibration cache, TENSOR=PATH; exact calibrated inventory required.
    #[arg(long, requires = "calibrated_plan", conflicts_with = "plan", value_parser = parse_named_path)]
    calibration_cache: Vec<NamedPath>,
    /// Stable identity for the mandatory conversion log, separate from CLI summary evidence.
    #[arg(long, requires = "conversion_plan")]
    conversion_run_id: Option<String>,
    /// Positive physical JSON plan bound (default 16MiB); not a process-RSS bound.
    #[arg(long, requires = "conversion_plan")]
    max_plan_bytes: Option<u64>,
    /// Immutable original-captured B .trit stage; no raw source or implicit ancestry upgrade.
    #[arg(long, requires_all = ["calibrated_plan", "transform_evidence", "original_salience_evidence", "capture_evidence", "stage_limits", "conversion_limits"])]
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
    /// Positive aggregate bound on the three input logs (default 16MiB), not RSS.
    #[arg(long, requires = "original_captured_stage")]
    max_evidence_bytes: Option<u64>,
}
impl PlanArgs {
    pub(crate) fn enabled(&self) -> bool {
        self.plan.is_some() || self.calibrated_plan.is_some()
    }
}

enum Request {
    Plain(QuantPlan),
    Calibrated(CalibratedQuantPlan),
}
impl Request {
    fn view(&self) -> ModelQuantPlan<'_> {
        match self {
            Self::Plain(p) => ModelQuantPlan::Uncalibrated(p),
            Self::Calibrated(p) => ModelQuantPlan::Calibrated(p),
        }
    }
}

fn paths(bindings: &[NamedPath]) -> Result<BTreeMap<String, PathBuf>> {
    let mut result = BTreeMap::new();
    for binding in bindings {
        ensure!(
            result
                .insert(binding.name.clone(), binding.path.clone())
                .is_none(),
            "duplicate input binding: {}",
            binding.name
        );
    }
    Ok(result)
}
fn inventory<'a>(
    paths: &BTreeMap<String, PathBuf>,
    expected: impl Iterator<Item = &'a str>,
    kind: &str,
) -> Result<()> {
    ensure!(
        paths.keys().map(String::as_str).eq(expected),
        "{kind} input inventory does not match plan"
    );
    Ok(())
}

fn regular_file(path: &Path) -> Result<File> {
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
fn read_bounded(path: &Path, max: u64) -> Result<Vec<u8>> {
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

// Real file-backed raw Grams are loaded only when requested by the common engine;
// do not retain every model's square Gram in the safetensors adapter's memory map.
struct FileWeightSource {
    raw: SafeTensorsWeightSource<File>,
    grams: BTreeMap<String, PathBuf>,
    gram_sizes: BTreeMap<String, u64>,
}
impl WeightSource for FileWeightSource {
    fn inventory(&self) -> std::result::Result<Vec<QuantSourceTensor>, QuantEngineError> {
        self.raw.inventory()
    }
    fn asset_names(&self) -> Vec<String> {
        self.raw.asset_names()
    }
    fn tensor(
        &mut self,
        name: &str,
        max: usize,
    ) -> std::result::Result<Cow<'_, [u8]>, QuantEngineError> {
        self.raw.tensor(name, max)
    }
    fn asset(
        &mut self,
        name: &str,
        max: usize,
    ) -> std::result::Result<Cow<'_, [u8]>, QuantEngineError> {
        self.raw.asset(name, max)
    }
    fn input_gram(
        &mut self,
        name: &str,
        max: usize,
    ) -> std::result::Result<Cow<'_, [f64]>, QuantEngineError> {
        (|| -> Result<Vec<f64>> {
            let path = self
                .grams
                .get(name)
                .context("missing requested raw Gram file")?;
            let expected = *self
                .gram_sizes
                .get(name)
                .context("missing requested Gram shape")?;
            ensure!(
                expected / 8 <= u64::try_from(max)?,
                "requested Gram exceeds engine limit"
            );
            let bytes = read_bounded(path, expected)?;
            ensure!(
                u64::try_from(bytes.len())? == expected,
                "Gram file has wrong square length: {name}"
            );
            let mut values = Vec::new();
            values
                .try_reserve_exact(bytes.len() / 8)
                .context("reserve bounded Gram")?;
            values.extend(
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| f64::from_le_bytes(*b)),
            );
            Ok(values)
        })()
        .map(Cow::Owned)
        .map_err(|e| QuantEngineError::Source(format!("raw Gram {name}: {e:#}")))
    }
}

pub(crate) fn run(args: &PlanArgs, output: &Path) -> Result<()> {
    let max = args.max_plan_bytes.unwrap_or(DEFAULT_MAX_PLAN_BYTES);
    ensure!(max > 0, "--max-plan-bytes must be positive");
    let request = match (&args.plan, &args.calibrated_plan) {
        (Some(path), None) => Request::Plain(
            serde_json::from_slice(&read_bounded(path, max)?).context("parse Rust QuantPlan")?,
        ),
        (None, Some(path)) => Request::Calibrated(
            serde_json::from_slice(&read_bounded(path, max)?)
                .context("parse Rust CalibratedQuantPlan")?,
        ),
        _ => bail!("exactly one typed conversion plan is required"),
    };
    let view = request.view();
    view.validate().context("validate conversion plan")?;
    let gram_paths = paths(&args.input_gram)?;
    let mut gram_sizes = BTreeMap::new();
    let mut cache_names = Vec::new();
    for tensor in view.tensors() {
        match tensor.action {
            QuantActionRef::Fit {
                plan,
                metric: DirectFitMetric::InputGramF64 { .. },
            } => {
                let entries = plan
                    .layout
                    .cols
                    .checked_mul(plan.layout.cols)
                    .context("Gram shape overflow")?;
                ensure!(
                    entries <= plan.limits.max_metric_entries,
                    "Gram shape exceeds requested metric limit"
                );
                gram_sizes.insert(
                    tensor.source.name.clone(),
                    entries.checked_mul(8).context("Gram byte overflow")?,
                );
            }
            QuantActionRef::CalibratedFit(_) => cache_names.push(tensor.source.name.as_str()),
            QuantActionRef::KeepDense
            | QuantActionRef::Fit {
                metric: DirectFitMetric::Identity,
                ..
            } => {}
            _ => bail!("unsupported typed plan action or metric"),
        }
    }
    inventory(
        &gram_paths,
        gram_sizes.keys().map(String::as_str),
        "raw Gram",
    )?;
    let caches = paths(&args.calibration_cache)?;
    inventory(&caches, cache_names.into_iter(), "calibration cache")?;
    let run = args
        .conversion_run_id
        .as_deref()
        .context("missing --conversion-run-id")?;
    ensure!(!run.is_empty(), "--conversion-run-id must not be empty");
    let recorder = recorder(run)?;
    if args.original_captured_stage.is_some() {
        let Request::Calibrated(plan) = request else {
            bail!("original-captured stage requires a calibrated plan");
        };
        let result = run_stage(args, &plan, caches, output, recorder, max)?;
        return print_result(output, &result);
    }
    let assets = paths(&args.asset)?;
    inventory(&assets, view.assets().iter().map(String::as_str), "asset")?;
    ensure!(
        !args.source_shard.is_empty(),
        "at least one --source-shard is required"
    );
    let readers = args
        .source_shard
        .iter()
        .map(|p| regular_file(p))
        .collect::<Result<Vec<_>>>()?;
    let mut remaining = view.limits().max_asset_bytes;
    let assets = assets
        .into_iter()
        .map(|(name, path)| {
            let bytes = read_bounded(&path, remaining)?;
            remaining -= u64::try_from(bytes.len()).context("asset platform size")?;
            Ok((name, bytes))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut source = FileWeightSource {
        raw: SafeTensorsWeightSource::new(readers, assets, BTreeMap::new())?,
        grams: gram_paths,
        gram_sizes,
    };
    let mut sink = DirectoryTensorSink::new(output);
    let result = match request {
        Request::Plain(plan) => Engine::run(&plan, &mut source, &mut sink, recorder)?,
        Request::Calibrated(plan) => Engine::run_calibrated(
            &plan,
            &mut source,
            &mut TritCalibrationSource::new(caches)?,
            &mut sink,
            recorder,
        )?,
    };
    print_result(output, &result)
}

fn recorder(run: &str) -> Result<EvidenceRecorder> {
    Ok(EvidenceRecorder::new(
        run,
        [
            DIRECT_LEVELS_FITTED_REGISTRATION,
            QUANTIZATION_STARTED_REGISTRATION,
            QUANTIZATION_COMPLETED_REGISTRATION,
            CALIBRATED_LEVELS_FITTED_REGISTRATION,
            CALIBRATED_QUANTIZATION_STARTED_REGISTRATION,
            CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
        ],
    )?)
}

fn read_evidence(path: &Path, remaining: &mut u64) -> Result<VerifiedLog> {
    let bytes = read_bounded(path, *remaining)?;
    *remaining -= u64::try_from(bytes.len()).context("evidence platform size")?;
    let text = std::str::from_utf8(&bytes).context("evidence must be UTF-8 JSONL")?;
    verify_jsonl(text).with_context(|| format!("verify evidence {}", path.display()))
}

fn run_stage(
    args: &PlanArgs,
    plan: &CalibratedQuantPlan,
    caches: BTreeMap<String, PathBuf>,
    output: &Path,
    recorder: EvidenceRecorder,
    max_json: u64,
) -> Result<QuantizedModel> {
    ensure!(
        args.source_shard.is_empty() && args.asset.is_empty() && args.input_gram.is_empty(),
        "original-captured stage conflicts with raw source inputs"
    );
    let required = |path: &Option<PathBuf>| path.clone().context("missing required stage input");
    let stage_limits: SalienceStageLimits =
        serde_json::from_slice(&read_bounded(&required(&args.stage_limits)?, max_json)?)
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
    let conversion_limits: SalienceConversionLimits = serde_json::from_slice(&read_bounded(
        &required(&args.conversion_limits)?,
        max_json,
    )?)
    .context("parse Rust SalienceConversionLimits")?;
    conversion_limits.validate()?;
    let mut remaining = args.max_evidence_bytes.unwrap_or(DEFAULT_MAX_PLAN_BYTES);
    ensure!(remaining > 0, "--max-evidence-bytes must be positive");
    let transform = read_evidence(&required(&args.transform_evidence)?, &mut remaining)?;
    let original = read_evidence(&required(&args.original_salience_evidence)?, &mut remaining)?;
    let capture = read_evidence(&required(&args.capture_evidence)?, &mut remaining)?;
    let bytes = read_bounded(
        &required(&args.original_captured_stage)?,
        stage_limits.max_package_bytes,
    )?;
    let mut stage =
        StagedSalienceSource::open_original_captured(&bytes, &transform, &original, stage_limits)?;
    Ok(Engine::run_original_captured_salience(
        plan,
        &mut stage,
        &capture,
        &mut TritCalibrationSource::new(caches)?,
        &mut DirectoryTensorSink::new(output),
        recorder,
        conversion_limits,
    )?)
}

fn print_result(output: &Path, result: &QuantizedModel) -> Result<()> {
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "output": output,
            "source_id": crate::hex::hex_digest(result.source_id.as_bytes()),
            "package_id": crate::hex::hex_digest(result.package_id.as_bytes()),
            "model_ids": result.model_ids,
            "evidence_root": result.evidence.log().root_digest,
            "conversion_run_id": result.evidence.run(),
            "publication_warnings": result.warnings.iter().map(|w| format!("{w:?}")).collect::<Vec<_>>(),
        }))?
    );
    Ok(())
}
