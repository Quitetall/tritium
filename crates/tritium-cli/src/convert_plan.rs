//! Shared convert/quantize filesystem receiver for the actual engine (ADR0063).
//! No fitting, identities, package assembly or publication logic lives here.
use crate::stage_input::{StageArgs, read_bounded};
use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use tritium_evidence::{
    CALIBRATED_LEVELS_FITTED_REGISTRATION, CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION,
    CALIBRATED_QUANTIZATION_STARTED_REGISTRATION, DIRECT_LEVELS_FITTED_REGISTRATION,
    EvidenceRecorder, QUANTIZATION_COMPLETED_REGISTRATION, QUANTIZATION_STARTED_REGISTRATION,
};
use tritium_quantize::engine::{
    DirectoryTensorSink, Engine, FileWeightSource, QuantizedModel, TritCalibrationSource,
    file_calibration_source,
};
use tritium_schema::{CalibratedQuantPlan, ModelQuantPlan, QuantPlan};

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
    #[command(flatten)]
    stage: StageArgs,
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
    let caches = paths(&args.calibration_cache)?;
    let mut calibration = file_calibration_source(view, caches)?;
    let run = args
        .conversion_run_id
        .as_deref()
        .context("missing --conversion-run-id")?;
    ensure!(!run.is_empty(), "--conversion-run-id must not be empty");
    let recorder = recorder(run)?;
    if args.stage.enabled() {
        let Request::Calibrated(plan) = request else {
            bail!("original-captured stage requires a calibrated plan");
        };
        let result = run_stage(args, &plan, &mut calibration, output, recorder, max)?;
        return print_result(output, &result);
    }
    let assets = paths(&args.asset)?;
    let mut source = FileWeightSource::open(view, &args.source_shard, assets, gram_paths)?;
    let mut sink = DirectoryTensorSink::new(output);
    let result = match request {
        Request::Plain(plan) => Engine::run(&plan, &mut source, &mut sink, recorder)?,
        Request::Calibrated(plan) => {
            Engine::run_calibrated(&plan, &mut source, &mut calibration, &mut sink, recorder)?
        }
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

fn run_stage(
    args: &PlanArgs,
    plan: &CalibratedQuantPlan,
    calibration: &mut TritCalibrationSource,
    output: &Path,
    recorder: EvidenceRecorder,
    max_json: u64,
) -> Result<QuantizedModel> {
    ensure!(
        args.source_shard.is_empty() && args.asset.is_empty() && args.input_gram.is_empty(),
        "original-captured stage conflicts with raw source inputs"
    );
    let inputs = args.stage.read(max_json)?;
    let mut stage = inputs.open()?;
    Ok(Engine::run_original_captured_salience(
        plan,
        &mut stage,
        &inputs.capture,
        calibration,
        &mut DirectoryTensorSink::new(output),
        recorder,
        inputs.conversion_limits,
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
