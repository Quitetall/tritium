//! Reproducible benchmark/validation reports for `tritium report`.
//!
//! These commands are intentionally plain CLI reports, not divan harnesses: they
//! run one requested scenario and emit stable JSON/table output suitable for local
//! runs and CI logs.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context as _, bail};
use rayon::prelude::*;
use serde::Serialize;
use tritium_format::{SafeTensors, dequant_salt_row};
use tritium_nn::{ModelRunner, sample_greedy};
use tritium_quantize::{BaseScaleScope, QuantConfig, ReconAccum, Sensitivity, quantize_tensor};
use tritium_spec::DeviceCaps;

use crate::quantize::ScaleGroupArg;
use crate::{ReportFormat, SaltSensitivityArg};

const RTX_4090_PEAK_HBM_BW_BYTES_PER_SEC: f64 = 1008.0e9;
const BITNET_2B4T_I2S_BYTES: f64 = 1_187_801_280.0;
/// Keep in sync with `tritium_benches::TRITIUM_2B4T_DECODE_4090` (the CLI does
/// not depend on the bench crate). Re-recorded 2026-07-06 after the v1.x decode
/// passes (measured 289–336 tok/s on the build box; pinned below the
/// contended-desktop floor).
const TRITIUM_2B4T_DECODE_4090_BASELINE: f64 = 270.0;

#[derive(Debug, Serialize)]
struct DecodeReport {
    report: &'static str,
    backend: String,
    prompt_tokens: usize,
    decode_steps: usize,
    warmup_steps: usize,
    elapsed_ms: f64,
    tokens_per_sec: f64,
    ms_per_token: f64,
    roofline_4090_pct: f64,
    baseline_4090_drop_pct: f64,
}

#[derive(Debug, Serialize)]
struct TtftReport {
    report: &'static str,
    backend: String,
    prompt_tokens: usize,
    runs: usize,
    total_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    tokens_per_sec: f64,
}

/// Environment capture for the reproducibility ledger (ADR 0026 Track R):
/// every BENCHMARKS.md number carries the box state it was measured on.
/// Captured via `nvidia-smi` when present ("unavailable" otherwise) — an
/// observation of the environment, not a measurement dependency.
#[derive(Debug, Serialize)]
struct EnvCapture {
    gpu: String,
    driver: String,
    vram_total_mib: String,
    vram_used_mib: String,
    /// Other compute processes resident on the GPU at capture time — the
    /// honest contention disclosure (a co-resident llama-server changes
    /// every number).
    co_resident: Vec<String>,
    date_utc: String,
    git_commit: String,
}

#[derive(Debug, Serialize)]
struct CompareReport {
    report: &'static str,
    model: String,
    model_bytes: u64,
    environment: EnvCapture,
    decode_runs: Vec<DecodeReport>,
    decode_median_tok_s: f64,
    prefill: TtftReport,
    prefill_tok_s: f64,
}

#[derive(Debug, Serialize)]
struct ParityReport {
    report: &'static str,
    prompt_tokens: usize,
    max_new: usize,
    matched_tokens: usize,
    cpu_tokens: Vec<u32>,
    cuda_tokens: Vec<u32>,
    exact_match: bool,
}

#[derive(Debug, Serialize)]
struct SaltReport {
    report: &'static str,
    rows: usize,
    k: usize,
    sensitivity: &'static str,
    budgets: Vec<SaltBudgetReport>,
}

#[derive(Debug, Serialize)]
struct SaltBudgetReport {
    requested_bpw: f64,
    logical_bpw: f64,
    /// Legacy nominal dense rate: maximum plane count times ternary entropy.
    dense_stored_bpw: f64,
    /// Packed TQ2_0 plane bytes, including block scales but excluding SALT headers.
    packed_weight_bpw: f64,
    /// Serialized SALT row bytes, including plane data and row headers.
    sidecar_bpw: f64,
    mse: f64,
    rmse: f64,
    max_abs_error: f64,
    plane_histogram: Vec<PlaneCount>,
}

#[derive(Debug, Serialize)]
struct PlaneCount {
    planes: usize,
    groups: usize,
}

pub(crate) fn decode(
    model_path: &Path,
    tokens: &[u32],
    backend: &str,
    decode_steps: usize,
    warmup: usize,
    format: ReportFormat,
) -> anyhow::Result<()> {
    if tokens.is_empty() {
        bail!("decode report requires at least one prompt token");
    }
    if decode_steps == 0 {
        bail!("decode report requires --decode-steps > 0");
    }
    let mut runner = load_runner(model_path, backend)?;
    let report = decode_core(&mut runner, tokens, backend, decode_steps, warmup)?;
    emit(format, &decode_table(&report), &report)
}

/// One timed decode run on an already-loaded runner (shared by `decode` and
/// `compare` — compare repeats it without re-loading the model).
fn decode_core(
    runner: &mut ModelRunner,
    tokens: &[u32],
    backend: &str,
    decode_steps: usize,
    warmup: usize,
) -> anyhow::Result<DecodeReport> {
    runner.reset();
    let positions: Vec<usize> = (0..tokens.len()).collect();
    let mut logits = runner
        .forward(tokens, &positions)
        .context("prefill failed")?;

    for step in 0..warmup {
        let next = sample_greedy(&logits).context("decode warmup produced empty logits")?;
        logits = runner
            .forward(&[next], &[tokens.len() + step])
            .context("decode warmup failed")?;
    }

    let start = Instant::now();
    for step in 0..decode_steps {
        let next = sample_greedy(&logits).context("decode step produced empty logits")?;
        logits = runner
            .forward(&[next], &[tokens.len() + warmup + step])
            .context("decode step failed")?;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let tokens_per_sec = decode_steps as f64 / elapsed;
    let ceiling = RTX_4090_PEAK_HBM_BW_BYTES_PER_SEC / BITNET_2B4T_I2S_BYTES;
    Ok(DecodeReport {
        report: "decode",
        backend: backend.to_owned(),
        prompt_tokens: tokens.len(),
        decode_steps,
        warmup_steps: warmup,
        elapsed_ms: elapsed * 1000.0,
        tokens_per_sec,
        ms_per_token: 1000.0 / tokens_per_sec,
        roofline_4090_pct: 100.0 * tokens_per_sec / ceiling,
        baseline_4090_drop_pct: 100.0
            * ((TRITIUM_2B4T_DECODE_4090_BASELINE - tokens_per_sec)
                / TRITIUM_2B4T_DECODE_4090_BASELINE)
                .max(0.0),
    })
}

pub(crate) fn ttft(
    model_path: &Path,
    tokens: &[u32],
    backend: &str,
    runs: usize,
    format: ReportFormat,
) -> anyhow::Result<()> {
    if tokens.is_empty() {
        bail!("ttft report requires at least one prompt token");
    }
    if runs == 0 {
        bail!("ttft report requires --runs > 0");
    }

    let mut runner = load_runner(model_path, backend)?;
    let report = ttft_core(&mut runner, tokens, backend, runs)?;
    emit(format, &ttft_table(&report), &report)
}

/// One prefill-latency measurement on an already-loaded runner (shared by
/// `ttft` and `compare`). `tokens_per_sec` here is the prefill throughput —
/// the pp512 ledger number when the prompt is 512 tokens.
fn ttft_core(
    runner: &mut ModelRunner,
    tokens: &[u32],
    backend: &str,
    runs: usize,
) -> anyhow::Result<TtftReport> {
    let positions: Vec<usize> = (0..tokens.len()).collect();
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        runner.reset();
        let start = Instant::now();
        runner
            .forward(tokens, &positions)
            .context("prefill failed")?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.total_cmp(b));
    let total_ms: f64 = samples.iter().sum();
    Ok(TtftReport {
        report: "ttft",
        backend: backend.to_owned(),
        prompt_tokens: tokens.len(),
        runs,
        total_ms,
        p50_ms: percentile_sorted(&samples, 0.50),
        p95_ms: percentile_sorted(&samples, 0.95),
        tokens_per_sec: (tokens.len() * runs) as f64 / (total_ms / 1000.0),
    })
}

/// One shell-out line of `nvidia-smi --query...`; "unavailable" on any error.
fn smi(args: &[&str]) -> String {
    std::process::Command::new("nvidia-smi")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unavailable".to_owned())
}

/// Resolve the ledger's `gpu` field from the two sources that can know it.
///
/// `nvidia-smi` wins whenever it answers: every historical BENCHMARKS.md row was
/// recorded with its string, and changing the basis under them would silently
/// invalidate the ledger. Off NVIDIA it reports nothing, so fall back to the
/// adapter name the initialised backend already carries in its [`DeviceCaps`] —
/// otherwise an AMD/Vulkan `report compare` emits a JSON that identifies no GPU at
/// all, which is not a receipt (issue #4). The `[backend]` suffix marks which
/// source answered so the two are never confused when read back.
fn resolve_gpu(smi_gpu: &str, caps: Option<&DeviceCaps>) -> String {
    if !smi_gpu.is_empty() && smi_gpu != "unavailable" {
        return smi_gpu.to_owned();
    }
    match caps {
        Some(c) if !c.device_name.is_empty() => {
            format!("{} [{}]", c.device_name, c.backend)
        }
        _ => "unavailable".to_owned(),
    }
}

/// Capture the box state for the ledger. Called BEFORE the model is loaded, so
/// `co_resident` and `vram_used_mib` describe the contention we are measuring
/// *against* rather than including our own allocation. The `gpu` field may come
/// back "unavailable" off NVIDIA; [`resolve_gpu`] backfills it from the backend
/// once one has initialised.
fn env_capture() -> EnvCapture {
    let co_resident = smi(&[
        "--query-compute-apps=process_name,used_memory",
        "--format=csv,noheader",
    ]);
    EnvCapture {
        gpu: smi(&["--query-gpu=name", "--format=csv,noheader"]),
        driver: smi(&["--query-gpu=driver_version", "--format=csv,noheader"]),
        vram_total_mib: smi(&["--query-gpu=memory.total", "--format=csv,noheader"]),
        vram_used_mib: smi(&["--query-gpu=memory.used", "--format=csv,noheader"]),
        co_resident: if co_resident == "unavailable" || co_resident.is_empty() {
            vec![]
        } else {
            co_resident.lines().map(str::to_owned).collect()
        },
        date_utc: std::process::Command::new("date")
            .args(["-u", "-Is"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .unwrap_or_default(),
        git_commit: std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .unwrap_or_default(),
    }
}

/// The one-command ledger bundle (ADR 0026 Track R): decode ×`reps` +
/// prefill p50 on one loaded runner, with environment capture. Every number
/// in docs/BENCHMARKS.md must reproduce from this command's JSON within
/// run-to-run spread.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compare(
    model_path: &Path,
    tokens: &[u32],
    backend: &str,
    prompt_len: usize,
    decode_steps: usize,
    warmup: usize,
    reps: usize,
    runs: usize,
    format: ReportFormat,
) -> anyhow::Result<()> {
    if tokens.is_empty() {
        bail!("compare report requires at least one prompt token");
    }
    if reps == 0 || runs == 0 || decode_steps == 0 {
        bail!("compare report requires --reps, --runs and --decode-steps > 0");
    }
    // Cycle/truncate to the requested prompt shape (512 = the pp512 ledger).
    let prompt: Vec<u32> = if prompt_len == 0 {
        tokens.to_vec()
    } else {
        tokens.iter().copied().cycle().take(prompt_len).collect()
    };
    // Snapshot the box BEFORE loading, so co-resident processes and used VRAM
    // describe the contention this run competes with, not our own model.
    let mut environment = env_capture();
    let (mut runner, caps) = load_runner_with_caps(model_path, backend)?;
    // Backfill only the GPU *identity* from the backend — off NVIDIA it is the sole
    // source, and without it the emitted JSON names no device (issue #4).
    environment.gpu = resolve_gpu(&environment.gpu, caps.as_ref());

    let mut decode_runs = Vec::with_capacity(reps);
    for _ in 0..reps {
        decode_runs.push(decode_core(
            &mut runner,
            &prompt,
            backend,
            decode_steps,
            warmup,
        )?);
    }
    let mut tok_s: Vec<f64> = decode_runs.iter().map(|r| r.tokens_per_sec).collect();
    tok_s.sort_by(|a, b| a.total_cmp(b));
    let decode_median_tok_s = tok_s[tok_s.len() / 2];

    let prefill = ttft_core(&mut runner, &prompt, backend, runs)?;
    let prefill_tok_s = prefill.tokens_per_sec;

    let report = CompareReport {
        report: "compare",
        model: model_path.display().to_string(),
        model_bytes: std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0),
        environment,
        decode_runs,
        decode_median_tok_s,
        prefill,
        prefill_tok_s,
    };
    let table = format!(
        "compare report\nmodel: {}\ngpu: {} (driver {})\nco-resident: {}\n\
         decode median: {:.1} tok/s ({} reps x {} steps)\n\
         prefill: {:.1} tok/s (p50 {:.1} ms over {} tokens, {} runs)\ngit: {}",
        report.model,
        report.environment.gpu,
        report.environment.driver,
        if report.environment.co_resident.is_empty() {
            "none".to_owned()
        } else {
            report.environment.co_resident.join("; ")
        },
        report.decode_median_tok_s,
        reps,
        decode_steps,
        report.prefill_tok_s,
        report.prefill.p50_ms,
        report.prefill.prompt_tokens,
        runs,
        report.environment.git_commit,
    );
    emit(format, &table, &report)
}

pub(crate) fn parity(
    model_path: &Path,
    tokens: &[u32],
    max_new: usize,
    eos: u32,
    format: ReportFormat,
) -> anyhow::Result<()> {
    if tokens.is_empty() {
        bail!("parity report requires at least one prompt token");
    }
    let mut cpu = load_runner(model_path, "cpu")?;
    let mut cuda = load_runner(model_path, "cuda").context(
        "failed to load cuda backend; build with `--features cuda` and run on CUDA host",
    )?;
    let cpu_tokens = cpu
        .generate(tokens, max_new, eos)
        .context("cpu generate failed")?;
    let cuda_tokens = cuda
        .generate(tokens, max_new, eos)
        .context("cuda generate failed")?;
    let matched_tokens = cpu_tokens
        .iter()
        .zip(cuda_tokens.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let report = ParityReport {
        report: "parity",
        prompt_tokens: tokens.len(),
        max_new,
        matched_tokens,
        exact_match: cpu_tokens == cuda_tokens,
        cpu_tokens,
        cuda_tokens,
    };
    emit(format, &parity_table(&report), &report)
}

pub(crate) fn salt(
    input_path: &Path,
    rows: usize,
    k: usize,
    budgets: &str,
    sensitivity: SaltSensitivityArg,
    format: ReportFormat,
) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(input_path)
        .with_context(|| format!("failed to read `{}`", input_path.display()))?;
    let weights: Vec<f32> = serde_json::from_str(&text).with_context(|| {
        format!(
            "failed to parse `{}` as a JSON array of f32 weights",
            input_path.display()
        )
    })?;
    if weights.len() != rows.saturating_mul(k) {
        bail!(
            "shape mismatch: rows*k = {} but input has {} weights",
            rows.saturating_mul(k),
            weights.len()
        );
    }
    let budget_values = parse_budgets(budgets)?;
    let mut reports = Vec::with_capacity(budget_values.len());
    for budget in budget_values {
        let cfg = QuantConfig {
            budget_bpw: budget,
            sensitivity: match sensitivity {
                SaltSensitivityArg::Uniform => Sensitivity::Uniform,
                SaltSensitivityArg::Energy => Sensitivity::Energy,
            },
            ..Default::default()
        };
        let qt = quantize_tensor(&weights, rows, k, &cfg).context("SALT quantize failed")?;
        let deq = dequantized_rows(&qt.salt_rows)?;
        reports.push(salt_budget_report(SaltBudgetInputs {
            requested_bpw: budget,
            plane_counts: &qt.plane_counts,
            rows,
            k,
            packed_weight_bpw: qt.packed_weight_bpw(),
            sidecar_bpw: qt.sidecar_bpw(),
            weights: &weights,
            deq: &deq,
        }));
    }
    let report = SaltReport {
        report: "salt",
        rows,
        k,
        sensitivity: match sensitivity {
            SaltSensitivityArg::Uniform => "uniform",
            SaltSensitivityArg::Energy => "energy",
        },
        budgets: reports,
    };
    emit(format, &salt_table(&report), &report)
}

#[derive(Debug, Serialize)]
struct SaltModelReport {
    report: &'static str,
    input: String,
    sensitivity: &'static str,
    scale_group: &'static str,
    tensors: usize,
    total_params: u64,
    budgets: Vec<SaltModelBudget>,
}

#[derive(Debug, Serialize)]
struct SaltModelBudget {
    requested_bpw: f64,
    /// Param-weighted average logical bpw the allocator actually realized.
    logical_bpw: f64,
    mse: f64,
    rmse: f64,
    mae: f64,
    max_abs: f64,
    frob_rel: f64,
    cosine: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    per_tensor: Vec<TensorReconRow>,
}

#[derive(Debug, Serialize)]
struct TensorReconRow {
    name: String,
    rows: usize,
    k: usize,
    logical_bpw: f64,
    frob_rel: f64,
    cosine: f64,
}

/// One tensor's contribution to the sweep, computed off-thread.
struct TensorContribution {
    params: u64,
    per_budget: Vec<BudgetContribution>,
}

/// One (tensor, budget) cell: the reconstruction moments, the realized bit cost, and the
/// per-tensor row (when `--per-tensor` is set).
struct BudgetContribution {
    accum: ReconAccum,
    bits: f64,
    row: Option<TensorReconRow>,
}

/// Quantize one 2D tensor at every budget and return its reconstruction contribution.
/// Pure + `Send` so it runs on the worker pool; reads the tensor's bytes from the shared
/// (mmap-backed) `SafeTensors` view.
fn tensor_contribution(
    st: &SafeTensors<'_>,
    name: &str,
    budgets: &[f64],
    sensitivity: &Sensitivity,
    scale_group: BaseScaleScope,
    per_tensor: bool,
) -> anyhow::Result<TensorContribution> {
    let shape = st.shape(name).expect("filtered to a 2D shape");
    let (rows, k) = (shape[0], shape[1]);
    let w = st
        .tensor_f32(name)
        .with_context(|| format!("read tensor `{name}`"))?;
    let mut per_budget = Vec::with_capacity(budgets.len());
    for &bpw in budgets {
        let cfg = QuantConfig {
            budget_bpw: bpw,
            t_min: 1,
            t_max: 3,
            sensitivity: sensitivity.clone(),
            scale_group,
        };
        let qt = quantize_tensor(&w, rows, k, &cfg)
            .with_context(|| format!("quantize `{name}` at {bpw} bpw"))?;
        let lbpw = qt.logical_bpw();
        // One dequant pass: fold into a local accumulator, derive the per-tensor stat from
        // it, then hand the moments back for the global merge.
        let mut accum = ReconAccum::default();
        accum
            .accumulate(&w, &qt)
            .with_context(|| format!("reconstruction of `{name}`"))?;
        let row = per_tensor.then(|| {
            let s = accum.finish();
            TensorReconRow {
                name: name.to_owned(),
                rows,
                k,
                logical_bpw: lbpw,
                frob_rel: s.frob_rel,
                cosine: s.cosine,
            }
        });
        per_budget.push(BudgetContribution {
            accum,
            bits: lbpw * (rows * k) as f64,
            row,
        });
    }
    Ok(TensorContribution {
        params: (rows * k) as u64,
        per_budget,
    })
}

/// SALT reconstruction-fidelity report over a real fp safetensors **master**: quantize
/// every 2D weight at each bpw budget and report whole-model (and optionally per-tensor)
/// error. The arch-agnostic proxy for output divergence — needs the fp (bf16/f16/f32)
/// master, NOT an already-quantized checkpoint (those carry no fp reference to measure
/// against and won't even parse as fp here).
pub(crate) fn salt_model(
    input: &Path,
    budgets: &str,
    sensitivity: SaltSensitivityArg,
    scale_group: ScaleGroupArg,
    limit: usize,
    per_tensor: bool,
    format: ReportFormat,
) -> anyhow::Result<()> {
    let budget_values = parse_budgets(budgets)?;
    let sens = match sensitivity {
        SaltSensitivityArg::Uniform => Sensitivity::Uniform,
        SaltSensitivityArg::Energy => Sensitivity::Energy,
    };
    let sg = match scale_group {
        ScaleGroupArg::Block => BaseScaleScope::Block,
        ScaleGroupArg::Tensor => BaseScaleScope::Tensor,
    };

    let shards = resolve_shards(input)?;

    // Bounded worker pool: parallelism speeds a multi-GB sweep ~linearly, but each worker
    // holds one tensor's f32 widening (the embeddings are multi-GB), so cap threads to
    // keep peak RSS sane rather than taking every core.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(12);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("build worker pool")?;

    // One accumulator per budget; read each tensor once (mmap-paged) and fold it into
    // every budget. Shards stream one at a time so a 50GB+ master never lands in RAM.
    let nb = budget_values.len();
    let mut accums = vec![ReconAccum::default(); nb];
    let mut bits = vec![0.0f64; nb]; // Σ logical_bpw·params, for the param-weighted avg
    let mut per_tensor_rows: Vec<Vec<TensorReconRow>> = if per_tensor {
        (0..nb).map(|_| Vec::new()).collect()
    } else {
        Vec::new()
    };
    let mut total_params = 0u64;
    let mut tensors_done = 0usize;

    'shards: for shard in &shards {
        let file = std::fs::File::open(shard)
            .with_context(|| format!("open shard `{}`", shard.display()))?;
        // SAFETY: the model file is read-only input for the lifetime of this report — we
        // never write through the map, and the standard mmap-the-weights contract assumes
        // nothing truncates it mid-run (a concurrent external truncation could fault).
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mmap shard `{}`", shard.display()))?;
        let st = SafeTensors::parse(&mmap)
            .with_context(|| format!("parse shard `{}` as safetensors", shard.display()))?;

        // Quantize every 2D weight matrix; skip 1D tensors (norms/biases) and degenerate
        // shapes — mirrors `tritium quantize`.
        let names: Vec<String> = st
            .names()
            .filter(|n| {
                st.shape(n)
                    .is_some_and(|s| s.len() == 2 && s[0] >= 2 && s[1] >= 2)
            })
            .map(str::to_owned)
            .collect();

        // How many tensors from this shard to process under --limit.
        let take = if limit > 0 {
            limit.saturating_sub(tensors_done).min(names.len())
        } else {
            names.len()
        };
        if take == 0 {
            break 'shards;
        }

        // Quantize this shard's (independent) 2D tensors in parallel. `par_iter().collect()`
        // preserves input order, so the per-tensor output stays deterministic; the global
        // accumulators are then folded sequentially in that order.
        let contribs: Vec<TensorContribution> = pool.install(|| {
            names[..take]
                .par_iter()
                .map(|name| tensor_contribution(&st, name, &budget_values, &sens, sg, per_tensor))
                .collect::<anyhow::Result<Vec<_>>>()
        })?;

        for c in contribs {
            total_params += c.params;
            tensors_done += 1;
            for (bi, bc) in c.per_budget.into_iter().enumerate() {
                accums[bi].merge(&bc.accum);
                bits[bi] += bc.bits;
                if let Some(row) = bc.row {
                    per_tensor_rows[bi].push(row);
                }
            }
        }
    }

    if tensors_done == 0 {
        bail!("no 2D weight tensors found in `{}`", input.display());
    }

    let reports = budget_values
        .iter()
        .enumerate()
        .map(|(bi, &bpw)| {
            let s = accums[bi].finish();
            SaltModelBudget {
                requested_bpw: bpw,
                logical_bpw: if total_params > 0 {
                    bits[bi] / total_params as f64
                } else {
                    0.0
                },
                mse: s.mse,
                rmse: s.rmse,
                mae: s.mae,
                max_abs: s.max_abs,
                frob_rel: s.frob_rel,
                cosine: s.cosine,
                per_tensor: if per_tensor {
                    std::mem::take(&mut per_tensor_rows[bi])
                } else {
                    Vec::new()
                },
            }
        })
        .collect();

    let report = SaltModelReport {
        report: "salt-model",
        input: input.display().to_string(),
        sensitivity: match sensitivity {
            SaltSensitivityArg::Uniform => "uniform",
            SaltSensitivityArg::Energy => "energy",
        },
        scale_group: match scale_group {
            ScaleGroupArg::Block => "block",
            ScaleGroupArg::Tensor => "tensor",
        },
        tensors: tensors_done,
        total_params,
        budgets: reports,
    };
    emit(format, &salt_model_table(&report), &report)
}

/// Resolve a model path to its safetensors shard files, in deterministic order. Accepts a
/// single `.safetensors` file, a `*.index.json` (sharded model — reads its `weight_map`),
/// or a directory. Directory resolution is ordered to avoid pulling in non-model weight
/// files (e.g. `adapter_model.safetensors`): (1) `model.safetensors.index.json` if present,
/// (2) a lone canonical `model.safetensors`, (3) otherwise every `*.safetensors` in the
/// directory, sorted (an unindexed multi-shard layout).
fn resolve_shards(input: &Path) -> anyhow::Result<Vec<PathBuf>> {
    if input.is_dir() {
        let idx = input.join("model.safetensors.index.json");
        if idx.exists() {
            return shards_from_index(&idx);
        }
        let canonical = input.join("model.safetensors");
        if canonical.is_file() {
            return Ok(vec![canonical]);
        }
        let mut v: Vec<PathBuf> = std::fs::read_dir(input)
            .with_context(|| format!("read dir `{}`", input.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        v.sort();
        if v.is_empty() {
            bail!("no `.safetensors` files in `{}`", input.display());
        }
        return Ok(v);
    }
    if input.extension().is_some_and(|x| x == "json") {
        return shards_from_index(input);
    }
    Ok(vec![input.to_path_buf()])
}

/// Read a HF `*.safetensors.index.json` and return the unique shard files it references,
/// resolved relative to the index's directory, sorted.
fn shards_from_index(idx: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let txt =
        std::fs::read_to_string(idx).with_context(|| format!("read index `{}`", idx.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&txt).with_context(|| format!("parse index `{}`", idx.display()))?;
    let wm = json
        .get("weight_map")
        .and_then(|v| v.as_object())
        .with_context(|| format!("`{}` has no object `weight_map`", idx.display()))?;
    let dir = idx.parent().unwrap_or_else(|| Path::new("."));
    let mut set = std::collections::BTreeSet::new();
    for v in wm.values() {
        if let Some(file) = v.as_str() {
            set.insert(dir.join(file));
        }
    }
    if set.is_empty() {
        bail!("index `{}` lists no shards", idx.display());
    }
    Ok(set.into_iter().collect())
}

fn salt_model_table(r: &SaltModelReport) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = write!(
        s,
        "salt-model report\ninput: {}\nsensitivity: {}\nscale_group: {}\ntensors: {}\ntotal_params: {:.3}M\n\n",
        r.input,
        r.sensitivity,
        r.scale_group,
        r.tensors,
        r.total_params as f64 / 1e6,
    );
    let _ = writeln!(
        s,
        "  {:>10} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "req_bpw", "log_bpw", "mse", "frob_rel", "cosine", "max_abs"
    );
    for b in &r.budgets {
        let _ = writeln!(
            s,
            "  {:>10.4} {:>10.4} {:>12.3e} {:>12.6} {:>12.8} {:>10.4}",
            b.requested_bpw, b.logical_bpw, b.mse, b.frob_rel, b.cosine, b.max_abs
        );
    }
    // Per-tensor breakdown (only present when requested), grouped by budget.
    for b in &r.budgets {
        if b.per_tensor.is_empty() {
            continue;
        }
        let _ = write!(s, "\n  per-tensor @ {:.4} bpw:\n", b.requested_bpw);
        let _ = writeln!(
            s,
            "    {:>10} {:>8} {:>12} {:>12}  name",
            "log_bpw", "k", "frob_rel", "cosine"
        );
        for t in &b.per_tensor {
            let _ = writeln!(
                s,
                "    {:>10.4} {:>8} {:>12.6} {:>12.8}  {}",
                t.logical_bpw, t.k, t.frob_rel, t.cosine, t.name
            );
        }
    }
    s.push('\n');
    s
}

fn load_runner(model_path: &Path, backend: &str) -> anyhow::Result<ModelRunner> {
    load_runner_with_caps(model_path, backend).map(|(runner, _)| runner)
}

/// As [`load_runner`], but also returns the backend's [`DeviceCaps`] — read off the
/// live object before it is moved into the runner, which is the only point it is
/// still reachable. `None` for the CPU path (`load_cpu` owns its backend privately,
/// and a CPU run has no GPU to identify anyway).
fn load_runner_with_caps(
    model_path: &Path,
    backend: &str,
) -> anyhow::Result<(ModelRunner, Option<DeviceCaps>)> {
    let bytes = std::fs::read(model_path)
        .with_context(|| format!("failed to read model `{}`", model_path.display()))?;
    if backend == "cpu" {
        let runner = ModelRunner::load_cpu(&bytes)
            .with_context(|| format!("failed to load model `{}` on cpu", model_path.display()))?;
        return Ok((runner, None));
    }

    let init = tritium_runtime::BACKENDS
        .iter()
        .find(|entry| entry.name == backend)
        .map(|entry| entry.init)
        .with_context(|| format!("backend `{backend}` is not registered"))?;
    let backend_obj = init().with_context(|| format!("backend `{backend}` failed to init"))?;
    let caps = backend_obj.capabilities();
    let file = tritium_format::read_gguf(&bytes)
        .with_context(|| format!("failed to parse model `{}`", model_path.display()))?;
    let runner = ModelRunner::load(&file, &bytes, backend_obj).with_context(|| {
        format!(
            "failed to load model `{}` on {backend}",
            model_path.display()
        )
    })?;
    Ok((runner, Some(caps)))
}

fn dequantized_rows(rows: &[tritium_format::SaltRow]) -> anyhow::Result<Vec<f32>> {
    let mut out = Vec::new();
    for row in rows {
        out.extend(dequant_salt_row(row).context("SALT dequant failed")?);
    }
    Ok(out)
}

struct SaltBudgetInputs<'a> {
    requested_bpw: f64,
    plane_counts: &'a [usize],
    rows: usize,
    k: usize,
    packed_weight_bpw: f64,
    sidecar_bpw: f64,
    weights: &'a [f32],
    deq: &'a [f32],
}

fn salt_budget_report(inputs: SaltBudgetInputs<'_>) -> SaltBudgetReport {
    let SaltBudgetInputs {
        requested_bpw,
        plane_counts,
        rows,
        k,
        packed_weight_bpw,
        sidecar_bpw,
        weights,
        deq,
    } = inputs;
    let mut squared = 0.0f64;
    let mut max_abs_error = 0.0f64;
    for (&w, &q) in weights.iter().zip(deq.iter()) {
        let err = (w - q) as f64;
        squared += err * err;
        max_abs_error = max_abs_error.max(err.abs());
    }
    let mse = if weights.is_empty() {
        0.0
    } else {
        squared / weights.len() as f64
    };
    let max_planes = plane_counts.iter().copied().max().unwrap_or(0);
    let logical_bpw = logical_bpw(plane_counts, rows, k);
    SaltBudgetReport {
        requested_bpw,
        logical_bpw,
        // Preserve the 1.0 report schema's historical metric. Exact physical
        // rates are exposed separately instead of silently changing this key.
        dense_stored_bpw: max_planes as f64 * tritium_quantize::TRIT_BITS,
        packed_weight_bpw,
        sidecar_bpw,
        mse,
        rmse: mse.sqrt(),
        max_abs_error,
        plane_histogram: plane_histogram(plane_counts),
    }
}

fn logical_bpw(plane_counts: &[usize], rows: usize, k: usize) -> f64 {
    if rows == 0 || k == 0 {
        return 0.0;
    }
    let blocks = k.div_ceil(tritium_format::QK_K);
    let mut bits = 0.0;
    for (flat_idx, &planes) in plane_counts.iter().enumerate() {
        let block = flat_idx % blocks;
        let start = block * tritium_format::QK_K;
        let size = (start + tritium_format::QK_K).min(k) - start;
        bits += size as f64 * planes as f64 * tritium_quantize::TRIT_BITS;
    }
    bits / (rows * k) as f64
}

fn plane_histogram(plane_counts: &[usize]) -> Vec<PlaneCount> {
    let max_planes = plane_counts.iter().copied().max().unwrap_or(0);
    let mut hist = vec![0usize; max_planes + 1];
    for &planes in plane_counts {
        hist[planes] += 1;
    }
    hist.into_iter()
        .enumerate()
        .filter(|(_, groups)| *groups > 0)
        .map(|(planes, groups)| PlaneCount { planes, groups })
        .collect()
}

fn parse_budgets(raw: &str) -> anyhow::Result<Vec<f64>> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: f64 = trimmed
            .parse()
            .with_context(|| format!("invalid bpw budget `{trimmed}`"))?;
        if !value.is_finite() || value < tritium_quantize::TRIT_BITS {
            bail!(
                "invalid bpw budget `{trimmed}`: must be finite and >= {:.6}",
                tritium_quantize::TRIT_BITS
            );
        }
        out.push(value);
    }
    if out.is_empty() {
        bail!("at least one bpw budget is required");
    }
    Ok(out)
}

fn emit<T: Serialize>(format: ReportFormat, table: &str, report: &T) -> anyhow::Result<()> {
    match format {
        ReportFormat::Both => {
            print!("{table}");
            println!("{}", serde_json::to_string_pretty(report)?);
        }
        ReportFormat::Json => println!("{}", serde_json::to_string_pretty(report)?),
        ReportFormat::Table => print!("{table}"),
    }
    Ok(())
}

fn decode_table(r: &DecodeReport) -> String {
    format!(
        "decode report\nbackend: {}\nprompt_tokens: {}\ndecode_steps: {}\nwarmup_steps: {}\nelapsed_ms: {:.3}\ntokens_per_sec: {:.3}\nms_per_token: {:.3}\nroofline_4090_pct: {:.3}\nbaseline_4090_drop_pct: {:.3}\n\n",
        r.backend,
        r.prompt_tokens,
        r.decode_steps,
        r.warmup_steps,
        r.elapsed_ms,
        r.tokens_per_sec,
        r.ms_per_token,
        r.roofline_4090_pct,
        r.baseline_4090_drop_pct,
    )
}

fn ttft_table(r: &TtftReport) -> String {
    format!(
        "ttft report\nbackend: {}\nprompt_tokens: {}\nruns: {}\ntotal_ms: {:.3}\np50_ms: {:.3}\np95_ms: {:.3}\ntokens_per_sec: {:.3}\n\n",
        r.backend, r.prompt_tokens, r.runs, r.total_ms, r.p50_ms, r.p95_ms, r.tokens_per_sec,
    )
}

fn parity_table(r: &ParityReport) -> String {
    format!(
        "parity report\nprompt_tokens: {}\nmax_new: {}\nmatched_tokens: {}\nexact_match: {}\ncpu_tokens: {:?}\ncuda_tokens: {:?}\n\n",
        r.prompt_tokens, r.max_new, r.matched_tokens, r.exact_match, r.cpu_tokens, r.cuda_tokens,
    )
}

fn salt_table(r: &SaltReport) -> String {
    let mut out = format!(
        "salt report\nrows: {}\nk: {}\nsensitivity: {}\n\nbudget_bpw logical_bpw dense_bpw mse rmse max_abs_error planes packed_bpw sidecar_bpw\n",
        r.rows, r.k, r.sensitivity
    );
    for b in &r.budgets {
        let hist = b
            .plane_histogram
            .iter()
            .map(|p| format!("{}:{}", p.planes, p.groups))
            .collect::<Vec<_>>()
            .join(",");
        out.push_str(&format!(
            "{:.6} {:.6} {:.6} {:.6e} {:.6e} {:.6e} {} {:.6} {:.6}\n",
            b.requested_bpw,
            b.logical_bpw,
            b.dense_stored_bpw,
            b.mse,
            b.rmse,
            b.max_abs_error,
            hist,
            b.packed_weight_bpw,
            b.sidecar_bpw,
        ));
    }
    out.push('\n');
    out
}

fn percentile_sorted(samples: &[f64], p: f64) -> f64 {
    assert!(
        !samples.is_empty(),
        "percentile requires at least one sample"
    );
    let idx = ((samples.len() as f64 * p).ceil() as usize).saturating_sub(1);
    samples[idx.min(samples.len() - 1)]
}

/// `tritium report sparsity` — the Track-A ternary census: per-tensor
/// element-zero fraction, all-zero-256-block fraction, and the model-level
/// entropy/format math that decides which memory-access strategy pays
/// (block-skip vs dense entropy packing vs bitmap+signs).
pub(crate) fn sparsity(model: &Path) -> anyhow::Result<()> {
    use tritium_core::Trit;
    use tritium_format::{
        GGML_TYPE_I2_S, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0, TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES,
        unpack_i2s_tensor, unpack_tq1_0_row, unpack_tq2_0_row,
    };
    let bytes = std::fs::read(model).with_context(|| format!("read {}", model.display()))?;
    let file = tritium_format::read_gguf(&bytes).context("parse gguf")?;
    let data0 = file.tensor_data_offset as usize;

    struct Row {
        name: String,
        n: u64,
        zeros: u64,
        zero_blocks: u64,
        blocks: u64,
    }
    let ternary: Vec<_> = file
        .tensors
        .iter()
        .filter(|t| {
            matches!(
                t.ggml_type,
                GGML_TYPE_I2_S | GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0
            )
        })
        .collect();
    if ternary.is_empty() {
        bail!(
            "no ternary tensors (I2_S/TQ1_0/TQ2_0) in {}",
            model.display()
        );
    }

    let rows: Vec<Row> = ternary
        .par_iter()
        .map(|info| -> anyhow::Result<Row> {
            let k = info.dims.first().copied().unwrap_or(0) as usize;
            let n_el: usize = info.dims.iter().product::<u64>() as usize;
            let n_rows = n_el.checked_div(k).unwrap_or(0);
            let start = data0 + info.offset as usize;
            // I2_S is a bitnet.cpp extension the generic reader sizes as 0 —
            // compute its payload length (packed 2-bit body + f32 scale) the
            // way the model loader does; TQ1/TQ2 are known types.
            let len = if info.ggml_type == GGML_TYPE_I2_S {
                n_el / 4 + tritium_format::I2S_SCALE_BYTES
            } else {
                info.n_bytes as usize
            };
            let payload = bytes
                .get(start..start + len)
                .context("tensor payload out of bounds")?;
            let mut trits = vec![Trit::ZERO; n_el];
            match info.ggml_type {
                GGML_TYPE_I2_S => {
                    unpack_i2s_tensor(payload, n_el, &mut trits)
                        .map_err(|e| anyhow::anyhow!("{}: {e}", info.name))?;
                }
                t @ (GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0) => {
                    let nb = k.div_ceil(256);
                    let row_bytes = nb
                        * if t == GGML_TYPE_TQ1_0 {
                            TQ1_0_BLOCK_BYTES
                        } else {
                            TQ2_0_BLOCK_BYTES
                        };
                    let mut scales = vec![half::f16::ZERO; nb];
                    if n_rows * row_bytes > payload.len() {
                        anyhow::bail!(
                            "{}: payload {} B < {} rows x {} B (k % 256 != 0?)",
                            info.name,
                            payload.len(),
                            n_rows,
                            row_bytes
                        );
                    }
                    for r in 0..n_rows {
                        let row = &payload[r * row_bytes..(r + 1) * row_bytes];
                        let out = &mut trits[r * k..(r + 1) * k];
                        if t == GGML_TYPE_TQ1_0 {
                            unpack_tq1_0_row(row, out, &mut scales)
                        } else {
                            unpack_tq2_0_row(row, out, &mut scales)
                        }
                        .map_err(|e| anyhow::anyhow!("{}: {e}", info.name))?;
                    }
                }
                _ => unreachable!(),
            }
            let zeros = trits.iter().filter(|t| t.get() == 0).count() as u64;
            // Block stats on the SAME geometry the skip kernels use: 256-trit
            // blocks along k, per row.
            let (mut zero_blocks, mut blocks) = (0u64, 0u64);
            for r in 0..n_rows {
                for c in trits[r * k..(r + 1) * k].chunks(256) {
                    blocks += 1;
                    if c.iter().all(|t| t.get() == 0) {
                        zero_blocks += 1;
                    }
                }
            }
            Ok(Row {
                name: info.name.clone(),
                n: n_el as u64,
                zeros,
                zero_blocks,
                blocks,
            })
        })
        .collect::<anyhow::Result<_>>()?;

    let (tot, tot_z, tot_zb, tot_b) = rows.iter().fold((0u64, 0u64, 0u64, 0u64), |a, r| {
        (
            a.0 + r.n,
            a.1 + r.zeros,
            a.2 + r.zero_blocks,
            a.3 + r.blocks,
        )
    });
    let p = tot_z as f64 / tot as f64;
    let pnz = (1.0 - p) / 2.0;
    // x·log2(x) with the 0·log2(0)=0 convention — keeps the floor total at
    // degenerate p (synthetic all-zero / no-zero fixtures).
    let xlx = |x: f64| if x > 0.0 { x * x.log2() } else { 0.0 };
    let entropy = -(xlx(p) + 2.0 * xlx(pnz));

    println!("ternary sparsity census — {}", model.display());
    println!(
        "{:<28} {:>12} {:>8} {:>10}",
        "tensor", "elements", "zero%", "0-block%"
    );
    for r in &rows {
        println!(
            "{:<28} {:>12} {:>7.1}% {:>9.3}%",
            r.name,
            r.n,
            r.zeros as f64 / r.n as f64 * 100.0,
            r.zero_blocks as f64 / r.blocks.max(1) as f64 * 100.0,
        );
    }
    println!(
        "\nTOTAL: {tot} weights | element zeros {:.2}% | all-zero 256-blocks {:.3}% ({tot_zb}/{tot_b})",
        p * 100.0,
        tot_zb as f64 / tot_b.max(1) as f64 * 100.0,
    );
    println!("format math at this sparsity (bits/weight, lower = less weight traffic):");
    println!("  entropy floor        {entropy:.3}");
    println!("  TQ2_0 (current)      2.000  (+ scales)");
    // TQ1_0's REAL payload rate: 48B qs @5 trits/byte + 4B qh @4 trits/byte
    // per 256-trit block = 1.625 b/w (1.6875 stored, with the f16 scale).
    println!(
        "  TQ1_0 (dense pack)   1.625  ({:.0}% of floor; 1.688 with scales)",
        1.625 / entropy * 100.0
    );
    println!(
        "  bitmap+signs         {:.3}  (1 + (1-p); sparsity-adaptive)",
        1.0 + (1.0 - p)
    );
    println!(
        "  block-skip savings   {:.3}%  (all-zero-block fraction — what the _sparse kernels skip)",
        tot_zb as f64 / tot_b.max(1) as f64 * 100.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_parse_csv() {
        let got = parse_budgets("1.585, 2.0").expect("parse budgets");
        assert_eq!(got.len(), 2);
        assert!((got[1] - 2.0).abs() < f64::EPSILON);
    }

    /// On an NVIDIA box `nvidia-smi` answers and its string is what the existing
    /// ledger entries were recorded with — the backend's own name must NOT displace
    /// it, or every historical row silently changes basis.
    #[test]
    fn gpu_field_prefers_the_vendor_tool_when_it_answers() {
        let caps = DeviceCaps::new("wgpu", "NVIDIA GeForce RTX 4090 (DiscreteGpu)");
        let got = resolve_gpu("NVIDIA GeForce RTX 4090", Some(&caps));
        assert_eq!(got, "NVIDIA GeForce RTX 4090");
    }

    /// The issue-#4 case: on an AMD/Vulkan box `nvidia-smi` is absent, so every env
    /// field reads "unavailable" and the emitted JSON identifies no GPU at all — not
    /// a receipt. The backend already knows its adapter name; use it.
    #[test]
    fn gpu_field_falls_back_to_the_backend_adapter_name_off_nvidia() {
        let caps = DeviceCaps::new("wgpu", "AMD Radeon RX 9070 XT (RADV GFX1201)");
        let got = resolve_gpu("unavailable", Some(&caps));
        assert_eq!(got, "AMD Radeon RX 9070 XT (RADV GFX1201) [wgpu]");
    }

    /// No vendor tool and no device-backed run (e.g. `--backend cpu`) must stay
    /// "unavailable" rather than inventing a name.
    #[test]
    fn gpu_field_stays_unavailable_with_neither_source() {
        assert_eq!(resolve_gpu("unavailable", None), "unavailable");
        // An empty `nvidia-smi` line is also "no answer", not a valid GPU name.
        assert_eq!(resolve_gpu("", None), "unavailable");
    }

    #[test]
    fn budgets_reject_below_floor() {
        let err = parse_budgets("1.0").expect_err("below floor must error");
        assert!(format!("{err:#}").contains(">="));
    }

    #[test]
    fn salt_budget_metrics_are_finite() {
        let weights = vec![0.0, 1.0, -1.0, 0.5];
        let deq = vec![0.0, 0.75, -1.0, 0.25];
        let report = salt_budget_report(SaltBudgetInputs {
            requested_bpw: 2.0,
            plane_counts: &[1],
            rows: 1,
            k: 4,
            packed_weight_bpw: 2.0625,
            sidecar_bpw: 2.375,
            weights: &weights,
            deq: &deq,
        });
        assert_eq!(report.dense_stored_bpw, tritium_quantize::TRIT_BITS);
        assert_eq!(report.packed_weight_bpw, 2.0625);
        assert_eq!(report.sidecar_bpw, 2.375);
        assert!(report.mse > 0.0);
        assert_eq!(report.plane_histogram[0].planes, 1);
        assert_eq!(report.plane_histogram[0].groups, 1);
    }

    #[test]
    fn decode_table_contains_key_metrics() {
        let report = DecodeReport {
            report: "decode",
            backend: "cpu".to_owned(),
            prompt_tokens: 3,
            decode_steps: 2,
            warmup_steps: 1,
            elapsed_ms: 10.0,
            tokens_per_sec: 200.0,
            ms_per_token: 5.0,
            roofline_4090_pct: 25.0,
            baseline_4090_drop_pct: 0.0,
        };
        let table = decode_table(&report);
        assert!(table.contains("tokens_per_sec"));
        assert!(table.contains("roofline_4090_pct"));
    }
}
