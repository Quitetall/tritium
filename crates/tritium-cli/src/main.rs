//! The `tritium` command-line tool.
//!
//! Three subcommands:
//! - `tritium inspect <PATH.gguf>` — parse a GGUF container and print a summary of
//!   its version, metadata, architecture, alignment, and tensor table.
//! - `tritium list-backends` — enumerate every backend the runtime discovered.
//! - `tritium generate` — load a GGUF model and greedily decode tokens from a
//!   reproducible JSON file of input token IDs.
//!
//! Errors are surfaced through [`anyhow`]: a missing, short, or corrupt GGUF file
//! prints a clean message and exits non-zero rather than panicking.

// `tritium_cpu` is otherwise unused by name here; the `as _` import forces it to be
// linked so its `linkme` self-registration into `tritium_runtime::BACKENDS` is
// present and `list-backends` can see the `cpu` backend.
use tritium_cpu as _;
#[cfg(feature = "cuda")]
use tritium_cuda as _;
// Same force-link for the non-NVIDIA backends: `list-backends` and
// `report --backend {wgpu,rocm}` resolve names out of the linkme-populated runtime
// registry, and an entry only exists if the backend crate is linked into this
// binary. Without these two lines an AMD box reports only `cpu` however the GPU
// crates were built (issue #4).
#[cfg(feature = "metal")]
use tritium_metal as _;
#[cfg(feature = "rocm")]
use tritium_rocm as _;
#[cfg(feature = "wgpu")]
use tritium_wgpu as _;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

mod backends;
mod campaign;
#[cfg(feature = "cuda")]
mod campaign_artifact;
#[cfg(feature = "nccl")]
mod campaign_world;
mod generate;
#[cfg(feature = "cuda")]
mod hestia_gate;
mod inspect;
#[cfg(feature = "cuda")]
mod nvml_probe;
mod pull;
mod quantize;
mod release;
mod repack;
mod report;
mod salt;
mod stage7_evidence;

/// BitNet 2B4T uses the LLaMA-3 tokenizer, whose end-of-text token is `128001`.
/// Used as the default stop token for `generate` when `--eos` is not given.
const DEFAULT_EOS: u32 = 128_001;

/// `tritium`: inspect ternary GGUF models and list available compute backends.
#[derive(Parser, Debug)]
#[command(
    name = "tritium",
    about = "Inspect ternary GGUF models and list available compute backends.",
    version
)]
struct Cli {
    /// The subcommand to run.
    #[command(subcommand)]
    command: Command,
}

/// The `tritium` subcommands.
#[derive(Subcommand, Debug)]
enum Command {
    /// Parse a GGUF file and print its version, metadata, and tensor table.
    Inspect {
        /// Path to the `.gguf` file to inspect.
        path: PathBuf,
    },
    /// List every backend the runtime discovered, with its capabilities.
    ListBackends,
    /// Build offline teacher targets or run a resumable packed-SALT campaign.
    Campaign {
        /// Campaign operation.
        #[command(subcommand)]
        campaign: campaign::CampaignCommand,
    },
    /// Admit sources and run resumable SALT V2 synthesis workflows.
    Salt {
        /// SALT V2 operation.
        #[command(subcommand)]
        salt: salt::SaltCommand,
    },
    /// Download a GGUF model from the HuggingFace hub into the local cache
    /// (~/.cache/tritium-models, override with TRITIUM_MODEL_CACHE). Resumes
    /// partial downloads on re-run.
    Pull {
        /// Hub repo, `owner/name` (e.g. `microsoft/bitnet-b1.58-2B-4T-gguf`).
        repo: String,
        /// Which file to pull when the repo holds several .gguf files.
        #[arg(long)]
        file: Option<String>,
        /// Git revision (branch, tag or commit) to pull from.
        #[arg(long, default_value = "main")]
        revision: String,
    },
    /// Load a GGUF model and greedily generate tokens from a JSON file of input IDs.
    Generate {
        /// Path to the `.gguf` model file.
        #[arg(long)]
        model: PathBuf,
        /// Path to a JSON file holding the input token IDs, e.g. `[1, 128000, 9906]`.
        #[arg(long)]
        tokens: PathBuf,
        /// Maximum number of new tokens to generate.
        #[arg(long, default_value_t = 16)]
        max_new: usize,
        /// Decode greedily (the only v0.20 strategy). `--greedy=false` still decodes
        /// greedily but prints a note; sampling lands in a later wave.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        greedy: bool,
        /// End-of-sequence token ID that stops generation early.
        #[arg(long, default_value_t = DEFAULT_EOS)]
        eos: u32,
    },
    /// Emit reproducible benchmark/validation reports.
    Report {
        /// The report to run.
        #[command(subcommand)]
        report: ReportCommand,
    },
    /// Inspect and verify immutable release-candidate inputs.
    Release {
        /// Release evidence operation.
        #[command(subcommand)]
        release: release::ReleaseCommand,
    },
    /// Repack ternary GGUF tensors while preserving dequantized weight values.
    Repack {
        /// Path to source `.gguf` (I2_S, standard Q2_0, TQ1_0 or TQ2_0 tensors).
        #[arg(long)]
        input: PathBuf,
        /// Path to write the repacked `.gguf`.
        #[arg(long)]
        output: PathBuf,
        /// Target ternary format.
        #[arg(long, value_enum)]
        to: repack::RepackTarget,
    },
    /// SALT-quantize an fp safetensors model to a SALT bundle.
    Quantize {
        /// Path to the fp16/bf16/f32 `.safetensors` source model.
        #[arg(long)]
        input: PathBuf,
        /// Path to write the SALT bundle (`.tslb`).
        #[arg(long)]
        output: PathBuf,
        /// Target average bits-per-weight (`1.585` = all base ternary … `~4.75` at T=3).
        #[arg(long, default_value_t = 2.0)]
        bpw: f64,
        /// Base-plane scale granularity: `block` (per-256-block) or `tensor` (per-tensor,
        /// for a BitNet b1.58 master).
        #[arg(long, value_enum, default_value_t = quantize::ScaleGroupArg::Block)]
        scale_group: quantize::ScaleGroupArg,
        /// Plane-allocation sensitivity: `uniform` (allocate purely by reconstruction-error
        /// reduction) or `energy` (weight `‖w‖²` proxy — spend planes on high-energy groups).
        #[arg(long, value_enum, default_value_t = SaltSensitivityArg::Uniform)]
        sensitivity: SaltSensitivityArg,
        /// Optional diagonal-Fisher sensitivity sidecar: a `.safetensors` file mapping each
        /// weight-tensor name to its per-weight Fisher `E[(∂L/∂w)²]` (same shape as the weight).
        /// When set it OVERRIDES `--sensitivity`, allocating planes by loss curvature per tile
        /// (plan 0039) — spend bits where the loss is sensitive, not merely where magnitude is.
        #[arg(long)]
        fisher: Option<PathBuf>,
        /// Output container: `sidecar` (single-file `.tslb` bundle) or `gguf`
        /// (GGUF container holding the SALT rows).
        #[arg(long, value_enum, default_value_t = quantize::OutputFormat::Sidecar)]
        format: quantize::OutputFormat,
    },
}

/// Benchmark/validation reports.
#[derive(Subcommand, Debug)]
enum ReportCommand {
    /// Ternary weight sparsity census: element-zero %, all-zero-block %,
    /// entropy bits/weight and projected traffic savings per tensor (Track A
    /// ground truth — run per model, per checkpoint).
    Sparsity {
        /// Path to the `.gguf` model file (I2_S / Q2_0 / TQ1_0 / TQ2_0 tensors).
        #[arg(long)]
        model: PathBuf,
    },
    /// Decode-only throughput after prefill.
    Decode {
        /// Path to the `.gguf` model file.
        #[arg(long)]
        model: PathBuf,
        /// Path to a JSON file holding the input token IDs.
        #[arg(long)]
        tokens: PathBuf,
        /// Backend name from the runtime registry.
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Timed single-token decode steps.
        #[arg(long, default_value_t = 8)]
        decode_steps: usize,
        /// Untimed decode warmup steps after prefill.
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        /// Output format.
        #[arg(long, value_enum, default_value_t = ReportFormat::Both)]
        format: ReportFormat,
    },
    /// Time to first token / prefill latency.
    Ttft {
        /// Path to the `.gguf` model file.
        #[arg(long)]
        model: PathBuf,
        /// Path to a JSON file holding the input token IDs.
        #[arg(long)]
        tokens: PathBuf,
        /// Backend name from the runtime registry.
        #[arg(long, default_value = "cpu")]
        backend: String,
        /// Number of full prefill runs.
        #[arg(long, default_value_t = 1)]
        runs: usize,
        /// Output format.
        #[arg(long, value_enum, default_value_t = ReportFormat::Both)]
        format: ReportFormat,
    },
    /// One-command benchmark bundle for the public ledger (ADR 0026 Track R):
    /// decode ×N (order-stable) + prefill/ttft, plus environment capture
    /// (GPU, driver, VRAM co-residency) — the JSON that docs/BENCHMARKS.md
    /// numbers must reproduce from.
    Compare {
        /// Path to the `.gguf` model file.
        #[arg(long)]
        model: PathBuf,
        /// Path to a JSON file holding the input token IDs.
        #[arg(long)]
        tokens: PathBuf,
        /// Backend name from the runtime registry.
        #[arg(long, default_value = "cuda")]
        backend: String,
        /// Cycle/truncate the token file to exactly this prompt length
        /// (512 = the pp512 ledger shape). 0 = use the file as-is.
        #[arg(long, default_value_t = 512)]
        prompt_len: usize,
        /// Timed decode steps per decode repetition.
        #[arg(long, default_value_t = 256)]
        decode_steps: usize,
        /// Untimed decode warmup steps after prefill.
        #[arg(long, default_value_t = 16)]
        warmup: usize,
        /// Decode repetitions (median reported).
        #[arg(long, default_value_t = 3)]
        reps: usize,
        /// Prefill runs for the ttft p50.
        #[arg(long, default_value_t = 5)]
        runs: usize,
        /// Output format.
        #[arg(long, value_enum, default_value_t = ReportFormat::Both)]
        format: ReportFormat,
    },
    /// CPU-vs-CUDA greedy parity.
    Parity {
        /// Path to the `.gguf` model file.
        #[arg(long)]
        model: PathBuf,
        /// Path to a JSON file holding the input token IDs.
        #[arg(long)]
        tokens: PathBuf,
        /// Maximum generated tokens to compare.
        #[arg(long, default_value_t = 16)]
        max_new: usize,
        /// End-of-sequence token ID.
        #[arg(long, default_value_t = DEFAULT_EOS)]
        eos: u32,
        /// Output format.
        #[arg(long, value_enum, default_value_t = ReportFormat::Both)]
        format: ReportFormat,
    },
    /// SALT bpw/error report for a flat JSON fp32 matrix.
    Salt {
        /// Path to a JSON array of row-major fp32 weights.
        #[arg(long)]
        input: PathBuf,
        /// Number of matrix rows.
        #[arg(long)]
        rows: usize,
        /// Input features per row.
        #[arg(long)]
        k: usize,
        /// Comma-separated bpw budgets, e.g. `1.585,2.0,2.5`.
        #[arg(long)]
        budgets: String,
        /// Sensitivity proxy.
        #[arg(long, value_enum, default_value_t = SaltSensitivityArg::Uniform)]
        sensitivity: SaltSensitivityArg,
        /// Output format.
        #[arg(long, value_enum, default_value_t = ReportFormat::Both)]
        format: ReportFormat,
    },
    /// SALT reconstruction-fidelity over a real fp safetensors master: quantize every
    /// 2D weight at each bpw budget and report whole-model (± per-tensor) error
    /// (Uniform vs sensitivity-allocated). Needs the fp master, not a quantized model.
    SaltModel {
        /// Path to the fp (bf16/f16/f32) safetensors master.
        #[arg(long)]
        input: PathBuf,
        /// Comma-separated bpw budgets, e.g. `1.585,2.0,2.5,3.0`.
        #[arg(long)]
        budgets: String,
        /// Sensitivity proxy for plane allocation.
        #[arg(long, value_enum, default_value_t = SaltSensitivityArg::Uniform)]
        sensitivity: SaltSensitivityArg,
        /// Base-plane scale granularity.
        #[arg(long, value_enum, default_value_t = quantize::ScaleGroupArg::Block)]
        scale_group: quantize::ScaleGroupArg,
        /// Only quantize the first N 2D tensors (0 = all) — a quick smoke on huge models.
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// Include the per-tensor breakdown in the report.
        #[arg(long, default_value_t = false)]
        per_tensor: bool,
        /// Output format.
        #[arg(long, value_enum, default_value_t = ReportFormat::Both)]
        format: ReportFormat,
    },
}

/// Report output format.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum ReportFormat {
    /// Human-readable table plus JSON.
    Both,
    /// JSON only.
    Json,
    /// Human-readable table only.
    Table,
}

/// SALT report sensitivity proxy.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum SaltSensitivityArg {
    /// Uniform group weighting.
    Uniform,
    /// Weight-energy proxy.
    Energy,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { path } => inspect::run(&path)?,
        Command::ListBackends => backends::run(),
        Command::Campaign { campaign: command } => campaign::run(command)?,
        Command::Salt { salt: command } => salt::run(command)?,
        Command::Pull {
            repo,
            file,
            revision,
        } => pull::run(&repo, file.as_deref(), &revision)?,
        Command::Generate {
            model,
            tokens,
            max_new,
            greedy,
            eos,
        } => {
            let ids = generate::read_token_file(&tokens)?;
            generate::run(&model, &ids, max_new, greedy, eos)?;
        }
        Command::Repack { input, output, to } => repack::run(&input, &output, to)?,
        Command::Release { release: command } => release::run(command)?,
        Command::Report { report: command } => match command {
            ReportCommand::Sparsity { model } => report::sparsity(&model)?,
            ReportCommand::Decode {
                model,
                tokens,
                backend,
                decode_steps,
                warmup,
                format,
            } => {
                let ids = generate::read_token_file(&tokens)?;
                report::decode(&model, &ids, &backend, decode_steps, warmup, format)?;
            }
            ReportCommand::Compare {
                model,
                tokens,
                backend,
                prompt_len,
                decode_steps,
                warmup,
                reps,
                runs,
                format,
            } => {
                let ids = generate::read_token_file(&tokens)?;
                report::compare(
                    &model,
                    &ids,
                    &backend,
                    prompt_len,
                    decode_steps,
                    warmup,
                    reps,
                    runs,
                    format,
                )?
            }
            ReportCommand::Ttft {
                model,
                tokens,
                backend,
                runs,
                format,
            } => {
                let ids = generate::read_token_file(&tokens)?;
                report::ttft(&model, &ids, &backend, runs, format)?;
            }
            ReportCommand::Parity {
                model,
                tokens,
                max_new,
                eos,
                format,
            } => {
                let ids = generate::read_token_file(&tokens)?;
                report::parity(&model, &ids, max_new, eos, format)?;
            }
            ReportCommand::Salt {
                input,
                rows,
                k,
                budgets,
                sensitivity,
                format,
            } => report::salt(&input, rows, k, &budgets, sensitivity, format)?,
            ReportCommand::SaltModel {
                input,
                budgets,
                sensitivity,
                scale_group,
                limit,
                per_tensor,
                format,
            } => report::salt_model(
                &input,
                &budgets,
                sensitivity,
                scale_group,
                limit,
                per_tensor,
                format,
            )?,
        },
        Command::Quantize {
            input,
            output,
            bpw,
            scale_group,
            sensitivity,
            fisher,
            format,
        } => quantize::run(
            &input,
            &output,
            bpw,
            scale_group,
            sensitivity,
            fisher.as_deref(),
            format,
        )?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn campaign_teacher_cache_cli_parses_fixed_window_inputs() {
        let cli = Cli::try_parse_from([
            "tritium",
            "campaign",
            "teacher-cache",
            "--model-dir",
            "model",
            "--corpus",
            "tokens.json",
            "--seq-len",
            "32",
            "--output",
            "teacher.ttpr",
        ])
        .expect("teacher-cache CLI");

        assert!(matches!(
            cli.command,
            Command::Campaign {
                campaign: campaign::CampaignCommand::TeacherCache { seq_len: 32, .. }
            }
        ));
    }

    #[test]
    fn campaign_run_cli_parses_config() {
        let cli = Cli::try_parse_from(["tritium", "campaign", "run", "--config", "campaign.json"])
            .expect("campaign run CLI");

        assert!(matches!(
            cli.command,
            Command::Campaign {
                campaign: campaign::CampaignCommand::Run { config }
            } if config == std::path::Path::new("campaign.json")
        ));
    }

    #[test]
    fn qwen36_preflight_cli_parses_immutable_candidate_output() {
        let cli = Cli::try_parse_from([
            "tritium",
            "salt",
            "qwen36-preflight",
            "--model-dir",
            "model",
            "--work-root",
            "work",
            "--output",
            "candidate.json",
        ])
        .expect("Qwen3.6 preflight CLI");

        assert!(matches!(
            cli.command,
            Command::Salt {
                salt: salt::SaltCommand::Qwen36Preflight {
                    model_dir,
                    work_root,
                    output,
                }
            } if model_dir == std::path::Path::new("model")
                && work_root == std::path::Path::new("work")
                && output == std::path::Path::new("candidate.json")
        ));
    }
}
