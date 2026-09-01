//! The `tritium` command-line tool.
//!
//! One binary, fourteen subcommands, grouped by concern:
//! - **inspect / list-backends / generate** — parse a GGUF and print its
//!   metadata + tensor table, enumerate the discovered compute backends,
//!   greedily decode from a reproducible JSON token file.
//! - **pull** — fetch GGUF models from the HuggingFace hub into the shared
//!   local cache (resumable).
//! - **report** — the reproducible benchmark/validation reports
//!   (`sparsity`, `decode`, `ttft`, `compare`, `parity`, `salt`,
//!   `salt-model`) that every docs/BENCHMARKS.md number reproduces from.
//! - **repack / transport / convert / quantize** — ternary container
//!   repacking, the seekable outer transport, HF→ternary conversion, and
//!   SALT quantization.
//! - **campaign / salt / release** — offline teacher targets, resumable
//!   SALT V2 synthesis workflows, and release-candidate verification.
//!
//! Serving is the separate `tritium-serve` binary.
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
mod convert;
mod generate;
#[cfg(feature = "cuda")]
mod hestia_gate;
mod hex;
mod inspect;
#[cfg(feature = "cuda")]
mod nvml_probe;
mod pull;
mod quantize;
mod quantize_ladder;
mod release;
mod repack;
mod report;
mod salt;
mod stage7_evidence;
mod transport;

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
    /// (~/.cache/tritium-models; override with TRITIUM_MODEL_DIR or the
    /// legacy TRITIUM_MODEL_CACHE — the same directory the report/bench
    /// harnesses read). Gated repos need HF_TOKEN. Resumes partial
    /// downloads on re-run.
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
        /// A `.gguf` model file, or a directory written by `tritium convert`.
        #[arg(long, alias = "input")]
        model: PathBuf,
        /// Path to a JSON file holding the input token IDs, e.g. `[1, 128000, 9906]`. The
        /// reproducible path: no tokenizer is consulted, so the same ids always go in and out.
        #[arg(long, conflicts_with = "prompt", required_unless_present = "prompt")]
        tokens: Option<PathBuf>,
        /// Text prompt, tokenized with the MODEL'S OWN tokenizer (`tokenizer.json` in a
        /// `tritium convert` directory, or the BPE embedded in a GGUF). Token ids are only
        /// meaningful relative to the vocabulary that produced them, so there is no fallback.
        #[arg(long, conflicts_with = "tokens")]
        prompt: Option<String>,
        /// Maximum number of new tokens to generate.
        #[arg(long, default_value_t = 16)]
        max_new: usize,
        /// Decode greedily — the only strategy this subcommand implements.
        /// `--greedy=false` is REJECTED (sampling lives in `tritium-serve`'s
        /// OpenAI API); the flag exists so scripts can be explicit.
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
        #[arg(long, alias = "model")]
        input: PathBuf,
        /// Path to write the repacked `.gguf`.
        #[arg(long, alias = "out")]
        output: PathBuf,
        /// Target ternary format.
        #[arg(long, value_enum)]
        to: repack::RepackTarget,
    },
    /// Pack or restore a seekable outer transport without changing resident-byte accounting.
    Transport {
        /// Transport operation.
        #[command(subcommand)]
        transport: transport::TransportCommand,
    },
    /// Convert a Hugging Face fp model directory to a ready-to-run ternary model directory.
    ///
    /// Unlike `quantize`, this loads the whole model, so it can run the activation-aware salience
    /// fold (`--calib`) — the lever every published SALT number was measured under — and it writes
    /// the folded norms alongside the bundle. Writing only a bundle would produce a model whose
    /// weights carry the fold and whose norms do not.
    Convert {
        /// Hugging Face model directory (config.json + safetensors shards).
        #[arg(long, alias = "input")]
        model: PathBuf,
        /// Output directory: config.json + model.safetensors (folded norms) + model.tslb.
        #[arg(long, alias = "output")]
        out: PathBuf,
        /// Calibration corpus for the salience fold: either a corpus JSON with a `train_ids`
        /// array, or a UTF-8 text file tokenized with the model's own tokenizer. Without it the
        /// fold is skipped and the result matches `tritium quantize`.
        #[arg(long)]
        calib: Option<PathBuf>,
        /// Calibration tokens to use (rounded down to whole 512-token windows).
        #[arg(long, default_value_t = 4096)]
        calib_tokens: usize,
        /// Salience-fold strength. 0.75 is the value every published SALT number used, but the
        /// optimum shifts DOWN with model size (0.75 -> 0.50 observed), so it is worth sweeping.
        #[arg(long, default_value_t = 0.75)]
        fold_alpha: f64,
        /// Plane count. 4 measures 1.024x fp on SmolLM2-360M without any fold; 3 measures 1.335x.
        #[arg(long, default_value_t = 4)]
        planes: usize,
        /// Scale-group width. Must be a multiple of 256 (one f16 scale per TQ2_0 block).
        #[arg(long, default_value_t = 256)]
        group: usize,
        /// Delta candidates per group for the ladder's `s0` grid search.
        #[arg(long, default_value_t = 16)]
        grid: usize,
    },
    /// SALT-quantize an fp safetensors model to a SALT bundle.
    Quantize {
        /// Path to the fp16/bf16/f32 `.safetensors` source model.
        #[arg(long, alias = "model")]
        input: PathBuf,
        /// Path to write the SALT bundle (`.tslb`).
        #[arg(long, alias = "out")]
        output: PathBuf,
        /// Target average bits-per-weight (`1.585` = all base ternary … `~4.75` at T=3).
        /// Applies to `--ladder itf` only (default 2.0 there); the geometric
        /// ladder derives its rate from `--planes` and REJECTS this flag.
        #[arg(long)]
        bpw: Option<f64>,
        /// Base-plane scale granularity: `block` (per-256-block) or `tensor` (per-tensor,
        /// for a BitNet b1.58 master). `--ladder itf` only (default `block`).
        #[arg(long, value_enum)]
        scale_group: Option<quantize::ScaleGroupArg>,
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
        /// Fitter. `geometric` (default) is the balanced-ternary ladder: one anchor per group,
        /// `s_p = s0*3^-(p-1)`. `itf` is the previous free-scale fit, kept for reproducing
        /// published numbers and because it beats the ladder below 3 planes.
        #[arg(long, value_enum, default_value_t = quantize_ladder::LadderArg::Geometric)]
        ladder: quantize_ladder::LadderArg,
        /// Plane count for `--ladder geometric`. 4 measures 1.024x fp on SmolLM2-360M in the
        /// configuration this command can write (no calibration fold, no rotation); 3 measures
        /// 1.335x. Below 3 the ladder is refused — see the error text.
        #[arg(long, default_value_t = 4)]
        planes: usize,
        /// Scale-group width for `--ladder geometric`. Must be a multiple of 256: a TQ2_0 block
        /// carries one f16 scale per 256 trits, so a smaller group would need two anchors in one
        /// block.
        #[arg(long, default_value_t = 256)]
        group: usize,
        /// Delta candidates per group for the ladder's `s0` grid search.
        #[arg(long, default_value_t = 16)]
        grid: usize,
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
        #[arg(long, alias = "input")]
        model: PathBuf,
    },
    /// Decode-only throughput after prefill.
    Decode {
        /// Path to the `.gguf` model file.
        #[arg(long, alias = "input")]
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
        #[arg(long, alias = "input")]
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
        #[arg(long, alias = "input")]
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
        #[arg(long, alias = "input")]
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
        #[arg(long, alias = "model")]
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
        #[arg(long, alias = "model")]
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
            prompt,
            max_new,
            greedy,
            eos,
        } => {
            // clap guarantees exactly one of the two is present.
            let ids = tokens.map(|p| generate::read_token_file(&p)).transpose()?;
            let source = match (&ids, &prompt) {
                (Some(ids), _) => generate::Prompt::Ids(ids),
                (None, Some(text)) => generate::Prompt::Text(text),
                (None, None) => unreachable!("clap requires --tokens or --prompt"),
            };
            generate::run(&model, &source, max_new, greedy, eos)?;
        }
        Command::Repack { input, output, to } => repack::run(&input, &output, to)?,
        Command::Transport { transport: command } => transport::run(command)?,
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
        Command::Convert {
            model,
            out,
            calib,
            calib_tokens,
            fold_alpha,
            planes,
            group,
            grid,
        } => convert::run(
            &model,
            &out,
            &convert::ConvertConfig {
                calib,
                calib_tokens,
                fold_alpha,
                ladder: quantize_ladder::LadderConfig {
                    planes,
                    group,
                    grid,
                },
            },
        )?,
        Command::Quantize {
            input,
            output,
            bpw,
            scale_group,
            sensitivity,
            fisher,
            format,
            ladder,
            planes,
            group,
            grid,
        } => quantize::run(
            &input,
            &output,
            bpw,
            scale_group,
            sensitivity,
            fisher.as_deref(),
            format,
            ladder,
            quantize_ladder::LadderConfig {
                planes,
                group,
                grid,
            },
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
