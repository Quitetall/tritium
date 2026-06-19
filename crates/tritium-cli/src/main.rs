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

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

mod backends;
mod generate;
mod inspect;
mod quantize;
mod report;

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
        /// Output container: `sidecar` (single-file `.tslb` bundle) or `gguf`
        /// (GGUF container holding the SALT rows).
        #[arg(long, value_enum, default_value_t = quantize::OutputFormat::Sidecar)]
        format: quantize::OutputFormat,
    },
}

/// Benchmark/validation reports.
#[derive(Subcommand, Debug)]
enum ReportCommand {
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
        Command::Report { report: command } => match command {
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
        },
        Command::Quantize {
            input,
            output,
            bpw,
            scale_group,
            format,
        } => quantize::run(&input, &output, bpw, scale_group, format)?,
    }
    Ok(())
}
