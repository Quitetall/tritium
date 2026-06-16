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

use std::path::PathBuf;

use clap::{Parser, Subcommand};

mod backends;
mod generate;
mod inspect;

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
    }
    Ok(())
}
