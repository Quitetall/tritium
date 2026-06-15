//! The `tritium` command-line tool.
//!
//! Two subcommands:
//! - `tritium inspect <PATH.gguf>` — parse a GGUF container and print a summary of
//!   its version, metadata, architecture, alignment, and tensor table.
//! - `tritium list-backends` — enumerate every backend the runtime discovered.
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
mod inspect;

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
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { path } => inspect::run(&path)?,
        Command::ListBackends => backends::run(),
    }
    Ok(())
}
