//! Inspection and verification of ADR 0044 evidence JSONL logs.

use std::fs;
use std::path::PathBuf;

use clap::Subcommand;
use serde::Serialize;
use tritium_evidence::verify_jsonl;

/// Evidence log operation.
#[derive(Subcommand, Debug)]
pub(crate) enum EvidenceCommand {
    /// Verify canonical encoding, event digests, span chains, and run root.
    Verify {
        /// Evidence JSONL file.
        path: PathBuf,
    },
    /// Verify a log and print its events as formatted JSON.
    View {
        /// Evidence JSONL file.
        path: PathBuf,
    },
}

#[derive(Serialize)]
struct VerifySummary<'a> {
    integrity: &'static str,
    event_count: usize,
    span_count: usize,
    root_digest: &'a str,
}

/// Run an evidence subcommand.
pub(crate) fn run(command: EvidenceCommand) -> anyhow::Result<()> {
    let (path, view) = match command {
        EvidenceCommand::Verify { path } => (path, false),
        EvidenceCommand::View { path } => (path, true),
    };
    let input = fs::read_to_string(&path)
        .map_err(|error| anyhow::anyhow!("read evidence log {}: {error}", path.display()))?;
    let verified = verify_jsonl(&input)
        .map_err(|error| anyhow::anyhow!("verify evidence log {}: {error}", path.display()))?;
    if view {
        println!("{}", serde_json::to_string_pretty(&verified.events)?);
    } else {
        let summary = VerifySummary {
            integrity: "verified",
            event_count: verified.events.len(),
            span_count: verified
                .events
                .iter()
                .map(|event| event.span.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            root_digest: &verified.root_digest,
        };
        println!("{}", serde_json::to_string_pretty(&summary)?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{EvidenceCommand, run};
    use serde::Serialize;
    use tritium_evidence::{EventRegistration, EvidenceEvent, EvidenceRecorder};

    #[derive(Serialize)]
    struct ProbeEvent {
        value: u32,
    }

    impl EvidenceEvent for ProbeEvent {
        const SCHEMA_ID: &'static str = "tritium.test.cli-probe";
        const VERSION: u16 = 1;
    }

    #[test]
    fn verify_command_accepts_a_valid_recorded_log() {
        let mut recorder = EvidenceRecorder::new(
            "cli-test-run",
            [EventRegistration {
                schema: ProbeEvent::SCHEMA_ID,
                version: ProbeEvent::VERSION,
            }],
        )
        .expect("valid registry");
        recorder
            .emit("root", None, 0, &ProbeEvent { value: 42 })
            .expect("registered event");
        let path =
            std::env::temp_dir().join(format!("tritium-evidence-cli-{}.jsonl", std::process::id()));
        std::fs::write(&path, recorder.to_jsonl().expect("canonical log"))
            .expect("write temporary log");
        let result = run(EvidenceCommand::Verify { path: path.clone() });
        let cleanup = std::fs::remove_file(path);
        cleanup.expect("remove temporary log");
        result.expect("CLI verification succeeds");
    }
}
