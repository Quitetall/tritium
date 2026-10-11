//! Inspection and verification of ADR 0044 evidence JSONL logs.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::{Subcommand, ValueEnum};
use serde::Serialize;
use tritium_evidence::{
    CalibratedFitReceiptView, CalibratedQuantizationReceiptView, CalibrationCaptureView,
    DirectFitReceiptView, EventRegistration, EvidenceEvent, EvidenceRecorder, ModelLoadReceiptView,
    QuantizationReceiptView, SalienceTransformView, verify_jsonl,
};

/// Evidence detail level for a CLI invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum EvidenceMode {
    /// Do not record events.
    Off,
    /// Record a bounded command outcome summary.
    Summary,
    /// Record command start and completion events.
    Full,
    /// Record deterministic logical-time events; requires a caller-stable run id.
    Det,
}

#[derive(Serialize)]
struct CliCommandEvent {
    command: &'static str,
    phase: &'static str,
    success: Option<bool>,
}

impl EvidenceEvent for CliCommandEvent {
    const SCHEMA_ID: &'static str = "tritium.cli.command";
    const VERSION: u16 = 1;
}

/// Evidence session surrounding one CLI command.
pub(crate) struct EvidenceSession {
    mode: EvidenceMode,
    writer: Option<std::fs::File>,
    recorder: Option<EvidenceRecorder>,
}

impl EvidenceSession {
    /// Start evidence recording and emit the start event for `full` mode.
    pub(crate) fn start(
        mode: EvidenceMode,
        output: Option<PathBuf>,
        run_id: Option<String>,
        command: &'static str,
    ) -> anyhow::Result<Self> {
        if mode == EvidenceMode::Off {
            anyhow::ensure!(
                output.is_none() && run_id.is_none(),
                "--evidence-out and --run-id require evidence recording to be enabled"
            );
            return Ok(Self {
                mode,
                writer: None,
                recorder: None,
            });
        }
        if mode == EvidenceMode::Det {
            anyhow::ensure!(
                run_id.is_some(),
                "--evidence det requires an explicit stable --run-id"
            );
        }
        let run_id = match run_id {
            Some(run_id) if !run_id.is_empty() => run_id,
            Some(_) => anyhow::bail!("--run-id must not be empty"),
            None => generated_run_id()?,
        };
        let mut recorder = EvidenceRecorder::new(
            run_id,
            [EventRegistration {
                schema: CliCommandEvent::SCHEMA_ID,
                version: CliCommandEvent::VERSION,
            }],
        )?;
        if mode == EvidenceMode::Full {
            recorder.emit(
                "cli",
                None,
                0,
                &CliCommandEvent {
                    command,
                    phase: "started",
                    success: None,
                },
            )?;
        }
        let writer = output
            .map(|path| {
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "reserve evidence log {} without replacing existing data: {error}",
                            path.display()
                        )
                    })
            })
            .transpose()?;
        Ok(Self {
            mode,
            writer,
            recorder: Some(recorder),
        })
    }

    /// Record completion, then write a new immutable JSONL artifact or stderr.
    pub(crate) fn finish(mut self, command: &'static str, success: bool) -> anyhow::Result<()> {
        if self.mode == EvidenceMode::Off {
            return Ok(());
        }
        let recorder = self
            .recorder
            .as_mut()
            .expect("enabled evidence session has a recorder");
        recorder.emit(
            "cli",
            None,
            0,
            &CliCommandEvent {
                command,
                phase: "completed",
                success: Some(success),
            },
        )?;
        let jsonl = recorder.to_jsonl()?;
        if let Some(mut writer) = self.writer {
            writer.write_all(jsonl.as_bytes())?;
            writer.sync_all()?;
        } else {
            let stderr = std::io::stderr();
            let mut locked = stderr.lock();
            locked.write_all(jsonl.as_bytes())?;
            locked.flush()?;
        }
        Ok(())
    }
}

fn generated_run_id() -> anyhow::Result<String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| anyhow::anyhow!("system clock is before Unix epoch: {error}"))?
        .as_nanos();
    Ok(format!("cli-{}-{nanos}", std::process::id()))
}

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
    // Known load observations use the shared typed fold, not a CLI schema copy.
    // Generic integrity is still not qualification or a registry for all events.
    ModelLoadReceiptView::fold(&verified)
        .map_err(|error| anyhow::anyhow!("verify model-load view {}: {error}", path.display()))?;
    DirectFitReceiptView::fold(&verified)
        .map_err(|error| anyhow::anyhow!("verify direct-fit view {}: {error}", path.display()))?;
    CalibratedFitReceiptView::fold(&verified).map_err(|error| {
        anyhow::anyhow!("verify calibrated-fit view {}: {error}", path.display())
    })?;
    QuantizationReceiptView::fold(&verified)
        .map_err(|error| anyhow::anyhow!("verify quantization view {}: {error}", path.display()))?;
    CalibratedQuantizationReceiptView::fold(&verified).map_err(|error| {
        anyhow::anyhow!("verify calibrated model view {}: {error}", path.display())
    })?;
    SalienceTransformView::fold(&verified).map_err(|error| {
        anyhow::anyhow!("verify salience transform view {}: {error}", path.display())
    })?;
    CalibrationCaptureView::fold(&verified).map_err(|error| {
        anyhow::anyhow!(
            "verify calibration capture view {}: {error}",
            path.display()
        )
    })?;
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
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{EvidenceCommand, EvidenceMode, EvidenceSession, run};
    use serde::Serialize;
    use tritium_evidence::{EventRegistration, EvidenceEvent, EvidenceRecorder, verify_jsonl};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    fn temp_log() -> PathBuf {
        std::env::temp_dir().join(format!(
            "tritium-cli-evidence-{}-{}.jsonl",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[derive(Serialize)]
    struct ProbeEvent {
        value: u32,
    }

    impl EvidenceEvent for ProbeEvent {
        const SCHEMA_ID: &'static str = "tritium.test.cli-probe";
        const VERSION: u16 = 1;
    }

    fn actual_transform() -> tritium_quantize::engine::RecordedSalienceSource {
        use std::collections::BTreeMap;
        use tritium_quantize::engine::{
            DeclaredSalienceStatistics, Engine, InMemoryWeightSource, RawTensor, SourceTransform,
            WeightSource,
        };
        use tritium_schema::*;
        let raw = |shape: Vec<u64>, values: &[f32]| RawTensor {
            shape,
            precision: ScalePrecision::F32,
            bytes: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        };
        let mut source = InMemoryWeightSource {
            tensors: BTreeMap::from([
                ("norm".into(), raw(vec![2], &[1.0, 1.0])),
                ("q".into(), raw(vec![1, 2], &[2.0, -2.0])),
            ]),
            assets: BTreeMap::from([
                ("config.json".into(), b"{}".to_vec()),
                ("tokenizer.json".into(), b"{}".to_vec()),
            ]),
            ..Default::default()
        };
        let observe = QuantPlan {
            architecture: "fixture".into(),
            tensors: source
                .inventory()
                .unwrap()
                .into_iter()
                .map(|source| QuantTensorPlan {
                    source,
                    action: QuantAction::KeepDense,
                })
                .collect(),
            assets: source.asset_names(),
            levels: vec!["dense".into()],
            limits: QuantLimits {
                max_source_bytes: 128,
                max_asset_bytes: 128,
                max_retained_blob_bytes: 128,
                max_package_bytes: 10_000,
            },
        };
        let stats = DeclaredSalienceStatistics {
            moments: vec![1.0, 16.0],
            rows: 1,
        };
        let plan = SalienceTransformPlan {
            architecture: "fixture".into(),
            original_source: Engine::observe_source(&observe, &mut source).unwrap(),
            tensors: source.inventory().unwrap(),
            assets: source.asset_names(),
            operations: vec![SalienceTransformOperation {
                name: "attn".into(),
                projections: vec!["q".into()],
                inverse: "norm".into(),
                inverse_axis: SalienceInverseAxis::NormChannels,
                moments: stats.binding().unwrap(),
                alpha_bits: 0.75f64.to_le_bytes(),
            }],
            numerics: SalienceNumerics::LegacyF64ScaleF32V1,
            statistics_kind: SalienceStatisticsKind::Declared,
            limits: SalienceTransformLimits {
                max_source_bytes: 128,
                max_asset_bytes: 128,
                max_derived_bytes: 128,
                max_statistics_bytes: 16,
                max_pair_working_bytes: 128,
            },
        };
        let recorder = EvidenceRecorder::new(
            "cli-transform",
            [
                tritium_evidence::SALIENCE_TRANSFORM_STARTED_REGISTRATION,
                tritium_evidence::SALIENCE_TRANSFORM_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap();
        SourceTransform::run_recorded(
            &plan,
            &mut source,
            &BTreeMap::from([("attn".into(), stats)]),
            recorder,
        )
        .unwrap()
    }

    #[test]
    fn verify_and_view_accept_actual_declared_transform_observations() {
        let recorded = actual_transform();
        let path = temp_log();
        std::fs::write(&path, recorded.evidence.jsonl()).unwrap();
        run(EvidenceCommand::Verify { path: path.clone() }).unwrap();
        run(EvidenceCommand::View { path: path.clone() }).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn verify_and_view_reject_known_capture_payload_with_unknown_fields() {
        #[derive(Serialize)]
        struct BadCapture {
            windows: Vec<Vec<u32>>,
            caches: Vec<serde_json::Value>,
            claimed_quality: bool,
        }
        impl EvidenceEvent for BadCapture {
            const SCHEMA_ID: &'static str = "tritium.calibration.capture_completed";
            const VERSION: u16 = 1;
        }
        let mut recorder = EvidenceRecorder::new(
            "bad-capture",
            [tritium_evidence::CALIBRATION_CAPTURE_COMPLETED_REGISTRATION],
        )
        .unwrap();
        recorder
            .emit(
                "capture",
                None,
                0,
                &BadCapture {
                    windows: vec![vec![1, 2]],
                    caches: vec![],
                    claimed_quality: true,
                },
            )
            .unwrap();
        let sealed = recorder.seal().unwrap();
        verify_jsonl(sealed.jsonl()).unwrap();
        let path = temp_log();
        std::fs::write(&path, sealed.jsonl()).unwrap();
        for command in [
            EvidenceCommand::Verify { path: path.clone() },
            EvidenceCommand::View { path: path.clone() },
        ] {
            assert!(
                run(command)
                    .unwrap_err()
                    .to_string()
                    .contains("calibration capture")
            );
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn verify_and_view_reject_invalid_transform_payloads_with_valid_hashes() {
        use tritium_schema::SalienceTransformStarted;
        #[derive(Serialize)]
        struct BadCompletion {
            observation: serde_json::Value,
        }
        impl EvidenceEvent for BadCompletion {
            const SCHEMA_ID: &'static str = "tritium.quantize.salience_transform_completed";
            const VERSION: u16 = 1;
        }
        let actual = actual_transform();
        let observation = actual.source.observation();
        let mut value = serde_json::to_value(observation).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("claimed_capture".into(), serde_json::json!(true));
        let mut recorder = EvidenceRecorder::new(
            "bad-transform",
            [
                tritium_evidence::SALIENCE_TRANSFORM_STARTED_REGISTRATION,
                tritium_evidence::SALIENCE_TRANSFORM_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap();
        recorder
            .emit(
                "transform",
                None,
                0,
                &SalienceTransformStarted {
                    plan: observation.plan.clone(),
                },
            )
            .unwrap();
        recorder
            .emit("transform", None, 1, &BadCompletion { observation: value })
            .unwrap();
        let sealed = recorder.seal().unwrap();
        verify_jsonl(sealed.jsonl()).unwrap();
        let path = temp_log();
        std::fs::write(&path, sealed.jsonl()).unwrap();
        assert!(
            run(EvidenceCommand::Verify { path: path.clone() })
                .unwrap_err()
                .to_string()
                .contains("salience transform")
        );
        assert!(
            run(EvidenceCommand::View { path: path.clone() })
                .unwrap_err()
                .to_string()
                .contains("salience transform")
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn verify_and_view_accept_registered_load_observations() {
        use tritium_evidence::{MODEL_LOADED_REGISTRATION, ModelLoaded};
        let mut recorder = EvidenceRecorder::new("load-view", [MODEL_LOADED_REGISTRATION]).unwrap();
        recorder
            .emit_model_loaded(
                "load",
                None,
                0,
                &ModelLoaded {
                    source: tritium_evidence::LoadedSourceIdentity::HuggingFace {
                        reason: tritium_evidence::UnknownReason::MissingCapability,
                    },
                    architecture: "llama".into(),
                    backend: "controlled-test".into(),
                    physical_device: "unknown".into(),
                    vocab_size: 10,
                    layer_count: 2,
                },
            )
            .unwrap();
        let path = temp_log();
        std::fs::write(&path, recorder.to_jsonl().unwrap()).unwrap();
        let verify = run(EvidenceCommand::Verify { path: path.clone() });
        let view = run(EvidenceCommand::View { path: path.clone() });
        std::fs::remove_file(path).unwrap();
        verify.unwrap();
        view.unwrap();
    }

    #[test]
    fn verify_and_view_reject_load_payloads_with_valid_hashes_but_wrong_schema() {
        use tritium_evidence::{MODEL_LOADED_REGISTRATION, ModelLoaded};
        #[derive(Serialize)]
        struct InvalidLoad {
            unauthorized_release_verdict: bool,
        }
        impl EvidenceEvent for InvalidLoad {
            const SCHEMA_ID: &'static str = ModelLoaded::SCHEMA_ID;
            const VERSION: u16 = 1;
        }
        let mut recorder =
            EvidenceRecorder::new("wrong-load", [MODEL_LOADED_REGISTRATION]).unwrap();
        recorder
            .emit(
                "load",
                None,
                0,
                &InvalidLoad {
                    unauthorized_release_verdict: true,
                },
            )
            .unwrap();
        let path = temp_log();
        std::fs::write(&path, recorder.to_jsonl().unwrap()).unwrap();
        let verify = run(EvidenceCommand::Verify { path: path.clone() });
        let view = run(EvidenceCommand::View { path: path.clone() });
        std::fs::remove_file(path).unwrap();
        assert!(verify.unwrap_err().to_string().contains("InvalidPayload"));
        assert!(view.unwrap_err().to_string().contains("InvalidPayload"));
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
        let path = temp_log();
        std::fs::write(&path, recorder.to_jsonl().expect("canonical log"))
            .expect("write temporary log");
        let result = run(EvidenceCommand::Verify { path: path.clone() });
        let cleanup = std::fs::remove_file(path);
        cleanup.expect("remove temporary log");
        result.expect("CLI verification succeeds");
    }

    #[test]
    fn verify_and_view_reject_invalid_calibrated_model_payloads_with_valid_hashes() {
        #[derive(Serialize)]
        struct InvalidStart {
            release_approved: bool,
        }
        impl EvidenceEvent for InvalidStart {
            const SCHEMA_ID: &'static str = "tritium.quantize.calibrated_model_started";
            const VERSION: u16 = 1;
        }
        #[derive(Serialize)]
        struct InvalidCompletion {
            release_approved: bool,
        }
        impl EvidenceEvent for InvalidCompletion {
            const SCHEMA_ID: &'static str = "tritium.quantize.calibrated_model_completed";
            const VERSION: u16 = 1;
        }
        fn reject<E: EvidenceEvent>(event: &E) {
            let mut recorder = EvidenceRecorder::new(
                "invalid-calibrated-model",
                [EventRegistration {
                    schema: E::SCHEMA_ID,
                    version: E::VERSION,
                }],
            )
            .unwrap();
            recorder.emit("quantize", None, 0, event).unwrap();
            let path = temp_log();
            std::fs::write(&path, recorder.to_jsonl().unwrap()).unwrap();
            let verify = run(EvidenceCommand::Verify { path: path.clone() });
            let view = run(EvidenceCommand::View { path: path.clone() });
            std::fs::remove_file(path).unwrap();
            assert!(
                verify
                    .unwrap_err()
                    .to_string()
                    .contains("calibrated model view")
            );
            assert!(
                view.unwrap_err()
                    .to_string()
                    .contains("calibrated model view")
            );
        }
        reject(&InvalidStart {
            release_approved: true,
        });
        reject(&InvalidCompletion {
            release_approved: true,
        });
    }

    #[test]
    fn verify_and_view_reject_invalid_calibrated_payload_with_valid_hashes() {
        #[derive(Serialize)]
        struct InvalidCalibratedFit {
            release_approved: bool,
        }
        impl EvidenceEvent for InvalidCalibratedFit {
            const SCHEMA_ID: &'static str = "tritium.quantize.calibrated_levels_fitted";
            const VERSION: u16 = 1;
        }
        let mut recorder = EvidenceRecorder::new(
            "invalid-calibrated-fit",
            [tritium_evidence::CALIBRATED_LEVELS_FITTED_REGISTRATION],
        )
        .unwrap();
        recorder
            .emit(
                "fit",
                None,
                0,
                &InvalidCalibratedFit {
                    release_approved: true,
                },
            )
            .unwrap();
        let path = temp_log();
        std::fs::write(&path, recorder.to_jsonl().unwrap()).unwrap();
        let verify = run(EvidenceCommand::Verify { path: path.clone() });
        let view = run(EvidenceCommand::View { path: path.clone() });
        std::fs::remove_file(path).unwrap();
        assert!(
            verify
                .unwrap_err()
                .to_string()
                .contains("calibrated-fit view")
        );
        assert!(
            view.unwrap_err()
                .to_string()
                .contains("calibrated-fit view")
        );
    }

    #[test]
    fn verify_and_view_reject_invalid_fit_payload_even_with_valid_hashes() {
        use tritium_evidence::DIRECT_LEVELS_FITTED_REGISTRATION;
        #[derive(Serialize)]
        struct InvalidFit {
            release_approved: bool,
        }
        impl EvidenceEvent for InvalidFit {
            const SCHEMA_ID: &'static str = "tritium.quantize.direct_levels_fitted";
            const VERSION: u16 = 1;
        }
        let mut recorder =
            EvidenceRecorder::new("invalid-fit", [DIRECT_LEVELS_FITTED_REGISTRATION]).unwrap();
        recorder
            .emit(
                "fit",
                None,
                0,
                &InvalidFit {
                    release_approved: true,
                },
            )
            .unwrap();
        let path = temp_log();
        std::fs::write(&path, recorder.to_jsonl().unwrap()).unwrap();
        let verify = run(EvidenceCommand::Verify { path: path.clone() });
        let view = run(EvidenceCommand::View { path: path.clone() });
        std::fs::remove_file(path).unwrap();
        assert!(verify.unwrap_err().to_string().contains("InvalidPayload"));
        assert!(view.unwrap_err().to_string().contains("InvalidPayload"));
    }

    #[test]
    fn det_mode_requires_stable_run_id_and_replays_identically() {
        let path_a = temp_log();
        let path_b = temp_log();
        assert!(EvidenceSession::start(EvidenceMode::Det, None, None, "inspect").is_err());

        EvidenceSession::start(
            EvidenceMode::Det,
            Some(path_a.clone()),
            Some("stable-run-1".into()),
            "inspect",
        )
        .expect("det session A")
        .finish("inspect", true)
        .expect("write deterministic session A");
        EvidenceSession::start(
            EvidenceMode::Det,
            Some(path_b.clone()),
            Some("stable-run-1".into()),
            "inspect",
        )
        .expect("det session B")
        .finish("inspect", true)
        .expect("write deterministic session B");

        let log_a = verify_jsonl(&std::fs::read_to_string(&path_a).expect("read log A"))
            .expect("verify log A");
        let log_b = verify_jsonl(&std::fs::read_to_string(&path_b).expect("read log B"))
            .expect("verify log B");
        assert_eq!(log_a.root_digest, log_b.root_digest);
        assert_eq!(log_a.events.len(), 1);
        assert_eq!(log_a.events[0].t, 0);
        std::fs::remove_file(path_a).expect("remove log A");
        std::fs::remove_file(path_b).expect("remove log B");
    }

    #[test]
    fn full_mode_records_start_and_completion() {
        let path = temp_log();
        EvidenceSession::start(
            EvidenceMode::Full,
            Some(path.clone()),
            Some("full-run".into()),
            "inspect",
        )
        .expect("full session")
        .finish("inspect", false)
        .expect("write full log");
        let log = verify_jsonl(&std::fs::read_to_string(&path).expect("read log"))
            .expect("verify full log");
        assert_eq!(log.events.len(), 2);
        assert_eq!(log.events[0].payload["phase"], "started");
        assert_eq!(log.events[1].payload["phase"], "completed");
        assert_eq!(log.events[1].payload["success"], false);
        std::fs::remove_file(path).expect("remove log");
    }

    #[test]
    fn existing_evidence_output_is_rejected_before_command_execution() {
        let path = temp_log();
        std::fs::write(&path, b"preserve me").expect("write sentinel file");
        let result = EvidenceSession::start(
            EvidenceMode::Summary,
            Some(path.clone()),
            Some("existing-output-run".into()),
            "inspect",
        );
        assert!(result.is_err());
        assert_eq!(std::fs::read(&path).expect("read sentinel"), b"preserve me");
        std::fs::remove_file(path).expect("remove sentinel");
    }

    #[test]
    fn recognized_malformed_quantization_payload_is_not_integrity_success() {
        #[derive(serde::Serialize)]
        struct Malformed {
            unexpected: bool,
        }
        impl tritium_evidence::EvidenceEvent for Malformed {
            const SCHEMA_ID: &'static str = "tritium.quantize.quantization_completed";
            const VERSION: u16 = 1;
        }
        let path = temp_log();
        let mut recorder = EvidenceRecorder::new(
            "malformed-model-fit",
            [tritium_evidence::QUANTIZATION_COMPLETED_REGISTRATION],
        )
        .unwrap();
        recorder
            .emit("fit", None, 0, &Malformed { unexpected: true })
            .unwrap();
        std::fs::write(&path, recorder.to_jsonl().unwrap()).unwrap();
        for command in [
            EvidenceCommand::Verify { path: path.clone() },
            EvidenceCommand::View { path: path.clone() },
        ] {
            assert!(run(command).is_err());
        }
        std::fs::remove_file(path).unwrap();
    }
}
