use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, write_salt_v2_package,
};
use tritium_quantize::{LogicalTritCount, MeasuredPackage, PhysicalSizeReport};
use tritium_salt::{
    ContentId, DriverFailure, EvidenceRef, HardwareUsage, Metric, PhysicalLedger,
    PublishedArtifact, QualityEvidence, RecipeRef, SaltDriver, SaltProfile, SaltSpec, SaltStage,
    SaltV2, SourceRef, StageArtifact, StageOutput, StageRequest,
};

use std::{fs, path::PathBuf};

fn spec() -> SaltSpec {
    let source_id = ContentId::of_bytes(b"source model");
    SaltSpec::new(
        SourceRef::new(source_id, "fixture://source").expect("source"),
        EvidenceRef::new(
            ContentId::of_bytes(b"evidence"),
            source_id,
            "fixture://evidence",
        )
        .expect("evidence"),
        RecipeRef::new(ContentId::of_bytes(b"recipe"), "fixture-backend", "rev-1").expect("recipe"),
        "fixture://published",
        SaltProfile::CompactV1,
    )
    .expect("spec")
}

#[test]
fn explain_exposes_the_canonical_pipeline() {
    let explanation = SaltV2::explain(&spec()).expect("explain");

    assert_eq!(explanation.profile(), SaltProfile::CompactV1);
    assert_eq!(explanation.stages(), &SaltStage::ALL);
    assert_eq!(explanation.work_id(), spec().work_id());
}

#[test]
fn content_id_hashes_files_and_sharded_directories_deterministically() {
    let root = unique_temp_dir("content-id");
    fs::create_dir_all(root.join("weights")).expect("create fixture");
    fs::write(root.join("config.json"), b"config").expect("config");
    fs::write(root.join("weights/part-2.bin"), b"two").expect("part 2");
    fs::write(root.join("weights/part-1.bin"), b"one").expect("part 1");

    let first = ContentId::from_path(&root).expect("hash directory");
    let second = ContentId::from_path(&root).expect("hash directory again");
    assert_eq!(first, second);
    assert_eq!(
        ContentId::from_file(root.join("config.json")).expect("hash file"),
        ContentId::of_bytes(b"config")
    );

    fs::write(root.join("weights/part-1.bin"), b"changed").expect("change shard");
    assert_ne!(ContentId::from_path(&root).expect("rehash"), first);
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn stages_have_stable_machine_names() {
    assert_eq!(SaltStage::Search.as_str(), "search");
    assert_eq!(SaltStage::Validate.to_string(), "validate");
}

#[test]
fn reconcile_is_idempotent_after_publication() {
    let root = unique_temp_dir("idempotent");
    let spec = spec();
    let mut first_driver = FixtureDriver::default();

    let first = SaltV2::reconcile(&spec, &root, &mut first_driver).expect("first reconcile");
    assert_eq!(
        first_driver.stages.as_slice(),
        SaltV2::explain(&spec).expect("explain").stages()
    );
    assert_eq!(first.stage_receipts().len(), SaltStage::ALL.len());
    assert_eq!(first.provenance().source_id(), spec.source().id());
    assert_eq!(first.provenance().evidence_id(), spec.evidence().id());
    assert_eq!(first.total_gpu_seconds(), 8 * 90);
    assert!((first.total_gpu_hours() - 0.2).abs() < f64::EPSILON);
    assert_eq!(first.published().expect("published").physical_bytes(), 230);
    let physical = first.physical().expect("physical ledger");
    assert_eq!(physical.resident_preserved_bytes(), 30);
    assert_eq!(physical.resident_shadow_bytes(), 40);
    assert_eq!(physical.resident_total_bytes(), 230);

    let mut second_driver = FixtureDriver::default();
    let second = SaltV2::reconcile(&spec, &root, &mut second_driver).expect("idempotent reconcile");
    assert!(second_driver.stages.is_empty());
    assert_eq!(second, first);
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn retryable_failure_resumes_the_same_stage() {
    let root = unique_temp_dir("resume");
    let spec = spec();
    let mut failing = FixtureDriver {
        fail_once: Some(SaltStage::Search),
        ..FixtureDriver::default()
    };

    let error = SaltV2::reconcile(&spec, &root, &mut failing).expect_err("injected failure");
    assert!(matches!(
        error,
        tritium_salt::SaltError::DriverFailure {
            stage: SaltStage::Search,
            retryable: true,
            ..
        }
    ));

    let mut resumed = FixtureDriver::default();
    let receipt = SaltV2::reconcile(&spec, &root, &mut resumed).expect("resume");
    assert_eq!(resumed.stages.first(), Some(&SaltStage::Search));
    let search = receipt
        .stage_receipts()
        .iter()
        .find(|record| record.stage() == SaltStage::Search)
        .expect("search receipt");
    assert_eq!(search.attempt(), 2);
    assert!(receipt.published().is_some());
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn resume_rehashes_every_accepted_upstream_artifact() {
    let root = unique_temp_dir("artifact-rehash");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    let pack = root
        .join(spec.work_id().to_string())
        .join("artifacts/pack.bin");
    fs::write(pack, b"tampered package").expect("tamper packed artifact");

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "accepted stage artifact changed"
        ))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn incompatible_source_and_evidence_fail_before_work_exists() {
    let source_id = ContentId::of_bytes(b"source-a");
    let result = SaltSpec::new(
        SourceRef::new(source_id, "fixture://source-a").expect("source"),
        EvidenceRef::new(
            ContentId::of_bytes(b"evidence-b"),
            ContentId::of_bytes(b"source-b"),
            "fixture://evidence-b",
        )
        .expect("evidence"),
        RecipeRef::new(ContentId::of_bytes(b"recipe"), "fixture", "rev").expect("recipe"),
        "fixture://published",
        SaltProfile::CompactV1,
    );

    assert!(matches!(
        result,
        Err(tritium_salt::SaltError::EvidenceSourceMismatch { .. })
    ));
}

#[test]
fn zero_content_identities_fail_at_reference_construction() {
    let zero = ContentId::from_digest([0; 32]);
    let valid = ContentId::of_bytes(b"valid");

    assert!(matches!(
        SourceRef::new(zero, "fixture://source"),
        Err(tritium_salt::SaltError::InvalidField(
            "source content identity"
        ))
    ));
    assert!(matches!(
        EvidenceRef::new(zero, valid, "fixture://evidence"),
        Err(tritium_salt::SaltError::InvalidField(
            "evidence content identity"
        ))
    ));
    assert!(matches!(
        EvidenceRef::new(valid, zero, "fixture://evidence"),
        Err(tritium_salt::SaltError::InvalidField(
            "evidence source identity"
        ))
    ));
    assert!(matches!(
        RecipeRef::new(zero, "fixture", "rev"),
        Err(tritium_salt::SaltError::InvalidField(
            "recipe content identity"
        ))
    ));
}

#[test]
fn failed_quality_evidence_is_retained_and_publish_never_runs() {
    let root = unique_temp_dir("quality-failure");
    let spec = spec();
    let mut rejecting = FixtureDriver {
        reject_quality: true,
        ..FixtureDriver::default()
    };

    let error = SaltV2::reconcile(&spec, &root, &mut rejecting).expect_err("quality failure");
    assert!(matches!(
        error,
        tritium_salt::SaltError::QualityGateFailed { .. }
    ));
    assert!(!rejecting.stages.contains(&SaltStage::Publish));

    let pipeline = tritium_salt::SaltPipeline::resume(&spec, &root).expect("resume evidence");
    let receipt = pipeline.receipt();
    assert_eq!(receipt.quality().map(QualityEvidence::passed), Some(false));
    assert!(receipt.published().is_none());
    assert_eq!(
        receipt
            .stage_receipts()
            .last()
            .map(|record| record.accepted()),
        Some(false)
    );

    let mut must_not_run = FixtureDriver::default();
    assert!(matches!(
        SaltV2::reconcile(&spec, &root, &mut must_not_run),
        Err(tritium_salt::SaltError::QualityGateFailed { .. })
    ));
    assert!(must_not_run.stages.is_empty());
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn quality_evidence_for_another_package_cannot_authorize_publication() {
    let root = unique_temp_dir("wrong-quality-package");
    let spec = spec();
    let mut driver = FixtureDriver {
        wrong_quality_package: true,
        ..FixtureDriver::default()
    };

    assert!(matches!(
        SaltV2::reconcile(&spec, &root, &mut driver),
        Err(tritium_salt::SaltError::StageContractViolation {
            stage: SaltStage::Validate,
            ..
        })
    ));
    assert!(!driver.stages.contains(&SaltStage::Publish));
    let pipeline = tritium_salt::SaltPipeline::resume(&spec, &root).expect("resume evidence");
    assert!(pipeline.receipt().quality().is_none());
    assert!(pipeline.receipt().published().is_none());
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn inappropriate_evidence_is_rejected_without_promoting_it_to_the_receipt() {
    let root = unique_temp_dir("inappropriate-evidence");
    let spec = spec();
    let mut driver = FixtureDriver {
        publish_early: true,
        ..FixtureDriver::default()
    };

    assert!(matches!(
        SaltV2::reconcile(&spec, &root, &mut driver),
        Err(tritium_salt::SaltError::StageContractViolation {
            stage: SaltStage::Ingest,
            ..
        })
    ));
    let pipeline = tritium_salt::SaltPipeline::resume(&spec, &root).expect("resume evidence");
    assert!(pipeline.receipt().published().is_none());
    assert!(pipeline.receipt().physical().is_none());
    assert_eq!(pipeline.receipt().stage_receipts().len(), 1);
    assert!(!pipeline.receipt().stage_receipts()[0].accepted());
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn resident_core_rate_is_gated_before_validation() {
    let root = unique_temp_dir("resident-budget");
    let spec = spec();
    let mut driver = FixtureDriver {
        resident_core_bytes: 220,
        ..FixtureDriver::default()
    };

    assert!(matches!(
        SaltV2::reconcile(&spec, &root, &mut driver),
        Err(tritium_salt::SaltError::StageContractViolation {
            stage: SaltStage::Pack,
            ..
        })
    ));
    assert!(!driver.stages.contains(&SaltStage::Validate));
    let pipeline = tritium_salt::SaltPipeline::resume(&spec, &root).expect("resume evidence");
    assert!(pipeline.receipt().physical().is_none());
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn physical_report_converts_without_parallel_accounting() {
    let plane = SaltV2Plane::new(vec![0; 256], vec![half::f16::ZERO; 2]).expect("zero plane");
    let tensor = SaltV2Tensor::new(
        "w",
        vec![256],
        vec![SaltV2Tile::new(vec![plane]).expect("tile")],
    )
    .expect("tensor");
    let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).expect("package");
    let encoded = write_salt_v2_package(&package).expect("encode package");
    let package_bytes = encoded.bytes;
    let measured = MeasuredPackage::from_bytes(&package_bytes).expect("measure package");
    let report = PhysicalSizeReport::from_salt_v2_package_bytes(&package_bytes, 256, None)
        .expect("physical report");
    let package_id = ContentId::of_bytes(&package_bytes);

    let ledger = PhysicalLedger::from_physical_size_report(package_id, report)
        .expect("convert physical report");

    assert_eq!(ledger.package_id(), package_id);
    assert_eq!(ledger.transport_package_id(), measured.id().as_bytes());
    assert_eq!(ledger.logical_core_trits(), 256);
    assert_eq!(ledger.serialized_core_bytes(), 68);
    assert_eq!(ledger.metadata_bytes(), 100);
    assert_eq!(ledger.allocation_map_bits(), 2);
    assert_eq!(ledger.allocation_map_embedded_bits(), 2);
    assert_eq!(ledger.preserved_bytes(), 0);
    assert_eq!(ledger.resident_core_bytes(), 68);
    assert_eq!(ledger.resident_metadata_bytes(), 0);
    assert_eq!(ledger.resident_allocation_map_bits(), 2);
    assert_eq!(ledger.resident_allocation_map_embedded_bits(), 2);
    assert_eq!(ledger.resident_shadow_bytes(), 0);
    assert_eq!(ledger.resident_total_bytes(), 68);
    assert_eq!(ledger.package_bytes(), package_bytes.len() as u64);
}

#[test]
fn pipeline_state_v3_is_rejected_after_logical_trit_schema_migration() {
    let root = unique_temp_dir("state-v3-rejected");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    let state_path = root.join(spec.work_id().to_string()).join("state.bin");
    let mut bytes = fs::read(&state_path).expect("read state");
    bytes[4] = 3;
    fs::write(&state_path, bytes).expect("write v3 marker");

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "unsupported pipeline state version"
        ))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn forged_checksummed_resident_overflow_is_rejected_before_receipt_exposure() {
    let root = unique_temp_dir("forged-resident-overflow");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    rewrite_checksummed_state(&root, &spec, |state| {
        state["receipt"]["physical"]["resident_core_bytes"] = serde_json::json!(u64::MAX);
        state["receipt"]["physical"]["resident_metadata_bytes"] = serde_json::json!(1_u64);
    });

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "invalid loaded physical ledger"
        ))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn forged_checksummed_full_provenance_drift_is_rejected() {
    let root = unique_temp_dir("forged-provenance");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    rewrite_checksummed_state(&root, &spec, |state| {
        state["receipt"]["provenance"]["recipe_implementation"] =
            serde_json::json!("forged-backend");
    });

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "checkpoint provenance mismatch"
        ))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn forged_checksummed_physical_package_binding_is_rejected() {
    let root = unique_temp_dir("forged-package-binding");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    rewrite_checksummed_state(&root, &spec, |state| {
        state["receipt"]["physical"]["package_id"] =
            serde_json::Value::Array(vec![serde_json::json!(1_u8); 32]);
    });

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "physical ledger package does not match accepted pack"
        ))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn forged_checksummed_transport_package_binding_is_rejected() {
    let root = unique_temp_dir("forged-transport-binding");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    rewrite_checksummed_state(&root, &spec, |state| {
        state["receipt"]["physical"]["transport_package_id"] =
            serde_json::Value::Array(vec![serde_json::json!(1_u8); 32]);
    });

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "physical ledger does not match packed artifact"
        ))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn forged_checksummed_stage_acceptance_drift_is_rejected() {
    let root = unique_temp_dir("forged-stage-acceptance");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    rewrite_checksummed_state(&root, &spec, |state| {
        state["receipt"]["stages"][0]["accepted"] = serde_json::json!(false);
    });

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "receipt/run stage mismatch"
        ))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn forged_checksummed_quality_package_binding_is_rejected() {
    let root = unique_temp_dir("forged-quality-binding");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    rewrite_checksummed_state(&root, &spec, |state| {
        state["receipt"]["quality"]["package_id"] =
            serde_json::Value::Array(vec![serde_json::json!(1_u8); 32]);
    });

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "quality evidence binding mismatch"
        ))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn forged_checksummed_success_without_publication_is_rejected() {
    let root = unique_temp_dir("forged-publish-state");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    rewrite_checksummed_state(&root, &spec, |state| {
        state["receipt"]["published"] = serde_json::Value::Null;
    });

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "publication evidence is inconsistent with stage state"
        ))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn forged_checksummed_metric_constructor_bypass_is_rejected() {
    let root = unique_temp_dir("forged-metric");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    rewrite_checksummed_state(&root, &spec, |state| {
        state["receipt"]["metrics"][0]["metric"]["name"] = serde_json::json!("");
    });

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint("invalid loaded metric"))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn forged_checksummed_metric_output_placement_is_rejected() {
    let root = unique_temp_dir("forged-metric-placement");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    rewrite_checksummed_state(&root, &spec, |state| {
        state["receipt"]["metrics"][0]["output_id"] =
            serde_json::Value::Array(vec![serde_json::json!(1_u8); 32]);
    });

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "metric lacks stage output"
        ))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn forged_checksummed_gpu_total_overflow_is_rejected() {
    let root = unique_temp_dir("forged-gpu-overflow");
    let spec = spec();
    SaltV2::reconcile(&spec, &root, &mut FixtureDriver::default()).expect("initial reconcile");
    rewrite_checksummed_state(&root, &spec, |state| {
        state["receipt"]["hardware"][0]["usage"]["gpu_seconds"] = serde_json::json!(u64::MAX);
        state["receipt"]["hardware"][1]["usage"]["gpu_seconds"] = serde_json::json!(1_u64);
    });

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint("GPU seconds overflow"))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn forged_checksummed_failure_provenance_is_rejected() {
    let root = unique_temp_dir("forged-failure-binding");
    let spec = spec();
    let mut driver = FixtureDriver {
        fail_once: Some(SaltStage::Search),
        ..FixtureDriver::default()
    };
    SaltV2::reconcile(&spec, &root, &mut driver).expect_err("injected failure");
    rewrite_checksummed_state(&root, &spec, |state| {
        state["receipt"]["failures"][0]["message"] = serde_json::json!("forged failure");
    });

    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "run/failure receipt mismatch"
        ))
    ));
    fs::remove_dir_all(root).expect("clean fixture");
}

#[derive(Default)]
struct FixtureDriver {
    stages: Vec<SaltStage>,
    fail_once: Option<SaltStage>,
    reject_quality: bool,
    publish_early: bool,
    resident_core_bytes: u64,
    wrong_quality_package: bool,
}

impl SaltDriver for FixtureDriver {
    fn run_stage(&mut self, request: StageRequest<'_>) -> Result<StageOutput, DriverFailure> {
        let stage = request.stage();
        self.stages.push(stage);
        if self.fail_once == Some(stage) {
            self.fail_once = None;
            return Err(DriverFailure::new("injected", "fixture interruption", true)
                .expect("driver failure"));
        }
        let artifact_bytes = if stage == SaltStage::Pack {
            vec![0x5a; 230]
        } else {
            stage.as_str().as_bytes().to_vec()
        };
        let output_id = ContentId::of_bytes(&artifact_bytes);
        let mut output = StageOutput::new(output_id)
            .with_hardware(HardwareUsage::new("fixture-gpu", 1, 90, 1_024).expect("hardware"));
        if stage != SaltStage::Publish {
            let relative = format!("artifacts/{}.bin", stage.as_str());
            let path = request.work_dir().join(&relative);
            fs::create_dir_all(path.parent().expect("artifact parent"))
                .expect("artifact directory");
            fs::write(&path, &artifact_bytes).expect("write stage artifact");
            output = output.with_artifact(StageArtifact::new(relative).expect("stage artifact"));
        }
        if stage == SaltStage::Ingest && self.publish_early {
            output = output.with_published(
                PublishedArtifact::new(ContentId::of_bytes(b"early"), 230).expect("early artifact"),
            );
        }
        if stage == SaltStage::Pack {
            let resident_core_bytes = if self.resident_core_bytes == 0 {
                150
            } else {
                self.resident_core_bytes
            };
            output = output.with_physical(
                PhysicalLedger::new(
                    output_id,
                    *MeasuredPackage::from_bytes(&artifact_bytes)
                        .expect("measure fixture package")
                        .id()
                        .as_bytes(),
                    800,
                    LogicalTritCount::new(800).expect("logical trits"),
                    200,
                    resident_core_bytes,
                    10,
                    0,
                    0,
                    10,
                    0,
                    0,
                    20,
                    30,
                    40,
                    230,
                )
                .expect("physical ledger"),
            );
        }
        if stage == SaltStage::Validate {
            output = output
                .with_metric(
                    Metric::new("perplexity_delta", "ratio", 0.0)
                        .expect("metric")
                        .with_confidence_interval(-0.01, 0.01)
                        .expect("confidence interval"),
                )
                .with_quality(
                    QualityEvidence::new(
                        request.spec().evidence().id(),
                        if self.wrong_quality_package {
                            ContentId::of_bytes(b"another package")
                        } else {
                            accepted_pack_id(request.receipt())
                        },
                        ContentId::of_bytes(b"fixture harness"),
                        !self.reject_quality,
                        if self.reject_quality {
                            "strict non-inferiority failed"
                        } else {
                            "strict non-inferiority passed"
                        },
                    )
                    .expect("quality"),
                );
        }
        if stage == SaltStage::Publish {
            output = output.with_published(
                PublishedArtifact::new(accepted_pack_id(request.receipt()), 230)
                    .expect("published artifact"),
            );
        }
        Ok(output)
    }
}

fn accepted_pack_id(receipt: &tritium_salt::SaltReceipt) -> ContentId {
    receipt
        .stage_receipts()
        .iter()
        .find(|record| record.stage() == SaltStage::Pack && record.accepted())
        .map(|record| record.output_id())
        .expect("accepted pack receipt")
}

fn unique_temp_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "tritium-salt-{label}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ))
}

fn rewrite_checksummed_state(
    root: &std::path::Path,
    spec: &SaltSpec,
    mutate: impl FnOnce(&mut serde_json::Value),
) {
    const HEADER_BYTES: usize = 13;
    const CHECKSUM_BYTES: usize = 32;
    const HASH_CONTEXT: &str = "tritium salt pipeline state v1";

    let path = root.join(spec.work_id().to_string()).join("state.bin");
    let original = fs::read(&path).expect("read checkpoint");
    let checksum_offset = original.len() - CHECKSUM_BYTES;
    let mut state: serde_json::Value =
        serde_json::from_slice(&original[HEADER_BYTES..checksum_offset]).expect("state json");
    mutate(&mut state);
    let payload = serde_json::to_vec(&state).expect("encode forged state");
    let mut forged = Vec::new();
    forged.extend_from_slice(b"TSV2");
    forged.push(original[4]);
    forged.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    forged.extend_from_slice(&payload);
    let mut hasher = blake3::Hasher::new_derive_key(HASH_CONTEXT);
    hasher.update(&forged);
    forged.extend_from_slice(hasher.finalize().as_bytes());
    fs::write(path, forged).expect("write forged checkpoint");
}
