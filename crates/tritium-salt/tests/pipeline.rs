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
fn work_lock_rejects_concurrent_reconcile_before_driver_and_releases_on_drop() {
    use std::{sync::mpsc, thread};

    let root = unique_temp_dir("work-lock");
    let spec = spec();
    let holder_root = root.clone();
    let holder_spec = spec.clone();
    let (locked_tx, locked_rx) = mpsc::sync_channel(0);
    let (release_tx, release_rx) = mpsc::sync_channel(0);
    let holder = thread::spawn(move || {
        let pipeline = tritium_salt::SaltPipeline::start(&holder_spec, &holder_root)
            .expect("first process-local owner");
        locked_tx.send(()).expect("report held lock");
        release_rx.recv().expect("release requested");
        drop(pipeline);
    });

    locked_rx.recv().expect("lock acquired");
    let mut blocked_driver = FixtureDriver::default();
    assert!(matches!(
        SaltV2::reconcile(&spec, &root, &mut blocked_driver),
        Err(tritium_salt::SaltError::Checkpoint(
            "pipeline work item is already locked"
        ))
    ));
    assert!(blocked_driver.stages.is_empty());
    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "pipeline work item is already locked"
        ))
    ));

    release_tx.send(()).expect("release holder");
    holder.join().expect("holder thread");
    let mut resumed_driver = FixtureDriver::default();
    let receipt = SaltV2::reconcile(&spec, &root, &mut resumed_driver)
        .expect("lock released when owner dropped");
    assert!(receipt.published().is_some());
    assert_eq!(resumed_driver.stages.as_slice(), &SaltStage::ALL);
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn work_lock_survives_work_directory_replacement() {
    let root = unique_temp_dir("work-lock-replaced-directory");
    let spec = spec();
    let holder = tritium_salt::SaltPipeline::start(&spec, &root).expect("first owner");
    let work_dir = holder.work_dir().to_path_buf();
    let displaced = root.join("displaced-work-directory");

    fs::rename(&work_dir, &displaced).expect("move live work directory");
    fs::remove_dir_all(&displaced).expect("unlink moved work directory");
    fs::create_dir_all(&work_dir).expect("recreate work directory");

    assert!(matches!(
        tritium_salt::SaltPipeline::start(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "pipeline work item is already locked"
        ))
    ));
    assert!(matches!(
        tritium_salt::SaltPipeline::resume(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "pipeline work item is already locked"
        ))
    ));
    let mut blocked_driver = FixtureDriver::default();
    assert!(matches!(
        SaltV2::reconcile(&spec, &root, &mut blocked_driver),
        Err(tritium_salt::SaltError::Checkpoint(
            "pipeline work item is already locked"
        ))
    ));
    assert!(blocked_driver.stages.is_empty());

    drop(holder);
    let mut resumed_driver = FixtureDriver::default();
    let receipt = SaltV2::reconcile(&spec, &root, &mut resumed_driver)
        .expect("replacement work directory becomes available after drop");
    assert!(receipt.published().is_some());
    fs::remove_dir_all(root).expect("clean fixture");
}

#[cfg(unix)]
#[test]
fn forked_child_drop_does_not_release_parent_work_lock() {
    use std::{
        io::{Read, Write},
        os::unix::net::UnixStream,
        time::Duration,
    };

    let root = unique_temp_dir("work-lock-fork-child-drop");
    let spec = spec();
    let holder = tritium_salt::SaltPipeline::start(&spec, &root).expect("parent lock owner");
    let (mut parent_signal, mut child_signal) =
        UnixStream::pair().expect("create child teardown signal");

    // SAFETY: this test immediately confines the child to dropping its inherited
    // pipeline, one socket write, and `_exit`; the parent owns and reaps the PID.
    let child_pid = unsafe { raw_fork() };
    assert!(
        child_pid >= 0,
        "fork failed: {}",
        std::io::Error::last_os_error()
    );
    if child_pid == 0 {
        drop(parent_signal);
        drop(holder);
        let exit_code = if child_signal.write_all(b"dropped").is_ok() {
            0
        } else {
            1
        };
        // SAFETY: this is the fork child; `_exit` avoids running the inherited
        // test harness or any unrelated parent destructors.
        unsafe { raw_exit(exit_code) }
    }

    drop(child_signal);
    let mut child = RawForkChildGuard::new(child_pid);
    parent_signal
        .set_read_timeout(Some(Duration::from_secs(5)))
        .expect("bound fork child teardown wait");
    let mut signal = [0_u8; 7];
    parent_signal
        .read_exact(&mut signal)
        .expect("fork child dropped inherited pipeline");
    assert_eq!(&signal, b"dropped");
    assert_eq!(child.wait().expect("reap fork child"), 0);

    assert!(matches!(
        tritium_salt::SaltPipeline::start(&spec, &root),
        Err(tritium_salt::SaltError::Checkpoint(
            "pipeline work item is already locked"
        ))
    ));

    drop(holder);
    let resumed = tritium_salt::SaltPipeline::resume(&spec, &root)
        .expect("creator drop releases parent lock");
    drop(resumed);
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn work_lock_is_released_after_process_crash_and_running_stage_is_recovered() {
    use std::{
        process::{Command, Stdio},
        thread,
        time::{Duration, Instant},
    };

    let root = unique_temp_dir("work-lock-process");
    let spec = spec();
    let ready = root.join("child.ready");
    let mut child = ChildProcessGuard::new(
        Command::new(std::env::current_exe().expect("integration-test executable"))
            .arg("--exact")
            .arg("work_lock_process_crash_helper")
            .arg("--nocapture")
            .env("TRITIUM_SALT_LOCK_TEST_ROOT", &root)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn lock holder"),
    );

    let deadline = Instant::now() + Duration::from_secs(10);
    while !ready.exists() {
        if Instant::now() >= deadline {
            panic!("child did not enter a running locked stage");
        }
        thread::sleep(Duration::from_millis(5));
    }

    let mut blocked_driver = FixtureDriver::default();
    assert!(matches!(
        SaltV2::reconcile(&spec, &root, &mut blocked_driver),
        Err(tritium_salt::SaltError::Checkpoint(
            "pipeline work item is already locked"
        ))
    ));
    assert!(blocked_driver.stages.is_empty());

    child
        .kill_and_wait()
        .expect("terminate and reap lock holder");
    let mut recovery_driver = FixtureDriver::default();
    let receipt = SaltV2::reconcile(&spec, &root, &mut recovery_driver)
        .expect("crash releases lock and running stage recovers");
    let ingest = receipt
        .stage_receipts()
        .iter()
        .find(|record| record.stage() == SaltStage::Ingest && record.accepted())
        .expect("accepted recovered ingest");
    assert_eq!(ingest.attempt(), 2);
    assert_eq!(receipt.failures().len(), 1);
    assert_eq!(receipt.failures()[0].code(), "interrupted");
    fs::remove_dir_all(root).expect("clean fixture");
}

#[test]
fn work_lock_process_crash_helper() {
    let Some(root) = std::env::var_os("TRITIUM_SALT_LOCK_TEST_ROOT") else {
        return;
    };
    let root = PathBuf::from(root);
    let spec = spec();
    let mut pipeline = tritium_salt::SaltPipeline::start(&spec, &root).expect("child pipeline");
    let mut driver = CrashHoldingDriver {
        ready: root.join("child.ready"),
    };
    let _ = pipeline.advance(&mut driver);
    panic!("crash holder unexpectedly returned");
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
    drop(pipeline);

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
    drop(pipeline);
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
    drop(pipeline);
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
    drop(pipeline);
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

struct CrashHoldingDriver {
    ready: PathBuf,
}

struct ChildProcessGuard {
    child: Option<std::process::Child>,
}

#[cfg(unix)]
struct RawForkChildGuard {
    pid: Option<std::ffi::c_int>,
}

#[cfg(unix)]
impl RawForkChildGuard {
    fn new(pid: std::ffi::c_int) -> Self {
        assert!(pid > 0, "fork child PID must be positive");
        Self { pid: Some(pid) }
    }

    fn wait(&mut self) -> std::io::Result<std::ffi::c_int> {
        let pid = self.pid.expect("fork child is live");
        let status = wait_for_raw_child(pid)?;
        self.pid = None;
        Ok(status)
    }
}

#[cfg(unix)]
impl Drop for RawForkChildGuard {
    fn drop(&mut self) {
        if let Some(pid) = self.pid.take() {
            // SAFETY: `pid` is the positive PID returned by `fork` and remains
            // owned by this guard until a successful `waitpid`.
            let _ = unsafe { raw_kill(pid, RAW_SIGKILL) };
            let _ = wait_for_raw_child(pid);
        }
    }
}

#[cfg(unix)]
fn wait_for_raw_child(pid: std::ffi::c_int) -> std::io::Result<std::ffi::c_int> {
    loop {
        let mut status = 0;
        // SAFETY: `status` is writable for the call and `pid` is a child PID
        // returned by `fork`; options zero requests a blocking reap.
        let waited = unsafe { raw_waitpid(pid, &mut status, 0) };
        if waited == pid {
            return Ok(status);
        }
        if waited < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
    }
}

#[cfg(unix)]
const RAW_SIGKILL: std::ffi::c_int = 9;

#[cfg(unix)]
unsafe extern "C" {
    #[link_name = "fork"]
    fn raw_fork() -> std::ffi::c_int;
    #[link_name = "_exit"]
    fn raw_exit(status: std::ffi::c_int) -> !;
    #[link_name = "waitpid"]
    fn raw_waitpid(
        pid: std::ffi::c_int,
        status: *mut std::ffi::c_int,
        options: std::ffi::c_int,
    ) -> std::ffi::c_int;
    #[link_name = "kill"]
    fn raw_kill(pid: std::ffi::c_int, signal: std::ffi::c_int) -> std::ffi::c_int;
}

impl ChildProcessGuard {
    fn new(child: std::process::Child) -> Self {
        Self { child: Some(child) }
    }

    fn kill_and_wait(&mut self) -> std::io::Result<std::process::ExitStatus> {
        let child = self.child.as_mut().expect("child process is live");
        // The helper may already have terminated between the readiness signal
        // and teardown. Reaping is authoritative; a successful wait means no
        // child remains even if the best-effort kill raced with its exit.
        let _ = child.kill();
        let wait_result = child.wait();
        if wait_result.is_ok() {
            self.child = None;
        }
        wait_result
    }
}

impl Drop for ChildProcessGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl SaltDriver for CrashHoldingDriver {
    fn run_stage(&mut self, _request: StageRequest<'_>) -> Result<StageOutput, DriverFailure> {
        use std::{thread, time::Duration};

        fs::write(&self.ready, b"running").expect("signal running stage");
        loop {
            thread::sleep(Duration::from_secs(1));
        }
    }
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
