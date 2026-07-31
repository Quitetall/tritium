use std::path::Path;

use tritium_salt::{
    Qwen36AdmittedSource, Qwen36CampaignPreflight, Qwen36CampaignPreflightError,
    Qwen36SourceIdentityStatus, Qwen36SourceProof, Qwen36TensorWorkError, Qwen36TensorWorkStore,
};

const PINNED_METADATA_DIGEST: [u8; 32] = [
    0xad, 0xd3, 0x32, 0xd2, 0x3a, 0x10, 0x12, 0xaa, 0x1d, 0x77, 0x33, 0x9e, 0x04, 0x61, 0x30, 0x42,
    0xaa, 0x56, 0xfb, 0x7c, 0xab, 0x49, 0x08, 0xf7, 0x72, 0x6a, 0xc7, 0x8b, 0x78, 0xef, 0x2d, 0x8e,
];

#[test]
fn wrong_revision_fails_before_source_open() {
    let result = Qwen36CampaignPreflight::open(
        Path::new("this-path-must-not-be-opened"),
        "mutable-main-branch",
    );
    assert!(matches!(
        result,
        Err(Qwen36CampaignPreflightError::WrongRevision)
    ));
}

#[test]
#[ignore = "streams the complete local Qwen3.6-27B checkpoint"]
fn real_revision_declared_checkpoint_earns_candidate_receipt() {
    let model_dir = std::env::var_os("TRITIUM_QWEN36_27B_DIR")
        .expect("set TRITIUM_QWEN36_27B_DIR to the pinned local snapshot");
    let preflight =
        Qwen36CampaignPreflight::open(Path::new(&model_dir), tritium_nn::QWEN36_27B_REVISION)
            .expect("pinned campaign preflight");

    assert_eq!(preflight.receipt().repository(), "Qwen/Qwen3.6-27B");
    assert_eq!(
        preflight.receipt().revision(),
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    );
    assert_eq!(preflight.receipt().total_tensors(), 1_199);
    assert_eq!(preflight.receipt().included_tensors(), 866);
    assert_eq!(preflight.receipt().payload_bytes(), 55_562_855_904);
    assert_eq!(preflight.receipt().metadata_record_bytes(), 75_705);
    assert_eq!(
        preflight.receipt().metadata_digest(),
        &PINNED_METADATA_DIGEST
    );
    assert_eq!(
        preflight.receipt().identity_status(),
        Qwen36SourceIdentityStatus::MeasuredAwaitingOfficialRegistration
    );

    let coverage = preflight.receipt().coverage();
    assert_eq!(coverage.language().tensors(), 851);
    assert_eq!(coverage.mtp().tensors(), 15);
    assert_eq!(coverage.vision().tensors(), 333);
    assert_eq!(coverage.additive_ternary().tensors(), 506);
    assert_eq!(coverage.preserve_source().tensors(), 360);

    let language = preflight.receipt().language();
    assert_eq!(language.language_tensors(), 851);
    assert_eq!(language.language_matrices(), 498);
    assert_eq!(language.language_preserved_tensors(), 353);
    assert_eq!(language.deferred_mtp_tensors(), 15);
    assert_eq!(language.deferred_vision_tensors(), 333);

    assert_eq!(preflight.source_manifest().tensors().len(), 1_199);
    assert_eq!(
        preflight.receipt().source_model_id(),
        preflight.source_manifest().model_id()
    );

    // Canonicalize: macos temp_dir sits under the /var symlink and the store
    // rejects symlinked ancestors.
    let work_root = std::env::temp_dir()
        .canonicalize()
        .expect("canonicalize temp dir")
        .join(format!(
            "tritium-qwen36-real-admission-{}",
            std::process::id()
        ));
    let _ = std::fs::remove_dir_all(&work_root);
    let admitted =
        Qwen36AdmittedSource::admit(preflight, &work_root).expect("durable candidate admission");
    let proof_bytes = std::fs::read(admitted.proof_path()).expect("read durable proof");
    let reopened = Qwen36SourceProof::from_canonical_bytes(&proof_bytes).expect("reopen proof");
    assert_eq!(&reopened, admitted.proof());
    assert_eq!(reopened.proof_id().unwrap(), admitted.receipt().proof_id());
    assert_eq!(
        reopened.source_model_id(),
        admitted.receipt().source_model_id()
    );
    let work = Qwen36TensorWorkStore::open(&admitted).expect("open tensor workspace");
    let workspace = work
        .reconcile_preserved()
        .expect("persist exact preserved tensors");
    assert_eq!(workspace.summary().active_tensors(), 866);
    assert_eq!(workspace.summary().additive_required(), 506);
    assert_eq!(workspace.summary().preserved_tensors(), 360);
    assert_eq!(workspace.summary().preserved_payload_bytes(), 5_343_232);
    assert_eq!(
        workspace.identity_status(),
        Qwen36SourceIdentityStatus::MeasuredAwaitingOfficialRegistration
    );
    assert!(matches!(
        work.require_complete(),
        Err(Qwen36TensorWorkError::MissingAdditiveArtifacts {
            expected: 506,
            present: 0
        })
    ));
    let _ = std::fs::remove_dir_all(work_root);
}
