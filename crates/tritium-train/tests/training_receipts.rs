use tritium_spec::{TrainBackendV1, TrainingVectorSetV1};
use tritium_testkit::{
    TrainingReceiptBundleError, TrainingReceiptSourcePolicyV1, admit_training_receipts,
    render_development_training_capability_table, render_training_capability_table,
    seal_training_receipts,
};
use tritium_train::CpuTrainBackendV1;

fn vectors() -> TrainingVectorSetV1 {
    TrainingVectorSetV1::parse_json(include_bytes!("../../../spec/training/v1/vectors/v1.json"))
        .expect("canonical vectors")
}

#[test]
fn cpu_report_seals_reopens_and_generates_table() {
    let vectors = vectors();
    let backend = CpuTrainBackendV1::new();
    let sealed = seal_training_receipts(&backend, &vectors).expect("seal receipts");
    let repeated = seal_training_receipts(&backend, &vectors).expect("reseal receipts");
    assert_eq!(sealed, repeated);
    assert_eq!(sealed.digest(), *blake3::hash(sealed.bytes()).as_bytes());
    let admitted = admit_training_receipts(
        sealed.bytes(),
        &vectors,
        sealed.digest(),
        TrainingReceiptSourcePolicyV1::Development,
    )
    .expect("admit receipts");
    assert_eq!(admitted.backend_id(), backend.capabilities().backend_id);
    assert_eq!(admitted.operation_count(), 35);
    assert_eq!(admitted.case_count(), 114);
    assert_eq!(admitted.bundle_digest(), sealed.digest());

    assert_eq!(
        render_training_capability_table(std::slice::from_ref(&admitted)),
        Err(TrainingReceiptBundleError::DevelopmentEvidence)
    );
    let table = render_development_training_capability_table(&[admitted])
        .expect("render development table");
    assert!(table.starts_with("> **Development evidence only.**"));
    assert!(table.contains("| cpu.reference.v1 |"));
    assert!(table.contains("| 35 | 114 |"));
    assert!(table.ends_with('\n'));
}

#[test]
fn table_rejects_duplicate_identity() {
    let vectors = vectors();
    let backend = CpuTrainBackendV1::new();
    let sealed = seal_training_receipts(&backend, &vectors).expect("seal receipts");
    let admitted = admit_training_receipts(
        sealed.bytes(),
        &vectors,
        sealed.digest(),
        TrainingReceiptSourcePolicyV1::Development,
    )
    .expect("admit receipts");
    assert!(matches!(
        render_development_training_capability_table(&[admitted.clone(), admitted]),
        Err(TrainingReceiptBundleError::DuplicateBackend { .. })
    ));
}

#[test]
fn admission_rejects_noncanonical_unknown_and_nonresident_evidence() {
    let vectors = vectors();
    let backend = CpuTrainBackendV1::new();
    let sealed = seal_training_receipts(&backend, &vectors).expect("seal receipts");

    let mut noncanonical = sealed.bytes().to_vec();
    noncanonical.push(b'\n');
    assert_eq!(
        admit_training_receipts(
            &noncanonical,
            &vectors,
            *blake3::hash(&noncanonical).as_bytes(),
            TrainingReceiptSourcePolicyV1::Development,
        ),
        Err(TrainingReceiptBundleError::NonCanonical)
    );

    let mut value: serde_json::Value =
        serde_json::from_slice(sealed.bytes()).expect("parse sealed JSON");
    value["unexpected"] = serde_json::json!(true);
    let mut unknown = serde_json::to_vec_pretty(&value).expect("serialize mutation");
    unknown.push(b'\n');
    assert!(matches!(
        admit_training_receipts(
            &unknown,
            &vectors,
            *blake3::hash(&unknown).as_bytes(),
            TrainingReceiptSourcePolicyV1::Development,
        ),
        Err(TrainingReceiptBundleError::Json(_))
    ));

    let canonical = String::from_utf8(sealed.bytes().to_vec()).expect("UTF-8 receipt");
    let nonresident =
        canonical.replacen("\"device_resident\": true", "\"device_resident\": false", 1);
    assert!(matches!(
        admit_training_receipts(
            nonresident.as_bytes(),
            &vectors,
            *blake3::hash(nonresident.as_bytes()).as_bytes(),
            TrainingReceiptSourcePolicyV1::Development,
        ),
        Err(TrainingReceiptBundleError::Capabilities(_))
    ));

    let excessive_scratch =
        canonical.replacen("\"scratch_bytes\": 0", "\"scratch_bytes\": 999999999", 1);
    assert!(matches!(
        admit_training_receipts(
            excessive_scratch.as_bytes(),
            &vectors,
            *blake3::hash(excessive_scratch.as_bytes()).as_bytes(),
            TrainingReceiptSourcePolicyV1::Development,
        ),
        Err(TrainingReceiptBundleError::Receipt { .. })
    ));

    let false_residency = canonical.replacen(
        "\"peak_resident_bytes\": 52",
        "\"peak_resident_bytes\": 0",
        1,
    );
    assert!(matches!(
        admit_training_receipts(
            false_residency.as_bytes(),
            &vectors,
            *blake3::hash(false_residency.as_bytes()).as_bytes(),
            TrainingReceiptSourcePolicyV1::Development,
        ),
        Err(TrainingReceiptBundleError::Receipt { .. })
    ));

    assert_eq!(
        admit_training_receipts(
            sealed.bytes(),
            &vectors,
            [0; 32],
            TrainingReceiptSourcePolicyV1::Development,
        ),
        Err(TrainingReceiptBundleError::ContentDigest)
    );
}
