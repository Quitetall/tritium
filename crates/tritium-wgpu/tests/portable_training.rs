#![cfg(feature = "wgpu")]

use tritium_spec::{TrainBackendV1, TrainingVectorSetV1};
use tritium_testkit::run_supported_training_conformance;
use tritium_wgpu::WgpuTrainBackendV1;

#[test]
fn wgpu_executes_every_vector_for_its_advertised_operations() {
    let vectors = TrainingVectorSetV1::parse_json(include_bytes!(
        "../../../spec/training/v1/vectors/v1.json"
    ))
    .expect("parse canonical training vectors");
    let backend = WgpuTrainBackendV1::new().expect("open native wgpu adapter");
    let report = run_supported_training_conformance(&backend, &vectors);
    assert!(
        report.is_ok(),
        "{} wgpu portable-training failures: {:?}",
        report.failed.len(),
        report.failed
    );
    assert_eq!(report.passed.len(), 9);
    assert_eq!(
        backend.capabilities().supported_operations,
        [
            "lifecycle.checkpoint",
            "lifecycle.resume",
            "lifecycle.export",
            "lifecycle.reload"
        ]
    );
    assert!(
        report
            .passed
            .iter()
            .filter_map(|case| case.receipt.as_ref())
            .all(|receipt| receipt
                .physical_device
                .as_deref()
                .is_some_and(|id| id.contains("NVIDIA")))
    );
}
