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
    assert_eq!(report.passed.len(), 67);
    assert_eq!(
        backend.capabilities().supported_operations,
        [
            "graph.ste_surrogate",
            "graph.lsq_ste",
            "graph.dense_matmul",
            "graph.ternary_matmul",
            "graph.embedding_gather",
            "graph.transpose",
            "graph.slice_cols",
            "graph.concat_cols",
            "graph.detach",
            "graph.scale_const",
            "graph.bias",
            "graph.add",
            "graph.mul",
            "graph.relu2",
            "graph.silu",
            "graph.causal_mask",
            "graph.rmsnorm",
            "graph.softmax",
            "loss.mse",
            "optimizer.sgd",
            "optimizer.adamw",
            "optimizer.cautious_adamw",
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
