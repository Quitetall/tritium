#![cfg(feature = "cuda")]

use tritium_cuda::train::CudaTrainBackendV1;
use tritium_spec::{TrainBackendV1, TrainingVectorSetV1};
use tritium_testkit::run_supported_training_conformance;

#[test]
fn cuda_executes_every_vector_for_its_advertised_operations() {
    let vectors = TrainingVectorSetV1::parse_json(include_bytes!(
        "../../../spec/training/v1/vectors/v1.json"
    ))
    .expect("parse canonical training vectors");
    let backend = CudaTrainBackendV1::new(0).expect("open CUDA device 0");
    let report = run_supported_training_conformance(&backend, &vectors);
    assert!(
        report.is_ok(),
        "{} CUDA portable-training failures: {:?}",
        report.failed.len(),
        report.failed
    );
    assert_eq!(report.passed.len(), 72);
    assert_eq!(
        backend.capabilities().supported_operations,
        [
            "graph.ste_surrogate",
            "graph.lsq_ste",
            "graph.fsq",
            "graph.dense_matmul",
            "graph.ternary_matmul",
            "graph.transpose",
            "graph.embedding_gather",
            "graph.slice_cols",
            "graph.concat_cols",
            "graph.detach",
            "graph.scale_const",
            "graph.bias",
            "graph.add",
            "graph.mul",
            "graph.relu2",
            "graph.silu",
            "graph.rmsnorm",
            "graph.softmax",
            "graph.causal_mask",
            "graph.rope",
            "loss.mse",
            "loss.softmax_cross_entropy"
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
