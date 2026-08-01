#![cfg(feature = "wgpu")]

use tritium_spec::{TrainBackendV1, TrainingVectorSetV2};
use tritium_testkit::run_supported_training_conformance;
use tritium_wgpu::WgpuTrainBackendV1;

#[test]
fn wgpu_executes_every_vector_for_its_advertised_operations() {
    let vectors = TrainingVectorSetV2::parse_json(include_bytes!(
        "../../../spec/training/v2/vectors/v2.json"
    ))
    .expect("parse canonical training vectors");
    let backend = WgpuTrainBackendV1::new().expect("open native wgpu adapter");
    backend
        .validate_dispatch_catalog()
        .expect("compile every shared WebGPU dispatch stage");
    let report = run_supported_training_conformance(&backend, &vectors);
    assert!(
        report.is_ok(),
        "{} wgpu portable-training failures: {:?}",
        report.failed.len(),
        report.failed
    );
    assert_eq!(report.passed.len(), 117);
    assert_eq!(
        backend.capabilities().supported_operations,
        [
            "graph.ste_surrogate",
            "graph.salt_ste",
            "graph.lsq_ste",
            "graph.fsq",
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
            "graph.conv1d",
            "graph.conv2d",
            "graph.attention",
            "graph.relu2",
            "graph.silu",
            "graph.causal_mask",
            "graph.rope",
            "graph.rmsnorm",
            "graph.softmax",
            "loss.mse",
            "loss.softmax_cross_entropy",
            "loss.topk_knowledge_distillation",
            "optimizer.sgd",
            "optimizer.adamw",
            "optimizer.cautious_adamw",
            "optimizer.int8_adamw",
            "optimizer.muon",
            "lifecycle.checkpoint",
            "lifecycle.resume",
            "lifecycle.export",
            "lifecycle.reload"
        ]
    );
    // Every receipt must name the physical adapter that executed the cases, and
    // it must be real silicon: a software rasterizer (llvmpipe / lavapipe /
    // SwiftShader) passes conformance while proving nothing about GPU execution.
    // Any hardware vendor is acceptable — this crate's point is portability, and
    // AMD RADV receipts from RDNA4 are as load-bearing as NVIDIA ones (the
    // previous NVIDIA-substring assert failed on every non-NVIDIA adapter by
    // construction; issue #1 finding 2).
    let mut receipts = 0usize;
    for receipt in report
        .passed
        .iter()
        .filter_map(|case| case.receipt.as_ref())
    {
        receipts += 1;
        let device = receipt
            .physical_device
            .as_deref()
            .expect("conformance receipt must record the physical device");
        let lower = device.to_lowercase();
        assert!(
            !["llvmpipe", "lavapipe", "swiftshader"]
                .iter()
                .any(|soft| lower.contains(soft)),
            "conformance receipt names a software rasterizer, not a GPU: {device}"
        );
    }
    assert!(receipts > 0, "no passed case carried a device receipt");
}
