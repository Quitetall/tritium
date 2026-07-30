//! Canonical plan-0049 vectors replayed through the public CPU backend seam.

use tritium_spec::{
    TrainBackendError, TrainBackendV1, TrainCapabilitiesV1, TrainDTypeV1, TrainOutputV1,
    TrainReceiptV1, TrainRequestV1, TrainingVectorSetV2, train_output_digest_v1,
    train_request_digest_v1,
};
use tritium_testkit::{TrainingVectorFailureReason, run_training_conformance};
use tritium_train::CpuTrainBackendV1;

#[test]
fn canonical_tracer_vectors_pass_with_corpus_bound_receipts() {
    let vectors = TrainingVectorSetV2::parse_json(TrainingVectorSetV2::canonical_json()).unwrap();
    let report =
        std::panic::catch_unwind(|| run_training_conformance(&CpuTrainBackendV1::new(), &vectors))
            .expect("canonical valid/error corpus must never panic the CPU request path");
    assert!(report.is_ok(), "{:#?}", report.failed);
    assert_eq!(report.passed.len(), 117);
    let mut receipt_count = 0;
    let mut checked_conv2d_scratch = false;
    let mut checked_attention_forward_scratch = false;
    let mut checked_attention_vjp_scratch = false;
    let mut checked_int8_adam_scratch = false;
    let mut checked_muon_scratch = false;
    let mut checked_optimizer_scratch = [false; 4];
    let mut checked_lifecycle_scratch = [false; 4];
    for passed in report.passed {
        if let Some(receipt) = passed.receipt {
            receipt_count += 1;
            assert_eq!(receipt.vector_digest, Some(vectors.source_digest()));
            assert_eq!(receipt.manifest_digest, vectors.manifest_digest());
            if passed.case_id == "graph.conv2d.forward.depthwise_asymmetric" {
                assert_eq!(receipt.scratch_bytes, 168);
                checked_conv2d_scratch = true;
            }
            if passed.case_id == "graph.attention.forward.causal_gqa" {
                assert_eq!(receipt.scratch_bytes, 84);
                checked_attention_forward_scratch = true;
            }
            if passed.case_id == "graph.attention.vjp.causal_gqa" {
                assert_eq!(receipt.scratch_bytes, 168);
                checked_attention_vjp_scratch = true;
            }
            if passed.case_id == "optimizer.int8_adamw.step.quiet_spike_blocks" {
                assert_eq!(receipt.scratch_bytes, 2584);
                checked_int8_adam_scratch = true;
            }
            if passed.case_id == "optimizer.muon.step.resumed_rectangular" {
                assert_eq!(receipt.scratch_bytes, 144);
                checked_muon_scratch = true;
            }
            let optimizer_scratch = [
                ("optimizer.adamw.step.resumed_state", 32),
                ("optimizer.cautious_adamw.step.masked_state", 48),
                ("optimizer.int8_adamw.step.quiet_spike_blocks", 2584),
                ("optimizer.muon.step.resumed_rectangular", 144),
            ];
            for (index, &(case_id, scratch_bytes)) in optimizer_scratch.iter().enumerate() {
                if passed.case_id == case_id {
                    assert_eq!(receipt.scratch_bytes, scratch_bytes);
                    checked_optimizer_scratch[index] = true;
                }
            }
            let lifecycle_scratch = [
                ("lifecycle.checkpoint.adamw_multileaf", 153),
                ("lifecycle.resume.adamw_multileaf", 60),
                ("lifecycle.export.salt_v2_package", 132032),
                ("lifecycle.reload.salt_v2_package", 132032),
            ];
            for (index, &(case_id, scratch_bytes)) in lifecycle_scratch.iter().enumerate() {
                if passed.case_id == case_id {
                    assert_eq!(receipt.scratch_bytes, scratch_bytes);
                    checked_lifecycle_scratch[index] = true;
                }
            }
        }
    }
    assert_eq!(receipt_count, 72);
    assert!(checked_conv2d_scratch);
    assert!(checked_attention_forward_scratch);
    assert!(checked_attention_vjp_scratch);
    assert!(checked_int8_adam_scratch);
    assert!(checked_muon_scratch);
    assert!(
        checked_optimizer_scratch
            .into_iter()
            .all(core::convert::identity)
    );
    assert!(
        checked_lifecycle_scratch
            .into_iter()
            .all(core::convert::identity)
    );
    assert_eq!(
        CpuTrainBackendV1::new().capabilities().supported_operations,
        [
            "graph.ste_surrogate",
            "graph.salt_ste",
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
            "graph.conv1d",
            "graph.conv2d",
            "graph.relu2",
            "graph.silu",
            "graph.rmsnorm",
            "graph.softmax",
            "graph.causal_mask",
            "graph.rope",
            "graph.attention",
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
            "lifecycle.reload",
        ]
    );
}

struct CorruptReceiptBackend(CpuTrainBackendV1);

impl TrainBackendV1 for CorruptReceiptBackend {
    fn capabilities(&self) -> TrainCapabilitiesV1 {
        self.0.capabilities()
    }

    fn execute(
        &self,
        request: TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<TrainReceiptV1, TrainBackendError> {
        let mut receipt = self.0.execute(request, output)?;
        receipt.input_digest = [0; 32];
        Ok(receipt)
    }
}

#[test]
fn harness_rejects_fabricated_receipt_content_digests() {
    let vectors = TrainingVectorSetV2::parse_json(TrainingVectorSetV2::canonical_json()).unwrap();
    let report =
        run_training_conformance(&CorruptReceiptBackend(CpuTrainBackendV1::new()), &vectors);
    assert!(report.failed.iter().any(|failure| {
        failure.case_id == "graph.add.forward.basic"
            && failure.reason == TrainingVectorFailureReason::Receipt("input_digest".to_owned())
    }));
}

struct NoWriteBackend;

impl TrainBackendV1 for NoWriteBackend {
    fn capabilities(&self) -> TrainCapabilitiesV1 {
        CpuTrainBackendV1::new().capabilities()
    }

    fn execute(
        &self,
        request: TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<TrainReceiptV1, TrainBackendError> {
        let capabilities = self.capabilities();
        Ok(TrainReceiptV1 {
            backend_id: capabilities.backend_id,
            backend_build: "test.no-write".to_owned(),
            physical_device: None,
            manifest_digest: capabilities.manifest_digest,
            vector_digest: request.vector_digest,
            operation: request.operation.to_owned(),
            execution: request.execution,
            dtype: TrainDTypeV1::F32,
            limits: capabilities.limits,
            input_digest: train_request_digest_v1(&request),
            output_digest: train_output_digest_v1(output),
            peak_resident_bytes: 0,
            scratch_bytes: 0,
            host_transfers: 0,
            device_resident: true,
        })
    }
}

#[test]
fn harness_poisons_success_outputs_so_unwritten_zeroes_fail() {
    let vectors = TrainingVectorSetV2::parse_json(TrainingVectorSetV2::canonical_json()).unwrap();
    let report = run_training_conformance(&NoWriteBackend, &vectors);
    assert!(report.failed.iter().any(|failure| {
        failure.case_id == "graph.add.forward.zero"
            && matches!(
                failure.reason,
                TrainingVectorFailureReason::OutputMismatch { .. }
            )
    }));
}
