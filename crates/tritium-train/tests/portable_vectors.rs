//! Canonical plan-0049 vectors replayed through the public CPU backend seam.

use tritium_spec::{
    train_output_digest_v1, train_request_digest_v1, TrainBackendError, TrainBackendV1,
    TrainCapabilitiesV1, TrainDTypeV1, TrainOutputV1, TrainReceiptV1, TrainRequestV1,
    TrainingVectorSetV1,
};
use tritium_testkit::{run_training_conformance, TrainingVectorFailureReason};
use tritium_train::CpuTrainBackendV1;

#[test]
fn canonical_tracer_vectors_pass_with_corpus_bound_receipts() {
    let vectors = TrainingVectorSetV1::parse_json(TrainingVectorSetV1::canonical_json()).unwrap();
    let report = run_training_conformance(&CpuTrainBackendV1::new(), &vectors);
    assert!(report.is_ok(), "{:#?}", report.failed);
    assert_eq!(report.passed.len(), 93);
    let mut receipt_count = 0;
    let mut checked_conv2d_scratch = false;
    let mut checked_attention_forward_scratch = false;
    let mut checked_attention_vjp_scratch = false;
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
        }
    }
    assert_eq!(receipt_count, 59);
    assert!(checked_conv2d_scratch);
    assert!(checked_attention_forward_scratch);
    assert!(checked_attention_vjp_scratch);
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
            "optimizer.sgd",
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
    let vectors = TrainingVectorSetV1::parse_json(TrainingVectorSetV1::canonical_json()).unwrap();
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
    let vectors = TrainingVectorSetV1::parse_json(TrainingVectorSetV1::canonical_json()).unwrap();
    let report = run_training_conformance(&NoWriteBackend, &vectors);
    assert!(report.failed.iter().any(|failure| {
        failure.case_id == "graph.add.forward.zero"
            && matches!(
                failure.reason,
                TrainingVectorFailureReason::OutputMismatch { .. }
            )
    }));
}
