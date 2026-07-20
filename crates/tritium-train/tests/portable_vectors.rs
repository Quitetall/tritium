//! Canonical plan-0049 vectors replayed through the public CPU backend seam.

use tritium_spec::{TrainBackendV1, TrainingVectorSetV1};
use tritium_testkit::run_training_conformance;
use tritium_train::CpuTrainBackendV1;

#[test]
fn canonical_tracer_vectors_pass_with_corpus_bound_receipts() {
    let vectors = TrainingVectorSetV1::parse_json(TrainingVectorSetV1::canonical_json()).unwrap();
    let report = run_training_conformance(&CpuTrainBackendV1::new(), &vectors);
    assert!(report.is_ok(), "{:#?}", report.failed);
    assert_eq!(report.passed.len(), 4);
    let mut receipt_count = 0;
    for passed in report.passed {
        if let Some(receipt) = passed.receipt {
            receipt_count += 1;
            assert_eq!(receipt.vector_digest, Some(vectors.source_digest()));
            assert_eq!(receipt.manifest_digest, vectors.manifest_digest());
        }
    }
    assert_eq!(receipt_count, 3);
    assert_eq!(
        CpuTrainBackendV1::new().capabilities().supported_operations,
        ["graph.add", "optimizer.sgd"]
    );
}
