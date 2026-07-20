//! Public schema tests for plan-0049 portable training vectors.

use tritium_spec::{
    TrainExecutionV1, TrainingOpManifestV1, TrainingToleranceV1, TrainingVectorBufferDataV1,
    TrainingVectorExpectedV1, TrainingVectorSetV1,
};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn parses_one_exact_forward_vector_bound_to_the_manifest() {
    let manifest_digest = hex(&TrainingOpManifestV1::digest());
    let json = format!(
        r#"{{
  "schema_id": "tritium.training_vectors",
  "schema_version": 1,
  "manifest_digest": "{manifest_digest}",
  "cases": [
    {{
      "case_id": "graph.add.forward.basic",
      "operation": "graph.add",
      "execution": "forward",
      "tolerance": {{"kind": "bit_exact"}},
      "inputs": [
        {{"name": "left", "shape": [2], "data": {{"dtype": "f32", "bits": [1065353216, 3221225472]}}}},
        {{"name": "right", "shape": [2], "data": {{"dtype": "f32", "bits": [1077936128, 1082130432]}}}}
      ],
      "attributes": [],
      "expected": {{
        "kind": "success",
        "outputs": [
          {{"name": "result", "shape": [2], "data": {{"dtype": "f32", "bits": [1082130432, 1073741824]}}}}
        ],
        "scratch_bytes_max": 0
      }}
    }}
  ]
}}"#
    );

    let vectors = TrainingVectorSetV1::parse_json(json.as_bytes()).unwrap();
    assert_eq!(vectors.manifest_digest(), TrainingOpManifestV1::digest());
    assert_eq!(vectors.cases().len(), 1);
    let case = &vectors.cases()[0];
    assert_eq!(case.case_id, "graph.add.forward.basic");
    assert_eq!(case.operation, "graph.add");
    assert_eq!(case.execution, TrainExecutionV1::Forward);
    assert_eq!(case.tolerance, TrainingToleranceV1::BitExact);
    assert_eq!(
        case.inputs[0].data,
        TrainingVectorBufferDataV1::F32Bits(vec![1065353216, 3221225472])
    );
    assert!(matches!(
        &case.expected,
        TrainingVectorExpectedV1::Success {
            outputs,
            scratch_bytes_max: 0,
        } if outputs.len() == 1
    ));
}

#[test]
fn canonical_partial_tracer_corpus_has_frozen_seed_order() {
    let bytes = TrainingVectorSetV1::canonical_json();
    assert_eq!(bytes.last(), Some(&b'\n'));
    let vectors = TrainingVectorSetV1::parse_json(bytes).unwrap();
    assert_eq!(
        vectors
            .cases()
            .iter()
            .map(|case| (case.operation.as_str(), case.execution))
            .collect::<Vec<_>>(),
        [
            ("graph.add", TrainExecutionV1::Forward),
            ("graph.add", TrainExecutionV1::Forward),
            ("graph.add", TrainExecutionV1::Vjp),
            ("optimizer.sgd", TrainExecutionV1::Step),
            ("graph.add", TrainExecutionV1::Forward),
        ]
    );
    assert_eq!(vectors.source_digest(), TrainingVectorSetV1::digest());
    assert_eq!(
        hex(&TrainingVectorSetV1::digest()),
        "4f5ab35ea2a77dec22cff12e134466fb0b86bfa882cd9b904dc64cc9dc751a78"
    );
}
