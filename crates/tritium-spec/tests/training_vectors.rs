//! Public schema tests for plan-0049 portable training vectors.

use tritium_spec::{
    TrainExecutionV1, TrainingOpManifestV1, TrainingToleranceV1, TrainingVectorBufferDataV1,
    TrainingVectorErrorCategoryV1, TrainingVectorExpectedV1, TrainingVectorSetV1,
};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn expected_invalid_request_preserves_duplicate_roles_for_backend_replay() {
    let manifest_digest = hex(&TrainingOpManifestV1::digest());
    let json = format!(
        r#"{{
  "schema_id": "tritium.training_vectors",
  "schema_version": 1,
  "manifest_digest": "{manifest_digest}",
  "cases": [
    {{
      "case_id": "graph.add.forward.duplicate_input",
      "operation": "graph.add",
      "execution": "forward",
      "tolerance": {{"kind": "bit_exact"}},
      "inputs": [
        {{"name": "left", "shape": [1], "data": {{"dtype": "f32", "bits": [1065353216]}}}},
        {{"name": "left", "shape": [1], "data": {{"dtype": "f32", "bits": [1073741824]}}}}
      ],
      "attributes": [],
      "expected": {{
        "kind": "error",
        "category": "invalid_request",
        "code": "duplicate_name.input.left",
        "outputs": [
          {{"name": "result", "shape": [1], "data": {{"dtype": "f32", "bits": [1123418112]}}}}
        ]
      }}
    }}
  ]
}}"#
    );

    let vectors = TrainingVectorSetV1::parse_json(json.as_bytes()).unwrap();
    assert!(matches!(
        &vectors.cases()[0].expected,
        TrainingVectorExpectedV1::Error {
            category: TrainingVectorErrorCategoryV1::InvalidRequest,
            code,
            outputs,
        } if code == "duplicate_name.input.left" && outputs.len() == 1
    ));
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
            .map(|case| case.case_id.as_str())
            .collect::<Vec<_>>(),
        [
            "graph.add.forward.basic",
            "graph.add.forward.zero",
            "graph.add.vjp.basic",
            "graph.dense_matmul.forward.basic",
            "graph.dense_matmul.vjp.basic",
            "graph.ternary_matmul.forward.basic",
            "graph.ternary_matmul.vjp.basic",
            "graph.transpose.forward.basic",
            "graph.transpose.vjp.basic",
            "graph.embedding_gather.forward.repeated",
            "graph.embedding_gather.vjp.repeated",
            "graph.slice_cols.forward.basic",
            "graph.slice_cols.vjp.basic",
            "graph.concat_cols.forward.basic",
            "graph.concat_cols.vjp.basic",
            "graph.mul.forward.basic",
            "graph.mul.vjp.basic",
            "graph.detach.forward.basic",
            "graph.detach.vjp.zero",
            "graph.scale_const.forward.basic",
            "graph.scale_const.vjp.basic",
            "graph.bias.forward.basic",
            "graph.bias.vjp.basic",
            "graph.relu2.forward.basic",
            "graph.relu2.vjp.basic",
            "graph.silu.forward.basic",
            "graph.silu.vjp.basic",
            "graph.rmsnorm.forward.basic",
            "graph.rmsnorm.vjp.basic",
            "graph.softmax.forward.basic",
            "graph.softmax.vjp.basic",
            "graph.causal_mask.forward.basic",
            "graph.causal_mask.vjp.basic",
            "loss.mse.forward.basic",
            "loss.mse.vjp.basic",
            "loss.softmax_cross_entropy.forward.basic",
            "loss.softmax_cross_entropy.vjp.basic",
            "optimizer.sgd.step.basic",
            "graph.add.forward.nonfinite",
            "graph.add.forward.duplicate_input",
            "graph.transpose.forward.shape_error",
            "graph.embedding_gather.forward.token_oob",
            "graph.slice_cols.forward.bounds_error",
            "graph.concat_cols.forward.shape_error",
            "graph.dense_matmul.forward.shape_error",
            "graph.ternary_matmul.forward.nonfinite_scale",
            "graph.rmsnorm.forward.shape_error",
            "graph.softmax.forward.shape_error",
            "graph.causal_mask.forward.shape_error",
            "loss.softmax_cross_entropy.forward.shape_error",
        ]
    );
    assert_eq!(vectors.source_digest(), TrainingVectorSetV1::digest());
    assert_eq!(
        hex(&TrainingVectorSetV1::digest()),
        "7adc028d7f05c839de3deb4e4e0a40929ecdf5a100c1f32dad024eb55c104527"
    );
}
