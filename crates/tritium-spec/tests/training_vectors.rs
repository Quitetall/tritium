//! Public schema tests for plan-0049 portable training vectors.

use tritium_spec::{
    TrainExecutionV1, TrainingOpManifestV1, TrainingOpManifestV2, TrainingOpManifestV3,
    TrainingToleranceV1, TrainingVectorBufferDataV1, TrainingVectorErrorCategoryV1,
    TrainingVectorExpectedV1, TrainingVectorSetV1, TrainingVectorSetV2, TrainingVectorSetV3,
};

#[test]
fn manifest_v3_adds_only_hestia_relax_to_v2() {
    let v2 = TrainingOpManifestV2::operations();
    let v3 = TrainingOpManifestV3::operations();
    assert_eq!(&v3[..v2.len()], v2);
    assert_eq!(v3.len(), v2.len() + 1);
    let hestia = v3.last().unwrap();
    assert_eq!(hestia.id, "graph.hestia_relax");
    assert!(hestia.forward);
    assert_eq!(hestia.vjp, tritium_spec::TrainingVjpV1::FirstOrder);
    assert!(!hestia.mutates);
    assert!(hestia.checkpoint_planes.is_empty());
    assert_eq!(
        TrainingOpManifestV3::parse_json(TrainingOpManifestV3::canonical_json()),
        Ok(TrainingOpManifestV3)
    );
}

#[test]
fn vector_v3_extends_v2_with_hestia_forward_vjp_and_errors() {
    let v2 = TrainingVectorSetV2::parse_json(TrainingVectorSetV2::canonical_json()).unwrap();
    let v3 = TrainingVectorSetV3::parse_json(TrainingVectorSetV3::canonical_json()).unwrap();
    assert_eq!(v3.manifest_digest(), TrainingOpManifestV3::digest());
    assert_eq!(v3.source_digest(), TrainingVectorSetV3::digest());
    assert_eq!(v3.cases().len(), v2.cases().len() + 5);
    assert_eq!(
        v3.cases()[..v2.cases().len()]
            .iter()
            .map(|case| case.case_id.as_str())
            .collect::<Vec<_>>(),
        v2.cases()
            .iter()
            .map(|case| case.case_id.as_str())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        v3.cases()[v2.cases().len()..]
            .iter()
            .map(|case| (case.case_id.as_str(), case.execution))
            .collect::<Vec<_>>(),
        [
            (
                "graph.hestia_relax.forward.basic",
                TrainExecutionV1::Forward,
            ),
            ("graph.hestia_relax.vjp.basic", TrainExecutionV1::Vjp),
            (
                "graph.hestia_relax.forward.invalid_tau",
                TrainExecutionV1::Forward,
            ),
            (
                "graph.hestia_relax.vjp.unrepresentable_tau",
                TrainExecutionV1::Vjp,
            ),
            (
                "graph.hestia_relax.vjp.invalid_shape",
                TrainExecutionV1::Vjp,
            ),
        ]
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn expected_invalid_request_preserves_duplicate_roles_for_backend_replay() {
    let manifest_digest = hex(&TrainingOpManifestV2::digest());
    let json = format!(
        r#"{{
  "schema_id": "tritium.training_vectors",
  "schema_version": 2,
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

    let vectors = TrainingVectorSetV2::parse_json(json.as_bytes()).unwrap();
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
    let manifest_digest = hex(&TrainingOpManifestV2::digest());
    let json = format!(
        r#"{{
  "schema_id": "tritium.training_vectors",
  "schema_version": 2,
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

    let vectors = TrainingVectorSetV2::parse_json(json.as_bytes()).unwrap();
    assert_eq!(vectors.manifest_digest(), TrainingOpManifestV2::digest());
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
    let bytes = TrainingVectorSetV2::canonical_json();
    assert_eq!(bytes.last(), Some(&b'\n'));
    let vectors = TrainingVectorSetV2::parse_json(bytes).unwrap();
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
            "graph.ste_surrogate.forward.basic",
            "graph.ste_surrogate.vjp.basic",
            "graph.salt_ste.forward.two_planes",
            "graph.salt_ste.vjp.identity",
            "graph.lsq_ste.forward.basic",
            "graph.lsq_ste.vjp.basic",
            "graph.fsq.forward.soft_round",
            "graph.fsq.vjp.soft_round",
            "graph.fsq.forward.hard_half_ties",
            "graph.fsq.vjp.hard_tanh",
            "graph.fsq.forward.stochastic_seed_7",
            "graph.fsq.forward.stochastic_seed_8",
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
            "graph.conv1d.forward.depthwise_asymmetric",
            "graph.conv1d.vjp.depthwise_asymmetric",
            "graph.conv2d.forward.depthwise_asymmetric",
            "graph.conv2d.vjp.depthwise_asymmetric",
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
            "graph.rope.forward.basic",
            "graph.rope.vjp.basic",
            "graph.attention.forward.causal_gqa",
            "graph.attention.forward.noncausal_gqa",
            "graph.attention.vjp.causal_gqa",
            "graph.attention.vjp.noncausal_mqa",
            "graph.attention.forward.multigroup_gqa",
            "graph.attention.vjp.multigroup_gqa",
            "loss.mse.forward.basic",
            "loss.mse.vjp.basic",
            "loss.softmax_cross_entropy.forward.basic",
            "loss.softmax_cross_entropy.vjp.basic",
            "loss.topk_knowledge_distillation.forward.duplicate_indices",
            "loss.topk_knowledge_distillation.vjp.duplicate_indices",
            "optimizer.sgd.step.basic",
            "optimizer.adamw.step.resumed_state",
            "optimizer.cautious_adamw.step.masked_state",
            "optimizer.int8_adamw.step.quiet_spike_blocks",
            "optimizer.muon.step.resumed_rectangular",
            "lifecycle.checkpoint.adamw_multileaf",
            "lifecycle.resume.adamw_multileaf",
            "lifecycle.export.salt_v2_package",
            "lifecycle.reload.salt_v2_package",
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
            "loss.topk_knowledge_distillation.forward.index_out_of_range",
            "graph.rope.forward.odd_head_dim",
            "loss.softmax_cross_entropy.forward.zero_rows",
            "loss.softmax_cross_entropy.forward.zero_cols",
            "graph.rope.forward.position_overflow",
            "graph.ste_surrogate.forward.shape_error",
            "graph.salt_ste.forward.zero_planes",
            "graph.lsq_ste.forward.shape_error",
            "graph.fsq.forward.invalid_levels",
            "graph.salt_ste.forward.zero_rows_huge_cols",
            "graph.salt_ste.forward.too_many_planes",
            "graph.salt_ste.forward.reordered_attributes",
            "graph.lsq_ste.vjp.zero_cols",
            "graph.fsq.forward.zero_len",
            "graph.fsq.forward.invalid_alpha",
            "graph.fsq.forward.unknown_ste",
            "graph.conv1d.forward.zero_groups",
            "graph.conv1d.forward.ragged_groups",
            "graph.conv1d.forward.axis_u32_overflow",
            "graph.conv1d.forward.scratch_limit",
            "graph.conv2d.forward.zero_groups",
            "graph.conv2d.forward.oversized_kernel",
            "optimizer.adamw.step.zero_step",
            "optimizer.cautious_adamw.step.invalid_beta1",
            "optimizer.int8_adamw.step.scale_shape",
            "optimizer.muon.step.zero_ns_steps",
            "lifecycle.checkpoint.negative_second_moment",
            "lifecycle.resume.bad_magic",
            "lifecycle.resume.negative_second_moment",
            "lifecycle.export.unknown_format",
            "lifecycle.reload.bad_magic",
            "graph.attention.forward.ragged_gqa",
            "graph.attention.forward.product_limit",
        ]
    );
    assert_eq!(vectors.source_digest(), TrainingVectorSetV2::digest());
    assert_eq!(
        hex(&TrainingVectorSetV2::digest()),
        "38b17f4c76c1d2f85cb35c713652a3d77627d02ba47933d2c8f31a88e0c594a7"
    );
}

#[test]
fn v1_corpus_remains_backward_readable_and_distinct_from_v2() {
    let vectors = TrainingVectorSetV1::parse_json(TrainingVectorSetV1::canonical_json()).unwrap();
    assert_eq!(vectors.cases().len(), 114);
    assert_eq!(vectors.manifest_digest(), TrainingOpManifestV1::digest());
    assert_eq!(TrainingVectorSetV2::canonical_json().last(), Some(&b'\n'));
    assert!(TrainingVectorSetV1::parse_json(TrainingVectorSetV2::canonical_json()).is_err());
    assert!(TrainingVectorSetV2::parse_json(TrainingVectorSetV1::canonical_json()).is_err());
}
