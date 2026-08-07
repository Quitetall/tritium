use tritium_nn::{
    AppliedIntermediateGrowthReceipt, ArchSpec, DENSE_GROWTH_ORACLE_ALGORITHM_V1,
    DENSE_GROWTH_ORACLE_TOLERANCE, DenseLinear, FixedEmbeddingPolicy, GrowthPlanError,
    GrowthResultModelId, GrowthSourceModelId, GrowthTarget, Mlp, MlpKind, ModelConfig, ModelRunner,
    ModelWeights, Projection, ProjectionGeometry, ProjectionPlaneCounts, SwiGluMlp,
    SwiGluTrainingModel, TiedSwiGluTrainingModel, TokenEmbedding, TransformerBlock,
};

fn dense(rows: usize, cols: usize, marker: f32) -> Projection {
    Projection::Dense(DenseLinear::new_exact(vec![marker; rows * cols], rows, cols).unwrap())
}

fn config() -> ModelConfig {
    ModelConfig {
        arch: "llama".to_owned(),
        n_layers: 2,
        n_embd: 4,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 2,
        n_ff: 6,
        n_ctx: 32,
        rope_theta: 100_000.0,
        rms_eps: 1e-5,
    }
}

fn spec() -> ArchSpec {
    ArchSpec {
        mlp: MlpKind::SwiGlu,
        attn_sub_norm: false,
        ffn_sub_norm: false,
        qk_norm: false,
        qkv_bias: false,
        tied_embeddings: true,
    }
}

fn weights() -> ModelWeights {
    let layers = (0..2)
        .map(|layer| {
            let marker = 10.0 * layer as f32;
            TransformerBlock {
                attn_norm: vec![100.0 + marker; 4],
                q_proj: dense(4, 4, 1.0 + marker),
                k_proj: dense(2, 4, 2.0 + marker),
                v_proj: dense(2, 4, 3.0 + marker),
                o_proj: dense(4, 4, 4.0 + marker),
                attn_sub_norm: Vec::new(),
                q_bias: Vec::new(),
                k_bias: Vec::new(),
                v_bias: Vec::new(),
                q_norm: Vec::new(),
                k_norm: Vec::new(),
                ffn_norm: vec![200.0 + marker; 4],
                mlp: Mlp::SwiGlu(SwiGluMlp {
                    gate: dense(6, 4, 5.0 + marker),
                    up: dense(6, 4, 6.0 + marker),
                    down: dense(4, 6, 7.0 + marker),
                }),
            }
        })
        .collect();
    ModelWeights {
        token_embd: TokenEmbedding::from_dense(vec![0.5; 5 * 4], 5, 4).unwrap(),
        vocab: 5,
        n_embd: 4,
        layers,
        output_norm: vec![300.0; 4],
        lm_head: None,
    }
}

fn dense_training_logits(
    config: ModelConfig,
    model: &TiedSwiGluTrainingModel,
    tokens: &[u32],
) -> Vec<f32> {
    let dense = model.to_dense_weights().expect("dense oracle weights");
    let mut runner =
        ModelRunner::from_weights(config, dense, Box::new(tritium_cpu::CpuBackend::new()));
    let positions: Vec<_> = (0..tokens.len()).collect();
    runner.forward(tokens, &positions).expect("dense oracle")
}

fn untied_spec() -> ArchSpec {
    let mut spec = spec();
    spec.tied_embeddings = false;
    spec
}

fn qwen_attention_spec() -> ArchSpec {
    let mut spec = spec();
    spec.qkv_bias = true;
    spec.qk_norm = true;
    spec
}

fn qwen_attention_weights() -> ModelWeights {
    let mut weights = weights();
    for (layer_index, layer) in weights.layers.iter_mut().enumerate() {
        let marker = layer_index as f32 / 16.0;
        layer.q_bias = vec![0.125 + marker, -0.25, 0.375, -0.5];
        layer.k_bias = vec![0.25 + marker, -0.125];
        layer.v_bias = vec![-0.375, 0.5 + marker];
        layer.q_norm = vec![0.75 + marker, 1.25];
        layer.k_norm = vec![1.125, 0.875 + marker];
    }
    weights
}

fn untied_weights(marker: f32) -> ModelWeights {
    let mut weights = weights();
    weights.lm_head = Some(dense(5, 4, marker));
    weights
}

fn ramp_dense(rows: usize, cols: usize, start: f32) -> Projection {
    let values = (0..rows * cols).map(|index| start + index as f32).collect();
    Projection::Dense(DenseLinear::new_exact(values, rows, cols).unwrap())
}

fn widening_weights() -> ModelWeights {
    let mut weights = weights();
    for (layer_index, layer) in weights.layers.iter_mut().enumerate() {
        let Mlp::SwiGlu(mlp) = &mut layer.mlp else {
            unreachable!("test fixture is SwiGLU")
        };
        let layer_offset = 1_000.0 * layer_index as f32;
        mlp.gate = ramp_dense(6, 4, 100.0 + layer_offset);
        mlp.up = ramp_dense(6, 4, 200.0 + layer_offset);
        mlp.down = ramp_dense(4, 6, 300.0 + layer_offset);
    }
    weights
}

fn patterned_dense(rows: usize, cols: usize, seed: usize) -> Projection {
    let values = (0..rows * cols)
        .map(|index| {
            let centered = ((index * 7 + seed * 3) % 19) as f32 - 9.0;
            centered / 32.0
        })
        .collect();
    Projection::Dense(DenseLinear::new_exact(values, rows, cols).unwrap())
}

fn equivalence_weights() -> ModelWeights {
    let mut weights = weights();
    for (layer_index, layer) in weights.layers.iter_mut().enumerate() {
        layer.ffn_norm = vec![0.75 + 0.125 * layer_index as f32, 1.0, 1.25, 0.875];
        let Mlp::SwiGlu(mlp) = &mut layer.mlp else {
            unreachable!("test fixture is SwiGLU")
        };
        mlp.gate = patterned_dense(6, 4, 10 + layer_index);
        mlp.up = patterned_dense(6, 4, 20 + layer_index);
        mlp.down = patterned_dense(4, 6, 30 + layer_index);
    }
    weights
}

fn dense_swiglu_stack(model: &TiedSwiGluTrainingModel, input: &[f32]) -> Vec<f32> {
    let arch = model.architecture();
    assert_eq!(input.len() % arch.n_embd, 0);
    let mut hidden = input.to_vec();
    for layer_index in 0..arch.n_layers {
        let base = 1 + 7 * layer_index;
        let gate = &model.parameters()[base + 4];
        let up = &model.parameters()[base + 5];
        let down = &model.parameters()[base + 6];
        for row in hidden.chunks_exact_mut(arch.n_embd) {
            let mean_square =
                row.iter().map(|value| value * value).sum::<f32>() / arch.n_embd as f32;
            let inverse_rms = (mean_square + arch.rms_eps).sqrt().recip();
            let normalized: Vec<_> = row
                .iter()
                .zip(&arch.ffn_norms[layer_index])
                .map(|(&value, &norm)| value * inverse_rms * norm)
                .collect();
            let intermediate: Vec<_> = gate
                .master
                .chunks_exact(arch.n_embd)
                .zip(up.master.chunks_exact(arch.n_embd))
                .map(|(gate_row, up_row)| {
                    let gate_value = gate_row
                        .iter()
                        .zip(&normalized)
                        .map(|(&weight, &value)| weight * value)
                        .sum::<f32>();
                    let up_value = up_row
                        .iter()
                        .zip(&normalized)
                        .map(|(&weight, &value)| weight * value)
                        .sum::<f32>();
                    (gate_value / (1.0 + (-gate_value).exp())) * up_value
                })
                .collect();
            for (output, down_row) in row.iter_mut().zip(down.master.chunks_exact(arch.n_ff)) {
                *output += down_row
                    .iter()
                    .zip(&intermediate)
                    .map(|(&weight, &value)| weight * value)
                    .sum::<f32>();
            }
        }
    }
    hidden
}

#[test]
fn extraction_preserves_canonical_hf_parameter_order_names_and_shapes() {
    let model = TiedSwiGluTrainingModel::extract(&config(), &spec(), &weights()).unwrap();

    let names: Vec<&str> = model
        .parameters()
        .iter()
        .map(|parameter| parameter.name.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "model.layers.1.self_attn.v_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
        ]
    );
    let shapes: Vec<(usize, usize)> = model
        .parameters()
        .iter()
        .map(|parameter| (parameter.rows, parameter.cols))
        .collect();
    assert_eq!(
        shapes,
        [
            (5, 4),
            (4, 4),
            (2, 4),
            (2, 4),
            (4, 4),
            (6, 4),
            (6, 4),
            (4, 6),
            (4, 4),
            (2, 4),
            (2, 4),
            (4, 4),
            (6, 4),
            (6, 4),
            (4, 6),
        ]
    );
    assert_eq!(model.parameters()[0].master, vec![0.5; 20]);
    let master_markers: Vec<f32> = model
        .parameters()
        .iter()
        .skip(1)
        .map(|parameter| parameter.master[0])
        .collect();
    assert_eq!(
        master_markers,
        [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0
        ]
    );
    assert_eq!(
        model.architecture().attn_norms,
        [vec![100.0; 4], vec![110.0; 4]]
    );
    assert_eq!(
        model.architecture().ffn_norms,
        [vec![200.0; 4], vec![210.0; 4]]
    );
    assert_eq!(model.architecture().output_norm, vec![300.0; 4]);
    assert!(model.is_lm_head_tied());
}

#[test]
fn untied_extraction_appends_head_without_reordering_existing_parameters() {
    let tied = SwiGluTrainingModel::extract(&config(), &spec(), &weights()).unwrap();
    let untied =
        SwiGluTrainingModel::extract(&config(), &untied_spec(), &untied_weights(9.0)).unwrap();

    assert!(!untied.is_lm_head_tied());
    assert_eq!(untied.parameters().len(), tied.parameters().len() + 1);
    assert_eq!(
        &untied.parameters()[..tied.parameters().len()],
        tied.parameters()
    );
    let head = untied.parameters().last().expect("untied head");
    assert_eq!(head.name, "lm_head.weight");
    assert_eq!((head.rows, head.cols), (5, 4));
    assert_eq!(head.master, vec![9.0; 20]);
}

#[test]
fn parameter_masters_can_move_to_optimizer_without_losing_graph_metadata() {
    let mut model = TiedSwiGluTrainingModel::extract(&config(), &spec(), &weights()).unwrap();
    let expected_elements: Vec<_> = model
        .parameters()
        .iter()
        .map(|parameter| parameter.rows * parameter.cols)
        .collect();

    let masters = model.take_parameter_masters();

    assert_eq!(masters.len(), expected_elements.len());
    assert_eq!(
        masters.iter().map(Vec::len).collect::<Vec<_>>(),
        expected_elements
    );
    assert!(
        model
            .parameters()
            .iter()
            .all(|parameter| parameter.master.is_empty())
    );
    assert_eq!(
        model
            .parameters()
            .iter()
            .map(|parameter| parameter.elements())
            .collect::<Vec<_>>(),
        expected_elements
    );
}

#[test]
fn intermediate_growth_applies_one_mapping_to_every_layer_without_reordering_parameters() {
    let mut model =
        TiedSwiGluTrainingModel::extract(&config(), &spec(), &widening_weights()).unwrap();
    let names_before: Vec<_> = model
        .parameters()
        .iter()
        .map(|parameter| parameter.name.clone())
        .collect();

    let plan = model.widen_intermediate(9, 0).unwrap();

    assert_eq!(plan.source_indices(), [0, 1, 2, 3, 4, 5, 1, 0, 1]);
    assert_eq!(plan.replication_counts(), [2, 3, 1, 1, 1, 1]);
    let split_numerators = plan
        .split_numerators()
        .expect("actual growth uses unequal dyadic splits");
    let split_denominator = 1_u32 << plan.split_denominator_log2().expect("v2 denominator");
    for source in 0..plan.replication_counts().len() {
        let source_numerators: Vec<_> = plan
            .source_indices()
            .iter()
            .zip(split_numerators)
            .filter_map(|(&candidate, &numerator)| (candidate == source).then_some(numerator))
            .collect();
        assert_eq!(source_numerators.iter().sum::<u32>(), split_denominator);
        for (index, &numerator) in source_numerators.iter().enumerate() {
            assert!(
                source_numerators[index + 1..]
                    .iter()
                    .all(|&other| numerator != other)
            );
        }
    }
    assert_eq!(model.architecture().n_ff, 9);
    assert_eq!(
        model
            .parameters()
            .iter()
            .map(|parameter| parameter.name.clone())
            .collect::<Vec<_>>(),
        names_before
    );

    for layer_index in 0..2 {
        let base = 1 + 7 * layer_index;
        let gate = &model.parameters()[base + 4];
        let up = &model.parameters()[base + 5];
        let down = &model.parameters()[base + 6];
        assert_eq!((gate.rows, gate.cols), (9, 4));
        assert_eq!((up.rows, up.cols), (9, 4));
        assert_eq!((down.rows, down.cols), (4, 9));

        let layer_offset = 1_000.0 * layer_index as f32;
        assert_eq!(
            gate.master
                .chunks_exact(4)
                .map(|row| row[0])
                .collect::<Vec<_>>(),
            [
                100.0 + layer_offset,
                104.0 + layer_offset,
                108.0 + layer_offset,
                112.0 + layer_offset,
                116.0 + layer_offset,
                120.0 + layer_offset,
                104.0 + layer_offset,
                100.0 + layer_offset,
                104.0 + layer_offset,
            ]
        );
        assert_eq!(&gate.master[24..28], &gate.master[4..8]);
        assert_eq!(&gate.master[28..32], &gate.master[0..4]);
        assert_eq!(&up.master[24..28], &up.master[4..8]);
        assert_eq!(&up.master[28..32], &up.master[0..4]);

        let source_row_start = 300.0 + layer_offset;
        let expected_down_row: Vec<_> = plan
            .source_indices()
            .iter()
            .zip(split_numerators)
            .map(|(&source, &numerator)| {
                (source_row_start + source as f32) * (numerator as f32 / split_denominator as f32)
            })
            .collect();
        assert_eq!(&down.master[..9], expected_down_row.as_slice());
    }
}

#[test]
fn intermediate_growth_keeps_the_untied_head_final_and_unchanged() {
    let mut weights = widening_weights();
    weights.lm_head = Some(patterned_dense(5, 4, 91));
    let mut model =
        SwiGluTrainingModel::extract(&config(), &untied_spec(), &weights).expect("untied model");
    let head_before = model.parameters().last().expect("head").clone();

    model.widen_intermediate(9, 0x27).expect("widen");

    assert_eq!(model.parameters().last(), Some(&head_before));
    assert_eq!(model.parameters().last().unwrap().name, "lm_head.weight");
}

#[test]
fn intermediate_growth_preserves_the_whole_multilayer_swiglu_function_before_salt() {
    let mut model =
        TiedSwiGluTrainingModel::extract(&config(), &spec(), &equivalence_weights()).unwrap();
    let input = [
        0.25, -0.5, 0.75, -1.0, // token 0
        -0.125, 0.375, 0.625, -0.875, // token 1
        1.0, -0.25, -0.75, 0.5, // token 2
    ];
    let original = dense_swiglu_stack(&model, &input);

    model.widen_intermediate(11, 0x0027).unwrap();
    let widened = dense_swiglu_stack(&model, &input);

    assert_eq!(widened.len(), original.len());
    let max_absolute_error = original
        .iter()
        .zip(&widened)
        .map(|(&expected, &actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_absolute_error <= 2e-6,
        "Net2Wider changed the two-layer SwiGLU function by {max_absolute_error:e}"
    );
}

#[test]
fn intermediate_growth_rejects_narrowing_and_drained_masters_before_mutation() {
    let mut model = TiedSwiGluTrainingModel::extract(&config(), &spec(), &weights()).unwrap();
    let original = model.clone();

    let error = model.widen_intermediate(5, 7).unwrap_err();
    assert!(error.to_string().contains("6 -> 5"), "{error}");
    assert_eq!(model, original);

    let masters = model.take_parameter_masters();
    assert!(masters.iter().all(|master| !master.is_empty()));
    let error = model.widen_intermediate(9, 7).unwrap_err();
    assert!(error.to_string().contains("drained"), "{error}");
    assert_eq!(model.architecture().n_ff, 6);
    assert!(model.parameters().iter().all(|parameter| {
        parameter.master.is_empty()
            && parameter.elements() == parameter.rows.saturating_mul(parameter.cols)
    }));
}

#[test]
fn checked_growth_recomputes_source_identity_before_mutation_and_issues_receipt_after_apply() {
    let config = config();
    let spec = spec();
    let mut source = TiedSwiGluTrainingModel::extract(&config, &spec, &weights()).unwrap();
    let source_id = GrowthSourceModelId::from_training_model(&config, &spec, &source).unwrap();
    let geometry = ProjectionGeometry::new(
        usize::try_from(config.n_layers).unwrap(),
        usize::try_from(config.n_embd).unwrap(),
        usize::try_from(config.n_head * config.head_dim).unwrap(),
        usize::try_from(config.n_head_kv * config.head_dim).unwrap(),
        usize::try_from(config.n_ff).unwrap(),
        weights().vocab,
        ProjectionPlaneCounts::new(1, 1, 1, 1, 1, 1, 1).unwrap(),
        FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
    )
    .unwrap();
    let target =
        GrowthTarget::intermediate_at_least(geometry.core_coefficient_count(7).unwrap()).unwrap();
    let plan = geometry.plan(source_id, target, 0x27).unwrap();

    let mut other = TiedSwiGluTrainingModel::extract(&config, &spec, &widening_weights()).unwrap();
    let other_id = GrowthSourceModelId::from_training_model(&config, &spec, &other).unwrap();
    let other_before = other.clone();
    assert_eq!(
        plan.apply(&config, &spec, &mut other),
        Err(GrowthPlanError::SourceModelMismatch {
            expected: source_id,
            actual: other_id,
        })
    );
    assert_eq!(
        other, other_before,
        "identity rejection must precede mutation"
    );

    let oracle_source = source.clone();
    let receipt = plan.apply(&config, &spec, &mut source).unwrap();
    assert_eq!(source.architecture().n_ff, 7);
    assert_eq!(receipt.source_model_id(), source_id);
    let mut grown_config = config.clone();
    grown_config.n_ff = receipt.new_width();
    let expected_result_id =
        GrowthResultModelId::from_training_model(&grown_config, &spec, &source).unwrap();
    assert_eq!(receipt.result_model_id(), expected_result_id);
    receipt
        .validate_result_model(&grown_config, &spec, &source)
        .unwrap();

    let mut changed_mlp_weights = source.to_dense_weights().unwrap();
    let Mlp::SwiGlu(changed_mlp) = &mut changed_mlp_weights.layers[0].mlp else {
        unreachable!("fixture uses SwiGLU")
    };
    let Projection::Dense(changed_gate) = &mut changed_mlp.gate else {
        unreachable!("training reconstruction is dense")
    };
    changed_gate.weights[0] += 0.125;
    let changed_mlp_model =
        TiedSwiGluTrainingModel::extract(&grown_config, &spec, &changed_mlp_weights).unwrap();
    assert!(matches!(
        receipt.validate_result_model(&grown_config, &spec, &changed_mlp_model),
        Err(GrowthPlanError::ResultModelMismatch { .. })
    ));

    let mut changed_attention_weights = source.to_dense_weights().unwrap();
    let Projection::Dense(changed_query) = &mut changed_attention_weights.layers[0].q_proj else {
        unreachable!("training reconstruction is dense")
    };
    changed_query.weights[0] -= 0.125;
    let changed_attention_model =
        TiedSwiGluTrainingModel::extract(&grown_config, &spec, &changed_attention_weights).unwrap();
    assert!(matches!(
        receipt.validate_result_model(&grown_config, &spec, &changed_attention_model),
        Err(GrowthPlanError::ResultModelMismatch { .. })
    ));
    assert_eq!(receipt.target(), target);
    assert_eq!(
        receipt.resulting_core_coefficient_count(),
        plan.resulting_core_coefficient_count()
    );
    assert_eq!(receipt.seed(), 0x27);
    assert_eq!(receipt.old_width(), 6);
    assert_eq!(receipt.new_width(), 7);
    let replay = receipt.replay_plan().unwrap();
    assert_eq!(
        receipt.source_indices(),
        replay
            .source_indices()
            .iter()
            .map(|&value| u32::try_from(value).unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        receipt.replication_counts(),
        replay
            .replication_counts()
            .iter()
            .map(|&value| u32::try_from(value).unwrap())
            .collect::<Vec<_>>()
    );
    assert_eq!(receipt.split_numerators(), replay.split_numerators());
    let oracle = receipt.function_preservation();
    assert_eq!(oracle.algorithm(), DENSE_GROWTH_ORACLE_ALGORITHM_V1);
    assert_eq!(oracle.vocabulary(), 5);
    assert_eq!(oracle.context_length(), 32);
    assert_eq!(oracle.tokens(), [0, 4, 2, 3]);
    assert_eq!(oracle.logit_count(), 5);
    assert_eq!(oracle.tolerance(), DENSE_GROWTH_ORACLE_TOLERANCE);
    assert!(oracle.max_absolute_error() <= oracle.tolerance());
    assert!(!oracle.tokens().is_empty());
    assert!(oracle.logit_count() > 0);
    assert_ne!(oracle.source_logits_digest(), [0; 32]);
    assert_ne!(oracle.grown_logits_digest(), [0; 32]);
    let independent_source = dense_training_logits(config.clone(), &oracle_source, oracle.tokens());
    let independent_grown = dense_training_logits(grown_config.clone(), &source, oracle.tokens());
    let independent_max = independent_source
        .iter()
        .zip(independent_grown)
        .map(|(&before, after)| (before - after).abs())
        .fold(0.0_f32, f32::max);
    assert_eq!(
        oracle.max_absolute_error().to_bits(),
        independent_max.to_bits()
    );
    plan.validate_receipt(&receipt).unwrap();

    let canonical = receipt.canonical_bytes().unwrap();
    let digest = receipt.digest().unwrap();
    let reopened =
        AppliedIntermediateGrowthReceipt::from_canonical_bytes_verified(&canonical, digest)
            .unwrap();
    assert_eq!(reopened, receipt);
    plan.validate_receipt(&reopened).unwrap();

    let mut tampered = canonical.clone();
    let last = tampered.last_mut().expect("nonempty receipt");
    *last ^= 1;
    assert!(
        AppliedIntermediateGrowthReceipt::from_canonical_bytes_verified(&tampered, digest).is_err()
    );

    let mut bad_version = canonical.clone();
    bad_version[4..6].copy_from_slice(&2_u16.to_le_bytes());
    assert!(matches!(
        AppliedIntermediateGrowthReceipt::from_canonical_bytes(&bad_version),
        Err(GrowthPlanError::UnsupportedReceiptVersion(2))
    ));

    // Fixed v1 header: magic/version/reserved/source/result/targets/count, then seed.
    let mut seed_mapping_mismatch = canonical.clone();
    seed_mapping_mismatch[96..104].copy_from_slice(&0x28_u64.to_le_bytes());
    assert!(matches!(
        AppliedIntermediateGrowthReceipt::from_canonical_bytes(&seed_mapping_mismatch),
        Err(GrowthPlanError::ReceiptMismatch)
    ));

    let mut hostile_source_count = canonical.clone();
    hostile_source_count[116..120].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        AppliedIntermediateGrowthReceipt::from_canonical_bytes(&hostile_source_count),
        Err(GrowthPlanError::InvalidReceiptField(_))
    ));

    // This fixture has new_width=7 and old_width=6, placing oracle tolerance at 224.
    let mut relaxed_tolerance = canonical.clone();
    relaxed_tolerance[224..228].copy_from_slice(&1.0_f32.to_bits().to_le_bytes());
    assert!(matches!(
        AppliedIntermediateGrowthReceipt::from_canonical_bytes(&relaxed_tolerance),
        Err(GrowthPlanError::InvalidOracleEvidence(_))
    ));

    let mut trailing = canonical;
    trailing.push(0);
    assert!(AppliedIntermediateGrowthReceipt::from_canonical_bytes(&trailing).is_err());
}

#[test]
fn checked_growth_validates_descriptors_before_hashing_or_mutating() {
    let config = config();
    let spec = spec();
    let model = TiedSwiGluTrainingModel::extract(&config, &spec, &weights()).unwrap();
    let mut wrong_config = config.clone();
    wrong_config.n_ff += 1;

    assert!(matches!(
        GrowthSourceModelId::from_training_model(&wrong_config, &spec, &model),
        Err(GrowthPlanError::SourceDescriptorMismatch(
            "intermediate_width"
        ))
    ));

    let source_id = GrowthSourceModelId::from_training_model(&config, &spec, &model).unwrap();
    let geometry = ProjectionGeometry::new(
        usize::try_from(config.n_layers).unwrap(),
        usize::try_from(config.n_embd).unwrap(),
        usize::try_from(config.n_head * config.head_dim).unwrap(),
        usize::try_from(config.n_head_kv * config.head_dim).unwrap(),
        usize::try_from(config.n_ff).unwrap(),
        weights().vocab,
        ProjectionPlaneCounts::new(1, 1, 1, 1, 1, 1, 1).unwrap(),
        FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
    )
    .unwrap();
    let target =
        GrowthTarget::intermediate_at_least(geometry.core_coefficient_count(7).unwrap()).unwrap();
    let plan = geometry.plan(source_id, target, 0x27).unwrap();
    let mut application_model = model.clone();
    let before = application_model.clone();

    assert!(matches!(
        plan.apply(&wrong_config, &spec, &mut application_model),
        Err(GrowthPlanError::SourceDescriptorMismatch(
            "intermediate_width"
        ))
    ));
    assert_eq!(application_model, before);
}

#[test]
fn applied_growth_receipts_explicitly_distinguish_target_and_seed() {
    let config = config();
    let spec = spec();
    let source = TiedSwiGluTrainingModel::extract(&config, &spec, &weights()).unwrap();
    let source_id = GrowthSourceModelId::from_training_model(&config, &spec, &source).unwrap();
    let geometry = ProjectionGeometry::new(
        usize::try_from(config.n_layers).unwrap(),
        usize::try_from(config.n_embd).unwrap(),
        usize::try_from(config.n_head * config.head_dim).unwrap(),
        usize::try_from(config.n_head_kv * config.head_dim).unwrap(),
        usize::try_from(config.n_ff).unwrap(),
        weights().vocab,
        ProjectionPlaneCounts::new(1, 1, 1, 1, 1, 1, 1).unwrap(),
        FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
    )
    .unwrap();
    let base = geometry.core_coefficient_count(6).unwrap();
    let widened = geometry.core_coefficient_count(7).unwrap();
    assert!(widened > base + 1);
    let first_target = GrowthTarget::intermediate_at_least(widened - 1).unwrap();
    let second_target = GrowthTarget::intermediate_at_least(widened).unwrap();
    let first_plan = geometry.plan(source_id, first_target, 0x27).unwrap();
    let different_target_plan = geometry.plan(source_id, second_target, 0x27).unwrap();
    let different_seed_plan = geometry.plan(source_id, first_target, 0x28).unwrap();
    assert_eq!(first_plan.new_width(), different_target_plan.new_width());
    assert_eq!(first_plan.new_width(), different_seed_plan.new_width());

    let mut first_model = source.clone();
    let mut different_target_model = source.clone();
    let mut different_seed_model = source;
    let first = first_plan.apply(&config, &spec, &mut first_model).unwrap();
    let different_target = different_target_plan
        .apply(&config, &spec, &mut different_target_model)
        .unwrap();
    let different_seed = different_seed_plan
        .apply(&config, &spec, &mut different_seed_model)
        .unwrap();

    assert_eq!(first.target(), first_target);
    assert_eq!(different_target.target(), second_target);
    assert_eq!(first.seed(), 0x27);
    assert_eq!(different_seed.seed(), 0x28);
    assert_ne!(first.digest().unwrap(), different_target.digest().unwrap());
    assert_ne!(first.digest().unwrap(), different_seed.digest().unwrap());
    assert!(first_plan.validate_receipt(&different_target).is_err());
    assert!(first_plan.validate_receipt(&different_seed).is_err());
}

#[test]
fn identity_growth_receipts_bind_seed_even_when_mapping_is_identical() {
    let config = config();
    let spec = spec();
    let source = TiedSwiGluTrainingModel::extract(&config, &spec, &weights()).unwrap();
    let source_id = GrowthSourceModelId::from_training_model(&config, &spec, &source).unwrap();
    let geometry = ProjectionGeometry::new(
        usize::try_from(config.n_layers).unwrap(),
        usize::try_from(config.n_embd).unwrap(),
        usize::try_from(config.n_head * config.head_dim).unwrap(),
        usize::try_from(config.n_head_kv * config.head_dim).unwrap(),
        usize::try_from(config.n_ff).unwrap(),
        weights().vocab,
        ProjectionPlaneCounts::new(1, 1, 1, 1, 1, 1, 1).unwrap(),
        FixedEmbeddingPolicy::PreservedDense { tied_lm_head: true },
    )
    .unwrap();
    let target =
        GrowthTarget::intermediate_at_least(geometry.core_coefficient_count(6).unwrap()).unwrap();
    let first_plan = geometry.plan(source_id, target, 0x27).unwrap();
    let second_plan = geometry.plan(source_id, target, 0x28).unwrap();
    let mut first_model = source.clone();
    let mut second_model = source;
    let first = first_plan.apply(&config, &spec, &mut first_model).unwrap();
    let second = second_plan
        .apply(&config, &spec, &mut second_model)
        .unwrap();

    assert_eq!(first.source_indices(), second.source_indices());
    assert_eq!(first.replication_counts(), second.replication_counts());
    assert_eq!(first.split_numerators(), None);
    assert_eq!(second.split_numerators(), None);
    assert_eq!(
        first.result_model_id().as_bytes(),
        first.source_model_id().as_bytes()
    );
    assert_eq!(
        second.result_model_id().as_bytes(),
        second.source_model_id().as_bytes()
    );
    assert_ne!(first.seed(), second.seed());
    assert_ne!(first.digest().unwrap(), second.digest().unwrap());
    assert!(first_plan.validate_receipt(&second).is_err());
}

#[test]
fn official_smollm2_135m_360m_and_1_7b_configs_are_supported() {
    let cases = [
        (
            include_str!("fixtures/smollm2-135m-config.json"),
            (30, 576, 1536, 9, 3),
        ),
        (
            include_str!("fixtures/smollm2-360m-config.json"),
            (32, 960, 2560, 15, 5),
        ),
        (
            include_str!("fixtures/smollm2-1.7b-config.json"),
            (24, 2048, 8192, 32, 32),
        ),
    ];

    for (json, expected) in cases {
        let (config, spec) = ModelConfig::from_hf_config(json).unwrap();
        TiedSwiGluTrainingModel::validate_config(&config, &spec).unwrap();
        assert_eq!(
            (
                config.n_layers,
                config.n_embd,
                config.n_ff,
                config.n_head,
                config.n_head_kv,
            ),
            expected
        );
    }
}

#[test]
fn config_validation_accepts_standard_qwen_attention_constants() {
    TiedSwiGluTrainingModel::validate_config(&config(), &qwen_attention_spec()).unwrap();
}

#[test]
fn config_validation_accepts_tied_and_untied_lm_heads() {
    TiedSwiGluTrainingModel::validate_config(&config(), &spec()).unwrap();
    TiedSwiGluTrainingModel::validate_config(&config(), &untied_spec()).unwrap();
}

#[test]
fn extraction_rejects_hidden_bias_qk_norm_and_untied_weight_mismatches() {
    let mut packed_embedding = weights();
    packed_embedding.token_embd = TokenEmbedding::from_packed_salt(
        (0..5)
            .map(|_| tritium_format::PackedSaltRow::new(4, Vec::new()).unwrap())
            .collect(),
        5,
        4,
    )
    .unwrap();
    let error =
        TiedSwiGluTrainingModel::extract(&config(), &spec(), &packed_embedding).unwrap_err();
    assert!(error.to_string().contains("latent fp32"), "{error}");

    let mut transposed_embedding = weights();
    transposed_embedding.token_embd = TokenEmbedding::from_dense(vec![0.5; 20], 4, 5).unwrap();
    let error =
        TiedSwiGluTrainingModel::extract(&config(), &spec(), &transposed_embedding).unwrap_err();
    assert!(error.to_string().contains("rows"), "{error}");

    let mut biased = weights();
    biased.layers[0].q_bias = vec![0.0; 4];
    let error = TiedSwiGluTrainingModel::extract(&config(), &spec(), &biased).unwrap_err();
    assert!(error.to_string().contains("QKV bias"), "{error}");

    let mut qk_normalized = weights();
    qk_normalized.layers[0].q_norm = vec![1.0; 2];
    let error = TiedSwiGluTrainingModel::extract(&config(), &spec(), &qk_normalized).unwrap_err();
    assert!(error.to_string().contains("QK norm"), "{error}");

    let error = TiedSwiGluTrainingModel::extract(&config(), &qwen_attention_spec(), &weights())
        .unwrap_err();
    assert!(error.to_string().contains("q_proj.bias"), "{error}");

    let mut malformed_qwen = qwen_attention_weights();
    malformed_qwen.layers[1].k_norm.pop();
    let error =
        TiedSwiGluTrainingModel::extract(&config(), &qwen_attention_spec(), &malformed_qwen)
            .unwrap_err();
    assert!(error.to_string().contains("k_norm.weight"), "{error}");

    let mut non_finite_qwen = qwen_attention_weights();
    non_finite_qwen.layers[0].q_bias[0] = f32::NAN;
    let error =
        TiedSwiGluTrainingModel::extract(&config(), &qwen_attention_spec(), &non_finite_qwen)
            .unwrap_err();
    assert!(error.to_string().contains("non-finite"), "{error}");

    let error =
        TiedSwiGluTrainingModel::extract(&config(), &spec(), &untied_weights(9.0)).unwrap_err();
    assert!(
        error.to_string().contains("requires no separate"),
        "{error}"
    );

    let error =
        TiedSwiGluTrainingModel::extract(&config(), &untied_spec(), &weights()).unwrap_err();
    assert!(error.to_string().contains("requires a separate"), "{error}");

    let mut wrong_head = untied_weights(9.0);
    wrong_head.lm_head = Some(dense(4, 4, 9.0));
    let error =
        TiedSwiGluTrainingModel::extract(&config(), &untied_spec(), &wrong_head).unwrap_err();
    assert!(error.to_string().contains("lm_head.weight"), "{error}");
}

#[test]
fn extraction_and_dense_reconstruction_preserve_qwen_attention_constants() {
    let config = config();
    let spec = qwen_attention_spec();
    let source = qwen_attention_weights();
    let model = TiedSwiGluTrainingModel::extract(&config, &spec, &source).unwrap();

    assert_eq!(model.parameters().len(), 15);
    assert_eq!(model.architecture().attention_constants.len(), 2);
    assert_eq!(
        model.architecture().attention_constants[0].q_bias,
        source.layers[0].q_bias
    );
    assert_eq!(
        model.architecture().attention_constants[1].k_norm,
        source.layers[1].k_norm
    );

    let reconstructed = model.to_dense_weights().unwrap();
    for (expected, actual) in source.layers.iter().zip(&reconstructed.layers) {
        assert_eq!(actual.q_bias, expected.q_bias);
        assert_eq!(actual.k_bias, expected.k_bias);
        assert_eq!(actual.v_bias, expected.v_bias);
        assert_eq!(actual.q_norm, expected.q_norm);
        assert_eq!(actual.k_norm, expected.k_norm);
    }

    let tokens = [0, 3, 4, 1];
    let positions = [0, 1, 2, 3];
    let mut source_runner = ModelRunner::from_weights(
        config.clone(),
        source,
        Box::new(tritium_cpu::CpuBackend::new()),
    );
    let mut reconstructed_runner = ModelRunner::from_weights(
        config,
        reconstructed,
        Box::new(tritium_cpu::CpuBackend::new()),
    );
    assert_eq!(
        source_runner.forward(&tokens, &positions).unwrap(),
        reconstructed_runner.forward(&tokens, &positions).unwrap()
    );
}

#[cfg(feature = "cuda")]
fn cuda_backend_or_skip(test: &str) -> Option<tritium_cuda::CudaBackend> {
    match tritium_cuda::CudaBackend::new(0) {
        Ok(backend) => Some(backend),
        Err(error) => {
            eprintln!("skipping {test}: CUDA device unavailable: {error}");
            None
        }
    }
}

#[cfg(feature = "cuda")]
fn upload_resident_parameters(
    backend: &tritium_cuda::CudaBackend,
    model: &TiedSwiGluTrainingModel,
) -> Vec<tritium_cuda::train::DeviceTensor> {
    model
        .parameters()
        .iter()
        .map(|parameter| {
            tritium_cuda::train::DeviceTensor::upload(backend, &parameter.master)
                .expect("upload resident parameter")
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn upload_packed_parameters(
    backend: &tritium_cuda::CudaBackend,
    model: &TiedSwiGluTrainingModel,
) -> Vec<tritium_cuda::train::DevicePackedSaltWeight> {
    model
        .parameters()
        .iter()
        .map(|parameter| {
            tritium_cuda::train::DevicePackedSaltWeight::from_host(
                backend,
                &parameter.master,
                parameter.rows,
                parameter.cols,
                2,
            )
            .expect("upload packed parameter")
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn hestia_test_fixture() -> (ModelConfig, TiedSwiGluTrainingModel) {
    let mut config = config();
    config.n_layers = 1;
    let mut source = weights();
    source.layers.truncate(1);
    source.token_embd = TokenEmbedding::from_dense(
        (0..source.vocab * source.n_embd)
            .map(|index| (((index * 5 + 3) % 17) as f32 - 8.0) / 16.0)
            .collect(),
        source.vocab,
        source.n_embd,
    )
    .unwrap();
    source.output_norm = vec![0.875, 1.125, 0.75, 1.25];
    let layer = &mut source.layers[0];
    layer.attn_norm = vec![1.0, 0.75, 1.25, 0.875];
    layer.ffn_norm = vec![0.875, 1.125, 1.0, 0.75];
    layer.q_proj = patterned_dense(4, 4, 1);
    layer.k_proj = patterned_dense(2, 4, 2);
    layer.v_proj = patterned_dense(2, 4, 3);
    layer.o_proj = patterned_dense(4, 4, 4);
    let Mlp::SwiGlu(mlp) = &mut layer.mlp else {
        unreachable!("test fixture is SwiGLU")
    };
    mlp.gate = patterned_dense(6, 4, 5);
    mlp.up = patterned_dense(6, 4, 6);
    mlp.down = patterned_dense(4, 6, 7);
    let model = TiedSwiGluTrainingModel::extract(&config, &spec(), &source).unwrap();
    (config, model)
}

#[cfg(feature = "cuda")]
fn cpu_hestia_logits(
    config: &ModelConfig,
    model: &TiedSwiGluTrainingModel,
    masters: &[Vec<f32>],
    scales: &[Vec<f32>],
    temperatures: &[f32],
    tokens: &[u32],
) -> Vec<f32> {
    use tritium_train::ops::hestia::hestia_forward;

    let arch = model.architecture();
    let relaxed: Vec<Vec<f32>> = model
        .parameters()
        .iter()
        .zip(masters)
        .zip(scales)
        .zip(temperatures)
        .map(|(((parameter, master), scale), &tau)| {
            hestia_forward(master, scale, &[tau], parameter.rows, parameter.cols)
        })
        .collect();
    let dense = |index: usize| {
        let parameter = &model.parameters()[index];
        Projection::Dense(
            DenseLinear::new_exact(relaxed[index].clone(), parameter.rows, parameter.cols).unwrap(),
        )
    };
    let build_weights = || {
        let mut layers = Vec::with_capacity(arch.n_layers);
        for layer_index in 0..arch.n_layers {
            let base = 1 + 7 * layer_index;
            layers.push(TransformerBlock {
                attn_norm: arch.attn_norms[layer_index].clone(),
                q_proj: dense(base),
                k_proj: dense(base + 1),
                v_proj: dense(base + 2),
                o_proj: dense(base + 3),
                attn_sub_norm: Vec::new(),
                q_bias: Vec::new(),
                k_bias: Vec::new(),
                v_bias: Vec::new(),
                q_norm: Vec::new(),
                k_norm: Vec::new(),
                ffn_norm: arch.ffn_norms[layer_index].clone(),
                mlp: Mlp::SwiGlu(SwiGluMlp {
                    gate: dense(base + 4),
                    up: dense(base + 5),
                    down: dense(base + 6),
                }),
            });
        }
        ModelWeights {
            token_embd: TokenEmbedding::from_dense(relaxed[0].clone(), arch.vocab, arch.n_embd)
                .unwrap(),
            vocab: arch.vocab,
            n_embd: arch.n_embd,
            layers,
            output_norm: arch.output_norm.clone(),
            lm_head: None,
        }
    };
    let mut logits = Vec::with_capacity(tokens.len() * arch.vocab);
    for end in 1..=tokens.len() {
        let mut runner = ModelRunner::from_weights(
            config.clone(),
            build_weights(),
            Box::new(tritium_cpu::CpuBackend::new()),
        );
        let positions: Vec<_> = (0..end).collect();
        logits.extend(runner.forward(&tokens[..end], &positions).unwrap());
    }
    logits
}

#[cfg(feature = "cuda")]
fn mean_cross_entropy(logits: &[f32], target: &[f32], rows: usize, cols: usize) -> f32 {
    logits
        .chunks_exact(cols)
        .zip(target.chunks_exact(cols))
        .map(|(row, target_row)| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp = max
                + row
                    .iter()
                    .map(|value| (*value - max).exp())
                    .sum::<f32>()
                    .ln();
            target_row
                .iter()
                .zip(row)
                .map(|(&probability, &logit)| probability * (log_sum_exp - logit))
                .sum::<f32>()
        })
        .sum::<f32>()
        / rows as f32
}

#[cfg(feature = "cuda")]
#[test]
fn hestia_device_forward_matches_cpu_oracle_and_master_gradients() {
    use tritium_cuda::train::{
        CheckpointPolicy, DeviceTape, DeviceTensor, DeviceTrainParam, DeviceTrainer,
        DeviceTrainerWeightStorage,
    };
    use tritium_nn::hestia_device_forward;
    use tritium_train::{AdamW, ops::ste::absmean_scale_per_row};

    let Some(backend) =
        cuda_backend_or_skip("hestia_device_forward_matches_cpu_oracle_and_master_gradients")
    else {
        return;
    };
    let (config, model) = hestia_test_fixture();
    let parameter_specs: Vec<_> = model
        .parameters()
        .iter()
        .map(|parameter| DeviceTrainParam {
            master: &parameter.master,
            rows: parameter.rows,
            cols: parameter.cols,
            salt_planes: 2,
            optimizer: AdamW::new(1e-3),
        })
        .collect();
    let mut trainer = DeviceTrainer::new_with_weight_storage(
        &backend,
        &parameter_specs,
        DeviceTrainerWeightStorage::Packed,
    )
    .unwrap();
    let mut packed: Vec<_> = (0..model.parameters().len())
        .map(|index| trainer.packed_weight(index).unwrap())
        .collect();
    let masters: Vec<_> = (0..model.parameters().len())
        .map(|index| trainer.master_tensor(index).unwrap())
        .collect();
    let host_masters: Vec<_> = model
        .parameters()
        .iter()
        .map(|parameter| parameter.master.clone())
        .collect();
    let scales: Vec<_> = model
        .parameters()
        .iter()
        .map(|parameter| absmean_scale_per_row(&parameter.master, parameter.rows, parameter.cols))
        .collect();
    let temperatures: Vec<_> = model
        .parameters()
        .iter()
        .enumerate()
        .map(|(index, _)| 0.55 + index as f32 * 0.025)
        .collect();
    let tokens = [0_i32, 3];
    let token_ids = [0_u32, 3];
    let arch = model.architecture();

    let foreign_master = DeviceTensor::upload(&backend, &host_masters[0]).unwrap();
    let foreign_packed = tritium_cuda::train::DevicePackedSaltWeight::from_device_master(
        &backend,
        &foreign_master,
        model.parameters()[0].rows,
        model.parameters()[0].cols,
        2,
    )
    .unwrap();
    let original_packed = core::mem::replace(&mut packed[0], foreign_packed);
    let mut rejected_tape = DeviceTape::new(&backend, arch.vocab.max(arch.n_ff)).unwrap();
    let error = hestia_device_forward(
        &mut rejected_tape,
        &model,
        &masters,
        &packed,
        &temperatures,
        &tokens,
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("different resident master"),
        "{error}"
    );
    packed[0] = original_packed;

    let mut tape = DeviceTape::new_with_checkpoint_policy(
        &backend,
        arch.vocab.max(arch.n_ff),
        CheckpointPolicy::SqrtDepth(arch.n_layers),
    )
    .unwrap();
    let forward =
        hestia_device_forward(&mut tape, &model, &masters, &packed, &temperatures, &tokens)
            .unwrap();
    let actual_logits = tape.value(forward.logits).unwrap();
    let expected_logits = cpu_hestia_logits(
        &config,
        &model,
        &host_masters,
        &scales,
        &temperatures,
        &token_ids,
    );
    let forward_error = actual_logits
        .iter()
        .zip(&expected_logits)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        forward_error <= 2e-4,
        "HESTIA forward error {forward_error:e}"
    );

    let mut target = vec![0.0; tokens.len() * arch.vocab];
    target[1] = 1.0;
    target[arch.vocab + 4] = 1.0;
    let device_target = DeviceTensor::upload(&backend, &target).unwrap();
    let gradients = tape
        .xent_backward_device(
            forward.logits,
            &device_target,
            tokens.len(),
            arch.vocab,
            &forward.master_leaves,
        )
        .unwrap();

    let epsilon = 2e-3_f32;
    let mut max_gradient_error = 0.0_f32;
    for (parameter_index, parameter) in model.parameters().iter().enumerate() {
        let actual = gradients.download(&backend, parameter_index).unwrap();
        for element_index in 0..parameter.elements() {
            let mut plus = host_masters.clone();
            plus[parameter_index][element_index] += epsilon;
            let plus_loss = mean_cross_entropy(
                &cpu_hestia_logits(&config, &model, &plus, &scales, &temperatures, &token_ids),
                &target,
                tokens.len(),
                arch.vocab,
            );
            let mut minus = host_masters.clone();
            minus[parameter_index][element_index] -= epsilon;
            let minus_loss = mean_cross_entropy(
                &cpu_hestia_logits(&config, &model, &minus, &scales, &temperatures, &token_ids),
                &target,
                tokens.len(),
                arch.vocab,
            );
            let expected = (plus_loss - minus_loss) / (2.0 * epsilon);
            max_gradient_error = max_gradient_error.max((actual[element_index] - expected).abs());
        }
    }
    assert!(
        max_gradient_error <= 3e-3,
        "HESTIA master-gradient error {max_gradient_error:e}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn resident_qwen_attention_constants_match_cpu_and_packed_path_runs() {
    use tritium_cuda::train::{DeviceTape, DeviceTensor};
    use tritium_nn::{packed_device_forward, resident_device_forward};

    let Some(backend) =
        cuda_backend_or_skip("resident_qwen_attention_constants_match_cpu_and_packed_path_runs")
    else {
        return;
    };
    let config = config();
    let spec = qwen_attention_spec();
    let source = qwen_attention_weights();
    let model = TiedSwiGluTrainingModel::extract(&config, &spec, &source).unwrap();
    let tokens = [0_i32, 3, 4, 1];
    let token_ids: Vec<u32> = tokens
        .iter()
        .map(|token| u32::try_from(*token).unwrap())
        .collect();
    let positions = [0, 1, 2, 3];

    let mut cpu_runner =
        ModelRunner::from_weights(config, source, Box::new(tritium_cpu::CpuBackend::new()));
    let expected = cpu_runner.forward(&token_ids, &positions).unwrap();

    let resident = upload_resident_parameters(&backend, &model);
    let resident_refs: Vec<_> = resident.iter().collect();
    let arch = model.architecture();
    let mut resident_tape =
        DeviceTape::new(&backend, arch.vocab.max(arch.n_ff).max(tokens.len())).unwrap();
    let resident_forward =
        resident_device_forward(&mut resident_tape, &model, &resident_refs, &tokens).unwrap();
    let actual = resident_tape.value(resident_forward.logits).unwrap();
    let max_error = actual
        .iter()
        .zip(&expected)
        .map(|(&actual, &expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_error <= 2e-4, "CUDA Qwen constants error {max_error:e}");

    let packed = upload_packed_parameters(&backend, &model);
    let mut packed_tape =
        DeviceTape::new(&backend, arch.vocab.max(arch.n_ff).max(tokens.len())).unwrap();
    let packed_forward = packed_device_forward(&mut packed_tape, &model, &packed, &tokens).unwrap();
    let packed_logits = packed_tape.value(packed_forward.logits).unwrap();
    assert_eq!(packed_logits.len(), tokens.len() * arch.vocab);
    assert!(packed_logits.iter().all(|value| value.is_finite()));
    let target = DeviceTensor::upload(
        &backend,
        &vec![1.0 / arch.vocab as f32; tokens.len() * arch.vocab],
    )
    .unwrap();
    let gradients = packed_tape
        .xent_backward_device(
            packed_forward.logits,
            &target,
            tokens.len(),
            arch.vocab,
            &packed_forward.master_leaves,
        )
        .unwrap();
    for (index, parameter) in model.parameters().iter().enumerate() {
        let gradient = gradients.download(&backend, index).unwrap();
        assert_eq!(gradient.len(), parameter.elements());
        assert!(gradient.iter().all(|value| value.is_finite()));
    }
}

#[cfg(feature = "cuda")]
#[test]
fn resident_forward_borrows_canonical_leaves_and_backpropagates_tied_model() {
    use tritium_cuda::train::{
        CheckpointPolicy, DeviceTape, DeviceTensor, DeviceTrainParam, DeviceTrainer,
    };
    use tritium_nn::resident_device_forward;
    use tritium_train::AdamW;

    let Some(backend) = cuda_backend_or_skip(
        "resident_forward_borrows_canonical_leaves_and_backpropagates_tied_model",
    ) else {
        return;
    };
    let model = TiedSwiGluTrainingModel::extract(&config(), &spec(), &weights()).unwrap();
    let parameter_specs: Vec<_> = model
        .parameters()
        .iter()
        .map(|parameter| DeviceTrainParam {
            master: &parameter.master,
            rows: parameter.rows,
            cols: parameter.cols,
            salt_planes: 2,
            optimizer: AdamW::new(1e-3),
        })
        .collect();
    let mut trainer = DeviceTrainer::new(&backend, &parameter_specs).expect("resident trainer");
    trainer
        .prepare_quantized()
        .expect("prepare resident SALT weights");
    let resident_refs: Vec<_> = (0..model.parameters().len())
        .map(|index| trainer.quantized(index).expect("prepared resident weight"))
        .collect();
    let tokens = [0_i32, 1, 4];
    let arch = model.architecture();
    let ones_max = arch.vocab.max(arch.n_ff).max(tokens.len());
    let mut tape = DeviceTape::new_with_checkpoint_policy(
        &backend,
        ones_max,
        CheckpointPolicy::SqrtDepth(arch.n_layers),
    )
    .expect("resident checkpointed tape");

    let forward = resident_device_forward(&mut tape, &model, &resident_refs, &tokens)
        .expect("resident forward");

    assert_eq!(forward.master_leaves.len(), model.parameters().len());
    let mut unique_leaves = forward.master_leaves.clone();
    unique_leaves.sort_unstable();
    unique_leaves.dedup();
    assert_eq!(unique_leaves.len(), model.parameters().len());
    let logits = tape
        .value(forward.logits)
        .expect("download resident logits");
    assert_eq!(logits.len(), tokens.len() * arch.vocab);
    assert!(logits.iter().all(|value| value.is_finite()));

    let target = DeviceTensor::upload(
        &backend,
        &vec![1.0 / arch.vocab as f32; tokens.len() * arch.vocab],
    )
    .expect("upload target");
    let gradients = tape
        .xent_backward_device(
            forward.logits,
            &target,
            tokens.len(),
            arch.vocab,
            &forward.master_leaves,
        )
        .expect("resident backward");
    assert_eq!(gradients.len(), model.parameters().len());
    for (index, parameter) in model.parameters().iter().enumerate() {
        let gradient = gradients
            .download(&backend, index)
            .expect("download resident gradient");
        assert_eq!(gradient.len(), parameter.elements(), "{}", parameter.name);
        assert!(
            gradient.iter().all(|value| value.is_finite()),
            "{} produced a non-finite gradient",
            parameter.name
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn resident_forward_uses_appended_untied_head() {
    use tritium_cuda::train::DeviceTape;
    use tritium_nn::resident_device_forward;

    let Some(backend) = cuda_backend_or_skip("resident_forward_uses_appended_untied_head") else {
        return;
    };
    let tokens = [0_i32, 1, 4];

    let mut zero_head_weights = weights();
    zero_head_weights.lm_head = Some(dense(5, 4, 0.0));
    let zero_head =
        SwiGluTrainingModel::extract(&config(), &untied_spec(), &zero_head_weights).unwrap();
    let zero_resident = upload_resident_parameters(&backend, &zero_head);
    let zero_refs: Vec<_> = zero_resident.iter().collect();
    let arch = zero_head.architecture();
    let mut zero_tape =
        DeviceTape::new(&backend, arch.vocab.max(arch.n_ff).max(tokens.len())).unwrap();
    let zero_forward = resident_device_forward(&mut zero_tape, &zero_head, &zero_refs, &tokens)
        .expect("zero-head forward");
    let zero_logits = zero_tape.value(zero_forward.logits).expect("zero logits");
    assert!(zero_logits.iter().all(|value| *value == 0.0));

    let mut patterned_head_weights = weights();
    patterned_head_weights.lm_head = Some(patterned_dense(5, 4, 77));
    let patterned =
        SwiGluTrainingModel::extract(&config(), &untied_spec(), &patterned_head_weights).unwrap();
    let patterned_resident = upload_resident_parameters(&backend, &patterned);
    let patterned_refs: Vec<_> = patterned_resident.iter().collect();
    let mut patterned_tape =
        DeviceTape::new(&backend, arch.vocab.max(arch.n_ff).max(tokens.len())).unwrap();
    let patterned_forward =
        resident_device_forward(&mut patterned_tape, &patterned, &patterned_refs, &tokens)
            .expect("patterned-head forward");
    let patterned_logits = patterned_tape
        .value(patterned_forward.logits)
        .expect("patterned logits");

    assert_eq!(
        patterned_forward.master_leaves.len(),
        patterned.parameters().len()
    );
    assert!(patterned_logits.iter().all(|value| value.is_finite()));
    assert!(patterned_logits.iter().any(|value| value.abs() > 1e-6));
}

#[cfg(feature = "cuda")]
#[test]
fn packed_forward_uses_appended_untied_head() {
    use tritium_cuda::train::DeviceTape;
    use tritium_nn::packed_device_forward;

    let Some(backend) = cuda_backend_or_skip("packed_forward_uses_appended_untied_head") else {
        return;
    };
    let tokens = [0_i32, 1, 4];

    let mut zero_head_weights = weights();
    zero_head_weights.lm_head = Some(dense(5, 4, 0.0));
    let zero_head =
        SwiGluTrainingModel::extract(&config(), &untied_spec(), &zero_head_weights).unwrap();
    let zero_packed = upload_packed_parameters(&backend, &zero_head);
    let arch = zero_head.architecture();
    let mut zero_tape =
        DeviceTape::new(&backend, arch.vocab.max(arch.n_ff).max(tokens.len())).unwrap();
    let zero_forward = packed_device_forward(&mut zero_tape, &zero_head, &zero_packed, &tokens)
        .expect("zero packed-head forward");
    let zero_logits = zero_tape.value(zero_forward.logits).expect("zero logits");
    assert!(zero_logits.iter().all(|value| *value == 0.0));

    let mut patterned_head_weights = weights();
    patterned_head_weights.lm_head = Some(patterned_dense(5, 4, 77));
    let patterned =
        SwiGluTrainingModel::extract(&config(), &untied_spec(), &patterned_head_weights).unwrap();
    let patterned_packed = upload_packed_parameters(&backend, &patterned);
    let mut patterned_tape =
        DeviceTape::new(&backend, arch.vocab.max(arch.n_ff).max(tokens.len())).unwrap();
    let patterned_forward =
        packed_device_forward(&mut patterned_tape, &patterned, &patterned_packed, &tokens)
            .expect("patterned packed-head forward");
    let patterned_logits = patterned_tape
        .value(patterned_forward.logits)
        .expect("patterned logits");

    assert_eq!(
        patterned_forward.master_leaves.len(),
        patterned.parameters().len()
    );
    assert!(patterned_logits.iter().all(|value| value.is_finite()));
    assert!(patterned_logits.iter().any(|value| value.abs() > 1e-6));
}

#[cfg(feature = "cuda")]
#[test]
fn resident_forward_rejects_invalid_inputs_and_parameter_lengths() {
    use tritium_cuda::train::{DeviceTape, DeviceTensor};
    use tritium_nn::{TrainingAdapterError, resident_device_forward};

    let Some(backend) =
        cuda_backend_or_skip("resident_forward_rejects_invalid_inputs_and_parameter_lengths")
    else {
        return;
    };
    let model = TiedSwiGluTrainingModel::extract(&config(), &spec(), &weights()).unwrap();
    let resident = upload_resident_parameters(&backend, &model);
    let resident_refs: Vec<_> = resident.iter().collect();

    let mut tape = DeviceTape::new(&backend, model.architecture().vocab).expect("device tape");
    let expected_parameters = model.parameters().len();
    let got_parameters = expected_parameters - 1;
    let error = resident_device_forward(&mut tape, &model, &resident_refs[..got_parameters], &[0])
        .unwrap_err();
    assert_eq!(
        error,
        TrainingAdapterError::TensorShape {
            name: "resident parameter count".to_owned(),
            expected: expected_parameters,
            got: got_parameters,
        }
    );

    let expected_embedding_elements = model.parameters()[0].elements();
    let got_embedding_elements = expected_embedding_elements - 1;
    let short_embedding = DeviceTensor::upload(
        &backend,
        &model.parameters()[0].master[..got_embedding_elements],
    )
    .expect("upload short embedding");
    let short_refs: Vec<_> = core::iter::once(&short_embedding)
        .chain(resident.iter().skip(1))
        .collect();
    let error = resident_device_forward(&mut tape, &model, &short_refs, &[0]).unwrap_err();
    assert_eq!(
        error,
        TrainingAdapterError::TensorShape {
            name: "resident model.embed_tokens.weight".to_owned(),
            expected: expected_embedding_elements,
            got: got_embedding_elements,
        }
    );

    let arch = model.architecture();
    for (tokens, message) in [
        (Vec::new(), "training sequence must be non-empty"),
        (vec![0; arch.n_ctx + 1], "exceeds max_position_embeddings"),
        (vec![-1], "outside the vocabulary"),
        (vec![arch.vocab as i32], "outside the vocabulary"),
    ] {
        let error =
            resident_device_forward(&mut tape, &model, &resident_refs, &tokens).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }

    if let Ok(other_backend) = tritium_cuda::CudaBackend::new(1) {
        let foreign_embedding = DeviceTensor::upload(&other_backend, &model.parameters()[0].master)
            .expect("upload foreign embedding");
        let foreign_refs: Vec<_> = core::iter::once(&foreign_embedding)
            .chain(resident.iter().skip(1))
            .collect();
        let mut context_tape =
            DeviceTape::new(&backend, model.architecture().vocab).expect("context tape");
        let error =
            resident_device_forward(&mut context_tape, &model, &foreign_refs, &[0]).unwrap_err();
        assert!(
            error.to_string().contains("different CUDA context"),
            "{error}"
        );
    }
}
