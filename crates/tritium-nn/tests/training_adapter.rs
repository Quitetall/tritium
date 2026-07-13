use tritium_nn::{
    ArchSpec, DenseLinear, Mlp, MlpKind, ModelConfig, ModelWeights, Projection, SwiGluMlp,
    TiedSwiGluTrainingModel, TransformerBlock,
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
        token_embd: vec![0.5; 5 * 4],
        vocab: 5,
        n_embd: 4,
        layers,
        output_norm: vec![300.0; 4],
        lm_head: None,
    }
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
        assert_eq!(
            &down.master[..9],
            &[
                source_row_start / 2.0,
                (source_row_start + 1.0) / 3.0,
                source_row_start + 2.0,
                source_row_start + 3.0,
                source_row_start + 4.0,
                source_row_start + 5.0,
                (source_row_start + 1.0) / 3.0,
                source_row_start / 2.0,
                (source_row_start + 1.0) / 3.0,
            ]
        );
    }
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
fn config_validation_rejects_qkv_bias() {
    let mut unsupported = spec();
    unsupported.qkv_bias = true;
    let error = TiedSwiGluTrainingModel::validate_config(&config(), &unsupported).unwrap_err();
    assert!(error.to_string().contains("QKV bias"), "{error}");
}

#[test]
fn config_validation_rejects_qk_norm() {
    let mut unsupported = spec();
    unsupported.qk_norm = true;
    let error = TiedSwiGluTrainingModel::validate_config(&config(), &unsupported).unwrap_err();
    assert!(error.to_string().contains("QK norm"), "{error}");
}

#[test]
fn config_validation_rejects_untied_lm_head() {
    let mut unsupported = spec();
    unsupported.tied_embeddings = false;
    let error = TiedSwiGluTrainingModel::validate_config(&config(), &unsupported).unwrap_err();
    assert!(error.to_string().contains("tied embeddings"), "{error}");
}

#[test]
fn extraction_rejects_hidden_bias_qk_norm_and_untied_weight_mismatches() {
    let mut biased = weights();
    biased.layers[0].q_bias = vec![0.0; 4];
    let error = TiedSwiGluTrainingModel::extract(&config(), &spec(), &biased).unwrap_err();
    assert!(error.to_string().contains("QKV bias"), "{error}");

    let mut qk_normalized = weights();
    qk_normalized.layers[0].q_norm = vec![1.0; 2];
    let error = TiedSwiGluTrainingModel::extract(&config(), &spec(), &qk_normalized).unwrap_err();
    assert!(error.to_string().contains("QK norm"), "{error}");

    let mut untied = weights();
    untied.lm_head = Some(dense(5, 4, 9.0));
    let error = TiedSwiGluTrainingModel::extract(&config(), &spec(), &untied).unwrap_err();
    assert!(error.to_string().contains("untied LM head"), "{error}");
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
