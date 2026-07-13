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
