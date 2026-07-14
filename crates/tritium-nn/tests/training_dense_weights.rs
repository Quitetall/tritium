use tritium_nn::{
    ArchSpec, DenseLinear, Mlp, MlpKind, ModelConfig, ModelRunner, ModelWeights, Projection,
    SwiGluMlp, SwiGluTrainingModel, TransformerBlock,
};

fn config() -> ModelConfig {
    ModelConfig {
        arch: "llama".to_owned(),
        n_layers: 2,
        n_embd: 4,
        n_head: 2,
        n_head_kv: 1,
        head_dim: 2,
        n_ff: 6,
        n_ctx: 16,
        rope_theta: 10_000.0,
        rms_eps: 1e-5,
    }
}

fn spec(tied: bool) -> ArchSpec {
    ArchSpec {
        mlp: MlpKind::SwiGlu,
        attn_sub_norm: false,
        ffn_sub_norm: false,
        qk_norm: false,
        qkv_bias: false,
        tied_embeddings: tied,
    }
}

fn dense(rows: usize, cols: usize, seed: usize) -> Projection {
    let weights = (0..rows * cols)
        .map(|index| {
            let centered = ((index * 11 + seed * 7) % 29) as f32 - 14.0;
            centered / 128.0
        })
        .collect();
    Projection::Dense(DenseLinear::new_exact(weights, rows, cols).expect("valid fixture shape"))
}

fn weights(tied: bool) -> ModelWeights {
    let layers = (0..2)
        .map(|layer| TransformerBlock {
            attn_norm: vec![1.0, 0.875, 1.125, 0.75],
            q_proj: dense(4, 4, 10 + 10 * layer),
            k_proj: dense(2, 4, 11 + 10 * layer),
            v_proj: dense(2, 4, 12 + 10 * layer),
            o_proj: dense(4, 4, 13 + 10 * layer),
            attn_sub_norm: Vec::new(),
            q_bias: Vec::new(),
            k_bias: Vec::new(),
            v_bias: Vec::new(),
            q_norm: Vec::new(),
            k_norm: Vec::new(),
            ffn_norm: vec![0.75, 1.0, 0.875, 1.25],
            mlp: Mlp::SwiGlu(SwiGluMlp {
                gate: dense(6, 4, 14 + 10 * layer),
                up: dense(6, 4, 15 + 10 * layer),
                down: dense(4, 6, 16 + 10 * layer),
            }),
        })
        .collect();
    let token_embd = (0..7 * 4)
        .map(|index| (((index * 5) % 17) as f32 - 8.0) / 64.0)
        .collect();
    ModelWeights {
        token_embd,
        vocab: 7,
        n_embd: 4,
        layers,
        output_norm: vec![1.0, 0.875, 1.125, 0.75],
        lm_head: (!tied).then(|| dense(7, 4, 91)),
    }
}

fn cpu() -> Box<dyn tritium_spec::TernaryBackend> {
    Box::new(tritium_cpu::CpuBackend::new())
}

#[test]
fn widened_training_model_builds_a_dense_exact_runner_for_tied_and_untied_heads() {
    for tied in [true, false] {
        let source_config = config();
        let source_weights = weights(tied);
        let extraction_weights = weights(tied);
        let mut model =
            SwiGluTrainingModel::extract(&source_config, &spec(tied), &extraction_weights)
                .expect("extract training model");
        let mut source_runner =
            ModelRunner::from_weights(source_config.clone(), source_weights, cpu());

        model
            .widen_intermediate(11, 0x0027)
            .expect("widen intermediate axis");
        let dense_weights = model.to_dense_weights().expect("build dense evaluator");
        let mut widened_config = source_config;
        widened_config.n_ff = 11;
        let mut widened_runner = ModelRunner::from_weights(widened_config, dense_weights, cpu());

        let tokens = [0, 3, 6, 2];
        let positions = [0, 1, 2, 3];
        let source_logits = source_runner
            .forward(&tokens, &positions)
            .expect("source forward");
        let widened_logits = widened_runner
            .forward(&tokens, &positions)
            .expect("widened forward");
        assert_eq!(widened_logits.len(), source_logits.len());
        let max_absolute_error = source_logits
            .iter()
            .zip(&widened_logits)
            .map(|(&source, &widened)| (source - widened).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_absolute_error <= 2e-6,
            "tied={tied}: widening changed dense exact logits by {max_absolute_error:e}"
        );
        assert_eq!(widened_runner.weights.lm_head.is_none(), tied);
    }
}

#[test]
fn dense_weight_reconstruction_rejects_drained_training_masters() {
    let mut model =
        SwiGluTrainingModel::extract(&config(), &spec(true), &weights(true)).expect("extract");
    let masters = model.take_parameter_masters();
    assert!(masters.iter().all(|master| !master.is_empty()));

    let error = match model.to_dense_weights() {
        Ok(_) => panic!("drained masters must fail"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("drained"), "{error}");
}
