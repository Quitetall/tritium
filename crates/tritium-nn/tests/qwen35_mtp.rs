//! Public structural contract for the deliberately unverified Qwen3.5-family MTP drafter.

use tritium_cpu::CpuBackend;
use tritium_nn::{
    DenseLinear, NnError, Projection, QWEN35_MTP_UNVERIFIED_REASON, Qwen35DeltaNetConfig,
    Qwen35Dtype, Qwen35FullAttentionConfig, Qwen35FullAttentionWeights, Qwen35LayerType,
    Qwen35MtpConfig, Qwen35MtpLayerWeights, Qwen35MtpStatus, Qwen35MtpWeights,
    Qwen35NormWeightSemantics, Qwen35OutputGate, Qwen35RopeConfig, Qwen35RopeType,
    Qwen35TextConfig, Qwen35TextLayerWeights, Qwen35TextMixerWeights, Qwen35TextRunner,
    Qwen35TextWeights, SwiGluMlp, TokenEmbedding, UnverifiedQwen35Mtp,
};

const H: usize = 4;
const I: usize = 8;
const V: usize = 7;

fn text_config() -> Qwen35TextConfig {
    Qwen35TextConfig {
        model_type: "qwen3_5_text".to_owned(),
        num_hidden_layers: 1,
        hidden_size: H as u32,
        intermediate_size: I as u32,
        vocab_size: V as u32,
        max_position_embeddings: 32,
        full_attention_interval: 1,
        layer_types: vec![Qwen35LayerType::FullAttention],
        full_attention: Qwen35FullAttentionConfig {
            num_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            bias: false,
            dropout: 0.0,
            output_gate: Qwen35OutputGate::Sigmoid,
            norm_weight_semantics: Qwen35NormWeightSemantics::ZeroCenteredOnePlusWeight,
        },
        delta_net: Qwen35DeltaNetConfig {
            conv_kernel_dim: 4,
            num_key_heads: 1,
            num_value_heads: 1,
            key_head_dim: 2,
            value_head_dim: 2,
            state_arithmetic_dtype: Qwen35Dtype::Float32,
            output_gate: Qwen35OutputGate::Swish,
            gated_norm_weight_semantics: Qwen35NormWeightSemantics::UnitCenteredDirectWeight,
        },
        rope: Qwen35RopeConfig {
            theta: 10_000.0,
            partial_rotary_factor: 1.0,
            rotary_dim: 2,
            rope_type: Qwen35RopeType::Default,
            mrope_interleaved: true,
            mrope_section: [1, 0, 0],
        },
        rms_norm_eps: 1e-6,
        source_dtype: Qwen35Dtype::Bfloat16,
        use_cache: true,
        tied_embeddings: false,
        mtp: Qwen35MtpConfig {
            num_hidden_layers: 1,
            dedicated_embeddings: false,
        },
    }
}

fn projection(rows: usize, columns: usize, exact: bool) -> Projection {
    let weights = vec![0.0; rows * columns];
    Projection::Dense(if exact {
        DenseLinear::new_exact(weights, rows, columns).unwrap()
    } else {
        DenseLinear::new(weights, rows, columns).unwrap()
    })
}

fn attention(exact: bool) -> Qwen35FullAttentionWeights {
    Qwen35FullAttentionWeights::new(
        projection(8, H, exact),
        projection(2, H, exact),
        projection(2, H, exact),
        projection(H, 4, exact),
        vec![0.0; 2],
        vec![0.0; 2],
    )
}

fn wrong_query_shape_attention() -> Qwen35FullAttentionWeights {
    Qwen35FullAttentionWeights::new(
        projection(7, H, true),
        projection(2, H, true),
        projection(2, H, true),
        projection(H, 4, true),
        vec![0.0; 2],
        vec![0.0; 2],
    )
}

fn mlp(exact: bool) -> SwiGluMlp {
    SwiGluMlp::new(
        projection(I, H, exact),
        projection(I, H, exact),
        projection(H, I, exact),
    )
    .unwrap()
}

fn assemble_with(
    target: &Qwen35TextRunner,
    fc: Projection,
    attention: Qwen35FullAttentionWeights,
    mlp: SwiGluMlp,
) -> Result<UnverifiedQwen35Mtp, NnError> {
    UnverifiedQwen35Mtp::new(
        target,
        Qwen35MtpWeights::new(
            vec![0.0; H],
            vec![0.0; H],
            fc,
            Qwen35MtpLayerWeights::new(vec![0.0; H], attention, vec![0.0; H], mlp),
            vec![0.0; H],
        ),
    )
}

fn language_runner(config: &Qwen35TextConfig) -> Qwen35TextRunner {
    let layer = Qwen35TextLayerWeights::new(
        vec![0.0; H],
        Qwen35TextMixerWeights::FullAttention(attention(true)),
        vec![0.0; H],
        mlp(true),
    );
    let weights = Qwen35TextWeights::new(
        TokenEmbedding::from_dense(vec![0.0; V * H], V, H).unwrap(),
        vec![layer],
        vec![0.0; H],
        projection(V, H, true),
    );
    Qwen35TextRunner::new(config, weights, Box::new(CpuBackend::new())).unwrap()
}

#[test]
fn alignment_shifts_tokens_but_preserves_target_output_positions_and_hidden_rows() {
    let config = text_config();
    let runner = language_runner(&config);
    let mut cache = runner.new_cache(16).unwrap();
    runner.forward(&[0, 0], &mut cache).unwrap();
    let target_output = runner.forward(&[1, 4, 2], &mut cache).unwrap();
    let mtp = assemble_with(
        &runner,
        projection(H, 2 * H, true),
        attention(true),
        mlp(true),
    )
    .unwrap();

    let plan = mtp.align_step(&target_output, 6).unwrap();

    assert_eq!(target_output.position_start(), 2);
    assert_eq!(plan.shifted_token_ids(), &[4, 2, 6]);
    assert_eq!(plan.positions(), &[2, 3, 4]);
    assert!(std::ptr::eq(
        plan.target_hidden_states().as_ptr(),
        target_output.final_hidden_states().as_ptr()
    ));
    assert_eq!(
        plan.target_hidden_states(),
        target_output.final_hidden_states()
    );
}

#[test]
fn assembly_is_explicitly_unverified_and_not_a_runnable_claim() {
    let config = text_config();
    let runner = language_runner(&config);
    let mtp = assemble_with(
        &runner,
        projection(H, 2 * H, true),
        attention(true),
        mlp(true),
    )
    .unwrap();

    assert_eq!(mtp.status(), Qwen35MtpStatus::Unverified);
    assert_eq!(mtp.status().reason(), QWEN35_MTP_UNVERIFIED_REASON);
}

#[test]
fn assembly_rejects_wrong_mtp_contract_shapes_and_activation_modes() {
    let mut wrong_layer_count = text_config();
    wrong_layer_count.mtp.num_hidden_layers = 2;
    let wrong_layer_count_runner = language_runner(&wrong_layer_count);
    assert!(
        assemble_with(
            &wrong_layer_count_runner,
            projection(H, 2 * H, true),
            attention(true),
            mlp(true),
        )
        .is_err()
    );

    let mut dedicated = text_config();
    dedicated.mtp.dedicated_embeddings = true;
    let dedicated_runner = language_runner(&dedicated);
    assert!(
        assemble_with(
            &dedicated_runner,
            projection(H, 2 * H, true),
            attention(true),
            mlp(true),
        )
        .is_err()
    );

    let config = text_config();
    let runner = language_runner(&config);
    assert!(
        assemble_with(
            &runner,
            projection(H, 2 * H - 1, true),
            attention(true),
            mlp(true),
        )
        .is_err()
    );
    assert!(
        assemble_with(
            &runner,
            projection(H, 2 * H, true),
            wrong_query_shape_attention(),
            mlp(true),
        )
        .is_err()
    );
    assert!(
        assemble_with(
            &runner,
            projection(H, 2 * H, true),
            attention(true),
            mlp(false),
        )
        .is_err()
    );
    assert!(
        UnverifiedQwen35Mtp::new(
            &runner,
            Qwen35MtpWeights::new(
                vec![f32::NAN; H],
                vec![0.0; H],
                projection(H, 2 * H, true),
                Qwen35MtpLayerWeights::new(vec![0.0; H], attention(true), vec![0.0; H], mlp(true),),
                vec![0.0; H],
            ),
        )
        .is_err()
    );
}

#[test]
fn alignment_rejects_foreign_outputs_and_out_of_vocabulary_samples() {
    let config = text_config();
    let runner = language_runner(&config);
    let mut cache = runner.new_cache(16).unwrap();
    let target_output = runner.forward(&[1, 4, 2], &mut cache).unwrap();
    let mtp = assemble_with(
        &runner,
        projection(H, 2 * H, true),
        attention(true),
        mlp(true),
    )
    .unwrap();

    assert!(mtp.align_step(&target_output, V as u32).is_err());

    let foreign_runner = language_runner(&config);
    let mut foreign_cache = foreign_runner.new_cache(16).unwrap();
    let foreign_output = foreign_runner
        .forward(&[1, 4, 2], &mut foreign_cache)
        .unwrap();
    assert!(mtp.align_step(&foreign_output, 6).is_err());
}
