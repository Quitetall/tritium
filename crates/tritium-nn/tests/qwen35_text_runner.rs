use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use tritium_core::{GemmShape, TernaryFormat, Trit};
use tritium_nn::{
    DenseLinear, NnError, Projection, Qwen35DeltaNetConfig, Qwen35DeltaNetWeights, Qwen35Dtype,
    Qwen35FullAttentionConfig, Qwen35FullAttentionWeights, Qwen35LayerType, Qwen35MtpConfig,
    Qwen35NormWeightSemantics, Qwen35OutputGate, Qwen35RopeConfig, Qwen35RopeType,
    Qwen35TextConfig, Qwen35TextLayerWeights, Qwen35TextMixerWeights, Qwen35TextRunner,
    Qwen35TextWeights, SwiGluMlp, TernaryLinear, TokenEmbedding,
};
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, MpGemm, TernaryBackend};

const H: usize = 4;
const I: usize = 6;
const V: usize = 7;

fn parameter(ordinal: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let residue = (17 * index + 13 * ordinal + 5) % 29;
            (residue as i32 - 14) as f32 / 32.0
        })
        .collect()
}

fn dense_exact(ordinal: usize, n_out: usize, k_in: usize) -> Projection {
    Projection::Dense(
        DenseLinear::new_exact(parameter(ordinal, n_out * k_in), n_out, k_in).unwrap(),
    )
}

fn dense_a8(ordinal: usize, n_out: usize, k_in: usize) -> Projection {
    Projection::Dense(DenseLinear::new(parameter(ordinal, n_out * k_in), n_out, k_in).unwrap())
}

fn config(layer_types: Vec<Qwen35LayerType>, interval: u32) -> Qwen35TextConfig {
    Qwen35TextConfig {
        model_type: "qwen3_5_text".to_owned(),
        num_hidden_layers: u32::try_from(layer_types.len()).unwrap(),
        hidden_size: H as u32,
        intermediate_size: I as u32,
        vocab_size: V as u32,
        max_position_embeddings: 32,
        full_attention_interval: interval,
        layer_types,
        full_attention: Qwen35FullAttentionConfig {
            num_heads: 2,
            num_key_value_heads: 1,
            head_dim: 4,
            bias: false,
            dropout: 0.0,
            output_gate: Qwen35OutputGate::Sigmoid,
            norm_weight_semantics: Qwen35NormWeightSemantics::ZeroCenteredOnePlusWeight,
        },
        delta_net: Qwen35DeltaNetConfig {
            conv_kernel_dim: 4,
            num_key_heads: 1,
            num_value_heads: 2,
            key_head_dim: 2,
            value_head_dim: 2,
            state_arithmetic_dtype: Qwen35Dtype::Float32,
            output_gate: Qwen35OutputGate::Swish,
            gated_norm_weight_semantics: Qwen35NormWeightSemantics::UnitCenteredDirectWeight,
        },
        rope: Qwen35RopeConfig {
            theta: 10_000.0,
            partial_rotary_factor: 0.5,
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

fn delta_weights(exact: bool) -> Qwen35DeltaNetWeights {
    let projection = |ordinal, n_out, k_in| {
        if exact {
            dense_exact(ordinal, n_out, k_in)
        } else {
            dense_a8(ordinal, n_out, k_in)
        }
    };
    Qwen35DeltaNetWeights::new(
        projection(6, 8, H),
        projection(7, 4, H),
        projection(8, 2, H),
        projection(9, 2, H),
        projection(5, H, 4),
        parameter(3, 8 * 4),
        parameter(4, 2),
        parameter(1, 2),
        parameter(2, 2),
    )
}

fn full_weights(exact: bool) -> Qwen35FullAttentionWeights {
    let projection = |ordinal, n_out, k_in| {
        if exact {
            dense_exact(ordinal, n_out, k_in)
        } else {
            dense_a8(ordinal, n_out, k_in)
        }
    };
    Qwen35FullAttentionWeights::new(
        projection(15, 16, H),
        projection(16, 4, H),
        projection(17, 4, H),
        projection(18, H, 8),
        parameter(19, 4),
        parameter(20, 4),
    )
}

fn mlp(layer: usize, exact: bool) -> SwiGluMlp {
    let base = if layer == 0 { 10 } else { 21 };
    let projection = |ordinal, n_out, k_in| {
        if exact {
            dense_exact(ordinal, n_out, k_in)
        } else {
            dense_a8(ordinal, n_out, k_in)
        }
    };
    SwiGluMlp::new(
        projection(base, I, H),
        projection(base + 1, I, H),
        projection(base + 2, H, I),
    )
    .unwrap()
}

fn raw_layer(kind: Qwen35LayerType, layer: usize, exact: bool) -> Qwen35TextLayerWeights {
    let (input_norm, post_norm) = if layer == 0 { (13, 14) } else { (24, 25) };
    let mixer = match kind {
        Qwen35LayerType::DeltaNet => Qwen35TextMixerWeights::DeltaNet(delta_weights(exact)),
        Qwen35LayerType::FullAttention => {
            Qwen35TextMixerWeights::FullAttention(full_weights(exact))
        }
    };
    Qwen35TextLayerWeights::new(
        parameter(input_norm, H),
        mixer,
        parameter(post_norm, H),
        mlp(layer.min(1), exact),
    )
}

fn exact_weights() -> Qwen35TextWeights {
    Qwen35TextWeights::new(
        TokenEmbedding::from_dense(parameter(0, V * H), V, H).unwrap(),
        vec![
            raw_layer(Qwen35LayerType::DeltaNet, 0, true),
            raw_layer(Qwen35LayerType::FullAttention, 1, true),
        ],
        parameter(26, H),
        dense_exact(27, V, H),
    )
}

fn exact_runner() -> Qwen35TextRunner {
    Qwen35TextRunner::new(
        &config(
            vec![Qwen35LayerType::DeltaNet, Qwen35LayerType::FullAttention],
            2,
        ),
        exact_weights(),
        Box::new(tritium_cpu::CpuBackend::new()),
    )
    .unwrap()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[test]
#[allow(clippy::excessive_precision)]
fn transformers_5_5_3_two_layer_oracle_and_cached_decode_match() {
    // Generated modeling_qwen3_5.py SHA-256:
    // aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba
    // transformers source tag: c6c8503869367af938666810e01a71866ca4fe93
    let expected_prefill_hidden = [
        0.2664346993,
        -1.6725171804,
        0.4617776573,
        -0.3494569063,
        0.4306403697,
        -1.5771758556,
        0.5621415973,
        -0.2753130794,
        2.1214075089,
        -1.0040738583,
        0.2510471940,
        0.2221434563,
    ];
    let expected_prefill_logits = [
        -0.8410053253,
        0.5659754872,
        0.8355028033,
        -0.7913014293,
        0.6156793833,
        -1.0373188257,
        -0.7415975332,
    ];
    let expected_decode_hidden = [-1.2401254177, 0.7075314522, 0.2221986055, -1.6879242659];
    let expected_decode_logits = [
        1.1545130014,
        -0.11116229,
        -0.9370046258,
        1.0920654535,
        -0.1736097634,
        0.1244115233,
        1.0296180248,
    ];

    let runner = exact_runner();
    let mut cache = runner.new_cache(16).unwrap();
    let prefill = runner.forward(&[1, 4, 2], &mut cache).unwrap();
    assert_eq!(prefill.sequence(), 3);
    assert_eq!(prefill.hidden_size(), H);
    assert_close(
        prefill.final_hidden_states(),
        &expected_prefill_hidden,
        3e-6,
    );
    assert_close(prefill.last_logits(), &expected_prefill_logits, 3e-6);
    assert_eq!(cache.len(), 3);

    let decode = runner.forward(&[6], &mut cache).unwrap();
    assert_close(decode.final_hidden_states(), &expected_decode_hidden, 3e-6);
    assert_close(decode.last_logits(), &expected_decode_logits, 3e-6);
    assert_eq!(cache.len(), 4);

    let one_shot_runner = exact_runner();
    let mut one_shot_cache = one_shot_runner.new_cache(16).unwrap();
    let one_shot = one_shot_runner
        .forward(&[1, 4, 2, 6], &mut one_shot_cache)
        .unwrap();
    assert_close(
        decode.final_hidden_states(),
        &one_shot.final_hidden_states()[3 * H..],
        3e-6,
    );
    assert_close(decode.last_logits(), one_shot.last_logits(), 3e-6);
}

#[test]
fn exact_schedule_rejects_a_wrong_mixer_in_a_64_layer_table() {
    let schedule: Vec<_> = (1..=64)
        .map(|layer| {
            if layer % 4 == 0 {
                Qwen35LayerType::FullAttention
            } else {
                Qwen35LayerType::DeltaNet
            }
        })
        .collect();
    let mut layers: Vec<_> = schedule
        .iter()
        .copied()
        .enumerate()
        .map(|(index, kind)| raw_layer(kind, index.min(1), true))
        .collect();
    layers[0] = raw_layer(Qwen35LayerType::FullAttention, 0, true);
    let weights = Qwen35TextWeights::new(
        TokenEmbedding::from_dense(parameter(0, V * H), V, H).unwrap(),
        layers,
        parameter(26, H),
        dense_exact(27, V, H),
    );
    let error = Qwen35TextRunner::new(
        &config(schedule, 4),
        weights,
        Box::new(tritium_cpu::CpuBackend::new()),
    )
    .err()
    .expect("wrong layer kind must fail closed");
    assert!(matches!(error, NnError::MissingConfig(message) if message.contains("layer 0")));
}

#[test]
fn cache_identity_oov_capacity_and_delta_continuation_fail_atomically() {
    let runner = exact_runner();
    let foreign = exact_runner();
    let mut cache = runner.new_cache(4).unwrap();
    let foreign_error = foreign
        .forward(&[1], &mut cache)
        .expect_err("cross-runner cache must fail");
    assert!(
        matches!(foreign_error, NnError::Backend(message) if message.contains("different runner"))
    );
    assert_eq!(cache.len(), 0);

    runner.forward(&[1], &mut cache).unwrap();
    assert!(matches!(
        runner.forward(&[V as u32], &mut cache),
        Err(NnError::MissingTensor(_))
    ));
    assert_eq!(cache.len(), 1);
    assert!(matches!(
        runner.forward(&[2, 3], &mut cache),
        Err(NnError::MissingConfig(_))
    ));
    assert_eq!(cache.len(), 1);

    let mut full = runner.new_cache(2).unwrap();
    runner.forward(&[1, 2], &mut full).unwrap();
    assert!(matches!(
        runner.forward(&[3], &mut full),
        Err(NnError::Shape { .. })
    ));
    assert_eq!(full.len(), 2);
    full.reset();
    assert!(full.is_empty());
}

#[derive(Debug)]
struct SwitchBackend {
    cpu: tritium_cpu::CpuBackend,
    fail: Arc<AtomicBool>,
}

impl TernaryBackend for SwitchBackend {
    fn device_id(&self) -> &str {
        "qwen35-switch"
    }

    fn capabilities(&self) -> DeviceCaps {
        self.cpu.capabilities()
    }

    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.cpu.upload_weights(packed, shape, format)
    }

    fn mpgemm(&self, parameters: MpGemm<'_>) -> Result<(), BackendError> {
        if self.fail.load(Ordering::SeqCst) {
            parameters.out.fill(1234.0);
            Err(BackendError::Backend(
                "intentional language-head failure".to_owned(),
            ))
        } else {
            self.cpu.mpgemm(parameters)
        }
    }
}

fn a8_runner(fail: Arc<AtomicBool>) -> Qwen35TextRunner {
    let backend = SwitchBackend {
        cpu: tritium_cpu::CpuBackend::new(),
        fail,
    };
    let head =
        Projection::Ternary(TernaryLinear::new(&backend, &[Trit::ZERO; V * H], V, H, 1.0).unwrap());
    let weights = Qwen35TextWeights::new(
        TokenEmbedding::from_dense(parameter(0, V * H), V, H).unwrap(),
        vec![
            raw_layer(Qwen35LayerType::DeltaNet, 0, false),
            raw_layer(Qwen35LayerType::FullAttention, 1, false),
        ],
        parameter(26, H),
        head,
    );
    Qwen35TextRunner::new(
        &config(
            vec![Qwen35LayerType::DeltaNet, Qwen35LayerType::FullAttention],
            2,
        ),
        weights,
        Box::new(backend),
    )
    .unwrap()
}

#[test]
fn language_head_failure_rolls_back_every_mixer_then_retry_matches_fresh() {
    let fail = Arc::new(AtomicBool::new(false));
    let runner = a8_runner(Arc::clone(&fail));
    let mut cache = runner.new_cache(8).unwrap();
    runner.forward(&[1, 4, 2], &mut cache).unwrap();
    assert_eq!(cache.len(), 3);

    fail.store(true, Ordering::SeqCst);
    let error = runner
        .forward(&[6], &mut cache)
        .expect_err("late language-head failure must propagate");
    assert!(
        matches!(error, NnError::Backend(message) if message.contains("language-head failure"))
    );
    assert_eq!(cache.len(), 3);

    fail.store(false, Ordering::SeqCst);
    let retry = runner.forward(&[6], &mut cache).unwrap();
    assert_eq!(cache.len(), 4);

    let reference_fail = Arc::new(AtomicBool::new(false));
    let reference = a8_runner(reference_fail);
    let mut reference_cache = reference.new_cache(8).unwrap();
    reference.forward(&[1, 4, 2], &mut reference_cache).unwrap();
    let expected = reference.forward(&[6], &mut reference_cache).unwrap();
    assert_eq!(retry, expected);
}

#[test]
fn constructor_rejects_mixed_mode_shape_and_nonfinite_weights() {
    let cfg = config(
        vec![Qwen35LayerType::DeltaNet, Qwen35LayerType::FullAttention],
        2,
    );
    let mut mixed = exact_weights();
    mixed.lm_head = dense_a8(27, V, H);
    assert!(matches!(
        Qwen35TextRunner::new(&cfg, mixed, Box::new(tritium_cpu::CpuBackend::new())),
        Err(NnError::MissingConfig(_))
    ));

    let mut wrong_shape = exact_weights();
    wrong_shape.final_norm.pop();
    assert!(matches!(
        Qwen35TextRunner::new(&cfg, wrong_shape, Box::new(tritium_cpu::CpuBackend::new())),
        Err(NnError::Shape { .. })
    ));

    let mut nonfinite = exact_weights();
    nonfinite.layers[1].input_norm[2] = f32::NAN;
    assert!(matches!(
        Qwen35TextRunner::new(&cfg, nonfinite, Box::new(tritium_cpu::CpuBackend::new())),
        Err(NnError::Backend(_))
    ));

    let mut bad_qkv = parameter(6, 8 * H);
    bad_qkv[3] = f32::NAN;
    let bad_delta = Qwen35DeltaNetWeights::new(
        Projection::Dense(DenseLinear::new_exact(bad_qkv, 8, H).unwrap()),
        dense_exact(7, 4, H),
        dense_exact(8, 2, H),
        dense_exact(9, 2, H),
        dense_exact(5, H, 4),
        parameter(3, 8 * 4),
        parameter(4, 2),
        parameter(1, 2),
        parameter(2, 2),
    );
    let mut nonfinite_delta = exact_weights();
    nonfinite_delta.layers[0] = Qwen35TextLayerWeights::new(
        parameter(13, H),
        Qwen35TextMixerWeights::DeltaNet(bad_delta),
        parameter(14, H),
        mlp(0, true),
    );
    assert!(matches!(
        Qwen35TextRunner::new(
            &cfg,
            nonfinite_delta,
            Box::new(tritium_cpu::CpuBackend::new())
        ),
        Err(NnError::Backend(_))
    ));

    let mut bad_query = parameter(15, 16 * H);
    bad_query[7] = f32::INFINITY;
    let bad_full = Qwen35FullAttentionWeights::new(
        Projection::Dense(DenseLinear::new_exact(bad_query, 16, H).unwrap()),
        dense_exact(16, 4, H),
        dense_exact(17, 4, H),
        dense_exact(18, H, 8),
        parameter(19, 4),
        parameter(20, 4),
    );
    let mut nonfinite_full = exact_weights();
    nonfinite_full.layers[1] = Qwen35TextLayerWeights::new(
        parameter(24, H),
        Qwen35TextMixerWeights::FullAttention(bad_full),
        parameter(25, H),
        mlp(1, true),
    );
    assert!(matches!(
        Qwen35TextRunner::new(
            &cfg,
            nonfinite_full,
            Box::new(tritium_cpu::CpuBackend::new())
        ),
        Err(NnError::Backend(_))
    ));
}
