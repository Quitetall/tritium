//! Dense reference conformance for Qwen3.5-family gated full attention.

use std::sync::atomic::{AtomicBool, Ordering};
use tritium_core::Trit;
use tritium_cpu::CpuBackend;
use tritium_nn::{
    DenseLinear, NnError, Projection, ProjectionActivationMode, Qwen35DeltaNetConfig, Qwen35Dtype,
    Qwen35FullAttention, Qwen35FullAttentionConfig, Qwen35FullAttentionWeights, Qwen35LayerType,
    Qwen35MtpConfig, Qwen35NormWeightSemantics, Qwen35OutputGate, Qwen35RopeConfig, Qwen35RopeType,
    Qwen35TextConfig, TernaryLinear,
};

use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, GemmShape, MpGemm, TernaryBackend};

const Q_PROJ: [f32; 32] = [
    0.20, -0.10, 0.05, 0.30, -0.25, 0.40, 0.15, -0.05, 0.10, 0.20, -0.30, 0.25, -0.15, 0.05, 0.35,
    0.10, -0.30, 0.10, 0.25, 0.20, 0.05, -0.20, 0.30, 0.15, 0.40, -0.25, 0.10, -0.05, -0.10, 0.30,
    0.20, -0.15,
];
const K_PROJ: [f32; 8] = [0.30, 0.10, -0.20, 0.05, -0.10, 0.25, 0.15, 0.20];
const V_PROJ: [f32; 8] = [0.20, -0.30, 0.10, 0.40, -0.25, 0.15, 0.35, -0.10];
const O_PROJ: [f32; 16] = [
    0.50, -0.10, 0.20, 0.0, 0.10, 0.30, -0.20, 0.40, -0.30, 0.20, 0.40, -0.10, 0.25, 0.0, 0.15,
    0.35,
];
const Q_NORM: [f32; 2] = [0.10, -0.20];
const K_NORM: [f32; 2] = [-0.15, 0.25];

fn text_config(num_heads: u32, num_kv_heads: u32, head_dim: u32) -> Qwen35TextConfig {
    Qwen35TextConfig {
        model_type: "qwen3_5_text".to_owned(),
        num_hidden_layers: 1,
        hidden_size: 4,
        intermediate_size: 8,
        vocab_size: 16,
        max_position_embeddings: 32,
        full_attention_interval: 1,
        layer_types: vec![Qwen35LayerType::FullAttention],
        full_attention: Qwen35FullAttentionConfig {
            num_heads,
            num_key_value_heads: num_kv_heads,
            head_dim,
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
            rotary_dim: head_dim,
            rope_type: Qwen35RopeType::Default,
            mrope_interleaved: true,
            mrope_section: [head_dim / 2, 0, 0],
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

fn dense(values: &[f32], n_out: usize, k_in: usize) -> Projection {
    Projection::Dense(DenseLinear::new_exact(values.to_vec(), n_out, k_in).unwrap())
}

fn dense_a8(values: &[f32], n_out: usize, k_in: usize) -> Projection {
    Projection::Dense(DenseLinear::new(values.to_vec(), n_out, k_in).unwrap())
}

fn official_weights() -> Qwen35FullAttentionWeights {
    Qwen35FullAttentionWeights::new(
        dense(&Q_PROJ, 8, 4),
        dense(&K_PROJ, 2, 4),
        dense(&V_PROJ, 2, 4),
        dense(&O_PROJ, 4, 4),
        Q_NORM.to_vec(),
        K_NORM.to_vec(),
    )
}

fn official_layer() -> Qwen35FullAttention {
    Qwen35FullAttention::new(&text_config(2, 1, 2), official_weights()).unwrap()
}

fn assert_close(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (lane, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= 2e-6,
            "lane {lane}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn transformers_5_5_3_prefill_and_incremental_goldens() {
    // Frozen from the official Transformers 5.5.3 Qwen3_5Attention fp32 path.
    // The fused q_proj rows use the per-head [Q, gate] layout; treating the two
    // global halves as Q and gate changes these outputs.
    let layer = official_layer();
    let backend = CpuBackend::new();
    let mut cache = layer.new_cache(8).unwrap();
    let mut prefill_out = [f32::NAN; 8];
    layer
        .forward(
            &backend,
            &[1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -0.25],
            &[0, 1],
            &mut cache,
            &mut prefill_out,
        )
        .unwrap();

    assert_close(
        &prefill_out,
        &[
            0.521_853_4,
            -0.207_832_57,
            0.044_160_135,
            0.227_438_76,
            -0.040_666_483,
            -0.011_874_773,
            0.303_812_74,
            -0.051_141_255,
        ],
    );
    assert_close(
        cache.keys(),
        &[0.980_920_73, 1.021_792_4, -1.409_578_7, -0.240_446_03],
    );
    assert_close(cache.values(), &[1.175_000_1, -0.4375, -0.500_000_06, 0.95]);
    assert_eq!(cache.len(), 2);

    let mut decode_out = [f32::NAN; 4];
    layer
        .forward(
            &backend,
            &[0.5, 1.5, -1.0, 0.75],
            &[2],
            &mut cache,
            &mut decode_out,
        )
        .unwrap();
    assert_close(
        &decode_out,
        &[-0.018_088_907, -0.111_877_49, 0.229_311_5, -0.037_711_766],
    );
    assert_close(&cache.keys()[4..], &[-1.259_783_4, 0.554_716_2]);
    assert_close(cache.values().get(4..).unwrap(), &[-0.15, -0.325]);
    assert_eq!(cache.len(), 3);
}

#[test]
fn one_shot_and_two_plus_one_streaming_match_the_official_trace() {
    let layer = official_layer();
    let backend = CpuBackend::new();
    let inputs = [
        1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -0.25, 0.5, 1.5, -1.0, 0.75,
    ];
    let expected = [
        0.521_853_4,
        -0.207_832_57,
        0.044_160_135,
        0.227_438_76,
        -0.040_666_483,
        -0.011_874_773,
        0.303_812_74,
        -0.051_141_255,
        -0.018_088_907,
        -0.111_877_49,
        0.229_311_5,
        -0.037_711_766,
    ];

    let mut one_shot_cache = layer.new_cache(3).unwrap();
    let mut one_shot = [f32::NAN; 12];
    layer
        .forward(
            &backend,
            &inputs,
            &[0, 1, 2],
            &mut one_shot_cache,
            &mut one_shot,
        )
        .unwrap();

    let mut streamed_cache = layer.new_cache(3).unwrap();
    let mut streamed = [f32::NAN; 12];
    layer
        .forward(
            &backend,
            &inputs[..8],
            &[0, 1],
            &mut streamed_cache,
            &mut streamed[..8],
        )
        .unwrap();
    layer
        .forward(
            &backend,
            &inputs[8..],
            &[2],
            &mut streamed_cache,
            &mut streamed[8..],
        )
        .unwrap();

    assert_close(&one_shot, &expected);
    assert_close(&streamed, &expected);
    assert_close(&one_shot, &streamed);
    assert_close(one_shot_cache.keys(), streamed_cache.keys());
    assert_close(one_shot_cache.values(), streamed_cache.values());
}

#[test]
fn full_mixer_applies_rope_only_to_the_configured_prefix() {
    // Independent Transformers 5.5.3 fp32 trace with head_dim=4,
    // partial_rotary_factor=0.5, rotary_dim=2, position=1. The identity K
    // projection exposes the post-Qwen-RMSNorm cache row directly.
    let mut config = text_config(1, 1, 4);
    config.rope.partial_rotary_factor = 0.5;
    config.rope.rotary_dim = 2;
    config.rope.mrope_section = [1, 0, 0];
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let weights = Qwen35FullAttentionWeights::new(
        dense(&[0.0; 32], 8, 4),
        dense(&identity, 4, 4),
        dense(&[0.0; 16], 4, 4),
        dense(&[0.0; 16], 4, 4),
        vec![0.0; 4],
        vec![0.0; 4],
    );
    let layer = Qwen35FullAttention::new(&config, weights).unwrap();
    let mut cache = layer.new_cache(2).unwrap();
    let mut output = [f32::NAN; 4];

    layer
        .forward(
            &CpuBackend::new(),
            &[1.0, 2.0, 3.0, 4.0],
            &[1],
            &mut cache,
            &mut output,
        )
        .unwrap();

    assert_close(
        cache.keys(),
        &[-0.417_232_96, 0.701_842_8, 1.095_445_2, 1.460_593_5],
    );
    // The rotary suffix is bit-identical to the independent normalized input;
    // using head_dim instead of rotary_dim would rotate these two lanes.
    assert_eq!(cache.keys()[2].to_bits(), 1.095_445_2f32.to_bits());
    assert_eq!(cache.keys()[3].to_bits(), 1.460_593_5f32.to_bits());
}

#[test]
fn transformers_partial_rope_golden_observes_the_query_path() {
    // Frozen from Transformers 5.5.3 Qwen3_5Attention in fp32. Unlike the
    // cache-only partial-RoPE contract above, this two-token output depends on
    // the rotated query. Rotating all four query lanes instead of the configured
    // two changes the second row and must fail this integration contract.
    let mut config = text_config(1, 1, 4);
    config.rope.partial_rotary_factor = 0.5;
    config.rope.rotary_dim = 2;
    config.rope.mrope_section = [1, 0, 0];
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let value = [
        0.20, -0.30, 0.10, 0.40, -0.25, 0.15, 0.35, -0.10, 0.30, 0.05, -0.20, 0.25, -0.10, 0.40,
        0.15, -0.30,
    ];
    let weights = Qwen35FullAttentionWeights::new(
        dense(&Q_PROJ, 8, 4),
        dense(&identity, 4, 4),
        dense(&value, 4, 4),
        dense(&identity, 4, 4),
        vec![0.10, -0.20, 0.05, -0.15],
        vec![-0.15, 0.25, 0.20, -0.10],
    );
    let layer = Qwen35FullAttention::new(&config, weights).unwrap();
    let mut cache = layer.new_cache(2).unwrap();
    let mut output = [f32::NAN; 8];

    layer
        .forward(
            &CpuBackend::new(),
            &[1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -0.25],
            &[0, 1],
            &mut cache,
            &mut output,
        )
        .unwrap();

    assert_close(
        &output,
        &[
            0.620_512_07,
            -0.274_888_34,
            0.442_713_5,
            -0.325_628_85,
            0.185_875_56,
            0.164_657_65,
            -0.019_909_615,
            -0.007_883_976,
        ],
    );
}

#[test]
fn fused_query_and_gate_are_deinterleaved_per_head() {
    let mut q_proj = vec![0.0; 8 * 4];
    // Per-head output rows are [q0, q1, gate0, gate1]. Only the gate rows matter
    // for this one-token case; a global-half split would give head 0 gates [0, 0].
    for (row, value) in [(2, 10.0), (3, -10.0), (6, -10.0), (7, 10.0)] {
        q_proj[row * 4] = value;
    }
    let v_proj = [2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0];
    let o_proj = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let weights = Qwen35FullAttentionWeights::new(
        dense(&q_proj, 8, 4),
        dense(&[0.0; 8], 2, 4),
        dense(&v_proj, 2, 4),
        dense(&o_proj, 4, 4),
        vec![0.0; 2],
        vec![0.0; 2],
    );
    let layer = Qwen35FullAttention::new(&text_config(2, 1, 2), weights).unwrap();
    let mut cache = layer.new_cache(1).unwrap();
    let mut out = [0.0; 4];
    layer
        .forward(
            &CpuBackend::new(),
            &[1.0, 0.0, 0.0, 0.0],
            &[0],
            &mut cache,
            &mut out,
        )
        .unwrap();

    let positive = 1.0 / (1.0 + (-10.0f32).exp());
    let negative = 1.0 / (1.0 + 10.0f32.exp());
    assert_close(
        &out,
        &[2.0 * positive, -negative, 2.0 * negative, -positive],
    );
}

#[test]
fn constructor_rejects_invalid_geometry_and_weight_shapes() {
    let mut config = text_config(2, 1, 2);
    config.full_attention.num_heads = 3;
    config.full_attention.num_key_value_heads = 2;
    assert!(matches!(
        Qwen35FullAttention::new(&config, official_weights()),
        Err(NnError::MissingConfig(_))
    ));

    let wrong_q = Qwen35FullAttentionWeights::new(
        dense(&[0.0; 28], 7, 4),
        dense(&K_PROJ, 2, 4),
        dense(&V_PROJ, 2, 4),
        dense(&O_PROJ, 4, 4),
        Q_NORM.to_vec(),
        K_NORM.to_vec(),
    );
    assert!(matches!(
        Qwen35FullAttention::new(&text_config(2, 1, 2), wrong_q),
        Err(NnError::Shape {
            expected: 8,
            got: 7
        })
    ));

    let wrong_norm = Qwen35FullAttentionWeights::new(
        dense(&Q_PROJ, 8, 4),
        dense(&K_PROJ, 2, 4),
        dense(&V_PROJ, 2, 4),
        dense(&O_PROJ, 4, 4),
        vec![0.0; 1],
        K_NORM.to_vec(),
    );
    assert!(matches!(
        Qwen35FullAttention::new(&text_config(2, 1, 2), wrong_norm),
        Err(NnError::Shape {
            expected: 2,
            got: 1
        })
    ));
}

#[test]
fn constructor_rejects_mixed_projection_activation_modes() {
    let mixed = Qwen35FullAttentionWeights::new(
        dense(&Q_PROJ, 8, 4),
        dense_a8(&K_PROJ, 2, 4),
        dense(&V_PROJ, 2, 4),
        dense(&O_PROJ, 4, 4),
        Q_NORM.to_vec(),
        K_NORM.to_vec(),
    );

    assert!(matches!(
        Qwen35FullAttention::new(&text_config(2, 1, 2), mixed),
        Err(NnError::MissingConfig(_))
    ));

    let uniform_a8 = Qwen35FullAttentionWeights::new(
        dense_a8(&Q_PROJ, 8, 4),
        dense_a8(&K_PROJ, 2, 4),
        dense_a8(&V_PROJ, 2, 4),
        dense_a8(&O_PROJ, 4, 4),
        Q_NORM.to_vec(),
        K_NORM.to_vec(),
    );
    let layer = Qwen35FullAttention::new(&text_config(2, 1, 2), uniform_a8).unwrap();
    assert_eq!(layer.activation_mode(), ProjectionActivationMode::A8);
}

#[test]
fn forward_rejects_wrong_positions_cache_input_and_output_without_mutation() {
    let layer = official_layer();
    let backend = CpuBackend::new();
    let mut cache = layer.new_cache(4).unwrap();
    let mut sentinel = [17.0; 4];
    assert!(
        layer
            .forward(&backend, &[0.0; 4], &[32], &mut cache, &mut sentinel)
            .is_err()
    );
    assert_eq!(cache.len(), 0);
    assert_eq!(sentinel, [17.0; 4]);

    assert!(
        layer
            .forward(&backend, &[0.0; 3], &[0], &mut cache, &mut sentinel)
            .is_err()
    );
    assert!(
        layer
            .forward(&backend, &[0.0; 4], &[0], &mut cache, &mut sentinel[..3])
            .is_err()
    );
    assert_eq!(cache.len(), 0);

    // Same cache row width (2), different head factorization: the typed cache
    // retains its bound spec, so row-width coincidence cannot cross-wire layers.
    let other_weights = Qwen35FullAttentionWeights::new(
        dense(&[0.0; 16], 4, 4),
        dense(&[0.0; 8], 2, 4),
        dense(&[0.0; 8], 2, 4),
        dense(&[0.0; 8], 4, 2),
        vec![0.0; 2],
        vec![0.0; 2],
    );
    let other = Qwen35FullAttention::new(&text_config(1, 1, 2), other_weights).unwrap();
    let mut wrong_cache = other.new_cache(4).unwrap();
    assert!(
        layer
            .forward(&backend, &[0.0; 4], &[0], &mut wrong_cache, &mut sentinel)
            .is_err()
    );
    assert_eq!(wrong_cache.len(), 0);

    // Geometry equality is not provenance: all 16 flagship attention layers
    // share shapes but must never accept one another's KV state.
    let same_geometry_layer = official_layer();
    let mut wrong_instance_cache = same_geometry_layer.new_cache(4).unwrap();
    assert!(
        layer
            .forward(
                &backend,
                &[0.0; 4],
                &[0],
                &mut wrong_instance_cache,
                &mut sentinel,
            )
            .is_err()
    );
    assert!(wrong_instance_cache.is_empty());

    // RoPE coordinates and the causal cache watermark are separate. A language
    // suffix can start at a nonzero bounded position without inventing prefix KV.
    layer
        .forward(&backend, &[0.0; 4], &[7], &mut cache, &mut sentinel)
        .unwrap();
    assert_eq!(cache.len(), 1);

    assert!(layer.new_cache(0).is_err());
    assert!(layer.new_cache(33).is_err());
}

#[derive(Debug, Default)]
struct SwitchBackend {
    cpu: CpuBackend,
    fail: AtomicBool,
}

impl SwitchBackend {
    fn fail(&self) {
        self.fail.store(true, Ordering::SeqCst);
    }
}

impl TernaryBackend for SwitchBackend {
    fn device_id(&self) -> &str {
        "switchable-test-backend"
    }

    fn capabilities(&self) -> DeviceCaps {
        self.cpu.capabilities()
    }

    fn upload_weights(
        &self,
        packed: &[u8],
        shape: GemmShape,
        format: tritium_core::TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.cpu.upload_weights(packed, shape, format)
    }

    fn mpgemm(&self, p: MpGemm<'_>) -> Result<(), BackendError> {
        if self.fail.load(Ordering::SeqCst) {
            Err(BackendError::Backend("intentional test failure".to_owned()))
        } else {
            self.cpu.mpgemm(p)
        }
    }
}

#[test]
fn projection_failure_rolls_back_cache_and_does_not_publish_output() {
    let cpu = CpuBackend::new();
    let ternary_o =
        Projection::Ternary(TernaryLinear::new(&cpu, &[Trit::ZERO; 16], 4, 4, 1.0).unwrap());
    let weights = Qwen35FullAttentionWeights::new(
        dense_a8(&Q_PROJ, 8, 4),
        dense_a8(&K_PROJ, 2, 4),
        dense_a8(&V_PROJ, 2, 4),
        ternary_o,
        Q_NORM.to_vec(),
        K_NORM.to_vec(),
    );
    let layer = Qwen35FullAttention::new(&text_config(2, 1, 2), weights).unwrap();
    let backend = SwitchBackend::default();
    let mut cache = layer.new_cache(4).unwrap();
    let mut prefill_output = [0.0; 8];
    layer
        .forward(
            &backend,
            &[1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -0.25],
            &[0, 1],
            &mut cache,
            &mut prefill_output,
        )
        .unwrap();
    let committed_keys = cache.keys().to_vec();
    let committed_values = cache.values().to_vec();
    backend.fail();
    let mut output = [23.0; 4];

    let error = layer
        .forward(
            &backend,
            &[0.5, 1.5, -1.0, 0.75],
            &[2],
            &mut cache,
            &mut output,
        )
        .unwrap_err();

    assert!(matches!(error, NnError::Backend(_)));
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.keys(), committed_keys);
    assert_eq!(cache.values(), committed_values);
    assert_eq!(output, [23.0; 4]);
}
