//! Dense reference conformance for Qwen3.5-family Gated DeltaNet.

use std::sync::atomic::{AtomicBool, Ordering};

use tritium_core::Trit;
use tritium_cpu::CpuBackend;
use tritium_nn::{
    DenseLinear, NnError, Projection, ProjectionActivationMode, Qwen35DeltaNet,
    Qwen35DeltaNetConfig, Qwen35DeltaNetWeights, Qwen35Dtype, Qwen35FullAttentionConfig,
    Qwen35LayerType, Qwen35MtpConfig, Qwen35NormWeightSemantics, Qwen35OutputGate,
    Qwen35RopeConfig, Qwen35RopeType, Qwen35TextConfig, TernaryLinear,
};
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, GemmShape, MpGemm, TernaryBackend};

const QKV_PROJ: [f32; 32] = [
    0.20, -0.10, 0.05, 0.30, -0.25, 0.40, 0.15, -0.05, 0.10, 0.20, -0.30, 0.25, -0.15, 0.05, 0.35,
    0.10, -0.30, 0.10, 0.25, 0.20, 0.05, -0.20, 0.30, 0.15, 0.40, -0.25, 0.10, -0.05, -0.10, 0.30,
    0.20, -0.15,
];
const Z_PROJ: [f32; 16] = [
    0.10, 0.20, -0.10, 0.05, -0.20, 0.10, 0.25, 0.15, 0.30, -0.15, 0.10, -0.25, 0.05, 0.20, 0.15,
    -0.10,
];
const B_PROJ: [f32; 8] = [0.25, -0.10, 0.20, 0.05, -0.15, 0.30, 0.10, -0.20];
const A_PROJ: [f32; 8] = [-0.20, 0.10, 0.15, 0.25, 0.10, -0.25, 0.30, -0.05];
const A_LOG: [f32; 2] = [-std::f32::consts::LN_2, 0.223_143_55];
const OUT_PROJ: [f32; 16] = [
    0.50, -0.10, 0.20, 0.0, 0.10, 0.30, -0.20, 0.40, -0.30, 0.20, 0.40, -0.10, 0.25, 0.0, 0.15,
    0.35,
];
const CONV: [f32; 32] = [
    0.10, -0.20, 0.30, 0.80, -0.15, 0.05, 0.20, 0.70, 0.05, 0.10, -0.10, 0.90, -0.20, 0.25, 0.15,
    0.65, 0.10, 0.05, 0.20, 0.75, -0.05, 0.15, -0.20, 0.85, 0.20, -0.10, 0.05, 0.80, -0.10, 0.20,
    0.10, 0.70,
];

fn text_config() -> Qwen35TextConfig {
    Qwen35TextConfig {
        model_type: "qwen3_5_text".to_owned(),
        num_hidden_layers: 1,
        hidden_size: 4,
        intermediate_size: 8,
        vocab_size: 16,
        max_position_embeddings: 16,
        full_attention_interval: 4,
        layer_types: vec![Qwen35LayerType::DeltaNet],
        full_attention: Qwen35FullAttentionConfig {
            num_heads: 1,
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
            partial_rotary_factor: 1.0,
            rotary_dim: 4,
            rope_type: Qwen35RopeType::Default,
            mrope_interleaved: true,
            mrope_section: [2, 0, 0],
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

fn official_weights() -> Qwen35DeltaNetWeights {
    Qwen35DeltaNetWeights::new(
        dense(&QKV_PROJ, 8, 4),
        dense(&Z_PROJ, 4, 4),
        dense(&B_PROJ, 2, 4),
        dense(&A_PROJ, 2, 4),
        dense(&OUT_PROJ, 4, 4),
        CONV.to_vec(),
        vec![1.10, 0.90],
        vec![-0.40, 0.20],
        A_LOG.to_vec(),
    )
}

fn official_layer() -> Qwen35DeltaNet {
    Qwen35DeltaNet::new(&text_config(), official_weights()).unwrap()
}

fn assert_close(got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (lane, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= tolerance,
            "lane {lane}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn transformers_5_5_3_prefill_and_incremental_goldens() {
    // Independently frozen from official Transformers 5.5.3
    // Qwen3_5GatedDeltaNet CPU fp32 fallback (source SHA-256
    // aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba).
    // Geometry exercises 1:2 query/key-to-value head repetition and a K=4
    // depthwise kernel whose newest raw QKV value multiplies tap K-1.
    let layer = official_layer();
    let backend = CpuBackend::new();
    let mut cache = layer.new_cache().unwrap();
    let mut prefill_out = [f32::NAN; 12];
    layer
        .forward(
            &backend,
            &[
                1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -0.25, 0.5, 1.5, -1.0, 0.75,
            ],
            3,
            &mut cache,
            &mut prefill_out,
        )
        .unwrap();

    assert_close(
        &prefill_out,
        &[
            -0.014_841_75,
            0.062_813_4,
            -0.019_188_508,
            0.017_104_98,
            -0.029_569_693,
            0.054_782_387,
            0.085_238_725,
            0.055_527_557,
            0.240_592_72,
            0.050_655_216,
            -0.114_517_88,
            0.145_351_37,
        ],
        2e-6,
    );
    assert_close(
        cache.conv_state(),
        &[
            0.0,
            0.862_500_1,
            -0.387_500_02,
            0.125,
            0.0,
            -0.512_500_05,
            0.875,
            0.287_5,
            0.0,
            0.425,
            -0.437_500_03,
            0.837_500_04,
            0.0,
            0.112_499_99,
            0.675,
            -0.275,
            0.0,
            0.112_499_98,
            0.787_5,
            -0.099_999_994,
            0.0,
            0.525,
            0.112_500_01,
            -0.462_5,
            0.0,
            0.450_000_02,
            -0.650_000_04,
            -0.312_5,
            0.0,
            -0.5,
            0.662_5,
            0.087_500_006,
        ],
        2e-6,
    );
    assert_close(
        cache.recurrent_state(),
        &[
            -0.008_753_672,
            -0.019_824_14,
            0.117_432_125,
            0.029_484_391,
            -0.051_991_09,
            -0.016_510_192,
            -0.047_485_854,
            0.070_001_12,
        ],
        2e-6,
    );
    assert_eq!(cache.len(), 3);

    let mut decode_out = [f32::NAN; 4];
    layer
        .forward(
            &backend,
            &[-0.75, 0.20, 1.10, -0.40],
            1,
            &mut cache,
            &mut decode_out,
        )
        .unwrap();
    assert_close(
        &decode_out,
        &[-0.064_640_11, 0.082_407_944, 0.044_142_3, 0.024_430_798],
        2e-6,
    );
    assert_close(
        cache.conv_state(),
        &[
            0.862_500_1,
            -0.387_500_02,
            0.125,
            -0.235,
            -0.512_500_05,
            0.875,
            0.287_5,
            0.452_500_02,
            0.425,
            -0.437_500_03,
            0.837_500_04,
            -0.465,
            0.112_499_99,
            0.675,
            -0.275,
            0.467_5,
            0.112_499_98,
            0.787_5,
            -0.099_999_994,
            0.44,
            0.525,
            0.112_500_01,
            -0.462_5,
            0.192_500_01,
            0.450_000_02,
            -0.650_000_04,
            -0.312_5,
            -0.220_000_01,
            -0.5,
            0.662_5,
            0.087_500_006,
            0.415_000_02,
        ],
        2e-6,
    );
    assert_close(
        cache.recurrent_state(),
        &[
            -0.050_060_652,
            -0.049_420_837,
            0.141_320_39,
            0.065_561_64,
            -0.010_263_69,
            -0.106_890_015,
            -0.022_483_414,
            0.150_035_07,
        ],
        2e-6,
    );
    assert_eq!(cache.len(), 4);
}

#[test]
fn one_shot_and_prefill_plus_decode_are_chunk_equivalent() {
    let layer = official_layer();
    let backend = CpuBackend::new();
    let input = [
        1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 1.25, -0.25, 0.5, 1.5, -1.0, 0.75, -0.75, 0.20, 1.10,
        -0.40,
    ];

    let mut one_shot_cache = layer.new_cache().unwrap();
    let mut one_shot = [f32::NAN; 16];
    layer
        .forward(&backend, &input, 4, &mut one_shot_cache, &mut one_shot)
        .unwrap();

    let mut streamed_cache = layer.new_cache().unwrap();
    let mut streamed = [f32::NAN; 16];
    layer
        .forward(
            &backend,
            &input[..12],
            3,
            &mut streamed_cache,
            &mut streamed[..12],
        )
        .unwrap();
    layer
        .forward(
            &backend,
            &input[12..],
            1,
            &mut streamed_cache,
            &mut streamed[12..],
        )
        .unwrap();

    // Official Transformers CPU chunk-vs-recurrent trace differs by at most
    // 1.49e-8 on this fixture. Tritium's scalar recurrence stays within fp32
    // conformance tolerance and ends at the same logical state.
    assert_close(&one_shot, &streamed, 2e-6);
    assert_close(
        one_shot_cache.conv_state(),
        streamed_cache.conv_state(),
        2e-6,
    );
    assert_close(
        one_shot_cache.recurrent_state(),
        streamed_cache.recurrent_state(),
        2e-6,
    );
}

#[test]
fn constructor_rejects_geometry_shapes_semantics_and_mixed_arithmetic() {
    let mut bad_geometry = text_config();
    bad_geometry.delta_net.num_key_heads = 2;
    bad_geometry.delta_net.num_value_heads = 3;
    assert!(matches!(
        Qwen35DeltaNet::new(&bad_geometry, official_weights()),
        Err(NnError::MissingConfig(_))
    ));

    let mut bad_semantics = text_config();
    bad_semantics.delta_net.state_arithmetic_dtype = Qwen35Dtype::Bfloat16;
    assert!(matches!(
        Qwen35DeltaNet::new(&bad_semantics, official_weights()),
        Err(NnError::MissingConfig(_))
    ));

    let wrong_qkv = Qwen35DeltaNetWeights::new(
        dense(&[0.0; 28], 7, 4),
        dense(&Z_PROJ, 4, 4),
        dense(&B_PROJ, 2, 4),
        dense(&A_PROJ, 2, 4),
        dense(&OUT_PROJ, 4, 4),
        CONV.to_vec(),
        vec![1.10, 0.90],
        vec![-0.40, 0.20],
        A_LOG.to_vec(),
    );
    assert!(matches!(
        Qwen35DeltaNet::new(&text_config(), wrong_qkv),
        Err(NnError::Shape {
            expected: 8,
            got: 7
        })
    ));

    let wrong_conv = Qwen35DeltaNetWeights::new(
        dense(&QKV_PROJ, 8, 4),
        dense(&Z_PROJ, 4, 4),
        dense(&B_PROJ, 2, 4),
        dense(&A_PROJ, 2, 4),
        dense(&OUT_PROJ, 4, 4),
        vec![0.0; 31],
        vec![1.10, 0.90],
        vec![-0.40, 0.20],
        A_LOG.to_vec(),
    );
    assert!(matches!(
        Qwen35DeltaNet::new(&text_config(), wrong_conv),
        Err(NnError::Shape {
            expected: 32,
            got: 31
        })
    ));

    let mixed = Qwen35DeltaNetWeights::new(
        dense(&QKV_PROJ, 8, 4),
        dense_a8(&Z_PROJ, 4, 4),
        dense(&B_PROJ, 2, 4),
        dense(&A_PROJ, 2, 4),
        dense(&OUT_PROJ, 4, 4),
        CONV.to_vec(),
        vec![1.10, 0.90],
        vec![-0.40, 0.20],
        A_LOG.to_vec(),
    );
    assert!(matches!(
        Qwen35DeltaNet::new(&text_config(), mixed),
        Err(NnError::MissingConfig(_))
    ));

    let uniform_a8 = Qwen35DeltaNetWeights::new(
        dense_a8(&QKV_PROJ, 8, 4),
        dense_a8(&Z_PROJ, 4, 4),
        dense_a8(&B_PROJ, 2, 4),
        dense_a8(&A_PROJ, 2, 4),
        dense_a8(&OUT_PROJ, 4, 4),
        CONV.to_vec(),
        vec![1.10, 0.90],
        vec![-0.40, 0.20],
        A_LOG.to_vec(),
    );
    let layer = Qwen35DeltaNet::new(&text_config(), uniform_a8).unwrap();
    assert_eq!(layer.activation_mode(), ProjectionActivationMode::A8);
}

#[test]
fn cache_provenance_layout_and_reset_are_fail_closed() {
    let layer = official_layer();
    let backend = CpuBackend::new();
    let other = official_layer();
    let mut foreign_cache = other.new_cache().unwrap();
    let mut sentinel = [17.0; 4];
    assert!(matches!(
        layer.forward(
            &backend,
            &[1.0, 0.0, 0.0, 0.0],
            1,
            &mut foreign_cache,
            &mut sentinel,
        ),
        Err(NnError::Backend(_))
    ));
    assert!(foreign_cache.is_empty());
    assert_eq!(sentinel, [17.0; 4]);

    let mut cache = layer.new_cache().unwrap();
    assert_eq!(cache.max_context(), 16);
    assert_eq!(cache.conv_kernel_dim(), 4);
    assert_eq!(cache.conv_width(), 8);
    assert_eq!(cache.num_key_heads(), 1);
    assert_eq!(cache.num_value_heads(), 2);
    assert_eq!(cache.key_head_dim(), 2);
    assert_eq!(cache.value_head_dim(), 2);

    assert!(matches!(
        layer.forward(&backend, &[], 0, &mut cache, &mut []),
        Err(NnError::Shape { .. })
    ));
    assert!(matches!(
        layer.forward(&backend, &[0.0; 3], 1, &mut cache, &mut sentinel),
        Err(NnError::Shape { .. })
    ));
    assert!(cache.is_empty());
    assert_eq!(sentinel, [17.0; 4]);

    layer
        .forward(
            &backend,
            &[1.0, -0.5, 0.25, 2.0],
            1,
            &mut cache,
            &mut sentinel,
        )
        .unwrap();
    let committed_conv = cache.conv_state().to_vec();
    let committed_recurrent = cache.recurrent_state().to_vec();
    let mut rejected = [23.0; 8];
    assert!(matches!(
        layer.forward(
            &backend,
            &[-1.5, 0.75, 1.25, -0.25, 0.5, 1.5, -1.0, 0.75],
            2,
            &mut cache,
            &mut rejected,
        ),
        Err(NnError::MissingConfig(_))
    ));
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.conv_state(), committed_conv);
    assert_eq!(cache.recurrent_state(), committed_recurrent);
    assert_eq!(rejected, [23.0; 8]);

    cache.reset();
    assert!(cache.is_empty());
    assert!(cache.conv_state().iter().all(|&value| value == 0.0));
    assert!(cache.recurrent_state().iter().all(|&value| value == 0.0));
}

#[test]
fn configured_context_limit_is_enforced_without_mutation() {
    let mut config = text_config();
    config.max_position_embeddings = 3;
    let layer = Qwen35DeltaNet::new(&config, official_weights()).unwrap();
    let backend = CpuBackend::new();
    let mut cache = layer.new_cache().unwrap();
    let mut prefill = [0.0; 12];
    layer
        .forward(&backend, &[0.0; 12], 3, &mut cache, &mut prefill)
        .unwrap();
    let committed_conv = cache.conv_state().to_vec();
    let committed_recurrent = cache.recurrent_state().to_vec();
    let mut out = [31.0; 4];

    assert!(matches!(
        layer.forward(&backend, &[0.0; 4], 1, &mut cache, &mut out),
        Err(NnError::Shape {
            expected: 3,
            got: 4
        })
    ));
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.conv_state(), committed_conv);
    assert_eq!(cache.recurrent_state(), committed_recurrent);
    assert_eq!(out, [31.0; 4]);
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
        "switchable-deltanet-test-backend"
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

    fn mpgemm(&self, parameters: MpGemm<'_>) -> Result<(), BackendError> {
        if self.fail.load(Ordering::SeqCst) {
            Err(BackendError::Backend("intentional test failure".to_owned()))
        } else {
            self.cpu.mpgemm(parameters)
        }
    }
}

#[test]
fn output_projection_failure_rolls_back_state_and_output() {
    let cpu = CpuBackend::new();
    let ternary_out =
        Projection::Ternary(TernaryLinear::new(&cpu, &[Trit::ZERO; 16], 4, 4, 1.0).unwrap());
    let weights = Qwen35DeltaNetWeights::new(
        dense_a8(&QKV_PROJ, 8, 4),
        dense_a8(&Z_PROJ, 4, 4),
        dense_a8(&B_PROJ, 2, 4),
        dense_a8(&A_PROJ, 2, 4),
        ternary_out,
        CONV.to_vec(),
        vec![1.10, 0.90],
        vec![-0.40, 0.20],
        A_LOG.to_vec(),
    );
    let layer = Qwen35DeltaNet::new(&text_config(), weights).unwrap();
    let backend = SwitchBackend::default();
    let mut cache = layer.new_cache().unwrap();
    let mut prefill = [0.0; 4];
    layer
        .forward(
            &backend,
            &[1.0, -0.5, 0.25, 2.0],
            1,
            &mut cache,
            &mut prefill,
        )
        .unwrap();
    let committed_conv = cache.conv_state().to_vec();
    let committed_recurrent = cache.recurrent_state().to_vec();
    backend.fail();
    let mut output = [29.0; 4];

    let error = layer
        .forward(
            &backend,
            &[-0.75, 0.20, 1.10, -0.40],
            1,
            &mut cache,
            &mut output,
        )
        .unwrap_err();

    assert!(matches!(error, NnError::Backend(_)));
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.conv_state(), committed_conv);
    assert_eq!(cache.recurrent_state(), committed_recurrent);
    assert_eq!(output, [29.0; 4]);
}
