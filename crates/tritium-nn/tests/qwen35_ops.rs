//! Public-op conformance for Qwen3.5 normalization and rotary embedding.

use tritium_nn::{rmsnorm, rmsnorm_zero_centered, rope_apply, rope_apply_partial_neox};

fn assert_close(got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= tolerance,
            "lane {index}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn zero_centered_rmsnorm_matches_qwen35_transformers_golden() {
    // Pinned from transformers 5.5.3 Qwen3_5RMSNorm in fp32. The two rows
    // exercise the official `norm(x.float()) * (1 + weight.float())` rule,
    // including a negative scale from weight = -1.5.
    let x = [1.0, -2.0, 3.0, -4.0, 0.25, -0.5, 0.75, -1.0];
    let weight = [0.0, 0.5, -0.25, -1.5];
    let expected = [
        0.365_148_37,
        -1.095_445_2,
        0.821_583_87,
        0.730_296_73,
        0.365_147_98,
        -1.095_444,
        0.821_583,
        0.730_295_96,
    ];
    let mut out = [0.0; 8];

    for (x, out) in x
        .as_chunks::<4>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<4>().0.iter_mut())
    {
        rmsnorm_zero_centered(x, &weight, 1e-6, out).expect("valid Qwen3.5 RMSNorm");
    }

    assert_close(&out, &expected, 2e-6);
}

#[test]
fn zero_centered_zero_weights_equal_ordinary_unit_weights() {
    let x = [3.0, 4.0, -5.0, 12.0];
    let mut zero_centered = [0.0; 4];
    let mut ordinary = [0.0; 4];

    rmsnorm_zero_centered(&x, &[0.0; 4], 1e-6, &mut zero_centered).unwrap();
    rmsnorm(&x, &[1.0; 4], 1e-6, &mut ordinary).unwrap();

    assert_eq!(zero_centered.map(f32::to_bits), ordinary.map(f32::to_bits));
}

#[test]
fn zero_centered_rmsnorm_rejects_weight_and_output_shape_errors() {
    let mut short_out = [0.0; 1];
    assert!(matches!(
        rmsnorm_zero_centered(&[1.0, 2.0], &[0.0], 1e-6, &mut [0.0; 2]),
        Err(tritium_nn::NnError::Shape { .. })
    ));
    assert!(matches!(
        rmsnorm_zero_centered(&[1.0, 2.0], &[0.0; 2], 1e-6, &mut short_out),
        Err(tritium_nn::NnError::Shape { .. })
    ));
}

#[test]
fn partial_neox_rope_matches_qwen35_transformers_golden() {
    // Pinned from transformers 5.5.3 `apply_rotary_pos_emb` with two tokens,
    // two heads, head_dim=8, rotary_dim=4, positions=[1, 3], theta=10000.
    let mut x = [
        0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 2.25, -2.5, 2.75, -3.0, 3.25, -3.5, 3.75,
        -4.0, -0.125, 0.375, -0.625, 0.875, -1.125, 1.375, -1.625, 1.875, 0.2, 0.4, 0.6, 0.8, 1.0,
        1.2, 1.4, 1.6,
    ];
    let expected = [
        -0.496_027_65,
        -0.489_975_15,
        0.615_594_5,
        -1.004_949_9,
        1.25,
        -1.5,
        1.75,
        -2.0,
        -1.098_365,
        -2.469_875_6,
        3.379_140_9,
        -3.024_849_7,
        3.25,
        -3.5,
        3.75,
        -4.0,
        0.211_949_07,
        0.348_585_2,
        0.601_105_33,
        0.885_854_6,
        -1.125,
        1.375,
        -1.625,
        1.875,
        -0.282_670_5,
        0.375_823_62,
        -0.565_771_5,
        0.811_638_24,
        1.0,
        1.2,
        1.4,
        1.6,
    ];

    rope_apply_partial_neox(&mut x, &[1, 3], 2, 8, 4, 10_000.0).expect("valid partial NeoX RoPE");

    assert_close(&x, &expected, 2e-6);
}

#[test]
fn partial_neox_rope_matches_qwen36_high_position_transformers_goldens() {
    // Frozen from Transformers 5.5.3 Qwen3_5TextRotaryEmbedding followed by
    // apply_rotary_pos_emb in fp32, using the pinned Qwen3.6-27B parameters:
    // head_dim=256, partial_rotary_factor=0.25, and rope_theta=10_000_000.
    // Each rotary half starts as (1, 0), so all 32 output pairs expose
    // Transformers' fp32 cosine and sine values directly.
    const POSITIONS: [usize; 4] = [4_096, 32_768, 131_072, 262_143];
    const EXPECTED_COS: [[f32; 32]; 4] = [
        [
            0.803_990_6,
            0.929_768_44,
            0.937_579_45,
            0.621_257_9,
            0.910_322_6,
            -0.978_916_5,
            -0.029_230_535,
            0.405_207_8,
            -0.835_532_84,
            0.999_434_23,
            0.104_672_6,
            -0.933_939,
            -0.958_708_94,
            0.915_693_76,
            -0.918_945_8,
            -0.541_850_1,
            0.272_054_4,
            0.708_993_5,
            0.890_206_2,
            0.959_427_6,
            0.985_119_76,
            0.994_557_56,
            0.998_011_4,
            0.999_273_66,
            0.999_734_76,
            0.999_903_14,
            0.999_964_65,
            0.999_987_07,
            0.999_995_3,
            0.999_998_3,
            0.999_999_4,
            0.999_999_76,
        ],
        [
            0.372_937_83,
            -0.992_136_6,
            -0.955_321_6,
            0.605_494_2,
            -0.963_161_4,
            -0.074_800_82,
            0.972_775_1,
            -0.980_842_4,
            -0.058_799_304,
            0.964_003_86,
            0.668_268_5,
            -0.976_449_9,
            -0.671_445_25,
            -0.986_099_36,
            -0.994_844_6,
            -0.130_917_95,
            -0.591_906_55,
            0.999_771_6,
            -0.800_662_1,
            -0.656_262_64,
            0.187_858_27,
            0.671_158_5,
            0.875_366_45,
            0.953_868_03,
            0.983_070_6,
            0.993_806_7,
            0.997_736_9,
            0.999_173_34,
            0.999_698_1,
            0.999_889_73,
            0.999_959_77,
            0.999_985_3,
        ],
        [
            0.042_090_815,
            0.876_643_54,
            0.362_170_1,
            -0.857_685,
            0.463_280_77,
            0.955_489_16,
            0.593_407_7,
            0.707_935_2,
            0.972_436_8,
            0.474_411_5,
            -0.977_172_8,
            0.644_967_3,
            -0.980_665_3,
            0.785_232_9,
            0.918_572_84,
            0.865_234,
            -0.820_846_9,
            0.996_347_25,
            -0.840_817,
            -0.961_558_64,
            0.727_637_65,
            -0.980_361_34,
            -0.432_817_67,
            0.343_909_23,
            0.740_439_65,
            0.902_434_1,
            0.963_994_74,
            0.986_801_27,
            0.995_173_4,
            0.998_236_54,
            0.999_355_9,
            0.999_764_8,
        ],
        [
            -0.609_161_5,
            0.923_175_3,
            -0.930_872_5,
            0.268_614_5,
            -0.674_451_53,
            0.868_371_25,
            -0.342_007_64,
            -0.026_949_033,
            0.883_157_9,
            -0.540_863_5,
            0.907_028_2,
            -0.171_883_7,
            0.922_492_7,
            0.231_786_44,
            0.688_183_5,
            0.496_809_66,
            0.347_872_88,
            0.985_383_3,
            0.413_840_62,
            0.849_152_7,
            0.058_871_25,
            0.922_226_55,
            -0.625_349_64,
            -0.763_446_87,
            0.096_507_39,
            0.628_777_3,
            0.858_572_66,
            0.947_553_9,
            0.980_740_37,
            0.992_952_47,
            0.997_424_5,
            0.999_059_26,
        ],
    ];
    const EXPECTED_SIN: [[f32; 32]; 4] = [
        [
            -0.594_642,
            -0.368_144_87,
            0.347_771_14,
            -0.783_606_2,
            -0.413_899_48,
            -0.204_260_66,
            -0.999_572_7,
            0.914_224_6,
            -0.549_440_5,
            0.033_634_31,
            0.994_506_7,
            -0.357_432_5,
            -0.284_389_14,
            -0.401_876_75,
            -0.394_383_85,
            0.840_475_14,
            0.962_281_9,
            0.705_215,
            0.455_557_76,
            0.281_955_2,
            0.171_869_31,
            0.104_188_81,
            0.063_033_57,
            0.038_107_004,
            0.023_031_466,
            0.013_918_611,
            0.008_411_139,
            0.005_082_859,
            0.003_071_562,
            0.001_856_135_7,
            0.001_121_656_5,
            0.000_677_813,
        ],
        [
            0.927_856_3,
            -0.125_159_71,
            0.295_568_3,
            -0.795_849_74,
            0.268_923_97,
            0.997_198_46,
            -0.231_751_28,
            0.194_802_88,
            -0.998_269_8,
            0.265_888_27,
            -0.743_920_15,
            0.215_744_23,
            0.741_054_2,
            0.166_156_83,
            -0.101_410_83,
            -0.991_393_2,
            -0.806_006_6,
            -0.021_372_9,
            -0.599_116_15,
            0.754_532_5,
            0.982_196_15,
            0.741_313_9,
            0.483_460_04,
            0.300_226_27,
            0.183_226_99,
            0.111_122_51,
            0.067_239_14,
            0.040_651_843,
            0.024_570_06,
            0.014_848_548,
            0.008_973_134,
            0.005_422_478,
        ],
        [
            -0.999_113_8,
            0.481_140_46,
            -0.932_112_04,
            0.514_175_4,
            -0.886_211_6,
            0.295_026_27,
            -0.804_902,
            -0.706_277_4,
            -0.233_166_77,
            0.880_303_2,
            0.212_445_96,
            -0.764_210_16,
            0.195_692_42,
            -0.619_200_5,
            0.395_251_7,
            -0.501_368_2,
            -0.571_148_16,
            -0.085_393_99,
            0.541_319_43,
            0.274_599_76,
            -0.685_961_66,
            -0.197_209_54,
            0.901_481_5,
            0.939_002_9,
            0.672_122_84,
            0.430_827_9,
            0.265_921_4,
            0.161_935_96,
            0.098_131_955,
            0.059_361_458,
            0.035_885_308,
            0.021_688_318,
        ],
        [
            0.793_046_2,
            0.384_379_3,
            -0.365_344_14,
            -0.963_247_8,
            -0.738_319_16,
            0.495_914_73,
            -0.939_697_15,
            -0.999_636_8,
            -0.469_075_8,
            0.841_110_35,
            -0.421_069_83,
            -0.985_117_26,
            -0.386_014_52,
            -0.972_766_7,
            0.725_536_7,
            -0.867_859_54,
            0.937_541_7,
            -0.170_352_07,
            -0.910_349_37,
            -0.528_147_46,
            -0.998_265_56,
            0.386_649_9,
            -0.780_344_67,
            0.645_870_6,
            0.995_332_24,
            0.777_585_4,
            0.512_691_9,
            0.319_596_05,
            0.195_315_88,
            0.118_513_1,
            0.071_724_12,
            0.043_366_27,
        ],
    ];

    let mut x = vec![0.0; POSITIONS.len() * 256];
    for head in x.as_chunks_mut::<256>().0 {
        head[..32].fill(1.0);
    }

    rope_apply_partial_neox(&mut x, &POSITIONS, 1, 256, 64, 10_000_000.0)
        .expect("valid Qwen3.6 partial NeoX RoPE");

    for (token, head) in x.as_chunks::<256>().0.iter().enumerate() {
        assert_close(&head[..32], &EXPECTED_COS[token], 2e-6);
        assert_close(&head[32..64], &EXPECTED_SIN[token], 2e-6);
    }
}

#[test]
fn partial_neox_rope_preserves_suffix_bits() {
    let suffix = [0x7fc1_2345, (-0.0f32).to_bits(), f32::INFINITY.to_bits(), 1];
    let mut x = [
        0.25,
        -0.5,
        0.75,
        -1.0,
        f32::from_bits(suffix[0]),
        f32::from_bits(suffix[1]),
        f32::from_bits(suffix[2]),
        f32::from_bits(suffix[3]),
    ];

    rope_apply_partial_neox(&mut x, &[7], 1, 8, 4, 10_000.0).unwrap();

    assert_eq!(
        x[4..].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        suffix
    );
}

#[test]
fn full_rope_preserves_legacy_f64_golden() {
    let input = [
        0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 2.25, -2.5, 2.75, -3.0, 3.25, -3.5, 3.75,
        -4.0,
    ];
    let mut wrapped = input;
    let expected_bits = [
        3_214_855_653,
        3_200_618_846,
        1_061_075_819,
        3_212_833_295,
        3_197_497_764,
        3_217_310_760,
        1_071_662_408,
        3_221_225_918,
        1_079_026_337,
        3_210_861_200,
        1_076_641_486,
        3_225_409_959,
        3_222_213_627,
        3_230_053_872,
        1_081_259_378,
        3_229_617_759,
    ];

    rope_apply(&mut wrapped, &[2, 11], 1, 8, 500_000.0).unwrap();

    assert_eq!(wrapped.map(f32::to_bits), expected_bits);
}

#[test]
fn full_rope_and_fp32_partial_rope_remain_semantically_equivalent() {
    let mut legacy = vec![0.0; 64];
    legacy[..32].fill(1.0);
    let mut transformers_fp32 = legacy.clone();

    rope_apply(&mut legacy, &[262_143], 1, 64, 10_000_000.0).unwrap();
    rope_apply_partial_neox(&mut transformers_fp32, &[262_143], 1, 64, 64, 10_000_000.0).unwrap();

    // Both APIs implement the same rotation, but `rope_apply` intentionally
    // retains its historical f64-derived table while the Qwen path follows
    // Transformers' fp32 table construction. Long-position phase error makes
    // bit identity an invalid contract; the observed semantic delta is <6e-3.
    assert_close(&legacy, &transformers_fp32, 6e-3);
}

#[test]
fn full_rope_preserves_empty_width_and_reports_extreme_odd_width() {
    let mut empty = [];
    rope_apply(&mut empty, &[0, 1], 3, 0, 500_000.0)
        .expect("historical zero-width layout is valid");

    assert!(matches!(
        rope_apply(&mut empty, &[], 0, usize::MAX, 500_000.0),
        Err(tritium_nn::NnError::Shape { .. })
    ));
    assert!(matches!(
        rope_apply(&mut empty, &[0], usize::MAX, 2, 500_000.0),
        Err(tritium_nn::NnError::Shape { .. })
    ));
}

#[test]
fn partial_neox_rope_rejects_invalid_dimensions_and_layout_overflow() {
    for (head_dim, rotary_dim) in [(8, 0), (8, 3), (8, 10), (7, 4)] {
        let mut x = vec![0.0; head_dim];
        assert!(matches!(
            rope_apply_partial_neox(&mut x, &[0], 1, head_dim, rotary_dim, 10_000.0),
            Err(tritium_nn::NnError::Shape { .. })
        ));
    }

    let mut wrong_layout = [0.0; 15];
    assert!(matches!(
        rope_apply_partial_neox(&mut wrong_layout, &[0], 2, 8, 4, 10_000.0),
        Err(tritium_nn::NnError::Shape { .. })
    ));

    let mut empty = [];
    assert!(matches!(
        rope_apply_partial_neox(&mut empty, &[0], usize::MAX, 2, 2, 10_000.0),
        Err(tritium_nn::NnError::Shape { .. })
    ));
}
