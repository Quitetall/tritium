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

    for (x, out) in x.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
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
fn full_rope_wrapper_is_bit_identical_to_full_width_partial_rope() {
    let input = [
        0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 2.25, -2.5, 2.75, -3.0, 3.25, -3.5, 3.75,
        -4.0,
    ];
    let mut wrapped = input;
    let mut explicit = input;

    rope_apply(&mut wrapped, &[2, 11], 1, 8, 500_000.0).unwrap();
    rope_apply_partial_neox(&mut explicit, &[2, 11], 1, 8, 8, 500_000.0).unwrap();

    assert_eq!(wrapped.map(f32::to_bits), explicit.map(f32::to_bits));
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
