//! Public failure-contract regressions for naive grouped-query attention.

use tritium_nn::gqa_attention;

#[test]
fn causal_extent_past_context_rejects_without_publishing_output() {
    let q = [1.0f32, 2.0];
    let k = [1.0f32];
    let v = [3.0f32];
    let mut out = [17.0f32, 19.0];
    let before = out;

    let result = gqa_attention(&q, &k, &v, 2, 1, 1, 1, 1, 1.0, 0, &mut out);

    assert!(result.is_err(), "causal_offset + seq must fit ctx");
    assert_eq!(
        out, before,
        "a rejected call must not publish partial output"
    );
}

#[test]
fn overflowing_causal_extent_rejects_without_publishing_output() {
    let q = [1.0f32];
    let k = [1.0f32];
    let v = [3.0f32];
    let mut out = [17.0f32];
    let before = out;

    let result = gqa_attention(&q, &k, &v, 1, 1, 1, 1, 1, 1.0, usize::MAX, &mut out);

    assert!(result.is_err(), "causal extent overflow must be typed");
    assert_eq!(out, before, "a rejected call must not publish output");
}

#[test]
fn zero_attention_dimensions_are_rejected() {
    for (label, n_head, n_head_kv, head_dim) in [
        ("query heads", 0, 1, 1),
        ("kv heads", 1, 0, 1),
        ("head width", 1, 1, 0),
    ] {
        let q = vec![0.0f32; n_head * head_dim];
        let k = vec![0.0f32; n_head_kv * head_dim];
        let v = k.clone();
        let mut out = vec![23.0f32; q.len()];
        let before = out.clone();

        let result = gqa_attention(
            &q, &k, &v, 1, 1, n_head, n_head_kv, head_dim, 1.0, 0, &mut out,
        );

        assert!(result.is_err(), "zero {label} must be rejected");
        assert_eq!(out, before, "zero {label} must not publish output");
    }
}

#[test]
fn nonfinite_scale_is_rejected_without_publishing_output() {
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let q = [1.0f32];
        let k = [2.0f32];
        let v = [3.0f32];
        let mut out = [29.0f32];
        let before = out;

        let result = gqa_attention(&q, &k, &v, 1, 1, 1, 1, 1, scale, 0, &mut out);

        assert!(
            result.is_err(),
            "nonfinite scale {scale:?} must be rejected"
        );
        assert_eq!(out, before, "a rejected scale must not publish output");
    }
}

#[test]
fn overflowing_dimension_products_return_errors_instead_of_panicking() {
    let result = gqa_attention(
        &[],
        &[0.0; 2],
        &[0.0; 2],
        1,
        1,
        usize::MAX,
        1,
        2,
        1.0,
        0,
        &mut [],
    );
    assert!(result.is_err(), "query row-width overflow must be typed");

    let q = [0.0f32; 4];
    let mut out = [31.0f32; 4];
    let before = out;
    let result = gqa_attention(
        &q,
        &[],
        &[],
        1,
        usize::MAX / 2 + 1,
        2,
        1,
        2,
        1.0,
        0,
        &mut out,
    );
    assert!(result.is_err(), "KV buffer-length overflow must be typed");
    assert_eq!(out, before, "an overflow must not publish output");
}

#[test]
fn empty_context_rejects_nonempty_sequence_without_publishing_output() {
    let q = [1.0f32];
    let mut out = [37.0f32];
    let before = out;

    let result = gqa_attention(&q, &[], &[], 1, 0, 1, 1, 1, 1.0, 0, &mut out);

    assert!(result.is_err(), "nonempty attention needs at least one key");
    assert_eq!(out, before, "an empty context must not publish output");
}

#[test]
fn empty_sequence_remains_a_valid_noop() {
    let mut out = [];

    gqa_attention(&[], &[], &[], 0, 0, 1, 1, 1, 1.0, usize::MAX, &mut out)
        .expect("an empty sequence must remain a no-op");
}

#[test]
fn valid_grouped_attention_keeps_its_exact_bits() {
    // Two query heads share one KV head. Zero query scores make the worked
    // causal probabilities exactly [1, 0] then [1/2, 1/2].
    let q = [0.0f32; 4];
    let k = [11.0f32, -7.0];
    let v = [2.0f32, 4.0];
    let mut out = [0.0f32; 4];

    gqa_attention(&q, &k, &v, 2, 2, 2, 1, 1, 1.0, 0, &mut out).expect("worked GQA case");

    assert_eq!(
        out.map(f32::to_bits),
        [
            2.0f32.to_bits(),
            2.0f32.to_bits(),
            3.0f32.to_bits(),
            3.0f32.to_bits()
        ]
    );
}
