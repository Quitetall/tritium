use tritium_quantize::{
    ActivationCacheBuilder, ActivationCacheSpec, ActivationChunk, ActivationDType, ActivationDigest,
};

fn cache(dtype: ActivationDType) -> tritium_quantize::ActivationCache {
    let spec = ActivationCacheSpec::new(
        7,
        "model.layers.7.mlp.down_proj.input",
        5,
        2,
        dtype,
        ActivationDigest::from_bytes([9; 32]),
        2,
    )
    .expect("valid cache spec");
    let mut builder = ActivationCacheBuilder::new(spec.clone());
    builder
        .ingest(
            ActivationChunk::new(
                &spec,
                0,
                5,
                vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0],
                vec![true, false, true, true, false],
                vec![2, 5],
            )
            .expect("valid activation chunk"),
        )
        .expect("ingest activation chunk");
    builder.finalize().expect("complete cache")
}

#[test]
fn bounded_window_decodes_values_mask_and_global_sequence_ends() {
    let original = cache(ActivationDType::Float16);
    let cache = tritium_quantize::ActivationCache::from_encoded(original.encoded())
        .expect("reopen canonical cache");

    let window = cache
        .read_window(1, 3, 64)
        .expect("decode bounded activation window");

    assert_eq!(window.token_start(), 1);
    assert_eq!(window.token_count(), 3);
    assert_eq!(window.feature_width(), 2);
    assert_eq!(window.values(), &[2.0, -2.0, 3.0, -3.0, 4.0, -4.0]);
    assert_eq!(window.token_mask(), &[false, true, true]);
    assert_eq!(window.sequence_ends(), &[2]);
}

#[test]
fn bounded_window_decodes_bfloat16_and_rejects_oversized_or_invalid_ranges() {
    let cache = cache(ActivationDType::BFloat16);

    let window = cache
        .read_window(4, 1, 17)
        .expect("decode one-token window");
    assert_eq!(window.values(), &[5.0, -5.0]);
    assert_eq!(window.token_mask(), &[false]);
    assert_eq!(window.sequence_ends(), &[5]);
    assert_eq!(window.decoded_byte_estimate(), 17);

    assert!(matches!(
        cache.read_window(1, 3, 50),
        Err(
            tritium_quantize::ActivationCacheError::DecodedWindowLimitExceeded {
                required: 51,
                limit: 50,
            }
        )
    ));
    assert!(matches!(
        cache.read_window(5, 1, 64),
        Err(tritium_quantize::ActivationCacheError::WindowOutOfBounds {
            start: 5,
            count: 1,
            total: 5,
        })
    ));
    assert!(matches!(
        cache.read_window(0, 0, 64),
        Err(tritium_quantize::ActivationCacheError::EmptyChunk)
    ));
}

#[test]
fn bounded_window_decodes_f32_at_an_exact_sequence_boundary() {
    let cache = cache(ActivationDType::Float32);
    let window = cache
        .read_window(0, 2, 34)
        .expect("decode exact-budget f32 window");

    assert_eq!(window.values(), &[1.0, -1.0, 2.0, -2.0]);
    assert_eq!(window.token_mask(), &[true, false]);
    assert_eq!(window.sequence_ends(), &[2]);
    assert_eq!(window.decoded_byte_estimate(), 34);
}
