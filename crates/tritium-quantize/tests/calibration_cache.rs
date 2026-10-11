use tritium_quantize::engine::{
    CalibrationBatch, CalibrationCache, CalibrationCacheBuilder, CalibrationCacheError,
    CalibrationCollector, CalibrationInput, CalibrationOutputBatch,
};
use tritium_schema::*;

fn request() -> CalibrationCacheRequest {
    CalibrationCacheRequest {
        tensor: QuantSourceTensor {
            name: "weight".into(),
            shape: vec![2, 3],
            precision: ScalePrecision::F32,
        },
        model: OriginalSourceId::from_bytes([7; 32]),
        consumer: CalibrationConsumer::Matmul,
        output: CalibrationOutput::DenseFisher,
        sample_start: 9,
        limits: CalibrationCacheLimits {
            max_payload_bytes: 1_000_000,
            max_samples: 10,
            max_windows: 10,
        },
    }
}
fn batch<'a>(
    input: &'a [f32],
    gradients: &'a [f32],
    weights: Option<&'a [f64]>,
    mask: Option<&'a [bool]>,
) -> CalibrationBatch<'a> {
    CalibrationBatch {
        samples: 2,
        input: CalibrationInput::Activations(input),
        output: CalibrationOutputBatch::Dense(gradients),
        weights,
        mask,
    }
}
fn build(
    input: &[f32],
    gradients: &[f32],
    weights: Option<&[f64]>,
    mask: Option<&[bool]>,
    tokenizer: &[u8],
    objective: &[u8],
) -> CalibrationCache {
    let mut builder = CalibrationCacheBuilder::new(request(), tokenizer, objective).unwrap();
    builder
        .capture(&[10, 11], batch(input, gradients, weights, mask))
        .unwrap();
    builder.finish().unwrap()
}

#[test]
fn actual_payloads_finalize_before_bound_replay_and_match_existing_reducers() {
    let input = [1., 2., 3., 4., 5., 6.];
    let gradients = [0.5, -0.25, 1.5, 2.];
    let weights = [0.5, 2.];
    let cache = build(
        &input,
        &gradients,
        Some(&weights),
        Some(&[true, true]),
        b"tokenizer",
        b"declared-test-objective",
    );
    let binding = cache.binding();
    assert_eq!(binding.source.cache.0, *cache.id().as_bytes());
    assert_eq!(binding.samples, 2);
    assert_eq!(binding.windows, 1);
    assert_eq!(cache.tokenizer(), b"tokenizer");
    assert_eq!(cache.objective(), b"declared-test-objective");
    let result = cache
        .collect(CalibrationPartition::Full, 0.125, 1_000_000)
        .unwrap();
    let chunked = cache
        .collect_chunked(CalibrationPartition::Full, 0.125, 1_000_000, 1)
        .unwrap();
    assert_eq!(result.factors(), chunked.factors());
    let mut reference = CalibrationCollector::new(CalibrationPlan {
        tensor: request().tensor,
        source: binding.source,
        partition: CalibrationPartition::Full,
        output: request().output,
        consumer: request().consumer,
        sample_start: 9,
        damping: 0.125,
        max_numeric_bytes: 1_000_000,
    })
    .unwrap();
    reference
        .accumulate(batch(
            &input,
            &gradients,
            Some(&weights),
            Some(&[true, true]),
        ))
        .unwrap();
    let expected = reference.finish().unwrap();
    assert_eq!(result.factors(), expected.factors());
    for row in 0..2 {
        assert_eq!(
            result.kfac_metric(0, row).unwrap(),
            expected.kfac_metric(0, row).unwrap()
        );
    }
    let grouped = cache
        .collect_chunked(
            CalibrationPartition::Groups { columns: 2 },
            0.125,
            1_000_000,
            1,
        )
        .unwrap();
    assert_eq!(grouped.factors().inputs.len(), 2);
    assert_eq!(grouped.factors().inputs[1].dimension, 1);
    assert!(
        cache
            .collect_chunked(CalibrationPartition::Full, 0.125, 1_000_000, 0)
            .is_err()
    );
}

#[test]
fn cache_identity_covers_context_tokens_pair_values_weights_masks_and_exact_recipe_bytes() {
    let input = [1., 2., 3., 4., 5., 6.];
    let gradients = [0.5, -0.25, 1.5, 2.];
    let original = build(&input, &gradients, None, None, b"tokenizer", b"objective");
    assert_eq!(
        original.id(),
        build(
            &input,
            &gradients,
            Some(&[1., 1.]),
            Some(&[true, true]),
            b"tokenizer",
            b"objective"
        )
        .id()
    );
    for (tokenizer, objective) in [
        (b"other".as_slice(), b"objective".as_slice()),
        (b"tokenizer".as_slice(), b"other".as_slice()),
    ] {
        assert_ne!(
            original.id(),
            build(&input, &gradients, None, None, tokenizer, objective).id()
        );
    }
    let mut changed = input;
    changed[0] = 7.;
    assert_ne!(
        original.id(),
        build(&changed, &gradients, None, None, b"tokenizer", b"objective").id()
    );
    let mut changed = gradients;
    changed[0] = 7.;
    assert_ne!(
        original.id(),
        build(&input, &changed, None, None, b"tokenizer", b"objective").id()
    );
    assert_ne!(
        original.id(),
        build(
            &input,
            &gradients,
            Some(&[1., 2.]),
            None,
            b"tokenizer",
            b"objective"
        )
        .id()
    );
    assert_ne!(
        original.id(),
        build(
            &input,
            &gradients,
            None,
            Some(&[false, true]),
            b"tokenizer",
            b"objective"
        )
        .id()
    );
    for change in 0..4 {
        let mut context = request();
        match change {
            0 => context.model = OriginalSourceId::from_bytes([8; 32]),
            1 => context.tensor.name = "other".into(),
            2 => context.tensor.precision = ScalePrecision::Bf16,
            _ => context.sample_start = 8,
        }
        let mut builder =
            CalibrationCacheBuilder::new(context, b"tokenizer", b"objective").unwrap();
        builder
            .capture(&[10, 11], batch(&input, &gradients, None, None))
            .unwrap();
        assert_ne!(original.id(), builder.finish().unwrap().id());
    }
    let mut builder = CalibrationCacheBuilder::new(request(), b"tokenizer", b"objective").unwrap();
    builder
        .capture(&[10, 12], batch(&input, &gradients, None, None))
        .unwrap();
    let changed_tokens = builder.finish().unwrap();
    assert_ne!(original.id(), changed_tokens.id());
    assert_ne!(
        original.binding().source.tokens,
        changed_tokens.binding().source.tokens
    );
    let mut higher_limit = request();
    higher_limit.limits.max_payload_bytes *= 2;
    let mut builder =
        CalibrationCacheBuilder::new(higher_limit, b"tokenizer", b"objective").unwrap();
    builder
        .capture(&[10, 11], batch(&input, &gradients, None, None))
        .unwrap();
    assert_eq!(original.id(), builder.finish().unwrap().id());
}

#[test]
fn invalid_masked_batches_and_limit_failures_do_not_advance_capture() {
    let input = [1., 2., 3., 4., 5., 6.];
    let gradients = [0.5, -0.25, 1.5, 2.];
    let reference = build(&input, &gradients, None, None, b"tokenizer", b"objective");
    let mut builder = CalibrationCacheBuilder::new(request(), b"tokenizer", b"objective").unwrap();
    let mut invalid = gradients;
    invalid[0] = f32::NAN;
    assert!(matches!(
        builder.capture(
            &[10, 11],
            batch(&input, &invalid, None, Some(&[false, true]))
        ),
        Err(CalibrationCacheError::Batch)
    ));
    assert!(
        builder
            .capture(&[10], batch(&input, &gradients, None, None))
            .is_err()
    );
    assert!(
        builder
            .capture(&[10, 11], batch(&input, &gradients, Some(&[-1., 1.]), None))
            .is_err()
    );
    assert!(
        builder
            .capture(&[10, 11], batch(&input[..5], &gradients, None, None))
            .is_err()
    );
    builder
        .capture(&[10, 11], batch(&input, &gradients, None, None))
        .unwrap();
    assert_eq!(reference.id(), builder.finish().unwrap().id());
    for limit in 0..3 {
        let mut context = request();
        match limit {
            0 => context.limits.max_payload_bytes = 20,
            1 => context.limits.max_samples = 1,
            _ => context.sample_start = u64::MAX,
        }
        let mut builder = CalibrationCacheBuilder::new(context, b"t", b"o").unwrap();
        assert!(
            builder
                .capture(&[10, 11], batch(&input, &gradients, None, None))
                .is_err()
        );
        assert!(matches!(
            builder.finish(),
            Err(CalibrationCacheError::Empty)
        ));
    }
}

#[test]
fn gather_identity_and_input_only_are_explicit_and_no_output_factors_are_invented() {
    let mut context = request();
    context.consumer = CalibrationConsumer::GatherIdentity;
    context.output = CalibrationOutput::IndexedFisher;
    let mut builder = CalibrationCacheBuilder::new(context, b"tokenizer", b"objective").unwrap();
    let batch = CalibrationBatch {
        samples: 2,
        input: CalibrationInput::GatherIdentity,
        output: CalibrationOutputBatch::Indexed {
            indices: &[0, 1],
            factors: &[1., 2.],
        },
        weights: None,
        mask: None,
    };
    builder.capture(&[10, 11], batch).unwrap();
    let cache = builder.finish().unwrap();
    let factors = cache
        .collect(CalibrationPartition::Full, 0.125, 1_000_000)
        .unwrap();
    assert_eq!(factors.factors().output.as_ref().unwrap().dimension, 2);
    let mut context = request();
    context.output = CalibrationOutput::None;
    let mut builder = CalibrationCacheBuilder::new(context, b"tokenizer", b"input-only").unwrap();
    builder
        .capture(
            &[10, 11],
            CalibrationBatch {
                samples: 2,
                input: CalibrationInput::Activations(&[1., 2., 3., 4., 5., 6.]),
                output: CalibrationOutputBatch::None,
                weights: None,
                mask: None,
            },
        )
        .unwrap();
    let cache = builder.finish().unwrap();
    assert!(
        cache
            .collect(CalibrationPartition::Full, 0., 1_000_000)
            .unwrap()
            .factors()
            .output
            .is_none()
    );
    assert!(
        cache
            .collect(CalibrationPartition::Full, 0.1, 1_000_000)
            .is_err()
    );
}
