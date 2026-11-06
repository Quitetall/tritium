//! Common collector tests cross its actual seam and existing reducers, not a copied oracle.
use tritium_quantize::{
    CurvatureSourceId, IndexedOutputFisherAccumulator, InputGramAccumulator,
    OutputFisherAccumulator, build_kfac_metric,
    engine::{
        CalibrationBatch, CalibrationCollector, CalibrationError, CalibrationInput,
        CalibrationOutputBatch,
    },
};
use tritium_schema::{
    CalibrationConsumer, CalibrationOutput, CalibrationPartition, CalibrationPlan,
    CalibrationSource, FitInputDigest, QuantSourceTensor, ScalePrecision,
};

const X: [f32; 20] = [
    1., 2., 3., 4., 5., -2., 1., 0.5, 3., -1., 0., 2., -1., 0.25, 4., 0.5, -1., 2., 1., 3.,
];
const Y: [f32; 12] = [1., 0.5, -2., 0.25, 3., 1., -1., 0., 2., 0.5, -0.25, 1.];
const WEIGHTS: [f64; 4] = [1., 2., 0., 3.];
const MASK: [bool; 4] = [true, false, true, true];
fn plan(partition: CalibrationPartition, output: CalibrationOutput) -> CalibrationPlan {
    CalibrationPlan {
        tensor: QuantSourceTensor {
            name: "weight".into(),
            shape: vec![3, 5],
            precision: ScalePrecision::F32,
        },
        source: CalibrationSource {
            model: FitInputDigest([1; 32]),
            cache: FitInputDigest([2; 32]),
            tokens: FitInputDigest([3; 32]),
        },
        partition,
        output,
        consumer: CalibrationConsumer::Matmul,
        sample_start: 0,
        damping: if output == CalibrationOutput::None {
            0.
        } else {
            0.125
        },
        max_numeric_bytes: 4 * 1024 * 1024,
    }
}
fn source() -> CurvatureSourceId {
    CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap()
}
fn batch(start: usize, end: usize) -> CalibrationBatch<'static> {
    CalibrationBatch {
        samples: end - start,
        input: CalibrationInput::Activations(&X[start * 5..end * 5]),
        output: CalibrationOutputBatch::Dense(&Y[start * 3..end * 3]),
        weights: Some(&WEIGHTS[start..end]),
        mask: Some(&MASK[start..end]),
    }
}

#[test]
fn full_pair_is_bit_exact_to_actual_existing_reducers_and_row_metrics() {
    let mut collector = CalibrationCollector::new(plan(
        CalibrationPartition::Full,
        CalibrationOutput::DenseFisher,
    ))
    .unwrap();
    collector.accumulate(batch(0, 4)).unwrap();
    let result = collector.finish().unwrap();
    let mut input = InputGramAccumulator::new_bound(5, source(), 0).unwrap();
    input
        .accumulate_batch(&X, 4, Some(&WEIGHTS), Some(&MASK))
        .unwrap();
    let input = input.finish().unwrap();
    let mut output = OutputFisherAccumulator::new_bound(3, source(), 0).unwrap();
    output
        .accumulate_batch(&Y, 4, Some(&WEIGHTS), Some(&MASK))
        .unwrap();
    let output = output.finish().unwrap();
    assert_eq!(result.inputs(), std::slice::from_ref(&input));
    assert_eq!(result.output(), Some(&output));
    assert_eq!(result.factors().inputs[0].content.0, input.digest());
    assert_eq!(
        result.factors().output.as_ref().unwrap().content.0,
        output.digest()
    );
    let mut diagonal = [0.; 5];
    result.diagonal_into(&mut diagonal).unwrap();
    for (i, value) in diagonal.iter().enumerate() {
        assert_eq!(value.to_bits(), input.as_slice()[i * 5 + i].to_bits());
    }
    for row in 0..3 {
        let metric = result.kfac_metric(0, row).unwrap();
        let original = build_kfac_metric(&input, &output, row, 0.125).unwrap();
        assert_eq!(metric.canonical_bytes(), original.canonical_bytes());
    }
    assert_ne!(
        result.kfac_metric(0, 0).unwrap().digest(),
        result.kfac_metric(0, 1).unwrap().digest()
    );
}

#[test]
fn grouped_ragged_widths_and_diagonal_reuse_same_actual_reductions() {
    let mut collector = CalibrationCollector::new(plan(
        CalibrationPartition::Groups { columns: 2 },
        CalibrationOutput::DenseFisher,
    ))
    .unwrap();
    collector.accumulate(batch(0, 4)).unwrap();
    let result = collector.finish().unwrap();
    assert_eq!(
        result
            .inputs()
            .iter()
            .map(|g| g.dimension())
            .collect::<Vec<_>>(),
        [2, 2, 1]
    );
    for (group, start) in [0, 2, 4].into_iter().enumerate() {
        let width = (5 - start).min(2);
        let values: Vec<_> = X
            .as_chunks::<5>()
            .0
            .iter()
            .flat_map(|sample| sample[start..start + width].iter().copied())
            .collect();
        let mut expected = InputGramAccumulator::new_bound(width, source(), 0).unwrap();
        expected
            .accumulate_batch(&values, 4, Some(&WEIGHTS), Some(&MASK))
            .unwrap();
        assert_eq!(result.inputs()[group], expected.finish().unwrap());
        assert_eq!(result.factors().inputs[group].dimension, width as u64);
        result.kfac_metric(group, 2).unwrap();
    }
    let mut full = CalibrationCollector::new(plan(
        CalibrationPartition::Full,
        CalibrationOutput::DenseFisher,
    ))
    .unwrap();
    full.accumulate(batch(0, 4)).unwrap();
    let mut a = [0.; 5];
    let mut b = [0.; 5];
    result.diagonal_into(&mut a).unwrap();
    full.finish().unwrap().diagonal_into(&mut b).unwrap();
    assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
}

#[test]
fn invalid_masked_gradient_is_atomic_for_both_factors_and_ordinals() {
    let p = plan(CalibrationPartition::Full, CalibrationOutput::DenseFisher);
    let mut collector = CalibrationCollector::new(p.clone()).unwrap();
    collector.accumulate(batch(0, 1)).unwrap();
    let before = collector.finish().unwrap();
    let bad = [f32::NAN, 1., 2.];
    let mut invalid = batch(1, 2);
    invalid.output = CalibrationOutputBatch::Dense(&bad);
    assert!(collector.accumulate(invalid).is_err());
    assert_eq!(collector.finish().unwrap().factors(), before.factors());
    collector.accumulate(batch(1, 4)).unwrap();
    let mut expected = CalibrationCollector::new(p).unwrap();
    expected.accumulate(batch(0, 4)).unwrap();
    assert_eq!(
        collector.finish().unwrap().factors(),
        expected.finish().unwrap().factors()
    );
}

#[test]
fn invalid_last_group_is_atomic_and_does_not_advance_fisher() {
    let mut collector = CalibrationCollector::new(plan(
        CalibrationPartition::Groups { columns: 2 },
        CalibrationOutput::DenseFisher,
    ))
    .unwrap();
    collector.accumulate(batch(0, 1)).unwrap();
    let before = collector.finish().unwrap();
    let bad = [1., 2., 3., 4., f32::INFINITY];
    let mut invalid = batch(1, 2);
    invalid.input = CalibrationInput::Activations(&bad);
    assert!(collector.accumulate(invalid).is_err());
    let after = collector.finish().unwrap();
    assert_eq!(after.inputs(), before.inputs());
    assert_eq!(after.output(), before.output());
}

#[test]
fn batching_resume_and_shard_processing_order_preserve_actual_factor_identities() {
    let p = plan(
        CalibrationPartition::Groups { columns: 2 },
        CalibrationOutput::DenseFisher,
    );
    let mut all = CalibrationCollector::new(p.clone()).unwrap();
    all.accumulate(batch(0, 4)).unwrap();
    let expected = all.finish().unwrap();
    let mut split = CalibrationCollector::new(p.clone()).unwrap();
    split.accumulate(batch(0, 1)).unwrap();
    split.accumulate(batch(1, 4)).unwrap();
    assert_eq!(split.finish().unwrap().factors(), expected.factors());
    let mut first = CalibrationCollector::new(p.clone()).unwrap();
    first.accumulate(batch(0, 2)).unwrap();
    let mut tail_plan = p.clone();
    tail_plan.sample_start = 2;
    let mut tail = CalibrationCollector::new(tail_plan).unwrap();
    tail.accumulate(batch(2, 4)).unwrap();
    let mut merged = CalibrationCollector::new(p).unwrap();
    merged.merge_shard(&tail).unwrap();
    assert!(merged.finish().is_err());
    merged.merge_shard(&first).unwrap();
    assert_eq!(merged.finish().unwrap().factors(), expected.factors());
    assert_eq!(merged.finish().unwrap().inputs(), expected.inputs());
}

#[test]
fn source_selection_and_overlap_errors_leave_parent_unchanged() {
    let p = plan(CalibrationPartition::Full, CalibrationOutput::DenseFisher);
    let mut parent = CalibrationCollector::new(p.clone()).unwrap();
    parent.accumulate(batch(0, 2)).unwrap();
    let before = parent.finish().unwrap();
    let mut overlapping = CalibrationCollector::new(p.clone()).unwrap();
    overlapping.accumulate(batch(0, 2)).unwrap();
    assert!(parent.merge_shard(&overlapping).is_err());
    let mut other_plan = p.clone();
    other_plan.sample_start = 2;
    other_plan.source.tokens = FitInputDigest([9; 32]);
    let mut other = CalibrationCollector::new(other_plan).unwrap();
    other.accumulate(batch(2, 4)).unwrap();
    assert!(matches!(
        parent.merge_shard(&other),
        Err(CalibrationError::Shard)
    ));
    let after = parent.finish().unwrap();
    assert_eq!(after.factors(), before.factors());
    assert_eq!(after.inputs(), before.inputs());
    assert_eq!(after.output(), before.output());
    let mut malformed = batch(2, 4);
    malformed.weights = Some(&[1.]);
    assert!(parent.accumulate(malformed).is_err());
    assert_eq!(parent.finish().unwrap().factors(), before.factors());
}

#[test]
fn indexed_output_factors_match_actual_sparse_reducer_not_dense_expansion() {
    let indices = [0, 2, 2, 0];
    let values = [1., 0.5, -0.25, 0.75];
    let mut collector = CalibrationCollector::new(plan(
        CalibrationPartition::Full,
        CalibrationOutput::IndexedFisher,
    ))
    .unwrap();
    let mut b = batch(0, 4);
    b.output = CalibrationOutputBatch::Indexed {
        indices: &indices,
        factors: &values,
    };
    collector.accumulate(b).unwrap();
    let result = collector.finish().unwrap();
    let mut expected = IndexedOutputFisherAccumulator::new_bound(3, source(), 0).unwrap();
    expected
        .accumulate_batch(&indices, &values, 4, Some(&WEIGHTS), Some(&MASK))
        .unwrap();
    assert_eq!(result.output(), Some(&expected.finish().unwrap()));
    assert_eq!(result.output().unwrap().as_slice()[1], 0.);
    result.kfac_metric(0, 1).unwrap(); // positive damping admits a zero factor
    let mut p = result.factors().plan.clone();
    p.damping = 0.;
    let mut zero = CalibrationCollector::new(p).unwrap();
    zero.accumulate(b).unwrap();
    assert!(zero.finish().unwrap().kfac_metric(0, 1).is_err());
}

#[test]
fn explicit_gather_identity_uses_sparse_rows_without_one_hot_activations() {
    let mut p = plan(
        CalibrationPartition::Groups { columns: 2 },
        CalibrationOutput::IndexedFisher,
    );
    p.consumer = CalibrationConsumer::GatherIdentity;
    let mut collector = CalibrationCollector::new(p).unwrap();
    collector
        .accumulate(CalibrationBatch {
            samples: 2,
            input: CalibrationInput::GatherIdentity,
            output: CalibrationOutputBatch::Indexed {
                indices: &[0, 2],
                factors: &[1., 2.],
            },
            weights: None,
            mask: None,
        })
        .unwrap();
    let result = collector.finish().unwrap();
    let mut diagonal = [0.; 5];
    result.diagonal_into(&mut diagonal).unwrap();
    assert_eq!(diagonal, [1.; 5]);
    assert_eq!(result.output().unwrap().as_slice(), [0.5, 0., 2.]);
    for gram in result.inputs() {
        let n = gram.dimension();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(gram.as_slice()[i * n + j], if i == j { 1. } else { 0. });
            }
        }
    }
    let before = result.factors().clone();
    assert!(collector.accumulate(batch(0, 1)).is_err());
    assert_eq!(collector.finish().unwrap().factors(), &before);
}

#[test]
fn explicit_input_only_statistics_never_claim_or_substitute_a_fisher() {
    let mut collector =
        CalibrationCollector::new(plan(CalibrationPartition::Full, CalibrationOutput::None))
            .unwrap();
    let mut b = batch(0, 4);
    b.output = CalibrationOutputBatch::None;
    collector.accumulate(b).unwrap();
    let result = collector.finish().unwrap();
    assert!(result.output().is_none());
    assert!(result.factors().output.is_none());
    assert!(matches!(
        result.kfac_metric(0, 0),
        Err(CalibrationError::MissingFisher)
    ));
    assert!(collector.accumulate(batch(0, 1)).is_err());
}

#[test]
fn numerical_budget_rejects_before_state_and_group_scratch_mutation() {
    let mut p = plan(CalibrationPartition::Full, CalibrationOutput::DenseFisher);
    let required = CalibrationCollector::required_numeric_bytes(&p).unwrap();
    p.max_numeric_bytes = required - 1;
    assert!(matches!(
        CalibrationCollector::new(p),
        Err(CalibrationError::Budget { .. })
    ));
    let mut p = plan(
        CalibrationPartition::Groups { columns: 2 },
        CalibrationOutput::DenseFisher,
    );
    p.max_numeric_bytes = CalibrationCollector::required_numeric_bytes(&p).unwrap() + 8;
    let mut collector = CalibrationCollector::new(p).unwrap();
    collector.accumulate(batch(0, 1)).unwrap();
    let before = collector.finish().unwrap();
    assert!(matches!(
        collector.accumulate(batch(1, 4)),
        Err(CalibrationError::Budget { .. })
    ));
    assert_eq!(collector.finish().unwrap().factors(), before.factors());
}

#[test]
fn invalid_requests_and_overflow_fail_without_geometry_fallback() {
    let base = plan(CalibrationPartition::Full, CalibrationOutput::DenseFisher);
    for field in 0..8 {
        let mut p = base.clone();
        match field {
            0 => p.source.model = FitInputDigest([0; 32]),
            1 => p.source.cache = FitInputDigest([0; 32]),
            2 => p.source.tokens = FitInputDigest([0; 32]),
            3 => p.partition = CalibrationPartition::Groups { columns: 0 },
            4 => p.tensor.shape = vec![5],
            5 => p.damping = f64::NAN,
            6 => p.consumer = CalibrationConsumer::GatherIdentity,
            7 => p.max_numeric_bytes = 0,
            _ => unreachable!(),
        }
        assert!(CalibrationCollector::new(p).is_err());
    }
    let mut p = base.clone();
    p.tensor.shape = vec![1, u64::MAX / 4];
    assert!(matches!(
        CalibrationCollector::new(p),
        Err(CalibrationError::Allocation)
    ));
    let mut p = base;
    p.output = CalibrationOutput::None;
    assert!(CalibrationCollector::new(p).is_err()); // unused damping is not silently accepted
}

#[test]
fn destination_shape_rejection_and_zero_weight_are_fail_closed() {
    let mut collector = CalibrationCollector::new(plan(
        CalibrationPartition::Full,
        CalibrationOutput::DenseFisher,
    ))
    .unwrap();
    let mut b = batch(0, 4);
    b.weights = Some(&[0.; 4]);
    collector.accumulate(b).unwrap();
    assert!(collector.finish().is_err());
    let mut valid = CalibrationCollector::new(plan(
        CalibrationPartition::Full,
        CalibrationOutput::DenseFisher,
    ))
    .unwrap();
    valid.accumulate(batch(0, 4)).unwrap();
    let result = valid.finish().unwrap();
    let mut out = [99.; 4];
    assert!(result.diagonal_into(&mut out).is_err());
    assert_eq!(out, [99.; 4]);
    assert!(result.kfac_metric(1, 0).is_err());
    assert!(result.kfac_metric(0, 3).is_err());
}

#[test]
fn candidate_views_round_trip_and_reject_unknown_nested_fields() {
    use tritium_schema::CalibrationFactors;
    let mut collector = CalibrationCollector::new(plan(
        CalibrationPartition::Groups { columns: 2 },
        CalibrationOutput::DenseFisher,
    ))
    .unwrap();
    collector.accumulate(batch(0, 4)).unwrap();
    let result = collector.finish().unwrap();
    let encoded = serde_json::to_vec(result.factors()).unwrap();
    assert_eq!(
        serde_json::from_slice::<CalibrationFactors>(&encoded).unwrap(),
        *result.factors()
    );
    for field in [
        "",
        "/plan",
        "/plan/source",
        "/plan/tensor",
        "/inputs/0",
        "/output",
    ] {
        let mut value = serde_json::to_value(result.factors()).unwrap();
        value
            .pointer_mut(field)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("invented_receipt".into(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<CalibrationFactors>(value).is_err());
    }
    let mut value = serde_json::to_value(result.factors()).unwrap();
    value["plan"]["partition"]["Groups"]["invented_group"] = true.into();
    assert!(serde_json::from_value::<CalibrationFactors>(value).is_err());
}

#[test]
fn oversized_group_is_one_real_factor_not_a_legacy_alignment_restriction() {
    let mut full = CalibrationCollector::new(plan(
        CalibrationPartition::Full,
        CalibrationOutput::DenseFisher,
    ))
    .unwrap();
    let mut grouped = CalibrationCollector::new(plan(
        CalibrationPartition::Groups { columns: u32::MAX },
        CalibrationOutput::DenseFisher,
    ))
    .unwrap();
    full.accumulate(batch(0, 4)).unwrap();
    grouped.accumulate(batch(0, 4)).unwrap();
    let (full, grouped) = (full.finish().unwrap(), grouped.finish().unwrap());
    assert_eq!(full.inputs(), grouped.inputs());
    assert_eq!(full.output(), grouped.output());
    // Numerical equality does not rewrite the explicitly requested partition.
    assert_ne!(
        full.factors().plan.partition,
        grouped.factors().plan.partition
    );
}

#[test]
fn overflowing_sample_ordinal_rejects_the_pair_without_losing_previous_factors() {
    let mut p = plan(CalibrationPartition::Full, CalibrationOutput::DenseFisher);
    p.sample_start = u64::MAX - 1;
    let mut collector = CalibrationCollector::new(p).unwrap();
    collector.accumulate(batch(0, 1)).unwrap();
    let before = collector.finish().unwrap();
    assert!(matches!(
        collector.accumulate(batch(1, 2)),
        Err(CalibrationError::Curvature(_))
    ));
    assert_eq!(collector.finish().unwrap().factors(), before.factors());
}

#[test]
fn bounded_shard_segments_reject_atomically_without_advancing_local_ordinal() {
    let p = plan(CalibrationPartition::Full, CalibrationOutput::DenseFisher);
    let mut collector = CalibrationCollector::new(p.clone()).unwrap();
    for ordinal in 0..64 {
        let mut shard_plan = p.clone();
        shard_plan.sample_start = ordinal;
        let mut shard = CalibrationCollector::new(shard_plan).unwrap();
        shard.accumulate(batch(0, 1)).unwrap();
        collector.merge_shard(&shard).unwrap();
    }
    let before = collector.finish().unwrap();
    let mut shard_plan = p;
    shard_plan.sample_start = 64;
    let mut extra = CalibrationCollector::new(shard_plan).unwrap();
    extra.accumulate(batch(0, 1)).unwrap();
    assert!(matches!(
        collector.merge_shard(&extra),
        Err(CalibrationError::Segments)
    ));
    assert_eq!(collector.finish().unwrap().factors(), before.factors());
    // A failed merge has not silently advanced either factor's local ordinal.
    assert!(collector.accumulate(batch(0, 1)).is_err()); // ordinal zero still overlaps
    assert_eq!(collector.finish().unwrap().factors(), before.factors());
}
