//! F64 reference feedback with an explicit shared decay policy; not model evidence.

use tritium_core::FeedbackDecay;
use tritium_quantize::{
    ColumnGroup, FeedbackMetric, FeedbackProblem, FeedbackState, fit_with_feedback,
    fit_with_feedback_decay, fit_with_feedback_precision,
};
use tritium_schema::WorkPrecision;

fn problem<'a>(
    weights: &'a [f64],
    metric: &'a [f64],
    groups: &'a [ColumnGroup],
) -> FeedbackProblem<'a> {
    FeedbackProblem {
        rows: 1,
        columns: weights.len(),
        weights,
        groups,
        metric: FeedbackMetric::InverseHessian(metric),
    }
}

fn assert_close(left: &[f64], right: &[f64]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert!((left - right).abs() < 1e-12);
    }
}

#[test]
fn unit_decay_matches_the_default_public_interface() {
    let weights = [0.74, 0.74, 0.74];
    let metric = [1.0, 0.5, 0.25, 0.5, 1.0, 0.4, 0.25, 0.4, 1.0];
    let groups = [
        ColumnGroup { start: 0, end: 1 },
        ColumnGroup { start: 1, end: 2 },
        ColumnGroup { start: 2, end: 3 },
    ];
    let data = problem(&weights, &metric, &groups);
    let fit = |request: tritium_quantize::GroupFitRequest<'_>| {
        Ok::<_, ()>(
            request
                .working_weights
                .iter()
                .map(|v| (v * 2.0).round() * 0.5)
                .collect(),
        )
    };
    let original = fit_with_feedback(data, fit).unwrap();
    for ramp in [false, true] {
        let explicit =
            fit_with_feedback_decay(data, FeedbackDecay::new(1.0, ramp).unwrap(), fit).unwrap();
        assert_eq!(original, explicit);
        let precise = fit_with_feedback_precision(
            data,
            FeedbackDecay::new(1.0, ramp).unwrap(),
            WorkPrecision::F64,
            fit,
        )
        .unwrap();
        assert_eq!(original, precise);
    }
}

#[test]
fn f32_feedback_and_refits_preserve_explicit_arithmetic_and_atomic_failure() {
    let weights = [1.125, 1.5];
    let metric = [1.0, 0.5, 0.5, 1.0];
    let groups = [
        ColumnGroup { start: 0, end: 1 },
        ColumnGroup { start: 1, end: 2 },
    ];
    let mut state = fit_with_feedback_precision(
        problem(&weights, &metric, &groups),
        FeedbackDecay::default(),
        WorkPrecision::F32Accum64,
        |request| {
            Ok::<_, ()>(if request.group_index == 0 {
                vec![0.25]
            } else {
                request.working_weights.to_vec()
            })
        },
    )
    .unwrap();
    assert_eq!(state.precision(), WorkPrecision::F32Accum64);
    assert_eq!(state.working_weights(), &[1.125, 1.0625]);
    assert_eq!(state.reconstruction(), &[0.25, 1.0625]);
    state.replace_group_reconstruction(0, &[0.75]).unwrap();
    assert_eq!(state.working_weights(), &[1.125, 1.3125]);
    state
        .refit_suffix(1, |request| Ok::<_, ()>(request.working_weights.to_vec()))
        .unwrap();
    assert_eq!(state.reconstruction(), &[0.75, 1.3125]);
    let before = state.clone();
    for invalid in [f64::MAX, f64::NAN, f64::INFINITY] {
        assert!(state.replace_group_reconstruction(0, &[invalid]).is_err());
        assert_eq!(state, before);
    }
}

#[test]
fn f32_narrowing_is_not_silently_f64_and_residual_overflow_is_rejected() {
    let metric = [1.0];
    let groups = [ColumnGroup { start: 0, end: 1 }];
    let weights = [1.0000000001];
    for precision in [WorkPrecision::F32Accum64, WorkPrecision::F64] {
        let state = fit_with_feedback_precision(
            problem(&weights, &metric, &groups),
            FeedbackDecay::default(),
            precision,
            |request| Ok::<_, ()>(request.working_weights.to_vec()),
        )
        .unwrap();
        assert_eq!(
            state.working_weights()[0],
            if precision == WorkPrecision::F64 {
                weights[0]
            } else {
                1.0
            }
        );
    }
    let mut calls = 0;
    let too_large = [f64::MAX];
    assert!(
        fit_with_feedback_precision(
            problem(&too_large, &metric, &groups),
            FeedbackDecay::default(),
            WorkPrecision::F32Accum64,
            |request| {
                calls += 1;
                Ok::<_, ()>(request.working_weights.to_vec())
            }
        )
        .is_err()
    );
    assert_eq!(calls, 0);
    let huge = [f64::from(f32::MAX)];
    assert!(
        fit_with_feedback_precision(
            problem(&huge, &metric, &groups),
            FeedbackDecay::default(),
            WorkPrecision::F32Accum64,
            |_| Ok::<_, ()>(vec![-f64::from(f32::MAX)])
        )
        .is_err()
    );
}

#[test]
fn zero_decay_still_validates_the_metric_before_invoking_the_fitter() {
    let weights = [1.0, 1.0];
    let groups = [
        ColumnGroup { start: 0, end: 1 },
        ColumnGroup { start: 1, end: 2 },
    ];
    for metric in [[1.0, f64::NAN, f64::NAN, 1.0], [1.0, 2.0, 2.0, 1.0]] {
        let mut invoked = false;
        let result = fit_with_feedback_decay(
            problem(&weights, &metric, &groups),
            FeedbackDecay::new(0.0, false).unwrap(),
            |request| {
                invoked = true;
                Ok::<_, ()>(request.working_weights.to_vec())
            },
        );
        assert!(result.is_err());
        assert!(!invoked);
    }
}

#[test]
fn scalar_decay_uses_global_column_order_and_active_schur_metric() {
    let weights = [1.0, 1.0, 0.0];
    let metric = [2.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0];
    let groups = [
        ColumnGroup { start: 0, end: 1 },
        ColumnGroup { start: 1, end: 2 },
        ColumnGroup { start: 2, end: 3 },
    ];
    for (decay, ramp, expected) in [(0.0, false, 0.0), (0.5, false, -0.125), (0.5, true, -0.375)] {
        let result = fit_with_feedback_decay(
            problem(&weights, &metric, &groups),
            FeedbackDecay::new(decay, ramp).unwrap(),
            |request| {
                Ok::<_, ()>(if request.group_index < 2 {
                    vec![0.0]
                } else {
                    request.working_weights.to_vec()
                })
            },
        )
        .unwrap();
        assert_close(result.reconstruction(), &[0.0, 0.0, expected]);
    }
}

#[test]
fn multi_column_group_weights_each_residual_by_its_source_column() {
    let weights = [2.0, 1.0, 3.0];
    let metric = [2.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0];
    let groups = [
        ColumnGroup { start: 0, end: 2 },
        ColumnGroup { start: 2, end: 3 },
    ];
    for (ramp, expected) in [(false, 2.5), (true, 23.0 / 12.0)] {
        let result = fit_with_feedback_decay(
            problem(&weights, &metric, &groups),
            FeedbackDecay::new(0.5, ramp).unwrap(),
            |request| {
                Ok::<_, ()>(if request.group_index == 0 {
                    vec![0.0, 0.0]
                } else {
                    request.working_weights.to_vec()
                })
            },
        )
        .unwrap();
        assert_close(result.reconstruction(), &[0.0, 0.0, expected]);
    }
}

#[test]
fn refit_retains_decay_and_matches_clean_recomputation() {
    let weights = [0.74, 0.74, 0.74];
    let metric = [1.0, 0.5, 0.25, 0.5, 1.0, 0.4, 0.25, 0.4, 1.0];
    let groups = [
        ColumnGroup { start: 0, end: 1 },
        ColumnGroup { start: 1, end: 2 },
        ColumnGroup { start: 2, end: 3 },
    ];
    let data = problem(&weights, &metric, &groups);
    let decay = FeedbackDecay::new(0.5, true).unwrap();
    let round = |request: tritium_quantize::GroupFitRequest<'_>| {
        Ok::<_, ()>(
            request
                .working_weights
                .iter()
                .map(|v| (v * 2.0).round() * 0.5)
                .collect(),
        )
    };
    let mut state: FeedbackState = fit_with_feedback_decay(data, decay, round).unwrap();
    state.replace_group_reconstruction(0, &[0.25]).unwrap();
    state.refit_suffix(1, round).unwrap();
    let fresh = fit_with_feedback_decay(data, decay, |request| {
        if request.group_index == 0 {
            Ok(vec![0.25])
        } else {
            round(request)
        }
    })
    .unwrap();
    assert_close(state.working_weights(), fresh.working_weights());
    assert_close(state.reconstruction(), fresh.reconstruction());
    let before = state.clone();
    assert!(state.replace_group_reconstruction(0, &[f64::NAN]).is_err());
    assert_eq!(state, before);
}
