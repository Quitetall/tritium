//! F64 reference feedback with an explicit shared decay policy; not model evidence.

use tritium_core::FeedbackDecay;
use tritium_quantize::{
    ColumnGroup, FeedbackMetric, FeedbackProblem, FeedbackState, fit_with_feedback,
    fit_with_feedback_decay,
};

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
    }
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
