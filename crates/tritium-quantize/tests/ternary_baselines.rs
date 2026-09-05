use tritium_core::Trit;
use tritium_quantize::{TernaryBaselineError, TtqState, TwnConfig, project_ttq, project_twn};

fn values(trits: &[Trit]) -> Vec<i8> {
    trits.iter().map(|trit| trit.get()).collect()
}

#[test]
fn twn_matches_threshold_and_selected_absmean_definition() {
    let weights = [-2.0, -0.5, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0];
    let projection = project_twn(&weights, 2, 4, TwnConfig::new(0.7).unwrap()).unwrap();
    assert_eq!(projection.rows(), 2);
    assert_eq!(projection.columns(), 4);
    assert_eq!(projection.planes().len(), 1);
    assert_eq!(
        values(projection.planes()[0].trits()),
        [-1, 0, 0, 1, 0, 0, 0, 0]
    );
    assert_eq!(projection.planes()[0].row_scales(), [2.0, 0.0]);
    assert_eq!(
        projection.decode(),
        [-2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn twn_threshold_comparison_is_strict() {
    let projection = project_twn(&[-1.0, 1.0], 1, 2, TwnConfig::new(1.0).unwrap()).unwrap();
    assert_eq!(values(projection.planes()[0].trits()), [0, 0]);
    assert_eq!(projection.decode(), [0.0, 0.0]);
}

#[test]
fn ttq_exports_two_asymmetric_sign_planes() {
    let weights = [-2.0, -0.5, 0.5, 2.0];
    let projection = project_ttq(&weights, 1, 4, TtqState::new(2.0, 3.0, 0.5).unwrap()).unwrap();
    assert_eq!(projection.planes().len(), 2);
    assert_eq!(values(projection.planes()[0].trits()), [0, 0, 0, 1]);
    assert_eq!(projection.planes()[0].row_scales(), [2.0]);
    assert_eq!(values(projection.planes()[1].trits()), [-1, 0, 0, 0]);
    assert_eq!(projection.planes()[1].row_scales(), [3.0]);
    assert_eq!(projection.decode(), [-3.0, 0.0, 0.0, 2.0]);
}

#[test]
fn baseline_projectors_fail_closed_on_invalid_input() {
    assert!(matches!(
        TwnConfig::new(0.0),
        Err(TernaryBaselineError::InvalidThresholdFactor)
    ));
    assert!(matches!(
        TtqState::new(1.0, 1.0, 1.0),
        Err(TernaryBaselineError::InvalidThresholdRatio)
    ));
    assert!(matches!(
        project_twn(&[1.0], usize::MAX, 2, TwnConfig::new(0.7).unwrap()),
        Err(TernaryBaselineError::ShapeOverflow { .. })
    ));
    assert!(matches!(
        project_ttq(
            &[1.0, f32::NAN],
            1,
            2,
            TtqState::new(1.0, 1.0, 0.5).unwrap()
        ),
        Err(TernaryBaselineError::NonFiniteWeight { index: 1 })
    ));
}
