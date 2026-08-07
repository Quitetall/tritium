//! HESTIA temperature-policy contract (ADR 0035 / plan 0054 WS-C2).

use tritium_train::{
    HestiaSensitivityProfile, RecoveryPhase, RecoverySchedule, TempSchedule, TemperatureError,
    TemperatureSchedule,
};

#[test]
fn exponential_temperature_schedule_hits_exact_anchors() {
    let schedule = TempSchedule::new(1.0, 0.01, 80).expect("valid temperature schedule");

    assert_eq!(schedule.tau(0), 1.0);
    assert!((schedule.tau(40) - 0.1).abs() < 1e-12);
    assert_eq!(schedule.tau(80), 0.01);
    assert_eq!(schedule.tau(800), 0.01);
}

#[test]
fn temperature_schedule_rejects_non_physical_configuration() {
    assert!(matches!(
        TempSchedule::new(f64::NAN, 0.01, 80),
        Err(TemperatureError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        TempSchedule::new(0.01, 1.0, 80),
        Err(TemperatureError::InvalidConfiguration(_))
    ));
    assert!(matches!(
        TempSchedule::new(1.0, 0.01, 0),
        Err(TemperatureError::InvalidConfiguration(_))
    ));
}

#[test]
fn exponential_schedule_stays_finite_across_extreme_valid_range() {
    let schedule = TempSchedule::new(f64::MAX, f64::MIN_POSITIVE, 2)
        .expect("finite positive endpoints remain representable in log space");
    let midpoint = schedule.tau(1);

    assert!(midpoint.is_finite());
    assert!(midpoint >= f64::MIN_POSITIVE);
    assert!(midpoint <= f64::MAX);
}

#[test]
fn standardized_sigmoid_is_ordered_and_centred() {
    let traces = [(-1.0_f64).exp(), 1.0, 1.0_f64.exp()];
    let profile = HestiaSensitivityProfile::standardized_sigmoid(&traces, 1.0, 1e-12)
        .expect("valid trace profile");
    let scores = profile.scores();

    assert!(scores[0] < scores[1] && scores[1] < scores[2]);
    assert_eq!(scores[1], 0.5);
    let expected_low = 1.0 / (1.0 + (1.5_f64).sqrt().exp());
    assert!((scores[0] - expected_low).abs() < 1e-12);
    assert!((scores[2] - (1.0 - expected_low)).abs() < 1e-12);
}

#[test]
fn standardized_sigmoid_rejects_missing_or_invalid_trace_signal() {
    for traces in [&[][..], &[0.0][..], &[f64::INFINITY][..]] {
        assert!(matches!(
            HestiaSensitivityProfile::standardized_sigmoid(traces, 1.0, 1e-12),
            Err(TemperatureError::InvalidConfiguration(_))
        ));
    }
}

#[test]
fn tensor_temperature_stays_sensitivity_ordered_then_reaches_floor_by_hard_phase() {
    let recovery = RecoverySchedule::new(100).expect("exact 80/20 schedule");
    let profile = HestiaSensitivityProfile::standardized_sigmoid(&[0.25, 1.0, 4.0], 1.0, 1e-12)
        .expect("valid trace profile");
    let schedule =
        TemperatureSchedule::new(recovery, 1.0, 0.01, 1.0, profile).expect("valid HESTIA schedule");

    let low = schedule.tau_at(0, 40).expect("low sensitivity temperature");
    let high = schedule
        .tau_at(2, 40)
        .expect("high sensitivity temperature");
    let scores = schedule.sensitivity_scores();
    assert!((low - 0.1 * scores[0].exp()).abs() < 1e-12);
    assert!((high - 0.1 * scores[2].exp()).abs() < 1e-12);
    assert!(high > low);
    assert_eq!(recovery.phase_at(80), Ok(RecoveryPhase::Hard));
    for tensor in 0..3 {
        assert_eq!(schedule.tau_at(tensor, 80), Ok(0.01));
        assert_eq!(schedule.tau_at(tensor, 99), Ok(0.01));
    }
    assert!(matches!(
        schedule.tau_at(3, 0),
        Err(TemperatureError::TensorOutOfRange { .. })
    ));
    assert!(matches!(
        schedule.tau_at(0, 100),
        Err(TemperatureError::StepOutOfRange { .. })
    ));
}
