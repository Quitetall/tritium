use half::f16;
use tritium_train::{
    AdamW, PvTernaryPlane, PvTernaryStructure, PvTernaryWeight, PvTuningConfig, PvTuningError,
    PvTuningSession,
};

fn adam(lr: f32) -> AdamW {
    AdamW {
        lr,
        beta1: 0.0,
        beta2: 0.0,
        eps: 1e-8,
        weight_decay: 0.0,
    }
}

fn pv_config(
    continuous: AdamW,
    code: AdamW,
    fraction: f32,
    trust_ratio: Option<f32>,
) -> Result<PvTuningConfig, PvTuningError> {
    let builder = PvTuningConfig::builder(continuous, code).max_code_change_fraction(fraction);
    match trust_ratio {
        Some(ratio) => builder.max_relative_code_change(ratio).build(),
        None => builder.build(),
    }
}

fn parent() -> PvTernaryWeight {
    PvTernaryWeight::new(
        1,
        4,
        4,
        PvTernaryStructure::Dense,
        vec![PvTernaryPlane::new(vec![1, -1, 1, -1], vec![f16::ONE])],
    )
    .unwrap()
}

#[test]
fn pv_step_runs_real_continuous_p_then_bounded_discrete_v() {
    let parent = parent();
    let config = pv_config(adam(0.1), adam(0.6), 0.5, None).unwrap();
    let mut session = PvTuningSession::new(parent, config).unwrap();

    let receipt = session.step(&[1.0, -1.0, 0.0, 0.0], 1).unwrap();

    assert_eq!(session.weight().planes()[0].trits(), &[0, 0, 1, -1]);
    assert_eq!(
        session.weight().planes()[0].scales()[0].to_bits(),
        f16::from_f32(0.9).to_bits()
    );
    assert_eq!(receipt.optimizer_step(), 1);
    assert_eq!(receipt.selected_code_units(), 2);
    assert_eq!(receipt.changed_code_units(), 2);
    assert_eq!(receipt.trust_limited_code_units(), 0);
    assert_eq!(receipt.changed_scales(), 1);
    assert!(receipt.v_surrogate_after() < receipt.v_surrogate_before());
    assert_eq!(receipt.representation_digest(), session.weight().digest());
}

#[test]
fn s34_v_step_moves_zero_without_breaking_per_plane_structure() {
    let parent = PvTernaryWeight::new(
        1,
        4,
        4,
        PvTernaryStructure::S34,
        vec![PvTernaryPlane::new(vec![0, 1, 1, 1], vec![f16::ONE])],
    )
    .unwrap();
    let config = pv_config(adam(0.001), adam(1.0), 1.0, None).unwrap();
    let mut session = PvTuningSession::new(parent, config).unwrap();

    let receipt = session.step(&[-1.0, 1.0, 0.0, 0.0], 1).unwrap();

    assert_eq!(session.weight().planes()[0].trits(), &[1, 0, 1, 1]);
    assert_eq!(
        session.weight().planes()[0]
            .trits()
            .iter()
            .filter(|&&trit| trit == 0)
            .count(),
        1
    );
    assert_eq!(receipt.selected_code_units(), 1);
    assert_eq!(receipt.changed_code_units(), 1);
    assert!(receipt.v_surrogate_after() < receipt.v_surrogate_before());
}

#[test]
fn rejected_steps_are_transactional() {
    let config = pv_config(adam(0.05), adam(0.4), 0.5, None).unwrap();
    let mut session = PvTuningSession::new(parent(), config).unwrap();
    let before = session.checkpoint_bytes().unwrap();

    assert!(matches!(
        session.step(&[0.0, f32::NAN, 0.0, 0.0], 1),
        Err(PvTuningError::Step(_))
    ));
    assert_eq!(session.checkpoint_bytes().unwrap(), before);
    assert!(matches!(
        session.step(&[0.0; 4], 2),
        Err(PvTuningError::Step(_))
    ));
    assert_eq!(session.checkpoint_bytes().unwrap(), before);
    assert_eq!(session.completed_step(), 0);
}

#[test]
fn constructors_reject_noncanonical_recipe_and_representation() {
    let negative_zero_scale = PvTernaryWeight::new(
        1,
        4,
        4,
        PvTernaryStructure::Dense,
        vec![PvTernaryPlane::new(
            vec![0; 4],
            vec![f16::from_bits(0x8000)],
        )],
    );
    assert!(matches!(
        negative_zero_scale,
        Err(PvTuningError::InvalidWeight(_))
    ));

    let broken_s34 = PvTernaryWeight::new(
        1,
        4,
        4,
        PvTernaryStructure::S34,
        vec![PvTernaryPlane::new(vec![0, 0, 1, 1], vec![f16::ONE])],
    );
    assert!(matches!(broken_s34, Err(PvTuningError::InvalidWeight(_))));

    let mut invalid_adam = adam(0.1);
    invalid_adam.weight_decay = -0.0;
    assert!(matches!(
        pv_config(invalid_adam, adam(0.1), 0.5, None),
        Err(PvTuningError::InvalidConfig(_))
    ));
}

#[test]
fn dense_multi_plane_v_step_enumerates_additive_code_tuples() {
    let parent = PvTernaryWeight::new(
        1,
        3,
        3,
        PvTernaryStructure::Dense,
        vec![
            PvTernaryPlane::new(vec![1, 1, 1], vec![f16::ONE]),
            PvTernaryPlane::new(vec![1, -1, 0], vec![f16::from_f32(0.5)]),
        ],
    )
    .unwrap();
    let config = pv_config(adam(0.001), adam(0.75), 1.0, None).unwrap();
    let mut session = PvTuningSession::new(parent, config).unwrap();

    let receipt = session.step(&[1.0, 1.0, -2.0], 1).unwrap();

    assert_eq!(session.weight().planes()[0].trits(), &[1, 0, 1]);
    assert_eq!(session.weight().planes()[1].trits(), &[-1, -1, 1]);
    assert_eq!(
        session.weight().planes()[0].scales()[0].to_bits(),
        f16::ONE.to_bits()
    );
    assert_eq!(
        session.weight().planes()[1].scales()[0].to_bits(),
        f16::from_f32(0.5).to_bits()
    );
    assert_eq!(receipt.changed_code_units(), 3);
    assert_eq!(receipt.changed_scales(), 0);
    assert!(receipt.v_surrogate_after() < receipt.v_surrogate_before());
}

#[test]
fn trust_ratio_reports_and_rejects_oversized_improving_moves() {
    let config = pv_config(adam(0.1), adam(0.6), 0.5, Some(0.1)).unwrap();
    let mut session = PvTuningSession::new(parent(), config).unwrap();

    let receipt = session.step(&[1.0, -1.0, 0.0, 0.0], 1).unwrap();

    assert_eq!(session.weight().planes()[0].trits(), &[1, -1, 1, -1]);
    assert_eq!(receipt.selected_code_units(), 2);
    assert_eq!(receipt.changed_code_units(), 0);
    assert_eq!(receipt.trust_limited_code_units(), 2);
    assert_eq!(receipt.relative_code_change(), 0.0);
    assert_eq!(receipt.v_surrogate_after(), receipt.v_surrogate_before());
}
