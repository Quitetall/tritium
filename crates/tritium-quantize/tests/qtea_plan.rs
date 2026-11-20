use tritium_format::qtea::{QteaError, QteaLimits};
use tritium_quantize::qtea_reference::{ArithmeticError, TensorFit, TensorFitConfig};
use tritium_quantize::{CurvatureSourceId, InputGram, InputGramAccumulator};

fn gram(seed: u8, scale: f32) -> InputGram {
    let source = CurvatureSourceId::new([seed; 32], [2; 32], [3; 32]).unwrap();
    let mut input = [0.; 16];
    for c in 0..4 {
        input[c * 4 + c] = if c == 0 { scale } else { 1. };
    }
    let mut builder = InputGramAccumulator::new_bound(4, source, 0).unwrap();
    builder.accumulate_batch(&input, 4, None, None).unwrap();
    builder.finish().unwrap()
}
fn config() -> TensorFitConfig {
    TensorFitConfig {
        group_size: 2,
        ..TensorFitConfig::default()
    }
}

#[test]
fn actual_values_gram_source_and_recipe_are_bound_not_only_shape() {
    let weights = [-1., 1., -2., 2.].repeat(4);
    let input = gram(1, 1.);
    let cap = TensorFit::required_work_bytes(4, 4).unwrap();
    let original = TensorFit::plan(&weights, 4, 4, &input, config(), cap).unwrap();
    let same = TensorFit::plan(&weights, 4, 4, &input, config(), cap).unwrap();
    assert_eq!(original.input_digest(), same.input_digest());
    let larger_cap = TensorFit::plan(&weights, 4, 4, &input, config(), cap + 1).unwrap();
    assert_eq!(original.input_digest(), larger_cap.input_digest()); // Operational budget is not mathematical identity.
    assert_eq!(original.metadata(), same.metadata());
    let mut changed = weights.clone();
    changed[0] = f32::from_bits(changed[0].to_bits() + 1);
    let alternate = TensorFit::plan(&changed, 4, 4, &input, config(), cap).unwrap();
    assert_ne!(original.input_digest(), alternate.input_digest());
    assert_ne!(
        original.metadata().bindings.weights,
        alternate.metadata().bindings.weights
    );
    for alternate in [gram(1, 2.), gram(9, 1.)] {
        let plan = TensorFit::plan(&weights, 4, 4, &alternate, config(), cap).unwrap();
        assert_ne!(plan.input_digest(), original.input_digest());
        assert_ne!(
            plan.metadata().bindings.gram,
            original.metadata().bindings.gram
        );
    }
    let alternate = TensorFit::plan(
        &weights,
        4,
        4,
        &input,
        TensorFitConfig {
            decay: 0.,
            ..config()
        },
        cap,
    )
    .unwrap();
    assert_ne!(alternate.input_digest(), original.input_digest());
    assert_ne!(
        alternate.metadata().recipe.digest(),
        original.metadata().recipe.digest()
    );
    let text = format!("{original:?}");
    assert!(text.contains("input_digest"));
    assert!(!text.contains("InputGram"));
    assert!(!text.contains("weights: ["));
    assert!(!text.contains("values:"));
}

#[test]
fn planning_refuses_caps_before_scans_and_wire_limits_before_fit() {
    let weights = [-1., 1., -2., 2.].repeat(4);
    let input = gram(1, 1.);
    let cap = TensorFit::required_work_bytes(4, 4).unwrap();
    let mut invalid = weights.clone();
    invalid[0] = f32::NAN;
    assert_eq!(
        TensorFit::plan(&invalid, 4, 4, &input, config(), cap - 1).unwrap_err(),
        ArithmeticError::ResourceLimit {
            required: cap,
            limit: cap - 1
        }
    );
    assert_eq!(
        TensorFit::plan(&invalid, 4, 4, &input, config(), cap).unwrap_err(),
        ArithmeticError::NonFiniteInput { index: 0 }
    );
    let plan = TensorFit::plan(&weights, 4, 4, &input, config(), cap).unwrap();
    assert_eq!(plan.required_work_bytes(), cap);
    let generous = QteaLimits {
        max_tensor_bytes: 4096,
        max_validation_bytes: 4,
    };
    let bytes = plan.encoded_ledger(generous).unwrap();
    assert_eq!(
        plan.encoded_ledger(QteaLimits {
            max_tensor_bytes: bytes.total_bytes - 1,
            ..generous
        }),
        Err(QteaError::TensorLimit)
    );
    assert_eq!(
        plan.encoded_ledger(QteaLimits {
            max_validation_bytes: 3,
            ..generous
        }),
        Err(QteaError::ValidationLimit)
    );
    assert_eq!(plan.metadata().byte_ledger(generous).unwrap(), bytes);
}

#[test]
fn a_plan_does_not_assert_positive_definiteness_or_fit_completion() {
    let source = CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap();
    let mut builder = InputGramAccumulator::new_bound(4, source, 0).unwrap();
    builder.accumulate_batch(&[1.; 4], 1, None, None).unwrap();
    let input = builder.finish().unwrap();
    let weights = [-1., 1., -2., 2.].repeat(4);
    let recipe = TensorFitConfig {
        damping: 0.,
        ..config()
    };
    let plan = TensorFit::plan(
        &weights,
        4,
        4,
        &input,
        recipe,
        TensorFit::required_work_bytes(4, 4).unwrap(),
    )
    .unwrap();
    assert_eq!(plan.fit().unwrap_err(), ArithmeticError::InvalidGram);
}
