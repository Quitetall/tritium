use serde::Deserialize;
use tritium_quantize::qtea_reference::{ArithmeticError, TensorFit, TensorFitConfig};
use tritium_quantize::{CurvatureSourceId, InputGram, InputGramAccumulator};

#[derive(Deserialize)]
struct Fixtures {
    source: String,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
struct Case {
    pivot_changed: bool,
    damping: f32,
    rows: usize,
    columns: usize,
    group_size: usize,
    ratio: f64,
    decay: f64,
    weights: Vec<f32>,
    activations: Vec<f32>,
    order: Vec<usize>,
    trits: Vec<i8>,
    column_v: Vec<f32>,
    dense: Vec<f32>,
    salient_columns: Vec<usize>,
    residual_indices: Vec<Vec<u8>>,
}

fn binding(seed: u8) -> CurvatureSourceId {
    CurvatureSourceId::new([seed; 32], [2; 32], [3; 32]).unwrap()
}
fn gram(values: &[f32], columns: usize, source: Option<CurvatureSourceId>) -> InputGram {
    let mut builder = match source {
        Some(source) => InputGramAccumulator::new_bound(columns, source, 0).unwrap(),
        None => InputGramAccumulator::new(columns).unwrap(),
    };
    builder
        .accumulate_batch(values, values.len() / columns, None, None)
        .unwrap();
    builder.finish().unwrap()
}
fn identity(columns: usize) -> Vec<f32> {
    let mut values = vec![0.; columns * columns];
    for i in 0..columns {
        values[i * columns + i] = 1.;
    }
    values
}
fn config(width: usize) -> TensorFitConfig {
    TensorFitConfig {
        group_size: width,
        ..TensorFitConfig::default()
    }
}
fn near(actual: f32, expected: f32) {
    assert!(actual.is_finite() && expected.is_finite());
    assert!(
        (actual - expected).abs() <= 5e-5 * (1. + expected.abs()),
        "{actual:?} != {expected:?}"
    );
}

#[test]
fn complete_tensor_sweep_matches_pinned_author_driver() {
    let fixtures: Fixtures =
        serde_json::from_str(include_str!("fixtures/qtea_reference/tensor.json")).unwrap();
    assert_eq!(fixtures.source, "e39bd8c0297e3427daeb404427c74cb6e202b683");
    assert_eq!(fixtures.cases.len(), 72);
    let mut non_identity_orders = 0;
    let mut pivot_changes = 0;
    for case in fixtures.cases {
        pivot_changes += usize::from(case.pivot_changed);
        let input = gram(&case.activations, case.columns, Some(binding(1)));
        let recipe = TensorFitConfig {
            group_size: case.group_size,
            salient_ratio: case.ratio,
            decay: case.decay,
            damping: case.damping,
            ..TensorFitConfig::default()
        };
        let fitted = TensorFit::fit(
            &case.weights,
            case.rows,
            case.columns,
            &input,
            recipe,
            TensorFit::required_work_bytes(case.rows, case.columns).unwrap(),
        )
        .unwrap();
        assert_eq!(
            fitted.fitting_order(),
            case.order,
            "C={} G={} ratio={}",
            case.columns,
            case.group_size,
            case.ratio
        );
        non_identity_orders += usize::from(case.order != (0..case.columns).collect::<Vec<_>>());
        assert_eq!(
            fitted.trits().iter().map(|x| x.get()).collect::<Vec<_>>(),
            case.trits,
            "C={} G={} ratio={} decay={}",
            case.columns,
            case.group_size,
            case.ratio,
            case.decay
        );
        for (actual, expected) in fitted.column_multipliers().iter().zip(case.column_v) {
            near(*actual, expected);
        }
        assert_eq!(
            fitted
                .residuals()
                .iter()
                .map(|x| x.column())
                .collect::<Vec<_>>(),
            case.salient_columns
        );
        for (actual, expected) in fitted.residuals().iter().zip(case.residual_indices) {
            assert_eq!(actual.residual().indices(), expected);
        }
        let mut decoded = vec![0.; case.rows * case.columns];
        fitted.decode_into(&mut decoded).unwrap();
        for (actual, expected) in decoded.iter().zip(case.dense) {
            near(*actual, expected);
        }
        assert_eq!(fitted.rows(), case.rows);
        assert_eq!(fitted.columns(), case.columns);
        assert_eq!(fitted.curvature_source(), binding(1));
        assert_eq!(fitted.gram_digest(), input.digest());
        assert_eq!(fitted.config(), recipe);
        let mut wrong = [123.; 3];
        assert_eq!(
            fitted.decode_into(&mut wrong),
            Err(ArithmeticError::InvalidShape)
        );
        assert_eq!(wrong, [123.; 3]);
    }
    assert!(non_identity_orders > 0);
    assert!(
        pivot_changes > 0,
        "must exercise actual LDL swaps, not only saliency order"
    );
}

#[test]
fn streaming_gram_hash_preserves_canonical_identity_for_bound_and_unbound_evidence() {
    for source in [None, Some(binding(1))] {
        let evidence = gram(&[0., -0., 1., 2., 3., -4., 0.25, 1.], 2, source);
        let mut hash = blake3::Hasher::new_derive_key("tritium salt v2 input gram v1");
        hash.update(&evidence.canonical_bytes());
        assert_eq!(evidence.digest(), *hash.finalize().as_bytes());
    }
}

#[test]
fn fit_is_deterministic_and_actual_input_and_recipe_identities_change() {
    let weights = [-1., 1., -2., 2.].repeat(4);
    let evidence = gram(&identity(4), 4, Some(binding(1)));
    let bound = TensorFit::required_work_bytes(4, 4).unwrap();
    let first = TensorFit::fit(&weights, 4, 4, &evidence, config(2), bound).unwrap();
    assert_eq!(
        first,
        TensorFit::fit(&weights, 4, 4, &evidence, config(2), bound).unwrap()
    );
    let other = gram(&identity(4), 4, Some(binding(4)));
    let changed = TensorFit::fit(&weights, 4, 4, &other, config(2), bound).unwrap();
    assert_ne!(first.gram_digest(), changed.gram_digest());
    let mut alternate = weights.clone();
    alternate[0] = f32::from_bits(alternate[0].to_bits() + 1);
    let changed = TensorFit::fit(&alternate, 4, 4, &evidence, config(2), bound).unwrap();
    assert_ne!(first.weights_digest(), changed.weights_digest());
    let changed = TensorFit::fit(
        &weights,
        4,
        4,
        &evidence,
        TensorFitConfig {
            decay: 0.,
            ..config(2)
        },
        bound,
    )
    .unwrap();
    assert_ne!(first.recipe_digest(), changed.recipe_digest());
}

#[test]
fn missing_curvature_singular_geometry_budgets_and_nonfinite_inputs_fail_without_fallback() {
    let weights = [-1., 1., -2., 2.].repeat(4);
    let bound = TensorFit::required_work_bytes(4, 4).unwrap();
    let good = gram(&identity(4), 4, Some(binding(1)));
    let unbound = gram(&identity(4), 4, None);
    assert_eq!(
        TensorFit::fit(&weights, 4, 4, &unbound, config(2), bound),
        Err(ArithmeticError::UnboundGram)
    );
    assert_eq!(
        TensorFit::fit(&weights, 4, 4, &good, config(2), bound - 1),
        Err(ArithmeticError::ResourceLimit {
            required: bound,
            limit: bound - 1
        })
    );
    assert_eq!(
        TensorFit::required_work_bytes(3, 4),
        Err(ArithmeticError::InvalidShape)
    );
    assert_eq!(
        TensorFit::required_work_bytes(4, (1 << 24) + 1),
        Err(ArithmeticError::InvalidShape)
    );
    assert!(TensorFit::required_work_bytes(4, 8192).is_ok()); // No allocation/fit.
    let singular = gram(&[1.; 16], 4, Some(binding(1)));
    assert_eq!(
        TensorFit::fit(
            &weights,
            4,
            4,
            &singular,
            TensorFitConfig {
                damping: 0.,
                ..config(2)
            },
            bound
        ),
        Err(ArithmeticError::InvalidGram)
    );
    assert!(TensorFit::fit(&weights, 4, 4, &singular, config(2), bound).is_ok());
    assert_eq!(
        TensorFit::fit(&[1.; 16], 4, 4, &good, config(2), bound),
        Err(ArithmeticError::DegenerateRow { row: 0 })
    );
    let mut invalid = weights.clone();
    invalid[2] = f32::NAN;
    assert_eq!(
        TensorFit::fit(&invalid, 4, 4, &good, config(2), bound),
        Err(ArithmeticError::NonFiniteInput { index: 2 })
    );
    for damping in [-1., f32::NAN, f32::INFINITY] {
        assert_eq!(
            TensorFit::fit(
                &weights,
                4,
                4,
                &good,
                TensorFitConfig {
                    damping,
                    ..config(2)
                },
                bound
            ),
            Err(ArithmeticError::InvalidConfig)
        );
    }
    for floor in [0., -1., 1., f32::NAN] {
        assert_eq!(
            TensorFit::fit(
                &weights,
                4,
                4,
                &good,
                TensorFitConfig {
                    relative_pivot_floor: floor,
                    ..config(2)
                },
                bound
            ),
            Err(ArithmeticError::InvalidConfig)
        );
    }
}

#[test]
fn finite_decay_with_overflowing_exponent_rejects_instead_of_silently_zeroing_feedback() {
    let weights = [-3., -1., 1., 3.].repeat(4);
    let evidence = gram(&identity(4), 4, Some(binding(1)));
    let bound = TensorFit::required_work_bytes(4, 4).unwrap();
    assert_eq!(
        TensorFit::fit(
            &weights,
            4,
            4,
            &evidence,
            TensorFitConfig {
                decay: f64::MAX,
                ..config(4)
            },
            bound
        ),
        Err(ArithmeticError::NonFiniteArithmetic)
    );
}
