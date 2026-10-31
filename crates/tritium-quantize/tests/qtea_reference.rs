use serde::Deserialize;
use tritium_quantize::qtea_reference::{
    ARITHMETIC_ID, ArithmeticError, GroupParameters, SalientRanking, SparseResidualColumn,
};

#[derive(Deserialize)]
struct Fixtures {
    source: String,
    groups: Vec<GroupFixture>,
    rounding: Vec<(f32, u8, f32)>,
    sparse: Vec<SparseFixture>,
    ranking: Vec<RankingFixture>,
}

#[derive(Deserialize)]
struct RankingFixture {
    weights: Vec<f32>,
    rows: usize,
    columns: usize,
    diagonal: Vec<f32>,
    group_size: usize,
    ratio: f64,
    order: Vec<usize>,
    salient_count: usize,
}

#[derive(Deserialize)]
struct GroupFixture {
    weights: Vec<f32>,
    rows: usize,
    columns: usize,
    coefficient: f32,
    iterations: u32,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    delta: Vec<f32>,
    refinements: Vec<ColumnFixture>,
}

#[derive(Deserialize)]
struct ColumnFixture {
    column: Vec<f32>,
    iterations: u32,
    trits: Vec<i8>,
    multiplier: f32,
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct SparseFixture {
    residual: Vec<f32>,
    indices: Vec<u8>,
    codes: Vec<u8>,
    scale: f32,
    dense: Vec<f32>,
}

fn fixtures() -> Fixtures {
    serde_json::from_str(include_str!("fixtures/qtea_reference/golden.json")).unwrap()
}

fn near(actual: f32, expected: f32) {
    assert!(actual.is_finite() && expected.is_finite());
    assert!(
        (actual - expected).abs() <= 4e-6 * (1.0 + expected.abs()),
        "{actual:?} != {expected:?}"
    );
}

fn near_array(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        near(*actual, *expected);
    }
}

#[test]
fn pinned_author_group_and_column_fixtures() {
    let data = fixtures();
    assert_eq!(data.source, "e39bd8c0297e3427daeb404427c74cb6e202b683");
    assert_eq!(ARITHMETIC_ID, "qtea.cpu-arithmetic.f32.v1");
    assert_eq!(data.groups.len(), 12);
    let mut columns = 0;
    for fixture in data.groups {
        let group = GroupParameters::fit(
            &fixture.weights,
            fixture.rows,
            fixture.columns,
            fixture.coefficient,
            fixture.iterations,
            12 * fixture.rows + fixture.columns,
        )
        .unwrap();
        assert_eq!(group.columns(), fixture.columns);
        near_array(group.alpha(), &fixture.alpha);
        near_array(group.beta(), &fixture.beta);
        near_array(group.delta(), &fixture.delta);
        for column in fixture.refinements {
            let fitted = group
                .refine_column(&column.column, column.iterations, 9 * fixture.rows)
                .unwrap();
            assert_eq!(
                fitted.trits().iter().map(|x| x.get()).collect::<Vec<_>>(),
                column.trits
            );
            near(fitted.multiplier(), column.multiplier);
            near_array(fitted.values(), &column.values);
            columns += 1;
        }
    }
    assert_eq!(columns, 108);
}

#[test]
fn column_uses_reference_code_rule_not_paper_equation_and_handles_dead_zone_ties() {
    let group = GroupParameters::fit(&[-1., 0., 1., -10., 0., 10.], 2, 3, 0.96, 1, 27).unwrap();
    let fitted = group.refine_column(&[2., 10.], 1, 18).unwrap();
    assert_eq!(fitted.multiplier(), 12.0 / 11.0);
    assert_ne!(fitted.multiplier(), 1.5); // Paper's mean of normalized magnitudes.
    for sign in [1.0, -1.0] {
        let column = [sign * 0.5, sign * 5.0];
        let strict = group.refine_column(&column, 0, 18).unwrap();
        assert!(strict.trits().iter().all(|x| x.is_zero()));
        let outer = group.refine_column(&column, 1, 18).unwrap();
        assert!(outer.trits().iter().all(|x| x.to_f32() == sign));
        assert_eq!(outer.multiplier(), 1.0); // No initial nonzero assignment.
    }
    let zero = group.refine_column(&[0., 0.], 3, 18).unwrap();
    assert_eq!(zero.multiplier(), 1.0);
    assert!(zero.trits().iter().all(|x| x.is_zero()));
}

#[test]
fn all_finite_fp8_levels_midpoints_and_neighboring_f32s_match_author_torch_bytes() {
    let cases = fixtures().rounding;
    assert_eq!(cases.len(), 1010);
    for (input, code, decoded) in cases {
        let sparse = SparseResidualColumn::fit(&[448., 0., 0., 0., input, 0., 0., 0.], 4).unwrap();
        assert_eq!(sparse.indices(), [0, 0]);
        assert_eq!(sparse.fp8_codes(), [126, code], "input {input:?}");
        assert_eq!(sparse.scale_multiplier(), 1.0);
        let mut output = [123.; 8];
        sparse.decode_into(&mut output).unwrap();
        assert_eq!(output[4].to_bits(), decoded.to_bits(), "input {input:?}");
    }
}

#[test]
fn sparse_argmax_ties_scaling_and_signed_zero_match_author_fixtures() {
    for case in fixtures().sparse {
        let sparse = SparseResidualColumn::fit(&case.residual, case.residual.len() / 2).unwrap();
        assert_eq!(sparse.rows(), case.residual.len());
        assert_eq!(sparse.indices(), case.indices);
        assert_eq!(sparse.fp8_codes(), case.codes);
        near(sparse.scale_multiplier(), case.scale);
        let mut output = vec![123.; sparse.rows()];
        sparse.decode_into(&mut output).unwrap();
        near_array(&output, &case.dense);
        for (actual, expected) in output.iter().zip(&case.dense) {
            if *expected == 0.0 {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        let mut wrong = [123.; 3];
        assert_eq!(
            sparse.decode_into(&mut wrong),
            Err(ArithmeticError::InvalidShape)
        );
        assert_eq!(wrong, [123.; 3]);
    }
}

#[test]
fn exact_logical_array_budgets_are_required_without_fallback() {
    let weights = [-1., 0., 1., -10., 0., 10.];
    assert_eq!(
        GroupParameters::fit(&weights, 2, 3, 0.96, 1, 26),
        Err(ArithmeticError::ResourceLimit {
            required: 27,
            limit: 26
        })
    );
    let group = GroupParameters::fit(&weights, 2, 3, 0.96, 1, 27).unwrap();
    assert_eq!(
        group.refine_column(&[1., 10.], 1, 17),
        Err(ArithmeticError::ResourceLimit {
            required: 18,
            limit: 17
        })
    );
    assert_eq!(
        SparseResidualColumn::fit(&[0.; 8], 3),
        Err(ArithmeticError::ResourceLimit {
            required: 4,
            limit: 3
        })
    );
}

#[test]
fn invalid_shapes_config_nonfinite_and_degenerate_groups_refuse() {
    for (rows, columns) in [
        (0, 3),
        (1, 0),
        (1, 4097),
        (usize::MAX, 2),
        ((1 << 24) + 1, 1),
    ] {
        assert_eq!(
            GroupParameters::fit(&[], rows, columns, 0.96, 1, usize::MAX),
            Err(ArithmeticError::InvalidShape)
        );
    }
    for coefficient in [0., -1., f32::NAN, f32::INFINITY] {
        assert_eq!(
            GroupParameters::fit(&[-1., 1.], 1, 2, coefficient, 1, 14),
            Err(ArithmeticError::InvalidConfig)
        );
    }
    assert_eq!(
        GroupParameters::fit(&[-1., 1.], 1, 2, 0.96, 65, 14),
        Err(ArithmeticError::InvalidConfig)
    );
    for rounds in [0, 1, 64] {
        assert_eq!(
            GroupParameters::fit(&[3., 3.], 1, 2, 0.96, rounds, 14),
            Err(ArithmeticError::DegenerateRow { row: 0 })
        );
    }
    assert_eq!(
        GroupParameters::fit(&[-1., 1., 0., 0.], 2, 2, 0.96, 1, 26),
        Err(ArithmeticError::DegenerateRow { row: 1 })
    );
    assert_eq!(
        GroupParameters::fit(&[-1., f32::NAN], 1, 2, 0.96, 1, 14),
        Err(ArithmeticError::NonFiniteInput { index: 1 })
    );
    assert_eq!(
        GroupParameters::fit(&[f32::MAX, f32::MAX], 1, 2, 0.96, 1, 14),
        Err(ArithmeticError::NonFiniteArithmetic)
    );
    let group = GroupParameters::fit(&[-1., 1.], 1, 2, 0.96, 1, 14).unwrap();
    assert_eq!(
        group.refine_column(&[], 1, 9),
        Err(ArithmeticError::InvalidShape)
    );
    assert_eq!(
        group.refine_column(&[0.], 65, 9),
        Err(ArithmeticError::InvalidConfig)
    );
    assert_eq!(
        group.refine_column(&[f32::INFINITY], 1, 9),
        Err(ArithmeticError::NonFiniteInput { index: 0 })
    );
    assert_eq!(
        SparseResidualColumn::fit(&[], 0),
        Err(ArithmeticError::InvalidShape)
    );
    assert_eq!(
        SparseResidualColumn::fit(&[0.; 3], 10),
        Err(ArithmeticError::InvalidShape)
    );
    assert_eq!(
        SparseResidualColumn::fit(&[0., f32::NAN, 0., 0.], 2),
        Err(ArithmeticError::NonFiniteInput { index: 1 })
    );
}

#[test]
fn zero_round_column_does_not_reduce_unused_group_alpha_and_sparse_never_decodes_infinity() {
    let magnitude = f32::MAX / 4.0;
    let weights = [-magnitude, 0.0, magnitude].repeat(8);
    let group = GroupParameters::fit(&weights, 8, 3, 0.96, 0, 99).unwrap();
    assert!(group.refine_column(&[0.; 8], 0, 72).is_ok());
    assert_eq!(
        group.refine_column(&[0.; 8], 1, 72),
        Err(ArithmeticError::NonFiniteArithmetic)
    );
    for value in [f32::MAX, f32::MIN_POSITIVE, f32::from_bits(1), -f32::MAX] {
        match SparseResidualColumn::fit(&[value, 0., 0., 0.], 2) {
            Ok(column) => {
                let mut output = [0.; 4];
                column.decode_into(&mut output).unwrap();
                assert!(output.iter().all(|value| value.is_finite()));
            }
            Err(error) => assert_eq!(error, ArithmeticError::NonFiniteArithmetic),
        }
    }
}

#[test]
fn salient_ranking_matches_author_and_freezes_ties_to_original_index() {
    let cases = fixtures().ranking;
    assert_eq!(cases.len(), 7);
    for case in cases {
        let ranking = SalientRanking::rank(
            &case.weights,
            case.rows,
            case.columns,
            &case.diagonal,
            case.group_size,
            case.ratio,
            case.columns * (size_of::<usize>() + 4),
        )
        .unwrap();
        assert_eq!(ranking.columns(), case.order);
        assert_eq!(ranking.salient_count(), case.salient_count);
    }
    let bound = 8 * (size_of::<usize>() + 4);
    for (ratio, expected) in [(0.0, 0), (0.125, 2), (0.625, 4), (0.875, 8)] {
        let ranking = SalientRanking::rank(&[1.; 8], 1, 8, &[1.; 8], 2, ratio, bound).unwrap();
        assert_eq!(ranking.columns(), [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(ranking.salient_count(), expected);
    }
    assert_eq!(
        SalientRanking::rank(&[1.; 8], 1, 8, &[1.; 8], 2, 0.5, bound - 1),
        Err(ArithmeticError::ResourceLimit {
            required: bound,
            limit: bound - 1
        })
    );
    assert_eq!(
        SalientRanking::rank(&[1.; 8], 1, 8, &[1.; 8], 3, 0.5, bound),
        Err(ArithmeticError::InvalidShape)
    );
    assert_eq!(
        SalientRanking::rank(&[1.; 8], 1, 8, &[1.; 8], 2, f64::NAN, bound),
        Err(ArithmeticError::InvalidConfig)
    );
    assert_eq!(
        SalientRanking::rank(&[1.; 8], 1, 8, &[-1.; 8], 2, 0.5, bound),
        Err(ArithmeticError::InvalidDiagonal { column: 0 })
    );
    assert_eq!(
        SalientRanking::rank(&[f32::MAX; 8], 1, 8, &[1.; 8], 2, 0.5, bound),
        Err(ArithmeticError::NonFiniteArithmetic)
    );
}
