use tritium_quantize::engine::{SalienceFoldError as Error, SaliencePair};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn scales_both_halves_in_row_major_order() {
    let mut q = [2.0, 3.0, -4.0, -5.0];
    let mut k = [6.0, 7.0];
    let mut norm = [8.0, 10.0];
    let scales = SaliencePair {
        projections: &mut [&mut q, &mut k],
        inverse: &mut norm,
        inverse_columns: 1,
    }
    .apply(&[1.0, 16.0], 1, 1.0)
    .unwrap();
    assert_eq!(scales, [0.5, 2.0]);
    assert_eq!(q, [1.0, 6.0, -2.0, -10.0]);
    assert_eq!(k, [3.0, 14.0]);
    assert_eq!(norm, [16.0, 5.0]);

    let mut down = [2.0, 3.0, -4.0, -5.0];
    let mut up = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
    SaliencePair {
        projections: &mut [&mut down],
        inverse: &mut up,
        inverse_columns: 3,
    }
    .apply(&[1.0, 16.0], 1, 1.0)
    .unwrap();
    assert_eq!(down, [1.0, 6.0, -2.0, -10.0]);
    assert_eq!(up, [12.0, 14.0, 16.0, 4.5, 5.0, 5.5]);
}

#[test]
fn alpha_zero_preserves_all_finite_bits_and_signed_zero() {
    for moments in [[0.0, 0.0], [1e-30, 1e30], [1.0, 16.0]] {
        let mut projection = [0.0, -0.0, f32::MIN_POSITIVE, -3.25];
        let mut inverse = [-0.0, f32::MAX];
        let before = (bits(&projection), bits(&inverse));
        let scales = SaliencePair {
            projections: &mut [&mut projection],
            inverse: &mut inverse,
            inverse_columns: 1,
        }
        .apply(&moments, 32, 0.0)
        .unwrap();
        assert_eq!(scales, [1.0, 1.0]);
        assert_eq!((bits(&projection), bits(&inverse)), before);
    }
}

#[test]
fn bad_statistics_and_alpha_reject_before_either_half_changes() {
    let cases = [
        (vec![], 1, 0.75, Error::EmptyStatistics),
        (vec![1.0, 4.0], 0, 0.75, Error::EmptyStatistics),
        (vec![1.0, 4.0], 1, f64::NAN, Error::Alpha),
        (vec![1.0, 4.0], 1, f64::INFINITY, Error::Alpha),
        (vec![1.0, 4.0], 1, -0.1, Error::Alpha),
        (vec![1.0, 4.0], 1, 1.1, Error::Alpha),
        (vec![1.0, -1.0], 1, 0.75, Error::Moments),
        (vec![1.0, f64::NAN], 1, 0.75, Error::Moments),
        (vec![1.0, f64::INFINITY], 1, 0.75, Error::Moments),
        (vec![0.0, f64::MAX], 1, 1.0, Error::Scale),
    ];
    for (moments, samples, alpha, expected) in cases {
        let mut p = [1.0, -2.0];
        let mut inverse = [3.0, -4.0];
        let before = (bits(&p), bits(&inverse));
        assert_eq!(
            SaliencePair {
                projections: &mut [&mut p],
                inverse: &mut inverse,
                inverse_columns: 1,
            }
            .apply(&moments, samples, alpha),
            Err(expected)
        );
        assert_eq!((bits(&p), bits(&inverse)), before);
    }
}

#[test]
fn geometry_rejects_without_mutation() {
    for (projection_len, inverse_len, columns) in [
        (0, 2, 1),
        (3, 2, 1),
        (4, 3, 1),
        (4, 2, 0),
        (4, 2, usize::MAX),
    ] {
        let mut p = vec![1.0; projection_len];
        let mut inverse = vec![2.0; inverse_len];
        let before = (bits(&p), bits(&inverse));
        assert_eq!(
            SaliencePair {
                projections: &mut [&mut p],
                inverse: &mut inverse,
                inverse_columns: columns,
            }
            .apply(&[1.0, 4.0], 1, 0.75),
            Err(Error::Geometry)
        );
        assert_eq!((bits(&p), bits(&inverse)), before);
    }
    assert_eq!(
        SaliencePair {
            projections: &mut [],
            inverse: &mut [1.0, 1.0],
            inverse_columns: 1,
        }
        .apply(&[1.0, 4.0], 1, 0.75),
        Err(Error::Geometry)
    );
}

#[test]
fn last_projection_or_inverse_failure_cannot_partially_mutate_the_pair() {
    for (late, partner, expected) in [
        ([1.0, f32::NAN], [3.0, 4.0], Error::NonFiniteSource),
        ([1.0, 2.0], [3.0, f32::INFINITY], Error::NonFiniteSource),
        ([1.0, f32::MAX], [3.0, 4.0], Error::NonFiniteOutput),
        ([1.0, 2.0], [f32::MAX, 4.0], Error::NonFiniteOutput),
    ] {
        let mut first = [1.0, 2.0];
        let mut last = late;
        let mut inverse = partner;
        let before = (bits(&first), bits(&last), bits(&inverse));
        assert_eq!(
            SaliencePair {
                projections: &mut [&mut first, &mut last],
                inverse: &mut inverse,
                inverse_columns: 1,
            }
            .apply(&[1.0, 16.0], 1, 1.0),
            Err(expected)
        );
        assert_eq!((bits(&first), bits(&last), bits(&inverse)), before);
    }
}
