//! Numerical native fixtures; not packed speed, model quality or release gates.
use tritium_core::{
    AdditiveView, CodeProjectionError, OwnedPerTileIndex, Trit, UniformPlan,
    project_additive_code_unit,
};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};

struct Fixture {
    layout: AdditiveLayout,
    planes: u8,
    trits: Vec<Trit>,
    scales: Vec<f32>,
    index: Option<OwnedPerTileIndex>,
}
impl Fixture {
    fn new(law: usize, codec: PlaneCodec, basis: Basis, mode: u8, cols: u64) -> Self {
        let planes = ADMITTED_LAWS[law].max_planes;
        let layout = AdditiveLayout {
            rows: 2,
            cols,
            tile: 256,
            group: 32,
            max_planes: planes,
            allocation: if mode == 0 {
                PlaneAllocation::Uniform
            } else {
                PlaneAllocation::PerTile
            },
            codec,
            law: ADMITTED_LAWS[law].law,
            basis,
            transport: Transport::Raw,
        };
        let index = (mode != 0).then(|| {
            OwnedPerTileIndex::new(
                layout,
                (0..2 * (cols as usize).div_ceil(256))
                    .map(|i| {
                        if mode == 2 {
                            0
                        } else {
                            [planes, 0, 1, planes][i % 4]
                        }
                    })
                    .collect(),
            )
            .unwrap()
        });
        let (trits, scales) = if let Some(index) = &index {
            (
                index.as_index().trit_count(),
                index.as_index().scale_count(),
            )
        } else {
            let plan = UniformPlan::new(layout, planes).unwrap();
            (plan.trit_count(), plan.scale_count())
        };
        Self {
            layout,
            planes,
            index,
            trits: (0..trits)
                .map(|i| {
                    if codec == PlaneCodec::S34 {
                        [Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS][i % 4]
                    } else {
                        [Trit::NEG, Trit::ZERO, Trit::POS][i % 3]
                    }
                })
                .collect(),
            scales: (0..scales)
                .map(|i| {
                    if law == 1 || law == 3 {
                        [0.0, f32::from_bits(0x3f80_0001), 0.333_333_34, 0.5][i % 4]
                    } else {
                        [0.0, 0.25, 0.5, 1.0][i % 4]
                    }
                })
                .collect(),
        }
    }
    fn view<'a>(&'a self, trits: &'a [Trit]) -> AdditiveView<'a> {
        if let Some(index) = &self.index {
            AdditiveView::new_per_tile(index.as_index(), trits, &self.scales).unwrap()
        } else {
            AdditiveView::new(self.layout, self.planes, trits, &self.scales).unwrap()
        }
    }
    fn row(&self, trits: &[Trit], row: usize) -> Vec<f32> {
        let mut out = vec![0.0; self.layout.cols as usize];
        self.view(trits).dequant_row_into(row, &mut out).unwrap();
        out
    }
}
fn square_error(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum()
}

#[test]
fn native_profile_matrix_preserves_structure_and_never_increases_surrogate() {
    let mut checked = 0;
    for law in 0..ADMITTED_LAWS.len() {
        for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
            for basis in [
                Basis::Identity,
                Basis::Hadamard { block: 16 },
                Basis::SignedRht {
                    block: 16,
                    seed: u64::MAX,
                    domain: 91,
                },
            ] {
                for mode in 0..=2 {
                    let f = Fixture::new(
                        law,
                        codec,
                        basis,
                        mode,
                        if basis != Basis::Identity {
                            272
                        } else if codec == PlaneCodec::S34 {
                            260
                        } else {
                            257
                        },
                    );
                    let width = if codec == PlaneCodec::S34 { 4 } else { 1 };
                    let cols = f.layout.cols as usize;
                    let mut output = f.trits.clone();
                    for row in 0..2 {
                        let original = f.row(&f.trits, row);
                        for col in (0..cols).step_by(width) {
                            let target: Vec<f32> = original[col..col + width]
                                .iter()
                                .map(|value| -value + 0.375)
                                .collect();
                            let stats = project_additive_code_unit(
                                f.view(&f.trits),
                                (row * cols + col) / width,
                                &target,
                                None,
                                &mut output,
                            )
                            .unwrap();
                            assert!(!stats.trust_limited);
                            assert_eq!(
                                stats.surrogate_before,
                                square_error(&original[col..col + width], &target)
                            );
                            let actual = f.row(&output, row);
                            assert_eq!(
                                stats.surrogate_after,
                                square_error(&actual[col..col + width], &target)
                            );
                            assert_eq!(
                                stats.squared_change,
                                square_error(
                                    &actual[col..col + width],
                                    &original[col..col + width]
                                )
                            );
                            assert!(stats.surrogate_after <= stats.surrogate_before);
                            checked += 1;
                        }
                    }
                    if codec == PlaneCodec::S34 {
                        assert!(
                            output.as_chunks::<4>().0.iter().all(|block| block
                                .iter()
                                .filter(|code| code.is_zero())
                                .count()
                                == 1)
                        );
                    }
                    assert_eq!(f.view(&output).scales(), f.scales);
                    assert_eq!(f.view(&output).layout(), f.layout);
                }
            }
        }
    }
    eprintln!("NATIVE_CODE_PROJECTION checked_units={checked} profile_combinations=108");
}

#[test]
fn dense_search_matches_independent_exhaustive_decode_and_tie_order() {
    for law in 0..ADMITTED_LAWS.len() {
        let mut f = Fixture::new(law, PlaneCodec::D2, Basis::Identity, 0, 1);
        f.scales.fill(0.5);
        for target in [-2.0, -0.75, -0.25, 0.0, 0.25, 0.75, 2.0] {
            let view = f.view(&f.trits);
            let mut expected = f.trits.clone();
            let mut best = square_error(&f.row(&expected, 0)[..1], &[target]);
            for tuple in 0..3usize.pow(u32::from(f.planes)) {
                let mut candidate = f.trits.clone();
                let mut cursor = tuple;
                for plane in 0..usize::from(f.planes) {
                    candidate[plane * 2] = [Trit::NEG, Trit::ZERO, Trit::POS][cursor % 3];
                    cursor /= 3;
                }
                let loss = square_error(&f.row(&candidate, 0)[..1], &[target]);
                if loss < best {
                    best = loss;
                    expected = candidate;
                }
            }
            let mut actual = f.trits.clone();
            let stats = project_additive_code_unit(view, 0, &[target], None, &mut actual).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(stats.surrogate_after, best);
        }
    }
}

#[test]
fn one_plane_sparse_search_matches_all_thirty_two_patterns() {
    let mut f = Fixture::new(3, PlaneCodec::S34, Basis::Identity, 0, 4);
    f.scales.fill(0.5);
    for target in [[0.5, 0.0, -0.5, 0.5], [-0.2, 0.9, 0.1, -0.8], [0.0; 4]] {
        let mut expected = f.trits.clone();
        let mut best = square_error(&f.row(&expected, 0), &target);
        for zero in 0..4 {
            for signs in 0..8 {
                let mut candidate = f.trits.clone();
                let mut bit = 0;
                for (i, code) in candidate[..4].iter_mut().enumerate() {
                    *code = if i == zero {
                        Trit::ZERO
                    } else {
                        let code = if signs & (1 << bit) == 0 {
                            Trit::NEG
                        } else {
                            Trit::POS
                        };
                        bit += 1;
                        code
                    };
                }
                let loss = square_error(&f.row(&candidate, 0), &target);
                if loss < best {
                    best = loss;
                    expected = candidate;
                }
            }
        }
        let mut actual = f.trits.clone();
        let stats =
            project_additive_code_unit(f.view(&f.trits), 0, &target, None, &mut actual).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(stats.surrogate_after, best);
    }
}

#[test]
fn trust_boundary_and_ties_leave_output_untouched() {
    let mut f = Fixture::new(3, PlaneCodec::D2, Basis::Identity, 0, 1);
    f.trits.fill(Trit::POS);
    f.scales.fill(1.0);
    for (target, budget, changed, blocked) in [
        (0.5, None, false, false),
        (0.0, Some(0.999), false, true),
        (0.0, Some(1.0), true, false),
    ] {
        let mut output = vec![Trit::NEG; f.trits.len()];
        let initial = output.clone();
        let stats = project_additive_code_unit(f.view(&f.trits), 0, &[target], budget, &mut output)
            .unwrap();
        assert_eq!(stats.changed, changed);
        assert_eq!(stats.trust_limited, blocked);
        assert_eq!(stats.squared_change, if changed { 1.0 } else { 0.0 });
        if changed {
            assert_eq!(output, [Trit::ZERO, Trit::NEG]);
        } else {
            assert_eq!(output, initial);
        }
    }
}

#[test]
fn invalid_requests_are_transactional_even_with_bad_structure_in_later_plane() {
    let f = Fixture::new(1, PlaneCodec::D2, Basis::Identity, 0, 4);
    let mut output = vec![Trit::POS; f.trits.len()];
    for (unit, target, budget, expected) in [
        (8, vec![0.0], None, CodeProjectionError::Shape),
        (0, vec![], None, CodeProjectionError::Shape),
        (0, vec![f32::NAN], None, CodeProjectionError::NonFinite),
        (0, vec![f32::INFINITY], None, CodeProjectionError::NonFinite),
        (
            0,
            vec![0.0],
            Some(f64::INFINITY),
            CodeProjectionError::NonFinite,
        ),
        (0, vec![0.0], Some(-0.0), CodeProjectionError::InvalidBudget),
        (0, vec![0.0], Some(-1.0), CodeProjectionError::InvalidBudget),
    ] {
        let before = output.clone();
        assert_eq!(
            project_additive_code_unit(f.view(&f.trits), unit, &target, budget, &mut output),
            Err(expected)
        );
        assert_eq!(output, before);
    }
    let mut short = output[..2].to_vec();
    let before = short.clone();
    assert_eq!(
        project_additive_code_unit(f.view(&f.trits), 0, &[0.0], None, &mut short),
        Err(CodeProjectionError::Shape)
    );
    assert_eq!(short, before);
    let mut sparse = Fixture::new(0, PlaneCodec::S34, Basis::Identity, 0, 4);
    sparse.trits[16] = Trit::POS;
    let mut output = sparse.trits.clone();
    let before = output.clone();
    assert_eq!(
        project_additive_code_unit(sparse.view(&sparse.trits), 0, &[0.0; 4], None, &mut output),
        Err(CodeProjectionError::InvalidStructure)
    );
    assert_eq!(output, before);
}

#[test]
fn overflowing_trials_are_skipped_but_overflowing_original_is_rejected() {
    let mut f = Fixture::new(1, PlaneCodec::D2, Basis::Identity, 0, 1);
    f.scales.fill(f32::MAX);
    f.trits.fill(Trit::ZERO);
    let mut output = f.trits.clone();
    let stats =
        project_additive_code_unit(f.view(&f.trits), 0, &[f32::MAX], None, &mut output).unwrap();
    assert!(stats.changed);
    assert_eq!(stats.surrogate_after, 0.0);
    f.trits.fill(Trit::POS);
    let before = output.clone();
    assert_eq!(
        project_additive_code_unit(f.view(&f.trits), 0, &[0.0], None, &mut output),
        Err(CodeProjectionError::NonFinite)
    );
    assert_eq!(output, before);
}

#[test]
fn projection_only_touches_selected_present_offsets_and_does_not_rotate_targets() {
    let mut identity = Fixture::new(1, PlaneCodec::B3, Basis::Identity, 1, 272);
    identity.scales.fill(1.0);
    identity.trits.fill(Trit::NEG);
    let mut rotated = Fixture::new(
        1,
        PlaneCodec::B3,
        Basis::SignedRht {
            block: 16,
            seed: 123,
            domain: 456,
        },
        1,
        272,
    );
    rotated.scales = identity.scales.clone();
    rotated.trits = identity.trits.clone();
    let mut a = vec![Trit::ZERO; identity.trits.len()];
    let mut b = a.clone();
    let x = project_additive_code_unit(identity.view(&identity.trits), 31, &[3.0], None, &mut a)
        .unwrap();
    let y =
        project_additive_code_unit(rotated.view(&rotated.trits), 31, &[3.0], None, &mut b).unwrap();
    assert!(x.changed);
    assert_eq!(x, y);
    assert_eq!(a, b);
    assert_eq!(a.iter().filter(|&&code| code == Trit::POS).count(), 3);
    assert_eq!(a.iter().filter(|&&code| code != Trit::ZERO).count(), 3);
    let before = a.clone();
    let pruned =
        project_additive_code_unit(identity.view(&identity.trits), 256, &[3.0], None, &mut a)
            .unwrap();
    assert!(!pruned.changed);
    assert_eq!(a, before);
}
