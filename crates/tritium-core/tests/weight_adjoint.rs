//! Continuous logical-weight Jacobians, not discrete-code or model qualification.
use tritium_core::{
    AdditiveError, AdditiveView, AdditiveWeightVjpPlan, OwnedPerTileIndex, PerTilePlan, Trit,
    UniformPlan, accumulate_embed_weight_vjp, accumulate_ternary_matmul_weight_vjp, apply_basis,
    apply_inverse_basis,
};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};

fn values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 13 + i / 7) % 19) as f32 / 16.0 - 0.5)
        .collect()
}
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 3e-6 * (1.0 + b.abs()), "{a} != {b}");
}

#[test]
fn present_coordinate_order_and_original_basis_jacobians_all_profiles() {
    let mut profiles = 0;
    for admitted in ADMITTED_LAWS {
        for planes in 1..=admitted.max_planes {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                for (cols, basis) in [
                    (257, Basis::Identity),
                    (260, Basis::Hadamard { block: 4 }),
                    (
                        260,
                        Basis::SignedRht {
                            block: 4,
                            seed: 19,
                            domain: 7,
                        },
                    ),
                    (
                        512,
                        Basis::SignedRht {
                            block: 128,
                            seed: 19,
                            domain: 7,
                        },
                    ),
                ] {
                    if codec == PlaneCodec::S34 && cols == 257 {
                        continue;
                    }
                    for mode in 0..3 {
                        let layout = AdditiveLayout {
                            rows: 3,
                            cols: cols as u64,
                            tile: 256,
                            group: 32,
                            max_planes: admitted.max_planes,
                            allocation: if mode == 0 {
                                PlaneAllocation::Uniform
                            } else {
                                PlaneAllocation::PerTile
                            },
                            codec,
                            law: admitted.law,
                            basis,
                            transport: Transport::Raw,
                        };
                        let counts: Vec<u8> = (0..6)
                            .map(|i| {
                                if mode == 2 || i % 3 == 0 {
                                    0
                                } else if i % 3 == 1 {
                                    planes
                                } else {
                                    1
                                }
                            })
                            .collect();
                        let index = (mode != 0)
                            .then(|| OwnedPerTileIndex::new(layout, counts.clone()).unwrap());
                        let (nt, ns) = if mode == 0 {
                            let p = UniformPlan::new(layout, planes).unwrap();
                            (p.trit_count(), p.scale_count())
                        } else {
                            let p = PerTilePlan::new(layout, &counts).unwrap();
                            (p.trit_count(), p.scale_count())
                        };
                        // All-zero coefficients still have nonzero present-weight
                        // adjoints: this is NOT the scale VJP or a trit gradient.
                        let trits = vec![Trit::ZERO; nt];
                        let scales = vec![0.0; ns];
                        let view = if let Some(i) = &index {
                            AdditiveView::new_per_tile(i.as_index(), &trits, &scales).unwrap()
                        } else {
                            AdditiveView::new(layout, planes, &trits, &scales).unwrap()
                        };
                        let plan = if let Some(i) = &index {
                            AdditiveWeightVjpPlan::from_per_tile(i.as_index())
                        } else {
                            AdditiveWeightVjpPlan::from_uniform(
                                UniformPlan::new(layout, planes).unwrap(),
                            )
                        };
                        // Independent compact mapping, from input counts rather
                        // than the sealed row/tile cursors used by the VJP.
                        let coordinates: Vec<_> = (0..3)
                            .flat_map(|r| {
                                (0..cols).filter_map({
                                    let counts = &counts;
                                    move |c| {
                                        (mode == 0 || counts[r * 2 + c / 256] > 0).then_some((r, c))
                                    }
                                })
                            })
                            .collect();
                        assert_eq!(plan.present_weight_count(), coordinates.len());
                        let x = values(2 * cols);
                        let dy = values(6);
                        let ids = [2, 0, 2, 1];
                        let de = values(ids.len() * cols);
                        let mut row = vec![17.0; cols];
                        let mut gm = vec![0.25; coordinates.len()];
                        let mut ge = gm.clone();
                        accumulate_ternary_matmul_weight_vjp(
                            &x,
                            &view.bind_matmul(),
                            &dy,
                            2,
                            &mut row,
                            &mut gm,
                        )
                        .unwrap();
                        accumulate_embed_weight_vjp(
                            &view.bind_gather(),
                            &ids,
                            &de,
                            &mut row,
                            &mut ge,
                        )
                        .unwrap();
                        if mode == 2 {
                            assert!(row.iter().all(|&v| v == 17.0));
                        }
                        // Unit stored-coordinate perturbations become B^T e_c
                        // in the original row. Dense forward Jacobians below do
                        // not rotate x or dy, unlike the implementation.
                        for c in [0, 1, 127, 255, 256, cols - 1] {
                            let mut direction = vec![0.0; cols];
                            direction[c] = 1.0;
                            apply_inverse_basis(&mut direction, basis).unwrap();
                            for (i, &(r, col)) in coordinates
                                .iter()
                                .enumerate()
                                .filter(|(_, (_, col))| *col == c)
                            {
                                assert_eq!(col, c);
                                let expected: f64 = (0..2)
                                    .map(|b| {
                                        dot(&direction, &x[b * cols..(b + 1) * cols])
                                            * f64::from(dy[b * 3 + r])
                                    })
                                    .sum();
                                close(gm[i] - 0.25, expected);
                                let expected: f64 = ids
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, id)| **id == r)
                                    .map(|(b, _)| dot(&direction, &de[b * cols..(b + 1) * cols]))
                                    .sum();
                                close(ge[i] - 0.25, expected);
                            }
                        }
                        let previous = gm.clone();
                        plan.matmul(&x, &dy, 2, &mut row, &mut gm).unwrap();
                        for (twice, once) in gm.iter().zip(previous) {
                            close(*twice - 0.25, 2.0 * (once - 0.25));
                        }
                        profiles += 1;
                    }
                }
            }
        }
    }
    assert_eq!(profiles, 330);
}

fn plan() -> AdditiveWeightVjpPlan<'static> {
    AdditiveWeightVjpPlan::from_uniform(
        UniformPlan::new(
            AdditiveLayout {
                rows: 3,
                cols: 4,
                tile: 256,
                group: 32,
                max_planes: 3,
                allocation: PlaneAllocation::Uniform,
                codec: PlaneCodec::D2,
                law: ADMITTED_LAWS[0].law,
                basis: Basis::SignedRht {
                    block: 4,
                    seed: 19,
                    domain: 7,
                },
                transport: Transport::Raw,
            },
            3,
        )
        .unwrap(),
    )
}

#[test]
fn preflight_and_empty_uses_preserve_both_buffers() {
    let p = plan();
    for case in 0..7 {
        let mut row = [17.0; 4];
        let mut out = [19.0; 12];
        let result = match case {
            0 => p.matmul(&[0.0; 3], &[0.0; 3], 1, &mut row, &mut out),
            1 => p.matmul(&[0.0; 4], &[0.0; 2], 1, &mut row, &mut out),
            2 => p.matmul(&[], &[], usize::MAX, &mut row, &mut out),
            3 => p.embed(&[0, 3], &[0.0; 8], &mut row, &mut out),
            4 => p.embed(&[0], &[0.0; 3], &mut row, &mut out),
            5 => p.matmul(&[], &[], 0, &mut row[..3], &mut out),
            _ => p.embed(&[], &[], &mut row, &mut out[..11]),
        };
        assert!(result.is_err());
        assert_eq!(row, [17.0; 4]);
        assert_eq!(out, [19.0; 12]);
    }
    let mut row = [17.0; 4];
    let mut out = [19.0; 12];
    assert_eq!(p.matmul(&[], &[], 0, &mut row, &mut out), Ok(()));
    assert_eq!(p.embed(&[], &[], &mut row, &mut out), Ok(()));
    assert_eq!(row, [17.0; 4]);
    assert_eq!(out, [19.0; 12]);
    assert_eq!(
        p.embed(&[usize::MAX], &[0.0; 4], &mut row, &mut out),
        Err(AdditiveError::RowOutOfRange)
    );
}

#[test]
fn signed_forward_basis_is_not_inverse_and_finite_policy_is_callers() {
    let p = plan();
    let x = values(4);
    let dy = [0.25, -0.75, 0.5];
    let basis = Basis::SignedRht {
        block: 4,
        seed: 19,
        domain: 7,
    };
    let mut inverse = x.clone();
    apply_inverse_basis(&mut inverse, basis).unwrap();
    let mut forward = x.clone();
    apply_basis(&mut forward, basis).unwrap();
    let mut row = [0.0; 4];
    let mut out = [0.0; 12];
    p.matmul(&x, &dy, 1, &mut row, &mut out).unwrap();
    assert!(
        out[..4]
            .iter()
            .zip(inverse)
            .any(|(a, b)| (*a - f64::from(b) * 0.25).abs() > 0.01)
    );
    for (a, b) in out[..4].iter().zip(forward) {
        close(*a, f64::from(b) * 0.25);
    }
    p.embed(&[0], &[f32::NAN; 4], &mut row, &mut out).unwrap();
    assert!(out[..4].iter().all(|v| v.is_nan()));
}

#[test]
fn continuous_dense_forward_finite_differences_match_combined_tied_consumers() {
    // Independently perturb continuous logical stored weights, NOT their hard
    // trits (which have no ordinary infinitesimal derivative).
    let p = plan();
    let basis = Basis::SignedRht {
        block: 4,
        seed: 19,
        domain: 7,
    };
    let x = values(8);
    let dy = values(6);
    let de = values(12);
    let ids = [2, 0, 2];
    let mut gradients = [0.0; 12];
    p.matmul(&x, &dy, 2, &mut [0.0; 4], &mut gradients).unwrap();
    p.embed(&ids, &de, &mut [0.0; 4], &mut gradients).unwrap();
    let loss = |stored: &[f32]| {
        let mut original = stored.to_vec();
        for row in original.as_chunks_mut::<4>().0 {
            apply_inverse_basis(row, basis).unwrap();
        }
        let mut loss = 0.0;
        for b in 0..2 {
            for r in 0..3 {
                loss += dot(&original[r * 4..r * 4 + 4], &x[b * 4..b * 4 + 4])
                    * f64::from(dy[b * 3 + r]);
            }
        }
        for (b, &r) in ids.iter().enumerate() {
            loss += dot(&original[r * 4..r * 4 + 4], &de[b * 4..b * 4 + 4]);
        }
        loss
    };
    let initial = values(12);
    for i in 0..12 {
        let mut plus = initial.clone();
        let mut minus = initial.clone();
        plus[i] += 0.01;
        minus[i] -= 0.01;
        let numerical = (loss(&plus) - loss(&minus)) / (f64::from(plus[i]) - f64::from(minus[i]));
        assert!(
            (gradients[i] - numerical).abs() < 2e-5,
            "coordinate={i} actual={} finite_difference={numerical}",
            gradients[i]
        );
    }
}
