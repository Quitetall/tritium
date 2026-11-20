//! CPU semantic fixtures, not packed/hardware/model-quality qualification.
use tritium_core::{
    AdditiveError, AdditiveScaleVjpPlan, AdditiveView, OwnedPerTileIndex, PerTilePlan, Trit,
    UniformPlan, accumulate_embed_scale_vjp, accumulate_ternary_matmul_scale_vjp,
    apply_inverse_basis, reference_embed, reference_ternary_matmul,
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
    fn new(law: usize, basis: Basis, group: u16, cols: u64, planes: u8, mode: u8) -> Self {
        let layout = AdditiveLayout {
            rows: 3,
            cols,
            tile: 256,
            group,
            max_planes: ADMITTED_LAWS[law].max_planes,
            allocation: if mode == 0 {
                PlaneAllocation::Uniform
            } else {
                PlaneAllocation::PerTile
            },
            codec: PlaneCodec::D2,
            law: ADMITTED_LAWS[law].law,
            basis,
            transport: Transport::Raw,
        };
        let (trits, scales, index) = if mode == 0 {
            let p = UniformPlan::new(layout, planes).unwrap();
            (p.trit_count(), p.scale_count(), None)
        } else {
            let tiles = (cols as usize).div_ceil(256);
            let counts = (0..3 * tiles)
                .map(|i| {
                    if mode == 2 || i % 3 == 0 {
                        0
                    } else if i % 3 == 1 {
                        planes
                    } else {
                        1
                    }
                })
                .collect::<Vec<_>>();
            let p = PerTilePlan::new(layout, &counts).unwrap();
            let sizes = (p.trit_count(), p.scale_count());
            (
                sizes.0,
                sizes.1,
                Some(OwnedPerTileIndex::new(layout, counts).unwrap()),
            )
        };
        Self {
            layout,
            planes,
            trits: (0..trits)
                .map(|i| [Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS, Trit::NEG][i % 5])
                .collect(),
            scales: vec![0.5; scales],
            index,
        }
    }
    fn view<'a>(&'a self, scales: &'a [f32]) -> AdditiveView<'a> {
        if let Some(index) = &self.index {
            AdditiveView::new_per_tile(index.as_index(), &self.trits, scales).unwrap()
        } else {
            AdditiveView::new(self.layout, self.planes, &self.trits, scales).unwrap()
        }
    }
}
fn close(a: f64, b: f64, t: f64) {
    assert!(
        (a - b).abs() <= t * (1.0 + b.abs()),
        "actual={a} expected={b} tolerance={t}"
    );
}
fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 7) as f32 * 0.125 - 0.375).collect()
}
fn matmul(f: &Fixture, scales: &[f32], x: &[f32], batch: usize) -> Vec<f32> {
    let mut out = vec![0.0; batch * f.layout.rows as usize];
    reference_ternary_matmul(
        x,
        &f.view(scales).bind_matmul(),
        batch,
        &mut vec![0.0; x.len()],
        &mut out,
    )
    .unwrap();
    out
}
fn embed(f: &Fixture, scales: &[f32], ids: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0; ids.len() * f.layout.cols as usize];
    reference_embed(&f.view(scales).bind_gather(), ids, &mut out).unwrap();
    out
}
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum()
}
fn check(f: &Fixture, batch: usize) {
    let view = f.view(&f.scales);
    let cols = f.layout.cols as usize;
    let x = values(batch * cols);
    let dy = values(batch * 3);
    let ids = if batch == 0 { vec![] } else { vec![2, 0, 2, 1] };
    let de = values(ids.len() * cols);
    let mut workspace = vec![91.0; cols];
    let mut gm = vec![0.25; f.scales.len()];
    let mut ge = gm.clone();
    accumulate_ternary_matmul_scale_vjp(
        &x,
        &view.bind_matmul(),
        &dy,
        batch,
        &mut workspace,
        &mut gm,
    )
    .unwrap();
    if batch == 0 {
        assert!(workspace.iter().all(|&v| v == 91.0));
        assert!(gm.iter().all(|&v| v == 0.25));
    }
    accumulate_embed_scale_vjp(&view.bind_gather(), &ids, &de, &mut workspace, &mut ge).unwrap();
    if std::env::var_os("TRITIUM_SCALE_ADJOINT_CAPTURE").is_some() {
        println!(
            "SCALE_BITS batch={batch} layout={:?} planes={} counts={:?} matmul={:?} embed={:?} row={:?}",
            f.layout,
            view.plane_count(),
            view.tile_counts(),
            gm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            ge.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            workspace.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    if batch == 0 {
        assert!(workspace.iter().all(|&v| v == 91.0));
        assert!(ge.iter().all(|&v| v == 0.25));
    }
    // Unit-scale forward Jacobians use the existing deployed forward, not the
    // new scale-offset/gradient implementation. Exercise every stored scale.
    for scale in 0..f.scales.len() {
        let mut unit = vec![0.0; f.scales.len()];
        unit[scale] = 1.0;
        close(
            gm[scale] - 0.25,
            dot(&matmul(f, &unit, &x, batch), &dy),
            2e-5,
        );
        close(ge[scale] - 0.25, dot(&embed(f, &unit, &ids), &de), 2e-5);
    }
    let first = gm.clone();
    accumulate_ternary_matmul_scale_vjp(
        &x,
        &view.bind_matmul(),
        &dy,
        batch,
        &mut workspace,
        &mut gm,
    )
    .unwrap();
    for (twice, once) in gm.iter().zip(first) {
        close(*twice - 0.25, 2.0 * (once - 0.25), 1e-12);
    }
}

#[test]
fn scale_adjoints_match_forward_jacobians_all_laws_groups_planes_bases_and_allocations() {
    let mut cases = 0;
    for (law, admitted) in ADMITTED_LAWS.iter().enumerate() {
        for planes in 1..=admitted.max_planes {
            for group in [32, 64, 128, 256] {
                for basis in [
                    Basis::Identity,
                    Basis::Hadamard { block: 4 },
                    Basis::SignedRht {
                        block: 4,
                        seed: 19,
                        domain: 7,
                    },
                    Basis::Hadamard { block: 128 },
                    Basis::SignedRht {
                        block: 128,
                        seed: 19,
                        domain: 7,
                    },
                ] {
                    for mode in 0..3 {
                        let cols = if matches!(
                            basis,
                            Basis::Hadamard { block: 128 } | Basis::SignedRht { block: 128, .. }
                        ) {
                            512
                        } else {
                            260
                        };
                        let f = Fixture::new(law, basis, group, cols, planes, mode);
                        for batch in [0, 1, 3] {
                            check(&f, batch);
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 1800);
}

#[test]
fn actual_forward_finite_differences_agree_for_both_consumers() {
    let mut partials = 0;
    for (law, admitted) in ADMITTED_LAWS.iter().enumerate() {
        for basis in [
            Basis::Identity,
            Basis::Hadamard { block: 4 },
            Basis::SignedRht {
                block: 4,
                seed: 19,
                domain: 7,
            },
            Basis::Hadamard { block: 128 },
            Basis::SignedRht {
                block: 128,
                seed: 19,
                domain: 7,
            },
        ] {
            for mode in [0, 1] {
                let cols = if matches!(
                    basis,
                    Basis::Hadamard { block: 128 } | Basis::SignedRht { block: 128, .. }
                ) {
                    128
                } else {
                    4
                };
                let f = Fixture::new(law, basis, 32, cols, admitted.max_planes, mode);
                let x = values(2 * cols as usize);
                let dy = values(6);
                let ids = [2, 0, 2];
                let de = values(3 * cols as usize);
                let mut row = vec![0.0; cols as usize];
                let mut gm = vec![0.0; f.scales.len()];
                let mut ge = gm.clone();
                accumulate_ternary_matmul_scale_vjp(
                    &x,
                    &f.view(&f.scales).bind_matmul(),
                    &dy,
                    2,
                    &mut row,
                    &mut gm,
                )
                .unwrap();
                accumulate_embed_scale_vjp(
                    &f.view(&f.scales).bind_gather(),
                    &ids,
                    &de,
                    &mut row,
                    &mut ge,
                )
                .unwrap();
                for i in 0..f.scales.len() {
                    let mut plus = f.scales.clone();
                    let mut minus = plus.clone();
                    plus[i] += 0.01;
                    minus[i] -= 0.01;
                    let h = f64::from(plus[i]) - f64::from(minus[i]);
                    close(
                        gm[i],
                        (dot(&matmul(&f, &plus, &x, 2), &dy)
                            - dot(&matmul(&f, &minus, &x, 2), &dy))
                            / h,
                        2e-5,
                    );
                    close(
                        ge[i],
                        (dot(&embed(&f, &plus, &ids), &de) - dot(&embed(&f, &minus, &ids), &de))
                            / h,
                        2e-5,
                    );
                    partials += 2;
                }
            }
        }
    }
    // Small profiles:33 scales *3 bases *2 consumers; wide profiles:
    //126 scales *2 bases *2 consumers.
    assert_eq!(partials, 702);
}

#[test]
fn signed_rotation_uses_forward_not_inverse_for_both_scale_adjoints() {
    let basis = Basis::SignedRht {
        block: 4,
        seed: 19,
        domain: 7,
    };
    let f = Fixture::new(1, basis, 32, 4, 3, 0);
    let identity = Fixture {
        layout: AdditiveLayout {
            basis: Basis::Identity,
            ..f.layout
        },
        planes: f.planes,
        trits: f.trits.clone(),
        scales: f.scales.clone(),
        index: None,
    };
    let x = values(4);
    let dy = [0.25, -0.75, 0.5];
    let ids = [2, 0, 2];
    let de = values(12);
    let mut correct = vec![0.0; f.scales.len()];
    let mut wrong = correct.clone();
    accumulate_ternary_matmul_scale_vjp(
        &x,
        &f.view(&f.scales).bind_matmul(),
        &dy,
        1,
        &mut [0.0; 4],
        &mut correct,
    )
    .unwrap();
    let mut wrong_x = x.clone();
    apply_inverse_basis(&mut wrong_x, basis).unwrap();
    accumulate_ternary_matmul_scale_vjp(
        &wrong_x,
        &identity.view(&f.scales).bind_matmul(),
        &dy,
        1,
        &mut [0.0; 4],
        &mut wrong,
    )
    .unwrap();
    assert!(
        correct
            .iter()
            .zip(&wrong)
            .any(|(a, b)| (a - b).abs() > 0.01)
    );
    correct.fill(0.0);
    wrong.fill(0.0);
    let mut wrong_dy = de.clone();
    for row in wrong_dy.as_chunks_mut::<4>().0 {
        apply_inverse_basis(row, basis).unwrap();
    }
    accumulate_embed_scale_vjp(
        &f.view(&f.scales).bind_gather(),
        &ids,
        &de,
        &mut [0.0; 4],
        &mut correct,
    )
    .unwrap();
    accumulate_embed_scale_vjp(
        &identity.view(&f.scales).bind_gather(),
        &ids,
        &wrong_dy,
        &mut [0.0; 4],
        &mut wrong,
    )
    .unwrap();
    assert!(
        correct
            .iter()
            .zip(&wrong)
            .any(|(a, b)| (a - b).abs() > 0.01)
    );
}

#[test]
fn shapes_overflow_and_late_invalid_index_reject_before_any_mutation() {
    let f = Fixture::new(1, Basis::Identity, 32, 4, 3, 0);
    let view = f.view(&f.scales);
    for case in 0..5 {
        let mut row = [17.0; 4];
        let mut out = vec![19.0; f.scales.len()];
        let result = match case {
            0 => accumulate_ternary_matmul_scale_vjp(
                &[0.0; 3],
                &view.bind_matmul(),
                &[0.0; 3],
                1,
                &mut row,
                &mut out,
            ),
            1 => accumulate_ternary_matmul_scale_vjp(
                &[0.0; 4],
                &view.bind_matmul(),
                &[0.0; 2],
                1,
                &mut row,
                &mut out,
            ),
            2 => accumulate_ternary_matmul_scale_vjp(
                &[],
                &view.bind_matmul(),
                &[],
                usize::MAX,
                &mut row,
                &mut out,
            ),
            3 => accumulate_ternary_matmul_scale_vjp(
                &[0.0; 4],
                &view.bind_matmul(),
                &[0.0; 3],
                1,
                &mut row[..3],
                &mut out,
            ),
            _ => accumulate_ternary_matmul_scale_vjp(
                &[0.0; 4],
                &view.bind_matmul(),
                &[0.0; 3],
                1,
                &mut row,
                &mut out[..1],
            ),
        };
        assert!(result.is_err());
        assert_eq!(row, [17.0; 4]);
        assert!(out.iter().all(|&v| v == 19.0));
    }
    let mut row = [17.0; 4];
    let mut out = vec![19.0; f.scales.len()];
    assert_eq!(
        accumulate_embed_scale_vjp(
            &view.bind_gather(),
            &[0, 2, 3],
            &[0.0; 12],
            &mut row,
            &mut out
        ),
        Err(AdditiveError::RowOutOfRange)
    );
    assert!(
        accumulate_embed_scale_vjp(&view.bind_gather(), &[0], &[], &mut row, &mut out).is_err()
    );
    assert_eq!(row, [17.0; 4]);
    assert!(out.iter().all(|&v| v == 19.0));
}

#[test]
fn checked_plan_offsets_are_absolute_and_failed_preflight_never_calls_reader() {
    for (law, admitted) in ADMITTED_LAWS.iter().enumerate() {
        for mode in 0..3 {
            let f = Fixture::new(
                law,
                Basis::SignedRht {
                    block: 4,
                    seed: 19,
                    domain: 7,
                },
                128,
                260,
                admitted.max_planes,
                mode,
            );
            let plan = if let Some(index) = &f.index {
                AdditiveScaleVjpPlan::from_per_tile(index.as_index())
            } else {
                AdditiveScaleVjpPlan::from_uniform(UniformPlan::new(f.layout, f.planes).unwrap())
            };
            assert_eq!(plan.stored_scale_count(), f.scales.len());
            let mut starts = vec![0];
            for plane in 0..f.view(&f.scales).plane_count() {
                let n = if let Some(index) = &f.index {
                    PerTilePlan::new(f.layout, index.tile_counts())
                        .unwrap()
                        .plane_trit_count(plane)
                        .unwrap()
                } else {
                    780
                };
                starts.push(starts.last().unwrap() + n);
            }
            let calls = std::cell::Cell::new(0);
            let mut actual = vec![0.25; f.scales.len()];
            let mut expected = actual.clone();
            let x = values(520);
            let dy = values(6);
            let mut row = vec![0.0; 260];
            plan.matmul(&x, &dy, 2, &mut row, &mut actual, |plane, index| {
                assert!(index >= starts[plane] && index < starts[plane + 1]);
                calls.set(calls.get() + 1);
                f.trits[index]
            })
            .unwrap();
            accumulate_ternary_matmul_scale_vjp(
                &x,
                &f.view(&f.scales).bind_matmul(),
                &dy,
                2,
                &mut row,
                &mut expected,
            )
            .unwrap();
            assert_eq!(calls.get(), 2 * f.trits.len());
            assert_eq!(actual, expected);
            let mut row = vec![17.0; 260];
            let mut out = vec![19.0; f.scales.len()];
            assert!(
                plan.matmul(&[], &[], usize::MAX, &mut row, &mut out, |_, _| panic!(
                    "reader called before overflow refusal"
                ))
                .is_err()
            );
            assert!(
                plan.embed(&[0, 3], &[0.25; 520], &mut row, &mut out, |_, _| panic!(
                    "reader called before index refusal"
                ))
                .is_err()
            );
            assert!(
                plan.matmul(&[], &[], 0, &mut row[..259], &mut out, |_, _| panic!(
                    "reader called before shape refusal"
                ))
                .is_err()
            );
            assert!(row.iter().all(|&v| v == 17.0));
            assert!(out.iter().all(|&v| v == 19.0));
            plan.matmul(&[], &[], 0, &mut row, &mut out, |_, _| {
                panic!("reader called for empty batch")
            })
            .unwrap();
            plan.embed(&[], &[], &mut row, &mut out, |_, _| {
                panic!("reader called for empty ids")
            })
            .unwrap();
            assert!(row.iter().all(|&v| v == 17.0));
            assert!(out.iter().all(|&v| v == 19.0));
        }
    }
}
