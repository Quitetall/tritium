//! Stored-coordinate fragments, not new owners or model-memory qualification.
use tritium_core::{
    AdditiveAdjointSpan, AdditiveError, AdditiveScaleVjpPlan, AdditiveWeightVjpPlan,
    OwnedPerTileIndex, PerTilePlan, Trit, UniformPlan,
};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|n| n.to_bits()).collect()
}
fn values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| [0.125, -0.75, 1024.0, -1024.0, 0.0][i % 5])
        .collect()
}

#[test]
fn disjoint_spans_preserve_full_accumulation_bits_across_profiles() {
    let mut cases = 0;
    for law in ADMITTED_LAWS {
        for planes in 1..=law.max_planes {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                for basis in [
                    Basis::Identity,
                    Basis::Hadamard { block: 4 },
                    Basis::SignedRht {
                        block: 4,
                        seed: 19,
                        domain: 7,
                    },
                ] {
                    for mode in 0..3 {
                        let layout = AdditiveLayout {
                            rows: 3,
                            cols: 260,
                            tile: 256,
                            group: 32,
                            max_planes: law.max_planes,
                            allocation: if mode == 0 {
                                PlaneAllocation::Uniform
                            } else {
                                PlaneAllocation::PerTile
                            },
                            codec,
                            law: law.law,
                            basis,
                            transport: Transport::Raw,
                        };
                        let counts =
                            [0, planes, 1, planes, 0, 1].map(|n| if mode == 2 { 0 } else { n });
                        let index = (mode != 0)
                            .then(|| OwnedPerTileIndex::new(layout, counts.to_vec()).unwrap());
                        let (ns, nt, scale) = if let Some(index) = &index {
                            let plan = PerTilePlan::new(layout, &counts).unwrap();
                            (
                                plan.scale_count(),
                                plan.trit_count(),
                                AdditiveScaleVjpPlan::from_per_tile(index.as_index()),
                            )
                        } else {
                            let plan = UniformPlan::new(layout, planes).unwrap();
                            (
                                plan.scale_count(),
                                plan.trit_count(),
                                AdditiveScaleVjpPlan::from_uniform(plan),
                            )
                        };
                        let trits: Vec<_> = (0..nt)
                            .map(|i| {
                                if codec == PlaneCodec::S34 {
                                    [Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS][i % 4]
                                } else {
                                    Trit::from_i8((i % 3) as i8 - 1).unwrap()
                                }
                            })
                            .collect();
                        let weight = AdditiveWeightVjpPlan::from(scale);
                        let x = values(3 * 260);
                        let dy = values(9);
                        let ids = [2, 0, 2];
                        let de = values(ids.len() * 260);
                        let mut row = vec![17.0; 260];
                        for embedding in [false, true] {
                            for weights in [false, true] {
                                let count = if weights {
                                    weight.present_weight_count()
                                } else {
                                    ns
                                };
                                let mut expected = vec![0.25; count];
                                let mut actual = expected.clone();
                                for _ in 0..2 {
                                    match (embedding, weights) {
                                        (false, false) => scale.matmul(
                                            &x,
                                            &dy,
                                            3,
                                            &mut row,
                                            &mut expected,
                                            |_, i| trits[i],
                                        ),
                                        (false, true) => {
                                            weight.matmul(&x, &dy, 3, &mut row, &mut expected)
                                        }
                                        (true, false) => scale.embed(
                                            &ids,
                                            &de,
                                            &mut row,
                                            &mut expected,
                                            |_, i| trits[i],
                                        ),
                                        (true, true) => {
                                            weight.embed(&ids, &de, &mut row, &mut expected)
                                        }
                                    }
                                    .unwrap();
                                    // Deliberately cut across row, plane and scale-group boundaries.
                                    for (ordinal, out) in actual.chunks_mut(37).enumerate() {
                                        let span = AdditiveAdjointSpan::new(ordinal * 37, out);
                                        match (embedding, weights) {
                                            (false, false) => scale.matmul_span(
                                                &x,
                                                &dy,
                                                3,
                                                &mut row,
                                                span,
                                                |_, i| trits[i],
                                            ),
                                            (false, true) => {
                                                weight.matmul_span(&x, &dy, 3, &mut row, span)
                                            }
                                            (true, false) => scale.embed_span(
                                                &ids,
                                                &de,
                                                &mut row,
                                                span,
                                                |_, i| trits[i],
                                            ),
                                            (true, true) => {
                                                weight.embed_span(&ids, &de, &mut row, span)
                                            }
                                        }
                                        .unwrap();
                                    }
                                    assert_eq!(bits(&actual), bits(&expected));
                                }
                                cases += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 1000);
}

#[test]
fn span_and_input_refusals_leave_outputs_scratch_and_reader_untouched() {
    let layout = AdditiveLayout {
        rows: 2,
        cols: 64,
        tile: 256,
        group: 32,
        max_planes: 3,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: ADMITTED_LAWS[0].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let scale = AdditiveScaleVjpPlan::from_uniform(UniformPlan::new(layout, 3).unwrap());
    let weight = AdditiveWeightVjpPlan::from(scale);
    let x = vec![0.5; 64];
    let dy = [1.0, -0.5];
    for weights in [false, true] {
        let count = if weights {
            weight.present_weight_count()
        } else {
            scale.stored_scale_count()
        };
        for start in [count, usize::MAX] {
            let mut out = [17.0; 2];
            let mut row = vec![19.0; 64];
            let mut reads = 0;
            let result = if weights {
                weight.matmul_span(
                    &x,
                    &dy,
                    1,
                    &mut row,
                    AdditiveAdjointSpan::new(start, &mut out),
                )
            } else {
                scale.matmul_span(
                    &x,
                    &dy,
                    1,
                    &mut row,
                    AdditiveAdjointSpan::new(start, &mut out),
                    |_, _| {
                        reads += 1;
                        Trit::POS
                    },
                )
            };
            assert!(matches!(
                result,
                Err(AdditiveError::MatmulShape | AdditiveError::SizeOverflow)
            ));
            assert_eq!(reads, 0);
            assert_eq!(out, [17.0; 2]);
            assert_eq!(row, vec![19.0; 64]);
        }
        let mut out = [];
        let mut row = vec![19.0; 64];
        let result = if weights {
            weight.embed_span(
                &[0, 2],
                &vec![0.5; 128],
                &mut row,
                AdditiveAdjointSpan::new(count, &mut out),
            )
        } else {
            scale.embed_span(
                &[0, 2],
                &vec![0.5; 128],
                &mut row,
                AdditiveAdjointSpan::new(count, &mut out),
                |_, _| panic!("invalid id must refuse before payload read"),
            )
        };
        assert_eq!(result, Err(AdditiveError::RowOutOfRange));
        assert_eq!(row, vec![19.0; 64]);
    }
}
