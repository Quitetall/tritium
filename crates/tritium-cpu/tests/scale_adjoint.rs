//! Actual packed handles; numerical conformance, not complete training acceptance.
use tritium_core::{
    AdditiveView, GemmShape, OwnedPerTileIndex, PerTilePlan, TernaryFormat, Trit, UniformPlan,
    accumulate_embed_scale_vjp, accumulate_ternary_matmul_scale_vjp,
};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};
use tritium_spec::{
    AdditiveMetadata, BackendError, DeviceBuffer, TensorMetadata, TensorStreamEncoding, TensorView,
    TernaryBackend,
};
#[path = "common/stream_source.rs"]
mod stream_source;

fn layout(law: usize, codec: PlaneCodec, group: u16, cols: usize, basis: Basis) -> AdditiveLayout {
    AdditiveLayout {
        rows: 3,
        cols: cols as u64,
        tile: 256,
        group,
        max_planes: ADMITTED_LAWS[law].max_planes,
        allocation: PlaneAllocation::Uniform,
        codec,
        law: ADMITTED_LAWS[law].law,
        basis,
        transport: Transport::Raw,
    }
}
fn codes(n: usize, codec: PlaneCodec) -> Vec<Trit> {
    (0..n)
        .map(|i| {
            if codec == PlaneCodec::S34 {
                [Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS][i % 4]
            } else {
                Trit::from_i8(((i * 7 + i / 11) % 3) as i8 - 1).unwrap()
            }
        })
        .collect()
}
fn scales(n: usize) -> Vec<f32> {
    (0..n).map(|i| [0.125, 0.5, 1.0, 0.0][i % 4]).collect()
}
fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 7) as f32 * 0.125 - 0.375).collect()
}
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn check(view: AdditiveView<'_>) -> usize {
    let cpu = CpuBackend::new();
    let rows = view.layout().rows as usize;
    let cols = view.layout().cols as usize;
    let encoded = AdditiveTensor::from_view(view)
        .unwrap()
        .encode_payload()
        .unwrap();
    let source = stream_source::Source::new(
        TensorMetadata::Additive(AdditiveMetadata::from_view(view)),
        TensorStreamEncoding::CanonicalRawAdditive,
        &encoded,
    );
    let direct = cpu.upload_tensor(TensorView::Additive(view)).unwrap();
    let streamed = cpu.upload_tensor_stream(&source).unwrap();
    assert_eq!(source.calls.get(), 1);
    assert_eq!(direct.len_bytes(), streamed.len_bytes());
    for handle in [direct, streamed] {
        let bytes = handle.len_bytes();
        assert_eq!(
            cpu.additive_stored_scale_count(&*handle).unwrap(),
            view.scales().len()
        );
        for batch in [0, 1, 3] {
            let x = values(batch * cols);
            let dy = values(batch * rows);
            let ids = if batch == 0 {
                vec![]
            } else {
                vec![rows - 1, 0, rows - 1]
            };
            let de = values(ids.len() * cols);
            let mut row = vec![17.0; cols];
            let mut expected_row = row.clone();
            let mut actual = vec![0.25; view.scales().len()];
            let mut expected = actual.clone();
            for _ in 0..2 {
                accumulate_ternary_matmul_scale_vjp(
                    &x,
                    &view.bind_matmul(),
                    &dy,
                    batch,
                    &mut expected_row,
                    &mut expected,
                )
                .unwrap();
                cpu.accumulate_additive_matmul_scale_vjp(
                    &*handle,
                    &x,
                    &dy,
                    batch,
                    &mut row,
                    &mut actual,
                )
                .unwrap();
                assert_eq!(bits(&actual), bits(&expected));
                assert_eq!(row, expected_row);
                accumulate_embed_scale_vjp(
                    &view.bind_gather(),
                    &ids,
                    &de,
                    &mut expected_row,
                    &mut expected,
                )
                .unwrap();
                cpu.accumulate_additive_embed_scale_vjp(&*handle, &ids, &de, &mut row, &mut actual)
                    .unwrap();
                assert_eq!(bits(&actual), bits(&expected));
                assert_eq!(row, expected_row);
            }
            assert_eq!(handle.len_bytes(), bytes);
        }
    }
    6
}

#[test]
fn direct_and_stream_uniform_handles_match_shared_scale_gradient_bits() {
    let mut cases = 0;
    for (law, admitted) in ADMITTED_LAWS.iter().enumerate() {
        for group in [32, 128] {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                for (cols, basis) in [
                    (257, Basis::Identity),
                    (260, Basis::Identity),
                    (260, Basis::Hadamard { block: 4 }),
                    (
                        260,
                        Basis::SignedRht {
                            block: 4,
                            seed: 19,
                            domain: 7,
                        },
                    ),
                    (512, Basis::Hadamard { block: 128 }),
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
                    for planes in 1..=admitted.max_planes {
                        let layout = layout(law, codec, group, cols, basis);
                        let p = UniformPlan::new(layout, planes).unwrap();
                        cases += check(
                            AdditiveView::new(
                                layout,
                                planes,
                                &codes(p.trit_count(), codec),
                                &scales(p.scale_count()),
                            )
                            .unwrap(),
                        );
                    }
                }
            }
        }
    }
    assert_eq!(cases, 2040);
}

#[test]
fn compact_variable_pruned_and_rotated_handles_match_shared_scale_gradient_bits() {
    let mut cases = 0;
    for (law, admitted) in ADMITTED_LAWS.iter().enumerate() {
        for group in [32, 128] {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                for (cols, basis) in [
                    (260, Basis::Identity),
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
                    let layout = AdditiveLayout {
                        allocation: PlaneAllocation::PerTile,
                        ..layout(law, codec, group, cols, basis)
                    };
                    for counts in [
                        [0, admitted.max_planes, 1, 0, admitted.max_planes, 1],
                        [0; 6],
                    ] {
                        let p = PerTilePlan::new(layout, &counts).unwrap();
                        let (n, s) = (p.trit_count(), p.scale_count());
                        let index = OwnedPerTileIndex::new(layout, counts.to_vec()).unwrap();
                        cases += check(
                            AdditiveView::new_per_tile(
                                index.as_index(),
                                &codes(n, codec),
                                &scales(s),
                            )
                            .unwrap(),
                        );
                    }
                }
            }
        }
    }
    assert_eq!(cases, 1152);
}

#[test]
fn s34_physical_quartets_cross_ragged_logical_rows_and_compact_holes() {
    let layout = AdditiveLayout {
        rows: 4,
        ..layout(1, PlaneCodec::S34, 32, 257, Basis::Identity)
    };
    let p = UniformPlan::new(layout, 3).unwrap();
    assert_eq!(
        check(
            AdditiveView::new(
                layout,
                3,
                &codes(p.trit_count(), PlaneCodec::S34),
                &scales(p.scale_count())
            )
            .unwrap()
        ),
        6
    );
    let layout = AdditiveLayout {
        allocation: PlaneAllocation::PerTile,
        ..layout
    };
    let counts = vec![0, 3, 1, 3, 2, 3, 3, 3];
    let p = PerTilePlan::new(layout, &counts).unwrap();
    let (n, s) = (p.trit_count(), p.scale_count());
    let index = OwnedPerTileIndex::new(layout, counts).unwrap();
    assert_eq!(
        check(
            AdditiveView::new_per_tile(index.as_index(), &codes(n, PlaneCodec::S34), &scales(s))
                .unwrap()
        ),
        6
    );
}

#[test]
fn bad_shapes_overflow_and_late_invalid_indices_preserve_buffers() {
    let cpu = CpuBackend::new();
    let layout = layout(1, PlaneCodec::D2, 32, 4, Basis::Identity);
    let handle = cpu
        .upload_tensor(TensorView::Additive(
            AdditiveView::new(layout, 1, &[Trit::POS; 12], &[0.5; 3]).unwrap(),
        ))
        .unwrap();
    for (batch, x, g, r, s) in [
        (1, 3, 3, 4, 3),
        (1, 4, 2, 4, 3),
        (1, 4, 3, 3, 3),
        (1, 4, 3, 4, 2),
        (0, 1, 0, 4, 3),
        (0, 0, 0, 0, 3),
        (usize::MAX, 0, 0, 4, 3),
        (usize::MAX / 3, 0, 0, 4, 3),
    ] {
        let mut row = vec![17.0; r];
        let mut out = vec![19.0; s];
        assert!(matches!(
            cpu.accumulate_additive_matmul_scale_vjp(
                &*handle,
                &vec![0.25; x],
                &vec![0.25; g],
                batch,
                &mut row,
                &mut out
            ),
            Err(BackendError::InvalidInput(_))
        ));
        assert!(row.iter().all(|&v| v == 17.0));
        assert!(out.iter().all(|&v| v == 19.0));
    }
    for (ids, g, r, s) in [
        (vec![0, 2, 3], 12, 4, 3),
        (vec![0], 3, 4, 3),
        (vec![0], 4, 3, 3),
        (vec![0], 4, 4, 2),
        (vec![], 0, 0, 3),
    ] {
        let mut row = vec![17.0; r];
        let mut out = vec![19.0; s];
        assert!(matches!(
            cpu.accumulate_additive_embed_scale_vjp(
                &*handle,
                &ids,
                &vec![0.25; g],
                &mut row,
                &mut out
            ),
            Err(BackendError::InvalidInput(_))
        ));
        assert!(row.iter().all(|&v| v == 17.0));
        assert!(out.iter().all(|&v| v == 19.0));
    }
}

#[derive(Debug)]
struct Foreign;
impl DeviceBuffer for Foreign {
    fn len_bytes(&self) -> usize {
        0
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn dense_foreign_and_legacy_handles_do_not_fall_back_to_other_representations() {
    let cpu = CpuBackend::new();
    let dense = cpu
        .upload_tensor(TensorView::Dense {
            rows: 1,
            cols: 4,
            values: &[1.0; 4],
        })
        .unwrap();
    let mut bytes = vec![0; tritium_format::TQ2_0_BLOCK_BYTES];
    tritium_format::pack_tq2_0_row(&[Trit::POS; 4], &[half::f16::ONE], &mut bytes).unwrap();
    let legacy = cpu
        .upload_weights(&bytes, GemmShape { m: 1, n: 1, k: 4 }, TernaryFormat::Tq2_0)
        .unwrap();
    for (index, handle) in [&*dense, &*legacy, &Foreign as &dyn DeviceBuffer]
        .into_iter()
        .enumerate()
    {
        let mut row = [17.0; 4];
        let mut out = [19.0; 1];
        let errors = [
            cpu.additive_stored_scale_count(handle).unwrap_err(),
            cpu.accumulate_additive_matmul_scale_vjp(
                handle,
                &[0.25; 4],
                &[0.25],
                1,
                &mut row,
                &mut out,
            )
            .unwrap_err(),
            cpu.accumulate_additive_embed_scale_vjp(handle, &[0], &[0.25; 4], &mut row, &mut out)
                .unwrap_err(),
        ];
        for error in errors {
            assert!(if index == 0 {
                matches!(error, BackendError::UnsupportedTensor)
            } else {
                matches!(error, BackendError::InvalidInput(_))
            });
        }
        assert_eq!(row, [17.0; 4]);
        assert_eq!(out, [19.0; 1]);
    }
}

#[test]
fn shared_packed_handle_allows_concurrent_scale_adjoint_callers() {
    let cpu = CpuBackend::new();
    let layout = layout(
        1,
        PlaneCodec::B3,
        128,
        260,
        Basis::SignedRht {
            block: 4,
            seed: 19,
            domain: 7,
        },
    );
    let p = UniformPlan::new(layout, 3).unwrap();
    let trits = codes(p.trit_count(), layout.codec);
    let scales = scales(p.scale_count());
    let view = AdditiveView::new(layout, 3, &trits, &scales).unwrap();
    let handle = cpu.upload_tensor(TensorView::Additive(view)).unwrap();
    let x = values(260);
    let dy = values(3);
    let ids = [2, 0, 2];
    let de = values(780);
    let mut expected = vec![0.25; scales.len()];
    accumulate_ternary_matmul_scale_vjp(
        &x,
        &view.bind_matmul(),
        &dy,
        1,
        &mut vec![0.0; 260],
        &mut expected,
    )
    .unwrap();
    accumulate_embed_scale_vjp(
        &view.bind_gather(),
        &ids,
        &de,
        &mut vec![0.0; 260],
        &mut expected,
    )
    .unwrap();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let handle = &*handle;
            let x = &x;
            let dy = &dy;
            let de = &de;
            let expected = &expected;
            scope.spawn(move || {
                let mut row = vec![0.0; 260];
                let mut actual = vec![0.25; expected.len()];
                let cpu = CpuBackend::new();
                cpu.accumulate_additive_matmul_scale_vjp(handle, x, dy, 1, &mut row, &mut actual)
                    .unwrap();
                cpu.accumulate_additive_embed_scale_vjp(handle, &ids, de, &mut row, &mut actual)
                    .unwrap();
                assert_eq!(bits(&actual), bits(expected));
            });
        }
    });
}
