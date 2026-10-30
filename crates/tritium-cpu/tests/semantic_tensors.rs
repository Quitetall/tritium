//! Production CPU execution through the shared semantic tensor contract.

use tritium_core::{AdditiveView, Trit};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_schema::{
    AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
    ScalePrecision, Transport,
};
use tritium_spec::{TensorMatmul, TensorView, TernaryBackend};

#[test]
fn compact_and_all_pruned_views_use_allocation_aware_cpu_storage() {
    let cpu = CpuBackend::new();
    let mut layout = layout(Basis::Identity, PlaneCodec::D2);
    layout.allocation = PlaneAllocation::PerTile;
    for counts in [[1, 0], [0, 0]] {
        let mut spans = vec![
            tritium_core::PerTileRowSpan::default();
            tritium_core::PerTileIndex::required_storage(layout, &counts).unwrap()
        ];
        let index = tritium_core::PerTileIndex::build(layout, &counts, &mut spans).unwrap();
        let trits = vec![Trit::POS; index.trit_count()];
        let scales = vec![0.5; index.scale_count()];
        let view = AdditiveView::new_per_tile(index, &trits, &scales).unwrap();
        let caps = cpu
            .tensor_caps(TensorView::Additive(view))
            .unwrap()
            .unwrap();
        let uploaded = cpu.upload_tensor(TensorView::Additive(view)).unwrap();
        assert_eq!(caps.execution, tritium_schema::TensorExecution::Native);
        assert_eq!(caps.payload_bytes, uploaded.len_bytes() as u64);
        let mut out = [99.; 2];
        cpu.matmul(TensorMatmul {
            act: &[1.; 4],
            tensor: &*uploaded,
            batch: 1,
            transformed_act: &mut [0.; 4],
            out: &mut out,
        })
        .unwrap();
        assert_eq!(out, if counts[0] == 0 { [0., 0.] } else { [2., 0.] });
        let owned = AdditiveTensor::from_view(view).unwrap();
        assert_eq!(owned.tile_counts(), Some(counts.as_slice()));
        assert_eq!(owned.trits(), view.trits());
        assert_eq!(owned.scales(), view.scales());
    }
}

#[test]
fn compact_cpu_replays_shared_candidate_vectors_in_one_and_four_threads() {
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                assert_eq!(
                    tritium_testkit::assert_candidate_per_tile_conformance(
                        &CpuBackend::new(),
                        tritium_schema::TensorExecution::Native
                    ),
                    624
                );
            });
    }
}

#[test]
fn compact_upload_counts_every_owned_buffer_and_rejects_invalid_s34() {
    use tritium_schema::{ADMITTED_LAWS, TensorUploadPolicy};
    use tritium_spec::BackendError;
    let cpu = CpuBackend::new();
    for (codec, payload) in [(PlaneCodec::D2, 213), (PlaneCodec::B3, 174)] {
        let mut meta = layout(Basis::Identity, codec);
        meta.cols = 257;
        meta.group = 128;
        meta.law = ADMITTED_LAWS[0].law;
        meta.allocation = PlaneAllocation::PerTile;
        let owned = AdditiveTensor::new_per_tile(
            meta,
            vec![1, 3, 2, 0],
            vec![Trit::POS; 771],
            vec![0.5; 9],
        )
        .unwrap();
        let expected = payload
            + 4
            + 6 * core::mem::size_of::<tritium_core::PerTileRowSpan>()
            + 3 * 2 * core::mem::size_of::<usize>();
        let view = TensorView::Additive(owned.view());
        assert_eq!(
            cpu.tensor_caps(view).unwrap().unwrap().payload_bytes,
            expected as u64
        );
        assert!(matches!(
            cpu.upload_tensor_checked(
                view,
                TensorUploadPolicy {
                    native_only: true,
                    max_payload_bytes: Some(expected as u64 - 1),
                }
            ),
            Err(BackendError::TensorPayloadBudgetExceeded { .. })
        ));
        let (uploaded, _) = cpu
            .upload_tensor_checked(
                view,
                TensorUploadPolicy {
                    native_only: true,
                    max_payload_bytes: Some(expected as u64),
                },
            )
            .unwrap();
        assert_eq!(uploaded.len_bytes(), expected);
    }
    let mut meta = layout(Basis::Identity, PlaneCodec::S34);
    meta.allocation = PlaneAllocation::PerTile;
    for (cols, trits) in [(4, vec![Trit::POS; 4]), (5, vec![Trit::ZERO; 5])] {
        meta.rows = 1;
        meta.cols = cols;
        let owned = AdditiveTensor::new_per_tile(meta, vec![1], trits, vec![1.]).unwrap();
        let view = TensorView::Additive(owned.view());
        assert!(cpu.tensor_caps(view).is_err());
        assert!(
            cpu.upload_tensor_checked(view, TensorUploadPolicy::default())
                .is_err()
        );
    }
}

fn layout(basis: Basis, codec: PlaneCodec) -> AdditiveLayout {
    AdditiveLayout {
        rows: 2,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: 3,
        allocation: PlaneAllocation::Uniform,
        codec,
        law: ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F32,
        },
        basis,
        transport: Transport::Raw,
    }
}

#[test]
fn encoded_additive_tensors_execute_with_frozen_logical_weights() {
    let plane = [
        Trit::POS,
        Trit::NEG,
        Trit::ZERO,
        Trit::POS,
        Trit::ZERO,
        Trit::POS,
        Trit::NEG,
        Trit::POS,
    ];
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        for (basis, rows) in [
            (
                Basis::Identity,
                [[1.5, -1.5, 0., 1.5], [0., 2.25, -2.25, 2.25]],
            ),
            (
                Basis::Hadamard { block: 4 },
                [[0.75, 0.75, -0.75, 2.25], [1.125, -3.375, 1.125, 1.125]],
            ),
            (
                Basis::SignedRht {
                    block: 4,
                    seed: 7,
                    domain: 9,
                },
                [[0.75, -0.75, 0.75, -2.25], [1.125, 3.375, -1.125, -1.125]],
            ),
        ] {
            let owned = AdditiveTensor::new(
                layout(basis, codec),
                2,
                plane.into_iter().chain(plane).collect(),
                vec![1., 2., 0.5, 0.25],
            )
            .unwrap();
            let payload = owned.encode_payload().unwrap();
            let decoded =
                AdditiveTensor::decode_payload(layout(basis, codec), 2, &payload).unwrap();
            let cpu = CpuBackend::new();
            let uploaded = cpu
                .upload_tensor(TensorView::Additive(decoded.view()))
                .unwrap();
            // Two canonical packed planes plus four retained f32 scales.
            assert_eq!(uploaded.len_bytes(), 20);
            let mut gathered = [0.; 12];
            cpu.embed_rows(&*uploaded, &[1, 0, 1], &mut gathered)
                .unwrap();
            assert_eq!(&gathered[..4], &rows[1]);
            assert_eq!(&gathered[4..8], &rows[0]);
            assert_eq!(&gathered[8..], &rows[1]);
            let act = [1., 2., 3., 4., -2., 1., 0., 3.];
            let mut scratch = [0.; 8];
            let mut out = [0.; 4];
            cpu.matmul(TensorMatmul {
                act: &act,
                tensor: &*uploaded,
                batch: 2,
                transformed_act: &mut scratch,
                out: &mut out,
            })
            .unwrap();
            for (batch, activation) in act.as_chunks::<4>().0.iter().enumerate() {
                for (row, weights) in rows.iter().enumerate() {
                    let expected: f32 = activation.iter().zip(weights).map(|(a, w)| a * w).sum();
                    assert_eq!(out[batch * 2 + row], expected);
                }
            }
        }
    }
}

#[test]
fn cpu_dense_contract_and_invalid_gather_are_fail_closed() {
    let cpu = CpuBackend::new();
    let tensor = cpu
        .upload_tensor(TensorView::Dense {
            rows: 2,
            cols: 3,
            values: &[1., 2., 3., -1., 0.5, 2.],
        })
        .unwrap();
    assert_eq!(tensor.len_bytes(), 24);
    let mut scratch = [99.; 3];
    let mut out = [99.; 2];
    cpu.matmul(TensorMatmul {
        act: &[2., -1., 4.],
        tensor: &*tensor,
        batch: 1,
        transformed_act: &mut scratch,
        out: &mut out,
    })
    .unwrap();
    assert_eq!(out, [12., 5.5]);
    assert_eq!(scratch, [2., -1., 4.]);
    let mut gathered = [99.; 6];
    cpu.embed_rows(&*tensor, &[1, 0], &mut gathered).unwrap();
    assert_eq!(gathered, [-1., 0.5, 2., 1., 2., 3.]);
    gathered.fill(99.);
    assert!(cpu.embed_rows(&*tensor, &[0, 2], &mut gathered).is_err());
    assert_eq!(gathered, [99.; 6]);
    let zero = cpu
        .upload_tensor(TensorView::Dense {
            rows: 2,
            cols: 0,
            values: &[],
        })
        .unwrap();
    cpu.embed_rows(&*zero, &[1, 0], &mut []).unwrap();
    assert!(cpu.embed_rows(&*zero, &[2], &mut []).is_err());
    assert!(
        cpu.upload_tensor(TensorView::Dense {
            rows: usize::MAX,
            cols: 2,
            values: &[]
        })
        .is_err()
    );
    assert!(
        cpu.upload_tensor(TensorView::Dense {
            rows: 2,
            cols: 3,
            values: &[0.; 5]
        })
        .is_err()
    );
    let empty = cpu
        .upload_tensor(TensorView::Dense {
            rows: 0,
            cols: 0,
            values: &[],
        })
        .unwrap();
    cpu.matmul(TensorMatmul {
        act: &[],
        tensor: &*empty,
        batch: usize::MAX,
        transformed_act: &mut [],
        out: &mut [],
    })
    .unwrap();
    scratch.fill(99.);
    out.fill(99.);
    assert!(
        cpu.matmul(TensorMatmul {
            act: &[2., -1.],
            tensor: &*tensor,
            batch: 1,
            transformed_act: &mut scratch,
            out: &mut out
        })
        .is_err()
    );
    assert_eq!(scratch, [99.; 3]);
    assert_eq!(out, [99.; 2]);
}

#[test]
fn cpu_executes_each_currently_admitted_law_without_changing_precision() {
    for (anchor, relation, precision, planes, scales, expected) in [
        (
            ScaleAnchor::Group,
            PlaneRelation::Free,
            ScalePrecision::F32,
            2,
            vec![1., 2., 0.5, 0.25],
            [6., 9.],
        ),
        (
            ScaleAnchor::Group,
            PlaneRelation::Free,
            ScalePrecision::F16,
            2,
            vec![1., 2., 0.5, 0.25],
            [6., 9.],
        ),
        (
            ScaleAnchor::Group,
            PlaneRelation::Tied { num: 1, den: 3 },
            ScalePrecision::F16,
            2,
            vec![3., 6.],
            [16., 32.],
        ),
        (
            ScaleAnchor::Tensor,
            PlaneRelation::Free,
            ScalePrecision::F32,
            1,
            vec![1.5],
            [6., 6.],
        ),
    ] {
        let mut meta = layout(Basis::Identity, PlaneCodec::D2);
        meta.law = ScaleLaw {
            anchor,
            relation,
            precision,
        };
        meta.max_planes = planes;
        let owned = AdditiveTensor::new(
            meta,
            planes,
            vec![Trit::POS; usize::from(planes) * 8],
            scales,
        )
        .unwrap();
        let cpu = CpuBackend::new();
        let uploaded = cpu
            .upload_tensor(TensorView::Additive(owned.view()))
            .unwrap();
        let mut scratch = [0.; 4];
        let mut out = [0.; 2];
        cpu.matmul(TensorMatmul {
            act: &[1.; 4],
            tensor: &*uploaded,
            batch: 1,
            transformed_act: &mut scratch,
            out: &mut out,
        })
        .unwrap();
        assert_eq!(out, expected);
    }
}

#[test]
fn cpu_upload_owns_data_and_rejects_foreign_buffers_and_inexact_scales() {
    let cpu = CpuBackend::new();
    let mut trits = [Trit::POS; 8];
    let mut scales = [1., 2.];
    let uploaded = {
        let view =
            AdditiveView::new(layout(Basis::Identity, PlaneCodec::D2), 1, &trits, &scales).unwrap();
        cpu.upload_tensor(TensorView::Additive(view)).unwrap()
    };
    trits.fill(Trit::ZERO);
    scales.fill(0.);
    let mut out = [0.; 4];
    cpu.embed_rows(&*uploaded, &[1], &mut out).unwrap();
    assert_eq!(out, [2.; 4]);
    let foreign = tritium_testkit::ReferenceBackend::new()
        .upload_tensor(TensorView::Dense {
            rows: 1,
            cols: 4,
            values: &[1.; 4],
        })
        .unwrap();
    out.fill(99.);
    assert!(cpu.embed_rows(&*foreign, &[0], &mut out).is_err());
    assert_eq!(out, [99.; 4]);
    let mut meta = layout(Basis::Identity, PlaneCodec::D2);
    meta.law.precision = ScalePrecision::F16;
    let view = AdditiveView::new(meta, 1, &trits, &[0.1, 0.2]).unwrap();
    assert!(cpu.upload_tensor(TensorView::Additive(view)).is_err());
}
