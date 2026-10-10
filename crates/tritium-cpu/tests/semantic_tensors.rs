//! Production CPU execution through the shared semantic tensor contract.

use tritium_core::{AdditiveView, Trit};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_schema::{
    AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
    ScalePrecision, Transport,
};
use tritium_spec::{TensorMatmul, TensorView, TernaryBackend};

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
            // Decoded i8 trits and f32 scales, not packed payload size.
            assert_eq!(uploaded.len_bytes(), 32);
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
