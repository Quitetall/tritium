//! ADR 0044 D3/D8 consumer binding through the public core/backend seam.

use tritium_core::{
    AdditiveError, AdditiveView, Trit, apply_basis, apply_inverse_basis, reference_embed,
    reference_ternary_matmul,
};
use tritium_schema::{
    AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
    ScalePrecision, Transport,
};
use tritium_spec::{TensorView, TernaryBackend};

fn layout(basis: Basis) -> AdditiveLayout {
    AdditiveLayout {
        rows: 2,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: 3,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F32,
        },
        basis,
        transport: Transport::Raw,
    }
}

fn trits() -> [Trit; 8] {
    [
        Trit::POS,
        Trit::NEG,
        Trit::ZERO,
        Trit::POS,
        Trit::ZERO,
        Trit::POS,
        Trit::NEG,
        Trit::POS,
    ]
}

#[test]
fn frozen_gather_vectors_unrotate_before_returning_rows() {
    let trits = trits();
    for (basis, first, second) in [
        (Basis::Identity, [1., -1., 0., 1.], [0., 2., -2., 2.]),
        (
            Basis::Hadamard { block: 4 },
            [0.5, 0.5, -0.5, 1.5],
            [1., -3., 1., 1.],
        ),
        (
            Basis::SignedRht {
                block: 4,
                seed: 7,
                domain: 9,
            },
            [0.5, -0.5, 0.5, -1.5],
            [1., 3., -1., -1.],
        ),
    ] {
        let view = AdditiveView::new(layout(basis), 1, &trits, &[1., 2.]).unwrap();
        let mut out = [99.; 12];
        reference_embed(&view.bind_gather(), &[1, 0, 1], &mut out).unwrap();
        assert_eq!(&out[..4], &second);
        assert_eq!(&out[4..8], &first);
        assert_eq!(&out[8..], &second);

        // The same stored matrix must represent the same logical weights for
        // matmul and gather, with consumer semantics selected by the type.
        let mut transformed = [0.; 8];
        let mut multiplied = [0.; 4];
        let activations = [1., 2., 3., 4., -2., 1., 0., 3.];
        reference_ternary_matmul(
            &activations,
            &view.bind_matmul(),
            2,
            &mut transformed,
            &mut multiplied,
        )
        .unwrap();
        for (batch, activation) in activations.as_chunks::<4>().0.iter().enumerate() {
            for (row, logical) in [first, second].iter().enumerate() {
                let expected: f32 = activation.iter().zip(logical).map(|(x, w)| x * w).sum();
                assert_eq!(multiplied[batch * 2 + row], expected);
            }
        }
        let backend = tritium_testkit::ReferenceBackend::new();
        let uploaded = backend.upload_tensor(TensorView::Additive(view)).unwrap();
        out.fill(99.);
        backend
            .embed_rows(&*uploaded, &[1, 0, 1], &mut out)
            .unwrap();
        assert_eq!(&out[..4], &second);
        assert_eq!(&out[4..8], &first);
        assert_eq!(&out[8..], &second);
    }
}

#[test]
fn signed_inverse_is_not_the_forward_transform_and_replays_across_blocks() {
    let basis = Basis::SignedRht {
        block: 4,
        seed: 7,
        domain: 9,
    };
    let original = [1., 2., 3., 4., 5., 6., 7., 8.];
    let mut rotated = original;
    apply_basis(&mut rotated, basis).unwrap();
    let mut wrong = rotated;
    apply_basis(&mut wrong, basis).unwrap();
    assert_ne!(wrong, original);
    apply_inverse_basis(&mut rotated, basis).unwrap();
    assert_eq!(rotated, original);
}

#[test]
fn every_basis_roundtrips_multiple_blocks_and_odd_normalization() {
    let original = core::array::from_fn::<_, 32, _>(|i| i as f32 - 15.);
    for block in [1, 2, 4, 8, 16, 32] {
        for basis in [
            Basis::Hadamard { block },
            Basis::SignedRht {
                block,
                seed: 19,
                domain: 23,
            },
        ] {
            let mut values = original;
            apply_basis(&mut values, basis).unwrap();
            apply_inverse_basis(&mut values, basis).unwrap();
            for (actual, expected) in values.iter().zip(original) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "{basis:?}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn inverse_invalid_geometry_leaves_values_untouched() {
    for basis in [
        Basis::Hadamard { block: 4 },
        Basis::SignedRht {
            block: 4,
            seed: 7,
            domain: 9,
        },
        Basis::Hadamard { block: 0 },
        Basis::SignedRht {
            block: 3,
            seed: 7,
            domain: 9,
        },
    ] {
        let mut values = [1., 2., 3.];
        assert!(apply_inverse_basis(&mut values, basis).is_err());
        assert_eq!(values, [1., 2., 3.]);
    }
}

#[test]
fn gather_reconstructs_every_admitted_scale_law_and_additive_planes() {
    let plane = trits();
    let trits: Vec<_> = plane.into_iter().chain(plane).collect();
    for (anchor, relation, precision, planes, scales, first, second) in [
        (
            ScaleAnchor::Group,
            PlaneRelation::Free,
            ScalePrecision::F32,
            2,
            vec![1., 2., 0.5, 0.25],
            [0.75, 0.75, -0.75, 2.25],
            [1.125, -3.375, 1.125, 1.125],
        ),
        (
            ScaleAnchor::Group,
            PlaneRelation::Free,
            ScalePrecision::F16,
            2,
            vec![1., 2., 0.5, 0.25],
            [0.75, 0.75, -0.75, 2.25],
            [1.125, -3.375, 1.125, 1.125],
        ),
        (
            ScaleAnchor::Group,
            PlaneRelation::Tied { num: 1, den: 3 },
            ScalePrecision::F16,
            2,
            vec![3., 6.],
            [2., 2., -2., 6.],
            [4., -12., 4., 4.],
        ),
        (
            ScaleAnchor::Tensor,
            PlaneRelation::Free,
            ScalePrecision::F32,
            1,
            vec![1.5],
            [0.75, 0.75, -0.75, 2.25],
            [0.75, -2.25, 0.75, 0.75],
        ),
    ] {
        let mut meta = layout(Basis::Hadamard { block: 4 });
        meta.law = ScaleLaw {
            anchor,
            relation,
            precision,
        };
        meta.max_planes = planes;
        let view =
            AdditiveView::new(meta, planes, &trits[..usize::from(planes) * 8], &scales).unwrap();
        let mut out = [0.; 8];
        reference_embed(&view.bind_gather(), &[0, 1], &mut out).unwrap();
        assert_eq!(&out[..4], &first);
        assert_eq!(&out[4..], &second);
    }
}

#[test]
fn gather_rejects_all_invalid_ids_and_lengths_before_writing() {
    let trits = trits();
    let view =
        AdditiveView::new(layout(Basis::Hadamard { block: 4 }), 1, &trits, &[1., 2.]).unwrap();
    let mut out = [99.; 8];
    assert_eq!(
        reference_embed(&view.bind_gather(), &[0, 2], &mut out),
        Err(AdditiveError::RowOutOfRange)
    );
    assert_eq!(out, [99.; 8]);
    assert!(matches!(
        reference_embed(&view.bind_gather(), &[0], &mut out),
        Err(AdditiveError::OutputLength {
            expected: 4,
            got: 8
        })
    ));
    assert_eq!(out, [99.; 8]);
    reference_embed(&view.bind_gather(), &[], &mut []).unwrap();
}

#[test]
fn reference_backend_dense_gather_preserves_order_and_fail_closed_output() {
    let backend = tritium_testkit::ReferenceBackend::new();
    let tensor = backend
        .upload_tensor(TensorView::Dense {
            rows: 2,
            cols: 3,
            values: &[1., 2., 3., 4., 5., 6.],
        })
        .unwrap();
    let mut out = [99.; 9];
    backend.embed_rows(&*tensor, &[1, 0, 1], &mut out).unwrap();
    assert_eq!(out, [4., 5., 6., 1., 2., 3., 4., 5., 6.]);
    out.fill(99.);
    assert!(backend.embed_rows(&*tensor, &[0, 2, 0], &mut out).is_err());
    assert_eq!(out, [99.; 9]);
    assert!(backend.embed_rows(&*tensor, &[0], &mut out).is_err());
    assert_eq!(out, [99.; 9]);
    backend.embed_rows(&*tensor, &[], &mut []).unwrap();

    let zero_width = backend
        .upload_tensor(TensorView::Dense {
            rows: 2,
            cols: 0,
            values: &[],
        })
        .unwrap();
    backend
        .embed_rows(&*zero_width, &[1, 0, 1], &mut [])
        .unwrap();
    assert!(backend.embed_rows(&*zero_width, &[2], &mut []).is_err());
}
