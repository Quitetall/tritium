//! Shared frozen semantic vectors; no backend-specific arithmetic oracle.

use tritium_core::{AdditiveView, Trit};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor,
    TensorExecution, TensorUploadPolicy, Transport,
};
use tritium_spec::{TensorMatmul, TensorView, TernaryBackend};

/// Small owned SignedRht/F32 fixture for upload policy, evidence and ownership.
///
/// Logical rows are `[0.5,-0.5,0.5,-1.5]` and twice that row. Decoded scalar
/// source bytes are 16; dense f32 materialization is 32 bytes.
pub fn additive_upload_fixture() -> tritium_format::AdditiveTensor {
    tritium_format::AdditiveTensor::new(
        AdditiveLayout {
            rows: 2,
            cols: 4,
            tile: 256,
            group: 32,
            max_planes: 1,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::D2,
            law: ADMITTED_LAWS[1].law,
            basis: Basis::SignedRht {
                block: 4,
                seed: 7,
                domain: 9,
            },
            transport: Transport::Raw,
        },
        1,
        [Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS]
            .into_iter()
            .cycle()
            .take(8)
            .collect(),
        vec![1., 2.],
    )
    .unwrap()
}

/// Assert rejection of unadmitted groups and noncanonical stored scales.
///
/// These checks exercise checked upload, not the legacy policy-unchecked path.
pub fn assert_additive_upload_rejections(backend: &dyn TernaryBackend) {
    let mut layout = additive_upload_fixture().view().layout();
    layout.rows = 1;
    layout.basis = Basis::Identity;
    for group in [64, 256] {
        layout.group = group;
        let view =
            TensorView::Additive(AdditiveView::new(layout, 1, &[Trit::POS; 4], &[1.]).unwrap());
        assert_eq!(backend.tensor_caps(view).unwrap(), None);
        assert!(matches!(
            backend.upload_tensor_checked(view, TensorUploadPolicy::default()),
            Err(tritium_spec::BackendError::UnsupportedTensor)
        ));
    }
    layout.group = 32;
    layout.law = ADMITTED_LAWS[0].law;
    for scales in [[0.1], [-0.]] {
        let view =
            TensorView::Additive(AdditiveView::new(layout, 1, &[Trit::POS; 4], &scales).unwrap());
        assert!(backend.tensor_caps(view).is_err());
        assert!(
            backend
                .upload_tensor_checked(view, TensorUploadPolicy::default())
                .is_err()
        );
    }
}

/// Execute frozen additive upload/matmul/gather vectors through a backend.
///
/// Covers every current law/plane count, G32/G128, three bases and three codecs.
/// Panics on unsupported capability or a frozen numerical/policy mismatch;
/// returns the number of combinations, not empirical release approval.
pub fn assert_additive_tensor_conformance(
    backend: &dyn TernaryBackend,
    execution: TensorExecution,
) -> usize {
    let native = TensorUploadPolicy {
        native_only: execution == TensorExecution::Native,
        max_payload_bytes: None,
    };
    let mut cases = 0;
    for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
        for admitted in ADMITTED_LAWS {
            for group in [32_u16, 128] {
                for planes in 1..=admitted.max_planes {
                    for (basis, logical) in [
                        (Basis::Identity, [1., -1., 0., 1.]),
                        (Basis::Hadamard { block: 4 }, [0.5, 0.5, -0.5, 1.5]),
                        (
                            Basis::SignedRht {
                                block: 4,
                                seed: 7,
                                domain: 9,
                            },
                            [0.5, 0.5, -0.5, 1.5],
                        ),
                    ] {
                        let cols = 2 * usize::from(group) + 4;
                        let layout = AdditiveLayout {
                            rows: 2,
                            cols: cols as u64,
                            tile: 256,
                            group,
                            max_planes: planes,
                            allocation: PlaneAllocation::Uniform,
                            codec,
                            law: admitted.law,
                            basis,
                            transport: Transport::Raw,
                        };
                        let trits: Vec<_> = [Trit::POS, Trit::NEG, Trit::ZERO, Trit::POS]
                            .into_iter()
                            .cycle()
                            .take(2 * cols * usize::from(planes))
                            .collect();
                        let units: &[f32] = if admitted.law.anchor == ScaleAnchor::Tensor {
                            &[1.]
                        } else {
                            &[1., 2., 3., 4., 5., 6.]
                        };
                        let scales: Vec<_> = match admitted.law.relation {
                            PlaneRelation::Free => (0..planes)
                                .flat_map(|p| {
                                    units
                                        .iter()
                                        .map(move |s| s * [1., 0.5, 0.25][usize::from(p)])
                                })
                                .collect(),
                            PlaneRelation::Tied { .. } => units.to_vec(),
                            _ => panic!("new relation needs frozen vectors"),
                        };
                        let plane_sum = match admitted.law.relation {
                            PlaneRelation::Free => [1., 1.5, 1.75][usize::from(planes - 1)],
                            PlaneRelation::Tied { .. } => {
                                [1., 4. / 3., 13. / 9.][usize::from(planes - 1)]
                            }
                            _ => unreachable!(),
                        };
                        let view = TensorView::Additive(
                            AdditiveView::new(layout, planes, &trits, &scales).unwrap(),
                        );
                        let caps = backend.tensor_caps(view).unwrap().unwrap();
                        assert_eq!(caps.execution, execution);
                        if execution == TensorExecution::Emulated {
                            assert_eq!(caps.payload_bytes, (2 * cols * 4) as u64);
                        }
                        let (uploaded, reported) =
                            backend.upload_tensor_checked(view, native).unwrap();
                        assert_eq!(reported, caps);
                        assert_eq!(reported.payload_bytes, uploaded.len_bytes() as u64);
                        // Frozen negative-sign bits for seed=7/domain=9, indexed
                        // across the input axis (not repeated independently per block).
                        let negative = [
                            0x2e11eedbb6181d8e_u64,
                            0xa340545129c2dc12,
                            0xc63f1323db21c779,
                            0x04eab3cb584fb6a1,
                            0xa,
                        ];
                        let mut expected_rows = vec![0.; 2 * cols];
                        for row in 0..2 {
                            for k in 0..cols {
                                let unit = if units.len() == 1 {
                                    1.
                                } else {
                                    units[row * 3 + k / usize::from(group)]
                                };
                                let sign = if matches!(basis, Basis::SignedRht { .. })
                                    && (negative[k / 64] >> (k % 64)) & 1 != 0
                                {
                                    -1.
                                } else {
                                    1.
                                };
                                expected_rows[row * cols + k] =
                                    unit * plane_sum * logical[k % 4] * sign;
                            }
                        }
                        let mut rows = vec![0.; 2 * cols];
                        backend.embed_rows(&*uploaded, &[0, 1], &mut rows).unwrap();
                        for (actual, expected) in rows.iter().zip(&expected_rows) {
                            assert!(
                                (actual - expected).abs() <= 2e-6,
                                "{} {layout:?} planes={planes}: actual={actual} expected={expected}",
                                backend.device_id()
                            );
                        }
                        let act = vec![1.; cols];
                        let mut scratch = vec![0.; cols];
                        let mut out = [0.; 2];
                        backend
                            .matmul(TensorMatmul {
                                act: &act,
                                tensor: &*uploaded,
                                batch: 1,
                                transformed_act: &mut scratch,
                                out: &mut out,
                            })
                            .unwrap();
                        for row in 0..2 {
                            let expected = expected_rows[row * cols..(row + 1) * cols]
                                .iter()
                                .map(|&value| f64::from(value))
                                .sum::<f64>() as f32;
                            assert!(
                                (out[row] - expected).abs() < 1e-3,
                                "{} {layout:?} planes={planes}: actual={} expected={expected}",
                                backend.device_id(),
                                out[row]
                            );
                        }
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 180);
    cases
}
