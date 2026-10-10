//! Checked production upload and frozen scalar execution combinations.

use tritium_core::{AdditiveView, Trit};
use tritium_cpu::CpuBackend;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor,
    TensorExecution, TensorUploadPolicy, Transport,
};
use tritium_spec::{BackendError, TensorMatmul, TensorView, TernaryBackend};
use tritium_testkit::ReferenceBackend;

#[test]
fn production_capabilities_execute_admitted_combinations() {
    let cpu = CpuBackend::new();
    let reference = ReferenceBackend::new();
    let native = TensorUploadPolicy {
        native_only: true,
        max_payload_bytes: None,
    };
    let mut cases = 0;
    for backend in [&cpu as &dyn TernaryBackend, &reference] {
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
                            codec: PlaneCodec::D2,
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
                        assert_eq!(caps.execution, TensorExecution::Native);
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
    assert_eq!(cases, 120);
}

#[test]
fn cpu_checked_upload_enforces_budget_and_keeps_unadmitted_groups_unknown() {
    let cpu = CpuBackend::new();
    let empty = TensorView::Dense {
        rows: 0,
        cols: 0,
        values: &[],
    };
    let zero_budget = TensorUploadPolicy {
        native_only: true,
        max_payload_bytes: Some(0),
    };
    assert_eq!(
        cpu.upload_tensor_checked(empty, zero_budget)
            .unwrap()
            .1
            .payload_bytes,
        0
    );
    let dense = TensorView::Dense {
        rows: 1,
        cols: 2,
        values: &[1., 2.],
    };
    let too_small = TensorUploadPolicy {
        native_only: true,
        max_payload_bytes: Some(7),
    };
    assert!(matches!(
        cpu.upload_tensor_checked(dense, too_small),
        Err(BackendError::TensorPayloadBudgetExceeded {
            requested: 8,
            limit: 7
        })
    ));
    let exact = TensorUploadPolicy {
        max_payload_bytes: Some(8),
        ..too_small
    };
    assert_eq!(
        cpu.upload_tensor_checked(dense, exact)
            .unwrap()
            .1
            .payload_bytes,
        8
    );
    for group in [64, 256] {
        let layout = AdditiveLayout {
            rows: 1,
            cols: 4,
            tile: 256,
            group,
            max_planes: 1,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::D2,
            law: ADMITTED_LAWS[1].law,
            basis: Basis::Identity,
            transport: Transport::Raw,
        };
        let view =
            TensorView::Additive(AdditiveView::new(layout, 1, &[Trit::POS; 4], &[1.]).unwrap());
        assert_eq!(cpu.tensor_caps(view).unwrap(), None);
        assert!(matches!(
            cpu.upload_tensor_checked(view, exact),
            Err(BackendError::UnsupportedTensor)
        ));
    }
}

#[test]
fn capability_query_rejects_invalid_dense_and_inexact_scale_inputs() {
    let cpu = CpuBackend::new();
    assert!(
        cpu.tensor_caps(TensorView::Dense {
            rows: usize::MAX,
            cols: 2,
            values: &[]
        })
        .is_err()
    );
    assert!(
        cpu.tensor_caps(TensorView::Dense {
            rows: 1,
            cols: 2,
            values: &[1.]
        })
        .is_err()
    );
    let layout = AdditiveLayout {
        rows: 1,
        cols: 4,
        tile: 256,
        group: 32,
        max_planes: 1,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: ADMITTED_LAWS[0].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    for scales in [[0.1], [-0.]] {
        let view =
            TensorView::Additive(AdditiveView::new(layout, 1, &[Trit::POS; 4], &scales).unwrap());
        assert!(cpu.tensor_caps(view).is_err());
        assert!(
            cpu.upload_tensor_checked(view, TensorUploadPolicy::default())
                .is_err()
        );
    }
}
