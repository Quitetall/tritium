//! Strict physical-adapter tests: absence of a GPU is not a passing result.
#![cfg(feature = "wgpu")]

use tritium_evidence::{
    EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION, verify_jsonl_with_registry,
};
use tritium_runtime::{TensorUploadContext, TensorUploadError, upload_tensor_logged};
use tritium_spec::{
    BackendError, TensorExecution, TensorMatmul, TensorUploadPolicy, TensorView, TernaryBackend,
};
use tritium_wgpu::WgpuBackend;

fn gpu() -> WgpuBackend {
    let backend = WgpuBackend::new().expect("requires a physical Vulkan or Metal adapter");
    let identity = backend.capabilities().device_name.to_lowercase();
    assert!(
        !["llvmpipe", "lavapipe", "swiftshader", "cpu"]
            .iter()
            .any(|software| identity.contains(software)),
        "not a physical GPU: {identity}"
    );
    backend
}

#[test]
fn additive_executes_shared_frozen_conformance_without_self_skip() {
    let backend = gpu();
    assert_eq!(
        tritium_testkit::assert_additive_tensor_conformance(&backend, TensorExecution::Emulated),
        180
    );
    tritium_testkit::assert_additive_upload_rejections(&backend);
}

#[test]
fn dense_device_owned_values_match_frozen_and_shared_core_oracles() {
    let backend = gpu();
    let mut values = [1., 2., 3., -1., 0.5, 2.];
    let (tensor, caps) = backend
        .upload_tensor_checked(
            TensorView::Dense {
                rows: 2,
                cols: 3,
                values: &values,
            },
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(24),
            },
        )
        .unwrap();
    assert_eq!(caps.execution, TensorExecution::Native);
    assert_eq!(caps.payload_bytes, 24);
    assert_eq!(tensor.len_bytes(), 24);
    values.fill(99.);
    let mut out = [99.; 4];
    let mut scratch = [99.; 6];
    backend
        .matmul(TensorMatmul {
            act: &[2., -1., 4., 1., 2., 3.],
            tensor: &*tensor,
            batch: 2,
            transformed_act: &mut scratch,
            out: &mut out,
        })
        .unwrap();
    assert_eq!(out, [12., 5.5, 14., 6.]);
    assert_eq!(scratch, [2., -1., 4., 1., 2., 3.]);
    let mut rows = [99.; 9];
    backend.embed_rows(&*tensor, &[1, 0, 1], &mut rows).unwrap();
    assert_eq!(rows, [-1., 0.5, 2., 1., 2., 3., -1., 0.5, 2.]);
    for (rows, cols, batch) in [
        (1, 1, 1),
        (3, 7, 2),
        (2, 31, 3),
        (4, 33, 2),
        (3, 129, 2),
        (2, 3, 0),
        (0, 3, 2),
        (3, 0, 2),
    ] {
        let weights: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 11) as f32 - 5.) * 0.125)
            .collect();
        let act: Vec<f32> = (0..batch * cols)
            .map(|i| ((i % 7) as f32 - 3.) * 0.25)
            .collect();
        let core = tritium_core::DenseView::new(rows, cols, &weights).unwrap();
        let tensor = backend
            .upload_tensor(TensorView::Dense {
                rows,
                cols,
                values: &weights,
            })
            .unwrap();
        let mut expected = vec![99.; batch * rows];
        let mut core_scratch = vec![99.; act.len()];
        core.matmul(&act, batch, &mut core_scratch, &mut expected)
            .unwrap();
        let mut actual = vec![99.; expected.len()];
        let mut scratch = vec![99.; act.len()];
        backend
            .matmul(TensorMatmul {
                act: &act,
                tensor: &*tensor,
                batch,
                transformed_act: &mut scratch,
                out: &mut actual,
            })
            .unwrap();
        for (&actual, &expected) in actual.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() <= 1e-4,
                "shape {rows}x{cols} batch {batch}: {actual} != {expected}"
            );
        }
        assert_eq!(scratch, core_scratch);
        if rows > 0 {
            let ids = [rows - 1, 0, rows - 1];
            let mut expected = vec![99.; ids.len() * cols];
            let mut actual = vec![99.; expected.len()];
            core.embed(&ids, &mut expected).unwrap();
            backend.embed_rows(&*tensor, &ids, &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn invalid_shapes_foreign_owners_budgets_and_zero_dimensions_fail_closed() {
    let backend = gpu();
    let other = gpu();
    let view = TensorView::Dense {
        rows: 1,
        cols: 2,
        values: &[1., 2.],
    };
    let tensor = backend.upload_tensor(view).unwrap();
    let foreign = tritium_testkit::ReferenceBackend::new()
        .upload_tensor(view)
        .unwrap();
    let mut out = [99.];
    let mut scratch = [99.; 2];
    for (executor, handle, batch, act) in [
        (&other, &*tensor, 1, &[1., 1.][..]),
        (&backend, &*foreign, 1, &[1., 1.][..]),
        (&backend, &*tensor, 1, &[1.][..]),
        (&backend, &*tensor, usize::MAX, &[][..]),
    ] {
        assert!(
            executor
                .matmul(TensorMatmul {
                    act,
                    tensor: handle,
                    batch,
                    transformed_act: &mut scratch,
                    out: &mut out
                })
                .is_err()
        );
        assert_eq!(out, [99.]);
        assert_eq!(scratch, [99.; 2]);
    }
    assert!(other.embed_rows(&*tensor, &[0], &mut scratch).is_err());
    assert!(backend.embed_rows(&*foreign, &[0], &mut scratch).is_err());
    assert!(backend.embed_rows(&*tensor, &[1], &mut scratch).is_err());
    assert_eq!(scratch, [99.; 2]);
    assert!(matches!(
        backend.upload_tensor_checked(
            view,
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(7)
            }
        ),
        Err(BackendError::TensorPayloadBudgetExceeded {
            requested: 8,
            limit: 7
        })
    ));
    for view in [
        TensorView::Dense {
            rows: usize::MAX,
            cols: 2,
            values: &[],
        },
        TensorView::Dense {
            rows: 1,
            cols: 2,
            values: &[1.],
        },
    ] {
        assert!(backend.tensor_caps(view).is_err());
        assert!(backend.upload_tensor(view).is_err());
    }
    let (zero, caps) = backend
        .upload_tensor_checked(
            TensorView::Dense {
                rows: 2,
                cols: 0,
                values: &[],
            },
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(0),
            },
        )
        .unwrap();
    assert_eq!(caps.payload_bytes, 0);
    let mut out = [99.; 4];
    backend
        .matmul(TensorMatmul {
            act: &[],
            tensor: &*zero,
            batch: 2,
            transformed_act: &mut [],
            out: &mut out,
        })
        .unwrap();
    assert_eq!(out, [0.; 4]);
    backend.embed_rows(&*zero, &[1, 0], &mut []).unwrap();
    assert!(backend.embed_rows(&*zero, &[2], &mut []).is_err());
    let empty = backend
        .upload_tensor(TensorView::Dense {
            rows: 0,
            cols: 0,
            values: &[],
        })
        .unwrap();
    backend
        .matmul(TensorMatmul {
            act: &[],
            tensor: &*empty,
            batch: usize::MAX,
            transformed_act: &mut [],
            out: &mut [],
        })
        .unwrap();
}

#[test]
fn emulation_policy_evidence_and_inverse_gather_are_explicit() {
    let backend = gpu();
    let fixture = tritium_testkit::additive_upload_fixture();
    let view = TensorView::Additive(fixture.view());
    let mut recorder =
        EvidenceRecorder::new("wgpu-emulation-fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    for (policy, native) in [
        (
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(32),
            },
            true,
        ),
        (
            TensorUploadPolicy {
                native_only: false,
                max_payload_bytes: Some(31),
            },
            false,
        ),
    ] {
        let result = upload_tensor_logged(
            &backend,
            view,
            policy,
            "embedding",
            TensorUploadContext {
                recorder: &mut recorder,
                span: "load",
                parent: None,
                logical_time: 0,
            },
        );
        if native {
            assert!(matches!(
                result,
                Err(TensorUploadError::Backend(
                    BackendError::NativeTensorRequired
                ))
            ));
        } else {
            assert!(matches!(
                result,
                Err(TensorUploadError::Backend(
                    BackendError::TensorPayloadBudgetExceeded {
                        requested: 32,
                        limit: 31
                    }
                ))
            ));
        }
        assert!(recorder.events().is_empty());
    }
    let policy = TensorUploadPolicy {
        native_only: false,
        max_payload_bytes: Some(32),
    };
    let (tensor, caps) = upload_tensor_logged(
        &backend,
        view,
        policy,
        "embedding",
        TensorUploadContext {
            recorder: &mut recorder,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    assert_eq!(caps.execution, TensorExecution::Emulated);
    assert_eq!(caps.payload_bytes, 32);
    assert_eq!(tensor.len_bytes(), 32);
    let mut replay =
        EvidenceRecorder::new("wgpu-emulation-fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let _replayed = upload_tensor_logged(
        &backend,
        view,
        policy,
        "embedding",
        TensorUploadContext {
            recorder: &mut replay,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    assert_eq!(
        recorder.root_digest().unwrap(),
        replay.root_digest().unwrap()
    );
    let event = &recorder.events()[0];
    assert_eq!(event.payload["caps"]["execution"], "Emulated");
    assert_eq!(event.payload["caps"]["payload_bytes"], 32);
    assert_eq!(event.payload["decoded_source_bytes"], 16);
    assert_eq!(
        event.payload["physical_device"],
        backend.physical_device_id()
    );
    assert_ne!(backend.physical_device_id(), backend.device_id());
    assert!(backend.physical_device_id().contains("wgpu:"));
    let verified = verify_jsonl_with_registry(
        &recorder.to_jsonl().unwrap(),
        [TENSOR_UPLOADED_REGISTRATION],
    )
    .unwrap();
    assert_eq!(verified.root_digest, recorder.root_digest().unwrap());
    drop(fixture);
    let mut out = [99.; 2];
    let mut scratch = [99.; 4];
    backend
        .matmul(TensorMatmul {
            act: &[1.; 4],
            tensor: &*tensor,
            batch: 1,
            transformed_act: &mut scratch,
            out: &mut out,
        })
        .unwrap();
    assert_eq!(out, [-1., -2.]);
    assert_eq!(scratch, [-1., 1., 1., 1.]);
    let mut rows = [99.; 12];
    backend.embed_rows(&*tensor, &[1, 0, 1], &mut rows).unwrap();
    assert_eq!(
        rows,
        [1., -1., 1., -3., 0.5, -0.5, 0.5, -1.5, 1., -1., 1., -3.]
    );
    rows.fill(99.);
    assert!(backend.embed_rows(&*tensor, &[0, 2, 1], &mut rows).is_err());
    assert_eq!(rows, [99.; 12]);
    backend.embed_rows(&*tensor, &[], &mut []).unwrap();
}

#[test]
fn additive_expansion_overflow_rejects_before_device_upload() {
    let backend = gpu();
    let fixture = tritium_testkit::additive_upload_fixture();
    let mut layout = fixture.view().layout();
    layout.max_planes = 3;
    let tensor = tritium_format::AdditiveTensor::new(
        layout,
        3,
        vec![tritium_core::Trit::POS; 24],
        vec![f32::MAX; 6],
    )
    .unwrap();
    assert!(matches!(
        backend.upload_tensor_checked(
            TensorView::Additive(tensor.view()),
            TensorUploadPolicy::default()
        ),
        Err(BackendError::InvalidInput(_))
    ));
}

#[test]
fn shared_device_handles_support_concurrent_matmul_and_gather() {
    let backend = gpu();
    let fixture = tritium_testkit::additive_upload_fixture();
    let tensor = backend
        .upload_tensor(TensorView::Additive(fixture.view()))
        .unwrap();
    std::thread::scope(|scope| {
        for _ in 0..3 {
            scope.spawn(|| {
                for _ in 0..8 {
                    let mut out = [99.; 2];
                    let mut scratch = [99.; 4];
                    backend
                        .matmul(TensorMatmul {
                            act: &[1.; 4],
                            tensor: &*tensor,
                            batch: 1,
                            transformed_act: &mut scratch,
                            out: &mut out,
                        })
                        .unwrap();
                    assert_eq!(out, [-1., -2.]);
                    let mut row = [99.; 4];
                    backend.embed_rows(&*tensor, &[0], &mut row).unwrap();
                    assert_eq!(row, [0.5, -0.5, 0.5, -1.5]);
                }
            });
        }
    });
}
