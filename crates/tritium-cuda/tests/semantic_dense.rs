//! Physical CUDA execution through the shared semantic tensor interface.
#![cfg(feature = "cuda")]

use tritium_cuda::CudaBackend;
use tritium_spec::{
    BackendError, TensorExecution, TensorMatmul, TensorUploadPolicy, TensorView, TernaryBackend,
};

fn gpu() -> CudaBackend {
    // This test never self-skips into a green result without a physical device.
    CudaBackend::new(0).expect("semantic dense conformance requires physical CUDA device 0")
}

#[test]
fn dense_gpu_upload_matmul_and_gather_match_frozen_values() {
    let backend = gpu();
    let native = TensorUploadPolicy {
        native_only: true,
        max_payload_bytes: Some(24),
    };
    let mut weights = [1., 2., 3., -1., 0.5, 2.];
    let view = TensorView::Dense {
        rows: 2,
        cols: 3,
        values: &weights,
    };
    let (tensor, caps) = backend.upload_tensor_checked(view, native).unwrap();
    assert_eq!(caps.execution, TensorExecution::Native);
    assert_eq!(caps.payload_bytes, 24);
    assert_eq!(tensor.len_bytes(), 24);
    weights.fill(99.); // Upload owns the original device payload.
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
    backend.embed_rows(&*tensor, &[], &mut []).unwrap();
    rows.fill(99.);
    assert!(backend.embed_rows(&*tensor, &[0, 2, 1], &mut rows).is_err());
    assert_eq!(rows, [99.; 9]);
    out.fill(99.);
    scratch.fill(99.);
    assert!(
        backend
            .matmul(TensorMatmul {
                act: &[1.; 5],
                tensor: &*tensor,
                batch: 2,
                transformed_act: &mut scratch,
                out: &mut out
            })
            .is_err()
    );
    assert_eq!(out, [99.; 4]);
    assert_eq!(scratch, [99.; 6]);
}

#[test]
fn gpu_zero_shapes_budgets_and_foreign_handles_fail_closed() {
    let backend = gpu();
    let other = gpu();
    let native = TensorUploadPolicy {
        native_only: true,
        max_payload_bytes: Some(8),
    };
    let (tensor, _) = backend
        .upload_tensor_checked(
            TensorView::Dense {
                rows: 1,
                cols: 2,
                values: &[1., 2.],
            },
            native,
        )
        .unwrap();
    let mut out = [99.];
    let mut scratch = [99.; 2];
    assert!(
        other
            .matmul(TensorMatmul {
                act: &[1., 1.],
                tensor: &*tensor,
                batch: 1,
                transformed_act: &mut scratch,
                out: &mut out
            })
            .is_err()
    );
    assert_eq!(out, [99.]);
    assert_eq!(scratch, [99.; 2]);
    assert!(other.embed_rows(&*tensor, &[0], &mut scratch).is_err());
    assert_eq!(scratch, [99.; 2]);
    assert!(
        backend
            .matmul(TensorMatmul {
                act: &[],
                tensor: &*tensor,
                batch: usize::MAX,
                transformed_act: &mut [],
                out: &mut [],
            })
            .is_err()
    );
    let cpu = tritium_cpu::CpuBackend::new();
    let foreign = cpu
        .upload_tensor(TensorView::Dense {
            rows: 1,
            cols: 2,
            values: &[1., 2.],
        })
        .unwrap();
    assert!(backend.embed_rows(&*foreign, &[0], &mut scratch).is_err());
    assert_eq!(scratch, [99.; 2]);
    assert!(matches!(
        backend.upload_tensor_checked(
            TensorView::Dense {
                rows: 1,
                cols: 2,
                values: &[1., 2.]
            },
            TensorUploadPolicy {
                max_payload_bytes: Some(7),
                ..native
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
    let zero_policy = TensorUploadPolicy {
        native_only: true,
        max_payload_bytes: Some(0),
    };
    let (zero, caps) = backend
        .upload_tensor_checked(
            TensorView::Dense {
                rows: 2,
                cols: 0,
                values: &[],
            },
            zero_policy,
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
    let (empty, _) = backend
        .upload_tensor_checked(
            TensorView::Dense {
                rows: 0,
                cols: 0,
                values: &[],
            },
            zero_policy,
        )
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
fn gpu_dense_shapes_match_shared_reference_without_self_skip() {
    let backend = gpu();
    // Binary-exact fractional values, ragged/reduction boundaries, multiple
    // activation rows and empty dimensions. This uses the shared core oracle,
    // not a second arithmetic implementation hidden in the test.
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
