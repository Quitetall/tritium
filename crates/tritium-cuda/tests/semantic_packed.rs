//! Physical shared-interface conformance; missing CUDA is a hard failure.
#![cfg(feature = "cuda")]

use tritium_cuda::CudaBackend;
use tritium_spec::{
    BackendError, TensorExecution, TensorMatmul, TensorUploadPolicy, TensorView, TernaryBackend,
};

#[test]
fn physical_packed_uniform_and_compact_vectors_match_shared_reference() {
    let cuda = CudaBackend::new(0).expect("packed semantic conformance requires physical CUDA");
    let uniform =
        tritium_testkit::assert_additive_tensor_conformance(&cuda, TensorExecution::Native);
    let compact =
        tritium_testkit::assert_candidate_per_tile_conformance(&cuda, TensorExecution::Native);
    assert_eq!(uniform, 180);
    assert!(compact > uniform);
    eprintln!("physical CUDA semantic vectors: uniform={uniform}, compact={compact}");
    tritium_testkit::assert_additive_upload_rejections(&cuda);
}

#[test]
fn packed_native_payload_budget_and_stream_ownership_are_checked_before_writes() {
    let cuda = CudaBackend::new(0).expect("packed semantic conformance requires physical CUDA");
    let other = CudaBackend::new(0).expect("owner test requires second stream on physical CUDA");
    let source = tritium_testkit::additive_upload_fixture();
    let view = TensorView::Additive(source.view());
    let caps = cuda.tensor_caps(view).unwrap().unwrap();
    assert_eq!(caps.execution, TensorExecution::Native);
    // This deliberately tiny fixture has more header bytes than data bytes.
    // Physical accounting must include the header, not claim compression here.
    assert_eq!(caps.payload_bytes, 60 + 2 + 8);
    assert!(matches!(
        cuda.upload_tensor_checked(
            view,
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(caps.payload_bytes - 1)
            }
        ),
        Err(BackendError::TensorPayloadBudgetExceeded { .. })
    ));
    let (handle, admitted) = cuda
        .upload_tensor_checked(
            view,
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(caps.payload_bytes),
            },
        )
        .unwrap();
    assert_eq!(handle.len_bytes() as u64, admitted.payload_bytes);
    let cols = source.layout().cols as usize;
    let rows = source.layout().rows as usize;
    drop(source);
    let act = vec![1.0; cols];
    let mut scratch = vec![123.0; cols];
    let mut out = vec![123.0; rows];
    assert!(
        other
            .matmul(TensorMatmul {
                act: &act,
                tensor: &*handle,
                batch: 1,
                transformed_act: &mut scratch,
                out: &mut out
            })
            .is_err()
    );
    assert!(scratch.iter().chain(&out).all(|n| *n == 123.0));
    assert!(other.embed_rows(&*handle, &[0], &mut scratch).is_err());
    assert!(scratch.iter().all(|n| *n == 123.0));
    cuda.matmul(TensorMatmul {
        act: &act,
        tensor: &*handle,
        batch: 1,
        transformed_act: &mut scratch,
        out: &mut out,
    })
    .unwrap();
    assert!(out.iter().all(|n| n.is_finite()));
    out.fill(123.0);
    scratch.fill(123.0);
    for batch in [2, usize::MAX] {
        assert!(
            cuda.matmul(TensorMatmul {
                act: &act,
                tensor: &*handle,
                batch,
                transformed_act: &mut scratch,
                out: &mut out,
            })
            .is_err()
        );
        assert!(scratch.iter().chain(&out).all(|n| *n == 123.0));
    }
    assert!(cuda.embed_rows(&*handle, &[rows], &mut scratch).is_err());
    assert!(scratch.iter().all(|n| *n == 123.0));
    cuda.embed_rows(&*handle, &[], &mut []).unwrap();
}

#[test]
fn physical_packed_payload_is_smaller_than_dense_with_all_metadata_counted() {
    let cuda = CudaBackend::new(0).expect("packed semantic conformance requires physical CUDA");
    let mut layout = tritium_testkit::additive_upload_fixture().layout();
    layout.cols = 132;
    let source = tritium_format::AdditiveTensor::new(
        layout,
        1,
        [
            tritium_core::Trit::POS,
            tritium_core::Trit::NEG,
            tritium_core::Trit::ZERO,
            tritium_core::Trit::POS,
        ]
        .into_iter()
        .cycle()
        .take(264)
        .collect(),
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10.],
    )
    .unwrap();
    let (tensor, caps) = cuda
        .upload_tensor_checked(
            TensorView::Additive(source.view()),
            TensorUploadPolicy {
                native_only: true,
                max_payload_bytes: Some(166),
            },
        )
        .unwrap();
    assert_eq!(caps.payload_bytes, 166); // 60 header + 66 plane + 40 scale.
    assert_eq!(tensor.len_bytes(), 166);
    assert!(caps.payload_bytes * 6 < 264 * 4);
    let mut rows = vec![0.; 264];
    cuda.embed_rows(&*tensor, &[0, 1], &mut rows).unwrap();
    let mut expected = vec![0.; 264];
    tritium_core::reference_embed(&source.view().bind_gather(), &[0, 1], &mut expected).unwrap();
    assert_eq!(rows, expected);
}
