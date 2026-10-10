//! Physical GPU emulation must execute, not self-skip without a device.
#![cfg(feature = "cuda")]

use tritium_cuda::CudaBackend;
use tritium_evidence::{
    EvidenceRecorder, TENSOR_UPLOADED_REGISTRATION, verify_jsonl_with_registry,
};
use tritium_runtime::{TensorUploadContext, TensorUploadError, upload_tensor_logged};
use tritium_spec::{
    BackendError, TensorExecution, TensorMatmul, TensorUploadPolicy, TensorView, TernaryBackend,
};

#[test]
fn gpu_additive_emulation_executes_shared_frozen_vectors() {
    let backend = CudaBackend::new(0).expect("requires physical CUDA device 0");
    assert_eq!(
        tritium_testkit::assert_additive_tensor_conformance(&backend, TensorExecution::Emulated),
        180
    );
    tritium_testkit::assert_additive_upload_rejections(&backend);
}

#[test]
fn gpu_emulation_policy_evidence_and_owned_basis_are_explicit() {
    let backend = CudaBackend::new(0).expect("requires physical CUDA device 0");
    let fixture = tritium_testkit::additive_upload_fixture();
    let view = TensorView::Additive(fixture.view());
    let mut recorder =
        EvidenceRecorder::new("cuda-emulation-fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
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
        let rejected = upload_tensor_logged(
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
                rejected,
                Err(TensorUploadError::Backend(
                    BackendError::NativeTensorRequired
                ))
            ));
        } else {
            assert!(matches!(
                rejected,
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
    let (tensor, caps) = upload_tensor_logged(
        &backend,
        view,
        TensorUploadPolicy {
            native_only: false,
            max_payload_bytes: Some(32),
        },
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
    // Successful upload observations replay deterministically on this backend.
    let mut replay =
        EvidenceRecorder::new("cuda-emulation-fixture", [TENSOR_UPLOADED_REGISTRATION]).unwrap();
    let _replayed = upload_tensor_logged(
        &backend,
        view,
        TensorUploadPolicy {
            native_only: false,
            max_payload_bytes: Some(32),
        },
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
        replay.root_digest().unwrap(),
        recorder.root_digest().unwrap()
    );
    drop(fixture); // No caller trit/scale payload is needed after device upload.
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
    out.fill(99.);
    scratch.fill(99.);
    assert!(
        backend
            .matmul(TensorMatmul {
                act: &[1.; 3],
                tensor: &*tensor,
                batch: 1,
                transformed_act: &mut scratch,
                out: &mut out
            })
            .is_err()
    );
    assert_eq!(out, [99.; 2]);
    assert_eq!(scratch, [99.; 4]);
    let event = &recorder.events()[0];
    assert_eq!(event.schema, "tritium.runtime.tensor_uploaded");
    assert_eq!(event.payload["caps"]["execution"], "Emulated");
    assert_eq!(event.payload["caps"]["payload_bytes"], 32);
    assert_eq!(event.payload["decoded_source_bytes"], 16);
    assert_eq!(event.payload["backend"], backend.device_id());
    let typed: tritium_evidence::TensorUploaded =
        serde_json::from_value(event.payload.clone()).unwrap();
    assert_eq!(typed.caps, caps);
    let mut malformed = event.payload.clone();
    malformed["caps"]["unknown"] = serde_json::json!(true);
    assert!(serde_json::from_value::<tritium_evidence::TensorUploaded>(malformed).is_err());
    let verified = verify_jsonl_with_registry(
        &recorder.to_jsonl().unwrap(),
        [TENSOR_UPLOADED_REGISTRATION],
    )
    .unwrap();
    assert_eq!(verified.root_digest, recorder.root_digest().unwrap());
}

#[test]
fn gpu_additive_expansion_overflow_is_rejected_before_device_upload() {
    let backend = CudaBackend::new(0).expect("requires physical CUDA device 0");
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
