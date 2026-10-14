//! Pure structural planning. No GPU context, transfer or physical qualification.
#![cfg(feature = "cuda")]
use tritium_cuda::CudaBackend;
use tritium_spec::{AdditiveMetadata, TensorExecution, TensorMetadata};

#[test]
fn cuda_payload_plans_use_the_actual_header_and_explicit_host_staging_without_init() {
    let source = tritium_testkit::additive_upload_fixture();
    let view = source.view();
    let metadata = TensorMetadata::Additive(
        AdditiveMetadata::uniform(source.layout(), view.plane_count() as u8).unwrap(),
    );
    let plan = CudaBackend::plan_tensor_metadata(metadata)
        .unwrap()
        .unwrap();
    assert_eq!(plan.caps.execution, TensorExecution::Native);
    assert_eq!(plan.caps.payload_bytes, 60 + 2 + 8);
    assert_eq!(
        plan.host_workspace_bytes,
        (60 + 2 + 8).max(source.layout().cols * 4)
    );
    assert_eq!(plan.device_workspace_bytes, 0);
    let dense = CudaBackend::plan_tensor_metadata(TensorMetadata::Dense { rows: 7, cols: 32 })
        .unwrap()
        .unwrap();
    assert_eq!(dense.caps.payload_bytes, 7 * 32 * 4);
    assert_eq!(dense.host_workspace_bytes, 0);
    assert!(
        CudaBackend::plan_tensor_metadata(TensorMetadata::Dense {
            rows: usize::MAX,
            cols: 2
        })
        .is_err()
    );
}

#[test]
fn large_geometry_is_planned_without_coefficients_or_runtime_indices() {
    let source = tritium_testkit::additive_upload_fixture();
    let mut layout = source.layout();
    layout.rows = 248320;
    layout.cols = 5120;
    layout.max_planes = 2;
    let plan = CudaBackend::plan_tensor_metadata(TensorMetadata::Additive(
        AdditiveMetadata::uniform(layout, 2).unwrap(),
    ))
    .unwrap()
    .unwrap();
    assert!(plan.caps.payload_bytes > 500_000_000);
    assert!(plan.host_workspace_bytes >= plan.caps.payload_bytes);
    // This is a structural software test, not observed allocation/VRAM or support.
}

#[test]
fn encoded_stream_plans_account_for_native_staging_and_bounded_scratch_without_init() {
    let source = tritium_testkit::additive_upload_fixture();
    let metadata = TensorMetadata::Additive(AdditiveMetadata::uniform(source.layout(), 1).unwrap());
    let plan = CudaBackend::plan_tensor_stream(
        metadata,
        tritium_spec::TensorStreamEncoding::CanonicalRawAdditive,
    )
    .unwrap()
    .unwrap();
    assert_eq!(plan.upload.caps.payload_bytes, 70);
    assert_eq!(plan.encoded_bytes, 10);
    assert_eq!(plan.upload.host_workspace_bytes, 70 + 65536);
    assert_eq!(plan.upload.device_workspace_bytes, 0);
    for precision in [
        tritium_schema::ScalePrecision::F16,
        tritium_schema::ScalePrecision::Bf16,
        tritium_schema::ScalePrecision::F32,
    ] {
        let plan = CudaBackend::plan_tensor_stream(
            TensorMetadata::Dense { rows: 7, cols: 32 },
            tritium_spec::TensorStreamEncoding::DenseLittleEndian(precision),
        )
        .unwrap()
        .unwrap();
        assert_eq!(plan.upload.caps.payload_bytes, 7 * 32 * 4);
        assert_eq!(plan.upload.host_workspace_bytes, 7 * 32 * 4 + 65536);
        assert_eq!(
            plan.encoded_bytes,
            7 * 32
                * if precision == tritium_schema::ScalePrecision::F32 {
                    4
                } else {
                    2
                }
        );
    }
    assert!(
        CudaBackend::plan_tensor_stream(
            metadata,
            tritium_spec::TensorStreamEncoding::DenseLittleEndian(
                tritium_schema::ScalePrecision::F32
            )
        )
        .unwrap()
        .is_none()
    );
}
