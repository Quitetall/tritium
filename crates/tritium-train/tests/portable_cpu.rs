//! CPU adapter tracer vectors through the public TrainBackendV1 seam.

use tritium_spec::{
    TrainAttributeV1, TrainAttributeValueV1, TrainBackendError, TrainBackendV1,
    TrainBufferDataMutV1, TrainBufferDataRefV1, TrainDTypeV1, TrainExecutionV1,
    TrainNamedBufferMutV1, TrainNamedBufferRefV1, TrainOutputV1, TrainRequestV1,
    TrainingOpManifestV1,
};
use tritium_train::CpuTrainBackendV1;

#[test]
fn cpu_add_forward_and_vjp_match_literal_vectors_and_emit_receipts() {
    let backend = CpuTrainBackendV1::new();
    let capabilities = backend.capabilities();
    assert_eq!(capabilities.dtypes, [TrainDTypeV1::F32, TrainDTypeV1::U32]);
    assert_eq!(
        capabilities.supported_operations,
        [
            "graph.transpose",
            "graph.embedding_gather",
            "graph.slice_cols",
            "graph.concat_cols",
            "graph.detach",
            "graph.scale_const",
            "graph.bias",
            "graph.add",
            "graph.mul",
            "graph.relu2",
            "graph.silu",
            "loss.mse",
            "optimizer.sgd",
        ]
    );
    assert_eq!(capabilities.manifest_digest, TrainingOpManifestV1::digest());

    let left = [1.0_f32, -2.0, 0.5];
    let right = [3.0_f32, 4.0, -1.5];
    let inputs = [
        TrainNamedBufferRefV1::new("left", &[3], TrainBufferDataRefV1::F32(&left)),
        TrainNamedBufferRefV1::new("right", &[3], TrainBufferDataRefV1::F32(&right)),
    ];
    let request = TrainRequestV1::new("graph.add", TrainExecutionV1::Forward, &inputs, &[]);
    let mut result = [0.0_f32; 3];
    let mut buffers = [TrainNamedBufferMutV1::new(
        "result",
        &[3],
        TrainBufferDataMutV1::F32(&mut result),
    )];
    let mut output = TrainOutputV1::new(&mut buffers);
    let receipt = backend.execute(request, &mut output).unwrap();
    assert_eq!(result, [4.0, 2.0, -1.0]);
    assert_eq!(receipt.operation, "graph.add");
    assert_eq!(receipt.execution, TrainExecutionV1::Forward);
    let build_digest = receipt
        .backend_build
        .strip_prefix(concat!(
            "tritium-train@",
            env!("CARGO_PKG_VERSION"),
            "+source-blake3:"
        ))
        .unwrap();
    assert_eq!(build_digest.len(), 64);
    assert!(build_digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    assert_eq!(receipt.physical_device, None);
    assert_eq!(receipt.vector_digest, None);
    assert_eq!(
        receipt.input_digest,
        [
            82, 83, 162, 16, 153, 77, 48, 152, 28, 138, 66, 39, 163, 176, 131, 161, 245, 101, 60,
            30, 201, 216, 245, 144, 163, 55, 42, 70, 150, 39, 153, 160,
        ]
    );
    assert_eq!(
        receipt.output_digest,
        [
            30, 138, 229, 175, 36, 160, 192, 254, 105, 221, 118, 2, 39, 115, 132, 122, 145, 90, 76,
            153, 179, 6, 175, 154, 208, 5, 233, 204, 41, 27, 247, 103,
        ]
    );
    assert_eq!(receipt.scratch_bytes, 0);
    assert_eq!(receipt.peak_resident_bytes, 36);
    assert_eq!(receipt.host_transfers, 0);
    assert!(receipt.device_resident);

    let grad_output = [0.25_f32, -0.5, 2.0];
    let inputs = [TrainNamedBufferRefV1::new(
        "grad_output",
        &[3],
        TrainBufferDataRefV1::F32(&grad_output),
    )];
    let request = TrainRequestV1::new("graph.add", TrainExecutionV1::Vjp, &inputs, &[]);
    let mut grad_left = [0.0_f32; 3];
    let mut grad_right = [0.0_f32; 3];
    let mut buffers = [
        TrainNamedBufferMutV1::new("grad_left", &[3], TrainBufferDataMutV1::F32(&mut grad_left)),
        TrainNamedBufferMutV1::new(
            "grad_right",
            &[3],
            TrainBufferDataMutV1::F32(&mut grad_right),
        ),
    ];
    let mut output = TrainOutputV1::new(&mut buffers);
    backend.execute(request, &mut output).unwrap();
    assert_eq!(grad_left, grad_output);
    assert_eq!(grad_right, grad_output);
}

#[test]
fn cpu_sgd_step_matches_portable_literal() {
    let backend = CpuTrainBackendV1::new();
    let parameter = [1.0_f32, -2.0];
    let gradient = [0.5_f32, -0.25];
    let inputs = [
        TrainNamedBufferRefV1::new("parameter", &[2], TrainBufferDataRefV1::F32(&parameter)),
        TrainNamedBufferRefV1::new("gradient", &[2], TrainBufferDataRefV1::F32(&gradient)),
    ];
    let attributes = [
        TrainAttributeV1::new("step", TrainAttributeValueV1::U64(1)),
        TrainAttributeV1::new("lr", TrainAttributeValueV1::F32(0.1)),
    ];
    let request = TrainRequestV1::new(
        "optimizer.sgd",
        TrainExecutionV1::Step,
        &inputs,
        &attributes,
    );
    let mut updated = [0.0_f32; 2];
    let mut buffers = [TrainNamedBufferMutV1::new(
        "parameter",
        &[2],
        TrainBufferDataMutV1::F32(&mut updated),
    )];
    let mut output = TrainOutputV1::new(&mut buffers);
    backend.execute(request, &mut output).unwrap();
    assert_eq!(updated, [0.95, -1.975]);
}

#[test]
fn cpu_mse_vjp_preserves_reference_f32_evaluation_order() {
    let backend = CpuTrainBackendV1::new();
    let prediction = [f32::from_bits(0x4fc3_5373); 257];
    let target = [0.0_f32; 257];
    let grad_output = [f32::from_bits(0xceb6_cdd2)];
    let inputs = [
        TrainNamedBufferRefV1::new("prediction", &[257], TrainBufferDataRefV1::F32(&prediction)),
        TrainNamedBufferRefV1::new("target", &[257], TrainBufferDataRefV1::F32(&target)),
        TrainNamedBufferRefV1::new("grad_output", &[], TrainBufferDataRefV1::F32(&grad_output)),
    ];
    let request = TrainRequestV1::new("loss.mse", TrainExecutionV1::Vjp, &inputs, &[]);
    let mut gradient = [0.0_f32; 257];
    let mut buffers = [TrainNamedBufferMutV1::new(
        "grad_prediction",
        &[257],
        TrainBufferDataMutV1::F32(&mut gradient),
    )];
    let mut output = TrainOutputV1::new(&mut buffers);
    backend.execute(request, &mut output).unwrap();
    assert!(gradient.iter().all(|value| value.to_bits() == 0xdb8a_ef6f));
}

#[test]
fn cpu_adapter_rejects_unsupported_or_malformed_requests_before_mutation() {
    let backend = CpuTrainBackendV1::new();
    let input = [1.0_f32];
    let inputs = [TrainNamedBufferRefV1::new(
        "x",
        &[1],
        TrainBufferDataRefV1::F32(&input),
    )];
    let request = TrainRequestV1::new("graph.rmsnorm", TrainExecutionV1::Forward, &inputs, &[]);
    let mut sentinel = [123.0_f32];
    let mut buffers = [TrainNamedBufferMutV1::new(
        "result",
        &[1],
        TrainBufferDataMutV1::F32(&mut sentinel),
    )];
    let mut output = TrainOutputV1::new(&mut buffers);
    assert!(matches!(
        backend.execute(request, &mut output),
        Err(TrainBackendError::UnsupportedOperation(operation)) if operation == "graph.rmsnorm"
    ));
    assert_eq!(sentinel, [123.0]);

    let left = [1.0_f32];
    let right = [2.0_f32];
    let reordered = [
        TrainNamedBufferRefV1::new("right", &[1], TrainBufferDataRefV1::F32(&right)),
        TrainNamedBufferRefV1::new("left", &[1], TrainBufferDataRefV1::F32(&left)),
    ];
    let request = TrainRequestV1::new("graph.add", TrainExecutionV1::Forward, &reordered, &[]);
    let mut sentinel = [456.0_f32];
    let mut buffers = [TrainNamedBufferMutV1::new(
        "result",
        &[1],
        TrainBufferDataMutV1::F32(&mut sentinel),
    )];
    let mut output = TrainOutputV1::new(&mut buffers);
    assert!(matches!(
        backend.execute(request, &mut output),
        Err(TrainBackendError::InvalidOperation(_))
    ));
    assert_eq!(sentinel, [456.0]);

    let grad_output = [7.0_f32];
    let inputs = [TrainNamedBufferRefV1::new(
        "grad_output",
        &[1],
        TrainBufferDataRefV1::F32(&grad_output),
    )];
    let request = TrainRequestV1::new("graph.add", TrainExecutionV1::Vjp, &inputs, &[]);
    let mut grad_left = [789.0_f32];
    let mut wrong_grad_right = [321_u32];
    let mut buffers = [
        TrainNamedBufferMutV1::new("grad_left", &[1], TrainBufferDataMutV1::F32(&mut grad_left)),
        TrainNamedBufferMutV1::new(
            "grad_right",
            &[1],
            TrainBufferDataMutV1::U32(&mut wrong_grad_right),
        ),
    ];
    let mut output = TrainOutputV1::new(&mut buffers);
    assert!(matches!(
        backend.execute(request, &mut output),
        Err(TrainBackendError::InvalidOperation(_))
    ));
    assert_eq!(grad_left, [789.0]);
    assert_eq!(wrong_grad_right, [321]);
}
