//! Public contract tests for the fallible plan-0049 backend seam.

use tritium_spec::{
    TrainAttributeV1, TrainAttributeValueV1, TrainBufferDataMutV1, TrainBufferDataRefV1,
    TrainExecutionV1, TrainNamedBufferMutV1, TrainNamedBufferRefV1, TrainOutputV1,
    TrainRequestError, TrainRequestV1,
};

#[test]
fn valid_forward_request_accepts_named_f32_buffers_and_attributes() {
    let left = [1.0_f32, 2.0];
    let right = [3.0_f32, 4.0];
    let mut result = [0.0_f32; 2];
    let inputs = [
        TrainNamedBufferRefV1::new("left", &[2], TrainBufferDataRefV1::F32(&left)),
        TrainNamedBufferRefV1::new("right", &[2], TrainBufferDataRefV1::F32(&right)),
    ];
    let attributes = [TrainAttributeV1::new("rows", TrainAttributeValueV1::U64(1))];
    let request = TrainRequestV1::new("graph.add", TrainExecutionV1::Forward, &inputs, &attributes);
    let mut buffers = [TrainNamedBufferMutV1::new(
        "result",
        &[2],
        TrainBufferDataMutV1::F32(&mut result),
    )];
    let output = TrainOutputV1::new(&mut buffers);
    assert_eq!(request.validate(&output), Ok(()));
}

#[test]
fn validation_fails_closed_on_unknown_ops_illegal_phases_and_duplicate_names() {
    let data = [1.0_f32];
    let mut result = [0.0_f32];
    let duplicate_inputs = [
        TrainNamedBufferRefV1::new("x", &[1], TrainBufferDataRefV1::F32(&data)),
        TrainNamedBufferRefV1::new("x", &[1], TrainBufferDataRefV1::F32(&data)),
    ];
    let mut buffers = [TrainNamedBufferMutV1::new(
        "result",
        &[1],
        TrainBufferDataMutV1::F32(&mut result),
    )];
    let output = TrainOutputV1::new(&mut buffers);

    assert!(matches!(
        TrainRequestV1::new(
            "graph.unknown",
            TrainExecutionV1::Forward,
            &duplicate_inputs[..1],
            &[],
        )
        .validate(&output),
        Err(TrainRequestError::UnknownOperation(_))
    ));
    assert!(matches!(
        TrainRequestV1::new(
            "optimizer.adamw",
            TrainExecutionV1::Forward,
            &duplicate_inputs[..1],
            &[],
        )
        .validate(&output),
        Err(TrainRequestError::IllegalExecution { .. })
    ));
    assert!(matches!(
        TrainRequestV1::new(
            "graph.add",
            TrainExecutionV1::Forward,
            &duplicate_inputs,
            &[],
        )
        .validate(&output),
        Err(TrainRequestError::DuplicateName { name, .. }) if name == "x"
    ));
}

#[test]
fn validation_checks_shape_products_lengths_and_attribute_finiteness() {
    let data = [1.0_f32];
    let mut result = [0.0_f32];
    let bad_length = [TrainNamedBufferRefV1::new(
        "x",
        &[2],
        TrainBufferDataRefV1::F32(&data),
    )];
    let mut buffers = [TrainNamedBufferMutV1::new(
        "result",
        &[1],
        TrainBufferDataMutV1::F32(&mut result),
    )];
    let output = TrainOutputV1::new(&mut buffers);
    assert!(matches!(
        TrainRequestV1::new(
            "graph.relu2",
            TrainExecutionV1::Forward,
            &bad_length,
            &[],
        )
        .validate(&output),
        Err(TrainRequestError::BufferLength { name, expected: 2, got: 1 }) if name == "x"
    ));

    let overflow = [TrainNamedBufferRefV1::new(
        "x",
        &[u64::MAX, 2],
        TrainBufferDataRefV1::F32(&data),
    )];
    assert!(matches!(
        TrainRequestV1::new(
            "graph.relu2",
            TrainExecutionV1::Forward,
            &overflow,
            &[],
        )
        .validate(&output),
        Err(TrainRequestError::ShapeOverflow { name }) if name == "x"
    ));

    let attributes = [TrainAttributeV1::new(
        "epsilon",
        TrainAttributeValueV1::F32(f32::NAN),
    )];
    assert!(matches!(
        TrainRequestV1::new(
            "graph.relu2",
            TrainExecutionV1::Forward,
            &overflow[..0],
            &attributes,
        )
        .validate(&output),
        Err(TrainRequestError::NonFiniteAttribute(name)) if name == "epsilon"
    ));
}
