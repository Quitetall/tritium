//! Regenerate the checked-in plan-0049 portable-training tracer corpus.
//!
//! Run deliberately when widening semantic coverage:
//!
//! ```text
//! cargo run -p tritium-train --example freeze_training_vectors
//! ```

use std::path::Path;

use serde::Serialize;
use tritium_spec::{TrainingOpManifestV1, TrainingVectorSetV1};
use tritium_train::{
    Optimizer, Sgd,
    ops::{act, bias, dense, elementwise, embed, loss, shape},
};

#[derive(Serialize)]
struct Corpus {
    schema_id: &'static str,
    schema_version: u32,
    manifest_digest: String,
    cases: Vec<Case>,
}

#[derive(Serialize)]
struct Case {
    case_id: &'static str,
    operation: &'static str,
    execution: &'static str,
    tolerance: Tolerance,
    inputs: Vec<Buffer>,
    attributes: Vec<Attribute>,
    expected: Expected,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Tolerance {
    BitExact,
    AbsoluteRelative {
        absolute_bits: u32,
        relative_bits: u32,
    },
}

#[derive(Serialize)]
struct Buffer {
    name: &'static str,
    shape: Vec<u64>,
    data: Data,
}

#[derive(Serialize)]
#[serde(tag = "dtype", rename_all = "snake_case")]
enum Data {
    F32 { bits: Vec<u32> },
    U32 { values: Vec<u32> },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Attribute {
    F32 {
        name: &'static str,
        bits: u32,
    },
    U64 {
        name: &'static str,
        value: u64,
    },
    U64List {
        name: &'static str,
        values: Vec<u64>,
    },
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Expected {
    Success {
        outputs: Vec<Buffer>,
        scratch_bytes_max: u64,
    },
    Error {
        category: &'static str,
        code: &'static str,
        outputs: Vec<Buffer>,
    },
}

fn f32_buffer(name: &'static str, shape: &[u64], values: &[f32]) -> Buffer {
    Buffer {
        name,
        shape: shape.to_vec(),
        data: Data::F32 {
            bits: values.iter().map(|value| value.to_bits()).collect(),
        },
    }
}

fn u32_buffer(name: &'static str, shape: &[u64], values: &[u32]) -> Buffer {
    Buffer {
        name,
        shape: shape.to_vec(),
        data: Data::U32 {
            values: values.to_vec(),
        },
    }
}

fn main() {
    let left = [1.0_f32, -2.0, 0.5];
    let right = [3.0_f32, 4.0, -1.5];
    let add: Vec<_> = left.iter().zip(right).map(|(&x, y)| x + y).collect();
    let grad_output = [0.25_f32, -0.5, 2.0];

    let mul_left = [2.0_f32, -3.0, 0.0];
    let mul_right = [-4.0_f32, 0.5, 7.0];
    let mul = elementwise::mul_forward(&mul_left, &mul_right);
    let mul_grad_output = [0.25_f32, -2.0, 3.0];
    let mul_grads = elementwise::mul_vjp(&mul_left, &mul_right, &mul_grad_output);

    let matrix = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let transpose = dense::transpose_forward(&matrix, 2, 3);
    let transpose_grad_output = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let transpose_grad = dense::transpose_vjp(2, 3, &transpose_grad_output);

    let embedding_weight = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let tokens = [2_u32, 0, 2];
    let embedding = embed::gather_forward(&embedding_weight, &tokens, 2);
    let embedding_grad_output = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let embedding_grad = embed::gather_vjp(4, &tokens, 2, &embedding_grad_output);

    let column_matrix = [0.0_f32, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
    let column_slice = shape::slice_cols_forward(&column_matrix, 2, 4, 1, 2);
    let slice_grad_output = [1.0_f32, 2.0, 3.0, 4.0];
    let slice_grad = shape::slice_cols_vjp(2, 4, 1, 2, &slice_grad_output);

    let concat_left = [1.0_f32, 2.0, 3.0, 4.0];
    let concat_right = [5.0_f32, 6.0];
    let concatenated = shape::concat_cols_forward(&[&concat_left, &concat_right], 2, &[2, 1]);
    let concat_grad_output = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let concat_grads = shape::concat_cols_vjp(2, &[2, 1], &concat_grad_output);

    let unary_input = [-2.0_f32, 0.0, 3.0];
    let unary_grad_output = [1.0_f32, 2.0, 0.5];
    let scale = 0.25_f32;
    let scaled: Vec<_> = unary_input.iter().map(|value| value * scale).collect();
    let scaled_grad: Vec<_> = unary_grad_output
        .iter()
        .map(|value| value * scale)
        .collect();
    let relu2 = act::relu2_forward(&unary_input);
    let relu2_grad = act::relu2_vjp(&unary_input, &unary_grad_output);

    let silu_input = [-1.0_f32, 0.0, 2.0];
    let silu_grad_output = [0.5_f32, -1.0, 2.0];
    let silu = act::silu_forward(&silu_input);
    let silu_grad = act::silu_vjp(&silu_input, &silu_grad_output);

    let bias_input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias_value = [0.5_f32, -1.0, 2.0];
    let bias_grad_output = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bias_result = bias::forward(&bias_input, &bias_value, 2, 3);
    let bias_grads = bias::vjp(&bias_input, &bias_value, 2, 3, &bias_grad_output);

    let prediction = [1.0_f32, -1.0, 2.0];
    let target = [0.0_f32, 1.0, 2.5];
    let loss_grad_output = [0.5_f32];
    let mse = loss::mse_forward(&prediction, &target);
    let mse_grad = loss::mse_vjp(&prediction, &target, &loss_grad_output);

    let parameter = [1.0_f32, -2.0];
    let gradient = [0.5_f32, -0.25];
    let optimizer = Sgd::new(0.1);
    let mut updated = parameter;
    let mut state = optimizer.init_state(updated.len());
    optimizer.step(1, &mut updated, &gradient, &mut state);

    let corpus = Corpus {
        schema_id: TrainingVectorSetV1::SCHEMA_ID,
        schema_version: TrainingVectorSetV1::SCHEMA_VERSION,
        manifest_digest: hex(&TrainingOpManifestV1::digest()),
        cases: vec![
            Case {
                case_id: "graph.add.forward.basic",
                operation: "graph.add",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("left", &[3], &left),
                    f32_buffer("right", &[3], &right),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &add)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.add.forward.zero",
                operation: "graph.add",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-4_f32.to_bits(),
                    relative_bits: 1.0e-4_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("left", &[1], &[0.0]),
                    f32_buffer("right", &[1], &[0.0]),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[1], &[0.0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.add.vjp.basic",
                operation: "graph.add",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[3], &grad_output)],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_left", &[3], &grad_output),
                        f32_buffer("grad_right", &[3], &grad_output),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.transpose.forward.basic",
                operation: "graph.transpose",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[2, 3], &matrix)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3, 2], &transpose)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.transpose.vjp.basic",
                operation: "graph.transpose",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[3, 2], &transpose_grad_output)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[2, 3], &transpose_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.embedding_gather.forward.repeated",
                operation: "graph.embedding_gather",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[4, 2], &embedding_weight),
                    u32_buffer("tokens", &[3], &tokens),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "vocab",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "n_embd",
                        value: 2,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3, 2], &embedding)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.embedding_gather.vjp.repeated",
                operation: "graph.embedding_gather",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[4, 2], &embedding_weight),
                    u32_buffer("tokens", &[3], &tokens),
                    f32_buffer("grad_output", &[3, 2], &embedding_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "vocab",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "n_embd",
                        value: 2,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_weight", &[4, 2], &embedding_grad)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.slice_cols.forward.basic",
                operation: "graph.slice_cols",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[2, 4], &column_matrix)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "start",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "len",
                        value: 2,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 2], &column_slice)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.slice_cols.vjp.basic",
                operation: "graph.slice_cols",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[2, 2], &slice_grad_output)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "start",
                        value: 1,
                    },
                    Attribute::U64 {
                        name: "len",
                        value: 2,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[2, 4], &slice_grad)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.concat_cols.forward.basic",
                operation: "graph.concat_cols",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("part.0", &[2, 2], &concat_left),
                    f32_buffer("part.1", &[2, 1], &concat_right),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64List {
                        name: "lens",
                        values: vec![2, 1],
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 3], &concatenated)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.concat_cols.vjp.basic",
                operation: "graph.concat_cols",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[2, 3], &concat_grad_output)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64List {
                        name: "lens",
                        values: vec![2, 1],
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_part.0", &[2, 2], &concat_grads[0]),
                        f32_buffer("grad_part.1", &[2, 1], &concat_grads[1]),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.mul.forward.basic",
                operation: "graph.mul",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("left", &[3], &mul_left),
                    f32_buffer("right", &[3], &mul_right),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &mul)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.mul.vjp.basic",
                operation: "graph.mul",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("left", &[3], &mul_left),
                    f32_buffer("right", &[3], &mul_right),
                    f32_buffer("grad_output", &[3], &mul_grad_output),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_left", &[3], &mul_grads[0]),
                        f32_buffer("grad_right", &[3], &mul_grads[1]),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.detach.forward.basic",
                operation: "graph.detach",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[3], &unary_input)],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &unary_input)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.detach.vjp.zero",
                operation: "graph.detach",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[3], &unary_grad_output)],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[3], &[0.0; 3])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.scale_const.forward.basic",
                operation: "graph.scale_const",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[3], &unary_input)],
                attributes: vec![Attribute::F32 {
                    name: "scale",
                    bits: scale.to_bits(),
                }],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &scaled)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.scale_const.vjp.basic",
                operation: "graph.scale_const",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("grad_output", &[3], &unary_grad_output)],
                attributes: vec![Attribute::F32 {
                    name: "scale",
                    bits: scale.to_bits(),
                }],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[3], &scaled_grad)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.bias.forward.basic",
                operation: "graph.bias",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[2, 3], &bias_input),
                    f32_buffer("bias", &[3], &bias_value),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[2, 3], &bias_result)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.bias.vjp.basic",
                operation: "graph.bias",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[2, 3], &bias_input),
                    f32_buffer("bias", &[3], &bias_value),
                    f32_buffer("grad_output", &[2, 3], &bias_grad_output),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![
                        f32_buffer("grad_x", &[2, 3], &bias_grads[0]),
                        f32_buffer("grad_bias", &[3], &bias_grads[1]),
                    ],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.relu2.forward.basic",
                operation: "graph.relu2",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[3], &unary_input)],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &relu2)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.relu2.vjp.basic",
                operation: "graph.relu2",
                execution: "vjp",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("x", &[3], &unary_input),
                    f32_buffer("grad_output", &[3], &unary_grad_output),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[3], &relu2_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.silu.forward.basic",
                operation: "graph.silu",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![f32_buffer("x", &[3], &silu_input)],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[3], &silu)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.silu.vjp.basic",
                operation: "graph.silu",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-5_f32.to_bits(),
                    relative_bits: 1.0e-5_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("x", &[3], &silu_input),
                    f32_buffer("grad_output", &[3], &silu_grad_output),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_x", &[3], &silu_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "loss.mse.forward.basic",
                operation: "loss.mse",
                execution: "forward",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("prediction", &[3], &prediction),
                    f32_buffer("target", &[3], &target),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("result", &[], &mse)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "loss.mse.vjp.basic",
                operation: "loss.mse",
                execution: "vjp",
                tolerance: Tolerance::AbsoluteRelative {
                    absolute_bits: 1.0e-6_f32.to_bits(),
                    relative_bits: 1.0e-6_f32.to_bits(),
                },
                inputs: vec![
                    f32_buffer("prediction", &[3], &prediction),
                    f32_buffer("target", &[3], &target),
                    f32_buffer("grad_output", &[], &loss_grad_output),
                ],
                attributes: vec![],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("grad_prediction", &[3], &mse_grad[0])],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "optimizer.sgd.step.basic",
                operation: "optimizer.sgd",
                execution: "step",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("parameter", &[2], &parameter),
                    f32_buffer("gradient", &[2], &gradient),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "step",
                        value: 1,
                    },
                    Attribute::F32 {
                        name: "lr",
                        bits: 0.1_f32.to_bits(),
                    },
                ],
                expected: Expected::Success {
                    outputs: vec![f32_buffer("parameter", &[2], &updated)],
                    scratch_bytes_max: 0,
                },
            },
            Case {
                case_id: "graph.add.forward.nonfinite",
                operation: "graph.add",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("left", &[1], &[f32::NAN]),
                    f32_buffer("right", &[1], &[1.0]),
                ],
                attributes: vec![],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "non_finite.left",
                    outputs: vec![f32_buffer("result", &[1], &[123.0])],
                },
            },
            Case {
                case_id: "graph.add.forward.duplicate_input",
                operation: "graph.add",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("left", &[1], &[1.0]),
                    f32_buffer("left", &[1], &[2.0]),
                ],
                attributes: vec![],
                expected: Expected::Error {
                    category: "invalid_request",
                    code: "duplicate_name.input.left",
                    outputs: vec![f32_buffer("result", &[1], &[456.0])],
                },
            },
            Case {
                case_id: "graph.transpose.forward.shape_error",
                operation: "graph.transpose",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[1, 6], &matrix)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 3,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[3, 2], &[123.0; 6])],
                },
            },
            Case {
                case_id: "graph.embedding_gather.forward.token_oob",
                operation: "graph.embedding_gather",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("weight", &[4, 2], &embedding_weight),
                    u32_buffer("tokens", &[1], &[4]),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "vocab",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "n_embd",
                        value: 2,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[1, 2], &[123.0; 2])],
                },
            },
            Case {
                case_id: "graph.slice_cols.forward.bounds_error",
                operation: "graph.slice_cols",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![f32_buffer("x", &[2, 4], &column_matrix)],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64 {
                        name: "cols",
                        value: 4,
                    },
                    Attribute::U64 {
                        name: "start",
                        value: 3,
                    },
                    Attribute::U64 {
                        name: "len",
                        value: 2,
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "attribute_value.start.slice_bounds",
                    outputs: vec![f32_buffer("result", &[2, 2], &[123.0; 4])],
                },
            },
            Case {
                case_id: "graph.concat_cols.forward.shape_error",
                operation: "graph.concat_cols",
                execution: "forward",
                tolerance: Tolerance::BitExact,
                inputs: vec![
                    f32_buffer("part.0", &[1, 4], &concat_left),
                    f32_buffer("part.1", &[2, 1], &concat_right),
                ],
                attributes: vec![
                    Attribute::U64 {
                        name: "rows",
                        value: 2,
                    },
                    Attribute::U64List {
                        name: "lens",
                        values: vec![2, 1],
                    },
                ],
                expected: Expected::Error {
                    category: "invalid_operation",
                    code: "shape",
                    outputs: vec![f32_buffer("result", &[2, 3], &[123.0; 6])],
                },
            },
        ],
    };

    let mut bytes = serde_json::to_vec_pretty(&corpus).expect("serialize tracer corpus");
    bytes.push(b'\n');
    TrainingVectorSetV1::parse_json(&bytes).expect("generated corpus must validate");

    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/training/v1/vectors/v1.json");
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, &bytes).expect("write temporary vector corpus");
    std::fs::rename(&temporary, &path).expect("atomically replace vector corpus");
    eprintln!(
        "froze {} cases -> {} ({})",
        corpus.cases.len(),
        path.display(),
        hex(blake3::hash(&bytes).as_bytes())
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
