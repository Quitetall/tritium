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
use tritium_train::{Optimizer, Sgd};

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
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Attribute {
    F32 { name: &'static str, bits: u32 },
    U64 { name: &'static str, value: u64 },
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

fn main() {
    let left = [1.0_f32, -2.0, 0.5];
    let right = [3.0_f32, 4.0, -1.5];
    let add: Vec<_> = left.iter().zip(right).map(|(&x, y)| x + y).collect();
    let grad_output = [0.25_f32, -0.5, 2.0];

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
