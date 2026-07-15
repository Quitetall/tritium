#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use half::f16;
use tritium_cuda::CudaBackend;
use tritium_format::PackageId;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, write_salt_v2_package,
};
use tritium_nn::{ModelRunner, NnError, Projection};
use tritium_quantize::PhysicalSizeReport;

const HIDDEN: usize = 8;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = HIDDEN / HEADS;
const FF: usize = 16;
const VOCAB: usize = 12;
const Q_WIDTH: usize = HEADS * HEAD_DIM;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
const PLANE_SCALES: [f32; 3] = [0.125, 0.0625, 0.03125];

static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
struct MatrixSpec {
    name: &'static str,
    dims: [usize; 2],
}

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let nonce = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "tritium-salt-v2-model-{label}-{}-{nonce}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create isolated SALT V2 model fixture");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct ModelFixture {
    dir: TestDir,
    tensors: Vec<SaltV2Tensor>,
    package_order: Vec<String>,
    quantized_parameters: u64,
    q_bias: Vec<f32>,
    k_bias: Vec<f32>,
    v_bias: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
}

impl ModelFixture {
    fn llama() -> Self {
        Self::new("llama", "llama", "", true)
    }

    fn untied_llama() -> Self {
        Self::new("untied-llama", "llama", "", false)
    }

    fn qwen2() -> Self {
        Self::new("qwen2", "qwen2", ",\"attention_bias\":true", true)
    }

    fn qwen3() -> Self {
        Self::new(
            "qwen3",
            "qwen3",
            &format!(",\"head_dim\":{HEAD_DIM},\"qk_norm\":true"),
            true,
        )
    }

    fn new(label: &str, model_type: &str, config_extras: &str, tied_embeddings: bool) -> Self {
        let dir = TestDir::new(label);
        std::fs::write(
            dir.path().join("config.json"),
            format!(
                r#"{{
                    "model_type":"{model_type}",
                    "hidden_size":{HIDDEN},
                    "num_hidden_layers":1,
                    "num_attention_heads":{HEADS},
                    "num_key_value_heads":{KV_HEADS},
                    "intermediate_size":{FF},
                    "max_position_embeddings":32,
                    "rope_theta":10000.0,
                    "rms_norm_eps":1e-5,
                    "hidden_act":"silu",
                    "tie_word_embeddings":{tied_embeddings},
                    "vocab_size":{VOCAB}{config_extras}
                }}"#
            ),
        )
        .expect("write config.json");

        let specs = matrix_specs(!tied_embeddings);
        let mut tensors = Vec::with_capacity(specs.len());
        let mut dense = Vec::with_capacity(specs.len() + 3);
        for spec in &specs {
            let (tensor, values) = semantic_matrix(spec.name, spec.dims);
            tensors.push(tensor);
            dense.push((spec.name.to_owned(), spec.dims.to_vec(), values));
        }
        dense.extend([
            (
                "model.norm.weight".to_owned(),
                vec![HIDDEN],
                patterned_vector(HIDDEN, 1.0, 0.025),
            ),
            (
                "model.layers.0.input_layernorm.weight".to_owned(),
                vec![HIDDEN],
                patterned_vector(HIDDEN, 0.9, 0.02),
            ),
            (
                "model.layers.0.post_attention_layernorm.weight".to_owned(),
                vec![HIDDEN],
                patterned_vector(HIDDEN, 1.1, -0.015),
            ),
        ]);

        let (q_bias, k_bias, v_bias) = if model_type == "qwen2" {
            let q = patterned_vector(Q_WIDTH, 0.35, -0.045);
            let k = patterned_vector(KV_WIDTH, -0.22, 0.07);
            let v = patterned_vector(KV_WIDTH, 0.18, 0.055);
            dense.extend([
                (
                    "model.layers.0.self_attn.q_proj.bias".to_owned(),
                    vec![Q_WIDTH],
                    q.clone(),
                ),
                (
                    "model.layers.0.self_attn.k_proj.bias".to_owned(),
                    vec![KV_WIDTH],
                    k.clone(),
                ),
                (
                    "model.layers.0.self_attn.v_proj.bias".to_owned(),
                    vec![KV_WIDTH],
                    v.clone(),
                ),
            ]);
            (q, k, v)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let (q_norm, k_norm) = if model_type == "qwen3" {
            let q = vec![0.55, 0.85, 1.2, 1.55];
            let k = vec![1.45, 1.1, 0.75, 0.5];
            dense.extend([
                (
                    "model.layers.0.self_attn.q_norm.weight".to_owned(),
                    vec![HEAD_DIM],
                    q.clone(),
                ),
                (
                    "model.layers.0.self_attn.k_norm.weight".to_owned(),
                    vec![HEAD_DIM],
                    k.clone(),
                ),
            ]);
            (q, k)
        } else {
            (Vec::new(), Vec::new())
        };
        std::fs::write(dir.path().join("model.safetensors"), safetensors(&dense))
            .expect("write model.safetensors");

        let package_order = tensors
            .iter()
            .map(|tensor| tensor.name().to_owned())
            .collect::<Vec<_>>();
        let quantized_parameters = tensors
            .iter()
            .map(|tensor| tensor.logical_coefficients() as u64)
            .sum();
        Self {
            dir,
            tensors,
            package_order,
            quantized_parameters,
            q_bias,
            k_bias,
            v_bias,
            q_norm,
            k_norm,
        }
    }

    fn write_package(
        &self,
        label: &str,
        codec: SaltV2Codec,
        tensors: Vec<SaltV2Tensor>,
    ) -> (PathBuf, PackageId) {
        let package = SaltV2Package::new(codec, tensors).expect("construct SALT V2 package");
        let encoded = write_salt_v2_package(&package).expect("encode SALT V2 package");
        let id = PackageId::from_package_bytes(&encoded.bytes);
        let path = self.dir.path().join(format!("{label}.tsv2"));
        std::fs::write(&path, encoded.bytes).expect("write SALT V2 package");
        (path, id)
    }
}

fn matrix_specs(include_lm_head: bool) -> Vec<MatrixSpec> {
    // Deliberately not loader request order. The receipt must preserve this package order.
    let mut specs = vec![
        MatrixSpec {
            name: "model.layers.0.mlp.down_proj.weight",
            dims: [HIDDEN, FF],
        },
        MatrixSpec {
            name: "model.embed_tokens.weight",
            dims: [VOCAB, HIDDEN],
        },
        MatrixSpec {
            name: "model.layers.0.self_attn.v_proj.weight",
            dims: [KV_WIDTH, HIDDEN],
        },
        MatrixSpec {
            name: "model.layers.0.mlp.gate_proj.weight",
            dims: [FF, HIDDEN],
        },
        MatrixSpec {
            name: "model.layers.0.self_attn.o_proj.weight",
            dims: [HIDDEN, Q_WIDTH],
        },
        MatrixSpec {
            name: "model.layers.0.self_attn.k_proj.weight",
            dims: [KV_WIDTH, HIDDEN],
        },
        MatrixSpec {
            name: "model.layers.0.mlp.up_proj.weight",
            dims: [FF, HIDDEN],
        },
        MatrixSpec {
            name: "model.layers.0.self_attn.q_proj.weight",
            dims: [Q_WIDTH, HIDDEN],
        },
    ];
    if include_lm_head {
        specs.insert(
            3,
            MatrixSpec {
                name: "lm_head.weight",
                dims: [VOCAB, HIDDEN],
            },
        );
    }
    specs
}

fn semantic_matrix(name: &'static str, dims: [usize; 2]) -> (SaltV2Tensor, Vec<f32>) {
    let len = dims[0] * dims[1];
    assert!(
        len.is_multiple_of(4),
        "S34 fixture matrices need full groups"
    );
    let seed = name
        .bytes()
        .fold(0usize, |sum, byte| sum.wrapping_add(byte as usize));
    let planes = (0..PLANE_SCALES.len())
        .map(|plane| structured_trits(len, seed, plane))
        .collect::<Vec<_>>();
    let mut dense = vec![0.0; len];
    for (plane, (&scale, trits)) in PLANE_SCALES.iter().zip(&planes).enumerate() {
        for (index, &trit) in trits.iter().enumerate() {
            dense[index] += f32::from(trit) * scale;
        }
        assert!(
            dense.iter().any(|value| *value != 0.0),
            "plane {plane} must contribute nonzero values"
        );
    }

    let mut tiles = Vec::new();
    for tile_start in (0..len).step_by(256) {
        let tile_end = (tile_start + 256).min(len);
        let tile_planes = planes
            .iter()
            .zip(PLANE_SCALES)
            .map(|(trits, scale)| {
                let slice = trits[tile_start..tile_end].to_vec();
                let scales = vec![f16::from_f32(scale); slice.len().div_ceil(128)];
                SaltV2Plane::new(slice, scales).expect("construct additive plane")
            })
            .collect();
        tiles.push(SaltV2Tile::new(tile_planes).expect("construct additive tile"));
    }
    let tensor = SaltV2Tensor::new(name, dims.map(|dim| dim as u64).to_vec(), tiles)
        .expect("construct semantic matrix");
    (tensor, dense)
}

fn structured_trits(len: usize, seed: usize, plane: usize) -> Vec<i8> {
    (0..len)
        .map(|index| {
            let group = index / 4;
            let lane = index % 4;
            let zero_lane = (seed + plane * 3 + group) % 4;
            if lane == zero_lane {
                0
            } else if (seed + plane * 11 + index * 5).is_multiple_of(2) {
                1
            } else {
                -1
            }
        })
        .collect()
}

fn patterned_vector(len: usize, base: f32, step: f32) -> Vec<f32> {
    (0..len).map(|index| base + index as f32 * step).collect()
}

fn safetensors(tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
    let mut data = Vec::new();
    let mut header = String::from("{");
    for (index, (name, shape, values)) in tensors.iter().enumerate() {
        let expected: usize = shape.iter().product();
        assert_eq!(values.len(), expected, "safetensors shape for {name}");
        let start = data.len();
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        if index != 0 {
            header.push(',');
        }
        let shape = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape}],\"data_offsets\":[{start},{}]}}",
            data.len()
        ));
    }
    header.push('}');
    let header = header.into_bytes();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&data);
    bytes
}

fn cuda_or_skip(test: &str) -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(cuda) => Some(cuda),
        Err(error) => {
            eprintln!("skipping {test}: CUDA device 0 unavailable: {error}");
            None
        }
    }
}

fn assert_logits_close(codec: SaltV2Codec, expected: &[f32], actual: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        expected.iter().chain(actual).all(|value| value.is_finite()),
        "{codec:?} end-to-end logits must be finite"
    );
    let maximum = expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    let range = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - expected.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        maximum <= 2e-3 || maximum / range.max(1e-6) <= 2e-3,
        "{codec:?} end-to-end logits diverged from dense reconstruction: max_abs={maximum:e}, range={range:e}"
    );
}

#[test]
fn complete_packages_run_on_gpu_for_every_codec_with_exact_receipts() {
    let fixture = ModelFixture::llama();
    let mut oracle =
        ModelRunner::from_hf(fixture.dir.path(), Box::new(tritium_cpu::CpuBackend::new()))
            .expect("load dense reconstruction oracle");
    let tokens = [1, 7, 3];
    let positions = [0, 1, 2];
    let expected = oracle
        .forward(&tokens, &positions)
        .expect("run dense reconstruction oracle");

    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let Some(cuda) = cuda_or_skip("complete SALT V2 model load") else {
            return;
        };
        let (package, package_id) = fixture.write_package(
            &format!("complete-{codec:?}"),
            codec,
            fixture.tensors.clone(),
        );
        let (mut runner, receipt) =
            ModelRunner::from_salt_v2(fixture.dir.path(), &package, package_id, Box::new(cuda))
                .expect("load complete SALT V2 model");

        assert_eq!(receipt.package_id(), package_id);
        assert_eq!(receipt.codec(), codec);
        assert_eq!(receipt.quantized_parameters(), fixture.quantized_parameters);
        assert_eq!(
            receipt
                .tensors()
                .iter()
                .map(|tensor| tensor.name())
                .collect::<Vec<_>>(),
            fixture
                .package_order
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        assert_eq!(receipt.tensors().len(), receipt.runtime_ledgers().len());
        for (tensor, runtime) in receipt.tensors().iter().zip(receipt.runtime_ledgers()) {
            assert_eq!(tensor.runtime(), *runtime);
            assert_eq!(runtime.dense_shadow_bytes(), 0);
        }
        assert_eq!(
            receipt.payload_bytes(),
            receipt
                .runtime_ledgers()
                .iter()
                .map(|runtime| runtime.payload_bytes())
                .sum::<u64>()
        );
        assert_eq!(
            receipt.scale_bytes(),
            receipt
                .runtime_ledgers()
                .iter()
                .map(|runtime| runtime.scale_bytes())
                .sum::<u64>()
        );
        assert_eq!(
            receipt.v2_resident_bytes(),
            receipt
                .runtime_ledgers()
                .iter()
                .map(|runtime| runtime.steady_resident_bytes())
                .sum::<u64>()
        );
        assert_eq!(
            receipt.tracked_weight_bytes(),
            receipt.v2_resident_bytes() + receipt.preserved_fp32_bytes()
        );
        let package_bytes = std::fs::read(&package).expect("reopen SALT V2 package");
        PhysicalSizeReport::from_salt_v2_package_bytes_with_runtime_receipts(
            &package_bytes,
            receipt.quantized_parameters() + receipt.preserved_parameters(),
            receipt.runtime_ledgers(),
            None,
        )
        .expect("runtime receipts must preserve encoded package order");
        assert!(runner.weights.token_embd.is_packed_salt());
        assert!(runner.weights.token_embd.as_dense().is_none());
        for layer in &runner.weights.layers {
            let projections = match &layer.mlp {
                tritium_nn::Mlp::SwiGlu(mlp) => vec![
                    &layer.q_proj,
                    &layer.k_proj,
                    &layer.v_proj,
                    &layer.o_proj,
                    &mlp.gate,
                    &mlp.up,
                    &mlp.down,
                ],
                tritium_nn::Mlp::Relu2(_) => panic!("fixture is SwiGLU"),
            };
            assert!(
                projections
                    .into_iter()
                    .all(|projection| matches!(projection, Projection::SaltV2(_)))
            );
        }

        let actual = runner
            .forward(&tokens, &positions)
            .expect("run SALT V2 model on CUDA");
        assert_logits_close(codec, &expected, &actual);
    }
}

#[test]
fn untied_head_is_a_separate_salt_v2_resident_and_matches_dense_oracle() {
    let fixture = ModelFixture::untied_llama();
    let (_, embedding_values) = semantic_matrix("model.embed_tokens.weight", [VOCAB, HIDDEN]);
    let (_, head_values) = semantic_matrix("lm_head.weight", [VOCAB, HIDDEN]);
    assert_ne!(
        embedding_values, head_values,
        "fixture must make accidental embedding/head aliasing observable"
    );

    let mut oracle =
        ModelRunner::from_hf(fixture.dir.path(), Box::new(tritium_cpu::CpuBackend::new()))
            .expect("load dense untied-head oracle");
    assert!(oracle.weights.lm_head.is_some(), "dense fixture is untied");
    let tokens = [6, 1, 10];
    let positions = [0, 1, 2];
    let expected = oracle
        .forward(&tokens, &positions)
        .expect("run dense untied-head oracle");

    let (package, package_id) =
        fixture.write_package("untied-complete", SaltV2Codec::B3, fixture.tensors.clone());
    let Some(cuda) = cuda_or_skip("untied SALT V2 model load") else {
        return;
    };
    let (mut runner, receipt) =
        ModelRunner::from_salt_v2(fixture.dir.path(), &package, package_id, Box::new(cuda))
            .expect("load untied SALT V2 model");

    assert_eq!(
        receipt
            .tensors()
            .iter()
            .map(|tensor| tensor.name())
            .collect::<Vec<_>>(),
        fixture
            .package_order
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );
    assert_eq!(receipt.quantized_parameters(), fixture.quantized_parameters);
    assert_eq!(receipt.tensors().len(), matrix_specs(true).len());
    assert!(
        receipt
            .runtime_ledgers()
            .iter()
            .all(|runtime| runtime.dense_shadow_bytes() == 0)
    );
    let embedding_receipt = receipt
        .tensors()
        .iter()
        .find(|tensor| tensor.name() == "model.embed_tokens.weight")
        .expect("embedding receipt");
    let head_receipt = receipt
        .tensors()
        .iter()
        .find(|tensor| tensor.name() == "lm_head.weight")
        .expect("untied head receipt");
    assert_eq!(
        runner.weights.token_embd.resident_bytes() as u64,
        embedding_receipt.runtime().steady_resident_bytes()
    );
    assert!(runner.weights.token_embd.is_packed_salt());
    assert!(runner.weights.token_embd.as_dense().is_none());
    let head = match runner.weights.lm_head.as_ref() {
        Some(Projection::SaltV2(head)) => head,
        Some(_) => panic!("untied head must be a SALT V2 resident"),
        None => panic!("untied config must publish a separate head"),
    };
    assert_eq!(head.rows(), VOCAB);
    assert_eq!(head.columns(), HIDDEN);
    assert_eq!(
        head.allocation_receipt().runtime_ledger(),
        head_receipt.runtime()
    );

    let actual = runner
        .forward(&tokens, &positions)
        .expect("run untied SALT V2 model on CUDA");
    assert_logits_close(SaltV2Codec::B3, &expected, &actual);
}

#[test]
fn package_identity_and_extra_tensor_are_rejected() {
    let fixture = ModelFixture::llama();
    let (complete, package_id) =
        fixture.write_package("identity", SaltV2Codec::D2, fixture.tensors.clone());
    let Some(cuda) = cuda_or_skip("SALT V2 identity rejection") else {
        return;
    };
    let wrong_id = PackageId::from_package_bytes(b"definitely not this package");
    let error =
        match ModelRunner::from_salt_v2(fixture.dir.path(), &complete, wrong_id, Box::new(cuda)) {
            Ok(_) => panic!("wrong package identity must fail"),
            Err(error) => error,
        };
    assert!(
        matches!(&error, NnError::MissingTensor(message) if message.contains("identity mismatch") && message.contains(&package_id.to_string())),
        "unexpected identity error: {error}"
    );

    let mut with_extra = fixture.tensors.clone();
    with_extra.push(semantic_matrix("unused.extra.weight", [4, 4]).0);
    let (extra, extra_id) = fixture.write_package("extra", SaltV2Codec::D2, with_extra);
    let Some(cuda) = cuda_or_skip("SALT V2 extra-tensor rejection") else {
        return;
    };
    let error =
        match ModelRunner::from_salt_v2(fixture.dir.path(), &extra, extra_id, Box::new(cuda)) {
            Ok(_) => panic!("an unowned package tensor must fail exact coverage"),
            Err(error) => error,
        };
    assert!(
        matches!(&error, NnError::Backend(message) if message.contains("coverage differs") && message.contains("unused.extra.weight")),
        "unexpected extra-tensor error: {error}"
    );
}

#[test]
fn missing_name_and_wrong_matrix_shape_are_rejected() {
    let fixture = ModelFixture::llama();
    let mut missing = fixture.tensors.clone();
    missing.retain(|tensor| tensor.name() != "model.layers.0.self_attn.q_proj.weight");
    let (package, package_id) = fixture.write_package("missing-q", SaltV2Codec::D2, missing);
    let Some(cuda) = cuda_or_skip("SALT V2 missing-name rejection") else {
        return;
    };
    let error =
        match ModelRunner::from_salt_v2(fixture.dir.path(), &package, package_id, Box::new(cuda)) {
            Ok(_) => panic!("a missing required package tensor must fail"),
            Err(error) => error,
        };
    assert!(
        matches!(&error, NnError::MissingTensor(message) if message.contains("model.layers.0.self_attn.q_proj.weight")),
        "unexpected missing-name error: {error}"
    );

    let mut wrong_shape = fixture.tensors.clone();
    let q = wrong_shape
        .iter_mut()
        .find(|tensor| tensor.name() == "model.layers.0.self_attn.q_proj.weight")
        .expect("q projection fixture");
    // Preserve the coefficient count so this catches exact matrix geometry checks,
    // rather than only a flat-length disagreement.
    *q = semantic_matrix("model.layers.0.self_attn.q_proj.weight", [4, 16]).0;
    let (package, package_id) =
        fixture.write_package("wrong-q-shape", SaltV2Codec::D2, wrong_shape);
    let Some(cuda) = cuda_or_skip("SALT V2 matrix-shape rejection") else {
        return;
    };
    let error =
        match ModelRunner::from_salt_v2(fixture.dir.path(), &package, package_id, Box::new(cuda)) {
            Ok(_) => panic!("a wrong required matrix shape must fail"),
            Err(error) => error,
        };
    assert!(
        matches!(error, NnError::Shape { .. }),
        "unexpected wrong-shape error: {error}"
    );
}

#[test]
fn qwen_optional_attention_vectors_survive_loading_and_match_dense_oracle() {
    for (label, fixture) in [
        ("qwen2-bias", ModelFixture::qwen2()),
        ("qwen3-qk-norm", ModelFixture::qwen3()),
    ] {
        let mut oracle =
            ModelRunner::from_hf(fixture.dir.path(), Box::new(tritium_cpu::CpuBackend::new()))
                .expect("load dense Qwen oracle");
        let tokens = [2, 9, 4];
        let positions = [0, 1, 2];
        let expected = oracle
            .forward(&tokens, &positions)
            .expect("run dense Qwen oracle");

        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let (package, package_id) = fixture.write_package(
                &format!("{label}-{codec:?}-complete"),
                codec,
                fixture.tensors.clone(),
            );
            let Some(cuda) = cuda_or_skip(label) else {
                return;
            };
            let (mut runner, receipt) =
                ModelRunner::from_salt_v2(fixture.dir.path(), &package, package_id, Box::new(cuda))
                    .expect("load complete Qwen SALT V2 model");
            let layer = &runner.weights.layers[0];

            assert_eq!(layer.q_bias, fixture.q_bias, "{label} q bias");
            assert_eq!(layer.k_bias, fixture.k_bias, "{label} k bias");
            assert_eq!(layer.v_bias, fixture.v_bias, "{label} v bias");
            assert_eq!(layer.q_norm, fixture.q_norm, "{label} q norm");
            assert_eq!(layer.k_norm, fixture.k_norm, "{label} k norm");
            assert!(
                layer
                    .q_bias
                    .iter()
                    .chain(&layer.k_bias)
                    .chain(&layer.v_bias)
                    .chain(&layer.q_norm)
                    .chain(&layer.k_norm)
                    .all(|value| *value != 0.0),
                "{label} optional vectors must be nonzero"
            );

            let mut expected_preserved = vec![
                "model.layers.0.input_layernorm.weight",
                "model.layers.0.post_attention_layernorm.weight",
                "model.norm.weight",
            ];
            if !fixture.q_bias.is_empty() {
                expected_preserved.extend([
                    "model.layers.0.self_attn.k_proj.bias",
                    "model.layers.0.self_attn.q_proj.bias",
                    "model.layers.0.self_attn.v_proj.bias",
                ]);
            }
            if !fixture.q_norm.is_empty() {
                expected_preserved.extend([
                    "model.layers.0.self_attn.k_norm.weight",
                    "model.layers.0.self_attn.q_norm.weight",
                ]);
            }
            expected_preserved.sort_unstable();
            assert_eq!(
                receipt
                    .preserved_tensors()
                    .iter()
                    .map(|tensor| tensor.name())
                    .collect::<Vec<_>>(),
                expected_preserved,
                "{label} preserved receipt names"
            );
            let expected_preserved_parameters = 3 * HIDDEN
                + fixture.q_bias.len()
                + fixture.k_bias.len()
                + fixture.v_bias.len()
                + fixture.q_norm.len()
                + fixture.k_norm.len();
            assert_eq!(
                receipt.preserved_parameters(),
                expected_preserved_parameters as u64,
                "{label} preserved parameter accounting"
            );
            assert_eq!(
                receipt.preserved_fp32_bytes(),
                (expected_preserved_parameters * core::mem::size_of::<f32>()) as u64,
                "{label} preserved byte accounting"
            );
            assert!(
                receipt
                    .runtime_ledgers()
                    .iter()
                    .all(|runtime| runtime.dense_shadow_bytes() == 0),
                "{label} must not retain a dense quantized-weight shadow"
            );

            let actual = runner
                .forward(&tokens, &positions)
                .expect("run Qwen SALT V2 model on CUDA");
            assert_logits_close(codec, &expected, &actual);
        }
    }
}
