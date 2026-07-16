use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};

use serde_json::json;
use tritium_nn::{Qwen35HfLanguageModel, Qwen35HfSource, Qwen35TensorStreamError};

const H: usize = 4;
const I: usize = 6;
const V: usize = 7;

#[derive(Clone)]
struct TensorFixture {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn parameter(ordinal: usize, len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let residue = (17 * index + 13 * ordinal + 5) % 29;
            (residue as i32 - 14) as f32 / 32.0
        })
        .collect()
}

fn tensor(name: impl Into<String>, shape: &[usize], ordinal: usize) -> TensorFixture {
    let len = shape.iter().product();
    TensorFixture {
        name: name.into(),
        shape: shape.to_vec(),
        values: parameter(ordinal, len),
    }
}

fn oracle_tensors() -> Vec<TensorFixture> {
    let mut tensors = vec![
        tensor("model.language_model.embed_tokens.weight", &[V, H], 0),
        tensor("model.language_model.norm.weight", &[H], 26),
        tensor("lm_head.weight", &[V, H], 27),
        tensor("mtp.norm.weight", &[H], 28),
        tensor("model.visual.pos_embed.weight", &[1, H], 29),
    ];
    let delta = "model.language_model.layers.0";
    tensors.extend([
        tensor(format!("{delta}.linear_attn.dt_bias"), &[2], 1),
        tensor(format!("{delta}.linear_attn.A_log"), &[2], 2),
        tensor(format!("{delta}.linear_attn.conv1d.weight"), &[8, 1, 4], 3),
        tensor(format!("{delta}.linear_attn.norm.weight"), &[2], 4),
        tensor(format!("{delta}.linear_attn.out_proj.weight"), &[H, 4], 5),
        tensor(
            format!("{delta}.linear_attn.in_proj_qkv.weight"),
            &[8, H],
            6,
        ),
        tensor(format!("{delta}.linear_attn.in_proj_z.weight"), &[4, H], 7),
        tensor(format!("{delta}.linear_attn.in_proj_b.weight"), &[2, H], 8),
        tensor(format!("{delta}.linear_attn.in_proj_a.weight"), &[2, H], 9),
        tensor(format!("{delta}.mlp.gate_proj.weight"), &[I, H], 10),
        tensor(format!("{delta}.mlp.up_proj.weight"), &[I, H], 11),
        tensor(format!("{delta}.mlp.down_proj.weight"), &[H, I], 12),
        tensor(format!("{delta}.input_layernorm.weight"), &[H], 13),
        tensor(format!("{delta}.post_attention_layernorm.weight"), &[H], 14),
    ]);
    let full = "model.language_model.layers.1";
    tensors.extend([
        tensor(format!("{full}.self_attn.q_proj.weight"), &[16, H], 15),
        tensor(format!("{full}.self_attn.k_proj.weight"), &[4, H], 16),
        tensor(format!("{full}.self_attn.v_proj.weight"), &[4, H], 17),
        tensor(format!("{full}.self_attn.o_proj.weight"), &[H, 8], 18),
        tensor(format!("{full}.self_attn.q_norm.weight"), &[4], 19),
        tensor(format!("{full}.self_attn.k_norm.weight"), &[4], 20),
        tensor(format!("{full}.mlp.gate_proj.weight"), &[I, H], 21),
        tensor(format!("{full}.mlp.up_proj.weight"), &[I, H], 22),
        tensor(format!("{full}.mlp.down_proj.weight"), &[H, I], 23),
        tensor(format!("{full}.input_layernorm.weight"), &[H], 24),
        tensor(format!("{full}.post_attention_layernorm.weight"), &[H], 25),
    ]);
    tensors
}

fn safetensors(tensors: &[TensorFixture]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for tensor in tensors {
        let start = payload.len();
        for value in &tensor.values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            tensor.name.clone(),
            json!({
                "dtype": "F32",
                "shape": tensor.shape,
                "data_offsets": [start, payload.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&payload);
    bytes
}

fn config_json() -> String {
    json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "language_model_only": false,
        "model_type": "qwen3_5",
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "attn_output_gate": true,
            "dtype": "bfloat16",
            "full_attention_interval": 2,
            "head_dim": 4,
            "hidden_act": "silu",
            "hidden_size": H,
            "intermediate_size": I,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 2,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 2,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 32,
            "model_type": "qwen3_5_text",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "output_gate_type": "swish",
            "partial_rotary_factor": 0.5,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [1, 0, 0],
                "partial_rotary_factor": 0.5,
                "rope_theta": 10000.0,
                "rope_type": "default"
            },
            "tie_word_embeddings": false,
            "use_cache": true,
            "vocab_size": V
        },
        "tie_word_embeddings": false,
        "vision_config": {"model_type": "qwen3_5"}
    })
    .to_string()
}

fn fixture_dir(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("tritium-qwen35-hf-{label}-{}", std::process::id()))
}

fn write_fixture(dir: &Path) {
    write_fixture_tensors(dir, oracle_tensors());
}

fn write_fixture_tensors(dir: &Path, tensors: Vec<TensorFixture>) {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), config_json()).unwrap();
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut weight_map = BTreeMap::new();
    for (index, tensor) in tensors.into_iter().enumerate() {
        let shard = if index.is_multiple_of(2) {
            left.push(tensor.clone());
            "model-00001-of-00002.safetensors"
        } else {
            right.push(tensor.clone());
            "model-00002-of-00002.safetensors"
        };
        weight_map.insert(tensor.name, shard);
    }
    std::fs::write(
        dir.join("model-00001-of-00002.safetensors"),
        safetensors(&left),
    )
    .unwrap();
    std::fs::write(
        dir.join("model-00002-of-00002.safetensors"),
        safetensors(&right),
    )
    .unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        json!({"weight_map": weight_map}).to_string(),
    )
    .unwrap();
}

fn write_single_shard_fixture(dir: &Path, mut tensors: Vec<TensorFixture>) {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut parsed: serde_json::Value = serde_json::from_str(&config_json()).unwrap();
    parsed["transformers_version"] = json!("ignored-provenance-field");
    parsed["text_config"]["initializer_range"] = json!(0.123);
    parsed["vision_config"]["deferred_fixture_detail"] = json!(true);
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&parsed).unwrap(),
    )
    .unwrap();
    tensors.reverse();
    std::fs::write(dir.join("model.safetensors"), safetensors(&tensors)).unwrap();
}

fn mutate_single_shard_tensor_byte(dir: &Path, name: &str) {
    let path = dir.join("model.safetensors");
    let bytes = std::fs::read(&path).unwrap();
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let relative = header[name]["data_offsets"][0].as_u64().unwrap();
    let absolute = u64::try_from(8 + header_len).unwrap() + relative;
    let current = bytes[usize::try_from(absolute).unwrap()];
    let mut file = OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(absolute)).unwrap();
    file.write_all(&[current ^ 1]).unwrap();
    file.sync_data().unwrap();
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "value {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

#[test]
fn source_identity_is_independent_of_shard_layout_and_json_formatting() {
    let sharded_dir = fixture_dir("source-identity-sharded");
    let single_dir = fixture_dir("source-identity-single");
    let tensors = oracle_tensors();
    write_fixture_tensors(&sharded_dir, tensors.clone());
    write_single_shard_fixture(&single_dir, tensors.clone());

    let sharded = Qwen35HfSource::open(&sharded_dir).unwrap();
    let single = Qwen35HfSource::open(&single_dir).unwrap();
    assert_eq!(sharded.metadata(), single.metadata());
    assert_eq!(sharded.metadata().len(), tensors.len());
    assert!(
        sharded
            .metadata()
            .windows(2)
            .all(|pair| pair[0].name() < pair[1].name())
    );
    let embedding = sharded
        .metadata()
        .iter()
        .find(|tensor| tensor.name() == "model.language_model.embed_tokens.weight")
        .unwrap();
    assert_eq!(embedding.dtype(), "F32");
    assert_eq!(embedding.shape(), &[V as u64, H as u64]);

    assert_eq!(
        sharded.canonical_config_bytes(),
        single.canonical_config_bytes()
    );
    assert_eq!(
        blake3::hash(sharded.canonical_config_bytes())
            .to_hex()
            .as_str(),
        "b43c6b9e4292d976de93a0e4a4c73e91669eb9ab27cb341eefc0087d25928bf0"
    );
    let sharded = sharded.verify_semantic_identity().unwrap();
    let single = single.verify_semantic_identity().unwrap();
    assert_eq!(
        sharded.canonical_config_bytes(),
        single.canonical_config_bytes()
    );
    assert!(
        sharded
            .identity()
            .manifest()
            .matches_canonical_config(sharded.canonical_config_bytes())
    );
    let source_model_id = sharded.model_id();
    assert_eq!(source_model_id, single.model_id());
    assert_eq!(
        sharded.identity().payload_bytes(),
        tensors
            .iter()
            .map(|tensor| (tensor.values.len() * size_of::<f32>()) as u64)
            .sum::<u64>()
    );
    assert_eq!(sharded.identity().manifest().model_id(), source_model_id);

    let model = sharded
        .load_language(Box::new(tritium_cpu::CpuBackend::new()))
        .unwrap();
    assert_eq!(model.receipt().language_tensors(), 28);
    assert_eq!(model.source_identity().model_id(), source_model_id);

    let _ = std::fs::remove_dir_all(&sharded_dir);
    let _ = std::fs::remove_dir_all(&single_dir);
}

#[test]
fn source_identity_binds_payload_bits_not_only_metadata() {
    let original_dir = fixture_dir("source-identity-original");
    let mutated_dir = fixture_dir("source-identity-mutated");
    let original = oracle_tensors();
    let mut mutated = original.clone();
    mutated[0].values[0] = f32::from_bits(mutated[0].values[0].to_bits() ^ 1);
    write_single_shard_fixture(&original_dir, original);
    write_single_shard_fixture(&mutated_dir, mutated);

    let original = Qwen35HfSource::open(&original_dir).unwrap();
    let mutated = Qwen35HfSource::open(&mutated_dir).unwrap();
    assert_eq!(original.metadata(), mutated.metadata());
    assert_ne!(
        original.verify_semantic_identity().unwrap().model_id(),
        mutated.verify_semantic_identity().unwrap().model_id()
    );

    let _ = std::fs::remove_dir_all(&original_dir);
    let _ = std::fs::remove_dir_all(&mutated_dir);
}

#[derive(Debug, PartialEq, Eq)]
struct SinkStop;

#[test]
fn verified_source_streams_bounded_raw_chunks_and_preserves_sink_errors() {
    let dir = fixture_dir("verified-byte-stream");
    let tensors = oracle_tensors();
    let expected = tensors
        .iter()
        .find(|tensor| tensor.name == "model.language_model.norm.weight")
        .unwrap()
        .values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_single_shard_fixture(&dir, tensors);
    let source = Qwen35HfSource::open(&dir)
        .unwrap()
        .verify_semantic_identity()
        .unwrap();

    let mut actual = Vec::new();
    let payload_bytes = source
        .try_visit_tensor_bytes(
            "model.language_model.norm.weight",
            5,
            |chunk| -> Result<(), SinkStop> {
                assert!(!chunk.is_empty());
                assert!(chunk.len() <= 5);
                actual.extend_from_slice(chunk);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(payload_bytes, expected.len() as u64);
    assert_eq!(actual, expected);

    let error = source
        .try_visit_tensor_bytes("model.language_model.norm.weight", 3, |_| Err(SinkStop))
        .unwrap_err();
    assert!(matches!(error, Qwen35TensorStreamError::Sink(SinkStop)));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn source_identity_binds_canonical_execution_config() {
    let original_dir = fixture_dir("source-config-original");
    let changed_dir = fixture_dir("source-config-changed");
    let tensors = oracle_tensors();
    write_single_shard_fixture(&original_dir, tensors.clone());
    write_single_shard_fixture(&changed_dir, tensors);
    let config_path = changed_dir.join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config["text_config"]["max_position_embeddings"] = json!(64);
    std::fs::write(&config_path, serde_json::to_vec_pretty(&config).unwrap()).unwrap();

    let original = Qwen35HfSource::open(&original_dir)
        .unwrap()
        .verify_semantic_identity()
        .unwrap();
    let changed = Qwen35HfSource::open(&changed_dir)
        .unwrap()
        .verify_semantic_identity()
        .unwrap();
    assert_ne!(original.model_id(), changed.model_id());

    let _ = std::fs::remove_dir_all(&original_dir);
    let _ = std::fs::remove_dir_all(&changed_dir);
}

#[test]
fn verified_load_rejects_in_place_payload_mutation() {
    let dir = fixture_dir("source-identity-toctou");
    write_single_shard_fixture(&dir, oracle_tensors());
    let verified = Qwen35HfSource::open(&dir)
        .unwrap()
        .verify_semantic_identity()
        .unwrap();
    mutate_single_shard_tensor_byte(&dir, "model.language_model.embed_tokens.weight");

    let result = verified.load_language(Box::new(tritium_cpu::CpuBackend::new()));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(matches!(
        result,
        Err(tritium_nn::NnError::MissingTensor(reason))
            if reason.contains("embed_tokens.weight")
                && reason.contains("changed after semantic identity verification")
    ));
}

#[test]
#[allow(clippy::excessive_precision)]
fn two_shard_adapter_matches_transformers_5_5_3_oracle() {
    let dir = fixture_dir("oracle");
    write_fixture(&dir);

    let model =
        Qwen35HfLanguageModel::load_family(&dir, Box::new(tritium_cpu::CpuBackend::new())).unwrap();
    assert_eq!(model.receipt().language_tensors(), 28);
    assert_eq!(model.receipt().language_matrices(), 17);
    assert_eq!(model.receipt().language_preserved_tensors(), 11);
    assert_eq!(model.receipt().deferred_mtp_tensors(), 1);
    assert_eq!(model.receipt().deferred_vision_tensors(), 1);

    let mut cache = model.runner().new_cache(16).unwrap();
    let output = model.runner().forward(&[1, 4, 2], &mut cache).unwrap();
    let expected_hidden = [
        0.2664346993,
        -1.6725171804,
        0.4617776573,
        -0.3494569063,
        0.4306403697,
        -1.5771758556,
        0.5621415973,
        -0.2753130794,
        2.1214075089,
        -1.0040738583,
        0.2510471940,
        0.2221434563,
    ];
    let expected_logits = [
        -0.8410053253,
        0.5659754872,
        0.8355028033,
        -0.7913014293,
        0.6156793833,
        -1.0373188257,
        -0.7415975332,
    ];
    assert_close(output.final_hidden_states(), &expected_hidden, 3e-6);
    assert_close(output.last_logits(), &expected_logits, 3e-6);
    assert_eq!(cache.len(), 3);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn missing_required_language_tensor_fails_closed() {
    let dir = fixture_dir("missing-language");
    let tensors = oracle_tensors()
        .into_iter()
        .filter(|tensor| tensor.name != "lm_head.weight")
        .collect();
    write_fixture_tensors(&dir, tensors);

    let result = Qwen35HfLanguageModel::load_family(&dir, Box::new(tritium_cpu::CpuBackend::new()));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(matches!(
        result,
        Err(tritium_nn::NnError::MissingTensor(name)) if name == "lm_head.weight"
    ));
}

#[test]
fn wrong_language_tensor_shape_fails_before_assembly() {
    let dir = fixture_dir("wrong-shape");
    let mut tensors = oracle_tensors();
    let conv = tensors
        .iter_mut()
        .find(|tensor| tensor.name.ends_with("linear_attn.conv1d.weight"))
        .unwrap();
    conv.shape = vec![8, 4];
    write_fixture_tensors(&dir, tensors);

    let result = Qwen35HfLanguageModel::load_family(&dir, Box::new(tritium_cpu::CpuBackend::new()));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(matches!(
        result,
        Err(tritium_nn::NnError::MissingTensor(reason))
            if reason.contains("linear_attn.conv1d.weight")
                && reason.contains("expected [8, 1, 4]")
    ));
}

#[test]
fn unknown_language_tensor_fails_closed() {
    let dir = fixture_dir("unknown-language");
    let mut tensors = oracle_tensors();
    tensors.push(tensor(
        "model.language_model.layers.0.unrecognized.weight",
        &[1],
        30,
    ));
    write_fixture_tensors(&dir, tensors);

    let result = Qwen35HfLanguageModel::load_family(&dir, Box::new(tritium_cpu::CpuBackend::new()));
    let _ = std::fs::remove_dir_all(&dir);
    assert!(matches!(
        result,
        Err(tritium_nn::NnError::MissingTensor(reason))
            if reason.contains("unrecognized.weight")
    ));
}
