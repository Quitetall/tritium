//! Self-contained training-SALT GGUF loading.
//!
//! Unlike [`ModelWeights::load_salt`](super::ModelWeights::load_salt), this path
//! has no model-directory fallback. The one GGUF v3 artifact contains its
//! canonical HuggingFace configuration, every quantized two-dimensional weight,
//! and every fp32 norm needed by the SwiGLU inference graph.

use std::collections::{BTreeSet, HashMap};

use tritium_format::{
    GGML_TYPE_TRITIUM_SALT, GgufFile, GgufValue, SALT_GGUF_FORMAT_KEY, SALT_GGUF_FORMAT_VALUE,
    SaltTensor, read_gguf, read_salt_gguf, salt_rows_to_dense,
};
use tritium_train::grow::{NET2WIDER_ALGORITHM_V1, NET2WIDER_ALGORITHM_V2, Net2WiderPlan};

use crate::config::{ArchSpec, MlpKind, ModelConfig};
use crate::error::NnError;
use crate::layers::{DenseLinear, Projection};
use crate::model::ModelWeights;
use crate::model::hf::{NameSchema, build_standard_model};
use crate::training::TiedSwiGluTrainingModel;

/// Metadata key marking a self-contained SwiGLU training-SALT model.
pub const TRAINING_SALT_FORMAT_KEY: &str = "tritium.training_salt.format";
/// Versioned self-contained tied-head training-SALT model format.
pub const TRAINING_SALT_FORMAT_VALUE: &str = "tied-swiglu.v1";
/// Versioned self-contained untied-head training-SALT model format.
pub const TRAINING_SALT_UNTIED_FORMAT_VALUE: &str = "untied-swiglu.v1";
/// Canonical compact HuggingFace configuration JSON embedded in the artifact.
pub const TRAINING_SALT_HF_CONFIG_KEY: &str = "tritium.training_salt.hf_config_json";
/// Fixed number of SALT planes in every quantized row.
pub const TRAINING_SALT_PLANES_KEY: &str = "tritium.training_salt.salt_planes";
/// Immutable campaign-plan fingerprint carried by the artifact.
pub const TRAINING_SALT_PLAN_FINGERPRINT_KEY: &str = "tritium.training_salt.plan_fingerprint";
/// Digest of the source teacher/student model before growth.
pub const TRAINING_SALT_SOURCE_MODEL_DIGEST_KEY: &str = "tritium.training_salt.source_model_digest";
/// Digest of the deterministically widened student before training.
pub const TRAINING_SALT_INITIAL_STUDENT_DIGEST_KEY: &str =
    "tritium.training_salt.initial_student_digest";
/// Last completed training step represented by the artifact.
pub const TRAINING_SALT_COMPLETED_STEP_KEY: &str = "tritium.training_salt.completed_step";
/// Canonical compact deterministic Net2Wider receipt JSON.
pub const TRAINING_SALT_GROWTH_RECEIPT_KEY: &str = "tritium.training_salt.growth_receipt_json";

const GGML_TYPE_F32: u32 = 0;

/// Replayable Net2Wider receipt embedded in a training-SALT artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingSaltGrowthReceipt {
    /// Versioned deterministic growth algorithm.
    pub algorithm: String,
    /// Intermediate width before widening.
    pub old_width: u64,
    /// Intermediate width after widening.
    pub new_width: u64,
    /// Seed used to choose replicated source units.
    pub seed: u64,
    /// Source unit for each widened unit, including the identity prefix.
    pub source_indices: Vec<u64>,
    /// Exact number of copies of each original unit.
    pub replication_counts: Vec<u64>,
    /// Base-two denominator exponent for v2 outgoing split coefficients.
    pub split_denominator_log2: Option<u32>,
    /// V2 outgoing split numerator for each widened unit.
    pub split_numerators: Option<Vec<u32>>,
}

/// Validated metadata and provenance from a self-contained training-SALT artifact.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingSaltArtifactMetadata {
    /// Architecture name, equal to the embedded HF config's `model_type`.
    pub architecture: String,
    /// Canonical compact embedded HuggingFace config JSON.
    pub hf_config_json: String,
    /// Fixed SALT row plane count in `1..=3`.
    pub salt_planes: u32,
    /// Immutable campaign-plan fingerprint.
    pub plan_fingerprint: [u8; 32],
    /// Digest of the source model before growth.
    pub source_model_digest: [u8; 32],
    /// Digest of the widened student before training.
    pub initial_student_digest: [u8; 32],
    /// Last completed training step represented by the artifact.
    pub completed_step: u64,
    /// Validated deterministic Net2Wider replay receipt.
    pub growth: TrainingSaltGrowthReceipt,
}

#[derive(Clone, Debug)]
struct ExpectedTensor {
    name: String,
    dims: Vec<u64>,
    kind: ExpectedKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedKind {
    Salt,
    F32,
}

impl ModelWeights {
    /// Load a self-contained training-SALT model from one GGUF v3 byte buffer.
    ///
    /// The loader is intentionally strict: both format markers and every
    /// provenance field are required; the embedded HF config must be canonical
    /// SwiGLU and agree with the tied/untied format marker; the tensor table must equal the canonical HF
    /// tensor set implied by that config. Quantized rows are dequantized once into
    /// exact-fp [`DenseLinear`] projections, providing a deterministic reference
    /// evaluator without consulting a model directory.
    ///
    /// # Errors
    /// Returns [`NnError`] when the GGUF, metadata, config, tensor set, tensor
    /// type/shape, SALT plane count, or payload is malformed.
    pub fn load_training_salt_gguf(bytes: &[u8]) -> Result<(ModelConfig, ModelWeights), NnError> {
        let file = read_gguf(bytes)
            .map_err(|error| NnError::Backend(format!("parse training SALT GGUF: {error}")))?;
        let metadata = parse_metadata(&file)?;
        let lm_head_tied = training_salt_head_tied(&file)?;
        let config_value: serde_json::Value = serde_json::from_str(&metadata.hf_config_json)
            .map_err(|error| NnError::MissingConfig(format!("embedded HF config: {error}")))?;
        let (config, spec) = validate_embedded_config(&metadata.hf_config_json, lm_head_tied)?;
        if metadata.architecture != config.arch {
            return Err(NnError::MissingConfig(format!(
                "general.architecture={} differs from embedded config architecture {}",
                metadata.architecture, config.arch
            )));
        }
        if metadata.growth.new_width != u64::from(config.n_ff) {
            return Err(NnError::MissingConfig(format!(
                "growth receipt new_width={} differs from intermediate_size={}",
                metadata.growth.new_width, config.n_ff
            )));
        }
        let vocab = config_value
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|&value| value != 0)
            .ok_or_else(|| NnError::MissingConfig("vocab_size".to_owned()))?;

        let planes = metadata.salt_planes as usize;

        let expected = expected_tensors(&config, vocab, spec.tied_embeddings)?;
        validate_tensor_table(&file, &expected)?;

        let parsed_salt = read_salt_gguf(bytes).map_err(|error| {
            NnError::MissingTensor(format!("parse training SALT rows: {error}"))
        })?;
        let salt_by_name: HashMap<&str, &SaltTensor> = parsed_salt
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect();
        if salt_by_name.len() != parsed_salt.len() {
            return Err(NnError::MissingTensor(
                "duplicate training SALT tensor name".to_owned(),
            ));
        }

        let mut dense = HashMap::<String, Vec<f32>>::with_capacity(expected.len());
        for tensor in &expected {
            let values = match tensor.kind {
                ExpectedKind::Salt => {
                    let salt = salt_by_name.get(tensor.name.as_str()).ok_or_else(|| {
                        NnError::MissingTensor(format!("{} (SALT payload)", tensor.name))
                    })?;
                    if salt.salt_rows.iter().any(|row| row.plane_count() != planes) {
                        return Err(NnError::MissingTensor(format!(
                            "{} has a row whose plane count differs from metadata T={planes}",
                            tensor.name
                        )));
                    }
                    salt_rows_to_dense(&salt.salt_rows).map_err(|error| {
                        NnError::MissingTensor(format!("dequantize {}: {error}", tensor.name))
                    })?
                }
                ExpectedKind::F32 => load_f32(&file, bytes, &tensor.name)?,
            };
            if values.iter().any(|value| !value.is_finite()) {
                return Err(NnError::MissingTensor(format!(
                    "{} contains a non-finite value",
                    tensor.name
                )));
            }
            dense.insert(tensor.name.clone(), values);
        }

        let provider = |name: &str| {
            dense
                .get(name)
                .cloned()
                .ok_or_else(|| NnError::MissingTensor(name.to_owned()))
        };
        let weights = build_standard_model(
            &config,
            &spec,
            NameSchema::Hf,
            |name, _expected_len| provider(name),
            |name, n_out, k_in| {
                Ok(Projection::Dense(DenseLinear::new_exact(
                    provider(name)?,
                    n_out,
                    k_in,
                )?))
            },
        )?;
        if weights.vocab != vocab {
            return Err(NnError::Shape {
                expected: vocab,
                got: weights.vocab,
            });
        }
        Ok((config, weights))
    }
}

fn require_marker(file: &GgufFile, key: &str, value: &str) -> Result<(), NnError> {
    if file.get_metadata(key).and_then(GgufValue::as_str) == Some(value) {
        Ok(())
    } else {
        Err(NnError::MissingMetadata(key.to_owned()))
    }
}

fn training_salt_head_tied(file: &GgufFile) -> Result<bool, NnError> {
    match file
        .get_metadata(TRAINING_SALT_FORMAT_KEY)
        .and_then(GgufValue::as_str)
    {
        Some(TRAINING_SALT_FORMAT_VALUE) => Ok(true),
        Some(TRAINING_SALT_UNTIED_FORMAT_VALUE) => Ok(false),
        _ => Err(NnError::MissingMetadata(
            TRAINING_SALT_FORMAT_KEY.to_owned(),
        )),
    }
}

fn validate_embedded_config(
    hf_config_json: &str,
    marker_tied: bool,
) -> Result<(ModelConfig, ArchSpec), NnError> {
    let value: serde_json::Value = serde_json::from_str(hf_config_json)
        .map_err(|error| NnError::MissingConfig(format!("embedded HF config: {error}")))?;
    let explicit_tie = value
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool);
    if value.get("hidden_act").and_then(serde_json::Value::as_str) != Some("silu")
        || explicit_tie != Some(marker_tied)
    {
        let marker = if marker_tied {
            TRAINING_SALT_FORMAT_VALUE
        } else {
            TRAINING_SALT_UNTIED_FORMAT_VALUE
        };
        return Err(NnError::MissingConfig(format!(
            "training SALT {marker} requires explicit hidden_act=silu and tie_word_embeddings={marker_tied}"
        )));
    }
    let (config, spec) = ModelConfig::from_hf_config(hf_config_json)?;
    if spec.mlp != MlpKind::SwiGlu || spec.tied_embeddings != marker_tied {
        return Err(NnError::MissingConfig(
            "training SALT format marker and embedded architecture disagree".to_owned(),
        ));
    }
    TiedSwiGluTrainingModel::validate_config(&config, &spec)
        .map_err(|error| NnError::MissingConfig(error.to_string()))?;
    Ok((config, spec))
}

fn require_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str, NnError> {
    file.get_metadata(key)
        .and_then(GgufValue::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| NnError::MissingMetadata(key.to_owned()))
}

/// Parse and validate the metadata/provenance envelope of a training-SALT GGUF.
///
/// Tensor payloads are not materialized. Campaign resume code can therefore bind
/// an existing artifact to its exact plan/model/growth evidence before deciding
/// whether evaluation may be reused.
///
/// # Errors
/// Returns [`NnError`] for malformed GGUF, wrong format markers/version, missing
/// or mistyped metadata, noncanonical JSON, malformed digests, or an inconsistent
/// Net2Wider receipt.
pub fn parse_training_salt_artifact_metadata(
    bytes: &[u8],
) -> Result<TrainingSaltArtifactMetadata, NnError> {
    let file = read_gguf(bytes)
        .map_err(|error| NnError::Backend(format!("parse training SALT GGUF: {error}")))?;
    parse_metadata(&file)
}

fn parse_metadata(file: &GgufFile) -> Result<TrainingSaltArtifactMetadata, NnError> {
    if file.version != 3 {
        return Err(NnError::MissingConfig(format!(
            "training SALT requires GGUF v3, got v{}",
            file.version
        )));
    }
    require_marker(file, SALT_GGUF_FORMAT_KEY, SALT_GGUF_FORMAT_VALUE)?;
    let lm_head_tied = training_salt_head_tied(file)?;
    if !matches!(
        file.get_metadata("general.alignment"),
        Some(GgufValue::U32(32))
    ) {
        return Err(NnError::MissingMetadata(
            "general.alignment (expected U32 32)".to_owned(),
        ));
    }
    let architecture = require_string(file, "general.architecture")?.to_owned();
    let hf_config_json = require_canonical_json(file, TRAINING_SALT_HF_CONFIG_KEY)?;
    let (config, _) = validate_embedded_config(&hf_config_json, lm_head_tied)?;
    if architecture != config.arch {
        return Err(NnError::MissingConfig(format!(
            "general.architecture={architecture} differs from embedded config architecture {}",
            config.arch
        )));
    }
    let salt_planes = match file.get_metadata(TRAINING_SALT_PLANES_KEY) {
        Some(GgufValue::U32(value)) if (1..=3).contains(value) => *value,
        _ => {
            return Err(NnError::MissingMetadata(
                TRAINING_SALT_PLANES_KEY.to_owned(),
            ));
        }
    };
    let plan_fingerprint = parse_hex_digest(file, TRAINING_SALT_PLAN_FINGERPRINT_KEY)?;
    let source_model_digest = parse_hex_digest(file, TRAINING_SALT_SOURCE_MODEL_DIGEST_KEY)?;
    let initial_student_digest = parse_hex_digest(file, TRAINING_SALT_INITIAL_STUDENT_DIGEST_KEY)?;
    let completed_step = match file.get_metadata(TRAINING_SALT_COMPLETED_STEP_KEY) {
        Some(GgufValue::U64(value)) => *value,
        _ => {
            return Err(NnError::MissingMetadata(
                TRAINING_SALT_COMPLETED_STEP_KEY.to_owned(),
            ));
        }
    };
    let growth = parse_growth_receipt(file)?;
    if growth.new_width != u64::from(config.n_ff) {
        return Err(NnError::MissingConfig(format!(
            "growth receipt new_width={} differs from intermediate_size={}",
            growth.new_width, config.n_ff
        )));
    }
    Ok(TrainingSaltArtifactMetadata {
        architecture,
        hf_config_json,
        salt_planes,
        plan_fingerprint,
        source_model_digest,
        initial_student_digest,
        completed_step,
        growth,
    })
}

fn parse_hex_digest(file: &GgufFile, key: &str) -> Result<[u8; 32], NnError> {
    let digest = require_string(file, key)?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(NnError::MissingMetadata(format!(
            "{key} (expected lowercase 64-hex)"
        )));
    }
    let mut parsed = [0_u8; 32];
    for (index, pair) in digest.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let text = core::str::from_utf8(pair).expect("validated ASCII hex");
        parsed[index] = u8::from_str_radix(text, 16).expect("validated lowercase hex");
    }
    if parsed == [0; 32] {
        return Err(NnError::MissingMetadata(format!(
            "{key} (all-zero digest is not provenance)"
        )));
    }
    Ok(parsed)
}

fn require_canonical_json(file: &GgufFile, key: &str) -> Result<String, NnError> {
    let json = require_string(file, key)?;
    let value: serde_json::Value = serde_json::from_str(json)
        .map_err(|error| NnError::MissingConfig(format!("{key}: {error}")))?;
    let canonical = serde_json::to_string(&value)
        .map_err(|error| NnError::MissingConfig(format!("{key}: {error}")))?;
    if canonical != json {
        return Err(NnError::MissingConfig(format!(
            "{key} is not canonical compact JSON"
        )));
    }
    Ok(canonical)
}

fn parse_growth_receipt(file: &GgufFile) -> Result<TrainingSaltGrowthReceipt, NnError> {
    let receipt = require_string(file, TRAINING_SALT_GROWTH_RECEIPT_KEY)?.to_owned();
    let value: serde_json::Value = serde_json::from_str(&receipt)
        .map_err(|error| NnError::MissingConfig(format!("growth receipt: {error}")))?;
    let object = value
        .as_object()
        .ok_or_else(|| NnError::MissingConfig("growth receipt object".to_owned()))?;
    let algorithm = object
        .get("algorithm")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| NnError::MissingConfig("growth receipt algorithm".to_owned()))?
        .to_owned();
    let expected_keys = match algorithm.as_str() {
        NET2WIDER_ALGORITHM_V1 => BTreeSet::from([
            "algorithm",
            "new_width",
            "old_width",
            "replication_counts",
            "seed",
            "source_indices",
        ]),
        NET2WIDER_ALGORITHM_V2 => BTreeSet::from([
            "algorithm",
            "new_width",
            "old_width",
            "replication_counts",
            "seed",
            "source_indices",
            "split_denominator_log2",
            "split_numerators",
        ]),
        _ => {
            return Err(NnError::MissingConfig(
                "growth receipt algorithm".to_owned(),
            ));
        }
    };
    let actual_keys: BTreeSet<&str> = object.keys().map(String::as_str).collect();
    if actual_keys != expected_keys {
        return Err(NnError::MissingConfig(format!(
            "growth receipt keys do not match {algorithm}"
        )));
    }
    let get_u64 = |key: &str| {
        object
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| NnError::MissingConfig(format!("growth receipt {key}")))
    };
    let old_width = get_u64("old_width")?;
    let new_width = get_u64("new_width")?;
    let seed = get_u64("seed")?;
    if old_width == 0 || new_width < old_width {
        return Err(NnError::MissingConfig(
            "growth receipt widths must satisfy 0 < old_width <= new_width".to_owned(),
        ));
    }
    let get_array = |key: &str| -> Result<Vec<u64>, NnError> {
        object
            .get(key)
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| NnError::MissingConfig(format!("growth receipt {key}")))?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .ok_or_else(|| NnError::MissingConfig(format!("growth receipt {key}")))
            })
            .collect()
    };
    let source_indices = get_array("source_indices")?;
    let replication_counts = get_array("replication_counts")?;
    let (split_denominator_log2, split_numerators) = match algorithm.as_str() {
        NET2WIDER_ALGORITHM_V1 => (None, None),
        NET2WIDER_ALGORITHM_V2 => {
            let denominator_log2 =
                u32::try_from(get_u64("split_denominator_log2")?).map_err(|_| {
                    NnError::MissingConfig(
                        "growth receipt split_denominator_log2 overflows u32".to_owned(),
                    )
                })?;
            let numerators = get_array("split_numerators")?
                .into_iter()
                .map(|value| {
                    u32::try_from(value).map_err(|_| {
                        NnError::MissingConfig(
                            "growth receipt split_numerators value overflows u32".to_owned(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            (Some(denominator_log2), Some(numerators))
        }
        _ => unreachable!("algorithm was validated above"),
    };
    let old = usize::try_from(old_width)
        .map_err(|_| NnError::MissingConfig("growth old_width overflows usize".to_owned()))?;
    let new = usize::try_from(new_width)
        .map_err(|_| NnError::MissingConfig("growth new_width overflows usize".to_owned()))?;
    if source_indices.len() != new || replication_counts.len() != old {
        return Err(NnError::MissingConfig(
            "growth receipt mapping/count lengths disagree with widths".to_owned(),
        ));
    }
    let mut actual_counts = vec![0_u64; old];
    for (index, &source) in source_indices.iter().enumerate() {
        let source = usize::try_from(source)
            .ok()
            .filter(|&source| source < old)
            .ok_or_else(|| NnError::MissingConfig("growth source index out of range".to_owned()))?;
        if index < old && source != index {
            return Err(NnError::MissingConfig(
                "growth source mapping lacks the identity prefix".to_owned(),
            ));
        }
        actual_counts[source] = actual_counts[source].checked_add(1).ok_or_else(|| {
            NnError::MissingConfig("growth replication count overflow".to_owned())
        })?;
    }
    if actual_counts != replication_counts {
        return Err(NnError::MissingConfig(
            "growth replication counts disagree with source mapping".to_owned(),
        ));
    }
    let replay = match algorithm.as_str() {
        NET2WIDER_ALGORITHM_V1 => Net2WiderPlan::replay_v1(old, new, seed),
        NET2WIDER_ALGORITHM_V2 => Net2WiderPlan::seeded(old, new, seed),
        _ => unreachable!("algorithm was validated above"),
    }
    .map_err(|error| NnError::MissingConfig(format!("growth receipt replay: {error}")))?;
    let replay_sources = replay
        .source_indices()
        .iter()
        .map(|&value| u64::try_from(value).expect("replayed source index originated from u64"))
        .collect::<Vec<_>>();
    let replay_counts = replay
        .replication_counts()
        .iter()
        .map(|&value| u64::try_from(value).expect("replayed count is bounded by a u64 width"))
        .collect::<Vec<_>>();
    if algorithm != replay.algorithm()
        || source_indices != replay_sources
        || replication_counts != replay_counts
        || split_denominator_log2 != replay.split_denominator_log2()
        || split_numerators.as_deref() != replay.split_numerators()
    {
        return Err(NnError::MissingConfig(
            "growth receipt does not match deterministic replay".to_owned(),
        ));
    }
    let canonical_source = serde_json::to_string(&source_indices)
        .map_err(|error| NnError::MissingConfig(format!("growth receipt: {error}")))?;
    let canonical_counts = serde_json::to_string(&replication_counts)
        .map_err(|error| NnError::MissingConfig(format!("growth receipt: {error}")))?;
    let canonical = match (&split_denominator_log2, &split_numerators) {
        (None, None) => format!(
            "{{\"algorithm\":\"{algorithm}\",\"old_width\":{old_width},\"new_width\":{new_width},\"seed\":{seed},\"source_indices\":{canonical_source},\"replication_counts\":{canonical_counts}}}"
        ),
        (Some(denominator_log2), Some(numerators)) => {
            let canonical_numerators = serde_json::to_string(numerators)
                .map_err(|error| NnError::MissingConfig(format!("growth receipt: {error}")))?;
            format!(
                "{{\"algorithm\":\"{algorithm}\",\"old_width\":{old_width},\"new_width\":{new_width},\"seed\":{seed},\"source_indices\":{canonical_source},\"replication_counts\":{canonical_counts},\"split_denominator_log2\":{denominator_log2},\"split_numerators\":{canonical_numerators}}}"
            )
        }
        _ => unreachable!("split metadata is parsed as an all-or-nothing pair"),
    };
    if receipt != canonical {
        return Err(NnError::MissingConfig(format!(
            "growth receipt is not canonical {algorithm} JSON"
        )));
    }
    Ok(TrainingSaltGrowthReceipt {
        algorithm,
        old_width,
        new_width,
        seed,
        source_indices,
        replication_counts,
        split_denominator_log2,
        split_numerators,
    })
}

fn expected_tensors(
    config: &ModelConfig,
    vocab: usize,
    lm_head_tied: bool,
) -> Result<Vec<ExpectedTensor>, NnError> {
    let embd = usize::try_from(config.n_embd)
        .map_err(|_| NnError::MissingConfig("hidden_size overflows usize".to_owned()))?;
    let head_dim = usize::try_from(config.head_dim())
        .map_err(|_| NnError::MissingConfig("head_dim overflows usize".to_owned()))?;
    let q_width = usize::try_from(config.n_head)
        .ok()
        .and_then(|heads| heads.checked_mul(head_dim))
        .ok_or_else(|| NnError::MissingConfig("attention width overflow".to_owned()))?;
    let kv_width = usize::try_from(config.n_head_kv)
        .ok()
        .and_then(|heads| heads.checked_mul(head_dim))
        .ok_or_else(|| NnError::MissingConfig("KV width overflow".to_owned()))?;
    let ff = usize::try_from(config.n_ff)
        .map_err(|_| NnError::MissingConfig("intermediate_size overflows usize".to_owned()))?;

    let salt = |name: String, rows: usize, cols: usize| -> Result<ExpectedTensor, NnError> {
        Ok(ExpectedTensor {
            name,
            dims: vec![
                u64::try_from(cols).map_err(|_| NnError::MissingConfig("tensor cols".into()))?,
                u64::try_from(rows).map_err(|_| NnError::MissingConfig("tensor rows".into()))?,
            ],
            kind: ExpectedKind::Salt,
        })
    };
    let norm = |name: String| ExpectedTensor {
        name,
        dims: vec![embd as u64],
        kind: ExpectedKind::F32,
    };

    let layer_count = usize::try_from(config.n_layers)
        .map_err(|_| NnError::MissingConfig("num_hidden_layers overflows usize".to_owned()))?;
    let tensor_count = layer_count
        .checked_mul(9)
        .and_then(|count| count.checked_add(2))
        .and_then(|count| count.checked_add(usize::from(!lm_head_tied)))
        .ok_or_else(|| NnError::MissingConfig("tensor count overflow".to_owned()))?;
    let mut tensors = Vec::with_capacity(tensor_count);
    tensors.push(salt("model.embed_tokens.weight".to_owned(), vocab, embd)?);
    for layer in 0..layer_count {
        let p = format!("model.layers.{layer}");
        tensors.push(salt(format!("{p}.self_attn.q_proj.weight"), q_width, embd)?);
        tensors.push(salt(
            format!("{p}.self_attn.k_proj.weight"),
            kv_width,
            embd,
        )?);
        tensors.push(salt(
            format!("{p}.self_attn.v_proj.weight"),
            kv_width,
            embd,
        )?);
        tensors.push(salt(format!("{p}.self_attn.o_proj.weight"), embd, q_width)?);
        tensors.push(salt(format!("{p}.mlp.gate_proj.weight"), ff, embd)?);
        tensors.push(salt(format!("{p}.mlp.up_proj.weight"), ff, embd)?);
        tensors.push(salt(format!("{p}.mlp.down_proj.weight"), embd, ff)?);
    }
    if !lm_head_tied {
        tensors.push(salt("lm_head.weight".to_owned(), vocab, embd)?);
    }
    for layer in 0..layer_count {
        let p = format!("model.layers.{layer}");
        tensors.push(norm(format!("{p}.input_layernorm.weight")));
        tensors.push(norm(format!("{p}.post_attention_layernorm.weight")));
    }
    tensors.push(norm("model.norm.weight".to_owned()));
    Ok(tensors)
}

fn validate_tensor_table(file: &GgufFile, expected: &[ExpectedTensor]) -> Result<(), NnError> {
    let actual_names: BTreeSet<&str> = file
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    if actual_names.len() != file.tensors.len() {
        return Err(NnError::MissingTensor(
            "duplicate tensor name in training SALT GGUF".to_owned(),
        ));
    }
    let expected_names: BTreeSet<&str> =
        expected.iter().map(|tensor| tensor.name.as_str()).collect();
    if actual_names != expected_names {
        let missing: Vec<&str> = expected_names.difference(&actual_names).copied().collect();
        let unexpected: Vec<&str> = actual_names.difference(&expected_names).copied().collect();
        return Err(NnError::MissingTensor(format!(
            "training SALT tensor set mismatch; missing={missing:?}, unexpected={unexpected:?}"
        )));
    }
    for (got, want) in file.tensors.iter().zip(expected) {
        if got.name != want.name {
            return Err(NnError::MissingTensor(format!(
                "training SALT tensor order mismatch: expected {}, got {}",
                want.name, got.name
            )));
        }
        let want_type = match want.kind {
            ExpectedKind::Salt => GGML_TYPE_TRITIUM_SALT,
            ExpectedKind::F32 => GGML_TYPE_F32,
        };
        if got.ggml_type != want_type {
            return Err(NnError::UnsupportedTensorType(got.ggml_type));
        }
        if got.dims != want.dims {
            return Err(NnError::MissingTensor(format!(
                "{} shape mismatch: expected {:?}, got {:?}",
                want.name, want.dims, got.dims
            )));
        }
    }
    Ok(())
}

fn load_f32(file: &GgufFile, bytes: &[u8], name: &str) -> Result<Vec<f32>, NnError> {
    let info = file
        .tensor(name)
        .ok_or_else(|| NnError::MissingTensor(name.to_owned()))?;
    let start = file
        .tensor_data_offset
        .checked_add(info.offset)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| NnError::MissingTensor(format!("{name} offset overflow")))?;
    let len = usize::try_from(info.n_bytes)
        .map_err(|_| NnError::MissingTensor(format!("{name} length overflow")))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| NnError::MissingTensor(format!("{name} payload end overflow")))?;
    let payload = bytes
        .get(start..end)
        .ok_or_else(|| NnError::MissingTensor(format!("{name} payload out of bounds")))?;
    if payload.len() % 4 != 0 {
        return Err(NnError::MissingTensor(format!(
            "{name} F32 payload is not word-aligned"
        )));
    }
    Ok(payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use serde_json::json;
    use tritium_format::{GgufValue, TensorOut, pack_salt_row, write_gguf};
    use tritium_quantize::export_training_salt_row;

    use super::*;
    use crate::{ModelRunner, Projection};

    #[derive(Clone)]
    struct OwnedTensor {
        name: String,
        dims: Vec<u64>,
        ggml_type: u32,
        data: Vec<u8>,
    }

    struct Fixture {
        metadata: BTreeMap<String, GgufValue>,
        tensors: Vec<OwnedTensor>,
        config: ModelConfig,
        spec: crate::ArchSpec,
        dense: HashMap<String, Vec<f32>>,
    }

    impl Fixture {
        fn bytes(&self) -> Vec<u8> {
            let tensors: Vec<TensorOut<'_>> = self
                .tensors
                .iter()
                .map(|tensor| TensorOut {
                    name: tensor.name.clone(),
                    dims: tensor.dims.clone(),
                    ggml_type: tensor.ggml_type,
                    data: &tensor.data,
                })
                .collect();
            write_gguf(3, &self.metadata, &tensors).expect("write fixture")
        }
    }

    fn replace_with_valid_f32_payload(tensor: &mut OwnedTensor) {
        let elements = tensor
            .dims
            .iter()
            .try_fold(1_usize, |product, &dim| product.checked_mul(dim as usize))
            .expect("F32 fixture shape");
        tensor.ggml_type = GGML_TYPE_F32;
        tensor.data = vec![0; elements.checked_mul(4).expect("F32 fixture bytes")];
    }

    fn append_valid_salt_row(tensor: &mut OwnedTensor) {
        assert_eq!(tensor.ggml_type, GGML_TYPE_TRITIUM_SALT);
        let rows = usize::try_from(tensor.dims[1]).expect("SALT fixture rows");
        assert!(rows > 0 && tensor.data.len().is_multiple_of(rows));
        let row_bytes = tensor.data.len() / rows;
        let extra_row = tensor.data[..row_bytes].to_vec();
        tensor.data.extend_from_slice(&extra_row);
        tensor.dims[1] += 1;
    }

    fn fixture() -> Fixture {
        fixture_with_tie(true)
    }

    fn untied_fixture() -> Fixture {
        fixture_with_tie(false)
    }

    fn fixture_with_tie(lm_head_tied: bool) -> Fixture {
        let config_json = serde_json::to_string(&json!({
            "hidden_act": "silu",
            "hidden_size": 4,
            "intermediate_size": 6,
            "max_position_embeddings": 16,
            "model_type": "llama",
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "num_key_value_heads": 1,
            "rms_norm_eps": 0.00001,
            "rope_theta": 10000.0,
            "tie_word_embeddings": lm_head_tied,
            "vocab_size": 5
        }))
        .expect("config JSON");
        let (config, spec) = ModelConfig::from_hf_config(&config_json).expect("config");
        let expected = expected_tensors(&config, 5, lm_head_tied).expect("expected tensors");
        let names = expected
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();
        let mut expected_names = vec![
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ];
        if !lm_head_tied {
            expected_names.push("lm_head.weight");
        }
        expected_names.extend([
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
        ]);
        assert_eq!(names, expected_names);

        let mut tensors = Vec::new();
        let mut dense = HashMap::new();
        for (tensor_index, tensor) in expected.into_iter().enumerate() {
            let elements = tensor
                .dims
                .iter()
                .try_fold(1_usize, |product, &dim| product.checked_mul(dim as usize))
                .expect("fixture shape");
            match tensor.kind {
                ExpectedKind::Salt => {
                    let rows = tensor.dims[1] as usize;
                    let cols = tensor.dims[0] as usize;
                    let mut data = Vec::new();
                    let mut values = Vec::with_capacity(elements);
                    for row in 0..rows {
                        let master: Vec<f32> = (0..cols)
                            .map(|col| {
                                let index = tensor_index * 17 + row * cols + col;
                                ((index as f32 + 0.25) * 0.37).sin() * 1.5 + 0.05
                            })
                            .collect();
                        let (salt, _) =
                            export_training_salt_row(&master, 2).expect("training SALT row");
                        data.extend_from_slice(&pack_salt_row(&salt).expect("pack SALT row"));
                        values.extend(salt_rows_to_dense(&[salt]).expect("dequant row"));
                    }
                    dense.insert(tensor.name.clone(), values);
                    tensors.push(OwnedTensor {
                        name: tensor.name,
                        dims: tensor.dims,
                        ggml_type: GGML_TYPE_TRITIUM_SALT,
                        data,
                    });
                }
                ExpectedKind::F32 => {
                    let values: Vec<f32> = (0..elements)
                        .map(|index| 0.9 + (tensor_index * 11 + index) as f32 * 0.001)
                        .collect();
                    let data = values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect();
                    dense.insert(tensor.name.clone(), values);
                    tensors.push(OwnedTensor {
                        name: tensor.name,
                        dims: tensor.dims,
                        ggml_type: GGML_TYPE_F32,
                        data,
                    });
                }
            }
        }

        let receipt = concat!(
            "{\"algorithm\":\"net2wider.intermediate-swiglu.splitmix64.v1\",",
            "\"old_width\":6,\"new_width\":6,\"seed\":99,",
            "\"source_indices\":[0,1,2,3,4,5],",
            "\"replication_counts\":[1,1,1,1,1,1]}"
        );
        let metadata = BTreeMap::from([
            ("general.alignment".to_owned(), GgufValue::U32(32)),
            (
                "general.architecture".to_owned(),
                GgufValue::String("llama".to_owned()),
            ),
            (
                SALT_GGUF_FORMAT_KEY.to_owned(),
                GgufValue::String(SALT_GGUF_FORMAT_VALUE.to_owned()),
            ),
            (
                TRAINING_SALT_FORMAT_KEY.to_owned(),
                GgufValue::String(
                    if lm_head_tied {
                        TRAINING_SALT_FORMAT_VALUE
                    } else {
                        TRAINING_SALT_UNTIED_FORMAT_VALUE
                    }
                    .to_owned(),
                ),
            ),
            (
                TRAINING_SALT_HF_CONFIG_KEY.to_owned(),
                GgufValue::String(config_json),
            ),
            (TRAINING_SALT_PLANES_KEY.to_owned(), GgufValue::U32(2)),
            (
                TRAINING_SALT_PLAN_FINGERPRINT_KEY.to_owned(),
                GgufValue::String("11".repeat(32)),
            ),
            (
                TRAINING_SALT_SOURCE_MODEL_DIGEST_KEY.to_owned(),
                GgufValue::String("22".repeat(32)),
            ),
            (
                TRAINING_SALT_INITIAL_STUDENT_DIGEST_KEY.to_owned(),
                GgufValue::String("33".repeat(32)),
            ),
            (
                TRAINING_SALT_COMPLETED_STEP_KEY.to_owned(),
                GgufValue::U64(17),
            ),
            (
                TRAINING_SALT_GROWTH_RECEIPT_KEY.to_owned(),
                GgufValue::String(receipt.to_owned()),
            ),
        ]);
        Fixture {
            metadata,
            tensors,
            config,
            spec,
            dense,
        }
    }

    #[test]
    fn self_contained_load_matches_manual_exact_dense_reference() {
        let fixture = fixture();
        let bytes = fixture.bytes();
        let metadata = parse_training_salt_artifact_metadata(&bytes).expect("metadata");
        assert_eq!(metadata.salt_planes, 2);
        assert_eq!(metadata.plan_fingerprint, [0x11; 32]);
        assert_eq!(metadata.source_model_digest, [0x22; 32]);
        assert_eq!(metadata.initial_student_digest, [0x33; 32]);
        assert_eq!(metadata.completed_step, 17);
        assert_eq!(metadata.growth.old_width, 6);
        assert_eq!(metadata.growth.new_width, 6);
        assert_eq!(metadata.growth.split_denominator_log2, None);
        assert_eq!(metadata.growth.split_numerators, None);

        let loaded_backend = Box::new(tritium_cpu::CpuBackend::new());
        let mut loaded =
            ModelRunner::from_training_salt_gguf(&bytes, loaded_backend).expect("load artifact");

        let provider = |name: &str| {
            fixture
                .dense
                .get(name)
                .cloned()
                .ok_or_else(|| NnError::MissingTensor(name.to_owned()))
        };
        let manual_weights = build_standard_model(
            &fixture.config,
            &fixture.spec,
            NameSchema::Hf,
            |name, _expected_len| provider(name),
            |name, rows, cols| {
                Ok(Projection::Dense(DenseLinear::new_exact(
                    provider(name)?,
                    rows,
                    cols,
                )?))
            },
        )
        .expect("manual exact model");
        let manual_backend = Box::new(tritium_cpu::CpuBackend::new());
        let mut manual =
            ModelRunner::from_weights(fixture.config.clone(), manual_weights, manual_backend);

        let loaded_logits = loaded
            .forward(&[1, 3, 2], &[0, 1, 2])
            .expect("loaded logits");
        let manual_logits = manual
            .forward(&[1, 3, 2], &[0, 1, 2])
            .expect("manual logits");
        assert_eq!(loaded_logits, manual_logits);
        assert!(
            loaded.weights.lm_head.is_none(),
            "artifact must use tied head"
        );
        match &loaded.weights.layers[0].q_proj {
            Projection::Dense(dense) => assert_eq!(
                dense.weights,
                fixture
                    .dense
                    .get("model.layers.0.self_attn.q_proj.weight")
                    .expect("q dense")
                    .as_slice()
            ),
            Projection::Salt(_)
            | Projection::Ternary(_)
            | Projection::Q2(_)
            | Projection::HostSaltV2(_) => {
                panic!("reference loader must dequantize to exact dense")
            }
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => {
                panic!("reference loader must not publish resident SALT V2")
            }
        }
    }

    #[test]
    fn parses_replayable_v2_growth_receipt() {
        let mut fixture = fixture();
        fixture.metadata.insert(
            TRAINING_SALT_GROWTH_RECEIPT_KEY.to_owned(),
            GgufValue::String(
                concat!(
                    "{\"algorithm\":\"net2wider.intermediate-swiglu.splitmix64.dyadic-unequal.v2\",",
                    "\"old_width\":4,\"new_width\":6,\"seed\":99,",
                    "\"source_indices\":[0,1,2,3,3,0],",
                    "\"replication_counts\":[2,1,1,2],",
                    "\"split_denominator_log2\":24,",
                    "\"split_numerators\":[4194304,16777216,16777216,4194304,12582912,12582912]}"
                )
                .to_owned(),
            ),
        );

        let metadata =
            parse_training_salt_artifact_metadata(&fixture.bytes()).expect("v2 metadata");
        assert_eq!(
            metadata.growth.algorithm,
            "net2wider.intermediate-swiglu.splitmix64.dyadic-unequal.v2"
        );
        assert_eq!(metadata.growth.split_denominator_log2, Some(24));
        assert_eq!(
            metadata.growth.split_numerators,
            Some(vec![
                4_194_304, 16_777_216, 16_777_216, 4_194_304, 12_582_912, 12_582_912
            ])
        );
    }

    #[test]
    fn parses_legacy_v1_widened_growth_receipt() {
        const RECEIPT: &str = concat!(
            "{\"algorithm\":\"net2wider.intermediate-swiglu.splitmix64.v1\",",
            "\"old_width\":4,\"new_width\":6,\"seed\":99,",
            "\"source_indices\":[0,1,2,3,3,0],",
            "\"replication_counts\":[2,1,1,2]}"
        );
        let mut legacy_fixture = fixture();
        legacy_fixture.metadata.insert(
            TRAINING_SALT_GROWTH_RECEIPT_KEY.to_owned(),
            GgufValue::String(RECEIPT.to_owned()),
        );

        let bytes = legacy_fixture.bytes();
        let metadata =
            parse_training_salt_artifact_metadata(&bytes).expect("legacy v1 widened metadata");
        assert_eq!(metadata.growth.algorithm, NET2WIDER_ALGORITHM_V1);
        assert_eq!(metadata.growth.old_width, 4);
        assert_eq!(metadata.growth.new_width, 6);
        assert_eq!(metadata.growth.source_indices, [0, 1, 2, 3, 3, 0]);
        assert_eq!(metadata.growth.replication_counts, [2, 1, 1, 2]);
        assert_eq!(metadata.growth.split_denominator_log2, None);
        assert_eq!(metadata.growth.split_numerators, None);
        ModelRunner::from_training_salt_gguf(&bytes, Box::new(tritium_cpu::CpuBackend::new()))
            .expect("load legacy v1 widened artifact");

        let mut tampered = fixture();
        tampered.metadata.insert(
            TRAINING_SALT_GROWTH_RECEIPT_KEY.to_owned(),
            GgufValue::String(RECEIPT.replace("\"seed\":99", "\"seed\":98")),
        );
        let error = parse_training_salt_artifact_metadata(&tampered.bytes())
            .expect_err("legacy v1 replay must bind the seed");
        assert!(error.to_string().contains("deterministic replay"));
    }

    #[test]
    fn rejects_v2_growth_receipt_tampering_and_noncanonical_json() {
        const RECEIPT: &str = concat!(
            "{\"algorithm\":\"net2wider.intermediate-swiglu.splitmix64.dyadic-unequal.v2\",",
            "\"old_width\":4,\"new_width\":6,\"seed\":99,",
            "\"source_indices\":[0,1,2,3,3,0],",
            "\"replication_counts\":[2,1,1,2],",
            "\"split_denominator_log2\":24,",
            "\"split_numerators\":[4194304,16777216,16777216,4194304,12582912,12582912]}"
        );
        let cases = [
            (
                "seed replay",
                RECEIPT.replace("\"seed\":99", "\"seed\":98"),
                "deterministic replay",
            ),
            (
                "coefficient replay",
                RECEIPT.replace(
                    "[4194304,16777216,16777216,4194304,12582912,12582912]",
                    "[4194305,16777216,16777216,4194304,12582912,12582911]",
                ),
                "deterministic replay",
            ),
            (
                "denominator",
                RECEIPT.replace(
                    "\"split_denominator_log2\":24",
                    "\"split_denominator_log2\":23",
                ),
                "deterministic replay",
            ),
            (
                "missing split field",
                RECEIPT.replace(",\"split_denominator_log2\":24", ""),
                "keys do not match",
            ),
            (
                "u32 overflow",
                RECEIPT.replacen("4194304", "4294967296", 1),
                "overflows u32",
            ),
            (
                "noncanonical key order",
                RECEIPT.replace(
                    "\"split_denominator_log2\":24,\"split_numerators\":[4194304,16777216,16777216,4194304,12582912,12582912]",
                    "\"split_numerators\":[4194304,16777216,16777216,4194304,12582912,12582912],\"split_denominator_log2\":24",
                ),
                "not canonical",
            ),
        ];

        for (name, receipt, expected_error) in cases {
            let mut fixture = fixture();
            fixture.metadata.insert(
                TRAINING_SALT_GROWTH_RECEIPT_KEY.to_owned(),
                GgufValue::String(receipt),
            );
            let error = parse_training_salt_artifact_metadata(&fixture.bytes())
                .expect_err("tampered receipt must fail");
            assert!(
                error.to_string().contains(expected_error),
                "{name}: expected {expected_error:?}, got {error}"
            );
        }
    }

    #[test]
    fn untied_format_loads_a_separate_exact_head() {
        let fixture = untied_fixture();
        let bytes = fixture.bytes();
        parse_training_salt_artifact_metadata(&bytes).expect("metadata");

        let (_, weights) =
            ModelWeights::load_training_salt_gguf(&bytes).expect("load untied artifact");
        let head = weights.lm_head.as_ref().expect("separate LM head");
        match head {
            Projection::Dense(dense) => assert_eq!(
                dense.weights.as_slice(),
                fixture
                    .dense
                    .get("lm_head.weight")
                    .expect("head dense")
                    .as_slice()
            ),
            Projection::Salt(_)
            | Projection::Ternary(_)
            | Projection::Q2(_)
            | Projection::HostSaltV2(_) => {
                panic!("reference loader must dequantize the head to exact dense")
            }
            #[cfg(feature = "cuda")]
            Projection::SaltV2(_) => {
                panic!("reference loader must not publish a resident SALT V2 head")
            }
        }
    }

    #[test]
    fn rejects_wrong_markers_planes_provenance_and_growth() {
        let mut wrong_marker = fixture();
        wrong_marker.metadata.insert(
            TRAINING_SALT_FORMAT_KEY.to_owned(),
            GgufValue::String("other".to_owned()),
        );
        assert!(ModelWeights::load_training_salt_gguf(&wrong_marker.bytes()).is_err());

        let mut bad_planes = fixture();
        bad_planes
            .metadata
            .insert(TRAINING_SALT_PLANES_KEY.to_owned(), GgufValue::U32(4));
        assert!(ModelWeights::load_training_salt_gguf(&bad_planes.bytes()).is_err());

        let mut bad_digest = fixture();
        bad_digest.metadata.insert(
            TRAINING_SALT_PLAN_FINGERPRINT_KEY.to_owned(),
            GgufValue::String("AA".repeat(32)),
        );
        assert!(parse_training_salt_artifact_metadata(&bad_digest.bytes()).is_err());

        let mut bad_growth = fixture();
        bad_growth.metadata.insert(
            TRAINING_SALT_GROWTH_RECEIPT_KEY.to_owned(),
            GgufValue::String(
                concat!(
                    "{\"algorithm\":\"net2wider.intermediate-swiglu.splitmix64.v1\",",
                    "\"old_width\":4,\"new_width\":6,\"seed\":99,",
                    "\"source_indices\":[0,1,2,3,0,1],",
                    "\"replication_counts\":[1,1,1,1]}"
                )
                .to_owned(),
            ),
        );
        assert!(parse_training_salt_artifact_metadata(&bad_growth.bytes()).is_err());
    }

    #[test]
    fn rejects_architecture_config_and_tensor_contract_mismatches() {
        let mut arch = fixture();
        arch.metadata.insert(
            "general.architecture".to_owned(),
            GgufValue::String("qwen".to_owned()),
        );
        assert!(ModelWeights::load_training_salt_gguf(&arch.bytes()).is_err());

        let mut untied = fixture();
        let mut config: serde_json::Value = serde_json::from_str(
            untied
                .metadata
                .get(TRAINING_SALT_HF_CONFIG_KEY)
                .and_then(GgufValue::as_str)
                .expect("config"),
        )
        .expect("parse config");
        config["tie_word_embeddings"] = serde_json::Value::Bool(false);
        untied.metadata.insert(
            TRAINING_SALT_HF_CONFIG_KEY.to_owned(),
            GgufValue::String(serde_json::to_string(&config).expect("config JSON")),
        );
        assert!(ModelWeights::load_training_salt_gguf(&untied.bytes()).is_err());

        let mut marker_mismatch = untied_fixture();
        marker_mismatch.metadata.insert(
            TRAINING_SALT_FORMAT_KEY.to_owned(),
            GgufValue::String(TRAINING_SALT_FORMAT_VALUE.to_owned()),
        );
        assert!(ModelWeights::load_training_salt_gguf(&marker_mismatch.bytes()).is_err());

        let mut missing_head = untied_fixture();
        missing_head
            .tensors
            .retain(|tensor| tensor.name != "lm_head.weight");
        assert!(ModelWeights::load_training_salt_gguf(&missing_head.bytes()).is_err());

        let mut reordered_head = untied_fixture();
        reordered_head.tensors.swap(7, 8);
        assert!(ModelWeights::load_training_salt_gguf(&reordered_head.bytes()).is_err());

        let mut wrong_head_type = untied_fixture();
        let head = wrong_head_type
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "lm_head.weight")
            .expect("head");
        replace_with_valid_f32_payload(head);
        assert!(matches!(
            ModelWeights::load_training_salt_gguf(&wrong_head_type.bytes()),
            Err(NnError::UnsupportedTensorType(_))
        ));

        let mut wrong_head_shape = untied_fixture();
        let head = wrong_head_shape
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "lm_head.weight")
            .expect("head");
        append_valid_salt_row(head);
        assert!(ModelWeights::load_training_salt_gguf(&wrong_head_shape.bytes()).is_err());

        let mut wrong_head_planes = untied_fixture();
        let one_plane_rows = wrong_head_planes
            .dense
            .get("lm_head.weight")
            .expect("head dense")
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|row| {
                let (salt, _) = export_training_salt_row(row, 1).expect("one-plane head row");
                pack_salt_row(&salt).expect("pack one-plane head row")
            })
            .collect();
        wrong_head_planes
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == "lm_head.weight")
            .expect("head")
            .data = one_plane_rows;
        assert!(ModelWeights::load_training_salt_gguf(&wrong_head_planes.bytes()).is_err());

        let mut reordered = fixture();
        reordered.tensors.swap(1, 2);
        assert!(ModelWeights::load_training_salt_gguf(&reordered.bytes()).is_err());

        let mut extra = fixture();
        extra.tensors.push(OwnedTensor {
            name: "lm_head.weight".to_owned(),
            dims: vec![4, 5],
            ggml_type: GGML_TYPE_F32,
            data: vec![0; 4 * 5 * 4],
        });
        assert!(ModelWeights::load_training_salt_gguf(&extra.bytes()).is_err());

        let mut bad_type = fixture();
        replace_with_valid_f32_payload(&mut bad_type.tensors[1]);
        assert!(matches!(
            ModelWeights::load_training_salt_gguf(&bad_type.bytes()),
            Err(NnError::UnsupportedTensorType(_))
        ));

        let mut bad_shape = fixture();
        append_valid_salt_row(&mut bad_shape.tensors[1]);
        assert!(ModelWeights::load_training_salt_gguf(&bad_shape.bytes()).is_err());
    }

    #[test]
    fn rejects_row_plane_mismatch_and_nonfinite_norm() {
        let mut wrong_t = fixture();
        wrong_t
            .metadata
            .insert(TRAINING_SALT_PLANES_KEY.to_owned(), GgufValue::U32(3));
        assert!(ModelWeights::load_training_salt_gguf(&wrong_t.bytes()).is_err());

        let mut nonfinite = fixture();
        let norm = nonfinite
            .tensors
            .iter_mut()
            .find(|tensor| tensor.ggml_type == GGML_TYPE_F32)
            .expect("norm");
        norm.data[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(ModelWeights::load_training_salt_gguf(&nonfinite.bytes()).is_err());
    }
}
