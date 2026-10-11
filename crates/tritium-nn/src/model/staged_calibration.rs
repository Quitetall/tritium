//! Actual admitted B payloads use the existing standard graph admission/assembler.
use std::{cell::RefCell, collections::BTreeMap};
use tritium_format::trit_model::decode_dense_f32;
use tritium_quantize::engine::{StagedSalienceSource, WeightSource};

use super::{
    ModelWeights,
    hf::{NameSchema, declared_vocab_size, resolve_optional_attention_names},
    standard::{StandardModelPlan, TensorGeometry},
};
use crate::{DenseLinear, MlpKind, ModelConfig, NnError, Projection, TokenEmbedding};

pub(crate) fn dense_stage_weights(
    source: &mut StagedSalienceSource<'_>,
    max_config_bytes: u64,
    max_dense_bytes: u64,
) -> Result<(ModelConfig, ModelWeights), NnError> {
    let maximum = usize::try_from(max_config_bytes)
        .map_err(|_| NnError::MissingConfig("config limit exceeds usize".into()))?;
    let bytes = source
        .asset("config.json", maximum)
        .map_err(|e| NnError::InvalidArtifact(e.to_string()))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| NnError::MissingConfig(format!("config.json: {e}")))?;
    let (config, mut spec) = ModelConfig::from_hf_config(text)?;
    let json = serde_json::from_str(text)
        .map_err(|e| NnError::MissingConfig(format!("config.json: {e}")))?;
    drop(bytes);
    if source.manifest().observation.plan.architecture != config.arch {
        return Err(NnError::MissingConfig(
            "stage/config architecture mismatch".into(),
        ));
    }
    let inventory = source
        .inventory()
        .map_err(|e| NnError::InvalidArtifact(e.to_string()))?;
    // A necessary coarse coverage bound prevents a tiny inventory from causing
    // a config-controlled enormous layer-metadata allocation. Exact coverage
    // remains owned by StandardModelPlan, not a copied tensor-count formula.
    if config.n_layers as usize > inventory.len() {
        return Err(NnError::MissingTensor(
            "declared layers exceed stage inventory".into(),
        ));
    }
    let inventory: BTreeMap<_, _> = inventory.iter().map(|t| (t.name.as_str(), t)).collect();
    (spec.qkv_bias, spec.qk_norm) =
        resolve_optional_attention_names(&config, &json, inventory.keys().copied())?;
    if spec.mlp != MlpKind::SwiGlu
        || !spec.tied_embeddings
        || spec.qkv_bias
        || spec.attn_sub_norm
        || spec.ffn_sub_norm
    {
        return Err(NnError::MissingConfig(
            "staged calibration graph requires tied SwiGLU without bias or sub-norms".into(),
        ));
    }
    let embedding = inventory
        .get(NameSchema::Hf.top("token_embd"))
        .ok_or_else(|| NnError::MissingTensor("embedding absent".into()))?;
    let [rows, cols] = embedding.shape.as_slice() else {
        return Err(NnError::MissingTensor("embedding must be a matrix".into()));
    };
    let vocab = usize::try_from(*rows)
        .map_err(|_| NnError::MissingTensor("vocabulary exceeds usize".into()))?;
    if usize::try_from(*cols) != Ok(config.n_embd as usize)
        || declared_vocab_size(&json)?.is_some_and(|n| n != vocab)
    {
        return Err(NnError::MissingTensor(
            "embedding geometry disagrees with config".into(),
        ));
    }
    let plan = StandardModelPlan::new(&config, &spec, NameSchema::Hf, vocab)?;
    plan.admit_inventory(inventory.len(), |name| {
        inventory
            .get(name)
            .map(|t| TensorGeometry::Stored(&t.shape))
    })?;
    let overflow = || NnError::Backend("decoded dense size overflows u64".into());
    let decoded = plan.tensors().try_fold(0u64, |total, request| {
        let bytes = request.shape.iter().try_fold(4u64, |n, &d| {
            n.checked_mul(u64::try_from(d).map_err(|_| overflow())?)
                .ok_or_else(overflow)
        })?;
        total.checked_add(bytes).ok_or_else(overflow)
    })?;
    if decoded > max_dense_bytes {
        return Err(NnError::Backend(
            "stage exceeds decoded dense byte limit".into(),
        ));
    }
    let source = RefCell::new(source);
    let values = |name: &str| -> Result<Vec<f32>, NnError> {
        let meta = inventory
            .get(name)
            .ok_or_else(|| NnError::MissingTensor(name.into()))?;
        let maximum = usize::try_from(
            meta.byte_len()
                .map_err(|e| NnError::InvalidArtifact(e.to_string()))?,
        )
        .map_err(|_| NnError::Backend("stage tensor exceeds usize".into()))?;
        let mut source = source.borrow_mut();
        let bytes = source
            .tensor(name, maximum)
            .map_err(|e| NnError::InvalidArtifact(e.to_string()))?;
        decode_dense_f32(name, &meta.shape, meta.precision, &bytes)
            .map_err(|e| NnError::InvalidArtifact(e.to_string()))
    };
    let embedding =
        TokenEmbedding::from_dense(values(&plan.embedding.name)?, vocab, config.n_embd as usize)?;
    let weights = plan.assemble(
        &config,
        &spec,
        embedding,
        |name, _| values(name),
        |name, rows, cols| DenseLinear::new_exact(values(name)?, rows, cols).map(Projection::Dense),
    )?;
    Ok((config, weights))
}
