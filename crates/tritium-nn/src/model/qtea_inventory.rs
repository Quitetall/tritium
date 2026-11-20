//! Shared metadata-only original-shape admission for QTEA builders and loaders.
use super::{
    hf::{NameSchema, declared_vocab_size, resolve_optional_attention_names},
    standard::{StandardModelPlan, TensorGeometry},
};
use crate::{ModelConfig, NnError, Qwen35CheckpointConfig, Qwen35TensorSchemaRole};
use std::collections::BTreeMap;

/// Explicit model profile; never inferred from unknown tensor names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QteaModelProfile {
    /// Existing homogeneous transformer architecture/variation plan.
    Standard,
    /// Complete supported Qwen-family language/MTP inventory; vision deferred.
    Qwen35Hybrid,
}
pub(super) struct StandardInventory {
    pub(super) config: ModelConfig,
    #[cfg(feature = "host-qtea")]
    pub(super) spec: crate::ArchSpec,
    #[cfg(feature = "host-qtea")]
    pub(super) plan: StandardModelPlan,
    #[cfg(feature = "host-qtea")]
    pub(super) vocab: usize,
}
pub(super) enum PreparedGraph {
    Standard(Box<StandardInventory>),
    Hybrid(Box<Qwen35CheckpointConfig>),
}
/// Admitted complete original names/shapes, not payload/source/runtime qualification.
/// Rank-two entries are fitted matrices in these explicit profiles; other entries
/// are preserved state. No raw bytes, numeric arrays or machine plans are owned.
pub struct QteaModelInventory {
    pub(super) graph: PreparedGraph,
    profile: QteaModelProfile,
    tensors: BTreeMap<String, Vec<u64>>,
}
impl core::fmt::Debug for QteaModelInventory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QteaModelInventory")
            .field("profile", &self.profile)
            .field("tensors", &self.tensors.len())
            .finish_non_exhaustive()
    }
}
fn invalid(message: &str) -> NnError {
    NnError::InvalidArtifact(message.into())
}
impl QteaModelInventory {
    #[cfg(feature = "host-qtea")]
    pub(super) fn from_manifest(
        config: &[u8],
        profile: QteaModelProfile,
        manifest: &tritium_format::qtea::model::QteaModelManifest,
        max_owners: usize,
    ) -> Result<Self, NnError> {
        use tritium_format::qtea::model::QteaModelTensor;
        let shapes = manifest
            .tensors
            .iter()
            .map(|(name, tensor)| {
                (
                    name.clone(),
                    match tensor {
                        QteaModelTensor::Qtea { rows, columns, .. } => vec![*rows, *columns],
                        QteaModelTensor::Vector { length, .. } => vec![*length],
                        QteaModelTensor::Preserved { shape, .. } => shape.clone(),
                    },
                )
            })
            .collect();
        let inventory = Self::prepare(config, profile, shapes, max_owners)?;
        for (name, shape) in &inventory.tensors {
            let matches = match &manifest.tensors[name] {
                QteaModelTensor::Qtea { .. } => shape.len() == 2,
                QteaModelTensor::Vector { .. } => shape.len() == 1,
                QteaModelTensor::Preserved { .. } => {
                    profile == QteaModelProfile::Qwen35Hybrid && shape.len() > 2
                }
            };
            if !matches {
                return Err(invalid("QTEA original role/shape mismatch"));
            }
        }
        Ok(inventory)
    }
    /// Admit source geometry through the same architecture plans as cold loading.
    /// Input metadata is caller-owned; config is bounded by the existing global
    /// config ceiling and plan growth by actual capped owner count. Shapes alone
    /// do not validate numeric bytes, original precision/custody or runtime fit.
    /// # Errors
    /// Invalid config/profile, empty/oversized/incomplete names/shapes or geometry.
    pub fn prepare(
        config: &[u8],
        profile: QteaModelProfile,
        tensors: BTreeMap<String, Vec<u64>>,
        max_owners: usize,
    ) -> Result<Self, NnError> {
        if tensors.is_empty()
            || tensors.len() > max_owners
            || config.len() as u64 > super::hf::MAX_CONFIG_JSON_BYTES
        {
            return Err(invalid("QTEA source metadata exceeds owner/config caps"));
        }
        let text = std::str::from_utf8(config).map_err(|e| invalid(&e.to_string()))?;
        let graph = match profile {
            QteaModelProfile::Standard => {
                let (config, mut spec) = ModelConfig::from_hf_config(text)?;
                let minimum = (config.n_layers as usize)
                    .checked_mul(9)
                    .and_then(|n| n.checked_add(2))
                    .ok_or_else(|| invalid("QTEA architecture inventory overflow"))?;
                if minimum > tensors.len() {
                    return Err(invalid("QTEA architecture exceeds owner inventory"));
                }
                let json = serde_json::from_str(text).map_err(|e| invalid(&e.to_string()))?;
                (spec.qkv_bias, spec.qk_norm) = resolve_optional_attention_names(
                    &config,
                    &json,
                    tensors.keys().map(String::as_str),
                )?;
                let shape = tensors
                    .get(NameSchema::Hf.top("token_embd"))
                    .ok_or_else(|| invalid("QTEA source embedding absent"))?;
                let [rows, cols] = shape.as_slice() else {
                    return Err(invalid("QTEA embedding must be a matrix"));
                };
                let vocab = usize::try_from(*rows).map_err(|e| invalid(&e.to_string()))?;
                if *cols != u64::from(config.n_embd)
                    || declared_vocab_size(&json)?.is_some_and(|v| v != vocab)
                {
                    return Err(invalid("QTEA embedding/config geometry mismatch"));
                }
                let plan = StandardModelPlan::new(&config, &spec, NameSchema::Hf, vocab)?;
                plan.admit_inventory(tensors.len(), |name| {
                    tensors.get(name).map(|shape| TensorGeometry::Stored(shape))
                })?;
                PreparedGraph::Standard(Box::new(StandardInventory {
                    config,
                    #[cfg(feature = "host-qtea")]
                    spec,
                    #[cfg(feature = "host-qtea")]
                    plan,
                    #[cfg(feature = "host-qtea")]
                    vocab,
                }))
            }
            QteaModelProfile::Qwen35Hybrid => {
                let config = Qwen35CheckpointConfig::from_hf_config(text)?;
                let minimum = config
                    .text
                    .layer_types
                    .len()
                    .checked_mul(7)
                    .and_then(|n| n.checked_add(17))
                    .ok_or_else(|| invalid("hybrid schema size overflow"))?;
                if minimum > tensors.len() {
                    return Err(invalid(
                        "hybrid architecture exceeds capped owner inventory",
                    ));
                }
                let schema = crate::qwen35_language_mtp_tensor_schema(&config.text)?;
                if schema.len() != tensors.len() {
                    return Err(invalid(
                        "hybrid QTEA requires complete exact language/MTP owner inventory",
                    ));
                }
                for entry in schema {
                    if (entry.role == Qwen35TensorSchemaRole::Matrix) != (entry.shape.len() == 2) {
                        return Err(invalid("unsupported hybrid source role geometry"));
                    }
                    let actual = tensors
                        .get(&entry.name)
                        .ok_or_else(|| invalid("missing canonical hybrid QTEA owner"))?;
                    if !actual
                        .iter()
                        .copied()
                        .eq(entry.shape.iter().map(|d| *d as u64))
                    {
                        return Err(invalid("hybrid QTEA original role/shape mismatch"));
                    }
                }
                PreparedGraph::Hybrid(Box::new(config))
            }
        };
        Ok(Self {
            graph,
            profile,
            tensors,
        })
    }
    /// Complete immutable original names and shapes, not raw-content bindings.
    #[must_use]
    pub fn tensors(&self) -> &BTreeMap<String, Vec<u64>> {
        &self.tensors
    }
    /// Actual explicitly selected profile.
    #[must_use]
    pub const fn profile(&self) -> QteaModelProfile {
        self.profile
    }
    /// Configured architecture label, not authenticated original-checkpoint identity.
    #[must_use]
    pub fn architecture_name(&self) -> &str {
        match &self.graph {
            PreparedGraph::Standard(p) => &p.config.arch,
            PreparedGraph::Hybrid(c) => &c.model_type,
        }
    }
}
