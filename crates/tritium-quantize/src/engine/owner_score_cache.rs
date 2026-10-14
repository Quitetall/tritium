//! Immutable one-owner/one-window content; not a fit or recipe execution receipt.
#[cfg(test)]
mod tests;
mod transport;
use super::{OwnedOwnerScoreWindow, OwnerScoreError, OwnerScoreUse, cache_scalar};
use std::sync::Arc;
use tritium_schema::{
    BlobId, OriginalSourceId, OwnerContextBinding, OwnerContextDtype, OwnerContextMeta,
    OwnerInvocationBinding, OwnerScoreCacheBinding, OwnerScoreCacheContent, OwnerScoreCacheId,
    OwnerScoreCacheLimits, OwnerScoreCacheManifest, OwnerScoreEstimator, OwnerScoreRole,
    OwnerScoreWindowMeta, QuantSourceTensor,
};

/// Cache admission/content/transport rejection; no old-profile fallback.
#[derive(Debug)]
#[non_exhaustive]
pub enum OwnerScoreCacheError {
    /// Invalid structural context, geometry or scalar encoding.
    Structure,
    /// Actual identity/inventory differs from expected content.
    Binding,
    /// Logical/physical receiving or producer admission exceeded.
    Limit,
    /// Allocation or arithmetic overflow.
    Allocation,
    /// Shared numerical validation/reconstruction failed.
    Numerical(OwnerScoreError),
    /// Common package/metadata codec failed.
    Transport(String),
    /// Common writer rejected output or I/O; a partial prefix may remain.
    Write(tritium_format::TritPackageWriteError),
}
impl core::fmt::Display for OwnerScoreCacheError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "owner cache rejected: {self:?}")
    }
}
impl std::error::Error for OwnerScoreCacheError {}
impl From<OwnerScoreError> for OwnerScoreCacheError {
    fn from(value: OwnerScoreError) -> Self {
        Self::Numerical(value)
    }
}
impl From<cache_scalar::ScalarError> for OwnerScoreCacheError {
    fn from(value: cache_scalar::ScalarError) -> Self {
        match value {
            cache_scalar::ScalarError::Allocation => Self::Allocation,
            cache_scalar::ScalarError::Length => Self::Structure,
        }
    }
}

/// Actual context and immutable observations; source declarations need verification.
#[derive(Debug)]
pub struct OwnerScoreCacheInput<'a> {
    /// Original complete source content identity.
    pub source: OriginalSourceId,
    /// Exact original matrix interpretation.
    pub tensor: QuantSourceTensor,
    /// Actual tokenizer bytes.
    pub tokenizer: &'a [u8],
    /// Actual declared scalar-objective bytes.
    pub objective: &'a [u8],
    /// Actual sorted context/invocation metadata.
    pub meta: OwnerScoreWindowMeta,
    /// Actual canonical logical context buffers, paired with metadata.
    pub context: &'a [&'a [u8]],
    /// Validated immutable numerical window, shared without another dense copy.
    pub observations: Arc<OwnedOwnerScoreWindow>,
}

/// Immutable numerical/context ownership and recomputed content binding.
#[derive(Debug)]
pub struct OwnerScoreCache {
    binding: OwnerScoreCacheBinding,
    limits: OwnerScoreCacheLimits,
    tokenizer: Vec<u8>,
    objective: Vec<u8>,
    context: Vec<Vec<u8>>,
    observations: Arc<OwnedOwnerScoreWindow>,
}
impl OwnerScoreCache {
    /// Validate all buffers/metadata and admit before copying context payloads.
    /// Numerical observations remain shared immutable storage. No recipe/source
    /// execution proof or total RSS claim is inferred from declared provenance.
    pub fn new(
        input: OwnerScoreCacheInput<'_>,
        limits: OwnerScoreCacheLimits,
    ) -> Result<Self, OwnerScoreCacheError> {
        use OwnerScoreCacheError as E;
        let OwnerScoreCacheInput {
            source,
            tensor,
            tokenizer,
            objective,
            meta,
            context,
            observations,
        } = input;
        if !limits.valid()
            || tokenizer.is_empty()
            || objective.is_empty()
            || source.as_bytes() == &[0; 32]
            || tensor.shape.len() != 2
            || tensor.byte_len().is_err()
            || tensor.shape[0] != observations.rows() as u64
            || tensor.shape[1] != observations.columns() as u64
            || meta.context.len() != context.len()
            || meta.invocations.len() != observations.use_count()
            || meta.context.is_empty()
            || meta.invocations.is_empty()
            || !meta.context.windows(2).all(|p| p[0].name < p[1].name)
            || !meta
                .invocations
                .windows(2)
                .all(|p| p[0].ordinal < p[1].ordinal)
            || meta.invocations.iter().any(|m| !m.valid())
        {
            return Err(E::Structure);
        }
        let context_bytes = context
            .iter()
            .try_fold(0u64, |n, v| n.checked_add(v.len() as u64))
            .ok_or(E::Allocation)?;
        let samples = meta
            .invocations
            .iter()
            .try_fold(0u64, |n, m| n.checked_add(m.samples))
            .ok_or(E::Allocation)?;
        let payload_bytes = context_bytes
            .checked_add(tokenizer.len() as u64)
            .and_then(|n| n.checked_add(objective.len() as u64))
            .and_then(|n| n.checked_add(observations.observation_bytes()))
            .ok_or(E::Allocation)?;
        if payload_bytes > limits.max_payload_bytes
            || context_bytes > limits.max_context_bytes
            || meta.context.len() as u64 > limits.max_context_tensors
            || meta.invocations.len() as u64 > limits.max_invocations
            || samples > limits.max_samples
        {
            return Err(E::Limit);
        }
        for (m, bytes) in meta.context.iter().zip(context) {
            validate_context(m, bytes)?;
        }
        if !meta.context.iter().any(|m| m.name == "input_ids") {
            return Err(E::Structure);
        }
        let mut invocations = cache_scalar::reserved(meta.invocations.len())?;
        for (m, usage) in meta.invocations.into_iter().zip(observations.uses()) {
            let (role, count, ins, outs) = match usage {
                OwnerScoreUse::Matmul {
                    samples,
                    inputs,
                    adjoints,
                } => (
                    OwnerScoreRole::Matmul,
                    samples,
                    cache_scalar::digest(inputs, |v| v.to_bits().to_le_bytes()),
                    cache_scalar::digest(adjoints, |v| v.to_bits().to_le_bytes()),
                ),
                OwnerScoreUse::Gather { indices, adjoints } => (
                    OwnerScoreRole::Gather,
                    indices.len(),
                    cache_scalar::digest(indices, |v| v.to_le_bytes()),
                    cache_scalar::digest(adjoints, |v| v.to_bits().to_le_bytes()),
                ),
            };
            if m.role != role || m.samples != count as u64 {
                return Err(E::Structure);
            }
            invocations.push(OwnerInvocationBinding {
                meta: m,
                input: ins,
                adjoint: outs,
            });
        }
        let content = OwnerScoreCacheContent {
            source,
            tensor,
            estimator: OwnerScoreEstimator::ScalarWindowGradient,
            tokenizer: blob(tokenizer),
            objective: blob(objective),
            context: meta
                .context
                .into_iter()
                .zip(context)
                .map(|(meta, v)| OwnerContextBinding {
                    meta,
                    content: blob(v),
                })
                .collect(),
            invocations,
        };
        let binding = OwnerScoreCacheBinding {
            cache: content_id(&content)?,
            content,
            payload_bytes,
            context_bytes,
            samples,
        };
        if !(OwnerScoreCacheManifest {
            binding: binding.clone(),
            limits,
        })
        .valid(limits)
        {
            return Err(E::Structure);
        }
        Ok(Self {
            binding,
            limits,
            tokenizer: copy(tokenizer)?,
            objective: copy(objective)?,
            context: context.iter().map(|v| copy(v)).collect::<Result<_, _>>()?,
            observations,
        })
    }
    /// Actual immutable content/counts; not an audited objective or fit.
    pub fn binding(&self) -> &OwnerScoreCacheBinding {
        &self.binding
    }
    /// Derived logical content identity, independent of physical packaging.
    pub fn id(&self) -> OwnerScoreCacheId {
        self.binding.cache
    }
    /// Exact declared objective bytes, not execution authorization/proof.
    pub fn objective(&self) -> &[u8] {
        &self.objective
    }
    /// Logical numerical observations only, excluding context/recipe/tokenizer.
    pub fn observation_bytes(&self) -> u64 {
        self.observations.observation_bytes()
    }
    /// Producer logical ceilings; excluded from content identity.
    pub fn limits(&self) -> OwnerScoreCacheLimits {
        self.limits
    }
    /// Reconstruct one f64 row using the same owned numerical implementation.
    pub fn row_score(
        &self,
        row: usize,
        max_workspace_bytes: u64,
    ) -> Result<Vec<f64>, OwnerScoreCacheError> {
        Ok(self.observations.row_score(row, max_workspace_bytes)?)
    }
}

fn content_id(content: &OwnerScoreCacheContent) -> Result<OwnerScoreCacheId, OwnerScoreCacheError> {
    let bytes = tritium_format::trit_owner_score::encode_owner_score_content(content)
        .map_err(transport_error)?;
    let mut hash = blake3::Hasher::new_derive_key("tritium.owner-score.cache.v1");
    hash.update(&bytes);
    Ok(OwnerScoreCacheId::from_bytes(*hash.finalize().as_bytes()))
}
fn blob(bytes: &[u8]) -> BlobId {
    BlobId::from_bytes(*blake3::hash(bytes).as_bytes())
}
fn copy(bytes: &[u8]) -> Result<Vec<u8>, OwnerScoreCacheError> {
    let mut result = cache_scalar::reserved(bytes.len())?;
    result.extend_from_slice(bytes);
    Ok(result)
}
fn transport_error(error: impl core::fmt::Display) -> OwnerScoreCacheError {
    OwnerScoreCacheError::Transport(error.to_string())
}

fn validate_context(meta: &OwnerContextMeta, bytes: &[u8]) -> Result<(), OwnerScoreCacheError> {
    use OwnerContextDtype as D;
    if meta.byte_len() != Some(bytes.len() as u64) {
        return Err(OwnerScoreCacheError::Structure);
    }
    let finite = match meta.dtype {
        D::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|v| half::f16::from_bits(u16::from_le_bytes(*v)).is_finite()),
        D::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|v| half::bf16::from_bits(u16::from_le_bytes(*v)).is_finite()),
        D::F32 | D::C64 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|v| f32::from_le_bytes(*v).is_finite()),
        D::F64 | D::C128 => bytes
            .as_chunks::<8>()
            .0
            .iter()
            .all(|v| f64::from_le_bytes(*v).is_finite()),
        D::Bool => bytes.iter().all(|v| *v <= 1),
        D::I8 | D::I16 | D::I32 | D::I64 | D::U8 => true,
        _ => false,
    };
    if !finite {
        return Err(OwnerScoreCacheError::Structure);
    }
    if meta.name == "input_ids"
        && (bytes.is_empty()
            || !match meta.dtype {
                D::I32 => bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|v| i32::from_le_bytes(*v) >= 0),
                D::I64 => bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .all(|v| u32::try_from(i64::from_le_bytes(*v)).is_ok()),
                _ => false,
            })
    {
        return Err(OwnerScoreCacheError::Structure);
    }
    Ok(())
}
