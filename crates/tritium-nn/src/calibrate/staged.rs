//! Immutable dense B masters behind the actual shared calibration graph.
use super::{Arch, ProjectionTapFn, QkNorm, forward_observed};
use crate::{Mlp, ModelConfig, ModelWeights, NnError, Projection};
use tritium_quantize::engine::{ObservedWeightSource, StagedSalienceSource, WeightSource};
use tritium_schema::OriginalSourceId;
use tritium_train::{Tape, tape::ValueId};

mod capture;
pub use capture::StagedCaptureLimits;
mod recorded;
pub use recorded::{RecordedStagedCapture, replay_staged_capture};
mod original;
pub use original::{RecordedOriginalSalienceCapture, replay_original_salience_capture};
#[cfg(feature = "tokenizer")]
mod text;
#[cfg(feature = "tokenizer")]
pub use text::StagedTextCaptureLimits;

/// Explicit input/master limits; NOT source, tape, cache, parsed metadata or RSS limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagedCalibrationLimits {
    /// Config input bytes before JSON parsing.
    pub max_config_bytes: u64,
    /// Aggregate retained widened f32 matrices and vectors before decoding.
    pub max_dense_bytes: u64,
    /// Nonempty forward windows, also bounded by the model's context length.
    pub max_window_tokens: usize,
}

/// Original dense B weights, consumed from an admitted mandatory stage (ADR0064).
/// Immutable masters retain B's stored-byte identity; execution widens to f32.
/// The caller owns objective/backward. This does not certify captured caches,
/// hybrid/MTP support, source arithmetic replay, out-of-core fitting or quality.
pub struct StagedCalibrationModel {
    core: DenseCalibrationModel,
}

/// Actual observed original-source masters, not a transformed B or capture receipt.
/// Reuses the same admitted dense loader, immutable masters and graph as B.
/// Does not certify salience statistics, raw text, out-of-core/RSS or model quality.
pub struct SourceCalibrationModel {
    core: DenseCalibrationModel,
}

struct DenseCalibrationModel {
    source: OriginalSourceId,
    arch: Arch,
    masters: Vec<Vec<f32>>,
    max_window_tokens: usize,
    limits: StagedCalibrationLimits,
}

impl std::fmt::Debug for StagedCalibrationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedCalibrationModel")
            .field("source", &self.core.source)
            .field("layers", &self.core.arch.n_layers)
            .field("vocabulary", &self.core.arch.vocab)
            .field("max_window_tokens", &self.core.max_window_tokens)
            .finish_non_exhaustive()
    }
}

impl StagedCalibrationModel {
    /// Admit the actual B graph and move owned dense weights into immutable masters.
    /// # Errors
    /// Invalid limits/config/inventory, unsupported graph, decode or allocation failure.
    pub fn new(
        source: &mut StagedSalienceSource<'_>,
        limits: StagedCalibrationLimits,
    ) -> Result<Self, NnError> {
        DenseCalibrationModel::validate_limits(limits)?;
        let source_id = source.source_id();
        let (config, weights) = crate::model::dense_stage_weights(
            source,
            limits.max_config_bytes,
            limits.max_dense_bytes,
        )?;
        Ok(Self {
            core: DenseCalibrationModel::from_weights(source_id, config, weights, limits)?,
        })
    }

    /// Exact complete B stored-byte identity; not a PackageId.
    #[must_use]
    pub const fn source_id(&self) -> OriginalSourceId {
        self.core.source
    }

    /// Immutable graph/norm descriptor from the actual shared loader.
    #[must_use]
    pub const fn arch(&self) -> &Arch {
        &self.core.arch
    }

    /// Execute the shared graph on caller-owned tape with actual projection taps.
    /// Each window copies immutable masters into leaves; this is not RSS admission.
    /// # Errors
    /// Invalid/oversized tokens or overflowing geometry before tape mutation.
    pub fn forward_observed(
        &self,
        tape: &mut Tape,
        tokens: &[u32],
        observe: &mut ProjectionTapFn<'_>,
    ) -> Result<ValueId, NnError> {
        self.core.forward_observed(tape, tokens, observe)
    }
}

impl std::fmt::Debug for SourceCalibrationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceCalibrationModel")
            .field("source", &self.core.source)
            .field("layers", &self.core.arch.n_layers)
            .field("vocabulary", &self.core.arch.vocab)
            .field("max_window_tokens", &self.core.max_window_tokens)
            .finish_non_exhaustive()
    }
}

impl SourceCalibrationModel {
    /// Decode actual checked original reads and move weights into immutable masters.
    /// Retains no borrowed source or second full raw model. This is not capture evidence.
    /// # Errors
    /// Changed source/assets/inventory, invalid limits/config/geometry or unsupported graph.
    pub fn new(
        source: &mut ObservedWeightSource<'_>,
        limits: StagedCalibrationLimits,
    ) -> Result<Self, NnError> {
        DenseCalibrationModel::validate_limits(limits)?;
        let source_id = source.source_id();
        let architecture = source.architecture().to_owned();
        let (config, weights) = crate::model::dense_source_weights(
            source,
            &architecture,
            limits.max_config_bytes,
            limits.max_dense_bytes,
        )?;
        source
            .inventory()
            .map_err(|e| NnError::InvalidArtifact(e.to_string()))?;
        Ok(Self {
            core: DenseCalibrationModel::from_weights(source_id, config, weights, limits)?,
        })
    }

    /// Exact original stored-byte identity, preserved across execution widening.
    #[must_use]
    pub const fn source_id(&self) -> OriginalSourceId {
        self.core.source
    }

    /// Immutable admitted graph/norm descriptor.
    #[must_use]
    pub const fn arch(&self) -> &Arch {
        &self.core.arch
    }

    /// Execute original masters through the existing forward/projection graph.
    /// Caller owns objective/backward; no salience/cache receipt is fabricated.
    /// # Errors
    /// Invalid/oversized tokens or overflowing geometry before tape mutation.
    pub fn forward_observed(
        &self,
        tape: &mut Tape,
        tokens: &[u32],
        observe: &mut ProjectionTapFn<'_>,
    ) -> Result<ValueId, NnError> {
        self.core.forward_observed(tape, tokens, observe)
    }
}

impl DenseCalibrationModel {
    fn validate_limits(limits: StagedCalibrationLimits) -> Result<(), NnError> {
        if limits.max_config_bytes == 0
            || limits.max_dense_bytes == 0
            || limits.max_window_tokens == 0
        {
            return Err(NnError::Backend(
                "staged calibration limits must be positive".into(),
            ));
        }
        Ok(())
    }

    fn from_weights(
        source_id: OriginalSourceId,
        config: ModelConfig,
        weights: ModelWeights,
        limits: StagedCalibrationLimits,
    ) -> Result<Self, NnError> {
        let ModelWeights {
            token_embd,
            vocab,
            n_embd,
            layers,
            output_norm,
            lm_head: _,
        } = weights;
        let mut arch = Arch {
            attn_norms: Vec::new(),
            ffn_norms: Vec::new(),
            qk_norms: Vec::new(),
            out_norm: output_norm,
            n_embd,
            n_head: config.n_head as usize,
            n_head_kv: config.n_head_kv as usize,
            head_dim: config.head_dim() as usize,
            ff: config.n_ff as usize,
            vocab,
            eps: config.rms_eps,
            theta: config.rope_theta,
            n_layers: layers.len(),
        };
        let mut masters = vec![token_embd.into_dense()?];
        for layer in layers {
            arch.attn_norms.push(layer.attn_norm);
            arch.ffn_norms.push(layer.ffn_norm);
            arch.qk_norms.push(if layer.q_norm.is_empty() {
                None
            } else {
                Some(QkNorm {
                    query: layer.q_norm,
                    key: layer.k_norm,
                })
            });
            let Mlp::SwiGlu(mlp) = layer.mlp else {
                return Err(NnError::MissingConfig("unsupported staged MLP".into()));
            };
            for projection in [
                layer.q_proj,
                layer.k_proj,
                layer.v_proj,
                layer.o_proj,
                mlp.gate,
                mlp.up,
                mlp.down,
            ] {
                let Projection::Dense(dense) = projection else {
                    return Err(NnError::Backend(
                        "staged calibration requires dense masters".into(),
                    ));
                };
                masters.push(dense.weights);
            }
        }
        Ok(Self {
            source: source_id,
            arch,
            masters,
            max_window_tokens: limits.max_window_tokens.min(config.n_ctx as usize),
            limits,
        })
    }

    /// Execute B with actual pre-transform projection taps on the caller's tape.
    /// Each call copies masters into tape-owned leaves; this memory is not RSS-admitted.
    /// # Errors
    /// Invalid/oversized tokens or overflowing window geometry, before tape mutation.
    fn forward_observed(
        &self,
        tape: &mut Tape,
        tokens: &[u32],
        observe: &mut ProjectionTapFn<'_>,
    ) -> Result<ValueId, NnError> {
        self.validate_window(tokens)?;
        let ids: Vec<_> = self.masters.iter().map(|w| tape.leaf(w.clone())).collect();
        Ok(forward_observed(tape, &ids, &self.arch, tokens, observe))
    }

    fn validate_window(&self, tokens: &[u32]) -> Result<(), NnError> {
        if tokens.is_empty()
            || tokens.len() > self.max_window_tokens
            || tokens
                .iter()
                .any(|&id| usize::try_from(id).map_or(true, |id| id >= self.arch.vocab))
        {
            return Err(NnError::Backend(
                "invalid staged calibration token window".into(),
            ));
        }
        for width in [
            self.arch.n_embd,
            self.arch.vocab,
            self.arch.ff,
            self.arch.n_head * self.arch.head_dim,
            self.arch.n_head_kv * self.arch.head_dim,
        ] {
            tokens
                .len()
                .checked_mul(width)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| NnError::Backend("staged window geometry overflows usize".into()))?;
        }
        tokens
            .len()
            .checked_mul(tokens.len())
            .and_then(|n| n.checked_mul(self.arch.n_head))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| NnError::Backend("staged attention geometry overflows usize".into()))?;
        Ok(())
    }
}
