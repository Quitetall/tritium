//! Immutable dense B masters behind the actual shared calibration graph.
use super::{Arch, ProjectionTapFn, QkNorm, forward_observed};
use crate::{Mlp, ModelWeights, NnError, Projection};
use tritium_quantize::engine::StagedSalienceSource;
use tritium_schema::OriginalSourceId;
use tritium_train::{Tape, tape::ValueId};

mod capture;
pub use capture::StagedCaptureLimits;
mod recorded;
pub use recorded::{RecordedStagedCapture, replay_staged_capture};
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
    source: OriginalSourceId,
    arch: Arch,
    masters: Vec<Vec<f32>>,
    max_window_tokens: usize,
    limits: StagedCalibrationLimits,
}

impl std::fmt::Debug for StagedCalibrationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagedCalibrationModel")
            .field("source", &self.source)
            .field("layers", &self.arch.n_layers)
            .field("vocabulary", &self.arch.vocab)
            .field("max_window_tokens", &self.max_window_tokens)
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
        if limits.max_config_bytes == 0
            || limits.max_dense_bytes == 0
            || limits.max_window_tokens == 0
        {
            return Err(NnError::Backend(
                "staged calibration limits must be positive".into(),
            ));
        }
        let source_id = source.source_id();
        let (config, weights) = crate::model::dense_stage_weights(
            source,
            limits.max_config_bytes,
            limits.max_dense_bytes,
        )?;
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

    /// Exact complete B source identity checked by stage admission; not a PackageId.
    #[must_use]
    pub const fn source_id(&self) -> OriginalSourceId {
        self.source
    }

    /// Immutable graph/norm descriptor using the loader's actual geometry.
    #[must_use]
    pub const fn arch(&self) -> &Arch {
        &self.arch
    }

    /// Execute B with actual pre-transform projection taps on the caller's tape.
    /// Each call copies masters into tape-owned leaves; this memory is not RSS-admitted.
    /// # Errors
    /// Invalid/oversized tokens or overflowing window geometry, before tape mutation.
    pub fn forward_observed(
        &self,
        tape: &mut Tape,
        tokens: &[u32],
        observe: &mut ProjectionTapFn<'_>,
    ) -> Result<ValueId, NnError> {
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
        let ids: Vec<_> = self.masters.iter().map(|w| tape.leaf(w.clone())).collect();
        Ok(forward_observed(tape, &ids, &self.arch, tokens, observe))
    }
}
