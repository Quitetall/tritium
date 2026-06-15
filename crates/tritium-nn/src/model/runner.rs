//! `ModelRunner`: load a GGUF model and generate tokens.
//!
//! The top of the inference spine. [`load`](ModelRunner::load) reads the config +
//! weights and picks a backend; [`forward`](ModelRunner::forward) runs one decode
//! step (embedding → 30 blocks → final norm → LM head → logits);
//! [`generate`](ModelRunner::generate) loops `forward` + sampling until `eos` or a
//! length cap. The acceptance gate (greedy IDs == transformers) lands in WF-4.

use tritium_format::GgufFile;
use tritium_spec::TernaryBackend;

use crate::config::ModelConfig;
use crate::error::NnError;
use crate::kv_cache::KvCache;
use crate::model::weights::ModelWeights;

/// A loaded model plus its per-layer KV caches and execution backend.
#[allow(missing_debug_implementations)]
pub struct ModelRunner {
    /// Model dimensions.
    pub config: ModelConfig,
    /// Loaded weights (embeddings, blocks, final norm, LM head).
    pub weights: ModelWeights,
    /// One KV cache per transformer block; length `config.n_layers`.
    pub kv: Vec<KvCache>,
    /// The execution backend for ternary GEMMs.
    pub backend: Box<dyn TernaryBackend>,
}

impl ModelRunner {
    /// Load a runner from a parsed GGUF `file` onto `backend`.
    ///
    /// Reads [`ModelConfig::from_gguf`], loads [`ModelWeights`], and allocates one
    /// [`KvCache`] per layer sized to the context length.
    ///
    /// # Errors
    /// [`NnError::MissingMetadata`] / [`NnError::MissingTensor`] /
    /// [`NnError::UnsupportedTensorType`] on a malformed file;
    /// [`NnError::BackendUnavailable`] if `backend` cannot run the model;
    /// [`NnError::Backend`] on an upload failure.
    pub fn load(file: &GgufFile, backend: Box<dyn TernaryBackend>) -> Result<Self, NnError> {
        let _ = (file, backend);
        todo!("WF-4: config + weights + per-layer KV cache + backend selection")
    }

    /// Run one decode step over `tokens` at absolute positions `positions`,
    /// returning the next-token logits `[vocab]`.
    ///
    /// `tokens` are the new token IDs to process (the full prompt on the first
    /// call, one token per step thereafter); the KV caches are advanced in place.
    ///
    /// # Errors
    /// [`NnError::Shape`] on inconsistent lengths, or [`NnError::Backend`] on a
    /// backend failure.
    pub fn forward(&mut self, tokens: &[u32], positions: &[usize]) -> Result<Vec<f32>, NnError> {
        let _ = (tokens, positions);
        todo!("WF-4: embedding → blocks → final norm → LM head → logits")
    }

    /// Greedily generate up to `max_new` tokens continuing `prompt` (token IDs),
    /// returning the generated IDs (not including the prompt). Stops early at the
    /// supplied `eos` token.
    ///
    /// # Errors
    /// [`NnError::Shape`] / [`NnError::Backend`] propagated from
    /// [`forward`](ModelRunner::forward).
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: u32,
    ) -> Result<Vec<u32>, NnError> {
        let _ = (prompt, max_new, eos);
        todo!("WF-4: prefill + greedy decode loop with KV cache")
    }
}
