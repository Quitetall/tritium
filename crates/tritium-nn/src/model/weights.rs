//! Loaded model weights, read from a GGUF file.
//!
//! Ternary weights take the I2_S → internal path ([`tritium_format`]) and are
//! uploaded to the backend as [`TernaryLinear`]; norms, the token embedding, and
//! the LM head are widened to host-side fp32 ([`crate::tensor`]). The loader maps
//! GGUF tensor names (`token_embd.weight`, `blk.N.*`, `output_norm.weight`,
//! `output.weight`) to these fields. Real loading lands in WF-4.

use tritium_format::GgufFile;
use tritium_spec::TernaryBackend;

use crate::config::ModelConfig;
use crate::error::NnError;
use crate::layers::TransformerBlock;

/// The weights for one decoder layer, ready to run.
///
/// A thin alias around [`TransformerBlock`] today; kept as a distinct loader-side
/// type so the GGUF-name → block-field mapping has a home if it grows.
pub type LayerWeights = TransformerBlock;

/// All weights for a model: embeddings, per-layer blocks, final norm, LM head.
#[allow(missing_debug_implementations)]
pub struct ModelWeights {
    /// Token embedding table, fp32, `[vocab, n_embd]` row-major.
    pub token_embd: Vec<f32>,
    /// Per-layer transformer blocks, length `n_layers`.
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm weight before the LM head; length `n_embd`.
    pub output_norm: Vec<f32>,
    /// LM head (unembedding) weight, fp32, `[vocab, n_embd]` row-major. BitNet
    /// ties this to the token embedding, but it is stored separately here.
    pub output: Vec<f32>,
}

impl ModelWeights {
    /// Load all weights from a parsed GGUF `file` per `config`, uploading ternary
    /// tensors to `backend`.
    ///
    /// # Errors
    /// - [`NnError::MissingTensor`] if a required tensor is absent.
    /// - [`NnError::UnsupportedTensorType`] if a tensor uses an unexpected ggml
    ///   type-id.
    /// - [`NnError::Backend`] if a weight upload fails.
    pub fn load(
        file: &GgufFile,
        config: &ModelConfig,
        backend: &dyn TernaryBackend,
    ) -> Result<Self, NnError> {
        let _ = (file, config, backend);
        todo!("WF-4: GGUF → ModelWeights (I2_S→internal ternary upload, fp32 norms/embeds/head)")
    }
}
