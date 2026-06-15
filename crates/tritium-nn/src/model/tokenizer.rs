//! The tokenizer contract.
//!
//! v0.20 tokenizes in Python (HF LLaMA-3 BPE) and commits pre-tokenized prompt
//! fixtures, so the offline Rust lane needs no tokenizer. This trait is the seam
//! a native Rust tokenizer plugs into at v0.80; the runner depends only on it.

use crate::error::NnError;

/// Encode text to token IDs and back.
///
/// Implementations are model-specific (BitNet uses the LLaMA-3 BPE vocabulary).
pub trait Tokenizer {
    /// Encode `text` to a sequence of token IDs.
    ///
    /// # Errors
    /// [`NnError::Tokenizer`] if `text` cannot be encoded.
    fn encode(&self, text: &str) -> Result<Vec<u32>, NnError>;

    /// Decode a sequence of token IDs back to text.
    ///
    /// # Errors
    /// [`NnError::Tokenizer`] if `tokens` contains an out-of-vocabulary ID or
    /// decodes to invalid UTF-8.
    fn decode(&self, tokens: &[u32]) -> Result<String, NnError>;

    /// The beginning-of-sequence token ID.
    fn bos(&self) -> u32;

    /// The end-of-sequence token ID (generation stops when it is produced).
    fn eos(&self) -> u32;
}
