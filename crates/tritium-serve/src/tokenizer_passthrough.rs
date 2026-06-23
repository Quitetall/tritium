//! A token-ID passthrough [`Tokenizer`] for the MVP / contract lane.
//!
//! Tritium has no in-repo BPE yet ([`tritium_nn::Tokenizer`] is a seam with no
//! implementation; v0.20 tokenized in Python). So v0.80 `serve` ships this
//! passthrough: `encode` parses whitespace-separated integer token IDs, `decode`
//! renders IDs space-separated. This unblocks the OpenAI wire contract + the
//! model-free contract tests + a token-ID e2e. Real LLaMA-3 BPE is a separate
//! `Tokenizer`-seam task — inject a `tokenizers`-crate-backed impl for text input.

use tritium_nn::{NnError, Tokenizer};

/// Default LLaMA-3 / BitNet special-token IDs.
const DEFAULT_BOS: u32 = 128_000;
const DEFAULT_EOS: u32 = 128_001;

/// A [`Tokenizer`] that treats text as whitespace-separated token IDs.
#[derive(Debug, Clone)]
pub struct IdPassthroughTokenizer {
    bos: u32,
    eos: u32,
}

impl Default for IdPassthroughTokenizer {
    fn default() -> Self {
        Self {
            bos: DEFAULT_BOS,
            eos: DEFAULT_EOS,
        }
    }
}

impl IdPassthroughTokenizer {
    /// Construct with explicit BOS/EOS IDs.
    #[must_use]
    pub fn new(bos: u32, eos: u32) -> Self {
        Self { bos, eos }
    }
}

impl Tokenizer for IdPassthroughTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, NnError> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }
        trimmed
            .split_whitespace()
            .map(|tok| {
                tok.parse::<u32>().map_err(|_| {
                    NnError::Tokenizer(format!(
                        "id-passthrough tokenizer expects whitespace-separated integer token IDs; \
                         got non-numeric {tok:?}. Inject a BPE-backed Tokenizer for text input."
                    ))
                })
            })
            .collect()
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, NnError> {
        let mut out = String::new();
        for (i, t) in tokens.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&t.to_string());
        }
        Ok(out)
    }

    fn bos(&self) -> u32 {
        self.bos
    }
    fn eos(&self) -> u32 {
        self.eos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ids() {
        let t = IdPassthroughTokenizer::default();
        assert_eq!(t.encode("10 11 12").unwrap(), vec![10, 11, 12]);
        assert_eq!(t.decode(&[10, 11, 12]).unwrap(), "10 11 12");
        assert_eq!(t.encode("").unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn rejects_nonnumeric() {
        let t = IdPassthroughTokenizer::default();
        assert!(matches!(t.encode("hello"), Err(NnError::Tokenizer(_))));
    }
}
