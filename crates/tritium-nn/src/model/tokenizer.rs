//! The tokenizer contract.
//!
//! v0.20 tokenizes in Python (HF LLaMA-3 BPE) and commits pre-tokenized prompt
//! fixtures, so the offline Rust lane needs no tokenizer. This trait is the seam
//! a native Rust tokenizer plugs into at v0.80; the runner depends only on it.

use crate::error::NnError;

/// Typed inputs to the source chat template, not guarantees about model behavior.
///
/// Default omits optional variables, preserving the source's undefined-variable
/// behavior. Explicit controls must not silently fall back to a legacy formatter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct ChatRenderOptions {
    enable_thinking: Option<bool>,
}

impl ChatRenderOptions {
    /// Supply the exact `enable_thinking` boolean to the source template.
    /// A template may ignore it; this does not enforce an output policy.
    #[must_use]
    pub const fn with_thinking(mut self, enabled: bool) -> Self {
        self.enable_thinking = Some(enabled);
        self
    }

    /// Explicit source input, or `None` when the variable must remain undefined.
    #[must_use]
    pub const fn enable_thinking(self) -> Option<bool> {
        self.enable_thinking
    }
}

/// Encode text to token IDs and back.
///
/// Implementations are model-specific (BitNet uses the LLaMA-3 BPE vocabulary).
pub trait Tokenizer {
    /// Render a generation prompt using this tokenizer's source-bound template.
    ///
    /// `None` means no source template exists; only that case permits an explicit
    /// legacy formatter. A declared but invalid/unsupported template returns an
    /// error, never `None`. Implementations must respect `max_output_bytes`.
    /// Text-role messages only; this does not imply tools or multimodal support.
    ///
    /// # Errors
    /// [`NnError::Tokenizer`] for invalid input, unsupported syntax or limits.
    fn render_chat_prompt(
        &self,
        _messages: &[(&str, &str)],
        _max_output_bytes: usize,
    ) -> Result<Option<String>, NnError> {
        Ok(None)
    }

    /// Render with typed source-template controls, without rewriting its text.
    ///
    /// Defaults delegate to the existing method, including its absence/error
    /// semantics. Adapters must explicitly support nondefault controls; they
    /// cannot return absence and silently lose a requested control in fallback.
    ///
    /// # Errors
    /// [`NnError::Tokenizer`] for unsupported controls or original render errors.
    fn render_chat_prompt_with_options(
        &self,
        messages: &[(&str, &str)],
        options: ChatRenderOptions,
        max_output_bytes: usize,
    ) -> Result<Option<String>, NnError> {
        if options != ChatRenderOptions::default() {
            return Err(NnError::Tokenizer(
                "explicit source chat controls are unsupported by this tokenizer".into(),
            ));
        }
        self.render_chat_prompt(messages, max_output_bytes)
    }

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

    /// Every end-of-sequence token id. Newer chat models (Qwen3-family) stop
    /// on several ids; the default wraps [`eos`](Self::eos) so existing
    /// implementors keep working. Generation loops should prefer this.
    fn eos_ids(&self) -> Vec<u32> {
        vec![self.eos()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct OldAdapter(Cell<usize>);
    impl Tokenizer for OldAdapter {
        fn render_chat_prompt(
            &self,
            _: &[(&str, &str)],
            _: usize,
        ) -> Result<Option<String>, NnError> {
            self.0.set(self.0.get() + 1);
            Ok(Some("unchanged".into()))
        }
        fn encode(&self, _: &str) -> Result<Vec<u32>, NnError> {
            Ok(Vec::new())
        }
        fn decode(&self, _: &[u32]) -> Result<String, NnError> {
            Ok(String::new())
        }
        fn bos(&self) -> u32 {
            0
        }
        fn eos(&self) -> u32 {
            0
        }
    }

    #[test]
    fn default_options_delegate_through_existing_trait_object_adapter() {
        let adapter = OldAdapter(Cell::new(0));
        let tokenizer: &dyn Tokenizer = &adapter;
        assert_eq!(
            tokenizer
                .render_chat_prompt_with_options(
                    &[("user", "Hi")],
                    ChatRenderOptions::default(),
                    64
                )
                .unwrap(),
            Some("unchanged".into())
        );
        assert_eq!(adapter.0.get(), 1);
        assert_eq!(ChatRenderOptions::default().enable_thinking(), None);
    }

    #[test]
    fn explicit_unsupported_controls_do_not_call_legacy_rendering() {
        let adapter = OldAdapter(Cell::new(0));
        let tokenizer: &dyn Tokenizer = &adapter;
        for enabled in [true, false] {
            let options = ChatRenderOptions::default().with_thinking(enabled);
            assert_eq!(options.enable_thinking(), Some(enabled));
            assert!(
                tokenizer
                    .render_chat_prompt_with_options(&[("user", "Hi")], options, 64)
                    .is_err()
            );
        }
        assert_eq!(adapter.0.get(), 0);
    }
}
