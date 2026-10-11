//! Actual runtime tokenizer, then the same fixed executed B capture controller.
use super::{StagedCalibrationModel, StagedCaptureLimits};
use crate::{HfJsonTokenizer, NnError, Tokenizer};
use tritium_quantize::engine::{CalibrationCache, StagedSalienceSource, WeightSource};

/// Input-byte limits, NOT parser/tokenizer/token/tape/source or complete RSS limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagedTextCaptureLimits {
    /// Aggregate raw UTF-8 input bytes in supplied text windows.
    pub max_text_bytes: u64,
    /// Exact tokenizer_config.json input bytes before parsing.
    pub max_tokenizer_config_bytes: u64,
}

impl StagedCalibrationModel {
    /// Encode actual ordered text windows using B's shared runtime tokenizer.
    ///
    /// No automatic BOS/EOS/chat-template insertion, truncation or fallback.
    /// Each encoded text is one next-token context (at least two tokens). Equal
    /// encoded windows produce the same token/objective caches as token-ID capture.
    /// This executes tokenization in-process but does not record/certify raw text
    /// identity or independent tokenizer/graph replay. Uses bounded temporary
    /// owned tokenizer asset copies; input-byte caps are not complete RSS limits.
    /// # Errors
    /// Foreign B, invalid/exceeded input caps, missing/malformed tokenizer assets,
    /// unused vocabulary IDs outside the model, encode or common capture failure.
    pub fn capture_next_token_text(
        &self,
        source: &mut StagedSalienceSource<'_>,
        texts: &[&str],
        text_limits: StagedTextCaptureLimits,
        capture_limits: StagedCaptureLimits,
    ) -> Result<Vec<CalibrationCache>, NnError> {
        if source.source_id() != self.source {
            return Err(reject(
                "text capture stage differs from immutable B masters",
            ));
        }
        if texts.is_empty()
            || texts.len() as u64 > capture_limits.per_cache.max_windows
            || text_limits.max_text_bytes == 0
            || text_limits.max_tokenizer_config_bytes == 0
            || capture_limits.per_cache.max_payload_bytes == 0
            || capture_limits.per_cache.max_samples == 0
            || capture_limits.max_total_payload_bytes == 0
        {
            return Err(reject("invalid staged text capture limits/windows"));
        }
        let text_bytes = texts.iter().try_fold(0u64, |sum, text| {
            sum.checked_add(text.len() as u64)
                .ok_or_else(|| reject("staged text byte count overflow"))
        })?;
        if text_bytes > text_limits.max_text_bytes {
            return Err(reject("staged text byte limit exceeded"));
        }
        let tokenizer_bytes = source
            .asset(
                "tokenizer.json",
                usize::try_from(capture_limits.per_cache.max_payload_bytes)
                    .map_err(|_| reject("staged tokenizer limit overflows usize"))?,
            )
            .map_err(|e| reject(&format!("staged text tokenizer: {e}")))?
            .into_owned();
        let config_bytes = source
            .asset(
                "tokenizer_config.json",
                usize::try_from(text_limits.max_tokenizer_config_bytes)
                    .map_err(|_| reject("staged tokenizer config limit overflows usize"))?,
            )
            .map_err(|e| reject(&format!("staged text tokenizer config: {e}")))?
            .into_owned();
        let tokenizer = HfJsonTokenizer::from_bytes(&tokenizer_bytes, &config_bytes)?;
        tokenizer.validate_model_vocab(self.arch.vocab)?;
        tokenizer.validate_capture_windows()?;
        let mut windows = Vec::new();
        let mut samples = 0u64;
        for text in texts {
            let ids = tokenizer.encode(text)?;
            if ids.len() < 2 || ids.len() - 1 > self.max_window_tokens {
                return Err(reject("invalid encoded staged text window"));
            }
            samples = samples
                .checked_add((ids.len() - 1) as u64)
                .ok_or_else(|| reject("staged text sample count overflow"))?;
            if samples > capture_limits.per_cache.max_samples {
                return Err(reject("staged text sample limit exceeded"));
            }
            windows.push(ids);
        }
        let windows: Vec<_> = windows.iter().map(Vec::as_slice).collect();
        self.capture_next_token(source, &windows, capture_limits)
    }
}

fn reject(message: &str) -> NnError {
    NnError::Tokenizer(message.into())
}
