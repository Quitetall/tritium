//! Executed B capture vocabulary; structural agreement is not graph replay.
use crate::{
    BlobId, CalibrationCacheBinding, CalibrationCacheLimits, CalibrationCacheManifest,
    CalibrationConsumer, CalibrationOutput, EvidenceReference, OriginalSourceId, PackageId,
    QuantSourceTensor,
};
use alloc::{string::String, vec::Vec};

/// The fixed actually executed objective and widened-f32 graph profile.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
pub enum CalibrationCaptureProfile {
    /// Per-window mean next-token CE, sparse k1 labels, unit weights/true masks.
    NextTokenWindowMeanK1F32V1,
}

/// Logical model/input/cache bounds, NOT tape/parser/package/evidence/full RSS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationCaptureLimits {
    /// Actual config input-byte cap before model parsing.
    pub max_config_bytes: u64,
    /// Retained widened-f32 master matrix/vector cap before decoding.
    pub max_dense_bytes: u64,
    /// Effective maximum input tokens per context (excluding final label).
    pub max_window_tokens: u64,
    /// Existing per-cache payload/sample/window admission.
    pub per_cache: CalibrationCacheLimits,
    /// Aggregate retained payload, including duplicated context bytes.
    pub max_total_payload_bytes: u64,
    /// Aggregate raw UTF-8 text bytes; zero for pretokenized capture.
    pub max_text_bytes: u64,
    /// tokenizer_config.json bytes; zero for pretokenized capture.
    pub max_tokenizer_config_bytes: u64,
}
impl CalibrationCaptureLimits {
    /// Check each recorded logical bound against a receiver's explicit admission.
    #[must_use]
    pub fn admitted_by(&self, receiver: &Self) -> bool {
        self.max_config_bytes <= receiver.max_config_bytes
            && self.max_dense_bytes <= receiver.max_dense_bytes
            && self.max_window_tokens <= receiver.max_window_tokens
            && self.per_cache.max_payload_bytes <= receiver.per_cache.max_payload_bytes
            && self.per_cache.max_samples <= receiver.per_cache.max_samples
            && self.per_cache.max_windows <= receiver.per_cache.max_windows
            && self.max_total_payload_bytes <= receiver.max_total_payload_bytes
            && self.max_text_bytes <= receiver.max_text_bytes
            && self.max_tokenizer_config_bytes <= receiver.max_tokenizer_config_bytes
    }
}

/// Exact ordered private calibration inputs, not an asserted input digest.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub enum CalibrationCaptureInput {
    /// Pretokenized windows, each including its final target.
    Tokens {
        /// Ordered complete token-ID windows.
        windows: Vec<Vec<u32>>,
    },
    /// Actual runtime encoding: raw text, no special tokens/padding/truncation.
    RawText {
        /// Exact UTF-8 windows; private evidence may contain private corpus text.
        windows: Vec<String>,
    },
}

/// Controller-owned actual B capture request, not caller provenance assertions.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationCapturePlan {
    /// Actual admitted dense B package.
    pub stage_package: PackageId,
    /// Actual complete stored-byte B source, not widened-f32 identity.
    pub source: OriginalSourceId,
    /// Actual mandatory A->B transform log reference.
    pub transform: EvidenceReference,
    /// Exact admitted tokenizer.json bytes.
    pub tokenizer: BlobId,
    /// Exact admitted tokenizer_config.json bytes for text; absent for token IDs.
    pub tokenizer_config: Option<BlobId>,
    /// Explicit actually executed objective and numerical graph profile.
    pub profile: CalibrationCaptureProfile,
    /// Actual ordered projection interpretations, followed by tied head.
    pub tensors: Vec<QuantSourceTensor>,
    /// Exact actual private inputs, including context boundaries/final labels.
    pub input: CalibrationCaptureInput,
    /// Actual positive logical admission; not total memory qualification.
    pub limits: CalibrationCaptureLimits,
}

/// Distinct registered root request.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationCaptureStarted {
    /// Actual complete controller-owned request.
    pub plan: CalibrationCapturePlan,
}

/// Actual immutable cache outputs; integrity alone does not prove execution.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct CalibrationCaptureCompleted {
    /// Actual encoded windows, including the final labels.
    pub windows: Vec<Vec<u32>>,
    /// Complete actual ordered cache bindings, not just output digest assertions.
    pub caches: Vec<CalibrationCacheBinding>,
}

impl CalibrationCapturePlan {
    /// Structural request admission, not complete B/graph/capture truth.
    #[must_use]
    pub fn valid(&self) -> bool {
        let l = self.limits;
        let count = match &self.input {
            CalibrationCaptureInput::Tokens { windows } => {
                if self.tokenizer_config.is_some()
                    || l.max_text_bytes != 0
                    || l.max_tokenizer_config_bytes != 0
                    || windows
                        .iter()
                        .any(|w| w.len() < 2 || w.len() as u64 - 1 > l.max_window_tokens)
                {
                    return false;
                }
                windows.len()
            }
            CalibrationCaptureInput::RawText { windows } => {
                let bytes = windows
                    .iter()
                    .try_fold(0u64, |n, w| n.checked_add(w.len() as u64));
                if self.tokenizer_config.is_none()
                    || l.max_text_bytes == 0
                    || l.max_tokenizer_config_bytes == 0
                    || bytes.is_none_or(|n| n > l.max_text_bytes)
                {
                    return false;
                }
                windows.len()
            }
        };
        self.stage_package.as_bytes() != &[0; 32]
            && self.source.as_bytes() != &[0; 32]
            && self.tokenizer.as_bytes() != &[0; 32]
            && self
                .tokenizer_config
                .is_none_or(|v| v.as_bytes() != &[0; 32])
            && !self.transform.run.is_empty()
            && !self.transform.span.is_empty()
            && self.transform.root != [0; 32]
            && l.max_config_bytes > 0
            && l.max_dense_bytes > 0
            && l.max_window_tokens > 0
            && l.per_cache.max_payload_bytes > 0
            && l.per_cache.max_samples > 0
            && count > 0
            && count as u64 <= l.per_cache.max_windows
            && l.max_total_payload_bytes > 0
            && !self.tensors.is_empty()
            && self.tensors.iter().enumerate().all(|(i, t)| {
                t.shape.len() == 2
                    && t.byte_len().is_ok()
                    && !self.tensors[..i].iter().any(|v| v.name == t.name)
            })
    }
}

impl CalibrationCaptureCompleted {
    /// Check complete ordered structural context; actual cache bytes need replay.
    #[must_use]
    pub fn agrees_with(&self, plan: &CalibrationCapturePlan) -> bool {
        if !plan.valid() || self.windows.is_empty() || self.caches.len() != plan.tensors.len() {
            return false;
        }
        let input_count = match &plan.input {
            CalibrationCaptureInput::Tokens { windows } => {
                if windows != &self.windows {
                    return false;
                }
                windows.len()
            }
            CalibrationCaptureInput::RawText { windows } => windows.len(),
        };
        if input_count != self.windows.len()
            || self
                .windows
                .iter()
                .any(|w| w.len() < 2 || w.len() as u64 - 1 > plan.limits.max_window_tokens)
        {
            return false;
        }
        let window_samples: Vec<_> = self.windows.iter().map(|w| w.len() as u64 - 1).collect();
        let samples = window_samples
            .iter()
            .try_fold(0u64, |n, w| n.checked_add(*w));
        let total = self
            .caches
            .iter()
            .try_fold(0u64, |n, b| n.checked_add(b.payload_bytes));
        total.is_some_and(|n| n <= plan.limits.max_total_payload_bytes)
            && self.caches.iter().zip(&plan.tensors).all(|(b, t)| {
                b.request.tensor == *t
                    && b.request.model == plan.source
                    && b.request.consumer == CalibrationConsumer::Matmul
                    && b.request.output == CalibrationOutput::DenseFisher
                    && b.request.sample_start == 0
                    && b.request.limits == plan.limits.per_cache
                    && samples == Some(b.samples)
                    && b.windows == self.windows.len() as u64
                    && b.tokenizer == plan.tokenizer
                    && b.source.model.0 == *plan.source.as_bytes()
                    && [b.source.cache.0, b.source.tokens.0]
                        .iter()
                        .all(|v| v != &[0; 32])
                    && CalibrationCacheManifest {
                        binding: b.clone(),
                        window_samples: window_samples.clone(),
                    }
                    .validate(plan.limits.per_cache)
                    .is_ok()
            })
    }
}

/// Existing fixed objective bytes, shared by execution and recorded validation.
/// Includes every label, including final targets absent from model input tokens.
#[must_use]
pub fn next_token_capture_objective(windows: &[&[u32]]) -> Option<Vec<u8>> {
    if windows.is_empty() || windows.iter().any(|w| w.len() < 2) {
        return None;
    }
    let mut size = b"tritium.capture.next-token-window-mean-k1-f32.v1"
        .len()
        .checked_add(8)?;
    for window in windows {
        size = size.checked_add(
            window
                .len()
                .checked_sub(1)?
                .checked_mul(4)?
                .checked_add(8)?,
        )?;
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).ok()?;
    bytes.extend_from_slice(b"tritium.capture.next-token-window-mean-k1-f32.v1");
    bytes.extend_from_slice(&(windows.len() as u64).to_le_bytes());
    for window in windows {
        bytes.extend_from_slice(&((window.len() - 1) as u64).to_le_bytes());
        for target in &window[1..] {
            bytes.extend_from_slice(&target.to_le_bytes());
        }
    }
    Some(bytes)
}
