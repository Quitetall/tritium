//! Original all-token forward salience, distinct from B loss/gradient capture.
//! Structural validation never establishes numerical execution or graph eligibility.
use crate::{
    NamedAssetDigest, OriginalSourceId, QuantSourceBinding, SalienceInverseAxis,
    SalienceMomentBinding, SalienceTransformPlanError, validate_original_source_inventory,
};
use alloc::{collections::BTreeSet, string::String, vec::Vec};

/// Fixed ordered legacy reducer; no loss, backward, weighting or averaging.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
pub enum OriginalSalienceCaptureProfile {
    /// Every input token, f32 activation widened to f64, ordered per-channel x*x sums.
    AllTokenF64SecondMomentV1,
}

/// Logical input/master/moment caps, not parser, tape, filesystem or full RSS admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OriginalSalienceCaptureLimits {
    /// Complete raw tensor payload cap for actual source observation.
    pub max_source_bytes: u64,
    /// Complete raw asset payload cap for actual source observation.
    pub max_asset_bytes: u64,
    /// Config bytes before parsing.
    pub max_config_bytes: u64,
    /// Aggregate retained widened f32 master bytes.
    pub max_dense_bytes: u64,
    /// Every token in one context, including its final token.
    pub max_window_tokens: u64,
    /// Complete number of ordered contexts.
    pub max_windows: u64,
    /// Aggregate executed tokens/rows, without shared-consumer multiplication.
    pub max_total_tokens: u64,
    /// Aggregate retained raw f64 moment-vector bytes.
    pub max_moment_bytes: u64,
    /// Aggregate exact UTF-8 text bytes; zero for token-ID input.
    pub max_text_bytes: u64,
    /// Actual tokenizer_config.json bytes; zero for token-ID input.
    pub max_tokenizer_config_bytes: u64,
}
impl OriginalSalienceCaptureLimits {
    /// Check every recorded cap against explicit receiving admission.
    #[must_use]
    pub fn admitted_by(&self, receiver: &Self) -> bool {
        self.max_source_bytes <= receiver.max_source_bytes
            && self.max_asset_bytes <= receiver.max_asset_bytes
            && self.max_config_bytes <= receiver.max_config_bytes
            && self.max_dense_bytes <= receiver.max_dense_bytes
            && self.max_window_tokens <= receiver.max_window_tokens
            && self.max_windows <= receiver.max_windows
            && self.max_total_tokens <= receiver.max_total_tokens
            && self.max_moment_bytes <= receiver.max_moment_bytes
            && self.max_text_bytes <= receiver.max_text_bytes
            && self.max_tokenizer_config_bytes <= receiver.max_tokenizer_config_bytes
    }
    fn positive(&self) -> bool {
        self.max_source_bytes > 0
            && self.max_asset_bytes > 0
            && self.max_config_bytes > 0
            && self.max_dense_bytes > 0
            && self.max_window_tokens > 0
            && self.max_windows > 0
            && self.max_total_tokens > 0
            && self.max_moment_bytes > 0
    }
}

/// Exact ordered inputs. Tokens are all inputs, never next-token targets.
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub enum OriginalSalienceCaptureInput {
    /// Complete pretokenized nonempty windows.
    Tokens {
        /// Every token is executed, in this exact order and context partition.
        windows: Vec<Vec<u32>>,
    },
    /// Exact private raw text, encoded without special tokens/template/fallback.
    RawText {
        /// UTF-8 input windows, without padding or truncation.
        windows: Vec<String>,
    },
}

/// Ordered actual graph pair interpretation, without fabricated moment bindings.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OriginalSalienceCapturePair {
    /// Unique logical operation name, in graph execution order.
    pub name: String,
    /// Exact raw matrices sharing one observed input, without duplicate accumulation.
    pub projections: Vec<String>,
    /// Actual preceding norm or linear output-row partner.
    pub inverse: String,
    /// Explicit partner axis.
    pub inverse_axis: SalienceInverseAxis,
    /// Actual input columns/partner output channels.
    pub channels: u64,
}

/// Controller-derived complete original content and graph context.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OriginalSalienceCapturePlan {
    /// Actual raw-source architecture, not a caller-owned Arch.
    pub architecture: String,
    /// Complete stored-byte A identity; the typed view rederives it.
    pub source: OriginalSourceId,
    /// Complete sorted raw interpretations and content bindings.
    pub tensors: Vec<QuantSourceBinding>,
    /// Complete sorted raw asset bindings, including config and tokenizer.
    pub assets: Vec<NamedAssetDigest>,
    /// One fixed all-token ordered forward-only reducer.
    pub profile: OriginalSalienceCaptureProfile,
    /// Complete ordered actual graph pairs; structural geometry is not graph proof.
    pub pairs: Vec<OriginalSalienceCapturePair>,
    /// Exact input/context boundaries, private when raw text is used.
    pub input: OriginalSalienceCaptureInput,
    /// Actual logical admission, separately bounded during cold replay.
    pub limits: OriginalSalienceCaptureLimits,
}

/// Distinct registered original forward-salience root request.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OriginalSalienceCaptureStarted {
    /// Actual complete original request.
    pub plan: OriginalSalienceCapturePlan,
}

/// Exact observed moment content and count for one operation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OriginalSalienceCaptureOutput {
    /// Exact operation name at this position in the request.
    pub operation: String,
    /// Existing exact little-endian f64 content/channel/row binding.
    pub moments: SalienceMomentBinding,
}

/// Actual ordered encoded windows and moment outputs, not execution qualification.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
pub struct OriginalSalienceCaptureCompleted {
    /// Every executed token in each context, including final tokens.
    pub windows: Vec<Vec<u32>>,
    /// One moment vector binding per ordered pair, never one per shared projection.
    pub outputs: Vec<OriginalSalienceCaptureOutput>,
}

impl OriginalSalienceCapturePlan {
    /// Validate inventories, exact input policy, pair geometry and logical caps.
    /// The evidence view additionally derives the source identity; actual bytes,
    /// graph eligibility, tokenizer results and moments require execution/replay.
    /// # Errors
    /// Invalid inventories, duplicate/foreign pairs, overflow or insufficient caps.
    pub fn validate(&self) -> Result<(), SalienceTransformPlanError> {
        use SalienceTransformPlanError as E;
        validate_original_source_inventory(&self.architecture, &self.tensors, &self.assets)
            .map_err(E::Source)?;
        if self.source.as_bytes() == &[0; 32]
            || !self.limits.positive()
            || self
                .tensors
                .iter()
                .any(|t| t.content.as_bytes() == &[0; 32])
            || self.assets.iter().any(|a| a.content.as_bytes() == &[0; 32])
            || !self.assets.iter().any(|a| a.name == "config.json")
            || !self.assets.iter().any(|a| a.name == "tokenizer.json")
        {
            return Err(E::Observation);
        }
        let count = match &self.input {
            OriginalSalienceCaptureInput::Tokens { windows } => {
                if self.limits.max_text_bytes != 0
                    || self.limits.max_tokenizer_config_bytes != 0
                    || admitted_rows(windows, self.limits).is_none()
                {
                    return Err(E::Limit);
                }
                windows.len()
            }
            OriginalSalienceCaptureInput::RawText { windows } => {
                let bytes = windows
                    .iter()
                    .try_fold(0u64, |n, s| n.checked_add(s.len() as u64));
                if self.limits.max_text_bytes == 0
                    || self.limits.max_tokenizer_config_bytes == 0
                    || !self
                        .assets
                        .iter()
                        .any(|a| a.name == "tokenizer_config.json")
                    || windows.iter().any(String::is_empty)
                    || bytes.is_none_or(|n| n > self.limits.max_text_bytes)
                {
                    return Err(E::Limit);
                }
                windows.len()
            }
        };
        if count == 0
            || count as u64 > self.limits.max_windows
            || self.pairs.is_empty()
            || self.pairs.len() > 1_000_000
        {
            return Err(E::Operations);
        }
        let mut source_bytes = 0u64;
        let mut dense_bytes = 0u64;
        for tensor in &self.tensors {
            let bytes = tensor.source.byte_len().map_err(E::Source)?;
            source_bytes = source_bytes.checked_add(bytes).ok_or(E::Limit)?;
            let width = if tensor.source.precision == crate::ScalePrecision::F32 {
                4
            } else {
                2
            };
            dense_bytes = dense_bytes
                .checked_add((bytes / width).checked_mul(4).ok_or(E::Limit)?)
                .ok_or(E::Limit)?;
        }
        if source_bytes > self.limits.max_source_bytes || dense_bytes > self.limits.max_dense_bytes
        {
            return Err(E::Limit);
        }
        let mut names = BTreeSet::new();
        let mut moments = 0u64;
        for pair in &self.pairs {
            crate::model::name(&pair.name, false)
                .map_err(|e| E::Source(crate::QuantPlanError::Manifest(e)))?;
            if !names.insert(pair.name.as_str()) || pair.projections.len() > self.tensors.len() {
                return Err(E::Operations);
            }
            crate::salience_transform::pair_geometry(
                |name| {
                    self.tensors
                        .binary_search_by(|t| t.source.name.as_str().cmp(name))
                        .map(|i| &self.tensors[i].source)
                        .map_err(|_| E::Pair)
                },
                &pair.projections,
                &pair.inverse,
                pair.inverse_axis,
                pair.channels,
            )?;
            moments = moments
                .checked_add(pair.channels.checked_mul(8).ok_or(E::Limit)?)
                .ok_or(E::Limit)?;
        }
        if moments > self.limits.max_moment_bytes {
            return Err(E::Limit);
        }
        Ok(())
    }
}

impl OriginalSalienceCaptureCompleted {
    /// Check complete ordered windows/outputs and actual all-token row counts.
    /// This does not check actual tokenizer encoding or moment bytes.
    #[must_use]
    pub fn agrees_with(&self, plan: &OriginalSalienceCapturePlan) -> bool {
        if plan.validate().is_err() || self.outputs.len() != plan.pairs.len() {
            return false;
        }
        let input_count = match &plan.input {
            OriginalSalienceCaptureInput::Tokens { windows } => {
                if *windows != self.windows {
                    return false;
                }
                windows.len()
            }
            OriginalSalienceCaptureInput::RawText { windows } => windows.len(),
        };
        let Some(rows) = admitted_rows(&self.windows, plan.limits) else {
            return false;
        };
        input_count == self.windows.len()
            && self.outputs.iter().zip(&plan.pairs).all(|(output, pair)| {
                output.operation == pair.name
                    && output.moments.channels == pair.channels
                    && output.moments.rows == rows
                    && output.moments.content.as_bytes() != &[0; 32]
            })
    }
}

fn admitted_rows(windows: &[Vec<u32>], limits: OriginalSalienceCaptureLimits) -> Option<u64> {
    if windows.is_empty() || windows.len() as u64 > limits.max_windows {
        return None;
    }
    let rows = windows.iter().try_fold(0u64, |total, window| {
        let rows = window.len() as u64;
        if rows == 0 || rows > limits.max_window_tokens {
            return None;
        }
        total.checked_add(rows)
    })?;
    (rows <= limits.max_total_tokens).then_some(rows)
}
