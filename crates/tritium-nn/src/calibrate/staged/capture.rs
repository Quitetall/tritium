//! Actual B graph, fixed executed objective, existing immutable cache/reducer.
use super::StagedCalibrationModel;
use crate::{NnError, calibrate::weight_names};
use tritium_quantize::engine::{
    CalibrationBatch, CalibrationCache, CalibrationCacheBuilder, CalibrationInput,
    CalibrationOutputBatch, StagedSalienceSource, WeightSource,
};
use tritium_schema::{
    CalibrationCacheLimits, CalibrationCacheRequest, CalibrationConsumer, CalibrationOutput,
};
use tritium_train::Tape;

const OBJECTIVE: &[u8] = b"tritium.capture.next-token-window-mean-k1-f32.v1";

/// Retained paired-cache payload limits, not master/tape/metadata/source/RSS limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StagedCaptureLimits {
    /// Existing exact scalar/token/mask/recipe limit for each projection cache.
    pub per_cache: CalibrationCacheLimits,
    /// Sum of retained payloads, including repeated tokenizer/objective bytes.
    pub max_total_payload_bytes: u64,
}

impl StagedCalibrationModel {
    /// Execute next-token cross-entropy and capture B's actual matrix inputs/VJPs.
    ///
    /// Each window includes its final label: execute `window[..N-1]` and predict
    /// `window[1..]`, with the existing per-window mean loss. Uses actual B tensor
    /// interpretations/tokenizer bytes; callers supply pretokenized IDs, not proof
    /// of text tokenization. Results follow shared per-layer weight order, then
    /// the tied head. The embedding gather is intentionally NOT a dense consumer.
    ///
    /// This returns actual in-process buffers, not independent graph/quality
    /// clearance, a registered capture receipt, or an out-of-core/RSS guarantee.
    /// # Errors
    /// Foreign B, invalid window/limits, retained payload admission, nonfinite
    /// graph/loss/gradients, missing stage asset or existing cache rejection.
    pub fn capture_next_token(
        &self,
        source: &mut StagedSalienceSource<'_>,
        windows: &[&[u32]],
        limits: StagedCaptureLimits,
    ) -> Result<Vec<CalibrationCache>, NnError> {
        if source.source_id() != self.source {
            return Err(reject("capture stage differs from immutable B masters"));
        }
        if windows.is_empty()
            || windows.len() as u64 > limits.per_cache.max_windows
            || limits.per_cache.max_payload_bytes == 0
            || limits.per_cache.max_samples == 0
            || limits.max_total_payload_bytes == 0
        {
            return Err(reject("invalid B capture windows or limits"));
        }
        let mut samples = 0u64;
        let mut objective_bytes = OBJECTIVE.len() as u64 + 8;
        for window in windows {
            if window.len() < 2
                || window.len() - 1 > self.max_window_tokens
                || window
                    .iter()
                    .any(|&id| usize::try_from(id).map_or(true, |id| id >= self.arch.vocab))
            {
                return Err(reject("invalid B capture token window"));
            }
            let n = (window.len() - 1) as u64;
            samples = checked(samples.checked_add(n))?;
            objective_bytes = checked(
                n.checked_mul(4)
                    .and_then(|n| n.checked_add(8))
                    .and_then(|n| objective_bytes.checked_add(n)),
            )?;
        }
        if samples > limits.per_cache.max_samples {
            return Err(reject("B capture sample limit exceeded"));
        }
        // Exact B interpretations, including retained F16/Bf16 untouched weights.
        let inventory = source
            .inventory()
            .map_err(|e| reject(&format!("B capture inventory: {e}")))?;
        let names = weight_names(&self.arch);
        let selected: Vec<_> = names[1..]
            .iter()
            .chain(&names[..1])
            .map(|name| {
                inventory
                    .iter()
                    .find(|tensor| &tensor.name == name)
                    .cloned()
                    .ok_or_else(|| reject("missing actual B capture tensor"))
            })
            .collect::<Result<_, _>>()?;
        let tokenizer = source
            .asset(
                "tokenizer.json",
                usize::try_from(limits.per_cache.max_payload_bytes)
                    .map_err(|_| reject("B capture tokenizer limit overflows usize"))?,
            )
            .map_err(|e| reject(&format!("B capture tokenizer: {e}")))?;
        let context_bytes = checked((tokenizer.len() as u64).checked_add(objective_bytes))?;
        let mut total = 0u64;
        for tensor in &selected {
            if tensor.shape.len() != 2 {
                return Err(reject("B capture requires a matrix interpretation"));
            }
            // Per sample: input/output f32s, u32 token, f64 unit weight, bool mask.
            let width = checked(tensor.shape[0].checked_add(tensor.shape[1]))?;
            let per_sample = checked(width.checked_mul(4).and_then(|n| n.checked_add(13)))?;
            let payload = checked(
                per_sample
                    .checked_mul(samples)
                    .and_then(|n| context_bytes.checked_add(n)),
            )?;
            if payload > limits.per_cache.max_payload_bytes {
                return Err(reject("B capture per-cache payload limit exceeded"));
            }
            total = checked(total.checked_add(payload))?;
        }
        if total > limits.max_total_payload_bytes {
            return Err(reject("B capture aggregate payload limit exceeded"));
        }
        // Fixed controller-owned recipe records actual labels, including each
        // final target absent from the input token stream. No arbitrary recipe.
        let mut objective = Vec::new();
        objective
            .try_reserve_exact(
                usize::try_from(objective_bytes)
                    .map_err(|_| reject("B capture objective limit overflows usize"))?,
            )
            .map_err(|_| reject("B capture objective allocation failed"))?;
        objective.extend_from_slice(OBJECTIVE);
        objective.extend_from_slice(&(windows.len() as u64).to_le_bytes());
        for window in windows {
            objective.extend_from_slice(&((window.len() - 1) as u64).to_le_bytes());
            for target in &window[1..] {
                objective.extend_from_slice(&target.to_le_bytes());
            }
        }
        let mut builders: Vec<_> = selected
            .into_iter()
            .map(|tensor| {
                CalibrationCacheBuilder::new(
                    CalibrationCacheRequest {
                        tensor,
                        model: self.source,
                        consumer: CalibrationConsumer::Matmul,
                        output: CalibrationOutput::DenseFisher,
                        sample_start: 0,
                        limits: limits.per_cache,
                    },
                    &tokenizer,
                    &objective,
                )
                .map_err(|e| reject(&format!("B capture builder: {e}")))
            })
            .collect::<Result<_, _>>()?;
        for window in windows {
            let input = &window[..window.len() - 1];
            let mut tape = Tape::new();
            let mut taps = Vec::new();
            let logits = self.forward_observed(&mut tape, input, &mut |_, tap| {
                if tap.input.is_some() {
                    taps.push(tap);
                }
            })?;
            if taps.len() != builders.len()
                || taps.iter().enumerate().any(|(i, tap)| {
                    tap.weight_index != if i + 1 == builders.len() { 0 } else { i + 1 }
                })
                || tape.value(logits).iter().any(|v| !v.is_finite())
            {
                return Err(reject("invalid/nonfinite actual B capture graph"));
            }
            // Reuse sparse CE, not an additional dense vocabulary-sized target.
            let loss = tape.topk_kd(
                logits,
                window[1..].to_vec(),
                vec![1.0; input.len()],
                input.len(),
                self.arch.vocab,
                1,
            );
            if !tape.value(loss)[0].is_finite() {
                return Err(reject("nonfinite actual B capture objective"));
            }
            let gradients = tape.backward(loss);
            for (builder, tap) in builders.iter_mut().zip(taps) {
                builder
                    .capture(
                        input,
                        CalibrationBatch {
                            samples: tap.samples,
                            input: CalibrationInput::Activations(tape.value(tap.input.unwrap())),
                            output: CalibrationOutputBatch::Dense(&gradients[tap.output]),
                            weights: None,
                            mask: None,
                        },
                    )
                    .map_err(|e| reject(&format!("B capture actual buffers: {e}")))?;
            }
        }
        builders
            .into_iter()
            .map(|builder| {
                builder
                    .finish()
                    .map_err(|e| reject(&format!("B capture finalization: {e}")))
            })
            .collect()
    }
}

fn checked(value: Option<u64>) -> Result<u64, NnError> {
    value.ok_or_else(|| reject("B capture payload arithmetic overflow"))
}

fn reject(message: &str) -> NnError {
    NnError::Backend(message.into())
}
