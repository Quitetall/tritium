//! Actual original masters -> existing ordered salience reducer -> sealed record/replay.
mod lineage;
use super::{SourceCalibrationModel, StagedCalibrationLimits};
use crate::{
    NnError,
    calibrate::{Calib, Tap, accumulate, calibrate_tapped},
};
pub use lineage::{RecordedOriginalCapturedStage, verify_original_captured_stage};
use std::collections::BTreeMap;
use tritium_evidence::{
    EvidenceRecorder, OriginalSalienceCaptureView, SealedEvidence, VerifiedLog,
};
use tritium_quantize::engine::{DeclaredSalienceStatistics, ObservedWeightSource, WeightSource};
use tritium_schema::{
    OriginalSalienceCaptureCompleted, OriginalSalienceCaptureInput, OriginalSalienceCaptureLimits,
    OriginalSalienceCaptureOutput, OriginalSalienceCapturePair, OriginalSalienceCapturePlan,
    OriginalSalienceCaptureProfile, OriginalSalienceCaptureStarted, QuantAction, QuantLimits,
    QuantPlan, QuantTensorPlan, SalienceInverseAxis,
};

/// Actual ordered salience sums and standalone evidence, not captured B/C or qualification.
/// Read-only access prevents replacement of returned vectors or sealed context in-place.
pub struct RecordedOriginalSalienceCapture {
    statistics: BTreeMap<String, DeclaredSalienceStatistics>,
    evidence: SealedEvidence,
}
impl std::fmt::Debug for RecordedOriginalSalienceCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordedOriginalSalienceCapture")
            .field("operations", &self.statistics.len())
            .field("root", &self.evidence.log().root_digest)
            .finish_non_exhaustive()
    }
}
impl RecordedOriginalSalienceCapture {
    /// Actual immutable f64 sums/counts for the existing transform seam.
    /// That seam's statistics kind remains Declared until mandatory carrier ancestry exists.
    #[must_use]
    pub fn statistics(&self) -> &BTreeMap<String, DeclaredSalienceStatistics> {
        &self.statistics
    }
    /// Actual standalone registered log; raw text may be private.
    #[must_use]
    pub fn evidence(&self) -> &SealedEvidence {
        &self.evidence
    }
}

impl SourceCalibrationModel {
    /// Execute every token through the existing block graph/legacy f64 reducer and record it.
    /// No loss/gradient/identity B stage, caller Arch or asserted statistics are accepted.
    /// Config/master caps must match construction; window admission may be narrower.
    /// Logical bounds are not parser/tape/RSS/GPU or independent qualification.
    /// # Errors
    /// Registration, source, model/input/moment caps, coverage or finite-buffer failure.
    pub fn capture_salience_recorded(
        &self,
        source: &mut ObservedWeightSource<'_>,
        windows: &[&[u32]],
        limits: OriginalSalienceCaptureLimits,
        recorder: EvidenceRecorder,
    ) -> Result<RecordedOriginalSalienceCapture, NnError> {
        check_recorder(&recorder)?;
        self.admit_original_windows(windows, limits)?;
        check_source(self, source, limits)?;
        let plan = self.original_plan(
            source,
            OriginalSalienceCaptureInput::Tokens {
                windows: windows.iter().map(|w| w.to_vec()).collect(),
            },
            limits,
        )?;
        self.record_original(source, plan, windows, recorder)
    }

    /// Actually encode exact private UTF-8 contexts with the shared runtime policy,
    /// then execute all tokens (one-token windows are valid). No automatic specials,
    /// chat template, padding, truncation, fallback, loss or backward is introduced.
    /// # Errors
    /// Registration/source/assets, tokenizer policy/vocab or common salience admission.
    #[cfg(feature = "tokenizer")]
    pub fn capture_salience_text_recorded(
        &self,
        source: &mut ObservedWeightSource<'_>,
        texts: &[&str],
        limits: OriginalSalienceCaptureLimits,
        recorder: EvidenceRecorder,
    ) -> Result<RecordedOriginalSalienceCapture, NnError> {
        check_recorder(&recorder)?;
        self.admit_original_limits(limits)?;
        let text_bytes = texts
            .iter()
            .try_fold(0u64, |n, s| n.checked_add(s.len() as u64));
        if texts.is_empty()
            || texts.len() as u64 > limits.max_windows
            || limits.max_text_bytes == 0
            || limits.max_tokenizer_config_bytes == 0
            || texts.iter().any(|s| s.is_empty())
            || text_bytes.is_none_or(|n| n > limits.max_text_bytes)
        {
            return Err(reject("original text input admission rejected"));
        }
        check_source(self, source, limits)?;
        let mut rows = 0u64;
        let windows = super::text::encode_checked_text_windows(
            source,
            texts,
            self.core.arch.vocab,
            super::text::TextInputLimits {
                max_text_bytes: limits.max_text_bytes,
                max_tokenizer_bytes: limits.max_asset_bytes,
                max_tokenizer_config_bytes: limits.max_tokenizer_config_bytes,
            },
            |window| {
                self.core.validate_window(window)?;
                if window.len() as u64 > limits.max_window_tokens {
                    return Err(reject("original encoded window limit exceeded"));
                }
                rows = rows
                    .checked_add(window.len() as u64)
                    .ok_or_else(|| reject("original encoded row count overflow"))?;
                if rows > limits.max_total_tokens {
                    return Err(reject("original encoded total row limit exceeded"));
                }
                Ok(())
            },
        )?;
        let windows: Vec<_> = windows.iter().map(Vec::as_slice).collect();
        self.admit_original_windows(&windows, limits)?;
        let plan = self.original_plan(
            source,
            OriginalSalienceCaptureInput::RawText {
                windows: texts.iter().map(|s| (*s).into()).collect(),
            },
            limits,
        )?;
        self.record_original(source, plan, &windows, recorder)
    }

    fn admit_original_limits(&self, limits: OriginalSalienceCaptureLimits) -> Result<(), NnError> {
        if limits.max_config_bytes != self.core.limits.max_config_bytes
            || limits.max_dense_bytes != self.core.limits.max_dense_bytes
            || limits.max_window_tokens > self.core.max_window_tokens as u64
            || [
                limits.max_source_bytes,
                limits.max_asset_bytes,
                limits.max_window_tokens,
                limits.max_windows,
                limits.max_total_tokens,
                limits.max_moment_bytes,
            ]
            .contains(&0)
        {
            return Err(reject("original salience model/receiving limits disagree"));
        }
        let bytes = self.original_pairs().iter().try_fold(0u64, |n, p| {
            p.channels
                .checked_mul(8)
                .and_then(|bytes| n.checked_add(bytes))
        });
        if bytes.is_none_or(|n| n > limits.max_moment_bytes) {
            return Err(reject("original salience moment byte limit exceeded"));
        }
        Ok(())
    }

    fn admit_original_windows(
        &self,
        windows: &[&[u32]],
        limits: OriginalSalienceCaptureLimits,
    ) -> Result<(), NnError> {
        self.admit_original_limits(limits)?;
        if windows.is_empty() || windows.len() as u64 > limits.max_windows {
            return Err(reject("original salience window count rejected"));
        }
        let mut rows = 0u64;
        for window in windows {
            self.core.validate_window(window)?;
            if window.len() as u64 > limits.max_window_tokens {
                return Err(reject("original salience window limit exceeded"));
            }
            rows = rows
                .checked_add(window.len() as u64)
                .ok_or_else(|| reject("original salience row count overflow"))?;
        }
        if rows > limits.max_total_tokens || usize::try_from(rows).is_err() {
            return Err(reject("original salience total row limit exceeded"));
        }
        Ok(())
    }

    fn original_pairs(&self) -> Vec<OriginalSalienceCapturePair> {
        let schema = crate::model::NameSchema::Hf;
        (0..self.core.arch.n_layers)
            .flat_map(|layer| {
                [
                    OriginalSalienceCapturePair {
                        name: schema.layer(layer, "q"),
                        projections: ["q", "k", "v"].map(|slot| schema.layer(layer, slot)).into(),
                        inverse: schema.layer(layer, "attn_norm"),
                        inverse_axis: SalienceInverseAxis::NormChannels,
                        channels: self.core.arch.n_embd as u64,
                    },
                    OriginalSalienceCapturePair {
                        name: schema.layer(layer, "gate"),
                        projections: ["gate", "up"].map(|slot| schema.layer(layer, slot)).into(),
                        inverse: schema.layer(layer, "ffn_norm"),
                        inverse_axis: SalienceInverseAxis::NormChannels,
                        channels: self.core.arch.n_embd as u64,
                    },
                    OriginalSalienceCapturePair {
                        name: schema.layer(layer, "down"),
                        projections: vec![schema.layer(layer, "down")],
                        inverse: schema.layer(layer, "up"),
                        inverse_axis: SalienceInverseAxis::LinearOutputRows,
                        channels: self.core.arch.ff as u64,
                    },
                ]
            })
            .collect()
    }

    fn original_plan(
        &self,
        source: &ObservedWeightSource<'_>,
        input: OriginalSalienceCaptureInput,
        limits: OriginalSalienceCaptureLimits,
    ) -> Result<OriginalSalienceCapturePlan, NnError> {
        if source.source_id() != self.core.source {
            return Err(reject("original salience has foreign source"));
        }
        let plan = OriginalSalienceCapturePlan {
            architecture: source.architecture().into(),
            source: self.core.source,
            tensors: source.tensor_bindings().to_vec(),
            assets: source.asset_bindings().to_vec(),
            profile: OriginalSalienceCaptureProfile::AllTokenF64SecondMomentV1,
            pairs: self.original_pairs(),
            input,
            limits,
        };
        plan.validate().map_err(|e| reject(&e.to_string()))?;
        Ok(plan)
    }

    fn record_original(
        &self,
        source: &ObservedWeightSource<'_>,
        plan: OriginalSalienceCapturePlan,
        windows: &[&[u32]],
        mut recorder: EvidenceRecorder,
    ) -> Result<RecordedOriginalSalienceCapture, NnError> {
        recorder
            .emit(
                "salience",
                None,
                0,
                &OriginalSalienceCaptureStarted { plan: plan.clone() },
            )
            .map_err(|e| reject(&e.to_string()))?;
        let statistics = self.original_moments(windows, &plan)?;
        source.inventory().map_err(|e| reject(&e.to_string()))?;
        let completed = completion(windows, &plan, &statistics)?;
        recorder
            .emit("salience", None, 1, &completed)
            .map_err(|e| reject(&e.to_string()))?;
        let evidence = recorder.seal().map_err(|e| reject(&e.to_string()))?;
        let views = OriginalSalienceCaptureView::fold(evidence.log())
            .map_err(|e| reject(&e.to_string()))?;
        if views.len() != 1
            || evidence.log().events.len() != 2
            || *views[0].completed() != completed
        {
            return Err(reject("original salience standalone completion differs"));
        }
        Ok(RecordedOriginalSalienceCapture {
            statistics,
            evidence,
        })
    }

    fn original_moments(
        &self,
        windows: &[&[u32]],
        plan: &OriginalSalienceCapturePlan,
    ) -> Result<BTreeMap<String, DeclaredSalienceStatistics>, NnError> {
        self.admit_original_windows(windows, plan.limits)?;
        let mut calib = Calib::new(&self.core.arch);
        for window in windows {
            let mut cursor = 0usize;
            let mut failure = false;
            calibrate_tapped(
                &self.core.masters,
                &self.core.arch,
                window,
                &mut |kind, layer, tape, id, rows, cols| {
                    let expected_layer = cursor / 3;
                    let expected_kind = [Tap::AttnIn, Tap::FfnIn, Tap::DownIn][cursor % 3];
                    let pair = plan.pairs.get(cursor);
                    if failure
                        || layer != expected_layer
                        || kind != expected_kind
                        || pair.is_none_or(|p| p.channels != cols as u64)
                        || rows != window.len()
                        || rows.checked_mul(cols) != Some(tape.value(id).len())
                        || tape.value(id).iter().any(|v| !v.is_finite())
                    {
                        failure = true;
                        return;
                    }
                    let acc = match kind {
                        Tap::AttnIn => &mut calib.attn_in[layer],
                        Tap::FfnIn => &mut calib.ffn_in[layer],
                        Tap::DownIn => &mut calib.down_in[layer],
                        _ => {
                            failure = true;
                            return;
                        }
                    };
                    accumulate(tape, id, rows, cols, acc);
                    if acc.iter().any(|v| !v.is_finite()) {
                        failure = true;
                    }
                    cursor += 1;
                },
            );
            if failure || cursor != plan.pairs.len() {
                return Err(reject(
                    "original salience actual tap coverage/values rejected",
                ));
            }
            calib.rows = calib
                .rows
                .checked_add(window.len())
                .ok_or_else(|| reject("original salience rows overflow"))?;
        }
        let mut statistics = BTreeMap::new();
        for (index, pair) in plan.pairs.iter().enumerate() {
            let layer = index / 3;
            let moments = match index % 3 {
                0 => std::mem::take(&mut calib.attn_in[layer]),
                1 => std::mem::take(&mut calib.ffn_in[layer]),
                _ => std::mem::take(&mut calib.down_in[layer]),
            };
            statistics.insert(
                pair.name.clone(),
                DeclaredSalienceStatistics {
                    moments,
                    rows: calib.rows as u64,
                },
            );
        }
        Ok(statistics)
    }
}

/// Cold actual original-source replay under explicit receiving caps, not caller hashes/models.
/// Reruns source observation, checked decoding, tokenizer/block graph/legacy reducer and
/// compares complete request/windows/moment bindings. Not historical authentication,
/// independent qualification, captured B/C ancestry, RSS or GPU admission.
/// # Errors
/// Invalid/incomplete log, receiving caps, actual source/assets/graph or numerical disagreement.
pub fn replay_original_salience_capture(
    log: &VerifiedLog,
    source: &mut dyn WeightSource,
    receiving: OriginalSalienceCaptureLimits,
) -> Result<BTreeMap<String, DeclaredSalienceStatistics>, NnError> {
    let views = OriginalSalienceCaptureView::fold(log).map_err(|e| reject(&e.to_string()))?;
    if views.len() != 1 || log.events.len() != 2 {
        return Err(reject("original replay needs complete standalone capture"));
    }
    let plan = &views[0].started().plan;
    if !plan.limits.admitted_by(&receiving) {
        return Err(reject("original replay receiver limit escalation rejected"));
    }
    #[cfg(not(feature = "tokenizer"))]
    if matches!(&plan.input, OriginalSalienceCaptureInput::RawText { .. }) {
        return Err(reject("original text replay requires tokenizer feature"));
    }
    let limits = plan.limits;
    // Observation-only metadata: no fitting, package or output retention is executed.
    let request = QuantPlan {
        architecture: plan.architecture.clone(),
        tensors: plan
            .tensors
            .iter()
            .map(|b| QuantTensorPlan {
                source: b.source.clone(),
                action: QuantAction::KeepDense,
            })
            .collect(),
        assets: plan.assets.iter().map(|a| a.name.clone()).collect(),
        levels: vec!["dense".into()],
        limits: QuantLimits {
            max_source_bytes: limits.max_source_bytes,
            max_asset_bytes: limits.max_asset_bytes,
            max_retained_blob_bytes: 0,
            max_package_bytes: 0,
        },
    };
    let mut observed =
        ObservedWeightSource::new(&request, source).map_err(|e| reject(&e.to_string()))?;
    if observed.source_id() != plan.source {
        return Err(reject("original replay actual source differs"));
    }
    let model = SourceCalibrationModel::new(
        &mut observed,
        StagedCalibrationLimits {
            max_config_bytes: limits.max_config_bytes,
            max_dense_bytes: limits.max_dense_bytes,
            max_window_tokens: usize::try_from(limits.max_window_tokens)
                .map_err(|_| reject("original replay window overflows usize"))?,
        },
    )?;
    let recorder = EvidenceRecorder::new(
        "original-cold-replay",
        [
            tritium_evidence::ORIGINAL_SALIENCE_CAPTURE_STARTED_REGISTRATION,
            tritium_evidence::ORIGINAL_SALIENCE_CAPTURE_COMPLETED_REGISTRATION,
        ],
    )
    .map_err(|e| reject(&e.to_string()))?;
    let actual = match &plan.input {
        OriginalSalienceCaptureInput::Tokens { windows } => {
            let windows: Vec<_> = windows.iter().map(Vec::as_slice).collect();
            model.capture_salience_recorded(&mut observed, &windows, limits, recorder)?
        }
        OriginalSalienceCaptureInput::RawText { windows } => {
            #[cfg(feature = "tokenizer")]
            {
                let texts: Vec<_> = windows.iter().map(String::as_str).collect();
                model.capture_salience_text_recorded(&mut observed, &texts, limits, recorder)?
            }
            #[cfg(not(feature = "tokenizer"))]
            {
                let _ = windows;
                return Err(reject("original text replay requires tokenizer feature"));
            }
        }
        _ => return Err(reject("unsupported original capture input")),
    };
    let actual_views = OriginalSalienceCaptureView::fold(actual.evidence.log())
        .map_err(|e| reject(&e.to_string()))?;
    if actual_views[0].started().plan != *plan
        || actual_views[0].completed() != views[0].completed()
    {
        return Err(reject(
            "original replay actual request/windows/moments differ",
        ));
    }
    Ok(actual.statistics)
}

fn check_recorder(recorder: &EvidenceRecorder) -> Result<(), NnError> {
    recorder
        .check_emit::<OriginalSalienceCaptureStarted>("salience", None, 0)
        .map_err(|e| reject(&e.to_string()))?;
    recorder
        .check_emit::<OriginalSalienceCaptureCompleted>("salience", None, 1)
        .map_err(|e| reject(&e.to_string()))
}
fn check_source(
    model: &SourceCalibrationModel,
    source: &mut ObservedWeightSource<'_>,
    limits: OriginalSalienceCaptureLimits,
) -> Result<(), NnError> {
    if source.source_id() != model.core.source {
        return Err(reject("original salience has foreign source"));
    }
    source.inventory().map_err(|e| reject(&e.to_string()))?;
    let total = source.tensor_bindings().iter().try_fold(0u64, |n, b| {
        b.source
            .byte_len()
            .ok()
            .and_then(|bytes| n.checked_add(bytes))
    });
    if total.is_none_or(|n| n > limits.max_source_bytes) {
        return Err(reject("original salience raw source cap exceeded"));
    }
    for index in 0..source.tensor_bindings().len() {
        let tensor = &source.tensor_bindings()[index].source;
        let name = tensor.name.clone();
        let bytes = tensor.byte_len().map_err(|e| reject(&e.to_string()))?;
        source
            .tensor(&name, platform(bytes)?)
            .map_err(|e| reject(&e.to_string()))?;
    }
    let mut remaining = limits.max_asset_bytes;
    for name in source.asset_names() {
        let bytes = source
            .asset(&name, platform(remaining)?)
            .map_err(|e| reject(&e.to_string()))?;
        remaining = remaining
            .checked_sub(bytes.len() as u64)
            .ok_or_else(|| reject("original salience asset cap exceeded"))?;
    }
    source.inventory().map_err(|e| reject(&e.to_string()))?;
    Ok(())
}
fn completion(
    windows: &[&[u32]],
    plan: &OriginalSalienceCapturePlan,
    statistics: &BTreeMap<String, DeclaredSalienceStatistics>,
) -> Result<OriginalSalienceCaptureCompleted, NnError> {
    let outputs = plan
        .pairs
        .iter()
        .map(|pair| {
            Ok(OriginalSalienceCaptureOutput {
                operation: pair.name.clone(),
                moments: statistics
                    .get(&pair.name)
                    .ok_or_else(|| reject("original salience output missing"))?
                    .binding()
                    .map_err(|e| reject(&e.to_string()))?,
            })
        })
        .collect::<Result<_, NnError>>()?;
    let completed = OriginalSalienceCaptureCompleted {
        windows: windows.iter().map(|w| w.to_vec()).collect(),
        outputs,
    };
    if !completed.agrees_with(plan) {
        return Err(reject("original salience completed context differs"));
    }
    Ok(completed)
}
fn platform(bytes: u64) -> Result<usize, NnError> {
    usize::try_from(bytes).map_err(|_| reject("original salience byte cap overflows usize"))
}
fn reject(detail: &str) -> NnError {
    NnError::Backend(format!("original salience: {detail}"))
}
