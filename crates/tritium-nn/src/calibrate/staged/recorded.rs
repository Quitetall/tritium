//! Recorded capture and cold actual-source replay using one execution controller.
use super::{StagedCalibrationLimits, StagedCalibrationModel, StagedCaptureLimits};
use crate::NnError;
use tritium_evidence::{CalibrationCaptureView, EvidenceRecorder, SealedEvidence, VerifiedLog};
use tritium_quantize::engine::{CalibrationCache, StagedSalienceSource};
use tritium_schema::{
    CalibrationCaptureCompleted, CalibrationCaptureInput, CalibrationCaptureLimits,
    CalibrationCapturePlan, CalibrationCaptureProfile, CalibrationCaptureStarted,
};

/// Actual complete caches and sealed capture observations; not qualification.
#[derive(Debug)]
pub struct RecordedStagedCapture {
    /// Existing immutable cache payloads with actual B/consumer/context binding.
    pub caches: Vec<CalibrationCache>,
    /// Standalone registered request/completion log; may contain private text.
    pub evidence: SealedEvidence,
}

impl StagedCalibrationModel {
    /// Record actual token-ID capture, binding B/package/transform and full inputs.
    /// No caller provenance hashes, tokenization assertion or independent clearance.
    /// # Errors
    /// Missing registration, foreign stage, actual input/cache limits or execution failure.
    pub fn capture_next_token_recorded(
        &self,
        source: &mut StagedSalienceSource<'_>,
        windows: &[&[u32]],
        limits: StagedCaptureLimits,
        recorder: EvidenceRecorder,
    ) -> Result<RecordedStagedCapture, NnError> {
        check_recorder(&recorder)?;
        if source.source_id() != self.core.source {
            return Err(reject("recorded capture has foreign B"));
        }
        self.admit_capture_windows(windows, limits)?;
        let plan = self.capture_plan(
            source,
            CalibrationCaptureInput::Tokens {
                windows: windows.iter().map(|w| w.to_vec()).collect(),
            },
            limits,
            0,
            0,
        )?;
        self.record_capture(source, plan, windows, limits, recorder)
    }

    /// Actually encode and record exact private raw text with the existing runtime policy.
    /// Equal encoded tokens may yield equal caches but retain distinct raw-text logs.
    /// # Errors
    /// Missing registration/assets, actual encoding/graph/capture/input-cap rejection.
    #[cfg(feature = "tokenizer")]
    pub fn capture_next_token_text_recorded(
        &self,
        source: &mut StagedSalienceSource<'_>,
        texts: &[&str],
        text_limits: super::StagedTextCaptureLimits,
        capture_limits: StagedCaptureLimits,
        recorder: EvidenceRecorder,
    ) -> Result<RecordedStagedCapture, NnError> {
        check_recorder(&recorder)?;
        let windows = self.encode_capture_text(source, texts, text_limits, capture_limits)?;
        let plan = self.capture_plan(
            source,
            CalibrationCaptureInput::RawText {
                windows: texts.iter().map(|s| (*s).into()).collect(),
            },
            capture_limits,
            text_limits.max_text_bytes,
            text_limits.max_tokenizer_config_bytes,
        )?;
        let windows: Vec<_> = windows.iter().map(Vec::as_slice).collect();
        self.record_capture(source, plan, &windows, capture_limits, recorder)
    }

    fn capture_plan(
        &self,
        source: &StagedSalienceSource<'_>,
        input: CalibrationCaptureInput,
        capture: StagedCaptureLimits,
        max_text_bytes: u64,
        max_tokenizer_config_bytes: u64,
    ) -> Result<CalibrationCapturePlan, NnError> {
        if source.source_id() != self.core.source {
            return Err(reject("recorded capture has foreign B"));
        }
        let asset = |name: &str| {
            source
                .manifest()
                .observation
                .assets
                .iter()
                .find(|a| a.name == name)
                .map(|a| a.content)
                .ok_or_else(|| reject("recorded capture asset missing from actual B"))
        };
        let plan = CalibrationCapturePlan {
            stage_package: source.package_id(),
            source: self.core.source,
            transform: source.manifest().transform.clone(),
            tokenizer: asset("tokenizer.json")?,
            tokenizer_config: if max_text_bytes > 0 {
                Some(asset("tokenizer_config.json")?)
            } else {
                None
            },
            profile: CalibrationCaptureProfile::NextTokenWindowMeanK1F32V1,
            tensors: self.capture_tensors(source)?,
            input,
            limits: CalibrationCaptureLimits {
                max_config_bytes: self.core.limits.max_config_bytes,
                max_dense_bytes: self.core.limits.max_dense_bytes,
                max_window_tokens: self.core.max_window_tokens as u64,
                per_cache: capture.per_cache,
                max_total_payload_bytes: capture.max_total_payload_bytes,
                max_text_bytes,
                max_tokenizer_config_bytes,
            },
        };
        if !plan.valid() {
            return Err(reject("invalid recorded capture request"));
        }
        Ok(plan)
    }

    fn record_capture(
        &self,
        source: &mut StagedSalienceSource<'_>,
        plan: CalibrationCapturePlan,
        windows: &[&[u32]],
        limits: StagedCaptureLimits,
        mut recorder: EvidenceRecorder,
    ) -> Result<RecordedStagedCapture, NnError> {
        check_recorder(&recorder)?;
        recorder
            .emit("capture", None, 0, &CalibrationCaptureStarted { plan })
            .map_err(evidence_error)?;
        let caches = self.capture_next_token(source, windows, limits)?;
        let completed = completion(windows, &caches);
        recorder
            .emit("capture", None, 1, &completed)
            .map_err(evidence_error)?;
        let evidence = recorder.seal().map_err(evidence_error)?;
        let views = CalibrationCaptureView::fold(evidence.log()).map_err(evidence_error)?;
        if evidence.log().events.len() != 2
            || views.len() != 1
            || views[0].completed() != &completed
        {
            return Err(reject(
                "recorded capture is not one complete standalone run",
            ));
        }
        Ok(RecordedStagedCapture { caches, evidence })
    }
}

/// Cold replay from actual admitted B and explicit receiving logical limits.
/// Rebuilds masters and reruns encoding/forward/loss/backward/cache construction;
/// never trusts a supplied model or reuses recorded gradients. Checks all ordered
/// bindings and encoded windows. NOT RSS/out-of-core/quality/independent clearance.
/// # Errors
/// Invalid standalone lifecycle, receiver caps, stage/source/assets, unsupported
/// text feature, actual execution failure or a rehashed fabricated cache claim.
pub fn replay_staged_capture(
    log: &VerifiedLog,
    source: &mut StagedSalienceSource<'_>,
    receiving: CalibrationCaptureLimits,
) -> Result<Vec<CalibrationCache>, NnError> {
    let views = CalibrationCaptureView::fold(log).map_err(evidence_error)?;
    if views.len() != 1 || log.events.len() != 2 {
        return Err(reject("replay requires standalone complete capture"));
    }
    let view = &views[0];
    let plan = &view.started().plan;
    if !plan.limits.admitted_by(&receiving)
        || source.package_id() != plan.stage_package
        || source.source_id() != plan.source
        || source.manifest().transform != plan.transform
    {
        return Err(reject("capture replay stage or receiving limits disagree"));
    }
    let limits = plan.limits;
    let assets = &source.manifest().observation.assets;
    if !assets
        .iter()
        .any(|a| a.name == "tokenizer.json" && a.content == plan.tokenizer)
        || plan.tokenizer_config.is_some_and(|content| {
            !assets
                .iter()
                .any(|a| a.name == "tokenizer_config.json" && a.content == content)
        })
    {
        return Err(reject(
            "capture replay actual tokenizer asset identities differ",
        ));
    }
    let model = StagedCalibrationModel::new(
        source,
        StagedCalibrationLimits {
            max_config_bytes: limits.max_config_bytes,
            max_dense_bytes: limits.max_dense_bytes,
            max_window_tokens: usize::try_from(limits.max_window_tokens)
                .map_err(|_| reject("capture replay window overflows usize"))?,
        },
    )?;
    let capture = StagedCaptureLimits {
        per_cache: limits.per_cache,
        max_total_payload_bytes: limits.max_total_payload_bytes,
    };
    let windows = match &plan.input {
        CalibrationCaptureInput::Tokens { windows } => windows.clone(),
        CalibrationCaptureInput::RawText { windows } => {
            #[cfg(feature = "tokenizer")]
            {
                let texts: Vec<_> = windows.iter().map(String::as_str).collect();
                model.encode_capture_text(
                    source,
                    &texts,
                    super::StagedTextCaptureLimits {
                        max_text_bytes: limits.max_text_bytes,
                        max_tokenizer_config_bytes: limits.max_tokenizer_config_bytes,
                    },
                    capture,
                )?
            }
            #[cfg(not(feature = "tokenizer"))]
            {
                let _ = windows;
                return Err(reject("text capture replay requires tokenizer feature"));
            }
        }
        _ => return Err(reject("unsupported capture input profile")),
    };
    let actual_plan = model.capture_plan(
        source,
        plan.input.clone(),
        capture,
        limits.max_text_bytes,
        limits.max_tokenizer_config_bytes,
    )?;
    if actual_plan != *plan {
        return Err(reject("capture replay actual B context differs"));
    }
    let windows: Vec<_> = windows.iter().map(Vec::as_slice).collect();
    let caches = model.capture_next_token(source, &windows, capture)?;
    if completion(&windows, &caches) != *view.completed() {
        return Err(reject(
            "capture replay actual encoded windows or cache outputs differ",
        ));
    }
    Ok(caches)
}

fn completion(windows: &[&[u32]], caches: &[CalibrationCache]) -> CalibrationCaptureCompleted {
    CalibrationCaptureCompleted {
        windows: windows.iter().map(|w| w.to_vec()).collect(),
        caches: caches.iter().map(|c| c.binding().clone()).collect(),
    }
}
fn check_recorder(recorder: &EvidenceRecorder) -> Result<(), NnError> {
    recorder
        .check_emit::<CalibrationCaptureStarted>("capture", None, 0)
        .map_err(evidence_error)?;
    recorder
        .check_emit::<CalibrationCaptureCompleted>("capture", None, 1)
        .map_err(evidence_error)
}
fn evidence_error(error: tritium_evidence::EvidenceError) -> NnError {
    reject(&format!("capture evidence: {error}"))
}
fn reject(message: &str) -> NnError {
    NnError::Backend(message.into())
}
