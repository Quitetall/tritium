//! Actual private capture -> shared transform/join -> mandatory B, without resupplied moments.
use super::{RecordedOriginalSalienceCapture, replay_original_salience_capture};
use crate::NnError;
use tritium_evidence::{
    EvidenceRecorder, OriginalSalienceCaptureView, SealedEvidence, VerifiedLog,
};
use tritium_quantize::engine::{
    DirectoryTensorSink, ObservedWeightSource, PublishedSalienceStage, RecordedSalienceSource,
    SourceTransform, StagedSalienceSource, WeightSource,
};
use tritium_schema::{
    OriginalSalienceCaptureLimits, PackageId, SalienceStageLimits, SalienceTransformPlan,
};

/// Owns actual A capture and recorded immutable B; does not qualify capture/quality/RSS.
/// Private fields prevent replacement of moments/logs after actual NN execution.
pub struct RecordedOriginalCapturedStage {
    capture: RecordedOriginalSalienceCapture,
    transform: RecordedSalienceSource,
}
impl std::fmt::Debug for RecordedOriginalCapturedStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordedOriginalCapturedStage")
            .field(
                "source",
                &self.transform.source.observation().derived_source,
            )
            .field(
                "original_capture_root",
                &self.capture.evidence().log().root_digest,
            )
            .finish_non_exhaustive()
    }
}
impl RecordedOriginalSalienceCapture {
    /// Consume actual immutable capture through the shared original-bound transform.
    /// Explicit plan policy must match the derived complete capture context; no source,
    /// graph partner, statistic assertion or replacement capture is authoritative.
    /// The returned module publishes only the mandatory original-captured B profile.
    /// # Errors
    /// Foreign source/plan/consumers/moments, shared transform/numerical or recorder failure.
    pub fn transform_recorded(
        self,
        plan: &SalienceTransformPlan,
        source: &mut ObservedWeightSource<'_>,
        recorder: EvidenceRecorder,
    ) -> Result<RecordedOriginalCapturedStage, NnError> {
        let transform = SourceTransform::run_original_captured_recorded(
            plan,
            source,
            self.statistics(),
            self.evidence().log(),
            recorder,
        )
        .map_err(error)?;
        Ok(RecordedOriginalCapturedStage {
            capture: self,
            transform,
        })
    }
}
impl RecordedOriginalCapturedStage {
    /// Bounded mandatory stage encoding, no resupplied statistics or provenance.
    /// # Errors
    /// Complete original/transform/payload disagreement, limits or allocation failure.
    pub fn encode_stage_package(&self, limits: SalienceStageLimits) -> Result<Vec<u8>, NnError> {
        self.transform
            .encode_original_captured_stage_package(
                self.capture.statistics(),
                self.capture.evidence().log(),
                limits,
            )
            .map_err(error)
    }

    /// The same single common writer without a retained second package buffer.
    /// # Errors
    /// Admission/caps or writer failure; a failed destination may contain a prefix.
    pub fn write_stage_package_to(
        &self,
        limits: SalienceStageLimits,
        destination: &mut impl std::io::Write,
    ) -> Result<PackageId, NnError> {
        self.transform
            .write_original_captured_stage_package_to(
                self.capture.statistics(),
                self.capture.evidence().log(),
                limits,
                destination,
            )
            .map_err(error)
    }

    /// Existing no-clobber publication of actual B and transform evidence.
    /// Original evidence remains private/available separately through its accessor.
    /// # Errors
    /// Original/transform/payload/cap disagreement or pre-commit I/O failure.
    pub fn publish_stage(
        &self,
        limits: SalienceStageLimits,
        destination: &mut DirectoryTensorSink,
    ) -> Result<PublishedSalienceStage, NnError> {
        self.transform
            .publish_original_captured_stage(
                self.capture.statistics(),
                self.capture.evidence().log(),
                limits,
                destination,
            )
            .map_err(error)
    }

    /// Complete actual A-capture evidence; may contain private raw text.
    #[must_use]
    pub fn original_capture_evidence(&self) -> &SealedEvidence {
        self.capture.evidence()
    }

    /// Actual complete transform evidence; metadata is not independent qualification.
    #[must_use]
    pub fn transform_evidence(&self) -> &SealedEvidence {
        &self.transform.evidence
    }
}

/// Cold actual A salience and A->B arithmetic replay for an admitted original-captured B.
/// Keeps structural, numerical and independent quality/release verification distinct.
/// # Errors
/// Wrong stage/reference/log/caps, actual A/tokenizer/moments or folded B disagreement.
pub fn verify_original_captured_stage(
    stage: &StagedSalienceSource<'_>,
    original_capture_log: &VerifiedLog,
    original: &mut dyn WeightSource,
    receiving: OriginalSalienceCaptureLimits,
) -> Result<(), NnError> {
    let mut views = OriginalSalienceCaptureView::fold(original_capture_log).map_err(error)?;
    if views.len() != 1
        || original_capture_log.events.len() != 2
        || stage.original_capture_reference() != Some(views.remove(0).reference())
    {
        return Err(error("stage differs from required original capture"));
    }
    let actual = replay_original_salience_capture(original_capture_log, original, receiving)?;
    let stored = stage.statistics().map_err(error)?;
    if actual.len() != stored.len()
        || actual.iter().any(|(name, values)| {
            stored.get(name).is_none_or(|expected| {
                values.rows != expected.rows
                    || values.moments.len() != expected.moments.len()
                    || values
                        .moments
                        .iter()
                        .zip(&expected.moments)
                        .any(|(a, b)| a.to_bits() != b.to_bits())
            })
        })
    {
        return Err(error("actual original moments differ from stage payloads"));
    }
    stage.verify_against_original(original).map_err(error)
}
fn error(detail: impl std::fmt::Display) -> NnError {
    NnError::Backend(format!("original captured stage: {detail}"))
}
