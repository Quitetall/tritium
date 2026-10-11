//! Separate calibrated model lifecycle over the shared complete-stage validator.
use crate::{
    CalibratedFitReceiptView, DirectFitReceiptView, EventRegistration, EvidenceError,
    EvidenceEvent, VerifiedLog,
};
use tritium_schema::{
    CalibratedQuantizationCompleted, CalibratedQuantizationStarted, ModelManifest, OriginalSourceId,
};

impl EvidenceEvent for CalibratedQuantizationStarted {
    const SCHEMA_ID: &'static str = "tritium.quantize.calibrated_model_started";
    const VERSION: u16 = 1;
}
impl EvidenceEvent for CalibratedQuantizationCompleted {
    const SCHEMA_ID: &'static str = "tritium.quantize.calibrated_model_completed";
    const VERSION: u16 = 1;
}
/// Explicit registration for actual calibrated model requests.
pub const CALIBRATED_QUANTIZATION_STARTED_REGISTRATION: EventRegistration = EventRegistration {
    schema: CalibratedQuantizationStarted::SCHEMA_ID,
    version: 1,
};
/// Explicit registration for successful calibrated fitting, not publication.
pub const CALIBRATED_QUANTIZATION_COMPLETED_REGISTRATION: EventRegistration = EventRegistration {
    schema: CalibratedQuantizationCompleted::SCHEMA_ID,
    version: 1,
};

/// Complete calibrated fit-stage ancestry, not independent numerical truth.
#[derive(Clone, Debug, PartialEq)]
pub struct CalibratedQuantizationReceiptView {
    pub(crate) start: CalibratedQuantizationStarted,
    pub(crate) completed: CalibratedQuantizationCompleted,
    pub(crate) fits: Vec<DirectFitReceiptView>,
    pub(crate) calibrated: Vec<CalibratedFitReceiptView>,
    pub(crate) manifest: ModelManifest,
    pub(crate) source_id: OriginalSourceId,
}
impl CalibratedQuantizationReceiptView {
    /// Actual typed complete model request.
    #[must_use]
    pub fn started(&self) -> &CalibratedQuantizationStarted {
        &self.start
    }
    /// Actual consumed original source and complete encoded outputs.
    #[must_use]
    pub fn completed(&self) -> &CalibratedQuantizationCompleted {
        &self.completed
    }
    /// Actual explicitly uncalibrated fits within the mixed request.
    #[must_use]
    pub fn fits(&self) -> &[DirectFitReceiptView] {
        &self.fits
    }
    /// Actual cache-bound calibrated fits, never reinterpreted as plain events.
    #[must_use]
    pub fn calibrated_fits(&self) -> &[CalibratedFitReceiptView] {
        &self.calibrated
    }
    /// Generic direct-source manifest with actual sealed calibrated ancestry.
    #[must_use]
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }
    /// Source identity derived from recorded raw inventory and checked against request.
    #[must_use]
    pub fn source_id(&self) -> OriginalSourceId {
        self.source_id
    }
    /// Shared complete-stage fold, including source/cache/fit-class agreement.
    /// # Errors
    /// Invalid integrity, payload/version, inventories, source or fit/output ancestry.
    pub fn fold(log: &VerifiedLog) -> Result<Vec<Self>, EvidenceError> {
        super::quantization::fold_calibrated(log)
    }
}
