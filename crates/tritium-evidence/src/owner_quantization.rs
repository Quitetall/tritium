//! Distinct owner-model lifecycle over the shared complete-stage validator.
use crate::{
    CalibratedFitReceiptView, DirectFitReceiptView, EventRegistration, EvidenceError,
    EvidenceEvent, OwnerScoreFitReceiptView, VerifiedLog,
};
use tritium_schema::{
    EvidenceReference, ModelManifest, OriginalSourceId, OwnerQuantizationCompleted,
    OwnerQuantizationStarted,
};
impl EvidenceEvent for OwnerQuantizationStarted {
    const SCHEMA_ID: &'static str = "tritium.quantize.owner_model_started";
    const VERSION: u16 = 1;
}
impl EvidenceEvent for OwnerQuantizationCompleted {
    const SCHEMA_ID: &'static str = "tritium.quantize.owner_model_completed";
    const VERSION: u16 = 1;
}
/// Registered complete owner-model request, not an old K-FAC start.
pub const OWNER_QUANTIZATION_STARTED_REGISTRATION: EventRegistration = EventRegistration {
    schema: OwnerQuantizationStarted::SCHEMA_ID,
    version: 1,
};
/// Registered actual owner-model fitting completion, not publication/qualification.
pub const OWNER_QUANTIZATION_COMPLETED_REGISTRATION: EventRegistration = EventRegistration {
    schema: OwnerQuantizationCompleted::SCHEMA_ID,
    version: 1,
};

/// Complete verified fit ancestry for a model containing actual owner fits.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnerQuantizationReceiptView {
    pub(crate) start: OwnerQuantizationStarted,
    pub(crate) completed: OwnerQuantizationCompleted,
    pub(crate) fits: Vec<DirectFitReceiptView>,
    pub(crate) calibrated: Vec<CalibratedFitReceiptView>,
    pub(crate) owners: Vec<OwnerScoreFitReceiptView>,
    pub(crate) manifest: ModelManifest,
    pub(crate) source_id: OriginalSourceId,
    pub(crate) reference: EvidenceReference,
}
impl OwnerQuantizationReceiptView {
    /// Actual run/span/sealed-root reference, not independent qualification.
    pub fn reference(&self) -> &EvidenceReference {
        &self.reference
    }
    /// Actual complete original request and cache expectations.
    pub fn started(&self) -> &OwnerQuantizationStarted {
        &self.start
    }
    /// Actual consumed source and complete encoded outputs.
    pub fn completed(&self) -> &OwnerQuantizationCompleted {
        &self.completed
    }
    /// Explicit plain direct fits, not owner fits with relabeled metrics.
    pub fn fits(&self) -> &[DirectFitReceiptView] {
        &self.fits
    }
    /// Actual legacy calibrated fits in a mixed request.
    pub fn calibrated_fits(&self) -> &[CalibratedFitReceiptView] {
        &self.calibrated
    }
    /// Actual recorded owner fits with complete row metric bindings.
    pub fn owner_fits(&self) -> &[OwnerScoreFitReceiptView] {
        &self.owners
    }
    /// Generic model manifest with actual verified Direct ancestry.
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }
    /// Original identity derived from recorded consumed inventory.
    pub fn source_id(&self) -> OriginalSourceId {
        self.source_id
    }
    /// Shared complete-stage validation, not independent source/recipe/numerical proof.
    pub fn fold(log: &VerifiedLog) -> Result<Vec<Self>, EvidenceError> {
        super::quantization::fold_owner(log)
    }
}
