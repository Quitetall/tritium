//! Mandatory structural joined admission over the actual B stage and fit log.
use super::model::{ConversionView, bound, conversion_views, platform, verify_conversion_model};
use super::{QuantEngineError, SalienceSourceError, StagedSalienceSource, WeightSource};
use tritium_evidence::{CalibrationCaptureView, EvidenceEvent, VerifiedLog};
use tritium_format::read_trit_package_with_manifest_limit;
use tritium_format::trit_model::{TritModel, bind_trit_model_payloads};
use tritium_format::trit_salience_conversion::{
    decode_captured_salience_conversion_manifest,
    decode_original_captured_salience_conversion_manifest, decode_salience_conversion_manifest,
    encode_captured_salience_conversion_manifest,
    encode_original_captured_salience_conversion_manifest, encode_salience_conversion_manifest,
};
use tritium_schema::{
    CalibratedLevelsFitted, CalibratedQuantAction, CalibratedQuantPlan,
    CalibratedQuantizationCompleted, CalibratedQuantizationStarted,
    CapturedSalienceConversionManifest, DirectLevelsFitted, EvidenceReference,
    OriginalCapturedSalienceConversionManifest, PackageId, QuantizationCompleted,
    QuantizationStarted, SalienceConversionLimits, SalienceConversionManifest,
    SalienceSourceStageManifest, SchemaId,
};

/// A complete C carrier bound to actual B payload and both real logs.
/// Structural/source/output agreement is not independent transform arithmetic,
/// captured-graph execution, fit optimality, model quality or release qualification.
#[derive(Debug)]
pub struct JoinedSalienceModel<'a> {
    model: TritModel<'a>,
    manifest: SalienceConversionManifest,
    capture: Option<EvidenceReference>,
    original_capture: Option<EvidenceReference>,
}
impl<'a> JoinedSalienceModel<'a> {
    /// Open only with an actual checked B stage (including its transform log)
    /// and complete actual conversion log. Ordinary readers reject this profile.
    /// # Errors
    /// Unknown/malformed carrier, payloads, caps or any joined stage/log/output disagreement.
    pub fn open(
        bytes: &'a [u8],
        stage: &StagedSalienceSource<'_>,
        fit_log: &VerifiedLog,
        limits: SalienceConversionLimits,
    ) -> Result<Self, QuantEngineError> {
        SalienceProductionContext::from_stage(stage, limits)?.open(bytes, fit_log)
    }

    /// Open the distinct captured profile only with actual B and both complete logs.
    /// Reuses joined payload/source/output admission; numerical capture needs cold replay.
    /// The resulting view retains C metadata/reference, not B or raw capture text.
    /// # Errors
    /// Missing/foreign/partial capture, stage/fit/cache/root disagreement or caps.
    pub fn open_captured(
        bytes: &'a [u8],
        stage: &StagedSalienceSource<'_>,
        capture_log: &VerifiedLog,
        fit_log: &VerifiedLog,
        limits: SalienceConversionLimits,
    ) -> Result<Self, QuantEngineError> {
        SalienceProductionContext::from_capture(stage, capture_log, limits)?.open(bytes, fit_log)
    }

    /// Open only the distinct original-A plus B capture carrier and admitted stage.
    /// Reuses common output/runtime admission, retaining references rather than text/B.
    /// # Errors
    /// Missing original-captured stage, either foreign root/cache/fit/payload or byte cap.
    pub fn open_original_captured(
        bytes: &'a [u8],
        stage: &StagedSalienceSource<'_>,
        capture_log: &VerifiedLog,
        fit_log: &VerifiedLog,
        limits: SalienceConversionLimits,
    ) -> Result<Self, QuantEngineError> {
        SalienceProductionContext::from_original_capture(stage, capture_log, limits)?
            .open(bytes, fit_log)
    }

    /// Mandatory original-A capture ancestry; absent from all older C profiles.
    /// This is structural provenance, not numerical or independent qualification.
    #[must_use]
    pub fn original_capture_reference(&self) -> Option<&EvidenceReference> {
        self.original_capture.as_ref()
    }

    /// Mandatory capture ancestry for the captured profile; absent for legacy joined C.
    /// This is structural provenance, not numerical or independent quality clearance.
    #[must_use]
    pub fn capture_reference(&self) -> Option<&EvidenceReference> {
        self.capture.as_ref()
    }

    /// Checked mandatory carrier, not a release receipt.
    #[must_use]
    pub fn manifest(&self) -> &SalienceConversionManifest {
        &self.manifest
    }

    /// The same common named payload/semantic model view, after joined admission.
    #[must_use]
    pub fn model(&self) -> &TritModel<'a> {
        &self.model
    }

    /// Independently re-execute actual A->B numerics with the stage's stored moments.
    /// Caller must supply the same actual B stage; no log-only proof or recreated A.
    /// This is still not captured-graph/model-quality or release qualification.
    /// # Errors
    /// Foreign stage, actual A or numerical replay differs from the mandatory carrier.
    pub fn verify_transform(
        &self,
        stage: &StagedSalienceSource<'_>,
        original: &mut dyn WeightSource,
    ) -> Result<(), SalienceSourceError> {
        if self.manifest.stage_package != stage.package_id()
            || self.manifest.transform != stage.manifest().transform
            || self.manifest.original_source != stage.manifest().observation.original_source
            || self.manifest.derived_source != stage.source_id()
            || self.original_capture.as_ref() != stage.original_capture_reference()
        {
            return Err(QuantEngineError::Binding("foreign stage for joined replay").into());
        }
        stage.verify_against_original(original)
    }

    pub(super) fn into_model(self) -> TritModel<'a> {
        self.model
    }
}

/// Issued only from a checked immutable stage; never accepted from caller assertions.
/// Retains metadata, not another dense source. Used by the single engine package path.
#[derive(Debug)]
pub(super) struct SalienceProductionContext {
    stage_package: PackageId,
    stage: SalienceSourceStageManifest,
    limits: SalienceConversionLimits,
    capture: Option<CalibrationCaptureView>,
    original_capture: Option<EvidenceReference>,
}

enum OriginalCaptureRequirement {
    Absent,
    Required,
}
impl SalienceProductionContext {
    pub(super) fn from_stage(
        stage: &StagedSalienceSource<'_>,
        limits: SalienceConversionLimits,
    ) -> Result<Self, QuantEngineError> {
        Self::from_stage_with_requirement(stage, limits, OriginalCaptureRequirement::Absent)
    }

    fn from_stage_with_requirement(
        stage: &StagedSalienceSource<'_>,
        limits: SalienceConversionLimits,
        requirement: OriginalCaptureRequirement,
    ) -> Result<Self, QuantEngineError> {
        match (stage.original_capture_reference(), requirement) {
            (None, OriginalCaptureRequirement::Absent)
            | (Some(_), OriginalCaptureRequirement::Required) => {}
            _ => {
                return Err(QuantEngineError::Binding(
                    "stage original capture profile differs from requested carrier",
                ));
            }
        }
        limits
            .validate()
            .map_err(|_| QuantEngineError::Limit("salience conversion byte caps"))?;
        Ok(Self {
            stage_package: stage.package_id(),
            stage: stage.manifest().clone(),
            limits,
            capture: None,
            original_capture: stage.original_capture_reference().cloned(),
        })
    }

    pub(super) fn from_capture(
        stage: &StagedSalienceSource<'_>,
        log: &VerifiedLog,
        limits: SalienceConversionLimits,
    ) -> Result<Self, QuantEngineError> {
        Self::from_capture_with_requirement(stage, log, limits, OriginalCaptureRequirement::Absent)
    }

    pub(super) fn from_original_capture(
        stage: &StagedSalienceSource<'_>,
        log: &VerifiedLog,
        limits: SalienceConversionLimits,
    ) -> Result<Self, QuantEngineError> {
        Self::from_capture_with_requirement(
            stage,
            log,
            limits,
            OriginalCaptureRequirement::Required,
        )
    }

    fn from_capture_with_requirement(
        stage: &StagedSalienceSource<'_>,
        log: &VerifiedLog,
        limits: SalienceConversionLimits,
        requirement: OriginalCaptureRequirement,
    ) -> Result<Self, QuantEngineError> {
        let mut context = Self::from_stage_with_requirement(stage, limits, requirement)?;
        let mut views = CalibrationCaptureView::fold(log)?;
        if views.len() != 1 || log.events.len() != 2 {
            return Err(QuantEngineError::Binding(
                "captured conversion requires one complete standalone capture",
            ));
        }
        let view = views.remove(0);
        let plan = &view.started().plan;
        let observation = &context.stage.observation;
        if plan.stage_package != context.stage_package
            || plan.source != observation.derived_source
            || plan.transform != context.stage.transform
            || !observation
                .assets
                .iter()
                .any(|a| a.name == "tokenizer.json" && a.content == plan.tokenizer)
            || plan.tokenizer_config.is_some_and(|content| {
                !observation
                    .assets
                    .iter()
                    .any(|a| a.name == "tokenizer_config.json" && a.content == content)
            })
            || plan.tensors.iter().any(|t| {
                !observation
                    .derived_tensors
                    .iter()
                    .any(|actual| actual.source == *t)
            })
        {
            return Err(QuantEngineError::Binding(
                "capture differs from actual B stage/assets/interpretations",
            ));
        }
        context.capture = Some(view);
        Ok(context)
    }

    pub(super) fn check_capture_plan(
        &self,
        plan: &CalibratedQuantPlan,
    ) -> Result<(), QuantEngineError> {
        let capture = self
            .capture
            .as_ref()
            .ok_or(QuantEngineError::Binding("required capture absent"))?;
        for tensor in &plan.tensors {
            match &tensor.action {
                CalibratedQuantAction::KeepDense => {}
                CalibratedQuantAction::CalibratedFit { plan }
                    if capture.completed().caches.iter().any(|b| b == &plan.cache) => {}
                _ => {
                    return Err(QuantEngineError::Binding(
                        "fit action/cache differs from mandatory capture",
                    ));
                }
            }
        }
        Ok(())
    }

    pub(super) fn manifest_bytes(
        &self,
        view: &ConversionView,
    ) -> Result<Vec<u8>, QuantEngineError> {
        self.check_view(view)?;
        let carrier = SalienceConversionManifest {
            schema: SchemaId { major: 1, minor: 0 },
            model: view.manifest().clone(),
            stage_package: self.stage_package,
            original_source: self.stage.observation.original_source,
            derived_source: self.stage.observation.derived_source,
            transform: self.stage.transform.clone(),
            conversion: view.reference().clone(),
        };
        let maximum = platform(
            self.limits.max_manifest_bytes,
            "salience conversion metadata",
        )?;
        Ok(match (&self.capture, &self.original_capture) {
            (Some(capture), Some(original_capture)) => {
                encode_original_captured_salience_conversion_manifest(
                    &OriginalCapturedSalienceConversionManifest {
                        lineage: CapturedSalienceConversionManifest {
                            lineage: carrier,
                            capture: capture.reference().clone(),
                        },
                        original_capture: original_capture.clone(),
                    },
                    maximum,
                )?
            }
            (Some(capture), None) => encode_captured_salience_conversion_manifest(
                &CapturedSalienceConversionManifest {
                    lineage: carrier,
                    capture: capture.reference().clone(),
                },
                maximum,
            )?,
            (None, None) => encode_salience_conversion_manifest(&carrier, maximum)?,
            (None, Some(_)) => {
                return Err(QuantEngineError::Binding(
                    "original-captured C needs actual B capture",
                ));
            }
        })
    }

    pub(super) fn max_package_bytes(&self) -> u64 {
        self.limits.max_package_bytes
    }

    pub(super) fn open<'a>(
        &self,
        bytes: &'a [u8],
        fit_log: &VerifiedLog,
    ) -> Result<JoinedSalienceModel<'a>, QuantEngineError> {
        bound(
            bytes.len() as u64,
            self.limits.max_package_bytes,
            "salience conversion package",
        )?;
        let maximum = platform(
            self.limits.max_manifest_bytes,
            "salience conversion metadata",
        )?;
        let package = read_trit_package_with_manifest_limit(bytes, maximum)?;
        let carrier = match (&self.capture, &self.original_capture) {
            (Some(capture), Some(original)) => {
                let carrier = decode_original_captured_salience_conversion_manifest(
                    package.manifest(),
                    maximum,
                )?;
                if carrier.original_capture != *original
                    || carrier.lineage.capture != *capture.reference()
                {
                    return Err(QuantEngineError::Binding(
                        "mandatory original/B capture root differs",
                    ));
                }
                carrier.lineage.lineage
            }
            (Some(capture), None) => {
                let carrier =
                    decode_captured_salience_conversion_manifest(package.manifest(), maximum)?;
                if carrier.capture != *capture.reference() {
                    return Err(QuantEngineError::Binding("mandatory capture root differs"));
                }
                carrier.lineage
            }
            (None, None) => decode_salience_conversion_manifest(package.manifest(), maximum)?,
            (None, Some(_)) => {
                return Err(QuantEngineError::Binding(
                    "original-captured C needs actual B capture",
                ));
            }
        };
        let mut views = conversion_views(fit_log)?;
        if views.len() != 1 {
            return Err(QuantEngineError::Binding(
                "expected one complete joined conversion",
            ));
        }
        let view = views.remove(0);
        complete_fit_log(&view, fit_log)?;
        self.check_view(&view)?;
        if carrier.stage_package != self.stage_package
            || carrier.original_source != self.stage.observation.original_source
            || carrier.derived_source != self.stage.observation.derived_source
            || carrier.transform != self.stage.transform
            || carrier.conversion != *view.reference()
        {
            return Err(QuantEngineError::Binding(
                "mandatory joined stage/root disagreement",
            ));
        }
        let model = bind_trit_model_payloads(package, carrier.model.clone())?;
        model.validate_complete_payload_inventory()?;
        verify_conversion_model(&model, &view)?;
        Ok(JoinedSalienceModel {
            model,
            manifest: carrier,
            capture: self.capture.as_ref().map(|c| c.reference().clone()),
            original_capture: self.original_capture.clone(),
        })
    }

    fn check_view(&self, view: &ConversionView) -> Result<(), QuantEngineError> {
        let observation = &self.stage.observation;
        let completed = view.completed();
        if view.source_id() != observation.derived_source
            || completed.sources != observation.derived_tensors
            || completed.assets != observation.assets
            || view.manifest().architecture != observation.plan.architecture
        {
            return Err(QuantEngineError::Binding(
                "conversion differs from actual complete B stage",
            ));
        }
        if let Some(capture) = &self.capture {
            let ConversionView::Calibrated(calibrated) = view else {
                return Err(QuantEngineError::Binding(
                    "captured conversion cannot admit plain fitting",
                ));
            };
            self.check_capture_plan(&calibrated.started().plan)?;
            if !view.fits().is_empty()
                || view.calibrated_fits().iter().any(|fit| {
                    !capture
                        .completed()
                        .caches
                        .iter()
                        .any(|b| b == &fit.observation().plan.cache)
                })
            {
                return Err(QuantEngineError::Binding(
                    "actual fit observations differ from required capture",
                ));
            }
        }
        Ok(())
    }
}

fn complete_fit_log(view: &ConversionView, log: &VerifiedLog) -> Result<(), QuantEngineError> {
    let admitted = [
        QuantizationStarted::SCHEMA_ID,
        QuantizationCompleted::SCHEMA_ID,
        CalibratedQuantizationStarted::SCHEMA_ID,
        CalibratedQuantizationCompleted::SCHEMA_ID,
        DirectLevelsFitted::SCHEMA_ID,
        CalibratedLevelsFitted::SCHEMA_ID,
    ];
    let count = 2usize
        .checked_add(view.fits().len())
        .and_then(|n| n.checked_add(view.calibrated_fits().len()))
        .ok_or(QuantEngineError::Binding("conversion event count overflow"))?;
    if log.events.len() != count
        || log
            .events
            .iter()
            .any(|event| !admitted.contains(&event.schema.as_str()))
    {
        return Err(QuantEngineError::Binding(
            "unknown or unconsumed joined conversion event",
        ));
    }
    Ok(())
}
