//! Mandatory structural joined admission over the actual B stage and fit log.
use super::model::{ConversionView, bound, conversion_views, platform, verify_conversion_model};
use super::{QuantEngineError, SalienceSourceError, StagedSalienceSource, WeightSource};
use tritium_evidence::{EvidenceEvent, VerifiedLog};
use tritium_format::read_trit_package_with_manifest_limit;
use tritium_format::trit_model::{TritModel, bind_trit_model_payloads};
use tritium_format::trit_salience_conversion::{
    decode_salience_conversion_manifest, encode_salience_conversion_manifest,
};
use tritium_schema::{
    CalibratedLevelsFitted, CalibratedQuantizationCompleted, CalibratedQuantizationStarted,
    DirectLevelsFitted, PackageId, QuantizationCompleted, QuantizationStarted,
    SalienceConversionLimits, SalienceConversionManifest, SalienceSourceStageManifest, SchemaId,
};

/// A complete C carrier bound to actual B payload and both real logs.
/// Structural/source/output agreement is not independent transform arithmetic,
/// captured-graph execution, fit optimality, model quality or release qualification.
#[derive(Debug)]
pub struct JoinedSalienceModel<'a> {
    model: TritModel<'a>,
    manifest: SalienceConversionManifest,
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
}
impl SalienceProductionContext {
    pub(super) fn from_stage(
        stage: &StagedSalienceSource<'_>,
        limits: SalienceConversionLimits,
    ) -> Result<Self, QuantEngineError> {
        limits
            .validate()
            .map_err(|_| QuantEngineError::Limit("salience conversion byte caps"))?;
        Ok(Self {
            stage_package: stage.package_id(),
            stage: stage.manifest().clone(),
            limits,
        })
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
        Ok(encode_salience_conversion_manifest(
            &carrier,
            platform(
                self.limits.max_manifest_bytes,
                "salience conversion metadata",
            )?,
        )?)
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
        let carrier = decode_salience_conversion_manifest(package.manifest(), maximum)?;
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
