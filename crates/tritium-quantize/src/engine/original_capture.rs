//! One original-A capture join for recorded transforms and mandatory B/C ancestry.
//! Structural consistency is not numerical replay or historical authentication.
use super::{SalienceSourceError, model::QuantEngineError};
use tritium_evidence::{OriginalSalienceCaptureView, VerifiedLog};
use tritium_schema::{EvidenceReference, SalienceTransformObservation, SalienceTransformPlan};

pub(super) struct OriginalCaptureJoin {
    view: OriginalSalienceCaptureView,
}
impl OriginalCaptureJoin {
    pub(super) fn from_log(log: &VerifiedLog) -> Result<Self, SalienceSourceError> {
        let mut views = OriginalSalienceCaptureView::fold(log)?;
        if views.len() != 1 || log.events.len() != 2 {
            return Err(binding(
                "original ancestry needs one complete standalone capture",
            ));
        }
        Ok(Self {
            view: views.remove(0),
        })
    }

    pub(super) fn reference(&self) -> &EvidenceReference {
        self.view.reference()
    }

    pub(super) fn check_plan(
        &self,
        plan: &SalienceTransformPlan,
    ) -> Result<(), SalienceSourceError> {
        plan.validate().map_err(SalienceSourceError::Plan)?;
        let original = &self.view.started().plan;
        if plan.architecture != original.architecture
            || plan.original_source != original.source
            || plan
                .tensors
                .iter()
                .ne(original.tensors.iter().map(|b| &b.source))
            || plan
                .assets
                .iter()
                .ne(original.assets.iter().map(|a| &a.name))
            || plan.operations.len() != original.pairs.len()
            || plan
                .operations
                .iter()
                .zip(&original.pairs)
                .zip(&self.view.completed().outputs)
                .any(|((operation, pair), output)| {
                    operation.name != pair.name
                        || operation.projections != pair.projections
                        || operation.inverse != pair.inverse
                        || operation.inverse_axis != pair.inverse_axis
                        || operation.moments.channels != pair.channels
                        || operation.name != output.operation
                        || operation.moments != output.moments
                })
        {
            return Err(binding(
                "transform request differs from mandatory original capture",
            ));
        }
        Ok(())
    }

    pub(super) fn check_observation(
        &self,
        observation: &SalienceTransformObservation,
    ) -> Result<(), SalienceSourceError> {
        observation.validate().map_err(SalienceSourceError::Plan)?;
        self.check_plan(&observation.plan)?;
        let original = &self.view.started().plan;
        if observation.original_source != original.source
            || observation.original_tensors != original.tensors
            || observation.assets != original.assets
        {
            return Err(binding(
                "actual complete original bytes/assets differ from capture",
            ));
        }
        Ok(())
    }
}
fn binding(detail: &'static str) -> SalienceSourceError {
    QuantEngineError::Binding(detail).into()
}
