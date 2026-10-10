//! Source-bound recording around the existing production direct-level controller.

use super::{DirectFitError, DirectTensorFit, fit_direct_levels};
use tritium_evidence::{DirectFitReceiptView, EvidenceError, EvidenceRecorder};
use tritium_format::trit_model::{TritModelError, additive_semantic_digest};
use tritium_schema::{
    DirectFitMetric, DirectFitOutput, DirectFitPlan, DirectLevelsFitted, FitInputDigest,
};

/// Successful fits and their immutable initial observation from the actual log.
/// Does not assign model `Direct` provenance or seal a completed-run root.
#[derive(Debug)]
pub struct RecordedDirectFit {
    /// Owned validated outputs; mutation cannot change the recorded observation.
    pub levels: Vec<DirectTensorFit>,
    /// Initial observation, not a whole-model/release receipt.
    pub observation: DirectFitReceiptView,
}

/// Typed controller/semantic-hash/recording failure; no fallback or success event.
#[derive(Debug)]
#[non_exhaustive]
pub enum RecordedDirectFitError {
    /// Logical tensor name is empty, too long or contains control characters.
    InvalidTensorName,
    /// Existing production controller admission/numerical failure.
    Fit(DirectFitError),
    /// Actual output semantic hashing failed.
    Semantic(TritModelError),
    /// Evidence context or observation recording failed.
    Evidence(EvidenceError),
    /// Actual Rayon pool size cannot be represented in the observation.
    ThreadCountOverflow,
}

impl core::fmt::Display for RecordedDirectFitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "recorded direct fit rejected: {self:?}")
    }
}
impl std::error::Error for RecordedDirectFitError {}

/// Execute the production controller and bind its original inputs and outputs.
///
/// Checks the recorder context before numerical work. Input digests preserve
/// actual f32/f64 bits; they never claim original BF16/HF file or calibration
/// dataset identity. Calls the unchanged `fit_direct_levels`, then uses the
/// format's shared semantic hash, not a parallel tensor identity implementation.
/// This is still the per-tensor step toward the full model source/sink engine.
/// # Errors
/// Invalid name/context, production fit, output hashing or event recording.
#[allow(clippy::too_many_arguments)]
pub fn fit_direct_levels_recorded(
    tensor_name: &str,
    plan: &DirectFitPlan,
    weights: &[f32],
    gram: Option<&[f64]>,
    recorder: &mut EvidenceRecorder,
    span: &str,
    parent: Option<&str>,
    logical_time: u64,
) -> Result<RecordedDirectFit, RecordedDirectFitError> {
    if tensor_name.is_empty()
        || tensor_name.len() > 4096
        || tensor_name.chars().any(char::is_control)
    {
        return Err(RecordedDirectFitError::InvalidTensorName);
    }
    recorder
        .check_emit::<DirectLevelsFitted>(span, parent, logical_time)
        .map_err(RecordedDirectFitError::Evidence)?;
    let rayon_threads = u32::try_from(rayon::current_num_threads())
        .map_err(|_| RecordedDirectFitError::ThreadCountOverflow)?;
    let levels = fit_direct_levels(plan, weights, gram).map_err(RecordedDirectFitError::Fit)?;

    let mut source = blake3::Hasher::new_derive_key("tritium.quantize.source-f32.v1");
    source.update(&plan.layout.rows.to_le_bytes());
    source.update(&plan.layout.cols.to_le_bytes());
    for value in weights {
        source.update(&value.to_bits().to_le_bytes());
    }
    let metric = match gram {
        None => DirectFitMetric::Identity,
        Some(values) => {
            let mut hash = blake3::Hasher::new_derive_key("tritium.quantize.input-gram-f64.v1");
            hash.update(&plan.layout.cols.to_le_bytes());
            for value in values {
                hash.update(&value.to_bits().to_le_bytes());
            }
            DirectFitMetric::InputGramF64 {
                content: FitInputDigest(*hash.finalize().as_bytes()),
            }
        }
    };
    let outputs = levels
        .iter()
        .map(|level| {
            Ok(DirectFitOutput {
                level: level.level.clone(),
                planes: level.tensor.plane_count(),
                content: additive_semantic_digest(&level.tensor)
                    .map_err(RecordedDirectFitError::Semantic)?,
            })
        })
        .collect::<Result<Vec<_>, RecordedDirectFitError>>()?;
    let observation = recorder
        .emit_direct_levels_fitted(
            span,
            parent,
            logical_time,
            &DirectLevelsFitted {
                tensor: tensor_name.into(),
                plan: plan.clone(),
                source_f32: FitInputDigest(*source.finalize().as_bytes()),
                metric,
                rayon_threads,
                outputs,
            },
        )
        .map_err(RecordedDirectFitError::Evidence)?;
    Ok(RecordedDirectFit {
        levels,
        observation,
    })
}
