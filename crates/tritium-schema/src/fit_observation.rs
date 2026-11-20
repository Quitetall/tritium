//! Candidate input-bound fitting observations (ADR 0057), not release verdicts.

use crate::{DirectFitPlan, SemanticTensorDigest};
use alloc::{string::String, vec::Vec};

/// Exact original fitting input bits under a named candidate digest domain.
/// Not a stored blob, package, HF identity or calibration dataset identity.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FitInputDigest(pub [u8; 32]);

/// Original metric supplied to the controller, before any basis transform.
#[non_exhaustive]
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirectFitMetric {
    /// Explicit plan invocation without an input Gram.
    Identity,
    /// Exact row-major f64 Gram bits; not original calibration data identity.
    InputGramF64 {
        /// Candidate `tritium.quantize.input-gram-f64.v1` digest.
        content: FitInputDigest,
    },
}

/// One independently fitted output's actual logical identity.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectFitOutput {
    /// Level label from the executed plan.
    pub level: String,
    /// Actual uniform plane count of the fitted tensor.
    pub planes: u8,
    /// Existing candidate additive semantic digest, not stored codec bytes.
    pub content: SemanticTensorDigest,
}

/// Successful per-tensor controller observation, not a whole-model receipt.
#[cfg_attr(feature = "schema-gen", derive(schemars::JsonSchema))]
#[cfg_attr(
    feature = "serde",
    derive(serde_nostd::Serialize, serde_nostd::Deserialize)
)]
#[cfg_attr(feature = "serde", serde(crate = "serde_nostd", deny_unknown_fields))]
#[derive(Clone, Debug, PartialEq)]
pub struct DirectLevelsFitted {
    /// Logical tensor name, not a filesystem path or model identity.
    pub tensor: String,
    /// Executed numerical profile, geometry, basis, level requests and limits.
    pub plan: DirectFitPlan,
    /// Candidate `tritium.quantize.source-f32.v1` digest of actual f32 inputs.
    /// This does not assert the identity of original BF16/HF source files.
    pub source_f32: FitInputDigest,
    /// Actual original metric selection/content.
    pub metric: DirectFitMetric,
    /// Active Rayon pool size observed by the production controller.
    pub rayon_threads: u32,
    /// Actual returned outputs in the plan's execution order, not prefixes.
    pub outputs: Vec<DirectFitOutput>,
}
