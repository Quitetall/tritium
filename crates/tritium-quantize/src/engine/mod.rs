//! Shared production numerical kernels for conversion and training (ADR 0044 D11).
//!
//! These kernels preserve the legacy F32Accum64 ladder semantics. They are the
//! numerical implementation shared by the typed model source/sink engine.
//! The f64 feedback oracle remains an explicitly separate reference profile.

mod activation_aware;
mod cache_scalar;
mod calibrated_fit;
mod calibration;
mod calibration_cache;
mod calibration_sources;
mod direct;
mod file_source;
mod joint_feedback;
mod ladder;
mod model;
mod model_import;
mod observed_source;
mod original_capture;
mod owner_score_cache;
mod owner_score_metric;
pub use owner_score_metric::{
    OwnerRowMetric, OwnerScoreMetricError, OwnerScoreMetricLimits, OwnerScoreMetrics,
    WeightedOwnerWindow, fit_owner_score_levels,
};
mod owner_scores;
pub use owner_score_cache::{OwnerScoreCache, OwnerScoreCacheError, OwnerScoreCacheInput};
mod salience;
mod salience_model;
mod sinks;
mod source_stage;
mod source_transform;
mod sources;

pub use observed_source::ObservedWeightSource;
pub use owner_scores::{OwnedOwnerScoreWindow, OwnerScoreError, OwnerScoreUse, OwnerScoreWindow};
pub use salience::{SalienceFoldError, SaliencePair, divide_rows, scale_cols, smooth_scales};
pub use salience_model::JoinedSalienceModel;

pub use file_source::{
    FileWeightSource, file_calibration_source, open_original_file, read_original_file_bounded,
};
pub use model::{
    Engine, PublicationWarning, QuantEngineError, QuantizedModel, TensorSink, WeightSource,
    verify_package_with_fit_log,
};
pub use model_import::{
    ImportedModelEvidenceView, ImportedSaltV2Model, SaltV2MatrixImportPlan, SaltV2ModelGuardError,
    SaltV2ModelImportError, SaltV2ModelImportPlan, SaltV2ModelImportRequest, import_salt_v2_model,
    import_salt_v2_model_bound, inspect_salt_v2_model_import, verify_imported_model_evidence,
    verify_imported_model_with_log,
};
pub use sinks::{DirectoryTensorSink, MemoryTensorSink, TritModelPublicationError};
pub use source_stage::{PublishedSalienceStage, StagedSalienceSource};
pub use source_transform::{
    DeclaredSalienceStatistics, DerivedSalienceSource, RecordedSalienceSource, SalienceSourceError,
    SourceTransform,
};
pub use sources::{InMemoryWeightSource, RawTensor, SafeTensorsWeightSource};

pub use calibrated_fit::{CalibratedFitError, fit_calibrated_levels};
pub use calibration::{
    CalibratedTensor, CalibrationBatch, CalibrationCollector, CalibrationError, CalibrationInput,
    CalibrationOutputBatch,
};
pub use calibration_cache::{CalibrationCache, CalibrationCacheBuilder, CalibrationCacheError};
pub use calibration_sources::{
    CalibrationSource, InMemoryCalibrationSource, TritCalibrationSource,
};

pub use direct::{DirectFitError, DirectTensorFit, fit_direct_levels};

mod recorded;
pub use recorded::{RecordedDirectFit, RecordedDirectFitError, fit_direct_levels_recorded};

mod recorded_calibrated;
mod recorded_owner_score;
pub use recorded_calibrated::{
    RecordedCalibratedFit, RecordedCalibratedFitError, fit_calibrated_levels_recorded,
};
pub use recorded_owner_score::{
    RecordedOwnerScoreFit, RecordedOwnerScoreFitError, fit_owner_score_levels_recorded,
};

pub use activation_aware::{
    ActivationAwareConfig, ScaleRefitMode, auto_decay, fit_tensor,
    fit_tensor_with_scale_refit_mode, fit_to_dense,
};
pub use ladder::{fit_ladder_group, ladder_quantize_at, ladder_step};
