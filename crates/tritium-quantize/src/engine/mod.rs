//! Shared production numerical kernels for conversion and training (ADR 0044 D11).
//!
//! These kernels preserve the legacy F32Accum64 ladder semantics. They are the
//! numerical implementation shared by the typed model source/sink engine.
//! The f64 feedback oracle remains an explicitly separate reference profile.

mod activation_aware;
mod calibrated_fit;
mod calibration;
mod calibration_cache;
mod calibration_sources;
mod direct;
mod joint_feedback;
mod ladder;
mod model;
mod sinks;
mod sources;

pub use model::{
    Engine, PublicationWarning, QuantEngineError, QuantizedModel, TensorSink, WeightSource,
    verify_package_with_fit_log,
};
pub use sinks::{DirectoryTensorSink, MemoryTensorSink};
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
pub use recorded_calibrated::{
    RecordedCalibratedFit, RecordedCalibratedFitError, fit_calibrated_levels_recorded,
};

pub use activation_aware::{
    ActivationAwareConfig, ScaleRefitMode, auto_decay, fit_tensor,
    fit_tensor_with_scale_refit_mode, fit_to_dense,
};
pub use ladder::{fit_ladder_group, ladder_quantize_at, ladder_step};
