//! Existing common full-Gram capture to the actual native QTEA work seam.
use crate::{QteaTensorOrigin, QteaTensorWorkError, QteaTensorWorkPlan};
use std::{error::Error, fmt};
use tritium_format::qtea::QteaLimits;
use tritium_quantize::{
    engine::CalibratedTensor,
    qtea_reference::{TensorFit, TensorFitConfig},
};
use tritium_schema::{
    CalibrationConsumer, CalibrationOutput, CalibrationPartition, OriginalSourceId,
    QuantSourceTensor,
};

/// Actual numerical capture and expected context; assertions are not source custody.
pub struct QteaCalibratedTensorInput<'a> {
    /// Expected original tensor interpretation, including stored precision.
    pub original: &'a QuantSourceTensor,
    /// Expected calibration-source model content, not the campaign artifact ID.
    pub source_model: OriginalSourceId,
    /// Actual widened numerical input; raw stored bytes are verified by its producer.
    pub weights: &'a [f32],
    /// Actual immutable common collector result, not factors reconstructed from labels.
    pub calibration: &'a CalibratedTensor,
    /// Canonical owner and declared original-storage/campaign binding.
    pub origin: QteaTensorOrigin<'a>,
}
impl fmt::Debug for QteaCalibratedTensorInput<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QteaCalibratedTensorInput")
            .field("original", self.original)
            .field("source_model", &self.source_model)
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}

impl<'a> QteaTensorWorkPlan<'a> {
    /// Consume actual input-only Full/Matmul calibration without copying its Gram.
    ///
    /// Preserves the common selected-weight normalization, masks, ordinals and
    /// source/data digests. Not author-sequence-normalized capture reproduction,
    /// original-byte authentication, full-corpus proof or physical RAM admission.
    /// Capture/cache/metadata residency remains caller scope beyond these fit caps.
    /// The capture cannot be dropped while its actual Gram is borrowed:
    /// ```compile_fail,E0505
    /// use tritium_salt::{ContentId, QteaCalibratedTensorInput, QteaTensorOrigin, QteaTensorWorkPlan};
    /// use tritium_quantize::{engine::CalibratedTensor, qtea_reference::TensorFitConfig};
    /// use tritium_schema::{OriginalSourceId, QuantSourceTensor};
    /// use tritium_format::qtea::QteaLimits;
    /// fn invalid(calibration: CalibratedTensor, original: QuantSourceTensor, weights: &[f32]) {
    ///     let work = QteaTensorWorkPlan::from_calibrated(QteaCalibratedTensorInput {
    ///         original: &original, source_model: OriginalSourceId::from_bytes([1;32]),
    ///         weights, calibration: &calibration, origin: QteaTensorOrigin {
    ///             source_id: ContentId::of_bytes(b"campaign"), owner: &original.name,
    ///             tensor_index: 0, source_tensor_digest: [9;32],
    ///         },
    ///     }, TensorFitConfig { group_size: 2, ..Default::default() },
    ///     QteaLimits { max_tensor_bytes: 4096, max_validation_bytes: 4096 }, usize::MAX, usize::MAX).unwrap();
    ///     drop(calibration);
    ///     let _ = work.fit_to(&mut Vec::new());
    /// }
    /// ```
    ///
    /// # Errors
    /// Refuses context/owner/source disagreement, approximated/grouped/gather or
    /// output-factor capture, incompatible shape and existing numerical/codec/cap
    /// failures. No fitting or sink access occurs during this planning call.
    pub fn from_calibrated(
        input: QteaCalibratedTensorInput<'a>,
        config: TensorFitConfig,
        limits: QteaLimits,
        max_native_work_bytes: usize,
        max_logical_bytes: usize,
    ) -> Result<Self, QteaCalibrationError> {
        let factors = input.calibration.factors();
        let binding = &factors.plan;
        if binding.tensor != *input.original
            || binding.source.model.0 != *input.source_model.as_bytes()
            || input.origin.owner != input.original.name
            || binding.validate().is_err()
        {
            return Err(QteaCalibrationError::ContextMismatch);
        }
        if binding.partition != CalibrationPartition::Full
            || binding.consumer != CalibrationConsumer::Matmul
            || binding.output != CalibrationOutput::None
            || input.calibration.output().is_some()
        {
            return Err(QteaCalibrationError::RequiresInputOnlyFullGram);
        }
        let [rows, columns] = input.original.shape.as_slice() else {
            return Err(QteaCalibrationError::Shape);
        };
        let [gram] = input.calibration.inputs() else {
            return Err(QteaCalibrationError::RequiresInputOnlyFullGram);
        };
        if u64::try_from(gram.dimension()).ok() != Some(*columns) {
            return Err(QteaCalibrationError::Shape);
        }
        let source = gram
            .source_id()
            .ok_or(QteaCalibrationError::ContextMismatch)?;
        if source.source_model_digest() != binding.source.model.0
            || source.activation_cache_digest() != binding.source.cache.0
            || source.token_stream_digest() != binding.source.tokens.0
        {
            return Err(QteaCalibrationError::ContextMismatch);
        }
        let rows = usize::try_from(*rows).map_err(|_| QteaCalibrationError::Shape)?;
        let columns = usize::try_from(*columns).map_err(|_| QteaCalibrationError::Shape)?;
        let numerical = TensorFit::plan(
            input.weights,
            rows,
            columns,
            gram,
            config,
            max_native_work_bytes,
        )
        .map_err(QteaTensorWorkError::from)?;
        Ok(Self::new(
            numerical,
            input.origin,
            limits,
            max_logical_bytes,
        )?)
    }
}

/// Calibration-context or actual native planning refusal; no successful work.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum QteaCalibrationError {
    /// Expected source/tensor/owner or actual Gram source binding disagrees.
    ContextMismatch,
    /// Only explicitly full original Matmul input statistics, without Fisher.
    RequiresInputOnlyFullGram,
    /// Unsupported platform/tensor/full-Gram geometry.
    Shape,
    /// Existing native numerical, origin, codec or logical admission error.
    Work(QteaTensorWorkError),
}
impl fmt::Display for QteaCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QTEA calibration rejected: {self:?}")
    }
}
impl Error for QteaCalibrationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Work(e) => Some(e),
            _ => None,
        }
    }
}
impl From<QteaTensorWorkError> for QteaCalibrationError {
    fn from(e: QteaTensorWorkError) -> Self {
        Self::Work(e)
    }
}
