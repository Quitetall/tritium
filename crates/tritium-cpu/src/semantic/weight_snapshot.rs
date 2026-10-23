//! Complete packed versions; unlike scale overlays, no old root is retained.
use super::{CpuTensor, PackedAdditive};
use std::{any::Any, sync::Arc};
use tritium_core::AdditiveView;
use tritium_runtime::{TensorPayloadBudget, TensorStorageReservation};
use tritium_spec::{BackendError, DeviceBuffer, TensorView};

struct PackedSnapshot {
    // Release packed payload before its admission; forward the concrete backend
    // type exactly as runtime admitted uploads do. No extra execution variant.
    tensor: Box<dyn DeviceBuffer>,
    _reservation: TensorStorageReservation,
}
impl DeviceBuffer for PackedSnapshot {
    fn len_bytes(&self) -> usize {
        self.tensor.len_bytes()
    }
    fn as_any(&self) -> &dyn Any {
        self.tensor.as_any()
    }
}

impl CpuTensor {
    pub(crate) fn snapshot_weights(
        parent: &Arc<dyn DeviceBuffer>,
        candidate: AdditiveView<'_>,
        budget: &TensorPayloadBudget,
    ) -> Result<Arc<dyn DeviceBuffer>, BackendError> {
        let selected = Self::checked(&**parent)?;
        if parent.len_bytes() != selected.len_bytes() {
            return Err(BackendError::InvalidInput(
                "CPU weight snapshot parent payload bytes differ from backend storage".into(),
            ));
        }
        let base = match selected {
            Self::Additive(base) => base,
            Self::ScaleSnapshot(snapshot) => snapshot.base()?,
            Self::Dense { .. } => return Err(BackendError::UnsupportedTensor),
        };
        if !base.matches_geometry(candidate) {
            return Err(BackendError::InvalidInput(
                "CPU weight snapshot candidate geometry differs from parent".into(),
            ));
        }
        // Validate exact scale precision and complete canonical symbols before
        // admission/allocation. Reuse the uploader's plan and packed encoder.
        let caps =
            Self::caps(TensorView::Additive(candidate))?.ok_or(BackendError::UnsupportedTensor)?;
        let reservation = budget.reserve_storage(caps.payload_bytes)?;
        let tensor: Box<dyn DeviceBuffer> =
            Box::new(Self::Additive(PackedAdditive::upload(candidate)?));
        if u64::try_from(tensor.len_bytes()).ok() != Some(caps.payload_bytes) {
            return Err(BackendError::InvalidInput(
                "CPU weight snapshot packed bytes differ from admitted plan".into(),
            ));
        }
        Ok(Arc::new(PackedSnapshot {
            tensor,
            _reservation: reservation,
        }))
    }
}
