use super::{CpuTensor, PackedAdditive, packed::PackedScaleView};
use std::{fmt, sync::Arc};
use tritium_runtime::{TensorPayloadBudget, TensorStorageReservation};
use tritium_spec::{BackendError, DeviceBuffer};

/// Logical bytes reachable through an immutable CPU scale snapshot.
///
/// Not unique aggregate residency, allocator capacity, process RSS or VRAM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuScaleSnapshotBytes {
    /// Original packed root, including its original scales and compact index.
    /// Shared by snapshots and retained through the root's original handle/lease.
    pub shared_root_bytes: usize,
    /// One newly owned scale buffer; this is the incremental budget charge.
    pub owned_scale_bytes: usize,
    /// Checked sum reported by this snapshot's DeviceBuffer.len_bytes.
    /// Summing several snapshots double-counts their shared roots.
    pub reachable_bytes: usize,
}

pub(crate) struct ScaleSnapshot {
    // Free the scale payload and root before releasing this payload reservation.
    scales: Vec<u8>,
    root: Arc<dyn DeviceBuffer>,
    // Process-local comparison only; never dereferenced or persisted as identity.
    root_identity: usize,
    bytes: CpuScaleSnapshotBytes,
    _reservation: TensorStorageReservation,
}
impl fmt::Debug for ScaleSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScaleSnapshot")
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}
impl ScaleSnapshot {
    pub(super) fn base(&self) -> Result<&PackedAdditive, BackendError> {
        match CpuTensor::checked(&*self.root)? {
            CpuTensor::Additive(tensor)
                if std::ptr::from_ref(tensor).addr() == self.root_identity =>
            {
                Ok(tensor)
            }
            _ => Err(BackendError::InvalidInput(
                "CPU scale snapshot original backend root changed".into(),
            )),
        }
    }
    pub(super) fn view(&self) -> Result<PackedScaleView<'_>, BackendError> {
        Ok(self.base()?.with_scales(&self.scales))
    }
    pub(super) fn bytes(&self) -> CpuScaleSnapshotBytes {
        self.bytes
    }
}

impl CpuTensor {
    pub(crate) fn snapshot_scales(
        parent: &Arc<dyn DeviceBuffer>,
        scales: &[f32],
        budget: &TensorPayloadBudget,
    ) -> Result<(Arc<dyn DeviceBuffer>, CpuScaleSnapshotBytes), BackendError> {
        let selected = Self::checked(&**parent)?;
        if parent.len_bytes() != selected.len_bytes() {
            return Err(BackendError::InvalidInput(
                "CPU scale snapshot parent payload bytes differ from backend storage".into(),
            ));
        }
        let (root, base) = match selected {
            Self::Additive(base) => (parent, base),
            Self::ScaleSnapshot(snapshot) => (&snapshot.root, snapshot.base()?),
            Self::Dense { .. } => return Err(BackendError::UnsupportedTensor),
        };
        let owned_scale_bytes = base.validate_scale_replacement(scales)?;
        let shared_root_bytes = base.len_bytes();
        let reachable_bytes = shared_root_bytes
            .checked_add(owned_scale_bytes)
            .ok_or_else(|| {
                BackendError::InvalidInput("CPU scale snapshot reachable bytes overflow".into())
            })?;
        let reserved =
            budget.reserve_storage(u64::try_from(owned_scale_bytes).map_err(|_| {
                BackendError::InvalidInput("CPU scale snapshot budget size overflow".into())
            })?)?;
        let scales = base.encode_scales(scales, owned_scale_bytes)?;
        let bytes = CpuScaleSnapshotBytes {
            shared_root_bytes,
            owned_scale_bytes,
            reachable_bytes,
        };
        let result: Arc<dyn DeviceBuffer> = Arc::new(Self::ScaleSnapshot(ScaleSnapshot {
            scales,
            root: Arc::clone(root),
            root_identity: std::ptr::from_ref(base).addr(),
            bytes,
            _reservation: reserved,
        }));
        Ok((result, bytes))
    }
}
