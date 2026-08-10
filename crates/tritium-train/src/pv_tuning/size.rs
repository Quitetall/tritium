use super::checkpoint::encoded_checkpoint_len;
use super::{PvTuningError, PvTuningSession};

/// Exact logical payload sizes for one PV session.
///
/// Counts exclude allocator capacity and Rust container metadata. They expose
/// deployed host representation, dense optimizer moments, active campaign
/// accumulators, and canonical serialized checkpoint as separate quantities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PvTuningSizeLedger {
    host_representation_bytes: usize,
    host_optimizer_bytes: usize,
    host_campaign_bytes: usize,
    serialized_checkpoint_bytes: usize,
}

impl PvTuningSizeLedger {
    #[must_use]
    pub const fn host_representation_bytes(self) -> usize {
        self.host_representation_bytes
    }

    #[must_use]
    pub const fn host_optimizer_bytes(self) -> usize {
        self.host_optimizer_bytes
    }

    #[must_use]
    pub const fn host_campaign_bytes(self) -> usize {
        self.host_campaign_bytes
    }

    #[must_use]
    pub const fn serialized_checkpoint_bytes(self) -> usize {
        self.serialized_checkpoint_bytes
    }
}

impl PvTuningSession {
    /// Return overflow-checked payload accounting without serializing state.
    pub fn size_ledger(&self) -> Result<PvTuningSizeLedger, PvTuningError> {
        let code_bytes = self
            .weight
            .len()
            .checked_mul(self.weight.planes.len())
            .ok_or_else(size_error)?;
        let scale_bytes = self
            .weight
            .total_scale_count()
            .checked_mul(core::mem::size_of::<half::f16>())
            .ok_or_else(size_error)?;
        let host_representation_bytes =
            code_bytes.checked_add(scale_bytes).ok_or_else(size_error)?;
        let optimizer_values = self
            .code_state
            .m
            .len()
            .checked_add(self.code_state.v.len())
            .and_then(|count| count.checked_add(self.scale_state.m.len()))
            .and_then(|count| count.checked_add(self.scale_state.v.len()))
            .ok_or_else(size_error)?;
        let host_optimizer_bytes = optimizer_values
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(size_error)?;
        let host_campaign_bytes = self.blockwise.as_ref().map_or(Ok(0), |state| {
            state
                .scale_gradient
                .len()
                .checked_mul(core::mem::size_of::<f64>())
                .ok_or_else(size_error)
        })?;
        Ok(PvTuningSizeLedger {
            host_representation_bytes,
            host_optimizer_bytes,
            host_campaign_bytes,
            serialized_checkpoint_bytes: encoded_checkpoint_len(self)?,
        })
    }
}

fn size_error() -> PvTuningError {
    PvTuningError::checkpoint("PV state size overflows host range")
}
