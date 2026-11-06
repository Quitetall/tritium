//! Optional backend-owned, transactional Gated DeltaNet execution.

use crate::{BackendError, TernaryBackend};

/// Recurrent layout `[value_heads, key_head_dim, value_head_dim]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaNetGeometry {
    /// Number of value heads.
    pub value_heads: usize,
    /// Per-head key/query width.
    pub key_head_dim: usize,
    /// Per-head value width.
    pub value_head_dim: usize,
    /// Number of value heads sharing one key/query head.
    pub group_size: usize,
}

impl DeltaNetGeometry {
    /// Validate nonzero dimensions, whole key-head groups and state byte size.
    ///
    /// Returns the element count of one state buffer, without allocating.
    /// Device-specific kernel limits are separate unsupported-capability checks.
    ///
    /// # Errors
    /// Returns invalid input for malformed or overflowing geometry.
    pub fn elements(self) -> Result<usize, BackendError> {
        if self.value_heads == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
            || self.group_size == 0
            || !self.value_heads.is_multiple_of(self.group_size)
        {
            return Err(BackendError::InvalidInput(
                "DeltaNet geometry requires nonzero dimensions and whole key-head groups".into(),
            ));
        }
        self.value_heads
            .checked_mul(self.key_head_dim)
            .and_then(|n| n.checked_mul(self.value_head_dim))
            .filter(|&n| n.checked_mul(core::mem::size_of::<f32>()).is_some())
            .ok_or_else(|| BackendError::InvalidInput("DeltaNet state byte size overflows".into()))
    }
}

/// One token's pre-normalized recurrent operands.
///
/// Keys/queries already carry L2 normalization (and queries their scale);
/// beta/decay already carry sigmoid/exponential. The recurrence performs only
/// multiply/add; no device-specific transcendental approximation is requested.
#[derive(Debug)]
pub struct DeltaNetStep<'a> {
    /// Key-head-major keys, `[value_heads / group_size, key_head_dim]`.
    pub key: &'a [f32],
    /// Same layout as key, including query scaling.
    pub query: &'a [f32],
    /// Value-head-major values, `[value_heads, value_head_dim]`.
    pub value: &'a [f32],
    /// Sigmoid update gate, `[value_heads]`.
    pub beta: &'a [f32],
    /// Exponential decay, `[value_heads]`.
    pub decay: &'a [f32],
    /// Output, `[value_heads, value_head_dim]`.
    pub output: &'a mut [f32],
}

/// Optional recurrence executor with owned backend resources and double buffers.
///
/// Stage copies committed state; step/write mutate staging only. Failure may
/// dirty staging, never committed state. A new stage overwrites an abandoned
/// stage, so abort requires no driver work. Callers commit only after all layers
/// succeed and preflight their pending lengths. Ordinary step must not read the
/// whole state back to the host. No full-model residency or performance claim
/// follows from implementing this interface.
pub trait ResidentDeltaNet: core::fmt::Debug + Send {
    /// Exact layout accepted by this executor.
    fn geometry(&self) -> DeltaNetGeometry;

    /// Reject a backend other than this executor's owning instance/stream.
    ///
    /// Same device label alone is not sufficient identity. The executor owns
    /// its resources even if the backend is subsequently dropped.
    ///
    /// # Errors
    /// Returns invalid input for an owner mismatch, before state is mutated.
    fn validate_owner(&self, backend: &dyn TernaryBackend) -> Result<(), BackendError>;

    /// Begin a transaction from committed state, replacing any abandoned stage.
    ///
    /// # Errors
    /// Returns transfer/driver failures without changing committed state.
    fn stage(&mut self) -> Result<(), BackendError>;

    /// Execute one token on staging, validating every operand length first.
    ///
    /// # Errors
    /// Returns shape or execution failures; committed state remains unchanged.
    fn step(&mut self, operands: DeltaNetStep<'_>) -> Result<(), BackendError>;

    /// Explicit diagnostic readback of staging, never an ordinary-step duty.
    ///
    /// # Errors
    /// Returns shape/transfer failures; committed state remains unchanged.
    fn read_staged(&self, output: &mut [f32]) -> Result<(), BackendError>;

    /// Initialize/replace staging from an explicitly supplied host snapshot.
    ///
    /// # Errors
    /// Returns shape/transfer failures; committed state remains unchanged.
    fn write_staged(&mut self, values: &[f32]) -> Result<(), BackendError>;

    /// Publish staging by swapping buffers; no allocation or fallible device work.
    fn commit(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_rejects_invalid_and_overflowing_state_before_allocation() {
        let valid = DeltaNetGeometry {
            value_heads: 6,
            key_head_dim: 8,
            value_head_dim: 8,
            group_size: 3,
        };
        assert_eq!(valid.elements().unwrap(), 384);
        for invalid in [
            DeltaNetGeometry {
                value_heads: 0,
                ..valid
            },
            DeltaNetGeometry {
                key_head_dim: 0,
                ..valid
            },
            DeltaNetGeometry {
                value_head_dim: 0,
                ..valid
            },
            DeltaNetGeometry {
                group_size: 0,
                ..valid
            },
            DeltaNetGeometry {
                group_size: 4,
                ..valid
            },
            DeltaNetGeometry {
                key_head_dim: usize::MAX,
                ..valid
            },
            DeltaNetGeometry {
                value_heads: 1,
                key_head_dim: 1,
                value_head_dim: usize::MAX / 2,
                group_size: 1,
            },
        ] {
            assert!(matches!(
                invalid.elements(),
                Err(BackendError::InvalidInput(_))
            ));
        }
    }
}
