//! Process-local owned-payload admission and lifetime-bound reservations.

use core::any::Any;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tritium_spec::{BackendError, DeviceBuffer};

#[derive(Debug)]
struct Ledger {
    limit: u64,
    used: AtomicU64,
}

/// Shared cap on payload bytes owned by admitted tensor handles.
///
/// Clones share one ledger. A caller may use one ledger for a model, a device or
/// several models; independently created ledgers do not coordinate. Each upload
/// reserves before allocation and holds its reservation until the returned
/// tensor's final owner drops it. Tied consumers sharing an `Arc` count once.
///
/// This is not a physical-memory measurement or complete VRAM admission. It
/// excludes host expansion, temporary/scratch buffers, driver/allocator overhead,
/// KV and unguarded uploads. Production loaders must use the admitted facade and
/// reserve those other resources separately.
#[derive(Clone, Debug)]
pub struct TensorPayloadBudget {
    ledger: Arc<Ledger>,
}

impl TensorPayloadBudget {
    /// Create an empty ledger with a fixed byte limit; zero is legal.
    #[must_use]
    pub fn new(limit: u64) -> Self {
        Self {
            ledger: Arc::new(Ledger {
                limit,
                used: AtomicU64::new(0),
            }),
        }
    }

    /// Fixed limit, shared by every clone.
    #[must_use]
    pub fn limit_bytes(&self) -> u64 {
        self.ledger.limit
    }

    /// Currently reserved upload/owned-handle bytes, including in-flight uploads.
    /// A snapshot can change immediately when other threads upload/drop tensors.
    #[must_use]
    pub fn used_bytes(&self) -> u64 {
        self.ledger.used.load(Ordering::Acquire)
    }

    pub(super) fn reserve(&self, bytes: u64) -> Result<Reservation, BackendError> {
        let mut used = self.used_bytes();
        loop {
            let requested = used.checked_add(bytes).ok_or_else(|| {
                BackendError::InvalidInput("aggregate tensor payload size overflows u64".into())
            })?;
            if requested > self.ledger.limit {
                return Err(BackendError::TensorPayloadBudgetExceeded {
                    requested,
                    limit: self.ledger.limit,
                });
            }
            match self.ledger.used.compare_exchange_weak(
                used,
                requested,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(Reservation {
                        ledger: Arc::clone(&self.ledger),
                        bytes,
                    });
                }
                Err(current) => used = current,
            }
        }
    }
}

pub(super) struct Reservation {
    ledger: Arc<Ledger>,
    bytes: u64,
}

impl Reservation {
    pub(super) fn attach(self, buffer: Box<dyn DeviceBuffer>) -> Box<dyn DeviceBuffer> {
        Box::new(AdmittedBuffer {
            buffer,
            _reservation: self,
        })
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

struct AdmittedBuffer {
    // Field order matters: free the owned buffer before releasing its admission.
    buffer: Box<dyn DeviceBuffer>,
    _reservation: Reservation,
}

impl DeviceBuffer for AdmittedBuffer {
    fn len_bytes(&self) -> usize {
        self.buffer.len_bytes()
    }
    fn as_any(&self) -> &dyn Any {
        // Preserve concrete backend ownership/downcast checks. The returned
        // owning handle keeps its reservation without adding an unwrap method.
        self.buffer.as_any()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_arithmetic_never_wraps_and_zero_reservations_release() {
        let budget = TensorPayloadBudget::new(u64::MAX);
        let all = budget.reserve(u64::MAX).unwrap();
        assert!(matches!(
            budget.reserve(1),
            Err(BackendError::InvalidInput(_))
        ));
        let zero = budget.reserve(0).unwrap();
        drop(zero);
        assert_eq!(budget.used_bytes(), u64::MAX);
        drop(all);
        assert_eq!(budget.used_bytes(), 0);
        let empty = TensorPayloadBudget::new(0);
        assert!(matches!(
            empty.reserve(1),
            Err(BackendError::TensorPayloadBudgetExceeded {
                requested: 1,
                limit: 0
            })
        ));
        drop(empty.reserve(0).unwrap());
        assert_eq!(empty.used_bytes(), 0);
    }
}
