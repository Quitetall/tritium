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
    // A child keeps its parent reservation until the final child owner drops.
    _parent: Option<Reservation>,
}

/// Shared logical cap for admitted tensor payloads and explicit storage guards.
///
/// Clones share one ledger. A caller may use one ledger for a model, a device or
/// several models; independently created ledgers do not coordinate. Each upload
/// reserves before allocation and holds its reservation until the returned
/// tensor's final owner drops it. Tied consumers sharing an `Arc` count once.
///
/// This is not a physical-memory measurement or complete VRAM admission. It
/// Upload admission alone excludes host expansion, temporary/scratch buffers,
/// driver/allocator overhead, KV and unguarded uploads. Explicit storage facades
/// may separately charge applicable domains to the same ledger; nothing is
/// implicitly admitted. Production loaders must reserve their covered resources.
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
                _parent: None,
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

    /// Atomically reserve a fixed partition before a model starts allocating.
    /// Child uploads charge this partition, not the parent twice. The entire
    /// partition remains charged until its last budget/handle/reservation drops.
    /// This is logical admission, not a physical-memory measurement.
    pub fn partition(&self, bytes: u64) -> Result<Self, BackendError> {
        let parent = self.reserve(bytes)?;
        Ok(Self {
            ledger: Arc::new(Ledger {
                limit: bytes,
                used: AtomicU64::new(0),
                _parent: Some(parent),
            }),
        })
    }

    /// Reserve caller-owned logical storage before allocation. The non-cloneable
    /// guard must outlive the storage; dropping it releases the reservation.
    pub fn reserve_storage(&self, bytes: u64) -> Result<TensorStorageReservation, BackendError> {
        self.reserve(bytes)
            .map(|reservation| TensorStorageReservation {
                _reservation: reservation,
            })
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

#[derive(Debug)]
pub(super) struct Reservation {
    ledger: Arc<Ledger>,
    bytes: u64,
}

/// Opaque lifetime guard for caller-owned logical storage, not physical memory.
/// Declare it after the owned storage so storage drops before admission releases.
#[derive(Debug)]
pub struct TensorStorageReservation {
    _reservation: Reservation,
}

impl TensorStorageReservation {
    /// Logical bytes held by this guard, not allocated or measured physical bytes.
    #[must_use]
    pub fn reserved_bytes(&self) -> u64 {
        self._reservation.bytes
    }

    /// Atomically change this guard's logical reservation, retaining its ledger
    /// and parent lifetime. Reserve growth BEFORE allocating the added storage;
    /// release/shrink only AFTER corresponding storage has been dropped. This
    /// method does not allocate/free user storage or establish physical fit.
    /// # Errors
    /// Failed growth leaves both this guard and the ledger unchanged.
    pub fn resize(&mut self, bytes: u64) -> Result<(), BackendError> {
        let old = self._reservation.bytes;
        if bytes > old {
            let budget = TensorPayloadBudget {
                ledger: Arc::clone(&self._reservation.ledger),
            };
            let mut additional = budget.reserve(bytes - old)?;
            // Transfer the charge, not the Arc: dropping the zero-byte guard
            // releases its ledger reference without releasing the transferred bytes.
            additional.bytes = 0;
            self._reservation.bytes = bytes;
        } else if bytes < old {
            self._reservation
                .ledger
                .used
                .fetch_sub(old - bytes, Ordering::AcqRel);
            self._reservation.bytes = bytes;
        }
        Ok(())
    }
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
    fn storage_resize_is_atomic_and_retains_parent_until_final_guard_drop() {
        let parent = TensorPayloadBudget::new(100);
        let child = parent.partition(80).unwrap();
        let mut storage = child.reserve_storage(40).unwrap();
        storage.resize(70).unwrap();
        assert_eq!(
            (
                storage.reserved_bytes(),
                child.used_bytes(),
                parent.used_bytes()
            ),
            (70, 70, 80)
        );
        assert!(storage.resize(81).is_err());
        assert_eq!((storage.reserved_bytes(), child.used_bytes()), (70, 70));
        storage.resize(20).unwrap();
        assert_eq!(child.used_bytes(), 20);
        storage.resize(0).unwrap();
        assert_eq!(child.used_bytes(), 0);
        drop(child);
        assert_eq!(
            parent.used_bytes(),
            80,
            "zero-byte storage guard still owns its parent lifetime"
        );
        storage.resize(80).unwrap();
        assert!(storage.resize(u64::MAX).is_err());
        assert_eq!(storage.reserved_bytes(), 80);
        drop(storage);
        assert_eq!(parent.used_bytes(), 0);
    }

    #[test]
    fn concurrent_storage_growth_cannot_overcommit() {
        let budget = TensorPayloadBudget::new(100);
        let start = Arc::new(std::sync::Barrier::new(3));
        let finish = Arc::new(std::sync::Barrier::new(3));
        std::thread::scope(|scope| {
            let jobs = (0..2)
                .map(|_| {
                    let mut storage = budget.reserve_storage(0).unwrap();
                    let start = Arc::clone(&start);
                    let finish = Arc::clone(&finish);
                    scope.spawn(move || {
                        start.wait();
                        let result = storage.resize(60);
                        finish.wait();
                        result.is_ok()
                    })
                })
                .collect::<Vec<_>>();
            start.wait();
            finish.wait();
            assert_eq!(
                jobs.into_iter()
                    .map(|j| j.join().unwrap())
                    .filter(|ok| *ok)
                    .count(),
                1
            );
        });
        assert_eq!(budget.used_bytes(), 0);
    }

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

    #[test]
    fn partition_is_atomic_and_follows_final_child_owner() {
        let parent = TensorPayloadBudget::new(100);
        let child = parent.partition(80).unwrap();
        assert_eq!(parent.used_bytes(), 80);
        assert!(parent.partition(21).is_err());
        let clone = child.clone();
        let storage = child.reserve_storage(60).unwrap();
        assert_eq!(child.used_bytes(), 60);
        assert!(child.reserve_storage(21).is_err());
        drop(child);
        drop(clone);
        assert_eq!(parent.used_bytes(), 80);
        drop(storage);
        assert_eq!(parent.used_bytes(), 0);
    }

    #[test]
    fn concurrent_partitions_cannot_overcommit_the_parent() {
        let parent = TensorPayloadBudget::new(100);
        let start = Arc::new(std::sync::Barrier::new(9));
        let finish = Arc::new(std::sync::Barrier::new(9));
        std::thread::scope(|scope| {
            let jobs = (0..8)
                .map(|_| {
                    let parent = parent.clone();
                    let start = start.clone();
                    let finish = finish.clone();
                    scope.spawn(move || {
                        start.wait();
                        let child = parent.partition(60);
                        finish.wait();
                        child.is_ok()
                    })
                })
                .collect::<Vec<_>>();
            start.wait();
            finish.wait();
            assert_eq!(
                jobs.into_iter()
                    .map(|job| job.join().unwrap())
                    .filter(|ok| *ok)
                    .count(),
                1
            );
        });
        assert_eq!(parent.used_bytes(), 0);
    }
}
