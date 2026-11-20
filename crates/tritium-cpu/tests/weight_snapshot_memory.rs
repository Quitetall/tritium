//! Allocator-request and injected fallible-allocation probes, not physical RSS.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tritium_core::Trit;
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_runtime::TensorPayloadBudget;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};
use tritium_spec::{BackendError, DeviceBuffer, TensorView, TernaryBackend};

struct Allocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static BYTES: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static FAIL_CALL: AtomicUsize = AtomicUsize::new(0);
fn reject(size: usize) -> bool {
    if !TRACK.load(Ordering::Relaxed) {
        return false;
    }
    BYTES.fetch_add(size, Ordering::Relaxed);
    let call = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    call == FAIL_CALL.load(Ordering::Relaxed)
}
// SAFETY: successful allocations/pointers/layouts are forwarded to System;
// injected failures return null, as permitted by GlobalAlloc.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if reject(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY: caller provides a valid allocation layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: caller provides the live pointer and original layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if reject(size) {
            return std::ptr::null_mut();
        }
        // SAFETY: caller provides the live pointer, original layout and new size.
        unsafe { System.realloc(pointer, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn measured<T>(fail_call: usize, call: impl FnOnce() -> T) -> (T, usize) {
    BYTES.store(0, Ordering::Relaxed);
    CALLS.store(0, Ordering::Relaxed);
    FAIL_CALL.store(fail_call, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let result = call();
    TRACK.store(false, Ordering::Relaxed);
    FAIL_CALL.store(0, Ordering::Relaxed);
    (result, BYTES.load(Ordering::Relaxed))
}

#[test]
fn only_new_packed_payload_is_allocated_and_every_fallible_upload_fault_releases_admission() {
    let (control, bytes) = measured(0, || vec![7u8; 65536]);
    assert_eq!(std::hint::black_box(control)[0], 7);
    assert!(bytes >= 65536);
    let cpu = CpuBackend::new();
    for elements in [1024, 1_048_576] {
        for compact in [false, true] {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                let layout = AdditiveLayout {
                    rows: if compact { elements / 256 } else { 1 },
                    cols: if compact { 256 } else { elements },
                    tile: 256,
                    group: 32,
                    max_planes: 1,
                    allocation: if compact {
                        PlaneAllocation::PerTile
                    } else {
                        PlaneAllocation::Uniform
                    },
                    codec,
                    law: ADMITTED_LAWS[3].law,
                    basis: Basis::Identity,
                    transport: Transport::Raw,
                };
                let trits = (0..elements)
                    .map(|i| [Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS][i as usize % 4])
                    .collect();
                let mut candidate = if compact {
                    AdditiveTensor::new_per_tile(
                        layout,
                        vec![1; layout.rows as usize],
                        trits,
                        vec![0.5],
                    )
                    .unwrap()
                } else {
                    AdditiveTensor::new(layout, 1, trits, vec![0.5]).unwrap()
                };
                let root: Arc<dyn DeviceBuffer> = Arc::from(
                    cpu.upload_tensor(TensorView::Additive(candidate.view()))
                        .unwrap(),
                );
                let next_trits = candidate
                    .trits()
                    .iter()
                    .map(|trit| Trit::from_i8(-trit.get()).unwrap())
                    .collect();
                candidate.replace_trits(next_trits).unwrap();
                candidate.replace_scales(vec![0.75]).unwrap();
                let payload = root.len_bytes();
                let budget = TensorPayloadBudget::new(payload as u64);
                let short = TensorPayloadBudget::new(payload as u64 - 1);
                let (result, allocated) = measured(0, || {
                    cpu.snapshot_additive_weights(&root, candidate.view(), &short)
                });
                assert!(matches!(
                    result,
                    Err(BackendError::TensorPayloadBudgetExceeded { .. })
                ));
                assert_eq!(
                    allocated, 0,
                    "budget refusal must precede payload allocation"
                );
                assert_eq!(short.used_bytes(), 0);
                let (result, allocated) = measured(0, || {
                    cpu.snapshot_additive_weights(&root, candidate.view(), &budget)
                });
                let snapshot = result.unwrap();
                assert_eq!(snapshot.len_bytes(), payload);
                assert_eq!(budget.used_bytes(), payload as u64);
                assert!(
                    allocated >= payload && allocated < payload + 2048,
                    "allocated={allocated} payload={payload} elements={elements} compact={compact} codec={codec:?}"
                );
                assert!(
                    cpu.additive_equals_view(&*snapshot, candidate.view())
                        .unwrap()
                );
                assert_eq!(Arc::strong_count(&root), 1);
                drop(snapshot);
                assert_eq!(budget.used_bytes(), 0);
                // Uniform: packed Vec. Compact: counts, row spans, plane starts,
                // packed Vec. Never inject into infallible handle metadata allocs.
                for fail_call in 1..=if compact { 4 } else { 1 } {
                    let (result, _) = measured(fail_call, || {
                        cpu.snapshot_additive_weights(&root, candidate.view(), &budget)
                    });
                    assert!(
                        matches!(result, Err(BackendError::OutOfMemory { .. })),
                        "fallible allocation={fail_call}"
                    );
                    assert_eq!(budget.used_bytes(), 0);
                    assert_eq!(Arc::strong_count(&root), 1);
                }
                let (snapshot, _) = measured(0, || {
                    cpu.snapshot_additive_weights(&root, candidate.view(), &budget)
                });
                let snapshot = snapshot.unwrap();
                drop(candidate);
                drop(root);
                assert_eq!(snapshot.len_bytes(), payload);
                drop(snapshot);
                assert_eq!(budget.used_bytes(), 0);
                eprintln!(
                    "FULL_SNAPSHOT_ALLOCATION elements={elements} compact={compact} codec={codec:?} packed={payload} allocated={allocated}"
                );
            }
        }
    }
}
