//! Allocator-request probes, not full-process or physical-memory qualification.
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
use tritium_spec::{DeviceBuffer, TensorView, TernaryBackend};
struct Allocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
// SAFETY: all requests/pointers/layouts are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: allocator caller supplies a valid layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: caller supplies the original pointer/layout.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: caller supplies a live pointer/layout and valid new size.
        let result = unsafe { System.realloc(ptr, layout, size) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(size, Ordering::Relaxed);
        }
        result
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn measured<T>(call: impl FnOnce() -> T) -> (T, usize) {
    REQUESTED.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let out = call();
    TRACK.store(false, Ordering::Relaxed);
    (out, REQUESTED.load(Ordering::Relaxed))
}
#[test]
fn snapshots_allocate_for_scales_not_coefficient_or_compact_index_size() {
    let (control, bytes) = measured(|| vec![7u8; 65536]);
    assert_eq!(std::hint::black_box(control)[0], 7);
    assert!(bytes >= 65536);
    let mut observations = Vec::new();
    for elements in [1024, 1_048_576] {
        for compact in [false, true] {
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
                codec: PlaneCodec::D2,
                law: ADMITTED_LAWS[3].law,
                basis: Basis::Identity,
                transport: Transport::Raw,
            };
            let source = if compact {
                AdditiveTensor::new_per_tile(
                    layout,
                    vec![1; layout.rows as usize],
                    vec![Trit::POS; elements as usize],
                    vec![0.5],
                )
                .unwrap()
            } else {
                AdditiveTensor::new(layout, 1, vec![Trit::POS; elements as usize], vec![0.5])
                    .unwrap()
            };
            let root: Arc<dyn DeviceBuffer> = Arc::from(
                CpuBackend::new()
                    .upload_tensor(TensorView::Additive(source.view()))
                    .unwrap(),
            );
            drop(source);
            let budget = TensorPayloadBudget::new(4);
            let (result, allocated) =
                measured(|| CpuBackend::new().snapshot_additive_scales(&root, &[0.75], &budget));
            let (snapshot, bytes) = result.unwrap();
            assert_eq!(bytes.owned_scale_bytes, 4);
            assert_eq!(budget.used_bytes(), 4);
            assert!(
                allocated < 2048,
                "snapshot allocated {allocated} bytes for {elements} coefficients"
            );
            eprintln!(
                "SCALE_SNAPSHOT_ALLOC elements={elements} compact={compact} bytes={allocated}"
            );
            observations.push(allocated);
            drop(snapshot);
            assert_eq!(budget.used_bytes(), 0);
            assert_eq!(Arc::strong_count(&root), 1);
            let (result, allocated) = measured(|| {
                CpuBackend::new().snapshot_additive_scales(&root, &[f32::NAN], &budget)
            });
            assert!(result.is_err());
            assert!(allocated < 2048);
            assert_eq!(budget.used_bytes(), 0);
            assert_eq!(Arc::strong_count(&root), 1);
        }
    }
    assert!(observations.iter().all(|bytes| *bytes == observations[0]));
}
