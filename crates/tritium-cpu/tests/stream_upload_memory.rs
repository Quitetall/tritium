//! Isolated cumulative heap requests, not RSS/page-cache/whole-model qualification.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tritium_core::{PerTilePlan, Trit, UniformPlan};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};
use tritium_spec::{AdditiveMetadata, TensorMetadata, TensorStreamEncoding, TernaryBackend};
#[path = "common/stream_source.rs"]
mod stream_source;
struct TrackingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
// SAFETY: all requests, pointers and layouts are forwarded unchanged to System.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: caller supplied a valid allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: caller supplied the original allocation pointer and layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: caller supplied original pointer/layout and a valid new size.
        let output = unsafe { System.realloc(pointer, layout, size) };
        if !output.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(size, Ordering::Relaxed);
        }
        output
    }
}
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn canonical_stream_allocates_one_native_payload_plus_bounded_validation_not_masters() {
    for compact in [false, true] {
        for cols in [512usize, 65536] {
            let layout = AdditiveLayout {
                rows: 8,
                cols: cols as u64,
                tile: 256,
                group: 128,
                max_planes: 3,
                allocation: if compact {
                    PlaneAllocation::PerTile
                } else {
                    PlaneAllocation::Uniform
                },
                codec: PlaneCodec::B3,
                law: ADMITTED_LAWS[0].law,
                basis: Basis::Identity,
                transport: Transport::Raw,
            };
            let tensor = if compact {
                let counts = (0..8 * cols.div_ceil(256))
                    .map(|i| (i % 4) as u8)
                    .collect::<Vec<_>>();
                let plan = PerTilePlan::new(layout, &counts).unwrap();
                let trits = (0..plan.trit_count())
                    .map(|i| Trit::from_i8((i % 3) as i8 - 1).unwrap())
                    .collect();
                let scales = vec![0.5; plan.scale_count()];
                AdditiveTensor::new_per_tile(layout, counts, trits, scales).unwrap()
            } else {
                let plan = UniformPlan::new(layout, 3).unwrap();
                AdditiveTensor::new(
                    layout,
                    3,
                    (0..plan.trit_count())
                        .map(|i| Trit::from_i8((i % 3) as i8 - 1).unwrap())
                        .collect(),
                    vec![0.5; plan.scale_count()],
                )
                .unwrap()
            };
            let bytes = tensor.encode_payload().unwrap();
            let source = stream_source::Source::new(
                TensorMetadata::Additive(AdditiveMetadata::from_view(tensor.view())),
                TensorStreamEncoding::CanonicalRawAdditive,
                &bytes,
            );
            REQUESTED.store(0, Ordering::Relaxed);
            TRACK.store(true, Ordering::Relaxed);
            let buffer = CpuBackend::new().upload_tensor_stream(&source).unwrap();
            TRACK.store(false, Ordering::Relaxed);
            let requested = REQUESTED.load(Ordering::Relaxed);
            let retained = buffer.len_bytes();
            eprintln!(
                "CPU_STREAM_HEAP compact={compact} cols={cols} encoded={} retained={retained} cumulative_heap_requests={requested}",
                bytes.len()
            );
            assert!(requested >= retained);
            assert!(
                requested < retained + 4096,
                "unexpected temporary master allocation: requested={requested} retained={retained}"
            );
        }
    }
}
