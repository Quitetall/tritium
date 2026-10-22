//! Heap requests during the reference call only, not a physical-memory bound.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_core::{AdditiveView, Trit, UniformPlan, reference_ternary_matmul_input_vjp};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};

struct Allocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);

// SAFETY: all requests/pointers/layouts are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: caller provides a valid allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: caller provides the original live pointer and layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: caller provides the live pointer, original layout and new size.
        let output = unsafe { System.realloc(pointer, layout, size) };
        if !output.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(size, Ordering::Relaxed);
        }
        output
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn measured<T>(call: impl FnOnce() -> T) -> (T, usize) {
    REQUESTED.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let result = call();
    TRACK.store(false, Ordering::Relaxed);
    (result, REQUESTED.load(Ordering::Relaxed))
}

#[test]
fn input_adjoint_uses_only_caller_buffers_at_small_and_large_shapes() {
    let (control, bytes) = measured(|| vec![7u8; 8192]);
    assert_eq!(std::hint::black_box(control)[0], 7);
    assert!(bytes >= 8192, "allocation tracking must be active");
    for rows in [1, 256] {
        let layout = AdditiveLayout {
            rows,
            cols: 256,
            tile: 256,
            group: 32,
            max_planes: 3,
            allocation: PlaneAllocation::Uniform,
            codec: PlaneCodec::D2,
            law: ADMITTED_LAWS[0].law,
            basis: Basis::SignedRht {
                block: 16,
                seed: 19,
                domain: 7,
            },
            transport: Transport::Raw,
        };
        let plan = UniformPlan::new(layout, 3).unwrap();
        let trits = vec![Trit::POS; plan.trit_count()];
        let scales = vec![0.5; plan.scale_count()];
        let view = AdditiveView::new(layout, 3, &trits, &scales)
            .unwrap()
            .bind_matmul();
        let gradients = vec![0.25; 2 * rows as usize];
        let mut scratch = vec![0.0; 256];
        let mut output = vec![0.0; 512];
        let (result, bytes) = measured(|| {
            reference_ternary_matmul_input_vjp(&gradients, &view, 2, &mut scratch, &mut output)
        });
        assert_eq!(result, Ok(()));
        assert_eq!(bytes, 0);
        assert!(output.iter().all(|v| v.is_finite()));
        eprintln!("INPUT_ADJOINT_HEAP_REQUESTS rows={rows} cols=256 batch=2 bytes={bytes}");
    }
}
