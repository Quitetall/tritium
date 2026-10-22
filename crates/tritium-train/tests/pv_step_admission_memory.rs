//! Rejected public P/V steps must not copy deployed state or optimizer arrays.
//! Counts allocator requests, not physical RSS, caller buffers or page cache.
use half::f16;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_train::{
    AdamW, PvTernaryPlane, PvTernaryStructure, PvTernaryWeight, PvTuningConfig, PvTuningError,
    PvTuningSession,
};

struct TrackingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);

// SAFETY: requests, pointers and layouts are forwarded unchanged to System.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the allocator caller supplies a valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies the original pointer and allocation layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller supplies a live pointer/layout and valid new size.
        let output = unsafe { System.realloc(pointer, layout, size) };
        if !output.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(size, Ordering::Relaxed);
        }
        output
    }
}
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

fn measured<T>(call: impl FnOnce() -> T) -> (T, usize) {
    REQUESTED.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let result = call();
    TRACK.store(false, Ordering::Relaxed);
    (result, REQUESTED.load(Ordering::Relaxed))
}

fn session(elements: usize) -> PvTuningSession {
    let parent = PvTernaryWeight::new(
        elements / 256,
        256,
        32,
        PvTernaryStructure::Dense,
        vec![PvTernaryPlane::new(
            vec![1; elements],
            vec![f16::ONE; elements / 32],
        )],
    )
    .unwrap();
    let adam = AdamW {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.99,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    PvTuningSession::new(
        parent,
        PvTuningConfig::builder(adam, adam)
            .max_code_change_fraction(0.5)
            .build()
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn invalid_whole_steps_reject_before_model_sized_heap_work_and_preserve_state() {
    let (control, allocated) = measured(|| vec![7u8; 65536]);
    assert_eq!(std::hint::black_box(control)[0], 7);
    assert!(allocated >= 65536, "allocation tracking must be active");
    let mut observations = Vec::new();
    for elements in [1024, 65536] {
        let mut state = session(elements);
        let original = state.checkpoint_bytes().unwrap();
        let (control, allocated) = measured(|| state.begin_blockwise_step(0, 32));
        assert_eq!(
            control,
            Err(PvTuningError::Step(
                "expected optimizer step 1, got 0".into()
            ))
        );
        assert!(allocated < 4096);
        eprintln!("PV_PREFLIGHT_CONTROL elements={elements} bytes={allocated}");
        let mut gradient = vec![0.0; elements];
        let mut requested = Vec::new();
        for invalid in 0..4 {
            gradient.fill(0.0);
            if invalid == 2 {
                gradient[elements - 1] = f32::NAN;
            }
            if invalid == 3 {
                state.begin_blockwise_step(1, 32).unwrap();
            }
            let before = if invalid == 3 {
                state.blockwise_checkpoint_bytes().unwrap()
            } else {
                original.clone()
            };
            let (result, bytes) = measured(|| match invalid {
                0 => state.step(&gradient, 0),
                1 => state.step(&gradient[..1], 1),
                2 => state.step(&gradient, 1),
                _ => state.step(&[], u64::MAX),
            });
            let expected = match invalid {
                0 => "expected optimizer step 1, got 0",
                1 => "gradient length mismatch",
                2 => "gradient contains a non-finite value",
                _ => "cannot run a whole-gradient step while a blockwise step is active",
            };
            assert_eq!(result, Err(PvTuningError::Step(expected.into())));
            let after = if invalid == 3 {
                state.blockwise_checkpoint_bytes().unwrap()
            } else {
                state.checkpoint_bytes().unwrap()
            };
            assert_eq!(after, before);
            assert_eq!(state.completed_step(), 0);
            requested.push(bytes);
        }
        eprintln!("PV_REJECTED_STEP_REQUESTS elements={elements} bytes={requested:?}");
        observations.push(requested);
    }
    // The same shape-independent error messages may allocate small strings, but
    // rejecting an input must not scale with the deployed parent/optimizer state.
    for (small, large) in observations[0].iter().zip(&observations[1]) {
        assert!(
            *small < 4096 && *large <= small + 256,
            "rejected-step heap requests grew with parent state: {observations:?}"
        );
    }
}
