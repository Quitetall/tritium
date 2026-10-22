//! Allocator requests for the CPU reference; not physical memory qualification.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_core::Trit;
use tritium_format::AdditiveTensor;
use tritium_schema::*;
use tritium_train::pv_tuning::NativeScaleRefinementSession;

struct TrackingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
// SAFETY: each allocation operation is forwarded unchanged to System.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc supplies a valid layout.
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        result
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies the original pointer and layout.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller supplies a live pointer/layout and valid new size.
        let result = unsafe { System.realloc(ptr, layout, size) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(size, Ordering::Relaxed);
        }
        result
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
fn session(elements: usize) -> NativeScaleRefinementSession {
    let layout = AdditiveLayout {
        rows: 1,
        cols: elements as u64,
        tile: 256,
        group: 32,
        max_planes: 1,
        allocation: PlaneAllocation::Uniform,
        codec: PlaneCodec::D2,
        law: ADMITTED_LAWS[3].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    let weight = AdditiveTensor::new(layout, 1, vec![Trit::POS; elements], vec![0.5]).unwrap();
    let plan = NativeRefinementPlan {
        parent: NativeRefinementParent {
            package: PackageId::from_bytes([1; 32]),
            level: "parent".into(),
            model: ModelId::from_bytes([2; 32]),
            source: OriginalSourceId::from_bytes([3; 32]),
            evidence: EvidenceReference {
                run: "synthetic".into(),
                span: "parent".into(),
                root: [4; 32],
            },
        },
        child_level: "child".into(),
        owners: vec!["weight".into()],
        recipe: NativeRecoveryRecipe::ScaleOnly {
            optimizer: NativeAdamWRecipe {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-8,
                weight_decay: 0.0,
            },
        },
        objective: [5; 32],
        teacher_source: None,
        training: vec![NativeRefinementWindow {
            content: [6; 32],
            weight: 1.0,
        }],
        validation: vec![NativeRefinementWindow {
            content: [7; 32],
            weight: 1.0,
        }],
        limits: NativeRefinementLimits {
            max_state_bytes: 2_000_000,
            max_workspace_bytes: 16,
            max_checkpoint_bytes: 137,
            max_package_bytes: 2_000_000,
        },
    };
    NativeScaleRefinementSession::new(plan, vec![weight]).unwrap()
}
#[test]
fn scale_updates_do_not_copy_codes_or_allocate_dense_or_code_optimizer_state() {
    let (control, bytes) = measured(|| vec![7u8; 65536]);
    assert_eq!(std::hint::black_box(control)[0], 7);
    assert!(bytes >= 65536);
    let mut observations = Vec::new();
    for elements in [1024, 1_048_576] {
        let mut state = session(elements);
        assert_eq!(
            state.size_ledger().state_payload_bytes,
            elements as u64 + 12
        );
        assert_eq!(state.size_ledger().update_workspace_bytes, 16);
        let address = state.weights().next().unwrap().trits().as_ptr();
        let (result, bytes) = measured(|| state.step(&[&[0.25]]));
        assert_eq!(result.unwrap(), 1);
        assert!(
            bytes < 1024,
            "update requested {bytes} bytes for {elements} codes"
        );
        assert_eq!(state.weights().next().unwrap().trits().as_ptr(), address);
        eprintln!("NATIVE_SCALE_UPDATE elements={elements} allocated_request_bytes={bytes}");
        observations.push(bytes);
    }
    assert_eq!(
        observations[0], observations[1],
        "one-scale update allocation must not scale with coefficient count"
    );
}
