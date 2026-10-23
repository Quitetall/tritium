//! Allocation requests, not physical RSS or model qualification.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_core::Trit;
use tritium_format::AdditiveTensor;
use tritium_schema::*;
use tritium_train::pv_tuning::NativePvRefinementSession as Session;
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
fn plan(owners: usize) -> NativeRefinementPlan {
    NativeRefinementPlan {
        parent: NativeRefinementParent {
            package: PackageId::from_bytes([1; 32]),
            level: "explicit-parent".into(),
            model: ModelId::from_bytes([2; 32]),
            source: OriginalSourceId::from_bytes([3; 32]),
            evidence: EvidenceReference {
                run: "synthetic-parent".into(),
                span: "import".into(),
                root: [4; 32],
            },
        },
        child_level: "one-child".into(),
        owners: (0..owners).map(|i| format!("owner-{i:04}")).collect(),
        recipe: NativeRecoveryRecipe::AlternatingPv {
            continuous_optimizer: NativeAdamWRecipe {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-8,
                weight_decay: 0.01,
            },
            code_optimizer: NativeAdamWRecipe {
                lr: 0.6,
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-8,
                weight_decay: 0.01,
            },
            max_code_change_fraction: 0.5,
            max_relative_code_change: Some(0.9),
        },
        objective: [5; 32],
        teacher_source: None,
        training: (0..4)
            .map(|i| NativeRefinementWindow {
                content: [i + 10; 32],
                weight: if i == 0 { 2.0 } else { 1.0 },
            })
            .collect(),
        validation: vec![NativeRefinementWindow {
            content: [20; 32],
            weight: 1.0,
        }],
        limits: NativeRefinementLimits {
            max_state_bytes: 1_000_000,
            max_workspace_bytes: 1_000_000,
            max_checkpoint_bytes: 1_000_000,
            max_package_bytes: 1_000_000,
        },
    }
}

fn parent(rows: u64, pruned: bool) -> AdditiveTensor {
    let layout = AdditiveLayout {
        rows,
        cols: 256,
        tile: 256,
        group: 32,
        max_planes: 1,
        allocation: if pruned {
            PlaneAllocation::PerTile
        } else {
            PlaneAllocation::Uniform
        },
        codec: PlaneCodec::D2,
        law: ADMITTED_LAWS[3].law,
        basis: Basis::Identity,
        transport: Transport::Raw,
    };
    if pruned {
        AdditiveTensor::new_per_tile(layout, vec![0; rows as usize], vec![], vec![]).unwrap()
    } else {
        AdditiveTensor::new(layout, 1, vec![Trit::POS; rows as usize * 256], vec![0.5]).unwrap()
    }
}
#[test]
fn pruned_coordinates_have_no_dense_moments_and_caps_reject_before_payload_allocations() {
    let (control, bytes) = measured(|| vec![1u8; 65536]);
    assert!(bytes >= 65536);
    drop(control);
    for rows in [1, 4096] {
        let weight = parent(rows, true);
        let expected = weight.view().auxiliary_buffer_len_bytes().unwrap() as u64;
        let mut p = plan(1);
        p.limits.max_state_bytes = 64_000_000;
        p.limits.max_workspace_bytes = 64_000_000;
        let (result, bytes) = measured(|| Session::new(p, vec![weight]));
        let mut s = result.unwrap();
        assert!(bytes < 65536, "pruned admission allocated {bytes}");
        assert_eq!(s.size_ledger().state_payload_bytes, expected);
        assert_eq!(s.checkpoint_bytes().unwrap().len(), 141);
        let (result, bytes) = measured(|| s.step(&[&[]], &[&[]]));
        assert_eq!(result.unwrap().owners[0].selected_units, 0);
        assert!(bytes < 32768, "pruned step allocated {bytes}");
        eprintln!(
            "NATIVE_PV_PRUNED rows={rows} logical_coordinates={} step_allocated_requests={bytes}",
            rows * 256
        );
    }
    let mut p = plan(1);
    p.limits.max_state_bytes = 1;
    let weight = parent(4096, false);
    let (result, bytes) = measured(|| Session::new(p, vec![weight]));
    assert!(result.is_err());
    assert!(bytes < 65536, "rejected admission allocated {bytes}");
    eprintln!("NATIVE_PV_REJECTED_STATE logical_coordinates=1048576 allocated_requests={bytes}");
}
