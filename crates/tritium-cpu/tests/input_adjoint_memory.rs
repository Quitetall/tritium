//! Allocator requests during packed backward only, not full-model RSS.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_core::{PerTilePlan, Trit, UniformPlan};
use tritium_cpu::CpuBackend;
use tritium_format::AdditiveTensor;
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, Transport,
};
use tritium_spec::{TensorView, TernaryBackend};

struct Allocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
// SAFETY: requests, pointers and layouts are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: caller supplies a valid allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: caller supplies the original pointer and allocation layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: caller supplies a live pointer, original layout and valid size.
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
    let value = call();
    TRACK.store(false, Ordering::Relaxed);
    (value, REQUESTED.load(Ordering::Relaxed))
}

#[test]
fn packed_input_adjoint_allocates_no_decoded_master_or_hidden_scratch() {
    let (control, bytes) = measured(|| vec![7u8; 8192]);
    assert_eq!(std::hint::black_box(control)[0], 7);
    assert!(bytes >= 8192, "allocation tracking must be active");
    for rows in [1, 256] {
        for compact in [false, true] {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                let meta = AdditiveLayout {
                    rows,
                    cols: 260,
                    tile: 256,
                    group: 128,
                    max_planes: 3,
                    allocation: if compact {
                        PlaneAllocation::PerTile
                    } else {
                        PlaneAllocation::Uniform
                    },
                    codec,
                    law: ADMITTED_LAWS[0].law,
                    basis: Basis::SignedRht {
                        block: 4,
                        seed: 19,
                        domain: 7,
                    },
                    transport: Transport::Raw,
                };
                let make_trits = |count| {
                    (0..count)
                        .map(|i| [Trit::ZERO, Trit::POS, Trit::NEG, Trit::POS][i % 4])
                        .collect()
                };
                let source = if compact {
                    let counts: Vec<u8> = (0..rows * 2).map(|i| (i % 4) as u8).collect();
                    let plan = PerTilePlan::new(meta, &counts).unwrap();
                    let (trit_count, scale_count) = (plan.trit_count(), plan.scale_count());
                    AdditiveTensor::new_per_tile(
                        meta,
                        counts,
                        make_trits(trit_count),
                        vec![0.5; scale_count],
                    )
                    .unwrap()
                } else {
                    let plan = UniformPlan::new(meta, 3).unwrap();
                    AdditiveTensor::new(
                        meta,
                        3,
                        make_trits(plan.trit_count()),
                        vec![0.5; plan.scale_count()],
                    )
                    .unwrap()
                };
                let handle = CpuBackend::new()
                    .upload_tensor(TensorView::Additive(source.view()))
                    .unwrap();
                // The operation cannot use a borrowed decoded source after this.
                drop(source);
                let before = handle.len_bytes();
                let gradients = vec![0.25; 2 * rows as usize];
                let mut scratch = vec![0.0; 260];
                let mut output = vec![0.0; 520];
                let (result, bytes) = measured(|| {
                    CpuBackend::new().additive_matmul_input_vjp(
                        &*handle,
                        &gradients,
                        2,
                        &mut scratch,
                        &mut output,
                    )
                });
                assert_eq!(result, Ok(()));
                assert_eq!(bytes, 0);
                assert_eq!(handle.len_bytes(), before);
                assert!(output.iter().all(|value| value.is_finite()));
                eprintln!(
                    "PACKED_INPUT_ADJOINT_HEAP rows={rows} compact={compact} codec={codec:?} bytes={bytes}"
                );
                let (count, bytes) =
                    measured(|| CpuBackend::new().additive_stored_scale_count(&*handle));
                let count = count.unwrap();
                assert_eq!(bytes, 0);
                let mut scale_gradients = vec![0.25; count];
                let x = vec![0.125; 520];
                let (result, bytes) = measured(|| {
                    CpuBackend::new().accumulate_additive_matmul_scale_vjp(
                        &*handle,
                        &x,
                        &gradients,
                        2,
                        &mut scratch,
                        &mut scale_gradients,
                    )
                });
                assert_eq!(result, Ok(()));
                assert_eq!(bytes, 0);
                eprintln!(
                    "PACKED_MATMUL_SCALE_HEAP rows={rows} compact={compact} codec={codec:?} bytes={bytes}"
                );
                let ids = [rows as usize - 1, 0, rows as usize - 1];
                let de = vec![0.25; 780];
                let (result, bytes) = measured(|| {
                    CpuBackend::new().accumulate_additive_embed_scale_vjp(
                        &*handle,
                        &ids,
                        &de,
                        &mut scratch,
                        &mut scale_gradients,
                    )
                });
                assert_eq!(result, Ok(()));
                assert_eq!(bytes, 0);
                assert_eq!(handle.len_bytes(), before);
                assert!(scale_gradients.iter().all(|v| v.is_finite()));
                eprintln!(
                    "PACKED_EMBED_SCALE_HEAP rows={rows} compact={compact} codec={codec:?} bytes={bytes}"
                );
            }
        }
    }
}
