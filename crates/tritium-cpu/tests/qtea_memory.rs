//! Allocator requests for caller-scratch CPU forward, not physical model RSS.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_core::Trit;
use tritium_cpu::qtea::matmul_into;
use tritium_format::qtea::{
    QteaBindings, QteaGroupRef, QteaLimits, QteaMetadata, QteaRecipe, QteaResidualRef, QteaTensor,
};

struct Allocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static REQUESTS: AtomicUsize = AtomicUsize::new(0);
fn requested() {
    if TRACK.load(Ordering::Relaxed) {
        REQUESTS.fetch_add(1, Ordering::Relaxed);
    }
}
// SAFETY: pointers/layouts are forwarded unchanged to System; only requests counted.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        requested();
        // SAFETY: allocator caller provides a valid layout.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        requested();
        // SAFETY: allocator caller provides a valid layout.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: allocator caller provides original pointer/layout.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        requested();
        // SAFETY: caller provides original pointer/layout and valid requested size.
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn measured<T>(call: impl FnOnce() -> T) -> (T, usize) {
    REQUESTS.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let output = call();
    TRACK.store(false, Ordering::Relaxed);
    (output, REQUESTS.load(Ordering::Relaxed))
}

#[test]
fn caller_scratch_forward_requests_no_heap_or_dense_weight_buffer() {
    let (control, count) = measured(|| vec![7_u8; 8192]);
    assert_eq!(std::hint::black_box(control)[0], 7);
    assert!(count > 0);
    let recipe = QteaRecipe {
        group_size: 2,
        salient_ratio: 1.,
        delta_coefficient: 0.96,
        fit_iterations: 1,
        rescale_iterations: 2,
        damping: 0.007,
        decay: 1.,
        relative_pivot_floor: 1e-7,
    };
    for (rows, columns) in [(4, 4), (68, 258), (256, 512)] {
        let meta = QteaMetadata {
            rows,
            columns,
            recipe,
            bindings: QteaBindings {
                weights: [1; 32],
                gram: [2; 32],
                source_model: [3; 32],
                activation_cache: [4; 32],
                token_stream: [5; 32],
            },
        };
        let trits = vec![Trit::POS; rows * columns];
        let alpha = vec![1.; rows];
        let beta = vec![0.; rows];
        let v = vec![1.; columns];
        let order = (0..columns).collect::<Vec<_>>();
        let positions = vec![0; rows / 4];
        let codes = vec![0x38; rows / 4];
        let tensor = QteaTensor::from_parts(
            meta,
            &trits,
            &v,
            &order,
            std::iter::repeat_n(
                QteaGroupRef {
                    alpha: &alpha,
                    beta: &beta,
                },
                columns / 2,
            ),
            (0..columns).map(|column| QteaResidualRef {
                column,
                scale_multiplier: 2.,
                indices: &positions,
                codes: &codes,
            }),
            QteaLimits {
                max_tensor_bytes: 4 << 20,
                max_validation_bytes: columns,
            },
        )
        .unwrap();
        drop((trits, alpha, beta, v, order, positions, codes));
        let input = vec![0.25; 3 * columns];
        let mut output = vec![99.; 3 * rows];
        let mut scratch = vec![88.; 3 * rows];
        let (result, requests) =
            measured(|| matmul_into(tensor.view(), &input, 3, &mut output, &mut scratch));
        result.unwrap();
        assert_eq!(requests, 0, "R={rows} C={columns}");
        for b in 0..3 {
            for row in 0..rows {
                assert_eq!(
                    output[b * rows + row],
                    columns as f32 * if row % 4 == 0 { 0.375 } else { 0.25 }
                );
            }
        }
    }
}
