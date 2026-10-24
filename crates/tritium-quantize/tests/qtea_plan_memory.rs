//! Planning allocator requests, not model RSS or machine-level admission.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_format::qtea::QteaLimits;
use tritium_quantize::qtea_reference::{TensorFit, TensorFitConfig};
use tritium_quantize::{CurvatureSourceId, InputGramAccumulator};

struct Allocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static REQUESTS: AtomicUsize = AtomicUsize::new(0);
fn request() {
    if TRACK.load(Ordering::Relaxed) {
        REQUESTS.fetch_add(1, Ordering::Relaxed);
    }
}
// SAFETY: all requests/pointers/layouts are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        request();
        // SAFETY: caller provides a valid layout.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        request();
        // SAFETY: caller provides a valid layout.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: caller provides original pointer/layout.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        request();
        // SAFETY: caller provides original pointer/layout and valid requested size.
        unsafe { System.realloc(p, l, n) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn measured<T>(call: impl FnOnce() -> T) -> (T, usize) {
    REQUESTS.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let value = call();
    TRACK.store(false, Ordering::Relaxed);
    (value, REQUESTS.load(Ordering::Relaxed))
}
#[test]
fn actual_planning_and_byte_ledger_request_no_heap() {
    let (control, count) = measured(|| vec![7u8; 8192]);
    assert_eq!(std::hint::black_box(control)[0], 7);
    assert!(count > 0);
    for columns in [4, 12, 128] {
        let rows = 4;
        let weights = (0..rows * columns)
            .map(|i| (i % 7) as f32 * 0.125 - 0.3)
            .collect::<Vec<_>>();
        let source = CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap();
        let mut builder = InputGramAccumulator::new_bound(columns, source, 0).unwrap();
        builder
            .accumulate_batch(&vec![1.; columns], 1, None, None)
            .unwrap();
        let gram = builder.finish().unwrap();
        let config = TensorFitConfig {
            group_size: 2,
            ..TensorFitConfig::default()
        };
        let cap = TensorFit::required_work_bytes(rows, columns).unwrap();
        let (plan, requests) =
            measured(|| TensorFit::plan(&weights, rows, columns, &gram, config, cap));
        let plan = plan.unwrap();
        assert_eq!(requests, 0, "C={columns}");
        let (ledger, requests) = measured(|| {
            plan.encoded_ledger(QteaLimits {
                max_tensor_bytes: 1 << 20,
                max_validation_bytes: columns,
            })
        });
        assert!(ledger.unwrap().total_bytes > 0);
        assert_eq!(requests, 0);
    }
}
