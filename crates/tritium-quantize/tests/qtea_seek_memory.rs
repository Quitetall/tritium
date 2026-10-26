//! Heap-request and real file-backed scope, not RSS/page-cache qualification.
use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};
use tritium_quantize::qtea_reference::{SeekBackedFitLimits, TensorFit, TensorFitConfig};
use tritium_quantize::{CurvatureSourceId, InputGramAccumulator};

struct Allocator;
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);
static TRACK: AtomicBool = AtomicBool::new(false);
static MATRIX_BYTES: AtomicUsize = AtomicUsize::new(0);
static MATRIX_REQUESTS: AtomicUsize = AtomicUsize::new(0);
fn grow(bytes: usize) {
    let live = LIVE.fetch_add(bytes as isize, Ordering::Relaxed) + bytes as isize;
    if TRACK.load(Ordering::Relaxed) {
        PEAK.fetch_max(live, Ordering::Relaxed);
        if bytes == MATRIX_BYTES.load(Ordering::Relaxed) {
            MATRIX_REQUESTS.fetch_add(1, Ordering::Relaxed);
        }
    }
}
// SAFETY: pointers/layouts are forwarded unchanged; counters do not access storage.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            grow(layout.size());
        }
        pointer
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid layout.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            grow(layout.size());
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        // SAFETY: the caller supplies the original pointer/layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller supplies the original pointer/layout and valid size.
        let output = unsafe { System.realloc(pointer, layout, size) };
        if !output.is_null() {
            LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
            grow(size);
        }
        output
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
fn measured<T>(call: impl FnOnce() -> T) -> (T, usize, usize) {
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    MATRIX_REQUESTS.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let output = call();
    TRACK.store(false, Ordering::Relaxed);
    (
        output,
        (PEAK.load(Ordering::Relaxed) - baseline) as usize,
        MATRIX_REQUESTS.load(Ordering::Relaxed),
    )
}
struct Scratch {
    file: Option<File>,
    path: PathBuf,
}
impl Drop for Scratch {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = std::fs::remove_file(&self.path);
    }
}

#[test]
fn real_file_replays_exactly_without_a_second_full_feedback_allocation() {
    let (rows, columns, group_size) = (256, 64, 8);
    let weights = (0..rows * columns)
        .map(|i| ((i * 17 + i / columns) % 41) as f32 / 16. - 1.)
        .collect::<Vec<_>>();
    let mut identity = vec![0.; columns * columns];
    for c in 0..columns {
        identity[c * columns + c] = 1.;
    }
    let source = CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap();
    let mut builder = InputGramAccumulator::new_bound(columns, source, 0).unwrap();
    builder
        .accumulate_batch(&identity, columns, None, None)
        .unwrap();
    let gram = builder.finish().unwrap();
    let config = TensorFitConfig {
        group_size,
        salient_ratio: 0.25,
        ..TensorFitConfig::default()
    };
    let path =
        std::env::temp_dir().join(format!("tritium-qtea-seek-{}.scratch", std::process::id()));
    let mut scratch = Scratch {
        file: Some(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap(),
        ),
        path: path.clone(),
    };
    MATRIX_BYTES.store(rows * columns * 4, Ordering::Relaxed);
    let (normal, resident_peak, resident_requests) =
        measured(|| TensorFit::fit(&weights, rows, columns, &gram, config, usize::MAX).unwrap());
    let caps = SeekBackedFitLimits {
        max_work_bytes: TensorFit::required_seek_work_bytes(rows, columns, group_size).unwrap(),
        max_scratch_bytes: TensorFit::required_seek_scratch_bytes(rows, columns).unwrap(),
    };
    let (actual, seek_peak, seek_requests) = measured(|| {
        TensorFit::fit_seek_backed(
            &weights,
            rows,
            columns,
            &gram,
            config,
            scratch.file.as_mut().unwrap(),
            caps,
        )
        .unwrap()
    });
    assert_eq!(actual, normal);
    let codec = tritium_format::qtea::QteaLimits {
        max_tensor_bytes: 1 << 20,
        max_validation_bytes: columns,
    };
    assert_eq!(
        actual.to_qtea_tensor(codec).unwrap().as_bytes(),
        normal.to_qtea_tensor(codec).unwrap().as_bytes()
    );
    assert_eq!(resident_requests, 2, "original copy plus mutable feedback");
    assert_eq!(
        seek_requests, 1,
        "original copy only; no RC feedback allocation"
    );
    assert!(
        seek_peak < resident_peak,
        "seek={seek_peak}, resident={resident_peak}"
    );
    assert!(seek_peak <= caps.max_work_bytes);
    assert!(caps.max_work_bytes < TensorFit::required_work_bytes(rows, columns).unwrap());
    assert_eq!(
        scratch.file.as_ref().unwrap().metadata().unwrap().len(),
        caps.max_scratch_bytes
    );
    println!(
        "QTEA_SEEK_HEAP_SCOPE resident_requested_peak={resident_peak} seek_requested_peak={seek_peak} scratch_bytes={} (not RSS/page cache/qualification)",
        caps.max_scratch_bytes
    );
    drop(scratch);
    assert!(!path.exists());
}
