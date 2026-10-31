//! Allocator-request checks, not physical memory/RSS qualification.
#[path = "qtea/common.rs"]
mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tritium_nn::HostQteaLinear;

struct Allocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static REQUESTS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
fn requested(bytes: usize) {
    if TRACK.load(Ordering::Relaxed) {
        REQUESTS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(bytes, Ordering::Relaxed);
    }
}
// SAFETY: requests are counted and all original pointers/layouts passed to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        requested(layout.size());
        // SAFETY: allocator caller supplies a valid allocation layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        requested(layout.size());
        // SAFETY: allocator caller supplies a valid allocation layout.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: caller supplies the original live allocation and layout.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        requested(new_size);
        // SAFETY: caller supplies a valid allocation/layout and requested size.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn measured<T>(call: impl FnOnce() -> T) -> (T, usize, usize) {
    REQUESTS.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let result = call();
    TRACK.store(false, Ordering::Relaxed);
    (
        result,
        REQUESTS.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

#[test]
fn packed_move_and_caller_workspace_do_not_expand_weights() {
    let (control, count, bytes) = measured(|| vec![7_u8; 8192]);
    assert_eq!(std::hint::black_box(control)[0], 7);
    assert!(count > 0 && bytes >= 8192);
    for (rows, columns) in [(4, 4), (68, 258), (128, 512)] {
        let (tensor, decoded) = common::fixture(rows, columns, 2, 1., 0);
        let tokens = [0, rows as u32 - 1, 0];
        let gathered = tokens
            .iter()
            .flat_map(|&row| {
                decoded[row as usize * columns..(row as usize + 1) * columns]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        drop(decoded);
        let encoded_capacity = tensor.buffer_capacity_bytes();
        let (matrix, count, bytes) =
            measured(|| HostQteaLinear::from_tensor(tensor, 3 * rows.max(columns) * 4));
        assert_eq!((count, bytes), (0, 0));
        assert_eq!(matrix.encoded_capacity_bytes(), encoded_capacity);
        let matrix = Arc::new(matrix);
        let (clone, count, bytes) = measured(|| Arc::clone(&matrix));
        assert_eq!((count, bytes), (0, 0));
        assert!(Arc::ptr_eq(&matrix, &clone));
        let input = vec![0.25; 3 * columns];
        let mut output = vec![55.; 3 * rows];
        let mut staging = vec![66.; 3 * rows];
        let (result, count, bytes) =
            measured(|| matrix.forward_into(&input, 3, &mut output, &mut staging));
        result.unwrap();
        assert_eq!((count, bytes), (0, 0));
        let expected = output.clone();
        let (result, count, bytes) = measured(|| matrix.forward(&input, 3, &mut output));
        result.unwrap();
        assert_eq!((count, bytes), (1, 3 * rows * 4));
        assert_eq!(output, expected);
        let mut gather_output = vec![55.; 3 * columns];
        let mut gather_staging = vec![66.; gather_output.len()];
        let (result, count, bytes) =
            measured(|| matrix.gather_into(&tokens, &mut gather_output, &mut gather_staging));
        result.unwrap();
        assert_eq!((count, bytes), (0, 0));
        assert_eq!(gather_output, gathered);
        let (result, count, bytes) = measured(|| matrix.gather(&tokens, &mut gather_output));
        result.unwrap();
        assert_eq!((count, bytes), (1, 3 * columns * 4));
        assert_eq!(gather_output, gathered);
        let (result, count, bytes) = measured(|| matrix.forward(&[], 0, &mut []));
        result.unwrap();
        assert_eq!((count, bytes), (0, 0));
    }
}
