use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tritium_format::{SaltBundleReader, SaltRow, TQ2_0_BLOCK_BYTES, num_blocks, write_salt_bundle};

struct TrackingAllocator;

static TRACK_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every operation delegates to `System` with the exact layout/pointer it received.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: delegated under the caller's GlobalAlloc contract.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: delegated under the caller's GlobalAlloc contract.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: delegated under the caller's GlobalAlloc contract.
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() && TRACK_ALLOCATIONS.load(Ordering::Relaxed) {
            ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        replacement
    }
}

#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn strict_reader_allocation_is_row_bounded_not_bundle_bounded() {
    let k = 1_000_000;
    let plane_bytes = num_blocks(k) * TQ2_0_BLOCK_BYTES;
    let large_rows = (0..80)
        .map(|_| SaltRow {
            k,
            planes: vec![vec![0; plane_bytes]],
        })
        .collect::<Vec<_>>();
    let small_rows = [SaltRow {
        k: 256,
        planes: vec![vec![0; TQ2_0_BLOCK_BYTES]],
    }];
    let bundle = write_salt_bundle(&[
        ("large.unselected", large_rows.as_slice()),
        ("small.selected", small_rows.as_slice()),
    ])
    .unwrap();
    drop(large_rows);
    assert!(bundle.len() > 16 * 1024 * 1024);

    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);
    let mut reader = SaltBundleReader::new_strict(Cursor::new(bundle.as_slice())).unwrap();
    reader
        .visit_packed_tensor("small.selected", |_| {})
        .unwrap();
    TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);

    let allocated = ALLOCATED_BYTES.load(Ordering::Relaxed);
    assert!(
        allocated < 4 * 1024 * 1024,
        "strict scan allocated {allocated} bytes for a {}-byte borrowed bundle",
        bundle.len()
    );
}
