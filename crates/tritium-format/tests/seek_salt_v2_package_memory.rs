use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use half::f16;
use tritium_format::salt_v2_package::{
    SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_PACKAGE_ALIGNMENT, SALT_V2_PACKAGE_MAGIC,
    SALT_V2_PACKAGE_VERSION, SaltV2PackageReader,
};

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

fn large_zero_package() -> Vec<u8> {
    let name = b"large.zero";
    let coefficients = 32 * 1024 * 1024usize;
    assert!(coefficients.is_multiple_of(SALT_V2_ALLOCATION_TILE_SIZE));
    let tile_count = coefficients / SALT_V2_ALLOCATION_TILE_SIZE;
    let payload_bytes = coefficients / 4;
    let scale_bytes = tile_count * 2 * core::mem::size_of::<u16>();
    let map_bytes = tile_count * 2 / 8;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&SALT_V2_PACKAGE_MAGIC);
    bytes.extend_from_slice(&SALT_V2_PACKAGE_VERSION.to_le_bytes());
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());

    bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(coefficients as u64).to_le_bytes());
    bytes.extend_from_slice(&(tile_count as u64).to_le_bytes());
    bytes.extend_from_slice(&(payload_bytes as u64).to_le_bytes());
    bytes.extend_from_slice(&(scale_bytes as u64).to_le_bytes());
    bytes.extend_from_slice(&[0; 24]);
    bytes.extend_from_slice(name);
    bytes.extend_from_slice(&(coefficients as u64).to_le_bytes());
    bytes.resize(bytes.len() + payload_bytes, 0x55);
    for _ in 0..tile_count * 2 {
        bytes.extend_from_slice(&f16::ONE.to_bits().to_le_bytes());
    }
    bytes.resize(bytes.len() + map_bytes, 0);
    let padding = (SALT_V2_PACKAGE_ALIGNMENT - bytes.len() % SALT_V2_PACKAGE_ALIGNMENT)
        % SALT_V2_PACKAGE_ALIGNMENT;
    bytes.resize(bytes.len() + padding, 0);
    let total = bytes.len() as u64;
    bytes[16..24].copy_from_slice(&total.to_le_bytes());
    bytes
}

#[test]
fn strict_reader_reuses_decode_scratch_instead_of_materializing_semantic_trits() {
    let package = large_zero_package();
    assert!(package.len() > 8 * 1024 * 1024);

    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    TRACK_ALLOCATIONS.store(true, Ordering::SeqCst);
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(package.as_slice())).unwrap();
    let mut visited = 0usize;
    reader
        .visit_packed_tensor("large.zero", |_| visited += 1)
        .unwrap();
    reader.verify_unchanged().unwrap();
    TRACK_ALLOCATIONS.store(false, Ordering::SeqCst);

    assert_eq!(visited, 32 * 1024 * 1024 / SALT_V2_ALLOCATION_TILE_SIZE);
    let allocated = ALLOCATED_BYTES.load(Ordering::Relaxed);
    assert!(
        allocated < 2 * 1024 * 1024,
        "strict reader cumulatively allocated {allocated} bytes for a {}-byte borrowed package",
        package.len()
    );
}
