//! Reader-owned cumulative heap requests, excluding prepared source/page cache.
use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::{self, File};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_format::{
    TritBlob, TritBlobKind, TritPackageReadLimits, TritPackageReader, read_trit_package,
    write_trit_package, write_trit_package_to,
};

struct TrackingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

// SAFETY: all allocation requests/pointers are forwarded unchanged to System.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && TRACK.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies the original pointer and layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller supplies the original pointer/layout and valid size.
        let output = unsafe { System.realloc(pointer, layout, size) };
        if !output.is_null() && TRACK.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(size, Ordering::Relaxed);
        }
        output
    }
}
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

#[test]
fn actual_file_open_visit_and_terminal_verify_do_not_allocate_payload_sized_buffers() {
    let mut requests = Vec::new();
    for size in [1024 * 1024, 16 * 1024 * 1024] {
        let path = std::env::temp_dir().join(format!(
            "tritium-seek-package-memory-{}-{size}.trit",
            std::process::id()
        ));
        let payload = vec![3u8; size];
        let mut destination = File::create_new(&path).unwrap();
        let expected = write_trit_package_to(
            &[0xa0],
            &[TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &payload,
            }],
            usize::MAX,
            &mut destination,
        )
        .unwrap();
        drop(destination);
        drop(payload);
        let source = File::open(&path).unwrap();
        let length = source.metadata().unwrap().len();
        let limits = TritPackageReadLimits {
            max_source_bytes: length,
            max_manifest_bytes: 1,
            max_directory_bytes: 128,
        };
        ALLOCATED.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::Relaxed);
        let mut reader = TritPackageReader::open(source, limits).unwrap();
        let mut seen = 0usize;
        let mut hash = blake3::Hasher::new();
        reader
            .visit_blob(0, |chunk| -> Result<(), ()> {
                seen += chunk.len();
                hash.update(chunk);
                Ok(())
            })
            .unwrap();
        reader.verify().unwrap();
        TRACK.store(false, Ordering::Relaxed);
        let requested = ALLOCATED.load(Ordering::Relaxed);
        assert_eq!(reader.package_id(), expected);
        assert_eq!(hash.finalize().as_bytes(), reader.blobs()[0].id.as_bytes());
        assert_eq!(seen, size);
        assert!(
            requested < 64 * 1024,
            "reader requested {requested} heap bytes"
        );
        requests.push(requested);
        drop(reader);
        fs::remove_file(path).unwrap();
        eprintln!(
            "SEEK_PACKAGE_MEMORY payload={size} source={length} cumulative_heap_requests={requested}"
        );
    }
    assert_eq!(
        requests[0], requests[1],
        "payload growth must not grow reader heap requests"
    );
    // Isolated process and one test: the cumulative allocator window cannot
    // overlap another test. Prepared source/asset metadata is excluded.
    let blobs: Vec<_> = (0..4096)
        .map(|index| TritBlob {
            kind: TritBlobKind::Asset(format!("asset-{index:06}")),
            bytes: &[],
        })
        .collect();
    let bytes = write_trit_package(&[0xa0], &blobs).unwrap();
    let expected = tritium_format::PackageId::from_package_bytes(&bytes);
    ALLOCATED.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let package = read_trit_package(&bytes).unwrap();
    TRACK.store(false, Ordering::Relaxed);
    let requested = ALLOCATED.load(Ordering::Relaxed);
    assert_eq!(package.package_id(), expected);
    assert_eq!(package.blobs().len(), blobs.len());
    eprintln!(
        "DIRECTORY_MEMORY entries=4096 adapter=borrowed cumulative_heap_requests={requested}"
    );
    assert!(
        requested < 512 * 1024,
        "directory requested {requested} heap bytes"
    );
    drop(package);
    let limits = TritPackageReadLimits {
        max_source_bytes: bytes.len() as u64,
        max_manifest_bytes: 1,
        max_directory_bytes: bytes.len(),
    };
    ALLOCATED.store(0, Ordering::Relaxed);
    TRACK.store(true, Ordering::Relaxed);
    let mut reader = TritPackageReader::open(std::io::Cursor::new(bytes), limits).unwrap();
    reader.verify().unwrap();
    TRACK.store(false, Ordering::Relaxed);
    let requested = ALLOCATED.load(Ordering::Relaxed);
    assert_eq!(reader.package_id(), expected);
    assert_eq!(reader.blobs().len(), blobs.len());
    eprintln!("DIRECTORY_MEMORY entries=4096 adapter=seek cumulative_heap_requests={requested}");
    assert!(
        requested < 1024 * 1024,
        "seek directory requested {requested} heap bytes"
    );
}
