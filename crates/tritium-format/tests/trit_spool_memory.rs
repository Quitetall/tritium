//! Source/output payloads are generated/hashed without retaining their bodies.
//! This measures Rust heap allocation for this writer, not complete process RSS.
use std::alloc::{GlobalAlloc, Layout, System};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_format::{PackageHasher, TritBlobKind, TritSpoolBlob, write_trit_package_from_spool};
use tritium_schema::BlobId;

struct TrackingAllocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

// SAFETY: delegates exact allocator layouts/pointers to System.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller supplies a valid allocation layout.
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        result
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the caller supplies the original pointer and layout.
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller supplies the original pointer/layout and valid size.
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() && TRACK.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(size, Ordering::Relaxed);
        }
        result
    }
}
#[global_allocator]
static ALLOCATOR: TrackingAllocator = TrackingAllocator;

struct GeneratedSource {
    size: u64,
    at: u64,
    largest_request: usize,
}
impl Read for GeneratedSource {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        self.largest_request = self.largest_request.max(out.len());
        let count = (self.size - self.at.min(self.size)).min(out.len() as u64) as usize;
        out[..count].fill(23);
        self.at += count as u64;
        Ok(count)
    }
}
impl Seek for GeneratedSource {
    fn seek(&mut self, at: SeekFrom) -> io::Result<u64> {
        self.at = match at {
            SeekFrom::Start(at) => at,
            SeekFrom::End(0) => self.size,
            _ => return Err(io::Error::other("unsupported test seek")),
        };
        Ok(self.at)
    }
}
struct HashSink {
    hash: PackageHasher,
    bytes: usize,
}
impl Write for HashSink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.hash.update(bytes);
        self.bytes += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn spool_heap_allocation_does_not_scale_with_payload_size() {
    let mut totals = Vec::new();
    for size in [1024 * 1024, 16 * 1024 * 1024] {
        let mut digest = blake3::Hasher::new();
        for _ in 0..size / 65536 {
            digest.update(&[23; 65536]);
        }
        let blobs = [TritSpoolBlob {
            kind: TritBlobKind::AdditiveTensor,
            id: BlobId::from_bytes(*digest.finalize().as_bytes()),
            offset: 0,
            length: size as u64,
        }];
        let mut source = GeneratedSource {
            size: size as u64,
            at: 0,
            largest_request: 0,
        };
        let mut sink = HashSink {
            hash: PackageHasher::new(),
            bytes: 0,
        };
        ALLOCATED.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::SeqCst);
        let result =
            write_trit_package_from_spool(&[0x80], &blobs, &mut source, size + 1024, &mut sink);
        TRACK.store(false, Ordering::SeqCst);
        let allocated = ALLOCATED.load(Ordering::Relaxed);
        assert_eq!(result.unwrap(), sink.hash.finalize());
        assert!(sink.bytes > size);
        assert_eq!(source.largest_request, 65536);
        assert!(
            allocated < 256 * 1024,
            "writer allocated {allocated} heap bytes for {size}-byte source"
        );
        totals.push(allocated);
    }
    assert!(
        totals[1].abs_diff(totals[0]) < 4096,
        "metadata-equivalent payload growth changed heap allocation: {totals:?}"
    );
}
