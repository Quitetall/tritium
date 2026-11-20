//! Cumulative typed-reader allocation requests, not source buffers/RSS/page cache.
use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::{self, File};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_format::trit_model::decode_additive_payload;
use tritium_format::trit_model::{TritModelReader, read_trit_model};
#[path = "support/typed_seek_fixture.rs"]
mod fixture;
#[path = "support/seek_model_parity.rs"]
mod parity;

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
fn retained_file_typed_validation_does_not_allocate_a_dense_or_encoded_payload_master() {
    let mut requests = Vec::new();
    for dense_size in [1024 * 1024, 16 * 1024 * 1024] {
        let bytes = fixture::fixture(257, dense_size);
        parity::assert_matches(&bytes);
        let expected = {
            let model = read_trit_model(&bytes).unwrap();
            (
                model.package_id(),
                model.model_id("child").unwrap(),
                model.model_id("parent").unwrap(),
                model.tensor_digest("child", "norm.weight").unwrap(),
            )
        };
        let limits = parity::limits(&bytes);
        let path = std::env::temp_dir().join(format!(
            "tritium-typed-seek-memory-{}-{dense_size}.trit",
            std::process::id()
        ));
        let mut destination = File::create_new(&path).unwrap();
        std::io::Write::write_all(&mut destination, &bytes).unwrap();
        drop(destination);
        drop(bytes);
        let source = File::open(&path).unwrap();
        ALLOCATED.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::Relaxed);
        let mut reader = TritModelReader::open(source, limits).unwrap();
        reader.validate_complete_payload_inventory().unwrap();
        let mut seen = 0usize;
        reader
            .visit_tensor("child", "norm.weight", |chunk| -> Result<(), ()> {
                seen += chunk.len();
                Ok(())
            })
            .unwrap();
        reader.verify().unwrap();
        TRACK.store(false, Ordering::Relaxed);
        let requested = ALLOCATED.load(Ordering::Relaxed);
        assert_eq!(reader.package_id(), expected.0);
        assert_eq!(reader.model_id("child").unwrap(), expected.1);
        assert_eq!(reader.model_id("parent").unwrap(), expected.2);
        assert_eq!(
            reader.tensor_digest("child", "norm.weight").unwrap(),
            expected.3
        );
        assert_eq!(seen, dense_size);
        drop(reader);
        fs::remove_file(path).unwrap();
        requests.push(requested);
        eprintln!(
            "TYPED_SEEK_MEMORY dense_payload={dense_size} cumulative_heap_requests={requested}"
        );
        assert!(
            requested < 256 * 1024,
            "typed reader requested {requested} heap bytes"
        );
    }
    assert!(
        requests[0].abs_diff(requests[1]) <= 1024,
        "payload growth must not grow a reader-owned payload master: {requests:?}"
    );

    // The output owner is deliberately allocated, but no encoded-owner copy
    // accompanies it. Both Uniform and PerTile use an actual retained File.
    let bytes = fixture::fixture(1_048_577, 68);
    let path = std::env::temp_dir().join(format!(
        "tritium-owner-decode-memory-{}.trit",
        std::process::id()
    ));
    let mut file = File::create_new(&path).unwrap();
    std::io::Write::write_all(&mut file, &bytes).unwrap();
    drop(file);
    let mut reader =
        TritModelReader::open(File::open(&path).unwrap(), parity::limits(&bytes)).unwrap();
    for level in ["child", "parent"] {
        let model = read_trit_model(&bytes).unwrap();
        let (meta, payload) = model.tensor(level, "weight").unwrap();
        let expected = decode_additive_payload(meta, payload).unwrap();
        let mut stream = reader.tensor_stream(level, "weight").unwrap();
        let required = stream.additive_decode_working_bytes().unwrap();
        ALLOCATED.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::Relaxed);
        assert!(stream.decode_additive(required - 1).is_err());
        TRACK.store(false, Ordering::Relaxed);
        let rejected = ALLOCATED.load(Ordering::Relaxed);
        assert!(
            rejected < 16384,
            "cap refusal allocated owner payload {rejected}"
        );
        ALLOCATED.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::Relaxed);
        let actual = stream.decode_additive(required).unwrap();
        TRACK.store(false, Ordering::Relaxed);
        let requested = ALLOCATED.load(Ordering::Relaxed);
        assert_eq!(actual, expected);
        assert!(
            (requested as u64) < required - 65536 + 16384,
            "encoded-owner copy or duplicate decoded payload: {requested}, budget {required}"
        );
        eprintln!(
            "OWNER_STREAM_DECODE level={level} encoded_bytes={} decoded_trits={} cumulative_heap_requests={requested} declared_working={required} rejected_heap_requests={rejected}",
            payload.len(),
            actual.trits().len()
        );
    }
    reader.verify().unwrap();
    drop(reader);
    fs::remove_file(path).unwrap();
}
