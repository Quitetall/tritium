//! Cumulative staging heap requests with actual Files, not RSS or source storage.
use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_format::TritPackageReadLimits;
use tritium_format::trit_model::{read_trit_model, stage_trit_model_from_spool};
#[path = "support/typed_seek_fixture.rs"]
mod fixture;
#[path = "support/model_spool_fixture.rs"]
mod preparation;

struct Allocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
// SAFETY: requests, layouts and pointers are forwarded unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: caller provides a valid layout.
        let output = unsafe { System.alloc(layout) };
        if !output.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        output
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: caller provides the original pointer/layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: caller provides the original pointer/layout and valid size.
        let output = unsafe { System.realloc(pointer, layout, size) };
        if !output.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(size, Ordering::Relaxed);
        }
        output
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

#[test]
fn file_to_file_complete_model_staging_has_no_payload_proportional_heap_requests() {
    let mut requests = Vec::new();
    for dense in [1024 * 1024, 16 * 1024 * 1024] {
        let bytes = fixture::fixture(257, dense);
        let (manifest, blobs) = preparation::prepared(&bytes);
        let expected = {
            let model = read_trit_model(&bytes).unwrap();
            (
                model.package_id(),
                model.model_id("child").unwrap(),
                model.model_id("parent").unwrap(),
            )
        };
        let source_path = std::env::temp_dir().join(format!(
            "tritium-model-stage-memory-{}-{dense}.spool",
            std::process::id()
        ));
        let output_path = source_path.with_extension("pending");
        struct Cleanup(Vec<std::path::PathBuf>);
        impl Drop for Cleanup {
            fn drop(&mut self) {
                for path in &self.0 {
                    let _ = fs::remove_file(path);
                }
            }
        }
        let mut source = File::create_new(&source_path).unwrap();
        source.write_all(&bytes).unwrap();
        drop(source);
        let cleanup = Cleanup(vec![source_path.clone(), output_path.clone()]);
        let mut source = File::open(&source_path).unwrap();
        let output = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&output_path)
            .unwrap();
        let limits = TritPackageReadLimits {
            max_source_bytes: bytes.len() as u64,
            max_manifest_bytes: 64 * 1024,
            max_directory_bytes: 64 * 1024,
        };
        drop(bytes); // prepared encoded source is not retained in the measured window
        REQUESTED.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::Relaxed);
        let mut calls = 0;
        let mut staged = stage_trit_model_from_spool(
            &manifest,
            &blobs,
            &mut source,
            limits,
            output,
            || -> Result<(), ()> {
                calls += 1;
                Ok(())
            },
        )
        .unwrap();
        staged.validate_complete_payload_inventory().unwrap();
        let mut visited = 0;
        staged
            .visit_tensor("child", "norm.weight", |chunk| -> Result<(), ()> {
                visited += chunk.len();
                Ok(())
            })
            .unwrap();
        staged.verify().unwrap();
        TRACK.store(false, Ordering::Relaxed);
        let requested = REQUESTED.load(Ordering::Relaxed);
        assert_eq!(calls, 1);
        assert_eq!(visited, dense);
        assert_eq!(staged.package_id(), expected.0);
        assert_eq!(staged.model_id("child").unwrap(), expected.1);
        assert_eq!(staged.model_id("parent").unwrap(), expected.2);
        drop(staged);
        drop(source);
        drop(cleanup);
        assert!(!source_path.exists() && !output_path.exists());
        eprintln!(
            "MODEL_SPOOL_STAGE_MEMORY dense_payload={dense} cumulative_heap_requests={requested}"
        );
        assert!(
            requested < 256 * 1024,
            "staging requested {requested} heap bytes"
        );
        requests.push(requested);
    }
    assert!(
        requests[0].abs_diff(requests[1]) <= 1024,
        "payload growth changed staging heap requests: {requests:?}"
    );
}
