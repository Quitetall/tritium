//! Actual public publisher/guard heap requests, not caller/source/process RSS.
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tritium_evidence::{
    DIRECT_LEVELS_FITTED_REGISTRATION, EvidenceRecorder, QUANTIZATION_COMPLETED_REGISTRATION,
    QUANTIZATION_STARTED_REGISTRATION, verify_jsonl,
};
use tritium_format::trit_model::decode_model_manifest;
use tritium_format::{TritPackageReadLimits, TritSpoolBlob, read_trit_package};
use tritium_quantize::engine::{
    DirectoryTensorSink, Engine, InMemoryWeightSource, MemoryTensorSink, QuantEngineError,
    RawTensor, WeightSource, verify_package_with_fit_log,
};
use tritium_schema::*;

struct Allocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static REQUESTED: AtomicUsize = AtomicUsize::new(0);
// SAFETY: every pointer/layout/request is delegated unchanged to System.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: caller supplies a valid layout.
        let output = unsafe { System.alloc(layout) };
        if !output.is_null() && TRACK.load(Ordering::Relaxed) {
            REQUESTED.fetch_add(layout.size(), Ordering::Relaxed);
        }
        output
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: caller supplies original pointer/layout.
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: caller supplies original pointer/layout and valid size.
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
fn public_file_publisher_and_actual_source_guard_have_no_payload_sized_heap_growth() {
    let root = std::env::temp_dir().join(format!(
        "tritium-spool-directory-memory-{}",
        std::process::id()
    ));
    fs::create_dir(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let cleanup = Cleanup(root.clone());
    let mut allocations = Vec::new();
    for dense in [1024 * 1024, 16 * 1024 * 1024] {
        let mut original = InMemoryWeightSource {
            tensors: BTreeMap::from([(
                "norm".into(),
                RawTensor {
                    shape: vec![(dense / 4) as u64],
                    precision: ScalePrecision::F32,
                    bytes: vec![0; dense],
                },
            )]),
            assets: BTreeMap::from([
                ("config.json".into(), b"{}".to_vec()),
                ("tokenizer.json".into(), b"{}".to_vec()),
            ]),
            grams: BTreeMap::new(),
        };
        let request = QuantPlan {
            architecture: "fixture".into(),
            tensors: original
                .inventory()
                .unwrap()
                .into_iter()
                .map(|source| QuantTensorPlan {
                    source,
                    action: QuantAction::KeepDense,
                })
                .collect(),
            assets: original.asset_names(),
            levels: vec!["imported".into()],
            limits: QuantLimits {
                max_source_bytes: dense as u64,
                max_asset_bytes: 100,
                max_retained_blob_bytes: dense as u64 + 100,
                max_package_bytes: dense as u64 + 65536,
            },
        };
        let recorder = EvidenceRecorder::new(
            "publisher-memory",
            [
                DIRECT_LEVELS_FITTED_REGISTRATION,
                QUANTIZATION_STARTED_REGISTRATION,
                QUANTIZATION_COMPLETED_REGISTRATION,
            ],
        )
        .unwrap();
        let mut memory = MemoryTensorSink::default();
        let prepared = Engine::run(&request, &mut original, &mut memory, recorder).unwrap();
        let bytes = memory.artifact().unwrap().0;
        let (manifest, blobs) = {
            let package = read_trit_package(bytes).unwrap();
            (
                decode_model_manifest(package.manifest()).unwrap(),
                package
                    .blobs()
                    .iter()
                    .map(|blob| TritSpoolBlob {
                        kind: blob.kind.clone(),
                        id: blob.id,
                        offset: blob.offset,
                        length: blob.length,
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let limits = TritPackageReadLimits {
            max_source_bytes: bytes.len() as u64,
            max_manifest_bytes: 64 * 1024,
            max_directory_bytes: 64 * 1024,
        };
        let spool_path = root.join(format!("{dense}.spool"));
        fs::write(&spool_path, bytes).unwrap();
        drop(memory); // complete prepared package no longer retained in measured publisher
        let mut spool = File::open(spool_path).unwrap();
        let out = root.join(format!("{dense}.out"));
        let mut sink = DirectoryTensorSink::new(&out);
        REQUESTED.store(0, Ordering::Relaxed);
        TRACK.store(true, Ordering::Relaxed);
        let mut calls = 0;
        let (mut reader, _) = sink
            .publish_spooled_model(
                &manifest,
                &blobs,
                &mut spool,
                limits,
                &prepared.evidence,
                || {
                    calls += 1;
                    if Engine::observe_source(&request, &mut original)? != prepared.source_id {
                        return Err(QuantEngineError::Binding("actual source changed"));
                    }
                    Ok(())
                },
            )
            .unwrap();
        reader.validate_complete_payload_inventory().unwrap();
        reader.verify().unwrap();
        TRACK.store(false, Ordering::Relaxed);
        let requested = REQUESTED.load(Ordering::Relaxed);
        assert_eq!(calls, 1);
        assert_eq!(
            reader.package_id().as_bytes(),
            prepared.package_id.as_bytes()
        );
        assert_eq!(
            reader.model_id("imported").unwrap(),
            prepared.model_ids["imported"]
        );
        let published = fs::read(out.join("model.trit")).unwrap();
        let log = fs::read_to_string(out.join("evidence.jsonl")).unwrap();
        verify_package_with_fit_log(&published, &verify_jsonl(&log).unwrap()).unwrap();
        assert!(!out.join(".pending.trit").exists());
        eprintln!(
            "SPOOLED_DIRECTORY_MEMORY dense_payload={dense} cumulative_heap_requests={requested}"
        );
        assert!(
            requested < 256 * 1024,
            "publisher requested {requested} heap bytes"
        );
        allocations.push(requested);
        drop(reader);
        drop(spool);
    }
    assert!(
        allocations[0].abs_diff(allocations[1]) <= 4096,
        "payload grew publisher heap: {allocations:?}"
    );
    drop(cleanup);
    assert!(!root.exists());
    // Prepared source/plan/log/expected IDs, engine conversion and File construction
    // are excluded. Original in-memory source stays caller-owned. Not aggregate
    // live/peak memory, stack, page cache, device/KV, throughput or real Qwen proof.
}
