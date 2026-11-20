//! Explicit File-backed caller-chain I/O diagnostic; not model-quality or speed evidence.
use super::*;
use tritium_nn::{ModelLoadResources, StreamLimits, inspect_trit_file_streamed};

#[test]
#[ignore = "explicit CPU File/read-syscall diagnostic; trace with strace -f -y -s 0"]
fn actual_public_streamed_load_read_io() {
    read_io(true);
}

#[test]
#[ignore = "explicit CPU inspected-identity caller/read-syscall diagnostic"]
fn inspected_identity_public_streamed_load_read_io() {
    read_io(false);
}

fn read_io(preliminary_reopen: bool) {
    let began = std::time::Instant::now();
    let root = Scratch::new("load-read-io");
    let mut package = text_package(true, false);
    // An unused, referenced asset separates full-source scans from tiny graph
    // payload work. It is synthetic, not a real checkpoint or hidden tensor.
    let padding = 1024 * 1024;
    asset(&mut package, "read-count-padding.bin", vec![0x5a; padding]);
    let bytes = package.bytes();
    let prepared_seconds = began.elapsed().as_secs_f64();
    let receiving = TritPackageReadLimits {
        max_source_bytes: bytes.len() as u64,
        ..receiving()
    };
    let paths = ["reader.trit", "inspection.trit", "load.trit"].map(|name| root.0.join(name));
    for path in &paths {
        std::fs::write(path, &bytes).unwrap();
    }
    let wrote_seconds = began.elapsed().as_secs_f64();
    // Fixture bytes supply the expected identity, just as the real replay's
    // export/verification evidence does. Never perform this borrowed materialization
    // for the real model. Compare legacy extra admission against using the actual
    // pure inspection's terminally verified identity before backend initialization.
    let package_id = tritium_format::read_trit_package(&bytes)
        .unwrap()
        .package_id();
    if preliminary_reopen {
        let mut reader = TritModelReader::open(File::open(&paths[0]).unwrap(), receiving).unwrap();
        reader.validate_complete_payload_inventory().unwrap();
        assert_eq!(reader.package_id(), package_id);
        reader.verify().unwrap();
    }
    let reader_seconds = began.elapsed().as_secs_f64();
    let stream = StreamLimits {
        max_context: 16,
        max_segment_tokens: 4,
        max_host_cache_payload_bytes: 1_000_000,
    };
    let plan = inspect_trit_file_streamed(
        &paths[1],
        "typed",
        receiving,
        limits(),
        stream,
        TensorUploadPolicy::default(),
        |metadata, encoding| tritium_cpu::CpuBackend.tensor_stream_plan(metadata, encoding),
    )
    .unwrap();
    match plan.source_plan().source() {
        LoadedSourceIdentity::Trit {
            package_id: observed,
            level,
            ..
        } => {
            assert_eq!(observed.as_bytes(), package_id.as_bytes());
            assert_eq!(level, "typed");
        }
        _ => panic!("inspection must bind the exact typed source"),
    }
    let inspection_seconds = began.elapsed().as_secs_f64();
    let ledgers: [TensorPayloadBudget; 5] =
        std::array::from_fn(|_| TensorPayloadBudget::new(1_000_000));
    let mut evidence = recorder();
    let mut loaded = tritium_nn::load_streamed_with_resources(
        ModelSource::TritFile {
            path: &paths[2],
            level: "typed",
            receiving,
            limits: limits(),
        },
        Box::new(tritium_cpu::CpuBackend),
        TensorUploadPolicy::default(),
        ModelLoadResources {
            payload: &ledgers[0],
            source_workspace: &ledgers[1],
            host_workspace: &ledgers[2],
            device_workspace: &ledgers[3],
            retained_host: &ledgers[4],
        },
        stream,
        TensorUploadContext {
            recorder: &mut evidence,
            span: "load",
            parent: None,
            logical_time: 0,
        },
    )
    .unwrap();
    let loaded_seconds = began.elapsed().as_secs_f64();
    assert_eq!(
        plan.source_plan().source(),
        &loaded.receipt.observation().source
    );
    assert!(
        loaded
            .runner
            .prefill(&[1, 2])
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    );
    drop(loaded);
    assert!(ledgers.iter().all(|b| b.used_bytes() == 0));
    eprintln!(
        "LOAD_READ_IO source_bytes={} padding_bytes={padding} package_id={package_id} preliminary_reopen={preliminary_reopen} root={}",
        bytes.len(),
        root.0.display()
    );
    eprintln!(
        "LOAD_READ_IO_PHASES fixture_seconds={prepared_seconds} writes_seconds={} reader_seconds={} inspection_seconds={} load_seconds={} finish_seconds={}",
        wrote_seconds - prepared_seconds,
        reader_seconds - wrote_seconds,
        inspection_seconds - reader_seconds,
        loaded_seconds - inspection_seconds,
        began.elapsed().as_secs_f64() - loaded_seconds
    );
}
