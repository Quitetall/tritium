//! The same complete actual conversion is admitted through both reader adapters.
use super::*;
use std::{
    cell::{Cell, RefCell},
    io::{Read, Seek, SeekFrom},
    rc::Rc,
};
use tritium_format::trit_model::TritModelReadError;
use tritium_format::{TritPackageReadError, TritPackageReadLimits};
use tritium_quantize::engine::open_model_reader_with_fit_log;

fn limits(size: usize) -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: size as u64,
        max_manifest_bytes: size,
        max_directory_bytes: size,
    }
}

#[test]
fn seek_reader_matches_complete_byte_admission_for_all_source_precisions() {
    for precision in [
        ScalePrecision::F16,
        ScalePrecision::Bf16,
        ScalePrecision::F32,
    ] {
        let mut source = source(precision);
        let request = plan(&source);
        let mut sink = MemoryTensorSink::default();
        Engine::run(&request, &mut source, &mut sink, recorder()).unwrap();
        let (package, evidence) = sink.artifact().unwrap();
        verify_package_with_fit_log(package, evidence.log()).unwrap();
        let expected = read_trit_model(package).unwrap();
        let mut actual = open_model_reader_with_fit_log(
            Cursor::new(package),
            limits(package.len()),
            evidence.log(),
        )
        .unwrap();
        assert_eq!(actual.package_id(), expected.package_id());
        assert_eq!(actual.manifest(), expected.manifest());
        for level in &expected.manifest().levels {
            assert_eq!(
                actual.model_id(&level.name).unwrap(),
                expected.model_id(&level.name).unwrap()
            );
            for tensor in &level.tensors {
                assert_eq!(
                    actual.tensor_digest(&level.name, &tensor.name).unwrap(),
                    expected.tensor_digest(&level.name, &tensor.name).unwrap()
                );
            }
        }
        actual.verify().unwrap();
    }
}

#[test]
fn seek_reader_rejects_corrupt_packages_and_changed_or_unrelated_logs() {
    let mut source = source(ScalePrecision::F32);
    let request = plan(&source);
    let mut sink = MemoryTensorSink::default();
    Engine::run(&request, &mut source, &mut sink, recorder()).unwrap();
    let (package, evidence) = sink.artifact().unwrap();
    let mut corrupt = package.to_vec();
    corrupt[256] ^= 1;
    assert!(
        open_model_reader_with_fit_log(
            Cursor::new(&corrupt),
            limits(corrupt.len()),
            evidence.log()
        )
        .is_err()
    );
    let mut changed = evidence.log().clone();
    changed.root_digest.replace_range(..1, "z");
    assert!(
        open_model_reader_with_fit_log(Cursor::new(package), limits(package.len()), &changed)
            .is_err()
    );
    source.tensors.get_mut("weight").unwrap().bytes[..4].copy_from_slice(&3.0f32.to_le_bytes());
    let mut other = MemoryTensorSink::default();
    let other_result = Engine::run(&request, &mut source, &mut other, recorder()).unwrap();
    assert!(
        open_model_reader_with_fit_log(
            Cursor::new(package),
            limits(package.len()),
            other_result.evidence.log()
        )
        .is_err()
    );
    let mut too_small = limits(package.len());
    too_small.max_source_bytes -= 1;
    assert!(
        open_model_reader_with_fit_log(Cursor::new(package), too_small, evidence.log()).is_err()
    );
}

#[derive(Debug)]
struct SharedSource {
    bytes: Rc<RefCell<Cursor<Vec<u8>>>>,
    fail: Rc<Cell<bool>>,
}
impl Read for SharedSource {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        if self.fail.get() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "seek admission sentinel",
            ));
        }
        self.bytes.borrow_mut().read(output)
    }
}
impl Seek for SharedSource {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.bytes.borrow_mut().seek(position)
    }
}

#[test]
fn seek_reader_keeps_mutation_checks_and_original_io_errors() {
    let mut source = source(ScalePrecision::F32);
    let request = plan(&source);
    let mut sink = MemoryTensorSink::default();
    Engine::run(&request, &mut source, &mut sink, recorder()).unwrap();
    let (package, evidence) = sink.artifact().unwrap();
    let bytes = Rc::new(RefCell::new(Cursor::new(package.to_vec())));
    let fail = Rc::new(Cell::new(false));
    let mut reader = open_model_reader_with_fit_log(
        SharedSource {
            bytes: bytes.clone(),
            fail: fail.clone(),
        },
        limits(package.len()),
        evidence.log(),
    )
    .unwrap();
    bytes.borrow_mut().get_mut()[256] ^= 1;
    assert!(reader.verify().is_err());
    fail.set(true);
    let error = open_model_reader_with_fit_log(
        SharedSource { bytes, fail },
        limits(package.len()),
        evidence.log(),
    )
    .unwrap_err();
    match error {
        QuantEngineError::ModelRead(TritModelReadError::Source(TritPackageReadError::Io {
            source,
            ..
        })) => {
            assert_eq!(source.kind(), std::io::ErrorKind::PermissionDenied);
            assert_eq!(source.to_string(), "seek admission sentinel");
        }
        other => panic!("original typed I/O error lost: {other:?}"),
    }
}
