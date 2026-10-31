//! Exact original-artifact export through the public engine and directory sink.
use super::*;
use std::{
    fs,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
};
use tritium_format::trit_model::{TritModelReadError, TritModelVisitError};
use tritium_format::{PackageId as PhysicalPackageId, TritPackageReadLimits};

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "tritium-stream-export-{name}-{}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        Self(root)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn limits(size: usize) -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: size as u64,
        max_manifest_bytes: size,
        max_directory_bytes: size,
    }
}

#[test]
fn original_streamed_exports_preserve_bytes_ids_and_no_clobber_for_all_precisions() {
    let root = Scratch::new("exact");
    for precision in [
        ScalePrecision::F16,
        ScalePrecision::Bf16,
        ScalePrecision::F32,
    ] {
        let mut original = source(precision);
        let mut memory = MemoryTensorSink::default();
        Engine::run(&plan(&original), &mut original, &mut memory, recorder()).unwrap();
        let (bytes, sealed) = memory.artifact().unwrap();
        let id = PhysicalPackageId::from_package_bytes(bytes);
        let out = root.0.join(format!("{precision:?}"));
        let mut sink = DirectoryTensorSink::new(&out);
        let warnings = sink
            .publish_verified_model_reader(
                Cursor::new(bytes),
                limits(bytes.len()),
                id,
                sealed.jsonl(),
            )
            .unwrap();
        let expected_warnings = if fs::File::open(&out).and_then(|f| f.sync_all()).is_err() {
            vec![PublicationWarning::DirectorySync]
        } else {
            Vec::new()
        };
        assert_eq!(warnings, expected_warnings);
        assert_eq!(fs::read(out.join("model.trit")).unwrap(), bytes);
        assert_eq!(
            fs::read_to_string(out.join("evidence.jsonl")).unwrap(),
            sealed.jsonl()
        );
        assert!(!out.join(".pending.trit").exists());
        verify_package_with_fit_log(&fs::read(out.join("model.trit")).unwrap(), sealed.log())
            .unwrap();
        fs::write(out.join("foreign"), b"retain").unwrap();
        assert!(matches!(
            sink.publish_verified_model_reader(
                Cursor::new(bytes),
                limits(bytes.len()),
                id,
                sealed.jsonl()
            ),
            Err(QuantEngineError::DestinationExists)
        ));
        assert_eq!(fs::read(out.join("foreign")).unwrap(), b"retain");
        assert_eq!(fs::read(out.join("model.trit")).unwrap(), bytes);
    }
}

#[test]
fn streamed_export_rejects_unbound_corrupt_or_over_limit_inputs_before_creating_output() {
    let root = Scratch::new("refused");
    let mut original = source(ScalePrecision::F32);
    let mut memory = MemoryTensorSink::default();
    Engine::run(&plan(&original), &mut original, &mut memory, recorder()).unwrap();
    let (bytes, sealed) = memory.artifact().unwrap();
    let id = PhysicalPackageId::from_package_bytes(bytes);
    let mut other_source = source(ScalePrecision::F32);
    other_source.tensors.get_mut("weight").unwrap().bytes[..4]
        .copy_from_slice(&5.0f32.to_le_bytes());
    let mut other = MemoryTensorSink::default();
    Engine::run(
        &plan(&other_source),
        &mut other_source,
        &mut other,
        recorder(),
    )
    .unwrap();
    for variant in 0..6 {
        let out = root.0.join(format!("refusal-{variant}"));
        let mut payload = bytes.to_vec();
        let mut bound = limits(bytes.len());
        let mut expected = id;
        let mut log = sealed.jsonl().to_owned();
        match variant {
            0 => payload[256] ^= 1,
            1 => log = "not jsonl".into(),
            2 => log = format!(" {}", sealed.jsonl()),
            3 => log = other.artifact().unwrap().1.jsonl().to_owned(),
            4 => expected = PhysicalPackageId::from_digest([7; 32]),
            _ => bound.max_source_bytes -= 1,
        }
        assert!(
            DirectoryTensorSink::new(&out)
                .publish_verified_model_reader(Cursor::new(payload), bound, expected, &log)
                .is_err()
        );
        assert!(!out.exists());
    }
}

struct FailingSource {
    bytes: Cursor<Vec<u8>>,
    destination: PathBuf,
    mode: u8,
    changed: bool,
}
impl Read for FailingSource {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        assert!(output.len() <= 65536);
        if self.destination.exists() && self.mode == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "stream export read sentinel",
            ));
        }
        let count = self.bytes.read(output)?;
        // Mutate only after admission and after returning original bytes to the
        // visitor's current buffer. No internal read-count/order is assumed.
        if self.destination.exists() && !self.changed {
            self.changed = true;
            match self.mode {
                1 => self.bytes.get_mut()[256] ^= 1,
                2 => self.bytes.get_mut().push(0),
                _ => {
                    self.bytes.get_mut().pop();
                }
            }
        }
        Ok(count)
    }
}
impl Seek for FailingSource {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.bytes.seek(position)
    }
}

#[test]
fn streamed_export_cleans_private_effects_after_original_io_mutation_growth_or_truncation() {
    let root = Scratch::new("precommit");
    let mut original = source(ScalePrecision::F32);
    let mut memory = MemoryTensorSink::default();
    Engine::run(&plan(&original), &mut original, &mut memory, recorder()).unwrap();
    let (bytes, sealed) = memory.artifact().unwrap();
    let id = PhysicalPackageId::from_package_bytes(bytes);
    for mode in 0..4 {
        let out = root.0.join(format!("fault-{mode}"));
        let input = FailingSource {
            bytes: Cursor::new(bytes.to_vec()),
            destination: out.clone(),
            mode,
            changed: false,
        };
        let result = DirectoryTensorSink::new(&out).publish_verified_model_reader(
            input,
            limits(bytes.len()),
            id,
            sealed.jsonl(),
        );
        assert!(result.is_err());
        assert!(!out.exists());
        if mode == 0 {
            match result {
                Err(QuantEngineError::ModelVisit(TritModelVisitError::Read(
                    TritModelReadError::Source(tritium_format::TritPackageReadError::Io {
                        source,
                        ..
                    }),
                ))) => {
                    assert_eq!(source.kind(), std::io::ErrorKind::PermissionDenied);
                    assert_eq!(source.to_string(), "stream export read sentinel");
                }
                other => panic!("original typed source failure lost: {other:?}"),
            }
        }
    }
}
