//! Public bounded staging -> existing directory publisher with actual fit evidence.
use super::*;
use std::cell::Cell;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use tritium_format::trit_model::{TritModelStageError, decode_model_manifest};
use tritium_format::{TritPackageReadLimits, TritPackageReader, TritSpoolBlob};
use tritium_quantize::engine::TritModelPublicationError;

struct Scratch(std::path::PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "tritium-spool-publish-{name}-{}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn receiving() -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: 1_000_000,
        max_manifest_bytes: 64 * 1024,
        max_directory_bytes: 64 * 1024,
    }
}
fn inventory(bytes: &[u8]) -> (ModelManifest, Vec<TritSpoolBlob>) {
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
            .collect(),
    )
}

#[test]
fn actual_fitted_model_and_original_log_publish_from_file_spool_and_replay_exactly() {
    let root = Scratch::new("success");
    for precision in [
        ScalePrecision::F32,
        ScalePrecision::F16,
        ScalePrecision::Bf16,
    ] {
        let mut original = source(precision);
        let request = plan(&original);
        let mut memory = MemoryTensorSink::default();
        let fitted = Engine::run(&request, &mut original, &mut memory, recorder()).unwrap();
        let (bytes, _) = memory.artifact().unwrap();
        let (manifest, blobs) = inventory(bytes);
        let spool_path = root.0.join(format!("{precision:?}.spool"));
        fs::write(&spool_path, bytes).unwrap();
        let mut spool = File::open(spool_path).unwrap();
        let out = root.0.join(format!("{precision:?}"));
        let mut sink = DirectoryTensorSink::new(&out);
        let calls = Cell::new(0);
        let (mut reader, warnings) = sink
            .publish_spooled_model(
                &manifest,
                &blobs,
                &mut spool,
                receiving(),
                &fitted.evidence,
                || {
                    calls.set(calls.get() + 1);
                    if Engine::observe_source(&request, &mut original)? != fitted.source_id {
                        return Err(QuantEngineError::Binding(
                            "original source changed after fit",
                        ));
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(calls.get(), 1);
        let expected_warnings = if File::open(&out).and_then(|file| file.sync_all()).is_err() {
            vec![PublicationWarning::DirectorySync]
        } else {
            Vec::new()
        };
        assert_eq!(warnings, expected_warnings);
        assert!(!out.join(".pending.trit").exists());
        assert_eq!(reader.manifest(), &manifest);
        assert_eq!(reader.package_id().as_bytes(), fitted.package_id.as_bytes());
        for (level, id) in &fitted.model_ids {
            assert_eq!(reader.model_id(level).unwrap(), *id);
        }
        reader.validate_complete_payload_inventory().unwrap();
        reader.verify().unwrap();
        let published = fs::read(out.join("model.trit")).unwrap();
        let log = fs::read_to_string(out.join("evidence.jsonl")).unwrap();
        assert_eq!(published, bytes);
        assert_eq!(log, fitted.evidence.jsonl());
        let verified = verify_jsonl(&log).unwrap();
        verify_package_with_fit_log(&published, &verified).unwrap();
        let mut reopened =
            TritPackageReader::open(File::open(out.join("model.trit")).unwrap(), receiving())
                .unwrap();
        assert_eq!(reopened.package_id(), reader.package_id());
        reopened.verify().unwrap();
        // A second request refuses before source callbacks and leaves both files intact.
        assert!(matches!(
            sink.publish_spooled_model(
                &manifest,
                &blobs,
                &mut spool,
                receiving(),
                &fitted.evidence,
                || -> Result<(), u8> { panic!("no callback for existing destination") },
            ),
            Err(TritModelPublicationError::Publication(
                QuantEngineError::DestinationExists
            ))
        ));
        assert_eq!(fs::read(out.join("model.trit")).unwrap(), published);
        assert_eq!(fs::read_to_string(out.join("evidence.jsonl")).unwrap(), log);
        drop(reopened);
        drop(reader);
        drop(spool);
    }
}

#[test]
fn all_typed_staging_refusals_clean_owned_files_without_running_source_guard() {
    let root = Scratch::new("refusals");
    let mut original = source(ScalePrecision::F32);
    let request = plan(&original);
    let mut memory = MemoryTensorSink::default();
    let fitted = Engine::run(&request, &mut original, &mut memory, recorder()).unwrap();
    let (bytes, _) = memory.artifact().unwrap();
    let (manifest, blobs) = inventory(bytes);
    for refusal in 0..7 {
        let mut metadata = manifest.clone();
        let mut entries = blobs.clone();
        let mut limits = receiving();
        match refusal {
            0 => entries[0].id = BlobId::from_bytes([7; 32]),
            1 => metadata.levels[0].tensors[0].blob.id = BlobId::from_bytes([9; 32]),
            2 => {
                let mut extra = entries[0].clone();
                extra.kind = TritBlobKind::Asset("unreferenced.bin".into());
                entries.push(extra);
            }
            3 => limits.max_source_bytes = bytes.len() as u64 - 1,
            4 => limits.max_manifest_bytes = 0,
            5 => limits.max_directory_bytes = 0,
            _ => {
                let parent = metadata.levels[0].name.clone();
                metadata.levels.last_mut().unwrap().fit =
                    LevelFit::ImportedPrefixOf { level: parent };
            }
        }
        let out = root.0.join(format!("refusal-{refusal}"));
        let mut sink = DirectoryTensorSink::new(&out);
        assert!(matches!(
            sink.publish_spooled_model(
                &metadata,
                &entries,
                &mut Cursor::new(bytes),
                limits,
                &fitted.evidence,
                || -> Result<(), u8> { panic!("no source guard before typed admission") },
            ),
            Err(TritModelPublicationError::Staging(_))
        ));
        assert!(!out.exists(), "no owned output after refusal {refusal}");
    }
}

#[test]
fn actual_original_source_change_and_original_guard_errors_never_publish() {
    let root = Scratch::new("source");
    let mut original = source(ScalePrecision::F32);
    let request = plan(&original);
    let mut memory = MemoryTensorSink::default();
    let fitted = Engine::run(&request, &mut original, &mut memory, recorder()).unwrap();
    let (bytes, _) = memory.artifact().unwrap();
    let (manifest, blobs) = inventory(bytes);
    original.tensors.get_mut("weight").unwrap().bytes[..4].copy_from_slice(&3.0f32.to_le_bytes());
    let out = root.0.join("changed");
    let mut sink = DirectoryTensorSink::new(&out);
    let calls = Cell::new(0);
    let result = sink.publish_spooled_model(
        &manifest,
        &blobs,
        &mut Cursor::new(bytes),
        receiving(),
        &fitted.evidence,
        || {
            calls.set(calls.get() + 1);
            if Engine::observe_source(&request, &mut original)? != fitted.source_id {
                return Err(QuantEngineError::Binding(
                    "original source changed after fit",
                ));
            }
            Ok(())
        },
    );
    assert!(matches!(
        result,
        Err(TritModelPublicationError::Staging(
            TritModelStageError::Guard(QuantEngineError::Binding(
                "original source changed after fit"
            ))
        ))
    ));
    assert_eq!(calls.get(), 1);
    assert!(!out.exists());
    let out = root.0.join("guard-error");
    let mut sink = DirectoryTensorSink::new(&out);
    assert!(matches!(
        sink.publish_spooled_model(
            &manifest,
            &blobs,
            &mut Cursor::new(bytes),
            receiving(),
            &fitted.evidence,
            || Err(37u8),
        ),
        Err(TritModelPublicationError::Staging(
            TritModelStageError::Guard(37)
        ))
    ));
    assert!(!out.exists());
}

#[test]
fn late_output_mutation_commit_collision_and_foreign_injection_preserve_owned_scope() {
    let root = Scratch::new("late");
    let mut original = source(ScalePrecision::F32);
    let request = plan(&original);
    let mut memory = MemoryTensorSink::default();
    let fitted = Engine::run(&request, &mut original, &mut memory, recorder()).unwrap();
    let (bytes, _) = memory.artifact().unwrap();
    let (manifest, blobs) = inventory(bytes);
    for failure in 0..3 {
        let out = root.0.join(format!("late-{failure}"));
        let mut sink = DirectoryTensorSink::new(&out);
        let result = sink.publish_spooled_model(
            &manifest,
            &blobs,
            &mut Cursor::new(bytes),
            receiving(),
            &fitted.evidence,
            || -> Result<(), u8> {
                match failure {
                    0 => {
                        let mut pending = fs::OpenOptions::new()
                            .read(true)
                            .write(true)
                            .open(out.join(".pending.trit"))
                            .unwrap();
                        pending.seek(SeekFrom::Start(256)).unwrap();
                        let mut byte = [0];
                        pending.read_exact(&mut byte).unwrap();
                        byte[0] ^= 1;
                        pending.seek(SeekFrom::Start(256)).unwrap();
                        pending.write_all(&byte).unwrap();
                        Ok(())
                    }
                    1 => {
                        fs::write(out.join("model.trit"), b"foreign model sentinel").unwrap();
                        Ok(())
                    }
                    _ => {
                        fs::write(out.join("foreign.txt"), b"preserve me").unwrap();
                        Err(41)
                    }
                }
            },
        );
        match failure {
            0 => {
                assert!(matches!(
                    result,
                    Err(TritModelPublicationError::Staging(
                        TritModelStageError::Read(_)
                    ))
                ));
                assert!(!out.exists());
            }
            1 => {
                assert!(
                    matches!(result, Err(TritModelPublicationError::Publication(QuantEngineError::Io(ref error))) if error.kind() == std::io::ErrorKind::AlreadyExists)
                );
                assert_eq!(
                    fs::read(out.join("model.trit")).unwrap(),
                    b"foreign model sentinel"
                );
            }
            _ => {
                assert!(matches!(
                    result,
                    Err(TritModelPublicationError::Staging(
                        TritModelStageError::Guard(41)
                    ))
                ));
                assert_eq!(fs::read(out.join("foreign.txt")).unwrap(), b"preserve me");
                assert!(!out.join("model.trit").exists());
            }
        }
        assert!(!out.join("evidence.jsonl").exists());
        assert!(!out.join(".pending.trit").exists());
    }
}

#[test]
fn existing_writer_keeps_write_only_pending_handle_and_original_callback_errors() {
    let root = Scratch::new("ordinary");
    let mut original = source(ScalePrecision::F32);
    let request = plan(&original);
    let mut memory = MemoryTensorSink::default();
    let fitted = Engine::run(&request, &mut original, &mut memory, recorder()).unwrap();
    let (bytes, _) = memory.artifact().unwrap();
    let out = root.0.join("ordinary");
    let mut sink = DirectoryTensorSink::new(&out);
    let ((), _) = sink
        .publish_with_writer(&fitted.evidence, |file| {
            file.write_all(bytes)?;
            file.seek(SeekFrom::Start(0))?;
            assert!(
                file.read(&mut [0]).is_err(),
                "existing callback remains write-only"
            );
            Ok(())
        })
        .unwrap();
    assert_eq!(fs::read(out.join("model.trit")).unwrap(), bytes);
    let out = root.0.join("refused");
    let mut sink = DirectoryTensorSink::new(&out);
    let result: Result<((), _), _> = sink.publish_with_writer(&fitted.evidence, |file| {
        file.write_all(b"partial")?;
        Err(QuantEngineError::Binding("original emitter error"))
    });
    assert!(matches!(
        result,
        Err(QuantEngineError::Binding("original emitter error"))
    ));
    assert!(!out.exists());
}
