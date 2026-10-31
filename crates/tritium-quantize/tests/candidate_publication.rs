//! Real no-overwrite filesystem transaction; not power-loss/security qualification.
use std::{
    fs,
    io::{Read, Write},
    path::PathBuf,
};
use tritium_format::{
    PackageId, TritBlob, TritBlobKind, TritPackageReadLimits, write_trit_package_bounded,
};
use tritium_quantize::engine::{
    CandidatePublicationError, DirectoryTensorSink, UnqualifiedCandidateRecord,
};
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "candidate-publish-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn package() -> Vec<u8> {
    write_trit_package_bounded(
        &[0xa0],
        &[TritBlob {
            kind: TritBlobKind::Asset("payload.bin".into()),
            bytes: b"actual bounded payload",
        }],
        1 << 20,
    )
    .unwrap()
}
fn limits() -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: 1 << 20,
        max_manifest_bytes: 1 << 16,
        max_directory_bytes: 1 << 16,
    }
}
#[test]
fn publishes_actual_identity_without_sealed_evidence_and_never_overwrites() {
    let root = Scratch::new();
    let dir = root.0.join("candidate");
    let bytes = package();
    let id = PackageId::from_package_bytes(&bytes);
    let mut sink = DirectoryTensorSink::new(&dir);
    let (value, warnings) = sink
        .publish_unqualified_candidate(limits(), |f| -> std::io::Result<_> {
            f.write_all(&bytes)?;
            Ok((42, id))
        })
        .unwrap();
    assert_eq!(value, 42);
    assert!(warnings.is_empty());
    assert_eq!(fs::read(dir.join("model.trit")).unwrap(), bytes);
    assert!(!dir.join("evidence.jsonl").exists());
    assert!(!dir.join(".pending.trit").exists());
    let record =
        UnqualifiedCandidateRecord::from_slice(&fs::read(dir.join("candidate.json")).unwrap())
            .unwrap();
    assert_eq!(record.package_id().as_bytes(), id.as_bytes());
    let mut calls = 0;
    assert!(
        sink.publish_unqualified_candidate(limits(), |_| -> std::io::Result<((), PackageId)> {
            calls += 1;
            unreachable!()
        })
        .is_err()
    );
    assert_eq!(calls, 0);
    assert_eq!(fs::read(dir.join("model.trit")).unwrap(), bytes);
}
#[test]
fn failed_partial_invalid_or_mismatched_emission_never_commits() {
    for mode in 0..4 {
        let root = Scratch::new();
        let dir = root.0.join("candidate");
        let mut bytes = package();
        if mode == 2 {
            let parsed = tritium_format::read_trit_package(&bytes).unwrap();
            let offset = parsed.blobs()[0].offset as usize;
            bytes[offset] ^= 1;
        }
        let actual = PackageId::from_package_bytes(&bytes);
        let mut cap = limits();
        if mode == 3 {
            cap.max_source_bytes = 1;
        }
        let result = DirectoryTensorSink::new(&dir).publish_unqualified_candidate(
            cap,
            |f| -> std::io::Result<_> {
                if mode == 0 {
                    f.write_all(b"partial")?;
                    return Err(std::io::Error::other("original emitter refusal"));
                }
                f.write_all(&bytes)?;
                Ok((
                    (),
                    if mode == 1 {
                        PackageId::from_digest([9; 32])
                    } else {
                        actual
                    },
                ))
            },
        );
        match mode {
            0 => assert!(
                matches!(result,Err(CandidatePublicationError::Emission(e)) if e.to_string()=="original emitter refusal")
            ),
            1 => assert!(matches!(
                result,
                Err(CandidatePublicationError::IdentityMismatch { .. })
            )),
            2 => assert!(matches!(
                result,
                Err(CandidatePublicationError::BlobValidation(_)
                    | CandidatePublicationError::Validation(_))
            )),
            _ => assert!(matches!(
                result,
                Err(CandidatePublicationError::Validation(_))
            )),
        };
        assert!(!dir.exists(), "owned pre-commit paths should be cleaned");
    }
}
#[test]
fn cleanup_preserves_injected_foreign_files_and_model_collisions() {
    for collision in [false, true] {
        let root = Scratch::new();
        let dir = root.0.join("candidate");
        let bytes = package();
        let id = PackageId::from_package_bytes(&bytes);
        let result = DirectoryTensorSink::new(&dir).publish_unqualified_candidate(
            limits(),
            |f| -> std::io::Result<_> {
                fs::write(
                    dir.join(if collision {
                        "model.trit"
                    } else {
                        "foreign.txt"
                    }),
                    b"foreign sentinel",
                )?;
                f.write_all(&bytes)?;
                if !collision {
                    return Err(std::io::Error::other("late original guard refusal"));
                }
                Ok(((), id))
            },
        );
        assert!(result.is_err());
        assert!(!dir.join("candidate.json").exists());
        assert!(!dir.join(".pending.trit").exists());
        assert_eq!(
            fs::read(dir.join(if collision {
                "model.trit"
            } else {
                "foreign.txt"
            }))
            .unwrap(),
            b"foreign sentinel"
        );
    }
}
#[test]
fn preexisting_files_directories_and_symlinks_refuse_before_emitter() {
    let root = Scratch::new();
    let file = root.0.join("file");
    fs::write(&file, b"sentinel").unwrap();
    let directory = root.0.join("directory");
    fs::create_dir(&directory).unwrap();
    let paths = [file.clone(), directory.clone()];
    #[cfg(unix)]
    let paths = {
        let link = root.0.join("link");
        std::os::unix::fs::symlink(&directory, &link).unwrap();
        paths.into_iter().chain(std::iter::once(link))
    };
    for path in paths {
        let mut called = false;
        let error = DirectoryTensorSink::new(path)
            .publish_unqualified_candidate(limits(), |_| -> std::io::Result<((), PackageId)> {
                called = true;
                unreachable!()
            })
            .unwrap_err();
        assert!(!called);
        assert!(matches!(error, CandidatePublicationError::Publication(_)));
    }
    assert_eq!(fs::read(file).unwrap(), b"sentinel");
    assert_eq!(fs::read_dir(directory).unwrap().count(), 0);
}

#[test]
fn invalid_receiving_caps_refuse_before_filesystem_or_emitter() {
    let root = Scratch::new();
    for mode in 0..3 {
        let dir = root.0.join(format!("bad-{mode}"));
        let mut caps = limits();
        match mode {
            0 => caps.max_source_bytes = 0,
            1 => caps.max_manifest_bytes = 0,
            _ => caps.max_directory_bytes = 0,
        };
        let mut called = false;
        let result = DirectoryTensorSink::new(&dir).publish_unqualified_candidate(
            caps,
            |_| -> std::io::Result<((), PackageId)> {
                called = true;
                unreachable!()
            },
        );
        assert!(result.is_err());
        assert!(!called);
        assert!(!dir.exists());
    }
}
#[test]
fn record_parser_never_accepts_qualification_or_ambiguous_metadata() {
    let record = serde_json::json!({"schema":"tritium.unqualified-candidate.v1","qualification":false,"package_id":tritium_schema::PackageId::from_bytes([3;32])});
    let encoded = serde_json::to_vec(&record).unwrap();
    UnqualifiedCandidateRecord::from_slice(&encoded).unwrap();
    for mode in 0..4 {
        let mut v = record.clone();
        match mode {
            0 => v["qualification"] = true.into(),
            1 => v["schema"] = "unknown".into(),
            2 => {
                v["package_id"] =
                    serde_json::to_value(tritium_schema::PackageId::from_bytes([0; 32])).unwrap()
            }
            _ => v["unexpected"] = 0.into(),
        };
        assert!(UnqualifiedCandidateRecord::from_slice(&serde_json::to_vec(&v).unwrap()).is_err());
    }
    let duplicate =
        String::from_utf8(encoded)
            .unwrap()
            .replacen('{', "{\"qualification\":false,", 1);
    assert!(UnqualifiedCandidateRecord::from_slice(duplicate.as_bytes()).is_err());
    assert!(UnqualifiedCandidateRecord::from_slice(&vec![b' '; 4097]).is_err());
}
#[test]
fn sealed_metadata_order_and_names_remain_unchanged() {
    let root = Scratch::new();
    let dir = root.0.join("sealed");
    #[derive(serde::Serialize)]
    struct TestStarted {
        purpose: &'static str,
    }
    impl tritium_evidence::EvidenceEvent for TestStarted {
        const SCHEMA_ID: &'static str = "tritium.test.publication_order_started";
        const VERSION: u16 = 1;
    }
    let mut recorder = tritium_evidence::EvidenceRecorder::new(
        "publication-order-test",
        [tritium_evidence::EventRegistration {
            schema: "tritium.test.publication_order_started",
            version: 1,
        }],
    )
    .unwrap();
    recorder
        .emit(
            "test",
            None,
            0,
            &TestStarted {
                purpose: "test metadata ordering, not model fitting or qualification",
            },
        )
        .unwrap();
    let evidence = recorder.seal().unwrap();
    DirectoryTensorSink::new(&dir)
        .publish_with_writer(&evidence, |f| {
            assert_eq!(
                fs::read_to_string(dir.join("evidence.jsonl"))?,
                evidence.jsonl()
            );
            assert!(!dir.join("candidate.json").exists());
            f.write_all(b"unqualified opaque test bytes")?;
            Ok(())
        })
        .unwrap();
    let mut file = fs::File::open(dir.join("model.trit")).unwrap();
    let mut data = vec![];
    file.read_to_end(&mut data).unwrap();
    assert_eq!(data, b"unqualified opaque test bytes");
}
