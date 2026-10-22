//! Common companion transaction, not checkpoint semantics or release qualification.
use super::*;
use std::{
    cell::Cell,
    fs,
    io::Read,
    path::{Path, PathBuf},
};
use tritium_format::trit_model::{TritModelStageError, decode_model_manifest};
use tritium_format::{TritPackageReadLimits, TritSpoolBlob, inspect_trit_spool_layout};
use tritium_quantize::engine::{
    TritModelCompanion, TritModelCompanions, TritModelPublicationError, TritModelSpool,
};

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let p =
            std::env::temp_dir().join(format!("tritium-companions-{name}-{}", std::process::id()));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn id(b: &[u8]) -> BlobId {
    BlobId::from_bytes(*blake3::hash(b).as_bytes())
}
fn fixture() -> (Vec<u8>, tritium_evidence::SealedEvidence) {
    let mut source = source(ScalePrecision::F32);
    let mut sink = MemoryTensorSink::default();
    let fit = Engine::run(&plan(&source), &mut source, &mut sink, recorder()).unwrap();
    (sink.artifact().unwrap().0.to_vec(), fit.evidence)
}
fn inventory(b: &[u8]) -> (ModelManifest, Vec<TritSpoolBlob>) {
    let p = read_trit_package(b).unwrap();
    (
        decode_model_manifest(p.manifest()).unwrap(),
        p.blobs()
            .iter()
            .map(|e| TritSpoolBlob {
                kind: e.kind.clone(),
                id: e.id,
                offset: e.offset,
                length: e.length,
            })
            .collect(),
    )
}
fn limits() -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: 1_000_000,
        max_manifest_bytes: 100_000,
        max_directory_bytes: 100_000,
    }
}
fn publish<E>(
    out: &Path,
    bytes: &[u8],
    evidence: &tritium_evidence::SealedEvidence,
    companions: TritModelCompanions<'_>,
    receiving: TritPackageReadLimits,
    guard: impl FnOnce() -> Result<(), E>,
) -> Result<
    (
        tritium_format::trit_model::TritModelReader<fs::File>,
        Vec<PublicationWarning>,
    ),
    TritModelPublicationError<E>,
> {
    let (manifest, blobs) = inventory(bytes);
    DirectoryTensorSink::new(out)
        .publish_spooled_model_with_companions(
            TritModelSpool {
                manifest: &manifest,
                blobs: &blobs,
                source: &mut Cursor::new(bytes),
                limits: receiving,
            },
            evidence,
            companions,
            |_| guard(),
        )
        .map(|(model, (), warnings)| (model, warnings))
}

#[test]
fn exact_streamed_companions_commit_with_identical_model_and_log() {
    let root = Scratch::new("success");
    let out = root.0.join("out");
    let (bytes, evidence) = fixture();
    let payload = vec![17; 33000];
    let largest = Cell::new(0);
    struct Bounded<'a> {
        data: Cursor<&'a [u8]>,
        largest: &'a Cell<usize>,
    }
    impl Read for Bounded<'_> {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            assert!(b.len() <= 8192);
            self.largest.set(self.largest.get().max(b.len()));
            self.data.read(b)
        }
    }
    let mut first = Cursor::new(b"opaque checkpoint".as_slice());
    let mut second = Bounded {
        data: Cursor::new(&payload),
        largest: &largest,
    };
    let companions = TritModelCompanions {
        files: vec![
            TritModelCompanion {
                name: "checkpoint.tnsr",
                source: &mut first,
                bytes: 17,
                content: id(b"opaque checkpoint"),
            },
            TritModelCompanion {
                name: "gradients.tnsj",
                source: &mut second,
                bytes: payload.len() as u64,
                content: id(&payload),
            },
        ],
        max_bytes: 33017,
    };
    let (mut reader, _) = publish(&out, &bytes, &evidence, companions, limits(), || {
        assert!(out.join("checkpoint.tnsr").exists());
        assert!(!out.join("model.trit").exists());
        Ok::<_, u8>(())
    })
    .unwrap();
    reader.verify().unwrap();
    assert_eq!(largest.get(), 8192);
    assert_eq!(fs::read(out.join("model.trit")).unwrap(), bytes);
    assert_eq!(
        fs::read_to_string(out.join("evidence.jsonl")).unwrap(),
        evidence.jsonl()
    );
    assert_eq!(
        fs::read(out.join("checkpoint.tnsr")).unwrap(),
        b"opaque checkpoint"
    );
    assert_eq!(fs::read(out.join("gradients.tnsj")).unwrap(), payload);
    assert_eq!(fs::read_dir(out).unwrap().count(), 4);
}

struct NoRead;
impl Read for NoRead {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        panic!("companion must not be read before admission")
    }
}
#[test]
fn invalid_names_duplicate_reserved_and_byte_caps_refuse_before_destination_or_reads() {
    let root = Scratch::new("names");
    let (bytes, evidence) = fixture();
    for (i, name) in [
        "",
        ".",
        "..",
        "a/b",
        "a\\b",
        "Upper.bin",
        ".hidden",
        "a:",
        "bad ",
        "model.trit",
        "evidence.jsonl",
        "con",
        "aux.bin",
        "com1.dat",
        "lpt9",
        "end.",
    ]
    .iter()
    .enumerate()
    {
        let out = root.0.join(format!("bad-{i}"));
        let mut source = NoRead;
        assert!(
            publish(
                &out,
                &bytes,
                &evidence,
                TritModelCompanions {
                    files: vec![TritModelCompanion {
                        name,
                        source: &mut source,
                        bytes: 1,
                        content: id(b"x")
                    }],
                    max_bytes: 1
                },
                limits(),
                || Ok::<_, u8>(())
            )
            .is_err()
        );
        assert!(!out.exists());
    }
    for mode in 0..3 {
        let out = root.0.join(format!("cap-{mode}"));
        let mut a = NoRead;
        let mut b = NoRead;
        let mut files = vec![TritModelCompanion {
            name: "one.bin",
            source: &mut a,
            bytes: 2,
            content: id(b"xx"),
        }];
        if mode == 1 {
            files.push(TritModelCompanion {
                name: "one.bin",
                source: &mut b,
                bytes: 2,
                content: id(b"xx"),
            });
        } else if mode == 2 {
            files[0].bytes = u64::MAX;
            files.push(TritModelCompanion {
                name: "two.bin",
                source: &mut b,
                bytes: 2,
                content: id(b"xx"),
            });
        }
        assert!(
            publish(
                &out,
                &bytes,
                &evidence,
                TritModelCompanions {
                    files,
                    max_bytes: 1
                },
                limits(),
                || Ok::<_, u8>(())
            )
            .is_err()
        );
        assert!(!out.exists());
    }
}

#[test]
fn exact_container_caps_are_known_before_companion_copy_and_match_encoder() {
    let root = Scratch::new("container-caps");
    let (bytes, evidence) = fixture();
    let (manifest, blobs) = inventory(&bytes);
    let size = inspect_trit_spool_layout(
        &tritium_format::trit_model::encode_model_manifest(&manifest).unwrap(),
        &blobs,
    )
    .unwrap();
    assert_eq!(size.package_bytes, bytes.len() as u64);
    for mode in 0..3 {
        let out = root.0.join(format!("bad-{mode}"));
        let mut source = NoRead;
        let mut cap = limits();
        match mode {
            0 => cap.max_source_bytes = size.package_bytes - 1,
            1 => cap.max_manifest_bytes = size.manifest_bytes as usize - 1,
            _ => cap.max_directory_bytes = size.directory_bytes as usize - 1,
        }
        assert!(matches!(
            publish(
                &out,
                &bytes,
                &evidence,
                TritModelCompanions {
                    files: vec![TritModelCompanion {
                        name: "data.bin",
                        source: &mut source,
                        bytes: 1,
                        content: id(b"x")
                    }],
                    max_bytes: 1
                },
                cap,
                || Ok::<_, u8>(())
            ),
            Err(TritModelPublicationError::Staging(_))
        ));
        assert!(!out.exists());
    }
    let out = root.0.join("exact");
    let mut payload = Cursor::new(b"x");
    publish(
        &out,
        &bytes,
        &evidence,
        TritModelCompanions {
            files: vec![TritModelCompanion {
                name: "data.bin",
                source: &mut payload,
                bytes: 1,
                content: id(b"x"),
            }],
            max_bytes: 1,
        },
        TritPackageReadLimits {
            max_source_bytes: size.package_bytes,
            max_manifest_bytes: size.manifest_bytes as usize,
            max_directory_bytes: size.directory_bytes as usize,
        },
        || Ok::<_, u8>(()),
    )
    .unwrap();
}

#[test]
fn wrong_hash_truncation_trailing_and_guard_failure_remove_only_owned_files() {
    let root = Scratch::new("refusals");
    let (bytes, evidence) = fixture();
    for mode in 0..4 {
        let out = root.0.join(format!("bad-{mode}"));
        let mut source = Cursor::new(b"abc".as_slice());
        let result = publish(
            &out,
            &bytes,
            &evidence,
            TritModelCompanions {
                files: vec![TritModelCompanion {
                    name: "data.bin",
                    source: &mut source,
                    bytes: if mode == 1 {
                        4
                    } else if mode == 2 {
                        2
                    } else {
                        3
                    },
                    content: if mode == 0 { id(b"bad") } else { id(b"abc") },
                }],
                max_bytes: 4,
            },
            limits(),
            || if mode == 3 { Err(37u8) } else { Ok(()) },
        );
        if mode == 3 {
            assert!(matches!(
                result,
                Err(TritModelPublicationError::Staging(
                    TritModelStageError::Guard(37)
                ))
            ));
        } else {
            assert!(result.is_err());
        }
        assert!(
            !out.exists(),
            "owned-only directory must be cleaned for mode{mode}"
        );
    }
}

#[test]
fn companion_mutation_after_source_guard_is_refused_before_commit() {
    let root = Scratch::new("mutation");
    let out = root.0.join("out");
    let (bytes, evidence) = fixture();
    let mut source = Cursor::new(b"abc");
    assert!(
        publish(
            &out,
            &bytes,
            &evidence,
            TritModelCompanions {
                files: vec![TritModelCompanion {
                    name: "data.bin",
                    source: &mut source,
                    bytes: 3,
                    content: id(b"abc")
                }],
                max_bytes: 3
            },
            limits(),
            || {
                fs::write(out.join("data.bin"), b"xyz").unwrap();
                Ok::<_, u8>(())
            }
        )
        .is_err()
    );
    assert!(!out.exists());
}

#[test]
fn existing_destination_refuses_before_companion_read_and_preserves_every_file() {
    let root = Scratch::new("no-clobber");
    let out = root.0.join("out");
    fs::create_dir(&out).unwrap();
    fs::write(out.join("user.txt"), b"keep").unwrap();
    let (bytes, evidence) = fixture();
    let mut source = NoRead;
    assert!(matches!(
        publish(
            &out,
            &bytes,
            &evidence,
            TritModelCompanions {
                files: vec![TritModelCompanion {
                    name: "data.bin",
                    source: &mut source,
                    bytes: 1,
                    content: id(b"x")
                }],
                max_bytes: 1
            },
            limits(),
            || Ok::<_, u8>(())
        ),
        Err(TritModelPublicationError::Publication(
            QuantEngineError::DestinationExists
        ))
    ));
    assert_eq!(fs::read(out.join("user.txt")).unwrap(), b"keep");
    assert_eq!(fs::read_dir(out).unwrap().count(), 1);
}

#[test]
fn original_read_error_and_injected_foreign_companion_path_are_preserved() {
    let root = Scratch::new("foreign");
    let (bytes, evidence) = fixture();
    struct Inject {
        data: Cursor<Vec<u8>>,
        path: PathBuf,
        fail: bool,
        done: bool,
    }
    impl Read for Inject {
        fn read(&mut self, b: &mut [u8]) -> std::io::Result<usize> {
            if !self.done {
                self.done = true;
                fs::write(&self.path, b"foreign-owned").unwrap();
            }
            if self.fail {
                return Err(std::io::Error::other("original injected companion read"));
            }
            self.data.read(b)
        }
    }
    for fail in [true, false] {
        let out = root.0.join(format!("{fail}"));
        let mut first = Inject {
            data: Cursor::new(b"abc".to_vec()),
            path: out.join("second.bin"),
            fail,
            done: false,
        };
        let mut second = Cursor::new(b"def");
        let error = publish(
            &out,
            &bytes,
            &evidence,
            TritModelCompanions {
                files: vec![
                    TritModelCompanion {
                        name: "first.bin",
                        source: &mut first,
                        bytes: 3,
                        content: id(b"abc"),
                    },
                    TritModelCompanion {
                        name: "second.bin",
                        source: &mut second,
                        bytes: 3,
                        content: id(b"def"),
                    },
                ],
                max_bytes: 6,
            },
            limits(),
            || Ok::<_, u8>(()),
        )
        .unwrap_err();
        if fail {
            assert!(
                error
                    .to_string()
                    .contains("original injected companion read")
            );
        }
        assert_eq!(fs::read(out.join("second.bin")).unwrap(), b"foreign-owned");
        assert_eq!(fs::read_dir(out).unwrap().count(), 1);
    }
}
