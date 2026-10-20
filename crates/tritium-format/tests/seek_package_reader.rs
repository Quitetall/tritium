use std::cell::{Cell, RefCell};
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::rc::Rc;

use tritium_core::DType;
use tritium_format::{
    TritBlob, TritBlobKind, TritBlobVisitError, TritPackageError, TritPackageReadError,
    TritPackageReadLimits, TritPackageReader, read_trit_package, write_trit_package,
};

const CHUNK: usize = 64 * 1024;

fn fixture(size: usize) -> Vec<u8> {
    let payload: Vec<_> = (0..size).map(|i| (i % 251) as u8).collect();
    write_trit_package(
        &[0xa0],
        &[
            TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &payload,
            },
            TritBlob {
                kind: TritBlobKind::DenseTensor(DType::F32),
                bytes: &payload,
            },
            TritBlob {
                kind: TritBlobKind::Asset("config.json".into()),
                bytes: b"{}",
            },
            TritBlob {
                kind: TritBlobKind::Asset("empty".into()),
                bytes: b"",
            },
        ],
    )
    .unwrap()
}

fn limits(bytes: &[u8]) -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: bytes.len() as u64,
        max_manifest_bytes: 1,
        max_directory_bytes: usize::try_from(u64::from_le_bytes(bytes[40..48].try_into().unwrap()))
            .unwrap(),
    }
}

#[derive(Debug)]
struct Source {
    bytes: Rc<RefCell<Vec<u8>>>,
    position: u64,
    reads: Rc<Cell<usize>>,
    max_request: Rc<Cell<usize>>,
    fail_read: Rc<Cell<bool>>,
    fail_seek: Rc<Cell<bool>>,
    wrong_seek: Rc<Cell<bool>>,
    short_read: Rc<Cell<bool>>,
    mutate_at: Option<(u64, usize, usize)>,
    matching_seeks: usize,
}

impl Source {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Rc::new(RefCell::new(bytes)),
            position: 37,
            reads: Rc::new(Cell::new(0)),
            max_request: Rc::new(Cell::new(0)),
            fail_read: Rc::new(Cell::new(false)),
            fail_seek: Rc::new(Cell::new(false)),
            wrong_seek: Rc::new(Cell::new(false)),
            short_read: Rc::new(Cell::new(false)),
            mutate_at: None,
            matching_seeks: 0,
        }
    }
}

impl Read for Source {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        self.reads.set(self.reads.get() + 1);
        self.max_request
            .set(self.max_request.get().max(output.len()));
        assert!(output.len() <= CHUNK, "source read exceeds fixed chunk");
        if self.fail_read.get() {
            return Err(io::Error::other("injected read"));
        }
        if self.short_read.get() {
            return Ok(0);
        }
        let bytes = self.bytes.borrow();
        let start = usize::try_from(self.position).unwrap();
        let count = output.len().min(13).min(bytes.len().saturating_sub(start));
        if count != 0 {
            output[..count].copy_from_slice(&bytes[start..start + count]);
        }
        drop(bytes);
        self.position += count as u64;
        Ok(count)
    }
}

impl Seek for Source {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        if self.fail_seek.get() {
            return Err(io::Error::other("injected seek"));
        }
        if let SeekFrom::Start(offset) = from
            && let Some((trigger, byte, nth)) = self.mutate_at
            && offset == trigger
        {
            self.matching_seeks += 1;
            if self.matching_seeks == nth {
                self.bytes.borrow_mut()[byte] ^= 1;
                self.mutate_at = None;
            }
        }
        let position = match from {
            SeekFrom::Start(n) => i128::from(n),
            SeekFrom::End(n) => self.bytes.borrow().len() as i128 + i128::from(n),
            SeekFrom::Current(n) => i128::from(self.position) + i128::from(n),
        };
        self.position = u64::try_from(position).map_err(|_| io::Error::other("negative seek"))?;
        Ok(self.position + u64::from(self.wrong_seek.get()))
    }
}

#[test]
fn cursor_and_bounded_short_read_adapter_match_the_borrowed_package() {
    for size in [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK + 7] {
        let bytes = fixture(size);
        let expected = read_trit_package(&bytes).unwrap();
        let mut cursor = Cursor::new(bytes.as_slice());
        cursor.set_position(91);
        let mut reader = TritPackageReader::open(cursor, limits(&bytes)).unwrap();
        assert_eq!(reader.manifest(), expected.manifest());
        assert_eq!(reader.blobs(), expected.blobs());
        assert_eq!(reader.package_id(), expected.package_id());
        for index in 0..reader.blobs().len() {
            let mut actual = Vec::new();
            reader
                .visit_blob(index, |chunk| -> Result<(), ()> {
                    assert!(!chunk.is_empty() && chunk.len() <= CHUNK);
                    actual.extend_from_slice(chunk);
                    Ok(())
                })
                .unwrap();
            assert_eq!(actual, expected.blob(&expected.blobs()[index]).unwrap());
        }
        reader.verify().unwrap();
        let source = Source::new(bytes.clone());
        let max_request = source.max_request.clone();
        let mut bounded = TritPackageReader::open(source, limits(&bytes)).unwrap();
        assert_eq!(bounded.package_id(), expected.package_id());
        bounded.verify().unwrap();
        assert!(max_request.get() <= CHUNK);
    }
}

#[test]
fn exact_source_and_metadata_caps_refuse_before_body_reads() {
    let bytes = fixture(CHUNK + 1);
    let exact = limits(&bytes);
    for cap in [0, exact.max_source_bytes - 1] {
        let source = Source::new(bytes.clone());
        let reads = source.reads.clone();
        assert!(matches!(
            TritPackageReader::open(
                source,
                TritPackageReadLimits {
                    max_source_bytes: cap,
                    ..exact
                }
            ),
            Err(TritPackageReadError::SourceLimitExceeded { .. })
        ));
        assert_eq!(reads.get(), 0);
    }
    for caps in [
        TritPackageReadLimits {
            max_manifest_bytes: 0,
            ..exact
        },
        TritPackageReadLimits {
            max_directory_bytes: exact.max_directory_bytes - 1,
            ..exact
        },
    ] {
        let source = Source::new(bytes.clone());
        let reads = source.reads.clone();
        assert!(matches!(
            TritPackageReader::open(source, caps),
            Err(TritPackageReadError::Format(
                TritPackageError::MetadataLimitExceeded
            ))
        ));
        // Header/footer only, with 13-byte partial reads. No whole-body scan.
        assert_eq!(reads.get(), 11);
    }
}

#[test]
fn common_format_refusals_match_for_malformed_and_mutated_packages() {
    let original = fixture(257);
    let mut cases = vec![Vec::new(), original[..100].to_vec()];
    for offset in [
        0,
        8,
        10,
        12,
        48,
        64,
        256,
        original.len() - 72,
        original.len() - 1,
    ] {
        let mut bytes = original.clone();
        bytes[offset] ^= 1;
        cases.push(bytes);
    }
    for offset in [16, 24, 32, 40] {
        let mut bytes = original.clone();
        bytes[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        cases.push(bytes);
    }
    let mut trailing = original.clone();
    trailing.push(0);
    cases.push(trailing);
    for cut in 0..original.len() {
        cases.push(original[..cut].to_vec());
    }
    // Deterministic single-byte mutation property, including metadata framing.
    for index in 0..512 {
        let mut bytes = original.clone();
        let offset = index * 7919 % bytes.len();
        bytes[offset] ^= ((index % 255) + 1) as u8;
        cases.push(bytes);
    }
    eprintln!("SEEK_PACKAGE_REFUSAL_PARITY cases={}", cases.len());
    for bytes in cases {
        let expected = read_trit_package(&bytes);
        let actual = TritPackageReader::open(
            Cursor::new(&bytes),
            TritPackageReadLimits {
                max_source_bytes: u64::MAX,
                max_manifest_bytes: usize::MAX,
                max_directory_bytes: usize::MAX,
            },
        );
        match (expected, actual) {
            (Err(expected), Err(TritPackageReadError::Format(actual))) => {
                assert_eq!(actual, expected)
            }
            (Ok(expected), Ok(actual)) => assert_eq!(actual.package_id(), expected.package_id()),
            other => panic!("reader refusal parity failed: {other:?}"),
        }
    }
}

#[test]
fn source_faults_and_wrong_seek_are_not_successful_admission() {
    let bytes = fixture(257);
    for fault in 0..4 {
        let source = Source::new(bytes.clone());
        match fault {
            0 => source.fail_read.set(true),
            1 => source.fail_seek.set(true),
            2 => source.wrong_seek.set(true),
            _ => source.short_read.set(true),
        }
        assert!(matches!(
            TritPackageReader::open(
                source,
                TritPackageReadLimits {
                    max_source_bytes: u64::MAX,
                    ..limits(&bytes)
                }
            ),
            Err(TritPackageReadError::Io { .. })
        ));
    }
    let mut source = Source::new(bytes.clone());
    source.mutate_at = Some((0, 0, 1));
    // First header read mutation is caught by the common magic validator.
    assert!(matches!(
        TritPackageReader::open(source, limits(&bytes)),
        Err(TritPackageReadError::Format(TritPackageError::BadMagic))
    ));
}

#[test]
fn admission_detects_header_and_same_size_payload_mutation_between_passes() {
    let bytes = fixture(257);
    let payload = read_trit_package(&bytes)
        .unwrap()
        .blobs()
        .iter()
        .find(|blob| blob.kind == TritBlobKind::AdditiveTensor)
        .unwrap()
        .offset as usize;
    // First full scan changes the header after preflight, or the terminal full
    // scan changes already validated payload. Neither returns a mixed identity.
    for (byte, nth) in [(8, 2), (payload, 4)] {
        let mut source = Source::new(bytes.clone());
        source.mutate_at = Some((0, byte, nth));
        assert!(matches!(
            TritPackageReader::open(source, limits(&bytes)),
            Err(TritPackageReadError::SourceChanged)
        ));
    }
}

#[test]
#[cfg(unix)]
fn actual_file_retains_the_original_handle_after_path_replacement() {
    use std::fs::{self, File};
    use std::io::Write;
    let bytes = fixture(CHUNK + 1);
    let path = std::env::temp_dir().join(format!(
        "tritium-seek-package-retained-{}-{:?}.trit",
        std::process::id(),
        std::thread::current().id()
    ));
    let moved = path.with_extension("retained");
    let mut output = File::create_new(&path).unwrap();
    output.write_all(&bytes).unwrap();
    drop(output);
    let mut reader = TritPackageReader::open(File::open(&path).unwrap(), limits(&bytes)).unwrap();
    fs::rename(&path, &moved).unwrap();
    let mut replacement = File::create_new(&path).unwrap();
    replacement
        .write_all(b"unrelated replacement source")
        .unwrap();
    drop(replacement);
    let expected = read_trit_package(&bytes).unwrap();
    for index in 0..reader.blobs().len() {
        let mut actual = Vec::new();
        reader
            .visit_blob(index, |chunk| -> io::Result<()> {
                actual.extend_from_slice(chunk);
                Ok(())
            })
            .unwrap();
        assert_eq!(actual, expected.blob(&expected.blobs()[index]).unwrap());
    }
    reader.verify().unwrap();
    drop(reader);
    fs::remove_file(path).unwrap();
    fs::remove_file(moved).unwrap();
}

#[test]
fn visits_preserve_typed_failures_and_detect_changed_blobs_and_terminal_source() {
    let bytes = fixture(2 * CHUNK + 7);
    let source = Source::new(bytes.clone());
    let mutable = source.bytes.clone();
    let fail_read = source.fail_read.clone();
    let mut reader = TritPackageReader::open(source, limits(&bytes)).unwrap();
    let index = reader
        .blobs()
        .iter()
        .position(|b| b.kind == TritBlobKind::AdditiveTensor)
        .unwrap();
    assert!(matches!(
        reader.visit_blob(usize::MAX, |_| Ok::<_, ()>(())),
        Err(TritBlobVisitError::Source(TritPackageReadError::Format(
            TritPackageError::InvalidDirectory
        )))
    ));
    assert!(matches!(
        reader.visit_blob(index, |_| Err("typed visitor stop")),
        Err(TritBlobVisitError::Visitor("typed visitor stop"))
    ));
    fail_read.set(true);
    assert!(matches!(
        reader.visit_blob(index, |_| Ok::<_, ()>(())),
        Err(TritBlobVisitError::Source(TritPackageReadError::Io { .. }))
    ));
    fail_read.set(false);
    let offset = reader.blobs()[index].offset as usize;
    let mut calls = 0;
    assert!(matches!(
        reader.visit_blob(index, |_| {
            calls += 1;
            if calls == 1 {
                mutable.borrow_mut()[offset + CHUNK] ^= 1;
            }
            Ok::<_, ()>(())
        }),
        Err(TritBlobVisitError::Source(TritPackageReadError::Format(
            TritPackageError::BlobDigestMismatch(_)
        )))
    ));
    assert!(matches!(
        reader.verify(),
        Err(TritPackageReadError::SourceChanged)
    ));
    mutable.borrow_mut()[offset + CHUNK] ^= 1;
    reader.verify().unwrap();
    mutable.borrow_mut()[48] ^= 1;
    assert!(matches!(
        reader.verify(),
        Err(TritPackageReadError::SourceChanged)
    ));
    mutable.borrow_mut()[48] ^= 1;
    mutable.borrow_mut().push(0);
    assert!(matches!(
        reader.visit_blob(index, |_| Ok::<_, ()>(())),
        Err(TritBlobVisitError::Source(
            TritPackageReadError::SourceChanged
        ))
    ));
    assert!(matches!(
        reader.verify(),
        Err(TritPackageReadError::SourceChanged)
    ));
}
