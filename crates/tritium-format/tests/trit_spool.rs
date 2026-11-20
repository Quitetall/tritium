//! Exercise the canonical writer through real bounded seekable source adapters.
use std::cell::{Cell, RefCell};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::rc::Rc;
use tritium_core::DType;
use tritium_format::{
    PackageId, TritBlob, TritBlobKind, TritPackageError, TritPackageReadLimits, TritPackageReader,
    TritPackageWriteError, TritSpoolBlob, read_trit_package, write_trit_package,
    write_trit_package_from_spool,
};
use tritium_schema::BlobId;

const MANIFEST: &[u8] = &[0x82, 0x01, 0x02];
const CHUNK: usize = 64 * 1024;

fn stage(blobs: &[TritBlob<'_>]) -> (Cursor<Vec<u8>>, Vec<TritSpoolBlob>) {
    // Separate ranges for duplicates, with deliberately unreferenced gaps.
    let mut bytes = vec![13; 17];
    let entries = blobs
        .iter()
        .map(|blob| {
            let entry = TritSpoolBlob {
                kind: blob.kind.clone(),
                id: BlobId::from_bytes(*blake3::hash(blob.bytes).as_bytes()),
                offset: bytes.len() as u64,
                length: blob.bytes.len() as u64,
            };
            bytes.extend_from_slice(blob.bytes);
            bytes.extend_from_slice(&[9; 3]);
            entry
        })
        .collect();
    (Cursor::new(bytes), entries)
}

struct BoundedSource {
    cursor: Cursor<Vec<u8>>,
    largest_request: usize,
    fail_read: bool,
    fail_seek: bool,
}
impl Read for BoundedSource {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        self.largest_request = self.largest_request.max(out.len());
        assert!(out.len() <= CHUNK, "no blob-size source read");
        if self.fail_read {
            return Err(io::Error::other("source read fault"));
        }
        let size = out.len().min(1031); // real short reads, not full-buffer mock reads
        self.cursor.read(&mut out[..size])
    }
}
impl Seek for BoundedSource {
    fn seek(&mut self, at: SeekFrom) -> io::Result<u64> {
        if self.fail_seek {
            return Err(io::Error::other("source seek fault"));
        }
        self.cursor.seek(at)
    }
}
struct ShortSink(Vec<u8>);
impl Write for ShortSink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let count = bytes.len().min(791);
        self.0.extend_from_slice(&bytes[..count]);
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        panic!("writer must not flush a caller-owned destination")
    }
}

#[test]
fn spool_matches_borrowed_bytes_kinds_dedup_empty_order_and_short_io() {
    let large: Vec<u8> = (0..CHUNK * 3 + 7).map(|i| (i % 251) as u8).collect();
    let mut blobs = vec![
        TritBlob {
            kind: TritBlobKind::AdditiveTensor,
            bytes: &large,
        },
        TritBlob {
            kind: TritBlobKind::DenseTensor(DType::F16),
            bytes: b"dense",
        },
        TritBlob {
            kind: TritBlobKind::Asset("config.json".into()),
            bytes: b"{}",
        },
        TritBlob {
            kind: TritBlobKind::Asset("empty".into()),
            bytes: b"",
        },
        TritBlob {
            kind: TritBlobKind::AdditiveTensor,
            bytes: &large,
        },
        TritBlob {
            kind: TritBlobKind::Asset("alias".into()),
            bytes: &large,
        },
    ];
    for _ in 0..blobs.len() {
        blobs.rotate_left(1);
        let expected = write_trit_package(MANIFEST, &blobs).unwrap();
        let (cursor, entries) = stage(&blobs);
        let mut source = BoundedSource {
            cursor,
            largest_request: 0,
            fail_read: false,
            fail_seek: false,
        };
        let mut out = ShortSink(Vec::new());
        let id = write_trit_package_from_spool(
            MANIFEST,
            &entries,
            &mut source,
            expected.len(),
            &mut out,
        )
        .unwrap();
        assert_eq!(out.0, expected);
        assert_eq!(id, PackageId::from_package_bytes(&expected));
        assert_eq!(source.largest_request, CHUNK);
        let parsed = read_trit_package(&out.0).unwrap();
        assert_eq!(parsed.blobs().len(), 5);
        for info in parsed.blobs() {
            parsed.blob(info).unwrap();
        }
        let mut reopened = TritPackageReader::open(
            Cursor::new(out.0.as_slice()),
            TritPackageReadLimits {
                max_source_bytes: out.0.len() as u64,
                max_manifest_bytes: MANIFEST.len(),
                max_directory_bytes: 1024,
            },
        )
        .unwrap();
        assert_eq!(reopened.package_id(), id);
        assert_eq!(reopened.blobs(), parsed.blobs());
        for (index, info) in parsed.blobs().iter().enumerate() {
            let mut hash = blake3::Hasher::new();
            reopened
                .visit_blob(index, |chunk| -> Result<(), ()> {
                    hash.update(chunk);
                    Ok(())
                })
                .unwrap();
            assert_eq!(hash.finalize().as_bytes(), info.id.as_bytes());
        }
        reopened.verify().unwrap();
    }
    let expected = write_trit_package(MANIFEST, &[]).unwrap();
    let mut out = Vec::new();
    write_trit_package_from_spool(
        MANIFEST,
        &[],
        &mut Cursor::new(Vec::new()),
        expected.len(),
        &mut out,
    )
    .unwrap();
    assert_eq!(out, expected);
}

#[test]
fn bad_metadata_ranges_and_digests_leave_destination_untouched() {
    let blobs = [TritBlob {
        kind: TritBlobKind::AdditiveTensor,
        bytes: b"payload",
    }];
    let (source, entries) = stage(&blobs);
    let size = write_trit_package(MANIFEST, &blobs).unwrap().len();
    let mut cases = Vec::new();
    let mut bad = entries.clone();
    bad[0].id = BlobId::from_bytes([7; 32]);
    cases.push(bad);
    let mut bad = entries.clone();
    bad[0].length = 1000;
    cases.push(bad);
    let mut bad = entries.clone();
    bad[0].offset = u64::MAX;
    cases.push(bad);
    let mut bad = entries.clone();
    bad[0].kind = TritBlobKind::Asset(String::new());
    cases.push(bad);
    for entries in cases {
        let mut out = Vec::new();
        assert!(
            write_trit_package_from_spool(
                MANIFEST,
                &entries,
                &mut source.clone(),
                usize::MAX,
                &mut out
            )
            .is_err()
        );
        assert!(out.is_empty());
    }
    for (manifest, limit) in [(&[0x18, 0x01][..], usize::MAX), (MANIFEST, size - 1)] {
        let mut out = Vec::new();
        assert!(
            write_trit_package_from_spool(manifest, &entries, &mut source.clone(), limit, &mut out)
                .is_err()
        );
        assert!(out.is_empty());
    }
    // Size admission precedes source I/O, avoiding a large scan of refused output.
    let mut source = BoundedSource {
        cursor: source,
        largest_request: 0,
        fail_read: true,
        fail_seek: true,
    };
    assert!(matches!(
        write_trit_package_from_spool(MANIFEST, &entries, &mut source, size - 1, &mut Vec::new()),
        Err(TritPackageWriteError::Package(
            TritPackageError::PackageLimitExceeded { .. }
        ))
    ));
}

#[test]
fn exact_layout_inspection_matches_actual_deduplication_and_ignores_source_offsets() {
    use tritium_format::inspect_trit_spool_layout;
    for length in [0, 1, 23, 255, 256, 65536] {
        let data = vec![19; length];
        let other = vec![23; length];
        let blobs = [
            TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: &data,
            },
            TritBlob {
                kind: TritBlobKind::Asset("same.bin".into()),
                bytes: &data,
            },
            TritBlob {
                kind: TritBlobKind::Asset("other.bin".into()),
                bytes: &other,
            },
        ];
        let (_, mut entries) = stage(&blobs);
        let actual = write_trit_package(MANIFEST, &blobs).unwrap();
        let size = inspect_trit_spool_layout(MANIFEST, &entries).unwrap();
        assert_eq!(size.package_bytes, actual.len() as u64);
        assert_eq!(size.manifest_bytes, MANIFEST.len() as u64);
        assert_eq!(
            size.directory_bytes,
            u64::from_le_bytes(actual[40..48].try_into().unwrap())
        );
        for entry in &mut entries {
            entry.offset = u64::MAX;
        }
        assert_eq!(
            inspect_trit_spool_layout(MANIFEST, &entries).unwrap(),
            size,
            "inspection is metadata layout, not source-range verification"
        );
    }
}

#[test]
fn source_errors_are_distinct_and_preflight_does_not_write() {
    let (cursor, entries) = stage(&[TritBlob {
        kind: TritBlobKind::AdditiveTensor,
        bytes: b"payload",
    }]);
    for (fail_read, fail_seek) in [(true, false), (false, true)] {
        let mut source = BoundedSource {
            cursor: cursor.clone(),
            largest_request: 0,
            fail_read,
            fail_seek,
        };
        let mut out = Vec::new();
        assert!(matches!(
            write_trit_package_from_spool(MANIFEST, &entries, &mut source, usize::MAX, &mut out),
            Err(TritPackageWriteError::Source(_))
        ));
        assert!(out.is_empty());
    }
}

struct SharedSource {
    bytes: Rc<RefCell<Vec<u8>>>,
    at: u64,
    fail: Rc<Cell<bool>>,
}
impl Read for SharedSource {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.fail.get() {
            return Err(io::Error::other("source failed after preflight"));
        }
        let bytes = self.bytes.borrow();
        let start = (self.at as usize).min(bytes.len());
        let count = out.len().min(bytes.len() - start);
        out[..count].copy_from_slice(&bytes[start..start + count]);
        self.at += count as u64;
        Ok(count)
    }
}
impl Seek for SharedSource {
    fn seek(&mut self, at: SeekFrom) -> io::Result<u64> {
        self.at = match at {
            SeekFrom::Start(at) => at,
            SeekFrom::End(0) => self.bytes.borrow().len() as u64,
            _ => return Err(io::Error::other("unsupported test seek")),
        };
        Ok(self.at)
    }
}
struct MutatingSink<F> {
    bytes: Vec<u8>,
    mutate: Option<F>,
    after: usize,
}
impl<F: FnOnce()> Write for MutatingSink<F> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.bytes.len() >= self.after
            && let Some(mutate) = self.mutate.take()
        {
            mutate();
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn same_length_mutation_of_emitted_or_deduplicated_range_returns_no_identity() {
    for alias in [false, true] {
        let (cursor, entries) = stage(&[
            TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: b"same bytes",
            },
            TritBlob {
                kind: TritBlobKind::Asset("alias".into()),
                bytes: b"same bytes",
            },
        ]);
        let bytes = Rc::new(RefCell::new(cursor.into_inner()));
        let changed = bytes.clone();
        let index = usize::from(alias);
        let offset = entries[index].offset as usize;
        let mut source = SharedSource {
            bytes,
            at: 0,
            fail: Rc::new(Cell::new(false)),
        };
        let mut out = MutatingSink {
            bytes: Vec::new(),
            mutate: Some(move || changed.borrow_mut()[offset] ^= 1),
            after: 0,
        };
        assert!(matches!(
            write_trit_package_from_spool(MANIFEST, &entries, &mut source, usize::MAX, &mut out),
            Err(TritPackageWriteError::Package(
                TritPackageError::BlobDigestMismatch(_)
            ))
        ));
        assert!(!out.bytes.is_empty());
        assert!(read_trit_package(&out.bytes).is_err()); // footer never seals pending output
    }
}

#[test]
fn source_failure_after_admission_leaves_unsealed_pending_bytes() {
    let (cursor, entries) = stage(&[TritBlob {
        kind: TritBlobKind::AdditiveTensor,
        bytes: b"payload",
    }]);
    let fail = Rc::new(Cell::new(false));
    let changed = fail.clone();
    let mut source = SharedSource {
        bytes: Rc::new(RefCell::new(cursor.into_inner())),
        at: 0,
        fail,
    };
    let mut out = MutatingSink {
        bytes: Vec::new(),
        mutate: Some(move || changed.set(true)),
        after: 0,
    };
    assert!(matches!(
        write_trit_package_from_spool(MANIFEST, &entries, &mut source, usize::MAX, &mut out),
        Err(TritPackageWriteError::Source(_))
    ));
    assert!(!out.bytes.is_empty());
    assert!(read_trit_package(&out.bytes).is_err());
}

#[test]
fn mutation_after_payload_emission_is_checked_before_footer() {
    let payload = b"already emitted";
    let (cursor, entries) = stage(&[TritBlob {
        kind: TritBlobKind::AdditiveTensor,
        bytes: payload,
    }]);
    let bytes = Rc::new(RefCell::new(cursor.into_inner()));
    let changed = bytes.clone();
    let offset = entries[0].offset as usize;
    let mut source = SharedSource {
        bytes,
        at: 0,
        fail: Rc::new(Cell::new(false)),
    };
    let mut out = MutatingSink {
        bytes: Vec::new(),
        mutate: Some(move || changed.borrow_mut()[offset] ^= 1),
        after: 256 + payload.len(), // mutate on manifest write, after payload hash
    };
    assert!(matches!(
        write_trit_package_from_spool(MANIFEST, &entries, &mut source, usize::MAX, &mut out),
        Err(TritPackageWriteError::Package(
            TritPackageError::BlobDigestMismatch(_)
        ))
    ));
    assert!(
        out.bytes
            .windows(payload.len())
            .any(|bytes| bytes == payload)
    );
    assert!(read_trit_package(&out.bytes).is_err());
}

#[test]
fn destination_failures_do_not_return_an_identity() {
    struct FaultSink {
        remaining: usize,
        zero: bool,
    }
    impl Write for FaultSink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                if self.zero {
                    return Ok(0);
                }
                return Err(io::Error::other("destination fault"));
            }
            let count = bytes.len().min(self.remaining);
            self.remaining -= count;
            Ok(count)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let (cursor, entries) = stage(&[TritBlob {
        kind: TritBlobKind::AdditiveTensor,
        bytes: b"payload",
    }]);
    let size = write_trit_package(
        MANIFEST,
        &[TritBlob {
            kind: TritBlobKind::AdditiveTensor,
            bytes: b"payload",
        }],
    )
    .unwrap()
    .len();
    for remaining in [0, 1, 64, 256, size - 1] {
        for zero in [false, true] {
            assert!(matches!(
                write_trit_package_from_spool(
                    MANIFEST,
                    &entries,
                    &mut cursor.clone(),
                    usize::MAX,
                    &mut FaultSink { remaining, zero }
                ),
                Err(TritPackageWriteError::Io(_))
            ));
        }
    }
}

#[test]
fn actual_file_spool_roundtrip_and_cleanup() {
    struct FileGuard(std::path::PathBuf);
    impl Drop for FileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let path = std::env::temp_dir().join(format!("tritium-canonical-spool-{}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let guard = FileGuard(path.clone());
    let blobs = [TritBlob {
        kind: TritBlobKind::DenseTensor(DType::F32),
        bytes: b"dense tensor",
    }];
    let (cursor, entries) = stage(&blobs);
    file.write_all(cursor.get_ref()).unwrap();
    let expected = write_trit_package(MANIFEST, &blobs).unwrap();
    let mut out = Vec::new();
    let result =
        write_trit_package_from_spool(MANIFEST, &entries, &mut file, expected.len(), &mut out);
    drop(file);
    drop(guard);
    assert!(!path.exists());
    result.unwrap();
    assert_eq!(out, expected);
    read_trit_package(&out).unwrap();
}
