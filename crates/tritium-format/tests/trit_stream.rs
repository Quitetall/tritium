use std::io::{self, Read, Write};
use tritium_format::{
    PackageHasher, PackageId, TritBlob, TritBlobKind, TritPackageWriteError, read_trit_package,
    write_trit_package_bounded, write_trit_package_to,
};

const MANIFEST: &[u8] = &[0x82, 0x01, 0x02];
struct ShortWriter {
    bytes: Vec<u8>,
    width: usize,
    interrupted: bool,
}
impl Write for ShortWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.interrupted {
            self.interrupted = false;
            return Err(io::ErrorKind::Interrupted.into());
        }
        let n = self.width.min(bytes.len());
        self.bytes.extend_from_slice(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        panic!("writer must not flush caller's destination")
    }
}

#[test]
fn stream_matches_buffer_dedup_order_alignment_empty_and_short_write_parity() {
    for length in [0, 1, 63, 64, 255, 256, 257, 4096] {
        let payload: Vec<_> = (0..length).map(|i| (i % 251) as u8).collect();
        let mut blobs = vec![
            TritBlob {
                kind: TritBlobKind::Asset("a".into()),
                bytes: &payload,
            },
            TritBlob {
                kind: TritBlobKind::Asset("alias".into()),
                bytes: &payload,
            },
            TritBlob {
                kind: TritBlobKind::Asset("a".into()),
                bytes: &payload,
            },
            TritBlob {
                kind: TritBlobKind::AdditiveTensor,
                bytes: b"other",
            },
        ];
        for _ in 0..2 {
            let expected = write_trit_package_bounded(MANIFEST, &blobs, 100_000).unwrap();
            for width in [1, 3, 255, 4096] {
                let mut writer = ShortWriter {
                    bytes: Vec::new(),
                    width,
                    interrupted: true,
                };
                let id =
                    write_trit_package_to(MANIFEST, &blobs, expected.len(), &mut writer).unwrap();
                assert_eq!(writer.bytes, expected);
                assert_eq!(id, PackageId::from_package_bytes(&expected));
                assert_eq!(read_trit_package(&writer.bytes).unwrap().package_id(), id);
            }
            blobs.reverse();
        }
    }
    let expected = write_trit_package_bounded(MANIFEST, &[], 100_000).unwrap();
    let mut actual = Vec::new();
    assert_eq!(
        write_trit_package_to(MANIFEST, &[], expected.len(), &mut actual).unwrap(),
        PackageId::from_package_bytes(&expected)
    );
    assert_eq!(actual, expected);
}

#[test]
fn invalid_inputs_and_complete_byte_bound_reject_before_any_destination_write() {
    let blobs = [TritBlob {
        kind: TritBlobKind::Asset("a".into()),
        bytes: b"payload",
    }];
    let bytes = write_trit_package_bounded(MANIFEST, &blobs, 100_000).unwrap();
    let mut destination = vec![9, 8, 7];
    for limit in [0, 63, bytes.len() - 1] {
        assert!(matches!(
            write_trit_package_to(MANIFEST, &blobs, limit, &mut destination),
            Err(TritPackageWriteError::Package(_))
        ));
        assert_eq!(destination, [9, 8, 7]);
    }
    assert!(write_trit_package_to(&[0xff], &blobs, 100_000, &mut destination).is_err());
    let invalid = [TritBlob {
        kind: TritBlobKind::Asset(String::new()),
        bytes: b"x",
    }];
    assert!(write_trit_package_to(MANIFEST, &invalid, 100_000, &mut destination).is_err());
    assert_eq!(destination, [9, 8, 7]);
}

#[test]
fn destination_faults_and_write_zero_return_errors_with_only_actual_prefix() {
    struct Fault {
        bytes: Vec<u8>,
        remaining: usize,
        zero: bool,
    }
    impl Write for Fault {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                return if self.zero {
                    Ok(0)
                } else {
                    Err(io::ErrorKind::BrokenPipe.into())
                };
            }
            let n = self.remaining.min(bytes.len());
            self.remaining -= n;
            self.bytes.extend_from_slice(&bytes[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            panic!("no implicit flush")
        }
    }
    let blobs = [TritBlob {
        kind: TritBlobKind::Asset("a".into()),
        bytes: b"payload",
    }];
    let expected = write_trit_package_bounded(MANIFEST, &blobs, 100_000).unwrap();
    for position in [0, 7, 64, 255, 258, expected.len() - 1] {
        for zero in [false, true] {
            let mut destination = Fault {
                bytes: Vec::new(),
                remaining: position,
                zero,
            };
            let Err(TritPackageWriteError::Io(e)) =
                write_trit_package_to(MANIFEST, &blobs, 100_000, &mut destination)
            else {
                panic!("must return I/O failure, no identity")
            };
            assert_eq!(
                e.kind(),
                if zero {
                    io::ErrorKind::WriteZero
                } else {
                    io::ErrorKind::BrokenPipe
                }
            );
            assert_eq!(destination.bytes, expected[..position]);
        }
    }
}

#[test]
fn large_borrowed_payload_streams_to_hash_sink_without_retaining_output() {
    struct HashSink {
        hash: PackageHasher,
        bytes: usize,
    }
    impl Write for HashSink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.hash.update(bytes);
            self.bytes += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let payload = vec![23; 8 * 1024 * 1024];
    let blobs = [TritBlob {
        kind: TritBlobKind::Asset("large".into()),
        bytes: &payload,
    }];
    let mut sink = HashSink {
        hash: PackageHasher::new(),
        bytes: 0,
    };
    let id = write_trit_package_to(MANIFEST, &blobs, 9 * 1024 * 1024, &mut sink).unwrap();
    assert_eq!(id, sink.hash.finalize());
    assert!(sink.bytes > payload.len());
    // Caller retains 8 MiB input; sink retains only its incremental hash/count.
    // This structural scenario does not measure or qualify complete RSS.
}

#[test]
fn actual_owned_file_stream_is_readable_and_removed_after_roundtrip() {
    struct FileGuard(std::path::PathBuf);
    impl Drop for FileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let path = std::env::temp_dir().join(format!("tritium-stream-{}.trit", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let guard = FileGuard(path.clone());
    let blobs = [TritBlob {
        kind: TritBlobKind::Asset("a".into()),
        bytes: b"payload",
    }];
    let id = write_trit_package_to(MANIFEST, &blobs, 100_000, &mut file).unwrap();
    file.sync_all().unwrap();
    drop(file);
    let mut bytes = Vec::new();
    std::fs::File::open(&path)
        .unwrap()
        .take(100_001)
        .read_to_end(&mut bytes)
        .unwrap();
    assert_eq!(read_trit_package(&bytes).unwrap().package_id(), id);
    drop(guard);
    assert!(!path.exists());
}
