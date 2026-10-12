//! Actual pending-stream staging, not a model/release qualification receipt.
use std::cell::{Cell, RefCell};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::rc::Rc;
use tritium_format::trit_model::{
    TritModelStageError, read_trit_model, stage_trit_model_from_spool,
};
use tritium_format::{TritPackageReadLimits, TritPackageWriteError};
#[path = "support/typed_seek_fixture.rs"]
mod fixture;
#[path = "support/model_spool_fixture.rs"]
mod preparation;
use preparation::prepared;

fn limits() -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: 32 * 1024 * 1024,
        max_manifest_bytes: 1024 * 1024,
        max_directory_bytes: 1024 * 1024,
    }
}
struct Pending {
    bytes: Rc<RefCell<Cursor<Vec<u8>>>>,
    fault: u8,
    replacement: Option<Vec<u8>>,
}
impl Read for Pending {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        assert!(out.len() <= 64 * 1024);
        if self.fault == 1 {
            return Err(io::Error::other("pending read"));
        }
        let count = out.len().min(13);
        self.bytes.borrow_mut().read(&mut out[..count])
    }
}
impl Write for Pending {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.fault == 2 {
            return Err(io::Error::other("pending write"));
        }
        if self.fault == 6 && self.bytes.borrow().get_ref().len() >= 100 {
            return Ok(0);
        }
        let count = bytes.len().min(17);
        self.bytes.borrow_mut().write(&bytes[..count])
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.fault == 3 {
            return Err(io::Error::other("pending flush"));
        }
        if let Some(replacement) = self.replacement.take() {
            *self.bytes.borrow_mut() = Cursor::new(replacement);
        }
        Ok(())
    }
}
impl Seek for Pending {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        if self.fault == 4 {
            return Err(io::Error::other("pending seek"));
        }
        if self.fault == 5 && matches!(position, SeekFrom::Start(0)) {
            return Ok(1);
        }
        self.bytes.borrow_mut().seek(position)
    }
}
fn pending(fault: u8) -> (Pending, Rc<RefCell<Cursor<Vec<u8>>>>) {
    let bytes = Rc::new(RefCell::new(Cursor::new(Vec::new())));
    (
        Pending {
            bytes: Rc::clone(&bytes),
            fault,
            replacement: None,
        },
        bytes,
    )
}

#[test]
fn exact_staging_reopens_all_levels_and_visits_with_short_io_and_one_source_guard() {
    for cols in [1, 257, 65537] {
        let bytes = fixture::fixture(cols, 131076);
        let (manifest, blobs) = prepared(&bytes);
        let expected = read_trit_model(&bytes).unwrap();
        let (destination, observed) = pending(0);
        let calls = Cell::new(0);
        let mut reader = stage_trit_model_from_spool(
            &manifest,
            &blobs,
            &mut Cursor::new(&bytes),
            limits(),
            destination,
            || -> Result<(), u8> {
                calls.set(calls.get() + 1);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(observed.borrow().get_ref(), &bytes);
        assert_eq!(reader.manifest(), expected.manifest());
        assert_eq!(reader.package_id(), expected.package_id());
        reader.validate_complete_payload_inventory().unwrap();
        for level in &manifest.levels {
            assert_eq!(
                reader.model_id(&level.name).unwrap(),
                expected.model_id(&level.name).unwrap()
            );
            for tensor in &level.tensors {
                assert_eq!(
                    reader.tensor_digest(&level.name, &tensor.name).unwrap(),
                    expected.tensor_digest(&level.name, &tensor.name).unwrap()
                );
                let mut output = Vec::new();
                reader
                    .visit_tensor(&level.name, &tensor.name, |chunk| -> Result<(), ()> {
                        output.extend_from_slice(chunk);
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(
                    output,
                    expected.tensor(&level.name, &tensor.name).unwrap().1
                );
            }
        }
        reader.verify().unwrap();
    }
}

#[test]
fn all_pending_io_failures_caps_and_nonempty_destination_return_no_staged_success() {
    let bytes = fixture::fixture(257, 16);
    let (manifest, blobs) = prepared(&bytes);
    for fault in 1..=6 {
        let (destination, _) = pending(fault);
        let calls = Cell::new(0);
        let result = stage_trit_model_from_spool(
            &manifest,
            &blobs,
            &mut Cursor::new(&bytes),
            limits(),
            destination,
            || -> Result<(), u8> {
                calls.set(calls.get() + 1);
                Ok(())
            },
        );
        assert!(result.is_err(), "fault {fault}");
        assert_eq!(calls.get(), 0);
    }
    let (destination, observed) = pending(0);
    observed
        .borrow_mut()
        .write_all(b"existing pending bytes")
        .unwrap();
    assert!(matches!(
        stage_trit_model_from_spool(
            &manifest,
            &blobs,
            &mut Cursor::new(&bytes),
            limits(),
            destination,
            || -> Result<(), u8> { panic!("no guard on refusal") },
        ),
        Err(TritModelStageError::NonemptyDestination)
    ));
    assert_eq!(observed.borrow().get_ref(), b"existing pending bytes");
    for receiving in [
        TritPackageReadLimits {
            max_source_bytes: bytes.len() as u64 - 1,
            ..limits()
        },
        TritPackageReadLimits {
            max_manifest_bytes: 0,
            ..limits()
        },
        TritPackageReadLimits {
            max_directory_bytes: 0,
            ..limits()
        },
    ] {
        let (destination, observed) = pending(0);
        assert!(
            stage_trit_model_from_spool(
                &manifest,
                &blobs,
                &mut Cursor::new(&bytes),
                receiving,
                destination,
                || -> Result<(), u8> { panic!("no guard on refusal") },
            )
            .is_err()
        );
        if receiving.max_source_bytes < bytes.len() as u64 {
            assert!(observed.borrow().get_ref().is_empty());
        }
    }
}

#[test]
fn source_errors_unreferenced_payload_and_invalid_typed_metadata_cannot_run_guard() {
    let bytes = fixture::fixture(257, 16);
    let (mut manifest, mut blobs) = prepared(&bytes);
    for fault in [1, 4] {
        let (mut source, source_bytes) = pending(fault);
        *source_bytes.borrow_mut() = Cursor::new(bytes.clone());
        let (destination, output) = pending(0);
        assert!(matches!(
            stage_trit_model_from_spool(
                &manifest,
                &blobs,
                &mut source,
                limits(),
                destination,
                || -> Result<(), u8> { panic!("no guard on source I/O refusal") },
            ),
            Err(TritModelStageError::Write(TritPackageWriteError::Source(_)))
        ));
        assert!(output.borrow().get_ref().is_empty());
    }
    blobs[0].id = tritium_schema::BlobId::from_bytes([7; 32]);
    let result = stage_trit_model_from_spool(
        &manifest,
        &blobs,
        &mut Cursor::new(&bytes),
        limits(),
        Cursor::new(Vec::new()),
        || -> Result<(), u8> { panic!("no guard on source error") },
    );
    assert!(matches!(
        result,
        Err(TritModelStageError::Write(TritPackageWriteError::Package(
            _
        )))
    ));
    let (_, blobs) = prepared(&bytes);
    manifest.assets.remove(0);
    assert!(matches!(
        stage_trit_model_from_spool(
            &manifest,
            &blobs,
            &mut Cursor::new(&bytes),
            limits(),
            Cursor::new(Vec::new()),
            || -> Result<(), u8> { panic!("no guard on unreferenced payload") },
        ),
        Err(TritModelStageError::Model(_))
    ));
    manifest.levels[0].tensors[0].blob.id = tritium_schema::BlobId::from_bytes([9; 32]);
    assert!(
        stage_trit_model_from_spool(
            &manifest,
            &blobs,
            &mut Cursor::new(&bytes),
            limits(),
            Cursor::new(Vec::new()),
            || -> Result<(), u8> { panic!("no guard on absent typed binding") },
        )
        .is_err()
    );
}

#[test]
fn emitted_identity_guard_errors_and_post_guard_mutation_remain_fail_closed() {
    let bytes = fixture::fixture(257, 16);
    let (manifest, blobs) = prepared(&bytes);
    let (mut destination, _) = pending(0);
    destination.replacement = Some(fixture::fixture(258, 16));
    assert!(matches!(
        stage_trit_model_from_spool(
            &manifest,
            &blobs,
            &mut Cursor::new(&bytes),
            limits(),
            destination,
            || -> Result<(), u8> { panic!("no guard on replaced output") },
        ),
        Err(TritModelStageError::IdentityMismatch { .. })
    ));
    let (destination, _) = pending(0);
    assert!(matches!(
        stage_trit_model_from_spool(
            &manifest,
            &blobs,
            &mut Cursor::new(&bytes),
            limits(),
            destination,
            || Err(37u8),
        ),
        Err(TritModelStageError::Guard(37))
    ));
    for mutation in 0..3 {
        let (destination, observed) = pending(0);
        let result = stage_trit_model_from_spool(
            &manifest,
            &blobs,
            &mut Cursor::new(&bytes),
            limits(),
            destination,
            || -> Result<(), u8> {
                let mut output = observed.borrow_mut();
                match mutation {
                    0 => output.get_mut()[256] ^= 1,
                    1 => {
                        output.get_mut().pop();
                    }
                    _ => output.get_mut().push(0),
                }
                Ok(())
            },
        );
        assert!(matches!(result, Err(TritModelStageError::Read(_))));
    }
}

#[test]
fn actual_file_spool_and_pending_handle_support_caller_no_clobber_commit() {
    let root = std::env::temp_dir().join(format!("tritium-model-stage-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let cleanup = Cleanup(root.clone());
    let bytes = fixture::fixture(257, 131076);
    let (manifest, blobs) = prepared(&bytes);
    let spool_path = root.join("spool");
    let pending_path = root.join("pending.trit");
    let destination = root.join("model.trit");
    fs::write(&spool_path, &bytes).unwrap();
    let mut spool = File::open(spool_path).unwrap();
    let output = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&pending_path)
        .unwrap();
    let sync_handle = output.try_clone().unwrap();
    let mut reader = stage_trit_model_from_spool(
        &manifest,
        &blobs,
        &mut spool,
        limits(),
        output,
        || -> Result<(), ()> { Ok(()) },
    )
    .unwrap();
    sync_handle.sync_all().unwrap();
    fs::write(&destination, b"existing output sentinel").unwrap();
    assert_eq!(
        fs::hard_link(&pending_path, &destination)
            .unwrap_err()
            .kind(),
        io::ErrorKind::AlreadyExists
    );
    assert_eq!(fs::read(&destination).unwrap(), b"existing output sentinel");
    fs::remove_file(&destination).unwrap(); // only this test's own sentinel
    reader.verify().unwrap();
    fs::hard_link(&pending_path, &destination).unwrap();
    fs::remove_file(&pending_path).unwrap();
    reader.verify().unwrap(); // retained handle survives caller pathname changes
    assert_eq!(fs::read(&destination).unwrap(), bytes);
    drop(reader);
    drop(sync_handle);
    drop(spool);
    drop(cleanup);
    assert!(!root.exists());
    // Test's publication is not a production filesystem/durability/release gate.
}
