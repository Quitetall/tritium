//! Real retained-source scenarios through the public typed reader.
use std::cell::{Cell, RefCell};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::rc::Rc;
use tritium_format::TritPackageReadError;
use tritium_format::trit_model::{TritModelReadError, TritModelReader, TritModelVisitError};
#[path = "support/typed_seek_fixture.rs"]
mod fixture;
#[path = "support/seek_model_parity.rs"]
mod parity;

struct Source {
    bytes: Rc<RefCell<Vec<u8>>>,
    position: usize,
    full_reads: usize,
    scan: bool,
    max_read: Rc<Cell<usize>>,
    fault: Rc<Cell<u8>>,
    mutate_on_terminal: bool,
}

#[test]
fn named_tensor_stream_borrows_exact_metadata_repeats_and_preserves_faults() {
    let bytes = fixture::fixture(65537, 131076);
    let source = Source::new(bytes.clone());
    let mutable = source.bytes.clone();
    let limits = parity::limits(&bytes);
    let mut reader = TritModelReader::open(source, limits).unwrap();
    let pointer = reader.tensor_meta("child", "weight").unwrap() as *const _;
    let expected = {
        let borrowed = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
        borrowed.tensor("child", "weight").unwrap().1.to_vec()
    };
    {
        let mut stream = reader.tensor_stream("child", "weight").unwrap();
        assert_eq!(stream.metadata() as *const _, pointer);
        assert_eq!(stream.encoded_bytes(), expected.len() as u64);
        for _ in 0..2 {
            let mut actual = Vec::new();
            stream
                .visit(|chunk| -> Result<(), u8> {
                    assert!(chunk.len() <= 65536);
                    actual.extend_from_slice(chunk);
                    Ok(())
                })
                .unwrap();
            assert_eq!(actual, expected);
        }
        assert!(matches!(
            stream.visit(|_| Err(37u8)),
            Err(TritModelVisitError::Visitor(37))
        ));
        let raw = mutable
            .borrow()
            .windows(expected.len())
            .position(|s| s == expected)
            .unwrap();
        mutable.borrow_mut()[raw] ^= 1;
        assert!(stream.visit(|_| Ok::<_, u8>(())).is_err());
    }
    assert!(reader.verify().is_err());
}
impl Source {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes: Rc::new(RefCell::new(bytes)),
            position: 0,
            full_reads: 0,
            scan: false,
            max_read: Rc::new(Cell::new(0)),
            fault: Rc::new(Cell::new(0)),
            mutate_on_terminal: false,
        }
    }
}
impl Read for Source {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        self.max_read.set(self.max_read.get().max(out.len()));
        if self.position == 0 && out.len() == self.bytes.borrow().len().min(64 * 1024) {
            self.full_reads += 1;
            self.scan = true;
            if self.mutate_on_terminal && self.full_reads == 3 {
                self.bytes.borrow_mut()[256] ^= 1;
            }
        }
        if self.full_reads >= 2 && (!self.scan || self.full_reads >= 3) && self.fault.get() == 1 {
            return Err(io::Error::other("injected typed read"));
        }
        let bytes = self.bytes.borrow();
        if self.position >= bytes.len() {
            return Ok(0);
        }
        let count = out.len().min(bytes.len() - self.position).min(13);
        out[..count].copy_from_slice(&bytes[self.position..self.position + count]);
        self.position += count;
        if self.position == bytes.len() {
            self.scan = false;
        }
        Ok(count)
    }
}
impl Seek for Source {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        if self.full_reads >= 2 && matches!(from, SeekFrom::Start(n) if n >= 256) {
            if self.fault.get() == 2 {
                return Err(io::Error::other("injected typed seek"));
            }
            if self.fault.get() == 3
                && let SeekFrom::Start(n) = from
            {
                return Ok(n + 1);
            }
        }
        let at = match from {
            SeekFrom::Start(n) => i128::from(n),
            SeekFrom::End(n) => self.bytes.borrow().len() as i128 + i128::from(n),
            SeekFrom::Current(n) => self.position as i128 + i128::from(n),
        };
        self.position =
            usize::try_from(at).map_err(|_| io::Error::other("bad synthetic offset"))?;
        Ok(self.position as u64)
    }
}

#[test]
fn admitted_files_and_large_cache_crossings_keep_exact_typed_identity_and_visits() {
    let bytes = fixture::fixture(65537, 131076);
    parity::assert_matches(&bytes);
    let path = std::env::temp_dir().join(format!(
        "tritium-typed-seek-file-{}.trit",
        std::process::id()
    ));
    let mut destination = File::create_new(&path).unwrap();
    std::io::Write::write_all(&mut destination, &bytes).unwrap();
    drop(destination);
    let mut reader =
        TritModelReader::open(File::open(&path).unwrap(), parity::limits(&bytes)).unwrap();
    let expected = tritium_format::trit_model::read_trit_model(&bytes).unwrap();
    reader.validate_complete_payload_inventory().unwrap();
    for level in ["child", "parent"] {
        assert_eq!(
            reader.model_id(level).unwrap(),
            expected.model_id(level).unwrap()
        );
        for name in ["norm.weight", "weight"] {
            let (_, raw) = expected.tensor(level, name).unwrap();
            let mut at = 0;
            reader
                .visit_tensor(level, name, |chunk| -> Result<(), ()> {
                    assert_eq!(chunk, &raw[at..at + chunk.len()]);
                    at += chunk.len();
                    Ok(())
                })
                .unwrap();
            assert_eq!(at, raw.len());
        }
    }
    reader.verify().unwrap();
    drop(reader);
    fs::remove_file(path).unwrap();
}

#[test]
fn short_reads_and_original_visitor_failures_use_the_same_named_model_seam() {
    let bytes = fixture::fixture(257, 68);
    let source = Source::new(bytes.clone());
    let max_read = source.max_read.clone();
    let mut reader = TritModelReader::open(source, parity::limits(&bytes)).unwrap();
    assert!(max_read.get() <= 64 * 1024);
    assert!(matches!(
        reader.visit_tensor("child", "weight", |_| Err("visitor sentinel")),
        Err(TritModelVisitError::Visitor("visitor sentinel"))
    ));
    assert!(matches!(
        reader.visit_asset("config.json", |_| Err("asset sentinel")),
        Err(TritModelVisitError::Visitor("asset sentinel"))
    ));
    assert!(matches!(
        reader.visit_tensor("missing", "weight", |_| Ok::<_, ()>(())),
        Err(TritModelVisitError::Read(TritModelReadError::Model(_)))
    ));
    assert!(reader.model_id("missing").is_err());
    assert!(reader.tensor_meta("child", "missing").is_err());
    assert!(reader.tensor_digest("child", "missing").is_err());
    assert!(reader.tensor_byte_len("child", "missing").is_err());
    assert!(reader.tensor_byte_len("missing", "weight").is_err());
    assert!(reader.asset_byte_len("absent.json").is_err());
    reader.verify().unwrap();
}

#[test]
fn typed_range_io_and_wrong_seek_after_package_admission_cannot_return_a_model() {
    let bytes = fixture::fixture(257, 68);
    for fault in [1, 2, 3] {
        let source = Source::new(bytes.clone());
        source.fault.set(fault);
        assert!(matches!(
            TritModelReader::open(source, parity::limits(&bytes)),
            Err(TritModelReadError::Source(TritPackageReadError::Io { .. }))
        ));
    }
}

#[test]
fn explicit_source_caps_refuse_before_body_reads_or_typed_payload_validation() {
    let bytes = fixture::fixture(257, 68);
    let source = Source::new(bytes.clone());
    let max_read = source.max_read.clone();
    let mut limits = parity::limits(&bytes);
    limits.max_source_bytes -= 1;
    assert!(matches!(
        TritModelReader::open(source, limits),
        Err(TritModelReadError::Source(
            TritPackageReadError::SourceLimitExceeded { .. }
        ))
    ));
    assert_eq!(max_read.get(), 0);
}

#[test]
fn final_full_package_reverification_refuses_same_length_mutation() {
    let bytes = fixture::fixture(257, 68);
    let mut source = Source::new(bytes.clone());
    source.mutate_on_terminal = true;
    assert!(matches!(
        TritModelReader::open(source, parity::limits(&bytes)),
        Err(TritModelReadError::Source(
            TritPackageReadError::SourceChanged
        ))
    ));
}

#[test]
fn later_blob_mutation_truncation_growth_and_read_faults_fail_before_terminal_success() {
    let bytes = fixture::fixture(257, 68);
    for change in 0..4 {
        let source = Source::new(bytes.clone());
        let data = source.bytes.clone();
        let fault = source.fault.clone();
        let mut reader = TritModelReader::open(source, parity::limits(&bytes)).unwrap();
        match change {
            0 => {
                let package = tritium_format::read_trit_package(&bytes).unwrap();
                let raw = reader.tensor_meta("child", "weight").unwrap().blob.id;
                let offset = package.blobs().iter().find(|b| b.id == raw).unwrap().offset as usize;
                data.borrow_mut()[offset] ^= 1;
            }
            1 => {
                data.borrow_mut().pop();
            }
            2 => data.borrow_mut().push(0),
            _ => fault.set(1),
        }
        assert!(matches!(
            reader.visit_tensor("child", "weight", |_| Ok::<_, ()>(())),
            Err(TritModelVisitError::Read(_))
        ));
        assert!(reader.verify().is_err());
    }
}

#[cfg(unix)]
#[test]
fn retained_file_is_not_replaced_by_a_different_pathname_target() {
    let bytes = fixture::fixture(257, 68);
    let path = std::env::temp_dir().join(format!(
        "tritium-typed-seek-pin-{}.trit",
        std::process::id()
    ));
    let old = path.with_extension("retained");
    let mut file = File::create_new(&path).unwrap();
    std::io::Write::write_all(&mut file, &bytes).unwrap();
    drop(file);
    let mut reader =
        TritModelReader::open(File::open(&path).unwrap(), parity::limits(&bytes)).unwrap();
    fs::rename(&path, &old).unwrap();
    let mut file = File::create_new(&path).unwrap();
    std::io::Write::write_all(&mut file, b"replacement").unwrap();
    drop(file);
    reader.verify().unwrap();
    let mut seen = Vec::new();
    reader
        .visit_asset("config.json", |chunk| -> Result<(), ()> {
            seen.extend_from_slice(chunk);
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, b"{}");
    drop(reader);
    fs::remove_file(path).unwrap();
    fs::remove_file(old).unwrap();
}

#[test]
fn complete_model_refusals_match_under_header_metadata_payload_and_prefix_mutations() {
    let bytes = fixture::fixture(257, 68);
    for cut in 0..bytes.len() {
        parity::assert_matches(&bytes[..cut]);
    }
    for index in 0..1024 {
        let mut changed = bytes.clone();
        let at = index * 7919 % changed.len();
        changed[at] ^= (index % 255 + 1) as u8;
        parity::assert_matches(&changed);
    }
    eprintln!("TYPED_SEEK_PACKAGE_REFUSALS cases={}", bytes.len() + 1024);
}
