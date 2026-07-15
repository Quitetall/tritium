use std::cell::{Cell, RefCell};
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::rc::Rc;

use half::f16;
use tritium_core::Trit;
use tritium_format::{
    DEFAULT_SPARSE_RESIDUAL_DENSITY, PackedSaltRow, SaltBundleIndex, SaltBundleReadError,
    SaltBundleReader, SaltRow, TQ2_0_BLOCK_BYTES, num_blocks, pack_salt_row, pack_tq2_0_row,
    unpack_packed_salt_row, write_progressive_salt_bundle, write_salt_bundle,
};

fn dense_plane(k: usize, seed: usize, sparse_stride: Option<usize>) -> Vec<u8> {
    let trits = (0..k)
        .map(|index| {
            let value = match sparse_stride {
                Some(stride) if !index.is_multiple_of(stride) => 0,
                _ => ((index + seed) % 3) as i8 - 1,
            };
            Trit::from_i8(value).unwrap()
        })
        .collect::<Vec<_>>();
    let scales = (0..num_blocks(k))
        .map(|block| f16::from_f32(0.25 + (block + seed) as f32 * 0.01))
        .collect::<Vec<_>>();
    let mut bytes = vec![0; num_blocks(k) * TQ2_0_BLOCK_BYTES];
    pack_tq2_0_row(&trits, &scales, &mut bytes).unwrap();
    bytes
}

fn row(k: usize, seed: usize, sparse_residual: bool) -> SaltRow {
    let mut planes = vec![dense_plane(k, seed, None)];
    if sparse_residual {
        planes.push(dense_plane(k, seed + 1, Some(64)));
    }
    SaltRow { k, planes }
}

struct ShortTrackedReader {
    cursor: Cursor<Vec<u8>>,
    max_request: Rc<Cell<usize>>,
    chunk: usize,
}

impl Read for ShortTrackedReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.max_request
            .set(self.max_request.get().max(buffer.len()));
        let len = buffer.len().min(self.chunk);
        self.cursor.read(&mut buffer[..len])
    }
}

impl Seek for ShortTrackedReader {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.cursor.seek(position)
    }
}

#[test]
fn strict_seek_reader_streams_dense_sparse_rows_in_arbitrary_order() {
    let a = vec![row(300, 1, true), row(300, 2, false)];
    let b = vec![row(512, 7, true)];
    let zero = vec![SaltRow {
        k: 300,
        planes: Vec::new(),
    }];
    let refs = [
        ("b", b.as_slice()),
        ("zero", zero.as_slice()),
        ("a", a.as_slice()),
    ];
    let bytes = write_progressive_salt_bundle(&refs, DEFAULT_SPARSE_RESIDUAL_DENSITY).unwrap();
    let borrowed = SaltBundleIndex::new(&bytes).unwrap();
    let expected_a = borrowed.tensor("a").unwrap().decode_packed().unwrap();
    let expected_b = borrowed.tensor("b").unwrap().decode_packed().unwrap();
    let max_request = Rc::new(Cell::new(0));
    let source = ShortTrackedReader {
        cursor: Cursor::new(bytes),
        max_request: Rc::clone(&max_request),
        chunk: 7,
    };
    let mut reader = SaltBundleReader::new_strict(source).unwrap();

    assert_eq!(
        reader.tensor_names().collect::<Vec<_>>(),
        ["a", "b", "zero"]
    );
    assert_eq!(reader.tensor_info("a").unwrap().shape(), (2, 300));
    assert_eq!(reader.tensor_info("b").unwrap().shape(), (1, 512));
    let requirements = reader.tensor_info("a").unwrap().storage_requirements();
    assert_eq!(requirements.rows(), 2);
    assert_eq!(requirements.planes(), 3);
    assert_eq!(requirements.sparse_planes(), 1);

    for (name, expected) in [
        ("b", expected_b.salt_rows.as_slice()),
        ("a", expected_a.salt_rows.as_slice()),
        ("b", expected_b.salt_rows.as_slice()),
    ] {
        let mut decoded = Vec::<PackedSaltRow>::new();
        reader
            .visit_packed_tensor(name, |row| {
                decoded.push(unpack_packed_salt_row(row.encoded_bytes()).unwrap());
            })
            .unwrap();
        assert_eq!(decoded, expected);
    }
    assert!(
        max_request.get() <= 64 * 1024,
        "largest Read request was {} bytes",
        max_request.get()
    );
}

#[test]
fn malformed_unselected_tensor_fails_strict_construction() {
    let a = vec![row(256, 1, false)];
    let b = vec![row(256, 8, false)];
    let mut bytes = write_salt_bundle(&[("a", &a), ("b", &b)]).unwrap();
    let encoded_b = pack_salt_row(&b[0]).unwrap();
    let start = bytes
        .windows(encoded_b.len())
        .position(|window| window == encoded_b)
        .unwrap();
    bytes[start + 10] = 0xff;
    assert!(SaltBundleReader::new_strict(Cursor::new(bytes)).is_err());
}

#[derive(Clone)]
struct SharedCursor {
    bytes: Rc<RefCell<Vec<u8>>>,
    position: u64,
}

impl Read for SharedCursor {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let bytes = self.bytes.borrow();
        let start = usize::try_from(self.position).unwrap();
        let len = output.len().min(bytes.len().saturating_sub(start));
        output[..len].copy_from_slice(&bytes[start..start + len]);
        self.position += len as u64;
        Ok(len)
    }
}

impl Seek for SharedCursor {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        let len = self.bytes.borrow().len() as i128;
        let next = match position {
            SeekFrom::Start(position) => position as i128,
            SeekFrom::End(delta) => len + delta as i128,
            SeekFrom::Current(delta) => self.position as i128 + delta as i128,
        };
        if !(0..=u64::MAX as i128).contains(&next) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid seek",
            ));
        }
        self.position = next as u64;
        Ok(self.position)
    }
}

#[test]
fn selected_tensor_digest_detects_valid_same_length_mutation() {
    let rows = vec![row(256, 1, false)];
    let bytes = write_salt_bundle(&[("w", &rows)]).unwrap();
    let encoded = pack_salt_row(&rows[0]).unwrap();
    let row_start = bytes
        .windows(encoded.len())
        .position(|window| window == encoded)
        .unwrap();
    let shared = Rc::new(RefCell::new(bytes));
    let mut reader = SaltBundleReader::new_strict(SharedCursor {
        bytes: Rc::clone(&shared),
        position: 0,
    })
    .unwrap();
    shared.borrow_mut()[row_start + 10 + 64] ^= 1;

    let visited = Rc::new(Cell::new(false));
    let visited_in_callback = Rc::clone(&visited);
    let error = reader
        .visit_packed_tensor("w", |_| visited_in_callback.set(true))
        .unwrap_err();
    assert!(
        visited.get(),
        "sink mutation precedes final digest verification"
    );
    assert!(matches!(error, SaltBundleReadError::SourceChanged(name) if name == "w"));
}

#[test]
fn missing_and_trailing_data_fail_closed() {
    let rows = vec![row(256, 1, false)];
    let bytes = write_salt_bundle(&[("w", &rows)]).unwrap();
    let mut reader = SaltBundleReader::new_strict(Cursor::new(bytes.clone())).unwrap();
    assert!(matches!(
        reader.visit_packed_tensor("absent", |_| {}),
        Err(SaltBundleReadError::TensorNotFound(name)) if name == "absent"
    ));

    let mut trailing = bytes;
    trailing.push(0);
    assert!(SaltBundleReader::new_strict(Cursor::new(trailing)).is_err());
}

#[test]
fn impossible_index_count_fails_before_metadata_reservation() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"TSLB");
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&1_000_000u32.to_le_bytes());
    let error = SaltBundleReader::new_strict(Cursor::new(bytes)).unwrap_err();
    assert!(matches!(error, SaltBundleReadError::Format(_)));
}

#[test]
fn model_reader_reports_its_plane_policy_boundary() {
    let rows = vec![SaltRow {
        k: 256,
        planes: (0..9).map(|seed| dense_plane(256, seed, None)).collect(),
    }];
    let bytes = write_salt_bundle(&[("w", &rows)]).unwrap();
    SaltBundleIndex::new(&bytes).unwrap();
    let error = SaltBundleReader::new_strict(Cursor::new(bytes)).unwrap_err();
    assert!(matches!(
        error,
        SaltBundleReadError::LimitExceeded { resource, limit: 8, actual: 9 }
            if resource == "planes per row"
    ));
}

#[derive(Debug)]
struct LyingStartSeek {
    cursor: Cursor<Vec<u8>>,
}

impl Read for LyingStartSeek {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        self.cursor.read(output)
    }
}

impl Seek for LyingStartSeek {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        if position == SeekFrom::Start(0) {
            Ok(1)
        } else {
            self.cursor.seek(position)
        }
    }
}

#[test]
fn lying_initial_seek_is_a_typed_io_error() {
    let error = SaltBundleReader::new_strict(LyingStartSeek {
        cursor: Cursor::new(Vec::new()),
    })
    .unwrap_err();
    assert!(matches!(
        error,
        SaltBundleReadError::Io {
            kind: std::io::ErrorKind::InvalidData,
            ..
        }
    ));
}
