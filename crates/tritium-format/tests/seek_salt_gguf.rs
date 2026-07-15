use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use half::f16;
use tritium_core::Trit;
use tritium_format::{
    FormatError, GGML_TYPE_TQ2_0, GGML_TYPE_TRITIUM_SALT, GgufError, GgufValue,
    SALT_GGUF_FORMAT_KEY, SALT_GGUF_FORMAT_VALUE, SALT_HEADER_BYTES, SPARSE_HEADER_BYTES,
    SPARSE_MAGIC, SaltGgufReadError, SaltGgufReader, SaltRow, TQ2_0_BLOCK_BYTES, TensorOut,
    num_blocks, pack_progressive_salt_row, pack_salt_row, pack_tq2_0_row, read_gguf,
    read_salt_gguf_packed, unpack_packed_salt_row, write_gguf, write_salt_gguf,
};

fn metadata(alignment: u32) -> BTreeMap<String, GgufValue> {
    BTreeMap::from([
        ("general.alignment".to_owned(), GgufValue::U32(alignment)),
        (
            SALT_GGUF_FORMAT_KEY.to_owned(),
            GgufValue::String(SALT_GGUF_FORMAT_VALUE.to_owned()),
        ),
    ])
}

fn row(k: usize, planes: usize, seed: usize) -> SaltRow {
    let scales = vec![f16::from_f32(0.5); num_blocks(k)];
    let planes = (0..planes)
        .map(|plane| {
            let trits = (0..k)
                .map(|index| {
                    Trit::from_i8(((index + plane + seed) % 3) as i8 - 1).expect("valid trit")
                })
                .collect::<Vec<_>>();
            let mut bytes = vec![0; num_blocks(k) * TQ2_0_BLOCK_BYTES];
            pack_tq2_0_row(&trits, &scales, &mut bytes).expect("pack row");
            bytes
        })
        .collect();
    SaltRow { k, planes }
}

fn progressive_sparse_row_payload() -> Vec<u8> {
    let base = row(256, 1, 0).planes.pop().expect("base plane");
    let zero = Trit::from_i8(0).expect("zero trit");
    let mut residual_trits = vec![zero; 256];
    residual_trits[7] = Trit::from_i8(1).expect("positive trit");
    let mut residual = vec![0; TQ2_0_BLOCK_BYTES];
    pack_tq2_0_row(&residual_trits, &[f16::from_f32(0.5)], &mut residual)
        .expect("pack sparse residual");
    let packed = pack_progressive_salt_row(
        &SaltRow {
            k: 256,
            planes: vec![base, residual],
        },
        1.0,
    )
    .expect("pack progressive row");
    assert_eq!(
        unpack_packed_salt_row(&packed)
            .expect("parse progressive row")
            .sparse_plane_count(),
        1
    );
    packed
}

fn write_private_salt_payload(name: &str, k: u64, rows: u64, payload: &[u8]) -> Vec<u8> {
    write_gguf(
        3,
        &metadata(32),
        &[TensorOut {
            name: name.to_owned(),
            dims: vec![k, rows],
            ggml_type: GGML_TYPE_TRITIUM_SALT,
            data: payload,
        }],
    )
    .expect("write private SALT payload")
}

fn mutate_tensor_type(
    bytes: &mut [u8],
    name: &str,
    dims: &[u64],
    original_type: u32,
    replacement_type: u32,
) {
    let mut record_prefix = Vec::new();
    push_string(&mut record_prefix, name);
    record_prefix.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for dimension in dims {
        record_prefix.extend_from_slice(&dimension.to_le_bytes());
    }
    record_prefix.extend_from_slice(&original_type.to_le_bytes());
    let matches = bytes
        .windows(record_prefix.len())
        .enumerate()
        .filter_map(|(offset, window)| (window == record_prefix).then_some(offset))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "tensor record prefix must be unique");
    let type_offset = matches[0] + record_prefix.len() - size_of::<u32>();
    bytes[type_offset..type_offset + size_of::<u32>()]
        .copy_from_slice(&replacement_type.to_le_bytes());
}

#[test]
fn indexes_and_visits_without_owned_tensor_decode() {
    let first = vec![row(269, 2, 0), row(269, 1, 1)];
    let second = vec![row(512, 3, 7)];
    let bytes = write_salt_gguf(&[("a", &first), ("b", &second)]).expect("write");
    let mut reader = SaltGgufReader::new_strict(Cursor::new(bytes)).expect("strict reader");

    assert_eq!(reader.tensor_names().collect::<Vec<_>>(), ["a", "b"]);
    let info = reader.tensor_info("a").expect("tensor info");
    assert_eq!(info.shape(), (2, 269));
    assert_eq!(info.storage_requirements().rows(), 2);
    assert_eq!(info.storage_requirements().planes(), 3);

    let mut plane_counts = Vec::new();
    reader
        .visit_packed_tensor("a", |row| plane_counts.push(row.plane_count()))
        .expect("visit");
    assert_eq!(plane_counts, [2, 1]);
}

#[test]
fn rejects_corruption_in_an_unselected_salt_tensor() {
    let first = vec![row(256, 1, 0)];
    let second = vec![row(256, 1, 1)];
    let mut bytes =
        write_salt_gguf(&[("selected", &first), ("unselected", &second)]).expect("write");
    let gguf = read_gguf(&bytes).expect("GGUF envelope");
    let unselected = gguf.tensor("unselected").expect("unselected table entry");
    let payload = usize::try_from(gguf.tensor_data_offset + unselected.offset).expect("offset");
    bytes[payload] ^= 0xff;

    assert!(SaltGgufReader::new_strict(Cursor::new(bytes)).is_err());
}

#[test]
fn validates_and_ignores_sized_standard_tensors() {
    let salt_row = row(256, 1, 0);
    let salt_payload = pack_salt_row(&salt_row).expect("pack");
    let norm_payload = [1.0f32, 2.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let scalar_payload = 3.0f32.to_le_bytes();
    let metadata = metadata(32);
    let tensors = [
        TensorOut {
            name: "scalar".to_owned(),
            dims: Vec::new(),
            ggml_type: 0,
            data: &scalar_payload,
        },
        TensorOut {
            name: "norm".to_owned(),
            dims: vec![2],
            ggml_type: 0,
            data: &norm_payload,
        },
        TensorOut {
            name: "weight".to_owned(),
            dims: vec![256, 1],
            ggml_type: GGML_TYPE_TRITIUM_SALT,
            data: &salt_payload,
        },
    ];
    let bytes = write_gguf(3, &metadata, &tensors).expect("write mixed GGUF");

    let reader = SaltGgufReader::new_strict(Cursor::new(bytes)).expect("strict reader");
    assert_eq!(reader.tensor_names().collect::<Vec<_>>(), ["weight"]);
}

#[test]
fn alignment_24_matches_the_eager_reader() {
    let salt_rows = [row(269, 2, 3), row(269, 1, 4)];
    let payload = salt_rows
        .iter()
        .flat_map(|row| pack_salt_row(row).expect("pack"))
        .collect::<Vec<_>>();
    let tensor = TensorOut {
        name: "weight".to_owned(),
        dims: vec![269, 2],
        ggml_type: GGML_TYPE_TRITIUM_SALT,
        data: &payload,
    };
    let bytes = write_gguf(3, &metadata(24), &[tensor]).expect("write alignment-24 GGUF");
    let eager = read_salt_gguf_packed(&bytes).expect("eager reader");
    let mut seek = SaltGgufReader::new_strict(Cursor::new(bytes)).expect("seek reader");
    let mut streamed = Vec::new();
    seek.visit_packed_tensor("weight", |row| {
        streamed.push(unpack_packed_salt_row(row.encoded_bytes()).expect("owned row"));
    })
    .expect("visit");

    assert_eq!(eager.len(), 1);
    assert_eq!(eager[0].salt_rows, streamed);
}

#[test]
fn rejects_unselected_nonfinite_and_negative_scales() {
    let selected = vec![row(256, 1, 0)];
    let unselected = vec![row(256, 1, 1)];
    let original =
        write_salt_gguf(&[("selected", &selected), ("unselected", &unselected)]).expect("write");
    let gguf = read_gguf(&original).expect("GGUF envelope");
    let tensor = gguf.tensor("unselected").expect("unselected tensor");
    let scale_offset = usize::try_from(gguf.tensor_data_offset + tensor.offset).expect("offset")
        + SALT_HEADER_BYTES
        + TQ2_0_BLOCK_BYTES
        - 2;

    for bits in [
        f16::NAN.to_bits(),
        f16::INFINITY.to_bits(),
        f16::from_bits(0x8000).to_bits(),
        f16::from_f32(-0.5).to_bits(),
    ] {
        let mut bytes = original.clone();
        bytes[scale_offset..scale_offset + 2].copy_from_slice(&bits.to_le_bytes());
        assert!(matches!(
            SaltGgufReader::new_strict(Cursor::new(bytes)),
            Err(SaltGgufReadError::InvalidScale(actual)) if actual == bits
        ));
    }
}

#[test]
fn rejects_negative_zero_in_a_sparse_residual_scale() {
    let payload = progressive_sparse_row_payload();
    let sparse_offset = payload
        .windows(SPARSE_MAGIC.len())
        .position(|window| window == SPARSE_MAGIC)
        .expect("sparse sidecar");
    let mut bytes = write_private_salt_payload("weight", 256, 1, &payload);
    let row_offset = bytes
        .windows(payload.len())
        .position(|window| window == payload)
        .expect("progressive row payload");
    let scale_offset = row_offset + sparse_offset + SPARSE_HEADER_BYTES;
    bytes[scale_offset..scale_offset + 2].copy_from_slice(&0x8000u16.to_le_bytes());

    assert!(matches!(
        SaltGgufReader::new_strict(Cursor::new(bytes)),
        Err(SaltGgufReadError::InvalidScale(0x8000))
    ));
}

#[test]
fn rejects_truncated_huge_tensor_table_before_reserving_it() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1_000_000u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    push_string(&mut bytes, SALT_GGUF_FORMAT_KEY);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut bytes, SALT_GGUF_FORMAT_VALUE);

    assert!(matches!(
        SaltGgufReader::new_strict(Cursor::new(bytes)),
        Err(SaltGgufReadError::Format(FormatError::Gguf(
            GgufError::Truncated
        )))
    ));
}

#[test]
fn accepts_nested_metadata_arrays_and_rejects_invalid_bool_bytes() {
    let nested = empty_salt_gguf_with_extra("nested", 9, |bytes| {
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&2u64.to_le_bytes());
        bytes.extend_from_slice(&7u32.to_le_bytes());
        bytes.extend_from_slice(&11u32.to_le_bytes());
    });
    let eager = read_gguf(&nested).expect("eager nested array");
    assert!(matches!(
        eager.get_metadata("nested"),
        Some(GgufValue::Array(outer))
            if matches!(outer.as_slice(), [GgufValue::Array(inner)] if inner == &[GgufValue::U32(7), GgufValue::U32(11)])
    ));
    assert!(
        SaltGgufReader::new_strict(Cursor::new(nested))
            .expect("nested array")
            .is_empty()
    );

    let invalid_bool = empty_salt_gguf_with_extra("bad_bool", 7, |bytes| bytes.push(2));
    assert!(matches!(
        read_gguf(&invalid_bool),
        Err(GgufError::InvalidBoolean(2))
    ));
    assert!(SaltGgufReader::new_strict(Cursor::new(invalid_bool)).is_err());
}

#[test]
fn eager_and_seek_readers_bound_nested_metadata_elements() {
    let oversized = empty_salt_gguf_with_extra("oversized", 9, |bytes| {
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&16_000_001u64.to_le_bytes());
    });
    assert!(matches!(
        read_gguf(&oversized),
        Err(GgufError::DimsOverflow)
    ));
    assert!(SaltGgufReader::new_strict(Cursor::new(oversized)).is_err());
}

#[test]
fn declared_alignment_must_be_nonzero_u32_and_multiple_of_eight() {
    let invalid = [
        (4, 0u32.to_le_bytes().to_vec()),
        (4, 12u32.to_le_bytes().to_vec()),
        (5, (-8i32).to_le_bytes().to_vec()),
        (10, 32u64.to_le_bytes().to_vec()),
    ];
    for (value_type, payload) in invalid {
        let bytes = empty_salt_gguf_with_alignment(value_type, &payload);
        assert!(matches!(
            read_gguf(&bytes),
            Err(GgufError::InvalidAlignment)
        ));
        assert!(matches!(
            SaltGgufReader::new_strict(Cursor::new(bytes)),
            Err(SaltGgufReadError::Format(FormatError::Gguf(
                GgufError::InvalidAlignment
            )))
        ));
    }
}

#[test]
fn rejects_quantized_standard_tensors_whose_rows_are_not_block_aligned() {
    let quantized = vec![0; TQ2_0_BLOCK_BYTES];
    let salt_row = row(256, 1, 0);
    let salt = pack_salt_row(&salt_row).expect("pack SALT row");
    let tensors = [
        TensorOut {
            name: "bad.quantized".to_owned(),
            dims: vec![128, 2],
            ggml_type: 169,
            data: &quantized,
        },
        TensorOut {
            name: "weight".to_owned(),
            dims: vec![256, 1],
            ggml_type: GGML_TYPE_TRITIUM_SALT,
            data: &salt,
        },
    ];
    let mut bytes = write_gguf(3, &metadata(32), &tensors).expect("write custom fixture");
    mutate_tensor_type(&mut bytes, "bad.quantized", &[128, 2], 169, GGML_TYPE_TQ2_0);
    assert!(matches!(
        read_gguf(&bytes),
        Err(GgufError::InvalidTensorShape)
    ));
    assert!(matches!(
        SaltGgufReader::new_strict(Cursor::new(bytes)),
        Err(SaltGgufReadError::Format(FormatError::Gguf(
            GgufError::InvalidTensorShape
        )))
    ));
}

#[test]
fn caps_each_underlying_read_at_64_kib() {
    let rows = vec![row(300_000, 1, 0)];
    let bytes = write_salt_gguf(&[("large", &rows)]).expect("write");
    let maximum = Arc::new(AtomicUsize::new(0));
    let source = CappedCursor {
        inner: Cursor::new(bytes),
        maximum: Arc::clone(&maximum),
    };
    let mut reader = SaltGgufReader::new_strict(source).expect("strict reader");
    reader
        .visit_packed_tensor("large", |_| {})
        .expect("visit large row");
    assert!(maximum.load(Ordering::Relaxed) <= 64 * 1024);
}

#[test]
fn selected_tensor_digest_detects_valid_same_length_mutation() {
    let rows = vec![row(256, 1, 0)];
    let bytes = write_salt_gguf(&[("w", &rows)]).expect("write");
    let encoded = pack_salt_row(&rows[0]).expect("pack");
    let row_start = bytes
        .windows(encoded.len())
        .position(|window| window == encoded)
        .expect("row payload");
    let shared = Rc::new(RefCell::new(bytes));
    let mut reader = SaltGgufReader::new_strict(SharedCursor {
        bytes: Rc::clone(&shared),
        position: 0,
    })
    .expect("strict reader");
    shared.borrow_mut()[row_start + 10 + 64] ^= 1;

    let visited = Rc::new(Cell::new(false));
    let visited_in_callback = Rc::clone(&visited);
    let error = reader
        .visit_packed_tensor("w", |_| visited_in_callback.set(true))
        .expect_err("digest mismatch");
    assert!(visited.get(), "sink mutation precedes digest verification");
    assert!(matches!(error, SaltGgufReadError::SourceChanged(name) if name == "w"));
}

#[test]
fn visit_rejects_scale_mutated_to_negative_zero_after_construction() {
    let rows = vec![row(256, 1, 0)];
    let bytes = write_salt_gguf(&[("w", &rows)]).expect("write");
    let encoded = pack_salt_row(&rows[0]).expect("pack");
    let row_start = bytes
        .windows(encoded.len())
        .position(|window| window == encoded)
        .expect("row payload");
    let shared = Rc::new(RefCell::new(bytes));
    let mut reader = SaltGgufReader::new_strict(SharedCursor {
        bytes: Rc::clone(&shared),
        position: 0,
    })
    .expect("strict reader");
    let scale_offset = row_start + SALT_HEADER_BYTES + TQ2_0_BLOCK_BYTES - 2;
    shared.borrow_mut()[scale_offset..scale_offset + 2].copy_from_slice(&0x8000u16.to_le_bytes());

    let visited = Cell::new(false);
    let error = reader
        .visit_packed_tensor("w", |_| visited.set(true))
        .expect_err("negative-zero scale mutation");
    assert!(!visited.get(), "scale validation precedes the callback");
    assert!(matches!(error, SaltGgufReadError::InvalidScale(0x8000)));
}

fn push_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn empty_salt_gguf_with_extra(
    key: &str,
    value_type: u32,
    value: impl FnOnce(&mut Vec<u8>),
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&3u64.to_le_bytes());

    push_string(&mut bytes, "general.alignment");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut bytes, SALT_GGUF_FORMAT_KEY);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut bytes, SALT_GGUF_FORMAT_VALUE);
    push_string(&mut bytes, key);
    bytes.extend_from_slice(&value_type.to_le_bytes());
    value(&mut bytes);
    let padding = (8 - bytes.len() % 8) % 8;
    bytes.resize(bytes.len() + padding, 0);
    bytes
}

fn empty_salt_gguf_with_alignment(value_type: u32, payload: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());
    push_string(&mut bytes, "general.alignment");
    bytes.extend_from_slice(&value_type.to_le_bytes());
    bytes.extend_from_slice(payload);
    push_string(&mut bytes, SALT_GGUF_FORMAT_KEY);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    push_string(&mut bytes, SALT_GGUF_FORMAT_VALUE);
    let padding = (8 - bytes.len() % 8) % 8;
    bytes.resize(bytes.len() + padding, 0);
    bytes
}

struct CappedCursor {
    inner: Cursor<Vec<u8>>,
    maximum: Arc<AtomicUsize>,
}

impl Read for CappedCursor {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        assert!(bytes.len() <= 64 * 1024, "oversized read: {}", bytes.len());
        self.maximum.fetch_max(bytes.len(), Ordering::Relaxed);
        self.inner.read(bytes)
    }
}

impl Seek for CappedCursor {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.inner.seek(position)
    }
}

#[derive(Clone)]
struct SharedCursor {
    bytes: Rc<RefCell<Vec<u8>>>,
    position: u64,
}

impl Read for SharedCursor {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let bytes = self.bytes.borrow();
        let start = usize::try_from(self.position).expect("position");
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
