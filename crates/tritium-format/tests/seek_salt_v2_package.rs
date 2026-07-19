use std::cell::{Cell, RefCell};
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::rc::Rc;

use half::f16;
use tritium_format::PackageId;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    PackedSaltV2PlaneRef, SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_PACKAGE_MAGIC,
    SALT_V2_PACKAGE_VERSION, SaltV2IndexedRuntimeLedger, SaltV2Package, SaltV2PackageError,
    SaltV2PackageReadError, SaltV2PackageReader, SaltV2Plane, SaltV2Tensor, SaltV2Tile,
    pack_salt_v2_plane, read_salt_v2_package, unpack_salt_v2_plane, write_salt_v2_package,
};

fn plane(len: usize, seed: usize) -> SaltV2Plane {
    let trits = (0..len)
        .map(|index| ((index * 5 + seed) % 3) as i8 - 1)
        .collect::<Vec<_>>();
    let scales = (0..len.div_ceil(128))
        .map(|group| f16::from_f32(0.25 + (seed + group) as f32 / 32.0))
        .collect();
    SaltV2Plane::new(trits, scales).unwrap()
}

fn tile(len: usize, planes: usize, seed: usize) -> SaltV2Tile {
    SaltV2Tile::new(
        (0..planes)
            .map(|plane_index| plane(len, seed + plane_index))
            .collect(),
    )
    .unwrap()
}

fn tensor(name: &str, lengths_and_planes: &[(usize, usize)], seed: usize) -> SaltV2Tensor {
    let len = lengths_and_planes.iter().map(|(len, _)| len).sum::<usize>();
    let tiles = lengths_and_planes
        .iter()
        .enumerate()
        .map(|(tile_index, &(tile_len, planes))| tile(tile_len, planes, seed + tile_index * 5))
        .collect();
    SaltV2Tensor::new(name, vec![len as u64], tiles).unwrap()
}

fn package() -> SaltV2Package {
    package_with_codec(SaltV2Codec::D2)
}

fn package_with_codec(codec: SaltV2Codec) -> SaltV2Package {
    SaltV2Package::new(
        codec,
        vec![
            tensor("z", &[(256, 1), (256, 3), (87, 2)], 1),
            tensor("a", &[(256, 2)], 17),
        ],
    )
    .unwrap()
}

fn s34_plane(len: usize, seed: usize) -> SaltV2Plane {
    let trits = (0..len)
        .map(|index| match index % 4 {
            0 => 0,
            1 => -1,
            2 => 1,
            _ if seed.is_multiple_of(2) => -1,
            _ => 1,
        })
        .collect::<Vec<_>>();
    let scales = (0..len.div_ceil(128))
        .map(|group| f16::from_f32(0.5 + (seed + group) as f32 / 64.0))
        .collect();
    SaltV2Plane::new(trits, scales).unwrap()
}

fn s34_tile(len: usize, planes: usize, seed: usize) -> SaltV2Tile {
    SaltV2Tile::new(
        (0..planes)
            .map(|plane_index| s34_plane(len, seed + plane_index))
            .collect(),
    )
    .unwrap()
}

fn s34_package() -> SaltV2Package {
    SaltV2Package::new(
        SaltV2Codec::S34,
        vec![
            SaltV2Tensor::new(
                "z",
                vec![599],
                vec![
                    s34_tile(256, 1, 1),
                    s34_tile(256, 3, 7),
                    s34_tile(87, 2, 11),
                ],
            )
            .unwrap(),
            SaltV2Tensor::new("a", vec![256], vec![s34_tile(256, 2, 19)]).unwrap(),
        ],
    )
    .unwrap()
}

#[derive(Debug, PartialEq)]
struct OwnedPlane {
    tile_index: usize,
    plane_index: usize,
    plane_count: usize,
    logical_len: usize,
    packed: Vec<u8>,
    scale_bits: Vec<u16>,
}

impl OwnedPlane {
    fn from_ref(plane: PackedSaltV2PlaneRef<'_>) -> Self {
        Self {
            tile_index: plane.tile_index(),
            plane_index: plane.plane_index(),
            plane_count: plane.plane_count(),
            logical_len: plane.logical_len(),
            packed: plane.packed_bytes().to_vec(),
            scale_bits: plane.scales().iter().map(|scale| scale.to_bits()).collect(),
        }
    }
}

fn expected_planes(tensor: &SaltV2Tensor, codec: SaltV2Codec) -> Vec<OwnedPlane> {
    tensor
        .tiles()
        .iter()
        .enumerate()
        .flat_map(|(tile_index, tile)| {
            tile.planes()
                .iter()
                .enumerate()
                .map(move |(plane_index, plane)| OwnedPlane {
                    tile_index,
                    plane_index,
                    plane_count: tile.planes().len(),
                    logical_len: tile.logical_len(),
                    packed: pack_salt_v2_plane(codec, plane.trits()).unwrap(),
                    scale_bits: plane.scales().iter().map(|scale| scale.to_bits()).collect(),
                })
        })
        .collect()
}

struct ShortTrackedReader {
    cursor: Cursor<Vec<u8>>,
    max_request: Rc<Cell<usize>>,
    chunk: usize,
}

impl Read for ShortTrackedReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        self.max_request
            .set(self.max_request.get().max(output.len()));
        let len = output.len().min(self.chunk);
        self.cursor.read(&mut output[..len])
    }
}

impl Seek for ShortTrackedReader {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        self.cursor.seek(position)
    }
}

#[test]
fn strict_seek_reader_visits_canonical_planes_in_arbitrary_tensor_order() {
    let encoded = write_salt_v2_package(&package()).unwrap();
    let expected_package_id = PackageId::from_package_bytes(&encoded.bytes);
    let eager = read_salt_v2_package(&encoded.bytes).unwrap();
    let max_request = Rc::new(Cell::new(0));
    let source = ShortTrackedReader {
        cursor: Cursor::new(encoded.bytes),
        max_request: Rc::clone(&max_request),
        chunk: 7,
    };
    let mut reader = SaltV2PackageReader::new_strict(source).unwrap();

    assert_eq!(reader.codec(), SaltV2Codec::D2);
    assert_eq!(reader.package_id(), expected_package_id);
    assert_eq!(reader.ledger(), encoded.ledger);
    assert_eq!(
        reader.indexed_runtime_ledger().unwrap(),
        SaltV2IndexedRuntimeLedger::for_package(&eager.package).unwrap()
    );
    assert_eq!(reader.tensor_names().collect::<Vec<_>>(), ["a", "z"]);
    assert_eq!(
        reader.tensor_names_encoded_order().collect::<Vec<_>>(),
        ["z", "a"]
    );
    assert_eq!(reader.len(), 2);
    assert!(!reader.is_empty());

    for name in ["z", "a", "z"] {
        let tensor = eager
            .package
            .tensors()
            .iter()
            .find(|tensor| tensor.name() == name)
            .unwrap();
        let info = reader.tensor_info(name).unwrap();
        assert_eq!(info.dims(), tensor.dims());
        assert_eq!(info.logical_coefficients(), tensor.logical_coefficients());
        assert_eq!(info.tile_count(), tensor.tiles().len());
        assert_eq!(info.transform(), tensor.transform());
        assert_eq!(
            info.runtime_ledger(),
            SaltV2IndexedRuntimeLedger::for_tensor(tensor, eager.package.codec()).unwrap()
        );

        let mut actual = Vec::new();
        reader
            .visit_packed_tensor(name, |plane| {
                actual.push(OwnedPlane::from_ref(plane));
            })
            .unwrap();
        assert_eq!(actual, expected_planes(tensor, eager.package.codec()));
    }

    assert!(
        max_request.get() <= 64 * 1024,
        "largest Read request was {} bytes",
        max_request.get()
    );
}

#[test]
fn strict_seek_reader_exposes_exact_mixed_and_ragged_tile_plane_counts() {
    let encoded = write_salt_v2_package(&package()).unwrap();
    let reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();

    let mut counts = reader.tensor_plane_counts("z").unwrap();
    assert_eq!(counts.len(), 3);
    assert_eq!(counts.next(), Some(1));
    assert_eq!(counts.len(), 2);
    assert_eq!(counts.collect::<Vec<_>>(), [3, 2]);
    assert_eq!(
        reader.tensor_plane_counts("a").unwrap().collect::<Vec<_>>(),
        [2]
    );
    assert!(matches!(
        reader.tensor_plane_counts("absent"),
        Err(SaltV2PackageReadError::TensorNotFound(name)) if name == "absent"
    ));
}

#[test]
fn public_s34_plane_decoder_removes_canonical_ragged_shape_padding() {
    let plane = SaltV2Plane::new(vec![0, -1, 1, -1, 1, -1], vec![f16::from_f32(0.5)]).unwrap();
    let packed = pack_salt_v2_plane(SaltV2Codec::S34, plane.trits()).unwrap();

    let decoded = unpack_salt_v2_plane(SaltV2Codec::S34, &packed, 6).unwrap();

    assert_eq!(decoded, plane.trits());
}

#[test]
fn public_plane_decoder_rejects_non_plane_lengths_before_allocation() {
    for logical_len in [0, SALT_V2_ALLOCATION_TILE_SIZE + 1, usize::MAX] {
        assert_eq!(
            unpack_salt_v2_plane(SaltV2Codec::D2, &[], logical_len),
            Err(SaltV2PackageError::InvalidPlaneLength { got: logical_len })
        );
    }
}

#[test]
fn strict_seek_reader_matches_eager_decode_for_every_physical_codec() {
    for package in [
        package_with_codec(SaltV2Codec::D2),
        package_with_codec(SaltV2Codec::B3),
        s34_package(),
    ] {
        let encoded = write_salt_v2_package(&package).unwrap();
        let expected_package_id = PackageId::from_package_bytes(&encoded.bytes);
        let eager = read_salt_v2_package(&encoded.bytes).unwrap();
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();
        assert_eq!(reader.package_id(), expected_package_id);
        assert_eq!(reader.codec(), eager.package.codec());
        assert_eq!(reader.ledger(), eager.ledger);
        for tensor in eager.package.tensors() {
            let mut actual = Vec::new();
            reader
                .visit_packed_tensor(tensor.name(), |plane| {
                    actual.push(OwnedPlane::from_ref(plane));
                })
                .unwrap();
            assert_eq!(actual, expected_planes(tensor, eager.package.codec()));
        }
    }
}

#[test]
fn malformed_unselected_tensor_fails_strict_construction() {
    let package = package();
    let mut encoded = write_salt_v2_package(&package).unwrap().bytes;
    let target = pack_salt_v2_plane(
        SaltV2Codec::D2,
        package.tensors()[0].tiles()[1].planes()[2].trits(),
    )
    .unwrap();
    let payload_start = encoded
        .windows(target.len())
        .position(|window| window == target)
        .unwrap();
    encoded[payload_start] = (encoded[payload_start] & !0b11) | 0b11;

    assert!(SaltV2PackageReader::new_strict(Cursor::new(encoded)).is_err());
}

#[test]
fn malformed_scale_in_an_unselected_tensor_fails_strict_construction() {
    let package = SaltV2Package::new(
        SaltV2Codec::D2,
        vec![
            tensor("selected", &[(256, 1)], 3),
            tensor("unselected", &[(256, 1)], 29),
        ],
    )
    .unwrap();
    let mut encoded = write_salt_v2_package(&package).unwrap().bytes;
    let scale_bytes = package.tensors()[1].tiles()[0].planes()[0]
        .scales()
        .iter()
        .flat_map(|scale| scale.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let matches = encoded
        .windows(scale_bytes.len())
        .enumerate()
        .filter(|(_, window)| *window == scale_bytes)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "fixture scale sequence must be unique");
    encoded[matches[0] + 1] |= 0x80;

    assert!(SaltV2PackageReader::new_strict(Cursor::new(encoded)).is_err());
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
    let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor("w", &[(256, 1)], 23)]).unwrap();
    let bytes = write_salt_v2_package(&package).unwrap().bytes;
    let target = pack_salt_v2_plane(
        SaltV2Codec::D2,
        package.tensors()[0].tiles()[0].planes()[0].trits(),
    )
    .unwrap();
    let payload_start = bytes
        .windows(target.len())
        .position(|window| window == target)
        .unwrap();
    let shared = Rc::new(RefCell::new(bytes));
    let mut reader = SaltV2PackageReader::new_strict(SharedCursor {
        bytes: Rc::clone(&shared),
        position: 0,
    })
    .unwrap();
    let old_code = shared.borrow()[payload_start] & 0b11;
    let new_code = if old_code == 0 { 2 } else { 0 };
    let old_byte = shared.borrow()[payload_start];
    shared.borrow_mut()[payload_start] = (old_byte & !0b11) | new_code;

    let visited = Rc::new(Cell::new(false));
    let visited_in_callback = Rc::clone(&visited);
    let error = reader
        .visit_packed_tensor("w", |_| visited_in_callback.set(true))
        .unwrap_err();
    assert!(visited.get(), "callback precedes final digest verification");
    assert!(matches!(
        error,
        SaltV2PackageReadError::SourceChanged(name) if name == "w"
    ));
}

#[test]
fn terminal_package_check_detects_same_length_mutation_after_an_earlier_visit() {
    let bytes = write_salt_v2_package(&package()).unwrap().bytes;
    let shared = Rc::new(RefCell::new(bytes));
    let mut reader = SaltV2PackageReader::new_strict(SharedCursor {
        bytes: Rc::clone(&shared),
        position: 0,
    })
    .unwrap();

    reader.visit_packed_tensor("a", |_| {}).unwrap();
    reader.verify_unchanged().unwrap();

    shared.borrow_mut()[0] ^= 0x80;
    assert!(matches!(
        reader.verify_unchanged(),
        Err(SaltV2PackageReadError::PackageChanged)
    ));

    shared.borrow_mut()[0] ^= 0x80;
    shared.borrow_mut().push(0);
    assert!(matches!(
        reader.verify_unchanged(),
        Err(SaltV2PackageReadError::PackageChanged)
    ));
}

#[test]
fn selected_tensor_detects_map_and_length_mutation_before_callbacks() {
    let package = SaltV2Package::new(
        SaltV2Codec::D2,
        vec![tensor("w", &[(256, 1), (256, 2), (256, 3), (256, 1)], 31)],
    )
    .unwrap();
    let encoded = write_salt_v2_package(&package).unwrap();
    assert_eq!(encoded.ledger.maps_bytes, 1);
    let map_offset = encoded.bytes.len()
        - encoded.ledger.padding_bytes as usize
        - encoded.ledger.maps_bytes as usize;
    let shared = Rc::new(RefCell::new(encoded.bytes));
    let mut reader = SaltV2PackageReader::new_strict(SharedCursor {
        bytes: Rc::clone(&shared),
        position: 0,
    })
    .unwrap();
    shared.borrow_mut()[map_offset] ^= 1;

    let visited = Rc::new(Cell::new(false));
    let visited_in_callback = Rc::clone(&visited);
    assert!(matches!(
        reader.visit_packed_tensor("w", |_| visited_in_callback.set(true)),
        Err(SaltV2PackageReadError::SourceChanged(name)) if name == "w"
    ));
    assert!(!visited.get());

    shared.borrow_mut()[map_offset] ^= 1;
    shared.borrow_mut().push(0);
    assert!(matches!(
        reader.visit_packed_tensor("w", |_| {}),
        Err(SaltV2PackageReadError::SourceChanged(name)) if name == "w"
    ));
}

#[test]
fn missing_truncated_and_trailing_packages_fail_closed() {
    let bytes = write_salt_v2_package(&package()).unwrap().bytes;
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(bytes.clone())).unwrap();
    assert!(matches!(
        reader.visit_packed_tensor("absent", |_| {}),
        Err(SaltV2PackageReadError::TensorNotFound(name)) if name == "absent"
    ));

    let mut truncated = bytes.clone();
    truncated.pop();
    assert!(SaltV2PackageReader::new_strict(Cursor::new(truncated)).is_err());

    let mut trailing = bytes;
    trailing.push(0);
    assert!(SaltV2PackageReader::new_strict(Cursor::new(trailing)).is_err());
}

#[test]
fn model_reader_rejects_impossible_tensor_count_before_reserving_metadata() {
    let tensor_count = 1_000_001u32;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&SALT_V2_PACKAGE_MAGIC);
    bytes.extend_from_slice(&SALT_V2_PACKAGE_VERSION.to_le_bytes());
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&tensor_count.to_le_bytes());
    bytes.extend_from_slice(&24u64.to_le_bytes());
    let error = SaltV2PackageReader::new_strict(Cursor::new(bytes)).unwrap_err();
    assert!(matches!(
        error,
        SaltV2PackageReadError::LimitExceeded {
            resource,
            limit: 1_000_000,
            actual: 1_000_001
        } if resource == "tensor count"
    ));
}
