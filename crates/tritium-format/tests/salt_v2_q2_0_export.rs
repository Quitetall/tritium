use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::rc::Rc;

use half::f16;
use tritium_core::Trit;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    COMPACT_Q2_EXPORT_PROFILE_KEY, COMPACT_Q2_SOURCE_PACKAGE_ID_KEY, CompactQ2ExportError,
    CompactQ2GgufExportError, SaltV2Package, SaltV2PackageReader, SaltV2Plane, SaltV2Tensor,
    SaltV2Tile, SaltV2Transform, export_compact_q2_0_tensor, write_compact_q2_0_gguf,
    write_salt_v2_package,
};
use tritium_format::{
    GGML_TYPE_Q2_0, GgufValue, Q2_0_BLOCK_BYTES, q2_0_num_blocks, read_gguf, unpack_q2_0_row,
};

fn plane(raw_trits: Vec<i8>, scales: Vec<f16>, group_size: usize) -> SaltV2Plane {
    SaltV2Plane::new_with_scale_group_size(raw_trits, scales, group_size).unwrap()
}

fn tile(planes: Vec<SaltV2Plane>) -> SaltV2Tile {
    SaltV2Tile::new(planes).unwrap()
}

fn reader_with_codec(
    codec: SaltV2Codec,
    tensor: SaltV2Tensor,
) -> SaltV2PackageReader<Cursor<Vec<u8>>> {
    let package = SaltV2Package::new(codec, vec![tensor]).unwrap();
    let bytes = write_salt_v2_package(&package).unwrap().bytes;
    SaltV2PackageReader::new_strict(Cursor::new(bytes)).unwrap()
}

fn reader(tensor: SaltV2Tensor) -> SaltV2PackageReader<Cursor<Vec<u8>>> {
    reader_with_codec(SaltV2Codec::D2, tensor)
}

#[test]
fn compact_p1_g128_exports_every_salt_codec_exactly() {
    let raw = (0..320)
        .map(|index| [-1, 0, 1, 1][index % 4])
        .collect::<Vec<_>>();
    for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        let tensor = SaltV2Tensor::new(
            "weight",
            vec![5, 64],
            vec![
                tile(vec![plane(
                    raw[..256].to_vec(),
                    vec![f16::from_f32(0.5), f16::from_f32(1.5)],
                    128,
                )]),
                tile(vec![plane(
                    raw[256..].to_vec(),
                    vec![f16::from_f32(2.5)],
                    128,
                )]),
            ],
        )
        .unwrap();
        let mut reader = reader_with_codec(codec, tensor);

        let packed = export_compact_q2_0_tensor(&mut reader, "weight").unwrap();

        assert_eq!(packed.len(), q2_0_num_blocks(raw.len()) * Q2_0_BLOCK_BYTES);
        let mut decoded = vec![Trit::ZERO; raw.len()];
        let mut scales = vec![f16::ZERO; q2_0_num_blocks(raw.len())];
        unpack_q2_0_row(&packed, &mut decoded, &mut scales).unwrap();
        assert_eq!(
            decoded.iter().map(|trit| trit.get()).collect::<Vec<_>>(),
            raw
        );
        assert_eq!(
            scales
                .iter()
                .map(|scale| scale.to_bits())
                .collect::<Vec<_>>(),
            [0.5_f32, 0.5, 1.5, 1.5, 2.5].map(|scale| f16::from_f32(scale).to_bits())
        );
    }
}

#[test]
fn export_fails_closed_for_any_multi_plane_tile() {
    let base = vec![0; 256];
    let residual = vec![1; 256];
    let tensor = SaltV2Tensor::new(
        "weight",
        vec![256],
        vec![tile(vec![
            plane(base, vec![f16::ONE, f16::ONE], 128),
            plane(residual, vec![f16::ONE, f16::ONE], 128),
        ])],
    )
    .unwrap();
    let mut reader = reader(tensor);

    assert_eq!(
        export_compact_q2_0_tensor(&mut reader, "weight"),
        Err(CompactQ2ExportError::IncompatiblePlaneCount {
            tile_index: 0,
            got: 2,
        })
    );
}

#[test]
fn export_reports_nonuniform_plane_map_at_first_incompatible_tile() {
    let tensor = SaltV2Tensor::new(
        "weight",
        vec![512],
        vec![
            tile(vec![plane(vec![0; 256], vec![f16::ONE, f16::ONE], 128)]),
            tile(vec![
                plane(vec![0; 256], vec![f16::ONE, f16::ONE], 128),
                plane(vec![1; 256], vec![f16::ONE, f16::ONE], 128),
            ]),
        ],
    )
    .unwrap();
    let mut reader = reader(tensor);

    assert_eq!(
        export_compact_q2_0_tensor(&mut reader, "weight"),
        Err(CompactQ2ExportError::IncompatiblePlaneCount {
            tile_index: 1,
            got: 2,
        })
    );
}

#[test]
fn export_rejects_g64_source_instead_of_guessing_profile_semantics() {
    let tensor = SaltV2Tensor::new_with_layout(
        "weight",
        vec![64],
        SaltV2Transform::None,
        64,
        vec![tile(vec![plane(vec![0; 64], vec![f16::ONE], 64)])],
    )
    .unwrap();
    let mut reader = reader(tensor);

    assert_eq!(
        export_compact_q2_0_tensor(&mut reader, "weight"),
        Err(CompactQ2ExportError::IncompatibleScaleGroupSize { got: 64 })
    );
}

#[test]
fn export_rejects_transform_metadata_that_q2_0_cannot_carry() {
    let tensor = SaltV2Tensor::new_with_transform(
        "weight",
        vec![256],
        SaltV2Transform::SignedRht { seed: 7, domain: 9 },
        vec![tile(vec![plane(
            vec![0; 256],
            vec![f16::ONE, f16::ONE],
            128,
        )])],
    )
    .unwrap();
    let mut reader = reader(tensor);

    assert_eq!(
        export_compact_q2_0_tensor(&mut reader, "weight"),
        Err(CompactQ2ExportError::IncompatibleTransform {
            got: SaltV2Transform::SignedRht { seed: 7, domain: 9 },
        })
    );
}

#[test]
fn export_rejects_rows_that_would_be_flattened_across_q2_0_blocks() {
    let tensor = SaltV2Tensor::new(
        "weight",
        vec![2, 65],
        vec![tile(vec![plane(
            vec![0; 130],
            vec![f16::ONE, f16::ONE],
            128,
        )])],
    )
    .unwrap();
    let mut reader = reader(tensor);

    assert_eq!(
        export_compact_q2_0_tensor(&mut reader, "weight"),
        Err(CompactQ2ExportError::IncompatibleRowWidth { got: 65 })
    );
}

#[test]
fn package_export_writes_shaped_identity_bound_q2_0_gguf() {
    let left_raw = (0..128)
        .map(|index| [-1, 0, 1, 1][index % 4])
        .collect::<Vec<_>>();
    let right_raw = (0..256)
        .map(|index| [1, 0, -1, -1][index % 4])
        .collect::<Vec<_>>();
    let package = SaltV2Package::new(
        SaltV2Codec::S34,
        vec![
            SaltV2Tensor::new(
                "left.weight",
                vec![2, 64],
                vec![tile(vec![plane(
                    left_raw.clone(),
                    vec![f16::from_f32(0.5)],
                    128,
                )])],
            )
            .unwrap(),
            SaltV2Tensor::new(
                "right.weight",
                vec![4, 64],
                vec![tile(vec![plane(
                    right_raw.clone(),
                    vec![f16::from_f32(1.5), f16::from_f32(2.5)],
                    128,
                )])],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let encoded = write_salt_v2_package(&package).unwrap();
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();
    let source_package_id = reader.package_id().to_string();
    let metadata = BTreeMap::from([(
        "general.architecture".to_owned(),
        GgufValue::String("fixture".to_owned()),
    )]);
    let mut output = Vec::new();

    write_compact_q2_0_gguf(&mut reader, &mut output, 3, metadata).unwrap();

    let gguf = read_gguf(&output).unwrap();
    assert_eq!(
        gguf.metadata.get("general.architecture"),
        Some(&GgufValue::String("fixture".to_owned()))
    );
    assert_eq!(
        gguf.metadata.get(COMPACT_Q2_SOURCE_PACKAGE_ID_KEY),
        Some(&GgufValue::String(source_package_id))
    );
    assert_eq!(
        gguf.metadata.get(COMPACT_Q2_EXPORT_PROFILE_KEY),
        Some(&GgufValue::String("compact-v1-p1-g128".to_owned()))
    );
    for (info, (expected_dims, expected_raw)) in gguf
        .tensors
        .iter()
        .zip([(vec![64, 2], left_raw), (vec![64, 4], right_raw)])
    {
        assert_eq!(info.ggml_type, GGML_TYPE_Q2_0);
        assert_eq!(info.dims, expected_dims);
        let start = (gguf.tensor_data_offset + info.offset) as usize;
        let payload = &output[start..start + info.n_bytes as usize];
        let mut decoded = vec![Trit::ZERO; expected_raw.len()];
        let mut scales = vec![f16::ZERO; q2_0_num_blocks(expected_raw.len())];
        unpack_q2_0_row(payload, &mut decoded, &mut scales).unwrap();
        assert_eq!(
            decoded.iter().map(|trit| trit.get()).collect::<Vec<_>>(),
            expected_raw
        );
    }
}

#[test]
fn package_export_preflights_every_tensor_before_writing_header() {
    let package = SaltV2Package::new(
        SaltV2Codec::D2,
        vec![
            SaltV2Tensor::new(
                "good.weight",
                vec![1, 64],
                vec![tile(vec![plane(vec![0; 64], vec![f16::ONE], 128)])],
            )
            .unwrap(),
            SaltV2Tensor::new(
                "bad.weight",
                vec![1, 64],
                vec![tile(vec![
                    plane(vec![0; 64], vec![f16::ONE], 128),
                    plane(vec![1; 64], vec![f16::ONE], 128),
                ])],
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let encoded = write_salt_v2_package(&package).unwrap();
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();
    let mut output = Vec::new();

    let error = write_compact_q2_0_gguf(&mut reader, &mut output, 3, BTreeMap::new())
        .expect_err("P>1 tensor must fail package preflight");

    assert!(matches!(
        error,
        CompactQ2GgufExportError::Tensor {
            name,
            source: CompactQ2ExportError::IncompatiblePlaneCount {
                tile_index: 0,
                got: 2,
            },
        } if name == "bad.weight"
    ));
    assert!(output.is_empty(), "preflight failure wrote a GGUF prefix");
}

#[test]
fn package_export_rejects_spoofed_reserved_provenance_before_writing() {
    let tensor = SaltV2Tensor::new(
        "weight",
        vec![1, 64],
        vec![tile(vec![plane(vec![0; 64], vec![f16::ONE], 128)])],
    )
    .unwrap();
    let mut reader = reader(tensor);
    let metadata = BTreeMap::from([(
        COMPACT_Q2_SOURCE_PACKAGE_ID_KEY.to_owned(),
        GgufValue::String("spoofed".to_owned()),
    )]);
    let mut output = Vec::new();

    let error = write_compact_q2_0_gguf(&mut reader, &mut output, 3, metadata)
        .expect_err("reserved provenance must be exporter-owned");

    assert!(matches!(
        error,
        CompactQ2GgufExportError::ReservedMetadataKey(key)
            if key == COMPACT_Q2_SOURCE_PACKAGE_ID_KEY
    ));
    assert!(output.is_empty());
}

#[derive(Clone)]
struct SharedCursor {
    bytes: Rc<RefCell<Vec<u8>>>,
    position: u64,
    read_bytes: Option<Rc<Cell<usize>>>,
}

impl Read for SharedCursor {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let bytes = self.bytes.borrow();
        let start = usize::try_from(self.position).unwrap_or(usize::MAX);
        if start >= bytes.len() {
            return Ok(0);
        }
        let count = output.len().min(bytes.len() - start);
        output[..count].copy_from_slice(&bytes[start..start + count]);
        self.position += count as u64;
        if let Some(read_bytes) = &self.read_bytes {
            read_bytes.set(read_bytes.get().saturating_add(count));
        }
        Ok(count)
    }
}

impl Seek for SharedCursor {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        let len = self.bytes.borrow().len() as i128;
        let next = match position {
            SeekFrom::Start(value) => i128::from(value),
            SeekFrom::End(delta) => len + i128::from(delta),
            SeekFrom::Current(delta) => i128::from(self.position) + i128::from(delta),
        };
        if !(0..=i128::from(u64::MAX)).contains(&next) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid shared cursor seek",
            ));
        }
        self.position = next as u64;
        Ok(self.position)
    }
}

#[derive(Debug)]
struct MutatingOutput {
    output: Rc<RefCell<Vec<u8>>>,
    source: Rc<RefCell<Vec<u8>>>,
    mutated: bool,
}

impl Write for MutatingOutput {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.output.borrow_mut().extend_from_slice(bytes);
        if !self.mutated
            && bytes.len() == Q2_0_BLOCK_BYTES
            && bytes
                .get(2..)
                .is_some_and(|codes| codes.iter().all(|&code| code == 0x55))
        {
            let mut source = self.source.borrow_mut();
            let last = source.last_mut().expect("nonempty package");
            *last ^= 1;
            self.mutated = true;
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn package_export_detects_source_mutation_after_partial_output() {
    let tensor = SaltV2Tensor::new(
        "weight",
        vec![1, 64],
        vec![tile(vec![plane(vec![0; 64], vec![f16::ONE], 128)])],
    )
    .unwrap();
    let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).unwrap();
    let source = Rc::new(RefCell::new(write_salt_v2_package(&package).unwrap().bytes));
    let source_cursor = SharedCursor {
        bytes: Rc::clone(&source),
        position: 0,
        read_bytes: None,
    };
    let mut reader = SaltV2PackageReader::new_strict(source_cursor).unwrap();
    let output = Rc::new(RefCell::new(Vec::new()));
    let writer = MutatingOutput {
        output: Rc::clone(&output),
        source,
        mutated: false,
    };

    let error = write_compact_q2_0_gguf(&mut reader, writer, 3, BTreeMap::new())
        .expect_err("terminal source mutation must invalidate output");

    assert!(matches!(
        error,
        CompactQ2GgufExportError::Read(
            tritium_format::salt_v2_package::SaltV2PackageReadError::PackageChanged
        )
    ));
    assert!(
        output.borrow().len() > Q2_0_BLOCK_BYTES,
        "destination did not receive partial GGUF bytes before failure"
    );
}

#[derive(Debug)]
struct FailOnQ2Payload;

impl Write for FailOnQ2Payload {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let scale = f16::ONE.to_bits().to_le_bytes();
        let q2_payload =
            bytes.len() == 4 * Q2_0_BLOCK_BYTES
                && bytes.as_chunks::<Q2_0_BLOCK_BYTES>().0.iter().all(|block| {
                    block[..2] == scale && block[2..].iter().all(|&code| code == 0x55)
                });
        if q2_payload {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "injected Q2_0 sink failure",
            ))
        } else {
            Ok(bytes.len())
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn package_export_stops_source_scan_on_first_sink_failure() {
    let tile_count = 2_048;
    let tiles = (0..tile_count)
        .map(|_| tile(vec![plane(vec![0; 256], vec![f16::ONE, f16::ONE], 128)]))
        .collect();
    let tensor = SaltV2Tensor::new("weight", vec![tile_count as u64, 256], tiles).unwrap();
    let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).unwrap();
    let encoded = write_salt_v2_package(&package).unwrap();
    let package_len = encoded.bytes.len();
    let read_bytes = Rc::new(Cell::new(0));
    let source = SharedCursor {
        bytes: Rc::new(RefCell::new(encoded.bytes)),
        position: 0,
        read_bytes: Some(Rc::clone(&read_bytes)),
    };
    let mut reader = SaltV2PackageReader::new_strict(source).unwrap();
    read_bytes.set(0);

    let error = write_compact_q2_0_gguf(&mut reader, FailOnQ2Payload, 3, BTreeMap::new())
        .expect_err("injected sink failure must stop conversion");

    assert!(matches!(
        error,
        CompactQ2GgufExportError::Gguf(tritium_format::GgufWriteError::Io(ref io))
            if io.kind() == std::io::ErrorKind::BrokenPipe
    ));
    assert!(
        read_bytes.get() < package_len,
        "sink failure scanned full source: read {} of {package_len} bytes",
        read_bytes.get()
    );
}
